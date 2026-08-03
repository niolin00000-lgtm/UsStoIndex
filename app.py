import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 頁面配置
st.set_page_config(page_title="雙層多因子美股 24-48H 預測模型", layout="wide", initial_sidebar_state="expanded")

# 側邊欄設定
st.sidebar.header("⚙️ 系統參數")
days_options = {"1 個月 (30天)": 30, "3 個月 (90天)": 90, "6 個月 (180天)": 180, "1 年 (365天)": 365}
selected_option = st.sidebar.selectbox("觀察歷史區間", list(days_options.keys()), index=1)
days = days_options[selected_option]

st.sidebar.markdown("---")

# 手動刷新按鈕
if st.sidebar.button("🔄 重新載入最新 API 數據", use_container_width=True):
    st.cache_data.clear()
    st.success("快取已清空，正在向 API 抓取最新數據...")
    st.rerun()

# 頁面標題與時間戳記
st.title("🔮 美股雙層多因子預測模型 (日級衝擊 + 週級氛圍)")
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"🕒 **最後更新時間：** `{now_str}` ｜ 結合 8 個週級總體氛圍指標與 7 個日級極速衝擊指標，避免單一數據誤判。")

@st.cache_data(ttl=1800)
def load_multi_factor_data(days_back):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 250) # 預留足夠天數算 200MA
    
    # 標的與對應 ticker
    tickers = {
        'SPY': 'SPY',         # 標普 500 現貨
        'US10Y': '^TNX',       # 10年期美債殖利率
        'VIX': '^VIX',         # 30天恐慌指數
        'VIX1D': '^VIX1D',     # 1天期極速恐慌指數
        'HYG': 'HYG',         # 高收益債
        'LQD': 'LQD',         # 投資級債
        'DXY': 'DX-Y.NYB',    # 美元指數
        'IWM': 'IWM',         # 羅素2000小型股
        'TSM': 'TSM',         # 台積電 ADR (科技/AI先導)
        'XLY': 'XLY',         # 非必需消費 (消費信心)
        'XLP': 'XLP'          # 必需消費 (防禦屬性)
    }
    
    ticker_list = list(tickers.values())
    raw_data = yf.download(ticker_list, start=start_date, end=end_date, progress=False)
    
    if raw_data.empty:
        return pd.DataFrame()
        
    # 安全獲取 Close 欄位（相容 MultiIndex 與單層 Index）
    if isinstance(raw_data.columns, pd.MultiIndex):
        if 'Close' in raw_data.columns.levels[0]:
            close_data = raw_data['Close']
        else:
            close_data = raw_data.xs('Close', axis=1, level=0, droplevel=False)
    else:
        close_data = raw_data

    # 重構 DataFrame
    df = pd.DataFrame()
    for key, symbol in tickers.items():
        if symbol in close_data.columns:
            df[key] = close_data[symbol]
            
    # 資料清理：先用前向填充（處理交易日不一），再用後向填充補齊開頭
    df = df.ffill().bfill()
    
    # 若核心欄位依然缺失則回傳空值
    required_cols = ['SPY', 'VIX', 'HYG', 'LQD', 'US10Y']
    if not all(col in df.columns for col in required_cols) or df.empty:
        return pd.DataFrame()

    # 若特定變動較大的 API 欄位缺失，以備用邏輯補齊
    if 'VIX1D' not in df.columns or df['VIX1D'].isnull().all():
        df['VIX1D'] = df['VIX'] # 備用防護

    # ==========================================
    # 🔵 第一層：週級別總體氛圍指標 (Weekly Layer)
    # ==========================================
    df['W_Discretionary_Defensive'] = df['XLY'] / df['XLP']
    df['W_Credit_Trend'] = (df['HYG'] / df['LQD']).rolling(20, min_periods=5).mean()
    df['W_SPY_SMA50_Ratio'] = df['SPY'] / df['SPY'].rolling(50, min_periods=10).mean()
    df['W_SPY_SMA200_Ratio'] = df['SPY'] / df['SPY'].rolling(200, min_periods=20).mean()
    df['W_VIX_Baseline'] = df['VIX'].rolling(20, min_periods=5).mean()
    df['W_DXY_Trend'] = df['DXY'].rolling(20, min_periods=5).mean()
    df['W_Breadth_Trend'] = (df['IWM'] / df['SPY']).rolling(20, min_periods=5).mean()

    # 計算週級綜合氣氛分數 (Weekly Sentiment Score: 0~100)
    w_w = 60
    z_w1 = (df['W_Discretionary_Defensive'] - df['W_Discretionary_Defensive'].rolling(w_w, min_periods=10).mean()) / (df['W_Discretionary_Defensive'].rolling(w_w, min_periods=10).std() + 1e-6)
    z_w2 = (df['W_Credit_Trend'] - df['W_Credit_Trend'].rolling(w_w, min_periods=10).mean()) / (df['W_Credit_Trend'].rolling(w_w, min_periods=10).std() + 1e-6)
    z_w3 = (df['W_SPY_SMA50_Ratio'] - df['W_SPY_SMA50_Ratio'].rolling(w_w, min_periods=10).mean()) / (df['W_SPY_SMA50_Ratio'].rolling(w_w, min_periods=10).std() + 1e-6)
    z_w4 = -1 * (df['W_VIX_Baseline'] - df['W_VIX_Baseline'].rolling(w_w, min_periods=10).mean()) / (df['W_VIX_Baseline'].rolling(w_w, min_periods=10).std() + 1e-6)
    
    weekly_logit = (z_w1 * 0.3) + (z_w2 * 0.3) + (z_w3 * 0.2) + (z_w4 * 0.2)
    df['Weekly_Regime_Score'] = (1 / (1 + np.exp(-weekly_logit.fillna(0)))) * 100

    # ==========================================
    # 🔴 第二層：日級別極速衝擊指標 (Daily Layer)
    # ==========================================
    df['D_VIX_Structure'] = df['VIX1D'] / df['VIX']
    df['D_VIX_1D_Pct'] = df['VIX'].pct_change(1)
    df['D_US10Y_1D_Chg'] = df['US10Y'].diff(1)
    df['D_DXY_1D_Pct'] = df['DXY'].pct_change(1)
    df['D_Credit_1D_Pct'] = (df['HYG'] / df['LQD']).pct_change(1)
    df['D_Breadth_1D_Pct'] = (df['IWM'] / df['SPY']).pct_change(1)
    df['D_TSM_1D_Pct'] = df['TSM'].pct_change(1)

    # 計算日級極速風險分數 (Daily Shock Score: 0~100)
    w_d = 20
    z_d1 = (df['D_VIX_Structure'] - df['D_VIX_Structure'].rolling(w_d, min_periods=5).mean()) / (df['D_VIX_Structure'].rolling(w_d, min_periods=5).std() + 1e-6)
    z_d2 = (df['D_US10Y_1D_Chg'] - df['D_US10Y_1D_Chg'].rolling(w_d, min_periods=5).mean()) / (df['D_US10Y_1D_Chg'].rolling(w_d, min_periods=5).std() + 1e-6)
    z_d3 = (df['D_DXY_1D_Pct'] - df['D_DXY_1D_Pct'].rolling(w_d, min_periods=5).mean()) / (df['D_DXY_1D_Pct'].rolling(w_d, min_periods=5).std() + 1e-6)
    z_d4 = -1 * (df['D_Credit_1D_Pct'] - df['D_Credit_1D_Pct'].rolling(w_d, min_periods=5).mean()) / (df['D_Credit_1D_Pct'].rolling(w_d, min_periods=5).std() + 1e-6)
    z_d5 = -1 * (df['D_TSM_1D_Pct'] - df['D_TSM_1D_Pct'].rolling(w_d, min_periods=5).mean()) / (df['D_TSM_1D_Pct'].rolling(w_d, min_periods=5).std() + 1e-6)

    daily_logit = (z_d1 * 1.0) + (z_d2 * 0.8) + (z_d3 * 0.6) + (z_d4 * 0.6) + (z_d5 * 0.5)
    df['Daily_Shock_Score'] = (1 / (1 + np.exp(-daily_logit.fillna(0)))) * 100

    # ==========================================
    # 🟣 雙層融合預測機率 (Integrated Forecast)
    # ==========================================
    df['Final_Prob_Down_2448H'] = (df['Weekly_Regime_Score'] * 0.4) + (df['Daily_Shock_Score'] * 0.6)

    # 清理所有中間計算產生的極少數 NaN
    # ✅ 新版相容語法
    df = df.ffill().bfill()

    return df.tail(days_back)

with st.spinner("正在加載多維度指標並計算雙層模型..."):
    df = load_multi_factor_data(days)

if not df.empty and len(df) >= 2:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    final_prob = round(latest['Final_Prob_Down_2448H'], 1)
    prob_change = round(latest['Final_Prob_Down_2448H'] - prev['Final_Prob_Down_2448H'], 1)
    weekly_score = round(latest['Weekly_Regime_Score'], 1)
    daily_score = round(latest['Daily_Shock_Score'], 1)

    # 頂部三大儀表卡片
    m1, m2, m3 = st.columns(3)
    m1.metric("🔮 綜合預測下行/高波動機率 (24-48H)", f"{final_prob}%", f"{prob_change:+}%", delta_color="inverse")
    m2.metric("🔵 第一層：週級總體脆弱度 (Macro Regime)", f"{weekly_score}%", "高位過熱/脆弱" if weekly_score > 60 else "健康/穩健")
    m3.metric("🔴 第二層：日級極速衝擊力 (Daily Shock)", f"{daily_score}%", "閃電避險觸發" if daily_score > 60 else "極速指標平穩")

    st.markdown("---")

    # 歷史走勢圖表
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Final_Prob_Down_2448H'], mode='lines', name='綜合預測機率 (%)', line=dict(color='crimson', width=3)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Weekly_Regime_Score'], mode='lines', name='週級脆弱度濾網 (%)', line=dict(color='royalblue', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Daily_Shock_Score'], mode='lines', name='日級衝擊觸發值 (%)', line=dict(color='orange', width=1.5, dash='dash')))

    fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="高風險警戒 (75%)")
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="中性分界 (50%)")

    fig.update_layout(
        title="<b>雙層模型歷史走勢對比圖 (週級氣氛 vs. 日級衝擊 vs. 綜合結果)</b>",
        xaxis_title="日期", yaxis_title="百分比 (%)", yaxis=dict(range=[0, 100]),
        template="plotly_white", height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    # 底層數據面板
    st.subheader("🔍 底層基底指標透明化面板")
    st.write("以下為模型計算所使用的所有原始指數與衍生特徵當前最新數值：")

    col_w, col_d = st.columns(2)

    with col_w:
        st.markdown("### 🔵 週級總體與氛圍指標")
        weekly_data = {
            "基底指標名稱": [
                "標普500現貨 (SPY)", 
                "消費信心比值 (XLY/XLP)", 
                "信用趨勢 (HYG/LQD 20MA)", 
                "SPY / 50日均線比率", 
                "SPY / 200日均線比率", 
                "VIX 20日基底", 
                "美元指數 20日趨勢", 
                "小型股廣度 20日趨勢"
            ],
            "最新數值": [
                f"${latest['SPY']:.2f}",
                f"{latest['W_Discretionary_Defensive']:.3f}",
                f"{latest['W_Credit_Trend']:.3f}",
                f"{latest['W_SPY_SMA50_Ratio']:.3f}",
                f"{latest['W_SPY_SMA200_Ratio']:.3f}",
                f"{latest['W_VIX_Baseline']:.2f}",
                f"{latest['W_DXY_Trend']:.2f}",
                f"{latest['W_Breadth_Trend']:.3f}"
            ]
        }
        st.table(pd.DataFrame(weekly_data))

    with col_d:
        st.markdown("### 🔴 日級極速與衝擊指標")
        daily_data = {
            "基底指標名稱": [
                "1日極速恐慌比 (VIX1D/VIX)", 
                "VIX 恐慌指數現貨", 
                "10年美債殖利率 (US10Y)", 
                "美債 1日衝擊量", 
                "美元指數 (DXY) 1日變動", 
                "信用風險 1日變動", 
                "台積電 ADR (TSM) 1日變動"
            ],
            "最新數值": [
                f"{latest['D_VIX_Structure']:.3f}",
                f"{latest['VIX']:.2f}",
                f"{latest['US10Y']:.2f}%",
                f"{latest['D_US10Y_1D_Chg']:+.2f}%",
                f"{latest['D_DXY_1D_Pct']*100:+.2f}%",
                f"{latest['D_Credit_1D_Pct']*100:+.2f}%",
                f"{latest['D_TSM_1D_Pct']*100:+.2f}%"
            ]
        }
        st.table(pd.DataFrame(daily_data))

else:
    st.error("⚠️ 無法獲取多因子數據，請重新整理頁面或點擊左側「🔄 重新載入最新 API 數據」按鈕。")
