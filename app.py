import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 頁面配置
st.set_page_config(page_title="美股 24-48H 預測模型 Dashboard", layout="wide", initial_sidebar_state="expanded")

# 側邊欄設定
st.sidebar.header("⚙️ 參數設定")
days_options = {"1 個月 (30天)": 30, "3 個月 (90天)": 90, "6 個月 (180天)": 180, "1 年 (365天)": 365}
selected_option = st.sidebar.selectbox("回測與觀察歷史區間", list(days_options.keys()), index=1)
days = days_options[selected_option]

st.sidebar.markdown("---")

# 🔄 手動刷新按鈕
if st.sidebar.button("🔄 重新載入最新數據", use_container_width=True):
    st.cache_data.clear()
    st.success("快取已清除，正在重新呼叫 API...")
    st.rerun()

st.title("🔮 美股 24–48 小時市場波動與方向預測模型")
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"🕒 **最後更新時間：** `{now_str}` ｜ 透過領先指標鏈推算未來 1-2 個交易日大盤修正與波動擴大的概率。")

@st.cache_data(ttl=1800)
def load_predictive_data(days_back):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 40)
    
    tickers = {
        'SPY': 'SPY',         # 標普 500 現貨
        'US10Y': '^TNX',       # 10年期美債殖利率
        'VIX': '^VIX',         # 30天恐慌指數
        'VIX1D': '^VIX1D',     # 1天期極速恐慌指數
        'HYG': 'HYG',         # 高收益債
        'LQD': 'LQD',         # 投資級債
        'DXY': 'DX-Y.NYB',    # 美元指數
        'IWM': 'IWM'          # 小型股
    }
    
    ticker_list = list(tickers.values())
    raw_data = yf.download(ticker_list, start=start_date, end=end_date, progress=False)
    
    if raw_data.empty or 'Close' not in raw_data:
        return pd.DataFrame()
        
    close_data = raw_data['Close']
    df = pd.DataFrame()
    
    for key, symbol in tickers.items():
        if symbol in close_data:
            df[key] = close_data[symbol]
            
    df = df.ffill().dropna()
    if df.empty:
        return pd.DataFrame()
        
    # --- 1. 計算先期特徵變化量 ---
    df['VIX_Structure'] = df['VIX1D'] / df['VIX']
    df['Credit_Ratio'] = df['HYG'] / df['LQD']
    df['Breadth_Ratio'] = df['IWM'] / df['SPY']
    
    df['US10Y_1D_Chg'] = df['US10Y'].diff(1)
    df['DXY_1D_Chg'] = df['DXY'].pct_change(1)
    df['Credit_3D_Chg'] = df['Credit_Ratio'].pct_change(3)
    
    # --- 2. 建立預測機率模型 ---
    w = 20
    z_vix_struct = (df['VIX_Structure'] - df['VIX_Structure'].rolling(w).mean()) / df['VIX_Structure'].rolling(w).std()
    z_us10y_chg = (df['US10Y_1D_Chg'] - df['US10Y_1D_Chg'].rolling(w).mean()) / df['US10Y_1D_Chg'].rolling(w).std()
    z_dxy_chg = (df['DXY_1D_Chg'] - df['DXY_1D_Chg'].rolling(w).mean()) / df['DXY_1D_Chg'].rolling(w).std()
    z_credit_chg = -1 * (df['Credit_3D_Chg'] - df['Credit_3D_Chg'].rolling(w).mean()) / df['Credit_3D_Chg'].rolling(w).std()
    
    pred_logit = (
        (z_vix_struct * 1.2) + 
        (z_us10y_chg * 0.9) + 
        (z_dxy_chg * 0.7) + 
        (z_credit_chg * 0.7)
    )
    
    df['Prob_Down_2448H'] = (1 / (1 + np.exp(-pred_logit))) * 100
    
    return df.tail(days_back).dropna()

with st.spinner("正在向 Yahoo Finance API 請求最新市場數據..."):
    df = load_predictive_data(days)

if not df.empty and len(df) >= 2:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    prob = round(latest['Prob_Down_2448H'], 1)
    prob_change = round(latest['Prob_Down_2448H'] - prev['Prob_Down_2448H'], 1)
    
    # --- 預測結論邏輯判斷 ---
    if prob >= 75:
        forecast_title = "🔴 未來 24–48 小時：高風險 / 預期回檔修正"
        action_advice = "機構避險需求急升，債市與衍生品同步發出警訊。建議降低槓桿、避開高估值科技股，不宜追高。"
    elif prob >= 50:
        forecast_title = "🟡 未來 24–48 小時：中性偏弱 / 震盪整理"
        action_advice = "市場出現局部壓力因子（如殖利率或美元短線拉高），多空交戰加劇，大盤容易開高走低或區間震盪。"
    else:
        forecast_title = "🟢 未來 24–48 小時：環境安全 / 偏多或平穩"
        action_advice = "先期壓力指標處於低位，流動性與衍生品情緒穩定，短期大盤出現系統性急跌的概率較低。"

    # --- 頂部預測儀表卡片 ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("未來 24–48H 市場下行/衝擊機率", f"{prob}%", f"{prob_change:+}%", delta_color="inverse")
    
    with col2:
        st.subheader(forecast_title)
        st.write(f"💡 **模型建議：** {action_advice}")

    st.markdown("---")

    # --- 預測走勢圖表 ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Prob_Down_2448H'],
        mode='lines', name='預測下行/高波動概率 (%)',
        line=dict(color='crimson', width=2.5)
    ))

    fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="高風險警戒線 (75%)")
    fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="中性分界線 (50%)")

    fig.update_layout(
        title="<b>歷史預測概率走勢圖 (未來 24–48H 預測)</b>",
        xaxis_title="日期",
        yaxis_title="預測概率 (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- 預測因子診斷 ---
    st.subheader("🔍 預測因子即時診斷 (24-48H 驅動來源)")
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("VIX 期貨倒掛比 (VIX1D/VIX)", f"{latest['VIX_Structure']:.2f}", 
              "倒掛警訊" if latest['VIX_Structure'] >= 1.0 else "結構正常")
    c2.metric("10年美債 1日衝擊", f"{latest['US10Y_1D_Chg']:+.2f}%", delta_color="inverse")
    c3.metric("美元指數 1日變動", f"{latest['DXY_1D_Chg']*100:+.2f}%", delta_color="inverse")
    c4.metric("高收益債比 3日趨勢", f"{latest['Credit_3D_Chg']*100:+.2f}%", delta_color="normal")

else:
    st.error("⚠️ 無法獲取預測數據，請重新整理頁面或按左側按鈕重試。")
