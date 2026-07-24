import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 頁面配置
st.set_page_config(page_title="美股先期情緒警戒 Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("🛡️ 美股先期風險警戒儀表板 (Early Warning Risk Dashboard)")
st.caption("整合美債殖利率 (10Y)、VIX 波動率與高收益債信用利差 (HYG/LQD)，提早 24-48 小時感知市場氣氛變革。")

# 側邊欄設定
st.sidebar.header("⚙️ 參數設定")
days_options = {"1 個月 (30天)": 30, "3 個月 (90天)": 90, "6 個月 (180天)": 180, "1 年 (365天)": 365}
selected_option = st.sidebar.selectbox("資料歷史區間", list(days_options.keys()), index=1)
days = days_options[selected_option]

@st.cache_data(ttl=1800)  # 快取 30 分鐘
def load_data(days_back):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 30) # 多抓 30 天算移動平均與 Rolling Z-score
    
    tickers = {
        'US10Y': '^TNX',   # 10年期美債殖利率
        'VIX': '^VIX',     # CBOE 恐慌指數
        'HYG': 'HYG',      # 高收益債 ETF
        'LQD': 'LQD'       # 投資級債 ETF
    }
    
    # 一次性下載所有數據，減少 API 請求次數
    ticker_list = list(tickers.values())
    raw_data = yf.download(ticker_list, start=start_date, end=end_date, progress=False)
    
    if raw_data.empty or 'Close' not in raw_data:
        return pd.DataFrame()
        
    close_data = raw_data['Close']
    
    df = pd.DataFrame()
    df['US10Y'] = close_data['^TNX']
    df['VIX'] = close_data['^VIX']
    df['HYG'] = close_data['HYG']
    df['LQD'] = close_data['LQD']
    
    # 處理假日/時區造成的空值：先用前一日數據填補，最後才刪除前段無法填補的 Na
    df = df.ffill().dropna()
    
    if df.empty:
        return pd.DataFrame()
    
    # 計算 HYG / LQD 比值（風險偏好指標）
    df['Credit_Ratio'] = df['HYG'] / df['LQD']
    
    # 計算 20 日滾動 Z-Score 衡量極端風險
    window = 20
    z_us10y = (df['US10Y'] - df['US10Y'].rolling(window).mean()) / df['US10Y'].rolling(window).std()
    z_vix = (df['VIX'] - df['VIX'].rolling(window).mean()) / df['VIX'].rolling(window).std()
    z_credit = -1 * (df['Credit_Ratio'] - df['Credit_Ratio'].rolling(window).mean()) / df['Credit_Ratio'].rolling(window).std()
    
    # 綜合風險分數 (Weights: VIX 40%, 10Y 30%, Credit 30%)
    composite_z = (z_vix * 0.4) + (z_us10y * 0.3) + (z_credit * 0.3)
    
    # Sigmoid 函數映射至 0 - 100 分
    df['Risk_Index'] = (1 / (1 + np.exp(-composite_z))) * 100
    
    # 只回傳使用者選擇的天數範圍
    return df.tail(days_back).dropna()

with st.spinner("正在抓取最新市場數據..."):
    df = load_data(days)

if not df.empty and len(df) >= 2:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    risk_score = round(latest['Risk_Index'], 1)
    risk_change = round(latest['Risk_Index'] - prev['Risk_Index'], 1)
    
    # 頂部卡片
    col1, col2, col3, col4 = st.columns(4)
    
    if risk_score >= 70:
        status = "🔴 高度警戒 (Fear)"
    elif risk_score >= 45:
        status = "🟡 中性謹慎 (Neutral)"
    else:
        status = "🟢 風險低/偏樂觀 (Greed)"

    col1.metric("綜合先期風險指數", f"{risk_score} / 100", f"{risk_change:+} 分", delta_color="inverse")
    col2.metric("當前市場狀態", status)
    col3.metric("10年期美債殖利率", f"{latest['US10Y']:.2f}%", f"{latest['US10Y'] - prev['US10Y']:+.2f}%", delta_color="inverse")
    col4.metric("VIX 恐慌指數", f"{latest['VIX']:.2f}", f"{latest['VIX'] - prev['VIX']:+.2f}", delta_color="inverse")

    st.markdown("---")

    # 圖表
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Risk_Index'],
        mode='lines', name='綜合先期風險指數',
        line=dict(color='crimson', width=2.5)
    ))

    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="高度風險警戒線 (70)")
    fig.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="中性分界線 (45)")

    fig.update_layout(
        title="<b>先期風險指數歷史走勢圖 (均值回歸與突發風險監測)</b>",
        xaxis_title="日期",
        yaxis_title="風險指數 (0-100)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # 頁籤數據
    st.subheader("📊 三大指標分項數據")
    tab1, tab2, tab3 = st.tabs(["10年期美債殖利率 (^TNX)", "VIX 恐慌指數", "信用利差比值 (HYG/LQD)"])

    with tab1:
        st.line_chart(df['US10Y'])
    with tab2:
        st.line_chart(df['VIX'])
    with tab3:
        st.line_chart(df['Credit_Ratio'])

else:
    st.error("⚠️ 無法順利讀取數據，可能是 Yahoo Finance 暫時限制存取或正在休市。請嘗試重新整理頁面。")
