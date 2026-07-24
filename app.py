import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 頁面標題與配置
st.set_page_config(page_title="美股先期情緒警戒 Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("🛡️ 美股先期風險警戒儀表板 (Early Warning Risk Dashboard)")
st.caption("整合美債殖利率 (10Y)、VIX 波動率與高收益債信用利差 (HYG/LQD)，提早 24-48 小時感知市場氣氛變革。")

# 側邊欄設定
st.sidebar.header("⚙️ 參數設定")
period_options = {"1 個月": "1mo", "3 個月": "3mo", "6 個月": "6mo", "1 年": "1y"}
selected_period = st.sidebar.selectbox("資料歷史區間", list(period_options.keys()), index=1)
period = period_options[selected_period]

@st.cache_data(ttl=3600)  # 快取 1 小時
def load_data(period_str):
    tickers = {
        'US10Y': '^TNX',   # 10年期美債殖利率
        'VIX': '^VIX',     # CBOE 恐慌指數
        'HYG': 'HYG',     # 高收益債 ETF
        'LQD': 'LQD'      # 投資級債 ETF
    }
    
    df = pd.DataFrame()
    for name, ticker in tickers.items():
        data = yf.Ticker(ticker).history(period=period_str)
        if not data.empty:
            df[name] = data['Close']
            
    df = df.dropna()
    
    # 計算 HYG / LQD 比值（代表風險偏好；比值下滑 = 風險升高）
    df['Credit_Ratio'] = df['HYG'] / df['LQD']
    
    # 計算指標變化與標準化 (Z-Score) 進行綜合
    # 1. 10Y 殖利率高點 = 風險高 (+)
    # 2. VIX 高點 = 風險高 (+)
    # 3. Credit Ratio 走低 = 風險高 (-) -> 取負值
    
    # 使用近 20 日移動算 Z-Score 衡量短期突發風險
    window = 20
    z_us10y = (df['US10Y'] - df['US10Y'].rolling(window).mean()) / df['US10Y'].rolling(window).std()
    z_vix = (df['VIX'] - df['VIX'].rolling(window).mean()) / df['VIX'].rolling(window).std()
    z_credit = -1 * (df['Credit_Ratio'] - df['Credit_Ratio'].rolling(window).mean()) / df['Credit_Ratio'].rolling(window).std()
    
    # 綜合風險分數 (Weights: VIX 40%, 10Y 30%, Credit 30%)
    composite_z = (z_vix * 0.4) + (z_us10y * 0.3) + (z_credit * 0.3)
    
    # 映射至 0 - 100 分數
    # Z-score 範圍約 -3 到 +3，轉換為 0-100
    df['Risk_Index'] = (1 / (1 + np.exp(-composite_z))) * 100
    
    return df.dropna()

with st.spinner("正在抓取最新市場數據..."):
    df = load_data(period)

if not df.empty:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    risk_score = round(latest['Risk_Index'], 1)
    risk_change = round(latest['Risk_Index'] - prev['Risk_Index'], 1)
    
    # 儀表板頂部卡片
    col1, col2, col3, col4 = st.columns(4)
    
    # 警戒等級判斷
    if risk_score >= 70:
        status = "🔴 高度警戒 (Fear)"
        status_color = "red"
    elif risk_score >= 45:
        status = "🟡 中性謹慎 (Neutral)"
        status_color = "orange"
    else:
        status = "🟢 風險低/偏樂觀 (Greed)"
        status_color = "green"

    col1.metric("綜合先期風險指數", f"{risk_score} / 100", f"{risk_change:+} 分", delta_color="inverse")
    col2.metric("當前市場狀態", status)
    col3.metric("10年期美債殖利率", f"{latest['US10Y']:.2f}%", f"{latest['US10Y'] - prev['US10Y']:+.2f}%", delta_color="inverse")
    col4.metric("VIX 恐慌指數", f"{latest['VIX']:.2f}", f"{latest['VIX'] - prev['VIX']:+.2f}", delta_color="inverse")

    st.markdown("---")

    # 繪製圖表
    fig = go.Figure()

    # 綜合指數線
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Risk_Index'],
        mode='lines', name='綜合先期風險指數',
        line=dict(color='crimson', width=2.5)
    ))

    # 風險警戒線 (70 分)
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

    # 各分項細節折線圖
    st.subheader("📊 三大指標分項數據")
    tab1, tab2, tab3 = st.tabs(["10年期美債殖利率 (^TNX)", "VIX 恐慌指數", "信用利差比值 (HYG/LQD)"])

    with tab1:
        st.line_chart(df['US10Y'])
    with tab2:
        st.line_chart(df['VIX'])
    with tab3:
        st.line_chart(df['Credit_Ratio'])

else:
    st.error("無法取得數據，請稍後再試。")