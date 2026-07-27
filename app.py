import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="美股 7 大先期風險儀表板 Pro", layout="wide", initial_sidebar_state="expanded")

st.title("🛡️ 美股 7 大先期風險警戒儀表板 (EWRI Pro)")
st.caption("融合 VIX/VIX1D 衍生品結構、美債殖利率、美元流動性、信用利差、半導體 ADR 與小型股相對強弱，提早 24-48 小時預警。")

st.sidebar.header("⚙️ 參數設定")
days_options = {"1 個月 (30天)": 30, "3 個月 (90天)": 90, "6 個月 (180天)": 180, "1 年 (365天)": 365}
selected_option = st.sidebar.selectbox("資料歷史區間", list(days_options.keys()), index=1)
days = days_options[selected_option]

@st.cache_data(ttl=1800)
def load_pro_data(days_back):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 40)
    
    # 7 大指標來源
    tickers = {
        'US10Y': '^TNX',       # 10年期美債殖利率
        'VIX': '^VIX',         # 30天恐慌指數
        'VIX1D': '^VIX1D',     # 1天期極速恐慌指數
        'HYG': 'HYG',         # 高收益債
        'LQD': 'LQD',         # 投資級債
        'DXY': 'DX-Y.NYB',    # 美元指數
        'TSM': 'TSM',         # 台積電 ADR (科技龍頭先導)
        'IWM': 'IWM',         # 小型股
        'SPY': 'SPY'          # 標普 500
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
        
    # 特徵工程與衍生指標計算
    df['Credit_Ratio'] = df['HYG'] / df['LQD']      # 信用偏好 (越高越安全)
    df['Small_Large_Ratio'] = df['IWM'] / df['SPY'] # 小型股相對強弱 (越高越偏風險)
    
    # 計算 20 日滾動 Z-Score (正規化極端值)
    w = 20
    z_vix = (df['VIX'] - df['VIX'].rolling(w).mean()) / df['VIX'].rolling(w).std()
    z_vix1d = (df['VIX1D'] - df['VIX1D'].rolling(w).mean()) / df['VIX1D'].rolling(w).std()
    z_us10y = (df['US10Y'] - df['US10Y'].rolling(w).mean()) / df['US10Y'].rolling(w).std()
    z_dxy = (df['DXY'] - df['DXY'].rolling(w).mean()) / df['DXY'].rolling(w).std()
    
    # 負向指標：數值下跌代表風險升高 (取 negative Z-score)
    z_credit = -1 * (df['Credit_Ratio'] - df['Credit_Ratio'].rolling(w).mean()) / df['Credit_Ratio'].rolling(w).std()
    z_tsm = -1 * (df['TSM'] - df['TSM'].rolling(w).mean()) / df['TSM'].rolling(w).std()
    z_small = -1 * (df['Small_Large_Ratio'] - df['Small_Large_Ratio'].rolling(w).mean()) / df['Small_Large_Ratio'].rolling(w).std()
    
    # 權重分配 (總和 100%)
    # VIX1D(20%) + VIX(15%) + 美債(20%) + 美元流動性(15%) + 信用利差(15%) + TSM先導(10%) + 小型股(5%)
    composite_z = (
        (z_vix1d * 0.20) + 
        (z_vix * 0.15) + 
        (z_us10y * 0.20) + 
        (z_dxy * 0.15) + 
        (z_credit * 0.15) + 
        (z_tsm * 0.10) + 
        (z_small * 0.05)
    )
    
    # Sigmoid 映射至 0 - 100
    df['Risk_Index'] = (1 / (1 + np.exp(-composite_z))) * 100
    return df.tail(days_back).dropna()

with st.spinner("正在計算 7 大先期指標與流動性數據..."):
    df = load_pro_data(days)

if not df.empty and len(df) >= 2:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = round(latest['Risk_Index'], 1)
    score_change = round(latest['Risk_Index'] - prev['Risk_Index'], 1)
    
    # 狀態面板
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if score >= 70:
        status = "🔴 高度風險警戒"
    elif score >= 45:
        status = "🟡 中性防禦觀望"
    else:
        status = "🟢 樂觀/風險較低"

    col1.metric("綜合先期風險指數", f"{score} / 100", f"{score_change:+} 分", delta_color="inverse")
    col2.metric("當前市場狀態", status)
    col3.metric("10年美債殖利率", f"{latest['US10Y']:.2f}%", f"{latest['US10Y']-prev['US10Y']:+.2f}%", delta_color="inverse")
    col4.metric("VIX1D 極速恐慌", f"{latest['VIX1D']:.2f}", f"{latest['VIX1D']-prev['VIX1D']:+.2f}", delta_color="inverse")
    col5.metric("美元指數 (DXY)", f"{latest['DXY']:.2f}", f"{latest['DXY']-prev['DXY']:+.2f}", delta_color="inverse")

    st.markdown("---")

    # 走勢圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Risk_Index'], mode='lines', name='7大指標綜合警戒值', line=dict(color='crimson', width=2.5)))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70 警戒線")
    fig.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="45 中性線")
    fig.update_layout(title="<b>7 大先期指標綜合趨勢圖</b>", yaxis=dict(range=[0, 100]), template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # 分項子頁籤
    st.subheader("🔍 先期子指標細節觀察")
    t1, t2, t3, t4 = st.tabs(["短線情緒 (VIX vs VIX1D)", "流動性 (美債10Y & 美元DXY)", "信用與小型股偏好", "科技龍頭先導 (TSM)"])
    
    with t1:
        st.line_chart(df[['VIX', 'VIX1D']])
    with t2:
        st.line_chart(df[['US10Y', 'DXY']])
    with t3:
        st.line_chart(df[['Credit_Ratio', 'Small_Large_Ratio']])
    with t4:
        st.line_chart(df['TSM'])
else:
    st.error("⚠️ 數據獲取失敗，請重新整理。")
