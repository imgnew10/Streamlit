import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import warnings
warnings.filterwarnings('ignore')

from recommendation_engine import fetch_stock_data, calculate_all_signals

st.set_page_config(page_title="Top 50 Stocks", page_icon="🏆", layout="wide")

st.markdown("<h1 style='text-align:center;color:#2ca02c;'>🏆 Top 50 Recommended Stocks</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;'>Stocks ranked by our Unified Score using Smart Score + Trend Meter + Alpha Signal + Technical Rating.</p>", unsafe_allow_html=True)

STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "V", "XOM", "WMT", "JPM", "PG", "MA", "LLY", "HD", "CVX", "MRK",
    "PEP", "KO", "BAC", "ABBV", "AVGO", "PFE", "TMO", "COST", "DIS", "CSCO",
    "VZ", "ADBE", "WFC", "ACN", "ABT", "CRM", "LIN", "NKE", "TXN", "NEE",
    "PM", "RTX", "HON", "BMY", "QCOM", "UPS", "LOW", "ORCL", "IBM", "GS"
]

@st.cache_data(ttl=3600)
def analyze_all_stocks():
    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(STOCK_UNIVERSE):
        status.text(f"Analyzing {symbol}... ({i+1}/{len(STOCK_UNIVERSE)})")
        progress.progress((i + 1) / len(STOCK_UNIVERSE))

        try:
            data, info, ticker = fetch_stock_data(symbol, period="6mo")
            if data is None or data.empty or len(data) < 50:
                continue

            results_data = calculate_all_signals(data, info, ticker, spy_close=None)
            if results_data is None:
                continue

            unified = results_data['unified']
            score = unified['score']
            signal_text = unified['signal']
            signal_emoji = unified['emoji']
            color = unified['color']
            analyst_target = unified['analyst_target']
            analyst_count = unified['analyst_count']

            last_price = data['Close'].iloc[-1]
            change_1d = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            change_1m = ((data['Close'].iloc[-1] - data['Close'].iloc[-21]) / data['Close'].iloc[-21]) * 100 if len(data) >= 21 else 0
            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100

            upside = None
            if analyst_target and not np.isnan(analyst_target):
                upside = ((analyst_target - last_price) / last_price) * 100

            results.append({
                "Symbol": symbol,
                "Name": info.get('shortName', symbol) if info else symbol,
                "Price": last_price,
                "1D Change": change_1d,
                "1M Change": change_1m,
                "Score": score,
                "Signal": f"{signal_emoji} {signal_text}",
                "Color": color,
                "Analyst Target": analyst_target,
                "Upside %": upside,
                "Analysts": analyst_count,
                "Volatility": volatility,
                "Sector": info.get('sector', 'N/A') if info else 'N/A'
            })
        except Exception:
            continue

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df

st.sidebar.markdown("## 🔍 Filter")
min_score = st.sidebar.slider("Minimum Unified Score", -1.0, 1.0, -0.5, 0.1)
max_volatility = st.sidebar.slider("Max Volatility %", 0.0, 100.0, 50.0, 5.0)
sector_filter = st.sidebar.multiselect("Sector", ["All", "Technology", "Healthcare", "Financials", "Consumer", "Energy", "Industrials"], default=["All"])

if st.sidebar.button("🚀 Analyze All 50 Stocks", type="primary"):
    df = analyze_all_stocks()

    if df.empty:
        st.error("Could not fetch data. Yahoo Finance may be rate limiting. Please wait a few minutes and try again.")
        st.stop()

    filtered = df[df["Score"] >= min_score]
    filtered = filtered[filtered["Volatility"] <= max_volatility]
    if "All" not in sector_filter:
        filtered = filtered[filtered["Sector"].isin(sector_filter)]

    st.markdown(f"### Showing {len(filtered)} of {len(df)} stocks")

    top5 = filtered.head(5)
    cols = st.columns(5)
    for idx, (_, row) in enumerate(top5.iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);border-radius:12px;padding:15px;text-align:center;border-top:4px solid {'#2ca02c' if row['Score'] > 0.2 else '#ffa500' if row['Score'] > -0.2 else '#dc3545'};">
                <h4 style="margin:0;color:#1f77b4;">#{idx+1} {row['Symbol']}</h4>
                <p style="margin:4px 0;font-size:0.85rem;color:#666;">{row['Name'][:15]}</p>
                <h3 style="margin:8px 0;color:#333;">${row['Price']:.2f}</h3>
                <p style="margin:0;font-size:1.1rem;font-weight:bold;color:{'#2ca02c' if row['Score'] > 0.2 else '#ffa500' if row['Score'] > -0.2 else '#dc3545'};">{row['Signal']}</p>
                <p style="margin:4px 0;font-size:0.8rem;color:#888;">Score: {row['Score']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    display_df = filtered[["Symbol", "Name", "Price", "1D Change", "1M Change", "Score", "Signal", "Analyst Target", "Upside %", "Analysts", "Volatility", "Sector"]].copy()
    display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:.2f}")
    display_df["1D Change"] = display_df["1D Change"].apply(lambda x: f"{x:+.2f}%")
    display_df["1M Change"] = display_df["1M Change"].apply(lambda x: f"{x:+.2f}%")
    display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f}")
    display_df["Analyst Target"] = display_df["Analyst Target"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    display_df["Upside %"] = display_df["Upside %"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
    display_df["Volatility"] = display_df["Volatility"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(display_df, use_container_width=True, height=600)

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Results", csv, "top50_recommended_stocks.csv", "text/csv")

    st.markdown("---")
    st.subheader("📊 Unified Score Distribution")

    import plotly.express as px
    fig = px.histogram(
        df, x="Score", nbins=20,
        color="Score",
        color_continuous_scale=[(0, "red"), (0.5, "orange"), (1, "green")],
        title="How Unified Scores Are Distributed Across 50 Stocks"
    )
    fig.add_vline(x=0.2, line_dash="dash", line_color="green", annotation_text="Buy Threshold")
    fig.add_vline(x=-0.2, line_dash="dash", line_color="red", annotation_text="Sell Threshold")
    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Click **Analyze All 50 Stocks** in the sidebar to start. Analysis takes ~2 minutes due to rate limiting protection.")

st.markdown("---")
st.markdown("<center><small>Data from Yahoo Finance | Rate-limited requests to avoid errors | Not financial advice</small></center>", unsafe_allow_html=True)
