import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from recommendation_engine import fetch_stock_data, fetch_market_data, calculate_all_signals, generate_forecast

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

st.markdown("<h1 style='text-align:center;color:#1f77b4;font-size:2.2rem;'>📈 Stock Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;font-size:1rem;'>Smart Score + Trend Meter + Alpha Signal + Technical Rating combined into one unified prediction.</p>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.markdown("---")

stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", placeholder="e.g. AAPL, TSLA, MSFT").upper().strip()

prediction_options = {
    "1 Day": 1, "3 Days": 3, "1 Week": 5, "2 Weeks": 10,
    "1 Month": 21, "2 Months": 42, "3 Months": 63,
    "6 Months": 126, "1 Year": 252
}
prediction_label = st.sidebar.selectbox("Prediction Horizon", list(prediction_options.keys()), index=4)
forecast_days = prediction_options[prediction_label]

st.sidebar.markdown("---")
st.sidebar.markdown("<small>Data from Yahoo Finance</small>", unsafe_allow_html=True)

if not stock_symbol:
    st.info("👈 Enter a stock symbol in the sidebar to begin.")
    st.stop()

# ==================== FETCH DATA ====================
with st.spinner(f"Analyzing {stock_symbol}..."):
    data, info, ticker = fetch_stock_data(stock_symbol, period="2y")
    spy_close = fetch_market_data(period="2y")

if data is None or data.empty:
    st.error(f"❌ No data found for '{stock_symbol}'. Try another ticker like AAPL, TSLA, MSFT, NVDA, AMZN.")
    st.stop()

close = data['Close']
high = data['High']
low = data['Low']
last_price = close.iloc[-1]

# ==================== CALCULATE ALL SIGNALS ====================
results = calculate_all_signals(data, info, ticker, spy_close)
unified = results['unified']
score = unified['score']
signal_text = unified['signal']
signal_emoji = unified['emoji']
color = unified['color']
analyst_target = unified['analyst_target']
analyst_count = unified['analyst_count']

# ==================== STOCK NOTE CARD ====================
st.markdown("---")
company_name = info.get('shortName', stock_symbol) if info else stock_symbol
sector = info.get('sector', 'N/A') if info else 'N/A'
industry = info.get('industry', 'N/A') if info else 'N/A'
market_cap = info.get('marketCap', None)
mcap_str = f"${market_cap/1e9:.1f}B" if market_cap else "N/A"
pe_ratio = info.get('trailingPE', 'N/A') if info else 'N/A'
employees = info.get('fullTimeEmployees', None)
emp_str = f"{employees:,}" if employees else "N/A"
website = info.get('website', '') if info else ''

summary = info.get('longBusinessSummary', '') if info else ''
summary_short = summary[:350] + "..." if len(summary) > 350 else summary

st.markdown(f"""
<div style="background:linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);border-radius:16px;padding:20px;margin:10px 0;border-left:6px solid {color};box-shadow:0 2px 8px rgba(0,0,0,0.08);">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
            <h2 style="margin:0 0 6px 0;color:#1f77b4;font-size:1.6rem;">{company_name} ({stock_symbol})</h2>
            <p style="margin:0 0 10px 0;color:#555;font-size:0.95rem;">
                <b>Sector:</b> {sector} &nbsp;|&nbsp; <b>Industry:</b> {industry} &nbsp;|&nbsp; 
                <b>Market Cap:</b> {mcap_str} &nbsp;|&nbsp; <b>P/E:</b> {pe_ratio if isinstance(pe_ratio, str) else f'{pe_ratio:.1f}'} &nbsp;|&nbsp;
                <b>Employees:</b> {emp_str}
            </p>
        </div>
        <div style="text-align:right;">
            <div style="font-size:2rem;font-weight:bold;color:{color};">{signal_emoji}</div>
            <div style="font-size:0.9rem;color:{color};font-weight:bold;">{signal_text}</div>
        </div>
    </div>
    <p style="margin:8px 0 0 0;color:#444;line-height:1.6;font-size:0.95rem;">{summary_short}</p>
    {f'<p style="margin:8px 0 0 0;font-size:0.85rem;"><a href="{website}" target="_blank">🌐 {website}</a></p>' if website else ''}
</div>
""", unsafe_allow_html=True)

# ==================== METRICS ROW ====================
change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
volatility = close.pct_change().std() * np.sqrt(252) * 100
period_return = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
yr_high = high.max()
yr_low = low.min()

future_dates, mean_f, p10, p90 = generate_forecast(data, score, forecast_days)
pred_price = mean_f[-1]
pred_change = ((pred_price - last_price) / last_price) * 100

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
with m1:
    st.metric("Price", f"${last_price:.2f}")
with m2:
    st.metric("1D Change", f"{change:+.2f}%")
with m3:
    st.metric(f"{prediction_label} Target", f"${pred_price:.2f}", f"{pred_change:+.1f}%")
with m4:
    st.metric("Smart Score", f"{unified['smart_score']:+.2f}")
with m5:
    st.metric("Trend Meter", f"{unified['trend_score']:+.2f}")
with m6:
    st.metric("Alpha Signal", f"{unified['alpha_score']:+.2f}")
with m7:
    st.metric("Tech Rating", f"{unified['tech_score']:+.2f}")

# ==================== MAIN CHART ====================
sma20 = close.rolling(20).mean()
sma50 = close.rolling(50).mean()
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
bb_sma = close.rolling(20).mean()
bb_std = close.rolling(20).std()
bb_upper = bb_sma + 2 * bb_std
bb_lower = bb_sma - 2 * bb_std
tp = (high + low + close) / 3
vwap = (tp * data['Volume']).cumsum() / data['Volume'].cumsum()

macd_line = ema12 - ema26
macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
macd_hist = macd_line - macd_signal_line

delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rsi = 100 - (100 / (1 + gain / loss))

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.55, 0.25, 0.20],
    subplot_titles=(f"{company_name} ({stock_symbol}) — Price & {prediction_label} Forecast", "MACD", "RSI")
)

fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Price", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=sma20.index, y=sma20.values, name="SMA 20", line=dict(color="orange", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=sma50.index, y=sma50.values, name="SMA 50", line=dict(color="purple", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=bb_upper.index, y=bb_upper.values, name="BB Upper", line=dict(color="rgba(255,0,0,0.3)", width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=bb_lower.index, y=bb_lower.values, name="BB Lower", line=dict(color="rgba(255,0,0,0.3)", width=1), fill="tonexty", fillcolor="rgba(255,0,0,0.05)", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=vwap.index, y=vwap.values, name="VWAP", line=dict(color="cyan", width=1)), row=1, col=1)

fig.add_trace(go.Scatter(x=future_dates, y=mean_f, name=f"Forecast ({prediction_label})", line=dict(color="#2ca02c", width=2.5, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=list(future_dates)+list(future_dates)[::-1], y=list(p90)+list(p10)[::-1], fill="toself", fillcolor="rgba(44,160,44,0.15)", line=dict(color="rgba(0,0,0,0)"), name="Confidence (10%-90%)", hoverinfo="skip"), row=1, col=1)

if analyst_target and not np.isnan(analyst_target):
    fig.add_hline(y=analyst_target, line=dict(color="gold", width=2, dash="dashdot"), annotation_text=f"Analyst Target: ${analyst_target:.2f}", annotation_position="top right", row=1, col=1)

fig.add_vline(x=close.index[-1], line=dict(color="gray", width=1, dash="dash"), annotation_text="Forecast Start", annotation_position="top", row=1, col=1)

fig.add_trace(go.Scatter(x=data.index, y=macd_line.values, name="MACD", line=dict(color="blue", width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=macd_signal_line.values, name="Signal", line=dict(color="red", width=1)), row=2, col=1)
fig.add_trace(go.Bar(x=data.index, y=macd_hist.values, name="Histogram", marker_color=['green' if h >= 0 else 'red' for h in macd_hist]), row=2, col=1)
fig.add_hline(y=0, line=dict(color="black", width=0.5), row=2, col=1)

fig.add_trace(go.Scatter(x=data.index, y=rsi.values, name="RSI", line=dict(color="purple", width=1.5)), row=3, col=1)
fig.add_hline(y=70, line=dict(color="red", width=1, dash="dash"), row=3, col=1)
fig.add_hline(y=30, line=dict(color="green", width=1, dash="dash"), row=3, col=1)

fig.update_layout(
    height=780,
    template="plotly_white",
    hovermode="x unified",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_rangeslider_visible=False,
    margin=dict(l=50, r=50, t=100, b=50)
)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="MACD", row=2, col=1)
fig.update_yaxes(title_text="RSI", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# ==================== SIGNAL BREAKDOWN ====================
st.markdown("---")
st.subheader(f"🔍 Unified Score: {score:+.2f}/1.00 — {signal_emoji} {signal_text}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**📊 Smart Score** (40% weight)")
    ss = unified['smart_score']
    st.progress((ss + 1) / 2, text=f"{ss:+.2f}")
    for k, v in results['smart_score'].items():
        st.write(f"• {k}: {v}")
with c2:
    st.markdown("**📈 Trend Meter** (25% weight)")
    ts = unified['trend_score']
    st.progress((ts + 1) / 2, text=f"{ts:+.2f}")
    for k, v in results['trend_meter'].items():
        st.write(f"• {k}: {v}")
with c3:
    st.markdown("**⚡ Alpha Signal** (20% weight)")
    als = unified['alpha_score']
    st.progress((als + 1) / 2, text=f"{als:+.2f}")
    for k, v in results['alpha_signal'].items():
        st.write(f"• {k}: {v}")
with c4:
    st.markdown("**🔮 Technical Rating** (15% weight)")
    tr = unified['tech_score']
    st.progress((tr + 1) / 2, text=f"{tr:+.2f}")
    for k, v in results['technical_rating'].items():
        st.write(f"• {k}: {v}")
    if results['patterns']:
        st.markdown("**📋 Patterns Detected:**")
        for pattern, direction in results['patterns'].items():
            st.write(f"• {pattern}: {direction}")

if analyst_count > 0 or analyst_target:
    st.markdown("---")
    acol1, acol2, acol3 = st.columns(3)
    with acol1:
        st.metric("Analysts Covering", f"{analyst_count}")
    with acol2:
        if analyst_target and not np.isnan(analyst_target):
            upside = ((analyst_target - last_price) / last_price) * 100
            st.metric("Analyst Target", f"${analyst_target:.2f}", f"{upside:+.1f}%")
    with acol3:
        st.metric("52W Range", f"${yr_low:.2f} - ${yr_high:.2f}")

with st.expander("📋 View Forecast Data"):
    forecast_df = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        "Predicted": mean_f.round(2),
        "Low (10%)": p10.round(2),
        "High (90%)": p90.round(2)
    })
    st.dataframe(forecast_df, use_container_width=True)
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", csv, f"{stock_symbol}_forecast.csv", "text/csv")

st.markdown("---")
st.markdown("<center><small>Data from Yahoo Finance | Smart Score + Trend Meter + Alpha Signal + Technical Rating combined | Not financial advice</small></center>", unsafe_allow_html=True)
