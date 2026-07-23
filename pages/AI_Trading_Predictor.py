import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
import json
import time
import random

warnings.filterwarnings('ignore')

from recommendation_engine import fetch_stock_data, fetch_market_data, calculate_all_signals, generate_forecast

st.set_page_config(page_title="AI Code Predict & Stock Predictor", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center;color:#1f77b4;font-size:2.2rem;'>🤖 AI Code Predict & Stock Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;font-size:1rem;'>Unified Quantitative Signals + Technical Forecast + Multi-Agent LLM Trading Predictions</p>", unsafe_allow_html=True)

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
st.sidebar.markdown("### 🤖 AI Agent Settings")

provider_choice = st.sidebar.selectbox(
    "LLM Provider",
    ["OpenRouter", "OpenAI", "Anthropic", "Ollama (Local)"],
    index=0
)

api_key = ""
ollama_url = "http://localhost:11434"

if provider_choice == "OpenRouter":
    api_key = st.sidebar.text_input("OpenRouter API Key", type="password", placeholder="sk-or-v1-...", help="Get a key at openrouter.ai")
    models = [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-large-2411",
        "google/gemini-flash-1.5",
        "deepseek/deepseek-r1"
    ]
    model_choice = st.sidebar.selectbox("Model", models, index=0)
elif provider_choice == "OpenAI":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...", help="Get a key at platform.openai.com")
    models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    model_choice = st.sidebar.selectbox("Model", models, index=0)
elif provider_choice == "Anthropic":
    api_key = st.sidebar.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...", help="Get a key at console.anthropic.com")
    models = ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-5-haiku-20241022"]
    model_choice = st.sidebar.selectbox("Model", models, index=0)
else:  # Ollama
    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
    model_choice = st.sidebar.text_input("Model Name", value="llama3.2", help="e.g. llama3.2, mistral, qwen2.5")

st.sidebar.markdown("---")
st.sidebar.markdown("<small>Data from Yahoo Finance & Multi-Agent LLMs</small>", unsafe_allow_html=True)

# ==================== LLM CALL & PARSE FUNCTIONS ====================
def call_llm(system_prompt, user_prompt, provider, api_key, model, ollama_url="http://localhost:11434"):
    if provider == "Ollama (Local)":
        try:
            response = requests.post(
                f"{ollama_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", ""), None
        except Exception as e:
            return None, f"Ollama Connection Error: {str(e)}. Ensure Ollama is running at {ollama_url}"

    elif provider == "OpenRouter":
        if not api_key or not api_key.startswith("sk-or"):
            return None, "Invalid API key. Please enter a valid OpenRouter key starting with 'sk-or-v1-'."
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://stock-predictor.streamlit.app",
            "X-Title": "Stock Predictor AI"
        }
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }
    elif provider == "OpenAI":
        if not api_key:
            return None, "Invalid API key. Please enter a valid OpenAI key."
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        endpoint = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }
    elif provider == "Anthropic":
        if not api_key:
            return None, "Invalid API key. Please enter a valid Anthropic key."
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        endpoint = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }
    else:
        return None, "Unsupported provider selected."

    for attempt in range(3):
        try:
            time.sleep(1.2 + random.uniform(0.3, 0.8))
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            if response.status_code == 429:
                time.sleep((attempt + 1) * 3)
                continue
            response.raise_for_status()
            data = response.json()
            if provider == "Anthropic":
                if "content" in data and isinstance(data["content"], list):
                    content_parts = [block.get("text", "") for block in data["content"] if isinstance(block, dict) and block.get("type") == "text"]
                    if content_parts:
                        return "".join(content_parts), None
                return None, f"Unexpected Anthropic response: {data}"
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                return content, None
            else:
                return None, f"Unexpected response: {data}"
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 2)
                continue
            return None, f"API Error: {str(e)}"
    return None, "Max retries exceeded. Please try again later."


def parse_agent_response(content):
    content_lower = content.lower()
    if any(word in content_lower for word in ["strong buy", "strong_buy", "strongbuy"]):
        signal = "STRONG BUY"
        score = 1.0
        emoji = "🟢"
    elif any(word in content_lower for word in ["buy", "bullish", "long", "accumulate"]):
        signal = "BUY"
        score = 0.5
        emoji = "🟢"
    elif any(word in content_lower for word in ["strong sell", "strong_sell", "strongsell"]):
        signal = "STRONG SELL"
        score = -1.0
        emoji = "🔴"
    elif any(word in content_lower for word in ["sell", "bearish", "short", "reduce"]):
        signal = "SELL"
        score = -0.5
        emoji = "🔴"
    else:
        signal = "HOLD"
        score = 0.0
        emoji = "🟡"
    return {"signal": signal, "score": score, "emoji": emoji, "reasoning": content}

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
mcap_str = f"${market_cap/1e9:.1f}B" if (market_cap and isinstance(market_cap, (int, float))) else "N/A"
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

future_dates, mean_f, p10, p90, future_support_levels, future_resistance_levels = generate_forecast(
    data,
    score,
    forecast_days,
    support_levels=results.get('support_levels', []),
    resistance_levels=results.get('resistance_levels', [])
)
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

forecast_support_price = future_support_levels[0][1] if future_support_levels else None
forecast_resistance_price = future_resistance_levels[0][1] if future_resistance_levels else None
forecast_mean_price = mean_f[-1] if len(mean_f) else None

if forecast_support_price is not None and forecast_resistance_price is not None:
    fs_pct = ((forecast_support_price - last_price) / last_price) * 100
    fr_pct = ((forecast_resistance_price - last_price) / last_price) * 100
    fm_pct = ((forecast_mean_price - last_price) / last_price) * 100 if forecast_mean_price is not None else None
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Forecast Support (10%)", f"${forecast_support_price:.2f}", f"{fs_pct:+.1f}%")
    with c2:
        st.metric("Forecast Mean", f"${forecast_mean_price:.2f}", f"{fm_pct:+.1f}%")
    with c3:
        st.metric("Forecast Resistance (90%)", f"${forecast_resistance_price:.2f}", f"{fr_pct:+.1f}%")

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
    subplot_titles=(f"{company_name} ({stock_symbol}) — Price & {prediction_label} Technical AI Forecast", "MACD", "RSI")
)

fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Price", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=sma20.index, y=sma20.values, name="SMA 20", line=dict(color="orange", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=sma50.index, y=sma50.values, name="SMA 50", line=dict(color="purple", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=bb_upper.index, y=bb_upper.values, name="BB Upper", line=dict(color="rgba(255,0,0,0.3)", width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=bb_lower.index, y=bb_lower.values, name="BB Lower", line=dict(color="rgba(255,0,0,0.3)", width=1), fill="tonexty", fillcolor="rgba(255,0,0,0.05)", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=vwap.index, y=vwap.values, name="VWAP", line=dict(color="cyan", width=1)), row=1, col=1)

support_levels = results.get('support_levels', [])
resistance_levels = results.get('resistance_levels', [])

def _extract_sr(level):
    if isinstance(level, (list, tuple)) and len(level) == 2:
        return level
    return (None, level)

for idx, entry in enumerate(support_levels):
    dt, level = _extract_sr(entry)
    fig.add_hline(y=level, line=dict(color="green", width=1, dash="dot"), annotation_text=f"Support {idx+1}: ${level:.2f}", annotation_position="bottom left", row=1, col=1)
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=close.index[0],
        x1=future_dates[-1],
        y0=level * 0.995,
        y1=level * 1.005,
        fillcolor="rgba(0,128,0,0.08)",
        line=dict(width=0),
        row=1,
        col=1
    )
for idx, entry in enumerate(resistance_levels):
    dt, level = _extract_sr(entry)
    fig.add_hline(y=level, line=dict(color="red", width=1, dash="dot"), annotation_text=f"Resistance {idx+1}: ${level:.2f}", annotation_position="top left", row=1, col=1)
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=close.index[0],
        x1=future_dates[-1],
        y0=level * 0.995,
        y1=level * 1.005,
        fillcolor="rgba(255,0,0,0.08)",
        line=dict(width=0),
        row=1,
        col=1
    )

if future_support_levels:
    fs_label, fs_value = future_support_levels[0]
    fig.add_hline(y=fs_value, line=dict(color="green", width=1, dash="dash"), annotation_text=fs_label, annotation_position="bottom right", row=1, col=1)
if future_resistance_levels:
    fr_label, fr_value = future_resistance_levels[0]
    fig.add_hline(y=fr_value, line=dict(color="red", width=1, dash="dash"), annotation_text=fr_label, annotation_position="top right", row=1, col=1)

buy_signals = [s for s in results.get('trade_signals', []) if s['signal'] == 'Buy']
sell_signals = [s for s in results.get('trade_signals', []) if s['signal'] == 'Sell']
fig.add_trace(go.Scatter(
    x=[s['date'] for s in buy_signals],
    y=[s['price'] for s in buy_signals],
    mode='markers',
    marker=dict(symbol='triangle-up', color='green', size=12),
    name='Buy Signal',
    hovertemplate='Buy: %{y:.2f}<br>%{x|%Y-%m-%d}<br>%{text}',
    text=[s['reason'] for s in buy_signals]
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=[s['date'] for s in sell_signals],
    y=[s['price'] for s in sell_signals],
    mode='markers',
    marker=dict(symbol='triangle-down', color='red', size=12),
    name='Sell Signal',
    hovertemplate='Sell: %{y:.2f}<br>%{x|%Y-%m-%d}<br>%{text}',
    text=[s['reason'] for s in sell_signals]
), row=1, col=1)

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
st.subheader(f"🔍 Technical Unified Score: {score:+.2f}/1.00 — {signal_emoji} {signal_text}")

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
