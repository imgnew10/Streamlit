import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import time
import random
import warnings

from recommendation_engine import fetch_stock_data, fetch_market_data, calculate_all_signals, generate_forecast

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Trading Prediction", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center;color:#1f77b4;font-size:2.2rem;'>🤖 AI Trading Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;font-size:1rem;'>Forecast prices using technical signals, AI agent analysis, and a hybrid buy/sell indicator.</p>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.markdown("---")

stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", placeholder="e.g. AAPL, TSLA, MSFT").upper().strip()

prediction_options = {
    "1 Day": 1,
    "3 Days": 3,
    "1 Week": 5,
    "2 Weeks": 10,
    "1 Month": 21,
    "2 Months": 42,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
}
prediction_label = st.sidebar.selectbox("Prediction Horizon", list(prediction_options.keys()), index=4)
forecast_days = prediction_options[prediction_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Provider")

provider_choice = st.sidebar.selectbox("Provider", ["OpenRouter", "OpenAI", "Anthropic", "Ollama (Local)"], index=0)

api_key = ""
ollama_url = "http://localhost:11434"

if provider_choice == "OpenRouter":
    api_key = st.sidebar.text_input("OpenRouter API Key", type="password", placeholder="sk-or-v1-...", help="Get a key at openrouter.ai/keys")
    model_options = [
        "openrouter/auto",
        "openrouter/free",
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.3-70b-instruct",
    ]
elif provider_choice == "OpenAI":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...", help="Get a key at platform.openai.com/api-keys")
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
elif provider_choice == "Anthropic":
    api_key = st.sidebar.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...", help="Get a key at console.anthropic.com/settings/keys")
    model_options = ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
else:
    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434", help="Local Ollama endpoint")
    model_options = ["llama3.2", "mistral", "qwen2.5:7b"]

model_choice = st.sidebar.selectbox("Model", model_options, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("<small>Combined price forecasting, technical signals, and AI trade signal analysis.</small>", unsafe_allow_html=True)

# ==================== LLM FUNCTIONS ====================

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
        except Exception as exc:
            return None, f"Ollama error: {exc}"

    if provider == "OpenRouter":
        if not api_key or not api_key.startswith("sk-or"):
            return None, "Enter a valid OpenRouter key starting with sk-or-v1-."
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800,
        }
    elif provider == "OpenAI":
        if not api_key:
            return None, "Enter a valid OpenAI API key."
        endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800,
        }
    elif provider == "Anthropic":
        if not api_key:
            return None, "Enter a valid Anthropic API key."
        endpoint = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.7,
            "max_tokens": 800,
        }
    else:
        return None, "Unsupported provider selected."

    for attempt in range(3):
        try:
            time.sleep(1.0 + random.uniform(0.2, 0.6))
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            if response.status_code == 429:
                time.sleep((attempt + 1) * 3)
                continue
            response.raise_for_status()
            data = response.json()
            if provider == "Anthropic":
                if "content" in data and isinstance(data["content"], list):
                    return "".join([block.get("text", "") for block in data["content"] if isinstance(block, dict)]), None
                return None, f"Unexpected Anthropic response: {data}"
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"], None
            return None, f"Unexpected response: {data}"
        except requests.exceptions.RequestException as exc:
            if attempt < 2:
                time.sleep((attempt + 1) * 2)
                continue
            return None, f"API Error: {exc}"
    return None, "Max retries exceeded."


def parse_agent_response(content):
    text = content.lower()
    if any(token in text for token in ["strong buy", "strong_buy", "strongbuy"]):
        return {"signal": "STRONG BUY", "score": 1.0, "emoji": "🟢", "reasoning": content}
    if any(token in text for token in ["buy", "bullish", "long", "accumulate"]):
        return {"signal": "BUY", "score": 0.5, "emoji": "🟢", "reasoning": content}
    if any(token in text for token in ["strong sell", "strong_sell", "strongsell"]):
        return {"signal": "STRONG SELL", "score": -1.0, "emoji": "🔴", "reasoning": content}
    if any(token in text for token in ["sell", "bearish", "short", "reduce"]):
        return {"signal": "SELL", "score": -0.5, "emoji": "🔴", "reasoning": content}
    return {"signal": "HOLD", "score": 0.0, "emoji": "🟡", "reasoning": content}


def aggregate_indicator_signal(technical_score, ai_score):
    combined = technical_score * 0.5 + ai_score * 0.5
    if combined >= 0.70:
        return "STRONG BUY", combined
    if combined >= 0.25:
        return "BUY", combined
    if combined <= -0.70:
        return "STRONG SELL", combined
    if combined <= -0.25:
        return "SELL", combined
    return "HOLD", combined


if provider_choice != "Ollama (Local)" and not api_key:
    st.info("Enter your API key in the sidebar to enable AI agent analysis.")
    st.stop()

if not stock_symbol:
    st.info("Enter a stock symbol in the sidebar to begin.")
    st.stop()

# ==================== FETCH MARKET DATA ====================
with st.spinner(f"Fetching market data for {stock_symbol}..."):
    data, info, ticker = fetch_stock_data(stock_symbol, period="2y")
    spy_close = fetch_market_data(period="2y")

if data is None or data.empty:
    st.error(f"No data found for '{stock_symbol}'. Try another ticker like AAPL, TSLA, MSFT, NVDA.")
    st.stop()

close = data["Close"]
high = data["High"]
low = data["Low"]
volume = data["Volume"]
last_price = close.iloc[-1]

results = calculate_all_signals(data, info, ticker, spy_close)
unified = results["unified"]
technical_score = unified["score"]
signal_text = unified["signal"]
signal_emoji = unified["emoji"]
color = unified["color"]

company_name = info.get("shortName", stock_symbol) if info else stock_symbol
sector = info.get("sector", "N/A") if info else "N/A"
industry = info.get("industry", "N/A") if info else "N/A"
market_cap = info.get("marketCap", None)
mcap_str = f"${market_cap/1e9:.1f}B" if (market_cap and isinstance(market_cap, (int, float))) else "N/A"
pe_ratio = info.get("trailingPE", "N/A") if info else "N/A"
employees = info.get("fullTimeEmployees", None)
emp_str = f"{employees:,}" if employees else "N/A"
website = info.get("website", "") if info else ""
summary = info.get("longBusinessSummary", "") if info else ""
summary_short = summary[:350] + "..." if len(summary) > 350 else summary

# ==================== FORECAST ====================
future_dates, mean_f, p10, p90, future_support_levels, future_resistance_levels = generate_forecast(
    data,
    technical_score,
    forecast_days,
    support_levels=results.get("support_levels", []),
    resistance_levels=results.get("resistance_levels", []),
)
pred_price = mean_f[-1]
pred_change = ((pred_price - last_price) / last_price) * 100

# ==================== PAGE HEADER CARD ====================
st.markdown("---")
st.markdown(f"""
<div style="background:linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);border-radius:16px;padding:22px;margin:10px 0;border-left:6px solid {color};box-shadow:0 2px 14px rgba(0,0,0,0.08);">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;">
        <div style="max-width:70%;">
            <h2 style="margin:0 0 8px 0;color:#1f77b4;font-size:1.7rem;">{company_name} ({stock_symbol})</h2>
            <p style="margin:0 0 10px 0;color:#555;font-size:0.95rem;">
                <b>Sector:</b> {sector} &nbsp;|&nbsp; <b>Industry:</b> {industry} &nbsp;|&nbsp;
                <b>Market Cap:</b> {mcap_str} &nbsp;|&nbsp; <b>P/E:</b> {pe_ratio if isinstance(pe_ratio, str) else f'{pe_ratio:.1f}'} &nbsp;|&nbsp;
                <b>Employees:</b> {emp_str}
            </p>
            <p style="margin:6px 0 0 0;color:#444;line-height:1.6;font-size:0.95rem;">{summary_short}</p>
            {f'<p style="margin:8px 0 0 0;font-size:0.87rem;"><a href="{website}" target="_blank">🌐 {website}</a></p>' if website else ''}
        </div>
        <div style="text-align:right;min-width:180px;">
            <div style="font-size:2.4rem;font-weight:bold;color:{color};">{signal_emoji}</div>
            <div style="font-size:1rem;color:{color};font-weight:bold;">{signal_text}</div>
            <div style="margin-top:12px;font-size:0.95rem;color:#333;">Technical Score: <strong>{technical_score:+.2f}</strong></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== METRICS ROW ====================
change = ((last_price - close.iloc[-2]) / close.iloc[-2]) * 100
volatility = close.pct_change().std() * np.sqrt(252) * 100
period_return = ((last_price / close.iloc[0]) - 1) * 100

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
with m1:
    st.metric("Price", f"${last_price:.2f}")
with m2:
    st.metric("1D Change", f"{change:+.2f}%")
with m3:
    st.metric(f"{prediction_label} Target", f"${pred_price:.2f}", f"{pred_change:+.1f}%")
with m4:
    st.metric("Neural Score", f"{unified['smart_score']:+.2f}")
with m5:
    st.metric("Trend Meter", f"{unified['trend_score']:+.2f}")
with m6:
    st.metric("Alpha Signal", f"{unified['alpha_score']:+.2f}")
with m7:
    st.metric("Tech Rating", f"{unified['tech_score']:+.2f}")

# ==================== PRICE CHART ====================

sma20 = close.rolling(20).mean()
sma50 = close.rolling(50).mean()

macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
macd_signal = macd_line.ewm(span=9, adjust=False).mean()

delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    row_heights=[0.55, 0.23, 0.22], subplot_titles=(f"{company_name} ({stock_symbol}) — Price & AI Forecast", "MACD", "RSI"))

fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Price", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=sma20.index, y=sma20.values, name="SMA 20", line=dict(color="orange", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=sma50.index, y=sma50.values, name="SMA 50", line=dict(color="purple", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=future_dates, y=mean_f, name="Forecast", line=dict(color="#2ca02c", width=2, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=list(future_dates) + list(future_dates)[::-1], y=list(p90) + list(p10)[::-1], fill="toself", fillcolor="rgba(44,160,44,0.15)", line=dict(color="rgba(0,0,0,0)"), name="Forecast range", hoverinfo="skip"), row=1, col=1)

for idx, (dt, level) in enumerate(results.get("support_levels", [])):
    fig.add_hline(y=level, line=dict(color="green", width=1, dash="dot"), annotation_text=f"Support {idx+1}: ${level:.2f}", annotation_position="bottom left", row=1, col=1)
for idx, (dt, level) in enumerate(results.get("resistance_levels", [])):
    fig.add_hline(y=level, line=dict(color="red", width=1, dash="dot"), annotation_text=f"Resistance {idx+1}: ${level:.2f}", annotation_position="top left", row=1, col=1)

fig.add_trace(go.Scatter(x=close.index, y=macd_line.values, name="MACD", line=dict(color="#2c3e50", width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=close.index, y=macd_signal.values, name="Signal", line=dict(color="#e74c3c", width=1)), row=2, col=1)
fig.add_trace(go.Bar(x=close.index, y=(macd_line - macd_signal).values, name="Histogram", marker_color=["green" if v >= 0 else "red" for v in (macd_line - macd_signal).values]), row=2, col=1)
fig.add_hline(y=0, line=dict(color="black", width=0.5), row=2, col=1)

fig.add_trace(go.Scatter(x=close.index, y=rsi.values, name="RSI", line=dict(color="#9b59b6", width=1.5)), row=3, col=1)
fig.add_hline(y=70, line=dict(color="red", width=1, dash="dash"), row=3, col=1)
fig.add_hline(y=30, line=dict(color="green", width=1, dash="dash"), row=3, col=1)

fig.update_layout(height=840, template="plotly_white", hovermode="x unified", showlegend=True,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=40, t=90, b=40))
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="MACD", row=2, col=1)
fig.update_yaxes(title_text="RSI", row=3, col=1)
st.plotly_chart(fig, use_container_width=True)

# ==================== TECHNICAL BREAKDOWN ====================
st.markdown("---")
st.subheader("🔍 Technical Score Breakdown")

breakdown_cols = st.columns(4)
with breakdown_cols[0]:
    st.markdown("**📊 Smart Score**")
    st.write(f"{unified['smart_score']:+.2f}")
with breakdown_cols[1]:
    st.markdown("**📈 Trend Meter**")
    st.write(f"{unified['trend_score']:+.2f}")
with breakdown_cols[2]:
    st.markdown("**⚡ Alpha Signal**")
    st.write(f"{unified['alpha_score']:+.2f}")
with breakdown_cols[3]:
    st.markdown("**🔮 Tech Rating**")
    st.write(f"{unified['tech_score']:+.2f}")

with st.expander("View technical scoring details"):
    st.write("**Smart Score details**")
    for k, v in results.get("smart_score", {}).items():
        st.write(f"- {k}: {v}")
    st.write("**Trend Meter details**")
    for k, v in results.get("trend_meter", {}).items():
        st.write(f"- {k}: {v}")
    st.write("**Alpha Signal details**")
    for k, v in results.get("alpha_signal", {}).items():
        st.write(f"- {k}: {v}")
    st.write("**Technical Rating details**")
    for k, v in results.get("technical_rating", {}).items():
        st.write(f"- {k}: {v}")
    if results.get("patterns"):
        st.write("**Patterns detected**")
        for pattern, direction in results["patterns"].items():
            st.write(f"- {pattern}: {direction}")

# ==================== AI AGENT ANALYSIS ====================
st.markdown("---")
st.subheader("🤖 AI Agent Analysis")

technical_context = (
    f"Current price: ${last_price:.2f}\n"
    f"52W high/low: ${high.max():.2f}/${low.min():.2f}\n"
    f"Forecast target: ${pred_price:.2f} ({pred_change:+.1f}%)\n"
    f"Trend score: {unified['trend_score']:+.2f}\n"
    f"RSI: {results.get('smart_score', {}).get('RSI', 'N/A')}\n"
    f"MACD: {results.get('smart_score', {}).get('MACD', 'N/A')}\n"
    f"Patterns: {', '.join(results.get('patterns', {}).keys()) or 'None'}\n"
)

agents = [
    {
        "name": "Fundamentals Analyst",
        "icon": "📊",
        "system": "You are a fundamentals analyst. Analyze valuation, earnings, analyst sentiment, and company context. Provide a concise final signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": (
            f"Analyze {company_name} ({stock_symbol}) fundamentals."
            f"\nSector: {sector}\nIndustry: {industry}\nMarket Cap: {mcap_str}\nP/E: {pe_ratio}\n"
            f"Current Price: ${last_price:.2f}\n{technical_context}\n"
            "Give a final trading signal with reasoning."
        ),
    },
    {
        "name": "Technical Analyst",
        "icon": "📈",
        "system": "You are a technical analyst. Review price action, indicators, support/resistance, and trend strength. Provide a concise final signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": (
            f"Analyze {company_name} ({stock_symbol}) technicals using the current trend and forecast.\n{technical_context}"
            "Include support/resistance, momentum, and any pattern evidence, then give a final signal."
        ),
    },
    {
        "name": "Sentiment Analyst",
        "icon": "😊",
        "system": "You are a sentiment analyst. Evaluate investor mood, market psychology, and momentum bias. Provide a concise final signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": (
            f"Assess sentiment for {company_name} ({stock_symbol}).\nCurrent price: ${last_price:.2f}\n{technical_context}"
            "Consider overall market tone and risk appetite, then give a final signal."
        ),
    },
    {
        "name": "News Analyst",
        "icon": "📰",
        "system": "You are a news analyst. Consider macro headwinds, sector momentum, and relevant flows. Provide a concise final signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": (
            f"Evaluate news and macro factors for {company_name} ({stock_symbol}).\n{technical_context}"
            "If exact headlines are not available, use sector outlook and recent market direction. Provide a final signal."
        ),
    },
    {
        "name": "Risk Manager",
        "icon": "⚠️",
        "system": "You are a risk manager. Assess downside risk, volatility, and appropriate position sizing. Provide a concise final signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": (
            f"Assess risk for {company_name} ({stock_symbol}) at current price ${last_price:.2f}.\n{technical_context}"
            "Recommend a final risk-aware signal and if possible a sensible position size."
        ),
    },
]

agent_results = {}
agent_cols = st.columns(len(agents))
for idx, agent in enumerate(agents):
    with agent_cols[idx]:
        st.markdown(f"<h4 style='margin-bottom:0.4rem;'>{agent['icon']} {agent['name']}</h4>", unsafe_allow_html=True)
        content, error = call_llm(agent["system"], agent["prompt"], provider_choice, api_key, model_choice, ollama_url)
        if error:
            st.error(error)
            agent_results[agent["name"]] = {"signal": "ERROR", "score": 0, "emoji": "⚪", "reasoning": error}
        else:
            parsed = parse_agent_response(content)
            agent_results[agent["name"]] = parsed
            st.markdown(
                f"<div style='background:#f6f7fb;padding:14px;border-radius:12px;border:1px solid #dcdfe6;'>"
                f"<p style='margin:0 0 8px 0;font-weight:700;color:#111;'>{parsed['emoji']} {parsed['signal']}</p>"
                f"<p style='margin:0;color:#444;white-space:pre-wrap;font-size:0.95rem;'>{parsed['reasoning']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ==================== FINAL HYBRID DECISION ====================
st.markdown("---")
st.subheader("🧠 Hybrid Signal & Strong Buy/Sell Indicator")

valid_scores = [result["score"] for result in agent_results.values() if result["signal"] != "ERROR"]
if valid_scores:
    ai_avg_score = float(np.mean(valid_scores))
    combined_label, combined_score = aggregate_indicator_signal(technical_score, ai_avg_score)
    final_pm_prompt = (
        f"You are a portfolio manager. Review the following analyst signals for {company_name} ({stock_symbol}):\n"
        + "\n".join([f"{name}: {result['signal']}" for name, result in agent_results.items() if result['signal'] != 'ERROR'])
        + f"\nTechnical unified score: {technical_score:+.2f}. Forecast target: ${pred_price:.2f} ({pred_change:+.1f}%).\n"
        + f"AI consensus score: {ai_avg_score:+.2f}.\n"
        + "Provide a final, concise recommendation with a strong buy/sell/hold signal and suggested allocation percentage."
    )
    pm_content, pm_error = call_llm("You are a portfolio manager.", final_pm_prompt, provider_choice, api_key, model_choice, ollama_url)
    if pm_error:
        st.error(pm_error)
        pm_signal = "ERROR"
        pm_reasoning = pm_error
    else:
        pm_parsed = parse_agent_response(pm_content)
        pm_signal = pm_parsed["signal"]
        pm_reasoning = pm_parsed["reasoning"]

    final_color = "#2ca02c" if "BUY" in combined_label else "#ff4500" if "SELL" in combined_label else "#ffa500"
    st.markdown(
        f"<div style='background:linear-gradient(135deg, {final_color}22 0%, {final_color}11 100%);border-radius:16px;padding:24px;margin:12px 0;border:2px solid {final_color};'>"
        f"<h2 style='margin:0 0 10px 0;color:{final_color};'>{signal_emoji} {combined_label}</h2>"
        f"<p style='margin:0 0 8px 0;color:#333;line-height:1.6;'>Combined hybrid score: {combined_score:+.2f}. AI consensus score: {ai_avg_score:+.2f}.</p>"
        f"<p style='margin:0;color:#444;line-height:1.6;'>Portfolio manager view: {pm_reasoning}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**AI Ensemble Consensus:**")
    consensus_fig = go.Figure()
    names = [name for name, result in agent_results.items() if result['signal'] != "ERROR"]
    scores = [result['score'] for name, result in agent_results.items() if result['signal'] != "ERROR"]
    colors = ["#2ca02c" if s > 0.2 else "#ffa500" if s >= -0.2 else "#dc143c" for s in scores]
    consensus_fig.add_trace(go.Bar(x=names, y=scores, marker_color=colors, text=[f"{s:+.1f}" for s in scores], textposition="outside"))
    consensus_fig.add_hline(y=ai_avg_score, line=dict(color="#3498db", dash="dash"), annotation_text=f"AI Avg {ai_avg_score:+.2f}", annotation_position="top left")
    consensus_fig.update_layout(title="Agent Signal Scores", yaxis_title="Score", template="plotly_white", showlegend=False, height=340)
    st.plotly_chart(consensus_fig, use_container_width=True)
else:
    st.warning("AI agent analysis did not produce valid signals. Check your API key and provider settings.")

st.markdown("---")
st.markdown("<center><small>Forecasts and signals are for educational purposes only. Not financial advice.</small></center>", unsafe_allow_html=True)
