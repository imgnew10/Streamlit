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

st.set_page_config(page_title="AI Trading Predictor", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center;color:#1f77b4;font-size:2.2rem;'>🤖 AI Trading Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;font-size:1rem;'>Combined technical forecasting + unified trading signals + multi-agent AI decision support.</p>", unsafe_allow_html=True)

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
st.sidebar.markdown("### 🤖 AI Provider")

provider_choice = st.sidebar.selectbox(
    "Provider",
    ["OpenRouter", "OpenAI", "Anthropic", "Ollama (Local)"],
    index=0
)

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
        "meta-llama/llama-3.3-70b-instruct"
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
st.sidebar.markdown("<small>Data from Yahoo Finance + multi-agent AI pricing predictions.</small>", unsafe_allow_html=True)

# ==================== LLM HELPERS ====================

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
            return None, f"Ollama Connection Error: {e}"

    if provider == "OpenRouter":
        if not api_key or not api_key.startswith("sk-or"):
            return None, "Enter a valid OpenRouter key starting with sk-or-v1-."
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
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
            return None, "Enter a valid OpenAI API key."
        endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
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
            return None, "Enter a valid Anthropic API key."
        endpoint = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.7,
            "max_tokens": 800
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
                    parts = [block.get("text", "") for block in data["content"] if isinstance(block, dict)]
                    return "".join(parts), None
                return None, f"Unexpected Anthropic response: {data}"
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"], None
            return None, f"Unexpected response: {data}"
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 2)
                continue
            return None, f"API Error: {e}"
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

if provider_choice != "Ollama (Local)" and not api_key:
    st.info("Enter your API key in the sidebar to use AI predictions.")
    st.stop()

if not stock_symbol:
    st.info("Enter a stock symbol in the sidebar to begin.")
    st.stop()

# ==================== FETCH DATA ====================
with st.spinner(f"Fetching market data for {stock_symbol}..."):
    data, info, ticker = fetch_stock_data(stock_symbol, period="2y")
    spy_close = fetch_market_data(period="2y")

if data is None or data.empty:
    st.error(f"No data found for '{stock_symbol}'. Try a ticker such as AAPL, TSLA, MSFT, NVDA.")
    st.stop()

close = data['Close']
high = data['High']
low = data['Low']
last_price = close.iloc[-1]

results = calculate_all_signals(data, info, ticker, spy_close)
unified = results['unified']
score = unified['score']
signal_text = unified['signal']
signal_emoji = unified['emoji']
color = unified['color']

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

# ==================== STOCK SUMMARY CARD ====================
st.markdown("---")
st.markdown(f"""
<div style="background:linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);border-radius:16px;padding:22px;margin:10px 0;border-left:6px solid {color};box-shadow:0 2px 14px rgba(0,0,0,0.08);">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;">
        <div style="max-width:72%;">
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
            <div style="margin-top:12px;font-size:0.95rem;color:#333;">Unified Score: <strong>{score:+.2f}</strong></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== METRICS ====================
change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
forecast_args = {
    "support_levels": results.get('support_levels', []),
    "resistance_levels": results.get('resistance_levels', [])
}
future_dates, mean_f, p10, p90, forecast_support_levels, forecast_resistance_levels = generate_forecast(
    data,
    score,
    forecast_days,
    **forecast_args
)
pred_price = mean_f[-1]
pred_change = ((pred_price - last_price) / last_price) * 100

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1:
    st.metric("Price", f"${last_price:.2f}")
with c2:
    st.metric("1D Change", f"{change:+.2f}%")
with c3:
    st.metric(f"{prediction_label} Target", f"${pred_price:.2f}", f"{pred_change:+.1f}%")
with c4:
    st.metric("Smart Score", f"{unified['smart_score']:+.2f}")
with c5:
    st.metric("Trend Meter", f"{unified['trend_score']:+.2f}")
with c6:
    st.metric("Alpha Signal", f"{unified['alpha_score']:+.2f}")
with c7:
    st.metric("Tech Rating", f"{unified['tech_score']:+.2f}")

# ==================== CHART ====================

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
                    row_heights=[0.55, 0.23, 0.22], subplot_titles=(f"{company_name} ({stock_symbol}) — Price & Forecast", "MACD", "RSI"))

fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Close", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=sma20.index, y=sma20.values, name="SMA 20", line=dict(color="orange", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=sma50.index, y=sma50.values, name="SMA 50", line=dict(color="purple", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=future_dates, y=mean_f, name="Forecast", line=dict(color="#2ca02c", width=2, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=list(future_dates) + list(future_dates)[::-1], y=list(p90) + list(p10)[::-1], fill="toself", fillcolor="rgba(44,160,44,0.15)", line=dict(color="rgba(0,0,0,0)"), name="Confidence band", hoverinfo="skip"), row=1, col=1)

for idx, (dt, level) in enumerate(results.get('support_levels', [])):
    fig.add_hline(y=level, line=dict(color="green", width=1, dash="dot"), annotation_text=f"Support {idx+1}: ${level:.2f}", annotation_position="bottom left", row=1, col=1)
for idx, (dt, level) in enumerate(results.get('resistance_levels', [])):
    fig.add_hline(y=level, line=dict(color="red", width=1, dash="dot"), annotation_text=f"Resistance {idx+1}: ${level:.2f}", annotation_position="top left", row=1, col=1)

fig.add_trace(go.Scatter(x=close.index, y=macd_line.values, name="MACD", line=dict(color="#2c3e50", width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=close.index, y=macd_signal.values, name="Signal", line=dict(color="#e74c3c", width=1)), row=2, col=1)
fig.add_trace(go.Bar(x=close.index, y=(macd_line - macd_signal).values, name="Histogram", marker_color=["green" if v >= 0 else "red" for v in (macd_line - macd_signal).values]), row=2, col=1)
fig.add_hline(y=0, line=dict(color="black", width=0.5), row=2, col=1)

fig.add_trace(go.Scatter(x=close.index, y=rsi.values, name="RSI", line=dict(color="#9b59b6", width=1.5)), row=3, col=1)
fig.add_hline(y=70, line=dict(color="red", width=1, dash="dash"), row=3, col=1)
fig.add_hline(y=30, line=dict(color="green", width=1, dash="dash"), row=3, col=1)

fig.update_layout(height=820, template="plotly_white", hovermode="x unified", showlegend=True,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=40, t=90, b=40))
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="MACD", row=2, col=1)
fig.update_yaxes(title_text="RSI", row=3, col=1)
st.plotly_chart(fig, use_container_width=True)

# ==================== SIGNAL BREAKDOWN ====================
st.markdown("---")
st.subheader("🔍 Unified Technical Score")
cols = st.columns(4)
for label, metric in [
    ("Smart Score", unified['smart_score']),
    ("Trend Meter", unified['trend_score']),
    ("Alpha Signal", unified['alpha_score']),
    ("Tech Rating", unified['tech_score'])
]:
    with cols.pop(0):
        st.metric(label, f"{metric:+.2f}")

with st.expander("See score components"):
    st.write("**Smart Score details**")
    for k, v in results.get('smart_score', {}).items():
        st.write(f"- {k}: {v}")
    st.write("**Trend Meter details**")
    for k, v in results.get('trend_meter', {}).items():
        st.write(f"- {k}: {v}")
    st.write("**Alpha Signal details**")
    for k, v in results.get('alpha_signal', {}).items():
        st.write(f"- {k}: {v}")
    st.write("**Technical Rating details**")
    for k, v in results.get('technical_rating', {}).items():
        st.write(f"- {k}: {v}")

# ==================== AGENT PREDICTIONS ====================
st.markdown("---")
st.subheader("🤖 AI Multi-Agent Trading Predictions")

tech_summary = [
    f"Current Price: ${last_price:.2f}",
    f"52W High: ${high.max():.2f} | 52W Low: ${low.min():.2f}",
    f"Last Close: ${last_price:.2f}",
    f"Trend: {results.get('trend_meter', {}).get('ADX', 'N/A')}"
]
tech_summary.extend([f"{k}: {v}" for k, v in results.get('smart_score', {}).items() if k in ["RSI", "MACD", "SMA Cross", "Bollinger", "Stochastic", "VWAP"]])
tech_summary.append(f"Patterns: {', '.join(results.get('patterns', {}).keys()) or 'None'}")
tech_summary_text = "\n".join(tech_summary)

agents = [
    {
        "name": "Fundamentals Analyst",
        "icon": "📊",
        "system": "You are a fundamentals analyst. Analyze company financials, valuation, and earnings. Give a concise signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze {company_name} ({stock_symbol}) using the following company attributes: sector {sector}, market cap {mcap_str}, P/E {pe_ratio}, industry {industry}. Provide reasoning and a final signal."
    },
    {
        "name": "Technical Analyst",
        "icon": "📈",
        "system": "You are a technical analyst. Review price action, indicators, and trend strength. Give a concise signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze {company_name} ({stock_symbol}) technicals from this summary:\n{tech_summary_text}\nProvide reasoning and a final signal."
    },
    {
        "name": "Sentiment Analyst",
        "icon": "😊",
        "system": "You are a sentiment analyst. Interpret investor mood and market bias. Give a concise signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze market sentiment around {company_name} ({stock_symbol}) based on current market conditions and price behavior. Provide reasoning and a final signal."
    },
    {
        "name": "News Analyst",
        "icon": "📰",
        "system": "You are a news analyst. Consider macro factors and sector risks. Give a concise signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Evaluate news flow for {company_name} ({stock_symbol}) in the {sector} sector and describe how it impacts the trade. Provide reasoning and a final signal."
    },
    {
        "name": "Risk Manager",
        "icon": "⚠️",
        "system": "You are a risk manager. Assess downside risk and position sizing. Give a concise signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Assess risk for {company_name} ({stock_symbol}) at current price ${last_price:.2f}. Recommend a signal and an approximate position size."
    }
]

agent_results = {}
agents_cols = st.columns(len(agents))
for idx, agent in enumerate(agents):
    with agents_cols[idx]:
        st.markdown(f"<h4 style='color:#333;margin-bottom:0.25rem;'>{agent['icon']} {agent['name']}</h4>", unsafe_allow_html=True)
        content, error = call_llm(agent['system'], agent['prompt'], provider_choice, api_key, model_choice, ollama_url)
        if error:
            st.error(error)
            agent_results[agent['name']] = {"signal": "ERROR", "score": 0, "emoji": "⚪", "reasoning": error}
        else:
            parsed = parse_agent_response(content)
            agent_results[agent['name']] = parsed
            st.markdown(f"<div style='background:#f4f5f7;padding:12px;border-radius:10px;border:1px solid #dcdde1;'>"
                        f"<p style='margin:0;font-weight:700;color:#111;'>{parsed['emoji']} {parsed['signal']}</p>"
                        f"<p style='margin:8px 0 0 0;color:#444;white-space:pre-wrap;font-size:0.94rem;'>{parsed['reasoning']}</p>"
                        f"</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("🧠 Portfolio Manager Final Recommendation")
valid_scores = [res['score'] for res in agent_results.values() if res['signal'] != "ERROR"]
if valid_scores:
    avg_score = np.mean(valid_scores)
    consensus = "\n".join([f"{name}: {res['emoji']} {res['signal']}" for name, res in agent_results.items() if res['signal'] != "ERROR"])
    pm_system = "You are a portfolio manager. Review the agent signals and make a final trading recommendation with position sizing. Be concise."
    pm_prompt = f"Review this consensus for {company_name} ({stock_symbol}):\n{consensus}\nAverage score: {avg_score:+.2f}. Give a final signal and recommended position size."
    pm_content, pm_error = call_llm(pm_system, pm_prompt, provider_choice, api_key, model_choice, ollama_url)
    if pm_error:
        st.error(pm_error)
    else:
        pm_parsed = parse_agent_response(pm_content)
        final_color = "#2ca02c" if "BUY" in pm_parsed['signal'] else "#ff4500" if "SELL" in pm_parsed['signal'] else "#ffa500"
        st.markdown(f"<div style='background:linear-gradient(135deg, {final_color}22 0%, {final_color}11 100%);border-radius:16px;padding:24px;margin:12px 0;border:2px solid {final_color};'>"
                    f"<h2 style='margin:0 0 10px 0;color:{final_color};'>{pm_parsed['emoji']} {pm_parsed['signal']}</h2>"
                    f"<p style='margin:0;color:#333;line-height:1.6;'>{pm_parsed['reasoning']}</p>"
                    f"</div>", unsafe_allow_html=True)
        fig = go.Figure()
        names = [name for name, res in agent_results.items() if res['signal'] != "ERROR"]
        scores = [res['score'] for name, res in agent_results.items() if res['signal'] != "ERROR"]
        colors = ["#2ca02c" if s > 0.2 else "#ffa500" if s >= -0.2 else "#dc143c" for s in scores]
        fig.add_trace(go.Bar(x=names, y=scores, marker_color=colors, text=[f"{s:+.1f}" for s in scores], textposition="outside"))
        fig.add_hline(y=avg_score, line=dict(color="#3498db", dash="dash"), annotation_text=f"Consensus {avg_score:+.2f}", annotation_position="top left")
        fig.update_layout(title="Agent Consensus Scores", yaxis_title="Score", template="plotly_white", showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No valid AI agent predictions were generated.")

st.markdown("---")
st.markdown("<center><small>Built as a combined technical & AI-driven prediction page. Not financial advice.</small></center>", unsafe_allow_html=True)
