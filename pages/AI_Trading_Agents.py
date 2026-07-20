import streamlit as st
import requests
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import warnings
warnings.filterwarnings('ignore')

from recommendation_engine import fetch_stock_data, calculate_all_signals

st.set_page_config(page_title="AI Trading Agents", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center;color:#6C5CE7;font-size:2.2rem;'>🤖 AI Trading Agents</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;font-size:1rem;'>Multi-Agent LLM Debate: Fundamentals + Sentiment + News + Technical + Risk + Portfolio Manager</p>", unsafe_allow_html=True)

# ==================== SIDEBAR: API KEY + SETTINGS ====================
st.sidebar.markdown("## 🔐 OpenRouter API")
st.sidebar.markdown("---")

api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password",
    placeholder="sk-or-v1-...",
    help="Get your key at https://openrouter.ai/keys"
)

model_choice = st.sidebar.selectbox(
    "LLM Model",
    [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Analysis Settings")

stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", placeholder="e.g. AAPL, TSLA, NVDA").upper().strip()
include_social = st.sidebar.checkbox("Include Social Sentiment Analysis", value=True)
include_news = st.sidebar.checkbox("Include News Analysis", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("<small>Powered by OpenRouter API</small>", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
def call_openrouter(system_prompt, user_prompt, api_key, model):
    """Call OpenRouter API with retry logic."""
    if not api_key or not api_key.startswith("sk-or"):
        return None, "Invalid API key. Please enter a valid OpenRouter key starting with 'sk-or-v1-'."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://stock-predictor.streamlit.app",
        "X-Title": "Stock Predictor"
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

    for attempt in range(3):
        try:
            time.sleep(1.5 + random.uniform(0.5, 1.5))  # Rate limiting
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 429:
                time.sleep((attempt + 1) * 3)
                continue

            response.raise_for_status()
            data = response.json()

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
    """Extract signal and reasoning from agent response."""
    content_lower = content.lower()

    # Determine signal
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

# ==================== MAIN ====================
if not api_key:
    st.info("👈 Enter your **OpenRouter API Key** in the sidebar to start the AI Trading Agents analysis.")
    st.markdown("""
    ### How to get an API Key:
    1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
    2. Sign up / Log in
    3. Click **"Create Key"**
    4. Copy the key (starts with `sk-or-v1-`)
    5. Paste it in the sidebar

    ### What the Agents Do:
    | Agent | Role |
    |-------|------|
    | 📊 **Fundamentals Analyst** | Company financials & valuation |
    | 😊 **Sentiment Analyst** | Social media & market mood |
    | 📰 **News Analyst** | Global news & macro trends |
    | 📈 **Technical Analyst** | Indicators & price patterns |
    | ⚠️ **Risk Manager** | Risk assessment & position sizing |
    | 💼 **Portfolio Manager** | Final decision & execution |
    """)
    st.stop()

if not stock_symbol:
    st.info("👈 Enter a stock symbol in the sidebar.")
    st.stop()

# Fetch stock data
with st.spinner(f"Fetching data for {stock_symbol}..."):
    data, info, ticker = fetch_stock_data(stock_symbol, period="6mo")
    if data is None or data.empty:
        st.error(f"❌ No data found for '{stock_symbol}'.")
        st.stop()

# Calculate technical signals for context
results = calculate_all_signals(data, info, ticker, spy_close=None)
unified = results['unified']

# Prepare stock context
close = data['Close']
last_price = close.iloc[-1]
company_name = info.get('shortName', stock_symbol) if info else stock_symbol
sector = info.get('sector', 'N/A') if info else 'N/A'
market_cap = info.get('marketCap', 'N/A') if info else 'N/A'
pe_ratio = info.get('trailingPE', 'N/A') if info else 'N/A'
eps = info.get('trailingEps', 'N/A') if info else 'N/A'

# Technical summary
tech_summary = f"""
Current Price: ${last_price:.2f}
52W High: ${data['High'].max():.2f} | 52W Low: ${data['Low'].min():.2f}
RSI: {results['smart_score'].get('RSI', 'N/A')}
MACD: {results['smart_score'].get('MACD', 'N/A')}
SMA: {results['smart_score'].get('SMA Cross', 'N/A')}
Bollinger: {results['smart_score'].get('Bollinger', 'N/A')}
Stochastic: {results['smart_score'].get('Stochastic', 'N/A')}
VWAP: {results['smart_score'].get('VWAP', 'N/A')}
Volume: {results['smart_score'].get('Volume', 'N/A')}
Trend: {results['trend_meter'].get('ADX', 'N/A')}
Patterns: {', '.join([f'{k} ({v})' for k, v in results['patterns'].items()]) if results['patterns'] else 'None detected'}
"""

# ==================== RUN AGENTS ====================
st.markdown("---")
st.subheader(f"🎯 Analyzing {company_name} ({stock_symbol})")

agents_config = [
    {
        "name": "📊 Fundamentals Analyst",
        "icon": "📊",
        "color": "#3498db",
        "system": "You are a fundamentals analyst. Analyze company financials, valuation metrics, earnings, and growth prospects. Be concise (3-4 sentences). End with a clear signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze {company_name} ({stock_symbol}) fundamentals.

Company Info:
- Sector: {sector}
- Market Cap: {market_cap}
- P/E Ratio: {pe_ratio}
- EPS: {eps}

Provide your assessment and signal."
    },
    {
        "name": "📈 Technical Analyst", 
        "icon": "📈",
        "color": "#9b59b6",
        "system": "You are a technical analyst. Analyze price action, indicators, and patterns. Be concise (3-4 sentences). End with a clear signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze {company_name} ({stock_symbol}) technicals.

{tech_summary}

Provide your assessment and signal."
    },
    {
        "name": "😊 Sentiment Analyst",
        "icon": "😊",
        "color": "#e67e22",
        "system": "You are a sentiment analyst. Analyze market mood, social media trends, and investor psychology. Be concise (3-4 sentences). End with a clear signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze market sentiment for {company_name} ({stock_symbol}). Current price ${last_price:.2f}, sector {sector}. Consider recent price action and general market mood. Provide your assessment and signal."
    },
    {
        "name": "📰 News Analyst",
        "icon": "📰",
        "color": "#1abc9c",
        "system": "You are a news analyst. Consider recent global events, sector trends, and macroeconomic factors. Be concise (3-4 sentences). End with a clear signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze news and macro factors affecting {company_name} ({stock_symbol}) in the {sector} sector. Current price ${last_price:.2f}. Provide your assessment and signal."
    },
    {
        "name": "⚠️ Risk Manager",
        "icon": "⚠️",
        "color": "#e74c3c",
        "system": "You are a risk manager. Assess downside risk, volatility, and position sizing. Be concise (3-4 sentences). End with a clear signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Assess risk for {company_name} ({stock_symbol}) at ${last_price:.2f}. Consider volatility, downside protection, and portfolio impact. Provide risk assessment and signal."
    }
]

# Run all agents
agent_results = {}
agent_cols = st.columns(len(agents_config))

for idx, agent in enumerate(agents_config):
    with agent_cols[idx]:
        st.markdown(f"<h4 style='color:{agent['color']};margin:0;'>{agent['icon']} {agent['name']}</h4>", unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            content, error = call_openrouter(
                agent['system'],
                agent['prompt'],
                api_key,
                model_choice
            )

        if error:
            st.error(f"❌ {error}")
            agent_results[agent['name']] = {"signal": "ERROR", "score": 0, "emoji": "⚪", "reasoning": error}
        else:
            parsed = parse_agent_response(content)
            agent_results[agent['name']] = parsed

            st.markdown(f"""
            <div style="background-color:{agent['color']}15;border-radius:8px;padding:10px;margin:5px 0;border-left:3px solid {agent['color']};">
                <p style="margin:0;font-size:1.2rem;font-weight:bold;color:{agent['color']};">{parsed['emoji']} {parsed['signal']}</p>
                <p style="margin:5px 0 0 0;font-size:0.85rem;color:#444;line-height:1.4;">{parsed['reasoning'][:200]}...</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== PORTFOLIO MANAGER - FINAL DECISION ====================
st.markdown("---")
st.subheader("💼 Portfolio Manager - Final Decision")

# Aggregate agent scores
valid_scores = [r['score'] for r in agent_results.values() if r['signal'] != "ERROR"]
if valid_scores:
    avg_score = np.mean(valid_scores)

    # Build consensus prompt
    consensus_summary = "
".join([
        f"{name}: {r['emoji']} {r['signal']} (Score: {r['score']:+.1f})"
        for name, r in agent_results.items() if r['signal'] != "ERROR"
    ])

    pm_system = "You are a portfolio manager. Review all analyst inputs and make a final trading decision with position sizing. Be decisive and concise (4-5 sentences)."
    pm_prompt = f"""Review the following analyst consensus for {company_name} ({stock_symbol}):

{consensus_summary}

Average Score: {avg_score:+.2f}/1.00

Make a final decision: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.
Include recommended position size (e.g., 5% portfolio allocation) and stop-loss level."""

    with st.spinner("Portfolio Manager deliberating..."):
        pm_content, pm_error = call_openrouter(pm_system, pm_prompt, api_key, model_choice)

    if pm_error:
        st.error(f"❌ {pm_error}")
    else:
        pm_parsed = parse_agent_response(pm_content)

        # Color based on final signal
        final_color = "#2ca02c" if "BUY" in pm_parsed['signal'] else "#DC143C" if "SELL" in pm_parsed['signal'] else "#FFA500"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, {final_color}15 0%, {final_color}08 100%);border-radius:16px;padding:25px;margin:15px 0;border:2px solid {final_color};text-align:center;">
            <h2 style="margin:0 0 10px 0;color:{final_color};font-size:2rem;">{pm_parsed['emoji']} {pm_parsed['signal']}</h2>
            <p style="margin:0;color:#333;font-size:1rem;line-height:1.6;max-width:800px;margin:0 auto;">{pm_parsed['reasoning']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Agent consensus chart
        import plotly.graph_objects as go

        agent_names = [name.split()[-1] for name in agent_results.keys() if agent_results[name]['signal'] != "ERROR"]
        agent_scores = [agent_results[name]['score'] for name in agent_results.keys() if agent_results[name]['signal'] != "ERROR"]
        agent_colors = ["#2ca02c" if s > 0.2 else "#FFA500" if s > -0.2 else "#DC143C" for s in agent_scores]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=agent_names,
            y=agent_scores,
            marker_color=agent_colors,
            text=[f"{s:+.1f}" for s in agent_scores],
            textposition="outside"
        ))
        fig.add_hline(y=avg_score, line_dash="dash", line_color="blue", annotation_text=f"Consensus: {avg_score:+.2f}")
        fig.add_hline(y=0.2, line_dash="dot", line_color="green", annotation_text="Buy")
        fig.add_hline(y=-0.2, line_dash="dot", line_color="red", annotation_text="Sell")
        fig.update_layout(
            title="Agent Consensus Scores",
            yaxis_title="Score (-1 to +1)",
            height=350,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<center><small>Powered by OpenRouter API | Multi-Agent LLM Debate | Not financial advice</small></center>", unsafe_allow_html=True)
