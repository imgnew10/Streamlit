import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import warnings
warnings.filterwarnings('ignore')

from recommendation_engine import fetch_stock_data, calculate_all_signals

st.set_page_config(page_title="AI Trading Agents", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center;color:#6C5CE7;font-size:2.2rem;'>🤖 AI Trading Agents</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;font-size:1rem;'>Multi-Agent LLM Debate using OpenRouter API</p>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("## 🔐 AI Provider")
st.sidebar.markdown("---")

provider_choice = st.sidebar.selectbox(
    "Provider",
    ["OpenRouter", "OpenAI", "Anthropic", "Ollama (Local)"],
    index=0
)

if provider_choice == "OpenRouter":
    api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-v1-...",
        help="Get your key at openrouter.ai/keys"
    )
    model_options = [
        "openrouter/auto",
        "openrouter/free",
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat"
    ]
elif provider_choice == "OpenAI":
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Get your key at platform.openai.com/api-keys"
    )
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
elif provider_choice == "Anthropic":
    api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com/settings/keys"
    )
    model_options = ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest", "claude-3-haiku-20240307"]
else:
    api_key = ""
    model_options = ["llama3.1:8b", "mistral", "qwen2.5:7b", "phi3:mini"]

if provider_choice == "OpenRouter":
    default_model_index = 1
else:
    default_model_index = 0

model_choice = st.sidebar.selectbox("LLM Model", model_options, index=default_model_index)

if provider_choice == "Ollama (Local)":
    ollama_url = st.sidebar.text_input(
        "Ollama URL",
        value="http://localhost:11434",
        help="Make sure Ollama is running locally and the selected model is installed."
    )
else:
    ollama_url = "http://localhost:11434"

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Settings")

stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", placeholder="e.g. AAPL, TSLA, NVDA").upper().strip()

st.sidebar.markdown("---")
st.sidebar.markdown("<small>Switch between OpenRouter, OpenAI, Anthropic, or local Ollama</small>", unsafe_allow_html=True)

# HELPER FUNCTIONS
def call_llm(system_prompt, user_prompt, provider, api_key, model, ollama_url):
    if provider == "Ollama (Local)":
        endpoint = f"{ollama_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }

        for attempt in range(3):
            try:
                time.sleep(1.0)
                response = requests.post(endpoint, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()

                if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                    return data["message"]["content"], None
                return None, f"Unexpected Ollama response: {data}"
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    time.sleep((attempt + 1) * 2)
                    continue
                return None, f"Ollama Error: {str(e)}"

        return None, "Max retries exceeded. Please try again later."

    if provider == "OpenRouter":
        if not api_key or not api_key.startswith("sk-or"):
            return None, "Invalid API key. Please enter a valid OpenRouter key starting with 'sk-or-v1-'."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://stock-predictor.streamlit.app",
            "X-Title": "Stock Predictor"
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
            time.sleep(1.5 + random.uniform(0.5, 1.5))
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

# MAIN
if provider_choice != "Ollama (Local)" and not api_key:
    st.info("Enter your API key in the sidebar to start.")
    if provider_choice == "OpenRouter":
        st.markdown("**How to get an API Key:**")
        st.markdown("1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)")
        st.markdown("2. Sign up / Log in")
        st.markdown("3. Click **Create Key**")
        st.markdown("4. Copy the key (starts with `sk-or-v1-`)")
        st.markdown("5. Paste it in the sidebar")
    elif provider_choice == "OpenAI":
        st.markdown("**How to get an API Key:**")
        st.markdown("1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)")
        st.markdown("2. Create a new secret key")
        st.markdown("3. Paste it in the sidebar")
    elif provider_choice == "Anthropic":
        st.markdown("**How to get an API Key:**")
        st.markdown("1. Go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)")
        st.markdown("2. Create a new API key")
        st.markdown("3. Paste it in the sidebar")
    st.markdown("---")
    st.markdown("**What the Agents Do:**")
    st.markdown("- 📊 **Fundamentals Analyst** - Company financials & valuation")
    st.markdown("- 😊 **Sentiment Analyst** - Social media & market mood")
    st.markdown("- 📰 **News Analyst** - Global news & macro trends")
    st.markdown("- 📈 **Technical Analyst** - Indicators & price patterns")
    st.markdown("- ⚠️ **Risk Manager** - Risk assessment & position sizing")
    st.markdown("- 💼 **Portfolio Manager** - Final decision & execution")
    st.stop()

if not stock_symbol:
    st.info("Enter a stock symbol in the sidebar.")
    st.stop()

# Fetch stock data
with st.spinner(f"Fetching data for {stock_symbol}..."):
    data, info, ticker = fetch_stock_data(stock_symbol, period="6mo")
    if data is None or data.empty:
        st.error(f"No data found for '{stock_symbol}'.")
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
tech_summary = f"Current Price: ${last_price:.2f}\n"
tech_summary += f"52W High: ${data['High'].max():.2f} | 52W Low: ${data['Low'].min():.2f}\n"
tech_summary += f"RSI: {results['smart_score'].get('RSI', 'N/A')}\n"
tech_summary += f"MACD: {results['smart_score'].get('MACD', 'N/A')}\n"
tech_summary += f"SMA: {results['smart_score'].get('SMA Cross', 'N/A')}\n"
tech_summary += f"Bollinger: {results['smart_score'].get('Bollinger', 'N/A')}\n"
tech_summary += f"Stochastic: {results['smart_score'].get('Stochastic', 'N/A')}\n"
tech_summary += f"VWAP: {results['smart_score'].get('VWAP', 'N/A')}\n"
tech_summary += f"Volume: {results['smart_score'].get('Volume', 'N/A')}\n"
tech_summary += f"Trend: {results['trend_meter'].get('ADX', 'N/A')}\n"
patterns_str = ', '.join([f"{k} ({v})" for k, v in results['patterns'].items()]) if results['patterns'] else 'None detected'
tech_summary += f"Patterns: {patterns_str}"

# RUN AGENTS
st.markdown("---")
st.subheader(f"Analyzing {company_name} ({stock_symbol})")

agents_config = [
    {
        "name": "Fundamentals Analyst",
        "icon": "📊",
        "color": "#3498db",
        "system": "You are a fundamentals analyst. Analyze company financials, valuation metrics, earnings, and growth prospects. Be concise (3-4 sentences). End with a clear signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze {company_name} ({stock_symbol}) fundamentals.\n\nCompany Info:\n- Sector: {sector}\n- Market Cap: {market_cap}\n- P/E Ratio: {pe_ratio}\n- EPS: {eps}\n\nProvide your assessment and signal."
    },
    {
        "name": "Technical Analyst", 
        "icon": "📈",
        "color": "#9b59b6",
        "system": "You are a technical analyst. Analyze price action, indicators, and patterns. Be concise (3-4 sentences). End with a clear signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze {company_name} ({stock_symbol}) technicals.\n\n{tech_summary}\n\nProvide your assessment and signal."
    },
    {
        "name": "Sentiment Analyst",
        "icon": "😊",
        "color": "#e67e22",
        "system": "You are a sentiment analyst. Analyze market mood, social media trends, and investor psychology. Be concise (3-4 sentences). End with a clear signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze market sentiment for {company_name} ({stock_symbol}). Current price ${last_price:.2f}, sector {sector}. Consider recent price action and general market mood. Provide your assessment and signal."
    },
    {
        "name": "News Analyst",
        "icon": "📰",
        "color": "#1abc9c",
        "system": "You are a news analyst. Consider recent global events, sector trends, and macroeconomic factors. Be concise (3-4 sentences). End with a clear signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.",
        "prompt": f"Analyze news and macro factors affecting {company_name} ({stock_symbol}) in the {sector} sector. Current price ${last_price:.2f}. Provide your assessment and signal."
    },
    {
        "name": "Risk Manager",
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
            content, error = call_llm(
                agent['system'],
                agent['prompt'],
                provider_choice,
                api_key,
                model_choice,
                ollama_url
            )

        if error:
            st.error(f"Error: {error}")
            agent_results[agent['name']] = {"signal": "ERROR", "score": 0, "emoji": "⚪", "reasoning": error}
        else:
            parsed = parse_agent_response(content)
            agent_results[agent['name']] = parsed

            st.markdown(f"<div style='background-color:{agent['color']}15;border-radius:8px;padding:10px;margin:5px 0;border-left:3px solid {agent['color']};'>"
                       f"<p style='margin:0;font-size:1.2rem;font-weight:bold;color:{agent['color']};'>{parsed['emoji']} {parsed['signal']}</p>"
                       f"<p style='margin:5px 0 0 0;font-size:0.85rem;color:#444;line-height:1.4;'>{parsed['reasoning'][:200]}...</p>"
                       f"</div>", unsafe_allow_html=True)

# PORTFOLIO MANAGER - FINAL DECISION
st.markdown("---")
st.subheader("Portfolio Manager - Final Decision")

# Aggregate agent scores
valid_scores = [r['score'] for r in agent_results.values() if r['signal'] != "ERROR"]
if valid_scores:
    avg_score = np.mean(valid_scores)

    # Build consensus prompt
    consensus_summary = "\n".join([
        f"{name}: {r['emoji']} {r['signal']} (Score: {r['score']:+.1f})"
        for name, r in agent_results.items() if r['signal'] != "ERROR"
    ])

    pm_system = "You are a portfolio manager. Review all analyst inputs and make a final trading decision with position sizing. Be decisive and concise (4-5 sentences)."
    pm_prompt = f"Review the following analyst consensus for {company_name} ({stock_symbol}):\n\n{consensus_summary}\n\nAverage Score: {avg_score:+.2f}/1.00\n\nMake a final decision: STRONG BUY / BUY / HOLD / SELL / STRONG SELL. Include recommended position size (e.g., 5% portfolio allocation) and stop-loss level."

    with st.spinner("Portfolio Manager deliberating..."):
        pm_content, pm_error = call_llm(pm_system, pm_prompt, provider_choice, api_key, model_choice, ollama_url)

    if pm_error:
        st.error(f"Error: {pm_error}")
    else:
        pm_parsed = parse_agent_response(pm_content)

        final_color = "#2ca02c" if "BUY" in pm_parsed['signal'] else "#DC143C" if "SELL" in pm_parsed['signal'] else "#FFA500"

        st.markdown(f"<div style='background:linear-gradient(135deg, {final_color}15 0%, {final_color}08 100%);border-radius:16px;padding:25px;margin:15px 0;border:2px solid {final_color};text-align:center;'>"
                   f"<h2 style='margin:0 0 10px 0;color:{final_color};font-size:2rem;'>{pm_parsed['emoji']} {pm_parsed['signal']}</h2>"
                   f"<p style='margin:0;color:#333;font-size:1rem;line-height:1.6;max-width:800px;margin:0 auto;'>{pm_parsed['reasoning']}</p>"
                   f"</div>", unsafe_allow_html=True)

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
st.markdown("<center><small>Powered by OpenRouter / OpenAI / Anthropic / Ollama | Multi-Agent LLM Debate | Not financial advice</small></center>", unsafe_allow_html=True)
