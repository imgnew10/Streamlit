import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

st.markdown("<h1 style='text-align:center;color:#1f77b4;'>📈 Stock Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Enter a stock ticker. We analyze analyst ratings, technical indicators & trends to predict the future price.</p>", unsafe_allow_html=True)

# ============ SINGLE INPUT ============
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    stock_symbol = st.text_input("", value="AAPL", placeholder="e.g. AAPL, TSLA, MSFT, NVDA", label_visibility="collapsed").upper().strip()

if not stock_symbol:
    st.stop()

with st.spinner(f"Analyzing {stock_symbol}..."):
    try:
        ticker = yf.Ticker(stock_symbol)
        hist = ticker.history(period="2y", interval="1d")
        info = ticker.info

        if hist.empty:
            st.error(f"No data found for '{stock_symbol}'. Try another ticker.")
            st.stop()

        # ============ ANALYST DATA ============
        analyst_score = 0  # -1 (sell) to +1 (buy)
        analyst_target = None
        analyst_count = 0

        # Try recommendations summary
        try:
            rec_sum = ticker.recommendations_summary
            if rec_sum is not None and not rec_sum.empty:
                latest = rec_sum.iloc[-1]
                strong_buy = latest.get('strongBuy', 0)
                buy = latest.get('buy', 0)
                hold = latest.get('hold', 0)
                sell = latest.get('sell', 0)
                strong_sell = latest.get('strongSell', 0)
                total = strong_buy + buy + hold + sell + strong_sell
                if total > 0:
                    analyst_score = (strong_buy*2 + buy*1 + hold*0 + sell*(-1) + strong_sell*(-2)) / (total*2)
                    analyst_count = total
        except:
            pass

        # Try analyst price target
        try:
            pt = ticker.analyst_price_target
            if pt is not None and not pt.empty:
                analyst_target = pt.get('current', pt.iloc[-1] if hasattr(pt, 'iloc') else None)
        except:
            pass

        # Fallback: use info keys
        if analyst_target is None:
            analyst_target = info.get('targetMeanPrice') or info.get('targetHighPrice') or info.get('currentPrice')

        # Try recommendation key
        rec_key = info.get('recommendationKey', '')
        if 'buy' in rec_key.lower() and analyst_score == 0:
            analyst_score = 0.3
        elif 'sell' in rec_key.lower() and analyst_score == 0:
            analyst_score = -0.3

        # ============ TECHNICAL INDICATORS ============
        close = hist['Close']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        # Bollinger Bands
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std

        # SMA cross
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()

        # Stochastic
        low14 = hist['Low'].rolling(14).min()
        high14 = hist['High'].rolling(14).max()
        stoch_k = 100 * ((close - low14) / (high14 - low14))
        stoch_d = stoch_k.rolling(3).mean()

        # ATR
        tr1 = hist['High'] - hist['Low']
        tr2 = abs(hist['High'] - close.shift())
        tr3 = abs(hist['Low'] - close.shift())
        atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

        # VWAP
        tp = (hist['High'] + hist['Low'] + close) / 3
        vwap = (tp * hist['Volume']).cumsum() / hist['Volume'].cumsum()

        # ============ MIROFISH SIGNAL (Composite) ============
        # Combine all signals into one score: -1 (strong sell) to +1 (strong buy)
        signals = []

        # 1. RSI signal
        if latest_rsi < 30:
            signals.append(0.5)   # oversold = buy
        elif latest_rsi > 70:
            signals.append(-0.5)  # overbought = sell
        else:
            signals.append(0)

        # 2. MACD signal
        if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
            signals.append(0.6)   # bullish crossover
        elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
            signals.append(-0.6)  # bearish crossover
        elif macd.iloc[-1] > macd_signal.iloc[-1]:
            signals.append(0.3)
        else:
            signals.append(-0.3)

        # 3. SMA crossover
        if sma20.iloc[-1] > sma50.iloc[-1] and sma20.iloc[-5] <= sma50.iloc[-5]:
            signals.append(0.5)   # golden cross
        elif sma20.iloc[-1] < sma50.iloc[-1] and sma20.iloc[-5] >= sma50.iloc[-5]:
            signals.append(-0.5)  # death cross
        elif sma20.iloc[-1] > sma50.iloc[-1]:
            signals.append(0.2)
        else:
            signals.append(-0.2)

        # 4. Bollinger position
        if close.iloc[-1] < bb_lower.iloc[-1]:
            signals.append(0.4)   # below lower band = buy
        elif close.iloc[-1] > bb_upper.iloc[-1]:
            signals.append(-0.4)  # above upper band = sell
        else:
            signals.append(0)

        # 5. Stochastic
        if stoch_k.iloc[-1] < 20 and stoch_d.iloc[-1] < 20:
            signals.append(0.4)
        elif stoch_k.iloc[-1] > 80 and stoch_d.iloc[-1] > 80:
            signals.append(-0.4)
        else:
            signals.append(0)

        # 6. Price vs VWAP
        if close.iloc[-1] > vwap.iloc[-1]:
            signals.append(0.2)
        else:
            signals.append(-0.2)

        # 7. Analyst rating
        signals.append(analyst_score * 0.8)

        # Composite Mirofish score
        mirofish_score = np.mean(signals)  # -1 to +1
        mirofish_signal = "🟢 STRONG BUY" if mirofish_score > 0.5 else                          "🟢 BUY" if mirofish_score > 0.2 else                          "🟡 HOLD" if mirofish_score > -0.2 else                          "🔴 SELL" if mirofish_score > -0.5 else "🔴 STRONG SELL"

        # ============ FORECAST ============
        forecast_days = 30
        last_price = close.iloc[-1]
        returns = close.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()

        # Trend from last 20 days
        trend = (close.iloc[-1] - close.iloc[-20]) / 20

        # Adjust trend by Mirofish score
        adjusted_trend = trend * (1 + mirofish_score * 2)

        # Monte Carlo with adjusted drift
        simulations = 500
        sim_results = []
        for _ in range(simulations):
            prices = [last_price]
            for _ in range(forecast_days):
                drift = adjusted_trend + mu * prices[-1]
                shock = np.random.normal(0, sigma * prices[-1])
                new_price = prices[-1] + drift + shock
                prices.append(max(new_price, 0.01))
            sim_results.append(prices[1:])

        sim_array = np.array(sim_results)
        mean_forecast = np.mean(sim_array, axis=0)
        p10 = np.percentile(sim_array, 10, axis=0)
        p90 = np.percentile(sim_array, 90, axis=0)

        # Also compute weighted forecast using Mirofish
        if mirofish_score > 0.3:
            # Bullish: shift mean up
            bullish_adjust = last_price * 0.003 * mirofish_score
            mean_forecast = mean_forecast + np.linspace(0, bullish_adjust * forecast_days, forecast_days)
        elif mirofish_score < -0.3:
            # Bearish: shift mean down
            bearish_adjust = last_price * 0.003 * abs(mirofish_score)
            mean_forecast = mean_forecast - np.linspace(0, bearish_adjust * forecast_days, forecast_days)

        future_dates = pd.date_range(start=close.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')

        # ============ DISPLAY METRICS ============
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("Current Price", f"${last_price:.2f}")
        with mcol2:
            pred = mean_forecast[-1]
            change = ((pred - last_price) / last_price) * 100
            st.metric("30-Day Target", f"${pred:.2f}", f"{change:+.1f}%")
        with mcol3:
            st.metric("Mirofish Signal", mirofish_signal)
        with mcol4:
            st.metric("Analyst Rating", f"{'Buy' if analyst_score > 0.2 else 'Sell' if analyst_score < -0.2 else 'Hold'} ({analyst_count} analysts)" if analyst_count > 0 else "N/A")
        with mcol5:
            st.metric("RSI", f"{latest_rsi:.0f}")

        # ============ MAIN CHART ============
        fig = go.Figure()

        # Historical price
        fig.add_trace(go.Scatter(
            x=close.index, y=close.values,
            name="Historical Price",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        ))

        # SMA 20
        fig.add_trace(go.Scatter(
            x=sma20.index, y=sma20.values,
            name="SMA 20",
            line=dict(color="orange", width=1.5, dash="dot"),
            opacity=0.7
        ))

        # SMA 50
        fig.add_trace(go.Scatter(
            x=sma50.index, y=sma50.values,
            name="SMA 50",
            line=dict(color="purple", width=1.5, dash="dot"),
            opacity=0.7
        ))

        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=bb_upper.index, y=bb_upper.values,
            name="BB Upper",
            line=dict(color="rgba(255,0,0,0.2)", width=1),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=bb_lower.index, y=bb_lower.values,
            name="BB Lower",
            line=dict(color="rgba(255,0,0,0.2)", width=1),
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.05)",
            showlegend=False
        ))

        # Forecast mean
        fig.add_trace(go.Scatter(
            x=future_dates, y=mean_forecast,
            name="Predicted Price",
            line=dict(color="#2ca02c", width=2.5, dash="dash"),
            hovertemplate="Date: %{x}<br>Predicted: $%{y:.2f}<extra></extra>"
        ))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(p90) + list(p10)[::-1],
            fill="toself",
            fillcolor="rgba(44,160,44,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band (10%-90%)",
            hoverinfo="skip"
        ))

        # Analyst target line if available
        if analyst_target and not np.isnan(analyst_target):
            fig.add_hline(
                y=analyst_target,
                line=dict(color="gold", width=2, dash="dashdot"),
                annotation_text=f"Analyst Target: ${analyst_target:.2f}",
                annotation_position="top right"
            )

        fig.update_layout(
            title=dict(
                text=f"<b>{info.get('shortName', stock_symbol)} ({stock_symbol})</b><br><sup>2-Year History + 30-Day AI Forecast</sup>",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=600,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                rangeslider=dict(visible=False),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)"
            ),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
        )

        # Add vertical line at forecast start
        fig.add_vline(x=close.index[-1], line=dict(color="gray", width=1, dash="dash"),
                      annotation_text="Forecast Start", annotation_position="top")

        st.plotly_chart(fig, use_container_width=True)

        # ============ SIGNAL BREAKDOWN ============
        st.subheader("🔍 Signal Breakdown")
        bcol1, bcol2, bcol3 = st.columns(3)

        with bcol1:
            st.markdown("**Technical Indicators**")
            st.write(f"• RSI: {latest_rsi:.1f} ({'Oversold' if latest_rsi < 30 else 'Overbought' if latest_rsi > 70 else 'Neutral'})")
            st.write(f"• MACD: {'Bullish' if macd_hist.iloc[-1] > 0 else 'Bearish'} crossover")
            st.write(f"• SMA: {'Golden' if sma20.iloc[-1] > sma50.iloc[-1] else 'Death'} cross trend")
            st.write(f"• Bollinger: {'Below lower' if close.iloc[-1] < bb_lower.iloc[-1] else 'Above upper' if close.iloc[-1] > bb_upper.iloc[-1] else 'Inside bands'}")
            st.write(f"• Stochastic: {stoch_k.iloc[-1]:.1f} / {stoch_d.iloc[-1]:.1f}")

        with bcol2:
            st.markdown("**Analyst Consensus**")
            if analyst_count > 0:
                st.write(f"• Rating: {'Strong Buy' if analyst_score > 0.5 else 'Buy' if analyst_score > 0.2 else 'Hold' if analyst_score > -0.2 else 'Sell' if analyst_score > -0.5 else 'Strong Sell'}")
                st.write(f"• Analysts: {analyst_count}")
                if analyst_target:
                    st.write(f"• Price Target: ${analyst_target:.2f}")
                    upside = ((analyst_target - last_price) / last_price) * 100
                    st.write(f"• Upside: {upside:+.1f}%")
            else:
                st.write("• No analyst data available")
                st.write("• Using technical signals only")

        with bcol3:
            st.markdown("**Mirofish Score**")
            st.write(f"• Composite: {mirofish_score:.2f} / 1.00")
            st.write(f"• Signal: {mirofish_signal}")
            st.write(f"• Volatility (ATR): ${atr.iloc[-1]:.2f}")
            st.write(f"• Daily Return: {mu*100:.2f}%")
            st.write(f"• Trend (20d): ${trend:.3f}/day")

        # ============ FORECAST TABLE ============
        with st.expander("📋 View Forecast Data"):
            forecast_df = pd.DataFrame({
                "Date": future_dates.strftime("%Y-%m-%d"),
                "Predicted": mean_forecast.round(2),
                "Low (10%)": p10.round(2),
                "High (90%)": p90.round(2)
            })
            st.dataframe(forecast_df, use_container_width=True)

            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast CSV", csv, f"{stock_symbol}_forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"Error analyzing {stock_symbol}: {str(e)}")
        st.info("Try a different ticker symbol. Examples: AAPL, MSFT, TSLA, GOOGL, AMZN, NVDA, META, BTC-USD, ETH-USD")

st.markdown("---")
st.markdown("<center><small>Data from Yahoo Finance | Forecasts are estimates, not financial advice</small></center>", unsafe_allow_html=True)
