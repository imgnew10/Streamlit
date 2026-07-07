
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import warnings
warnings.filterwarnings('ignore')

# ==================== RATE LIMITING & RETRY ====================
_last_request_time = 0
_min_delay = 1.5  # seconds between requests

def _wait_for_rate_limit():
    """Ensure minimum delay between Yahoo Finance requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _min_delay:
        time.sleep(_min_delay - elapsed + random.uniform(0.2, 0.5))
    _last_request_time = time.time()

def _fetch_with_retry(ticker_symbol, period="2y", max_retries=3):
    """Fetch stock data with retry logic and rate limiting."""
    for attempt in range(max_retries):
        try:
            _wait_for_rate_limit()
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period=period, interval="1d")
            info = ticker.info
            if data.empty:
                return None, None, None
            return data, info, ticker
        except Exception as e:
            error_msg = str(e).lower()
            if "too many requests" in error_msg or "rate limited" in error_msg or "429" in error_msg:
                wait_time = (attempt + 1) * 3 + random.uniform(1, 3)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
            return None, None, None
    return None, None, None

def fetch_stock_data(ticker_symbol, period="2y"):
    """Fetch historical data for a stock with rate limiting."""
    return _fetch_with_retry(ticker_symbol, period)

def fetch_market_data(period="2y"):
    """Fetch SPY for market comparison with rate limiting."""
    try:
        _wait_for_rate_limit()
        spy = yf.Ticker("SPY").history(period=period, interval="1d")
        return spy['Close'] if not spy.empty else None
    except Exception:
        return None

# ==================== CANDLESTICK PATTERNS ====================
def detect_candlestick_patterns(data):
    """Detect basic candlestick patterns."""
    patterns = {}
    if len(data) < 2:
        return patterns

    o = data['Open'].iloc[-1]
    h = data['High'].iloc[-1]
    l = data['Low'].iloc[-1]
    c = data['Close'].iloc[-1]
    body = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    total_range = h - l

    if total_range == 0:
        return patterns

    if lower_shadow > 2 * body and upper_shadow < body:
        patterns['Hammer'] = 'Bullish' if c > o else 'Bearish'

    if body < total_range * 0.1:
        patterns['Doji'] = 'Neutral/Reversal'

    if len(data) >= 2:
        prev_o = data['Open'].iloc[-2]
        prev_c = data['Close'].iloc[-2]
        prev_body = abs(prev_c - prev_o)

        if c > o and prev_c < prev_o and c > prev_o and o < prev_c and body > prev_body:
            patterns['Bullish Engulfing'] = 'Strong Bullish'
        elif c < o and prev_c > prev_o and c < prev_o and o > prev_c and body > prev_body:
            patterns['Bearish Engulfing'] = 'Strong Bearish'

    return patterns

# ==================== ALL SIGNALS CALCULATION ====================
def calculate_all_signals(data, info, ticker, spy_close=None):
    """
    Calculate TradingView-style combined signals:
    - Smart Score (multi-indicator composite)
    - Trend Meter (directional strength)
    - Alpha Signal (momentum vs market)
    - Technical Rating (pattern-based)
    Returns unified score and detailed breakdown.
    """
    if data is None or data.empty:
        return None

    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    last_price = close.iloc[-1]

    results = {
        'smart_score': {},
        'trend_meter': {},
        'alpha_signal': {},
        'technical_rating': {},
        'patterns': {},
        'analyst': {}
    }

    # ==================== SMART SCORE (10 indicators) ====================
    smart_votes = []

    # 1. RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.iloc[-1]
    if latest_rsi < 30:
        smart_votes.append(1.0)
        results['smart_score']['RSI'] = 'Oversold (+1.0)'
    elif latest_rsi > 70:
        smart_votes.append(-1.0)
        results['smart_score']['RSI'] = 'Overbought (-1.0)'
    else:
        smart_votes.append(0)
        results['smart_score']['RSI'] = f'Neutral ({latest_rsi:.0f})'

    # 2. MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
        smart_votes.append(1.0)
        results['smart_score']['MACD'] = 'Bullish Crossover (+1.0)'
    elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
        smart_votes.append(-1.0)
        results['smart_score']['MACD'] = 'Bearish Crossover (-1.0)'
    elif macd_line.iloc[-1] > macd_signal.iloc[-1]:
        smart_votes.append(0.5)
        results['smart_score']['MACD'] = 'Bullish (+0.5)'
    else:
        smart_votes.append(-0.5)
        results['smart_score']['MACD'] = 'Bearish (-0.5)'

    # 3. SMA 20 vs 50
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    if sma20.iloc[-1] > sma50.iloc[-1] and sma20.iloc[-5] <= sma50.iloc[-5]:
        smart_votes.append(1.0)
        results['smart_score']['SMA Cross'] = 'Golden Cross (+1.0)'
    elif sma20.iloc[-1] < sma50.iloc[-1] and sma20.iloc[-5] >= sma50.iloc[-5]:
        smart_votes.append(-1.0)
        results['smart_score']['SMA Cross'] = 'Death Cross (-1.0)'
    elif sma20.iloc[-1] > sma50.iloc[-1]:
        smart_votes.append(0.5)
        results['smart_score']['SMA Cross'] = 'Above 50-day (+0.5)'
    else:
        smart_votes.append(-0.5)
        results['smart_score']['SMA Cross'] = 'Below 50-day (-0.5)'

    # 4. Bollinger Bands
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    if close.iloc[-1] < bb_lower.iloc[-1]:
        smart_votes.append(1.0)
        results['smart_score']['Bollinger'] = 'Below Lower Band (+1.0)'
    elif close.iloc[-1] > bb_upper.iloc[-1]:
        smart_votes.append(-1.0)
        results['smart_score']['Bollinger'] = 'Above Upper Band (-1.0)'
    else:
        smart_votes.append(0)
        results['smart_score']['Bollinger'] = 'Inside Bands (0)'

    # 5. Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * ((close - low14) / (high14 - low14))
    stoch_d = stoch_k.rolling(3).mean()
    if stoch_k.iloc[-1] < 20 and stoch_d.iloc[-1] < 20:
        smart_votes.append(1.0)
        results['smart_score']['Stochastic'] = 'Oversold (+1.0)'
    elif stoch_k.iloc[-1] > 80 and stoch_d.iloc[-1] > 80:
        smart_votes.append(-1.0)
        results['smart_score']['Stochastic'] = 'Overbought (-1.0)'
    else:
        smart_votes.append(0)
        results['smart_score']['Stochastic'] = f'Neutral ({stoch_k.iloc[-1]:.0f})'

    # 6. VWAP
    tp = (high + low + close) / 3
    vwap = (tp * volume).cumsum() / volume.cumsum()
    if close.iloc[-1] > vwap.iloc[-1]:
        smart_votes.append(0.5)
        results['smart_score']['VWAP'] = 'Above VWAP (+0.5)'
    else:
        smart_votes.append(-0.5)
        results['smart_score']['VWAP'] = 'Below VWAP (-0.5)'

    # 7. Volume Trend
    vol_sma20 = volume.rolling(20).mean()
    if volume.iloc[-1] > vol_sma20.iloc[-1] * 1.5 and close.iloc[-1] > close.iloc[-2]:
        smart_votes.append(1.0)
        results['smart_score']['Volume'] = 'High Volume Breakout (+1.0)'
    elif volume.iloc[-1] > vol_sma20.iloc[-1] * 1.5 and close.iloc[-1] < close.iloc[-2]:
        smart_votes.append(-1.0)
        results['smart_score']['Volume'] = 'High Volume Sell-off (-1.0)'
    else:
        smart_votes.append(0)
        results['smart_score']['Volume'] = 'Normal Volume (0)'

    # 8. Price vs 52-week range
    yr_high = high.max()
    yr_low = low.min()
    if yr_high > yr_low:
        position = (close.iloc[-1] - yr_low) / (yr_high - yr_low)
        if position < 0.2:
            smart_votes.append(1.0)
            results['smart_score']['52W Range'] = 'Near 52W Low (+1.0)'
        elif position > 0.8:
            smart_votes.append(-1.0)
            results['smart_score']['52W Range'] = 'Near 52W High (-1.0)'
        else:
            smart_votes.append(0)
            results['smart_score']['52W Range'] = f'Mid Range ({position*100:.0f}%)'

    # 9. Analyst Ratings
    analyst_score = 0
    analyst_count = 0
    analyst_target = None
    try:
        rec_sum = ticker.recommendations_summary
        if rec_sum is not None and not rec_sum.empty:
            latest = rec_sum.iloc[-1]
            sb = latest.get('strongBuy', 0)
            b = latest.get('buy', 0)
            h = latest.get('hold', 0)
            s = latest.get('sell', 0)
            ss = latest.get('strongSell', 0)
            total = sb + b + h + s + ss
            if total > 0:
                analyst_score = (sb*2 + b*1 + h*0 + s*(-1) + ss*(-2)) / (total*2)
                analyst_count = total
    except Exception:
        pass

    if analyst_count == 0 and info:
        rec_key = info.get('recommendationKey', '')
        if 'buy' in rec_key.lower():
            analyst_score = 0.3
        elif 'sell' in rec_key.lower():
            analyst_score = -0.3

    try:
        pt = ticker.analyst_price_target
        if pt is not None and not pt.empty:
            analyst_target = pt.get('current', pt.iloc[-1] if hasattr(pt, 'iloc') else None)
    except Exception:
        pass

    if analyst_target is None and info:
        analyst_target = info.get('targetMeanPrice') or info.get('targetHighPrice')

    smart_votes.append(analyst_score)
    results['smart_score']['Analysts'] = f'{"Strong Buy" if analyst_score > 0.5 else "Buy" if analyst_score > 0.2 else "Hold" if analyst_score > -0.2 else "Sell" if analyst_score > -0.5 else "Strong Sell"} ({analyst_count} analysts)' if analyst_count > 0 else f'Info: {info.get("recommendationKey", "N/A")}' if info else 'N/A'

    # 10. EPS Trend
    eps_growth = info.get('earningsGrowth') if info else None
    if eps_growth and not np.isnan(eps_growth):
        if eps_growth > 0.2:
            smart_votes.append(1.0)
            results['smart_score']['EPS Growth'] = 'Strong Growth (+1.0)'
        elif eps_growth > 0:
            smart_votes.append(0.5)
            results['smart_score']['EPS Growth'] = 'Positive Growth (+0.5)'
        else:
            smart_votes.append(-0.5)
            results['smart_score']['EPS Growth'] = 'Negative Growth (-0.5)'
    else:
        smart_votes.append(0)
        results['smart_score']['EPS Growth'] = 'No Data (0)'

    smart_score = np.mean(smart_votes)

    # ==================== TREND METER ====================
    trend_votes = []

    # ADX
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = abs(100 * (minus_dm.rolling(14).mean() / atr14))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(14).mean()

    latest_adx = adx.iloc[-1] if not adx.empty else 0
    if latest_adx > 25:
        if plus_di.iloc[-1] > minus_di.iloc[-1]:
            trend_votes.append(1.0)
            results['trend_meter']['ADX'] = 'Strong Uptrend (+1.0)'
        else:
            trend_votes.append(-1.0)
            results['trend_meter']['ADX'] = 'Strong Downtrend (-1.0)'
    else:
        trend_votes.append(0)
        results['trend_meter']['ADX'] = f'Weak/No Trend ({latest_adx:.0f})'

    # Volume trend
    vol_change = ((volume.iloc[-1] - volume.iloc[-20]) / volume.iloc[-20]) * 100 if len(volume) >= 20 else 0
    if vol_change > 50 and close.iloc[-1] > close.iloc[-20]:
        trend_votes.append(1.0)
        results['trend_meter']['Volume Trend'] = 'Rising Volume + Price (+1.0)'
    elif vol_change > 50 and close.iloc[-1] < close.iloc[-20]:
        trend_votes.append(-1.0)
        results['trend_meter']['Volume Trend'] = 'Rising Volume - Price (-1.0)'
    else:
        trend_votes.append(0)
        results['trend_meter']['Volume Trend'] = f'Vol Change {vol_change:+.0f}% (0)'

    # Price slope
    x = np.arange(20)
    y = close.iloc[-20:].values
    slope = np.polyfit(x, y, 1)[0] if len(y) == 20 else 0
    if slope > last_price * 0.002:
        trend_votes.append(1.0)
        results['trend_meter']['Price Slope'] = 'Steep Rise (+1.0)'
    elif slope < -last_price * 0.002:
        trend_votes.append(-1.0)
        results['trend_meter']['Price Slope'] = 'Steep Fall (-1.0)'
    else:
        trend_votes.append(0)
        results['trend_meter']['Price Slope'] = 'Flat (0)'

    trend_score = np.mean(trend_votes)

    # ==================== ALPHA SIGNAL ====================
    alpha_votes = []

    # Relative strength vs SPY
    if spy_close is not None and len(spy_close) >= 20:
        stock_return_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100
        spy_return_20d = (spy_close.iloc[-1] / spy_close.iloc[-20] - 1) * 100
        relative_strength = stock_return_20d - spy_return_20d
        if relative_strength > 5:
            alpha_votes.append(1.0)
            results['alpha_signal']['vs S&P 500'] = f'Outperforming +{relative_strength:.1f}% (+1.0)'
        elif relative_strength < -5:
            alpha_votes.append(-1.0)
            results['alpha_signal']['vs S&P 500'] = f'Underperforming {relative_strength:.1f}% (-1.0)'
        else:
            alpha_votes.append(0)
            results['alpha_signal']['vs S&P 500'] = f'Tracking Market ({relative_strength:+.1f}%)'
    else:
        alpha_votes.append(0)
        results['alpha_signal']['vs S&P 500'] = 'No Market Data (0)'

    # Momentum
    roc_10 = ((close.iloc[-1] / close.iloc[-10]) - 1) * 100 if len(close) >= 10 else 0
    if roc_10 > 10:
        alpha_votes.append(1.0)
        results['alpha_signal']['10-Day Momentum'] = f'Strong +{roc_10:.1f}% (+1.0)'
    elif roc_10 < -10:
        alpha_votes.append(-1.0)
        results['alpha_signal']['10-Day Momentum'] = f'Weak {roc_10:.1f}% (-1.0)'
    else:
        alpha_votes.append(0)
        results['alpha_signal']['10-Day Momentum'] = f'Moderate ({roc_10:+.1f}%)'

    # Volatility regime
    vol_20d = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
    if vol_20d < 20:
        alpha_votes.append(0.5)
        results['alpha_signal']['Volatility'] = f'Low Vol {vol_20d:.1f}% (+0.5)'
    elif vol_20d > 40:
        alpha_votes.append(-0.5)
        results['alpha_signal']['Volatility'] = f'High Vol {vol_20d:.1f}% (-0.5)'
    else:
        alpha_votes.append(0)
        results['alpha_signal']['Volatility'] = f'Medium Vol {vol_20d:.1f}% (0)'

    alpha_score = np.mean(alpha_votes)

    # ==================== TECHNICAL RATING ====================
    patterns = detect_candlestick_patterns(data)
    results['patterns'] = patterns

    if 'Bullish Engulfing' in patterns or 'Hammer' in patterns:
        tech_score = 1.0
        results['technical_rating']['Pattern'] = 'Bullish Pattern Detected (+1.0)'
    elif 'Bearish Engulfing' in patterns or 'Hanging Man' in patterns:
        tech_score = -1.0
        results['technical_rating']['Pattern'] = 'Bearish Pattern Detected (-1.0)'
    elif 'Doji' in patterns:
        tech_score = 0
        results['technical_rating']['Pattern'] = 'Doji - Reversal Possible (0)'
    else:
        tech_score = 0
        results['technical_rating']['Pattern'] = 'No Clear Pattern (0)'

    # ==================== COMBINED UNIFIED SCORE ====================
    unified_score = smart_score * 0.40 + trend_score * 0.25 + alpha_score * 0.20 + tech_score * 0.15

    if unified_score > 0.5:
        signal_text = "STRONG BUY"
        signal_emoji = "🟢"
        color = "#2ca02c"
    elif unified_score > 0.2:
        signal_text = "BUY"
        signal_emoji = "🟢"
        color = "#32CD32"
    elif unified_score > -0.2:
        signal_text = "HOLD"
        signal_emoji = "🟡"
        color = "#FFA500"
    elif unified_score > -0.5:
        signal_text = "SELL"
        signal_emoji = "🔴"
        color = "#FF6347"
    else:
        signal_text = "STRONG SELL"
        signal_emoji = "🔴"
        color = "#DC143C"

    results['unified'] = {
        'score': unified_score,
        'signal': signal_text,
        'emoji': signal_emoji,
        'color': color,
        'smart_score': smart_score,
        'trend_score': trend_score,
        'alpha_score': alpha_score,
        'tech_score': tech_score,
        'analyst_target': analyst_target,
        'analyst_count': analyst_count
    }

    return results

def generate_forecast(data, score, days):
    """Generate price forecast based on unified recommendation score."""
    close = data['Close']
    last_price = close.iloc[-1]
    returns = close.pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()

    trend = (close.iloc[-1] - close.iloc[-20]) / 20
    adjusted_trend = trend * (1 + score * 2)

    simulations = 500
    sim_results = []
    for _ in range(simulations):
        prices = [last_price]
        for _ in range(days):
            drift = adjusted_trend + mu * prices[-1]
            shock = np.random.normal(0, sigma * prices[-1])
            new_price = prices[-1] + drift + shock
            prices.append(max(new_price, 0.01))
        sim_results.append(prices[1:])

    sim_array = np.array(sim_results)
    mean_f = np.mean(sim_array, axis=0)
    p10 = np.percentile(sim_array, 10, axis=0)
    p90 = np.percentile(sim_array, 90, axis=0)

    if score > 0.3:
        adjust = last_price * 0.003 * score
        mean_f = mean_f + np.linspace(0, adjust * days, days)
    elif score < -0.3:
        adjust = last_price * 0.003 * abs(score)
        mean_f = mean_f - np.linspace(0, adjust * days, days)

    future_dates = pd.date_range(start=close.index[-1] + timedelta(days=1), periods=days, freq='B')
    return future_dates, mean_f, p10, p90
