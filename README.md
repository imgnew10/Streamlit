# 📈 Stock Forecast & Technical Analysis App

A lightweight Streamlit app for stock analysis with 15+ technical indicators and forecasting models.

## 🚀 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

## 📦 Local Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ✨ Features

### Technical Indicators
- Candlestick Chart with Volume
- Bollinger Bands | SMA (20, 50) | EMA (12, 26)
- RSI | MACD | Stochastic | ADX | MFI | VWAP
- Ichimoku Cloud | Parabolic SAR | Fibonacci Retracement

### Forecasting Models
- **Always Available:** Moving Average + Trend, Monte Carlo Simulation
- **Optional (install separately):** Prophet, ARIMA

## 📝 Usage

1. Enter a stock symbol (e.g., AAPL, TSLA, MSFT, BTC-USD)
2. Select time period and interval
3. Toggle indicators on/off
4. Choose forecast model and days
5. Click **Analyze Stock**

## 🔧 Optional: Enable Advanced Forecasting

To add Prophet and ARIMA models, update `requirements.txt`:
```
streamlit>=1.28.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
prophet>=1.1.5
statsmodels>=0.14.0
```

> ⚠️ Note: Prophet requires system build tools and may fail on Streamlit Cloud free tier due to memory limits.
