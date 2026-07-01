# 📈 Stock Forecast & Technical Analysis App

A comprehensive Streamlit application for stock analysis with 15+ technical indicators and 4 forecasting models.

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
- Bollinger Bands
- SMA (20, 50)
- EMA (12, 26)
- RSI
- MACD
- Stochastic Oscillator
- ADX
- MFI (Money Flow Index)
- VWAP
- Ichimoku Cloud
- Parabolic SAR
- Fibonacci Retracement

### Forecasting Models
- Prophet (Facebook)
- ARIMA
- Moving Average + Trend
- Monte Carlo Simulation

## 📝 Usage

1. Enter a stock symbol (e.g., AAPL, TSLA, MSFT)
2. Select time period and interval
3. Toggle indicators on/off
4. Choose forecast model and days
5. Click **Analyze Stock**
