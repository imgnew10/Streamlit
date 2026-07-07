# 📈 Stock Price Predictor

One-click stock analysis with AI forecast.

## How to Use
1. Enter a stock ticker (e.g. AAPL, TSLA, MSFT)
2. See the predicted price chart instantly

## What It Shows
- **2-year historical price** with SMA 20/50 and Bollinger Bands
- **30-day predicted price** with confidence band
- **Analyst target price** (if available)
- **Mirofish composite signal** combining all indicators

## Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repo, set main file to `app.py`
4. Deploy

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
