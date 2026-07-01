
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Stock Forecast & Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .forecast-header {
        font-size: 1.5rem;
        color: #2ca02c;
        font-weight: bold;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INDICATOR FUNCTIONS ====================
class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def calculate_sma(data, period):
        return data['Close'].rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data, period):
        return data['Close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_stochastic(data, k_period=14, d_period=3):
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def calculate_atr(data, period=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    @staticmethod
    def calculate_obv(data):
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=data.index)

    @staticmethod
    def calculate_adx(data, period=14):
        plus_dm = data['High'].diff()
        minus_dm = data['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - data['Close'].shift(1))
        tr3 = abs(data['Low'] - data['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(window=period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def calculate_fibonacci_retracement(data):
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        levels = {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low
        }
        return levels

    @staticmethod
    def calculate_ichimoku(data):
        high_9 = data['High'].rolling(window=9).max()
        low_9 = data['Low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2

        high_26 = data['High'].rolling(window=26).max()
        low_26 = data['Low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        high_52 = data['High'].rolling(window=52).max()
        low_52 = data['Low'].rolling(window=52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)

        chikou_span = data['Close'].shift(-26)

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

    @staticmethod
    def calculate_vwap(data):
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap

    @staticmethod
    def calculate_mfi(data, period=14):
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']

        positive_flow = [0]
        negative_flow = [0]

        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)

        positive_mf = pd.Series(positive_flow, index=data.index).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow, index=data.index).rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

    @staticmethod
    def calculate_parabolic_sar(data, af=0.02, max_af=0.2):
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values

        sar = close.copy()
        trend = [1]
        ep = [high[0]]
        af_values = [af]

        for i in range(1, len(close)):
            if trend[-1] == 1:
                sar[i] = sar[i-1] + af_values[-1] * (ep[-1] - sar[i-1])
                if low[i] < sar[i]:
                    trend.append(-1)
                    sar[i] = ep[-1]
                    ep.append(low[i])
                    af_values.append(af)
                else:
                    trend.append(1)
                    if high[i] > ep[-1]:
                        ep.append(high[i])
                        af_values.append(min(af_values[-1] + af, max_af))
                    else:
                        ep.append(ep[-1])
                        af_values.append(af_values[-1])
            else:
                sar[i] = sar[i-1] + af_values[-1] * (ep[-1] - sar[i-1])
                if high[i] > sar[i]:
                    trend.append(1)
                    sar[i] = ep[-1]
                    ep.append(high[i])
                    af_values.append(af)
                else:
                    trend.append(-1)
                    if low[i] < ep[-1]:
                        ep.append(low[i])
                        af_values.append(min(af_values[-1] + af, max_af))
                    else:
                        ep.append(ep[-1])
                        af_values.append(af_values[-1])

        return pd.Series(sar, index=data.index)


# ==================== FORECASTING FUNCTIONS ====================
class Forecasting:
    @staticmethod
    def arima_forecast(data, periods=30):
        if not STATSMODELS_AVAILABLE:
            return None, "Statsmodels not available"

        try:
            series = data['Close'].dropna()
            model = ARIMA(series, order=(5, 1, 0))
            fitted = model.fit()

            forecast = fitted.forecast(steps=periods)
            conf_int = fitted.get_forecast(steps=periods).conf_int()

            future_dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=periods, freq='B')
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast.values,
                'Lower': conf_int.iloc[:, 0].values,
                'Upper': conf_int.iloc[:, 1].values
            })
            return forecast_df, "ARIMA Forecast"
        except Exception as e:
            return None, f"ARIMA Error: {str(e)}"

    @staticmethod
    def prophet_forecast(data, periods=30):
        if not PROPHET_AVAILABLE:
            return None, "Prophet not available. Install with: pip install prophet"

        try:
            df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            df = df.dropna()

            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            forecast_df.columns = ['Date', 'Forecast', 'Lower', 'Upper']
            return forecast_df, "Prophet Forecast"
        except Exception as e:
            return None, f"Prophet Error: {str(e)}"

    @staticmethod
    def moving_average_forecast(data, periods=30, window=20):
        try:
            last_value = data['Close'].iloc[-1]
            ma = data['Close'].rolling(window=window).mean().iloc[-1]
            trend = (data['Close'].iloc[-1] - data['Close'].iloc[-window]) / window

            forecasts = []
            lower = []
            upper = []

            for i in range(1, periods + 1):
                pred = ma + (trend * i)
                std = data['Close'].tail(window).std()
                forecasts.append(pred)
                lower.append(pred - 1.96 * std)
                upper.append(pred + 1.96 * std)

            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=periods, freq='B')
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecasts,
                'Lower': lower,
                'Upper': upper
            })
            return forecast_df, "Moving Average + Trend Forecast"
        except Exception as e:
            return None, f"MA Error: {str(e)}"

    @staticmethod
    def monte_carlo_simulation(data, periods=30, simulations=1000):
        try:
            returns = data['Close'].pct_change().dropna()
            mu = returns.mean()
            sigma = returns.std()

            last_price = data['Close'].iloc[-1]

            simulation_results = []
            for _ in range(simulations):
                prices = [last_price]
                for _ in range(periods):
                    price = prices[-1] * (1 + np.random.normal(mu, sigma))
                    prices.append(price)
                simulation_results.append(prices[1:])

            simulation_array = np.array(simulation_results)
            mean_forecast = np.mean(simulation_array, axis=0)
            lower = np.percentile(simulation_array, 5, axis=0)
            upper = np.percentile(simulation_array, 95, axis=0)

            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=periods, freq='B')
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': mean_forecast,
                'Lower': lower,
                'Upper': upper
            })
            return forecast_df, "Monte Carlo Simulation"
        except Exception as e:
            return None, f"Monte Carlo Error: {str(e)}"


# ==================== CHARTING FUNCTIONS ====================
class ChartBuilder:
    @staticmethod
    def create_candlestick_chart(data, indicators=None):
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price', 'Volume', 'RSI')
        )

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        ), row=1, col=1)

        if indicators and 'BB' in indicators:
            upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(data)
            fig.add_trace(go.Scatter(x=data.index, y=upper, name='BB Upper', line=dict(color='rgba(255,0,0,0.3)')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=middle, name='BB Middle', line=dict(color='rgba(255,0,0,0.5)')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=lower, name='BB Lower', line=dict(color='rgba(255,0,0,0.3)')), row=1, col=1)

        if indicators and 'SMA' in indicators:
            sma20 = TechnicalIndicators.calculate_sma(data, 20)
            sma50 = TechnicalIndicators.calculate_sma(data, 50)
            fig.add_trace(go.Scatter(x=data.index, y=sma20, name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=sma50, name='SMA 50', line=dict(color='blue')), row=1, col=1)

        if indicators and 'EMA' in indicators:
            ema12 = TechnicalIndicators.calculate_ema(data, 12)
            ema26 = TechnicalIndicators.calculate_ema(data, 26)
            fig.add_trace(go.Scatter(x=data.index, y=ema12, name='EMA 12', line=dict(color='purple')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=ema26, name='EMA 26', line=dict(color='green')), row=1, col=1)

        if indicators and 'VWAP' in indicators:
            vwap = TechnicalIndicators.calculate_vwap(data)
            fig.add_trace(go.Scatter(x=data.index, y=vwap, name='VWAP', line=dict(color='cyan')), row=1, col=1)

        if indicators and 'Ichimoku' in indicators:
            tenkan, kijun, senkou_a, senkou_b, chikou = TechnicalIndicators.calculate_ichimoku(data)
            fig.add_trace(go.Scatter(x=data.index, y=tenkan, name='Tenkan-sen', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=kijun, name='Kijun-sen', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=senkou_a, name='Senkou A', line=dict(color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=senkou_b, name='Senkou B', line=dict(color='orange')), row=1, col=1)

        if indicators and 'Parabolic SAR' in indicators:
            psar = TechnicalIndicators.calculate_parabolic_sar(data)
            fig.add_trace(go.Scatter(x=data.index, y=psar, name='Parabolic SAR', 
                                     mode='markers', marker=dict(size=3, color='purple')), row=1, col=1)

        colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors), row=2, col=1)

        rsi = TechnicalIndicators.calculate_rsi(data)
        fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(
            title='Stock Price Analysis',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_forecast_chart(data, forecast_df, model_name):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name='Historical',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Forecast'],
            name=f'{model_name} Forecast',
            line=dict(color='red', dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
            y=forecast_df['Upper'].tolist() + forecast_df['Lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))

        fig.update_layout(
            title=f'Price Forecast - {model_name}',
            xaxis_title='Date',
            yaxis_title='Price',
            height=500,
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_macd_chart(data):
        macd, signal, hist = TechnicalIndicators.calculate_macd(data)

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=data.index, y=macd, name='MACD', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=signal, name='Signal', line=dict(color='red')))

        colors = ['green' if h >= 0 else 'red' for h in hist]
        fig.add_trace(go.Bar(x=data.index, y=hist, name='Histogram', marker_color=colors))

        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(title='MACD', height=300, template='plotly_white')
        return fig

    @staticmethod
    def create_stochastic_chart(data):
        k, d = TechnicalIndicators.calculate_stochastic(data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=k, name='%K', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=d, name='%D', line=dict(color='red')))
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        fig.add_hline(y=20, line_dash="dash", line_color="green")
        fig.update_layout(title='Stochastic Oscillator', height=300, template='plotly_white')
        return fig

    @staticmethod
    def create_adx_chart(data):
        adx, plus_di, minus_di = TechnicalIndicators.calculate_adx(data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=adx, name='ADX', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=data.index, y=plus_di, name='+DI', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data.index, y=minus_di, name='-DI', line=dict(color='red')))
        fig.add_hline(y=25, line_dash="dash", line_color="gray")
        fig.update_layout(title='ADX', height=300, template='plotly_white')
        return fig


# ==================== MAIN APP ====================
def main():
    st.markdown('<div class="main-header">📈 Stock Forecast & Technical Analysis</div>', unsafe_allow_html=True)

    st.sidebar.header("⚙️ Configuration")

    stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()

    period = st.sidebar.selectbox(
        "Select Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )

    interval = st.sidebar.selectbox(
        "Select Interval",
        ["1d", "1wk", "1mo"],
        index=0
    )

    st.sidebar.header("📊 Technical Indicators")
    show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
    show_sma = st.sidebar.checkbox("SMA (20, 50)", value=True)
    show_ema = st.sidebar.checkbox("EMA (12, 26)", value=False)
    show_vwap = st.sidebar.checkbox("VWAP", value=False)
    show_ichimoku = st.sidebar.checkbox("Ichimoku Cloud", value=False)
    show_psar = st.sidebar.checkbox("Parabolic SAR", value=False)

    st.sidebar.header("🔮 Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)
    forecast_model = st.sidebar.selectbox(
        "Forecast Model",
        ["Prophet", "ARIMA", "Moving Average + Trend", "Monte Carlo Simulation"]
    )

    if st.sidebar.button("🚀 Analyze Stock", type="primary"):
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            try:
                ticker = yf.Ticker(stock_symbol)
                data = ticker.history(period=period, interval=interval)
                info = ticker.info

                if data.empty:
                    st.error(f"No data found for {stock_symbol}. Please check the symbol.")
                    return

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Company", info.get('shortName', stock_symbol))
                with col2:
                    st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                with col3:
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                    st.metric("Daily Change", f"{change:.2f}%", delta=f"{change:.2f}%")
                with col4:
                    st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")

                st.subheader("📈 Key Statistics")
                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
                with mcol1:
                    st.metric("52W High", f"${data['High'].max():.2f}")
                with mcol2:
                    st.metric("52W Low", f"${data['Low'].min():.2f}")
                with mcol3:
                    st.metric("Avg Volume", f"{data['Volume'].mean():,.0f}")
                with mcol4:
                    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                    st.metric("Volatility", f"{volatility:.1f}%")
                with mcol5:
                    returns = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    st.metric("Period Return", f"{returns:.1f}%")

                st.subheader("📊 Price Chart with Indicators")
                indicators = []
                if show_bb: indicators.append('BB')
                if show_sma: indicators.append('SMA')
                if show_ema: indicators.append('EMA')
                if show_vwap: indicators.append('VWAP')
                if show_ichimoku: indicators.append('Ichimoku')
                if show_psar: indicators.append('Parabolic SAR')

                main_chart = ChartBuilder.create_candlestick_chart(data, indicators)
                st.plotly_chart(main_chart, use_container_width=True)

                st.subheader("📉 Additional Indicators")
                c1, c2 = st.columns(2)
                with c1:
                    macd_chart = ChartBuilder.create_macd_chart(data)
                    st.plotly_chart(macd_chart, use_container_width=True)
                with c2:
                    stoch_chart = ChartBuilder.create_stochastic_chart(data)
                    st.plotly_chart(stoch_chart, use_container_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    adx_chart = ChartBuilder.create_adx_chart(data)
                    st.plotly_chart(adx_chart, use_container_width=True)
                with c4:
                    mfi = TechnicalIndicators.calculate_mfi(data)
                    fig_mfi = go.Figure()
                    fig_mfi.add_trace(go.Scatter(x=data.index, y=mfi, name='MFI', line=dict(color='orange')))
                    fig_mfi.add_hline(y=80, line_dash="dash", line_color="red")
                    fig_mfi.add_hline(y=20, line_dash="dash", line_color="green")
                    fig_mfi.update_layout(title='Money Flow Index (MFI)', height=300, template='plotly_white')
                    st.plotly_chart(fig_mfi, use_container_width=True)

                st.subheader("📏 Fibonacci Retracement Levels")
                fib_levels = TechnicalIndicators.calculate_fibonacci_retracement(data)
                fib_df = pd.DataFrame(list(fib_levels.items()), columns=['Level', 'Price'])
                st.dataframe(fib_df, use_container_width=True)

                st.markdown('<div class="forecast-header">🔮 Price Forecast</div>', unsafe_allow_html=True)

                with st.spinner(f"Running {forecast_model} forecast..."):
                    if forecast_model == "Prophet":
                        forecast_df, msg = Forecasting.prophet_forecast(data, forecast_days)
                    elif forecast_model == "ARIMA":
                        forecast_df, msg = Forecasting.arima_forecast(data, forecast_days)
                    elif forecast_model == "Moving Average + Trend":
                        forecast_df, msg = Forecasting.moving_average_forecast(data, forecast_days)
                    else:
                        forecast_df, msg = Forecasting.monte_carlo_simulation(data, forecast_days)

                    if forecast_df is not None:
                        forecast_chart = ChartBuilder.create_forecast_chart(data, forecast_df, forecast_model)
                        st.plotly_chart(forecast_chart, use_container_width=True)

                        st.subheader("📋 Forecast Details")
                        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(forecast_df, use_container_width=True)

                        last_price = data['Close'].iloc[-1]
                        predicted_price = forecast_df['Forecast'].iloc[-1]
                        predicted_change = ((predicted_price - last_price) / last_price) * 100

                        fcol1, fcol2, fcol3 = st.columns(3)
                        with fcol1:
                            st.metric("Current Price", f"${last_price:.2f}")
                        with fcol2:
                            st.metric(f"Predicted Price ({forecast_days}d)", f"${predicted_price:.2f}")
                        with fcol3:
                            st.metric("Predicted Change", f"{predicted_change:.2f}%", 
                                    delta=f"{predicted_change:.2f}%")
                    else:
                        st.warning(msg)

                with st.expander("📋 View Raw Data"):
                    st.dataframe(data.tail(50), use_container_width=True)

                csv = data.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f'{stock_symbol}_data.csv',
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check the stock symbol and try again.")

    st.markdown("---")
    st.markdown("<center>Built with ❤️ using Streamlit, Plotly, and Yahoo Finance</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
