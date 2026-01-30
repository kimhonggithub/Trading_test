import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from binance.client import Client
from sklearn.linear_model import LinearRegression
from PIL import Image

# Initialize Binance Client
API_KEY = 'XHZeiuSX08HCZ76Ky9MVedZLnQArQLb0j6BDSf5g2WsBAHY16tk6Gx8IjwuF2RSN'
API_SECRET = 'N0h480KvxsrbKNCrTnglEtDL13RynQnZq3GuGhfHzCoGUGnTZtEFcOIjgPGODJoi'

# Initialize Binance Client
client = Client(API_KEY, API_SECRET, tld='us')
# Full Pair List
crypto_pairs = [
    'BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','SOLUSDT',
    'ADAUSDT','DOGEUSDT','AVAXUSDT','LINKUSDT',
    'DOTUSDT','LTCUSDT','SHIBUSDT','UNIUSDT'
]

# Function to fetch data from Binance
def get_binance_data(symbol, interval, lookback):
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['open'] = pd.to_numeric(df['open'])
    return df[['timestamp', 'open', 'high', 'low', 'close']]

# Function to calculate correlations and sensitivity
def calculate_correlation(data, base_symbol='BTCUSDT'):
    base_data = data[base_symbol].set_index('timestamp')['close']
    correlation_results = []
    
    for pair, df in data.items():
        if pair != base_symbol:
            target_data = df.set_index('timestamp')['close']
            
            # Calculate percentage changes
            btc_pct_change = base_data.pct_change().dropna()
            target_pct_change = target_data.pct_change().dropna()
            
            # Align indices
            aligned_data = pd.concat([btc_pct_change, target_pct_change], axis=1).dropna()
            aligned_data.columns = ['BTC_pct_change', 'Target_pct_change']
            
            # Correlation Coefficient
            correlation = aligned_data.corr().iloc[0, 1]
            
            # Regression for sensitivity (percentage change relationship)
            X = aligned_data['BTC_pct_change'].values.reshape(-1, 1)
            y = aligned_data['Target_pct_change'].values
            reg = LinearRegression()
            reg.fit(X, y)
            sensitivity = reg.coef_[0]  # Beta coefficient
            
            # Append results
            correlation_results.append({
                'Pair': pair,
                'Correlation Coefficient': correlation,
                'Sensitivity (% Change)': sensitivity
            })

    # Create DataFrame
    correlation_df = pd.DataFrame(correlation_results)
    return correlation_df.sort_values(by='Correlation Coefficient', ascending=False).reset_index(drop=True)

# Function to predict value based on BTC price
def predict_pair_value(data, base_symbol='BTCUSDT', target_symbol='ETHUSDT', target_btc_price=150000):
    base_data = data[base_symbol][['timestamp', 'close']].rename(columns={'close': 'close_base'})
    target_data = data[target_symbol][['timestamp', 'close']].rename(columns={'close': 'close_target'})
    merged = pd.merge(base_data, target_data, on='timestamp')

    X = merged['close_base'].values.reshape(-1, 1)
    y = merged['close_target'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    return model.predict([[target_btc_price]])[0][0]

def predict_trend(df, future_steps=12):
    """
    Simple linear trend prediction
    """
    df = df.copy()
    df['index'] = range(len(df))

    X = df[['index']]
    y = df['close']

    model = LinearRegression()
    model.fit(X, y)

    future_index = range(len(df), len(df) + future_steps)
    future_X = pd.DataFrame({'index': future_index})
    future_prices = model.predict(future_X)

    future_timestamps = pd.date_range(
        start=df['timestamp'].iloc[-1],
        periods=future_steps + 1,
        freq='H'
    )[1:]

    return pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_price': future_prices
    })



# Streamlit layout configuration
st.set_page_config(layout="wide")

eth_df = st.session_state.crypto_data['ETHUSDT']
prediction_df = predict_trend(eth_df, future_steps=12)


# Sidebar
image1 = Image.open("Pic1.png")
st.sidebar.image(image1, use_column_width=True)

st.sidebar.header("Data Parameters")
interval = st.sidebar.selectbox("Select Interval", ['1m', '5m', '15m', '1h', '4h', '1d'], index=3)
lookback_days = st.sidebar.slider("Select Lookback Period (Days)", min_value=1, max_value=30, value=7)  # Slider for lookback
lookback = f"{lookback_days} days ago UTC"  # Convert slider value to Binance-compatible format
get_data_button = st.sidebar.button("Get Data")

st.sidebar.header("Prediction Inputs")
target_symbol = st.sidebar.selectbox("Select Target Symbol", crypto_pairs, index=0)  # Default ETHUSDT
btc_price_input = st.sidebar.number_input("Enter BTC Price ($)", min_value=0, step=1000, value=150000)
predict_button = st.sidebar.button("Predict")

# Data fetch and correlation calculation
if get_data_button:
    crypto_data = {pair: get_binance_data(pair, interval, lookback) for pair in crypto_pairs}
    correlation_df = calculate_correlation(crypto_data)
    st.session_state.crypto_data = crypto_data
    st.session_state.correlation_df = correlation_df

# Main Page
image2 = Image.open("Pic2.png")
st.image(image2, use_column_width=True)

st.markdown("""
<style>
    .block-container {padding-top: 0 !important;}
    .card {border-radius: 10px; padding: 20px; text-align: center; font-size: 18px;}
    h1, h2, h3, .subheader {text-align: center !important;}
</style>
""", unsafe_allow_html=True)

st.title("Crypto Strength Analysis, Correlations and Predictions")

# Cards on top
if "crypto_data" in st.session_state:
    btc_latest_price = st.session_state.crypto_data['BTCUSDT']['close'].iloc[-1]
    target_latest_price = st.session_state.crypto_data[target_symbol]['close'].iloc[-1]
    predicted_price = None

    if predict_button:
        predicted_price = predict_pair_value(st.session_state.crypto_data, 'BTCUSDT', target_symbol, btc_price_input)

    col1, col2, col3 = st.columns(3)

    # BTC Latest Price Card
    col1.markdown(f"""<div class="card" style="background-color: #d4edda; color:black">BTC Price<br><b>${btc_latest_price:,.2f}</b></div>""", unsafe_allow_html=True)
    
    # Target Symbol Latest Price Card
    col2.markdown(f"""<div class="card" style="background-color: #f8f9fa;  color:black">{target_symbol} Price<br><b>${target_latest_price:,.2f}</b></div>""", unsafe_allow_html=True)
    
    # Predicted Price Card (Handle NoneType)
    if predicted_price is not None:
        col3.markdown(f"""<div class="card" style="background-color: #f8f9fa;  color:black">Predicted Price<br><b>${predicted_price:,.2f}</b></div>""", unsafe_allow_html=True)
    else:
        col3.markdown(f"""<div class="card" style="background-color: #f8f9fa;  color:black">Predicted Price<br><b>Not Available</b></div>""", unsafe_allow_html=True)
        
# Section 1: Correlation and BTC Candlestick
if "crypto_data" in st.session_state:
    col1, col2 = st.columns(2)
    col1.subheader("Correlation and Sensitivity Data")
    col1.dataframe(st.session_state.correlation_df, use_container_width=True)

    col2.subheader("BTC Candlestick Chart")
    btc_fig = go.Figure(data=[
        go.Candlestick(
            x=st.session_state.crypto_data['BTCUSDT']['timestamp'],
            open=st.session_state.crypto_data['BTCUSDT']['open'],
            high=st.session_state.crypto_data['BTCUSDT']['high'],
            low=st.session_state.crypto_data['BTCUSDT']['low'],
            close=st.session_state.crypto_data['BTCUSDT']['close']
        )
    ])
    btc_fig.update_layout(xaxis_title=None, yaxis_title=None, template="plotly_white")
    col2.plotly_chart(btc_fig, use_container_width=True)



fig = go.Figure()

# Real price
fig.add_trace(go.Scatter(
    x=eth_df['timestamp'],
    y=eth_df['close'],
    mode='lines',
    name='ETH Price'
))

# Prediction
fig.add_trace(go.Scatter(
    x=prediction_df['timestamp'],
    y=prediction_df['predicted_price'],
    mode='lines',
    line=dict(dash='dash'),
    name='Predicted Trend'
))

fig.update_layout(
    title='ETHUSDT Price Prediction (Next 12 Hours)',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)


# Section 2: Scatter Plot and Target Candlestick
if predict_button and "crypto_data" in st.session_state:
    col1, col2 = st.columns(2)

    col1.subheader(f"Scatter Plot: BTC vs {target_symbol}")
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=st.session_state.crypto_data['BTCUSDT']['close'],
        y=st.session_state.crypto_data[target_symbol]['close'],
        mode='markers'
    ))
    scatter_fig.update_layout(xaxis_title=None, yaxis_title=None, template="plotly_white")
    col1.plotly_chart(scatter_fig, use_container_width=True)

    col2.subheader(f"{target_symbol} Candlestick Chart")
    target_fig = go.Figure(data=[
        go.Candlestick(
            x=st.session_state.crypto_data[target_symbol]['timestamp'],
            open=st.session_state.crypto_data[target_symbol]['open'],
            high=st.session_state.crypto_data[target_symbol]['high'],
            low=st.session_state.crypto_data[target_symbol]['low'],
            close=st.session_state.crypto_data[target_symbol]['close']
        )
    ])
    target_fig.update_layout(xaxis_title=None, yaxis_title=None, template="plotly_white")
    col2.plotly_chart(target_fig, use_container_width=True)

# Streamlit run Binance_Correlations.py
