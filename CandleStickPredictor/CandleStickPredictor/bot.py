import ccxt
import pandas as pd
import time
import yfinance as yf
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import webbrowser

# Binance API Credentials
API_KEY = 'althoUYXvlKUlBJ8NvUiM8RnH0UEA1mM56MdFAeB9JHewYvawZbcFkfajIwct1GN'
API_SECRET = 'v7wZcwfMZEV3HI3mrKDzFy8nnGdyUmK1G63XF0QarPvSDhZD5tUxx0jSkKAfiXIE'

# Trading Parameters
SYMBOL = 'LTC/BTC'     # For ccxt (crypto)
YF_SYMBOL = 'AAPL'      # For yfinance (stock)
TIMEFRAME = '1h'
SHORT_MA = 10
LONG_MA = 50
TRADE_AMOUNT = 1  # LTC amount
USE_YFINANCE = False  # Set to True to use Yahoo Finance instead of Binance

def init_exchange():
    return ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True,
            'recvWindow': 5000
        }
    })

def fetch_data_ccxt(exchange):
    bars = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LONG_MA + 2)
    df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df

def fetch_data_yfinance():
    interval_map = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '60m',
        '1d': '1d', '1wk': '1wk'
    }
    interval = interval_map.get(TIMEFRAME, '1h')
    data = yf.download(tickers=YF_SYMBOL, period='10d', interval=interval)
    df = data.reset_index()
    df.rename(columns={'Date': 'ts', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'vol'}, inplace=True)
    df['close'] = df['close'].astype(float)
    return df.tail(LONG_MA + 2)

def calculate_indicators(df):
    # Moving Averages
    df['ma_short'] = df['close'].rolling(SHORT_MA).mean()
    df['ma_long'] = df['close'].rolling(LONG_MA).mean()
    
    # Percentage Changes
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['ma_short_pct'] = df['ma_short'].pct_change() * 100
    df['ma_long_pct'] = df['ma_long'].pct_change() * 100
    
    # Volume Analysis
    df['volume_change_pct'] = df['vol'].pct_change() * 100
    
    return df

def create_candlestick_chart(df, signal=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3])

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df['ts'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(
        x=df['ts'],
        y=df['ma_short'],
        name=f'{SHORT_MA} MA',
        line=dict(color='blue')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['ts'],
        y=df['ma_long'],
        name=f'{LONG_MA} MA',
        line=dict(color='red')
    ), row=1, col=1)

    # Volume Chart
    fig.add_trace(go.Bar(
        x=df['ts'],
        y=df['vol'],
        name='Volume'
    ), row=2, col=1)

    # Add signal markers if there's a signal
    if signal:
        last_price = df['close'].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[df['ts'].iloc[-1]],
            y=[last_price],
            mode='markers',
            marker=dict(
                symbol='triangle-up' if signal == 'buy' else 'triangle-down',
                size=15,
                color='green' if signal == 'buy' else 'red'
            ),
            name=f'{signal.upper()} Signal'
        ), row=1, col=1)

    # Update layout
    fig.update_layout(
        title=f'{SYMBOL} Price Chart with {SHORT_MA}/{LONG_MA} Moving Averages',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800
    )

    # Save and open the chart
    chart_file = 'trading_chart.html'
    fig.write_html(chart_file)
    webbrowser.open('file://' + os.path.realpath(chart_file))

def apply_strategy(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Calculate signal
    signal = None
    if prev['ma_short'] <= prev['ma_long'] and latest['ma_short'] > latest['ma_long']:
        signal = 'buy'
    elif prev['ma_short'] >= prev['ma_long'] and latest['ma_short'] < latest['ma_long']:
        signal = 'sell'
    
    # Print analysis
    print("\n=== Market Analysis ===")
    print(f"Current Price: {latest['close']:.8f}")
    print(f"Price Change: {latest['price_change_pct']:.2f}%")
    print(f"Volume Change: {latest['volume_change_pct']:.2f}%")
    print(f"Short MA: {latest['ma_short']:.8f} ({latest['ma_short_pct']:.2f}%)")
    print(f"Long MA: {latest['ma_long']:.8f} ({latest['ma_long_pct']:.2f}%)")
    
    return signal

def execute_trade(exchange, signal):
    if signal == 'buy':
        print("Placing buy order...")
        return exchange.create_limit_buy_order(
            SYMBOL,
            TRADE_AMOUNT,
            0.1,  # price
            {'timeInForce': 'GTC'}
        )
    if signal == 'sell':
        print("Placing sell order...")
        return exchange.create_limit_sell_order(
            SYMBOL,
            TRADE_AMOUNT,
            0.1,  # price
            {'timeInForce': 'GTC'}
        )
    return None

def main():
    exchange = init_exchange()
    while True:
        try:
            if USE_YFINANCE:
                df = fetch_data_yfinance()
            else:
                df = fetch_data_ccxt(exchange)

            df = calculate_indicators(df)
            signal = apply_strategy(df)
            
            # Create and display the chart
            create_candlestick_chart(df, signal)
            
            if signal:
                if not USE_YFINANCE:
                    order = execute_trade(exchange, signal)
                    print("Order executed:", order)
                else:
                    print(f"Signal: {signal} for {YF_SYMBOL} (no trade executed)")
            else:
                print("No action:", time.ctime())
                
            time.sleep(60 * 60)  # wait one hour
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            time.sleep(60)  # wait a minute before retrying

if __name__ == "__main__":
    main()
