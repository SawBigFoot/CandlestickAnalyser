import ccxt
import pandas as pd
import time
import yfinance as yf  # <--- Added yfinance
import os

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
    df['close'] = df['close'].astype(float)
    return df

def fetch_data_yfinance():
    # Map your timeframe to yfinance interval:
    interval_map = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '60m',
        '1d': '1d', '1wk': '1wk'
    }
    interval = interval_map.get(TIMEFRAME, '1h')

    data = yf.download(tickers=YF_SYMBOL, period='10d', interval=interval)
    df = data.reset_index()
    df.rename(columns={'Date': 'ts', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'vol'}, inplace=True)
    df['close'] = df['close'].astype(float)
    return df.tail(LONG_MA + 2)  # Keep same amount of data points

def apply_strategy(df):
    df['ma_short'] = df['close'].rolling(SHORT_MA).mean()
    df['ma_long'] = df['close'].rolling(LONG_MA).mean()
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if prev['ma_short'] <= prev['ma_long'] and latest['ma_short'] > latest['ma_long']:
        return 'buy'
    if prev['ma_short'] >= prev['ma_long'] and latest['ma_short'] < latest['ma_long']:
        return 'sell'
    return None

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
        if USE_YFINANCE:
            df = fetch_data_yfinance()
        else:
            df = fetch_data_ccxt(exchange)

        signal = apply_strategy(df)
        if signal:
            if not USE_YFINANCE:
                order = execute_trade(exchange, signal)
                print("Order executed:", order)
            else:
                # For yfinance stocks, no direct trading via this script (just print)
                print(f"Signal: {signal} for {YF_SYMBOL} (no trade executed)")
        else:
            print("No action:", time.ctime())
        time.sleep(60 * 60)  # wait one hour

if __name__ == "__main__":
    main()
