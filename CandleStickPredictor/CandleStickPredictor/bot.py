import ccxt
import pandas as pd
import time
import yfinance as yf
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import webbrowser
import json
from flask import Flask, render_template_string
import threading
import numpy as np

# Initialize Flask app
app = Flask(__name__)

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

# Rate limiting parameters
UPDATE_INTERVAL = 300  # 5 minutes between updates
MAX_RETRIES = 3
RETRY_DELAY = 60  # 1 minute between retries

# Global variables for chart data
current_chart_data = None
current_analysis = None
last_update_time = 0

def identify_candlestick_patterns(df):
    patterns = []
    signals = []
    
    # Get the last 3 candles for pattern analysis
    last_3 = df.tail(3)
    
    # Doji
    def is_doji(row):
        body = abs(row['close'] - row['open'])
        total = row['high'] - row['low']
        return body <= (total * 0.1)  # Body is less than 10% of total range
    
    # Hammer
    def is_hammer(row):
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        return (lower_wick > (body * 2)) and (upper_wick < body)
    
    # Engulfing
    def is_bullish_engulfing(row1, row2):
        return (row1['close'] < row1['open'] and  # First candle is bearish
                row2['close'] > row2['open'] and  # Second candle is bullish
                row2['open'] < row1['close'] and  # Second opens below first's close
                row2['close'] > row1['open'])     # Second closes above first's open
    
    def is_bearish_engulfing(row1, row2):
        return (row1['close'] > row1['open'] and  # First candle is bullish
                row2['close'] < row2['open'] and  # Second candle is bearish
                row2['open'] > row1['close'] and  # Second opens above first's close
                row2['close'] < row1['open'])     # Second closes below first's open
    
    # Check patterns
    if is_doji(last_3.iloc[-1]):
        patterns.append("Doji")
        signals.append("Neutral - Potential trend reversal")
    
    if is_hammer(last_3.iloc[-1]):
        patterns.append("Hammer")
        signals.append("Bullish - Potential reversal")
    
    if len(last_3) >= 2:
        if is_bullish_engulfing(last_3.iloc[-2], last_3.iloc[-1]):
            patterns.append("Bullish Engulfing")
            signals.append("Strong Buy Signal")
        elif is_bearish_engulfing(last_3.iloc[-2], last_3.iloc[-1]):
            patterns.append("Bearish Engulfing")
            signals.append("Strong Sell Signal")
    
    return patterns, signals

def calculate_trend_strength(df):
    # Calculate ADX (Average Directional Index)
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(period).mean()
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1]
    
    # Calculate RSI
    def calculate_rsi(close, period=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    adx = calculate_adx(df['high'], df['low'], df['close'])
    rsi = calculate_rsi(df['close']).iloc[-1]
    
    # Determine trend strength
    if adx > 25:
        trend_strength = "Strong"
    elif adx > 20:
        trend_strength = "Moderate"
    else:
        trend_strength = "Weak"
    
    # Determine overbought/oversold
    if rsi > 70:
        rsi_signal = "Overbought"
    elif rsi < 30:
        rsi_signal = "Oversold"
    else:
        rsi_signal = "Neutral"
    
    return {
        'trend_strength': trend_strength,
        'adx': round(adx, 2),
        'rsi': round(rsi, 2),
        'rsi_signal': rsi_signal
    }

def generate_prediction(df, patterns, signals, trend_analysis):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Price movement
    price_change = ((latest['close'] - prev['close']) / prev['close']) * 100
    
    # Volume analysis
    volume_change = ((latest['vol'] - prev['vol']) / prev['vol']) * 100
    
    # Generate prediction
    prediction = {
        'price_movement': f"{price_change:.2f}%",
        'volume_movement': f"{volume_change:.2f}%",
        'patterns_detected': patterns,
        'signals': signals,
        'trend_analysis': trend_analysis,
        'recommendation': 'HOLD'  # Default recommendation
    }
    
    # Generate trading recommendation
    if trend_analysis['trend_strength'] == 'Strong':
        if 'Bullish Engulfing' in patterns or 'Hammer' in patterns:
            if trend_analysis['rsi_signal'] != 'Overbought':
                prediction['recommendation'] = 'STRONG BUY'
        elif 'Bearish Engulfing' in patterns:
            if trend_analysis['rsi_signal'] != 'Oversold':
                prediction['recommendation'] = 'STRONG SELL'
    elif trend_analysis['trend_strength'] == 'Moderate':
        if 'Bullish Engulfing' in patterns:
            prediction['recommendation'] = 'BUY'
        elif 'Bearish Engulfing' in patterns:
            prediction['recommendation'] = 'SELL'
    
    # Add explanation
    prediction['explanation'] = f"""
    Current market analysis shows a {trend_analysis['trend_strength'].lower()} trend (ADX: {trend_analysis['adx']}).
    RSI is at {trend_analysis['rsi']} indicating {trend_analysis['rsi_signal'].lower()} conditions.
    Price has moved {price_change:.2f}% with volume change of {volume_change:.2f}%.
    Detected patterns: {', '.join(patterns) if patterns else 'None'}
    Trading signals: {', '.join(signals) if signals else 'None'}
    """
    
    return prediction

# HTML template for the live chart
CHART_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Trading Chart</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <meta http-equiv="refresh" content="300">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px;
            background-color: #f0f2f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .analysis-box { 
            background-color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .analysis-box:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }
        .signal { 
            font-weight: bold; 
            color: #2c3e50;
            cursor: pointer;
        }
        .buy { color: #27ae60; }
        .sell { color: #c0392b; }
        .hold { color: #7f8c8d; }
        .expandable {
            cursor: pointer;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 5px 0;
        }
        .expandable:hover {
            background-color: #f8f9fa;
        }
        .details {
            display: none;
            padding: 10px;
            margin-top: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .chart-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .pattern-info {
            margin-top: 10px;
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: 4px;
        }
        .indicator-value {
            font-weight: bold;
            color: #2c3e50;
        }
        .update-time {
            text-align: right;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="chart-container">
            <div id="chart"></div>
        </div>
        
        <div class="analysis-box">
            <h2>Market Analysis</h2>
            <div id="analysis"></div>
        </div>
        
        <div class="analysis-box">
            <h2>Trading Signals</h2>
            <div id="signals"></div>
        </div>
        
        <div class="analysis-box">
            <h2>Prediction</h2>
            <div id="prediction"></div>
        </div>
        
        <div class="analysis-box">
            <h2>Pattern Analysis</h2>
            <div id="patterns"></div>
        </div>
        
        <div class="update-time" id="lastUpdate"></div>
    </div>

    <script>
        var chartData = {{ chart_data | safe }};
        var analysis = {{ analysis | safe }};
        var prediction = {{ prediction | safe }};
        var lastUpdate = "{{ last_update }}";
        
        // Initialize the chart with interactive features
        Plotly.newPlot('chart', chartData.data, chartData.layout, {
            displayModeBar: true,
            modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape'],
            displaylogo: false
        });
        
        // Market Analysis Section
        var analysisDiv = document.getElementById('analysis');
        analysisDiv.innerHTML = `
            <div class="expandable" onclick="toggleDetails('price-details')">
                <h3>Price Information</h3>
                <p>Current Price: <span class="indicator-value">${analysis.current_price}</span></p>
                <p>Price Change: <span class="indicator-value">${analysis.price_change}</span></p>
            </div>
            <div id="price-details" class="details">
                <p>24h High: ${analysis.high_24h || 'N/A'}</p>
                <p>24h Low: ${analysis.low_24h || 'N/A'}</p>
                <p>Volume: ${analysis.volume || 'N/A'}</p>
            </div>
            
            <div class="expandable" onclick="toggleDetails('ma-details')">
                <h3>Moving Averages</h3>
                <p>Short MA: <span class="indicator-value">${analysis.short_ma}</span></p>
                <p>Long MA: <span class="indicator-value">${analysis.long_ma}</span></p>
            </div>
            <div id="ma-details" class="details">
                <p>MA Crossover: ${analysis.ma_crossover || 'None'}</p>
                <p>MA Trend: ${analysis.ma_trend || 'Neutral'}</p>
            </div>
        `;
        
        // Trading Signals Section
        var signalsDiv = document.getElementById('signals');
        signalsDiv.innerHTML = `
            <div class="expandable" onclick="toggleDetails('trend-details')">
                <h3>Trend Analysis</h3>
                <p class="signal">Trend Strength: <span class="indicator-value">${prediction.trend_analysis.trend_strength}</span></p>
                <p>ADX: <span class="indicator-value">${prediction.trend_analysis.adx}</span></p>
            </div>
            <div id="trend-details" class="details">
                <p>ADX Interpretation: ${getADXInterpretation(prediction.trend_analysis.adx)}</p>
                <p>Trend Direction: ${getTrendDirection(prediction.trend_analysis)}</p>
            </div>
            
            <div class="expandable" onclick="toggleDetails('rsi-details')">
                <h3>RSI Analysis</h3>
                <p>RSI: <span class="indicator-value">${prediction.trend_analysis.rsi}</span></p>
                <p>Signal: <span class="indicator-value">${prediction.trend_analysis.rsi_signal}</span></p>
            </div>
            <div id="rsi-details" class="details">
                <p>RSI Interpretation: ${getRSIInterpretation(prediction.trend_analysis.rsi)}</p>
                <p>Potential Action: ${getRSIAction(prediction.trend_analysis.rsi)}</p>
            </div>
        `;
        
        // Prediction Section
        var predictionDiv = document.getElementById('prediction');
        predictionDiv.innerHTML = `
            <div class="expandable" onclick="toggleDetails('recommendation-details')">
                <h3 class="signal ${prediction.recommendation.toLowerCase().replace(' ', '')}">
                    Recommendation: ${prediction.recommendation}
                </h3>
            </div>
            <div id="recommendation-details" class="details">
                <p>${prediction.explanation}</p>
                <div class="pattern-info">
                    <h4>Pattern Analysis:</h4>
                    <p>${getPatternAnalysis(prediction.patterns_detected)}</p>
                </div>
            </div>
        `;
        
        // Pattern Analysis Section
        var patternsDiv = document.getElementById('patterns');
        patternsDiv.innerHTML = `
            <div class="expandable" onclick="toggleDetails('pattern-details')">
                <h3>Detected Patterns</h3>
                <p>Patterns: ${prediction.patterns_detected.length ? prediction.patterns_detected.join(', ') : 'None'}</p>
            </div>
            <div id="pattern-details" class="details">
                ${getDetailedPatternInfo(prediction.patterns_detected)}
            </div>
        `;
        
        document.getElementById('lastUpdate').innerHTML = `Last Update: ${lastUpdate}`;
        
        // Helper functions
        function toggleDetails(id) {
            var details = document.getElementById(id);
            if (details.style.display === "block") {
                details.style.display = "none";
            } else {
                details.style.display = "block";
            }
        }
        
        function getADXInterpretation(adx) {
            if (adx > 25) return "Strong trend - Good for trend following strategies";
            if (adx > 20) return "Moderate trend - Consider trend following with caution";
            return "Weak trend - Consider range trading strategies";
        }
        
        function getTrendDirection(analysis) {
            if (analysis.adx > 25) {
                return analysis.rsi > 50 ? "Upward" : "Downward";
            }
            return "Sideways";
        }
        
        function getRSIInterpretation(rsi) {
            if (rsi > 70) return "Overbought - Potential reversal or correction";
            if (rsi < 30) return "Oversold - Potential reversal or bounce";
            return "Neutral - No clear signal";
        }
        
        function getRSIAction(rsi) {
            if (rsi > 70) return "Consider taking profits or short positions";
            if (rsi < 30) return "Consider buying opportunities";
            return "Monitor for clearer signals";
        }
        
        function getPatternAnalysis(patterns) {
            if (!patterns.length) return "No significant patterns detected";
            return patterns.map(pattern => {
                switch(pattern) {
                    case "Doji": return "Doji indicates market indecision and potential trend reversal";
                    case "Hammer": return "Hammer suggests potential bullish reversal after downtrend";
                    case "Bullish Engulfing": return "Bullish Engulfing shows strong buying pressure";
                    case "Bearish Engulfing": return "Bearish Engulfing shows strong selling pressure";
                    default: return pattern;
                }
            }).join("<br>");
        }
        
        function getDetailedPatternInfo(patterns) {
            if (!patterns.length) return "No patterns to analyze";
            return patterns.map(pattern => {
                let info = "";
                switch(pattern) {
                    case "Doji":
                        info = "A Doji represents indecision in the market. The opening and closing prices are virtually equal, creating a cross-like pattern. This often indicates a potential trend reversal.";
                        break;
                    case "Hammer":
                        info = "A Hammer is a bullish reversal pattern that forms after a downtrend. It has a small body at the top and a long lower shadow, suggesting that sellers pushed the price down but buyers were able to push it back up.";
                        break;
                    case "Bullish Engulfing":
                        info = "A Bullish Engulfing pattern occurs when a small bearish candle is followed by a larger bullish candle that completely engulfs the previous candle. This indicates strong buying pressure and potential trend reversal.";
                        break;
                    case "Bearish Engulfing":
                        info = "A Bearish Engulfing pattern occurs when a small bullish candle is followed by a larger bearish candle that completely engulfs the previous candle. This indicates strong selling pressure and potential trend reversal.";
                        break;
                }
                return `<div class="pattern-info">
                    <h4>${pattern}</h4>
                    <p>${info}</p>
                </div>`;
            }).join("");
        }
    </script>
</body>
</html>
'''

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

def fetch_data_with_retry(exchange, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            if USE_YFINANCE:
                return fetch_data_yfinance()
            else:
                return fetch_data_ccxt(exchange)
        except ccxt.RateLimitExceeded:
            if attempt < max_retries - 1:
                print(f"Rate limit exceeded. Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            else:
                raise
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise

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

def create_chart_data(df, signal=None):
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

    return fig.to_json()

def create_analysis_data(df):
    latest = df.iloc[-1]
    return {
        'current_price': f"{latest['close']:.8f}",
        'price_change': f"{latest['price_change_pct']:.2f}%",
        'volume_change': f"{latest['volume_change_pct']:.2f}%",
        'short_ma': f"{latest['ma_short']:.8f} ({latest['ma_short_pct']:.2f}%)",
        'long_ma': f"{latest['ma_long']:.8f} ({latest['ma_long_pct']:.2f}%)"
    }

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

@app.route('/')
def index():
    global current_chart_data, current_analysis, last_update_time
    if current_chart_data is None or current_analysis is None:
        return "Waiting for data..."
    return render_template_string(CHART_TEMPLATE, 
                                chart_data=current_chart_data,
                                analysis=current_analysis,
                                last_update=datetime.fromtimestamp(last_update_time).strftime('%Y-%m-%d %H:%M:%S'))

def run_flask():
    app.run(host='0.0.0.0', port=5000)

def main():
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    print("Flask server started at http://localhost:5000")
    print("Open this URL in your browser to see live updates")
    print(f"Updates will occur every {UPDATE_INTERVAL} seconds")

    exchange = init_exchange()
    while True:
        try:
            # Fetch and process data
            df = fetch_data_with_retry(exchange)
            df = calculate_indicators(df)
            
            # Analyze patterns and trends
            patterns, signals = identify_candlestick_patterns(df)
            trend_analysis = calculate_trend_strength(df)
            prediction = generate_prediction(df, patterns, signals, trend_analysis)
            
            # Update global variables for the web interface
            global current_chart_data, current_analysis, last_update_time
            current_chart_data = create_chart_data(df, prediction['recommendation'].split()[0].lower())
            current_analysis = create_analysis_data(df)
            last_update_time = time.time()
            
            # Execute trades based on strong signals
            if prediction['recommendation'] in ['STRONG BUY', 'STRONG SELL']:
                signal = 'buy' if prediction['recommendation'] == 'STRONG BUY' else 'sell'
                if not USE_YFINANCE:
                    order = execute_trade(exchange, signal)
                    print("Order executed:", order)
                else:
                    print(f"Signal: {signal} for {YF_SYMBOL} (no trade executed)")
            else:
                print("No action:", time.ctime())
                
            print(f"Next update in {UPDATE_INTERVAL} seconds...")
            time.sleep(UPDATE_INTERVAL)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    main()
