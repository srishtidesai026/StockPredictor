from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Helper function: Fetch data from Yahoo Finance
def fetch_data(ticker, interval, period):
    try:
        data = yf.download(ticker, interval=interval, period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Helper function: Calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Intraday Strategy
def intraday_strategy(data):
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Signal'] = np.where(data['SMA_5'] > data['SMA_20'], 1, 0)
    data['Position'] = data['Signal'].diff()
    data['Buy_Signal'] = data['Position'] == 1
    data['Sell_Signal'] = data['Position'] == -1
    return data

# Swing Strategy
def swing_strategy(data):
    data = calculate_rsi(data)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['SMA_20'] + (2 * data['Close'].rolling(window=20).std())
    data['Lower_Band'] = data['SMA_20'] - (2 * data['Close'].rolling(window=20).std())
    data['Buy_Signal'] = (data['RSI'] < 30) & (data['Close'] <= data['Lower_Band'])
    data['Sell_Signal'] = (data['RSI'] > 70) & (data['Close'] >= data['Upper_Band'])
    return data

# Sentiment Analysis
def fetch_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    return np.mean(sentiments)

# Generate plot and return as base64 string
def plot_to_base64(data, strategy):
    plt.figure(figsize=(14, 7))
    if strategy == "swing":
        plt.plot(data['Close'], label='Close Price', alpha=0.6)
        plt.plot(data['Upper_Band'], label='Upper Band', linestyle='--', color='orange')
        plt.plot(data['Lower_Band'], label='Lower Band', linestyle='--', color='blue')
        plt.scatter(data.index[data['Buy_Signal']], data['Close'][data['Buy_Signal']],
                    label='Buy Signal', marker='^', color='green')
        plt.scatter(data.index[data['Sell_Signal']], data['Close'][data['Sell_Signal']],
                    label='Sell Signal', marker='v', color='red')
    elif strategy == "intraday":
        plt.plot(data['Close'], label='Close Price', alpha=0.6)
        plt.plot(data['SMA_5'], label='SMA 5', color='blue')
        plt.plot(data['SMA_20'], label='SMA 20', color='orange')
        plt.scatter(data.index[data['Buy_Signal']], data['Close'][data['Buy_Signal']],
                    label='Buy Signal', marker='^', color='green')
        plt.scatter(data.index[data['Sell_Signal']], data['Close'][data['Sell_Signal']],
                    label='Sell Signal', marker='v', color='red')
    plt.legend()
    plt.title(f"{strategy.capitalize()} Trading Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form['ticker']
    strategy = request.form['strategy']

    if strategy == 'intraday':
        data = fetch_data(ticker, interval='5m', period='5d')
        if data is None:
            return jsonify({"error": "No data found for the provided ticker."})
        data = intraday_strategy(data)
    elif strategy == 'swing':
        data = fetch_data(ticker, interval='1d', period='6mo')
        if data is None:
            return jsonify({"error": "No data found for the provided ticker."})
        data = swing_strategy(data)
    else:
        return jsonify({"error": "Invalid strategy selected."})

    # Generate plot
    plot_url = plot_to_base64(data, strategy)
    
    # Example news headlines for sentiment analysis
    news_headlines = [
        "Markets rally as inflation shows signs of slowing",
        "Tech stocks rise sharply amid strong earnings",
        "Analysts fear a market correction is imminent"
    ]
    sentiment_score = fetch_sentiment(news_headlines)

    return render_template('result.html', plot_url=plot_url, sentiment=sentiment_score, strategy=strategy)

if __name__ == "__main__":
    # Run the app on 0.0.0.0 to make it accessible externally
    app.run(host="0.0.0.0", port=5000, debug=True)





