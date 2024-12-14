Simplified Trading Strategies with Flask Frontend

Overview

This project implements simplified intraday and swing trading strategies using Python. The backend leverages Yahoo Finance API to fetch stock data. A Flask frontend allows users to interact with the application, input stock tickers, and visualize trading signals.

Features

Visualization:

Dynamic plots of trading signals for swing strategies, including Bollinger Bands and buy/sell indicators.

Flask Frontend:

A user-friendly interface to input stock tickers, view trading signals, and sentiment analysis.

Technologies Used

Backend:

Python

Libraries: pandas, numpy, yfinance, matplotlib, VADER Sentiment Analysis

Frontend:

Flask

HTML, CSS, JavaScript (if applicable)


Project Structure
.
├── app.py                     # Flask application script
├── simplified_trading_strategy.py # Core trading strategy logic
├── templates/
│   └── index.html             # Frontend template for user interface
├── static/
│   ├── styles.css             # Styles for the frontend
│   └── scripts.js             # Optional JavaScript for interactivity
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
