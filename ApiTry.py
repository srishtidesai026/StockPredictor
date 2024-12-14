import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Fetch historical stock data
ticker = "AAPL"
stock_data = yf.Ticker(ticker)

# Fetch data for the last 1 year (or adjust as necessary)
data = stock_data.history(period="1y")

# Display the data
print(data)

# Step 2: Preprocess data - Use only the 'Close' prices for ARIMA
data = data[['Close']]

# Check for stationarity (optional, but a good practice)
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['Close'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# If the p-value is greater than 0.05, the data is non-stationary, so we'll need to difference it
if result[1] > 0.05:
    data['Close'] = data['Close'].diff().dropna()

# Step 3: Split data - ARIMA typically uses all available data for training
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

print(f"Train samples: {len(train)}, Test samples: {len(test)}")

# Step 4: Train the ARIMA model
# ARIMA parameters (p=5, d=1, q=0) can be adjusted as needed
model = ARIMA(train, order=(5,1,0))  # p=5, d=1, q=0 (adjust these parameters as needed)
model_fit = model.fit()

# Step 5: Make predictions
predictions = model_fit.forecast(steps=len(test))

# Step 6: Plot the predicted vs actual prices
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['Close'], label='Actual Price', color='blue')
plt.plot(test.index, predictions, label='Predicted Price', color='orange')
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()


# Step 7: Evaluate the model
mse = mean_squared_error(test['Close'], predictions)
r2 = r2_score(test['Close'], predictions)
rmse = np.sqrt(mse)  # Root Mean Squared Error


print(f'Mean Squared Error: {mse:.2f}')
print(f'R-Squared: {r2:.2f}')
print(f"RMSE: {rmse}")


# Step 8: Save the trained ARIMA model
with open('arima_stock_model.pkl', 'wb') as file:
    pickle.dump(model_fit, file)

print("ARIMA model saved successfully!")




'''
#Step 5: Plot the closing prices
data['4. close'].plot(title=f"Stock Prices for {symbol}")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()


# Fetch MACD data
macd_data, meta_data = ts.get_macd(symbol=symbol, interval='daily', series_type='close')

# Display and plot MACD data
print(macd_data.head())
macd_data.plot(title=f"MACD for {symbol}")
plt.xlabel("Date")
plt.ylabel("MACD Value")
plt.show()

# Combine stock data with RSI
data = data.join(rsi_data)

# Create Buy/Sell signals
data['Signal'] = 'Hold'
data.loc[data['RSI'] < 30, 'Signal'] = 'Buy'
data.loc[data['RSI'] > 70, 'Signal'] = 'Sell'

# Display signals
print(data[['4. close', 'RSI', 'Signal']])

import time

# Pause for 12 seconds between requests
time.sleep(12)
data.to_csv("stock_data.csv")
'''