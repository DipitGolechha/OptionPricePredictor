#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 1 04:28:37 2023

@author: dipit
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load and preprocess the dataset
file_path = 'cleaned_niftybankfinaldata.csv'
df = pd.read_csv(file_path)
df['Date-Time'] = pd.to_datetime(df['Date-Time'], format='%d-%m-%Y')
df_sorted = df.sort_values('Date-Time')
split_idx = int(len(df_sorted) * 0.8)
features = df_sorted.iloc[:, 1:]  # Excluded the date column
target = df_sorted.iloc[:, 4]  # The target variable is the closing price


# Normalize the features
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_features.fit(features)
train_features_scaled = scaler_features.transform(features.iloc[:split_idx])
test_features_scaled = scaler_features.transform(features.iloc[split_idx:])

# Normalize the target
scaler_target = MinMaxScaler(feature_range=(0, 1))
scaler_target.fit(target.values.reshape(-1, 1))  # Fit on the entire target data
train_target_scaled = scaler_target.transform(target.iloc[:split_idx].values.reshape(-1, 1))
test_target_scaled = scaler_target.transform(target.iloc[split_idx:].values.reshape(-1, 1))

# Create the LSTM dataset
def create_lstm_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps), :])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys).reshape(-1, 1)  # Ensure y is 2D

time_steps = 60
X_train, y_train = create_lstm_dataset(train_features_scaled, train_target_scaled, time_steps)
X_test, y_test = create_lstm_dataset(test_features_scaled, test_target_scaled, time_steps)


quantiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=len(quantiles)))


def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)


model.compile(optimizer='adam', loss=[lambda y, f: quantile_loss(quantiles[i], y, f) for i in range(len(quantiles))])

history = model.fit(
    X_train, [y_train for _ in range(len(quantiles))],
    epochs=50,
    batch_size=32,
    verbose=1,
    validation_data=(X_test, [y_test for _ in range(len(quantiles))])
)


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


y_pred_train = y_pred_train.reshape(-1, len(quantiles))
y_pred_test = y_pred_test.reshape(-1, len(quantiles))


y_pred_train_inv = scaler_target.inverse_transform(y_pred_train)
y_pred_test_inv = scaler_target.inverse_transform(y_pred_test)
y_train_inv = scaler_target.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1))


train_rmse = [np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv[:, i])) for i in range(len(quantiles))]
test_rmse = [np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv[:, i])) for i in range(len(quantiles))]


train_percentage_error = [(rmse / np.mean(y_train_inv)) * 100 for rmse in train_rmse]
test_percentage_error = [(rmse / np.mean(y_test_inv)) * 100 for rmse in test_rmse]


for i, q in enumerate(quantiles):
    print(f"Quantile {q}: Train RMSE: {train_rmse[i]}, Test RMSE: {test_rmse[i]}")
    print(f"Quantile {q}: Train Percentage Error: {train_percentage_error[i]}, Test Percentage Error: {test_percentage_error[i]}")


for i, q in enumerate(quantiles):
    plt.figure(figsize=(10, 4))
    plt.plot(y_train_inv.flatten(), label='Actual')
    plt.plot(y_pred_train_inv[:, i].flatten(), label=f'Predicted Q{int(q*100)}')
    plt.title(f'Training Set: Actual vs Predicted Q{int(q*100)}')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

for i, q in enumerate(quantiles):
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_inv.flatten(), label='Actual')
    plt.plot(y_pred_test_inv[:, i].flatten(), label=f'Predicted Q{int(q*100)}')
    plt.title(f'Test Set: Actual vs Predicted Q{int(q*100)}')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()



last_known_day = test_features_scaled[-time_steps:]  
predictions_7_days = []
current_batch = last_known_day.reshape(1, time_steps, -1)


target_feature_index = 4  

for i in range(7):  # Predict 7 days into the future
    current_pred = model.predict(current_batch)[0]
    predictions_7_days.append(current_pred)
    
    current_batch = np.roll(current_batch, -1, axis=1)
    
    current_batch[0, -1, target_feature_index] = current_pred[-1]  # Update only the target feature
    
    current_batch[0, -1, :target_feature_index] = current_batch[0, -2, :target_feature_index]

predictions_7_days = np.array(predictions_7_days).reshape(-1, len(quantiles))
predictions_7_days_inv = scaler_target.inverse_transform(predictions_7_days)

for i in range(predictions_7_days_inv.shape[0]):
    predictions_7_days_inv[i, :] = np.sort(predictions_7_days_inv[i, :])

predictions_table = pd.DataFrame(predictions_7_days_inv, columns=[f'Predicted Q{int(q*100)}' for q in quantiles])
print(predictions_table)

for i, q in enumerate(quantiles):
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 8), predictions_7_days_inv[:, i], label=f'Predicted Q{int(q*100)}')
    plt.title(f'Predicted Closing Price for Q{int(q*100)} over t+7 Days')
    plt.xlabel('Day')
    plt.ylabel('Predicted Closing Price')
    plt.legend()
    plt.show()
    
arrays_quantiles = {col: predictions_table[col].values for col in predictions_table}


def black_scholes_call(S, K, T, r,sigma):
    # S: spot price
    q=0
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    call_price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

# Last known stock price (spot price)
S = df.iloc[0, 4]
# Strike price
K = df.iloc[0, 4]

# Time to maturity (7 days in terms of year fraction)
T = 7 / 365  

# Risk-free rate (annual)
r = 0.0685  

# Volatility of the stock 
sigma = 0.19

# Calculate call option price at t=0
option_price_t0 = black_scholes_call(S, S-100, T, r, sigma)

quantiles = ['Q10', 'Q20', 'Q40', 'Q60', 'Q80', 'Q90']

for quantile in quantiles:
    predicted_stock_prices = arrays_quantiles[f'Predicted {quantile}']
    
    option_prices = []
    
    for price in predicted_stock_prices:
        option_price = black_scholes_call(price, K, T, r, sigma)
        option_prices.append(option_price)
    
    option_prices = np.array(option_prices)
    
    # Calculate the percentage profit/loss
    percentage_profit_loss = ((option_prices - option_price_t0) / option_price_t0) * 100
    
    days = np.arange(1, len(predicted_stock_prices) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(days, option_prices, marker='o', linestyle='-', color='blue')
    plt.title(f'Option Prices for Predicted Stock Paths ({quantile})')
    plt.xlabel('Day')
    plt.ylabel('Option Price')
    plt.show()
    
    # Plotting the percentage profit/loss histogram
    plt.figure(figsize=(12, 6))
    plt.bar(days, percentage_profit_loss, color='green')
    plt.title(f'Histogram of % Profit/Loss for Predicted Option Prices ({quantile})')
    plt.xlabel('Day')
    plt.ylabel('% Profit/Loss')
    plt.axhline(y=0, color='black', linestyle='-') 
    plt.show()

    last_option_price = option_prices[-1]

    # Calculate the profitability
    profitability = ((last_option_price - option_price_t0)/option_price_t0) *100

    print(f"Quantile : {quantile}")
    print("Last Option Price from Predictions:", last_option_price)
    print("Option Price at Time 0:", option_price_t0)
    print("Profitability of the Trade:", profitability)
    
sell_threshold=500
buy_threshold=500




for quantile in quantiles:
    initial_investment = 1000  # Starting with 1000 rupees
    current_investment = initial_investment
    investment_over_time = []
    option_prices = []

    predicted_stock_prices = arrays_quantiles[f'Predicted {quantile}']
    for price in predicted_stock_prices:
        option_price = black_scholes_call(price, K, T, r, sigma)
        option_prices.append(option_price)

    # Convert the list of option prices to a numpy array
    option_prices = np.array(option_prices)

    buy_signals = []
    sell_signals = []

    for time_span in range(1, 6):  # Check price changes over 1, 2, 3, 4, 5 days
        for i in range(time_span, len(predicted_stock_prices)):
            price_change = predicted_stock_prices[i] - predicted_stock_prices[i - time_span]

            if price_change < buy_threshold and not buy_signals:  # Ensure no concurrent buy signals
                buy_price = black_scholes_call(predicted_stock_prices[i], predicted_stock_prices[i] - 100, T, r, sigma)
                buy_signals.append((i, buy_price))  # Store the day and buy price

            elif price_change > sell_threshold and buy_signals and i > buy_signals[-1][0]:  # Only sell if it's a future day
                sell_price = black_scholes_call(predicted_stock_prices[i], predicted_stock_prices[i] - 100, T, r, sigma)
                sell_signals.append((i, sell_price))

                # Calculate percentage profit
                profit_percent = ((sell_price - buy_signals[-1][1]) / buy_signals[-1][1]) * 100
                current_investment *= (1 + profit_percent / 100)  # Update the investment
                investment_over_time.append(current_investment)

                buy_signals.pop()  # Clear the buy signal

    print(f"Results for Quantile {quantile}:")
    print("Final Investment:", current_investment)

    # plot the investment over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(investment_over_time)), investment_over_time, marker='o', linestyle='-')
    plt.title(f"Investment Over Time for Quantile {quantile}")
    plt.xlabel("Trade Number")
    plt.ylabel("Investment Amount")
    plt.show()
