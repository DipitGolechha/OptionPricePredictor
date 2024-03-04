#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:53:20 2023

@author: dipit
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import TimeDistributed

file_path = 'cleaned_niftybankfinaldata.csv'
df = pd.read_csv(file_path)

df['Date-Time'] = pd.to_datetime(df['Date-Time'], format='%d-%m-%Y')

df_sorted = df.sort_values('Date-Time')

split_idx = int(len(df_sorted) * 0.8)

# Spliting the data into features and target
features = df_sorted.iloc[:, 1:] #excluded the date coloum  
target = df_sorted.iloc[:, 2]  # The target variable is the closing price

# Spliting into test and train sets
test_features = features[split_idx:]
train_features = features[:split_idx]
test_target = target[split_idx:]
train_target = target[:split_idx]

# Normalizing the features
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_features.fit(features) 
train_features_scaled = scaler_features.transform(train_features)
test_features_scaled = scaler_features.transform(test_features)

# Normalizing the target
scaler_target = MinMaxScaler(feature_range=(0, 1))
train_target_scaled = scaler_target.fit_transform(train_target.values.reshape(-1, 1))
test_target_scaled = scaler_target.transform(test_target.values.reshape(-1, 1))

# Creating the LSTM dataset
def create_lstm_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps), :])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60
X_train, y_train = create_lstm_dataset(train_features_scaled, train_target_scaled, time_steps)
X_test, y_test = create_lstm_dataset(test_features_scaled, test_target_scaled, time_steps)

# Defining the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fiting the model to the training data
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Making predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Inverse transforming the normalization
y_pred_train_inv = scaler_target.inverse_transform(y_pred_train)
y_pred_test_inv = scaler_target.inverse_transform(y_pred_test)
y_train_inv = scaler_target.inverse_transform(y_train)
y_test_inv = scaler_target.inverse_transform(y_test)

# Calculating RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))

# Calculating percentage error
train_percentage_error = (train_rmse / np.mean(y_train_inv)) * 100
test_percentage_error = (test_rmse / np.mean(y_test_inv)) * 100

# Plotting the results
plt.figure(figsize=(15, 6))

# Training predictions
plt.subplot(1, 2, 1)
plt.plot(y_train_inv.flatten(), label='Actual')
plt.plot(y_pred_train_inv.flatten(), label='Predicted')
plt.title('Training Set: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()

# Test predictions
plt.subplot(1, 2, 2)
plt.plot(y_test_inv.flatten(), label='Actual')
plt.plot(y_pred_test_inv.flatten(), label='Predicted')
plt.title('Test Set: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()

plt.tight_layout()
plt.show()

print(train_rmse, test_rmse, train_percentage_error, test_percentage_error)

