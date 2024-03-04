import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# Load and preprocess the dataset
file_path = 'cleaned_niftybankfinaldata.csv'
df = pd.read_csv(file_path)
df['Date-Time'] = pd.to_datetime(df['Date-Time'], format='%d-%m-%Y')
df_sorted = df.sort_values('Date-Time')
split_idx = int(len(df_sorted) * 0.8)
features = df_sorted.iloc[:, 1:]  # Excluded the date column
target = df_sorted.iloc[:, 2]  # The target variable is the closing price

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

# Define the quantiles for prediction
quantiles = [0.1,0.2, 0.4,0.6,0.8,0.9]

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=len(quantiles)))

# Custom quantile loss for each quantile
def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

# Compile the model with the quantile loss for each output neuron
model.compile(optimizer='adam', loss=[lambda y, f: quantile_loss(quantiles[i], y, f) for i in range(len(quantiles))])

# Fit the model on the training data
history = model.fit(
    X_train, [y_train for _ in range(len(quantiles))],
    epochs=200,
    batch_size=32,
    verbose=1,
    validation_data=(X_test, [y_test for _ in range(len(quantiles))])
)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Reshape the predictions to comply with the scaler's expected input
y_pred_train = y_pred_train.reshape(-1, len(quantiles))
y_pred_test = y_pred_test.reshape(-1, len(quantiles))

# Inverse transform the predictions and actuals
y_pred_train_inv = scaler_target.inverse_transform(y_pred_train)
y_pred_test_inv = scaler_target.inverse_transform(y_pred_test)
y_train_inv = scaler_target.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE for each quantile
train_rmse = [np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv[:, i])) for i in range(len(quantiles))]
test_rmse = [np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv[:, i])) for i in range(len(quantiles))]

# Calculate percentage error for each quantile
train_percentage_error = [(rmse / np.mean(y_train_inv)) * 100 for rmse in train_rmse]
test_percentage_error = [(rmse / np.mean(y_test_inv)) * 100 for rmse in test_rmse]

# Print the RMSE and percentage errors
for i, q in enumerate(quantiles):
    print(f"Quantile {q}: Train RMSE: {train_rmse[i]}, Test RMSE: {test_rmse[i]}")
    print(f"Quantile {q}: Train Percentage Error: {train_percentage_error[i]}, Test Percentage Error: {test_percentage_error[i]}")

# Plot the results
# Plot the results for each quantile in separate plots
for i, q in enumerate(quantiles):
    plt.figure(figsize=(10, 4))
    plt.plot(y_train_inv.flatten(), label='Actual')
    plt.plot(y_pred_train_inv[:, i].flatten(), label=f'Predicted Q{int(q*100)}')
    plt.title(f'Training Set: Actual vs Predicted Q{int(q*100)}')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()


# Predict the next 7 days
# Predict the next 7 days
last_known_day = test_features_scaled[-time_steps:]
predictions_7_days = []
current_batch = last_known_day.reshape(1, time_steps, -1)

for i in range(7):
    current_pred = model.predict(current_batch)[0]
    predictions_7_days.append(current_pred)
    # Update only the last feature with the predicted values
    # You may want to devise a better method for updating other features
    current_batch[0, -1, -1] = current_pred[-1]  # Assuming the last feature is the target
    # Shift the window
    current_batch = np.roll(current_batch, -1, axis=1)

# Convert predictions to their original scale and create DataFrame
predictions_7_days = np.array(predictions_7_days).reshape(-1, len(quantiles))
predictions_7_days_inv = scaler_target.inverse_transform(predictions_7_days)
# After obtaining predictions_7_days_inv
for i in range(predictions_7_days_inv.shape[0]):
    predictions_7_days_inv[i, :] = np.sort(predictions_7_days_inv[i, :])

# Now create the DataFrame with the post-processed, sorted predictions
predictions_table = pd.DataFrame(predictions_7_days_inv, columns=[f'Predicted Q{int(q*100)}' for q in quantiles])
print(predictions_table)


# Plot the predicted quantiles for t+7 days in separate plots
for i, q in enumerate(quantiles):
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 8), predictions_7_days_inv[:, i], label=f'Predicted Q{int(q*100)}')
    plt.title(f'Predicted Closing Price for Q{int(q*100)} over t+7 Days')
    plt.xlabel('Day')
    plt.ylabel('Predicted Closing Price')
    plt.legend()
    plt.show()
