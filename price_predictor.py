import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# ------------------------------
# Fetch Data
# ------------------------------
def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if 'Close' not in df.columns:
        raise ValueError("No 'Close' column found in data.")
    return df[['Close']]

# ------------------------------
# Prepare Data for LSTM
# ------------------------------
def prepare_data(df, window_size=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(df[['Close']])

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# ------------------------------
# Build LSTM Model
# ------------------------------
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ------------------------------
# Predict Future Prices
# ------------------------------
def predict_future_prices(df, model, scaler, window_size=60, days_ahead=30):
    last_window = df['Close'][-window_size:].values
    last_scaled = scaler.transform(last_window.reshape(-1, 1))

    future_predictions = []
    current_input = last_scaled.reshape(1, window_size, 1)

    for _ in range(days_ahead):
        next_scaled = model.predict(current_input, verbose=0)
        future_predictions.append(next_scaled[0, 0])
        current_input = np.append(current_input[:, 1:, :], [[next_scaled]], axis=1)

    # Inverse transform predictions back to price scale
    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_prices

# ------------------------------
# Full Pipeline
# ------------------------------
def train_and_forecast(ticker, start_date, end_date, days_ahead=30):
    df = fetch_stock_data(ticker, start_date, end_date)
    X, y, scaler = prepare_data(df)
    model = build_lstm_model((X.shape[1], 1))

    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    future_prices = predict_future_prices(df, model, scaler, days_ahead=days_ahead)

    # Create a date index for future predictions
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date, periods=days_ahead+1, freq='B')[1:]

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': future_prices.flatten()
    })

    return df, forecast_df, model
