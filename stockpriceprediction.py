import streamlit as st
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.io as pio
pio.renderers.default = "colab"

st.title("📈 Stock Prediction App")

# --------------------------
# User Inputs
# --------------------------
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, INFY.NS):", "AAPL")
years = st.number_input("Enter how many years of data you want", min_value=1, max_value=20, value=5)

if st.button("Run Model"):

    # ORIGINAL CODE STARTS (unchanged)
    period = f"{years}y"

    data = yf.download(ticker, period=period, interval="1d")
    data.dropna(inplace=True)

    # Moving Averages
    data["MA100"] = data["Close"].rolling(100).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    st.write("Data Loaded Successfully!")

    plt.figure(figsize=(14,6))
    plt.plot(data["Close"], label="Close Price")
    plt.plot(data["MA100"], label="100-Day MA")
    plt.plot(data["MA200"], label="200-Day MA")
    plt.title(f"{ticker} Stock Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(dataset):
        X, y = [], []
        for i in range(60, len(dataset)):
            X.append(dataset[i-60:i])
            y.append(dataset[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_prices = scaler.inverse_transform(y_test)

    # ACCURACY CHECK
    mae = mean_absolute_error(real_prices, predicted_prices)
    mse = mean_squared_error(real_prices, predicted_prices)
    rmse = np.sqrt(mse)

    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)

    candlestick_data = data.iloc[-len(real_prices):]

    # Debug prints
    print("candlestick_data shape:", candlestick_data.shape)
    print("predicted_prices shape:", predicted_prices.shape)
    print(candlestick_data.columns)
    print(candlestick_data[["Open","High","Low","Close"]].dtypes)
    print("NaNs per column:\n", candlestick_data[["Open","High","Low","Close"]].isna().sum())
    print("Index dtype:", candlestick_data.index.dtype)

    # Defensive copy & cleaning
    df = data.copy() 

    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass  

    for col in ["Open","High","Low","Close"]:
        df[(col, ticker)] = pd.to_numeric(df[(col, ticker)], errors="coerce")
    df = df.dropna(subset=[(c, ticker) for c in ["Open","High","Low","Close"]])
    if df.empty:
        st.error("candlestick_data has no valid OHLC rows after cleaning.")
        st.stop()

    pred = np.array(predicted_prices).flatten()

    if len(pred) != len(df):
        if len(pred) < len(df):
            df = df.iloc[-len(pred):]
        else:
            pred = pred[-len(df):]

    print("Final df length:", len(df), "pred len:", len(pred))

    # Plotting Candlestick & Prediction
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df[("Open", ticker)],
        high=df[("High", ticker)],
        low=df[("Low", ticker)],
        close=df[("Close", ticker)],
        name="Real Price",
        increasing=dict(line=dict(color="green")),
        decreasing=dict(line=dict(color="red")),
        whiskerwidth=0.5,
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=pred,
        mode="lines",
        name="Predicted Price",
        line=dict(width=2, color="white"),
        opacity=0.8
    ))

    fig.update_layout(
        title=f"{ticker} – Real Price (Candlestick) vs Predicted Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        autosize=True,
        xaxis=dict(rangeslider=dict(visible=False)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

       # -------------------------------
    # PREDICT NEXT 60 DAYS
    # -------------------------------
    future_days = 60
    last_60 = scaled_data[-60:].reshape(1, 60, 1)

    future_predictions = []

    for _ in range(future_days):
        next_pred = model.predict(last_60, verbose=0)  # shape (1,1)
        future_predictions.append(next_pred[0][0])

        # reshape prediction for LSTM window
        next_pred_reshaped = next_pred.reshape(1, 1, 1)

        # shift the last_60 window and append new prediction
        last_60 = np.append(last_60[:, 1:, :], next_pred_reshaped, axis=1)

    # Inverse scale
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    # -------------------------------
    # CREATE FAKE OHLC FOR 60-DAY CANDLESTICK
    # -------------------------------
    pred_close = future_predictions.flatten()

    pred_open = np.roll(pred_close, 1)
    pred_open[0] = real_prices[-1]

    volatility = (np.max(real_prices) - np.min(real_prices)) * 0.002

    pred_high = pred_close + (volatility * np.random.uniform(0.5, 1.2, len(pred_close)))
    pred_low = pred_close - (volatility * np.random.uniform(0.5, 1.2, len(pred_close)))

    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=future_days
    )

    # -------------------------------
    # STREAMLIT FUTURE CANDLESTICK
    # -------------------------------
    fig_future = go.Figure()

    fig_future.add_trace(go.Candlestick(
        x=future_dates,
        open=pred_open,
        high=pred_high,
        low=pred_low,
        close=pred_close,
        increasing=dict(line=dict(color="cyan")),
        decreasing=dict(line=dict(color="magenta")),
        name="Predicted Candlesticks"
    ))

    fig_future.update_layout(
        title=f"{ticker} — Predicted Next 60 Days (Candlestick)",
        xaxis_title="Date",
        yaxis_title="Predicted Price",
        template="plotly_dark",
        xaxis=dict(rangeslider=dict(visible=False))
    )

    st.subheader("📅 Next 60-Day Forecast")
    st.plotly_chart(fig_future, use_container_width=True)
