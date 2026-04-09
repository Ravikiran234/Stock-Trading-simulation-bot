import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import yfinance as yf
import numpy as np
from rl_trading import fetch_data, SimpleRLTrader, moving_average_strategy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from sentiment_utils import get_news_sentiment, send_telegram_alert

NEWS_API_KEY = "d4183b87b84f4966b265c54f3ef2b5e9"
BOT_TOKEN = "8248866353:AAHRgVYnlsayjcTVRXOR_wJeUkf1hPnZ62w"
CHAT_ID = "1417336803"


# -------------------------------
# Initialize Session State
# -------------------------------
if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False
    st.session_state.sim_data = None
    st.session_state.rl_values = None
    st.session_state.ma_values = None
    st.session_state.ticker = None

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Stock Trading Bot", layout="wide")
st.title("📈 Stock Trading Bot Simulation ")
st.subheader("🌍 Live Market Overview")

# -------------------------------
# Function to get Top Gainers
# -------------------------------
def get_auto_top_gainers(region="india", top_n=5):
    try:
        if region == "india":
            tickers = [
                "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS", "ITC.NS",
                "BHARTIARTL.NS", "KOTAKBANK.NS", "SBIN.NS", "SUNPHARMA.NS", "WIPRO.NS", "TITAN.NS",
                "TATASTEEL.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "ONGC.NS", "POWERGRID.NS", "ULTRACEMCO.NS"
            ]
        else:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "PG", "UNH", "DIS"]

        data = yf.download(tickers, period="5d", interval="1d", progress=False)["Close"]
        if data.empty:
            return pd.DataFrame()

        latest = data.iloc[-1]
        previous = data.iloc[-2]
        percent_change = ((latest - previous) / previous) * 100

        df = pd.DataFrame({
            "Stock": latest.index,
            "Latest Price": latest.values.round(2),
            "Change (%)": percent_change.values.round(2),
            "Trend": ["📈" if x > 0 else "📉" for x in percent_change]
        }).sort_values("Change (%)", ascending=False)

        return df.head(top_n)
    except Exception as e:
        st.warning(f"⚠️ Could not fetch data: {e}")
        return pd.DataFrame()

# -------------------------------
# Show Top 5 Gainers (India & USA)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🇮🇳 Top 5 Indian Gainers Today")
    india_gainers = get_auto_top_gainers(region="india")
    if not india_gainers.empty:
        st.dataframe(india_gainers, hide_index=True)
    else:
        st.warning("Could not fetch Indian stock data right now.")

with col2:
    st.markdown("#### 🇺🇸 Top 5 US Gainers Today")
    us_gainers = get_auto_top_gainers(region="usa")
    if not us_gainers.empty:
        st.dataframe(us_gainers, hide_index=True)
    else:
        st.warning("Could not fetch US stock data right now.")

# -------------------------------
# User Inputs for Simulation
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS, INFY.NS)", value="AAPL")

with col2:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))

with col3:
    end_date = st.date_input("End Date", value=pd.to_datetime("2021-01-01"))

initial_balance = st.number_input("💰 Enter Starting Balance", min_value=1000, max_value=1000000, value=10000, step=500)

# -------------------------------
# Run Simulation
# -------------------------------
if st.button("🚀 Run Simulation"):
    st.info(f"Fetching stock data for **{ticker}**...")
    data = fetch_data(ticker, start_date, end_date)

    if data.empty:
        st.error("⚠️ No data found for the given ticker and date range.")
    else:
        st.success("✅ Data fetched successfully!")

        st.subheader("📊 Original Stock Price Trend")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(data.index, data['Close'], color='purple', linewidth=2)
        ax1.set_title(f"{ticker} - Original Close Price Data", fontsize=14)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close Price ($)")
        st.pyplot(fig1)

        st.subheader("⚙️ Strategy Simulation in Progress...")
        rl_bot = SimpleRLTrader(balance=initial_balance)
        rl_values = rl_bot.train_and_trade(data)
        ma_values = moving_average_strategy(data, initial_balance=initial_balance)

        final_rl = rl_values[-1]
        final_ma = ma_values[-1]
        rl_profit_ratio = ((final_rl - initial_balance) / initial_balance) * 100
        ma_profit_ratio = ((final_ma - initial_balance) / initial_balance) * 100

        recommended = "Reinforcement Learning (RL) Trader" if final_rl > final_ma else "Moving Average Strategy"
        st.session_state.recommended_strategy = recommended
        rec_profit = (final_rl - initial_balance) if final_rl > final_ma else (final_ma - initial_balance)  # Add this
        rec_ratio = rl_profit_ratio if final_rl > final_ma else ma_profit_ratio
        # Portfolio Comparison
        st.subheader("📉 Strategy Portfolio Performance Comparison")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(data.index, rl_values, label="RL Trader", color="blue", linewidth=2)
        ax2.plot(data.index, ma_values, label="Moving Average Strategy", color="orange", linewidth=2)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Portfolio Value ($)")
        ax2.legend()
        st.pyplot(fig2)
        st.markdown("### 🧮 Final Portfolio Summary")
        col1, col2 = st.columns(2)
        col1.metric("RL Trader Final Value", f"${final_rl:,.2f}", f"{rl_profit_ratio:.2f}%")
        col2.metric("MA Strategy Final Value", f"${final_ma:,.2f}", f"{ma_profit_ratio:.2f}%")

        st.markdown("---")
        st.markdown(f"### ✅ **Recommended Strategy: {recommended}**")
        st.markdown(f"**Expected Profit:** ${rec_profit:,.2f}  |  **Profit Ratio:** {rec_ratio:.2f}%")
        st.success(f"🎯 The **{recommended}** performed better with a profit ratio of **{rec_ratio:.2f}%**.")

        # Save simulation data
        # Save simulation data
        st.session_state.simulation_done = True
        st.session_state.sim_data = data
        st.session_state.ticker = ticker
        st.session_state.initial_balance = initial_balance
        st.session_state.rl_values = rl_values
        st.session_state.ma_values = ma_values

# -------------------------------
# 🔮 Future Price Prediction (LSTM)
# -------------------------------
st.markdown("---")
st.subheader("🔮 Future Stock Price Prediction (Next 30 Days Using LSTM)")

if st.button("📊 Predict with LSTM"):
    if not st.session_state.simulation_done:
        st.warning("⚠️ Please run a simulation first.")
    else:
        ticker = st.session_state.ticker
        pred_data = yf.download(ticker, period="1y", interval="1d", progress=False)[["Close"]]

        if pred_data.empty:
            st.error("No data found for prediction.")
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(pred_data["Close"].values.reshape(-1, 1))
            # --------------------------------
            # News Sentiment Analysis
            # --------------------------------
            sentiment_score = get_news_sentiment(NEWS_API_KEY, ticker)

            st.markdown("### 📰 Market Sentiment From News")

            if sentiment_score > 0.05:
                sentiment_label = "📈 Positive"
            elif sentiment_score < -0.05:
                sentiment_label = "📉 Negative"
            else:
                sentiment_label = "➖ Neutral"

            st.write(f"Sentiment Score: **{sentiment_score:.3f}**")
            st.write(f"Overall Sentiment: **{sentiment_label}**")

            seq_len = 60
            X_train, y_train = [], []
            for i in range(seq_len, len(scaled_data)):
                X_train.append(scaled_data[i - seq_len:i, 0])
                y_train.append(scaled_data[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mean_squared_error")

            with st.spinner("Training LSTM model..."):
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            st.success("✅ Model training complete!")

            # Predict next 30 days
            last_60_days = scaled_data[-60:]
            future_predictions = []
            input_seq = last_60_days.copy()

            for _ in range(30):
                X_test = np.reshape(input_seq, (1, input_seq.shape[0], 1))
                pred = model.predict(X_test, verbose=0)
                future_predictions.append(pred[0, 0])
                input_seq = np.append(input_seq[1:], pred, axis=0)

            future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_dates = pd.date_range(pred_data.index[-1] + timedelta(days=1), periods=30, freq='B')
            # --------------------------------
            # Adjust predictions using sentiment
            # --------------------------------

            if sentiment_score > 0.05:
                future_prices = future_prices * 1.02

            elif sentiment_score < -0.05:
                future_prices = future_prices * 0.98

            pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_prices.flatten()})
            st.dataframe(pred_df)

            plt.figure(figsize=(10, 5))
            plt.plot(pred_data.index, pred_data["Close"], label="Historical Close", color="blue")
            plt.plot(pred_df["Date"], pred_df["Predicted Close"], label="Predicted Close (LSTM)", color="orange")
            plt.title(f"{ticker} - Next 30 Days Price Forecast (LSTM)")
            plt.legend()
            st.pyplot(plt)

            # -------------------------------
            # 💰 Investment Suggestion Based on LSTM Predictions
            # -------------------------------

            # Detect currency based on ticker
        currency = "₹" if ".NS" in ticker.upper() else "$"

        # Find absolute lowest and highest prices (not dependent on order)
        min_price_idx = np.argmin(future_prices)
        max_price_idx = np.argmax(future_prices)

        buy_date = pred_df.iloc[min_price_idx]['Date']
        sell_date = pred_df.iloc[max_price_idx]['Date']
        buy_price = future_prices[min_price_idx][0]
        sell_price = future_prices[max_price_idx][0]
        profit = sell_price - buy_price
        return_pct = (profit / buy_price) * 100

        st.markdown("### 💡 Investment Recommendation")
        st.write(f"📅 **Best Day to Buy (Lowest Price):** {buy_date.date()} at {currency}{buy_price:.2f}")
        st.write(f"📅 **Best Day to Sell (Highest Price):** {sell_date.date()} at {currency}{sell_price:.2f}")
        st.success(
            f"💰 **Expected Profit (if bought low and sold high):** {currency}{profit:.2f}  |  **Return:** {return_pct:.2f}%")

        # --------------------------------
        # Telegram Alert
        # --------------------------------
        investment = st.session_state.initial_balance

        message = f"""
        🚨 STOCK PREDICTION ALERT

        Ticker: {ticker}
        
        💰 Initial Investment Used In Simulation:
         {currency}{investment}

        Recommended Strategy: {st.session_state.recommended_strategy}

        Best Day To Buy:
        {buy_date.date()} at {currency}{buy_price:.2f}

        Best Day To Sell:
        {sell_date.date()} at {currency}{sell_price:.2f}

        Expected Profit:
        {currency}{profit:.2f}

        Return:
        {return_pct:.2f}%

        Sentiment:
        {sentiment_label}

        """

        send_telegram_alert(BOT_TOKEN, CHAT_ID, message)

        # Visualize buy/sell points on prediction chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pred_df['Date'], pred_df['Predicted Close'], label='Predicted Close (LSTM)', color='blue')
        ax.scatter(buy_date, buy_price, color='green', label='Best Buy Day', s=100, marker='^')
        ax.scatter(sell_date, sell_price, color='red', label='Best Sell Day', s=100, marker='v')
        ax.legend()
        ax.set_title(f"📈 LSTM Predicted Prices with Best Buy/Sell Days ({currency})")
        st.pyplot(fig)

        # -------------------------------
        # 💾 Download Buttons for Simulation & Prediction Data
        # -------------------------------

        # ✅ 1️⃣ Download Simulation Data (already generated earlier)
        if st.session_state.sim_data is not None:
            sim_df = pd.DataFrame({
                "Date": st.session_state.sim_data.index,
                "Close Price": st.session_state.sim_data["Close"].values,
                "RL Portfolio Value": st.session_state.rl_values,
                "MA Portfolio Value": st.session_state.ma_values
            })
            csv_buffer = io.StringIO()
            sim_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Simulation Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"{st.session_state.ticker}_simulation_results.csv",
                mime="text/csv"
            )

        # ✅ 2️⃣ Download LSTM Prediction Data
        pred_df_export = pred_df.copy()
        pred_df_export["Currency"] = currency
        pred_df_export["Best Buy Date"] = buy_date
        pred_df_export["Best Sell Date"] = sell_date
        pred_df_export["Expected Profit"] = round(float(profit), 2)
        pred_df_export["Return (%)"] = round(float(return_pct), 2)

        csv_pred = io.StringIO()
        pred_df_export.to_csv(csv_pred, index=False)
        st.download_button(
            label="📈 Download LSTM Prediction Results (CSV)",
            data=csv_pred.getvalue(),
            file_name=f"{ticker}_LSTM_predictions.csv",
            mime="text/csv"
        )

