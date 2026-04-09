# 📈 AI-Powered Stock Trading Bot

## 🚀 Overview

This project is an **AI-based stock trading bot** that combines **Reinforcement Learning, LSTM (Deep Learning), and News Sentiment Analysis** to simulate trading strategies and predict future stock prices.

It provides a complete **end-to-end trading system** including:

* Strategy comparison
* Price prediction
* Investment recommendations
* Real-time alerts

---

## 🎯 Features

### 📊 Trading Strategies

* Reinforcement Learning (Q-Learning based trader)
* Moving Average Strategy

### 🔮 Prediction

* LSTM-based **30-day stock price forecasting**

### 📰 Sentiment Analysis

* News-based sentiment using News API
* VADER sentiment scoring
* Adjusts predictions based on market sentiment

### 💡 Investment Insights

* Best day to **Buy** (lowest predicted price)
* Best day to **Sell** (highest predicted price)
* Profit and Return calculation

### 📈 Visualization

* Historical stock trends
* Strategy performance comparison
* Predicted vs actual price graph
* Buy/Sell markers

### 🤖 Alerts

* Telegram bot integration
* Automatic alert after prediction with:

  * Ticker
  * Recommended strategy
  * Best buy/sell days
  * Expected profit & return

### 🌍 Market Overview

* Top gainers (India & US markets)

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Machine Learning:**

  * LSTM (TensorFlow/Keras)
  * Reinforcement Learning (Q-Learning)
* **Data Source:** Yahoo Finance (yfinance)
* **Sentiment Analysis:** VADER + News API
* **Visualization:** Matplotlib
* **Alerts:** Telegram Bot API

---

## 📂 Project Structure

```
stock_trading_bot/
│
├── app.py                  # Main Streamlit application
├── rl_trading.py           # RL trader & moving average strategy
├── sentiment_utils.py      # News sentiment + Telegram alerts
├── live_trading_bot.py     # Real-time alert bot
├── requirements.txt        # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/Ravikiran234/stock-trading-bot.git
cd stock-trading-bot
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Add API Keys

Create a folder `.streamlit/` and add `secrets.toml`:

```
NEWS_API_KEY = "your_news_api_key"
BOT_TOKEN = "your_telegram_bot_token"
CHAT_ID = "your_chat_id"
```

---

### 5️⃣ Run the App

```
streamlit run app.py
```

---

## 📊 How It Works

1. User selects stock ticker and date range
2. System fetches historical stock data
3. Runs:

   * Reinforcement Learning strategy
   * Moving Average strategy
4. Compares performance and recommends best strategy
5. LSTM model predicts next 30 days prices
6. News sentiment is analyzed and applied
7. Best buy/sell days are identified
8. Telegram alert is sent automatically

---

## 📸 Output Screens

* 📊 Stock price trends
* 📉 Strategy comparison graph
* 🔮 LSTM prediction graph
* 💡 Buy/Sell recommendations

---

## ⚠️ Limitations

* Free News API may not work in deployed environments
* LSTM predictions are based on historical data only
* Not intended for real financial trading decisions

---

## 🚀 Future Enhancements

* Add technical indicators (RSI, MACD)
* Improve prediction accuracy using transformers
* Multi-stock portfolio optimization
* Real-time trading dashboard
* FinBERT sentiment analysis

---

## 📌 Use Cases

* Stock market analysis
* Algorithmic trading research
* Machine learning projects
* Educational purposes

---

## 👨‍💻 Author

**Your Name**

* CSE Student
* Interested in AI, ML, and FinTech

---

## 📄 License

This project is for **educational purposes only**.

---

## ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 📢 Share with others

---
