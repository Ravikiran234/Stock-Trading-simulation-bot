import yfinance as yf
import pandas as pd
import numpy as np

# Fetch historical stock data
def fetch_data(ticker, start, end):
    # Download stock data
    data = yf.download(ticker, start=str(start), end=str(end), progress=False)

    # If no data returned
    if data.empty:
        return pd.DataFrame()

    # Some tickers return multi-level columns from yfinance, fix that
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Ensure 'Close' column exists
    if 'Close' not in data.columns:
        return pd.DataFrame()

    # Keep only Close price and clean data
    data = data[['Close']].dropna()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])

    return data



# Q-Learning Based RL Trader
class SimpleRLTrader:
    def __init__(self, balance=10000):
        self.balance = float(balance)
        self.initial_balance = float(balance)
        self.position = 0
        self.history = []

        # Q-Learning parameters
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

        self.actions = ["buy", "sell", "hold"]

    # -----------------------------
    def get_state(self, prev_price, current_price):
        if current_price > prev_price:
            return "up"
        elif current_price < prev_price:
            return "down"
        else:
            return "flat"

    # -----------------------------
    def decide_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)

        return max(self.q_table[state], key=self.q_table[state].get)

    # -----------------------------
    def update_q(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}

        current_q = self.q_table[state][action]
        max_future_q = max(self.q_table[next_state].values())

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )

        self.q_table[state][action] = new_q

    # -----------------------------
    def train_and_trade(self, data):

        prices = data['Close'].values

        # Add first portfolio value to match length
        if len(prices) > 0:
            first_price = float(prices[0])
            initial_value = self.balance + self.position * first_price
            self.history.append(initial_value)

        for i in range(1, len(prices)):
            prev_price = float(prices[i - 1])
            current_price = float(prices[i])

            state = self.get_state(prev_price, current_price)
            action = self.decide_action(state)

            reward = 0

            if action == "buy" and self.balance >= current_price:
                self.position += 1
                self.balance -= current_price

            elif action == "sell" and self.position > 0:
                self.position -= 1
                self.balance += current_price
                reward = current_price - prev_price

            total_value = self.balance + self.position * current_price
            self.history.append(total_value)

            if i < len(prices) - 1:
                next_state = self.get_state(current_price, float(prices[i + 1]))
            else:
                next_state = state

            self.update_q(state, action, reward, next_state)

        return self.history


# Simple Moving Average Strategy
def moving_average_strategy(data, short_window=5, long_window=20, initial_balance=10000):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_ma'] = data['Close'].rolling(window=short_window).mean()
    signals['long_ma'] = data['Close'].rolling(window=long_window).mean()

    position = 0
    balance = float(initial_balance)
    history = []

    for i in range(len(signals)):
        price = signals['price'].iloc[i]
        if pd.isna(price) or pd.isna(signals['short_ma'].iloc[i]) or pd.isna(signals['long_ma'].iloc[i]):
            history.append(balance + position * (price if not pd.isna(price) else 0))
            continue

        if signals['short_ma'].iloc[i] > signals['long_ma'].iloc[i] and balance >= price:
            position += 1
            balance -= price
        elif signals['short_ma'].iloc[i] < signals['long_ma'].iloc[i] and position > 0:
            position -= 1
            balance += price

        history.append(balance + position * price)

    return history
