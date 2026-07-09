import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------
# Get News Sentiment (Guardian API - free tier works in production)
# --------------------------------
def get_news_sentiment(api_key, ticker):

    analyzer = SentimentIntensityAnalyzer()

    try:
        url = "https://content.guardianapis.com/search"
        params = {
            "q": ticker,
            "api-key": api_key,
            "order-by": "newest",
            "page-size": 20
        }

        response = requests.get(url, params=params, timeout=10)
        articles = response.json().get("response", {}).get("results", [])

        sentiments = []

        for article in articles:
            title = article["webTitle"]
            score = analyzer.polarity_scores(title)["compound"]
            sentiments.append(score)

        if len(sentiments) == 0:
            return 0

        avg_sentiment = sum(sentiments) / len(sentiments)
        return avg_sentiment

    except:
        return 0


# --------------------------------
# Telegram Alert
# --------------------------------
def send_telegram_alert(bot_token, chat_id, message):

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message
    }

    requests.post(url, data=payload)
