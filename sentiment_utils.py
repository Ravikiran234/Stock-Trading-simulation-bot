import requests
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------
# Get News Sentiment
# --------------------------------
def get_news_sentiment(api_key, ticker):

    analyzer = SentimentIntensityAnalyzer()
    newsapi = NewsApiClient(api_key=api_key)

    try:
        news = newsapi.get_everything(
            q=ticker,
            language="en",
            sort_by="publishedAt",
            page_size=20
        )

        sentiments = []

        for article in news["articles"]:
            title = article["title"]
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
