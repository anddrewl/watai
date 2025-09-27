import praw
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import os
import time

REDDIT_CLIENT_ID = 'bsKMMMkoaVuWVWQkTleVrw' # Replace with your own
REDDIT_CLIENT_SECRET = 'oz78ES_veF_MHmjjCbMtSoi03bT7Dw' # Replace with your own
REDDIT_USER_AGENT = 'data_fetching_stock'
TICKER = 'TSLA' # Change to your desired stock ticker symbol
SUBREDDITS = ['wallstreetbets', 'stocks', 'investing']
START_DATE = datetime(2022, 9, 27)
END_DATE = datetime(2025, 9, 27)
DUMP_FILE = 'path_to_dump/processed.jsonl' # Replace with your pre-downloaded Reddit dump file if you have one

os.makedirs('reddit_data', exist_ok=True)
os.makedirs('financial_data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

analyzer = SentimentIntensityAnalyzer()

def fetch_historical_with_praw():
    """Fetches the most recent 1000 posts per subreddit via PRAW (fallback if no dump)."""
    posts = []
    for sub in SUBREDDITS:
        try:
            for submission in reddit.subreddit(sub).search(
                    f"{TICKER} OR ${TICKER}", # search for posts with substrings like "TSLA" or "$TSLA"
                    time_filter='all',
                    sort='new',
                    limit=1000):
                created = datetime.fromtimestamp(submission.created_utc)
                if START_DATE <= created <= END_DATE:
                    text = f"{submission.title} {submission.selftext}".lower()
                    if TICKER.lower() in text or f"${TICKER}".lower() in text:
                        posts.append({
                            'id': submission.id,
                            'title': submission.title,
                            'selftext': submission.selftext,
                            'score': submission.score,
                            'created_utc': created,
                            'subreddit': sub,
                            'url': submission.url
                        })
        except Exception as e:
            print(f"Error fetching r/{sub}: {e}")
    print(f"[PRAW] Retrieved {len(posts)} posts for {TICKER}")
    # save
    with open(f"reddit_data/historical_{TICKER}.json", 'w') as f:
        json.dump(posts, f, default=str, indent=2)
    return posts

def process_historical_dump(dump_path):
    """Load a Pushshift JSONL dump and filter by date/subreddit/ticker."""
    posts = []
    with open(dump_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(obj['created_utc'])
            if not (START_DATE <= created <= END_DATE):
                continue
            if obj.get('subreddit', '').lower() not in SUBREDDITS:
                continue
            text = f"{obj.get('title','')} {obj.get('selftext','')}".lower()
            if TICKER.lower() in text or f"${TICKER}".lower() in text:
                posts.append({
                    'id': obj['id'],
                    'title': obj['title'],
                    'selftext': obj.get('selftext', ''),
                    'score': obj.get('score', 0),
                    'created_utc': created,
                    'subreddit': obj['subreddit'],
                    'url': f"https://reddit.com{obj.get('permalink','')}"
                })
    print(f"[Dump] Processed {len(posts)} posts for {TICKER}")
    with open(f"reddit_data/filtered_{TICKER}.json", 'w') as f:
        json.dump(posts, f, default=str, indent=2)
    return posts

def load_historical_posts():
    """Attempts to use pre-downloaded Reddit data dump if available for efficiency sake, otherwise it runs live PRAW API queries."""
    if os.path.isfile(DUMP_FILE):
        return process_historical_dump(DUMP_FILE)
    else:
        return fetch_historical_with_praw()

def compute_sentiment(posts):
    """Computes daily average VADER sentiment scores (-1 for negative, +1 for positive)."""
    if not posts:
        return pd.DataFrame(columns=['date', 'sentiment'])
    records = []
    for p in posts:
        txt = f"{p['title']} {p.get('selftext','')}".strip()
        if len(txt) < 5: # ignore very short Reddit posts
            continue
        score = analyzer.polarity_scores(txt)['compound'] # compute sentiment score of the Reddit post
        date = p['created_utc'].date() if isinstance(p['created_utc'], datetime) else datetime.fromisoformat(p['created_utc']).date()
        records.append({'date': date, 'sentiment': score})
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=['date', 'sentiment'])
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby('date')['sentiment'].mean().reset_index()
    return daily

def fetch_financial_data(ticker, start, end):
    """Retrieve historical stock data from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end)
    if hist.empty:
        raise ValueError(f"No financial data for {ticker}")
    hist = hist.reset_index()[['Date', 'Close', 'Volume']]
    hist['Date'] = pd.to_datetime(hist['Date']).dt.date
    hist.set_index('Date', inplace=True)
    hist.to_csv(f"financial_data/{ticker}_historical.csv")
    print(f"Fetched {len(hist)} days of {ticker} data")
    return hist

def align_and_plot(sent_df, fin_df):
    """Join the sentiment and stock price data columns, computes the correlation btw sentiment and next-day stock returns, and plots both time series."""
    if sent_df.empty or fin_df.empty:
        print("No data to align/plot.")
        return None, None

    sent_df = sent_df.set_index('date').rename_axis('Date')
    fin_df = fin_df.copy()
    combo = fin_df.join(sent_df, how='inner')
    if combo.empty:
        print("No overlapping dates.")
        return None, None

    combo['next_return'] = combo['Close'].pct_change().shift(-1)
    corr = combo['sentiment'].corr(combo['next_return'])
    print(f"Correlation sentiment → next‐day return: {corr:.4f}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(combo.index, combo['Close'], color='tab:blue', label='Close')
    ax1.set_ylabel('Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', colors='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(combo.index, combo['sentiment'], color='tab:red', label='Sentiment')
    ax2.set_ylabel('Sentiment', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red',  colors='tab:red')

    plt.title(f"{TICKER} Price vs Reddit Sentiment, corr={corr:.3f}")
    fig.tight_layout()
    plt.savefig(f"plots/{TICKER}_sentiment_analysis.png", dpi=300)
    # plt.show()

    return combo, corr

def main():
    posts = load_historical_posts()
    daily_sent = compute_sentiment(posts)
    fin = fetch_financial_data(TICKER, START_DATE, END_DATE)
    aligned, _ = align_and_plot(daily_sent, fin)
    if aligned is not None:
        print("Analysis successful ✅")

if __name__ == "__main__":
    main()
