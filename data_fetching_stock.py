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
import sqlite3
import re

REDDIT_CLIENT_ID = 'bsKMMMkoaVuWVWQkTleVrw' # Replace with your own
REDDIT_CLIENT_SECRET = 'oz78ES_veF_MHmjjCbMtSoi03bT7Dw' # Replace with your own
REDDIT_USER_AGENT = 'data_fetching_stock'
TICKER = 'AAPL' # Change to your desired stock ticker symbol
SUBREDDITS = ['wallstreetbets', 'stocks', 'investing']
START_DATE = datetime(2022, 10, 3)
END_DATE = datetime(2025, 10, 3)
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
    comments_list = []
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
                        media_url = None
                        if submission.media and 'reddit_video' in submission.media:
                            media_url = submission.media['reddit_video']['fallback_url']
                        elif not submission.is_self and submission.url.endswith(('.jpg', '.png', '.gif', '.mp4')):
                            media_url = submission.url
                        post_dict = {
                            'id': submission.id,
                            'title': submission.title,
                            'selftext': submission.selftext,
                            'score': submission.score,
                            'created_utc': created,
                            'subreddit': sub,
                            'url': submission.url,
                            'author': submission.author.name if submission.author else '[deleted]',
                            'upvote_ratio': submission.upvote_ratio,
                            'num_comments': submission.num_comments,
                            'media': media_url
                        }
                        posts.append(post_dict)
                        # fetch comments
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list():
                            com_dict = {
                                'post_platform_id': submission.id,
                                'platform_comment_id': comment.id,
                                'body': comment.body,
                                'author': comment.author.name if comment.author else '[deleted]',
                                'created_utc': datetime.fromtimestamp(comment.created_utc),
                                'score_likes': comment.score,
                                'media': None
                            }
                            comments_list.append(com_dict)
        except Exception as e:
            print(f"Error fetching r/{sub}: {e}")
    print(f"[PRAW] Retrieved {len(posts)} posts for {TICKER}")
    # save
    with open(f"reddit_data/historical_{TICKER}.json", 'w') as f:
        json.dump(posts, f, default=str, indent=2)
    return posts, comments_list

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
                media_url = obj.get('url') if obj.get('is_self', False) else None
                if 'media' in obj:
                    media_url = obj['media']
                post_dict = {
                    'id': obj['id'],
                    'title': obj['title'],
                    'selftext': obj.get('selftext', ''),
                    'score': obj.get('score', 0),
                    'created_utc': created,
                    'subreddit': obj['subreddit'],
                    'url': f"https://reddit.com{obj.get('permalink','')}",
                    'author': obj.get('author', '[deleted]'),
                    'upvote_ratio': obj.get('upvote_ratio'),
                    'num_comments': obj.get('num_comments', 0),
                    'media': media_url
                }
                posts.append(post_dict)
    print(f"[Dump] Processed {len(posts)} posts for {TICKER}")
    with open(f"reddit_data/filtered_{TICKER}.json", 'w') as f:
        json.dump(posts, f, default=str, indent=2)
    return posts, []

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

def extract_entities(text):
    """Extract tickers, hashtags, and mentions from text."""
    tickers = re.findall(r'\$\s*([A-Z]{1,5})\b', text.upper())
    hashtags = re.findall(r'#\s*(\w+)', text)
    mentions = re.findall(r'@\s*(\w+)', text)
    entities = [('ticker', t) for t in tickers] + [('hashtag', h) for h in hashtags] + [('mention', m) for m in mentions]
    return entities

def compute_sentiment_score_and_label(txt):
    if len(txt.strip()) < 5:
        return None, None
    score = analyzer.polarity_scores(txt)['compound']
    label = 'positive' if score > 0.05 else 'negative' if score < -0.05 else 'neutral'
    return score, label

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
    posts, comments = load_historical_posts()
    daily_sent = compute_sentiment(posts)
    fin = fetch_financial_data(TICKER, START_DATE, END_DATE)
    aligned, _ = align_and_plot(daily_sent, fin)
    if aligned is not None:
        print("Analysis successful ✅")
    
    conn = sqlite3.connect('data_fetched.sqlite')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY,
        platform TEXT,
        platform_post_id TEXT,
        title TEXT,
        body TEXT,
        author TEXT,
        subreddit_channel TEXT,
        created_utc DATETIME,
        score_likes INTEGER,
        upvote_ratio REAL,
        num_comments INTEGER,
        url TEXT,
        collected_at DATETIME,
        media TEXT
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY,
        post_id INTEGER,
        platform_comment_id TEXT,
        body TEXT,
        author TEXT,
        created_utc DATETIME,
        score_likes INTEGER,
        collected_at DATETIME,
        media TEXT
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        platform TEXT,
        platform_user_id TEXT,
        username TEXT,
        display_name TEXT,
        followers_count INTEGER,
        description TEXT,
        collected_at DATETIME
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS post_entities (
        post_id INTEGER,
        entity_type TEXT,
        value TEXT,
        PRIMARY KEY (post_id, entity_type, value)
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS features (
        id INTEGER PRIMARY KEY,
        ref_type TEXT,
        ref_id INTEGER,
        sentiment_label TEXT,
        sentiment_score REAL,
        topics TEXT,
        model_version TEXT,
        processed_at DATETIME
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS prices (
        id INTEGER PRIMARY KEY,
        ticker TEXT,
        ts DATETIME,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        source TEXT
    )''')

    collected_at = datetime.now()

    # NEED TO FIX THE DATABASE INSERTIONS BELOW (running into some errors)

    # insert posts and related data
    post_ids = {}
    for p in posts:
        cursor.execute('''INSERT INTO posts (platform, platform_post_id, title, body, author, subreddit_channel, created_utc, score_likes, upvote_ratio, num_comments, url, collected_at, media)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  ('reddit', p['id'], p['title'], p['selftext'], p['author'], p['subreddit'], p['created_utc'], p['score'], p.get('upvote_ratio'), p['num_comments'], p['url'], collected_at, p.get('media')))
        post_id = cursor.lastrowid
        post_ids[p['id']] = post_id

        # post features
        txt = f"{p['title']} {p['selftext']}"
        sentiment_score, sentiment_label = compute_sentiment_score_and_label(txt)
        if sentiment_score is not None:
            cursor.execute('''INSERT INTO features (ref_type, ref_id, sentiment_label, sentiment_score, topics, model_version, processed_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      ('post', post_id, sentiment_label, sentiment_score, None, 'vader', datetime.now()))

        # post entities
        entities = extract_entities(txt)
        for entity_type, value in entities:
            try:
                cursor.execute('''INSERT INTO post_entities (post_id, entity_type, value) VALUES (?, ?, ?)''',
                          (post_id, entity_type, value))
            except sqlite3.IntegrityError:
                pass  # Duplicate entry

    # insert comments and features
    for com in comments:
        post_id = post_ids.get(com['post_platform_id'])
        if post_id:
            cursor.execute('''INSERT INTO comments (post_id, platform_comment_id, body, author, created_utc, score_likes, collected_at, media)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (post_id, com['platform_comment_id'], com['body'], com['author'], com['created_utc'], com['score_likes'], collected_at, com['media']))
            comment_id = cursor.lastrowid

            sentiment_score, sentiment_label = compute_sentiment_score_and_label(com['body'])
            if sentiment_score is not None:
                cursor.execute('''INSERT INTO features (ref_type, ref_id, sentiment_label, sentiment_score, topics, model_version, processed_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?)''',
                          ('comment', comment_id, sentiment_label, sentiment_score, None, 'vader', datetime.now()))

    # insert users into tables
    authors = set(p['author'] for p in posts if p['author'] != '[deleted]') | set(com['author'] for com in comments if com['author'] != '[deleted]')
    for author in authors:
        try:
            cursor.execute('''INSERT INTO users (platform, platform_user_id, username, display_name, followers_count, description, collected_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      ('reddit', author, author, author, None, None, collected_at))
        except sqlite3.IntegrityError:
            pass

    # insert prices to tables
    fin_reset = fin.reset_index()
    fin_reset.rename(columns={'Date': 'ts', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    fin_reset['ticker'] = TICKER
    fin_reset['source'] = 'Yahoo Finance'
    fin_reset['ts'] = pd.to_datetime(fin_reset['ts'])
    for _, row in fin_reset.iterrows():
        cursor.execute('''INSERT INTO prices (ticker, ts, open, high, low, close, volume, source)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (row['ticker'], row['ts'], row['open'], row['high'], row['low'], row['close'], row['volume'], row['source']))

    conn.commit()
    conn.close()
    print("Data inserted into database successfully.")

if __name__ == "__main__":
    main()
