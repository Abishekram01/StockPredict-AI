"""
News scraper using Google News RSS feeds.
Fetches latest headlines for Indian stocks and computes sentiment scores.
Supports industry/domain-aware news for custom stock searches.
Smart caching: skips re-fetch if cache is fresh (< NEWS_STALE_HOURS old).
"""
import os
import json
import time
import re
from datetime import datetime
from urllib.parse import quote
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NIFTY50_TICKERS, NEWS_CACHE_DIR, NEWS_STALE_HOURS

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Google News RSS base URL
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"


def _clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _cache_path(ticker: str) -> str:
    return os.path.join(NEWS_CACHE_DIR, f"{ticker.replace('.', '_')}_news.json")


def _is_news_fresh(ticker: str) -> bool:
    """Check if cached news exists and is within the staleness window."""
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    age_hours = (datetime.now() - mtime).total_seconds() / 3600
    return age_hours < NEWS_STALE_HOURS


def _fetch_rss(query: str, max_articles: int = 15) -> list:
    """Fetch and score articles from a single RSS query."""
    url = GOOGLE_NEWS_RSS.format(query=quote(query))
    articles = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_articles]:
            headline = _clean_html(entry.get("title", ""))
            if not headline:
                continue
            scores = analyzer.polarity_scores(headline)
            compound = scores["compound"]

            if compound >= 0.05:
                sentiment_label = "positive"
            elif compound <= -0.05:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            try:
                pub_date = datetime(*entry.published_parsed[:6]).isoformat()
            except (AttributeError, TypeError):
                pub_date = datetime.now().isoformat()

            articles.append({
                "headline": headline,
                "sentiment_score": round(compound, 4),
                "sentiment_label": sentiment_label,
                "url": entry.get("link", ""),
                "source": entry.get("source", {}).get("title", "Unknown"),
                "published_at": pub_date,
            })
    except Exception:
        pass
    return articles


def fetch_news_for_stock(ticker: str, company_name: str, max_articles: int = 15) -> list:
    """
    Fetch latest news articles for a stock from Google News RSS.
    Returns list of dicts with headline, sentiment, url, etc.
    """
    query = f"{company_name} stock NSE India"
    articles = _fetch_rss(query, max_articles)

    # Tag each article with ticker/company
    for a in articles:
        a["ticker"] = ticker
        a["company"] = company_name
        a["news_type"] = "company"

    return articles


def fetch_industry_news(
    company_name: str,
    sector: str = None,
    industry: str = None,
    ticker: str = "",
    max_articles: int = 10,
) -> list:
    """
    Fetch news about the company's industry/sector/domain.
    These are macro-level news that can affect the stock indirectly:
      - Regulatory changes, policy updates
      - Sector earnings, peer company results
      - Global supply chain, commodity prices, etc.
    Returns list of articles tagged with news_type='industry'.
    """
    queries = []
    if sector and sector != "N/A":
        queries.append(f"{sector} India stock market")
        queries.append(f"{sector} industry news India 2026")
    if industry and industry != "N/A" and industry != sector:
        queries.append(f"{industry} India market outlook")
    # Always add a domain-specific query using the company name
    queries.append(f"{company_name} sector industry outlook India")

    seen_headlines = set()
    all_articles = []

    for query in queries:
        arts = _fetch_rss(query, max_articles)
        for a in arts:
            if a["headline"] not in seen_headlines:
                seen_headlines.add(a["headline"])
                a["ticker"] = ticker
                a["company"] = company_name
                a["news_type"] = "industry"
                a["query"] = query
                all_articles.append(a)
        time.sleep(0.3)

    return all_articles


def fetch_all_news(tickers: dict = None, force: bool = False) -> dict:
    """
    Fetch news for all configured stocks with smart caching.
    Returns dict of {ticker: [articles]}.
    """
    if tickers is None:
        tickers = NIFTY50_TICKERS

    all_news = {}
    total = len(tickers)
    fetched = 0
    cached = 0

    print(f"\n{'='*60}")
    print(f"  📰  Processing news for {total} stocks")
    print(f"  ⏱   Staleness threshold: {NEWS_STALE_HOURS} hours")
    print(f"{'='*60}\n")

    for i, (ticker, name) in enumerate(tickers.items(), 1):
        if not force and _is_news_fresh(ticker):
            articles = load_cached_news(ticker)
            all_news[ticker] = articles
            cached += 1
            print(f"  [{i:02d}/{total}] {name:25s} → 💾 cached ({len(articles)} articles)")
            continue

        print(f"  [{i:02d}/{total}] {name:25s} → 🌐 fetching...", end=" ")
        articles = fetch_news_for_stock(ticker, name)
        all_news[ticker] = articles
        fetched += 1
        print(f"✓ {len(articles)} articles")

        # Cache to file
        with open(_cache_path(ticker), "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    print(f"\n  ✅  {total} stocks processed "
          f"(📥 {fetched} fetched, 💾 {cached} from cache)\n")
    return all_news


def load_cached_news(ticker: str) -> list:
    """Load cached news articles for a ticker."""
    path = _cache_path(ticker)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def get_average_sentiment(ticker: str) -> float:
    """Get average sentiment score for a ticker from cached news."""
    articles = load_cached_news(ticker)
    if not articles:
        return 0.0
    scores = [a["sentiment_score"] for a in articles]
    return round(sum(scores) / len(scores), 4)


if __name__ == "__main__":
    fetch_all_news()
