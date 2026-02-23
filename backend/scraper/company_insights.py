"""
Company Insights Scraper.
Fetches comprehensive company data from yfinance + Google News for
the top BUY/SELL stock picks. Provides fundamental data, financials,
analyst recommendations, and enriched news to help users decide.
"""
import os
import json
import sys
import time
from datetime import datetime

import yfinance as yf
import feedparser
from urllib.parse import quote
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NIFTY50_TICKERS, NEWS_CACHE_DIR

analyzer = SentimentIntensityAnalyzer()
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"


def _clean_html(text):
    return re.sub(r"<[^>]+>", "", text).strip()


def _safe_get(info, key, default="N/A"):
    """Safely get a value from yfinance info dict."""
    val = info.get(key, default)
    if val is None:
        return default
    return val


def fetch_detailed_news(company_name, ticker, max_articles=20):
    """Fetch detailed news for a company with multiple search queries."""
    all_articles = []
    queries = [
        f"{company_name} stock India",
        f"{company_name} quarterly results 2025 2026",
        f"{company_name} business outlook India",
    ]

    seen_headlines = set()
    for query in queries:
        url = GOOGLE_NEWS_RSS.format(query=quote(query))
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                headline = _clean_html(entry.get("title", ""))
                if not headline or headline in seen_headlines:
                    continue
                seen_headlines.add(headline)

                scores = analyzer.polarity_scores(headline)
                compound = scores["compound"]
                label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")

                try:
                    pub_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d %H:%M")
                except:
                    pub_date = "Unknown"

                all_articles.append({
                    "headline": headline,
                    "sentiment_score": round(compound, 4),
                    "sentiment_label": label,
                    "url": entry.get("link", ""),
                    "source": entry.get("source", {}).get("title", "Unknown"),
                    "published_at": pub_date,
                })
            time.sleep(0.3)
        except Exception as e:
            pass

    # Sort by sentiment score (most positive first for BUY, most negative first for SELL)
    all_articles.sort(key=lambda x: -abs(x["sentiment_score"]))
    return all_articles[:25]


def get_company_insights(ticker: str) -> dict:
    """
    Fetch comprehensive company insights from yfinance.
    Returns fundamentals, financials, analyst data, and detailed news.
    """
    company_name = NIFTY50_TICKERS.get(ticker, ticker.replace(".NS", ""))

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception as e:
        print(f"  ✗  Error fetching info for {ticker}: {e}")
        info = {}

    # ─── Company Profile ─────────────────────────────────────────
    profile = {
        "name": _safe_get(info, "longName", company_name),
        "sector": _safe_get(info, "sector"),
        "industry": _safe_get(info, "industry"),
        "description": _safe_get(info, "longBusinessSummary", "No description available."),
        "website": _safe_get(info, "website"),
        "employees": _safe_get(info, "fullTimeEmployees", 0),
        "city": _safe_get(info, "city"),
        "country": _safe_get(info, "country", "India"),
    }

    # ─── Key Financials ──────────────────────────────────────────
    financials = {
        "market_cap": _safe_get(info, "marketCap", 0),
        "market_cap_formatted": _format_large_number(_safe_get(info, "marketCap", 0)),
        "enterprise_value": _safe_get(info, "enterpriseValue", 0),
        "revenue": _safe_get(info, "totalRevenue", 0),
        "revenue_formatted": _format_large_number(_safe_get(info, "totalRevenue", 0)),
        "net_income": _safe_get(info, "netIncomeToCommon", 0),
        "net_income_formatted": _format_large_number(_safe_get(info, "netIncomeToCommon", 0)),
        "profit_margin": _format_pct(_safe_get(info, "profitMargins", 0)),
        "operating_margin": _format_pct(_safe_get(info, "operatingMargins", 0)),
        "gross_margin": _format_pct(_safe_get(info, "grossMargins", 0)),
        "ebitda": _format_large_number(_safe_get(info, "ebitda", 0)),
        "free_cash_flow": _format_large_number(_safe_get(info, "freeCashflow", 0)),
        "debt_to_equity": _safe_get(info, "debtToEquity", 0),
        "current_ratio": _safe_get(info, "currentRatio", 0),
        "return_on_equity": _format_pct(_safe_get(info, "returnOnEquity", 0)),
        "return_on_assets": _format_pct(_safe_get(info, "returnOnAssets", 0)),
    }

    # ─── Valuation Metrics ───────────────────────────────────────
    valuation = {
        "pe_ratio": round(_safe_get(info, "trailingPE", 0), 2) if isinstance(_safe_get(info, "trailingPE", 0), (int, float)) else "N/A",
        "forward_pe": round(_safe_get(info, "forwardPE", 0), 2) if isinstance(_safe_get(info, "forwardPE", 0), (int, float)) else "N/A",
        "peg_ratio": _safe_get(info, "pegRatio", "N/A"),
        "price_to_book": round(_safe_get(info, "priceToBook", 0), 2) if isinstance(_safe_get(info, "priceToBook", 0), (int, float)) else "N/A",
        "price_to_sales": round(_safe_get(info, "priceToSalesTrailing12Months", 0), 2) if isinstance(_safe_get(info, "priceToSalesTrailing12Months", 0), (int, float)) else "N/A",
        "ev_to_ebitda": round(_safe_get(info, "enterpriseToEbitda", 0), 2) if isinstance(_safe_get(info, "enterpriseToEbitda", 0), (int, float)) else "N/A",
        "ev_to_revenue": round(_safe_get(info, "enterpriseToRevenue", 0), 2) if isinstance(_safe_get(info, "enterpriseToRevenue", 0), (int, float)) else "N/A",
    }

    # ─── Dividend Info ───────────────────────────────────────────
    dividend = {
        "dividend_yield": _format_pct(_safe_get(info, "dividendYield", 0)),
        "dividend_rate": _safe_get(info, "dividendRate", 0),
        "payout_ratio": _format_pct(_safe_get(info, "payoutRatio", 0)),
        "ex_dividend_date": _safe_get(info, "exDividendDate", "N/A"),
    }

    # ─── Price Targets & Analyst Opinions ────────────────────────
    analyst = {
        "target_high": _safe_get(info, "targetHighPrice", 0),
        "target_low": _safe_get(info, "targetLowPrice", 0),
        "target_mean": _safe_get(info, "targetMeanPrice", 0),
        "target_median": _safe_get(info, "targetMedianPrice", 0),
        "recommendation": _safe_get(info, "recommendationKey", "N/A"),
        "recommendation_mean": _safe_get(info, "recommendationMean", 0),
        "num_analysts": _safe_get(info, "numberOfAnalystOpinions", 0),
    }

    # ─── Current Price Info ──────────────────────────────────────
    price_info = {
        "current_price": _safe_get(info, "currentPrice", _safe_get(info, "regularMarketPrice", 0)),
        "previous_close": _safe_get(info, "previousClose", 0),
        "open": _safe_get(info, "open", _safe_get(info, "regularMarketOpen", 0)),
        "day_high": _safe_get(info, "dayHigh", _safe_get(info, "regularMarketDayHigh", 0)),
        "day_low": _safe_get(info, "dayLow", _safe_get(info, "regularMarketDayLow", 0)),
        "fifty_two_week_high": _safe_get(info, "fiftyTwoWeekHigh", 0),
        "fifty_two_week_low": _safe_get(info, "fiftyTwoWeekLow", 0),
        "fifty_day_avg": round(_safe_get(info, "fiftyDayAverage", 0), 2) if isinstance(_safe_get(info, "fiftyDayAverage", 0), (int, float)) else 0,
        "two_hundred_day_avg": round(_safe_get(info, "twoHundredDayAverage", 0), 2) if isinstance(_safe_get(info, "twoHundredDayAverage", 0), (int, float)) else 0,
        "volume": _safe_get(info, "volume", _safe_get(info, "regularMarketVolume", 0)),
        "avg_volume": _safe_get(info, "averageVolume", 0),
        "beta": round(_safe_get(info, "beta", 0), 3) if isinstance(_safe_get(info, "beta", 0), (int, float)) else "N/A",
    }

    # ─── Detailed News ───────────────────────────────────────────
    news = fetch_detailed_news(company_name, ticker)

    # News sentiment summary
    if news:
        pos = sum(1 for n in news if n["sentiment_label"] == "positive")
        neg = sum(1 for n in news if n["sentiment_label"] == "negative")
        neu = sum(1 for n in news if n["sentiment_label"] == "neutral")
        avg_sent = round(sum(n["sentiment_score"] for n in news) / len(news), 4)
    else:
        pos, neg, neu, avg_sent = 0, 0, 0, 0.0

    news_summary = {
        "total_articles": len(news),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "average_sentiment": avg_sent,
        "sentiment_verdict": "Bullish" if avg_sent > 0.05 else ("Bearish" if avg_sent < -0.05 else "Neutral"),
    }

    return {
        "ticker": ticker,
        "profile": profile,
        "financials": financials,
        "valuation": valuation,
        "dividend": dividend,
        "analyst": analyst,
        "price_info": price_info,
        "news": news,
        "news_summary": news_summary,
    }


def _format_large_number(val):
    """Format large numbers to human-readable ₹Cr format."""
    if not isinstance(val, (int, float)) or val == 0:
        return "N/A"
    cr = val / 1e7  # Convert to Crores
    if cr >= 100000:
        return f"₹{cr/100000:.1f}L Cr"
    elif cr >= 1000:
        return f"₹{cr/1000:.1f}K Cr"
    elif cr >= 1:
        return f"₹{cr:.0f} Cr"
    else:
        return f"₹{val:,.0f}"


def _format_pct(val):
    """Format percentage values."""
    if not isinstance(val, (int, float)) or val == 0:
        return "N/A"
    return f"{val * 100:.1f}%"


if __name__ == "__main__":
    # Test with Reliance
    insights = get_company_insights("RELIANCE.NS")
    print(json.dumps(insights, indent=2, default=str))
