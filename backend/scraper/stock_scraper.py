"""
Stock data scraper using yfinance.
Downloads 1 year of OHLCV data for NIFTY 50 stocks.
Smart caching: skips re-scrape if Dataset CSV is fresh (< DATASET_STALE_HOURS old).
"""
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    NIFTY50_TICKERS, START_DATE, END_DATE,
    STOCK_DATA_DIR, DATASET_STALE_HOURS
)


def _csv_path(ticker: str) -> str:
    return os.path.join(STOCK_DATA_DIR, f"{ticker.replace('.', '_')}.csv")


def _is_fresh(csv_path: str) -> bool:
    """Check if a cached CSV exists and is within the staleness window."""
    if not os.path.exists(csv_path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(csv_path))
    age_hours = (datetime.now() - mtime).total_seconds() / 3600
    return age_hours < DATASET_STALE_HOURS


def download_stock_data(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Download OHLCV data for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        if df.empty:
            print(f"  ⚠  No data returned for {ticker}")
            return pd.DataFrame()
        # Clean up columns
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index.name = "Date"
        df = df.dropna()
        return df
    except Exception as e:
        print(f"  ✗  Error downloading {ticker}: {e}")
        return pd.DataFrame()


def scrape_stock(ticker: str, force: bool = False) -> pd.DataFrame:
    """
    Download data for a single ticker with smart caching.
    - If force=True, always re-scrape.
    - Otherwise, skip if the CSV exists and is fresh.
    Returns the DataFrame (from download or cache).
    """
    path = _csv_path(ticker)

    if not force and _is_fresh(path):
        return load_stock_data(ticker)

    df = download_stock_data(ticker)
    if not df.empty:
        df.to_csv(path)
    return df


def scrape_all_stocks(tickers: dict = None, force: bool = False) -> dict:
    """
    Download data for all configured tickers with smart caching.
    Returns dict of {ticker: DataFrame}.
    """
    if tickers is None:
        tickers = NIFTY50_TICKERS

    results = {}
    total = len(tickers)
    scraped = 0
    cached = 0

    print(f"\n{'='*60}")
    print(f"  📊  Processing {total} stocks ({START_DATE} → {END_DATE})")
    print(f"  📁  Dataset folder: {os.path.abspath(STOCK_DATA_DIR)}")
    print(f"  ⏱   Staleness threshold: {DATASET_STALE_HOURS} hours")
    print(f"{'='*60}\n")

    for i, (ticker, name) in enumerate(tickers.items(), 1):
        path = _csv_path(ticker)

        if not force and _is_fresh(path):
            # Use cached data
            df = load_stock_data(ticker)
            if not df.empty:
                results[ticker] = df
                cached += 1
                print(f"  [{i:02d}/{total}] {name:25s} → 💾 cached ({len(df)} rows)")
                continue

        # Need to scrape
        print(f"  [{i:02d}/{total}] {name:25s} → 🌐 scraping...", end=" ")
        df = download_stock_data(ticker)

        if not df.empty:
            df.to_csv(path)
            results[ticker] = df
            scraped += 1
            print(f"✓ {len(df)} rows")
        else:
            print("✗ skipped")

        time.sleep(0.3)

    print(f"\n  ✅  {len(results)}/{total} stocks ready "
          f"(📥 {scraped} scraped, 💾 {cached} from cache)\n")
    return results


def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load previously scraped data from CSV in Dataset folder."""
    path = _csv_path(ticker)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        return df
    return pd.DataFrame()


def load_all_stock_data() -> dict:
    """Load all available stock CSVs."""
    results = {}
    for ticker in NIFTY50_TICKERS:
        df = load_stock_data(ticker)
        if not df.empty:
            results[ticker] = df
    return results


if __name__ == "__main__":
    scrape_all_stocks()
