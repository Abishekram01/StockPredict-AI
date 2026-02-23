"""
Macro-economic feature fetcher.
Downloads and caches India VIX, USD/INR, US 10Y, sector indices, Gold
to provide external signals for stock prediction.
"""
import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import STOCK_DATA_DIR, START_DATE, END_DATE, DATASET_STALE_HOURS

MACRO_TICKERS = {
    "INDIAVIX":  "^INDIAVIX",
    "USDINR":    "USDINR=X",
    "US10Y":     "^TNX",
    "NIFTY50":   "^NSEI",
    "NIFTYIT":   "^CNXIT",
    "NIFTYBANK": "^NSEBANK",
    "GOLD":      "GC=F",
}

MACRO_CSV = os.path.join(STOCK_DATA_DIR, "_macro_data.csv")


def _is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime).total_seconds() / 3600 < DATASET_STALE_HOURS


def fetch_macro_data(force: bool = False) -> pd.DataFrame:
    """Download macro indicators and cache to Dataset folder."""
    if not force and _is_fresh(MACRO_CSV):
        return pd.read_csv(MACRO_CSV, index_col="Date", parse_dates=True)

    frames = {}
    for label, ticker in MACRO_TICKERS.items():
        try:
            df = yf.Ticker(ticker).history(start=START_DATE, end=END_DATE)
            if not df.empty:
                frames[label] = df["Close"].rename(label)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    macro = pd.concat(frames.values(), axis=1).sort_index()
    macro.index.name = "Date"
    macro = macro.ffill().bfill()  # fill holidays

    # Derive features
    for col in macro.columns:
        macro[f"{col}_ret1d"] = macro[col].pct_change(1)
        macro[f"{col}_ret5d"] = macro[col].pct_change(5)
        macro[f"{col}_mom10d"] = macro[col].pct_change(10)

    macro = macro.dropna()
    macro.to_csv(MACRO_CSV)
    return macro


def get_macro_feature_columns() -> list:
    """Return the list of macro feature columns."""
    cols = []
    for label in MACRO_TICKERS:
        cols.extend([
            label,
            f"{label}_ret1d",
            f"{label}_ret5d",
            f"{label}_mom10d",
        ])
    return cols


def merge_macro_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Merge macro features onto a stock DataFrame by date index."""
    macro = fetch_macro_data()
    if macro.empty:
        return stock_df

    macro_cols = get_macro_feature_columns()
    available = [c for c in macro_cols if c in macro.columns]
    if not available:
        return stock_df

    merged = stock_df.join(macro[available], how="left")
    merged[available] = merged[available].ffill().bfill()
    return merged


if __name__ == "__main__":
    df = fetch_macro_data(force=True)
    print(f"Macro data: {df.shape}")
    print(df.tail())
