"""
Feature engineering for stock prediction.
Computes technical indicators, macro-economic features,
fractional differencing, and triple barrier labels.
"""
import os
import pandas as pd
import numpy as np
import ta
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NIFTY50_TICKERS
from scraper.news_scraper import get_average_sentiment


# ─── Fractional Differencing ────────────────────────────────────────────

def _get_weights_ffd(d: float, thres: float = 1e-5, max_len: int = 100) -> np.ndarray:
    """Fixed-width window fractional differencing weights (de Prado)."""
    w = [1.0]
    for k in range(1, max_len):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff(series: pd.Series, d: float = 0.4, thres: float = 1e-5) -> pd.Series:
    """
    Apply fractional differencing of order d to a series.
    d=0 → no differencing, d=1 → standard differencing.
    d≈0.3-0.5 achieves stationarity while retaining memory.
    """
    weights = _get_weights_ffd(d, thres)
    width = len(weights)
    result = pd.Series(index=series.index, dtype=float)

    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1: i + 1].values
        result.iloc[i] = np.dot(weights.T, window)[0]

    return result


def find_min_d(series: pd.Series, d_range=None, p_threshold: float = 0.05) -> float:
    """
    Find minimum d that achieves stationarity (ADF test p < threshold).
    Returns the optimal d value.
    """
    if d_range is None:
        d_range = np.arange(0.1, 1.1, 0.1)

    try:
        from statsmodels.tsa.stattools import adfuller
        for d in d_range:
            diffed = frac_diff(series, d).dropna()
            if len(diffed) < 20:
                continue
            adf_result = adfuller(diffed, maxlag=1)
            if adf_result[1] < p_threshold:
                return round(d, 1)
    except ImportError:
        pass

    return 0.4  # sensible default


def apply_fractional_differencing(df: pd.DataFrame, d: float = 0.4) -> pd.DataFrame:
    """Apply fractional differencing to price columns."""
    df = df.copy()
    price_cols = ["Close", "Open", "High", "Low"]
    for col in price_cols:
        if col in df.columns:
            df[f"{col}_frac"] = frac_diff(df[col], d=d)
    if "Volume" in df.columns:
        df["Volume_frac"] = frac_diff(np.log1p(df["Volume"]), d=d)
    return df


# ─── Triple Barrier Method ──────────────────────────────────────────────

def triple_barrier_labels(
    df: pd.DataFrame,
    pt_multiplier: float = 2.0,
    sl_multiplier: float = 2.0,
    max_holding: int = 5,
) -> pd.Series:
    """
    Triple Barrier Method labeling (de Prado).

    For each day, set three barriers:
      - Upper: Close + ATR * pt_multiplier (profit-take)
      - Lower: Close - ATR * sl_multiplier (stop-loss)
      - Time:  max_holding days ahead

    Label = which barrier is hit first:
      1 → profit-take hit (upside)
      0 → stop-loss or timeout (downside/flat)
    """
    close = df["Close"].values
    atr = df["ATR"].values if "ATR" in df.columns else np.full(len(df), np.nan)

    labels = np.zeros(len(df), dtype=int)

    for i in range(len(df) - max_holding):
        if np.isnan(atr[i]) or atr[i] <= 0:
            # Fallback: simple next-day direction
            if i + 1 < len(df):
                labels[i] = 1 if close[i + 1] > close[i] else 0
            continue

        upper = close[i] + atr[i] * pt_multiplier
        lower = close[i] - atr[i] * sl_multiplier

        label = 0  # default: timeout
        for j in range(1, max_holding + 1):
            idx = i + j
            if idx >= len(df):
                break

            if close[idx] >= upper:
                label = 1  # profit-take hit
                break
            elif close[idx] <= lower:
                label = 0  # stop-loss hit
                break

        labels[i] = label

    # Last max_holding rows: use simple direction as fallback
    for i in range(max(0, len(df) - max_holding), len(df) - 1):
        labels[i] = 1 if close[i + 1] > close[i] else 0

    return pd.Series(labels, index=df.index, name="Target")


# ─── Technical Indicators ────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the OHLCV DataFrame."""
    df = df.copy()

    # ─── Trend ────────────────────────────────────────
    df["SMA_5"]  = ta.trend.sma_indicator(df["Close"], window=5)
    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)

    df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)
    df["EMA_26"] = ta.trend.ema_indicator(df["Close"], window=26)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"] = adx.adx()

    # CCI
    df["CCI"] = ta.trend.cci(df["High"], df["Low"], df["Close"])

    # ─── Momentum ─────────────────────────────────────
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    df["ROC"] = ta.momentum.roc(df["Close"], window=10)
    df["Williams_R"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"])

    # ─── Volatility ───────────────────────────────────
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()

    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])

    # ─── Volume ───────────────────────────────────────
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["Volume_SMA_20"] = ta.trend.sma_indicator(df["Volume"].astype(float), window=20)
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]

    # ─── Price-based ──────────────────────────────────
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["Price_vs_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    df["Price_vs_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]

    df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]

    # Opening gap
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    return df


def add_target(df: pd.DataFrame, method: str = "triple_barrier") -> pd.DataFrame:
    """
    Add target labels.
    method='triple_barrier': Triple Barrier Method (de Prado)
    method='simple': Original next-day direction
    """
    df = df.copy()

    if method == "triple_barrier" and "ATR" in df.columns:
        df["Target"] = triple_barrier_labels(df)
    else:
        df["Next_Close"] = df["Close"].shift(-1)
        df["Next_Return"] = (df["Next_Close"] - df["Close"]) / df["Close"]
        df["Target"] = (df["Next_Return"] > 0).astype(int)

    return df


def prepare_features(
    df: pd.DataFrame,
    ticker: str = None,
    use_macro: bool = True,
    use_frac_diff: bool = True,
    target_method: str = "triple_barrier",
) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    1. Add technical indicators
    2. Triple Barrier or simple target
    3. Fractional differencing on price columns
    4. Merge macro-economic features
    5. Add sentiment score
    6. Drop NaN rows
    """
    df = add_technical_indicators(df)
    df = add_target(df, method=target_method)

    # Fractional differencing
    if use_frac_diff:
        df = apply_fractional_differencing(df, d=0.4)

    # Macro features
    if use_macro:
        try:
            from features.macro_features import merge_macro_features
            df = merge_macro_features(df)
        except Exception:
            pass

    # Sentiment
    if ticker:
        sentiment = get_average_sentiment(ticker)
    else:
        sentiment = 0.0
    df["Sentiment"] = sentiment

    df = df.dropna()
    return df


def get_feature_columns() -> list:
    """Return the list of feature column names used for training."""
    base_features = [
        # Trend
        "SMA_5", "SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "MACD_Hist", "ADX", "CCI",
        # Momentum
        "RSI", "Stoch_K", "Stoch_D", "ROC", "Williams_R",
        # Volatility
        "BB_Upper", "BB_Lower", "BB_Width", "ATR",
        # Volume
        "OBV", "Volume_SMA_20", "Volume_Ratio",
        # Price
        "Return_1d", "Return_5d", "Return_10d", "Log_Return",
        "Price_vs_SMA20", "Price_vs_SMA50", "HL_Range", "Gap",
        # Fractional differencing
        "Close_frac", "Open_frac", "High_frac", "Low_frac", "Volume_frac",
        # Sentiment
        "Sentiment",
        # Raw OHLCV
        "Open", "High", "Low", "Close", "Volume",
    ]
    return base_features


def get_macro_columns() -> list:
    """Return macro feature columns if available."""
    try:
        from features.macro_features import get_macro_feature_columns
        return get_macro_feature_columns()
    except ImportError:
        return []


def get_all_feature_columns() -> list:
    """Return ALL feature columns (base + macro)."""
    return get_feature_columns() + get_macro_columns()


if __name__ == "__main__":
    from scraper.stock_scraper import load_stock_data

    df = load_stock_data("RELIANCE.NS")
    if not df.empty:
        featured = prepare_features(df, "RELIANCE.NS")
        print(f"\nFeatures shape: {featured.shape}")
        print(f"Target distribution:\n{featured['Target'].value_counts()}")

        # Check stationarity
        try:
            from statsmodels.tsa.stattools import adfuller
            for col in ["Close", "Close_frac"]:
                if col in featured.columns:
                    adf = adfuller(featured[col].dropna())
                    print(f"  {col:15s} → ADF stat={adf[0]:.4f}, p={adf[1]:.6f} {'✓ stationary' if adf[1] < 0.05 else '✗ non-stationary'}")
        except ImportError:
            pass
