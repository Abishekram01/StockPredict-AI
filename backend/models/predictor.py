"""
Prediction engine.
Loads the best saved model for each stock and generates BUY/HOLD/SELL signals.
Uses the feature columns stored in each model's metadata for compatibility.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NIFTY50_TICKERS, MODEL_DIR, LSTM_LOOKBACK
from features.feature_engineer import prepare_features, get_feature_columns
from scraper.stock_scraper import load_stock_data
from scraper.news_scraper import get_average_sentiment, load_cached_news


def _load_model(ticker: str):
    """Load the best saved model, scaler, and metadata for a ticker."""
    model_dir = os.path.join(MODEL_DIR, ticker.replace(".", "_"))
    meta_path = os.path.join(model_dir, "metadata.json")
    scaler_path = os.path.join(model_dir, "scaler.joblib")

    if not os.path.exists(meta_path) or not os.path.exists(scaler_path):
        return None, None, None

    try:
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        scaler = joblib.load(scaler_path)
        best_model_name = metadata.get("best_model")
        if not best_model_name:
            return None, None, None

        if best_model_name == "LSTM":
            keras_path = os.path.join(model_dir, "best_model.keras")
            if not os.path.exists(keras_path):
                return None, None, None
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            model = tf.keras.models.load_model(keras_path)
        else:
            joblib_path = os.path.join(model_dir, "best_model.joblib")
            if not os.path.exists(joblib_path):
                return None, None, None
            model = joblib.load(joblib_path)

        return model, scaler, metadata
    except Exception:
        return None, None, None


def predict_stock(ticker: str) -> dict:
    """
    Generate prediction for a single stock.
    Uses feature columns from the model's metadata for compatibility.
    """
    model, scaler, metadata = _load_model(ticker)
    if model is None:
        return {"ticker": ticker, "signal": "NO_DATA", "confidence": 0}

    df = load_stock_data(ticker)
    if df.empty:
        return {"ticker": ticker, "signal": "NO_DATA", "confidence": 0}

    # Determine how many features the scaler/model actually expects
    n_expected = getattr(scaler, 'n_features_in_', None)
    model_feature_cols = metadata.get("feature_columns", get_feature_columns())

    featured_df = prepare_features(df, ticker)
    if featured_df.empty:
        return {"ticker": ticker, "signal": "NO_DATA", "confidence": 0}

    # Filter to columns that exist in the DataFrame
    available_cols = [c for c in model_feature_cols if c in featured_df.columns]

    # If scaler expects fewer features than metadata says, trust the scaler
    if n_expected is not None and len(available_cols) > n_expected:
        available_cols = available_cols[:n_expected]
    elif n_expected is not None and len(available_cols) < n_expected:
        # Pad with zeros for missing columns
        for col in model_feature_cols:
            if col not in featured_df.columns:
                featured_df[col] = 0.0
        available_cols = [c for c in model_feature_cols if c in featured_df.columns][:n_expected]

    latest = featured_df[available_cols].iloc[-1:].values
    latest_scaled = scaler.transform(latest)

    best_model_name = metadata["best_model"]

    if best_model_name == "Stacking":
        try:
            model_dir = os.path.join(MODEL_DIR, ticker.replace(".", "_"))
            base_models = {}
            for name in model.get("base_model_names", []):
                safe = name.lower().replace(" ", "_")
                path = os.path.join(model_dir, f"model_{safe}.joblib")
                if os.path.exists(path):
                    base_models[name] = joblib.load(path)

            from models.trainer import predict_with_stacking
            prob = float(predict_with_stacking(model, base_models, latest_scaled)[0])
        except Exception:
            prob = 0.5
    elif best_model_name == "LSTM":
        lookback = min(LSTM_LOOKBACK, len(featured_df) - 1)
        all_data = featured_df[available_cols].values
        all_scaled = scaler.transform(all_data)
        sequence = all_scaled[-lookback:].reshape(1, lookback, len(available_cols))
        prob = float(model.predict(sequence, verbose=0)[0][0])
    else:
        prob = float(model.predict_proba(latest_scaled)[0][1])

    # Signal with sentiment blending
    sentiment = get_average_sentiment(ticker)
    adjusted_prob = prob * 0.85 + (sentiment + 1) / 2 * 0.15

    if adjusted_prob >= 0.6:
        signal = "BUY"
    elif adjusted_prob <= 0.4:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = abs(adjusted_prob - 0.5) * 200

    current_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2])
    price_change = round((current_price - prev_price) / prev_price * 100, 2)
    company_name = NIFTY50_TICKERS.get(ticker, ticker)

    return {
        "ticker": ticker,
        "company": company_name,
        "signal": signal,
        "confidence": round(confidence, 1),
        "probability": round(adjusted_prob, 4),
        "current_price": round(current_price, 2),
        "price_change_pct": price_change,
        "sentiment_score": sentiment,
        "best_model": best_model_name,
        "best_auc": metadata.get("best_val_auc", metadata.get("best_auc", 0)),
        "best_test_auc": metadata.get("best_test_auc", 0),
    }


def predict_all_stocks(tickers: dict = None) -> list:
    """Generate predictions for all stocks."""
    if tickers is None:
        tickers = NIFTY50_TICKERS

    predictions = []
    total = len(tickers)

    print(f"\n{'='*60}")
    print(f"  🔮  Generating predictions for {total} stocks")
    print(f"{'='*60}\n")

    for i, (ticker, name) in enumerate(tickers.items(), 1):
        pred = predict_stock(ticker)
        if pred["signal"] != "NO_DATA":
            predictions.append(pred)
            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
            print(f"  [{i:02d}/{total}] {name:25s} → {emoji.get(pred['signal'], '⚪')} {pred['signal']:4s} ({pred['confidence']:.0f}%)")
        else:
            print(f"  [{i:02d}/{total}] {name:25s} → ⚪ NO DATA")

    signal_order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    predictions.sort(key=lambda x: (signal_order.get(x["signal"], 3), -x["confidence"]))

    print(f"\n  ✅  Generated {len(predictions)} predictions\n")
    return predictions


def get_stock_details(ticker: str) -> dict:
    """Get detailed info for a single stock."""
    prediction = predict_stock(ticker)

    df = load_stock_data(ticker)
    history = []
    if not df.empty:
        recent = df.tail(90)
        for date, row in recent.iterrows():
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

    model_dir = os.path.join(MODEL_DIR, ticker.replace(".", "_"))
    meta_path = os.path.join(model_dir, "metadata.json")
    model_metrics = {}
    test_metrics = {}
    best_model_on_test = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            model_metrics = meta.get("val_metrics", meta.get("all_metrics", {}))
            test_metrics = meta.get("test_metrics", {})
            best_model_on_test = meta.get("best_model_on_test", None)

    news = load_cached_news(ticker)

    return {
        "prediction": prediction,
        "history": history,
        "model_metrics": model_metrics,
        "test_metrics": test_metrics,
        "best_model_on_test": best_model_on_test,
        "news": news[:10],
    }


if __name__ == "__main__":
    predictions = predict_all_stocks()
    if predictions:
        print("\n  Top 5 BUY signals:")
        for p in [x for x in predictions if x["signal"] == "BUY"][:5]:
            print(f"    {p['company']:25s} → {p['signal']} ({p['confidence']:.0f}%)")
