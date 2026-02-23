"""
On-demand stock analysis pipeline.
Accepts any ticker, scrapes data (with Dataset caching), trains models,
evaluates, predicts, and returns comprehensive results with company insights
and industry-aware news — all in one call.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    START_DATE, END_DATE, MODEL_DIR, RANDOM_STATE,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LOOKBACK,
    STOCK_DATA_DIR, NEWS_CACHE_DIR
)
from scraper.stock_scraper import scrape_stock, load_stock_data
from scraper.news_scraper import (
    fetch_news_for_stock, get_average_sentiment,
    fetch_industry_news, _cache_path
)
from scraper.company_insights import get_company_insights
from features.feature_engineer import prepare_features, get_feature_columns
from models.trainer import (
    MODEL_TRAINERS, _evaluate, _three_way_split,
    _evaluate_model_on_set, _get_model_dir
)

from sklearn.preprocessing import StandardScaler
import joblib


def analyze_custom_stock(ticker: str, company_name: str = None) -> dict:
    """
    Full on-demand pipeline for any stock:
      1. Scrape 1-year OHLCV (cached if fresh)
      2. Fetch & score company news + industry/domain news
      3. Engineer features
      4. Train 6 models (60/20/20 split)
      5. Evaluate all models on held-out test set
      6. Pick the best model
      7. Generate BUY/HOLD/SELL signal
      8. Fetch company insights
    Returns a single dict with everything the frontend needs.
    """
    # Normalize ticker
    if not ticker.endswith(".NS"):
        ticker_ns = ticker.upper() + ".NS"
    else:
        ticker_ns = ticker.upper()

    ticker_clean = ticker_ns.replace(".NS", "")

    if company_name is None:
        company_name = ticker_clean

    result = {
        "ticker": ticker_ns,
        "company": company_name,
        "status": "processing",
        "steps": [],
    }

    # ─── Step 1: Scrape (with smart caching) ─────────────────────
    result["steps"].append("Scraping stock data...")
    df = scrape_stock(ticker_ns)
    if df.empty:
        result["status"] = "error"
        result["error"] = f"No data found for ticker '{ticker_ns}'. Please check the ticker symbol."
        return result

    csv_path = os.path.join(STOCK_DATA_DIR, f"{ticker_ns.replace('.', '_')}.csv")
    is_cached = os.path.exists(csv_path)
    result["steps"].append(
        f"{'Loaded from Dataset cache' if is_cached else 'Scraped'}: {len(df)} days of data"
    )
    result["data_rows"] = len(df)

    # ─── Step 2: Fetch Company News + Industry News ──────────────
    result["steps"].append("Fetching news & sentiment...")
    news_articles = fetch_news_for_stock(ticker_ns, company_name)

    # Cache company news
    cache_path = _cache_path(ticker_ns)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(news_articles, f, indent=2, ensure_ascii=False)

    # Fetch industry/domain news — get sector from yfinance first
    sector = None
    industry = None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker_ns).info
        sector = info.get("sector", None)
        industry = info.get("industry", None)
        if company_name == ticker_clean and info.get("longName"):
            company_name = info["longName"]
            result["company"] = company_name
    except Exception:
        pass

    industry_articles = fetch_industry_news(
        company_name, sector=sector, industry=industry, ticker=ticker_ns
    )

    result["steps"].append(
        f"Fetched {len(news_articles)} company + {len(industry_articles)} industry news articles"
    )

    # ─── Step 3: Feature Engineering ─────────────────────────────
    result["steps"].append("Engineering features...")
    feature_cols = get_feature_columns()
    featured_df = prepare_features(df, ticker_ns)

    if len(featured_df) < 100:
        result["status"] = "error"
        result["error"] = f"Not enough data for training ({len(featured_df)} rows, need at least 100)."
        return result

    # Filter to only columns that exist in the DataFrame
    available_cols = [c for c in feature_cols if c in featured_df.columns]
    if len(available_cols) < 5:
        result["status"] = "error"
        result["error"] = "Not enough features could be generated from the data."
        return result

    result["steps"].append(f"Generated {len(available_cols)} features for {len(featured_df)} samples")

    # ─── Step 4: Train & Evaluate ────────────────────────────────
    result["steps"].append("Training ML models...")

    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_end, val_end, test_dates) = _three_way_split(featured_df, available_cols)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    val_results = {}
    test_results = {}
    trained_models = {}
    best_model = None
    best_model_name = None
    best_val_auc = -1

    for model_name, trainer_fn in MODEL_TRAINERS.items():
        try:
            model, val_metrics = trainer_fn(X_train_scaled, y_train, X_val_scaled, y_val)
            val_results[model_name] = val_metrics
            trained_models[model_name] = model

            test_metrics = _evaluate_model_on_set(model, model_name, X_test_scaled, y_test)
            test_results[model_name] = test_metrics

            auc = val_metrics["roc_auc"]
            if model is not None and auc > best_val_auc:
                best_val_auc = auc
                best_model = model
                best_model_name = model_name
        except Exception as e:
            zero = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0}
            val_results[model_name] = zero
            test_results[model_name] = zero

    # Fallback: if no best model selected (all AUCs were 0.5/nan), pick by accuracy
    if best_model is None and trained_models:
        best_acc = -1
        for name, model in trained_models.items():
            if model is not None:
                acc = val_results.get(name, {}).get("accuracy", 0)
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_model_name = name
                    best_val_auc = val_results.get(name, {}).get("roc_auc", 0.5)

    # Try stacking ensemble
    try:
        from models.trainer import train_stacking_ensemble, predict_with_stacking
        stacking_model, stack_val = train_stacking_ensemble(
            X_train_scaled, y_train, X_val_scaled, y_val, trained_models
        )
        if stacking_model is not None:
            val_results["Stacking"] = stack_val
            stack_test_prob = predict_with_stacking(stacking_model, trained_models, X_test_scaled)
            stack_test_pred = (stack_test_prob > 0.5).astype(int)
            stack_test = _evaluate(y_test, stack_test_pred, stack_test_prob)
            test_results["Stacking"] = stack_test
            if stack_val["roc_auc"] > best_val_auc:
                best_val_auc = stack_val["roc_auc"]
                best_model_name = "Stacking"
                best_model = stacking_model
    except Exception:
        pass

    result["steps"].append(f"Trained {len(val_results)} models — best: {best_model_name}")

    # Save models
    model_dir = _get_model_dir(ticker_ns)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

    for model_name, model in trained_models.items():
        if model is None:
            continue
        safe_name = model_name.lower().replace(" ", "_")
        if model_name == "LSTM":
            model.save(os.path.join(model_dir, f"model_{safe_name}.keras"))
        else:
            joblib.dump(model, os.path.join(model_dir, f"model_{safe_name}.joblib"))

    if best_model is not None:
        if best_model_name == "LSTM":
            best_model.save(os.path.join(model_dir, "best_model.keras"))
        elif best_model_name == "Stacking":
            joblib.dump(best_model, os.path.join(model_dir, "best_model.joblib"))
        else:
            joblib.dump(best_model, os.path.join(model_dir, "best_model.joblib"))

    # Save metadata with actual available columns (not the full list)
    meta = {
        "ticker": ticker_ns,
        "best_model": best_model_name,
        "best_val_auc": best_val_auc,
        "best_test_auc": test_results.get(best_model_name, {}).get("roc_auc", 0),
        "val_metrics": val_results,
        "test_metrics": test_results,
        "all_metrics": val_results,
        "feature_columns": available_cols,
        "data_rows": len(featured_df),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "has_stacking": "Stacking" in val_results,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ─── Step 5: Generate Prediction ─────────────────────────────
    result["steps"].append("Generating prediction...")

    latest = featured_df[available_cols].iloc[-1:].values
    latest_scaled = scaler.transform(latest)

    if best_model_name == "Stacking" and best_model is not None:
        try:
            prob = float(predict_with_stacking(best_model, trained_models, latest_scaled)[0])
        except Exception:
            prob = 0.5
    elif best_model_name == "LSTM" and best_model is not None:
        lookback = min(LSTM_LOOKBACK, len(featured_df) - 1)
        all_data = featured_df[available_cols].values
        all_scaled = scaler.transform(all_data)
        sequence = all_scaled[-lookback:].reshape(1, lookback, len(available_cols))
        prob = float(best_model.predict(sequence, verbose=0)[0][0])
    elif best_model is not None:
        prob = float(best_model.predict_proba(latest_scaled)[0][1])
    else:
        prob = 0.5

    sentiment = get_average_sentiment(ticker_ns)
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

    prediction = {
        "ticker": ticker_ns,
        "company": company_name,
        "signal": signal,
        "confidence": round(confidence, 1),
        "probability": round(adjusted_prob, 4),
        "current_price": round(current_price, 2),
        "price_change_pct": price_change,
        "sentiment_score": sentiment,
        "best_model": best_model_name,
        "best_val_auc": round(best_val_auc, 4),
        "best_test_auc": round(test_results.get(best_model_name, {}).get("roc_auc", 0), 4),
    }

    # ─── Step 6: Historical data for chart ───────────────────────
    history = []
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

    # ─── Step 7: Company Insights ────────────────────────────────
    result["steps"].append("Fetching company insights...")
    try:
        insights = get_company_insights(ticker_ns)
    except Exception:
        insights = None

    # ─── Final Result ────────────────────────────────────────────
    result["status"] = "success"
    result["prediction"] = prediction
    result["val_metrics"] = val_results
    result["test_metrics"] = test_results
    result["history"] = history
    result["insights"] = insights
    result["news"] = news_articles[:15]
    result["industry_news"] = industry_articles[:15]
    result["sector"] = sector
    result["industry"] = industry
    result["split_info"] = {
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test),
        "total": len(featured_df),
    }
    result["steps"].append("Analysis complete!")

    return result


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "WIPRO.NS"
    result = analyze_custom_stock(ticker)
    print(json.dumps({
        "status": result["status"],
        "prediction": result.get("prediction"),
        "steps": result["steps"],
    }, indent=2, default=str))
