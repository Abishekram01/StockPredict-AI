"""
Model Evaluator — tests all saved models against the held-out test set.
Run standalone:  python -m models.evaluator
"""
import os
import json
import numpy as np
import joblib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NIFTY50_TICKERS, MODEL_DIR, LSTM_LOOKBACK
from models.trainer import _evaluate

# Model file mapping
MODEL_FILES = {
    "Random Forest":        "model_random_forest.joblib",
    "XGBoost":              "model_xgboost.joblib",
    "LightGBM":             "model_lightgbm.joblib",
    "Logistic Regression":  "model_logistic_regression.joblib",
    "SVM":                  "model_svm.joblib",
    "LSTM":                 "model_lstm.keras",
    "Stacking":             "model_stacking.joblib",
}


def load_test_data(ticker: str):
    """Load the held-out test set for a ticker."""
    model_dir = os.path.join(MODEL_DIR, ticker.replace(".", "_"))
    test_path = os.path.join(model_dir, "test_data.json")
    if not os.path.exists(test_path):
        return None, None, None
    with open(test_path, "r") as f:
        data = json.load(f)
    X_test = np.array(data["X_test"])
    y_test = np.array(data["y_test"])
    test_dates = data.get("test_dates", [])
    return X_test, y_test, test_dates


def evaluate_all_models_for_stock(ticker: str) -> dict:
    """
    Load every saved model for a ticker, evaluate each on the held-out
    test set, and return per-model metrics.
    """
    model_dir = os.path.join(MODEL_DIR, ticker.replace(".", "_"))
    X_test, y_test, test_dates = load_test_data(ticker)
    if X_test is None:
        return {}

    results = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = os.path.join(model_dir, filename)
        if not os.path.exists(model_path):
            continue

        try:
            if model_name == "Stacking":
                stacking = joblib.load(model_path)
                # Load base models for stacking
                base_models = {}
                for name in stacking.get("base_model_names", []):
                    safe = name.lower().replace(" ", "_")
                    bp = os.path.join(model_dir, f"model_{safe}.joblib")
                    if os.path.exists(bp):
                        base_models[name] = joblib.load(bp)
                if len(base_models) >= 2:
                    from models.trainer import predict_with_stacking
                    y_prob = predict_with_stacking(stacking, base_models, X_test)
                    y_pred = (y_prob > 0.5).astype(int)
                    results[model_name] = _evaluate(y_test, y_pred, y_prob)
                else:
                    continue
            elif model_name == "LSTM":
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                import tensorflow as tf
                tf.get_logger().setLevel("ERROR")
                model = tf.keras.models.load_model(model_path)

                lookback = min(LSTM_LOOKBACK, len(X_test) - 1)
                if lookback < 10:
                    results[model_name] = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5}
                    continue
                Xs, ys = [], []
                for i in range(lookback, len(X_test)):
                    Xs.append(X_test[i - lookback:i])
                    ys.append(y_test[i])
                Xs, ys = np.array(Xs), np.array(ys)
                y_prob = model.predict(Xs, verbose=0).flatten()
                y_pred = (y_prob > 0.5).astype(int)
                results[model_name] = _evaluate(ys, y_pred, y_prob)
            else:
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                results[model_name] = _evaluate(y_test, y_pred, y_prob)
        except Exception as e:
            print(f"  ✗ {model_name} eval failed for {ticker}: {e}")
            results[model_name] = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0}

    return results


def run_full_evaluation(tickers: dict = None) -> dict:
    """
    Evaluate all models for all stocks on their held-out test sets.
    Prints a comprehensive report.
    """
    if tickers is None:
        tickers = NIFTY50_TICKERS

    all_results = {}
    total = len(tickers)

    print(f"\n{'='*70}")
    print(f"  🧪  HELD-OUT TEST SET EVALUATION")
    print(f"  📐  Testing every model on the unseen 20% test data")
    print(f"{'='*70}\n")

    summary_rows = []

    for i, (ticker, name) in enumerate(tickers.items(), 1):
        test_metrics = evaluate_all_models_for_stock(ticker)
        if not test_metrics:
            continue

        all_results[ticker] = test_metrics

        # Find best model on test set
        best_name = max(test_metrics, key=lambda k: test_metrics[k].get("roc_auc", 0))
        best_auc = test_metrics[best_name]["roc_auc"]
        best_acc = test_metrics[best_name]["accuracy"]

        print(f"  [{i:02d}/{total}] {name:25s} │ Best on test: {best_name:22s} │ "
              f"AUC: {best_auc:.4f}  Acc: {best_acc:.4f}")

        summary_rows.append({
            "ticker": ticker,
            "company": name,
            "best_model_test": best_name,
            "test_auc": best_auc,
            "test_acc": best_acc,
            "all_test_metrics": test_metrics,
        })

        # Update metadata with test evaluation results
        meta_path = os.path.join(MODEL_DIR, ticker.replace(".", "_"), "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["test_metrics"] = test_metrics
            meta["best_model_on_test"] = best_name
            meta["best_test_auc"] = best_auc
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    # Print summary table
    if summary_rows:
        print(f"\n{'='*70}")
        print(f"  📊  EVALUATION SUMMARY ({len(summary_rows)} stocks)")
        print(f"{'='*70}")
        print(f"  {'Stock':25s} │ {'Best Model (Test)':22s} │ {'AUC':>6s} │ {'Acc':>6s}")
        print(f"  {'─'*25}─┼─{'─'*22}─┼─{'─'*6}─┼─{'─'*6}")
        for row in sorted(summary_rows, key=lambda r: -r["test_auc"]):
            print(f"  {row['company']:25s} │ {row['best_model_test']:22s} │ "
                  f"{row['test_auc']:.4f} │ {row['test_acc']:.4f}")

        avg_auc = sum(r["test_auc"] for r in summary_rows) / len(summary_rows)
        avg_acc = sum(r["test_acc"] for r in summary_rows) / len(summary_rows)
        print(f"  {'─'*25}─┼─{'─'*22}─┼─{'─'*6}─┼─{'─'*6}")
        print(f"  {'AVERAGE':25s} │ {'':22s} │ {avg_auc:.4f} │ {avg_acc:.4f}")
        print()

    return all_results


if __name__ == "__main__":
    run_full_evaluation()
