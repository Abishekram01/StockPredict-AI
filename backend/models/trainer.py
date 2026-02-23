"""
ML Training Pipeline — Advanced Version.
Supports:
  - Standard 3-way split (60/20/20) — fast mode
  - Walk-Forward Cross-Validation with purged embargo gap
  - Ensemble Stacking Meta-Learner (7th model)
  - 6 base models: RF, XGBoost, LightGBM, LogReg, SVM, LSTM

All trained models are saved so the evaluator can test each one independently.
The best model (by validation ROC-AUC) is marked for future predictions.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
import xgboost as xgb
import lightgbm as lgb
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    NIFTY50_TICKERS, MODEL_DIR, RANDOM_STATE,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LOOKBACK
)
from features.feature_engineer import prepare_features, get_feature_columns
from scraper.stock_scraper import load_stock_data

warnings.filterwarnings("ignore")

# ─── Split Ratios ────────────────────────────────────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

# Walk-Forward settings
WF_TRAIN_DAYS = 120   # training window
WF_TEST_DAYS  = 30    # test window per fold
WF_EMBARGO    = 5     # purge gap between train and test


def _get_model_dir(ticker: str) -> str:
    d = os.path.join(MODEL_DIR, ticker.replace(".", "_"))
    os.makedirs(d, exist_ok=True)
    return d


# ─── Data Splitting ─────────────────────────────────────────────────────

def _three_way_split(df: pd.DataFrame, feature_cols: list):
    """Chronological 3-way split: Train (60%) / Validation (20%) / Test (20%)."""
    available = [c for c in feature_cols if c in df.columns]
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X = df[available].values
    y = df["Target"].values

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    test_dates = df.index[val_end:].tolist()
    return X_train, y_train, X_val, y_val, X_test, y_test, train_end, val_end, test_dates


def _walk_forward_splits(df: pd.DataFrame, feature_cols: list):
    """
    Walk-Forward Cross-Validation with purge embargo gap.
    Yields (X_train, y_train, X_val, y_val) for each fold.
    """
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    y = df["Target"].values
    n = len(X)

    folds = []
    start = 0
    while start + WF_TRAIN_DAYS + WF_EMBARGO + WF_TEST_DAYS <= n:
        train_end = start + WF_TRAIN_DAYS
        val_start = train_end + WF_EMBARGO  # purge gap
        val_end = val_start + WF_TEST_DAYS

        folds.append((
            X[start:train_end], y[start:train_end],
            X[val_start:val_end], y[val_start:val_end],
        ))
        start += WF_TEST_DAYS  # slide forward

    return folds


# ─── Metrics ────────────────────────────────────────────────────────────

def _evaluate(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            # Handle nan from degenerate predictions
            import math
            metrics["roc_auc"] = round(auc, 4) if not math.isnan(auc) else 0.5
        except (ValueError, TypeError):
            metrics["roc_auc"] = 0.5
    else:
        metrics["roc_auc"] = 0.5
    return metrics


def _evaluate_model_on_set(model, model_name, X_scaled, y_true):
    if model is None:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5}

    if model_name == "LSTM":
        lookback = min(LSTM_LOOKBACK, len(X_scaled) - 1)
        if lookback < 10:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5}
        Xs, ys = [], []
        for i in range(lookback, len(X_scaled)):
            Xs.append(X_scaled[i - lookback:i])
            ys.append(y_true[i])
        if not Xs:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5}
        Xs, ys = np.array(Xs), np.array(ys)
        y_prob = model.predict(Xs, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        return _evaluate(ys, y_pred, y_prob)
    else:
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        return _evaluate(y_true, y_pred, y_prob)


# ─── Individual Model Trainers ──────────────────────────────────────────

def train_random_forest(X_train, y_train, X_eval, y_eval):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return model, _evaluate(y_eval, y_pred, y_prob)


def train_xgboost(X_train, y_train, X_eval, y_eval):
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, use_label_encoder=False,
        eval_metric="logloss", verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return model, _evaluate(y_eval, y_pred, y_prob)


def train_lightgbm(X_train, y_train, X_eval, y_eval):
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, verbosity=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return model, _evaluate(y_eval, y_pred, y_prob)


def train_logistic_regression(X_train, y_train, X_eval, y_eval):
    model = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, C=1.0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return model, _evaluate(y_eval, y_pred, y_prob)


def train_svm(X_train, y_train, X_eval, y_eval):
    model = SVC(
        kernel="rbf", probability=True, random_state=RANDOM_STATE, C=1.0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return model, _evaluate(y_eval, y_pred, y_prob)


def train_lstm(X_train, y_train, X_eval, y_eval):
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM as LSTMLayer, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        tf.get_logger().setLevel("ERROR")

        lookback = min(LSTM_LOOKBACK, len(X_train) - 1)
        if lookback < 10:
            return None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5}

        def create_sequences(X, y, lookback):
            Xs, ys = [], []
            for i in range(lookback, len(X)):
                Xs.append(X[i - lookback:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
        X_eval_seq, y_eval_seq = create_sequences(X_eval, y_eval, lookback)

        if len(X_eval_seq) == 0:
            return None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5}

        model = Sequential([
            LSTMLayer(64, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
            Dropout(0.2),
            LSTMLayer(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(
            X_train_seq, y_train_seq,
            epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
            validation_split=0.1, callbacks=[early_stop], verbose=0
        )

        y_prob = model.predict(X_eval_seq, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        return model, _evaluate(y_eval_seq, y_pred, y_prob)

    except Exception as e:
        print(f"    LSTM error: {e}")
        return None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5}


# ─── Model Registry ─────────────────────────────────────────────────────

MODEL_TRAINERS = {
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
    "Logistic Regression": train_logistic_regression,
    "SVM": train_svm,
    "LSTM": train_lstm,
}


# ─── Stacking Meta-Learner ──────────────────────────────────────────────

def train_stacking_ensemble(X_train, y_train, X_val, y_val, trained_models: dict):
    """
    Build a stacking ensemble using the 6 base models' predictions as meta-features.
    Uses a Logistic Regression as the final meta-learner.
    """
    # Collect base model predictions on validation set as meta-features
    base_names = []
    base_models_for_stack = []

    for name, model in trained_models.items():
        if model is None or name == "LSTM":
            continue  # skip LSTM (different input shape)
        base_names.append(name)
        base_models_for_stack.append(model)

    if len(base_models_for_stack) < 2:
        return None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5}

    # Generate meta-features: each base model's probability output
    train_meta = np.column_stack([
        m.predict_proba(X_train)[:, 1] for m in base_models_for_stack
    ])
    val_meta = np.column_stack([
        m.predict_proba(X_val)[:, 1] for m in base_models_for_stack
    ])

    # Train meta-learner
    meta_learner = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    meta_learner.fit(train_meta, y_train)

    y_pred = meta_learner.predict(val_meta)
    y_prob = meta_learner.predict_proba(val_meta)[:, 1]

    # Wrap everything for saving
    stacking_model = {
        "meta_learner": meta_learner,
        "base_model_names": base_names,
    }

    return stacking_model, _evaluate(y_val, y_pred, y_prob)


def predict_with_stacking(stacking_model, base_models, X_scaled):
    """Generate prediction from stacking ensemble."""
    base_names = stacking_model["base_model_names"]
    meta_learner = stacking_model["meta_learner"]

    meta_features = np.column_stack([
        base_models[name].predict_proba(X_scaled)[:, 1]
        for name in base_names
        if name in base_models and base_models[name] is not None
    ])

    y_prob = meta_learner.predict_proba(meta_features)[:, 1]
    return y_prob


# ─── Walk-Forward Training ──────────────────────────────────────────────

def train_walk_forward(ticker: str, df: pd.DataFrame) -> dict:
    """
    Walk-Forward Cross-Validation with purged embargo gap.
    Trains all models across multiple time windows for robust evaluation.
    """
    feature_cols = get_feature_columns()
    featured_df = prepare_features(df, ticker)
    if len(featured_df) < 100:
        print(f"  ⚠  Not enough data for {ticker} ({len(featured_df)} rows)")
        return {}

    available_cols = [c for c in feature_cols if c in featured_df.columns]
    folds = _walk_forward_splits(featured_df, available_cols)

    if len(folds) < 2:
        print(f"  ⚠  Not enough data for walk-forward ({len(folds)} folds)")
        return train_all_models_for_stock(ticker, df)  # fallback

    print(f"    Walk-Forward: {len(folds)} folds "
          f"(train={WF_TRAIN_DAYS}d, test={WF_TEST_DAYS}d, embargo={WF_EMBARGO}d)")

    # Aggregate metrics across folds
    agg_metrics = {name: [] for name in MODEL_TRAINERS}

    for fold_idx, (X_tr, y_tr, X_va, y_va) in enumerate(folds):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        for model_name, trainer_fn in MODEL_TRAINERS.items():
            try:
                _, metrics = trainer_fn(X_tr_s, y_tr, X_va_s, y_va)
                agg_metrics[model_name].append(metrics)
            except Exception:
                pass

    # Average metrics across folds
    avg_metrics = {}
    for name, fold_results in agg_metrics.items():
        if not fold_results:
            avg_metrics[name] = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0}
            continue
        avg = {}
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            vals = [r[key] for r in fold_results]
            avg[key] = round(np.mean(vals), 4)
        avg["std_auc"] = round(np.std([r["roc_auc"] for r in fold_results]), 4)
        avg["n_folds"] = len(fold_results)
        avg_metrics[name] = avg

    # Print summary
    for name in MODEL_TRAINERS:
        m = avg_metrics.get(name, {})
        print(f"    {name:25s} → Avg AUC: {m.get('roc_auc', 0):.4f} "
              f"(±{m.get('std_auc', 0):.4f}, {m.get('n_folds', 0)} folds)")

    # Now do final training on full data with standard split for model saving
    result = train_all_models_for_stock(ticker, df)

    # Add walk-forward metrics to metadata
    if result:
        model_dir = _get_model_dir(ticker)
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["walkforward_metrics"] = avg_metrics
            meta["walkforward_folds"] = len(folds)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    return result


# ─── Main Training Orchestrator ─────────────────────────────────────────

def train_all_models_for_stock(ticker: str, df: pd.DataFrame) -> dict:
    """
    Train all models for a single stock using 3-way split.
    Includes Stacking Meta-Learner as 7th model.
    Cleans old model files only after confirming we have enough data.
    """
    feature_cols = get_feature_columns()
    featured_df = prepare_features(df, ticker)
    if len(featured_df) < 100:
        print(f"  ⚠  Not enough data for {ticker} ({len(featured_df)} rows)")
        return {}

    # Data validated — safe to delete old model files now
    model_dir = _get_model_dir(ticker)
    import glob
    for old_file in glob.glob(os.path.join(model_dir, "*.joblib")):
        os.remove(old_file)
    for old_file in glob.glob(os.path.join(model_dir, "*.keras")):
        os.remove(old_file)

    available_cols = [c for c in feature_cols if c in featured_df.columns]
    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_end, val_end, test_dates) = _three_way_split(featured_df, available_cols)

    print(f"    Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    val_results = {}
    test_results = {}
    trained_models = {}
    best_model = None
    best_model_name = None
    best_val_auc = -1

    # Train 6 base models
    for model_name, trainer_fn in MODEL_TRAINERS.items():
        try:
            model, val_metrics = trainer_fn(X_train_scaled, y_train, X_val_scaled, y_val)
            val_results[model_name] = val_metrics
            trained_models[model_name] = model

            test_metrics = _evaluate_model_on_set(model, model_name, X_test_scaled, y_test)
            test_results[model_name] = test_metrics

            print(f"    {model_name:25s} → Val AUC: {val_metrics['roc_auc']:.4f}  "
                  f"Test AUC: {test_metrics['roc_auc']:.4f}  "
                  f"Test Acc: {test_metrics['accuracy']:.4f}")

            if val_metrics["roc_auc"] > best_val_auc and model is not None:
                best_val_auc = val_metrics["roc_auc"]
                best_model = model
                best_model_name = model_name

        except Exception as e:
            print(f"    ✗ {model_name} failed: {e}")
            zero = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0}
            val_results[model_name] = zero
            test_results[model_name] = zero

    # Fallback: if no best model by AUC (all 0.5/nan), pick by accuracy
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
        if best_model_name:
            print(f"    ⚠  All AUCs = 0.5 — fell back to {best_model_name} (acc={best_acc:.4f})")

    # ─── 7th Model: Stacking Meta-Learner ───────────────────
    try:
        stacking_model, stack_val = train_stacking_ensemble(
            X_train_scaled, y_train, X_val_scaled, y_val, trained_models
        )
        if stacking_model is not None:
            val_results["Stacking"] = stack_val

            # Evaluate stacking on test set
            stack_test_prob = predict_with_stacking(stacking_model, trained_models, X_test_scaled)
            stack_test_pred = (stack_test_prob > 0.5).astype(int)
            stack_test = _evaluate(y_test, stack_test_pred, stack_test_prob)
            test_results["Stacking"] = stack_test

            print(f"    {'Stacking':25s} → Val AUC: {stack_val['roc_auc']:.4f}  "
                  f"Test AUC: {stack_test['roc_auc']:.4f}  "
                  f"Test Acc: {stack_test['accuracy']:.4f}")

            if stack_val["roc_auc"] > best_val_auc:
                best_val_auc = stack_val["roc_auc"]
                best_model_name = "Stacking"
                best_model = stacking_model
    except Exception as e:
        print(f"    ✗ Stacking failed: {e}")

    # ─── Save Everything ────────────────────────────────────
    model_dir = _get_model_dir(ticker)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

    # Save test data
    test_data = {
        "X_test": X_test_scaled.tolist(),
        "y_test": y_test.tolist(),
        "test_dates": [str(d) for d in test_dates],
    }
    with open(os.path.join(model_dir, "test_data.json"), "w") as f:
        json.dump(test_data, f)

    # Save each model
    for model_name, model in trained_models.items():
        if model is None:
            continue
        safe_name = model_name.lower().replace(" ", "_")
        if model_name == "LSTM":
            model.save(os.path.join(model_dir, f"model_{safe_name}.keras"))
        else:
            joblib.dump(model, os.path.join(model_dir, f"model_{safe_name}.joblib"))

    # Save stacking model
    if "Stacking" in val_results and best_model_name == "Stacking":
        joblib.dump(stacking_model, os.path.join(model_dir, "model_stacking.joblib"))

    # Save best model
    if best_model is not None:
        if best_model_name == "LSTM":
            best_model.save(os.path.join(model_dir, "best_model.keras"))
        elif best_model_name == "Stacking":
            joblib.dump(best_model, os.path.join(model_dir, "best_model.joblib"))
        else:
            joblib.dump(best_model, os.path.join(model_dir, "best_model.joblib"))

    # Metadata
    meta = {
        "ticker": ticker,
        "best_model": best_model_name,
        "best_val_auc": best_val_auc,
        "best_test_auc": test_results.get(best_model_name, {}).get("roc_auc", 0),
        "val_metrics": val_results,
        "test_metrics": test_results,
        "all_metrics": val_results,
        "feature_columns": [c for c in feature_cols if c in featured_df.columns],
        "data_rows": len(featured_df),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "has_stacking": "Stacking" in val_results,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if best_model_name:
        best_test = test_results.get(best_model_name, {})
        print(f"    ★ Best: {best_model_name} "
              f"(Val AUC: {best_val_auc:.4f} → Test AUC: {best_test.get('roc_auc', 0):.4f})")

    return {"val_metrics": val_results, "test_metrics": test_results}


def train_all_stocks(tickers: dict = None, stock_data: dict = None, walkforward: bool = False) -> dict:
    """Train models for all stocks."""
    if tickers is None:
        tickers = NIFTY50_TICKERS
    if stock_data is None:
        stock_data = {t: load_stock_data(t) for t in tickers}

    all_results = {}
    total = len(tickers)
    mode = "Walk-Forward CV" if walkforward else f"{int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}"

    print(f"\n{'='*60}")
    print(f"  🤖  Training models for {total} stocks")
    print(f"  📐  Mode: {mode} + Stacking Meta-Learner")
    print(f"{'='*60}\n")

    for i, (ticker, name) in enumerate(tickers.items(), 1):
        print(f"\n  [{i:02d}/{total}] {name} ({ticker})")
        print(f"  {'─'*50}")

        df = stock_data.get(ticker, pd.DataFrame())
        if df.empty:
            print(f"  ⚠  No data for {ticker}, skipping")
            continue

        if walkforward:
            results = train_walk_forward(ticker, df)
        else:
            results = train_all_models_for_stock(ticker, df)

        if results:
            all_results[ticker] = results

    print(f"\n  ✅  Trained models for {len(all_results)}/{total} stocks\n")
    return all_results


if __name__ == "__main__":
    test_tickers = {k: v for i, (k, v) in enumerate(NIFTY50_TICKERS.items()) if i < 3}
    train_all_stocks(test_tickers)
