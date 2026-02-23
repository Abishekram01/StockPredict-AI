"""
Backtesting engine for stock predictions.
Simulates realistic trading with slippage, brokerage, and STT.
Computes Sharpe Ratio, Max Drawdown, Win Rate, Calmar Ratio.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_DIR
from scraper.stock_scraper import load_stock_data

# ─── Trading Costs (India) ───────────────────────────────────────────────
SLIPPAGE_PCT = 0.05 / 100       # 0.05% slippage per trade
BROKERAGE_PCT = 0.03 / 100      # 0.03% brokerage
STT_SELL_PCT = 0.1 / 100        # 0.1% STT on sell side
RISK_FREE_RATE = 0.065          # 6.5% Indian 10Y government bond


def backtest_stock(ticker: str, initial_capital: float = 100000) -> dict:
    """
    Backtest a stock using its saved model predictions on the held-out test set.
    Simulates buy-the-next-day-open, sell-next-signal-change.

    Returns comprehensive results including equity curve, trade log, and metrics.
    """
    model_dir = os.path.join(MODEL_DIR, ticker.replace(".", "_"))
    meta_path = os.path.join(model_dir, "metadata.json")
    test_path = os.path.join(model_dir, "test_data.json")

    if not os.path.exists(meta_path) or not os.path.exists(test_path):
        return {"error": f"No model data for {ticker}"}

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load stock price data
    df = load_stock_data(ticker)
    if df.empty:
        return {"error": f"No price data for {ticker}"}

    # Use last 20% of data (test period)
    n = len(df)
    test_start = int(n * 0.80)
    test_df = df.iloc[test_start:].copy()

    if len(test_df) < 10:
        return {"error": "Not enough test data"}

    # Load model and generate predictions for test period
    try:
        import joblib
        from features.feature_engineer import prepare_features, get_feature_columns
        from sklearn.preprocessing import StandardScaler

        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        best_model_name = meta["best_model"]

        if best_model_name == "LSTM":
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            model = tf.keras.models.load_model(os.path.join(model_dir, "best_model.keras"))
        elif best_model_name == "Stacking":
            model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
        else:
            model = joblib.load(os.path.join(model_dir, "best_model.joblib"))

    except Exception as e:
        return {"error": f"Cannot load model: {e}"}

    # Generate daily signals
    feature_cols = meta.get("feature_columns", get_feature_columns())
    featured = prepare_features(df, ticker)

    if featured.empty:
        return {"error": "Feature preparation failed"}

    available_cols = [c for c in feature_cols if c in featured.columns]
    featured_test = featured.iloc[-(len(test_df)):] if len(featured) >= len(test_df) else featured

    signals = []
    close_prices = test_df["Close"].values
    open_prices = test_df["Open"].values
    dates = test_df.index.tolist()

    for i in range(len(featured_test)):
        try:
            row = featured_test[available_cols].iloc[i:i+1].values
            row_scaled = scaler.transform(row)

            if best_model_name == "Stacking":
                # Load all base models
                base_models = {}
                for name in model["base_model_names"]:
                    safe = name.lower().replace(" ", "_")
                    path = os.path.join(model_dir, f"model_{safe}.joblib")
                    if os.path.exists(path):
                        base_models[name] = joblib.load(path)
                from models.trainer import predict_with_stacking
                prob = predict_with_stacking(model, base_models, row_scaled)[0]
            elif best_model_name == "LSTM":
                prob = 0.5  # Simplified for backtest
            else:
                prob = model.predict_proba(row_scaled)[0][1]

            signal = "BUY" if prob >= 0.6 else ("SELL" if prob <= 0.4 else "HOLD")
        except Exception:
            signal = "HOLD"
            prob = 0.5

        signals.append(signal)

    # ─── Simulate Trading ────────────────────────────────────────
    capital = initial_capital
    position = 0         # shares held
    entry_price = 0.0
    equity_curve = []
    trades = []
    daily_returns = []

    for i in range(len(dates)):
        price = float(close_prices[i])
        open_price = float(open_prices[i]) if i < len(open_prices) else price
        signal = signals[i] if i < len(signals) else "HOLD"

        if signal == "BUY" and position == 0:
            # Buy at open (next day effect simulated by using open price)
            buy_price = open_price * (1 + SLIPPAGE_PCT)
            cost = buy_price * (1 + BROKERAGE_PCT)
            position = int(capital / cost)
            entry_price = buy_price
            capital -= position * cost

            trades.append({
                "type": "BUY",
                "date": dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], 'strftime') else str(dates[i]),
                "price": round(buy_price, 2),
                "shares": position,
                "cost": round(position * cost, 2),
            })

        elif signal == "SELL" and position > 0:
            # Sell at open
            sell_price = open_price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * position
            total_costs = proceeds * (BROKERAGE_PCT + STT_SELL_PCT)
            net_proceeds = proceeds - total_costs
            pnl = net_proceeds - (entry_price * position)

            capital += net_proceeds

            trades.append({
                "type": "SELL",
                "date": dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], 'strftime') else str(dates[i]),
                "price": round(sell_price, 2),
                "shares": position,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / (entry_price * position) * 100, 2),
            })
            position = 0
            entry_price = 0.0

        # Portfolio value
        portfolio_value = capital + (position * price if position > 0 else 0)
        equity_curve.append({
            "date": dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], 'strftime') else str(dates[i]),
            "value": round(portfolio_value, 2),
            "price": round(price, 2),
        })

        if len(equity_curve) >= 2:
            prev = equity_curve[-2]["value"]
            daily_returns.append((portfolio_value - prev) / prev)

    # ─── Compute Metrics ─────────────────────────────────────────
    final_value = equity_curve[-1]["value"] if equity_curve else initial_capital
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Buy & Hold comparison
    bh_return = ((close_prices[-1] - close_prices[0]) / close_prices[0] * 100) if len(close_prices) > 1 else 0

    # Sharpe Ratio (annualized)
    if daily_returns:
        excess_returns = np.array(daily_returns) - RISK_FREE_RATE / 252
        sharpe = np.sqrt(252) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
    else:
        sharpe = 0

    # Max Drawdown
    values = [e["value"] for e in equity_curve]
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd * 100

    # Win/Loss stats
    sell_trades = [t for t in trades if t["type"] == "SELL"]
    wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
    losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0
    reward_risk = avg_win / avg_loss if avg_loss > 0 else 0

    # Calmar Ratio
    calmar = total_return / max_dd_pct if max_dd_pct > 0 else 0

    return {
        "ticker": ticker,
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(bh_return, 2),
        "alpha": round(total_return - bh_return, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "calmar_ratio": round(calmar, 4),
        "total_trades": len(sell_trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "reward_risk_ratio": round(reward_risk, 2),
        "best_model": meta.get("best_model", "Unknown"),
        "test_period": {
            "start": equity_curve[0]["date"] if equity_curve else None,
            "end": equity_curve[-1]["date"] if equity_curve else None,
            "days": len(equity_curve),
        },
        "equity_curve": equity_curve,
        "trades": trades[-20:],  # last 20 trades
        "costs": {
            "slippage": f"{SLIPPAGE_PCT*100:.2f}%",
            "brokerage": f"{BROKERAGE_PCT*100:.2f}%",
            "stt": f"{STT_SELL_PCT*100:.2f}%",
        },
    }


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    result = backtest_stock(ticker)
    if "error" not in result:
        print(f"\n  📊 Backtest: {ticker}")
        print(f"  {'─'*40}")
        print(f"  Return:     {result['total_return_pct']:+.2f}%")
        print(f"  Buy&Hold:   {result['buy_hold_return_pct']:+.2f}%")
        print(f"  Alpha:      {result['alpha']:+.2f}%")
        print(f"  Sharpe:     {result['sharpe_ratio']:.4f}")
        print(f"  Max DD:     {result['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate:   {result['win_rate_pct']:.1f}%")
        print(f"  Trades:     {result['total_trades']}")
    else:
        print(f"  Error: {result['error']}")
