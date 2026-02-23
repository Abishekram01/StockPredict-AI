"""
Risk calculator for stock positions.
Computes Historical VaR, Parametric VaR, and Conditional VaR (Expected Shortfall).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scraper.stock_scraper import load_stock_data


def calculate_var(ticker: str, position_value: float = 10000, confidence_levels: list = None) -> dict:
    """
    Calculate Value at Risk (VaR) for a stock position.

    Args:
        ticker: Stock ticker
        position_value: Hypothetical position size in INR (default 10,000)
        confidence_levels: List of confidence levels (default [0.95, 0.99])

    Returns dict with:
        - historical_var: from actual return distribution
        - parametric_var: assuming normal distribution
        - cvar (Expected Shortfall): average loss beyond VaR
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    df = load_stock_data(ticker)
    if df.empty or len(df) < 30:
        return {"error": "Not enough data for VaR calculation"}

    # Daily returns
    returns = df["Close"].pct_change().dropna()

    result = {
        "ticker": ticker,
        "position_value": position_value,
        "data_days": len(returns),
        "mean_daily_return": round(float(returns.mean()) * 100, 4),
        "daily_volatility": round(float(returns.std()) * 100, 4),
        "annualized_volatility": round(float(returns.std() * np.sqrt(252)) * 100, 2),
        "var": {},
    }

    for cl in confidence_levels:
        cl_label = f"{int(cl*100)}%"
        alpha = 1 - cl

        # Historical VaR: actual percentile of returns
        hist_var = np.percentile(returns, alpha * 100)
        hist_var_amount = abs(hist_var) * position_value

        # Parametric VaR: assume normal distribution
        from scipy import stats
        z_score = stats.norm.ppf(alpha)
        para_var = returns.mean() + z_score * returns.std()
        para_var_amount = abs(para_var) * position_value

        # CVaR (Expected Shortfall): mean of losses beyond VaR
        tail_returns = returns[returns <= hist_var]
        cvar = tail_returns.mean() if len(tail_returns) > 0 else hist_var
        cvar_amount = abs(cvar) * position_value

        result["var"][cl_label] = {
            "historical": {
                "pct": round(abs(hist_var) * 100, 4),
                "amount": round(hist_var_amount, 2),
                "interpretation": f"On a ₹{position_value:,.0f} position, you could lose ≥ ₹{hist_var_amount:,.0f} in {alpha*100:.0f}% of days",
            },
            "parametric": {
                "pct": round(abs(para_var) * 100, 4),
                "amount": round(para_var_amount, 2),
            },
            "cvar": {
                "pct": round(abs(cvar) * 100, 4),
                "amount": round(cvar_amount, 2),
                "interpretation": f"When losses exceed VaR, average loss ≈ ₹{cvar_amount:,.0f}",
            },
        }

    # Additional risk metrics
    max_daily_loss = float(returns.min()) * 100
    max_daily_gain = float(returns.max()) * 100

    # Worst-case scenarios
    worst_5 = returns.nsmallest(5).values * 100

    result["extremes"] = {
        "max_daily_loss_pct": round(max_daily_loss, 2),
        "max_daily_gain_pct": round(max_daily_gain, 2),
        "worst_5_days_pct": [round(w, 2) for w in worst_5],
        "skewness": round(float(returns.skew()), 4),
        "kurtosis": round(float(returns.kurtosis()), 4),
    }

    return result


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    result = calculate_var(ticker)
    if "error" not in result:
        print(f"\n  📉 VaR Analysis: {ticker}")
        print(f"  {'─'*40}")
        print(f"  Daily Vol:  {result['daily_volatility']}%")
        print(f"  Annual Vol: {result['annualized_volatility']}%")
        for cl, data in result["var"].items():
            print(f"\n  {cl} Confidence:")
            print(f"    Historical VaR: {data['historical']['pct']}% (₹{data['historical']['amount']:,.0f})")
            print(f"    CVaR (ES):      {data['cvar']['pct']}% (₹{data['cvar']['amount']:,.0f})")
