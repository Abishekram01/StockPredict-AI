"""
Flask REST API for the Trading Bot.
Serves predictions, stock data, model metrics, and news to the frontend.
"""
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import NIFTY50_TICKERS, MODEL_DIR
from models.predictor import predict_all_stocks, predict_stock, get_stock_details
from scraper.stock_scraper import load_stock_data, load_all_stock_data
from scraper.news_scraper import load_cached_news, fetch_all_news
from config import API_HOST, API_PORT

app = Flask(__name__)
CORS(app)


@app.route("/api/stocks", methods=["GET"])
def get_all_stocks():
    """Get all stocks with their latest predictions."""
    try:
        predictions = predict_all_stocks()
        # Summary counts
        buy_count = sum(1 for p in predictions if p["signal"] == "BUY")
        hold_count = sum(1 for p in predictions if p["signal"] == "HOLD")
        sell_count = sum(1 for p in predictions if p["signal"] == "SELL")

        return jsonify({
            "success": True,
            "summary": {
                "total": len(predictions),
                "buy": buy_count,
                "hold": hold_count,
                "sell": sell_count,
            },
            "stocks": predictions
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stocks/<ticker>", methods=["GET"])
def get_stock_detail(ticker):
    """Get detailed info for a specific stock."""
    try:
        # Normalize ticker (frontend sends without .NS)
        if not ticker.endswith(".NS"):
            ticker = ticker + ".NS"

        details = get_stock_details(ticker)
        return jsonify({"success": True, "data": details})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def get_model_comparisons():
    """Get model comparison metrics for all stocks."""
    try:
        all_metrics = {}
        for ticker in NIFTY50_TICKERS:
            model_dir = os.path.join(MODEL_DIR, ticker.replace(".", "_"))
            meta_path = os.path.join(model_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    all_metrics[ticker] = {
                        "company": NIFTY50_TICKERS.get(ticker, ticker),
                        "best_model": meta.get("best_model", "N/A"),
                        "best_val_auc": meta.get("best_val_auc", meta.get("best_auc", 0)),
                        "best_test_auc": meta.get("best_test_auc", 0),
                        "best_model_on_test": meta.get("best_model_on_test", "N/A"),
                        "val_metrics": meta.get("val_metrics", meta.get("all_metrics", {})),
                        "test_metrics": meta.get("test_metrics", {}),
                        "all_metrics": meta.get("all_metrics", {}),
                        "split_ratios": meta.get("split_ratios", {}),
                    }
        return jsonify({"success": True, "models": all_metrics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/news/<ticker>", methods=["GET"])
def get_stock_news(ticker):
    """Get latest news and sentiment for a stock."""
    try:
        if not ticker.endswith(".NS"):
            ticker = ticker + ".NS"
        news = load_cached_news(ticker)
        return jsonify({"success": True, "news": news})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/refresh", methods=["POST"])
def refresh_data():
    """Trigger a full re-scrape and re-train cycle."""
    try:
        from scraper.stock_scraper import scrape_all_stocks
        from models.trainer import train_all_stocks
        from models.evaluator import run_full_evaluation

        # Re-scrape
        stock_data = scrape_all_stocks()
        # Re-fetch news
        fetch_all_news()
        # Re-train
        train_all_stocks(stock_data=stock_data)
        # Evaluate on held-out test set
        run_full_evaluation()

        return jsonify({"success": True, "message": "Data refreshed, models retrained and evaluated."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/top-picks", methods=["GET"])
def get_top_picks():
    """Get top 2 BUY and top 2 SELL stocks with full company insights."""
    try:
        from scraper.company_insights import get_company_insights

        predictions = predict_all_stocks()

        # Top 2 BUY (highest confidence)
        buy_stocks = [p for p in predictions if p["signal"] == "BUY"]
        buy_stocks.sort(key=lambda x: -x["confidence"])
        top_buys = buy_stocks[:2]

        # Top 2 SELL (highest confidence)
        sell_stocks = [p for p in predictions if p["signal"] == "SELL"]
        sell_stocks.sort(key=lambda x: -x["confidence"])
        top_sells = sell_stocks[:2]

        # Fetch insights for each
        result = {"buy": [], "sell": []}

        for pred in top_buys:
            insights = get_company_insights(pred["ticker"])
            result["buy"].append({
                "prediction": pred,
                "insights": insights,
            })

        for pred in top_sells:
            insights = get_company_insights(pred["ticker"])
            result["sell"].append({
                "prediction": pred,
                "insights": insights,
            })

        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/insights/<ticker>", methods=["GET"])
def get_company_insight(ticker):
    """Get comprehensive company insights for a specific stock."""
    try:
        from scraper.company_insights import get_company_insights

        if not ticker.endswith(".NS"):
            ticker = ticker + ".NS"
        insights = get_company_insights(ticker)
        return jsonify({"success": True, "data": insights})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_stock():
    """
    On-demand analysis for any stock ticker.
    Runs full pipeline: scrape → train → evaluate → predict → insights.
    Body: { "ticker": "TATAMOTORS", "company_name": "Tata Motors" }
    """
    try:
        from models.on_demand import analyze_custom_stock

        data = request.get_json()
        ticker = (data.get("ticker") or "").strip()
        company_name = (data.get("company_name") or "").strip() or None

        if not ticker:
            return jsonify({"success": False, "error": "Ticker symbol is required."}), 400

        result = analyze_custom_stock(ticker, company_name)

        if result["status"] == "error":
            return jsonify({"success": False, "error": result.get("error", "Unknown error")}), 400

        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/backtest/<ticker>", methods=["GET"])
def get_backtest(ticker):
    """Backtest a stock's model predictions on the held-out test period."""
    try:
        from models.backtester import backtest_stock

        if not ticker.upper().endswith(".NS"):
            ticker = ticker.upper() + ".NS"
        result = backtest_stock(ticker)
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 400
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/var/<ticker>", methods=["GET"])
def get_var(ticker):
    """Get Value at Risk analysis for a stock."""
    try:
        from models.risk_calculator import calculate_var

        if not ticker.upper().endswith(".NS"):
            ticker = ticker.upper() + ".NS"
        position = request.args.get("position", 10000, type=float)
        result = calculate_var(ticker, position_value=position)
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 400
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sentiment-heatmap", methods=["GET"])
def sentiment_heatmap():
    """Sector-wise sentiment heatmap data."""
    try:
        import yfinance as yf
        from scraper.news_scraper import get_average_sentiment

        sectors = {}
        for ticker, name in NIFTY50_TICKERS.items():
            try:
                info = yf.Ticker(ticker).info
                sector = info.get("sector", "Other")
            except Exception:
                sector = "Other"

            sentiment = get_average_sentiment(ticker)

            if sector not in sectors:
                sectors[sector] = {"stocks": [], "total_sentiment": 0, "count": 0}

            sectors[sector]["stocks"].append({
                "ticker": ticker,
                "company": name,
                "sentiment": sentiment,
            })
            sectors[sector]["total_sentiment"] += sentiment
            sectors[sector]["count"] += 1

        # Aggregate
        heatmap = []
        for sector, data in sectors.items():
            avg = data["total_sentiment"] / data["count"] if data["count"] > 0 else 0
            positive = sum(1 for s in data["stocks"] if s["sentiment"] > 0.05)
            negative = sum(1 for s in data["stocks"] if s["sentiment"] < -0.05)
            neutral = data["count"] - positive - negative

            label = "Bullish" if avg > 0.05 else ("Bearish" if avg < -0.05 else "Neutral")

            heatmap.append({
                "sector": sector,
                "avg_sentiment": round(avg, 4),
                "verdict": label,
                "stock_count": data["count"],
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "stocks": sorted(data["stocks"], key=lambda s: s["sentiment"], reverse=True),
            })

        heatmap.sort(key=lambda s: s["avg_sentiment"], reverse=True)
        return jsonify({"success": True, "data": heatmap})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Trading Bot API is running"})


def start_server():
    """Start the Flask development server."""
    print(f"\n  🚀  API server starting on http://{API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=False)


if __name__ == "__main__":
    start_server()
