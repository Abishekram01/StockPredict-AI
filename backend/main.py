"""
Main orchestrator for the Trading Bot.
Runs the full pipeline: scrape → news → train → evaluate → predict → serve.
Supports --walkforward for rigorous cross-validation.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NIFTY50_TICKERS
from scraper.stock_scraper import scrape_all_stocks, load_all_stock_data
from scraper.news_scraper import fetch_all_news
from models.trainer import train_all_stocks
from models.evaluator import run_full_evaluation
from models.predictor import predict_all_stocks
from api.server import start_server


def run_full_pipeline(walkforward: bool = False):
    """Run the complete pipeline: scrape, train, predict, serve."""
    print("\n" + "="*60)
    print("  🤖  Indian Stock Trading Bot — Advanced ML")
    print("  📊  NIFTY 50 Prediction System")
    print("  🔬  Triple Barrier + Stacking + Macro Features")
    print("="*60)

    # Step 1: Scrape stock data
    print("\n  STEP 1/6: Scraping stock data...")
    stock_data = scrape_all_stocks()

    # Step 2: Fetch news
    print("\n  STEP 2/6: Fetching news and sentiment...")
    fetch_all_news()

    # Step 3: Fetch macro data
    print("\n  STEP 3/6: Fetching macro-economic data...")
    try:
        from features.macro_features import fetch_macro_data
        macro = fetch_macro_data()
        print(f"  ✅  Macro data: {macro.shape[0]} days, {macro.shape[1]} features")
    except Exception as e:
        print(f"  ⚠  Macro data unavailable: {e}")

    # Step 4: Train models
    mode_str = "Walk-Forward CV" if walkforward else "60/20/20 Split"
    print(f"\n  STEP 4/6: Training ML models ({mode_str} + Stacking)...")
    train_all_stocks(stock_data=stock_data, walkforward=walkforward)

    # Step 5: Evaluate on held-out test set
    print("\n  STEP 5/6: Evaluating all models on held-out test set (20%)...")
    run_full_evaluation()

    # Step 6: Generate predictions
    print("\n  STEP 6/6: Generating predictions...")
    predictions = predict_all_stocks()

    # Summary
    buy = [p for p in predictions if p["signal"] == "BUY"]
    hold = [p for p in predictions if p["signal"] == "HOLD"]
    sell = [p for p in predictions if p["signal"] == "SELL"]

    print(f"\n{'='*60}")
    print(f"  📊  PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"  🟢 BUY:  {len(buy)} stocks")
    print(f"  🟡 HOLD: {len(hold)} stocks")
    print(f"  🔴 SELL: {len(sell)} stocks")

    if buy:
        print(f"\n  Top BUY picks:")
        for p in buy[:5]:
            print(f"    • {p['company']:25s} → {p['confidence']:.0f}% confidence (₹{p['current_price']:,.2f})")

    # Start API server
    print(f"\n  Starting API server...")
    start_server()


def main():
    parser = argparse.ArgumentParser(description="Indian Stock Trading Bot — Advanced ML")
    parser.add_argument("--scrape-only", action="store_true", help="Only scrape data")
    parser.add_argument("--train-only", action="store_true", help="Only train models")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate models on held-out test set")
    parser.add_argument("--predict-only", action="store_true", help="Only generate predictions")
    parser.add_argument("--serve-only", action="store_true", help="Only start API server")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (default)")
    parser.add_argument("--walkforward", action="store_true",
                        help="Use Walk-Forward CV instead of single split (slower, more robust)")

    args = parser.parse_args()

    if args.scrape_only:
        scrape_all_stocks()
    elif args.train_only:
        stock_data = load_all_stock_data()
        fetch_all_news()
        train_all_stocks(stock_data=stock_data, walkforward=args.walkforward)
    elif args.evaluate_only:
        run_full_evaluation()
    elif args.predict_only:
        predictions = predict_all_stocks()
    elif args.serve_only:
        start_server()
    else:
        run_full_pipeline(walkforward=args.walkforward)


if __name__ == "__main__":
    main()
