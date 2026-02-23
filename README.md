<p align="center">
  <h1 align="center">📊 StockPredict AI</h1>
  <p align="center">
    <b>ML-Powered NIFTY 50 Stock Prediction System</b><br/>
    Triple Barrier Labeling · 7-Model Ensemble · Macro Features · Real-Time Dashboard
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/react-18-61DAFB?logo=react&logoColor=white" />
  <img src="https://img.shields.io/badge/flask-REST_API-black?logo=flask" />
  <img src="https://img.shields.io/badge/models-XGBoost%20·%20LightGBM%20·%20LSTM-green" />
  <img src="https://img.shields.io/badge/license-MIT-yellow" />
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **7 ML Models** | Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, LSTM, + Stacking Meta-Learner |
| **Triple Barrier Labeling** | ATR-based regime detection — profit-take, stop-loss, or timeout |
| **70+ Features** | Technical indicators, fractional differencing, macro-economic data, news sentiment |
| **Macro Indicators** | India VIX, USD/INR, US 10Y Treasury, Nifty indices, Gold |
| **Smart Caching** | Stock data (12h), news (4h) — skip re-download if fresh |
| **Walk-Forward CV** | Sliding window validation with 5-day purged embargo gap |
| **Backtesting Engine** | Realistic Indian market simulation (slippage, STT, brokerage) |
| **Risk Analysis** | Historical VaR, Parametric VaR, CVaR (Expected Shortfall) |
| **FinBERT Sentiment** | Financial-domain NLP (optional), VADER fallback |
| **Real-Time Dashboard** | React + Vite with Top Picks, Custom Search, Sentiment Heatmap |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + Vite)                  │
│  Dashboard · Top Picks · Custom Search · Sentiment Heatmap  │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────────┐
│                   Flask API Server (13 endpoints)            │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│  Predictor  │  Backtester  │ Risk Calc    │  On-Demand      │
├─────────────┴──────────────┴──────────────┴─────────────────┤
│           Trainer (7 models + Walk-Forward CV)               │
├─────────────────────────────────────────────────────────────┤
│    Feature Engineer (Technical + Frac Diff + Macro + NLP)    │
├──────────────┬──────────────────┬───────────────────────────┤
│ Stock Scraper│   News Scraper   │   Macro Features          │
│  (yfinance)  │ (Google News RSS)│ (VIX, USD/INR, Gold...)   │
└──────────────┴──────────────────┴───────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Scrape → News → Macro → Train → Evaluate → Predict → Serve
python main.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — the dashboard connects to the backend on port 5000.

---

## 🎮 CLI Options

| Command | What It Does |
|---------|-------------|
| `python main.py` | Full pipeline (scrape + train + predict + serve) |
| `python main.py --walkforward` | Rigorous walk-forward cross-validation |
| `python main.py --serve-only` | Start API server only (use existing models) |
| `python main.py --scrape-only` | Only download stock data |
| `python main.py --train-only` | Retrain all models |
| `python main.py --evaluate-only` | Re-evaluate on test sets |
| `python main.py --predict-only` | Regenerate predictions |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stocks` | All stocks with predictions |
| `GET` | `/api/stocks/<ticker>` | Detailed stock info |
| `GET` | `/api/top-picks` | Top 2 BUY + top 2 SELL |
| `POST` | `/api/analyze` | On-demand analysis for any ticker |
| `GET` | `/api/backtest/<ticker>` | Backtest results + equity curve |
| `GET` | `/api/var/<ticker>` | Value at Risk analysis |
| `GET` | `/api/sentiment-heatmap` | Sector-wise sentiment |
| `GET` | `/api/models` | Model comparison metrics |
| `POST` | `/api/refresh` | Re-scrape + retrain cycle |

---

## 🧠 ML Pipeline

### Models

| # | Model | Type |
|---|-------|------|
| 1 | Random Forest | Ensemble (bagging) |
| 2 | XGBoost | Gradient boosting |
| 3 | LightGBM | Gradient boosting |
| 4 | Logistic Regression | Linear |
| 5 | SVM | Kernel (RBF) |
| 6 | LSTM | Deep learning (optional) |
| 7 | **Stacking** | Meta-learner on top of 1–5 |

### Feature Engineering (70+ features)

- **Technical Indicators** — SMA, EMA, MACD, RSI, Bollinger Bands, ATR, ADX, OBV, CCI, Stochastic, Williams %R
- **Fractional Differencing** — Order d=0.4 for stationarity with memory preservation
- **Macro Features** — India VIX, USD/INR, US 10Y, Nifty 50/IT/Bank, Gold (returns + momentum)
- **Sentiment** — VADER (default) or FinBERT (optional)

### Triple Barrier Target

```
Upper barrier = Close + ATR × 2.0   → profit-take → label 1
Lower barrier = Close − ATR × 2.0   → stop-loss   → label 0
Time barrier  = 5 days max hold      → timeout     → label 0
```

### Prediction Signal

```
adjusted_prob = ML_probability × 0.85 + sentiment × 0.15

≥ 0.6 → BUY       confidence = |prob − 0.5| × 200%
≤ 0.4 → SELL
else  → HOLD
```

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🏆 **Top Picks** | Top 2 BUY/SELL with company profiles, price targets, news |
| 📋 **All / Buy / Hold / Sell** | Signal cards with confidence %, price change |
| 🌡️ **Heatmap** | Color-coded sector sentiment grid |
| 🔍 **Search Stock** | Analyze ANY NSE ticker on-demand |

---

## 🔧 Configuration

| Setting | Value | Location |
|---------|-------|----------|
| Stock universe | 49 NIFTY 50 tickers | `config.py` |
| Data range | Last 365 days | `config.py` |
| Train/Val/Test | 60% / 20% / 20% | `trainer.py` |
| Walk-forward | 120d train, 30d test, 5d embargo | `trainer.py` |
| BUY threshold | probability ≥ 0.6 | `predictor.py` |
| SELL threshold | probability ≤ 0.4 | `predictor.py` |
| Backtest slippage | 0.05% | `backtester.py` |
| Backtest STT | 0.1% on sell | `backtester.py` |
| Cache staleness | 12h (stocks), 4h (news) | `config.py` |

---

## 📦 Project Structure

```
├── backend/
│   ├── config.py                    # Configuration & NIFTY 50 tickers
│   ├── main.py                      # CLI orchestrator
│   ├── api/server.py                # Flask REST API
│   ├── scraper/
│   │   ├── stock_scraper.py         # yfinance data + caching
│   │   ├── news_scraper.py          # Google News RSS + sentiment
│   │   └── company_insights.py      # Company profiles
│   ├── features/
│   │   ├── feature_engineer.py      # 70+ features + Triple Barrier
│   │   ├── macro_features.py        # India VIX, USD/INR, Gold...
│   │   └── finbert_sentiment.py     # FinBERT / VADER fallback
│   └── models/
│       ├── trainer.py               # 7 models + stacking + walk-forward
│       ├── evaluator.py             # Held-out test evaluation
│       ├── predictor.py             # Signal generation
│       ├── on_demand.py             # Custom stock analysis
│       ├── backtester.py            # Trading simulation
│       └── risk_calculator.py       # VaR / CVaR
├── frontend/
│   └── src/components/
│       ├── Dashboard.jsx            # Main layout + tabs
│       ├── TopPicks.jsx             # Top picks with insights
│       ├── CustomSearch.jsx         # On-demand analysis
│       ├── SentimentHeatmap.jsx     # Sector sentiment grid
│       ├── StockCard.jsx            # Signal cards
│       ├── PredictionChart.jsx      # Price chart
│       ├── ModelComparison.jsx      # Model metrics table
│       └── NewsPanel.jsx            # News headlines
└── .gitignore
```

---

## 📋 Dependencies

| Package | Purpose |
|---------|---------|
| `flask`, `flask-cors` | REST API |
| `yfinance` | Stock & macro data |
| `pandas`, `numpy` | Data processing |
| `scikit-learn` | ML models & preprocessing |
| `xgboost`, `lightgbm` | Gradient boosting |
| `ta` | Technical indicators |
| `vaderSentiment` | Sentiment analysis |
| `statsmodels` | Stationarity tests |
| `scipy` | Parametric VaR |
| `tensorflow` | LSTM (optional) |
| `transformers` + `torch` | FinBERT (optional, ~2GB) |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Stock market predictions are inherently uncertain — always do your own research before making investment decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.