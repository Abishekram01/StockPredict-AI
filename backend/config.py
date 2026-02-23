"""
Configuration for the Indian Stock Trading Bot.
Contains NIFTY 50 tickers, model settings, and paths.
"""
import os
from datetime import datetime, timedelta

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DATASET_DIR = os.path.join(BASE_DIR, "..", "Dataset")   # User-visible dataset folder
STOCK_DATA_DIR = DATASET_DIR                            # CSVs stored here
MODEL_DIR = os.path.join(DATA_DIR, "models")
NEWS_CACHE_DIR = os.path.join(DATA_DIR, "news")

for d in [DATASET_DIR, MODEL_DIR, NEWS_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Cache / Staleness ───────────────────────────────────────────────────
DATASET_STALE_HOURS = 12   # Re-scrape stock data if CSV older than this
NEWS_STALE_HOURS = 4       # Re-fetch news if cache older than this

# ─── Date Range ──────────────────────────────────────────────────────────
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

# ─── NIFTY 50 Tickers (Yahoo Finance format: .NS suffix) ────────────────
NIFTY50_TICKERS = {
    "RELIANCE.NS":  "Reliance Industries",
    "TCS.NS":       "Tata Consultancy Services",
    "HDFCBANK.NS":  "HDFC Bank",
    "INFY.NS":      "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS":"Hindustan Unilever",
    "ITC.NS":       "ITC Limited",
    "SBIN.NS":      "State Bank of India",
    "BHARTIARTL.NS":"Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS":        "Larsen & Toubro",
    "AXISBANK.NS":  "Axis Bank",
    "ASIANPAINT.NS":"Asian Paints",
    "MARUTI.NS":    "Maruti Suzuki",
    "HCLTECH.NS":   "HCL Technologies",
    "SUNPHARMA.NS": "Sun Pharma",
    "TITAN.NS":     "Titan Company",
    "BAJFINANCE.NS":"Bajaj Finance",
    "WIPRO.NS":     "Wipro",
    "ULTRACEMCO.NS":"UltraTech Cement",
    "ONGC.NS":      "ONGC",
    "NTPC.NS":      "NTPC",
    "POWERGRID.NS": "Power Grid Corp",
    "TATAMOTORS.NS":"Tata Motors",
    "M&M.NS":       "Mahindra & Mahindra",
    "JSWSTEEL.NS":  "JSW Steel",
    "TATASTEEL.NS": "Tata Steel",
    "ADANIENT.NS":  "Adani Enterprises",
    "ADANIPORTS.NS":"Adani Ports",
    "COALINDIA.NS": "Coal India",
    "BPCL.NS":      "BPCL",
    "GRASIM.NS":    "Grasim Industries",
    "TECHM.NS":     "Tech Mahindra",
    "INDUSINDBK.NS":"IndusInd Bank",
    "HINDALCO.NS":  "Hindalco",
    "EICHERMOT.NS": "Eicher Motors",
    "DRREDDY.NS":   "Dr. Reddy's",
    "DIVISLAB.NS":  "Divi's Laboratories",
    "CIPLA.NS":     "Cipla",
    "APOLLOHOSP.NS":"Apollo Hospitals",
    "BAJAJ-AUTO.NS":"Bajaj Auto",
    "BAJAJFINSV.NS":"Bajaj Finserv",
    "BRITANNIA.NS": "Britannia",
    "HEROMOTOCO.NS":"Hero MotoCorp",
    "NESTLEIND.NS": "Nestle India",
    "SBILIFE.NS":   "SBI Life Insurance",
    "HDFCLIFE.NS":  "HDFC Life",
    "TATACONSUM.NS":"Tata Consumer Products",
    "LTIM.NS":      "LTIMindtree",
    "WIPRO.NS":     "Wipro",
}

# ─── Model Training Settings ────────────────────────────────────────────
TRAIN_TEST_SPLIT = 0.8          # 80% train, 20% test (chronological)
RANDOM_STATE = 42
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_LOOKBACK = 60              # 60-day sequences for LSTM

# ─── API Settings ────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
