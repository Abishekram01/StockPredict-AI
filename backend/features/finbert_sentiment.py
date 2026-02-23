"""
FinBERT-based financial sentiment analysis.
Uses ProsusAI/finbert transformer model for accurate financial text classification.
Falls back to VADER if transformers/torch is not installed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try FinBERT, fall back to VADER
_USE_FINBERT = False
_finbert_pipeline = None

try:
    from transformers import pipeline as hf_pipeline
    import torch

    _finbert_pipeline = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1,  # CPU
        truncation=True,
        max_length=512,
    )
    _USE_FINBERT = True
    print("  ✓ FinBERT loaded for financial sentiment analysis")
except (ImportError, Exception):
    _USE_FINBERT = False


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of financial text.
    Returns: {"label": "positive"|"negative"|"neutral", "score": float, "compound": float}

    - If FinBERT is available: uses transformer model
    - Otherwise: falls back to VADER
    """
    if _USE_FINBERT and _finbert_pipeline is not None:
        return _finbert_analyze(text)
    else:
        return _vader_analyze(text)


def _finbert_analyze(text: str) -> dict:
    """FinBERT analysis — understands financial language nuances."""
    try:
        result = _finbert_pipeline(text[:512])[0]
        label = result["label"].lower()  # positive, negative, neutral
        score = result["score"]

        # Map to VADER-like compound: -1 to +1
        if label == "positive":
            compound = score
        elif label == "negative":
            compound = -score
        else:
            compound = 0.0

        return {
            "label": label,
            "score": score,
            "compound": round(compound, 4),
            "engine": "finbert",
        }
    except Exception:
        return _vader_analyze(text)


def _vader_analyze(text: str) -> dict:
    """VADER fallback for when FinBERT is not available."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": abs(compound),
        "compound": round(compound, 4),
        "engine": "vader",
    }


def batch_analyze(texts: list) -> list:
    """Analyze a batch of texts efficiently."""
    if _USE_FINBERT and _finbert_pipeline is not None:
        try:
            results = _finbert_pipeline([t[:512] for t in texts])
            out = []
            for r in results:
                label = r["label"].lower()
                score = r["score"]
                compound = score if label == "positive" else (-score if label == "negative" else 0.0)
                out.append({
                    "label": label, "score": score,
                    "compound": round(compound, 4), "engine": "finbert",
                })
            return out
        except Exception:
            pass

    return [_vader_analyze(t) for t in texts]


def get_engine_name() -> str:
    return "finbert" if _USE_FINBERT else "vader"


if __name__ == "__main__":
    test_texts = [
        "The stock crashed after disappointing earnings report",
        "Breaking through resistance level, bullish momentum continues",
        "Company announces record quarterly revenue growth",
        "RBI raises interest rates, banking stocks fall",
        "Neutral market sentiment, trading sideways",
    ]
    print(f"\nSentiment Engine: {get_engine_name()}\n")
    for text in test_texts:
        r = analyze_sentiment(text)
        print(f"  [{r['label']:8s}] {r['compound']:+.4f}  {text}")
