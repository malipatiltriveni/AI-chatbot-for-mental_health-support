"""
sentiment_tracker.py

Scores journal entries with VADER (rule-based, no API needed, fast).
Not a clinical tool -- purely for showing the user their own trend
over time.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_entry(text: str) -> float:
    """Returns a compound sentiment score from -1 (very negative)
    to +1 (very positive)."""
    return _analyzer.polarity_scores(text)["compound"]


def label_for_score(score: float) -> str:
    if score >= 0.3:
        return "positive"
    if score <= -0.3:
        return "negative"
    return "neutral"