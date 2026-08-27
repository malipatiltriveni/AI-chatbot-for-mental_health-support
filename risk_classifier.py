"""
risk_classifier.py

Trains and serves a 3-class risk classifier: normal / distress / crisis.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from crisis_responses import contains_crisis_language

MODEL_PATH = "data/risk_classifier.joblib"
DEFAULT_DATA_PATH = "data/seed_training_data.csv"

LABELS = ["normal", "distress", "crisis"]


def train(data_path: str = DEFAULT_DATA_PATH, model_path: str = MODEL_PATH):
    df = pd.read_csv(data_path)
    assert set(df["label"].unique()) <= set(LABELS), "Unexpected label found"

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, preds, labels=LABELS))
    print("=== Confusion Matrix (rows=true, cols=pred) ===")
    print(pd.DataFrame(
        confusion_matrix(y_test, preds, labels=LABELS),
        index=[f"true_{l}" for l in LABELS],
        columns=[f"pred_{l}" for l in LABELS],
    ))

    crisis_mask = y_test == "crisis"
    if crisis_mask.sum() > 0:
        crisis_recall = (preds[crisis_mask.values] == "crisis").mean()
        print(f"\nCrisis-class recall (1 - false negative rate): {crisis_recall:.2%}")

    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")
    return pipeline


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def classify_risk(text: str, model=None) -> str:
    if contains_crisis_language(text):
        return "crisis"

    if model is None:
        model = load_model()
    return model.predict([text])[0]


if __name__ == "__main__":
    train()