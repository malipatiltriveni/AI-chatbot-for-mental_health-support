# 🌱 Mind Companion — Mental Health Support Chatbot (Student Project)

A safety-first mental health support tool: mood check-ins, journaling with sentiment trends, CBT exercise suggestions, and a gated supportive chat powered by Google Gemini.

> ⚠️ **This is a student portfolio project, not a medical device.**
> It is not a replacement for therapy or professional care.
> If you are in crisis, contact 988 (US) or your local emergency services.

## Demo



![Chat interface showing a supportive response](screenshots/chat_demo.png)



*(Add more screenshots here — crisis path, mood trends, CBT suggestions)*

## Why this project

Most student "mental health chatbot" projects let an LLM handle every message, including crisis language — which is a real safety risk. This project is built around one core design decision instead:

**The crisis path never touches the LLM.** Every message is checked against a risk classifier first. If it's flagged as crisis-level, the app responds with a fixed, hardcoded message containing real crisis resources — no model generation involved, no chance of hallucination.

## Architecture

User message
     │
     ▼
Risk classifier (risk_classifier.py)
  - keyword safety net (crisis_responses.py) checked first
  - ML model (TF-IDF + Logistic Regression) as a second signal
  - if EITHER flags "crisis" → crisis path
     │
     ├── crisis   → crisis_responses.py (fixed text, hotline info, NO LLM call)
     ├── distress → Gemini chat (strict system prompt) + CBT suggestion
     └── normal   → Gemini chat (strict system prompt)

## Features

- **Risk classification** — 3-class model (normal / distress / crisis) combining a trained ML classifier with a hardcoded keyword safety net
- **Crisis response path** — fixed, non-generative, always shows real crisis hotline info
- **Gated supportive chat** — Google Gemini, strictly prompted to never diagnose, never give medical advice, and always redirect to professional help for anything beyond mild stress
- **CBT exercise library** — retrieved from a fixed, curated set of exercises rather than generated, for consistency and safety
- **Mood check-in & journaling** — with VADER-based sentiment scoring
- **Trend visualization** — mood and sentiment charted over time

## Tech stack

Python, Streamlit, scikit-learn (TF-IDF + Logistic Regression), VADER (sentiment analysis), Google Gemini API, SQLite

## Setup

Clone the repo and install dependencies:

git clone your-repo-url-here
cd mental_health_support_app
pip install -r requirements.txt

Train the risk classifier (uses the bundled seed dataset):

python risk_classifier.py

Get a free Gemini API key at https://aistudio.google.com/apikey, then set it (PowerShell):

$env:GEMINI_API_KEY="your-key-here"

Run the app:

python -m streamlit run app.py

## About the training data

data/seed_training_data.csv has about 30 rows — just enough to make the pipeline run end-to-end. It is not enough data for a trustworthy classifier. Training on it shows weak crisis-class recall, which is expected. This is why the keyword safety net in crisis_responses.py exists as a second line of defense — the ML model is never the only thing standing between a user and a crisis response.

To improve it: merge in a real dataset (e.g. Kaggle's "Suicide and Depression Detection" or HuggingFace's "Dreaddit") in the same text,label format, then re-run python risk_classifier.py and check the crisis-recall number printed at the end.

## What this project deliberately does NOT do

- Does not diagnose any condition
- Does not give medical or medication advice
- Does not let the LLM handle crisis messages — that's a fixed lookup
- Does not claim clinical validation of any kind

## Evaluation metrics

- Crisis-class false negative rate — the most important number in this project, printed automatically by risk_classifier.py
- Manual review of sampled chat responses for appropriateness
- Response latency

## Limitations

- Training data is a small seed set, not production-scale
- Keyword list is a starting point, not exhaustive
- No clinical validation — this is a self-awareness/support tool, not a diagnostic or therapeutic instrument
- LLM responses, while prompted strictly, are not 100% guaranteed safe — this is why the crisis path bypasses the LLM entirely

## Author

Built by Your Name as a portfolio project exploring safety-first design in AI-assisted mental health tools.
