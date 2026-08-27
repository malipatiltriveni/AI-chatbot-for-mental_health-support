"""
crisis_responses.py

Fixed, non-generative responses for crisis-level messages.
This module NEVER calls an LLM. Every response here is hardcoded on
purpose -- crisis situations should never depend on a model generating
text on the fly.
"""

from datetime import datetime

CRISIS_KEYWORDS = [
    "kill myself", "end my life", "suicide", "suicidal",
    "want to die", "wanna die", "no reason to live",
    "better off dead", "end it all", "can't go on",
    "hurt myself", "self harm", "self-harm", "cutting myself",
    "overdose", "not worth living", "give up on life",
]

CRISIS_RESOURCES = {
    "US": {
        "name": "988 Suicide & Crisis Lifeline",
        "phone": "988",
        "text": "Text 'HELLO' to 741741 (Crisis Text Line)",
        "url": "https://988lifeline.org",
    },
}


def contains_crisis_language(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in CRISIS_KEYWORDS)


def get_crisis_response(region: str = "US") -> str:
    resource = CRISIS_RESOURCES.get(region, CRISIS_RESOURCES["US"])
    return (
        "It sounds like you might be going through something really "
        "difficult right now, and I want to make sure you get real support -- "
        "more than I can safely give you here.\n\n"
        f"**{resource['name']}**\n"
        f"Call or text: {resource['phone']}\n"
        f"{resource['text']}\n\n"
        "If you are in immediate danger, please contact your local "
        "emergency services.\n\n"
        "This app is not a substitute for professional care."
    )


def log_crisis_event(log_path: str = "data/crisis_events.log"):
    with open(log_path, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()} - crisis_path_triggered\n")