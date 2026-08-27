"""
llm_chat.py

Wraps the Google Gemini API (free tier) for the supportive-chat layer.
This module must NEVER be called when risk_classifier.classify_risk()
returns "crisis" -- that path is handled entirely by crisis_responses.py
with zero model generation involved.

Setup:
  1. Get a free API key at https://aistudio.google.com/apikey
  2. Set it as an environment variable: GEMINI_API_KEY
"""

import os
import google.generativeai as genai

SYSTEM_PROMPT = """You are a supportive listening companion in a mental \
health self-help app. Follow these rules strictly, with no exceptions:

1. You are NOT a therapist and must never claim to be one.
2. Never diagnose any condition (e.g. "it sounds like you have depression").
3. Never give medical or medication advice.
4. Use reflective listening: validate feelings, ask gentle open questions,
   avoid being clinical or robotic.
5. For anything beyond mild, everyday stress, explicitly encourage the
   person to talk to a licensed therapist or counselor.
6. Keep responses brief (3-5 sentences). This is a supplement to real
   support, not a replacement for it.
7. If the user expresses anything resembling hopelessness, self-harm, or
   suicidal ideation, do not continue engaging conversationally -- this
   should never reach you, as the app routes those messages to a fixed
   crisis response before calling you. If it ever does reach you anyway,
   immediately and only respond by telling them to contact 988 (US) or
   their local emergency services, and nothing else.
"""

_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        _configured = True


def get_supportive_response(user_message: str, conversation_history=None,
                             model: str = "gemini-3.6-flash") -> str:
    """conversation_history: list of {"role": "user"/"assistant", "content": str}
    Only call this for messages already classified as 'normal' or 'distress'.
    """
    _ensure_configured()

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
    )

    # Gemini's chat history format uses "model" instead of "assistant"
    history = []
    for m in (conversation_history or []):
        role = "model" if m["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [m["content"]]})

    chat = gemini_model.start_chat(history=history)
    response = chat.send_message(user_message)
    return response.text