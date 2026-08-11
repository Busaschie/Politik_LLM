"""
Dünner Wrapper um die Groq API.
"""
from __future__ import annotations

import os
import random

# 1. Erst versuchen, den Key aus den Umgebungsvariablen zu holen
_API_KEY = os.environ.get("GROQ_API_KEY")

# 2. Falls nicht da, versuchen aus Streamlit Secrets (.streamlit/secrets.toml) zu laden
if not _API_KEY:
    try:
        import streamlit as st
        _API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

_client = None

# 3. Erst JETZT den Client initialisieren, nachdem _API_KEY final feststeht
if _API_KEY:
    try:
        from groq import Groq
        _client = Groq(api_key=_API_KEY)
        print("[llm_client] Groq Client erfolgreich initialisiert!")
    except ImportError:
        print("[llm_client] 'groq' Paket nicht installiert -- falle auf Mock-Modus zurück.")
else:
    print("[llm_client] Kein API Key gefunden -- Mock-Modus aktiv.")


MOCK_REPLIES = [
    "Haha, klassisch. *rollt digital mit den Augen*",
    "Ich hab zugehört -- und ich hab dazu was zu sagen: nice!",
    "Moment, DAS musste ich kommentieren.",
    "Pixel meldet sich: läuft bei euch, oder?",
    "Ganz ehrlich? Bisschen chaotisch hier, aber ich mag's.",
]


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    if _client is not None:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    return random.choice(MOCK_REPLIES)


def is_mock_mode() -> bool:
    return _client is None