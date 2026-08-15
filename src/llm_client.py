"""
Dünner Wrapper um die Groq API.
"""
from __future__ import annotations

import os

# 1. API-Key aus Umgebungsvariablen oder Streamlit Secrets laden
_API_KEY = os.environ.get("GROQ_API_KEY")

if not _API_KEY:
    try:
        import streamlit as st
        _API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        _API_KEY = None

# 2. Groq-Client initialisieren
_client = None

if _API_KEY:
    try:
        from groq import Groq
        _client = Groq(api_key=_API_KEY)
        print("[llm_client] Groq Client erfolgreich initialisiert!")
    except ImportError:
        print("[llm_client] 'groq' Paket ist nicht installiert.")
else:
    print("[llm_client] Kein GROQ_API_KEY gefunden.")


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    """
    Generiert eine Antwort via Groq LLM.
    """
    if _client is None:
        return "⚠️ Fehler: Kein Groq API-Key konfiguriert oder Client nicht verfügbar."

    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[llm_client] API-Fehler: {e}")
        return f"⚠️ API-Fehler bei der Anfrage: {str(e)}"


def is_mock_mode() -> bool:
    """Gibt True zurück, wenn kein funktionierender API-Client vorhanden ist."""
    return _client is None