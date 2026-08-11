"""
Dünner Wrapper um die Gemini API (Google). Läuft ohne GEMINI_API_KEY im
Mock-Modus weiter.
"""
from __future__ import annotations

import os
import random

_API_KEY = os.environ.get("GEMINI_API_KEY")
_client = None

if _API_KEY:
    try:
        from google import genai
        from google.genai import types
        _client = genai.Client(api_key=_API_KEY)
    except ImportError:
        print("[llm_client] 'google-genai' Paket nicht installiert -- falle auf Mock-Modus zurück.")

MOCK_REPLIES = [
    "Haha, klassisch. *rollt digital mit den Augen*",
    "Ich hab zugehört -- und ich hab dazu was zu sagen: nice!",
    "Moment, DAS musste ich kommentieren.",
    "Pixel meldet sich: läuft bei euch, oder?",
    "Ganz ehrlich? Bisschen chaotisch hier, aber ich mag's.",
]


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    if _client is not None:
        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text

    return random.choice(MOCK_REPLIES)


def is_mock_mode() -> bool:
    return _client is None