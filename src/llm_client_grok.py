"""
Dünner Wrapper um die Grok API (xAI). Läuft ohne XAI_API_KEY im
Mock-Modus weiter.
"""
from __future__ import annotations

import os
import random

_API_KEY = os.environ.get("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")
_client = None

if _API_KEY:
    try:
        from openai import OpenAI
        _client = OpenAI(
            api_key=_API_KEY,
            base_url="https://api.x.ai/v1",
        )
    except ImportError:
        print("[llm_client] 'openai' Paket nicht installiert -- falle auf Mock-Modus zurück.")

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
            model="grok-2-latest",
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