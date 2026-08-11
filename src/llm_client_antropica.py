"""
Dünner Wrapper um die Anthropic API. Läuft ohne ANTHROPIC_API_KEY im
Mock-Modus weiter (canned/regelbasierte Antworten), damit das restliche
System (Trigger, Kontext, Orchestrierung) jederzeit ohne Kosten/Key
testbar bleibt. Für "echte" Antworten: ANTHROPIC_API_KEY setzen.
"""
from __future__ import annotations

import os
import random
grok-4.1
_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
_client = None

if _API_KEY:
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=_API_KEY)
    except ImportError:
        print("[llm_client] 'anthropic' Paket nicht installiert -- falle auf Mock-Modus zurück.")

MOCK_REPLIES = [
    "Haha, klassisch. *rollt digital mit den Augen*",
    "Ich hab zugehört -- und ich hab dazu was zu sagen: nice!",
    "Moment, DAS musste ich kommentieren.",
    "Pixel meldet sich: läuft bei euch, oder?",
    "Ganz ehrlich? Bisschen chaotisch hier, aber ich mag's.",
]


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    if _client is not None:
        response = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
    # Mock-Modus: deterministisch genug für Demo, aber variabel
    return random.choice(MOCK_REPLIES)


def is_mock_mode() -> bool:
    return _client is None
