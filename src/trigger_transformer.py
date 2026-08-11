"""
Trigger-Modell Variante C: eigener Token-Embedding + Transformer-Encoder
(custom_transformer.TextEncoder), end-to-end zusammen mit dem
Klassifikations-Kopf trainiert -- im Gegensatz zu trigger.py
(Variante B), das ein eingefrorenes, vortrainiertes Satz-Embedding nutzt.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from custom_transformer import VOCAB_PATH, SimpleTokenizer, TextEncoder

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "trigger_model_transformer.pt"


class TriggerNetTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        handcrafted_dim: int = 8,
        token_embed_dim: int = 32,
        hidden: int = 64,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, embed_dim=token_embed_dim)
        in_dim = handcrafted_dim + token_embed_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self, handcrafted: torch.Tensor, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        text_vec = self.text_encoder(token_ids, attention_mask)
        x = torch.cat([handcrafted, text_vec], dim=-1)
        logits = self.head(x).squeeze(-1)
        return torch.sigmoid(logits)


def load_trigger_model_transformer() -> tuple[TriggerNetTransformer, SimpleTokenizer] | tuple[None, None]:
    """Lädt Modell + Vokabular, falls train_trigger_transformer.py schon
    gelaufen ist. Gibt (None, None) zurück, wenn noch nicht trainiert --
    der Aufrufer (app.py) entscheidet dann, wie er das anzeigt/abfängt."""
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        return None, None
    tokenizer = SimpleTokenizer.load(VOCAB_PATH)
    model = TriggerNetTransformer(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def should_react(
    model: TriggerNetTransformer,
    tokenizer: SimpleTokenizer,
    handcrafted: torch.Tensor,
    text: str,
    threshold: float = 0.40,
) -> tuple[bool, float]:
    """Pendant zu trigger.should_react() für Variante C -- gleiche
    Signatur-Idee (gibt (reagieren?, Wahrscheinlichkeit) zurück), nur dass
    hier zusätzlich der Rohtext getokenisiert werden muss."""
    token_ids, mask = tokenizer.encode(text)
    prob = model(handcrafted.unsqueeze(0), token_ids.unsqueeze(0), mask.unsqueeze(0)).item()
    return prob >= threshold, prob
