"""
Kleines MLP, das aus den Turn-Features die Wahrscheinlichkeit schätzt,
dass das Maskottchen jetzt reagieren sollte.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from features import N_FEATURES

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "trigger_model.pt"


class TriggerNet(nn.Module):
    def __init__(self, in_dim: int = N_FEATURES, hidden: int = 64):
        # hidden=64 statt vorher 16: bei 392 Input-Dims (8 handgebaut + 384
        # Embedding) braucht das Netz mehr Kapazität, um die semantischen
        # Embedding-Dimensionen sinnvoll zu komprimieren, bevor die
        # handgebauten Features mit hineinspielen.
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),  # bei so vielen Input-Dims und wenig Trainingsdaten: Overfitting-Schutz
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits)


def load_trigger_model() -> TriggerNet:
    model = TriggerNet()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    else:
        print(f"[trigger] Kein trainiertes Modell unter {MODEL_PATH} gefunden — "
              f"nutze zufällig initialisierte Gewichte. Erst 'python src/train_trigger.py' ausführen.")
    model.eval()
    return model


@torch.no_grad()
def should_react(model: TriggerNet, features: torch.Tensor, threshold: float = 0.40) -> tuple[bool, float]:
    # threshold=0.40: per tune_threshold() (train_trigger.py) gegen
    # data/manual_validation.json ermittelt. Sweep-Ergebnis zeigte ein
    # Plateau 0.30-0.40 mit identischem F1=0.87 (precision=100%,
    # recall=76.9%) -- 0.40 als Mitte des Plateaus gewählt statt des
    # Randwerts 0.30, für mehr Robustheit gegenüber neuen Trainingsdaten.
    # Nach jedem Neutraining den Sweep erneut prüfen, ob sich das ändert.
    prob = model(features.unsqueeze(0)).item()
    return prob >= threshold, prob
