"""
Lazy-geladenes Sentence-Embedding-Modell (Variante B).

Wird von features.py genutzt, um den vollen Bedeutungsvektor eines
Turns zusätzlich zu den handgebauten Features ins Trigger-Modell zu geben.

Warum lazy + Singleton: das Modell (~90 MB) soll nur EINMAL pro Prozess
geladen werden, nicht bei jedem Feature-Extraktions-Aufruf -- sonst
wird main.py bei 20 Turns unbenutzbar langsam.
"""
from __future__ import annotations

import torch

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # feste Ausgabedimension dieses Modells

_model = None
_embedding_available = True


def _get_model():
    global _model, _embedding_available
    if _model is not None or not _embedding_available:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"[embeddings] '{EMBEDDING_MODEL_NAME}' geladen ({EMBEDDING_DIM} Dimensionen).")
    except Exception as exc:  # ImportError ODER Netzwerkfehler beim Modell-Download
        print(f"[embeddings] Konnte Embedding-Modell nicht laden ({exc!r}). "
              f"Falle auf Null-Vektor zurück -- Trigger-Modell nutzt dann effektiv "
              f"nur die 8 handgebauten Features.")
        _embedding_available = False
        _model = None
    return _model


def embed(text: str) -> torch.Tensor:
    model = _get_model()
    if model is None:
        return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)
    vec = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    return vec.to(dtype=torch.float32)
