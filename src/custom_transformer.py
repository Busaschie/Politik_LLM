# src/custom_transformer.py
import json
import torch
import torch.nn as nn
from pathlib import Path


class Vocabulary:
    def __init__(self, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.pad_id = 0
        self.unk_id = 1

    def build_vocab(self, texts: list[str]):
        """Baut das Vokabular aus einer Liste von Texten auf."""
        idx = len(self.word2idx)
        for text in texts:
            for word in text.split():
                word = word.lower()
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def encode(self, text: str) -> list[int]:
        """Wandelt Text in Token-IDs um."""
        return [self.word2idx.get(word.lower(), self.unk_id) for word in text.split()]

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f, ensure_ascii=False)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class CustomTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        # 1. EMBEDDING-LAYER DEFINIEREN:
        # Erstellt eine Tabelle mit [vocab_size x embed_dim] lernbaren Werten
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embedding(x)
        out = self.transformer(out)
        logits = self.fc_out(out)
        return logits

    def save_model(self, filepath: str):
        """Speichert die PyTorch Modellgewichte."""
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str, device: torch.device):
        """Lädt die PyTorch Modellgewichte."""
        self.load_state_dict(torch.load(filepath, map_location=device))