import json
import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class Vocabulary:
    """Wrapper um den Hugging Face BPE-Subword-Tokenizer.

    Ermöglicht Subword-Tokenisierung (Byte-Pair Encoding), um Out-of-Vocabulary (OOV)
    Probleme zu vermeiden und deutsche Komposita besser abzubilden.
    """

    def __init__(self, tokenizer_path: str = None, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_id = 0
        self.unk_id = 1

        if tokenizer_path and Path(tokenizer_path).exists():
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            # Standard-Setup für ein neues BPE-Modell
            self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
            self.tokenizer.pre_tokenizer = Whitespace()

    def build_vocab(self, texts: list[str], vocab_size: int = 2000):
        """Trainiert das BPE-Subword-Vokabular auf einer Liste von Texten."""
        trainer = BpeTrainer(
            special_tokens=[self.pad_token, self.unk_token],
            vocab_size=vocab_size
        )
        self.tokenizer.train_from_iterator(texts, trainer)

    def encode(self, text: str) -> list[int]:
        """Wandelt Text in Subword-Token-IDs um."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer wurde nicht geladen oder trainiert!")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Wandelt eine Liste von Token-IDs zurück in lesbaren Text um."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer wurde nicht geladen oder trainiert!")
        return self.tokenizer.decode(ids)

    def save(self, filepath: str):
        """Speichert das trainierte BPE-Tokenizer-Modell als JSON."""
        self.tokenizer.save(str(filepath))

    def load(self, filepath: str):
        """Lädt ein bestehendes BPE-Tokenizer-Modell aus einer Datei."""
        self.tokenizer = Tokenizer.from_file(str(filepath))

    def __len__(self):
        """Gibt die Vokabelgröße (Anzahl der Subwords + Spezial-Tokens) zurück."""
        if self.tokenizer is None:
            return 2
        return self.tokenizer.get_vocab_size()


class CustomTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        # 1. EMBEDDING-LAYER DEFINIEREN:
        # Erstellt eine Tabelle mit [vocab_size x embed_dim] lernbaren Werten
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Definiert eine einzelne Encoder-Schicht (Layer) der Transformer-Architektur.
        # Diese Schicht kombiniert Multi-Head Self-Attention mit einem Feed-Forward Neural Network (FFNN).
        encoder_layer = nn.TransformerEncoderLayer(
            # d_model: Dimension der Embeddings / Feature-Vektoren (z. B. 128 oder 512).
            # Bestimmt, wie groß die Vektor-Repräsentation für jedes einzelne Token ist.
            d_model=embed_dim,

            # nhead: Anzahl der parallelen Attention-Köpfe (Attention Heads).
            # Erlaubt dem Modell, gleichzeitig verschiedene Beziehungen zwischen Wörtern
            # zu lernen (z.B. grammatikalische vs. semantische Verbindungen).
            # Wichtig: d_model muss ohne Rest durch num_heads teilbar sein!
            nhead=num_heads,

            # dropout: Wahrscheinlichkeit (z. B. 0.1 für 10%), mit der Neuronen während
            # des Trainings zufällig deaktiviert werden, um Overfitting (Überanpassung) zu verhindern.
            dropout=dropout,

            # activation: Aktivierungsfunktion im Feed-Forward-Netzwerk innerhalb des Layers.
            # "gelu" (Gaussian Error Linear Unit) ist glatter als ReLU und der Standard bei
            # modernen Sprachmodellen wie BERT, GPT und RoBERTa.
            activation="gelu",

            # batch_first: Gibt an, wie die Daten-Tensoren strukturiert sind.
            # True  -> Tensor-Shape: [Batch_Size, Sequence_Length, Embedding_Dim]
            # False -> Tensor-Shape: [Sequence_Length, Batch_Size, Embedding_Dim] (PyTorch-Oldschool-Standard)
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("\n" + "=" * 50)
        print("🔍 --- INSIDE TRANSFORMER FORWARD PASS ---")

        # 1. Input Tokens (Subword Integer IDs)
        print(f"1. Input Tokens (Shape {x.shape}):\n   {x[0, :8].tolist()} ...")

        # 2. Embedding Layer (Konvertierung in hochdimensionale Vektoren)
        out = self.embedding(x)
        print(f"2. Embedding Vectors (Shape {out.shape}):")
        print(f"   Ausschnitt des 1. Subword-Vektors (erste 5 Dimensionen):\n   {out[0, 0, :5].detach().tolist()}")

        # 3. Transformer Encoder (Self-Attention & Contextualization)
        out = self.transformer(out)
        print(f"3. Transformer Output Tensoren (Shape {out.shape})")

        # 4. Linear Output Head (Logits über das gesamte Subword-Vokabular)
        logits = self.fc_out(out)
        print(f"4. Unnormalized Logits (Shape {logits.shape}):")
        print(f"   Logits für das nächste Token (erste 5 Subword-IDs):\n   {logits[0, -1, :5].detach().tolist()}")
        print("=" * 50 + "\n")

        return logits

    def save_model(self, filepath: str):
        """Speichert die PyTorch Modellgewichte."""
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str, device: torch.device):
        """Lädt die PyTorch Modellgewichte."""
        self.load_state_dict(torch.load(filepath, map_location=device))