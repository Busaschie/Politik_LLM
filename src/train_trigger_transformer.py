# src/train_trigger_transformer.py
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.custom_transformer import CustomTransformer, Vocabulary
from src.llm_client import generate

BASE_DIR = Path(__file__).resolve().parent.parent
DISCUSSIONS_DIR = BASE_DIR / "data" / "discussions"
MODEL_SAVE_PATH = BASE_DIR / "data" / "custom_transformer_model.pt"
VOCAB_SAVE_PATH = BASE_DIR / "data" / "transformer_vocab.json"


def load_all_discussions() -> list[str]:
    """
    Lädt alle gespeicherten Diskussionen aus dem Ordner data/discussions.
    Unterstützt sowohl neue .json-Dateien als auch alte .txt-Dateien.
    """
    texts = []
    if not DISCUSSIONS_DIR.exists():
        return texts

    # 1. JSON-Dateien einlesen
    for json_file in DISCUSSIONS_DIR.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "full_text" in data:
                    texts.append(data["full_text"])
                elif "rounds" in data:
                    full_str = "\n".join([f"{item['speaker']}: {item['text']}" for item in data["rounds"]])
                    texts.append(full_str)
        except Exception as e:
            print(f"Fehler beim Laden von {json_file.name}: {e}")

    # 2. Falls noch alte TXT-Dateien vorhanden sind, auch diese mitladen
    for txt_file in DISCUSSIONS_DIR.glob("*.txt"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
        except Exception as e:
            print(f"Fehler beim Laden von {txt_file.name}: {e}")

    return texts


class DebateDataset(Dataset):
    def __init__(self, texts: list[str], vocab: Vocabulary, max_len: int = 64):
        self.examples = []
        for text in texts:
            tokens = vocab.encode(text)
            if len(tokens) < max_len:
                tokens += [vocab.pad_id] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
            self.examples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def retrain_custom_transformer(m_prefs: dict, t_prefs: dict, status_container=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    discussions = load_all_discussions()
    if not discussions:
        if status_container:
            status_container.error("Keine Diskussionsdaten im Ordner 'data/discussions/' gefunden!")
        return [], [], [], []

    # 1. Vokabular dynamisch aufbauen
    vocab = Vocabulary()
    vocab.build_vocab(discussions)
    vocab.save(str(VOCAB_SAVE_PATH))

    # 2. Train / Test Split
    full_dataset = DebateDataset(discussions, vocab)
    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size

    if train_size <= 0:
        if status_container:
            status_container.error("Zu wenige Daten für Train/Test-Split!")
        return [], [], [], []

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # 3. Modell mit dynamischen Einstellungen aus der UI
    model = CustomTransformer(
        vocab_size=len(vocab),
        embed_dim=m_prefs["embed_dim"],
        num_heads=m_prefs["num_heads"],
        num_layers=m_prefs["pipelen"],
        dropout=m_prefs["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t_prefs["lr"], weight_decay=t_prefs["decay"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    # 4. Training & Validation Loop
    step = 0
    loss_history = []
    val_loss_history = []
    ppl_history = []
    acc_history = []

    model.train()
    while step < t_prefs["steps"]:
        for batch in train_loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            # 1. FORWARD PASS (Vorwärtslauf):
            # Das Modell macht eine Vorhersage und berechnet den Fehler (Loss).
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))

            # 2. GRADIENTEN ZURÜCKSETZEN:
            # Verhindert, dass sich alte mathematische Steigungen aufsummieren.
            optimizer.zero_grad()

            # 3. BACKPROPAGATION (Rückwärtslauf):
            # PyTorch berechnet automatisch die Steigung (Gradienten) für jedes einzelne Gewicht.
            loss.backward()

            # 4. GEWICHTE ANPASSEN (Optimizer Step):
            # Der Optimizer (AdamW) nutzt die eben berechneten Gradienten, um das Modell schauer zu machen.
            optimizer.step()

            step += 1

            if step % t_prefs["interval"] == 0 or step == t_prefs["steps"]:
                loss_val = loss.item()

                model.eval()
                val_loss, correct, total_tokens = 0.0, 0, 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        v_in, v_target = val_batch[:, :-1], val_batch[:, 1:]
                        v_logits = model(v_in)

                        v_loss = criterion(v_logits.reshape(-1, len(vocab)), v_target.reshape(-1))
                        val_loss += v_loss.item()

                        preds = v_logits.argmax(dim=-1)
                        mask = (v_target != vocab.pad_id)
                        correct += (preds[mask] == v_target[mask]).sum().item()
                        total_tokens += mask.sum().item()

                val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else loss_val
                accuracy = (correct / total_tokens) if total_tokens > 0 else 0.0
                perplexity = math.exp(val_loss_avg) if val_loss_avg < 20 else float("inf")

                loss_history.append(loss_val)
                val_loss_history.append(val_loss_avg)
                ppl_history.append(perplexity)
                acc_history.append(accuracy)

                if status_container:
                    status_container.write(
                        f"📍 **Schritt {step}/{t_prefs['steps']}** — "
                        f"Train Loss: `{loss_val:.4f}` | "
                        f"Test Loss: `{val_loss_avg:.4f}` | "
                        f"Accuracy: `{accuracy * 100:.1f}%` | "
                        f"Perplexity: `{perplexity:.2f}`"
                    )
                model.train()

            if step >= t_prefs["steps"]:
                break

    model.save_model(str(MODEL_SAVE_PATH))
    return loss_history, val_loss_history, ppl_history, acc_history


def get_training_recommendation(m_prefs: dict, t_prefs: dict, results: dict) -> str:
    """Nutzt das LLM, um die Trainingsergebnisse zu analysieren und Empfehlungen zu geben."""
    sys_prompt = (
        "Du bist ein Senior Machine Learning Engineer für Transformer-Modelle. "
        "Analysiere die Hyperparameter und die Ergebnisse des PyTorch-Trainings. "
        "Gib kurze, präzise Empfehlungen (max. 3-4 Bulletpoints), wie das Modell "
        "weiter verbessert werden kann (z.B. Overfitting vermeiden, Kapazität erhöhen, Lernrate anpassen)."
    )

    user_prompt = f"""
    KONFIGURATION:
    - Layers (pipelen): {m_prefs.get('pipelen')}
    - Embedding Dim: {m_prefs.get('embed_dim')}
    - Attention Heads: {m_prefs.get('num_heads')}
    - Dropout: {m_prefs.get('dropout')}
    - Steps: {t_prefs.get('steps')}
    - Learning Rate: {t_prefs.get('lr')}
    - Weight Decay: {t_prefs.get('decay')}

    ERGEBNISSE:
    - Finaler Train Loss: {results.get('train_loss', 0):.4f}
    - Finaler Test Loss: {results.get('test_loss', 0):.4f}
    - Finale Accuracy: {results.get('accuracy', 0) * 100:.1f}%
    - Finale Perplexity: {results.get('perplexity', 0):.2f}

    Was sind deine konkreten Empfehlungen für den nächsten Trainingslauf?
    """

    try:
        return generate(system_prompt=sys_prompt, user_prompt=user_prompt)
    except Exception as e:
        return f"⚠️ KI-Empfehlung konnte nicht generiert werden: {str(e)}"


def sample_text(prompt: str, m_prefs: dict, max_tokens: int = 20, temperature: float = 0.8) -> str:
    """Generiert Text mit dem trainierten PyTorch Transformer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_SAVE_PATH.exists() or not VOCAB_SAVE_PATH.exists():
        return "⚠️ Noch kein trainiertes Modell gefunden. Bitte zuerst das Training ausführen!"

    vocab = Vocabulary()
    vocab.load(str(VOCAB_SAVE_PATH))

    # 🔥 DYNAMISCHE ARCHITEKTUR AUS m_prefs NUTZEN:
    model = CustomTransformer(
        vocab_size=len(vocab),
        embed_dim=m_prefs.get("embed_dim", 128),
        num_heads=m_prefs.get("num_heads", 4),
        num_layers=m_prefs.get("pipelen", 8),
        dropout=m_prefs.get("dropout", 0.2)
    ).to(device)

    model.load_model(str(MODEL_SAVE_PATH), device)
    model.eval()

    tokens = vocab.encode(prompt)
    if not tokens:
        tokens = [vocab.unk_id]

    generated = list(tokens)

    with torch.no_grad():
        for _ in range(max_tokens):
            input_tensor = torch.tensor([generated[-64:]], dtype=torch.long).to(device)
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_token_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

    words = [vocab.idx2word.get(idx, "<UNK>") for idx in generated]
    return " ".join(words)