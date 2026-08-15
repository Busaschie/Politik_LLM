# src/train_trigger_transformer.py
import json
import math
import re
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
    texts = []
    if not DISCUSSIONS_DIR.exists():
        return texts

    # 1. JSON-Dateien einlesen
    for json_file in DISCUSSIONS_DIR.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "full_text" in data:
                    cleaned = clean_debate_text(data["full_text"])
                    if cleaned:
                        texts.append(cleaned)
                elif "rounds" in data:
                    full_str = "\n".join([f"{item['speaker']}: {item['text']}" for item in data["rounds"]])
                    cleaned = clean_debate_text(full_str)
                    if cleaned:
                        texts.append(cleaned)
        except Exception as e:
            print(f"Fehler beim Laden von {json_file.name}: {e}")

    # 2. Falls noch alte TXT-Dateien vorhanden sind
    for txt_file in DISCUSSIONS_DIR.glob("*.txt"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    cleaned = clean_debate_text(content)
                    if cleaned:
                        texts.append(cleaned)
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


def clean_debate_text(text: str) -> str:
    """Entfernt wiederkehrende Begrüßungen und Höflichkeitsfloskeln aus dem Text."""
    phrases_to_remove = [
        r"guten abend(,? meine damen und herren)?!?",
        r"herzlich willkommen( zu unserer heutigen debatte)?!?",
        r"vielen dank(,? herr moderator|,? frau moderatorin)?!?",
        r"danke für die einladung!?",
        r"vielen dank für die frage!?",
        r"als vertreter der \w+ stehe ich!?",
        r"sehr geehrte damen und herren,!?"
    ]

    cleaned = text
    for pattern in phrases_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Doppelte Leerzeichen aufräumen
    return re.sub(r"\s+", " ", cleaned).strip()


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

    # LERNRATEN-SCHEDULER ERSTELLEN
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_prefs["steps"],
        eta_min=1e-6
    )

    # 🛑 EARLY STOPPING EINSTELLUNGEN
    patience = t_prefs.get("patience", 5)  # Maximale Anzahl an Checks ohne Verbesserung
    best_val_loss = float("inf")
    patience_counter = 0

    # 4. Training & Validation Loop
    step = 0
    loss_history = []
    val_loss_history = []
    ppl_history = []
    acc_history = []

    model.train()
    stop_early = False

    while step < t_prefs["steps"] and not stop_early:
        for batch in train_loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            # 1. FORWARD PASS
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))

            # 2. GRADIENTEN ZURÜCKSETZEN
            optimizer.zero_grad()

            # 3. BACKPROPAGATION
            loss.backward()

            # 4. GRADIENT CLIPPING
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 5. GEWICHTE ANPASSEN
            optimizer.step()

            # 6. LERNRATE ANPASSEN
            scheduler.step()

            step += 1

            # 🧮 Terminal-Log für Mathematik & Softmax (alle 100 Schritte)
            if step % 100 == 0:
                last_token_logits = logits[0, -1, :]
                probs = F.softmax(last_token_logits, dim=-1)

                top_prob, top_idx = torch.topk(probs, 3)
                top_words = [vocab.idx2word.get(idx.item(), "<UNK>") for idx in top_idx]

                embed_grad_norm = model.embedding.weight.grad.norm().item() if model.embedding.weight.grad is not None else 0.0
                linear_grad_norm = model.fc_out.weight.grad.norm().item() if model.fc_out.weight.grad is not None else 0.0

                print("\n" + "📊 " * 10)
                print(f"🧮 MATHE-CHECK SCHRITT {step}:")
                print(f"   • Raw Logits (Min/Max): {last_token_logits.min().item():.2f} / {last_token_logits.max().item():.2f}")
                print(f"   • Softmax-Wahrscheinlichkeiten (Top 3 Wörter):")
                for w, p in zip(top_words, top_prob):
                    print(f"     -> '{w}': {p.item() * 100:.2f}%")
                print(f"   • Berechneter Cross-Entropy Loss: {loss.item():.4f}")
                print(f"   • ⚡ Backprop Gradient Norm (Embedding Layer): {embed_grad_norm:.6f}")
                print(f"   • ⚡ Backprop Gradient Norm (Output Linear Layer): {linear_grad_norm:.6f}")
                print("📊 " * 10 + "\n")

            # 🧪 VALIDIERUNG & EVALUIERUNG INTERVALL
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

                current_lr = scheduler.get_last_lr()[0]

                # 🛑 EARLY STOPPING PRÜFUNG & MODELL-SPEICHERUNG
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    patience_counter = 0
                    model.save_model(str(MODEL_SAVE_PATH))
                    saved_msg = " [Bestes Modell gespeichert]"
                else:
                    patience_counter += 1
                    saved_msg = f" [Keine Verbesserung ({patience_counter}/{patience})]"

                if status_container:
                    status_container.write(
                        f"📍 **Schritt {step}/{t_prefs['steps']}** — "
                        f"Train Loss: `{loss_val:.4f}` | "
                        f"Test Loss: `{val_loss_avg:.4f}` | "
                        f"Accuracy: `{accuracy * 100:.1f}%` | "
                        f"Perplexity: `{perplexity:.2f}` | "
                        f"LR: `{current_lr:.6f}`"
                        f"{saved_msg}"
                    )

                if patience_counter >= patience:
                    print(f"\n🛑 EARLY STOPPING: Test Loss hat sich {patience} Überprüfungen lang nicht verbessert. Abbruch!")
                    if status_container:
                        status_container.warning(f"🛑 **Early Stopping ausgelöst!** Abbruch bei Schritt {step}/{t_prefs['steps']}.")
                    stop_early = True
                    break

                model.train()

            if step >= t_prefs["steps"]:
                break

    return loss_history, val_loss_history, ppl_history, acc_history


def get_training_recommendation(m_prefs: dict, t_prefs: dict, results: dict) -> str:
    """Nutzt das LLM, um die Trainingsergebnisse zu analysieren und Empfehlungen zu geben."""
    sys_prompt = (
        "Du bist ein Senior Machine Learning Engineer für Transformer-Modelle. "
        "Analysiere die Hyperparameter und die Ergebnisse des PyTorch-Trainings. "
        "Gib kurze, präzise Empfehlungen (max. 3-4 Bulletpoints), wie das Modell "
        "weiter verbessert werden kann (z.B. Overfitting vermeiden, Kapazität erhöhen, Lernrate anpassen)."
        "Wichtig: Fasse dich kurz und beende jeden Satz vollständig!"
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
        return generate(system_prompt=sys_prompt, user_prompt=user_prompt, max_tokens=300)
    except Exception as e:
        return f"⚠️ KI-Empfehlung konnte nicht generiert werden: {str(e)}"


def sample_text(
        prompt: str,
        m_prefs: dict,
        max_tokens: int = 20,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9
) -> str:
    """Generiert Text mit Top-k und Top-p (Nucleus) Sampling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_SAVE_PATH.exists() or not VOCAB_SAVE_PATH.exists():
        return "⚠️ Noch kein trainiertes Modell gefunden. Bitte zuerst das Training ausführen!"

    vocab = Vocabulary()
    vocab.load(str(VOCAB_SAVE_PATH))

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

    # 🖨️ TERMINAL HEADER
    print("\n" + "🎲 " * 15)
    print(f"🎯 INFERENZ / SAMPLING START (Prompt: '{prompt}')")
    print("🎲 " * 15)

    with torch.no_grad():
        for step_idx in range(max_tokens):
            input_tensor = torch.tensor([generated[-64:]], dtype=torch.long).to(device)
            logits = model(input_tensor)

            # 1. Temperature anwenden
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)

            # 2. TOP-K SAMPLING
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # 3. TOP-P (NUCLEUS) SAMPLING
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # 4. Softmax für finale Wahrscheinlichkeiten
            probs = F.softmax(next_token_logits, dim=-1)

            # 📊 TOP 5 KANDIDATEN UND WAHRSCHEINLICHKEITEN EXTRAHIEREN
            top5_probs, top5_indices = torch.topk(probs, k=min(5, len(vocab)))

            # 5. Nächstes Wort ziehen
            next_token = torch.multinomial(probs, num_samples=1).item()
            picked_word = vocab.idx2word.get(next_token, "<UNK>")

            # 🖨️ SCHRITT-FÜR-SCHRITT AUSGABE IM TERMINAL
            current_context = " ".join([vocab.idx2word.get(idx, "<UNK>") for idx in generated])
            print(f"\n🔹 Schritt {step_idx + 1} | Context: \"{current_context}\"")
            print("   Mögliche Top-5 Wörter (Softmax %):")
            for rank, (p, idx) in enumerate(zip(top5_probs, top5_indices), 1):
                word = vocab.idx2word.get(idx.item(), "<UNK>")
                prob_pct = p.item() * 100
                is_picked = "👈 [GEPICK T]" if idx.item() == next_token else ""
                print(f"     {rank}. '{word}': {prob_pct:.2f}% {is_picked}")

            # Fallback-Anzeige, falls das gezogene Wort außerhalb der Top 5 lag
            if next_token not in [idx.item() for idx in top5_indices]:
                print(f"   ➔ Gepickt: '{picked_word}' (ID: {next_token})")

            generated.append(next_token)

    words = [vocab.idx2word.get(idx, "<UNK>") for idx in generated]
    final_text = " ".join(words)

    # 🖨️ TERMINAL FOOTER
    print("\n" + "✅ " * 15)
    print(f"🏁 FINALE GENERIERUNG: \"{final_text}\"")
    print("✅ " * 15 + "\n")

    return final_text