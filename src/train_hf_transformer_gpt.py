# src/train_hf_transformer.py
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

BASE_DIR = Path(__file__).resolve().parent.parent
DISCUSSIONS_DIR = BASE_DIR / "data" / "discussions"
HF_MODEL_SAVE_PATH = BASE_DIR / "data" / "hf_fine_tuned_model_gpt"
BASE_MODEL_NAME = "dbmdz/german-gpt2"

class StreamlitStatusCallback(TrainerCallback):
    """Liest die Trainings-Logs von Hugging Face aus und schreibt sie live in die Streamlit Status-Box."""

    def __init__(self, status_container):
        self.status_container = status_container

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.status_container:
            epoch = logs.get("epoch", 0)
            loss = logs.get("loss", None)
            lr = logs.get("learning_rate", None)
            grad_norm = logs.get("grad_norm", None)

            # Zeige Log-Daten an, sobald ein Loss vorhanden ist
            if loss is not None:
                msg = f"📍 **Epoche {epoch:.2f}** — Loss: `{loss:.4f}`"
                if lr is not None:
                    msg += f" | LR: `{lr:.6f}`"
                if grad_norm is not None:
                    msg += f" | Grad Norm: `{grad_norm:.2f}`"

                self.status_container.write(msg)


def load_all_discussions() -> list[str]:
    """Lädt alle .json-Dateien aus data/discussions/."""
    texts = []
    if not DISCUSSIONS_DIR.exists():
        return texts

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

    return texts


def retrain_hf_transformer(epochs: int = 3, lr: float = 5e-5, status_container=None):
    """
    Fine-tunt das german-gpt2 Modell mit allen vorhandenen Diskussions-JSONs.
    """
    discussions = load_all_discussions()
    if not discussions:
        if status_container:
            status_container.error("Keine Diskussionsdaten im Ordner 'data/discussions/' gefunden!")
        return False

    if status_container:
        status_container.info("📥 Lade Modell & Tokenizer von Hugging Face...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

    # Dataset erstellen & tokenisieren
    dataset = Dataset.from_dict({"text": discussions})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(HF_MODEL_SAVE_PATH / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        logging_steps=5,
        learning_rate=lr,
        weight_decay=0.01,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
    )

    callbacks = []
    if status_container:
        callbacks.append(StreamlitStatusCallback(status_container))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if status_container:
        status_container.info("🚀 Starte Fine-Tuning mit euren JSON-Diskussionen...")

    trainer.train()

    # Fertiges Modell & Tokenizer speichern
    HF_MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(HF_MODEL_SAVE_PATH))
    tokenizer.save_pretrained(str(HF_MODEL_SAVE_PATH))

    if status_container:
        status_container.success("✅ Hugging Face Modell erfolgreich ge-finetunt und gespeichert!")

    return True


def sample_hf_text(prompt: str, max_tokens: int = 40, temperature: float = 0.8) -> str:
    """Generiert Text mit dem ge-finetunten Hugging Face Modell."""
    if not HF_MODEL_SAVE_PATH.exists():
        return "⚠️ Noch kein ge-finetunktes Hugging Face Modell gefunden. Bitte zuerst das Training ausführen!"

    tokenizer = AutoTokenizer.from_pretrained(str(HF_MODEL_SAVE_PATH))
    model = AutoModelForCausalLM.from_pretrained(str(HF_MODEL_SAVE_PATH))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)