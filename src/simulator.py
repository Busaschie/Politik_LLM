# src/simulator.py
import json
import random
from datetime import datetime
from pathlib import Path
from src.llm_client import generate

class PartyDebateSimulator:
    def __init__(self, personas_path="data/personas.json"):
        # Pfad dynamisch bestimmen
        self.base_dir = Path(__file__).resolve().parent.parent
        full_path = self.base_dir / personas_path

        with open(full_path, "r", encoding="utf-8") as f:
            self.personas = json.load(f)

        # Ordner-Pfade definieren
        self.discussions_dir = self.base_dir / "data" / "discussions"
        self.discussions_dir.mkdir(parents=True, exist_ok=True)

        self.programmes_dir = self.base_dir / "data" / "programmes"
        self.programmes_dir.mkdir(parents=True, exist_ok=True)

    def _load_party_programme(self, party_name: str) -> str:
        """Lädt das ausführliche Parteiprogramm aus der entsprechenden .txt-Datei."""
        # Dateinamen-Mapping für Sonderzeichen (z. B. Grüne -> Gruene.txt)
        filename_map = {
            "Grüne": "Gruene.txt",
            "CSU": "CSU.txt",
            "SPD": "SPD.txt",
            "Linke": "Linke.txt",
            "AfD": "AfD.txt"
        }

        file_name = filename_map.get(party_name, f"{party_name}.txt")
        file_path = self.programmes_dir / file_name

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            # Fallback, falls die Datei noch nicht existiert
            return self.personas.get(party_name, {}).get("programm", "")

    def run_round(self, topic: str) -> list[str]:
        discussion_history = [f"Thema der Debatte: {topic}"]
        structured_entries = []
        # Reihenfolge dynamisch vor jeder Runde mischen
        parties = ["CSU", "SPD", "Grüne", "Linke", "AfD"]
        random.shuffle(parties)

        print(f"=== STARTE DEBATTE ZUM THEMA: {topic} ===")
        print(f"Reihenfolge heute: {', '.join(parties)}\n")

        # 1. Moderator eröffnet die Runde
        mod_info = self.personas.get("Moderator", {})
        mod_sys = mod_info.get("system_prompt", "Du bist ein neutraler Moderator.")
        mod_user = f"Das Thema lautet: '{topic}'. Eröffne die Debatte kurz, energisch und fordere die Parteien auf."

        mod_reply = generate(system_prompt=mod_sys, user_prompt=mod_user)
        discussion_history.append(f"Moderator: {mod_reply}")
        structured_entries.append({"speaker": "Moderator", "text": mod_reply})

        # 2. Jede Partei antwortet nacheinander
        for party in parties:
            p_info = self.personas.get(party, {})

            # 🔥 Ausführliches Parteiprogramm aus data/programmes/ laden:
            detailed_programme = self._load_party_programme(party)

            p_sys = (
                f"{p_info.get('system_prompt', '')}\n\n"
                f"Ausführliches Parteiprogramm & Kernpositionen:\n{detailed_programme}"
            )

            p_user = (
                    f"Bisheriger Verlauf der Debatte:\n" + "\n".join(discussion_history[-3:]) +
                    f"\n\nNimm als Vertreter der {party} kurz und prägnant Stellung im Sinne deines Parteiprogramms."
            )

            p_reply = generate(system_prompt=p_sys, user_prompt=p_user)
            discussion_history.append(f"{party}: {p_reply}")
            structured_entries.append({"speaker": party, "text": p_reply})

        # 3. Diskussion als JSON speichern
        self._save_as_json(topic, discussion_history, structured_entries)

        return discussion_history

    def _save_as_json(self, topic: str, history: list[str], structured_entries: list[dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"discussion_{timestamp}.json"
        filepath = self.discussions_dir / filename

        data = {
            "topic": topic,
            "timestamp": timestamp,
            "full_text": "\n".join(history),
            "rounds": structured_entries
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 Diskussion erfolgreich gespeichert unter: {filepath}")