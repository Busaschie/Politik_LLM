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

    def run_round(self, topic: str, max_rounds: int = 3) -> list[str]:
        discussion_history = [f"Thema der Debatte: {topic}"]
        structured_entries = []
        parties = ["CSU", "SPD", "Grüne", "Linke", "AfD"]

        print(f"=== STARTE DEBATTE ZUM THEMA: {topic} ({max_rounds} Runden) ===")

        # 1. Eröffnung durch den Moderator
        mod_info = self.personas.get("Moderator", {})
        mod_sys = mod_info.get("system_prompt", "Du bist ein neutraler Moderator.")
        mod_user = f"Das Thema lautet: '{topic}'. Eröffne die Debatte kurz, energisch und fordere die Parteien auf."

        mod_reply = generate(system_prompt=mod_sys, user_prompt=mod_user)
        discussion_history.append(f"Moderator: {mod_reply}")
        structured_entries.append({"speaker": "Moderator", "text": mod_reply})

        # 2. Runden-Schleife (z.B. 3 Runden)
        for r in range(1, max_rounds + 1):
            random.shuffle(parties)

            for party in parties:
                p_info = self.personas.get(party, {})
                detailed_programme = self._load_party_programme(party)

                p_sys = (
                    f"{p_info.get('system_prompt', '')}\n\n"
                    f"Ausführliches Parteiprogramm & Kernpositionen:\n{detailed_programme}"
                )

                p_user = (
                        f"Runde {r} von {max_rounds}.\n"
                        f"Bisheriger Verlauf der Debatte:\n" + "\n".join(discussion_history[-4:]) +
                        f"\n\nNimm als Vertreter der {party} kurz in 2 Sätzen Stellung dazu."
                )

                p_reply = generate(system_prompt=p_sys, user_prompt=p_user)
                discussion_history.append(f"{party}: {p_reply}")
                structured_entries.append({"speaker": party, "text": p_reply, "round": r})

            # 3. Moderator greift am Ende von Runde 1 und 2 ein
            if r < max_rounds:
                mod_intervene_user = (
                        f"Runde {r} ist vorbei. Bisheriger Verlauf:\n" + "\n".join(discussion_history[-6:]) +
                        f"\n\nFasse als Moderator in 1-2 Sätzen kurz zusammen und stelle eine kritische Nachfrage für Runde {r + 1}."
                )
                mod_intervene_reply = generate(system_prompt=mod_sys, user_prompt=mod_intervene_user)
                discussion_history.append(f"Moderator: {mod_intervene_reply}")
                structured_entries.append({"speaker": "Moderator", "text": mod_intervene_reply, "round": r})

        # 4. Fazit des Moderators ganz am Ende
        mod_final_user = f"Die Debatte ist zu Ende. Bisheriger Verlauf:\n" + "\n".join(
            discussion_history[-6:]) + f"\n\nZiehe als Moderator ein kurzes, neutrales Fazit in 2 Sätzen."
        mod_final_reply = generate(system_prompt=mod_sys, user_prompt=mod_final_user)
        discussion_history.append(f"Moderator (Fazit): {mod_final_reply}")
        structured_entries.append({"speaker": "Moderator", "text": mod_final_reply, "round": "final"})

        # 5. Diskussion als JSON speichern
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