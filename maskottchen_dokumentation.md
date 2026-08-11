# Maskottchen — Ausführliche Dokumentation

Modelle, Trainingsprozess und alle Stellschrauben im Detail. Stand: nach
Mood-State-Machine, Variante C (eigener Transformer) und Feedback-Sammlung.

---

## Inhalt

1. [Architekturüberblick](#1-architekturüberblick)
2. [Die drei Trigger-Modell-Varianten](#2-die-drei-trigger-modell-varianten)
3. [Trainingsprozess im Detail](#3-trainingsprozess-im-detail)
4. [Mood-State-Machine](#4-mood-state-machine)
5. [Feedback-Sammlung (echte Daten ins Training)](#5-feedback-sammlung-echte-daten-ins-training)
6. [Stellschrauben — alle einstellbaren Parameter](#6-stellschrauben)
7. [Glossar](#7-glossar)

---

## 1. Architekturüberblick

```
simulator.py --Turn--> main.py/app.py --Turn+Context--> features.py --Vektor--> trigger*.py
                                                                                      |
                                                                              reagieren? (Sigmoid)
                                                                                      |
                                                                                     Ja
                                                                                      v
                                                                                mascot.py (+ mood.py)
                                                                                      |
                                                                                llm_client.py --> Antwort
```

Zusätzlich, unabhängig vom Live-Betrieb:

```
train_trigger.py / train_trigger_transformer.py
   -> synthetische Daten (build_dataset)
   -> + echte, bestätigte Daten (data_collection.py, optional)
   -> Training
   -> Evaluation gegen manual_validation.json (unabhängig!)
   -> Threshold-Sweep
```

---

## 2. Die drei Trigger-Modell-Varianten

Alle drei beantworten dieselbe Frage — *"soll das Maskottchen bei diesem
Turn reagieren?"* — aber auf unterschiedliche Weise, wie der Text in
Zahlen übersetzt wird, die ein Netz verarbeiten kann.

### Variante A: reines Lexikon (Ausgangspunkt, nicht mehr aktiv)

8 handgebaute Features (`features.py`, `extract_handcrafted_features`):

| # | Feature | Was es misst |
|---|---|---|
| 0 | `length_norm` | Textlänge, normiert (gedeckelt bei 200 Zeichen) |
| 1 | `has_question` | Enthält `?` |
| 2 | `has_exclamation` | Enthält `!` |
| 3 | `mentions_mascot` | Wird "Pixel" per Regex mit Wortgrenzen gefunden |
| 4 | `positive_score` | Treffer aus `POSITIVE_WORDS` (Lexikon) |
| 5 | `negative_score` | Treffer aus `NEGATIVE_WORDS` (Lexikon) |
| 6 | `turns_since_last_reaction_norm` | Wie lange war Pixel schon still |
| 7 | `active_speaker_count_norm` | Wie viele Personen reden gerade aktiv mit |

**Schwäche:** blind für alles, was nicht wörtlich im Lexikon steht
(Synonyme, Verneinung: "nicht schlecht", indirekte Ansprache: "unser
kleiner Kumpel").

### Variante B: vortrainiertes Satz-Embedding (aktiv, Standard in main.py/app.py)

`embeddings.py` + `features.py` (`extract_features`).

Die 8 handgebauten Features werden mit einem **384-dimensionalen
Satz-Embedding** aus `sentence-transformers/all-MiniLM-L6-v2`
konkateniert → 392 Dimensionen Input fürs Netz (`trigger.py`,
`TriggerNet`).

**Wie das Embedding funktioniert:** Ein auf Millionen Sätzen
vortrainiertes Modell übersetzt einen Satz in einen Punkt im
"Bedeutungsraum" — Sätze mit ähnlicher Bedeutung liegen nah beieinander,
unabhängig von der exakten Wortwahl. Das Modell selbst wird **nicht**
mittrainiert (eingefroren) — nur der Trigger-Kopf lernt, wie er dieses
Embedding nutzt.

**`TriggerNet`-Architektur:**
```
Input (392) -> Linear(64) -> GELU -> Dropout(0.2) -> Linear(32) -> GELU -> Linear(1) -> Sigmoid
```

**Braucht:** Internetzugriff beim ersten Lauf (Modell-Download, ~90 MB),
danach lokal gecacht.

### Variante C: eigener Tokenizer + Token-Embedding + Transformer (Vergleichsmodell)

`custom_transformer.py` + `trigger_transformer.py` + `train_trigger_transformer.py`.

Hier wird **alles selbst gebaut und mittrainiert** — kein vortrainiertes
Modell:

1. **`SimpleTokenizer`** — eigenes Vokabular, nur aus den 34
   Trainingssätzen aufgebaut (wortbasiert, kein Sub-Word/BPE)
2. **Token-Embedding** — `nn.Embedding(vocab_size, 32)`, lernt für jedes
   bekannte Wort einen 32-dim-Vektor
3. **Positions-Embedding** — da Transformer von sich aus keine
   Reihenfolge kennen, wird Positionsinfo addiert
4. **`nn.TransformerEncoder`** (2 Schichten, 2 Attention-Heads) —
   verarbeitet die Token-Sequenz über Self-Attention
5. **Mean-Pooling** — fasst die Token-Vektoren (ohne Padding) zu einem
   Satzvektor zusammen
6. Dieser 32-dim-Satzvektor wird mit den 8 handgebauten Features
   kombiniert (40 Dimensionen total) und geht in denselben Kopf-Aufbau
   wie bei Variante B

**Entscheidender Unterschied zu B:** Alles — Token-Embedding,
Transformer-Gewichte, Klassifikations-Kopf — wird **gemeinsam** über
denselben `BCELoss` trainiert. Nichts ist eingefroren.

**Braucht:** kein Internet, keine externe Bibliothek außer PyTorch
selbst.

**Bekannte Schwäche:** Out-of-Vocabulary — jedes Wort, das nicht in den
34 Trainingssätzen vorkam, wird zu `<unk>`, das Modell hat dann kein
Signal aus dem Transformer-Teil dafür (nur noch die handgebauten
Features helfen).

### Vergleichstabelle (eure tatsächlichen Ergebnisse)

| | A (Lexikon, geschätzt schwächer) | B (vortrainiertes Embedding) | C (eigener Transformer) |
|---|---|---|---|
| Precision | — | 100 % / 80 % (je nach Lauf) | 100 % |
| Recall | — | 61–77 % | 61,5 % |
| F1 | — | 0,70–0,87 | 0,76 |
| Braucht Internet | Nein | Ja (einmalig) | Nein |
| Wörter außerhalb des Trainings verstehen | Nein | Ja | Nein (OOV-Problem) |

---

## 3. Trainingsprozess im Detail

### 3.1 Woher kommen die Trainingsdaten?

`build_dataset()` in `train_trigger.py` (bzw. das Pendant in
`train_trigger_transformer.py`) generiert **synthetisch**:

1. Zufälliger Satz aus `SAMPLE_SENTENCES` (34 Vorlagesätze, kategorisiert:
   neutral, positiv, negativ, starke Emotion ohne Lexikon-Treffer,
   direkte Ansprache, Gruppenfragen, kurze Aussagen)
2. Zufällige Person als Sprecher
3. Label per `_rule_based_label()` — einer **selbst geschriebenen
   Regel**, die eure eigene Vorstellung davon kodiert, wann eine
   Reaktion sinnvoll ist (Ansprache → fast immer ja, starke Emotion →
   meistens, lange Stille → manchmal, sonst eher nein, plus etwas
   Zufallsrauschen)

Das ist **Weak Supervision**: Statt hunderte Beispiele von Hand zu
labeln, wird die Regel automatisch auf beliebig viele generierte
Beispiele angewendet (Standard: 800).

### 3.2 Der Trainings-Loop

```python
model = TriggerNet()
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-2)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train)
    loss = criterion(preds, y_train)
    loss.backward()
    optimizer.step()
```

- **`BCELoss`** (Binary Cross-Entropy): das Standard-Fehlermaß für
  binäre Klassifikation mit Sigmoid-Output. Bestraft selbstbewusste
  Fehlentscheidungen (z. B. p=0.95 bei Label 0) stärker als unsichere
  (p=0.55 bei Label 0).
- **`AdamW`**: passt die Gewichte basierend auf den Gradienten an —
  mit Momentum (nutzt die Richtung vorheriger Updates) und Weight
  Decay (hält die Gewichte klein, wirkt Overfitting entgegen).
- **`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`**:
  der klassische PyTorch-Dreischritt — Gradienten zurücksetzen,
  Gradienten berechnen (Backpropagation), Gewichte aktualisieren.

### 3.3 Train/Val-Split

85 % Training, 15 % Validierung (`n_val = int(len(X) * 0.15)`). Der
Val-Split zeigt, ob das Netz *generalisiert* (auch auf ungesehenen
synthetischen Daten funktioniert) statt nur auswendig zu lernen.

### 3.4 Die unabhängige Prüfung: `manual_validation.json`

24 von Hand konzipierte Beispiele, **unabhängig** von der
Trainingsregel — mit bewussten Grenzfällen (Verneinung, indirekte
Ansprache, Sarkasmus, starke Emotion ohne Lexikon-Wort). Beantwortet die
Frage: *"Lernt das Netz nur meine eigene Formel auswendig, oder
überträgt sich das auf echtes Urteilsvermögen?"*

`evaluate_manual_set()` druckt pro Beispiel richtig/falsch plus
Gesamt-Metriken (Accuracy, Precision, Recall, F1, Konfusionsmatrix).

### 3.5 Threshold-Tuning

`should_react()` entscheidet über einen Schwellwert: `p >= threshold`
→ reagieren. `tune_threshold()` testet mehrere Schwellwerte (0.30 bis
0.70) gegen das manuelle Set und wählt den mit dem besten F1-Score.

**Bei euch gefunden:** ein Plateau zwischen 0.30 und 0.40 (identisches
F1=0.87) — als Default wurde die Mitte (0.40) gewählt statt des
Randwerts (0.30), für mehr Robustheit gegenüber leicht veränderten
Trainingsdaten.

---

## 4. Mood-State-Machine

`mood.py` — zwei unabhängige Zahlen, die Pixels "Stimmung" beschreiben:

- **`valence`** (-1 bis +1): wie positiv/negativ sich das Gespräch bisher
  angefühlt hat
- **`energy`** (0 bis 1): wie aufgedreht/müde

**Update-Regel pro Turn** (nicht nur bei Reaktionen — Pixel "hört mit"):

```python
self.valence *= DECAY          # erst Richtung neutral abklingen (0.85)
self.energy *= ENERGY_DECAY    # (0.90)
self.valence += POS_STEP * positive_score - NEG_STEP * negative_score
self.energy += ENERGY_STEP * intensity
```

Erst Decay, dann der neue Turn — ein einzelner Ausreißer kippt so nicht
die ganze Stimmung, aber ein anhaltendes Muster (mehrere frustrierte
Turns) schon. `label()` bildet daraus einen Text+Emoji (z. B. "genervt
😤"), der sowohl angezeigt als auch in den LLM-Prompt für die
Maskottchen-Antwort eingebaut wird (`mascot.py`).

---

## 5. Feedback-Sammlung (echte Daten ins Training)

`data_collection.py` — sammelt **von Menschen bestätigte/korrigierte**
Trigger-Entscheidungen aus echten Sessions:

- **`app.py`**: nach jeder Entscheidung erscheinen Buttons "✅ Richtig"
  / "❌ Falsch" — Klick schreibt einen Datensatz in
  `data/collected_turns.jsonl`
- **`main.py --collect`**: dasselbe über Konsoleneingabe

**Wichtige Designentscheidung:** Das gespeicherte `label` ist die
**menschliche** Entscheidung, nicht automatisch die Modellvorhersage
(`predicted`). Würde man ungeprüft die eigene Vorhersage als Label
speichern, würde sich das Modell beim Nachtrainieren nur auf seinen
eigenen — möglicherweise falschen — Entscheidungen selbst bestätigen
(Feedback-Loop, verstärkt bestehende Fehler statt sie zu korrigieren).

**Einbindung ins Training:** `train_trigger.py` lädt diese Datei
(falls vorhanden) und mischt sie **nur dem Trainings-Split** bei — der
Validierungs-Split (gegen `manual_validation.json`) bleibt unverändert,
damit die Auswertung über die Zeit vergleichbar bleibt.

---

## 6. Stellschrauben

Alle Parameter, die sich lohnen anzupassen, mit Datei, Effekt und
Faustregel.

### Trigger-Entscheidung

| Parameter | Datei | Aktuell | Effekt beim Erhöhen | Effekt beim Senken |
|---|---|---|---|---|
| `threshold` | `trigger.py`, `should_react()` | 0.40 | weniger, aber sichere Reaktionen (höhere Precision, niedrigerer Recall) | mehr Reaktionen (höherer Recall, mehr Fehlalarme) |
| `window` | `context.py`, `ConversationContext(window=...)` | 15 | Maskottchen "erinnert" sich an mehr Gesprächsverlauf | kürzeres Gedächtnis, schneller "vergessen" |

### TriggerNet-Architektur (Variante B)

| Parameter | Datei | Aktuell | Effekt |
|---|---|---|---|
| `hidden` | `trigger.py`, `TriggerNet.__init__` | 64 | mehr Kapazität, aber höheres Overfitting-Risiko bei wenig Daten |
| `Dropout`-Rate | `trigger.py` | 0.2 | höher = mehr Regularisierung (hilft gegen Overfitting, kann aber Underfitting verursachen) |

### Training (Variante B, `train_trigger.py`)

| Parameter | Aktuell | Effekt |
|---|---|---|
| `epochs` | 200 | mehr = potenziell besseres Fitting, aber auch mehr Overfitting-Risiko ohne Early Stopping |
| `lr` (Lernrate) | 1e-2 | höher = schnelleres, aber instabileres Training; niedriger = langsamer, aber stabiler |
| `n_samples` in `build_dataset()` | 800 | mehr synthetische Daten = stabileres Training, aber immer noch begrenzt durch die 34 Vorlagesätze |
| `SAMPLE_SENTENCES`-Liste | 34 Sätze | mehr/vielfältigere Sätze = mehr Angriffsfläche fürs Embedding-Signal |
| `_rule_based_label()` | — | die eigentliche "Wahrheit", die gelernt werden soll — hier ansetzen, um das gewünschte Reaktionsverhalten grundsätzlich zu ändern |

### Variante C (eigener Transformer)

| Parameter | Datei | Aktuell | Effekt |
|---|---|---|---|
| `embed_dim` (Token-Embedding-Größe) | `custom_transformer.py`, `TextEncoder` | 32 | größer = mehr Ausdruckskraft, aber mehr Overfitting-Risiko bei wenig Daten/Vokabular |
| `n_heads` | `custom_transformer.py` | 2 | mehr Attention-Heads = mehr parallele "Blickwinkel" auf den Satz, braucht aber mehr Daten um sinnvoll genutzt zu werden |
| `n_layers` | `custom_transformer.py` | 2 | mehr Schichten = tiefere Verarbeitung, aber deutlich mehr Parameter bei wenig Trainingsdaten riskant |
| `MAX_LEN` | `custom_transformer.py` | 20 Tokens | längere Sätze werden abgeschnitten, falls überschritten |
| `lr` | `train_trigger_transformer.py` | 1e-3 (niedriger als bei B, absichtlich) | Transformer + Embedding gemeinsam trainieren braucht eine vorsichtigere Lernrate als nur der kleine Kopf bei Variante B |

### Mood-State-Machine

| Parameter | Datei | Aktuell | Effekt |
|---|---|---|---|
| `DECAY` | `mood.py` | 0.85 | höher = Stimmung hält länger an (weniger schnell neutral) |
| `ENERGY_DECAY` | `mood.py` | 0.90 | höher = Energie klingt langsamer ab |
| `POS_STEP` / `NEG_STEP` | `mood.py` | 0.35 | höher = einzelne Turns schlagen stärker auf die Stimmung durch |
| `ENERGY_STEP` | `mood.py` | 0.40 | höher = emotionale Turns pushen die Energie stärker |
| Bucket-Schwellwerte in `label()` | `mood.py` | 0.15 / 0.5 | verschiebt, ab wann z. B. "müde" oder "aufgedreht" angezeigt wird |

### Feature-Extraktion

| Parameter | Datei | Aktuell | Effekt |
|---|---|---|---|
| `POSITIVE_WORDS` / `NEGATIVE_WORDS` | `features.py` | je ~5–9 Wörter | mehr Wörter = bessere Lexikon-Abdeckung, aber Pflegeaufwand |
| Textlängen-Deckel (`/200`) | `features.py` | 200 Zeichen | ab welcher Länge `length_norm` bei 1.0 sättigt |
| `EMBEDDING_MODEL_NAME` | `embeddings.py` | `all-MiniLM-L6-v2` | größere/andere Modelle = potenziell bessere Embeddings, aber langsamer/größerer Download |

### Feedback-Sammlung

| Parameter | Datei | Effekt |
|---|---|---|
| Ob `collected_turns.jsonl` genutzt wird | automatisch, sobald die Datei existiert | mehr gesammelte, bestätigte Beispiele = das Training lernt zunehmend von echtem statt nur synthetischem Verhalten |

---

## 7. Glossar

| Begriff | Kurzerklärung |
|---|---|
| **Token-Embedding** | Numerischer Vektor pro Wort/Token, den das Modell selbst lernt (Variante C) — im Gegensatz zum ganzen-Satz-Embedding aus Variante B, das fertig vortrainiert ist. |
| **Positional Encoding** | Zusatzinformation, die einem Transformer mitteilt, an welcher Position im Satz ein Token steht — ohne das hätte "Tom nervt Lena" dieselbe Repräsentation wie "Lena nervt Tom". |
| **Self-Attention** | Mechanismus, mit dem jedes Token "entscheidet", wie stark es jedes andere Token im selben Satz für seine eigene Repräsentation berücksichtigt. |
| **Sigmoid** | Aktivierungsfunktion, quetscht jeden Wert auf 0–1 — macht aus rohem Netzwerk-Output eine Wahrscheinlichkeit. |
| **BCE-Loss** | Fehlermaß für binäre Klassifikation; bestraft falsche, selbstbewusste Vorhersagen stärker als unsichere. |
| **AdamW** | Optimierungsverfahren mit Momentum + Weight Decay, robuster als reines SGD. |
| **Dropout** | Schaltet beim Training zufällig Neuronen ab — Schutz gegen Overfitting. |
| **Overfitting** | Modell lernt Trainingsdaten auswendig statt zugrunde liegende Muster — erkennbar an hoher Trainings-, aber niedriger Validierungs-Genauigkeit. |
| **Weak Supervision** | Trainingsdaten werden automatisch über eine Regel/Heuristik gelabelt statt von Hand — spart Zeit, braucht aber unabhängige Prüfung. |
| **Precision** | Von allem, was als "reagieren" vorhergesagt wurde: wie viel Anteil war korrekt. |
| **Recall** | Von allem, was tatsächlich eine Reaktion verdient hätte: wie viel Anteil wurde erkannt. |
| **F1-Score** | Harmonisches Mittel aus Precision und Recall. |
| **Out-of-Vocabulary (OOV)** | Ein Wort, das der Tokenizer nicht kennt (wird zu `<unk>`) — Schwachpunkt kleiner, selbst gebauter Vokabulare gegenüber Sub-Word-Tokenizern. |
| **Feedback-Loop (Bias-Verstärkung)** | Wenn ein Modell auf seinen eigenen, ungeprüften Vorhersagen weitertrainiert wird und dadurch bestehende Fehler verstärkt statt sie zu korrigieren. |
