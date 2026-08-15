# app.py
import streamlit as st
import pandas as pd
from src.simulator import PartyDebateSimulator
from src.train_trigger_transformer import (
    retrain_custom_transformer,
    sample_text,
    get_training_recommendation
)
#from src.train_hf_transformer_qwen import (
from src.train_hf_transformer_gpt import (
    retrain_hf_transformer,
    sample_hf_text
)

st.set_page_config(page_title="Polit-LLM Debatten Simulator", layout="wide", page_icon="🏛️")

st.title("🏛️ Polit-LLM: Parteiendiskussion & PyTorch Retraining")

# =========================================================
# SEITENLEISTE: MODELL-AUSWAHL & HYPERPARAMETER
# =========================================================
st.sidebar.header("⚙️ Modell & Einstellungen")

# 🔥 1. MODELL-AUSWAHL
selected_model_type = st.sidebar.radio(
    "Modell-Architektur wählen:",
    ["Eigener Custom Transformer (from Scratch)", "Hugging Face German-GPT2 (Fine-Tuning)"]
)

st.sidebar.divider()

# 🔥 2. DYNAMISCHE HYPERPARAMETER
if selected_model_type == "Eigener Custom Transformer (from Scratch)":
    st.sidebar.subheader("🏗️ Custom Modell-Architektur")
    pipelen = st.sidebar.slider("Anzahl Layer (pipelen/num_layers)", min_value=1, max_value=16, value=8)
    embed_dim = st.sidebar.select_slider("Embedding Dim (embed_dim)", options=[32, 64, 128, 256, 512], value=128)
    num_heads = st.sidebar.selectbox("Attention Heads (num_heads)", options=[1, 2, 4, 8], index=2)
    dropout = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)

    st.sidebar.subheader("🎯 Trainings-Parameter")
    steps = st.sidebar.slider("Trainings-Schritte (steps)", min_value=100, max_value=3000, value=800, step=100)
    interval = st.sidebar.select_slider("Evaluierungs-Intervall", options=[10, 50, 100, 200], value=100)
    lr = st.sidebar.select_slider("Lernrate (lr)", options=[1e-4, 3e-4, 5e-4, 1e-3], value=3e-4,
                                  format_func=lambda x: f"{x:.4f}")
    decay = st.sidebar.slider("Weight Decay", min_value=0.0, max_value=0.1, value=0.01, step=0.005)

    m_prefs = {
        "pipelen": pipelen,
        "vocab_size": 10000,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "dropout": dropout
    }

    t_prefs = {
        "seed": 0,
        "steps": steps,
        "interval": interval,
        "lr": lr,
        "decay": decay
    }
else:
    st.sidebar.subheader("🎯 Hugging Face Fine-Tuning Parameter")
    hf_epochs = st.sidebar.slider("Anzahl Epochen", min_value=1, max_value=10, value=3)
    hf_lr = st.sidebar.select_slider("Lernrate (lr)", options=[1e-5, 3e-5, 5e-5, 1e-4], value=5e-5,
                                     format_func=lambda x: f"{x:.5f}")


# Layout in zwei Hauptspalten
col_left, col_right = st.columns([1, 1])

# =========================================================
# SPALTE 1: DISKUSSIONS-ANSICHT
# =========================================================
with col_left:
    st.header("💬 Live-Diskussion")
    topic = st.text_input("Debattenthema eingeben:", "Zukunft der Energieversorgung und Wirtschaft")

    if st.button("🚀 Diskussion starten", use_container_width=True):
        simulator = PartyDebateSimulator()
        with st.spinner("Parteien und Moderator debattieren..."):
            history = simulator.run_round(topic)
            st.session_state["last_discussion"] = history

    if "last_discussion" in st.session_state:
        st.subheader("📋 Protokoll der Debatte")
        for line in st.session_state["last_discussion"]:
            # line.startswith("Moderator") erfasst sowohl "Moderator:" als auch "Moderator (Fazit):"
            if line.startswith("Moderator"):
                st.info(line)
            elif line.startswith("CSU:"):
                st.markdown(f"🔵 **{line}**")
            elif line.startswith("SPD:"):
                st.markdown(f"🔴 **{line}**")
            elif line.startswith("Grüne:"):
                st.markdown(f"🟢 **{line}**")
            elif line.startswith("Linke:"):
                st.markdown(f"🟣 **{line}**")
            elif line.startswith("AfD:"):
                st.markdown(f"🔷 **{line}**")

# =========================================================
# SPALTE 2: TRAINING & METRIKEN + KI-EMPFEHLUNG
# =========================================================
with col_right:
    st.header("🔥 Model Retraining")
    st.caption(f"Aktuell ausgewähltes Modell: **{selected_model_type}**")

    if st.button("🧠 Model jetzt feintunen", use_container_width=True):
        status_box = st.container(height=250)

        # -------------------------------------------------------------
        # FALL A: CUSTOM TRANSFORMER TRAINING
        # -------------------------------------------------------------
        if selected_model_type == "Eigener Custom Transformer (from Scratch)":
            with st.spinner("PyTorch Training läuft..."):
                res = retrain_custom_transformer(m_prefs, t_prefs, status_container=status_box)

            if isinstance(res, tuple) and len(res) == 4:
                t_hist, v_hist, p_hist, a_hist = res

                if t_hist and len(t_hist) > 0:
                    st.success("🎉 Custom Transformer erfolgreich trainiert!")

                    last_results = {
                        "train_loss": t_hist[-1],
                        "test_loss": v_hist[-1],
                        "accuracy": a_hist[-1] if a_hist else 0.0,
                        "perplexity": p_hist[-1]
                    }

                    with st.spinner("🤖 KI analysiert Ergebnisse und generiert Empfehlungen..."):
                        recommendation = get_training_recommendation(m_prefs, t_prefs, last_results)
                        st.session_state["training_recommendation"] = recommendation

                        st.session_state["loss_chart_data"] = pd.DataFrame({
                            "Train Loss": t_hist,
                            "Test Loss": v_hist
                        })

                        st.session_state["ppl_chart_data"] = pd.DataFrame({
                            "Perplexity": p_hist
                        })
                else:
                    st.error("⚠️ Keine Trainingsschritte ausgeführt. Bitte starte zuerst links eine Diskussion!")
            else:
                st.error("⚠️ Die Trainingsfunktion hat ein unerwartetes Format zurückgegeben.")

        # -------------------------------------------------------------
        # FALL B: HUGGING FACE GERMAN-GPT2 FINE-TUNING
        # -------------------------------------------------------------
        else:
            with st.spinner("Hugging Face Fine-Tuning läuft..."):
                success = retrain_hf_transformer(epochs=hf_epochs, lr=hf_lr, status_container=status_box)

            if success:
                st.success("🎉 Hugging Face Modell erfolgreich ge-finetunt!")
                # Bei HF entfernen wir alte Verlaufs-Charts, falls vorhanden
                st.session_state.pop("loss_chart_data", None)
                st.session_state.pop("ppl_chart_data", None)
                st.session_state.pop("training_recommendation", None)

    # Zeige Charts und KI-Empfehlung an (nur bei Custom Transformer vorhanden)
    if "training_recommendation" in st.session_state and selected_model_type == "Eigener Custom Transformer (from Scratch)":
        st.subheader("📊 Trainings-Ergebnisse")

        st.markdown("#### 📉 Loss-Verlauf (Train vs. Test)")
        st.line_chart(st.session_state["loss_chart_data"])

        st.markdown("#### 🌀 Perplexity (Unsicherheit des Modells)")
        st.line_chart(st.session_state["ppl_chart_data"])

        st.info("💡 **KI-Empfehlung zur Optimierung**")
        st.markdown(st.session_state["training_recommendation"])

    st.divider()

    # =========================================================
    # INFERENZ / MODELL TESTEN
    # =========================================================
    st.subheader("🤖 Trainiertes Modell testen (Inferenz)")
    prompt_input = st.text_input("Prompt für das ausgewählte Modell:", "Die CSU fordert")

    if st.button("🔮 Text weitergenerieren", use_container_width=True):
        with st.spinner("Modell generiert Text..."):
            if selected_model_type == "Eigener Custom Transformer (from Scratch)":
                gen_text = sample_text(prompt=prompt_input, m_prefs=m_prefs, max_tokens=80)
            else:
                gen_text = sample_hf_text(prompt=prompt_input, max_tokens=150)

            st.write(f"**Generierte Antwort ({selected_model_type}):**")
            st.info(gen_text)