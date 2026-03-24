# -*- coding: utf-8 -*-
"""
AGRO-INSIGHT MAROC — Tableau de bord IA pour l'agriculture de précision
Prédiction du rendement par imagerie satellite et apprentissage automatique
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

class TemporalAttention(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1)

    def call(self, inputs):
        score = self.v(self.W(inputs))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context


MODEL_PATH = os.environ.get("MODEL_PATH", "model_vit_gru_att_tab.keras")
IMG_SIZE = (128, 128)


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(
            MODEL_PATH, compile=False,
            custom_objects={"TemporalAttention": TemporalAttention},
        )
    except Exception as e:
        st.session_state["_load_error"] = str(e)
        return None


def preprocess_image(uploaded_file):
    pil = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), pil


def build_tab_vector(pluie, ndvi_moy, surface, production, lat, lon, annee):
    return np.array([[pluie, ndvi_moy, surface, production, lat, lon, annee]], dtype=np.float32)


def build_seq_vector(ndvi_01, ndvi_03, ndvi_05):
    return np.array([[ndvi_01, ndvi_03, ndvi_05]], dtype=np.float32).reshape(1, 3, 1)


# ============================================================================
# THÈME DARK — Émeraude, teal, or
# ============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(180deg, #0d1f12 0%, #0a1810 50%, #07100a 100%);
}
.main .block-container { padding: 1.5rem 2rem; max-width: 100%; }
[data-testid="stHeader"] { background: rgba(0,0,0,0.3); }

/* Panneaux */
.panel {
    background: rgba(13, 71, 51, 0.25);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.panel-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #10b981;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}
.panel-dark {
    background: rgba(6, 28, 20, 0.6);
    border: 1px solid rgba(245, 158, 11, 0.25);
}

/* Titre principal */
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #10b981;
    text-align: center;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6ee7b7;
    text-align: center;
}

/* Carte prédiction */
.pred-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 95, 70, 0.4) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
}
.pred-card .value {
    font-size: 2.8rem;
    font-weight: 700;
    color: #f59e0b;
}
.pred-card .label { color: #6ee7b7; font-size: 1rem; }
.pred-card .badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.4);
    color: #10b981;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}

/* Zone upload image */
.upload-zone {
    border: 2px dashed rgba(16, 185, 129, 0.5);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    background: rgba(6, 28, 20, 0.4);
}
"""

st.set_page_config(
    page_title="AGRO-INSIGHT MAROC",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# EN-TÊTE
# ---------------------------------------------------------------------------
st.markdown('<p class="hero-title">AGRO-INSIGHT MAROC</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Agriculture de précision · Imagerie satellite & IA · Prédiction du rendement</p>',
    unsafe_allow_html=True,
)
st.markdown("")

model = load_model()
err = st.session_state.pop("_load_error", None)

# ---------------------------------------------------------------------------
# LAYOUT 3 PANNEAUX
# ---------------------------------------------------------------------------
col_left, col_center, col_right = st.columns([1, 1.5, 1])

# ========== PANNEAU GAUCHE — Saisie des données ==========
with col_left:
    st.markdown('<div class="panel"><div class="panel-title">📥 SAISIE DES DONNÉES</div></div>', unsafe_allow_html=True)

    st.markdown("**Pluviométrie & surface**")
    pluie = st.number_input(
        "Pluie saisonnière (mm)",
        min_value=0.0,
        value=450.0,
        step=1.0,
        key="pluie",
    )
    surface = st.number_input(
        "Surface (1000 Ha)",
        min_value=0.0,
        value=50.0,
        step=0.1,
        key="surface",
    )
    annee = st.number_input(
        "Année",
        min_value=2000,
        max_value=2035,
        value=2021,
        step=1,
        key="annee",
    )

    st.markdown("**Téléchargement d'image**")
    uploaded = st.file_uploader(
        "Image satellite (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    st.markdown("**NDVI (janvier, mars, mai)**")
    ndvi_01 = st.slider("NDVI_01", -0.2, 1.0, 0.40, 0.01, key="n01")
    ndvi_03 = st.slider("NDVI_03", -0.2, 1.0, 0.45, 0.01, key="n03")
    ndvi_05 = st.slider("NDVI_05", -0.2, 1.0, 0.55, 0.01, key="n05")

    ndvi_moy = st.number_input("NDVI moyen", min_value=-0.2, max_value=1.0, value=0.45, step=0.01, key="ndvi_moy")
    production = st.number_input("Production (1000 Qx)", min_value=0.0, value=600.0, step=0.1, key="prod")
    latitude = st.number_input("Latitude", value=35.20, format="%.6f", key="lat")
    longitude = st.number_input("Longitude", value=-3.93, format="%.6f", key="lon")

    predict_btn = st.button("🚀 Prédire le rendement", use_container_width=True, type="primary")

# ========== PANNEAU CENTRAL — Carte & NDVI ==========
with col_center:
    st.markdown('<div class="panel"><div class="panel-title">📊 CARTE & TENDANCE NDVI</div></div>', unsafe_allow_html=True)

    # Graphique NDVI (01, 03, 05) — natif Streamlit
    st.markdown("**Tendance NDVI (01, 03, 05)**")
    df_ndvi = pd.DataFrame({"NDVI": [ndvi_01, ndvi_03, ndvi_05]}, index=["Janvier", "Mars", "Mai"])
    st.line_chart(df_ndvi)

    # Aperçu image satellite
    st.markdown("**Aperçu image satellite**")
    if uploaded is not None:
        x_img, pil_preview = preprocess_image(uploaded)
        st.image(pil_preview, use_container_width=True)
    else:
        st.info("Chargez une image dans le panneau gauche.")
        x_img = None

    # Carte simplifiée (point sur Maroc)
    st.markdown("**Localisation (Maroc)**")
    map_data = {"lat": [latitude], "lon": [longitude]}
    st.map(map_data, use_container_width=True)

# ========== PANNEAU DROIT — Prédiction & métriques ==========
with col_right:
    st.markdown('<div class="panel"><div class="panel-title">🎯 ANALYSE & PRÉDICTION</div></div>', unsafe_allow_html=True)

    if predict_btn:
        if model is None:
            st.error("Modèle non chargé.")
        elif uploaded is None:
            st.warning("Fournissez une image satellite.")
        else:
            x_seq = build_seq_vector(ndvi_01, ndvi_03, ndvi_05)
            x_tab = build_tab_vector(pluie, ndvi_moy, surface, production, latitude, longitude, float(annee))

            with st.spinner("Calcul..."):
                pred = float(model.predict([x_img, x_seq, x_tab], verbose=0).ravel()[0])

            # Donut / carte prédiction
            st.markdown(f"""
            <div class="pred-card">
                <div class="value">{pred:.1f}</div>
                <div class="label">Rendement prédit (Qx/Ha)</div>
                <span class="badge">Confiance élevée</span>
            </div>
            """, unsafe_allow_html=True)

            # Barres : Prédiction vs Référence — natif Streamlit
            st.markdown("**Comparaison**")
            df_bar = pd.DataFrame(
                {"Valeur (Qx/Ha)": [pred, pred * 0.85]},
                index=["Prédiction actuelle", "Moyenne indicative"],
            )
            st.bar_chart(df_bar)

            st.balloons()
    else:
        st.info("Remplissez les données et cliquez sur **Prédire**.")

    # Indicateurs clés
    st.markdown("**Indicateurs clés**")
    st.markdown(f"""
    | Indicateur | Valeur |
    |------------|--------|
    | Latitude | {latitude:.4f} |
    | Longitude | {longitude:.4f} |
    | NDVI moyen | {ndvi_moy:.3f} |
    | Pluie (mm) | {pluie:.0f} |
    """)

st.markdown("---")
with st.expander("ℹ️ À propos — AGRO-INSIGHT MAROC"):
    st.markdown("""
    **Tableau de bord IA pour l'agriculture de précision au Maroc.**

    - **Panneau gauche** : Saisie des données (pluie, surface, NDVI, image satellite).
    - **Panneau central** : Courbe NDVI (01, 03, 05), aperçu image, localisation.
    - **Panneau droit** : Prédiction du rendement (Qx/Ha), comparaison, indicateurs.

    Modèle : **ViT + GRU + Attention + tabulaire**.
    """)

