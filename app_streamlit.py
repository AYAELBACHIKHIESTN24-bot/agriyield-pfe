# -*- coding: utf-8 -*-
"""
AgriYield Morocco — Application de prédiction du rendement agricole par IA
PFE · Imagerie satellite + NDVI + variables tabulaires
"""

from __future__ import annotations

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Couche personnalisée (identique au notebook)
# ---------------------------------------------------------------------------
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
TABULAR_COLS = [
    "Pluie_Saisonniere_mm", "NDVI_Moyen", "Surface (1000 Ha)",
    "Production (1000 Qx)", "Latitude", "Longitude", "Année",
]


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
# STYLES — Thème agricole professionnel
# ============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

:root {
    --primary: #1b5e20;
    --primary-light: #2e7d32;
    --primary-dark: #0d3d10;
    --accent: #c6ff00;
    --gold: #ffc107;
    --bg-dark: #0a1f0a;
    --card-bg: linear-gradient(135deg, #1a3d1a 0%, #0d2810 100%);
    --text-light: #e8f5e9;
    --text-muted: #a5d6a7;
}

/* Fond global */
.stApp {
    background: linear-gradient(180deg, #0a1f0a 0%, #0f2d0f 30%, #133013 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2810 0%, #0a1f0a 100%);
    border-right: 1px solid rgba(198, 255, 0, 0.2);
}
[data-testid="stSidebar"] .stMarkdown { color: #e8f5e9 !important; }
[data-testid="stSidebar"] label { color: #a5d6a7 !important; }

/* Titre principal */
.hero {
    background: linear-gradient(135deg, rgba(27,94,32,0.9) 0%, rgba(13,61,16,0.95) 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(198, 255, 0, 0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero h1 {
    font-family: 'Poppins', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #c6ff00;
    margin: 0 0 0.5rem 0;
    text-shadow: 0 0 20px rgba(198,255,0,0.3);
}
.hero p {
    color: #a5d6a7;
    font-size: 1rem;
    margin: 0;
}

/* Cartes */
.metric-card {
    background: linear-gradient(135deg, rgba(27,94,32,0.6) 0%, rgba(13,61,16,0.8) 100%);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(198,255,0,0.2);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #c6ff00;
}
.metric-card .label {
    font-size: 0.85rem;
    color: #a5d6a7;
}

/* Zone image */
.img-box {
    border-radius: 12px;
    overflow: hidden;
    border: 2px solid rgba(198,255,0,0.3);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}

/* Bouton prédire */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #2e7d32, #1b5e20) !important;
    color: #c6ff00 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
}

/* Résumé entrées */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(198,255,0,0.2);
}
"""

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriYield Morocco | PFE",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🌾 AgriYield Morocco</h1>
    <p><b>Prédiction du rendement agricole</b> par imagerie satellite, séries NDVI et variables tabulaires — Projet PFE · Intelligence artificielle appliquée à l'agriculture marocaine</p>
</div>
""", unsafe_allow_html=True)

model = load_model()
err = st.session_state.pop("_load_error", None)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Données d'entrée")
    st.markdown("*Image satellite · NDVI · Pluviométrie · Localisation*")
    st.markdown("---")

    uploaded = st.file_uploader(
        "**Image satellite** (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        help="Parcelle agricole · 128×128 px",
    )

    st.markdown("---")
    st.markdown("**🌿 Série NDVI (janvier · mars · mai)**")
    ndvi_01 = st.slider("NDVI_01 (Janvier)", -0.2, 1.0, 0.40, 0.01)
    ndvi_03 = st.slider("NDVI_03 (Mars)", -0.2, 1.0, 0.45, 0.01)
    ndvi_05 = st.slider("NDVI_05 (Mai)", -0.2, 1.0, 0.55, 0.01)

    st.markdown("---")
    st.markdown("**📊 Variables tabulaires**")
    pluie = st.number_input("Pluie saisonnière (mm)", min_value=0.0, value=450.0, step=1.0)
    ndvi_moy = st.number_input("NDVI moyen", min_value=-0.2, max_value=1.0, value=0.45, step=0.01)
    surface = st.number_input("Surface (1000 Ha)", min_value=0.0, value=50.0, step=0.1)
    production = st.number_input("Production (1000 Qx)", min_value=0.0, value=600.0, step=0.1)
    latitude = st.number_input("Latitude", value=35.20, format="%.6f")
    longitude = st.number_input("Longitude", value=-3.93, format="%.6f")
    annee = st.number_input("Année", min_value=2000, max_value=2035, value=2021, step=1)

    st.markdown("---")
    predict_btn = st.button("🚀 Prédire le rendement", use_container_width=True, type="primary")

    # Statut modèle
    st.markdown("---")
    if model is None:
        st.error("Modèle non chargé")
        if err:
            st.caption(str(err)[:200])
    else:
        st.success("✅ Modèle prêt (ViT + GRU + Attention)")

# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
col_img, col_summary = st.columns([1.2, 1])

with col_img:
    st.markdown("#### 📷 Image chargée")
    if uploaded is not None:
        x_img, pil_preview = preprocess_image(uploaded)
        st.markdown('<div class="img-box">', unsafe_allow_html=True)
        st.image(pil_preview, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Chargez une image satellite dans la barre latérale pour commencer.")
        x_img = None

with col_summary:
    st.markdown("#### 📋 Résumé des entrées")
    preview = {
        "NDVI (01, 03, 05)": f"{ndvi_01:.2f}, {ndvi_03:.2f}, {ndvi_05:.2f}",
        "Pluie (mm)": pluie,
        "NDVI moyen": ndvi_moy,
        "Surface (1000 Ha)": surface,
        "Production (1000 Qx)": production,
        "Coordonnées": f"({latitude:.4f}, {longitude:.4f})",
        "Année": annee,
    }
    st.dataframe(preview, use_container_width=True, height=280)

# ---------------------------------------------------------------------------
# PRÉDICTION
# ---------------------------------------------------------------------------
if predict_btn:
    if model is None:
        st.error("Impossible de prédire : modèle introuvable.")
    elif uploaded is None:
        st.warning("Veuillez fournir une image satellite.")
    else:
        x_seq = build_seq_vector(ndvi_01, ndvi_03, ndvi_05)
        x_tab = build_tab_vector(pluie, ndvi_moy, surface, production, latitude, longitude, float(annee))

        with st.spinner("Calcul en cours..."):
            pred = float(model.predict([x_img, x_seq, x_tab], verbose=0).ravel()[0])

        st.markdown("---")
        st.markdown("### 🎯 Résultat de la prédiction")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{pred:.2f}</div>
                <div class="label">Rendement prédit (Qx/Ha)</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.metric("NDVI moyen saisi", f"{ndvi_moy:.3f}")
        with c3:
            st.metric("Pluie (mm)", f"{pluie:.1f}")

        st.balloons()
        st.success("✅ Prédiction terminée.")

# ---------------------------------------------------------------------------
# FOOTER — À propos
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ À propos du projet et du modèle"):
    st.markdown("""
    **AgriYield Morocco** — Projet de Fin d'Études
    
    - **Objectif** : prédire le rendement agricole (Qx/Ha) à partir de l'imagerie satellite, des indices NDVI et des données météo/géographiques.
    - **Modèle** : **ViT + GRU + Attention + tabulaire** — architecture hybride combinant Vision Transformer (image), GRU (série NDVI), couche d'attention temporelle et variables tabulaires.
    - **Données** : base multi-sources (HCP, NASA, Copernicus, Yandex Maps) sur les régions agricoles du Maroc.
    """)
