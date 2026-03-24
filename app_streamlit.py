# -*- coding: utf-8 -*-
"""
AgriYield Morocco — Prédiction du rendement agricole par IA (PFE)
"""

from __future__ import annotations

import os
import numpy as np
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
# STYLES — Thème CLAIR
# ============================================================================
CUSTOM_CSS = """
.stApp {
    background: linear-gradient(180deg, #f1f8e9 0%, #dcedc8 50%, #c5e1a5 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f5e9 0%, #c8e6c9 100%);
}
.step-box {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #4caf50;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.step-box .icon { font-size: 1.5rem; }
.step-box .title { font-weight: 600; color: #2e7d32; }
.step-box .desc { font-size: 0.85rem; color: #555; }
.result-card {
    background: linear-gradient(135deg, #81c784 0%, #66bb6a 100%);
    color: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 6px 20px rgba(76,175,80,0.4);
}
.result-card .value { font-size: 2.5rem; font-weight: 700; }
.result-card .label { font-size: 1rem; opacity: 0.95; }
"""

st.set_page_config(page_title="AgriYield Morocco | PFE", page_icon="🌾", layout="wide")
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# EXPLICATION : CE QUE CONTIENT L'APPLICATION
# ---------------------------------------------------------------------------
st.markdown("# 🌾 AgriYield Morocco")
st.markdown("**Prédiction du rendement agricole (Qx/Ha) — Projet PFE**")
st.markdown("")

st.markdown("### 📖 Comment ça marche ?")
st.markdown("""
L'application a besoin de **3 types de données** pour prédire le rendement.  
Remplis chaque section dans la barre à gauche, puis clique sur **Prédire**.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="step-box">
        <span class="icon">📷</span> <span class="title">1. Image satellite</span><br>
        <span class="desc">Une photo de la parcelle agricole (PNG/JPG). Le modèle analyse la végétation visible.</span>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="step-box">
        <span class="icon">🌿</span> <span class="title">2. NDVI (3 valeurs)</span><br>
        <span class="desc">Indices de végétation pour Janvier, Mars et Mai. Mesurent la santé des cultures.</span>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="step-box">
        <span class="icon">🌧️</span> <span class="title">3. Variables tabulaires</span><br>
        <span class="desc">Pluie, surface, production, coordonnées, année. Contexte météo et géographique.</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("**Après avoir cliqué sur « Prédire »** : l'application affiche la **valeur prédite** (rendement en Qx/Ha). Il n'y a **pas de graphique** — juste le nombre calculé par le modèle.")
st.markdown("---")

model = load_model()
err = st.session_state.pop("_load_error", None)

# ---------------------------------------------------------------------------
# SIDEBAR — Avec icônes et explications
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📥 Entrées")

    st.markdown("### 📷 Image satellite")
    st.caption("Parcelle agricole · PNG ou JPG")
    uploaded = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🌿 NDVI (janvier · mars · mai)")
    st.caption("Indice de végétation entre -0.2 et 1")
    ndvi_01 = st.slider("NDVI_01 (Janvier)", -0.2, 1.0, 0.40, 0.01)
    ndvi_03 = st.slider("NDVI_03 (Mars)", -0.2, 1.0, 0.45, 0.01)
    ndvi_05 = st.slider("NDVI_05 (Mai)", -0.2, 1.0, 0.55, 0.01)

    st.markdown("---")
    st.markdown("### 🌧️ Pluie & météo")
    st.caption("Pluie saisonnière en mm")
    pluie = st.number_input("Pluie (mm)", min_value=0.0, value=450.0, step=1.0)

    st.markdown("### 📊 Données parcelle")
    st.caption("Surface, production, localisation")
    ndvi_moy = st.number_input("NDVI moyen", min_value=-0.2, max_value=1.0, value=0.45, step=0.01)
    surface = st.number_input("Surface (1000 Ha)", min_value=0.0, value=50.0, step=0.1)
    production = st.number_input("Production (1000 Qx)", min_value=0.0, value=600.0, step=0.1)
    latitude = st.number_input("Latitude", value=35.20, format="%.6f")
    longitude = st.number_input("Longitude", value=-3.93, format="%.6f")
    annee = st.number_input("Année", min_value=2000, max_value=2035, value=2021, step=1)

    st.markdown("---")
    predict_btn = st.button("🚀 Prédire le rendement", use_container_width=True, type="primary")

    st.markdown("---")
    if model is None:
        st.error("Modèle non chargé")
        if err:
            st.caption(str(err)[:150])
    else:
        st.success("✅ Modèle prêt")

# ---------------------------------------------------------------------------
# ZONE PRINCIPALE
# ---------------------------------------------------------------------------
col_img, col_info = st.columns([1, 1])

with col_img:
    st.markdown("### 📷 Image chargée")
    if uploaded is not None:
        x_img, pil_preview = preprocess_image(uploaded)
        st.image(pil_preview, use_container_width=True)
    else:
        st.info("Chargez une image dans la barre à gauche.")
        x_img = None

with col_info:
    st.markdown("### 📋 Récapitulatif des entrées")
    st.markdown(f"""
    | Donnée | Valeur |
    |--------|--------|
    | NDVI (01, 03, 05) | {ndvi_01:.2f}, {ndvi_03:.2f}, {ndvi_05:.2f} |
    | Pluie (mm) | {pluie:.0f} |
    | NDVI moyen | {ndvi_moy:.2f} |
    | Surface (1000 Ha) | {surface:.1f} |
    | Production (1000 Qx) | {production:.1f} |
    | Coordonnées | ({latitude:.4f}, {longitude:.4f}) |
    | Année | {int(annee)} |
    """)

# ---------------------------------------------------------------------------
# RÉSULTAT (valeur uniquement — pas de graphique)
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
        st.markdown("*Le modèle affiche la **valeur prédite** — pas de graphique.*")
        st.markdown("")
        st.markdown(f"""
        <div class="result-card">
            <div class="value">{pred:.2f}</div>
            <div class="label">Rendement prédit (Qx/Ha)</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
        st.success("✅ Prédiction terminée.")

st.markdown("---")
with st.expander("ℹ️ À propos"):
    st.markdown("""
    **AgriYield Morocco** — PFE
    
    - **Données requises** : Image satellite + 3 NDVI + pluie + surface + production + coordonnées + année
    - **Modèle** : ViT + GRU + Attention + tabulaire
    - **Sortie** : une seule valeur (rendement en Qx/Ha), pas de graphique
    """)
