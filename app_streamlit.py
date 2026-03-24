# -*- coding: utf-8 -*-
"""
Application Streamlit — alignée sur le notebook Colab (ViT + GRU + Attention + tabulaire).

Export depuis Colab (après entraînement du modèle final) :
    model_vit_gru_att_tab.save("model_vit_gru_att_tab.keras")

Copiez le fichier .keras dans le même dossier que ce script (ou définissez MODEL_PATH).

Entrées : même ordre que tabular_cols du notebook + NDVI_01/03/05 + image 128×128 (/255).
"""

from __future__ import annotations

import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Couche personnalisée (identique au notebook) — obligatoire pour load_model
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

# Ordre strict = notebook Colab `tabular_cols`
TABULAR_COLS = [
    "Pluie_Saisonniere_mm",
    "NDVI_Moyen",
    "Surface (1000 Ha)",
    "Production (1000 Qx)",
    "Latitude",
    "Longitude",
    "Année",
]


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"TemporalAttention": TemporalAttention},
        )
    except Exception as e:
        st.session_state["_load_error"] = str(e)
        return None


def preprocess_image(uploaded_file) -> tuple[np.ndarray, Image.Image]:
    pil = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x_img = np.expand_dims(arr, axis=0)
    return x_img, pil


def build_tab_vector(
    pluie: float,
    ndvi_moy: float,
    surface: float,
    production: float,
    lat: float,
    lon: float,
    annee: float,
) -> np.ndarray:
    row = np.array(
        [[pluie, ndvi_moy, surface, production, lat, lon, annee]],
        dtype=np.float32,
    )
    return row


def build_seq_vector(ndvi_01: float, ndvi_03: float, ndvi_05: float) -> np.ndarray:
    seq = np.array([[ndvi_01, ndvi_03, ndvi_05]], dtype=np.float32).reshape(1, 3, 1)
    return seq


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriYield Maroc | PFE",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style discret (vert / pro)
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .main-title { font-size: 2rem; font-weight: 700; color: #1b5e20; }
    .sub { color: #555; font-size: 1rem; margin-top: -0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🌾 Prédiction du rendement agricole — Maroc</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Modèle <b>ViT + GRU + Attention + tabulaire</b> · Image satellite 128×128 + 3 NDVI + 7 variables tabulaires</p>',
    unsafe_allow_html=True,
)

model = load_model()
err = st.session_state.pop("_load_error", None)

col_info, col_status = st.columns([2, 1])
with col_info:
    st.caption(f"Fichier modèle : `{MODEL_PATH}`")
with col_status:
    if model is None:
        st.error("Modèle non chargé")
        if err:
            st.code(err, language="text")
    else:
        st.success("Modèle prêt")

st.sidebar.markdown("### 📥 Entrées")
st.sidebar.markdown("**Image** · **NDVI (jan, mar, mai)** · **Données tabulaires**")

uploaded = st.sidebar.file_uploader(
    "Image satellite (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
    help="Même prétraitement que le notebook : RGB, redimensionnement 128×128, normalisation /255",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Série NDVI**")
ndvi_01 = st.sidebar.slider("NDVI_01", -0.2, 1.0, 0.40, 0.01, help="Janvier (dataset)")
ndvi_03 = st.sidebar.slider("NDVI_03", -0.2, 1.0, 0.45, 0.01, help="Mars")
ndvi_05 = st.sidebar.slider("NDVI_05", -0.2, 1.0, 0.55, 0.01, help="Mai")

st.sidebar.markdown("---")
st.sidebar.markdown("**Variables tabulaires**")
pluie = st.sidebar.number_input("Pluie saisonnière (mm)", min_value=0.0, value=450.0, step=1.0)
ndvi_moy = st.sidebar.number_input("NDVI moyen", min_value=-0.2, max_value=1.0, value=0.45, step=0.01)
surface = st.sidebar.number_input("Surface (1000 Ha)", min_value=0.0, value=50.0, step=0.1)
production = st.sidebar.number_input("Production (1000 Qx)", min_value=0.0, value=600.0, step=0.1)
latitude = st.sidebar.number_input("Latitude", value=35.20, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-3.93, format="%.6f")
annee = st.sidebar.number_input("Année", min_value=2000, max_value=2035, value=2021, step=1)

predict_btn = st.sidebar.button("🚀 Prédire le rendement", use_container_width=True, type="primary")

left, right = st.columns([1.1, 1])

with left:
    st.subheader("Aperçu")
    if uploaded is not None:
        x_img, pil_preview = preprocess_image(uploaded)
        st.image(pil_preview, caption="Image chargée (128×128 après traitement)", use_container_width=True)
    else:
        st.info("Chargez une image satellite dans la barre latérale.")
        x_img = None

with right:
    st.subheader("Résumé des entrées")
    preview = {
        "NDVI_01": ndvi_01,
        "NDVI_03": ndvi_03,
        "NDVI_05": ndvi_05,
        **{TABULAR_COLS[i]: v for i, v in enumerate([pluie, ndvi_moy, surface, production, latitude, longitude, float(annee)])},
    }
    st.dataframe(preview, use_container_width=True, height=320)

if predict_btn:
    if model is None:
        st.error(
            "Impossible de prédire : modèle introuvable ou erreur de chargement. "
            "Exportez depuis Colab : `model_vit_gru_att_tab.save('model_vit_gru_att_tab.keras')` "
            "et placez le fichier à côté de `app_streamlit.py`."
        )
    elif uploaded is None:
        st.warning("Veuillez fournir une image satellite.")
    else:
        x_seq = build_seq_vector(ndvi_01, ndvi_03, ndvi_05)
        x_tab = build_tab_vector(pluie, ndvi_moy, surface, production, latitude, longitude, float(annee))

        with st.spinner("Inférence en cours…"):
            pred = float(model.predict([x_img, x_seq, x_tab], verbose=0).ravel()[0])

        m1, m2, m3 = st.columns(3)
        m1.metric("Rendement prédit", f"{pred:.2f} Qx/Ha")
        m2.metric("NDVI moyen (saisi)", f"{ndvi_moy:.3f}")
        m3.metric("Pluie (mm)", f"{pluie:.1f}")

        st.balloons()
        st.success("Prédiction terminée.")

st.markdown("---")
with st.expander("ℹ️ Correspondance avec le notebook Colab"):
    st.markdown(
        """
        - **Cible** : `Rendement (Qx/Ha)` (sortie du modèle).
        - **Entrées** : `[Image_Input, NDVI_Sequence_Input, Tabular_Input]` dans le même ordre que `model.predict([X_img, X_seq, X_tab])`.
        - **Colonnes tabulaires** (ordre fixe) :  
          `Pluie_Saisonniere_mm`, `NDVI_Moyen`, `Surface (1000 Ha)`, `Production (1000 Qx)`, `Latitude`, `Longitude`, `Année`.
        - Si vous réentraînez le modèle, **remplacez seulement le fichier `.keras`** : l’application ne change pas tant que l’architecture et les entrées restent identiques.
        """
    )
