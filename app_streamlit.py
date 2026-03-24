# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
import joblib

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="AGRI-PREDICT PRO",
    page_icon="🌿",
    layout="wide",
)

# --- STYLE CSS : CLAIR, MODERNE ET ATTIRANT ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

/* Fond de l'application - Dégradé très clair */
.stApp {
    background: linear-gradient(135deg, #f0f9f1 0%, #ffffff 100%);
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Header Professionnel */
.main-header {
    background: white;
    padding: 1rem 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-left: 8px solid #4ade80;
}

/* Cartes (Panneaux) */
.css-1r6slb0, .stVerticalBlock {
    gap: 1.5rem;
}

div[data-testid="stVerticalBlock"] > div:has(div.panel) {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.03);
    border: 1px solid #f0f0f0;
}

/* Titres des sections */
.section-title {
    color: #1a3a32;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Bouton Prédire avec dégradé vert frais */
.stButton>button {
    background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3) !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(34, 197, 94, 0.4) !important;
}

/* Carte de résultat (Grande valeur) */
.result-card {
    background: #f0fdf4;
    border: 2px solid #bbf7d0;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

.result-value {
    font-size: 3.5rem;
    font-weight: 800;
    color: #15803d;
    line-height: 1;
}

.result-unit {
    font-size: 1.2rem;
    color: #166534;
}

/* Custom inputs */
.stNumberInput, .stSlider {
    margin-bottom: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- LOGIQUE DU MODÈLE (Simplifiée pour l'exemple) ---
class TemporalAttention(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1)
    def call(self, inputs):
        score = self.v(self.W(inputs))
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(weights * inputs, axis=1)

@st.cache_resource
def load_prediction_model():
    # Remplacez par votre vrai fichier .keras ou .h5
    # return tf.keras.models.load_model("votre_modele.keras", custom_objects={"TemporalAttention": TemporalAttention})
    return None 

# --- INTERFACE UTILISATEUR ---

# 1. HEADER
st.markdown("""
    <div class="main-header">
        <div>
            <h1 style='margin:0; color:#1a3a32; font-size:1.8rem;'>🌿 AGRI-PREDICT PRO</h1>
            <p style='margin:0; color:#6b7280;'>Moteur d'intelligence artificielle pour le rendement agricole</p>
        </div>
        <div style='text-align:right'>
            <span style='background:#dcfce7; color:#166534; padding:5px 15px; border-radius:20px; font-size:0.8rem; font-weight:600;'>
                SATELLITE CONNECTED
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

col_input, col_viz = st.columns([1, 1.8], gap="large")

# --- COLONNE GAUCHE : SAISIE (Claire) ---
with col_input:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Paramètres du Champ</div>', unsafe_allow_html=True)
    
    region = st.selectbox("Région", ["Béni Mellal-Khénifra", "Souss-Massa", "Gharb", "Haouz"])
    
    c1, c2 = st.columns(2)
    with c1:
        pluie = st.number_input("Précipitations (mm)", value=400)
    with c2:
        temp = st.number_input("Température (°C)", value=24)
        
    surface = st.slider("Surface cultivée (Ha)", 1, 500, 50)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛰️ Données Satellite</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Télécharger l'image NDVI", type=['jpg','png','jpeg'])
    
    ndvi_03 = st.slider("Indice NDVI (Mars)", 0.0, 1.0, 0.45)
    
    predict_btn = st.button("CALCULER LE RENDEMENT")
    st.markdown('</div>', unsafe_allow_html=True)

# --- COLONNE DROITE : VISUALISATION ---
with col_viz:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    
    if predict_btn:
        # Simulation de prédiction
        rendement_simule = 42.8
        
        st.markdown('<div class="section-title">📊 Analyse des Résultats</div>', unsafe_allow_html=True)
        
        # Carte de résultat style "Apple/Premium"
        st.markdown(f"""
            <div class="result-card">
                <p style='margin:0; color:#166534; font-weight:600; text-transform:uppercase; letter-spacing:1px;'>Rendement Estimé</p>
                <div class="result-value">{rendement_simule}</div>
                <div class="result-unit">Quintaux par Hectare (Qx/Ha)</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Graphiques
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Comparaison Historique**")
            chart_data = pd.DataFrame({
                'Année': ['2022', '2023', '2024', 'Prédiction'],
                'Rendement': [35, 38, 32, rendement_simule]
            })
            st.bar_chart(chart_data.set_index('Année'))
            
        with c2:
            st.markdown("**Indice de Confiance**")
            st.progress(88)
            st.write("Le modèle est confiant à **88%** basé sur les données météo et satellite.")
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Analyse de la biomasse", use_column_width=True)
        
        st.success("✅ Analyse terminée avec succès pour la région " + region)
    else:
        # État vide (Empty State)
        st.info("Veuillez configurer les paramètres à gauche et cliquer sur le bouton pour lancer l'analyse.")
        
        # Petite image d'illustration ou icône
        st.markdown("""
            <div style='text-align:center; padding: 5rem 0; opacity: 0.2;'>
                <img src='https://cdn-icons-png.flaticon.com/512/2910/2910768.png' width='150'>
                <p>En attente de données...</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><hr><center style='color:#9ca3af; font-size:0.8rem;'>AGRI-PREDICT IA © 2024 - Technologie Nano-Precision</center>", unsafe_allow_html=True)

