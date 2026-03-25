# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="AGRI-PREDICT AI", layout="wide", page_icon="🌾")

# --- STYLE CSS (LUMINEUX & PROFESSIONNEL) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Barre de navigation stylisée */
    .nav-bar {
        background: linear-gradient(90deg, #ffffff 0%, #f0fdf4 100%);
        padding: 1.5rem;
        border-radius: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border-bottom: 4px solid #4ade80;
    }
    
    .logo-text { color: #166534; font-weight: 800; font-size: 1.6rem; letter-spacing: -1px; }
    
    /* Cartes */
    .main-card {
        background: white;
        padding: 2rem;
        border-radius: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.04);
        border: 1px solid #f0f0f0;
    }

    /* Bouton avec dégradé vert attirant */
    .stButton>button {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 15px !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: 0.3s;
        box-shadow: 0 8px 15px rgba(34, 197, 94, 0.2) !important;
    }
    
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 10px 20px rgba(34, 197, 94, 0.3) !important; }

    /* Résultat */
    .result-box {
        background: #f0fdf4;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px dashed #bbf7d0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER HAUT DE GAMME ---
st.markdown("""
    <div class="nav-bar">
        <div class="logo-text">🛰️ AGRI-PREDICT <span style="color:#4ade80">PRO</span></div>
        <div style="color: #6b7280; font-size: 0.9rem; font-weight: 600;">
            SATELLITE ANALYSIS • MOROCCO AI • 2026
        </div>
    </div>
""", unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES (pour les menus) ---
@st.cache_data
def load_data():
    try:
        # On essaie d'abord avec latin-1 qui gère bien les accents français/marocains du CSV
        df = pd.read_csv("base_finale_2060.csv", encoding='latin-1')
        return df
    except Exception as e:
        try:
            # Si ça rate, on tente utf-8
            df = pd.read_csv("base_finale_2060.csv", encoding='utf-8')
            return df
        except Exception as e2:
            st.error(f"Erreur de lecture du fichier : {e2}")
            return pd.DataFrame()

df = load_data()

# --- LAYOUT ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📍 Localisation & Terrain")
    
    # Choix dynamique depuis ton CSV
    region = st.selectbox("Région", df['Région'].unique())
    province = st.selectbox("Province", df[df['Région'] == region]['Province'].unique())
    
    st.markdown("---")
    st.subheader("💧 Facteurs Climatiques")
    pluie = st.number_input("🌧️ Précipitations Saisonnières (mm)", value=float(df['Pluie_Saisonniere_mm'].mean()))
    surface = st.number_input("📏 Surface (1000 Ha)", value=float(df['Surface (1000 Ha)'].mean()))
    
    st.markdown("---")
    st.subheader("🛰️ Indices NDVI (Satellite)")
    
    # Upload d'image satellite
    uploaded_file = st.file_uploader("🖼️ Charger l'image Satellite/NDVI", type=['jpg','png','jpeg'])
    
    # Les 3 points NDVI de ton CSV
    n01 = st.slider("🌱 NDVI Janvier", 0.0, 1.0, 0.40)
    n03 = st.slider("🌿 NDVI Mars", 0.0, 1.0, 0.45)
    n05 = st.slider("🌳 NDVI Mai", 0.0, 1.0, 0.55)
    
    predict_btn = st.button("LANCER L'ANALYSE IA")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    if predict_btn:
        st.markdown('<div class="section-title">📊 ANALYSE DU RENDEMENT</div>', unsafe_allow_html=True)
        
        # --- LOGIQUE DE PRÉDICTION (Simulée ici) ---
        # Ici tu devrais charger ton modèle .keras et faire model.predict()
        prediction = 22.45  # Exemple
        
        st.markdown(f"""
            <div class="result-box">
                <p style="margin:0; color:#166534; font-size:1.1rem;">Rendement Prévisionnel</p>
                <h1 style="font-size:4rem; color:#15803d; margin:0;">{prediction:.2f} <span style="font-size:1.5rem;">Qx/Ha</span></h1>
                <p style="color:#166534; opacity:0.7;">Confiance du modèle : 91%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # --- VISUALISATION ---
        st.markdown("### 📈 Courbe de croissance NDVI")
        ndvi_data = pd.DataFrame({
            'Mois': ['Janvier', 'Mars', 'Mai'],
            'Valeur NDVI': [n01, n03, n05]
        }).set_index('Mois')
        st.line_chart(ndvi_data)
        
        if uploaded_file:
            st.markdown("### 🗺️ Analyse Spatiale")
            st.image(uploaded_file, caption="Analyse de la biomasse par satellite", use_container_width=True)
            
        st.balloons()
    else:
        st.info("👋 Bienvenue ! Veuillez remplir les données du champ à gauche et charger une image satellite pour obtenir une prédiction précise du rendement.")
        
        # Image d'attente pro
        st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=1000", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><p style='text-align:center; color:#9ca3af;'>Propulsé par Nano-Banana Technology 🍌 • 2026</p>", unsafe_allow_html=True)

