"""
Malaysian License Plate Price Estimator — Streamlit Dashboard
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Malaysian Plate Estimator",
    page_icon="🚗",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), 'train_v2/models/best_model_v2.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

bundle       = load_model()
model        = bundle['model']
le           = bundle['label_encoder']
feature_cols = bundle['features']
model_name   = bundle['model_name']

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_interactions(X):
    X = X.copy()
    X['rarity_score']     = 1.0 / (X['num_digits'].clip(lower=1) * X['prefix_len'].clip(lower=1))
    X['same_and_lucky']   = X['all_same'] * X['lucky_count']
    X['single_short']     = X['is_single'] * (X['prefix_len'] <= 2).astype(int)
    X['palin_lucky']      = X['is_palindrome'] * X['lucky_count']
    X['digits_x_repeat']  = X['num_digits'] * X['max_repeat']
    X['unlucky_weighted'] = X['unlucky_count'] * X['num_digits']
    # Sécurité : remplacer inf et NaN par 0
    X = X.replace([float('inf'), float('-inf')], 0).fillna(0)
    return X

def predict_price(plate: str):
    feats = extract_features(plate)
    prefix = feats.pop('_prefix')
    feats['prefix_enc'] = int(le.transform([prefix])[0]) if prefix in le.classes_ else -1
    X = pd.DataFrame([feats])
    X = add_interactions(X)
    X = X[feature_cols]
    log_price = model.predict(X)[0]
    price = float(np.exp(log_price))
    # Fourchette ±30% (MAPE du modèle)
    low  = price * 0.70
    high = price * 1.30
    return price, low, high

def get_factors(plate: str):
    plate = plate.strip().upper()
    letters    = re.sub(r'[^A-Z]', '', plate)
    digits_str = re.sub(r'[^0-9]', '', plate)
    digits     = int(digits_str) if digits_str else 0
    factors = []

    if len(digits_str) == 1:
        factors.append(("🏆", "Single digit — le plus rare du marché"))
    elif len(digits_str) == 2:
        factors.append(("⭐", "Numéro à 2 chiffres — très rare"))

    if len(letters) <= 2:
        factors.append(("⭐", f"Préfixe court ({len(letters)} lettre{'s' if len(letters)>1 else ''}) — série ancienne"))

    if len(set(digits_str)) == 1 and len(digits_str) > 1:
        factors.append(("🔢", f"Tous les chiffres identiques ({digits_str})"))

    if digits_str == digits_str[::-1] and len(digits_str) > 1:
        factors.append(("🔄", f"Palindrome ({digits_str})"))

    if (len(digits_str) == 4 and digits_str[0] == digits_str[2]
            and digits_str[1] == digits_str[3] and digits_str[0] != digits_str[1]):
        factors.append(("🔁", f"Pattern ABAB ({digits_str})"))

    if '8888' in digits_str:
        factors.append(("🎰", "Contient 8888 — très chanceux"))
    elif '888' in digits_str:
        factors.append(("🎰", "Contient 888 — chanceux"))

    lucky = sum(1 for d in digits_str if d in '8916')
    if lucky == len(digits_str) and len(digits_str) > 1:
        factors.append(("🍀", "Tous les chiffres chanceux (8/9/1/6)"))
    elif lucky >= 2:
        factors.append(("🍀", f"{lucky} chiffres chanceux (8/9/1/6)"))

    if '4' in digits_str:
        factors.append(("⚠️", "Contient un 4 — légèrement pénalisant culturellement"))

    if not factors:
        factors.append(("📋", "Plaque standard — pas de pattern particulier"))

    return factors

def is_valid_plate(plate: str) -> bool:
    return bool(re.match(r'^[A-Z0-9\s]+$', plate.strip().upper()))

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🚗 Malaysian Plate Price Estimator")
st.caption("Estimez la valeur marchande d'une plaque d'immatriculation spéciale malaisienne")

st.divider()

# Input
plate_input = st.text_input(
    "Entrez une plaque",
    placeholder="ex: WMA 4, EV 8888, KH 1...",
    max_chars=20
).strip().upper()
plate_input = re.sub(r'\s+', ' ', plate_input)

col_btn, _ = st.columns([1, 3])
with col_btn:
    estimate = st.button("Estimer le prix", type="primary", use_container_width=True)

# Résultat
if estimate and plate_input:
    if not is_valid_plate(plate_input):
        st.error("Plaque invalide — utilisez uniquement des lettres et des chiffres.")
    else:
        try:
            price, low, high = predict_price(plate_input)

            st.divider()

            # Prix central
            st.markdown(f"### Plaque : `{plate_input}`")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prix bas estimé", f"RM {low:,.0f}")
            col2.metric("Prix central", f"RM {price:,.0f}")
            col3.metric("Prix haut estimé", f"RM {high:,.0f}")

            st.caption(f"Fourchette basée sur la marge d'erreur du modèle (±30% — MAPE=31.7%)")

            # Facteurs
            st.divider()
            st.markdown("**Facteurs de valeur détectés :**")
            for icon, text in get_factors(plate_input):
                st.markdown(f"{icon} {text}")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

elif estimate and not plate_input:
    st.warning("Veuillez entrer une plaque.")

# ── Liens ─────────────────────────────────────────────────────────────────────

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report/report_license_plate.pdf')
    if os.path.exists(pdf_path):
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        st.download_button(
            label="📄 Télécharger le rapport du projet",
            data=pdf_bytes,
            file_name="report_license_plate.pdf",
            mime="application/pdf",
            use_container_width=True
        )

with col_b:
    st.link_button(
        "🌐 Voir les plaques sur Motor Trader",
        url="https://www.motortrader.com.my/numberplate",
        use_container_width=True
    )

st.divider()
st.caption(f"Modèle : {model_name} | Données : Motor Trader (web scraping) | 2\,879 plaques")
