"""
License Plate Price Estimator
Usage:
  python predict.py "WMA 4"
  python predict.py              (mode interactif)
"""

import sys
import os
import re
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features

# ── Charger le modèle ─────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'train_v2/models/best_model_v2.pkl')

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

model        = bundle['model']
le           = bundle['label_encoder']
feature_cols = bundle['features']
model_name   = bundle['model_name']

# ── Features d'interaction (identiques à train_v2.py) ────────────────────────

def add_interactions(X):
    X = X.copy()
    X['rarity_score']     = 1.0 / (X['num_digits'].clip(lower=1) * X['prefix_len'].clip(lower=1))
    X['same_and_lucky']   = X['all_same'] * X['lucky_count']
    X['single_short']     = X['is_single'] * (X['prefix_len'] <= 2).astype(int)
    X['palin_lucky']      = X['is_palindrome'] * X['lucky_count']
    X['digits_x_repeat']  = X['num_digits'] * X['max_repeat']
    X['unlucky_weighted'] = X['unlucky_count'] * X['num_digits']
    X = X.replace([float('inf'), float('-inf')], 0).fillna(0)
    return X

# ── Prédiction ────────────────────────────────────────────────────────────────

def predict_price(plate: str) -> float:
    feats = extract_features(plate)
    prefix = feats.pop('_prefix')
    feats['prefix_enc'] = int(le.transform([prefix])[0]) if prefix in le.classes_ else -1

    X = pd.DataFrame([feats])
    X = add_interactions(X)
    X = X[feature_cols]

    log_price = model.predict(X)[0]
    return float(np.exp(log_price))

# ── Explication ───────────────────────────────────────────────────────────────

def explain(plate: str, price: float) -> list[str]:
    plate = plate.strip().upper()
    letters   = re.sub(r'[^A-Z]', '', plate)
    digits_str = re.sub(r'[^0-9]', '', plate)
    digits    = int(digits_str) if digits_str else 0

    reasons = []

    if len(digits_str) == 1:
        reasons.append("numéro single digit — le plus rare du marché")
    elif len(digits_str) == 2:
        reasons.append("numéro à 2 chiffres — très rare")

    if len(letters) <= 2:
        reasons.append(f"préfixe court ({len(letters)} lettre{'s' if len(letters)>1 else ''}) — série ancienne/rare")

    if len(set(digits_str)) == 1 and len(digits_str) > 1:
        reasons.append(f"tous les chiffres identiques ({digits_str})")

    if digits_str == digits_str[::-1] and len(digits_str) > 1:
        reasons.append(f"palindrome ({digits_str})")

    if (len(digits_str) == 4 and digits_str[0] == digits_str[2]
            and digits_str[1] == digits_str[3] and digits_str[0] != digits_str[1]):
        reasons.append(f"pattern ABAB ({digits_str})")

    if '8888' in digits_str:
        reasons.append("contient 8888 — très chanceux")
    elif '888' in digits_str:
        reasons.append("contient 888 — chanceux")

    lucky = sum(1 for d in digits_str if d in '8916')
    if lucky == len(digits_str) and len(digits_str) > 1:
        reasons.append(f"tous les chiffres chanceux (8/9/1/6)")
    elif lucky >= 2:
        reasons.append(f"{lucky} chiffres chanceux (8/9/1/6)")

    if '4' in digits_str:
        reasons.append("contient un 4 — légèrement pénalisant culturellement")

    if not reasons:
        reasons.append("plaque standard — pas de pattern particulier")

    return reasons

# ── Affichage ─────────────────────────────────────────────────────────────────

def run(plate: str):
    plate_clean = plate.strip().upper()
    plate_clean = re.sub(r'\s+', ' ', plate_clean)

    try:
        price = predict_price(plate_clean)
    except Exception as e:
        print(f"  Erreur : {e}")
        return

    reasons = explain(plate_clean, price)

    print()
    print(f"  Plaque        : {plate_clean}")
    print(f"  Prix estimé   : RM {price:,.0f}")
    print(f"  Facteurs      :")
    for r in reasons:
        print(f"    • {r}")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Modèle chargé : {model_name} (V2)")

    if len(sys.argv) > 1:
        run(' '.join(sys.argv[1:]))
    else:
        print("=== Estimateur de prix — plaques malaisiennes ===")
        print("Tapez 'quit' pour quitter.\n")
        while True:
            plate = input("Plaque (ex: WMA 4, EV 8888, KH 1) : ").strip()
            if plate.lower() in ('quit', 'exit', 'q'):
                break
            if plate:
                run(plate)
