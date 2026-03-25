"""
License Plate Price — Data Cleaning
Input  : licence_plate_price.csv
Output : licence_plate_price_cleaned.csv
"""

import pandas as pd
import re

df = pd.read_csv('licence_plate_price.csv', header=0)
df.columns = ['plate', 'price']

print("=" * 50)
print("LICENSE PLATE PRICE — CLEANING")
print("=" * 50)
print(f"Entrées initiales : {len(df)}")

# ── 1. Supprimer les prix POA (Price On Application) ──
poa_mask = df['price'].str.strip().str.upper() == 'POA'
print(f"\n[1] Suppression POA : {poa_mask.sum()} lignes")
df = df[~poa_mask].copy()

# ── 2. Nettoyer les plaques mal formées ───────────────

# 2a. Supprimer les alias entre parenthèses ex: "JAB 4 (VIP 3)" → "JAB 4"
df['plate'] = df['plate'].str.replace(r'\(.*?\)', '', regex=True).str.strip()

# 2b. Mettre en majuscules
df['plate'] = df['plate'].str.upper().str.strip()

# 2c. Remplacer les caractères spéciaux (?, *, etc.) — supprimer la ligne si plaque illisible
def is_clean_plate(val):
    return bool(re.match(r'^[A-Z0-9\s]+$', str(val).strip()))

bad_plate_mask = ~df['plate'].apply(is_clean_plate)
print(f"[2] Suppression plaques illisibles : {bad_plate_mask.sum()} lignes")
if bad_plate_mask.any():
    print("    →", df[bad_plate_mask]['plate'].tolist())
df = df[~bad_plate_mask].copy()

# ── 3. Nettoyer et parser les prix ────────────────────

def parse_price(val):
    """'RM 3,800' → 3800.0"""
    try:
        return float(re.sub(r'[^\d.]', '', str(val)))
    except:
        return None

df['price_rm'] = df['price'].apply(parse_price)

# Supprimer les prix non parsables
bad_price = df['price_rm'].isna()
print(f"[3] Suppression prix non parsables : {bad_price.sum()} lignes")
df = df[~bad_price].copy()

# ── 4. Normaliser les espaces dans les plaques ────────
df['plate'] = df['plate'].str.strip().str.replace(r'\s+', ' ', regex=True)

# ── 5. Supprimer les prix anormalement bas (< RM 1000) ─
low_price = df['price_rm'] < 1000
print(f"[4] Suppression prix < RM 1000 : {low_price.sum()} lignes")
if low_price.any():
    print("    →", df[low_price][['plate', 'price_rm']].values.tolist())
df = df[~low_price].copy()

# ── 6. Supprimer les doublons ─────────────────────────
dupes = df.duplicated(subset=['plate'], keep='first').sum()
print(f"[5] Suppression doublons (même plaque) : {dupes}")
df = df.drop_duplicates(subset=['plate'], keep='first')

# ── 7. Résultat final ─────────────────────────────────
print(f"\nEntrées finales : {len(df)}")
print(f"Supprimées      : {3420 - len(df)}")
print(f"\nPrix — Min    : RM {df['price_rm'].min():>10,.0f}")
print(f"Prix — Max    : RM {df['price_rm'].max():>10,.0f}")
print(f"Prix — Médiane: RM {df['price_rm'].median():>10,.0f}")

# ── 8. Sauvegarder ───────────────────────────────────
df_out = df[['plate', 'price_rm']].rename(columns={'price_rm': 'price'})
df_out.to_csv('licence_plate_price_cleaned.csv', index=False)
print(f"\nSauvegardé → licence_plate_price_cleaned.csv")
print("=" * 50)
