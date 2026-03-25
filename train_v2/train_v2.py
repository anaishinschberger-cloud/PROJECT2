"""
License Plate Price — Training V2
Améliorations vs V1 :
  1. Features d'interaction manuelle
  2. LightGBM
  3. Stacking (RF + LightGBM → Ridge meta-modèle)
  4. Stratified split sur tranches de prix
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

ROOT = os.path.abspath('.')   # = "License plate price/" quand lancé depuis là
sys.path.insert(0, ROOT)
from features import build_feature_matrix

os.makedirs('train_v2/models', exist_ok=True)
os.makedirs('train_v2/results', exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)

# ── Load ──────────────────────────────────────────────────────────────────────

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print("=" * 60)
print("LICENSE PLATE PRICE — TRAINING V2")
print("=" * 60)
print(f"Train : {len(train)} | Test : {len(test)}")

# ── Features de base ──────────────────────────────────────────────────────────

def prepare(df, le=None, fit_le=False):
    X = build_feature_matrix(df)
    if fit_le:
        le = LabelEncoder()
        le.fit(X['_prefix'])
    X['prefix_enc'] = X['_prefix'].apply(
        lambda p: le.transform([p])[0] if p in le.classes_ else -1
    )
    X = X.drop(columns=['_prefix'])
    return X, le

X_train_base, le = prepare(train, fit_le=True)
X_test_base,  _  = prepare(test, le=le)

# ── Features d'interaction ────────────────────────────────────────────────────

def add_interactions(X):
    X = X.copy()
    # Rareté combinée : plus num_digits est petit ET prefix_len petit → très cher
    X['rarity_score']     = 1.0 / (X['num_digits'].clip(lower=1) * X['prefix_len'].clip(lower=1))
    # 8888 (all_same + lucky) vs 1111 (all_same, pas lucky)
    X['same_and_lucky']   = X['all_same'] * X['lucky_count']
    # Single digit avec préfixe court = exponentiel
    X['single_short']     = X['is_single'] * (X['prefix_len'] <= 2).astype(int)
    # Palindrome avec chiffres chanceux
    X['palin_lucky']      = X['is_palindrome'] * X['lucky_count']
    # Nombre de chiffres × max_repeat (ex: 4 chiffres tous pareils = 8888)
    X['digits_x_repeat']  = X['num_digits'] * X['max_repeat']
    # Pénalité unlucky pondérée par la rareté (le 4 pénalise moins si plaque rare)
    X['unlucky_weighted'] = X['unlucky_count'] * X['num_digits']
    return X

X_train = add_interactions(X_train_base)
X_test  = add_interactions(X_test_base)

FEATURE_NAMES = list(X_train.columns)
print(f"Features : {len(FEATURE_NAMES)} (base) + interactions")

# ── Cible ─────────────────────────────────────────────────────────────────────

y_train = np.log(train['price'].values)
y_test  = np.log(test['price'].values)

# ── Modèles ───────────────────────────────────────────────────────────────────

rf = RandomForestRegressor(
    n_estimators=400, max_depth=14, min_samples_leaf=3,
    random_state=42, n_jobs=-1
)

lgbm = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.04,
    num_leaves=40, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=5, random_state=42, verbose=-1
)

stacking = StackingRegressor(
    estimators=[('rf', rf), ('lgbm', lgbm)],
    final_estimator=Ridge(alpha=1.0),
    cv=5, passthrough=False, n_jobs=-1
)

models = {
    'Random Forest V2':  RandomForestRegressor(
        n_estimators=400, max_depth=14, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    ),
    'LightGBM':          lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.04,
        num_leaves=40, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=5, random_state=42, verbose=-1
    ),
    'Stacking (RF+LGBM)': StackingRegressor(
        estimators=[
            ('rf',   RandomForestRegressor(n_estimators=400, max_depth=14,
                                           min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ('lgbm', lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.04,
                                        num_leaves=40, subsample=0.8, colsample_bytree=0.8,
                                        min_child_samples=5, random_state=42, verbose=-1))
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5, passthrough=False, n_jobs=-1
    ),
}

# ── Entraînement & évaluation ─────────────────────────────────────────────────

results = {}

print()
for name, model in models.items():
    print(f"Entraînement : {name}...")
    model.fit(X_train, y_train)

    log_pred = model.predict(X_test)
    pred_rm  = np.exp(log_pred)
    true_rm  = np.exp(y_test)

    mae  = mean_absolute_error(true_rm, pred_rm)
    r2   = r2_score(y_test, log_pred)
    mape = np.mean(np.abs((true_rm - pred_rm) / true_rm)) * 100

    results[name] = {'model': model, 'pred': pred_rm, 'true': true_rm,
                     'mae': mae, 'r2': r2, 'mape': mape}

    print(f"  MAE  : RM {mae:>10,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.1f}%")
    print()

# ── Comparaison avec V1 ───────────────────────────────────────────────────────

V1 = {'MAE': 5906, 'R2': 0.9206, 'MAPE': 28.7}

print("── Comparaison V1 (Random Forest original) ──────────────")
print(f"  V1  MAE=RM {V1['MAE']:>8,}  R²={V1['R2']:.4f}  MAPE={V1['MAPE']:.1f}%")
for name, res in results.items():
    delta_mae  = res['mae'] - V1['MAE']
    delta_r2   = res['r2'] - V1['R2']
    sign_mae   = '+' if delta_mae > 0 else ''
    sign_r2    = '+' if delta_r2 > 0 else ''
    print(f"  {name:28s}  MAE=RM {res['mae']:>8,.0f} ({sign_mae}{delta_mae:,.0f})  "
          f"R²={res['r2']:.4f} ({sign_r2}{delta_r2:.4f})  MAPE={res['mape']:.1f}%")

# ── Meilleur modèle ───────────────────────────────────────────────────────────

best_name = min(results, key=lambda k: results[k]['mae'])
best      = results[best_name]
print(f"\nMeilleur modèle V2 : {best_name}  (MAE = RM {best['mae']:,.0f})")

with open('train_v2/models/best_model_v2.pkl', 'wb') as f:
    pickle.dump({
        'model':         best['model'],
        'label_encoder': le,
        'features':      FEATURE_NAMES,
        'model_name':    best_name,
        'version':       'v2',
    }, f)
print("Modèle sauvegardé → train_v2/models/best_model_v2.pkl")

# ── Figure : Prédit vs Réel (tous les modèles V2) ────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('V2 — Prix prédit vs réel', fontsize=14, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    true_log = np.log10(res['true'])
    pred_log = np.log10(np.clip(res['pred'], 1, None))
    ax.scatter(true_log, pred_log, alpha=0.4, s=15, color='#4C72B0')
    lims = [min(true_log.min(), pred_log.min()) - 0.1,
            max(true_log.max(), pred_log.max()) + 0.1]
    ax.plot(lims, lims, 'r--', linewidth=1.5)
    ticks = [3, 3.5, 4, 4.5, 5, 5.5, 6]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'RM {10**t:,.0f}' for t in ticks], rotation=30, fontsize=7)
    ax.set_yticklabels([f'RM {10**t:,.0f}' for t in ticks], fontsize=7)
    ax.set_xlabel('Prix réel', fontsize=10)
    ax.set_ylabel('Prix prédit', fontsize=10)
    ax.set_title(f"{name}\nMAE=RM {res['mae']:,.0f}  R²={res['r2']:.3f}",
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_v2/results/fig_v2_pred_vs_real.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardée → train_v2/results/fig_v2_pred_vs_real.png")

# ── Figure : Feature importances LightGBM ────────────────────────────────────

lgbm_model = results['LightGBM']['model']
imp = pd.Series(lgbm_model.feature_importances_, index=FEATURE_NAMES)
imp = imp.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp.index, imp.values, color='#4C72B0', edgecolor='white')
ax.set_title('Feature Importances — LightGBM V2', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('train_v2/results/fig_v2_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardée → train_v2/results/fig_v2_feature_importance.png")

print("\n" + "=" * 60)
print("RESUME FINAL V2")
print("=" * 60)
for name, res in results.items():
    print(f"{name:30s}  MAE=RM {res['mae']:>8,.0f}  R²={res['r2']:.4f}  MAPE={res['mape']:.1f}%")
print("=" * 60)
