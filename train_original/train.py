"""
License Plate Price — Model Training
Entraîne Gradient Boosting + Random Forest sur log(prix)
Sauvegarde le meilleur modèle dans models/
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, '.')
from features import build_feature_matrix

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('report', exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)

# ── Load ──────────────────────────────────────────────────────────────────────

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print("=" * 55)
print("LICENSE PLATE PRICE — TRAINING")
print("=" * 55)
print(f"Train : {len(train)} | Test : {len(test)}")

# ── Features ──────────────────────────────────────────────────────────────────

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

X_train, le = prepare(train, fit_le=True)
X_test,  _  = prepare(test, le=le)

# Cible : log(prix)
y_train = np.log(train['price'].values)
y_test  = np.log(test['price'].values)

FEATURE_NAMES = list(X_train.columns)

# ── Modèles ───────────────────────────────────────────────────────────────────

models = {
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    ),
}

# ── Entraînement & évaluation ─────────────────────────────────────────────────

results = {}

print()
for name, model in models.items():
    print(f"Entraînement : {name}...")
    model.fit(X_train, y_train)

    log_pred  = model.predict(X_test)
    pred_rm   = np.exp(log_pred)
    true_rm   = np.exp(y_test)

    mae   = mean_absolute_error(true_rm, pred_rm)
    r2    = r2_score(y_test, log_pred)          # R² sur log
    mape  = np.mean(np.abs((true_rm - pred_rm) / true_rm)) * 100

    results[name] = {'model': model, 'pred': pred_rm, 'true': true_rm,
                     'mae': mae, 'r2': r2, 'mape': mape}

    print(f"  MAE  : RM {mae:>10,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.1f}%")
    print()

# ── Meilleur modèle ───────────────────────────────────────────────────────────

best_name = min(results, key=lambda k: results[k]['mae'])
best      = results[best_name]
print(f"Meilleur modèle : {best_name}  (MAE = RM {best['mae']:,.0f})")

# ── Sauvegarde ────────────────────────────────────────────────────────────────

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump({
        'model':         best['model'],
        'label_encoder': le,
        'features':      FEATURE_NAMES,
        'model_name':    best_name,
    }, f)
print("Modèle sauvegardé → models/best_model.pkl")

# ── Figure 1 : Prédit vs Réel ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prédit vs Réel — comparaison des modèles', fontsize=14, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    true_log = np.log10(res['true'])
    pred_log = np.log10(np.clip(res['pred'], 1, None))

    ax.scatter(true_log, pred_log, alpha=0.4, s=18, color='#4C72B0')
    lims = [min(true_log.min(), pred_log.min()) - 0.1,
            max(true_log.max(), pred_log.max()) + 0.1]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Prédiction parfaite')

    ticks = [3, 3.5, 4, 4.5, 5, 5.5, 6]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'RM {10**t:,.0f}' for t in ticks], rotation=25, fontsize=8)
    ax.set_yticklabels([f'RM {10**t:,.0f}' for t in ticks], fontsize=8)
    ax.set_xlabel('Prix réel', fontsize=11)
    ax.set_ylabel('Prix prédit', fontsize=11)
    ax.set_title(f"{name}\nMAE=RM {res['mae']:,.0f}  R²={res['r2']:.3f}  MAPE={res['mape']:.1f}%",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/fig4_pred_vs_real.png', dpi=150, bbox_inches='tight')
plt.savefig('report/fig4_pred_vs_real.png',  dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardée → fig4_pred_vs_real.png")

# ── Figure 2 : Feature importances (meilleur modèle) ─────────────────────────

importances = pd.Series(best['model'].feature_importances_, index=FEATURE_NAMES)
importances = importances.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(importances.index, importances.values,
               color='#4C72B0', edgecolor='white')
ax.set_title(f'Feature Importances — {best_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance relative', fontsize=12)
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.savefig('report/fig5_feature_importance.png',  dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardée → fig5_feature_importance.png")

print("\n" + "=" * 55)
print("RESUME FINAL")
print("=" * 55)
for name, res in results.items():
    print(f"{name:25s}  MAE=RM {res['mae']:>8,.0f}  R²={res['r2']:.4f}  MAPE={res['mape']:.1f}%")
print("=" * 55)
