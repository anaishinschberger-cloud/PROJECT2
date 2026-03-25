"""
License Plate Price — Data Exploration
Génère 3 figures séparées sauvegardées dans results/ et report/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

os.makedirs('results', exist_ok=True)
os.makedirs('report', exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv('database/licence_plate_price.csv', header=0)
df.columns = ['plate', 'price']

def is_valid_price(val):
    if pd.isna(val): return False
    return bool(re.match(r'^RM\s[\d,]+$', str(val).strip()))

df_valid = df[df['price'].apply(is_valid_price)].copy()
df_valid['price_rm'] = df_valid['price'].apply(
    lambda x: float(re.sub(r'[^\d.]', '', x))
)

print(f"Entrées totales : {len(df)}")
print(f"Entrées valides : {len(df_valid)}")
print(f"Prix min : RM {df_valid['price_rm'].min():,.0f}")
print(f"Prix max : RM {df_valid['price_rm'].max():,.0f}")
print(f"Médiane  : RM {df_valid['price_rm'].median():,.0f}")
print(f"Moyenne  : RM {df_valid['price_rm'].mean():,.0f}")

# ── Figure 1 — Qualité des données ───────────────────────────────────────────
# Pie chart : valides vs invalides, avec détail des invalides

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Qualité des données brutes', fontsize=14, fontweight='bold')

# Pie : valides vs invalides
total = len(df)
poa = df['price'].str.strip().str.upper().eq('POA').sum()
bad_plate = (~df['plate'].str.match(r'^[A-Za-z0-9\s\(\)]+$')).sum()
valid = len(df_valid)
other_invalid = total - valid - poa - bad_plate

sizes  = [valid, poa, bad_plate]
labels = [f'Valides\n({valid})', f'Prix POA\n({poa})', f'Plaques\nmal formées\n({bad_plate})']
colors = ['#4C72B0', '#DD8452', '#C44E52']

wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontweight('bold')
axes[0].set_title('Répartition des entrées', fontweight='bold')

# Bar : détail des suppressions lors du nettoyage
categories = ['Prix POA', 'Plaques\nillisibles', 'Prix < RM 1000', 'Doublons']
counts     = [107, 1, 1, 432]
bars = axes[1].bar(categories, counts, color=['#DD8452','#C44E52','#937860','#8172B2'],
                   edgecolor='white', width=0.55)
axes[1].set_title('Entrées supprimées lors du nettoyage', fontweight='bold')
axes[1].set_ylabel('Nombre d\'entrées')
for bar, val in zip(bars, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                 str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)
axes[1].set_ylim(0, 500)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/fig1_data_quality.png', dpi=150, bbox_inches='tight')
plt.savefig('report/fig1_data_quality.png',  dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 sauvegardée → fig1_data_quality.png")

# ── Figure 2 — Distribution des prix (échelle log) ───────────────────────────
# C'est la visualisation la plus informative : montre l'asymétrie et la structure

fig, ax = plt.subplots(figsize=(11, 5))

log_prices = np.log10(df_valid['price_rm'])
ax.hist(log_prices, bins=55, color='#4C72B0', edgecolor='white', alpha=0.88)

med_log = np.log10(df_valid['price_rm'].median())
moy_log = np.log10(df_valid['price_rm'].mean())

ax.axvline(med_log, color='#C44E52', linestyle='--', linewidth=2,
           label=f"Médiane : RM {df_valid['price_rm'].median():,.0f}")
ax.axvline(moy_log, color='#DD8452', linestyle='--', linewidth=2,
           label=f"Moyenne : RM {df_valid['price_rm'].mean():,.0f}")

ticks = [3, 3.5, 4, 4.5, 5, 5.5, 6]
ax.set_xticks(ticks)
ax.set_xticklabels([f'RM {10**t:,.0f}' for t in ticks], fontsize=10)
ax.set_xlabel('Prix (RM) — echelle logarithmique', fontsize=12)
ax.set_ylabel('Nombre de plaques', fontsize=12)
ax.set_title('Distribution des prix des plaques (echelle log)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/fig2_price_distribution.png', dpi=150, bbox_inches='tight')
plt.savefig('report/fig2_price_distribution.png',  dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 sauvegardée → fig2_price_distribution.png")

# ── Figure 3 — Répartition par tranches de prix ───────────────────────────────

bins   = [0, 5000, 10000, 20000, 50000, 100000, float('inf')]
labels_cat = ['< 5K', '5K–10K', '10K–20K', '20K–50K', '50K–100K', '> 100K']
df_valid['price_cat'] = pd.cut(df_valid['price_rm'], bins=bins, labels=labels_cat)
cat_counts = df_valid['price_cat'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
palette = sns.color_palette('Blues_d', len(cat_counts))
bars = ax.bar(cat_counts.index, cat_counts.values, color=palette, edgecolor='white', width=0.6)

for bar, val in zip(bars, cat_counts.values):
    pct = val / len(df_valid) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Répartition des plaques par tranche de prix', fontsize=14, fontweight='bold')
ax.set_xlabel('Tranche de prix (RM)', fontsize=12)
ax.set_ylabel('Nombre de plaques', fontsize=12)
ax.set_ylim(0, cat_counts.max() * 1.2)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/fig3_price_brackets.png', dpi=150, bbox_inches='tight')
plt.savefig('report/fig3_price_brackets.png',  dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 sauvegardée → fig3_price_brackets.png")

print("\nToutes les figures générées.")
