"""
Feature Engineering — Malaysian License Plate Price Estimator

Features extraites du texte brut de la plaque :

STRUCTURE
  - prefix         : lettres (ex: "WMA", "EV")
  - prefix_len     : longueur du préfixe (1=très rare, 2=rare, 3=commun)
  - num_digits     : nombre de chiffres (1=très cher, 4=commun)
  - total_len      : longueur totale sans espaces
  - digits_value   : valeur numérique du numéro (ex: 4 → 4)

NUMÉROLOGIE / CULTURE
  - lucky_count    : nb de chiffres chanceux (8, 9, 6, 1)
  - lucky_ratio    : proportion de chiffres chanceux
  - unlucky_count  : nb de 4 (malchance en culture chinoise)
  - digit_sum      : somme des chiffres (numérologie)
  - ends_8         : se termine par 8
  - starts_8       : commence par 8
  - contains_888   : contient 888
  - contains_8888  : contient 8888

PATTERNS
  - all_same       : tous les chiffres identiques (1111, 8888)
  - is_palindrome  : palindrome (1221, 8558, 9999)
  - is_abab        : pattern ABAB (1212, 8989)
  - max_repeat     : longueur max d'une séquence répétée
  # is_sequential et is_rev_seq retirés — trop peu de cas dans la base (8 et 4)
  # pour être statistiquement fiables. À réintégrer avec une base plus large.

RARETÉ
  - is_single      : numéro à 1 chiffre (1–9)
  - is_double      : numéro à 2 chiffres (10–99)
  - is_round       : multiple de 100 (100, 200, 500...)
  - is_short_prefix: préfixe ≤ 2 lettres (série ancienne/rare)
"""

import re
import itertools
import pandas as pd


def extract_features(plate: str) -> dict:
    plate = str(plate).strip().upper()
    plate = re.sub(r'\s+', ' ', plate)

    letters = re.sub(r'[^A-Z]', '', plate)
    digits_str = re.sub(r'[^0-9]', '', plate)
    digits = int(digits_str) if digits_str else 0
    num_digits = len(digits_str)
    total_len = len(plate.replace(' ', ''))
    prefix_len = len(letters)

    # ── Répétitions ──────────────────────────────────────
    max_repeat = max(
        (len(list(g)) for _, g in itertools.groupby(digits_str)), default=0
    ) if digits_str else 0

    all_same = int(len(set(digits_str)) == 1 and num_digits > 0)
    is_palindrome = int(digits_str == digits_str[::-1] and num_digits > 1)

    is_abab = int(
        num_digits == 4
        and digits_str[0] == digits_str[2]
        and digits_str[1] == digits_str[3]
        and digits_str[0] != digits_str[1]
    ) if num_digits == 4 else 0

    # ── Séquences ─────────────────────────────────────────
    # Retirées : trop peu d'exemples dans la base (is_sequential=8 cas, is_rev_seq=4 cas)
    # Signal trop faible, risque de bruit. À réintégrer avec une base plus large.

    # ── Numérologie / culture ─────────────────────────────
    lucky_digits = set('8916')
    lucky_count  = sum(1 for d in digits_str if d in lucky_digits)
    lucky_ratio  = lucky_count / num_digits if num_digits > 0 else 0
    unlucky_count = digits_str.count('4')
    digit_sum    = sum(int(d) for d in digits_str) if digits_str else 0

    ends_8   = int(digits_str.endswith('8')) if digits_str else 0
    starts_8 = int(digits_str.startswith('8')) if digits_str else 0
    contains_888  = int('888' in digits_str)
    contains_8888 = int('8888' in digits_str)

    # ── Rareté ────────────────────────────────────────────
    is_single       = int(0 < digits <= 9   and num_digits == 1)
    is_double       = int(10 <= digits <= 99 and num_digits == 2)
    is_round        = int(digits > 0 and digits % 100 == 0)
    is_short_prefix = int(prefix_len <= 2)

    return {
        # Structure
        'prefix_len':      prefix_len,
        'num_digits':      num_digits,
        'total_len':       total_len,
        'digits_value':    digits,
        # Numérologie
        'lucky_count':     lucky_count,
        'lucky_ratio':     lucky_ratio,
        'unlucky_count':   unlucky_count,
        'digit_sum':       digit_sum,
        'ends_8':          ends_8,
        'starts_8':        starts_8,
        'contains_888':    contains_888,
        'contains_8888':   contains_8888,
        # Patterns
        'all_same':        all_same,
        'is_palindrome':   is_palindrome,
        'is_abab':         is_abab,
        'max_repeat':      max_repeat,
        # Rareté
        'is_single':       is_single,
        'is_double':       is_double,
        'is_round':        is_round,
        'is_short_prefix': is_short_prefix,
        # Pour encodage
        '_prefix':         letters[:3],
    }


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Applique extract_features sur tout le dataframe."""
    return pd.DataFrame(df['plate'].apply(extract_features).tolist())


# ── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_plates = [
        'WMA 4',      # single digit, préfixe court → très cher
        'EV 8888',    # 4x8, préfixe court → très cher
        'VPA 6363',   # ABAB → moyen
        'VNT 8558',   # palindrome → moyen+
        'BSP 1221',   # palindrome → moyen+
        'JYV 1234',   # séquentiel → peu valorisé
        'VMW 3133',   # commun → bas
    ]

    rows = []
    for p in test_plates:
        f = extract_features(p)
        f['plate'] = p
        rows.append(f)

    display_cols = ['plate', 'prefix_len', 'num_digits', 'digits_value',
                    'all_same', 'is_palindrome', 'is_abab', 'lucky_count',
                    'unlucky_count', 'is_single', 'is_double', 'max_repeat',
                    'contains_888']
    print(pd.DataFrame(rows)[display_cols].to_string(index=False))
