"""
preprocessing.py — raw UCI CSV → ML-ready CSVs for the full team

Input:  UCI_Credit_Card.csv  (original UCI Taiwan dataset, 30,000 rows × 25 cols)

Outputs (all use random_state=42, stratified on default_payment):
  credit_card_data.csv    — full ML-ready dataset (~29,965 rows), for custom splitting
  train.csv               — 80% training split, original class distribution
  train_balanced.csv      — same training split with SMOTENC applied (~50/50 balance)
  test.csv                — 20% test split, original distribution, never touched by SMOTE

Runs all data hygiene, feature engineering, column selection, log-transforms,
and one-hot encoding so downstream scripts need no further feature work.

train_balanced.csv requires imbalanced-learn (pip install imbalanced-learn).
If not available the script still produces the other three files.

For EDA (distributions, heatmaps, group analyses) see eda.py.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _signed_log1p(s):
    """Log transform that preserves sign; handles negative bill values (refunds ~2% of rows)."""
    return np.sign(s) * np.log1p(np.abs(s))


df = pd.read_csv("UCI_Credit_Card.csv")

# ── Data hygiene ───────────────────────────────────────────────────────────────
df.rename(columns={"default.payment.next.month": "default_payment"}, inplace=True)

df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)  # undocumented unknowns → others
df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)             # 54 undocumented zeros → others

df.drop("ID", axis=1, inplace=True)
df = df.drop_duplicates().reset_index(drop=True)          # removes 35 exact duplicate rows

# ── Feature engineering ────────────────────────────────────────────────────────
pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bills = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
pays_amts = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

df["Pay_delay_mean"] = df[pays].mean(axis=1)
df["Pay_delay_max"] = df[pays].max(axis=1)
df["Pay_delay_std"] = df[pays].std(axis=1)

df["Bill_mean"] = df[bills].mean(axis=1)
df["Bill_std"] = df[bills].std(axis=1)
df["Bill_max"] = df[bills].max(axis=1)

df["Pays_amts_total"] = df[pays_amts].sum(axis=1)
df["Pays_amts_mean"] = df[pays_amts].mean(axis=1)

df["Utilization_1"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)
df["Utilization_mean"] = df[bills].mean(axis=1) / (df["LIMIT_BAL"] + 1)

# Linear combo kept in CSV for interpretability; downstream drops it before training
df["Risk_score"] = (
    df["Pay_delay_mean"]
    + df["Utilization_mean"]
    + df["Bill_mean"] / (df["LIMIT_BAL"] + 1)
)

# Months where outstanding bill exceeded the credit limit (0–6 count)
# Must be computed before BILL_AMTs are dropped below
df["n_months_over_limit"] = (
    df[bills].gt(df["LIMIT_BAL"], axis=0).sum(axis=1).astype(np.int8)
)

# ── Column selection ───────────────────────────────────────────────────────────
# Drop raw temporal columns; engineered summaries above replace them.
# PAY_0 is kept — it is the single strongest predictor and is used raw downstream.
# Pays_amts_total, Utilization_1, Risk_score stay in the CSV for interpretability
# even though model.py and benchmark.py drop them before training.
df = df.drop(columns=pays[1:] + bills + pays_amts)

# ── Log transform heavy-tailed monetary features ───────────────────────────────
# Without this, extreme rows hit z > 70 under StandardScaler and dominate MSE loss.
log_cols = [
    "LIMIT_BAL", "Bill_mean", "Bill_std", "Bill_max",
    "Pays_amts_mean", "Utilization_mean",
]
df[log_cols] = df[log_cols].apply(_signed_log1p)

# ── One-hot nominal categoricals ───────────────────────────────────────────────
# MSE on ordinal-coded SEX/EDUCATION/MARRIAGE is meaningless for autoencoder training.
df = pd.get_dummies(
    df, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=True, dtype=np.float32
)

print("Output shape:", df.shape)
print("Columns:", df.columns.tolist())
df.to_csv("credit_card_data.csv", index=False)
print("Saved credit_card_data.csv")

# ── Train / test split ────────────────────────────────────────────────────────
X = df.drop(columns=["default_payment"])
y = df["default_payment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.assign(default_payment=y_train.values)
test_df  = X_test.assign(default_payment=y_test.values)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
print(f"Saved train.csv ({len(train_df)} rows, default rate {y_train.mean():.3f})")
print(f"Saved test.csv  ({len(test_df)} rows, default rate {y_test.mean():.3f})")

# ── SMOTE balancing (requires imbalanced-learn) ───────────────────────────────
# SMOTENC is used instead of plain SMOTE because the one-hot dummy columns
# (SEX_2, EDUCATION_*, MARRIAGE_*) are binary categoricals — interpolating them
# as continuous values would produce fractional dummies (e.g. SEX_2 = 0.3).
# SMOTENC keeps categorical columns at valid integer values during oversampling.
try:
    from imblearn.over_sampling import SMOTENC

    cat_cols = ["SEX_2", "EDUCATION_2", "EDUCATION_3", "EDUCATION_4", "MARRIAGE_2", "MARRIAGE_3"]
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    smote = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    train_bal_df = pd.DataFrame(X_bal, columns=X_train.columns)
    train_bal_df["default_payment"] = y_bal
    train_bal_df.to_csv("train_balanced.csv", index=False)
    print(f"Saved train_balanced.csv ({len(train_bal_df)} rows, default rate {y_bal.mean():.3f})")

except ImportError:
    print("imbalanced-learn not installed — skipping train_balanced.csv")
    print("  Install with: pip install imbalanced-learn")
