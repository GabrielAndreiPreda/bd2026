"""
preprocessing.py -- raw UCI CSV to ML-ready CSVs.

Input:  UCI_Credit_Card.csv  (30,000 rows x 25 cols)

Outputs (random_state=42, stratified):
  credit_card_data.csv    full ML-ready dataset
  train.csv               80% train split, original class distribution
  train_balanced.csv      SMOTENC-balanced training split (~50/50)
  test.csv                20% test split, never touched by SMOTE

train_balanced.csv requires imbalanced-learn.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _signed_log1p(s):
    """sign-preserving log; handles negative bill values (~2% of rows)."""
    return np.sign(s) * np.log1p(np.abs(s))


# ---------------------------------------------------------------------------
# Load and clean
# ---------------------------------------------------------------------------

df = pd.read_csv("UCI_Credit_Card.csv")
df.rename(columns={"default.payment.next.month": "default_payment"}, inplace=True)

df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)  # undocumented unknowns -> others
df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)

df.drop("ID", axis=1, inplace=True)
df = df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bills = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
pays_amts = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

df["Pay_delay_mean"] = df[pays].mean(axis=1)
df["Pay_delay_max"]  = df[pays].max(axis=1)
df["Pay_delay_std"]  = df[pays].std(axis=1)

df["Bill_mean"] = df[bills].mean(axis=1)
df["Bill_std"]  = df[bills].std(axis=1)
df["Bill_max"]  = df[bills].max(axis=1)

df["Pays_amts_total"] = df[pays_amts].sum(axis=1)
df["Pays_amts_mean"]  = df[pays_amts].mean(axis=1)

df["Utilization_1"]    = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)
df["Utilization_mean"] = df[bills].mean(axis=1) / (df["LIMIT_BAL"] + 1)

df["Risk_score"] = (
    df["Pay_delay_mean"]
    + df["Utilization_mean"]
    + df["Bill_mean"] / (df["LIMIT_BAL"] + 1)
)

df["n_months_over_limit"] = (
    df[bills].gt(df["LIMIT_BAL"], axis=0).sum(axis=1).astype(np.int8)
)


# ---------------------------------------------------------------------------
# Column selection and transforms
# ---------------------------------------------------------------------------

df = df.drop(columns=pays[1:] + bills + pays_amts)

log_cols = [
    "LIMIT_BAL", "Bill_mean", "Bill_std", "Bill_max",
    "Pays_amts_mean", "Utilization_mean",
]
df[log_cols] = df[log_cols].apply(_signed_log1p)

df = pd.get_dummies(
    df, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=True, dtype=np.float32,
)

df.to_csv("credit_card_data.csv", index=False)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

X = df.drop(columns=["default_payment"])
y = df["default_payment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

X_train.assign(default_payment=y_train.values).to_csv("train.csv", index=False)
X_test.assign(default_payment=y_test.values).to_csv("test.csv", index=False)


# ---------------------------------------------------------------------------
# SMOTENC balancing (preserves one-hot dummy integers during oversampling)
# ---------------------------------------------------------------------------

try:
    from imblearn.over_sampling import SMOTENC

    cat_cols = ["SEX_2", "EDUCATION_2", "EDUCATION_3", "EDUCATION_4", "MARRIAGE_2", "MARRIAGE_3"]
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    smote = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    pd.DataFrame(X_bal, columns=X_train.columns).assign(default_payment=y_bal).to_csv(
        "train_balanced.csv", index=False,
    )

except ImportError:
    print("imbalanced-learn not installed -- skipping train_balanced.csv")
    print("  pip install imbalanced-learn")
