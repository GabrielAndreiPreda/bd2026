"""
eda.py — EDA companion for the credit card default prediction project.

Reads the ML-ready credit_card_data.csv produced by preprocessing.py.
Run any time you want to re-inspect distributions, correlations, and group
statistics. Does not modify the CSV or any model inputs.

Outputs: eda_plots/*.png, correlation_results.csv, default_summary.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
OUT  = HERE / "eda_plots"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(HERE / "credit_card_data.csv")

engineered_feats = [
    "Pay_delay_mean", "Pay_delay_max",
    "Bill_mean", "Pays_amts_mean",
    "Utilization_mean", "Risk_score",
]

# ── Target distribution ────────────────────────────────────────────────────────
target_counts = df["default_payment"].value_counts()
target_percent = df["default_payment"].value_counts(normalize=True) * 100
print("Target Counts:\n", target_counts)
print("\nTarget Percentages:\n", target_percent)

plt.figure(figsize=(6, 4))
sns.countplot(x="default_payment", data=df)
plt.title("Default vs Non-Default Distribution")
plt.tight_layout()
plt.savefig(OUT / "target_distribution.png", dpi=150)
plt.close()

# ── Credit limit distribution (signed-log scale after preprocessing) ──────────
plt.figure(figsize=(6, 4))
sns.histplot(df["LIMIT_BAL"], bins=30, kde=True)
plt.title("Credit Limit Distribution (signed-log scale)")
plt.tight_layout()
plt.savefig(OUT / "credit_limit_distribution.png", dpi=150)
plt.close()

# ── Age distribution ──────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.histplot(df["AGE"], bins=30, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig(OUT / "age_distribution.png", dpi=150)
plt.close()

# ── Full correlation heatmap ──────────────────────────────────────────────────
plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", linewidths=0.2)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap_full.png", dpi=150)
plt.close()

# ── PAY_0 distribution ────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
df["PAY_0"].hist(bins=20)
plt.title("PAY_0 (September payment status) Distribution")
plt.xlabel("PAY_0")
plt.tight_layout()
plt.savefig(OUT / "pay0_distribution.png", dpi=150)
plt.close()

# ── Outlier boxplots ──────────────────────────────────────────────────────────
for col in ["LIMIT_BAL", "AGE", "Bill_mean", "Pays_amts_mean"]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.tight_layout()
    plt.savefig(OUT / f"boxplot_{col.lower()}.png", dpi=150)
    plt.close()

# ── Engineered feature distributions ─────────────────────────────────────────
df[engineered_feats].hist(figsize=(12, 8), bins=20)
plt.suptitle("Engineered Feature Distributions")
plt.tight_layout()
plt.savefig(OUT / "engineered_features_distributions.png", dpi=150)
plt.close()

# ── Engineered features correlation with target ───────────────────────────────
plt.figure(figsize=(12, 10))
sns.heatmap(
    df[engineered_feats + ["default_payment"]].corr(),
    cmap="coolwarm", annot=True,
)
plt.title("Correlation with Engineered Features")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap_engineered.png", dpi=150)
plt.close()

# ── All-feature correlation with target ───────────────────────────────────────
corr_target = (
    df.corr(numeric_only=True)["default_payment"]
    .sort_values(ascending=False)
)
print("\nTop Positive Correlations:")
print(corr_target.head(10))
print("\nTop Negative Correlations:")
print(corr_target.tail(10))

# ── Default rate by most recent payment delay ─────────────────────────────────
pay_delay_analysis = df.groupby("PAY_0")["default_payment"].mean() * 100
print("\nDefault rate % by PAY_0:")
print(pay_delay_analysis.sort_index())

# ── Group summary by default status ──────────────────────────────────────────
summary = df.groupby("default_payment")[[
    "LIMIT_BAL", "Bill_mean", "Pays_amts_mean",
    "Utilization_mean", "Pay_delay_mean", "Risk_score",
]].mean()
print("\nGroup summary by default_payment:")
print(summary)

# ── Utilization and risk by default status ────────────────────────────────────
print("\nMean Utilization_mean by default_payment:")
print(df.groupby("default_payment")["Utilization_mean"].mean())

print("\nMean Risk_score by default_payment:")
print(df.groupby("default_payment")["Risk_score"].mean())

# ── IQR outlier counts ────────────────────────────────────────────────────────
for col in ["LIMIT_BAL", "Bill_mean", "Pays_amts_mean"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    print(f"{col}: {n_out} outliers")

# ── Save EDA outputs ──────────────────────────────────────────────────────────
corr_target.to_csv(HERE / "correlation_results.csv")
summary.to_csv(HERE / "default_summary.csv")
print(f"\nSaved correlation_results.csv and default_summary.csv")
print(f"Saved {len(list(OUT.glob('*.png')))} plots to {OUT}/")
