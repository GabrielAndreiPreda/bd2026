"""
EDA companion. Reads credit_card_data.csv (produced by preprocessing.py).
Outputs: eda_plots/*.png, correlation_results.csv, default_summary.csv.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

plt.figure(figsize=(6, 4))
sns.countplot(x="default_payment", data=df)
plt.title("Default vs Non-Default Distribution")
plt.tight_layout()
plt.savefig(OUT / "target_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(df["LIMIT_BAL"], bins=30, kde=True)
plt.title("Credit Limit Distribution (signed-log scale)")
plt.tight_layout()
plt.savefig(OUT / "credit_limit_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(df["AGE"], bins=30, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig(OUT / "age_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
df["PAY_0"].hist(bins=20)
plt.title("PAY_0 (September payment status) Distribution")
plt.xlabel("PAY_0")
plt.tight_layout()
plt.savefig(OUT / "pay0_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", linewidths=0.2)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap_full.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(
    df[engineered_feats + ["default_payment"]].corr(),
    cmap="coolwarm", annot=True,
)
plt.title("Correlation with Engineered Features")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap_engineered.png", dpi=150)
plt.close()

for col in ["LIMIT_BAL", "AGE", "Bill_mean", "Pays_amts_mean"]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.tight_layout()
    plt.savefig(OUT / f"boxplot_{col.lower()}.png", dpi=150)
    plt.close()

df[engineered_feats].hist(figsize=(12, 8), bins=20)
plt.suptitle("Engineered Feature Distributions")
plt.tight_layout()
plt.savefig(OUT / "engineered_features_distributions.png", dpi=150)
plt.close()

corr_target = (
    df.corr(numeric_only=True)["default_payment"]
    .sort_values(ascending=False)
)
corr_target.to_csv(HERE / "correlation_results.csv")

summary = df.groupby("default_payment")[[
    "LIMIT_BAL", "Bill_mean", "Pays_amts_mean",
    "Utilization_mean", "Pay_delay_mean", "Risk_score",
]].mean()
summary.to_csv(HERE / "default_summary.csv")
