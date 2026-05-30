# %%
# !pip install pandas numpy matplotlib seaborn scikit-learn

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# %%
# from google.colab import files

# uploaded = files.upload()

# %%
df = pd.read_csv("UCI_Credit_Card.csv")

# %%
print(df.head())
print(df.shape)
print(df.info())
print("missing values")
print(df.isnull().sum())
print("duplicates rows", df.duplicated().sum())

# %%
#EDA: "default.payment.next.month" distribution
plt.figure(figsize=(6,4))
sns.countplot(x="default.payment.next.month", data=df)
plt.title("Default vs Non-Default Distribution")

plt.show()

# %%
#EDA: "Credit Limit" distribution

plt.figure(figsize=(6,4))
sns.histplot(df["LIMIT_BAL"], bins=30, kde=True)
plt.title("Credit Limit Distribution")
plt.show()

# %%
#EDA: AGE DISTRIBUTION (for profiling/ demographics)
plt.figure(figsize=(6,4))
sns.histplot(df["AGE"], bins=30, kde=True)
plt.title("Age Distribution")

plt.show()

# %%
#EDA: correlation heatmap
plt.figure(figsize=(18,12))
corr = df.corr()

sns.heatmap(corr, cmap="coolwarm", linewidths=0.2)
plt.title("Feature Correlation Heatmap")

plt.show()

# %%
#EDA: PAYMENT PATTERNS
pays = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

df[pays].hist(figsize=(12,8), bins=20)
plt.suptitle("Payment status Distribution over time")
plt.show()

# %%
#EDA: billing amounts distribution
bills = [
    "BILL_AMT1","BILL_AMT2","BILL_AMT3",
    "BILL_AMT4","BILL_AMT5","BILL_AMT6"
]

df[bills].hist(figsize=(12,8), bins=20)
plt.suptitle("Billing Amount Distributions")
plt.show()

# %%
#EDA: outliers
feats = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]

for col in feats:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()

# %%
df.rename(columns={
    "default.payment.next.month": "default_payment"
}, inplace=True)



# %%
print(df.columns)

# %% [markdown]
# The original dataset had:
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# but EDUCATION 4,5,6 can all fall under 1 category
# 
# 
# 

# %%
df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)

# %%
df.drop("ID", axis=1, inplace=True)
df = df.drop_duplicates().reset_index(drop=True)

# %%
pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bills = [
    "BILL_AMT1","BILL_AMT2","BILL_AMT3",
    "BILL_AMT4","BILL_AMT5","BILL_AMT6"
]
pays_amts = [
    "PAY_AMT1","PAY_AMT2","PAY_AMT3",
    "PAY_AMT4","PAY_AMT5","PAY_AMT6"
]

# %%
#for pays...payment history features
df["Pay_delay_mean"] = df[pays].mean(axis=1)
df["Pay_delay_max"] = df[pays].max(axis=1)
df["Pay_delay_std"] = df[pays].std(axis=1)


# %%
#for bills...billing behavior feats
df["Bill_mean"] = df[bills].mean(axis=1)
df["Bill_std"] = df[bills].std(axis=1)
df["Bill_max"] = df[bills].max(axis=1)

# %%
#for pay_amts...payment amount feats
df["Pays_amts_total"] = df[pays_amts].sum(axis=1)
df["Pays_amts_mean"] = df[pays_amts].mean(axis=1)

# %%
#feats for credit card utilization

df["Utilization_1"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)

df["Utilization_mean"] = df[bills].mean(axis=1) / (df["LIMIT_BAL"] + 1)

# %%
#intended as a feature that combines payment delay, utilization and billing behavior to serve as a risk indicator (easy for interpretation)
df["Risk_score"] = (
    df["Pay_delay_mean"] +
    df["Utilization_mean"] +
    df["Bill_mean"] / (df["LIMIT_BAL"] + 1)
)

# %%
print("dataset shape final", df.shape)
print(df.head())

# %%
engineered_feats = [
    "Pay_delay_mean",
    "Pay_delay_max",
    "Bill_mean",
    "Pays_amts_total",
    "Utilization_mean",
    "Risk_score"
]

df[engineered_feats].describe()

# %%
#EDA after feat engineering
df[engineered_feats].hist(figsize=(12,8), bins=20)

plt.suptitle("Engineered Feature Distributions")
plt.show()

# %%
#EDA: engineered feats + default_payment correlation
plt.figure(figsize=(12,10))

sns.heatmap(df[engineered_feats + ["default_payment"]].corr(),
            cmap="coolwarm",
            annot=True)

plt.title("Correlation with Engineered Features")
plt.show()

# %%
target_counts = df["default_payment"].value_counts()

target_percent = df["default_payment"].value_counts(normalize=True) * 100

print("Target Counts:")
print(target_counts)

print("\nTarget Percentages:")
print(target_percent)

# %%
pay_delay_analysis = df.groupby("PAY_0")["default_payment"].mean() * 100

print(pay_delay_analysis.sort_index())

# %%
corr_target = (
    df.corr(numeric_only=True)["default_payment"]
    .sort_values(ascending=False)
)

print(corr_target)

# %%

print("Top Positive Correlations")
print(corr_target.head(10))

print("\nTop Negative Correlations")
print(corr_target.tail(10))

# %%
default_util = df.groupby("default_payment")["Utilization_mean"].mean()

print(default_util)

# %%
risk_analysis = df.groupby("default_payment")["Risk_score"].mean()

print(risk_analysis)

# %%
summary = df.groupby("default_payment")[[
    "LIMIT_BAL",
    "Bill_mean",
    "Pays_amts_mean",
    "Utilization_mean",
    "Pay_delay_mean",
    "Risk_score"
]].mean()

print(summary)

# %%
for col in ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"]:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"{col}: {len(outliers)} outliers")

# %%

engineered_corr = (
    df[engineered_feats + ["default_payment"]]
    .corr()["default_payment"]
    .sort_values(ascending=False)
)

print(engineered_corr)

# %%
#saving EDA summary/corr
corr_target.to_csv("correlation_results.csv")

summary.to_csv("default_summary.csv")

# %% [markdown]
# Scaling should be applied later on in modeling by Member(s) 2 and 3 (to avoid data leakages)

# %%
df.to_csv("credit_card_dataset.csv", index=False)

# %%


# %%


# %%



