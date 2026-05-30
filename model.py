# %%
!pip install xgboost  # tensorflow and sklearn are pre-installed on Kaggle

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("GPUs available:", tf.config.list_physical_devices("GPU"))

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# %%
df = pd.read_csv("/kaggle/input/datasets/gabrielpredaz/creditcarddata/credit_card_data.csv")

print(df.shape)
print(df.head())

# %%
bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
df["n_months_over_limit"] = df[bill_cols].gt(df["LIMIT_BAL"], axis=0).sum(axis=1).astype(np.int8)

# %%
# Keep: demographics, PAY_0 (strongest single predictor), engineered summaries
# Drop: raw temporal months (PAY_2-6, all BILL/PAY_AMTs) + redundant engineered features
cols_to_drop = [
    # Raw temporal — replaced by engineered summaries
    "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    # Redundant engineered: Pays_amts_total = mean×6, Utilization_1 = single month,
    # Risk_score = linear combo of features already kept
    "Pays_amts_total", "Utilization_1", "Risk_score",
]
df.drop(columns=cols_to_drop, inplace=True)
print("Shape after column selection:", df.shape)

# %%
# Signed-log on heavy-tailed monetary features so MSE reconstruction isn't dominated
# by rare extreme rows (PAY_AMT2 hits z=72.9 under plain StandardScaler).
def signed_log1p(s):
    return np.sign(s) * np.log1p(np.abs(s))

log_cols = ["LIMIT_BAL", "Bill_mean", "Bill_std", "Bill_max", "Pays_amts_mean", "Utilization_mean"]
df[log_cols] = df[log_cols].apply(signed_log1p)

# %%
# One-hot nominal categoricals — MSE on ordinal-coded SEX/EDUCATION/MARRIAGE
# treats them as continuous distances, which is meaningless.
df = pd.get_dummies(
    df,
    columns=["SEX", "EDUCATION", "MARRIAGE"],
    drop_first=True,
    dtype=np.float32,
)
print("Features:", df.drop("default_payment", axis=1).columns.tolist())

# %%
X = df.drop("default_payment", axis=1)
y = df["default_payment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train default rate:", y_train.mean().round(3))
print("Test default rate:", y_test.mean().round(3))

# %%
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Max |z| per feature (should be roughly 5–8 post-log; >20 means a feature still has fat tails):")
print(pd.DataFrame(X_train_scaled, columns=X_train.columns).abs().max().sort_values(ascending=False).round(2))

# %%
# Autoencoder architecture: 18 -> 24 -> 16 -> 10 -> 16 -> 24 -> 18
# Dropout added to encoder layers to force more robust latent representations
input_dim = X_train_scaled.shape[1]
encoding_dim = 10

inputs = Input(shape=(input_dim,))

# Encoder
x = Dense(24, activation="relu")(inputs)
x = Dropout(0.2)(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.2)(x)
encoded = Dense(encoding_dim, activation="relu")(x)

# Decoder
x = Dense(16, activation="relu")(encoded)
x = Dense(24, activation="relu")(x)
decoded = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inputs, decoded, name="autoencoder")
encoder = Model(inputs, encoded, name="encoder")

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# %%
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# %%
# Reconstruction loss curve
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Autoencoder Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Per-feature reconstruction MSE — diagnoses whether the autoencoder is spending
# its capacity on a handful of high-variance features at the expense of the rest.
recon = autoencoder.predict(X_train_scaled, verbose=0)
per_col_mse = ((X_train_scaled - recon) ** 2).mean(axis=0)
per_col_mse = pd.Series(per_col_mse, index=X_train.columns).sort_values()

plt.figure(figsize=(8, max(5, len(per_col_mse) * 0.25)))
sns.barplot(x=per_col_mse.values, y=per_col_mse.index, palette="viridis")
plt.title("Per-Feature Reconstruction MSE (train)")
plt.xlabel("MSE")
plt.tight_layout()
plt.show()

# %%
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

print("Encoded train shape:", X_train_encoded.shape)
print("Encoded test shape:", X_test_encoded.shape)

# Move to GPU so XGBoost doesn't have to copy on every call
import cupy as cp
X_train_encoded = cp.array(X_train_encoded)
X_test_encoded = cp.array(X_test_encoded)

# %%
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

param_grid = {
    "n_estimators":     [200, 400],
    "max_depth":        [4, 6],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 0.9],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
    "gamma":            [0, 0.1],
}

base_xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    device="cuda",      # GPU acceleration
    tree_method="hist"  # required for GPU in XGBoost >= 2.0
)

grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=1,     # must be 1 when XGBoost uses GPU — parallelism happens inside XGBoost
    verbose=2
)

grid_search.fit(X_train_encoded, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV AUC-ROC:", grid_search.best_score_)

xgb_model = grid_search.best_estimator_

# %%
y_pred = xgb_model.predict(X_test_encoded)
y_pred_proba = xgb_model.predict_proba(X_test_encoded)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
print()
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# %%
# Confusion matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# %%
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.title("ROC Curve — Autoencoder + XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# XGBoost feature importance on encoded dimensions
importances = xgb_model.feature_importances_
latent_labels = [f"Latent Dim {i}" for i in range(encoding_dim)]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=latent_labels, palette="viridis")
plt.title("XGBoost Feature Importance (Encoded Dimensions)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# %%
# Mean absolute Jacobian: how much each raw input feature drives each bottleneck neuron
# Computed on a subset of training data for efficiency
feature_names = X.columns.tolist()

n_samples = min(3000, len(X_train_scaled))
X_jac = X_train_scaled[:n_samples]

all_jac = []
jac_batch = 256

for i in range(0, n_samples, jac_batch):
    x_batch = tf.constant(X_jac[i:i+jac_batch], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        enc_out = encoder(x_batch, training=False)
    jac = tape.jacobian(enc_out, x_batch).numpy()  # (n, 10, n, 34)
    n = jac.shape[0]
    # Extract per-sample diagonal: jac[i, :, i, :] -> (n, 10, 34)
    jac = jac[np.arange(n), :, np.arange(n), :]
    all_jac.append(np.abs(jac))

mean_jac = np.mean(np.concatenate(all_jac, axis=0), axis=0)  # (10, 34)

plt.figure(figsize=(18, 6))
sns.heatmap(
    mean_jac,
    xticklabels=feature_names,
    yticklabels=[f"Latent Dim {i}" for i in range(encoding_dim)],
    cmap="YlOrRd",
    annot=False,
    linewidths=0.3
)
plt.title("Mean Absolute Jacobian: Raw Feature → Bottleneck Neuron")
plt.xlabel("Input Feature")
plt.ylabel("Bottleneck Neuron")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %%
# Top 5 contributing features per bottleneck neuron
print("Top 5 input features per bottleneck neuron:\n")
for dim in range(encoding_dim):
    top_idx = np.argsort(mean_jac[dim])[::-1][:5]
    top = [(feature_names[j], round(float(mean_jac[dim][j]), 4)) for j in top_idx]
    print(f"  Latent Dim {dim}: {top}")
