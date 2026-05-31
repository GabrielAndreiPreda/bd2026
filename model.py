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
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV

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
# CSV is fully preprocessed by preprocessing.py (log transforms, one-hot, feature
# engineering). Drop the three interpretability-only columns kept in the CSV but
# not used for training (Pays_amts_total = mean×6, Utilization_1 = single month,
# Risk_score = linear combo of features already kept).
X = df.drop(columns=["default_payment", "Pays_amts_total", "Utilization_1", "Risk_score"])
y = df["default_payment"]

print("Features:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Carve a 10% calibration slice from train. The AE never sees X_cal, and
# GridSearchCV never sees it either — it's the held-out set used to fit the
# isotonic probability calibrator and to pick the F1-optimal threshold.
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train, y_train,
    test_size=0.10,
    random_state=42,
    stratify=y_train,
)

print("Train shape:", X_train.shape)
print("Cal shape:", X_cal.shape)
print("Test shape:", X_test.shape)
print("Train default rate:", y_train.mean().round(3))
print("Cal default rate:", y_cal.mean().round(3))
print("Test default rate:", y_test.mean().round(3))

# %%
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
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

# B.4: weight defaulters' reconstruction loss by the inverse class ratio so the
# AE preserves minority-class structure in the latent space (was class-blind before).
ae_sample_weight = np.where(y_train.values == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0)

history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    sample_weight=ae_sample_weight,
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
X_cal_encoded = encoder.predict(X_cal_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

# C.3: concat PAY_0 (strongest single predictor) with the latents so XGBoost
# doesn't have to recover it through the AE bottleneck. PAY_0 is taken from the
# scaled feature matrix to keep it on roughly the same magnitude as the latents.
pay_0_idx = X_train.columns.get_loc("PAY_0")
X_train_features = np.concatenate([X_train_encoded, X_train_scaled[:, pay_0_idx:pay_0_idx+1]], axis=1)
X_cal_features   = np.concatenate([X_cal_encoded,   X_cal_scaled[:, pay_0_idx:pay_0_idx+1]],   axis=1)
X_test_features  = np.concatenate([X_test_encoded,  X_test_scaled[:, pay_0_idx:pay_0_idx+1]],  axis=1)

print("Train feature shape:", X_train_features.shape)
print("Cal feature shape:", X_cal_features.shape)
print("Test feature shape:", X_test_features.shape)

# Move to GPU so XGBoost doesn't have to copy on every call (numpy versions kept
# for sklearn calibration / threshold tuning, which don't accept cupy arrays).
import cupy as cp
X_train_features_gpu = cp.array(X_train_features)

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
    scoring="average_precision",   # PR-AUC: more honest than ROC-AUC for imbalanced data
    cv=5,
    n_jobs=1,     # must be 1 when XGBoost uses GPU — parallelism happens inside XGBoost
    verbose=2
)

grid_search.fit(X_train_features_gpu, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV PR-AUC:", grid_search.best_score_)

xgb_model = grid_search.best_estimator_

# %%
# D.2: isotonic probability calibration on the held-out X_cal slice. Without
# this, scale_pos_weight inflates the raw predict_proba output and Brier score
# is meaningless. cv='prefit' uses xgb_model as already-fitted.
calibrator = CalibratedClassifierCV(xgb_model, method="isotonic", cv="prefit")
calibrator.fit(X_cal_features, y_cal)

# %%
# D.1: pick the F1-optimal threshold on calibrated cal-slice predictions.
# Threshold 0.5 is wrong when scale_pos_weight upweights positives — even after
# calibration, F1-optimal threshold is rarely 0.5 for imbalanced problems.
y_cal_proba = calibrator.predict_proba(X_cal_features)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_cal, y_cal_proba)
f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
best_idx = int(np.argmax(f1s))
tau_star = float(thresholds[best_idx])
print(f"Optimal threshold (max F1 on cal): tau* = {tau_star:.4f}, F1 = {f1s[best_idx]:.4f}")

# %%
y_pred_proba = calibrator.predict_proba(X_test_features)[:, 1]
y_pred = (y_pred_proba >= tau_star).astype(int)

print("PR-AUC (test):", average_precision_score(y_test, y_pred_proba))
print("Brier score (test):", brier_score_loss(y_test, y_pred_proba))
print(f"F1 @ tau*={tau_star:.4f} (test):", f1_score(y_test, y_pred))
print("Accuracy @ tau* (test):", accuracy_score(y_test, y_pred))
print("AUC-ROC (test, for reference):", roc_auc_score(y_test, y_pred_proba))
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
# XGBoost feature importance — latents + raw PAY_0
importances = xgb_model.feature_importances_
feature_labels = [f"Latent Dim {i}" for i in range(encoding_dim)] + ["PAY_0 (raw)"]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_labels, palette="viridis")
plt.title("XGBoost Feature Importance (Latents + PAY_0)")
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
