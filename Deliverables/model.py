"""
AE + XGBoost credit-default classifier.

Pipeline:
  1. StandardScaler on 18 preprocessed features.
  2. Autoencoder (18 -> 24 -> 16 -> 10 -> 16 -> 24 -> 18) with dropout 0.2
     and class-weighted reconstruction loss.
  3. Concat 10 latents with raw scaled PAY_0.
  4. XGBoost GridSearchCV on PR-AUC (128 cells, 5-fold CV).
  5. Isotonic probability calibration on a held-out 10% slice.
  6. F1-optimal threshold tau* on the same slice.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
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

from xgboost import XGBClassifier

df = pd.read_csv("/kaggle/input/datasets/gabrielpredaz/creditcarddata/credit_card_data.csv")

X = df.drop(columns=["default_payment", "Pays_amts_total", "Utilization_1", "Risk_score"])
y = df["default_payment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42, stratify=y_train,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled   = scaler.transform(X_cal)
X_test_scaled  = scaler.transform(X_test)

input_dim = X_train_scaled.shape[1]
encoding_dim = 10

inputs = Input(shape=(input_dim,))
x = Dense(24, activation="relu")(inputs)
x = Dropout(0.2)(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.2)(x)
encoded = Dense(encoding_dim, activation="relu")(x)
x = Dense(16, activation="relu")(encoded)
x = Dense(24, activation="relu")(x)
decoded = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inputs, decoded, name="autoencoder")
encoder = Model(inputs, encoded, name="encoder")
autoencoder.compile(optimizer="adam", loss="mse")

ae_sample_weight = np.where(
    y_train.values == 1,
    (y_train == 0).sum() / (y_train == 1).sum(),
    1.0,
)

history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    sample_weight=ae_sample_weight,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    verbose=1,
)

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Autoencoder Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()

recon = autoencoder.predict(X_train_scaled, verbose=0)
per_col_mse = pd.Series(
    ((X_train_scaled - recon) ** 2).mean(axis=0),
    index=X_train.columns,
).sort_values()

plt.figure(figsize=(8, max(5, len(per_col_mse) * 0.25)))
sns.barplot(x=per_col_mse.values, y=per_col_mse.index, palette="viridis")
plt.title("Per-Feature Reconstruction MSE (train)")
plt.xlabel("MSE")
plt.tight_layout()
plt.show()

X_train_encoded = encoder.predict(X_train_scaled)
X_cal_encoded   = encoder.predict(X_cal_scaled)
X_test_encoded  = encoder.predict(X_test_scaled)

pay_0_idx = X_train.columns.get_loc("PAY_0")
X_train_features = np.concatenate([X_train_encoded, X_train_scaled[:, pay_0_idx:pay_0_idx + 1]], axis=1)
X_cal_features   = np.concatenate([X_cal_encoded,   X_cal_scaled[:, pay_0_idx:pay_0_idx + 1]],   axis=1)
X_test_features  = np.concatenate([X_test_encoded,  X_test_scaled[:, pay_0_idx:pay_0_idx + 1]],  axis=1)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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
    device="cuda",
    tree_method="hist",
)

grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    scoring="average_precision",
    cv=5,
    n_jobs=1,
    verbose=2,
)
grid_search.fit(X_train_features, y_train)
xgb_model = grid_search.best_estimator_

calibrator = CalibratedClassifierCV(xgb_model, method="isotonic", cv="prefit")
calibrator.fit(X_cal_features, y_cal)

y_cal_proba = calibrator.predict_proba(X_cal_features)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_cal, y_cal_proba)
f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
tau_star = float(thresholds[int(np.argmax(f1s))])

y_pred_proba = calibrator.predict_proba(X_test_features)[:, 1]
y_pred = (y_pred_proba >= tau_star).astype(int)

print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Default", "Default"],
    yticklabels=["No Default", "Default"],
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", label=f"AUC = {roc_auc_score(y_test, y_pred_proba):.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.title("ROC Curve -- Autoencoder + XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

feature_labels = [f"Latent Dim {i}" for i in range(encoding_dim)] + ["PAY_0 (raw)"]
plt.figure(figsize=(8, 5))
sns.barplot(x=xgb_model.feature_importances_, y=feature_labels, palette="viridis")
plt.title("XGBoost Feature Importance (Latents + PAY_0)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

n_samples = min(3000, len(X_train_scaled))
X_jac = X_train_scaled[:n_samples]
all_jac = []
jac_batch = 256

for i in range(0, n_samples, jac_batch):
    x_batch = tf.constant(X_jac[i:i + jac_batch], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        enc_out = encoder(x_batch, training=False)
    jac = tape.jacobian(enc_out, x_batch).numpy()
    n = jac.shape[0]
    jac = jac[np.arange(n), :, np.arange(n), :]
    all_jac.append(np.abs(jac))

mean_jac = np.mean(np.concatenate(all_jac, axis=0), axis=0)

plt.figure(figsize=(18, 6))
sns.heatmap(
    mean_jac,
    xticklabels=X.columns.tolist(),
    yticklabels=[f"Latent Dim {i}" for i in range(encoding_dim)],
    cmap="YlOrRd",
    annot=False,
    linewidths=0.3,
)
plt.title("Mean Absolute Jacobian: Raw Feature -> Bottleneck Neuron")
plt.xlabel("Input Feature")
plt.ylabel("Bottleneck Neuron")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
