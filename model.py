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
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBClassifier

# %%
df = pd.read_csv("/kaggle/input/<your-dataset-slug>/credit_card_dataset.csv")

print(df.shape)
print(df.head())

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

# %%
# Autoencoder architecture: 34 -> 24 -> 16 -> 10 -> 16 -> 24 -> 34
input_dim = X_train_scaled.shape[1]
encoding_dim = 10

inputs = Input(shape=(input_dim,))

# Encoder
x = Dense(24, activation="relu")(inputs)
x = Dense(16, activation="relu")(x)
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
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

print("Encoded train shape:", X_train_encoded.shape)
print("Encoded test shape:", X_test_encoded.shape)

# %%
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb_model.fit(X_train_encoded, y_train)

# %%
y_pred = xgb_model.predict(X_test_encoded)
y_pred_proba = xgb_model.predict_proba(X_test_encoded)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred).round(4))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba).round(4))
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
