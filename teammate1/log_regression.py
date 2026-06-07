"""
Logistic Regression baseline: 3 training variants x 2 feature sets.

Variants: original, balanced_cw (class_weight='balanced'), smote (SMOTENC).
Feature sets: full 18 features, PCA(95% variance).
Metrics: PR-AUC (primary), Brier, F1@tau*, Accuracy, ROC-AUC.

Outputs in results/: results.csv, coefficients_<variant>.csv,
pr_curve_<variant>.npz, pr_curves.png, confusion_matrix_best.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

DROP_COLS = ["Pays_amts_total", "Utilization_1", "Risk_score"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["default_payment"] + DROP_COLS)
    y = df["default_payment"]
    return X, y


X_train_raw, y_train_raw = load_xy(DATA_DIR / "train.csv")
X_train_bal, y_train_bal = load_xy(DATA_DIR / "train_balanced.csv")
X_test,      y_test      = load_xy(DATA_DIR / "test.csv")

X_tr, X_cal, y_tr, y_cal = train_test_split(
    X_train_raw, y_train_raw,
    test_size=0.10, random_state=42, stratify=y_train_raw,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tau(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1s)])


def evaluate(y_true, y_proba, tau):
    y_pred = (y_proba >= tau).astype(int)
    return {
        "pr_auc":   float(average_precision_score(y_true, y_proba)),
        "brier":    float(brier_score_loss(y_true, y_proba)),
        "f1":       float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc":  float(roc_auc_score(y_true, y_proba)),
        "tau_star": tau,
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

VARIANTS = [
    ("original",    X_tr,        y_tr,        None),
    ("balanced_cw", X_tr,        y_tr,        "balanced"),
    ("smote",       X_train_bal, y_train_bal, None),
]

records = []
coeff_tables = {}

for variant_name, X_fit, y_fit, class_weight in VARIANTS:
    scaler = StandardScaler()
    X_fit_s  = scaler.fit_transform(X_fit)
    X_cal_s  = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    for use_pca in [False, True]:
        tag = variant_name + ("_PCA" if use_pca else "")

        if use_pca:
            pca = PCA(n_components=0.95, random_state=42)
            Xf, Xc, Xt = pca.fit_transform(X_fit_s), pca.transform(X_cal_s), pca.transform(X_test_s)
            n_feats = Xf.shape[1]
        else:
            Xf, Xc, Xt = X_fit_s, X_cal_s, X_test_s
            n_feats = X_fit_s.shape[1]

        model = LogisticRegression(max_iter=2000, random_state=42, class_weight=class_weight)
        model.fit(Xf, y_fit)

        tau = find_tau(y_cal, model.predict_proba(Xc)[:, 1])
        y_test_proba = model.predict_proba(Xt)[:, 1]
        metrics = evaluate(y_test, y_test_proba, tau)

        records.append({"condition": tag, "n_features": n_feats, "train_rows": len(y_fit), **metrics})
        print(f"[{tag}] PR-AUC={metrics['pr_auc']:.4f} Brier={metrics['brier']:.4f} F1@t*={metrics['f1']:.4f}")

        if not use_pca:
            p, r, _ = precision_recall_curve(y_test, y_test_proba)
            np.savez(OUT_DIR / f"pr_curve_{variant_name}.npz", precision=p, recall=r)

            coeff_df = pd.DataFrame({
                "feature": X_test.columns,
                "coefficient": model.coef_[0],
            }).sort_values("coefficient", ascending=False)
            coeff_tables[variant_name] = coeff_df
            coeff_df.to_csv(OUT_DIR / f"coefficients_{variant_name}.csv", index=False)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(records)
results_df.to_csv(OUT_DIR / "results.csv", index=False)

plt.figure(figsize=(8, 6))
for variant_name, _, _, _ in VARIANTS:
    npz = np.load(OUT_DIR / f"pr_curve_{variant_name}.npz")
    pr_auc = results_df.loc[results_df["condition"] == variant_name, "pr_auc"].values[0]
    plt.plot(npz["recall"], npz["precision"], label=f"{variant_name} (PR-AUC={pr_auc:.4f})")
plt.axhline(y_test.mean(), color="gray", linestyle="--", label=f"Random ({y_test.mean():.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves -- Logistic Regression variants")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curves.png", dpi=150)
plt.close()

non_pca = results_df[~results_df["condition"].str.endswith("_PCA")]
best_row = non_pca.loc[non_pca["pr_auc"].idxmax()]
best_condition = best_row["condition"]
best_tau = best_row["tau_star"]

variant_name, X_fit, y_fit, class_weight = next(v for v in VARIANTS if v[0] == best_condition)
scaler = StandardScaler()
X_fit_s = scaler.fit_transform(X_fit)
X_test_s = scaler.transform(X_test)
best_model = LogisticRegression(max_iter=2000, random_state=42, class_weight=class_weight)
best_model.fit(X_fit_s, y_fit)
y_pred_best = (best_model.predict_proba(X_test_s)[:, 1] >= best_tau).astype(int)

plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred_best),
    annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Default", "Default"],
    yticklabels=["No Default", "Default"],
)
plt.title(f"Confusion Matrix -- {best_condition} (tau*={best_tau:.3f})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_best.png", dpi=150)
plt.close()
