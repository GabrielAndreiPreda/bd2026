"""
SVM (RBF) and SVM + LDA classification.

Variants: original (train.csv), balanced (train_balanced.csv).
Feature sets: full 18 features, 1D LDA projection.
Classifier: SVC(kernel='rbf', probability=True, C=1.0, gamma='scale').
Metrics: PR-AUC (primary), Brier, F1@tau*, Accuracy, ROC-AUC.

Outputs in results/: results.csv, confusion_matrix_*.png,
lda_distribution_*.png, roc_curve_*.png.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

DROP_COLS = ["Pays_amts_total", "Utilization_1", "Risk_score"]

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

VARIANTS = [
    ("original", X_tr,        y_tr),
    ("balanced", X_train_bal, y_train_bal),
]

records = []

for variant_name, X_fit, y_fit in VARIANTS:
    scaler = StandardScaler()
    X_fit_s  = scaler.fit_transform(X_fit)
    X_cal_s  = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    for use_lda in [False, True]:
        tag = variant_name + ("_LDA" if use_lda else "")

        if use_lda:
            lda = LDA(n_components=1)
            Xf = lda.fit_transform(X_fit_s, y_fit)
            Xc = lda.transform(X_cal_s)
            Xt = lda.transform(X_test_s)
            n_feats = 1
        else:
            Xf, Xc, Xt = X_fit_s, X_cal_s, X_test_s
            n_feats = X_fit_s.shape[1]

        t0 = time.time()
        svm = SVC(kernel="rbf", probability=True, random_state=42, cache_size=2000)
        svm.fit(Xf, y_fit)
        elapsed = time.time() - t0

        y_cal_proba  = svm.predict_proba(Xc)[:, 1]
        y_test_proba = svm.predict_proba(Xt)[:, 1]
        tau = find_tau(y_cal, y_cal_proba)
        metrics = evaluate(y_test, y_test_proba, tau)

        print(f"[{tag}] PR-AUC={metrics['pr_auc']:.4f} F1@t*={metrics['f1']:.4f} fit={elapsed:.1f}s")

        records.append({
            "condition":  tag,
            "n_features": n_feats,
            "train_rows": len(y_fit),
            "fit_time_s": round(elapsed, 1),
            **metrics,
        })

        y_pred = (y_test_proba >= tau).astype(int)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
        )
        ax.set_title(f"Confusion Matrix -- {tag} (t*={tau:.3f})")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_{tag}.png", dpi=150)
        plt.close()

        if use_lda:
            lda_test_df = pd.DataFrame({
                "LDA_Component": Xt.flatten(),
                "True_Class":    y_test.values,
            })
            fig, ax = plt.subplots(figsize=(8, 5))
            for cls, color, lbl in [(0, "blue", "Non-default"), (1, "red", "Default")]:
                data = lda_test_df.loc[lda_test_df["True_Class"] == cls, "LDA_Component"]
                data.plot.kde(ax=ax, color=color, label=lbl)
            ax.set_title(f"LDA 1D Distribution -- {tag}")
            ax.set_xlabel("LDA Component"); ax.set_ylabel("Density")
            ax.legend(); plt.tight_layout()
            plt.savefig(OUT_DIR / f"lda_distribution_{tag}.png", dpi=150)
            plt.close()

            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc(fpr, tpr):.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve -- {tag}")
            ax.legend(); plt.tight_layout()
            plt.savefig(OUT_DIR / f"roc_curve_{tag}.png", dpi=150)
            plt.close()

results_df = pd.DataFrame(records)
results_df.to_csv(OUT_DIR / "results.csv", index=False)
