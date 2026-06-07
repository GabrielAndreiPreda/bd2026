"""
UMAP visualisation + KNN classification.

Variants: original (train.csv), smote (train_balanced.csv).
Feature sets: full 18 features, 2D UMAP (n_neighbors=15, min_dist=0.1).
KNN tuning: GridSearchCV (3-fold stratified, scoring=F1).
Metrics: PR-AUC (primary), Brier, F1@tau*, Accuracy, ROC-AUC.

Outputs in results/: results.csv, umap_*.png, confusion_matrix_*.png,
knn_performance_comparison.png.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

DROP_COLS = ["Pays_amts_total", "Utilization_1", "Risk_score"]

KNN_GRID = {
    "n_neighbors": [5, 11, 15, 21, 31],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["default_payment"] + DROP_COLS)
    y = df["default_payment"]
    return X, y


train_df_full = pd.read_csv(DATA_DIR / "train.csv")
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


def tune_knn(X_fit, y_fit):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        KNN_GRID, scoring="f1", cv=cv, n_jobs=-1, verbose=0,
    )
    t0 = time.time()
    gs.fit(X_fit, y_fit)
    return gs.best_estimator_, gs.best_params_, time.time() - t0


# ---------------------------------------------------------------------------
# UMAP visualisation (fit on full original training set)
# ---------------------------------------------------------------------------

scaler_viz = StandardScaler()
X_train_viz_s = scaler_viz.fit_transform(X_train_raw)

umap_viz = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                metric="euclidean", random_state=42)
X_train_umap_viz = umap_viz.fit_transform(X_train_viz_s)

viz_df = pd.DataFrame(X_train_umap_viz, columns=["UMAP_1", "UMAP_2"])
viz_df["default_payment"] = y_train_raw.values
viz_df["Risk_score"] = train_df_full["Risk_score"].values
viz_df["Pay_delay_mean"] = train_df_full["Pay_delay_mean"].values

plt.figure(figsize=(10, 7))
for cls, label in [(0, "Non-default"), (1, "Default")]:
    m = viz_df["default_payment"] == cls
    plt.scatter(viz_df.loc[m, "UMAP_1"], viz_df.loc[m, "UMAP_2"], s=6, alpha=0.5, label=label)
plt.title("UMAP -- Original Training Set (coloured by class)")
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUT_DIR / "umap_original_train_class.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 7))
sc = plt.scatter(viz_df["UMAP_1"], viz_df["UMAP_2"], c=viz_df["Risk_score"], s=6, alpha=0.6)
plt.colorbar(sc, label="Risk_score")
plt.title("UMAP -- Original Training Set (coloured by Risk_score)")
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUT_DIR / "umap_original_train_risk_score.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 7))
sc = plt.scatter(viz_df["UMAP_1"], viz_df["UMAP_2"], c=viz_df["Pay_delay_mean"], s=6, alpha=0.6)
plt.colorbar(sc, label="Pay_delay_mean")
plt.title("UMAP -- Original Training Set (coloured by Pay_delay_mean)")
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUT_DIR / "umap_original_train_pay_delay.png", dpi=150)
plt.close()

scaler_bal_viz = StandardScaler()
X_bal_viz_s = scaler_bal_viz.fit_transform(X_train_bal)
umap_bal_viz = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="euclidean", random_state=42)
X_bal_umap_viz = umap_bal_viz.fit_transform(X_bal_viz_s)

bal_viz_df = pd.DataFrame(X_bal_umap_viz, columns=["UMAP_1", "UMAP_2"])
bal_viz_df["default_payment"] = y_train_bal.values

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for cls, label in [(0, "Non-default"), (1, "Default")]:
    m = viz_df["default_payment"] == cls
    axes[0].scatter(viz_df.loc[m, "UMAP_1"], viz_df.loc[m, "UMAP_2"], s=5, alpha=0.45, label=label)
axes[0].set_title("Original Training Set")
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
axes[0].grid(alpha=0.3); axes[0].legend()
for cls, label in [(0, "Non-default"), (1, "Default")]:
    m = bal_viz_df["default_payment"] == cls
    axes[1].scatter(bal_viz_df.loc[m, "UMAP_1"], bal_viz_df.loc[m, "UMAP_2"], s=5, alpha=0.45, label=label)
axes[1].set_title("SMOTE-Balanced Training Set")
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
axes[1].grid(alpha=0.3); axes[1].legend()
plt.suptitle("UMAP: Original vs SMOTE-Balanced", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "umap_comparison_original_vs_balanced.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# Classification experiments
# ---------------------------------------------------------------------------

scaler = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_cal_s  = scaler.transform(X_cal)
X_test_s = scaler.transform(X_test)
X_bal_s  = scaler.transform(X_train_bal)

records = []

for variant, X_fit_s, y_fit in [
    ("original", X_tr_s,  y_tr),
    ("smote",    X_bal_s, y_train_bal),
]:
    for use_umap in [False, True]:
        tag = variant + ("_UMAP" if use_umap else "")

        if use_umap:
            t0 = time.time()
            reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                           metric="euclidean", random_state=42)
            Xf = reducer.fit_transform(X_fit_s)
            Xc = reducer.transform(X_cal_s)
            Xt = reducer.transform(X_test_s)
            umap_time = time.time() - t0
            n_feats = 2
        else:
            Xf, Xc, Xt = X_fit_s, X_cal_s, X_test_s
            umap_time = 0.0
            n_feats = X_fit_s.shape[1]

        best_knn, best_params, tune_time = tune_knn(Xf, y_fit)

        y_cal_proba  = best_knn.predict_proba(Xc)[:, 1]
        y_test_proba = best_knn.predict_proba(Xt)[:, 1]
        tau = find_tau(y_cal, y_cal_proba)
        metrics = evaluate(y_test, y_test_proba, tau)

        print(f"[{tag}] PR-AUC={metrics['pr_auc']:.4f} F1@t*={metrics['f1']:.4f}")

        records.append({
            "condition":   tag,
            "n_features":  n_feats,
            "train_rows":  len(y_fit),
            "best_k":      best_params["n_neighbors"],
            "best_weight": best_params["weights"],
            "best_metric": best_params["metric"],
            "tune_time_s": round(tune_time, 1),
            "umap_time_s": round(umap_time, 1),
            **metrics,
        })

        y_pred = (y_test_proba >= tau).astype(int)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                               display_labels=["No Default", "Default"]).plot(ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix -- {tag} (t*={tau:.3f})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_{tag}.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(records)
results_df.to_csv(OUT_DIR / "results.csv", index=False)

plot_df = results_df.set_index("condition")[["pr_auc", "f1", "accuracy", "roc_auc"]]
plot_df.plot(kind="bar", figsize=(12, 6))
plt.title("KNN Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15, ha="right")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "knn_performance_comparison.png", dpi=150)
plt.close()
