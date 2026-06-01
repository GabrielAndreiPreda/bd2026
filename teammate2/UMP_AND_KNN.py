"""
teammate2/UMP_AND_KNN.py — UMAP visualization + KNN classification

Loads pre-split data from the parent directory (produced by preprocessing.py).
Evaluates 4 conditions (2 training variants x 2 feature sets) and reports the
same metrics as benchmark.py so results are directly comparable.

Training variants:
  original  — train.csv, no balancing
  smote     — train_balanced.csv, SMOTENC-balanced 50/50

Feature sets:
  full  — all 18 features after StandardScaler
  UMAP  — 2D projection (n_neighbors=15, min_dist=0.1, euclidean)

KNN tuning: GridSearchCV, 3-fold stratified CV, scoring=f1
Grid: n_neighbors in [5,11,15,21,31], weights in [uniform,distance],
      metric in [euclidean,manhattan]

Metrics: PR-AUC, Brier, F1@t*, Accuracy, ROC-AUC  (primary: PR-AUC)

Outputs: results/results.csv, results/report.md, results/*.png
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

DROP_COLS = ["Pays_amts_total", "Utilization_1", "Risk_score"]


# ── Load data ──────────────────────────────────────────────────────────────────
def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["default_payment"] + DROP_COLS)
    y = df["default_payment"]
    return X, y


train_df_full    = pd.read_csv(DATA_DIR / "train.csv")   # keep for Risk_score viz
X_train_raw, y_train_raw = load_xy(DATA_DIR / "train.csv")
X_train_bal, y_train_bal = load_xy(DATA_DIR / "train_balanced.csv")
X_test,      y_test      = load_xy(DATA_DIR / "test.csv")

# Calibration slice (same 10%/seed=42 as benchmark.py and teammate1)
X_tr, X_cal, y_tr, y_cal = train_test_split(
    X_train_raw, y_train_raw,
    test_size=0.10, random_state=42, stratify=y_train_raw,
)

print(f"Train (original):  {len(X_tr)} rows, default rate {y_tr.mean():.3f}")
print(f"Train (balanced):  {len(X_train_bal)} rows, default rate {y_train_bal.mean():.3f}")
print(f"Cal slice:         {len(X_cal)} rows")
print(f"Test:              {len(X_test)} rows, default rate {y_test.mean():.3f}")
print()


# ── Helpers ────────────────────────────────────────────────────────────────────
def find_tau(y_true, y_proba):
    """F1-optimal threshold on a held-out calibration set."""
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


def tune_knn(X_fit, y_fit, label):
    param_grid = {
        "n_neighbors": [5, 11, 15, 21, 31],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan"],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    gs.fit(X_fit, y_fit)
    elapsed = time.time() - t0
    print(f"  [{label}] best params: {gs.best_params_}, CV F1={gs.best_score_:.4f}, "
          f"tuning={elapsed:.1f}s")
    return gs.best_estimator_, gs.best_params_, elapsed


# ── UMAP visualisation (fit on full original training set) ─────────────────────
print("Fitting UMAP for visualisation (full original train)...")
scaler_viz = StandardScaler()
X_train_viz_s = scaler_viz.fit_transform(X_train_raw)

t0 = time.time()
umap_viz = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                metric="euclidean", random_state=42)
X_train_umap_viz = umap_viz.fit_transform(X_train_viz_s)
umap_viz_time = time.time() - t0
print(f"  done in {umap_viz_time:.1f}s\n")

viz_df = pd.DataFrame(X_train_umap_viz, columns=["UMAP_1", "UMAP_2"])
viz_df["default_payment"] = y_train_raw.values
viz_df["Risk_score"] = train_df_full["Risk_score"].values
viz_df["Pay_delay_mean"] = train_df_full["Pay_delay_mean"].values

# Plot 1: coloured by class
plt.figure(figsize=(10, 7))
for cls, label in [(0, "Non-default"), (1, "Default")]:
    m = viz_df["default_payment"] == cls
    plt.scatter(viz_df.loc[m, "UMAP_1"], viz_df.loc[m, "UMAP_2"],
                s=6, alpha=0.5, label=label)
plt.title("UMAP — Original Training Set (coloured by class)")
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUT_DIR / "umap_original_train_class.png", dpi=150)
plt.close()

# Plot 2: coloured by Risk_score
plt.figure(figsize=(10, 7))
sc = plt.scatter(viz_df["UMAP_1"], viz_df["UMAP_2"],
                 c=viz_df["Risk_score"], s=6, alpha=0.6)
plt.colorbar(sc, label="Risk_score")
plt.title("UMAP — Original Training Set (coloured by Risk_score)")
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUT_DIR / "umap_original_train_risk_score.png", dpi=150)
plt.close()

# Plot 3: coloured by Pay_delay_mean
plt.figure(figsize=(10, 7))
sc = plt.scatter(viz_df["UMAP_1"], viz_df["UMAP_2"],
                 c=viz_df["Pay_delay_mean"], s=6, alpha=0.6)
plt.colorbar(sc, label="Pay_delay_mean")
plt.title("UMAP — Original Training Set (coloured by Pay_delay_mean)")
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUT_DIR / "umap_original_train_pay_delay.png", dpi=150)
plt.close()

print("Saved UMAP visualisation plots.\n")


# ── UMAP for balanced set visualisation ───────────────────────────────────────
print("Fitting UMAP for balanced set visualisation...")
scaler_bal_viz = StandardScaler()
X_bal_viz_s = scaler_bal_viz.fit_transform(X_train_bal)

t0 = time.time()
umap_bal_viz = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="euclidean", random_state=42)
X_bal_umap_viz = umap_bal_viz.fit_transform(X_bal_viz_s)
umap_bal_viz_time = time.time() - t0
print(f"  done in {umap_bal_viz_time:.1f}s\n")

bal_viz_df = pd.DataFrame(X_bal_umap_viz, columns=["UMAP_1", "UMAP_2"])
bal_viz_df["default_payment"] = y_train_bal.values

# Plot 4: side-by-side original vs balanced
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for cls, label in [(0, "Non-default"), (1, "Default")]:
    m = viz_df["default_payment"] == cls
    axes[0].scatter(viz_df.loc[m, "UMAP_1"], viz_df.loc[m, "UMAP_2"],
                    s=5, alpha=0.45, label=label)
axes[0].set_title("Original Training Set")
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
axes[0].grid(alpha=0.3); axes[0].legend()

for cls, label in [(0, "Non-default"), (1, "Default")]:
    m = bal_viz_df["default_payment"] == cls
    axes[1].scatter(bal_viz_df.loc[m, "UMAP_1"], bal_viz_df.loc[m, "UMAP_2"],
                    s=5, alpha=0.45, label=label)
axes[1].set_title("SMOTE-Balanced Training Set")
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
axes[1].grid(alpha=0.3); axes[1].legend()

plt.suptitle("UMAP: Original vs SMOTE-Balanced", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "umap_comparison_original_vs_balanced.png", dpi=150)
plt.close()
print("Saved balanced UMAP visualisation plot.\n")


# ── Classification experiments ─────────────────────────────────────────────────
# Scaler fit on X_tr (original training minus cal slice) — same convention as
# teammate1 and benchmark.
scaler = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_cal_s  = scaler.transform(X_cal)
X_test_s = scaler.transform(X_test)

X_bal_s = scaler.transform(X_train_bal)    # same scaler, fit on original train

records = []

for variant, X_fit_s, y_fit, X_fit_cal_s, y_fit_cal in [
    ("original", X_tr_s,  y_tr,         X_cal_s, y_cal),
    ("smote",    X_bal_s, y_train_bal,  X_cal_s, y_cal),
]:
    for use_umap in [False, True]:
        tag = variant + ("_UMAP" if use_umap else "")

        if use_umap:
            print(f"Fitting UMAP for {tag}...")
            t0 = time.time()
            reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                           metric="euclidean", random_state=42)
            Xf = reducer.fit_transform(X_fit_s)
            Xc = reducer.transform(X_fit_cal_s)
            Xt = reducer.transform(X_test_s)
            umap_time = time.time() - t0
            print(f"  done in {umap_time:.1f}s")
            n_feats = 2
        else:
            Xf, Xc, Xt = X_fit_s, X_fit_cal_s, X_test_s
            umap_time = 0.0
            n_feats = X_fit_s.shape[1]

        best_knn, best_params, tune_time = tune_knn(Xf, y_fit, tag)

        y_cal_proba  = best_knn.predict_proba(Xc)[:, 1]
        y_test_proba = best_knn.predict_proba(Xt)[:, 1]

        tau = find_tau(y_fit_cal, y_cal_proba)
        metrics = evaluate(y_test, y_test_proba, tau)

        print(f"  [{tag}]  PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier']:.4f}  "
              f"F1@t*={metrics['f1']:.4f}  t*={tau:.3f}  ROC-AUC={metrics['roc_auc']:.4f}\n")

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

        # Confusion matrix
        y_pred = (y_test_proba >= tau).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(
            ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix — {tag} (t*={tau:.3f})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_{tag}.png", dpi=150)
        plt.close()


# ── Save results ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
results_df.to_csv(OUT_DIR / "results.csv", index=False)

# Performance comparison bar chart
metrics_to_plot = ["pr_auc", "f1", "accuracy", "roc_auc"]
plot_df = results_df.set_index("condition")[metrics_to_plot]
ax = plot_df.plot(kind="bar", figsize=(12, 6))
plt.title("KNN Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15, ha="right")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "knn_performance_comparison.png", dpi=150)
plt.close()

print("\nResults:")
print(results_df[["condition", "pr_auc", "brier", "f1", "accuracy", "roc_auc", "tau_star"]].to_string(index=False))


# ── Best condition detail ──────────────────────────────────────────────────────
non_umap = results_df[~results_df["condition"].str.endswith("_UMAP")]
best_row = non_umap.loc[non_umap["pr_auc"].idxmax()]
print(f"\nBest non-UMAP condition: {best_row['condition']} (PR-AUC={best_row['pr_auc']:.4f})")

best_tag = best_row["condition"]
best_tau = best_row["tau_star"]
y_pred_best = (pd.read_csv(OUT_DIR / f"confusion_matrix_{best_tag}.png")
               if False else None)  # already saved; rebuild for classification report

# Rebuild predictions for classification report
variant = "original" if "original" in best_tag else "smote"
X_fit_s_b = X_tr_s if variant == "original" else X_bal_s
y_fit_b   = y_tr   if variant == "original" else y_train_bal
best_knn_b, _, _ = tune_knn(X_fit_s_b, y_fit_b, f"{best_tag}_report")
y_pred_report = (best_knn_b.predict_proba(X_test_s)[:, 1] >= best_tau).astype(int)
print(classification_report(y_test, y_pred_report, target_names=["No Default", "Default"]))


# ── Markdown report ────────────────────────────────────────────────────────────
lines = [
    "# UMAP + KNN Results",
    "",
    "Metrics aligned with `benchmark.py`. Primary metric: **PR-AUC**.",
    "",
    "## Conditions",
    "",
    "- **original** — `train.csv`, no balancing",
    "- **smote** — `train_balanced.csv`, SMOTENC 50/50 balance",
    "- **_UMAP** suffix — 2D UMAP (n_neighbors=15, min_dist=0.1, euclidean) applied after StandardScaler",
    "",
    "KNN grid: `n_neighbors` in [5, 11, 15, 21, 31], `weights` in [uniform, distance], "
    "`metric` in [euclidean, manhattan]. 3-fold stratified CV, scoring=F1.",
    "",
    "## Results",
    "",
    "| Condition | PR-AUC | Brier | F1@t* | Accuracy | ROC-AUC | t* | k | Features |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for _, row in results_df.sort_values("pr_auc", ascending=False).iterrows():
    lines.append(
        f"| **{row['condition']}** | {row['pr_auc']:.4f} | {row['brier']:.4f} | "
        f"{row['f1']:.4f} | {row['accuracy']:.4f} | {row['roc_auc']:.4f} | "
        f"{row['tau_star']:.3f} | {int(row['best_k'])} | {int(row['n_features'])} |"
    )

lines += [
    "| *(C0 XGB-only — benchmark ref)* | *0.5281* | *0.1350* | *0.5465* | *0.7959* | *0.7820* | *0.336* | — | *18* |",
    "",
    "## UMAP hyperparameters",
    "",
    "| Variant | Rows | Components | n_neighbors | min_dist | Metric |",
    "| --- | --- | --- | --- | --- | --- |",
    f"| original (viz) | {len(X_train_raw)} | 2 | 15 | 0.1 | euclidean |",
    f"| balanced (viz) | {len(X_train_bal)} | 2 | 15 | 0.1 | euclidean |",
    "",
    "## Notes",
    "",
    "- t* tuned on 10% calibration slice carved from `train.csv` (seed=42, same as benchmark).",
    "- UMAP fitted on the training portion only; calibration and test sets are transformed.",
    "- KNN hyperparameter search uses F1 CV score (not PR-AUC) to keep grid search tractable.",
    "- Benchmark reference row (C0) included for cross-model comparison.",
]

(OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"\nResults saved to {OUT_DIR}/")
print("  results.csv, report.md")
print("  umap_*.png (4 visualisation plots)")
print("  knn_performance_comparison.png")
print("  confusion_matrix_<condition>.png (x4)")
