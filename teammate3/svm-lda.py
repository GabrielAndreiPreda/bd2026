"""
teammate3/svm-lda.py — SVM and SVM+LDA classification

Loads pre-split data from the parent directory (produced by preprocessing.py).
Evaluates 4 conditions (2 training variants x 2 dimensionality settings) and
reports the same metrics as benchmark.py so results are directly comparable.

Training variants:
  original  — train.csv, no balancing
  balanced  — train_balanced.csv, SMOTENC 50/50 balance

Dimensionality:
  full  — 18 features after StandardScaler
  LDA   — 1D Linear Discriminant Analysis projection (binary classification)

Classifier: SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


X_train_raw, y_train_raw = load_xy(DATA_DIR / "train.csv")
X_train_bal, y_train_bal = load_xy(DATA_DIR / "train_balanced.csv")
X_test,      y_test      = load_xy(DATA_DIR / "test.csv")

# Calibration slice (same 10%/seed=42 as benchmark and teammates 1, 2)
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


# ── Experiments ────────────────────────────────────────────────────────────────
VARIANTS = [
    ("original", X_tr,         y_tr,         None),
    ("balanced", X_train_bal,  y_train_bal,  None),
]

records = []

for variant_name, X_fit, y_fit, _ in VARIANTS:
    scaler = StandardScaler()
    X_fit_s  = scaler.fit_transform(X_fit)
    X_cal_s  = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    for use_lda in [False, True]:
        tag = variant_name + ("_LDA" if use_lda else "")
        n_feats = 18

        if use_lda:
            lda = LDA(n_components=1)
            Xf = lda.fit_transform(X_fit_s, y_fit)
            Xc = lda.transform(X_cal_s)
            Xt = lda.transform(X_test_s)
            n_feats = 1
        else:
            Xf, Xc, Xt = X_fit_s, X_cal_s, X_test_s

        print(f"[{tag}] fitting SVM ({len(y_fit)} rows, {n_feats} features)...")
        t0 = time.time()
        svm = SVC(kernel="rbf", probability=True, random_state=42, cache_size=2000)
        svm.fit(Xf, y_fit)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")

        y_cal_proba  = svm.predict_proba(Xc)[:, 1]
        y_test_proba = svm.predict_proba(Xt)[:, 1]

        tau = find_tau(y_cal, y_cal_proba)
        metrics = evaluate(y_test, y_test_proba, tau)

        print(f"  PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier']:.4f}  "
              f"F1@t*={metrics['f1']:.4f}  t*={tau:.3f}  ROC-AUC={metrics['roc_auc']:.4f}\n")

        records.append({
            "condition":  tag,
            "n_features": n_feats,
            "train_rows": len(y_fit),
            "fit_time_s": round(elapsed, 1),
            **metrics,
        })

        # Confusion matrix
        y_pred = (y_test_proba >= tau).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Default", "Default"],
                    yticklabels=["No Default", "Default"])
        ax.set_title(f"Confusion Matrix — {tag} (t*={tau:.3f})")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_{tag}.png", dpi=150)
        plt.close()

        # For LDA variants: save the 1D LDA distribution and ROC curve
        if use_lda:
            # LDA distribution
            lda_test_df = pd.DataFrame({
                "LDA_Component": Xt.flatten(),
                "True_Class":    y_test.values,
            })
            fig, ax = plt.subplots(figsize=(8, 5))
            for cls, color, lbl in [(0, "blue", "Non-default"), (1, "red", "Default")]:
                data = lda_test_df.loc[lda_test_df["True_Class"] == cls, "LDA_Component"]
                data.plot.kde(ax=ax, color=color, label=lbl)
                ax.fill_between(
                    np.linspace(data.min(), data.max(), 200),
                    0,
                    np.zeros(200),   # placeholder; kde drawn above
                    alpha=0.2, color=color,
                )
            ax.set_title(f"LDA 1D Distribution — {tag}")
            ax.set_xlabel("LDA Component"); ax.set_ylabel("Density")
            ax.legend(); plt.tight_layout()
            plt.savefig(OUT_DIR / f"lda_distribution_{tag}.png", dpi=150)
            plt.close()

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_auc_val = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, color="darkorange", lw=2,
                    label=f"ROC (AUC={roc_auc_val:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {tag}")
            ax.legend(); plt.tight_layout()
            plt.savefig(OUT_DIR / f"roc_curve_{tag}.png", dpi=150)
            plt.close()


# ── Save results ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
results_df.to_csv(OUT_DIR / "results.csv", index=False)

print("\nResults:")
print(results_df[["condition", "pr_auc", "brier", "f1", "accuracy", "roc_auc", "tau_star"]].to_string(index=False))

best_condition = results_df.loc[results_df["pr_auc"].idxmax(), "condition"]
print(f"\nBest condition: {best_condition}")

# Classification report for best condition (refit to get predictions)
best_row = results_df.loc[results_df["pr_auc"].idxmax()]
var = "original" if "original" in best_condition else "balanced"
X_fit_b = X_tr if var == "original" else X_train_bal
y_fit_b = y_tr if var == "original" else y_train_bal
use_lda_b = "_LDA" in best_condition

scaler_b = StandardScaler()
X_fit_bs  = scaler_b.fit_transform(X_fit_b)
X_test_bs = scaler_b.transform(X_test)
if use_lda_b:
    lda_b = LDA(n_components=1)
    X_fit_bs  = lda_b.fit_transform(X_fit_bs, y_fit_b)
    X_test_bs = lda_b.transform(X_test_bs)

svm_b = SVC(kernel="rbf", probability=True, random_state=42, cache_size=2000)
svm_b.fit(X_fit_bs, y_fit_b)
y_pred_best = (svm_b.predict_proba(X_test_bs)[:, 1] >= best_row["tau_star"]).astype(int)
print(classification_report(y_test, y_pred_best, target_names=["No Default", "Default"]))


# ── Markdown report ────────────────────────────────────────────────────────────
lines = [
    "# SVM + LDA Results",
    "",
    "Metrics aligned with `benchmark.py`. Primary metric: **PR-AUC**.",
    "",
    "## Conditions",
    "",
    "- **original** — `train.csv`, no balancing",
    "- **balanced** — `train_balanced.csv`, SMOTENC 50/50 balance",
    "- **_LDA** suffix — 1D Linear Discriminant Analysis projection before SVM",
    "",
    "Classifier: `SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')`",
    "",
    "## Results",
    "",
    "| Condition | PR-AUC | Brier | F1@t* | Accuracy | ROC-AUC | t* | Features | Fit(s) |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for _, row in results_df.sort_values("pr_auc", ascending=False).iterrows():
    lines.append(
        f"| **{row['condition']}** | {row['pr_auc']:.4f} | {row['brier']:.4f} | "
        f"{row['f1']:.4f} | {row['accuracy']:.4f} | {row['roc_auc']:.4f} | "
        f"{row['tau_star']:.3f} | {int(row['n_features'])} | {row['fit_time_s']:.0f} |"
    )

lines += [
    "| *(C0 XGB-only — benchmark ref)* | *0.5281* | *0.1350* | *0.5465* | *0.7959* | *0.7820* | *0.336* | *18* | — |",
    "",
    "## Notes",
    "",
    "- t* tuned on 10% calibration slice carved from `train.csv` (seed=42, same as benchmark).",
    "- SVM uses Platt scaling (`probability=True`) for calibrated probabilities; Brier score is meaningful.",
    "- LDA reduces 18 features to 1 component (binary classification); SVM on 1D is very fast.",
    "- Benchmark reference row (C0) included for cross-model comparison.",
]

(OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"\nResults saved to {OUT_DIR}/")
print("  results.csv, report.md")
print("  confusion_matrix_<condition>.png (x4)")
print("  lda_distribution_<condition>.png (x2)")
print("  roc_curve_<condition>.png (x2)")
