"""
teammate1/log_regression.py — Logistic Regression baseline

Loads pre-split data from the parent directory (produced by preprocessing.py).
Evaluates 6 conditions (3 training variants × 2 feature sets) and reports the
same metrics as benchmark.py so results are directly comparable.

Training variants:
  original   — train.csv, no balancing
  balanced_cw — train.csv, class_weight='balanced' (upweights minority during fit)
  smote      — train_balanced.csv, SMOTENC-balanced 50/50

Feature sets:
  full — all 18 features
  PCA  — PCA(95% variance) applied after scaling

Metrics: PR-AUC, Brier, F1@τ*, Accuracy, ROC-AUC  (primary: PR-AUC)

Outputs: results/results.csv, results/report.md
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Same three cols dropped by model.py and benchmark.py
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

# Calibration slice carved from raw training set (same 10% / seed=42 as model.py)
# Used for F1-optimal threshold tuning on all variants (original distribution)
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


# ── Training variants ──────────────────────────────────────────────────────────
VARIANTS = [
    ("original",    X_tr,         y_tr,         None),
    ("balanced_cw", X_tr,         y_tr,         "balanced"),
    ("smote",       X_train_bal,  y_train_bal,  None),
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
            Xf = pca.fit_transform(X_fit_s)
            Xc = pca.transform(X_cal_s)
            Xt = pca.transform(X_test_s)
            n_feats = Xf.shape[1]
        else:
            Xf, Xc, Xt = X_fit_s, X_cal_s, X_test_s
            n_feats = X_fit_s.shape[1]

        model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight=class_weight,
        )
        model.fit(Xf, y_fit)

        tau = find_tau(y_cal, model.predict_proba(Xc)[:, 1])
        y_test_proba = model.predict_proba(Xt)[:, 1]
        metrics = evaluate(y_test, y_test_proba, tau)

        record = {"condition": tag, "n_features": n_feats, "train_rows": len(y_fit), **metrics}
        records.append(record)

        print(f"[{tag}]  PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier']:.4f}  "
              f"F1@t*={metrics['f1']:.4f}  t*={tau:.3f}  ROC-AUC={metrics['roc_auc']:.4f}")

        # Save full PR curve for the primary non-PCA conditions
        if not use_pca:
            p, r, _ = precision_recall_curve(y_test, y_test_proba)
            np.savez(OUT_DIR / f"pr_curve_{variant_name}.npz", precision=p, recall=r)

        # Feature coefficients for non-PCA variants (interpretable)
        if not use_pca:
            coeff_df = pd.DataFrame({
                "feature": X_test.columns,
                "coefficient": model.coef_[0],
            }).sort_values("coefficient", ascending=False)
            coeff_tables[variant_name] = coeff_df
            coeff_df.to_csv(OUT_DIR / f"coefficients_{variant_name}.csv", index=False)

print()

# ── Save results CSV ───────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
results_df.to_csv(OUT_DIR / "results.csv", index=False)

# ── Plots ──────────────────────────────────────────────────────────────────────
# PR curves (non-PCA variants)
plt.figure(figsize=(8, 6))
for variant_name, _, _, _ in VARIANTS:
    npz = np.load(OUT_DIR / f"pr_curve_{variant_name}.npz")
    pr_auc = results_df.loc[results_df["condition"] == variant_name, "pr_auc"].values[0]
    plt.plot(npz["recall"], npz["precision"], label=f"{variant_name} (PR-AUC={pr_auc:.4f})")
baseline = y_test.mean()
plt.axhline(baseline, color="gray", linestyle="--", label=f"Random ({baseline:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves — Logistic Regression variants")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curves.png", dpi=150)
plt.show()

# Confusion matrix for best variant (by PR-AUC, non-PCA)
non_pca = results_df[~results_df["condition"].str.endswith("_PCA")]
best_condition = non_pca.loc[non_pca["pr_auc"].idxmax(), "condition"]
best_tau = non_pca.loc[non_pca["pr_auc"].idxmax(), "tau_star"]

variant_name, X_fit, y_fit, class_weight = next(
    v for v in VARIANTS if v[0] == best_condition
)
scaler = StandardScaler()
X_fit_s = scaler.fit_transform(X_fit)
X_test_s = scaler.transform(X_test)
best_model = LogisticRegression(max_iter=2000, random_state=42, class_weight=class_weight)
best_model.fit(X_fit_s, y_fit)
y_pred_best = (best_model.predict_proba(X_test_s)[:, 1] >= best_tau).astype(int)

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"])
plt.title(f"Confusion Matrix — {best_condition} (τ*={best_tau:.3f})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_best.png", dpi=150)
plt.show()

print(f"\nBest condition: {best_condition}")
print(classification_report(y_test, y_pred_best, target_names=["No Default", "Default"]))

# Coefficients for best variant
if best_condition in coeff_tables:
    print(f"\nTop 10 positive features ({best_condition}):")
    print(coeff_tables[best_condition].head(10).to_string(index=False))
    print(f"\nTop 10 negative features ({best_condition}):")
    print(coeff_tables[best_condition].tail(10).to_string(index=False))

# ── Markdown report ────────────────────────────────────────────────────────────
metrics_order = ["pr_auc", "brier", "f1", "accuracy", "roc_auc", "tau_star", "n_features"]

lines = [
    "# Logistic Regression Results",
    "",
    "Metrics aligned with `benchmark.py` for direct comparison. Primary metric: **PR-AUC**.",
    "",
    "## Conditions",
    "",
    "- **original** — `train.csv`, no class balancing, default LR",
    "- **balanced_cw** — `train.csv`, `class_weight='balanced'` (upweights minority during fit)",
    "- **smote** — `train_balanced.csv`, SMOTENC-balanced 50/50",
    "- **_PCA** suffix — PCA(95% variance) applied after StandardScaler",
    "",
    "## Results",
    "",
    "| Condition | PR-AUC | Brier | F1@τ* | Accuracy | ROC-AUC | τ* | Features |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]

for _, row in results_df.sort_values("pr_auc", ascending=False).iterrows():
    lines.append(
        f"| **{row['condition']}** | {row['pr_auc']:.4f} | {row['brier']:.4f} | "
        f"{row['f1']:.4f} | {row['accuracy']:.4f} | {row['roc_auc']:.4f} | "
        f"{row['tau_star']:.3f} | {int(row['n_features'])} |"
    )

# Reference row from benchmark (C0 — best overall)
lines += [
    "| *(C0 XGB-only — benchmark ref)* | *0.5281* | *0.1350* | *0.5465* | *0.7959* | *0.7820* | *0.336* | *18* |",
    "",
    "## Notes",
    "",
    "- τ* tuned on 10% calibration slice carved from `train.csv` (same seed=42 as model.py).",
    "- Brier score is meaningful here because logistic regression is naturally well-calibrated.",
    "- XGBoost benchmark reference (C0) included for context.",
]

report_path = OUT_DIR / "report.md"
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"\nResults saved to {OUT_DIR}/")
print(f"  results.csv")
print(f"  report.md")
print(f"  pr_curves.png")
print(f"  confusion_matrix_best.png")
print(f"  pr_curve_<variant>.npz (×3)")
print(f"  coefficients_<variant>.csv (×3)")
