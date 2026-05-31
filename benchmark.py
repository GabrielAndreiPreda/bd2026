"""A/B benchmark harness for the AE+XGB credit-default pipeline.

Runs 7 conditions × 3 seeds with a reduced XGBoost grid; produces a markdown
report and CSV in ./bench_out/. See MEMORY.md for what each condition isolates.

Requires: xgboost, imbalanced-learn, tensorflow, scikit-learn, pandas, numpy.

Usage:
    python benchmark.py
"""

import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _find_csv():
    candidates = [
        "/kaggle/input/datasets/gabrielpredaz/creditcarddata/credit_card_data.csv",
        "./credit_card_data.csv",
        "/mnt/e/Projects/Master/BD/credit_card_data.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("credit_card_data.csv not found; edit CSV_PATH at the top of benchmark.py")


CSV_PATH = _find_csv()
OUT_DIR = Path("/kaggle/working/bench_out") if Path("/kaggle/working").exists() else Path(__file__).resolve().parent / "bench_out"
SEEDS = [0, 1, 2]

# Reduced grid: 16 cells × 5 folds = 80 fits per condition.
PARAM_GRID = {
    "n_estimators":     [200, 400],
    "max_depth":        [4, 6],
    "learning_rate":    [0.05, 0.1],
    "min_child_weight": [1, 3],
}

CONDITIONS = [
    dict(id="C0", use_ae=False, ae_weighted=None,  use_pay0=False, spw="auto", sampler=None),
    dict(id="C1", use_ae=False, ae_weighted=None,  use_pay0=False, spw=1.0,    sampler=None),
    dict(id="C2", use_ae=True,  ae_weighted=True,  use_pay0=True,  spw="auto", sampler=None),
    dict(id="C3", use_ae=True,  ae_weighted=False, use_pay0=True,  spw="auto", sampler=None),
    dict(id="C4", use_ae=True,  ae_weighted=True,  use_pay0=False, spw="auto", sampler=None),
    dict(id="C5", use_ae=True,  ae_weighted=True,  use_pay0=True,  spw=1.0,    sampler="smote"),
    dict(id="C6", use_ae=True,  ae_weighted=True,  use_pay0=True,  spw=1.0,    sampler="borderline"),
]

CONDITION_DESCRIPTIONS = {
    "C0": "XGB-only, scale_pos_weight=auto",
    "C1": "XGB-only, scale_pos_weight=1",
    "C2": "AE+XGB current (B.4 weighted AE + C.3 PAY_0 concat)",
    "C3": "AE+XGB without B.4 (class-blind AE)",
    "C4": "AE+XGB without C.3 (latents only)",
    "C5": "AE+XGB + SMOTE on latents (scale_pos_weight=1)",
    "C6": "AE+XGB + Borderline-SMOTE on latents (scale_pos_weight=1)",
}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ---------------------------------------------------------------------------
# Data loading — CSV is fully preprocessed by preprocessing.py
# ---------------------------------------------------------------------------

def load_data(csv_path):
    """Load the ML-ready CSV produced by preprocessing.py.

    Drops the three interpretability-only columns that are retained in the
    CSV but not used for training.
    """
    df = pd.read_csv(csv_path)
    drop_cols = ["Pays_amts_total", "Utilization_1", "Risk_score"]
    X = df.drop(columns=["default_payment"] + drop_cols)
    y = df["default_payment"]
    return X, y


def make_splits(X, y, seed):
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_tr_full, y_tr_full, test_size=0.10, random_state=seed, stratify=y_tr_full,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_cal_scaled = scaler.transform(X_cal).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    pay_0_idx = X_train.columns.get_loc("PAY_0")

    return dict(
        X_train=X_train,
        X_cal=X_cal,
        X_test=X_test,
        y_train=y_train.reset_index(drop=True),
        y_cal=y_cal.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        X_train_scaled=X_train_scaled,
        X_cal_scaled=X_cal_scaled,
        X_test_scaled=X_test_scaled,
        pay_0_idx=pay_0_idx,
        feature_names=list(X_train.columns),
    )


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

def build_ae(input_dim, seed):
    tf.keras.utils.set_random_seed(seed)
    inputs = Input(shape=(input_dim,))
    x = Dense(24, activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.2)(x)
    encoded = Dense(10, activation="relu")(x)
    x = Dense(16, activation="relu")(encoded)
    x = Dense(24, activation="relu")(x)
    decoded = Dense(input_dim, activation="linear")(x)
    autoencoder = Model(inputs, decoded, name="autoencoder")
    encoder = Model(inputs, encoded, name="encoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def train_ae(X_scaled, y_train, *, weighted, seed):
    autoencoder, encoder = build_ae(X_scaled.shape[1], seed)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    fit_kwargs = dict(
        x=X_scaled,
        y=X_scaled,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )
    if weighted:
        spw = (y_train == 0).sum() / (y_train == 1).sum()
        fit_kwargs["sample_weight"] = np.where(y_train.values == 1, spw, 1.0).astype(np.float32)

    autoencoder.fit(**fit_kwargs)
    return encoder


def encode(encoder, X_scaled):
    return encoder.predict(X_scaled, verbose=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def assemble_features(latents, X_scaled, pay_0_idx, *, use_latents, concat_pay0):
    if use_latents and concat_pay0:
        return np.concatenate([latents, X_scaled[:, pay_0_idx:pay_0_idx + 1]], axis=1).astype(np.float32)
    if use_latents:
        return latents.astype(np.float32)
    return X_scaled.astype(np.float32)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

def build_estimator(seed, *, sampler, scale_pos_weight):
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss",
        device="cuda",
        tree_method="hist",
    )
    if sampler is None:
        return xgb
    if sampler == "smote":
        return ImbPipeline([("samp", SMOTE(random_state=seed)), ("clf", xgb)])
    if sampler == "borderline":
        return ImbPipeline([("samp", BorderlineSMOTE(random_state=seed)), ("clf", xgb)])
    raise ValueError(f"Unknown sampler: {sampler!r}")


def prefix_grid(grid, has_sampler):
    if not has_sampler:
        return grid
    return {f"clf__{k}": v for k, v in grid.items()}


# ---------------------------------------------------------------------------
# Fit / calibrate / threshold
# ---------------------------------------------------------------------------

def fit_and_calibrate(estimator, X_train_feat, y_train, X_cal_feat, y_cal, grid, seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    t0 = time.time()

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,  # XGBoost on GPU does internal parallelism; multiple jobs cause CUDA contention
        verbose=0,
    )
    grid_search.fit(X_train_feat, y_train)
    best_est = grid_search.best_estimator_
    best_params = dict(grid_search.best_params_)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        calibrator = CalibratedClassifierCV(best_est, method="isotonic", cv="prefit")
        calibrator.fit(X_cal_feat, y_cal)

    y_cal_proba = calibrator.predict_proba(X_cal_feat)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_cal, y_cal_proba)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    tau_star = float(thresholds[best_idx])

    train_sec = time.time() - t0
    return calibrator, best_params, tau_star, train_sec, y_cal_proba


def evaluate(calibrator, X_test_feat, y_test, tau_star):
    y_test_proba = calibrator.predict_proba(X_test_feat)[:, 1]
    y_test_pred = (y_test_proba >= tau_star).astype(int)
    metrics = dict(
        pr_auc=float(average_precision_score(y_test, y_test_proba)),
        brier=float(brier_score_loss(y_test, y_test_proba)),
        f1=float(f1_score(y_test, y_test_pred)),
        accuracy=float(accuracy_score(y_test, y_test_pred)),
        roc_auc=float(roc_auc_score(y_test, y_test_proba)),
    )
    return metrics, y_test_proba


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------

def run_condition(cfg, seed, ae_cache, splits):
    cond_id = cfg["id"]
    y_train = splits["y_train"]

    if cfg["spw"] == "auto":
        spw = float((y_train == 0).sum() / (y_train == 1).sum())
    else:
        spw = float(cfg["spw"])

    if cfg["use_ae"]:
        encoder = ae_cache[cfg["ae_weighted"]]
        latents_train = encode(encoder, splits["X_train_scaled"])
        latents_cal = encode(encoder, splits["X_cal_scaled"])
        latents_test = encode(encoder, splits["X_test_scaled"])
    else:
        latents_train = latents_cal = latents_test = None

    X_train_feat = assemble_features(
        latents_train, splits["X_train_scaled"], splits["pay_0_idx"],
        use_latents=cfg["use_ae"], concat_pay0=cfg["use_pay0"],
    )
    X_cal_feat = assemble_features(
        latents_cal, splits["X_cal_scaled"], splits["pay_0_idx"],
        use_latents=cfg["use_ae"], concat_pay0=cfg["use_pay0"],
    )
    X_test_feat = assemble_features(
        latents_test, splits["X_test_scaled"], splits["pay_0_idx"],
        use_latents=cfg["use_ae"], concat_pay0=cfg["use_pay0"],
    )

    estimator = build_estimator(seed, sampler=cfg["sampler"], scale_pos_weight=spw)
    grid = prefix_grid(PARAM_GRID, has_sampler=cfg["sampler"] is not None)

    calibrator, best_params, tau_star, train_sec, y_cal_proba = fit_and_calibrate(
        estimator, X_train_feat, y_train, X_cal_feat, splits["y_cal"], grid, seed,
    )

    metrics, y_test_proba = evaluate(calibrator, X_test_feat, splits["y_test"], tau_star)

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / f"best_params_{cond_id}_{seed}.json", "w") as f:
        json.dump({str(k): v for k, v in best_params.items()}, f, indent=2)

    np.savez(
        OUT_DIR / f"cal_diag_{cond_id}_{seed}.npz",
        y_cal_proba=y_cal_proba, y_test_proba=y_test_proba,
        y_cal=splits["y_cal"].values, y_test=splits["y_test"].values,
    )
    p_t, r_t, th_t = precision_recall_curve(splits["y_test"], y_test_proba)
    np.savez(OUT_DIR / f"pr_curves_{cond_id}_{seed}.npz", precision=p_t, recall=r_t, thresholds=th_t)

    record = dict(
        cond_id=cond_id,
        seed=seed,
        tau_star=tau_star,
        train_sec=train_sec,
        scale_pos_weight=spw,
        n_features=X_train_feat.shape[1],
        **metrics,
    )
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_long_csv(records, path):
    pd.DataFrame(records).to_csv(path, index=False)


def write_markdown_report(records, path):
    df = pd.DataFrame(records)
    metrics = ["pr_auc", "brier", "f1", "accuracy", "roc_auc"]

    summary = df.groupby("cond_id")[metrics + ["tau_star", "train_sec"]].agg(["mean", "std"])
    n_seeds = df["seed"].nunique()

    lines = [
        "# Benchmark results",
        "",
        f"_{n_seeds} seeds × {df['cond_id'].nunique()} conditions × 16-cell grid × 5-fold CV._",
        "",
        "## Conditions",
        "",
    ]
    for cid in sorted(df["cond_id"].unique()):
        lines.append(f"- **{cid}** — {CONDITION_DESCRIPTIONS.get(cid, '(no description)')}")

    lines += [
        "",
        f"## Results (mean ± std across {n_seeds} seeds)",
        "",
    ]

    headers = ["Condition", "PR-AUC", "Brier", "F1@τ*", "Accuracy", "ROC-AUC", "τ* (mean)", "Train (s)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for cid in sorted(df["cond_id"].unique()):
        row = [f"**{cid}**"]
        for m in metrics:
            mu = summary.loc[cid, (m, "mean")]
            sd = summary.loc[cid, (m, "std")]
            row.append(f"{mu:.4f} ± {sd:.4f}")
        row.append(f"{summary.loc[cid, ('tau_star', 'mean')]:.3f}")
        row.append(f"{summary.loc[cid, ('train_sec', 'mean')]:.0f}")
        lines.append("| " + " | ".join(row) + " |")

    by_pr = summary["pr_auc"].sort_values("mean", ascending=False)
    best_cid = by_pr.index[0]
    best_mu = by_pr.iloc[0]["mean"]
    best_sd = by_pr.iloc[0]["std"]

    lines += [
        "",
        "## Conclusion",
        "",
        f"Top condition by PR-AUC: **{best_cid}** ({CONDITION_DESCRIPTIONS.get(best_cid)}) "
        f"at {best_mu:.4f} ± {best_sd:.4f}.",
        "",
        f"Conditions within 1 std of the top (statistically tied at n={n_seeds}):",
    ]
    for cid, row in by_pr.iterrows():
        if row["mean"] >= best_mu - best_sd:
            lines.append(f"- {cid}: {row['mean']:.4f} ± {row['std']:.4f}")

    lines += [
        "",
        f"_n={n_seeds} seeds; ΔPR-AUC smaller than ~1 std is noise. With n={n_seeds} "
        "the std is itself noisy (~50% relative error). Treat narrowly-separated "
        "conditions as undecided rather than ranked._",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"CSV: {CSV_PATH}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Seeds: {SEEDS}")
    print(f"Conditions: {[c['id'] for c in CONDITIONS]}")

    print("\nLoading features...")
    X, y = load_data(CSV_PATH)
    print(f"Shape: {X.shape}, positive rate: {y.mean():.3f}")

    records = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        set_global_seed(seed)
        splits = make_splits(X, y, seed=seed)

        ae_cache = {}
        flavors_needed = {c["ae_weighted"] for c in CONDITIONS if c["use_ae"]}
        for weighted in flavors_needed:
            label = "weighted" if weighted else "class-blind"
            t0 = time.time()
            ae_cache[weighted] = train_ae(
                splits["X_train_scaled"], splits["y_train"],
                weighted=weighted, seed=seed,
            )
            print(f"  AE ({label}) trained in {time.time() - t0:.0f}s")

        for cfg in CONDITIONS:
            print(f"  Running {cfg['id']} ({CONDITION_DESCRIPTIONS[cfg['id']]})...", flush=True)
            rec = run_condition(cfg, seed, ae_cache, splits)
            print(
                f"    PR-AUC={rec['pr_auc']:.4f}  Brier={rec['brier']:.4f}  "
                f"F1@τ*={rec['f1']:.4f}  τ*={rec['tau_star']:.3f}  train={rec['train_sec']:.0f}s"
            )
            records.append(rec)

    write_long_csv(records, OUT_DIR / "results_long.csv")
    write_markdown_report(records, OUT_DIR / "report.md")
    print(f"\nDone. {len(records)} records written.")
    print(f"  CSV:    {OUT_DIR / 'results_long.csv'}")
    print(f"  Report: {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
