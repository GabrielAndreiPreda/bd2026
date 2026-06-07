"""4x4 dim-reduction x classifier benchmark for credit-default prediction.

Runs every pairing of {AE, PCA, UMAP, LDA} x {XGBoost, LogReg, KNN, SVM} on the
same train/cal/test split with the same evaluation protocol, then writes the
results as a 4x4 matrix.

Configuration (locked by user choices):
  - 16 cells, no raw-feature baseline
  - Single seed = 42
  - Original training variant only (no SMOTE, no class reweighting)
  - No PAY_0 concat (clean dim-reduction comparison)
  - Each classifier uses its team's grid (XGB: 16 cells, KNN: 20 cells,
    LogReg/SVM: fixed hyperparameters)

Requires: xgboost, scikit-learn, tensorflow, umap-learn, pandas, numpy.
  pip install xgboost umap-learn

Usage:
    python benchmark.py
"""

import subprocess
import sys

try:
    from umap import UMAP  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "umap-learn"])

import json
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from umap import UMAP  # noqa: E402
from xgboost import XGBClassifier


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
SEED = 42

DIMREDS = ["ae", "pca", "umap", "lda"]
CLFS = ["xgb", "lr", "knn", "svm"]

XGB_GRID = {
    "n_estimators":     [200, 400],
    "max_depth":        [4, 6],
    "learning_rate":    [0.05, 0.1],
    "min_child_weight": [1, 3],
}

KNN_GRID = {
    "n_neighbors": [5, 11, 15, 21, 31],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"],
}

DIMRED_LABELS = {
    "ae":   "AE (10 latents)",
    "pca":  "PCA (95% var)",
    "umap": "UMAP (2D)",
    "lda":  "LDA (1D)",
}

CLF_LABELS = {
    "xgb": "XGBoost",
    "lr":  "LogReg",
    "knn": "KNN",
    "svm": "SVM (RBF)",
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
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path):
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

    return dict(
        y_train=y_train.reset_index(drop=True),
        y_cal=y_cal.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        X_train_scaled=X_train_scaled,
        X_cal_scaled=X_cal_scaled,
        X_test_scaled=X_test_scaled,
        feature_names=list(X_train.columns),
    )


# ---------------------------------------------------------------------------
# Dim-reduction factory
# ---------------------------------------------------------------------------

class AEReducer:
    """Keras encoder adapted to sklearn's .transform interface."""

    def __init__(self, encoder):
        self.encoder = encoder

    def transform(self, X):
        return self.encoder.predict(X, verbose=0).astype(np.float32)


def transform_all(reducer, splits):
    return tuple(
        np.asarray(reducer.transform(splits[k]), dtype=np.float32)
        for k in ("X_train_scaled", "X_cal_scaled", "X_test_scaled")
    )


def _build_ae(input_dim, seed):
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


def _train_ae(X_scaled, seed):
    autoencoder, encoder = _build_ae(X_scaled.shape[1], seed)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )
    return encoder


def fit_reducer(name, X_scaled, y_train, seed):
    if name == "ae":
        encoder = _train_ae(X_scaled, seed)
        return AEReducer(encoder)
    if name == "pca":
        return PCA(n_components=0.95, random_state=seed).fit(X_scaled)
    if name == "umap":
        return UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        ).fit(X_scaled)
    if name == "lda":
        return LinearDiscriminantAnalysis(n_components=1).fit(X_scaled, y_train)
    raise ValueError(f"Unknown dim-reduction: {name!r}")


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def fit_classifier(name, Z_train, y_train, seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    if name == "xgb":
        base = XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            device="cuda",
            tree_method="hist",
        )
        gs = GridSearchCV(base, XGB_GRID, scoring="average_precision", cv=cv, n_jobs=1, verbose=0)
        gs.fit(Z_train, y_train)
        return gs.best_estimator_, dict(gs.best_params_)

    if name == "lr":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(Z_train, y_train)
        return clf, {}

    if name == "knn":
        base = KNeighborsClassifier()
        gs = GridSearchCV(base, KNN_GRID, scoring="average_precision", cv=cv, n_jobs=-1, verbose=0)
        gs.fit(Z_train, y_train)
        return gs.best_estimator_, dict(gs.best_params_)

    if name == "svm":
        clf = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            cache_size=2000,
            random_state=seed,
        )
        clf.fit(Z_train, y_train)
        return clf, {}

    raise ValueError(f"Unknown classifier: {name!r}")


# ---------------------------------------------------------------------------
# Threshold tuning and evaluation
# ---------------------------------------------------------------------------

def f1_optimal_threshold(y_cal, proba_cal):
    precisions, recalls, thresholds = precision_recall_curve(y_cal, proba_cal)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx])


def evaluate(y_true, proba, tau):
    pred = (proba >= tau).astype(int)
    return dict(
        pr_auc=float(average_precision_score(y_true, proba)),
        brier=float(brier_score_loss(y_true, proba)),
        f1=float(f1_score(y_true, pred)),
        accuracy=float(accuracy_score(y_true, pred)),
        roc_auc=float(roc_auc_score(y_true, proba)),
    )


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

def run_cell(dimred_name, clf_name, features, splits, seed):
    Z_train, Z_cal, Z_test = features

    t0 = time.time()
    clf, best_params = fit_classifier(clf_name, Z_train, splits["y_train"], seed)
    fit_sec = time.time() - t0

    proba_cal = clf.predict_proba(Z_cal)[:, 1]
    tau = f1_optimal_threshold(splits["y_cal"], proba_cal)

    proba_test = clf.predict_proba(Z_test)[:, 1]
    metrics = evaluate(splits["y_test"], proba_test, tau)

    p_t, r_t, th_t = precision_recall_curve(splits["y_test"], proba_test)
    np.savez(
        OUT_DIR / f"pr_curves_{dimred_name}_{clf_name}.npz",
        precision=p_t, recall=r_t, thresholds=th_t,
        proba_test=proba_test, y_test=splits["y_test"].values,
    )
    if best_params:
        with open(OUT_DIR / f"best_params_{dimred_name}_{clf_name}.json", "w") as f:
            json.dump({str(k): v for k, v in best_params.items()}, f, indent=2)

    return dict(
        dimred=dimred_name,
        clf=clf_name,
        n_features=Z_train.shape[1],
        tau=tau,
        fit_sec=fit_sec,
        **metrics,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pivot(df, value):
    return df.pivot(index="dimred", columns="clf", values=value).reindex(index=DIMREDS, columns=CLFS)


def write_matrices(records, out_dir):
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "results_long.csv", index=False)

    for metric, fname in [
        ("pr_auc",   "matrix_pr_auc.csv"),
        ("f1",       "matrix_f1.csv"),
        ("brier",    "matrix_brier.csv"),
        ("roc_auc",  "matrix_roc_auc.csv"),
        ("accuracy", "matrix_accuracy.csv"),
        ("fit_sec",  "matrix_fit_sec.csv"),
        ("tau",      "matrix_tau.csv"),
    ]:
        _pivot(df, metric).to_csv(out_dir / fname, float_format="%.4f")


def plot_heatmap(records, metric, title, path, fmt=".4f", cmap="viridis"):
    df = pd.DataFrame(records)
    matrix = _pivot(df, metric)
    row_labels = [DIMRED_LABELS[d] for d in matrix.index]
    col_labels = [CLF_LABELS[c] for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix.values,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        cbar_kws={"label": metric},
    )
    ax.set_title(title)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Dim. reduction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"CSV: {CSV_PATH}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Seed: {SEED}")
    print(f"Grid: {len(DIMREDS)} dim-reductions x {len(CLFS)} classifiers = {len(DIMREDS) * len(CLFS)} cells")

    set_global_seed(SEED)

    print("\nLoading features...")
    X, y = load_data(CSV_PATH)
    print(f"Shape: {X.shape}, positive rate: {y.mean():.3f}")

    splits = make_splits(X, y, seed=SEED)
    print(f"Train: {splits['X_train_scaled'].shape}, Cal: {splits['X_cal_scaled'].shape}, Test: {splits['X_test_scaled'].shape}")

    records = []
    for dimred_name in DIMREDS:
        t_red = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            reducer = fit_reducer(dimred_name, splits["X_train_scaled"], splits["y_train"], seed=SEED)
            features = transform_all(reducer, splits)
        print(f"\n=== {DIMRED_LABELS[dimred_name]} fitted in {time.time() - t_red:.0f}s ===")

        for clf_name in CLFS:
            print(f"  -> {CLF_LABELS[clf_name]}...", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                rec = run_cell(dimred_name, clf_name, features, splits, seed=SEED)
            print(
                f"     PR-AUC={rec['pr_auc']:.4f}  F1@tau*={rec['f1']:.4f}  "
                f"Brier={rec['brier']:.4f}  tau*={rec['tau']:.3f}  fit={rec['fit_sec']:.0f}s"
            )
            records.append(rec)

    write_matrices(records, OUT_DIR)
    plot_heatmap(records, "pr_auc",  "PR-AUC",         OUT_DIR / "heatmap_pr_auc.png")
    plot_heatmap(records, "f1",      "F1 at tau*",     OUT_DIR / "heatmap_f1.png")
    plot_heatmap(records, "brier",   "Brier score",    OUT_DIR / "heatmap_brier.png", cmap="viridis_r")
    plot_heatmap(records, "fit_sec", "Fit time (s)",   OUT_DIR / "heatmap_fit_sec.png", fmt=".0f", cmap="rocket_r")

    print(f"\nDone. {len(records)} cells, outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
