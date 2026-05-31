# Project memory — credit card default prediction (AE + XGBoost)

Last updated: 2026-05-30

This document is the shared working memory for the project. It captures what
the codebase looked like, what we changed, why, and how the pipeline runs now.
Update it whenever a change lands that future-you (or a teammate) would need to
know about without re-reading every diff.

---

## What the project is

UCI Taiwan credit card default dataset (~30k rows, Apr–Sep 2005 history,
predicting October 2005 default). Binary classification, target prior ≈ 22%
positive.

The model choice is **fixed**: a Keras dense autoencoder feeds a low-dim latent
into XGBoost (GridSearchCV-tuned). All fixes go into preprocessing or feature
representation, never into a different model family.

Files in the repo:
- `credit_card_data.csv` — 30,000 × 35, output of `preprocessing.py` with all
  engineered features already computed. This is what `model.py` reads on
  Kaggle.
- `preprocessing.py` — runs on the original UCI CSV; recodes unknown
  categoricals, engineers payment/billing summary features, writes the CSV
  above.
- `model.py` — loads `credit_card_data.csv`, drops raw temporal columns, log/
  one-hot transforms, trains the AE, encodes into latents, trains XGBoost,
  calibrates, tunes a decision threshold, evaluates.
- `README.md` — original dataset documentation.

---

## Pipeline at a glance (current state)

```
credit_card_data.csv (30,000 × 35, engineered)
  │
  ▼
model.py
  │
  ├─ Compute n_months_over_limit (0–6) from raw BILL_AMTs
  ├─ Drop raw temporal cols + redundant engineered cols
  ├─ Signed-log heavy-tailed monetary cols
  ├─ One-hot SEX / EDUCATION / MARRIAGE (drop_first)
  │
  ├─ Stratified split: train 80% / test 20%
  ├─ Carve 10% calibration slice from train (stratified)
  │   → train ≈ 21,569 rows, cal ≈ 2,397, test ≈ 6,000
  │
  ├─ StandardScaler (fit on train only) → train/cal/test scaled
  │
  ├─ Autoencoder 18 → 24 → 16 → 10 → 16 → 24 → 18, MSE loss
  │   with sample_weight upweighting defaulters (~3.52×)
  │
  ├─ Encode train/cal/test → 10-dim latents
  ├─ Concat raw scaled PAY_0 → 11-dim feature matrices
  │
  ├─ XGBoost via GridSearchCV (scoring=average_precision, cv=5)
  │   with scale_pos_weight ≈ 3.52
  │
  ├─ CalibratedClassifierCV(isotonic, cv='prefit') fit on X_cal
  ├─ F1-optimal threshold tuned on calibrated cal-slice predictions
  │
  └─ Test eval: PR-AUC, Brier, F1@τ*, accuracy, ROC-AUC
```

---

## What we had before

The original pipeline (commits `8b533c4` initial + `8e1d47e` GPU) had:

- `preprocessing.py` recoded EDUCATION unknowns {0, 5, 6} → 4, but **missed
  MARRIAGE's undocumented 0s** (54 rows).
- `model.py` dropped the raw temporal columns and the obviously-redundant
  engineered ones (Pays_amts_total, Utilization_1, Risk_score), then
  StandardScaler-normalized the ~17 remaining features.
- The autoencoder was trained **unsupervised on all 24,000 train rows
  (class-blind)**, MSE loss, 50 epochs with EarlyStopping(patience=5).
- XGBoost used `scale_pos_weight ≈ 3.52` to handle class imbalance; GridSearchCV
  scored on **ROC-AUC**; the final prediction used a default threshold of 0.5.
- No probability calibration. No threshold tuning. No PR-AUC reporting.

---

## What we changed and why

### Stage 1 — Data hygiene (`preprocessing.py`)

| Change | Where | Why |
|---|---|---|
| `MARRIAGE.replace(0, 3)` | next to the EDUCATION recode (L121) | 54 undocumented zeros. The README only lists {1,2,3}; folding 0 into "others" matches the EDUCATION treatment. |
| `df.drop_duplicates().reset_index(drop=True)` | right after `df.drop("ID", ...)` (L125) | 35 exact duplicate rows. Must happen before train/test split or the same duplicate can land in both. |

### Stage 2 — Pre-scaling transforms (`model.py`)

| Change | Where | Why |
|---|---|---|
| `n_months_over_limit` (0–6) | L38–39, before raw BILL_AMTs are dropped | The over-limit signal would otherwise be destroyed by `cols_to_drop`. ~7% of rows have at least one over-limit month. |
| `signed_log1p` on `LIMIT_BAL, Bill_mean, Bill_std, Bill_max, Pays_amts_mean, Utilization_mean` | L62–68 | PAY_AMT skew was up to 30; after StandardScaler one row hit z=72.9 and dominated MSE loss. After log transform, max \|z\| drops to ~5–8. **`Utilization_mean` was added to this list in a second pass** after we saw it still hitting z=14.22 (originally classified as "bounded ratio" — wrong; over-limit pushes it to 5.36). |
| `signed_log1p` defined locally | L64–65 | Handles negative bills (refunds, ~2% of rows) without dropping sign information. |

### Stage 3 — Categorical handling (`model.py`)

| Change | Where | Why |
|---|---|---|
| `pd.get_dummies(SEX, EDUCATION, MARRIAGE, drop_first=True)` | L73–78 | MSE on ordinal-coded nominals is meaningless (predicting MARRIAGE=2.3 when truth is 2 has no semantics). One-hot adds 3 net features (SEX 0, EDUCATION +2, MARRIAGE +1). |
| `PAY_0` kept numeric (not bucketed, not one-hot) | unchanged | PAY_0 is genuinely ordinal (more delay = worse); it's the single strongest predictor. Spreading it across dummies would waste latent-dim capacity. |
| Architecture comment updated to `18 → 24 → 16 → 10 → 16 → 24 → 18` | L120 | Input dim went from ~14 to 18 after the dummies + over-limit indicator. Architecture kept the same; only the comment reflects the new dim. |

### Stage 4 — Imbalance handling (current iteration, model.py)

This is row 0 of the benchmark: **instrumented baseline** with the cheap fixes
in place. The bigger imbalance options (SMOTE, one-class AE, BalancedBagging,
focal loss) are **not yet implemented** — they're on the menu for the next
iteration.

| Change | Where | Why |
|---|---|---|
| 10% calibration slice carved from train (stratified) | L95–100 | Held out from BOTH the autoencoder training and GridSearchCV folds; used downstream for calibration + threshold tuning. Test set stays untouched. |
| `autoencoder.fit(..., sample_weight=...)` upweighting defaulters by ~3.52× (**B.4**) | L152–164 | The AE was class-blind under MSE — it preserved majority-class structure preferentially. Weighting makes it pay attention to defaulter patterns in latent space. Free fix (one extra line). |
| Concat scaled `PAY_0` with latents → 11-dim XGBoost input (**C.3**) | L197–203 | The AE bottleneck was probably losing PAY_0 signal (confirmed by the 2-neuron ≈ 10-neuron AUC finding — the AE is over-parameterized and effectively does PCA-equivalent compression). Letting XGBoost see PAY_0 directly bypasses the loss. |
| `GridSearchCV(scoring="average_precision")` instead of `"roc_auc"` | L239 | ROC-AUC is forgiving for imbalanced data. PR-AUC is the honest ranking metric for a 22%-positive problem. The grid now picks hyperparameters that actually help on the minority class. |
| `CalibratedClassifierCV(isotonic, cv='prefit')` on cal slice (**D.2**) | L256–257 | `scale_pos_weight` inflates `predict_proba` outputs; raw probabilities mean nothing. Isotonic regression on the held-out slice maps them back to honest probabilities so Brier score is meaningful. |
| F1-optimal threshold from PR curve on cal predictions (**D.1**) | L263–268 | Threshold 0.5 with `scale_pos_weight=3.52` is incoherent — the F1-optimal cutoff for a 22%-positive problem is rarely 0.5 anyway. Tuning on held-out cal predictions avoids test-set leakage. |
| New test metrics: PR-AUC, Brier, F1@τ* | L274–278 | Three numbers, not one. PR-AUC for ranking, Brier for calibration, F1@τ* for the binary decision quality at the operational threshold. Old ROC-AUC kept for reference only. |
| Feature importance labels include `PAY_0 (raw)` | L311–320 | 11 features now, not 10. |

### Verification diagnostics added along the way

- `print(pd.DataFrame(X_train_scaled, ...).abs().max().sort_values(...))` at
  L116–117 — sanity check that signed-log tamed the heavy tails.
- Per-feature reconstruction MSE bar chart at L181–190 — diagnoses whether the
  AE is spending its capacity on a handful of high-variance features. Before
  the log fix, monetary columns dominated by >10×; after, the distribution is
  roughly uniform.

---

## Things we noticed along the way

- **AE is over-parameterized.** A 2-neuron bottleneck (with 150 epochs) achieves
  almost the same AUC as the 10-neuron bottleneck. The data's effective
  dimensionality is very low, consistent with the high multicollinearity in the
  monetary columns. The AE is doing PCA-equivalent work, not learning a useful
  manifold.
- **MARRIAGE features look important** in the feature-importance / Jacobian
  views — but partly because rare one-hot dummies (~1% support) have unusually
  large amplitude after StandardScaler (z up to ~9.65 for MARRIAGE_3). The
  signal is real but the magnitude is inflated by scaling. A direct empirical
  check (`df.groupby("MARRIAGE")["default_payment"].mean()`) is the honest
  verification.
- **AGE has the highest per-feature reconstruction MSE.** Not a bug — AGE is
  relatively uncorrelated with payment behavior, so the bottleneck can't
  predict it from the other dims. The AE faithfully gives up.
- **`PAY_0`'s strong correlation with target (~0.32) is not leakage.** PAY_0 is
  September status; target is October default. It's a legitimate pre-target
  feature and the README is correct in calling it the strongest single
  predictor.

---

## Benchmark harness

`/mnt/e/Projects/Master/BD/benchmark.py` runs 7 conditions × 3 seeds with a
reduced 16-cell XGBoost grid and writes a markdown report + long CSV to
`bench_out/`. Conditions:

| ID | What it is | What it isolates |
|---|---|---|
| C0 | XGB-only, `scale_pos_weight=auto` | XGB-only baseline with imbalance correction |
| C1 | XGB-only, `scale_pos_weight=1` | XGB-only without imbalance correction |
| C2 | AE+XGB current (B.4 + C.3) | The instrumented baseline in model.py |
| C3 | AE+XGB without B.4 (class-blind AE) | Does weighted AE loss contribute? |
| C4 | AE+XGB without C.3 (latents only) | Does PAY_0 concat contribute? |
| C5 | AE+XGB + SMOTE on latents | Latent-space oversampling vs `scale_pos_weight` |
| C6 | AE+XGB + Borderline-SMOTE | Boundary-aware oversampling variant |

Run with `python benchmark.py`. Expected wall clock: ~2–3 hours on GPU with the
reduced grid. Outputs:
- `bench_out/results_long.csv` — one row per (condition, seed) with metrics
- `bench_out/report.md` — mean ± std table + conclusion
- `bench_out/best_params_<cond>_<seed>.json` — chosen grid cell per run
- `bench_out/cal_diag_<cond>_<seed>.npz` — probabilities for offline reliability diagrams
- `bench_out/pr_curves_<cond>_<seed>.npz` — test PR curves

**Results: not yet recorded — append a "Benchmark results (date)" section
below this one after the first run.**

`benchmark.py` is the canonical comparator; `model.py` stays as the
human-readable single-condition reference. If they diverge in feature
engineering, the benchmark wins.

## What's NOT been done (the rest of the imbalance menu)

A.3 SMOTE and A.5 Borderline-SMOTE on latents are now wired in as benchmark
conditions C5 and C6. Still queued for future benchmarks:

1. **C.1 — One-class AE on majority only, reconstruction error as feature.**
   Bigger architectural rearrangement; the theoretically correct use of an AE
   for imbalanced classification.
2. **E.1 — BalancedBaggingClassifier wrapping XGBoost** with a reduced grid
   (16–32 cells; the bagging multiplies cost).
3. **B.3 — Focal loss as custom XGBoost objective.** Last because high effort
   and unlikely to dominate a well-tuned SMOTE baseline.

Do not try:
- **A.6 SMOTE-ENN** — ENN deletes boundary majority rows, which on UCI default
  is where the signal lives.
- **Pre-AE plain SMOTE** on the 18-dim feature space with one-hots — produces
  fractional dummies that corrupt SEX/EDUCATION/MARRIAGE. Use SMOTENC if
  pre-AE is ever needed.
- **`scale_pos_weight=3.52` simultaneously with a 50/50 sampler** —
  double-correction.
- **C.2 class-conditional AEs** with the current architecture size — only ~5k
  defaulters; the minority-class AE would overfit.

For benchmarking discipline:
- Every condition uses the same train/test split (`random_state=42`), the same
  XGBoost grid, the same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`,
  and the same calibration + threshold-tuning protocol.
- Samplers go inside `imblearn.pipeline.Pipeline` so they only see the training
  fold within each CV iteration (no leakage into the validation fold).
- Reuse the cached AE across conditions where the AE doesn't change (most
  post-AE samplers); only retrain it for B.4 variants, one-class AE, or
  pre-AE samplers.
- Run 5 outer seeds and report mean ± std. A ΔPR-AUC smaller than ~1 std is
  noise.

---

## Operational notes

- Kaggle environment: `model.py` reads from
  `/kaggle/input/datasets/gabrielpredaz/creditcarddata/credit_card_data.csv`.
  Locally, the same file sits at `./credit_card_data.csv` (7.7 MB).
- GPU: XGBoost uses `device="cuda"` + `tree_method="hist"`. The training
  feature matrix is also moved to GPU via `cupy` to avoid per-fit copies.
  Numpy versions are kept for sklearn calibration / threshold tuning, which
  can't accept cupy arrays.
- TensorFlow is needed for the Keras autoencoder; GridSearchCV with
  `n_jobs=1` is required when XGBoost uses the GPU (parallelism happens
  inside XGBoost).
- Sklearn 1.6+ deprecation warning: `CalibratedClassifierCV(cv='prefit')` is
  flagged; replacement uses `FrozenEstimator`. Still works; switch when needed.
- Team project — `preprocessing.py` mentions "Member(s) 2 and 3" doing modeling
  later, which is why the engineered features stay in the saved CSV instead of
  being computed inline by `model.py`.

---

## How to read the new test output

```
PR-AUC (test):                       0.55ish — ranking quality on minority
Brier score (test):                  0.13–0.17ish — calibration quality (lower = better)
F1 @ tau*=0.XX (test):               binary decision quality at the operational threshold
Accuracy @ tau* (test):              for reference; misleading on imbalanced data
AUC-ROC (test, for reference):       legacy; do not headline this number
```

τ\* will typically land between 0.2 and 0.35. If it's near 0.5 something
unusual happened (calibration failed, sample is small).
