# Report Agent Brief — BigData2026 Credit Default Project

## Your task

Produce a detailed project report covering: dataset and preprocessing choices,
model architecture and training decisions, ablation benchmark results, metric
interpretation, limitations, and recommendations. All raw data you need is
already on disk — no Kaggle runs required.

---

## Working directory

`/mnt/e/Projects/Master/BD/`

---

## What the project is

UCI Taiwan credit card default dataset (~30k rows, Apr–Sep 2005 payment history,
predicting October 2005 default). Binary classification, ~22% positive class.

**Fixed model architecture (project constraint):** Keras dense autoencoder
compresses features to a low-dim latent space → XGBoost classifier trained on
the latents. No changing the classifier family.

---

## Key files to read first

| File | What it contains |
|---|---|
| `MEMORY.md` | Full project history — every decision made, why, and what was observed. Read this before anything else. |
| `bench_out/report.md` | Auto-generated benchmark summary with mean ± std table and conclusion. |
| `bench_out/results_long.csv` | Raw results: one row per (condition × seed), columns: `cond_id, seed, tau_star, train_sec, scale_pos_weight, n_features, pr_auc, brier, f1, accuracy, roc_auc`. |
| `bench_out/best_params_<C>_<seed>.json` | Winning XGBoost grid cell per condition/seed (21 files). |
| `bench_out/pr_curves_<C>_<seed>.npz` | Arrays `precision`, `recall`, `thresholds` for test PR curves (21 files). Load with `numpy.load`. |
| `bench_out/cal_diag_<C>_<seed>.npz` | Arrays `y_cal_proba`, `y_test_proba`, `y_cal`, `y_test` for reliability diagrams (21 files). |
| `preprocessing.py` | Full preprocessing pipeline with rationale in comments. |
| `model.py` | Single-condition reference implementation (AE+XGB, readable). |
| `benchmark.py` | The A/B harness — defines all 7 conditions in `CONDITIONS` list. |

---

## Benchmark conditions (7 total, 3 seeds each)

| ID | Description | What it isolates |
|---|---|---|
| C0 | XGB-only, `scale_pos_weight=auto` (~3.52) | XGB baseline with class-imbalance correction |
| C1 | XGB-only, `scale_pos_weight=1` | XGB baseline without imbalance correction |
| C2 | AE+XGB, weighted AE + PAY_0 concat | The full current model (both B.4 and C.3 active) |
| C3 | AE+XGB, class-blind AE + PAY_0 concat | Isolates whether weighted AE training (B.4) helps |
| C4 | AE+XGB, weighted AE, latents only | Isolates whether PAY_0 concat (C.3) helps |
| C5 | AE+XGB + SMOTE on latents, spw=1 | Latent-space oversampling vs scale_pos_weight |
| C6 | AE+XGB + Borderline-SMOTE on latents, spw=1 | Boundary-aware oversampling variant |

**AE architecture:** 18 → 24 → 16 → 10 → 16 → 24 → 18 (encoding_dim=10, Dropout 0.2)
**XGB grid:** 16 cells (n_estimators×max_depth×learning_rate×min_child_weight), 5-fold CV, scored on average_precision.

---

## Benchmark results (copy for the report)

| Condition | PR-AUC | Brier | F1@τ* | Accuracy | ROC-AUC | τ* | Train(s) |
|---|---|---|---|---|---|---|---|
| C0 | 0.5281 ± 0.0065 | 0.1350 ± 0.0011 | 0.5465 ± 0.0054 | 0.7959 ± 0.0155 | 0.7820 ± 0.0077 | 0.336 | 31 |
| C1 | 0.5280 ± 0.0107 | 0.1349 ± 0.0011 | 0.5463 ± 0.0061 | 0.7958 ± 0.0150 | 0.7831 ± 0.0069 | 0.343 | 31 |
| C2 | 0.5217 ± 0.0046 | 0.1362 ± 0.0014 | 0.5398 ± 0.0045 | 0.7680 ± 0.0091 | 0.7772 ± 0.0089 | 0.280 | 31 |
| C3 | 0.5235 ± 0.0059 | 0.1366 ± 0.0013 | 0.5367 ± 0.0055 | 0.7860 ± 0.0103 | 0.7764 ± 0.0068 | 0.317 | 31 |
| C4 | 0.5123 ± 0.0027 | 0.1382 ± 0.0014 | 0.5367 ± 0.0066 | 0.7825 ± 0.0133 | 0.7734 ± 0.0075 | 0.319 | 31 |
| C5 | 0.5177 ± 0.0019 | 0.1372 ± 0.0009 | 0.5389 ± 0.0041 | 0.7825 ± 0.0070 | 0.7743 ± 0.0050 | 0.316 | 44 |
| C6 | 0.4933 ± 0.0029 | 0.1398 ± 0.0011 | 0.5325 ± 0.0051 | 0.7581 ± 0.0249 | 0.7700 ± 0.0052 | 0.291 | 54 |

**Primary metric is PR-AUC** (more honest than ROC-AUC for 22%-positive data).
Brier score measures calibration quality (lower = better).
F1@τ* is the binary decision quality at the F1-optimal threshold tuned on a held-out calibration slice.

---

## model.py single-condition results (Kaggle, encoding_dim=4)

These are from the most recent Kaggle notebook run (v13), which used `encoding_dim=4`
(a deviation from the benchmark's 10 — see caveats below):

```
Best XGB params: lr=0.05, max_depth=4, n_estimators=200,
                 colsample_bytree=1.0, gamma=0.1, subsample=0.8, min_child_weight=1
Best CV PR-AUC:  0.5425
PR-AUC (test):   0.5152
Brier score:     0.1390
F1 @ τ*=0.399:  0.5176
Accuracy:        0.787
ROC-AUC:         0.762
τ*:              0.399
```

Classification report (test set, 5,993 rows):
- No Default: precision=0.86, recall=0.86, F1=0.86 (4,667 samples)
- Default:     precision=0.52, recall=0.52, F1=0.52 (1,326 samples)

---

## Important caveats and context

**1. encoding_dim discrepancy.**
The Kaggle notebook (`model.py`) was manually changed to `encoding_dim=4` at some
point before this project session. The benchmark harness uses `encoding_dim=10`
(the original value). The benchmark C2 condition (encoding_dim=10) achieves
PR-AUC=0.5217 vs the notebook's 0.5152 with dim=4. The 4-dim bottleneck is worse.
Recommendation: revert to 10 in the notebook.

**2. AE is not contributing positively.**
The benchmark shows plain XGB (C0/C1) outperforms every AE+XGB variant. The gap
is small but consistent across all 3 seeds. The AE is over-parameterized for this
dataset — it effectively performs PCA-equivalent compression on already low-
dimensional data (high multicollinearity in monetary features). This is confirmed
by the finding that a 2-neuron bottleneck achieves near-identical AUC to a 10-neuron
one (noted in MEMORY.md). The model choice is a project constraint; present this
finding honestly in the report.

**3. C4 < C2 → PAY_0 concat is load-bearing.**
C4 (latents only, no PAY_0 concat) is the worst AE variant. The AE bottleneck loses
PAY_0 signal. Concatenating raw PAY_0 directly (C.3) partially compensates but doesn't
recover to XGB-only level.

**4. B.4 (weighted AE) has marginal effect.**
C2 vs C3 (weighted AE vs class-blind AE) differ by only 0.0018 PR-AUC (within noise
at n=3 seeds). The weighted AE training is free to keep but isn't the deciding factor.

**5. Borderline-SMOTE hurts (C6).**
C6 is the worst overall condition. Boundary-aware oversampling on latents degrades
performance. C5 (plain SMOTE on latents) is neutral-to-slightly-negative.

**6. scale_pos_weight doesn't matter much for XGB-only.**
C0 vs C1 are statistically indistinguishable (0.0001 PR-AUC difference). The model
is robust to the imbalance correction choice when operating on raw features.

**7. Calibration worked.**
τ* lands between 0.28–0.34 for AE conditions and 0.34–0.37 for XGB-only, well away
from the default 0.5. The isotonic calibrator is doing its job.

---

## Teammate model results

These complement the main AE+XGB benchmark. All use identical train/test/cal splits (seed=42).
Outputs are in `teammate1/results/`, `teammate2/results/`, `teammate3/results/`.

---

### Teammate 1 — Logistic Regression (`teammate1/log_regression.py`)

6 conditions: original / balanced_cw (class_weight='balanced') / smote × full features / PCA(95%)

| Condition | PR-AUC | Brier | F1@t* | Accuracy | ROC-AUC | t* | Features |
|---|---|---|---|---|---|---|---|
| original | 0.5087 | 0.1423 | 0.5183 | 0.7900 | 0.7458 | 0.319 | 18 |
| original_PCA | 0.5031 | 0.1428 | 0.5277 | 0.7887 | 0.7470 | 0.319 | 12 |
| balanced_cw | 0.4969 | 0.1960 | 0.5272 | 0.7783 | 0.7495 | 0.577 | 18 |
| balanced_cw_PCA | 0.4941 | 0.1969 | 0.5275 | 0.7781 | 0.7497 | 0.593 | 12 |
| smote | 0.4870 | 0.1937 | 0.4965 | 0.7703 | 0.7227 | 0.543 | 18 |
| smote_PCA | 0.4851 | 0.2002 | 0.4945 | 0.7727 | 0.7269 | 0.522 | 12 |

**Best: original (PR-AUC=0.5087)**. Classification report on test set (t*=0.319):
- No Default: precision=0.86, recall=0.88, F1=0.87 (4,667)
- Default: precision=0.54, recall=0.50, F1=0.52 (1,326)

Key findings:
- class_weight='balanced' and SMOTE both hurt PR-AUC (inflated Brier: 0.196 vs 0.142 baseline)
- PCA makes no meaningful difference (within 0.006 PR-AUC of full-feature equivalent)
- Top positive coefficients: Pay_delay_mean (0.531), PAY_0 (0.388), Pay_delay_std (0.286)
- Top negative coefficients: Bill_std (-0.221), Bill_mean (-0.192), EDUCATION_4 (-0.130)
- All 6 LR conditions fall below the XGB-only benchmark (C0=0.5281)

Outputs: `results.csv`, `report.md`, `pr_curves.png`, `confusion_matrix_best.png`,
`pr_curve_<variant>.npz` (x3), `coefficients_<variant>.csv` (x3)

---

### Teammate 2 — UMAP + KNN (`teammate2/UMP_AND_KNN.py`)

4 conditions: original / smote × full features / 2D UMAP (n_neighbors=15, min_dist=0.1)
KNN tuning: GridSearchCV, 3-fold CV, scoring=F1; grid covers n_neighbors, weights, metric.

| Condition | PR-AUC | Brier | F1@t* | Accuracy | ROC-AUC | t* | k |
|---|---|---|---|---|---|---|---|
| original | 0.4999 | 0.1429 | 0.5099 | 0.7681 | 0.7502 | 0.333 | 15 |
| original_UMAP | 0.4879 | 0.1441 | 0.5040 | 0.7914 | 0.7470 | 0.381 | 21 |
| smote | 0.4749 | 0.1722 | 0.0903 | 0.7849 | 0.7435 | 1.000 | 15 |
| smote_UMAP | 0.4559 | 0.1977 | 0.4833 | 0.6975 | 0.7314 | 0.468 | 31 |

**Best: original (PR-AUC=0.4999)**. Optimal KNN params: k=15, manhattan, uniform weights.

Key findings:
- UMAP compression to 2D costs ~0.012 PR-AUC vs full 18 features
- SMOTE-trained KNN exhibits degenerate threshold (t*=1.0, F1=0.09): the model trained on
  balanced 50/50 data outputs systematically inflated probabilities on the natural-distribution
  cal set, so no threshold below 1.0 maximizes F1. Rank-based PR-AUC (0.4749) is still valid.
- smote_UMAP avoids the calibration collapse (t*=0.468) but is the weakest overall (PR-AUC=0.4559)
- All KNN conditions fall below LR original and SVM original

UMAP visualization plots: original train coloured by class, Risk_score, Pay_delay_mean;
side-by-side original vs balanced. See `results/umap_*.png`.

Outputs: `results.csv`, `report.md`, `knn_performance_comparison.png`,
`umap_*.png` (4), `confusion_matrix_<condition>.png` (x4)

---

### Teammate 3 — SVM + LDA (`teammate3/svm-lda.py`)

4 conditions: original / balanced x no-LDA / LDA
Classifier: SVC(kernel='rbf', probability=True, C=1.0, gamma='scale').
LDA reduces 18 features to 1 component (binary classification).

| Condition | PR-AUC | Brier | F1@t* | Accuracy | ROC-AUC | t* | Features | Fit(s) |
|---|---|---|---|---|---|---|---|---|
| original | 0.5144 | 0.1418 | 0.5129 | 0.7579 | 0.7127 | 0.152 | 18 | 38 |
| original_LDA | 0.4671 | 0.1468 | 0.5231 | 0.7886 | 0.6874 | 0.148 | 1 | 93 |
| balanced | 0.4730 | 0.1718 | 0.4990 | 0.7983 | 0.7408 | 0.709 | 18 | 215 |
| balanced_LDA | 0.3967 | 0.1969 | 0.4935 | 0.7579 | 0.7070 | 0.656 | 1 | 206 |

**Best: original (PR-AUC=0.5144)**. Classification report on test set (t*=0.152):
- No Default: precision=0.87, recall=0.81, F1=0.84 (4,667)
- Default: precision=0.46, recall=0.58, F1=0.51 (1,326)

Key findings:
- SVM original (rbf, 18 features) is the **strongest single-teammate model** at PR-AUC=0.5144,
  beating LR original (0.5087), KNN original (0.4999), and AE+XGB C2 (0.5217, but still below C0)
- LDA compression to 1D costs ~0.047 PR-AUC (original vs original_LDA); LDA loses discriminative
  information that the rbf kernel can exploit in high-dimensional space
- balanced (SMOTE) hurts both SVM conditions: PR-AUC drops 0.042 (no-LDA) and 0.070 (LDA)
- balanced_LDA is the worst condition across all teammates (PR-AUC=0.3967)
- t* is very low for SVM original (0.152): SVM decision boundary sits near 0.5 raw probability
  after Platt scaling, but the F1-optimal cut on the calibration set is much lower — model is
  overconfident about negatives
- ROC-AUC for SVM (0.713) is notably lower than LR (0.746) and KNN (0.750) despite higher PR-AUC;
  this means SVM ranks the top positives well but is weaker on the tail

Outputs: `results.csv`, `report.md`, `confusion_matrix_<condition>.png` (x4),
`lda_distribution_<condition>.png` (x2), `roc_curve_<condition>.png` (x2)

---

### Cross-model comparison (primary metric: PR-AUC)

| Model | Best condition | PR-AUC | Brier | F1@t* | ROC-AUC |
|---|---|---|---|---|---|
| XGB-only (C0) | scale_pos_weight=auto | **0.5281** | 0.1350 | 0.5465 | 0.7820 |
| AE+XGB (C2) | weighted AE + PAY_0 concat | 0.5217 | 0.1362 | 0.5398 | 0.7772 |
| SVM rbf | original (no SMOTE, no LDA) | 0.5144 | 0.1418 | 0.5129 | 0.7127 |
| Logistic Regression | original (no balancing) | 0.5087 | 0.1423 | 0.5183 | 0.7458 |
| KNN | original (no UMAP) | 0.4999 | 0.1429 | 0.5099 | 0.7502 |
| SVM rbf | balanced (SMOTE) | 0.4730 | 0.1718 | 0.4990 | 0.7408 |
| KNN | original + UMAP | 0.4879 | 0.1441 | 0.5040 | 0.7470 |

Cross-cutting patterns:
- **Original distribution consistently beats SMOTE** across all model families. Brier score
  degrades significantly with SMOTE (0.172–0.197 vs 0.142–0.143 for no-SMOTE).
- **Dimensionality reduction hurts**: LDA loses 0.047, UMAP loses 0.012, PCA loses <0.006.
  The rbf-SVM finding is especially clear: compressing 18D to 1D destroys PR-AUC.
- **XGB dominates**: even the weakest XGB-only condition (C1, PR-AUC=0.5280) outperforms
  every non-XGB model. The AE pipeline cannot beat the XGB baseline despite added complexity.
- **ROC-AUC vs PR-AUC divergence**: SVM has higher PR-AUC than KNN (0.514 vs 0.500) but lower
  ROC-AUC (0.713 vs 0.750). At 22% positive rate, PR-AUC is the reliable primary metric.

---

## Pipeline summary (for the report's Methods section)

```
UCI_Credit_Card.csv (30,000 × 25, raw)
  │
  preprocessing.py
  ├── Recode EDUCATION {0,5,6}→4, MARRIAGE 0→3 (undocumented unknowns)
  ├── Drop ID; remove 35 exact duplicate rows → 29,965 rows
  ├── Engineer: Pay_delay_{mean,max,std}, Bill_{mean,std,max},
  │             Pays_amts_{total,mean}, Utilization_{1,mean},
  │             Risk_score, n_months_over_limit
  ├── Drop raw temporal cols (PAY_2–6, BILL_AMT1–6, PAY_AMT1–6); keep PAY_0
  ├── signed_log1p on LIMIT_BAL, Bill_{mean,std,max}, Pays_amts_mean, Utilization_mean
  │   (tames z-scores from >70 to ~5–8 for autoencoder MSE stability)
  └── One-hot SEX / EDUCATION / MARRIAGE (drop_first) → 18 model features
  │
  credit_card_data.csv (29,965 × 22, ML-ready)
  │
  model.py / benchmark.py
  ├── Drop Pays_amts_total, Utilization_1, Risk_score → 18 training features
  ├── Stratified 80/20 train/test split (seed=42)
  ├── Carve 10% calibration slice from train (seed=42, stratified)
  │   → train ≈ 21,572, cal ≈ 2,397, test ≈ 5,993
  ├── StandardScaler fit on train only
  ├── Autoencoder 18→24→16→10→16→24→18, MSE loss
  │   sample_weight upweights defaulters ×3.52 (B.4)
  ├── Encode train/cal/test → 10-dim latents
  ├── Concat scaled PAY_0 → 11-dim feature matrices (C.3)
  ├── XGBoost via GridSearchCV (scoring=average_precision, 5-fold CV)
  │   scale_pos_weight ≈ 3.52
  ├── CalibratedClassifierCV(isotonic, cv='prefit') on cal slice (D.2)
  └── F1-optimal threshold τ* tuned on calibrated cal predictions (D.1)
```

---

## Data available for teammates (Kaggle dataset: gabrielpredaz/creditcarddata)

| File | Rows | Default rate | Notes |
|---|---|---|---|
| `credit_card_data.csv` | 29,965 | 22.1% | Full ML-ready dataset, custom splitting |
| `train.csv` | 23,972 | 22.1% | 80% split, seed=42, original distribution |
| `train_balanced.csv` | 37,336 | 50.0% | SMOTENC on train.csv (binary one-hots preserved) |
| `test.csv` | 5,993 | 22.1% | 20% split, seed=42, untouched by SMOTE |

---

## Suggested report structure

1. **Introduction** — problem, dataset, class imbalance, project constraints
2. **Data & Preprocessing** — hygiene decisions, feature engineering rationale,
   log-transform necessity (z-score diagnostics), one-hot rationale, dedup/recode
3. **Model Architecture** — AE topology, weighted loss (B.4), PAY_0 concat (C.3),
   calibration (D.2), threshold tuning (D.1)
4. **Experiment Design** — 7 AE+XGB ablation conditions; teammate baselines
   (LR, UMAP+KNN, SVM+LDA); shared train/cal/test splits; evaluation protocol
5. **Results**
   - AE+XGB benchmark table with mean ± std (from `bench_out/report.md`)
   - PR curves (from `.npz` files in `bench_out/`)
   - Calibration reliability diagrams (from `cal_diag_*.npz`)
   - Teammate results tables (from each `teammate*/results/results.csv`)
   - UMAP visualisation plots (from `teammate2/results/umap_*.png`)
   - LDA distribution and ROC plots (from `teammate3/results/`)
6. **Analysis**
   - XGB-only vs AE+XGB finding (AE adds no benefit; PAY_0 bottleneck issue)
   - Cross-model ranking: XGB > SVM > LR > KNN
   - SMOTE consistently hurts calibrated probability quality (Brier degrades 0.03–0.06)
   - Dimensionality reduction: PCA neutral, UMAP small loss, LDA large loss
   - SVM PR-AUC vs ROC-AUC divergence (high PR-AUC but low ROC-AUC)
   - SMOTE calibration collapse in KNN (t*=1.0)
   - encoding_dim=4 vs 10 comparison; scale_pos_weight insensitivity
7. **Limitations** — n=3 seeds (wide std), constrained model choice, AE over-parameterisation,
   no hyperparameter search for SVM or LR (only KNN was grid-searched)
8. **Recommendations** — revert encoding_dim to 10; add reconstruction error as feature;
   if unconstrained, XGB-only on raw features (C0) is the best-performing setup
9. **Conclusion**

---

## How to load the npz files

```python
import numpy as np

# PR curve for condition C2, seed 0
pr = np.load("bench_out/pr_curves_C2_0.npz")
precision, recall = pr["precision"], pr["recall"]

# Calibration diagnostics for C2, seed 0
cal = np.load("bench_out/cal_diag_C2_0.npz")
y_test_proba, y_test = cal["y_test_proba"], cal["y_test"]
```

## How to load best params

```python
import json, glob
params = {f.split("_")[2] + "_" + f.split("_")[3].replace(".json",""):
          json.load(open(f)) for f in glob.glob("bench_out/best_params_*.json")}
```

---

## Python environment

Use `.venv/bin/python` (uv-managed, has pandas, numpy, sklearn, matplotlib, seaborn).
The Kaggle CLI is at `.venv/bin/kaggle` if you need to pull any additional outputs.
