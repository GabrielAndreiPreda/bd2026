# UMAP + KNN Results

Metrics aligned with `benchmark.py`. Primary metric: **PR-AUC**.

## Conditions

- **original** — `train.csv`, no balancing
- **smote** — `train_balanced.csv`, SMOTENC 50/50 balance
- **_UMAP** suffix — 2D UMAP (n_neighbors=15, min_dist=0.1, euclidean) applied after StandardScaler

KNN grid: `n_neighbors` in [5, 11, 15, 21, 31], `weights` in [uniform, distance], `metric` in [euclidean, manhattan]. 3-fold stratified CV, scoring=F1.

## Results

| Condition | PR-AUC | Brier | F1@t* | Accuracy | ROC-AUC | t* | k | Features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **original** | 0.4999 | 0.1429 | 0.5099 | 0.7681 | 0.7502 | 0.333 | 15 | 18 |
| **original_UMAP** | 0.4879 | 0.1441 | 0.5040 | 0.7914 | 0.7470 | 0.381 | 21 | 2 |
| **smote** | 0.4749 | 0.1722 | 0.0903 | 0.7849 | 0.7435 | 1.000 | 15 | 18 |
| **smote_UMAP** | 0.4559 | 0.1977 | 0.4833 | 0.6975 | 0.7314 | 0.468 | 31 | 2 |
| *(C0 XGB-only — benchmark ref)* | *0.5281* | *0.1350* | *0.5465* | *0.7959* | *0.7820* | *0.336* | — | *18* |

## UMAP hyperparameters

| Variant | Rows | Components | n_neighbors | min_dist | Metric |
| --- | --- | --- | --- | --- | --- |
| original (viz) | 23972 | 2 | 15 | 0.1 | euclidean |
| balanced (viz) | 37336 | 2 | 15 | 0.1 | euclidean |

## Notes

- t* tuned on 10% calibration slice carved from `train.csv` (seed=42, same as benchmark).
- UMAP fitted on the training portion only; calibration and test sets are transformed.
- KNN hyperparameter search uses F1 CV score (not PR-AUC) to keep grid search tractable.
- Benchmark reference row (C0) included for cross-model comparison.
