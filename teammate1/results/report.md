# Logistic Regression Results

Metrics aligned with `benchmark.py` for direct comparison. Primary metric: **PR-AUC**.

## Conditions

- **original** — `train.csv`, no class balancing, default LR
- **balanced_cw** — `train.csv`, `class_weight='balanced'` (upweights minority during fit)
- **smote** — `train_balanced.csv`, SMOTENC-balanced 50/50
- **_PCA** suffix — PCA(95% variance) applied after StandardScaler

## Results

| Condition | PR-AUC | Brier | F1@τ* | Accuracy | ROC-AUC | τ* | Features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **original** | 0.5087 | 0.1423 | 0.5183 | 0.7934 | 0.7458 | 0.319 | 18 |
| **original_PCA** | 0.5031 | 0.1428 | 0.5277 | 0.7906 | 0.7470 | 0.319 | 13 |
| **balanced_cw** | 0.4969 | 0.1960 | 0.5272 | 0.7812 | 0.7495 | 0.577 | 18 |
| **balanced_cw_PCA** | 0.4941 | 0.1969 | 0.5275 | 0.7809 | 0.7497 | 0.593 | 13 |
| **smote** | 0.4870 | 0.1937 | 0.4965 | 0.7597 | 0.7227 | 0.543 | 18 |
| **smote_PCA** | 0.4851 | 0.2002 | 0.4945 | 0.7322 | 0.7269 | 0.522 | 13 |
| *(C0 XGB-only — benchmark ref)* | *0.5281* | *0.1350* | *0.5465* | *0.7959* | *0.7820* | *0.336* | *18* |

## Notes

- τ* tuned on 10% calibration slice carved from `train.csv` (same seed=42 as model.py).
- Brier score is meaningful here because logistic regression is naturally well-calibrated.
- XGBoost benchmark reference (C0) included for context.
