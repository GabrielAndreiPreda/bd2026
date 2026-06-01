# SVM + LDA Results

Metrics aligned with `benchmark.py`. Primary metric: **PR-AUC**.

## Conditions

- **original** — `train.csv`, no balancing
- **balanced** — `train_balanced.csv`, SMOTENC 50/50 balance
- **_LDA** suffix — 1D Linear Discriminant Analysis projection before SVM

Classifier: `SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')`

## Results

| Condition | PR-AUC | Brier | F1@t* | Accuracy | ROC-AUC | t* | Features | Fit(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **original** | 0.5144 | 0.1418 | 0.5129 | 0.7579 | 0.7127 | 0.152 | 18 | 38 |
| **balanced** | 0.4730 | 0.1718 | 0.4990 | 0.7983 | 0.7408 | 0.709 | 18 | 215 |
| **original_LDA** | 0.4671 | 0.1468 | 0.5231 | 0.7886 | 0.6874 | 0.148 | 1 | 93 |
| **balanced_LDA** | 0.3967 | 0.1969 | 0.4935 | 0.7579 | 0.7070 | 0.656 | 1 | 206 |
| *(C0 XGB-only — benchmark ref)* | *0.5281* | *0.1350* | *0.5465* | *0.7959* | *0.7820* | *0.336* | *18* | — |

## Notes

- t* tuned on 10% calibration slice carved from `train.csv` (seed=42, same as benchmark).
- SVM uses Platt scaling (`probability=True`) for calibrated probabilities; Brier score is meaningful.
- LDA reduces 18 features to 1 component (binary classification); SVM on 1D is very fast.
- Benchmark reference row (C0) included for cross-model comparison.
