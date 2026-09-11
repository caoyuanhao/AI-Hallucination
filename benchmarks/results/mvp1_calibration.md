### Why the train split cannot be used for calibration

| Split | n | Hallucinated | base's error rate | base F1 | F1-optimal thr |
|---|---|---|---|---|---|
| train | 15090 | 44.5% | 3.1% | 0.9652 | 0.4830 |
| test | 2700 | 34.9% | 16.6% | 0.7607 | 0.6869 |

The v1 checkpoints were trained on this train split, so those rows are
in-sample. A 5.4x lower error rate is memorisation, not generalisation, and a
threshold fitted against it is fitted against the wrong distribution.

### Threshold source compared, all reported on the same held-out half of test

Fitting half A: n=1350, 35.0% hallucinated. Reporting half B: n=1350, 34.8%.

| Fitted on | Threshold | F1 on B | vs shipped | of oracle headroom |
|---|---|---|---|---|
| shipped | 0.5000 | 0.7497 | +0.00pp | 0% |
| train-fitted, global | 0.4830 | 0.7486 | -0.11pp | -7% |
| train-fitted, per-task | 0.36/0.57/0.58 | 0.7412 | -0.85pp | -52% |
| train-fitted + prior shift | 0.5842 | 0.7489 | -0.09pp | -5% |
| A-fitted, global | 0.6857 | 0.7555 | +0.57pp | 35% |
| A-fitted, per-task | 0.70/0.51/0.73 | 0.7585 | +0.88pp | 54% |
| ORACLE on B, global | 0.4030 | 0.7574 | +0.77pp | 47% |
| ORACLE on B, per-task | - | 0.7661 | +1.64pp | 100% |

Per-task thresholds are shown as Summary/Data2txt/QA. Half A and half B are
stratified to the same task mix, so nothing here is explained by a base-rate
difference between them.
