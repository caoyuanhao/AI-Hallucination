### Gate premise, checked on the fitting half

| AUROC(low confidence -> error) | Overall error | Least-conf decile | Most-conf decile |
|---|---|---|---|
| 0.7467 | 16.4% | 47.4% | 3.0% |

Errors concentrate 16x in the least-confident decile, which is what
makes escalating that tail worth more than escalating the same volume at random.

### Cascade payoff, measured on the held-out half

base alone 0.7585 @ 180 ms  |  large alone 0.7747 @ 389 ms  (+1.62pp for 2.16x)  |  shipped base, no retuning 0.7497

| Escalated | Avg latency | confidence gate | random | oracle router |
|---|---|---|---|---|
| 5% | 199 ms | 0.7677 (+0.91) | 0.7600 (+0.14) | 0.8389 (+8.04) |
| 10% | 219 ms | 0.7756 (+1.70) | 0.7608 (+0.23) | 0.8494 (+9.08) |
| 15% | 238 ms | 0.7795 (+2.10) | 0.7611 (+0.26) | 0.8408 (+8.23) |
| 20% | 258 ms | 0.7783 (+1.97) | 0.7668 (+0.82) | 0.8330 (+7.44) |
| 25% | 277 ms | 0.7815 (+2.29) | 0.7670 (+0.84) | 0.8280 (+6.94) |
| 30% | 297 ms | 0.7815 (+2.29) | 0.7701 (+1.16) | 0.8239 (+6.54) |
| 40% | 336 ms | 0.7833 (+2.47) | 0.7699 (+1.14) | 0.8190 (+6.04) |
| 50% | 374 ms | 0.7797 (+2.12) | 0.7696 (+1.10) | 0.8141 (+5.55) |
| 75% | 472 ms | 0.7765 (+1.80) | 0.7612 (+0.26) | 0.7930 (+3.44) |
| 100% | 569 ms | 0.7747 (+1.62) | 0.7747 (+1.62) | 0.7747 (+1.62) |

F1 with the change over base-alone in pp. Both models use per-task thresholds
fitted on the other half; the gate and the budget are not fitted at all.
Latency is additive -- base runs on everything, large only on the escalated
fraction -- so below 50% the median request never reaches large and p50 stays
at 180 ms.

### Significance of the two comparisons that matter (paired bootstrap)

| Escalated | Latency | vs base alone (pp) | vs large alone (pp) |
|---|---|---|---|
| 10% | 219 ms (56% of large) | +1.70 [+0.32,+3.09]* | +0.09 [-2.29,+2.34]  |
| 15% | 238 ms (61% of large) | +2.10 [+0.37,+3.79]* | +0.48 [-1.68,+2.54]  |
| 25% | 277 ms (71% of large) | +2.29 [+0.32,+4.19]* | +0.68 [-1.20,+2.43]  |
| 40% | 336 ms (86% of large) | +2.47 [+0.10,+4.71]* | +0.86 [-0.47,+2.14]  |

2000 paired bootstrap resamples over the reporting half; * marks a 95% interval
excluding zero. The defensible claim is the left column plus cost: the cascade
improves significantly on base, and *matches* large while paying a fraction of
its latency. The apparent edge over large is not separable from noise at
n=1350 and should not be claimed.
