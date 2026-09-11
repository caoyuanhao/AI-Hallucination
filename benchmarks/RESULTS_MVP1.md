# MVP-1: cheaper serving for a hallucination detector

Building on the reproduced baseline in [RESULTS.md](RESULTS.md). Everything here is
inference plus CPU post-processing — no fine-tuning, no LLM judge, no cloud.

**Result.** A confidence-gated cascade over LettuceDetect's own two checkpoints reaches
the 395M model's accuracy while sending only 10% of traffic to it: **+1.70pp F1 over the
149M model alone (95% CI [+0.32, +3.09]) at 219 ms against the 395M model's 389 ms**.
The accuracy difference against always running the large model is not statistically
separable at this sample size — the win is the cost, not the accuracy.

Raw output: [results/mvp1_cascade.md](results/mvp1_cascade.md),
[results/mvp1_calibration.md](results/mvp1_calibration.md).

## Evaluation discipline

RAGTruth's `train` split cannot be used to fit anything here, because the v1
LettuceDetect checkpoints were trained on it — `vendor/LettuceDetect/scripts/train.py`
selects `sample.split == "train"`. Scores dumped for that split are in-sample.

So test is split in half, stratified by task type: **half A for fitting, half B for
reporting**. Half B is read once, at the end. The `train` split appears in this document
only as evidence for why it is unusable.

| Split | n | Hallucinated | base's error rate | base F1 | F1-optimal threshold |
|---|---|---|---|---|---|
| train | 15090 | 44.5% | **3.1%** | **0.9652** | 0.4830 |
| test | 2700 | 34.9% | 16.6% | 0.7607 | 0.6869 |

An F1 of 0.9652 in-sample against 0.7607 out-of-sample is memorisation. The threshold
that looks optimal against memorised predictions, 0.483, is nowhere near the one that is
actually optimal, 0.687.

## Finding 1 — calibrating on the published training split makes the detector worse

All rows reported on half B. "Headroom" is measured against a per-task oracle fitted
directly on B.

| Threshold fitted on | Value | F1 on B | vs shipped | of oracle headroom |
|---|---|---|---|---|
| shipped | 0.5000 | 0.7497 | +0.00pp | 0% |
| train, global | 0.4830 | 0.7486 | −0.11pp | −7% |
| train, per-task | 0.36/0.57/0.58 | 0.7412 | **−0.85pp** | −52% |
| train + prior-shift correction | 0.5842 | 0.7489 | −0.09pp | −5% |
| **half A, global** | 0.6857 | 0.7555 | +0.57pp | 35% |
| **half A, per-task** | 0.70/0.51/0.73 | **0.7585** | **+0.88pp** | **54%** |
| oracle on B, per-task | — | 0.7661 | +1.64pp | 100% |

Per-task thresholds are listed Summary/Data2txt/QA.

Retuning the threshold is worth a real +0.88pp — but only with clean data. Do the
obvious thing and fit on the published train split and you lose 0.85pp instead, a
1.73pp swing that depends entirely on *which* data you calibrate against.

A base-rate difference between the splits (44.5% vs 34.9% hallucinated) points the same
way, but it is not the explanation: correcting the train-fitted threshold for the prior
shift recovers almost nothing (−0.11pp → −0.09pp). Contamination dominates.

This generalises past this repo. Reusing any published detector means its training split
is off-limits for calibration, and for most released models that split is the most
convenient labelled data available.

## Finding 2 — a rank-based gate sidesteps the problem entirely

The cascade's gate has nothing fitted in it. It escalates the k% of traffic where base is
least confident, measured as `|p − 0.5|`, and k comes from a latency budget rather than
from labels. There is no threshold to place wrongly.

The premise holds — on half A, base's errors concentrate **16x** in its least-confident
decile (47.4% error) against its most-confident (3.0%), AUROC 0.7467 for confidence
predicting its own mistakes. Checking this needs no large-model scores at all.

**Payoff on half B.** Both models use per-task thresholds fitted on A. Latency is
additive: base runs on everything, large only on the escalated fraction.

| Escalated | Avg latency | Confidence gate | Random gate | Oracle router |
|---|---|---|---|---|
| 0% (base) | 180 ms | 0.7585 | — | — |
| 5% | 199 ms | 0.7677 (+0.91) | 0.7600 (+0.14) | 0.8389 (+8.04) |
| **10%** | **219 ms** | **0.7756 (+1.70)** | 0.7608 (+0.23) | 0.8494 (+9.08) |
| 15% | 238 ms | 0.7795 (+2.10) | 0.7611 (+0.26) | 0.8408 (+8.23) |
| 25% | 277 ms | 0.7815 (+2.29) | 0.7670 (+0.84) | 0.8280 (+6.94) |
| 40% | 336 ms | 0.7833 (+2.47) | 0.7699 (+1.14) | 0.8190 (+6.04) |
| 100% (large) | 569 ms | 0.7747 (+1.62) | 0.7747 (+1.62) | 0.7747 (+1.62) |

At 10% escalation the gate delivers **7.4x** the gain of routing the same volume at
random (+1.70pp vs +0.23pp), which is the evidence that the routing decision — not
merely the occasional use of a bigger model — is doing the work.

Below 50% escalation the *median* request never reaches large, so p50 stays at base's
180 ms and only the tail pays.

**Significance.** 2000 paired bootstrap resamples over half B; `*` marks a 95% interval
excluding zero.

| Escalated | Latency | vs base alone | vs large alone |
|---|---|---|---|
| 10% | 219 ms (56% of large) | +1.70 [+0.32, +3.09] * | +0.09 [−2.29, +2.34] |
| 15% | 238 ms (61% of large) | +2.10 [+0.37, +3.79] * | +0.48 [−1.68, +2.54] |
| 25% | 277 ms (71% of large) | +2.29 [+0.32, +4.19] * | +0.68 [−1.20, +2.43] |
| 40% | 336 ms (86% of large) | +2.47 [+0.10, +4.71] * | +0.86 [−0.47, +2.14] |

The improvement over base is significant everywhere. The apparent edge over large is
not, at n=1350, and is not claimed — the defensible statement is that the cascade
*matches* large at 56% of its latency.

## What did not work

**Alternative aggregations.** `max` over token probabilities is LettuceDetect's shipped
rule and nothing beats it — mean, top-5/top-10 mean, logsumexp, fraction and count above
0.5 all score lower on AUROC *and* on oracle F1 ([RESULTS.md](RESULTS.md#headroom-probe)).
The token-probability distribution carries nothing its maximum does not.

**Threshold tuning as a standalone contribution.** +0.88pp with clean data, and negative
with the obvious data. Folded into the cascade rather than sold separately.

## Limitations

- **The oracle router reaches +9.08pp at 10% escalation against the gate's +1.70pp.**
  Confidence captures under a fifth of what perfect routing would deliver. A better gate
  is the clearest remaining opportunity.
- **Fitting and reporting halves are 1350 samples each**, which is why sub-1pp
  differences are not resolvable.
- **Latency is modelled, not measured end-to-end.** The 180/389 ms figures are measured
  p50s at batch=1 ([RESULTS.md](RESULTS.md#serving-latency)); the cascade's cost is
  computed as `180 + k × 389` rather than benchmarked as a running service.
- **`large` was never scored on train**, so its thresholds are also fitted on half A.
  This is consistent but means both models share one fitting set.
- Single dataset, single language, single model family.
