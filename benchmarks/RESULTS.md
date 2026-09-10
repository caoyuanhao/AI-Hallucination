# MVP-0: RAGTruth baseline reproduction

Goal: independently reproduce LettuceDetect's reported RAGTruth numbers, so that every
later change in this project can be measured against a trusted reference point.

Setup and environment gotchas: [SETUP.md](SETUP.md).
Raw logs: [results/](results/). Parsed metrics: `results/baseline.json`.
Dataset statistics: `results/dataset_stats.json`.

## Dataset

RAGTruth, preprocessed into LettuceDetect's schema. Verified before evaluating:

| | |
|---|---|
| total samples | 17,790 |
| split | train 15,090 / test 2,700 |
| test task types | Summary 900 / Data2txt 900 / QA 900 |
| test hallucinated | 943 (34.9%) |

Matches the RAGTruth paper.

## Reproduction vs. reported

Example-level = binary "does this answer contain any hallucination". Metrics are for the
hallucinated class (class 1) on the whole 2,700-sample test set.

| Model | Params | Reported F1 | **Reproduced F1** | Gap |
|---|---|---|---|---|
| lettucedect-base-modernbert-en-v1 | 149M | 76.8% | **76.07%** | −0.73pp |
| lettucedect-large-modernbert-en-v1 | 395M | 79.2% | **79.22%** | **exact** |

The baseline is reproduced. Large matches to the reported decimal; base is 0.73pp low,
within run-to-run noise from tokenizer/transformers version differences.

## Full breakdown

| Model | Task type | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| base | Summary | 0.5389 | 0.4755 | 0.5052 | 0.7560 |
| base | Data2txt | 0.8930 | 0.8653 | 0.8789 | 0.9038 |
| base | QA | 0.6064 | 0.7125 | 0.6552 | 0.9005 |
| base | **whole** | 0.7664 | 0.7550 | **0.7607** | 0.8886 |
| large | Summary | 0.6404 | 0.5588 | 0.5969 | 0.8229 |
| large | Data2txt | 0.9045 | 0.8670 | 0.8854 | 0.9203 |
| large | QA | 0.6593 | 0.7500 | 0.7018 | 0.9176 |
| large | **whole** | 0.8044 | 0.7805 | **0.7922** | 0.9144 |

## Serving latency

Not in the paper. Measured at batch=1 (the real serving path), 100 test samples after
5 warmup calls, on a GTX 1650 4GB. The paper's "30–60 examples/s" is batched throughput
on a stronger GPU; these numbers are not comparable to it and are not meant to be.

| Model | p50 | p95 | p99 | Throughput | Peak VRAM |
|---|---|---|---|---|---|
| base | 180 ms | 414 ms | 608 ms | 4.8 req/s | 719 MB |
| large | 389 ms | 1069 ms | 1369 ms | 2.0 req/s | 1728 MB |

## What the per-task numbers hide

The 30pp spread between Summary and Data2txt invites a story about model capability.
Before telling one, it is worth looking at what differs in the *data* and in the
decision rule. Reproduce with `python benchmarks/analyze_dataset.py --markdown`
(raw: `results/dataset_stats.json`).

The example-level prediction is a single fixed threshold, shared by all three tasks
(`lettucedetect/models/evaluator.py`, `evaluate_model_example_level`):

```python
pred_example_label = 1 if (sample_preds == 1).any().item() else 0
```

An example is hallucinated iff *any* token's argmax is class 1 — i.e. any token above
p=0.5. Nothing about this rule is tuned, and nothing in it is task-aware. Meanwhile the
three tasks are far from interchangeable:

| Task type | Examples | Hallucinated | Base rate | Subtle share |
|---|---|---|---|---|
| Summary | 900 | 204 | 22.7% | 11.1% |
| Data2txt | 900 | 579 | **64.3%** | 9.5% |
| QA | 900 | 160 | 17.8% | **20.9%** |

| Task type | Spans | Spans/example | Median span | Answer covered |
|---|---|---|---|---|
| Summary | 244 | 1.20 | 48 ch | 13.1% |
| Data2txt | 1054 | 1.82 | 26 ch | 6.4% |
| QA | 235 | 1.47 | 98 ch | 22.6% |

| Taxonomy label | Summary | Data2txt | QA |
|---|---|---|---|
| Evident Conflict | 100 | 489 | 30 |
| Evident Baseless Info | 117 | 465 | 156 |
| Subtle Conflict | 11 | 5 | 0 |
| Subtle Baseless Info | 16 | 95 | 49 |

## Findings that shape the next step

**1. The per-task F1 spread mostly measures base rate, not detector quality.**
Data2txt and QA rank almost identically well and score 22pp apart:

| | AUROC | Base rate | F1 | Oracle-threshold F1 |
|---|---|---|---|---|
| Data2txt | 0.9038 | 64.3% | 0.8789 | 0.8870 |
| QA | 0.9005 | 17.8% | 0.6552 | 0.6879 |

A 0.0033 difference in ranking quality becomes a 22.4pp difference in F1. The detector
separates hallucinated from clean answers *equally well* on both; the gap is a property
of F1 under a 3.6x base-rate difference. Give each task its best possible threshold and
19.9pp of the 22.4pp gap survives — F1 on a 17.8%-positive task simply cannot reach F1
on a 64.3%-positive one at equal ranking quality.

So the headline per-task numbers are not commensurable, and "Data2txt is the easy task"
is largely an artifact of how often its answers hallucinate. AUROC, which is invariant
to base rate, says the two tasks are equally hard.

> **Correction.** An earlier version of this section concluded from the same table that
> a per-task threshold was "worth up to double digits of F1". That does not follow, and
> it is wrong: the base-rate component of the gap is not addressable by thresholding.
> Measured headroom is +2.20pp, not double digits — see below.

**2. Summary additionally has a genuine ranking deficit — and it is not about subtlety.**
Summary AUROC is 0.756 against ~0.90 for the other two. That gap is real capability,
not threshold placement, so calibration alone will not close it.

An earlier draft of this document attributed the gap to Summary hallucinations skewing
toward RAGTruth's "Subtle" label types. **The annotations refute that**: Summary is the
*least* subtle task (11.1% of spans), while QA — which scores 15pp higher — is the most
(20.9%). Nor is it a signal-density story: Summary's hallucinated answers have 2x the
character coverage of Data2txt's (13.1% vs 6.4%). What actually makes Summary harder to
rank is still open, and is the question the error analysis should answer.

**3. Cost of `large` is not worth it outside summarization.**
large buys +3.15pp overall F1 for 2.2x p50 latency, 2.6x p95, and 2.4x VRAM. Broken down,
almost the whole gain is on Summary (+9.2pp); Data2txt gains +0.65pp and QA +4.7pp.
For QA/structured workloads `base` is the better operating point.

Where the gain lands is consistent with finding 2: large helps most on Summary (+9.2pp),
the one task with a genuine ranking deficit, and barely at all on Data2txt (+0.65pp),
where base already ranks at AUROC 0.90. Extra capacity buys ranking quality — so it pays
only where ranking, not thresholding, is what is broken.

## Headroom probe

Three directions were on the table for MVP-1. Rather than build all three, each was
first given an *upper bound* — every threshold and escalation rate below is fitted
directly on the test set. That is deliberate cheating, reported only as a ceiling: an
honest method fitted on train can only land under these numbers. Two of the three
directions did not survive.

Reproduce with `python scripts/probe_headroom.py`.

**Thresholds — real but small.**

| Model | Shipped (max, 0.5) | Oracle global | Oracle per-task | Headroom |
|---|---|---|---|---|
| base | 0.7607 | 0.7726 | 0.7827 | +2.20pp |
| large | 0.7922 | 0.7979 | 0.8027 | +1.05pp |

Most of even that comes from moving the *global* threshold (+1.19pp on base); making it
per-task adds only ~1pp more. Worth taking, since it costs nothing at inference, but it
is not a headline.

**Aggregation — no headroom at all.**

| Aggregation | AUROC | Oracle global F1 | Oracle per-task F1 |
|---|---|---|---|
| **max** (shipped) | **0.8886** | **0.7726** | **0.7827** |
| mean | 0.8710 | 0.7486 | 0.7739 |
| top5mean | 0.8825 | 0.7603 | 0.7616 |
| top10mean | 0.8807 | 0.7601 | 0.7661 |
| logsumexp | 0.8675 | 0.7697 | 0.7825 |
| frac>0.5 | 0.8251 | 0.7627 | 0.7673 |
| count>0.5 | 0.8299 | 0.7607 | 0.7662 |

`max` wins on every metric. Nothing about the token-probability *distribution* carries
signal that its maximum does not already carry. This direction is closed.

**Cascade — this is where the result is.**
Run `base` on everything, escalate the samples it is least sure about (smallest
`|p - 0.5|`) to `large`. Cost is additive: 180 ms always, plus 389 ms on the escalated
fraction.

| Escalated | F1 | vs base | large's gain kept | Avg latency | vs base |
|---|---|---|---|---|---|
| 0% (base only) | 0.7607 | — | 0% | 180 ms | 1.00x |
| 10% | 0.7786 | +1.79pp | 57% | 219 ms | 1.22x |
| 15% | 0.7814 | +2.07pp | 66% | 238 ms | 1.32x |
| 30% | 0.7872 | +2.65pp | 84% | 297 ms | 1.65x |
| **40%** | **0.7930** | **+3.23pp** | **102%** | **336 ms** | 1.86x |
| 50% | 0.7946 | +3.39pp | 107% | 374 ms | 2.08x |
| 100% (large only) | 0.7922 | +3.16pp | 100% | 569 ms | 3.16x |

Two things stand out. Escalating 10% of traffic recovers 57% of large's gain for 22%
more latency. And from 40% on, the cascade **beats always-running-large outright** —
0.7930 vs 0.7922 — while costing 336 ms against large-only's 389 ms. Sending every
request to the bigger model is not just expensive, it is worse than routing, because
`base` is right about some cases `large` gets wrong.

## Where this goes next (MVP-1)

The probe reorders the plan. Under this project's constraints — a local 4GB GPU, no
cloud, no LLM API budget, so no fine-tuning and no LLM judge — the work is:

1. **The cascade, as the headline.** Fit the escalation gate on train, report the
   F1-versus-latency Pareto curve once on test. The oracle curve above says the shape is
   there; what remains is showing it survives honest fitting.
2. **Per-task thresholds, as a cheap add-on.** +2.20pp oracle, so expect ~+1pp real.
   Folded into the cascade rather than sold separately.
3. **A CPU-side fusion head, as the open question.** Aggregation statistics are dead, but
   a *learned* combination of `max`, answer length, task type and the linguistic features
   in `src/detector/confidence_scorer.py` has not been tested. Given how flat the
   aggregation table is, this one may well come back negative — which is worth reporting
   either way.

Everything above runs off one score dump per model per split
(`scripts/dump_scores.py`), which must re-derive 76.07 / 79.22 exactly before anything
is built on it. Both test dumps now pass that gate on all 16 metrics.

## Reference points for later comparison

From LettuceDetect's `docs/benchmarks.md`, same test set, example-level overall F1:

| System | Type | F1 |
|---|---|---|
| GPT-4 | LLM, zero-shot | 63.4% |
| Luna | Encoder | 65.4% |
| **lettucedect-base** | Encoder 149M | **76.8%** |
| Llama-2-13B fine-tuned | LLM | 78.7% |
| **lettucedect-large** | Encoder 395M | **79.2%** |
| RAG-HAT (Llama-3-8B) | LLM | 83.9% |

A 149M encoder already beats zero-shot GPT-4 by 12.7pp on this task, at ~180 ms and
719 MB. That is the bar: any added signal has to justify its latency and cost against a
model this cheap, which is why MVP-1 targets the decision layer rather than a bigger one.
