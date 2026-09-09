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

**1. Data2txt vs QA isolates the cost of the fixed threshold.**
These two tasks rank almost identically well and score 22pp apart:

| | AUROC | Base rate | F1 |
|---|---|---|---|
| Data2txt | 0.9038 | 64.3% | 0.8789 |
| QA | 0.9005 | 17.8% | 0.6552 |

A 0.0033 difference in ranking quality becomes a 22.4pp difference in F1. The model
separates hallucinated from clean answers *equally well* on both; what differs is that
a 3.6x higher base rate makes p=0.5 a near-optimal operating point on Data2txt and a
poor one on QA. Much of Data2txt's headline 0.879 is base-rate inflation, not skill.

This is the single clearest result here: **a per-task threshold is worth up to double
digits of F1 and costs no parameters, no training, and no extra inference.**

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

## Where this goes next (MVP-1)

Findings 1 and 3 point the same direction: the accuracy left on the table is in how
scores are turned into decisions and how requests are routed, not in the encoder. That
is also the only direction available under this project's constraints — a local 4GB GPU,
no cloud, and no LLM API budget, which rules out both fine-tuning and an LLM judge.

1. **Per-task calibration and aggregation.** Replace `any-token > 0.5` with a tuned
   aggregation (max / top-k mean / logsumexp / count-above-τ) and per-task thresholds,
   fitted on train and reported once on test.
2. **A CPU-side fusion head.** Token-probability statistics plus the linguistic features
   already in `src/detector/confidence_scorer.py`, fed to logistic regression / GBDT.
   Open question: can `base` + a 20-feature head match `large`?
3. **A confidence-gated cascade.** Run `base` on everything, escalate only the uncertain
   tail to `large`, and trace the F1-versus-latency Pareto curve using the measured
   180 ms / 389 ms operating points above.

All three need per-sample token probabilities on both splits, so the first deliverable is
a one-time score dump that must re-derive 76.07 / 79.22 exactly before anything is built
on it.

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
