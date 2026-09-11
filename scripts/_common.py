"""Shared loading, threshold fitting and table helpers for the MVP-1 analysis scripts.

One thing here is worth knowing before using any of it. The v1 LettuceDetect checkpoints
were trained on RAGTruth's own `train` split -- see
`vendor/LettuceDetect/scripts/train.py`, which selects `sample.split == "train"` -- so
the scores in `artifacts/scores_*_train.npz` are **in-sample**. The detector's error rate
there is 3.1% against 16.6% on test. Anything fitted on that split is fitted on
memorised predictions and does not transfer.

So the honest place to fit is a held-out slice of test, produced by
:func:`stratified_half_split`. Fit on A, report once on B, and never look at B while
choosing anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ART = Path("artifacts")
TASKS = ["Summary", "Data2txt", "QA"]
# measured p50 at batch=1 on a GTX 1650, see benchmarks/RESULTS.md
LATENCY_MS = {"base": 180.0, "large": 389.0}
SPLIT_SEED = 0


def load(model: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-example max token probability, true label, task type.

    `max` is LettuceDetect's own example-level rule and, per
    scripts/probe_headroom.py, no other aggregation of the token probabilities beats it.
    """
    z = np.load(ART / f"scores_{model}_{split}.npz", allow_pickle=False)
    # Read each member once. Indexing an NpzFile decompresses the whole array on every
    # access, so slicing it inside a loop re-inflates the entire dump per sample.
    off, flat = z["offsets"], z["probs"]
    scores = np.array(
        [flat[off[i] : off[i + 1]].max() if off[i + 1] > off[i] else 0.0 for i in range(len(off) - 1)]
    )
    return scores, z["y_true"].astype(int), z["task_type"]


def stratified_half_split(task: np.ndarray, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Split test into a fitting half A and a reporting half B, balanced per task type.

    Stratifying keeps both halves at the same task mix and hence the same base rate, so
    a threshold fitted on A is not handed a different prior when applied to B.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros(len(task), dtype=bool)
    for t in TASKS:
        idx = np.flatnonzero(task == t)
        rng.shuffle(idx)
        a[idx[: len(idx) // 2]] = True
    return a, ~a


def best_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Exact F1-optimal threshold and the F1 it achieves.

    Swept in closed form: sort descending, and the realisable decision boundaries are
    exactly the gaps between distinct scores. Cumulative true positives give F1 at every
    cut at once, so this is O(n log n) and cannot miss the optimum the way a grid can.
    """
    order = np.argsort(-scores, kind="stable")
    s, ys = scores[order], y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    f1 = np.where(np.r_[s[:-1] != s[1:], True], f1, -1.0)
    i = int(np.argmax(f1))
    thr = float(s[i + 1]) if i + 1 < len(s) else float(np.nextafter(s[i], -1))
    return float(f1[i]), thr


def emit(rows: list[str], header: list[str], table: list[list[str]], markdown: bool) -> None:
    """Append a table to *rows*, as markdown or aligned plain text."""
    if markdown:
        rows.append("| " + " | ".join(header) + " |")
        rows.append("|" + "---|" * len(header))
        rows.extend("| " + " | ".join(r) + " |" for r in table)
    else:
        w = [max(len(header[i]), *(len(r[i]) for r in table)) for i in range(len(header))]
        rows.append("  ".join(h.ljust(x) for h, x in zip(header, w)))
        rows.extend("  ".join(c.ljust(x) for c, x in zip(r, w)) for r in table)
