"""Upper-bound the three MVP-1 directions before spending GPU hours on them.

Every threshold and escalation rate here is chosen *on the test set*. These are oracle
numbers -- deliberate cheating, reported only as ceilings. A method fitted honestly on
train can only do worse. The point is to find out which directions are worth building
before building them, and two of the three turned out not to be.

    python scripts/probe_headroom.py
    python scripts/probe_headroom.py --markdown

Requires artifacts/scores_{base,large}_test.npz from scripts/dump_scores.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ART = Path("artifacts")
TASKS = ["Summary", "Data2txt", "QA"]
# measured p50 at batch=1 on a GTX 1650, see benchmarks/RESULTS.md
LATENCY_MS = {"base": 180.0, "large": 389.0}


def load(model: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    z = np.load(ART / f"scores_{model}_test.npz", allow_pickle=False)
    # Read each member once. Indexing an NpzFile decompresses the whole array on every
    # access, so `z["probs"][a:b]` inside a loop re-inflates 440k floats per sample.
    off, flat = z["offsets"], z["probs"]
    probs = [flat[off[i] : off[i + 1]] for i in range(len(off) - 1)]
    return probs, z["y_true"].astype(int), z["task_type"]


AGGREGATIONS = {
    "max": lambda p: p.max(),
    "mean": lambda p: p.mean(),
    "top5mean": lambda p: np.sort(p)[-5:].mean(),
    "top10mean": lambda p: np.sort(p)[-10:].mean(),
    "logsumexp": lambda p: np.log(np.exp(p * 10).sum()) / 10,
    "frac>0.5": lambda p: (p > 0.5).mean(),
    "count>0.5": lambda p: float((p > 0.5).sum()),
}


def aggregate(probs: list[np.ndarray], how: str) -> np.ndarray:
    fn = AGGREGATIONS[how]
    return np.array([fn(p) if len(p) else 0.0 for p in probs])


def oracle_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Best achievable F1 over *every* distinct threshold, and where it occurs.

    Swept in closed form rather than over a grid: sort by score descending, and the
    candidate decision boundaries are exactly the points between distinct scores.
    Cumulative true positives then give F1 at every cut at once. A grid would both
    miss the true optimum and cost thousands of f1_score calls.
    """
    order = np.argsort(-scores, kind="stable")
    s_sorted, y_sorted = scores[order], y[order]

    tp = np.cumsum(y_sorted)  # predicting the top k as positive
    k = np.arange(1, len(y_sorted) + 1)
    positives = y_sorted.sum()
    f1 = np.divide(2 * tp, k + positives, out=np.zeros(len(k)), where=(k + positives) > 0)

    # Only cuts that fall between two different scores are realisable thresholds.
    realisable = np.r_[s_sorted[:-1] != s_sorted[1:], True]
    f1 = np.where(realisable, f1, -1.0)

    i = int(np.argmax(f1))
    # Any threshold in [next score, this score) selects exactly the top i+1.
    thr = float(s_sorted[i + 1]) if i + 1 < len(s_sorted) else float(np.nextafter(s_sorted[i], -1))
    return float(f1[i]), thr


def oracle_per_task(scores: np.ndarray, y: np.ndarray, task: np.ndarray) -> float:
    """F1 when each task type gets its own oracle threshold."""
    pred = np.zeros(len(y), dtype=int)
    for t in TASKS:
        m = task == t
        _, thr = oracle_threshold(scores[m], y[m])
        pred[m] = (scores[m] > thr).astype(int)
    return f1_score(y, pred, zero_division=0)


def probe_base_rate_effect(rows: list[str]) -> None:
    """How much of the per-task F1 spread is irreducible under any threshold.

    Data2txt and QA have near-identical AUROC but a 3.6x base-rate difference. If the
    spread were a thresholding artifact it would collapse once each task is given its
    own optimum; the point of this table is that it does not.
    """
    probs, y, task = load("base")
    s = aggregate(probs, "max")

    rows.append("### Base-rate effect (base, max aggregation)\n")
    rows.append("| Task | AUROC | Base rate | Shipped F1 | Oracle-threshold F1 | Oracle thr |")
    rows.append("|---|---|---|---|---|---|")
    for t in TASKS:
        m = task == t
        f_now = f1_score(y[m], (s[m] > 0.5).astype(int), zero_division=0)
        f_best, thr = oracle_threshold(s[m], y[m])
        rows.append(
            f"| {t} | {roc_auc_score(y[m], s[m]):.4f} | {y[m].mean():.1%} "
            f"| {f_now:.4f} | {f_best:.4f} | {thr:.4f} |"
        )
    rows.append("")


# ---------------------------------------------------------------------------


def probe_thresholds(rows: list[str]) -> None:
    rows.append("### Threshold headroom (oracle, fitted on test)\n")
    rows.append("| Model | Shipped (max, 0.5) | Oracle global | Oracle per-task | Per-task gain |")
    rows.append("|---|---|---|---|---|")
    for model in ("base", "large"):
        probs, y, task = load(model)
        s = aggregate(probs, "max")
        shipped = f1_score(y, (s > 0.5).astype(int))
        g_f1, _ = oracle_threshold(s, y)
        pt = oracle_per_task(s, y, task)
        rows.append(
            f"| {model} | {shipped:.4f} | {g_f1:.4f} | {pt:.4f} | **{100 * (pt - shipped):+.2f}pp** |"
        )
    rows.append("")


def probe_aggregation(rows: list[str]) -> None:
    probs, y, task = load("base")
    rows.append("### Aggregation headroom (base, oracle thresholds)\n")
    rows.append("| Aggregation | AUROC | Oracle global F1 | Oracle per-task F1 |")
    rows.append("|---|---|---|---|")
    for how in AGGREGATIONS:
        s = aggregate(probs, how)
        g_f1, _ = oracle_threshold(s, y)
        rows.append(
            f"| {how} | {roc_auc_score(y, s):.4f} | {g_f1:.4f} | {oracle_per_task(s, y, task):.4f} |"
        )
    rows.append("")


def probe_cascade(rows: list[str]) -> None:
    pb, y, _ = load("base")
    pl, y2, _ = load("large")
    assert np.array_equal(y, y2), "base and large dumps disagree on labels"

    sb, sl = aggregate(pb, "max"), aggregate(pl, "max")
    f_base = f1_score(y, (sb > 0.5).astype(int))
    f_large = f1_score(y, (sl > 0.5).astype(int))

    # Escalate the samples base is least sure about, measured as distance from its
    # own decision boundary. base runs on everything, large only on the escalated tail,
    # so the cost is additive.
    order = np.argsort(np.abs(sb - 0.5))

    rows.append("### Cascade headroom (escalate base's least-confident tail to large)\n")
    rows.append(f"base only: F1 {f_base:.4f} at {LATENCY_MS['base']:.0f} ms  ")
    rows.append(
        f"large only: F1 {f_large:.4f} at {LATENCY_MS['large']:.0f} ms "
        f"({100 * (f_large - f_base):+.2f}pp for {LATENCY_MS['large'] / LATENCY_MS['base']:.2f}x)\n"
    )
    rows.append("| Escalated | F1 | vs base | large's gain kept | Avg latency | vs base |")
    rows.append("|---|---|---|---|---|---|")
    for frac in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00):
        k = int(round(frac * len(y)))
        escalate = np.zeros(len(y), dtype=bool)
        escalate[order[:k]] = True
        s = np.where(escalate, sl, sb)
        f = f1_score(y, (s > 0.5).astype(int))
        lat = LATENCY_MS["base"] + frac * LATENCY_MS["large"]
        kept = 100 * (f - f_base) / (f_large - f_base)
        rows.append(
            f"| {frac:.0%} | {f:.4f} | {100 * (f - f_base):+.2f}pp | {kept:.0f}% "
            f"| {lat:.0f} ms | {lat / LATENCY_MS['base']:.2f}x |"
        )
    rows.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="emit tables ready to paste")
    ap.parse_args()

    rows: list[str] = []
    probe_base_rate_effect(rows)
    probe_thresholds(rows)
    probe_aggregation(rows)
    probe_cascade(rows)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
