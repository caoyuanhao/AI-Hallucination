"""Confidence-gated cascade: run base on everything, escalate its least-sure cases.

The appeal of a cascade here is not only cost. Per scripts/eval_calibration_source.py,
the usual way to improve this detector -- retune its decision threshold -- depends on
having clean data to fit against, and RAGTruth's train split is not clean because the
v1 checkpoints were trained on it. A cascade gate has no such dependency: it escalates
the k% of traffic where base is least confident, which is a *rank* rule with nothing
fitted in it, and k comes from a latency budget rather than from labels.

Nothing is fitted on the reporting half. Following the same discipline as the
calibration script, test is split into a fitting half A and a reporting half B:

  * per-task decision thresholds for both models are fitted on A
  * the gate premise -- that low confidence predicts base's errors -- is checked on A
  * every number in the payoff table is measured on B

Two controls make the gate's own contribution legible:

  random   escalate a uniformly random k%, isolating the value of *routing* from the
           value of merely using large on some traffic
  oracle   escalate only cases base gets wrong and large gets right, the ceiling for
           any router at this budget

    python scripts/eval_cascade.py
    python scripts/eval_cascade.py --markdown

Requires artifacts/scores_{base,large}_test.npz from scripts/dump_scores.py.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from _common import LATENCY_MS, TASKS, best_threshold, emit, load, stratified_half_split

BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
SEED = 0


def fit_task_thresholds(s: np.ndarray, y: np.ndarray, task: np.ndarray, fit: np.ndarray) -> dict[str, float]:
    return {t: best_threshold(s[fit & (task == t)], y[fit & (task == t)])[1] for t in TASKS}


def apply_thresholds(s: np.ndarray, task: np.ndarray, thr: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(s), dtype=int)
    for t in TASKS:
        m = task == t
        out[m] = (s[m] > thr[t]).astype(int)
    return out


def validate_gate(rows: list[str], markdown: bool, s: np.ndarray, y: np.ndarray, fit: np.ndarray) -> None:
    """Check on the fitting half that |p - 0.5| ranks base's own mistakes.

    This needs no large scores at all, so it costs nothing to verify before committing
    to the design.
    """
    wrong = ((s[fit] > 0.5).astype(int) != y[fit]).astype(int)
    confidence = np.abs(s[fit] - 0.5)
    deciles = np.array_split(np.argsort(confidence), 10)
    err = [wrong[d].mean() for d in deciles]

    rows.append("### Gate premise, checked on the fitting half\n")
    emit(
        rows,
        ["AUROC(low confidence -> error)", "Overall error", "Least-conf decile", "Most-conf decile"],
        [[f"{roc_auc_score(wrong, -confidence):.4f}", f"{wrong.mean():.1%}", f"{err[0]:.1%}", f"{err[-1]:.1%}"]],
        markdown,
    )
    rows.append(
        f"\nErrors concentrate {err[0] / err[-1]:.0f}x in the least-confident decile, which is what"
        "\nmakes escalating that tail worth more than escalating the same volume at random."
    )


def payoff(rows: list[str], markdown: bool) -> None:
    s_b, y, task = load("base", "test")
    s_l, y2, _ = load("large", "test")
    assert np.array_equal(y, y2), "base and large dumps disagree on labels"

    a, b = stratified_half_split(task)

    # Both models get per-task thresholds fitted on A only.
    thr_b = fit_task_thresholds(s_b, y, task, a)
    thr_l = fit_task_thresholds(s_l, y, task, a)
    pred_b = apply_thresholds(s_b, task, thr_b)
    pred_l = apply_thresholds(s_l, task, thr_l)

    validate_gate(rows, markdown, s_b, y, a)

    f_base = f1_score(y[b], pred_b[b])
    f_large = f1_score(y[b], pred_l[b])
    f_base_shipped = f1_score(y[b], (s_b[b] > 0.5).astype(int))

    rng = np.random.default_rng(SEED)
    fixable = (pred_b != y) & (pred_l == y)
    orders = {
        "confidence": np.argsort(np.abs(s_b[b] - 0.5), kind="stable"),
        "random": rng.permutation(int(b.sum())),
        "oracle": np.argsort(~fixable[b], kind="stable"),
    }

    sb_b, sl_b, y_b = s_b[b], s_l[b], y[b]
    pb_b, pl_b = pred_b[b], pred_l[b]

    table = []
    for frac in BUDGETS:
        k = int(round(frac * len(y_b)))
        cells = [f"{frac:.0%}", f"{LATENCY_MS['base'] + frac * LATENCY_MS['large']:.0f} ms"]
        for name in ("confidence", "random", "oracle"):
            esc = np.zeros(len(y_b), bool)
            esc[orders[name][:k]] = True
            f = f1_score(y_b, np.where(esc, pl_b, pb_b))
            cells.append(f"{f:.4f} ({100 * (f - f_base):+.2f})")
        table.append(cells)

    # The reporting half is 1350 samples and the interesting differences are under 1pp,
    # so a point estimate alone cannot support "the cascade beats large". Paired
    # bootstrap over examples: resample the same indices for both systems, so the
    # interval is on the *difference* and the shared sampling noise cancels.
    def paired_ci(pred_x: np.ndarray, pred_y: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
        boot_rng = np.random.default_rng(SEED)
        n = len(y_b)
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            idx = boot_rng.integers(0, n, n)
            diffs[i] = f1_score(y_b[idx], pred_x[idx]) - f1_score(y_b[idx], pred_y[idx])
        return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    rows.append("\n### Cascade payoff, measured on the held-out half\n")
    rows.append(
        f"base alone {f_base:.4f} @ {LATENCY_MS['base']:.0f} ms  |  "
        f"large alone {f_large:.4f} @ {LATENCY_MS['large']:.0f} ms  "
        f"({100 * (f_large - f_base):+.2f}pp for {LATENCY_MS['large'] / LATENCY_MS['base']:.2f}x)  |  "
        f"shipped base, no retuning {f_base_shipped:.4f}\n"
    )
    emit(rows, ["Escalated", "Avg latency", "confidence gate", "random", "oracle router"], table, markdown)
    rows.append(
        "\nF1 with the change over base-alone in pp. Both models use per-task thresholds"
        "\nfitted on the other half; the gate and the budget are not fitted at all."
        f"\nLatency is additive -- base runs on everything, large only on the escalated"
        f"\nfraction -- so below 50% the median request never reaches large and p50 stays"
        f"\nat {LATENCY_MS['base']:.0f} ms."
    )

    rows.append("\n### Significance of the two comparisons that matter (paired bootstrap)\n")
    ci_table = []
    for frac in (0.10, 0.15, 0.25, 0.40):
        k = int(round(frac * len(y_b)))
        esc = np.zeros(len(y_b), bool)
        esc[orders["confidence"][:k]] = True
        pred_c = np.where(esc, pl_b, pb_b)
        lat = LATENCY_MS["base"] + frac * LATENCY_MS["large"]

        cells = [f"{frac:.0%}", f"{lat:.0f} ms ({lat / LATENCY_MS['large']:.0%} of large)"]
        for label, reference in (("base", pb_b), ("large", pl_b)):
            lo, hi = paired_ci(pred_c, reference)
            delta = f1_score(y_b, pred_c) - f1_score(y_b, reference)
            mark = "*" if lo > 0 else " "
            cells.append(f"{100 * delta:+.2f} [{100 * lo:+.2f},{100 * hi:+.2f}]{mark}")
        ci_table.append(cells)

    emit(rows, ["Escalated", "Latency", "vs base alone (pp)", "vs large alone (pp)"], ci_table, markdown)
    rows.append(
        "\n2000 paired bootstrap resamples over the reporting half; * marks a 95% interval"
        "\nexcluding zero. The defensible claim is the left column plus cost: the cascade"
        "\nimproves significantly on base, and *matches* large while paying a fraction of"
        "\nits latency. The apparent edge over large is not separable from noise at"
        "\nn=1350 and should not be claimed."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows: list[str] = []
    payoff(rows, args.markdown)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
