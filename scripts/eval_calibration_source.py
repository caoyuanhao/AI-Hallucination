"""Where can you calibrate a published detector? Not on its published training split.

Tuning the example-level decision threshold is the cheapest possible improvement to
LettuceDetect: no parameters, no training, no extra inference. The obvious way to do it
is to fit on RAGTruth's `train` split and apply to `test`. That is a trap, and this
script measures how expensive a trap it is.

The v1 checkpoints were trained on exactly that split
(`vendor/LettuceDetect/scripts/train.py` keeps `sample.split == "train"`), so the scores
dumped for it are in-sample. The detector's own error rate is 3.1% there against 16.6%
on test. Its score distribution is correspondingly overconfident, and the F1-optimal
threshold sits in a completely different place -- 0.483 fitted on train against 0.686
fitted on held-out data.

Two comparisons, both reported on the same untouched half of test:

  train-fitted   the obvious approach, using the split the model memorised
  A-fitted       fitted on the other half of test, which the model never saw

A base-rate difference between the splits (train 44.5% hallucinated, test 34.9%) pushes
in the same direction, so a prior-shift correction of the train-fitted threshold is
included to show how much of the damage that accounts for. It is not a deployable
method -- it needs the target base rate -- only a decomposition.

    python scripts/eval_calibration_source.py
    python scripts/eval_calibration_source.py --markdown
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score

from _common import TASKS, best_threshold, emit, load, stratified_half_split


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def contamination_evidence(rows: list[str], markdown: bool) -> None:
    """The reason train is unusable, stated in the detector's own error rate."""
    table = []
    for split in ("train", "test"):
        s, y, _ = load("base", split)
        pred = (s > 0.5).astype(int)
        table.append(
            [
                split,
                f"{len(y)}",
                f"{y.mean():.1%}",
                f"{(pred != y).mean():.1%}",
                f"{f1_score(y, pred):.4f}",
                f"{best_threshold(s, y)[1]:.4f}",
            ]
        )
    rows.append("### Why the train split cannot be used for calibration\n")
    emit(
        rows,
        ["Split", "n", "Hallucinated", "base's error rate", "base F1", "F1-optimal thr"],
        table,
        markdown,
    )
    rows.append(
        "\nThe v1 checkpoints were trained on this train split, so those rows are"
        "\nin-sample. A 5.4x lower error rate is memorisation, not generalisation, and a"
        "\nthreshold fitted against it is fitted against the wrong distribution."
    )


def compare_sources(rows: list[str], markdown: bool) -> None:
    s, y, task = load("base", "test")
    a, b = stratified_half_split(task)

    s_tr, y_tr, task_tr = load("base", "train")

    shipped = f1_score(y[b], (s[b] > 0.5).astype(int))
    _, oracle_thr_global = best_threshold(s[b], y[b])
    oracle_global = f1_score(y[b], (s[b] > oracle_thr_global).astype(int))

    # per-task oracle on B, the ceiling for any per-task rule
    pred = np.zeros(len(y), int)
    for t in TASKS:
        m = b & (task == t)
        _, thr = best_threshold(s[m], y[m])
        pred[m] = (s[m] > thr).astype(int)
    oracle_pt = f1_score(y[b], pred[b])

    def report(name: str, thresholds: dict[str, float] | float) -> list[str]:
        if isinstance(thresholds, float):
            p = (s[b] > thresholds).astype(int)
            shown = f"{thresholds:.4f}"
        else:
            p = np.zeros(len(y), int)
            for t in TASKS:
                m = b & (task == t)
                p[m] = (s[m] > thresholds[t]).astype(int)
            p = p[b]
            shown = "/".join(f"{thresholds[t]:.2f}" for t in TASKS)
        f = f1_score(y[b], p)
        head = oracle_pt - shipped
        return [name, shown, f"{f:.4f}", f"{100 * (f - shipped):+.2f}pp", f"{100 * (f - shipped) / head:.0f}%"]

    _, thr_train = best_threshold(s_tr, y_tr)
    _, thr_a = best_threshold(s[a], y[a])
    thr_train_pt = {t: best_threshold(s_tr[task_tr == t], y_tr[task_tr == t])[1] for t in TASKS}
    thr_a_pt = {t: best_threshold(s[a & (task == t)], y[a & (task == t)])[1] for t in TASKS}

    # prior-shift correction of the train-fitted global threshold, in logit space
    shifted = _logit(thr_train) - (_logit(float(y[b].mean())) - _logit(float(y_tr.mean())))
    thr_prior = float(1 / (1 + np.exp(-shifted)))

    table = [
        ["shipped", "0.5000", f"{shipped:.4f}", "+0.00pp", "0%"],
        report("train-fitted, global", thr_train),
        report("train-fitted, per-task", thr_train_pt),
        report("train-fitted + prior shift", thr_prior),
        report("A-fitted, global", thr_a),
        report("A-fitted, per-task", thr_a_pt),
        ["ORACLE on B, global", f"{oracle_thr_global:.4f}", f"{oracle_global:.4f}",
         f"{100 * (oracle_global - shipped):+.2f}pp", f"{100 * (oracle_global - shipped) / (oracle_pt - shipped):.0f}%"],
        ["ORACLE on B, per-task", "-", f"{oracle_pt:.4f}", f"{100 * (oracle_pt - shipped):+.2f}pp", "100%"],
    ]

    rows.append(f"\n### Threshold source compared, all reported on the same held-out half of test\n")
    rows.append(f"Fitting half A: n={a.sum()}, {y[a].mean():.1%} hallucinated. "
                f"Reporting half B: n={b.sum()}, {y[b].mean():.1%}.\n")
    emit(rows, ["Fitted on", "Threshold", "F1 on B", "vs shipped", "of oracle headroom"], table, markdown)
    rows.append(
        "\nPer-task thresholds are shown as Summary/Data2txt/QA. Half A and half B are"
        "\nstratified to the same task mix, so nothing here is explained by a base-rate"
        "\ndifference between them."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows: list[str] = []
    contamination_evidence(rows, args.markdown)
    compare_sources(rows, args.markdown)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
