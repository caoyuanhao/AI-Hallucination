"""Does a decision threshold fitted on train survive the move to test?

This is the honest version of the threshold row in scripts/probe_headroom.py. There the
threshold was fitted on test, as a ceiling. Here it is fitted on train and applied once
to test, which is the only defensible way to claim a gain.

The reason to expect trouble: RAGTruth's test split was built balanced (900 per task
type) while train was not, so the two splits differ substantially in how often answers
actually hallucinate --

    task       train    test
    Summary    31.1%   22.7%
    Data2txt   69.4%   64.3%
    QA         31.1%   17.8%

and the F1-optimal threshold moves with the base rate. A threshold fitted where
positives are common is too permissive where they are rare.

Three transfer rules are compared, all fitted on train only:

  absolute   carry the fitted threshold across unchanged
  quantile   carry the fitted *positive rate* across, re-deriving the threshold from
             the test score distribution -- adapts to score drift, not to base-rate drift
  prior-adj  shift the threshold in logit space by the train->test change in base-rate
             odds, which is the textbook correction for label shift and needs the target
             base rate to be known

Reported against the shipped rule (max token probability > 0.5) and against the
test-fitted oracle, so the fraction of available headroom actually captured is visible.

    python scripts/eval_threshold_transfer.py
    python scripts/eval_threshold_transfer.py --markdown

Requires artifacts/scores_base_{train,test}.npz from scripts/dump_scores.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ART = Path("artifacts")
TASKS = ["Summary", "Data2txt", "QA"]


def load(model: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-example max token probability, label, task type."""
    z = np.load(ART / f"scores_{model}_{split}.npz", allow_pickle=False)
    # Read each member once; indexing an NpzFile re-decompresses the whole array.
    off, flat = z["offsets"], z["probs"]
    scores = np.array(
        [flat[off[i] : off[i + 1]].max() if off[i + 1] > off[i] else 0.0 for i in range(len(off) - 1)]
    )
    return scores, z["y_true"].astype(int), z["task_type"]


def best_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Exact F1-optimal threshold: sweep every realisable cut, not a grid."""
    order = np.argsort(-scores, kind="stable")
    s, ys = scores[order], y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    f1 = np.where(np.r_[s[:-1] != s[1:], True], f1, -1.0)
    i = int(np.argmax(f1))
    thr = float(s[i + 1]) if i + 1 < len(s) else float(np.nextafter(s[i], -1))
    return float(f1[i]), thr


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def transfer_rules(
    s_tr: np.ndarray, y_tr: np.ndarray, s_te: np.ndarray, base_rate_te: float
) -> dict[str, float]:
    """Thresholds for test, each derived from train only (except prior-adj's target rate)."""
    _, thr = best_threshold(s_tr, y_tr)

    # Same fraction predicted positive, re-derived against the test score distribution.
    positive_rate = float((s_tr > thr).mean())
    thr_quantile = float(np.quantile(s_te, 1 - positive_rate)) if 0 < positive_rate < 1 else thr

    # Label-shift correction: move the operating point by the change in prior odds.
    rate_tr = float(y_tr.mean())
    shifted = _logit(thr) - (_logit(base_rate_te) - _logit(rate_tr))
    thr_prior = float(1 / (1 + np.exp(-shifted)))

    return {"absolute": thr, "quantile": thr_quantile, "prior-adj": thr_prior}


def evaluate(rows: list[str], markdown: bool) -> None:
    s_tr, y_tr, t_tr = load("base", "train")
    s_te, y_te, t_te = load("base", "test")

    header = ["Scope", "Shipped 0.5", "absolute", "quantile", "prior-adj", "Oracle (test)"]
    table: list[list[str]] = []

    scopes = [(t, t_tr == t, t_te == t) for t in TASKS]
    scopes.append(("per-task combined", None, None))
    scopes.append(("whole (single thr)", np.ones(len(y_tr), bool), np.ones(len(y_te), bool)))

    combined: dict[str, np.ndarray] = {k: np.zeros(len(y_te), int) for k in ("absolute", "quantile", "prior-adj")}

    for name, m_tr, m_te in scopes:
        if m_tr is None:  # filled in after the per-task rows have voted
            shipped = f1_score(y_te, (s_te > 0.5).astype(int))
            oracle_pt = _oracle_per_task(s_te, y_te, t_te)
            table.append(
                [name, f"{shipped:.4f}"]
                + [f"{f1_score(y_te, combined[k]):.4f}" for k in ("absolute", "quantile", "prior-adj")]
                + [f"{oracle_pt:.4f}"]
            )
            continue

        shipped = f1_score(y_te[m_te], (s_te[m_te] > 0.5).astype(int), zero_division=0)
        oracle, _ = best_threshold(s_te[m_te], y_te[m_te])
        thrs = transfer_rules(s_tr[m_tr], y_tr[m_tr], s_te[m_te], float(y_te[m_te].mean()))

        cells = [name, f"{shipped:.4f}"]
        for rule in ("absolute", "quantile", "prior-adj"):
            pred = (s_te[m_te] > thrs[rule]).astype(int)
            cells.append(f"{f1_score(y_te[m_te], pred, zero_division=0):.4f}")
            if name in TASKS:
                combined[rule][m_te] = pred
        cells.append(f"{oracle:.4f}")
        table.append(cells)

    rows.append("### Threshold transfer, fitted on train and applied once to test\n")
    _emit(rows, header, table, markdown)

    # How much of the reachable gain each rule captured, and where. The base-rate shift
    # differs a lot by task, so the per-task view is what shows whether the mechanism
    # is really label shift.
    rows.append("\n### Share of oracle headroom captured\n")
    cap: list[list[str]] = []
    for name, m_tr, m_te in [(t, t_tr == t, t_te == t) for t in TASKS]:
        shift = float(y_tr[m_tr].mean()) / float(y_te[m_te].mean())
        shipped = f1_score(y_te[m_te], (s_te[m_te] > 0.5).astype(int), zero_division=0)
        oracle, _ = best_threshold(s_te[m_te], y_te[m_te])
        head = oracle - shipped
        row = [name, f"{shift:.2f}x", f"{100 * head:+.2f}pp"]
        for rule in ("absolute", "quantile", "prior-adj"):
            gain = f1_score(y_te[m_te], combined[rule][m_te], zero_division=0) - shipped
            row.append(f"{100 * gain:+.2f}pp ({100 * gain / head:.0f}%)" if head > 0 else "n/a")
        cap.append(row)

    shipped = f1_score(y_te, (s_te > 0.5).astype(int))
    oracle_pt = _oracle_per_task(s_te, y_te, t_te)
    head = oracle_pt - shipped
    row = ["combined", f"{y_tr.mean() / y_te.mean():.2f}x", f"{100 * head:+.2f}pp"]
    for rule in ("absolute", "quantile", "prior-adj"):
        gain = f1_score(y_te, combined[rule]) - shipped
        row.append(f"{100 * gain:+.2f}pp ({100 * gain / head:.0f}%)")
    cap.append(row)

    _emit(rows, ["Scope", "Base-rate shift", "Headroom", "absolute", "quantile", "prior-adj"],
          cap, markdown)
    rows.append(f"\nShipped {shipped:.4f} -> per-task oracle {oracle_pt:.4f} on test.")


def _oracle_per_task(s: np.ndarray, y: np.ndarray, task: np.ndarray) -> float:
    pred = np.zeros(len(y), int)
    for t in TASKS:
        m = task == t
        _, thr = best_threshold(s[m], y[m])
        pred[m] = (s[m] > thr).astype(int)
    return f1_score(y, pred)


def _emit(rows: list[str], header: list[str], table: list[list[str]], markdown: bool) -> None:
    if markdown:
        rows.append("| " + " | ".join(header) + " |")
        rows.append("|" + "---|" * len(header))
        for r in table:
            rows.append("| " + " | ".join(r) + " |")
    else:
        widths = [max(len(header[i]), *(len(r[i]) for r in table)) for i in range(len(header))]
        rows.append("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        for r in table:
            rows.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    rows: list[str] = []
    evaluate(rows, args.markdown)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
