"""Characterise the RAGTruth test split: base rates, taxonomy mix, span geometry.

The example-level metrics in RESULTS.md vary hugely by task type (F1 0.51 on
Summary vs 0.88 on Data2txt). Before blaming the model, it is worth asking what
differs about the *data* — because LettuceDetect's example-level decision rule is
a single fixed threshold shared across all three tasks:

    pred = 1 if (argmax(logits, -1) == 1).any() else 0
    # lettucedetect/models/evaluator.py :: evaluate_model_example_level

That rule has no task-type awareness, so any per-task difference in how often
hallucinations occur, or in how much of the answer they cover, lands directly on
precision/recall.

Usage:
    python benchmarks/analyze_dataset.py
    python benchmarks/analyze_dataset.py --markdown --json-out benchmarks/results/dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

DATA = Path("data/ragtruth/ragtruth_data.json")
TASKS = ["Summary", "Data2txt", "QA"]
# RAGTruth's 2x2 annotation scheme: {Evident, Subtle} x {Conflict, Baseless Info}
LABELS = [
    "Evident Conflict",
    "Evident Baseless Info",
    "Subtle Conflict",
    "Subtle Baseless Info",
]


def integrity(samples: list[dict]) -> dict:
    """Re-check the counts quoted in SETUP.md, so a silent preprocessing drop is caught."""
    test = [s for s in samples if s["split"] == "test"]
    return {
        "total": len(samples),
        "train": sum(1 for s in samples if s["split"] == "train"),
        "test": len(test),
        "test_by_task": {t: sum(1 for s in test if s["task_type"] == t) for t in TASKS},
        "test_hallucinated": sum(1 for s in test if s["labels"]),
    }


def per_task(test: list[dict]) -> list[dict]:
    """Base rate, taxonomy mix and span geometry, one row per task type."""
    rows = []
    for task in TASKS:
        group = [s for s in test if s["task_type"] == task]
        hallu = [s for s in group if s["labels"]]

        spans = [lab for s in hallu for lab in s["labels"]]
        mix = Counter(lab["label"] for lab in spans)
        subtle = sum(v for k, v in mix.items() if k.startswith("Subtle"))

        # Fraction of the answer's characters that sit inside an annotated span.
        coverage = [
            sum(lab["end"] - lab["start"] for lab in s["labels"]) / len(s["answer"])
            for s in hallu
        ]

        rows.append(
            {
                "task_type": task,
                "n": len(group),
                "n_hallucinated": len(hallu),
                "base_rate": len(hallu) / len(group),
                "n_spans": len(spans),
                "spans_per_hallu_example": statistics.mean(len(s["labels"]) for s in hallu),
                "median_span_chars": statistics.median(
                    lab["end"] - lab["start"] for lab in spans
                ),
                "mean_answer_coverage": statistics.mean(coverage),
                "subtle_share": subtle / len(spans),
                "taxonomy": {lab: mix.get(lab, 0) for lab in LABELS},
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    out = "| Task type | Examples | Hallucinated | Base rate | Subtle share |\n"
    out += "|---|---|---|---|---|\n"
    for r in rows:
        out += (
            f"| {r['task_type']} | {r['n']} | {r['n_hallucinated']} "
            f"| {r['base_rate']:.1%} | {r['subtle_share']:.1%} |\n"
        )

    out += "\n| Task type | Spans | Spans/example | Median span | Answer covered |\n"
    out += "|---|---|---|---|---|\n"
    for r in rows:
        out += (
            f"| {r['task_type']} | {r['n_spans']} | {r['spans_per_hallu_example']:.2f} "
            f"| {r['median_span_chars']:.0f} ch | {r['mean_answer_coverage']:.1%} |\n"
        )

    out += "\n| Taxonomy label | " + " | ".join(r["task_type"] for r in rows) + " |\n"
    out += "|---|" + "---|" * len(rows) + "\n"
    for lab in LABELS:
        out += f"| {lab} | " + " | ".join(str(r["taxonomy"][lab]) for r in rows) + " |\n"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    samples = json.loads(args.data.read_text(encoding="utf-8"))
    checks = integrity(samples)
    rows = per_task([s for s in samples if s["split"] == "test"])

    print("Integrity (compare against SETUP.md):")
    print(f"  total          {checks['total']}")
    print(f"  split          train {checks['train']} / test {checks['test']}")
    print(f"  test tasks     " + " / ".join(f"{k} {v}" for k, v in checks["test_by_task"].items()))
    print(
        f"  hallucinated   {checks['test_hallucinated']} "
        f"({checks['test_hallucinated'] / checks['test']:.1%})"
    )
    print()

    if args.markdown:
        print(to_markdown(rows))
    else:
        for r in rows:
            print(
                f"{r['task_type']:10} base_rate={r['base_rate']:.1%} "
                f"subtle={r['subtle_share']:.1%} spans={r['n_spans']:5} "
                f"coverage={r['mean_answer_coverage']:.1%}"
            )

    if args.json_out:
        args.json_out.write_text(
            json.dumps({"integrity": checks, "per_task": rows}, indent=2)
        )
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
