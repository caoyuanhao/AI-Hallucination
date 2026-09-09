"""Parse the text output of LettuceDetect's scripts/evaluate.py into a table.

evaluate.py prints one metric block per task type (Summary, Data2txt, QA) and
then one for the whole test set. tqdm writes progress bars to the same stream,
so the raw file is noisy; this pulls out just the numbers.

Usage:
    python benchmarks/parse_results.py benchmarks/results/*.txt
    python benchmarks/parse_results.py --markdown benchmarks/results/*.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# "Task type: Summary" ... "Precision: 0.1234" ... "AUROC: 0.9876"
TASK_RE = re.compile(r"Task type: (.+?)\s*$", re.MULTILINE)
HALLU_RE = re.compile(
    r"Hallucination Detection \(Class 1\):\s*"
    r"Precision:\s*([\d.]+)\s*"
    r"Recall:\s*([\d.]+)\s*"
    r"F1:\s*([\d.]+)"
)
AUROC_RE = re.compile(r"AUROC:\s*([\d.]+)")


def parse_file(path: Path) -> list[dict]:
    """Extract one record per task-type section in an evaluate.py log."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # tqdm uses \r to redraw; split so its bars don't swallow the metric lines.
    text = text.replace("\r", "\n")

    sections = []
    marks = list(TASK_RE.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.start() : end]

        hallu = HALLU_RE.search(body)
        if not hallu:
            continue  # section did not finish (run still in progress / crashed)
        auroc = AUROC_RE.search(body)

        sections.append(
            {
                "run": path.stem,
                "task_type": m.group(1).strip(),
                "precision": float(hallu.group(1)),
                "recall": float(hallu.group(2)),
                "f1": float(hallu.group(3)),
                "auroc": float(auroc.group(1)) if auroc else None,
            }
        )
    return sections


def to_markdown(rows: list[dict]) -> str:
    head = "| Run | Task type | Precision | Recall | F1 | AUROC |\n"
    head += "|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        auroc = f"{r['auroc']:.4f}" if r["auroc"] is not None else "-"
        body += (
            f"| {r['run']} | {r['task_type']} | {r['precision']:.4f} "
            f"| {r['recall']:.4f} | {r['f1']:.4f} | {auroc} |\n"
        )
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table")
    ap.add_argument("--json-out", type=Path, help="also write the rows as JSON")
    args = ap.parse_args()

    rows: list[dict] = []
    for f in args.files:
        rows.extend(parse_file(f))

    if not rows:
        print("No completed metric blocks found.")
        return

    if args.markdown:
        print(to_markdown(rows))
    else:
        for r in rows:
            auroc = f"{r['auroc']:.4f}" if r["auroc"] is not None else "-"
            print(
                f"{r['run']:32} {r['task_type']:16} "
                f"P={r['precision']:.4f} R={r['recall']:.4f} "
                f"F1={r['f1']:.4f} AUROC={auroc}"
            )

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
