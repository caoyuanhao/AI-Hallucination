"""Dump per-token hallucination probabilities to disk, once, for reuse.

Everything in MVP-1 -- threshold tuning, aggregation sweeps, the fusion head, the
base->large cascade -- is a function of the model's per-token class-1 probabilities.
Recomputing those on a 4GB GPU for every experiment would make each iteration a
multi-hour job. Dumping them once turns the whole search into CPU-bound seconds.

The inference path here is deliberately identical to LettuceDetect's own evaluation
(`scripts/evaluate.py` -> `evaluate_model_example_level`): same HallucinationDataset,
same DataCollatorForTokenClassification with label_pad_token_id=-100, same softmax over
a 2-class token-classification head, `shuffle=False`. Only the output differs -- probs
are kept instead of being collapsed into metrics.

`--verify` is the acceptance gate: it re-derives example-level P/R/F1/AUROC from the
dump alone and diffs against benchmarks/results/baseline.json. If those do not match,
the dump is misaligned and nothing downstream can be trusted.

Note on batching: evaluate.py runs each task type as its own pass, so its batches hold
only same-task samples, while this script does a single pass over the whole split. That
changes how sequences are padded together. Attention masking makes the two
mathematically equivalent, and empirically they agree exactly -- all 16 example-level
metrics for base/test reproduce baseline.json to 4dp -- so padding composition does not
perturb the result here.

Usage:
    # smoke test first -- 64 samples, ~15s
    python scripts/dump_scores.py --model base --split test --limit 64

    # the real runs
    python scripts/dump_scores.py --model base  --split test
    python scripts/dump_scores.py --model base  --split train
    python scripts/dump_scores.py --model large --split test  --batch-size 2
    python scripts/dump_scores.py --model large --split train --batch-size 2

    # acceptance gate (test split only -- train has no published reference)
    python scripts/dump_scores.py --model base --split test --verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationDataset,
)

MODELS = {
    "base": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
    "large": "KRLabsOrg/lettucedect-large-modernbert-en-v1",
}
# batch sizes that fit in 4GB VRAM at max_length=4096 (see benchmarks/SETUP.md)
DEFAULT_BATCH = {"base": 4, "large": 2}

DATA = Path("data/ragtruth/ragtruth_data.json")
OUT_DIR = Path("artifacts")
BASELINE = Path("benchmarks/results/baseline.json")


def out_path(model: str, split: str, limit: int | None = None) -> Path:
    # A truncated smoke-test dump must never land on the canonical path, or it will
    # later be mistaken for a full run.
    suffix = f"_limit{limit}" if limit else ""
    return OUT_DIR / f"scores_{model}_{split}{suffix}.npz"


# ----------------------------------------------------------------------------
# Dump
# ----------------------------------------------------------------------------


def dump(model_key: str, split: str, batch_size: int, limit: int | None) -> Path:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    # Keep the index into the full file so dumps can be joined back to the source
    # rows (task_type, taxonomy labels, answer text) without re-deriving an order.
    keep = [(i, s) for i, s in enumerate(raw) if s["split"] == split]
    if limit:
        keep = keep[:limit]
    source_idx = np.array([i for i, _ in keep], dtype=np.int32)

    samples = HallucinationData.from_json([s for _, s in keep]).samples
    print(f"{model_key}/{split}: {len(samples)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        MODELS[model_key], trust_remote_code=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key])

    loader = DataLoader(
        HallucinationDataset(samples, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForTokenClassification(
            tokenizer=tokenizer, label_pad_token_id=-100
        ),
    )

    # Ragged per-sample token counts, so store one flat array plus slice boundaries.
    chunks: list[np.ndarray] = []
    lengths: list[int] = []
    y_true: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{model_key}/{split}"):
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=-1)

            for i in range(batch["labels"].size(0)):
                labels = batch["labels"][i]
                # -100 covers both context tokens and collator padding.
                valid = labels != -100

                if valid.sum().item() == 0:
                    # evaluate_model_example_level scores these as clean with p=0.0.
                    chunks.append(np.zeros(0, dtype=np.float32))
                    lengths.append(0)
                    y_true.append(0)
                    continue

                p1 = probs[i][valid][:, 1].float().cpu().numpy().astype(np.float32)
                chunks.append(p1)
                lengths.append(len(p1))
                y_true.append(int((labels[valid] == 1).any().item()))

    OUT_DIR.mkdir(exist_ok=True)
    path = out_path(model_key, split, limit)
    np.savez_compressed(
        path,
        probs=np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32),
        # offsets[i]:offsets[i+1] slices out sample i's tokens
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        y_true=np.array(y_true, dtype=np.int8),
        task_type=np.array([s["task_type"] for _, s in keep]),
        source_idx=source_idx,
        model=MODELS[model_key],
        split=split,
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, {sum(lengths)} tokens)")
    return path


# ----------------------------------------------------------------------------
# Verify
# ----------------------------------------------------------------------------


def load_dump(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    off = z["offsets"]
    return {
        "probs": [z["probs"][off[i] : off[i + 1]] for i in range(len(off) - 1)],
        "y_true": z["y_true"],
        "task_type": z["task_type"],
        "source_idx": z["source_idx"],
    }


def example_metrics(probs: list[np.ndarray], y_true: np.ndarray) -> dict:
    """Reproduce evaluate_model_example_level's decision rule from probabilities alone.

    That rule is `any(argmax == 1)`, which for a 2-class head is exactly
    `any(p1 > 0.5)`. AUROC is ranked on the per-example max of p1.
    """
    scores = np.array([p.max() if len(p) else 0.0 for p in probs])
    y_pred = (scores > 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "f1": float(f1[1]),
        "auroc": float(auc(fpr, tpr)),
    }


def verify(model_key: str, split: str, tol: float) -> bool:
    d = load_dump(out_path(model_key, split))
    reference = {
        r["task_type"]: r
        for r in json.loads(BASELINE.read_text(encoding="utf-8"))
        if r["run"] == f"{model_key}_example_level"
    }
    if not reference:
        print(f"No baseline rows for {model_key}_example_level in {BASELINE}")
        return False

    groups = [
        (t, np.flatnonzero(d["task_type"] == t)) for t in ["Summary", "Data2txt", "QA"]
    ]
    groups.append(("whole dataset", np.arange(len(d["y_true"]))))

    print(f"\n{'task':14}{'metric':11}{'dumped':>10}{'baseline':>10}{'delta':>10}")
    ok = True
    for task, idx in groups:
        if task not in reference:
            continue
        got = example_metrics([d["probs"][i] for i in idx], d["y_true"][idx])
        for metric in ("precision", "recall", "f1", "auroc"):
            want = reference[task][metric]
            delta = got[metric] - want
            # baseline.json is rounded to 4dp, so tolerance must absorb that.
            flag = "" if abs(delta) <= tol else "   <-- MISMATCH"
            if flag:
                ok = False
            print(
                f"{task:14}{metric:11}{got[metric]:>10.4f}{want:>10.4f}"
                f"{delta:>+10.4f}{flag}"
            )
    print("\nPASS: dump reproduces the baseline" if ok else "\nFAIL: dump is misaligned")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--batch-size", type=int, help="default: 4 for base, 2 for large")
    ap.add_argument("--limit", type=int, help="only dump the first N samples (smoke test)")
    ap.add_argument("--verify", action="store_true", help="check an existing dump, do not run the model")
    ap.add_argument("--tol", type=float, default=5e-4, help="max allowed deviation from baseline.json")
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(0 if verify(args.model, args.split, args.tol) else 1)

    dump(
        args.model,
        args.split,
        args.batch_size or DEFAULT_BATCH[args.model],
        args.limit,
    )


if __name__ == "__main__":
    main()
