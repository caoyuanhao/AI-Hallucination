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

Runs are chunked and resumable, because a full large/train pass is ~3h on a 4GB GPU and
the machine is also the user's daily driver. Each chunk is saved to artifacts/parts/ the
moment it finishes; re-running the same command skips whatever is already on disk and
merges into the canonical dump once every chunk is present. `--max-chunks` bounds how
much work one session does, so the GPU can be handed back on demand.

Usage:
    # smoke test first -- 64 samples, ~15s
    python scripts/dump_scores.py --model base --split test --limit 64

    # run a few chunks now, hand the GPU back, continue later with the same command
    python scripts/dump_scores.py --model large --split train --max-chunks 3
    python scripts/dump_scores.py --model large --split train --max-chunks 3
    ...

    # or run a whole split in one go
    python scripts/dump_scores.py --model large --split test

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


def part_path(model: str, split: str, start: int, end: int, limit: int | None = None) -> Path:
    # Both bounds go in the name. If a later run uses a different --chunk or --limit,
    # its filenames simply will not match, so a stale part can never be silently
    # mistaken for covering a range it does not.
    suffix = f"_limit{limit}" if limit else ""
    return OUT_DIR / "parts" / f"{model}_{split}{suffix}_{start:06d}_{end:06d}.npz"


# ----------------------------------------------------------------------------
# Dump
# ----------------------------------------------------------------------------


def _score_chunk(model, tokenizer, device, samples, batch_size, desc) -> dict:
    """Run the model over one contiguous slice of samples and keep the probabilities."""
    loader = DataLoader(
        HallucinationDataset(samples, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForTokenClassification(
            tokenizer=tokenizer, label_pad_token_id=-100
        ),
    )

    # Ragged per-sample token counts, so store one flat array plus slice boundaries.
    per_sample: list[np.ndarray] = []
    y_true: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
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
                    per_sample.append(np.zeros(0, dtype=np.float32))
                    y_true.append(0)
                    continue

                per_sample.append(probs[i][valid][:, 1].float().cpu().numpy().astype(np.float32))
                y_true.append(int((labels[valid] == 1).any().item()))

    return {"per_sample": per_sample, "y_true": y_true}


def _pack(per_sample: list[np.ndarray], y_true, task_type, source_idx, model_id, split) -> dict:
    lengths = [len(p) for p in per_sample]
    return {
        "probs": np.concatenate(per_sample) if per_sample else np.zeros(0, dtype=np.float32),
        # offsets[i]:offsets[i+1] slices out sample i's tokens
        "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "y_true": np.array(y_true, dtype=np.int8),
        "task_type": np.array(task_type),
        "source_idx": np.array(source_idx, dtype=np.int32),
        "model": model_id,
        "split": split,
    }


def dump(
    model_key: str,
    split: str,
    batch_size: int,
    limit: int | None,
    chunk: int,
    max_chunks: int | None,
) -> Path | None:
    """Score a split in resumable chunks, then merge them into one canonical dump.

    A full large/train pass is ~3h on a 4GB GPU. Chunking lets that be spread over
    several sessions: each chunk is written as soon as it finishes, and re-running the
    same command skips chunks that already exist. `--max-chunks` stops after a fixed
    amount of work so the GPU can be handed back.

    Chunk boundaries change which samples get padded together, relative to a single
    pass. That is provably harmless here -- see the note in the module docstring -- and
    `--verify` re-checks it against the baseline after merging anyway.
    """
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    # Keep the index into the full file so dumps can be joined back to the source
    # rows (task_type, taxonomy labels, answer text) without re-deriving an order.
    keep = [(i, s) for i, s in enumerate(raw) if s["split"] == split]
    if limit:
        keep = keep[:limit]
    total = len(keep)

    bounds = [(s, min(s + chunk, total)) for s in range(0, total, chunk)]
    todo = [b for b in bounds if not part_path(model_key, split, *b, limit).exists()]
    done = len(bounds) - len(todo)
    if max_chunks is not None:
        todo = todo[:max_chunks]

    print(f"{model_key}/{split}: {total} samples, {len(bounds)} chunks of {chunk}")
    print(f"  already done {done}, running {len(todo)} now, "
          f"{len(bounds) - done - len(todo)} left after this")

    if todo:
        (OUT_DIR / "parts").mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForTokenClassification.from_pretrained(
            MODELS[model_key], trust_remote_code=True
        ).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key])

        for start, end in todo:
            piece = keep[start:end]
            scored = _score_chunk(
                model,
                tokenizer,
                device,
                HallucinationData.from_json([s for _, s in piece]).samples,
                batch_size,
                f"{model_key}/{split} [{start}:{end}]",
            )
            np.savez_compressed(
                part_path(model_key, split, start, end, limit),
                **_pack(
                    scored["per_sample"],
                    scored["y_true"],
                    [s["task_type"] for _, s in piece],
                    [i for i, _ in piece],
                    MODELS[model_key],
                    split,
                ),
            )

    missing = [b for b in bounds if not part_path(model_key, split, *b, limit).exists()]
    if missing:
        print(f"\n{len(missing)} chunks still missing; re-run to continue "
              f"(next starts at sample {missing[0][0]})")
        return None

    return merge(model_key, split, bounds, keep, limit)


def merge(model_key: str, split: str, bounds, keep, limit: int | None) -> Path:
    """Concatenate finished chunk files into the canonical dump, in sample order."""
    per_sample: list[np.ndarray] = []
    y_true: list[int] = []
    for start, end in bounds:
        z = np.load(part_path(model_key, split, start, end, limit), allow_pickle=False)
        off, flat = z["offsets"], z["probs"]  # once each; see load_dump
        n = len(off) - 1
        assert n == end - start, f"part [{start}:{end}] holds {n} samples, expected {end - start}"
        per_sample.extend(flat[off[i] : off[i + 1]] for i in range(n))
        y_true.extend(z["y_true"].tolist())

    assert len(per_sample) == len(keep), f"merged {len(per_sample)} samples, expected {len(keep)}"

    OUT_DIR.mkdir(exist_ok=True)
    path = out_path(model_key, split, limit)
    np.savez_compressed(
        path,
        **_pack(
            per_sample,
            y_true,
            [s["task_type"] for _, s in keep],
            [i for i, _ in keep],
            MODELS[model_key],
            split,
        ),
    )
    tokens = sum(len(p) for p in per_sample)
    print(f"\nmerged {len(bounds)} chunks -> {path}  "
          f"({path.stat().st_size / 1e6:.1f} MB, {tokens} tokens)")
    return path


# ----------------------------------------------------------------------------
# Verify
# ----------------------------------------------------------------------------


def load_dump(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    # Read each member once. Indexing an NpzFile decompresses the whole array on every
    # access, so `z["probs"][a:b]` inside a loop re-inflates the entire dump per sample.
    off, flat = z["offsets"], z["probs"]
    return {
        "probs": [flat[off[i] : off[i + 1]] for i in range(len(off) - 1)],
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
    ap.add_argument("--chunk", type=int, default=1000, help="samples per resumable chunk")
    ap.add_argument(
        "--max-chunks",
        type=int,
        help="stop after this many chunks this session; re-run to continue",
    )
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
        args.chunk,
        args.max_chunks,
    )


if __name__ == "__main__":
    main()
