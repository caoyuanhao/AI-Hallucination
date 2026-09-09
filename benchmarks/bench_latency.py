"""Measure single-sample serving latency for the LettuceDetect models.

The accuracy numbers in the paper come from batched evaluation. A real service
answers one request at a time, so batch=1 latency is the number that actually
matters for deployment — and it is the axis where a 149M encoder beats an
LLM judge by orders of magnitude.

Usage:
    python benchmarks/bench_latency.py --model KRLabsOrg/lettucedect-base-modernbert-en-v1 -n 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from lettucedetect.models.inference import HallucinationDetector

DATA = Path("data/ragtruth/ragtruth_data.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=100, help="number of test samples to time")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    samples = [s for s in json.loads(DATA.read_text(encoding="utf-8")) if s["split"] == "test"]
    samples = samples[: args.n + args.warmup]

    detector = HallucinationDetector(method="transformer", model_path=args.model)

    latencies: list[float] = []
    for i, s in enumerate(samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        detector.predict_prompt(prompt=s["prompt"], answer=s["answer"], output_format="spans")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000  # ms

        if i >= args.warmup:  # discard warmup: first calls pay kernel-compile cost
            latencies.append(dt)

    latencies.sort()
    result = {
        "model": args.model,
        "n": len(latencies),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "mean_ms": round(statistics.mean(latencies), 1),
        "p50_ms": round(latencies[len(latencies) // 2], 1),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
        "max_ms": round(latencies[-1], 1),
        "throughput_per_s": round(1000 / statistics.mean(latencies), 2),
    }
    if torch.cuda.is_available():
        result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
