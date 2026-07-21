"""Point squishlab at a named benchmark: one call, one report. The generalized surface,
dogfooded -- MMLU and TruthfulQA flow through the SAME code despite different option counts.

    python experiments/run_benchmark.py                      # TruthfulQA MC1 (variable widths)
    python experiments/run_benchmark.py mmlu:world_religions # any MMLU subject
    python experiments/run_benchmark.py truthfulqa --n 40 --scoring gen

Writes results/run_<benchmark>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from squishlab import OllamaClient, evaluate
from squishlab.loaders import load_mmlu, load_truthfulqa_mc1

MODEL = "qwen3.5:0.8b"


def load(spec: str, n: int, seed: int):
    """`mmlu:<subject>` or `truthfulqa` -> (items, benchmark_label)."""
    if spec.startswith("mmlu"):
        subject = spec.split(":", 1)[1] if ":" in spec else "world_religions"
        return load_mmlu(subject, n, seed), f"mmlu:{subject}"
    if spec.startswith("truthful"):
        return load_truthfulqa_mc1(n, seed), "truthfulqa_mc1"
    raise SystemExit(f"unknown benchmark {spec!r} (try mmlu:<subject> or truthfulqa)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", nargs="?", default="truthfulqa")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scoring", choices=("gen", "ll"), default="ll")
    args = ap.parse_args()

    items, label = load(args.benchmark, args.n, args.seed)
    counts = sorted({len(it.options) for it in items})
    print(f"{label}: {len(items)} items · option counts present: {counts}\n")

    report = evaluate(
        OllamaClient(MODEL), items, benchmark=label, scoring=args.scoring, n_rerun=5
    )
    print(report.to_markdown())

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"run_{label.replace(':', '_')}.json"
    out.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"wrote {out.relative_to(results_dir.parent)}")


if __name__ == "__main__":
    main()
