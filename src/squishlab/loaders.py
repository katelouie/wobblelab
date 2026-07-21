"""Turn public benchmarks into `list[MCItem]`, so `evaluate()` runs on a named benchmark.

`datasets` is imported lazily inside each loader — installing it is the `[bench]` extra, and
the core library stays importable without it. Each loader returns plain `MCItem`s (variable
option counts welcome); the harness and Task machinery do the rest.
"""

from __future__ import annotations

import random

from squishlab.benchmark import MCItem


def _sample(n_total: int, n: int | None, seed: int) -> list[int]:
    idx = list(range(n_total))
    random.Random(seed).shuffle(idx)
    return idx if n is None else idx[:n]


def load_mmlu(
    subject: str = "world_religions", n: int | None = 40, seed: int = 0
) -> list[MCItem]:
    """MMLU (`cais/mmlu`): 4-option general knowledge across 57 subjects."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", subject, split="test")
    return [
        MCItem(
            id=f"mmlu:{subject}:{j}",
            question=ds[j]["question"],
            options=tuple(ds[j]["choices"]),
            answer_idx=int(ds[j]["answer"]),
        )
        for j in _sample(len(ds), n, seed)
    ]


def load_truthfulqa_mc1(n: int | None = 40, seed: int = 0) -> list[MCItem]:
    """TruthfulQA MC1 (`truthful_qa`, multiple_choice): pick the single true answer among a
    *variable* number of true/false-like choices. Exactly one label is 1.

    This is the loader that exercises the generalization: option counts vary per question
    (roughly 4–13), so it only works because the harness stopped assuming a fixed width.
    """
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    items = []
    for j in _sample(len(ds), n, seed):
        row = ds[j]
        choices = list(row["mc1_targets"]["choices"])
        labels = list(row["mc1_targets"]["labels"])
        items.append(
            MCItem(
                id=f"truthfulqa_mc1:{j}",
                question=row["question"],
                options=tuple(choices),
                answer_idx=labels.index(1),  # exactly one correct in MC1
            )
        )
    return items
