"""Turn public benchmarks into `list[MCItem]`, so `evaluate()` runs on a named benchmark.

`datasets` is imported lazily inside each loader — installing it is the `[bench]` extra, and
the core library stays importable without it. Each loader returns plain `MCItem`s (variable
option counts welcome); the harness and Task machinery do the rest.
"""

from __future__ import annotations

import random

from wobblelab.benchmark import MCItem


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


def load_gpqa_diamond(n: int | None = None, seed: int = 0) -> list[MCItem]:
    """GPQA Diamond (`Idavidrein/gpqa`, `gpqa_diamond`): 198 graduate-level 4-option MCQs
    in physics, chemistry, and biology — hard enough that PhD experts *in the field* score
    ~65% and skilled non-experts with web access ~34%. The interesting stress test: a real
    benchmark where the questions are long and technical, not trivia.

    **Gated dataset.** Accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa
    and authenticate (`hf auth login`, or set `HF_TOKEN`) before loading, or `datasets`
    raises `DatasetNotFoundError`.

    The four answers live in separate columns with the correct one always in the same field,
    so each question's options are shuffled with a per-item seed. Without that, `NaturalOrder`
    would show every correct answer at slot A and the model's position bias would masquerade
    as accuracy — see the harness lens. The shuffle is deterministic in `seed`, so the
    natural-order number is reproducible.
    """
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    items = []
    for j in _sample(len(ds), n, seed):
        row = ds[j]
        # index 0 is the correct answer; shuffle, then find where it landed
        answers = [
            row["Correct Answer"].strip(),
            row["Incorrect Answer 1"].strip(),
            row["Incorrect Answer 2"].strip(),
            row["Incorrect Answer 3"].strip(),
        ]
        order = [0, 1, 2, 3]
        random.Random(f"gpqa:{seed}:{j}").shuffle(order)
        items.append(
            MCItem(
                id=f"gpqa_diamond:{j}",
                question=row["Question"].strip(),
                options=tuple(answers[i] for i in order),
                answer_idx=order.index(0),
            )
        )
    return items


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
