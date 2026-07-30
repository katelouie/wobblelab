"""Benchmark = its item source + its canonical Harness + named landmark harnesses + published
reference scores. A thin config over the uniform `Harness` space (docs/design/architecture.md).

Named harnesses (lm-eval / Inspect / the authors') are *landmarks*; the handshake validates each
against `reference_scores`. Only the authors' GPQA harness is encoded so far -- it is the one we
have verbatim; lm-eval's and Inspect's get added and fidelity-checked at handshake time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from wobblelab.benchmark import MCItem
from wobblelab.harness import Harness, PromptSpec, SamplingSpec, ScoringSpec

# The official GPQA answer-extraction cascade (github.com/idavidrein/gpqa); first valid A-D wins.
_GPQA_EXTRACTION = (
    r"answer is \((.)\)",
    r"Answer: \((.)\)",
    r"answer: \((.)\)",
    r"answer \((.)\)",
    r"\((.)\)",
)

# The authors' reference harness: free generation, `(A)` options, "The correct answer is (X)",
# the extraction cascade, temp 0, a budget generous enough for chain-of-thought to finish.
GPQA_AUTHORS = Harness(
    name="gpqa:authors",
    prompt=PromptSpec(
        question_prefix="What is the correct answer to this question: ",
        choices_header="\n\nChoices:\n",
        option_format="({letter}) {text}",
        answer_trigger='Format your response as follows: "The correct answer is (insert answer here)"',
    ),
    scoring=ScoringSpec(method="gen", extraction_patterns=_GPQA_EXTRACTION),
    sampling=SamplingSpec(temperature=0.0, max_tokens=2048),
)


@dataclass(frozen=True)
class Benchmark:
    name: str
    load: Callable[
        [], list[MCItem]
    ]  # item source (lazy; network/auth deferred to call time)
    canonical: Harness
    landmarks: dict[str, Harness] = field(default_factory=dict)
    # reference_scores[harness_name][model] = published score, for the handshake gate
    reference_scores: dict[str, dict[str, float]] = field(default_factory=dict)


def _load_gpqa_diamond() -> list[MCItem]:
    from wobblelab.loaders import load_gpqa_diamond

    return load_gpqa_diamond()


GPQA_DIAMOND = Benchmark(
    name="gpqa_diamond",
    load=_load_gpqa_diamond,
    canonical=GPQA_AUTHORS,
    landmarks={"authors": GPQA_AUTHORS},
    reference_scores={},
)
