"""The Harness: the full, immutable recipe for turning a benchmark item into a prompt + a parse.

A named real harness (lm-eval's, the authors', Inspect's) is a `Harness` config in this *one*
uniform space -- which is what lets us sweep *between* them and lets our systems knobs compose
(docs/design/architecture.md). This module is pure: render an item into a prompt, parse a
response. Running against a model lives in `engine.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from wobblelab.benchmark import LETTERS, MCItem

Scoring = Literal["gen", "ll_letter", "ll_choice"]
# gen        -- sample/greedy generate, extract the answer with a regex cascade
# ll_letter  -- argmax over first-token letter logprobs (Provider.rank_letters); deterministic
# ll_choice  -- full-completion log-likelihood per choice (what lm-eval MC does); needs a
#               provider that scores completions -- not yet implemented (see engine.py)


@dataclass(frozen=True)
class PromptSpec:
    """How to render an `MCItem` into a prompt. Configurable enough to express the real named
    harnesses -- the `(A)` vs `A.` option format, the `Choices:`/`Options:` header, the answer
    instruction. Small differences here move the number a lot, so they are first-class fields."""

    question_prefix: str = ""
    choices_header: str = "\n"
    option_format: str = "{letter}. {text}"
    option_sep: str = "\n"
    answer_trigger: str = ""
    system_prompt: str = ""
    n_shots: int = 0
    cot: bool = False


@dataclass(frozen=True)
class ScoringSpec:
    method: Scoring = "gen"
    extraction_patterns: tuple[str, ...] = (
        r"\b([A-Za-z])\b",
    )  # first valid-letter capture wins


@dataclass(frozen=True)
class SamplingSpec:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop: tuple[str, ...] = ()
    seed: int = 0


@dataclass(frozen=True)
class OrderSpec:
    kind: Literal["canonical", "permutation"] = "canonical"
    order: tuple[int, ...] | None = (
        None  # explicit presentation order when kind == permutation
    )


@dataclass(frozen=True)
class Harness:
    """The bundled recipe. Immutable; a `Knob` produces a new one via `dataclasses.replace`."""

    name: str
    prompt: PromptSpec
    scoring: ScoringSpec
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    order: OrderSpec = field(default_factory=OrderSpec)


def presentation_order(harness: Harness, item: MCItem) -> tuple[int, ...]:
    """The order option originals are shown in (a tuple of original indices, one per slot)."""
    if harness.order.kind == "permutation" and harness.order.order is not None:
        return harness.order.order
    return tuple(range(len(item.options)))


def render_prompt(
    harness: Harness, item: MCItem, order: tuple[int, ...] | None = None
) -> str:
    """Render the item under this harness. Option at `order[pos]` is shown at slot `pos`
    (lettered `LETTERS[pos]`)."""
    if order is None:
        order = presentation_order(harness, item)
    p = harness.prompt
    if p.n_shots:
        raise NotImplementedError(
            "few-shot rendering is a V1 concern; n_shots must be 0 for now"
        )
    body = f"{p.question_prefix}{item.question}{p.choices_header}"
    body += p.option_sep.join(
        p.option_format.format(letter=LETTERS[pos], text=item.options[orig])
        for pos, orig in enumerate(order)
    )
    if p.answer_trigger:
        body += "\n\n" + p.answer_trigger
    return f"{p.system_prompt}\n\n{body}" if p.system_prompt else body


def extract(harness: Harness, text: str, n_options: int) -> int | None:
    """Regex cascade -> presentation position (0-based), or None. First pattern whose capture is
    a valid letter for this option count wins."""
    valid = LETTERS[:n_options]
    for pat in harness.scoring.extraction_patterns:
        m = re.search(pat, text)
        if m and m.group(1).upper() in valid:
            return valid.index(m.group(1).upper())
    return None
