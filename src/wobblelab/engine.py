"""Run a `Harness` over items against a `Provider` -> `Outcome`s.

The execution layer: concurrent, order-deterministic (identical numbers to a sequential run),
pure aggregation on top. Everything an `Outcome` carries is provenance-tagged so a study can be
audited (docs/design/architecture.md).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from wobblelab.benchmark import MCItem
from wobblelab.harness import Harness, extract, presentation_order, render_prompt
from wobblelab.provider import Provider


@dataclass(frozen=True)
class Outcome:
    """One (harness, item, seed) result, fully traceable."""

    item_id: str
    harness: str
    seed: int
    order: tuple[int, ...]
    raw: str | None  # model text (gen) or None (ll)
    parsed_pos: int | None  # chosen presentation position
    parsed_orig: int | None  # mapped back to the original option index
    correct: bool
    n_options: int


def run_item(provider: Provider, harness: Harness, item: MCItem, seed: int) -> Outcome:
    order = presentation_order(harness, item)
    prompt = render_prompt(harness, item, order)
    n = len(item.options)
    method = harness.scoring.method
    if method == "gen":
        raw = provider.ask(prompt, seed)
        pos = extract(harness, raw, n)
    elif method == "ll_letter":
        pos, _ = provider.rank_letters(prompt, n, seed)
        raw = None
    elif method == "ll_choice":
        raise NotImplementedError(
            "ll_choice (full-completion log-likelihood, what lm-eval MC uses) needs a provider "
            "that scores whole completions; the Provider seam exposes only ask + rank_letters "
            "today. Add that method at handshake time."
        )
    else:
        raise ValueError(f"unknown scoring method: {method!r}")
    orig = order[pos] if pos is not None and 0 <= pos < len(order) else None
    return Outcome(
        item_id=item.id,
        harness=harness.name,
        seed=seed,
        order=order,
        raw=raw,
        parsed_pos=pos,
        parsed_orig=orig,
        correct=(orig == item.answer_idx),
        n_options=n,
    )


def run_harness(
    provider: Provider,
    harness: Harness,
    items: list[MCItem],
    *,
    concurrency: int = 1,
    seed: int = 0,
) -> list[Outcome]:
    """Run every item through the harness once at `seed`. Order-preserving; concurrency changes
    only speed, never the returned outcomes."""
    if concurrency <= 1:
        return [run_item(provider, harness, it, seed) for it in items]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(lambda it: run_item(provider, harness, it, seed), items))


def accuracy(outcomes: list[Outcome]) -> float:
    return sum(o.correct for o in outcomes) / len(outcomes) if outcomes else 0.0


def coverage(outcomes: list[Outcome]) -> float:
    """Fraction that produced a parseable answer -- the health signal that catches a truncation
    or format collapse before it masquerades as a low score."""
    if not outcomes:
        return 0.0
    return sum(o.parsed_pos is not None for o in outcomes) / len(outcomes)
