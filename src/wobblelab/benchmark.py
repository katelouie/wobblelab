"""Multiple-choice benchmark machinery: the option-permutation perturbation.

The clean interventional axis for a benchmark is **permuting the answer options**. It is
guaranteed meaning-preserving (reordering cannot change which option is correct), so any
change in the model's answer is pure wobble — no paraphrase-quality confound (contrast
F-006) — and it exposes position bias for free.

An item's canonical option order is fixed; a *presentation order* is a tuple of original
option indices. The model sees letters A, B, C... assigned by presentation position, so
we parse the chosen letter -> presentation position -> original index, which makes the
chosen answer comparable across orderings.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass

LETTERS = (
    string.ascii_uppercase
)  # up to 26 options (MMLU-Pro 10, TruthfulQA-MC1 ~13, ...)


@dataclass(frozen=True)
class MCItem:
    id: str
    question: str
    options: tuple[str, ...]  # canonical order
    answer_idx: int  # index into options of the correct one


def format_prompt(item: MCItem, order: tuple[int, ...]) -> str:
    """Render the question with options in `order`, lettered by presentation position."""
    lines = [item.question, ""]
    for pos, orig_i in enumerate(order):
        lines.append(f"{LETTERS[pos]}. {item.options[orig_i]}")
    letters = "/".join(LETTERS[: len(order)])
    lines.append("")
    lines.append(f"Answer with only the letter ({letters}).")
    return "\n".join(lines)


def parse_answer(text: str, n_options: int) -> int | None:
    """Return the chosen *presentation position* (0-based), or None if unparseable.

    The match is bounded to exactly the valid letters for this item's option count, so a
    4-option question can't be "answered" by the pronoun "I" (the 9th letter) leaking out of
    a chatty response, and a 12-option question can legitimately land on "J". First in-range
    standalone letter wins, which is what the "answer with only the letter" prompt elicits.
    """
    if n_options < 1:
        return None
    valid = LETTERS[:n_options]
    m = re.search(rf"\b([{valid}])\b", text.upper())
    return LETTERS.index(m.group(1)) if m else None


def presented_to_original(pos: int | None, order: tuple[int, ...]) -> int | None:
    """Map a chosen presentation position back to the original option index."""
    if pos is None or not (0 <= pos < len(order)):
        return None
    return order[pos]


def orders_correct_at_each_position(item: MCItem) -> list[tuple[int, ...]]:
    """One presentation order per option slot, placing the correct answer at that slot.

    Distractors keep their relative order, so the only thing that varies is *where* the
    right answer sits — isolating position bias (index p -> correct answer at position p).
    """
    a, k = item.answer_idx, len(item.options)
    distractors = [i for i in range(k) if i != a]
    return [tuple(distractors[:p] + [a] + distractors[p:]) for p in range(k)]


def modal(values) -> tuple[Hashable | None, float]:
    """Most common non-None value and its fraction of the decided votes.

    Works over any hashable outcome, not just option indices — a code task's behavior hash
    or a judge's score bucket aggregate the same way, which is what lets one harness measure
    interventional/observational stability across benchmark families.
    """
    c = Counter(v for v in values if v is not None)
    if not c:
        return None, 0.0
    val, cnt = c.most_common(1)[0]
    return val, cnt / sum(c.values())
