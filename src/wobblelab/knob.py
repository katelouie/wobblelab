"""The Knob: a named transform `Harness -> [Harness]` that deviates the canonical harness in
exactly one dimension. Concrete knobs (reorder, temperature, budget, paraphrase, ...) arrive in
V1; this module defines the *type* so the architecture is expressed and the study layer can be
built against it. See docs/design/architecture.md and roadmap.md.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from wobblelab.harness import Harness

Family = Literal["harness", "perturbation", "systems", "task"]


@runtime_checkable
class Knob(Protocol):
    """A one-dimensional deviation from a harness.

    `meaning_preserving` tags whether turning it *should* change the correct answer: False means
    movement is choice-dependence (a harness knob like CoT/shots); True means movement is a defect
    (a perturbation like reorder/paraphrase). `family` groups the taxonomy.
    """

    name: str
    family: Family
    meaning_preserving: bool

    def variants(self, canonical: Harness) -> list[tuple[Any, Harness]]:
        """`(value, harness)` pairs -- the harness with this one knob turned to each value.
        `Knob.variants(anchor)` is the one-knob-at-a-time sweep from the anchor."""
        ...
