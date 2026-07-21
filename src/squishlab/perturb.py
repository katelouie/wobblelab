"""Perturbations: the semantic-preserving variations that make up the interventional axis.

Pillar 2, in full. A benchmark item can be presented many equivalent ways, options
reordered, the instruction reworded, the question paraphrased, the whole thing translated,
and a model's answer should not care which. How much it *does* care, broken out PER KIND of
change, is the squish we report. That per-kind breakdown is the point: "this model is solid
under reordering but flips 30% of the time when you reword the question" is the reliability
fact a leaderboard never tells you.

Two families:
  - **model-free** (ReorderOptions, RephraseInstruction): guaranteed meaning-preserving,
    because we author the transformation. Safe, cheap, no confound.
  - **model-based** (ParaphraseWithModel, TranslateWithModel): a *perturber* model generates
    the variant. Full breadth (reword the actual question, translate it), but meaning
    preservation is not guaranteed, the generator can drift, so a hot-spot here means
    "investigate", not "proven" (lab-journal D-006). Flagged, not hidden.

A `Presentation` is one concrete rendering; `origin[slot]` records which ORIGINAL option is
shown at each slot, so the chosen answer stays comparable across every kind of perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from squishlab.benchmark import LETTERS, MCItem, orders_correct_at_each_position
from squishlab.provider import Provider


@dataclass(frozen=True)
class Presentation:
    """One equivalent way to show an item to the model."""

    question: str
    options: tuple[
        str, ...
    ]  # display order and display text (maybe reworded/translated)
    origin: tuple[
        int, ...
    ]  # origin[slot] = the ORIGINAL option index shown at that slot
    kind: str  # "reorder" | "rephrase" | "paraphrase" | "translate:<lang>"
    instruction: str | None = (
        None  # overrides the answer instruction (framing perturbations)
    )
    probes_position: bool = False  # True only for the reorder set (it moves the answer)


def render(pres: Presentation) -> str:
    """Presentation -> the prompt string the model actually sees."""
    lines = [pres.question, ""]
    for i, opt in enumerate(pres.options):
        lines.append(f"{LETTERS[i]}. {opt}")
    letters = "/".join(LETTERS[: len(pres.options)])
    lines.append("")
    lines.append(pres.instruction or f"Answer with only the letter ({letters}).")
    return "\n".join(lines)


@runtime_checkable
class Perturbation(Protocol):
    """Maps an item to the set of equivalent presentations of one kind."""

    kind: str

    def present(self, item: MCItem) -> list[Presentation]: ...


class ReorderOptions:
    """Model-free, guaranteed meaning-preserving: the correct answer placed at each slot.

    Doubles as the position-debiasing set (so accuracy and position bias come from here) and
    as the reorder interventional axis (a shuffle must never change the chosen content).
    """

    kind = "reorder"

    def present(self, item: MCItem) -> list[Presentation]:
        return [
            Presentation(
                question=item.question,
                options=tuple(item.options[i] for i in order),
                origin=tuple(order),
                kind=self.kind,
                probes_position=True,
            )
            for order in orders_correct_at_each_position(item)
        ]


DEFAULT_INSTRUCTIONS = (
    "Which option is correct? Reply with only the letter ({letters}).",
    "Select the single best answer and respond with just its letter ({letters}).",
    "Choose the correct option; give only its letter ({letters}).",
)


class RephraseInstruction:
    """Model-free: the SAME question and options, under equivalent instruction phrasings.

    A guaranteed-safe slice of "prompt rephrasing" (we control the templates), so it isolates
    the model's sensitivity to how the *ask* is worded, with no meaning-drift confound.
    """

    kind = "rephrase"

    def __init__(self, instructions: tuple[str, ...] = DEFAULT_INSTRUCTIONS) -> None:
        self.instructions = instructions

    def present(self, item: MCItem) -> list[Presentation]:
        n = len(item.options)
        letters = "/".join(LETTERS[:n])
        ident = tuple(range(n))
        return [
            Presentation(
                question=item.question,
                options=tuple(item.options),
                origin=ident,
                kind=self.kind,
                instruction=tmpl.format(letters=letters),
            )
            for tmpl in self.instructions
        ]


class ParaphraseWithModel:
    """Model-based: reword the QUESTION with a perturber model (full prompt rephrasing).

    Meaning preservation is not guaranteed (the paraphraser can drift), so treat a paraphrase
    hot-spot as a lead to investigate, not a proven defect. Pass a *different* model as the
    perturber than the one under test where you can.
    """

    kind = "paraphrase"

    def __init__(self, perturber: Provider, n: int = 3) -> None:
        self.perturber = perturber
        self.n = n

    def present(self, item: MCItem) -> list[Presentation]:
        ask = (
            "Reword the following question so it means exactly the same thing, changing only "
            "the wording. Output only the reworded question.\n\n" + item.question
        )
        ident = tuple(range(len(item.options)))
        out = []
        for s in range(self.n):
            q = self.perturber.ask(ask, seed=s).strip() or item.question
            out.append(
                Presentation(
                    question=q,
                    options=tuple(item.options),
                    origin=ident,
                    kind=self.kind,
                )
            )
        return out


class TranslateWithModel:
    """Model-based: translate the question and options into another language.

    Pillar 2's "translation" axis. Same meaning-drift caveat as paraphrase, plus culture-
    boundedness for culturally-loaded items (a category can genuinely differ across languages,
    which is a finding, not squish). One perturber call per text.
    """

    def __init__(self, perturber: Provider, language: str) -> None:
        self.perturber = perturber
        self.language = language
        self.kind = f"translate:{language}"

    def _tr(self, text: str, seed: int) -> str:
        ask = (
            f"Translate the following into {self.language}, preserving its meaning exactly. "
            f"Output only the translation.\n\n{text}"
        )
        return self.perturber.ask(ask, seed=seed).strip() or text

    def present(self, item: MCItem) -> list[Presentation]:
        q = self._tr(item.question, 0)
        opts = tuple(self._tr(o, i + 1) for i, o in enumerate(item.options))
        return [
            Presentation(
                question=q,
                options=opts,
                origin=tuple(range(len(item.options))),
                kind=self.kind,
            )
        ]
