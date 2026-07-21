"""The benchmark-family seam: perturb an item, run one perturbation, score it into an Outcome.

squishlab's two axes are universal — rerun the same input (observational), or perturb it in a
meaning-preserving way (interventional) — so the *harness* (reruns, CIs, squish aggregation)
is written once and shared. What differs per benchmark family is only (a) what a meaning-
preserving perturbation IS and (b) how you score a response. A `Task` supplies exactly those
two things; `evaluate()` in report.py drives any Task without knowing its family.

`MultipleChoiceTask` is the implemented family (MMLU, MMLU-Pro, TruthfulQA-MC1, GPQA, ARC:
any number of options, one correct). `CodeExecutionTask` and `JudgeTask` are documented seams
— they raise until built, but they show that SWE-bench-style execution evals and judge-scored
evals slot into the SAME harness, changing only these two methods.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from squishlab.benchmark import (
    MCItem,
    format_prompt,
    orders_correct_at_each_position,
    parse_answer,
    presented_to_original,
)
from squishlab.provider import Provider


@dataclass(frozen=True)
class Outcome:
    """The scored result of running one perturbation once.

    Attributes:
        correct: graded correctness, or None if the task is ungraded / the response was
            unparseable. Aggregated into accuracy.
        content: the canonical "what the model produced", compared ACROSS perturbations
            (interventional squish) and ACROSS reruns (observational squish). For a choice
            task it's the chosen *original* option index; for a code task it'd be a hash of
            the observable behavior. None if the model produced nothing usable.
        slot / correct_slot / n_slots: position bookkeeping for *choice* tasks — which
            presentation slot the model picked, which slot held the answer, and how many
            slots there were. Left None by families with no notion of answer position
            (code, judge), which simply get no position-bias readout.
    """

    correct: bool | None
    content: Hashable | None
    slot: int | None = None
    correct_slot: int | None = None
    n_slots: int | None = None


@runtime_checkable
class Task(Protocol):
    """A benchmark family. Two methods + two attributes; the harness supplies everything else.

    - ``name``: short label for the report/provenance (e.g. "mc:ll").
    - ``deterministic``: if True, reruns collapse to one (log-likelihood argmax, greedy decode)
      so the harness doesn't waste calls re-sampling an identical answer.
    - ``perturbations(item)``: the meaning-preserving variants — the interventional axis.
    - ``run(provider, item, perturbation, seed)``: run one perturbation, score it -> Outcome.
    """

    name: str
    deterministic: bool

    def perturbations(self, item) -> list: ...

    def run(self, provider: Provider, item, perturbation, seed: int) -> Outcome: ...


class MultipleChoiceTask:
    """Any question with a fixed option set and one correct answer, 2–26 options.

    The interventional perturbation is an option *ordering*. We use the position-debiasing
    set (correct answer placed at each slot, distractors kept in relative order), which does
    double duty: it exposes position bias AND is a guaranteed meaning-preserving perturbation
    (a reorder must never change which content the model picks). Nothing here assumes a fixed
    option count, so variable-length benchmarks (TruthfulQA-MC1) work unchanged.
    """

    def __init__(self, scoring: str = "gen") -> None:
        if scoring not in ("gen", "ll"):
            raise ValueError(f"scoring must be 'gen' or 'll', got {scoring!r}")
        self.scoring = scoring
        self.name = f"mc:{scoring}"
        self.deterministic = (
            scoring == "ll"
        )  # log-likelihood argmax -> reruns collapse to 1

    def perturbations(self, item: MCItem) -> list[tuple[int, ...]]:
        return orders_correct_at_each_position(item)

    def run(
        self, provider: Provider, item: MCItem, order: tuple[int, ...], seed: int
    ) -> Outcome:
        n = len(item.options)
        prompt = format_prompt(item, order)
        if self.scoring == "ll":
            pos, _ = provider.rank_letters(prompt, n, seed=seed)
        else:
            pos = parse_answer(provider.ask(prompt, seed=seed), n)
        content = presented_to_original(
            pos, order
        )  # chosen ORIGINAL index (order-invariant)
        return Outcome(
            correct=(content == item.answer_idx) if pos is not None else False,
            content=content,
            slot=pos,
            correct_slot=order.index(
                item.answer_idx
            ),  # where the answer sits in this order
            n_slots=n,
        )


class CodeExecutionTask:
    """PLANNED seam (HumanEval / MBPP / SWE-bench). Not yet implemented — shown to prove the
    architecture holds for execution evals without harness changes.

    Sketch of the implementation:
      - ``perturbations(item)``: meaning-preserving rewrites of the problem statement
        (paraphrase the docstring, rename entities, reorder the given examples) — the code
        analogue of shuffling options. The canonical statement is one of them.
      - ``run(provider, item, perturbation, seed)``: render the problem, ``provider.ask()`` to
        generate a solution, execute it against the item's test suite in a sandbox, and return
        ``Outcome(correct=tests_pass, content=<which tests passed / behavior hash>)``. ``slot``
        stays None (no answer position). ``deterministic=False`` (generation).

    The harness needs nothing new: accuracy = pass rate (bootstrapped over problems),
    observational squish = pass/fail flips across reruns, interventional squish = pass/fail or
    behavior flips across paraphrases. The real work is sandboxed execution, not the seam.
    """

    name = "code:exec"
    deterministic = False

    def perturbations(self, item):
        raise NotImplementedError(
            "CodeExecutionTask is a planned seam, not yet implemented"
        )

    def run(self, provider, item, perturbation, seed):
        raise NotImplementedError(
            "CodeExecutionTask is a planned seam, not yet implemented"
        )


class JudgeTask:
    """PLANNED seam (MT-Bench / AlpacaEval). A second model scores a free-form response.

    Sketch: ``perturbations`` = prompt paraphrases; ``run`` = ``provider.ask()`` to produce a
    response, then a judge model scores it, returning ``Outcome(correct=score>=threshold or
    None, content=score_bucket)``. Because the judge is itself a ``Provider``, "does the SCORE
    move when you rerun or reword" is measurable with the very same machinery — judge-squish,
    for free.
    """

    name = "judge"
    deterministic = False

    def perturbations(self, item):
        raise NotImplementedError("JudgeTask is a planned seam, not yet implemented")

    def run(self, provider, item, perturbation, seed):
        raise NotImplementedError("JudgeTask is a planned seam, not yet implemented")
