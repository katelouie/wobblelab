"""The catalog artifact: one call, one model, a benchmark number you can actually trust.

`evaluate(provider, items)` runs the position-debiased benchmark and returns a `ModelReport`
-- accuracy WITH a confidence interval, the reorder squish (how often the chosen answer's
content flips under a meaning-preserving option shuffle), the position-bias profile (the
mirage detector), and full provenance. `compare()` runs it across models. This is the
benchmark-squish measurement from experiments/bench.py, lifted out of a one-off script into
the reusable surface the whole project points at.

Everything here is provider-agnostic (it drives the Provider seam) and item-agnostic (the
caller supplies MCItems; loading a dataset stays out of the library). Deterministic given a
deterministic provider, so it is fully testable with MockProvider and no model running.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from squishlab.benchmark import (
    LETTERS,
    MCItem,
    format_prompt,
    modal,
    orders_correct_at_each_position,
    parse_answer,
    presented_to_original,
)
from squishlab.provider import Provider
from squishlab.stats import bootstrap_ci


@dataclass(frozen=True)
class ModelReport:
    """A single (model, benchmark) catalog entry: the number and how much to trust it."""

    model: str
    benchmark: str
    scoring: str  # "gen" (sampled) or "ll" (log-likelihood)
    n_items: int
    n_rerun: int
    accuracy: float  # position-debiased
    accuracy_ci: tuple[float, float]  # bootstrapped over items
    reorder_squish: (
        float  # fraction of items whose modal answer-content flips on shuffle
    )
    position_swing: float  # max-min accuracy across answer positions (0 = unbiased)
    accuracy_by_position: tuple[float, ...]
    chosen_distribution: tuple[float, ...]  # which letter the model reaches for
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "scoring": self.scoring,
            "n_items": self.n_items,
            "n_rerun": self.n_rerun,
            "accuracy": round(self.accuracy, 3),
            "accuracy_ci": [round(x, 3) for x in self.accuracy_ci],
            "reorder_squish": round(self.reorder_squish, 3),
            "position_swing": round(self.position_swing, 3),
            "accuracy_by_position": [round(x, 3) for x in self.accuracy_by_position],
            "chosen_distribution": [round(x, 3) for x in self.chosen_distribution],
            "provenance": self.provenance,
        }

    def to_markdown(self) -> str:
        lo, hi = self.accuracy_ci
        k = len(self.accuracy_by_position)
        by_pos = "  ".join(
            f"{LETTERS[i]} {a:.2f}" for i, a in enumerate(self.accuracy_by_position)
        )
        chosen = "  ".join(
            f"{LETTERS[i]} {c:.0%}" for i, c in enumerate(self.chosen_distribution)
        )
        return (
            f"**{self.model}** on `{self.benchmark}` ({self.scoring}, "
            f"{self.n_items} items × {self.n_rerun})\n\n"
            f"- **accuracy: {self.accuracy:.1%}**  (95% CI {lo:.1%}–{hi:.1%})\n"
            f"- reorder squish: {self.reorder_squish:.1%} of answers flip content on option shuffle\n"
            f"- position swing: {self.position_swing:.2f} across {k} slots  "
            f"({'flat, unbiased' if self.position_swing < 0.1 else 'position-biased'})\n"
            f"- accuracy by answer position: {by_pos}\n"
            f"- letter the model reaches for: {chosen}\n"
        )

    def __str__(self) -> str:
        return self.to_markdown()


def _gen_choice(provider: Provider, prompt: str, n: int, seed: int) -> int | None:
    return parse_answer(provider.ask(prompt, seed=seed), n)


def _ll_choice(provider: Provider, prompt: str, n: int, seed: int) -> int | None:
    pos, _ = provider.rank_letters(prompt, n, seed=seed)
    return pos


def evaluate(
    provider: Provider,
    items: list[MCItem],
    *,
    benchmark: str = "benchmark",
    model: str | None = None,
    scoring: str = "gen",
    n_rerun: int = 5,
    boot_seed: int = 1,
) -> ModelReport:
    """Position-debiased accuracy + CI + reorder squish + position bias for one model.

    Places the correct answer at every option slot (distractors keep their order) and reruns
    each, so accuracy is debiased and position bias is measured directly. `scoring="ll"` reads
    the model's letter log-probabilities (deterministic, so reruns collapse to one);
    `scoring="gen"` samples and parses. Accuracy CI is bootstrapped over items -- the correct
    unit, since reruns of one item are correlated.
    """
    if scoring not in ("gen", "ll"):
        raise ValueError(f"scoring must be 'gen' or 'll', got {scoring!r}")
    if not items:
        raise ValueError("no items to evaluate")
    choose = _ll_choice if scoring == "ll" else _gen_choice
    rerun = 1 if scoring == "ll" else n_rerun  # ll is deterministic

    k = len(items[0].options)
    pos_correct = [0] * k
    pos_total = [0] * k
    chosen = [0] * k
    per_item_acc: list[float] = []
    per_item_reorder: list[float] = []

    for it in items:
        n = len(it.options)
        orders = orders_correct_at_each_position(
            it
        )  # index p -> correct sits at slot p
        modal_content = []
        correct = 0
        total = 0
        for p, order in enumerate(orders):
            chosen_orig = []
            for s in range(rerun):
                pos = choose(provider, format_prompt(it, order), n, s)
                if pos is not None:
                    chosen[pos] += 1
                    chosen_orig.append(presented_to_original(pos, order))
                pos_total[p] += 1
                if pos == p:
                    pos_correct[p] += 1
                    correct += 1
                total += 1
            mc, _ = modal(chosen_orig)
            modal_content.append(mc)
        per_item_acc.append(correct / total)
        _, content_frac = modal(modal_content)
        per_item_reorder.append(
            1 - content_frac
        )  # 0 = same content across all shuffles

    accuracy = sum(per_item_acc) / len(per_item_acc)
    acc_ci = bootstrap_ci(
        per_item_acc, lambda s: sum(s) / len(s), n_boot=4000, seed=boot_seed
    )
    acc_by_pos = tuple(c / t if t else 0.0 for c, t in zip(pos_correct, pos_total))
    total_chosen = sum(chosen) or 1
    return ModelReport(
        model=model or provider.config().get("model", "model"),
        benchmark=benchmark,
        scoring=scoring,
        n_items=len(items),
        n_rerun=rerun,
        accuracy=accuracy,
        accuracy_ci=tuple(acc_ci),
        reorder_squish=sum(per_item_reorder) / len(per_item_reorder),
        position_swing=max(acc_by_pos) - min(acc_by_pos),
        accuracy_by_position=acc_by_pos,
        chosen_distribution=tuple(c / total_chosen for c in chosen),
        provenance={
            "config": provider.config(),
            "scoring": scoring,
            "boot_seed": boot_seed,
        },
    )


def compare(
    providers: dict[str, Provider], items: list[MCItem], **kwargs
) -> list[ModelReport]:
    """Evaluate several named models on the same items. `model` label = the dict key."""
    return [evaluate(p, items, model=name, **kwargs) for name, p in providers.items()]


def compare_markdown(reports: list[ModelReport]) -> str:
    """A comparison table for a set of reports -- the shape a catalog page wants."""
    header = (
        "| model | accuracy | 95% CI | reorder squish | position swing |\n"
        "|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {r.model} | {r.accuracy:.1%} | "
        f"{r.accuracy_ci[0]:.1%}–{r.accuracy_ci[1]:.1%} | "
        f"{r.reorder_squish:.1%} | {r.position_swing:.2f} |\n"
        for r in reports
    )
    return header + rows
