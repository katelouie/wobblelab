"""The catalog artifact: one call, one model, a benchmark number you can actually trust.

`evaluate(provider, items)` runs a benchmark through a `Task` (default: multiple-choice) and
returns a `ModelReport` — accuracy WITH a confidence interval, both squish axes (does the
answer move across reruns / across meaning-preserving perturbations), and, for choice tasks,
the position-bias profile that catches the mirage. `compare()` runs it across models.

The harness is *family-agnostic*: it owns reruns, CIs, and squish aggregation, and delegates
"how to perturb" and "how to score" to the Task. So the same code produces a report for MMLU,
TruthfulQA (variable option counts), and — once those Tasks are built — SWE-bench-style
execution evals, changing nothing here. It is also item-agnostic (the caller supplies items;
dataset loading stays in squishlab.loaders) and deterministic given a deterministic provider,
so it's fully testable with MockProvider and no model running.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from squishlab.benchmark import LETTERS, modal
from squishlab.provider import Provider
from squishlab.stats import bootstrap_ci
from squishlab.task import MultipleChoiceTask, Task


@dataclass(frozen=True)
class ModelReport:
    """A single (model, benchmark) catalog entry: the number and how much to trust it.

    Universal fields apply to every benchmark family. The position fields are populated only
    for choice tasks (they stay None when the family has no notion of answer position).
    """

    model: str
    benchmark: str
    task: str  # the Task's name, e.g. "mc:ll"
    n_items: int
    n_rerun: int
    accuracy: float | None  # position-debiased; None if the task is ungraded
    accuracy_ci: tuple[float, float] | None  # bootstrapped over items
    interventional_squish: (
        float  # answer-content flips across meaning-preserving perturbations
    )
    observational_squish: float  # answer-content flips across reruns of the same input
    accuracy_by_position: tuple[float, ...] | None = None  # choice tasks only
    position_swing: float | None = None  # max-min accuracy across answer positions
    chosen_distribution: tuple[float, ...] | None = (
        None  # which slot the model reaches for
    )
    squish_by_kind: dict[str, float] = field(
        default_factory=dict
    )  # per perturbation kind
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "benchmark": self.benchmark,
            "task": self.task,
            "n_items": self.n_items,
            "n_rerun": self.n_rerun,
            "accuracy": round(self.accuracy, 3) if self.accuracy is not None else None,
            "accuracy_ci": [round(x, 3) for x in self.accuracy_ci]
            if self.accuracy_ci
            else None,
            "interventional_squish": round(self.interventional_squish, 3),
            "observational_squish": round(self.observational_squish, 3),
            "squish_by_kind": {k: round(v, 3) for k, v in self.squish_by_kind.items()},
            "provenance": self.provenance,
        }
        if self.accuracy_by_position is not None:
            d["accuracy_by_position"] = [round(x, 3) for x in self.accuracy_by_position]
            d["position_swing"] = round(self.position_swing, 3)
            d["chosen_distribution"] = [round(x, 3) for x in self.chosen_distribution]
        return d

    def to_markdown(self) -> str:
        lines = [
            f"**{self.model}** on `{self.benchmark}` ({self.task}, "
            f"{self.n_items} items × {self.n_rerun})\n"
        ]
        if self.accuracy is not None:
            lo, hi = self.accuracy_ci
            lines.append(
                f"- **accuracy: {self.accuracy:.1%}**  (95% CI {lo:.1%}–{hi:.1%})"
            )
        lines.append(
            f"- interventional squish: {self.interventional_squish:.1%} of answers flip "
            f"content under a meaning-preserving perturbation"
        )
        lines.append(
            f"- observational squish: {self.observational_squish:.1%} rerun instability"
        )
        if len(self.squish_by_kind) > 1:
            per_kind = " · ".join(
                f"{k} {v:.0%}" for k, v in sorted(self.squish_by_kind.items())
            )
            lines.append(f"- squish by perturbation kind: {per_kind}")
        if self.accuracy_by_position is not None:
            k = len(self.accuracy_by_position)
            by_pos = "  ".join(
                f"{LETTERS[i]} {a:.2f}" for i, a in enumerate(self.accuracy_by_position)
            )
            chosen = "  ".join(
                f"{LETTERS[i]} {c:.0%}" for i, c in enumerate(self.chosen_distribution)
            )
            flat = (
                "flat, unbiased"
                if (self.position_swing or 0) < 0.1
                else "position-biased"
            )
            lines.append(
                f"- position swing: {self.position_swing:.2f} across {k} slots  ({flat})"
            )
            lines.append(f"- accuracy by answer position: {by_pos}")
            lines.append(f"- letter the model reaches for: {chosen}")
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_markdown()


def evaluate(
    provider: Provider,
    items: list,
    *,
    task: Task | None = None,
    benchmark: str = "benchmark",
    model: str | None = None,
    scoring: str = "gen",
    n_rerun: int = 5,
    boot_seed: int = 1,
    seed_offset: int = 0,
    concurrency: int = 1,
) -> ModelReport:
    """Run one model on one benchmark and return its report.

    ``task`` selects the benchmark family (default: ``MultipleChoiceTask(scoring)``). For each
    item the harness runs every perturbation (the interventional axis) ``n_rerun`` times (the
    observational axis; collapsed to 1 when the task is deterministic), then aggregates:
    accuracy with a bootstrap-over-items CI (the correct unit — reruns of one item are
    correlated), interventional squish (does the modal content change across perturbations),
    observational squish (does it change across reruns), and — for choice tasks — accuracy by
    answer position + the chosen-letter distribution. Variable option counts are handled;
    position stats aggregate over whichever slots actually occur.

    ``concurrency`` > 1 dispatches the provider calls through a thread pool (the aggregation
    is unchanged and order-deterministic, so the numbers are identical to the sequential run).
    It only helps against a server that batches concurrent requests -- vLLM, llama.cpp's
    ``--parallel`` server, mlx-lm, or ollama with ``OLLAMA_NUM_PARALLEL`` set.
    """
    if not items:
        raise ValueError("no items to evaluate")
    task = task or MultipleChoiceTask(scoring=scoring)
    rerun = 1 if getattr(task, "deterministic", False) else n_rerun

    # Phase 1: the perturbations for each item (model-based kinds make perturber calls here).
    per_item_perts = [task.perturbations(item) for item in items]

    # Phase 2: run every (item, perturbation, rerun) provider call. The provider stays
    # synchronous; a thread pool just keeps many requests in flight so a batching server
    # (vLLM / llama.cpp / mlx) can coalesce them. concurrency=1 -> sequential, identical output.
    units = [
        (i, j, s)
        for i, perts in enumerate(per_item_perts)
        for j in range(len(perts))
        for s in range(rerun)
    ]

    def _run_unit(u):
        i, j, s = u
        return u, task.run(
            provider, items[i], per_item_perts[i][j], seed_offset * 1000 + s
        )

    outcomes: dict[tuple[int, int, int], object] = {}
    if concurrency > 1 and len(units) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for u, o in pool.map(_run_unit, units):
                outcomes[u] = o
    else:
        for u in units:
            outcomes[u] = _run_unit(u)[1]

    # Phase 3: aggregate. Pure, order-deterministic, reads the gathered outcomes -- so the
    # numbers are identical whether the calls ran sequentially or concurrently.
    per_item_acc: list[float] = []
    per_item_interv: list[float] = []
    obs_terms: list[float] = []
    slot_correct: dict[int, int] = {}
    slot_total: dict[int, int] = {}
    slot_chosen: dict[int, int] = {}
    kind_total: dict[str, int] = {}
    kind_flip: dict[str, int] = {}
    max_slots = 0

    for i, item in enumerate(items):
        modal_contents = []  # one modal content per presentation
        pres_kinds = []  # the kind of each presentation, aligned with modal_contents
        correct = 0
        graded = 0
        for j, pres in enumerate(per_item_perts[i]):
            contents = []
            kind = getattr(pres, "kind", "perturbation")
            for s in range(rerun):
                o = outcomes[(i, j, s)]
                if o.correct is not None:
                    graded += 1
                    correct += int(o.correct)
                if o.content is not None:
                    contents.append(o.content)
                if o.slot is not None:
                    slot_chosen[o.slot] = slot_chosen.get(o.slot, 0) + 1
                if o.correct_slot is not None:
                    slot_total[o.correct_slot] = slot_total.get(o.correct_slot, 0) + 1
                    if o.slot == o.correct_slot:
                        slot_correct[o.correct_slot] = (
                            slot_correct.get(o.correct_slot, 0) + 1
                        )
                if o.n_slots:
                    max_slots = max(max_slots, o.n_slots)
            mc, frac = modal(contents)
            obs_terms.append(1 - frac if contents else 0.0)  # instability across reruns
            modal_contents.append(mc)
            pres_kinds.append(kind)
        if graded:
            per_item_acc.append(correct / graded)
        # this item's dominant answer, and how often each perturbation kind knocks it off
        reference, content_frac = modal(modal_contents)
        per_item_interv.append(
            1 - content_frac
        )  # 0 = same content across all perturbations
        if reference is not None:
            for kind, mc in zip(pres_kinds, modal_contents):
                kind_total[kind] = kind_total.get(kind, 0) + 1
                if mc != reference:
                    kind_flip[kind] = kind_flip.get(kind, 0) + 1

    accuracy = sum(per_item_acc) / len(per_item_acc) if per_item_acc else None
    acc_ci = (
        tuple(
            bootstrap_ci(
                per_item_acc, lambda s: sum(s) / len(s), n_boot=4000, seed=boot_seed
            )
        )
        if per_item_acc
        else None
    )
    interventional = sum(per_item_interv) / len(per_item_interv)
    observational = sum(obs_terms) / len(obs_terms) if obs_terms else 0.0
    squish_by_kind = {
        k: kind_flip.get(k, 0) / kind_total[k] for k in kind_total if kind_total[k]
    }

    acc_by_pos = pos_swing = chosen_dist = None
    if slot_total and max_slots:
        acc_by_pos = tuple(
            (slot_correct.get(p, 0) / slot_total[p]) if slot_total.get(p) else 0.0
            for p in range(max_slots)
        )
        populated = [acc_by_pos[p] for p in range(max_slots) if slot_total.get(p)]
        pos_swing = (max(populated) - min(populated)) if populated else 0.0
        total_chosen = sum(slot_chosen.values()) or 1
        chosen_dist = tuple(
            slot_chosen.get(p, 0) / total_chosen for p in range(max_slots)
        )

    return ModelReport(
        model=model or provider.config().get("model", "model"),
        benchmark=benchmark,
        task=task.name,
        n_items=len(items),
        n_rerun=rerun,
        accuracy=accuracy,
        accuracy_ci=acc_ci,
        interventional_squish=interventional,
        observational_squish=observational,
        accuracy_by_position=acc_by_pos,
        position_swing=pos_swing,
        chosen_distribution=chosen_dist,
        squish_by_kind=squish_by_kind,
        provenance={
            "config": provider.config(),
            "task": task.name,
            "boot_seed": boot_seed,
            "seed_offset": seed_offset,
        },
    )


def compare(providers: dict[str, Provider], items: list, **kwargs) -> list[ModelReport]:
    """Evaluate several named models on the same items. `model` label = the dict key."""
    return [evaluate(p, items, model=name, **kwargs) for name, p in providers.items()]


def compare_markdown(reports: list[ModelReport]) -> str:
    """A comparison table for a set of reports — the shape a catalog page wants."""
    header = (
        "| model | accuracy | 95% CI | interv. squish | position swing |\n"
        "|---|---|---|---|---|\n"
    )

    def cell(r: ModelReport) -> str:
        acc = f"{r.accuracy:.1%}" if r.accuracy is not None else "—"
        ci = f"{r.accuracy_ci[0]:.1%}–{r.accuracy_ci[1]:.1%}" if r.accuracy_ci else "—"
        swing = f"{r.position_swing:.2f}" if r.position_swing is not None else "—"
        return (
            f"| {r.model} | {acc} | {ci} | {r.interventional_squish:.1%} | {swing} |\n"
        )

    return header + "".join(cell(r) for r in reports)


def score_stability(
    provider: Provider,
    items: list,
    *,
    n_runs: int = 5,
    scoring: str = "gen",
    benchmark: str = "benchmark",
    **kwargs,
) -> dict:
    """Pillar 1: run the WHOLE benchmark ``n_runs`` times and report the run-to-run spread of
    the aggregate score — the variance a non-deterministic model injects into "the number".

    Each run uses a fresh seed offset, so a sampling model (``scoring="gen"``) genuinely
    re-samples and the score moves; a deterministic one (``scoring="ll"``) reproduces exactly,
    which is itself the honest answer (its run-to-run variance is zero). Report a score as
    ``mean ± spread`` across runs, never as a lone point. Note the aggregate is often far more
    stable than individual answers — the wobble lives per-item (observational squish), which is
    why both numbers matter.
    """
    runs = []
    for r in range(n_runs):
        rep = evaluate(
            provider,
            items,
            scoring=scoring,
            benchmark=benchmark,
            seed_offset=r + 1,
            **kwargs,
        )
        runs.append(rep.accuracy)
    graded = [a for a in runs if a is not None]
    mean = sum(graded) / len(graded) if graded else None
    if len(graded) > 1:
        var = sum((a - mean) ** 2 for a in graded) / len(graded)
        std = var**0.5
    else:
        std = 0.0
    return {
        "runs": runs,
        "mean": mean,
        "std": std,
        "spread": (max(graded) - min(graded)) if graded else None,
        "n_runs": n_runs,
        "scoring": scoring,
    }
