"""Builders: turn raw harness output into a structured ReliabilityCard.

`benchmark_card(report, stability)` is the fully-wired one -- hand it a `ModelReport` (and
optionally a `score_stability` result) and it computes the panels, the plain-English
verdicts, the per-panel severity, and the headline squish factor. `production_card(...)`
takes the production-lens measurements (perturbation sensitivity, cross-lingual, squish
plane, config) since those come from separate probes, and assembles them the same way.

All the interpretation ("BENCHMARK UNRELIABLE", "position preference, not knowledge") lives
here, once, so a renderer never has to interpret data and every format tells the same story.
"""

from __future__ import annotations

from squishlab.benchmark import LETTERS
from squishlab.cards.model import Panel, ReliabilityCard
from squishlab.squish import squish_factor

_BAND_SEVERITY = {"high": "bad", "moderate": "caution", "low": "good"}


def _sev(x: float, caution: float, bad: float) -> str:
    return "bad" if x >= bad else "caution" if x >= caution else "good"


def _verdict_badge(lens: str, band: str) -> str:
    if lens == "benchmark":
        return {
            "high": "BENCHMARK UNRELIABLE",
            "moderate": "USE WITH CAUTION",
            "low": "RELIABLE",
        }[band]
    return {
        "high": "DO NOT ROUTE",
        "moderate": "ROUTE WITH CAUTION",
        "low": "ROUTE OK",
    }[band]


def benchmark_card(report, stability: dict | None = None) -> ReliabilityCard:
    """Assemble the Benchmark Reliability card ("can I trust this evaluation number?")."""
    panels: list[Panel] = []
    signals: dict[str, float] = {}

    # Accuracy + CI
    ci = list(report.accuracy_ci) if report.accuracy_ci else None
    width = (ci[1] - ci[0]) if ci else 0.0
    panels.append(
        Panel(
            kind="accuracy_ci",
            title="Accuracy — is the number trustworthy?",
            data={"value": report.accuracy, "ci": ci, "chance": _chance(report)},
            verdict=(
                f"Reported as {_pct(report.accuracy)}, but the honest range is "
                f"{_pct(ci[0])}–{_pct(ci[1])} — a {round(width * 100)}-point interval."
                if ci
                else None
            ),
            severity=_sev(width, 0.08, 0.15),
        )
    )

    # Run-to-run variance
    if stability and stability.get("runs"):
        spread = stability["spread"] or 0.0
        signals["run_spread"] = spread
        panels.append(
            Panel(
                kind="run_variance",
                title="Run-to-run variance — same model, same questions, different seeds",
                data={
                    "runs": stability["runs"],
                    "mean": stability["mean"],
                    "spread": spread,
                },
                verdict=(
                    "A single run is not a measurement. Any two runs of this model on this "
                    f"benchmark could differ by {round(spread * 100)} points."
                ),
                severity=_sev(spread, 0.05, 0.15),
                squish_signal=spread,
            )
        )

    # Position bias
    if report.accuracy_by_position:
        swing = report.position_swing or 0.0
        signals["position_swing"] = swing
        by_slot = {LETTERS[i]: a for i, a in enumerate(report.accuracy_by_position)}
        chosen = {LETTERS[i]: c for i, c in enumerate(report.chosen_distribution or ())}
        best = max(by_slot, key=by_slot.__getitem__)
        worst = min(by_slot, key=by_slot.__getitem__)
        panels.append(
            Panel(
                kind="position_bias",
                title="Position bias — accuracy by correct-answer slot",
                data={
                    "by_slot": by_slot,
                    "chosen": chosen,
                    "swing": swing,
                    "chance": _chance(report),
                },
                verdict=(
                    f'Scores {_pct(by_slot[best])} when the answer is "{best}" and '
                    f'{_pct(by_slot[worst])} when it is "{worst}". Much of the accuracy is '
                    "position preference, not knowledge."
                ),
                severity=_sev(swing, 0.15, 0.4),
                squish_signal=swing,
            )
        )

    # Option-order sensitivity (reorder flip rate)
    reorder = report.squish_by_kind.get("reorder", report.interventional_squish)
    signals["reorder"] = reorder
    panels.append(
        Panel(
            kind="flip_rate",
            title="Option-order sensitivity — shuffle answers, watch responses flip",
            data={
                "value": reorder,
                "caption": "of answers flip when options are reordered",
            },
            severity=_sev(reorder, 0.15, 0.3),
            squish_signal=reorder,
        )
    )

    factor = squish_factor(**signals)
    return ReliabilityCard(
        subject=_subject(report),
        lens="benchmark",
        verdict=_verdict_badge("benchmark", factor["band"]),
        severity=_BAND_SEVERITY[factor["band"]],
        squish_factor=factor,
        panels=panels,
        provenance=_provenance(report),
    )


def production_card(
    subject: dict,
    *,
    perturbation: list[dict] | None = None,
    cross_lingual: list[dict] | None = None,
    squish_plane: list[dict] | None = None,
    config_ab: dict | None = None,
) -> ReliabilityCard:
    """Assemble the Production Robustness card ("will it behave for real users?").

    Inputs are the production-lens measurements, already computed:
      - perturbation: [{name, value, example}, ...]  (value = avg shift / flip rate, 0-1)
      - cross_lingual: [{name, a, b, delta}, ...]     (per-probe language drift)
      - squish_plane: [{name, dispersion, margin, quadrant}, ...]
      - config_ab: {sensitive: [names], n_agree, n_total, arms: [a, b]}
    """
    panels: list[Panel] = []
    signals: dict[str, float] = {}

    if perturbation:
        worst = max(p["value"] for p in perturbation)
        signals["perturbation"] = worst
        panels.append(
            Panel(
                kind="perturbation",
                title="Prompt perturbation — same question, different wording",
                data={"kinds": perturbation, "unit": "avg shift in p(yes)"},
                severity=_sev(worst, 0.15, 0.3),
                squish_signal=worst,
            )
        )

    if cross_lingual:
        threshold = 0.30
        over = [p for p in cross_lingual if p["delta"] > threshold]
        worst = max((p["delta"] for p in cross_lingual), default=0.0)
        signals["cross_lingual"] = worst
        langs = (
            cross_lingual[0].get("langs", ["A", "B"]) if cross_lingual else ["A", "B"]
        )
        panels.append(
            Panel(
                kind="cross_lingual",
                title="Cross-lingual stability — same question in another language",
                data={
                    "probes": cross_lingual,
                    "threshold": threshold,
                    "n_over": len(over),
                    "n_total": len(cross_lingual),
                    "langs": langs,
                },
                verdict=(
                    f"{len(over)} of {len(cross_lingual)} probes show >{round(threshold * 100)}% "
                    "drift between languages. Multilingual deployment needs per-language validation."
                ),
                severity=_sev(worst, 0.15, 0.3),
                squish_signal=worst,
            )
        )

    if squish_plane:
        knife = [p["name"] for p in squish_plane if p.get("quadrant") == "KNIFE-EDGE"]
        panels.append(
            Panel(
                kind="squish_plane",
                title="Squish plane — reliability shape per question",
                data={"points": squish_plane},
                verdict=(
                    f"Knife-edge probes (solid on rerun, flip on reword): {', '.join(knife)}."
                    if knife
                    else None
                ),
                severity="caution" if knife else "good",
            )
        )

    if config_ab:
        n_sens = len(config_ab.get("sensitive", []))
        signals["config"] = config_ab.get("worst_shift", 0.0)
        panels.append(
            Panel(
                kind="config_ab",
                title="Config sensitivity — do sampling hyperparameters change the answer?",
                data=config_ab,
                verdict=(
                    f"{config_ab['n_agree']} of {config_ab['n_total']} probes agree across configs."
                    + (
                        f" Only {', '.join(config_ab['sensitive'])} shifts confidently."
                        if n_sens
                        else ""
                    )
                ),
                severity=_sev(n_sens / max(config_ab.get("n_total", 1), 1), 0.15, 0.4),
            )
        )

    factor = squish_factor(**signals)
    return ReliabilityCard(
        subject=subject,
        lens="production",
        verdict=_verdict_badge("production", factor["band"]),
        severity=_BAND_SEVERITY[factor["band"]],
        squish_factor=factor,
        panels=panels,
        provenance=subject.get(
            "provenance", {"recorded": ["model", "config", "seeds"], "complete": True}
        ),
    )


# --- small helpers ---------------------------------------------------------


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _chance(report) -> float:
    k = len(report.accuracy_by_position) if report.accuracy_by_position else 4
    return 1.0 / k if k else 0.25


def _subject(report) -> dict:
    cfg = report.provenance.get("config", {})
    return {
        "model": report.model,
        "context": report.benchmark,
        "n_items": report.n_items,
        "n_rerun": report.n_rerun,
        "quant": cfg.get("options", {}).get("quant") or report.provenance.get("quant"),
    }


def _provenance(report) -> dict:
    return {
        "recorded": ["model", "quantization", "config", "harness", "seed"],
        "complete": True,
        "detail": report.provenance,
    }
