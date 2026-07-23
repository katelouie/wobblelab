"""The squish score: how much a model's answer moves under what shouldn't matter.

Combines the two measurement channels into one headline, respecting their asymmetry
(lab-journal D-007):

  - interventional squish (phrasing-fragility, ``1 - margin``) is UNCONDITIONAL: a
    meaning-preserving rephrase should never change an answer, decidable or not.
  - observational squish (rerun ``dispersion``) is CONDITIONAL: a defect only when
    there is a fact to be stable about. On an undecidable prompt, rerun variation is
    appropriate uncertainty, so it is gated OFF. (Behaviorally you cannot tell the two
    apart from outputs alone -- F-013 -- so the decidability label is required, not a
    shortcut.)

Channels are combined worst-case (``max``), consistent with the worst-case margin
(D-004), and the full decomposition is retained so "bad on both axes" stays visible.

Scoring is a pure post-processing function of a measurement, so a battery can be
re-scored under a revised rule without re-running the model.
"""

from __future__ import annotations

import statistics


def squish_score(
    dispersion: float,
    margin_max: float,
    answer: str | None,
    disp_ci: tuple[float, float] | None = None,
    margin_ci: tuple[float, float] | None = None,
) -> dict:
    """Per-prompt squish score plus its decomposition.

    Args:
        dispersion: rerun dispersion in [0, 0.5].
        margin_max: worst-case interventional margin in [0, 1].
        answer: "yes"/"no" (decidable -> dispersion counts) or None (undecidable ->
            dispersion gated off).
        disp_ci, margin_ci: optional (lo, hi) intervals; if both given, a score CI is
            returned (a conservative interval: the max of the two channels' intervals).
    """
    interventional = 1.0 - margin_max
    decidable = answer in ("yes", "no")
    observational = 2.0 * dispersion if decidable else 0.0  # 2x maps [0,0.5] -> [0,1]
    score = max(interventional, observational)

    if interventional > observational:
        driver = "interventional"
    elif observational > interventional:
        driver = "observational"
    else:
        driver = "both"

    out = {
        "squish": round(score, 3),
        "interventional": round(interventional, 3),
        "observational": round(observational, 3) if decidable else None,
        "decidable": decidable,
        "driver": driver,
    }
    if disp_ci is not None and margin_ci is not None:
        interv_lo, interv_hi = 1 - margin_ci[1], 1 - margin_ci[0]
        obs_lo, obs_hi = (2 * disp_ci[0], 2 * disp_ci[1]) if decidable else (0.0, 0.0)
        out["squish_ci"] = (
            round(max(interv_lo, obs_lo), 3),
            round(max(interv_hi, obs_hi), 3),
        )
    return out


def squish_factor(**signals: float) -> dict:
    """A single 0-1 unreliability headline: the WORST of the measured squish signals.

    "Squish factor" answers one question for a report card: how much can this model's answer
    move under something that should not move it? We combine **worst-case (max)**, consistent
    with the plane's worst-case margin (D-004) and the score's worst-case channel-combine
    (D-007) -- a model is only as reliable as its most fragile axis, and averaging would let a
    catastrophic position bias hide behind four calm signals.

    Each keyword signal must already be a 0-1 fraction (a flip rate, an accuracy swing, a
    normalized spread). Returns the factor, the *driving* signal, a band label, and the full
    decomposition, so "why is it high" is always one glance away.

    >>> squish_factor(reorder=0.365, position_swing=0.667, run_spread=0.198)["factor"]
    0.667
    """
    clamped = {k: max(0.0, min(1.0, v)) for k, v in signals.items()}
    factor = max(clamped.values()) if clamped else 0.0
    driver = max(clamped, key=clamped.__getitem__) if clamped else None
    band = "high" if factor >= 0.5 else "moderate" if factor >= 0.2 else "low"
    return {
        "factor": round(factor, 3),
        "driver": driver,
        "band": band,
        "components": {k: round(v, 3) for k, v in clamped.items()},
    }


def model_squish(scored_rows: list[dict]) -> dict:
    """Model-level headline: central tendency across the battery + worst offenders.

    Each row must carry at least "name", "squish", "driver".
    """
    scores = [r["squish"] for r in scored_rows]
    worst = sorted(scored_rows, key=lambda r: r["squish"], reverse=True)
    return {
        "mean_squish": round(statistics.fmean(scores), 3),
        "median_squish": round(statistics.median(scores), 3),
        "max_squish": round(max(scores), 3),
        "n_prompts": len(scores),
        "worst_offenders": [
            {"name": r["name"], "squish": r["squish"], "driver": r["driver"]}
            for r in worst[:3]
        ],
    }
