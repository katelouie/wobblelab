"""Tests for the reliability-card data model, builders, and renderers."""

import json
import xml.dom.minidom as minidom

import pytest

from wobblelab import (
    JSONRenderer,
    MockProvider,
    SVGCardRenderer,
    benchmark_card,
    evaluate,
    production_card,
    score_stability,
)
from wobblelab.benchmark import MCItem

ITEMS = [
    MCItem(
        id=f"q{i}",
        question=f"Q{i}?",
        options=("CORRECT", "w1", "w2", "w3"),
        answer_idx=i % 4,
    )
    for i in range(8)
]


def _biased_report():
    # "always A" -> real position bias, so the card has non-trivial signals
    from wobblelab import always_position

    prov = MockProvider(always_position(0))
    return evaluate(prov, ITEMS, scoring="ll", benchmark="mmlu:toy"), score_stability(
        prov, ITEMS, n_runs=3, scoring="ll"
    )


def test_benchmark_card_structure_and_verdict():
    rep, stab = _biased_report()
    card = benchmark_card(rep, stab)
    assert card.lens == "benchmark"
    kinds = [p.kind for p in card.panels]
    assert kinds == ["accuracy_ci", "run_variance", "position_bias", "flip_rate"]
    # the position-bias panel carries a real swing and a plain-English verdict
    pos = card.panel("position_bias")
    assert pos.data["swing"] > 0.5 and "position preference" in pos.verdict
    # verdict badge tracks the wobble band
    assert card.verdict in ("BENCHMARK UNRELIABLE", "USE WITH CAUTION", "RELIABLE")
    assert card.wobble_factor["factor"] == pytest.approx(
        pos.data["swing"]
    )  # worst-case driver


def test_card_to_dict_is_json_serializable():
    rep, stab = _biased_report()
    card = benchmark_card(rep, stab)
    d = card.to_dict()
    dumped = json.dumps(d)  # must not raise
    assert json.loads(dumped)["lens"] == "benchmark"
    assert {"subject", "verdict", "wobble_factor", "panels", "provenance"} <= set(d)


def test_svg_renderer_produces_valid_svg():
    rep, stab = _biased_report()
    svg = SVGCardRenderer().render(benchmark_card(rep, stab))
    minidom.parseString(svg)  # well-formed XML or raises
    assert svg.startswith("<svg") and "WOBBLE FACTOR" in svg


def test_json_renderer_roundtrips():
    rep, stab = _biased_report()
    out = JSONRenderer().render(benchmark_card(rep, stab))
    assert json.loads(out)["wobble_factor"]["driver"] == "position_swing"


def test_production_card_from_measurements():
    card = production_card(
        {"model": "m", "context": "production", "n_items": 8},
        perturbation=[{"name": "Lexical", "value": 0.2, "example": "zero -> 0"}],
        cross_lingual=[
            {
                "name": "pluto",
                "a": 0.05,
                "b": 0.38,
                "delta": 0.33,
                "langs": ["EN", "ZH"],
            },
            {
                "name": "water",
                "a": 0.94,
                "b": 0.97,
                "delta": 0.03,
                "langs": ["EN", "ZH"],
            },
        ],
        config_ab={
            "sensitive": ["pluto"],
            "n_agree": 7,
            "n_total": 8,
            "worst_shift": 0.02,
        },
    )
    assert card.lens == "production"
    assert card.panel("cross_lingual").data["n_over"] == 1  # only pluto > 0.30
    assert "1 of 2 probes" in card.panel("cross_lingual").verdict
    minidom.parseString(SVGCardRenderer().render(card))  # renders both lenses


def test_renderer_falls_back_for_unknown_panel_kind():
    from wobblelab import Panel, ReliabilityCard

    card = ReliabilityCard(
        subject={"model": "m"},
        lens="benchmark",
        verdict="RELIABLE",
        severity="good",
        wobble_factor={"factor": 0.0, "band": "low", "driver": None},
        panels=[Panel(kind="a_brand_new_kind", title="Novel", data={"x": 1, "y": 0.5})],
    )
    svg = SVGCardRenderer().render(card)  # must not crash on an unknown kind
    minidom.parseString(svg)
    assert "Novel".upper() in svg
