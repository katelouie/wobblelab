"""Reliability cards: a structured card data model + pluggable renderers.

from wobblelab import evaluate, score_stability
from wobblelab.cards import benchmark_card, SVGCardRenderer, JSONRenderer

report = evaluate(provider, items)
card = benchmark_card(report, score_stability(provider, items))
open("card.svg", "w").write(SVGCardRenderer().render(card))   # visual artifact
data = card.to_dict()                                         # parseable API
"""

from wobblelab.cards.build import benchmark_card, production_card
from wobblelab.cards.model import Panel, ReliabilityCard
from wobblelab.cards.render import (
    CardRenderer,
    JSONRenderer,
    SVGCardRenderer,
)

__all__ = [
    "Panel",
    "ReliabilityCard",
    "benchmark_card",
    "production_card",
    "CardRenderer",
    "JSONRenderer",
    "SVGCardRenderer",
]
