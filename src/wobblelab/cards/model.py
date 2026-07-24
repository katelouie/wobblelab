"""The reliability card data model: structured, self-describing, presentation-free.

A `ReliabilityCard` is the complete reliability picture for one model in one context (a
"lens"), captured as data, not pixels. It is the parseable API: `to_dict()` gives you every
number the harness produced, and a renderer (SVG, JSON, HTML, PDF, ...) turns it into an
artifact. Redesign the card, add a format, or rearrange panels without touching the data.

A card is a list of typed `Panel`s. Each panel is one self-describing unit -- its `kind`
tells a renderer what it is, its `data` holds the kind-specific numbers, its `verdict` is the
plain-English takeaway, its `severity` drives colour, and its `wobble_signal` is the 0-1
contribution it makes to the card's headline wobble factor. New panel kinds are just new
`kind` strings with a documented `data` schema; a renderer that doesn't know a kind can fall
back to a generic view rather than break.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEVERITIES = ("good", "caution", "bad")


@dataclass(frozen=True)
class Panel:
    """One structured unit of a reliability card.

    Documented `data` schemas by `kind` (renderers read these; all values JSON-safe):
      - "accuracy_ci":       {value, ci: [lo, hi], chance}
      - "run_variance":      {runs: [...], mean, spread}
      - "position_bias":     {by_slot: {A: acc, ...}, chosen: {A: frac, ...}, swing, chance}
      - "flip_rate":         {value, caption}
      - "perturbation":      {kinds: [{name, value, example}, ...], unit}
      - "cross_lingual":     {probes: [{name, a, b, delta}, ...], threshold, n_over, n_total, langs: [a, b]}
      - "wobble_plane":      {points: [{name, dispersion, margin, quadrant}, ...]}
      - "config_ab":         {sensitive: [names], n_agree, n_total, arms: [a, b]}
      - "provenance":        {recorded: [fields], complete: bool}
    """

    kind: str
    title: str
    data: dict = field(default_factory=dict)
    verdict: str | None = None
    severity: str | None = None  # "good" | "caution" | "bad"
    wobble_signal: float | None = None  # 0-1 contribution to the card's wobble factor

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "data": self.data,
            "verdict": self.verdict,
            "severity": self.severity,
            "wobble_signal": self.wobble_signal,
        }


@dataclass(frozen=True)
class ReliabilityCard:
    """The complete reliability picture for one model in one context. `to_dict()` is the API."""

    subject: dict  # {model, quant, context, n_items, n_rerun, ...}
    lens: str  # "benchmark" | "production"
    verdict: str  # headline badge text, e.g. "BENCHMARK UNRELIABLE"
    severity: str  # overall "good" | "caution" | "bad"
    wobble_factor: dict  # from wobble_factor(): {factor, driver, band, components}
    panels: list[Panel]
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "lens": self.lens,
            "verdict": self.verdict,
            "severity": self.severity,
            "wobble_factor": self.wobble_factor,
            "panels": [p.to_dict() for p in self.panels],
            "provenance": self.provenance,
        }

    def panel(self, kind: str) -> Panel | None:
        """First panel of a given kind, or None -- so renderers/consumers can pick by kind."""
        return next((p for p in self.panels if p.kind == kind), None)
