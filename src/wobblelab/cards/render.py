"""Renderers: turn a ReliabilityCard into an artifact. Pluggable by format.

A renderer only reads the structured card -- it never measures or interprets. `JSONRenderer`
is the parseable output (free). `SVGCardRenderer` is the visual POC: it flow-lays-out the
panels top to bottom, one fragment method per panel `kind`, with a generic fallback for kinds
it doesn't know. Reorder panels in the builder, restyle a panel by editing its one method,
change the palette in one dict, or add an HTMLRenderer/PDFRenderer beside these -- the data
model and the harness never move.
"""

from __future__ import annotations

import html
import json
from typing import Protocol, runtime_checkable

from wobblelab.cards.model import Panel, ReliabilityCard


@runtime_checkable
class CardRenderer(Protocol):
    """Anything that turns a ReliabilityCard into a serialized artifact."""

    format: str

    def render(self, card: ReliabilityCard) -> str: ...


class JSONRenderer:
    """The parseable output: the full card as JSON. This is the machine-readable API."""

    format = "json"

    def render(self, card: ReliabilityCard) -> str:
        return json.dumps(card.to_dict(), indent=2)


# --- palette (one place to restyle everything) -----------------------------

TOKENS = {
    "bg": "#ffffff",
    "surface": "#f7f6f3",
    "hair": "#e5e2db",
    "ink": "#1b1d23",
    "muted": "#6b6f7a",
    "faint": "#9b9893",
    "good": "#3B6D11",
    "good_bg": "#EAF3DE",
    "good_fill": "#97C459",
    "caution": "#854F0B",
    "caution_bg": "#FAEEDA",
    "caution_fill": "#FAC775",
    "bad": "#A32D2D",
    "bad_bg": "#FCEBEB",
    "bad_fill": "#F09595",
    "accent": "#e07a3a",
    "blue": "#6b8fbf",
}
SANS = "-apple-system, 'Segoe UI', Helvetica, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, monospace"


class SVGCardRenderer:
    """Data-driven SVG. Flow-layout: each panel is a `_p_<kind>` fragment; unknown kinds fall
    back to `_p_generic`. The existing hand-authored cards are the visual reference."""

    format = "svg"
    HEADER_H = 88
    FOOTER_H = 40
    GAP = 20

    def __init__(self, width: int = 720, tokens: dict | None = None) -> None:
        self.w = width
        self.t = {**TOKENS, **(tokens or {})}

    # -- public -------------------------------------------------------------

    def render(self, card: ReliabilityCard) -> str:
        inner = self.w - 48  # 24px margins
        parts: list[str] = []
        y = self.HEADER_H + self.GAP
        for panel in card.panels:
            method = getattr(self, f"_p_{panel.kind}", self._p_generic)
            frag, h = method(panel, 24, y, inner)
            parts.append(frag)
            if panel.verdict and panel.kind not in ("flip_rate",):
                vfrag, vh = self._verdict_box(panel, 24, y + h + 8, inner)
                parts.append(vfrag)
                h += vh + 8
            parts.append(self._line(24, y + h + self.GAP / 2, self.w - 24))
            y += h + self.GAP
        total = y + self.FOOTER_H
        head = self._header(card)
        foot = self._footer(card, total - self.FOOTER_H + 6)
        return (
            f'<svg viewBox="0 0 {self.w} {total:.0f}" xmlns="http://www.w3.org/2000/svg" '
            f'role="img" style="width:100%;height:auto;font-family:{SANS}">'
            f"<title>WobbleLab {html.escape(card.lens)} reliability card</title>"
            f'<rect x="0" y="0" width="{self.w}" height="{total:.0f}" rx="8" '
            f'fill="{self.t["bg"]}" stroke="{self.t["hair"]}" stroke-width="0.5"/>'
            f"{head}{''.join(parts)}{foot}</svg>"
        )

    # -- header / footer ----------------------------------------------------

    def _header(self, card: ReliabilityCard) -> str:
        sev = card.severity
        badge_w = 12 + len(card.verdict) * 7.4
        subj = card.subject
        ctx = " · ".join(str(v) for v in (subj.get("context"), _items_line(subj)) if v)
        gx = self.w - 160
        sf = card.wobble_factor
        fill_w = round(136 * float(sf.get("factor", 0)))
        return (
            self._rect(24, 16, badge_w, 40, self.t[f"{sev}_bg"], rx=20)
            + self._text(
                24 + badge_w / 2,
                41,
                card.verdict,
                12,
                self.t[sev],
                "600",
                MONO,
                "middle",
            )
            + self._text(
                40 + badge_w,
                34,
                str(subj.get("model", "model")),
                15,
                self.t["ink"],
                "600",
            )
            + self._text(40 + badge_w, 52, ctx, 11, self.t["muted"])
            + self._text(gx, 28, "WOBBLE FACTOR", 10, self.t["muted"], "500")
            + self._rect(gx, 36, 136, 8, self.t["hair"], rx=4)
            + self._rect(gx, 36, fill_w, 8, self.t["accent"], rx=4)
            + self._text(
                gx, 58, f"{sf.get('factor', 0):.2f}", 11, self.t["accent"], "600", MONO
            )
            + self._text(
                gx + 30,
                58,
                f"/ 1.0 — {sf.get('band', '')} ({sf.get('driver', '')})",
                10,
                self.t["muted"],
            )
            + self._line(24, 76, self.w - 24)
        )

    def _footer(self, card: ReliabilityCard, y: float) -> str:
        rec = card.provenance.get("recorded", [])
        ok = card.provenance.get("complete", False)
        return (
            self._text(24, y + 14, "PROVENANCE", 10, self.t["muted"], "500")
            + self._rect(
                100, y + 2, 40, 18, self.t["good_bg"] if ok else self.t["bad_bg"], rx=9
            )
            + self._text(
                120,
                y + 15,
                "full" if ok else "partial",
                10,
                self.t["good"] if ok else self.t["bad"],
                "500",
                MONO,
                "middle",
            )
            + self._text(
                152,
                y + 15,
                " + ".join(rec) + " — recorded, reproducible",
                10,
                self.t["muted"],
            )
        )

    # -- panels -------------------------------------------------------------

    def _p_accuracy_ci(self, p: Panel, x, y, w) -> tuple[str, float]:
        d = p.data
        val, ci = d.get("value"), d.get("ci")
        s = self._label(p.title, x, y)
        s += self._text(
            x, y + 52, _pct(val), 44, self.t["ink"], "400", "ui-serif, Georgia, serif"
        )
        tx, tw = x + 116, w - 116
        s += self._rect(tx, y + 34, tw, 24, self.t["surface"], rx=4)
        if ci:
            lo, hi = ci
            s += self._rect(
                tx + tw * lo, y + 36, tw * (hi - lo), 20, self.t["caution_bg"], rx=3
            )
            s += f'<line x1="{tx + tw * val:.1f}" y1="{y + 34}" x2="{tx + tw * val:.1f}" y2="{y + 58}" stroke="{self.t["ink"]}" stroke-width="2"/>'
            s += self._text(
                tx,
                y + 74,
                f"{_pct(lo)} – {_pct(hi)}   ({round((hi - lo) * 100)}-pt honest range)",
                10,
                self.t["caution"],
                "500",
                MONO,
            )
        for frac in (0, 0.25, 0.5, 0.75, 1.0):
            s += self._text(
                tx + tw * frac,
                y + 90,
                f"{int(frac * 100)}%",
                9,
                self.t["faint"],
                "400",
                MONO,
                "middle",
            )
        return s, 96

    def _p_run_variance(self, p: Panel, x, y, w) -> tuple[str, float]:
        d = p.data
        runs = d.get("runs", [])
        s = self._label(p.title, x, y)
        s += self._rect(x, y + 14, w, 44, self.t["surface"], rx=4)
        lo, hi = (min(runs), max(runs)) if runs else (0, 0)
        for v in runs:
            cx = x + w * v
            s += f'<circle cx="{cx:.1f}" cy="{y + 36}" r="6" fill="{self.t["bad"]}" opacity="0.8"/>'
            s += self._text(
                cx, y + 54, f"{v * 100:.0f}", 8, self.t["bad"], "400", MONO, "middle"
            )
        if runs:
            s += f'<line x1="{x + w * lo:.1f}" y1="{y + 22}" x2="{x + w * hi:.1f}" y2="{y + 22}" stroke="{self.t["bad"]}" stroke-width="1"/>'
            s += self._text(
                x + w * (lo + hi) / 2,
                y + 18,
                f"{round((hi - lo) * 100)}-point spread",
                9,
                self.t["bad"],
                "600",
                MONO,
                "middle",
            )
        return s, 62

    def _p_position_bias(self, p: Panel, x, y, w) -> tuple[str, float]:
        d = p.data
        by = d.get("by_slot", {})
        s = self._label(p.title, x, y)
        n = max(len(by), 1)
        bw = (w - (n - 1) * 12) / n
        chance = d.get("chance", 0.25)
        for i, (slot, acc) in enumerate(by.items()):
            bx = x + i * (bw + 12)
            sev = "good" if acc >= 0.6 else "bad" if acc < chance else "caution"
            s += self._rect(bx, y + 16, bw, 32, self.t["surface"], rx=4)
            s += self._rect(bx, y + 16, bw * acc, 32, self.t[f"{sev}_fill"], rx=4)
            s += self._text(
                bx + 6, y + 36, f"{acc * 100:.0f}%", 12, self.t[sev], "600", MONO
            )
            s += self._text(bx + bw - 12, y + 36, slot, 11, self.t[sev])
        return s, 54

    def _p_flip_rate(self, p: Panel, x, y, w) -> tuple[str, float]:
        d = p.data
        val = d.get("value", 0.0)
        s = self._label(p.title, x, y)
        s += self._text(
            x,
            y + 30,
            f"{val * 100:.0f}% {d.get('caption', '')}",
            12,
            self.t["ink"],
            "500",
        )
        filled = val * 10
        for i in range(10):
            sq = min(max(filled - i, 0), 1)
            fill = (
                self.t["bad"]
                if sq >= 0.9
                else self.t["bad_fill"]
                if sq > 0
                else self.t["hair"]
            )
            op = "0.8" if sq >= 0.9 else "0.6" if sq > 0 else "1"
            s += (
                self._rect(x + i * 28, y + 40, 24, 24, fill, rx=3)
                if op == "1"
                else f'<rect x="{x + i * 28}" y="{y + 40}" width="24" height="24" rx="3" fill="{fill}" opacity="{op}"/>'
            )
        s += self._text(
            x + 300, y + 57, "each square = 10% · filled = flipped", 10, self.t["muted"]
        )
        return s, 70

    def _p_perturbation(self, p: Panel, x, y, w) -> tuple[str, float]:
        d = p.data
        kinds = d.get("kinds", [])
        s = self._label(p.title, x, y)
        rh = 44
        for i, k in enumerate(kinds):
            ky = y + 16 + i * (rh + 8)
            val = k["value"]
            sev = "good" if val < 0.1 else "bad" if val >= 0.3 else "caution"
            s += self._rect(x, ky, w, rh, self.t["surface"], rx=6)
            s += self._text(x + 12, ky + 18, k["name"], 11, self.t["ink"], "500")
            if k.get("example"):
                s += self._text(x + 12, ky + 34, k["example"][:64], 9, self.t["muted"])
            s += self._rect(x + w - 96, ky + 8, 80, 28, self.t[f"{sev}_bg"], rx=14)
            s += self._text(
                x + w - 56,
                ky + 26,
                f"{val * 100:.0f}% {d.get('unit', '')[:5]}",
                11,
                self.t[sev],
                "600",
                MONO,
                "middle",
            )
        return s, 16 + len(kinds) * (rh + 8)

    def _p_cross_lingual(self, p: Panel, x, y, w) -> tuple[str, float]:
        d = p.data
        probes = d.get("probes", [])[:6]
        s = self._label(p.title, x, y)
        s += self._rect(x, y + 14, w, 16 + len(probes) * 22, self.t["surface"], rx=4)
        tx, tw = x + 150, w - 260
        for i, pr in enumerate(probes):
            ry = y + 32 + i * 22
            a, b = pr["a"], pr["b"]
            big = pr["delta"] > 0.30
            s += self._text(
                x + 8, ry + 4, pr["name"][:16], 9, self.t["ink"], "400", MONO
            )
            s += f'<line x1="{tx + tw * a:.1f}" y1="{ry}" x2="{tx + tw * b:.1f}" y2="{ry}" stroke="{self.t["faint"]}" stroke-width="1"/>'
            s += f'<circle cx="{tx + tw * a:.1f}" cy="{ry}" r="4" fill="{self.t["blue"]}"/>'
            s += f'<circle cx="{tx + tw * b:.1f}" cy="{ry}" r="4" fill="{self.t["bad"]}"/>'
            s += self._text(
                x + w - 60,
                ry + 4,
                f"Δ={pr['delta']:.2f}",
                9,
                self.t["bad"] if big else self.t["good"],
                "600",
                MONO,
            )
        return s, 22 + len(probes) * 22

    def _p_generic(self, p: Panel, x, y, w) -> tuple[str, float]:
        s = self._label(p.title, x, y)
        items = list(p.data.items())[:6]
        for i, (k, v) in enumerate(items):
            s += self._text(
                x,
                y + 30 + i * 16,
                f"{k}: {_short(v)}",
                10,
                self.t["muted"],
                "400",
                MONO,
            )
        return s, 30 + len(items) * 16

    # -- primitives ---------------------------------------------------------

    def _verdict_box(self, p: Panel, x, y, w) -> tuple[str, float]:
        sev = p.severity or "caution"
        return (
            self._rect(x, y, w, 28, self.t[f"{sev}_bg"], rx=4)
            + self._text(x + 12, y + 18, "Verdict: " + p.verdict, 11, self.t[sev]),
            28,
        )

    def _label(self, s: str, x, y) -> str:
        return self._text(
            x, y + 8, s.upper(), 11, self.t["muted"], "500", SANS, "start", ".06em"
        )

    def _text(
        self,
        x,
        y,
        s,
        size,
        color,
        weight="400",
        family=SANS,
        anchor="start",
        spacing=None,
    ):
        sp = f' letter-spacing="{spacing}"' if spacing else ""
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{sp}>{html.escape(str(s))}</text>'
        )

    def _rect(self, x, y, w, h, fill, rx=0) -> str:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>'

    def _line(self, x1, y, x2) -> str:
        return f'<line x1="{x1}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" stroke="{self.t["hair"]}" stroke-width="0.5"/>'


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _items_line(subj: dict) -> str:
    bits = []
    if subj.get("n_items"):
        bits.append(f"{subj['n_items']} items")
    if subj.get("quant"):
        bits.append(str(subj["quant"]))
    return " · ".join(bits)


def _short(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    s = str(v)
    return s if len(s) < 40 else s[:37] + "..."
