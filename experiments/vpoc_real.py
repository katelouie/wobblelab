"""squishlab vPOC v0.1 -- the squish plane, done honestly.

Changes from v0 (see docs/lab-journal.md, entry 2026-07-20):
  - continuous shift-based interventional margin (max / mean / max_ci), not majority-flip
  - Wilson CIs on dispersion, Newcombe CIs on every paraphrase shift
  - typed paraphrase taxonomy (casual / formal / hedge / reorder / lexical) -> hot-spots
  - both axes carry 95% CI error bars; a point whose CI straddles a threshold is UNRESOLVED
  - N bumped (60 canonical / 25 per paraphrase)

Sampling config is the controlled, portable set (squishlab.client.CONTROLLED),
NOT ollama's Modelfile defaults. Everything logged into results/vpoc_v01.json.

Run:  python experiments/vpoc_real.py
"""

from __future__ import annotations

import collections
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from squishlab import OllamaClient, confident_shift, newcombe_diff_ci, wilson_ci  # noqa: E402

MODEL = "qwen3.5:0.8b"
QUANT = "Q8_0"
N_RERUN = 60  # canonical reruns (seeds 0..N-1)
N_PARA = 25  # reruns per paraphrase
DISP_T = 0.15
MARGIN_T = 0.60
SUFFIX = " Reply with only the single word yes or no."

client = OllamaClient(MODEL)

# Typed paraphrase battery. The five perturbation types are constant across prompts
# (comparability); the hot-spot then tells us WHICH kind of change broke each prompt.
BATTERY = {
    "water_wet": {
        "q": "Is water wet?",
        "paras": [
            ("casual", "Water's wet, right?"),
            ("formal", "Is it the case that water is wet?"),
            ("hedge", "Would you say water is wet?"),
            ("reorder", "Wet: is that what water is?"),
            ("lexical", "Is H2O wet?"),
        ],
    },
    "hotdog_sandwich": {
        "q": "Is a hotdog a sandwich?",
        "paras": [
            ("casual", "Is a hot dog basically a sandwich?"),
            ("formal", "Does a hot dog satisfy the definition of a sandwich?"),
            ("hedge", "Would you consider a hot dog a sandwich?"),
            ("reorder", "A sandwich: is that what a hot dog is?"),
            ("lexical", "Is a frankfurter a sandwich?"),
        ],
    },
    "seven_prime": {
        "q": "Is 7 a prime number?",
        "paras": [
            ("casual", "Is 7 prime?"),
            ("formal", "Is the integer 7 a prime number?"),
            ("hedge", "Would you say 7 is prime?"),
            ("reorder", "Prime: is 7 one?"),
            ("lexical", "Is seven a prime number?"),
        ],
    },
    "zero_even": {
        "q": "Is zero an even number?",
        "paras": [
            ("casual", "Is 0 even?"),
            ("formal", "Is the integer zero an even number?"),
            ("hedge", "Would you say zero is even?"),
            ("reorder", "Even: is zero one of those?"),
            ("lexical", "Is 0 an even number?"),
        ],
    },
    "tomato_fruit": {
        "q": "Is a tomato a fruit?",
        "paras": [
            ("casual", "So is a tomato a fruit?"),
            ("formal", "Does a tomato meet the botanical definition of a fruit?"),
            ("hedge", "Would you call a tomato a fruit?"),
            ("reorder", "A fruit: is that what a tomato is?"),
            ("lexical", "Are tomatoes fruits?"),
        ],
    },
    "pluto_planet": {
        "q": "Is Pluto a planet?",
        "paras": [
            ("casual", "So is Pluto a planet?"),
            ("formal", "Is Pluto classified as a planet?"),
            ("hedge", "Would you say Pluto is a planet?"),
            ("reorder", "A planet: is Pluto one?"),
            ("lexical", "Is Pluto one of the planets?"),
        ],
    },
    "rain_tomorrow": {
        "q": "Will it rain tomorrow?",
        "paras": [
            ("casual", "Gonna rain tomorrow?"),
            ("formal", "Is precipitation expected tomorrow?"),
            ("hedge", "Do you think it'll rain tomorrow?"),
            ("reorder", "Tomorrow: will it rain?"),
            ("lexical", "Will it be rainy tomorrow?"),
        ],
    },
    "cereal_soup": {
        "q": "Is cereal a soup?",
        "paras": [
            ("casual", "Is cereal basically soup?"),
            ("formal", "Does cereal meet the definition of a soup?"),
            ("hedge", "Would you call cereal a soup?"),
            ("reorder", "A soup: is cereal one?"),
            ("lexical", "Is a bowl of cereal soup?"),
        ],
    },
}

YES = re.compile(r"\byes\b", re.I)
NO = re.compile(r"\bno\b", re.I)


def parse(text: str) -> str | None:
    y, n = bool(YES.search(text)), bool(NO.search(text))
    return "yes" if (y and not n) else "no" if (n and not y) else None


def run(prompt: str, n: int) -> tuple[int, int, int]:
    """Return (yes, decided, unparseable) over n seeded reruns (seeds 0..n-1)."""
    labels = [parse(client.ask(prompt + SUFFIX, seed=s)) for s in range(n)]
    c = collections.Counter(labels)
    return c["yes"], c["yes"] + c["no"], c[None]


def dispersion_with_ci(yes: int, decided: int):
    if decided == 0:
        return 0.5, (0.0, 0.5)
    p = yes / decided
    d = min(p, 1 - p)
    lo_p, hi_p = wilson_ci(yes, decided)
    d_hi = 0.5 if lo_p <= 0.5 <= hi_p else max(min(lo_p, 1 - lo_p), min(hi_p, 1 - hi_p))
    d_lo = min(min(lo_p, 1 - lo_p), min(hi_p, 1 - hi_p))
    return d, (d_lo, d_hi)


def quadrant(disp: float, margin: float) -> str:
    return {
        (False, True): "SOLID",
        (False, False): "KNIFE-EDGE",
        (True, True): "NOISY-BUT-SURE",
        (True, False): "COIN-FLIP",
    }[(disp >= DISP_T, margin >= MARGIN_T)]


def measure(name: str, spec: dict) -> dict:
    y0, n0, u0 = run(spec["q"], N_RERUN)
    p0 = y0 / n0 if n0 else 0.0
    disp, (disp_lo, disp_hi) = dispersion_with_ci(y0, n0)

    paras = []
    for ptype, text in spec["paras"]:
        yk, nk, uk = run(text, N_PARA)
        pk = yk / nk if nk else 0.0
        delta = pk - p0
        lo, hi = newcombe_diff_ci(yk, nk, y0, n0)  # CI on (p_para - p_canonical)
        paras.append(
            {
                "type": ptype,
                "text": text,
                "p_yes": round(pk, 3),
                "delta": round(delta, 3),
                "abs_delta": abs(delta),
                "ci": (round(lo, 3), round(hi, 3)),
                "confident_shift": round(confident_shift(yk, nk, y0, n0), 3),
                "unparseable": uk,
            }
        )

    worst = max(paras, key=lambda d: d["abs_delta"])
    margin_max = 1 - worst["abs_delta"]
    margin_mean = 1 - sum(d["abs_delta"] for d in paras) / len(paras)
    margin_max_ci = 1 - max(d["confident_shift"] for d in paras)

    wl, wh = worst["ci"]
    abs_lo, abs_hi = (max(0.0, wl), wh) if worst["delta"] >= 0 else (max(0.0, -wh), -wl)
    margin_lo, margin_hi = 1 - abs_hi, 1 - abs_lo

    disp_unresolved = disp_lo < DISP_T < disp_hi
    margin_unresolved = margin_lo < MARGIN_T < margin_hi
    return {
        "name": name,
        "question": spec["q"],
        "yes": y0,
        "decided": n0,
        "unparseable": u0,
        "p_yes": round(p0, 3),
        "dispersion": round(disp, 3),
        "disp_ci": (round(disp_lo, 3), round(disp_hi, 3)),
        "paras": paras,
        "margin_max": round(margin_max, 3),
        "margin_mean": round(margin_mean, 3),
        "margin_max_ci": round(margin_max_ci, 3),
        "margin_ci": (round(margin_lo, 3), round(margin_hi, 3)),
        "hotspot": {
            "type": worst["type"],
            "text": worst["text"],
            "delta": worst["delta"],
            "confident": worst["confident_shift"] > 0,
        },
        "quadrant": quadrant(disp, margin_max),
        "disp_unresolved": disp_unresolved,
        "margin_unresolved": margin_unresolved,
        "unresolved": disp_unresolved or margin_unresolved,
    }


def plot(rows, path: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=150)
    fig.patch.set_facecolor("#12141a")
    ax.set_facecolor("#181b23")
    quads = [
        (MARGIN_T, 0, 1 - MARGIN_T, DISP_T, "#2f4a2c", "SOLID", "#9fd18f"),
        (0, 0, MARGIN_T, DISP_T, "#5a3410", "KNIFE-EDGE", "#f0a63c"),
        (0, DISP_T, MARGIN_T, 0.5 - DISP_T, "#4a2230", "COIN-FLIP", "#e08a8a"),
        (
            MARGIN_T,
            DISP_T,
            1 - MARGIN_T,
            0.5 - DISP_T,
            "#24374a",
            "NOISY-BUT-SURE",
            "#8fb8d8",
        ),
    ]
    for x0, y0, w, h, c, lab, lc in quads:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, color=c, alpha=0.5, zorder=0))
        ax.text(
            x0 + w / 2,
            y0 + h - 0.012,
            lab,
            color=lc,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            alpha=0.9,
        )
    ax.axvline(MARGIN_T, color="#5b6270", lw=1)
    ax.axhline(DISP_T, color="#5b6270", lw=1)
    for r in rows:
        x, y = r["margin_max"], r["dispersion"]
        xe = [[max(0, x - r["margin_ci"][0])], [max(0, r["margin_ci"][1] - x)]]
        ye = [[max(0, y - r["disp_ci"][0])], [max(0, r["disp_ci"][1] - y)]]
        hollow = r["unresolved"]
        ax.errorbar(
            x,
            y,
            xerr=xe,
            yerr=ye,
            fmt="o",
            ms=9,
            mfc=("none" if hollow else "#f6e8ce"),
            mec="#f6e8ce",
            ecolor="#7d8393",
            elinewidth=1,
            capsize=2,
            zorder=5,
        )
        ax.annotate(
            r["name"],
            (x, y),
            textcoords="offset points",
            xytext=(9, 5),
            color="#fbf4e4",
            fontsize=9,
            zorder=6,
        )
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(-0.02, 0.53)
    ax.set_xlabel(
        "interventional margin  →  robust to paraphrase (worst-case)", color="#c9b79e"
    )
    ax.set_ylabel("observational dispersion  →  noisier on rerun", color="#c9b79e")
    ax.set_title(
        f"THE SQUISH PLANE v0.1  ·  {MODEL} ({QUANT})",
        color="#f6e8ce",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    ax.text(
        0.5,
        1.017,
        "point = estimate · bars = 95% CI · hollow = unresolved (CI straddles a line)",
        transform=ax.transAxes,
        ha="center",
        color="#8a7358",
        fontsize=8,
    )
    for s in ax.spines.values():
        s.set_color("#3a3f4b")
    ax.tick_params(colors="#8a8f9a")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")


def main() -> None:
    t0 = time.time()
    rows = []
    print(
        f"{'prompt':17} {'disp':>13} {'margin_max':>13} mean maxCI hot-spot            quadrant"
    )
    print("-" * 100)
    for name, spec in BATTERY.items():
        r = measure(name, spec)
        flag = "?" if r["unresolved"] else " "
        dl, dh = r["disp_ci"]
        ml, mh = r["margin_ci"]
        print(
            f"{name:17} {r['dispersion']:.2f}[{dl:.2f},{dh:.2f}] "
            f"{r['margin_max']:.2f}[{ml:.2f},{mh:.2f}] "
            f"{r['margin_mean']:.2f} {r['margin_max_ci']:.2f} "
            f"{r['hotspot']['type']:8}({'sig' if r['hotspot']['confident'] else 'ns '}) "
            f"{r['quadrant']:14}{flag}"
        )
        rows.append(r)

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot(rows, results_dir / "squish_plane_v01.png")
    out = {
        "version": "vPOC-v0.1",
        "quantization": QUANT,
        "config": client.config(),
        "n_rerun": N_RERUN,
        "n_para": N_PARA,
        "seed_scheme": "explicit 0..N-1 per prompt",
        "disp_threshold": DISP_T,
        "margin_threshold": MARGIN_T,
        "margin_default": "margin_max (worst-case point) with Newcombe CI",
        "seconds": round(time.time() - t0, 1),
        "results": rows,
    }
    (results_dir / "vpoc_v01.json").write_text(json.dumps(out, indent=2))
    print(
        f"\nwrote results/squish_plane_v01.png and results/vpoc_v01.json in {out['seconds']}s"
    )


if __name__ == "__main__":
    main()
