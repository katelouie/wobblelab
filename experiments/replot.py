"""Regenerate the result figures from their saved JSON — no model calls.

The squishlab -> wobblelab rename fixed every *source* string, but the PNGs still
carried "SQUISH" baked into their pixels. Re-running the experiments would redraw
the text, but it would also resample the model and *change the numbers* the README
and pitch cite (65% at slot A, the cereal_soup knife-edge, the 0.48 swing). That
breaks the receipts.

So this reads each figure's saved JSON and calls the *same* plot() function the
original run used, with the current (already-renamed) title strings. Same data,
corrected labels. Pure post-processing.

Covers the six public-facing figures (README + pitch). The superseded v0.1 / "real"
wobble planes are journal-only dated snapshots and are left as-is.

Run:  python experiments/replot.py
"""

from __future__ import annotations

import json
from pathlib import Path

# The plot() functions read their module-level MODEL/QUANT/SUBJECT/VERSION globals
# for titles; those are already the renamed, current values, so importing and
# calling plot() reproduces each current figure faithfully. The OllamaClient these
# modules instantiate at import time is inert (stores config, no network call).
from bench import plot as plot_bench
from config_ab import plot as plot_config
from harness import plot as plot_harness
from run_pillars import plot as plot_pillars
from vpoc_real import plot as plot_plane
from xlingual import plot as plot_xlingual

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def replot_plane() -> str:
    j = _load("vpoc_v02.json")
    plot_plane(j["results"], RESULTS / "wobble_plane_v02.png")
    return "wobble_plane_v02.png"


def replot_harness() -> str:
    j = _load("harness_world_religions.json")
    order = [("gen", "0shot"), ("ll", "0shot"), ("gen", "5shot"), ("ll", "5shot")]
    grid = {(s, sh): j["grid"][f"{s}_{sh}"] for s, sh in order}
    plot_harness(grid, j["profiles_0shot"], RESULTS / "harness_world_religions.png")
    return "harness_world_religions.png"


def replot_bench() -> str:
    j = _load("bench_world_religions.json")
    plot_bench(
        j["accuracy_by_position"],
        j["accuracy_debiased"],
        j["accuracy_ci_bootstrap"],
        j["chosen_letter_dist"],
        RESULTS / "bench_world_religions.png",
    )
    return "bench_world_religions.png"


def replot_config() -> str:
    j = _load("config_ab.json")
    plot_config(j["results"], RESULTS / "config_ab.png")
    return "config_ab.png"


def replot_xlingual() -> str:
    j = _load("xlingual.json")
    plot_xlingual(j["results"], RESULTS / "xlingual.png")
    return "xlingual.png"


def replot_pillars() -> str:
    j = _load("pillars_world_religions.json")
    plot_pillars(
        j["report"]["wobble_by_kind"],
        j["stability"]["runs"],
        j["stability"]["mean"],
        RESULTS / "pillars_world_religions.png",
    )
    return "pillars_world_religions.png"


def main() -> None:
    for fn in (
        replot_plane,
        replot_harness,
        replot_bench,
        replot_config,
        replot_xlingual,
        replot_pillars,
    ):
        name = fn()
        print(f"redrew results/{name}")


if __name__ == "__main__":
    main()
