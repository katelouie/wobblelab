"""Apply the wobble score to a measurement and rank the battery.

Pure post-processing (no model calls): reads a results JSON from a vPOC run, applies
wobblelab.wobble with the battery's decidability labels, prints a ranked table + the
model-level headline, and plots the wobble ranking with its decomposition.

Run:  python experiments/score.py [results/vpoc_v02.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wobblelab.wobble import model_wobble, wobble_score  # noqa: E402
from vpoc_real import BATTERY, MODEL, QUANT  # noqa: E402  (answers live in the battery)

DRIVER_COLOR = {
    "interventional": "#f0a63c",  # phrasing-fragility (amber, the KNIFE-EDGE hue)
    "observational": "#8fb8d8",  # rerun-instability (blue, the NOISY hue)
    "both": "#c39bd8",  # both channels
}


def answer_for(row: dict) -> str | None:
    """Prefer an embedded label; fall back to the battery for older runs."""
    if "answer" in row:
        return row["answer"]
    return BATTERY.get(row["name"], {}).get("answer")


def main() -> None:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else (Path(__file__).resolve().parent.parent / "results" / "vpoc_v02.json")
    )
    data = json.loads(path.read_text())

    scored = []
    for row in data["results"]:
        s = wobble_score(
            dispersion=row["dispersion"],
            margin_max=row["margin_max"],
            answer=answer_for(row),
            disp_ci=row.get("disp_ci"),
            margin_ci=row.get("margin_ci"),
        )
        s["name"] = row["name"]
        s["answer"] = answer_for(row)
        s["hotspot_type"] = row.get("hotspot", {}).get("type")
        scored.append(s)

    scored.sort(key=lambda r: r["wobble"], reverse=True)
    head = model_wobble(scored)

    print(f"wobble ranking  ·  {MODEL} ({QUANT})  ·  from {path.name}")
    print(
        f"model headline: mean={head['mean_wobble']}  median={head['median_wobble']}  "
        f"max={head['max_wobble']}  (n={head['n_prompts']})"
    )
    print("-" * 74)
    print(
        f"{'prompt':17} {'wobble':>6} {'interv':>6} {'obs':>6}  {'driver':14} decidable"
    )
    for r in scored:
        obs = "--" if r["observational"] is None else f"{r['observational']:.2f}"
        print(
            f"{r['name']:17} {r['wobble']:6.2f} {r['interventional']:6.2f} {obs:>6}  "
            f"{r['driver']:14} {'yes' if r['decidable'] else 'no'}"
        )
    print(
        "worst offenders:",
        ", ".join(
            f"{w['name']}({w['wobble']},{w['driver'][:6]})"
            for w in head["worst_offenders"]
        ),
    )

    # ---- plot: wobble ranking, colored by driver, with score CIs ----
    plt.rcParams["font.family"] = "DejaVu Sans"
    order = list(reversed(scored))  # wobbleiest at top
    n = len(order)
    fig, ax = plt.subplots(figsize=(9.0, 0.62 * n + 1.8), dpi=150)
    fig.patch.set_facecolor("#12141a")
    ax.set_facecolor("#181b23")
    for i, r in enumerate(order):
        lo, hi = r.get("wobble_ci", (r["wobble"], r["wobble"]))
        ax.barh(
            i,
            r["wobble"],
            color=DRIVER_COLOR[r["driver"]],
            height=0.62,
            xerr=[[r["wobble"] - lo], [hi - r["wobble"]]],
            error_kw={"ecolor": "#c9b79e", "elinewidth": 1, "capsize": 3},
            zorder=3,
        )
        tag = r["hotspot_type"] if r["driver"] != "observational" else "rerun"
        ax.text(hi + 0.015, i, tag, va="center", color="#8a8f9a", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["name"] for r in order], color="#e7dcc8", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("wobble  →  more sensitive to what shouldn't matter", color="#c9b79e")
    ax.set_title(
        f"WOBBLE SCORE  ·  {MODEL} ({QUANT})",
        color="#f6e8ce",
        fontsize=14,
        fontweight="bold",
        pad=26,
        loc="left",
    )
    ax.text(
        0,
        1.04,
        f"mean {head['mean_wobble']} · median {head['median_wobble']} · "
        f"bars = 95% CI · color = driver (amber phrasing / blue rerun / violet both)",
        transform=ax.transAxes,
        color="#8a7358",
        fontsize=8.5,
    )
    for s in ax.spines.values():
        s.set_color("#3a3f4b")
    ax.tick_params(colors="#8a8f9a")
    fig.tight_layout()
    results_dir = path.parent
    png = results_dir / f"wobble_scores_{path.stem.split('_')[-1]}.png"
    fig.savefig(png, facecolor=fig.get_facecolor(), bbox_inches="tight")

    out = {"source": path.name, "model_headline": head, "scored": scored}
    (results_dir / f"wobble_scores_{path.stem.split('_')[-1]}.json").write_text(
        json.dumps(out, indent=2)
    )
    print(f"\nwrote {png.name} and wobble_scores_{path.stem.split('_')[-1]}.json")


if __name__ == "__main__":
    main()
