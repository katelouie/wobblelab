"""Config squish: how much does the *sampling config* alone move the measurement?

D-003 argued the sampling config is itself a squish source -- "as it ships" measures the
harness's packaging, not the model. F-019 sharpened it: ollama's Modelfile defaults are
Qwen's *own* recommended settings (`ollama show --modelfile`: temperature 1, top_k 20,
top_p 0.95, presence_penalty 1.5). This turns that argument into data: run the same
canonical battery under two configs and measure how far the answer distribution moves.

  A · controlled  -- pure temperature sampling from the full softmax (squishlab.CONTROLLED)
  B · qwen-rec     -- + Qwen's recommended top_p 0.95 / top_k 20 / presence_penalty 1.5

Every other knob is held at the controlled-neutral value in BOTH arms, so the contrast
isolates exactly the three choices Qwen makes on top of temperature (the truncation pair
+ the presence penalty), not ollama's opaque internal defaults for everything else.

Observational axis only (canonical reruns, no paraphrases) -- the config's most direct
effect is on rerun dispersion. Prediction on record (see lab journal): small, because
top_p/top_k barely truncate a 2-way split and presence_penalty can't touch the first
token. Run: python experiments/config_ab.py
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from squishlab import OllamaClient, newcombe_diff_ci, wilson_ci  # noqa: E402
from vpoc_real import BATTERY, SUFFIX, parse  # noqa: E402

MODEL = "qwen3.5:0.8b"
QUANT = "Q8_0"
N_RERUN = 150

# Qwen's recommended sampling (ollama Modelfile = Qwen's own settings, F-019), layered on
# the controlled base so only these three deliberate choices differ between the arms.
QWEN_REC = {"top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}

ARMS = {
    "controlled": OllamaClient(MODEL),
    "qwen_rec": OllamaClient(MODEL, options=QWEN_REC),
}


def run(client: OllamaClient, prompt: str, n: int) -> tuple[int, int]:
    """(yes, decided) over n seeded reruns -- same seeds across arms for a fair pair."""
    labels = [parse(client.ask(prompt + SUFFIX, seed=s)) for s in range(n)]
    c = collections.Counter(labels)
    return c["yes"], c["yes"] + c["no"]


def dispersion(yes: int, decided: int) -> float:
    if decided == 0:
        return 0.5
    p = yes / decided
    return min(p, 1 - p)


def majority(yes: int, decided: int) -> str | None:
    if decided == 0:
        return None
    p = yes / decided
    return "yes" if p > 0.5 else "no" if p < 0.5 else None


def measure(name: str, spec: dict) -> dict:
    ans = spec.get("answer")
    arms = {}
    for arm, client in ARMS.items():
        yes, decided = run(client, spec["q"], N_RERUN)
        p = yes / decided if decided else 0.0
        d = dispersion(yes, decided)
        arms[arm] = {
            "yes": yes,
            "decided": decided,
            "p_yes": round(p, 3),
            "dispersion": round(d, 3),
            "disp_ci": tuple(
                round(x, 3) for x in wilson_ci(min(yes, decided - yes), decided)
            )
            if decided
            else (0.0, 0.5),
            "majority": majority(yes, decided),
            "correct": (majority(yes, decided) == ans) if ans else None,
        }
    a, b = arms["controlled"], arms["qwen_rec"]
    # Newcombe CI on the shift in P(yes): the inferential anchor -- did the config move
    # the answer distribution at all, or is any difference within sampling noise?
    lo, hi = newcombe_diff_ci(b["yes"], b["decided"], a["yes"], a["decided"])
    d_shift_confident = lo > 0 or hi < 0  # CI on Δp_yes excludes zero
    return {
        "name": name,
        "question": spec["q"],
        "answer": ans,
        "controlled": a,
        "qwen_rec": b,
        "delta_p_yes": round(b["p_yes"] - a["p_yes"], 3),
        "delta_p_ci": (round(lo, 3), round(hi, 3)),
        "delta_dispersion": round(b["dispersion"] - a["dispersion"], 3),
        "shift_confident": d_shift_confident,
        "majority_flip": a["majority"] != b["majority"],
    }


def plot(rows, path: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=150)
    fig.patch.set_facecolor("#12141a")
    ax.set_facecolor("#181b23")
    rows = sorted(rows, key=lambda r: r["controlled"]["dispersion"])
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        c, q = r["controlled"]["dispersion"], r["qwen_rec"]["dispersion"]
        ax.plot([c, q], [y, y], color="#4a4f5b", lw=2, zorder=1)
        ax.scatter(
            [c],
            [y],
            color="#8fb8d8",
            s=70,
            zorder=2,
            label="controlled" if y == 0 else "",
        )
        ax.scatter(
            [q],
            [y],
            color="#f0a63c",
            s=70,
            zorder=2,
            marker="D" if r["shift_confident"] else "o",
            label="qwen-rec" if y == 0 else "",
        )
    ax.set_yticks(list(ys))
    ax.set_yticklabels(
        [f"{r['name']}  ({r['answer'] or 'null'})" for r in rows],
        color="#c9b79e",
        fontsize=9,
    )
    ax.set_xlim(-0.02, 0.52)
    ax.set_xlabel(
        "observational dispersion  (min(p, 1-p) over 150 reruns)", color="#c9b79e"
    )
    ax.set_title(
        f"CONFIG SQUISH · same battery, two sampling configs · {MODEL} ({QUANT})",
        color="#f6e8ce",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5,
        1.015,
        "controlled (full softmax) vs qwen-rec (top_p .95 / top_k 20 / presence 1.5) · ◆ = Δp_yes CI excludes 0",
        transform=ax.transAxes,
        ha="center",
        color="#8a7358",
        fontsize=8,
    )
    ax.legend(
        facecolor="#181b23",
        edgecolor="#3a3f4b",
        labelcolor="#e7dcc8",
        fontsize=9,
        loc="lower right",
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
        f"{'prompt':17} {'ans':5} {'ctrl_p':>7} {'ctrl_d':>7} {'qwen_p':>7} {'qwen_d':>7} {'Δp':>7} {'Δp CI':>16} flip"
    )
    print("-" * 96)
    for name, spec in BATTERY.items():
        r = measure(name, spec)
        c, q = r["controlled"], r["qwen_rec"]
        sig = "*" if r["shift_confident"] else " "
        print(
            f"{name:17} {str(r['answer'] or 'null'):5} "
            f"{c['p_yes']:>7.3f} {c['dispersion']:>7.3f} "
            f"{q['p_yes']:>7.3f} {q['dispersion']:>7.3f} "
            f"{r['delta_p_yes']:>+7.3f}{sig} [{r['delta_p_ci'][0]:+.2f},{r['delta_p_ci'][1]:+.2f}] "
            f"{'FLIP' if r['majority_flip'] else '--'}"
        )
        rows.append(r)

    mean_ctrl = sum(r["controlled"]["dispersion"] for r in rows) / len(rows)
    mean_qwen = sum(r["qwen_rec"]["dispersion"] for r in rows) / len(rows)
    n_confident = sum(r["shift_confident"] for r in rows)
    n_flip = sum(r["majority_flip"] for r in rows)
    mean_abs_dp = sum(abs(r["delta_p_yes"]) for r in rows) / len(rows)
    dec = [r for r in rows if r["answer"]]
    acc_ctrl = sum(r["controlled"]["correct"] for r in dec) / len(dec)
    acc_qwen = sum(r["qwen_rec"]["correct"] for r in dec) / len(dec)

    print(
        f"\nCONFIG SQUISH · {MODEL} ({QUANT}) · {len(rows)} prompts × {N_RERUN} reruns × 2 configs"
    )
    print(
        f"  mean dispersion:   controlled {mean_ctrl:.3f}  vs  qwen-rec {mean_qwen:.3f}   (Δ {mean_qwen - mean_ctrl:+.3f})"
    )
    print(
        f"  mean |Δp_yes|:     {mean_abs_dp:.3f}   · prompts with Δp CI excluding 0: {n_confident}/{len(rows)}"
    )
    print(f"  majority flips:    {n_flip}/{len(rows)}")
    print(
        f"  decidable accuracy: controlled {acc_ctrl:.3f}  vs  qwen-rec {acc_qwen:.3f}"
    )

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot(rows, results_dir / "config_ab.png")
    out = {
        "model": MODEL,
        "quant": QUANT,
        "n_rerun": N_RERUN,
        "arm_configs": {a: c.config()["options"] for a, c in ARMS.items()},
        "mean_dispersion": {
            "controlled": round(mean_ctrl, 3),
            "qwen_rec": round(mean_qwen, 3),
        },
        "mean_abs_delta_p": round(mean_abs_dp, 3),
        "n_confident_shift": n_confident,
        "n_majority_flip": n_flip,
        "decidable_accuracy": {
            "controlled": round(acc_ctrl, 3),
            "qwen_rec": round(acc_qwen, 3),
        },
        "seconds": round(time.time() - t0, 1),
        "results": rows,
    }
    (results_dir / "config_ab.json").write_text(json.dumps(out, indent=2))
    print(
        f"\nwrote results/config_ab.png and results/config_ab.json in {out['seconds']}s"
    )


if __name__ == "__main__":
    main()
