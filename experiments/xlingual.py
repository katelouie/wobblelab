"""Cross-lingual consistency probe: does a yes/no answer survive EN <-> 中文?

Symmetric. For each prompt, run the English and Chinese forms N times each and measure
cross-lingual disagreement |p_yes(EN) - p_yes(ZH)| with a Newcombe CI. On the
culturally-INVARIANT facts this is pure squish -- language cannot change whether 7 is
prime -- so those are the clean signal. The two cultural items (one Western, one
Chinese) are confounded by framing and included for balance and a bias look, not clean
measurement. The decidability labels also give per-language accuracy, so we can catch a
cross-lingual knowledge gap (right in one language, wrong in the other).

Controlled, portable config (squishlab.client.CONTROLLED), explicit seeds.
Run:  python experiments/xlingual.py
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

from squishlab import OllamaClient, newcombe_diff_ci, wilson_ci  # noqa: E402

MODEL = "qwen3.5:0.8b"
QUANT = "Q8_0"
N = 150
EN_SUFFIX = " Reply with only the single word yes or no."
ZH_SUFFIX = "只回答一个字：是或否。"

client = OllamaClient(MODEL)

BATTERY = {
    "seven_prime": {
        "en": "Is 7 a prime number?",
        "zh": "7是质数吗？",
        "answer": "yes",
        "kind": "invariant",
    },
    "zero_even": {
        "en": "Is zero an even number?",
        "zh": "0是偶数吗？",
        "answer": "yes",
        "kind": "invariant",
    },
    "water_wet": {
        "en": "Is water wet?",
        "zh": "水是湿的吗？",
        "answer": "yes",
        "kind": "invariant",
    },
    "pluto_planet": {
        "en": "Is Pluto a planet?",
        "zh": "冥王星是行星吗？",
        "answer": "no",
        "kind": "invariant",
    },
    "sun_star": {
        "en": "Is the sun a star?",
        "zh": "太阳是恒星吗？",
        "answer": "yes",
        "kind": "invariant",
    },
    "earth_round": {
        "en": "Is the earth round?",
        "zh": "地球是圆的吗？",
        "answer": "yes",
        "kind": "invariant",
    },
    "hotdog_sandwich": {
        "en": "Is a hotdog a sandwich?",
        "zh": "热狗是三明治吗？",
        "answer": None,
        "kind": "western",
    },
    "soymilk_soup": {
        "en": "Is soy milk a soup?",
        "zh": "豆浆是汤吗？",
        "answer": None,
        "kind": "chinese",
    },
}

YES = re.compile(r"\byes\b", re.I)
NO = re.compile(r"\bno\b", re.I)
ZH_NEG = ("否", "不", "没", "错")
ZH_POS = ("是", "对", "正确", "会", "有")


def parse_en(text: str) -> str | None:
    y, n = bool(YES.search(text)), bool(NO.search(text))
    return "yes" if (y and not n) else "no" if (n and not y) else None


def parse_zh(text: str) -> str | None:
    # Negative first: 不是 / 不对 / 否 are "no", and 不是 contains 是.
    if any(c in text for c in ZH_NEG):
        return "no"
    if any(c in text for c in ZH_POS):
        return "yes"
    return None


def run(text: str, suffix: str, parser, n: int) -> tuple[int, int, int]:
    labels = [parser(client.ask(text + suffix, seed=s)) for s in range(n)]
    c = collections.Counter(labels)
    return c["yes"], c["yes"] + c["no"], c[None]


def measure(name: str, spec: dict) -> dict:
    en_y, en_d, en_u = run(spec["en"], EN_SUFFIX, parse_en, N)
    zh_y, zh_d, zh_u = run(spec["zh"], ZH_SUFFIX, parse_zh, N)
    p_en = en_y / en_d if en_d else 0.0
    p_zh = zh_y / zh_d if zh_d else 0.0
    en_maj = "yes" if p_en >= 0.5 else "no"
    zh_maj = "yes" if p_zh >= 0.5 else "no"
    lo, hi = newcombe_diff_ci(en_y, en_d, zh_y, zh_d)  # CI on p_en - p_zh
    confident = lo > 0 or hi < 0
    ans = spec["answer"]
    en_correct = (en_maj == ans) if ans in ("yes", "no") else None
    zh_correct = (zh_maj == ans) if ans in ("yes", "no") else None
    return {
        "name": name,
        "kind": spec["kind"],
        "answer": ans,
        "p_en": round(p_en, 3),
        "p_zh": round(p_zh, 3),
        "en_maj": en_maj,
        "zh_maj": zh_maj,
        "en_ci": tuple(round(x, 3) for x in wilson_ci(en_y, en_d)),
        "zh_ci": tuple(round(x, 3) for x in wilson_ci(zh_y, zh_d)),
        "disagreement": round(abs(p_en - p_zh), 3),
        "diff_ci": (round(lo, 3), round(hi, 3)),
        "confident": confident,
        "en_correct": en_correct,
        "zh_correct": zh_correct,
        "gap": (en_correct is not None and en_correct != zh_correct),
        "en_unparseable": en_u,
        "zh_unparseable": zh_u,
    }


def plot(rows, path: Path) -> None:
    order = sorted(rows, key=lambda r: r["disagreement"])  # squishiest at top
    n = len(order)
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(9.2, 0.66 * n + 1.9), dpi=150)
    fig.patch.set_facecolor("#12141a")
    ax.set_facecolor("#181b23")
    EN_C, ZH_C, TRUTH_C = "#8fb8d8", "#e06666", "#7ec98f"
    for i, r in enumerate(order):
        ax.plot([r["p_en"], r["p_zh"]], [i, i], color="#5b6270", lw=2, zorder=2)
        # ground-truth tick for decidable prompts
        if r["answer"] in ("yes", "no"):
            tx = 1.0 if r["answer"] == "yes" else 0.0
            ax.plot([tx], [i], marker="|", ms=18, color=TRUTH_C, mew=2, zorder=1)
        ax.scatter(
            r["p_en"], i, s=90, color=EN_C, edgecolor="#12141a", lw=1.2, zorder=4
        )
        ax.scatter(
            r["p_zh"], i, s=90, color=ZH_C, edgecolor="#12141a", lw=1.2, zorder=4
        )
        tag = f"Δ={r['disagreement']:.2f}"
        if r["gap"]:
            tag += "  ⚠ gap"
        ax.text(1.02, i, tag, va="center", color="#c9b79e", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["name"] for r in order], color="#e7dcc8", fontsize=9)
    ax.set_xlim(-0.03, 1.28)
    ax.set_xlabel(
        "p(yes)  —  blue = English · red = Chinese · green tick = correct answer",
        color="#c9b79e",
    )
    ax.set_title(
        f"CROSS-LINGUAL CONSISTENCY  ·  EN vs Chinese  ·  {MODEL} ({QUANT})",
        color="#f6e8ce",
        fontsize=13,
        fontweight="bold",
        pad=16,
        loc="left",
    )
    ax.axvline(0.5, color="#3a3f4b", lw=1, ls=(0, (4, 4)))
    for s in ax.spines.values():
        s.set_color("#3a3f4b")
    ax.tick_params(colors="#8a8f9a")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")


def main() -> None:
    t0 = time.time()
    rows = [measure(name, spec) for name, spec in BATTERY.items()]

    invariant = [r for r in rows if r["kind"] == "invariant"]
    mean_disagree = round(sum(r["disagreement"] for r in invariant) / len(invariant), 3)
    gaps = [r["name"] for r in rows if r["gap"]]

    print(f"cross-lingual consistency  ·  {MODEL} ({QUANT})  ·  N={N}/lang")
    print(
        f"mean |Δ| on invariant facts: {mean_disagree}   |   accuracy gaps: {gaps or 'none'}"
    )
    print("-" * 88)
    print(
        f"{'prompt':16} {'kind':10} {'EN(p/maj)':>13} {'ZH(p/maj)':>13} {'|Δ|':>6} acc EN/ZH"
    )
    for r in sorted(rows, key=lambda r: r["disagreement"], reverse=True):
        acc = (
            "--"
            if r["en_correct"] is None
            else f"{'✓' if r['en_correct'] else '✗'}/{'✓' if r['zh_correct'] else '✗'}"
        )
        flag = " ⚠" if r["gap"] else ""
        print(
            f"{r['name']:16} {r['kind']:10} {r['p_en']:.2f} {r['en_maj']:>3}   "
            f"{r['p_zh']:.2f} {r['zh_maj']:>3}   {r['disagreement']:.2f}   {acc}{flag}"
        )

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot(rows, results_dir / "xlingual.png")
    out = {
        "model": MODEL,
        "quantization": QUANT,
        "config": client.config(),
        "n": N,
        "mean_disagreement_invariant": mean_disagree,
        "accuracy_gaps": gaps,
        "seconds": round(time.time() - t0, 1),
        "results": rows,
    }
    (results_dir / "xlingual.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print(
        f"\nwrote results/xlingual.png and results/xlingual.json in {out['seconds']}s"
    )


if __name__ == "__main__":
    main()
