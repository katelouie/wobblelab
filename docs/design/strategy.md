# WobbleLab strategy and viability

> An honest assessment of whether this tool finds wobble that *matters*, where the value
> actually is, and what the moat is. Written to guide build decisions, not to cheerlead.
> Confidence is flagged where a claim rests on possibly-stale knowledge and should be verified.

## The question that decides everything

Will WobbleLab find wobble that is **real** (not artifact), **meaningful** (changes a decision),
and **not already obvious**? "Real" is the easy bar. "Meaningful and non-obvious" is the bet.

## Detectability by axis

| Axis | Real? | Meaningful? | Novel? | Read |
|---|---|---|---|---|
| Position / option-order | yes | shrinks with capability | already known (Zheng et al.) | detectable, weak novelty, may be small on frontier models |
| Harness choice (scoring, shots, CoT, extraction) | yes | large swings (6-20pt) | partly | but mostly *choice-dependence*, not fragility; prompt-*wording* σ is small (~0.4-1.6pt) |
| Systems / batch nondeterminism (temp 0) | yes | sub-point on aggregate accuracy | distinctive | a reproducibility story, not "the score is wrong" (matters more in long-gen/agentic) |
| Production perturbations (paraphrase, x-lingual, knife-edge) | yes | direct | under-served | the strongest play (see below) |

Calibrated read: **very high chance of finding measurable wobble; moderate-to-low chance it is a
compelling "so what" for benchmark *comparison*; meaningfully higher for the production case.**
The tool will produce numbers. Whether they change a decision is axis- and tier-dependent.

## The benchmark lens: the overlapping-CI reframe

The natural worry: if honest CIs make model-comparison rankings overlap, is the whole thing a
lost cause? No, it inverts:

- **Overlapping CIs are the deliverable, not the failure.** "These adjacent leaderboard rows are
  statistically indistinguishable once you account for harness + run variance" polices
  over-claimed rankings and is genuinely sellable.
- **It cuts both ways.** The tool will also *confirm* that far comparisons (7B vs 70B, clearly
  separated frontier models) survive any wobble. Honest, less exciting, true for many cases.
- **Item-bootstrap CIs on big benchmarks are narrow.** On 12k items a 3-point gap is often
  significant on item variance alone. What *widens* the effective CI to overlap is the harness
  variance we add. So the sharp claim is: **the leaderboard's CI is too narrow because it ignores
  "which harness did you use"; here is the honest one, and here is which close gaps it dissolves.**

Net: benchmarks are not a lost cause. They are reported with dishonest error bars, which is
fixable, and the value is specifically in policing *close* comparisons.

## The leaderboard-harness reality

**No single canonical harness is used for reporting, and the divergence is the evidence that this
matters.** *(verified)* Concrete: HuggingFace's own writeup found **LLaMA 65B on MMLU scored 0.637
(Original) / 0.637 (HELM) / 0.488 (Eleuther Harness)** — a **~15-point** swing on the same model
and benchmark, purely from prompt formatting ("Choices:" prefix, dropped topic line) and answer
extraction (single-letter probability vs generated-text-match vs full-sequence log-likelihood).
Providers add a fourth number by running internal, prompt-engineered evals. And it is plural even
inside one framework: lm-eval's GPQA is a 0-shot multiple-choice readout, not the authors'
free-generation CoT harness.

Two consequences:
- Strongest real-world evidence that harness wobble is consequential: **one model, many numbers.**
- **"The canonical anchor" is a choice among several defensible harnesses.** Principled default:
  the benchmark *authors'* reference harness. Treat "provider-optimized vs authors' reference" as
  itself a wobble axis. The validation handshake must name *whose* published number it targets.

## The commoditized-infrastructure reality (the biggest build decision)

**Centralized canonical-harness batteries already exist.** *(verified)*
- **lm-evaluation-harness (EleutherAI)** — de facto standard, what the Open LLM Leaderboard runs.
  Declarative YAML + Jinja2; ships GPQA (`leaderboard_gpqa`, 0-shot MC) and MMLU-Pro
  (`leaderboard_mmlu_pro`, 5-shot CoT) with prompts/shots/extraction programmed in. Wrap-able.
- **HELM (Stanford)** — holistic eval that *already runs input-perturbation robustness*: a
  collection of perturbations (typos for robustness, dialect for fairness), scored as **worst-case
  accuracy** before vs after. So input perturbation is taken; it reports worst-case drop, not a
  per-comparison "is this gap real," and not the harness-choice or systems axes.
- **lighteval (HF), Inspect (UK AISI), OpenCompass** — others.

**Decision this forces: wrap, don't rebuild.** Wrap lm-eval-harness (or lighteval / Inspect) as
the anchor provider and add the wobble layer on top. It gives canonical-ish anchors for the whole
battery for free, it is what leaderboards actually use (easier handshake), and it keeps our claim
honest (we did not reinvent eval plumbing).

**Uncomfortable corollary: the measurement instrument is largely commoditized, and HELM already
does *some* perturbation robustness. The moat is not the instrument.** It is the framing, the
honest reporting (CIs + wobble score + the two-lens split), the reliability-card product, and the
curation. We must know exactly what HELM / lm-eval already surface to carve a sharp niche. This is
the immediate research task.

## The production lens: where the tool likely earns its keep

The benchmark lens fights the law of large numbers (12k items averages wobble away) and mostly
finds choice-dependence that within-leaderboard comparisons already control. The production lens
escapes all of that:

- **No averaging.** A production use is one prompt or a narrow class. If *your* prompt is a
  knife-edge, that is 100% of your traffic wobbling, not a 1/sqrt(N)-suppressed average.
- **Live sampling noise** (real usage is temp > 0).
- **The threat model matches**: users rephrase, switch languages, add typos, change framing. A
  model that flips its recommendation on formal-vs-casual phrasing is a concrete liability, and
  that phenomenon appears strongly even on tiny models.
- **Direct "so what"**: "X% of semantically-equivalent phrasings flip this yes/no decision" is
  immediately actionable for a product team.

Hard problems, stated honestly:
- **Usually no ground truth.** You measure *consistency* (does it wobble), often not *correctness*,
  and a model *should* wobble on genuinely ambiguous inputs (the decidability problem, D-007).
- **Defining a good meaning-preserving perturbation is hard** and model-dependent.
- **Harder to make a clean benchmark-style number**; it is more bespoke per use case.

Higher value, more bespoke. This is where a reliability card would actually change a deployment.

## The crisp deliverable (sharpened 2026-07-24)

The benchmark lens's headline is the **resolution limit**: "this benchmark distinguishes models
more than N points apart, and no finer." One number, legible without statistics, and nobody
publishes it. Its backbone is **Generalizability Theory** — a psychometrics formalism ML eval
almost never uses, which is the real comparative advantage (measurement-theory arbitrage, not
ML-research cleverness). Full treatment in [architecture.md](architecture.md); the build ladder
(POC → V0 → V1 → V2) in [roadmap.md](roadmap.md).

## Honest bottom line

- The **instrument is largely commoditized** (lm-eval, HELM). Wrap it. The moat is honest
  reporting and framing, not novel measurement.
- The **benchmark lens** earns a real but *narrow* claim, made crisp: **the resolution limit** —
  "which close leaderboard gaps are noise." Worth doing, genuinely useful, not world-shaking.
- **Closest prior art is SCORE / POSIX** (research/notes.md) — read before claiming novelty; our
  deltas are the systems-noise facet, the resolution limit, and the G-theory decomposition.
- The **production lens** is the differentiated, high-"so what" play: it escapes the averaging
  problem and matches the real threat model. Harder (no ground truth) but it is where the value is.
- **Strongest product shape:** lean on wrapped-canonical benchmarks for the "your leaderboard
  number ± the harness you used" credibility hook, then deliver the real value in a
  production-reliability card for a *specific* use case.

## A sharper benchmark-lens product (from the research)

The strongest benchmark-lens play is probably **not perturbations we invent** (HELM has input
perturbation; inventing our own invites "is that a real harness anyone uses?"). It is
**quantifying the spread across the harnesses people already use** — lm-eval vs HELM vs the
authors' reference vs the provider's reported number — and reporting **which model comparisons
survive that spread.** The LLaMA-65B 15-point example is the proof of concept: those are four
real, named, defensible harnesses, and the gap between them is the wobble. This is more credible
than a synthetic perturbation and it directly attacks the saturated-leaderboard problem.

## What this changes in the architecture

- Add a decision to `architecture.md`: **the canonical anchor is sourced by wrapping an existing
  harness (lm-eval-harness et al.), not hand-rolled per benchmark.** `Benchmark` becomes a thin
  adapter over the wrapped harness's task definition plus our knobs and gates.
- The harness-choice "knob" is often a **swap between named real harnesses**, not a synthetic
  tweak. Support "run this model under lm-eval's / HELM's / the authors' harness and report the
  spread" as a first-class study.
- Lead with the **production lens** for the sharp claims; keep the benchmark lens for credibility.

## Open empirical questions

*Resolved:* lm-eval ships GPQA/MMLU-Pro as wrap-able tasks; HELM runs input-perturbation
robustness (typos/dialect, worst-case); one-model-many-numbers is real and large (~15pt on
LLaMA-65B MMLU).

*Still open:*
- Does lm-eval expose the **hooks** we need to swap scoring / shots / budget and run our knobs
  around its task, or do we fork the task definitions? (Determines wrap effort.)
- **HELM's perturbation module** in code detail: exactly which perturbations, how worst-case is
  computed, whether any CI is reported. Sets the precise niche boundary.
- Does anyone ship a **per-model reliability card** with honest CIs + fragility? If so, what is
  missing? (The reporting-side gap is the moat; confirm it is open.)
