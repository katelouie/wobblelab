# WobbleLab roadmap (living doc)

> A versioned ladder, most-minimal first. The point of this doc is **scope control**: each version
> ships one coherent thing and has an explicit *out of scope* list. The rule that keeps the
> project alive: **do not build vN+1 features inside vN.** When you feel the pull to add "just one
> more facet / metric / benchmark," check which version it belongs to and defer it.
>
> Status is honest and dated. Revise freely; this is a living doc. Design detail lives in
> [architecture.md](architecture.md); positioning in [strategy.md](strategy.md).
>
> **Where we are (2026-07-24):** pre-POC on the measurement core. Infra exists (uniform provider,
> concurrency, remote-GPU runbook, a GPQA loader); the `Harness` / `Knob` / `Benchmark` / `Study`
> abstraction and everything downstream do not yet.

## The guiding cut

- **Dependent variable is pass/fail through V1.** Unimpeachable. Contested quality metrics (code
  complexity, readability) are a free exit for critics and are deferred to V2.
- **Instrument, not benchmark.** Every version runs *on top of* existing harnesses; we never build
  a benchmark.
- **Compute is the constraint that kills personal projects.** Morris-screen before any full sweep;
  start with 2 models, 1 benchmark, ~30 items; resist everything else.
- **`--quick` is sacred.** One interval + the resolution limit. The elegant math is `--full` and
  the presentation, never the default interface.

---

## POC — the engine reproduces a published number

**Goal.** Prove the uniform engine can run a model through one named harness and land on that
harness's published score. De-risks the whole approach: if the handshake fails, nothing else
matters.

**In scope.** `Harness` / `Benchmark` core (minimal). One benchmark (GPQA Diamond). One landmark
harness re-implemented as a config (lm-eval's, since it is the leaderboard's). One known model.
The **handshake gate**: anchor within tolerance of the published number.

**Out of scope.** Knobs. Systems floor. CIs. Multiple harnesses. Cards. A second benchmark.

**Deliverable.** "We reproduced `<model>`'s lm-eval GPQA number within ±X" + the audit JSON.

---

## V0 — the resolution limit

**Goal.** Produce the first genuinely useful, novel number: the per-benchmark resolution limit.

**In scope.** The **base run** (fixed concurrency × shuffle-seed factorial at temp 0) → the
systems noise floor, estimated **per difficulty stratum**. Two models. Compute the **resolution
limit**: the minimum score gap that is statistically distinguishable. Pass/fail DV. One benchmark,
one harness. `--quick` prints it.

**Out of scope.** Named-harness spread. OAT knob sweep. G-theory decomposition. Shapley charts.
The production/perturbation lens. Batch-invariant control arm (V2).

**Deliverable.** "This benchmark resolves models more than **N** points apart; your A-vs-B gap of
M is / is not real." One number, one benchmark, two models.

**Prior art to clear first.** SCORE, POSIX (research/notes.md) — know what is already done.

---

## V1 — the benchmark reliability instrument

**Goal.** The shippable open tool: a variance audit that runs on existing benchmarks and reports
how much you can trust the number.

**In scope.** The **named-harness landmarks** (3: authors' / lm-eval / Inspect) and their spread.
**OAT knob sweep** from the anchor (CoT, shots, budget, format, scoring). The **honest combined
CI** (item + harness + systems). The **G-theory variance decomposition** (basic crossed design) +
**Shapley-effect attribution** waterfall (`--full`). The audit JSON + a **benchmark reliability
card**. Two benchmarks (GPQA Diamond + MMLU-Pro). Pass/fail DV. Aggregation-level labels on
everything.

**Out of scope.** Meaning-preserving perturbations / production lens. Any code-quality metric.
pass^k / solution entropy. The batch-invariant kernel arm. The full-factorial interaction study.
The agentic-eval level.

**Deliverable.** `wobblelab run <model> <benchmark>` → the card: anchor number, spread across three
real harnesses, resolution limit, the wobble hierarchy, one auditable JSON. Public, documented,
with a `--quick` mode.

---

## V2 — the production lens + code-quality-and-reliability

**Goal.** The differentiated depth: reliability under real-world perturbation, and (finally) the
code-quality dimensions. This is the biggest scope-spiral risk, which is exactly why it is walled
off here.

**In scope.**
- **Production lens.** Meaning-preserving perturbations as knobs — typed paraphrase, dialect
  translation, rerun at temp > 0 — measuring consistency (does the answer survive), with a
  semantic-equivalence gate on the perturbations. Elasticity: ∂log(metric)/∂(perturbation).
- **Reliability statistics.** `pass^k` (all-of-k), the pass@k − pass^k gap as inconsistency;
  **solution entropy** (normalized-AST distance across a prompt's k passing samples) — reliability
  decoupled from correctness.
- **Code-quality dimensions** (validated against human judgment before shipping, per the notes):
  cognitive complexity vs reference baseline, **held-out test delta** (the brittleness metric),
  over-defensiveness checks.
- **The batch-invariant control arm** for a clean causal σ²_infra on open models.
- **Full G-theory** crossed/nested design; Sobol `S_i` vs `S_Ti`; the facet × difficulty heatmap.

**Out of scope.** Agentic-eval (model × agentic harness like Claude Code / Cursor). Hosted
service / leaderboard. A paper.

**Deliverable.** The production-reliability card for a specific use case; the code-quality
reliability profile; the full variance waterfall with per-facet posterior error bars.

---

## V3+ — parked (explicitly, so they do not leak earlier)

Named only so they stay out of V1/V2:

- **The agentic-eval level** — model × *agentic* harness (Claude Code, Cursor, open code), the
  "test the model on the harness you actually use" product Denis floated. Legitimately bigger and
  different from the benchmark lens; do not conflate.
- **Cross-backend / hardware** reproducibility (vLLM vs llama.cpp vs hosted; GPU arch drift).
- **The battery at scale** — the full multi-benchmark suite, many models.
- **A hosted service or a published leaderboard** of resolution limits.
- **A paper.** Priority does not matter for a tool; if someone publishes the resolution-limit idea
  first, it costs nothing.
