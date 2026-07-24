# WobbleLab — Lab Journal

The decision + discovery record for wobblelab. Every methodological fork, why it
went the way it did, and what the data taught us. Append-only, newest context at
the bottom of each section. When a decision is later overturned, don't delete it —
add a superseding entry so the reasoning trail survives.

**wobblelab measures *wobble*: how much a model's answer moves under things that
should not change it.** Two lenses, which turn out to be two axes:

- **observational** — rerun the *same* prompt; how much does the answer disperse? (sampling noise)
- **interventional** — perturb the prompt in a *meaning-preserving* way; does the answer shift? (fragility)

---

## Decisions ledger

| # | Decision | One-line rationale |
|---|---|---|
| **D-001** | The wobble *plane*: dispersion (y) × margin (x), four quadrants (SOLID / KNIFE-EDGE / NOISY-BUT-SURE / COIN-FLIP) | The two lenses are orthogonal; one axis can't tell a solid answer from a knife-edge (see F-001) |
| **D-002** | Model = `qwen3.5:0.8b` at **Q8_0** (8-bit), `think:false` | Quantization is itself a wobble source, so 8-bit not 4-bit; no-think avoids reasoning-trace variance polluting the measurement |
| **D-003** | **Controlled, portable sampling config** (pure temperature sampling from the full softmax, explicit per-call seeds) — NOT ollama's Modelfile defaults | Implicit harness defaults are a config-level wobble source; "as it ships" measures *ollama's packaging*, not the model, and won't reproduce on vLLM/llama.cpp |
| **D-004** | Interventional margin is **continuous shift-based**; plane default = **worst-case (`margin_max`)** within a prompt; model headline = **central tendency** across prompts; always compute `max`/`mean`/`max_ci`; not a user knob | A knife-edge is *defined* by the existence of a breaking rephrase → worst-case within-prompt; different aggregation levels want different stats (see 2026-07-20 entry) |
| **D-005** | Everything carries CIs (Wilson on rates, Newcombe on shifts). Point for placement, CI for honesty; **CI straddling a threshold → `unresolved`**, not forced into a quadrant. Both axes identical. | A reliability tool must not report point estimates without error bars; N=30 thresholds were slicing sampling noise (F-004) |
| **D-006** | **Typed paraphrase taxonomy** (casual / formal / hedge / reorder / lexical), constant K per battery | Makes the hot-spot diagnostic (*which kind* of change broke it) and keeps worst-case margins comparable |
| **D-007** | **Wobble score = `max(interventional_deficit, gated·observational)`.** Interventional `(1 − margin)` always counts; dispersion (`2·D`) counts only for *decidable* prompts, gated to 0 for undecidable ones. Worst-case combine (max), full decomposition + score CI retained. Battery gains an `answer: yes\|no\|null` label; null when decidability is disputed. | Phrasing-fragility is unconditional wobble; rerun-variation is wobble only when there's a fact to be stable about. Appropriate uncertainty and epistemic instability are behaviorally indistinguishable (F-013), so the decidability label is *required*, not a shortcut. |
| **D-008** | **Bootstrap is for CIs on *derived/composite* statistics, not a substitute for reruns.** Use it for the wobble score, the model-level headline (resample prompts), and benchmark accuracy (resample items — the correct unit, since reruns of one item are correlated). It is NOT a shortcut around actual sampling: resampling can't invent power, and for a plain proportion it just reproduces Wilson. The real way to spend fewer *calls* is adaptive/sequential sampling, not bootstrap. | Kate asked whether we could "bootstrap into" reruns; the honest answer is no for base rates, yes for composites — where there's no clean closed form and resampling recorded outcomes is free. |
| **D-009** | **Harness wobble = sweep the eval harness on fixed items.** Two axes: scoring (`gen` = sample+parse a letter vs `ll` = argmax over first-token letter logprobs, the official method) × shots (0 vs 5, exemplars from MMLU's `dev` split). Metric = **natural-position accuracy** (the leaderboard number), CI bootstrapped over items; spread across the 2×2 = harness wobble, main effects decompose it. `ll` via ollama native `/api/chat` `logprobs:true, top_logprobs:20`. | F-019 argued official-vs-ours is "not directly comparable"; this turns that argument into a *quantified decomposition* — how many points are scoring vs shots vs everything-else — instead of a hand-wave. Same items/seed as bench.py so `gen/0-shot` cross-checks F-017. |
| **D-010** | **Concurrency lives in the harness, not the provider; one OpenAI-compatible adapter reaches every batching backend.** `evaluate(concurrency=N)` thread-pools the provider calls while aggregation stays pure and order-deterministic (identical numbers to the sequential run). Providers stay synchronous (`ask` / `rank_letters`); a single `OpenAICompatibleProvider` points at vLLM / llama.cpp / mlx / ollama-`/v1` / hosted by `base_url`. | Backends batch *server-side* when you keep many requests in flight — so the client just needs concurrency + a portable HTTP shape, and one adapter covers local *and* remote. Keeps the harness backend-agnostic and the IP-clean local↔GPU switch a one-line change (F-024). |

## Findings ledger

| # | Finding | Where |
|---|---|---|
| **F-001** | In pure binary with only the observational axis, `dispersion = 0.25 − margin²` — the two collapse to one parabola and KNIFE-EDGE is unreachable. The second axis is *necessary*, not decorative. | vPOC-mock |
| **F-002** | The 0.8B's confidence **does not track truth or difficulty**: "water is wet" (trivially true) came back noisy; "is a hotdog a sandwich" (genuinely contested) came back rock-solid. | smoke test |
| **F-003** | **`zero_even` is paraphrase-sensitive.** "Is zero an even number?" → yes (correct), but most meaning-identical rephrasings → no (wrong). A definite math fact is phrasing-dependent and mostly wrong. The gem. | vPOC v0 |
| **F-004** | v0's majority-flip margin was too blunt (7/8 prompts pinned at 1.0); N=30 too few (dispersion threshold split a one-sample-wide cluster). Motivated v0.1. | vPOC v0 |
| **F-005** | **Genuine KNIFE-EDGES exist.** `pluto_planet` (disp 0.02, margin 0.58) and `cereal_soup` (disp 0.07, margin 0.47) are rock-solid on rerun but fragile to paraphrase — the exact phenomenon the two-axis design targets, and invisible to the observational axis alone. | vPOC v0.1 |
| **F-006** | **`reorder` is the model's dominant fragility** (hot-spot for 4/8 prompts). Caveat: our reorder form is a stilted colon-fronting ("A planet: is Pluto one?"), so it's partly a paraphrase-quality artifact — D-006's worst-case/quality caveat, live. | vPOC v0.1 |
| **F-007** | At N=60 / M=25, **all 8 points are `unresolved`** — the 95% CIs are too wide to confidently assign quadrants. The framework correctly refuses to over-claim. Under-powered; needs larger N/M or a continuous/deadzone treatment. | vPOC v0.1 |
| **F-008** | **v0.1's "reorder dominates" was a paraphrase-quality artifact.** With natural reorders, `reorder` drops from 4/8 hot-spots to **1/8**; fragility spreads across casual/lexical/hedge. Confirms F-006 and D-006's caveat — the framework self-corrected a false signal. | vPOC v0.2 |
| **F-009** | **`pluto_planet` was a *partial* artifact.** Its v0.1 "knife-edge" (margin 0.58) relaxed to borderline-SOLID (0.65) once the stilted reorder was replaced. Not a clean knife-edge; the colon-fronting was inflating its fragility. | vPOC v0.2 |
| **F-010** | **`cereal_soup` is the robust knife-edge.** Survives natural reorders *and* the power bump: near-unanimous "no" on rerun (disp 0.10) yet a lexical rephrase ("Is a bowl of cereal soup?") shifts it (margin 0.47, `max_ci` 0.61 = confident). The clearest confirmed two-axis payoff. | vPOC v0.2 |
| **F-011** | Even at N=150 / M=60, only **2/8** points resolve — prompts cluster right at the thresholds (margin ≈ 0.60, disp ≈ 0.15), which slice the densest band. Points to data-derived thresholds or continuous confident-membership, not a fixed grid. | vPOC v0.2 |
| **F-012** | **The model has a ~5–15% rerun-noise floor** even on rock-solid facts (water 0.06, pluto 0.05, prime 0.09). So a *fixed* dispersion threshold will always cut through the "solid" cluster — the hard grid was the wrong abstraction (→ the continuous wobble score, D-007). | vPOC v0.2 |
| **F-013** | **Appropriate uncertainty and epistemic instability are behaviorally indistinguishable** from outputs alone: a decidable fact the model has lost its grip on and a genuinely undecidable question produce the *same* output distribution (phrasing-stable dispersion). Only external ground truth separates them — which is why D-007's decidability gate is necessary, not a convenience. | reasoning → D-007 |
| **F-014** | **Qwen-0.8B is cross-lingually *consistent* on clear facts:** mean EN↔ZH `\|Δ\|` = 0.10, and **zero majority accuracy gaps** — it gets every decidable fact right in *both* languages. More robust than predicted; my "we'll catch an accuracy gap" bet lost. | x-lingual |
| **F-015** | **Pluto is the cross-lingual outlier among facts** (`\|Δ\|`=0.33): 95% "not a planet" in English vs only 62% in Chinese. The model's grip on the recent, English-discourse-heavy IAU reclassification is much weaker in Chinese — real cross-lingual wobble, on the one fact with a contested history. | x-lingual |
| **F-016** | **Cultural framing is language-bound.** Both cultural items swing hard cross-lingually (hotdog 0.32, soymilk 0.43), and 豆浆是汤吗 ("is soy milk a soup") **flips its majority**: *no* in English (0.43), *yes* in Chinese (0.86). Confounded (not clean wobble), but a crisp demo that category conventions live per-language — and the Chinese-culture item swung most. | x-lingual |
| **F-017** | **A bare benchmark number is nearly meaningless without the wobble breakdown.** On MMLU:world_religions the 0.8B's 36% accuracy is a *position-bias artifact*: accuracy swings **48 points** by answer-position (A 65% → B 17%, *below* the 25% chance floor), driven by a 56%-A chosen-letter bias. The whole wobblelab thesis, on a real benchmark, first run. | bench |
| **F-018** | **Bootstrap-over-items widens the accuracy CI honestly:** [0.31, 0.42] vs the naive Wilson-over-trials [0.33, 0.40] — item clustering matters (D-008). And **~29% of answers flip their content under option reordering** (interventional wobble 0.29 [0.24, 0.35]). | bench |
| **F-019** | **Official Qwen3.5-0.8B numbers exist** (HF model card, non-thinking mode: MMLU-Pro 29.7, MMLU-Redux 48.5, C-Eval 46.4, MMMLU 34.1; thinking mode ~10–13 pts higher). Two sharp points: **(a)** Qwen's *own* recommended inference settings are `top_p 0.95 / top_k 20 / presence_penalty 1.5 / temp 1.0` = **ollama's Modelfile default verbatim** — so the config we dismissed as "ollama's packaging" (D-003) is Qwen's *official* one. There is no *neutral* config; even the official number is a config choice, which strengthens, not weakens, the point. **(b)** Official scores are log-likelihood + few-shot + full-precision + full-benchmark; ours is generation + zero-shot + Q8 + a 40-item slice — *not directly comparable*, and none of the official numbers carry a CI or a position-bias/wobble breakdown. | web / model card |
| **F-020** | **ollama's logprobs are top-20-capped but near-lossless for MC.** `top_logprobs` hard-caps at 20 (50+ → HTTP 400); it returns the top-N alternatives, not the full vocab softmax. But on a lettered-MC first token the distribution is so peaked that the top-20 captures **99.9%** of the mass, and candidate-letter coverage measured **0.95 at 0-shot / 1.00 at 5-shot** (few-shot format-anchoring closes the gap). When a letter falls outside top-20 its true prob is < e⁻⁹ ≈ 0.0001, so treating it as −∞ for the argmax is harmless. The `ll` scorer is effectively exact. | probe / harness |
| **F-021** | **A 20-point spread from harness alone.** Same model, same 40 items: natural-position accuracy runs **0.30 → 0.50** across the four harnesses (`ll/0-shot` 0.50, `gen/0-shot` 0.41, `ll/5-shot` 0.38, `gen/5-shot` 0.30). Main effects: **scoring (ll−gen) +0.08** (reading the distribution beats sampling a letter), **shots (5−0) −0.12** (few-shot *hurts* this 0.8B, robustly across both scoring methods). So F-019's 36-vs-48.5 "gap" was mostly scoring-method + which-number-you-report, not capability: measured the official way (`ll`), our crude Q8 40-item slice brackets the official 48.5 (0.50 [0.35,0.65]). The leaderboard number is a *harness choice*, quantified. | harness |
| **F-023** | **Config wobble is small and localized on binary decisions — prediction confirmed.** Controlled full-softmax vs Qwen-recommended (top_p .95 / top_k 20 / presence 1.5), same 8-prompt battery, 150 reruns each: **mean dispersion 0.141 vs 0.140 (Δ −0.001)**, mean \|Δp_yes\| 0.019, **1/8** prompts with a Δp CI excluding zero, **0** majority flips, decidable accuracy 1.00 both. The lone confident shift is `pluto_planet` (dispersion 0.047 → **0.000**): truncation clips the residual minority mass on the one near-unanimous prompt, so "as it ships" makes the model look *slightly more* reliable exactly where it had a sliver of doubt. Pre-registered prediction (top_p/top_k barely truncate a 2-way split; presence can't touch the first token) held. Refines D-003: config wobble is real but its bite scales with the *width of the output tail* — negligible on binary yes/no, and expected to matter on the MC benchmark's chosen-letter tail (where truncation would nudge `gen` toward the more-A-biased argmax, linking to F-022). | config_ab |
| **F-024** | **Concurrency is the cheap win; llama.cpp is the fast local backend; mlx-lm's server does not batch.** Harness `concurrency` alone gave **~4.7×** on the existing ollama with no config change (10.9→2.3s on a 10-item ll pass). Head-to-head on the *identical* Qwen3-0.6B Q4_K_M GGUF (ollama vs llama.cpp) and Qwen3-0.6B-8bit (mlx), 40-prompt load, req/s: **llama.cpp** 11.3→**28** (best; ~1.9× ollama single-stream, saturates at its 8 slots), **ollama** 6.0→~18 (auto-batches to ~4), **mlx-lm** 6.1→~6 (*flat* — `mlx_lm.server` serializes concurrent requests). Turns the ~9h full-MMLU reorder pass into ~2h (ollama) / ~1.3h (llama.cpp), all local. mlx_parallm (batched `batch_generate`) fixes MLX's gap but is Python-only (no HTTP), generation-only (no logprobs → no `ll` scoring), and Qwen-unverified — so llama.cpp stays the pick. | backend bench |
| **F-022** | **Log-likelihood scoring does *not* cure position bias — it sharpens it.** Switching gen→ll, the 0-shot position swing *grows* 0.48 → 0.70 and the chosen-A share *rises* 56% → 67%, even as debiased accuracy improves 0.36 → 0.48. Sampling at temp 1 blurs the model's positional preference (it occasionally samples off-top); argmax commits to the top logit every time, so ll is simultaneously more accurate *and* more polarized. The bias lives in the weights, not the readout — the "cleaner" scoring method exposes it undiluted rather than removing it. My "ll will fix the bias" prediction lost. | harness |

---

## 2026-07-18 — design + vPOC-mock

Framed wobble as sensitivity to things that shouldn't matter, and split it into the
two lenses above. Built a synthetic mock (`scratchpad/wobble_poc.py`) with two
independent knobs per prompt: `true_p_yes` (drives observational dispersion) and
`true_flip_rate` (drives interventional margin).

**F-001 (the reason the plane has two axes):** with only the observational lens in a
pure binary task, `dispersion = 0.25 − vote_margin²` — dispersion and margin are the
*same number* on a parabola, so you can never separate a SOLID answer from a
KNIFE-EDGE (looks-decided-but-flips-on-paraphrase). Adding the interventional axis is
what opens the plane. → **D-001.**

Model plan: a tiny local model so we control everything. 8-bit not 4-bit because
aggressive quantization is *itself* a wobble source we don't want to confound with
the model's own behavior. → **D-002.**

## 2026-07-20 — vPOC-real: first light

Model downloaded: `qwen3.5:0.8b`, and `ollama show` confirmed **Q8_0** (the ~1 GB size
was the tell; a Q4 would be ~half). Default temperature 1 (good — we need >0 to see
dispersion), but the Modelfile also ships `presence_penalty 1.5 / top_k 20 / top_p
0.95` and a `thinking` capability.

**Smoke test:** API works, `think:false` suppresses the reasoning trace (message is
just role+content), yes/no parses cleanly. And **F-002**: on 8 reruns, "water is wet"
split 6/2 while "is a hotdog a sandwich" went 8/8 — the model is jittery on the
obvious and confident on the contested. Confidence ≠ difficulty.

**Config decision (D-003):** the smoke ran on ollama's Modelfile defaults. That's a
confound — it measures *ollama's packaging*, and another harness ships different
implicit defaults, so nothing would reproduce or compare. Switched to a fully
explicit, portable config: **pure temperature sampling from the full softmax**
(temp 1, top_p 1, top_k 0, min_p 0, all penalties 0, repeat_penalty 1) with
**explicit per-call seeds 0..N-1** so the whole run is bit-reproducible.

## 2026-07-20 — vPOC v0 (controlled, N=30)

720 controlled calls, 2.5 min. The plane lit up (3/4 quadrants). Two useful outcomes:

- **F-003 (the gem):** `zero_even`. Canonical "Is zero an even number?" → **yes**, but
  three of four rephrasings → **no** (wrong), and the flips don't follow a clean rule
  ("Is 0 even?" → yes, "Is the number zero an even number?" → no). A definite math
  fact comes out phrasing-dependent and mostly *incorrect*. This is the phenomenon
  wobblelab exists to catch, found on the first real run, un-designed.
- **F-004 (what v0 got wrong):** (a) the margin metric was majority-flip — it only
  registers a paraphrase that crosses 50%, so it pinned 7/8 prompts at margin 1.0 and
  the whole x-axis collapsed to the right edge. (b) N=30 gives dispersions with wide
  error bars; the 0.15 threshold split `water_wet` (4/30) from the
  `hotdog/prime/tomato` cluster (5/30) — a **one-sample** difference deciding a
  quadrant. Meaningless.

Artifacts: `results/vpoc_real.json`, `results/wobble_plane_real.png`.

## 2026-07-20 — the max-vs-mean decision (D-004), worked in full

The v0.1 fix for the blunt x-axis is a *continuous* shift-based margin: per paraphrase,
`δ = p_yes(paraphrase) − p_yes(canonical)`; aggregate the `|δ|`s into one margin. But
*how* to aggregate is the question that defines what "margin" means. Long reasoning,
preserved because it's load-bearing:

**Max and mean are not rival estimators of one truth — they're different quantities.**
`mean|δ|` = average fragility ("random rephrase, how much moves?"). `max|δ|` = peak
fragility ("does there *exist* a breaking rephrase?"). Both legitimate.

**The statistics punish naive max.** Each `|δ_k|` is noisy (~0.13 at M=15). So (a) max
is *noise-inflated* — even with all true shifts zero, the max of K folded-noise values
is biased up; and (b) max is *K-dependent* — more paraphrases → higher max → margin
drifts down as you add paraphrases, which **breaks comparability** (fatal for a
scoring framework; a determinism tool whose own scores wobble with paraphrase count is
a dogfooding failure). Mean has neither problem but **dilutes the signal**: 1 hard flip
in 10 → `mean|δ|`≈0.12 → "robust," and the knife-edge vanishes.

**The mission forces worst-case.** A KNIFE-EDGE is *defined* by the existence of a
breaking rephrase; there is no average-case definition of it. So the within-prompt
aggregation must be worst-case or the axis can't do its one job.

**Resolution (three moves):**
1. *Different aggregation levels want different stats.* **Within a prompt** (paraphrase
   shifts → margin): worst-case. **Across prompts** (per-prompt margins → the model's
   headline wobble): central tendency + named worst-offenders. The max-vs-mean fight
   was two questions in one coat.
2. *Tame max with CIs, don't replace it.* Plane plots the worst-case **point**
   (`1 − max|δ̂|`) with its **Newcombe CI** as an error bar, and flags any point whose
   CI straddles the threshold as `unresolved`. Max's noise-inflation becomes a *visible
   uncertainty* instead of a silent lie. (Same treatment as the dispersion axis — D-005.)
3. *Compute all three, always.* Store the full per-paraphrase shift vector; emit
   `margin_max` (plane), `margin_mean` (average fragility), `margin_max_ci`
   (`1 − max` of the CI-lower-bounds — fires only when a shift is statistically
   confident, never on noise), plus the **hot-spot** (argmax paraphrase + its type).
   That *is* "run with both for a while": we run all of them every time and let the
   data pick over the next few batteries.

**Not a user-facing knob.** A `p`-power-mean dial nobody can interpret is a bad
surface; a headline number should mean one thing.

**Two caveats worst-case amplifies:** (1) paraphrase quality is now load-bearing — a
meaning-*changing* paraphrase produces a false wobble; mitigations are the auditable
hot-spot every run and the confident-shift requirement. (2) constant K per battery or
comparability dies. → **D-006** (typed, constant-K taxonomy).

## 2026-07-20 — v0.1 built

Refactored the reusable core into the package: `src/wobblelab/stats.py` (Wilson +
Newcombe + `confident_shift`, with `tests/test_stats.py`, 6/6 green) and
`src/wobblelab/client.py` (the explicit `CONTROLLED` config + `OllamaClient`). The
experiment (`experiments/vpoc_real.py`) now: continuous margins with all three
aggregations + hot-spot; Wilson/Newcombe CIs on both axes; typed paraphrases
(N=60 canonical / 25 per paraphrase); plane with CI error bars and hollow markers for
`unresolved` points. Run launched → results pending in this entry.

**Results (308s, 1,480 calls).** `results/vpoc_v01.json`, `results/wobble_plane_v01.png`.

- **The x-axis came alive.** Margins now spread 0.45 (`hotdog_sandwich`, wobbleiest) to
  0.92 (`tomato_fruit`, most robust); v0 had 7/8 pinned at 1.0. The continuous
  shift-based metric fixed the dead axis.
- **F-005 — real knife-edges.** `pluto_planet` and `cereal_soup` sit low on *both* axes:
  the model is nearly unanimous on rerun (disp 0.02 / 0.07) yet a rephrasing shifts the
  answer (margin 0.58 / 0.47). The observational axis alone called these "solid"; the
  interventional axis exposed them. This is the payoff of the two-axis design, on real data.
- **F-006 — `reorder` dominates.** It's the hot-spot for water/hotdog/prime/pluto. Real
  signal *and* a lesson: the colon-fronting construction is stilted, so some of it is
  paraphrase quality, not pure model fragility. The auditable hot-spot did its job.
- **F-007 — honestly under-powered.** Every point plotted hollow (`unresolved`): at this N,
  the CIs straddle the thresholds, so quadrant labels aren't trustworthy yet. The
  framework flagged its own uncertainty instead of faking crisp quadrants. Good. The fix
  is more samples (or a continuous plane with deadzones), not a different metric.
- `margin_max_ci` earned its place: it separates *confident* fragility (hotdog/pluto/cereal,
  significant hot-spots) from *maybe-noise* (`tomato_fruit`, hot-spot `ns`, max_ci 1.00).

## 2026-07-20 — vPOC v0.2 (natural reorders + power)

N=150 / M=60, 3,600 calls, 12.4 min. Two changes: reorder paraphrases rewritten as
natural restructurings (killing the colon-fronting confound), and N/M bumped ~2.5× to
halve the CIs. `results/vpoc_v02.json`, `results/wobble_plane_v02.png`.

The headline: **the framework caught and corrected a false positive from our own v0.1.**

- **F-008 — reorder was an artifact.** With fluent reorders, `reorder` fell from 4/8
  hot-spots to 1/8; the model is *not* specially fragile to restructuring, it was
  fragile to the awkward phrasing we fed it. The worst-case metric's D-006 caveat
  (paraphrase quality is load-bearing) played out exactly as predicted, and the
  auditable hot-spot + a follow-up run fixed it.
- **F-009 — pluto downgraded.** Its v0.1 knife-edge was partly the same artifact:
  natural reorder → margin 0.58 → 0.65, borderline-SOLID now, not a clean knife-edge.
- **F-010 — cereal is the real one.** It's the single prompt that stays bottom-left
  through both the paraphrase fix and the power bump: solid on rerun, breaks on a
  lexical rephrase, confident (`max_ci` 0.61). If we publish one example of "looks
  decided, isn't," it's `cereal_soup`.
- **Power paid off, partly.** CIs visibly tighter (~halved); `water_wet` and
  `seven_prime` now plot **filled** (resolved SOLID). But **F-011**: 6/8 still hollow,
  because the point cloud piles up right on the threshold lines. The remaining problem
  isn't noise now, it's that the *thresholds* (0.15 / 0.60) sit in the densest part of
  the data. More N won't fix that; rethinking the thresholds will.
- Net picture of this model: **mostly solid, with diffuse mild fragility and one true
  knife-edge.** v0.1 over-stated the wobble (bad paraphrases inflated it); v0.2 is the
  honest version.

## 2026-07-20 — the wobble score + the decidability gate (D-007)

F-011 forced the question: the hard quadrant grid can't be decisive when the model's
own **~10% rerun-noise floor** (F-012) sits right under the dispersion line and every
point piles onto the thresholds. The fix isn't a better line, it's the continuous
**wobble score** from the original design (headline + factors + hot-spot).

But combining the two axes surfaced a real fork: **they aren't symmetric wobble.**

- **Interventional margin-deficit `(1 − M)` is unconditional.** A meaning-preserving
  rephrase should never move a factual answer, decidable or not. Always a defect.
- **Dispersion is conditional.** Seed-driven variation is a defect *iff there's a fact
  to be stable about*. On an undecidable prompt it's appropriate calibration.

The deep reason (**F-013**): you *cannot* tell appropriate uncertainty from epistemic
instability from outputs alone. A decidable fact the model has lost its grip on
(50/50, phrasing-stable) and a genuinely undecidable question produce the identical
output distribution. Only external ground truth distinguishes them. So a naive
`D + (1−M)` mis-scores `rain_tomorrow` (honestly uncertain) as maximally wobbly, and
a pure-`(1−M)` misses seed-wobble on decidable facts (a 50/50 "is water wet" would read
as perfectly reliable). Neither works.

**Decision (D-007):** `wobble = max( 1 − margin , [decidable]·2·dispersion )`. The
interventional channel always counts; dispersion is **gated by a decidability label**
(part of the supervision we already committed to), worst-case combined, decomposition
kept. Roadmap-consistent: supervised → full gated score; unsupervised/discovered →
fall back to the unconditionally-safe `1 − margin`, dispersion reported raw and un-gated.

**It behaves.** Applied to the v0.2 data (pure post-processing, no re-run):

```
cereal_soup     0.53  interv (lexical)    rain_tomorrow   0.28  interv (gated: -- )
hotdog_sandwich 0.39  interv (lexical)    seven_prime     0.23  interv
pluto_planet    0.35  interv (reorder)    water_wet       0.16  interv
zero_even       0.31  both  (~tie)        model headline: mean 0.32 / median 0.31
tomato_fruit    0.31  OBSERVATIONAL       ── the only rerun-driven one
```

`rain` correctly drops out of the danger zone (dispersion gated off); `tomato` correctly
*rises* on dispersion (it wobbles 15% about a botanical fact — real wobble that
pure-interventional would have missed). Both failure modes handled.

Built: `src/wobblelab/wobble.py` (`wobble_score` + `model_wobble`, tested,
`tests/test_wobble.py` 6/6), battery gained conservative `answer` labels (hotdog /
cereal / rain → null), and **scoring is a separate post-processing step**
(`experiments/score.py`) so we can re-score without re-running the model. Artifacts:
`results/wobble_scores_v02.{png,json}`.

## 2026-07-20 — cross-lingual probe (EN vs 中文)

A dedicated symmetric probe (`experiments/xlingual.py`): each prompt run in English and
Chinese N=150× each, disagreement `|p_yes(EN) − p_yes(ZH)|` with a Newcombe CI, plus
per-language accuracy from the `answer` labels. Battery = 6 culturally-invariant facts
(the clean signal) + 2 cultural items, one Western (hotdog), one Chinese (soy-milk-soup,
Kate's pick over wonton). Simplified for Qwen. `results/xlingual.{png,json}`.

The result was more nuanced than expected, and it corrected a prediction:

- **F-014 — the model is *good* at this.** Mean `|Δ|` on invariant facts is 0.10, and
  there are **zero majority accuracy gaps**: it gets 7-is-prime, 0-is-even, water-is-wet,
  sun-is-a-star, earth-is-round, and Pluto-is-not-a-planet correct in *both* languages.
  I'd bet we'd catch a right-in-EN/wrong-in-ZH flip; we didn't. Qwen being Chinese-origin,
  EN↔ZH is a genuinely fair, strong-on-both comparison, and it shows.
- **F-015 — except Pluto.** `|Δ|`=0.33: 95% "not a planet" in English, but only 62% in
  Chinese (majority still correct, but the grip is much looser). It's the single fact with
  a *recent, contested, English-discourse-heavy* history (the 2006 IAU reclassification),
  and that's exactly where the cross-lingual knowledge thins. A clean cross-lingual wobble.
- **F-016 — culture is language-bound.** Both cultural items swung hard (hotdog 0.32,
  soymilk 0.43), and **豆浆是汤吗 flips its majority across languages** — "no, soy milk
  isn't a soup" in English, "yes it is" in Chinese. That's not wobble (it's confounded,
  correctly labelled null), it's the model reflecting *different category conventions in
  each language*, which is a finding in its own right and a vindication of gating cultural
  items out of the clean measurement.

Net: the cleanest cross-lingual wobble signal is small-but-real (Pluto), the model is
otherwise cross-lingually solid on facts, and the cultural items behave as confounded-but-
interesting rather than as wobble. The design (invariant-facts-are-clean, cultural-items-
are-a-bias-look) held up.

## 2026-07-21 — benchmark wobble on MMLU (the original idea)

The application the whole project was aimed at: instead of a leaderboard's bare accuracy
number, report **accuracy + a CI + a wobble score**. Built the reusable MC machinery
(`src/wobblelab/benchmark.py`, tested — the option-permutation round-trip is the heart of
it) and the harness (`experiments/bench.py`). Also added `bootstrap_ci` to `stats.py`
(D-008), prompted by Kate's bootstrap question.

**Design.** For each item, place the correct answer at *every* option position
(distractors keep relative order) and rerun each. Option reordering is guaranteed
meaning-preserving, so it's a clean interventional axis with no paraphrase-quality
confound — and placing the answer at each position measures position bias directly.
Accuracy CI is **bootstrapped over items** (the correct resampling unit; reruns of one
item are correlated, so Wilson-over-trials lies).

**First run — MMLU:world_religions, 40 items × 4 orders × 5 reruns (4 min):**

- **F-017 — the number is a mirage.** Debiased accuracy 0.364, but that hides everything:
  accuracy by answer-position is **A 0.65 / B 0.17 / C 0.22 / D 0.42**, a **48-point swing**
  driven purely by *where the correct answer sits*. B-accuracy (0.17) is *below* the 0.25
  chance floor — the model actively avoids B. The mechanism is a brutal chosen-letter bias:
  **A 56% / B 8% / C 11% / D 25%.** The model mostly answers "A," so it looks competent
  only when the answer happens to be A. A leaderboard would print "36%"; the honest report
  is "36% ± 5, but really 65% if the answer's at A and 17% if it's at B."
- **F-018 — CI + reorder wobble.** Bootstrap-over-items CI [0.31, 0.42] is properly wider
  than naive Wilson-over-trials [0.33, 0.40]. ~29% of answers flip content under reordering.

This is exactly the pitch, demonstrated: the leaderboard number is nearly meaningless
without the wobble, and wobblelab surfaces the position bias, the reorder fragility, and
the honest CI in one shot. `results/bench_world_religions.{png,json}`. Next: TruthfulQA
MC1 (predict *higher* wobble — adversarial-by-construction), and more subjects/models.

## 2026-07-21 — harness wobble: the leaderboard number is a config choice, quantified

F-019 left an open charge: our 36% and the official 48.5 are "not directly comparable"
because the harnesses differ (generation + zero-shot + Q8 vs log-likelihood + few-shot +
full precision). That's an argument, not a measurement. This turns it into a measurement:
hold the 40 items fixed and *sweep the harness* — scoring (gen vs ll) × shots (0 vs 5) —
so the only thing moving is the eval convention. → **D-009.**

**Building the ll scorer meant answering Kate's question first:** does ollama return the
full distribution or just top-N? Probed it — **top-N only, hard-capped at 20** (50+ →
HTTP 400), same contract as OpenAI. But on a lettered-MC first token the softmax is so
peaked the top-20 holds **99.9%** of the mass, and the four candidate letters sit at the
very top (ranks 1,2,3,5 on a test item). So first-token argmax over the returned letter
logprobs *is* the official MMLU scoring method, and it's effectively exact here. The
native `/api/chat` endpoint respects `think:false` cleanly; the OpenAI-compat one leaked
the thinking trace, so native it is. Instrumented the harness to count coverage anyway —
it came back **0.95 (0-shot) / 1.00 (5-shot)**, the few-shot exemplars anchoring the
format enough to pull every letter into the top-20. → **F-020.**

**The result (579s, ~1,440 calls). `results/harness_world_religions.{png,json}`.**
Cross-check passed first: `gen/0-shot` debiased = **0.364**, identical to F-017 — same
seed, same items, reproducible instrument.

- **F-021 — a 20-point spread from harness alone.** Natural-position accuracy (what a
  leaderboard prints) runs **0.30 → 0.50** across the four cells. The *same model on the
  same items* is "30%" or "50%" depending only on eval convention. Scoring lifts +0.08
  (ll > gen: reading the distribution beats sampling a letter and paying the output-bias
  tax); **few-shot *costs* −0.12** — 5 exemplars *hurt* this 0.8B, robustly under both
  scorings. So the F-019 gap wasn't a capability gap: measured the official way (ll,
  0-shot), our crude Q8 40-item slice brackets the official 48.5 at **0.50 [0.35, 0.65]**.
  Once you match the harness, the mystery mostly evaporates — and what's left is that
  "the accuracy" was never a point, it's a range you pick a number from.
- **F-022 — ll doesn't cure position bias, it sharpens it.** I predicted log-likelihood
  scoring would flatten the brutal position swing (F-017's 48 points). It *widened* it:
  swing **0.48 → 0.70**, chosen-A share **56% → 67%**, even as debiased accuracy rose
  0.36 → 0.48. The mechanism is clean: temp-1 sampling occasionally samples off the top
  letter, blurring the model's positional preference toward the middle; argmax commits to
  the top logit every time, so ll is simultaneously *more accurate* and *more polarized*.
  The bias is in the weights, not the readout — the cleaner method exposes it undiluted.
  A better finding than the one I bet on: you cannot scoring-method your way out of a
  model that reaches for "A."

Net: harness wobble is real and large (20 pts here), it decomposes cleanly (scoring helps,
few-shot hurts, position bias is model-intrinsic), and the leaderboard's single number is
demonstrably one draw from a harness-dependent range. Next: TruthfulQA MC1 (still queued,
predict higher wobble); then config-as-wobble (D-003 → data).

## 2026-07-21 — config wobble: D-003's argument, turned into (a small) number

The third leg. D-003 held that the sampling config is itself a wobble source; F-019
sharpened it (ollama's Modelfile *is* Qwen's recommended config). Time to measure it, not
assert it. Two arms on the same 8-prompt battery: controlled full-softmax vs Qwen's
recommended top_p .95 / top_k 20 / presence 1.5, every other knob neutral in both so the
contrast isolates exactly Qwen's three deliberate choices. **Pre-registered a prediction**
in the journal before running: small effect, because top_p/top_k barely truncate a two-way
split and presence_penalty can't reach the first token.

**Result (`results/config_ab.{png,json}`, 487s): the prediction held, almost to the
decimal.** → **F-023.** Mean dispersion moved 0.141 → 0.140 (Δ −0.001); mean |Δp_yes|
0.019; **only pluto** shifted confidently, and by exactly the predicted mechanism —
truncation clipping its residual 4.7% minority mass down to a flat 0.0, i.e. the shipped
config makes the model look *marginally more sure* on the one prompt where it wasn't quite.

Two things worth keeping. **(1)** The prompt where config bit is `pluto_planet` — the same
fact that's the cross-lingual outlier (F-015) and the partial-artifact knife-edge (F-009).
Config wobble concentrates where the model's grip is *already loosest*: solid facts have no
tail to clip, genuine 50/50s (rain, dispersion 0.42) keep both answers far above the
truncation floor, and only the barely-decided prompt has a thin minority for top_p to cut.
**(2)** This doesn't weaken D-003, it *scopes* it: config wobble is real but its magnitude
tracks the width of the output tail. On binary yes/no it's a rounding error; on the
benchmark's chosen-letter distribution (4 letters + format tokens) truncation has real tail
to bite, and would push the `gen` path toward the sharper, more-A-biased argmax we measured
as `ll` (F-022). The natural follow-up is config-A/B on the *benchmark*, not the battery —
that's where the three legs (benchmark / harness / config wobble) converge.

Three legs now stand: **which items + scoring** (F-017), **which harness** (F-021/22),
**which config** (F-023) — a leaderboard number is a choice at every level, each now
quantified on the same model.

---

## 2026-07-23 — going faster: concurrency, an OpenAI adapter, and a backend bake-off

The single-stream client made a rented A100 pointless (barely faster than the laptop). Two
changes fixed it, and the same knob helps locally. → **D-010.** `evaluate(concurrency=N)`
thread-pools the provider calls (aggregation stays pure/order-deterministic, so the numbers
are byte-identical to sequential — a test asserts it), and `OpenAICompatibleProvider` reaches
any `/v1` endpoint. The concurrency alone bought **~4.7×** on the *existing* ollama with no
config change — ollama already batches concurrent requests up to its default parallelism.

Then a real bake-off, same model on three backends (Qwen3-0.6B; llama.cpp can't load the
qwen3.5 GGUF — a rope-config bug — so we dropped to the 0.6B, which also gave 8-bit/Q4 parity
via the *identical* ollama GGUF blob feeding both ollama and llama.cpp). → **F-024.**

- **llama.cpp** (`--parallel 8`) wins: 11.3 → **28 req/s**, ~1.9× ollama single-stream, and it
  works through our adapter *with* logprobs (so `ll` scoring survives).
- **ollama**: 6.0 → ~18, auto-batches to ~4. The zero-setup default.
- **mlx-lm**: 6.1 → ~6, **flat** — `mlx_lm.server` serializes. Apple-native but not for batched
  eval. `mlx_parallm` (Will Brown) fixes the batching via `batch_generate`, but it's Python-only
  (no HTTP → doesn't fit the adapter), generation-only (no logprobs → loses our clean `ll`
  path), and Qwen-unverified. Interesting, not worth it over llama.cpp yet.

A bonus: the `OpenAICompatibleProvider` drove both llama.cpp and mlx `/v1` first try — the same
adapter that will point at a vLLM pod. `experiments/backends.py` reproduces the bake-off.

## 2026-07-23 — RENAME IN PROGRESS: squishlab → wobblelab (branch `rename-wobblelab`)

Full rename (project + package + metric vocabulary). "wobble" is the more accessible umbrella
term; the squish/wobble distinction only matters in methodology prose, not the brand.

**DONE on branch `rename-wobblelab` (uncommitted):**
- `git mv src/squishlab → src/wobblelab`, `git mv squish.py → wobble.py`.
- Three-case content replace (`squish→wobble`, `Squish→Wobble`, `SQUISH→WOBBLE`) across 43
  text files (src, tests, experiments, docs, README, pyproject, .github, pitch/). Zero
  "squish" left outside results/.
- Renamed files whose *names* held squish: `test_squish.py→test_wobble.py`, results plots
  (`wobble_plane_*.png`, `wobble_scores_*`), pitch SVGs (`wobblelab_report_card_*.svg`), pitch
  docs (`wobblelab-pitch*.{html,md}`). Rewrote keys inside results/*.json. Fixed prose
  artifacts (`wobbley→wobbly`, `wobbleing→wobbling`).
- Metric API renamed: `wobble_score`, `wobble_factor`, `wobble_by_kind`, `interventional_wobble`,
  `observational_wobble`, `model_wobble`, "wobble plane".

**REMAINING (immediate, to finish the branch):**
1. Reinstall: `pip uninstall -y squishlab && pip install -e ".[dev,bench]"` in the pyenv venv
   (still named `squishlab`; keep using `PYENV_VERSION=squishlab`) so `import wobblelab` resolves.
2. `ruff format . && ruff check .` then `pytest` — must stay green (60 tests).
3. Bump `pyproject.toml` version `0.0.1 → 0.1.0`.
4. Commit on the branch; merge to `main`.
5. NOTE: result/pitch **PNGs still show "SQUISH" baked into the pixels** (can't sed an image) —
   regenerate charts later to fix; filenames + JSON already renamed.

**Kate's account-level TODOs (not code):** claim `wobblelab` on PyPI (it's free; 404 confirmed)
+ configure Trusted Publishing pending-publisher for it; leave/tombstone `squishlab` (published
v0.0.1, don't delete); rename the GitHub repo (auto-redirects); optionally rename the pyenv venv.

**Other pending (post-rename):**
- Full MMLU-Pro run was **killed ~25% into pass 1** (no checkpointing → lost). Redo as a ~2–3k
  subset (~30–45 min on llama.cpp) or on a rented GPU. My ~1.5h estimate was wrong — MMLU-Pro's
  long 10-option prompts run ~4× slower (~10 calls/s, prefill-bound), so full 12k ≈ 3–4h local.
- `Benchmark` **ABC** + concrete `ConfigMCQBenchmark` (config-driven flat mapping) + subclass
  for nested schemas (TruthfulQA) + a `BENCHMARKS` registry; port the loaders onto it (thin
  `load_*` shims). Chosen ABC over Protocol: we own the impls + want shared `load()` + enforce
  the blanks. Do this in wobblelab, after the rename lands.
- vLLM pod adapter for the real-model spike (adapter exists; needs pod config + IP clearance).
  Backend bake-off (F-024): llama.cpp `--parallel` is the fast local pick (~28 req/s, keeps ll).

## Open questions / backlog

- **A `BatchProvider` seam (stub).** Offline eval has the whole prompt set upfront, so a
  provider that takes a *batch* — `ask_batch(prompts, seeds) -> [str]` — fits better than
  thread-pooling one HTTP call per prompt, and it's the clean plug for in-process static
  batchers (`mlx_parallm`) and vLLM's batch API (no per-request HTTP overhead). The harness
  would detect batch-capable providers and hand them batches; single-call providers keep the
  thread-pool path. Only worth building if an MLX (or batch-API) backend beats llama.cpp in a
  head-to-head — until then llama.cpp + `concurrency` covers it (F-024, D-010).
- **vLLM adapter for the real-model spike.** `OpenAICompatibleProvider` already speaks vLLM's
  `/v1`; the remaining work is a pod/serverless config + the IP clearance to run on company
  vs personal resources. Est. ~$1–15 for a full-corpus 7B/70B run (rented A100/H100).
- ~~**Match the official harness to measure harness-wobble directly.**~~ **Done**
  (D-009, F-020/21/22, `experiments/harness.py`): ll + few-shot eval mode built, 2×2
  harness sweep run, 20-pt spread quantified and decomposed. Remaining threads it opened:
  **(a)** run the sweep on the *full* subject (not a 40-item slice) and on more subjects,
  to see if "few-shot hurts" and the scoring lift hold in aggregate; **(b)** 5-shot
  position profiles (we only profiled 0-shot) — does few-shot change the position bias?;
  **(c)** TruthfulQA MC1 still queued (predict higher wobble, adversarial-by-construction;
  variable option counts need the loader generalized past a fixed 4).
- ~~**Config as a wobble axis, quantified.**~~ **Done** (F-023, `experiments/config_ab.py`):
  measured on the binary battery — small and localized (Δ mean dispersion −0.001; only the
  near-unanimous pluto shifts confidently). Follow-up that would show the *large* version:
  run the same controlled-vs-Qwen-recommended A/B on the **MC benchmark**, where the wider
  chosen-letter tail gives truncation something to bite (predict it nudges `gen` toward the
  A-biased argmax, F-022).
- **Kernel/backend nondeterminism.** Even with identical explicit config, different
  backends/kernels diverge (the batch-invariance issue from the Thinking Machines
  piece). Probe: does `temperature 0` give bit-identical output across reruns here?
- **Multiple-comparisons on the worst-case margin.** As K grows, more chances for a
  false-significant shift. If the taxonomy scales past ~10 types, use 99% CIs or
  Bonferroni, or report a high quantile (p90) instead of the strict max.
- **Paraphrase validation.** Worst-case + a sloppy paraphrase = false wobble. Consider
  a cheap check that paraphrases are genuinely meaning-preserving (embedding distance,
  or a second model's agreement) before trusting a hot-spot.
- **The model headline.** Not yet built — a decomposable wobble factor across the whole
  battery (central tendency + worst-offenders + per-type breakdown), per Kate's
  "headline number + factors + hot-spots" design.
