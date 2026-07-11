# Soft-world causal graph plan (P3) — 2026-07-11

**Status:** Scoped, small increment implemented and verified. No large causal-discovery work in this pass.
**Priority:** P3 in `docs/reports/DEV_PLAN_2026-07-11.md:153` ("Paper V soft-world / causal graph maturity — next framework beyond 7-stat profile; targets κ 0.90–0.93; independent, longer horizon"). Also referenced at `DEV_PLAN_2026-07-11.md:138,142`: "Paper V causal graph beyond stub (`native/include/cypha/intelligence/causal_graph.hpp:13`)" and listed among "known stub functions."
**Relationship to the parallel P0 work:** independent. This plan and its implementation do not touch `native/build_math`, `bench/BASELINE_LOCK.json`, or any overnight orchestration script. All build/verification in this pass used a fresh scratch directory, `native/build_softworld`. Files also avoided per this task's constraints: `native/src/rpsm/*`, `native/src/intelligence/measurers.cpp`, `cyphalm_bench_native.cpp`'s `--lstm-hidden` path, kernel-LLR files.

---

## 1. The claim being scoped

Paper V (`docs/research/intelligence_stats/soft_world_paper5.md`) describes a "soft world model" — a causal-discovery-based simulation engine that closes the "grounded causality" gap left open by Paper IV, pushing κ from 0.89 (Paper IV) to 0.90–0.93 (Paper V, Phase 3/4 maturation, §5.1, §6.1). The repo's own implementation-status doc flags the current code as a stub:

> "Paper V causal graph stub | `causal_graph.hpp/cpp` | papers + `/intelligence/report`" — `docs/reports/INTELLIGENCE_STATS_IMPLEMENTATION.md:37`

> "Paper V causal graph beyond stub (`native/include/cypha/intelligence/causal_graph.hpp:13`)" — `docs/reports/DEV_PLAN_2026-07-11.md:138`

The header comment itself (before this pass) read: `/// Paper V causal graph stub: tracks P-space edges and soft-world maturation signals.` (`causal_graph.hpp:13`, original).

**This document's job:** determine precisely what "stub" means here (no-op? fixed/fake data? partial implementation?), cross-reference against Paper V's own Phase 1–5 roadmap to find which phase the current code corresponds to, identify the one specific gap that matters most (real vs. hardcoded edge weights), and either fix that one gap cheaply or document why it can't be fixed cheaply — before any larger causal-discovery investment is considered.

---

## 2. What "stub" concretely means (read of the actual code, before this pass)

`native/include/cypha/intelligence/causal_graph.hpp` + `native/src/intelligence/causal_graph.cpp` (both read in full) turned out to **not** be a no-op and **not** returning fake/fixed data. `CausalGraphMonitor` is a real, working class:

- `edges()` returns a real, growing vector of `{from, to, weight}` triples.
- `to_json()` / `trajectory_json()` produce real structured JSON, wired into `/intelligence/simulation` (`intelligence_rest_routes.cpp:88-98`) and into every bench domain that calls `intelligence_profile_report_json` (`profile_from_model.cpp:222-224`, feeding `d18_intelligence_profile` and the `intelligence_profile.causal_graph` block visible in every `cyphalm_bench_native --intelligence-profile` run and every D17 overnight log).
- It embeds a `SoftWorldMonitor` (`soft_world_monitor.hpp/cpp`) that does genuine online Bayesian inference: two `NigStatisticState` posteriors (`world_model_nig_`, `query_quality_nig_`) updated via the same real Normal-Inverse-Gamma update rule used everywhere else in the profiler stack (`nig_statistic_state.cpp:21-35`). `maturation_level()` is a real running NIG mean of observed resolution values, not a placeholder.

**Confirmed from a live production log** (`bench/results/production_overnight_20260628_170701.log:33-224`, June 28 D17 300k run) — the causal graph *is* already producing structured output with real edges today, exactly as the task's context described:

```34:65:bench/results/production_overnight_20260628_170701.log
"causal_graph": {
  "edges": [
    { "from": "alpha", "to": "calibration", "weight": 0.8361241973655229 },
    { "from": "tau", "to": "r_eu", "weight": 0.20136131980732883 },
    { "from": "query", "to": "r_eu", "weight": 0.22499999999999998 },
    { "from": "simulation", "to": "world_model", "weight": 0.22499999999999998 },
    { "from": "world_model", "to": "maturation", "weight": 0.15 },
    { "from": "maturation", "to": "tau", "weight": 0.06081480861243095 },
    ...
```

So the framing "stub == does nothing / returns fake data" is **wrong**. The correct, narrower framing is: **the edge *topology* is fixed by hand (six hardcoded `record_edge(...)` call sites, always the same six node-pairs, never discovered from data), and — this is the part that actually matters — two of those six edge *weights* (`alpha->calibration`, `tau->r_eu`) were computed by a **fixed deterministic formula of the current observation alone**, not fit, learned, or estimated from any accumulated statistical relationship between the two named variables.**

### 2.1 The two weight formulas, verbatim (before this pass)

```20:33:native/src/intelligence/causal_graph.cpp (original)
void CausalGraphMonitor::observe_profile(const ProfileObservation& obs) {
  ...
  record_edge("alpha", "calibration", clamp01(1.0 - std::abs(obs.alpha - 0.5)));
  record_edge("tau", "r_eu", clamp01(obs.tau * obs.r_eu));
}
```

Two concrete problems, found by direct inspection:

1. **`alpha->calibration` never reads `obs.calibration`.** `ProfileObservation` has a real `calibration` field (`intelligence_profiler.hpp:19`), populated from the profiler's actual measured calibration statistic (`profile_from_model.cpp:100`, `mean_observation()`). The edge that is supposed to represent the causal/associative link between `alpha` and `calibration` computes a number using **only** `alpha`, ignoring the second named variable entirely. This is not "a simple heuristic" so much as a formula that doesn't test the relationship it claims to report.
2. **`tau->r_eu` is the instantaneous product of the current sample**, not a relationship estimated across observations. A product of two numbers from one time point looks like a plausible weight (bounded, data-dependent, changes call to call) but encodes no actual claim about whether `tau` and `r_eu` co-vary — it would produce the exact same "confident" nonzero number on pure noise, on a single unrepeated sample, or on data where `tau` and `r_eu` are in fact unrelated.

The other four edges (`query->r_eu`, `simulation->world_model`, `world_model->maturation`, `maturation->tau`) are **not** in the same category — `query->r_eu` and `simulation->world_model` report a directly measured delta (`r_eu_before - r_eu_after`, `resolution`) that is the literal definition of the quantity being named, and `world_model->maturation` reports the real `SoftWorldMonitor` NIG mean. `maturation->tau` (`obs.tau * soft_world_.maturation_level()`) has the same "instantaneous product, not fitted" issue as `tau->r_eu` but was left out of scope for this pass (see §7).

### 2.2 Two distinct invocation paths, with different real-world implications

Reading every call site (`Grep` across the repo) found **two materially different usage patterns**, which the log evidence and the task's framing conflate:

| Call site | Pattern | Effect on the weight-computation gap |
|---|---|---|
| `native/apps/cypha_rest.cpp:73,879,975` | `g_causal_graph_monitor` is a **persistent, process-lifetime singleton**. `observe_profile()` is called on every `/update`/infer request; `simulation_step()` on every self-correction event. | Given enough real traffic, this instance genuinely accumulates a long history of `(alpha, calibration)` and `(tau, r_eu)` pairs — a real correlation *could* be estimated from it. The old fixed-formula code never did this; it recomputed a fresh number from the current sample on every call and threw the history away. |
| `native/src/intelligence/profile_from_model.cpp:222-224` (feeds `d18_intelligence_profile`, `cypha_intelligence_bench`, every bench-domain / overnight-log `intelligence_profile.causal_graph` block) | A **fresh `CausalGraphMonitor` is constructed inside `intelligence_profile_report_json()` on every call** and immediately fed a single `ProfileObservation` (the profiler's *mean* over the eval batch) via `run_simulation_trajectory(4, obs)`. | There is, by construction, only ever **one** real observation available to this instance — the "4 steps" are a synthetic decay loop over that one observation (see §2.3), not four independent samples. Any formula that claims to estimate a two-variable relationship from a single data point is statistically unfounded here, no matter how the formula is written. |

This distinction matters directly for §4 below (what the fix can and can't do in each path) and is why the production-log numbers looked "real" — they *are* real observed `alpha`/`tau`/`r_eu` values, run through a formula, not literal fabrication — while still not being an estimated relationship.

### 2.3 The four/`n_steps` "trajectory" is synthetic, not re-observed data

```54:70:native/src/intelligence/causal_graph.cpp (original)
void CausalGraphMonitor::run_simulation_trajectory(int n_steps, const ProfileObservation& obs, double resolution_scale) {
  ...
  observe_profile(obs);
  double r_eu = std::clamp(obs.r_eu, 0.1, 1.0);
  const double scale = std::max(0.01, resolution_scale);
  for (int i = 0; i < n_steps; ++i) {
    const double decay = scale * (0.75 + 0.05 * static_cast<double>(i));
    const double r_after = std::max(0.05, r_eu - decay);
    ...
```

`profile_from_model.cpp:223` calls this with a hardcoded `n_steps=4` and default `resolution_scale=0.3`. The four "simulation steps" that show up in every bench-domain causal-graph JSON block are a **fixed algebraic decay curve** (`decay = 0.3*(0.75 + 0.05*i)`) applied to the single observed `r_eu`, not four independent observations of the model doing anything. This is a separate, larger gap from the one this pass fixes (§7, out of scope) — it means `world_model->maturation` and `simulation->world_model` in the bench-domain path are also not measuring anything about the actual model, just replaying a canned shape. It is flagged here because it directly explains why, even after this pass's fix, the `alpha_calibration_n`/`tau_r_eu_n` counters in the bench-domain path stay at `1` (§5) — `observe_profile` is only called once per `intelligence_profile_report_json()` call.

---

## 3. Cross-reference against Paper V's own Phase 1–5 roadmap

`docs/research/intelligence_stats/soft_world_paper5.md` §8 (`:662-723`, read in full) gives an explicit five-phase roadmap:

| Paper V phase | Paper's own spec | Codebase status |
|---|---|---|
| **Phase 1** (weeks 1–6): Paper IV `ProfileGuidedCyphaCell` + self-correcting loop | — | Shipped (`docs/reports/INTELLIGENCE_STATS_IMPLEMENTATION.md` Phase 1–5 tables; `profile_guided_loss.hpp/cpp`, `self_correcting_infer.hpp/cpp`) |
| **Phase 2** (weeks 7–9): uncertainty-driven webscraper, `MinimalAcquisitionLayer` (`:669-683`) | Query generation from epistemic uncertainty, real search API acquisition | **Not implemented.** No webscraper, no search-API integration, anywhere in `native/`. `record_acquisition(r_eu_before, r_eu_after)` exists but is always called with *already-known* before/after values supplied by the caller — there is no code that decides *what to query* or *fetches new data* in response to high `r_eu`. |
| **Phase 3** (weeks 10–14): `MinimalSoftWorld` — causal discovery per domain, NIG-parameterised edges, `simulate(intervention, domain)` (`:687-706`) | `CausalDiscovery.discover()`: PC-algorithm-style conditional-independence skeleton discovery + v-structure orientation (§3.2, `:237-297`); per-edge NIG state fit from data; `do`-calculus intervention/simulation | **Partially implemented, and this is where "stub" applies precisely.** `SoftWorldMonitor`'s two aggregate NIG states (`world_model_nig_`, `query_quality_nig_`) are a genuine (if very coarse) realisation of "NIG parameterisation of world-model quality" — but there is no per-*edge* NIG state, no conditional-independence testing, no skeleton discovery, no v-structure orientation, and (before this pass) no statistical relationship of any kind behind the two weights this document targets. `CausalGraphMonitor` has a fixed topology chosen by the six `record_edge()` call sites in the code, not a topology discovered from `conditional_independence_test`-style analysis as Paper V specifies. There is also no `do_intervention`/`counterfactual`/`twin_network` — Pearl's rungs 2 and 3 (§3.4) are entirely absent. |
| **Phase 4** (weeks 15–18): full integration, measure end-to-end r_eu reduction, effective τ, κ, maturation trajectory (`:708-715`) | — | Not attempted; blocked on Phase 2/3. |
| **Phase 5** (weeks 19–24): domain validation on 3 diverse domains + 1 held-out (`:717-723`) | — | Not attempted; blocked on Phase 2/4. |

**Conclusion: the current code sits partway through Phase 3, specifically implementing the "NIG parameterisation of overall world-model quality" sub-piece of Phase 3 for real, while completely skipping Phase 3's actual headline mechanism (causal discovery / conditional-independence testing) and all of Phase 2, 4, and 5.** The task's framing of "stub" is accurate for the *edge-weight-computation* sub-piece specifically (which is what this pass targets), not for the whole `causal_graph.hpp/soft_world_monitor.hpp` file (which is a real, if narrow, partial implementation).

Cross-checked against `docs/research/intelligence_stats/cypha_self_correcting_paper4.md:707` ("The system models correlations and temporal dependencies but not interventional causal structure... This is a limitation of the architecture, not the profile framework") — Paper IV itself already flags "grounded causality" as unsolved before Paper V exists; Paper V's whole contribution is supposed to be closing that gap. The current code has not yet closed it; §4 identifies the one narrow piece of it that was cheap to move from "not even correlational" to "genuinely correlational."

---

## 4. The actual gap (answering the task's central question)

**Is the edge-weight computation using a real learned/inferred mechanism, or a hardcoded/heuristic formula that doesn't adapt from data?**

Before this pass: **hardcoded/heuristic, for the two weights this document targets** (`alpha->calibration`, `tau->r_eu`). Evidence:

- The formulas are pure functions of the *current* `ProfileObservation` — calling `observe_profile` twice with the same `obs` produces the exact same weight both times (deterministic replay), and calling it many times with *unrelated* `alpha`/`calibration` pairs produces exactly the same formula-shaped output as calling it with genuinely correlated pairs. There is no way, from the weight alone, to tell "real relationship" from "coincidence" from "no relationship at all" — the formula cannot express "I don't have enough data to know."
- `alpha->calibration` additionally never reads one of its own two named variables (`obs.calibration`), which is a correctness gap independent of the "is it learned" question — it wasn't computing *any* version of the relationship it claims to name.

This is not a full causal-discovery gap (that would require implementing conditional-independence testing and structure learning per Paper V §3.2 — out of scope, see §7) — it is a narrower, purely **weight-estimation** gap on an already-fixed topology, and that is exactly the kind of gap the task asked to check for ("is this a stub in the sense that matters ... vs. just an early/simple-but-real implementation").

---

## 5. The fix implemented in this pass

**Scope:** replace the two formula-based weights with weights estimated online from the real accumulated `(x, y)` history, using a small new `OnlineCorrelation` (Welford-style running Pearson correlation) class. No topology change, no causal-discovery algorithm, no new file — added to the existing `causal_graph.hpp`/`.cpp`. Total diff: ~70 new lines across the header/impl, plus one new CTest function (~65 lines).

### 5.1 What changed

`native/include/cypha/intelligence/causal_graph.hpp`:
- New `class OnlineCorrelation`: `update(x, y)` (Welford-style running mean/covariance/variance), `correlation()` (Pearson coefficient in `[-1, 1]`, exactly `0.0` when `n < 2` or either variable's variance is ~0 — i.e. explicitly reports "not enough data to estimate a relationship" rather than guessing).
- `CausalGraphMonitor` gained two private `OnlineCorrelation` members (`alpha_calibration_corr_`, `tau_r_eu_corr_`) plus read-only accessors for testing/diagnostics (`alpha_calibration_correlation()`, `alpha_calibration_n()`, `tau_r_eu_correlation()`, `tau_r_eu_n()`).

`native/src/intelligence/causal_graph.cpp`:
- `observe_profile()`: `record_edge("alpha", "calibration", clamp01(1.0 - std::abs(obs.alpha - 0.5)))` → `alpha_calibration_corr_.update(obs.alpha, obs.calibration); record_edge("alpha", "calibration", clamp01(std::abs(alpha_calibration_corr_.correlation())))`. Same pattern for `tau->r_eu` against `tau_r_eu_corr_`.
- `to_json()`: added an `edge_estimation` object (`alpha_calibration_correlation`, `alpha_calibration_n`, `tau_r_eu_correlation`, `tau_r_eu_n`) so any consumer of `/intelligence/simulation` or a bench-domain JSON can see directly how many real observations back each estimated edge, rather than having to infer it.
- Doc comment on `CausalEdge` (the flagged `causal_graph.hpp:13`) rewritten to state precisely what is and isn't real: fixed topology (not discovered), two weights now correlation-estimated from history, remaining weights are direct measured deltas/NIG means (already real), no causal-discovery structure learning (linked to this document).

### 5.2 CTest added

`native/tools/intelligence_profiler_papers.cpp`: `test_paper_v_causal_graph_correlation_edges()` (registered in `main()`, runs under the existing `native_intelligence_profiler_papers` CTest), three cases:
1. **Single observation** → both edges must report weight `0.0` and `n == 1` (cannot estimate a correlation from one point; this is the direct behavioural contrast with the old formula, which always produced a nonzero number from one sample).
2. **12 synthetic observations**, `alpha`/`calibration` rising together and `tau`/`r_eu` moving in strict opposition → `alpha_calibration_correlation() > 0.95`, `tau_r_eu_correlation() < -0.95`, and both edge weights (which take `abs(correlation)`) `> 0.95` in the exported JSON.
3. **8 constant observations** (zero variance) → both correlations fall back to the documented `0.0`, not a divide-by-zero artefact.

### 5.3 Build and test results (`native/build_softworld`, fresh scratch build, GNU 13.2.0/Ninja)

```
ctest --test-dir native/build_softworld -R "native_intelligence|native_d18|native_d39|native_cyphalm_bench_intelligence_profile"
100% tests passed, 0 tests failed out of 6
  native_intelligence_profiler_smoke .......... Passed
  native_intelligence_profiler_papers .......... Passed   (includes the 3 new cases)
  native_intelligence_bench_smoke .............. Passed
  native_cyphalm_bench_intelligence_profile ... Passed
  native_d39_intelligence_monitor_smoke ....... Passed
  native_intelligence_lm_monitor_smoke ......... Passed
```

No existing test asserted a literal numeric value for the `alpha->calibration`/`tau->r_eu` weights (`test_paper_v_causal_graph` only checks `edges` presence, `step_count >= 5`, trajectory size, and `maturation_level >= 0`), so the fix is behaviourally additive — no existing assertion needed to change.

### 5.4 Measured effect on the reference-fixture bench path (`cypha_intelligence_bench`, `d18_intelligence_profile`)

Ran `native/build_softworld/cypha_intelligence_bench.exe` against the committed `fixtures/reference.cypha` + `fixtures/native_parity.bin` (deterministic, no `n_train` knob — this tool profiles a fixed reference model, not a training run; this is the correct low-cost validation fixture for this specific change per the task's "reference fixture" framing).

Observed profile mean: `alpha=0.25`, `calibration=0.6634330349119797`, `tau=0.5`, `r_eu=0.5624999999992187`.

| Edge | Old formula (computed by hand from the formula + these values) | New (this pass, measured) |
|---|---|---|
| `alpha -> calibration` | `1 - \|0.25 - 0.5\| = 0.75` | `0.0` (`alpha_calibration_n = 1`) |
| `tau -> r_eu` | `0.5 * 0.5625 = 0.28125` | `0.0` (`tau_r_eu_n = 1`) |

This is the expected and *correct* behaviour change for this specific call path, not a regression: `intelligence_profile_report_json()` constructs a fresh `CausalGraphMonitor` and feeds it exactly one observation (§2.2/§2.3), so there is genuinely no relationship that can be estimated yet, and the new code says so (`weight = 0.0`, `n = 1`) instead of manufacturing a confident-looking number from a single point. The old formula's `0.75`/`0.28125` were not measuring anything about a relationship either — they were just less obviously so.

`criticality_score_obs` (`0.8007521950442218`) and every `landscape_kappa.*` entry were **bit-identical before and after this change** — confirmed both by direct inspection of `profile_from_model.cpp:188-226` (landscape/criticality are computed from `obs` directly via `IntelligenceProfiler::criticality_score_for`/`landscape_reference`, *before* `CausalGraphMonitor` is even constructed, on the following lines) and by running the tool. **This is an important, non-obvious finding in itself: the causal graph is currently a downstream diagnostic side-channel with zero feedback into κ, landscape comparison, or `criticality_score` anywhere in the codebase.** No change to `causal_graph.cpp` — this fix or a much larger one — can move `kappa` today, because nothing reads the causal graph's output back into the profile-matrix/criticality-score computation. Closing that feedback loop (i.e. making κ actually depend on causal-graph fidelity, which is implicitly what Paper V's κ 0.90–0.93 estimate assumes once "grounded causality" is closed) is itself a distinct, larger piece of future work — see §7.

### 5.5 Where the fix *does* have a measurable, meaningful effect: the persistent REST path

The bench-domain path (§5.4) can only ever see `n=1` by construction (§2.2), so it cannot demonstrate the fix adapting over time. The **persistent** `g_causal_graph_monitor` in `cypha_rest.cpp` (§2.2 table, row 1) is the path where this matters in practice — every `/update`/self-correction request calls `observe_profile`/`simulation_step` on the *same* long-lived instance, so `alpha_calibration_n`/`tau_r_eu_n` grow with real traffic and the correlation genuinely converges toward whatever relationship actually exists in the served data. This is exactly what CTest case 2 (§5.2) verifies synthetically (12 observations, correlation converges to `>0.95`/`<-0.95` in the expected direction) since spinning up the live REST server with hours of real traffic was out of scope for this pass; the synthetic CTest is the correct low-cost stand-in for "does it actually adapt with more data," matching the task's request to measure the effect "on a reference fixture/bench run at n_train<=5000" in spirit (n=5000 observations is not meaningful here since each `observe_profile` call corresponds to one already-aggregated profiler-matrix mean, not one training step; 12 synthetic points is the right order of magnitude to demonstrate convergence behaviour for a correlation estimator, and is what the CTest uses).

---

## 6. Files changed in this pass

| File | Change |
|---|---|
| `native/include/cypha/intelligence/causal_graph.hpp` | Added `OnlineCorrelation` class; added two private correlation-tracker members + 4 read-only accessors to `CausalGraphMonitor`; rewrote the `CausalEdge` doc comment to state precisely what's real vs. not |
| `native/src/intelligence/causal_graph.cpp` | Implemented `OnlineCorrelation::update`/`correlation`; replaced the two formula-based `record_edge` calls in `observe_profile` with correlation-estimated ones; added `edge_estimation` block to `to_json()` |
| `native/tools/intelligence_profiler_papers.cpp` | Added `test_paper_v_causal_graph_correlation_edges()` (3 cases), registered in `main()` |

No changes to `SoftWorldMonitor`, `soft_world_monitor.hpp/cpp`, `intelligence_rest_routes.cpp`, `profile_from_model.cpp`, `bench_domains.cpp`, or any file under `native/build_math`, `native/src/rpsm/`, `native/src/intelligence/measurers.cpp`, or the kernel-LLR/`--lstm-hidden` paths, per this task's constraints.

---

## 7. Explicitly out of scope for this pass (and why)

| Candidate next step | Why it's out of scope here |
|---|---|
| **Conditional-independence-based structure learning** (Paper V §3.2's `CausalDiscovery.discover()` — PC-algorithm skeleton discovery + v-structure orientation) | This is the actual "full causal discovery algorithm" the task said not to attempt. It requires: choosing a variable set beyond the 7 profile statistics, an independence-test implementation (partial correlation or a nonparametric test), a search over conditioning sets, edge orientation rules, and a decision about what "removing an edge" means for a graph whose topology currently has exactly one caller-specified shape. Multi-week effort at minimum; the paper itself allocates 5 weeks (Phase 3, `:687`) to this specifically. |
| **Per-edge NIG parameterisation** (Paper V §3.2 step 3: "for each oriented edge X→Y, fit a NIG distribution over the causal strength") | Natural follow-on to the correlation fix in this pass (replace the scalar `OnlineCorrelation` with a proper NIG posterior over the correlation/regression-slope estimate, giving the edge weight both a mean *and* an epistemic-uncertainty band, consistent with how every other statistic in this codebase is represented). Deliberately deferred: it's a real design decision (what's the observation model? correlation coefficient doesn't have an obvious conjugate NIG form the way a per-token statistic does) rather than a mechanical port, and the task asked for "well under a few hundred lines" — this would need its own scoping pass. |
| **`maturation->tau` weight** (`obs.tau * soft_world_.maturation_level()`) | Same "instantaneous product, not fitted" issue as the two edges fixed in this pass. Left out because `maturation_level()` is itself a running NIG mean already (not a raw per-call sample), so a `tau`-vs-`maturation_level` correlation would need to correctly handle the fact that `maturation_level()` barely changes call-to-call relative to `tau` — a different statistical situation from the two i.i.d.-ish variables fixed here, worth its own short pass rather than bundling in. |
| **Wiring the causal graph into `kappa`/`criticality_score`** (§5.4's finding) | This is a bigger, cross-cutting design decision (does κ get an eighth axis? does an existing axis like `L` or `C` get a causal-graph-derived adjustment term?) that touches `IntelligenceProfiler::criticality_score_for` and the whole profile-matrix contract used throughout the bench suite and `BASELINE_LOCK.json`-adjacent validation domains (d18, d39, etc.). Not attempted; flagged as the highest-leverage next step if the goal is actually moving κ toward Paper V's 0.90–0.93 target, since right now *no* causal-graph fidelity improvement (this pass's or a much larger one) can move κ at all. |
| **The Phase 2 data-acquisition layer** (webscraper, `MinimalAcquisitionLayer`) | Entirely unimplemented (§3); a genuinely new subsystem (network I/O, query generation, a search API integration) with its own security/dependency/config surface. Out of scope for a "small, well-defined, low-risk" pass by any reading of the task. |
| **Fixing the synthetic 4-step decay trajectory** (§2.3) in the bench-domain path | Would require deciding what a "real" multi-step trajectory means for a single aggregated profiler observation (there is no natural sequence of 4 independent samples available at that call site without restructuring `intelligence_profile_report_json`'s caller to pass in per-batch, not per-eval-mean, observations) — a real but separate design question from the weight-estimation gap this pass targeted. |

---

## 8. Recommended next-phase scope and effort estimate

Ranked by leverage-per-effort, for whoever picks up P3 next:

1. **(Half-day, highest leverage) Wire `edge_estimation`/edge weights into the REST persistent-monitor lifecycle more visibly** — e.g. expose `alpha_calibration_n`/`tau_r_eu_n` in the existing `/intelligence/simulation` payload's top-level (already done via `to_json()`, §5.1) and add a CTest or manual-verification note confirming it grows across repeated `/update` calls against a live `cypha_rest` process. Mechanical, no new algorithm, closes the loop on "does this fix actually do something in the one place it has enough data to matter."
2. **(2–3 days, medium leverage) Per-edge NIG parameterisation** (§7, row 2) — replaces the scalar correlation with a proper epistemic/aleatoric-decomposed causal-strength estimate, consistent with the rest of the profile framework's presentation. Needs a short design note first (what's the observation model for a running correlation coefficient's own uncertainty — e.g. Fisher z-transform + NIG on the z-scale is the standard approach and would fit this codebase's existing `NigStatisticState` machinery directly).
3. **(1–2 weeks, high leverage but high uncertainty) Wire causal-graph fidelity into `kappa`** (§7, row 4) — the highest-leverage item for actually moving the number Paper V cares about, but requires a design decision (new axis vs. adjustment term) that should be scoped as its own short planning pass (same rigour as this document) before implementation, not decided inline here.
4. **(multi-week, explicitly out of scope per the task) Real conditional-independence-based structure learning** (§7, row 1) — this is the item that would make "causal graph" mean what Paper V's Phase 3 actually specifies (a discovered, not hand-specified, topology). Should not be attempted without a dedicated scoping pass of its own; this document's §3/§4 are the starting point for that pass when it's prioritised.

**Bottom line for this pass:** the specific, narrow gap the task asked to find — hardcoded-formula edge weights masquerading as data-dependent ones, with one of them not even reading both of its named variables — is fixed, tested, and measured. It is a real (if small) improvement to the honesty of the diagnostic signal (a formula that can now say "not enough data" instead of guessing, and that genuinely converges with real history in the one place — the persistent REST monitor — where enough history exists). It does **not** move κ today, because the causal graph doesn't feed back into κ anywhere in the current architecture — that disconnection, not the weight-formula gap, is now the correctly-identified next bottleneck if the goal is Paper V's stated κ target.

---

## Appendix: commands reference

```powershell
# One-time: configure a separate scratch build (does not touch native/build_math)
cmake -S native -B native/build_softworld -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build the targets touched/exercised by this pass
cmake --build native/build_softworld --target intelligence_profiler_papers cypha_intelligence_bench cyphalm_bench_native cypha_bench_run intelligence_profiler_smoke intelligence_lm_monitor_smoke

# Full CTest coverage for this change
ctest --test-dir native/build_softworld -R "native_intelligence|native_d18|native_d39|native_cyphalm_bench_intelligence_profile" --output-on-failure

# Reference-fixture measurement used in §5.4 (deterministic, no n_train knob)
native/build_softworld/cypha_intelligence_bench.exe --out <scratch path>.json
# then inspect the "causal_graph" and "edge_estimation" blocks, and diff
# "criticality_score_obs"/"landscape_kappa" against a pre-change run to confirm no coupling.
```

### Files read/cited in this scoping pass

- `native/include/cypha/intelligence/causal_graph.hpp`, `native/src/intelligence/causal_graph.cpp` (full read, before and after this pass's edits)
- `native/include/cypha/intelligence/soft_world_monitor.hpp`, `native/src/intelligence/soft_world_monitor.cpp` (full read)
- `native/include/cypha/intelligence/nig_statistic_state.hpp`, `native/src/intelligence/nig_statistic_state.cpp` (full read)
- `native/include/cypha/intelligence/intelligence_profiler.hpp` (`ProfileObservation` fields)
- `native/src/intelligence/profile_from_model.cpp` (full read — `intelligence_profile_report_json`, `mean_observation`, causal-graph call site)
- `native/apps/intelligence_rest_routes.cpp`, `native/apps/cypha_rest.cpp` (persistent `g_causal_graph_monitor` wiring)
- `native/tools/intelligence_profiler_papers.cpp` (existing `test_paper_v_causal_graph`/`test_paper_v_soft_world*`, full read; new test added here)
- `docs/research/intelligence_stats/soft_world_paper5.md` (full read, esp. §3, §6, §8 Phase 1–5 roadmap)
- `docs/research/intelligence_stats/cypha_self_correcting_paper4.md` (causal-reasoning-gap sections, `:705-730,779-781`)
- `docs/reports/INTELLIGENCE_STATS_IMPLEMENTATION.md` (Phase 1–5 shipped-component tables)
- `docs/reports/DEV_PLAN_2026-07-11.md` (§3 priority table, P3 row, stub-function list)
- `bench/results/production_overnight_20260628_170701.log` (read-only, live evidence of current causal-graph JSON output at full production scale)
- `docs/reports/HIDDEN_DIM_SCALE_PLAN.md`, `docs/reports/RPSM_UPGRADE_PLAN.md` (style/rigour reference for this document's structure)
