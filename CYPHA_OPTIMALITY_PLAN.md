# Cypha — Optimality Upgrade Plan (Cursor-ready)

**Author:** Odin Loch
**Scope:** Native C++ only (Python parity retired). Turns the criticality/optimality discussion into a dependency-ordered build plan.
**Status as of 2026-07-17 (evening):** **Done:** P0 `4133054`; P1 `31bbb0c`/`7a07f8b`; P2 `de4fa16` (util caveat — `OPTIMALITY_PHASE2_2026-07-17.md`); P3 opt-in `1b59f3e` (XOR ~51%, default OFF); P5 `da9be39`; P9 `CriticalityVector` `c759e72`. **Open:** P4, P6–8; P3 default-on / XOR ≥75%; overnight finalize. See [`CYPHA_BILL_OF_WORK.md`](CYPHA_BILL_OF_WORK.md).

---

## Phase status summary (2026-07-17)

| Phase | Title | Status | Notes |
|-------|-------|--------|-------|
| 0 | Retire parity, keep regression net | [x] Done | `4133054` — `native/tests/regression/` + `*_golden` CTests; parity harness retired |
| 1 | EM keystone | [x] Done | `31bbb0c`/`7a07f8b` — `em_step.hpp`, `em_step.cpp`, `em_step_smoke` |
| 2 | Fix MoE with EM | [x] Done | `de4fa16` — EM responsibilities in `mke_scalar_train_step`; util caveat in `OPTIMALITY_PHASE2_2026-07-17.md` |
| 3 | Per-class GMM (real XOR fix) | [~] Opt-in; XOR no-go | `1b59f3e` — `use_class_gmm` default OFF; XOR ON≈51% — see `OPTIMALITY_PHASE3_2026-07-17.md` |
| 4 | Bayesian model averaging over Δk | [ ] Not started | |
| 5 | Orthogonal / leverage-score features | [x] Shipped (2026-07-17) | Leverage Nyström + SORF opt-in; CTest `native_kernel_approx_p5_smoke`; see `docs/reports/OPTIMALITY_PHASE5_2026-07-17.md` |
| 6 | Variational IB encoder | [ ] Not started | |
| 7 | Score matching → delete Bessel LUT | [~] Opt-in shipped | Score-match path + CTest; LUT retained — see `OPTIMALITY_PHASE7_2026-07-17.md` |
| 8 | Rao-Blackwellise sampling paths | [ ] Not started | |
| 9 | Runtime criticality monitor | [~] Partial | `CriticalityVector` tier/cadence `c759e72`; profiler, REST `/intelligence`, profile-guided loss shipped |

---

## Reading this plan

Each phase is self-contained and Cursor-actionable:

- **Objective** — what changes.
- **Bound saturated** — the *stated* optimum (argmax/argmin of a functional). Tier-1 items are true optima; the monitor gauges (Phase 9) are Tier-2 *locators* of an optimum, not optima themselves. Keep that distinction in code.
- **Files** — real targets in the repo.
- **Change** — the concrete edit.
- **Done when** — acceptance test.
- **Risk / goldens** — what breaks and how to re-baseline.
- **Cursor prompt** — paste-ready.

**Keystone:** Build EM *once* (Phase 1). Phases 2 and 3 both consume it. Do not write the E-step twice.

**Ordering rationale:** Phase 0 first (every later phase changes numerics; do it before re-baselining goldens). Then the keystone and the two changes that depend on it. Phase 5 (features) is independent and can run in parallel. Phase 9 (monitor) can start as read-only telemetry any time and grow.

---

## Phase 0 — Retire parity, keep the regression net

**Status:** [x] **Done** (2026-07-17) — `4133054`; `native/tests/regression/` + `*_golden` CTests; parity harness retired.

**Objective:** Split "parity" into dead Python-comparison scaffolding (delete) and frozen goldens (repurpose as native regression tests). Removing parity also *unblocks* Phases 1–3 and 7, which were previously parity-breakers.

**Bound saturated:** none — this is prep. But it removes a design constraint (Python-reproducible numerics), which unlocks free training-rule changes, native SIMD/threading, a faster RNG, and score matching (Phase 7).

**Files:**
- `native/tests/parity/` (36 files) → rename dir to `native/tests/regression/`.
- `native/tests/CMakeLists.txt` — rename CTest cases `*_parity` → `*_golden`.
- `docs/port/PORT_CONTRACT.md`, `docs/verify/VERIFICATION_STATUS.md` — strip dual-implementation language.
- `README.md` — remove "proof surface is parity with Python" framing (now false).
- `.github/workflows/*.yml` — drop any Python-setup step that existed only for parity regeneration.
- Any `.py` reference generators / fixture regenerators — delete.

**Change:** Mechanical rename + dead-scaffolding delete. The fixtures (38 dirs) stay: they are frozen expected-outputs and remain your only regression guard through the refactor wave. **Do not delete goldens.** The one exception: goldens that encode collapsing-MoE behaviour get regenerated in Phase 2, not kept.

**Done when:** build + CTest green under the new `regression`/`golden` names; no CI job installs Python; `grep -ri parity` returns only historical CHANGELOG entries.

**Risk / goldens:** Low. Pure rename + delete of dead code. Keep the golden count roughly constant.

**Cursor prompt:**
> Rename `native/tests/parity/` to `native/tests/regression/` and update `native/tests/CMakeLists.txt` so every `*_parity` CTest becomes `*_golden`, preserving all fixture loads and comparisons unchanged. Then remove Python-comparison language from `docs/port/PORT_CONTRACT.md`, `docs/verify/VERIFICATION_STATUS.md`, and `README.md`, and delete any CI step or `.py` script whose only purpose was regenerating or cross-checking fixtures against a Python implementation. Do not delete any fixture directory. Print a report of every file touched.

---

## Phase 1 — The EM keystone (shared primitive)

**Status:** [x] **Done** (2026-07-17) — `31bbb0c`/`7a07f8b`; `em_step.hpp`, `em_step.cpp`, `em_step_smoke`.

**Objective:** One reusable EM step: an E-step returning Bayes-optimal responsibilities `r_i ∝ π_i · p(x | component i)`, and a weighted M-step. Consumed by the MoE router (Phase 2) and per-class GMM (Phase 3).

**Bound saturated:** EM monotonically increases the mixture log-likelihood; E-step responsibilities are the exact posterior over latent assignment (Tier 1 — argmax of a stated functional).

**Files (new):**
- `native/include/cypha/em_step.hpp`
- `native/src/em_step.cpp`

**Change:** Implement `responsibilities(loglik[], prior[], K, temperature, eps) -> r[]` (log-sum-exp normalised) and `weighted_moment_update(...)` helpers. Diagonal-Gaussian component likelihoods reuse the existing `score_matrix_use_field` math. Keep it allocation-light and header-declared for inlining on the hot path.

**Done when:** a unit test recovers two well-separated 1-D Gaussian components from synthetic data (means within tolerance, responsibilities bimodal); log-likelihood is non-decreasing across iterations.

**Risk / goldens:** None yet — new module, no callers changed.

**Cursor prompt:**
> Create `native/include/cypha/em_step.hpp` and `native/src/em_step.cpp` implementing a reusable EM step for diagonal-Gaussian mixtures: (1) `responsibilities()` computing `r_i ∝ prior_i · exp(loglik_i)` with log-sum-exp normalisation, temperature, and an epsilon floor; (2) `weighted_moment_update()` for weighted mean/variance accumulation. Reuse the diagonal-Gaussian log-likelihood convention from `score_matrix_use_field`. Add a CTest that recovers two separated Gaussians from synthetic data and asserts the mixture log-likelihood is monotonically non-decreasing.

---

## Phase 2 — Fix MoE with EM (dead experts → live)

**Status:** [x] **Done** (2026-07-17) — EM E-step + `argmax r` router target in `mke_scalar_train_step_from_phi`; goldens regenerated; util smoke in `mke_train_step_golden`. Caveat: hard ≤60% mass floor not met under DIF prior (see report).

**Objective:** Kill the rich-get-richer collapse. Replace self-argmax router training with responsibility-based routing.

**Bound saturated:** same as Phase 1 — the router now targets the E-step posterior instead of confirming its own prior.

**Files:**
- `native/src/mke_scalar_train_step.cpp` (`mke_scalar_train_step_from_phi`)
- `native/include/cypha/mke_scalar_train_step.hpp`

**Change (four edits, all local to the train step):**
1. **Responsibilities, not routing probs.** You already compute `dp = w_i·φ` in the `y_hat` loop — reuse it: `r_i ∝ p_i · exp(−(y − w_i·φ)² / 2σ²)`, normalised. This is the E-step.
2. **Route training on `r`, not `argmax p`.** Replace the `router_label = argmax p` fallback with `argmax r` (hard EM) or train the DIF router toward soft `r`. Breaks self-confirmation because `r` depends on the target.
3. **Weight the M-step by `r_i`; drop the hard floor.** Pass `r_i` (not `p_i`) into `mke_expert_rls_scalar_step`; replace the hard `pi_floor` skip with a tiny epsilon so weak experts still get a trickle of gradient.
4. **Break symmetry + floor routing entropy.** Random per-expert init of `w_by_label`; keep routing temperature high during warmup, anneal down.

**Done when:** on a synthetic multi-cluster regression task, routing entropy stays above a floor through warmup and expert-utilisation is spread (no single expert >~60% of mass at convergence); error beats a single-head baseline.

**Risk / goldens:** Breaks `mke_train_step_golden` (was `_parity`) — correct, since it froze collapsing behaviour. Regenerate the golden from the fixed native code.

**Cursor prompt:**
> In `native/src/mke_scalar_train_step.cpp::mke_scalar_train_step_from_phi`, replace the current router training with EM using `native/cypha/em_step.hpp`: compute per-expert responsibilities `r_i ∝ p_i · exp(−(y − w_i·φ)²/2σ²)` reusing the `dp = w_i·φ` values already computed for `y_hat`; train the DIF router toward `argmax r` (not `argmax p`); pass `r_i` into `mke_expert_rls_scalar_step` and replace the hard `pi_floor` skip with an epsilon floor; add random per-expert initialisation of `w_by_label` and a routing-temperature warmup anneal. Then regenerate the `mke_train_step_golden` fixture from the updated native output and add an expert-utilisation assertion (no expert exceeds ~60% of routing mass at convergence).

---

## Phase 3 — Per-class GMM (the real XOR fix)

**Status:** [~] **Opt-in shipped; XOR no-go** (2026-07-17) — `1b59f3e`; `use_class_gmm` default OFF; XOR ON≈51% vs OFF≈51% (no kernel). Default-on / format bump deferred — see `OPTIMALITY_PHASE3_2026-07-17.md`.

**Objective:** Let each `ClassDifferential Δk` be a small mixture of Gaussians in latent space instead of one diagonal Gaussian. This removes the XOR *impossibility*, not just softens it.

**Bound saturated:** mixture log-likelihood via EM (Tier 1). **Why this is the real fix:** one diagonal Gaussian cannot cover two disconnected lobes — that is a representational impossibility, which is why kernels only patched XOR to ~76%. A 2–3 component class model occupies both lobes directly.

**Files:**
- `native/include/cypha/memory_train.hpp`, `native/src/` DIFMemory/ClassDifferential internals
- `native/include/cypha/nig_field.hpp` (component gate reuse)
- consumes `em_step.hpp`

**Change:** Generalise `Δk` from `(μ_k, v_k)` to `{(π_{k,m}, μ_{k,m}, v_{k,m})}` for a small `M_k` (start `M=2`, cap `M=4`). `score k = log Σ_m π_{k,m} · p(h | N(μ_{k,m}, v_{k,m}))` (log-sum-exp). Fit components online by the Phase-1 EM step. Keep `‖Δk‖_F ≤ C` MDL cap per component so simple classes stay single-mode (Solomonoff prior still holds; cold-start unchanged).

**Done when:** XOR (S3) breaks the ~50% linear-LLR wall using multimodal classes *without* kernels — target ≥75%; existing near-saturated tasks (Wine R2, Digits R3) do not regress.

**Risk / goldens:** Changes `.cypha` v3 serialisation (class now stores M components). Bump format version, add migration for single-component load, regenerate affected golden fixtures.

**Cursor prompt:**
> Generalise `ClassDifferential` in the DIFMemory path so each class holds a small Gaussian mixture `{(π, μ, v)}_m` (default M=2, cap 4) instead of a single diagonal Gaussian. Change the class score to `log Σ_m π_m · N(h | μ_m, v_m)` via log-sum-exp, fit components online using `cypha/em_step.hpp`, and keep the `‖Δk‖_F ≤ C` MDL cap per component. Bump the `.cypha` format version with a migration that loads legacy single-component classes as M=1. Add a CTest showing XOR (S3) exceeds 75% with multimodal classes and no kernel, and assert Wine and Digits accuracy do not regress.

---

## Phase 4 — Bayesian model averaging over Δk

**Status:** [ ] **Not started**.

**Objective:** Stop collapsing `Δk` to a point. Keep the conjugate posterior (NIG is already conjugate → closed-form) and average over it at prediction.

**Bound saturated:** BMA minimises expected log loss / is the admissible predictor under KL (Tier 1). No sampling — conjugacy makes the average analytic.

**Files:**
- `native/include/cypha/nig_field.hpp`, `native/include/cypha/nig_gig_math.hpp`
- inference path in `infer_cpu.hpp` / `dif_rest.hpp`

**Change:** At inference, integrate the class score over the NIG posterior rather than plugging the MAP `Δk`. Emit a posterior credible interval as `confidence` and `r_eff`, replacing the heuristic confidence scalar. Strengthens the OOD/calibration story you already lead with.

**Done when:** calibration (reliability-curve ECE) improves vs the point-estimate baseline on R1–R4; credible intervals have correct empirical coverage on a held-out canary.

**Risk / goldens:** Inference outputs shift (better-calibrated). Regenerate the top-level `cypha_golden` inference fixture.

**Cursor prompt:**
> In the CyphaDIF inference path, replace the MAP plug-in of `Δk` with a Bayesian model average: integrate the class score analytically over the conjugate NIG posterior in `nig_field.hpp`/`nig_gig_math.hpp`, and emit a posterior credible interval as the reported `confidence`. Add a calibration CTest (expected calibration error must not worsen, coverage of the credible interval within tolerance on a held-out set) and regenerate the top-level inference golden.

---

## Phase 5 — Orthogonal / leverage-score features (independent; parallelisable)

**Status:** [x] **Shipped** (2026-07-17). Leverage-score Nyström + SORF RFF opt-in paths; CTest `native_kernel_approx_p5_smoke`. Report: [`docs/reports/OPTIMALITY_PHASE5_2026-07-17.md`](docs/reports/OPTIMALITY_PHASE5_2026-07-17.md).

**Objective:** Raise the kernel-approximation quality at fixed feature budget. Complements Phase 3 from the other side.

**Bound saturated:** orthogonal random features are minimum-variance unbiased vs iid RFF (Yu et al.); ridge-leverage-score Nyström minimises the reconstruction-error bound `‖K̂ − K‖` at fixed landmark budget (Alaoui–Mahoney). Both Tier 1. **Note:** raises the nonlinear ceiling; the *impossibility* wall is removed by Phase 3, not here.

**Files:**
- `native/include/cypha/` RFFEncoder, `kernel_memory.hpp`, the Nyström LLR path

**Change:** Swap iid Gaussian RFF for structured orthogonal features (SORF/Fastfood blocks). Swap uniform Nyström landmarks for ridge-leverage-score sampling. You have the exact kernel to compare against, so the improvement is measurable, not assumed.

**Done when:** at fixed `rff_D=256`, `‖K̂ − K‖_F` drops vs the iid/uniform baseline; XOR kernel-LLR improves from the current ~76% (RFF auto-gamma baseline).

**Risk / goldens:** Feature values change → regenerate RFF-dependent golden fixtures (`rff_regression`, `preprocessor_fit_rff`).

**Cursor prompt:**
> Replace iid Gaussian random features in `RFFEncoder` with structured orthogonal features (SORF/Fastfood), and replace uniform Nyström landmark sampling in the kernel-LLR path with ridge-leverage-score sampling. Add a CTest that measures `‖K̂ − K‖_F` against the exact kernel and asserts it decreases vs the previous iid/uniform baseline at `rff_D=256`. Regenerate `rff_regression` and `preprocessor_fit_rff` goldens.

---

## Phase 6 — Variational IB encoder (make the stated architecture true)

**Status:** [ ] **Not started**.

**Objective:** Replace the heuristic Fisher–Rao push-pull encoder with an actual Information Bottleneck minimiser. You *cite* IB as a founding thread; currently the encoder only gauges it.

**Bound saturated:** variational IB bound (Alemi et al.) — a proper `argmin I(X;T) − β·I(T;Y)` (Tier 1). A nonlinear IB encoder also attacks nonlinear boundaries from the encoder side.

**Files:**
- `native/include/cypha/encoder_contrastive.hpp`, EncoderProjection update

**Change:** Replace the contrastive Fisher–Rao residual gradient with the variational IB loss; keep the `‖W‖_F ≤ 8.0` cap for stability. Report the `β` trade-off point.

**Done when:** latent mutual-information proxy tracks the target class better at fixed compression; classification improves or holds on R1–R4 with better OOD separation.

**Risk / goldens:** Encoder updates change → regenerate encoder-dependent goldens.

**Cursor prompt:**
> Replace the heuristic Fisher–Rao contrastive update in `encoder_contrastive.hpp` with a variational Information Bottleneck objective (`min I(X;T) − β·I(T;Y)` via the Alemi variational bound), retaining the `‖W‖_F ≤ 8.0` Frobenius cap. Expose `β` as a config parameter, add a CTest tracking a latent MI proxy vs class label at fixed compression, and regenerate affected encoder goldens.

---

## Phase 7 — Score matching → delete the Bessel LUT

**Status:** [ ] **Not started** — blocked on Phase 0.

**Objective:** Fit the GH/NIG gate without its partition function; drop `bessel_ratios.npz`. Only became possible once parity was retired (Phase 0).

**Bound saturated:** score matching minimises Fisher divergence and is a consistent estimator — no normalising constant required (Tier 1).

**Files:**
- `native/include/cypha/bessel_table.hpp`, `bessel_ratios.npz` (delete), NIG gate fit path

**Change:** Fit the heavy-tailed gate by score matching (Hyvärinen). Remove the 300 KB LUT and the Bessel-accuracy constraint entirely.

**Done when:** gate fit matches or beats the LUT-based density on held-out log-likelihood; `bessel_ratios.npz` and `bessel_table.hpp` removed; build shrinks.

**Risk / goldens:** Gate outputs shift → regenerate GH/NIG goldens (`gh_infer_deliberation`, quantile-DIF).

**Cursor prompt:**
> Replace the Bessel-ratio-LUT normalisation of the GH/NIG gate with a score-matching fit (Hyvärinen Fisher-divergence objective) that avoids the partition function. Delete `bessel_ratios.npz` and `bessel_table.hpp` and their references. Add a CTest asserting held-out log-likelihood of the gate is at least as good as the previous LUT-based version, and regenerate the `gh_infer_deliberation` and quantile-DIF goldens.

---

## Phase 8 — Rao-Blackwellise the sampling paths (free variance reduction)

**Status:** [ ] **Not started**.

**Objective:** Replace Monte-Carlo sample averages with conditional expectations given sufficient statistics, anywhere you sample (KDE generation, priority-replay estimates).

**Bound saturated:** Rao–Blackwell theorem — conditioning on a sufficient statistic *never increases* variance (Tier 1, unconditional improvement).

**Files:**
- `native/include/cypha/generation.hpp`, `replay_buffer.hpp`

**Change:** Where the replay/KDE paths average samples, substitute the analytic conditional expectation using the sufficient statistics you already carry.

**Done when:** estimator variance drops (or holds) at equal or lower compute on a fixed seed; generation quality metrics do not regress.

**Risk / goldens:** Low if kept numerically equivalent in expectation; regenerate generation goldens if outputs move.

**Cursor prompt:**
> In `generation.hpp` and `replay_buffer.hpp`, replace Monte-Carlo sample averages with Rao-Blackwellised conditional expectations computed from the sufficient statistics already stored in the model. Add a CTest comparing estimator variance before/after on a fixed seed and assert it does not increase; regenerate generation goldens only if outputs change.

---

## Phase 9 — Runtime criticality monitor (read-only first)

**Status:** [~] **Partially shipped** (2026-07-17) — `CriticalityVector` / `CriticalityField` with tier/cadence tags shipped `c759e72`; hot gauges + REST `/intelligence/criticality`; read-only (inference byte-identical). Prior Intelligence Stats work (`IntelligenceProfiler`, κ, `lm_intelligence_monitor.hpp`, profile-guided loss) remains. **Open:** wire session extras into hot input; connect mid estimators (spectral radius, stochastic `‖K̂−K‖_F`, forgetting canary); optional `experiment_db` ring buffer — see `OPTIMALITY_PHASE9_2026-07-17.md`.

**Objective:** Always-on self-monitoring vector: one normalised health scalar per component. Ship telemetry first (zero parity/golden risk), promote a gauge to closed-loop control only once it has its own fixture.

**Bound semantics (must be encoded in the struct):**
- **Tier-1 fields → distance-from-bound.** There is a real bound to subtract from: kernel-approx error `‖K̂ − K‖_F`, forgetting canary (LLR-residual drift on a frozen set).
- **Tier-2 fields → distance-from-critical-point.** A target value, no bound: `α` (target ~0.5, band 0.35–0.65), `ρ(J)` (target 1.0), MoE routing entropy.

Tag every field with its Tier so the readout never conflates "how far from optimal" with "how far from the critical point." Conflating them is the one lie to avoid.

**Cadence (mirrors your tiered context):**
- **Hot / every step (cheap, mostly already computed):** `α` from GRIA entropy, MoE routing entropy + dead-expert fraction off the gate softmax, `anomaly_score`, `drift_score`, NIG field confidence, effective sample size, OOD-flag rate.
- **Mid / every N steps on a subsample (estimators — label as such):** `ρ(J)` via a few power-iteration steps on the SSM transition, stochastic `‖K̂ − K‖_F` on a landmark subset, forgetting canary.
- **Cold / on demand only:** full leverage-score recompute, full kernel reconstruction, OT coupling suboptimality.

**Files:**
- `native/include/cypha/intelligence/intelligence_profiler.hpp`, `soft_world_monitor.hpp`, `cyphalm/lm_intelligence_monitor.hpp`
- `cyphalm/cyphalm_alpha_spectrum.hpp` (already reports `mean_alpha`, `fraction_near_edge_of_chaos`)
- sinks: `intelligence_rest.hpp` metrics field + optional `experiment_db` ring buffer for history

**Change:** Add a `CriticalityVector` struct (per-field: value, tier tag, target-or-bound, cadence). Populate hot fields from existing state; add mid estimators behind a subsample gate. Expose read-only via REST; optional SQLite ring buffer.

**Done when:** REST emits the live vector (e.g. `α 0.52 [T2], ρ(J) 1.03 [T2], routing-entropy 0.8 [T2], kernel-err 4e-3 [T1], forgetting 0.001 [T1]`); zero change to inference outputs (read-only); no golden touched.

**Risk / goldens:** None for telemetry (read-only). Any future closed-loop gauge (like GRIA already does with `control_interval=50`) needs hysteresis + its own fixture before it acts.

**Cursor prompt:**
> Add a `CriticalityVector` struct to the intelligence monitor surface (`intelligence_profiler.hpp`, `soft_world_monitor.hpp`, `lm_intelligence_monitor.hpp`) where each field carries: value, a Tier tag (Tier-1 = distance-from-bound, Tier-2 = distance-from-critical-point), its target-or-bound, and a cadence tag (hot/mid/cold). Populate hot fields (α from GRIA, MoE routing entropy and dead-expert fraction, anomaly/drift scores, NIG confidence, effective sample size, OOD rate) from existing state every step; add mid-cadence estimators (spectral radius via power iteration, stochastic `‖K̂−K‖_F`, a frozen-canary forgetting proxy) behind a subsample gate. Expose the vector read-only through `intelligence_rest.hpp` and optionally log to an `experiment_db` ring buffer. Assert inference outputs are byte-identical with the monitor on vs off (read-only guarantee).

---

## Suggested execution order

1. **Phase 0** — retire parity (unblocks the rest).
2. **Phase 1** — EM keystone.
3. **Phase 2** — MoE fix (consumes EM; regenerate MoE golden).
4. **Phase 3** — per-class GMM (consumes EM; the real XOR fix; format bump).
5. **Phase 4** — BMA over Δk (conjugate add-on).
6. **Phase 9** — start the monitor as read-only telemetry (can begin any time after Phase 2 so routing entropy is available).
7. **Phase 5** — orthogonal/leverage features (parallelisable throughout).
8. **Phase 6** — variational IB encoder.
9. **Phase 7** — score matching, delete Bessel LUT.
10. **Phase 8** — Rao-Blackwell sampling.

**Tier discipline (keep visible in code and docs):** Phases 1–8 are Tier-1 optima — each is the exact solution to a stated variational problem, and each acceptance test names the quantity it saturates. Phase 9 gauges are Tier-2 locators — they mark where a *different* objective is extremal and must be reported as distance-from-critical-point, never as optima. Do not let "monitored" drift into "optimised."

**Held for later (heavier Tier-1, lower payoff-per-effort now):** full natural gradient beyond diagonal Fisher (K-FAC); NML / stochastic complexity as the exact MDL code. Your diagonal Fisher and `‖Δk‖ ≤ C` are cheap approximations to these — revisit only after Phases 1–5 land.
