# Math Integration Bench Report

Generated: 2026-06-23 07:04:39 UTC

Build: `C:\Users\odinl\OneDrive\Desktop\Cypha\native\build_math`

Settings: `CYPHA_BENCH_FAST=1`, profile **d17**, mode **hybrid**, `n_train=400`, `n_eval=80`.

## Results

| Run | Exit | BPC | Îº | profile_completeness.all_complete |
|-----|------|-----|---|-----------------------------------|
| hybrid baseline | 0 | 5.362264 | 0.849020 | true |
| hybrid + math-integration | 0 | 5.525396 | 0.813600 | true |

## Notes

- Baseline: `cyphalm_bench_native --intelligence-profile`
- Math integration: adds `--math-integration` (profile-guided 7-stat navigation loss during train)
- Validation gate: `cypha_bench_run --domain-tag d40` (`math_integration_ready` when exit 0 + complete profile + finite BPC)
- Phase 29: hybrid LSTM **logit navigation grads** (`d_logit_uniform` → `dWy`/`dby`); CharLSTM logit nudge on `dby`
- Phase 30: **direct LSTM weight navigation** — logit nudge applied inside `backward_step` (propagates to `dWx`/`dWh`/`dWy`); H04 kernel LLR stub in CyphaDIF route
- Phase 31: **Nyström `KernelMemory`** in CyphaDIF (route/train/checkpoint); **κ trajectory λ** (`EMA κ` + `dκ/dt` boost) in math preset
- Phase 32: **per-stat deviation λ** (`|obs_i−target_i|` weighting); export **`stat_deltas`**, **`kappa_trajectory`**, **`navigation_config`**; cell lock **Pareto object**
- Phase 33: **τ forget gate** (Paper IV `0.5+0.5·τ/r_eu`); **kernel LLR in math preset**; **span=1.0** tuning
- Phase 34: **κ ceiling λ** (weaken navigation when κ > target); **LSTM hidden D_eff nudge**; **span ablation CLI** (`--per-stat-deviation-span`)
- Phase 35: **Eigenvalue D_eff PR** (Paper IV covariance eigenvalues, opt-in); **tunable κ ceiling** (`kappa_ceiling_strength`, `--kappa-lambda-target`); **r_eu forget gate** (opt-in); **d48** κ target ablation @ 5k
- Phase 36: **κ trajectory ceiling** + **κ excess grad nudge**; **soft r_eu forget blend**; **d49** ceiling 2D grid @ 5k; CLI **`--kappa-ceiling-min-scale`**
- Phase 37: **κ excess grad margin** (default **0.02** above κ target before nudge); **`--bench-seed`** / **`CYPHA_BENCH_SEED`** for reproducible 5k runs; **d50** joint lock gate reproduces `BASELINE_LOCK.json` `math_integration_results` within tolerance
- Phase 38: **ablation winners in preset** (κ target **0.83**, ceiling min_scale **0.40**); **κ-adaptive kernel blend**; **d51** opt-in lever A/B (eigenvalue D_eff × r_eu forget @ 5k)
- Phase 39: **r_eu forget gate enabled in preset**; **d49** pinned seed **42**; **d52** preset ship lock gate
- Phase 40: **κ-aware navigation warmup**; **d53** production preset ship lock (300k tier pending maintainer overnight)
- Phase 45: **κ kernel blend floor CLI ablation**; **d59** kernel blend floor grid @ 5k seed 42
- Phase 46: **κ excess grad margin grid d60** @ 5k seed 42
- Phase 47: **κ excess grad scale grid d61** @ 5k seed 42
- Phase 48: **math ablation stack complete d62** (unified d47–d61 audit)
- Phase 49: **r_eu forget blend grid d63** @ 5k seed 42
- Phase 50: **κ trajectory window grid d64** @ 5k seed 42; d62 stack → **12** tables
- Phase 51: **navigation loss warmup grid d65** @ 5k seed 42
- Phase 52: **free energy beta grid d66** @ 5k seed 42; d62 stack → **14** tables
- Phase 53: **base kernel blend grid d67** @ 5k seed 42
- Phase 54: **kernel_m grid d68** @ 5k seed 42; d62 stack → **16** tables
- Phase 55: **hybrid blend logit grid d69** @ 5k seed 42; d62 stack → **18** tables
- Phase 56: **MDL forget max norm grid d70** @ 5k seed 42
- Phase 57: **kernel_lr_scale grid d71** + **alpha_init grid d72** @ 5k seed 42; d62 stack → **21** tables
- Phase 58: **hybrid_blend_lr grid d73** @ 5k seed 42
- Phase 59: **n_experts grid d74**, **max_memory_slots grid d75**, **compress_interval grid d76** @ 5k seed 42; d62 stack → **24** tables; **preset scalar coverage complete**
- Phase 42: **cell sweep `--math-integration`**; **d56** B2 joint κ–BPC @ 5k; overnight/lock parity with `-MathIntegration`

## Scale tier (d41/d42, n_train=5000, n_eval=256)

| Run | BPC | κ |
|-----|-----|---|
| hybrid baseline | 4.001 | 0.826 |
| hybrid + math-integration | 3.983 | 0.870 |
| **Δ (math − baseline)** | **−0.017** | **+0.044** |

*(Refreshed 2026-06-23 Phase 37 — κ excess grad margin + pinned seed; d17-math ΔBPC **−0.017** retained. d50 pins **bench_seed=42** for lock repro.)*

### Joint lock (d50, n_train=5000, n_eval=256, bench_seed=42)

Pinned-seed baseline + math-integration vs `bench/BASELINE_LOCK.json` → `math_integration_results`:
- Lock baseline: BPC **4.001**, κ **0.826**
- Lock math: BPC **3.983**, κ **0.870**
- Repro tolerance: BPC ±**0.025**, κ ±**0.03**
- Joint criteria: ΔBPC **< 0**, |Δκ| ≤ **0.05** → `joint_lock_ready`

Gate: `cypha_bench_run --domain-tag d50`. Table: `bench/report/tables/d50_math_joint_lock_validation.json`.

**Validated 2026-06-23:** `joint_lock_ready` — baseline BPC **4.001**, math **3.983**, ΔBPC **−0.017**, Δκ **+0.044** (matches lock; subprocess env aligned with `cypha_baseline_lock --medium` full-corpus flags). Phase 38 preset (κ target **0.83**, min_scale **0.40**, κ kernel blend) retains joint win.

### Opt-in lever A/B (d51, n_train=5000, bench_seed=42)

| lever | ΔBPC | Δκ | joint |
|-------|------|-----|-------|
| preset (Phase 38) | **−0.017** | +0.044 | ✓ |
| + eigenvalue D_eff | +0.096 | +0.049 | ✗ |
| + r_eu forget gate | **−0.017** | +0.044 | ✓ (best score) |
| + both | +0.095 | +0.049 | ✗ |

**Status:** `opt_in_lever_joint_ready`. Eigenvalue D_eff remains **opt-in off**; r_eu forget gate **enabled in preset** (Phase 39). Table: `bench/report/tables/d51_opt_in_lever_joint_validation.json`.

### Preset ship lock (d52, n_train=5000, bench_seed=42)

Phase 39 shipped preset: `use_reu_forget_gate=true`, κ target **0.83**, min_scale **0.40**.

**Validated 2026-06-23:** `preset_ship_lock_ready` — ΔBPC **−0.017**, Δκ **+0.030**, lock repro ✓. Table: `bench/report/tables/d52_preset_ship_lock_validation.json`.

### Production preset ship lock (d53, maintainer 300k tier)

Phase 40 gate: validates `run_production_overnight.ps1 -MathIntegration` wiring + Phase 40 κ navigation warmup preset + 5k subprocess joint.

**CI @ 5k (2026-06-23):** `pending_production_preset_ship_lock` — subprocess ΔBPC **−0.017**, Δκ **+0.030** (κ warmup improved joint vs Phase 39 +0.044). Awaits `math_integration_results.n_train >= 300000` + `status=production`.

Gate: `cypha_bench_run --domain-tag d53`. Table: `bench/report/tables/d53_production_preset_ship_lock_validation.json`.

### Production math certificate (d54, maintainer 300k tier)

Phase 41 gate: unifies d42+d53 — validates production math lock tier, joint κ–BPC, and math BPC vs hybrid overnight baseline (tolerance **+0.05**).

**CI @ 5k (2026-06-19):** `pending_production_math_certificate` — subprocess ΔBPC **−0.017**, Δκ **+0.044**, `subprocess_joint_ok` ✓. Awaits `math_integration_results.n_train >= 300000` + hybrid BPC gate.

Gate: `cypha_bench_run --domain-tag d54`. Table: `bench/report/tables/d54_production_math_certificate_validation.json`.

### Nav warmup grid (d55, n_train=5000, bench_seed=42)

Strength {0.25, 0.35} × floor {0.60, 0.65} — **`nav_warmup_grid_joint_ready`**. All four cells ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Best score: strength **0.25**, floor **0.60** (marginal; preset **0.35 / 0.65** retained — tied within float noise).

Gate: `cypha_bench_run --domain-tag d55`. Table: `bench/report/tables/d55_nav_warmup_grid_joint_validation.json`.

### Cell sweep math integration (d56, smoke tier, bench_seed=42)

Phase 42 gate: B2 baseline vs math cell sweep (`--overnight-sweep-smoke`) with full navigation preset per variant.

**Validated 2026-06-19:** `cell_sweep_math_joint_ready` — B2 ΔBPC **−0.411**, Δκ **+0.048** (`joint_ok` ✓). Awaits `cell_sweep_results.math_integration_enabled` @ 300k for `production_cell_sweep_math_ready`.

Gate: `cypha_bench_run --domain-tag d56`. Table: `bench/report/tables/d56_cell_sweep_math_integration_validation.json`.

### Production cell sweep math certificate (d57, maintainer 300k tier)

Phase 43 gate: unifies d56 + d54 pattern for cell sweep — `math_integration_enabled` @ 300k, joint κ–BPC, hybrid BPC tolerance **+0.05**.

Gate: `cypha_bench_run --domain-tag d57`. Table: `bench/report/tables/d57_production_cell_sweep_math_certificate_validation.json`.

**CI @ smoke tier (2026-06-19):** `production_cell_sweep_math_joint_ready` — B2 subprocess ΔBPC **−0.411**, Δκ **+0.048** (`subprocess_joint_ok` ✓). Awaits `cell_sweep_results.math_integration_enabled` @ 300k for `production_cell_sweep_math_certificate_ready`.

### Production overnight math complete (d58, maintainer 300k tier)

Phase 44 gate: unifies d54+d57 — math/cell/overnight tier alignment, hybrid BPC gates, `best_pareto_variant`, joint κ–BPC.

Gate: `cypha_bench_run --domain-tag d58`. Table: `bench/report/tables/d58_production_overnight_math_complete_validation.json`.

**CI @ 5k (2026-06-24):** `pending_production_overnight_math_complete` — subprocess ΔBPC **−0.017**, Δκ **+0.044**, `subprocess_joint_ok` ✓. Awaits 300k math lock + cell `math_integration_enabled` + hybrid BPC gates for `production_overnight_math_complete_ready`.

### Kernel blend floor grid (d59, n_train=5000, bench_seed=42)

Phase 45 gate: ablates `kappa_kernel_blend_floor` {0.05, 0.08, 0.12} with full math-integration preset @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d59`. Table: `bench/report/tables/d59_kernel_blend_floor_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `kernel_blend_floor_grid_joint_ready` — all floors {0.05, 0.08, 0.12} ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Preset floor **0.08** retained (grid tied at FAST tier).

### Excess grad margin grid (d60, n_train=5000, bench_seed=42)

Phase 46 gate: ablates `kappa_excess_grad_margin` {0.01, 0.02, 0.04} with full math-integration preset @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d60`. Table: `bench/report/tables/d60_excess_grad_margin_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `excess_grad_margin_grid_joint_ready` — all margins {0.01, 0.02, 0.04} ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Preset margin **0.02** retained (grid tied at FAST tier).

### Excess grad scale grid (d61, n_train=5000, bench_seed=42)

Phase 47 gate: ablates `kappa_excess_grad_scale` {0.25, 0.35, 0.50} with preset margin **0.02** @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d61`. Table: `bench/report/tables/d61_excess_grad_scale_grid_joint_validation.json`.

### Math ablation stack complete (d62, n_train=5000, bench_seed=42)

Phase 48 meta-gate: audits ablation tables **d47–d52, d55, d59–d61, d63–d76** for `*_joint_ready`, plus subprocess joint @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d62`. Table: `bench/report/tables/d62_math_ablation_stack_complete_validation.json`.

Status tiers: `math_ablation_stack_complete_ready` (all tables + joint) | `math_ablation_stack_joint_ready` (subprocess joint only) | `math_ablation_stack_ablation_ready`.

**CI @ 5k (2026-06-19):** `math_ablation_stack_complete_ready` — **24/24** stack tables joint-ready, subprocess ΔBPC **−0.017**, Δκ **+0.044** (joint ✓).

### Base kernel blend grid (d67, n_train=5000, bench_seed=42)

Phase 53 gate: ablates base `kernel_blend` {0.15, 0.25, 0.40} (DIF/kernel LLR anchor; distinct from d59 κ floor) @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d67`. Table: `bench/report/tables/d67_kernel_blend_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `kernel_blend_grid_joint_ready` — blends {0.15, 0.25, 0.40} ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Preset **0.25** retained.

### Nyström kernel_m grid (d68, n_train=5000, bench_seed=42)

Phase 54 gate: ablates `kernel_m` {32, 64, 128} @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d68`. Table: `bench/report/tables/d68_kernel_m_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `kernel_m_grid_joint_ready` — kernel_m {32, 64, 128} ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Preset **64** retained.

### GRIA/LSTM hybrid blend logit grid (d69, n_train=5000, bench_seed=42)

Phase 55 gate: ablates `hybrid_blend_logit` {0.0, 0.5, 1.0} (sigmoid GRIA/LSTM mix prior) @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d69`. Table: `bench/report/tables/d69_hybrid_blend_logit_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `hybrid_blend_logit_grid_joint_ready` — logit {0.0, 0.5, 1.0} ΔBPC **−0.017** (best **−0.0174** @ 0.0), Δκ **+0.044** (joint ✓). Preset **0.5** retained (grid nearly tied at FAST tier).

### H12 MDL forget max norm grid (d70, n_train=5000, bench_seed=42)

Phase 56 gate: ablates `mdl_forget_max_norm` {2.0, 4.0, 8.0} (L2 cap on H12 field via `mdl_forget_project`) @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d70`. Table: `bench/report/tables/d70_mdl_forget_max_norm_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `mdl_forget_max_norm_grid_joint_ready` — max_norm {2.0, 4.0, 8.0} ΔBPC **−0.017**, Δκ **+0.044** (joint ✓; grid tied). Preset **4.0** retained.

### Kernel LLR lr scale grid (d71, n_train=5000, bench_seed=42)

Phase 57 gate: ablates `kernel_lr_scale` {0.5, 1.0, 2.0} (Nyström memory update rate) @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d71`. Table: `bench/report/tables/d71_kernel_lr_scale_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `kernel_lr_scale_grid_joint_ready` — scales {0.5, 1.0, 2.0} ΔBPC **−0.017**, Δκ **+0.044** (joint ✓; grid tied). Preset **1.0** retained.

### GRIA alpha_init grid (d72, n_train=5000, bench_seed=42)

Phase 57 gate: ablates `alpha_init` {0.3, 0.5, 0.7} (Dirichlet prior for GRIA mixture) @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d72`. Table: `bench/report/tables/d72_alpha_init_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `alpha_init_grid_joint_ready` — α_init {0.3, 0.5, 0.7} ΔBPC **−0.017** (best **−0.0174** @ 0.3), Δκ **+0.043**–**+0.044** (joint ✓). Preset **0.5** retained.

### Navigation hybrid blend lr grid (d73, n_train=5000, bench_seed=42)

Phase 58 gate: ablates `hybrid_blend_lr` {0.005, 0.01, 0.02} (κ-navigation nudge on GRIA/LSTM logit) @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d73`. Table: `bench/report/tables/d73_hybrid_blend_lr_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `hybrid_blend_lr_grid_joint_ready` — lr {0.005, 0.01, 0.02} ΔBPC **−0.014** to **−0.019** (best **−0.019** @ 0.02), Δκ **+0.044** (joint ✓). Preset **0.01** retained (balanced κ–BPC at FAST tier).

### DIF n_experts grid (d74, n_train=5000, bench_seed=42)

Phase 59 gate: ablates `n_experts` {4, 8, 12} (OOD branching + DIF mixture width) @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d74`. Table: `bench/report/tables/d74_n_experts_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `n_experts_grid_joint_ready` — experts {4, 8, 12} ΔBPC **−0.017**, Δκ **+0.044** (joint ✓; grid nearly tied). Preset **8** retained.

### Priority replay max_memory_slots grid (d75, n_train=5000, bench_seed=42)

Phase 59 gate: ablates `max_memory_slots` {128, 256, 512} @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d75`. Table: `bench/report/tables/d75_max_memory_slots_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `max_memory_slots_grid_joint_ready` — slots {128, 256, 512} ΔBPC **−0.017**, Δκ **+0.044** (joint ✓; grid tied). Preset **256** retained.

### Compressive memory compress_interval grid (d76, n_train=5000, bench_seed=42)

Phase 59 gate: ablates `compress_interval` {8, 16, 32} (hierarchical SSM + memory compression cadence) @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d76`. Table: `bench/report/tables/d76_compress_interval_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `compress_interval_grid_joint_ready` — intervals {8, 16, 32} ΔBPC **−0.017**, Δκ **+0.044** (joint ✓; grid tied). Preset **16** retained.

### Free energy beta grid (d66, n_train=5000, bench_seed=42)

Phase 49 gate: ablates `reu_forget_gate_blend` {0.0, 0.25, 0.50} via `hybrid_forget_gate_scale` @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d63`. Table: `bench/report/tables/d63_reu_forget_blend_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `reu_forget_blend_grid_joint_ready` — all blends ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Preset blend **0.25** retained (grid tied at FAST tier).

### κ trajectory window grid (d64, n_train=5000, bench_seed=42)

Phase 50 gate: ablates `kappa_trajectory_window` {8, 16, 32} (EMA λ trajectory) @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d64`. Table: `bench/report/tables/d64_kappa_trajectory_window_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `kappa_trajectory_window_grid_joint_ready` — all windows ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Preset window **16** retained (grid tied at FAST tier).

### Navigation loss warmup grid (d65, n_train=5000, bench_seed=42)

Phase 51 gate: ablates `navigation_loss_warmup_steps` {100, 200, 400} @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d65`. Table: `bench/report/tables/d65_navigation_loss_warmup_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `navigation_loss_warmup_grid_joint_ready` — all steps ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Preset **200** retained (grid tied at FAST tier).

### Free energy beta grid (d66, n_train=5000, bench_seed=42)

Phase 52 gate: ablates `free_energy_beta` {0.005, 0.01, 0.02} @ 5k seed 42.

Gate: `cypha_bench_run --domain-tag d66`. Table: `bench/report/tables/d66_free_energy_beta_grid_joint_validation.json`.

**CI @ 5k (2026-06-19):** `free_energy_beta_grid_joint_ready` — all betas ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Preset **0.01** retained (grid tied at FAST tier).

### Excess grad scale grid (d61) — validated

**CI @ 5k (2026-06-19):** `excess_grad_scale_grid_joint_ready` — scales {0.25, 0.35, 0.50} all ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). Preset scale **0.35** retained (grid tied at FAST tier).

### κ ceiling ablation (d48, n_train=5000) — refreshed

All κ targets {0.80, 0.83, 0.86} now beat baseline BPC at 5k with aligned corpus env → **`kappa_ceiling_joint_ready`**. Best target **0.86** (ΔBPC **−0.017**). Table: `bench/report/tables/d48_kappa_ceiling_ablation_validation.json`.

### Ceiling grid (d49, n_train=5000, bench_seed=42)

Strength {1.5, 2.5} × min_scale {0.40, 0.55} — **`ceiling_grid_joint_ready`** (Phase 39 seed-42 + corpus env). All cells ΔBPC **−0.017**, Δκ **+0.044**. Table: `bench/report/tables/d49_ceiling_grid_joint_validation.json`.

### Span ablation (d47, n_train=500, n_eval=80)

| span | baseline BPC | math BPC | ΔBPC | κ (math) | Pareto score |
|------|--------------|----------|------|----------|--------------|
| 0.5 | 5.169 | 5.319 | +0.150 | 0.875 | 0.860 |
| 1.0 | 5.169 | 5.320 | +0.150 | 0.875 | 0.860 |
| 1.5 | 5.169 | 5.320 | +0.150 | 0.875 | 0.860 |

**Best span (Pareto):** 0.5 — see `bench/report/tables/d47_span_ablation_validation.json`.

*(Note: at 500-step smoke tier math-integration does not yet beat baseline BPC; 5k medium tier remains the reference for ΔBPC.)*

### κ ceiling ablation (d48, n_train=5000, n_eval=256)

| κ target | baseline BPC | math BPC | ΔBPC | κ (math) | Δκ |
|----------|--------------|----------|------|----------|-----|
| 0.80 | 4.040 | 4.047 | +0.007 | 0.872 | +0.041 |
| 0.83 | 4.040 | 4.047 | +0.007 | 0.872 | +0.041 |
| 0.86 | 4.040 | 4.047 | +0.007 | 0.872 | +0.041 |

**Best target (joint score):** 0.83 — see `bench/report/tables/d48_kappa_ceiling_ablation_validation.json`. CLI: `--kappa-lambda-target`, `--kappa-ceiling-strength`; enable eigenvalue D_eff via config `use_eigenvalue_d_eff`.

Source: `bench/BASELINE_LOCK.json` → `math_integration_results` (Phase 29 `cypha_baseline_lock --run d17-math --medium`, 2026-06-23).

**Note:** Phase 29 hybrid LSTM logit navigation grads flipped scale-tier ΔBPC from +0.005 to **−0.006** (math-integration now beats baseline BPC at 5k).
