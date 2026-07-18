# Cypha — Bill of Work

**Compiled by:** Odin Loch
**Source:** Consolidated from `docs/FUTURE.md`, `docs/RESEARCH_STATUS.md`, `docs/verify/ROADMAP.md`,
`docs/verify/VERIFICATION_STATUS.md`, `docs/research/upgrades/*`, `docs/MULTI_VIEW_TRAINING_PLAN.md`,
`docs/CYPHALM_UPGRADE_V2.md`, `docs/CYPHA_TESTS_PHASE2.md`, `docs/native/*`, `CHANGELOG.md`.
**Repo state at compile time of original bill of work (2026-06-14):** native C++ sole runtime (P7 complete);
115 CTests blocking gate (116 when `d38` merged).
**Repo state as of this update (2026-07-18):** Bounded product/adjust wave closed out —
[`PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md`](docs/reports/PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md). **160 CTests**
blocking gate; **300k overnight COMPLETE** — H22 @ 25/25, finalize exit=0, lock `a552aee` — [`OVERNIGHT_COMPLETE_2026-07-18.md`](docs/reports/OVERNIGHT_COMPLETE_2026-07-18.md).

---

## Status as of 2026-07-18

| Area | Verdict | Evidence |
|------|---------|----------|
| **Optimality P0–2, P5, P9** | [x] Done | P0 `4133054`; P1 `31bbb0c`/`7a07f8b`; P2 `de4fa16`; P5 `da9be39`; P9 `CriticalityVector` `c759e72` — [`CYPHA_OPTIMALITY_PLAN.md`](CYPHA_OPTIMALITY_PLAN.md) |
| **Optimality P3 (class GMM)** | [x] Opt-in; XOR REJECT | hard-split warm-start still ~50.5% — [`OPTIMALITY_P3_GMM_WARMSTART_2026-07-18.md`](docs/reports/OPTIMALITY_P3_GMM_WARMSTART_2026-07-18.md); keep RFF |
| **Optimality P4 (BMA over Δk)** | [~] Opt-in shipped | `33125b8` default OFF — [`OPTIMALITY_PHASE4_2026-07-17.md`](docs/reports/OPTIMALITY_PHASE4_2026-07-17.md) |
| **Optimality P6 (IB)** | [~] Opt-in | `f0ea334` default OFF — [`OPTIMALITY_PHASE6_2026-07-17.md`](docs/reports/OPTIMALITY_PHASE6_2026-07-17.md) |
| **Optimality P7 (score match)** | [~] Opt-in; LUT kept | `f19e167` — [`OPTIMALITY_PHASE7_2026-07-17.md`](docs/reports/OPTIMALITY_PHASE7_2026-07-17.md) |
| **Optimality P8 (RB)** | [x] Audit no-go | `322cb68` — no MC estimators in scope — [`OPTIMALITY_PHASE8_2026-07-17.md`](docs/reports/OPTIMALITY_PHASE8_2026-07-17.md) |
| **B3 position weights** | [~] Opt-in; null @ 5k | `5445e40` default OFF; ~−0.00005 BPC — [`CYPHALM_B3_POSITION_WEIGHTS_2026-07-17.md`](docs/reports/CYPHALM_B3_POSITION_WEIGHTS_2026-07-17.md) |
| **Infer latency (RPSM scratch)** | [x] ~48% win | `4d3afa2` — [`INFER_LATENCY_PROFILE_2026-07-17.md`](docs/reports/INFER_LATENCY_PROFILE_2026-07-17.md) |
| **D17 perf Parts 1–6** | [x] Done | [`PERFORMANCE_PROFILE_2026-07-12.md`](docs/reports/PERFORMANCE_PROFILE_2026-07-12.md) Parts 1–6; Part 6 `12ad4b3` (skip dead BPTT slow-tier when EWC off) |
| **P1 XOR kernel gap** | [x] ~2.7pp remaining (was ~18pp) | RFF auto-gamma `rff_dim=4096` → **76.3%** vs sklearn ~79%; [`RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) Priority 1 |
| **P2 auto-gamma defaults** | [x] Shipped + smoke | Implicit CV default in `preprocessor_fit.cpp` (`d≤30`); D01/D08 FAST smoke 2026-07-17 |
| **Addendum 2 MC2/MS1** | [x] Shipped | ECE + train/held-out gap `412ded1`; [`GENERAL_METRICS_MC2_MS1_2026-07-17.md`](docs/reports/GENERAL_METRICS_MC2_MS1_2026-07-17.md) |
| **Addendum 2 MC4** | [x] Shipped | Margin distribution (mean/p50/p10) `b61543f`; [`GENERAL_METRICS_MC4_2026-07-17.md`](docs/reports/GENERAL_METRICS_MC4_2026-07-17.md) |
| **Addendum 2 MR3** | [x] Shipped | Residual autocorr + spectral flatness `b61543f`; [`GENERAL_METRICS_MR3_2026-07-17.md`](docs/reports/GENERAL_METRICS_MR3_2026-07-17.md) |
| **Cell-sweep summary.csv tool** | [~] 24/25 + fix | H15 NaN root-caused + fixed (FAST finite); 300k H15 re-run optional — [`H15_AXIOM_NAN_FIX_2026-07-18.md`](docs/reports/H15_AXIOM_NAN_FIX_2026-07-18.md) |
| **§0.5 BPC pin** | [x] Reconciled | canonical **2.873** `b0d39e7`; [`BASELINE_PIN_CANONICAL_2026-07-17.md`](docs/reports/BASELINE_PIN_CANONICAL_2026-07-17.md) |
| **P5 marketing claims** | [x] Aligned | D16B/D16F isolation caveat `3491da0` |
| **RPSM cheap hypotheses** | [x] Exhausted | [`RPSM_UPGRADE_PLAN.md`](docs/reports/RPSM_UPGRADE_PLAN.md) §13–§14 — five cheap-scale experiments; gap is zero-BPTT training, not config |
| **D10 ECG stale claim** | [x] Retired | D10A **60.67%** (~3× chance); never routed through `CellAISSM`; [`D10_ECG_SSM_DIAGNOSIS_2026-07-11.md`](docs/reports/D10_ECG_SSM_DIAGNOSIS_2026-07-11.md) |
| **D17B `n_experts=1`** | [x] Genuine dynamic | Not a warm-start reporting bug; `mean_expert_alpha` split shipped `e6d95d2`; [`D17B_EXPERT_REPORTING_2026-07-12.md`](docs/reports/D17B_EXPERT_REPORTING_2026-07-12.md) |
| **EWC D16B forgetting** | [~] Improved, not solved | Growable-`D` fix + λ sweep best **0.135→0.108** @ λ=2.0; shared-model CL still open; [`EWC_D16B_SCOPING_2026-07-12.md`](docs/reports/EWC_D16B_SCOPING_2026-07-12.md) |
| **Paper draft** | [x] Reconciled | `paper/CyphaLM_paper.md` canonical **2.873** + historical labels (2026-07-17) |
| **Curriculum / uncertainty-rank** | [x] Shipped | `curriculum.hpp` + bench `CYPHA_CURRICULUM_WINDOW`; `GET/POST /uncertainty-rank` + CTest `native_rest_uncertainty_rank` |
| **300k production overnight** | [x] Done | Lock `a552aee`; **2.864 BPC** @ 300k — [`POST_LOCK_STATUS_2026-07-18.md`](docs/reports/POST_LOCK_STATUS_2026-07-18.md) |

This is a task list with explicit done/in-progress/open markers — not a live status dashboard. See
[`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) for the canonical research journal.

---

## 0-bis. Math Integration / Intelligence Stats (Phases 24–59) — open items

Large subsystem landed: 7-statistic profile `P = (α, D_eff, σ_branch, τ, r_eu, L, C)` → scalar **κ**
(criticality score) → profile-guided navigation loss. Open questions:

- [x] **Ablation grids suspiciously flat** — confirmed still flat @ 20k for `hybrid_blend_lr` (ΔBPC≈0.001) — [`MATH_OPEN_ITEMS_2026-07-18.md`](docs/reports/MATH_OPEN_ITEMS_2026-07-18.md)
- [x] **Scale-dependent sign flip characterized** — worse@500/2k, better@5k/20k (−0.117), worse@300k (+0.209); recipe redesign only — [`MATH_OPEN_ITEMS_2026-07-18.md`](docs/reports/MATH_OPEN_ITEMS_2026-07-18.md)
- [x] **Is κ-targeting a generalization signal?** — no @ 5k (targets 0.70/0.83/0.95 flat); harmful @ 300k already known
- [x] **Reconcile kernel-LLR gap** — generalizable XOR `latent` gap now **~2.7pp** (RFF, 2026-07-11), not stale ~18pp; latent RFF auto-gamma promoted as exploratory default `beacef3` — §1 P1 (`xor_pair` prod default unchanged).
- [x] **Production-tier validation** — d53 → `preset_ship_production_wiring_ready` (`lock_joint_ok=false` @ 300k); see math production report.
- [x] **Eigenvalue `D_eff` vs variance-proxy** — +0.094 ΔBPC @ 5k reproduced; keep eigenvalue **OFF** in preset

---

## 0. Maintainer-only / release-blocking

| # | Task | Status | Source |
|---|------|--------|--------|
| [x] 0.1 | **300k production overnight** to completion | H22 @ 25/25 (`2026-07-17T15:39:44Z`) | [`OVERNIGHT_COMPLETE_2026-07-18.md`](docs/reports/OVERNIGHT_COMPLETE_2026-07-18.md) |
| [x] 0.2 | `poll_and_finalize_overnight.ps1 -AutoCommit` after 0.1 | Done; lock `a552aee` (2026-07-18 06:47 local, exit=0) | Phase 18, 24 |
| [ ] 0.3 | `gh auth login` + `publish_release.ps1` | **Blocked** — `gh auth status`: not logged in (agent dry-run needs `native/build`) | Phase 15, 19 |
| [x] 0.4 | Validate **d38** overnight certificate (domain + CTest in tree) | **PASS** `overnight_certificate_ready` vs lock `a552aee` — [`D38_STATUS_2026-07-18.md`](docs/reports/D38_STATUS_2026-07-18.md) | Phase 24 |
| [x] 0.5 | Reconcile three 300k hybrid BPC pins (2.873 / 2.892 / 2.897) | Canonical **2.873** `b0d39e7` | [`BASELINE_PIN_CANONICAL_2026-07-17.md`](docs/reports/BASELINE_PIN_CANONICAL_2026-07-17.md) |

---

## 1. Priority research queue

### P1 — Kernel LLR (Nyström RBF / RFF)
- [x] RFF auto-gamma shipped — **76.3%** @ `rff_dim=4096`, **~2.7pp** to sklearn (was ~18pp @ Nyström M=256)
- [x] RFF wired into `d03_xor` (opt-in `CYPHA_D03_KERNEL_BASIS=rff`)
- [x] Nyström M=512 impractical (`O(M³)`/step); M=384 marginal
- [x] Promote RFF auto-gamma to recommended exploratory default for generalizable `latent` mode — [`RFF_LATENT_PROMOTE_2026-07-17.md`](docs/reports/RFF_LATENT_PROMOTE_2026-07-17.md) `beacef3`; `xor_pair` prod default unchanged
- [ ] D14 kernelized routing — clean negative; do not default

### P2 — Auto-gamma RFF as default
- [x] Kernel-LLR RFF auto-gamma shipped (2026-07-11)
- [x] Promote `RFFEncoder.auto_rff_gamma_cv` to default preprocessor path (`fit_from_design_matrix` implicit when `d≤30`)
- [x] Re-run D08/D01 preprocessor smoke (FAST tier; D08 N/A, D01 0 pp delta)

### P3 — CyphaLM beat-bigram
- [x] Steps 1–6 done (hybrid **2.873 BPC** @ 300k)
- [x] D17B **`n_experts=1` genuine** — not a bug (`e6d95d2`, [`D17B_EXPERT_REPORTING_2026-07-12.md`](docs/reports/D17B_EXPERT_REPORTING_2026-07-12.md))
- [x] D17 perf Parts 1–6 (`12ad4b3`)
- [x] Step 7 — Multi-view Phase 2 gated — index-reorder dead; curriculum opt-in kept — [`P4_P5_CONTINUAL_LEARNING_DECISION_2026-07-18.md`](docs/reports/P4_P5_CONTINUAL_LEARNING_DECISION_2026-07-18.md)

### P4 — Multi-view online training (CyphaLM → CyphaDIF)
- [x] Phase 1 (LM) done @ 300k
- [x] Port attempts measured — 16I replay **negative**; class-block **negative**; curriculum **+7.4pp iris** opt-in
- [x] Fix D16 **16G** — confirmed negative control only ([`D16_MULTIVIEW_POLICY_2026-07-17.md`](docs/reports/D16_MULTIVIEW_POLICY_2026-07-17.md) §2)
- [x] Document early-stop policy — [`D16_MULTIVIEW_POLICY_2026-07-17.md`](docs/reports/D16_MULTIVIEW_POLICY_2026-07-17.md) §1 (`schedule_b` ≤24k; `same_order`×2 @ 40k; `--n-train` knob)

### P5 — Shared-model continual learning
- [x] Hybrid EWC + growable-`D` fix + λ sweep (`e6d95d2`) — best **0.135→0.108** @ λ=2.0; FAST re-sweep λ=2.0 → **0.0** forgetting
- [x] **Accept D16F isolation-only** for product; EWC optional overlay — [`P4_P5_CONTINUAL_LEARNING_DECISION_2026-07-18.md`](docs/reports/P4_P5_CONTINUAL_LEARNING_DECISION_2026-07-18.md)
- [x] Confirm marketing language live everywhere (`3491da0`)

### P6 — ECG / temporal — stale claim retired
- [x] D10A **60.67%**; not `CellAISSM` path ([`D10_ECG_SSM_DIAGNOSIS_2026-07-11.md`](docs/reports/D10_ECG_SSM_DIAGNOSIS_2026-07-11.md))
- [ ] Optional: push D10 >90% with real ECG5000 data

---

## 2. RPSM track

- [x] **Option A** matrix refactor — `batched_llr_gemm` default-on (`RPSM_UPGRADE_PLAN.md` §2)
- [x] **Option B** sequence layer — scaffold + BPTT tried (§14 negative); Small-tier capacity gate **STOP** — [`RPSM_SMALL_TIER_GATE_2026-07-18.md`](docs/reports/RPSM_SMALL_TIER_GATE_2026-07-18.md)
- [x] Global memory (Izaac VRF + GMM world) — **deprioritized** (Small-tier failed; BPTT/world-stats negative)
- [x] Final 300k D17 benchmark vs hybrid — lock stands (RPSM 7.336 vs hybrid 2.873); no further 300k RPSM this wave

---

## 3. Cell hypothesis testbench (28 variants)

- [x] Tier 1 sweep H01–H22 — overnight cell-sweep complete (H22 @ 25/25, lock `a552aee`)
- [ ] Tier 2 H07, H09–H13 native paths
- [x] Tier 3 real 300k run — complete via production overnight
- [~] Populate `results/summary.csv` vs locked baselines — 24/25 rows; H15 omitted (`bpc:null`) — [`CELL_SWEEP_SUMMARY_2026-07-18.md`](docs/reports/CELL_SWEEP_SUMMARY_2026-07-18.md)

---

## 4. CyphaLM upgrade V2

- [x] B3 position weights — opt-in `5445e40` (`--ngram-position-weights`); default OFF; null @ 5k (~−0.00005 BPC) — [`CYPHALM_B3_POSITION_WEIGHTS_2026-07-17.md`](docs/reports/CYPHALM_B3_POSITION_WEIGHTS_2026-07-17.md)
- [x] B4 bilinear fusion — opt-in `f185979` (`--ngram-bilinear-fusion`); default OFF; ~0 BPC @ 5k — [`CYPHALM_B4_BILINEAR_2026-07-17.md`](docs/reports/CYPHALM_B4_BILINEAR_2026-07-17.md)
- [x] B1 gated fusion — worse (+0.116 BPC), shelved
- [x] Learnable views — neutral, keep fixed

---

## 5. Engineering backlog

| Item | Status |
|------|--------|
| CUDA CI | [x] Local-only (policy) — [`ACCEL_CUDA.md`](docs/native/ACCEL_CUDA.md), [`FUTURE.md`](docs/FUTURE.md) §1 |
| Qt shell polish | [x] Compare + export | Empty hints + CSV/JSON export — [`QT_HARDENING_CHECKLIST_2026-07-18.md`](docs/reports/QT_HARDENING_CHECKLIST_2026-07-18.md) |
| Web UI | [~] Chat + empty/readiness polish | `b706647` + `436808f` — [`WEB_UI_POLISH_2026-07-17.md`](docs/reports/WEB_UI_POLISH_2026-07-17.md) |
| Multi-model `cypha_rest` | [x] Slice shipped `e3a5b63` — [`MULTI_MODEL_REST_2026-07-17.md`](docs/reports/MULTI_MODEL_REST_2026-07-17.md) |
| **Curriculum / active learning** | [x] Shipped (`curriculum.hpp`, `/uncertainty-rank`) |
| ONNX export | [x] encode→LLR→softmax `1cbdd8c` (header-only ModelProto smoke) |
| GGUF export | [x] Tensors packed `dad723d` (`enc_W`, `F_field`, `world.mu`, class `D`/`D_T`, `inv_v`, `llr_bias`) |
| Overnight health | [x] Complete — H22 @ 25/25, lock `a552aee` — [`OVERNIGHT_COMPLETE_2026-07-18.md`](docs/reports/OVERNIGHT_COMPLETE_2026-07-18.md) |
| Sample-efficiency curves | [x] MC5/MG5 `297f59c` — [`SAMPLE_EFFICIENCY_CURVE_2026-07-17.md`](docs/reports/SAMPLE_EFFICIENCY_CURVE_2026-07-17.md) |
| Parallel `score_matrix` | [x] ~3.4× @ n=256 `c788f5f` (OpenMP row-parallel) |
| Federated training | [~] Golden merge blocking; TLS optional — [`FEDERATED_TLS_STATUS_2026-07-17.md`](docs/reports/FEDERATED_TLS_STATUS_2026-07-17.md) (`d1a9bf1`) |
| Legacy sigmoid removal | [x] Removed (`kInferWorldGateApiVersion=2`) |

---

## 6. Paper / writeup

- [x] `paper/CyphaLM_paper.md` rewritten (`e9ac580`)
- [x] Narrative reconciliation vs lock (§0.5 BPC pin — canonical **2.873**, historical sweeps labeled)
- [~] Submit (2027 Q1 target) — figures + bibliography landed; venue choice / upload still human

---

## 7. Housekeeping

- [~] Full GPU training not implemented — gap documented; infer CUDA only — [`GPU_TRAINING_GAP_2026-07-18.md`](docs/reports/GPU_TRAINING_GAP_2026-07-18.md)
- [x] Real-data profiling pass logged — [`REAL_DATA_PROFILE_2026-07-17.md`](docs/reports/REAL_DATA_PROFILE_2026-07-17.md); `scripts/run_real_data_profile.ps1`
- [x] Qt shell manual hardening checklist + compare export — [`QT_HARDENING_CHECKLIST_2026-07-18.md`](docs/reports/QT_HARDENING_CHECKLIST_2026-07-18.md)
- [x] `cypha_som` archive — reads closed — [`docs/archive/failed_experiments/cypha_som/README.md`](docs/archive/failed_experiments/cypha_som/README.md)

---

## Suggested execution order

**Active plan:** human-gated release only — all agent-actionable BoW research closed 2026-07-18  
**Evidence:** [`UPGRADE_WAVE2_STATUS_2026-07-18.md`](docs/reports/UPGRADE_WAVE2_STATUS_2026-07-18.md), [`MATH_OPEN_ITEMS_2026-07-18.md`](docs/reports/MATH_OPEN_ITEMS_2026-07-18.md), [`PHASE_F_PAPER_CLOSEOUT_2026-07-18.md`](docs/reports/PHASE_F_PAPER_CLOSEOUT_2026-07-18.md)

1. ~~Agent research waves~~ — backlog, wave 2, BPE@300k STOP, math §0-bis mid-tier closed
2. Phase E — **`gh auth login`** then `publish_release.ps1` (only remaining blocker; dry-run already OK)
3. Phase F submit — venue/arXiv upload (human, 2027 Q1)
4. Optional future — math overnight recipe redesign (not preset promotion); residual RFF full-tier D14
