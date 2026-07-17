# Cypha — Bill of Work

**Compiled by:** Odin Loch
**Source:** Consolidated from `docs/FUTURE.md`, `docs/RESEARCH_STATUS.md`, `docs/verify/ROADMAP.md`,
`docs/verify/VERIFICATION_STATUS.md`, `docs/research/upgrades/*`, `docs/MULTI_VIEW_TRAINING_PLAN.md`,
`docs/CYPHALM_UPGRADE_V2.md`, `docs/CYPHA_TESTS_PHASE2.md`, `docs/native/*`, `CHANGELOG.md`.
**Repo state at compile time of original bill of work (2026-06-14):** native C++ sole runtime (P7 complete);
115 CTests blocking gate (116 when `d38` merged).
**Repo state as of this update (2026-07-17 late evening):** Bounded product/adjust wave closed out —
[`PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md`](docs/reports/PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md). **160 CTests**
blocking gate; overnight H17 @ 20/25 healthy — [`OVERNIGHT_HEALTH_2026-07-17.md`](docs/reports/OVERNIGHT_HEALTH_2026-07-17.md) §7.

---

## Status as of 2026-07-17

| Area | Verdict | Evidence |
|------|---------|----------|
| **Optimality P0–2, P5, P9** | [x] Done | P0 `4133054`; P1 `31bbb0c`/`7a07f8b`; P2 `de4fa16`; P5 `da9be39`; P9 `CriticalityVector` `c759e72` — [`CYPHA_OPTIMALITY_PLAN.md`](CYPHA_OPTIMALITY_PLAN.md) |
| **Optimality P3 (class GMM)** | [~] Opt-in; XOR no-go | `1b59f3e` default OFF; XOR ~51% — needs different approach or kernels |
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
| **§0.5 BPC pin** | [x] Reconciled | canonical **2.873** `b0d39e7`; [`BASELINE_PIN_CANONICAL_2026-07-17.md`](docs/reports/BASELINE_PIN_CANONICAL_2026-07-17.md) |
| **P5 marketing claims** | [x] Aligned | D16B/D16F isolation caveat `3491da0` |
| **RPSM cheap hypotheses** | [x] Exhausted | [`RPSM_UPGRADE_PLAN.md`](docs/reports/RPSM_UPGRADE_PLAN.md) §13–§14 — five cheap-scale experiments; gap is zero-BPTT training, not config |
| **D10 ECG stale claim** | [x] Retired | D10A **60.67%** (~3× chance); never routed through `CellAISSM`; [`D10_ECG_SSM_DIAGNOSIS_2026-07-11.md`](docs/reports/D10_ECG_SSM_DIAGNOSIS_2026-07-11.md) |
| **D17B `n_experts=1`** | [x] Genuine dynamic | Not a warm-start reporting bug; `mean_expert_alpha` split shipped `e6d95d2`; [`D17B_EXPERT_REPORTING_2026-07-12.md`](docs/reports/D17B_EXPERT_REPORTING_2026-07-12.md) |
| **EWC D16B forgetting** | [~] Improved, not solved | Growable-`D` fix + λ sweep best **0.135→0.108** @ λ=2.0; shared-model CL still open; [`EWC_D16B_SCOPING_2026-07-12.md`](docs/reports/EWC_D16B_SCOPING_2026-07-12.md) |
| **Paper draft** | [x] Reconciled | `paper/CyphaLM_paper.md` canonical **2.873** + historical labels (2026-07-17) |
| **Curriculum / uncertainty-rank** | [x] Shipped | `curriculum.hpp` + bench `CYPHA_CURRICULUM_WINDOW`; `GET/POST /uncertainty-rank` + CTest `native_rest_uncertainty_rank` |
| **300k production overnight** | [~] H17 healthy, wait | H17 @ 20/25; active compute per `a09b665` — [`OVERNIGHT_HEALTH_2026-07-17.md`](docs/reports/OVERNIGHT_HEALTH_2026-07-17.md); d27–d38 gates `pending_production` until lock lands |

This is a task list with explicit done/in-progress/open markers — not a live status dashboard. See
[`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) for the canonical research journal.

---

## 0-bis. Math Integration / Intelligence Stats (Phases 24–59) — open items

Large subsystem landed: 7-statistic profile `P = (α, D_eff, σ_branch, τ, r_eu, L, C)` → scalar **κ**
(criticality score) → profile-guided navigation loss. Open questions:

- [ ] **Ablation grids suspiciously flat** — 16 hyperparameters report identical ΔBPC/Δκ at FAST/5k tier; re-run at production scale before trusting preset values.
- [ ] **Scale-dependent sign flip unexplained** — math-integration worse at 500 train, better at 5k; needs scaling-law sweep.
- [ ] **Is κ-targeting a generalization signal?** — no held-out transfer tests beyond existing eval split.
- [~] **Reconcile kernel-LLR gap** — generalizable XOR `latent` gap now **~2.7pp** (RFF, 2026-07-11), not stale ~18pp; promote RFF to production default — §1 P1.
- [ ] **Production-tier validation** — d53–d58 all `pending_production` until 300k overnight with `-MathIntegration`.
- [ ] **Eigenvalue `D_eff` vs τ-based `r_eu` split** — eigenvalue estimator alone +0.096 ΔBPC; derive why.

---

## 0. Maintainer-only / release-blocking

| # | Task | Status | Source |
|---|------|--------|--------|
| [~] 0.1 | **300k production overnight** to completion | In progress: H17 @ 20/25 (healthy, wait) | RESEARCH_STATUS Phase 13–24; `a09b665` |
| [ ] 0.2 | `poll_and_finalize_overnight.ps1 -AutoCommit` after 0.1 | Blocked on 0.1 | Phase 18, 24 |
| [ ] 0.3 | `gh auth login` + `publish_release.ps1` | Auth still gated | Phase 15, 19 |
| [ ] 0.4 | Merge **d38** once 0.1–0.2 land | 115 → 116 CTests | Phase 24 |
| [x] 0.5 | Reconcile three 300k hybrid BPC pins (2.873 / 2.892 / 2.897) | Canonical **2.873** `b0d39e7` | [`BASELINE_PIN_CANONICAL_2026-07-17.md`](docs/reports/BASELINE_PIN_CANONICAL_2026-07-17.md) |

---

## 1. Priority research queue

### P1 — Kernel LLR (Nyström RBF / RFF)
- [x] RFF auto-gamma shipped — **76.3%** @ `rff_dim=4096`, **~2.7pp** to sklearn (was ~18pp @ Nyström M=256)
- [x] RFF wired into `d03_xor` (opt-in `CYPHA_D03_KERNEL_BASIS=rff`)
- [x] Nyström M=512 impractical (`O(M³)`/step); M=384 marginal
- [ ] Promote RFF auto-gamma to production default for generalizable `latent` mode
- [ ] D14 kernelized routing — clean negative; do not default

### P2 — Auto-gamma RFF as default
- [x] Kernel-LLR RFF auto-gamma shipped (2026-07-11)
- [x] Promote `RFFEncoder.auto_rff_gamma_cv` to default preprocessor path (`fit_from_design_matrix` implicit when `d≤30`)
- [x] Re-run D08/D01 preprocessor smoke (FAST tier; D08 N/A, D01 0 pp delta)

### P3 — CyphaLM beat-bigram
- [x] Steps 1–6 done (hybrid **2.873 BPC** @ 300k)
- [x] D17B **`n_experts=1` genuine** — not a bug (`e6d95d2`, [`D17B_EXPERT_REPORTING_2026-07-12.md`](docs/reports/D17B_EXPERT_REPORTING_2026-07-12.md))
- [x] D17 perf Parts 1–6 (`12ad4b3`)
- [ ] Step 7 — Multi-view Phase 2 (D16/DIF)

### P4 — Multi-view online training (CyphaLM → CyphaDIF)
- [x] Phase 1 (LM) done @ 300k
- [ ] Port multi-view scheduling to CyphaDIF (D16)
- [ ] Fix D16 **16G** task-block-shuffle regression — negative control documented; next = DIF-V3 replay-interleave ([`D16_MULTIVIEW_POLICY_2026-07-17.md`](docs/reports/D16_MULTIVIEW_POLICY_2026-07-17.md) §2)
- [x] Document early-stop policy — [`D16_MULTIVIEW_POLICY_2026-07-17.md`](docs/reports/D16_MULTIVIEW_POLICY_2026-07-17.md) §1 (`schedule_b` ≤24k; `same_order`×2 @ 40k; `--n-train` knob)

### P5 — Shared-model continual learning
- [x] Hybrid EWC + growable-`D` fix + λ sweep (`e6d95d2`) — best **0.135→0.108** @ λ=2.0
- [~] Forgetting improved but not solved (legacy **0.813** everyday profile)
- [ ] Further EWC / routing redesign vs accept D16F isolation-only
- [x] Confirm marketing language live everywhere (`3491da0`)

### P6 — ECG / temporal — stale claim retired
- [x] D10A **60.67%**; not `CellAISSM` path ([`D10_ECG_SSM_DIAGNOSIS_2026-07-11.md`](docs/reports/D10_ECG_SSM_DIAGNOSIS_2026-07-11.md))
- [ ] Optional: push D10 >90% with real ECG5000 data

---

## 2. RPSM track

- [x] **Option A** matrix refactor — `batched_llr_gemm` default-on (`RPSM_UPGRADE_PLAN.md` §2)
- [~] **Option B** sequence layer — scaffold + bug fixes; **cheap hypotheses exhausted** (§13–§14); zero-BPTT gap remains; D17 < **2.873** not met
- [ ] Global memory (Izaac VRF + Gaussian-mixture world model)
- [ ] Final 300k D17 benchmark vs hybrid

---

## 3. Cell hypothesis testbench (28 variants)

- [~] Tier 1 sweep H01–H05 — overnight cell-sweep in progress (H17 @ 20/25)
- [ ] Tier 2 H07, H09–H13 native paths
- [ ] Tier 3 real 300k run
- [ ] Populate `results/summary.csv` vs locked baselines

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
| Qt shell polish | [ ] Optional |
| Web UI | [ ] Partial |
| Multi-model `cypha_rest` | [x] Slice shipped `e3a5b63` — [`MULTI_MODEL_REST_2026-07-17.md`](docs/reports/MULTI_MODEL_REST_2026-07-17.md) |
| **Curriculum / active learning** | [x] Shipped (`curriculum.hpp`, `/uncertainty-rank`) |
| ONNX export | [x] encode→LLR→softmax `1cbdd8c` (header-only ModelProto smoke) |
| GGUF export | [x] Tensors packed `dad723d` (`enc_W`, `F_field`, `world.mu`, class `D`/`D_T`, `inv_v`, `llr_bias`) |
| Overnight health | [~] H17 healthy, wait `a09b665` — [`OVERNIGHT_HEALTH_2026-07-17.md`](docs/reports/OVERNIGHT_HEALTH_2026-07-17.md) |
| Sample-efficiency curves | [x] MC5/MG5 `297f59c` — [`SAMPLE_EFFICIENCY_CURVE_2026-07-17.md`](docs/reports/SAMPLE_EFFICIENCY_CURVE_2026-07-17.md) |
| Parallel `score_matrix` | [x] ~3.4× @ n=256 `c788f5f` (OpenMP row-parallel) |
| Federated training | [ ] TLS smoke only |
| Legacy sigmoid removal | [x] Removed (`kInferWorldGateApiVersion=2`) |

---

## 6. Paper / writeup

- [x] `paper/CyphaLM_paper.md` rewritten (`e9ac580`)
- [x] Narrative reconciliation vs lock (§0.5 BPC pin — canonical **2.873**, historical sweeps labeled)
- [ ] Submit (2027 Q1 target)

---

## 7. Housekeeping

- [ ] Full GPU training not implemented
- [x] Real-data profiling pass logged — [`REAL_DATA_PROFILE_2026-07-17.md`](docs/reports/REAL_DATA_PROFILE_2026-07-17.md); `scripts/run_real_data_profile.ps1`
- [ ] Qt shell manual hardening pass
- [ ] `cypha_som` archive — reads closed

---

## Suggested execution order

1. §0 — 300k overnight + lock commit + release
2. §0.5 — canonical BPC pin
3. §1 P1/P2 — promote RFF defaults
4. §2 RPSM — BPTT in training loop (cheap hypotheses done)
5. §1 P5 + §3 — forgetting + cell sweep in parallel
6. §4–§6 — opportunistic
