# Product adjust closeout — bounded profile wave (2026-07-17)

**Author:** Odin Loch (agent closeout)  
**Scope:** Summarize what landed in the bounded **product/profile adjust** wave vs what remains blocked on overnight or multi-day research.  
**Repo HEAD at closeout:** `4d5dbbf` (code); this doc is docs-only follow-up.  
**Overnight:** Read-only health — H18 @ 21/25 in flight; see [`OVERNIGHT_HEALTH_2026-07-17.md`](OVERNIGHT_HEALTH_2026-07-17.md) §8.

---

## Executive summary

The bounded wave shipped **infer/train floor profiling**, **Optimality P0–9** (done / opt-in / no-go mix), **REST multi-model + Studio Web chat**, **Addendum-2 metrics (MC/MR/MG/MS)**, **ONNX/GGUF export**, **parallel `score_matrix`**, **Qt compare polish**, and **federated merge golden parity** — all without touching `build_math`, `build_deff`, `BASELINE_*`, or the live overnight cell-sweep.

**Blocked on overnight / multi-day research:** 300k production cell-sweep completion (H19–H22 remain), baseline lock refresh + d27–d38 production gates, RFF default promotion, P3 XOR GMM, RPSM zero-BPTT gap, hidden=512 @ 300k D_eff Phase 3, math-integration production certificate, and release publish.

---

## Done (shipped this wave)

### Train / infer floor

| Item | Status | Evidence |
|------|--------|----------|
| D17 train throughput floor (Parts 1–7) | [x] Reached | ~252 chars/sec MSVC; compute-bound on LSTM/BPTT — [`PERFORMANCE_PROFILE_2026-07-12.md`](PERFORMANCE_PROFILE_2026-07-12.md), [`DEAD_WORK_AUDIT_2026-07-17.md`](DEAD_WORK_AUDIT_2026-07-17.md) |
| DIF infer `score_matrix` scratch reuse | [x] ~48% win | `4d3afa2` — [`INFER_LATENCY_PROFILE_2026-07-17.md`](INFER_LATENCY_PROFILE_2026-07-17.md) |
| CyphaLM `/generate` latency floor profiled | [x] Documented | ~0.6–1.2 ms/tok quiet; no further free skips — [`GENERATE_LATENCY_2026-07-17.md`](GENERATE_LATENCY_2026-07-17.md) |
| Production `n_train` floor (300k) | [x] Wired | `bench_full_n_train()` default 300k; overnight sweep uses it when `CYPHA_BENCH_OVERNIGHT=1` |
| Real-data D03 profile pass | [x] Logged | Iris CSV ingest — [`REAL_DATA_PROFILE_2026-07-17.md`](REAL_DATA_PROFILE_2026-07-17.md) |

### Optimality P0–9

| Phase | Title | Verdict | Commit / report |
|-------|-------|---------|-----------------|
| P0 | Regression net | [x] Done | `4133054` |
| P1 | EM keystone | [x] Done | `31bbb0c`/`7a07f8b` |
| P2 | MoE + EM | [x] Done | `de4fa16` |
| P3 | Per-class GMM | [~] Opt-in; XOR no-go (~51%) | `1b59f3e` — [`OPTIMALITY_PHASE3_2026-07-17.md`](OPTIMALITY_PHASE3_2026-07-17.md) |
| P4 | BMA over Δk | [~] Opt-in shipped | `33125b8` — [`OPTIMALITY_PHASE4_2026-07-17.md`](OPTIMALITY_PHASE4_2026-07-17.md) |
| P5 | Leverage Nyström + SORF | [x] Shipped opt-in | [`OPTIMALITY_PHASE5_2026-07-17.md`](OPTIMALITY_PHASE5_2026-07-17.md) |
| P6 | Variational IB | [~] Opt-in | `f0ea334` — [`OPTIMALITY_PHASE6_2026-07-17.md`](OPTIMALITY_PHASE6_2026-07-17.md) |
| P7 | Score matching | [~] Opt-in; LUT kept | `f19e167` — [`OPTIMALITY_PHASE7_2026-07-17.md`](OPTIMALITY_PHASE7_2026-07-17.md) |
| P8 | RB audit | [x] No-go | `322cb68` — [`OPTIMALITY_PHASE8_2026-07-17.md`](OPTIMALITY_PHASE8_2026-07-17.md) |
| P9 | CriticalityVector | [x] REST/report only | `c759e72` — [`OPTIMALITY_PHASE9_2026-07-17.md`](OPTIMALITY_PHASE9_2026-07-17.md) |

Canonical plan: [`CYPHA_OPTIMALITY_PLAN.md`](../../CYPHA_OPTIMALITY_PLAN.md).

### REST

| Item | Status | Evidence |
|------|--------|----------|
| Multi-model in-process map | [x] Shipped | `e3a5b63` — [`MULTI_MODEL_REST_2026-07-17.md`](MULTI_MODEL_REST_2026-07-17.md) |
| Curriculum + `/uncertainty-rank` | [x] Shipped | `curriculum.hpp`, CTest `native_rest_uncertainty_rank` |
| Legacy sigmoid world-gate | [x] Removed | API v2 `38620b4` |

### Metrics — MC / MR / MG / MS

| ID | Metric | Status | Report |
|----|--------|--------|--------|
| MC2 | ECE + mean confidence | [x] | [`GENERAL_METRICS_MC2_MS1_2026-07-17.md`](GENERAL_METRICS_MC2_MS1_2026-07-17.md) |
| MS1 | Train vs held-out gap | [x] | same |
| MR1 | CRPS | [x] | [`GENERAL_METRICS_MR1_MR2_2026-07-17.md`](GENERAL_METRICS_MR1_MR2_2026-07-17.md) |
| MR2 | 90% interval coverage | [x] | same |
| MC5 | BPC vs `n_train` curve | [x] | [`SAMPLE_EFFICIENCY_CURVE_2026-07-17.md`](SAMPLE_EFFICIENCY_CURVE_2026-07-17.md) |
| MG3 | Needle-in-haystack recall | [x] | [`MG3_NEEDLE_HAYSTACK_2026-07-17.md`](MG3_NEEDLE_HAYSTACK_2026-07-17.md) |
| MG4 | Memorization canary | [x] | [`MG4_MEMORIZATION_CANARY_2026-07-17.md`](MG4_MEMORIZATION_CANARY_2026-07-17.md) |
| MG5 | Sample-efficiency JSON format | [x] | same as MC5 |

### ONNX / GGUF export

| Format | Status | Evidence |
|--------|--------|----------|
| ONNX encode→LLR→softmax | [x] Structural smoke | `1cbdd8c` — [`ONNX_EXPORT_2026-07-17.md`](ONNX_EXPORT_2026-07-17.md) |
| GGUF v3 tensor pack | [x] Partial (custom `cypha-dif` arch) | `dad723d` — [`GGUF_EXPORT_2026-07-17.md`](GGUF_EXPORT_2026-07-17.md) |

### Parallel score

| Item | Status | Evidence |
|------|--------|----------|
| OpenMP row-parallel `batched_llr_gemm` | [x] ~3.4× @ n=256 | `c788f5f` — [`PARALLEL_SCORE_2026-07-17.md`](PARALLEL_SCORE_2026-07-17.md) |

### Web / Qt

| Item | Status | Evidence |
|------|--------|----------|
| Studio Web — CyphaLM chat pane (SSE) | [x] Shipped | `b706647` — [`WEB_UI_GENERATE_2026-07-17.md`](WEB_UI_GENERATE_2026-07-17.md) |
| Qt compare-run statistics table | [x] Shipped | `7578f13` — [`QT_POLISH_2026-07-17.md`](QT_POLISH_2026-07-17.md) |

### Federated

| Item | Status | Evidence |
|------|--------|----------|
| Merge golden parity CTest | [x] Shipped | `d1a9bf1` — [`FEDERATED_SLICE_2026-07-17.md`](FEDERATED_SLICE_2026-07-17.md) |
| TLS smoke | [~] Skip without OpenSSL build | existing |

### Other bounded landings

| Item | Status | Evidence |
|------|--------|----------|
| §0.5 BPC pin **2.873** | [x] | [`BASELINE_PIN_CANONICAL_2026-07-17.md`](BASELINE_PIN_CANONICAL_2026-07-17.md) |
| B3 position weights / B4 bilinear | [x] Opt-in, default OFF | BoW §4 |
| P2 auto-gamma RFF default (d≤30) | [x] Confirmed | `0280d4a` |
| D16 multi-view early-stop policy | [x] Documented | [`D16_MULTIVIEW_POLICY_2026-07-17.md`](D16_MULTIVIEW_POLICY_2026-07-17.md) |
| CUDA policy local-only | [x] Accepted | `b7a26e5`, [`ACCEL_CUDA.md`](../native/ACCEL_CUDA.md) |

---

## Blocked — overnight

| Blocker | State @ closeout | Unblocks |
|---------|------------------|----------|
| **300k cell-sweep H19–H22** | H18 @ 21/25 in flight (progress line 2026-07-17T11:45:54Z); PID 47108 active; poll `processes=4` | H22 child exit + `write_overnight_artifacts` |
| **`poll_and_finalize_overnight.ps1 -AutoCommit`** | Waiting on sweep | After H22 + artifact flush |
| **d27–d38 production gates** | `pending_production` | Baseline lock refresh post-overnight |
| **d38 merge (115→116 CTests)** | Blocked on 0.1–0.2 | Lock lands |
| **Cell-sweep `summary.csv` vs baselines** | No H15–H17 JSON yet (bulk write at end) | Sweep completion |
| **Math-integration production certificate (d53–d58)** | Needs 300k overnight with `-MathIntegration` | Same overnight chain |

Overnight health: [`OVERNIGHT_HEALTH_2026-07-17.md`](OVERNIGHT_HEALTH_2026-07-17.md).

---

## Blocked — multi-day research

| Area | Why blocked | Next step |
|------|-------------|-----------|
| **P3 class GMM default-on / XOR ≥75%** | XOR ~51% at FAST; needs different approach or kernels | RFF latent promotion (P1/P2) or new kernel path |
| **RPSM zero-BPTT training gap** | Cheap hypotheses exhausted (§13–§14) | BPTT in training loop — multi-day |
| **D17 < 2.873 via RPSM** | Not met at any tier tried | Depends on BPTT fix |
| **Hidden=512 @ 300k D_eff (Phase 3)** | Contention with overnight + ~16h wall under load | Schedule after sweep; see [`HIDDEN_DIM_SCALE_PLAN.md`](HIDDEN_DIM_SCALE_PLAN.md) |
| **EWC / shared-model CL (P5)** | Best 0.135→0.108 @ λ=2.0; not solved | Routing redesign or accept isolation-only |
| **Multi-view CyphaDIF (P4 Step 7)** | D16 16G regression documented | DIF-V3 replay-interleave |
| **κ-targeting / math-integration ablations** | Flat at FAST/5k; sign flip at 500 vs 5k | Production-scale grid post-overnight |
| **Federated TLS + coordinator HTTP** | Golden merge only; TLS skipped without OpenSSL | Optional infra pass |
| **Qt shell full polish / Web partial** | Compare stats only this wave | Manual hardening backlog |
| **Paper submit (2027 Q1)** | Narrative reconciled; results pending production lock | After overnight + lock |

---

## Suggested execution order (post-closeout)

> **Wave 2 update:** See [`PRODUCT_ADJUST_WAVE2_2026-07-17.md`](PRODUCT_ADJUST_WAVE2_2026-07-17.md) for post-closeout landings. Finalize command + prerequisites: [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md).

1. **Wait** for H19–H22 overnight (do not restart, do not kill).
2. **`poll_and_finalize_overnight.ps1 -AutoCommit -BuildDir native/build_math`** when H22 + artifact flush complete — exact checklist in [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md).
3. **`aggregate_cell_sweep_summary.ps1`** on flushed variant JSON ([`CELL_SWEEP_SUMMARY_TOOL_2026-07-17.md`](CELL_SWEEP_SUMMARY_TOOL_2026-07-17.md) — shipped `a04af20`).
4. Merge **d38**; run production validate env hooks.
5. Re-evaluate **P3 XOR GMM** using RFF latent profile — latent exploratory default promoted `beacef3` (`xor_pair` prod default unchanged); see [`RFF_LATENT_PROMOTE_2026-07-17.md`](RFF_LATENT_PROMOTE_2026-07-17.md).
6. Schedule **hidden=512 @ 300k** on uncontended machine.
7. RPSM **BPTT-in-training** research track in parallel with cell-sweep analysis.

---

## Cross-links

- Wave 2 closeout (DONE): [`PRODUCT_ADJUST_WAVE2_2026-07-17.md`](PRODUCT_ADJUST_WAVE2_2026-07-17.md)
- Finalize prep (post-H22): [`FINALIZE_PREP_2026-07-17.md`](FINALIZE_PREP_2026-07-17.md)
- Master task list: [`CYPHA_BILL_OF_WORK.md`](../../CYPHA_BILL_OF_WORK.md)
- Research journal: [`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md)
- Optimality plan: [`CYPHA_OPTIMALITY_PLAN.md`](../../CYPHA_OPTIMALITY_PLAN.md)
