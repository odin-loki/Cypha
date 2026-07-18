# Roadmap execution ledger (2026-07-18)

**Living pointer:** [`CYPHA_BILL_OF_WORK.md`](../../CYPHA_BILL_OF_WORK.md) · Optimality: [`CYPHA_OPTIMALITY_PLAN.md`](../../CYPHA_OPTIMALITY_PLAN.md)  
**Draft roadmaps:** [`QUALITY_RESEARCH_ROADMAP_2026-07-18.md`](QUALITY_RESEARCH_ROADMAP_2026-07-18.md) · [`PERF_RESEARCH_ROADMAP_2026-07-18.md`](PERF_RESEARCH_ROADMAP_2026-07-18.md)

Status keys: **Done** · **Wave1** · **Deferred** · **STOP**

---

## Optimality (P0–P9)

| Phase | Status | Evidence |
|-------|--------|----------|
| P0–P2, P5, P8 | Done | Goldens / EM / MoE / leverage-Nyström / RB audit |
| P3 GMM XOR default-on | **STOP** | REJECT ~50.5%; keep `use_class_gmm` OFF |
| P4 / P6 / P7 | Opt-in OFF | BMA / IB / score-match shipped; no promote |
| P9 mid-estimators | Deferred | Hot gauges + REST shipped; session extras / mid estimators open |
| Overnight H16/19/25 | Done (closed) | Cell sweep closed 2026-07-18; H15@300k = 5.262 no-promote |

---

## Quality roadmap

| Item | Status | Notes |
|------|--------|-------|
| §1.1 LSTM BPTT>1 | **Done** (Wave1) | Opt-in `lstm_bptt_steps` / `CYPHA_LSTM_BPTT` / `--bptt-lstm`; default **1** (pin path unchanged) |
| §1.2 Adam + grad clip | **Done** (Wave1) | Opt-in `lstm_optim=adam`, `lstm_grad_clip`; default SGD / clip 0 |
| §1.3 Full WikiText multi-epoch | Deferred | Later promote wave |
| §1.4 LayerNorm / dropout | Deferred | — |
| §1.5 Classic LSTM init | **Done** (Wave1) | Opt-in `lstm_init=classic` (orthogonal Wh, forget bias +1) |
| §1.6 Batch default | Deferred | `CyphaLMBatch` stays side path |
| §1.7 25-cell re-sweep | Deferred | After recipe promote |
| §2.x capacity / xLSTM / kNN-LM | Deferred | — |
| §4.1a Kernel default-on | **STOP** (D14 routing) | Keep opt-in |
| §4.1b MLP encoder | Deferred | — |
| MiniRocket / XGBoost | Deferred | — |

---

## Perf roadmap

| Item | Status | Notes |
|------|--------|-------|
| §0.1–0.3 profiler / noise | Deferred | Host tooling |
| §0.4 Throughput lock | **Done** (Wave1 scaffold) | `scripts/update_throughput_lock.ps1` + `bench/THROUGHPUT_LOCK.json`; local smoke only (no CI hard-fail) |
| §1.1 native arch | **Done** (Wave1) | `CYPHA_NATIVE_ARCH` CMake option |
| §1.2 LTO | **Done** (Wave1) | `CYPHA_ENABLE_LTO` CMake option |
| §1.3–1.6 PGO / mimalloc / bake-off | Deferred | — |
| §2.1 fp32 typedef | Deferred | — |
| §2.2 GEMM score_matrix | Deferred | — |
| §4.1 `g_mu` narrowing | **Done** (Wave1) | CPU hot path lock-free after init; CUDA still locked |
| §5 GPU train | **STOP** / gap | Documented; not this wave |
| REST dynamic batching | Deferred | — |

---

## Explicit STOPs (do not reopen)

RPSM BPTT/Small · GMM XOR default-on · BPE@300k over char · D14 kernel routing default-on · sticky CL promote · eigenvalue `D_eff` default-on · GPU full training.

---

## Wave 1 acceptance

- Default D17 / goldens / CI green with recipe flags OFF
- Recipe smoke CTest green with flags ON
- Accel CPU path unlocked after init; CUDA smokes unchanged when available
- This ledger + Optimality header match tip reality
