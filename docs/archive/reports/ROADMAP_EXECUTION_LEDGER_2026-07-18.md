# Roadmap execution ledger (2026-07-18)

**Living pointer:** [`CYPHA_BILL_OF_WORK.md`](../../CYPHA_BILL_OF_WORK.md) · Optimality: [`CYPHA_OPTIMALITY_PLAN.md`](../../CYPHA_OPTIMALITY_PLAN.md)  
**Draft roadmaps:** [`QUALITY_RESEARCH_ROADMAP_2026-07-18.md`](QUALITY_RESEARCH_ROADMAP_2026-07-18.md) · [`PERF_RESEARCH_ROADMAP_2026-07-18.md`](PERF_RESEARCH_ROADMAP_2026-07-18.md)  
**Recipe evidence:** [`QUALITY_RECIPE_WAVE2_2026-07-18.md`](QUALITY_RECIPE_WAVE2_2026-07-18.md)

Status keys: **Done** · **Wave1** · **Wave2** · **Deferred** · **STOP**

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
| §1.1 LSTM BPTT>1 | **Done** (Wave1) | Opt-in; default **1** |
| §1.2 Adam + grad clip | **Done** (Wave1) | Opt-in; default SGD / clip 0 |
| §1.2b AdamW + LR schedule | **Done** (Wave2) | Opt-in `lstm_weight_decay`, warmup/cosine; default OFF |
| §1.3 Full WikiText multi-epoch | Deferred | After 300k recipe decision |
| §1.4 LayerNorm / dropout | Deferred | — |
| §1.5 Classic LSTM init | **Done** (Wave1) | Opt-in `lstm_init=classic` |
| §1.6 Batch default | Deferred | `CyphaLMBatch` stays side path |
| §1.7 25-cell re-sweep | Deferred | After recipe promote |
| Recipe promote (default flip) | **HOLD** | 20k candidate wins at Adam **lr=1e-3** (3.365 vs 3.475); Adam at 0.05 collapses; awaiting 300k A/B |
| §2.x / MiniRocket / MLP | Deferred | — |
| §4.1a Kernel default-on | **STOP** | Keep opt-in |

---

## Perf roadmap

| Item | Status | Notes |
|------|--------|-------|
| §0.4 Throughput lock | **Done** (Wave1 scaffold) | Local smoke; no CI hard-fail |
| §1.1 native arch / §1.2 LTO | **Done** (Wave1) | CMake options |
| §3.1 score_matrix inv_v fold | **Done** (Wave2) | CPU precompute `D_scaled`; CTest `native_score_matrix_inv_v_fold_smoke` |
| §2.2 full GEMM score_matrix | Deferred | Fold is step 1 toward GEMM |
| §4.1 `g_mu` narrowing | **Done** (Wave1) | CPU lock-free after init |
| §5 GPU train | **STOP** / gap | — |
| REST dynamic batching | Deferred | — |

---

## Explicit STOPs (do not reopen)

RPSM BPTT/Small · GMM XOR default-on · BPE@300k over char · D14 kernel routing default-on · sticky CL promote · eigenvalue `D_eff` default-on · GPU full training · recipe-as-default before 300k evidence.

---

## Wave 1 + Wave 2 acceptance

- Default D17 / goldens / CI green with recipe flags OFF
- Wave1/Wave2 recipe smokes green with flags ON
- Accel inv_v fold smoke green
- Ledger matches tip; promote HOLD pending 300k
