# Cypha -- Research Status

**Last updated:** 2026-07-19  
**Runtime:** native C++ only -- `cypha_rest`, `cypha_bench_run`, **~160 CTests** *(authoritative tally: `scripts/cypha_native_validate_all.ps1`)*  
**Product:** one type **`cypha::Cypha`** -- classify + regress + latent sample + sequence tokens  
**Planning:** [`CYPHA_BILL_OF_WORK.md`](../CYPHA_BILL_OF_WORK.md), cutover [`reports/ONE_CYPHA_CUTOVER.md`](reports/ONE_CYPHA_CUTOVER.md)

Canonical research journal. Historical subsystem names (CyphaDIF / CyphaLM) appear only where needed for wire-format or archived study pointers. Dated reports -> [`archive/`](archive/README.md).

---

## Quick state summary

| System | Status | Verdict |
|--------|--------|---------|
| **Cypha classifier** | Working, benchmarked | Strong on linear/tabular; nonlinear ceiling mitigated by kernel LLR |
| **Cypha regressor** | Working | Near Ridge on smooth domains; weak on hard nonlinear equations |
| **Native stack (M1-M6 + P7)** | Shipped | Sole runtime; Kernel LLR in `native/src/kernel_memory.cpp`; CTest-gated CI |
| **cypha::accel** | Working | Optional CUDA (`-DCYPHA_ENABLE_CUDA=ON`); ISO C++ thread fallback |
| **Cypha sequence** | Living default **PGM->Wy (U06)** | Hybrid GRIA+LSTM **2.873 BPC** @ 300k is a **historical pin** only (`bench/BASELINE_LOCK.json`) |
| **cypha_som** | Archived, default OFF | Failed experiment -- [`archive/failed_experiments/cypha_som/`](archive/failed_experiments/cypha_som/README.md) |
| **Bench harness** | 17 domains | Configs + reports under `bench/` |
| **cypha_qt_shell / cypha_rest** | Working | `/predict`, `/update`, `/generate`, `/sample`, `/retrieve`, `/sequence/*` |

---

## Current regression pins (2026-07-19)

Restored after MSVC / hybrid-default drift (portable shuffle, D01 golden draws, opt-in n-gram prior, native D16A ARI). Source: [`bench/BASELINE_REPORT.md`](../bench/BASELINE_REPORT.md).

| Domain | Pin | Notes |
|--------|-----|-------|
| **D01** linear 2-class | Cypha **0.9875** vs logistic **0.8875** | Golden synthetic draws + portable shuffle |
| **D01** 4-Gaussian blobs | Cypha **0.8875** | Same harness |
| **D04** Gutenberg @ 8k | BPC **~4.14-4.16** (hybrid bench mode) | Historical hybrid profile; living product sequence is PGM->Wy |
| **D16A** task discovery | **ARI = 1.0** | Native ARI recovery |
| **D17** hybrid @ 300k | **2.873 BPC** (lock) / **2.883** re-run 2026-07-19 | **Historical pin only** -- not the living production spine; Δ=+0.010 within ±0.05 gate |

**Reproduce historical hybrid pin** (must pass `--mode hybrid`; CLI default is `pgm_logits`):

```powershell
$env:CYPHA_BENCH_FULL_CORPUS="1"; $env:CYPHA_BENCH_OVERNIGHT="1"
cyphalm_bench_native --profile d17 --mode hybrid --overnight --n-train 300000 --n-eval 2000 --threads 1 --bench-seed 42
```

Scale check (same flags, seed 42): 5k → ~4.05 BPC; 40k → ~3.42 BPC; 300k → ~2.88 BPC. Keep `use_ngram_count_prior=false` (default).

Living sequence: `Cypha::init_default_sequence` -> U06 / `apply_pgm_logits_recipe`. Bare `CyphaLMConfig` may still default Hybrid for **bench** compatibility.

---

## Benchmark snapshot (17 domains)

Everyday-profile era numbers (2026-05-31) plus later refresh notes. Full tables: [`bench/BASELINE_REPORT.md`](../bench/BASELINE_REPORT.md). Detailed dated writeups: [`archive/reports/`](archive/reports/).

### Classification (highlights)

| Domain | Cypha | Notes |
|--------|-------|-------|
| D01 linear / blobs | **0.9875 / 0.8875** | Current pin (2026-07-19) |
| D06 Go / D07 Poker | 99.5% / 93.3% | Strong / good |
| D08 MNIST HOG | 89.6% | Raw pixels weaker |
| D10A ECG5000 | **90.11%** | Enriched features; see archive D10 reports |
| D16A continual ARI | **1.0** | Perfect routing |

### Regression / OOD / continual

| Area | Verdict |
|------|---------|
| D05 chess R^2 ~0.65 | Near Ridge |
| Cross-domain OOD AUROC ~0.84 | Good |
| D16B shared-model forgetting | Real; EWC modest help -- [`archive/reports/EWC_D16B_SCOPING_2026-07-12.md`](archive/reports/EWC_D16B_SCOPING_2026-07-12.md) |
| D16F isolated models | Zero forgetting by architecture |

### Sequence (D04 / D17)

| Pin | Value | Role |
|-----|-------|------|
| Hybrid D17 @ 300k | **2.873 BPC** | Historical lock -- beats bigram / char-LSTM bench of that era |
| Hybrid D04 @ 8k | **~4.14 BPC** | Bench regression pin |
| Living default | **PGM->Wy (U06)** | Product `predict_next` / `generate` |
| GRIA-only @ 300k | 3.838 BPC | Ablation / archive |

Studies (algorithm, long-range, model class): [`archive/studies/`](archive/studies/). One Cypha sequence notes: [`archive/reports/one_cypha/`](archive/reports/one_cypha/).

---

## Confirmed properties

| Property | Evidence |
|----------|----------|
| Per-task isolation -> no forgetting | D16F |
| Shared-model forgetting is real | D16B |
| OOD AUROC > 0.80 | Cross-domain mean ~0.84 |
| Task routing / discovery | D16A ARI = 1.0 |
| Save/restore fidelity | D16E retention = 1.0 |
| Linear LLR ceiling on XOR | Kernel LLR closes most of gap (~2.7pp to sklearn RBF) |

## Confirmed hard limits

| Limit | Status |
|-------|--------|
| Nonlinear boundaries | Kernel LLR shipped; not default everyday profile |
| Shared-model continual learning | Open; EWC opt-in only |
| Sequence BPC race vs hybrid 2.873 | Hybrid is historical; living work is PGM->Wy quality, not "beat 2.873 as product default" |

---

## Current priorities

1. **PGM->Wy quality** -- living sequence spine under `cypha::Cypha` ([cutover](reports/ONE_CYPHA_CUTOVER.md)).
2. **Kernel LLR / RFF** -- promote or keep opt-in after domain re-bench ([`research/upgrades/NONLINEAR_BOUNDARY.md`](research/upgrades/NONLINEAR_BOUNDARY.md)).
3. **Shared-model continual learning** -- EWC scoping archived; product claim remains isolation (D16F).
4. **Human arXiv upload** -- paper bundle ready ([`CYPHA_BILL_OF_WORK.md`](../CYPHA_BILL_OF_WORK.md)).

Closed / STOP tracks (RPSM Option B, cell sweep, SOM, WikiText BPE vs char pin): see [`research/upgrades/README.md`](research/upgrades/README.md) and [`archive/plans/`](archive/plans/).

---

## Archive map

| Bucket | Path |
|--------|------|
| Policy | [`archive/README.md`](archive/README.md) |
| Dated reports | [`archive/reports/`](archive/reports/) |
| One Cypha sequence notes | [`archive/reports/one_cypha/`](archive/reports/one_cypha/) |
| CyphaLM / multi-view studies | [`archive/studies/`](archive/studies/) |
| Mega plans / roadmaps | [`archive/plans/`](archive/plans/) |
| Failed SOM | [`archive/failed_experiments/cypha_som/`](archive/failed_experiments/cypha_som/README.md) |

Phase-by-phase history (Intelligence Stats Phases 7-59, overnight lock, cell sweep, etc.) lives in those archive files and in [`CHANGELOG.md`](../CHANGELOG.md) -- not duplicated here.

---

## How to reproduce

```bash
cypha_bench_run --list-domains
cypha_bench_run --from-domain 1
cypha_bench_run --domain 4
cypha_bench_run --domain 17
cypha_bench_run --report-only
```

```powershell
powershell -File scripts\cypha_native_validate_all.ps1
```

Parity fixture updates: [`verify/MAINTENANCE.md`](verify/MAINTENANCE.md).

---

*Update the quick-state and regression-pin tables when a milestone or bench pin changes.*
