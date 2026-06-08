# Cypha Tests — Phase 2 (Hebbian / Biochemical Networks)

**Maps:** [`Cypha Tests.txt`](../Cypha%20Tests.txt) Phase 2 → Cypha stack  
**Prerequisite:** Phase 1 complete for CyphaLM — see [`CYPHALM_LONG_RANGE_TESTS.md`](CYPHALM_LONG_RANGE_TESTS.md)

---

## Phase 1 recap (verified)

| Experiment | CyphaLM result | Verdict |
|------------|----------------|---------|
| **1A** Sequential vs shuffled | Block shuffle flat; **char shuffle +4.54 BPC @ 300k hybrid** | Pass @ char level |
| **1A** Field vs zeroed | SSM ablation: gria vs no_ssm ≈ flat @ stream eval | Marginal @ aggregate BPC |
| **1C** Field as context | Warm-up **3.03→2.86**; reset-every-8 **+0.27 BPC** | Pass |

Phase 1 for **CyphaDIF classifier** (fixed-vector tasks) remains in [`docs/reports/DIAGNOSTIC_REPORT.md`](reports/DIAGNOSTIC_REPORT.md).

---

## Phase 2 mapping

| Cypha Tests | Question | Cypha implementation | Status |
|-------------|----------|----------------------|--------|
| **2A** Replace `contrastive_update` with Hebbian | Does Hebbian encoder converge on classification? | `Cypha.py` → `Encoder.contrastive_update`; compare vs external Hebbian rule | **Planned** — needs user biochem rule API |
| **2B** Hebbian encoder → DIF input | Are Hebbian features useful to CyphaDIF? | `cypha_som/hebbian_topology.py`; optional encoder hook | **Planned** |
| **2C** Hebbian lateral field | Richer temporal dynamics vs `A_eff @ h`? | `CellAISSM.sparse_hebbian_update` (`use_sparse_hebbian`) | **Partial** — flag exists, not LM-benchmarked |

---

## 2C — SSM sparse Hebbian (ready to bench)

**Code:** `cypha_lm/temporal/cellai_ssm.py` → `sparse_hebbian_update`  
**Config:** `CyphaLMConfig.use_sparse_hebbian` (default **off** — overhead per step)

**Component study** (`cyphalm_component_study.py`) already sweeps `{spectral, multiscale, hebbian}` on `gria_ngram`; Hebbian alone was neutral @ 40k.

### Proposed sweep (Phase 2C LM)

```powershell
# Toggle sparse Hebbian on hybrid @ 40k (fast)
python cypha_bench/tuning/cyphalm_hebbian_phase2_sweep.py --profile d17 --n-train 40000 --write

# @ 300k if 40k shows ≥0.05 BPC gain
python cypha_bench/tuning/cyphalm_hebbian_phase2_sweep.py --profile d17 --n-train 300000 --write
```

Success criterion: held-out BPC **≥0.05 lower** than hybrid baseline with `use_sparse_hebbian=False`.

**Result @ 40k hybrid (2026-06-08):** `hebbian_on_minus_off_bpc = 0.0` — identical BPC (3.346). SSM Hebbian updates do not affect hybrid LM eval when LSTM head dominates. **No gain @ 40k; skip 300k unless GRIA-only path retested.**

---

## 2A / 2B — Integration points

### 2A — Encoder swap (CyphaDIF)

1. Implement `HebbianEncoder.update(f, h, label)` matching `contrastive_update` signature.
2. Bench D01/D03/D09 with `CyphaDIF(encoder="hebbian")` vs default.
3. Metrics: accuracy, train steps to 90%, wall time.

### 2B — Frozen Hebbian front-end

1. Train multi-layer Hebbian net on token/id streams (or sentence embeddings for D09).
2. Freeze; feed activations into `CyphaDIF` field input instead of RFF/Vector encoder.
3. Compare OOD AUROC + accuracy vs RFF baseline.

**Branch A from Cypha Tests.txt** (CyphaDIF on frozen transformer embeddings) is the production path for NLP routing — separate from char-LM but shares 2B methodology.

---

## Phase 3 preview (sequence generation)

| Experiment | CyphaLM path |
|------------|--------------|
| **3A** Next-state prediction | Field → linear next-embed head (not yet implemented) |
| **3B** Greedy generation | `CyphaLM.generate()` exists; quality gated by BPC |
| **3C** Perplexity | D17/D04 BPC vs n-gram / LSTM baselines — **hybrid @ 2.87 BPC @ 300k** |

Phase 3 minimal LM is **done** via hybrid + char-LSTM head. Frontier scale (Branch C in Cypha Tests.txt) remains a separate transformer pretrain track.

---

## Model-class C1 — `char_lstm` mode

**Shipped:** `context_mode=char_lstm` — LSTM-only inside `cypha_lm` (no GRIA/SSM/DIF path).

```powershell
python cypha_bench/tuning/cyphalm_hybrid_lstm_sweep.py --cells char_lstm --profile d17 --n-train 300000 --write --out cypha_bench/config/cyphalm_char_lstm_300k.json
```

Compare to hybrid and bench `char_lstm_baseline_bpc` to quantify GRIA path value.

**Result @ 300k WikiText (2026-06-08):** `char_lstm` **2.876 BPC** vs hybrid **2.873** (Δ +0.003) vs bench baseline **2.979**. In-package LSTM-only matches hybrid — GRIA path contributes ~0.003 BPC at convergence when blend is ~99.6% LSTM.

---

## References

- Phase 1 LM: [`CYPHALM_LONG_RANGE_TESTS.md`](CYPHALM_LONG_RANGE_TESTS.md)
- Model class C2: [`CYPHALM_MODEL_CLASS_RESEARCH.md`](CYPHALM_MODEL_CLASS_RESEARCH.md)
- SSM Hebbian: [`cypha_lm/temporal/cellai_ssm.py`](../cypha_lm/temporal/cellai_ssm.py)
- CyphaDIF encoder: [`Cypha.py`](../Cypha.py) `contrastive_update`
