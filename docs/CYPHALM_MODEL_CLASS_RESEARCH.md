# CyphaLM Model-Class Research — Char-LSTM & Hybrids

**Status:** M2 complete — **hybrid GRIA+LSTM is default** (D17 + D04 @ 300k)  
**Last updated:** 2026-06-08  
**Related:** [`CYPHALM_UPGRADE_V2.md`](CYPHALM_UPGRADE_V2.md), [`CYPHALM_LONG_RANGE_TESTS.md`](CYPHALM_LONG_RANGE_TESTS.md)

---

## Summary (2026-06-08)

| Model | D17 @ 300k | D04 Moby Dick @ 300k | vs bigram |
|-------|------------|----------------------|-----------|
| GRIA stack (`gria_ngram`) | 3.838 | 3.965 | +0.27 / +0.33 |
| Char-LSTM baseline | 2.979 | 3.047 | −0.50 / −0.59 |
| **Hybrid (`hybrid_gria_lstm`)** | **2.873** | **2.993** (bench) / **2.859** (sweep) | **−0.61 / −0.64** |
| **Char-LSTM in-package (`char_lstm`)** | **2.876** | — | **−0.60** |

Blend learns ~**99.6% LSTM**. Cypha Tests **1A passes @ char shuffle** (+4.54 BPC @ 300k); block shuffle remains flat.

---

## Why a second model class?

Char-LSTM beat the GRIA-only CyphaLM stack by **~0.74 BPC @ 300k** while also beating bigram. The gap was larger than the CyphaLM-vs-bigram gap — recurrent depth was the missing piece, not more GRIA sweeps.

---

## Candidate architectures

| ID | Description | Verdict |
|----|-------------|---------|
| **C1** | Char-LSTM head only (drop GRIA) | **Shipped** — `char_lstm` @ 300k **2.876** ≈ hybrid |
| **C2** | Dual head GRIA + LSTM with online blend | **Default profile** |
| **C3** | LSTM on embed history → GRIA | Not evaluated; C2 sufficient |
| **C4** | Promote bench LSTM into `cypha_lm` | **Done** — `char_lstm_head.py` |

---

## Implementation

| Module | Role |
|--------|------|
| `cypha_lm/model/char_lstm_head.py` | NumPy LSTM head + blend logit |
| `cypha_lm/model/cypha_lm.py` | `context_mode=hybrid_gria_lstm` |
| `bench/tuning/cyphalm_hybrid_lstm_sweep.py` | GRIA vs hybrid sweep |
| D17 **17J_hybrid_lstm** | Bench experiment |

Config fields: `lstm_hidden`, `lstm_lr`, `hybrid_blend_learnable`, `hybrid_blend_lr`.

---

## Phase results

### M1 — Char-LSTM extended baseline

Artifact: `cyphalm_char_lstm_extended.json`

| n_train | Char-LSTM BPC | vs bigram |
|---------|---------------|-----------|
| 40k | 3.615 | −0.30 |
| 300k | **3.098** | −0.47 |

### M2 — Dual head @ 300k

Artifacts: `cyphalm_hybrid_lstm_300k.json`, `cyphalm_hybrid_lstm_d04_300k.json`

| Cell | WikiText BPC | Moby Dick BPC |
|------|--------------|---------------|
| `gria_ngram` | 3.842 | 3.965 |
| **hybrid** | **2.870** | **2.859** |
| Hybrid − GRIA | **−0.972** | **−1.105** |

Profiles: `cyphalm_d17_wikitext.json`, `cyphalm_d04_gutenberg.json`, `cyphalm_llm.json`.

### M3 — Integration decision

**Ship hybrid as default.** GRIA-only path retained for SSM ablation / long-range probes.

Optional future work:
- **`context_mode=char_lstm`** — LSTM-only shipped; bench via hybrid sweep `--cells char_lstm`
- C3 embed-LSTM fusion before GRIA
- Per-token blend weights
- **Cypha Tests Phase 2** (Hebbian) — see [`CYPHA_TESTS_PHASE2.md`](CYPHA_TESTS_PHASE2.md)

---

## Parallel with Upgrade V2

| Track | Result |
|-------|--------|
| V2 Track A (learnable views) | Neutral @ 300k — keep fixed views |
| V2 Track B (gated fusion) | +0.12 BPC worse @ 40k — keep sum fusion |
| **Model class C2** | **−0.97 BPC vs GRIA @ 300k** |

---

## Commands

```powershell
# Hybrid sweep (WikiText or Gutenberg)
cypha_tune_run --config bench/config/cyphalm_hybrid_lstm_sweep.py --profile d17 --n-train 300000 --write
cypha_tune_run --config bench/config/cyphalm_hybrid_lstm_sweep.py --profile d04 --corpus gutenberg --n-train 300000 --write

# D04 full bench refresh (figures + tables)
cypha_tune_run --config bench/config/run_d04_hybrid_refresh.py

# Char-level 1A probe
cypha_tune_run --config bench/config/cyphalm_long_range_suite.py --n-train 300000 --skip-ablation --write --out bench/config/cyphalm_long_range_300k_char1a.json
```

---

## References

- Bench char-LSTM: `bench/adapters/char_lstm_baseline.py`
- Long-range / 1A: `bench/adapters/cyphalm_long_range.py`
- Branch A embeddings: [`CYPHA_BRANCH_A_EMBEDDINGS.md`](CYPHA_BRANCH_A_EMBEDDINGS.md)
- Findings: [`FINDINGS_CYPHALM_TRAINING.md`](FINDINGS_CYPHALM_TRAINING.md)
