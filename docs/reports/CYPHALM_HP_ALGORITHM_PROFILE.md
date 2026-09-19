# CyphaLM hp integration profile

**Date:** 2026-09-19  
**Branch:** `cursor/bit-tree-gate24-eval-9d44`  
**Upstream canonical profile:** [CompressionAlgorithm `docs/reports/HP_ALGORITHM_PROFILE.md`](https://github.com/odin-loki/CompressionAlgorithm/blob/master/docs/reports/HP_ALGORITHM_PROFILE.md) (PR #3, commit `cb8f938`)

This document covers **Cypha-specific** integration, SKU builds, and enwik8.8mb measurements. For hp codec architecture (experts → MixerNet → 3×APM → binary coder), redundancy `--profile` semantics, and upstream RECORD bars, use the upstream report — **do not treat Cypha docs as a second source of CPU stage splits** (upstream has no per-stage wall timers).

---

## Reference bars (CompressionAlgorithm RECORD, same corpus)

Corpus: `bench/data/enwik8/enwik8.8mb` — SHA256 `09f6dd7241a8ae21edfd6762f3c6712a1fd02f7f322c5e77cab8bb88f292ee8e`

| Config | Archive bytes | BPC | Peak RSS | Wall (upstream VM) |
|--------|---------------|-----|----------|-------------------|
| v82-era flags, `SLOT_MAX=24`, mem 22 | **1,685,481** | **1.607** | **1.54 GB** | **502 s** |
| RECORD v82 champ, `SLOT_MAX=35`, mem 22 | **1,689,157** | **1.610** | ~15–27 GB | *(not re-run @ 15 GiB)* |

Source: upstream [`HP_ALGORITHM_PROFILE.md` §3.1](https://github.com/odin-loki/CompressionAlgorithm/blob/master/docs/reports/HP_ALGORITHM_PROFILE.md).

---

## Cypha measured enwik8.8mb (this VM, vendored `native/third_party/hp`)

**VM:** Linux 6.12.94+, Intel Xeon 4 cores, **15 GiB RAM**, g++ 13.3.0

| SKU | v78 flags | `SLOT_MAX` | XSIMD | hp archive BPC | Cypha observe BPC | Δ obs−arch | RT SHA |
|-----|-----------|------------|-------|----------------|-------------------|------------|--------|
| **light** | 0/78 | 24 | OFF | **1.721362** (1,804,979 B) | **1.721331** | −0.000031 | PASS |
| **gate24** | **78/78** | 24 | ON | **1.611759** (1,690,052 B) | **1.611729** | −0.000030 | PASS |
| **champ** | **78/78** | 35 | ON | — | — | — | **OOM** (exit 137) |

### vs upstream reference

| SKU | Δ archive vs s24 ref (1.607) | Δ archive vs champ ref (1.610) |
|-----|------------------------------|--------------------------------|
| gate24 (Cypha vendored hp) | **+4,571 B** (+0.0048 BPC) | **+895 B** (+0.0009 BPC) |
| light | +119,498 B (+0.114 BPC) | — |

**Vendored hp drift:** Cypha's pinned `native/third_party/hp` produces **1,690,052 B** on gate24 where upstream CompressionAlgorithm master produces **1,685,481 B** with the same v82-era flag recipe (§4.4 upstream). Observe≡archive holds within Cypha; the gap to **1.607** is vendored-tree version drift, not a Cypha integration math bug.

---

## Cypha integration map

```
CyphaLMModel::eval_bpc / eval_bpc_compress_equivalent
    → HpSequenceBackend::observe_next_byte  (8 bits, live pred_, update on truth)
    ≡ hp archive BPC @ same compile flags

CyphaLMModel::predict_next / generate
    → HpSequenceBackend::next_byte_log_probs
        light:  bit-tree DFS (checkpoint pool, 9× Predictor slots)
        gate24/champ: legacy 256-clone (pool OOMs on v78 table sizes)
```

**BPC metric:** bit-serial observe only. Full-vocab `predict_next` latency is separate (see below).

### Redundancy `--profile` (gate24, vendored hp)

Not CPU stages — CTW-style bit-cost decomposition per upstream §2.1:

```
total spent       1690105 B   1.611 bpc
  model redundancy (mixer vs best expert): +1531856 B
  coding redundancy (APM+coder vs mixer):  -25718 B
  parameter share (sparse contexts):       47%
```

Full stderr: [`enwik_sku_profiles/gate24_redundancy_profile.txt`](enwik_sku_profiles/gate24_redundancy_profile.txt).

---

## RAM (Cypha `HpSequenceBackend`)

| SKU | Construct RSS (measured) | Notes |
|-----|--------------------------|-------|
| light | ~4.3 GB | `pred_` + `scratch_` + 9 DFS checkpoints |
| gate24 | ~1.5–2 GB class (hp CLI) | No DFS pool; legacy 256-clone for scoring |
| champ | OOM @ 15 GiB | upstream RECORD: ~15 GB @ SLOT_MAX=35 |

---

## Throughput & bit-tree honesty

enwik8.8mb, light SKU, 64-byte warm context:

| Path | µs/call | vs legacy |
|------|---------|-----------|
| bit-tree `next_byte_log_probs` | **~37M** | **~1.9× slower** |
| legacy 256-clone | **~19M** | 1.00× |
| observe `eval_bpc` | **~44k B/s** | correct BPC path |

Bit-tree matches legacy exactly (`hp_bit_tree_smoke` Δ=0) but is **not** a latency win. **Next step:** hp `undo` stack to replace checkpoint pool (see [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md)).

---

## SKU builds

```bash
# gate24 (quality screen, fits 15 GiB)
cmake -S native -B native/build-gate24 -DCMAKE_BUILD_TYPE=Release \
  -DCYPHA_HP_PROFILE=gate24 -DCMAKE_CXX_COMPILER=g++
cmake --build native/build-gate24 --target cyphalm_hp_sku_measure -j$(nproc)

# champ (needs ≥32 GiB for enwik8.8mb)
cmake ... -DCYPHA_HP_PROFILE=champ
```

Flag diff: `python3 scripts/hp_v78_flag_diff.py` → gate24/champ **78/78**.

---

## Related

- [`CYPHALM_LLM_EVAL.md`](CYPHALM_LLM_EVAL.md) — headline enwik table  
- [Upstream `HP_ALGORITHM_PROFILE.md`](https://github.com/odin-loki/CompressionAlgorithm/blob/master/docs/reports/HP_ALGORITHM_PROFILE.md) — canonical hp architecture & RECORD bars  
- [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md) — observe≡archive parity
