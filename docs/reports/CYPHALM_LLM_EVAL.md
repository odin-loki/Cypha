# CyphaLM LLM Eval — enwik8.8mb SKU measurements

**Date:** 2026-09-19  
**Branch:** `cursor/bit-tree-gate24-eval-9d44`  
**Harness:** `native/tools/cyphalm_hp_sku_measure.cpp`, `scripts/measure_enwik_skus.sh`, `scripts/hp_v78_flag_diff.py`  
**Corpus:** `bench/data/enwik8/enwik8.8mb` (8,388,608 B)

**Win metric:** enwik8.8mb archive + bit-serial observe BPC vs user **1.610906** (champ) / **1.611759** (gate24). WikiText numbers are secondary only.

---

## Headline — enwik8.8mb BPC vs user bar

Reference bars (CompressionAlgorithm encyclopedia / re-measured hp CLI):

| Bar | Archive bytes | BPC |
|-----|---------------|-----|
| **User champ (v82)** | 1,689,157 | **1.610906** |
| **gate24 screen** | 1,690,052 | **1.611759** |

Measured on this VM (g++ 13.3, 4-core Xeon, **15 GiB RAM**):

| SKU | v78 flags | `SLOT_MAX` | XSIMD | hp archive BPC | Cypha observe BPC | Δ obs−arch | RT SHA | Status |
|-----|-----------|------------|-------|----------------|-------------------|------------|--------|--------|
| **light** | 0/78 | 24 | OFF | **1.721362** (1,804,979 B) | **1.721331** | +0.000031 | PASS | OK |
| **gate24** | **78/78** | 24 | ON | **1.611759** (1,690,052 B) | **1.611729** | +0.000030 | PASS | OK |
| **champ** | **78/78** | 35 | ON | — | — | — | — | **OOM** (exit 137, ~5–11 s) |

| SKU | Δ observe vs champ (1.610906) | Δ observe vs gate24 ref |
|-----|------------------------------|-------------------------|
| light | **+0.110425** | +0.109572 |
| gate24 | **+0.000823** | **−0.000030** |
| champ | *not measured* | *not measured* |

**Conclusion:** gate24 matches the PLAN 8 MB screen bar within **0.001 BPC** of champ reference. light is **~0.11 BPC** worse — expected (0/78 v78 flags). champ requires **≥32 GiB** RAM on this workload; not completed here.

---

## Flag parity (78/78)

```bash
python3 scripts/hp_v78_flag_diff.py
# light: 0/78   gate24: 78/78   champ: 78/78
# HP_SLOT_MAX: light=24  gate24=24  champ=35  v82=35
```

gate24 and champ use full `v78_flags.ps1` + `HP_XSIMD=1` + `-msse4.1` (v82 recipe). light uses `features.hpp` defaults only.

---

## Timing (enwik8.8mb, measured)

| SKU | hp `c` wall | Cypha observe wall | Observe throughput |
|-----|-------------|--------------------|--------------------|
| light | ~203 s | ~187 s | **44,813 B/s** |
| gate24 | ~743 s | ~902 s | **~9,300 B/s** |
| champ | OOM @ ~5 s | OOM @ ~11 s | — |

---

## Full-vocab latency (light only — honest bit-tree profile)

64-byte warm context, 2 timed iterations (`cyphalm_hp_sku_measure`):

| Path | µs/call | vs legacy |
|------|---------|-----------|
| bit-tree `next_byte_log_probs` (default light) | **36,894,104** | 1.86× slower |
| legacy 256-clone | **19,811,989** | 1.00× |

gate24/champ use **legacy 256-clone** for full-vocab scoring (bit-tree checkpoint pool OOMs on v78 table sizes). Bit-tree code is kept; it is not a latency win until hp undo stack lands. Parity: `hp_bit_tree_smoke` max Δ = 0 on light.

---

## hp `--profile` (redundancy decomposition)

gate24 enwik8.8mb (`hp_gate24 c --mem 22 --lr 2 --profile`):

- **model redundancy** (mixer vs best expert): **+1,531,856 B**
- **coding redundancy** (APM+coder vs mixer): **−25,718 B**
- **parameter share** (sparse contexts): **47%**

See [`CYPHALM_HP_ALGORITHM_PROFILE.md`](CYPHALM_HP_ALGORITHM_PROFILE.md) and [`enwik_sku_profiles/`](enwik_sku_profiles/).

---

## WikiText-2 (secondary — not comparable to 1.610)

light SKU only, `cyphalm_llm_eval`, n=100,000 bytes, bit-serial observe:

| Slice | BPC | Throughput |
|-------|-----|------------|
| train | 2.106972 | 30,476 B/s |
| eval | 2.138972 | 34,326 B/s |

Different corpus, different SKU — **do not** headline these vs enwik champ.

---

## Reproduce

```bash
# Download corpus (see bench/data/enwik8/README.md)
bash scripts/measure_enwik_skus.sh bench/data/enwik8/enwik8.8mb

# Or per-SKU:
cmake -S native -B native/build-gate24 -DCMAKE_BUILD_TYPE=Release \
  -DCYPHA_HP_PROFILE=gate24 -DCMAKE_CXX_COMPILER=g++
cmake --build native/build-gate24 --target cyphalm_hp_sku_measure -j$(nproc)
./native/build-gate24/cyphalm_hp_sku_measure \
  --corpus bench/data/enwik8/enwik8.8mb \
  --hp-tool native/build-sku-measure/hp_gate24 \
  --work-dir /tmp/gate24_measure
```

---

## Related

- [`CYPHALM_HP_ALGORITHM_PROFILE.md`](CYPHALM_HP_ALGORITHM_PROFILE.md) — architecture, RAM, bottlenecks  
- [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md) — observe≡archive parity proof  
- [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md) — undo stack / lossy LLM roadmap
