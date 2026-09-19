# CyphaLM LLM Eval — enwik8.8mb SKU measurements

**Date:** 2026-09-19  
**Branch:** `cursor/bit-tree-gate24-eval-9d44`  
**Harness:** `native/tools/cyphalm_hp_sku_measure.cpp`, `scripts/measure_enwik_skus.sh`  
**Corpus:** `bench/data/enwik8/enwik8.8mb` (8,388,608 B, SHA256 `09f6dd72…`)

**Win metric:** enwik8.8mb archive + bit-serial observe BPC vs CompressionAlgorithm RECORD bars **1.607** (s24) / **1.610** (champ). See [upstream `HP_ALGORITHM_PROFILE.md`](https://github.com/odin-loki/CompressionAlgorithm/blob/master/docs/reports/HP_ALGORITHM_PROFILE.md).

WikiText ~2.1 (light SKU) is **secondary only** — not comparable to 1.607/1.610.

---

## Headline — enwik8.8mb vs RECORD bars

### Reference (CompressionAlgorithm master, same corpus SHA)

| Bar | Archive bytes | BPC | Source |
|-----|---------------|-----|--------|
| **s24 screen** (`SLOT_MAX=24`, mem 22) | **1,685,481** | **1.607** | upstream measured 2026-09-19 |
| **v82 champ** (`SLOT_MAX=35`, mem 22) | **1,689,157** | **1.610** | RECORD (not re-run @ 15 GiB) |

### Cypha measured (vendored hp, this VM: 15 GiB, 4-core Xeon, g++ 13.3)

| SKU | v78 | `SLOT_MAX` | hp archive BPC | Cypha observe BPC | Δ vs **1.607** | Δ vs **1.610** | RT |
|-----|-----|------------|----------------|-------------------|----------------|----------------|-----|
| **light** | 0/78 | 24 | **1.721362** | **1.721331** | +0.114 | +0.110 | PASS |
| **gate24** | 78/78 | 24 | **1.611759** | **1.611729** | +0.0048 | +0.0008 | PASS |
| **champ** | 78/78 | 35 | — | — | — | — | **OOM** |

**gate24 observe** re-measured 2026-09-19: **1.611729** in **430 s** (Δ obs−archive **−0.000030**).

**Vendored hp note:** Cypha gate24 archive is **+4,571 B** vs upstream s24 ref (1,685,481) — tree version drift in `native/third_party/hp`, not observe math. Observe≡archive within Cypha.

---

## Flag parity

```bash
python3 scripts/hp_v78_flag_diff.py
# light: 0/78   gate24: 78/78   champ: 78/78
```

gate24/champ: `v78_flags.ps1` + `HP_XSIMD=1` + `-msse4.1`. Upstream v82-era recipe also strips post-v82 accepts (`HP_LR1_SCALE`, `HP_WIKIBOLD_MOD`, …) — see upstream §4.2–4.4.

---

## Timing (measured)

| SKU | hp `c` wall | Cypha observe wall | Throughput |
|-----|-------------|--------------------|------------|
| light | ~200 s | ~191–212 s | **~40–45k B/s** |
| gate24 | ~425 s | **430 s** | **~19.5k B/s** |
| champ | OOM ~5 s | OOM ~11 s | — |

Upstream s24 reference: **502 s**, **1.54 GB** peak RSS.

---

## Bit-tree latency (light, honest)

64 B warm context, 2 iters:

| Path | µs/call |
|------|---------|
| bit-tree (default light) | **~37M** |
| legacy 256-clone | **~19M** |

gate24/champ: legacy 256-clone only (checkpoint pool OOM). Parity: `hp_bit_tree_smoke` Δ=0. **Undo stack** is the planned speed fix.

---

## hp `--profile` (redundancy, not CPU)

gate24 vendored hp: model redundancy **+1,531,856 B**; parameter sparse share **47%**. See upstream §2.1 and [`CYPHALM_HP_ALGORITHM_PROFILE.md`](CYPHALM_HP_ALGORITHM_PROFILE.md).

---

## WikiText-2 (secondary)

light SKU, n=100k observe: train **2.107**, eval **2.139** BPC. Different corpus + SKU — **do not headline vs 1.607**.

---

## Reproduce

```bash
bash scripts/measure_enwik_skus.sh bench/data/enwik8/enwik8.8mb
```

---

## Related

- [Upstream `HP_ALGORITHM_PROFILE.md`](https://github.com/odin-loki/CompressionAlgorithm/blob/master/docs/reports/HP_ALGORITHM_PROFILE.md)  
- [`CYPHALM_HP_ALGORITHM_PROFILE.md`](CYPHALM_HP_ALGORITHM_PROFILE.md) — Cypha integration supplement  
- [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md)
