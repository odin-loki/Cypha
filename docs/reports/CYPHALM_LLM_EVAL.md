# CyphaLM LLM Eval — enwik8.8mb gate24

**Date:** 2026-09-19  
**Harness:** `native/tools/cyphalm_hp_sku_measure.cpp`, `scripts/measure_enwik_gate24.sh`  
**Corpus:** `bench/data/enwik8/enwik8.8mb` (8,388,608 B, SHA256 `09f6dd72…`)

CyphaLM builds **gate24 only** (v78_flags.ps1 + `HP_SLOT_MAX=24`). Removed SKUs: [`REMOVED_HP_SKUS.md`](../history/REMOVED_HP_SKUS.md).

**Win metric:** enwik8.8mb archive + bit-serial observe BPC vs upstream RECORD **1.607** (s24). See [upstream `HP_ALGORITHM_PROFILE.md`](https://github.com/odin-loki/CompressionAlgorithm/blob/master/docs/reports/HP_ALGORITHM_PROFILE.md).

---

## Headline — gate24 enwik8.8mb

| Metric | Value |
|--------|-------|
| hp archive BPC | **1.611759** (1,690,052 B) |
| Cypha observe BPC | **1.611729** |
| Δ observe − archive | **−0.000030** |
| Δ vs upstream **1.607** | **+0.004729** |
| Roundtrip | PASS (SHA256) |
| Observe wall | **430 s** (~19.5k B/s) |

**Vendored hp note:** Cypha gate24 archive is **+4,571 B** vs upstream s24 ref (1,685,481) — tree version drift in `native/third_party/hp`, not observe math.

Historical (removed SKUs, same VM): light observe **1.721331**; champ OOM @ 15 GiB.

---

## Reproduce

```bash
bash scripts/measure_enwik_gate24.sh bench/data/enwik8/enwik8.8mb
```

---

## Related

- [Upstream `HP_ALGORITHM_PROFILE.md`](https://github.com/odin-loki/CompressionAlgorithm/blob/master/docs/reports/HP_ALGORITHM_PROFILE.md)  
- [`CYPHALM_HP_ALGORITHM_PROFILE.md`](CYPHALM_HP_ALGORITHM_PROFILE.md)  
- [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md)
