# CyphaLM hp integration profile

**Date:** 2026-09-19  
**Upstream canonical profile:** [CompressionAlgorithm `docs/reports/HP_ALGORITHM_PROFILE.md`](https://github.com/odin-loki/CompressionAlgorithm/blob/master/docs/reports/HP_ALGORITHM_PROFILE.md)

CyphaLM builds **gate24 only** (v78_flags.ps1 + `HP_SLOT_MAX=24`). Removed light/champ SKUs: [`REMOVED_HP_SKUS.md`](../history/REMOVED_HP_SKUS.md).

---

## Reference bar (upstream s24, same corpus)

Corpus: `bench/data/enwik8/enwik8.8mb` — SHA256 `09f6dd7241a8ae21edfd6762f3c6712a1fd02f7f322c5e77cab8bb88f292ee8e`

| Config | Archive bytes | BPC |
|--------|---------------|-----|
| v82-era flags, `SLOT_MAX=24`, mem 22 | **1,685,481** | **1.607** |

---

## Cypha measured gate24 (vendored hp, 2026-09-19)

| hp archive BPC | Cypha observe BPC | Δ obs−arch | RT |
|----------------|-------------------|------------|-----|
| **1.611759** (1,690,052 B) | **1.611729** | −0.000030 | PASS |

Δ vs upstream **1.607**: **+4,571 B** archive (+0.0048 BPC) — vendored-tree drift in `native/third_party/hp`, not observe math.

---

## Integration map

```
CyphaLMModel::eval_bpc
    → HpSequenceBackend bit-serial observe (8 bits, live pred_)
    ≡ hp archive BPC @ same compile flags

CyphaLMModel::serve_predict_next / generate_decode
    → HpSequenceBackend::serve_next_byte_log_probs (bit-tree + delta undo on pred_)
    → greedy: serve_greedy_next (O(8) undo fork on pred_)

Post-undo bench: [`GATE24_POST_UNDO_BENCH.md`](GATE24_POST_UNDO_BENCH.md).

See [`docs/native/CYPHALM_SERVE.md`](../native/CYPHALM_SERVE.md).
```

Redundancy `--profile` stderr: [`enwik_sku_profiles/gate24_redundancy_profile.txt`](enwik_sku_profiles/gate24_redundancy_profile.txt).

---

## Build & measure

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --target cyphalm_hp_sku_measure -j$(nproc)
bash scripts/measure_enwik_gate24.sh bench/data/enwik8/enwik8.8mb
python3 scripts/hp_v78_flag_diff.py
```

---

## Related

- [`CYPHALM_LLM_EVAL.md`](CYPHALM_LLM_EVAL.md)  
- [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md)
