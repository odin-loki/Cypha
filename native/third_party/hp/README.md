# hp (gate24) — CyphaLM's context mixer

Integer-exact PAQ-lineage context-mixing predictor, vendored from
[odin-loki/CompressionAlgorithm](https://github.com/odin-loki/CompressionAlgorithm)
and reduced to the one recipe CyphaLM ships: **gate24** (the v78 leftover flag
set with `HP_SLOT_MAX=24`, ~1.61 bpc observe on enwik8 8 MiB).

The upstream tree is an ablation lab: ~290 compile flags, most of them rejected
experiments. Here every flag was resolved to its gate24 value and the losing
branches deleted (`unifdef`, then literal folding). The archive bytes and the
per-bit predictions are identical to the flag build; see
`docs/reports/CYPHALM_HP_GATE24_STRIP.md` in the Cypha repo for the method and
the checks. The only build switches left are:

| macro | default | meaning |
|---|---|---|
| `HP_XSIMD` | 1 | SSE4.1 mixer dots (bit-identical to scalar; 0 on non-x86) |
| `HP_SLOT_MAX` | 24 | per-model table-bit cap |

Runtime lossy knobs (all default to exact gate24) live on `hp::Config`:
`cm_drop`, `cm_bits_cap`, `gate_drop`, `mixer_skip`, `match_bits_cap`,
`pool_slots`, `pool_bits_cap`. They are CyphaLM serve tiers, not archive
formats: `hp c` never sets them.

## Layout

| path | what |
|---|---|
| `include/hp/predictor.hpp` | `hp::Predictor`: 35 hashed context models, 13 byte-match models, word match, DMC, LZP, Hebbian, discovery pool, 10-set mixer, 3 APMs |
| `include/hp/mixer.hpp` | two-layer gated mixer + APM |
| `include/hp/undo.hpp` | delta undo for speculative bit updates (CyphaLM bit-tree scoring) |
| `include/hp/checkpoint.hpp` | `HPCP` v1 binary checkpoint |
| `src/main.cpp` | `hp c|d` CLI (same binary encodes and decodes) |
| `tools/` | `dump_profile`, `dump_experts`, `rank`, `bench` diagnostics |
| `test/` | round-trip, determinism, no-float, ASan, gdb smoke scripts |

No `float`/`double` in `include/` or `src/` (`test/no_float.sh`).
