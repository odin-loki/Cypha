# GGUF export (BoW §5) — 2026-07-17

**Scope:** Pack CyphaDIF inference tensors from `.cypha` into a GGUF v3 container for structural load / tooling.

**OFF-LIMITS:** `build_math`, `build_deff`, `BASELINE_*`, overnight.

**Tool:** `cypha_export_gguf` (`native/tools/cypha_export_gguf.cpp`)  
**Build dir:** `native/build_gguf` (Ninja, Release, MinGW).

## Status

Honest **partial** export: valid GGUF v3 magic/header + metadata + F32 tensor info and (default) embedded weight blobs. Not a llama.cpp LLM architecture; architecture tag is `cypha-dif`. Suitable for structural load / custom loaders, not drop-in llama.cpp chat inference.

## Tensors packed

| Name | Shape | Source |
|------|-------|--------|
| `enc_W` | `[d, d]` | `CyphaInferModel::enc_w` |
| `F_field` | `[d, field_dim]` | `CyphaInferModel::f_field` (when present) |
| `field_h` | `[1, field_dim]` | `CyphaInferModel::field_h` (when present) |
| `world.mu` | `[1, d]` | world μ + field shift baked at export |
| `inv_v` | `[1, d]` | `CyphaInferModel::inv_v` |
| `D` | `[K, d]` | class centroids / deltas |
| `D_T` | `[d, K]` | transpose of `D` |
| `llr_bias` | `[1, K]` | MDL + context prior terms frozen at export |

Sidecar: `<out>.manifest.json` (`cypha-gguf-manifest-v1`) lists shapes/labels; default export also embeds float32 blobs in the `.gguf`.

## Modes

| Flag | Behavior |
|------|----------|
| (default) | Write GGUF with embedded weights + manifest |
| `--header-only` | Metadata + tensor info only (no weight bytes) |
| `--verify PATH` | Magic `GGUF`, version 3, kv/tensor counts, `general.architecture=cypha-dif`, require `enc_W` + `D` |
| `--dry-run` | Load + print summary; no files |

## CTest

- `native_export_gguf_help`
- `native_export_gguf_smoke` — export `fixtures/reference.cypha` → `gguf_export_reference.gguf`
- `native_export_gguf_validate` — `--verify` on that file (depends on smoke; may be local CMake only until staged)

## Reproduce

```text
cmake -S native -B native/build_gguf -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_gguf -j8 --target cypha_export_gguf
native/build_gguf/cypha_export_gguf --cypha fixtures/reference.cypha --out native/build_gguf/gguf_export_reference.gguf
native/build_gguf/cypha_export_gguf --verify native/build_gguf/gguf_export_reference.gguf
```

## Notes

- Field-conditioned `world.mu` and `llr_bias` are frozen snapshots; live field / context dynamics are not exported as graph ops.
- Class GMM / NIG BMA / kernel LLR are out of scope for this pack.
