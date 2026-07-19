# PGM Cell Integration (H23) — 2026-07-18

Plastic Graph Machine (PGM) is available as a native CyphaLM cell under `cypha_lm_native`
(float64 / double, no PyTorch). Research reference: Better LSTM `PGM_spec_and_results.md`.

## What landed

| Piece | Path |
|-------|------|
| Header | `native/include/cypha/cyphalm/pgm_cell.hpp` |
| Source | `native/src/cyphalm/pgm_cell.cpp` (picked up by `CYPHALM_SOURCES` GLOB) |
| Config | `use_pgm_cell` + `pgm_*` knobs on `CyphaLMConfig` |
| Hypothesis | `H23` via `apply_cell_variant` |
| Model wire | `CyphaLMModel::init_components` / `reset_context` / `predict_next` field blend |
| Smoke | `native/tools/pgm_cell_smoke.cpp` → target `pgm_cell_smoke` |

## Design (complexity constraint)

Addressing is **hierarchical log-N**, not dense attention:

- Branching factor `n_sub` (b), depth `levels` (L) → slot count \(N = b^L\)
- Candidate generation: level-wise beam expand → \(O(L \cdot \mathrm{beam} \cdot b \cdot d)\)
- T1 chunk buffer; consolidate **adjacent** pairs at chunk boundary
- T2 sparse Hebbian edges with per-row `topk` sparsify + row-norm
- T3 content bank `V[slot]` with **rehash-t** assign
- Retrieval: beam-2 max-plus hops over the sparse edge map

No \(O(n^2)\) attention matrices.

## How to enable

**Option A — hypothesis id (preferred for sweeps):**

```text
--cell-variant H23
```

or in code:

```cpp
cypha::cyphalm::apply_cell_variant("H23", cfg);
// sets use_pgm_cell=true and default pgm_* knobs; bench mode hybrid
```

**Option B — config flag:**

```cpp
CyphaLMConfig cfg;
cfg.context_mode = ContextMode::Hybrid;  // needs SSM field path
cfg.use_pgm_cell = true;
cfg.pgm_n_sub = 16;
cfg.pgm_levels = 3;
cfg.pgm_chunk_len = 16;
cfg.pgm_topk = 4;
cfg.pgm_beam = 2;
cfg.pgm_rehash_t = 16;
cfg.pgm_hops = 2;
```

When enabled, `init_components` constructs `PGMCell` with `d_input = hidden = field_dim`.
In `predict_next`, after the field projection (and optional NIG blend), PGM `step(field_x_)`
is blended: `field_x = 0.6 * field_x + 0.4 * pgm_h` (same site as H06 NIG).

Accessor: `model.pgm_cell()` / `model.pgm_cell()->step(x)`.

## Build / smoke

From `native/`:

```powershell
cmake -S . -B build -DCYPHA_ENABLE_CUDA=OFF
cmake --build build --config Release --target pgm_cell_smoke
.\build\Release\pgm_cell_smoke.exe
# or: ctest -R native_pgm_cell_smoke -C Release
```

`pgm_cell.cpp` is included automatically via the `src/cyphalm/*.cpp` GLOB into `cypha_lm_native`.
