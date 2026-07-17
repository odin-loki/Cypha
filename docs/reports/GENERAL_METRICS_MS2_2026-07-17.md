# General metrics MS2 (algebraic fingerprint vector) — 2026-07-17

**Scope:** Bill of Work Addendum 2, build-order item MS2.

## What shipped

| ID | Metric | Helper / path | Report field |
|----|--------|---------------|--------------|
| MS2 | Algebraic fingerprint vector | `cypha::cyphalm::compute_text_algebraic_fingerprint` in `text_algebraic_fingerprint.hpp` | `vector`, per-feature keys, `spectrum_position` |
| — | CyphaLM smoke runner | `cyphalm_algebraic_fingerprint` | `generated`, `scored_train` blocks |
| — | CTest FAST smoke | `native_algebraic_fingerprint_smoke` | — |

Features (6-vector, normalized to [0, 1]):

1. **linear_complexity** — Berlekamp–Massey complexity on LSB bitstream, divided by length.
2. **spectral_flatness** — geometric / arithmetic mean ratio of DFT power bins.
3. **run_length_entropy** — Shannon entropy of run-length histogram, normalized.
4. **ngram_entropy_1/2/3** — normalized char n-gram entropies at scales 1–3.

**spectrum_position** is the mean of normalized vector entries (optional scalar centroid in [0, 1]).

Existing H22 `algebraic_fingerprint.hpp` (field-state tag for GRIA) is unchanged; MS2 targets **text output** fingerprints.

## Result (FAST, seed 42)

Run `cyphalm_algebraic_fingerprint --write-table` after build; smoke asserts finite values on both `generated` and `scored_train` blocks.

## Build & test

```powershell
cmake -S native -B native/build_ms2 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_ms2 --target cyphalm_algebraic_fingerprint algebraic_fingerprint_smoke
ctest --test-dir native/build_ms2 -R native_algebraic_fingerprint_smoke --output-on-failure
$env:CYPHA_BENCH_FAST=1; native/build_ms2/cyphalm_algebraic_fingerprint.exe --write-table
```

Smoke asserts ≥4 finite vector entries and finite `spectrum_position` on generated and scored text.
