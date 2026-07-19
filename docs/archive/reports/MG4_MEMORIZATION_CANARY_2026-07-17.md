# Memorization canary (MG4) — 2026-07-17

**Scope:** Bill of Work Addendum 2, build-order item MG4.

## What shipped

| ID | Deliverable | Path |
|----|-------------|------|
| MG4 | Canary inject + train + recall runner | `cyphalm_memorization_canary` |
| — | CTest FAST smoke | `native_memorization_canary_smoke` |
| — | Result JSON | `bench/results/memorization_canary.json` |

Unique random canary strings are inserted **once** each into a short synthetic train text (`[[CANARY]]` slots). CyphaLM (char-LSTM) trains briefly; recall is teacher-forced suffix completion given the canary prefix.

## Protocol

1. Sample `n_canaries` unique uppercase strings (prefix `MG…`).
2. Build train text with each canary appearing exactly once.
3. Train `ContextMode::CharLstm` for `train_epochs` over the short corpus.
4. For each canary: teacher-force gold tokens through `[[` + prefix; require greedy next-token match on the remaining suffix.

FAST defaults (`CYPHA_BENCH_FAST=1`): **3** canaries, len **6**, prefix **4**, **60** epochs.

## Result (FAST, seed 42)

| Metric | Value |
|--------|-------|
| **recall_rate** | **1.0** (3/3) |
| n_canaries | 3 |
| canary_len / prefix_len | 6 / 4 |
| train_epochs | 60 |
| context_mode | char_lstm |
| recall_mode | teacher_forced_suffix |

## Build & test

```powershell
cmake -S native -B native/build_mg4 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_mg4 --target cyphalm_memorization_canary memorization_canary_smoke
ctest --test-dir native/build_mg4 -R native_memorization_canary_smoke --output-on-failure
$env:CYPHA_BENCH_FAST=1; native/build_mg4/cyphalm_memorization_canary.exe --write-table
```

Smoke asserts finite `recall_rate ∈ [0,1]`, non-empty canaries array, and **≥1** canary recalled.
