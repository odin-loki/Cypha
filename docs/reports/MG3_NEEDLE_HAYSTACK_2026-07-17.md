# Needle-in-haystack long-range recall (MG3) — 2026-07-17

**Scope:** Bill of Work Addendum 2, MG3 — plant a unique fact early in context, pad with filler, ask at the end; score char-LSTM answer-token recall + BPC.

## What shipped

| ID | Deliverable | Path |
|----|-------------|------|
| MC3 | Needle-haystack runner | `native/tools/cyphalm_needle_haystack.cpp` |
| MG3 | JSON result + table | `bench/results/needle_haystack.json` |
| — | CTest smoke | `native_needle_haystack_smoke` (`CYPHA_BENCH_FAST=1`) |

Build dir: `native/build_mg3`.

## Protocol

1. Build synthetic sequence: `FACT: The secret code is [[<needle>]].` + filler haystack + `QUESTION: ... ANSWER: ` + needle.
2. Train tiny CharLSTM on that sequence (`CYPHA_BENCH_FAST=1`: epochs=15, hidden=64, lr=0.25).
3. Score answer span:
   - **token_recall** — fraction of answer chars where argmax next-token equals ground truth (teacher-forced after full context warm-up).
   - **bpc_answer** — bits-per-char on the answer tokens only.
   - **exact_recall** — greedy generation of the full needle matches (stricter; often false at FAST scale).

Default FAST haystack depths: **32 / 64** chars. Non-FAST: **256 / 512 / 1024**.

## Measured numbers (`CYPHA_BENCH_FAST=1`, seed=42)

| haystack_chars | context_chars | token_recall | bpc_answer | exact_recall |
|----------------|---------------|--------------|------------|--------------|
| 32 | 116 | 0.00 | 5.154 | false |
| 64 | 148 | 0.10 | 5.005 | false |

- **Mean token_recall (`recall_rate`): 0.05** (0/2 depths fully recalled)
- Needle: `MG31x1e6du` (len 10)
- CTest: `native_needle_haystack_smoke` **PASS**

## Build & test

```powershell
cmake -S native -B native/build_mg3 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_mg3 --target cyphalm_needle_haystack needle_haystack_smoke
ctest --test-dir native/build_mg3 -R native_needle_haystack_smoke --output-on-failure
$env:CYPHA_BENCH_FAST=1; native/build_mg3/cyphalm_needle_haystack.exe --seed 42 --write-table
```

## Notes

- Cheap synthetic domain; not a production long-context claim.
- At FAST scale, exact greedy needle replay fails; token-level recall on the answer span is the primary MG3 metric.
- Smoke requires finite `bpc_answer` and `token_recall > 0` on at least one depth.
