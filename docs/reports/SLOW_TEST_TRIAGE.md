# Slow CTest triage: the 38 failing `cypha_slow` tests

**Odin Loch** · 2026-10-28 · HANDOFF §4.5 "Outside CI"

**Question.** 38 CTests labelled `cypha_slow` fail on a full local `ctest`. CI runs `-LE cypha_slow`, so they never turn a job red. Why do they fail, and what should happen to each?

**Answer.** None of the 38 fails because of the container, and none is a flake. All 38 fail because of one change: on 2026-09-18 (`0653e25`, "Integrate hp context mixer as Cypha LLM algorithm") `CyphaLMModel` stopped being the Hybrid GRIA+LSTM model and became a wrapper over hp. Release 2.5.0 (2026-09-20) made that the default and moved the hybrid sources behind `CYPHA_BUILD_LEGACY_CYPHALM` ([LEGACY_LLM](../history/LEGACY_LLM.md)). The tests still check the hybrid model.

- **35 tests** measure statistics and wiring that only the hybrid model had. They are cut candidates (class d). Deleting them is Odin's call.
- **3 tests** check behaviour hp should still have, and two of them show real defects in the hp wrapper when the vocabulary is smaller than 256 symbols. The fixes are in model code, outside these tests, so nothing was changed here.

No code was changed. No test was skipped, relabelled or loosened.

## Method

1. Release build of every target (`/root/work/slow/build`), then `ctest -L cypha_slow --output-on-failure -j1`: 91 tests, **38 failed** (477 s). The list matches the earlier full run except `native_cyphalm_neural_smoke`, fixed on 2026-10-28 (HANDOFF §4.5).
2. Each failure was re-run to tell a flake from a fixed result. Every one failed again the same way.
3. A logging wrapper around `cyphalm_bench_native` (this build only, removed after) recorded every call the domain tests make and the error each call printed.
4. The history came from a separate blob-less clone of the repository (603 commits), because the working clone is shallow (it starts at 2026-09-25).

## Results by class

| Class | Tests | Cause |
|---|---:|---|
| (a) bitrot in the test, fixable inside it | 0 | — |
| (a′) real defect, outside the test | 2 | The hp wrapper with a vocabulary under 256 symbols: generation emits out-of-vocabulary tokens; a memorised canary is not recalled |
| (a″) cost changed under the test | 1 | `forecast_smoke`: the pipeline now builds an hp model where it built a small hybrid one; it passes in 6 min 4 s alone, 82% of it kernel time, past the 300 s limit |
| (b) environment | 0 | — |
| (c) threshold or flake | 0 | Every failure repeats exactly |
| (d) obsolete: measures the removed hybrid model | 35 | See below |

### Class d: 35 tests that measure the removed hybrid model

**`native_cyphalm_bench_intelligence_profile` and 27 domain smokes** (d39, d40, d41, d45, d47, d48, d49, d51–d55, d58–d62, d64, d68–d76) all call `cyphalm_bench_native --intelligence-profile`. All 28 calls fail the same way:

```
cyphalm_bench_native: incomplete intelligence profile; missing stats: alpha d_eff sigma_branch tau r_eu lipschitz calibration
```

These seven statistics came from the hybrid model's training loop. Since `0653e25`, `CyphaLMModel::accumulate_intelligence_profile` ignores the profiler (it only re-trains), and `cyphalm_model.hpp` says "Hybrid GRIA+LSTM removed; hp is the LLM core". The profile can never be complete on hp, so the bench exits 1 by design.

**Seven more domain smokes fail before calling the bench**, on checks of the hybrid model's source or tools:

| Test | Error | Why |
|---|---|---|
| d44 | `missing kernel_llm_h04_smoke` | No CMake target builds it any more (the H04 kernel-LLR stub, Phase 30) |
| d46 | `Phase 33 math stack wiring incomplete` | Looks for `use_tau_forget_gate`, `hybrid_forget_gate_scale`, `use_kernel_llr` in `cyphalm_model.cpp`; none are there |
| d50 | `Phase 37 joint lock tuning sources missing` | The same kind of source check |
| d63 | `cyphalm_model missing reu forget blend scaling` | Hybrid feature, not in `cyphalm_model.cpp` |
| d65 | `cyphalm_model missing navigation loss warmup ramp` | Hybrid feature |
| d66 | `cyphalm_model missing free energy penalty` | Hybrid feature |
| d67 | `cyphalm_model missing runtime kernel blend` | Hybrid feature |

**Superseded by:** hp on 2026-09-18/20 (`0653e25`, release 2.5.0), and since 2026-10 the packed trie, which is the model taken forward (CURRENT.md). hp stays only as a comparison.

**Size if cut:** about 4,100 of the 10,459 lines of `native/src/bench/bench_domains.cpp` (the d39–d76 domain functions), 38 profiles in `bench/config/` (`d39_*` to `d76_*`), and 35 CTest blocks. Four domain tests in the same range pass (d42, d43, d56, d57: "math integration" checks over `bench/BASELINE_LOCK.json` and cell-sweep files, 0.01–7 s). They belong to the same hybrid-era series and should be looked at in the same cut.

### Class a′: real defects in the hp wrapper (vocabulary under 256)

**`native_predictive_codec_smoke`** fails on one sub-check:

```
FAIL: mixer should beat cold neural on repetitive stream mix=0.241022 neural=0.240311
```

- The sub-check assumes the cold "neural" model is weak, as an untrained LSTM was, so the online n-gram experts must win by 0.05 bits. hp is the neural model now and already codes the pattern at 0.240 bits by itself. The premise is stale; the check needs a new design, not a new number.
- The sub-check also hides the rest of the test. With it switched off in a scratch build (reverted), the round trip passes and the next check fails: **`generated token out of vocab`**. `generate_via_bits` returns a token ≥ 64 for a 64-symbol model. That is a real defect, in `predictive_codec.cpp` or the hp wrapper, not in the test.

**`native_memorization_canary_smoke`** fails: `smoke requires >=1 canary recalled`.

- `cyphalm_memorization_canary` trains 40 passes over 90 characters with 6 planted 10-character canaries, then predicts each canary's last 5 characters after its first 5. It recalls **0 of 6**, and the predictions are noise (`P?TCC` for `DXCOJ`; `?` is a symbol outside the canary text).
- The tool asks for `char_lstm`, a mode that now runs hp without hp's production recipe. A scratch build with the living `Hp` mode (reverted) also recalls 0 of 6. So the failure is not the stale mode name.
- A context-mixing model that has seen the text 40 times should recall these. Like the codec case, this points at the wrapper's handling of a small vocabulary. The test's expectation (at least 1 of 6) is reasonable for any working model.

### Class a″: `native_forecast_smoke`, a time limit

- It times out at 300 s. Alone on the machine, with no limit, it **passes** (`node_eval_acc=1`, 998 tree nodes) in **6 min 4 s**: 33 s of user time and **4 min 57 s of kernel time**.
- `ForecastPipeline` still calls `apply_hybrid_production_recipe` and builds a `CyphaLMModel`, which is hp since `0653e25`. The process holds about 845 MB and runs 200 rollouts of depth 8 through it. The 300 s limit dates from the small hybrid model.
- The kernel time says the cost is memory mapping around hp's large tables (allocated with `mmap`, CHANGELOG "hp tables are demand-zero again"), not prediction work. Where the mapping churn comes from (a table reset or copy per rollout) was not traced: it is in the hp wrapper, outside this test.
- Raising the limit would hide that cost. The choice is Odin's: find the mapping churn, a smaller hp tier (or the packed trie) for the forecast pipeline, or a limit set from the measured time.

## For Odin

1. **Cut the 35 class-d tests** and the code only they use (about 4,100 lines of `bench_domains.cpp` and 38 bench profiles)? They test a model that was removed on 2026-09-18.
2. **The hp wrapper's small-vocabulary defects** (out-of-vocabulary generation; canary recall 0/6): fix, or retire `CyphaLMModel`'s non-byte vocabularies? hp is comparison-only now.
3. **The forecast pipeline:** trace the kernel-time churn (4 min 57 s of 6 min 4 s), use a smaller hp tier or the packed trie, or set a longer limit?

## How to reproduce

```
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
ctest --test-dir build -L cypha_slow --output-on-failure -j1          # 91 tests, 38 fail
CYPHA_BENCH_FAST=1 build/cyphalm_bench_native --profile d17 --mode hybrid \
  --n-train 100 --n-eval 40 --intelligence-profile                   # exit 1: the 7 missing statistics
build/cyphalm_memorization_canary --out canary.json --seed 42        # recall 0/6
git log -S'cypha::intelligence::IntelligenceProfiler&) {' -- native/src/cyphalm/cyphalm_model.cpp   # 0653e25, full clone
```

Running the slow tests rewrites `bench/BASELINE_REPORT.md`; restore it with `git checkout` afterwards.
