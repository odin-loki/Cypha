# D17 production training throughput: profiling + memory-allocation fix (2026-07-12)

**Scope:** D17 production training (`cyphalm_bench_native --profile d17 --threads 1`) runs at
~96-126 chars/sec single-threaded, making it the bottleneck for essentially every benchmark/research
workflow in this repo. This pass profiles the actual hot path, implements the highest-leverage safe
fix the profiling data supports, and verifies the D17 BPC pin is bit-identical before/after.

**HEAD at start:** `8833611`. Fresh build dir: `native/build_perf` (Ninja, MinGW/Strawberry
`g++.exe`, `-DCMAKE_BUILD_TYPE=RelWithDebInfo`). Did not touch `native/build_math`,
`bench/BASELINE_LOCK.json`, `bench/BASELINE_REPORT.md`, or the overnight orchestration scripts. A
sibling agent was concurrently committing an unrelated `lstm_hidden_d_eff` history-buffer fix
(`8cb9df1`, `3894d99`) to `cyphalm_model.hpp`/`cyphalm_model.cpp` during this session — confirmed via
`git diff`/`git log` and kept fully separate (see "Concurrency note" below); this report's changes are
layered on top of that commit, not in conflict with it.

**Bottom line:**
- **Profiling:** MSVC/WPT weren't available (no Visual Studio C++ toolset installed on this
  machine — see CUDA section), and `gprof` produces empty output under MinGW/Windows (known
  limitation). Used env-gated `std::chrono::steady_clock` phase instrumentation
  (`CYPHA_PERF_TRACE=1`) around every subsystem in `CyphaLMModel::train_step`. Over 17,978 real
  `train_step` calls: **`lstm_backward` (hybrid LSTM backward pass) = 45.3%**, **`predict_next`
  (forward: GRIA+LSTM+hybrid blend) = 31.7%**, **`bptt_ssm_update` = 18.2%** — LSTM
  forward+backward and BPTT together account for **>95%** of per-step time. GRIA backward, DIF
  memory training, and end-of-step bookkeeping are all under 3% combined.
- **Root cause:** not raw compute — at `lstm_hidden=128` the matvecs themselves are trivially
  cheap. It's **heap allocation/deallocation churn**: `CharLSTMHead::forward_step` and
  `backward_step` freshly `std::vector`-allocated ~15 temporary buffers (gates, gate activations,
  gradients, up to `4*hidden` and `hidden*4*hidden` elements) on **every single character step**,
  and `CyphaLMModel::train_step` freshly allocated a full `CharLSTMGrad` (four weight-sized
  gradient matrices, ~1.5MB at `hidden=128`) every step just to discard it after `apply_grads`.
- **Fix:** replaced the hot-loop temporaries with `thread_local static` buffers (reused/resized in
  place instead of reallocated) in `char_lstm.cpp`, and added a persistent
  `CyphaLMModel::hybrid_lstm_grad_scratch_` member filled in-place via a new out-param
  `CharLSTMHead::backward_step(..., CharLSTMGrad& out, ...)` overload. **Zero change to any
  arithmetic** — every element is fully overwritten before use each step, so this is a pure
  allocator-pressure fix, not a numerical one.
- **Measured speedup:** 126.4 → 138.3 chars/sec on `--n-train 20000` (paired same-session run,
  see "Measurement caveat" below) — **~9.5% faster**, entirely from removing allocator overhead.
- **Determinism:** confirmed bit-identical/CTest-clean before and after — `native_d17_wikitext_smoke`,
  `native_cyphalm_model_parity`, `native_cyphalm_char_lstm_parity`,
  `native_baseline_lock_validate_smoke`, and 11 other native regression tests (see "Verification" §)
  all pass on the optimized build with zero flag changes to the default code path.
- **CUDA feasibility:** **blocked, concretely** — `nvcc` 13.2.51 is installed, but there is
  **no MSVC or any other CUDA-supported host compiler on this machine at all** (`vswhere -all`
  returns zero registered Visual Studio installations; no `cl.exe`/`clang-cl` anywhere). `nvcc`
  does not support MinGW-w64 GCC as a host compiler, matching `native/CMakeLists.txt`'s existing
  `FATAL_ERROR` guard. No CUDA build was attempted (would not compile). See CUDA section for exactly
  what's missing.

---

## 1. Profiling methodology

### 1.1 Toolchain check (Phase 1 step 1)

- `where cl` / `vswhere -latest -products *` initially suggested a Visual Studio Build Tools
  installation might exist, but a direct search for `cl.exe` under any `Program Files*\Microsoft
  Visual Studio*` path found nothing, and a later `vswhere.exe -all -products *` (re-run after the
  fact) returned **zero installations** — there is no MSVC C++ toolset on this machine right now.
  Windows Performance Toolkit / VS diagnostics were therefore unavailable.
- Built `cyphalm_bench_native` in a fresh `native/build_perf` with Ninja + MinGW/Strawberry
  `g++.exe` 13.2, `-DCMAKE_BUILD_TYPE=RelWithDebInfo` (`CMAKE_CXX_FLAGS_RELWITHDEBINFO = -O2 -g
  -DNDEBUG` — optimizations on, debug symbols present, confirmed from `CMakeCache.txt`). No
  `-march=native`/AVX flags are enabled by default (see §5.3 below for why that's a candidate, not
  a bug fix here).

### 1.2 `gprof` attempt (Phase 1 step 2, second choice)

Built a `-pg`-instrumented variant and ran a training pass to produce `gmon.out`. `gprof` against
that file produced an **empty flat profile and call graph** (no samples) despite a substantial
`gmon.out` size. This is a known limitation of `gprof`'s interval-timer-based sampling under
MinGW/Windows — the SIGPROF-equivalent sampling mechanism it relies on doesn't work correctly in
this environment. Not pursued further; moved to the fallback.

### 1.3 Manual `chrono` phase instrumentation (Phase 1 step 2, fallback — what actually produced results)

Added an env-gated (`CYPHA_PERF_TRACE=1`), zero-overhead-when-unset instrumentation block to
`native/src/cyphalm/cyphalm_model.cpp`: a `PerfTracePhases` struct accumulates
`std::chrono::steady_clock` durations into 8 named buckets across every `train_step` call, and
prints a percentage breakdown to `stderr` at process exit. `perf_trace_scope(idx, fn)` /
`perf_trace_begin()`/`perf_trace_end()` wrap each subsystem call inside `train_step`; when the env
var is unset, `perf_trace_scope` degenerates to a single branch-and-call with no clock read at all,
so this is safe to leave permanently in the tree opt-in (see the comment block in the source for
the exact safety argument re: the D17 BPC pin).

Representative result, running `cyphalm_bench_native --profile d17 --n-train 6000 --n-eval 500
--threads 1` with `CYPHA_PERF_TRACE=1` on the **optimized** binary (post-fix; the pre-fix run showed
the same two phases dominating, just with a different internal split — allocator overhead is
folded into whichever phase happens to trigger the realloc):

```
=== CYPHA_PERF_TRACE: train_step phase breakdown over 17978 calls (64.8726s instrumented) ===
  predict_next (forward: GRIA+LSTM+hybrid blend): 20.5559s (31.6865%)
  gria_backward (cross_entropy_gradients+update_weights/alpha/bias): 1.72249s (2.65519%)
  dif_train_step (kernel-LLR memory): 1.15658s (1.78285%)
  hebbian_stack (encoder_train_step): 0.0029298s (0.00451624%)
  bptt_ssm_update: 11.814s (18.2111%)
  lstm_backward (hybrid path): 29.3802s (45.2891%)
  rpsm_train_step: 0.0019984s (0.0030805%)
  tail (ewc/ngram/gng/laplace bookkeeping): 0.238506s (0.367653%)
```

**Interpretation:** LSTM forward (`predict_next`, which runs the LSTM cell alongside GRIA) and LSTM
backward (`lstm_backward`) together are ~77% of step time; `bptt_ssm_update` (a separate BPTT pass
over the SSM/GRIA field) adds another 18%. Everything else — GRIA weight updates, DIF/kernel
memory, Hebbian encoder, RPSM, EWC/n-gram/GNG bookkeeping — is noise-level (<3% combined). This
directly points at the two `CharLSTMHead` hot functions and the BPTT update as the only places
worth touching.

### 1.4 Was it compute-bound, allocation-bound, or something else? (Phase 1 step 3)

Reading `char_lstm.cpp`'s `forward_step`/`backward_step` (the dominant phases from §1.3) before
changing anything showed the answer directly: **allocation-bound, not compute-bound.** Every call
to `forward_step` freshly constructed ~8 `std::vector<double>` locals (`gates`, `wh`, `i_gate`,
`f_gate`, `g_gate`, `o_gate`, `logits`, `probs`), and every call to `backward_step` freshly
constructed ~9 more (`d_logits`, `dh_new`, `do_gate`, `dc_new`, `df_gate`, `di_gate`, `dg_gate`,
`dgates`, `dx`) — plus `CyphaLMModel::train_step` freshly heap-allocated a full `CharLSTMGrad`
(four gradient matrices sized `vocab_size*hidden` and `4*hidden*hidden`, ~1.5MB total at
`hidden=128`) on every step just to apply it once and discard it. At `hidden=128`, the actual
matvecs (`128x128`, `256x128`, etc.) are floating-point-trivial — nanoseconds of real arithmetic —
so a heap allocator round-trip (malloc + first-touch page faults + free) per buffer, repeated ~17
times per character, plausibly dominates. This matches the measured phase split: the two phases
that do the most allocation (`lstm_backward`, `predict_next`) are exactly the two that dominate
wall time, disproportionately to their arithmetic FLOP count.

---

## 2. Optimization implemented (Phase 2)

Since the data pointed squarely at allocation churn (not compute, not RNG/IO/vtables), the fix is a
pure memory-reuse change with **zero arithmetic change**:

1. **`native/src/cyphalm/char_lstm.cpp` — `forward_step`:** converted the 8 local `std::vector`
   temporaries to `thread_local static std::vector`s, resized (not reallocated) on first use per
   thread and reused every call thereafter. `thread_local` (not a plain function-static) because
   `CyphaLMBatch`'s `parallel_batch` (`cyphalm_batch.cpp`) calls these methods concurrently on the
   same shared `CharLSTMHead` object from multiple worker threads — a plain `static` would be a
   data race; `thread_local` gives every thread its own private backing storage with no aliasing.
2. **`native/src/cyphalm/char_lstm.cpp` — `backward_step`:** same treatment for its 9 local
   temporaries, plus a new **out-param overload**
   (`void backward_step(cache, target_id, CharLSTMGrad& out, ...)`) so callers that already own a
   persistent gradient buffer can fill it in place instead of receiving a freshly-allocated
   `CharLSTMGrad` by value every call. The original by-value overload now delegates to this one
   (so anyone still calling the old signature gets identical behavior/output, just via one extra
   copy — no existing caller's numerics change).
3. **`native/include/cypha/cyphalm/cyphalm_model.hpp` / `cyphalm_model.cpp` — `CyphaLMModel`:**
   added a `CharLSTMGrad hybrid_lstm_grad_scratch_` member, reused across every `train_step` call
   in the hybrid-mode LSTM backward path via the new out-param overload — eliminates the ~1.5MB
   `CharLSTMGrad` allocation that previously happened every single character step in D17's default
   `--mode hybrid` path.

All three changes only add/replace buffer *storage*; every buffer is fully overwritten before any
read in the same call, so there is no way for stale thread-local state from a previous call (on the
same or a different sequence) to leak into a result. `CyphaLMModel::train_step` itself still runs
single-threaded per model instance in the D17 production path (`--threads 1`); the `thread_local`
buffers only matter for `CyphaLMBatch`'s separate multi-thread batch path, where they're required
for correctness (not just performance) once introduced.

**Not pursued this pass (data didn't support it as highest-leverage, or out of scope):**
- **Compute/vectorization:** `-O2` (RelWithDebInfo default) already enables auto-vectorization for
  the plain scalar loops in `matvec_rowmajor` etc.; no `-march=native`/AVX2 is set, but given
  allocation — not arithmetic throughput — was the measured bottleneck, this wasn't the
  highest-leverage next step (see §5.3 for why it's still worth a follow-up, carefully).
- **Multi-threading the online recurrence:** correctly *not* attempted — `train_step`'s
  character-by-character LSTM/SSM state recurrence is inherently sequential (each step depends on
  the previous step's hidden state); parallelizing it would require either changing the algorithm
  or accepting non-determinism, neither of which is in scope for a "don't touch the BPC pin" pass.
- **CUDA:** see §4 — blocked by missing host compiler, not a code issue.

---

## 3. Measurement (Phase 3)

### Before/after throughput

Both runs: `cyphalm_bench_native.exe --profile d17 --n-train 20000 --n-eval 2000 --threads 1`
(RelWithDebInfo, `native/build_perf`), measured with `Measure-Command`/`Stopwatch` wall time:

| | Wall time | Throughput |
|---|---|---|
| Before (pre-fix, HEAD `8833611`) | 158,247 ms | **126.4 chars/sec** |
| After (post-fix) | 144,567 ms | **138.3 chars/sec** |
| **Speedup** | | **+9.5%** |

### Measurement caveat (important)

This machine is shared with other concurrently-running agents/processes today (a sibling agent's
`lstm_hidden_d_eff` fix landed mid-session as two new commits; a third process was independently
attempting a CUDA+VS2026 build in the same tree throughout). Re-running the identical
`--n-train 20000` command later in the session, under heavier concurrent load, produced wall times
ranging from **166s to 567s** for what should be the same ~145-160s workload — a >3x swing driven
entirely by system-level contention, not by anything in this change. **The 126.4 → 138.3 chars/sec
numbers above were captured back-to-back, in the same narrow time window, under comparable system
load**, and are the trustworthy same-session A/B comparison; later noisy re-runs were discarded as
uninformative rather than reported as contradictory data. The `CYPHA_PERF_TRACE` phase-percentage
breakdown (§1.3) is inherently more robust to this kind of noise, since it's a *relative* split of
instrumented time rather than an absolute wall-clock number, and independently corroborates that
LSTM forward/backward dominates regardless of which specific run you look at.

### Determinism / regression verification

All of the following pass identically before and after this change (`ctest`, `native/build_perf`):

- `native_d17_wikitext_smoke` — the closest thing to a direct D17-pin regression check (Passed,
  ~3.8s).
- `native_baseline_lock_validate_smoke` — validates against the existing (untouched)
  `bench/BASELINE_LOCK.json` (Passed).
- `native_cyphalm_model_parity`, `native_cyphalm_char_lstm_parity`, `native_cyphalm_checkpoint_parity`
  — exact parity/checkpoint round-trip tests for the two files touched (Passed).
- `native_ewc_cyphalm_smoke`, `native_ewc_hybrid_smoke`, `native_ewc_weights_smoke`,
  `native_navigation_loss_char_lstm_smoke`, `native_navigation_loss_hybrid_smoke`,
  `native_tau_forget_gate_smoke`, `native_lm_self_correct_smoke`, `native_kernel_llm_h04_smoke`,
  `native_intelligence_lm_monitor_smoke`, `native_corpus_smoke` — broader regression coverage of
  everything that calls into `CharLSTMHead`/`CyphaLMModel::train_step` (Passed).

**15/15 tests passed, 0 failures**, both immediately after implementing the change and again after
it was re-layered on top of the sibling agent's concurrently-landed commits (see "Concurrency note"
below) — confirming the fix and the sibling's unrelated change compose cleanly with no interaction.

---

## 4. CUDA feasibility investigation (Phase 2, as requested)

Checked concretely rather than assumed:

- `nvcc --version` → **CUDA 13.2.51 is installed** and functional.
- `vswhere.exe -all -products *` → **returns zero results.** There is no Visual Studio installation
  registered on this machine at all right now (an earlier same-session check had suggested Build
  Tools might be present via stale installer metadata, but no `cl.exe` exists on disk anywhere
  under any `Program Files*\Microsoft Visual Studio*` path, and the later `-all` sweep confirms no
  installation is registered).
- `Get-Command clang-cl` / `clang` / `clang++` → **none found.** No alternative CUDA-supported host
  compiler is available either.
- `native/CMakeLists.txt` already has an explicit `FATAL_ERROR` guard for
  `CYPHA_ENABLE_CUDA=ON` + MinGW ("not supported for MinGW targets (use native MSVC or Linux +
  nvcc)") — this matches upstream reality: `nvcc` on Windows requires MSVC `cl.exe` (or, on newer
  CUDA/MSVC combinations, `clang-cl`) as its host compiler; MinGW-w64 GCC is not a supported host
  compiler for `nvcc` on Windows at all, independent of this codebase.

**Concrete blocker:** install Visual Studio Build Tools with the "Desktop development with C++"
workload (which provides `cl.exe`/`vcvars64.bat`) on this machine. Once that's present,
`-DCYPHA_ENABLE_CUDA=ON` in a throwaway build dir becomes a real, low-risk thing to try next — the
CUDA source (`accel_cuda.cu`/`accel_backend.cpp`) and a `cuda_smoke` CTest already exist per
`docs/FUTURE.md` §1, so the remaining work is purely toolchain installation, not code. No CUDA
build was attempted this pass since it cannot succeed without that install (confirmed, not
guessed). Note: a third process was observed concurrently generating
`native/build-windows-vs2026-cuda_{configure,build}_log*.txt` in this same tree during this
session — i.e. someone else already appears to be mid-attempt at exactly this VS2026+CUDA install;
worth checking that work before repeating it.

---

## 5. Prioritized further opportunities (not implemented this pass)

Ordered by estimated impact/effort ratio, based on the measured phase breakdown in §1.3:

1. **`lstm_backward` further optimization (45% of step time) — Effort: Medium, Impact: High.**
   The allocation fix in this pass removes *allocator* overhead from this phase but doesn't touch
   its arithmetic. `CharLSTMHead::backward_step` still does several full `hidden`- and
   `4*hidden`-sized passes per call; profiling *within* this phase (e.g. re-running
   `CYPHA_PERF_TRACE` with finer-grained sub-phase buckets specifically inside `backward_step`)
   would show whether the remaining time is memory-bandwidth-bound (cache misses on the
   `4*hidden x hidden` weight matrices) or still allocation-adjacent (e.g. the `dx` accumulation
   pattern). Likely next-highest-value target.
2. **`bptt_ssm_update` (18% of step time) — Effort: Medium, Impact: Medium.** Not profiled in
   isolation this pass (out of the two files this task was scoped to). Same allocation-churn
   pattern is plausible here too — worth a quick `std::vector` audit in `bptt_ssm_update`'s
   implementation before assuming it needs an algorithmic change.
3. **`-march=native` / explicit AVX2 codegen — Effort: Low to try, Medium-High risk to adopt as
   default, Impact: Unknown until measured.** RelWithDebInfo currently builds with plain `-O2`, no
   target-arch flags (confirmed via `CMakeCache.txt`). This is very easy to *try* in a throwaway
   build (`-DCMAKE_CXX_FLAGS=-march=native`) and measure — but do **not** flip it on for the
   default D17 path without an explicit bit-identical-output check first: `-march=native` can
   change floating-point codegen (e.g. FMA instruction fusion, which changes rounding vs.
   separate multiply+add) and could silently break the BPC pin. Treat as "measure the speedup in
   an isolated build, then decide whether it's worth a dedicated determinism-audit pass" — not a
   quick win to just merge.
4. **Batch/eval-side CUDA offload once a host compiler is available — Effort: High (mostly
   toolchain, some integration), Impact: High for eval/batch-encode workloads specifically, zero
   for the serial online training recurrence itself (which is the actual D17 production
   bottleneck and is not GPU-parallelizable step-to-step without changing the algorithm).** See §4
   — the code already exists, only the MSVC toolchain install is missing.
5. **Sub-phase breakdown inside `predict_next` (31.7%) — Effort: Low, Impact: informational only
   (feeds into #1/#2 prioritization).** `predict_next` blends GRIA field lookup with the LSTM cell;
   splitting those two costs out with one more `CYPHA_PERF_TRACE` bucket would clarify whether
   further LSTM-forward-specific work is worth it independent of the GRIA field cost, which this
   pass didn't need to disambiguate (both are part of the same `thread_local` fix already).

---

## Concurrency note

This session ran alongside a sibling agent making an unrelated fix to
`CyphaLMModel::lstm_hidden_d_eff()`'s history-buffer sizing (`8cb9df1`, `3894d99`) in the same two
files this task also touched (`cyphalm_model.hpp`/`cyphalm_model.cpp`). The two changes are in
non-overlapping regions (their fix is in the constructor and `append_lstm_hidden_history`/
`lstm_hidden_d_eff_detail`; this task's changes are in the `train_step` phase-instrumentation
wrapping and the new `hybrid_lstm_grad_scratch_` member). Verified via hunk-level `git diff`
inspection that the final committed diff for this task contains only this task's changes, with the
sibling's already-committed work left completely untouched, and re-ran the full regression suite
against the final composed state (§3) to confirm the two changes compose without interaction.

---

## Follow-up (2026-07-12, part 2)

**Scope:** execute this report's own §5 prioritized follow-up list -- (1) a sub-phase breakdown
*inside* `lstm_backward` (45.3% share, §5 item 1), (2) an allocation audit of `bptt_ssm_update`
(18.2% share, §5 item 2), and (3) a determinism-gated `-march=native`/`/arch:AVX2` trial (§5 item
3). **Did not touch** `native/build_math`, `bench/BASELINE_LOCK.json`,
`bench/BASELINE_REPORT.md`, the overnight orchestration scripts, `native/build_deff` (a sibling
agent's live hidden=512 D_eff confirmation run), or the `lstm_h_history_rows_`/
`kLstmHiddenHistoryMax`/`lstm_hidden_d_eff` area of `cyphalm_model.hpp` (owned by the already-
landed `8cb9df1`/`3894d99`/`e733a89` fix).

**HEAD at start:** `f566fee` (the MSVC/CUDA toolchain-migration commit; see
`docs/reports/MSVC_TOOLCHAIN_MIGRATION_2026-07-12.md`). Fresh build dir: `native/build_perf2`
(MSVC, `windows-vs2026-release` preset, per that report's recommendation now that a clean MSVC
build "just works"). A second throwaway dir, `native/build_perf2_avx2`, was used for the Part C
trial and deleted afterward (all data below was captured before deletion).

**Bottom line:**
- **`lstm_backward` sub-phase profiling (Part A):** extended `CYPHA_PERF_TRACE` with six
  sub-phase timers *inside* `CharLSTMHead::backward_step`. The two single-column/single-row
  "transpose" matrix-vector products (`Wy`/`Wx`/`Wh` backprop into `dh_new`/`dx`/`dh_prev`) were
  reading their weight matrices with a `hidden`-doubles stride in the innermost loop --
  cache-hostile, not the row-major sequential access the weight-gradient outer products already
  used. Fixing this via loop interchange (provably summation-order-preserving, so
  bit-for-bit-identical output) collapsed `lstm_backward`'s own instrumented time from **17.618s
  to 7.708s over the same 59,922 calls (-56.3%)**, and dropped its share of total `train_step`
  time from **46.3% to 41.8%** (even though its *relative* share within backward itself shifted --
  see the sub-phase table below -- because the previously-second-largest phases shrank the most).
- **`bptt_ssm_update` allocation audit (Part B):** confirmed the same allocation-churn pattern
  the top-level report predicted: `grad_core`, `grad_field`, `delta_rows`, `inv_v`, and `grad_ctx`
  were all fresh `std::vector` locals every call (plus two more full-vector *copies*, `grad_h`/
  `grad_s`, that were provably redundant sub-range duplicates of `grad_ctx`). Converted the first
  group to persistent `CyphaLMModel` scratch members (same pattern as `hybrid_lstm_grad_scratch_`)
  and deleted `grad_h`/`grad_s` entirely in favor of indexing `grad_ctx` directly. Also applied
  the same transpose-matvec loop-interchange fix from Part A to the free-function
  `matvec_transpose` helper this function calls twice. Combined effect: `bptt_ssm_update`'s
  instrumented time dropped from **15.507s to 8.151s over the same call count (-47.4%)**, share of
  `train_step` from **16.5% to 11.0%**.
- **`-march=native`/`/arch:AVX2` trial (Part C):** built a second `cyphalm_bench_native` with
  `/arch:AVX2` added to the default MSVC flags (careful to preserve the rest of the default flag
  set -- see the pitfall noted below). **Determinism: PASS, no flag changes needed.** The AVX2
  build's D17 output (`bpc`, `bpc_lstm_only`, and every other field) is **byte-for-byte identical**
  to the non-AVX2 build at the same commit/seed -- MSVC's default `/fp:precise` (not overridden)
  evidently keeps this workload's FMA/rounding behavior unchanged even with AVX2 codegen enabled.
  **Speed: no measurable benefit** -- two paired timing runs showed the AVX2 build slightly
  *slower* (81.2s, 83.9s) than the plain build (79.2s, 79.4s) on the same `--n-train 20000`
  command, i.e. within noise but never faster. **Verdict: not adopted.** Determinism alone isn't
  a reason to change the default build flags when there's no offsetting speed win, and the default
  build is left completely unchanged.
- **Throughput:** paired same-session A/B on the same commit, same MSVC build dir, same
  `--n-train 20000 --n-eval 2000 --threads 1 --bench-seed 42` command used throughout this report:
  **211.3 -> 252.2 chars/sec (+19.4%)** from Parts A+B combined. See "Running total" below for how
  this composes with Part 1's MinGW numbers and the separate MSVC toolchain migration.
- **Determinism:** the D17 BPC pin is **bit-for-bit identical** before/after Parts A and B (not
  just "close") -- confirmed via full stdout diff (`Compare-Object`, zero differences), not just
  the `bpc` field. All 18 `cyphalm|d17|lstm|bptt`-matching CTests pass; the broader regression
  sweep (EWC/navigation/checkpoint/baseline-lock/corpus/self-correct/intelligence-monitor/hebbian,
  19 tests) is 18/19 green, with the one failure (`native_ewc_weights_smoke`) being the exact same
  pre-existing MSVC-vs-GCC floating-point-codegen finding already documented in
  `docs/reports/MSVC_TOOLCHAIN_MIGRATION_2026-07-12.md` §3 (identical failure message and drift
  numbers down to the last digit: `2.019481e-03->2.051795e-03` etc.) -- confirmed unrelated to
  today's changes, not a new regression.

---

### Part A: `lstm_backward` sub-phase breakdown

Re-ran `--profile d17 --n-train 20000 --n-eval 500 --threads 1 --bench-seed 42` with
`CYPHA_PERF_TRACE=1` on the fresh MSVC build, first with only the new instrumentation added (no
optimization yet), to get an honest breakdown of where the 45.3%-of-`train_step` `lstm_backward`
phase actually goes:

| Sub-phase | Before (s) | Before (%) | After fix (s) | After fix (%) | Δ absolute |
|---|---|---|---|---|---|
| 1. output-layer backward (`dWy`/`dby`/`dh_new`) | 3.238 | 18.4% | 1.494 | 19.4% | -53.9% |
| 2. activation gradient (`do_gate`/`dc_new`+base) | 0.047 | 0.3% | 0.041 | 0.5% | -12.4% |
| 3. gate derivative dispatch (per-gate loop) | 0.049 | 0.3% | 0.045 | 0.6% | -7.2% |
| 4. weight-gradient outer products (`dWx`/`dWh`/`db`) | 3.969 | 22.5% | 3.458 | 44.9% | -12.9% |
| 5. input-gradient backprop (`dx`/`dE`) | 5.190 | 29.5% | 1.360 | 17.6% | **-73.8%** |
| 6. hidden-gradient backprop (`dh_prev`) | 5.125 | 29.1% | 1.310 | 17.0% | **-74.4%** |
| **Total (instrumented)** | **17.618** | 100% | **7.708** | 100% | **-56.3%** |

(Both columns: 59,922 real `backward_step` calls in this run.)

**Root cause of the pre-fix split:** sub-phases 5 and 6 (`dx`/`dh_prev`, the input- and
hidden-gradient backprop into `Wx`/`Wh`) were doing the exact same total arithmetic as sub-phase 4
(the `dWx`/`dWh` outer products) -- all four operations touch `4*hidden x hidden` matrices -- yet
took noticeably *more* wall time each despite doing *less* total work (one matrix each vs. two).
The difference was memory access pattern, not FLOP count: `outer_rowmajor` (sub-phase 4) writes
its output row-major/sequentially, but the pre-fix `dx`/`dh_prev` loops read `Wx[r*hidden+j]`/
`Wh[r*hidden+j]` with the loop order `for (j) for (r)` -- i.e. `r` in the *inner* loop, striding
by `hidden` doubles (1024 bytes at `hidden=128`) on every iteration, a textbook cache-hostile
strided scan across a matrix too large to keep the touched cache lines resident. Sub-phase 1's
`dh_new` computation (`Wy^T * d_logits`) has the identical pattern against `Wy` and shows the
same signature (its 18.4%-before number is *higher* than its arithmetic share alone would suggest,
for the same reason).

**Fix implemented (Part A, item 2 of the task):** a pure loop-interchange, not an allocation fix
and not an arithmetic change. Added `matvec_transpose_rowmajor(M, rows, cols, x, y)` in
`char_lstm.cpp` that computes `y = M^T * x` by iterating the outer-product-source dimension `r`
*outer* and the output dimension `c` *inner* (`y[c] += M[r*cols+c] * x[r]`), reading each row of
`M` sequentially instead of striding through it. **This is provably summation-order-preserving:**
for any fixed output index `c`, the additions into `y[c]` still happen in the exact same
increasing-`r` order as the original `for (c) { s=0; for (r) s += M[r*cols+c]*x[r]; y[c]=s; }`
form -- the interchange only changes which output element gets touched at each step, never the
order two floating-point values are added for the *same* output element. Confirmed bit-for-bit
identical output (see "Determinism" above), not just "close." Applied to all three
transpose-matvecs inside `backward_step` (`Wy`/`Wx`/`Wh`) and, for consistency and because it's
the exact same pattern, to the standalone `matvec_transpose` helper in `cyphalm_model.cpp` used by
`bptt_ssm_update` (Part B) and two other call sites.

**Secondary fix (also implemented, smaller effect):** `backward_step`'s non-eml/non-axiom path
recomputed `std::tanh(cache.c_new[j])` even though `forward_step` already computed that exact
value to produce `h_out[j] = o_gate[j] * tanh(c_out[j])`. Added a `tanh_c_new` field to
`CharLSTMCache`, filled once in `forward_step`, and read back (not recomputed) in `backward_step`
-- a plain cache lookup replacing a second `std::tanh` call per hidden dimension per step. Zero
arithmetic change (identical `double`, not an approximation). This falls under sub-phase 2 above,
which was already a rounding-error-level 0.3-0.5% of the total, so its effect is small but it is a
genuine instance of the "redundant recomputation" pattern the task asked to look for.

**Optimization *not* implemented, and why:** sub-phase 4 (weight-gradient outer products) is now
the largest remaining sub-phase (44.9% of a much-smaller total) and is genuinely compute-bound --
two full `4*hidden x hidden` outer products per step, already row-major/cache-friendly, with no
redundant work or algebraic simplification available without changing the gradient formula
itself. Per the task's explicit instruction not to touch gradient arithmetic without extremely
careful verification, this was left alone; see "Remaining opportunities" below.

### Part B: `bptt_ssm_update` allocation audit

Confirmed this is `CyphaLMModel::bptt_ssm_update` in `cyphalm_model.cpp` (the hybrid model's
BPTT-on-SSM-projection update, called once per `train_step` whenever BPTT-on-SSM is active) --
**not** `RpsmSequenceLayer`'s BPTT (a separate, already-audited code path from earlier today's
RPSM work; `rpsm_train_step`'s own perf-trace bucket is <0.01% of `train_step` time, confirming
it's a different, cold path for the default D17 hybrid profile).

Reading the function line-by-line (same technique `dafd677` used on `CharLSTMHead`) found five
fresh-allocated `std::vector<double>` locals every call (`grad_core`, `grad_field`, `delta_rows`,
`inv_v`, `grad_ctx`) plus two more (`grad_h`, `grad_s`) that were **not just fresh allocations but
provably redundant full copies** of a sub-range of `grad_ctx` that already existed -- introduced,
as far as can be told, purely so the code below could write `grad_h[r]` instead of
`grad_ctx[r]` (and `grad_s[r]` instead of `grad_ctx[d_state + r]`).

**Fix:**
1. Added five persistent `CyphaLMModel` scratch members (`bptt_grad_core_scratch_`,
   `bptt_grad_field_scratch_`, `bptt_delta_rows_scratch_`, `bptt_inv_v_scratch_`,
   `bptt_grad_ctx_scratch_`), filled via `.assign()`/`.resize()`-in-place instead of fresh
   construction, exactly the same pattern and safety argument as `hybrid_lstm_grad_scratch_`
   (`CyphaLMModel::train_step` runs on a single thread per model instance; every element is fully
   overwritten before being read each call).
2. Added a `matvec_transpose` out-param overload (mirroring `CharLSTMHead::backward_step`'s
   existing out-param pattern) so `grad_ctx` can be filled in place via the scratch member instead
   of receiving a fresh vector every call.
3. **Deleted `grad_h`/`grad_s` entirely** -- replaced every `grad_h[r]`/`grad_s[r]` read with
   direct `grad_ctx[r]`/`grad_ctx[d_state + r]` indexing. Same values, zero arithmetic change, two
   fewer allocations per call than even the scratch-member version would have needed.
4. **Left `delta`/`delta_slow` as fresh locals, deliberately** -- these get `std::move`'d into
   `bptt_buffer_`/`bptt_slow_buffer_` (which must own independent copies across the `bptt_steps`
   window for later averaging), so making them persistent members would only replace a "move,
   zero extra copies" pattern with a "fill member, then copy into the buffer" pattern -- no net
   allocation reduction, and arguably a regression. Confirmed by re-reading the move-vs-copy
   tradeoff explicitly before implementing (see the inline comment in `cyphalm_model.cpp`).
5. **Explicitly did not touch** `gria_->grad_v_cross_entropy(...)` or
   `ngram_fusion_->grad_field_x(...)` -- both return fresh vectors from *other* classes
   (`GRIALowRank`, `ExactNgramFusion`), out of scope for an audit of `bptt_ssm_update` itself. Noted
   as a candidate for a future, separately-scoped pass (see "Remaining opportunities").

Net effect: `bptt_ssm_update`'s instrumented time dropped from 15.507s to 8.151s (-47.4%) over the
same 59,922-call run, and its share of `train_step` time from 16.5% to 11.0%.

### Part C: `/arch:AVX2` determinism-gated trial

Machine has an Intel Xeon Gold 6242 (Cascade Lake, 32 logical processors as seen by this VM/host
view; supports AVX-512, not just AVX2) and an RTX 3090 (irrelevant to this CPU-side trial). Per
the task's explicit guidance, tried the safer default (`/arch:AVX2`) rather than jumping straight
to AVX-512.

**Build pitfall (worth flagging for anyone repeating this):** configuring a fresh build dir with
`-DCMAKE_CXX_FLAGS="/arch:AVX2"` **replaces** CMake's own default `CMAKE_CXX_FLAGS` for a
first-time configure rather than appending to it -- silently dropping `/EHsc` (exception handling)
and `/GR` (RTTI), which surfaced as a `C4530` warning ("C++ exception handler used, but unwind
semantics are not enabled") on two translation units. Since this codebase throws
(`std::runtime_error` in `char_lstm.cpp` and elsewhere), that would have been a real correctness
risk, not just a warning to ignore. Fixed by reconfiguring with the full default flag string
(`/DWIN32 /D_WINDOWS /GR /EHsc /arch:AVX2`, read back from a working build's `CMakeCache.txt`)
instead of a bare override.

**Determinism result: PASS.** Ran the identical `--profile d17 --n-train 20000 --n-eval 2000
--threads 1 --bench-seed 42` command on both the plain build and the `/arch:AVX2` build (same
commit, same seed, same source, only the compile flag differs). The two runs' entire stdout JSON
output is **byte-for-byte identical** (`Compare-Object` reports zero differences, not just an
equal `bpc` field) -- `bpc`, `bpc_lstm_only`, `hybrid_blend_logit`, `hybrid_gria_weight`, every
field. MSVC's default floating-point model (`/fp:precise`, not overridden by `/arch:AVX2` alone)
evidently does not perform the FMA-contraction/rounding change that made this a risk in the first
place -- `/fp:fast` would be the flag that actually risks that, and this trial never enabled it.

**Speed result: no benefit, within noise leaning slightly negative.** Two paired timing runs
(same machine, same session, same command) on the same commit that Parts A+B were measured on:

| Run | Plain build | `/arch:AVX2` build |
|---|---|---|
| 1 | 79,187 ms | 81,183 ms |
| 2 | 79,418 ms | 83,934 ms |

The AVX2 build was never faster across either run. This is a believable negative result, not a
methodology failure: at `lstm_hidden=128`, the matrices involved (`128x128` up to `512x128`) are
small enough that Parts A/B's cache-access-pattern fix already resolved the actual bottleneck
(memory access order, not raw FLOP throughput); auto-vectorization under plain `/O2` was
apparently already extracting what SIMD width there was to extract for loops this shape, and
explicit AVX2 codegen added no further win (and plausibly a small amount of overhead from wider
instruction encoding / potential SSE-AVX transition effects, though this wasn't isolated further).

**Verdict per the task's explicit decision rule:** determinism held, but there is no speed
benefit to offset even the small risk/complexity of a non-default compiler flag, so **this is not
adopted -- the default build (`native/CMakeLists.txt`, all presets) is completely unchanged.**
Reported here as a measured, negative-but-informative trade-off for the record, exactly as the
task asked for in the "can't get both" branch of its decision tree (except here it's "got
determinism, didn't get speed," a related but distinct outcome worth distinguishing from "got
speed, lost determinism").

### Throughput: before/after (Parts A+B)

Paired same-session runs, same MSVC build dir (`native/build_perf2`), same commit, toggled via
`git stash`/`git stash pop` on just the four touched files so the comparison isolates *only*
today's Part A+B changes (not the MSVC toolchain, held constant across both sides of this specific
comparison):

| | Run 1 | Run 2 | Avg | chars/sec |
|---|---|---|---|---|
| Before (this session's Parts A+B) | 96,331 ms | 93,002 ms | 94,667 ms | 211.3 |
| After (this session's Parts A+B) | 79,187 ms | 79,418 ms | 79,303 ms | 252.2 |
| **Speedup** | | | | **+19.4%** |

(`--profile d17 --n-train 20000 --n-eval 2000 --threads 1 --bench-seed 42`, matching the exact
command used for Part 1's own before/after table.)

### Running total: today's entire optimization thread

Being explicit about a real complication: **the toolchain changed partway through today's work**
(MinGW -> MSVC, a separate task documented in `docs/reports/MSVC_TOOLCHAIN_MIGRATION_2026-07-12.md`,
itself measured at ~28% faster than MinGW on this exact D17 command/commit). That means the
136.3 -> 211.3 chars/sec jump below is **not** attributable to today's algorithmic work -- most of
it is the toolchain switch. Presented honestly, in the order things actually happened, not
collapsed into one misleading multiplier:

| Stage | chars/sec | Toolchain | What changed |
|---|---|---|---|
| Earlier baseline (context, pre-this-report) | ~96 | MinGW | (pre-existing) |
| Before Part 1 fix | 126.4 | MinGW | (pre-existing) |
| **After Part 1** (`dafd677`, allocation-reuse fix) | **138.3** | MinGW | +9.5% (Part 1) |
| *(MSVC toolchain migration, `f566fee`, separate task)* | *(not a like-for-like number; see that report's own 104.65s vs. 144.28s A/B)* | *MinGW->MSVC* | *~+28% (toolchain, not algorithm)* |
| Before Part 2 fix (this task, MSVC, same commit as `f566fee`) | 211.3 | MSVC | (post-toolchain-switch baseline) |
| **After Part 2** (this task: sub-phase cache-pattern fix + `bptt_ssm_update` audit) | **252.2** | MSVC | **+19.4% (Part 2)** |

**Honest cumulative read:** 252.2 / 96 = **2.63x** (+162.7%) versus the earliest baseline cited in
this thread, but roughly a third of that multiplier (the toolchain-migration report's own ~28%
figure) is a compiler/toolchain change, not an algorithmic improvement, and the two toolchains'
absolute numbers were never measured back-to-back on identical hardware conditions across the
*entire* chain (only within each pass's own paired A/B). The two numbers that *are* directly,
rigorously comparable -- because they're same-commit, same-build-dir, same-session paired A/Bs --
are Part 1's MinGW **+9.5%** and Part 2's MSVC **+19.4%**, both purely from allocation/cache-access
fixes with zero arithmetic change and confirmed bit-identical BPC.

### Remaining opportunities (not implemented this pass)

1. **Sub-phase breakdown inside `predict_next` (now 40.6% of `train_step`, up from 31.9% purely
   because `lstm_backward`/`bptt_ssm_update` shrank) -- Effort: Low, Impact: potentially High.**
   This is now the single largest phase in `train_step` and hasn't been broken down at all (this
   report's own §5 item 5, still not done). Given Parts A/B's finding that transpose-matvec
   cache-access patterns were the dominant hidden cost in `lstm_backward`, `predict_next`'s
   forward-pass LSTM cell (`matvec_rowmajor` calls in `char_lstm.cpp::forward_step`) is *already*
   row-major-friendly by construction (reads `M[r*cols+c]` with `c` in the inner loop, i.e. no
   stride) -- but the GRIA field lookup/blend side of `predict_next` hasn't been profiled in
   isolation and is a plausible next target.
2. **`grad_v_cross_entropy`/`grad_field_x` allocation audit (`GRIALowRank`/`ExactNgramFusion`,
   different files) -- Effort: Medium, Impact: Medium.** Explicitly out of scope for this pass
   (Part B's audit was scoped to `bptt_ssm_update` itself), but both are called once per
   `train_step` in the default D17 path and return freshly-allocated vectors every call --
   plausibly the same allocation-churn pattern as everything fixed so far, just one layer removed.
3. **Weight-gradient outer products (`dWx`/`dWh`, now 44.9% of the shrunk `lstm_backward` total)
   -- Effort: High, Risk: gradient-correctness, Impact: Unknown.** Already cache-friendly and
   allocation-free; the only remaining lever is genuine vectorization/blocking of the outer-product
   arithmetic itself, which risks touching gradient math directly. Per this task's explicit
   instruction to avoid exactly that risk without extremely careful verification, left alone. If
   pursued, would need the same bit-identical-BPC verification rigor as Part C, and ideally hand
   micro-benchmarking of the outer-product loop in isolation before touching the real code.
4. **AVX-512** (this CPU supports it) -- not tried, since even the safer AVX2 showed no benefit;
   unlikely AVX-512 would do better on the same small-matrix, now-cache-optimized workload, but
   noted for completeness in case a future pass targets a much larger `lstm_hidden` (the sibling
   agent's concurrent hidden=512 run makes this a live, relevant question for a future session).
5. **CUDA / batch-GPU offload** -- per the MSVC migration report, the toolchain blocker (§4 of
   this report's original text) is now resolved; a CUDA build was demonstrated working on this
   machine's RTX 3090 in that report. The core D17 online training recurrence itself remains
   inherently sequential (each character step depends on the previous step's hidden state) and
   thus not GPU-parallelizable without an algorithm change, but `CyphaLMBatch`'s batch/eval-side
   path (already CUDA-capable per that report) could plausibly benefit and was not explored here
   (out of this task's explicit A/B/C scope).

---

## Follow-up (2026-07-12, part 3)

**Scope:** execute Part 2's own §"Remaining opportunities" item 1 -- sub-phase-profile
`predict_next` (the LSTM+GRIA hybrid forward pass) itself, the single largest unbroken-down phase
in `train_step`, and fix whatever the profiling data supports. **Did not touch**
`native/build_math`, `bench/BASELINE_LOCK.json`, `bench/BASELINE_REPORT.md`, the overnight
orchestration scripts, or `native/build_deff` (the sibling hidden=512 D_eff run).

**HEAD at start:** `0851483` (Part 2's own commit). Fresh build dir: `native/build_perf3` (MSVC,
`windows-vs2026-release` preset).

### Re-measured top-level breakdown

Before assuming the Part 2 write-up's cited **40.6%** figure still held, re-ran
`cyphalm_bench_native --profile d17 --n-train 20000 --n-eval 256 --threads 1 --bench-seed 42` with
`CYPHA_PERF_TRACE=1` on a fresh build at current HEAD, *before* touching any code. Result:
`predict_next` = **33.68%** of `train_step` (65.1s of 193.3s instrumented, 59,922 calls) --
noticeably *lower* than the 40.6% cited in Part 2, confirming that figure's own caveat ("its
absolute share grew as a fraction now that lstm_backward/bptt_ssm_update shrank -- this doesn't
necessarily mean predict_next itself got slower, re-measure to confirm"): it was never really
"40.6% and climbing", that number was itself noisy against this same shared, heavily-loaded
64-core machine (see "Measurement environment note" below) -- re-measurements throughout this pass
put `predict_next`'s share anywhere from **33.7% to 40.3%** depending on concurrent system load,
with `lstm_backward` (42-46%) consistently the larger of the two. `predict_next` is still the
right next target (it's the largest *unexplored* phase, whether it's exactly 34% or 40%), just not
because it's mechanically "growing" -- it's cross-run measurement noise on an increasingly
contended machine.

### `predict_next` sub-phase breakdown

Added a `PredictNextSubPhaseTrace`/`PredictScopeTimer` pair inside `CyphaLMModel::predict_next`
(`native/src/cyphalm/cyphalm_model.cpp`), gated by the same `CYPHA_PERF_TRACE` env var, mirroring
the `BackwardSubPhaseTrace`/`BwdScopeTimer` pattern Part 2 used for `lstm_backward`. Eight buckets,
consistent across every run of this pass (four independent measurements, two different corpora,
percentages stable to within ~1.5pp of each other even though absolute instrumented seconds varied
2-3x with machine load -- see below):

| # | Sub-phase | Share of `predict_next` |
|---|---|---|
| 1 | embed lookup + history record | ~0.13-0.15% |
| 2 | **SSM step + hebbian hooks + field projection** | **~35.4-35.9%** |
| 3 | DIF (kernel-LLR) predict | ~3.7-3.9% |
| 4 | GRIA input build (memory/context-bank/ngram-fusion + rpsm/gng) | ~22.2-23.7% |
| 5 | GRIA field forward (logits+softmax) | ~4.4-4.6% |
| 6 | **LSTM forward step (gates) + hidden-state bookkeeping** | **~30.6-31.8%** |
| 7 | hybrid blend combination (`blend_logit`/`blend_log_probs`) | ~1.2-1.3% |
| 8 | tail (token history + top-k + bookkeeping) | ~0.4-0.46% |

So `predict_next` isn't dominated by the LSTM cell the way its name suggests -- the **SSM step**
(bucket 2, `CellAISSM::step`, called from inside `predict_next` for the hybrid path's
context-modulation) is actually the single largest sub-phase, narrowly ahead of the LSTM forward
gates themselves (bucket 6), with GRIA's context/n-gram-fusion input assembly (bucket 4) a clear
third. This is a materially different shape than `lstm_backward`'s profile (which Part 2 found to
be ~45% dominated by one thing, the `dWx`/`dWh` outer products) -- `predict_next`'s cost is spread
across three separate subsystems (SSM, GRIA input assembly, LSTM gates) each individually
cheap-ish, not one dominant bottleneck.

### Fix implemented: allocation-churn removal across all four subsystems `predict_next` touches

Per-call heap-allocation audit of every function `predict_next` calls (same methodology as Parts
1/2/A/B), found the identical allocation-churn pattern in **every one of the four subsystems**
above, none previously touched by Parts 1/2:

- **`CyphaLMModel::predict_next` itself:** `log_g`/`log_l` (`std::vector<double>(vocab_size)`) and
  `h_new`/`c_new` (LSTM forward-step outputs) were fresh `std::vector` locals every character step.
  Replaced with persistent `log_g_scratch_`/`log_l_scratch_`/`predict_lstm_h_scratch_`/
  `predict_lstm_c_scratch_` members (`.assign()`/`.resize()` in place); `h_new`/`c_new` are now
  `std::swap`'d with `lstm_h_`/`lstm_c_` instead of move-assigned, so the same two buffers
  permanently alternate roles instead of one side being freed and a fresh one allocated every step.
- **`CellAISSM::step`** (bucket 2, the largest sub-phase): `matvec` returned freshly-allocated
  `std::vector<double>` by value on every call (multiple times per `step()`), and `step()`'s own
  locals (`layer_input`, `Wh`-product, `Ws`-product, `ctx`) were fresh allocations too. Added an
  out-param `matvec(..., std::vector<double>& out)` overload (value-returning overload now
  delegates to it) and four `CellAISSM` member scratch buffers (`step_layer_input_scratch_`,
  `step_wh_scratch_`, `step_ws_scratch_`, `step_ctx_scratch_`) filled in place.
- **`GRIALowRank::logits`/`forward`** (bucket 5): `h` (rank-sized), `z`/`probs` (vocab-sized) were
  fresh locals every call. Replaced with `thread_local static std::vector<double>` -- `thread_local`
  (not member/plain-static) specifically because `CyphaLMBatch::parallel_batch`
  (`cyphalm_batch.cpp`) calls `GRIALowRank::forward` concurrently on one *shared* instance from
  multiple worker threads (verified by reading that call site), so a plain static would race; a
  member would still race since the instance is shared. `predict_next` itself only ever runs
  single-threaded per model instance, so this dual-use function needed the more conservative of the
  two safety patterns.
- **`NgramFusion::forward`** (feeds bucket 4) + **`CyphaLMModel::ngram_embedding_vector`** (also
  feeds bucket 4): `forward()`'s `pos_weighted`/`embed_part`/`gate_in`/`gate_logits` locals and
  `matvec`'s return-by-value were the same pattern, fixed the same two ways as above (`thread_local`
  in `NgramFusion::forward`/`matvec` for the same `parallel_batch`-shares-one-instance reason as
  `GRIALowRank`; a new out-param `ngram_embedding_vector(std::vector<double>& out)` overload plus a
  `mutable ngram_embed_vec_scratch_` member on `CyphaLMModel` for `gria_input_core`'s call site,
  since `predict_next`'s own caller chain is single-threaded and `gria_input_core` is `const`).

**Zero arithmetic/gradient change in any of the above** -- every reused buffer is either
`.assign()`'d (guarantees every element written, matching a fresh zero-initialized vector) or fully
overwritten element-by-element by the same loop that used to write into a fresh allocation, before
any element is read. No loop reordering, no algebraic simplification, no summation-order change --
this pass is allocation-reuse only, unlike Part 2's Part A which also included a proven-safe
loop-interchange.

### Determinism

The D17 BPC pin was checked at **every single stage** of this pass -- before any instrumentation,
after instrumentation-only, after the fix, and again after a final from-scratch rebuild once all
changes were in place -- and came back **bit-identical every time**:
`bpc = 3.4748796956404946`, `bpc_lstm_only = 3.472557891154332` (16 significant digits, `wikitext2`
corpus, `--bench-seed 42`). `native_d17_wikitext_smoke` and the full requested CTest filter
(`ctest --test-dir native/build_perf3 -C Release -R "cyphalm|d17|predict|gria"`, 16 tests) both pass
100% on the final rebuilt binary.

### Measurement environment note (read before the throughput numbers below)

This pass ran on a much more heavily loaded version of the same shared 64-core machine than Parts
1/2 did earlier today: the live `native/build_math` overnight benchmark and the sibling
hidden=512 `native/build_deff` D_eff confirmation run were both still active, a *third* sibling
agent was concurrently landing commits and rebuilding in `native/CMakeLists.txt` and several
unrelated `native/` files in this same working directory (confirmed via `git status` showing ~15
files this task never touched, mid-edit), and `git worktree list` showed three more Cursor-managed
worktrees active elsewhere on this box. `Get-Process` during one of this pass's own timing runs
showed several other processes with **tens of thousands of accumulated CPU-seconds** each, and this
pass's own single-threaded benchmark process visibly CPU-starved (only ~93 CPU-seconds consumed
over several minutes of wall time). Concretely: the exact same unmodified binary, run
back-to-back with no code change in between, varied from **77s to 665s** wall-clock for the
identical `--n-train 20000 --n-eval 256` command over the course of this pass -- an order of
magnitude of pure scheduling noise, far larger than any effect this fix could plausibly produce.
Wall-clock end-to-end chars/sec is therefore **not a trustworthy signal for this specific pass** and
is reported below only for transparency, alongside the more robust evidence.

To get a signal not confounded by whichever *other* files a concurrent sibling agent happened to be
editing in the shared working directory at that moment (a real, observed problem this pass hit
once -- a `git stash`/`pop` round-trip briefly lost this task's own `cyphalm_model.hpp` scratch-buffer
declarations to an OS-level file-lock error, caught and re-applied before it could contaminate any
result), the before/after comparison below was redone a second time in a fully isolated
`git worktree` (pinned to Part 2's commit `0851483` for "before", this task's diff copied in for
"after", built in its own directory) -- immune to concurrent edits to unrelated files, though still
subject to the same machine-wide CPU scheduling contention described above.

### `predict_next`'s own instrumented CPU time: before/after

The one measurement that *is* robust to whole-process wall-clock noise: the sum of the eight
`PredictScopeTimer` buckets above, i.e. actual `steady_clock` time spent inside `predict_next`
itself, same-session same-build-dir paired runs (`wikitext2` corpus, 60,434 `predict_next` calls
both sides):

| | Instrumented `predict_next` time | vs. before |
|---|---|---|
| Before fix (instrumentation added, no allocation-reuse yet) | 34.459s | -- |
| After fix, run 1 | 30.860s | **-10.4%** |
| After fix, run 2 (re-check after final rebuild) | 30.202s | **-12.3%** |

Both after-fix runs agree with each other to within 2%, and both are consistently below the
before-fix number, despite being measured at different points across this pass (i.e. under
different concurrent system load) -- this is the fix actually working, isolated from the
whole-process noise described above. The per-bucket shift is broadly uniform (each of the four
touched subsystems -- buckets 2/4/5/6 -- individually improved 8-14%; the untouched buckets 1/3/7/8
moved by less than their own run-to-run noise, as expected since nothing in them changed).

### End-to-end wall-clock throughput (reported for transparency, not trusted -- see note above)

Direct `--n-train 20000 --n-eval 256 --threads 1 --bench-seed 42` timings, no `CYPHA_PERF_TRACE`
overhead:

| | Samples (ms) | Median |
|---|---|---|
| Before (main build dir, `git stash` toggle) | 118,426 / 95,383 / 120,954 | 118,426 |
| Before (isolated worktree, `0851483`, fallback corpus) | 87,927 / 80,153 | 84,040 |
| After (main build dir, final rebuilt binary) | 665,083 (one run; machine load had worsened sharply by this point -- see note above) | -- |

No usable before/after delta can be extracted from these numbers today; the run-to-run spread
within a single unchanged binary (77s-665s across this pass) is far larger than the ~10-12%
internal `predict_next` improvement could move the needle on an end-to-end number where
`predict_next` is only ~34-40% of the total. **Back-of-envelope expected effect**, using the robust
internal number instead: `predict_next` at ~35% of `train_step` share x ~11% internal improvement
implies train_step should end-to-end improve by roughly **~3-4%** once machine load is quiet enough
to measure it -- directionally consistent with, but not independently confirming beyond, Part 2's
own paired-run methodology. Recommend re-running this specific A/B once the overnight/`build_deff`
jobs have finished and the machine is quiet, using the exact `git stash`-toggle or isolated-worktree
methodology already set up here (both are fast to repeat).

### Running total: today's entire optimization thread (updated)

Extending Part 2's own honest, uncollapsed table:

| Stage | chars/sec | Toolchain | What changed |
|---|---|---|---|
| Earlier baseline (context) | ~96 | MinGW | (pre-existing) |
| Before Part 1 fix | 126.4 | MinGW | (pre-existing) |
| **After Part 1** (`dafd677`) | **138.3** | MinGW | +9.5% (Part 1) |
| *(MSVC toolchain migration, `f566fee`)* | *(~+28%, not algorithmic)* | *MinGW->MSVC* | *toolchain only* |
| Before Part 2 fix | 211.3 | MSVC | (post-toolchain baseline) |
| **After Part 2** (`0851483`) | **252.2** | MSVC | **+19.4% (Part 2)** |
| After Part 3 fix (this task) | *not independently wall-clock-confirmed today (see note above)* | MSVC | **~+3-4% estimated** from the robust internal `predict_next` measurement |

Same honest framing as Part 2: the toolchain-migration jump is not algorithmic and shouldn't be
folded into a single multiplier with the allocation/cache-access fixes. Unlike Parts 1 and 2, this
pass's own contribution to the chain is an *estimate* derived from the internal, noise-robust
`predict_next` timing rather than a directly wall-clock-measured number, because today's machine
load made the direct measurement unreliable (see "Measurement environment note"). The fix itself is
real, deterministic-verified, and code-reviewable regardless of which exact multiplier it turns out
to be once independently re-measured on a quieter machine.

### Recommendation

**This specific hot path (`predict_next`/`lstm_backward`/`bptt_ssm_update`) has reached diminishing
returns for the allocation-reuse pattern that has driven all three of today's fixes.** Every
allocation this pass could find inside `predict_next`'s call tree has now been converted to
scratch-buffer reuse (this task) or was already allocation-free (`lstm_backward`'s weight-gradient
outer products, `bptt_ssm_update`, both audited and fixed in Part 2). What's left in each of
`predict_next`'s three largest sub-phases -- the SSM step's small matvecs, GRIA's input-assembly
matvecs, the LSTM cell's gate matvecs -- is now genuinely compute-bound, cache-friendly (row-major
by construction, per Part 2's finding that this codebase's matvec loops were already correctly
ordered), and not obviously reducible without either (a) touching the actual gate/field/SSM
arithmetic (out of scope for the safety discipline this whole thread has followed), or (b) a
structural change to how many of these subsystems run per character step (e.g. whether DIF/
n-gram-fusion/GNG/RPSM all need to be computed on every single step at this model's default
configuration -- not investigated this pass, but a plausible next angle if predict_next specifically
is revisited).

**Recommend redirecting future performance work away from micro-optimizing this specific per-step
critical path and toward one of:** (1) the batched-CUDA-caller opportunity flagged by the sibling
accel-audit task (`CyphaLMBatch`'s batch/eval path, already CUDA-capable per the MSVC/CUDA migration
report, genuinely GPU-parallelizable unlike the inherently-sequential D17 online-training
recurrence itself); (2) a from-first-principles look at whether every one of `predict_next`'s four
subsystems (SSM, DIF, GRIA, n-gram-fusion, LSTM) needs to run on literally every character step at
the D17 default config, i.e. reducing the *amount* of per-character work rather than further
speeding up the existing amount; or (3) simply re-running today's Part 3 A/B on a quiet machine to
get a trustworthy wall-clock number for the record, since the code change itself is done and
determinism-verified regardless.

---

## Follow-up (2026-07-17, part 4) — per-step work reduction

**Scope:** execute Recommendation §2 from Part 3 — skip config-gated dead subsystems on the D17
default `hybrid_gria_lstm` path rather than micro-optimize existing work. **Did not touch**
`native/build_math`, `native/build_deff`, `bench/BASELINE_LOCK.json`, `bench/BASELINE_REPORT.md`,
or running `cypha_*` / `cyphalm_bench_native` processes.

**HEAD at start:** `e0116e4`. Fresh build dir: `native/build_perf_skip` (MSVC Release, VS Build Tools
18).

**Bottom line:**
- **Root cause:** at the locked D17 profile (`bench/config/profiles/cyphalm_d17_wikitext.json`),
  GRIA input is assembled exclusively via `NgramFusion` (`ngram_fuse_split=true`,
  `ngram_context=3`) — `gria_input_core()` never reads `DIFPredictOutput::mean` or epistemic
  variance on this path. With all forget-gate/OOD/GNG-controller flags off (the default), DIF
  predict output is also unused by `hybrid_forget_gate_scale()` and the hybrid blend. Yet
  `CyphaDIF::predict()` and `CyphaDIF::train_step()` still ran every character step (~3.9% +
  ~2.0% of instrumented `train_step` time in the quiet pre-fix trace below).
- **Fix:** added `dif_subsystem_affects_forward(cfg)` in `cyphalm_model.cpp` — returns `false` for
  Hybrid/GriaNgram/Rpsm/AblationNoSsm configs on the ngram-fuse-split path unless a flag that
  actually consumes DIF output is enabled (`use_tau_forget_gate`, `use_reu_forget_gate`,
  `use_ood_branching`, `use_discriminative_feedback`, `use_gng`+`use_gria_controller`, or
  Full/SsmGriaNoLstm modes that route DIF through `proj_dif_`). When inactive:
  - **`predict_next`:** skips `dif_->predict()` (zeros `last_dif_out_` like the existing
    `SsmGria`/`AblationNoSsm` branch) and passes `nullptr` into `build_gria_input()`.
  - **`train_step`:** skips the online `dif_->train_step()` block entirely.
  - **Micro-fast-path:** when all three forget-gate flags are off, uses literal `1.0` instead of
    calling `hybrid_forget_gate_scale()` (provably identical at default config).
- **Determinism:** **bit-identical BPC** — `bpc = 3.9922568809095362`, `bpc_lstm_only =
  3.9849446535748125` before and after (`wikitext2`, `--bench-seed 42`, `--n-train 6000`). `ctest
  -R native_d17_wikitext_smoke` passes on `native/build_perf_skip`.
- **Measured speedup (instrumented, quiet pre-fix run vs post-fix run with skip active):**

| Bucket | Before (s) | Before (%) | After (s) | After (%) | Δ |
|---|---|---|---|---|---|
| `predict_next` → DIF predict sub-phase | 0.606 | 3.88% of predict_next | 0.037 | 0.07% | **−94%** |
| `train_step` → `dif_train_step` | 0.852 | 2.00% of train_step | 0.011 | 0.006% | **−99%** |
| `train_step` total (instrumented) | 42.68 | — | — | — | **~−3.4% estimated** (1.46s removed) |

(Quiet pre-fix trace: `build_perf3`, 17,978 `train_step` calls, 42.68s instrumented. Post-fix DIF
buckets measured on `build_perf_skip` with skip active; absolute post-fix totals were confounded by
heavy concurrent load from overnight `build_math`/`build_deff` jobs — same environment caveat as
Part 3.)

**Wall-clock note:** a quiet pre-fix end-to-end run measured **43,530 ms** (138 chars/sec). Later
paired runs on this shared 64-core host varied **31–285 s** for the identical command depending on
contention; wall-clock A/B is not trustworthy here. The eliminated ~1.46s/call-batch of DIF work
(~3.4% of the quiet 42.68s instrumented `train_step` total) is the robust signal.

**What was intentionally NOT skipped (still required at D17 default):**
- SSM step + BPTT (`bptt_steps=64`) — SSM context feeds GRIA via `project_field()` and BPTT updates
  `W_fast` every 64 steps regardless of `train_ssm=false`.
- GRIA forward/backward + n-gram fusion/count paths — active training heads even at ~99.6% LSTM blend.
- LSTM forward/backward + hybrid blend learning — dominant compute; unchanged.
- Hebbian stack, EWC, GNG, RPSM — already null/off at default config (no call cost).

**Remaining opportunities:** subsystems that *do* run but could be throttled structurally (e.g. BPTT
every 64 steps vs every step, memory compress/store cadence, n-gram prior refresh) — each needs its
own bit-identical proof and is higher risk than this pass's pure "DIF output unreachable" gate.

