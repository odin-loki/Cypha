> **Note (2026-07-18):** Draft research/audit note. Living status is [CONTINUUM_CLOSEOUT_2026-07-18.md](CONTINUUM_CLOSEOUT_2026-07-18.md) + [CYPHA_BILL_OF_WORK.md](../../CYPHA_BILL_OF_WORK.md). Release is **v2.3.24** / CI is **windows_msvc** (not MinGW). Treat numbers below as proposals unless cross-checked against those sources.
# Cypha — Performance Research & Profiling Roadmap

**Prepared for:** Odin Loch
**Basis:** Full read of `Cypha-main` @ v2.2.8 (native core, CyphaLM, accel backend, CUDA kernels, bench tree) plus your existing profiles: `PERFORMANCE_PROFILE_2026-07-12.md` (train_step: `lstm_backward` 45.3%, `predict_next` 31.7%, `bptt_ssm_update` 18.2%), `INFER_LATENCY_PROFILE_2026-07-17.md` (score_matrix n=32 → 0.16 µs/row vs 3.60 µs single), `GPU_TRAINING_GAP_2026-07-18.md`, and the Optimality Plan status table.

**Framing:** Your measured throughput ceiling today is ~130–140 chars/sec single-threaded D17 training. The evidence in your own reports says this is not an algorithm problem — it is a *scalar-double-precision, hand-rolled-kernel, no-SIMD, MinGW-libm* problem, plus a serialization problem on the serving path. The realistic aggregate headroom below is **10–50× on CPU training** and **1–2 orders of magnitude on GPU**, before touching any model math.

---

## 0. Fix the measurement substrate first (prerequisite, ~1 day)

Everything else depends on trustworthy numbers, and right now you don't have a working sampling profiler.

**0.1 — Get a real profiler.** `gprof` is confirmed dead under MinGW (your own report). Three viable paths, in order of preference:

1. **WSL2 + `perf`** — you already have `wsl_bench_gpu.sh` / `wsl_verify.sh`. `perf record -g --call-graph dwarf` + `perf report` / flamegraphs gives you instruction-level attribution, and `perf stat -d` gives IPC, cache-miss rate, branch mispredict rate — the counters that tell you *why* a loop is slow, not just *that* it is. This is the single highest-value profiling upgrade available to you and costs nothing.
2. **Windows-native samplers that don't need MSVC:** `samply` (Firefox Profiler UI, works on MinGW binaries with DWARF symbols), Tracy (instrumentation-based, in-app frame view — excellent for the train_step loop since you already have `CYPHA_PERF_TRACE` scopes to convert), AMD uProf / Intel VTune (both free, both sample MinGW binaries fine).
3. **Install MSVC Build Tools anyway** — it also unblocks CUDA on the Windows box (your GPU gap report's stated blocker) and lets you A/B MSVC STL vs libstdc++ codegen.

**0.2 — Valgrind under WSL.** `callgrind` for exact call-graph costs and `cachegrind` for simulated cache behaviour on the LSTM backward loop. Slow to run but deterministic — perfect for A/B-ing a layout change without host noise.

**0.3 — Kill the noise.** Your infer report says CyphaLM wall times vary ~2× run-to-run. Before any optimization A/B: pin to physical cores (`start /affinity` or `taskset`), Windows High Performance power plan / disable core parking, warmup iterations, report median-of-9 with IQR. Adopt Google Benchmark or nanobench for the microkernels (`score_matrix`, `matvec_rowmajor`, gate activations, `softmax_rows`) so each has a standalone, repeatable benchmark with statistical treatment.

**0.4 — Perf regression lock.** You already lock BPC (`BASELINE_LOCK.json`). Add a *throughput lock*: chars/sec on a fixed D17 slice, µs/row on `score_matrix` n∈{1,32,256}, µs/token on `generate`. CI gate at −5%. Otherwise the wins below erode silently.

**0.5 — Counter-driven questions to answer before optimizing** (one `perf stat` session each):

| Question | Counter | Decides |
|---|---|---|
| Is `matvec_rowmajor` memory- or compute-bound at hidden=128? | L1/L2 miss rate, IPC | SIMD vs layout work first |
| Is the backward transposed-weight access (noted at `char_lstm.cpp:110–115` as majority of backward) actually missing cache? | LLC misses on backward | Whether to keep a transposed weight copy |
| How expensive are `exp`/`tanh` in MinGW libm? | cycles in `libm` symbols | Vector-math library priority |
| Does OpenMP row-parallel actually scale on your core count? | per-thread cycles, migrations | Threshold retune |

---

## 1. Build & compiler (near-free wins, ~1–2 days)

Your Release build is plain `-O2` (RelWithDebInfo confirmed `-O2 -g` in your report), **no `-march`, no LTO, no PGO**. This tier is pure configuration.

- **1.1 `-O3 -march=native`** for local research builds; `-march=x86-64-v3` (AVX2+FMA baseline) for release bundles. Every dot product in `score_matrix`, `matvec_rowmajor`, and the Mahalanobis loops currently compiles to scalar SSE2 doubles. FMA alone is ~2× on those loops; AVX2 auto-vectorization on the clean `for (j < d)` loops is another 2–4× where the compiler takes it.
- **1.2 LTO/IPO** (`-flto`): `matvec_rowmajor` is defined in *two* translation units (`char_lstm.cpp:95`, `gria_lowrank.cpp:15`) and called cross-TU; LTO enables inlining and loop fusion across those seams.
- **1.3 PGO**: the D17 training loop is the ideal PGO workload — one dominant path, branchy gate dispatch (`use_axiom`/`use_eml`/`use_sr` checked *per element per step*). GCC `-fprofile-generate/-fprofile-use` typically gives 5–15% on exactly this shape. BOLT on Linux afterwards if you want to squeeze layout.
- **1.4 Floating-point contract**: full `-ffast-math` will break your golden fixtures; but `-ffp-contract=fast` (FMA fusion only) and `-fno-math-errno -fno-trapping-math` are usually golden-safe within tolerance and unlock vectorized libm inlining. Re-baseline goldens once (your Phase 0 already retired Python parity, so you're free to).
- **1.5 Compiler bake-off**: MinGW GCC 13 vs Clang vs MSVC on the three hot kernels. MinGW's `exp`/`tanh` (`msvcrt` era libm) are notoriously slow; Clang + compiler-rt or MSVC's UCRT can differ 2–3× on transcendental-heavy loops. Cheap experiment, occasionally huge.
- **1.6 Allocator swap**: link `mimalloc` (drop-in, excellent on Windows where the default heap is weak). Your 2026-07-12 fix removed the worst churn, but `infer_cpu.cpp` still has ~46 vector alloc sites and the REST path allocates per request. mimalloc is a one-line link flag and routinely worth 3–10% on alloc-touched paths.

---

## 2. The LSTM hot path (biggest CPU prize: 77–95% of train_step)

Your profile: `lstm_backward` 45.3% + `predict_next` 31.7% + BPTT 18.2%. The forward you showed me is: two hand-rolled matvecs (`Wx`: 4h×h, `Wh`: 4h×h, h=128), then a **per-element gate loop with per-element mode dispatch** (axiom grammar lookup / `eml_nand` / `sigmoid` chosen inside the j-loop), then output `Wy`: vocab×h matvec. Research areas, in leverage order:

**2.1 — Precision: double → float (the headline).** Everything is `double`. There is no statistical justification for fp64 in LSTM training at this scale — fp32 (with fp64 master accumulators if you're cautious) is the industry norm and halves memory bandwidth while *doubling* SIMD lane count (AVX2: 4 doubles → 8 floats). Combined with §1.1 this is the single largest CPU multiplier available (~3–4× on the matvec-bound fraction). Research question to settle empirically: does D17 BPC move at fp32? Run the pin with an fp32 build behind a compile-time `cypha::real` typedef. Also the prerequisite for GPU wins (§5).

**2.2 — Matvec → GEMM restructuring.**
- Fuse `Wx` and `Wh` into one `[Wx|Wh] (4h × 2h)` matrix acting on `[x;h]` — one kernel launch/loop instead of two, better cache reuse.
- Across the BPTT window, the *input* projections `Wx·x_t` for all t are independent of recurrence → batch them into a single `(T × h) · (h × 4h)` GEMM before the sequential loop. Standard cuDNN-style RNN optimization; applies on CPU too.
- Then benchmark three backends for that GEMM/matvec: (a) hand-SIMD with `xsimd`/Highway/`std::experimental::simd`, (b) **OpenBLAS/BLIS** `sgemv`/`sgemm`, (c) Eigen. At h=128 the matrices are small enough that BLAS call overhead matters — measure, don't assume. A dedicated small-GEMM library (libxsmm) is the research-grade option here and is *built* for exactly 128-sized panels.

**2.3 — Backward transposed access.** Your own comment (`char_lstm.cpp:110–115`) says column-strided reads of `Wy/Wx/Wh` are the majority of backward cost. Candidate fixes to profile: keep persistent transposed copies (`WxT`, `WhT`, `WyT`) updated in `apply_grads` (memory cost trivial at h=128), or blocked/tiled transposed matvec. Cachegrind will adjudicate.

**2.4 — Gate activation vectorization.** Hoist the `use_axiom`/`use_eml`/`use_sr` dispatch *out* of the j-loop (one branch per step, four flat loops per mode), then vectorize each mode's math:
- `sigmoid`/`tanh`: SLEEF or a padé/minimax rational approximation with documented ULP bound — vectorized `exp` is typically 4–8× scalar libm, and on MinGW possibly more (§1.5).
- `eml_nand` / `apply_axiom_gate`: audit their inner math for vectorizability; if axiom grammar entries force gather-style per-element lookups, research an SoA re-layout of the grammar tables so gates of the same op vectorize together.
- The `sr_laws_.predict()` path calls a per-element virtual/struct predict — same treatment: batch by law type.

**2.5 — Output layer (`Wy`: vocab×h).** For char-level vocab (~100–256) this is fine; if you push toward larger token vocabs the standard research menu applies: adaptive/sampled softmax for training, and for generation compute logits top-k lazily. Even now, `generate` at 1.8 ms/token vs `predict_next` at 0.69 ms says ~60% of generate is *outside* the model forward — profile what (sampling, JSON, context bookkeeping, RNG).

**2.6 — BPTT (18.2%).** Sweep truncation length vs BPC (quality-throughput frontier); research gradient checkpointing trade (recompute forward vs store activations — at h=128 storage is cheap, so likely keep storing, but measure the activation-buffer cache footprint across the window).

**2.7 — Sequence/data parallelism for training.** `CyphaLMModel::train_step` is sequential by design. Research directions that preserve your online-learning story: (a) batch multiple independent sequences per step (you already have `CyphaLMBatch` parallel workers touching a shared head — audit its scaling and false sharing), (b) Hogwild-style lock-free updates at h=128 (dense small weights = high collision but literature says it still converges), (c) local-SGD/periodic-merge reusing your existing `federated_aggregate` machinery as an intra-node parallelism primitive. That last one is a genuinely interesting research angle unique to your codebase.

---

## 3. The DIF score path (`score_matrix` family)

`cpu_parallel_score_matrix` inner loop: `cross += (H[i·d+j] − mu0[j]) · inv_v[j] · D[k·d+j]`.

- **3.1 — Fold `inv_v` into `D` offline.** `inv_v[j]·D[k·d+j]` is input-independent → precompute `D'[k][j] = inv_v[j]·D[k][j]` once per model update. Inner loop becomes a plain dot of `(H−mu0)` against `D'` rows — i.e. the whole thing is `(H − mu0) · D'ᵀ`, a **GEMM**. Centre H once per batch, call `sgemm`, add the `−½D_sq − u_k + ctx` rank-1 row correction. This converts your most-called inference kernel into a BLAS call and composes with fp32 (§2.1).
- **3.2 — The n=1 path.** Your report: single-row RPSM score is allocation-sensitive and 22× worse per row than n=32. Two directions: finish the scratch-reuse audit for n=1 (the `psi_matrices.cpp:61` pattern, extended), and **server-side dynamic batching** (§6.2) so n=1 rarely happens under load.
- **3.3 — Softmax/exp**: same vector-exp treatment as §2.4; `softmax_rows` is embarrassingly SIMD.
- **3.4 — GH gate / Bessel.** LUT already replaces `kv` — good. Remaining questions: linear interp on a 16 384-point uniform grid — check whether the grid resolution lets you drop to a smaller cache-resident table or a piecewise rational approx (fewer memory touches); profile `nig_r_eff_scalar` frequency on the batch path.
- **3.5 — RFF encoder.** `cos` over D=256 per sample: vectorized `sincos` (SLEEF), or lean on the SORF/FWHT path you shipped in Phase 5 — `fwht_core` is clean radix-2 and auto-vectorizes well with §1.1, but benchmark it against the dense projection at your actual d; FWHT only wins above a size threshold.
- **3.6 — Similarity index / kernel memory.** Audit `similarity_index.cpp` + `kernel_memory.cpp` (736 lines) for linear-scan retrieval; if retrieval grows with corpus size, the research area is ANN structures (HNSW) vs your exact scan, with your OOD/anomaly semantics as the constraint.

---

## 4. Concurrency & the global lock

- **4.1 — `g_mu` serializes the entire accel API.** Every call to `score_matrix`, `batch_encode`, `softmax_rows` takes the same global `std::lock_guard` (`accel_backend.cpp`), even on the pure-CPU path. Under a multi-threaded REST server this flatly serializes all inference. Fix pattern: `std::call_once`/atomic-flag init, then lock-free reads of the immutable backend selection; confine the mutex to the CUDA context if the CUDA runtime needs it (streams per thread are the better answer there). This is a correctness-of-scaling issue, not a micro-opt — profile REST throughput vs client concurrency before/after.
- **4.2 — Thread pool, not thread spawn.** The non-OpenMP fallback in `parallel_rows.hpp` spawns `std::thread`s per call. Replace with a persistent pool (or just require OpenMP). Also retune `kScoreParallelMinWork = 1e6` per §0.5's scaling measurement — it was tuned on MinGW OpenMP once; it changes with fp32 and SIMD.
- **4.3 — Nested parallelism interaction**: REST worker threads × OpenMP teams oversubscribes. Research a policy: OMP threads = physical cores / active requests, or single-level parallelism with request batching.
- **4.4 — False sharing audit** on `CyphaLMBatch`'s shared-head workers and on any per-thread accumulators; `alignas(64)` the scratch structs.

---

## 5. GPU (largest absolute ceiling)

Current CUDA: fp64 kernels, one-thread-per-(i,k) score with an inner d-loop, infer-only, blocked on Windows by the missing MSVC host compiler.

- **5.1 — fp64 → fp32 on device.** Consumer NVIDIA silicon runs fp64 at 1/32–1/64 of fp32 rate. Your kernels are all fp64. This is potentially a **30–60× throughput change on the same GPU** and is the first thing to do before judging any GPU results you've collected.
- **5.2 — Unblock the toolchain**: either MSVC Build Tools on the Windows box (§0.1.3) or do all CUDA work under WSL2 (nvcc + gcc is supported there; your `wsl_bench_gpu.sh` suggests the environment half-exists).
- **5.3 — Kernel quality**: `k_score_matrix` re-reads `H[i·d+j]` for every k and `D` uncoalesced across the k-strided threads. With §3.1's fold, replace it with **cuBLAS `sgemm`** entirely. `k_softmax_rows` is one-thread-per-row (no warp reduction) — fine at small K, rewrite with warp shuffles if K grows.
- **5.4 — Transfer amortization**: check whether model constants (`mu0`, `inv_v`, `D'`, Bessel table) live persistently on device vs re-uploaded per call; pinned host memory for H batches; CUDA graphs for the generate token loop.
- **5.5 — LSTM training on GPU** (your stated gap): at h=128 sequential, per-step GPU launches lose to CPU. The research menu: (a) batch B sequences so each step is a `(B×2h)·(2h×4h)` GEMM — GPU wins from B≈32–64 up; (b) cuDNN RNN API as a baseline oracle even if you keep custom gates; (c) persistent-kernel RNNs (weights resident in registers/SMEM — h=128 fits, this is the classic Baidu persistent-RNN result and is exactly your size class). Honest framing: GPU training only pays if you adopt §2.7 batching; single-stream online learning stays CPU.

---

## 6. Serving path (`cypha_rest`)

- **6.1 — cpp-httplib is blocking thread-per-connection** and you build with optional OpenSSL. Profile end-to-end `/predict` latency budget: JSON parse → encode → score → JSON serialize. If JSON is >10% (likely at your µs-scale kernels), research simdjson for parse and a non-allocating writer for response.
- **6.2 — Dynamic batching.** Your own data: 0.16 vs 3.60 µs/row. A micro-batching queue (collect requests for ≤200 µs or until n=32) converts the worst path into the best path under load. This is the highest-leverage *serving* change and is a well-trodden design (Triton-style).
- **6.3 — SQLite experiment DB**: opt-in, but when enabled verify WAL mode + prepared statements + transaction batching so it never sits on a request thread.

---

## 7. Memory & data layout (cross-cutting)

- fp32 conversion (§2.1) halves every working set; recompute cache residency after: at h=128 fp32, `Wx+Wh+Wy(vocab=256)` ≈ 0.9 MB — fits L2, which changes which optimizations matter (compute-bound, favouring SIMD/FMA over blocking).
- Continue the allocation audit beyond the 2026-07-12 fix: `infer_cpu.cpp` (46 sites), `generation.cpp`, REST handlers. Pattern: request-scoped monotonic arena reset per call.
- Alignment: ensure weight matrices are 64-byte aligned (`std::assume_aligned` / aligned alloc) so the SIMD paths don't pay unaligned penalties.
- The 49 KLoC `bessel_table_data.cpp` compiles a 16 384-entry static table — fine at runtime, but consider emitting it as a binary blob to cut compile time of the core lib.

---

## 8. System-level & Windows-specific research

- MinGW libm vs UCRT vs vendor math (§1.5) — likely the cheapest 2× nobody has measured yet on this codebase.
- Timer resolution / power plan / core parking hygiene for all benchmarking (§0.3).
- libstdc++ (MinGW) vs MSVC STL codegen on the hot TUs once MSVC exists.
- Huge pages: probably irrelevant at your working-set sizes; deprioritize.

---

## 9. Prioritized execution order (expected compounding)

| # | Work | Cost | Expected effect | Depends on |
|---|------|------|-----------------|-----------|
| 1 | §0 profiler + noise control + perf lock | 1–2 d | Truthful numbers | — |
| 2 | §1.1–1.4 `-O3 -march` + LTO + fp-contract + PGO | 1–2 d | 1.5–3× hot kernels | 1 |
| 3 | §2.1 fp32 typedef + golden re-baseline | 2–4 d | 2–4× matvec paths | 1 |
| 4 | §2.2–2.4 fused/GEMM matvec + hoisted vectorized gates | 1–2 wk | 2–5× on the 77% | 2,3 |
| 5 | §4.1–4.2 kill global lock, thread pool | 2 d | Linear REST scaling | 1 |
| 6 | §3.1 score_matrix → GEMM + §6.2 dynamic batching | 3–5 d | ~20× worst-case serve path | 3 |
| 7 | §5.1–5.4 fp32 CUDA + cuBLAS + persistent buffers | 1–2 wk | 10–50× bulk paths | toolchain |
| 8 | §2.7 batched/parallel training research | open-ended | Unlocks GPU training (§5.5) | 4 |
| 9 | §2.6, §3.4–3.6, §7 residual audits | ongoing | 10–30% aggregate | 4,6 |

Items 2+3+4 multiply on the same 95% of train_step: a conservative compound estimate is **10–20× single-thread training throughput** (≈130 → 1 500–2 500+ chars/sec) with zero model-quality change, which transforms every sweep and overnight in your bench tree. Item 6 changes the product story for the REST surface; item 7 changes what benchmarks are even feasible.

---

## 10. Open research questions worth writing up (paper-adjacent)

1. **fp32 sensitivity of the DIF natural-gradient rule** — does Cramér–Rao-efficient updating degrade measurably at reduced precision? (Ties to your information-geometry framing; a precision-ablation section strengthens the paper.)
2. **Federated-aggregate as intra-node parallel SGD** — convergence of periodic-merge local training for the θ₀⊕Δk decomposition specifically; the shared world prior may merge more gracefully than generic weights.
3. **Persistent-RNN kernels for custom gate algebras** (eml/axiom) — nobody publishes GPU kernels for non-standard LSTM gates; h=128 register-resident weights make this tractable and it's a differentiator for the Cypha story.
4. **Criticality-monitor-guided compute allocation** (your Phase 9) — use the runtime criticality signal to gate expensive paths (GMM, BMA, IB) adaptively: spend FLOPs only when the monitor says the model is near a boundary. Performance and theory in one.

