# O1: a faster build with the same outputs (GENERATION_PLAN §13.41)

2026-11-12 · Odin Loch

**Odin:** "Keep looking. Come up with more. Try optimisation techniques … Go crazy on it."

The job (§13.41, O1): profile-guided optimisation, link-time optimisation and `-march=native` (AVX-512) with `-ffp-contract=off`, as a CMake option, off by default. Exact. Predictions: `context8k` time down at least 10%; K5 sampling down at least 10%. Branch `wip/opt-build`. Only `native/CMakeLists.txt` and the new `scripts/pgo_build.sh` change.

The work began on 2026-11-05 and stopped twice on Odin's word (§13.44, §13.47) before its exactness pair finished. It resumed on 2026-11-12. The branch first merged main 9cbd277 (clean).

## What changed

- `CYPHA_FAST_BUILD` (off by default) builds `cyphalm_trie` and `scribe` with `-march=native -ffp-contract=off` and link-time optimisation (GCC `-flto=4`; Clang ThinLTO with lld, 4 jobs).
- `CYPHA_PGO=generate|use` adds profile-guided optimisation (GCC or Clang).
- Nothing that reorders floating point: no `-ffast-math` or any part of it, no `-mrecip`. `-ffp-contract=off` stops fused multiply-adds.
- `scripts/pgo_build.sh` builds the instrumented binaries, trains them on runs that are not the checked ones, and builds again with the profile:
  - `context8k --train 524288 --sample 655360` (the small check is 1048576 / 1179648);
  - `sample --k5 fuzzy/train.txt` on 12 prompts (`sample`'s own, not the GEN-A dev or test files);
  - `scribe ladder` (v2) on that sample.
- With `-march=native` the binaries run only on CPUs like the build machine's.
- When the option is off, nothing changes.

## Checks (registered 2026-11-12, before any run)

Old: this branch with the option off, CMake Release, GCC 13 (`-O3`, no `-march`): the same build as main's. New: `scripts/pgo_build.sh --cc gcc` and `--cc clang` (Clang 18), each with LTO and `-march=native`.

**Choosing the compiler.** The small default `context8k` (`--train 1048576 --sample 1179648`) runs old, GCC-PGO and Clang-PGO together, each pinned to its own 4 logical CPUs. Of those whose stdout is `cmp` equal to old's, the one with the shorter wall time is O1's build. All further checks use it.

Then, old against that build, each pair started together, each side pinned to its own 4 logical CPUs, default threads:
1. `context8k` small: `--no-meta-mix` and `--no-full-mix` (default is the screen above). `cmp` of the whole stdout.
2. GEN-A dev, 250 prompts, `--k5 /mnt/d/cypha_llm/fuzzy/train.txt`, `K5_THREADS=4`: the planner (the default) and `--no-k5-plan`. `cmp` of the GEN file, the `sample` stdout, both SCRIBE reports (v2 and v3.1) and the summary; the `sample` stderr with its timing lines removed. Old and new use their own `scribe`.
3. Both selftests; FMA instructions in each binary (`objdump`).
4. One full default `context8k` with the new build, inside `flock ~/.cypha_full.lock`: `context8k_bpc` 1.4962 must reproduce, and its stdout is `cmp`'d against O5's full run of main's code (`/mnt/d/cypha_llm/opt/o5/F_new.out`). If that differs, old's full run is made too and they are compared.

If any check differs, O1 is reported failed. Timings come only from the pairs started together.

Measures: wall time, user time and peak RAM (`/usr/bin/time -v`); K5's sampling time (stderr's `k5: generated N bytes in X s`).

## Results

(to come)
