# Vendored hp reduced to gate24

**Date:** 2026-09-22
**Scope:** `native/third_party/hp` only. The rest of Cypha is unchanged.

CyphaLM ships one hp recipe, gate24: the v78 leftover flag set from
`v78_flags.ps1` plus `HP_SLOT_MAX=24`. The vendored tree still carried the
whole CompressionAlgorithm ablation lab: 289 `HP_*` switches in
`features.hpp`, 1,067 `#if` blocks in `predictor.hpp` alone, and the
mixers, gates and context models that were tried and rejected on the Hutter
board (LSTM mixer, low-rank `HP_MIXER_RANK`, NLMS, backprop, hedge-L1, CTW,
PPMD, SR, stemmer streams, 60+ wiki context models that never paid, and so on).
None of that was reachable from CyphaLM. This change removes it.

## Method

1. **Resolve.** Compile `features.hpp` and every header with the gate24 flags,
   then read back the effective value of all 297 `HP_*` macros, including the
   derived and header-local ones (`vals.cpp` prints each macro).
2. **unifdef.** Run `unifdef -D<name>=<value>` for every macro except
   `HP_XSIMD` (platform switch) and `HP_SLOT_MAX` (build knob).
3. **Fold literals.** Replace the flags still used inside C++ expressions,
   such as `(HP_WMATCH_4 ? 1 : 0)` and `HP_PY_EXPERT ? 2 : 1`, with their
   gate24 values. Then run `unifdef -k` to evaluate the constant `#if`s.
   A first attempt folded every `0 ? a : b` in the text and broke unrelated
   ternaries (`n >= 8 ? 0 : …`). The shipped fold only rewrites ternaries whose
   condition is an `HP_*` macro.
4. **Tidy.** Hand-collapse the folded sums (`kExtraCtx = 24`,
   `kWordMatch = 4`, `slot_bits`). Delete code that nothing reaches any more:
   `stemmer.hpp`, `reorder.hpp`, `stage_profile.hpp`, `SrModel`, and the
   `Branch3` / `shape6` / `disp_var` / `mlen2` / `agree` gates.
   `features.hpp` keeps just `HP_XSIMD` and `HP_SLOT_MAX` (default now 24).
5. **Tools.** Drop the flag-sweep tooling that only makes sense in the lab:
   `*.cu` GPU searches, `*.ps1` sweeps, `*.py` corpus scripts,
   `hp_harness.sh`, `v78_flags.ps1`, `test/ablation.sh`,
   `test/pattern_cache.sh`, `UPGRADES.md`, and the stale `simd_dot_check.cpp`
   (it called a helper that no longer existed). `dump_profile.cpp` and
   `dump_experts.cpp` went through the same resolve and fold passes.
   `cmake/HpFlags.cmake` no longer parses `v78_flags.ps1`.
   `scripts/hp_v78_flag_diff.py` is removed, and the two measure scripts build
   the CLI without a flag list.

## Checks

| check | result |
|---|---|
| Observe bpc, first 131,072 B of enwik8, mem 18, stripped vs `-D` flag build | 1.990576 vs 1.990576 |
| `hp c --mem 18` on 64 KiB of enwik8 | archives byte-identical (`cmp`), 17,775 B |
| `hp d` round trip | identical to input |
| After the unifdef pass alone (before folding) | 1.990576 |
| After the first, too-greedy fold | 1.990569, rejected and redone |
| `-Wall -Wextra` on `predictor.hpp` | clean (also fixed the old `cfg_`/`byte_ring_` `-Wreorder`) |
| Cypha build + CTest | see `CYPHALM_LOSSY_MIXER_REPORT.md` |

## Size

| file | before | after |
|---|---:|---:|
| `predictor.hpp` | 4,370 | 1,044 |
| `wiki.hpp` | 3,955 | 931 |
| `checkpoint.hpp` | 1,054 | 204 |
| `features.hpp` | 886 | 16 |
| `stemmer.hpp` / `reorder.hpp` / `stage_profile.hpp` | 646 | 0 |
| everything else | 4,795 | 4,492 |
| **hp include + main.cpp** | **15,706** | **6,687** |

The "after" column also includes the runtime lossy knobs and the undo fixes
described in `CYPHALM_LOSSY_MIXER_REPORT.md`. Together they add about 120 lines.

## What this means for callers

- gate24 is now the only code path. Passing `-DHP_<FLAG>=…` for a removed flag
  does nothing.
- The full ablation lab is still upstream in odin-loki/CompressionAlgorithm
  (`hp/`), which remains the place to try a new flag. This repo's pre-strip
  copy is at tag `v2.5.0`: `git show v2.5.0:native/third_party/hp/include/hp/predictor.hpp`.
