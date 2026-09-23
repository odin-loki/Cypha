# CyphaLM lossy context mixer: profile, tiers, serve fixes

**Date:** 2026-09-23
**Branch:** `claude/llm-profiling-optimization-y5cdrr`
**Machine:** Linux KVM, 4 vCPU Xeon, 15 GiB, g++ 13.3, `-O3 -msse4.1`, THP `madvise`
**Corpus:** enwik8 first 8 MiB (`enwik8.8mb`, SHA256 `09f6dd72…ee8e`), screens on its first 1 MiB
**Metric:** compress-faithful observe bpc (`eval_bpc` = hp encode math), mem 22

Continues [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md). That plan
found the only RAM lever it had, a smaller global `table_bits` (mem20), costs
+0.0063 bpc on enwik8. This round profiles where gate24 actually spends time
and RAM, adds per-component lossy knobs, and screens them. It also fixes two
serve-path problems that turned up on the way.

## TL;DR

| tier (`CYPHA_HP_LOSSY_TIER`) | enwik8 8 MiB bpc | Δ vs gate24 | peak RSS | observe time |
|---|---:|---:|---:|---:|
| `gate24` (default, exact) | 1.611729 | — | 1,538 MB | 1.00× |
| **`lean`** | **1.609866** | **−0.0019** | **1,078 MB (−30%)** | **1.35× faster** |
| `balanced` | 1.612457 | +0.0007 | 814 MB (−47%) | 1.38× |
| `compact` | 1.617400 | +0.0057 | 670 MB (−56%) | 1.37× |
| `small` | 1.629798 | +0.0181 | 404 MB (−74%) | 1.37× |
| `tiny` | 1.652317 | +0.0406 | 253 MB (−84%) | 1.39× |
| old `mem20` (plan, for reference) | 1.618017 | +0.0063 | 1,913 MB* | — |

\* The plan's mem20 RSS was measured on the old eager-zero tables with a
second predictor, so it isn't directly comparable. Its bpc is.

Observe time is solo on the first 2 MiB (one process on an idle box), relative
to gate24 on the same binary. gate24 on this branch is itself 1.16× faster
than the v2.5.0 vendored tree (huge-page tables), so `lean` is ~1.55× faster
than v2.5.0. The tiers ran on the build that still had prefetch; it made no
measurable difference (last two gate24 rows), so it was removed afterwards:

| build (2 MiB, mem 22) | seconds | bytes/s | bpc |
|---|---:|---:|---:|
| v2.5.0 vendored hp (eager `std::vector` tables) | 274.9 (+1.1 construct) | 7,629 | 1.684211 |
| this branch, gate24, with next-slot prefetch (the build the tiers ran on) | 239.6 (+0.06 construct) | 8,752 | 1.684211 |
| this branch, gate24, no `MADV_HUGEPAGE` | 312.9 | 6,703 | 1.684211 |
| this branch, gate24, prefetch removed (shipped) | 237.0 | 8,849 | 1.684211 |
| `lean` | 177.2 | 11,837 | 1.681291 |
| `balanced` | 173.3 | 12,103 | 1.681735 |
| `compact` | 174.4 | 12,028 | 1.682953 |
| `small` | 175.4 | 11,955 | 1.686952 |
| `tiny` | 172.8 | 12,138 | 1.695866 |

Tier speed comes from doing 20% fewer context lookups per bit (8 models and
4 pool slots fewer). Smaller tables barely help beyond that once huge pages
are on. The lever for speed is fewer experts, and the lever for RAM is
smaller tables.

- **`lean` beats gate24 on every axis at once.** It drops 8 wiki context
  models, shrinks the discovery pool to 8 slots and caps the pool and
  byte-match tables. On enwik8 that is lower bpc, 30% less RAM and faster.
- **`balanced`** is near-lossless (+0.0007) at 53% of gate24's RAM.
  **`compact`** is under half the RAM for +0.0057, about what mem20 cost,
  at a third of mem20's measured RSS. `small` / `tiny` are for tight memory.
- **The mixer weights themselves are not a lever.** All ten layer-1 weight
  sets pay (+0.0035 to +0.018 bpc each to drop). A higher update-skip
  threshold only loses (+0.0023 at 64). The mixer is ~300 KB, and time goes
  into the context lookups that feed it. So the lossy "mixer" levers here
  are its inputs: which experts it gets, and how big their tables are.
- **Serve path fixes (lossless):** speculative bit-tree scoring was leaking
  into the live model (up to 0.18 nats on later predictions). It is now
  exact. Full-vocab `predict_next` at mem 22 goes from 208 ms to 8 ms (26×),
  and the predictor constructs in ~60 ms instead of ~1–8 s. See [Serve path](#serve-path-correctness-and-speed).

## Where the time and RAM go (gate24, mem 22)

`perf` on 512 KiB of observe (the train / BPC path):

| symbol | share |
|---|---:|
| `ContextModel::predict` (35 hashed models + 12 pool slots, ~2 cache misses each per bit) | 62% |
| `Predictor::update` (mixer + APM + counters) | 13% |
| `Predictor::predict` (mixer dots, gates, APM) | 10% |
| kernel `clear_page` (eager zeroing of the tables) | 9% |
| everything else | 6% |

Mixer inputs: 115 experts = 35 context models × 2 outputs + 9 byte-match +
4 word-match + sparse UTF-8 + DMC + LZP + 3 skip matches + Hebbian + bias +
24 discovery-pool outputs. There are 10 layer-1 weight sets and 1 final set.

RAM at mem 22 (1,528 MB):

| component | MB | note |
|---|---:|---|
| 35 context models | 864 | 12 of them at 2^24 slots (48 MB each), the rest 2^22 (12 MB) |
| 13 byte-match hash tables | ~416 | 2^23 × 4 B each |
| discovery pool, 12 context models | 144 | |
| byte ring, Hebbian, LZP, DMC, word match, APMs | ~100 | |
| mixer weights | <1 | |

## Knobs added

All are runtime fields on `hp::Config`, mapped from `CyphaLMConfig`, and all
default to exact gate24. They change predictions, so they are saved with the
model (`.json`) and must match its `.hpbin`.

| `CyphaLMConfig` | `hp::Config` | effect |
|---|---|---|
| `hp_cm_drop` | `cm_drop` | bit *i* drops context model *i* (`hp::Predictor::CmId`): no table, zero mixer input, no lookup |
| `hp_cm_bits_cap` | `cm_bits_cap` | cap every context-model table at 2^N slots |
| `hp_gate_drop` | `gate_drop` | bit *j* drops mixer weight set *j* |
| `hp_mixer_skip` | `mixer_skip` | skip mixer update when \|err\| < N (gate24: 32) |
| `hp_match_bits_cap` | `match_bits_cap` | cap the 13 byte-match hash tables |
| `hp_pool_slots` | `pool_slots` | keep N of the 12 discovery-pool slots |
| `hp_pool_bits_cap` | `pool_bits_cap` | cap discovery-pool tables |

`apply_hp_lossy_tier(cfg, name)` / `CYPHA_HP_LOSSY_TIER=name` sets a
measured bundle. `cyphalm_lossy_bench --tiers gate24,lean,…` measures them
(its numbers match the standalone screen harness exactly).

## Screens (1 MiB, mem 22, gate24 = 1.711967)

Three runs at a time on 4 cores, so the seconds column is comparable only
within this table. Full data:
[`CYPHALM_LOSSY_MIXER_SCREEN_1M.tsv`](CYPHALM_LOSSY_MIXER_SCREEN_1M.tsv).

### Drop one context model

| dropped | Δ bpc | | dropped | Δ bpc |
|---|---:|---|---|---:|
| paramod | −0.00060 | | sp24 | +0.00054 |
| nestmod | −0.00054 | | sentmem_cm | +0.00063 |
| infokeymod | −0.00043 | | sp13 | +0.00066 |
| linkpipemod | −0.00031 | | sengrp | +0.00069 |
| tplmod | −0.00030 | | tag | +0.00072 |
| o6b | −0.00022 | | wikistackmod | +0.00076 |
| headingmod | −0.00017 | | wstr_sp | +0.00078 |
| capmaskmod | −0.00006 | | o1 | +0.00121 |
| linemod | +0.00001 | | sen | +0.00128 |
| catmod | +0.00003 | | col | +0.00129 |
| statemod | +0.00005 | | o6 | +0.00165 |
| titlemod | +0.00007 | | sentst | +0.00278 |
| wordlenmod | +0.00016 | | wbi | +0.00279 |
| num | +0.00024 | | o3 | +0.00488 |
| link | +0.00040 | | brk | +0.00567 |
| uppergapmod | +0.00041 | | o2 | +0.00630 |
| sectitlemod | +0.00054 | | o4 | +0.00908 |
| | | | word | +0.02487 |

Stacked: the 8 with Δ < 0 (**dropA**) −0.0023; plus the 4 neutral ones
+0.0006; plus 4 more +0.0033. Only dropA carries forward.

### Tables, pool, mixer

| variant | Δ bpc | RSS MB |
|---|---:|---:|
| context tables ≤ 2^23 | +0.00012 | 1,240 |
| context tables ≤ 2^22 | +0.00039 | 1,084 |
| context tables ≤ 2^21 | +0.00178 | 874 |
| context tables ≤ 2^20 | +0.00530 | 769 |
| match tables ≤ 2^22 | +0.00002 | 1,320 |
| match tables ≤ 2^21 | +0.00005 | 1,216 |
| match tables ≤ 2^20 | +0.00011 | 1,164 |
| pool tables ≤ 2^20 | −0.00003 | 1,420 |
| pool tables ≤ 2^18 | +0.00001 | 1,393 |
| pool 8 slots | −0.00171 | 1,480 |
| pool 4 slots | −0.00151 | 1,432 |
| mixer skip 64 / 128 / 256 | +0.0023 / +0.0068 / +0.0149 | 1,528 |
| drop weight set 0 (c0) | +0.0177 | |
| drop weight set 1 (GRIA α) | +0.0035 | |
| drop weight set 2 (prev byte) | +0.0142 | |
| drop weight set 3 (match len) | +0.0058 | |
| drop weight set 4 (entropy) | +0.0036 | |
| drop weight set 5 (Hebbian) | +0.0036 | |
| drop weight set 6 (wiki state) | +0.0047 | |
| drop weight set 7 (pattern) | +0.0041 | |
| drop weight set 8 (argmax) | +0.0071 | |
| drop weight set 9 (word pos) | +0.0096 | |

## 8 MiB confirmation (full `enwik8.8mb`, mem 22)

Three to five runs at a time on 4 cores, so wall time isn't reported here.
The solo timings are in the TL;DR.

| run | knobs | bpc | Δ vs gate24 | peak RSS |
|---|---|---:|---:|---:|
| base (gate24) | — | **1.611729** | — | 1,538 MB |
| dropA | drop 8 CMs | 1.610643 | −0.0011 | 1,406 MB |
| pool8 | pool 8 slots | 1.610558 | −0.0012 | 1,490 MB |
| **lean** | dropA + pool 8 slots, pool ≤2^20, match ≤2^22 | **1.609866** | **−0.0019** | **1,078 MB** |
| balanced | lean + CM ≤2^23 | 1.612457 | +0.0007 | 814 MB |
| compact | lean + CM ≤2^22 | 1.617400 | +0.0057 | 670 MB |
| small | lean + CM ≤2^21, match ≤2^21 | 1.629798 | +0.0181 | 404 MB |
| tiny | lean + CM ≤2^20, match ≤2^20, pool ≤2^18 | 1.652317 | +0.0406 | 253 MB |

dropA and pool8 each hold up on the full file, and they stack
sub-additively. Past `balanced`, each halving of the context tables costs
about 3× more than the one before.

`base` reproduces the published gate24 figure (1.611729) exactly. So the
gate24 strip, the demand-zero tables and the prefetch leave predictions
untouched on the full file.

## Serve path: correctness and speed

`serve_predict_next` scores all 256 next bytes by walking the MSB bit tree on
the live predictor and undoing each speculative update. That only works if
the undo log records every cell `predict()`/`update()` writes. Two did not:

- `DmcModel::update` recorded nothing: counts, node splits (`push_back`), `cur_`.
- `WordMatchModel::predict` reset `len_` on a mismatching bit, unrecorded.

**Effect on main:** after one `serve_predict_next`, the live model's later
log-probs moved by up to 0.18 nats compared with an identical model that never
served. The served distribution itself was off from fresh-clone scoring by up
to 0.27 nats on text. `hp_bit_tree_smoke` did not catch it: 128 random warmup
bytes never trigger a DMC split or a word match. Found by diffing serialized
component state around one speculative step, then around a full DFS.

**Fixes:** record the DMC cells and its node count (`UndoFrame::note_size`),
with DMC capacity reserved so recorded cells never move. Record the
word-match reset. Reserve the undo frame stack so live scopes never point at
relocated frames. Drop the O(n²) duplicate scan in `note_bytes`: restoring
newest-first already leaves the oldest value.

**DFS:** one `predict()` per node, a re-predict only before the bit-1 update
(`update()` reads `predict()`'s mixer scratch, which the bit-0 subtree
overwrites), and no update at leaves. That's 382 predicts + 254 updates per
call instead of 510 + 510.

| check (mem 16, text) | before | after |
|---|---:|---:|
| max \|Δ log p\| bit-tree vs fresh clone | 0.27 | 0 |
| live model drift after serving (50 bytes) | 0.18 | 0 |
| `next_byte_log_probs(256)`, old vs new DFS on the fixed undo log | 17.8 ms | 9.6 ms |

Serve latency at mem 22, `hp_inference_bench --iters 5` (64 KiB warmup), solo on
this machine, `main` build vs this branch:

| | `main` | this branch | speedup |
|---|---:|---:|---:|
| `predict_next` (full 256-way distribution) | 207.6 / 211.4 ms | 8.1 / 8.7 ms | 25× |
| `next_byte_log_probs(256)` | 207.2 ms | 7.3 ms | 28× |
| `serve_greedy_next` | 4.37 ms | 0.26 ms | 17× |
| RSS after model construct | 1,564 MB | 81 MB | |

Most of the gain is the undo-log duplicate scan: each speculative update
records ~1,500 cells, and the scan was quadratic in that. The DFS change
supplies the rest. (`main`'s number is what shipped. Given the ODR split
below, its tools may also have been running a mixed predictor.)

**Tables:** the vendored `HashTable` had become a `std::vector` (zero-filled
up front). It's back to demand-zero pages (`mmap`; `calloc` on Windows) with
`MADV_HUGEPAGE`. Predictions are identical; construct drops from ~8 s to
~55 ms at mem 22, and RSS starts at ~80 MB and grows with use.

**Prefetch (tried, dropped):** prefetching every model's next slot as soon as
the coded bit is known looked ~25% faster in a noisy early test. Solo, with
huge-page tables, it measured 237.0 s against 239.6 s without, which is noise.
Huge pages already remove the TLB misses the prefetch was hiding. It was
removed again; upstream hp's "no prefetch" note stands.

## Tests

- `hp_bit_tree_smoke`: adds a text warmup, a served-vs-twin leak check (must
  be 0) and parity against fresh-copy scoring. Tables capped at 2^16 via the
  knobs so its 512 predictor copies stay fast.
- `hp_checkpoint_roundtrip_smoke`: lossy knobs survive save/load; the loaded
  live predictor reproduces the trained one (with a fresh-model control).
- Full CTest on this branch versus unmodified `main`, both built here: the same
  38 tests fail on both (intelligence-profile / d39–d76 `cypha_slow` smokes that
  expect LSTM statistics the hp backend doesn't produce, `forecast_smoke`
  timeout, canary, predictive codec, CUDA/TLS without hardware). Every hp /
  CyphaLM test passes.

### Latent build bug fixed by the strip

On `main`, `HpFlags.cmake` applied the v78 flags `PRIVATE` to `cypha_core`.
Every tool or test that included the hp headers therefore compiled a
different `hp::Predictor`: 86 KB and 64 experts instead of 172 KB and 115.
Because those methods are inline, the linker could mix both definitions in
one binary. That's why `hp_bit_tree_smoke` took 2 s on `main`: it wasn't
really exercising gate24. With the flags resolved into the source (see
[`CYPHALM_HP_GATE24_STRIP.md`](CYPHALM_HP_GATE24_STRIP.md)) there is one
definition everywhere.

## Reproduce

```bash
cmake -S native -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
# tiers through Cypha (one process per tier gives clean RSS numbers)
build/cyphalm_lossy_bench --corpus enwik8.8mb --warmup-n 0 --eval-n 8388608 \
    --latency-iters 0 --tiers lean
CYPHA_HP_LOSSY_TIER=compact build/cyphalm_generate --prompt "The " --max-bytes 32
```

## Not done / next

- **Default tier.** `lean` is better than gate24 on enwik8 but stays opt-in:
  gate24 is the published reference and the upstream lab's leftover models
  were accepted on it. Making `lean` the production default is a one-line
  change in `apply_hp_production_recipe` once confirmed on a second corpus.
- Only enwik8 was measured. The dropped models are wiki-markup models, so
  expect `lean` to be neutral-to-better on plain text, but that is untested.
- Windows uses `calloc` for tables (may zero eagerly), and MSVC gets
  no huge pages. Not compiled here: no MSVC/mingw on this machine.
