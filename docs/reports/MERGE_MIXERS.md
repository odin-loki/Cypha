# Merging the mixers: one mixer library

2026-11-05 · Odin Loch

**Odin:** "see if there are any algebraic, factorial or combinatorical reductions of these algorithms and see if we cant merge some of these algorithms together and lighten the load".

Branch `wip/merge-mixers`, from main 7265b3e. Plan: [GENERATION_PLAN §13.40](../GENERATION_PLAN.md), job M2. Odin's PC (WSL). Other sessions (M1, M3, M4) ran on the PC at the same time.

## Why

PAQ's logistic mixing is geometric mixing over a binary alphabet: a product of experts with a convex log loss (Mattern 2012–13). CyphaLM has four mixers that are all forms of it, each with its own copy of the maths:
- the content mixer (`online/content_mixer.hpp`): the chain's stages and the online final layer, by exponentiated gradient;
- the full mix (`online/full_mixer.hpp`): a replica of the chain and final layer, PAQ's bitwise logistic mixer, a byte-level geometric mixer, APMs;
- the meta mix (`online/meta_mixer.hpp`): a wider logistic mixer with float weights, fixed-share masters;
- K5's structure ranker (`lm/structure/mixer.hpp`): a softmax over candidates (geometric mixing), signed exponentiated gradient, a second layer by exponentiated gradient.

## What was merged

New: `native/include/cypha/online/mix_core.hpp`. The four mixers now call it.

| Part of mix_core | What it is | Replaces |
|---|---|---|
| `reweight`, `eg_floor`, `eg_step` | the multiplicative step w_j ← w_j f(j), then back to the simplex with a floor | the step in `eg_stage`, `OnlineFinal::update`, the full mix's `EgState` and final layer, the structure mixer's second layer, `FixedShare::update` |
| `EgSets` | weight sets on the simplex, chosen by a context, starting at the last input | `OnlineFinal`'s weights and the full mix's copy of them (`fw_`) |
| `EgLambdas` | the chain's (λ, 1 − λ) pair per bucket | `eg_stage`'s two vectors and the full mix's `EgState` |
| `eg_pm` | exponentiated gradient with signed weights (EG±) | `structure::Ranker::update` |
| `FixedShare` | exponential weights with fixed share | the meta mix's `FixedShare` |
| `softmax`, `expect` | geometric mixing's normaliser and its log-loss gradient E_P[f] − f(target) | `GeoMix::dist` and `Ranker::dist`; `GeoMix::update` and `Ranker::update` |
| `GeoMixer` | the byte-level geometric mixer | the full mix's `GeoMix` (also the meta mix's M2G) |
| `LogisticMixer<W>` | PAQ8's two-layer gated logistic mixer; W = double or float; layer 2 chosen by its own context | the full mix's `BitMixer` (double, layer 2 by bit position) and the meta mix's `Mixer2` (float, layer 2 by partial byte or bit position) |
| `Apm`, `path_bits`, `squash`, `stretch_eps`, `lerp`, `byte_class`, `parse_kv` | small shared parts | `byte_class` was in two places; both settings parsers had the same loop |

Kept in their own files: the full mix's `Sse` (it reads `ByteIn`), the arms' wiring in `FullMixer::step` and `MetaMixer::bit`, the meta mix's selectors, the dump and replay, `fit_lambdas`, `fit_doc`, `mix_doc_into`, and the structure model's file format.

**Where the mixers differ, each keeps its own order.** This is what makes the merge exact:
- The structure mixer's second layer caps the exponent after the step (min(30, η q/p)); the others cap the ratio (η min(q/p, 30)). Each passes its own exponent to `reweight`.
- The structure mixer's second layer does not renormalise after its floor; the others do (`eg_floor`'s `renorm`).
- The ranker's gradient starts from −z_target and adds the expectation; the geometric mixer adds the expectation and then subtracts. `expect` takes the starting value.
- `LogisticMixer` keeps step counters only when the decay is not 0. With decay 0 the old rate was lr / (1 + 0 · t) = lr exactly, so the counters changed nothing.

Nothing in the four mixers had to be left unmerged for exactness. Three other mixers were not in M2's list and were not touched: the online n-gram's own logistic mixer (`online_ngram.hpp`), the GLN (`bitwise_gln.hpp`) and `adaptive_predictor_mixer.hpp`.

## Exactness checks

Old: main 7265b3e (`~/build_lead`). New: this branch. Same compiler and flags (GCC 13.3, Release, -O3, no -march=native). Each pair was started together.

| Check | What was compared | Result |
|---|---|---|
| Old-header vs new-header harness | every mixer on the same random inputs, every number printed in hex (365 lines): the content mixer's frozen and online chain in six modes (two rates, step multipliers, copy's clip, online lambdas out), `OnlineFinal`; the full mix in four set-ups (n-gram on and off, two rates on and off, LSTM on and off, the ig gate, lr decay, learned SSE) with the meta mix on it in three settings (default; l2 0, m1r 0, m2m 1, m3lctx 1, six selectors; three scales); `BitMixer` with and without decay, `Sse`, `path_bits`; the structure ranker with one set, with 12 sets and a mask, and with nine rankers and a second layer (learn, loss, dist, write, read) | `cmp` equal |
| Selftests | `cyphalm_trie`, `scribe`, `k5_planner`, `k5_structure` (old and new) | all pass; `k5_structure selftest`'s output equal old and new |
| `context8k`, small setup, default (meta mix on) | whole stdout, 260 lines | `cmp` equal |
| `context8k --no-meta-mix` | whole stdout, 245 lines | `cmp` equal |
| `context8k --no-full-mix` | whole stdout, 233 lines | `cmp` equal |
| Planner, dev GEN (`gen_eval.sh dev … --prompts 250 --k5 fuzzy/train.txt`, the default planner) | GEN file, `sample` stdout, SCRIBE v2 and v3.1 tables, summary | `cmp` equal (all five) |
| `--k5-struct`, dev GEN (the same with `--k5-struct ~/k5s_data/runs/S210.Rnok5.chosen.model`, K5_STRUCTURE.md's W5) | GEN file, `sample` stdout, SCRIBE v2 and v3.1 tables, summary | `cmp` equal (all five) |
| `full-mix-replay`, `meta-mix-replay` over the small setup's dump | whole stdout, three rounds each | `cmp` equal |

The small setup is `cyphalm_trie context8k /mnt/d/cypha_llm/enwik8 /mnt/d/cypha_llm/corpus/labelled/dolly/databricks-dolly-15k.jsonl --train 1048576 --sample 1179648`. Its headline: `context8k_bpc 1.7501 (meta_mix M1Gs)`, 1.7633 with `--no-meta-mix`, 1.8384 with `--no-full-mix` (old and new alike). The W5 GEN file gives SCRIBE v2 0.453 [0.431, 0.474], as in K5_STRUCTURE.md; the planner's gives 0.532 [0.502, 0.564].

**The merge is exact.**

## Lines of code, time and RAM

**Lines.** "Code" counts lines that are not blank and not only a `//` comment.

| File | Lines before | Lines after | Code before | Code after |
|---|---:|---:|---:|---:|
| `online/content_mixer.hpp` | 360 | 332 | 287 | 261 |
| `online/full_mixer.hpp` | 861 | 649 | 737 | 547 |
| `online/meta_mixer.hpp` | 497 | 412 | 444 | 366 |
| `lm/structure/mixer.hpp` | 182 | 177 | 159 | 151 |
| **the four mixers** | **1,900** | **1,570** | **1,627** | **1,325 (−18.6%)** |
| `online/mix_core.hpp` (new) | 0 | 339 | 0 | 257 |
| **all mixing code** | **1,900** | **1,909 (+0.5%)** | **1,627** | **1,582 (−2.8%)** |

The four files lost 302 lines of code to the library, but the library holds 257. Only about 45 lines were true duplicates: the two logistic mixers, the two copies of the final layer and of the chain's stage, the two softmaxes, `byte_class`, the settings loop. Most of the four files is not shared maths: the arms' wiring, the meta mix's selectors, the full mix's inputs, the dump and replay, the document stage's fit and read, the structure model's file format.

**Time and RAM.** `/usr/bin/time -v`; old and new started together, the same flags and threads, one run each. Other sessions loaded the PC (load 8–25), so a change of 1–2% in one run is within noise.

| Run | Wall, old | Wall, new | Change | User, old | User, new | Peak RSS, old | Peak RSS, new |
|---|---:|---:|---:|---:|---:|---:|---:|
| `context8k`, default | 600.6 s | 604.2 s | +0.6% | 745.6 s | 746.9 s | 1,303 MB | 1,302 MB |
| `context8k --no-meta-mix` | 346.7 s | 351.4 s | +1.4% | 472.8 s | 476.7 s | 905 MB | 906 MB |
| `context8k --no-full-mix` | 500.2 s | 509.2 s | +1.8% | 587.4 s | 597.5 s | 671 MB | 670 MB |
| planner GEN (`gen_eval.sh`, 8 threads) | 221.9 s | 221.8 s | 0.0% | 1,077.5 s | 995.1 s | 760 MB | 761 MB |
| `--k5-struct` GEN (`gen_eval.sh`, 8 threads) | 169.5 s | 168.1 s | −0.8% | 894.7 s | 890.9 s | 1,814 MB | 1,814 MB |

- The full mix's own seconds (its stderr line): 60.6 against 61.0 (default), 39.4 against 39.4 (`--no-meta-mix`).
- Its parts (chain, final layer, bits, geometric, bit loop, learning): 1.4/2.7/1.9/16.0/27.6/7.0 s old, 1.3/2.6/2.0/16.4/27.1/7.0 s new.
- `sample`'s own time: planner 184.1 s old, 172.7 s new; `--k5-struct` 136.1 s old, 133.8 s new. The mixer is a small part of either.
- The state the mixers report is the same byte for byte: `full_mix state_bytes 178538240`, `meta_mix state_bytes 461393472`.

**The mixers alone.** Most of `context8k`'s time is not the mixers (the full mix is about 60 s of 600). So the mixers were also timed alone: `full-mix-replay` and `meta-mix-replay` over a dump of the small setup's inputs (`context8k --full-mix-dump`, 3.5 GB, copied to the WSL disk so the replay is not held up by `/mnt/d`). One thread each; old and new started together; three rounds.

| Replay | Wall, old (3 rounds) | Wall, new | Mean change | User, old (mean) | User, new (mean) | Peak RSS |
|---|---|---|---:|---:|---:|---:|
| `full-mix-replay` | 33.6, 31.8, 31.5 s | 33.5, 31.3, 30.9 s | −1.2% | 30.3 s | 29.9 s | 174 MB both |
| `meta-mix-replay` | 73.7, 71.3, 74.7 s | 72.4, 72.2, 73.2 s | −0.9% | 71.0 s | 70.3 s | 572 MB both |

Every replay's stdout was the same old and new (`cmp`), and its `full_mix_arm` and `meta_mix_arm` lines equal `context8k`'s. A first round read the dump from `/mnt/d` and was held up by it (wall 159 s and 564 s for 33 s and 114–127 s of CPU), so it is not used for time.

**What this means.** The merge is time-neutral within this PC's noise, and RAM is unchanged (±1 MB). That is expected: the arithmetic is the same, operation for operation. The one saving in the arithmetic (no division for the rate when there is no decay) is too small to see.

## The §13.40 M2 predictions, scored

Registered: "M2: exact; code down at least 25% across the four mixers; time within ±3%."

| Prediction | Result | Score |
|---|---|---|
| exact | every check `cmp` equal: the harness, three `context8k` runs, two dev GEN files (and their SCRIBE tables) | **hit** |
| code down at least 25% | the four files' code −18.6%; all mixing code, the new library included, −2.8% (physical lines +0.5%) | **miss** |
| time within ±3% | `context8k` +0.6%, +1.4%, +1.8%; GEN runs 0.0%, −0.8%; the mixers alone (replay) −1.2% and −0.9% | **hit** |

**The rule** (§13.40: kept if exact and it lowers RAM or time or code with nothing worse): it is exact, the code is a little lower (45 lines in all, 302 in the four files), and time and RAM are not worse. So it passes, but the gain is small. Its real value is that each piece of mixing maths now exists once: the next mixer (or a fix to one) is written in one place.

Why the code prediction missed: it assumed the four mixers were mostly the same maths written four times. They share the maths, but the maths is a small part of each file. The rest is each mixer's own wiring and plumbing, which no merge removes.

## Files

- Code: `native/include/cypha/online/mix_core.hpp` (new); `online/content_mixer.hpp`, `online/full_mixer.hpp`, `online/meta_mixer.hpp`, `lm/structure/mixer.hpp`. No caller changed (`cyphalm_trie.cpp`, the planner, the structure writer, `k5_structure.cpp`, `k5_planner.cpp`).
- Runs on the PC: `/mnt/d/cypha_llm/merge/m2/` (`ctx/`, `gen_old/`, `gen_new/`, `selftest/`, `replay_ext4/`, `run_all.log`, `replay.log`). The harness and the run scripts: `/mnt/d/cypha_llm/merge/m2/harness/`. The dump (`small.fmx`, 3.5 GB) is only for the replay timing; it can be deleted.
- To run the harness again: build `harness/mixers_harness.cpp` twice (`g++ -O3 -DNDEBUG -std=gnu++23 -pthread -I native/include -I native/third_party`), once in a copy of main and once in the branch, and `cmp` the two outputs.
