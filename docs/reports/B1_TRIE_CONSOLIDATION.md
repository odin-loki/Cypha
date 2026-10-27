# B1: one packed-trie library and one tool

**Odin Loch** · 2026-10-27

**Question.** Can the seven forked packed-trie tools and the three eval tools that included `trie_chat.cpp` become one library and one tool, with the same numbers?

**Answer.** Yes.
- **Parity.** Every number line the old tools print comes out of the new tool unchanged, to the last printed digit:
  - `chat`, `probe` and `context8k` stdout is byte-identical to the old stdout;
  - the two `sample` files are byte-identical;
  - the 30 number lines of `trie_fold_lm` + `trie_online_mix` all appear unchanged in `bpc`.
- **Size.** Ten files, 9,574 lines, became two files, 2,072 lines (−7,502, −78%).
- **Kept as it is.** `native/tools/trie_stream_train.cpp` was not touched.

| New | Lines | What it holds |
|---|---:|---|
| `native/include/cypha/cyphalm/packed_trie.hpp` | 1,293 | Header-only, `namespace cypha::cyphalm::trie`. Quantiser, `Node`/`Fol`/`Hit`, `Model` (observe, fill_experts, mix, mix_dist, compact_match, and the PPMD / chain / inheritance negatives), `eg_update`, `train_span`, layout measurement (`measure`, `hash_*`, `trie_stats`), `entropy_u`, the Dolly loader, `printable`, `self_test`, sampling with the no-repeat rule. Paths and constants are `Config` fields whose defaults equal the old constants |
| `native/tools/cyphalm_trie.cpp` | 779 | Subcommands `bpc`, `chat`, `probe`, `context8k`, `sample`, `selftest`. Paths are arguments (`--wiki`, `--dolly` or positional), then `$CYPHA_WIKI` / `$CYPHA_DOLLY`, then the old defaults |

PLAN rule 10 still holds: there is no string-search answering in either file. `probe` and `context8k` use `find` only to *grade* what the model wrote, as the old tools did.

## Parity, old vs new

Setup:
- **Machine.** 2 cores, shared with two other agents. Every run was under `nice -n 5`, one heavy run at a time for the new tool.
- **Data.** enwik8's first 8 MiB (the same bytes as enwik9's) at `/root/cdata/enwik8`, and Dolly at `/root/cdata/dolly.jsonl` (15,015 records).
- **Old tools.** Built from their committed source, with the one hard-coded path changed to the files above and nothing else (checked with `diff`). Build: `g++ -O2 -std=c++20`.
- **New tool.** Built with the B1 build line below, which compiles cleanly under `-Wall -Wextra`. The binary used for the parity runs is byte-identical to the final source's binary, because the later edits touched only comments.

| Old tool | New subcommand | Number | Old | New | Match |
|---|---|---|---|---|:-:|
| `trie_fold_lm` | `bpc --negative` | fixed backoff `synth_bpc` | 2.386293 | 2.386293 | yes |
| `trie_fold_lm` | `bpc --negative` | `mean_u` / `match_rate` | 0.486240 / 0.960441 | 0.486240 / 0.960441 | yes |
| `trie_fold_lm` | `bpc --negative` | `full_stored` (bytes) | 9,112,775 | 9,112,775 | yes |
| `trie_fold_lm` | `bpc --negative` | `drop_ctx_stored` | 6,623,887 | 6,623,887 | yes |
| `trie_fold_lm` | `bpc --negative` | `synth_stored` (the 5.4 MB packed layout) | 5,385,965 | 5,385,965 | yes |
| `trie_fold_lm` | `bpc --negative` | `full` / `drop_ctx` / `synth` node, follower and trie-byte rows | 3 rows | 3 rows identical | yes |
| `trie_fold_lm` | `bpc --negative` | context counts, `hash_*`, `build_heap`, order-2 row | 5 rows | 5 rows identical | yes |
| `trie_fold_lm` | `bpc --negative` | `ppmd_bpc` / `chain_bpc` / `inherit_bpc` (CURRENT.md "~2.42") | 2.415026 / 2.417500 / 2.416818 | 2.415026 / 2.417500 / 2.416818 | yes |
| `trie_fold_lm` | `bpc --negative` | `chain` / `inherit` layout rows, `chain_stored` / `inherit_stored`, `deep_nodes_total_ge2` | 5 rows | 5 rows identical | yes |
| `trie_online_mix` | `bpc` | `backoff_bpc` | 2.386293 | 2.386293 | yes |
| `trie_online_mix` | `bpc` | `mix_bpc`, learned mixture, frozen | 2.297794 | 2.297794 | yes |
| `trie_online_mix` | `bpc` | `w0`…`w3` | 0.007465 0.000190 0.000127 0.000410 | same | yes |
| `trie_online_mix` | `bpc` | `w4` / `wmatch` | 0.930724 / 0.061085 | 0.930724 / 0.061085 | yes |
| `trie_online_mix` | `bpc` | `mix_stored` / `mix_beats_backoff` | 5,386,013 / 1 | 5,386,013 / 1 | yes |
| `trie_chat` | `chat` | `wiki_bpc_after_chat` | 2.344571 | 2.344571 | yes |
| `trie_chat` | `chat` | `w4` / `wmatch` / `mix_stored` | 0.960948 / 0.038609 / 11,181,225 | same | yes |
| `trie_chat` | `chat` | whole stdout (10 lines, with `unseen_sample`) | — | byte-identical | yes |
| `trie_qa_probe` | `probe` | whole stdout (84 lines, 14 questions; `model_contains_expect` 0 on all 14) | — | byte-identical | yes |
| `trie_context_8k` | `context8k` | `trie_only_bpc` | 2.3199 | 2.3199 | yes |
| `trie_context_8k` | `context8k` | `trie_plus_context_{4,16,64,512,8192}_bpc` | 2.2660 2.0637 2.0711 2.0715 2.0715 | same | yes |
| `trie_context_8k` | `context8k` | median matched context, 8 histogram rows | 10 bytes; 8 rows | same | yes |
| `trie_context_8k` | `context8k` | TALLY seen / first letter / Thanks / paraphrase / held-out Dolly / wiki fact | 3/3, 3/3, 0/3, 0/3, 1/30, 0/8 | same | yes |
| `trie_context_8k` | `context8k` | whole stdout (221 lines) | — | byte-identical | yes |
| `trie_sample` | `sample` | `gen_dev` file, 492,226 bytes, sha256 `fcc2b9c423ee588a…` | — | byte-identical | yes |
| `trie_sample` | `sample` | `gen_fresh` file, 491,694 bytes, sha256 `8592fb7ee543a3f5…` | — | byte-identical | yes |
| — | `selftest` | tiny-corpus golden: backoff / mixture BPC | — | 1.222293339 / 1.130916116 | golden |

`selftest` needs no data files. It runs `self_test()` and then trains on a 40,000-byte built-in corpus (train 36,000, holdout 4,000). It fails if either BPC is more than 1e-6 from its golden value, or if the weights do not sum to 1. It takes 0.07–0.25 s. The golden values also hold for a `-std=c++23 -O3 -march=native` build.

**Not carried, on purpose.** `trie_fold_lm` prints 12 more lines: the surface score `S` (H, D, L, d4, loop8) and one sample text for each of the PPMD, chain and inheritance variants. Their BPC rows are carried under `--negative`. The surface lines and the samples of those three negative variants are not. Surface scoring lives in `native/tools/surface_score.cpp`, and samples come from `cyphalm_trie sample`.

The old tools also printed `match_filled 4 …` and `self-test ok` on stderr. Those were progress lines, not numbers. The self-test still runs before every subcommand, but it is now silent.

Run times on this machine (new tool, niced): `bpc --negative` 98 s, `chat` 270 s, `probe` 235 s, `context8k` 274 s, `sample` 52 s.

**A note on CURRENT.md's 2.341.** CURRENT.md lists "After 15,011 Dolly turns: 2.341 wiki". Both the old `trie_chat` and the new `chat` print **2.344571** on the Dolly copy here, which has 15,015 records. The difference comes from the data copy, not the consolidation. CYPHALM_QA_AND_CONTEXT_TEST.md already records the 15,011 vs 15,015 difference.

## Line counts

| Before | Lines | | After | Lines |
|---|---:|---|---|---:|
| `trie_fold_lm.cpp` | 1,068 | | `packed_trie.hpp` | 1,293 |
| `trie_online_mix.cpp` | 1,222 | | `cyphalm_trie.cpp` | 779 |
| `trie_together.cpp` | 1,616 | | | |
| `trie_adaptive.cpp` | 1,275 | | | |
| `trie_gria_gate.cpp` | 1,219 | | | |
| `trie_online_holdout.cpp` | 1,200 | | | |
| `trie_chat.cpp` | 1,323 | | | |
| `trie_qa_probe.cpp` | 175 | | | |
| `trie_context_8k.cpp` | 272 | | | |
| `trie_sample.cpp` | 204 | | | |
| **Total** | **9,574** | | **Total** | **2,072** |

`trie_stream_train.cpp` (unchanged) is not counted on either side.

## Retired experiments

These four forks were separate experiments, not mode flags of the model. None of them makes a number that current work builds on, so they were not carried forward. CURRENT.md keeps their results as settled rows. For the record, each one was run once more on 2026-10-27, from its committed source with only the data path changed:

| Tool | What it tested | Result (2026-10-27 rerun) | CURRENT.md row | Last commit containing it |
|---|---|---|---|---|
| `trie_together` | Six-expert mixture plus the GRIA gate, frozen and then still adapting on the holdout; a current-word trie as a seventh expert | `together_frozen_bpc` 2.346144, `together_online_bpc` **2.286208** (adapts on the test bytes); `word_bpc` 2.297358 with word-trie weight **0.000531**; samples `S` 0.742 and 0.790 | "Same mixture, weights still adapting on the holdout: 2.286"; "word-sized trie … weight 0.0005"; surface "0.74–0.83" | `a50c51f` |
| `trie_gria_gate` | GRIA (`hp::GriaGate`) used as a probability gate on the synth trie | backoff 2.38629283, **gated 2.64151118**, GRIA-steered mixture 2.29495819, `beats` 0 (the gate loses to backoff) | "GRIA as a probability gate: 2.642, worse than backoff" | `a50c51f` |
| `trie_online_holdout` | Frozen against online updates on the holdout | frozen 2.386293, online counts 2.117701, online weights (five experts) 2.317383. The online numbers learn the test bytes, so they are not frozen scores | none | `a50c51f` |
| `trie_adaptive` | One online loop: EG weights, GRIA step size and gate, `u`-driven depth, `D_eff` leak, epistemic threshold, match EMA | `frozen_bpc` 2.342228, `online_bpc` 2.337343, `online_counts_bpc` 2.251548; synth samples `S` 0.830 and 0.825 | surface "0.74–0.83" (upper end) | `a50c51f` |

The six tools now served by `cyphalm_trie` were deleted too. Their last commits are:
- `trie_fold_lm` and `trie_online_mix`: `a50c51f`
- `trie_chat` and `trie_qa_probe`: `9f35798`
- `trie_context_8k`: `ab4f9f6`
- `trie_sample`: `2491fe2`

Each of the ten deleted files is unchanged at HEAD from its commit above.

Recovery:
```bash
git show a50c51f:native/tools/trie_together.cpp > trie_together.cpp     # same for the other three a50c51f files
git show 9f35798:native/tools/trie_chat.cpp > trie_chat.cpp            # needed by trie_qa_probe / trie_context_8k
```
The old tools read `/mnt/d/cypha_llm/enwik9` (and `/mnt/d/cypha_llm/chat/dolly.jsonl`) from a fixed path. `trie_qa_probe`, `trie_context_8k` and `trie_sample` take the paths as arguments. The others need the path edited.

## Build and run

```bash
gcc -O2 -c native/third_party/libsais/libsais.c -o libsais.o && g++ -O2 -std=c++20 -I native/include -I native/third_party -I native/third_party/libsais native/tools/cyphalm_trie.cpp native/src/cyphalm/infinigram.cpp libsais.o -o cyphalm_trie

./cyphalm_trie selftest
./cyphalm_trie bpc       enwik8 --negative              # fold_lm + online_mix numbers
./cyphalm_trie chat      enwik8 dolly.jsonl
./cyphalm_trie probe     enwik8 dolly.jsonl
./cyphalm_trie context8k enwik8 dolly.jsonl
./cyphalm_trie sample    enwik8 gen_dev.txt gen_fresh.txt  # input to scribe_ladder
```

Options: `--wiki P`, `--dolly P`, `--train N`, `--sample N` (bytes; the defaults are 7,340,032 and 8,388,608), and `--no-guard`. The `bpc` guard stops the run if the backoff is outside 2.36–2.41 on the default split. `chat` also takes `--prompt TEXT` and `--no-repeat`. `sample` also takes `--prompts N` and `--bytes N`.

## CMake block (not applied; `native/CMakeLists.txt` belongs to another agent)

```cmake
# --- B1: CyphaLM packed trie, one library and one tool (PLAN Track B1) ----------------------------
# Header-only library include/cypha/cyphalm/packed_trie.hpp. cyphalm_trie replaces trie_fold_lm,
# trie_online_mix, trie_chat, trie_qa_probe, trie_context_8k and trie_sample. Only the selftest is a
# CTest (no corpus); bpc/chat/probe/context8k/sample need enwik8 and Dolly and are run by hand.
add_executable(cyphalm_trie
  tools/cyphalm_trie.cpp
  src/cyphalm/infinigram.cpp
  third_party/libsais/libsais.c)
target_include_directories(cyphalm_trie PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libsais")
cypha_mingw_apply_exe_link_options(cyphalm_trie)
add_test(NAME native_cyphalm_trie_selftest COMMAND "$<TARGET_FILE:cyphalm_trie>" selftest)
set_tests_properties(native_cyphalm_trie_selftest PROPERTIES TIMEOUT 60)
# --- end B1 block ---------------------------------------------------------------------------------
```

This block was checked in a scratch CMake project that links `native/`'s `tools`, `src`, `include` and `third_party`, using CMake 3.28 in Release mode. It builds, and `ctest` passes `native_cyphalm_trie_selftest` in 0.07 s. It compiles `infinigram.cpp` and `libsais.c` into the tool instead of linking `cypha_core`, so the tool does not pull in the rest of the core library. Linking `cypha_core` would also work, because `cypha_core` already holds both files. PLAN B1 also lists Q/A-probe and 8K golden CTests. Those need enwik8 and Dolly, so they are not in this block.

## References updated

- `docs/CURRENT.md`: the "Tools:" line.
- `docs/reports/CYPHALM_QA_AND_CONTEXT_TEST.md`: the setup rows, the tools table, and the build and run commands.
- `docs/reports/SCRIBE_V1.md`: the code line and the reproduce commands.
- `native/tools/scribe_ladder.cpp`: comments and the usage string (`trie_sample` → `cyphalm_trie sample`).
- `docs/reports/REMOVED_NON_CYPHALM_QA.md`: a note on where the tools went.
- `docs/PLAN.md`: one sentence (the chat model loads Dolly only).
- `CYPHA_BILL_OF_WORK.md`: one line.
- `CHANGELOG.md`: an Added entry and a Removed entry.
- Not changed: `README.md` and `MODEL_CARD.md`, which do not name the files.

Older CHANGELOG entries and the narrative in the reports still name the old tools. They describe what was run at the time.

## Still to do

`native/tools/trie_stream_train.cpp` still has its own copy of the model. Odin's live 1 TB training run uses it through `scripts/run_train.sh`, so it was left completely untouched. It should move onto `packed_trie.hpp` after the 1 TB run ends. That is Odin's call. When it does, parity can be checked the same way: same split, then a diff of the printed numbers.
