# Handoff for Claude Code: Cypha / CyphaLM

**Odin Loch** · rewritten 2026-10-28 at the end of the second working session of the day; updated the same day for B4c, S1, RAM M10–M15, S4, S6 and A4c · repo `odin-loki/Cypha` (branch `main`)

Read this first. Then read the documents in §3 for whatever you are working on. The goal, the rules and Odin's standing instructions are in §1–§2 and are not optional.

---

## 1. What this project is

Cypha is Odin Loch's own C++ AI. **CyphaLM** is its language model: a packed byte trie (orders 1–4 plus a 4-byte match, a learned 6-expert mixture) and an InfiniGram ∞-gram expert. Core Cypha is a statistical classifier (DIF: WorldPrior θ₀ plus ClassDifferential Δk diagonal Gaussians, natural-gradient updates, MDL shrinkage, epistemic variance, NIG/GH gate, OOD flag).

**The goal.** Working general NLP / LLM behaviour with:
- symbolic plus statistical machinery, **no neural nets at runtime**;
- low RAM;
- online learning;
- being picky about what it learns;
- saying it doesn't know rather than guessing.

**Where it stands.**
- **Language model.** BPC 2.298 on the enwik 8 MiB protocol (chat model 2.345). Fresh text: 2.166 with InfiniGram; 1.912 with the class and copy experts, 2.027 with class and similar-context. The similar-context expert is on by default in `qa`, `sample` and `context8k`, off inside answers (Odin, 2026-10-28); class and copy are options.
- **Text.** SCRIBE v2, the generation score that passes its ladder, calls the trie's text word salad (0.014; human 0.770). Surface S does not see it.
- **Questions.** CyphaLM answers almost nothing by itself: 2 right, 0 wrong at a 2% wrong rate on dev with the fact shard; 0 on the frozen test. Its "I don't know" is real: precision 0.635, recall 0.94 on the test.
- **RAM.** The qa model's resident memory fell from 228 MB to 76 MB (§4.3).

---

## 2. Rules and Odin's standing instructions

1. **CyphaLM does every task** (PLAN rule 10).
   - No string search, retrieval, or hand-written lookup may produce an answer. Tools may only add training text, add experts that predict the next byte, or gate.
   - Three Q/A tools that broke this were deleted on 2026-10-27 ([record](reports/REMOVED_NON_CYPHALM_QA.md)).
   - **Do not cheat. If in doubt, ask Odin.**
2. **"Doesn't know" is a feature, not a refusal** (rule 9).
   - The output is a plain message plus a tag, `<|unknown reason=<CODE> entity="…" relation=… gap=<id>|>` (DISCOVERY §3.5). CyphaLM is trained to write it itself.
   - A gate that decides whether to abstain on CyphaLM's own answer counts as CyphaLM doing the task (Odin, DISCOVERY §8a). The gate in use is `qa`'s threshold gate; the Cypha-classifier gate was measured and deleted (TOOL_PROFILE P4).
   - The familiarity token is allowed on the gate side, after `Assistant: ` (Odin, 2026-10-28).
   - A reason code describes what CyphaLM itself has read, not the wider fact store (Odin: "most honest", 2026-10-28). Where the data isn't there, saying so is right; S1 stays at 82% (PLAN ledger 61).
   - Next-byte experts that hurt answers are turned off inside answers (after `Assistant: `, and from the end of a `qa` prompt), so they can be defaults for text (A4b, B4c).
3. **Everything has a process.** The AI does nothing it has not been given data or a process for. The library is `docs/thought/processes.yaml`.
4. **Minimal code debt: "the minimal and best" tools.** Profile before adding, cut what doesn't earn its place (`docs/TOOL_PROFILE_PLAN.md` §5). **Keep `scripts/gguf_mine.py`** (Odin's decision).
5. **Metrics, always together:** held-out answers (right / abstained / wrong), the text score, BPC. BPC alone is a poor metric (Odin). The text score in the rules is SCRIBE v2, with surface S reported alongside (Odin, 2026-10-29).
6. **Science hygiene** (PLAN rules 1–4): register predictions before a run, then score each one hit or miss; a dev split for choices; the frozen test read once per change; Wilson / bootstrap intervals; LCI ≥ 0.95 to keep a change.
7. **Writing style.** Docs and reports are dated, with the byline **Odin Loch**. Markdown only, never .docx; no images. Plain, short sentences.
8. **Git.** Commit and push finished, verified work to `main`. End commit messages with:
   ```
   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   Claude-Session: <your session's URL>
   ```
   Keep CI green: all six jobs of `.github/workflows/ci.yml` passed on 2026-10-28 (§4.5).
9. **Never touch `native/tools/trie_stream_train.cpp` or `scripts/run_train.sh`.** Odin's live 1 TB training run uses them. The tool keeps its own copy of the trie and moves to the library only after the run ends, at Odin's call.
10. **Work style.** Odin likes parallel agents in small usage chunks, each with a clear brief and file ownership, so they don't collide. Another Claude session may be pushing to `main` at the same time (SCRIBE v2 did on 2026-10-28): fetch before every publish.
11. **No Qwen inference in the cloud container** (PLAN ledger 54). Analytics on the weights only; inference and distillation run on Odin's PC.
12. **Settle open design questions by measurement**, with the evidence recorded (PLAN ledger 55).

---

## 3. Map of the documents

| Document | What it is |
|---|---|
| `docs/PLAN.md` | The default plan: rules, tracks A–S, and a ledger of every point Odin raised (rows 1–58) |
| `CYPHA_BILL_OF_WORK.md` | Open, future and closed work items. **Keep it current** |
| `CHANGELOG.md` | The Unreleased section lists everything added, changed or removed |
| `docs/CURRENT.md` | Current numbers and which tool makes them |
| `docs/DISCOVERY.md` | Knowing what it knows: epistemic states, reason codes, the don't-know tag, the hypothesis store (S5 built), the gap map (S3 built), generators |
| `docs/THOUGHT.md`, `docs/thought/processes.yaml` | The thought-process library: 144 processes |
| `docs/SEMANTICS.md` | Core Cypha as the meaning layer. S-1 failed; S-1b reached PPMI-SVD's level but failed its gates |
| `docs/EMBEDDING_GEOMETRY.md` | Copying Qwen3-Embedding's geometry into Cypha. QE0–QE2 inputs done from the weights; the rest on Odin's PC |
| `docs/CYPHALM_ANSWERING.md` | How CyphaLM answers by itself: routes R1–R3, steps A0–A5 with results |
| `docs/TOOL_PROFILE_PLAN.md` | How support tools earn their place; P1–P4 done |
| `docs/GGUF_MINING.md`, `scripts/gguf_mine.py` | Mining a GGUF teacher |
| `docs/reports/RAM_PROFILE.md` | CyphaLM's memory, M1–M8, every cut measured |
| `docs/reports/*.md` | One dated report per piece of work, with every number |

---

## 4. State at handoff (2026-10-28)

### 4.1 Done today (all on `main`, each with a dated report)

| Item | Result | Report |
|---|---|---|
| D4 Wikidata seeds, fact-text shard | Kept patterns 44 → 70, admitted facts 832 → 1,216, Wikidata agreement 0.951; `pattern_learn --shard` writes question-form fact text with a leakage check | `D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md` |
| F2R raw-byte patterns | 81 patterns over the dump's own bytes, 1,025 facts at 0.950; R3 frames from them | `F2R_RAW_BYTE_PATTERNS.md` |
| B4 class-context expert | Fresh text 2.166 → 2.088 BPC (shape classes) | `B4_CLASS_CONTEXT_EXPERT.md` |
| A1c, A2, A3 | Entity familiarity (AUROC 0.795), the copy expert, the fact shard (2 right, 0 wrong on dev), statement scoring (0) | `A1C_…`, `A2_A3_…` |
| A1d / A2b answering variants | Span candidates pick the right entity 76 / 100 (E1: 63). The `low`-only don't-know token: dev 0.600 / 1.00, **test 0.635 / 0.94**. No variant names the entity in the tag | `A1D_A2B_ANSWERING_VARIANTS.md` |
| A4 / A4b similar-context expert | Fresh text −0.061 BPC over trie + IG + class; no answers by itself. **Now a default** in `qa`, `sample`, `context8k`, gated off inside answers (answers unchanged: C1L keeps 2 right, 0 wrong); 2.2 MB shipped vectors (`data/cyphalm/`); +42 MB resident | `A4_SIMILAR_CONTEXT_EXPERT.md` |
| B4B language-model defaults | Class (shape, 3 orders) + copy (m 8, window 16,384): fresh text 1.912. Not defaults then: with the fact shard they echo the question (rule 3) | `B4B_LM_DEFAULTS.md` |
| **B4c echo gate** (Odin: all three in parallel) | Class + copy off inside answers, like the similar-context expert. **Defaults** in `qa` and `context8k` (class + sim + copy); `sample` takes sim + copy, the class expert only with `--cls` (SCRIBE v2 falls with it). Rule 3 holds: dev 520 / 520 and test 800 / 800 answer rows identical. `qa` holdout trie + IG 2.0729 → **1.8345** | `B4C_ECHO_GATE.md` |
| **S1 reason codes, evidence cards** | Codes right 82 / 100 dev, 169 / 200 test; the 90% gate is missed (17 of 18 dev errors: the `low` token on in-slice entities) and Odin closed it there. "Unknown" = what CyphaLM has read (Odin: "most honest"). `qa --cards` via `FmInfiniGram::locate` (+10 MB, 0.02 ms a card); answers byte-identical | `S1_REASON_CODES_AND_EVIDENCE.md` |
| **S4 unseen estimates, frontier** | Gate met: Good–Turing with the page as the unit is 0.968 / 0.980 / 1.000 of the held-out new share (words / entities / facts; plain n₁/N: 0.55–0.80); Chapman a lower bound in 7 / 7 relations; the frontier: 7,124 slots known, about 9,970 unseen on pages already read | `S4_UNSEEN_AND_FRONTIER.md` |
| **S6 hypothesis scoring** | P0 0.738 [0.717, 0.758] on a fresh planted set, gate 0.75 missed; the store's pairwise rules carry it (hard constraint +0.010, path +0.004); value terms untested until S10 | `S6_HYPOTHESIS_SCORING.md` |
| **A4c similar-context, distinct contexts** | `dedupe` k 8,192 is the default: fresh −0.0040 [−0.0048, −0.0032], S and SCRIBE up, answers unchanged, the expert 40 → 19 MB; counts per k-means list were worse and are removed | `A4_SIMILAR_CONTEXT_EXPERT.md` |
| **M15 class build in key buckets** | Built `qa` 226 → 194 MB (defaults), 259 → 235 MB (C1L + defaults), repeating within 0.1 MB; lossless | `RAM_PROFILE.md` |
| **S7 hypothesis tests** | I5, I6 met (`PreregLog`, schema triggers); s and c measured for T1, T3–T6 (T4 0.99 / 0.83, T5 0.77 / 1.00); T2 cut (no rule fires); the loop supports 56, all true, but decides only 6% | `S7_HYPOTHESIS_TESTS.md` |
| **S15 temporal holdout** | Built and frozen: 38,393 facts (a 10,521 / b 8,574 / c 1,917 / u 17,381) plus false pairs, dev / test by subject; Dolly holds 35 of the c pairs, so CyphaLM-based generators drop them | `S15_TEMPORAL_HOLDOUT.md` |
| **Slow CTest triage** | All 38 failures date from the hp cutover (`0653e25`): 35 test the removed hybrid model, 2 are hp-wrapper defects (vocabularies under 256), 1 time limit | `SLOW_TEST_TRIAGE.md` |
| Tool profile P2–P4 | Tagger and word-vector class tables cut from the class expert (the tagger stays as the question parser); `cypha_gate` deleted; `pattern_learn` fails the Q/A cut, flagged for Odin | `TOOL_PROFILE_P2_GRID.md` |
| S-1b learned contexts | Cypha's own learner reaches PPMI-SVD's level (0.683 / 0.246 / 0.669); gates failed on rare words | `S1B_LEARNED_CONTEXTS.md` |
| S5 hypothesis store, S3 gap map | `hypothesis.hpp` (I1, I4, I7, I8), `gap_map.hpp`, `pattern_learn --gaps`: 27,842 gaps with VOI. CyphaLM's dev tags name their question's slot 0 / 750 | `S3_S5_GAP_MAP_AND_HYPOTHESES.md` |
| Qwen3-Embedding, weights only | QE0 hit on 0.6B; QE1 static dissection of 0.6B and 8B (Q4 hit); the hashed probe set and its builder in `mined/probe/` | `QWEN3_EMBEDDING_DISSECTION.md` |
| **CyphaLM's RAM** (Odin's request) | §4.3; M10–M14: `qa --save DIR` / `--model DIR`, served `qa` under 200 MB in every configuration | `RAM_PROFILE.md` |
| CI | All six jobs green (§4.5) | |
| SCRIBE v2 and its paper | From a parallel session, merged into `main` | `SCRIBE_V2.md`, `paper/scribe/` |

### 4.2 CyphaLM numbers now

| Read | Value |
|---|---|
| BPC, holdout, learned mixture | 2.298 (bpc model); chat model 2.345 |
| Fresh text | trie + IG 2.166; + class + copy 1.912; + class + similar-context 2.027 |
| `qa` holdout, default model | trie + IG 2.0729 → **1.8269** with class + similar-context + copy (defaults since B4c, off inside answers; A4c's similar-context default: B4c read 1.8345); `context8k` 2.0715 → 1.8271 |
| Class expert's table | default `shape` = three orders, compact, seen once dropped (fresh 2.0853; Odin, 2026-10-28) |
| Text, SCRIBE v2 | trie 0.014, trie + class 0.002, `sample`'s default (trie + similar-context + copy) 0.064 [0.052, 0.078] (A4c; B4c's 0.056), human 0.770 |
| Reason codes (S1) | 82 / 100 dev, 169 / 200 test; `qa --cards` gives evidence cards |
| Q/A, frozen test (C1L) | 0 right kept at 2% wrong; don't-know precision 0.635, recall 0.94 |
| Q/A, dev, with D4's shard (C1L) | 2 right, 0 wrong at 2% wrong |
| R3 with raw frames | test 1 right, 0 wrong |

### 4.3 Memory

Odin asked on 2026-10-28 to reduce CyphaLM's RAM. Every cut was measured against a rule fixed before it ([`reports/RAM_PROFILE.md`](reports/RAM_PROFILE.md), M1–M9).

| Part (qa model) | Before | Now |
|---|---:|---:|
| Trie | 38.5 MB training form | **7.7 MB** packed (M4), bit-identical; a stored file loads in 17 ms (M7) |
| InfiniGram | 77 MB | **26.0 MB** FM-index (M5), the same answers; a stored file loads in 0.06 s (M8) |
| Class expert, 3 orders | 77.7 MB | **18.2 MB** (M2 compact, M3 seen-once dropped: +0.0009 BPC) |
| Class build, memory it adds | 560 MB | 132 MB (M1, identical table) |
| Whole qa model, resident (harness) | 228 MB | **76 MB** |
| `cyphalm_trie qa`, peak | 264 MB | **171 MB** (M9; with the don't-know token and shard 368 → 199 MB) |
| `chat` from a stored model | 104 MB and 476 s (trains) | **21 MB and 12 s** (`--model FILE`) |
| `qa` served from a store (M10–M14) | built: A0 186.8, + class + copy 245.3, C1L + class + copy 275.1 MB | **138.4 / 169.4 / 189.1 MB**: `qa --save DIR`, then `qa --model DIR`; parts load in 0.15 s; 3–5× faster; every number unchanged |

- Every printed number of `cyphalm_trie` is unchanged by M9 (bpc, chat, sample, context8k, dktext, cls, qa, the `--sim` paths).
- **Served `qa` is under 200 MB** in every configuration (M10–M14). With B4c's defaults (C1L, class + similar-context + copy): built 264–275 MB and about 705 s; served **189 MB and 229 s**, the same rows (checked on the merged tool, 2026-10-28). A served model refuses the token statistics that read the wiki-stage trie (`question_lp`, `ent_lp`), `--sim src=all` and `--cards`.
- **Building** (M15): the class build no longer sets the peak (it adds about 20 MB). Built peaks: defaults 194 MB, C1L + defaults 235 MB, repeating within 0.1 MB. The FM-index build sets them now (libsais's 4-byte scratch per text byte). The earlier 226–302 MB spread was the C library's heap under the old per-chunk merge.

### 4.4 CTests

Run these on every change: `native_cyphalm_trie_selftest`, `native_fm_infinigram_test`, `native_epistemic_test`, `native_thought_test`, `native_geometry_test`, `native_ecs_lci_test`, `native_scribe_selftest`, `native_cyphalm_neural_smoke`, `native_class_context_test`. `native_fm_infinigram_test` runs 191,885 checks (locate included), `native_epistemic_test` 654 (S4, S6, S7), `native_class_context_test` 1,043 (M15). `pattern_learn` builds with `g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp`; its S3 `--gaps` run must stay byte-identical.

### 4.5 CI

- All six jobs of `ci.yml` passed on 2026-10-28. Two causes of red were fixed: `cypha_gate.cpp` included `unistd.h` (Windows MSVC; the tool is deleted), and `native_cyphalm_neural_smoke`'s kernel-parity limit was a flat 1e-6 on log-probabilities down to −15, inside float32 rounding (now scaled by 1 + |log p|; planted kernel bugs still fail it).
- **Outside CI:** the 38 failing `cypha_slow` tests are triaged ([SLOW_TEST_TRIAGE](reports/SLOW_TEST_TRIAGE.md)). 35 test the Hybrid GRIA+LSTM model removed on 2026-09-18 (cut candidates, §5); 2 show hp-wrapper defects with small vocabularies; `forecast_smoke` needs about 6 min (mostly kernel time) against a 300 s limit. CI runs `-LE cypha_slow`, so none of them turns a job red.

---

## 5. Decisions waiting for Odin

1. ~~`pattern_learn` fails the tool-profile cut.~~ **Answered 2026-10-29: keep it** (PLAN ledger 65). It is Track S's fact and hypothesis tool (fact store, D4, F2R, the gap map, S4, S6, S7, S8); it never answers a question (rule 10).
2. **The `low`-only don't-know token** (A1D/A2B): the form that lets the token and the fact shard work together (dev 2 right, 0 wrong). Odin allowed the token; this is a narrower form.
3. **`ent_title` against `ent_bold`.** `ent_title` reads XML title tags; `ent_bold` reads none and meets D2 on dev (0.833 / 0.50) but halves recall.
4. **The entity in the tag** needs a copy weight that depends on context (inside the answer or not). That changes the mixture.
5. ~~Class and copy with the fact shard.~~ **Answered by B4c:** the same gate works for class and copy; rule 3 holds, and both are defaults where the rules hold.
6. ~~Which text score the rules use.~~ **Answered 2026-10-29: SCRIBE v2** (PLAN ledger 64), with surface S reported alongside. So the class expert stays out of `sample`'s default (SCRIBE v2 falls with it, 0.061 → 0.025).
7. **Track S:** where gap demand comes from; Wikidata's type-level rate as P(exists); finer entity types (S3/S5 report). Answered 2026-10-28: the reason code describes what CyphaLM has read ("most honest"), and S1 stays at 82% ("if the data isn't there, it isn't there").
8. **PPMI-SVD without the evaluation bytes** as the standard vector file (its leak is worth 0.001 BPC).
9. **Decided by measurement (PLAN ledger 62), open to Odin's veto:** DISCOVERY §3.3's missing mass for text is Good–Turing with the page as the unit (S4: within 4%; n₁/N undercounts by 20–50%); S6's P0 keeps the constraint term and judges its parts by one joint test (S6b).
10. **Slow-test cuts: answered 2026-10-29, delete** the 35 hybrid-era tests and the code only they use (PLAN ledger 63; the cut is running). Still open: the hp wrapper's small-vocabulary defects and `forecast_smoke`'s time.
11. **S15 defaults, open to Odin's veto:** the Wikidata response cache (2.35 MB) is committed for audit; D4's T-QA exclusion is kept, so the benchmark is thin on CyphaLM's own 8 MiB slice (105 train subjects).
12. **S7's questions** (batch control too strict; a "not found" outcome as evidence; re-estimating φ) are being settled by measurement (S7b, PLAN ledger 55).
13. **When the 1 TB run ends:** move `trie_stream_train.cpp` onto `packed_trie.hpp` (and `pack()`, M4).

---

## 6. Next work, in order

1. **S8** generators batch 1 (G2 rule induction, G14 contradiction resolution, G6 and G7 link prediction; G16 teacher runs on Odin's PC) and **S7b** (the test loop's likelihoods, batch control and φ): running 2026-10-29, with S15 as the fresh source (its test half read once per change, logged). S6's next change rides on S8's link evidence. (S1, S4, S6, S7, S15, the echo fix, A4c, M15 and serving from stored files were done 2026-10-28.)
2. **RAM:** the FM-index build's peak when building (M16, running: libsais scratch, 4 bytes per text byte; C1L + defaults 235 MB). When the class expert joins `sample`'s or `chat`'s defaults, their `--save` / `--model` should carry its table.
3. **The slow-test cuts** (§5.10), after Odin's call.
4. **Similar-context expert:** more probed lists (k 8,192's recall is 0.65), online keys. **S4:** the §3.4 unknown-unknown signals other than vocabulary (no producers yet).
5. **On Odin's PC:** the 8B QE0 gate and timing, QE2 probe vectors, QE3, QE4 `pairs`, teacher Q/A text (QWEN3_EMBEDDING_DISSECTION §9).
6. **Bill of work upkeep** with every commit.

---

## 7. Environment notes (cloud container)

| Thing | Where |
|---|---|
| Repo | `/home/user/Cypha` (session branch kept in step with `main`) |
| Data | `/root/cdata/enwik8`, `/root/cdata/dolly.jsonl` (15,015 records, GitHub copy d000e30; the Hugging Face copy has 15,011) |
| Build | `cmake -S native -B <dir> -DCMAKE_BUILD_TYPE=Release && cmake --build <dir> --target <t> -j2 && ctest --test-dir <dir> -R <test>` |
| Scratch results | `/root/work/<track>/` (lost when the container is reclaimed; every report says how to rebuild) |
| Python with llama-cpp-python | `/root/work/teacher/venv/bin/python` (weights-only analytics) |
| Similar-context vectors | `data/cyphalm/simvec_ppmi32.bin`, compiled in by CMake; `$CYPHA_SIMVEC` overrides; `--no-sim` turns the expert off |

**Gotchas.**
- **The machine:** 4 shared cores, 16 GB RAM. Use `nice`, give each agent its own files.
- **Network:** Hugging Face and Wikidata SPARQL are reachable now; the Wikipedia action API throttles (429).
- **A wait loop that greps for a process name** matches its own command line (`pgrep -f`), so it never ends. Wait on a file instead.
- **Publishing:** a clean worktree on `origin/main` (`git fetch && git reset --hard origin/main`), copy the verified files, commit, push; then merge `main` into the session branch.
