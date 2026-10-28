# Handoff for Claude Code: Cypha / CyphaLM

**Odin Loch** · rewritten 2026-10-28 at the end of the second working session of the day · repo `odin-loki/Cypha` (branch `main`)

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
3. **Everything has a process.** The AI does nothing it has not been given data or a process for. The library is `docs/thought/processes.yaml`.
4. **Minimal code debt: "the minimal and best" tools.** Profile before adding, cut what doesn't earn its place (`docs/TOOL_PROFILE_PLAN.md` §5). **Keep `scripts/gguf_mine.py`** (Odin's decision).
5. **Metrics, always together:** held-out answers (right / abstained / wrong), the text score, BPC. BPC alone is a poor metric (Odin). SCRIBE v2 now passes its ladder where surface S does not (§4.1); which text score the rules use is Odin's call (§5).
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
| B4B language-model defaults | Class (shape, 3 orders) + copy (m 8, window 16,384): fresh text 1.912. **Defaults unchanged**: with the fact shard they echo the question (rule 3) | `B4B_LM_DEFAULTS.md` |
| Tool profile P2–P4 | Tagger and word-vector class tables cut from the class expert (the tagger stays as the question parser); `cypha_gate` deleted; `pattern_learn` fails the Q/A cut, flagged for Odin | `TOOL_PROFILE_P2_GRID.md` |
| S-1b learned contexts | Cypha's own learner reaches PPMI-SVD's level (0.683 / 0.246 / 0.669); gates failed on rare words | `S1B_LEARNED_CONTEXTS.md` |
| S5 hypothesis store, S3 gap map | `hypothesis.hpp` (I1, I4, I7, I8), `gap_map.hpp`, `pattern_learn --gaps`: 27,842 gaps with VOI. CyphaLM's dev tags name their question's slot 0 / 750 | `S3_S5_GAP_MAP_AND_HYPOTHESES.md` |
| Qwen3-Embedding, weights only | QE0 hit on 0.6B; QE1 static dissection of 0.6B and 8B (Q4 hit); the hashed probe set and its builder in `mined/probe/` | `QWEN3_EMBEDDING_DISSECTION.md` |
| **CyphaLM's RAM** (Odin's request) | §4.3 | `RAM_PROFILE.md` |
| CI | All six jobs green (§4.5) | |
| SCRIBE v2 and its paper | From a parallel session, merged into `main` | `SCRIBE_V2.md`, `paper/scribe/` |

### 4.2 CyphaLM numbers now

| Read | Value |
|---|---|
| BPC, holdout, learned mixture | 2.298 (bpc model); chat model 2.345 |
| Fresh text | trie + IG 2.166; + class + copy 1.912; + class + similar-context 2.027 |
| `qa` holdout, default model | trie + IG 2.0716 → 1.9686 with the similar-context expert (default since A4b) |
| Class expert's table | default `shape` = three orders, compact, seen once dropped (fresh 2.0853; Odin, 2026-10-28) |
| Text, SCRIBE v2 | trie 0.014, trie + class 0.002, human 0.770 |
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

- Every printed number of `cyphalm_trie` is unchanged by M9 (bpc, chat, sample, context8k, dktext, cls, qa, the `--sim` paths).
- **Missed:** qa's peak with 3-order classes is 235 MB, not under 200: the builds come after training, because A1c's tokens need the wiki trie. Next: load stored index and class files in `qa`.

### 4.4 CTests

Run these on every change: `native_cyphalm_trie_selftest`, `native_fm_infinigram_test`, `native_epistemic_test`, `native_thought_test`, `native_geometry_test`, `native_ecs_lci_test`, `native_scribe_selftest`, `native_cyphalm_neural_smoke`.

### 4.5 CI

- All six jobs of `ci.yml` passed on 2026-10-28. Two causes of red were fixed: `cypha_gate.cpp` included `unistd.h` (Windows MSVC; the tool is deleted), and `native_cyphalm_neural_smoke`'s kernel-parity limit was a flat 1e-6 on log-probabilities down to −15, inside float32 rounding (now scaled by 1 + |log p|; planted kernel bugs still fail it).
- **Outside CI:** 38 tests labelled `cypha_slow` fail on a full local `ctest` (older research benchmarks: intelligence-profile statistics, the d39–d54 domain smokes, a canary recall check, a mixer margin). CI runs `-LE cypha_slow`, so they do not turn a job red. Not investigated.

---

## 5. Decisions waiting for Odin

1. **`pattern_learn` fails the tool-profile cut** as a Q/A tool (+1.5 right per 100 for 3,026 lines; needs +6). The fact store, D4, F2R and the gap map depend on it. Keep or delete?
2. **The `low`-only don't-know token** (A1D/A2B): the form that lets the token and the fact shard work together (dev 2 right, 0 wrong). Odin allowed the token; this is a narrower form.
3. **`ent_title` against `ent_bold`.** `ent_title` reads XML title tags; `ent_bold` reads none and meets D2 on dev (0.833 / 0.50) but halves recall.
4. **The entity in the tag** needs a copy weight that depends on context (inside the answer or not). That changes the mixture.
5. **Class and copy with the fact shard.** Both lower BPC but hurt answering when the shard is in. The similar-context expert now has a gate that turns it off inside answers (Odin: use it as soon as it is useful). The same gate for class and copy?
6. **Which text score the rules use.** SCRIBE v2 passes its ladder; surface S does not, and scores the experts' word salad above human text.
7. **Track S:** where gap demand comes from; Wikidata's type-level rate as P(exists); which knowledge the reason code describes; finer entity types (S3/S5 report).
8. **PPMI-SVD without the evaluation bytes** as the standard vector file (its leak is worth 0.001 BPC).
9. **When the 1 TB run ends:** move `trie_stream_train.cpp` onto `packed_trie.hpp` (and `pack()`, M4).

---

## 6. Next work, in order

1. **S1 on CyphaLM's answers** (DISCOVERY §13): reason codes and evidence cards. The tags' reason is right on 45 of 72 on the test; the gate is 90%. Evidence cards need positions in the training text: `FmInfiniGram` gives counts, not positions, so add sampled suffix-array entries (about n/32 × 4 bytes) for `locate`.
2. **The echo fix** (§5.5), then read B4B's rule 3 again for class, copy and similar-context.
3. **S4** (unseen estimates and the frontier report), then S6.
4. **Similar-context expert, counts version:** next-word counts per cluster of two-word contexts instead of k neighbours (k keeps paying).
5. **RAM next:** serve from the stored files only (M7 + M8, and a stored class table), so serving never trains or builds.
6. **On Odin's PC:** the 8B QE0 gate and timing, QE2 probe vectors, QE3, QE4 `pairs`, teacher Q/A text (QWEN3_EMBEDDING_DISSECTION §9).
7. **Bill of work upkeep** with every commit.

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
