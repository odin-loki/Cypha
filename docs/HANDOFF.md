# Handoff for Claude Code: Cypha / CyphaLM

**Odin Loch** · written 2026-10-28 at the end of a long working session · repo `odin-loki/Cypha` (branch `main`)

Read this first. Then read the documents in §3 for whatever you are working on. The goal, the rules and Odin's standing instructions are in §1–§2 and are not optional.

---

## 1. What this project is

Cypha is Odin Loch's own C++ AI. **CyphaLM** is its language model: a packed byte trie (orders 1–4 plus a 4-byte match, a learned 6-expert mixture) and an InfiniGram suffix-array ∞-gram expert. Core Cypha is a statistical classifier (DIF: WorldPrior θ₀ plus ClassDifferential Δk diagonal Gaussians, natural-gradient updates, MDL shrinkage, epistemic variance, NIG/GH gate, OOD flag).

**The goal.** Working general NLP / LLM behaviour with:
- symbolic plus statistical machinery, **no neural nets at runtime**;
- low RAM;
- online learning;
- being picky about what it learns;
- saying it doesn't know rather than guessing.

**Where it stands.**
- **Language model.** CyphaLM's BPC is 2.298 on the enwik 8 MiB protocol (fixed backoff 2.386).
- **Text.** Generation is passable.
- **Questions.** By itself it scored 0/8 on Wikipedia facts and 1/30 on held-out Dolly. See `docs/CYPHALM_ANSWERING.md` §1 for why: there is a missing path from question bytes to answer bytes.

---

## 2. Rules and Odin's standing instructions

1. **CyphaLM does every task** (PLAN rule 10).
   - No string search, retrieval, or hand-written lookup may produce an answer. Tools may only add training text, add experts that predict the next byte, or gate.
   - Three Q/A tools that broke this were deleted on 2026-10-27 ([record](reports/REMOVED_NON_CYPHALM_QA.md)).
   - **Do not cheat. If in doubt, ask Odin.**
2. **"Doesn't know" is a feature, not a refusal** (rule 9).
   - The output is a plain message plus a tag, `<|unknown reason=<CODE> entity="…" relation=… gap=<id>|>` (DISCOVERY §3.5).
   - CyphaLM is trained to produce it itself.
   - The Cypha-classifier abstain gate **counts as CyphaLM doing the task for now**. Discovery mode investigates it later (DISCOVERY §8a).
3. **Everything has a process.**
   - The AI does nothing it has not been given data or a process for.
   - It can learn new things.
   - The process library is `docs/thought/processes.yaml`.
4. **Minimal code debt: "the minimal and best" tools.** Profile before adding, cut what doesn't earn its place (`docs/TOOL_PROFILE_PLAN.md`). **Keep `scripts/gguf_mine.py`** (Odin's decision).
5. **Metrics, always together:**
   - held-out answers (right / abstained / wrong);
   - the surface score;
   - BPC.

   BPC alone is a poor metric (Odin).
6. **Science hygiene** (PLAN rules 1–4):
   - register predictions before a run, then score each one hit or miss;
   - a dev split for choices;
   - the frozen test read once per change;
   - Wilson / bootstrap intervals;
   - LCI ≥ 0.95 to keep a change.
7. **Writing style.**
   - Docs and reports are dated, with the byline **Odin Loch**.
   - Markdown only, never .docx; no images.
   - Plain, short sentences.
8. **Git.** Commit and push finished, verified work to `main`. End commit messages with:
   ```
   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_01WdQxkeK7PxswL73mseNWCy
   ```
   Replace the session line with your own session's, if you have one.
9. **Never touch `native/tools/trie_stream_train.cpp`.** Odin's live 1 TB training run uses it through `scripts/run_train.sh`. It still has its own copy of the trie model and moves to the library only after the run ends, at Odin's call.
10. **Work style.** Odin likes parallel agents working in small usage chunks (about 1% of a Max plan each), each with a clear brief and file ownership, so they don't collide.

---

## 3. Map of the documents

| Document | What it is |
|---|---|
| `docs/PLAN.md` | The default plan: rules, tracks A–S, and a ledger of every point Odin raised (rows 1–50) |
| `CYPHA_BILL_OF_WORK.md` | Open, future and closed work items. **Keep it current** |
| `CHANGELOG.md` | The Unreleased section lists everything added or removed |
| `docs/CURRENT.md` | Current numbers and which tool makes them |
| `docs/DISCOVERY.md` | Knowing what it knows: epistemic states, reason codes, the don't-know tag, the hypothesis store, 20 generators, 16 discovery processes |
| `docs/THOUGHT.md`, `docs/thought/processes.yaml` | The thought-process library: 144 processes, 25 categories, 856 steps, each a learnable option |
| `docs/SEMANTICS.md` | Core Cypha as the meaning layer. **S-1 has been run and failed its gate**; the lesson is in §7 |
| `docs/EMBEDDING_GEOMETRY.md` | **The newest plan:** copy Qwen3-Embedding-8B's geometry into Cypha. Dissection list, forces, distillation, QE0–QE9, predictions Q1–Q10 |
| `docs/CYPHALM_ANSWERING.md` | **Proposal waiting for Odin**: how CyphaLM answers questions by itself (routes R1–R3, steps A0–A5) |
| `docs/TOOL_PROFILE_PLAN.md` | How support tools earn their place (the B / B+T / F / F−T grid) |
| `docs/GGUF_MINING.md`, `scripts/gguf_mine.py` | Mining a GGUF teacher. 27 subcommands; §14 covers the embedding geometry ones |
| `docs/reports/*.md` | One dated report per piece of work, with every number |

---

## 4. State at handoff (2026-10-28)

### 4.1 Done this session (all on `main`)

| Item | Result | Report |
|---|---|---|
| B1 trie consolidation | Ten forked tools (9,574 lines) became `packed_trie.hpp` plus `cyphalm_trie` (2,072 lines), with exact parity on every printed number. Subcommands `bpc`, `chat`, `probe`, `context8k`, `sample`, `selftest` | `reports/B1_TRIE_CONSOLIDATION.md` |
| K2/K3 thought interpreter and learners | `native/include/cypha/thought/`; CTest with 5,540 checks. half_split takes 6 tests against 31.3 for random probing; SPRT errors are at their nominal rates; selection regret falls; ECE 0.015; replay is bit-identical | `reports/K2_K3_THOUGHT_INTERPRETER.md` |
| S-1 words as classes | **Negative.** Cypha class means reach count level (WordSim 0.240) and stay far below PPMI-SVD (0.723). There is no is-a signal from variance, and the capital-of map finds 0–12% of capitals. P1–P3 missed. PPMI-SVD stays the vector source | `reports/C5_WORDS_AS_CLASSES.md` |
| Embedding geometry plan | Qwen3-Embedding-8B dissection: 7 static and 22 dynamic analyses, 6 mining jobs with Qwen3.5-9B | `EMBEDDING_GEOMETRY.md` |
| Geometry library | `native/include/cypha/semantics/geometry.hpp`; CTest with 32,096 checks. On a planted teacher, distillation reaches pair Spearman 0.943, against 0.367 for InfoNCE alone. The mask protects paraphrases, and CSLS cuts hub skew from 3.18 to 0.32 | `reports/QE_GEOMETRY_LIBRARY.md` |
| `gguf_mine.py` geometry commands | `embedtexts`, `sensitivity`, `occlusion`, `instruct`, `pairs`, `geometry`, `compare`, `wdiff`, plus `vectors --pooling last` with an EOS check. The selftest passes on two vocabularies, with known-answer maths checks | `GGUF_MINING.md` §14 |
| Thought library additions | `synthetic_pair_mining` and `geometry_distillation` processes | `THOUGHT.md` |
| Answering proposal | Three routes that keep CyphaLM writing every byte | `CYPHALM_ANSWERING.md` |

### 4.2 CyphaLM Q/A baseline (A0) and "don't know" training (A1)

**Done 2026-10-28** ([report](reports/A0_A1_CYPHALM_QA_AND_DONT_KNOW.md)). `cyphalm_trie qa` (the harness) and `cyphalm_trie dktext` (the don't-know training-text builder, with a leakage check).

- **A0: CyphaLM answers none by itself.**
  - **Dev:** 0–2 of 130 right, both lucky words inside nonsense. Coverage at 2% wrong is 0.0% (5-fold CV inside dev).
  - **Frozen test:** 0–2 of 200 raw. The dev gate kept 3 answers, all wrong.
  - **BPC:** 2.344571, unchanged.
- **A1: it learns to write the message, not when to.** 3,963 don't-know rows, with 0 leakage.
  - **Naive text (A1a):** the message comes out at about its training share, indiscriminately. Precision 0.23–0.27 against a base rate of 0.23. **D1 hit.**
  - **Familiarity token (A1b):** CyphaLM follows a token placed after `Assistant: ` exactly. But the familiarity statistic, the model's own mean log-probability of the question, separates not-in-slice from in-slice questions with an AUROC of only 0.54–0.60. Dev precision 0.50, recall 0.43. **D2 missed.**
  - **Tags:** no tag names the question's entity.
  - **BPC:** changes by ≤ 0.0033. **D3 hit.**
- **What this says.**
  - The bottleneck is a better familiarity signal: an entity-level count rather than the whole question's probability, or the R2 experts.
  - CyphaLM needs a way to copy the entity from the question into its answer and its tag.
  - Odin still has to rule on the familiarity token.

### 4.3 Earlier results still in force

| Item | Result |
|---|---|
| E1 tagger | UPOS 94.81% |
| F2 pattern learner | 832 new facts, almost all dates and places |
| C1 PPMI-SVD | WordSim 0.672, SimLex 0.255, MEN 0.637 |
| E4 | Cypha's classifier (75.0) loses to an averaged perceptron (89–91.6) on TREC. Cypha's text learning is an open risk |
| SCRIBE v1 | Passes 3 of 6 ladder pairs |
| ECS / LCI / epistemic libraries | Built, with CTests |

### 4.4 CTests

Run these on every change:
- `native_thought_test`
- `native_geometry_test`
- `native_cyphalm_trie_selftest`
- `native_ecs_lci_test`, `native_epistemic_test` and `native_scribe_selftest`, which predate this session

---

## 5. Decisions waiting for Odin

1. **Answering proposal: ruled 2026-10-28.** R1 + R3 first; teacher text only as a separate low-trust shard; R3 frames from the question parse and learned patterns; the abstain gate stays "for now" (`CYPHALM_ANSWERING.md` §5).
2. **Familiarity control token: ruled 2026-10-28.** Allowed on the gate side; try an entity-level familiarity; token after `Assistant: `.
3. **F2's per-relation base rate.**
4. **When the 1 TB run ends:** move `trie_stream_train.cpp` onto `packed_trie.hpp`.

---

## 6. Next work, in order

### 6.1 Can run anywhere (cloud container or Odin's machine)

1. **A2 and A3** of CYPHALM_ANSWERING, once Odin agrees: question-form fact shards, then statement scoring, in `cyphalm_trie` (`qa` is the harness; `dktext` shows how to build a training shard with a leakage check).
   - In parallel, test a better familiarity signal for A1: the InfiniGram count of the question's entity span in the training text, a statistic of CyphaLM's own model.
   - Add a copy path, so the tag can name the entity.
2. **TOOL_PROFILE P1–P4.**
   - P1: adapters of at most about 200 lines each.
   - P2: the 12-configuration grid with the leakage check, about 6–8 min each on one core.
   - P3 and P4: decisions, then cuts.
3. **S-1b**, the semantics retry SEMANTICS §7 calls for:
   - learned context vectors (two tables) or PMI-weighted inputs;
   - the `geometry.hpp` pull–push rule;
   - in `native/tools/cypha_word_classes.cpp`, the single semantics tool, which grows `train` / `distill` / `eval` / `export` subcommands;
   - the PPMI-SVD baseline on the same pairs.
4. **QE5 without a teacher download.** Model2Vec-style static distillation needs teacher vectors. Until they exist, build the Cypha side: export Cypha vectors as `.npy` plus an index, so `gguf_mine.py compare` can read them.
5. **Bill of work upkeep.** Keep `CYPHA_BILL_OF_WORK.md`, `CHANGELOG.md` and `docs/CURRENT.md` current with every commit.

### 6.2 Needs Odin's machine (Hugging Face and Wikidata are blocked in the cloud container; 2 cores there)

| Step | Command (from the repo root, with Odin's Python having `llama-cpp-python`, `gguf`, `numpy`) |
|---|---|
| Downloads | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0) for the pilot; `Qwen/Qwen3-Embedding-8B-GGUF` (Q8_0, 8.05 GB); Qwen3-0.6B / Qwen3-8B GGUF (base and instruct) for `wdiff`; a community Qwen3.5-9B GGUF (Q4_K_M 5.6 GB or Q8_0 9.5 GB) plus a recent llama.cpp with `--jinja` |
| QE0 gate | `python scripts/gguf_mine.py info --model $E --out mined/qwen3emb/info`, then compare `embedtexts --pooling last` against `llama-embedding -m $E --pooling last` on 20 sentences. The cosine must be at least 0.999. Also check that llama-cpp-python loads Qwen3.5-9B at all |
| Timing | `embedtexts --input probe.txt --limit 1000`, then size the full jobs (the probe set is about 3M tokens) |
| QE1–QE3 | The runbook in GGUF_MINING §14: `geometry`, `sensitivity`, `occlusion`, `instruct`, `wdiff`. Pilot on 0.6B, then 8B. Score predictions Q1–Q6 and Q10 |
| QE4 | `pairs` with Qwen3.5-9B (thinking off; `--think` only for the audit) |
| D4 Wikidata seeds | For F2's Q/A relations. Wikidata is unreachable from the cloud container |
| Results back | Commit the small outputs (`*.json`, `*.tsv`) under `mined/`, or put the large `.npy` files on a GitHub release. Write `docs/reports/QWEN3_EMBEDDING_DISSECTION.md` |

Probe-set inputs (`probe.txt`, `pos_pairs.tsv`, `classes.tsv`, `held_out.txt`, `questions.txt`) do not exist yet. Build them first, from:
- held-out enwik8 (after 8 MiB);
- Dolly;
- GitHub copies of STS-B, MRPC, PAWS and SNLI;
- the E1 chunker for phrases.

Never use T-QA test questions in them.

---

## 7. Environment notes (cloud container)

| Thing | Where |
|---|---|
| Repo | `/home/claude/cypha` |
| Data | `/root/cdata/enwik8` (the first 8 MiB equal enwik9's), `/root/cdata/dolly.jsonl` (15,015 records), `/root/cdata/wiki8m` |
| Build directory (Ninja, Release) | `/root/cb_ecs`: `cmake -S native -B /root/cb_ecs && cmake --build /root/cb_ecs --target <t> -j1 && ctest --test-dir /root/cb_ecs -R <test>` |
| Core library | `/root/cb_core/libcypha_core.a` (used by `cypha_word_classes`; its header has the g++ line) |
| Python with llama-cpp-python 0.3.36 | `/root/mine/venv/bin/python` (system python3 lacks it) |
| Vocab-only GGUFs for the selftest | `/root/mine/llama_cpp_python-0.3.36/vendor/llama.cpp/models/ggml-vocab-{qwen2,llama-spm}.gguf` |
| Word-similarity sets, PPMI-SVD vectors, F2 facts | The session scratchpad (`ev/`, `out/`, `f2/`). Lost if the container is reclaimed; regenerate with the tools in each report |

**Gotchas.**
- **The machine is small:** 2 cores and about 7 GB RAM. Use `nice`, run one heavy job at a time, and give each agent its own files.
- **No network beyond the package registries and GitHub.** Hugging Face, Wikidata and dumps.wikimedia.org are blocked from the shell.
- **The remote URL is the old lowercase `odin-loki/cypha`.** Pushing prints "This repository moved" (to `odin-loki/Cypha`); it is harmless and the push lands.
- **Deletions and globs.** When committing deletions, use `git rm` or `git add -A` on the directory. A shell glob over deleted files matches nothing, which once left B1's deletions out of a commit.
- **The `wip/agents-2026-10-28` branch** holds old snapshots of in-progress agent work. Everything in it is now on `main` in verified form. Odin can delete it (the session token may not delete branches).
