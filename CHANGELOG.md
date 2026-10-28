# Changelog

All notable changes to Cypha are recorded here. The project follows a
**milestone** release model: each entry corresponds to a named engineering
milestone or a significant self-contained change.

---

## [Unreleased]

### Removed
- **`cypha_gate`** and CTest `native_cypha_gate_selftest` (TOOL_PROFILE P4, [report](docs/reports/TOOL_PROFILE_P2_GRID.md)): over ten configurations the Cypha-classifier gate kept 21.7 right and 18.0 wrong answers (mean of 3 seeds) against the threshold gate's 15 and 5, and broke the 2% target in 6 of 30 seed runs. `qa`'s threshold gate stays. Last version: commit `e82a4a4`.
- **`native/tools/scribe_ladder.cpp`** (2026-10-28): replaced by `native/tools/scribe.cpp`, which reproduces v1's ladder numbers to the printed digit with `--profile v1` ([SCRIBE v2 report](docs/reports/SCRIBE_V2.md)).
- **Forked trie tools** (PLAN B1, 2026-10-27, [report](docs/reports/B1_TRIE_CONSOLIDATION.md)): `trie_fold_lm`, `trie_online_mix`, `trie_chat`, `trie_qa_probe`, `trie_context_8k`, `trie_sample` (now `cyphalm_trie` subcommands with identical output) and the retired experiments `trie_together`, `trie_adaptive`, `trie_gria_gate`, `trie_online_holdout` (results and recovery commits in the report). 9,574 lines became 2,072. `trie_stream_train.cpp` is unchanged.
- **Code-debt cuts** (2026-10-27): `native/tools/question_classifier.cpp` (its finding stays in the E4 report; git `707629a`) and `native/tools/surface_controls.cpp` (superseded by `scribe_ladder`; git `ab4f9f6`).
- **Question answering that did not use CyphaLM** (PLAN rule 10, [record](docs/reports/REMOVED_NON_CYPHALM_QA.md)): `copied_answer()` in `trie_chat.cpp` (a string search), `native/tools/qa_infobox.cpp` (hand-written fact lookup) and `native/tools/chat_index.cpp` (BM25 retrieval). Their Q/A numbers are not CyphaLM results. The frozen question sets stay as CyphaLM's test and dev data.

### Changed
- **Language-model defaults unchanged** ([B4B](docs/reports/B4B_LM_DEFAULTS.md)): the class expert (shape, 3 orders) and copy (m 8, window 16,384) take fresh text from 2.166 to 1.912 BPC with surface S 0.907, but with the fact shard they echo the question or let a wrong answer through (rule 3), so they stay options.
- **Odin's rulings, 2026-10-28** ([CYPHALM_ANSWERING.md](docs/CYPHALM_ANSWERING.md) §5; PLAN ledger rows 51–53): R1 question-form fact text and R3 statement scoring go first, with R3 frames from the question parse and the learned patterns; teacher text only as a separate low-trust shard; the familiarity control token is allowed on the gate side, after `Assistant: `, and an entity-level familiarity is to be tried.

### Added
- **A4 similar-context expert** ([report](docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md)): `native/include/cypha/cyphalm/similar_context.hpp` (header-only: int8 PCA keys of the last W words' mean vector, spherical k-means inverted file, neighbours live through the word being written) and `cyphalm_trie cls --sim` / `--sim-fresh` / `--sim-surface`, `qa --sim`.
  - Fresh text, read once: trie + IG + class 2.0879 → 2.0267 (−0.0612 [−0.0658, −0.0572]); surface S unchanged. Meaning carries 44% of the gain (random word vectors give −0.033).
  - Answers: 0 more at 2% wrong on dev, 0 / 200 on the test; with the fact shard it lets look-alike wrong answers through. An option, not a default.
- **A1d / A2b answering variants** ([report](docs/reports/A1D_A2B_ANSWERING_VARIANTS.md)): `qa` adds entity candidates (`a|b|c`, the best by `--fam-stat`), `ent_tb`, `ENTITY` and `FAMTOK` lines, `--fam-train low`, `--cls` together with `--copy`, `--copy-min N`, and `--r3` reading F2R's raw frames. Tried and deleted: `--fam-ent`, `--fam-train matched`, plain-text shard rows.
  - The don't-know token (span candidates, `ent_title`, `low` only): dev precision 0.600 / recall 1.00, frozen test 0.635 / 0.94. With D4's shard the gate keeps 2 right, 0 wrong on dev; test 0 / 0.
  - Class + copy: holdout BPC 2.072 → 1.914 (trie + IG), but with the shard CyphaLM echoes the question and 6 raw right answers fall to 2. Neither is a default for answering.
- **M7 stored packed trie** ([RAM report](docs/reports/RAM_PROFILE.md)): `PackedModel::save` / `load`, a checksummed little-endian file with the model, the mixture weights and the split. The chat model is an 8.1 MB file that loads in 17 ms with bit-identical scores; a serving process peaks at 20.1 MB, against 95.9 MB when it trains.
- **CyphaLM's RAM** ([report](docs/reports/RAM_PROFILE.md)), each change measured against a rule fixed before it:
  - **M4 packed serving form:** `PackedModel`, `pack()` and `Model::release()` in `packed_trie.hpp`, with `fill_experts` and `mix` as one template body for both forms. Bit-identical to the training model in Synth mode; the chat model 38.5 → 7.7 MB; `fill_experts` about 5× faster.
  - **M5 `FmInfiniGram`** (`fm_infinigram.hpp`/`.cpp`): InfiniGram's `query` and `match_prefix` (count, no position) over a bidirectional FM-index in weight-shaped alphabetic wavelet trees. 77 → 26 MB on the qa training text, the same answers; with its `Cursor`, as fast as the suffix array on text and faster on long copies. CTest `native_fm_infinigram_test` (178,635 checks). `save` / `load` store it (M8): 27 MB for the qa text, loaded in 0.06 s with the same answers, so serving never pays the build.
  - **M1–M3 in `class_context.hpp`:** the build sorts and merges counted runs chunk by chunk, each order on its own (identical table; the memory the 3-order build adds at its peak 560 → 132 MB). Options `compact` (32-bit keys, 1-byte counts: table −53%, +0.00002 to +0.00008 BPC) and `min_total` (contexts seen once dropped: 3 orders compact 50.2 → 10.8 MB on the wiki text, +0.0009 BPC, surface S −0.003 [−0.010, +0.003]).
- **Qwen3-Embedding dissection, weights only** ([report](docs/reports/QWEN3_EMBEDDING_DISSECTION.md)): the QE0 gate on 0.6B (hit, cosine ≥ 0.9999999 against `llama-embedding`), the hashed QE2 probe set with its builder (`mined/probe/`), and the QE1 static dissection of the 0.6B and 8B embedders (`mined/qwen3emb/`). Q4 hit on both. Inference moved to Odin's PC on 2026-10-28; the report's §9 has the commands.
  - **Fixed in `gguf_mine.py`:** `--threads` was ignored for batch decodes (19.0 → 58.3 tokens/s, same vectors); `wdiff` on padded vocabularies; `spectra` copied a whole tensor to float64 before sampling.
- **S5 hypothesis store and S3 gap map** ([report](docs/reports/S3_S5_GAP_MAP_AND_HYPOTHESES.md)): `native/include/cypha/eval/hypothesis.hpp` (record, lifecycle, tiers, taint and firewall, promotion through ECS on independent evidence, refutation block, labelled cards) and `gap_map.hpp` (tag parser, CyphaLM gap id, gap map, VOI).
  - `native_epistemic_test` covers I1, I4, I7 and I8: 195 → 516 checks. Four planted bugs, one per invariant, each fail it.
  - `pattern_learn --gaps FILE [--demand ROWS,…] [--parse P] [--gap-floor F] [--demand-weight W]`: 27,842 gaps on the v2 store; the store itself is byte-identical to the run without `--gaps`.
  - CyphaLM's dev "I don't know" tags name their question's slot 0 / 750 (Wilson upper 0.005). Predictions 8 hit, 8 missed.
- **F2R raw-byte patterns** ([report](docs/reports/F2R_RAW_BYTE_PATTERNS.md)): `pattern_learn --raw` learns patterns over the dump's own bytes (markup kept) and exports R3 frames (a `frame` column); `--join` merges another run's facts into the shard store; `--reach` measures frame reach on dev (grading only). Raw: 81 kept patterns, 1,025 facts at 0.950 Wikidata agreement; 41 facts the cleaned run lacks (30/30 hand check) join the store (1,257 at 0.951). Hard relations still get no pattern. Raw frames with the raw-form shard reach the dev answer for 7 of 13 eligible questions (cleaned: 4).
  - A year fact is now written as a year in every mode: 6 rows of the default shard change ("died in November 22, 1963" → "died in 1963"); facts unchanged. The raw mode's link-cut rule was tried in the cleaned mode and rejected (death places 30 → 0).
- **S-1b learned contexts** ([report](docs/reports/S1B_LEARNED_CONTEXTS.md)): `cypha_word_classes` gains `train` (`--method sgns|pullpush`), `distill`, `eval` (paired bootstrap, relations, hypernymy) and `export` (.npy + index, SIF sentences, k-means classes); `s1 --obs ppmi|ppmi-row --save`. Cypha's vectors reach PPMI-SVD's level (WordSim 0.750, SimLex 0.270) but fail both gates, so PPMI-SVD stays the source.
- **SCRIBE v2 and its paper** ([report](docs/reports/SCRIBE_V2.md), [plan with registered predictions](docs/SCRIBE_PAPER_PLAN.md), [paper](paper/scribe/SCRIBE_paper.md)):
  - `native/tools/scribe.cpp`, one tool: `docs`, `prompts`, `samples`, `prepare`, `windows`, `ladder`, `score`, `selftest` (CTest `native_scribe_selftest`, 76 checks). JSONL in and out; enwik ranges, plain-text JSONL and raw JSONL as sources.
  - v2 scores each statistic as a robust z against human reference samples and combines them with one rule: lexical and syntax failures are categorical and saturate, degeneration is the worst failure, coherence is graded. New statistics: context gain of the word and skeleton trigrams over unigram frequency (s3g, s2g), lexical cohesion (c2), adjacent cohesion, verbatim 8-gram copy, and a prompted relevance axis (profile `v2p`).
  - Designed on development ladders only, frozen, then read once on fresh enwik8 [72, 95.37) MiB: **5 of 5 control pairs** (v1 2 of 5, surface S 1 of 5); also 5 of 5 on web text and 3 of 5 on books.
  - `scripts/scribe_eval/`: dataset preparation (WebText and the GPT-2 output dataset, Gutenberg, WritingPrompts, HANNA, OpenMEVA, MAUVE human evaluation, Dolly, UD EWT), model-ladder generation, baseline metrics and an LLM judge, held-out corruptions, attacks, prompted and Q/A ladders, and the meta-evaluation statistics.
  - `cyphalm_trie cls --out FILE` writes its samples and the human continuations for SCRIBE.
- **B4 class-context expert** ([report](docs/reports/B4_CLASS_CONTEXT_EXPERT.md)): `native/include/cypha/cyphalm/class_context.hpp` (200 lines) predicts the next byte from the classes of the last two words plus up to 6 bytes since; `cyphalm_trie cls` (the experiment) and `qa --cls` (the tool-profile grid).
  - On fresh enwik8 text, read once: trie + InfiniGram 2.1660 → 2.0879 BPC with shape classes only, 2.0751 with E1 tags; surface S 0.761 → 0.905 / 0.877 (human 0.870).
  - The tags add −0.0128 BPC [−0.0147, −0.0110] over shape classes but lower surface S by 0.029. Q/A comes with the grid.
- **A1c, A2, A3 in `cyphalm_trie qa`** ([A1c report](docs/reports/A1C_ENTITY_FAMILIARITY_AND_COPY.md), [A2/A3 report](docs/reports/A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md)): options `--parse` (E1 question parse), `--fam-stat`, `--fam-train stat|outcome`, `--fam-only`, `--copy` (copy expert over the current context), `--r3` (statement scoring with F2's patterns as frames), `--shard` / `--shard2` (question-form fact text; holdout pages dropped; exits on a dev or test question), `--leak`; `dktext --mention`.
  - **Familiarity:** the entity's count as a page title gives dev AUROC 0.795 (A1: 0.54–0.60); as a gate token, precision 0.435 / 0.371 at recall ≈ 1.00. D2 still missed.
  - **Copy expert:** holdout BPC trie 2.320 → 2.114, trie + InfiniGram 2.072 → 1.926; an option, not a default (no tag names the entity yet).
  - **A2:** 2 right / 0 wrong out of fold on dev (coverage 1.5%), only on questions the shard covers; test +0. **A3:** 0% at 2% wrong.
- **`cypha_gate`** (TOOL_PROFILE P1, the adapter for the Cypha classifier as abstain gate; DISCOVERY §8a): the Core Cypha classifier learns right vs wrong from CyphaLM's own confidence features in `cyphalm_trie qa --out` rows, with 5-fold CV inside dev and a threshold at 2% wrong. It only keeps or withholds answers CyphaLM wrote. CTest `native_cypha_gate_selftest` (synthetic rows: CV coverage 0.293 at a 2.0% wrong rate). On the A0 rows it answers nothing, as there is nothing right to keep.
- **D4 Wikidata seeds and the fact-text shard** ([report](docs/reports/D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md)):
  - `scripts/wikidata_seeds.py` (180 lines, stdlib only) maps enwik8 titles to Wikidata (98.7%) and fetches relation values as distant supervision. T-QA dev and test subjects are excluded before any query.
  - `pattern_learn` gains `--wikidata`, `--wikidata-gold-only`, `--phi` and `--shard` (question-form fact text with a built-in leakage check), and a seed fix that drops 664 citation-template seeds.
  - Kept patterns 44 → 70; admitted facts 832 → 1,216; agreement with Wikidata 0.951 [0.909, 0.974]. Capital, author and director still get no kept pattern.
- **CyphaLM Q/A harness and don't-know training** (CYPHALM_ANSWERING A0–A1, [report](docs/reports/A0_A1_CYPHALM_QA_AND_DONT_KNOW.md)): `cyphalm_trie qa` and `cyphalm_trie dktext`.
  - **Q/A:** CyphaLM alone has 0.0% coverage at a 2% wrong rate on dev.
  - **Don't know:** trained on its own message and tag, it writes them, but only as well as its familiarity signal allows (AUROC 0.54–0.60). D1 hit, D2 missed, D3 hit (BPC change ≤ 0.0033).
- **Embedding-geometry mining commands** in `scripts/gguf_mine.py` ([GGUF_MINING.md](docs/GGUF_MINING.md) §14), now 4,199 lines:
  - **With a model:** `embedtexts` (last-token pooling with the EOS check, query instructions, MRL prefix, resume, timing), `sensitivity`, `occlusion`, `instruct`, `pairs` (Qwen3.5-style synthetic pairs, hard negatives, verification, mask audit).
  - **Offline:** `geometry`, `compare` and `wdiff`.
  - **Changed:** `vectors` gains `--pooling`.
  - **Selftest:** passes with two vocabularies, with known-answer maths checks.
- **Embedding geometry library** (EMBEDDING_GEOMETRY §4–5, [report](docs/reports/QE_GEOMETRY_LIBRARY.md)): header-only `native/include/cypha/semantics/geometry.hpp`. It provides Gaussian items, the softmax pull–push rule with the false-negative mask and confidence-scaled natural-gradient steps, relational distillation, MRL ordering and nested loss, whitening, CSLS and the agreement metrics.
  - **CTest `native_geometry_test`:** 32,096 checks. On a planted teacher, distillation reaches pair Spearman 0.943 against 0.367 for InfoNCE alone.
  - **CMake:** the `cyphalm_trie` target and the `native_cyphalm_trie_selftest` CTest are added.
- **Words as classes** (SEMANTICS S-1, [report](docs/reports/C5_WORDS_AS_CLASSES.md)): `native/tools/cypha_word_classes.cpp` runs Core Cypha's own class update over 10,000 enwik8 word classes with sampled competitors.
  - **Result, negative:** WordSim / SimLex / MEN 0.240 / 0.134 / 0.372. That is level with a matched count baseline and far below PPMI-SVD (0.723 / 0.265 / 0.732). Hypernym direction is below the frequency baseline, and the capital-of map reaches 0–12% in the top 10. P1–P3 missed.
  - **What stays:** PPMI-SVD stays the vector source.
- **Answering proposal** ([`docs/CYPHALM_ANSWERING.md`](docs/CYPHALM_ANSWERING.md)): how CyphaLM can answer questions itself under rule 10. Three routes: question-form training text, generalising experts, and answering by scoring the declarative statement. Steps A0–A5 with predictions. Awaiting Odin's decision.
- **One packed-trie library and tool** (PLAN B1): `native/include/cypha/cyphalm/packed_trie.hpp` (header-only, `cypha::cyphalm::trie`, paths and constants as parameters) and `native/tools/cyphalm_trie.cpp` with `bpc`, `chat`, `probe`, `context8k`, `sample` and `selftest` (tiny-corpus golden BPC, for the CTest `native_cyphalm_trie_selftest`; the CMake block is in the report).
- **Two teacher-geometry processes** in the thought library: `synthetic_pair_mining` (generate, verify, filter, cross-check, audit the mask, admit as hypotheses) and `geometry_distillation` (teacher-minus-Cypha forces, mask, natural-gradient step, gate by LCI). The library now has 144 processes and 856 steps.
- **Embedding geometry plan** ([`docs/EMBEDDING_GEOMETRY.md`](docs/EMBEDDING_GEOMETRY.md), Track C): make Cypha's vectors attract and repel like Qwen3-Embedding-8B's without a neural net at runtime.
  - **Mechanics:** the teacher's verified training (three stages, InfoNCE with hard negatives, the +0.1 false-negative mask, last-token pooling, MRL, slerp merging) written as forces.
  - **Cypha's version:** a softmax pull–push rule in Fisher geometry with confidence-scaled steps; relational distillation where each pair's force is p_teacher − p_Cypha.
  - **Dissection:** 7 static and 22 dynamic analyses, including the weight change from the base model; 6 mining jobs for Qwen3.5-9B.
  - **Measures and plan:** geometry agreement, tasks and cost; work items QE0–QE9 with 10 predictions registered in advance.
- **Thought interpreter and learners** (THOUGHT K2–K3, [report](docs/reports/K2_K3_THOUGHT_INTERPRETER.md)): header-only `native/include/cypha/thought/{learners,interpreter}.hpp` run the process library (`docs/thought/processes.json`, written by `thought_check.py --json`). Portable Beta, Dirichlet, Welford, Thompson selection, ACT-R utility, TD(λ), VOC stop, calibration log. CTest `native_thought_test`: 5,534 checks; half_split 6 tests against 31.3 for random probing; SPRT errors 0.028 / 0.038 at 0.05; selection regret 243 → 92.5; bit-identical replay.
- **Semantics design** ([`docs/SEMANTICS.md`](docs/SEMANTICS.md), Track C): Core Cypha reinterpreted as a statistical meaning layer.
  - **How it works:** words and concepts become classes and contexts become observations. That gives online embeddings, Gaussian variance for breadth and is-a direction, relation maps with abstention, and sense splitting.
  - **Into CyphaLM:** through class-context and similar-context experts.
  - **From a teacher:** it lists what to extract from an LLM.
  - **Experiments:** S-1 to S-6, with predictions registered in advance.
- **"Don't know" message and tag** (PLAN rule 9, DISCOVERY §3.5): when Cypha doesn't know, CyphaLM writes a plain message and a machine tag (`<|unknown reason=... gap=...|>`). It is trained to produce both itself. The tag logs the gap for later discovery. This is "doesn't know", not a refusal.
- **Tool profiling plan** ([`docs/TOOL_PROFILE_PLAN.md`](docs/TOOL_PROFILE_PLAN.md)):
  - It decides which support tools stay (tagger, pattern learner, word vectors, teacher text, InfiniGram, Cypha abstain gate), by what each adds to CyphaLM's own Q/A and BPC.
  - It uses an ablation grid with LCI, a lines-per-gain bar and predictions registered in advance.
- **Pattern learning** (PLAN F2, [report](docs/reports/F2_PATTERN_LEARNING.md)): `native/tools/pattern_learn.cpp` learns relation patterns from infobox seeds by distant supervision, keeps those with a Wilson precision bound ≥ 0.7, and extracts ECS-scored facts.
  - On enwik8: 832 new admitted facts (mostly dates and birth places), at 49/49 verified precision, in 16 s.
- **Question classifier** (PLAN E4, [report](docs/reports/E4_QUESTION_CLASSIFIER.md)): `native/tools/question_classifier.cpp` on TREC. The averaged perceptron scores 89.0–91.6 coarse and 84.6 fine; Cypha's classifier 75.0 and 62–67. This confirms the PLAN §11 risk, so the perceptron is used for question types and Cypha's text learning is an open item.
- **Part-of-speech tagger and chunker** (PLAN E1, [report](docs/reports/E1_POS_TAGGER.md)): `native/tools/pos_tagger.cpp`, an averaged perceptron on UD English EWT v2.18.
  - Test UPOS 94.81%, passing the 94% gate; about 1M tokens/s.
  - Rule-based noun-phrase chunker, F1 83.8 on test.
  - Question-parse mode: wh-word, auxiliary, main verb, entity.
- **AI and data-science processes** in the thought library ([`docs/THOUGHT.md`](docs/THOUGHT.md) §3.9): 36 more processes, for 142 in total.
  - 17 data science: CRISP-DM, EDA, cleaning, quality scorecard, tests, A/B, regression, time series, clustering, anomalies, observational causal effects, honest charts, features, sampling, record linkage, near-duplicates, ETL.
  - 19 AI/ML: workflow, cross-validation, successive-halving search, error analysis, evaluation, ablation, leakage, drift (PSI/KS/CUSUM), active learning, safe online update, model selection, retrieval, teacher queries, benchmark hygiene, tracking, slice audit, bandits, model cards.
  - A `daily_upkeep` routine for Cypha itself.
- **Thought-process library** (PLAN Track K, [`docs/THOUGHT.md`](docs/THOUGHT.md)):
  - 106 human thinking and work processes in 23 categories (`docs/thought/processes.yaml`), each a control graph with its source and core maths.
  - The model is a learnable option with Beta, Dirichlet and Welford learners, DIF applicability, Thompson selection, TD(λ) credit, a value-of-computation stop, fast/slow arbitration, chunking, process mining and MDL edits promoted through LCI.
  - `scripts/thought_check.py` validates the library, writes `docs/thought/INDEX.md`, and exports any process as PlantUML or Mermaid text.
- **SCRIBE v1** ([report](docs/reports/SCRIBE_V1.md)): real packed-trie samples (`native/tools/trie_sample.cpp`), SIF sentence vectors, an I gate, length parity. On a fresh ladder it passes 3 of 6 adjacent pairs (v0 passed 1); surface S passes 2. Human text beats trie output in 99.3% of pairs.
- **Epistemic layer and chat-index reason codes** (DISCOVERY S1/S2, [report](docs/reports/S1_S2_EPISTEMIC.md)): header-only `epistemic.hpp` covers states, degree of conflict, uncertainty split, 11 reason codes, Good–Turing, Chao1, Chapman, Heaps and the four card formats; CTest `native_epistemic_test`, 195 checks. `chat_index` explains every refusal with a reason code, and answers in 1.3 s instead of 53 s.
- **SCRIBE v0** (PLAN §6, [report](docs/reports/SCRIBE_V0.md)): header-only `native/include/cypha/eval/scribe.hpp` covers the open-text axes S, C, I and E, with human-reference normalisation. `native/tools/scribe_ladder.cpp` runs the V1 control ladder; CTest `native_scribe_selftest`.
  - It fails the ladder: 1 of 6 adjacent pairs pass, as does surface S. Recorded as a negative result, with the fixes.
- **Count-based word vectors** (PLAN C1, [report](docs/reports/C1_PPMI_SVD_VECTORS.md)): `native/tools/ppmi_svd.cpp` builds PPMI with smoothed contexts and a home-built randomised SVD, plus random indexing, from enwik8. No neural network is involved. Scores: WordSim-353 0.672, SimLex-999 0.255, MEN 0.637. This is the baseline the other Cypha encoders must beat.
- **T-QA-wiki, 200 blind questions** (`bench/eval/t_qa_wiki.tsv`, [report](docs/reports/T_QA_WIKI_200.md)): 150 answerable from the training slice and 50 not.
  - `qa_infobox`: 3 right / 197 abstained / 0 wrong. The 8/8 seed score was in-sample.
  - `chat_index`: 2 right / 194 abstained / 4 wrong.
- **Discovery design** ([`docs/DISCOVERY.md`](docs/DISCOVERY.md), PLAN Track S):
  - answer mode and discovery mode behind a firewall;
  - epistemic states and the two kinds of uncertainty;
  - a competence self-model;
  - reason codes for every "I don't know";
  - a gap map, with estimates of the unseen (Good–Turing, Chao1, capture–recapture);
  - 20 hypothesis generators;
  - scoring by novelty, impact, compression gain and expected information gain;
  - 10 test kinds with pre-registration;
  - 16 discovery processes, a curiosity scheduler, possible-worlds reasoning, 10 honesty invariants, benchmarks and work items S1–S15.
- **Fact-store Q/A** (PLAN F1, [report](docs/reports/F1_INFOBOX_QA.md)): `native/tools/qa_infobox.cpp` extracts infobox fields and a few sentence patterns from the training slice into 141 facts. A question classifier v0 answers from them or abstains.
  - 8/8 on the seed questions (in-sample), up from the trie's 0/8.
  - Fact precision 0.90, with a Wilson lower bound of 0.744.
  - Seed suite at `bench/eval/t_qa_wiki_seed.tsv`.
- **Fuzzy chat index** (PLAN B7, [report](docs/reports/B7_CHAT_INDEX.md)): `native/tools/chat_index.cpp`, a BM25 + character 3-gram retrieval index over Dolly that returns a stored answer or abstains.
  - Every perturbed form (lowercased, " Thanks.", no punctuation, synonym) keeps ≥ 0.99 of verbatim accuracy with no abstain, and 0.957–0.964 with abstain on.
  - It abstains on 92.6% of held-out questions.
  - Query p50 0.34 ms, 28 MB index.
- **ECS and LCI library** (PLAN M1, [report](docs/reports/M1_ECS_LCI_LIBRARY.md)): header-only `native/include/cypha/eval/ecs.hpp` and `lci.hpp`. Covers evidence fusion, grades, calibration, the Bayesian-bootstrap learning gain, canaries, e-process with alpha spending, Benjamini–Hochberg, Thresholdout, BWT and learning-curve fit. CTest `native_ecs_lci_test` runs 242 known-answer checks.
- **GGUF mining** ([`docs/GGUF_MINING.md`](docs/GGUF_MINING.md), [`scripts/gguf_mine.py`](scripts/gguf_mine.py)): PLAN Track H.
  - The full list of what can be statistically extracted from a neural model in GGUF format, with references, status and how each output feeds the data pack and ECS.
  - A tool with 18 subcommands:
    - **static:** tokenizer priors and morphology; embedding neighbours, clusters, Brown-style bit paths, PCA, relation offsets and direct-path bigrams; FFN neuron read/write vocabularies; attention-head copy scores; weight spectra with power-law alpha;
    - **dynamic, through llama-cpp-python:** Wikidata relation probing over 52 relations with paraphrase, sample, verification and inverse statistics; an LMCRAWL-style crawl; lexical, taxonomy and commonsense relations; question paraphrases; Q/A; synthetic corpora with text statistics; corpus bits per byte; model-pooled vectors and word senses; calibration into ECS priors; ID linking.
  - The self-test builds a tiny random GGUF and passes all 18 subcommands on SPM and BPE vocabularies.
- **Default plan** ([`docs/PLAN.md`](docs/PLAN.md), 2026-10-26):
  - Language understanding from statistics plus a data pack (Wikidata, Open English WordNet, Wiktionary, ConceptNet, OpenCyc upper ontology, UD treebank), with the Cypha classifier making the decisions and learning online through rewards. No neural network at runtime.
  - Self-improvement is promoted only through a referee.
  - Defines **SCRIBE** (generated-text metric, six axes, normalised against human text, with control ladders it must pass), **ECS** (evidence certainty: subjective-logic fusion of source trust, extraction confidence, independence, time and derivation) and **LCI** (learning certainty: paired Bayesian gain, canary non-inferiority, generalisation ratio, calibration, stability, cost, e-value multiplicity control, reusable holdout).
  - The 1 TB trie run becomes Track A, with predictions registered in advance.
- **Question-answering and count-context test** ([report](docs/reports/CYPHALM_QA_AND_CONTEXT_TEST.md)). New tools `native/tools/trie_qa_probe.cpp`, `trie_context_8k.cpp` and `surface_controls.cpp` reuse `trie_chat.cpp`'s model and take paths as arguments. Findings:
  - The chat "answers" are a string search for the whole prompt: 975/975 verbatim, 0/975 with one letter lowercased.
  - An order-8192 count context takes BPC from 2.32 to 2.06, all of it by 16 bytes (median match 10 bytes). It answers 0/8 Wikipedia facts and 1/30 held-out questions.
  - Surface `S` rates random corpus words (0.918) above the trie's sample (0.775).
- **Corpus fetch** (`scripts/fetch_corpus.py`, `scripts/run_fetch.sh`): stoppable download toward about 1 TB on `D:\cypha_llm\corpus`. Books (Gutenberg, Wikisource, Wikibooks), code (RedPajama GitHub), articles (C4, RedPajama Common Crawl, English Wikipedia, a capped Common Crawl WET sample), papers (RedPajama arXiv), structured (Stack Exchange), labelled (Dolly, OpenAssistant, AG News, SQuAD, IMDB). The RedPajama book slice is excluded. Progress bar and time left. `--check` probes the seed URLs.
- **Language model decision:** the packed trie is the model to train. hp gate24 stays as the comparison (~1.61 BPC, ~1.5 GB) and is not the terabyte train. Plan: [`docs/CURRENT.md`](docs/CURRENT.md).
- **CyphaLM winner v2 and the ∞-gram as a cache** ([report](docs/reports/CYPHALM_LM_QUALITY_REPORT.md), "Winner v2"): occupancy fold (`--fold-auto`), upstream CompressionAlgorithm mixer gains and context models (config fields, checkpoint v4/v5), a byte-LSTM neural expert (`--neural`, bf16, 0.66 ms/byte), libsais just-in-time ∞-gram index (no stored index), `cyphalm_trace` (corpus attribution / contamination), session cache (`--session-cache`). Default winner: 4 shards, wiki / Alice / lcet10 1.6192 / 2.0191 / 1.5160 at 1.0 GB private RAM.
- **CyphaLM as a language model** ([`docs/reports/CYPHALM_LM_QUALITY_REPORT.md`](docs/reports/CYPHALM_LM_QUALITY_REPORT.md), Reference appendix for every tool, flag, config field, env var and file format): full next-byte distributions scored on held-out text; frozen scoring; exact greedy, min-p, word lookahead; serve-time ensembles of shard models with learned weights and threaded scoring (`cyphalm_shard_train`, ensemble manifests); ∞-gram suffix-array expert (`cyphalm_infinigram_build`, IGR2); packed 16-bit context slots (checkpoint v3), mmap loading, serve-time table folding and model drops (`slim` tier); bit-tree pruning (~5× faster distributions). Winner manifests in `models/cyphalm_winner/` (held-out wiki 1.6516 bits/byte), rebuilt by `scripts/build_cyphalm_winner.sh`. `cyphalm_lm_quality --dump-dist` writes the full distributions.
- **CyphaLM lossy tiers** (`apply_hp_lossy_tier`, env `CYPHA_HP_LOSSY_TIER`): `lean`, `balanced`, `compact`, `small`, `tiny`, built from new runtime `hp::Config` knobs (context-model drop, table caps for context/match/discovery-pool models, pool slot count, mixer weight-set drop, mixer update skip). Measured on enwik8 8 MiB against gate24's 1.611729 bpc / 1,538 MB: `lean` **1.609866 / 1,078 MB** (better and 30% smaller), `balanced` 1.612457 / 814 MB, `compact` 1.617400 / 670 MB, `small` 1.629798 / 404 MB, `tiny` 1.652317 / 253 MB. Knobs are saved in the checkpoint JSON. `cyphalm_lossy_bench --tiers`. Report: [`docs/reports/CYPHALM_LOSSY_MIXER_REPORT.md`](docs/reports/CYPHALM_LOSSY_MIXER_REPORT.md).

### Changed
- **Vendored hp is gate24-only:** all 289 ablation flags resolved into the source and the rejected branches deleted (hp include + main: 15,706 → 6,687 lines). Predictions and archives are bit-identical; `HpFlags.cmake` no longer parses `v78_flags.ps1`. Lab-only tooling removed. [`docs/reports/CYPHALM_HP_GATE24_STRIP.md`](docs/reports/CYPHALM_HP_GATE24_STRIP.md).
- **Bit-tree serve DFS:** one `predict()` per node and no update at leaves (382 + 254 calls instead of 510 + 510). Undo log appends without the O(n²) duplicate scan.
- **hp tables are demand-zero again** (`mmap` + `MADV_HUGEPAGE`; `calloc` on Windows): gate24 constructs in ~55 ms instead of ~8 s, and RSS grows with use. Observe is 1.16× faster than v2.5.0 (huge pages), `lean` 1.55×; full-vocab `predict_next` at mem 22 208 ms → 8 ms. Prediction-identical.

### Fixed
- **Serve scoring leaked into the live model.** `DmcModel::update` and the `WordMatchModel` mismatch reset were not undo-recorded, so each `serve_predict_next` could change later predictions (up to 0.18 nats) and the served distribution differed from fresh-clone scoring (up to 0.27 nats). Now exact; `hp_bit_tree_smoke` checks it on text.
- **ODR split between `cypha_core` and tools:** the v78 flags were `PRIVATE` to `cypha_core`, so tools including hp headers compiled a different `hp::Predictor` (64 vs 115 experts). Gone with the gate24-only tree.
- **`hp_lossy_mem` was not saved** in the checkpoint JSON, so a lossy-mem model reloaded at mem 22 against mismatched `.hpbin` tables.

## [2.5.0] — 2026-09-20 · Cross-platform release + CyphaLM hp

### Added
- **macOS arm64 release packages:** `scripts/package_release_macos.sh` + `packaging/install_release_macos.sh`; GitHub Actions `release.yml` `build-macos` job publishes `cypha-*-macos-arm64.tar.gz` alongside Linux and Windows.
- **CyphaLM eval harnesses (honest numbers):** `cyphalm_generation_harness` + `scripts/cyphalm_generation_harness.sh` (greedy/temperature qualitative samples); `scripts/cyphalm_hp_shard_holdout.sh` (fair holdout shard-merge BPC); `scripts/cyphalm_lossy_enwik_screen.sh` (full enwik8MB mem20 vs mem22). Reports: `docs/reports/CYPHALM_GENERATION_HARNESS.{json,md}`, `CYPHALM_SHARD_HOLDOUT.json`, refreshed `CYPHALM_LOSSY_ENWIK_SCREEN.json`.
- **hp shard-merge holdout improvements:** `ShardMergeOptions` confidence gating, `BoundaryReplayConfig` with distance-weighted replay and full-train bridge fine-tune; `prepare_merged_predictor_for_holdout`; holdout ablation mode in `cyphalm_hp_shard_spike`. Measured: full-train bridge closes 57–63% of holdout gap vs PR #15 baseline (still +0.02–0.06 BPC vs single-stream). Report: `docs/reports/CYPHALM_SHARD_HOLDOUT.md`.
- **hp delta undo API:** `hp/undo.hpp` with `UndoRecorderScope`, patch-based `UndoFrame`, `PredictorUndoStack`, and `Predictor::update_tracked`. Bit-tree inference records table/counter mutations only; speculative scoring skips `end_of_byte` stream updates (log-prob parity with legacy clones). Smokes: `hp_undo_smoke`, `hp_bit_tree_smoke`, `hp_inference_bench`.
- **CyphaLM serve / generation surface:** explicit `serve_predict_next` / `serve_advance` on `CyphaLMModel`; `serve_*` helpers on `HpSequenceBackend`; `cyphalm_generate` CLI; `cyphalm_serve_smoke` CTest.
- **CyphaLM beam + nucleus CLI:** byte-level `generate_beam` (pruned bit-tree expansions, predictor replay); `cyphalm_generate` flags `--beam`, `--top-p`, `--temperature`, `--max-bytes`; `cyphalm_generate_smoke` CTest; sample outputs in `docs/reports/CYPHALM_GENERATION_SAMPLES.md`.
- **CyphaLM cold-start generation levers:** serve-time warmup (`--warmup-file` / `--warmup-bytes`), repetition penalty / `ban_last_k`, optional `text_like_prior`; `cyphalm_generation_harness` cold vs primed before/after samples + `decode_ms` in `docs/reports/CYPHALM_GENERATION_HARNESS.md`.
- **Docs:** `MODEL_CARD.md` + `README.md` CyphaLM sections — hp framed as byte LLM (train/serve, generation, ~1.61 gate24 BPC, inference-first).
- **hp LLM integration:** Vendored `native/third_party/hp/` from [odin-loki/CompressionAlgorithm](https://github.com/odin-loki/CompressionAlgorithm). `CyphaLMModel` delegates to `HpSequenceBackend` / `hp::Predictor`. New tests: `hp_llm_smoke`, `hp_roundtrip_smoke`. CMake: `cmake/HpIntegration.cmake`, optional `-DCYPHA_HP_XSIMD=OFF` on non-SSE hosts.

### Changed
- **Sole public GitHub Release:** `v2.5.0` is the only retained release (Linux + Windows + macOS arm64); older tags/releases retired after publish.
- **CyphaLM hp profile:** Single **gate24** build only (v78_flags.ps1 + `HP_SLOT_MAX=24`). Default CMake no longer exposes `CYPHA_HP_PROFILE=light|champ`. See [`docs/history/REMOVED_HP_SKUS.md`](docs/history/REMOVED_HP_SKUS.md).
- **CyphaLM inference (gate24):** `next_byte_log_probs` / `predict_next` use MSB **bit-tree** with delta undo on the live `pred_` (single-predictor serve — no standing `scratch_` twin). Speculative scoring skips `end_of_byte` stream side effects (log-prob parity with legacy per-byte clones). Legacy 256× clone path: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`. Greedy generation uses O(8) `serve_greedy_next_byte`; temperature/top-k sampling via `serve_sample_next_byte`. **Measured** on Linux VM (`hp_inference_bench`, 64 KB warm `consume_byte`, `--iters 5`): @ `table_bits=22` **163664 µs/call** `predict_next`, VmRSS after construct **1564100 kB** (~1.5 GiB). Prior checkpoint tree @ tb22: **~48×10⁶ µs**, **~4.3 GB RSS** (PR #7 pre-undo baseline) — **~293×** faster, **~−64%** construct RSS vs 4.3 GB class. Report: [`docs/reports/GATE24_POST_UNDO_BENCH.md`](docs/reports/GATE24_POST_UNDO_BENCH.md).
- **Production sequence algorithm:** Hybrid GRIA+LSTM is **no longer** the default LLM path. `ContextMode::Hp` + `apply_hp_production_recipe()` replace `apply_hybrid_production_recipe()` for sequence duties. `hybrid` CLI/profile aliases map to hp.
- **CI tiers:** PR gate (Linux + macOS) runs `scripts/ci_native_hp_smoke.sh` (hp/CyphaLM smokes + goldens only). Bench/lock/forecast/d21–d76 smokes labeled `cypha_slow`. Optional `Native slow CTest` job runs `ci_native_fast.sh`. Full gate: `scripts/ci_native_linux.sh`.
- **Legacy CyphaLM tools:** Pre-hp GRIA/LSTM/SSM regression targets gated behind `-DCYPHA_BUILD_LEGACY_CYPHALM=ON` (default OFF) so default `cmake --build` and `ctest -R native_` succeed.
- **Docs:** `docs/native/CYPHALM_TIER2_MODEL.md`, `MODEL_CARD.md`, `README.md` updated; Cypha BPC vs hp archive metrics explicitly separated. Removal narratives: `docs/history/REMOVED_RPSM.md`, `docs/history/LEGACY_LLM.md`.
- **Checkpoints:** `save_cyphalm_model` persists hp config; predictor tables are session-local (online adaptation).
- **Paper camera-ready:** `paper/CyphaLM_paper.md` + arXiv bundle HTML/PDF regenerated to cite living Hybrid **2.664 BPC** (L1 2.873 historical) — **superseded by hp for LLM; paper text not yet regenerated.**
- **Cell-sweep B2 / H06:** 300k / eval 2k rerun after CTest clobber; both **3.681 BPC** (math-integration) — not a promote. Archived under `data/archive/cell_sweep/`.

### Removed
- **`native_navigation_loss_char_lstm_smoke`:** legacy CharLSTM math-integration smoke (broken under the hp default path); dropped executable + CTest from the optional slow suite.
- **RPSM (full retirement):** All RPSM sources, headers, CTests, CMake targets, bench profiles (`cyphalm_d21_rpsm*.json`, `d21_rpsm_profile.json`), and `scripts/run_rpsm_overnight.ps1`. Retired `ContextMode::Rpsm`, `BenchMode::Rpsm`, `rpsm_*` config fields, and `CYPHA_USE_RPSM_LLR` / `rpsm_score_matrix_batched` in `infer_cpu`. d21 lock section `rpsm_results` now records `mode=hp`; schema key kept for compat. **Why:** unused; superseded by hp CyphaLM. **History:** [`docs/history/REMOVED_RPSM.md`](docs/history/REMOVED_RPSM.md); git history on `main`/PR #1 retains pre-removal code.
- **RPSM CTests (representative):** `native_rpsm_sequence_smoke`, `native_rpsm_batched_llr_smoke`, `native_cyphalm_bench_rpsm_smoke`, `native_d21_rpsm_smoke`, `native_score_matrix_parallel_parity`, `native_rpsm_*` regression suite — removed from default build/CI.
- **Default build:** Legacy cyphalm component sources (GRIA, LSTM, SSM, compressive_memory, etc.) excluded from `cypha_core`; headers remain for reference. `kernel_llm_h04_smoke` replaced by `hp_llm_smoke`. **Gated** (not deleted): `-DCYPHA_BUILD_LEGACY_CYPHALM=ON` rebuilds hybrid tools — see [`docs/history/LEGACY_LLM.md`](docs/history/LEGACY_LLM.md).
- **hp SKU matrix (light / champ):** Removed `CYPHA_HP_PROFILE`, champ/light CMake branches, `apply_hp_champ_recipe` / `apply_hp_gate24_recipe`, `profiles/cyphalm_hp_champ.json`. Quality path is gate24 only (~1.6117 BPC enwik observe). See [`docs/history/REMOVED_HP_SKUS.md`](docs/history/REMOVED_HP_SKUS.md).

## [2.4.0] — 2026-08-16 · Competition lock + forecasting

### Added
- **Competition housekeeping:** root [`LICENSE`](LICENSE) (CC BY 4.0) and [`MODEL_CARD.md`](MODEL_CARD.md); living docs aligned to 2.664 BPC / 36 cell-sweep variants. Closed BoW / optimality / forecasting design moved under `docs/archive/`; root BoW is open items only. Forecast/codec sources rewritten UTF-8 (OneDrive UTF-16 broke GCC/Linux CI); `.gitattributes` pins `working-tree-encoding=UTF-8`. Historical XOR/D17 traces moved from `artifacts/profiles/` to [`data/archive/`](data/README.md); repo-root `d*_smoke.json` leaks removed. **GPU training closed** (CPU is faster); future CPU SIMD via xsimd in [`docs/FUTURE.md`](docs/FUTURE.md) §1b.
- **Cell-sweep resume / checkpoints:** `cypha_cell_hypothesis_sweep --resume` plus `scripts/resume_cell_sweep.ps1`, `scripts/wait_cell_sweep_and_lock.ps1`, `scripts/run_cell_sweep_parallel.ps1`. **36** runnable variants (B0–B2, H01–H23, U01–U10). Lock `cell_sweep_results.status=historical` (July, 25 variants, H19 @ 2.921); living spine is Hybrid **2.664**, not the cell-sweep winner.
- **Latent RFF XOR default:** D03 / kernel path defaults to latent features + RFF kernel LLR (`rff_dim=4096`, ~76% vs sklearn ~79%). Legacy `xor_pair` via `CYPHA_D03_KERNEL_FEATURE_MODE=xor_pair`.
- **MSVC C++ workload helper:** `scripts/install_msvc_cpp.ps1` installs the Visual Studio Native Desktop workload when STL headers are missing.
- **GDELT live monitor:** `GdeltCsvTail`, `forecast_gdelt_poll`, REST `POST /forecast/ingest` + `/forecast/monitor/reset`.
- **SORF gate alignment:** `orf_encoder_bench` uses D01 golden linear-sep draws (400×10, seed 42). `bench/BASELINE_LOCK.json` updated to **2.664 BPC** @300k (L2 + Adam BPTT=8).
- **VIEWS leaderboard baselines:** Conflictology mean, observed Markov, NegBin GLMM + `views_leaderboard_bench` tool.
- **Forecast data script:** `scripts/fetch_forecast_data.ps1` materializes `gml_mid.csv` / `gdelt_events.csv` / `views_bulk.csv` (GDELT + CoW MID + VIEWS API bulk).
- **Forecast bench + bulk loaders:** `cypha_bench_run --domain-tag forecast`; CSV loaders for GML MID, GDELT, VIEWS (`split` column); `validate_views_holdout`; expanded `gml_mid.csv` sample.
- **Predictive arithmetic coding (LLMZip-style):** 32-bit range coder + `predictive_codec` on `predict_next`; `Cypha::{compress,decompress}_tokens`, `generate_via_bits`; REST `POST /sequence/compress` + `/sequence/decompress`. CTests `native_predictive_codec_smoke`, `native_predictive_codec_bench_smoke`.
- **Hybrid production recipe:** `apply_hybrid_production_recipe` (GRIA+LSTM, ngram fuse-split, no count prior).
- **Event-forecasting framework (Phases 1–9):** `native/include/cypha/forecast/` — CAMEO token vocabulary, MID node estimators, CyphaLM rollout tree, VIEWS CRPS scoring, GDELT drift monitor; tools `forecast_smoke`, `cypha_forecast_run`, `orf_encoder_bench`, `forecast_drift_backtest`; REST `POST /forecast/run`; sample data under `bench/data/forecast/`.
- **Orthogonal RFF (ORF/SORF):** `RffProjectionKind::Orf`, `OrthogonalRffEncoder`, `PreprocessorState::rff_orf`; `active_query_score_from_probs`, `Cypha::drift_score()`.

### Changed
- **VIEWS bulk path resolution:** bench/REST/leaderboard prefer `views_bulk.csv` over `sample_views.csv` when bulk fetch has run (`resolve_views_csv_path`).
- **SORF promotion gate:** `orf_encoder_bench` aligned with D01 golden split/training; `promote=sorf` when XOR +5pp and D01 +30pp lift vs iid. `apply_hybrid_production_recipe` now sets L2 + Wave2 BPTT (Adam, bptt=8, lr=0.001) matching the 2.664 BPC lock. `everyday_profile.json` enables `rff_sorf: true` for tabular RFF (SORF wins XOR smoke without breaking preprocessor goldens).
- **Living sequence default → Hybrid L2+Wave2 BPTT (2.664 BPC):** `Cypha::init_default_sequence` and `cyphalm_bench_native` default `--mode hybrid` (U06 PGM→Wy opt-in). Docs / D17 profile `_meta.status=production` updated.

### Fixed
- **MSVC / overnight script pathing:** `install_msvc_cpp.ps1` for missing STL headers; `run_d17_overnight.ps1` cell-sweep banner reports **36** variants (was 28).
- **Overnight scripts with external build dir:** `Resolve-NativeBuildDir` / `Resolve-NativeExePath` fix production overnight when `BuildDir` is absolute (e.g. `%LOCALAPPDATA%\cypha_native_build`); all overnight/finalize/poll scripts default to `Get-DefaultNativeBuildDir`. restored locked accuracy / BPC / routing numbers after MSVC and hybrid-default drift -- portable Fisher-Yates shuffle (`portable_shuffle.hpp`), golden D01 synthetic draws (`d01_synthetic_golden.inc`), raw-feature logistic baselines, **opt-in** n-gram count prior (off unless enabled), and native D16A ARI. Pins: D01 linear **0.9875** vs logistic **0.8875**; D04 ~**4.14 BPC** @ 8k; D16A **ARI = 1.0**.
- **D17 hybrid pin re-verified (prior L1 lock):** WikiText-2 300k / eval 2k / seed 42 / `--mode hybrid` → **2.883 BPC** vs archived L1 lock **2.873** (Δ=+0.010 within ±0.05). Living production lock is Hybrid L2+Wave2 BPTT **2.664 BPC**. Scale path (L1-era): 5k≈4.05, 40k≈3.42, 300k≈2.88.
- **`native_rest_schema_contract` (Linux):** `ChildProcess` is now move-only so assigning the spawned `cypha_rest` no longer kills the server before `/health` (was flaky `server did not become ready` at ~15s).
- **`cypha_cell_hypothesis_sweep`:** `--output-dir` now writes `variant_*.json` / `summary.csv` even for single `--cell-variant` runs (previously overnight-sweep only).

### Added
- **D10A ECG5000 >90%:** enriched temporal features + 44 passes -> **90.11%** default (`CYPHA_D10_ECG_ENRICH=0` keeps legacy 85.96%).
- **Paper PDF:** `paper/arxiv_bundle/CyphaLM_paper.pdf` (pandoc + XeLaTeX).

### Changed
- **Docs archive:** historical dated reports / CyphaLM studies / mega-plans moved under `docs/archive/`; living hub rewritten for One Cypha (`cypha::Cypha`, PGM->Wy default, hybrid **2.873** historical pin only).
- **Windows toolchain / CI:** blocking `mingw_cross` job replaced by `windows_msvc` (native MSVC Release on `windows-latest`). Release zip is packaged via `scripts/package_release_windows.ps1` from an MSVC build; MinGW cross-compile is no longer a release or CI gate.

### Removed
- **Legacy sigmoid world gate** (`gh_chi <= 0` / `gh_psi <= 0` fallback in `classify_at_h` and `world_gate_vector_use_field`). GH–NIG gate now required (`gh_chi > 0`, `gh_psi > 0`); non-positive hyperparameters throw. **`kInferWorldGateApiVersion`** bumped to **2** (`native/include/cypha/infer_cpu.hpp`). Unused in-tree; all native callers already pass `gh_chi=gh_psi=1`.
- **`Cypha Possible Upgrades/`** root folder — content backported to [`docs/research/upgrades/`](docs/research/upgrades/README.md).
- **Python runtime decommissioned (P7):** `Cypha.py`, `cypha_studio/`, `cypha_core/`, `cypha_accel/`, `bench/` (Python package removed), `cypha_lm/`, and related packages removed from the product path. Native C++ (`cypha_core` library, `cypha_rest`, `cypha_qt_shell`, `cypha_bench_run`) is the sole runtime.
- **pytest CI gate:** ~274 pytest tests no longer run in CI; validation is CTest-only.
- **`run_all.py`:** replaced by `cypha_bench_run`.
- **CUDA CI jobs:** **`windows_cuda_msvc`** and **`linux_cuda`** removed from `.github/workflows/ci.yml`; CUDA remains an optional local build.

### Changed
- **CI gate:** native-only — **CTests grow over time, see `scripts/cypha_native_validate_all.ps1` for the current authoritative tally** (`ctest -R native_`) across **two blocking jobs** (`build_and_test`, `windows_msvc`); optional **`federated_tls`** job (**`-DCYPHA_ENABLE_OPENSSL=ON`**, `native_federated_tls_smoke`, `continue-on-error`; local mirrors **`scripts/ci_federated_tls_linux.sh`**, **`scripts/ci_federated_tls_windows.ps1`**); optional **`corpus_and_d25`** job (WikiText fetch + **`native_corpus_smoke`** / **`native_d25_corpus_smoke`**, `continue-on-error`). Production overnight tier (300k) is **not** run in CI — maintainer-only via **`scripts/run_production_overnight.ps1`** (+ **`finalize_production_overnight.ps1`**).
- **`CYPHA_ACCEL_GPU_MIN_BATCH_ROWS` default:** **1** (was 16) — CUDA used for all batch sizes n≥1 when a GPU is available.
- **Docs:** README, CONTRIBUTING, docs hub, NATIVE_QUICKSTART, PORT_FULL_STACK, RESEARCH_STATUS, FUTURE, and C++ framework plan updated for native-first workflow.
- **C++23** standard for native build (`native/CMakeLists.txt`).
- **`cyphalm_parity`:** Windows subprocess fix (`CreateProcess` instead of `std::system`).
- **Intelligence Stats papers** moved to `docs/research/intelligence_stats/`.
- **cypha_som** removed; documented as failed experiment under `docs/archive/failed_experiments/cypha_som/`.
- **C++2023 migration plan:** `docs/native/migration/CPLUSPLUS_2023_MASTER_PLAN.md`.
- **Repo layout (Phase A):** `cypha_bench/` → `bench/`, `parity_fixtures/` → `fixtures/`, `install/` → `packaging/`.
- **Release install scripts:** under `packaging/` (`install_release_linux.sh`, `install_release_windows.ps1`).

### Added
- **Kernel LLR XOR pair features:** d03_xor + smoke historically used `xor_pair` — **97.8%** kernel acc (3 seeds, 8 passes); closes diagnostic 32 pp gap. **Historical / non-default** — living default is latent RFF (~76% vs sklearn ~79%); restore `xor_pair` with `CYPHA_D03_KERNEL_FEATURE_MODE=xor_pair`.
- **RPSM Option A scaffold:** `PsiMatrices` + `rpsm_score_matrix_batched` + CTest `native_rpsm_batched_llr_smoke`.
- **Intelligence Stats Phase 3:** CyphaLM profiler hook (`--intelligence-profile`), Qt self-correct checkbox, `CausalGraphMonitor`, `profile_guided_loss`, cell hypothesis sweep (`d19`, `cypha_cell_hypothesis_sweep`).
- **Intelligence Stats Phase 4:** EWC regularizer stub + curriculum sampler; `GET /intelligence/simulation`; REST `/update` batch+curriculum; Qt curriculum checkbox; epistemic halt on `/generate`; federated JSON merge (`cypha_federated_merge`).
- **RPSM Option B scaffold:** `rpsm_sequence_layer`, CyphaLM `--mode rpsm`, profile-guided loss in `train_step`; CTests `native_rpsm_sequence_smoke`, `native_cyphalm_bench_rpsm_smoke`.
- **Cell hypothesis H02–H14:** `cypha_cell_hypothesis` + EML activation in char-LSTM; all 28 variants runnable; Tier-2 smoke (`native_cell_hypothesis_tier2_smoke`).
- **WikiText D17 full profile:** `d17_wikitext_full_profile.json` + `CYPHA_BENCH_FULL_CORPUS`; CTest `native_d17_wikitext_smoke`.
- **WikiText D17 overnight profile:** `d17_wikitext_overnight_profile.json`, `--overnight` / `CYPHA_BENCH_OVERNIGHT=1` on `cyphalm_bench_native` and `cypha_bench_run`; CTest `native_d17_wikitext_overnight_smoke`.
- **Qt CyphaLM epistemic halt:** generate tab checkbox mirrors REST `/generate` `epistemic_halt` (Paper IV r_eu gate).
- **Intelligence Stats Phase 5:** RPSM hierarchy (`W_up`/`W_down`, global memory, Izaac init); RPSM batched LLR default path; NIG-state cell H06 + OOD branching H14; profile-guided loss in CyphaLM train-step backprop; EWC D16B smoke + REST `ewc_lambda`; federated coordinator CLI; Paper V multi-step simulation loop.
- **Intelligence Stats Phase 6:** 28-variant `--overnight-sweep`; bench **d20**; `scripts/run_d17_overnight.ps1`.
- **Intelligence Stats Phase 7:** `bench/BASELINE_LOCK.json`; `scripts/publish_release.ps1`; `native_overnight_mini_smoke`.
- **Intelligence Stats Phase 8:** bench **d22** cross-domain intelligence profile (d18 + d16 EWC + d20 cell sweep); `d22_intelligence_cross_profile.json`; **`cypha_baseline_lock`** + `scripts/update_baseline_lock.ps1`; hybrid EWC (SSM/GRIA α + B0 ngram count path, H01 α forget gate); CTests `native_d22_cross_smoke`, `native_ewc_hybrid_smoke`, `native_baseline_lock_smoke`.
- **Intelligence Stats Phase 9 (v2.3.9) — shipped:** hybrid EWC **weight** Fisher on GRIA **U**/**V** + SSM **W_fast** (extends Phase 8 α Fisher); **`scripts/run_overnight_all.ps1`** unified overnight runner (D17 + d21 RPSM + 28-variant cell sweep + baseline-lock refresh); bench **d23** overnight lock validation; EWC anchor/Fisher persistence in CyphaLM **`checkpoint.json`**; CTests `native_d23_overnight_lock_smoke`, `native_ewc_weights_smoke`.
- **Intelligence Stats Phase 10 (v2.3.10) — shipped:** hybrid EWC Fisher on GRIA **bias** + SSM **W_slow** (extends Phase 9 weight Fisher); bench **d24** production lock validation; **`cypha_baseline_lock --run all`** (D17 + d21 + cell-sweep in one CLI); Windows TLS CI mirror **`scripts/ci_federated_tls_windows.ps1`**; CTests `native_d24_production_lock_smoke`, `native_ewc_weights_smoke` (**99 total**).
- **Intelligence Stats Phase 11 (v2.3.11) — shipped:** WikiText-2 download scripts (**`scripts/download_wikitext2.ps1`**, **`scripts/download_wikitext2.sh`**) + **`bench/data/wikitext2/README.md`**; **Gutenberg fallback** for d17/d21 when WikiText absent (`gutenberg_fallback` source tag in **`cyphalm_corpus.cpp`**); **`corpus_smoke`** CLI + bench **d25** corpus readiness validation; overnight **`-Fast`** propagates **`CYPHA_BENCH_FAST=1`** through **`run_d17_overnight.ps1`**, **`run_rpsm_overnight.ps1`**, **`run_overnight_all.ps1`**, **`update_baseline_lock.ps1`** (synthetic corpus without WikiText); CTests `native_d25_corpus_smoke`, `native_corpus_smoke` (**101 total** at Phase 11).
- **Intelligence Stats Phase 12 (v2.3.12) — shipped everywhere:** **medium overnight tier** — **`-Medium`** on **`run_d17_overnight.ps1`**, **`run_rpsm_overnight.ps1`**, **`run_overnight_all.ps1`**, **`update_baseline_lock.ps1`** and **`--medium`** on **`cypha_baseline_lock`** (5k train / 256 eval, real WikiText/gutenberg, `status=medium_smoke`); bench **d26** medium overnight lock validation; profile **`bench/config/d26_medium_overnight_profile.json`**; **`scripts/validate_baseline_lock.ps1`** (+ **`baseline_lock_validate`** CLI) — schema + d17 pin check on **`bench/BASELINE_LOCK.json`**; **`scripts/publish_release.ps1 -DryRun`** / **`-NotesOnly`** preview without `gh`; optional CI **`corpus_and_d25`** job; CTests `native_d26_medium_overnight_smoke`, `native_baseline_lock_validate_smoke` (**103 total** at Phase 12).
- **Intelligence Stats Phase 13 (v2.3.13) — shipped:** **production overnight tier** — **`-Production`** on **`run_d17_overnight.ps1`**, **`run_rpsm_overnight.ps1`**, **`run_overnight_all.ps1`**, **`update_baseline_lock.ps1`** and **`--production`** on **`cypha_baseline_lock`** (300k train / 2000 eval, real WikiText/gutenberg, `status=production`, `CYPHA_BENCH_FULL_CORPUS=1`, `CYPHA_BENCH_OVERNIGHT=1`); dedicated **`scripts/run_production_overnight.ps1`** wrapper (chains **`run_overnight_all.ps1 -Production`**, logs to **`bench/results/production_overnight_<timestamp>.log`**); bench **d27** production overnight lock validation; profile **`bench/config/d27_production_lock_profile.json`**; **`scripts/validate_baseline_lock.ps1 -Production`** (+ **`baseline_lock_validate --production`**) — when `overnight_results.n_train >= 300000`, require `status=production` or `completed` and BPC within **0.05** of d17 hybrid **2.873** pin; CTest `native_d27_production_lock_smoke` (**104 total**)
- **Intelligence Stats Phase 14 (v2.3.14) — shipped:** **baseline lock status validator fix** — **`scripts/validate_baseline_lock.ps1`** and **`baseline_lock_validate`** accept **`medium_smoke`** and **`production`** (fixes production/medium overnight validation failure); **cell sweep artifact path** — default overnight output **`bench/results/cell_sweep`** via **`bench_paths::results_dir()`**, wired through **`cypha_baseline_lock --output-dir`**, **`update_baseline_lock.ps1`**, and **`run_overnight_all.ps1`**; bench **d28** unified overnight completion validation — cross-check **`overnight_results`**, **`rpsm_results`**, and **`cell_sweep_results`** for matching **`n_train`** / **`n_eval`** ( **`pending_overnight_complete`** when **< 300k**, full gate when **≥ 300k**); profile **`bench/config/d28_overnight_complete_profile.json`**; **`scripts/finalize_production_overnight.ps1`** post-overnight gate (**`-Production`** validate + d27/d28 bench, lock summary); **`run_production_overnight.ps1`** chains finalize on success; CTest `native_d28_overnight_complete_smoke`, `native_baseline_lock_validate_production_status` (**106 total**).
- **Intelligence Stats Phase 15 (v2.3.15) — shipped:** bench **d29** release readiness validation — schema + production + overnight-complete gates, release script presence check, optional **`baseline_lock_validate --production`**; profile **`bench/config/d29_release_readiness_profile.json`**; **`scripts/commit_production_lock.ps1`** post-overnight lock commit helper (**`-DryRun`** / **`-Force`**); **`scripts/watch_production_overnight.ps1`** stall-aware log/process watcher; **`cypha_native_validate_all.ps1`** env gates **`CYPHA_VALIDATE_OVERNIGHT_COMPLETE`**, **`CYPHA_VALIDATE_RELEASE_READINESS`**, **`CYPHA_STRICT_TEST_COUNT`**; CTest **`native_d29_release_readiness_smoke`** (**107 total**).
- **Intelligence Stats Phase 16 (v2.3.16) — shipped:** bench **d30** artifact path hygiene validation — legacy repo-root **`results/`** path detection in **`cell_sweep_results.artifact_path`**, verifies **`bench/results/.gitkeep`**; profile **`bench/config/d30_artifact_hygiene_profile.json`**; **overnight progress logging** — stderr **`[cyphalm]`** / **`[cell_sweep]`** (full sweep only); **`run_d17_overnight.ps1`** tees to **`bench/results/overnight_d17_<timestamp>.log`**; **`scripts/migrate_legacy_results.ps1`**; **`CYPHA_VALIDATE_ARTIFACT_HYGIENE=1`**; CTest **`native_d30_artifact_hygiene_smoke`** (**108 total**).
- **Intelligence Stats Phase 17 (v2.3.17) — shipped:** bench **d31** post-overnight pipeline validation — d27→d30 chain + pipeline script presence gate; profile **`bench/config/d31_post_overnight_pipeline_profile.json`**; **`scripts/poll_and_finalize_overnight.ps1`** — poll until processes exit, finalize + commit preview; **`scripts/cleanup_legacy_results.ps1`** + **`migrate_legacy_results.ps1 -ArchiveLegacy`**; **`run_production_overnight.ps1`** chains **`commit_production_lock.ps1 -DryRun`** after finalize; **`CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE=1`**; CTest **`native_d31_post_overnight_pipeline_smoke`** (**109 total**).
- **Intelligence Stats Phase 18 (v2.3.18) — shipped:** bench **d32** production complete validation — full production tier gate when **`n_train >= 300000`**; profile **`bench/config/d32_production_complete_profile.json`**; **`scripts/validate_production_complete.ps1`** — unified post-overnight validator (baseline lock + finalize + d31/d30); **`scripts/start_poll_finalize_background.ps1`**; cell sweep **`overnight_progress.log`** sidecar; **`publish_release.ps1`** **`gh auth`** preflight; **`CYPHA_VALIDATE_PRODUCTION_COMPLETE=1`**; CTest **`native_d32_production_complete_smoke`** (**110 total**; **113** with optional experiment DB smokes when SQLite amalgamation is fetched).
- **Intelligence Stats Phase 19 (v2.3.19) — shipped:** bench **d33** release publish validation — publish script presence + production/overnight-complete gates; profile **`bench/config/d33_release_publish_profile.json`**; **`scripts/verify_release_publish.ps1`** — production complete + d33 + **`publish_release.ps1 -DryRun`**; poll/finalize **BuildDir auto-detect**; **`CYPHA_VALIDATE_RELEASE_PUBLISH=1`**; CTest **`native_d33_release_publish_smoke`** (**111 total**).
- **Intelligence Stats Phase 20 (v2.3.20) — shipped:** bench **d34** repo smoke hygiene validation — repo-root **`d*_smoke.json`** leak detection + **`.gitignore`** patterns; profile **`bench/config/d34_repo_smoke_hygiene_profile.json`**; **`scripts/cleanup_repo_smoke_artifacts.ps1`**; poll **heartbeat** logging; **`CYPHA_VALIDATE_REPO_SMOKE_HYGIENE=1`**; CTest **`native_d34_repo_smoke_hygiene_smoke`** (**112 total**).
- **Intelligence Stats Phase 21 (v2.3.21) — shipped:** bench **d35** lock commit pipeline validation — **`commit_production_lock.ps1`** / **`poll_and_finalize_overnight.ps1`** presence + production tier gates; profile **`bench/config/d35_lock_commit_pipeline_profile.json`**; **`scripts/verify_production_pipeline.ps1`**; overnight watch variant progress; poll dedupe + heartbeat fix; **`CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE=1`**; CTest **`native_d35_lock_commit_pipeline_smoke`** (**113 total**).
- **Intelligence Stats Phase 22 (v2.3.22) — shipped:** bench **d36** production pipeline E2E validation — full maintainer overnight-to-publish toolchain script + profile presence gates (d27-d35); profile **`bench/config/d36_pipeline_e2e_profile.json`**; **`scripts/run_post_overnight.ps1`**; watch **effective_n_train**; **`CYPHA_VALIDATE_PIPELINE_E2E=1`**; CTest **`native_d36_pipeline_e2e_smoke`** (**114 total**).
- **Intelligence Stats Phase 23 (v2.3.23) — shipped:** bench **d37** overnight lock refresh validation — post-overnight baseline lock update toolchain; profile **`bench/config/d37_lock_refresh_profile.json`**; **`scripts/migrate_inflight_overnight_artifacts.ps1`**; finalize best-effort **`update_baseline_lock -Production`**; **`CYPHA_VALIDATE_LOCK_REFRESH=1`**; **`publish_release.ps1 -NotesPath`**; CTest **`native_d37_lock_refresh_smoke`** (**115 total**).
- **Intelligence Stats Phase 24 (v2.3.24) — shipped:** bench **d38** production overnight completion certificate — full 300k cross-section + 28-variant cell sweep; profile **`bench/config/d38_overnight_certificate_profile.json`**; **`poll_and_finalize_overnight.ps1 -AutoCommit`**; watch stall detector; **`CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE=1`**; CTest **`native_d38_overnight_certificate_smoke`** (**116 total**).
- **Intelligence Stats Phase 25 — shipped (in-tree; formal release tag v2.3.25 is One Cypha cutover):** bench **d39** intelligence profile completeness validation — 7-stat monitoring toolchain + **`cyphalm_bench_native --intelligence-profile`** smoke; profile **`bench/config/d39_intelligence_monitor_profile.json`**; **`CYPHA_VALIDATE_INTELLIGENCE_MONITOR=1`**; CTest **`native_d39_intelligence_monitor_smoke`** (**117 total**).
- **Intelligence Stats Phase 26 (v2.3.26) — shipped:** bench **d40** math integration validation — **`cyphalm_math_integration`** + 7-stat profile-guided navigation loss + **`cyphalm_bench_native --math-integration`** smoke; profile **`bench/config/d40_math_integration_profile.json`**; **`scripts/run_math_integration_bench.ps1`**; **`CYPHA_VALIDATE_MATH_INTEGRATION=1`**; CTest **`native_d40_math_integration_smoke`** (**118 total**).
- **Intelligence Stats Phase 27 (v2.3.27) — shipped:** bench **d41** math integration scale validation (5k/256, ΔBPC/Δκ); navigation-loss **warmup** + hybrid **blend nudge**; **profile curriculum** re-enabled in math preset; cell sweep **`--intelligence-profile`** κ overlay; LM **self-correct** REST auto-`epistemic_halt`; CTests **`native_d41_math_integration_scale_smoke`**, **`native_lm_self_correct_smoke`** (**121 total**).
- **Intelligence Stats Phase 28 (v2.3.28) — shipped:** **CharLSTM navigation loss** backprop; **adaptive λ from κ**; **d20/d22 κ ranking**; bench **d42** production math-integration gate + **`run_d17_overnight.ps1 -MathIntegration`**; CTests **`native_d42_math_integration_production_smoke`**, **`native_navigation_loss_char_lstm_smoke`** (**123 total**).
- **Intelligence Stats Phase 29 (v2.3.29) — shipped:** **hybrid LSTM logit navigation grads** (`d_logit_uniform`); **κ×BPC Pareto cell ranking**; **`cypha_baseline_lock --run d17-math`**; production overnight **`-MathIntegration`** + finalize **d42**; CTest **`native_navigation_loss_hybrid_smoke`** (**124 total**).
- **Intelligence Stats Phase 30 (v2.3.30) — shipped:** **direct LSTM weight navigation** (logit nudge inside `backward_step`); **Pareto default cell winner** in lock; bench **d43** math lock validation; **H04 kernel LLR stub** in CyphaDIF route; CTests **`native_d43_math_integration_lock_smoke`**, **`native_kernel_llm_h04_smoke`** (**126 total**).
- **Intelligence Stats Phase 31 (v2.3.31) — shipped:** **Nyström `KernelMemory`** in CyphaDIF route/train/checkpoint (replaces H04 RBF proxy when `n_basis≥4`); **κ trajectory λ schedule** (`scale_profile_guided_loss_from_trajectory`); bench **d44** kernel CyphaLM validation; CTest **`native_d44_kernel_nystrom_cyphalm_smoke`** (**127 total**).
- **Intelligence Stats Phase 32 (v2.3.32) — shipped:** **per-stat deviation λ scheduling** (`scale_profile_guided_loss_by_stat_deviation`); enriched **math integration export** (`stat_deltas`, `kappa_trajectory`, `navigation_config`); **cell sweep Pareto lock** (`--intelligence-profile` + full `best_pareto_variant` object); bench **d45**; CTest **`native_d45_per_stat_navigation_smoke`** (**128 total**).
- **Intelligence Stats Phase 33 (v2.3.33) — shipped:** **Paper IV τ forget gate** (`use_tau_forget_gate`, `hybrid_forget_gate_scale`); **kernel LLR in math preset**; **per_stat_deviation_span=1.0**; bench **d46**; CTests **`native_d46_math_stack_upgrade_smoke`**, **`native_tau_forget_gate_smoke`** (**130 total**).
- **Intelligence Stats Phase 34 (v2.3.34) — shipped:** **κ ceiling λ scheduler** (`use_kappa_ceiling_lambdas`); **LSTM hidden D_eff nudge**; **`--per-stat-deviation-span`** CLI; bench **d47**; CTest **`native_d47_span_ablation_smoke`** (**131 total**).
- **Intelligence Stats Phase 35 (v2.3.35) — shipped:** **Eigenvalue D_eff PR**; **tunable κ ceiling**; **r_eu forget gate** (opt-in); bench **d48**; CTest **`native_d48_kappa_ceiling_ablation_smoke`** (**132 total**).
- **Intelligence Stats Phase 37 (v2.3.37) — shipped:** **κ excess grad margin** (`kappa_excess_grad_margin`, default **0.02**); separate **`use_kappa_excess_grad_nudge`** + CLI **`--kappa-excess-grad-margin`**, **`--kappa-excess-grad-scale`**, **`--disable-kappa-excess-grad-nudge`**; **pinned bench seed** — **`--bench-seed N`** (+ **`CYPHA_BENCH_SEED`**); bench **d50** joint κ–BPC lock validation (repro **`bench/BASELINE_LOCK.json`** @ 5k, seed **42**); CTest **`native_d50_math_joint_lock_smoke`** (**134 total**).
- **Intelligence Stats Phase 36 (v2.3.36) — shipped:** **κ trajectory ceiling** + **κ excess grad nudge**; **soft r_eu forget blend**; ceiling **2D grid d49**; CLI **`--kappa-ceiling-min-scale`**, **`--use-eigenvalue-d-eff`**; CTest **`native_d49_ceiling_grid_joint_smoke`** (**133 total**).
- **Intelligence Stats Phase 38 (v2.3.38) — shipped:** **ablation winners in preset** (κ target **0.83**, min_scale **0.40**); **κ-adaptive kernel blend**; bench **d51** opt-in lever A/B; CTest **`native_d51_opt_in_lever_joint_smoke`** (**135 total**).
- **Intelligence Stats Phase 39 (v2.3.39) — shipped:** **r_eu forget gate enabled in preset**; **d49 seed-42** ceiling grid refresh; bench **d52** preset ship lock; CTest **`native_d52_preset_ship_lock_smoke`** (**136 total**).
- **Intelligence Stats Phase 40 (v2.3.40) — prep:** **κ-aware navigation warmup** (`scale_navigation_warmup_from_kappa`); bench **d53** production preset ship lock (300k tier + 5k subprocess); CTest **`native_d53_production_preset_ship_lock_smoke`** (**137 total**).
- **Intelligence Stats Phase 41 (v2.3.41) — prep:** **nav warmup CLI ablation**; bench **d54** production math certificate; **d55** nav warmup grid; CTests **`native_d54_production_math_certificate_smoke`**, **`native_d55_nav_warmup_grid_joint_smoke`** (**139 total**).
- **Intelligence Stats Phase 42 (v2.3.42) — prep:** **cell sweep `--math-integration`** wiring; bench **d56** cell sweep math joint @ 5k; CTest **`native_d56_cell_sweep_math_integration_smoke`** (**140 total**).
- **Intelligence Stats Phase 43 (v2.3.43) — prep:** bench **d57** production cell sweep math certificate; finalize **d56+d57**; **`update_baseline_lock.ps1 -MathIntegration`**; CTest **`native_d57_production_cell_sweep_math_certificate_smoke`** (**141 total**).
- **Intelligence Stats Phase 59 (v2.3.59) — prep:** **structural preset CLIs** (`--n-experts`, `--max-memory-slots`, `--compress-interval`); bench **d74** grid {4, 8, 12}, **d75** {128, 256, 512}, **d76** {8, 16, 32} @ 5k seed 42; d62 stack → **24** tables (**full preset scalar coverage**); CTests **`native_d74_*`**, **`native_d75_*`**, **`native_d76_*`** (**158–160 total**).
- **Intelligence Stats Phase 58 (v2.3.58) — prep:** **hybrid blend lr CLI** (`--hybrid-blend-lr`); bench **d73** grid {0.005, 0.01, 0.02} @ 5k seed 42; CTest **`native_d73_hybrid_blend_lr_grid_joint_smoke`** (**157 total**).
- **Intelligence Stats Phase 57 (v2.3.57) — prep:** **kernel lr scale CLI** (`--kernel-lr-scale`); **alpha init CLI** (`--alpha-init`); bench **d71** grid {0.5, 1.0, 2.0} + **d72** grid {0.3, 0.5, 0.7} @ 5k seed 42; d62 stack → **21** tables; CTests **`native_d71_kernel_lr_scale_grid_joint_smoke`**, **`native_d72_alpha_init_grid_joint_smoke`** (**155–156 total**).
- **Intelligence Stats Phase 56 (v2.3.56) — prep:** **MDL forget max norm CLI** (`--mdl-forget-max-norm`); bench **d70** grid {2.0, 4.0, 8.0} @ 5k seed 42; d62 stack → **18** tables; CTest **`native_d70_mdl_forget_max_norm_grid_joint_smoke`** (**154 total**).
- **Intelligence Stats Phase 55 (v2.3.55) — prep:** **hybrid blend logit CLI** (`--hybrid-blend-logit`); bench **d69** grid {0.0, 0.5, 1.0} @ 5k seed 42; CTest **`native_d69_hybrid_blend_logit_grid_joint_smoke`** (**153 total**).
- **Intelligence Stats Phase 54 (v2.3.54) — prep:** **kernel_m CLI** (`--kernel-m`); bench **d68** grid {32, 64, 128} @ 5k seed 42; d62 stack → **16** tables; CTest **`native_d68_kernel_m_grid_joint_smoke`** (**152 total**).
- **Intelligence Stats Phase 53 (v2.3.53) — prep:** **base kernel blend CLI** (`--kernel-blend`); bench **d67** grid {0.15, 0.25, 0.40} @ 5k seed 42; CTest **`native_d67_kernel_blend_grid_joint_smoke`** (**151 total**).
- **Intelligence Stats Phase 52 (v2.3.52) — prep:** **free energy beta CLI** (`--free-energy-beta`); bench **d66** grid {0.005, 0.01, 0.02} @ 5k seed 42; d62 stack → **14** tables; CTest **`native_d66_free_energy_beta_grid_joint_smoke`** (**150 total**).
- **Intelligence Stats Phase 51 (v2.3.51) — prep:** **navigation loss warmup CLI** (`--navigation-loss-warmup-steps`); bench **d65** grid {100, 200, 400} @ 5k seed 42; CTest **`native_d65_navigation_loss_warmup_grid_joint_smoke`** (**149 total**).
- **Intelligence Stats Phase 50 (v2.3.50) — prep:** **κ trajectory window CLI** (`--kappa-trajectory-window`); bench **d64** grid {8, 16, 32} @ 5k seed 42; d62 stack extended to **12** tables; CTest **`native_d64_kappa_trajectory_window_grid_joint_smoke`** (**148 total**).
- **Intelligence Stats Phase 49 (v2.3.49) — prep:** **r_eu forget blend CLI** (`--reu-forget-gate-blend`); bench **d63** grid {0.0, 0.25, 0.50} @ 5k seed 42; CTest **`native_d63_reu_forget_blend_grid_joint_smoke`** (**147 total**).
- **Intelligence Stats Phase 48 (v2.3.48) — prep:** bench **d62** unified math ablation stack complete (audits d47–d61 tables + subprocess joint); finalize **d59–d62**; CTest **`native_d62_math_ablation_stack_complete_smoke`** (**146 total**).
- **Intelligence Stats Phase 47 (v2.3.47) — prep:** bench **d61** κ excess grad scale grid {0.25, 0.35, 0.50} @ 5k seed 42; CTest **`native_d61_excess_grad_scale_grid_joint_smoke`** (**145 total**).
- **Intelligence Stats Phase 46 (v2.3.46) — prep:** bench **d60** κ excess grad margin grid {0.01, 0.02, 0.04} @ 5k seed 42; subprocess excess-grad CLI wiring; CTest **`native_d60_excess_grad_margin_grid_joint_smoke`** (**144 total**).
- **Intelligence Stats Phase 45 (v2.3.45) — prep:** **κ kernel blend floor CLI ablation** (`--kappa-kernel-blend-floor`, `--disable-kappa-kernel-blend-scale`); bench **d59** kernel blend floor grid {0.05, 0.08, 0.12} @ 5k seed 42; CTest **`native_d59_kernel_blend_floor_grid_joint_smoke`** (**143 total**).
- **Intelligence Stats Phase 44 (v2.3.44) — prep:** **best_pareto_variant** cell sweep export; bench **d58** unified production overnight math complete; finalize **d58**; CTest **`native_d58_production_overnight_math_complete_smoke`** (**142 total**).
- **H16 SR gate laws:** `sr_gate_laws.hpp/cpp` — fit closed-form forget-gate laws, apply in char_lstm; `native_sr_gate_laws_smoke`.
- **RPSM d21 bench:** end-to-end CyphaLM rpsm `train_sequence`; bench **d21**; `scripts/run_rpsm_overnight.ps1`; `native_d21_rpsm_smoke`.
- **CyphaLM EWC full:** embed + lm_head Fisher (diagonal grad²); extended `native_ewc_cyphalm_smoke`.
- **Federated TLS option:** `CYPHA_ENABLE_OPENSSL=ON` for HTTPS coordinator/worker; `native_federated_tls_smoke` (skipped without OpenSSL).
- **Hybrid EWC:** `HybridEwcRegularizer` — diagonal Fisher on SSM multiscale **α** + GRIA per-token **α** (char-LSTM embed/head Fisher retained); B0 **`ngram_count_table`** online prior when `ngram_context > 0`; H01 **`use_alpha_forget_gate`** scales char-LSTM forget gate by mean GRIA α; CTest `native_ewc_hybrid_smoke`.
- **`cypha_baseline_lock`:** CLI merges D17/d21/cell-sweep BPC into **`bench/BASELINE_LOCK.json`**; wrapper **`scripts/update_baseline_lock.ps1`**; CTest `native_baseline_lock_smoke`.
- **CyphaLM checkpoint:** save/load **`ngram_count_table`** in checkpoint JSON (B0 ngram count path round-trip).
- **CTests +3 (Phase 8):** `native_d22_cross_smoke`, `native_ewc_hybrid_smoke`, `native_baseline_lock_smoke` (**96 total** at Phase 8).
- **CTests +2 (Phase 9, shipped):** `native_d23_overnight_lock_smoke`, `native_ewc_weights_smoke` (**98 total**).
- **CTests +1 (Phase 10):** `native_d24_production_lock_smoke` (**99 total**); extended `native_ewc_weights_smoke` (GRIA bias + SSM W_slow Fisher).
- **CTests +2 (Phase 11):** `native_d25_corpus_smoke`, `native_corpus_smoke` (**101 total** at Phase 11).
- **CTests +2 (Phase 12):** `native_d26_medium_overnight_smoke`, `native_baseline_lock_validate_smoke` (**103 total** at Phase 12).
- **CTests +1 (Phase 13):** `native_d27_production_lock_smoke` (**104 total**).
- **CTests +2 (Phase 14):** `native_d28_overnight_complete_smoke`, `native_baseline_lock_validate_production_status` (**106 total**).
- **CTests +1 (Phase 15):** `native_d29_release_readiness_smoke` (**107 total**).
- **CTests +1 (Phase 16):** `native_d30_artifact_hygiene_smoke` (**108 total**).
- **CTests +1 (Phase 17):** `native_d31_post_overnight_pipeline_smoke` (**109 total**).
- **CTests +1 (Phase 18):** `native_d32_production_complete_smoke` (**110 total**).
- **CTests +3 (Phase 7):** `native_sr_gate_laws_smoke`, `native_d21_rpsm_smoke`, `native_federated_tls_smoke` (**93 total** at Phase 7).
- **Preprocessor:** auto `auto_rff_gamma_cv` when tabular dim ≤30 and `rff_gamma` left at default 1.0.
- **Release notes script:** `scripts/create_release_notes.ps1`; release workflow hook.
- **Web UI tabs:** CyphaLM generate, Experiments (`/models`), Intelligence report (`/intelligence/report`).
- **GGUF export:** tensor blobs for `enc_W`, `world.mu`, class `D`/`D_T`, `inv_v`, `llr_bias` from `.cypha`.
- **ONNX export smoke:** CTest `native_onnx_export_smoke` writes valid graph from `reference.cypha`.
- **Phase B native layout:** `native/apps/`, `native/tests/parity/`, `cmake/CyphaApps.cmake` + `CyphaParity.cmake`.
- **Web UI (§4):** vanilla SPA at `GET /` served by `cypha_rest`; CTest `native_rest_ui_smoke`.
- **Fixture generators:** `cypha_fixture_gen` for batch_llr, memory_train, preprocessor, train_step_vector, quantile_dif_train, regression_m4.
- **`cypha_onnx_export`:** header-only ONNX ModelProto writer for VectorEncoder+LLR inference path.
- **`cyphalm_train`:** native CyphaLM training CLI + checkpoint save; CTest `native_cyphalm_train_smoke`.
- **`cyphalm_ssm_diagnose`:** CellAI/SSM probe tool; `--ssm-diagnose` on `cypha_bench_run` for d10/d17.
- **Intelligence Profiler Papers II–V:** extended measurers + CTest `native_intelligence_profiler_papers`.
- **Kernel LLR tuning:** diverse landmarks, whitening fallback, M=512 tuned profile (XOR ~71% kernel acc).
- **`cypha_kernel_tune`:** grid search wrapper for XOR kernel bench.
- **Packaging (§3):** `packaging/build_appimage.sh`, `packaging/build_windows_bundle.ps1`; release workflow AppImage + native-only assets.
- **Qt dark theme + model card editor (§2d–e);** threaded bulk REST `/update`.
- **`auto_rff_gamma_cv`:** RFF bandwidth grid CV in native preprocessor; Qt Fit Preprocessor gamma mode combo; CTest `native_preprocessor_rff_gamma_cv`.
- **`cypha_fixture_gen`:** native fixture regeneration CLI (`--fixture batch_llr`); CTest `native_fixture_gen_list`.
- **Multi-model REST:** registry map, optional `"model"` on predict/update, `--preload-registry`; CTest `native_rest_multi_model`.
- **Qt bulk train worker:** background `QThread` with live loss chart and cancel.
- **Qt chart UX (§2b):** zoom, pan, tooltips on loss charts.
- **Experiment compare (§2c):** multi-run loss overlay in Experiments panel.
- **`/uncertainty-rank`:** active-learning entropy ranking; CTest `native_rest_uncertainty_rank`.
- **Auto-γ RFF (§0b):** native `PreprocessorState::auto_rff_gamma`; bench + Qt shell wired.
- **Nyström kernel LLR (native):** `KernelMemory` in C++ with train/infer wiring, `.cypha` persistence, XOR bench (`xor_kernel_bench`, CTest `native_xor_kernel_bench_smoke`), bench domain **`d03_xor`** (`cypha_bench_run --domain-tag d03_xor`), opt-in profile `bench/config/kernel_llr_profile.json`.
- **Intelligence Profiler (C++):** `native/include/cypha/intelligence/` — NIG statistic states, 7-stat measurers, κ, health signal; CTest `native_intelligence_profiler_smoke`.
- **Research upgrades hub:** RPSM, nonlinear boundary, cell hypothesis specs under `docs/research/upgrades/`.
- **`cypha_fixture_gen`:** all **32** sidecar parity fixtures regenerable; CTest `native_fixture_gen_list`.
- **`cypha_parity_run`:** unified parity driver `--fixture NAME`; CTest `native_parity_run_list`.
- **`bench_domains.cpp`:** domain runners split from `cypha_bench_run` CLI.
- **`mt19937_rng`:** renamed from `numpy_default_rng` (parity MT19937).
- **REST schema contract:** `native_rest_schema_contract` replaces pytest `test_api_contract`.
- **Studio Web UI tabs:** Update, Models/Load, LM generate, Session RNG, Metrics chart.
- **`embed_static_ui`:** C++ tool replaces Python embed script.
- **Kernel XOR pair features:** raw XOR polynomial kernel path (~97% acc smoke); `kernel_x` / `kernel_features` API.
- **`bench/config/auto_rff_gamma_cv_profile.json`** for d01 RFF CV preprocessor.
- **Diagnostics phase 5:** intelligence profiler inline + SSM recommendations JSON.
- **`GET /intelligence/profile`** and **`GET /intelligence/report`** on `cypha_rest` (report = reference fixture P-space).
- **Qt:** chunked CSV train, MKE bulk worker, uncertainty sort checkbox, Qt Charts `QRectF` fix.
- **`cypha_export_gguf`:** GGUF header + manifest stub; CTest `native_export_gguf_help`.

### Fixed
- **`lm_intelligence_monitor`:** remove **`batch.sequence`** OOB (τ via **`batch.tau`**); guard **`embed_dim != field_dim`** on **`batch.input`**.
- **`cyphalm_model`:** skip **`update_profiler_from_lm_token`** when LM intelligence monitor is active (avoid double-counting).
- **Phase 25+26:** bench **d39** / **d40** — throw when subprocess exit **≠ 0** (was silently passing on crash).

---

## [2.3.25] — 2026-07-18 · One Cypha cutover

### Added
- **One public type `cypha::Cypha`:** classify + regress + latent sample + sequence (next-token / text generate) in a single native runtime.
- **REST:** `POST /sample`, `POST /retrieve`, `/sequence/*`; health `model_type=Cypha`; primary `predict` / `update` / sample paths owned by `Cypha`.
- **Living sequence spine:** PGM→Wy (U06); D17 hybrid **2.873 BPC** retained as historical pin only.
- **Qt Studio:** `studio_cypha_` owns classify / sample / sequence; Sample latents UI.
- **Smokes:** `native_one_cypha_smoke`, `native_cypha_rest_one_smoke`, PGM cell checkpoint round-trip; CI green.

### Changed
- **Branding / API cutover:** dual CyphaDIF + CyphaLM surfaces collapsed into `cypha::Cypha`; `/dif/*` and `/lm/*` removed from the living REST contract (see [`docs/reports/ONE_CYPHA_CUTOVER.md`](docs/reports/ONE_CYPHA_CUTOVER.md)).

---

## [2.2.8] — 2026-06-11 · All four CI jobs blocking

### Changed
- **`linux_cuda`** CI job is now **blocking** (was `continue-on-error`); Jimver **`cudart`/`thrust`** + **`CUDAToolkit_ROOT`**; failure log artifact.
- Docs: CONTRIBUTING, VERIFY_PLAN §5, ROADMAP, ACCEL_CUDA, framework plan, README test counts synced to **four blocking CI jobs**.
- `cyphalm_checkpoint_parity`: log BPC delta on Python checkpoint load.
- Native CMake project version **2.2.8**.

---

## [2.2.7] — 2026-06-11 · MSVC CUDA CI green

### Fixed
- **MSVC + CUDA CI (Windows):** green on GHA — `ilammy/msvc-dev-cmd`, explicit `CUDA_PATH` from Jimver output, Ninja **1.12.1** pin, `thrust` subpackage, optional CUDA VS integration copy; failure log artifact.

### Changed
- **`windows_cuda_msvc`** is now a **blocking** CI job (was `continue-on-error`).
- Native CMake project version **2.2.7**.

---

## [2.2.6] — 2026-06-11 · Hybrid checkpoint atol + MSVC CUDA CI hardening

### Fixed
- **MSVC + CUDA CI:** `vswhere` + `vcvars64.bat` + Ninja with explicit `cl`/`nvcc` (Jimver nvcc-only install; no Nsight VSE hang).
- **CyphaLM hybrid checkpoint parity:** tighten `atol_bpc` from 0.35 → **0.02** (measured native load delta ~0.002 BPC).

### Changed
- Native CMake project version **2.2.6**; CyphaLM tracker marks `proj_dif` GRIA wiring complete.
- `bench/README.md`: note on when to commit `report/` baseline snapshots.

---

## [2.2.5] — 2026-06-11 · Docs sync + MSVC CUDA CI fix

### Changed
- Verify hub, ROADMAP, README test counts aligned with green CI (~274 pytest, 52 CTest).
- **MSVC + CUDA CI:** `ilammy/msvc-dev-cmd` + Ninja single-config (configure/build/ctest in one step so `cl`/`nvcc` env persists).

---

## [2.2.4] — 2026-06-11 · CI green (Linux CTest + pytest)

### Fixed
- **Linux CI CTest:** CyphaLM checkpoint sidecars use portable relative `checkpoint.json` paths; `cyphalm_checkpoint_parity` resolves paths against the sidecar directory.
- **Linux CI pytest:** restore empty Studio GUI shim modules as re-exports from `widgets.py`; extend CTest↔pytest registry; add bench/tune/diagnostics/SOM/GH infer subprocess tests; Qt shell `--help` documents PNG/SVG/CSV export keywords.
- **Linux CI Qt:** compile Qt targets on ubuntu-latest; exclude headless-incompatible GUI CTest exec (`native_qt_shell_smoke`, `native_qt_stub_load_reference`).

### Changed
- **`scripts/ci_native_linux.sh`:** skip Qt GUI CTests when `CYPHA_BUILD_QT=1` on headless hosts (matches CI).

---

## [2.2.3] — 2026-05-31 · Release packaging hardening

### Fixed
- **Windows release verify:** fail-fast packaging when required PE binaries are missing; clearer verify step; build `--target all`; rm staging before unzip check.
- **MSVC + CUDA CI:** try VS 2026 then VS 2022 generators; job marked non-blocking (`continue-on-error`).
- **Ruff CI:** pin `ruff>=0.9,<0.10` for reproducible lint.
- **CI MinGW job:** replace PE CTest (cannot run on Linux) with artifact existence smoke.
- **Ruff UP038:** `isinstance(exc, ValueError | TypeError)` for ruff 0.9 pin compatibility.
- **Linux CI Qt/CTest:** install XCB/EGL/GL runtime deps; compile Qt targets; exclude GUI exec tests from CTest on headless runners.

---

## [2.2.2] — 2026-05-31 · CI / release pipeline fixes

### Fixed
- **MinGW cross-compile:** vendored zlib via CMake FetchContent when system `ZLIB` is missing (fixes `cypha_bench_run` on release Windows PE builds).
- **Ruff CI:** auto-format + targeted `per-file-ignores`; all lint paths green.
- **MSVC + CUDA CI:** auto-detect VS 2026 vs 2022 generator on `windows-latest`.

---

## [2.2.1] — 2026-06-11 · Qt Studio dialogs + doc sync

### Added
- **Qt shell:** Settings dialog (QSettings, Studio-compatible prefs) and Confusion Matrix view after labeled batch predict.

### Changed
- Master plan and quickstart marked **v2.5 complete**; validation gate unchanged (52 CTests, 155 pytest).

---

## [2.2.0] — 2026-06-11 · Full C++ framework release

### Added
- **Full native production framework** — bench (`cypha_bench_run`, `cypha_bench_report`), tune (`cypha_tune_run`), diagnostics (`cypha_diagnostics_run`), and REST **`/dif/*`** routes in **`cypha_rest`**; Python required only for fixture generation and research prototyping.
- **`cypha_qt_shell`** — Qt Studio shell with **9 tabs** (Data, Model, Train, Predict, Registry, Server, Experiments, CyphaLM, Help).
- **Native validation gate** — `scripts/cypha_native_validate_all.ps1` orchestrates CTest, pytest parity, bench smoke, tune dry-run, and REST contract checks.

### Changed
- **CyphaLM 300k baseline lock** — native hybrid **2.897 BPC** on D17 @ 300k (release-time sweep figure; canonical pin remains **2.873** in `bench/BASELINE_LOCK.json` — see `docs/reports/BASELINE_PIN_CANONICAL_2026-07-17.md`).
- **Test coverage** — **52 CTests** (`native_*`) and **155 pytest** cases green on the full native gate.

### Installers
- Release bundles via `scripts/package_release_windows.sh` / `scripts/package_release_linux.sh`; see [`docs/native/NATIVE_QUICKSTART.md`](docs/native/NATIVE_QUICKSTART.md).

---

## [1.1.0] — 2026-05-31 · CyphaLM native release

### Added
- **`cypha_lm_native`** — C++ CyphaLM Tiers 0–2–4 (build6 sweep hybrid **2.892 BPC @ 300k** vs canonical pin **2.873**).
- Native REST LM routes in **`cypha_rest`**: `/lm/load`, `/lm/metrics`, `/lm/predict_next`, `/generate`, `/generate/stream`.
- Checkpoint save/load with DIF + SSM state; Python GRIA import via `load_from_full_w`.
- GitHub Release workflow: Linux `.tar.gz` + Windows `.zip` installer bundles (`scripts/package_release_*.sh`).

### Installers
- **Linux:** `cypha-1.1.0-linux-x86_64.tar.gz` → `bash install.sh`
- **Windows:** `cypha-1.1.0-windows-x86_64.zip` → `powershell -File install.ps1`

---

## [Unreleased] — 2026-06-10 CyphaLM native Tier 0–2–4 integration

### Added
- **`cypha_lm_native`** static library — `native/src/cyphalm/*.cpp` (CMake `GLOB`), OpenMP optional.
- **`cyphalm_bench_native`** — BPC bench CLI (`--mode`, `--profile d17|d04`, `--n-train`, `--n-eval`, `--threads`).
- **`cyphalm_parity`** — meta-runner for native CyphaLM parity tools.
- **`scripts/generate_cyphalm_native_fixtures.py`** — one-time Python → `fixtures/cyphalm_*/sidecar.json`.
- **`tests/test_cyphalm_native_parity.py`** — subprocess parity (skip if binary missing).
- **`native/include/cypha/cyphalm/cyphalm_config.hpp`** — unified config + bench mode mapping.
- PORT_CONTRACT **§4b** and `CYPHALM_NATIVE_UPGRADE_MASTER.md` integration / build notes.

---

## [Unreleased] — 2026-06-08 Branch A frozen embeddings

### Added
- **`frozen_text_embeddings.py`** — MiniLM or hashing frozen text vectors.
- **`cypha_branch_a_sweep.py`** — CyphaDIF on frozen ST vs TF-IDF on 20 Newsgroups.

### Changed
- Branch A @ 2k samples: CyphaDIF + frozen MiniLM **62.5%** vs TF-IDF path **34.0%**.
- D09 integration: `run_d09_branch_a.py`, Gutenberg OOD epistemic **6.3×** in-domain.

### Added (follow-up)
- `branch_a_documents.py`, `run_d09_branch_a.py`, `scripts/demo_branch_a_route.py`.
- **REST Branch A routing:** `POST /route/text`, `POST /route/generate`, Ollama fallback client.
- **Studio chat:** Branch A mode in Settings → Inference; `BranchADispatchWorker` routes text in chat.
- **Router checkpoint:** `save_checkpoint` / `load_checkpoint`, `scripts/save_branch_a_router.py`, `POST /route/save`.
- **Encoder sweep:** VectorEncoder **59.5%** vs RFF **4.5%** @ 384-d MiniLM — keep VectorEncoder.

---

## [Unreleased] — 2026-06-08 Cypha Tests Phase 2A encoder

### Added
- **`EncoderProjection.hebbian_update`** and `CyphaDIF.encoder_update_mode` (`contrastive` | `hebbian`).
- **`cypha_encoder_phase2a_sweep.py`** — D01 + 20 Newsgroups contrastive vs Hebbian benchmark.

### Changed
- Phase 2A baseline: competitive Hebbian **underperforms** contrastive on all 4 tasks; keep contrastive default.

---

## [Unreleased] — 2026-06-08 char_lstm mode + Phase 2 Hebbian

### Added
- **`context_mode=char_lstm`** — LSTM-only path in `CyphaLM` (C1 model-class).
- **`cyphalm_hebbian_phase2_sweep.py`** — Cypha Tests 2C sparse Hebbian toggle.
- **`docs/CYPHA_TESTS_PHASE2.md`** — Phase 2 experiment map (2A–2C).

### Changed
- Hybrid sweep supports `--cells` filter; third cell `char_lstm`.
- **Char-LSTM @ 300k:** **2.876 BPC** (≈ hybrid 2.873); Hebbian SSM **neutral @ 40k**.

---

## [Unreleased] — 2026-06-08 char 1A + D04 refresh

### Added
- **Char-level shuffle probe** (Cypha Tests 1A) in `eval_shuffled_stream_bpc`.
- **`run_d04_hybrid_refresh.py`** — D04 @ 300k with ablation stub from sweep.

### Changed
- **D04 @ 300k hybrid:** **2.993 BPC** (Moby Dick); figures and `d17.json` 17K char-shuffle fields updated.
- **Cypha Tests 1A:** passes @ char shuffle (+4.54 BPC @ 300k hybrid); block shuffle still flat.

---

## [Unreleased] — 2026-06-07 CyphaLM hybrid + long-range

### Added
- **`hybrid_gria_lstm`** context mode — GRIA + char-LSTM dual head with online blend (`cypha_lm/model/char_lstm_head.py`).
- **Long-range context suite** — `cyphalm_long_range.py`, `cyphalm_long_range_suite.py`, D17 **17K**, `docs/CYPHALM_LONG_RANGE_TESTS.md`.
- **Upgrade V2** — learnable views (`view_embed.py`), gated n-gram fusion (`ngram_fusion.py`); sweeps + D17 **17I/17J**.
- **Phase 1c runner** — `run_d17_phase1c.py`; 300k train cap via `cyphalm_bench_limits()`.
- **Profiles** — `cyphalm_d17_hybrid.json`; D17/D04/llm defaults → `hybrid_gria_lstm`.

### Changed
- **D17 @ 300k:** hybrid **2.873 BPC** (Phase 1c 17A) — beats bigram and char-LSTM bench.
- **`d17.json`**, **`BASELINE_REPORT.md`**, research docs refreshed for hybrid + 17K long-range.

---

## [Unreleased] — 2026-05-31 CyphaLM + LLM features

### Added
- **D04 rewritten for CyphaLM** — char-LM domain now runs Izaac → CellAI SSM → CyphaDIF → GRIA (not raw CyphaDIF + CharNgramEncoder).
- **`bench/adapters/cyphalm_bench.py`** — shared LM helpers for D04/D17: BPC eval, context-length curve, save/restore fidelity, sampling comparison, expert routing trace.
- **D04 experiments:** BPC vs context length, CyphaDIF expert routing during generation, checkpoint round-trip parity, sampling strategy bar chart (`fig04_context_bpc`, `fig04_expert_routing`, `fig04_sampling_strategies`).
- **CyphaLM generation:** `top_p_sample` (nucleus), unified `autoregressive_decode`, `stream_generate` SSE chunks; `predict_next` exposes `routing_probs`, `dominant_expert`, `active_experts`.
- **CyphaStudio LM REST (FastAPI-only):** `POST /lm/load`, `GET /lm/metrics`, `POST /lm/predict_next`, `POST /generate`, `POST /generate/stream` (SSE with epistemic gating).
- **`cypha_studio/core/lm_engine.py`** — `LMEngine` wrapper for CyphaLM inference and streaming.
- **`CYPHA_LM_CHECKPOINT`** env var — auto-load CyphaLM at FastAPI startup.
- **`tests/test_lm_api.py`** — generation utilities + REST route tests.
- **`examples/lm_generate_body.json`**, `curl_lm_generate_stream.sh/.ps1`.
- **`docs/port/PORT_CONTRACT.md` §4** — CyphaLM REST contract.
- **CyphaStudio GUI CyphaLM chat** — File → Load CyphaLM…; streamed char generation in chat with expert/epistemic status (`lm_generation_worker.py`, `chat_widget.py`).
- **`scripts/generate_demo_lm_checkpoint.py`** — trains a tiny demo checkpoint for Studio/REST smoke tests (`examples/demo_cyphalm/README.md`).
- **CI** — `pip install -e cypha_lm/`, scoped ruff, `cypha_lm/model/tests/` in pytest matrix.
- **Installers** — Windows/Linux scripts run `test_lm_api` and document demo checkpoint generation.

### Changed
- **`cypha_lm/model/cypha_lm.py`** — `generate()` accepts `strategy`, `top_k`, `top_p`; adds `stream_generate()`.
- **Documentation:** `cypha_lm/README.md`, `bench/README.md`, `cypha_studio/README.md`, `docs/studio/CYPHA_ENV.md`, `examples/README.md`, `docs/RESEARCH_STATUS.md`.
- **D10 time-series tuning** — ECG passes 4→8, encoder window 50→32, n_fft 10→16.
- **`DEFAULT_CYPHALM_CONFIG`** — `gria_lr: 0.06`, `online: True` for bench training.
- **Full D04/D17 benchmark refresh** — updated figures, tables, and `BASELINE_REPORT.md`.

---

## [Unreleased] — 2026-05-31 polish pass

### Added
- **`docs/RESEARCH_STATUS.md`** — canonical research journal: 17-domain benchmark table,
  confirmed properties, hard limits, hypothesis ledger, and forward research map.
- **`docs/reports/`** — permanent archive of investigation reports; all bench and diagnostic
  reports copied here (`BENCH_TUNING_REPORT.md`, `BENCH_ARCH_TUNING_REPORT.md`,
  `BENCH_ARCH_RESCORE_REPORT.md`, `BENCH_UPGRADE_REPORT.md`, `BENCH_PAPER.md`).
- **`CHANGELOG.md`** (this file).
- **Package READMEs:** `cypha_som/README.md`, `cypha_accel/README.md`;
  major rewrite of `cypha_lm/README.md` (architecture table, D04 bug warning,
  known limitations, configuration guide, empirical results).
- **`bench/README.md`** — 17-domain structure, run instructions, single-domain usage.
- **`examples/`** — `README.md`, `cypha_update_body.json`, `cypha_load_body.json`,
  `cypha_adapt_temperature_body.json`, `curl_predict.sh`, `curl_predict.ps1`.
- **`packaging/`** — `install_windows.ps1`, `install_linux.sh`, `README.md`.
- **`docs/studio/CYPHA_STUDIO_MASTER_PLAN.md`** — historical stub fixing broken links.
- `cypha_diagnostics/README.md` — explains package purpose and confirmed findings.
- Proper `[project]` metadata in `pyproject.toml` (name, version, description,
  dependencies, optional extras `studio`, `gpu`, `dev`).
- `cypha_lm/` submodule `__init__.py` docstrings (embeddings, temporal, model,
  expert_field, projection, analysis).
- `native/CMakeLists.txt` VERSION field (`0.1.0`).
- CI `concurrency` group (cancel redundant PR runs), `PYTHON_VERSION` env var,
  job/step names for clarity.
- `Makefile` full `.PHONY` declaration and `help` target.

### Fixed
- **D04 benchmark bug** — `d04_generation_language.py` was indexing `probs[char_id]`
  into a label-ordered probability array, yielding a nonsensical 33.2 bpc floor.
  Fixed to map via `memory._classes.keys()` order. All documentation references to
  "33.2 bpc CyphaLM failure" updated to reflect this was a benchmark bug.
- **D04 clarification propagated** to `CHANGELOG [1.0.0]`, `DIAGNOSTIC_REPORT.md`,
  `docs/FUTURE.md`, `README.md`, `cypha_lm/README.md`.
- `bench/BASELINE_REPORT.md` header clarified: this is the **post-diagnostic
  tuned** run, not a default-parameters baseline.
- `docs/port/PORT_FULL_STACK.md` M6 ExperimentDB API checkbox marked complete.
- 36 `pytest.importorskip` calls in `test_api_contract.py` and
  `test_cypha_rest_smoke.py` now include `reason=` strings.

### Changed
- **`docs/port/PORT_CONTRACT.md`** updated: Kernel LLR (Python-only), deliberation
  (Python-only, default OFF), `gh_infer` vs native `use_gh` gap, `/session/rng` routes,
  `field_sr_vec` key, fixture staleness note (2026-05-30 additions don't affect fixtures
  when generated with defaults).
- Moved `results/SOM_UPGRADE_REPORT.md` → `docs/reports/`.
- Moved `cypha_diagnostics/DIAGNOSTIC_REPORT.md` → `docs/reports/`.
- `scripts/README.md` extended with 15+ previously undocumented scripts (parity
  fixture generators, CyphaLM/SOM eval runners, analysis utilities, PS extras).
- `docs/verify/VERIFY_PLAN.md` — updated from single Ubuntu CI job to 2 jobs;
  `.sh` script notes.
- `docs/verify/VERIFICATION_STATUS.md` — 2 CI jobs, Not-in-CI table for
  `cypha_som/tests/` + `cypha_lm/`, D04 note, full 24-dir fixture inventory,
  `native_generation` in CTest list.
- `docs/verify/ROADMAP.md` — Phase 5 now lists Kernel LLR as top priority;
  Phase 6 entry added for CyphaLM (D17: 4.50 bpc).
- `CONTRIBUTING.md` — install scripts, `cypha_lm`/`cypha_som` test commands,
  2-job CI, `RESEARCH_STATUS.md` link.
- `docs/README.md` — research status section, bench report index, benchmark commands.
- `fixtures/README.md` — missing fixtures added (memory_train, preprocessor,
  f_field, regression_m4, rff_regression, two_stage_*, generation, registry_register).
- `benchmark.py` and `benchmark_baseline.py` — expanded docstrings and `--help`.
- `native/README.md` — M6 experiments section updated to reference CRUD parity.
- `.gitignore` — added MNIST `*.gz` files; removed `*.sh` rule; deduplicated entries.

### Removed
- Stale planning files: `CyphaDIF_TestBench.md`, `CyphaLM_Plan.md`,
  `cypha_diagnostic_plan.md`, `cypha_som_upgrades.md` — outcomes live in
  implemented packages and `docs/reports/`.
- Raw experiment JSON from `results/` and `cypha_diagnostics/results/`
  (ephemeral; regenerate via `scripts/run_som_upgrade_eval.py` and
  `cypha_diagnostics/run_diagnostics.py`).

---

## [1.0.0] — 2026-05-30 · commit `1dbfa13`

**Cypha comprehensive upgrade — diagnostics, CellAI, KernelLLR, multi-pass**

### Highlights
- **Three root-cause bugs found and fixed** via the full diagnostic plan
  (`cypha_diagnostics/run_diagnostics.py`):
  - Bug 1: Deliberation band `[0.4, 0.6]` was masking ~40% of predictions as
    `__unknown__` on binary problems. Fix: `deliberation_lo=1.0, deliberation_hi=0.0`
    (disabled). Effect: +23.5 pp on S1_2class_linear; regression R² −0.007 → 0.756.
  - Bug 2: `delta_lr=0.06` was too aggressive. Fix: `delta_lr=0.03`. Effect: +4 pp
    on R3_digits (0.882 → 0.922).
  - Bug 3: `VectorEncoder` inadequate for `input_dim ≤ 30`. Fix: auto-select
    `RFFEncoder(D=256)` for small inputs. Effect: +14 pp on S1_2class over
    `VectorEncoder` alone.
- **+23.5 pp** on linearly-separable 2-class benchmark; **+20.5 pp** on digits.
- SOM upgrade evaluation completed (`scripts/run_som_upgrade_eval.py`): all six
  upgrades benchmarked. Verdict: default flags remain OFF (U2 hurts accuracy;
  U1 doubles latency with no benefit; U3/U5/U6 are neutral on classification).
- Multi-pass training (`n_epochs` from profile) used in D01 domain loop: ~3 pp gain.

### Architecture — confirmed limits
- **XOR / nonlinear boundaries:** FDR=0.001, kernel(h)=0.835 vs linear(h)=0.512.
  Gap of 32.3 pp is a hard LLR-linearity ceiling. **Requires Kernel LLR (Nyström)**
  to close. Highest-priority future upgrade.
- **D04 char-LM metric is a benchmark bug** — D04 runs `CyphaDIF + CharNgramEncoder`
  (not CyphaLM); "33.2 bpc" is caused by wrong probability indexing (`probs[next_idx]`
  indexes by char ID into a label-ordered array, hitting the `1e-10` floor → -log2(1e-10)
  = 33.2). The SGD "0.66 bpc" is cherry-picked from step 1000 (final SGD is 1.51 bpc).
  Real CyphaLM evaluation: **D17 held-out BPC = 4.50** (bigram baseline 3.69).
- **CellAI / D10 ECG:** 17–20% accuracy on 5-class time-series; temporal SSM domain not yet tuned.

### Files changed
- `bench/config/everyday_profile.json` — deliberation disabled, delta_lr=0.03.
- `bench/adapters/bench_models.py` — auto-RFF for `input_dim ≤ 30`.
- `bench/domains/d01_statistical_baselines.py` — multi-pass with `n_epochs`.
- `cypha_diagnostics/` — new diagnostic package (`run_diagnostics.py`, `apply_upgrades.py`).
- `cypha_som/` — SOM/GNG/GRIA/Hebbian/temporal hooks (all flags OFF by default).
- `benchmark_baseline.py` — baseline runner for SOM upgrade evaluation.
- `scripts/run_som_upgrade_eval.py`, `scripts/run_cypha_lm_report.py`,
  `scripts/merge_final_profile.py` — new utility scripts.

---

## [0.9.0] — 2026-03-31 · commit `8945c95`

**Refactor, docs cleanup, Qt streaming training thread**

### Highlights
- Qt shell streaming training: `QThread` worker emitting `lossReported` and
  `valAccReported` signals; live loss chart updates every N steps.
- Documentation pass across all `docs/` subdirectories.
- Minor refactors to `Cypha.py` and `cypha_studio/` for consistency.

---

## [0.1.0] — 2026-03-31 · commit `127bbe9`

**Initial commit — Cypha native port M1–M6 complete**

### Summary

First committed state of the project. All six native port milestones signed off:

| Milestone | Description |
|-----------|-------------|
| M1 | Inference kernel: encode + LLR + GH gate + softmax vs `fixtures/`. |
| M2 | Registry + preprocessor: fit (scale/PCA), transform, CSV load. |
| M3 | Online `train_step`: DIF, GH, replay, NIG, context, OOD. |
| M4 | Regression stack: MKE / RFF / two-stage / ridge / EMA. |
| M5 | `cypha_rest` native server + Qt shell (`cypha_qt_shell`). |
| M6 | Experiments DB (SQLite amalgamation) + Qt M6 panel. |

- **188 pytest + 33 CTest** cases across 13 named parity fixtures.
- Python reference (`Cypha.py`, `cypha_studio/`) serving as golden spec.
- `cypha_lm/` research package (LM stack, embeddings, SSM, experts).
- `bench/` evaluation harness (17 domains, encoders, reports).
- `cypha_som/` optional SOM/GNG/GRIA hooks.
- `cypha_accel/` CuPy-accelerated LLR / projection / NIG helpers.
- GitHub Actions CI: **four blocking jobs** — Linux CTest + pytest, MinGW PE, MSVC + CUDA, GCC + CUDA.

---

[Unreleased]: https://github.com/odin-loki/Cypha/compare/v2.5.0...HEAD
[2.5.0]: https://github.com/odin-loki/Cypha/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/odin-loki/Cypha/compare/v2.3.25...v2.4.0
[2.3.25]: https://github.com/odin-loki/Cypha/compare/v2.3.24...v2.3.25
[2.3.24]: https://github.com/odin-loki/Cypha/compare/v2.2.8...v2.3.24
[2.2.8]: https://github.com/odin-loki/Cypha/compare/v2.2.7...v2.2.8
[2.2.7]: https://github.com/odin-loki/Cypha/compare/v2.2.6...v2.2.7
[2.2.6]: https://github.com/odin-loki/Cypha/compare/v2.2.5...v2.2.6
[2.2.5]: https://github.com/odin-loki/Cypha/compare/v2.2.4...v2.2.5
[2.2.4]: https://github.com/odin-loki/Cypha/compare/v2.2.3...v2.2.4
[2.2.3]: https://github.com/odin-loki/Cypha/compare/v2.2.2...v2.2.3
[2.2.2]: https://github.com/odin-loki/Cypha/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/odin-loki/Cypha/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/odin-loki/Cypha/releases/tag/v2.2.0
[1.1.0]: https://github.com/odin-loki/Cypha/releases/tag/v1.1.0
[1.0.0]: https://github.com/odin-loki/Cypha/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/odin-loki/Cypha/compare/v0.1.0...v0.9.0
[0.1.0]: https://github.com/odin-loki/Cypha/releases/tag/v0.1.0
