# Handoff for Claude Code: Cypha / CyphaLM

**Odin Loch** · rewritten 2026-10-28 at the end of the second working session of the day; updated the same day for B4c, S1, RAM M10–M15, S4, S6 and A4c, and on 2026-10-29 for S7–S9, S15, A4d, M16, M17, S4b and S8d · repo `odin-loki/Cypha` (branch `main`)

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
| **S7b loop calibration** | Three-way likelihoods (T4 silence BF 0.64; a T4 match 29.4), the store's φ for T1 (s 0.755 → 0.822), BH among the supported; the loop decides 23% (S7: 3.5%); confirmed on S15 dev (0.956) | `S7_HYPOTHESIS_TESTS.md` |
| **S8b G6 / G7 link prediction** | Hits@10 0.334 test (base 0.176); BH promotes nothing under S7's loop (gate not measurable); S15 b / c reachable 6–9%; S6's gate missed on S15's pairs (0.658) | `S8B_LINK_PREDICTION.md` |
| **S8a G2 rules, G14** | Rules readable (taxonomy, inverse, path); Hits@10 0.230; promotions 0.818 (S15 grader, n 11); gate not readable on the 2006-text store; G14 parked; T2 fires but is cut (never negative) | `S8A_RULES_AND_CONTRADICTIONS.md` |
| **Slow-test cut** (Odin) | 35 hybrid-era tests and 5,255 lines removed; the canary test fixed (it measured a cold model); `cypha_slow` 91 → 56 | `SLOW_TEST_TRIAGE.md` |
| **S7c batch rule** (Odin: "not up to scratch") | Hybrid: Known = null-model BH over the whole batch (FDR ≤ 0.1 if the null likelihoods are right), Believed = the other supported; 23% decided on dev at 0.974 (e-BH 11%); a fresh set at 0.954; with all hypotheses false it promotes 0.1 per batch (S7b's step: 34) | `S7_HYPOTHESIS_TESTS.md` |
| **S9 worlds, fragility** | I9 holds (43,409 worlds); T2 through worlds never says no (cut again); 76.9% of held facts are single points of failure; Wikidata's blast radius 62% | `S9_WORLDS_AND_FRAGILITY.md` |
| **S8c evidence sources** (Odin: "get another evidence source") | DBpedia infobox reliable and independent of Wikidata (yes right 30 / 31 on the generators' hypotheses); Wikipedia leads fail like co-mention; S8's gate still not passed (15 graded, 0.667) | `S8C_EVIDENCE_SOURCES.md` |
| **M16 FM index in place** | Built `qa` A0 132, defaults 170, C1L + defaults 185 MB (was 175 / 194 / 235); `context8k` 202; lossless | `RAM_PROFILE.md` |
| **A4d similar-context recall, online keys** | Probe 128 (recall 0.94) and keys from the text being read: fresh −0.145, SCRIBE not falling, answers unchanged, 1.87× scoring time; `qa` holdout 1.8269 → **1.7895** | `A4_SIMILAR_CONTEXT_EXPERT.md` |
| **hp-wrapper fixes** (Odin: "fix both") | Forecast rollouts read the trained model (8 min → 6 s); vocabularies other than 256 rejected; `cypha_slow` 56 / 56 | `SLOW_TEST_TRIAGE.md` |
| **S8d promotion** (decided by measurement, ledger 71) | `decide` needs an independent yes; sources promote only where LR+ ≥ 10 on the generators' errors (DBpedia dates, places, links; GeoNames places); a fresh read: 25 graded at 0.880 [0.700, 0.958], gate (0.9) not met; the 3 misses are S15 name or grain mismatches | `S8D_PROMOTION.md` |
| **S4b unknown-unknown signals** | One alarm rule; the vocabulary signal alarms on new page types (95.5% within 50 pages); surprise and unparsed do not pass; nothing sees an unseen domain in plain prose | `S4_UNSEEN_AND_FRONTIER.md` |
| **S4c out-of-scope signal** | No candidate passes: topic distance 38% within 50 pages on Dolly (gate 90%); the markup control holds; Dolly's long answers are mostly in scope by content | `S4_UNSEEN_AND_FRONTIER.md` |
| **S10 curiosity scheduler** | Gate not met (LCI 0.000): dev +126%, gate −5% (2.105 against 2.227); the gate set's value sits in one small arm that uniform also empties, and the highest-P0-first order reversed; uniform stays | `S10_CURIOSITY_SCHEDULER.md` |
| **S11a G9 concepts, G11 open relations** | G9's gate not met (0 of 60 classes; nearer than the null on 50); no promotions. G11: 21 of 38 proposals hold (0.861), educated_at among them | `S11A_CONCEPTS_AND_OPEN_RELATIONS.md` |
| **S11b G12 ABC** | Gate not met (LCI 0.000): open-form P@100 0 (degree 0.010); G12 removed by its rule 3; the closed form works (AUROC 0.68 against 0.47) | `S11B_ABC_DISCOVERY.md` |
| **S11c G13, G3, G4, G5** | G3 kept (recall@10 0.840); G4 and G5 earn their place (MRR +0.068, +0.029); G13 parked (real verdicts 0.647, baseline 0.706), but every fact it questioned was decided right by the sources (21 of 21); promotions 11 of 11 | `S11C_ANALOGY_ANOMALY_ABDUCTION.md` |
| **M17 lowered index freed after `--save`** | C1L + defaults with `--save` 204.5 → 186.7 MB; lossless | `RAM_PROFILE.md` |
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
| `qa` holdout, default model | trie + IG 2.0729 → **1.7895** with class + similar-context + copy (defaults since B4c, off inside answers; A4d's similar-context default; A4c read 1.8269, B4c 1.8345); `context8k` 2.0715 → 1.7897 |
| Class expert's table | default `shape` = three orders, compact, seen once dropped (fresh 2.0853; Odin, 2026-10-28) |
| Text, SCRIBE v2 | trie 0.014, trie + class 0.002, `sample`'s default (trie + similar-context + copy) 0.063 [0.051, 0.078] (A4d; A4c 0.064, B4c 0.056), human 0.770 |
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
- **Building** (M15, M16): built `qa` peaks at 132 MB (A0), 170 MB (defaults) and 185 MB (C1L + defaults), repeating within 0.2 MB; `context8k` 202 MB. The index builds in place, and the class table is built after it from `FmInfiniGram::text()`. With `--save`, C1L + defaults peaks at 187 MB (M17: the lowered index is freed once the store is written; it was 205).

### 4.4 CTests

Run these on every change: `native_cyphalm_trie_selftest`, `native_fm_infinigram_test`, `native_epistemic_test`, `native_thought_test`, `native_geometry_test`, `native_ecs_lci_test`, `native_scribe_selftest`, `native_cyphalm_neural_smoke`, `native_class_context_test`. `native_fm_infinigram_test` runs 191,885 checks (locate included), `native_epistemic_test` 1,293 (S4, S4b, S4b-b, S4c, S6, S7, S7b, S7c, S8a, S8b, S8c, S8d, S9, S10, S11a, S11c), `native_class_context_test` 1,043 (M15). `pattern_learn` builds with `g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp`; its S3 `--gaps` run must stay byte-identical. Since S11c it uses `std::thread`: older glibc (before 2.34) needs `-pthread`. `pattern_learn --signals DIR` (S4b) runs after S3's passes; without it the run is byte-identical. `--signals-scope dev|test` (S4c) runs after S4b's signals; without it the output is S4b's.

### 4.5 CI

- All six jobs of `ci.yml` passed on 2026-10-28. Two causes of red were fixed: `cypha_gate.cpp` included `unistd.h` (Windows MSVC; the tool is deleted), and `native_cyphalm_neural_smoke`'s kernel-parity limit was a flat 1e-6 on log-probabilities down to −15, inside float32 rounding (now scaled by 1 + |log p|; planted kernel bugs still fail it).
- **Outside CI:** all 56 `cypha_slow` tests pass (2026-10-29: the hybrid-era cut, the canary fix, and the forecast and vocabulary fixes). CI runs `-LE cypha_slow`.

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
10. ~~hp wrapper semantics.~~ **Fixed 2026-10-29** (PLAN ledger 69). New: forecast's event vocabulary has 842 ids, but hp reads bytes, so an event id ≥ 256 (any theater after Taiwan) throws during training. Map events onto bytes, or give forecast a model that is not byte-level?
11. **S15 defaults, open to Odin's veto:** the Wikidata response cache (2.35 MB) is committed for audit; D4's T-QA exclusion is kept, so the benchmark is thin on CyphaLM's own 8 MiB slice (105 train subjects).
12. **S7c's choice, open to Odin's veto** (PLAN ledger 70, answering ledger 68): the Known tier is null-model BH (K4), chosen over conformal selection (K3) by the registered tie order; they promoted the same sets on every read. K3's guarantee needs no model, only a labelled calibration set drawn like the batch; K4's rests on the null likelihoods, which run high for plausible false hypotheses. Switch to K3? Should fusion also use the store's φ (S7b)?
13. **S8's promotion gate** (ledger 67, 71; S8c and S8d done). `decide` now needs an independent yes, and a source promotes only where its LR+ on the generators' own errors is at least 10 (decided by measurement, open to veto). A fresh confirm-half set, read once: 25 graded at 0.880 [0.700, 0.958], so the gate (0.9) is not met. The 3 misses are S15 name or grain mismatches: Brussels for City of Brussels, Los Angeles for Century City, Chicago for Edgewater Hospital. Open:
    - Accept the miss, or register a grader that knows Wikidata's names and places inside places, then read a new set?
    - Read the gate on Supported (as now) or on Known (empty at this calibration size)?
    - Should a source need at least 10 graded positives before it may promote? GeoNames passed on 2.
    - Grade more DBpedia occupations and genres? They failed on 29 wrong values.
    - DBpedia's data licence.
14. **S9's questions:** DISCOVERY §4.4 use 1 (answering Odin's what-if questions) breaks rule 10, so nothing answers; drop it, or keep cards as data only? May a confirm-only test (T2: LR 3.35 dev, 10.5 S15) enter the loop by its three-way LR? Count Wikidata as one source or per claim (its 62% blast radius)? Merge S9's single points of failure into S3's gap map (it changes S3's output)?
15. **When the 1 TB run ends:** move `trie_stream_train.cpp` onto `packed_trie.hpp` (and `pack()`, M4).
16. **S4b's alarm threshold:** runs of pages on one topic raise false alarms (taxonomy pages for vocabulary, American football pages for surprise; 1 each in about 1,600 pages). Should such runs open gaps? A threshold of 100,000 instead of 20,000 silences both, at about 5 more pages of delay for the vocabulary signal.
17. **S4c's questions:** should the out-of-scope gate be read on Dolly's own-voice answers (brainstorming, creative writing) or on another corpus, since most of S4b's Dolly pages are encyclopedic prose? Adopt TF-IDF novelty beside the vocabulary signal for new page types (100% within 50 pages, median 16, against 95.5%, median 24)? Add a scope gap kind to DISCOVERY §3.2?
18. **S10's questions:**
    - A scheduler can only beat uniform where value is spread over large arms with different rates. Re-read S10 when the generators give larger pools (after §5.13's grader, or with S11), with a budget fixed in advance that does not empty the arms?
    - Order within an arm: highest P0 first won on dev and lost on the gate. Keep random order until the link between P0 and promotion is measured per arm?
    - S15's rule names only generators' final reads of test. S10 read test's bit-17 half (subjects and relations to build the set, the labels once; bit 17 = 0 untouched). Does a scheduler's read count, or where should such reads come from?
19. **S11a's questions:**
    - Name G11's 21 holding proposals, starting with educated_at, and hand them to F2 as new relations or phrasings? P10 needs Odin or the teacher for this.
    - G9's bar (Jaccard ≥ 0.5) cannot be reached on an occupation-heavy store. Accept the miss, or re-read on a richer store (for example with S15's added relations as attributes) with fresh held-out classes?
    - G11's dev choice between C1 and C2 was decided by date links (344 of C1's 419 holding pairs). Without them C2 wins (87 against 78) and proposes developer, headquarters, member_of, spouse and country. Register the skip with the choice as it falls, and read S15 test once (S11a-b)?
20. **S11b's questions:** keep G12's closed form (AUROC 0.68 on test) as a support term in S6's P0 or as a test kind? It needs its own registered read. A wider grader for open ABC (any Wikidata property between A and C, not S15's 34) would need new Wikidata fetches.
21. **S11c's questions:** use G13's anomalies only as questions for the source stage, never as verdicts (through the sources they were right 21 of 21)? Keep A6 (spouse gaps) for a larger store, or cut it? Drop G3's rule term from its order (the link predictor alone ranks the right fact first more often)? Stop G4 carrying dates as constants (every carried date was wrong)? Both halves of S15 test now carry a pool read, so a future fresh set needs new slots (S8e builds them).

---

## 6. Next work, in order

1. **S8, promotion:** after Odin's calls (§5.13): a grader that knows Wikidata's names and places inside places, then a new fresh read; then S9's consequence tests need confident rules into single-valued slots (a larger store); S10 waits for larger pools (§5.18). (S10, S4c, S8d, S4b, M17, S7b, S7c, S8a, S8b, S8c, S9, M16, A4d, the hp fixes and the slow-test cut were done 2026-10-29; S1, S4, S6, S7, S15, the echo fix, A4c, M15 and serving from stored files 2026-10-28.)
2. **RAM:** nothing open. Every built and served `qa` configuration is under 190 MB (M17). A blockwise BWT only if a larger corpus needs it.
3. **Forecast's event vocabulary** on a byte model (§5.10), after Odin's call.
4. **Similar-context expert:** a cap or window on online keys before streams pass a few MiB (they cost 5.3 MB per MiB read). **S4c:** no content signal sees Dolly, because most of it is in scope; a new planted pool needs Odin's call (§5.17). Source-trust shifts once source reliability is refit over time.
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
