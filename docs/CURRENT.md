# Where the numbers are

The language model being taken forward is the packed byte trie. hp gate24 stays in the tree as a comparison. It scores better on a small slice and it costs too much RAM to train on a terabyte or to keep loaded for day-to-day use.

**The default plan is now [`PLAN.md`](PLAN.md)** (2026-10-26). The 1 TB run below continues inside it as Track A, with its predictions registered there. The question-answering and 8K count-context results are in [`reports/CYPHALM_QA_AND_CONTEXT_TEST.md`](reports/CYPHALM_QA_AND_CONTEXT_TEST.md).

## Packed trie (the model to train)

Tools: `native/tools/cyphalm_trie.cpp` (`bpc`, `chat`, `probe`, `context8k`, `sample`, `selftest`) over the one model in `native/include/cypha/cyphalm/packed_trie.hpp`. On 2026-10-27 it replaced the forked `trie_*.cpp` tools and reproduces their numbers to the printed digit. The experiments behind the 2.286, 2.642 and word-trie rows (`trie_together`, `trie_gria_gate`) were retired; their results and the git commits that hold them are in [`reports/B1_TRIE_CONSOLIDATION.md`](reports/B1_TRIE_CONSOLIDATION.md). Surface score: `native/tools/surface_score.cpp`.

Protocol for the numbers below: enwik9 bytes `[0, 7340032)` train, `[7340032, 8388608)` frozen holdout, unless a row says otherwise.

| Read | BPC | What it means |
|------|----:|---------------|
| Fixed backoff, orders 1–4, 4-byte match | 2.386 | Counts only, alpha 0.4 |
| Learned mixture, weights frozen | **2.298** | About **0.93** on order 4, **0.06** on the match |
| Same mixture, weights still adapting on the holdout | 2.286 | The router is learning the test bytes. The counts are not. |
| PPMD, longer chains, or inheritance | ~2.42 | Worse than backoff |
| GRIA as a probability gate | 2.642 | Worse than backoff |
| Packed form (M4, `PackedModel`) | — | **3.9 MB** (bpc model) and **7.7 MB** (chat model), bit-identical to the training form in Synth mode; lookups about 5× faster ([RAM report](reports/RAM_PROFILE.md)) |
| After 15,011 Dolly turns | 2.341 wiki | Stored accounting about **11 MB** |
| + class-context expert, second half of the holdout (B4) | 1.999 / 1.982 | Over trie + InfiniGram (2.0725): shape classes / E1 tags, tail 6. `cyphalm_trie cls`. On fresh text 2.166 → 2.088 / 2.075 |
| Class expert's default table (Odin, 2026-10-28) | 2.0853 fresh | Three orders, compact, contexts seen once dropped (`shape` = `shape:6:3:cp2`); the B4 rows above are `shape:6:2:f` |
| + copy expert, second half of the holdout (A1c) | 1.927 | Over trie + InfiniGram (2.072), chat model; `qa --copy`; an option, not a default |
| + similar-context expert (A4), second half of the holdout | 1.939 | Over trie + InfiniGram + class (1.999); fresh text 2.088 → 2.027. `cls --sim`, `qa --sim`; an option, not a default ([report](reports/A4_SIMILAR_CONTEXT_EXPERT.md)) |
| Similar-context expert, default in `qa` / `sample` / `context8k` (A4b) | 1.9686 | `qa` holdout trie + IG 2.0716 → 1.9686; off inside answers, so answers are unchanged; `sample` SCRIBE v2 0.014 → 0.061 (still word salad); `--no-sim` turns it off |
| **Similar-context expert: distinct contexts stored once, k 8,192 (A4c, 2026-10-28)** | **1.8269** | `qa` holdout with class + copy (B4c: 1.8345); fresh trie + IG + class + sim 2.0275 → 2.0235 (−0.0040 [−0.0048, −0.0032]); the expert 40.0 → 19.3 MB; answers unchanged ([report](reports/A4_SIMILAR_CONTEXT_EXPERT.md), A4c) |
| **Similar-context expert: probe 128 and online keys (A4d, 2026-10-29)** | **1.7895** | `qa` holdout with class + copy (A4c: 1.8269); `context8k` 1.7897; fresh trie + IG + class + sim 2.0235 → 1.8783 (−0.145 [−0.176, −0.118], read once); neighbour recall 0.65 → 0.94; online keys (contexts already read join the index; checked causal) cost 5.3 MB per MiB read; scoring 1.87×; answers unchanged ([report](reports/A4_SIMILAR_CONTEXT_EXPERT.md), A4d) |
| **Class + similar-context + copy, defaults (B4c, 2026-10-28)** | **1.8345** | `qa` holdout (C1L chat model) trie + IG 2.0729 → 1.9944 (+ class) → 1.9224 (+ similar-context) → 1.8345 (+ copy, m 8, window 16,384); `context8k` the same. All three are off inside answers, so answers are unchanged (dev 520 / 520 rows, frozen test 800 / 800). `sample` takes similar-context + copy; the class expert only with `--cls` (SCRIBE v2 falls with it). `--no-cls`, `--no-sim`, `--no-copy` ([report](reports/B4C_ECHO_GATE.md)) |

Surface scores of trie samples are about **0.74–0.83**. That is readable fragments, not answers.

BPC is held-out `−log2 p`. It is not a generation score.

### What the measurements settled

- Context hashes are spread out. The bytes underneath are not. Order-2 prefixes on 2 MiB had about **6.2** extensions on average, and **47%** of order-4 contexts occurred once.
- Enumerating context masks does not shrink the tables.
- `32/(32+n)` and NIG `r_eu` die after many counts. They are not mix weights. The uncertainty used on the match is `u = H(order-1) / ln(256)`, about **0.49** on this holdout.
- A word-sized trie added as another expert fell to weight **0.0005** and did not change the samples.
- Dolly questions that occur byte-for-byte in the stored chat are copied back. The copy is a search for that prompt, not the order-4 model reading the question. An unseen question stays Wikipedia-shaped. A `<|doc|>` marker is written between the Wikipedia bytes and the chat so those contexts do not glue together.

### Memory (qa model, 8 MiB protocol)

[`reports/RAM_PROFILE.md`](reports/RAM_PROFILE.md), 2026-10-28. The qa model is the trie on wiki train and two Dolly passes, with InfiniGram and the class expert over the training text (19.6 MB).

| Part | Before | Now | How |
|---|---:|---:|---|
| Trie | 38.5 MB | **7.7 MB** | `pack()` (M4), bit-identical |
| InfiniGram | 77 MB | **26.0 MB** | `FmInfiniGram` (M5), the same answers |
| Class expert, 3 orders | 77.7 MB | **18.2 MB** | compact and seen-once dropped (M2, M3): +0.0009 BPC on fresh text, surface S unchanged within its interval |
| Class expert build, 3 orders: memory it adds at its peak | 560 MB | **132 MB** | chunked, each order merged on its own (M1), identical table |
| Whole qa model, resident | 228 MB | **76 MB** | M2–M5 together (scratch harness) |
| `cyphalm_trie qa`, peak | 264 MB | **171 MB** | M9: the tool on M4–M6, every number unchanged (C1L 368 → 199 MB; `context8k` 230 → 165 MB) |
| `chat` served from a stored model | 104 MB, 476 s (trains) | **21 MB, 12 s** | `chat --model FILE` (M7) |
| `cyphalm_trie qa` served from a store, peak | 186.8 / 245.3 / 275.1 MB (A0 / + class + copy / C1L + class + copy, built) | **138.4 / 169.4 / 189.1 MB** | `qa --save DIR`, then `qa --model DIR` (M10–M14): no training, no index or class build; parts load in 0.15 s; 3–5× faster; every printed number and the rows file unchanged |
| `cyphalm_trie qa` built, peak | 175 / 194 / 235 MB (A0 / E / C1L + E, after M15) | **132 / 170 / 185 MB** | M16: both BWTs built in the text's own buffer, trees filled in one pass, the class table built after the index from `FmInfiniGram::text()`; identical index files and rows; `context8k` 226 → 202 MB. With `--save`, C1L + E 204.5 → 186.7 MB (M17: the lowered index is freed once the store is written) |

### What grows, and what must stay separate

Training keeps a node for every context, including ones seen once. The **5.4 MB** figure is the later packed pass (drop counts below 2, one byte per kept count). It is not the resident size during the pass. On the 8 MiB slice the node and follower vectors are about **25 MB**, and the process peaks near **116 MB**.

Counted across all **10⁹** bytes of enwik9, distinct 4-byte contexts do not grow in proportion to the file:

| Bytes | Distinct 4-grams | Seen once | Peak training RAM |
|------:|-----------------:|----------:|------------------:|
| 8 MB | 331,145 | 43% | 116 MB |
| 64 MB | 979,017 | 41% | 128 MB |
| 256 MB | 1,865,580 | 39% | 256 MB |
| 10⁹ | 3,222,263 | 37% | 400 MB |

From 512 MB to 10⁹ bytes the 4-gram count grows like about N^0.47. The packed upper bound at 10⁹ bytes is about **45 MB**. A gigabyte of Wikipedia-like text fits this trie. The match record is a `uint32` offset into the file, so a corpus past 4 GiB needs the next byte stored in the record instead of an index.

Chat must not be poured into those counts. On the 8 MiB split, the 4-byte match fires on **96%** of holdout bytes, and **75.7%** of holdout bytes use a 4-gram that also occurs in Dolly. The match table keeps only the latest follower, so those wiki predictions become the last Dolly byte. Every Dolly prompt also ends in the same 12 bytes, `\nAssistant: `, so a fixed 8-byte or 16-byte key does not tell the questions apart (0 of 15,011 prompts are unique at 12 bytes; **19%** are unique at 16). The chat index has to be the whole prompt pointing at its answer, in its own table, after the wiki counts and the mixture weights are frozen.

## Generated-text quality (SCRIBE v2)

Tool: `native/tools/scribe.cpp` over `native/include/cypha/eval/scribe.hpp` ([report](reports/SCRIBE_V2.md), [paper](../paper/scribe/SCRIBE_paper.md)). Counts only; scores a text by its distance from human reference text of the same genre and length.

| Read | Value |
|---|---|
| Control ladder, fresh Wikipedia (read once) | 5 of 5 pairs (0.975, 0.999, 0.998, 0.998, 1.000); v1 2 of 5; surface S 1 of 5 |
| Control ladder, web text / books (own references) | 5 of 5 / 3 of 5 |
| Held-out degradations followed | 7 of 9 families (blind to sentence order and truncation) |
| CyphaLM trie samples / trie + class context / human, same prompts | 0.014 / 0.002 / 0.770 (word salad by SCRIBE; surface S said 0.749 / 0.901 / 0.904) |
| `sample` with its defaults: trie + similar-context (+ copy) / + class | A4d's default 0.063 [0.051, 0.078]; A4c's 0.064 [0.052, 0.078]; B4c's 0.061 [0.049, 0.074] (0.056 with copy) / 0.025 (surface S 0.801 → 0.880: the two scores disagree; B4c) |
| Document experts (GEN-C, `--doc`, off) | Held-out bits per byte −0.043 (cache); coherence distance −23 to −35%; dev SCRIBE 0.048–0.060 against the default's 0.063 (sentence form worsens): not adopted |
| Cost | about 1,000 texts/s on one core, about 0.4 GB, 14 s with the 32 MiB pack built |

BPC and surface S are not generation scores. SCRIBE v2 is the one that passes its ladder.

## Question answering, CyphaLM by itself

Tool: `cyphalm_trie qa` (dev 130 questions, frozen test 200; coverage at a 2% wrong rate by 5-fold CV inside dev). Reports: [A0/A1](reports/A0_A1_CYPHALM_QA_AND_DONT_KNOW.md), [A1c](reports/A1C_ENTITY_FAMILIARITY_AND_COPY.md), [A2/A3](reports/A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md).

| Read | Value |
|---|---|
| A0, CyphaLM alone | 0.0% coverage at 2% wrong on dev; raw right 0–2 / 130 |
| A1d don't-know token (span candidates, `ent_title`, `low` only) | dev precision 0.600, recall 1.00; frozen test 0.635 / 0.94 ([report](reports/A1D_A2B_ANSWERING_VARIANTS.md)) |
| Don't-know token with `ent_bold` (ANS, ledger 79) | low-only: frozen test trie precision 0.963 [0.817, 0.993], recall 0.52 [0.385, 0.652] (D2 met for the first time); trie + IG 0.540 / 0.68; gated 0 right, 0 wrong |
| Entity in the tag (ANS, ledger 80, veto recommended) | `--fam-ent` + `--copy-dk`: tags name the question's entity 49 of 50 on test (was 0 of 72); the message never agrees with the tag; 23 of 97 refusals write the frame token |
| With D4's shard (C1L) | dev 2 right, 0 wrong at 2% wrong; test 0 / 0 (the shard covers 3 test questions) |
| R3 with F2R raw frames | dev 2 right / 1 wrong out of fold; test 1 / 0 |
| trie + IG + class (tags) + copy, holdout second half | 1.8437 BPC (trie + IG 2.0735); with the shard it echoed the question until B4c turned the experts off inside answers |
| A2, + fact shard (R1) | dev 2 right / 0 wrong out of fold (1.5%, Wilson 0.4–5.4%); test +0 |
| A3, statement scoring (R3) | 0% at 2% wrong |
| A2 + A3, frozen test, gated | 1 right / 0 wrong |
| Entity familiarity (`ent_title`) | dev AUROC 0.795 [0.745, 0.847] |
| Reason codes on don't-knows (S1) | right 82 / 100 on the dev audit (Wilson 0.733–0.883), 169 / 200 on the frozen test (0.845); CyphaLM's own codes 0.60 / 0.63. The 90% gate is missed and closed there (Odin: "if the data isn't there, it isn't there"). "Unknown" means what CyphaLM has read. `qa --cards FILE` gives each answer its evidence card ([report](reports/S1_REASON_CODES_AND_EVIDENCE.md)) |
| Copy expert (option) | holdout BPC trie 2.114, trie + InfiniGram 1.926 |

## Word vectors

PPMI-SVD (`native/tools/ppmi_svd.cpp`) stays the source: WordSim-353 0.672, SimLex-999 0.255, MEN 0.637 on all of enwik8 (C1 pairs). Cypha's own learner, `cypha_word_classes train --method sgns --epochs 5 --var0 0.3` ([S-1b](reports/S1B_LEARNED_CONTEXTS.md)): 0.683 / 0.246 / 0.669 on the same pairs, 0.750 / 0.270 / 0.756 on the top-10,000 pairs; no worse on common words, worse on rare ones. S-1c (2026-10-29): a context prior lifts Cypha's own vectors on rare words (Card-660 +0.098 over (b)), and on common nouns they are above PPMI-SVD (MTurk-287 +0.066), but verbs fail the margin (VERB-143), so PPMI-SVD stays the source; the §4.2 step is 1.42× faster, bit-identical.

## Embedding teacher (weights only in the container)

Qwen3-Embedding is its base plus a low-rank change: 3.3% of the linear weights' norm on 0.6B, 6.3% on 8B; token embeddings and norms unchanged ([report](reports/QWEN3_EMBEDDING_DISSECTION.md)). 0.6B `embedtexts` runs at 58.3 tokens/s on 2 container threads. The probe set is 135,895 items and 2.46M tokens (`mined/probe/`). Probe vectors and the dynamic dissection run on Odin's PC.

## Fact store and fact text (for CyphaLM to learn from)

Tool: `native/tools/pattern_learn.cpp` ([D4 report](reports/D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md), 2026-10-28). These facts never answer anything. They become training text for CyphaLM (PLAN rule 10).

| Read | Value |
|---|---|
| Text seeds (infobox, table, category) | 7,484 (8,148 before a citation-template seed bug was fixed) |
| Kept patterns, `--cv --wikidata` | 70 (F2 alone: 44) |
| Admitted facts extracted from the text | 1,216 (F2 alone: 832) |
| Agreement with Wikidata on held-out subjects | 0.951, Wilson [0.909, 0.974] |
| Raw-byte patterns (`--raw`, F2R) | 81 kept, 1,025 facts, Wikidata 0.950 [0.904, 0.974]; joined store 1,257 facts (0.951); raw frames + raw shard reach the dev answer for 7/13 eligible questions (cleaned frames + D4 shard: 4/13) |
| Cross-article reading, wildcard and word-class gaps (F2b, [report](reports/F2_PATTERN_LEARNING.md)) | No kept pattern for capital, author or director in six arms (author 16 of 18 checks, two short of 20); the changes add dates; gold-test A1 0.951 [0.890, 0.979]. A person-lexicon leak from Wikidata founder seeds (8 of A1's 27 new facts wrong) blocks any switch. F2c: with `--lexicon-type-check`, Wikidata seeds widen the type lexicons only when Wikidata's type agrees (147 non-human "persons", 60 coordinate-less "places" kept out); admitted 1,216 → 1,209, gold agreement unchanged; S3's gap map `ba221cb2c79cc1ee` with it (off by default until the references are regenerated) |
| Fact shard (`--shard`) | 2,001 question-form rows from 675 facts on pages in the 8 MiB slice; 0 equal to a dev or test question; dev reach 16 / 100 in-slice questions |
| Gap map (`--gaps`, [S3/S5 report](reports/S3_S5_GAP_MAP_AND_HYPOTHESES.md)) | 27,842 gaps (schema 26,333, contradiction 128, certainty 1,342, demand 39); 6,062 not parked; P(exists) Brier 0.0301, calibrated. CyphaLM's dev tags name their question's slot 0 / 750 |
| Unseen estimates (S4, [report](reports/S4_UNSEEN_AND_FRONTIER.md)) | Good–Turing with the page as the unit is within 4% of what held-out pages bring (test half, read once: words 0.968, entities 0.980, facts 1.000); the plain n₁/N undercounts by 20–50%. Store coverage 0.42 of the slots that exist on pages already read (person 0.56, place 0.25, other 0.20); `pattern_learn --frontier` |
| Unknown-unknown signals (S4b, [report](reports/S4_UNSEEN_AND_FRONTIER.md)) | One alarm rule for surprise, unparsed and vocabulary. Planted new page types: vocabulary alarms within 50 pages in 95.5% of runs (median 24); surprise 56%, unparsed 0%. No signal sees Dolly prose (surprise is only 0.13 bits per byte above wiki text). Unshifted stream: at most 1 false alarm per signal in about 1,600 pages. S4b-b: U leaves out linked years and days (eligible sentences −5.9%; no alarm on S, as before) |
| Out-of-scope signal (S4c, [report](reports/S4_UNSEEN_AND_FRONTIER.md)) | Five content candidates on a page's first 50 prose words. The dev choice (topic distance) detects 38% of Dolly shifts within 50 pages on the one test read (gate 90%), with no false alarm on S. Wiki pages with their markup removed are never detected. Dolly's long answers are mostly encyclopedic prose. TF-IDF novelty detects list and disambiguation pages, 100% within 50 (median 16; not gated) |
| Hypothesis scoring (S6, [report](reports/S6_HYPOTHESIS_SCORING.md)) | P0 ranks known-true above known-false at 0.738 [0.717, 0.758] on a fresh planted set (gate 0.75 missed); the store's pairwise rules carry it; the hard date constraint lifts dates 0.55 → 0.66 |
| Hypothesis tests (S7, [report](reports/S7_HYPOTHESIS_TESTS.md)) | Sensitivity / specificity on test, read once: T1 consistency 0.78 / 0.89, T4 corpus 0.99 / 0.83, T5 Wikidata 0.77 / 1.00, T6 compression 0.20 / 0.95, T3 rules 1.00 / 0.67; T2 cut (no consequence rule fired). The §7.2 loop supports 56 (all true) and refutes 115 (92% false); BH at q 0.1 decides only 6% |
| Hypothesis loop (S7b, S7c) | Three-way likelihoods, T1's φ from the store; the batch step is S7c's two-tier hybrid (Known: BH at 0.1 on null-model p-values over the whole batch, FDR ≤ 0.1 if the null likelihoods hold; Believed: the other supported, never Known): dev 573 Known at 0.974, 23% decided (guaranteed e-BH alone: 11%); a fresh set, read once, 679 Known at 0.954 [0.936, 0.968], 29% decided. With every hypothesis false it promotes 0.1 per batch (S7b's step: 34). S7d: T2 stays out (+0.14 points); K4 stays; the store's φ recommended for fusion, default unchanged |
| Link prediction (S8b, [report](reports/S8B_LINK_PREDICTION.md)) | Rediscovery Hits@10 0.334 / MRR 0.203 on test (base rate 0.176 / 0.098); G7's paths carry it. Promotion through S7's loop: none (gate not measurable). S15: 6–9% of unstated and future values are candidates; S6's gate on S15's pairs 0.658 |
| Rules (S8a, [report](reports/S8A_RULES_AND_CONTRADICTIONS.md)) | G2 finds the taxonomy, inverse and path rules; rediscovery Hits@10 0.230 on test; promotions 0.82 on 11 graded; S8's gate cannot be read on the 2006-text store (1 graded promotion); G14 parked |
| Evidence sources (S8c, [report](reports/S8C_EVIDENCE_SOURCES.md)) | DBpedia infobox facts are the one reliable source independent of Wikidata (on the generators' own hypotheses its yes was right 30 / 31, its no 114 / 117); Wikipedia lead text fails like co-mention (yes right 25 / 43); GeoNames for places. S8's promotion on dev with them: 15 graded at 0.667 [0.417, 0.848], gate not passed |
| Promotion (S8d, [report](reports/S8D_PROMOTION.md)) | `decide` needs an independent yes; a source promotes only where its LR+ on the generators' errors is ≥ 10 (DBpedia dates, places, links; GeoNames places). A fresh confirm-half set, read once: 25 graded at 0.880 [0.700, 0.958]; the gate (0.9) is not met. The 3 misses are S15 name or grain mismatches (Brussels for City of Brussels). The Known tier is empty |
| Curiosity scheduler (S10, [report](reports/S10_CURIOSITY_SCHEDULER.md)) | Dev: +126% confirmed value per cost over uniform at a quarter budget. The gate (a fresh set on S15 test's bit-17 half, read once) is not met: 2.105 against 2.227, LCI 0.000. One small arm holds 8 of the 12 promotions, and the highest-P0-first order chosen on dev reversed. Uniform stays |
| Concepts and open relations (S11a, [report](reports/S11A_CONCEPTS_AND_OPEN_RELATIONS.md)) | G9: 0 of 60 held-out classes rediscovered at Jaccard ≥ 0.5 (null 0; nearer than chance on 50, p 2.8e-12); no hypotheses for hidden slots. G11: 21 of 38 phrase clusters hold on test (0.861); educated_at found twice; 19 of the known-relation proposals are linked years inside dates already read. S11a-b: skipping date links removes all 19 date proposals but moves G11's dev choice to C2 (21 non-date proposals, 0.613), so G11 is unchanged; the date pairs had decided S11a's choice **S11d 2026-10-29:** G11 with the date skip (C2): 6 of 21 hold, 0.492 [0.408, 0.577]; the skip stays opt-in. G9 on A3: 14 of 42 fresh classes rediscovered (0.333), 2 strictly (0.048 [0.013, 0.158]); not adopted. No hypotheses. |
| Literature-based discovery (S11b, G12, [report](reports/S11B_ABC_DISCOVERY.md)) | Open form: P@100 0 on S15 test (degree 0.010, random 0), LCI 0.000; about 1 pick in 800 is a hit; code removed (its rule 3). Closed form: AUROC 0.682 against popularity's 0.472. Typed pool: 0 promoted |
| Analogy, anomalies, abduction (S11c, [report](reports/S11C_ANALOGY_ANOMALY_ABDUCTION.md)) | G4 analog transfer ranks hidden facts best of the store's generators (test MRR 0.173; G2 0.153, base 0.105) and doubles the base rate on S15's unstated facts; G3 finds a deleted fact in its top 10 84% of the time; G5 helps on country and nationality only; G13 parked (real-anomaly verdicts 0.647). Test promotions 11 of 11 true; too few for S8's gate. S11e (dev): G3 by the link predictor alone (recall@1 +0.066), G4 without carried dates (ranking unchanged, its false dates gone), G13 keeps its verdict prior; the fresh read waits for S8e's set |
| Tokens T0–T2 ([report](reports/TOKENS_T0_T2.md)) | Dev bits per byte (canonical segmentation): Unigram 4k 1.9039, BPE 4k 1.9048 against the byte model's 2.0725; smaller vocabularies win at 8 MiB (BPE 1k 1.8856); byte-level Kneser–Ney matches them (order 8 1.8951, order 12 1.8831), so on 8 MiB the gain is smoothing, not tokens; T4 judges generation |
| Byte–token bridge (Tokens T3, [report](reports/TOKENS_T3_T4.md)) | Exact on a planted chain (5e-16) for BPE, longest match, Unigram and raw longest match; TOK-A's canonical bits per byte overstate by 0.0017; the naive next byte inside text is 4.0 bits worse than exact. The whole-words tokenizer is not prefix-closed (no exact bridge). T4 waits for GEN-A's harness; at T 0.6 a smoke run copied 94% of its bytes from training |
| Thought processes (K4, K5, [report](reports/K4_K5_THOUGHT_BINDINGS.md)) | All 33 primitives bound to Cypha's libraries; the rule-10 guard caught 100 of 100 planted fact-store answers. K5: differential_diagnosis 22% cheaper than half_split on skewed fault chains (Thompson picks it 95.8%); ach 0.830 against 0.580; SPRT errors 0.039 / 0.035; CUSUM 8.6 against Shewhart 29.1 (1σ); T-QA through confidence_monitor: 0 right, 0 wrong, all abstained. DIF applicability without a class: regret 0.27–0.34 (pooled Beta 0.70, class-given Beta 0.01–0.02), AUROC 0.93–0.98; not adopted (low-end calibration) |
| Chunking and process mining (K6, K7, [report](reports/K6_K7_CHUNKING_AND_MINING.md)) | K6: chunk accuracy 0.969 [0.963, 0.974]; System 1 answers 33% of tasks at +0.0019 [−0.0006, +0.0043] wrong answers (gate met; cost −26%); in answer mode chunks give only don't-know cards. K7a: planted processes recovered exactly in 0.96 of held-out cases with a noise floor of one (THOUGHT's miner 0.76; library 144 of 144). K7b: edit precision 0.685 (gate 0.9 not met; edge removal right 57% of the time); LCI promoted 73 edited models, none with a held-out loss. DIF calibration map (Platt): regret 0.33 → 0.21, but all-pairs ECE 0.084 → 0.113; not adopted |
| Topic routing (C0/C7, [report](reports/C0_C7_BASELINES_AND_ROUTING.md)) | 20 Newsgroups 2k, test: Cypha (`W_enc` frozen) on PPMI-SVD + TF-IDF 0.613 (gate met), hashing 0.203, TF-IDF + logistic regression 0.718 (0.835 on all 11k; Cypha 0.608); AG News 0.829 / 0.521 / 0.853. Not adopted into Branch A: LCI 0.5 (calibration) |
| Cypha-classifier tagger (E1, [report](reports/E1_CYPHA_TAGGER.md)) | UD EWT test 92.06% UPOS (gate 94%, perceptron 94.81%); whitening lifts Cypha 83.8 → 92.0 on dev; confidence below the perceptron's at every coverage; the OOD flag fires on 100% of tokens (fixed threshold) |
| Deep deduction, laws, wild tier (S12a, S12c, [report](reports/S12A_DEDUCTION_LAWS_REFINEMENT.md)) | Planted-law gate met on fresh tables after S12c (recall 0.967 [0.943, 0.981], no false law in 720); S12a's own test read (0.853) stays not met. G15 finds population density and the unit conversions in the 2006 infoboxes. G1 beyond depth 1 is right 7% of the time (depth 1: 37%); its prior can be calibrated by depth (Brier 0.117 against 0.316). G19 removed. G20 all refuted. No promotions. Prospective log of 46 |
| Possible worlds (S9, [report](reports/S9_WORLDS_AND_FRAGILITY.md)) | I9 holds over 43,409 worlds; consequence tests through worlds never say no on this store (cut again; their yes has LR 3.35 on dev, 10.5 on S15). Fragility: 76.9% of held facts are single points of failure; retracting Wikidata drops 62%. S9b: Wikidata stays one source (per claim changes no fact's state, only the blast radius); single points of failure as fragile gaps behind `--gaps-spof` (10,454; 22 of the top 100); whether they point at errors cannot be read here |
| Temporal holdout (S15, [report](reports/S15_TEMPORAL_HOLDOUT.md)) | Frozen in `bench/eval/`: 38,393 facts about 4,059 enwik8 subjects, each with a false partner; 8,574 true by 2006 but unstated, 1,917 true only after 2006-03-04 (99 deaths); test half unread; the Wikidata cache is committed |

## Corpus on D:

`scripts/fetch_corpus.py` writes `D:\cypha_llm\corpus` (`/mnt/d/cypha_llm/corpus` from WSL). Ctrl+C stops it. The same command continues. `scripts/run_fetch.sh` is the visible WSL launcher.

| Category | Cap | What it is |
|----------|----:|------------|
| Books | 40 GB | Gutenberg texts, Wikisource, Wikibooks |
| Code | 200 GB | RedPajama GitHub |
| Articles | 580 GB | C4, RedPajama Common Crawl, English Wikipedia, 50 GB of WET from `CC-MAIN-2026-43` |
| Papers | 90 GB | RedPajama arXiv |
| Structured | 80 GB | Stack Exchange |
| Labelled | 20 GB | Dolly, OpenAssistant, AG News, SQuAD, IMDB |

The RedPajama book files are not downloaded. That slice was removed for copyright. Gutenberg plus Wikisource is the book text, and it is tens of gigabytes, not hundreds. The old PubMed bulk directory and `https://arxiv.org/src/arXiv_src_manifest.xml` return 404. Papers come from the RedPajama arXiv list.

Seed URLs were probed on 2026-10-25 (`python scripts/fetch_corpus.py --check`). Each returned HTTP 206 with a real body.

## hp gate24 (comparison only)

Observe **1.611729 BPC**, archive **1.611759 BPC** on enwik8 (2026-09-19). Lab RSS at mem 22 is about **1.5–2 GB**. A tiny-tier run on the first 8 MiB used about **40 MB** and scored about **1.92 BPC**, with a best sample surface near **0.997**. Those tables are sized in advance. They are not the model trained on this corpus. Serve notes for that binary: [`native/CYPHALM_SERVE.md`](native/CYPHALM_SERVE.md).

The D17 number **2.664 BPC** in `bench/BASELINE_LOCK.json` is a bench lock for the retired hybrid. [`history/LEGACY_LLM.md`](history/LEGACY_LLM.md).

## Plan: train the trie on about 1 TB

This is Track A of [`PLAN.md`](PLAN.md). The experiments A1–A8 and predictions P-A1 to P-A5 are recorded there.

Unstructured text and labelled chat are different jobs.

1. **Hold out text first.** From each category, set aside a frozen slice before any count is updated. Score BPC and surface `S` there. Score the same 8 MiB enwik holdout with the existing gate24 and tiny checkpoints. That is the comparison. Gate24 is not trained on the terabyte.
2. **Shard the unstructured text at about 1 GB.** A full pass of enwik9 (10⁹ bytes) peaked near **400 MB**. One shard at a time stays in that range. Write a `<|doc|>` break between files so order 4 does not glue the end of one file to the start of the next.
3. **Pack the shard, then free it.** Drop counts below 2, store the kept counts in one byte, and drop the fat nodes before the next shard. Merge shards by adding the counts for the same context, then drop counts below 2 again. The mixture weights stay the ones fit on the 8 MiB slice (about **0.93** on order 4 and **0.06** on the match). They are not re-learned across the terabyte.
4. **Match record stores the next byte.** A `uint32` index into the file wraps past 4 GiB. After a shard is packed, the match entry is the predicted byte, not an offset that pins the corpus in RAM.
5. **Labelled chat is a second index.** Freeze the trie and the mixture. Build a map from the whole `User: … Assistant:` prompt to the stored answer. Do not `observe` Dolly into the trie. On the 8 MiB split, **75.7%** of holdout bytes shared a 4-gram with Dolly, and the match keeps only the latest follower, so those wiki predictions became the last Dolly byte. A fixed 8-byte or 16-byte key cannot separate the prompts: every one of the 15,011 ends in `\nAssistant: `.
6. **Ask both ways.** A prompt that is in the chat index returns the stored answer. A prompt that is not is a sample from the packed trie. Report BPC, surface `S`, and the resident size of the packed model next to the gate24 numbers on the small slice.

Order of the unstructured shards: Wikipedia and articles, then books, code, and papers, then Stack Exchange. The labelled index is last.
