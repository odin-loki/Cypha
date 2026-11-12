# E2: an entity recogniser whose decisions the Cypha classifier makes

**Odin Loch** · 2026-11-12 · PLAN Track E, E2 · branch `wip/e2-ner` · follows [E1_CYPHA_TAGGER.md](E1_CYPHA_TAGGER.md)

(The data, the design, the reads, the rules and the predictions were written and committed before any training run, in ca142f8b; amendment 1 before its runs, in ca7a35d5. Results are added around them.)

## Result (2026-11-12)

**The gate is not met.** The Cypha-classifier recogniser scores **0.678** micro F1 over the five types on the silver test (sentence bootstrap [0.647, 0.708]). The gate is 0.80. On the hand-checked set it scores **0.426** [0.275, 0.571]. Nothing uses it. The planner is unchanged.

| Read | Silver test (once) | Hand-checked set (once) | Silver dev |
|---|---|---|---|
| Micro F1, five types | **0.678** [0.647, 0.708] | **0.426** [0.275, 0.571] | 0.651 [0.620, 0.683] |
| PERSON | 0.409 [0.347, 0.468] (P 0.68, R 0.29) | 0.235 (12 gold) | 0.471 |
| PLACE | 0.564 [0.487, 0.642] | 0.381 (25) | 0.502 |
| ORG | 0.090 [0.000, 0.195] (R 0.05) | 0.000 (18) | 0.125 |
| DATE | 0.681 [0.619, 0.738] | 0.759 (14) | 0.579 |
| NUMBER | 0.853 [0.822, 0.878] (P 0.74, R 1.00) | 0.625 (5) | 0.819 |
| MISC (not gated) | 0.357 [0.321, 0.393] | 0.333 (43) | 0.392 |
| Six-type micro F1 | 0.571 | 0.379 | 0.557 |

- **The dev choice** (amendment 1): α 0.5, ε 0.1, `delta_lr` 0.003, `world_lr` 0.002, Cypha learns from a quarter of the O tokens, 1 pass. Training 34 s, tagging 5,504 tokens/s, 0.99 GB.
- **The ceiling of Cypha's rule is low too.** Exact class means (LDA) on the same input reach 0.708 on dev. A shared-variance Gaussian is a linear rule, and entity boundaries need conjunctions (a capital *and* the previous tag *and* the next word's shape). This is E1's finding again, and stronger.
- **Online Cypha is unstable here.** With E1's rates the dev F1 jumps between passes (0.55 at best). Smaller rates and fewer O tokens lift it to 0.65.
- **The typical error is a broken span.** Tags flip type inside a name ("Bay" PLACE, "Bengal" MISC) or run on past it ("General Motors New Zealand plant in Petone outside Wellington, until it closed in" as one MISC span). Most names come out as MISC (441 of 727 spans on the dev prompts).

**The planner read** (250 GEN-A dev prompts, store `e9_text`, offline; Wilson intervals):

| Candidate list | Prompts with a main-entity plan |
|---|---:|
| find1 (the current finder) | **24** (0.096 [0.065, 0.139]) |
| find2 (Q1's finder) | 24 |
| e2 alone | 18 (0.072 [0.046, 0.111]) |
| e2 without MISC | 5 |
| **e2 first, then find1** | **27** (0.108 [0.075, 0.153]) |
| ngram ceiling (any token run that is a store subject) | 36 (0.144 [0.106, 0.193]) |

- E2 alone loses 9 of find1's prompts and gains 3: −6 [−13, 0]. Its broken spans miss names find1 gets right (Gothenburg, Carl Friedrich Gauss, Charlemagne, Ken Livingstone).
- E2 first, then find1: +3 [0, +7] against find1, with 2 more prompts on a different subject.
- **By hand**, on the 5 prompts where E2 changes or adds the main subject: right 2 (Australia, Colin Powell), even 1 (Katharine Hepburn for The Philadelphia Story), wrong 2 ("Bengal" from "Bay of Bengal", "Greenwich" from "Greenwich Village").
- **The store is the main limit.** Even a lookup of every token run reaches only 36 of 250. Every candidate with a fact already gets a main plan: in each list "any fact" equals "main".

**Question.**
- Can the Cypha classifier find named entities in Wikipedia prose, with E1's machinery?
- The types are PLAN's: PERSON, PLACE, ORG, DATE, NUMBER. MISC (other names) is learned and reported, but not gated.
- The gate is PLAN's: at least 80% F1 on the silver test (micro F1 over the five types).
- And does it help the K5 planner find the prompt's main entity? Today the planner plans a main entity for 24 of the 250 GEN-A dev prompts (P1t, §13.46).

**Rule 10.** The recogniser marks spans and types. It answers no question. Every tag is the Cypha classifier's decision. The silver labels and the count tables are training data, prepared from the dump's markup. No list of names is consulted at tagging time. The planner is not changed in this item.

## Data

**Source.** enwik9 (`/mnt/d/cypha_llm/enwik9`, the 2006 English Wikipedia dump).

**Held out, never read for training:**
- enwik8 bytes [16 MiB, 72 MiB) (SCRIBE's human text) and [75497472, 1e8) (GEN-A's prompts and continuations).
- To be safe, every page whose text starts anywhere in [16 MiB, 1e8) is skipped, also for typing.
- The training sentences come only from enwik9 [1e8, 1e9).
- The overlap pages (`skip_offsets.txt`, the §13.38 list of 74,476 pages that repeat enwik8 text) are skipped.

**Silver labels** (`scripts/e2_silver.py`, Python, one pass to type pages and one to read sentences):
1. **Typing pages from their own markup.** A page is PERSON, PLACE or ORG by its infobox name ("Infobox Biography", "Infobox City", "Infobox Company") or, without one, by a vote of its categories ("1879 births", "Cities in Germany", "Record labels"). Any other article is MISC. Disambiguation pages are left untyped. Redirects are followed. 150,465 titles are typed: MISC 65,866, PLACE 43,399, PERSON 27,309, ORG 5,497, disambiguation 8,394. Wikidata is not used (PLAN's row says "typed through Wikidata"; the dump's own categories do the same job and keep the labels inside CyphaLM's reading).
2. **Spans from links.** In a prose line, a link `[[Target|anchor]]` whose anchor starts with a capital becomes a span of Target's type. A single-word anchor of a PLACE that is not in the place's name ("[[France|French]]") becomes MISC. A lowercase anchor ("[[dog|dogs]]") is not an entity.
3. **DATE and NUMBER.** Year, decade, century and day links ("[[1954]]", "[[March 5]]") are DATE. So are plain day and month expressions, decades and years 1000–2029 outside links. Other numerals written in digits are NUMBER. Numbers written as words are not labelled.
4. **Later mentions.** Inside a page, a name once typed (an anchor, the page's own title, and a person's surname) labels its later unlinked mentions.
5. **The coverage filter.** A sentence is kept only if every capitalised word is inside a span. The first word may stay outside only if it is a common lowercase word (seen lowercase at least 5 times in enwik9 [1e8, 1.5e8)). This keeps unlinked names out of the "O" class. Sentences of 6–60 words ending in . ! or ? only.
6. **Split by page:** md5(title) mod 20 = 0 is test, 1 is dev, the rest train. No page is in two splits.

| Split | Sentences | Words | PERSON | PLACE | ORG | DATE | NUMBER | MISC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 100,000 | 1,899,330 | 11,432 | 6,303 | 2,147 | 9,609 | 16,093 | 18,100 |
| dev | 3,000 | 56,532 | 289 | 150 | 45 | 270 | 472 | 565 |
| test | 3,000 | 56,532 | 319 | 220 | 64 | 316 | 543 | 537 |

(These are the script's span counts. A few spans merge or drop when the tool maps them to tokens.) ORG is rare in the silver data, so its intervals will be wide.

Files: `bench/e2_ner/silver_dev.tsv` (sha256 `72b2b51f2bdcf061…`), `silver_test.tsv` (`345c65696ad46725…`), `silver_stats.txt`. The train file (13.8 MB, `c12dbbde9136ddb2…`) is rebuilt by the command below and is not committed.

**The hand-checked set** (`bench/e2_ner/hand.tsv`, from `hand_pool.tsv` and `hand_labels.txt` by `scripts/e2_hand.py`):
- 60 prose sentences from the test pages, drawn in a seeded order *without* the coverage filter. So it has the unlinked names and the sentences the silver set leaves out.
- I labelled them by hand before any model was trained. They hold 117 entities: MISC 43, PLACE 25, ORG 18, DATE 14, PERSON 12, NUMBER 5.
- Annotation rules: PERSON is a person or a person-named character; PLACE is a country, region, city, valley, island, park or other place on Earth; ORG is a company, institution, agency, newspaper, broadcaster, party or team; DATE is a year, date, decade or century; NUMBER is a numeral in digits (the whole token, so "27-year-old" is NUMBER); MISC is any other proper name (demonyms, languages, religions, works, products, events, species names, planets). Titles such as "the King" are not entities.

## The model (fixed before any run)

`native/tools/classify/cypha_ner.cpp` (new). It is E1's tagger, retargeted to BIO entity tags.

- **Tags:** O plus B- and I- for the six types: 13 classes.
- **Tokens:** words (letters, digits, UTF-8, joined by - ' . & and by commas inside numbers), an abbreviation keeps its final period ("U.S.", "Dr."), "'s" is its own token, other punctuation is one token each. A token takes a span's type if it overlaps the span.
- **Input (D = 238):** 15 blocks of log P(tag | feature), 13 values each. Features: the form, the lowercase form, suffixes of 1–4 letters, the 2-letter prefix, the word shape, the previous and next lowercase words, the previous tag, the previous two tags, the previous word with this word, this word with the next word, and the shapes of the previous, this and next word. Smoothing as E1: P(t | f) = (c(f, t) + α P(t)) / (c(f) + α). Then 10 scalars: log(1 + form count), unseen form, initial capital, all capitals, has a digit, has a hyphen, first in sentence, the share of the word's mid-sentence uses that are capitalised, log(1 + lowercase count), punctuation. Then E1's committed vectors (`data/cyphalm/simvec_ppmi32.bin`, 32 + 1).
- **Cross-fitting:** the count tables come from all 100,000 training sentences, in 5 folds by sentence index. A training row's counts leave out its own fold. Dev and test use all of train.
- **Scaling:** standardised per dimension, then whitened by the pooled within-class covariance (E1's amendment 1; E1 lost 8 points without it).
- **The classifier:** `cypha::Cypha` from `create_fresh_model_root` (identity `W_enc`, frozen; field dimension 16), `Cypha::update` one token at a time, GH path, as E1. Cypha trains on a seeded sample of 15,000 training sentences (about 285,000 tokens; E1 trained on 204,000). Tagging is greedy left to right; an I- tag after O or another type opens a span.

## Comparisons

1. **Counts alone** (no learning): the argmax of the summed log-probability blocks.
2. **Exact class means** in the whitened space (LDA): the ceiling of Cypha's scoring rule, as in E1.
3. **The planner's finder** for the coverage read (below).

## Reads

- **Span F1** (exact boundaries and type) per type, and micro F1 over the five types; MISC and the six-type micro F1 beside them. 95% intervals by a bootstrap over sentences (2,000 resamples).
- On silver dev (for choices), silver test (read once) and the hand-checked set (read once, with the test read).
- Token accuracy. Training time, tagging speed, count-table size.

## Choices and decision rules (fixed before any run)

1. **Stage 1:** α ∈ {0.5, 2}, by the dev micro F1 of exact class means (whitened, ε 0.1). E1 found ε does not change this read.
2. **Stage 2:** on that α, ε ∈ {0.01, 0.1} × `delta_lr` ∈ {0.01, 0.03} × `world_lr` ∈ {0.002, 0.008}, 3 passes, dev read after each pass. The dev choice is the highest dev micro F1 (five types). Ties within 0.005 go to fewer passes, then the smaller `delta_lr`.
3. One design round, then the test split and the hand set are read once with the dev choice. Nothing changes after that read.
4. **The gate:** silver-test micro F1 over the five types ≥ 0.80.
5. If the gate is missed, E2 stays in the tree as a measured tool, and nothing uses it.

## The planner read (offline; the planner is not changed)

`scripts/e2_coverage.cpp` (new; it includes the planner's headers read only) loads `bench/fact_store/e9_text` as `sample --k5-plan` does (the holdout [7340032, 8388608) and `--plan-drop 75497472:100000000` dropped) and the skeleton bank `skel_e9.txt`. For each of the 250 GEN-A dev prompts and each candidate list, it runs make_plan's main-entity rule (the first candidate whose store subject has a fact of a bank relation with at most 2 values):

| List | What |
|---|---|
| find1 | today's finder (`ContentPlanStore::candidates`: bold lead, links, capitalised runs) |
| find2 | Q1's finder (`--k5-plan-find 2`) |
| e2 | E2's PERSON, PLACE, ORG and MISC spans, in text order |
| e2ppo | the same without MISC |
| e2+find1 | E2's spans first, then find1's candidates (the hook below) |
| ngram | every token run of the prompt that is a store subject (a lookup, for scale only: the ceiling of any finder on this store) |

Reads: prompts with a main-entity plan, and prompts with any candidate that has a fact, with Wilson intervals; the paired difference against find1 with a bootstrap interval. Where E2's main subject differs from find1's, I judge by hand which one is the prompt's real entity.

**Measured before E2 exists** (the tool with an empty span file, 2026-11-12): find1 plans 24 prompts (as §13.46), find2 24, ngram 36. So on this store no finder can plan more than 36 of 250. The store, not the finder, is the main limit.

## The proposed planner hook (registered for later; not built here)

- A flag `--k5-plan-find 3`, off by default. With it, `FactStore::candidates` lists E2's PERSON, PLACE, ORG and MISC spans first, in text order, then today's candidates not already listed. `find_main` is unchanged.
- It needs a saved E2 model (count tables, whitening and the Cypha file); `cypha_ner` does not save one yet.
- Adopt only if, on dev: (a) main-entity plans do not fall; (b) on the prompts where the main subject changes, the hand-judged right subject is at least as common as before; (c) then four blind raters prefer it, by §13.38's rule.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| P1 | Stage 1 picks α 0.5 (as E1) |
| P2 | Dev micro F1 (five types) of the dev choice: 0.72–0.86 (centre 0.80) |
| P3 | Silver test micro F1 within 0.03 of dev; the gate (0.80) is met or missed narrowly |
| P4 | Silver test per type: NUMBER ≥ 0.90, DATE ≥ 0.85, PERSON 0.60–0.85, PLACE 0.50–0.80, ORG ≤ 0.50 |
| P5 | Hand-checked set micro F1 (five types) at least 0.10 below the silver test (unlinked names, no filter) |
| P6 | Exact class means within 0.03 of online Cypha on dev; counts alone at least 0.20 below |
| P7 | Planner: e2 alone plans a main entity for 18–30 prompts (find1: 24); e2+find1 for 24–30; neither passes ngram's 36 |
| P8 | Where e2's main subject differs from find1's, E2's is the right entity in at least 60% of those prompts |
| P9 | Each training run under 15 minutes; tagging at least 2,000 tokens per second |

## Round 1 (dev, 2026-11-12)

**Stage 1** (exact class means, whitened, ε 0.1; dev micro F1 over the five types):

| α | Counts alone | Exact class means (LDA) |
|---|---:|---:|
| **0.5** | 0.529 | **0.708** |
| 2 | 0.507 | 0.704 |

α 0.5 is chosen (P1: hit). The LDA ceiling is low: PERSON 0.57, PLACE 0.39, ORG 0.00, DATE 0.80, NUMBER 0.94, MISC 0.14.

**Stage 2** (α 0.5, online Cypha, dev micro F1 after passes 1 / 2 / 3):

| ε | `delta_lr` | `world_lr` | Pass 1 | Pass 2 | Pass 3 |
|---|---|---|---:|---:|---:|
| 0.01 | 0.01 | 0.002 | 0.486 | 0.524 | 0.321 |
| 0.01 | 0.01 | 0.008 | 0.403 | 0.445 | 0.500 |
| 0.01 | 0.03 | 0.002 | 0.368 | 0.439 | 0.292 |
| 0.01 | 0.03 | 0.008 | 0.323 | 0.376 | 0.254 |
| 0.1 | 0.01 | 0.002 | 0.447 | 0.518 | **0.552** |
| 0.1 | 0.01 | 0.008 | 0.427 | 0.520 | 0.462 |
| 0.1 | 0.03 | 0.002 | 0.431 | 0.350 | 0.323 |
| 0.1 | 0.03 | 0.008 | 0.420 | 0.370 | 0.347 |

- The best cell is 0.552 (ε 0.1, `delta_lr` 0.01, `world_lr` 0.002, pass 3). It is 0.16 under its own ceiling (LDA 0.708), and close to counts alone (0.529).
- The online model is unstable from pass to pass. Types swap: in one cell NUMBER's recall is 0.99 at precision 0.46, because DATEs are tagged NUMBER.
- The smaller rates do better in every pair. About 87% of training tokens are O.

## Amendment 1 (registered 2026-11-12, after round 1 and before any amendment run)

Round 1 is far under its ceiling, unlike E1 (within 0.4 points). Two likely causes: the rates are too high for this input, and the O class floods the online updates. So, one more round, on dev only:

- ε 0.1, `world_lr` 0.002, α 0.5 (round 1's best);
- `delta_lr` ∈ {0.003, 0.001} × O kept with probability ∈ {1, 0.25} (`--o-keep`, seeded; entity tokens are always kept), 3 passes.
- The dev choice of the item is the best dev micro F1 over rounds 1 and 2 (rule 2's ties). Then the test split and the hand set are read once.
- `--o-keep` defaults to 1, so round 1's runs are unchanged.

**Predictions for the amendment:**

| # | Prediction |
|---|---|
| A1 | The amendment's best dev micro F1 is at least 0.60 |
| A2 | It stays at least 0.05 under the LDA ceiling (0.708) |
| A3 | O subsampling raises recall and lowers precision of every entity type |
| A4 | The gate (0.80 on silver test) is missed |

## Amendment 1 (dev, 2026-11-12)

ε 0.1, `world_lr` 0.002, α 0.5; dev micro F1 after passes 1 / 2 / 3:

| `delta_lr` | O kept | Pass 1 | Pass 2 | Pass 3 |
|---|---|---:|---:|---:|
| 0.003 | 1 | 0.485 | 0.590 | 0.568 |
| 0.003 | 0.25 | **0.651** | 0.631 | 0.624 |
| 0.001 | 1 | 0.566 | 0.561 | 0.594 |
| 0.001 | 0.25 | 0.621 | 0.624 | 0.624 |

- **The dev choice of the item:** `delta_lr` 0.003, O kept 0.25, 1 pass, dev 0.651. Round 1's best was 0.552. No other cell is within 0.005.
- The test split and the hand set were then read once with it (command below). Dev in that run reproduced 0.6512 exactly.

## The test read (once, 2026-11-12)

```
cypha_ner run --train train.tsv --dev bench/e2_ner/silver_dev.tsv --test bench/e2_ner/silver_test.tsv --read-test --hand bench/e2_ner/hand.tsv \
  --alpha 0.5 --whiten 0.1 --delta-lr 0.003 --world-lr 0.002 --o-keep 0.25 --passes 1 \
  --prompts bench/eval/gen_prompts_dev.tsv --prompt-out final_prompt_spans.tsv --pred-out final
```

Outputs in `bench/e2_ner/runs/` (every run log as `*.log.txt`): `final.log.txt`; the spans on the dev prompts, `final_prompt_spans.tsv`; the planner read, `e2_coverage.log.txt` and `cov_final.tsv`; the predictions on dev, test and the hand set, `final.{dev,test,hand}.tsv`.

## Decisions (by the registered rules)

- **Rule 4: the gate is not met** (0.678 < 0.80).
- **Rule 5:** E2 stays in the tree as a measured tool, with its selftest. Nothing uses it.
- **The planner hook is not built.** Its first condition fails for E2 alone (main plans fall from 24 to 18). E2 first, then find1, adds 3 [0, +7], but 2 of the 5 changed subjects are wrong. The hook stays registered for a recogniser that passes the gate.

## Predictions, scored

| # | Prediction | Result |
|---|---|---|
| P1 | Stage 1 picks α 0.5 | **Hit** (0.708 against 0.704) |
| P2 | Dev micro F1 of the dev choice 0.72–0.86 | **Miss:** 0.651 |
| P3 | Test within 0.03 of dev; gate met or missed narrowly | **Partial:** +0.027 (hit); missed by 0.12 (miss) |
| P4 | Test: NUMBER ≥ 0.90, DATE ≥ 0.85, PERSON 0.60–0.85, PLACE 0.50–0.80, ORG ≤ 0.50 | **Partial:** PLACE 0.56 and ORG 0.09 hit; NUMBER 0.85, DATE 0.68, PERSON 0.41 miss |
| P5 | Hand set at least 0.10 under silver test | **Hit:** 0.426 against 0.678 |
| P6 | LDA within 0.03 of online Cypha; counts alone at least 0.20 below | **Miss:** LDA 0.057 above; counts alone 0.12 below |
| P7 | e2 18–30 main plans, e2+find1 24–30, neither above 36 | **Hit:** 18, 27, 36 |
| P8 | E2's changed subject right in at least 60% | **Miss:** 2 of 5 (one even) |
| P9 | Runs under 15 minutes; tagging at least 2,000 tokens/s | **Hit:** at most 8.7 minutes; 5,504 tokens/s |
| A1 | Amendment's best at least 0.60 | **Hit:** 0.651 |
| A2 | Still at least 0.05 under the LDA ceiling | **Hit:** 0.057 under |
| A3 | O subsampling raises recall and lowers precision of every type | **Miss:** recall rose for all, but precision rose for PLACE, ORG and NUMBER |
| A4 | The gate is missed | **Hit** |

7 hits, 2 partial, 4 misses (P2, P6, P8, A3).

## What is weak, and next

- **The silver labels are not checked.** The hand set checks the model, not the silver labels. The page typing has visible errors ("Gustloff", a ship named after a man, typed PERSON). The coverage filter keeps easy sentences, so the silver test is kinder than real prose (0.68 against 0.43).
- **ORG is nearly absent** from the silver data (2,147 training spans) and is not learned.
- **Next, in order:**
  1. E1's advice for E2: the averaged perceptron (sparse conjunctions) on the same silver data and reads, as a second arm. E1's perceptron beat Cypha by 2.75 points on tags; spans need conjunctions more.
  2. BIO-constrained decoding (Viterbi over Cypha's scores, I-X only after B-X or I-X). It fixes run-on and broken spans without new learning.
  3. A non-linear Cypha decision (class-conditional covariances or a mixture of components), the library item E1 named.
- **For the planner, the store matters more than the finder:** 36 of 250 is the ceiling on `e9_text` for any finder.

## How to reproduce

```bash
# silver data (165 s, 1.5 GB)
python3 -I scripts/e2_silver.py /mnt/d/cypha_llm/enwik9 /mnt/d/cypha_llm/fact_store_e9/skip_offsets.txt OUT --train 100000 --dev 3000 --test 3000
python3 -I scripts/e2_hand.py bench/e2_ner/hand_pool.tsv bench/e2_ner/hand_labels.txt bench/e2_ner/hand.tsv
cmake -S native -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build cypha_ner && build/cypha_ner selftest
# stage 1 (diagnostics only)
build/cypha_ner run --train OUT/train.tsv --dev bench/e2_ner/silver_dev.tsv --alpha 0.5 --whiten 0.1 --passes 0 --diag
# stage 2, one cell
build/cypha_ner run --train OUT/train.tsv --dev bench/e2_ner/silver_dev.tsv --alpha A --whiten EPS --delta-lr X --world-lr Y --passes 3
# the planner read
g++ -std=c++20 -O2 -I native/include scripts/e2_coverage.cpp -o e2_coverage
./e2_coverage bench/fact_store/e9_text bench/eval/gen_prompts_dev.tsv bench/k5_planner/model/skel_e9.txt SPANS.tsv \
    --plan-drop 75497472:100000000 --stats ~/k5s_data/S10.stats --list cov.tsv
```
