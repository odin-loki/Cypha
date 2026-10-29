# S4: estimates of the unseen, and the frontier report

**Odin Loch** · 2026-10-28 · DISCOVERY §3.3, §3.4, §13 S4 · follows [S3_S5_GAP_MAP_AND_HYPOTHESES.md](S3_S5_GAP_MAP_AND_HYPOTHESES.md) and [S1_S2_EPISTEMIC.md](S1_S2_EPISTEMIC.md)

(The rules and predictions below were written before the runs. Results and decisions are added under them.)

**Question.** Cypha should know how much it has not seen, not only where. Do the standard unseen-species estimates, computed on what Cypha has read, predict what it finds when it reads more? And where would reading more teach it the most?

**Gate (DISCOVERY §13).** Missing-mass validity within ±25% on a held-out source.

**Rule 10.** This is Cypha's self-assessment. Nothing here produces an answer for CyphaLM, a gate decision or a tag. It counts what was read and predicts what reading more would bring.

## Rules (fixed before the runs)

### Data

- enwik8, all 8,587 main-namespace articles (`pattern_learn`'s page list), and the v2 fact store as S3 built it (`--cv --wikidata`).
- A page is the sampling unit. A page's place is the byte offset where its text starts.

### Splits

- **Store view (the gate).** Estimation set E: pages starting before byte 50,000,000 (the first half of the dump). Held-out set H: the rest. H is split by a bit of the title's FNV-1a hash into **H-dev** and **H-test**. H-dev chooses; H-test is read once, at the end.
- **CyphaLM's view.** E7: pages starting before byte 7,340,032 (CyphaLM's wiki train). H7: pages starting at or after 8,388,608, split into dev and test the same way.
- **Random split (diagnostic, dev only).** E and H by another hash bit over all pages. It tests the estimators with no drift between E and H.

### Items

| Kind | Item | Observation |
|---|---|---|
| words | a maximal run of ASCII letters in the page's raw text, lower-cased | each run |
| entities | the target of a `[[target]]` or `[[target|text]]` link, normalised (`norm_ent`), without `#section`; namespace and language links skipped | each link |
| facts | (subject, relation, object): a text seed (infobox, table or category) or a stored extracted fact (P ≥ 0.5) | each seed, and each evidence sentence of a stored fact, on the page that holds it |
| values | (relation, object) of the same facts | the same |

### Estimators, computed on E

- **GT0**, DISCOVERY §3.3's Good–Turing missing mass: n₁ / N.
- **GTu**, Good–Turing with the page as the unit: the observations of items seen on exactly one page, over N. It is the leave-one-page-out estimate of the same quantity. Text repeats an item inside a page, so a page-level repeat is no evidence that an item is common.
- **Q1/U**, the incidence form: items on exactly one page over all (page, item) incidences.
- **Chao extrapolation** (Chao et al. 2014), with pages as units: Ŝ(T + t) = S_obs + Q̂₀ [1 − (1 − Q₁/(T Q̂₀ + Q₁))^t], Q̂₀ = ((T−1)/T) Q₁² / (2 Q₂). Also the abundance form, over observations.
- **Heaps** (words): V = K N^β fit on E's growth curve (every 25th page). It predicts the vocabulary after more reading, and the new-word rate dV/dN = β V / N.
- **Chapman** per relation with a kept pattern: source A = slots (subject, relation) with a text seed; source B = slots where a kept pattern fires on the subject's page (any value); m = both.

### Targets, measured on H-dev or H-test

- For GT0 and GTu: the share of the held-out observations whose item is not in E.
- For Q1/U: the share of held-out (page, item) incidences whose item is not in E.
- For Chao: the distinct items in E ∪ H-x.
- For Heaps: the distinct words in E ∪ H-x, and the words target.
- For Chapman: a reference count N_ref = |A ∪ B ∪ W|, where W is the slots with a Wikidata value. Chapman is a lower bound because A and B are correlated (patterns are learned from seeds). The check is N̂ ≤ N_ref.

### Intervals

- The target's sampling error: a bootstrap over the held-out pages (2,000 resamples), E fixed. It gives a 95% interval for the ratio estimate / target.
- The estimate's own error: a delete-a-group jackknife over 20 groups of E's pages, reported as a standard error.

### The gate

- On H-dev, choose between GT0 and GTu: the one with the smaller mean |ln(estimate / target)| over words, entities and facts (relations pooled).
- **S4 passes** if, on H-test (read once), the chosen estimator's point ratio estimate / target is within [0.75, 1.25] for each of words, entities and facts.
- Per relation (facts and values, relations with ≥ 30 held-out observations), the share of relations within ±25% is reported, not gated. CyphaLM's view is reported with the same rule, not gated.

### The frontier report (`pattern_learn --frontier FILE`, with `--gaps`)

One row per (type, relation) cell of S3's map (types: country, person, place, other):

- **known**: filled slots (S3's `with_store`).
- **expected**: entities × P(exists) (S3's chosen estimator of f(T, R)).
- **coverage** = known / max(known, expected); **unseen in the slice** = max(0, expected − known): slots that should exist on pages already read, which better extraction or other sources would fill (P1).
- **certainty**: the mean P over the cell's filled slots (a text seed counts 20/21, an extracted fact its fused P).
- **demand**: CyphaLM's tags on the cell (S3's demand log).
- **reading more**: the chosen estimator of missing mass over the cell's fact observations, times the cell's fact observations per MiB of article text, times 10: the expected new facts per 10 MiB of comparable new text.
- **values**: Chao1 over the relation's distinct values (how many distinct values the relation has, seen and unseen).
- Two rankings: by expected new facts per 10 MiB read ("where reading more teaches the most"), and by unseen in the slice ("where better extraction teaches the most"), each with demand alongside.

### Unknown-unknown signals (§3.4)

Only the vocabulary signal is cheap now: the share of new words in held-out text against the Heaps prediction. The other signals (OOD rate, unparsed sentences, surprise spikes, routing, unnamed clusters, trust shifts) have no producer yet and are listed as not done.

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| P1 | GT0 underestimates the target for words and entities on H-dev (store view) by 10–40%: repeats inside one page make items look common |
| P2 | GTu is within ±15% of the target for words and entities on H-dev |
| P3 | Facts, pooled: the target is ≥ 0.95 (a fact sits on its subject's page); GTu within ±5%; GT0 below GTu |
| P4 | GTu is chosen on dev, and the gate passes on H-test |
| P5 | Values, pooled: GTu within ±25% on dev; per relation, at least half of the relations with ≥ 30 held-out observations within ±25% |
| P6 | Chao extrapolation (pages as units) of distinct words and entities in E ∪ H-x is within ±10%, and under rather than over |
| P7 | Heaps: the vocabulary of E ∪ H-x within ±5%; β between 0.5 and 0.8; the Heaps new-word rate within ±30% of the words target |
| P8 | Chapman(seeds, patterns) ≤ N_ref for every relation with a kept pattern |
| P9 | CyphaLM's view: GTu within ±25% for words and entities; its words target is at least twice the store view's |
| P10 | The top cells by reading more are person cells for birth and death dates and years |

## The dev choice (before the test read)

On H-dev (store view), the ratio estimate / target:

| Kind | GT0 | GTu |
|---|---:|---:|
| words | 0.509 | 0.904 |
| entities | 0.768 | 0.944 |
| facts | 0.555 | 1.000 |

Mean |ln ratio|: GT0 0.509, GTu 0.053. **GTu is chosen.** The test half is read once below with it.

## Results (run 2026-10-28)

Tool: `pattern_learn --frontier` on S3's run (`--cv --wikidata --gaps --demand`). Without `--frontier` the run writes S3's files byte for byte (gap map sha256 a77678d8ef7dfdef, cells eed14241a6d6eff2, demand dd1c8ddac831dcb6; the store files equal `f2/v2/aug`). 34–53 s, peak 226 MB.

Items on all of enwik8: 282,882 word types, 246,292 link entities, 8,306 facts, 4,666 values.

### The gate: missing mass on the held-out test half (read once)

Store view: E = 4,413 pages (the first 50,000,000 bytes); H-test = 2,029 pages. Ratio = GTu / target, with the 95% bootstrap interval over H-test's pages.

| Kind | GTu (jackknife se) | Target: share of H-test's observations that are new | GTu / target | GT0 / target |
|---|---:|---:|---|---:|
| words | 0.02279 (0.00055) | 0.02354 [0.02246, 0.02470] | **0.968** [0.923, 1.015] | 0.545 |
| entities | 0.30735 (0.00428) | 0.31358 [0.30360, 0.32493] | **0.980** [0.946, 1.012] | 0.797 |
| facts | 0.99967 (0.00046) | 1.00000 | **1.000** | 0.555 |
| values (not gated) | 0.61477 (0.00863) | 0.63027 [0.60396, 0.65608] | 0.975 [0.937, 1.018] | 0.420 |

**S4's gate is met.** Good–Turing with the page as the unit is within 4% of what reading more pages actually brings, for words, entities and facts, and the word and entity intervals include 1.

- DISCOVERY's n₁ / N (GT0) is not: it says 1.3% of the next words are new when 2.4% are, and 25% of the next links when 31% are. A page repeats what it talks about, so an item seen five times on one page looks common to GT0 and is not.
- Facts are trivially new: a fact sits on its own subject's page, so every held-out fact observation is a new fact (target 1.000).
- The **random split** (no drift between E and H) gives GTu / target 0.946 (words), 1.014 (entities), 1.000 (facts), 1.026 (values) on dev. So the store view's small shortfall on words (0.904 dev, 0.968 test) is drift: the second half of the dump brings slightly more new words.

**CyphaLM's view** (E = its 536 wiki-train pages; H = the pages from 8,388,608 on), test: words 0.956 [0.933, 0.978], entities 0.895 [0.883, 0.908], facts 1.000, values 0.949. All within ±25%. Its words target is 0.056: one word in 18 of the rest of the dump is a word CyphaLM has never read (2.4× the store view's).

### Per relation (not gated)

| Set | Facts within ±25% | Values within ±25% (relations with ≥ 30 held-out observations) | Outside |
|---|---|---|---|
| store, dev | 17 / 17 | 15 / 17 | occupation 5.94, order 1.31 |
| store, test | 14 / 14 | 13 / 14 | occupation 1.51 |
| CyphaLM, dev | 26 / 26 | 22 / 26 | located_in 0.67, occupation 0.63, order 1.39, class 2.09 |
| CyphaLM, test | 21 / 21 | 18 / 21 | located_in 0.68, language 1.45, occupation 0.52 |

Occupation has 38 values seen hundreds of times; its target is near 0 (0.004 on dev), so a relative error means little there.

### Richness and Heaps

| Check | Store dev | Store test | CyphaLM dev | CyphaLM test |
|---|---:|---:|---:|---:|
| Chao extrapolation (pages), words: predicted / observed distinct in E + H | 0.977 | 0.987 | 0.711 | 0.730 |
| the same, entities | 0.980 | 0.989 | 0.798 | 0.789 |
| the same, observations as draws, words | 0.935 | 0.949 | 0.518 | 0.532 |
| Heaps: vocabulary of E + H, predicted / observed | 1.007 | 1.020 | 1.117 | 1.158 |
| Heaps β | 0.611 | 0.611 | 0.661 | 0.661 |
| Heaps new-word rate / words target | 0.655 | 0.701 | 0.648 | 0.684 |

- In the store view (H about half of E) the page-unit Chao extrapolation is within 3%, and under, as registered. In CyphaLM's view it is asked to extrapolate 536 pages to about 4,500 (8×), past its known range (about 2×), and it falls 20–30% short. Heaps overshoots there by 12–16%.
- Heaps' new-word rate counts a new word once (where the vocabulary grows); the target counts every occurrence of it in the new pages. So the rate runs at 0.65–0.70 of the target even with no drift (random split 0.675). As an unknown-unknown alarm (§3.4) the baseline is that ratio, not 1: store view 0.655, random 0.675, so no drift alarm.

### Chapman per relation (the seven relations with a kept pattern)

| Relation | A text seeds | B patterns fire | m | N̂ (sd) | N_ref = A ∪ B ∪ Wikidata | N̂ / N_ref | Text vs Wikidata: N̂ |
|---|---:|---:|---:|---:|---:|---:|---:|
| birth_place | 362 | 226 | 89 | 915 (65) | 1,231 | 0.743 | 1,293 |
| death_place | 77 | 36 | 4 | 576 (212) | 1,026 | 0.562 | 1,128 |
| birth_date | 113 | 706 | 81 | 982 (54) | 1,068 | 0.919 | 1,119 |
| death_date | 81 | 479 | 57 | 678 (45) | 868 | 0.781 | 907 |
| birth_year | 1,135 | 737 | 714 | 1,172 (5) | 1,318 | 0.889 | 1,334 |
| death_year | 914 | 511 | 498 | 938 (5) | 1,027 | 0.913 | 1,037 |
| language | 121 | 27 | 27 | 121 (0) | 272 | 0.445 | 497 |

- Chapman over the two text sources is a lower bound in all 7 relations (0.44–0.92 of N_ref): the patterns are learned from the seeds, so the two sources are correlated.
- Text against Wikidata are closer to independent. Their Chapman estimates agree with the frontier's expected counts for person cells within 8–17% (birth_place 1,293 against 1,146; birth_date 1,119 against 1,035; death_place 1,128 against 964).

### The frontier report

`pattern_learn --frontier FILE` writes 132 cells (4 types × 33 relations). Over all cells the store holds 7,124 slots and an estimated 9,970 more exist on pages already read (coverage 0.42). By type: person 0.56, country 0.77, place 0.25, other 0.20. Of the 45 cells with at least 50 expected slots, 28 are under half covered.

**One definition was corrected after the dev run** (the estimates above are unchanged by it): S3's P(exists) is the chance that an *empty* store slot has a value (calibrated on empty slots), so the unseen slots are (entities − known) × P(exists), and expected = known + unseen. The registered "entities × P(exists)" gave expected below known in some cells (person birth_year 637 against 1,160 known). The frontier file and the lists below use the corrected definition.

Where reading more teaches the most (new fact observations per 10 MiB of comparable text; in brackets, distinct new facts):

| # | Cell | New per 10 MiB | Coverage | Unseen in the slice | Demand |
|---|---|---:|---:|---:|---:|
| 1 | person birth_date | 268.5 (84.9) | 0.710 | 300 | 31 |
| 2 | person death_date | 171.6 (56.0) | 0.601 | 334 | 16 |
| 3 | person birth_year | 143.0 (133.1) | 0.935 | 81 | 0 |
| 4 | person birth_place | 126.1 (70.9) | 0.435 | 647 | 0 |
| 5 | person occupation | 119.7 (89.0) | 0.480 | 644 | 0 |
| 6 | person nationality | 117.6 (102.7) | 0.647 | 330 | 0 |
| 7 | person death_year | 107.2 (104.1) | 0.960 | 39 | 0 |
| 8 | place located_in | 34.7 (32.9) | 0.680 | 114 | 0 |

Where better extraction (or another source) teaches the most (unseen slots on pages already read):

| # | Cell | Unseen | Coverage | Kept pattern |
|---|---|---:|---:|---|
| 1 | other country | 1,194 | 0.065 | no |
| 2 | person death_place | 855 | 0.113 | yes |
| 3 | other founded | 727 | 0.127 | no |
| 4 | person birth_place | 647 | 0.435 | yes |
| 5 | person occupation | 644 | 0.480 | no |
| 6 | person spouse | 613 | 0.038 | no |
| 7 | place country | 440 | 0.002 | no |

- Ranked by distinct new facts instead, the reading list is the same cells in a different order (birth_year, death_year and nationality first: their facts have one evidence sentence each).
- **Demand** sits on few cells: of CyphaLM's 750 dev tags, 31 land on person birth_date and 16 on death_date; most name a relation the store does not have (`definition`, 333) or an entity with no page (S3's finding).
- Values: Chao1 (pages as units) says person birth_date has about 28,000 distinct values (753 seen) and birth_place about 1,300 (396 seen).

### Unknown-unknown signals (§3.4)

| Signal | State |
|---|---|
| Vocabulary: new-word rate against Heaps | Done: the ratio is 0.65–0.70 with no drift; an alarm is a ratio well below that |
| OOD rate of incoming text and questions | Not done: no OOD detector feeds Cypha yet |
| Unparsed-sentence rate | Not done: needs the sentence parser's failure count per domain |
| Surprise spikes in CyphaLM | Not done |
| Uncertain routing | Not done: the question classifier was removed (E4) |
| Unnamed clusters (G9) | Not done: S11 |
| Source-trust shifts | Not done: no trust posteriors yet |

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| P1 | GT0 under by 10–40% for words and entities | **Miss** on size: under as predicted, but words by 49% (entities 23%) |
| P2 | GTu within ±15% (dev) | **Hit**: 0.904, 0.944 |
| P3 | Facts: target ≥ 0.95, GTu within ±5%, GT0 below | **Hit**: 1.000, 0.9997, 0.555 |
| P4 | GTu chosen; the gate passes on test | **Hit**: 0.968, 0.980, 1.000 |
| P5 | Values pooled within ±25%; half the relations within ±25% | **Hit**: 0.956; 15 / 17 (dev), 13 / 14 (test) |
| P6 | Chao (pages) within ±10%, under | **Hit** in the store view (0.977–0.989); **miss** in CyphaLM's (0.71–0.80, an 8× extrapolation) |
| P7 | Heaps vocabulary within ±5%; β in 0.5–0.8; new-word rate within ±30% | Vocabulary **hit** in the store view (1.007, 1.020), **miss** in CyphaLM's (1.12, 1.16); β **hit**; rate **miss** (0.655 dev, 0.701 test) |
| P8 | Chapman(seeds, patterns) ≤ N_ref, every relation | **Hit**: 7 / 7 |
| P9 | CyphaLM's view: GTu within ±25%; words target ≥ 2× | **Hit**: 0.90–0.96; 2.36× (dev), 2.39× (test) |
| P10 | Top cells by reading are person birth/death dates and years | **Hit**: ranks 1–3 (death_year 7) |

### Checks

- `native_epistemic_test`: 536 checks (20 new: clumped statistics, the page-unit and incidence missing mass, the Chao extrapolation). Three planted bugs are caught: q1 for q1_mass in `unit_missing_mass` (2 checks fail); the (n − 1) / n factor dropped from the extrapolation (6); `units <= 1` and `== 3` in `clumped_stats` (1).

## Decisions

1. **Good–Turing with the page as the unit (GTu) is Cypha's missing-mass estimator for anything read from text.** DISCOVERY §3.3's n₁ / N undercounts what is new by 20–50% on page-clumped text; GTu is within 4% on held-out pages. n₁ / N stays right where the draws really are independent.
2. **S4's gate is met** (store view, test: 0.968, 0.980, 1.000).
3. **The frontier report is `pattern_learn --frontier FILE`**, run after `--gaps`. For facts, reading more always brings new facts (missing mass ≈ 1); the useful numbers are the reading rates per cell, the slots unseen on pages already read, and the values' richness.
4. **The Chao extrapolation is trusted to about 2× the pages read.** Beyond it (CyphaLM's 7 MiB to the whole dump) it undercounts by 20–30%.

## For Odin

- DISCOVERY §3.3 names n₁ / N. The measurement says to use the page as the unit for text. Changing the document's formula is your call; the code has both.
- The demand question stays open (HANDOFF §5.7): most of CyphaLM's tags name relations the store does not have.

## Reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
E=/root/cdata/enwik8; W=/root/work/f2/wikidata/wd_values.tsv; DEV=bench/eval/t_qa_wiki_dev.tsv
A=/root/work/answer; Q=/root/work/qa2/runs
D=$A/d_rows.tsv,$A/d2_rows.tsv,$Q/E1a.rows.tsv,$Q/E1b.rows.tsv,$Q/E1c.rows.tsv,$Q/E1d.rows.tsv
O=/root/work/s4/runs; mkdir -p $O/dev2 $O/test
# dev (UNSEEN / UNSEEN_REL / HEAPS / CHAPMAN / FRONTIER lines on stdout; the frontier in frontier.tsv)
./pattern_learn $E $O/dev2 --cv --wikidata $W --dev $DEV --gaps $O/dev2/gap_map.tsv --demand $D --parse $A/parse_dev.tsv --frontier $O/dev2/frontier.tsv
# the test half (read once; it was read with the first frontier definition, which does not touch the estimates)
./pattern_learn $E $O/test --cv --wikidata $W --dev $DEV --gaps $O/test/gap_map.tsv --demand $D --parse $A/parse_dev.tsv --frontier $O/test/frontier.tsv --frontier-test
g++ -O2 -std=c++20 -I native/include native/tools/epistemic_test.cpp -o epistemic_test && ./epistemic_test   # 536 checks
```

- sha256 prefix of the frontier file: `frontier.tsv` 4306e17a24633044.

## S4b: unknown-unknown signals with alarms (2026-10-29)

**Task.** Build the DISCOVERY §3.4 signals that have producers now (surprise in CyphaLM, the unparsed-sentence rate, and S4's vocabulary signal), give them one alarm rule, validate the alarms on planted shifts, and open a gap for each alarm.

(The design, the alarm rule and the predictions below were written and committed before any run. Results and decisions are added under them.)

**Rule 10.** These are Cypha's self-assessment signals. They watch what Cypha reads. They never answer a question for CyphaLM, never feed a gate decision or a tag, and never change a fact.

### Design (fixed before any run)

#### Units and streams (shared by every signal)

- **Unit:** an enwik8 page (`pattern_learn`'s list: main namespace, not a redirect) and its text body (`<text>` to `</text>`, the dump's escaped bytes).
- **Calibration stream C:** article pages whose body starts in [9,437,184, 30,000,000), after CyphaLM's protocol slice, in dump order.
- **Test stream S:** article pages whose body starts in [30,000,000, 50,000,000), in dump order. This is the unshifted stream: held-out text from the training distribution.
- C and S leave out "List of …" and "… (disambiguation)" pages, the two page types the store's reader skips.
- **Planted pool D (an unseen domain):** Dolly responses (`/root/cdata/dolly.jsonl`) of at least 256 bytes, every 10th in file order. Each is written as a dump page (title `Dolly NNNNN`, body = the response, escaped as the dump escapes text) by `scripts/s4b_dolly_pages.py`. CyphaLM's wiki-only model has never read Dolly.
- **Planted pool L (a page type the store has never read):** the "List of …" and "… (disambiguation)" pages of enwik8 whose body starts at or after 9,437,184.
- Each signal has its own unit rule. A page outside that rule is not a unit of that signal and is skipped in its stream.

#### The signals

| Signal | Producer | Score of a unit (high = unusual) | Unit rule | Strata | Gap kind (DISCOVERY §3.2) |
|---|---|---|---|---|---|
| **Surprise (Σ)** | `cyphalm_trie surprise` | Bits per byte of the first ≤ 2,048 body bytes. Each page is read in a fresh context: only `<text xml:space="preserve">` before it | ≥ 256 body bytes | scored bytes 256–1,023, 1,024–2,047, 2,048 | prediction |
| **Unparsed (U)** | `pattern_learn --signals` | −ln P(X ≤ x \| n, q_τ), with X ~ Binomial(n, q_τ). n is the page's eligible sentences, x the parsed ones, and q_τ the parse rate of the page's type over C | n ≥ 3 | page type (country, person, place, other: `write_gaps`' rule) | coverage |
| **Vocabulary (V)** | `pattern_learn --signals` | ln((v + 0.5) / (e + 0.5)). v is the page's words that are new to the reader. e is Heaps' expected new words for its tokens | ≥ 50 word tokens | tokens 50–299, 300–1,499, ≥ 1,500 | lexical |
| **Source-trust shifts** | none | Not done: S8c fits each source's reliability once, on a fixed set of hypotheses. No series of trust posteriors exists to watch. Once a source's answers arrive as a stream, the alarm rule below applies to it unchanged | — | — | — |

Details:

- **Σ's model** is CyphaLM's default expert stack trained on the protocol's wiki train only, [0, 7,340,032), with no chat. The chat-trained serving model has read Dolly, so for it Dolly is not an unseen domain. The wiki-only stack is the one B4B's and A4's fresh BPC reads use.
  - The stack: the packed trie; InfiniGram over the train bytes (cap 8,192); the class expert (the default spec); the similar-context expert (the default spec on this base); and copy (m 8, its window over the page's own bytes).
  - Each expert's lambdas are fit on the whole holdout [7,340,032, 8,388,608), read as one stream, in qa's order: InfiniGram, class, similar-context, copy.
- **U's sentences** follow DISCOVERY §3.2's coverage gap: "sentences [that] mention two or more linked entities but no pattern parses them."
  - Eligible: a prose sentence (`page_sentences`) with at least two entity mentions. A mention is a wiki link or a span of the page's subject (`find_subject`).
  - Parsed: one of the store's kept patterns fires on it, exactly as in extraction (`apply_patterns` with the subject spans).
  - The patterns come from the normal S3 run (`--cv --wikidata`).
- **V's reader** is CyphaLM's view (S4).
  - The starting vocabulary is the words of the pages before byte 7,340,032. Words are S4's: maximal runs of ASCII letters, lower-cased.
  - Heaps is fit on that set's growth curve (every 25th page).
  - The reader then reads C and S in order and adds each page's words.
  - e = K[(N + n)^β − N^β], where N is the tokens read before the page and n its tokens.
  - The pools' scores are taken against the vocabulary at the start of S, adding the pool's own words in pool order. Plants change the S pages after them by a few words; that change is ignored.

#### The alarm rule (one rule for every signal)

1. **Reference.** For each stratum, a window of reference scores.
   - **Running** (primary: "the model's running expectation for that text type"): the last 500 units of that stratum read before this unit. C and S are read as one sequence. A unit is scored only once its window holds at least 100 units.
   - **Fixed** (comparison): all of C's units of that stratum.
2. **Conformal p-value.** p(u) = (#{w: s_w > s_u} + V_u (#{w: s_w = s_u} + 1)) / (M + 1). M is the window's size. V_u ∈ [0, 1) is a hash of the unit's identity, which randomises ties, so U's many tied scores stay exact. If the units are exchangeable, p(u) is uniform.
3. **Event.** e(u) = [p(u) < 0.05]. Under no shift, P(event) = 0.05 for every signal and stratum.
4. **Detector.** Shiryaev–Roberts on the events, against the registered alternative p1 = 0.25:
   - R_t = (1 + R_{t−1}) Λ_t, with Λ_t = 5 for an event and 0.75 / 0.95 otherwise, and R_0 = 0.
   - **Alarm** at R_t ≥ A = 20,000; R then resets to 0.
   - Under no shift, with independent units, R_t − t is a martingale. So P(an alarm within the first L units) ≤ L / A: at most 0.05 alarms per 1,000 units.
   - An event rate below about 0.13 cannot raise an alarm (E ln Λ < 0). This is by design: only a sustained rise is a region of ignorance.
5. **Gaps.** Every alarm opens one gap in S3's gap-map form, written to its own file, `--signals-gaps FILE` (S3's map is not touched).
   - The columns are S3's: gap_id, kind, status, entity, type, relation, p_exists, f_store, f_wd, p, eig_bits, demand, voi, note.
   - kind is prediction, coverage or lexical; status is open; the entity is the page where the alarm fired; the type is the stratum; the relation is `-`.
   - p is the event share since the last reset; p_exists, f_store and f_wd are −1; eig_bits and voi are 0.
   - The note holds the stream position, R, and the first page of the run.

#### Validation (what counts as a valid alarm)

- **False alarms.** Every alarm on the unshifted stream S is a false alarm. Reported:
  - the alarms on S in dump order (both references) and the event rate on S;
  - the mean alarms per 1,000 units over 200 random orders of S (fixed reference);
  - the alarms on C (running reference, after its warm-up).
- **Planted shifts.** 200 replicates per (signal, pool, mixture π ∈ {1, 0.5, 0.2}, reference).
  - The planted point t0 is uniform over S's units [100, |S| − 200].
  - From t0 on, each unit is a plant with probability π (drawn from the pool without replacement, in random order), else S's next unit.
  - The detector's state at t0 is the unshifted run's.
- **A valid alarm** (a detection) is the first alarm at or after t0 and within 200 units of it. Its delay is the units read from t0 to the alarm, both counted (1 = the first plant).
- An alarm before t0 is a false alarm, already counted on S. A replicate with no alarm in [t0, t0 + 200) is a miss.
- **A signal passes for a shift** (running reference) when both hold:
  - S in dump order has at most one alarm;
  - at π = 1, at least 90% of the replicates detect within 50 units.
- π = 0.5 and 0.2 are reported, not gated.
- **Decision.** A signal that passes for a shift is watched by Cypha for that kind of change, and its alarms open gaps. A signal that passes for no shift stays built and reported, with the reason it cannot see the shifts.

### Predictions (registered before any run)

| # | Prediction |
|---|---|
| B1 | Σ: on S (running reference) the event rate is in [0.03, 0.08] and there is at most one alarm. Over 200 random orders (fixed reference) at most 0.1 alarms per 1,000 units |
| B2 | Σ detects D (Dolly as pages): at π = 1, at least 90% of the replicates within 50 units, median delay ≤ 10 units. Dolly's conversational prose without wiki markup is surprising to the wiki-only model |
| B3 | Σ does not detect L: at π = 1, fewer than half the replicates within 200 units. Lists and disambiguation pages are ordinary wiki text to CyphaLM |
| B4 | U: the unparsed rate over C is 0.80–0.95 for person pages and ≥ 0.95 for the other types. Fewer than 10% of D's pages are U units (no links, no subject), so U cannot see D (no detection). U does not detect L (few eligible sentences) |
| B5 | U and V on S: event rate in [0.03, 0.08], at most one alarm each (running reference) |
| B6 | V does not detect D: Dolly's words are common, so its new-word ratio is below a wiki page's (at π = 1, fewer than half within 200 units) |
| B7 | V detects L: lists and disambiguation pages are dense in names never read before (at π = 1, at least 90% within 50 units) |
| B8 | With the fixed reference, V's event rate on S is below 0.05 (S4: Heaps overshoots in CyphaLM's view, so later pages bring fewer new words than predicted). Σ's and U's fixed-reference event rates are within [0.03, 0.08] |
| B9 | No signal detects every shift: Σ sees D, V sees L, and U sees neither. This is why the signals are watched together |

### Results (run 2026-10-29)

**The runs:**
- `cyphalm_trie surprise` scored 4,281 pages (7.2 MB) in two parts. Each part took 22.6 min and about 220 MB, and the two ran in parallel on a loaded machine.
- On the holdout the lambdas were fit on, the stack gives 2.086 (trie + InfiniGram), 2.008 (+ class), 1.944 (+ similar-context) and 1.851 (+ copy).
- The similar-context expert is A4c's default on this base (probe 32). A4d changes the default; a rerun follows it.
- `pattern_learn --signals` is S3's run plus the signals: 42 s, 154 MB peak.

**Units:**

| Signal | C | S | D | L |
|---|---:|---:|---:|---:|
| Surprise Σ (pages ≥ 256 bytes) | 1,829 | 1,623 | 616 | 213 |
| Unparsed U (n ≥ 3) | 1,543 | 1,364 | **0** | 21 |
| Vocabulary V (≥ 50 tokens) | 1,820 | 1,606 | 535 | 205 |

**The signals' levels:**

- **Σ, mean bits per byte:** C 2.006, S 1.947, D 2.194, L 2.134.
  - In the shortest band (256–1,023 bytes, where 512 of D's 616 pages fall): C 2.083, D 2.210, L 2.237.
  - Dolly's plain prose is only about 0.13 bits per byte more surprising than a wiki page of the same length.
- **U, unparsed rate over C by type:**
  - country 0.9953 (848 eligible sentences), person 0.9664 (6,226), place 1.0000 (4,163), other 1.0000 (19,215).
  - The kept patterns parse person sentences only, plus 4 country sentences.
- **V, new words per Heaps' expectation:** C 0.834, S 0.772, D 0.624, L 2.658.
  - Heaps on the reader's 536 pages (1.02M tokens): K 6.337, β 0.661.
  - Dolly brings *fewer* new words than a wiki page of its length. Lists and disambiguation pages bring 2.7 times more.

**False alarms (the unshifted streams):**

| Signal | Reference | C: scored, event rate, alarms | S: scored, event rate, alarms | S, 200 random orders |
|---|---|---|---|---|
| Σ | running | 1,529, 0.0510, 0 | 1,623, 0.0493, **1** | — |
| Σ | fixed | — | 1,623, 0.0407, 1 | 0.006 alarms per 1,000 |
| U | running | 1,230, 0.0553, 0 | 1,345, 0.0625, **0** | — |
| U | fixed | — | 1,345, 0.0647, 0 | 0.123 per 1,000 |
| V | running | 1,520, 0.0513, 0 | 1,606, 0.0554, **1** | — |
| V | fixed | — | 1,606, 0.0604, 2 | 0.084 per 1,000 |

- **Σ, on the training distribution's held-out text:** 0 alarms in C's 1,529 pages and 1 in S's 1,623 (0.62 per 1,000 pages). The event rate is 0.049–0.051.
- **The alarms on S are runs of pages on one topic.** The registered rule counts them as false alarms, because S is unshifted. Both are regions the reader had not met, which is what the signals watch for.
  - V at *Carolus Linnaeus* (S page 37), after Carnivora, Cheirogaleidae, Callitrichinae, Cebidae and Chondrichthyes: Latin names new to the reader.
  - Σ at *Chrono Trigger* (S page 793), after the Cleveland Browns, Carolina Panthers, Chicago Bears and Cincinnati Bengals pages: their leads are infoboxes at 2.45–2.72 bits per byte.
  - V with the fixed reference also fired at *Colin Maclaurin* (page 1,435).
- **The fixed reference drifts:**
  - U's and V's event rates on S are 0.065 and 0.060, so their random-order false-alarm rates exceed the 0.05 bound (0.12 and 0.08 per 1,000). The bound assumes the null rate is right.
  - Σ's rate falls to 0.041, because S's pages are less surprising than C's.
  - The running reference keeps every rate within 0.049–0.063.

**What a shift brings.** The event rate of each pool against the fixed reference:

| Signal | D | L |
|---|---:|---:|
| Σ | 0.104 | 0.188 |
| U | no units | 0.048 |
| V | 0.022 | 0.288 |

The rule raises an alarm only on a sustained rate above about 0.13 (p1 = 0.25).
- Σ on D (0.104) sits below that floor.
- Even the best test of 0.104 against 0.05 needs about ln A / KL = 9.9 / 0.024 ≈ 420 pages.
- So no detector can see Dolly within 200 pages from this score.

**Detection** (running reference). For each case: replicates detected within 50 and within 200 of 200, then the median delay (90th percentile) in pages.

| Signal | Pool | π = 1 | π = 0.5 | π = 0.2 |
|---|---|---|---|---|
| Σ | D | 12 / 47 | 8 / 33 | 5 / 27 |
| Σ | L | 112 / 200; 45 (92) | 41 / 116; 158 | 7 / 37 |
| U | D | invisible: no units | — | — |
| U | L | 0 / 0 | 0 / 0 | 0 / 0 |
| V | D | 0 / 0 | 0 / 0 | 0 / 0 |
| V | L | **191 / 200; 24 (44)** | 73 / 187; 63 (172) | 6 / 55 |

- With the fixed reference the numbers are close: V on L 192 / 200, median 22; Σ on L 102 / 200, median 50; Σ on D 7 / 63.
- At π < 1 some alarms fall on unshifted pages. For V on L, 155 of the 187 alarms at π = 0.5 fell on a plant.

**The registered verdict:**

| Signal | S in dump order: ≤ 1 alarm | D: ≥ 90% within 50 at π = 1 | L: the same |
|---|---|---|---|
| Σ | pass (1) | fail (6%) | fail (56%; all 200 within 200) |
| U | pass (0) | fail (invisible) | fail (0%) |
| V | pass (1) | fail (0%) | **pass (95.5%, median 24 pages)** |

**Gaps.** Each of the two running-reference alarms on S opened one gap in `signal_gaps.tsv`, in S3's columns:
- `G-C6C0 lexical open Carolus Linnaeus`;
- `G-4615 prediction open Chrono Trigger`.

The first detection of each planted pool is in `signal_gaps.tsv.planted.tsv` (3 rows; for example `lexical` at *List of people by name: Q*, 16 pages after the plant). S3's gap map is unchanged.

**Source-trust shifts:** not done. S8c gives no series to watch.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| B1 | Σ on S: event rate in [0.03, 0.08], ≤ 1 alarm, ≤ 0.1 per 1,000 in random orders | **Hit**: 0.049, 1 alarm, 0.006 |
| B2 | Σ detects D (≥ 90% within 50, median ≤ 10) | **Miss**: 6% within 50, 24% within 200. Dolly is only about 0.13 bits per byte above wiki pages of its length |
| B3 | Σ does not detect L | **Miss**: Σ detects L in every replicate within 200 pages (median 45). Lists and disambiguation pages surprise CyphaLM more than Dolly does |
| B4 | U: person 0.80–0.95, others ≥ 0.95; < 10% of D are units; U does not detect L | Person **miss** (0.966); others **hit** (0.995–1.000); D **hit** (0 units); L **hit** (0 / 200) |
| B5 | U and V on S: event rate in [0.03, 0.08], ≤ 1 alarm each | **Hit**: 0.063 and 0.055; 0 and 1 alarms |
| B6 | V does not detect D | **Hit**: 0 / 200 (D's event rate 0.022) |
| B7 | V detects L (≥ 90% within 50) | **Hit**: 95.5%, median 24 pages |
| B8 | Fixed reference: V's rate on S < 0.05; Σ's and U's in [0.03, 0.08] | V **miss** (0.060); Σ and U **hit** (0.041, 0.065) |
| B9 | Σ sees D, V sees L, U neither | **Miss**: Σ and V both see L; nothing sees D |

### Checks

- **`native_epistemic_test`:** 1,019 checks, 44 of them new (`test_signals_S4b`). The new checks cover:
  - the conformal p-value and its tie rule;
  - the Shiryaev–Roberts path of an all-event stream (an alarm at the 7th unit);
  - a flat stream silent over 100,000 units;
  - the L / A bound (2,000 streams of 1,000 units);
  - planted step changes detected (100 of 100 within 50 units, median 6–12), with no detection from plants drawn from the stream itself;
  - the running window, the delay statistics and U's binomial score.
- **Three planted bugs are caught:** the tie count without its +1 (3 checks fail); R not reset after an alarm (3); the running window not reversed (1).
- **The 9 CTests of HANDOFF §4.4 pass.**
- **S3 parity.** Without `--signals`, every file of S3's run is byte-identical to the aca60a5 build's: gap map `a77678d8ef7dfdef`, cells `eed14241a6d6eff2`, demand `dd1c8ddac831dcb6`, facts, patterns and seeds. With `--signals` those files are still identical; `stats.txt` gains the SIGNAL lines.
- **`cyphalm_trie`.** On a 1 MB split, `selftest`, `bpc`, `context8k` and `cls` print the same as the aca60a5 build, apart from one build-time field.

## S4b decisions

1. **The vocabulary signal (V, CyphaLM's view) is the watched alarm for text of a new type.** It is the only signal that passes for a shift (L, 95.5% within 50 pages). Its alarms open lexical gaps.
2. **Surprise (Σ) stays built and reported, not passing.**
   - It is valid on unshifted text: 0 and 1 alarms on C and S, event rate 0.05.
   - It sees new page types (L) in every replicate within 200 pages, but slower than the 50-page bar.
   - It cannot see Dolly: per page, the wiki-only stack finds Dolly's prose only 0.13 bits per byte harder than a wiki stub.
3. **The unparsed signal (U) stays built, not passing.**
   - The store's kept patterns parse only person sentences (3.4% of them). Place and other pages are 100% unparsed, so U has no room to rise.
   - Text with no links and no known subject (Dolly) yields no eligible sentence, so U never sees it.
   - U becomes a signal when extraction covers more relations (DISCOVERY P10).
4. **No signal here sees an unseen domain written in plain prose (D).** Σ and V see changes of form: lists, names, infoboxes. A content-level signal is needed for D: the OOD rate, DISCOVERY's first §3.4 signal.
5. **Source-trust shifts: not done.** S8c fits each source once, so there is no series to watch.

## S4b for Odin

- The two alarms on the unshifted stream are runs of pages on one topic: taxonomy pages and American football team infoboxes. The registered rule calls them false. As gaps, they point at regions the reader had not met. If such runs should not open gaps, A = 100,000 silences both (their R was 67,000 and 61,000). It costs about 5 more pages of delay on V's L detection at π = 1: ln 5 over V's drift on L, about 0.30 per page.

## S4b reproduce

```bash
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target cyphalm_trie epistemic_test -j1
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
R=/root/work/s4b/runs; mkdir -p $R/sig $R/pl/full
python3 scripts/s4b_dolly_pages.py /root/cdata/dolly.jsonl $R/dolly_pages.xml        # 616 pages
for k in 0 1; do   # from the repository root (the similar-context vectors); the two parts can run in parallel
  build/cyphalm_trie surprise /root/cdata/enwik8 --out $R/sig/surprise.$k.tsv --plant $R/dolly_pages.xml --part $k/2
done
E=/root/cdata/enwik8; W=/root/work/f2/wikidata/wd_values.tsv; DEV=bench/eval/t_qa_wiki_dev.tsv
A=/root/work/answer; Q=/root/work/qa2/runs
D=$A/d_rows.tsv,$A/d2_rows.tsv,$Q/E1a.rows.tsv,$Q/E1b.rows.tsv,$Q/E1c.rows.tsv,$Q/E1d.rows.tsv
O=$R/pl/full
./pattern_learn $E $O --cv --wikidata $W --dev $DEV --gaps $O/gap_map.tsv --demand $D --parse $A/parse_dev.tsv \
  --signals $O --signals-dolly $R/dolly_pages.xml --signals-gaps $O/signal_gaps.tsv --surprise $R/sig/surprise.0.tsv,$R/sig/surprise.1.tsv
build/epistemic_test   # 1,019 checks
```

- The SIGNAL lines are on stdout. Per-page scores: `units.tsv` (11,375 rows, sha256 prefix 9ef72f7c87b09d26). Gaps: `signal_gaps.tsv` (ad9133c8122637f7) and `.planted.tsv` (8b2dc7b9a23800fe). Surprise files: 0f97e1c7c29b9574 and b7be672fc81a5ca8.

## S4c: an out-of-scope signal that reads content (2026-10-29)

**Odin Loch** · 2026-10-29 · follows S4b's decision 4

**Task.** S4b found that no signal sees an unseen domain written in plain prose. Surprise rises only about 0.13 bits per byte on Dolly pages, the unparsed rate cannot rise, and vocabulary misses it. S4c builds DISCOVERY §3.4's first signal, out-of-scope inputs, so that it reads the words of a page and not its markup. It runs under S4b's alarm rule and on the same kinds of planted shifts.

(The design, the selection rule and the predictions below were written and committed before any S4c run. Results and decisions are added under them.)

**Rule 10.** Like S4b's signals, this one watches what Cypha reads. It never answers a question, never feeds a gate or a tag, and never changes a fact.

### Design (fixed before any run)

#### What the signal reads: prose words, never markup

- **The reader.** Each page is read by S3's prose reader (`page_sentences`): lines outside templates, tables, list items and headings, with references removed and every link reduced to its shown words.
- **One more list rule, for every page.** Lines that start with `-`, `•` or a number followed by `.` or `)` are list items too, and are dropped. The prose reader already drops wiki lists (`*`, `#`); Dolly writes its lists with `-` and `1.`. So a list never enters, whatever its markup.
- **Words.** The reader's word tokens that start with an ASCII letter, lower-cased (letters with inner apostrophes and hyphens).
- **The window.** The first m = 50 words of a page's prose. A page with fewer than 50 prose words is not a unit. One window of fixed size for every page means no score depends on page length.
- **The reference.** Word counts over all prose words of the reader pages: body before byte 7,340,032 (CyphaLM's train), not lists or disambiguation pages. F is its 100 most frequent words (ties by the word).

#### The candidates (one is chosen on dev)

| # | Candidate | Score of a window (high = unusual) |
|---|---|---|
| O1 | **Register** | G per word: (2 / m) Σ_b c_b ln(c_b / (m p_b)). The bins b are the 100 words of F and one bin for every other word; c_b is the window's count and p_b the reference's share |
| O2 | **Rare content words** | The mean of −ln p̂(w) over the window's words not in F, with p̂(w) = (n_w + 0.5) / (N + 0.5 (V + 1)) from the reference (n_w the word's count, N its words, V its distinct words) |
| O3 | **Topic distance** | 1 − the largest cosine between the page's vector and the reader pages' vectors. A page's vector is Σ a / (a + f_w) · vec_w over the window's words not in F that have a vector (`data/cyphalm/simvec_ppmi32.bin`), with a = 0.001 and f_w the word's reference share, then normalised. No such word: score 1 |

- O1 reads how a page is written (which common words it uses, and how often). O2 and O3 read what it is about.
- The vectors were built from enwik8 without the BPC holdout and the fresh text. They have read C's and S's words, but never Dolly.
- **Strata:** the page's type (country, person, place, other), as for U. Dolly pages are "other".
- **The alarm rule:** S4b's, unchanged (events at p < 0.05, Shiryaev–Roberts with p1 = 0.25, alarm at A = 20,000, running windows of 500, scored after 100).
- **Gaps.** An alarm opens a gap in S3's form in the `--signals-gaps` file, as in S4b. Its kind is `scope`. DISCOVERY §3.2 has no kind for out-of-scope text, so this is a proposal for Odin.

#### The control: a wiki page with its markup removed must score as itself

- **Pool P.** Article pages whose body starts at or after byte 50,000,000 (neither C, S nor S4b's reader has read them), not lists or disambiguation pages, every 4th in dump order. `scripts/s4c_plain_pages.py` writes each one as a Dolly-style page, with a separate renderer written in Python. It removes templates, tables, references, comments, tags, categories and images; it turns links into their shown words; it drops headings; and it writes list items as `- item`.
- **Two checks:**
  - **Same score.** For each P page, its score against the score of its source page in the dump.
  - **No shift.** P planted as a shift, like D. A signal that fires on P is reading markup, not content.

#### Reads

- **Dev (for the choice).** Calibration: C's pages whose body starts in [9,437,184, 19,718,592). Stream: C's pages in [19,718,592, 30,000,000). Pools: D-dev (Dolly records with an even line number in `dolly_pages.xml`), P-dev (P's pages with an even index), and L.
- **Test (the gate, read once).** S4b's streams, C then S. Pools: D-test (odd line numbers), P-test (odd index), and L.
- S4b read C, S, D and L with its own signals. No S4c score has been computed on any page before this commit. The dev read never scores S.
- Planted shifts, delays and replicates are S4b's: 200 replicates per pool, π ∈ {1, 0.5, 0.2}, both references, t0 uniform over the stream's units [100, |stream| − 200].

#### The choice on dev (fixed now)

- A candidate is **eligible** when, on the dev stream with the running reference, it has at most one alarm, and P-dev at π = 1 is detected within 50 units in at most 10% of the replicates.
- The chosen candidate is the eligible one with the most D-dev detections within 50 at π = 1. Ties go to the lower median delay, then to O1, O2, O3 in that order.
- If none is eligible, the one with the most D-dev detections is read on test, and is reported as having failed on dev.
- m, F's size, a, the strata and the list rule are fixed above. Dev chooses the candidate only.

#### The gate (test, read once, the chosen candidate)

The signal passes for an unseen domain in plain prose when all three hold:
1. S in dump order (running reference): at most one alarm.
2. D-test at π = 1: at least 90% of 200 replicates detected within 50 units.
3. P-test at π = 1: detected within 50 units in at most 10% of the replicates.

Detection shares get Wilson 95% intervals. L, π = 0.5 and 0.2, the fixed reference and the other candidates are reported, not gated.

### Predictions (registered before any run)

| # | Prediction |
|---|---|
| E1 | O1 (register) is chosen on dev |
| E2 | O1 on D-dev: at least 90% of replicates within 50 units at π = 1, median delay at most 10 |
| E3 | O2 does not detect D (fewer than half the replicates within 200 at π = 1): Dolly's words are common, as V found |
| E4 | O3 detects D in 30–90% of replicates within 50 at π = 1: Dolly's topics overlap wiki topics in part |
| E5 | Every candidate on the dev stream (running reference): event rate in [0.03, 0.08], at most one alarm |
| E6 | P: for O1 and O2, at least 90% of P pages score exactly as their source page; every candidate detects P-dev within 50 in at most 5% of replicates |
| E7 | The chosen candidate passes the test gate, with the Wilson lower bound of D-test's share at least 0.85 |
| E8 | Fewer than 30% of L's pages are units: a list page is mostly list items, which the reader skips |

### Round 2 (registered after round 1's dev read, before any test read)

**Round 1 on dev** (the full lines are in "Results" below). Every candidate is eligible: no alarm on the dev stream, and P-dev is never detected within 50 units. None comes near the bar on D-dev. At π = 1 with the running reference, the shares within 50 units are O1 1 / 200, O2 0 / 200 and O3 50 / 200. Against the fixed reference, D's event rate is O1 0.145, O2 0.056 and O3 0.184. A pass needs about 0.3.

**Why O1 is weak.** F holds the reference's 100 most frequent words. Words that wiki prose rarely uses and other registers use often all fall in O1's one "other" bin. O1 also weights each bin by its multinomial variance. Real pages vary far more than that in some words (he, his, she) and far less in others.

**Round 2 adds one candidate. Nothing else changes** (m, the reader, the list rule, the strata, the alarm rule, the reads, the choice rule and the gate):

| # | Candidate | Score of a window (high = unusual) |
|---|---|---|
| O4 | **Register profile (Burrows' Delta)** | (1 / (K + 1)) Σ_b \|x_b − μ_b\| / σ_b over the reference's K = 500 most frequent words and one bin for every other word. x_b is the bin's share of the window's 50 words. μ_b and σ_b are the bin's mean and standard deviation over every whole 50-word window of the reader pages' prose (windows do not cross pages). σ_b is at least √(μ_b (1 − μ_b) / 50) |

- O4 measures how far a window's use of the 500 commonest words is from wiki prose, in units of how much wiki windows vary. It is the stylometric distance of Burrows (2002), used here as a one-class score.
- **The choice** is the registered rule over O1–O4.
- **Predictions (round 2):**

| # | Prediction |
|---|---|
| E9 | O4 is eligible: at most one alarm on the dev stream, and P-dev within 50 in at most 10% of replicates |
| E10 | O4's D event rate against the fixed reference on dev is at least 0.25, and O4 is chosen |
| E11 | O4 on D-dev at π = 1 (running reference): at least 70% of replicates within 50 units |

### Round 3 (registered after round 2's dev read, before any test read)

**Round 2 on dev.** O4 is flat. Its mean on D-dev is 0.2865, against 0.2870 on the calibration half, and D's event rate is 0.034. With 50 words, most of the 501 bins are empty, so Delta counts mid-frequency words, not register. O4 also had one alarm in the calibration half (*Transportation in Brazil*).

**What D-dev is (dev analysis, no test page read).** By Dolly category, the area under the ROC curve against the dev stream's "other" pages:

| Candidate | All D-dev (234) | brainstorming (25) | creative writing (29) | general QA (70) | open QA (53) | closed QA (17) | summarisation (22) | information extraction (11) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| O1 register | 0.602 | 0.634 | 0.616 | 0.635 | 0.573 | 0.516 | 0.572 | 0.423 |
| O3 topic | 0.654 | 0.723 | 0.722 | 0.628 | 0.684 | 0.679 | 0.560 | 0.523 |

- Dolly answers of 256 bytes or more are mostly encyclopedic prose: "Lollapalooza is an annual musical festival held in Grant Park in Chicago, Illinois. It was started in 1991 …". Closed QA, summarisation and information extraction rewrite Wikipedia passages, and many open and general QA answers read like a lead paragraph.
- So D is mostly the same register as wiki prose. What differs is what it is about: topics the reader has not met, many after 2006. Topic distance (O3) separates best, in every category.

**Round 3 adds one candidate. Nothing else changes:**

| # | Candidate | Score of a window (high = unusual) |
|---|---|---|
| O5 | **Topic novelty (TF-IDF, the first-story-detection score)** | 1 − the largest cosine between the window's TF-IDF vector and each reader page's. Vectors use words outside F: tf × idf, with idf(w) = ln((N_d + 1) / (df_w + 1)) + 1 over the N_d reader pages (a word on no reader page gets ln(N_d + 1) + 1). A reader page's vector uses all of its prose words. No word outside F: score 1 |

- The bank is the same as O3's (the reader pages), so the two differ only in how a text is represented: 32 projected dimensions, or every word.
- **The choice** is the registered rule over O1–O5. After this round the choice is final, and the test is read once.
- **Predictions (round 3):**

| # | Prediction |
|---|---|
| E12 | O5 is eligible, and its AUROC on D-dev against the dev stream's "other" pages is above O3's (0.654) |
| E13 | No candidate reaches 90% of D-dev replicates within 50 at π = 1 with the running reference, so the chosen one fails the test gate. Most of D is encyclopedic text that a content signal should not call out of scope |

### The dev choice (fixed before the test read)

Dev read, π = 1, 200 replicates each. "Stream alarms" is the dev stream (C's second half, 752 units) with the running reference.

| Candidate | Stream alarms | P-dev within 50 | D-dev within 50 (running) | D-dev within 200 | D-dev AUROC against the stream's "other" pages | Eligible |
|---|---:|---:|---:|---:|---:|---|
| O1 register | 0 | 0 / 200 | 1 / 200 | 14 / 200 | 0.602 | yes |
| O2 rare content words | 0 | 0 / 200 | 0 / 200 | 3 / 200 | 0.488 | yes |
| **O3 topic distance** | 0 | 0 / 200 | **50 / 200** | 147 / 200 | 0.654 | yes |
| O4 Delta (round 2) | 0 | 0 / 200 | 0 / 200 | 0 / 200 | 0.515 | yes |
| O5 TF-IDF novelty (round 3) | 1 | 0 / 200 | 10 / 200 | 55 / 200 | 0.611 | yes |

**The choice: O3, topic distance.** It is eligible and has the most D-dev detections within 50 units. On dev it misses the gate's 90% by far (25%), so E13's expected fail is the likely test outcome. The test is read once, with O3 as the gated signal. The other four are reported beside it, not gated.

### Results (run 2026-10-29)

**The runs.** `pattern_learn --signals --signals-scope dev|test` is S3's run, S4b's signals and S4c's candidates: 30–32 s and 190 MB peak. Pool P: `scripts/s4c_plain_pages.py` wrote 1,022 of the 4,088 qualifying pages after byte 50,000,000. The dev read ran three times, once per round; each round only added a candidate (O1–O3's lines are identical across rounds). The test read ran once.

**Units** (pages with at least 50 prose words):

| Read | Calibration | Stream | D | L | P |
|---|---:|---:|---:|---:|---:|
| Dev | 926 of 1,015 | 752 of 830 | 234 of 307 (76%) | 55 of 218 (25%) | 469 of 511 |
| Test | 1,678 of 1,845 (C) | 1,474 of 1,640 (S) | 237 of 309 (77%) | 55 of 218 (25%) | 451 of 511 |

**The control, check 1: the same page with its markup removed.** Of the 446 P-test pages that are units in both forms, 353 (79%) give the reader exactly the same 50 words. The rest differ because the two renderers disagree at the edges: a template inside a sentence, a caption, an indented line. The mean score change is small against each score's spread: O1 0.026 (its mean is about 1.0), O3 0.006 (about 0.15), O5 0.006 (about 0.87).

**The control, check 2: P planted as a shift.** No candidate detects P. At π = 1 with the running reference, the shares within 50 units on test are O1 0 / 200, O2 0, O3 3 / 200 (1.5% [0.5%, 4.3%]), O4 0 and O5 0. Dev's were all 0 / 200. So no candidate reads markup.

**False alarms (test, running reference):**

| Candidate | S: event rate, alarms | C: alarms (warm-up, not gated) | S, 200 random orders (fixed) |
|---|---|---|---|
| O1 | 0.060, 0 | 0 | 0.165 per 1,000 |
| O2 | 0.048, 0 | 0 | 0 |
| **O3** | **0.041, 0** | 0 | 0 |
| O4 | 0.050, 0 | 1 (*Transportation in Brazil*) | 0.086 |
| O5 | 0.043, 0 | 2 (*Athlon*, *Bayonne*) | 0.003 |

**Detection (test, π = 1, 200 replicates).** Within 50 and within 200 units, then the median delay:

| Candidate | D, running | D, fixed | L, running | P, running |
|---|---|---|---|---|
| O1 register | 52 / 124; 102 | 86 / 200; 59 | 1 / 43 | 0 / 2 |
| O2 rare words | 0 / 2 | 0 / 0 | 0 / 0 | 0 / 0 |
| **O3 topic distance** | **76 / 151; 66** | 34 / 190; 100 | 0 / 0 | 3 / 12 |
| O4 Delta | 0 / 0 | 0 / 0 | 9 / 113; 129 | 0 / 2 |
| O5 TF-IDF novelty | 1 / 7 | 8 / 28 | **200 / 200; 16** | 0 / 0 |

- O3 at π = 0.5: 19 / 53 within 50 / 200; at π = 0.2: 4 / 11.
- D's event rate against the fixed reference: O1 0.173, O3 0.144, O5 0.089, O2 0.055, O4 0.046. A pass needs about 0.3.

**The registered verdict (O3, the dev choice):**

| Condition | Result |
|---|---|
| 1. S in dump order: at most one alarm | **pass** (0) |
| 2. D-test at π = 1: at least 90% within 50 | **fail**: 76 / 200 = 0.380 [0.316, 0.449] (within 200: 0.755 [0.691, 0.809]) |
| 3. P-test at π = 1: at most 10% within 50 | **pass**: 0.015 [0.005, 0.043] |

**O3 does not pass.** No candidate does.

**Why: what Dolly-as-pages is.** By Dolly category, O3's AUROC on test against S's "other" pages is 0.675 overall. It is highest where Dolly writes in its own voice: brainstorming 0.841 (20 units), classification 0.748 (8), creative writing 0.739 (28) and general QA 0.714 (66). It is lowest where Dolly rewrites Wikipedia: summarisation 0.579 (29), information extraction 0.606 (5), open QA 0.612 (71) and closed QA 0.615 (10). Most of D (the 256-byte-and-longer answers S4b planted) is encyclopedic prose about a topic. By content it is in scope. What sets it apart is mainly what it is about, and topic distance sees that part.

**Found on the way (not gated).** O5, TF-IDF novelty, detects L (list and disambiguation pages) in 200 of 200 replicates within 50 units on both reads, median 16 (S4b's V: 95.5%, median 24). It has no alarm on S. It does see L from the words of their few prose lines; only 25% of L pages have 50 prose words.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| E1 | O1 is chosen | **Miss**: O3 is chosen |
| E2 | O1 on D-dev ≥ 90% within 50, median ≤ 10 | **Miss**: 1 / 200 |
| E3 | O2 does not detect D | **Hit**: 3 / 200 within 200 (dev), 2 / 200 (test) |
| E4 | O3 detects D in 30–90% within 50 | **Miss on dev** (25%); in range on test (38%) |
| E5 | Every candidate on the dev stream: rate in [0.03, 0.08], ≤ 1 alarm | **Hit**: 0.044–0.063; O5 1 alarm, the rest 0 |
| E6 | ≥ 90% of P pages score as their source (O1, O2); P-dev ≤ 5% | **Miss** for the first part (O1 80%, O2 77% on dev); **hit** for the second (0 / 200 for every candidate) |
| E7 | O3 passes the test gate, lower bound ≥ 0.85 | **Miss**: 0.380 [0.316, 0.449] |
| E8 | Fewer than 30% of L's pages are units | **Hit**: 25% |
| E9 | O4 eligible | **Hit** |
| E10 | O4's D event rate ≥ 0.25 and O4 chosen | **Miss**: 0.034 (dev) |
| E11 | O4 ≥ 70% within 50 on D-dev | **Miss**: 0 / 200 |
| E12 | O5's AUROC on D-dev above O3's | **Miss**: 0.611 against 0.654 |
| E13 | No candidate reaches 90% on dev; the chosen one fails the test gate | **Hit** |

### Checks

- **`native_epistemic_test`:** 1,119 checks, 46 of them new (`test_scope_S4c`). The new checks cover:
  - the list rule;
  - the reference and F, with ties;
  - O1–O5 against values worked by hand;
  - O4's binomial floor;
  - an end-to-end case through S4b's rule: a planted language with another register is detected within 50 units in at least 90 of 100 replicates, and more of the reference language in at most 5.
- **Six planted bugs are caught:** the number list rule removed (2 checks fail); O1 without its factor 2 (2); O4 without its floor (2); O3 returning the cosine, not the distance (2); idf without its + 1 (1); O2 counting F's words (2).
- **S3 parity.** Without `--signals`, all 10 files of S3's run are byte-identical to main 7f77615's, and stdout matches apart from timings.
- **S4b parity.** With `--signals` and without `--signals-scope`, all 13 files (gap map, units, signal gaps, planted gaps, facts, patterns and seeds) and stdout are S4b's, byte for byte.
- **The nine CTests of HANDOFF §4.4 pass** (CMake Release build of their targets; `native_epistemic_test` 1,119 checks).

## S4c decisions

1. **No out-of-scope signal is adopted.** O3 (topic distance) was chosen on dev and read once: 38% of Dolly shifts are detected within 50 pages, against the 90% bar. It stays built and reported, like Σ and U.
2. **No candidate reads markup.** Wiki pages with their markup removed (P) are never detected as a shift, and on most pages they score exactly as their source. The reader (prose lines only, list items in any markup dropped, the first 50 words) is the one to use for any later content signal.
3. **Dolly-as-pages is mostly in scope by content.** Answers of 256 bytes or more are mostly encyclopedic prose. A content signal should not call them out of scope, and none does. Only the part in Dolly's own voice (brainstorming, creative writing) separates well (AUROC 0.74–0.84 with O3). A cleaner planted pool for "an unseen domain in plain prose" would be text with neither wiki's topics nor its register, such as Dolly's brainstorming and creative-writing answers, or another corpus.
4. **The `scope` gap kind** is written for the O alarms in the run's own gap file (three on C in the test read: O4's and O5's). Since no O signal is adopted, none of them opens a gap in Cypha's map.

## S4c for Odin

- **A different planted pool?** S4b's D is mostly encyclopedic prose, so the gate asks a content signal to flag in-scope text. Should the out-of-scope gate be read on Dolly's own-voice answers only (brainstorming, creative writing, general QA), or on another corpus? That needs a new registration and a fresh read.
- **O5 for new page types?** O5 sees lists and disambiguation pages faster than V (100% within 50, median 16 against 95.5%, median 24), with no alarm on S. L was not S4c's gate, so it is not adopted here. Adopt it beside V under S4b's rule, or leave V alone?
- **A `scope` gap kind** for DISCOVERY §3.2, for out-of-scope text, if a signal ever passes.

## S4c reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
g++ -O2 -std=c++20 -I native/include native/tools/epistemic_test.cpp -o epistemic_test && ./epistemic_test   # 1,119 checks
R=/root/work/s4c/runs; S4B=/root/work/s4b/runs   # S4b's Dolly pages and surprise files (S4b reproduce)
python3 scripts/s4c_plain_pages.py /root/cdata/enwik8 $R/plain_pages.xml   # 1,022 pages
E=/root/cdata/enwik8; W=/root/work/f2/wikidata/wd_values.tsv; DEV=bench/eval/t_qa_wiki_dev.tsv
A=/root/work/answer; Q=/root/work/qa2/runs
D=$A/d_rows.tsv,$A/d2_rows.tsv,$Q/E1a.rows.tsv,$Q/E1b.rows.tsv,$Q/E1c.rows.tsv,$Q/E1d.rows.tsv
for read in dev test; do   # from the repository root (the vectors); dev never scores S
  O=$R/$read; mkdir -p $O
  ./pattern_learn $E $O --cv --wikidata $W --dev $DEV --gaps $O/gap_map.tsv --demand $D --parse $A/parse_dev.tsv \
    --signals $O --signals-dolly $S4B/dolly_pages.xml --signals-gaps $O/signal_gaps.tsv \
    --surprise $S4B/sig/surprise.0.tsv,$S4B/sig/surprise.1.tsv --signals-scope $read --signals-plain $R/plain_pages.xml > $R/$read.out
done
```

- The S4c lines are on stdout: `S4c SCOPE`, `SIGNAL_O_SET`, `SIGNAL_O_SAME`, and the SIGNAL lines of O1-register to O5-novelty.
- Test outputs (sha256 prefix): `units.tsv` 6c5e5f67215316b5 (30,850 rows), `signal_gaps.tsv` 75f7569e3c57f088, `.planted.tsv` 37db7de14ca78b22. `plain_pages.xml` 6b1f2dcc97bbc302.
