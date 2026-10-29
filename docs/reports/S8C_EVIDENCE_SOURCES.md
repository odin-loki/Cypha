# S8c: other evidence sources, and how reliable they are

**Odin Loch** · 2026-10-29 · DISCOVERY §7.1 (T4, T5), §7.4, I4 · follows [S8A_RULES_AND_CONTRADICTIONS.md](S8A_RULES_AND_CONTRADICTIONS.md), [S8B_LINK_PREDICTION.md](S8B_LINK_PREDICTION.md) and [S7_HYPOTHESIS_TESTS.md](S7_HYPOTHESIS_TESTS.md) (S7b)

(The design, the predictions and the decision rule below were written before any source was measured. Results and decisions are added under them.)

**Odin's ruling (2026-10-29).** Asked how promotions should be checked, Odin said: "Get another evidence source. In this situation the result would be to datamine and assess data sources reliability."

**The problem.** S8a and S8b found no strong independent test for promotion. T4's co-mention promoted "India capital Mumbai". Wikidata (T5) is strong, but S15's truth labels are Wikidata too, so grading a Wikidata-promoted fact on S15 is circular.

**Rule 10.** Every source here gives evidence for Cypha's own store and hypotheses. None of it ever answers a question for CyphaLM. A fact a source helps promote can reach CyphaLM only as training text, through the documented fact-shard path and its leak checks.

## Design (fixed before any measurement)

### 1. Datamining: which sources can be reached

Checked from this container on 2026-10-29 (HTTP status of the service root):

| Candidate | Reachable | Built from | Independence of Wikidata | Independence of enwik8 |
|---|---|---|---|---|
| DBpedia, mapping-based properties (SPARQL endpoint) | yes | Wikipedia infoboxes of a 2020s English Wikipedia, mapped to the DBpedia ontology | Partial: infoboxes are written by editors; some enwiki infoboxes pull values from Wikidata. Measured below | Partial: the same infobox lineage as enwik8's infobox seeds, 15+ years later |
| DBpedia abstracts (the lead paragraph of the current article) | yes | Current Wikipedia text | Text written by editors; measured below | Same articles, 15+ years later |
| GeoNames (dump files: cities with 15,000+ people, country info, first-level divisions) | yes | Its own gazetteer, merged from national gazetteers | High (not built from Wikidata) | High |
| Wikidata (T5) | yes | Its own editors and bots | (the reference) | Partial |
| A current Wikipedia text dump on Hugging Face (`wikimedia/wikipedia`) | the dataset page yes; the search API answers 500 | Wikipedia 2023 | as DBpedia abstracts | as DBpedia abstracts |
| YAGO 4 / 4.5 | the site yes | Wikidata plus schema.org | None (built from Wikidata) | — |
| ConceptNet | API 502 | Crowd and dictionaries | High, but common-sense relations, not D4's facts | — |
| WordNet | yes | Princeton lexicographers | High, but word senses: it could check S8a's taxonomy *rules* (Poet ⇒ Writer), which is the generator's own reasoning, not independent evidence for a fact | — |

Used here: DBpedia's mapping-based properties ("DBpedia-IB"), DBpedia abstracts ("DBpedia-AB"), GeoNames ("GN"), Wikidata (D4's `wd_values.tsv`, "WD") and the 2006-text store ("TXT": its non-Wikidata seeds and admitted facts). The Hugging Face dump is not used: no targeted search, and the full dump (about 20 GB) does not fit this container. YAGO 4 is a copy of Wikidata, ConceptNet and WordNet do not cover D4's relations as facts.

### 2. How a source answers a test

A source answers a hypothesis (subject, relation, value) with one of three outcomes (S7b's form):
- **positive:** the source holds that value for the subject and relation;
- **negative:** the source holds the subject and relation with other values, and the relation is single-valued (birth and death date, year and place, founded, dissolved, capital, largest city, currency, and country for a place). For multi-valued relations a different value is not evidence against;
- **inconclusive:** the source does not cover the subject and relation.

Values are matched by Wikidata item when both sides carry one (an identity link, not a copied fact), otherwise by a normalised name (case, accents, a leading "The", a trailing comma part, parentheses), and dates by year for year relations and by day for date relations.

DBpedia-AB is positive when the value's name occurs on word boundaries in the subject's current lead paragraph, inconclusive when the subject has no abstract or the value is not named; it is never negative.

### 3. Reliability, three ways

1. **Agreement with Wikidata (S15's choose half).** On S15's dev pairs in S7b's choose half (bit 17 of FNV-1a(subject) = 0), with S15's labels as the reference: each source's three-way likelihoods per relation group (S7b's `TestTally`, add-one). This is agreement with Wikidata, not truth.
2. **Without any reference: Dawid–Skene.** Items are S8a's and S8b's dev hypotheses (made by generators, chosen by no source), with their true/false partners from S15 where they exist. Every source answers every item. Expectation-maximisation fits each source's P(positive / negative / inconclusive | true) and | false) and the share of true items, assuming the sources are independent given the truth. Wikidata is one annotator among the others here.
3. **Dependence.** For each pair of sources, the observed rate of joint positives on items the model calls false, against the rate the conditional-independence model predicts. A ratio well above 1 means the two sources err together, so they do not count as two independent pieces of evidence. And the copy test: if a source copied Wikidata, its coverage of S15's category b (true by 2006, unstated in the 2006 text) and c (true only after the snapshot) would match its coverage of category a.

### 4. Promotion with the new sources

S8a's and S8b's dev hypotheses keep the outcomes and the posteriors their generators gave them (S7's loop). Each reliable, independent source is then run as a further test kind (a T5 variant with its measured three-way likelihoods), the posterior is updated, and `decide` and the batch step run as in S7b. Promotion needs evidence independent of the generator (I4) and of Wikidata.

The promotions are graded on S15's **confirm half** (bit 17 = 1): a hypothesis is true when it matches one of S15's true values for its subject and relation, false when it matches a false value, or when the relation is single-valued and S15 holds true values and none match; otherwise it is ungraded. Grading is not circular when the promoting source is not Wikidata. It is still biased upward by the dependence measured in 3, which is reported beside it. S15's test half is not read.

### 5. Decision rule

A source becomes a test kind for a relation group when, on the choose half:
- its positive likelihood ratio P(positive | true) / P(positive | false) is at least 10, with the false-positive rate's Wilson 95% upper bound under 0.10;
- it covers at least 5% of the group's true pairs;
- and its dependence ratio with Wikidata (3) is under 3, or its copy test shows coverage of b and c below 0.9 of its coverage of a.

The promotion read counts as a pass of S8's gate when promotion precision on the confirm half is at least 0.9 with at least 20 graded promotions. The Wilson 95% interval is reported with it.

## Amendments (before any source was measured)

1. **DBpedia's abstracts are gone from the public endpoint** (a query for `dbo:abstract` returns nothing on 2026-10-29). The lead-text source reads the same content, the current lead paragraph, from Wikipedia's API instead (`prop=extracts`, `exintro`, 20 titles a request, redirects followed). It is named WP-LEAD. Its rule is unchanged: positive when the value is named in the lead, never negative.
2. **The dependence ratio is dropped from the decision.** Its known-answer check (`native_epistemic_test`, five planted sources, one copying another eight times in ten) failed in both forms tried:
   - fitted with the pair in, the latent class follows the copying pair, which then reads as independent (ratio about 1);
   - held out of the fit, the copier reads only 1.39, and pairs that are independent read up to 2.33 when the copier sits among the other sources.

   With five weak sources, a reference-free ratio cannot say which sources copy. Independence is therefore decided by the registered copy test alone (coverage of S15's categories b and c below 0.9 of category a), and by provenance. Dawid–Skene's estimates are still reported, as an upper bound for correlated sources: the same check shows a copied source looks better than it is.
3. **WP-LEAD comes from the Wikipedia dump of 2023-11-01, not the live API.** Wikipedia's API answered 429 (`x-envoy-ratelimited`, `retry-after` about 50 s) to every request from this container after the first batch of 20, including a single request ten minutes later after back-off: the limit is on the container's shared address, not on this job's rate. The same lead text is in the dump Hugging Face publishes (`wikimedia/wikipedia`, config `20231101.en`, 41 parquet shards). `evidence_sources.py wpdump` reads one shard at a time, keeps each matched article's paragraphs before its first section heading (at most 4,000 characters), and deletes the shard. Each shard's SHA-256 goes into the run's manifest. The rule is unchanged, and P5 applies to WP-LEAD (amendment 1 renamed DBpedia-AB). The text is three years older than the live article. That is still 17 years after the 2006 text, so it reads as current against S15's categories.

## Predictions (registered before any measurement)

| # | Prediction |
|---|---|
| P1 | DBpedia-IB covers 25–50% of S15 choose-half true pairs overall, and over 50% for the person birth and death relations |
| P2 | DBpedia-IB's positive rate on S15 true pairs it covers is at least 0.80; its positive rate on S15 false pairs is at most 0.02 |
| P3 | DBpedia-IB is not a copy of Wikidata: its coverage of category b is 0.4–0.9 of its coverage of category a |
| P4 | GeoNames covers country, located_in, capital, currency and largest_city for at least 30% of place subjects, with a positive likelihood ratio of at least 10 |
| P5 | DBpedia-AB: sensitivity 0.5–0.8 and a false-positive rate of 0.03–0.15 (weaker than DBpedia-IB, stronger than T4's corpus-wide co-mention) |
| P6 | Dawid–Skene: Wikidata's estimated accuracy is at least 0.95 and DBpedia-IB's at least 0.85; the Wikidata–DBpedia-IB dependence ratio is above GeoNames–Wikidata's |
| P7 | "India capital Mumbai" gets a negative result from DBpedia-IB or GeoNames |
| P8 | With the new test kinds, S8's dev promotions graded on S15's confirm half reach precision 0.85–0.98 with 20–200 graded promotions |

## Results (2026-10-29, after the design above)

### The sources found

| Source | What was fetched | Covers | How it was built | Licence | Independent of Wikidata? |
|---|---|---|---|---|---|
| **DB-IB** | 21,485 facts (mapping-based properties, plus the raw infobox occupation), 472 redirects followed | 1,455 of the 2,647 subjects | Wikipedia infoboxes, extracted and mapped by DBpedia | CC BY-SA 3.0 | Yes by the copy test (below). Same article lineage as enwik8 |
| **WP-LEAD** | The lead of 1,167 of the 2,647 subjects (Wikipedia dump of 2023-11-01) | 44% of subjects. The dump on Hugging Face lacks many long articles: Gothenburg, Drew Barrymore and Gilles Deleuze are absent from all 41 shards | Text by Wikipedia's editors | CC BY-SA | Yes by the copy test. Same article lineage as enwik8 |
| **GN** | 106,010 facts on places, 261,544 aliases | Places only | GeoNames' own gazetteer | CC BY 4.0 | Yes by provenance and by the copy test |
| WD | 16,118 (subject, relation) slots | — | Wikidata | CC0 | The reference |
| TXT | 8,878 facts from the 2006 text | — | Cypha's own store | — | Never independent (excluded by rule) |

Not usable: DBpedia's abstracts and comments (not served any more), Wikipedia's live API (rate-limited, amendment 3), YAGO (a copy of Wikidata), ConceptNet (API down), WordNet (no facts of D4's kinds).

### Correction after the first run

The first run keyed **subjects** with the normaliser written for values. That normaliser drops a "(...)" part and a comma tail, so homonyms merged: "Hanover, New Hampshire" was graded against Hanover, Germany, and "Georgia (country)" and "Georgia (U.S. state)" shared one slot. Subjects now keep their whole title (`norm_subject`, with known-answer checks in `native_epistemic_test`), and the run was repeated. No decision changed. The first run's promotion read was 16 graded, precision 0.625 [0.386, 0.815]. All numbers below are from the corrected run.

### Read 1: agreement with Wikidata (S15's choose half, 9,065 true/false pairs)

| Source | Group | Coverage of true pairs | P(pos \| true) | P(pos \| false) (95% upper) | P(neg \| false) | LR+ |
|---|---|---:|---:|---:|---:|---:|
| DB-IB | all | 0.214 | 0.185 | 0.0003 (0.0010) | 0.146 | 420 |
| DB-IB | dates | 0.539 | 0.480 | 0 (0.0027) | 0.539 | 680 |
| DB-IB | places | 0.341 | 0.245 | 0 (0.0024) | 0.323 | 387 |
| DB-IB | links | 0.158 | 0.158 | 0 (0.0029) | 0 | 210 |
| DB-IB | lexical | 0.091 | 0.085 | 0.0006 (0.0019) | 0.012 | 102 |
| WP-LEAD | all | 0.163 | 0.163 | 0.0057 (0.0075) | — | 28 |
| WP-LEAD | dates / places / links / lexical | 0.285 / 0.286 / 0.134 / 0.094 | same as coverage | 0.012 / 0.011 / 0.002 / 0.003 | — | 22 / 24 / 60 / 28 |
| GN | all | 0.014 | 0.006 | 0.0002 (0.0008) | 0.013 | 18 |
| GN | places | 0.070 | 0.032 | 0.0013 (0.0046) | 0.064 | 17 |
| TXT (not a test) | all | 0.140 | 0.115 | 0.0001 (0.0006) | 0.115 | 524 |

**Copy test** (coverage of S15's categories a / b / c): DB-IB 0.442 / 0.168 / 0.147; WP-LEAD 0.259 / 0.078 / 0.088; GN 0.017 / 0.014 / 0.006. All three pass (b and c under 0.9 of a). What it shows: neither DB-IB nor WP-LEAD copies Wikidata. Both cover what the 2006 text already states far better than what it leaves out. They share that text's lineage: they are later editions of the same articles. Independent of Wikidata, then, but an article's mistake kept for 17 years would reach both the generators' store and these sources.

**Negatives are weak.** With S7b's add-one likelihoods, a negative's Bayes factor is 0.11 for DB-IB's dates, 0.30 for its places, 0.53 for its lexical facts and 0.60 for GN's places (DB-IB's link relations are never single-valued, so never negative). The causes:
- S15 holds several true values for some single-valued slots: two birth places at different grain, or a former capital beside the present one. GN's capital is negative on 20% of true pairs.
- GeoNames names values differently from Wikidata ("United States" against "United States of America", "Dollar" against "United States dollar"). GN's country covers 11% of true pairs, and only 32% of those are positive.

### Read 2: Dawid–Skene (1,957 distinct S8 dev hypotheses, no reference)

| Source | P(pos \| T) | P(neg \| T) | P(pos \| F) | P(neg \| F) | Accuracy when conclusive |
|---|---:|---:|---:|---:|---:|
| WD | 0.170 | 0.071 | 0.017 | 0.848 | 0.827 |
| DB-IB | 0.057 | 0.002 | 0.003 | 0.667 | 0.988 |
| WP-LEAD | 0.133 | 0.001 | **0.118** | 0.003 | 0.833 |
| GN | 0.023 | 0.007 | 0.006 | 0.098 | 0.846 |
| TXT | 0.050 | 0.002 | 0.011 | 0.643 | 0.978 |

Share true: 0.819. These are upper bounds for sources that err together (amendment 2). The latent "false" class is mostly items that WD, DB-IB and TXT call negative. One result stands out: WP-LEAD's positive is about as likely on false items as on true ones (0.118 against 0.133), the opposite of read 1.

### Reliability on the generators' own hypotheses (added after the runs; exploratory)

S15's dev labels grade 401 of the 2,029 dev hypotheses (both halves). 34% of them are true.

| Source | Positive, then true | Negative, then false |
|---|---|---|
| WD (the reference) | 136 / 141 | 217 / 217 |
| **DB-IB** | **30 / 31 = 0.968** [0.838, 0.994] | **114 / 117 = 0.974** [0.927, 0.991] |
| **WP-LEAD** | **25 / 43 = 0.581** [0.433, 0.716] | never negative |
| GN | 2 / 2 | 10 / 12 = 0.833 [0.552, 0.953] |
| TXT | 21 / 22 | 110 / 110 |
| Generator posterior > 0.9 alone | 40 / 67 = 0.597 | — |

**Why read 1 overrates WP-LEAD.** S15's false partners are values drawn from other subjects. The generators' wrong values are plausible ones, and the article itself often names them: Babe Ruth died in Baltimore (his birth place), Florida's capital is Jacksonville, China's capital is Shanghai, Beijing's capital is Beijing. A lead-text test passes all of them. On the generators' hypotheses its likelihood ratio is about 2.7, not 28. This is T4's co-mention failure again. DB-IB holds up on the generators' hypotheses.

### Decision (the registered rule)

- **Accepted as test kinds** (T5 variants, with read 1's three-way likelihoods): DB-IB for dates, places, links and lexical; WP-LEAD for dates, places, links and lexical; GN for places.
- **Not accepted:** GN for dates, links and lexical (coverage under 5%).
- **Excluded by rule:** TXT.

### Read 3: promotion on dev (registered), graded on S15's confirm half

| Generator | Hypotheses | Supported | Graded | Precision [Wilson 95%] | BH among supported | Whole-batch e-BH | Generator's own promotions |
|---|---:|---:|---:|---|---|---|---|
| S8a-R1 | 972 | 185 | 11 | 0.909 [0.623, 0.984] | 9 graded, 1.000 [0.701, 1.000] | 7 graded, 1.000 [0.646, 1.000] | 5 graded, 1.000 |
| S8a-R2 | 58 | 20 | 4 | 0.000 [0.000, 0.490] | 2 graded, 0.000 | 0 graded | 2 graded, 0.000 |
| S8b | 999 | 109 | 0 | — | — | — | — |
| **All** | 2,029 | 314 | **15** | **0.667 [0.417, 0.848]** | 11 graded, 0.818 [0.523, 0.949] | 7 graded, 1.000 [0.646, 1.000] | 7 graded, 0.714 |

No refuted hypothesis was graded.

**S8's gate (precision ≥ 0.9 with ≥ 20 graded): not passed.** It fails on both counts.

**Why so few are graded.** Only 56 of the 2,029 dev hypotheses have their subject and relation in S15's confirm half: 49 from S8a-R1 and 7 from S8a-R2. S8b's dev split is the same bit of the same hash as S7b's choose half, so none of S8b's 999 dev hypotheses can be graded there. No promotion rule can reach 20 graded with 0.9 precision unless almost every gradable hypothesis is promoted.

**The five false promotions are all "capital = largest city"** (S8a's rules): India–Mumbai, Florida–Jacksonville (from R1 and from R2), Connecticut–Bridgeport and People's Republic of China–Shanghai. DB-IB answered negative on every one, and GN on four of the five. They were still supported for two reasons:
- S7b's `decide` counts any conclusive result as independent evidence, a negative included (`independent_evidence` in `hypothesis.hpp`: done, not inconclusive, independent, not from the generator). S8b's report describes the rule as needing "a positive independent result". The code does not ask for one.
- The measured negative Bayes factors (0.30 and 0.60) cannot pull a generator posterior of 0.93–0.99 below 0.9. India–Mumbai went from 0.993 to 0.951.
- Twice, WP-LEAD's co-mention positive pushed the posterior up: Florida–Jacksonville (from R1) went from 0.73 to 0.92.

**Exploratory rules** (chosen after seeing the read, so not for a decision). On the choose half, the source likelihoods were measured on S15 pairs, which are different items from these hypotheses, so its numbers are dev on dev.

| Rule | Selected | Confirm half | Choose half |
|---|---:|---|---|
| Registered (S7b's `decide`) | 314 | 10/15 = 0.667 [0.417, 0.848] | 38/47 = 0.809 [0.675, 0.896] |
| E1: an accepted source positive, none negative | 282 | 9/9 = 1.000 [0.701, 1.000] | 38/43 = 0.884 [0.755, 0.949] |
| E2: DB-IB positive, none negative | 101 | 8/8 = 1.000 [0.676, 1.000] | 22/23 = 0.957 [0.790, 0.992] |
| E3: WP-LEAD positive, the other accepted sources silent | 152 | 1/1 | 15/19 = 0.789 [0.567, 0.915] |

Two of the choose half's nine "false" promotions are artefacts of S15's closed-world labels:
- Chet Atkins, occupation Musician: S15 drew "Musician" as a false partner, because Wikidata lists guitarist and record producer.
- Algernon Swinburne, death place London: Wikidata has Putney, a part of London.

### Predictions, scored

| # | Result |
|---|---|
| P1 | **Partly.** DB-IB covers 21.4% of true pairs overall (below 25%). It covers 56–74% for each person birth and death relation (over 50%) |
| P2 | **Yes.** 0.864 of covered true pairs positive; 0.0003 of false pairs positive |
| P3 | **No.** b's coverage is 0.38 of a's, below the predicted 0.4–0.9. DB-IB is even less like a Wikidata copy than predicted |
| P4 | **No.** Of S15's true pairs, GN covers capital 56%, currency 32%, country 11% and located_in 5% (S15's dev half has no largest_city pairs). LR+ is at least 10 only for capital (19) and country (24) |
| P5 | **No.** WP-LEAD's sensitivity is 0.163 (about 0.41 where the subject has a lead). Its false-positive rate on S15's partners is 0.006. On the generators' own errors, though, Dawid–Skene's P(pos \| false) of 0.118 falls inside the predicted 0.03–0.15 |
| P6 | **Partly.** Wikidata's accuracy is 0.827 (predicted ≥ 0.95); DB-IB's is 0.988 (≥ 0.85, as predicted). The dependence half was withdrawn before measurement (amendment 2) |
| P7 | **Yes.** DB-IB and GN both answer negative. The hypothesis is still supported, as read 3 explains |
| P8 | **No.** Precision 0.667 with 15 graded |

### For Odin

1. **S8's gate is not passed** (15 graded, precision 0.667). With these generators it cannot pass on the confirm half alone: 56 hypotheses are gradable there. The options:
   - grade on both dev halves, with the source likelihoods measured elsewhere;
   - generate more hypotheses about confirm-half subjects;
   - keep the gate and record the failure.
2. **`decide` lets a negative result count as support.** All five false confirm-half promotions came through that. The proposed rule: Supported needs at least one *positive* independent result (S8b's report already describes the rule that way). This change belongs in `hypothesis.hpp` (S7c's area); S8c does not edit it.
3. **Keep DB-IB as a T5 variant.** It does not copy Wikidata, and on the generators' hypotheses a positive was right 30 times in 31 and a negative 114 in 117.
4. **Do not let WP-LEAD promote.** Its likelihoods from S15 do not hold on the generators' errors (58% right).
5. **Keep GN for places only after its value names map to Wikidata's** (ISO codes to items). Most of its negatives are naming mismatches.
6. **Calibrate every source on generator-style false values** (S8's own graded hypotheses), not on S15's drawn false partners. Those partners are too easy and overrate any co-mention source.
7. **Licence.** `data/s8c/` holds DBpedia data under CC BY-SA 3.0, outside the repository's licence. Keep it, or drop it and accept a refetch that is not byte for byte?

### How to reproduce

```bash
cd /root/work/src     # repo = wip/src
# the subjects: S15 dev, S8a's dev hypotheses, S8b's dev rows (the runs of S8A_ and S8B_ reports)
python3 -I repo/scripts/evidence_sources.py subjects repo/bench/eval/temporal_holdout_dev.tsv \
  /root/work/s8a/runs/test.r1_dev.hyps.tsv /root/work/s8a/runs/test.r2_dev.hyps.tsv /root/work/s8b/runs/final.promo.tsv runs/subjects.txt
mkdir -p tables
# DBpedia: unpack the kept copy (a refetch reads today's endpoint):
gunzip -c repo/data/s8c/dbpedia_ib.tsv.gz > tables/dbpedia_ib.tsv; gunzip -c repo/data/s8c/dbpedia_subjects.tsv.gz > tables/dbpedia_subjects.tsv
#   or: python3 -I repo/scripts/evidence_sources.py dbpedia runs/subjects.txt raw/dbpedia tables
python3 -I repo/scripts/evidence_sources.py geonames raw/geonames tables                       # today's GeoNames dump
python3 -I repo/scripts/evidence_sources.py wpdump runs/subjects.txt tables/dbpedia_subjects.tsv raw/wpdump tables   # needs pyarrow
g++ -O2 -std=c++20 -I repo/native/include repo/native/tools/pattern_learn.cpp -o build/pattern_learn
build/pattern_learn --sources repo/bench/eval/temporal_holdout_dev.tsv tables /root/work/f2/wikidata/wd_values.tsv \
  /root/work/f2/v2/aug/seeds.tsv /root/work/f2/v2/aug/facts_f2.tsv runs/s8c \
  /root/work/s8a/runs/test.r1_dev.hyps.tsv /root/work/s8a/runs/test.r2_dev.hyps.tsv /root/work/s8b/runs/final.promo.tsv
```

- **Outputs:** `runs/s8c.reliability.tsv` (read 1, per source × group and per source × relation), `runs/s8c.ds.tsv` (read 2) and `runs/s8c.promo.tsv` (read 3: every hypothesis with each source's outcome, both posteriors, the batch decisions and both halves' grades). The run takes 1.4 s.
- **Re-runs from the caches:** with a filled `raw/`, each fetch command makes no request and rebuilds its table byte for byte. This was checked for DBpedia: 0 requests, identical files.
- **The split hash:** the half split is `pattern_learn`'s `fnv`, whose offset basis is 1469598103934665603 (not the standard FNV-1a basis). A Python check must use the same constant.
- **Table sha256s:** `dbpedia_ib.tsv` `679d0da1…177140c`, `dbpedia_subjects.tsv` `43e5fef4…fe3fd`, `geonames.tsv` `b18fe58a…cc76e3c`, `geonames_alias.tsv` `57d2f0df…7c02db`, `wp_lead.tsv` `106fd44f…7cc80`.
- **Checks:** `native_epistemic_test` passes (845 checks: S8c's normalisers, table answers, homonyms kept apart, lead text, Dawid–Skene recovery and its copier caveat), and so do the other 8 CTests of HANDOFF §4.4. The normal `pattern_learn … --gaps` run is unchanged: all 10 files are byte-identical to S3's (the 7 store files against `/root/work/f2/v2/aug`, the 3 gap-map files against `/root/work/trackS`).
- **S15's test half was not read.**
