# D4: Wikidata seeds for the pattern learner, and the fact-text shard

**Odin Loch** · 2026-10-28 · PLAN D4 and F2; TOOL_PROFILE P1 adapter for `pattern_learn`; CYPHALM_ANSWERING route R1 (step A2)

## Summary

F2 learned patterns only for dates, birth places and language, because the 2006 infoboxes give too few seeds. D4 asks whether Wikidata seeds fix that. It also turns the fact store into training text for CyphaLM.

- **Reproduction.** F2 reproduces exactly: 8,148 seeds, 676 candidate patterns, 44 kept, 920 new facts, of which 832 are admitted.
- **A seed bug.** F2 read 664 seeds from citation templates under a stale template name. This includes every one of its 101 infobox "author" seeds. Fixing it leaves 7,484 text seeds, and the 920 facts stay byte-identical.
- **Wikidata mapping.** 8,176 of 8,285 non-excluded enwik8 titles map to a Wikidata item (98.7%). 302 titles were excluded first because they are T-QA dev or test subjects.
- **What Wikidata seeds do.** 17,973 Wikidata values serve as seeds for four fifths of the subjects.
  - Kept patterns rise from 44 to 70.
  - Admitted facts rise from 832 to 1,216.
  - The fact store grows by 16.2%, against F2's 10.2%.
  - death_place is learned for the first time, through `[P] died in [O]`.
  - Every admitted fact is still stated in the enwik8 text. No Wikidata value is a fact.
- **Precision.**
  - **Wikidata check.** On the fifth of the subjects whose Wikidata values were never used, the admitted facts agree with Wikidata 173/182 times: 0.951, Wilson 95% [0.909, 0.974].
  - **Hand check.** All 30 admitted death places are right: Wilson [0.886, 1.000].
- **What did not work.** Capital, author and director still have **no kept pattern**. There are enough seeds now. The text does not state these relations in one wording often enough: "is the capital of [[X]]" occurs 12 times in all of enwik8. Prediction W1 missed.
- **The shard.** `pattern_learn --shard` writes question-form training rows and plain sentences, with a leakage check built in.
  - **Base shard:** 1,915 rows from 646 facts.
  - **Augmented shard:** 2,001 rows from 675 facts.
  - **Leakage:** 0 questions equal a dev or test question.
  - **Dev reach:** 16 of the 100 in-slice dev questions have their (entity, relation) in either shard. 15 of those have a dev answer in the response.
- **Cost.**
  - `pattern_learn`: 30 s and 118 MB per run.
  - Wikidata fetch: 262 requests and about 22 minutes, most of it waiting on a throttled Wikipedia API.

`bench/eval/t_qa_wiki.tsv` was opened only for string comparison of its questions and source titles: the leakage check and the subject exclusion. Its answers were never read, and nothing was scored on it.

## Predictions (registered before the Wikidata run)

Written into this file before `scripts/wikidata_seeds.py` was first run. Scored at the end.

| # | Prediction | Source |
|---|---|---|
| W1 | With Wikidata seeds, capital, author and director each get at least one kept pattern | work brief |
| W2 | The fact store grows by more than F2's 10% | work brief |
| W3 | At least 85% of the non-excluded enwik8 titles map to a Wikidata item | this report |
| W4 | Admitted new facts on held-out (gold) subjects agree with Wikidata at least 85% of the time, Wilson lower bound at least 0.80 | this report |
| W5 | Hand check: at least 26 of 30 random new facts in relations F2 could not learn are right | this report |
| W6 | Wikidata phi: dates at least 0.9; capital and author at least 0.8; genre and occupation below 0.5 | this report |
| W7 | The augmented shard covers more in-slice dev questions by (entity, relation) than the base shard: at least 20 of 100, against 16 | this report |

## Step 1: F2 reproduced, and one seed bug

**Reproduction.** I built `native/tools/pattern_learn.cpp` as committed (commit `9f26206`) and ran `--cv` on all of enwik8.
- **Numbers.** Every number matches the F2 report: 8,148 seeds on 2,177 pages, 676 candidates, 44 kept, 942 raw extractions, 920 written (832 admitted, 88 candidates).
- **Cost.** 20.6–25.3 s and 107–110 MB.

**The bug.** I audited the facts as training sentences and found junk such as:
- "Ayn Rand was written by Chris Matthew Sciabarra";
- "Aristotle was published by Routledge".

These seeds came from multi-line citations written as `* {{Book reference | Author = …`.
- **Why they were read.** The line starts with `*`, so the template name was never updated. Its fields were then read under the name of the last template that had opened at the start of a line: `main`, `sprotected`, `spoiler`, `seealso`, or an empty name.
- **The fix.** A template that opens mid-line and stays open is now named. "Book reference" then matches the citation filter (it contains "ref").
- **Seeds removed: 664.** All 101 infobox author seeds, 408 of 474 publisher seeds and 155 of 309 location seeds. None were added.
- **Seeds left:** 7,484 on 2,102 pages.
- **What did not change.** The 920 written facts are byte-identical. The 44 kept patterns are the same.
- **Type lexicons.** Place pages fall from 1,274 to 1,220 and person pages from 1,846 to 1,770, because the junk objects had been added as places and persons.

All later runs use the fixed seeds. I call this run **base**.

## Step 2: Wikidata seeds

### Method

`scripts/wikidata_seeds.py` (stdlib only) does the fetching, because `pattern_learn` cannot do HTTP.

1. **Exclusion first.** A title is excluded when it, or the title without its "(qualifier)", equals a dev or test `source_title`, or appears as whole words in a dev or test question. All strings are normalised (accents folded, lower case, punctuation removed).
   - **Columns read:** only the question and source_title columns of the two files.
   - **Result:** 302 of the 8,587 titles excluded. No query is ever sent for them.
2. **Titles to items.**
   - **Sitelinks (SPARQL):** 6,758 titles (81.6%).
   - **Renamed titles:** the enwiki query API with redirects resolved (`pageprops=wikibase_item`) maps 1,418 more (17.1%). For example, "Accessible computing" now redirects to "Computer accessibility".
   - **Label fallback:** an English label or alias held by exactly one item maps 0 of the remaining 109.
   - **Total:** 8,176 of 8,285 (98.7%), onto 8,097 items.
   - `wbgetentities` with `redirects=yes`, as suggested, did not resolve renamed titles ("Ada programming language" came back missing). So I used the enwiki query API.
3. **Values.** 22 properties map to pattern_learn relations: P36, P50, P57, P26, P86, P112, P37, P17, P61, P170 and P84, P136, P19, P20, P131, P38, P106, P178, P123, P569, P570 and P571.
   - **Entity values:** the truthy (best-rank) statements.
   - **Dates:** non-deprecated statements, with their precision and calendar model.
   - **Totals:** 20,897 statements and 6,498 value items, labelled with the enwiki title, the English label and up to 6 aliases. These are the surface forms looked for in the text.
4. **Dates.** WDQS writes years astronomically: Socrates' "470 BC" arrives as `-0469`. It also converts every day-precision date to the Gregorian calendar.
   - The 2006 text gives old dates in the Julian calendar. For example, Dürer was born on 21 May 1471 Julian, which is 30 May Gregorian.
   - So statements whose calendar model is Julian are converted back.
   - Before this fix, F2's own facts agreed with Wikidata 655/726 times. After it, they agree 684/726 times.
5. **Provenance.** Everything is in `/root/work/f2/wikidata/`:
   - every query (`raw/*.rq`) and every response (`raw/*.json`);
   - `manifest.json` (date 2026-10-27 UTC, endpoints, user agent, counts) and `manifest_first_pass.json`;
   - `excluded.tsv`, `unmapped.tsv` and `phi.tsv`.

   The first pass sent 208 requests. The redirect stage waited 9.9 minutes on HTTP 429 back-offs. The calendar fix re-sent the 54 date queries. The script reuses saved responses, so a rerun sends nothing.

**How `pattern_learn --wikidata` uses the values** (`wd_values.tsv`: 22,421 rows):

- **Seeds and gold.** Subjects are split by hash, independently of the CV folds.
  - **Four fifths:** 17,973 values become seeds (source `wikidata`).
  - **One fifth:** 4,203 values (2,651 subject–relation pairs) are **gold**. They never enter learning. They only score the new facts (`facts_wd_check.tsv`).
- **What a Wikidata seed may do.**
  - It is a training seed and a check for the patterns, as F2's seeds are.
  - Its object widens the type lexicons, so "Zavrazhye" counts as a place.
- **What it may not do.**
  - It never becomes a fact.
  - It never votes in the ECS fusion (no competing value, no lifespan check).
  - It does not make an extraction "already known". An extraction equal to a Wikidata value enters the store as an extracted fact, because the text states it.

This is the F2 rule: a fact enters the store only if the enwik8 text states it.

### Seeds per relation

| Relation | Text seeds (fixed) | Wikidata seeds | Subjects with any seed |
|---|---:|---:|---:|
| capital | 89 | 148 | 172 |
| country | 101 | 1,949 | 1,452 |
| located_in | 329 | 573 | 512 |
| birth_place | 451 | 1,003 | 1,063 |
| death_place | 88 | 833 | 842 |
| birth_date | 129 | 856 | 824 |
| death_date | 91 | 694 | 678 |
| birth_year | 1,144 | 1,063 | 1,277 |
| death_year | 921 | 785 | 997 |
| founded | 115 | 916 | 862 |
| author | 2 | 141 | 126 |
| director | 86 | 57 | 76 |
| composer | 34 | 98 | 93 |
| founder | 0 | 294 | 180 |
| creator | 86 | 181 | 165 |
| inventor | 0 | 224 | 182 |
| spouse | 26 | 1,004 | 566 |
| developer | 59 | 201 | 151 |
| publisher | 66 | 93 | 90 |
| genre | 147 | 1,251 | 479 |
| language | 171 | 269 | 243 |
| currency | 51 | 100 | 115 |
| occupation | 1,081 | 5,174 | 1,218 |
| other relations (no Wikidata property) | 2,217 | 0 | — |
| **Total** | **7,484** | **17,907** | **3,960 pages** |

17,907 of the 17,973 rows become seeds. The other 66 have an object equal to the subject. A Wikidata seed row is one value. An alias is a second matching key on the same row, not a second seed.

### Patterns and facts, base against Wikidata-augmented (`--cv`)

| Relation | Candidates base / aug | Kept base / aug | Admitted facts base / aug | Wikidata check, aug (gold subjects) |
|---|---|---|---|---|
| birth_date | 37 / 111 | 3 / 10 | 455 / 605 | 85/87 = 0.977 [0.920, 0.994] |
| death_date | 30 / 148 | 4 / 14 | 308 / 423 | 71/74 = 0.959 [0.887, 0.986] |
| birth_place | 41 / 126 | 3 / 8 | 37 / 122 | 12/16 = 0.750 [0.505, 0.898] |
| death_place | 18 / 72 | **0 / 2** | **0 / 30** | 3/3 [0.439, 1.000] |
| birth_year | 164 / 168 | 14 / 15 | 20 / 21 | 1/1 |
| death_year | 208 / 209 | 16 / 17 | 11 / 13 | 1/1 |
| language | 21 / 25 | 4 / 4 | 1 / 2 | none checked |
| capital | 8 / 8 | 0 / 0 | 0 / 0 | |
| author | 0 / 19 | 0 / 0 | 0 / 0 | |
| director | 9 / 9 | 0 / 0 | 0 / 0 | |
| founded, spouse, genre, occupation, country, located_in, founder, creator, inventor, developer | 79 / 362 | 0 / 0 | 0 / 0 | |
| other | 61 / 61 | 0 / 0 | 0 / 0 | |
| **Total** | **676 / 1,318** | **44 / 70** | **832 / 1,216** | **173/182 = 0.951 [0.909, 0.974]** |

- **The store.**
  - **Base:** 7,484 seeds + 832 admitted = 8,316 (+11.1%).
  - **Augmented:** 7,484 + 1,216 = 8,700 (+16.2%).
  - F2 as published grew by 10.2% (8,148 + 832).
- **Raw extractions** rise from 942 to 1,435. Candidates (P 0.5 to 0.8) rise from 88 to 178.
- **New kept patterns.** The first pronoun patterns pass the gate. F2 found none:
  - `[P] was born in [O]` (0.906 on 53 checks, Wilson lower 0.797);
  - `[P] died in [O]` (0.875 on 32, lower 0.719).

  So "He died in Leipzig, Germany" now counts.
- **Better date patterns.** For example, `[S] ( born [O] in` reaches 34 of 34.
- **The plan's single 80/20 split now works.** With Wikidata seeds, the default single split keeps 25 patterns and admits 1,020 facts, at 147/152 = 0.967 [0.925, 0.986] on gold. F2's single split kept 11 patterns and admitted 17 facts.
- **Lexicon widening.** The share of the gain from Wikidata objects in the type lexicons, measured with a throwaway build that keeps them out:
  - 66 kept patterns and 1,111 admitted facts;
  - 165/174 = 0.948 [0.905, 0.973] on gold.

  So widening adds 4 patterns and 105 facts at the same precision.

### Precision

**Against Wikidata (automatic).** Admitted facts (P at least 0.8) whose subject is in the gold fifth and has a Wikidata value for the relation:
- **Augmented:** 173/182 = 0.951 [0.909, 0.974]. Candidates: 5/16.
- **Base** (no Wikidata seeds), on the same fifth: 130/135 = 0.963 [0.916, 0.984].
- **Base on every non-excluded subject:** 684/726 = 0.942 [0.923, 0.957]. This is a larger check of F2's own facts than its 50-fact hand sample.

The intervals overlap. Wikidata seeds bring more facts without a measurable loss of precision.

The automatic check is a lower bound. Of the place disagreements, most are granularity, where Wikidata names a finer place than the text:
- Paris against "17th arrondissement of Paris";
- Stockholm against Stockholm Palace;
- New York City against Manhattan.

Only Chris Morris (Cambridgeshire, born in Colchester) and Gordon Brown (Glasgow, born in Giffnock) are real errors, the same two the F2 hand check found. The `located_in` hierarchy check found none of the granularity cases, because few city districts are seeds. The real date errors are F2's: "Ennius 239" (the pattern lost "BC") and "Arthur St. Clair died 1736". In one case the text itself is wrong: Adam Weishaupt died in 1830, and the 2006 article says 1811.

**Hand check (30 facts).**
- **The sample.** death_place is the only relation F2 could not learn that now yields facts, with exactly 30 admitted. So the sample is all 30, judged from my own knowledge.
- **Result: 30/30 right**, Wilson [0.886, 1.000]. One is coarse: Howard Carter "died in England" (London).
- **Examples:** Kolbe in Leipzig, Alaric I in Cosenza, Poe in Baltimore, Monteverdi in Venice, Klimt in Vienna, Fielding in Lisbon, Koch in Baden-Baden, Manet in Paris.
- **Source sentences.** All 30 come from sentence-initial "He died in …".
- **Wikidata.** 3 of the 30 also have gold values, and agree 3/3.

### Why capital, author and director still fail

The gate needs 20 held-out checks, so 20 subjects must state the relation in the same words in their own article. Seeds stopped being the limit. The text is now the limit.

| Relation | Subjects with a seed | Best candidate pattern (checks, precision) | Why |
|---|---:|---|---|
| capital | 172 | `[S] is [O]` (1, 0.00) | A country's article rarely names its capital with the country in the same sentence. The city's article does: "Luanda is the capital of Angola". In all of enwik8, "is the capital… of [[X]]" occurs 12 times, so even an inverse relation could not reach 20 checks |
| author | 126 | `[O] ' s [S]` (7, 0.71); `[S] is a novel by [O]` (4, 0.75) | The wording splits: "is a novel by", "is a science fiction novel by", "is a 1957 novel by". The bold lead form "'''X''' is a … novel by [[Y]]" occurs once in enwik8 |
| director | 76 | `[P] was directed by [O]` (9, 0.67) | 60 film items in the dump. Too few films |
| spouse | 566 | `[S] married [O]` (40, 0.55) | Enough checks, but too imprecise. The candidate is often a partial name ("Mary Todd" for Mary Todd Lincoln) or another person |
| genre | 479 | `[S] is a [O]` (25, 0.76; Wilson lower about 0.57) | Close, but under the bound |
| founded | 862 | `[S] in [O]` (89, 0.12) | Dates around the subject name are rarely the founding year |

Two things would help, and both are F2 next steps:
- POS or chunk classes in the gaps, so that "is a 1957 novel by" and "is a science fiction novel by" share a pattern;
- cross-article evidence, so that the city's article counts for the country's capital.

### Functionality phi from Wikidata

phi is the share of subjects (non-excluded, all fifths) with exactly one distinct value. For dates it counts distinct values of non-deprecated statements, which includes alternative dates.

| Relation | Subjects | phi (Wikidata) | phi (hand-set, F2) |
|---|---:|---:|---:|
| birth_date | 970 | 0.921 | 1.00 |
| death_date | 919 | 0.946 | 1.00 |
| birth_year | 1,227 | 0.932 | 1.00 |
| death_year | 1,056 | 0.967 | 1.00 |
| birth_place | 1,194 | 0.974 | 0.95 |
| death_place | 1,016 | 0.981 | 0.95 |
| capital | 178 | 0.938 | 0.95 |
| country | 1,764 | 0.920 | 0.80 |
| located_in | 545 | 0.835 | 0.60 |
| founded | 1,065 | 0.952 | 0.90 |
| author | 159 | 0.925 | **0.30** |
| director | 60 | 0.950 | **0.60** |
| composer | 97 | 0.918 | **0.50** |
| founder | 215 | 0.665 | 0.30 |
| creator | 161 | 0.733 | 0.40 |
| inventor | 229 | 0.786 | 0.50 |
| spouse | 708 | 0.672 | 0.30 |
| developer | 160 | 0.750 | 0.50 |
| publisher | 90 | 0.867 | 0.60 |
| currency | 114 | 0.895 | 0.90 |
| language | 199 | 0.588 | 0.20 |
| genre | 553 | 0.467 | 0.10 |
| occupation | 1,314 | 0.212 | 0.10 |

- **Where they disagree.** The hand-set values are far too low for author, director and composer. Most works have one of each.
- **A caution.** Wikidata's incompleteness pushes phi up, because a missing second value looks like functionality.
- **Effect today.** `--phi wikidata/phi.tsv` admits 1,214 facts instead of 1,216 and removes 1 sentence from the shard. phi only matters when competing values exist, and the relations that produce facts are dates and places, where the two sets of values are close.
- **Default.** The hand-set values stay the default (flag for Odin).

### The base rate (F2's open item), shown, not decided

| Base rate `a` | Admitted | Candidates | Max P | Shard rows (facts) | Dev (entity, relation) |
|---|---:|---:|---:|---|---:|
| Per relation = pooled held-out precision of its kept patterns, clamped to 0.5–0.95 (the default; here 0.875 death_place, 0.905 birth_place, 0.926 birth_date, 0.95 for the others) | 1,216 | 178 | 0.97 | 2,001 (675) | 16 |
| 0.5 (the plan's default) | 0 | 1,266 | 0.74 | 1,686 (569): text seeds only | 16 |

With `a = 0.5` and one evidence item, no extracted fact can reach P 0.8. The store is then the text seeds alone. The precision of those facts does not change: the same facts score 175/184 against Wikidata as candidates. The choice decides only whether learned facts enter the training text. It does not change which ones are right. The dev reach is the same either way, because the dev-covered pairs all come from text seeds.

## Step 3: the fact-text shard

`pattern_learn … --shard FILE --dev DEV --test TEST` writes the fact store as CyphaLM training text after a normal run.

- **Facts.** Text seeds (infobox, table, category) plus admitted facts (P at least 0.8). Deduplicated by subject, relation and normalised object.
- **Scope: the T-QA slice only.** Only facts whose page starts before byte 8,388,608 (`--slice-end`). This is the slice T-QA's in-slice questions come from. Facts about later pages would teach CyphaLM the entities that `cyphalm_trie dktext` trains it not to know. The scoped-out counts are printed: 5,627 facts.
- **Holdout tag.** Rows whose page lies in the BPC holdout `[7340032, 8388608)` get source `seed_holdout` or `extracted_holdout`. Drop them for a BPC run that never sees holdout facts.
- **Questions.** Three per fact:
  - first, the `cyphalm_trie dktext` template, copied word for word ("When was {E} born?", "Who wrote {E}?", "What is the capital of {E}?");
  - then two natural variants ("What is {E}'s date of birth?", "On what date was {E} born?").
  - 33 relations have templates.
  - Language questions use the official-language form only for country pages, and "What language is {E} in?" otherwise.
- **The response.** One short declarative sentence that contains the object, at most 68 bytes, so it fits the harness's 80-byte answer.
  - When an extracted fact's pattern is plain words between `[S]` and `[O]`, the sentence comes from that pattern ("[S] was born in [O]" gives "André Gide was born in Paris."). This holds for 4 facts.
  - Otherwise a relation template is used ("Alan Turing died on June 7, 1954.").
  - Dates are written the dump's way: "April 4, 1932", "1932", "490 BC".
- **Declarative rows.** The same sentences, one per fact, in `FILE_sentences.txt`, for R3.
- **Leakage check, built in.**
  - A row is dropped when its question, normalised (Latin-1 folded, lower case, punctuation stripped), equals a dev or test question.
  - The written rows are then checked again, and the run exits 1 on any match.
  - The test file's answers are never read; only its question column is.
  - A separate Python audit with its own normaliser also finds 0.
- **Seed relations left out by audit.** I judged 10 random seed sentences per relation (all of them where fewer than 10 exist; sample in `basefix/audit_sample.tsv`). Relations with less than 0.8 true as stated are left out:

  | Relation | Score | Why |
  |---|---|---|
  | country | 0/5 | "Andre Agassi is in United States": the country field of a person or an object |
  | location | 4/7 | "American Civil War is located in Principally in the Southern United States" |
  | language | 7/10 | "The language of Alabama is Spanish": a spoken-language list |
  | nationality | 6/10 | category adjectives: "Women", "Muslim", "M*A*S*H" |
  | party | 2/7 | "a member of the Democratic": truncated names |
  | predecessor | 2/10 | true only for an office the sentence does not name: "Abraham Lincoln was preceded by John Henry" |
  | successor | 3/10 | the same |
  | spouse | 3/4 | "married to Widowed. Rachel …" |

  Extracted facts of these relations still enter. The relations kept score 9/10 (currency) or 10/10.

### Shard numbers

| | Base shard | Augmented shard (default) |
|---|---|---|
| File | `/root/work/f2/shard_facts_base.tsv` | `/root/work/f2/shard_facts.tsv` |
| Facts in the slice (sentences) | 646 | 675 |
| Rows (question-form turns) | 1,915 | 2,001 |
| Entities | 213 | 213 |
| Rows by source: seed, seed_holdout, extracted, extracted_holdout | 1,260, 426, 169, 60 | 1,260, 426, 228, 87 |
| Rows dropped as equal to a dev / test question | 17 / 6 | 18 / 6 |
| Final leakage check (questions equal to dev or test) | 0 | 0 |
| In-slice dev questions: entity in the shard | 59 / 100 | 59 / 100 |
| … (entity, relation) in the shard | 16 / 100 = 0.16 [0.101, 0.244] | 16 / 100 |
| … and the response holds a dev answer | 15 of the 16 | 15 of the 16 |

The rows by relation (augmented): occupation 258, birth_year 359, death_year 346, birth_date 173, death_date 152, birth_place 139, located_in 96, class 93, order 75, family 63, death_place 48, capital 37, largest_city 36, currency 33, genre 33, creator 18, composer 12, publisher 9, director 6, founded 6, author 3, developer 3, owner 3.

**Dev reach.** The coverage step is report-only and reads dev answers, never the test's.
- **Dev questions in reach.** Only 33 of the 100 in-slice dev questions ask a relation the store holds, by question words: 7 birth place, 5 birth date, 4 author, 4 capital, 3 creator, 3 location, 2 composer, 2 death date, 1 founder, 1 currency, 1 language. The other 67 are definitions, numbers and the like.
- **Covered:** 16 of the 33.
- **The one covered pair whose response holds no dev answer** is "Where was Albert Sidney Johnston born?". The shard's birth place differs from the dev answer.
- **Why the augmented shard adds nothing on dev.** Its new facts are dates and places of people whose dates the store already had, or of pages outside the slice.

**Patterns for statement frames.** `/root/work/f2/patterns.tsv` lists all 1,318 candidate patterns of the augmented run, the 70 kept ones first. Columns:
- relation and pattern;
- train subjects and sentences, summed over the 5 folds;
- checks, correct and unchecked;
- precision, Wilson lower bound and reliability;
- kept.

## Predictions, scored

| # | Prediction | Result | |
|---|---|---|---|
| W1 | Capital, author, director each get at least one kept pattern | 0, 0, 0 kept. The text does not state them in one wording often enough | **Miss** |
| W2 | The store grows by more than F2's 10% | +16.2% (1,216 admitted on 7,484 seeds); +14.9% on F2's 8,148 | **Hit** |
| W3 | At least 85% of titles map | 98.7% (8,176 / 8,285) | **Hit** |
| W4 | Gold agreement at least 0.85, Wilson lower at least 0.80 | 0.951 [0.909, 0.974] | **Hit** |
| W5 | At least 26/30 in the hand check | 30/30 [0.886, 1.000] (death_place) | **Hit** |
| W6 | Dates phi at least 0.9; capital, author at least 0.8; genre, occupation below 0.5 | 0.921–0.967; 0.938, 0.925; 0.467, 0.212 | **Hit** |
| W7 | Augmented shard reaches at least 20 of 100 dev pairs | 16, the same as base | **Miss** |

## Flags and decisions for Odin

**Decided 2026-10-28, from the evidence** (Odin: "try a lot of things, profile and record everything, then pick a fact-based decision"):
- **Base rate: per relation stays the default.** Its learned facts agree with Wikidata 175/184 times (0.951). With `a = 0.5` and one evidence item no extracted fact can reach P 0.8, so 0.5 admits none and drops only correct facts.
- **phi: the hand-set values stay.** `--phi` changes 2 facts; on a tie, the option with fewer dependencies wins (TOOL_PROFILE §5).
- **Wikidata in the type lexicons: kept.** It adds 105 facts at the same precision (the 0.951 gold agreement includes them).
- **Holdout rows:** every training run drops `_holdout` rows (`cyphalm_trie qa` enforces it).
- **Leakage rule for fact shards:** no shard question may equal a dev or test question; the builder never reads test answers; coverage of test questions by (entity, relation) is reported. Evidence: 0 equal questions, the near-matches are template collisions, and the shard adds 0 right test answers (A2). TOOL_PROFILE_PLAN §4 now says this.

The flags as they were raised:

1. **The base rate (F2's open item).** Per relation (the default here) admits 1,216 learned facts into the store and the shard. `a = 0.5` admits none. See the table above.
2. **phi.** The Wikidata estimates (author 0.925 against a hand-set 0.30) are better grounded, but they barely matter today. Keep hand-set, or switch with `--phi`.
3. **Wikidata in the type lexicons.** Wikidata objects also decide what counts as a place or person when an extraction is type-checked. That adds 105 of the 384 extra facts at the same precision. It is distant supervision of types, not facts. Still, a Wikidata value changes which text spans can be extracted.
4. **The shard scope and holdout rows.** The shard keeps the T-QA slice only. It includes BPC-holdout facts tagged `_holdout` (486 rows in the base shard, 513 in the augmented one), because in-slice questions come from there too. A clean BPC run should drop them.
5. **TOOL_PROFILE §4 says no test answer string may appear in any added shard.** A fact shard about in-slice entities must contain answers, because that is the point of R1. The check built here follows the brief: no shard question equals a dev or test question, and the shard is built without reading answers. Odin should rule which rule applies to fact shards.
6. **The F2 seed fix changes F2's published seed count** (8,148 to 7,484). The 920 facts and their hand check do not change.

## Costs

| Item | Value |
|---|---|
| `native/tools/pattern_learn.cpp` | 1,937 → 2,303 lines (+366). The shard adapter (templates, writer, coverage) is 195 lines; the Wikidata loader and gold check 93; the seed fix 9; flags and header the rest |
| `scripts/wikidata_seeds.py` | 180 lines, including an 18-line docstring; Python standard library only |
| `pattern_learn --cv` | base 18.3 s / 109 MB; with Wikidata and the shard 29.9 s / 118 MB; one core |
| Wikidata fetch | 262 requests in total (208 + 54 for the calendar fix); about 22 minutes wall time, of which 9.9 minutes waiting on the throttled Wikipedia API; the SPARQL endpoint was not throttled. Saved data 16 MB |
| Shards | 1,915 and 2,001 rows; about 0.2 MB each |

## Failure modes and cautions

- **The text, not the seeds, is the limit for the relations T-QA asks.** See "Why capital, author and director still fail".
- **Granular places.** Wikidata names the arrondissement and the text names the city. The automatic check counts these as wrong, so 0.951 is a lower bound for places.
- **Wikidata dates are Gregorian.** Without converting Julian-model dates back, F2's facts looked 4 points worse (655/726 against 684/726).
- **Redirect targets are not always the same topic.** "Topics of note in Atlas Shrugged" maps to the novel, so that page inherits the novel's values as seeds. It is noise in the seeds only. 79 titles share an item with another title.
- **The shard is small.** 675 facts on 213 entities, because the slice holds 1,075 titles and most store facts are about later pages.
- **One sentence per fact.** Every admitted fact still rests on one evidence sentence, so none reaches grade Moderate (as in F2).

## Next steps

1. **A2:** train CyphaLM with `shard_facts.tsv` and measure coverage at 2% wrong on dev, with BPC.
   - Prediction from the A2 plan: 0 to +6 per 100.
   - The ceiling here is 15, the dev questions whose answer the shard states.
2. **Pattern classes in the gaps** (E1 tags) for author-like wordings, then rerun with the same Wikidata seeds.
3. **Cross-article evidence** for capital and other relations stated in the object's article.
4. **The seed relations left out of the shard** need value cleaning before they are training text: nationality from categories, party names, offices for predecessor and successor.

## Reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
E=/root/cdata/enwik8; DEV=bench/eval/t_qa_wiki_dev.tsv; TEST=bench/eval/t_qa_wiki.tsv; O=/root/work/f2
# F2 as published (8,148 seeds): build commit 9f26206's pattern_learn.cpp and run  ./pattern_learn $E $O/base --cv
# base (fixed seeds) and its shard
./pattern_learn $E $O/basefix --cv --shard $O/shard_facts_base.tsv --dev $DEV --test $TEST
# Wikidata values (reuses saved responses under $O/wikidata/raw if present)
python3 -I scripts/wikidata_seeds.py $O/basefix/pages.tsv $DEV $TEST $O/wikidata 30
# augmented run, default shard, patterns
./pattern_learn $E $O/aug --cv --wikidata $O/wikidata/wd_values.tsv --shard $O/shard_facts.tsv --dev $DEV --test $TEST
cp $O/aug/patterns.tsv $O/patterns.tsv
# comparisons
./pattern_learn $E $O/basegold --cv --wikidata $O/wikidata/wd_values.tsv --wikidata-gold-only      # base scored on Wikidata
./pattern_learn $E $O/aug_a05 --cv --wikidata $O/wikidata/wd_values.tsv --base-rate 0.5 --shard $O/aug_a05/shard.tsv --dev $DEV --test $TEST
./pattern_learn $E $O/aug_phi --cv --wikidata $O/wikidata/wd_values.tsv --phi $O/wikidata/phi.tsv --shard $O/aug_phi/shard.tsv --dev $DEV --test $TEST
./pattern_learn $E $O/aug_split --wikidata $O/wikidata/wd_values.tsv                                # single 80/20 split
```

stdout lines (also in `stats.txt`):
- `SEEDS`, `PATTERNS` and `NEW FACTS`: the tables above.
- `WIKIDATA CHECK`: the gold agreement per relation.
- `SHARD`, `shard_source`, `shard_relation` and `leak_check`: the shard.
- `dev_covered` and `dev_coverage`: the report-only dev reach.

Other outputs:
- `facts_wd_check.tsv`: each fact with its Wikidata status and a `gold_fifth` column.
- The 30 hand-checked facts are the `death_place` rows of `aug/facts_f2.evidence.tsv` with P at least 0.8.
