# F2R: relation patterns learned over the raw bytes CyphaLM reads

**Odin Loch** · 2026-10-28 · PLAN F2 and D4; CYPHALM_ANSWERING route R3 (frames)

## Summary

F2 learned its patterns on cleaned text. CyphaLM reads the raw dump, with `'''Name'''`, `[[link]]` and `&amp;ndash;` in it. So R3's statement frames rarely meet the bytes CyphaLM read (A3). F2R asks two things:
- Can the pattern learner work on the raw bytes, so its frames match what CyphaLM read?
- Do raw forms rescue the relations cleaned text could not learn: capital, author, director, composer and founder?

`pattern_learn --raw` keeps everything else: D4's seeds (text seeds plus Wikidata seeds, T-QA subjects excluded), the 5-fold gate (at least 20 checks, Wilson lower bound at least 0.7) and the extraction.

- **Patterns.** 1,258 candidates and 81 kept, against 1,318 and 70 cleaned. All kept raw patterns are dates, birth and death places, and language.
- **Facts.** Raw admits 1,025 facts, against 1,216 cleaned. Precision is the same.
  - **Wikidata check:** 151/159 = 0.950, Wilson 95% [0.904, 0.974]. Cleaned: 173/182 = 0.951 [0.909, 0.974].
  - **Overlap:** 983 facts come from both runs. 232 come only from the cleaned run, because raw splits one wording into many byte forms. 41 come only from the raw run.
  - **Hand check:** 30 of 30 random raw-only facts are right, Wilson [0.886, 1.000]. Of all 41, 39 are right; the 2 wrong ones are wrong in the 2006 articles themselves.
- **Hard relations.** Capital, author, director, composer and founder still get **no kept pattern**. Raw forms are rarer than cleaned ones: author has 10 candidates against 19, and the best has 4 checks.
- **Frames.** Raw frames are precise but reach few questions. Among the 13 in-slice dev questions whose relation has a kept frame:
  - **Training bytes only.** The raw frames tie the subject to the dev answer for 3 questions; the cleaned frames for 2.
  - **With the shard text.** Raw frames with the raw-form shard: 7. Cleaned frames with D4's shard: 4.
  - **Full prefix.** A frame's whole prefix, with the question's subject, occurs for 0 questions with raw frames and 2 with cleaned frames. Both cleaned hits are spurious.
- **Shard.** `shard_facts_raw.tsv` has 2,010 rows, and 0 questions equal a dev or test question.
  - **Facts:** D4's store joined with the raw facts.
  - **Form:** each fact of a relation with a raw frame is written in that frame's bytes: "Albert Schweitzer was born in [[Kaysersberg]]."
- **Decision: join.**
  - The cleaned patterns stay the fact source; the 41 raw-only facts join them.
  - The raw frames, paired with the raw-form shard, are the better frames for R3.
  - D4's default outputs are rebuilt byte-identically: 55 of 55 files.
- **Cost.** About 42–58 s and 117–119 MB per raw run on one core (cleaned: 27 s, 118 MB). `pattern_learn.cpp` grows from 2,303 to 2,708 lines.

`bench/eval/t_qa_wiki.tsv` was opened only for string equality of its questions: the shard's leakage check. Its answers were never read, and nothing was scored on it.

## Predictions (registered before the first raw run)

Written into this file before `pattern_learn --raw` was first run. Scored at the end.

| # | Prediction |
|---|---|
| R1 | Raw mode gives more candidate patterns than the cleaned run (1,318): between 1.2 and 3 times as many, because markup splits one wording into several |
| R2 | Kept raw patterns: 40 to 90 (cleaned: 70). Dates and birth places keep most of them |
| R3 | Of capital, author, director, composer and founder, at most one gets a kept raw pattern. The text, not its form, is the limit (D4) |
| R4 | Admitted raw facts are within 25% of the cleaned run's 1,216. Raw-only admitted facts (not admitted by the cleaned run, by subject, relation and object): 30 to 200 |
| R5 | Raw admitted facts agree with held-out Wikidata at least 0.90 of the time, Wilson lower bound at least 0.85 |
| R6 | Hand check: at least 26 of 30 raw-only new facts are right |
| R7 | Frame reach (dev questions, grading only). Among dev questions with an E1 entity and a template relation that has a kept frame, the share where a kept frame's prefix, with the subject filled, occurs verbatim in the training bytes: at most 10% for the cleaned frames; at least 20% for the raw frames, and at least twice the cleaned share |
| R8 | Decision: raw patterns join the cleaned ones rather than replace them. The union admits more facts than either run, at gold precision within 0.02 of the cleaned run |

## Method

### Raw mode (`--raw`)

1. **Tokens with raw positions.**
   - Each prose line is cleaned as before, but in place.
   - The line is XML-decoded with a map back to the dump's bytes.
   - A removed byte (template, ref, comment, `''` run) becomes a blank marker, and a replaced span keeps its length. So every token knows its raw byte span.
   - Tokens, sentences, subject forms, candidate objects and type checks are the same as F2's.
2. **Mentions.** A mention's raw span is widened over the markup that wraps it:
   - the whole link, with its target, pipe and glued letters, when its first or last token is a link token;
   - then a run of 2 or more quotes present on both sides.

   So the subject is `'''Andrey Markov'''`, and the object is `[[Ryazan]]`, `[[Ryazan Oblast|Ryazan]]` or the date `[[June 14]], [[1856]]`.
3. **The pattern** is built from the dump's own bytes:
   - the subject slot with its markup (`'''[S]'''`);
   - the bytes between the two mentions, XML escapes kept, digit runs written `#` and month names `<M>`;
   - the object slot with its markup (`[[[O]]]`).

   For example: `'''[S]''' ([[[O]]]` and `[S] was born in [[[O]]]`. The four context variants are kept: the bytes out to the end of the next token on each side, `^` and `$` at sentence ends. The gap may be at most 200 bytes, and F2's limit of 1 to 6 tokens between the mentions still holds.
4. **Everything else is unchanged:** seeds, the 5-fold split, checks against seeds, the gate, extraction, ECS fusion and the Wikidata gold fifth.

### Details decided from the data

| Detail | Choice | Evidence |
|---|---|---|
| Mention that cuts a link | Not used | The first hand-check draw found 5 birth places such as "New York", a chunk inside the single link `[[Ballston Spa, New York]]`. Widening it to the whole link made `[S] was born in [[[O]]]` match while extracting part of the link. After the fix, raw extracts the whole link "Duluth, Minnesota" |
| An entity token's raw span | The whole entity | `&amp;ndash;` cleans to one `-` token. Its span first covered only `&amp;`, so contexts read `&amp;`. Fixed before any result was kept |
| XML escapes | Kept | They are the bytes CyphaLM read. 209 of 1,258 candidates and 21 of 81 kept patterns hold `&amp;` or `&quot;` |
| Dash forms | Kept apart | Kept patterns split three ways: `&amp;ndash;` 21, the UTF-8 en dash 12, a plain hyphen 8. Folding them to `-` (an experiment, then removed) gives 64 kept and 1,126 admitted, still 90 below the cleaned run |
| Templates and refs between mentions | Kept verbatim | 16 candidates hold `{{`, none is kept; no candidate holds a ref. Gaps: median 12 bytes, at most 67; kept, at most 36 |
| Slot markup | Part of the key | 976 of 1,258 candidates and 80 of 81 kept patterns link the object (`[[[O]]]`). 604 candidates and 67 kept patterns have a bold subject (`'''[S]'''`) |

### Frames for R3: `patterns_raw.tsv`

- **Columns:** the same 11 as D4's `patterns.tsv`, plus a 12th, `frame`.
- **The frame** is the pattern without its `^` / `$` ends, with `[P]` read as `[S]`. Write the subject in place of `[S]`. The bytes before `[O]` are the prefix CyphaLM continues; they end with the object's opening markup. For example, `[S] was born in [[[O]]]` gives the prefix `Andrey Markov was born in [[`.
- **A linked date** fills its slot as the dump writes it: `[[[O]]]` holds `June 16]], [[1874`.
- **Usable frames.** A frame R3 can use has `[S]` before `[O]` and no `#` or `<M>`. 52 kept raw patterns qualify, with 17 distinct frames.
- **Caution.** `cyphalm_trie --r3` reads only column 2 and splits it at spaces, so it does not substitute `'''[S]'''` or `[[[O]]]`. Its loader must read `frame` before this file is passed to it.

### Frame reach (`--reach PARSE`, grading only, dev only)

- **Questions.** For each dev question, the entity is E1's (A3's parse file) and the relation is the `cyphalm_trie dktext` template that, filled with the entity, equals the question. This picks the same 32 questions on which A3's R3 ran.
- **Frame sets.** "Kept" is every kept frame of the relation. "R3" is what `--r3` loads: kept frames first, then candidates by training subjects, at most 3.
- **Three measures per frame.** The text is the training bytes `[0, 7340032)`, and again with the shard's responses on rows not marked `_holdout`.
  - **Reach:** the prefix with the subject (article dropped) occurs verbatim.
  - **Tie:** the prefix from the subject's last word on occurs, which is the least InfiniGram needs to tie the subject to an object.
  - **Leads to a dev answer:** a dev answer (its year, for a date) lies in the 48 bytes after a tie.
- This reads dev answers, never the test's. Nothing answers a question.

### The raw-form shard (`--join FILE`, raw mode)

- **Facts.** `--join` adds another run's admitted facts to this run's store: D4's 1,216 cleaned facts join the 1,025 raw ones.
- **Response form.** In raw mode, each fact of a relation with a raw frame is written in that relation's first R3 frame that is a bare core:

  | Relation | Frame |
  |---|---|
  | birth_place | `[S] was born in [[[O]]]` |
  | death_place | `[S] died in [[[O]]]` |
  | birth_date | `'''[S]''' ([[[O]]]` |
  | death_date | `[S] died on [[[O]]]` |
  | language | `'''[S]''' ([[[O]]]` |

  An open "(" is closed: "'''Al Gore''' ([[March 31]], [[1948]])." Other relations keep D4's sentence.
- **Years** are written as years with D4's sentence ("Abraham Lincoln died in 1865."). The dump never writes "died on [[1865]]".
- Questions, the slice, the `_holdout` tag and the leakage check are D4's.

## Results

### Patterns and facts, cleaned against raw (`--cv`, Wikidata seeds)

| Relation | Candidates cleaned / raw | Kept cleaned / raw | Admitted cleaned / raw | Wikidata check cleaned / raw |
|---|---|---|---|---|
| birth_place | 126 / 106 | 8 / 8 | 122 / 122 | 12/16 / 11/15 |
| death_place | 72 / 66 | 2 / 1 | 30 / 15 | 3/3 / 1/1 |
| birth_date | 111 / 122 | 10 / 16 | 605 / 576 | 85/87 / 83/85 |
| death_date | 148 / 190 | 14 / 15 | 423 / 292 | 71/74 / 55/57 |
| birth_year | 168 / 187 | 15 / 18 | 21 / 13 | 1/1 / 1/1 |
| death_year | 209 / 253 | 17 / 19 | 13 / 5 | 1/1 / 0/0 |
| language | 25 / 24 | 4 / 4 | 2 / 2 | none checked |
| capital, author, director, composer, founder | 41 / 25 | 0 / 0 | 0 / 0 | |
| all other relations | 458 / 285 | 0 / 0 | 0 / 0 | |
| **Total** | **1,318 / 1,258** | **70 / 81** | **1,216 / 1,025** | **173/182 / 151/159** |

- **Raw extractions:** 1,077 (cleaned 1,435). Candidates (P 0.5 to 0.8): 52 (cleaned 178).
- **Top kept raw patterns** (precision, checks):
  - `^'''[S]''' ([[[O]]]` for birth_year (0.998, 450);
  - `'''[S]''' ([[<M> #]], [[#]] &amp;ndash; [[[O]]]` for death_year (0.995, 189);
  - `'''[S]''' ([[[O]]] &amp;ndash;` for birth_date (0.989, 179);
  - `[S] was born in [[[O]]],` for birth_place (0.920, 88);
  - `[S] died in [[[O]]]` for death_place (0.905, 21; lower bound 0.711);
  - `'''[S]''' ([[[O]]]` for language (1.000, 33).
- **Lost:** the cleaned run's pronoun death-place pattern `[P] died in [O]` splits into `[P] died in [[[O]]]`, which has 18 of 20 right and a Wilson lower bound of 0.699, just under the gate.

### Which facts differ

| | Facts | By relation |
|---|---:|---|
| Admitted by both runs | 983 | |
| Only cleaned | 232 | death_date 131, birth_date 41, death_place 30, birth_place 14, birth_year 8, death_year 8 |
| Only raw | 41 | death_place 15, birth_place 14, birth_date 12 |
| Union | 1,257 | Wikidata check 176/185 = 0.951 [0.910, 0.974] |

- **Why raw misses facts.** One cleaned pattern covers many byte forms. For example, `^ [S] ( # - [O] )` covers `([[1893]]-[[1980]])`, `([[1364]] -[[1430]])` and `([[239 BC|239]] - [[169 BC]])`. Others are `[[July 19]],[[1898]] &amp;ndash;` (no space), unlinked places ("Soddy was born in Eastbourne") and the 30 pronoun death places.
- **Why raw finds facts the cleaned run misses.**
  - **Birth places.** The cleaned run extracts both the chunk "Duluth" and the link "Duluth, Minnesota" from `[[Duluth, Minnesota]]`. The two values compete in ECS and both stay at P 0.72–0.73. Raw does not use a chunk that cuts a link, so the link alone is admitted at P 0.94.
  - **Death places** come from `[S] died in [[[O]]]`.
  - **Birth dates** come from dash variants.

### Other configurations

| Configuration | Cleaned: candidates / kept / admitted / Wikidata check | Raw: candidates / kept / admitted / Wikidata check |
|---|---|---|
| Text seeds only, `--cv`, checked on all subjects | 676 / 44 / 832 / 684/726 = 0.942 [0.923, 0.957] | 688 / 48 / 659 / 538/569 = 0.946 [0.924, 0.961] |
| Wikidata seeds, `--cv` (main) | 1,318 / 70 / 1,216 / 173/182 = 0.951 [0.909, 0.974] | 1,258 / 81 / 1,025 / 151/159 = 0.950 [0.904, 0.974] |
| Wikidata seeds, single 80/20 split | 1,050 / 25 / 1,020 / 147/152 = 0.967 [0.925, 0.986] | 1,008 / 24 / 847 / 122/125 = 0.976 [0.932, 0.992] |
| Wikidata seeds, `--cv`, dash forms folded (experiment, removed) | | 1,145 / 64 / 1,126 / 162/170 = 0.953 [0.910, 0.976] |

- **The same pattern every time.** Raw admits 16–21% fewer facts at the same precision.
- **Text seeds only.** 648 facts come from both runs, 184 only from cleaned, 11 only from raw.
- **Dash folding.** It closes 92 of the 232 cleaned-only facts and leaves 140. The rest are link layouts, unlinked objects and pronoun death places. Folding makes frames non-verbatim, so it was dropped.

### Precision

**Against Wikidata.** As in the table above. The two runs cannot be told apart: raw 0.950 [0.904, 0.974], cleaned 0.951 [0.909, 0.974]. Of the 41 raw-only facts, 3 have gold values, and all 3 agree.

**Hand check.** Judged from my own knowledge.
- **First draw (found a bug).** I took the 18 raw-only facts then present, plus 12 random facts from both runs.
  - 16 of the 18 were right, but 5 were coarse (a state for a town). That was the link-cutting bug above.
  - 2 were wrong: Damon Runyon born 1884 (he was born in 1880) and William Kidd born 1654 (most sources give about 1645). Both errors are in the 2006 articles.
  - The 12 shared facts were all right.
  - I fixed the bug and drew again. The fix was informed by the first draw, so the second draw is a fresh sample but not a blind design.
- **Second draw** (30 of the 41 raw-only facts, `random.seed(20261028)`): **30 of 30 right**, Wilson [0.886, 1.000].
  - 13 death places, for example Columbus in Valladolid, Cortés in Castilleja de la Cuesta, Schrödinger in Vienna.
  - 11 birth places at town level, for example Bob Dylan in Duluth, Seaborg in Ishpeming, Hackman in San Bernardino.
  - 6 birth dates.
  - The file is `hand30_raw3.tsv`.
- **All 41 raw-only facts:** 39 right, 0.951 [0.839, 0.987]. The 2 wrong are Runyon and Kidd again.

### Capital, author, director, composer, founder

| Relation | Candidates cleaned / raw | Best raw candidate (checks, right) | Best cleaned candidate (checks, right) |
|---|---|---|---|
| capital | 8 / 4 | `[S] is [[[O]]]` (1, 0) | `[S] is [O]` (1, 0) |
| author | 19 / 10 | `[O]'s [S]` (4, 4); `'''''[S]''''' is a [[science fiction]] novel by [[[O]]]` (3, 3) | `[O] ' s [S]` (7, 5) |
| director | 9 / 9 | `[P] was directed by [[[O]]]` (7, 6) | `[P] was directed by [O]` (9, 6) |
| composer | 0 / 0 | none | none |
| founder | 5 / 2 | `[S] was founded in [[#]] by [[[O]]]` (1, 1) | `[S] was founded in # by [O]` (7, 5) |

Raw forms do not make these wordings more regular; they make them rarer. "is a novel by" splits further by markup: `'''''[S]''''' is a [[science fiction]] novel by`. D4's diagnosis stands: the 2006 text does not state these relations in one wording often enough, and a country's capital is stated in the city's article, which F2 does not read.

### Frame reach (dev, grading only)

36 dev questions have an E1 entity and a template relation; 17 of them are in-slice. The table counts in-slice questions; "leads to a dev answer" is defined only for those.

| Frames | Text | With a frame | Prefix with subject occurs | Ties | Tie leads to a dev answer |
|---|---|---:|---:|---:|---:|
| Cleaned, kept | training bytes | 13 | 2 | 6 | 2 |
| Raw, kept | training bytes | 13 | 0 | 3 | **3** |
| Cleaned, kept | + D4's shard | 13 | 9 | 11 | 4 |
| Raw, kept | + raw-form shard | 13 | 5 | 7 | **7** |
| Cleaned, R3's top 3 | training bytes | 17 | 4 | 10 | 2 |
| Raw, R3's top 3 | training bytes | 16 | 1 | 4 | 3 |
| Cleaned, R3's top 3 | + D4's shard | 17 | 11 | 14 | 4 |
| Raw, R3's top 3 | + raw-form shard | 16 | 6 | 8 | 7 |

- **Kept frames, all 36 questions, training bytes:** cleaned frames reach 2 of 16 questions with a frame (0.125 [0.035, 0.360]); raw frames reach 0 of 16 ([0.000, 0.194]).
- **Where the raw ties lead to the answer** (training bytes):
  - Markov: `Markov was born in [[` gives Ryazan.
  - Johnston: `Johnston was born in [[` gives Washington, Kentucky.
  - Angola's language: `Angola''' ([[` gives Portuguese.
- **Shard reaches.**
  - **Raw-form shard:** Meighen, Al Gore (`'''Al Gore''' ([[`), Schweitzer, Johnston and Alcuin. All 5 lead to the answer.
  - **D4's shard:** 9 reaches. Only 4 lead to the answer.
  - **A false reach.** The birth-place frame `Allan Dwan was born in ` matches D4's birth-year sentence "Allan Dwan was born in 1885.". The raw frame `was born in [[` cannot match a year: the link markup tells a place from a year.
  - The other cleaned false reaches are "Angola (" and "Avicenna (" somewhere in the text, and the year sentences of De Morgan and Koestler.
- **Why the full prefix rarely occurs.** For all 5 in-slice birth-date questions, the lead does not take the common form `'''[S]''' ([[[O]]]` with the question's name:
  - **Another form:** `'''Arthur Meighen''', [[Queen's Privy Council for Canada|PC]]` and `'''Alan Kay''', born [[May 17]]`.
  - **Another name:** `'''Andrei Arsenyevich Tarkovsky'''` and `'''Albert Arnold Gore, Jr.'''`.
  - **No date in the lead:** `'''Avicenna''' was a [[Persian]]`.
- **Precision of ties.** On the training bytes, 3 of 3 in-slice raw ties lead to the dev answer ([0.439, 1.000]), against 2 of 6 cleaned ones (0.333 [0.097, 0.700]). With the shards: 7 of 13 in-slice questions with a frame (0.538 [0.291, 0.768]) against 4 of 13 (0.308 [0.127, 0.576]). The intervals overlap.
- **This is a ceiling, not an R3 result.** It says where the bytes exist. R3 itself must be run again with a loader that reads `frame`.

### The raw-form shard

| | D4's shard (default) | Raw-form shard |
|---|---|---|
| File | `/root/work/f2/shard_facts.tsv` | `/root/work/f2raw/shard_facts_raw.tsv` |
| Extracted facts in the store | 1,216 | 1,257 (1,025 raw + 232 joined) |
| Facts in the slice (sentences) | 675 | 678 |
| … written in a raw frame | 0 (4 from the fact's own pattern) | 177 |
| Rows | 2,001 | 2,010 |
| Rows by source: seed, seed_holdout, extracted, extracted_holdout | 1,260, 426, 228, 87 | 1,260, 426, 234, 90 |
| Rows dropped as equal to a dev / test question | 18 / 6 | 18 / 6 |
| Final leakage check (questions equal to dev or test) | 0 | 0 |
| In-slice dev (entity, relation) in the shard; response holds a dev answer | 16; 15 | 16; 15 |
| Year rows written with a full date | 6 | 0 |

- **The sentences file.** `shard_facts_raw_sentences.txt` has 678 lines.
- **The facts the raw rows add.** The 41 raw-only facts add 3 facts to the slice. The dev reach does not change.
- **What the raw rows change.** Their form matches the raw frames. 521 rows contain `[[`.

## Decision: join

| Option | What it gives | What it costs |
|---|---|---|
| Replace (raw only) | Frames in CyphaLM's bytes; 41 facts | 232 facts (19%), all 30 pronoun death places |
| **Join** | 1,257 facts at 176/185 = 0.951 [0.910, 0.974]; raw frames and a shard in their bytes | A second run (42–58 s) |
| No change | Nothing to do | 41 precise facts; frames that tie spuriously |

- **Facts.** The cleaned patterns stay the fact source. Raw splits one wording into many byte forms and admits 16–21% fewer facts in every configuration, at the same precision. The 41 raw-only facts are right 30 of 30 in a random draw, so they join.
  - The join is `--raw --join aug/facts_f2.tsv --shard ...`.
  - It adds no conflicting value: the 2 subject–relation pairs with two values were already in the cleaned run.
- **Frames for R3.** Use the raw frames with the raw-form shard.
  - Their ties lead to a dev answer for 7 of the 13 in-slice questions with a frame, against 4 for cleaned frames with D4's shard.
  - They make no false ties on the training bytes.
  - The cleaned frames stay paired with D4's plain shard: a raw frame `was born in [[` does not match "was born in Kaysersberg.".
- **The default does not change.** Without `--raw`, all six of D4's runs give byte-identical outputs: 55 of 55 files, and the stats are equal except for timing and paths.

## Predictions, scored

| # | Prediction | Result | |
|---|---|---|---|
| R1 | 1.2 to 3 times the cleaned 1,318 candidates | 1,258 (0.95 times). Markup splits wordings, but split forms seen in fewer than 2 training subjects are not candidates. Dates gain candidates (birth_year 168 to 187); the rest lose them (occupation 96 to 56) | **Miss** |
| R2 | 40 to 90 kept, mostly dates and birth places | 81: 68 dates and years, 9 places, 4 language | **Hit** |
| R3 | At most one of the five hard relations gets a kept pattern | 0 | **Hit** |
| R4 | Admitted within 25% of 1,216; raw-only 30 to 200 | 1,025 (−15.7%); 41 raw-only | **Hit** |
| R5 | Wikidata agreement at least 0.90, lower bound at least 0.85 | 0.950 [0.904, 0.974] | **Hit** |
| R6 | At least 26 of 30 raw-only facts right | 30/30 (second draw); 28/30 in the first draw (before the fix) | **Hit** |
| R7 | Kept-frame prefix reach at most 10% cleaned; at least 20% raw and at least twice cleaned | Cleaned 2/16 = 12.5% (both spurious); raw 0/16 | **Miss** (both parts) |
| R8 | Join: the union beats either run at gold precision within 0.02 of cleaned | 1,257 against 1,216 and 1,025; 0.951 against 0.951 | **Hit** |

## Flags for Odin

**Flags 3 and 4 decided 2026-10-28, from the evidence** (Odin: decide by measurement):
- **Flag 3 (year rows): fixed in every mode.** A year fact now keeps its year only. The default shard changes in exactly the 6 rows of the flag (Aldous Huxley's birth and death years); every fact and every count is unchanged (rows 2,001; Wikidata check 173/182). New default shard: `/root/work/f2/v2/shard_facts.tsv`.
- **Flag 4 (the link-cut rule in the cleaned mode): not adopted.** Tried in the cleaned mode (`--cv --wikidata`): birth places 122 → 132, birth and death dates +6 and +7, but **death places 30 → 0** (all 30 were right in the hand check), so admitted facts fall by 7 and the Wikidata check is 170/179 (0.950) against 173/182 (0.951). The rule stays raw-only.
- **Flags 1 and 2** are passed to the answering-variants run (A1D/A2B).

The flags as raised:

1. **`cyphalm_trie --r3` cannot read raw frames yet.** Its loader renders column 2 by splitting at spaces, so `'''[S]'''` and `[[[O]]]` are not filled, and it would fail silently. It must read column 12 (`frame`) instead: fill `[S]`, and continue from the bytes before `[O]`.
   - R3's candidate cut stops at `|`, so a continuation `Ryazan Oblast|Ryazan]]` gives "Ryazan Oblast".
   - It does not stop at `]]`, so `Ryazan]] in 1856` would give "Ryazan in 1856". It should stop there too.
2. **Run R3 with raw frames and the raw-form shard** against run F2 of A2/A3 (cleaned frames, D4's shard). The reach table bounds them at 7 and 4 of the 100 in-slice dev questions.
3. **D4's default shard writes 6 year rows with a full date**, for example "Aldous Huxley died in November 22, 1963.". An extracted year fact keeps its candidate's full date. The fix is one line (keep the year only). It changes `shard_facts.tsv`, so I applied it only in the raw-form shard. Odin should rule whether the default gets it.
4. **The cleaned run loses town-level birth places to its own chunks.** A chunk inside a link ("Duluth" in `[[Duluth, Minnesota]]`) competes with the link, and neither is admitted. The raw rule (no mention cuts a link) fixes this for raw. The same rule in the default would change D4's outputs, so it waits for a ruling.
5. **Hard relations need another route.** Raw bytes do not help capital, author, director, composer or founder. D4's next steps stand: pattern classes in the gaps, and cross-article evidence (a country's capital is stated in the city's article).

## Costs

| Item | Value |
|---|---|
| `native/tools/pattern_learn.cpp` | 2,303 → 2,708 lines (+427, −22). In-place cleaning 83; token positions about 30; raw keys and mentions 68; raw frames, `--join` and the raw-form shard about 75; `--reach` 128; flags and header about 20 |
| `pattern_learn --raw --cv` with Wikidata | 41.6–57.8 s, 117–119 MB, one core (the machine was shared) |
| The same with `--join`, `--shard` and `--reach` | 41.6 s, 119 MB |
| Cleaned run, for comparison | 26.6–27.4 s, 118–119 MB |
| Outputs | `patterns_raw.tsv` 1,258 patterns (0.1 MB); `shard_facts_raw.tsv` 2,010 rows (0.2 MB) |

## Failure modes and cautions

- **Fragmentation is the price of verbatim bytes.** One cleaned wording becomes several raw patterns (dash forms, spacing, linked or not), and each must reach 20 checks alone. Raw keeps more patterns (81 against 70) but extracts fewer facts.
- **The subject in the question is not the subject in the lead.** Frames with a bold subject need the lead's exact bold name and form. R3 fills the question's name. Neither fits any of the 5 in-slice birth-date questions.
- **R3's candidate frames are junk in both modes.** With no kept capital pattern, R3 loads candidates: `[S] are [[[O]]]` (raw) and `[S] are [O]` (cleaned).
- **The raw-form shard inherits D4's seed values**, for example two birth places for Arthur Meighen (Anderson and Manitoba).
- **Small numbers.** The frame comparison rests on 13 in-slice questions, and every interval overlaps.
- **Text errors pass through.** Runyon and Kidd are stated wrongly in the 2006 articles, and the extraction is faithful to them.

## Reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
E=/root/cdata/enwik8; DEV=bench/eval/t_qa_wiki_dev.tsv; TEST=bench/eval/t_qa_wiki.tsv
F=/root/work/f2; O=/root/work/f2raw; W=$F/wikidata/wd_values.tsv
PAR=parse_dev.tsv   # E1's parse of the dev questions, A3's --parse file (/root/work/answer/parse_dev.tsv)
# raw, main run: patterns, facts, the raw-form shard (D4's facts joined), frame reach
./pattern_learn $E $O/raw --raw --cv --wikidata $W --join $F/aug/facts_f2.tsv --shard $O/shard_facts_raw.tsv --dev $DEV --test $TEST --reach $PAR
cp $O/raw/patterns.tsv $O/patterns_raw.tsv
# cleaned frames with D4's shard, for the reach comparison (its outputs equal D4's)
./pattern_learn $E $O/reach_clean --cv --wikidata $W --shard $O/reach_clean/shard_facts.tsv --dev $DEV --test $TEST --reach $PAR
# raw against D4's other configurations
./pattern_learn $E $O/raw_basegold --raw --cv --wikidata $W --wikidata-gold-only
./pattern_learn $E $O/raw_split --raw --wikidata $W
# no regression: D4's six runs (D4 report, Reproduce) into $O/regress/<run>, then cmp every file against $F/<run>
```

- **stdout lines** (also in `stats.txt`):
  - `PATTERNS`, `NEW FACTS` and `WIKIDATA CHECK`, as in D4;
  - `join`, `shard_raw_frame`, `SHARD`, `shard_source` and `leak_check` for the shard;
  - `REACH_FRAME`, `REACH` and `REACH_SUMMARY` for the frame reach.
- **Overlap counts** (983, 232 and 41) come from comparing the two `facts_f2.tsv` files by subject, relation and object, at P at least 0.8. `join … new to this run's store 232` prints the cleaned-only count.
- **Hand checks:** `/root/work/f2raw/hand30_raw3.tsv` (the 30 judged facts, column `hand`) and `hand30_first_draw.tsv`.
- **sha256:**
  - `patterns_raw.tsv`: 57dbfb33…87653f;
  - `shard_facts_raw.tsv`: f073aad2…cf207e;
  - `shard_facts_raw_sentences.txt`: 3e5775c2…d37.
- **Runs whose code is not kept** (their outputs are in `/root/work/f2raw/`): `raw1` and `raw2` (before the entity-span and link-cut fixes), and `raw_fold` (the dash-folding experiment).
