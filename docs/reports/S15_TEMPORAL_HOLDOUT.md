# S15: the temporal-holdout benchmark (2006 text against current Wikidata)

**Odin Loch** · 2026-10-28 · DISCOVERY §7.1 T9, §12.2, §13 S15 · follows [D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md](D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md) and [S4_UNSEEN_AND_FRONTIER.md](S4_UNSEEN_AND_FRONTIER.md)

(The design, the predictions and the freezing rule below were written before the benchmark was built. Results are added under them.)

**Question.** enwik8 is a Wikipedia snapshot from March 2006. Wikidata is current. Which facts about the snapshot's subjects does Wikidata hold that the 2006 text does not state, and which of them only became true after the snapshot? Those facts are a fair test for Cypha's discovery processes: a generator that proposes them could not have read them.

**Gate (DISCOVERY §13).** The benchmark is built and frozen in `bench/eval/`.

**Rule 10.** This is an evaluation set for Cypha's discovery processes (T9). It never feeds an answer, a fact store or CyphaLM's training text.

## Design (fixed before the build)

### The snapshot date

The latest revision timestamp in enwik8 is `2006-03-04T06:14:28Z` (12,347 timestamps read). The snapshot date is **2006-03-04**. A fact "became true after the snapshot" if its date is after 2006-03-04.

### Subjects

- enwik8's 8,587 main-namespace articles (`pattern_learn`'s page list), mapped to Wikidata items as D4 mapped them: 8,176 titles, 8,097 items.
- D4's 302 excluded titles stay out: they are T-QA dev or test subjects, or named in a T-QA question. A discovery that reached the store and a shard could otherwise leak into the Q/A test.
- Each subject is marked by where its page starts: `train` (before byte 7,340,032, CyphaLM's 7 MiB train part), `holdout` (before byte 8,388,608, the rest of the 8 MiB slice), or `rest` (the rest of enwik8, which Track S reads).
- Each subject gets a type by a fixed rule: `person` (instance of Q5), `country` (instance of a country class: Q6256, Q3624078, Q3024240, Q7275), `place` (has a coordinate location, P625), else `other`.

### Facts

Every non-deprecated Wikidata statement of these properties, for every subject item:

- **D4's relations** (what Cypha's pattern learner extracts): capital P36, author P50, director P57, spouse P26, composer P86, founder P112, language P37, country P17, inventor P61, creator P170 / P84, genre P136, birth_place P19, death_place P20, located_in P131, currency P38, occupation P106, developer P178, publisher P123; dates birth P569, death P570, founded P571 (as birth_date / birth_year, death_date / death_year, founded, the way D4 writes them).
- **Dated relations added for S15** (they carry start time, end time or point in time qualifiers often): award_received P166, position_held P39, member_of_sports_team P54, member_of_political_party P102, member_of P463, head_of_government P6, head_of_state P35, educated_at P69, employer P108, owned_by P127, parent_organization P749, headquarters P159, chairperson P488, director_manager P1037, dissolved P576 (a date).
- Entity values keep their item, their enwiki title or English label, and their aliases (as D4 fetches them). Dates use D4's conversion (Julian-model dates back to the Julian calendar; years only below day precision).
- A statement's qualifiers start time P580, end time P582 and point in time P585 are kept (the earliest, if several).

### Categories (one per fact)

The fact's **date** is: the value itself for date relations; the death date for death_place; else the start time, or else the point in time.

- **c, after the snapshot:** its date is after 2006-03-04 (a year after 2006, a month after 2006-03, a day after 2006-03-04).
- **u, unknown when:** its date straddles the snapshot (the year 2006 or the month 2006-03 without a day, or the day 2006-03-04); or it has no date, only an end time after the snapshot; or it has no date at all and its relation can change over time (every relation except birth_*, death_*, founded, dissolved, author, composer, director, inventor, creator, founder, genre, language).
- Otherwise the fact was true by 2006 (a historical fact counts, ended or not). Then:
  - **a, stated in the 2006 text**, if any of: the v2 fact store holds it (`/root/work/f2/v2/aug/facts_f2.tsv`, admitted), a 2006 text seed holds it (`seeds.tsv`, any source but `wikidata`), or the subject's own 2006 page names the value (the **page rule**, below);
  - **b, true in 2006 but unstated** otherwise.
- **The page rule (for labelling only).** Text is lowered, accents stripped and every run of non-alphanumerics made one space (D4's `norm`). An entity value is named if one of its surface forms (enwiki title with any "(…)" suffix dropped, English label, aliases of 3 or more characters) occurs on word boundaries in the subject's page text. A year is named if the 4-digit year occurs. A day date is named if its year occurs and its day and month name occur next to each other ("19 February" or "February 19"). The rule is generous on purpose: a value named for another reason still counts as stated, so `b` stays clean.
- Every fact also gets `named_on_page` (the page rule, for every category) and `named_elsewhere` (a surface form occurs on another enwik8 page).

### Negatives

One false fact per true fact, by a fixed rule: the same subject and relation, with a value drawn from that relation's values over all subjects (each true fact one ticket, so frequent values are drawn as often as they occur), using a seeded generator per fact; a value the subject holds for that relation in any statement (ended or not) is redrawn. A negative takes its pair's category, so every category can be scored as a ranking. Type consistency comes from the relation (birth_place draws places, birth_year draws years).

### Splits

By subject: `test` if the first byte of sha256("S15|" + item) is 128 or more, else `dev`. The rule needs no seed file and does not depend on order.

### Files

- `bench/eval/temporal_holdout_dev.tsv`, `bench/eval/temporal_holdout_test.tsv`: one row per fact, true and false.
- `bench/eval/README.md`: a row per file with its sha256.
- The builder: `scripts/temporal_holdout.py` (Python standard library, like D4's fetcher). It reuses D4's `scripts/wikidata_seeds.py` (its title mapping, its fetcher with the response cache, its date conversion), so D4's script gains a `main()` guard and writes its title-to-item map; its other outputs stay byte-identical. Raw responses are cached under the run directory, so a rebuild from the cache is byte for byte.

## Freezing rule (fixed before the build)

1. The files are frozen at their commit. Their sha256 values are in this report and in `bench/eval/README.md`.
2. They are never edited. A fix makes a new version (`temporal_holdout_v2_*`), with its own section here; the old version stays.
3. **dev** may be read freely while a generator is developed. **test** is read only by a generator's final read, once per change. Each read is logged in that generator's report and in the "Reads" table below.
4. The files never become training text for CyphaLM. They never enter a fact store, a shard, a teacher prompt or a pattern seed. `scripts/temporal_holdout.py --check FILE …` reports any line of a text or shard file that names a subject together with the value of one of its `b` or `c` facts; run it on new training text. (qa's `--leak` compares question strings only, so it cannot see these facts.)
5. Nothing in this step scores a generator on the benchmark.

## Predictions (registered before the build)

| # | Prediction |
|---|---|
| T1 | D4's script, rerun from its cache with the `main()` guard, writes `wd_values.tsv`, `phi.tsv`, `excluded.tsv` and `unmapped.tsv` byte for byte |
| T2 | Category `c` holds 3,000–12,000 true facts; award_received and position_held together make at least half of them |
| T3 | 120–400 subjects died after the snapshot (death dates in `c`) |
| T4 | Among true facts that were true by 2006 (`a` + `b`), the share unstated (`b`) is 25–55% overall, and 15–40% for birth_place |
| T5 | Category `u` holds 20–50% of all true facts |
| T6 | 47–53% of subjects fall in `test` |
| T7 | Of `c` facts, 15–50% are named on the subject's 2006 page anyway (recurring values such as an award or a party already named), so the date, not the text, must decide `c` |
| T8 | The two files together are under 15 MB |
| T9 | A rebuild from the cache gives the same sha256 for both files |

## Results (built 2026-10-28)

**The gate is met.** The benchmark is built and frozen in `bench/eval/`:

| File | Rows (true + false) | Bytes | sha256 |
|---|---:|---:|---|
| `temporal_holdout_dev.tsv` | 38,162 | 4,860,728 | `9e1ae1ba9420b1b53e358ee815f431ecdeaad907d2a43eeb0235371a3e6ccc51` |
| `temporal_holdout_test.tsv` | 38,624 | 4,945,022 | `1535dcdd77a6d150604c7565e66566dd87f09a08a564941cca33baf855b32fb6` |

- **Columns:** `id`, `subject`, `subject_item`, `subject_type`, `region`, `relation`, `property`, `value`, `value_item`, `date`, `date_kind` (value, death, start, point, end or none), `category`, `stated_by` (store, seed or page, for `a`), `named_on_page`, `named_elsewhere`, `label` (1 true, 0 false), `pair` (the other half of the pair).
- **Size:** 38,393 true facts and 38,393 false ones, 4,059 subjects (1,997 dev, 2,062 test).
- **Wikidata:** 8,097 items; 33,729 entity-statement rows and 4,315 time-statement rows; 14,507 values labelled. 383 requests in about 17 minutes, cached (32 MB).

### Categories

| | a: stated | b: unstated by 2006 | c: after the snapshot | u: unknown when | Total |
|---|---:|---:|---:|---:|---:|
| dev | 5,215 | 4,228 | 1,003 | 8,635 | 19,081 |
| test | 5,306 | 4,346 | 914 | 8,746 | 19,312 |
| all | 10,521 | 8,574 | 1,917 | 17,381 | 38,393 |

- **Unstated share:** b is 44.9% of the facts that were true by 2006 (27.4% for birth_place).
- **`a` comes from:** the page rule 7,233, a 2006 text seed 2,349, the fact store 939.
- **Of `b`, 51.4%** name the value somewhere else in enwik8, which a link predictor could use.
- **`c`:** 99 subjects died after the snapshot. The largest `c` relations are award_received 526, head_of_government 380, chairperson 185, member_of 181 and head_of_state 131. Only 5.8% of `c` values are named on the subject's 2006 page.
- **`u`** is mostly undated occupation (6,428), country (2,321), member_of (2,134) and award_received (1,785).
- **Subjects** by type: other 2,076, person 1,246, place 660, country 77. By region: train 105, holdout 82, rest 3,872.

True facts by relation and category (both halves):

| Relation | a | b | c | u | Total |
|---|---:|---:|---:|---:|---:|
| occupation | 32 | 21 | 1 | 6,428 | 6,482 |
| award_received | 418 | 2,243 | 526 | 1,785 | 4,972 |
| member_of | 224 | 1,186 | 181 | 2,134 | 3,725 |
| country | 253 | 165 | 10 | 2,321 | 2,749 |
| educated_at | 180 | 188 | 0 | 1,231 | 1,599 |
| genre | 626 | 901 | 0 | 0 | 1,527 |
| birth_year | 1,079 | 246 | 0 | 0 | 1,325 |
| position_held | 502 | 449 | 19 | 343 | 1,313 |
| birth_place | 947 | 357 | 0 | 0 | 1,304 |
| spouse | 328 | 279 | 17 | 671 | 1,295 |
| founded | 787 | 347 | 10 | 0 | 1,144 |
| death_year | 899 | 100 | 99 | 0 | 1,098 |
| death_place | 684 | 300 | 105 | 0 | 1,089 |
| birth_date | 807 | 243 | 0 | 0 | 1,050 |
| death_date | 651 | 218 | 99 | 0 | 968 |
| located_in | 131 | 104 | 14 | 645 | 894 |
| head_of_government | 87 | 281 | 380 | 38 | 786 |
| employer | 138 | 121 | 8 | 396 | 663 |
| chairperson | 158 | 101 | 185 | 27 | 471 |
| headquarters | 12 | 11 | 8 | 356 | 387 |
| founder | 267 | 87 | 0 | 0 | 354 |
| language | 255 | 83 | 4 | 0 | 342 |
| head_of_state | 70 | 96 | 131 | 10 | 307 |
| inventor | 205 | 91 | 0 | 0 | 296 |
| developer | 5 | 8 | 4 | 209 | 226 |
| capital | 30 | 14 | 4 | 175 | 223 |
| creator | 161 | 61 | 0 | 0 | 222 |
| owned_by | 21 | 16 | 35 | 147 | 219 |
| member_of_political_party | 32 | 24 | 4 | 149 | 209 |
| currency | 48 | 46 | 8 | 87 | 189 |
| author | 163 | 23 | 0 | 0 | 186 |
| member_of_sports_team | 77 | 35 | 4 | 23 | 139 |
| director_manager | 16 | 58 | 33 | 23 | 130 |
| dissolved | 84 | 18 | 20 | 1 | 123 |
| publisher | 5 | 0 | 2 | 105 | 112 |
| composer | 72 | 39 | 0 | 0 | 111 |
| parent_organization | 14 | 3 | 6 | 77 | 100 |
| director | 53 | 11 | 0 | 0 | 64 |

### One correction before freezing

The first build drew each false value from the relation's values over **all** subjects. That made false facts such as "Guatemala member_of The Dave Brubeck Quartet": a country's membership drawn from a band's. They would be rejected for their type alone. False values are now drawn from the relation's values over subjects of the **same type**, and from the whole relation only when none fits (10 of 38,393). No generator had read the files. Every other rule is as registered.

### Checks

- **D4 unchanged (T1).** D4's script, rerun from its cache with the `main()` guard, writes `wd_values.tsv`, `phi.tsv`, `excluded.tsv` and `unmapped.tsv` byte for byte, with 0 requests. It also writes `items.tsv` (8,176 titles).
- **Rebuild (T9).** Two rebuilds from the cache give the same sha256 for both files: 55 s, 590 MB peak, 0 requests.
- **Spot checks.** `c`: "Italy head_of_state Giorgio Napolitano, 2006-05-15", "H. R. Giger death_place Zurich, 2014-05-12", "Ian Murdock employer Sun Microsystems, 2007-03-19". False facts: "Guatemala member_of Visa Waiver Program", "Hanover head_of_government Walther Schreiber".
- **Leak check** (`check`, entity values only; a bare year matches too much prose):
  - D4's fact shard (`shard_facts.tsv`, CyphaLM training text): 0 lines.
  - Dolly (CyphaLM's chat text, written in 2023): 126 lines name a subject with one of its `b` or `c` values, 131 distinct `b` pairs and 35 distinct `c` pairs (India with Narendra Modi, Slovenia and the euro, Croatia and NATO). These are co-occurrences, not always statements. **A generator built on CyphaLM has read them**, so its final read must drop the pairs `check` reports on its own training text. Track S's store (enwik8 only) has not read them.

### Limits

- **CyphaLM's own slice is thin here.** Of the 536 pages in the 7 MiB train part, D4 excluded 205 (T-QA subjects) and 327 are mapped. Most of those are concept articles (Albedo, Anarchism) with none of these relations. So 105 train and 82 holdout subjects carry facts (680 and 1,026 true facts). The benchmark is for Track S, whose store reads all of enwik8.
- **The page rule is a string rule.** It misses a value named in other words ("won a Locus Award" against "Locus Award for Best Science Fiction Novel"), so some `b` facts are stated loosely. It also counts a value named for another reason as stated, which keeps `b` clean but makes `a` generous.
- **D4's title mapping carries over.** A few titles map to a related item ("Dvorak" to the composer). Their facts are about that item.
- **`u` is large (45%).** Undated statements of relations that can change (occupation above all) cannot be placed before or after 2006. They stay in the files, marked, for generators that score them anyway.
- **Wikidata is current, not complete.** A false fact is false by Wikidata's record; a missing statement can make a true fact look false. Recall over `c` is the safer score; precision over `c` against the false facts is a lower bound.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| T1 | D4's outputs byte for byte | **Hit** (4 of 4 files, 0 requests) |
| T2 | `c` 3,000–12,000; award_received + position_held ≥ half | **Miss**: 1,917; 28.4% (heads of government and chairpersons are a large share; most positions were held before 2006, by historical people) |
| T3 | 120–400 died after the snapshot | **Miss**: 99 (most of enwik8's people are historical) |
| T4 | `b` 25–55% of `a + b`; birth_place 15–40% | **Hit**: 44.9%; 27.4% |
| T5 | `u` 20–50% | **Hit**: 45.3% |
| T6 | `test` 47–53% of subjects | **Hit**: 50.8% |
| T7 | 15–50% of `c` named on the 2006 page | **Miss**: 5.8%; recurring values are rarer than expected |
| T8 | Under 15 MB together | **Hit**: 9.8 MB |
| T9 | A rebuild gives the same sha256 | **Hit** |

6 hits, 3 misses.

## Reads

Every read of `temporal_holdout_test.tsv` by a generator is logged here, with its report.

| Date | Generator | Change | Report |
|---|---|---|---|
| 2026-10-29 | G6 and G7 link prediction, their combination (S8b) | read once, the final setting chosen on dev (commit d50fdcd): Hits by category on the 2006-text store, and promotion through S7's loop | [S8B_LINK_PREDICTION.md](S8B_LINK_PREDICTION.md) |
| 2026-10-29 | G2 rule induction and G14 contradiction resolution (S8a), on the 2006-text store | the first read; the registered setting, commit 10bef62 | [S8A_RULES_AND_CONTRADICTIONS.md](S8A_RULES_AND_CONTRADICTIONS.md) |
| 2026-10-29 | S10 curiosity scheduler (P2's allocator over G2 and G6/G7) | read once on a fresh set built on test's bit-17 half (commit dc2e632); bit 17 = 0 untouched. Whether a scheduler's read counts under rule 3 is open (HANDOFF §5.18) | [S10_CURIOSITY_SCHEDULER.md](S10_CURIOSITY_SCHEDULER.md) |
| 2026-10-29 | G9 concept formation and G11 open relations (S11a) | read once (commit 913d2a7): G9's promotion read on test's bit 17 = 0 half (slots only: it made no hypotheses, so no label was used), and G11's 38 proposals graded on test subjects' rows | [S11A_CONCEPTS_AND_OPEN_RELATIONS.md](S11A_CONCEPTS_AND_OPEN_RELATIONS.md) |
| 2026-10-29 | G12 literature-based discovery, Swanson's ABC (S11b) | read once, the setting chosen on dev (PPMI, hub cap 200; commit 0230741): open-form precision@k and the closed form on S15 test's subjects, and promotion of its typed pool | [S11B_ABC_DISCOVERY.md](S11B_ABC_DISCOVERY.md) |
| 2026-10-29 | G3, G4, G5 (S11c) | read once with the settings chosen on dev (commit 804d77c): G4 and G5's ranks of `b` and `c` facts on the 2006-text store (all of test), and the promotions of G3, G4 and G5 on a fresh set built on test's bit-17 = 0 half (pool hashes committed first) | [S11C_ANALOGY_ANOMALY_ABDUCTION.md](S11C_ANALOGY_ANOMALY_ABDUCTION.md) |
| 2026-10-29 | G1 deep deduction, G19 refinement, G20 the wild tier (S12a) | read once (commit 6307b17): G1 and G19 on S8a's R1 hide with S15 test's rows in the grader, G1 on store T for S15 test's subjects, and the G1 and G20 pools on test's bit-17 = 0 half (the half S11c's pools used) | [S12A_DEDUCTION_LAWS_REFINEMENT.md](S12A_DEDUCTION_LAWS_REFINEMENT.md) |
| 2026-10-29 | G11 with the date-link skip (C2 by S11a's rule) and G9 on a richer store (S11d) | read once (commit 64068c5): G11's proposals on test subjects' rows; G9's A3 store takes S15 test's true rows of the 14 added relations as attributes (fresh classes, S11a's 60 classes again); G9's promotion read on the bit 17 = 0 half (0 hypotheses) | [S11A_CONCEPTS_AND_OPEN_RELATIONS.md](S11A_CONCEPTS_AND_OPEN_RELATIONS.md) |

## How to rebuild

```bash
# D4's title map, from D4's cache (copy /root/work/f2/wikidata/raw to $D4/raw first): 0 requests
python3 -I scripts/wikidata_seeds.py /root/work/f2/basefix/pages.tsv bench/eval/t_qa_wiki_dev.tsv bench/eval/t_qa_wiki.tsv $D4 30
# S15 (fetches into $RUN/raw the first time, then reads only the cache)
python3 -I scripts/temporal_holdout.py build /root/cdata/enwik8 /root/work/f2/basefix/pages.tsv $D4 \
  /root/work/f2/v2/aug/facts_f2.tsv /root/work/f2/v2/aug/seeds.tsv $RUN bench/eval
# leak check on any new training text or shard
python3 -I scripts/temporal_holdout.py check bench/eval FILE ...
```

- The cache of this build is committed as `bench/eval/temporal_holdout_wikidata_cache.tar.gz` (2.35 MB, sha256 `1969d192f910a23407954037f8e924581e91c02fa0dbe6aaa033602473b5d948`; 2026-10-29, decided under PLAN ledger 55: Wikidata changes, so without it the benchmark could not be audited). Unpack it into `$RUN` (`tar -xzf … -C $RUN` gives `$RUN/raw`). Checked: a rebuild from this archive alone, with D4's cache for the title map, gives both files' sha256 above, with 0 requests.
- Wikidata changes, so a build from live Wikidata will differ. **The committed files are the benchmark**; the cache only allows an audit.
