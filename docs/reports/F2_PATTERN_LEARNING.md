# F2: relation patterns learned from text by distant supervision

**Odin Loch** · 2026-10-27

**Update 2026-10-28.** 664 of the 8,148 seeds came from citation templates read under a stale template name, including all 101 infobox "author" seeds. With the fix there are 7,484 text seeds; the 920 facts and their hand check are unchanged ([D4 report](D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md)).

## Summary

`T_QA_WIKI_200` showed that the fact store answered 3 of 150 answerable held-out questions. F2 asks whether patterns learned from the text itself can add facts. The tool `native/tools/pattern_learn.cpp` takes seeds from infobox and template fields, finds the sentences that carry them, learns lexical patterns, measures each pattern on held-out subjects, and applies the reliable ones to every article.

- **Seeds:** 8,148 facts on 2,177 pages, over 31 relations.
- **Patterns:** 676 candidates. 44 pass the plan's gate (at least 20 checks, Wilson 95% lower bound of precision at least 0.7). They cover six relations: birth date, death date, birth year, death year, birth place, and language.
- **New facts:** 920 with P at least 0.5. Of these, 832 are admitted (P at least 0.8) and 88 are candidates. The store grows from 8,148 to 8,980 facts, about 10%.
- **Precision:** 49 of 49 verifiable facts right in a hand check of 50 random admitted facts (Wilson 95% [0.927, 1.000]). One fact I could not verify. Counting it as wrong gives 49/50, [0.895, 0.996].
- **Cost:** 16 s on one core, peak RAM 107 MB.
- **What did not work:** capital, author, director, spouse, genre and the other relations got no kept pattern. They have 25 to 150 seeds each, which gives fewer than 20 held-out checks. The growth is almost all dates. The 2006 dump has too few infoboxes for distant supervision to learn the relations the Q/A test asks about. See "Failure modes".

`bench/eval/t_qa_wiki.tsv` was not opened or scored.

Files, all outside the repo, in `/tmp/claude-0/-home-claude/73363f2d-2615-591f-aa38-d54363629bae/scratchpad/f2/`:

| File | Content |
|---|---|
| `facts_f2.tsv` | The new facts for the Q/A tool (920 rows). Columns: subject, relation, object, P, grade, n_evidence, best_pattern, source_page, byte_offset |
| `facts_f2.evidence.tsv` | The strongest evidence sentence for each fact |
| `seeds.tsv`, `patterns.tsv`, `stats.txt` | Seeds, all 676 candidate patterns with checks, and the run log |
| `sample_admitted.tsv` | The 50 hand-judged facts |
| `split80_20/` | The same run with a single 80/20 split |

`byte_offset` is the start of the line in the dump that holds the strongest evidence sentence. `source_page` is the article, which is always the subject. Dates are written `YYYY-MM-DD`, `YYYY-MM` or `YYYY`, and `490 BC` for years before the common era.

## Method

1. **Seeds.**
   - Fields `| key = value` in templates whose name is not a citation (infoboxes, `persondata`, `succession box`) are mapped to 32 relations. 3,286 seeds come from here.
   - Wiki-table rows such as `| Capital || [[X]]` give 16 more.
   - Infoboxes are sparse in 2006, so category links give 4,846 seeds: `1905 births` and `1982 deaths` (birth and death year), `Cities in Y` (located_in), `Natives of Z` (birth place), `American novelists` (nationality and occupation), and `Novels by W` or `Films directed by V` (author, director).
   - Values are cleaned of links, refs and templates. Dates are parsed to year, month and day.
2. **Distant supervision.**
   - For each seed, the sentences of the subject's own article are searched (markup cleaned; templates, tables, lists and headings dropped). The subject can appear as the title, the bold lead form, the surname of a person page, or a sentence-initial he/she/it/his/her/its. A pronoun is written `[P]` in the pattern, so it is learned apart from `[S]`.
   - The object is a link, a capitalised chunk, a date, or an n-gram from the relation's seed vocabulary. It must equal the seed object after normalisation and pass the relation's type check.
   - The pattern is the text between the two mentions (1 to 6 tokens, digits written `#`, month names `<M>`), with slot order. There are four variants: no context, one token right, one token left, both. Examples: `[S] was born in [O]`, `^ [S] ( born [O] )`, `[S] ( <M> # , # - [O] )`.
   - A pattern is a candidate when at least 2 training subjects show it.
3. **Reliability.**
   - Subjects are split by hash into 80% training and 20% held-out. The candidate patterns are applied to the held-out subjects' articles.
   - An extraction is a check when the subject has a seed for that relation. It is correct when it equals a seed object. A place also counts when a `located_in` seed puts it inside a coarser seed place (Detroit inside Michigan), because the `Natives of` categories name a state about as often as a town.
   - A pattern is kept with at least 20 checks and a Wilson lower bound of at least 0.7. Its reliability is the mean of Beta(1 + correct, 1 + wrong).
   - Type constraints:
     - a date must parse, with year at most 2006;
     - capital, largest_city and country need a link to a place-like page;
     - person relations need a person-like page or 2 to 4 capitalised words;
     - lexicon relations need a seed value;
     - birth and death relations need a person page;
     - capital, language and currency need a country or state page.
   - "Place-like" and "person-like" come from the page's template, its categories (`... births`, `Cities in ...`) and the noun after "is/was" in the lead sentence.
4. **Extraction and ECS.**
   - Kept patterns are applied to all 8,587 articles. Facts that are not seeds get one evidence item per (sentence, pattern family), so the four variants of one match count once.
   - Mass is `omega * tau * pi`. `pi` is the pattern reliability. `tau` is 20/21, the plan's infobox prior, used for Wikipedia prose because the plan has no prior for prose. `omega` is `(1 + ln n)/n` for repeats of the same sentence text.
   - Other extracted values and conflicting seeds for the same (subject, relation) oppose the fact with weight `phi`, the relation's functionality. `phi` is hand-set (0.95 for capital and birth place, 1.0 for dates, 0.3 for author) and should be estimated from Wikidata later. A lifespan that is negative or over 125 years is a hard violation.
   - `ecs::fuse` gives the opinion and the grade.
   - The base rate `a` is the pooled held-out precision of the relation's kept patterns, clamped to 0.5 to 0.95. See the note under "Failure modes".
5. **Run modes.**
   - The default is one 80/20 split, as specified.
   - `--cv` repeats it over five folds and pools the checks. Fold 0 is the 80/20 split. The delivered facts come from `--cv`, because a single split gives fewer than 20 checks to almost every pattern.

## Seeds per relation

| Relation | Infobox / template | Table | Category | Total |
|---|---|---|---|---|
| capital | 83 | 6 | 0 | 89 |
| largest_city | 63 | 0 | 0 | 63 |
| country | 100 | 1 | 0 | 101 |
| location | 309 | 0 | 0 | 309 |
| located_in | 0 | 0 | 329 | 329 |
| birth_place | 120 | 0 | 331 | 451 |
| death_place | 88 | 0 | 0 | 88 |
| birth_date | 129 | 0 | 0 | 129 |
| death_date | 91 | 0 | 0 | 91 |
| birth_year | 0 | 0 | 1,144 | 1,144 |
| death_year | 0 | 0 | 921 | 921 |
| founded | 113 | 2 | 0 | 115 |
| author | 101 | 0 | 2 | 103 |
| director | 59 | 0 | 27 | 86 |
| composer | 26 | 0 | 8 | 34 |
| creator | 86 | 0 | 0 | 86 |
| spouse | 26 | 0 | 0 | 26 |
| predecessor | 197 | 0 | 0 | 197 |
| successor | 209 | 0 | 0 | 209 |
| developer | 59 | 0 | 0 | 59 |
| publisher | 474 | 0 | 0 | 474 |
| owner | 25 | 0 | 0 | 25 |
| genre | 147 | 0 | 0 | 147 |
| language | 168 | 3 | 0 | 171 |
| currency | 47 | 4 | 0 | 51 |
| occupation | 39 | 0 | 1,042 | 1,081 |
| nationality | 21 | 0 | 1,042 | 1,063 |
| party | 98 | 0 | 0 | 98 |
| family | 117 | 0 | 0 | 117 |
| order | 133 | 0 | 0 | 133 |
| class | 158 | 0 | 0 | 158 |
| **Total** | **3,286** | **16** | **4,846** | **8,148** |

`founder` is defined but has no seeds.

## Patterns

Counts are for the `--cv` run. "Checks at least 20" counts pattern variants. Kept patterns include the four variants of each match.

| Relation | Candidates | Checks at least 20 | Kept |
|---|---|---|---|
| birth_year | 164 | 17 | 14 |
| death_year | 208 | 27 | 16 |
| death_date | 30 | 6 | 4 |
| birth_date | 37 | 5 | 3 |
| birth_place | 41 | 7 | 3 |
| language | 21 | 4 | 4 |
| located_in | 34 | 1 | 0 |
| death_place | 18 | 1 | 0 |
| nationality | 40 | 3 | 0 |
| occupation | 10 | 1 | 0 |
| capital, largest_city, country, location, founded, director, spouse, successor, developer, genre, party, class | 73 in all | 0 | 0 |
| **Total** | **676** | | **44** |

The top kept patterns by Wilson lower bound (precision, checks, lower bound). `^` marks a sentence start.

| Relation | Pattern | Precision | Checks | Wilson lower |
|---|---|---|---|---|
| birth_year | `^ [S] ( [O]` | 0.993 | 457 | 0.981 |
| birth_year | `^ [S] ( [O] -` | 0.993 | 424 | 0.979 |
| birth_year | `^ [S] ( born [O]` | 1.000 | 154 | 0.976 |
| birth_year | `^ [S] ( born [O] )` | 1.000 | 99 | 0.963 |
| birth_year | `[S] ( [O] -` | 0.951 | 492 | 0.928 |
| death_year | `^ [S] ( <M> # , # - [O] )` | 0.994 | 311 | 0.977 |
| death_year | `[S] ( <M> # , # - [O] )` | 0.991 | 337 | 0.974 |
| death_year | `^ [S] ( <M> # , # - [O]` | 0.991 | 319 | 0.973 |
| death_year | `[S] ( <M> # , # - [O]` | 0.988 | 346 | 0.971 |
| death_year | `[S] ( <M> # # - [O]` | 1.000 | 35 | 0.901 |
| death_date | `^ [S] ( <M> # , # - [O]` | 1.000 | 40 | 0.912 |
| death_date | `^ [S] ( <M> # , # - [O] )` | 1.000 | 39 | 0.910 |
| death_date | `[S] ( <M> # , # - [O]` | 0.976 | 42 | 0.877 |
| death_date | `[S] ( <M> # , # - [O] )` | 0.976 | 41 | 0.874 |
| birth_date | `^ [S] ( [O]` | 1.000 | 56 | 0.936 |
| birth_date | `^ [S] ( [O] -` | 1.000 | 54 | 0.934 |
| birth_date | `[S] ( [O] -` | 0.846 | 65 | 0.739 |
| birth_place | `^ [S] was born in [O]` | 0.849 | 53 | 0.729 |
| birth_place | `^ [S] was born in [O] ,` | 0.846 | 39 | 0.703 |
| birth_place | `[S] was born in [O]` | 0.821 | 56 | 0.702 |
| language | `[S] ( [O]` | 0.968 | 31 | 0.838 |
| language | `[S] ( [O] :` | 0.964 | 28 | 0.823 |
| language | `the [S] ( [O]` | 0.950 | 20 | 0.764 |
| language | `the [S] ( [O] :` | 0.950 | 20 | 0.764 |

With a single 80/20 split only 7 birth_year and 4 death_year patterns pass. That run gives 77 raw new facts: 40 candidates and 17 admitted (13 birth year, 4 death year).

## Facts before and after (`--cv`)

"After" is seeds plus admitted facts (P at least 0.8). Candidates (0.5 to 0.8) are kept in the file but not counted.

| Relation | Before | Raw extractions | Candidates | Admitted | After |
|---|---|---|---|---|---|
| birth_date | 129 | 483 | 26 | 455 | 584 |
| death_date | 91 | 309 | 1 | 308 | 399 |
| birth_place | 451 | 55 | 18 | 37 | 488 |
| birth_year | 1,144 | 76 | 36 | 20 | 1,164 |
| death_year | 921 | 18 | 7 | 11 | 932 |
| language | 171 | 1 | 0 | 1 | 172 |
| all other relations | 5,241 | 0 | 0 | 0 | 5,241 |
| **Total** | **8,148** | **942** | **88** | **832** | **8,980** |

Grades: all 920 facts are Low. 919 have a single evidence sentence and one has two. Under the plan's rule (P at least 0.8 and grade at least Moderate) none would be admitted, because Moderate needs uncertainty at most 0.3, which means evidence mass of about 4.7. I followed the instruction (P at least 0.8 admits) and kept the grade in the file so the Q/A tool can filter.

## Precision of admitted facts

I drew 50 admitted facts with `std::mt19937` seed 20261027 and judged each from my own knowledge of the person. The sample has 30 birth dates, 16 death dates, 3 birth places and 1 birth year.

- 49 right. 1 unverifiable (Charles Robert Malden, death date 1855-05-23; the sentence supports it, but I do not know the person).
- Precision is 49/49 = 1.000, Wilson 95% [0.927, 1.000]. Counting the unverifiable fact as wrong gives 49/50 = 0.980, [0.895, 0.996].
- This clears the plan's 0.90 lower-bound target only if the unverifiable fact is excluded.

Two cautions:

- **The first draw found bugs.** It had two clear errors in 50: "Battle of Marathon" typed as a person (birth year 490 BC), and "Frederick William I of Prussia" matched through the surname "Prussia" (birth date 1713, the year his reign began). I fixed the lead-sentence typing and the surname rule, then re-ran. The numbers above are the second draw with the same seed, so it is a fresh sample, but the fixes were informed by looking at the first one.
- **The sample is 90% dates**, because 763 of the 832 admitted facts are birth or death dates. The 50 say little about the other relations. For these I checked all 37 admitted birth places by hand: 35 are right (95%, Wilson [0.823, 0.985]). Chris Morris is wrong (born in Colchester; the article says Cambridgeshire), and Gordon Brown is wrong (born in Giffnock; the article says Glasgow). Boston Corbett ("England", born in London) is right but coarse, and Angus McDuck (Glasgow) is right only inside the Duck fiction. The one language fact (Chad, Arabic) is right. This is not a random sample, so it is a supplementary check.

## Failure modes

- **Too few seeds.** The 2006 dump has 3,286 infobox seeds on 2,177 pages. Capital has 89, author 103, spouse 26. A relation needs enough seeded subjects that its patterns reach 20 checks on held-out pages, and these do not. Even five-fold pooling gives 0 kept patterns for capital, author, director, spouse, genre and currency. This is the main limit, and more extraction code will not fix it. Seeds will.
- **Coarse gold.** The `Natives of Michigan` category gives "Michigan" while the text says "Detroit". Exact matching counted these as errors until I added the `located_in` hierarchy and the comma-head match ("Philadelphia, Pennsylvania" against "Philadelphia"). The birth_place precision of 0.82 to 0.85 is mostly this plus a few wrong places in the articles.
- **Selection bias in the seed subjects.** The language pattern `[S] ( [O]` scores 0.97 on held-out country pages, because countries are written "Chad (Arabic: ...)". On a city the same shape gives "Aachen (French Aix-la-Chapelle)", which is a name gloss and not an official language. I found it by reading extractions. The fix was a subject-type gate (language, capital, currency need a country or state page), which cut the language extractions from 243 to 1. Any relation whose seeds come from one narrow class of page has this risk, and held-out precision will not show it.
- **Typing by heuristics.** Place and person types come from templates, categories and the noun after "is". They fail on lead sentences like "The Battle of Marathon ... was the culmination of King Darius". I now stop at the first preposition, but the heuristic is still crude.
- **Pronouns.** No `[P]` pattern passed the gate, so "He was born in Paris" sentences contribute nothing. Sentence-initial pronouns are only a rough stand-in for coreference.
- **ECS base rate.** With the plan default `a = 0.5` and `W = 2`, one evidence item cannot give P above 0.67, so no fact would be admitted. I set `a` per relation to the pooled held-out precision of its kept patterns. That partly counts pattern quality twice (once in `pi`, once in `a`), and the same held-out checks both select the patterns and set `a`. The 49/49 hand check is the independent test of that choice, and it passed on dates. With `--base-rate 0.5` (the plan default) the same run gives 845 facts at P 0.5 or more, a maximum P of 0.74, and none admitted.
- **One sentence per fact.** 919 of 920 facts rest on a single lead sentence, so the independence weight `(1 + ln n)/n` is almost never exercised, and no fact reaches Moderate.
- **Facts about the page subject only.** A sentence in the Algeria article that names the capital of Tunisia is not read.
- **Offsets.** `byte_offset` points to the start of the line, not the sentence.

## Next steps

1. **More seeds.** Add Wikidata seeds (D track) for capital, author, director, spouse, inventor and teacher. These are the relations the T-QA questions use, and 25 to 150 seeds is too few for any of them.
2. **Bootstrap.** Feed the admitted facts back as seeds for a second round (DIPRE style), with the Beta reliability and type checks as the drift guard. Date patterns are precise enough to start from.
3. **Read definitions.** The lead sentence "X is a novel by Y" and "X invented Y" gives author, inventor and occupation without an infobox.
4. **Cross-article evidence.** Link a mention in article A to the page of B, so that a sentence about B's capital in A counts. This is also what makes repeated, independent evidence possible.
5. **Better subjects.** Replace the sentence-initial pronoun rule with real coreference from E2/E3, and replace the lead-noun typing with the E5 entity linker.
6. **Pattern shape.** Use POS and chunk classes in the gaps (E1/E2) so that "was born at Guildford in Surrey" and "was born in Surrey" share a pattern.
7. **Calibration.** Fit the isotonic map of PLAN 7.5 on a 100-fact audit that is not drawn from the held-out checks, and estimate `phi` and `a` from Wikidata.
8. **Frozen test.** Load `facts_f2.tsv` into the Q/A tool and score `t_qa_wiki.tsv` once, with the full pipeline. This tool has not seen it.

---

## F2b: capital, author and director (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · closes F2's open item in the bill of work · follows [D4](D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md) ("Why capital, author and director still fail") and S11a's G11 ([S11A](S11A_CONCEPTS_AND_OPEN_RELATIONS.md))

(The design, the arms, the reads, the rules and the predictions below were written and committed before any F2b run. Results are added under them.)

**Question.** D4 gave capital, author and director enough Wikidata seeds, but no kept pattern: a subject's own article rarely states these relations, and when it does, the wording varies. Two fixes were named: reading other articles (cross-article evidence) and generalising the gap between the two mentions (pattern classes). Does either give these relations patterns that pass F2's gate, with facts that agree with Wikidata?

**Rule 10.** Patterns read text into facts for Cypha's store. Nothing here answers a question.

**What was looked at before this was written.** Counts of a few phrasings in all of enwik8 (no seed, label or gold value read): "capital of [[" 96, "capital, [[" 145, "is the capital of" 22, "novel by [[" 63, "written by [[" 223, "directed by [[" 242, "attended [[" 77, "educated at [[" 37. And S11a's G11 dev proposals (`/root/work/s11a/runs/dev.out`): own-article phrasings such as "[S] was directed by [O]" (10 pairs), "[S] is a novel by [O]" (5), "[S] was founded by [O]" (8), "[S] attended [O]" and "[S] was educated at [O]" (educated_at, which is not one of `pattern_learn`'s relations).

### The changes (flags; without them every output is byte-identical)

1. **Cross-article reading, `--f2b-xa`.** Every sentence of every article is read, not only the subject's own article. The fact's subject can be a **link** in the sentence: a link to an article (resolved through enwik8's redirects) whose page has seeds. Its slot is written `[L]`, so these patterns are learned apart from F2's `[S]` and `[P]`. The object is any candidate F2 already reads (link, capitalised chunk, date, lexicon string) or a mention of the page's own subject (the city's article: "[O] is the capital of [L]"). Type checks are F2's, with the subject's type taken from the linked page (a capital's subject must be a country-like page). The same 80/20 folds by subject apply, keyed by the linked page's title. A pattern's check counts once per (pattern, subject, object) in a fold, wherever it is stated, so a fact repeated in many articles does not inflate its pattern's checks. At extraction, every article can give a fact about a linked subject; each article's sentence is a separate evidence item.
2. **Wildcard gaps, `--f2b-wild 1|2`.** Beside each exact pattern, a generalised one: the gap's tokens are kept at its ends and the middle becomes `<*>` (one or more tokens). Mode 1 keeps the first gap token and the last two ("is <*> novel by": "is a novel by", "is a 1957 novel by", "is a science fiction novel by"); mode 2 keeps only the last two ("<*> novel by"). The gap limit stays 6 tokens. One pair of mentions gives one evidence item, however many of its patterns match.
3. **Word classes in gaps, `--f2b-classes FILE`.** Beside each exact pattern, one where every gap word that is not a function word (F2's stop-word list) becomes its k-means class (`classes_k256.tsv`: 256 classes over S-1b's 50,000 word vectors, [S1B](S1B_LEARNED_CONTEXTS.md)); digits and month names stay `#` and `<M>`.

F2's gate is unchanged: at least 20 checks and a Wilson lower bound of at least 0.7.

### Arms (all with `--cv --wikidata`, D4's published setting)

| Arm | Flags |
|---|---|
| A0 | none (D4) |
| A1 | `--f2b-xa` |
| A2 | `--f2b-wild 1` |
| A3 | `--f2b-xa --f2b-wild 1` |
| A4 | `--f2b-xa --f2b-wild 2` |
| A5 | `--f2b-xa --f2b-classes classes_k256.tsv` |

### The reads

- **Gold.** D4's gold fifth: subjects whose Wikidata values are never seeds, only scores. It is split in two by another hash bit (`(fnv(subject) >> 50) & 1`): **gold-dev** (0) for the choice, **gold-test** (1) read once, for the chosen arm and A0, with `--f2b-read-test`.
- **Measures, per relation and pooled:** kept patterns; admitted facts (P ≥ 0.8); agreement of admitted facts with Wikidata on gold subjects (right / checked, Wilson 95%), as D4's `score_gold`; the share of admitted facts with two or more evidence sentences, and with grade Moderate or better.
- **Hand check:** 30 admitted facts of the chosen arm in capital, author and director (fixed seed 20261029; all of them if fewer), judged from my own knowledge, after the test read.
- **G11 cross-reference (report only):** which of S11a's G11 dev phrasings map onto a relation `pattern_learn` already has, and whether the chosen arm keeps a pattern for it. Naming a new relation (educated_at) is Odin's (PLAN P10).

### Decision rules

1. **Dev choice.** Among A1–A5, the arms whose pooled gold-dev agreement has a Wilson lower bound of at least 0.85 and a point estimate no more than 0.03 below A0's are eligible. The eligible arm with the most admitted facts in capital, author and director together is chosen; ties go to fewer kept patterns. If none is eligible, F2b recommends nothing and the test read is A1's.
2. **Per relation (test).** F2's open item is closed for a relation if the chosen arm keeps at least one pattern for it and its admitted facts on gold-test agree with Wilson lower bound ≥ 0.7 on at least 10 checked facts. With fewer than 10 checked, it is "kept, too few gold facts to grade", and the hand check decides (at least 8 of 10 right).
3. **Recommendation for the default.** The chosen arm is recommended as `pattern_learn`'s default if its pooled gold-test agreement has a Wilson lower bound of at least 0.85 and is no more than 0.03 below A0's on the same half. It is **not switched here**: S3's gap map and every store built on it (S7–S11) are frozen on the default run. The switch is a decision for Odin, with the downstream stores rebuilt.

### Predictions

| # | Prediction |
|---|---|
| F1 | A0 reproduces D4: 70 kept patterns, 1,216 admitted facts, gold 173/182 |
| F2 | Cross-article reading (A1) keeps at least one pattern for at least two of capital, author and director |
| F3 | Wildcards alone (A2) keep no capital pattern, and add at most 5 kept patterns in all |
| F4 | The chosen arm admits at least 100 facts in capital, author and director together |
| F5 | The chosen arm's pooled gold-test agreement has a Wilson lower bound of at least 0.85 |
| F6 | With cross-article reading, at least 10% of admitted facts have two or more evidence sentences, and some reach grade Moderate (F2 and D4: 1 of 920) |
| F7 | The chosen arm runs in at most 3× A0's time and peaks under 300 MB |
| F8 | Hand check: at least 26 of 30 right |

### Results (run 2026-10-29)

The machine was shared with about twenty other jobs (load 27–36 on 4 cores), so every run took about 10 to 30 times the 21 s of an idle one. Outputs are in `/root/work/f2b/runs` (`dev/`, `test/`).

**Parity.** Without the new flags, `pattern_learn` is byte-identical to main's (ae631f5) on S3's run: 10 of 10 output files, and stdout apart from timings; the gap map files equal S3's reference (`a77678d8ef7dfdef`). `pattern_learn --f2b-selftest` passes 16 checks, and five planted bugs each make it fail: a wildcard keeping one end token instead of two, classes applied to stop words, the slots swapped, titles not capitalised, and redirects not followed. `pattern_learn.cpp` is not a CMake target, so no CTest compiles the changed file.

#### Dev (gold-dev half only)

| Arm | Kept (all) | Kept cap/auth/dir | Admitted (all) | Admitted cap/auth/dir | Gold-dev right/checked | Wilson 95% | >=2 evidence | Moderate+ | Time (s) | Peak MB |
|---|---:|---|---:|---|---|---|---:|---:|---:|---:|
| A0 | 70 | 0/0/0 | 1,216 | 0/0/0 | 77/81 = 0.951 | [0.880, 0.981] | 16 (0.013) | 0 | 581 | 121 |
| A1 | 74 | 0/0/0 | 1,243 | 0/0/0 | 77/81 = 0.951 | [0.880, 0.981] | 16 (0.013) | 0 | 1129 | 123 |
| A2 | 83 | 0/0/0 | 1,256 | 0/0/0 | 80/84 = 0.952 | [0.884, 0.981] | 18 (0.014) | 0 | 600 | 122 |
| A3 | 87 | 0/0/0 | 1,283 | 0/0/0 | 80/84 = 0.952 | [0.884, 0.981] | 18 (0.014) | 0 | 1153 | 125 |
| A4 | 104 | 0/0/0 | 1,385 | 0/0/0 | 86/93 = 0.925 | [0.853, 0.963] | 38 (0.027) | 0 | 1159 | 126 |
| A5 | 108 | 0/0/0 | 1,244 | 0/0/0 | 77/81 = 0.951 | [0.880, 0.981] | 17 (0.014) | 0 | 443 | 132 |

- **Capital, author and director: no arm keeps a pattern or admits a fact.**
  - **Author** comes closest. Cross-article reading finds "[O] ' s [L]" ("Tolkien's [[The Hobbit]]"): 16 of 18 checks right (0.889), Wilson lower bound 0.672. It is two checks short of the gate.
  - **Director:** the best is the wildcard "[P] <*> directed by [O]" (A4), 7 of 10.
  - **Capital:** cross-article candidates such as "in [O] , [L]" ("in [[Luanda]], [[Angola]]") state where a city is, not a capital: 0.10–0.23. "[O] is the capital of [L]" never reaches two training subjects. A capital's lead sentence usually puts a parenthesis between the city and the country ("Brussels (… ''Brüssel'') is the capital of [[Belgium]]"), so the gap exceeds 6 tokens, and only about six short "X is the capital of [[Y]]" leads exist in enwik8.
- **What the changes do add is dates.**
  - **A1:** 4 kept patterns, "[L] ( # - [O]", read people's years from other articles ("the British naturalist Charles Darwin (1809 - 1882)").
  - **A2:** wildcards keep 13 more date variants ("[S] ( <*> # - [O] )").
  - **A4:** most facts, +169, at gold-dev 0.925.
  - **A5:** word classes add 38 kept patterns and 28 facts.
- **Evidence.** Cross-article reading leaves almost every fact on one sentence: 1.3–2.7% have two or more evidence items, and none reaches grade Moderate.
- **The dev choice (rule 1).** Every arm is eligible (Wilson lower bound ≥ 0.85, point within 0.03 of A0's 0.951). Every arm admits 0 facts in the three relations, so the rule's tie-break (fewest kept patterns) chooses **A1**. The rule's main criterion was empty for every arm.

#### Test (gold-test half, read once)

| Arm | Gold-test right/checked | Wilson 95% |
|---|---|---|
| A0 | 96/101 = 0.950 | [0.889, 0.979] |
| A1 | 97/102 = 0.951 | [0.890, 0.979] |

A0's two halves together are 173/182, D4's number.

#### Decisions (by the registered rules)

- **Rule 2: F2's open item is not closed.** Capital, author and director have no kept pattern in the chosen arm or any other. The registered hand check of their facts has nothing to check.
- **Rule 3: A1 meets the recommendation rule.** Its gold-test lower bound is 0.890, and its point estimate is within 0.03 of A0's.
- **But I do not recommend switching yet.** A supplementary hand check (not registered) of A1's 27 new facts finds **19 right and 8 wrong**.
  - **The 8:** every one is a country's period read as a death date ("Belgium (1940 - 1945)", France 1944, Italy 1943, the Commonwealth of England 1653, each as death date and death year).
  - **Why they pass:** D4's Wikidata "founder" seeds widen the person lexicon with country names. The European Space Agency, the European Coal and Steel Community and Benelux list Belgium, France and Italy as founders. So these countries pass the person-subject check.
  - **Why gold misses them:** Wikidata has no death date for a country, so the gold check never sees these facts.
  - **The 19 right:** Darwin 1882, Celsius 1744, Fahrenheit 1736, Galileo 1642, Davy 1829, Capra 1991, and others.
  - The switch was never made here, because S3's store and everything built on it are frozen. It is a decision for Odin, after the person-lexicon leak is fixed.

#### G11's phrasings (report only)

| G11 dev phrasing | Relation in `pattern_learn` | Best F2 candidate (A1 = A4) | Kept |
|---|---|---|---|
| "[S] was directed by [O]" | director | "[P] was directed by [O]" 6/9 | no |
| "[S] is a novel by [O]" | author | 3/4 | no |
| "[S] was born at [O]" | birth_place | 8/8 (lower 0.676) | no |
| "[S] is a city located in [O]" | located_in | 11/11 (lower 0.741) | no (11 checks) |
| "[S] is the county seat of [O]" | located_in | 8/8 | no |
| "[S] was founded in [O]" | founded | 14/16 | no |
| "[S] died in [O]" | death_place | "[P] died in [O]" 28/32 | yes (D4's) |
| "[S] attended [O]", "[S] was educated at [O]" | none (educated_at is not a relation) | — | Odin's (P10) |

The text states these relations correctly but rarely. F2's 20-check gate is what stops them, not precision.

#### Why the item stays open, and what would close it

- **The limit is the text, not the patterns.** Each relation needs 20 held-out statements, in one wording, about subjects that have seeds. Cross-article reading mostly found statements about books, films and countries that have no seeds, or found long ones.
- **Options (for Odin):**
  1. A larger text, such as the 1 TB corpus: checks grow with the text.
  2. A gate that pools a relation's pattern family (the core with its context variants and wildcards) before counting checks. That changes F2's gate.
  3. Fix the person-lexicon leak (a Wikidata seed object should widen a type lexicon only when Wikidata's own type of the object agrees).
  4. educated_at as a new relation (PLAN P10).

#### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| F1 | A0 reproduces D4 | **Hit:** 70 kept, 1,216 admitted, gold 173/182 |
| F2 | A1 keeps a pattern for two of capital, author and director | **Miss:** none for any |
| F3 | A2 keeps no capital pattern and adds at most 5 kept patterns | **Partial:** no capital pattern, but 13 more kept (date variants) |
| F4 | The chosen arm admits at least 100 facts in the three relations | **Miss:** 0 |
| F5 | The chosen arm's gold-test Wilson lower bound ≥ 0.85 | **Hit:** 0.890 |
| F6 | ≥ 10% of admitted facts with two or more evidence items, some Moderate | **Miss:** 1.3%, none Moderate |
| F7 | At most 3× A0's time, under 300 MB | **Hit:** A1 1.9× (dev) and 2.6× (test), 124 MB |
| F8 | Hand check ≥ 26 of 30 | **Not scorable:** no facts in the three relations (supplementary: 19 of 27 cross-article facts right) |

3 hits, 1 partial, 3 misses, 1 not scorable.

#### Reproduce

```bash
g++ -O2 -std=c++20 -I native/include -pthread native/tools/pattern_learn.cpp -o pattern_learn
E=/root/cdata/enwik8; W=/root/work/f2/wikidata/wd_values.tsv; C=/root/work/sem/classes_k256.tsv
./pattern_learn --f2b-selftest
./pattern_learn $E A0 --cv --wikidata $W --f2b-gold-split                      # dev half only
./pattern_learn $E A1 --cv --wikidata $W --f2b-gold-split --f2b-xa
./pattern_learn $E A2 --cv --wikidata $W --f2b-gold-split --f2b-wild 1
./pattern_learn $E A3 --cv --wikidata $W --f2b-gold-split --f2b-xa --f2b-wild 1
./pattern_learn $E A4 --cv --wikidata $W --f2b-gold-split --f2b-xa --f2b-wild 2
./pattern_learn $E A5 --cv --wikidata $W --f2b-gold-split --f2b-xa --f2b-classes $C
./pattern_learn $E T0 --cv --wikidata $W --f2b-read-test                       # the test read, once (A0 and A1)
./pattern_learn $E T1 --cv --wikidata $W --f2b-read-test --f2b-xa
```

## F2c: the lexicon leak (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · follows F2b's supplementary check · decided by measurement (PLAN ledger 55), open to Odin's veto

(The definitions, the reads, the rule and the predictions below were written and committed before any fetch of types and before any run of the changed code. Results are added under them.)

**Question.** F2b found that D4's Wikidata seeds widen the type lexicons with values of the wrong type. The "founder" values of the European Space Agency, the ECSC and Benelux include Belgium, France and Italy. Founder is a person relation, so `pass 1` adds those country names to the person lexicon. A1's cross-article reading then took a country's period as a death date ("Belgium (1940 - 1945)"): 8 of A1's 27 new facts. Wikidata's gold check misses them, because a country has no death date. Should a Wikidata seed value widen a type lexicon only when Wikidata's own type for it agrees?

**Rule 10.** The lexicons read text into facts for Cypha's own store. Nothing here answers a question.

### The fix (behind a flag)

- **`--lexicon-type-check FILE`.** `FILE` is a table of Wikidata types for the seed values: item, its instance-of classes (P31), and whether it has a coordinate location (P625). It is fetched once from Wikidata's SPARQL endpoint for the person-relation and place-relation values in `wd_values.tsv`, and cached.
- **What it changes.** Only how Wikidata seeds (source 3) widen the lexicons in pass 1. Text seeds widen them as before. The seeds stay seeds: they still choose and check patterns.
  - **Arm P:** a Wikidata value of a person relation (author, director, composer, founder, creator, spouse, predecessor, successor, inventor) enters the person lexicon only if Wikidata says it is an instance of human (Q5).
  - **Arm PL:** arm P, and a Wikidata value of a place relation (capital, largest_city, country, location, located_in, birth_place, death_place) enters the place lexicon only if it has a coordinate location (P625).
  - A value with no row in the table counts as disagreeing (it does not widen).
- **Every copy of pass 1** (the main run and the modes that rebuild its lexicons: S7, S8a and the others) uses the same test, so the store stays one store.
- **Without the flag** every output is byte-identical to now.

### The reads

- **Dev** (choices): F2b's gold-dev half (`--f2b-gold-split`), with A0 (D4's run) and A1 (F2b's cross-article arm), each without the flag and with arms P and PL. Read: the lexicon entries each arm removes, by lexicon; admitted facts (P ≥ 0.8); the store's Wikidata agreement on gold-dev, right / checked with Wilson 95%; gold-dev facts Wikidata confirms ("right") that are lost; for A1, whether the 8 wrong country-period facts are gone and the 19 right ones kept.
- **S3's gap map** with the chosen arm: which gap kinds and entities move.
- **Gold-test** (the decision, read once): the chosen arm against no flag, with A0's settings, on F2b's gold-test half (`--f2b-read-test`).

### Decision rule (fixed before any run)

1. **Arm choice on dev.** PL replaces P only if it passes the rule below on dev and removes more misfiled place entries without losing any gold-dev "right" fact that P keeps. Otherwise P.
2. **Adopt as the default** if, with A0's settings: the leaked entries are removed (no Wikidata person-relation value that Wikidata types as a country enters the person lexicon); the store's Wikidata agreement does not fall (gold point estimate not lower than without the flag); and admitted facts that Wikidata confirms fall by at most 1%. Dev chooses; gold-test decides, read once.
3. If adopted, S3's reference outputs change. The report gives the new gap-map hash and lists every reference that must be regenerated. Other agents' references are not regenerated here.
4. Otherwise the flag stays and the default is unchanged.

### Predictions (registered before any run)

| # | Prediction |
|---|---|
| L1 | 10–35% of the person-relation Wikidata values are not humans (organisations, countries, bands, fictional characters) |
| L2 | Arm P removes 150–700 strings from the person lexicon that no page flag puts there |
| L3 | With A0's settings, admitted facts change by under 1% and gold-dev agreement moves by under 0.01 |
| L4 | With A1's settings, all 8 country-period facts are gone and at least 17 of the 19 right facts stay |
| L5 | S3's gap map changes (its hash differs), mostly in person-entity schema gaps |
| L6 | Arm PL removes 2–15% of the place-relation values from the place lexicon, and changes admitted facts by under 1% |
| L7 | The rule adopts arm P |

### F2c results on dev (2026-10-29)

**The types.** `seed_value_types.tsv`: P31 classes and P625 for the 4,592 values of person and place relations in `wd_values.tsv` (2,278 person-relation values, 2,328 place-relation values, 14 in both). Fetched once from `https://query.wikidata.org/sparql` on 2026-10-29, in ten batches of 500 at one a minute (the endpoint was limiting to one request a minute during an outage). Every item has a row.

- **Person-relation values that are not human:** 186 of 2,278 (0.082). By class (approximate): 87 mythological, biblical or fictional characters (the Japanese imperial consorts of legend, Zeus, Hephaestus, Fergus McDuck), 23 organisations (Sony, Philips, Cisco, Microsoft), 14 countries or states (Belgium, France, Italy, Spain, Sweden, the United Kingdom, West Germany), 62 other (bands, laboratories, legislatures, "group of authors").
- **Place-relation values without a coordinate location:** 76 of 2,328 (0.033), mostly historical states and palaces.

**The lexicons (pass 1, A0's settings).**

| Arm | Person lexicon | Place lexicon | Values kept out |
|---|---:|---:|---|
| Off | 3,429 | 4,137 | — |
| P | 3,283 | 4,137 | 147 person values; 146 of them no other source puts in the person lexicon ("david" stays, from a page) |
| PL | 3,283 | 4,046 | as P, and 60 place values (56 not in the place lexicon from another source) |

- **The leaked countries are out:** Belgium, France, Italy, Denmark, Spain, Sweden, Switzerland, Luxembourg, the Netherlands, the United Kingdom, the Republic of Ireland and West Germany no longer enter the person lexicon. No Wikidata person-relation value that Wikidata types as a country does.
- **PL's 56 place removals:** about 7 are not places (a journal, a city council, a government, "internationality", Esperantujo, the Yishuv, a dynasty). About 49 are places Wikidata gives no coordinates: historical states (the Mughal Empire, the Delhi Sultanate, the Kingdom of Poland, the Ptolemaic Kingdom), Japanese palaces of legend, fictional places (Arrakis, Skaro), Christiania (Oslo's old name).

**The store (gold-dev half, `--f2b-gold-split`).**

| Run | Admitted | Lost against off | Gold-dev Wikidata agreement | Admitted facts any Wikidata value confirms |
|---|---:|---|---|---:|
| A0 off | 1,216 | — | 77/81 = 0.951 [0.880, 0.981] | 1,005 |
| A0 P | 1,210 | 6, none checkable (Fergus McDuck ×5, Downy O'Drake ×1) | 77/81 = 0.951 [0.880, 0.981] | 1,005 (±0.00%) |
| A0 PL | 1,209 | the 6, and Ivar Aasen died in Christiania (confirmed) | 77/81 = 0.951 [0.880, 0.981] | 1,004 (−0.10%) |
| A1 off | 1,243 | — | 77/81 = 0.951 [0.880, 0.981] | 1,022 |
| A1 P | 1,229 | 14 (below) | 77/81 = 0.951 [0.880, 0.981] | 1,022 (±0.00%) |
| A1 PL | 1,228 | the 14 and Ivar Aasen's | 77/81 = 0.951 [0.880, 0.981] | 1,021 (−0.10%) |

("Any Wikidata value confirms" uses every row of `wd_values.tsv`, seeds included, matched on a surface form; reported, not the registered gold measure.) No arm adds a fact.

**A1's 27 cross-article facts (F2b's supplementary check).**
- **Gone with arm P (8):** Belgium 1945, France 1944, Italy 1943 (each as death date and death year): 6 of F2b's 8 wrong facts. And Downy O'Drake's death in 1897 (date and year): 2 of F2b's 19 right ones. Downy O'Drake is a fictional duck; Wikidata types her as a fictional character, and her 1897 death is the comics' own.
- **Kept (19):** 17 of the 19 right facts, and the Commonwealth of England's 1653 (date and year), 2 of the 8 wrong ones. The Commonwealth enters the person lexicon from a **text** seed, not Wikidata: Charles I of England's succession box names it as his successor. This fix leaves text seeds as they are (registered design), so that part of the leak stays.
- Fergus McDuck's five facts (born 1835 in Glasgow, died 1902) go for the same reason as Downy O'Drake's: fictional characters are not human in Wikidata.

**S3's gap map (the full run, `--gaps`).**

| Arm | Gaps | Entities by type | Gap map hash |
|---|---:|---|---|
| Off | 27,842 | person 1,329, place 568, other 6,359 | `a77678d8ef7dfdef` |
| P | 27,760 | person 1,320, place 570, other 6,366 | `7e71b1eae898ffb9` |
| PL | 27,752 | person 1,320, place 568, other 6,368 | `ba221cb2c79cc1ee` |

- With PL, 73 gap rows go (69 schema, 4 certainty; on 68 person entities and 5 place entities: France, Italy, Ezekiel, God, Hephaestus, the Free Software Foundation, the Futurians, Fergus McDuck, Christiania, Esperantujo) and 21 come (schema gaps of the same entities, now typed other or place). The demand file is unchanged. `facts_f2.tsv`, `patterns.tsv` and the cells file change with the store; `seeds.tsv` does not.

**Rule 1 (the arm).** PL removes more misfiled place entries than P (about 7 against none) and loses no gold-dev fact that P keeps, so by the rule as registered **PL is chosen**. It also keeps about 49 real places without coordinates out of the place lexicon and loses one confirmed fact outside gold-dev (Ivar Aasen in Christiania). The rule did not weigh that; it is noted for Odin, and arm P stays available behind the same flag.

**Rule 2 on dev (A0's settings, PL).** The leaked entries are removed; gold-dev agreement is unchanged (0.951); admitted facts any Wikidata value confirms fall by 0.10% (≤ 1%). Dev passes. The gold-test read decides.

**Planted bugs** (`--f2b-selftest`, now 25 checks, 9 of them F2c's): text seeds gated too (1 check fails), any class counted as human (2 fail), arm P gating places (1 fails). Each fails it.

### F2c test read and decision (2026-10-29)

**Gold-test, read once** (`--f2b-read-test`, A0's settings, arm PL; the flag-off side is F2b's own read of the same half, reproduced byte for byte by the flag-off binary):

| Run | Admitted | Gold-test Wikidata agreement | Admitted facts any Wikidata value confirms |
|---|---:|---|---:|
| Off (F2b's T0) | 1,216 | 96/101 = 0.950 [0.889, 0.979] | 1,005 |
| PL | 1,209 | 96/101 = 0.950 [0.889, 0.979] | 1,004 (−0.10%) |

The store with PL is the same on both halves' runs (the split only changes which subjects are scored). Peak 122 MB, 78 s.

**Rule 2 holds.** The leaked entries are gone (no Wikidata person-relation value typed as a country enters the person lexicon); gold-test agreement is unchanged; confirmed facts fall by 0.10% (≤ 1%). **Arm PL is adopted as the default** (decided by measurement under PLAN ledger 55, open to Odin's veto).

- **How it is the default** (at integration it waited behind `--lexicon-type-check FILE` until the references were regenerated; F2c-b below turned it on): the table is committed (`data/f2c/wd_seed_types.tsv`, 98 KB, with its README and `scripts/f2c_fetch_seed_types.py`). `pattern_learn` loads it from the working directory (the repository root, as every run here does) unless `--no-lexicon-type-check` is given. `--lexicon-type-check FILE` names another table; `--lexicon-type-arm P` keeps places as they were.
- **What changes by default:** S3's gap map becomes `ba221cb2c79cc1ee` (27,752 gaps; was `a77678d8ef7dfdef`, 27,842). With `--no-lexicon-type-check` every output is byte-identical to before.
- **References to regenerate** (done in F2c-b, below). Every mode that rebuilds the type lexicons from Wikidata seeds runs pass 1 through the new check:
  - the main run: S3's `--gaps` reference (`/root/work/s4/runs/test/`: `gap_map.tsv`, its cells file, `facts_f2.tsv`, `facts_f2.evidence.tsv`, `facts_wd_check.tsv`, `patterns.tsv`, `sample_admitted.tsv`; `seeds.tsv`, `pages.tsv` and the demand file do not change), the v2 store if it is rebuilt, F2/F2b/D4's reads, S4's frontier and S4b/S4c's `--signals` runs (U reads the store's patterns);
  - `--test-kinds` (S7, S7b, S7c), `--gen-links` (S8b; S8d's `--fresh-confirm` and S10's `--fresh-test-half` pools), `--gen-rules` (S8a, the same pools), `--worlds` (S9), `--gen-concepts` (S11a), `--gen-anomaly` and `--gen-analogy` (S11c);
  - modes that only read those outputs (`--sources`, `--schedule`, `--score-planted`) change only through their inputs.
  - Any pass-1 copy added by a branch merged later must call `lexfix::widen_ok` too.
- **Not fixed here:** the Commonwealth of England's 1653 "death" stays. It enters the person lexicon from a text seed (Charles I's succession box), not Wikidata.

#### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| L1 | 10–35% of person-relation values not human | **Miss:** 8.2% (186 of 2,278) |
| L2 | Arm P removes 150–700 person strings | **Miss:** 146 |
| L3 | A0: admitted −1% or less, gold agreement within 0.01 | **Hit:** −0.49% (P), −0.58% (PL); agreement unchanged |
| L4 | A1: all 8 country-period facts gone, ≥ 17 of 19 right kept | **Partial:** 6 of 8 gone (the Commonwealth's 2 come from a text seed); 17 of 19 kept |
| L5 | S3's gap map changes, mostly person schema gaps | **Hit:** 69 of 73 removed rows are schema gaps, 68 on person entities |
| L6 | PL removes 2–15% of place values, admitted −1% or less | **Hit:** 60 of 2,328 values (2.6%) kept out; admitted −0.58% |
| L7 | The rule adopts arm P | **Miss:** it adopts PL (rule 1, as registered) |

3 hits, 1 partial, 3 misses.

#### Reproduce

```bash
g++ -O2 -std=c++20 -I native/include -pthread native/tools/pattern_learn.cpp -o pattern_learn
E=/root/cdata/enwik8; W=/root/work/f2/wikidata/wd_values.tsv; T=data/f2c/wd_seed_types.tsv
./pattern_learn --f2b-selftest                                                                   # 25 checks
./pattern_learn $E A0off --cv --wikidata $W --f2b-gold-split --no-lexicon-type-check              # dev, off
./pattern_learn $E A0P --cv --wikidata $W --f2b-gold-split --lexicon-type-check $T --lexicon-type-arm P
./pattern_learn $E A0PL --cv --wikidata $W --f2b-gold-split                                       # dev, PL (the default)
./pattern_learn $E A1PL --cv --wikidata $W --f2b-gold-split --f2b-xa                              # A1 with PL
./pattern_learn $E T0PL --cv --wikidata $W --f2b-read-test                                        # the test read (once)
```

## F2c-b: the type check on by default (2026-10-29)

**Odin Loch** · 2026-10-29 · PLAN ledger 81 (decided by measurement under row 55, open to Odin's veto)

At integration F2c's check stayed behind `--lexicon-type-check FILE`, off by default, until the reference outputs were regenerated. F2c-b turns it on and regenerates them.

### What changed

- `lexfix::take_flags` reads the check's flags before any mode runs. The check is on by default, with arm PL and the committed table `data/f2c/wd_seed_types.tsv`.
- The table is looked for in `$CYPHA_F2C_TYPES`, then beside the source file the tool was built from, then in the working directory (the repository root, as for `bench/eval`).
- `--lexicon-type-check FILE` still names another table. `--lexicon-type-arm P|PL` picks the arm. `--no-lexicon-type-check` gives the outputs from before, byte for byte.
- If no table is found, the first Wikidata seed that needs it stops the run. The message names the places searched. The check never turns itself off. Modes that read no Wikidata seed (for example `--wilson`) run without the table.
- Every pass-1 lexicon copy already widens through `lexfix::widen_ok`, so one switch governs them all:
  - the main run (S3, S4b, S4c, S4d, F2b);
  - `--test-kinds` (S7, S7b, S7c);
  - `--gen-rules` (S8a);
  - `--gen-links` (S8b), and through its corpus loader `--gen-anomaly`, `--gen-analogy` (S11c) and `--gen-deep` (S12a);
  - `--worlds` (S9, S9b);
  - `--gen-concepts` (S11a, S11d).
- `--s7d`, `--gen-laws`, `--sources`, `--schedule` and `--score-planted` build no lexicon. They read files, so they change only if their input files are rebuilt.

### Parity: `--no-lexicon-type-check` against main's tool (53f4387)

| Read | Result |
|---|---|
| S3 `--gaps` with S4b's signals | 13 of 14 files identical; `stats.txt` and stdout differ only in paths and timings. S3's 10 reference files identical (gap map `a77678d8ef7dfdef`) |
| `--gen-rules` (S8a, dev) | 8 of 8 files and stdout identical (timings masked) |
| `--gen-concepts` (S11a, dev, A1) | 2 of 2 files and stdout identical (timings masked) |

### The new reference

`/root/work/s3ref_f2c/`, the S3 `--gaps` command with the default on. Its gap map is `ba221cb2c79cc1ee`, the same as F2c's own PL run, and all 10 files and the blocked-values list equal that run's.

| File | sha256, first 16 | Against the old reference |
|---|---|---|
| `gap_map.tsv` | `ba221cb2c79cc1ee` | changed |
| `gap_map.tsv.cells.tsv` | `0ccb8de4d20671c7` | changed |
| `gap_map.tsv.demand.tsv` | `dd1c8ddac831dcb6` | same |
| `facts_f2.tsv` | `0c3df03ee3003ff6` | changed |
| `facts_f2.evidence.tsv` | `35ad3196e19b1c8b` | changed |
| `facts_wd_check.tsv` | `ee9ffc41e7d1033b` | changed |
| `pages.tsv` | `aaf63bba31828c93` | same |
| `patterns.tsv` | `5aa1de8a40616f64` | changed |
| `sample_admitted.tsv` | `f9478cfa5bd5ac78` | changed |
| `seeds.tsv` | `3e958326bf11ddb3` | same |

### What moved (dev reads only)

Main's tool against the new default, on each mode's dev command. No frozen test set was read again: a test read with the new default would be a new read, and none was registered.

| Read | What changed |
|---|---|
| S3 (main run) | Person pages 3,429 → 3,283, place pages 4,137 → 4,046 (147 person and 60 place values kept out). Raw new facts 1,435 → 1,428 (birth_place 573 → 571, death_place 118 → 117). Gaps 27,842 → 27,752 (schema 26,333 → 26,250; certainty 1,342 → 1,335). Gap calibration moves by 0.0001–0.0002 Brier. In the top 100 questions, contradiction 45 → 40, schema 15 → 20 |
| S4b signals | No alarm changes (`signal_gaps.tsv` identical). U: one more eligible sentence (28,660 → 28,661; the rate is unchanged at 0.9927) |
| S7 / S7b / S7c (`--test-kinds`, dev) | Choices unchanged: φ stays "store", S7c's hybrid keeps K4 for Known. T1 c 0.918 → 0.919 (one more false hypothesis T1 calls negative). S7c dev K4 promoted 572 → 570 (precision 0.976 → 0.975); K1 215 → 214 |
| S8a (`--gen-rules`, dev) | 7 fewer hypotheses selected, all ungraded (supported 143 → 136, promoted 105 → 98); every graded count and precision is unchanged. R1's T4 c unchanged; R2's T4 c 0.914 → 0.915 |
| S8b (`--gen-links`, dev) | One more hypothesis refuted (391 → 392, false 334 → 335, precision 0.971) |
| S9 (`--worlds --s9b`, dev) | One item: "Free software movement" founder "Luxembourg" is now refuted by T1, because the country is no longer in the person lexicon. T1 c 0.911 → 0.912 |
| S11a (`--gen-concepts`, dev) | G11: one "died in" pair fewer (11/19 → 11/18 holding on dev, 0.579 → 0.611) |
| S12a (`--gen-deep --fresh-dev`) | Unchanged (the prospective log differs only in its timestamp) |
| S11c (`--gen-anomaly`, dev) | Unchanged: 2 of 2 files and stdout identical (timings masked) |

Not run: `--gen-analogy`. Its 300-dimension vectors (`ppmi_svd_output/enwik8`) are no longer on this machine. It builds its lexicons through the same loader as `--gen-anomaly`.

### Checks

- `pattern_learn --f2b-selftest`: 32 checks, 0 failed (7 new for F2c-b: on by default, no table found, `--no-lexicon-type-check`, `--lexicon-type-arm P`, a named table, an unreadable named table, a text seed without a table).
- Four planted bugs each fail it: the default left off (4 checks fail), `--no-lexicon-type-check` ignored (1), a missing table not marked (1), a named table ignored (2).
- Fail closed: run from a directory without the table and without `$CYPHA_F2C_TYPES`, the S3 command stops with exit 2 and the message; `--wilson` still runs.
- `native_epistemic_test`: 1,478 checks, 0 failed. `pattern_learn` builds clean with `-Wall`.
