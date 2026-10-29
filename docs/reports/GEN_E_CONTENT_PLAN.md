# GEN-E: the content-plan expert (the hybrid step of the generation track)

**Odin Loch** · 2026-10-29 · PLAN §3.2 item 4 (patterns are bidirectional), §3.3 step 5 (realise the text), B9 (copy expert), B4c (experts off inside answers)

(The design, the reads, the rule and the predictions below were written and committed before any run of the expert. Results are added under them.)

**Question.** CyphaLM's samples are word salad by SCRIBE v2 (0.063 on dev; human 0.923, sentences shuffled across documents 0.344). Nothing steers what the text is about. Odin approved the hybrid step on 2026-10-29 ("That's ok to do what you suggested"): the fact store picks what to say about the prompt's entity, and a next-byte expert raises the odds of the words that express the next planned fact, using learned patterns reversed. Does that make the samples better text, and does it keep them honest?

**Rule 10.** CyphaLM writes every byte. The plan is one more next-byte expert in the mixture, with lambdas fit on held-out text like the others. It is on in `sample` only. `qa` and `chat` do not build it, so their answers stay byte-identical. Using it to answer questions would need its own ruling.

## What was looked at before this was written

While designing I read the store and pattern files and ran one coverage count on the dev prompts (A4d's `SO_dev.txt`, 250 prompts) with a prototype of the entity rule below. 30 of 250 prompts named an entity that has facts in the store. No sample was generated and no score was computed.

## The design (fixed before any run)

**The store.** Cypha's v2 fact store as `pattern_learn` wrote it (`--plan DIR`): `seeds.tsv` (infobox, table and category seeds read from the text, and the Wikidata seeds of D4, the data pack) and `facts_f2.tsv` (facts F2 admitted from the text). Nothing is fetched at run time.
- A fact whose evidence byte offset lies in the sample's holdout `[train, sample)` is dropped. The prompts come from the holdout, so a prompt's own page can never feed its plan.
- A date (`birth_date`, `death_date`, an ISO value) becomes its year relation (`birth_year`, `death_year`) with the year, leading zeros dropped; the same is done for `founded`.
- A relation is planned for an entity only if the entity has at most 2 distinct values for it (lists such as a language's countries are left out). The value used is the first from the text (F2, infobox, table, category), else the first from Wikidata, in file order.
- A relation is planned only if it has at least one frame (below).

**The entity.** Read from the prompt only, by the reader's own markup:
1. the bold lead (`'''X'''`) at the start;
2. then each link target (`[[X]]`, `[[X|y]]`), in order;
3. then each run of capitalised words ("of" allowed inside; "The" alone dropped), in order, and the same run without a leading "The".

The first candidate whose name (lowercased) is a subject in the store with at least one plannable fact is the entity. No search over the training text is made.

**The plan.** The entity's plannable facts, at most 6, in a fixed relation order taken from how the 2006 text states facts:
- relations with F2 facts, by the median position of their evidence inside its page (bytes from the page start, `pages.tsv`), earliest first;
- then the other relations by their number of seeds, most first.

**The frames (patterns reversed).** A3's `load_frames` over F2's `patterns.tsv` (rendered, at most 3 a relation, kept patterns first, then candidates by training subjects) and over F2R's `patterns_raw.tsv` (the raw bytes with their markup, at most 3 a relation). A frame is filled with the store's subject title and the value. Frames with slots CyphaLM cannot fill (`<M>`, `#`) are skipped, as in A3.

**The expert.** Like the copy expert, but its source is the plan's realisations instead of the text so far.
- **Active facts:** every planned fact not yet expressed. A fact is expressed once its value (markup dropped, lowercased) appears in the generated text. Then its realisations leave the expert.
- **Alignment:** for every realisation r and every position j, L is the longest common suffix of the text so far and r[0, j), at most 64 bytes; only L ≥ 3 counts. The expert takes the largest L; ties go to the earlier fact in the plan. It predicts r[j], spread evenly over that fact's realisations and positions that reach the same L.
- **Weights:** 8 buckets by L (the copy expert's buckets). The lambdas are fit on the first half of the holdout over the mixture with the experts before it (trie, similar-context, copy), as the other experts are. For that fit, each paragraph line of the holdout gets the plan of its own first 64 bytes, exactly as a prompt would. Bucket 0 (no alignment) predicts nothing.
- **Order:** mixed last, after the copy expert.

**Deviation from the brief, stated.** The brief said "the next planned fact". The expert here keeps every unexpressed planned fact active and uses the plan order only to break ties. With one fact at a time the plan stalls whenever the text never reaches that fact's frame.

## The reads

1. **Coverage.** Prompts with a plan of at least one fact, out of all; and the plans' relations.
2. **BPC.** The mixture's bits per byte on the second half of the holdout, without and with the expert (lambdas from the first half).
3. **Text (SCRIBE v2, surface S).** A4d's pipeline: `cyphalm_trie sample` on the dev prompts (`SO_dev.txt`'s 250, T = 0.6, 0.8, 1.0), scored by `scribe ladder` with A4d's settings. The baseline is main's default sample (A4d's `SO_dev.txt`; the sample code path has not changed since, which a 5-prompt rerun checks). Reported overall and on the covered prompts. Copy share is SCRIBE's "share D_I > 0".
4. **Factual precision.** For every covered prompt, in each generated text: a statement is a frame of a relation the entity has in the store, found with a value after it. Each statement either matches the store's value (consistent) or not (contradiction). Counted for the baseline and with the plan, on the same prompts.
5. **Test.** Once, on GEN-A's frozen benchmark if it is ready (`/root/work/gena/HARNESS_READY`); otherwise dev only, and the test read is left for after the merge.

## Rule (fixed before any run)

The expert becomes a `sample` default only if, on the test read:
- SCRIBE v2 rises with disjoint 95% intervals;
- the copy share does not rise by more than 0.02;
- contradictions with the store do not rise;
- surface S does not fall (its interval's upper end stays above the baseline's mean).

Otherwise it stays behind `--plan` and this report keeps the numbers.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| E1 | Coverage 8–20% of dev prompts |
| E2 | Lambdas: buckets with L ≥ 8 get at least 0.1; L below 6 at most 0.05 |
| E3 | BPC on the holdout's second half changes by less than 0.002 |
| E4 | SCRIBE v2 overall on dev moves by less than 0.01; on the covered prompts the intervals overlap |
| E5 | Statements matching the store rise; contradictions do not |
| E6 | Copy share and surface S change by less than 0.01 |
| E7 | The rule is not met: the expert stays behind `--plan` |

## Amendment before the test read (written after the dev reads, before the test read)

GEN-A's harness became ready during the dev reads, so the test read is made on its frozen benchmark (`bench/eval/gen_prompts_test.tsv`, `scripts/gen_eval.sh`). Three things follow from that. Nothing else changes.

1. **The prompts' own bytes.** The benchmark's prompts come from enwik8 [72 MiB, end), bytes [75,497,472, 100,000,000). The store covers all of enwik8, so it holds facts read from those pages. Facts whose evidence lies there are dropped too (`--plan-drop 75497472:100000000`). This is the holdout rule applied to the benchmark: a prompt's own page never feeds its plan. The relation order still uses all of F2's offsets; it is a count over thousands of facts, not a fact about any page.
2. **Continuation BPC.** The harness scores each human continuation under sampling's mixture. The plan expert joins that mixture there too, with the prompt's plan and the same "expressed" rule. So the test read also shows whether the plan predicts what the human wrote next.
3. **Copy share.** As registered, SCRIBE's "share D_I > 0". The harness's 32-byte copy share is reported beside it.

The baseline arm is the same binary without `--plan` (main's default `sample`, with GEN-A's `--prompts-file`). Both arms run once on the test split.

## Results: dev (A4d's pipeline, 250 prompts of `SO_dev.txt`)

Runs: `cyphalm_trie sample enwik8 GE_dev.txt --sim default,online=1,probe=128 --plan /root/work/f2/v2/aug,/root/work/f2raw/patterns_raw.tsv` (wip/gene 3044a7d; 4,357 s, peak 75 MB). Baseline: A4d's `SO_dev.txt`.

**Parity.** Without `--plan` the new binary writes A4d's `SO_dev.txt` byte for byte (MODEL line and the first 5 prompts, 3 texts each). With `--plan`, all 222 uncovered prompts are byte-identical to the baseline; all 84 texts of the 28 covered prompts changed. So every difference below comes from the covered prompts.

**1. Coverage.** 28 of 250 dev prompts have a plan: 0.112 [0.079, 0.157]. The pre-run prototype found 30. On the holdout's paragraph lines (the lambda fit) 131 of 1,046 have a plan: 0.125 [0.107, 0.147]. The store has 3,940 subjects; 26 relations have frames (117 frames). The 28 plans name 19 entities (Agamemnon 4 times, Aeneas 3). Their facts: country 10, birth_place 7, occupation 7, death_year 6, birth_year 6, spouse 6, located_in 6, death_place 5, founded 5, capital 4, language 4, nationality 2, largest_city 1.

**2. Lambdas and BPC.** Over the holdout lines with a plan the expert aligned 5,552 bytes and put more than half its mass on the right byte at 2,398 of them (0.432 [0.419, 0.445]). Lambdas by match length:

| L | none | 3 | 4–5 | 6–7 | 8–11 | 12–15 | 16–31 | 32+ |
|---|---|---|---|---|---|---|---|---|
| lambda | 0 | 0.025 | 0.11 | 0.375 | 0.285 | 0.355 | 0.13 | 0 |

Second-half holdout BPC: 1.8953 without the expert, 1.8950 with it (−0.0003).

**3. Text (SCRIBE v2, 95% bootstrap intervals).**

| Read | Baseline | With the plan |
|---|---|---|
| SCRIBE v2, all prompts (608 / 616 texts kept) | 0.063 [0.051, 0.078] | 0.066 [0.053, 0.080] |
| surface S, all | 0.860 [0.854, 0.867] | 0.860 [0.853, 0.866] |
| T 0.6 / 0.8 / 1.0 | 0.081 / 0.070 / 0.037 | 0.079 / 0.075 / 0.041 |
| SCRIBE v2, covered prompts (63 / 71 texts kept) | 0.050 [0.019, 0.089] | 0.070 [0.035, 0.111] |
| surface S, covered | 0.860 [0.837, 0.880] | 0.852 [0.834, 0.869] |
| copy share (D_I > 0), all | 0 of 608 | 1 of 616 |
| copy share, covered | 0 of 63 | 1 of 71 |

Paired by prompt (27 covered prompts with a kept text in both arms): SCRIBE +0.027 [−0.012, +0.070], surface S −0.005 [−0.030, +0.018]. More covered texts survive SCRIBE's cleaning with the plan (71 of 84, against 63).

**4. Factual precision** (`scripts/gen_e_facts.py`, written before any planned text was read). A statement is a frame connective of a planned relation followed by a value the model wrote. The connectives come from the plan's own frames; one that frames several relations goes to the relation with the most F2 training subjects for it. Values: a year for year relations, else A3's head rule.

| On the 28 covered prompts (84 texts) | Baseline | With the plan |
|---|---|---|
| statements | 1 | 4 |
| consistent with the store | 0 | 0 |
| contradicting the store | 1 | 4 |
| subject named in the 60 bytes before | 1 (contradicting) | 1 (contradicting) |
| planned values written anywhere in the text | 0 of 207 | 38 of 207, 0.184 [0.137, 0.242] |

Paired bootstrap over prompts: contradictions +3 [0, +7]. The new three are all `birth_place`: "was born in the United and is sometimes", "was born in China", "was born in [[Canada". The plan wanted Salamis (Euripides) and Cheshire (Lewis Carroll, twice). The expert carries the frame ("he was born in [[" aligns 15 bytes) at lambda 0.355, and the value's first letter too. From the 16th aligned byte the lambda drops to 0.13, so the rest of the value is weakly held: "C" from Cheshire went on as Canada and China. The baseline's one statement (Ahmed III "born December 19, 1975") is in both arms; the expert had not fired yet at that point.

The values that do get written: United Kingdom 10, United States 4, Algeria 3, Sovereign 4, English language 2, Europe 2, Thebes 2, military leader 2, and one each of Arabic, 944, 1054, 886, Tokyo (twice: capital and largest city), East of England, mythical character, Cheshire. They are mostly common names, placed loosely ("the [[Thebes capital of [[Edward Tarsus|Marsh]]").

**Samples (T 0.8; before, then after).**

- Prompt 247, "In this passage Lewis Carroll incorporated references to everyon"; plan death 1898, birth 1832, Cheshire, The Chestnuts, British.
  - Before: "e be declared in the rectal [[Hungary]] ==External ches links== *[http://www.albca.com/wcd/abs/35/35/ IMDb name|id=ISBN 0-394-39, 45 to his first death [[superman of [[Benito Medicinematics]], American actor (d. [[1924]])"
  - After: "e be declared on 18]] * [[Jules Prize in the [[United showing of and as [[Internamed [[Moses]] to and the sixth is or part [[dies in the first international chess for [[composed seven … *''[[The Chest and he was born in [[Cana"
- Prompt 177, "While Alcmene was pregnant with [[Heracles]] (&quot;glory of Her"; plan Thebes, mythical character.
  - Before: "… [[Hammunist years of life hand and the [[Maldives]] and [[Peter Crowley (d. [[1941]])"
  - After: "… [[Hammunist years of life hand and the [[Thebes capital of [[Edward Tarsus|Marsh]], [[1930s, Stephen]], ''Grander]] that the mythology]] of [[moral states"
- Prompt 103, "Joanna died in March, [[1238]] in Essex, and in the following ye"; plan United Kingdom, East of England, Chelmsford.
  - Before: "ars, Austrian came extreme to decademy the line of and reasons for the first to be the definitional shield steel …"
  - After: "ars, Austrian came extreme to decademy the line of Egypt|Constitutionally]], [[Jordan]], a city is the reaction … renovated connected Kingdom]] in the formation of …"

The topic words arrive ("Chest", "Thebes", "mythology", "Kingdom"), but inside the same broken grammar.

## Results: the test read (GEN-A's frozen benchmark, read once)

Commands (wip/gene ca4f4ba, from the repo root):

```
CYPHALM_TRIE=<wip/gene build> SCRIBE=/root/work/gena/runs/scribe_gena bash scripts/gen_eval.sh test E0 OUT
CYPHALM_TRIE=<wip/gene build> SCRIBE=/root/work/gena/runs/scribe_gena bash scripts/gen_eval.sh test E1 OUT \
    --plan /root/work/f2/v2/aug,/root/work/f2raw/patterns_raw.tsv --plan-drop 75497472:100000000
```

Then `scripts/gen_e_facts.py` and the same subset and paired SCRIBE reads as on dev.

**Parity.** The 212 uncovered prompts are byte-identical between the arms, both in the generated text and in the continuation bits. All 114 texts of the 38 covered prompts changed.

**Coverage.** 38 of 250 test prompts have a plan: 0.152 [0.113, 0.202]. The store keeps 3,816 subjects once the benchmark's bytes are dropped. The test prompts are clean prose, often about countries, so the mix differs from dev: language 19, capital 18, founded 18, birth_place 6, birth_year 6, country 6, largest_city 6, death_place 5, death_year 5, founder 5, located_in 4, and 16 more facts in 9 other relations (genre, author, occupation, creator, inventor, spouse, developer, director, nationality).

| Read (test, 250 prompts) | E0, baseline | E1, with the plan |
|---|---|---|
| SCRIBE v2, pooled (617 / 626 texts kept) | 0.072 [0.059, 0.085] | 0.071 [0.059, 0.084] |
| T 0.6 / 0.8 / 1.0 | 0.106 / 0.064 / 0.046 | 0.102 / 0.068 / 0.044 |
| surface S, pooled | 0.861 [0.855, 0.866] | 0.860 [0.854, 0.866] |
| copy share (D_I > 0) | 0 of 617 | 0 of 626 |
| harness copy share (32-byte spans) | 0.0386 | 0.0401 |
| continuation BPC | 2.1296 [2.0956, 2.1659] | 2.1266 [2.0923, 2.1629] |
| continuation BPC, paired E1 − E0 | | −0.0030 [−0.0052, −0.0013] |
| SCRIBE v2, covered (89 / 98 texts kept) | 0.080 [0.042, 0.125] | 0.073 [0.043, 0.107] |
| SCRIBE v2, covered, paired (37 prompts) | | −0.021 [−0.082, +0.034] |
| surface S, covered | 0.857 [0.839, 0.873] | 0.854 [0.839, 0.867] |
| continuation BPC, covered, paired (38 prompts) | 2.1185 | −0.0199 [−0.0328, −0.0092] |
| statements (consistent / contradicting) | 0 / 0 | 0 / 0 |
| planned values written anywhere | 3 of 342 | 36 of 342, 0.105 [0.077, 0.142] |

**What the test read says.**
- **The plan knows what comes next.** On the covered prompts the human continuation costs 0.020 fewer bits a byte with the plan expert in the mixture (about 1%), with an interval clear of zero. Over all 250 prompts that is −0.0030 [−0.0052, −0.0013].
- **It does not make better text.** SCRIBE v2 does not move, on all prompts or on the covered ones. The planned values arrive (36 of 342 against 3), but inside the same broken grammar.
- **No harm found on test:** no copying, no contradictions, surface S flat.

**Samples (test, T 0.8; before, then after).**
- Prompt 78, "Dexter killed Morse in his last book … unlike Arthur Conan Doyle, who killed his main character only to bring …"; plan death 1930, birth 1859.
  - Before: "In it the [[self Art a few thousand degrees the or infant's accident. On [[August 2]], [[1951]] that the same line as or it sometimestable is method organic of amines …"
  - After: "In it the [[self Art a few thousand degrees the or infant's accident. On [[July 11]] [[1930]], [[music]] ''[[Serenity]], a ship's train, the [[Fabia astronauts …"
- Prompt 0, "The military has dominated Algerian politics up to today. …"; plan Arabic, Algiers (capital, largest city).
  - Before: "In the summer and to consider the Communion is a [[German state, and a few weeks have was a series of the first reflection. …"
  - After: "In the Appalachian government of the saints term ''Yank'' is considered the side later. === Major called [[seal]]s, which have significant state …"

## Decision

The rule is not met on the test read: SCRIBE v2 does not rise (0.072 → 0.071, intervals overlap). The other three conditions hold (copy share 0 → 0, contradictions 0 → 0, surface S's upper end 0.866 above 0.861). **The expert stays behind `--plan`.** `sample`'s default does not change.

What it is good for: the plan expert lowers the bits of real continuations where it fires (−0.020 BPC on covered test prompts). It carries knowledge the trie lacks. The text side fails because a byte expert can only nudge single bytes. Where it fires, the model takes the frame and the first letter, then drifts (dev: "was born in China" for Cheshire). A plan needs a writer that can hold a phrase; that is the word-level track's job (`docs/TOKENS_PLAN.md`), or a sentence gate (GEN-B) that picks candidates which state the planned fact.

## Predictions scored

| # | Prediction | Result |
|---|---|---|
| E1 | Coverage 8–20% of dev prompts | **Held.** 11.2% [7.9, 15.7] (test 15.2%) |
| E2 | Lambdas: L ≥ 8 get at least 0.1; L below 6 at most 0.05 | **Failed.** L 4–5 got 0.11; L ≥ 32 got 0 (8–31: 0.13–0.355, as predicted) |
| E3 | Holdout BPC changes by less than 0.002 | **Held.** −0.0003 |
| E4 | Dev SCRIBE moves by less than 0.01; covered intervals overlap | **Held.** +0.003; covered 0.050 → 0.070, overlapping |
| E5 | Consistent statements rise; contradictions do not | **Failed.** Dev: consistent 0 → 0, contradictions 1 → 4 (+3 [0, +7]). Test: 0 → 0 |
| E6 | Copy share and surface S change by less than 0.01 | **Held on all prompts** (copy +0.002, S 0.000); on the covered dev prompts copy rose by one text (+0.014) |
| E7 | The rule is not met; the expert stays behind `--plan` | **Held** |

## Checks

- The 9 CTests of HANDOFF §4.4 pass, before the merge (3044a7d) and after it (the merge with GEN-A and GEN-C, Release build): 9 of 9 both times. `cyphalm_trie selftest` passes. Its GEN-E block checks the entity order, dates to years, the holdout and list exclusions, the relation order, the realisation, alignment and its tie rule, and the "expressed" test.
- **Answers unchanged.** `qa` with A4d's exact flags (`--sim default,online=1,probe=128`, the C1LD set) writes `QO.rows.tsv` byte for byte: 520 rows, chat and wiki forms. Only latency lines differ. `qa` and `chat` never build the plan; `sample_with_class` is called by `sample` and `cls` only.
- **Sample unchanged without the flag.** Without `--plan` the binary writes A4d's `SO_dev.txt` byte for byte (5-prompt rerun), and with it every uncovered prompt is byte-identical (dev 222 of 222, test 212 of 212).
- Cost: the dev arm took 4,357 s (peak 75 MB) under a load of 7–10 on 4 cores. The plan adds a second pass over the holdout's second half (the lambda fit reads the whole holdout, as `--doc` does).
