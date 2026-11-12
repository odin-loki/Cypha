# H2: a fill that does not invent numbers and dates

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/honest-dates` · GENERATION_PLAN §13.48 (H1b), §13.53 (L1), [GEN_FACTS](GEN_FACTS.md)

(The arms, the screen, the measures, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- Today's default (L1, §13.53) states wrong facts as often as right ones. Its fact precision against CyphaLM's own store is 0.437 [0.369, 0.507] (GEN_FACTS v2).
- Most errors are dates. In W1b's file (the same fact sentences as L1), 53 of 59 graded death dates are wrong, and 55 of 125 birth dates are wrong at the store's full date. The plan holds a year; the fill adds a month, a day and a second date.
- H1b (§13.48, `--k5-plan-honest on`) fixed this: precision 0.958. But it refuses three things: numbers, months and capitalised names. It lost 0.055 of SCRIBE v2, and the raters gave it only 0.560 against D0.

**Hypothesis.** H1b lost fluency mainly by refusing names. Names are most of what a planned sentence says ("was a New Zealand politician and writer"). Dates are most of what it gets wrong. So refusing only numbers and months, unless they are the fact's own words, and dropping the skeleton when it asks for one, should keep most of H1b's precision gain at almost no cost in fluency.

## The change

One flag, off by default, byte-identical when off. `--k5-plan-honest` now takes `off`, `on` (H1b, as before), `dates` or `nations`. Only `planner/slots.hpp` (`FillOpts::honest` is now a mode, 0 to 3) and `planner/writer.hpp` (`Filler`) change. The discourse writer already passes `honest` to its fill, so `discourse.hpp` does not change.

At a class position of a skeleton:
- **`dates` (HD1):** a word with a digit, or a month name, is refused unless it is one of the fact's own words. Capitalised words are not refused.
- **`nations` (HD2):** as HD1, and also a nationality or country word (the one-word names of GEN_FACTS' name table, and the second words of its two-word names, such as "Zealand") is refused unless it is one of the fact's words, the prompt's or the text's so far, or the sentence's first word. Other names are not refused.
- **H1b's drop rule, for both:** if the class's most probable word at a position is a refused word, that fill hypothesis ends. A skeleton that cannot be filled is dropped, as any failed fill is. A refused word lower in the list is skipped.

Why not a separate rule for a second date in a "(born - died)" frame: under HD1 a second date can only be written from the fact's own numbers. A birth fact holds one year, so a frame "(1926 - [year])" asks for a number the fact does not hold, and HD1 drops the skeleton already. A separate frame rule would change almost nothing, so HD2 tests nationalities instead: they are the next most common contradiction (W1b: 8 nationality contradictions against 2 birth places).

## The baseline

**L1**, today's default: `scripts/gen_eval.sh dev L1 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`.
- Built with `g++ -O2` as `scripts/build_1tb_tools.sh` builds it, `libsais.o` linked into `scribe` too.
- `CYPHALM_TRIE`, `SCRIBE`, `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

**Checks first** (no arm counts until both hold):
- main's build: the GEN file equals `bench/k5_coherence/runs/L1.gen.txt` under `cmp`, MODEL line aside;
- my build with the flag off: the same.

## The arms

Dev, seed 0, 250 prompts, the baseline's command plus the flag:

| Arm | Flag |
|---|---|
| HD1 | `--k5-plan-honest dates` |
| HD2 | `--k5-plan-honest nations` |

Beside them, already committed and not rerun: H1b on the old D0 writer (`bench/gen_quality/runs/H1b.gen.txt`), for the fact check only.

**Machine limits:** at most 2 arms at once, at most 6 threads each, no start while `free -g` shows under 14 GB available.

## Measures (dev)

- SCRIBE v2 with its 95% interval (the score of record), v3.1 beside it, copy32, verbatim13, wall time, peak RAM.
- **Fact precision, a registered measure:** `python3 scripts/gen_fact_check.py run --v2 --out bench/honest_dates/facts L1=… HD1=… HD2=… H1b=…` (GEN_FACTS amendment A1, the store and drop rule as registered there; the script is copied unchanged from `wip/gen-facts`). Reported: statements a sample, share contradicted, share supported, precision S/(S+C), all with 95% bootstrap intervals; counts by relation.
- The number of samples that differ from L1, in all 250 and in the 50 rated prompts.

## The screen

**Reading.** I read the samples that differ from L1 on dev prompts the raters will not see (not in `bench/blind/llm_rating/prompts.json`), the first 20 by prompt number, each beside L1's.

**An arm passes if all hold:**
- verbatim13 is 0;
- copy32 at most 0.30;
- SCRIBE v2 at least L1's minus 0.03;
- by my reading, it reads at least as well as L1.

**Candidates:** at most 2 that pass. Both go to the raters if both pass and they differ from each other in at least 10 of the 50 rated prompts; else the one with the higher fact precision.

## The deciding judge

`python scripts/blind_rate.py make --out bench/blind/hd_pair --ids bench/blind/llm_rating/prompts.json --arm L1=… --arm X=… --mode pair --baseline L1 --seed 20261119`: the 50 prompts, four fresh blind Claude raters. Packets, `key.b64` and `RATER_PROMPT.md` are committed and pushed before the raters run. `KEY.json` stays out of the repo.

Samples identical to L1's are rated too (they come out as ties). This pulls the share toward 0.5, which is right: the rule asks only that the arm reads no worse.

**The adoption rule** (both parts must hold):
1. **Not worse to read.** The arm's win share against L1 (a tie counts half) is at least 0.50. (§13.38's interval: the share minus 0.5 with its 95% bootstrap interval, and the sign in each rater pair, is reported. A share of at least 0.50 cannot be significantly worse.)
2. **A clear gain in facts.** On dev, its fact precision is at least L1's plus 0.20, its 95% interval lies wholly above L1's interval, and its share contradicted is at most half of L1's.

If both arms pass, the one with the higher fact precision is adopted; if their precisions are within 0.02, the higher win share. The adopted arm becomes the default (Odin can veto). Then GEN-A test is read once, for that arm only, with SCRIBE and the fact check.

## Predictions

| # | Prediction |
|---|---|
| P1 | With the flag off, my build equals L1 (`cmp`); main's build equals L1 |
| P2 | HD1 changes 50 to 90 of the 250 dev samples (about 77 get a plan); HD2 changes a few more than HD1 |
| P3 | HD1: SCRIBE v2 within 0.02 of L1; copy32 within 0.01; verbatim13 0 |
| P4 | HD1: 0.8 to 1.3 statements a sample (L1 1.48, H1b 0.56) |
| P5 | HD1: share contradicted at most 0.08 (L1 0.324); precision at least 0.80 (L1 0.437) |
| P6 | HD1: at most 5 contradicted death dates, at most 10 contradicted birth dates |
| P7 | HD2: precision at least HD1's; at most 2 contradicted nationalities; SCRIBE v2 within 0.02 of HD1 |
| P8 | Raters: HD1 0.45 to 0.60 against L1 (many samples are equal); by the rule HD1 is adopted |

## Results

**Checks** (2026-11-12): main's build gives a GEN file equal to `bench/k5_coherence/runs/L1.gen.txt` under `cmp` (MODEL line aside), and so does my build with the flag off. **Hit.**

**Dev, seed 0, 250 prompts** (`bench/honest_dates/runs/`; fact check `bench/honest_dates/facts/`, `--v2`):

| Arm | SCRIBE v2 [95%] | v3.1 | copy32 | verbatim13 | differ from L1 (all / rated) | statements a sample | contradicted | precision S/(S+C) [95%] | wall | peak RAM |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 0.610 [0.577, 0.645] | 0.546 | 0.242 | 0 | – | 1.48 | 0.324 | 0.437 [0.369, 0.507] | 6:52 | 11.3 GB |
| HD1 | 0.570 [0.541, 0.601] | 0.512 | 0.246 | 0 | 70 / 11 | 0.71 | 0.045 | 0.765 [0.620, 0.920] | 7:31 | 11.3 GB |
| HD2 | 0.570 [0.540, 0.602] | 0.518 | 0.239 | 0 | 74 / 11 | 0.74 | 0.027 | 0.865 [0.757, 0.960] | 6:16 | 11.3 GB |
| H1b (old D0 writer, for reference) | – | – | – | – | – | 0.56 | 0.007 | 0.958 [0.880, 1.000] | – | – |

By relation (supported / contradicted): L1 birth date 70/55, death date 6/53, nationality 6/8; HD1 birth date 0/1, death date 0/1, nationality 11/0, birth place 5/4; HD2 birth date 0/0, death date none, nationality 16/0, birth place 5/4.

**Reading** (the first 20 samples that differ from L1 on unrated dev prompts, each beside L1's; and HD2 beside HD1):
- **The invented dates are gone, and so are all date sentences.** HD1 and HD2 write no birth date at all, not even the right ones (L1 had 70 right). The skeleton bank's date skeletons all ask for a month and a day ("(born [month] [day], O)"). A plan that holds only a year never fills one, so every date skeleton is dropped.
- **What replaces them is weaker.** L1's lead sentences are false but fluent and on topic ("Hirohito (November 10, 1901 - October 25, 1963) is an American actor."). In their place come free K5 sentences or other fact skeletons, often off the subject ("The U. S. states. He was accused of sexual assault."). SCRIBE sees this as a loss of 0.040.
- **Nationalities:** HD2 removes the remaining nationality contradictions and reads about like HD1 ("Jacques Tardi is a french writer and poet." in place of an invented "German composer").
- Other invented claims remain in both: places ("Auckland is a city located in the Belgian province of New Zealand"), and "lives in Ottawa, Ontario".
- By my reading, neither arm reads as well as L1 on these 20 prompts. The loss is small, and it sits in the first sentences.

**Screen:** both arms fail. SCRIBE v2 0.570 is below L1's minus 0.03 (0.580), and my reading has them a little below L1. **No candidate, so no packets and no raters.** By the registered rule, nothing is adopted.

**Predictions:**
- P1 byte-identical with the flag off: **hit**.
- P2 HD1 changes 50–90 samples: **hit** (70); HD2 a few more: **hit** (74).
- P3 HD1 SCRIBE v2 within 0.02 of L1: **miss** (−0.040); copy32 within 0.01: **hit** (+0.005); verbatim13 0: **hit**.
- P4 HD1 0.8–1.3 statements a sample: **miss** (0.71).
- P5 HD1 contradicted at most 0.08: **hit** (0.045); precision at least 0.80: **miss** (0.765).
- P6 at most 5 death and 10 birth dates contradicted: **hit** (1 and 1).
- P7 HD2 precision at least HD1's: **hit** (0.865); at most 2 nationality contradictions: **hit** (0); SCRIBE within 0.02 of HD1: **hit** (0.000).
- P8 raters, HD1 adopted: **not scored** (no arm passed the screen).

**What it says:**
- **The hypothesis is half right.** Refusing names was not what cost H1b its fluency here: HD1 refuses no names and loses as much SCRIBE as H1b did (−0.040 here, −0.055 there). The cost comes from the dates themselves. Every date skeleton wants a month and a day the plan does not hold, so dropping them removes the lead sentence.
- **The next step:** let a fact that holds only a year use a year-only skeleton ("X (born 1926) was ..."), or cut the month and day out of the skeleton instead of dropping it. That would keep the true birth years L1 writes (70 of them) without the invented days. It needs a change to the skeleton choice, not only to the fill.
