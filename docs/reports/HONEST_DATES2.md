# H3: true dates in place of dropped ones

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/honest-dates2` · after [HONEST_DATES](HONEST_DATES.md) (H2), GENERATION_PLAN §13.58, §13.61

(The arms, the screen, the measures, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Odin's word

Odin, on H2: "Fix H2 somehow and use it." So H3's aim is a fixed H2 that becomes the default. Two changes to H2's rules follow from it, both Odin's call:
- SCRIBE v2's floor (L1 − 0.03 = 0.580) is reported but is not part of the screen.
- The adoption rule below picks the best fixed arm the raters do not find worse.

## Why

- L1 (today's default) writes wrong dates. Fact precision 0.437 [0.369, 0.507] (GEN_FACTS v2); 53 of 59 death dates and 55 of 125 birth dates wrong.
- H2's `nations` (HD2) raised precision to 0.865, but SCRIBE v2 fell 0.610 → 0.570. The reason: every date skeleton in the bank asks for a month and a day. H2 lets the fill write a month or a number only from the fact's own words, and a fact holds a year. So no date skeleton is ever filled, and the lead sentence is dropped. HD2 wrote no birth date at all (L1 had 70 right).
- **Most of those months and days are in the store.** The store's rows hold full dates (`birth_date 1898-02-10`). The planner's store keeps only the year (`CPS::canon` makes every date a year). In `bench/fact_store/e9_text`, 7,463 of the 11,117 subjects with a birth year also have a full birth date of that year, and 3,348 of 8,097 with a death year have a full death date.

**Hypothesis.** H2 dropped the date sentences because the fill had no true month and day to write. Give it the store's own dates, and skeletons that ask only for what the store holds, and the date sentences come back true. That should keep HD2's precision and win back most of its loss in fluency.

## The change

One mode, off by default, byte-identical when off: `--k5-plan-honest life`. One more flag: `--k5-plan-skel-add FILE[,FILE]`. Code: `planner/life_dates.hpp` (new; `k5_dates selftest` covers the trim and fit rules), `planner/writer.hpp`, `planner/slots.hpp`, `planner/skeleton.hpp`, and the tool `native/tools/lm/k5_dates.cpp` (new; built as `g++ -std=c++23 -O2 -pthread -I native/include -I native/third_party -I native/third_party/libsais native/tools/lm/k5_dates.cpp libsais.o`, no CMake target). `discourse.hpp` does not change: it passes `honest` to its fill already, and it takes skeletons from the bank and `best_skeletons`.

**`life` is HD2 (`nations`) plus three things:**
1. **The life book.** For each store subject: its birth and death year, and the month and day where the store's rows give one full date of that year (`facts_f2.tsv`, `seeds.tsv`, the planner's drops applied). A year is used only if the planner's store holds it as the subject's one birth or death year. Two different full dates: no month or day.
2. **The fill may write the book's words.** At a class position, a month, a day or a year of the subject's life dates may be written, each once and in order: birth before death, a date's words together (its year ends it), O's own year only at its slot. Where the most probable word of a class is a date the plan does not hold, the position takes a date word of its class that the book allows; if there is none, the fill hypothesis ends, as in H2. Everything else is as HD2.
3. **The fit.** K5's classes are read from K5: a class is a month class if month names hold at least half its mass under P(w | class), a number class if words with a digit do (classes 88 months; 161, 166, 193, 222, 232 numbers: `k5_dates classes`). A skeleton that asks for more months, or more numbers, than the book holds for the subject (O's own year aside) is left out of `best_skeletons`. If that would leave none, nothing is left out (the fill decides, as in H2).

**Skeletons that ask only for a year** (rule 10: from real text, or by a stated rule from mined skeletons; no hand-written sentence):
- **Trimmed** (`bench/honest_dates2/skel_trim.txt`, `k5_dates trim` on `skel_e9.txt`): for a birth or death year skeleton whose O sits in a date inside brackets, inside those brackets only: **A** a date keeps only its year ("(born [month] [day], O)" → "(born O)"; "([month] [day], [year] - [month] [day], O)" → "([year] - O)"); **B** the dates after O's go, with the dash ("(O - [month] [day], [year])" → "(O)"); a date before O's stays (without it, "(date - O)" would read as a birth date); **AB** both. A variant that leaves "( O )" alone, or one already in the bank, is not kept. Each variant has its parent's count. 479 variants, 471 distinct (307 birth year, 172 death year).
- **Mined** (`bench/honest_dates2/skel_years.txt`, `k5_dates mine`): distant supervision as `k5_planner mine` (store `e9_aug`, clean enwik9 [0, 100 MB), the same drop), for birth and death years only, keeping a sentence only if no word outside the slots is a month name or a number other than a year ("Kurt Vonnegut (born 1922) is ...").

The added skeletons go after the bank's own, so the bank's indices stay.

## The baseline

**L1:** `scripts/gen_eval.sh dev L1 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, built with `g++ -O2` as `scripts/build_1tb_tools.sh` builds it (`libsais.o` linked into `scribe`); `CYPHALM_TRIE`, `SCRIBE`, `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

**Checks first:**
- main's build (origin/main 508a532e): the GEN file equals `bench/k5_coherence/runs/L1.gen.txt` under `cmp`, MODEL line aside. **Done before this was written: equal.**
- my build with the flags off: the same.
- my build, HD2: equal to `bench/honest_dates/runs/HD2.gen.txt` (MODEL line aside).

**Before this was written** (stated so the reader can weigh it): to check that the code runs, I ran HLT's flags on the first 25 dev prompts (`cyphalm_trie sample --prompts 25`, not `gen_eval.sh`, no SCRIBE) and looked at the 8 samples that differ from L1. The life book loaded (9,974 subjects; birth 8,972 years, 5,066 with month and day; death 6,087 years, 2,737 with month and day) and true store dates were written ("Björn Ulvaeus (born April 25, 1945)"). Nothing was changed after it. Those prompts overlap the 20 I read for the screen.

## The arms

Dev, seed 0, 250 prompts, the baseline's command plus:

| Arm | Flags |
|---|---|
| HD2 | `--k5-plan-honest nations` (H2's arm, rerun on my build; Odin's raters see it) |
| HL | `--k5-plan-honest life` |
| HLT | `--k5-plan-honest life --k5-plan-skel-add bench/honest_dates2/skel_trim.txt` |
| HLTM | `--k5-plan-honest life --k5-plan-skel-add bench/honest_dates2/skel_trim.txt,bench/honest_dates2/skel_years.txt` |

**Machine limits:** at most 2 arms at once, at most 6 threads each, no start while `free -g` shows under 14 GB available.

## Measures (dev)

- SCRIBE v2 with its 95% interval (reported; its floor is waived), v3.1 beside it, copy32, verbatim13, wall time, peak RAM.
- **Fact precision, the registered measure:** `python3 scripts/gen_fact_check.py run --v2 --out bench/honest_dates2/facts L1=… HD2=… HL=… HLT=… HLTM=…` (GEN_FACTS amendment A1, the store and drop rule as registered there). Reported: statements a sample, share contradicted, share supported, precision S/(S+C), with 95% bootstrap intervals; counts by relation.
- Samples that differ from L1, in all 250 and in the 50 rated prompts. From the planner's log: the book's coverage, skeletons left out by the fit.

## The screen

**Reading.** I read the samples that differ from L1 on dev prompts the raters will not see (not in `bench/blind/llm_rating/prompts.json`), the first 20 by prompt number, each beside L1's and beside HD2's.

**An H3 arm passes if all hold:**
- verbatim13 is 0;
- copy32 at most 0.30;
- by my reading, it reads at least as well as L1.

SCRIBE v2 is reported beside it, not screened (Odin's call).

**Candidates:** at most 2 H3 arms that pass. If more pass, the 2 with the highest fact precision. HD2 goes to the raters too, whatever the screen says (Odin asked that the raters see it). If no H3 arm passes, HD2 alone goes, and I say so.

## The deciding judge

`python scripts/blind_rate.py make --out bench/blind/hd2_pair --ids bench/blind/llm_rating/prompts.json --arm L1=… --arm HD2=… --arm X=… [--arm Y=…] --mode pair --baseline L1 --seed 20261121`: the 50 rated prompts, four fresh blind Claude raters, each arm against L1. Packets, `key.b64` and `RATER_PROMPT.md` are committed and pushed before the raters run. `KEY.json` stays out of the repo. The lead runs the raters.

Samples identical to L1's are rated too (they come out as ties).

**The adoption rule.** Among the arms rated against L1, an arm is **not worse to read** if the upper bound of the 95% bootstrap interval of its win share against L1 (a tie counts half) is at least 0.50 (§13.38's interval, as `blind_rate.py score` prints it). An arm is **eligible** if it is not worse to read and its dev fact precision is at least L1's plus 0.15 (0.437 + 0.15 = 0.587). **The eligible arm with the highest fact precision becomes the default** (if two are within 0.02, the one with the higher win share). If no arm is eligible because the raters find every arm worse, I report it and the lead asks Odin. The adopted arm's GEN-A test run, with SCRIBE and the fact check, is read once afterwards.

## Predictions

| # | Prediction |
|---|---|
| P1 | My build with the flags off equals L1 (`cmp`); my HD2 equals H2's HD2 file |
| P2 | HL writes birth dates again: at least 40 supported birth-date statements, at most 8 contradicted (L1 70/55, HD2 0/0) |
| P3 | HL, HLT, HLTM: fact precision at least 0.80 each; share contradicted at most 0.08 |
| P4 | HLT writes more date statements than HL (the year-only facts get a skeleton); HLTM at least as many as HLT |
| P5 | SCRIBE v2 (not screened): HL at least 0.585; HLT and HLTM within 0.015 of L1 (0.610) |
| P6 | copy32 within 0.01 of L1 (0.242) for every arm; verbatim13 0 |
| P7 | By my reading HLT or HLTM reads as well as L1 on the 20 unrated prompts, and better than HD2 |
| P8 | Raters: the best H3 arm's win share against L1 between 0.45 and 0.60; it is adopted by the rule; HD2's share below the H3 arm's |

## Results

**Checks** (2026-11-12): main's build equals `bench/k5_coherence/runs/L1.gen.txt` under `cmp` (MODEL line aside); so does my build with the flags off; my build's HD2 equals `bench/honest_dates/runs/HD2.gen.txt`. **Hit.**

**Dev, seed 0, 250 prompts** (`bench/honest_dates2/runs/`; fact check `bench/honest_dates2/facts/`, `--v2`):

| Arm | SCRIBE v2 [95%] (not screened) | v3.1 | copy32 | verbatim13 | differ from L1 (all / rated) | statements a sample | contradicted [95%] | precision S/(S+C) [95%] | wall | peak RAM |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 0.610 [0.577, 0.645] | 0.546 | 0.242 | 0 | – | 1.48 | 0.324 [0.260, 0.396] | 0.437 [0.369, 0.507] | 9:04 | 11.3 GB |
| HD2 | 0.570 [0.540, 0.602] | 0.518 | 0.239 | 0 | 74 / 11 | 0.74 | 0.027 [0.006, 0.051] | 0.865 [0.757, 0.960] | 7:31 | 11.3 GB |
| HL | 0.590 [0.560, 0.624] | 0.538 | 0.231 | 0 | 75 / 11 | 0.98 | 0.033 [0.012, 0.058] | 0.928 [0.871, 0.972] | 10:11 | 11.3 GB |
| HLT | 0.595 [0.563, 0.630] | 0.539 | 0.232 | 0 | 75 / 11 | 1.12 | 0.036 [0.013, 0.063] | **0.935 [0.888, 0.974]** | 10:03 | 11.3 GB |
| HLTM | 0.595 [0.563, 0.630] | 0.540 | 0.232 | 0 | 75 / 11 | 1.11 | 0.043 [0.018, 0.074] | 0.922 [0.871, 0.966] | 9:25 | 11.3 GB |

(Two arms ran at a time, so wall times are not comparable with a lone run.)

By relation (supported / contradicted):
- birth date: L1 70/55, HD2 0/0, HL 56/4, HLT 86/6, HLTM 83/8;
- death date: L1 6/53, HD2 none, HL 23/0, HLT 39/0, HLTM 37/0;
- nationality: L1 6/8, HD2 16/0, HL 10/0, HLT 8/0, HLTM 9/0; birth place 4/4 in every H3 arm (as HD2).

Planner: the life book holds 9,974 subjects (birth: 8,972 years, 5,066 with month and day; death: 6,087 years, 2,737 with month and day). The planner wrote 473 fact sentences under HL, 472 under HLT and HLTM, against HD2's 373. The mined year-only bank added only 58 skeletons the bank did not hold. (The discourse writer does not print the fit's count, so it is not reported.)

The 6 birth dates HLT gets wrong: four are the planner's store disagreeing with the checker's (George Grenville 1765 against 1712; Heinrich Lübke 1892 against 1894, twice; Chester A. Arthur November 18 against October 5, 1829); two ("Ernst Zündel (born 2005)", "Johan Asplund (born 1980)") come from K5's free sentences after the plan, which no flag here touches.

**Reading** (the first 20 samples that differ from L1 on unrated dev prompts, each beside L1's and HD2's):
- **The date sentences are back, and they are true.** "Björn Ulvaeus (born April 25, 1945)", "Louis Giorgio Bassani (March 4 1916 – April 13 2000)", "Friedrich Wöhler was born in ... on July 31, 1800", "Hirohito (1901 - 1989)". L1's were as fluent and mostly false ("Hirohito (November 10, 1901 - October 25, 1963) is an American actor").
- **Better than HD2.** HD2 put weak free sentences where the lead dates were ("The U. S. states. He was accused of sexual assault."). HL, HLT and HLTM keep a date sentence there.
- **Still weak:** a stilted shape repeats ("In 1901, Hirohito was born in Dublin.", "In 1546, François d'Aguilon was born in Dublin."), and places are still invented ("born in Vienna, Berlin"), as in HD2 and L1. "US" escapes the nationality list ("is a US musician").
- HL, HLT and HLTM read almost alike (HLT and HLTM differ in 22 samples).
- **By my reading, HLT and HL read about as well as L1 on these 20.** L1's leads are a little smoother; theirs are true. It is close. All three H3 arms pass my reading.

**Screen:** HL, HLT and HLTM all pass (verbatim13 0, copy32 0.23, reading). By the rule, the 2 with the highest fact precision go: **HLT (0.935) and HL (0.928)**. HD2 goes too (Odin's word). SCRIBE v2 (not screened): HLT 0.595 is 0.015 below L1, 0.025 above HD2.

**Packets** (`bench/blind/hd2_pair`, `blind_rate.py make ... --arm L1 --arm HD2 --arm HLT --arm HL --mode pair --baseline L1 --seed 20261121`): committed with `key.b64` and `RATER_PROMPT.md` before the raters run. `KEY.json` is kept out of the repo. HLT differs from L1 in 11 of the 50 rated prompts, from HL in 5, from HD2 in 10.

**Predictions so far:**
- P1 byte identity, flags off and HD2: **hit**.
- P2 HL at least 40 birth dates supported, at most 8 contradicted: **hit** (56 / 4).
- P3 precision at least 0.80 and contradicted at most 0.08 for each H3 arm: **hit** (0.922 to 0.935; 0.033 to 0.043).
- P4 HLT more date statements than HL: **hit** (131 against 83); HLTM at least as many as HLT: **miss** (128).
- P5 HL at least 0.585: **hit** (0.590); HLT and HLTM within 0.015 of L1: **hit**, just (0.595).
- P6 copy32 within 0.01 of L1: **miss** for HL and HLT (0.231, 0.232; lower, not higher), hit for HD2 and HLTM; verbatim13 0: **hit**.
- P7 HLT or HLTM reads as well as L1 and better than HD2, by my reading: **hit** (close against L1).
- P8 raters: **not yet scored** (the lead runs the raters).
