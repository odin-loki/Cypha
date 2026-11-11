# Q2: typed slots and a little smoothing in the planner's fill (dev)

**Odin Loch** · 2026-11-06 · branch `wip/plan-fill` · GENERATION_PLAN §13.42 Q2 (with §13.46: the text-only store)

**Outcome first.**
- **Typed slots** work. A name's slot now refuses words that do not suit the name's type. The types are learned from CyphaLM's own store and text, with no list. On the text-only store `e9_text` they halve the slot type errors: 22 of 67 checked names (0.33) with P1t, 11 of 68 (0.16) with typed slots. They also put the two names side by side more often (4 → 12 of 179 sentences). SCRIBE does not move: v2 0.511 → 0.510, v3.1 0.511 → 0.510.
- **No smoothing place helps beyond noise.** Each was tried alone at two small amounts. S2 (gate temperature) and S4 (fact choice) change 0–4 of about 180–250 fact sentences. S3 (add-k on skeleton counts) changes 8–14. S1 (smoothing the fill's distribution) changes the most and makes names and types a little worse on the first store.
- **Chosen arm Q2:** typed slots + S1 (K5 at order 3, 0.05) + S3 (add-5). The choice rule was written before any smoothing arm was read, and applied to the final store's arms. (On the first store the same rule picked S2 + S3.) Q2's lead over typed slots alone is about two sentences.
- **Byte identity:** with none of the new flags, the dev GEN file is byte-identical to main's.
- **Blind pair packets** (Q2T and Q2 against P1t) are built and committed in `bench/blind/q2_pair/`. The raters are left to the lead. Only 9 of the 50 rated prompts differ from P1t.

## What was built

Code: `native/include/cypha/lm/planner/slots.hpp` (new), the `Filler` in `writer.hpp`, `skeleton.hpp` (S4), a few flag lines in `cyphalm_trie.cpp`, and `k5_planner types` plus selftests in `k5_planner.cpp`. Every flag is off by default.

**Typed slots** (`--k5-plan-typed FILE`).
1. **Types.** `k5_planner types` fits a latent class model by EM over the store's entities. Each entity is seen by three things only: the positions it fills (`relation:S`, `relation:O`), the K5 class of its last token, and its shape (digits, capitals, lower case, length). Each entity gets a posterior over K types. 12 types, 4 restarts, 100 iterations.
2. **Context model.** In clean enwik9 [0, 100 MB), every mention of a store entity adds its posterior to the counts of the words and K5 classes at offsets −2, −1, +1, +2. The fit of word w next to a name is the ratio of its count next to the name's types to its count if type did not matter, with a pseudo-count of 2.
3. **In the fill.** A class word next to a typed slot is refused when its fit is below −τ bits. The beam then takes the next word of the class. A skeleton whose class codes next to a slot fit below −τ is dropped, unless every skeleton would be. If no typed fill succeeds, the step is filled untyped.
4. **Fit over the types** (`--k5-plan-typed-fit`): `mix` is log₂ of the posterior-mixed ratio; `log` is the posterior's mean of the log ratios. `mix` lets "born in" through for a university because it might be a town. `log` does not.

**The types on `e9_text`** (31,815 entities, 60 positions; `bench/k5_planner/runs/q2/types_e9t_K12.out`). The model found, among others: people with birth and death years; people with a nationality and an occupation; office holders; places that are inside other places (counties, towns, rivers); countries and cities as objects; years; nationality words ("English", "American"); occupations; taxa (bird families); films; works and games. Years like a month before them (+1.3 bits) and hate "North" or "President" (−7 to −8). Office holders like "President" before them (+5.5). Places that are inside other places like "near" and "at" before them.

**Smoothing places** (`slots.hpp` `FillOpts`):
- **S1** `--k5-plan-s1 L [--k5-plan-s1-mode class|low]`: the fill's next-word distribution becomes (1 − L) K5 + L q. q is K5's class model, or K5 at order 3 after the same context.
- **S2** `--k5-plan-s2 T`: each gate ranker's softmax at temperature T before the second layer mixes them.
- **S3** `--k5-plan-s3 K`: the skeleton score's count term is log₂(K + count). Today K = 1.
- **S4** `--k5-plan-s4 L`: the plan's relation transitions become (1 − L) P(b | a) + L P(b).

**The gate.** Typed slots (τ 1, `log`) change 55% of the fact sentences (134 of 244 on the first store). So the gate was retrained on typed fills, with the same recipe as `plan_e9.model` (`k5_planner train`, clean enwik9 [100, 120 MB), the real sentence's own fill as the target, not real against draft). The result is `bench/k5_planner/model/plan_q2.model`: test top-1 0.373, K5 alone 0.181, chance 0.125 (415 items; `plan_e9.model` had 0.373 too). The same recipe without types reproduces `plan_e9.model` byte for byte at 4 threads. The slot types reproduce byte for byte too (`k5_planner types`, counts written to 7 digits).

## Commands

```
rsync -a --exclude .git <worktree>/ ~/q2_src/
cmake -S ~/q2_src/native -B ~/build_q2 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ~/build_q2 -j4 --target cyphalm_trie scribe k5_planner
cd ~/q2_src
~/build_q2/k5_planner types --k5 fuzzy/train.txt --store bench/fact_store/e9_text --text k5scale/clean/train_all.txt \
    --lo 0 --hi 100000000 --drop 75497472:100000000 --types 12 --out bench/k5_planner/model/slots_e9t.txt --table types_e9t_K12
~/build_q2/k5_planner train --k5 fuzzy/train.txt --store bench/fact_store/e9_aug --stats ~/k5s_data/S10.stats \
    --skel bench/k5_planner/model/skel_e9.txt --text k5scale/clean/train_all.txt --lo 100000000 --hi 120000000 \
    --drop 75497472:100000000 --items 3000 --threads 4 --out plan_q2.model \
    --typed bench/k5_planner/model/slots_e9t.txt --typed-fit log --typed-tau 1
T="--k5-plan-typed bench/k5_planner/model/slots_e9t.txt --k5-plan-typed-fit log --k5-plan-typed-tau 1 --k5-plan-model bench/k5_planner/model/plan_q2.model"
K5_THREADS=4 K5_PLAN_LOG=$OUT/Q2T.plan.log scripts/gen_eval.sh dev Q2T $OUT --prompts 250 --k5 fuzzy/train.txt $T
K5_THREADS=4 K5_PLAN_LOG=$OUT/Q2.plan.log  scripts/gen_eval.sh dev Q2  $OUT --prompts 250 --k5 fuzzy/train.txt $T \
    --k5-plan-s1 0.05 --k5-plan-s1-mode low --k5-plan-s3 5
python3 -I scripts/plan_fill_measures.py $OUT/Q2.plan.log bench/fact_store/e9_text types_e9t_K12 --base $OUT/P1t.plan.log
python3 -I scripts/k5_planner_measures.py $OUT/Q2.plan.log bench/fact_store/e9_text
```

`OUT=/mnt/d/cypha_llm/plan/q2/dev_t` (first store: `dev2`). Each run had 4 threads. Up to 4 runs shared the PC. `gen_eval.sh` passes `--plan-drop 75497472:100000000`. The gate model still names `~/k5s_data/S10.stats`. Selftests pass: `cyphalm_trie`, `scribe` (120/120), `k5_planner` (with the Q2 checks).

## The measures

`scripts/plan_fill_measures.py`, over the stated fact sentences:
- **Names intact:** both names stand as written. Neither runs into a capitalised word next to it, nor into the other name.
- **Relation cue:** the sentence still holds one of the relation's pattern words (`k5_planner_measures.py`, as K5_PLANNER.md). Only about 30–50 sentences have cue lists, so one sentence moves it by 0.02–0.03.
- **Slot type errors:** a store pattern's words next to its `[S]` or `[O]` stand next to a name. That asserts the name fills that position. A position fits the name if the store has the name there, or the name's EM type holds at least 5% of the position's entities.
  - **Rule 1** (written before any Q2 arm): an error if any asserted position does not fit. But the same words often come from several relations' patterns. So it flags right sentences, such as "Carl Friedrich Gauss was born in Braunschweig" ("was born in" is also a `location:S` pattern).
  - **Rule 2** (set after reading P1's and the first typed arm's examples): an error only if no asserted position fits. Both rules are reported.
  - The measure uses the same EM types as the fill. So it is not independent of typed slots.
  - Many remaining errors are measure noise: "X is one of the most important … groups" reads as a `located_in:S` cue for a person.
- **Sentences changed:** fact sentences unlike the baseline's, prompt by prompt.

**The choice rule** (`bench/k5_planner/runs/q2/choice_rule.txt`, written before any S1–S4 arm was read): D = Δ names intact + Δ relation cue − Δ rule-2 type errors + Δ SCRIBE v3.1, all against typed slots alone. The amount of each place with the larger D. A place helps if D > 0. The two best places with D > 0 run together. The arm with the largest D is Q2.

## Byte identity (none of the new flags)

- **Against main** (c10935b, M1 and M2 merged; a fresh build from `git archive`): `sample --k5`, default planner, all 250 dev prompts. The GEN file and stdout are identical under `cmp` (sha256 c05220e2…0ce7). The plan logs are identical once sorted. Both SCRIBE reports are identical apart from the label. This equals `bench/k5_planner/P1_e9_s0.gen.txt` once carriage returns are removed (only the MODEL line differs: it names the store path).
- **After merging §13.46** (main 34da0ec, default store `e9_text`): the branch's default run equals the lead's `P1t` under `cmp`, and `bench/k5_planner/P1t_s0.gen.txt` with carriage returns removed, MODEL line included (sha256 35fb39c0…9192).

## The screen on the first store (`e9_aug`, against P1)

This ran before §13.46 found the Wikidata seeds in `e9_aug`. It chose the typed-slot settings and the smoothing amounts. GEN-A dev, seed 0, 250 prompts. Typed arms use 12 types fitted on `e9_aug`; measures use the same types.

| Arm | SCRIBE v2 [95%] | v3.1 [95%] | copy32 | names intact | cue | type errors r1 | type errors r2 | changed vs P1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 (no flags) | 0.532 [0.502, 0.564] | 0.537 [0.514, 0.558] | 0.0376 | 206/246 (0.837) | 7/48 | 48/80 (0.600) | 33/80 (0.412) | — |
| typed, mix, τ 2 | 0.533 [0.504, 0.565] | 0.537 [0.514, 0.558] | 0.0369 | 208/249 (0.835) | 8/49 | 49/82 (0.598) | 34/82 (0.415) | 56/249 |
| typed, mix, τ 1 | 0.521 [0.494, 0.551] | 0.536 [0.514, 0.559] | 0.0374 | 212/244 (0.869) | 10/49 | 54/90 (0.600) | 30/90 (0.333) | 135/244 |
| typed, log, τ 2 | 0.531 [0.503, 0.563] | 0.537 [0.515, 0.558] | 0.0374 | 209/249 (0.839) | 8/49 | 47/80 (0.588) | 32/80 (0.400) | 64/249 |
| typed, log, τ 1 | 0.524 [0.494, 0.553] | 0.537 [0.514, 0.560] | 0.0371 | 212/244 (0.869) | 10/49 | 52/87 (0.598) | 28/87 (0.322) | 134/244 |
| **Q2T**: log, τ 1, retrained gate | 0.527 [0.498, 0.558] | 0.542 [0.520, 0.564] | 0.0372 | 212/247 (0.858) | 10/50 | 52/91 (0.571) | 29/91 (0.319) | 147/247 |

The first four typed rows use P1's gate. Typed slots at τ 2 refuse too little: a name's posterior is spread, so the mixed ratio lets most words through. τ 1 with the `log` fit had the fewest rule-2 errors, so it is the typed setting of every later arm. The retrained gate and the old one are equal within a sentence or two (D −0.006).

Each smoothing place alone, on top of Q2T:

| Place, amount | v2 | v3.1 | names intact | cue | r2 errors | changed vs Q2T | D |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 class 0.05 | 0.529 | 0.544 | 209/247 (0.846) | 10/50 | 30/87 (0.345) | 82 | −0.036 |
| S1 class 0.15 | 0.540 | 0.544 | 208/248 (0.839) | 11/50 | 33/90 (0.367) | 131 | −0.046 |
| S1 low 0.05 | 0.527 | 0.540 | 209/247 (0.846) | 10/50 | 30/92 (0.326) | 27 | −0.022 |
| S1 low 0.15 | 0.527 | 0.542 | 209/247 (0.846) | 8/50 | 29/89 (0.326) | 67 | −0.059 |
| S2 T 1.5 | 0.528 | 0.542 | 212/247 (0.858) | 10/50 | 29/91 (0.319) | 1 | 0.000 |
| S2 T 3 | 0.529 | 0.542 | 213/247 (0.862) | 10/50 | 29/91 (0.319) | 2 | +0.004 |
| S3 add-2 | 0.528 | 0.544 | 212/248 (0.855) | 10/50 | 29/91 (0.319) | 9 | −0.001 |
| S3 add-5 | 0.530 | 0.545 | 212/248 (0.855) | 11/50 | 29/91 (0.319) | 14 | +0.020 |
| S4 0.1 | 0.527 | 0.542 | 212/247 (0.858) | 10/50 | 29/91 (0.319) | 0 | 0.000 |
| S4 0.3 | 0.526 | 0.541 | 212/247 (0.858) | 10/50 | 29/91 (0.319) | 4 | −0.001 |
| S2 T 3 + S3 add-5 | 0.531 | 0.545 | 213/248 (0.859) | 11/50 | 29/91 (0.319) | 15 | +0.024 |

All v2 intervals overlap P1's (about ±0.03). copy32 0.036–0.038 in every arm, no 13-word copies. On this store the rule picked S2 + S3.

Why the places do so little:
- **S2:** the gate takes the argmax of the mixed rankers. A temperature changes the argmax only when the rankers disagree, which is rare.
- **S3:** 2,551 of the 2,593 skeletons were seen once, so add-k barely changes their order.
- **S4:** it acts only on main-entity plans (50 prompts here), and only where two orders or the kept 4 facts score close. At 0.1 it changed no fact sentence; at 0.3, four.
- **S1:** it moves the fill's words most. On this store its fills kept fewer names intact and had more type errors.

## The final arms on the text-only store (`e9_text`, against P1t)

§13.46 moved every Q arm to a text-only store. So the types were fitted again on `e9_text`: the Wikidata seeds are not CyphaLM's own statistics. The gate was retrained on fills typed by them. Its training items still come from `e9_aug`, the same recipe as P1t's gate, which §13.46 keeps. The typed setting and the smoothing amounts are the screen's. Measures use the `e9_text` types.

| Arm | SCRIBE v2 [95%] | v3.1 [95%] | copy32 | names intact | cue | type errors r1 | type errors r2 | changed vs P1t | sampling / wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1t (default) | 0.511 [0.481, 0.542] | 0.511 [0.491, 0.532] | 0.0388 | 149/179 (0.832) | 6/29 | 37/67 (0.552) | 22/67 (0.328) | — | 194 s / 3:55 |
| **Q2T** (typed slots) | 0.510 [0.479, 0.542] | 0.510 [0.488, 0.530] | 0.0396 | 145/179 (0.810) | 6/30 | 27/68 (0.397) | 11/68 (0.162) | 104/179 | 194 s / 3:51 |
| + S1 low 0.05 | 0.513 [0.483, 0.545] | 0.511 [0.489, 0.531] | 0.0393 | 147/179 (0.821) | 6/30 | 27/66 (0.409) | 11/66 (0.167) | 104/179 | 188 s / 3:50 |
| + S2 T 3 | 0.510 [0.479, 0.543] | 0.510 [0.488, 0.530] | 0.0404 | 147/179 (0.821) | 6/30 | 29/68 (0.426) | 13/68 (0.191) | 103/179 | 183 s / 3:44 |
| + S3 add-5 | 0.509 [0.480, 0.541] | 0.510 [0.491, 0.530] | 0.0394 | 144/178 (0.809) | 6/30 | 27/69 (0.391) | 11/69 (0.159) | 106/178 | 188 s / 3:45 |
| + S4 0.1 | 0.510 [0.479, 0.542] | 0.510 [0.488, 0.530] | 0.0396 | 145/179 (0.810) | 6/30 | 27/68 (0.397) | 11/68 (0.162) | 104/179 | 188 s / 3:44 |
| + S2 T 3 + S3 add-5 | 0.509 [0.480, 0.541] | 0.510 [0.491, 0.530] | 0.0401 | 146/178 (0.820) | 6/30 | 29/69 (0.420) | 13/69 (0.188) | 105/178 | 195 s / 3:54 |
| **Q2**: + S1 low 0.05 + S3 add-5 | 0.511 [0.482, 0.544] | 0.512 [0.493, 0.531] | 0.0393 | 146/178 (0.820) | 6/30 | 27/67 (0.403) | 11/67 (0.164) | 106/178 | 201 s / 3:59 |

- **Paths** are P1t's in every arm: main entity 24, co-occurring 53, plain K5 173. The plan does not change.
- **D against Q2T:** S1 +0.007, S2 −0.018, S3 +0.001, S4 0.000; S2 + S3 −0.016; S1 + S3 +0.010.
- On this store S2 costs two names, so the screen's pair loses. By the same rule, the places with D > 0 are S1 and S3, and their pair has the largest D. **That pair is Q2.** It differs from Q2T in 22 of 178 fact sentences (14 of 77 planned prompts), and on 3 of the 50 rated prompts.
- **Names side by side** rise with typed slots: P1t 4 of 179 (0.022), Q2T 12 (0.067), Q2 11 (0.062). A refused word can leave a skeleton where the names meet.
- **Cost:** sampling time is the same within the machine's noise (4 runs shared the PC). The fill computes 90,000 K5 distributions in every arm. S1 at order 3 doubles that (180,080), but the fill is about 2% of sampling's thread time. Peak RAM 630 MB (P1t), 638 MB (Q2T and Q2).

**Examples** (the same fact, P1t then Q2T):
- Marc Riboud | nationality | french. P1t: "Marc Riboud are a cappella rock band , bordered in 2000 in England , France ." Q2T: "Its bone is removed by the french architect Marc Riboud in 1995 ."
- Paul Klee | birth_place | Bern. P1t: "Paul Klee is located in Bern in Asia , Europe , …" Q2T: "v . 105 ) . In the spring of this influence , he went up to the show in the state of Bern kept by Paul Klee ." The person is no longer "located", but the gate chose a broken skeleton.
- Tsuen Wan District | located_in | Hong Kong. P1t: "He died at Tsuen Wan District , Hong Kong , on the 20 th of April 1809 ." Q2T: "He was born on August 31 , 1918 in Tsuen Wan District in Hong Kong ."

## Blind packets (not rated here)

`bench/blind/q2_pair/` (R1–R4, RATER_PROMPT.md, key.b64; KEY.json is not committed), made with:

```
python3 scripts/blind_rate.py make --out bench/blind/q2_pair --ids bench/blind/llm_rating/prompts.json \
    --arm P1t=bench/k5_planner/P1t_s0.gen.txt --arm Q2T=bench/k5_planner/Q2T_typed_s0.gen.txt \
    --arm Q2=bench/k5_planner/Q2_typed_s1s3_s0.gen.txt --mode pair --baseline P1t --seed 43
```

Each rater reads 50 pairs. Q2T's and Q2's texts differ from P1t's on 9 of the 50 rated prompts each. The other 41 are the same text on both sides and can only tie.

## §13.42 Q2 predictions, scored (dev)

| Prediction | Result |
|---|---|
| Slot type errors down at least 50% | **Hit on rule 2 on the final store, miss otherwise.** `e9_text`: rule 2 0.328 → 0.162 (−51%; the intervals [0.228, 0.447] and [0.093, 0.267] just overlap), rule 1 0.552 → 0.397 (−28%). First store: rule 2 0.412 → 0.319 (−23%), rule 1 −5%. Rule 2 was set after reading examples, and the measure shares the fill's EM types |
| One smoothing place helps (most likely S1 or S2) | **Miss.** No place moves any measure by more than about two sentences. S1 hurts names and types on the first store; S2 changes 1–4 sentences |
| The raters prefer Q2 to P1 | Not scored here (the lead runs the raters). The packets are against P1t, after §13.46 |

## What is weak

- **The measures are thin.** About 70 checked names and 30 cue sentences per arm. The type-error rules lean on store patterns that are shared across relations. Rule 2 was chosen after reading examples.
- **The measure is not independent.** It uses the same EM types as the fill. A rater reading is the real check.
- **Names run together more.** Typed slots triple "S O" side by side (4 → 12).
- **The skeletons are the bigger fault.** Many fills are fluent nonsense around correct names ("is one of the most important and innovative … groups in the world", "v . 105 ) . In the spring …"). Typed slots cannot fix a skeleton mined from a bad sentence.
- **The context window is ±2 tokens.** "Terracina was the son of" puts a town as a person, but "son" is three tokens away.
- **The screen ran on the store with Wikidata seeds.** The settings carried over to `e9_text`. There S2 turned from +0.004 to −0.018, all within one or two sentences.
- **Little reaches the raters.** Only 9 of the 50 rated prompts differ from P1t, because 69% of prompts get no plan.
- The gate model still reads `~/k5s_data/S10.stats` (Odin's PC only).

## Files

- Code: `native/include/cypha/lm/planner/slots.hpp`; `writer.hpp` (`Filler`, `Model::best_skeletons`, `candidates`, `dist_temp`, `apply_plan_flags`); `skeleton.hpp` (S4); `native/tools/lm/cyphalm_trie.cpp` (flag lines); `native/tools/lm/k5_planner.cpp` (`types`, train's `--typed…`/`--s1…`, `selftest_q2`); `scripts/plan_fill_measures.py`.
- Models: `bench/k5_planner/model/slots_e9t.txt` (the slot types on `e9_text`), `bench/k5_planner/model/plan_q2.model` (the gate on typed fills).
- GEN files: `bench/k5_planner/Q2T_typed_s0.gen.txt`, `bench/k5_planner/Q2_typed_s1s3_s0.gen.txt` (P1t: `bench/k5_planner/P1t_s0.gen.txt`).
- Runs: `bench/k5_planner/runs/q2/` — the type reports and tables for both stores, training output, the choice rule, and per-arm summaries, measures and planner counters (`screen/`, `final/`; the plan logs of P1t, Q2T and Q2 as `final/*.plan.tsv`).
- Packets: `bench/blind/q2_pair/`.
- Full outputs (GEN, plan logs, SCRIBE): `/mnt/d/cypha_llm/plan/q2/dev2/` (first store) and `/mnt/d/cypha_llm/plan/q2/dev_t/` (final), on Odin's PC.
