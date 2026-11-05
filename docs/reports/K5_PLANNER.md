# K5 planner: plan → skeleton → fill → gate (dev)

**Odin Loch** · 2026-11-05 · branch `wip/k5-planner2` · GENERATION_PLAN §13.37, §13.38 B

**Outcome first.** The planner works end to end and puts store facts into the text: 246 facts on dev, where plain K5 states none of them. SCRIBE v2 and v3.1 both rise a little: v2 0.498 → 0.532, v3.1 0.483 → 0.537 (store e9_aug). With the flag off, the output is byte-identical to the base binary. The weak part is the fill: K5 writes the words around the fact, and only 15–23% of the fact sentences still say the relation ("born", "capital", ...). K5 also makes up dates and names around the facts. Pairwise blind rating, the deciding judge, is not run here.

## Design (what was built)

Code: `native/include/cypha/lm/planner/` (`facts.hpp`, `skeleton.hpp`, `writer.hpp`), `native/tools/lm/k5_planner.cpp` (mine, train, selftest), a few lines in `cyphalm_trie.cpp`, the `k5_planner` CMake target.

1. **Plan.** The content plan's entity finder (unchanged) gives the prompt's candidates. The first one that is a store subject with facts gives the plan: up to 4 facts, one per relation, as fact IDs. Every order of them is scored by the relation transitions mined from clean enwik9 and by the prompt's association with each fact's value; the best order is written.
   - **Fallback (cooc):** facts whose *value* is a prompt entity, one per subject, up to 3.
   - **Else:** plain K5 writes the prompt. The path is logged per prompt.
   - Not done: the relations expert's typed patterns as a second fallback.
2. **Skeleton.** Distant supervision over clean enwik9 [0, 100 MB): a sentence that holds a store fact's subject and value gives that relation a shape over K5's 256 classes, with S and O slots. Bank: 2,593 skeletons over 30 relations (e9_aug store), 1,447 (verify_f2r_aug). Per step the relation's skeletons are scored by the class 5-gram, the length model and their count; the best 8 are filled.
3. **Fill.** A beam (width 4, 3 words per class code) on K5 word IDs. Each class code takes a word of that class by K5's probabilities, with the repeat ban and the 12-word copy cap. The slots take the fact's tokens.
4. **Gate.** All 20 structure-expert features plus an MBR consensus on word-ID bigrams, joined by a meta-mixer (3 selectors × 3 learning rates, second layer). It is trained on clean enwik9 [100, 120 MB), which K5 never saw: the fill of the real sentence's own skeleton and fact is the target, against fills of 7 random other skeletons of the same relation. A K5 fluency floor applies (the targets' 5th percentile, −8.49 bits a word). **P2:** K5 alone (highest mean log₂ P) chooses.
   - Test items (last 20%, 83 items, 8 fills each), top-1: mixer 0.373, K5 alone 0.169, MBR alone 0.133, chance 0.125. Only 415 items: few real sentences state a store fact with a known-word skeleton.
   - A first version filled the *best-scored* skeletons as the other candidates. The mixer then learned "pick the odd one" (0.84 top-1, K5 weight negative), the trap of §13.36. That version was dropped before any dev run.
5. After the plan, K5's own sentence gate writes on to 600 bytes.

The search runs on fact IDs, word IDs and class codes. The experts read candidates as K5 tokens (vocabulary entries of the IDs). Only the winner is rendered to bytes.

## Commands

```
cmake -S native -B ~/build_pl2 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ~/build_pl2 --target k5_planner cyphalm_trie scribe cypha_report_card
k5_planner mine  --k5 fuzzy/train.txt --store bench/fact_store/e9_aug --text k5scale/clean/train_all.txt \
    --lo 0 --hi 100000000 --drop 75497472:100000000 --out skel_e9.txt
k5_planner train --k5 fuzzy/train.txt --store bench/fact_store/e9_aug --stats ~/k5s_data/S10.stats --skel skel_e9.txt \
    --text k5scale/clean/train_all.txt --lo 100000000 --hi 120000000 --drop 75497472:100000000 --items 3000 --out plan_e9.model
bench/k5_planner/arm.sh P1_e9_s0 bench/fact_store/e9_aug plan_e9.model full   # P2: ... k5
#  = scripts/gen_eval.sh dev LABEL OUT --prompts 250 --k5 fuzzy/train.txt \
#      --k5-plan STORE --k5-plan-model MODEL --k5-plan-gate full|k5 --plan-drop 75497472:100000000   (K5_PLAN_LOG = per-prompt log)
python3 -I scripts/k5_planner_measures.py P1_e9_s0.plan.log bench/fact_store/e9_aug K5_base_s0.gen.txt
```

The store directory is a plain argument. Facts from the prompts' bytes (`--plan-drop`) and from the sample holdout are dropped, as with `--plan`. Selftests pass: `cyphalm_trie`, `scribe` (119/119), `cypha_report_card` (29/29), `k5_planner`.

## Byte identity (flag off)

New binary against the base commit ae06ab2's binary, `sample --k5 fuzzy/train.txt`, all 250 dev prompts: GEN file and stdout identical under `cmp` (sha256 2b37f604…0a0a). The base output also equals `bench/k5_structure/W0_s0.gen.txt`.

## Results (GEN-A dev, seed 0, 250 prompts)

| Arm | SCRIBE v2 [95%] | v3.1 [95%] | copy32 | 13-word runs | wall / peak RAM |
|---|---:|---:|---:|---:|---:|
| K5 (today) | 0.498 [0.469, 0.527] | 0.483 [0.466, 0.500] | 0.0380 | 0 | 3:10 / 505 MB (6 threads) |
| **P1, e9_aug** (full gate) | 0.532 [0.502, 0.564] | 0.537 [0.514, 0.558] | 0.0376 | 0 | 1:46 / 782 MB |
| P2, e9_aug (K5 gate) | 0.527 [0.499, 0.559] | 0.539 [0.518, 0.560] | 0.0394 | 0 | 2:02 / 782 MB |
| P1, verify_f2r_aug (side) | 0.550 [0.515, 0.581] | 0.548 [0.525, 0.572] | 0.0407 | 0 | 1:53 / 767 MB |
| P2, verify_f2r_aug (side) | 0.525 [0.495, 0.558] | 0.533 [0.513, 0.554] | 0.0379 | 0 | 2:00 / 769 MB |

Wall time covers sampling and both SCRIBE runs. Sampling alone: K5 120 s on 6 threads; P1 56 s and P2 72 s on 8 threads. The planner path skips K5's 8-draft gate for the fact sentences, so it is faster per byte.

**Coverage by path** (Wilson 95%):

| Store | main entity | co-occurring | plain K5 |
|---|---:|---:|---:|
| e9_aug | 50 (0.200 [0.155, 0.254]) | 43 (0.172) | 157 (0.628) |
| verify_f2r_aug | 43 (0.172 [0.130, 0.224]) | 40 (0.160) | 167 (0.668) |

**Facts (e9_aug, P1):** 258 planned, 246 written and stated inside the 600-byte cut. All 246 are store facts (true by construction). None is covered by `facts_wd_check` (they are seeds, not F2 prose facts), so no Wikidata precision can be given. **Relation cue** (the sentence holds one of the relation's F2 pattern words): 7 of 48 sentences with cue lists (0.15 [0.07, 0.27]); P2 11 of 48 (0.23). Plain K5 states 0 of these facts.

## §13.38 B predictions, scored (dev)

| Prediction | Result |
|---|---|
| Byte-identical with the flag off | **Hit** (cmp, 250 prompts) |
| P1 fact precision ≥ 0.8 of stated facts | **Hit, but hollow**: 1.00 against the store by construction. Only 15% of the sentences still say the relation, and K5 makes up dates and names around the fact |
| P1 copy32 no higher than K5's | **Hit** (0.0376 vs 0.0380; e9). Side row f2r: 0.0407, a miss |
| SCRIBE v2: P1 within ±0.05 of K5 | **Hit** (+0.034) |
| SCRIBE v3.1: P1 above K5 | **Hit** (+0.054; the intervals barely touch) |
| Pairwise P1 beats K5; P2 below P1 | Not scored here (blind raters: the lead). On SCRIBE, P2 ≈ P1 |

## What is weak

- **The fill.** K5 fills class codes with its most probable words given four words of context, so a birth_place skeleton can come out as "He was born in Gothenburg University, Gothenburg Municipality." The slots are right; the sentence around them often means something else or false (made-up dates in the co-occurring path: "on March 1, 1781").
- **Co-occurring fallback** writes about other entities (people of the same nationality), which may read as off topic.
- **Coverage:** 63% of prompts still go to plain K5.
- **Gate training data:** 415 items only; the mixer's advantage over K5 alone (0.37 vs 0.17 top-1) is on 83 test items.
- SCRIBE rose, but SCRIBE rewarded topic stuffing before (§13.34). Fact sentences may be a similar effect. The pairwise raters decide.
- The model files point at `~/k5s_data/S10.stats` on Odin's PC (not in the repo).

## Files

- GEN files: `bench/k5_planner/{K5_base_s0, P1_e9_s0, P2_e9_s0, P1_f2r_s0, P2_f2r_s0}.gen.txt`.
- Runs: `bench/k5_planner/runs/` (summaries, per-prompt plan logs, skeleton banks, gate models, mine and train output).

## Blind reading and the test read (added by the lead session, 2026-11-05)

- **Pairwise blind raters** (§13.38 rule; packets `bench/blind/planner_pair/`, committed before the ratings): P1 wins **0.635 [0.575, 0.700]** against K5, rater pairs 0.66 and 0.61. P2 is the same. Where the planner's text differs from K5's (15 of 50 prompts), it wins 0.95 [0.85, 1.00]; the rest are plain K5 on both sides and tie.
- **P1 reads better than K5 by the rule and is adopted**, still off by default.
- **Test read, once** (P1, `bench/k5_planner/test_read/`): SCRIBE v2 0.561 [0.532, 0.590] (K5 0.524), v3.1 0.551 [0.529, 0.574], copy32 0.036.

## On by default (2026-11-05, GENERATION_PLAN §13.39)

`sample --k5` now uses the planner (P1); `--no-k5-plan` is K5 alone. The model and skeleton bank are in `bench/k5_planner/model/`. The default's dev samples equal P1's, and `--no-k5-plan` equals the old K5, on all 250 dev prompts.

## The text-only store (2026-11-06, GENERATION_PLAN §13.46)

- The store behind §13.38's results (`e9_aug`) held 17,907 Wikidata seeds. The planner stated them as facts, and `--plan-drop` never dropped them, since all had byte offset 0. So those results were partly built on facts CyphaLM did not read.
- The default store is now `e9_text`, the same store without them.
- **P1t against K5, blind pairwise:** 0.610 [0.560, 0.670].
- **Test SCRIBE v2:** 0.537 [0.505, 0.564] (P1: 0.561; K5: 0.524).
