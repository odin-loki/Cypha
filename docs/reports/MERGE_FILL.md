# Merge M3: the planner's fill by dynamic beam allocation (dev)

**Odin Loch** · 2026-11-05 · branch `wip/merge-fill` · GENERATION_PLAN §13.40 M3

**Outcome first.** Dynamic beam allocation (DBA) is in the planner's fill as an option: `sample --k5 … --k5-plan-fill dba [--k5-plan-beam N]`. With neither flag, the output is byte-identical to the old binary on all 250 dev prompts. DBA does not make the fill cheaper. At today's beam (F1) the fill costs 3% more time and 5% more K5 distributions; with the beam doubled (F2) it costs twice as much. SCRIBE does not move: v2 0.532 (P1), 0.535 (F1), 0.523 (F2), all intervals overlapping. But F1 often runs the fact's two names together ("He Gothenburg University Gothenburg Municipality was born in , ."): 21% of its fact sentences, against 4% for P1 and 7% for F2. It happens when the names have 4 or more tokens, so the banks outnumber the beam. The blind pairwise packets for F1 and F2 against P1 are built and committed; the raters are left to the lead.

## Method

**Today's fill (P1).** One beam of 4 per skeleton, 3 words per class code. The skeleton fixes where the subject (S) and the value (O) go. At a slot, every hypothesis writes the name's tokens. It is not grid beam search: there is no grid of beams by constraints met, so its cost is already one beam per step.

**Why the names must be able to move.** With fixed slots, every hypothesis has met the same number of fact tokens at every step. DBA's banks would then hold one group, and DBA would equal the plain beam. So the DBA fill (`Filler::fill_dba` in `lm/planner/writer.hpp`) treats the names as lexical constraints, as Post and Vilar (NAACL 2018) do:

- The constraints are S's tokens and O's tokens, C in all.
- Before any class position ahead of a name's slot, a hypothesis may start that name. A started name runs to its end; it is never broken off.
- At its slot, a name not yet written starts. The slot of a written name is skipped. So the slot is a deadline, and P1's placements stay in the search.
- Every step writes one token for every hypothesis (a class word or a name's token). All hypotheses finish together, with every name written.
- Class words are chosen as in P1: the best 3 of the class by K5, with the copy cap and the repeat ban. A name's token costs log₂ of its K5 probability, as at a slot.

**Banks.** One beam of k hypotheses is split over banks 0..C by the constraint tokens met. Each bank gets k / (C+1), and the remainder goes to the top bank. Then, from the top bank down, a bank with more slots than candidates gives its spare slots to the nearest banks that have more candidates than slots (the lower one first). Sockeye's single leftward pass can leave slots unused; this version does not while candidates remain. In the selftest, Sockeye's pass gave "Alan Smith lived born in Paris ."; this one gives P1's "Alan Smith was born in Paris .".

**Not as in Sockeye:** the class words come 3 per hypothesis, not from a global top-k over the beam; a started phrase is never broken and unwound.

**Cost.** Both fills compute one K5 distribution (32,000 words) per hypothesis per step. So DBA cannot be cheaper than P1 at the same beam. Its saving is against grid beam search (k·(C+1) hypotheses), which the planner never used.

**Instrumentation.** The planner counts the time inside `Filler::fill` (summed over threads), the fills, the K5 distributions and the time of its fact steps. One line goes to stderr at the end of `sample`; the GEN file is not touched.

**Unchanged:** the plan, the skeleton choice, the gate and its model (`bench/k5_planner/model/plan_e9.model`), and K5's write-on after the facts.

## Commands

```
rsync -a --exclude .git <worktree>/ ~/m3_src/
cmake -S ~/m3_src/native -B ~/build_m3 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ~/build_m3 --target cyphalm_trie scribe k5_planner
cd ~/m3_src      # the planner reads bench/fact_store/e9_aug and bench/k5_planner/model from here
export CYPHALM_TRIE=~/build_m3/cyphalm_trie SCRIBE=~/build_m3/scribe WIKI=/mnt/d/cypha_llm/enwik8 \
       ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=2
K5_PLAN_LOG=$OUT/P1.plan.log scripts/gen_eval.sh dev P1 $OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt
K5_PLAN_LOG=$OUT/F1.plan.log scripts/gen_eval.sh dev F1 $OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt --k5-plan-fill dba
K5_PLAN_LOG=$OUT/F2.plan.log scripts/gen_eval.sh dev F2 $OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt --k5-plan-fill dba --k5-plan-beam 8
python3 -I scripts/k5_planner_measures.py $OUT/F1.plan.log bench/fact_store/e9_aug
```

`OUT=/mnt/d/cypha_llm/merge/m3`. `gen_eval.sh` always passes `--plan-drop 75497472:100000000`. Selftests pass: `cyphalm_trie`, `scribe` (119/119), `k5_planner` (now with the bank sizes and the DBA fill).

## Byte identity (neither flag)

- New binary against the old one (`~/build_lead/cyphalm_trie`, main before the change), `sample --k5` through the default planner, 250 dev prompts: the GEN file and stdout are identical under `cmp` (sha256 c05220e2…0ce7). The plan logs are identical once sorted (their lines come in thread order).
- The fresh P1 samples also equal `bench/k5_planner/P1_e9_s0.gen.txt` (carriage returns removed: the Windows checkout has CRLF). Only the MODEL line differs; it names the store and model paths. That file is P1 in the packets.

## Results (GEN-A dev, seed 0, 250 prompts)

The four runs (old binary, P1, F1, F2) started together, each on 2 threads, so that they share the machine alike and stay near 8 threads in all. Other jobs ran on the PC at the same time. Fill time is summed over threads; the counts of K5 distributions do not depend on the machine.

| Arm | fill thread-s (ms per fill) | K5 distributions | planner fact steps thread-s | sampling wall | SCRIBE v2 [95%] | v3.1 [95%] | copy32 | 13-word runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 (plain beam 4) | 15.73 (7.72) | 130,298 | 20.42 | 252.6 s | 0.532 [0.502, 0.564] | 0.537 [0.514, 0.558] | 0.0376 | 0 |
| F1 (DBA, beam 4) | 16.14 (7.92) | 136,785 | 20.83 | 250.4 s | 0.535 [0.503, 0.568] | 0.546 [0.523, 0.567] | 0.0380 | 0 |
| F2 (DBA, beam 8) | 29.73 (14.60) | 258,956 | 34.39 | 258.0 s | 0.523 [0.494, 0.554] | 0.541 [0.519, 0.563] | 0.0388 | 0 |

- Against P1: F1 fill time +2.6%, distributions +5.0%; F2 fill time +89%, distributions +99%, fact steps +68%.
- 2,037 fills in each arm (93 planned prompts). F1's extra distributions are likely from its fuller early beam: a hypothesis can start a name as well as take a class word, so the beam fills sooner.
- The fill is about 3% of sampling's thread time, and the planner's fact steps about 4%. Most of the time is K5's write-on gate and the 157 plain-K5 prompts. So F2's doubled fill moves sampling wall time by only 2%.
- The old binary sampled in 246.2 s alongside. Whole `gen_eval` runs (sampling and both SCRIBEs): 5:09, 5:07, 5:13. Peak RAM 757 MB in every arm.
- SCRIBE kept 245 lines for P1 and 244 for F1 and F2.

**Coverage and facts.** The plan does not change, so coverage is the same in every arm: main entity 50 (0.200 [0.155, 0.254]), co-occurring 43 (0.172 [0.130, 0.224]), plain K5 157 (0.628 [0.567, 0.686]).

| Arm | facts stated | relation cue in the sentence | the two names side by side | fact sentences unlike P1's |
|---|---:|---:|---:|---:|
| P1 | 246 | 7 of 48 (0.146 [0.072, 0.272]) | 10 of 246 (0.041 [0.022, 0.073]) | — |
| F1 | 251 | 7 of 50 (0.140 [0.070, 0.262]) | 52 of 251 (0.207 [0.162, 0.262]) | 207 of 245 (88 of 93 prompts) |
| F2 | 247 | 9 of 48 (0.188 [0.102, 0.319]) | 17 of 247 (0.069 [0.043, 0.107]) | 168 of 241 (79 of 93 prompts) |

"Relation cue" is K5_PLANNER.md's measure (`scripts/k5_planner_measures.py`). "Side by side" means the sentence holds "S O" or "O S" (case folded). The 157 plain-K5 prompts are the same text in every arm.

**Examples** (the same fact, the gate's winner):

- Gothenburg University | located_in | Gothenburg Municipality. P1: "He was born in Gothenburg University , Gothenburg Municipality ." F1: "He Gothenburg University Gothenburg Municipality was born in , ." F2: "It is located in Gothenburg University , Gothenburg Municipality ."
- Emil Theodor Kocher | birth_place | Bern. P1 and F2: "Emil Theodor Kocher was born in Bern in Africa , France , …" F1: "Emil Theodor Kocher It Bern has led some to believe that …"
- Berkeley Software Distribution | family | Unix. P1: "This are often used as a family of the Berkeley Software Distribution book Unix ." F1: "Berkeley Software Distribution Unix became one to work in a state - only system in , etc ."

**Blind packets** (not rated here). Both arms are no worse than P1 on SCRIBE v2 (the intervals overlap), so both are packed: `bench/blind/fill_pair/` (R1–R4, RATER_PROMPT.md, key.b64; KEY.json is not committed), made with `scripts/blind_rate.py make --out bench/blind/fill_pair --ids bench/blind/llm_rating/prompts.json --arm P1=bench/k5_planner/P1_e9_s0.gen.txt --arm F1=bench/k5_planner/F1_dba4_s0.gen.txt --arm F2=bench/k5_planner/F2_dba8_s0.gen.txt --mode pair --baseline P1 --seed 40`. Each rater reads 50 pairs. Of the 50 rated prompts, F1's text differs from P1's on 15 and F2's on 14; the others are the same text on both sides and can only tie.

## §13.40 M3 predictions, scored (dev)

| Prediction | Result |
|---|---|
| F1: planner fill time down at least 30% | **Miss.** Fill time +2.6%, K5 distributions +5.0%. Today's fill was never grid beam search, so DBA has nothing to save at the same beam |
| F1: SCRIBE within ±0.02 of P1 | **Hit.** v2 +0.003, v3.1 +0.009 |
| F1: raters about even with P1 | Not scored here (the lead runs the raters) |
| F2: at about P1's time | **Miss** for the fill: +89% fill time, +99% distributions, +68% planner fact steps. Sampling wall time is +2%, because the fill is a small part of it |
| F2: raters prefer F2 to P1 | Not scored here |

## What is weak

- **F1 runs the names together when they are long.** With 4 or more name tokens (C ≥ 4: 67 of 246 stated facts), the C + 1 banks outnumber the beam of 4. Every bank but the top one then gets no slot of its own, and spare slots only flow down from the top. So the beam keeps hypotheses that wrote a name early even where that breaks the skeleton. There F1 runs the names together in 42 of 67 sentences; P1 in 0 of 67. With C ≤ 3, F1 does it in 9 of 178 and P1 in 10 of 179. (C is counted with a plain word-and-mark split, close to K5's tokens.) SCRIBE v2 does not see this (0.535 against 0.532); a reader will.
- **F2 does it less** (6 of 67 long facts, 17 of 247 in all) but costs twice the fill. Its v2 is 0.009 below P1, inside the noise.
- **The gate was trained on P1's fills** (415 items). DBA's fills shift the features it reads (K5's mean bits, the class sequence).
- **One-sided freedom.** A name may come early, never late. Letting it come late would need a stop rule before the final punctuation.
- **Not Sockeye exactly:** 3 class words per hypothesis, no phrase breaking, and the no-waste bank adjustment.
- **Timing.** Thread-seconds carry a few percent of noise (the old binary and P1, the same work, differ by 2.6% in sampling wall time). The distribution counts are exact.
- **Where a cheaper fill would come from instead:** a name's token needs only its own probability and the normaliser, not a whole 32,000-word distribution; and the 8 skeletons of one fact share their first distribution. Neither is DBA, and neither is built here.

## Files

- Code: `native/include/cypha/lm/planner/writer.hpp` (`bank_sizes`, `Filler::fill_dba`, the counters, `apply_plan_flags`), the two flag lines in `native/tools/lm/cyphalm_trie.cpp`, the DBA checks in `k5_planner selftest` (`native/tools/lm/k5_planner.cpp`).
- GEN files: `bench/k5_planner/F1_dba4_s0.gen.txt`, `bench/k5_planner/F2_dba8_s0.gen.txt` (P1: `bench/k5_planner/P1_e9_s0.gen.txt`).
- Summaries and timings: `bench/k5_planner/runs/{F1_dba4_s0,F2_dba8_s0}.summary.md`, `bench/k5_planner/runs/merge_fill_timing.txt`.
- Packets: `bench/blind/fill_pair/`.
- Plan logs, SCRIBE outputs and the old binary's run: `/mnt/d/cypha_llm/merge/m3/` on Odin's PC.
