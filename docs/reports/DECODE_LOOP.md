# L1: a decode loop with blind readers in it

**Odin Loch** · registered 2026-11-13, before any round runs · branch `wip/decode-loop` · GENERATION_PLAN §13.34, §13.38, §13.51, §13.53, §13.57, §13.66, §13.71, §13.74 · WRITING_SIGNAL, K5_DECODE, K5_MIXDEC

(The loop, the screen, the readers' rule, the search rule, the rule for which metrics to trust, the stop rule and the predictions below were written and committed before any round ran. Results are added under them.)

## Why

- Odin: "We can do all that in a loop, like you read it to minimise error and generate coherent text. We can have several metrics. SCRIBE can be one."
- The hard lesson (WRITING_SIGNAL, §13.57): over about 5,000 blind judgments, SCRIBE v2 agrees with the raters only 0.547, v3.1 is at chance, and a fit of all 17 signals fails on rounds it did not see (0.467).
- SCRIBE rose most for arms the readers marked down: topic stuffing (§13.34), R2 (§13.53), web prose (§13.67).
- So **the metrics do not steer the loop. They screen. The readers decide.** And the loop logs, round by round, which metrics agree with the readers on this writer.

## The loop, in one picture

1. Around the current best setting, make up to 4 candidates, each moving one knob (the search rule below).
2. Write with each on 150 fixed dev prompts: subset A (100, for the metrics) and the 50 rater prompts (for the packets).
3. Measure each on subset A. Drop those that break a hard rule or move far from human text (the screen).
4. Rank the rest by the trusted metrics. Keep at most 2.
5. Build a small blind packet: 20 rater prompts, each survivor against the current best. Two blind raters read it (the lead runs them).
6. A survivor that wins clearly becomes the new best.
7. Log the readers' verdicts beside every metric. From round 4, rank only with metrics that agree with the readers.
8. Every 3 rounds, a full 50-prompt, 4-rater confirmation against the starting default.

## The writer and the start

- **Base:** `origin/wip/mixdec-t8` (job D5: D4's decoder merged onto today's default, T8). Until it appears, the loop is built and registered on `origin/main` (deda93c3), whose default is T8 (§13.71). If D5's branch has not appeared by about 02:45 on 2026-11-13, the loop runs on main's T8 without the decoder, and I say so.
- **The starting best, START:** the base branch's default `sample --k5 /mnt/d/cypha_llm/fuzzy/train.txt` (T8: T 0.8, nucleus 0.8, gate 16 with `wp=1 wm=5`, LDA term `lda_w 1`, retrieval `k=1,rel=0.5`, the planner with the honest fill), or D5's winner if the lead says so.
- **Build:** `cyphalm_trie` and `scribe` with `g++ -O2` as `scripts/build_1tb_tools.sh` builds them (`libsais.o` linked into `scribe`), into `~/dloop_bin`.
- **Check first:** my build's default on all 250 dev prompts equals `bench/k5_decode/runs/T8.gen.txt` under `cmp`, MODEL line aside. If the base is D5's branch, the same check with its default against D5's GEN file.

## The prompts

- **Subset A** (`bench/decode_loop/prompt_sets.json`, "A"): the first 100 GEN-A dev prompts, by number, that are not among the 50 rater prompts of `bench/blind/llm_rating/prompts.json`. All metrics are read on A only.
- **Rater prompts R:** the 50 of `prompts.json`, in its order.
- **The loop file** `bench/decode_loop/loop_prompts.tsv`: A, then R (150 rows). `sample` seeds each prompt by its row, so every loop run, START included, writes from the same seeds. These texts are not the same draws as a 250-prompt dev run; nothing in the loop is compared with one.
- **GEN-A test is never read.** The lead reads it once for any default change.

## The knobs

Only flags that exist on the base branch:

| Group | Knob | Flag | START |
|---|---|---|---|
| sampling | temperature | `--temps T` | 0.8 |
| sampling | nucleus | `--k5-nucleus P` | 0.8 |
| gate | fluency weight | `--k5-gate 16,wf=W` | 0 |
| gate | class-grammar weight | `--k5-gate 16,wg=W` | 0 |
| gate | LDA topic weight | `--k5-lda 100 --k5-lda-w W --k5-disc-text …/train_all.txt --k5-disc-cache /mnt/d/cypha_llm/disc/cache` | 1 |
| retrieval | documents, their share | `--k5-retrieve k=N,rel=R` | k 1, rel 0.5 |
| decoder (D4) | context ratio, judge | `--mixdec ratio=R,judge=J,copy=0` | off on T8 (D5's branch may change it) |
| later | C1's controller targets (`--k5-control`, `wip/k5-control`); D6's learner options (`wip/mixdec-tune`) | — | added by an amendment, after I read their reports, only once they are on the base |

- The association weight `wp` stays 1: it is the gate's reference term (the others are z-scored against it).
- The LDA weight needs `--k5-lda 100` and the disc flags written out: the default sets `lda_w 1` whenever no `--k5-lda` flag is given, so `--k5-lda-w` alone would be overwritten. Checked before round 2: the written-out form with `W=1` equals START.
- A candidate's flags are the current best's flags with one knob moved.

## The metrics (subset A, every text cut near 460 characters as `blind_rate.py` cuts it)

`bench/decode_loop/loop_measures.py`, after `bench/decode_loop/loop_arm.sh`:

| Metric | Source | Good direction (registered) |
|---|---|---|
| SCRIBE v2, v3.1 | `scribe ladder --profile v2/v31` on A's GEN lines (as `gen_eval.sh`) | higher |
| fact precision | `python3 -I scripts/gen_fact_check.py run --v2` on A | higher |
| verbatim13 | `sample`'s GEN_EVAL line, all 150 prompts | must be 0 (Odin's 12-word cap) |
| copy32 | the same line | lower (reported, not screened, as in K5_DECODE) |
| mean surprise per word (`surp_mean`) | WRITING_SIGNAL's word trigram KN over `fuzzy/train.txt` | lower |
| spread of surprise (`surp_sd`) | the same | nearer human |
| words per sentence (`sl_mean`) | `surprise_profile.py`'s sentences | nearer human |
| link to the sentence before (`on_prev`) | `para_plan_mine.py`'s window (as `hook_screen.py`) | higher |
| verbless-fragment rate (`verbless`) | whole sentences of 2+ words with no finite verb by a fixed list (be, have, do, the modals, common verb forms) plus two rules (a 5+ letter word ending in "ed"; a word ending in "s" after he, she, it, which, that, who, this, they) | lower |
| keyword repetition (`stuff3`) | `coh_measures.py`: 3rd and later uses of a content word | lower |
| `commit_text`, `junk`, `rep3` | `surprise_profile.py` | reported |

- **Human ranges** are measured on the human continuations of A (`bench/decode_loop/human.gen.txt`), the same cut: the mean and the per-text spread of each metric. Measured before registration (they decide nothing by themselves): mean surprise 8.92 ± 0.80 bits, surprise spread 5.44 ± 0.37, 20.1 ± 5.2 words a sentence, link 0.571 ± 0.384, verbless 0.050 ± 0.147, stuff3 0.033 ± 0.042.
- The verbless rule is a word list, not a tagger (no tagger model is on the PC). It reads human text at 0.05 and T8's at about 0.21, so it separates them; it is not checked by hand.

## The screen (every candidate)

A candidate is dropped if any holds:
- verbatim13 is not 0 over the 150 prompts;
- fact precision on A is below 0.874;
- **it moves away from human text:** for any of `surp_mean`, `surp_sd`, `sl_mean`, `on_prev`, `verbless`, `stuff3`, its distance from the human mean, in human per-text spreads, grows by more than 0.5 over the current best's. (For mean surprise that is about 0.4 bits a word lower; for stuff3 about 0.021 higher.)

## The ranking (survivors of the screen)

- **Rounds 1–3:** the rank sum over four metrics, each in its registered direction: mean surprise (the best proxy known between settings of this writer, 0.643 on the 337 MB writer, WRITING_SIGNAL), SCRIBE v2, stuff3 and verbless. Lower sum first; ties by lower copy32.
- **From round 4:** the rank sum over the metrics the readers trust (the switch rule below), each in the direction the readers showed. If none qualifies, the rounds 1–3 ranking stays, and I say so.
- **At most 2 survivors go to the readers**, the two best ranked.

## The readers decide (every round)

- **Prompts:** 20 of the 50 rater prompts, in rotation through `prompts.json`'s order: round r takes positions 20(r−1) mod 50 onward, 20 of them, wrapping at 50. So rounds 1, 2, 3 take positions 0–19, 20–39, 40–49 + 0–9.
- **Packet:** `python3 scripts/blind_rate.py make --mode pair --baseline BEST --out bench/blind/loop_rN --ids bench/blind/loop_rN/ids.json --arm BEST=… --arm S1=… [--arm S2=…] --seed 20261210+N` (460 characters). `blind_rate.py` writes four files: R1 and R2 hold the first 10 prompts, R3 and R4 the second 10.
- **Two raters:** rater X reads R1.txt and R3.txt; rater Y reads R2.txt and R4.txt (each file is answered as a fresh task, with `RATER_PROMPT.md`'s text). So every prompt is judged once by each rater, for each survivor. The replies are saved as R1.json–R4.json.
- `KEY.json` is moved out of the repository (the scratchpad, `loop_rN_KEY.json`) before the commit. The packets, `ids.json`, `key.b64` and `RATER_PROMPT.md` are committed and pushed before the raters run. I message the lead; the lead runs the raters and replies with the result.
- **The round's rule:** a survivor's win share against the best (a tie counts half) over the 20 prompts and both raters. **It becomes the new (provisional) best if its share is at least 0.60 and each rater's own share is at least 0.5.** If both qualify: the higher share, then the higher bootstrap lower bound, then the lower copy32. Otherwise the best stays.
- This is a loose rule on purpose (20 prompts give an interval of about ±0.13); the confirmations below are the strict test.

## The search rule (coordinate search with halving)

- **Groups, in this order, then again:** sampling → gate → retrieval → decoder (only if the base has `--mixdec`) → (later groups, by amendment). One group a round.
- **First visit of a group** (from START; each candidate moves one knob):
  - sampling: T 0.75, T 0.85, nucleus 0.7, nucleus 0.9;
  - gate: `wf=0.5`, `wg=0.5`, `lda_w 0.5`, `lda_w 2`;
  - retrieval: `k=2`, `k=4` (rel kept), `rel=0.25`, `rel=1.0` (k kept);
  - decoder: if the base's default has the decoder off: `ratio=0.15`, `ratio=0.3`, `judge=1`, `ratio=0.3,judge=1` (all `copy=0`, as D4's arms); if it is on: ratio × 0.5, ratio × 2, judge 0, judge × 2.
- **Later visits:** for each knob of the group, the step is the last visit's step. If the knob's candidate won its round, the next visit tries one more step the same way, and half a step back. If no candidate on the knob won, the step halves and both ways are tried. A knob that starts at 0 (`wf`, `wg`) uses 0.5 as its first step and never goes below 0. At most 4 candidates: the knobs in the table's order, the winning knob first.
- Successive halving: up to 4 written and measured, at most 2 read, at most 1 kept a round.

## Learning which metrics to trust (registered before round 1)

- **Logged every round** (`bench/decode_loop/rounds.json`): each rater judgment (prompt, survivor, rater, win, loss or tie) beside both texts' metrics, and every arm's metrics on A.
- **Judgment level** (the per-text metrics): for every judgment that is not a tie, with different texts, does the metric, read in its registered direction, pick the text the rater chose? The share, with a 95% bootstrap over (round, prompt) clusters. Confirmation judgments count too (the best against START).
- **Arm level** (SCRIBE v2 and v3.1, fact precision, copy32, and the per-text metrics' means): for every rated survivor, does the sign of its difference from the best on A match the sign of its win share minus 0.5? Reported as k of n.
- **The switch rule:** from round 4 on (recomputed after every round, on all judgments so far), a metric is trusted if its judgment-level interval lies above 0.5 (used in its registered direction) or below 0.5 (used reversed). An arm-level-only metric (SCRIBE, fact precision, copy32) is trusted if it agrees in at least 80% of at least 5 rated survivors (or reversed at 20% or less).
- The trust rule changes only the ranking. The screen stays as registered: a metric the readers disagree with still guards Odin's cap, the facts and the human ranges.

## Confirmations and the stop

- **When:** after rounds 3 and 6, and after the last round if the best changed since the last confirmation. If the best is START, or the best has not changed since the last confirmation, no confirmation runs, and it counts as "no gain".
- **Packet:** the best against START, all 50 rater prompts, 460 characters, four raters (`blind_rate.py make --mode pair --baseline START`, seed 20261310 + c for confirmation c, `bench/blind/loop_cC`). The lead runs it.
- **§13.38's rule:** the best reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Gain:** the best passes the rule and is not the setting confirmed last time. If it fails with a share below 0.5, the best goes back to the last setting that passed (or START), and the loop goes on from there.
- **Stop** after round 8, or after two confirmations in a row with no gain.
- **Reported at the end:** the best setting and its flags, every confirmation's result, every round's table, and the metric-agreement table. Any default change is Odin's call; the lead reads GEN-A test once for it.

## Machine rules

One arm at a time (two only while `free -g` shows 30 GB or more available), 6 threads, no start under 16 GB available (`loop_arm.sh` waits). Waits on done-files, never `pgrep -f`. Only PIDs I started are killed.

## Predictions

| # | Prediction |
|---|---|
| L1 | My build's default equals T8's dev GEN file (`cmp`); the written-out LDA form with W=1 equals START |
| L2 | No first-visit sampling or retrieval candidate breaks the 12-word cap; at most one candidate in the whole loop does (TF did in K5_DECODE) |
| L3 | Mean surprise falls with lower T and lower nucleus and rises with retrieval k, by 0.1–0.4 bits a word |
| L4 | In rounds 1–3, at most one survivor becomes the new best |
| L5 | After round 3, mean surprise is the only per-text metric whose judgment-level interval lies above 0.5 (agreement 0.60–0.70); SCRIBE v2's arm-level sign agreement is near chance |
| L6 | verbless agrees 0.50–0.60 (interval touching 0.5); `on_prev` and `stuff3` near 0.5 |
| L7 | The final best beats START at a confirmation with a win share of 0.55–0.65; it may not pass §13.38 |
| L8 | The loop stops before round 8 on two confirmations with no gain |
| L9 | A loop arm takes 6–10 minutes on 150 prompts (plus SCRIBE and the fact check) at 11.3–12.6 GB peak |

## Amendment 1: X1's probe metrics (2026-11-13, before any candidate was screened or read)

- **Why:** X1 (WHERE_IT_BREAKS, §13.75) found the main fault: K5 sees four words, so it rates "splices" as likely (a 4-word context it has seen, followed by a word that never followed it). The lead asked for X1's break-marking as a screen metric. X1 marked breaks by reading; that cannot run per candidate. Its probe can.
- **The metrics** (`bench/decode_loop/loop_probe.py`; `k5_probe --modes map --no-learner`, built from main ea3a784c): on X1's 30 prompts (the first 30 of subset A, the same ids), each arm's text cut as the raters read it:
  - `splice`: the share of words whose 4-word context K5 has seen but never followed by that word;
  - `tail`: the share of words of rank above 10 in K5's distribution, or P below 0.02.
  - Both are read under T8's model (with the overlay), whatever the arm's settings.
- **Screen:** a candidate is also dropped if its splice share is more than 0.03 above the best's.
- **Ranking, rounds 1–3:** splice (lower) is added as a fifth metric to the rank sum.
- **Logged** beside the readers' verdicts like every other per-text metric, and trusted by the same rule. `tail` is logged, not screened: human text has far more tail words than any writer (half its words lie outside the nucleus, WHERE_IT_BREAKS).
- **Later knobs:** F1's `--k5-minp`, gate fluency floor and clause or splice judge (`wip/k5-fixes`) join the knob table by a further amendment once they are on the base, after I read their report.
- **What had run when this was written:** START and the first round-1 candidate (T 0.75), written and measured; no candidate was screened, ranked or read.

## Results

(Added after the runs, under this line.)

### Checks (2026-11-13)

- My build of main (deda93c3) with no flags, on all 250 dev prompts: the GEN file equals `bench/k5_decode/runs/T8.gen.txt` under `cmp`, MODEL line aside. 9:13 wall, 10.9 GB. **L1 hit.**
- The written-out LDA form with W=1 (`--k5-lda 100 --k5-lda-w 1 --k5-disc-text … --k5-disc-cache …`) equals the default on 6 loop prompts (`cmp`, and the same MODEL line). **Hit.**
- `wip/mixdec-t8` had not appeared on origin by the start of round 1. Rounds run on main's T8 (decoder absent). START: 7:41 wall for the 150 loop prompts, 11.0 GB.

### Round 1: sampling (from START)

Subset A (100 prompts); splice and tail on X1's 30. Full table: `bench/decode_loop/rounds/r1.screen.md`.

| | human | START | T 0.75 | T 0.85 | nucleus 0.7 | nucleus 0.9 |
|---|---|---|---|---|---|---|
| mean surprise (bits) | 8.92 | 7.58 | 7.70 | 7.53 | 7.74 | 7.61 |
| words a sentence | 20.1 | 13.2 | 13.1 | 12.3 | 12.7 | 12.6 |
| link to the sentence before | 0.571 | 0.301 | 0.357 | 0.271 | 0.366 | 0.267 |
| verbless | 0.050 | 0.204 | 0.221 | 0.226 | 0.184 | 0.208 |
| stuff3 | 0.033 | 0.030 | 0.029 | 0.026 | 0.034 | 0.024 |
| splice | 0.216 | 0.240 | 0.212 | 0.247 | 0.225 | 0.240 |
| SCRIBE v2 | | 0.817 | 0.844 | 0.776 | 0.863 | 0.787 |
| fact precision | | 0.938 | 0.938 | 0.938 | 0.938 | 0.938 |
| copy32 | | 0.498 | 0.525 | 0.438 | 0.531 | 0.424 |
| verbatim13 | | 0 | 0 | 0 | 0 | 0 |

- **Screen:** all four pass. **Ranking** (rank sum of surprise, SCRIBE v2, stuff3, verbless, splice): N9 6, T75 7, N7 7 (T75 by the copy32 tie-break), T85 10. **To the readers: N9 and T75.**
- **Readers** (`bench/blind/loop_r1`, 20 prompts, seed 20261211; packets committed first, 36b87324). The lead gave each of the four packet files to its own fresh rater, not two raters reading two files each. Scored as registered, with "rater X" = the R1 and R3 replies and "rater Y" = R2 and R4 (each covers all 20 prompts once):
  - **T 0.75: 0.625 [0.450, 0.775]**, X 0.60, Y 0.65: **leads, not significant** (the interval includes 0.5). By the round's loose rule it is carried forward as the provisional best. Nothing is adopted until a confirmation (all 50 prompts, four raters, against START) passes §13.38;
  - nucleus 0.9: 0.425 [0.275, 0.575], X 0.40, Y 0.45: no.
- SCRIBE v2 ranked them the same way the readers did; so did the link, splice and verbless. N9 had the lower stuff3. Too few rated arms to say more.
- Round 2 (gate) starts from T 0.75, the provisional best. The round-3 confirmation tests the round-3 best against START. Gate arms run speculatively from START while the readers read were stopped and are not used.

### Round 2: gate (from T 0.75)

Full table: `bench/decode_loop/rounds/r2.screen.md`. The WSL machine restarted at about 02:53 (not this job). It killed the last arm (lda_w 2), which was rerun whole. No other arm was affected.

| | T 0.75 (best) | wf 0.5 | wg 0.5 | lda_w 0.5 | lda_w 2 |
|---|---|---|---|---|---|
| mean surprise (bits) | 7.70 | 7.75 | 7.48 | 7.70 | 7.67 |
| words a sentence | 13.1 | 12.7 | 12.8 | 13.2 | 13.0 |
| link to the sentence before | 0.357 | 0.322 | 0.291 | 0.360 | 0.342 |
| verbless | 0.221 | 0.215 | 0.203 | 0.199 | 0.232 |
| stuff3 | 0.029 | 0.029 | 0.027 | 0.030 | 0.027 |
| splice | 0.212 | 0.205 | 0.214 | 0.217 | 0.216 |
| tail | 0.166 | 0.151 | 0.165 | 0.164 | 0.164 |
| SCRIBE v2 | 0.844 | 0.843 | 0.859 | 0.843 | 0.823 |
| copy32 | 0.525 | 0.557 | 0.522 | 0.528 | 0.523 |
| rated texts unlike the best (of 50) | – | 29 | 31 | 20 | 23 |

- **Screen:** all four pass (verbatim13 0, fact precision 0.938). **Ranking:** wg 0.5 3, wf 0.5 8, lda_w 2 9, lda_w 0.5 10. **To the readers: wg 0.5 and wf 0.5.**
- **Readers** (`bench/blind/loop_r2`, seed 20261212, packets committed first, 419e96aa; four fresh raters, one file each):
  - **wf 0.5: 0.700 [0.575, 0.825]**, X 0.75, Y 0.65: **the new (provisional) best.** Its interval excludes 0.5, but it is a 20-prompt round against T 0.75, not a confirmation;
  - wg 0.5: 0.500 [0.325, 0.675], X 0.55, Y 0.45: no.
- **Secondary** (the lead's request; the decision stays on the registered score): many pairs showed the same text twice (21 of the 50 rated prompts are unchanged by wf, 19 by wg), and the raters tied them. On the differing items only (`loop_diffscore.py`): wf 0.5 **0.808 [0.615, 0.962]** (13 prompts), wg 0.5 0.500 [0.250, 0.750] (14). Round 1 the same way: T 0.75 0.656 [0.438, 0.844] (16), nucleus 0.9 0.406 (16).
- The ranking put wg first by SCRIBE v2 and its low mean surprise; the readers did not prefer it. wf had the lowest splice and tail shares of the four.

## Amendment 2: packets from prompts whose texts differ (2026-11-13, before round 3's packet)

- **Why:** in round 2, 19–21 of the 50 rated prompts gave the same text for the best and the candidate, and the raters tied them (the lead's note). Identical pairs carry no signal and pull every share toward 0.5.
- **From round 3 on:** round r walks the 50 rater prompts in `prompts.json`'s order from position 20(r−1) mod 50, wrapping. It takes the first 20 where every survivor's text (as cut for the raters) differs from the best's. If fewer than 20 qualify, it adds prompts where at least one differs (`loop_round.py`, `rotation_differing`).
- Confirmations keep all 50 prompts, as registered (they are the comparable strict test). Their score on the differing items only is reported beside them as secondary.
- The round rule, the seeds and everything else stay as registered.
- What had run when this was written: round 3's candidates (retrieval, from wf 0.5) were being written; none was measured or read.
