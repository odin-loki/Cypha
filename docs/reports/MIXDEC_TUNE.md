# D6: tuning the decoder (more experts, more synergy)

**Odin Loch** · 2026-11-13 · branch `wip/mixdec-tune` (from `wip/mixdec-t8` 42206a76, D5) · design: [../DECODER.md](../DECODER.md) · D4: [K5_MIXDEC.md](K5_MIXDEC.md) · GENERATION_PLAN §13.38, §13.72

**Odin:** "we should go on with tuning the decoder. More experts, more synergy."

(Each part below was written and committed before the runs it governs. Results are added under it.)

## Why

- D4's decoder guides K5's words with a byte learner. Raters preferred it to K1 0.750–0.760.
- Its learner is light. It reads about 2.0 bits a byte on dev continuations; `context8k`'s full stack reads 1.4786 on its own holdout (§13.72).
- So: a stronger learner, better matched to the context, and better-tuned guidance. Readers judge.

## Part 1: the learner's bits a byte (registered before any measurement)

A cheap, offline step. Each change is measured alone on the learner's bits a byte. Only the changes that lower it go into writing arms.

**The measure.**
- The 250 GEN-A dev prompts and their human continuations (about 511 bytes each, 128 KB).
- The learner is primed as a continuation's learner is: the base (512 KiB of generic prose), then the retrieved article (its first 8,192 bytes), then the prompt. Then it reads the continuation frozen, as it reads its own text when writing.
- The articles are K5's own retrieval, dumped once (`CYPHA_MIXDEC_DUMP`, the first 4 documents of each prompt).
- Reported: the primed learner's bits a byte (**the number that decides**), the neutral learner's (the base alone), and the gain between them. The ratio arms use the gain.
- The learner is measured as the ratio arms run it: without the copy expert (`copy=0`).
- Tool: `CYPHA_MIXDEC_EVAL=250 CYPHA_MIXDEC_ARTS=DIR CYPHA_MIXDEC_EVAL_ONLY=1 cyphalm_trie sample … --mixdec SPEC` (new; off unless set).

**The changes** (new `--mixdec` fields, off by default; with all of them off the learner is the same to the last bit):

| Name | Field | What |
|---|---|---|
| M0 | — | D4's learner (`ratio=0.3,copy=0`) |
| CLS | `cls=1` | the class expert (word shapes, counted on K5's 7 MB). It is a function of the text read, so it rewinds for free. |
| DOC | `doc=1` | the document cache (DocTopic's cache part, cand 2048 as T1 tuned it). It learns the article and the prompt, then is frozen; a mark keeps its partial word and last bytes. |
| NG | `ngo=1:2:3:4:5:6:7:8:12:16,ngcap=8,nglr=0.000625` | T1's tuned n-gram |
| KN | `bkn=10:4:2:1` | T1's tuned Kneser-Ney |
| MX | `floor=1.67652e-05,mlr=0.00181686,decay=0.0001` | T1's tuned arm B (the full mix's geometric mixer, which the learner's mixer is) |
| T1 | `t1=1` | NG, KN and MX together |
| MIX2a | `mix2=1` | a second mixer, weight sets by the previous byte (256), joined to the first by a final geometric mixer |
| MIX2b | `mix2=2` | the same with an order-2 hash (4,096 sets) |
| PRE | `ngpre=8388608,ngtb=20` | the n-gram's counts pretrained on 8 MiB of generic prose (after the base's 512 KiB) |
| GEN | `--mixdec-generic-bytes 2097152` | the base primed on 2 MiB, not 512 KiB |
| ART4 | `artk=4,art=32768` | the learner primed on the 4 best retrieved documents (K5's own counts keep their 1) |
| LSTM0 | `lstm=0` | the LSTM left out: how much it carries |
| LSTMx | `--mixdec-lstm` (a new file) | the hidden-64 LSTM trained on 32 MiB of clean prose, not 7 MB |

Not built, and why:
- **The similar-context expert.** Its `predict` keeps mutable caches and online keys. It cannot be shared by six threads, and copying it per continuation copies its whole index. Its online keys from the text read are what the copy expert already gives.
- **InfiniGram's caps.** They need a suffix array of a training text. Over K5's 7 MB they would add little to the 1 TB trie.
- **A larger LSTM (hidden 128).** The learner holds the hidden-64 type; a second type needs a variant in every mark. LSTM0 says first how much the LSTM carries. If it carries more than 0.02 bits a byte, a larger LSTM is the next step; if not, it is not worth it.
- **The meta mix's bitwise weight sets.** Its contexts select by bit position inside a byte. The learner mixes whole distributions, so MIX2a and MIX2b take the meta mix's two best byte-level contexts (order 1, order-2 hash).

**The rule.** A change is kept if it lowers the primed learner's bits a byte by at least 0.005 against M0. Then the kept changes are measured together (ALL). The cost per byte distribution is reported beside each (ms, from the run's wall time).

**Predictions.**
- M0 near 1.95 primed (D4's smoke: 1.975 on the prompt alone, 12 prompts).
- NG, MX, T1: −0.01 to −0.04. PRE: −0.02 to −0.06 (the n-gram has seen only 512 KiB).
- CLS: −0.005 to −0.02. DOC: 0 to −0.01 (the article's words are mostly in the trie).
- MIX2a/b: −0.005 to −0.02.
- ART4: −0.01 to −0.03 primed, and a larger gain.
- GEN: −0.005 to −0.02.
- LSTM0: +0.01 to +0.03 (the LSTM carries little). LSTMx: −0.005 to −0.02.
- ALL: −0.05 to −0.12, so about 1.85 primed. Still far above `context8k`'s 1.48.

## Part 1 results (2026-11-13)

Dev, 250 prompts, 130,183 continuation bytes. Paired bootstrap over prompts (2,000 resamples), byte-weighted. Files: `bench/mixdec_tune/bits/` (each run's per-prompt rows).

| Change | primed bits a byte | − M0 [95%] | neutral | gain (neutral − primed) | kept |
|---|---:|---:|---:|---:|---|
| M0 | 1.9514 | — | 2.0365 | 0.0851 | |
| CLS | 1.9325 | −0.0189 [−0.0208, −0.0170] | 2.0138 | 0.0813 | yes |
| DOC | 1.9595 | +0.0082 [+0.0036, +0.0129] | 2.0291 | 0.0696 | no |
| NG | 1.9453 | −0.0060 [−0.0084, −0.0036] | 2.0521 | 0.1068 | yes |
| KN | **1.8960** | **−0.0553** [−0.0584, −0.0521] | 1.9806 | 0.0846 | yes |
| MX | 1.9635 | +0.0122 [+0.0094, +0.0151] | 2.0392 | 0.0757 | no |
| T1 (NG + KN + MX) | 1.8995 | −0.0518 [−0.0565, −0.0470] | 1.9946 | 0.0951 | (MX hurts) |
| MIX2a | 1.9492 | −0.0022 [−0.0038, −0.0006] | 2.0373 | 0.0881 | no |
| MIX2b | 1.9419 | −0.0095 [−0.0116, −0.0074] | 2.0238 | 0.0818 | yes |
| PRE | 1.9451 | −0.0063 [−0.0087, −0.0041] | 2.0217 | 0.0766 | yes |
| GEN | 1.9504 | −0.0010 [−0.0040, +0.0022] | 2.0606 | 0.1103 | no |
| ART4 | 1.9391 | −0.0123 [−0.0185, −0.0064] | 2.0365 | 0.0974 | yes |
| LSTM0 | 1.9514 | −0.0000 [−0.0010, +0.0011] | 2.0323 | 0.0809 | (ablation) |
| LSTMx | 1.9489 | −0.0025 [−0.0035, −0.0015] | 2.0295 | 0.0806 | no |
| **ALL** (CLS, NG, KN, MIX2b, PRE, ART4) | **1.8327** | **−0.1187** [−0.1253, −0.1119] | 1.9385 | **0.1058** | |
| ALL without ART4 | 1.8523 | −0.0990 [−0.1028, −0.0951] | 1.9385 | 0.0862 | |

- **ALL lowers the learner's bits a byte by 0.119** (1.951 → 1.833), and its gain from the context rises from 0.085 to 0.106. The ratio arms write with that gain.
- **Kneser-Ney carries most of it.** T1's 10:4:2:1 (order 10) alone is −0.055.
- **The LSTM carries nothing** (LSTM0 ±0.001). By the registered rule a larger LSTM is not built. Training it on 32 MiB did not help either (−0.0025).
- The document cache and T1's mixer settings make it worse. T1's mixer settings were tuned for a mixer that reads 90 MB; this one reads 520 KB, and its step decay slows it.
- The second mixer pays only with the order-2 hash (MIX2b).
- **Two additions after the registered list, stated here:** ALL without ART4, to separate the learner from the priming; and LEAN (CLS, KN, PRE, ART4: the cheaper changes), measured because ALL costs about twice M0 per distribution. LEAN reads 1.8680 with a gain of only 0.0783, below M0's. The n-gram's orders carry the gain. LEAN is not used.
- **The final binary** (the second mixer's log probabilities taken directly, a speed change): M0 and ALL measured again on the rebased build: 1.9514 and 1.8327, gain 0.0851 and 0.1058, as before to four places.
- **Cost.** The evaluation's own seconds (6 threads, shared PC: loads changed between runs) are too noisy to rank the changes (LEAN without ART4 took 87 s, ALL without ART4 54 s, M0 26 s). Loading and priming: M0 100 s, ALL 243 s (Kneser-Ney order 10, the class table, 8 MiB of n-gram counts).

**Predictions:** M0 near 1.95: **hit** (1.951). NG −0.01 to −0.04: **miss** (−0.006). MX −0.01 to −0.04: **miss** (+0.012). T1 −0.01 to −0.04: **miss** (−0.052; the gain is Kneser-Ney's). PRE −0.02 to −0.06: **miss** (−0.006). CLS: **hit** (−0.019). DOC 0 to −0.01: **miss** (+0.008). MIX2a/b −0.005 to −0.02: **miss** / **hit**. ART4 −0.01 to −0.03 and a larger gain: **hit** (−0.012; gain 0.097). GEN: **miss** (−0.001). LSTM0 +0.01 to +0.03: **miss** (0.000). LSTMx: **miss** (−0.0025). ALL −0.05 to −0.12: **hit** (−0.119).

## Part 2: the writing arms (registered before any arm runs)

**Base.** `wip/mixdec-t8` (D5, 42206a76): today's default T8 with D4's decoder merged. D5's branch appeared while part 1 ran, so part 1 ran on `wip/k5-mixdec` and the branch was rebased onto D5's before any arm (no conflicts). Selftest passes on the rebased build.

**Baseline: T8MR** (T8 + `--mixdec ratio=0.3,copy=0`, D5's spec). D5's raters have not run, so I take T8MR, as the job says by default. The judge (MRJ) is not used: it cost 37 minutes against 23 in D5's checks, and it moved no screen number in D4.

**Checks** (before the arms are screened; `cmp` of the GEN lines, MODEL line aside):
- with no `--mixdec`, the build equals T8 (`bench/k5_decode/runs/T8.gen.txt`);
- T8MR from this build equals D5's T8MR GEN file (when D5 commits it; else my T8MR is the baseline and this is stated): the tuning fields off leave the learner unchanged.

**Arms** (dev, seed 0, 250 prompts, `scripts/gen_eval.sh dev LABEL OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt` plus):

| Arm | `--mixdec` | What it tests |
|---|---|---|
| T8MR | `ratio=0.3,copy=0` | the baseline |
| TL | `ratio=0.3,copy=0,` + L | the stronger learner (ALL), guidance unchanged |
| TL5 | `ratio=0.5,copy=0,` + L | the stronger learner with stronger guidance |
| TL2 | `ratio=0.2,copy=0,` + L | the stronger learner with weaker guidance (against fragments and repetition) |
| TM5 | `ratio=0.5,copy=0` | D4's learner with stronger guidance: separates the exponent from the learner |

L = `cls=1,bkn=10:4:2:1,ngo=1:2:3:4:5:6:7:8:12:16,ngcap=8,nglr=0.000625,mix2=2,ngpre=8388608,ngtb=20,artk=4,art=32768`.

- **Not run: 32 scored candidates.** It doubles the scoring, and TL already costs about twice T8MR's distributions. With D5's T8MR at 28 minutes on this shared PC, a top-32 arm with L would take near 2 hours.
- **Cost:** T8MR and TM5 should take 20–30 minutes. TL, TL5 and TL2 will pass the ~30-minute target: I expect 40–60 minutes each (about twice the distributions' cost, plus 2.5 minutes to load). This is stated, not hidden; it is the price of the experts that lowered the bits.
- One arm at a time, at most 6 threads, none started while `free -g` shows under 16 GB available.

**Screen.** An arm passes if all hold:
- verbatim13 is 0;
- fact precision at least 0.874 (`python3 -I scripts/gen_fact_check.py run --v2`);
- **my strict reading**, D5's protocol against T8MR, not T8: the 12 unrated dev prompts D2 read (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), first 460 characters, each arm beside T8MR, the cut last sentence not counted. Each sentence is whole (a subject and a finite verb), a verbless fragment, or neither. The arm passes only if
  - its share of whole sentences is at least T8MR's;
  - its share of verbless fragments is at most T8MR's + 0.05 (D4's weak point);
  - no keyword patchwork: the subject's word is not repeated in three sentences running in more than 2 of the 12 prompts (D4's other weak point);
  - and it is no worse than T8MR in any other way I see (broken clauses, lists of names, phrases pasted from the article, leaving the subject).
- Reported, not screened: SCRIBE v2 and v3.1, copy32, mean surprise, on_prev and stuff3 (`bench/para_plan/hook_screen.py`, 460 characters, against T8MR, 95% paired bootstrap), sentences in 460 characters, wall time and peak RAM.
- At most 2 candidates. If more pass: the two with the most whole sentences in my reading, then the higher SCRIBE v2. If none passes, nothing goes to the raters, and I say so.

**Judge.** `python3 -I scripts/blind_rate.py make --out bench/blind/mixtune_pair --ids bench/blind/llm_rating/prompts.json --arm T8MR=… --arm X=… [--arm Y=…] --mode pair --baseline T8MR --seed 20261208`: the 50 prompts, 460 characters, four fresh blind raters. Packets, `key.b64`, `RATER_PROMPT.md` and the GEN files are committed and pushed before the raters run; `KEY.json` stays off the branch. The lead runs the raters. §13.38's rule: an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign. A winner goes to a 1–10 round (§13.69, §13.74) before any default changes; any default is Odin's call.

**Predictions.**

| # | Prediction |
|---|---|
| Q1 | Both checks equal (`cmp`) |
| Q2 | verbatim13 0 in every arm; fact precision within ±0.03 of T8MR's, all above 0.874 |
| Q3 | TL: SCRIBE v2 within ±0.03 of T8MR; copy32 within ±0.03; fewer verbless fragments than T8MR in my reading (a learner 0.12 bits better scores broken words lower) |
| Q4 | TL5 and TM5: stuff3 up against T8MR by 0.005–0.02, more sentences in 460 characters; at least one of them fails my reading on keyword repetition |
| Q5 | TL2: closest to T8 in reading; mean surprise up against T8MR |
| Q6 | TL passes the screen and is a candidate |
| Q7 | Raters: the best candidate 0.50–0.62 against T8MR; at most one reads better by the rule |
| Q8 | Wall: T8MR and TM5 20–30 minutes; TL, TL5, TL2 40–60 minutes |

## Status (2026-11-13, 04:20; stopped at Odin's request)

- **Check 1 done:** with no `--mixdec`, the rebased build equals T8 (`bench/k5_decode/runs/T8.gen.txt`) under `cmp`. SCRIBE v2 0.811 [0.785, 0.838], copy32 0.487, verbatim13 0; 17:02 wall on a loaded PC (`bench/mixdec_tune/checks/`).
- **Check 2, partly:** a T8MR run from this build wrote a GEN file equal to D5's T8MR GEN file under `cmp` (the tuning fields off leave the learner unchanged). WSL went down before its copy32 line and SCRIBE, so it is not a finished arm.
- **No arm has finished.** WSL went down three times (02:53, about 03:55, about 04:08) while five or more 10 GB jobs from several agents ran at once. Each time my run was lost with the rest.
- **Queue, when resumed:** T8MR, TL, TM5, TL5, TL2 (`bench/mixdec_tune/scripts/mt_queue_arms.sh`; it skips arms with a summary). Each arm runs under `~/heavy.sh` (2 global slots, 16 GB available), 6 threads. Then the screen (`mt_screen.sh`), my reading, and the packets in `bench/blind/mixtune_pair/` (seed 20261208).
