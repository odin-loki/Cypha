# D4: CyphaLM's decoder (K5 with the mixer)

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/k5-mixdec` (from main 4629f68c) · design: [../DECODER.md](../DECODER.md) · research: DECODER_RESEARCH.md (`wip/decoder-research` 7b5baf8b) · GENERATION_PLAN §13.11, §13.24, §13.38, §13.57, §13.66, §13.69

(The method, the arms, the screen, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- Odin: "What about the actual algorithm? Like a decoder? A mixer to write." Then: "Lets make a mixer decoder and an architecture like a LLM architecture … We should do the reverse of compression."
- CyphaLM's best predictor (`context8k`, 1.4900 bits a byte, §13.55) has never written a byte. Today's writer, K1 (§13.66), reads 2.28 of 10 to blind raters; human text 8.54 (§13.69).
- Every expert blended into K5's sampling distribution so far lowered bits and made the text worse (§13.11, §13.24). So the learner enters as a context ratio, a veto or a judge, never as a plain blend (DECODER.md §4).

## The method

New flag, off by default, byte-identical when off: `cyphalm_trie sample --k5 … --mixdec SPEC`. Code: `native/include/cypha/lm/decoder/{arith,learner,mixdec}.hpp` (new), with hooks:
- `lm/word_kn.hpp`: `ByteScorer::decode` and `ByteScorer::rerank` (virtuals; false and true by default, as before), `Options::scorer_top` (0 by default: every word scored, as §13.24), `MixDec` (an interface), `Options::mixdec`, `State::mix`, the article's text in `start()`, a fork per draft in `gate_z()` and `draft()`, the judge's term in `gate_z()`, the veto as scorer mode 4, and the decode hook where `fast_pick` and `full_pick` draw a word.
- `lm/planner/discourse.hpp`: the judge's term in `free_sentence()`.
- `online/online_ngram.hpp`: `history()` and `set_history()` (read and set the 64-byte history; nothing else uses them).
- `tools/lm/cyphalm_trie.cpp`: the flags, the models' loading, the selftest.

The learner, the priming, the freezing and the three ways in are in DECODER.md §4–7. Its files: the 1 TB trie (`/mnt/d/cypha_llm/corpus/trie/trie_1tb.pk`), byte Kneser–Ney counted on `--k5`'s text when the run starts, the byte LSTM `/mnt/d/cypha_llm/mixdec/clean64.lstm` (`cyphalm_trie lstm-train /mnt/d/cypha_llm/fuzzy/train.txt`, hidden 64, 7 MB online, last MiB 2.55 bits a byte), and the base's priming text: the first 512 KiB of `/mnt/d/cypha_llm/k5scale/clean/train_all.txt`.

`SPEC` fields: `ratio=R` (the context ratio, weight R), `veto=V` (the veto, V bits), `judge=J` (the judge, weight J), `ad=0|1` (arithmetic decoding; on whenever a word term is on), `art=N` (article bytes primed, 8,192), `top=N` (the nucleus's N most probable words are scored, 16; 0: all), and `trie=0`, `kn=0`, `copy=0`, `ngram=0`, `lstm=0` to leave an expert out (not used by the arms).

What the decoder touches, and what not:
- Only K5's own sentences: the gate's drafts (`gate_z`, prompts with no plan) and the discourse writer's free sentences (`draft`). The planner's fact sentences are untouched; the learner reads them as text.
- The nucleus 0.8, the repeat ban and the 12-word cap are applied before the decoder sees a word. Every word written is one of K5's candidates (rule 10).
- With the judge alone (MJ) K5 draws its own words; only the gate's choice changes.

## The baseline and the checks

**K1**, today's default: `scripts/gen_eval.sh dev K1 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `cyphalm_trie` and `scribe` built by `g++ -O2` as `scripts/build_1tb_tools.sh` builds them (`libsais.o` linked into `scribe`), `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

**Checks, done before this was written:**
- main's build (4629f68c): the GEN file equals `bench/k5_retrieve/runs/K1.gen.txt` under `cmp`, MODEL line aside. **Equal.** SCRIBE v2 0.704 [0.673, 0.735], copy32 0.275, verbatim13 0; 9:46 wall, 11.3 GB.
- my first build (b9afa478) with the flag off (`cyphalm_trie sample`, 250 prompts): **equal** to the same file (7:58 wall, 11.3 GB).
- selftest passes: the coder's round trip (20 × 2,000 random distributions), its frequencies, the learner's exact rewind and copy.
- To be repeated with the final binary before the arms: the flag-off `cmp`.

**Before this was written: a smoke run, and what it changed** (stated so the reader can weigh it). To time the decoder I ran `cyphalm_trie sample` on dev prompts 0–11 (no SCRIBE), with 6 threads, on a machine shared with other jobs (load 100–155 on 64 cores). I read two to four samples of each, and looked at copy32 and, for the first run, `hook_screen.py`'s measures.

| Smoke run (12 prompts) | copy32 | wall | byte distributions | Note |
|---|---:|---:|---:|---|
| `ratio=0.3,judge=1`, every nucleus word scored | 0.456 | 15:34 | 12.2 M | on_prev 0.250 → 0.440, mean surprise +0.33 bits, against K1's same 12 |
| `ad=1` | 0.272 | 18:56 | 16.0 M | **a bug**: the scorer reranked with the default product (w 0.25); fixed (`rerank()`) |
| `ratio=0.3`, top 16 | 0.373 | 10:02 | 1.2 M | article phrases patched together |
| `ratio=0.3,copy=0`, top 16 | 0.252 | 12:06 | — | on the subject, sentences as broken as K1's |
| `veto=10`, top 16 | 0.277 | 10:12 | 1.3 M | reads like K1 |
| `judge=1` | 0.336 | 11:05 | 0.23 M | |

- The learner's bits a byte on those 12 prompts' human continuations (`CYPHA_MIXDEC_EVAL=12`): **2.011** neutral, **1.975** primed on the prompt alone (−0.036). The base's priming read its 512 KiB at 1.923.
- **What I changed after it, before registering:**
  - scoring only the nucleus's 16 most probable words (`top=16`; the rest take the scored words' mean ratio, or are kept by the veto): scoring all of them cost 12.2 M distributions for 12 prompts, about 4 hours for 250;
  - the copy expert is left out of the learner in the ratio and judge arms (`copy=0`): with it, the ratio and the judge raise copy32 far above K1's (0.373, 0.336), because the copy expert makes the article's own phrases the most "explained" text. The veto keeps it (spelling and names);
  - the fix of the `ad=1` bug.
- No smoke run went through SCRIBE or the fact check. The arms below are fixed now and not changed again.

## The arms

Dev, seed 0, 250 prompts, K1's command plus:

| Arm | Flags | What it tests |
|---|---|---|
| MJ | `--mixdec judge=1,copy=0` | the judge in the gate; K5 draws its own words |
| MR | `--mixdec ratio=0.3,copy=0` | the context ratio inside K5's nucleus (top 16), arithmetic decoding |
| MV | `--mixdec veto=10` | the veto inside K5's nucleus (top 16), arithmetic decoding |
| MRJ | `--mixdec ratio=0.3,judge=1,copy=0` | the whole decoder: the ratio and the judge |

**A check, not an arm:** MA, `--mixdec ad=1` (arithmetic decoding of K5's own nucleus, no learner term). It should score like K1: decoding random bytes is sampling. It is screened and reported, not rated.

Each arm runs alone, as `~/mixdec_arm.sh` (it refuses to start under 16 GB available). If an arm's wall time passes 6 hours, it is stopped and screened on the prompts it finished, and that is stated.

## The screen

An arm passes if:
- verbatim13 is 0;
- copy32 is at most 0.30;
- fact precision (`python3 -I scripts/gen_fact_check.py run --v2`) is at least 0.874;
- SCRIBE v2 is at least K1's − 0.03;
- mean surprise (WRITING_SIGNAL's word-trigram surprise, `bench/para_plan/hook_screen.py`, 460 characters) is at most 0.5 bits a word below K1's (the guard against bland text);
- and it reads at least as well as K1 to me, strictly: no word salad or broken clauses that K1 does not have, no keyword patchwork, no runs copied from the article.

Reported beside it: wall time and peak RAM, the link to the sentence before and stuff3 (`hook_screen.py`), the words a sample, the share of samples unlike K1. The leak check is K1's (the retrieval is unchanged). At most 2 candidates go to the raters. If more pass, the two with the highest SCRIBE v2 among those I read as best, one per way in where possible.

## The judge

Blind pairwise raters against K1 at 460 characters: `scripts/blind_rate.py make --mode pair --baseline K1`, the 50 prompts of `bench/blind/llm_rating/prompts.json`, seed 20261203, packets in `bench/blind/mix_pair/`. Four fresh blind raters; packets and `key.b64` committed before the raters run; `KEY.json` kept off the branch. §13.38's rule: an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign. The best of those that pass (higher lower bound, then lower copy32) goes to a 1–10 round beside K1 and human text (§13.69's method) before any default changes; that, and any default, is Odin's call. The lead runs the raters.

## Predictions

- With the flag off, the GEN file equals K1's (`cmp`): hit expected.
- MA's SCRIBE v2 is within ±0.03 of K1's and its copy32 within ±0.02.
- MJ: copy32 within ±0.03 of K1; SCRIBE v2 within ±0.03; on_prev up by 0.02–0.08; raters 0.50–0.60 (not adopted).
- MR: copy32 within ±0.03 of K1; SCRIBE v2 −0.06 to +0.02; on_prev up 0.03–0.10; mean surprise up; raters 0.40–0.55.
- MV: SCRIBE v2 within ±0.02, copy32 within ±0.02; reads like K1; raters 0.45–0.55.
- MRJ: like MR, with on_prev up a little more.
- At least one arm fails the screen on my reading (broken clauses no better than K1's).
- No arm reaches a win share of 0.70. If one is adopted, the 1–10 round moves by less than 0.5 point.
- Wall time: 1.5–4 hours an arm on this shared machine (K1: 10 minutes).

## Results (2026-11-13; dev, seed 0, 250 prompts)

**Checks.** The final binary (060b4e52) with the flag off equals K1's GEN file under `cmp` (9:40 wall, 11.3 GB). The selftest passes. MA, the check, scores like K1, as predicted: decoding random bytes is sampling.

| Arm | SCRIBE v2 [95%] | v3.1 | copy32 | verbatim13 | fact precision [95%] | mean surprise − K1 (bits a word) [95%] | on_prev − K1 [95%] | stuff3 − K1 [95%] | sentences in 460 chars | wall | peak RAM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 0.704 [0.673, 0.735] | 0.665 | 0.275 | 0 | 0.918 [0.867, 0.962] | — (7.806) | — (0.232) | — (0.0214) | 5.30 | 9:46 | 11.3 GB |
| MA (check) | 0.712 [0.682, 0.742] | 0.667 | 0.280 | 0 | 0.935 [0.888, 0.974] | −0.061 [−0.134, +0.014] | +0.020 [−0.006, +0.048] | −0.0010 | 5.25 | 11:17 | 12.5 GB |
| MJ | 0.698 [0.668, 0.729] | 0.674 | 0.273 | 0 | 0.924 [0.874, 0.966] | −0.058 [−0.127, +0.011] | −0.020 [−0.039, −0.001] | +0.0008 | 6.04 | 15:09 | 12.5 GB |
| MR | **0.810** [0.784, 0.836] | 0.725 | 0.277 | 0 | 0.918 [0.867, 0.962] | −0.211 [−0.305, −0.115] | +0.015 [−0.016, +0.045] | **+0.0161** [+0.0117, +0.0209] | 6.72 | 21:03 | 12.6 GB |
| MV | 0.715 [0.686, 0.744] | 0.649 | 0.254 | 0 | 0.935 [0.888, 0.974] | −0.336 [−0.412, −0.258] | +0.019 [−0.006, +0.044] | +0.0005 | 5.32 | 20:01 | 12.8 GB |
| MRJ | **0.802** [0.778, 0.829] | 0.739 | 0.268 | 0 | 0.918 [0.870, 0.959] | −0.122 [−0.213, −0.031] | +0.001 [−0.027, +0.032] | **+0.0227** [+0.0171, +0.0284] | 7.60 | 20:22 | 12.6 GB |

Files: `bench/k5_mixdec/runs/` (GEN, SCRIBE, summaries, the decoder's counters, `time -v`), `bench/k5_mixdec/screen/facts/` (`gen_fact_check.py run --v2`), `bench/k5_mixdec/screen/hook.md` (`hook_screen.py`, 460 characters, 95% paired bootstrap).

**Cost.** The decoder's arms took 15–21 minutes for 250 prompts on 6 threads (K1: 10), and about 1.3 GB more RAM (the 1 TB trie, Kneser–Ney, the learners). About 1.5 minutes of each is loading the models and priming the base (512 KiB at 1.92–1.94 bits a byte). The ratio and veto arms computed 27–36 million byte distributions, about 0.23 ms each on one thread; the judge alone 5.9 million. The learner's bits a byte on the human continuations of 12 dev prompts: 2.011 neutral, 1.975 primed on the prompt (smoke run). It is far weaker than `context8k` (1.49 on its own holdout): the price of a learner that can rewind.

**Every arm passes the numbers** (verbatim13 0, copy32 ≤ 0.30, fact precision ≥ 0.874, SCRIBE v2 ≥ 0.674, mean surprise no more than 0.5 bits below K1). The decoder never changes a planned fact sentence: on planned prompts the first 460 characters are often K1's own.

**My reading** (the 50 rated prompts' first 460 characters, side by side; 11 read closely, the rest skimmed):
- **No arm writes grammatical sentences.** All read as K1 does: sentence-shaped word salad.
- **MR and MRJ stay on the prompt's subject.** K1 drifts within a sentence or two; MR keeps to it.
  - Monetary policy. K1: "Monetary policy in a church. … the British invaded by a policy of " one China"". MR: "A central idea of monetary policy. Monetary stability. … This entails managing the quantity of money in circulation, and setting reserve."
  - Archery. K1 drifts to coups and "6,000 Ghost". MR: "The identities of the top eight archers. The team, with that archery. All six events were for men."
  - Adjectives. K1: "the term " Achaeans " derives from the French words of core memory". MR: "Examples of adjectives, the adjectives light. The most widely recognized adjectives). In both Latin and Greek."
- **The price is fragments and repetition.** MR writes more, shorter sentences (6.7 in 460 characters against 5.3), many without a verb ("Monetary stability.", "In both Latin and Greek."). It repeats the subject's word more (stuff3 0.037 against 0.021): "Cadillac. All of this, … Cadillac production cars. Cadillac." MRJ more so (7.6 sentences, stuff3 0.044). By my strict reading this is keyword repetition at the edge, not the patchwork that failed before (§13.34, §13.65): most of its sentences add a different phrase from the subject's field. I pass MR and MRJ, and state the doubt.
- **SCRIBE v2 rose by 0.10** for MR and MRJ. SCRIBE has rewarded topic words that readers did not (§13.34, §13.53 R2, §13.65), so this is not evidence by itself.
- **MV reads like K1**, drifting the same way ("Cadillac is a brand of luxury automobile, part of the Church was formed on the coast of Japan"). Its mean surprise is 0.34 bits lower: the veto removes rare words more than wrong ones.
- **MJ reads like K1.** The judge changes the gate's choice in most prompts but not the kind of text.
- **Copying did not rise** with the copy expert left out (0.268–0.277 against 0.275). With it (smoke run) it rose to 0.34–0.46.
- **The link to the sentence before (on_prev) did not move** in any arm (±0.02). The smoke run's +0.19 came with the copy expert and the article's phrases.

**Predictions:**
- flag off equals K1: **hit**;
- MA within ±0.03 SCRIBE and ±0.02 copy32: **hit** (+0.008, +0.005);
- MJ: copy32 ±0.03 **hit**; SCRIBE ±0.03 **hit**; on_prev +0.02 to +0.08 **miss** (−0.020);
- MR: copy32 ±0.03 **hit**; SCRIBE −0.06 to +0.02 **miss** (+0.106); on_prev +0.03 to +0.10 **miss** (+0.015); mean surprise up **miss** (down 0.21);
- MV: SCRIBE ±0.02 **hit** (+0.011); copy32 ±0.02 **miss** (−0.021); reads like K1 **hit**;
- MRJ like MR with on_prev up more: **miss** (+0.001);
- at least one arm fails my reading: **miss** (all pass; MR and MRJ with a stated doubt);
- 1.5–4 hours an arm: **miss** (15–21 minutes: the cap of 16 scored words, and a quieter machine);
- the raters' predictions are scored when they have run.

## Candidates and packets

- **Candidates: MR and MRJ.** They are the two I read as best (on the subject where K1 drifts) and the two highest SCRIBE v2. Both use the ratio; MRJ adds the judge, the only other way in that changed the text with it. MJ and MV read like K1 and would cost the raters a round for no expected change.
- **Packets:** `bench/blind/mix_pair/`: `scripts/blind_rate.py make --mode pair --baseline K1 --arm K1=… --arm MR=… --arm MRJ=… --seed 20261203`, the 50 prompts of `bench/blind/llm_rating/prompts.json`, 460 characters, with `key.b64` and `RATER_PROMPT.md`. `KEY.json` is kept off the branch.
- **The rule, as registered:** §13.38's, on the 460-character reading; then the 1–10 round for the winner, before any default changes. The lead runs the raters.
