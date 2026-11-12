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
