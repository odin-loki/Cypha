# The generation track

**Odin Loch** · 2026-10-29 · PLAN ledger 72 · Odin's aim: CyphaLM writes coherent text

**Question.** CyphaLM answers questions, but its open text changes subject every few words and breaks its sentences. Today's default samples score 0.063 on SCRIBE v2 with A4d's prompts and 0.079 [0.064, 0.096] on GEN-A's dev benchmark; sentences shuffled across documents score 0.344; human text 0.923. Can we raise that without giving up rule 10?

## The four steps

| Step | Agent | What it does | Status |
|---|---|---|---|
| 1. Clean training text | GEN-A | `sample` trains on SCRIBE's markup-stripped prose of its own training slice | Done: harness and benchmark ready; clean text not adopted (dev SCRIBE 0.069 against 0.079), `--clean` stays a flag |
| 2. Sentence gate | GEN-B | A gate at sentence level: CyphaLM writes candidate sentences; a scorer picks | Done: not adopted (test SCRIBE 0.072 → 0.090 [0.076, 0.106], intervals overlap); `--gate` stays a flag ([report](GEN_B_SENTENCE_GATE.md)) |
| 3. Word and topic experts | GEN-C (topic), token track (words) | Document experts (PLAN B5); the word level is the token track (`docs/TOKENS_PLAN.md`) | GEN-C done: not adopted on bytes ([report](GEN_C_TOPIC_CACHE.md)) |
| 4. Content-plan expert | GEN-E | The fact store steers through a next-byte expert; CyphaLM writes every byte | Done: not adopted; `--plan` flag ([report](GEN_E_CONTENT_PLAN.md)) |

## Shared rules

1. **Rule 10.** A tool may only add training text, add a next-byte expert, or gate. CyphaLM writes every byte. Generation tools are off inside `qa` and chat answers.
2. **Metrics together.** SCRIBE v2 with its 95% interval, surface S, bits a byte on clean held-out text, and the copy share. Answers must not change.
3. **Register first.** The rule and the predictions are committed before any run.
4. **Dev for choices.** Every choice is made on the dev split.
5. **One test read per change.** Each read is logged in the Reads table below.
6. **Intervals.** Every number that decides something has a 95% interval.
7. No Qwen inference. `native/tools/trie_stream_train.cpp` and `scripts/run_train.sh` are not touched.

## Reads (frozen test split)

One row per read of `bench/eval/gen_prompts_test.tsv`. Dev runs are not logged here.

| Date | Change | Label | Command | SCRIBE v2 [95%] | Copy share | Decision |
|---|---|---|---|---|---|---|
| 2026-10-29 | Tokens T4: whole-token sampling from TOK-A's BPE-W 4,096 model at T 1.0 (chosen on dev) against the byte default at T 0.6 ([report](TOKENS_T3_T4.md), "T4") | TB; B0 | `cyphalm_tok_gen gen … t-bpe-4096.tok … --prompts-file bench/eval/gen_prompts_test.tsv --temps 1.0`; `scripts/gen_eval.sh test B0` (today's default) | TB 0.112 [0.086, 0.137]; B0 T 0.6 0.106 [0.078, 0.135] (pooled 0.072 [0.059, 0.085]) | TB 0.0746; B0 T 0.6 0.0561 (pooled 0.0386) | Not adopted: intervals overlap. Copy rule met (0.0746 ≤ 0.0761) |
| 2026-10-29 | Tokens T4b: the token sampler with its ∞-gram capped at 16 tokens when sampling, BPE-W 4,096 at T 1.0 (chosen on dev) against the byte default at T 0.6 ([report](TOKENS_T3_T4.md), "T4b") | K16; B0 (T4's run, reused) | `cyphalm_tok_gen gen … --prompts-file bench/eval/gen_prompts_test.tsv --inf-cap 16 --temps 1.0`, then `scribe ladder` and `gen_eval_summary.py` | K16 0.115 [0.091, 0.143]; B0 at T 0.6 0.106 [0.078, 0.135] | 0.0728 (B0 0.0561) | Not adopted: intervals overlap |
| 2026-10-29 | Tokens T4c: the byte sampler's nucleus 0.9 and repeat ban on tokens over Kneser–Ney alone (no ∞-gram), BPE-W 4,096 at T 1.0 (chosen on dev) against the byte default at T 0.6 ([report](TOKENS_T3_T4.md), "T4c") | NRK0; B0 (T4's run, reused) | `cyphalm_tok_gen gen … --prompts-file bench/eval/gen_prompts_test.tsv --inf-cap 0 --nucleus 0.9 --no-repeat --temps 1.0`, then `scribe ladder` and `gen_eval_summary.py` | NRK0 0.187 [0.155, 0.217]; B0 at T 0.6 0.106 [0.078, 0.135] | 0.0649 (B0 0.0561; limit 0.0761) | Gate met: intervals disjoint; a candidate for `sample`'s default (switch not made) |
| 2026-10-29 | GEN-E content-plan expert | E0 (baseline), E1 (`--plan`) | `gen_eval.sh test E1 OUT --plan /root/work/f2/v2/aug,/root/work/f2raw/patterns_raw.tsv --plan-drop 75497472:100000000` | E0 0.072 [0.059, 0.085]; E1 0.071 [0.059, 0.084] | D_I > 0: 0 → 0; 32-byte 0.0386 → 0.0401 | Not adopted: SCRIBE does not rise. `--plan` stays a flag |
| 2026-10-29 | GEN-B sentence gate: best of 4 CyphaLM sentences by fluency − 5 × markup share (chosen on dev) ([report](GEN_B_SENTENCE_GATE.md)) | genb_T1; B0 (reused; genb_T0 byte-identical) | `scripts/gen_eval.sh test genb_T1 OUT --gate 4,wm=5` | genb_T1 0.090 [0.076, 0.106]; B0 0.072 [0.059, 0.085] | 32-byte 0.0386 → 0.0557; D_I > 0: 0 → 0 | Not adopted: intervals overlap. Copy and surface S rules met. `--gate` stays a flag |
| 2026-10-29 | Tokens T4d: the byte default with byte Kneser–Ney (BKN2), its first read of this split, against T4c's token sampler NRK0 (T4c's test read reused) ([report](TOKENS_T3_T4.md), "T4d") | BK0; NRK0 (reused) | `CYPHALM_TRIE=cyphalm_trie_bkn2d scripts/gen_eval.sh test BK0` (main's default, no flags) | BK0 T 0.6 0.147 [0.115, 0.181] (pooled 0.115 [0.098, 0.133]); NRK0 0.187 [0.155, 0.217] | BK0 T 0.6 0.0769 (pooled 0.0458); NRK0 0.0649 | Token sampler not adopted: intervals overlap. The byte model with Kneser–Ney predicts the continuations 0.014 bits per byte better, paired |

## GEN-A

### Part 1: the frozen benchmark

**Source.** enwik8 [72 MiB, end), that is bytes [75,497,472, 100,000,000).

- CyphaLM's training text and holdout are [0, 8 MiB). They are not in this range.
- SCRIBE's pack [16, 48) MiB, reference [48, 56) MiB and dev test [56, 64) MiB are not in this range either.
- SCRIBE v2 checked itself once on [72, 95.37) MiB, its fresh region, after it was frozen. Nothing was tuned there.

**Prompts.** SCRIBE's own prompt maker builds the pool:

```
scribe prompts enwik:/root/cdata/enwik8:72:96 --n 100000 --seed 20261029 --set gen --out pool72.jsonl
```

- Each cleaned page gives at most one row: a prompt of 100–300 bytes and the true continuation of 400–600 bytes, with at least 3 sentences.
- The pool has 1,626 rows, in a seeded order.

**Contamination.** Some pages of enwik8 repeat text from earlier pages. A row is dropped when either holds:

- more than 10% of the continuation's bytes lie in 32-byte spans found in enwik8's first 8 MiB, raw or cleaned by SCRIBE;
- more than 25% of the prompt's bytes do.

The cleaned text comes from `scribe docs enwik:/root/cdata/enwik8:0:8 --out clean_0_8.jsonl` (620 pages, 4,125,159 bytes). The rule dropped 7 rows on the way to the dev rows and 6 on the way to the test rows.

**Split.** A row goes to dev when the FNV-1a 64 hash of its prompt's UTF-8 bytes is even, else to test. Each file keeps the first 250 clean rows of its half, in the pool's order.

**Builder.** `python3 -I scripts/gen_bench.py pool72.jsonl bench/eval 250 /root/cdata/enwik8 clean_0_8.jsonl`. A rebuild from scratch on 2026-10-29 gave the same pool, the same cleaned text and byte-identical files.

**Files.** Columns: id, doc (SCRIBE's page index in the range), prompt, continuation. Tab, newline and backslash are escaped as `\t`, `\n`, `\\`.

| File | Rows | Pages | Prompt bytes (mean, range) | Continuation bytes (mean, range) | sha256 |
|---|---:|---:|---|---|---|
| `bench/eval/gen_prompts_dev.tsv` | 250 | 250 | 172 (100–293) | 520 (408–600) | `e1bfa51d35f659e602f0ddd80c9a749a924ff196b0c09559515ac0b38e694625` |
| `bench/eval/gen_prompts_test.tsv` | 250 | 250 | 174 (100–293) | 522 (410–599) | `3819410288a512e6bc36066b38f5fec3b3d8e944b85c8153301f87fe419e55ce` |

No page is in both files. The inputs' hashes are `138dfccfdbb067aa649002462baeff77420975bbb5539bcd0fa269e3873c5608` (pool72.jsonl) and `55aa87016f09d532c4aee19ae35a18fc7e2865a2037f8f125b268fcf878d8f33` (clean_0_8.jsonl).

### Part 1: the harness

```
CYPHALM_TRIE=native/build/cyphalm_trie SCRIBE=native/build/scribe \
  scripts/gen_eval.sh dev|test LABEL OUTDIR [cyphalm_trie sample flags...]
```

It runs three things:

1. `cyphalm_trie sample /root/cdata/enwik8 OUTDIR/LABEL.gen.txt --prompts-file bench/eval/gen_prompts_SPLIT.tsv [flags]`. That is 250 prompts, T 0.6, 0.8 and 1.0, 600 bytes, and `sample`'s fixed seeds. Flags pass through, so any expert or gate can be scored.
2. `scribe ladder` with SCRIBE v2's control ladder, the same command as A4d and GEN-C: pack [16, 48) MiB, reference [48, 56), human and control rungs [56, 64), vectors `/root/work/scribe/vec32`, profile v2, seed 20261027.
3. `scripts/gen_eval_summary.py`, which writes `OUTDIR/LABEL.summary.md`.

**What it prints.**

- **SCRIBE v2** with its 95% interval (1,000 resamples), pooled and for each temperature.
- **Surface S** with its interval.
- **D_I**, SCRIBE's integrity axis (loops, repeats, word 8-grams copied from SCRIBE's pack).
- **Mean length** of the samples SCRIBE keeps (it drops samples under 100 bytes after its cleaning).
- **Continuation BPC.** `GEN_EVAL cont_bpc`: the bits a byte of each human continuation after its prompt, under sampling's mixture (no temperature). The interval comes from a bootstrap over prompts (2,000 draws). Per-row bits are in `LABEL.gen.txt.cont.tsv`.
- **Copy share.** `GEN_EVAL copy32`: the share of generated bytes that lie in a 32-byte span found in the training bytes [0, 7 MiB), raw or cleaned by SCRIBE. It is given per temperature and pooled. The reference is the same with and without `--clean`.
- **Five samples** at T 0.8.

**Paired BPC.** `python3 -I scripts/gen_eval_summary.py --pair OUTDIR LABEL_A LABEL_B` gives B minus A on the same rows, with a paired bootstrap interval.

**Other tools.** A tool that writes the same outputs can skip step 1 and run steps 2 and 3 as the script does. The outputs are PROMPT and GEN lines with kinds `trie_T0.6`, `trie_T0.8` and `trie_T1.0`, `OUT.cont.tsv`, and the GEN_EVAL lines. TOK-B's `cyphalm_tok_gen` does this.

**Unchanged outputs.** Without `--prompts-file` and `--clean`, `sample` writes byte-identical files. The scoring and the sampling now share one function (`sample_mix`), with the same order of experts. Checked at `91827e1`: `sample --sim default,online=1,probe=128` writes a dev file byte-identical to A4d's (`/root/work/a4d/runs/SO_dev.txt`, 492,276 bytes).

### Part 2: clean training text (registered 2026-10-29, before any run)

**Change.** `cyphalm_trie sample … --clean` replaces two ranges of `sample`'s text with SCRIBE's cleaned prose (`clean_wiki_documents`, the cleaner SCRIBE scores with), one page per line:

- the training bytes [0, 7 MiB);
- the holdout [7, 8 MiB), where the lambdas are fit and default prompts are drawn.

Pages cut by a range's ends are dropped. The trie, the similar-context keys, the copy expert and every lambda are then built on clean text only. Nothing else is added. No chat text is added, because the chat turns are questions and answers, not the prose the benchmark asks for. `qa`, `chat` and `context8k` do not read `--clean`.

**Arms on dev** (`scripts/gen_eval.sh dev`):

- **D0:** `sample` as it is today (similar-context expert, then copy).
- **D1:** the same with `--clean`.

**Rule.**

1. **Choice on dev.** Clean text is chosen if D1's pooled SCRIBE v2 is above D0's with disjoint 95% intervals, and D1's pooled copy share is at most D0's + 0.02.
2. **One test read**, if it is chosen: D0 and D1 on the test split, logged in the Reads table. `sample`'s default switches to clean text only if the test read meets the same two conditions. Otherwise `--clean` stays a flag, and both reads are reported.
3. Surface S, continuation BPC (paired), D_I and length are reported. They do not decide.
4. Nothing is changed after the test read.

**Predictions.**

| # | Prediction |
|---|---|
| G1 | D0's dev SCRIBE is within 0.03 of A4d's 0.063. The prompts are now clean prose, not raw holdout lines |
| G2 | D1's dev SCRIBE is above D0's |
| G3 | The dev intervals are disjoint, so rule 1 holds. I put this at about even odds |
| G4 | D1's surface S is at least 0.01 above D0's (no markup leftovers) |
| G5 | D1's continuation BPC is at least 0.05 below D0's, with the paired interval below 0. The raw model expects markup inside prose |
| G6 | Both copy shares are under 0.10, and D1 − D0 is within ±0.02 |
| G7 | SCRIBE drops fewer of D1's samples as too short (under 100 bytes after its cleaning) than of D0's |
| G8 | Without `--clean` and `--prompts-file`, `sample`'s files are byte-identical to before |
| G9 | If there is a test read, it agrees with dev on the direction of SCRIBE |

### Part 2: results (dev, 2026-10-29)

**Outcome.** Clean text removes the markup from the samples. Copying falls by about nine tenths, surface S rises, and the human continuations cost 0.029 fewer bits a byte. But SCRIBE v2 does not rise: 0.069 [0.056, 0.081] against the default's 0.079 [0.064, 0.096]. By the rule, clean text is not chosen. There is no test read, and `--clean` stays a flag. The sentences are as broken as before; the markup was not what held SCRIBE down.

Tool: `cyphalm_trie` at `91827e1` (wip/gena with main `bcd1d0d` merged). Command: `scripts/gen_eval.sh dev D0|D1 /root/work/gena/runs/dev [--clean]`. Wall times were 4,161 s (D0) and 3,146 s (D1), on a machine shared with about 12 other agents.

**Clean text.** SCRIBE's cleaner keeps 3,640,555 of the 7,340,032 training bytes and 474,934 of the 1,048,576 holdout bytes. So D1's trie learns from half as many bytes, all of them prose.

| Measure | D0 (default) | D1 (`--clean`) | D1 − D0 |
|---|---|---|---|
| SCRIBE v2, pooled [95%] | **0.079 [0.064, 0.096]** | 0.069 [0.056, 0.081] | −0.010 (intervals overlap) |
| SCRIBE at T 0.6 | 0.101 [0.072, 0.135] | 0.124 [0.094, 0.158] | +0.023 |
| SCRIBE at T 0.8 | 0.088 [0.059, 0.119] | 0.043 [0.029, 0.059] | −0.045 |
| SCRIBE at T 1.0 | 0.047 [0.031, 0.065] | 0.040 [0.026, 0.054] | −0.007 |
| Surface S, pooled [95%] | 0.855 [0.848, 0.862] | **0.878 [0.872, 0.884]** | +0.023 |
| Continuation BPC [95%] | 2.1339 [2.1016, 2.1659] | **2.1050 [2.0767, 2.1341]** | −0.0289 [−0.0378, −0.0205], paired |
| Copy share (copy32, pooled) | 0.0414 | **0.0049** | −0.0365 |
| Copy share at T 0.6 / 0.8 / 1.0 | 0.059 / 0.039 / 0.027 | 0.009 / 0.004 / 0.002 | |
| Samples SCRIBE keeps (of 750) | 618 | 667 | +49 |
| Mean bytes kept | 311 | 364 | +53 |
| D_S (sentence form) / D_C (coherence) | 18.16 / 4.68 | 18.57 / 5.24 | |
| D_I (SCRIBE's integrity axis) | 0.00 | 0.00 | |

- **What clean text does.** It removes markup. `[[` is in 749 of D0's 750 samples and page XML (`<text`, `<page`, `<id>`) in 175; D1's samples have none, and no `{{` or `&quot;`. SCRIBE keeps 49 more samples, because fewer shrink under 100 bytes when it strips markup. Surface S rises with disjoint intervals.
- **Copying.** The default copies 4.1% of its bytes in 32-byte runs of the training text. 93% of those bytes are in runs that hold markup or page XML (for example `comment>\n <text xml:space="preserve">{{Taxobox`). With clean text it copies 0.5%.
- **Why SCRIBE does not rise.** D_S (sentence form) is 18.2 to 18.6 in both arms, and D_C (coherence) rises a little. The byte trie still writes "the of the and" runs. Removing markup does not teach it grammar. At T 0.6 D1 is higher, at T 0.8 lower; pooled, it is lower.
- **SCRIBE's human rungs** are the same in both runs (human 0.923, sentences shuffled 0.344), as they must be.

**Samples (T 0.6, the same seeds).**

Prompt 0: "The first export of Holdens to New Zealand began in 1954, … until it closed in 1990."

- **D0:** "The [[United States States Standard of the term &quot;Apocrypha&quot; is understood subjective to the comment> <text xml:space="preserve">{{Taxobox | color = pink | name = Albatross | status and in a similar approach …"
- **D1:** "The product's of the and eastern America and the two and would have been involved concerns that the word century, and the twelve states of a second constitution …"

Prompt 3: "In lower concentrations (3%), it is used medically for cleaning wounds and removing dead tissue. …"

- **D0:** "The last [[Carthus as a result, Arabic speaking language|Hebrew]] and are usually a simple of an action of army of the field of and in the first the [[Characters [[chemistry|physical of [[star|stellar]] …"
- **D1:** "The engines badge he was a trained theory of an a survey for the Arts, which has been offsetting in 1928. He made many in systems, while the of province culture is the extent be to a modest trip in in 1902. …"

**Decision (by the registered rule).**

- **Rule 1 is not met.** D1's pooled SCRIBE is below D0's. Clean text is not chosen.
- **No test read.** The Reads table stays empty for this change.
- `sample`'s default is unchanged. `--clean` stays behind its flag, because it is a clean base for later steps: no markup to copy, and better bits on prose.
- `qa`, `chat` and `context8k` are unchanged.

**Predictions scored.**

| # | Prediction | Result |
|---|---|---|
| G1 | D0 within 0.03 of 0.063 | **Right** (0.079) |
| G2 | D1's SCRIBE above D0's | Wrong (0.069 against 0.079) |
| G3 | Disjoint intervals, even odds | Wrong |
| G4 | Surface S +0.01 or more | **Right** (+0.023, disjoint) |
| G5 | Continuation BPC −0.05 or more, paired interval below 0 | Wrong on size (−0.029; the interval is below 0) |
| G6 | Both copy shares under 0.10; D1 − D0 within ±0.02 | Half right (0.041 and 0.005; the fall, −0.037, is larger than ±0.02) |
| G7 | Fewer of D1's samples dropped as short | **Right** (83 against 132) |
| G8 | Default files byte-identical | **Right** (`sample --sim default,online=1,probe=128`: the dev file is byte-identical to A4d's, 492,276 bytes) |
| G9 | The test read agrees on direction | Not scored: no test read |

**What it means for the track.** The markup is a small part of the problem. Sentence form (D_S ≈ 18 of SCRIBE's D ≈ 23) is most of it, in both arms. That is GEN-B's target (sentence gate) and the token track's (whole words). Later steps can stack on `--clean` (and must compare against D1 as well as D0 if they do).

## GEN-B: the sentence gate

The sentence gate's design, rules, dev arms and test read are in [GEN_B_SENTENCE_GATE.md](GEN_B_SENTENCE_GATE.md).

## GEN-E: the content-plan expert

Full report: [GEN_E_CONTENT_PLAN.md](GEN_E_CONTENT_PLAN.md) (design, rule and predictions registered before any run).

**Change.** `cyphalm_trie sample … --plan STOREDIR[,RAWFRAMES] [--plan-drop LO:HI]`. The fact store (`pattern_learn`'s v2 files) plans up to 6 facts about the entity the prompt names. A next-byte expert follows those facts' realisations (F2's learned frames, filled with the entity and the value). It is mixed last, with lambdas fit on the holdout like the other experts. CyphaLM writes every byte. `qa` and `chat` never build it.

**Test read** (once, on this benchmark; facts read from the benchmark's own bytes dropped):
- 38 of 250 prompts have a plan (0.152). The other 212 are byte-identical to the baseline.
- SCRIBE v2: 0.072 [0.059, 0.085] → 0.071 [0.059, 0.084]. On the covered prompts, paired: −0.021 [−0.082, +0.034].
- Continuation BPC, paired: −0.0030 [−0.0052, −0.0013] overall; −0.0199 [−0.0328, −0.0092] on the covered prompts.
- Copy share 0 → 0; contradictions with the store 0 → 0; surface S 0.861 → 0.860.

**Decision.** The rule needs SCRIBE to rise with disjoint intervals. It does not, so `--plan` stays a flag. The plan carries real knowledge (the human continuation gets cheaper), but a byte expert cannot hold a phrase. On dev it wrote the frame and drifted on the value ("was born in China" for Cheshire). The next places to use the plan are the word-level writer (token track) and GEN-B's sentence gate.

