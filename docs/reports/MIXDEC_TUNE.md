# D6: tuning the decoder (more experts, more synergy)

**Odin Loch** · 2026-11-13 · branch `wip/mixdec-tune` · design: [../DECODER.md](../DECODER.md) · D4: [K5_MIXDEC.md](K5_MIXDEC.md) · GENERATION_PLAN §13.38, §13.72

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
