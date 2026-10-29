# GEN-C: document experts for long-range context (PLAN B5)

**Odin Loch** · 2026-10-29 · generation track step 3 (PLAN ledger 72) · section "GEN-C" of the track (this file stands in for `GEN_TRACK.md` until GEN-A's file is merged)

(The design, the reads, the rules and the predictions below were written and committed before any run of the new experts. Results are added under them.)

**Outcome (dev, 2026-10-29).** The document experts lower bits a byte on held-out text by 0.043–0.044 (the cache does almost all of it), and with the cache they cut SCRIBE's coherence distance by 23–35%. But no arm raises SCRIBE v2 over the default (best 0.060 against 0.063): topical words land in broken byte-level syntax, and an unfrozen cache loops on its own output. Nothing is adopted; `--doc` stays off by default; no test read.

**Question.** CyphaLM's samples change subject every few words. Odin's aim is coherent text. Can experts that remember the document, with no network, keep the samples on the subject across sentences? PLAN B5 names the tools: a document cache and trigger pairs (Kuhn & De Mori 1990; Rosenfeld 1996), and a topic mixture from the C-track vectors. The word-level model (PLAN B3) is now the token track's (`docs/TOKENS_PLAN.md`), so this is B5 alone, at the byte level.

**Rule 10.** These are next-byte experts. Each proposes one byte of a word that the document makes likely; the mixture decides. None writes a word or a passage. They are off inside `qa` and chat answers: only `sample` takes `--doc`, so answers are unchanged.

## The experts (`native/include/cypha/cyphalm/doc_cache.hpp`)

- **Words.** Runs of ASCII letters, compared without case. **Content words** have at least 3 letters and are not among the 200 words that most training documents contain (a stop list from counts).
- **Documents.** The text from one `<page>` to the next. A reader (stream) starts again after `<page>`.
- **Triggers.** From the train bytes [0, 7 MiB) only. In each training document, the 100 content words with the best tf-idf form pairs. A pair shared by at least 3 documents (both words in at least 3) is scored by local mutual information, co × ln(co N / (df_a df_b)), kept if positive. Each word keeps its 32 best, weights summing to 1. In a document, every content word seen for the first time adds its triggers' weights to a mass over words.
- **Cache.** In a document, each content word seen adds 1 to its own mass.
- **Topic** (optional). The mean of the document's content-word vectors (the similar-context expert's projected PPMI-SVD rows, `data/cyphalm/simvec_ppmi32.bin`; not SCRIBE's vectors), recomputed every 4 content words; its 512 nearest content words get mass exp((cos − cos_max) / 0.05).
- **Prediction.** Each part keeps its 1,024 largest masses. At a position, the bytes since the last non-letter are the partial word. Candidates that begin with it (without case) propose their next letter, in their most frequent training spelling. A candidate equal to the partial word proposes a word-ending byte, from the training distribution of bytes after a word. No candidate: uniform, bucket 0.
- **Buckets** (21): none; else the kind of position (word start after a space, word start after another byte, 1, 2, 3+ letters typed) × a confidence level (at a word start, the distinct content words seen in the document: 1, 2–4, 5–16, 17+; inside a word, the matching candidates' share of the part's mass: < 0.01, < 0.05, < 0.2, more).
- **Mixing.** In `sample`, after the default experts (similar-context, then copy), in the order trig, cache, topic. Each part's lambdas are fit on the first half of the holdout [7, 7.5 MiB) over the mixture before it, as the other experts' are.
- **Flag.** `cyphalm_trie sample … --doc SPEC`; SPEC is a list of parts (`trig`, `cache`, `topic`) and settings, or `default`. Without `--doc` nothing changes.

## The reads (fixed before any run)

1. **Holdout bits a byte** (PLAN B5's measure). On the second half of the holdout [7.5, 8 MiB), the full default mixture before and after the document experts. A bootstrap over 32 blocks of 16 KiB gives the interval of the difference. Split by the byte's offset in its document (under and over 4 KiB) for the long-document effect. Printed by `sample --doc` itself (lines `DOC read`).
2. **Dev generation.** `cyphalm_trie sample … --sim default,online=1,probe=128` (today's default, named as A4d named it) on its dev prompts (file 0: 250 prompts, T 0.6 / 0.8 / 1.0, 600 bytes), scored by `scribe ladder` with A4d's command (pack enwik8 [16, 48) MiB, reference [48, 56), test [56, 64), `/root/work/scribe/vec32`, profile v2, seed 20261027). Reported: SCRIBE v2 pooled with its 95% interval, surface S, the axes D_S, D_C and D_I, and the share of samples with D_I > 0 (SCRIBE's copy axis).
3. **Arms on dev:** A0 the default (no `--doc`); A1 `trig`; A2 `trig,cache`; A3 `trig,cache,topic`.
4. **Test read, once.** The chosen arm against A0 on GEN-A's frozen generation benchmark (`scripts/gen_eval.sh`, `/root/work/gena/HARNESS_READY`), if it is ready when the dev choice is made. If not, dev results are reported and the test read is left for later.

## Rules (fixed before any run)

1. **Dev choice.** Among A1–A3, the arm with the highest dev SCRIBE (pooled), if it is above A0's; its share of D_I > 0 must not rise by more than 0.02 and its surface S must not fall by more than 0.01. If no arm qualifies, nothing is adopted and there is no test read.
2. **Adoption in `sample`.** On the test read: SCRIBE rises over A0 with disjoint 95% intervals, the copy share (D_I > 0) does not rise by more than 0.02, and surface S does not fall. Then `--doc default` names that arm and becomes `sample`'s default; otherwise `--doc` stays off by default.
3. **Bits-a-byte paths** (`context8k`, `qa`'s holdout). Not wired here. They would need LCI ≥ 0.95 in their own read; read 1 is reported only.
4. Nothing is changed after the test read.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| C1 | Read 1: the triggers lower the holdout's second half by 0.003–0.03 bits a byte, with the interval below 0 |
| C2 | Read 1: the cache lowers it by more than the triggers do, and the gain is larger at offsets of 4 KiB and over than under |
| C3 | Read 1: the topic part adds under 0.003 bits a byte (the similar-context expert already reads these vectors) |
| C4 | Dev: the best arm's D_C (coherence distance) falls by at least 10% against A0 |
| C5 | Dev: the best arm's SCRIBE rises by +0.005 to +0.03 over A0 (0.063); the intervals overlap. Most of the gap to human text is D_S (sentence form), which these experts do not address |
| C6 | Dev: the copy share (D_I > 0) stays within 0.01 of A0; surface S within ±0.01 |
| C7 | Without `--doc` and with A4d's flags (`--sim default,online=1,probe=128`, which is today's default), `sample`'s dev file is byte-identical to A4d's (`/root/work/a4d/runs/SO_dev.txt`) |
| C8 | The test read, if made, does not meet rule 2 (disjoint intervals) |

## Amendment 1 (registered 2026-10-29 after A2's dev read, before any run of these arms)

**What A2 showed on dev.** A2 (`trig,cache`) lowers the coherence distance D_C from 4.79 to 3.39, but its samples loop on cached words ("offering … offering … offering"): in `sample` the parts read what CyphaLM itself writes, so a word it repeats raises its own odds again. SCRIBE falls (0.051 against 0.063) and surface S falls (0.813 against 0.860); A2 fails rule 1.

**The change.** A `frozen` setting: in `sample`, after the prompt, the parts stop learning (`Stream::freeze`); they still follow the partial word. The document state is then the prompt's alone, so the parts keep the prompt's subject but cannot feed on their own output. The holdout fit (read 1) is unchanged: there the text is real, not written by CyphaLM.

**New arms on dev:** A4 `trig,cache,frozen`; A5 `trig,cache,topic,frozen`. Rule 1 now chooses among A1–A5. Rules 2–4 are unchanged.

**Predictions for the amendment.**

| # | Prediction |
|---|---|
| C9 | A4's samples do not loop: surface S within 0.01 of A0 (0.860) |
| C10 | A4's SCRIBE is at least A0's (0.063), with D_C below A0's 4.79; the intervals overlap |
| C11 | A5 is within 0.01 SCRIBE of A4 (the topic part adds little over the triggers) |

## Results (2026-10-29, dev only)

Tool: `cyphalm_trie` at 9ea77fd (A1–A3) and b16f834 (A4, A5: the same code plus `frozen`). Every arm ran with today's default experts (`--sim default,online=1,probe=128`, then copy). The machine was shared with about twenty other agents, so the times below are wall times under load.

### A0: the default is unchanged

A0 was stopped after 74 of its 250 prompts to give its CPU to the other arms. Its file was byte-identical to A4d's dev file up to that point (143,735 bytes, the MODEL line included). So A0's SCRIBE read is A4d's (same tool, same command, same text).

### Read 1: bits a byte on the holdout's second half [7.5, 8 MiB)

The full default mixture before and after the document experts. Bootstrap over 32 blocks of 16 KiB.

| Arm | Parts | Before | After | Difference [95% interval] | Offset < 4 KiB | Offset ≥ 4 KiB |
|---|---|---:|---:|---|---:|---:|
| A1 | trig | 1.8953 | 1.8927 | −0.0026 [−0.0038, −0.0015] | −0.0028 | −0.0025 |
| A2, A4 | trig, cache | 1.8953 | 1.8522 | **−0.0431 [−0.0525, −0.0344]** | −0.0426 | −0.0435 |
| A3, A5 | trig, cache, topic | 1.8953 | 1.8508 | **−0.0445 [−0.0542, −0.0354]** | −0.0452 | −0.0440 |

- **The cache does almost all of it** (−0.041 on top of the triggers). Its lambdas reach 0.64–0.83 inside words whose matching candidates hold much of its mass. The triggers' lambdas stay at 0.2 or below; the topic part's at 0.35 or below.
- **Not only long documents.** The gain is about the same under and over 4 KiB into a page.
- A4's and A5's reads equal A2's and A3's to every printed digit: `frozen` changes only sampling.
- **Trigger statistics.** 917 training documents, 59,654 words; 36,298 trigger entries kept (a pair needs 3 shared documents). The topic part has 15,443 content words with a vector.

### Read 2: dev generation (250 prompts × T 0.6 / 0.8 / 1.0, SCRIBE v2)

| Arm | SCRIBE [95% interval] | Surface S | D_S (sentence form) | D_C (coherence) | Copy share (D_I > 0) | Peak RSS | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 (default; A4d's read) | 0.063 [0.051, 0.078] | 0.860 | 18.92 | 4.79 | 0% | 79 MB* | — |
| A1 trig | 0.060 [0.047, 0.074] | 0.854 | 19.45 | 4.94 | 0% | 115 MB | 6,316 s |
| A2 trig, cache | 0.051 [0.040, 0.064] | 0.813 | 20.82 | 3.39 | 0% | 116 MB | 6,426 s |
| A3 trig, cache, topic | 0.048 [0.037, 0.059] | 0.803 | 20.91 | **3.10** | 0% | 143 MB | 6,406 s |
| A4 trig, cache, frozen | 0.059 [0.046, 0.074] | 0.839 | 20.11 | 3.67 | 0% | 117 MB | 4,956 s |
| A5 trig, cache, topic, frozen | 0.060 [0.048, 0.072] | 0.830 | 19.66 | 3.64 | 0% | 143 MB | 5,008 s |

*A4d's run wrote two files and fit over half the holdout; `--doc` fits over the whole holdout (its second half is read 1), so it holds more.

- **The experts do what they were built for.** With the cache, the coherence distance falls by 23–35% (4.79 → 3.10–3.67). The samples stay with the prompt's subject.
- **But SCRIBE does not rise.** The sentence-form distance D_S rises in every arm (18.92 → 19.45–20.91), and surface S falls. Topical words land in broken byte-level syntax; the cache (unfrozen) also loops on its own output.
- **Frozen** (amendment 1) stops the loops and recovers most of the loss (A4 0.059 against A2 0.051), but not past the default.
- No arm copies: the copy share is 0% everywhere.

**One prompt, T 0.6** (prompt "It was probably in the year [[400]] that Alaric made his first i"):

- **A0:** "n the yeaths of and are the [[Northern times of the [[Allied]] [[1940'' [[Nazism|Protestant]], [[Evangelicalism in the yeast former served airport, and relationship the state of the [[Antarctic …"
- **A2:** "n the yeaths of alaric the of [[Alaric career. The career and the second battle stage of the origin of the orient [[Parliament of orient origin origin of [[March the same time …"
- **A4:** "n the yeaths of alaric the of [[Alaric career. The contract alaric thematicalled a composite de alabaster the [[Probably in [[1959]] he heard found in the yeast series, the [[Alaric character of the books …"

### Decision (by the registered rules)

- **Rule 1: no arm qualifies.** None has a dev SCRIBE above A0's 0.063 (the best, A5 and A1, are 0.060), and every arm with the cache lowers surface S by more than 0.01. So nothing is adopted, and there is **no test read** (GEN-A's harness was not ready either).
- `--doc` stays off by default. The code stays behind the flag: read 1 shows a large gain in bits a byte, and the token track (`docs/TOKENS_PLAN.md`, T6) can carry the parts to the token level, where a cached or triggered word is written whole instead of letter by letter.
- **Rule 3, the bits-a-byte paths.** Read 1 is a strong candidate for `context8k` and `qa`'s holdout (−0.044 on the holdout's second half), but it is not wired there and needs its own read with LCI ≥ 0.95.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| C1 | Triggers −0.003 to −0.03, interval below 0 | **Miss, narrowly:** −0.0026 [−0.0038, −0.0015], below 0 but under the range |
| C2 | Cache more than triggers; larger at ≥ 4 KiB | **Hit:** cache −0.041 against −0.003; −0.0435 against −0.0426 (barely) |
| C3 | Topic under 0.003 | **Hit:** −0.0013 |
| C4 | Best arm's D_C −10% or more | **Hit for every arm with the cache** (−23% to −35%); **miss for triggers alone** (+3%) |
| C5 | Best arm's SCRIBE +0.005 to +0.03 | **Miss:** no arm rises (best 0.060 against 0.063) |
| C6 | Copy share within 0.01; surface S within ±0.01 | **Partial:** copy share 0% everywhere; surface S held only for A1 (−0.006), fell 0.021–0.057 with the cache |
| C7 | Defaults byte-identical | **Hit** on the 74 prompts run (143,735 bytes) |
| C8 | The test read misses rule 2 | **Not scorable:** no test read (rule 1) |
| C9 | A4 does not loop; surface S within 0.01 | **Miss:** the loops are gone, but surface S fell 0.021 |
| C10 | A4's SCRIBE at least A0's; D_C lower | **Partial:** D_C 3.67 (lower); SCRIBE 0.059 (not at least 0.063) |
| C11 | A5 within 0.01 of A4 | **Hit:** 0.060 against 0.059 |

### Checks

- `cyphalm_trie selftest` passes, with the document experts' checks on a toy corpus. Three planted bugs fail it: no reset at `<page>`, the lowercase letter proposed instead of the training spelling, and triggers kept one way only.
- The default `sample` output is byte-identical to A4d's (above). `qa`, chat and every other caller of `sample_with_class` pass no document experts, so their outputs cannot change.
- The new header is included only by `cyphalm_trie.cpp`.
- The nine CTests of HANDOFF §4.4 pass (9 of 9) on this branch.

### How to reproduce

```bash
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target cyphalm_trie
./build/cyphalm_trie sample enwik8 A4.txt --sim default,online=1,probe=128 --doc trig,cache,frozen 2> A4.err
grep '^DOC' A4.err   # read 1
scribe ladder --pack enwik:enwik8:16:48 --ref enwik:enwik8:48:56 --test enwik:enwik8:56:64 \
  --vectors vec32 --profile v2 --seed 20261027 --gen-trie A4.txt --label A4   # read 2
```

Run from the repository root (the topic part reads `data/cyphalm/simvec_ppmi32.bin`). Runs are in `/root/work/genc/runs`.
