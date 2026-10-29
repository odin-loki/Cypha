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

## GEN-C2: the document experts where only prediction counts (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · the read GEN-C's rule 3 left open

**Question.** GEN-C's document experts lower held-out bits a byte by 0.043, but `sample` does not take them, because SCRIBE's sentence form falls. `context8k` and `qa`'s holdout lines are read by bits a byte alone. Do the document experts earn a default there, under the LM rules?

**Where they would go.**
- `context8k`: its holdout chain (trie + InfiniGram + class + similar-context + copy), then the document experts, as a new `_doc` line. Its answers are written by trie + InfiniGram alone (`gen`), so they cannot change.
- `qa`: its holdout lines (trie + InfiniGram + class + similar-context + copy), then the document experts. `qa`'s answers are written by their own mixers, which are never given the document experts. So the rows cannot change; this is checked against main's rows (`/root/work/a4e/runs/parity/QB.rows.tsv`, byte for byte).
- Never inside an answer, never in `sample` (GEN-C's flag stays as it is), `chat`, `probe` or `bpc`.

**The read's tool.** `cyphalm_trie cls <enwik8> shape --doc V1;V2;…`, a new branch of `cls`, as A4c's and A4d's fresh reads were made there. The model is trained on wiki train [0, 7 MiB) only. The chain is `qa`'s order and defaults: trie + InfiniGram (cap 8192), class (`shape`), similar-context (`default`, online keys on), copy (m 8), then the document experts. The document experts' words and triggers come from [0, 7 MiB) only (GEN-C).

**Variants (dev):** V1 `trig`; V2 `cache`; V3 `trig,cache`; V4 `trig,cache,topic`. (`frozen` changes only sampling, so it is not read.)

**Dev read.** The holdout's second half [7.5, 8 MiB). Every expert's lambdas are fit on the first half. Per variant: bits a byte before and after, and the paired change over 4 KiB blocks (block bootstrap, 2,000 resamples, seed 7): mean, 95% interval, P(change < 0).

**Fresh read (once, for the chosen variant).** enwik8 [54,530,849, 56,628,001): 2 MiB from the first `<page>` line at or after 52 MiB; 274 page starts. `cls … --sim-stream 54530849:2097152` places it right after the holdout. Every expert's lambdas are fit on the whole holdout [7, 8 MiB), read as one stream that goes on into the fresh slice (the similar-context expert's online keys and the copy window carry over, as in A4e's stream; the document experts start again at the slice's first `<page>`). The paired change over 4 KiB blocks (4,000 resamples, seed 11), as A4d's fresh read.
- **Who has read this slice.** No CyphaLM bits-a-byte read and no CyphaLM choice. SCRIBE's reference ladder ([48, 56) MiB) reads it as human text, and S4c's and S4d's pools (pages after byte 50,000,000) read its pages with content signals only. The shipped similar-context vectors were built on enwik8 without [7, 9) MiB, so they have seen it; that is the same in both arms of the paired read.

**Rules (fixed now).**
1. **Dev choice.** Among V1–V4, those whose dev 95% interval is below 0. The one with the lowest dev bits a byte is chosen, unless a variant with fewer parts is within 0.001 of it; then that one is. If none qualifies: no fresh read, nothing is wired, and the branch is removed (rule 4).
2. **Fresh.** The chosen variant's fresh change has mean below 0 and P(change < 0) ≥ 0.95 (LCI).
3. **Defaults.** If rule 2 holds, the chosen variant becomes the default in `context8k`'s chain and `qa`'s holdout lines only, with `--no-doc` to leave it out. Conditions, all checked before the switch: `qa`'s rows byte-identical to main's; built and served `qa` peak under 200 MB; `context8k`'s answer lines unchanged and its peak reported. If the RAM condition fails, the switch waits, and the read is reported as passed.
4. **Code debt.** If rule 1 or 2 fails, the new `cls --doc` branch is removed and `--doc` stays `sample`'s flag alone.

**Predictions.**

| # | Prediction | Falsified if |
|---|---|---|
| D1 | Over the full `qa` chain the cache variants gain less than GEN-C's −0.043 (InfiniGram and class already hold some of it): V3's dev change is between −0.035 and −0.010 | outside that range |
| D2 | V1 (triggers alone) changes dev by less than 0.003 in size, with its interval below 0 | size 0.003 or more, or interval not below 0 |
| D3 | V4 adds less than 0.002 over V3 | V4 − V3 below −0.002 |
| D4 | The rule picks V2 (cache alone): V3 is within 0.001 of it | another variant is chosen |
| D5 | The fresh change is below 0 with LCI ≥ 0.95, and within ±50% of the chosen variant's dev change | LCI below 0.95, or outside ±50% |
| D6 | `qa`'s rows are byte-identical with the document experts on | any byte differs |
| D7 | `qa`'s headline holdout (1.7895) falls by 0.010–0.035 | outside that range |
| D8 | Built and served `qa` stay under 200 MB | either at 200 MB or more |

### GEN-C2 dev results (2026-10-29, before the fresh read)

Tool: `cyphalm_trie` at befd5ec. Command: `cyphalm_trie cls enwik8 shape --doc 'trig;cache;trig,cache;trig,cache,topic'`, run from the repository root. Wall time 2,182 s under load; peak 303 MB (the `cls` tool holds per-byte arrays for every expert; this is not `qa`'s peak).

The chain is trie + InfiniGram + class + similar-context + copy, the model trained on wiki train only: 1.802237 bits a byte on the holdout's second half. Each variant goes on last; the change is paired over 4 KiB blocks.

| Variant | Parts | After | Change [95% interval] | P(change < 0) |
|---|---|---:|---|---:|
| V1 | trig | 1.801014 | −0.0012 [−0.0017, −0.0008] | 1.000 |
| **V2** | **cache** | **1.766556** | **−0.0357 [−0.0411, −0.0305]** | 1.000 |
| V3 | trig, cache | 1.768024 | −0.0342 [−0.0397, −0.0290] | 1.000 |
| V4 | trig, cache, topic | 1.767519 | −0.0347 [−0.0403, −0.0295] | 1.000 |

- The cache does almost all of it, as in GEN-C. Over this chain it gains −0.036, against −0.041 over `sample`'s chain (no InfiniGram, no class).
- The triggers hurt a little in front of the cache: V3 is 0.0015 worse than V2. Their lambdas are fit first, over the chain without the cache.
- The gain is the same under and over 4 KiB into a page (−0.0354 and −0.0359 for V2).
- Building the cache takes 1.3 s and fitting it 33 s here.

**Dev choice (rule 1).** Every variant's interval is below 0. V2 has the lowest bits a byte and the fewest parts, so **V2 (`cache`) is chosen**. The fresh read is made once, for V2 alone.

### GEN-C2 fresh read (2026-10-29, read once)

Command: `cyphalm_trie cls enwik8 shape --doc cache --sim-stream 54530849:2097152` at befd5ec, from the repository root. Wall time 5,784 s under load; peak 514 MB (`cls` holds per-byte arrays for the holdout and the 2 MiB stream).

| Read | Chain | With the cache | Change [95% interval] | P(change < 0) |
|---|---:|---:|---|---:|
| Holdout's second half (dev, reproduced) | 1.802237 | 1.766556 | −0.0357 [−0.0411, −0.0305] | 1.000 |
| **Fresh, enwik8 [54,530,849, 56,628,001)** | **1.915952** | **1.857202** | **−0.0588 [−0.0618, −0.0559]** | **1.000** |

- The chain is trie + InfiniGram + class + similar-context + copy, every lambda fit on the whole holdout; the cache goes on last.
- The fresh gain is larger than on dev. It holds under and over 4 KiB into a page (−0.063 and −0.057).
- The cache's lambdas refit on the whole holdout are close to the first half's (0.79 against 0.83 at the top bucket).

**Rule 2 holds:** the mean change is below 0 and P(change < 0) = 1.000 ≥ 0.95. Rule 3's conditions are checked next, before any default changes.

### GEN-C2 rule 3 and the decision (2026-10-29)

**First wiring, before BKN2 (main 4c47cf2 + DK2, A4e, GEN-E).** The cache was built before `qa`'s per-byte holdout arrays and fit with `fit_doc`.
- `qa`'s rows are byte-identical to main's, built, `--save` and served (`QB.rows.tsv`).
- `qa`'s headline holdout falls 1.789464 → 1.756115 (−0.0333; LCI P(gain > 0.005) = 1.000 over 256 blocks).
- `context8k` falls 1.7897 → 1.7555, and its answer lines are unchanged.
- **But built `qa` peaked at 204.3 MB** (main 184.7), over the 200 MB rule. Served `qa` was 193.3 MB, `--save` 202.6 and `context8k` 202.5.

**Final wiring (b2816b4).**
- The cache is built after `qa`'s per-byte holdout arrays are freed, in their memory, and freed again before the answers.
- `mix_doc_into` keeps each part's predictions over the first half only and mixes the second half as it streams. The selftest checks that it gives `fit_doc`'s numbers exactly, and a planted bug (the second half mixed wrongly) fails it.
- `context8k` uses it too.

**With BKN2's defaults (main c756ee3; byte Kneser–Ney on).** This is the configuration that decides.

| Path | Without the cache (BKN2) | With the cache | Peak, BKN2 | Peak, with the cache | Rows / answer lines |
|---|---:|---:|---:|---:|---|
| `qa` built (`--save`) | 1.727340 | **1.698891** (−0.0284; LCI 1.000) | 196.8–197.1 MB | **197.2 MB** | byte-identical to BKN2's R1 |
| `qa` served (`--model`) | 1.727340 | **1.698891** | 196.4 MB | **196.5 MB** | byte-identical to BKN2's R2 |
| `context8k` | 1.7276 | **1.6985** | 218.9 MB | 219.1 MB | identical to BKN2's C8 |

- Built `qa`'s peak comes before the cache's stage (197.18 MB already reached). The stage adds under 0.1 MB.
- The frozen-test configuration (BKN2: 199.9 MB) was not run. The test split is not read again for a change that cannot touch answers.
- The fresh read was made before BKN2. Its paired gain with BKN on, on the same fresh text, was not read: the session's budget was short. With BKN on, the gain on `qa`'s holdout is −0.0284 (LCI 1.000) and on `context8k`'s −0.0291.

**Decision (rule 3 holds).** The document experts' cache is the default in `context8k`'s chain and `qa`'s holdout lines. `--no-doc` leaves it out, and `--doc SPEC` reads another setting. `sample` keeps GEN-C's default (no cache). No answer changes: no answer mixer is given the cache.

**Predictions, scored.**

| # | Result |
|---|---|
| D1 | **Hit:** V3 −0.0342, inside [−0.035, −0.010] |
| D2 | **Hit:** V1 −0.0012, interval below 0 |
| D3 | **Hit:** V4 − V3 = −0.0005 |
| D4 | **Hit** on the choice (V2). The reason given was wrong: V3 is 0.0015 behind V2, not within 0.001 |
| D5 | **Partial:** LCI 1.000, but the fresh gain (−0.0588) is 65% larger than dev's (−0.0357), outside ±50% |
| D6 | **Hit:** rows byte-identical in every run |
| D7 | **Hit:** 1.7895 → 1.7561 (−0.0333) before BKN2 |
| D8 | **Miss** with the first wiring (built 204.3 MB); **hit** with the final one (197.2 / 196.5 MB with BKN) |

**Checks.** `cyphalm_trie selftest` passes, with the new equivalence check. The other CTests were not run for this report.

**How to reproduce.**

```bash
cyphalm_trie cls enwik8 shape --doc 'trig;cache;trig,cache;trig,cache,topic'           # dev
cyphalm_trie cls enwik8 shape --doc cache --sim-stream 54530849:2097152                # fresh, once
cyphalm_trie qa enwik8 dolly.jsonl $C1LD --save DIR --out rows.tsv; cyphalm_trie qa ... --model DIR
cyphalm_trie context8k enwik8 dolly.jsonl [--no-doc]
```

Runs are in `/root/work/genc2/runs` (`dev`, `fresh`, `adopt/` before BKN2, `bkn/` with it).
