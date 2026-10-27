# B4: the class-context expert, and what the tagger adds through it

**Odin Loch** · 2026-10-28 · PLAN B4 · TOOL_PROFILE_PLAN P1 (the adapter for the tagger and the word classes) and the BPC half of P2

**Question.** CyphaLM's trie reads four bytes. Does an expert that reads *word classes* help it? And how much of that comes from the classes themselves (here, E1's part-of-speech tags) rather than from the expert's own mechanics?

**Answer.** The expert helps a lot. Most of the gain is its own mechanics; the tags add a little.
- **BPC.** On fresh enwik8 text, read once, trie + InfiniGram goes from 2.1660 to 2.0879 with shape classes only (−0.078) and to 2.0751 with E1 tags (−0.091). Both intervals are far from zero.
- **What the tags add.** −0.0128 BPC over shape classes, interval [−0.0147, −0.0110]. That is inside the registered prediction for the tagger (−0.00 to −0.03), but just under TOOL_PROFILE's bar of −0.01 BPC per 1,000 lines of tool code: the tagger is 1,396 lines, so it needs −0.014.
- **Surface score.** The expert lifts surface S from 0.761 to 0.905 (shape classes) or 0.877 (tags); human text scores 0.870. The tags *lower* surface S by 0.029 against shape classes.
- **Q/A.** Not measured yet. It comes with the P2 grid, once the expert is wired into `cyphalm_trie qa`.

PLAN rule 10 holds. The expert predicts the next byte inside CyphaLM's mixture; it never answers.

---

## The expert

`native/include/cypha/cyphalm/class_context.hpp` (200 lines, header-only).

- **Context.** The classes of the last two completed words, plus the bytes typed since the last word ended (at most `tail` bytes, default 6). A word is a run of ASCII letters, digits or apostrophes.
- **Classes.** A `word<TAB>class` table, looked up in lower case. A word not in the table gets one of three shape classes: lower case, capitalised, number. With no table at all, every word is a shape class.
- **Estimate.** Two orders, (c2, c1, tail) and (c1, tail), interpolated Witten-Bell style over a uniform floor.
- **Mixing.** One lambda per bucket (the deepest order found × 4 count buckets = 12), on top of CyphaLM's distribution, exactly as `cyphalm_trie` mixes InfiniGram. The lambdas are fit on the first half of the holdout.
- **Counts.** Built once over the training bytes: (context, byte) keys are sorted and packed into one table, 16 bytes per context and 5 per follower. So this expert is batch-built, not online.

**The tag table (the tagger's adapter).** E1's tagger (`pos_tagger tag`, dev UPOS 94.81%) tags the training bytes `[0, 7340032)`. Each lower-cased word seen at least twice takes its most frequent tag: 25,517 words, 17 tags. A shell pipeline makes the table (Reproduce).

---

## Setup

| Item | Value |
|---|---|
| Text | enwik8, train `[0, 7340032)`, protocol holdout `[7340032, 8388608)`, fresh text `[8388608, 9437184)` |
| Trie | `train_span` on the training bytes, weights frozen, compacted match, Synth mode: the `bpc` model |
| InfiniGram | over the training bytes, cap 8192, 28 lambdas fit on the first holdout half (as `context8k`) |
| Expert | built on the training bytes; 12 lambdas fit on the first holdout half, on top of the trie or of trie + InfiniGram |
| Machine | 4 shared cores (three other agents running), `nice -n 5`, g++ 13.3 `-O2 -std=c++20`, clean under `-Wall -Wextra` |

**Order of work.** The tail and the class table were chosen on the second half of the protocol holdout. The chosen pair (tags vs shape classes, tail 6) was then read once on fresh text, with the lambdas refit on the whole protocol holdout. The fresh numbers are the result; the sweep is how the choice was made.

---

## The sweep (protocol holdout, second half, 524,288 bytes)

Trie alone 2.278066, trie + InfiniGram 2.072500.

| Classes | Tail | Contexts | Table | Trie + expert | Trie + IG + expert | Gain over trie + IG (95% block bootstrap, 4 KiB blocks, 2,000 resamples) |
|---|---:|---:|---:|---:|---:|---|
| shape only | 4 | 585,500 | 15.5 MB | 2.205460 | 2.034727 | −0.0378 [−0.0406, −0.0351] |
| shape only | 6 | 1,242,013 | 29.1 MB | 2.116367 | 1.998603 | −0.0739 [−0.0774, −0.0702] |
| shape only | 7 | 1,440,922 | 33.2 MB | 2.115441 | 1.997854 | −0.0746 [−0.0781, −0.0710] |
| E1 tags | 4 | 1,819,856 | 43.7 MB | 2.143545 | 2.003015 | −0.0695 [−0.0725, −0.0666] |
| E1 tags | 6 | 2,709,055 | 61.3 MB | 2.094396 | **1.982232** | −0.0903 [−0.0936, −0.0870] |
| E1 tags | 7 | 2,898,651 | 65.2 MB | 2.095311 | 1.982448 | −0.0901 [−0.0933, −0.0868] |
| word identity (35,819 words) | 4 | 7,872,801 | 170 MB | 2.134094 | 2.026396 | −0.0461 [−0.0483, −0.0439] |
| word identity | 6 | 8,186,615 | 176 MB | 2.120160 | 2.019990 | −0.0525 [−0.0549, −0.0501] |

- **The tail matters most.** Going from 4 to 6 bytes doubles the gain. At 7 there is nothing more.
- **Classes beat words.** With every word its own class, the expert is a word-bigram model, and it is worse than tags at every tail and worse than shape classes at tail 6. Generalising over words is what pays.
- **Tags beat shapes**, by 0.032 at tail 4 and 0.016 at tail 6. The longer tail already captures some of what a tag says about the word just finished.
- Tail 6 was chosen over 7 as the smaller model with the same BPC.

---

## Fresh text, read once (1,048,576 bytes)

Trie 2.367826, trie + InfiniGram 2.166032.

| Expert | Trie + expert | Trie + IG + expert | Gain over trie (95%) | Gain over trie + IG (95%) |
|---|---:|---:|---|---|
| shape only, tail 6 | 2.202063 | 2.087926 | −0.1658 [−0.1712, −0.1606] | −0.0781 [−0.0810, −0.0751] |
| E1 tags, tail 6 | 2.186263 | **2.075081** | −0.1816 [−0.1888, −0.1749] | −0.0910 [−0.0939, −0.0879] |
| tags minus shape only | | | | **−0.0128 [−0.0147, −0.0110]** |

Intervals are 95% block bootstraps over 4 KiB blocks (4,000 resamples). Each gain is negative in all 4,000 resamples, so LCI ≥ 0.95 holds for all three.

---

## Surface score

200 prompts (the first 64 bytes of holdout paragraphs, `cyphalm_trie sample`'s rule and seed), 400 bytes each at T = 0.8, nucleus 0.9, the no-repeat rule, the same seed for every model. Trie only (no InfiniGram), because that is how `sample` writes. S = H·D·L (`scribe.hpp`), with D and L referenced to the human continuations of the same 200 prompts.

| Text | Surface S | Gain over trie (95% bootstrap over prompts) |
|---|---:|---|
| human continuation | 0.8702 | |
| trie | 0.7612 | |
| trie + shape-only expert | **0.9052** | +0.1440 [+0.1341, +0.1538] |
| trie + E1-tag expert | 0.8766 | +0.1154 [+0.1052, +0.1256] |
| tags minus shape only | | −0.0286 [−0.0370, −0.0206] |

- **S saturates.** Both expert rows reach the human level, because S mostly counts real words, and the expert writes real words: *"names that his home or the referred such major a since the who legal of increasill actions and [[National of the states called…"* against the trie's *"Chan the same title>\n <id>3861</id>…"*. It is still word salad. S cannot see that; SCRIBE has not yet passed its ladder (PLAN §6.4), so it judges nothing here.
- **Tags lower S.** With tags, the samples drift into markup more often (`==`, `{{cite`, `[[Category:`). Tag contexts after markup are common, and the tag-level counts spread them.

---

## Predictions, scored

| Prediction (registered in TOOL_PROFILE_PLAN §6 before the runs) | Result | Score |
|---|---|---|
| `pos_tagger`: BPC −0.00 to −0.03 | −0.0128 over the shape-only expert on fresh text | **Hit** |
| `pos_tagger`: Q/A +0 to +2 per 100 | Not run yet (P2 grid) | — |
| `ppmi_svd` classes: BPC −0.00 to −0.03, overlapping with the tagger | Waiting for S-1b's word classes | — |

**Not predicted.** The gain from the expert's own mechanics (−0.078 BPC, +0.144 S) was not registered: the plan treated the expert only as a carrier for the tags. It is reported as a finding, chosen on the protocol holdout and confirmed once on fresh text.

---

## What it costs

| Item | Value |
|---|---|
| Code | `class_context.hpp`, 200 lines; in `cyphalm_trie`, the `cls` subcommand and `qa --cls` (326 lines with the self-test checks and help text); the tag table is a shell pipeline over `pos_tagger tag` |
| Build of the table | 4.0 s (shape only, tail 6) to 6.0 s (tags, tail 6) on 7 MiB; tagging the 7 MiB takes 5 s |
| Scoring | 1.4 s (shape only) to 2.6 s (tags) for the 1,048,576 holdout bytes, about 1.3–2.5 µs a byte |
| Table in RAM | 29.1 MB (shape only) or 61.3 MB (tags), against the trie's 25 MB of nodes |
| Peak RSS of the whole harness | 257 MB (shape only), 303 MB (tags) |
| Online learning | **None.** The table is batch-built. An online version needs a hash table in place of the sorted one |

---

## What this means for the tool profile

1. **The expert is a CyphaLM upgrade in its own right** (PLAN B4). Shape classes alone pass LCI on BPC and on surface S. It should be measured on Q/A in the P2 grid before it becomes a default.
2. **The tagger is a borderline case.** It helps BPC (−0.0128) but not by enough per line, and it lowers surface S. Its Q/A effect decides it (TOOL_PROFILE §5).
3. **Word classes next.** S-1b's k-means classes (256 and 1,024) go through the same harness, both alone and with the tags (the F − T rows).
4. **Wiring (done).** `cyphalm_trie cls` reproduces this report; `qa --cls` adds the expert to the Q/A harness; `selftest` checks it on fixed inputs.

---

## Reproduce

`cyphalm_trie cls` prints every number in this report. Its output matches the first runs to the last printed digit (BPC, intervals and surface S); those runs used scratch harnesses that `cls` replaces. `qa --cls` adds the expert to the Q/A harness for the tool-profile grid.

```bash
g++ -O2 -std=c++20 -I native/include -I native/third_party -I native/third_party/libsais \
    native/tools/cyphalm_trie.cpp native/src/cyphalm/infinigram.cpp libsais.o -o cyphalm_trie
./cyphalm_trie selftest        # includes fixed-input checks of the expert

# the tag table (E1 model as in docs/reports/E1_POS_TAGGER.md)
head -c 7340032 enwik8 > wiki_train.txt
./pos_tagger tag pos_model.bin < wiki_train.txt > wiki_train.tags
tr ' ' '\n' < wiki_train.tags | awk -F'/' 'NF==2 && $1 ~ /^[A-Za-z0-9'"'"']+$/ {print tolower($1)"\t"$2}' \
  | sort | uniq -c \
  | awk '{if($1>=2){ if(!($2 in best) || $1>bc[$2]){best[$2]=$3; bc[$2]=$1}}} END{for(w in best) print w"\t"best[w]}' \
  | sort > tags_lex_wiki.tsv          # 25,517 words

# the decision run: shape classes against tags, tail 6 (holdout, fresh text, surface S); about 6 minutes
./cyphalm_trie cls enwik8 shape:6 tags_lex_wiki.tsv:6

# the sweep (the holdout lines are the sweep table)
tr -cs "A-Za-z0-9'" '\n' < wiki_train.txt | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn \
  | awk '$1>=2 && NR<=60000 {print $2"\t"$2}' > words_lex_wiki.tsv     # word identity, 35,819 words
./cyphalm_trie cls enwik8 shape:4 shape:6 shape:7 tags_lex_wiki.tsv:4 tags_lex_wiki.tsv:6 tags_lex_wiki.tsv:7 \
    words_lex_wiki.tsv:4 words_lex_wiki.tsv:6
```

Output lines: `holdout2` and `fresh` (the model without the expert), `spec … holdout2 / fresh` (with it), `… gain_over_trie[_ig]` (mean, 95% interval, share of resamples below and above 0), `surface`, and the comparisons with the first spec.
