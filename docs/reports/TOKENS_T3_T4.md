# Tokens T3 and T4: the byte–token bridge, and generation at the token level

**Odin Loch** · 2026-10-29 · [TOKENS_PLAN](../TOKENS_PLAN.md) items T3 and T4 · PLAN ledger 73–74

(The design, the rules and the predictions below were written and committed before any run. Results are added under them.)

**Rule 10.** The bridge is part of CyphaLM itself. It turns a token-level CyphaLM's predictions into exact next-byte predictions. It answers nothing, and no neural network is involved.

## The question

A token model predicts whole tokens. CyphaLM is scored, mixed and prompted in bytes. Two things go wrong if the token model is used naively:
- **Scores do not line up.** A token model's bits must become bits per byte, exactly, to compare with every CyphaLM number so far.
- **Tokenization bias** (Phan et al., ICLR 2025, [arXiv 2410.09303](https://arxiv.org/abs/2410.09303v2)). When a prompt ends inside a word, the naive next-token prediction is wrong. Example: vocabulary {A, AA, B}; after the bytes "A", the canonical tokens are [A], and the model has learned that [A] is never followed by "A" (that would have been the token AA). So the naive next-byte probability of "A" is 0, though the true probability is 1 − α.

The bridge fixes both. It is exact, not an approximation.

## The bridge (fixed before any run)

**Definitions** (Phan et al., Definitions 3–4, Lemma 1, Proposition 1–2).
- A token sequence t is **valid** when `encode(decode(t)) = t`. For BPE, longest-match (MPE) and Viterbi (Unigram) tokenizers, a sequence is a prefix of some string's canonical encoding exactly when it is valid. So a model trained on canonical encodings gives invalid sequences probability 0.
- The **covers** of a byte prefix x are the valid sequences whose last token covers the end of x: `decode(t₁…t_{k−1})` is a strict prefix of x, and x is a prefix of `decode(t₁…t_k)`.
- **The lemma:** the covers partition all continuations of x, so P(x) = Σ over covers of P(t).
- A valid sequence that decodes exactly to a string s is unique: it is `encode(s)`. So the covers of x are `encode(x[0:i])` followed by one token v that starts with `x[i:n]`, for each split point i.
- **Truncate-renormalisation** (Proposition 2): a real count model gives some probability to invalid tokens. Each next-token distribution is set to 0 on invalid tokens and renormalised. This does not increase the KL divergence to the true distribution.

**What the header does** (`native/include/cypha/cyphalm/token_bridge.hpp`):
- **Inputs:** a deterministic tokenizer (interface: vocabulary, `encode`, and a stable prefix) and a token model (interface: next-token distribution given a token history).
- **Stable prefix:** the tokenizer names a position p in the bytes so far such that the encoding up to p can never change when bytes are appended. With word-boundary pre-tokenization, p is the start of the last pretoken. For longest match without pre-tokenization, it is the start of the first token whose longest-match test still needs unknown bytes. Only covers that start at or after p are searched, so each byte costs at most (longest token length) model calls.
- **Outputs:** the exact next-byte distribution after any bytes, the exact log₂ prefix probability (the marginal over covers), and the canonical-segmentation log₂ probability for comparison.
- **Sampling** (for T4): exact sampling of whole tokens after any byte prompt. First a cover of the prompt is drawn with probability proportional to its weight (this heals a prompt that ends mid-token), then tokens are drawn from the renormalised model, rejecting invalid ones.

## The planted check (fixed before any run)

The truth must be known exactly, so the test builds it.
- **A byte-level Markov chain** over {a, b, c, space}. The state is the last byte and the current word length. Random transitions; no space after a space; a space is forced after 4 letters (so every word ends within 5 bytes).
- **Four tokenizers over it:**
  - BPE with ranked merges, inside word pretokens (" ab", "cab");
  - longest match (MPE) inside word pretokens;
  - Unigram/Viterbi inside word pretokens;
  - longest match over the raw stream, with tokens that cross spaces, and no pre-tokenization.
  - All 256 bytes are in every vocabulary.
- **The true token model** of each: P(t) = the chain's probability that the stream's canonical encoding starts with t. It is computed exactly by enumerating the continuations that decide the last tokens (at most 5 bytes for word pretokens; longest token − 1 for raw longest match). Next-token probabilities are ratios of these.
- **The bridge is given only the true token model and the tokenizer.** Its next-byte probabilities must equal the chain's transition probabilities.

## Rules (fixed before any run)

1. **Exactness (the plan's T3 gate).** On the planted chain, for each tokenizer and for prompts ending mid-word, at a word end and after a space, the bridge's next-byte probabilities equal the chain's within 1e-9 (absolute).
2. **The prefix identity.** The sum of the bridge's log₂ next-byte probabilities over a prompt equals the bridge's log₂ prefix probability and the chain's log₂ probability within 1e-9 (relative).
3. **Renormalisation.** For a leaky model (0.7 × the true model + 0.3 × uniform over all tokens), the bridge with truncate-renormalisation matches an independent brute-force computation (all segmentations enumerated, validity checked by re-encoding) within 1e-9, and the next-byte probabilities summed over bytes equal the prefix probability ratio (Σₐ P(x a) = P(x)).
4. **Planted bugs must fail the CTest** (`native_token_bridge_test`), each by mutation of a copy of the code:
   - a missed partial-token continuation (only new tokens after the prompt are counted, not tokens that started inside it);
   - a probability not renormalised (the truncate step skipped);
   - a merge-order error in BPE (merges applied in the wrong order).
5. **The tokenizer contract** is checked in the CTest for every tokenizer: validity equals re-encoding, and the stable prefix holds for random strings and extensions.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| B1 | The bridge meets rule 1 for all four tokenizers at every prompt type (max error below 1e-12 in double precision) |
| B2 | The naive estimate (the first byte of the next token after `encode(x)`) is wrong where it matters: total-variation distance from the truth above 0.05 on average over prompts ending mid-word or after a space, for all three word-pretoken tokenizers |
| B3 | The canonical-segmentation log₂ probability of a prompt is below the exact one (more bits) by 0 to 3 bits per prompt; per byte the gap shrinks as the prompt grows |
| B4 | With the leaky model, skipping renormalisation changes some next-byte probability by more than 1e-3 |
| B5 | On real text with TOK-A's token model (when it exists), the exact marginal bits per byte (bridge, renormalised) are below TOK-A's canonical-segmentation bits per byte by 0.005 to 0.05: the renormalisation gain dominates, and the end-of-text cover gain is negligible |

## T4: generation at the token level (rules fixed now; run only when TOK-A's tokenizers and GEN-A's harness exist)

- **What is sampled:** whole tokens from TOK-A's best token model, with prompts healed by the bridge (above). Today's byte model is the baseline, scored by the same harness on the same prompts.
- **Reported together:** SCRIBE v2 with its 95% interval, surface S, the copy share (SCRIBE's verbatim-copy measure), bits per byte on the held-out continuations (through the bridge), and the **invented-word rate** (the share of generated words, as alphabetic runs, that never occur in the training text).
- **Dev choices:** temperature in {0.6, 0.8, 1.0} and the tokenizer among TOK-A's best two, by dev SCRIBE.
- **Gate (one read of GEN-A's frozen test):** SCRIBE rises over the byte model with disjoint 95% intervals, and the copy share rises by 0.02 at most.
- **Predictions carried from the plan:**
  - P-T2: token-level samples score SCRIBE v2 ≥ 0.15 (byte default 0.063), and the invented-word rate falls below 1%.
  - P-T6 (answers unchanged or better with token experts in the mixture) belongs to T10 and is not read here.
- **B6:** the byte model's invented-word rate is above 10%.

### T4 registration (2026-10-29, written and committed before any T4 run)

GEN-A's harness and benchmark are ready (`bench/eval/gen_prompts_{dev,test}.tsv`, `scripts/gen_eval.sh`; [GEN_TRACK](GEN_TRACK.md)). TOK-A's choice is final (`TOKENIZERS_READY`).

**The two tokenizers.** UNI 4,096 (dev 1.9039 bits per byte) and BPE-W 4,096 (1.9048), the two best at the registered sizes, within 0.005 of each other ([T0–T2](TOKENS_T0_T2.md)). Files: `t-uni-4096.tok` and `t-bpe-4096.tok` in TOK-A's run directory. The smaller vocabularies (1,024 and 2,048) and byte Kneser–Ney are not arms here. TOK-A named them for later work; adding them now would be a choice made without a rule.

**The token model.** TOK-A's: modified Kneser–Ney, orders 1–5, mixed with the ∞-gram over token IDs (match cap 64), λ per bucket fitted on the holdout's first half. Built on the training slice [0, 7,340,032), as in T3's bias read.

**The sampler** (`cyphalm_tok_gen gen`, unchanged since T3). The prompt is healed by the bridge: one cover reaching past its end is drawn by weight. Then whole tokens are drawn from the renormalised model at temperature T. There is no nucleus cut and no repeat ban. Each sample runs until it has at least 600 bytes, so it may end up to one token past 600. Seeds: `1000003 (k+1) + 7919 t` for prompt k and temperature index t.

**Known differences from the byte sampler**, recorded and not changed:
- `sample` keeps the top 0.9 of the mass (nucleus) and bans some repeats. The token sampler does neither.
- Its random generator is a 32-bit mix of the same seed.
- So loops and rare tokens are more likely at the token level. SCRIBE's integrity axis D_I is reported, so loops show.

**One fix before any run (measurement only).** The token tool's copy share now uses GEN-A's reference: 32-byte spans of the training bytes [0, 7 MiB), raw **or cleaned by SCRIBE** (`clean_wiki_documents`). Before, it used the raw bytes only. Now both tools measure the copy share the same way.

**Commands.**

```
B=/root/work/tokb/build/cyphalm_tok_gen; TK=/root/work/toka/runs/tok; E=/root/cdata/enwik8
SC=/root/work/gena/runs/scribe_gena; O=/root/work/tokb/runs/t4
# for L in TU (t-uni-4096) and TB (t-bpe-4096), split dev:
$B gen $E $TK/t-uni-4096.tok $O/dev/TU.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv > $O/dev/TU.sample.out
$SC ladder --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors /root/work/scribe/vec32 \
   --profile v2 --seed 20261027 --gen-trie $O/dev/TU.gen.txt --label TU > $O/dev/TU.scribe.md
python3 -I scripts/gen_eval_summary.py $O/dev TU dev
python3 -I scripts/gen_eval_summary.py --pair $O/dev D0 TU      # with GEN-A's D0 files copied in, read only
$B words $E $O/dev/TU.gen.txt /root/work/gena/runs/dev/D0.gen.txt   # the invented-word rate
```

**The byte baseline.** Today's default `sample` (similar-context expert, then copy), through `scripts/gen_eval.sh` with GEN-A's binary `cyphalm_trie_91827e1`. On dev it is GEN-A's D0, read as it stands (not re-run). On test it is read once, as part of this change, with the same command and the label B0.

**The dev choice.**
1. Each arm is a tokenizer at one temperature: 2 × 3 = 6 arms. SCRIBE v2 is read per temperature, from the ladder's per-temperature rungs.
2. **The byte comparator** is the byte default at its own best dev temperature, chosen the same way: D0's highest per-temperature SCRIBE, T 0.6 (0.101 [0.072, 0.135]; copy share 0.0586). So each model is compared at its own best setting. Pooled numbers for both are reported alongside; they do not decide.
3. **Eligible:** an arm's dev copy share is at most the comparator's dev copy share + 0.02, that is at most 0.0786.
4. **Chosen:** the eligible arm with the highest dev SCRIBE. Within 0.005, the lower copy share wins.
5. **No eligible arm:** T4 stops on dev. There is no test read, and the next design is written down, not run.
6. **No dev gain:** if the chosen arm's dev SCRIBE point estimate is not above the comparator's (0.101), there is no test read either. Dev already shows no gain.

**The test read (once).** The chosen arm at its temperature (`--temps T`) and the byte default B0 (all three temperatures, as `gen_eval.sh` runs it), both on `gen_prompts_test.tsv`. **Gate:** the token arm's SCRIBE is above the byte comparator's (B0 at T 0.6), with disjoint 95% intervals, and its copy share is at most B0's at T 0.6 + 0.02. The read is logged in GEN_TRACK's Reads table. A pass makes the token sampler a candidate for `sample`'s default. The switch itself is a later integration step (T10's mixture, P-T6), not part of this change.

**Reported together**, for every arm: SCRIBE v2 [95%] with surface S [95%] alongside, D_I, the samples SCRIBE keeps and their mean length, continuation bits per byte through the bridge [95%] (and paired with D0), the copy share, the invented-word rate, and five samples.

**Predictions** (P-T2 and B6 are carried from above and scored too):

| # | Prediction | Falsified if |
|---|---|---|
| Q1 | For both tokenizers the copy share falls as T rises: T 0.6 > T 0.8 > T 1.0 | Any step goes the other way |
| Q2 | At T 0.6 both tokenizers copy more than 0.30 of their bytes (T3's smoke: 0.94 on two prompts) | Either is at most 0.30 |
| Q3 | Only T 1.0 can be eligible, and at least one tokenizer's T 1.0 is eligible. About even odds | An eligible arm at T 0.6 or 0.8, or no eligible arm |
| Q4 | Continuation bits per byte through the bridge are below D0's 2.134 for both tokenizers, by 0.10 to 0.25, paired interval below 0 | A difference outside [−0.25, −0.10], or an interval touching 0 |
| Q5 | The chosen arm's dev SCRIBE is between 0.06 and 0.15. Whole tokens fix invented words, not sentence form (D_S) | Outside [0.06, 0.15] |
| Q6 | The token arms' invented-word rate is below the byte default's at every temperature | Any token arm at or above D0's at the same temperature |
| Q7 | D_I is above 0 for both tokenizers at T 0.6 (loops, with no repeat ban) | D_I is 0 for either |

## T3 results (2026-10-29)

Code: `native/include/cypha/cyphalm/token_bridge.hpp` and the CTest `native_token_bridge_test` (`native/tools/token_bridge_test.cpp`), commit 41bf8ee; stable hints added in b1fb8d3 (the numbers below are unchanged by it). The CTest runs in about 17 s on the loaded machine and makes 9,352 checks.

### The planted chain (rules 1–3)

24 prompts of 4 to 15 bytes were drawn from the chain. Each prompt's every prefix was checked, so the checks cover prompts that end inside a word, at the end of a word and after a space.

| Tokenizer | Max error, next byte | Prefix identity, relative error | Naive estimate, total-variation distance (after a letter / after a space) | Canonical segmentation, bits below the exact (mean / max) |
|---|---|---|---|---|
| BPE, word pretokens | 5.0e-16 | 3.1e-16 | 0.295 / 0.662 | 0.776 / 1.722 |
| Longest match, word pretokens | 4.4e-16 | 4.5e-16 | 0.328 / 0.712 | 0.822 / 1.797 |
| Unigram (Viterbi), word pretokens | 4.4e-16 | 4.3e-16 | 0.238 / 0.712 | 0.589 / 1.797 |
| Longest match, raw stream (tokens cross spaces) | 1.7e-16 | 2.5e-16 | 0.239 / 0.650 | 0.679 / 1.797 |

- **Exact.** For every tokenizer and prompt, the bridge's next-byte probabilities equal the chain's to within rounding (5e-16; the rule was 1e-9).
- **The naive estimate is badly wrong.** Reading the next byte off the token model after `encode(x)` misses by a total-variation distance of 0.24–0.33 after a letter and 0.65–0.71 after a space. After a space the canonical encoding nearly always merges the space with the next word (" ab", " ca"), so the token " " alone almost never continues, and the naive estimate spreads its mass wrongly.
- **The canonical segmentation undercounts.** Scoring a prompt by its canonical encoding alone gives 0.59–0.82 bits less probability on average (up to 1.8 bits), all at the prompt's end. That is the mass of the covers that overhang it.

**Leaky model (rule 3).** 0.7 × the true model + 0.3 × uniform over all tokens, truncate-renormalised; 6 prompts of 3 to 6 bytes, all 256 next bytes:

| Tokenizer | Max error against brute force | Σₐ P(x a) against P(x), relative |
|---|---|---|
| BPE | 4.6e-15 | 7.4e-15 |
| Longest match | 7.7e-15 | 1.5e-14 |
| Unigram | 5.7e-15 | 1.4e-14 |
| Longest match, raw | 4.1e-15 | 1.2e-14 |

**Sampling.** After the prompts "ab c" (BPE) and "ab ca" (longest match), both of which end inside a word, 4,000 healed samples each: the first generated byte follows the chain's next-byte distribution (worst deviation 0.71 and 1.05 standard errors).

### Planted bugs (rule 4)

Each was applied to a copy of the code; the CTest must fail.

| Bug | Failed checks | What catches it |
|---|---|---|
| Missed partial-token continuation (only new tokens after the prompt) | 862 of 9,352 | Exactness (errors up to 0.66–1.0), the prefix identity, the leaky check |
| Next-token distributions not renormalised | 25 | The leaky check: next-byte errors up to 1.0e-3, and the prefix identity |
| BPE merges applied highest rank first | 5 | The BPE known answers ("cab" → c + ab, "abc", "bca", " ab", "aab") |

- A wrong merge order is a tokenizer bug, not a bridge bug. With a deterministic tokenizer that keeps the contract, the bridge stays exact even when the merges are wrong, so only the known answers can catch it.

### The tokenizer contract on TOK-A's tokenizers (rule 5)

`cyphalm_tok_gen contract`: 150 snippets of 48 bytes from the dev holdout, every cut point.

| Tokenizer | Violations |
|---|---|
| BPE 4,096 and 8,192; Unigram 4,096 and 8,192; Entropy 8,192 | 0 |
| SpaceByte 4,096 and 8,192 | 0, after one fix in the adapter (below) |
| Words 8,192 | 360: not prefix-closed |

- **SpaceByte.** A chunk ends after a space-like byte that is not preceded by another. Encoded on its own, a substring has no byte before it, so a chunk start between two space-like bytes ("</") is not a stable position. The adapter now skips such starts.
- **Words** (a whole pretoken if it is in the vocabulary, else its bytes) is not prefix-closed: an unknown word is spelled in bytes, and a byte prefix such as "Go" re-encodes as the word "Go". So Proposition 1 does not hold for it, and the bridge is not exact for Words. BPE, Unigram, SpaceByte and Entropy are fine.
- **TOK-A's pretokens** (GPT-2's split, where a whitespace run gives its last space to the word after it): the stable position is the start of the last pretoken, moved back to the start of a whitespace run just before it.

### Predictions B1–B4, scored

| # | Prediction | Result |
|---|---|---|
| B1 | Rule 1 for all four tokenizers, max error below 1e-12 | **Hit:** max 5.0e-16 |
| B2 | Naive estimate's total-variation distance above 0.05 on average after a letter or a space | **Hit:** 0.24–0.33 after a letter, 0.65–0.71 after a space |
| B3 | Canonical segmentation 0 to 3 bits below the exact per prompt | **Hit:** mean 0.59–0.82, max 1.80. The gap sits at the prompt's end, so per byte it shrinks as the prompt grows (not measured separately) |
| B4 | Without renormalisation, some next-byte probability moves by more than 1e-3 | **Partial:** 1.01e-3 for longest match, 8.9e-4 for BPE |

### On real text: TOK-A's model through the bridge (2026-10-29)

Tool: `cyphalm_tok_gen bias` (commit b1fb8d3), with TOK-A's BPE 4,096 tokenizer and its model: modified Kneser–Ney (orders 1–5) mixed with the ∞-gram over token IDs, λ per bucket fitted on the holdout's first half, as `cyphalm_tok eval`. Built on the training slice [0, 7,340,032). Read on the first 65,536 bytes of the dev half (from byte 7,864,320). The headers used are a snapshot of TOK-A's working tree (`tokenizer.hpp` sha256 c174e327…, `token_model.hpp` d61beb24…).

| Bits per byte (65,536 bytes) | Raw model | Truncate-renormalised |
|---|---|---|
| Canonical segmentation (TOK-A's measure) | 1.9216 | 1.9201 |
| Exact marginal over covers (the bridge) | 1.9213 | **1.9199** |

- **The exact bits per byte are 0.0017 below the canonical number.** Renormalisation gives 0.0015 of it. The covers that overhang the end of the text give 0.0003, because the text is long and the gap sits only at its end.
- So TOK-A's canonical bits per byte are a fair measure of its models. They overstate the cost by about 0.002 bits per byte.

**Next byte inside the text.** At 398 random positions of the same slice (most inside a word), the exact next byte through the bridge costs 1.963 bits. The naive estimate (the first byte of the next token after the canonical encoding) costs 5.980 bits, a total-variation distance of 0.575. A token model asked for the next byte without the bridge is badly wrong exactly where a prompt ends inside a word.

**Cost.** Build 38 s; the read (both bridges, every byte, 398 next-byte distributions) 546 s on a machine with load average about 35 on 4 cores; peak 406 MB resident.

**Prediction B5, scored: miss.** The predicted gap was 0.005–0.05 bits per byte; the measured gap is 0.0017. Its direction holds: renormalisation dominates, and the end-of-text cover gain is negligible.

### T4 dev results (2026-10-29, before the test read)

Tool: `cyphalm_tok_gen` at dd9a704 (main 8ee4373 and `wip/gena` merged). Labels TU (UNI 4,096) and TB (BPE-W 4,096). Wall times under load about 22: TU 1,943 s, TB 1,331 s (generation and continuation scoring), ladder 22–36 s. Peak resident about 380 MB each. D0 is GEN-A's dev run, read as it stands.

| Arm | SCRIBE v2 [95%] | Surface S [95%] | D_I | Kept / mean bytes | Copy share | Invented words |
|---|---|---|---|---|---|---|
| D0 byte, T 0.6 (comparator) | 0.101 [0.072, 0.135] | 0.900 [0.892, 0.908] | 0.00 | 213 / 328 | 0.0586 | 0.0251 |
| D0 byte, T 0.8 | 0.088 [0.059, 0.119] | 0.879 [0.872, 0.887] | 0.00 | 200 / 318 | 0.0389 | 0.0461 |
| D0 byte, T 1.0 | 0.047 [0.031, 0.065] | 0.785 [0.774, 0.796] | 0.00 | 205 / 285 | 0.0267 | 0.1017 |
| D0 byte, pooled | 0.079 [0.064, 0.096] | 0.855 [0.848, 0.862] | 0.00 | 618 / 311 | 0.0414 | 0.0569 |
| TU, T 0.6 | 0.424 [0.379, 0.472] | 0.871 [0.856, 0.882] | 0.60 | 203 / 270 | 0.7296 | 0.0075 |
| TU, T 0.8 | 0.226 [0.188, 0.262] | 0.842 [0.831, 0.853] | 0.00 | 192 / 272 | 0.2766 | 0.0386 |
| TU, T 1.0 | 0.084 [0.063, 0.108] | 0.762 [0.753, 0.772] | 0.00 | 208 / 301 | **0.0730** | 0.0993 |
| TU, pooled | 0.244 [0.218, 0.267] | 0.824 [0.817, 0.831] | 0.20 | 603 / 281 | 0.3597 | 0.0497 |
| TB, T 0.6 | 0.423 [0.375, 0.468] | 0.868 [0.851, 0.881] | 0.67 | 194 / 269 | 0.7755 | 0.0053 |
| TB, T 0.8 | 0.299 [0.262, 0.341] | 0.847 [0.834, 0.860] | 0.51 | 200 / 263 | 0.2982 | 0.0346 |
| **TB, T 1.0** | **0.110 [0.084, 0.139]** | 0.757 [0.747, 0.766] | 0.00 | 208 / 298 | **0.0769** | 0.1044 |
| TB, pooled | 0.274 [0.249, 0.298] | 0.823 [0.814, 0.830] | 0.39 | 602 / 277 | 0.3834 | 0.0492 |

**Continuation bits per byte** (the 250 human continuations, exact through the bridge; paired with D0 over prompts):

| Model | Bits per byte [95%] | Minus D0, paired [95%] |
|---|---|---|
| D0 byte default | 2.1339 [2.1016, 2.1659] | |
| TU | 2.0292 [1.9975, 2.0605] | −0.1047 [−0.1161, −0.0922] |
| TB | 2.0260 [1.9954, 2.0556] | −0.1079 [−0.1193, −0.0955] |

TB minus TU: −0.0032 [−0.0077, +0.0012].

**The dev choice, by the registered steps.**
1. Eligible (copy share at most 0.0786): TU at T 1.0 (0.0730) and TB at T 1.0 (0.0769). Every arm at T 0.6 and 0.8 copies far more (0.28 to 0.78).
2. Chosen: **TB at T 1.0**, SCRIBE 0.110 against TU's 0.084 (more than 0.005 apart).
3. Its point estimate is above the comparator's 0.101, so the test read is made. The dev intervals overlap almost fully ([0.084, 0.139] against [0.072, 0.135]), so a pass on test (disjoint intervals) is unlikely.

**What dev shows.**
- **SCRIBE's high scores at T 0.6 are copying.** Three quarters of those bytes lie in 32-byte runs of the training text. The token ∞-gram follows long matches, and at low temperature it rarely leaves them. SCRIBE does not penalise copying from the training slice (its D_I counts only the pack), so the copy rule is what keeps this out.
- **At T 1.0 the token arms are as broken as the byte model.** Invented words are 10% for both (D0 at T 1.0: 10.2%). The samples are list markup, interwiki links and fragments.
- **Whole-token sampling still makes invented words.** A word spelled from two subword pieces that never met in training is invented, so 4,096 tokens do not stop it.
- **The bits are better.** Both token models predict the human prose 0.105 bits per byte better than the byte default, paired. That is in line with T0–T2 (TOK-A's 1.904 against 2.073 on the dev holdout).

### T4 test read (2026-10-29, read once)

Commands: `cyphalm_tok_gen gen … t-bpe-4096.tok runs/t4/test/TB.gen.txt --prompts-file bench/eval/gen_prompts_test.tsv --temps 1.0`, then SCRIBE's ladder and the summary as on dev. B0: `scripts/gen_eval.sh test B0 runs/t4/test` with GEN-A's `cyphalm_trie_91827e1` (today's default, no flags). Wall times: TB 353 s, B0 1,174 s.

| Arm | SCRIBE v2 [95%] | Surface S [95%] | D_I | Kept / mean bytes | Copy share | Invented words |
|---|---|---|---|---|---|---|
| **TB, T 1.0** (chosen on dev) | **0.112 [0.086, 0.137]** | 0.766 [0.755, 0.777] | 0.00 | 197 / 272 | **0.0746** | 0.1018 |
| **B0 byte, T 0.6** (comparator) | **0.106 [0.078, 0.135]** | 0.901 [0.893, 0.909] | 0.00 | 206 / 334 | **0.0561** | 0.0243 |
| B0 byte, T 0.8 | 0.064 [0.045, 0.086] | 0.883 [0.876, 0.891] | 0.00 | 207 / 321 | 0.0369 | 0.0453 |
| B0 byte, T 1.0 | 0.046 [0.027, 0.065] | 0.797 [0.787, 0.806] | 0.00 | 204 / 290 | 0.0227 | 0.0991 |
| B0 byte, pooled | 0.072 [0.059, 0.085] | 0.861 [0.855, 0.866] | 0.00 | 617 / 315 | 0.0386 | 0.0557 |

**Continuation bits per byte:** TB 2.0263 [1.9921, 2.0620], B0 2.1296 [2.0956, 2.1659]. TB minus B0, paired: **−0.1033 [−0.1142, −0.0927]**.

**The gate, as registered.**
- SCRIBE: TB 0.112 against B0 at T 0.6 0.106. The intervals overlap. **Not met.**
- Copy share: 0.0746 against 0.0561 + 0.02 = 0.0761. Met, narrowly.

**Decision: the token sampler is not adopted.** `sample`'s default stays the byte model. `cyphalm_tok_gen` stays a measuring tool. No default of any tool changed.

**Reported, not deciding.**
- At the same temperature the token model is far ahead. At T 1.0, TB scores 0.112 against B0's 0.046, with disjoint intervals. Against B0's pooled 0.072 [0.059, 0.085] it is also disjoint. But the registered comparator is each model at its own best dev temperature, and the byte model at T 0.6 is as good.
- Surface S is lower for TB (0.766 against 0.901). The T 1.0 token samples are more often list markup, interwiki links and broken fragments.
- Test agrees with dev in direction and size (dev 0.110 against 0.101).

**Samples** (test, the first prompts; TB at T 1.0, B0 at T 0.6):

- Prompt: "The military has dominated Algerian politics up to today. …"
  - TB: "N) argued in that alreadyagefer from [[Buty]] [[da:Attila]] [[de:Analyy come</title> <id>1331</id> …"
  - B0: "The Anglican Alps in the early [[Middle Ages and [[Alfred perform of a certain theoretical of the [[Trans-Mississippi Theater …"
- Prompt: "Guy de Maupassant is buried in the Cimetière du Montparnasse, Paris. …"
  - TB: "1906 as 1996 the [[semiconductor]] that serial killer (d. [[1838]]) *Tindouchi]], he later adopted the ideas …"
  - B0: "==Events ''[[The cartoon first company, Iowa.html Arthur Einstein]] [[Category:Alternative deaths|Karpov, Anatoly]] …"
- Prompt: "It was apparent to ancient humans that offspring resembled their parents. …"
  - TB: "{{ref|4500 ft/s) and bring him form investigations (i.e.php |mer]] and [[liquid of knowledge and oftenons telem …"
  - B0: "The [[Republican of the [[Marxist]] and [[John Williams Reinholt]], and in and a single in the sense of the [[Middle century]] …"
- Prompt: "As of 2004, 45 of the 96 GSSPs required have been approved. …"
  - TB: "alone might song. ==Comraventional|U.S. House of Commander 2 hours and to him was dismant territorials== …"
  - B0: "The last [[Carthus as a result, Arabic speaking language|Hebrew]] and are usually a simple of an action of army …"
- Prompt: "Application program functions such as termination, page-up, page-down, or help …"
  - TB: "By the end of the [[PHistidine]] (CH&lt;sub&gt;3&lt;/sub&gt;4&lt;/sup&gt;213 |[[Haspreserve Einstein and fossils …"
  - B0: "Also, when considered the of science of space the of screen a stralian character points to be found to be a common …"

### T4 predictions, scored

| # | Prediction | Result |
|---|---|---|
| P-T2 | Token samples score SCRIBE v2 ≥ 0.15, and the invented-word rate falls below 1% | **Miss.** The chosen arm scores 0.112 on test with 10.2% invented words. T 0.6 reaches 0.42 with under 1%, but by copying three quarters of its bytes |
| B6 | The byte model's invented-word rate is above 10% | **Miss.** 5.7% pooled on dev (5.6% on test). Only T 1.0 reaches 10% (10.2% dev, 9.9% test) |
| Q1 | Copy share falls as T rises, both tokenizers | **Hit:** TU 0.730 > 0.277 > 0.073; TB 0.776 > 0.298 > 0.077 |
| Q2 | Both copy more than 0.30 at T 0.6 | **Hit:** 0.730 and 0.776 |
| Q3 | Only T 1.0 eligible, at least one tokenizer | **Hit:** both T 1.0 arms eligible, no other |
| Q4 | Continuation bits per byte 0.10–0.25 below D0, paired interval below 0 | **Hit:** −0.105 and −0.108 on dev (−0.103 on test), intervals below 0 |
| Q5 | Chosen arm's dev SCRIBE in [0.06, 0.15] | **Hit:** 0.110 |
| Q6 | Token invented-word rate below D0's at every temperature | **Miss:** TB at T 1.0 is 0.1044 against D0's 0.1017. TU is below at every temperature |
| Q7 | D_I above 0 for both at T 0.6 | **Hit:** 0.60 and 0.67 |

6 hits, 3 misses.

### T4 status

**Done (2026-10-29). Not adopted.** Whole-token sampling from TOK-A's model does not beat the byte model at each model's best temperature. Its SCRIBE is high only where it copies the training text. Where it copies no more than the byte model, it is as broken as the byte model, with the same rate of invented words.

What T4 does show:
- **The token model predicts prose better.** About 0.10 bits per byte better than the byte default, exact through the bridge, on dev and on test.
- **The ∞-gram drives copying.** At T 0.6 and 0.8 the token sampler copies 28–78% of its bytes. The byte default copies 2–6%, because its nucleus cut and repeat ban keep it from following one long match.

**Next design (written down, not run).** The ∞-gram's weight at long matches is what makes the low temperatures copy. Two changes, each to be registered and read on GEN-A's dev split first:
1. **Cap the token ∞-gram** in sampling: a match-length cap (for example 8 tokens) or a λ ceiling at long matches. Then a lower temperature, where SCRIBE is higher, could meet the copy rule.
2. **Use the byte sampler's nucleus cut (0.9) and repeat ban at the token level**, so the two samplers differ only in what they sample.

The word-level gap (sentence form, D_S) remains. Tokens fix spelling at low temperature, not grammar.

**Cost.** About 380 MB peak resident per run, about 30 minutes per dev arm under load (three temperatures and the continuation bits), 6 minutes for the single-temperature test arm.

## T4b: a token sampler that does not follow long matches (registered 2026-10-29, before any T4b run)

**Odin Loch** · 2026-10-29 · decided by measurement under PLAN ledger 55, open to Odin's veto

**Question.** T4 left two designs for Odin's choice. Both target the same cause: at T 0.6 and 0.8 the token sampler copies 28–78% of its bytes, because the token ∞-gram follows long training matches. T4b settles the choice on GEN-A's dev split.

**The fitted weights, read before choosing the grid** (λ of the ∞-gram per match-length bucket, from the holdout's first half; `cyphalm_tok_gen gen` now prints them):

| Match length (tokens) | Count 1 | 2–3 | 4–15 | 16+ |
|---|---|---|---|---|
| 1 | 0 | 0.995 | 0 | 0.145 |
| 2, 3–4 | 0 | 0 | 0 | 0 |
| 5–8 | 0.11 | 0.08 | 0.10 | 0.05 |
| 9–16 | 0.315 | 0.29 | 0.215 | 0.20 |
| 17+ | 0.60 | 0.315 | 0.50 | 0.38 |

So a ceiling on λ of 0.3 or 0.5 would barely bind (only above 16 tokens), and a match cap of 4 tokens is almost the same as no ∞-gram (λ is 0 for 2–4 tokens). The grid below uses match caps instead.

**Arms** (all BPE-W 4,096, TOK-A's model, temperatures 0.6, 0.8 and 1.0, as T4's TB; only sampling changes):
- **C1, the ∞-gram capped when sampling** (`--inf-cap M`): the ∞-gram reads its match from at most the last M tokens. K0 (M = 0: no ∞-gram, KN alone), K8 (M = 8), K16 (M = 16). T4's TB is M = 64.
- **C2, the byte sampler's rules on tokens** (`--nucleus 0.9 --no-repeat`, label NR): the full model (M = 64). Per token: a token is banned if any of its bytes, written in turn, would follow an 11-byte window with the byte that followed the same window earlier in the text (prompt included). That is `mark_banned`'s rule, byte by byte. If every token with mass is banned, nothing is banned. Then q^(1/T), the smallest set of tokens holding at least 0.9 of the mass, and one draw within it. Healing (the first token, a cover past the prompt's end) stays the bridge's draw, as in T4.
- **C3, both** (label NRK*): C2 with the best C1 cap. The best C1 cap is the one whose best eligible arm (dev rule below) has the highest dev SCRIBE. If no C1 arm is eligible, it is the cap with the lowest pooled copy share. C3 runs after C1's results.

**Unchanged by design.** The human continuations are always scored by T4's model (M = 64, no nucleus) through the bridge. So every arm's continuation bits per byte must equal TB's 2.0260 exactly. That is checked, not assumed. With no T4b flag the tool reproduces T4's TB file byte for byte (checked on the first 10 dev prompts: 41 lines identical, and the continuation bits).

**Checks of the new code.** `cyphalm_tok_gen check-t4b` compares the lifted repeat rule with the byte rule applied byte by byte, on random texts over a two-letter alphabet: 18,000 checks, 1,383 of them banned, 0 failed. Two planted bugs fail it: checking only a token's first byte (755 failed), and missing windows that straddle a token boundary (65 failed).

**The dev choice** (as T4's):
1. Each arm is one setting at one temperature. SCRIBE v2 is read per temperature.
2. **Eligible:** dev copy share at most the byte comparator's (GEN-A's D0 at T 0.6: 0.0586) + 0.02, that is at most **0.0786**.
3. **Chosen:** the eligible arm with the highest dev SCRIBE, among the T4b arms and T4's TB at T 1.0 (0.110 [0.084, 0.139]; copy 0.0769). Within 0.005, the lower copy share wins.
4. **Test read only if** the chosen arm is a T4b arm and its dev SCRIBE point estimate is above **0.110** (T4's TB at T 1.0, itself above the byte comparator's 0.101). A T4b arm that does not beat T4's arm on dev is not a change worth a frozen-test read.

**The test read (once).** The chosen arm at its temperature (`--temps T`) on `gen_prompts_test.tsv`. The byte comparator is T4's B0 test run, reused: the same prompts file, the same binary (`cyphalm_trie_91827e1`, no flags) and a deterministic sampler, so a rerun would be bit-identical. **Gate:** SCRIBE above B0 at T 0.6 (0.106 [0.078, 0.135]) with disjoint 95% intervals, and the copy share at most B0's at T 0.6 + 0.02 (0.0761). Logged in GEN_TRACK's Reads table. A pass makes the sampler a candidate for `sample`'s default; the switch is a later integration step.

**Reported together**, every arm: SCRIBE v2 [95%] with surface S, D_I, kept samples and mean length, continuation bits per byte, copy share, invented-word rate, five samples.

**Commands** (`/root/work/tokc/runs/t4b/arm.sh SPLIT LABEL [gen flags]`):

```
B=/root/work/tokc/build/cyphalm_tok_gen; E=/root/cdata/enwik8; TK=/root/work/toka/runs/tok/t-bpe-4096.tok
SC=/root/work/gena/runs/scribe_gena; O=/root/work/tokc/runs/t4b/dev
$B gen $E $TK $O/K8.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv --inf-cap 8 > $O/K8.sample.out
$SC ladder --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors /root/work/scribe/vec32 \
   --profile v2 --seed 20261027 --gen-trie $O/K8.gen.txt --label K8 > $O/K8.scribe.md
python3 -I scripts/gen_eval_summary.py $O K8 dev
$B words $E $O/K8.gen.txt
```

**Predictions** (registered before any run):

| # | Prediction | Falsified if |
|---|---|---|
| R1 | K0 (KN alone) still copies more than 0.10 of its bytes at T 0.6: order-5 KN also follows contexts seen once | K0's T 0.6 copy share is at most 0.10 |
| R2 | Copying grows with the cap at every temperature: K0 ≤ K8 ≤ K16 ≤ TB | Any step falls by more than 0.01 |
| R3 | C2 (NR) copies more than TB at T 0.8 and 1.0: the nucleus cuts the tail and leaves the long match | NR's copy share is below TB's at either |
| R4 | At least one T4b arm at T 0.6 or 0.8 is eligible. About even odds | None is |
| R5 | The chosen arm's dev SCRIBE is in [0.08, 0.20] | Outside |
| R6 | Every arm's continuation bits per byte is 2.0260, as TB | Any differs |
| R7 | NR at T 1.0 invents fewer words than TB at T 1.0 (0.104) | Not fewer |
| R8 | If a test read is made, the gate is not met (the intervals overlap). About 70% | The gate is met |

### T4b dev results (2026-10-29, before the test read)

Tool: `cyphalm_tok_gen` at 7300e77. Four arms ran side by side at load about 8 on 4 cores: K0 974 s, K8 978 s, K16 998 s, NR 1,547 s; then NRK16 1,903 s. Ladder 27–34 s each. The byte comparator D0 and T4's TB are read as they stand.

| Arm, temperature | SCRIBE v2 [95%] | Surface S | D_I | Kept / mean bytes | Copy share | Invented words |
|---|---|---|---|---|---|---|
| D0 byte, T 0.6 (comparator) | 0.101 [0.072, 0.135] | 0.900 | 0.00 | 213 / 328 | 0.0586 | 0.0251 |
| TB (T4, cap 64), T 1.0 | 0.110 [0.084, 0.139] | 0.757 | 0.00 | 208 / 298 | 0.0769 | 0.1044 |
| K0 (no ∞-gram), T 0.6 / 0.8 / 1.0 | 0.379 / 0.273 / **0.090 [0.067, 0.114]** | 0.879 / 0.845 / 0.760 | 0.06 / 0 / 0 | 189 / 197 / 206 | 0.310 / 0.147 / **0.0472** | 0.014 / 0.042 / 0.107 |
| K8, T 0.6 / 0.8 / 1.0 | 0.360 / 0.289 / **0.085 [0.062, 0.109]** | 0.877 / 0.845 / 0.761 | 0 / 0 / 0 | 186 / 189 / 213 | 0.465 / 0.207 / **0.0659** | 0.010 / 0.039 / 0.104 |
| K16, T 0.6 / 0.8 / 1.0 | 0.402 / 0.293 / **0.112 [0.085, 0.141]** | 0.875 / 0.845 / 0.756 | 0.23 / 0.13 / 0 | 183 / 197 / 210 | 0.663 / 0.284 / **0.0749** | 0.008 / 0.036 / 0.105 |
| NR (nucleus, no repeat), T 0.6 / 0.8 / 1.0 | 0.607 / 0.339 / 0.181 [0.147, 0.216] | 0.894 / 0.873 / 0.804 | 0 / 0 / 0 | 216 / 200 / 205 | 0.897 / 0.464 / 0.116 | 0.003 / 0.016 / 0.070 |
| NRK16 (C3), T 0.6 / 0.8 / 1.0 | 0.452 / 0.365 / 0.180 [0.147, 0.216] | 0.879 / 0.862 / 0.803 | 0 / 0 / 0 | 198 / 195 / 206 | 0.804 / 0.415 / 0.109 | 0.004 / 0.017 / 0.071 |

Mean bytes kept are 245–301 for every token arm. **Continuation bits per byte: 2.0260 [1.9954, 2.0556] in every arm.** Each arm's `.cont.tsv` is byte-identical to TB's, as the design requires.

**The dev choice, by the registered steps.**
1. **Eligible** (copy share at most 0.0786): K0, K8 and K16 at T 1.0, and T4's TB at T 1.0. No arm at T 0.6 or 0.8 is eligible; NR and NRK16 are eligible at no temperature.
2. **Best C1 cap** (for C3): K16 (0.112). So C3 was NRK16.
3. **Chosen:** K16 at T 1.0 (0.112, copy 0.0749). It is within 0.005 of TB (0.110, copy 0.0769), and the lower copy share wins.
4. **Test read:** K16 is a T4b arm and 0.112 is above 0.110, so the rule calls for the read. The margin is 0.002, inside the tie band: the read was made because the rule says so, not because dev shows a real gain.

**What dev shows.**
- **Copying is not only the ∞-gram.** With no ∞-gram at all (K0), the sampler still copies 31% of its bytes at T 0.6 and 15% at T 0.8. Order-5 Kneser–Ney follows contexts seen once too. Capping the ∞-gram lowers copying at every temperature (K0 < K8 < K16 < TB), but not to the byte model's level below T 1.0.
- **The nucleus cut and repeat ban are the larger lever on SCRIBE, and they copy more.** NR at T 1.0 scores 0.181 [0.147, 0.216], with an interval clear of the byte comparator's, and invents 7.0% of its words against TB's 10.4%. But it copies 11.6% of its bytes (the limit is 7.9%). Cutting the tail leaves the long match as the likely token. Capping the ∞-gram at 16 tokens (NRK16) barely changes it (0.109).
- **SCRIBE rises with copying across these arms.** Within the eligible set (copy at most 0.0786), every arm scores 0.085–0.112.

### T4b test read (2026-10-29, read once)

Command: `arm.sh test K16 --inf-cap 16 --temps 1.0` (`cyphalm_tok_gen gen … --prompts-file bench/eval/gen_prompts_test.tsv --inf-cap 16 --temps 1.0`, then the ladder and the summary). 510 s. The byte comparator is T4's B0 test run, reused.

| Arm | SCRIBE v2 [95%] | Surface S [95%] | D_I | Kept / mean bytes | Copy share | Invented words |
|---|---|---|---|---|---|---|
| **K16, T 1.0** | **0.115 [0.091, 0.143]** | 0.765 [0.754, 0.776] | 0.00 | 198 / 275 | **0.0728** | 0.1017 |
| B0 byte, T 0.6 (T4's run) | 0.106 [0.078, 0.135] | 0.901 [0.893, 0.909] | 0.00 | 206 / 334 | 0.0561 | 0.0243 |
| T4's TB, T 1.0 (for reference) | 0.112 [0.086, 0.137] | 0.766 | 0.00 | 197 / 272 | 0.0746 | 0.1018 |

**Continuation bits per byte:** 2.0263 [1.9921, 2.0620], byte-identical per prompt to T4's TB test run; minus B0, paired: −0.1033 [−0.1142, −0.0927].

**The gate.** SCRIBE 0.115 against 0.106: the intervals overlap. **Not met.** Copy share 0.0728 against 0.0761: met.

**Decision: not adopted.** The capped sampler is the same as T4's on test (0.115 against 0.112). `cyphalm_tok_gen`'s new flags stay measuring options. No default changed.

### T4b predictions, scored

| # | Prediction | Result |
|---|---|---|
| R1 | K0 still copies more than 0.10 at T 0.6 | **Hit:** 0.310 |
| R2 | Copying grows with the cap at every temperature (K0 ≤ K8 ≤ K16 ≤ TB) | **Hit:** T 0.6 0.31 / 0.47 / 0.66 / 0.78; T 0.8 0.15 / 0.21 / 0.28 / 0.30; T 1.0 0.047 / 0.066 / 0.075 / 0.077 |
| R3 | NR copies more than TB at T 0.8 and 1.0 | **Hit:** 0.464 against 0.298; 0.116 against 0.077 |
| R4 | A T4b arm at T 0.6 or 0.8 is eligible | **Miss:** the lowest is K0 at T 0.8, 0.147 |
| R5 | The chosen arm's dev SCRIBE is in [0.08, 0.20] | **Hit:** 0.112 |
| R6 | Every arm's continuation bits per byte is 2.0260 | **Hit:** byte-identical files, dev and test |
| R7 | NR at T 1.0 invents fewer words than TB | **Hit:** 0.070 against 0.104 |
| R8 | A test read does not meet the gate | **Hit:** overlapping intervals |

7 hits, 1 miss.

### Samples (T 1.0, the first five prompts of each split, cut to 150 bytes)

**K0 (dev, T 1.0):**
- 5-[[dications shorter distances in Ethiopia]] which played in a [[h lyreement prevalence of the state.  In 18 s pers For this is that in August]] / [[ …
- cunyes to ready to researcher with a glass [[Arno through several volumes the 4thus free-prop | color1=#ffff99 | color2=green.  While language and adv …
- /  / With the breakh only one)''': / * [[Analysis.'' emer structure or as the fact that absolutely == / ===S texts instances of thised characters do.  …
- org/ would looking c small, Athens&quot;, ''l time asae redire and as Lincoln keptomus]] / |- / |0100&amp;nbsp;&amp; consonants or [[norming in [[Con  …
- and the beginning to a foeur Ruma. /  / * Num was mentioned above, unlike all other formsunk Isms (aor phase of St. George'cos''), meaning to Northern …

**K8 (dev, T 1.0):**
- 5-[[dications shorter distancesan (or ''m), the Employment, a ship's doctrine.jpg|In some ( tribes. At free kicks and field of the writer Information  …
- cunyes to ready to researcher with a glass [[Arno through several volumes. The best-known_omer A will in [[laserks in [[1976]] album, and surviving]]' …
- /  / With [[poetry]], [[Anglican]] in New York City features, out of contend that he was ex when [[earthquake]] near [[medicine of overlooking/cub at  …
- org/asterhar]] / The [[Angola)]], 1927]] - [[Characters in Atlas_Shrugged|Druid]] / *[[par epoch the head of the Academy's [[light]]. / * RFC 1908 and …
- [[Buddhisme]] / [[de:Aktuir]].  Records|ely water orphans Center''' which is the stretchesly viciousidedG, Casa formal Algeria: The oil would flying [ …

**K16 (dev, T 1.0):**
- 5-[[dications shorter distancesan (or ''m), the Employment, a ship's doctrine.jpg|In some ( tribes. At free kicks and field of the writer Information  …
- cunyes to ready to researcher with a glass [[Arno through several volumes. The best-known? spentow Jr. by the original by Hollow abolished) / |} /  /  …
- /  / With [[poetry]], [[Anglican]] in New York City features, out of contend that he was ex when [[earthquake]] near [[medicine of overlooking/cub at  …
- org/asterhar]] / The [[Angola)]], 1927]] - [[Characters in Atlas_Shrugged|Debus| /  sat served soil'' '''en Library Museum  in the true Johns Hopkins  …
- [[Buddhisme]] / [[de:Aktuir]].  Records|ely water orphans Center''' which is the stretchesly viciousidedG, Casa formal Algeria: The oil would flying [ …

**NR (dev, T 1.0):**
- 5-lov crimes. /  / It has been suggested that the field of view mainland, Mississippipeabi Remix EP) and Paul II Balli]], the Talibanate a source.13&q …
- cnrn working under a different meaning thanked before theological work of actions from him with generated and is difficult to natives and by [[patent] …
- /  / ==M== / {{details|Abbey ??obul, Publishing, whereas in which he said, &quot;lackeries and attempts === / * [[Madur.4%) of the apocryphal substant …
- org/contraceptive] / |- / | '''[[List of which had startediba]], [[1874]] / |Democratic / |- bgcolor=#DDEEFF / |[[List of country call it became the p …
- [[1</ip> /       </contributor> /       <comment>Removing root]] of Markowensized Alchemy has been closer competition in traffic. /  / Tex|Mobile purp …

**NRK16 (dev, T 1.0):**
- 5-lov crimes. /  / It has been suggested that the field of view mainland, Mississippipeabi Remix EP) and Paul II Balli]], the Talibanate a source.13&q …
- cnrn working under a different meaning thanked before theological work of actions from him with generated and is difficult to natives and by [[patent] …
- /  / ==M== / {{details|Abbey ??obul, Publishing, whereas in which he said, &quot;lackeries and attempts === / * [[Madur.4%) of the apocryphal substant …
- org/contraceptive] / |- / | '''[[List of which had startediba]], [[1874]] / |Democratic / |- bgcolor=#DDEEFF / |[[List of country call it became the p …
- [[1</ip> /       </contributor> /       <comment>Removing root]] of Markowensized Alchemy has been closer competition in traffic. /  / Tex|Mobile purp …

**K16 (test, T 1.0):**
- N) argued in that alreadyagefer from [[Buty]] / [[da:Attila]] / [[de:Analyy come</title> /     <id>1331</id> /       </contributor> /       <comment>{ …
- 1906 as 1996 the [[semiconductor]] that serial killer (d. [[1838]]) / *Tindouchi]], he later adopted the ideas onin]] and [[Methodism's into a gas sub …
- {{ref|4500 ft/s) and bring him form investigations (i.e.php |mer]] and [[liquid of knowledge and oftenons telem :aid of the most common function contr …
- alone might song. /  / ==Comraventional|U.S. House of Commander 2 hours and to him was dismant territorials== /  / Astron. ''The Americanid nowis. Amb …
- By the end of the [[PHistidine]] (CH&lt;sub&gt;3&lt;/sub&gt;3&lt;/sub&gt;Henry' can beokhy Select in very earlyisies]] and regna Kn?? for all43). Supe …

### T4b status

**Done (2026-10-29). Not adopted.** Capping the token ∞-gram when sampling lowers copying but does not lift SCRIBE: the best eligible arm (K16, T 1.0) equals T4's on test. The byte sampler's nucleus and repeat ban lift SCRIBE (dev 0.181 at T 1.0, clear of the byte model) but raise copying past the rule, so they are not eligible.

**Next designs (written down, not run; each needs its own registration):**
1. **NR with K0 at T 1.0:** the nucleus and repeat ban over Kneser–Ney alone. K0 copies 0.030 less than TB at T 1.0, NR 0.039 more, so the pair may sit at the limit.
2. **A wider nucleus** (0.95 or 0.97) with the repeat ban, which cuts less of the tail.
3. **A no-copy rule when sampling:** a token is banned when writing it would extend a span already found in the training text past 32 bytes. It is the repeat ban's twin, aimed at the training text. It looks text up, so it needs Odin's ruling under PLAN rule 10 (it gates; it never writes) before it is tried.

**Cost.** About 380 MB peak resident per arm. About 16 minutes per three-temperature arm at load 8 (25–32 minutes with the nucleus, which sorts 4,096 tokens per step); 8.5 minutes for the one-temperature test arm.

## T4c: the nucleus and repeat ban over Kneser–Ney alone, and a wider nucleus (registered 2026-10-29, before any T4c run)

**Odin Loch** · 2026-10-29 · decided by measurement under PLAN ledger 55, open to Odin's veto

**Question.** T4b found that the byte sampler's nucleus 0.9 and repeat ban (NR) lift dev SCRIBE at T 1.0 to 0.181, clear of the byte model, but copy 11.6% of their bytes (the limit is 7.9%). Kneser–Ney alone (K0) copies least. T4b wrote down two next designs: NR over Kneser–Ney alone, and a wider nucleus. T4c reads both on GEN-A's dev split. A no-copy sampling rule is not tried: it looks text up, so it needs Odin's ruling under PLAN rule 10.

**Tool.** `cyphalm_tok_gen` as on main 84c0547 (its source is T4b's 7300e77). No code changes: the flags exist (`--inf-cap`, `--nucleus`, `--no-repeat`, `--temps`). This build reproduces T4b's K0 dev file byte for byte (first 3 prompts, 13 lines) and its continuation bits, so T4b's dev arms are reused as they stand.

**Arms** (BPE-W 4,096, TOK-A's model, as T4 and T4b; only sampling changes):

| Label | Flags | Temperatures | Note |
|---|---|---|---|
| NRK0 | `--inf-cap 0 --nucleus 0.9 --no-repeat` | 0.8, 1.0 | NR over Kneser–Ney alone. T 0.6 is not run: K0 alone copies 0.310 there and NR raises copying |
| N95K0 | `--inf-cap 0 --nucleus 0.95 --no-repeat` | 1.0 | a wider nucleus, no ∞-gram |
| N97K0 | `--inf-cap 0 --nucleus 0.97 --no-repeat` | 1.0 | |
| N95K16 | `--inf-cap 16 --nucleus 0.95 --no-repeat` | 1.0 | a wider nucleus, T4b's chosen cap |
| N97K16 | `--inf-cap 16 --nucleus 0.97 --no-repeat` | 1.0 | |

Reused from T4b (dev, the same binary source and settings): K0, K16 (nucleus 1, no repeat ban), NR (cap 64, nucleus 0.9) and NRK16 (cap 16, nucleus 0.9). So at T 1.0 the grid has nucleus {0.9, 0.95, 0.97, 1} × cap {0, 16}.

**Unchanged by design.** The human continuations are scored by T4's model (cap 64, no nucleus) through the bridge, so every arm's continuation bits per byte must be 2.0260 on dev, with each `.cont.tsv` byte-identical to T4b's.

**The dev choice** (as T4b's):
1. Each arm is one setting at one temperature. SCRIBE v2 is read pooled at that temperature.
2. **Eligible:** dev copy share at most the byte comparator's (GEN-A's D0 at T 0.6: 0.0586) + 0.02, that is at most **0.0786**.
3. **Chosen:** the eligible arm with the highest dev SCRIBE, among the T4c arms and the eligible arms already read (K16 at T 1.0, 0.112; T4's TB at T 1.0, 0.110; K0 0.090; K8 0.085). Within 0.005 (T4b's tie band), the lower copy share wins.
4. **Test read only if** the chosen arm is a T4c arm and its dev SCRIBE point estimate is above **0.106** (the byte comparator's 0.101 plus the tie band) **and** above **0.117** (K16's 0.112, the best token arm already read on test, plus the tie band). An arm that ties K16 on dev is not a change worth a second frozen-test read; K16's test read already answered that question.

**The test read (once).** The chosen arm at its temperature (`--temps T`) on `gen_prompts_test.tsv`. The byte comparator is T4's B0 test run, reused (the same prompts, binary `cyphalm_trie_91827e1` with no flags, a deterministic sampler). **Gate:** SCRIBE above B0 at T 0.6 (0.106 [0.078, 0.135]) with disjoint 95% intervals, and the copy share at most B0's at T 0.6 + 0.02 (0.0761). Logged in GEN_TRACK's Reads table. A pass makes the sampler a candidate for `sample`'s default; the switch is a later integration step.

**Reported together**, every arm: SCRIBE v2 [95%] with surface S, D_I, kept samples and mean length, continuation bits per byte, copy share, invented-word rate, five samples.

**Commands** (`/root/work/tokd/runs/t4c/arm.sh SPLIT LABEL [gen flags]`, T4b's `arm.sh` with paths changed):

```
B=/root/work/tokd/build/cyphalm_tok_gen; E=/root/cdata/enwik8; TK=/root/work/toka/runs/tok/t-bpe-4096.tok
SC=/root/work/gena/runs/scribe_gena; O=/root/work/tokd/runs/t4c/dev
$B gen $E $TK $O/NRK0.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv --inf-cap 0 --nucleus 0.9 --no-repeat --temps 0.8,1.0 > $O/NRK0.sample.out
$SC ladder --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors /root/work/scribe/vec32 \
   --profile v2 --seed 20261027 --gen-trie $O/NRK0.gen.txt --label NRK0 > $O/NRK0.scribe.md
python3 -I scripts/gen_eval_summary.py $O NRK0 dev
$B words $E $O/NRK0.gen.txt
```

**Predictions** (registered before any run):

| # | Prediction | Falsified if |
|---|---|---|
| U1 | NRK0 at T 1.0 copies less than NRK16 at T 1.0 (0.109): the ∞-gram is part of NR's copying | NRK0's T 1.0 copy share is 0.109 or more |
| U2 | NRK0 at T 1.0 is eligible (copy at most 0.0786). About even odds | Its copy share is above 0.0786 |
| U3 | NRK0 at T 0.8 is not eligible: K0 alone copies 0.147 at T 0.8 and the nucleus raises copying | Its copy share is at most 0.0786 |
| U4 | At a fixed cap and T 1.0, widening the nucleus lowers copying: nucleus 0.9 ≥ 0.95 ≥ 0.97 ≥ 1 | Any step up in the nucleus raises copying by more than 0.005 |
| U5 | At a fixed cap and T 1.0, widening the nucleus lowers SCRIBE: nucleus 0.9 ≥ 0.95 ≥ 0.97 ≥ 1 | Any step up in the nucleus raises SCRIBE by more than 0.01 |
| U6 | The best eligible T4c arm's dev SCRIBE is in [0.12, 0.17] | Outside, or no T4c arm is eligible |
| U7 | Every arm's continuation bits per byte is 2.0260, `.cont.tsv` byte-identical to T4b's | Any differs |
| U8 | Every T4c arm at T 1.0 invents fewer words than TB at T 1.0 (0.104) | Any arm invents 0.104 or more |
| U9 | If a test read is made, the gate is not met (the intervals overlap). About 60% | The gate is met |

### T4c dev results (2026-10-29, before the test read)

Tool: `cyphalm_tok_gen` built from main 84c0547 (source as T4b's 7300e77). The five arms ran side by side at load 7–10 on 4 cores: NRK0 794 s (two temperatures), the others 583–608 s; ladder 11–27 s each. T4b's arms and the byte comparator D0 are read as they stand.

| Arm, temperature | Nucleus | ∞-gram cap | SCRIBE v2 [95%] | Surface S | Kept / mean bytes | Copy share | Invented words |
|---|---|---|---|---|---|---|---|
| D0 byte, T 0.6 (comparator) | 0.9 (byte) | (byte) | 0.101 [0.072, 0.135] | 0.900 | 213 / 328 | 0.0586 | 0.0251 |
| **NRK0, T 1.0** | **0.9** | **0** | **0.192 [0.157, 0.225]** | 0.811 | 200 / 276 | **0.0728** | 0.0701 |
| NRK0, T 0.8 | 0.9 | 0 | 0.344 [0.306, 0.384] | 0.868 | 208 / 263 | 0.1945 | 0.0235 |
| N95K0, T 1.0 | 0.95 | 0 | 0.119 [0.093, 0.147] | 0.786 | 212 / 293 | 0.0647 | 0.0817 |
| N97K0, T 1.0 | 0.97 | 0 | 0.134 [0.105, 0.164] | 0.776 | 206 / 279 | 0.0590 | 0.0868 |
| K0, T 1.0 (T4b) | 1 | 0 | 0.090 [0.067, 0.114] | 0.760 | 206 / – | 0.0472 | 0.107 |
| NRK16, T 1.0 (T4b) | 0.9 | 16 | 0.180 [0.147, 0.216] | 0.803 | 206 / – | 0.109 | 0.071 |
| N95K16, T 1.0 | 0.95 | 16 | 0.132 [0.106, 0.163] | 0.789 | 211 / 283 | 0.0919 | 0.0806 |
| N97K16, T 1.0 | 0.97 | 16 | 0.130 [0.104, 0.160] | 0.779 | 211 / 287 | 0.0793 | 0.0850 |
| K16, T 1.0 (T4b) | 1 | 16 | 0.112 [0.085, 0.141] | 0.756 | 210 / – | 0.0749 | 0.105 |

D_I is 0.00 in every arm. **Continuation bits per byte: 2.0260 [1.9954, 2.0556] in every arm**, each `.cont.tsv` byte-identical to T4b's.

**The dev choice, by the registered steps.**
1. **Eligible** (copy share at most 0.0786): NRK0 at T 1.0 (0.0728), N95K0 (0.0647), N97K0 (0.0590), and the arms already read (K16, TB, K0, K8 at T 1.0). Not eligible: NRK0 at T 0.8 (0.1945), N95K16 (0.0919) and N97K16 (0.0793, over by 0.0007).
2. **Chosen:** NRK0 at T 1.0, dev SCRIBE 0.192. The next eligible arm is N97K0 at 0.134, so no tie.
3. **Test read:** NRK0 is a T4c arm, and 0.192 is above both bars (0.106 and 0.117). The rule calls for the read. On dev its interval [0.157, 0.225] is clear of the byte comparator's [0.072, 0.135].

**What dev shows.**
- **Dropping the ∞-gram brings the nucleus sampler under the copy limit.** NR copies 0.116 with the full ∞-gram, 0.109 capped at 16 and 0.0728 with none, while SCRIBE stays where it was (0.181, 0.180, 0.192).
- **A wider nucleus copies less but scores lower.** At either cap, copying falls steadily from nucleus 0.9 to 1. SCRIBE drops sharply from 0.9 to 0.95 (0.192 to 0.119 with no ∞-gram), then moves little. The 0.9 cut is what lifts SCRIBE.
- **The nucleus sampler invents fewer words** at T 1.0: 0.070 with nucleus 0.9 against 0.107 for KN alone.

### T4c test read (2026-10-29, read once)

Command: `arm.sh test NRK0 --inf-cap 0 --nucleus 0.9 --no-repeat --temps 1.0` (`cyphalm_tok_gen gen … --prompts-file bench/eval/gen_prompts_test.tsv --inf-cap 0 --nucleus 0.9 --no-repeat --temps 1.0`, then the ladder and the summary). 276 s at load about 7. The byte comparator is T4's B0 test run, reused.

| Arm | SCRIBE v2 [95%] | Surface S [95%] | D_I | Kept / mean bytes | Copy share | Invented words |
|---|---|---|---|---|---|---|
| **NRK0, T 1.0** | **0.187 [0.155, 0.217]** | 0.809 [0.800, 0.818] | 0.00 | 213 / 280 | **0.0649** | 0.0687 |
| B0 byte, T 0.6 (T4's run) | 0.106 [0.078, 0.135] | 0.901 [0.893, 0.909] | 0.00 | 206 / 334 | 0.0561 | 0.0243 |
| B0 byte, T 1.0 (same run, for reference) | 0.046 [0.027, 0.065] | 0.797 | 0.00 | 204 / 290 | 0.0227 | – |
| T4b's K16, T 1.0 (for reference) | 0.115 [0.091, 0.143] | 0.765 | 0.00 | 198 / 275 | 0.0728 | 0.1017 |

**Continuation bits per byte:** 2.0263 [1.9921, 2.0620], `.cont.tsv` byte-identical to T4b's K16 test run; minus B0, paired: −0.1033 [−0.1142, −0.0927].

**The gate.** SCRIBE 0.187 against 0.106: the intervals are disjoint (0.155 above 0.135). **Met.** Copy share 0.0649 against the limit 0.0761: **met.**

**Decision: the gate is met.** By the registration, NRK0 (the token sampler with the nucleus 0.9 and repeat ban over Kneser–Ney alone, T 1.0) is a candidate for `sample`'s default. The switch is not made here: it is a later integration step, and it is not small. The token sampler is its own tool (`cyphalm_tok_gen`, peak about 380 MB per run), so making it `sample`'s default means bringing the token model and the bridge into `cyphalm_trie sample` (or `sample` calling the token tool), with its own parity and memory checks. No default changed.

**What it is and is not.** The text is still word salad by eye (samples below). SCRIBE rates it above the byte sampler's best and above the token sampler's earlier arms, but far below sentences shuffled across documents (0.344). The gain over T4 and T4b comes from the nucleus cut, with copying held down by dropping the ∞-gram from sampling. The human continuations are still scored by the full token model, 0.10 bits per byte better than the byte model.

**Samples** (dev, T 1.0, cut to 150 bytes):
- " Gra]]'' (c/ Indust. There is a more conservative members had worked as the [[Indo-European [[zinc]], [[game. Manycher]] and others. The accord of Sund…"
- " These giant succeeded to have a labor government.org/index.shtml Islam people her own. [[storyID=25, at the court of Pauline, [[super) has neverand t…"
- " When Karpov over the discontinuela, ammonia or '' in the media]], [[Hungarian [[National emphasizes the city that forms of an exchanged can, it has st…"

Test, T 1.0:
- " On the six schools of thought maintains of varying completeness theoremi. [[beta (province)|Cabinda]] and middle as immor, also ;dism database, most…"
- " 1943 | authorlink = Nice is elected a god or, in some, the Government of Alberta website] * [http://www.flight | comment= }} {{Elementbox_isotopes_e…"

### T4c predictions, scored

| # | Prediction | Result |
|---|---|---|
| U1 | NRK0 at T 1.0 copies less than NRK16 (0.109) | **Hit:** 0.0728 |
| U2 | NRK0 at T 1.0 is eligible (copy at most 0.0786) | **Hit:** 0.0728 |
| U3 | NRK0 at T 0.8 is not eligible | **Hit:** 0.1945 |
| U4 | At a fixed cap and T 1.0, a wider nucleus copies less | **Hit:** cap 0: 0.0728 / 0.0647 / 0.0590 / 0.0472; cap 16: 0.109 / 0.0919 / 0.0793 / 0.0749 (nucleus 0.9 / 0.95 / 0.97 / 1) |
| U5 | At a fixed cap and T 1.0, a wider nucleus scores lower | **Miss:** with no ∞-gram, 0.95 → 0.97 rises 0.119 → 0.134 (+0.015). The fall from 0.9 to 0.95 is large (0.192 → 0.119); above 0.95 SCRIBE moves within its intervals |
| U6 | The best eligible T4c arm's dev SCRIBE is in [0.12, 0.17] | **Miss:** 0.192 |
| U7 | Every arm's continuation bits per byte is 2.0260, `.cont.tsv` identical to T4b's | **Hit:** byte-identical on dev and on test |
| U8 | Every T4c arm at T 1.0 invents fewer words than TB (0.104) | **Hit:** 0.070–0.087 |
| U9 | A test read does not meet the gate | **Miss:** the gate is met |

6 hits, 3 misses.

### T4c status

**Done (2026-10-29). Gate met on test; no default changed.** The token sampler with the byte sampler's nucleus 0.9 and repeat ban, over Kneser–Ney alone (`--inf-cap 0 --nucleus 0.9 --no-repeat`, T 1.0), scores SCRIBE v2 0.187 [0.155, 0.217] on GEN-A's test split against the byte default's 0.106 [0.078, 0.135], copying 0.0649 of its bytes (limit 0.0761). It is the first generation change on this benchmark whose interval clears the byte default's as registered (T4's B0; see the caveat below).

**Next (integration, not run here):** bring the token sampler into `sample` as an option and then a default, with `qa` and `chat` untouched (answers stay the byte model's), parity of `sample`'s other paths, and a memory read (about 380 MB per token run today, against `sample`'s 70 MB). Per PLAN ledger 74, defaults switch on T4; this read is T4's gate met by the T4c sampler. Odin may veto.

**A caveat found at the end (2026-10-29).** While T4c ran, main changed `sample`'s byte default: BKN2 (main c756ee3, PLAN ledger 89) adds byte Kneser–Ney. T4c's registered comparator is T4's B0, the byte default as it was. The new byte default scores 0.157 on GEN-A's dev split at T 0.6 (pooled 0.125 [0.107, 0.143]; BYTE_KN.md), against B0's 0.101. NRK0's dev 0.192 [0.157, 0.225] is not clearly above it. So T4c's pass is against the old byte default only. Before any switch, the token sampler must be compared with the new byte default under its own registration (dev first; the new default has no test read on this benchmark yet).

**Cost.** About 380 MB peak resident per arm. 583–608 s per one-temperature dev arm and 794 s for two temperatures at load 7–10; 276 s for the one-temperature test arm.

## T4d: the token sampler against the byte default with Kneser–Ney (registered 2026-10-29, before the test read)

**Odin Loch** · 2026-10-29 · decided by measurement under PLAN ledger 55, open to Odin's veto

**Question.** T4c's token sampler (NRK0: `--inf-cap 0 --nucleus 0.9 --no-repeat`, BPE-W 4,096, T 1.0) met T4's gate against T4's byte default B0. During that read BKN2 (main 8e06368, PLAN ledger 89) made byte Kneser–Ney part of `sample`'s default. Does the token sampler still beat the byte default as it is now?

**The two arms.** Nothing new is sampled on the token side. Only the byte default is read on test.

| Side | What | Dev | Test |
|---|---|---|---|
| Token | NRK0 at T 1.0, unchanged (`cyphalm_tok_gen` from main 7f97541, md5 `22afb787…`, the same binary as T4c's) | T4c's run, reused | T4c's run, reused (read once in T4c, not read again) |
| Byte | `cyphalm_trie sample` with no flags: main's default (byte Kneser–Ney 6:4:2:1, way b, then similar-context and copy). Binary `cyphalm_trie_bkn2d` (md5 `11df0ff4…`), the one BKN2's dev read used; its source is main's `cyphalm_trie.cpp` | BKN2's GEN-A dev read (`/root/work/bkn/runs/bkn2/gen/BKN.*`), reused | **Read once here, label BK0**: `scripts/gen_eval.sh test BK0 OUT`. This is also the new byte default's first read of GEN-A's test split |

**Checks before reuse** (they decide nothing; dev numbers already exist):
1. Main's `cyphalm_tok_gen` reproduces NRK0's first 10 prompts of each split byte for byte (`--limit 10`: the gen file's lines and the continuation rows).
2. Every arm is scored with GEN-A's SCRIBE binary (`scribe_gena`), as in T4–T4c. BKN2's dev read was scored with a SCRIBE built in BKN's tree. Both binaries score `BKN.gen.txt` (dev); if any table number differs, the dev file is re-scored with `scribe_gena` and that score is used below.

These two checks were started a few minutes before this registration was committed.

**No byte-side variant.** `sample`'s temperatures are fixed in the code (0.6, 0.8, 1.0). A lower temperature, where the byte default might score higher, needs a code change, so it is not an arm here. It is a limit of this comparison.

**The dev choice** (T4's rules, with the new comparator):
1. **Byte comparator:** the byte default at its best dev temperature among 0.6, 0.8 and 1.0. From BKN2's read: T 0.6, SCRIBE 0.157 [0.122, 0.196], copy share 0.0691.
2. **Copy limit for a token arm:** the comparator's dev copy share + 0.02 = **0.0891**.
3. **Token arm:** the best eligible token arm on dev among every arm read in T4, T4b and T4c, under the new limit. Under 0.0891, NRK0 at T 1.0 (0.192, copy 0.0728) is still the best; the next are N97K0 (0.134), N97K16 (0.130, copy 0.0793), N95K0 (0.119). Not eligible: N95K16 (0.0919), NRK16 (0.109), NR (0.116), NRK0 at T 0.8 (0.1945).
4. **Test read** only if NRK0's dev point (0.192) is above the comparator's dev point plus T4b's tie band (0.157 + 0.005 = 0.162). It is, so BK0 is read on test. If check 2 changes the comparator's dev score, steps 1–4 are redone with the new number before the test read.

**The test read (once).** BK0: the byte default, all three temperatures, on `gen_prompts_test.tsv`. The comparator on test is BK0 at T 0.6, the temperature chosen on dev (not re-chosen on test). Logged in GEN_TRACK's Reads table.

**Gate** (T4's): NRK0's test SCRIBE (0.187 [0.155, 0.217]) is above BK0's at T 0.6 with disjoint 95% intervals, **and** NRK0's test copy share (0.0649) is at most BK0's at T 0.6 + 0.02.

**If the gate holds**, the switch is not made here. The report writes down what bringing the token sampler into `sample` needs (an option, then a default): parity of `sample`'s other paths, memory (about 380 MB per token run against `sample`'s 70 MB; 218.7 MB on GEN-A's prompts), cost, and the rule-10 check (CyphaLM writes every byte). **If it fails**, the report records why and what next.

**Reported together:** SCRIBE v2 [95%] with surface S [95%], D_I, kept samples and mean bytes, per temperature for BK0; continuation bits per byte with the paired difference, token through the bridge against byte with Kneser–Ney, on dev and on test (`gen_eval_summary.py --pair`); copy share; invented-word rate (`cyphalm_tok_gen words`) for BK0 and BKN's dev file; five samples each; BK0's peak memory and wall time.

**Commands.**

```
B=/root/work/toke/bin/cyphalm_trie_bkn2d; SC=/root/work/toke/bin/scribe_gena; O=/root/work/toke/runs/test
CYPHALM_TRIE=$B SCRIBE=$SC scripts/gen_eval.sh test BK0 $O
/root/work/toke/build/cyphalm_tok_gen words /root/cdata/enwik8 $O/BK0.gen.txt
python3 -I scripts/gen_eval_summary.py --pair /root/work/toke/runs/pair BK0 NRK0   # .cont.tsv files copied in
```

**Predictions** (registered before the test read):

| # | Prediction | Falsified if |
|---|---|---|
| V1 | BK0 at T 0.6 scores within ±0.04 of its dev 0.157 on test | Outside [0.117, 0.197] |
| V2 | The gate is not met: NRK0's test interval overlaps BK0's at T 0.6. About 70% | The intervals are disjoint with NRK0 above |
| V3 | BK0's test copy share at T 0.6 is in [0.054, 0.084], so the copy part of the gate holds | Outside, or the copy part fails |
| V4 | On test, the paired continuation difference, token minus byte with Kneser–Ney, lies in [−0.01, +0.03] bits per byte (dev: 2.0260 against 2.0145, +0.0115) | Outside |
| V5 | BK0's pooled test SCRIBE is in [0.10, 0.15] (dev 0.125) | Outside |
| V6 | BK0 at T 0.6 invents fewer words than NRK0 at T 1.0 (0.0687) | BK0 at T 0.6 invents 0.0687 or more |
| V7 | BK0 at T 0.6 has a higher surface S than NRK0, intervals disjoint (dev 0.927 against 0.811) | Not disjoint, or lower |
| V8 | The two SCRIBE binaries score `BKN.gen.txt` identically | Any table number differs |
| V9 | The 10-prompt reproduction is byte-identical on both splits | Any byte differs |

### T4d checks (2026-10-29)

- **Reproduction (V9).** Main's `cyphalm_tok_gen` (md5 `22afb787…`, the same as T4c's binary) reproduces NRK0's first 10 prompts byte for byte: test, 21 gen lines and 11 continuation rows; dev, 31 gen lines and 11 continuation rows. The dev check must use T4c's `--temps 0.8,1.0`: a sample's seed depends on its temperature's place in that list. My first dev check used `--temps 1.0` alone and differed for that reason only.
- **SCRIBE binaries (V8).** `scribe_gena` and the SCRIBE built in BKN's tree score `BKN.gen.txt` identically: every table line matches, and both match BKN2's own ladder file. So the comparator's dev score stays 0.157, and steps 1–4 of the dev choice stand as registered.

### T4d test read (2026-10-29, read once)

Command: `CYPHALM_TRIE=cyphalm_trie_bkn2d SCRIBE=scribe_gena scripts/gen_eval.sh test BK0 /root/work/toke/runs/test` (the sampler wrapped by `peak.sh`). 813 s at load about 11; peak 232.5 MB. NRK0's numbers are T4c's test read, reused.

| Arm | SCRIBE v2 [95%] | Surface S [95%] | D_I | Kept / mean bytes | Copy share | Invented words |
|---|---|---|---|---|---|---|
| **NRK0, T 1.0** (T4c) | **0.187 [0.155, 0.217]** | 0.809 [0.800, 0.818] | 0.00 | 213 / 280 | 0.0649 | 0.0687 |
| **BK0 byte + Kneser–Ney, T 0.6** | **0.147 [0.115, 0.181]** | 0.926 [0.919, 0.932] | 0.00 | 217 / 362 | 0.0769 | 0.0138 |
| BK0, T 0.8 | 0.125 [0.095, 0.156] | 0.903 [0.896, 0.910] | 0.03 | 194 / 328 | 0.0384 | 0.0254 |
| BK0, T 1.0 | 0.074 [0.051, 0.099] | 0.849 [0.840, 0.858] | 0.00 | 207 / 301 | 0.0222 | 0.0541 |
| BK0, pooled | 0.115 [0.098, 0.133] | 0.893 [0.888, 0.898] | 0.01 | 618 / 331 | 0.0458 | 0.0308 |
| T4's B0 (no Kneser–Ney), T 0.6, for reference | 0.106 [0.078, 0.135] | 0.901 | 0.00 | 206 / 334 | 0.0561 | 0.0243 |

**Continuation bits per byte** (the human continuations; the token model through the bridge, the byte model with its sampling mixture):

| Split | Token (NRK0's model) | Byte + Kneser–Ney | Token minus byte, paired |
|---|---|---|---|
| Dev | 2.0260 | 2.0145 | **+0.0115 [+0.0023, +0.0214]** |
| Test | 2.0263 | 2.0123 | **+0.0140 [+0.0049, +0.0232]** |

**The gate.** SCRIBE 0.187 [0.155, 0.217] against 0.147 [0.115, 0.181]: the intervals overlap (0.155 is below 0.181). **Not met.** The copy part holds: 0.0649 against the limit 0.0969 (BK0's 0.0769 + 0.02).

**Decision: not adopted.** The token sampler stays a measuring tool (`cyphalm_tok_gen`). No default changed. Nothing in `sample` changes.

**What the read shows.**
- **Byte Kneser–Ney closed the gap on prediction and passed it.** In T4 the token model predicted the human continuations 0.10 bits per byte better than the byte default. With byte Kneser–Ney, the byte model is now 0.012–0.014 bits per byte *better* than the token model, on dev and on test, paired intervals clear of zero.
- **SCRIBE does not separate them.** NRK0's point estimate is 0.040 above, but the intervals overlap. Its edge comes from the nucleus cut at T 1.0.
- **The byte text is cleaner on two other measures.** It invents a fifth as many words (0.014 against 0.069), and its surface S is higher, with disjoint intervals (0.926 against 0.809). It also copies a little more (0.077 against 0.065).
- **BKN2's first test read on this benchmark.** The new byte default scores 0.147 [0.115, 0.181] at T 0.6 and 0.115 [0.098, 0.133] pooled, against B0's 0.106 and 0.072. Its dev numbers hold on test (dev 0.157 and 0.125).
- **Memory.** `sample` on GEN-A's 250 prompts with the default peaks at 232.5 MB (BKN2 measured 218.7 MB without Kneser–Ney).

**Samples** (test, T 0.6, the first three prompts, cut to 150 bytes):
- BK0: " The term &quot; and a series and are about the continued use the word &quot; and\n* [[Johan interests and her the belief sound that the algebraic of …"
- BK0: "\n* [[John Arthur C. Clarke]] ([[1813]], [[Belgian poet and in a non-standard analysis]], the [[Germany]] and [[New Zealand and [[Paris]]. The [[Charl…"
- BK0: "\n\nThe of white major considered and [[phosphate between the [[Soviet Union of and the appellate constitution of and special with a different persona…"

NRK0's test samples are in T4c's section above.

### T4d predictions, scored

| # | Prediction | Result |
|---|---|---|
| V1 | BK0 at T 0.6 within ±0.04 of dev 0.157 on test | **Hit:** 0.147 |
| V2 | The gate is not met | **Hit:** intervals overlap |
| V3 | BK0's test copy share at T 0.6 in [0.054, 0.084]; the copy part holds | **Hit:** 0.0769; 0.0649 ≤ 0.0969 |
| V4 | Paired continuation difference, token minus byte, in [−0.01, +0.03] on test | **Hit:** +0.0140 [+0.0049, +0.0232] |
| V5 | BK0's pooled test SCRIBE in [0.10, 0.15] | **Hit:** 0.115 |
| V6 | BK0 at T 0.6 invents fewer words than NRK0 (0.0687) | **Hit:** 0.0138 |
| V7 | BK0 at T 0.6 has the higher surface S, intervals disjoint | **Hit:** 0.926 [0.919, 0.932] against 0.809 [0.800, 0.818] |
| V8 | The two SCRIBE binaries score `BKN.gen.txt` identically | **Hit** |
| V9 | The 10-prompt reproduction is byte-identical | **Hit**, once the dev check used T4c's temperature list (see Checks) |

9 hits, 0 misses.

### T4d status

**Done (2026-10-29). Not adopted.** Against the byte default with Kneser–Ney, the token sampler's SCRIBE edge is inside the noise (test 0.187 against 0.147, overlapping), and the byte model now predicts the human continuations better (−0.014 bits per byte for bytes, paired). T4c's pass held only against the old byte default.

**Next (written down, not run; each would be registered first):**
- **Per-prompt paired SCRIBE.** Both arms answer the same 250 prompts. A paired difference would have more power than two independent intervals. That is a change to the gate, so it needs its own registration and Odin may veto it.
- **The byte sampler at a lower temperature.** `sample` offers only 0.6, 0.8 and 1.0, and SCRIBE rises as the temperature falls (0.074 → 0.125 → 0.147 on test). A `--temps` option is a small code change; the read would be registered first.
- **Tokens with the byte model's other experts.** The byte default has similar-context and copy experts; the token model has neither. T10's mixture is where the two meet.
- **The token sampler's integration is not pursued now.** It costs about 380 MB per run, against `sample`'s 232.5 MB on these prompts, and the byte default already matches it.

**Cost.** BK0: 813 s for three temperatures at load about 11, peak 232.5 MB. The reproduction checks: about 5 minutes. The token side was not re-run.
