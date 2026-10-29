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

### T4 status

Not run. GEN-A's frozen generation benchmark and `scripts/gen_eval.sh` exist in GEN-A's working tree, but GEN-A has not marked them ready, and TOK-A's best-two choice is provisional. The token sampler is ready and checked:
- `cyphalm_tok_gen gen <enwik8> <TOK> <OUT> --prompts-file bench/eval/gen_prompts_{dev,test}.tsv` writes GEN-A's outputs for the token model: PROMPT and GEN lines (kinds `trie_T0.6/0.8/1.0`, so SCRIBE's per-temperature rungs read them), `OUT.cont.tsv` (each human continuation's bits through the bridge) and GEN_EVAL lines (continuation bits per byte, the 32-byte copy share). Then SCRIBE's ladder and GEN-A's summary run unchanged.
- `cyphalm_tok_gen words <enwik8> <files>` gives the invented-word rate.
- A two-prompt smoke run on the dev prompts: 600 bytes a sample in about 20 s a prompt under load; a cache for frequent contexts gives byte-identical output.
- **A risk for T4's copy rule.** At T = 0.6 the token model's samples copied 94% of their bytes in 32-byte spans from the training text (two prompts; T = 0.8: 27%, T = 1.0: 9%). The ∞-gram over tokens follows long matches. T4's dev choice of temperature, and the copy rule (+0.02 at most over the byte model), will decide.
