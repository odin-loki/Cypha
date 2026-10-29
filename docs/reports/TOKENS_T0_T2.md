# Tokens T0–T2: tokenizers learned by counting, and token count models

**Odin Loch** · 2026-10-29 · [TOKENS_PLAN](../TOKENS_PLAN.md) T0–T2 · PLAN ledger 73–74

(The setup, rules and predictions below were written and committed before any run. Results are added under them.)

**Question.** CyphaLM reads 4 bytes back, about 0.6 of a word on its training slice. Does a count model over tokens, learned by counting, predict the same text as well or better, scored in bits per byte? Which tokenizers and sizes are worth taking to generation (T4)?

**Rule 10.** The tokenizers and token models are CyphaLM itself: counts, no neural network. Nothing here answers a question. Nothing in `cyphalm_trie` changes.

## The text (fixed before any run)

- **Training:** enwik8 bytes [0, 7,340,032), the byte model's training slice (`packed_trie.hpp` `Config`).
- **Dev:** the byte model's frozen holdout [7,340,032, 8,388,608).
  - **First half** (to the first token boundary at or after byte 7,864,320): mixture weights are fit here.
  - **Second half** (from that boundary to 8,388,608): every bits-per-byte number in this report.
- **Not read:** the fresh text [8,388,608, 9,437,184) that decides byte-model changes; GEN-A's benchmark [72 MiB, end); SCRIBE's pack, reference and test [16, 72) MiB. T4 does the one test read.
- **Documents** for intervals: the holdout split at `<page>`. Intervals are bootstraps over documents (2,000 resamples) of the bits-per-byte ratio.

## The byte baseline (T0)

The byte model as `cyphalm_trie cls` builds it (wiki only): the trie with its learned mixture, and the trie with InfiniGram mixed by λ per (match length × count) bucket, λ fit on the first half. It must reproduce B4's numbers on the second half: trie 2.278066, trie + InfiniGram 2.072500. It is scored over exactly the same byte span as each token model.

## Tokenizers (T1)

All are deterministic. All keep the 256 single bytes as tokens, so every byte string encodes and decodes exactly.

| Arm | Boundaries | Vocabulary | Encoding |
|---|---|---|---|
| **BPE-W** | GPT-style pre-tokens: a space joins the word after it; runs of letters, digits, other symbols; whitespace runs | BPE merges, most frequent pair first (ties: lowest pair id) | Merges by rank inside each pre-token |
| **UNI** | The same pre-tokens | Unigram LM (Kudo): seeded with frequent substrings (2–16 bytes), EM, pruned by the loss of removing each piece | Viterbi inside each pre-token |
| **SPB** | SpaceByte's rule: a chunk ends after a space-like byte (not a letter, digit or UTF-8 continuation byte) not preceded by another | The most frequent chunks | A chunk in the vocabulary is one token; others by longest match |
| **ENT** | CyphaLM's own byte model: a cut where the next-byte entropy (scored before the byte is seen, as in training) passes a global threshold, set on the training text for a mean segment of 4.5 bytes | The most frequent segments | Longest match, left to right |
| **WRD** | The same pre-tokens as BPE-W | The most frequent whole pre-tokens | A pre-token in the vocabulary is one token; others are bytes |

Sizes: 4,096, 8,192, 16,384 and 32,768 tokens, the 256 bytes included (fewer if time runs short; the report says which).

Reported for each, never used to choose (plan §3): bytes per token, the share of word starts that are token starts, and the share of the vocabulary used on dev.

## Token count models (T2)

- **KN:** a token n-gram model, orders 1–5, interpolated modified Kneser–Ney (Chen and Goodman): continuation counts below the top order, three discounts per order from the count of counts, a uniform base over the vocabulary.
- **INF:** an ∞-gram over token IDs: a suffix array over the training tokens; the longest suffix of the history that occurs; P(next) = its count of each following token over its total.
- **KN + INF:** λ per (match length × count) bucket, fit on the first half by the byte model's grid search.
- **Score:** bits per byte, canonical segmentation: the bits of the token sequence divided by its bytes. For a deterministic tokenizer this is an upper bound on the exact byte-level probability (TOK-B builds the exact bridge).
- Costs: build time and peak RSS.

## Decision rule (fixed before any run)

1. Every arm and size is scored on the second half, KN and KN + INF.
2. **The two best** by KN + INF bits per byte go to T4 (TOKENIZERS_READY names them). Within 0.005 bits per byte, the smaller vocabulary is preferred.
3. Against the byte model: the difference in bits per byte with its bootstrap interval. No default changes here (T12 decides).

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| P-T1 | (plan) On the 8 MiB slice, the best token model's bits per byte is within ±0.05 of the byte model's (trie + InfiniGram, 2.0725) |
| P-T4 | (plan) ENT comes within 0.02 bits per byte of BPE-W at the same vocabulary size |
| P-T5 | (plan) UNI and BPE-W are the top two |
| P-T7 | (plan) A 16k token model on this slice stays under 50 MB resident |
| P-A1 | BPE-W at 8k averages 3.0–4.0 bytes per token on dev (raw wiki XML is markup-heavy) |
| P-A2 | INF lowers KN's bits per byte by at least 0.05 at every size (the byte model gains 0.21 from InfiniGram) |
| P-A3 | On this slice 32k is worse than 8k or 16k for BPE-W (counts too sparse) |
| P-A4 | WRD is the worst arm at every size (rare words fall back to bytes) |
| P-A5 | The best token model (KN + INF) beats the byte trie alone (2.2781) |

## Amendment 1 (registered after the first BPE, UNI and WRD results, before any of the runs it names)

**What was seen.** Every token arm beats the byte model by 0.12–0.17 bits per byte, and smaller vocabularies do better (BPE: 4k 1.905, 32k 1.963).

**The worry.** The token models use interpolated modified Kneser–Ney. The byte trie uses a fixed backoff (alpha 0.4) and a learned six-expert mixture. Part of the gain may be smoothing, not tokens.

**The control (added now).** The same Kneser–Ney and ∞-gram over single bytes (V = 256: BPE with no merges), at orders 5 (the byte trie's 4-byte context), 8 and 12. It is scored exactly like the token arms.

**Also added, report only.** BPE and UNI at 1,024 and 2,048 tokens, to see where the size trend turns. They are not eligible for the two-best rule; the registered sizes stay 4k–32k. T4 may register them.

| # | Prediction |
|---|---|
| P-A6 | Byte KN + ∞-gram at its best order (5, 8 or 12) lands between the byte model (2.0725) and BPE-4096 (1.9048): tokens keep at least 0.05 of their gain over byte KN |
| P-A7 | Byte KN at order 5 beats the byte trie alone (2.2781): part of the gain is smoothing |
| P-A8 | BPE at 2,048 is within 0.01 of BPE at 4,096 |

## Results (2026-10-29)

All numbers are bits per byte on the holdout's second half, [7,864,320, 8,388,608): 524,288 bytes, 74 documents (ENT arms: one byte later, 524,287). Intervals are 95% bootstraps over documents (2,000 resamples). The tool is `cyphalm_tok` at the commit after this report.

### T0: the byte baseline

`cyphalm_tok baseline` reproduces B4's numbers exactly: **trie 2.278066, trie + InfiniGram 2.072500** (104 s, 62 MB peak).

### T1: the tokenizers

| Arm | Time to learn (all sizes) | Peak | Notes |
|---|---:|---:|---|
| BPE-W | 3.8 s | 64 MB | One run to 32,768; smaller sizes are its first merges |
| UNI | 47 s | 97 MB | 200,000 seed pieces, EM, pruned by loss in steps of 25% |
| WRD | 0.7 s | 22 MB | Most frequent whole pre-tokens |
| SPB | 0.9 s | 35 MB | Most frequent SpaceByte chunks |
| ENT | 243 s entropy pass + 3.5 s | 75 MB | Threshold 3.23 bits; mean segment 4.44 bytes; 235,302 distinct segments |

### T2: token count models against the byte model

KN is order 5; INF is the ∞-gram over token IDs (longest match up to 64 tokens). "Aligned" is the share of word starts where a token starts (at the word or at the space before it).

| Arm | Size | Bytes/token | KN | KN + INF | Against the byte model (2.0725) | Aligned | Vocabulary used | KN tables | Peak |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **UNI** | 4,096 | 2.815 | 1.9119 | **1.9039** | −0.1686 [−0.1765, −0.1608] | 1.000 | 0.917 | 232 MB | 310 MB |
| **BPE-W** | 4,096 | 2.848 | 1.9118 | **1.9048** | −0.1677 [−0.1764, −0.1602] | 1.000 | 0.951 | 234 MB | 315 MB |
| UNI | 8,192 | 3.133 | 1.9260 | 1.9189 | −0.1536 [−0.1614, −0.1455] | 1.000 | 0.846 | 231 MB | 309 MB |
| BPE-W | 8,192 | 3.155 | 1.9270 | 1.9206 | −0.1519 [−0.1605, −0.1433] | 1.000 | 0.901 | 233 MB | 313 MB |
| WRD | 16,384 | 2.409 | 1.9573 | 1.9237 | −0.1488 [−0.1597, −0.1386] | 1.000 | 0.624 | 189 MB | 304 MB |
| WRD | 32,768 | 2.709 | 1.9464 | 1.9245 | −0.1480 [−0.1594, −0.1376] | 1.000 | 0.431 | 192 MB | 304 MB |
| WRD | 8,192 | 2.100 | 1.9836 | 1.9342 | −0.1383 [−0.1476, −0.1302] | 1.000 | 0.788 | 185 MB | 304 MB |
| UNI | 16,384 | 3.421 | 1.9438 | 1.9382 | −0.1343 [−0.1424, −0.1250] | 1.000 | 0.702 | 230 MB | 307 MB |
| BPE-W | 16,384 | 3.439 | 1.9468 | 1.9414 | −0.1311 [−0.1388, −0.1225] | 1.000 | 0.778 | 248 MB | 311 MB |
| ENT | 4,096 | 3.026 | 1.9527 | 1.9473 | −0.1252 [−0.1337, −0.1176] | 0.899 | 0.911 | 238 MB | 306 MB |
| WRD | 4,096 | 1.841 | 2.0162 | 1.9491 | −0.1234 [−0.1321, −0.1155] | 1.000 | 0.888 | 172 MB | 304 MB |
| UNI | 32,768 | 3.658 | 1.9598 | 1.9555 | −0.1170 [−0.1247, −0.1078] | 1.000 | 0.541 | 245 MB | 306 MB |
| BPE-W | 32,768 | 3.685 | 1.9682 | 1.9634 | −0.1091 [−0.1167, −0.0999] | 1.000 | 0.572 | 246 MB | 309 MB |
| SPB | 4,096 | 1.784 | 2.0351 | 1.9716 | −0.1009 [−0.1071, −0.0954] | 0.886 | 0.899 | 173 MB | 325 MB |
| SPB | 8,192 | 2.023 | 2.0261 | 1.9784 | −0.0941 [−0.0994, −0.0892] | 0.871 | 0.811 | 187 MB | 327 MB |
| ENT | 8,192 | 3.492 | 1.9941 | 1.9901 | −0.0824 [−0.0913, −0.0745] | 0.875 | 0.844 | 254 MB | 306 MB |
| SPB | 16,384 | 2.348 | 2.0281 | 1.9934 | −0.0791 [−0.0854, −0.0739] | 0.857 | 0.679 | 192 MB | 328 MB |
| SPB | 32,768 | 2.667 | 2.0457 | 2.0188 | −0.0537 [−0.0618, −0.0473] | 0.846 | 0.502 | 194 MB | 327 MB |
| ENT | 16,384 | 4.003 | 2.0495 | 2.0458 | −0.0267 [−0.0348, −0.0186] | 0.844 | 0.742 | 252 MB | 304 MB |
| ENT | 32,768 | 4.480 | 2.1039 | 2.1005 | +0.0280 [+0.0179, +0.0373] | 0.820 | 0.576 | 249 MB | 301 MB |

Each run encodes, builds and scores in 20–40 s on the shared machine.

### Amendment 1: the controls

| Model | KN | KN + INF | Against the byte model | Peak |
|---|---:|---:|---|---:|
| Byte KN, order 5 (V = 256) | 2.1611 | 1.9929 | −0.0796 [−0.0865, −0.0738] | 385 MB |
| Byte KN, order 8 | 1.9168 | 1.8951 | −0.1774 [−0.1860, −0.1707] | 526 MB |
| **Byte KN, order 12** | 1.8880 | **1.8831** | −0.1894 [−0.1999, −0.1806] | 1,303 MB |
| BPE-W 2,048 (report only) | 1.9023 | 1.8941 | −0.1784 [−0.1877, −0.1710] | 315 MB |
| BPE-W 1,024 (report only) | 1.8966 | 1.8856 | −0.1869 [−0.1956, −0.1795] | 311 MB |
| UNI 2,048 (report only) | 1.9052 | 1.8955 | −0.1770 [−0.1854, −0.1694] | 308 MB |
| UNI 1,024 (report only) | 1.9001 | 1.8887 | −0.1838 [−0.1918, −0.1763] | 305 MB |

**Tokens against byte Kneser–Ney, paired by document** (KN + INF, difference in bits per byte; negative favours the first):

| Pair | Difference |
|---|---|
| BPE-W 4,096 − byte KN 8 | +0.0097 [+0.0048, +0.0157] |
| UNI 4,096 − byte KN 8 | +0.0088 [+0.0025, +0.0163] |
| BPE-W 2,048 − byte KN 8 | −0.0010 [−0.0054, +0.0041] |
| BPE-W 1,024 − byte KN 8 | −0.0095 [−0.0136, −0.0049] |
| UNI 1,024 − byte KN 8 | −0.0064 [−0.0110, −0.0009] |
| BPE-W 4,096 − byte KN 12 | +0.0217 [+0.0159, +0.0292] |
| UNI 4,096 − byte KN 12 | +0.0207 [+0.0134, +0.0300] |
| BPE-W 1,024 − byte KN 12 | +0.0025 [−0.0028, +0.0091] |
| Byte KN 8 − byte KN 12 | +0.0120 [+0.0085, +0.0166] |

### What the numbers say

- **Every token arm with Kneser–Ney beats today's byte model** at its best size, by up to 0.17 bits per byte at the registered sizes.
- **Most of that gain is smoothing and context length, not tokens.** The same Kneser–Ney over single bytes at order 8 matches BPE-W at 2,048 and beats both 4,096 arms by about 0.01. At order 12 it beats every token arm except BPE-W at 1,024, which ties with it. On 8 MiB, bits per byte do not show an advantage for tokens once the smoothing is matched.
- **Smaller vocabularies win at this size.** BPE-W and UNI fall steadily from 32,768 to 1,024. Larger vocabularies spread 7 MB of text over more contexts, so counts get sparse. Whether tokens pull ahead with more text (P-T1's second clause) is untested here; that is T11.
- **Boundaries matter.** Arms whose tokens start at word starts (BPE-W, UNI, WRD) beat SpaceByte chunks and entropy segments at every size. Entropy segments get worse fast as the vocabulary grows (32,768 is worse than the byte model).
- **The ∞-gram adds little over a 5-gram of tokens** (−0.005 to −0.011 for BPE-W and UNI). It helps where many words fall back to bytes (WRD, SPB: −0.02 to −0.07), because there the 5-gram's reach is short.
- **A finding for the byte model itself (not changed here).** Byte-level Kneser–Ney at order 8–12 with the ∞-gram is 0.18–0.19 bits per byte better than the current byte trie with InfiniGram on this holdout. That is a candidate change for `packed_trie.hpp`, with its own registered rule and a fresh read; order 12 costs 1.3 GB unpacked.
- **Costs.** The token tables are unpacked (16-byte n-gram entries, 32-byte contexts): 170–255 MB, about 310 MB peak. Packing them like the byte trie (M4) is future work.

### Decision (the registered rule)

**The two best at the registered sizes are UNI 4,096 (1.9039) and BPE-W 4,096 (1.9048).** They are within 0.005 of each other at the same size, so both go to T4.

For T4 to register (not decided here): BPE-W and UNI at 1,024 and 2,048 are as good or better in bits per byte, and byte Kneser–Ney is a needed control in generation too. Bits per byte do not measure what tokens are for in generation: whole-word sampling, no invented words, reach in words.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| P-T1 | Best token model within ±0.05 of the byte model (2.0725) | **Miss:** 1.9039, −0.169. Against byte KN at matched smoothing the gap is within ±0.03, but the prediction named the byte model |
| P-T4 | ENT within 0.02 of BPE-W at the same size | **Miss:** +0.043 at 4,096, +0.070 at 8,192 |
| P-T5 | UNI and BPE-W the top two | **Hit:** UNI 1.9039, BPE-W 1.9048 (WRD 1.9237 next) |
| P-T7 | A 16k token model under 50 MB | **Miss:** 230–250 MB of tables, about 310 MB peak (unpacked) |
| P-A1 | BPE-W 8k at 3.0–4.0 bytes per token | **Hit:** 3.155 |
| P-A2 | INF lowers KN by at least 0.05 at every size | **Miss:** 0.005–0.011 for BPE-W and UNI |
| P-A3 | 32k worse than 8k or 16k for BPE-W | **Hit:** 1.9634 against 1.9206 and 1.9414 |
| P-A4 | WRD worst at every size | **Miss:** WRD is third overall and best at 16k and 32k; SPB and ENT are worse |
| P-A5 | Best token model beats the byte trie alone (2.2781) | **Hit** |
| P-A6 | Byte KN between the byte model and BPE-W 4,096 | **Miss:** byte KN beats BPE-W 4,096 (by 0.010 at order 8, 0.022 at order 12) |
| P-A7 | Byte KN order 5 beats the byte trie alone | **Hit:** 2.1611 against 2.2781 |
| P-A8 | BPE-W 2,048 within 0.01 of 4,096 | **Miss, narrowly:** 0.0107 |

5 hits, 7 misses.

### Checks

- `native_cyphalm_tok_selftest` (new): 180 checks. Pre-tokens cover the text in order. Every tokenizer keeps the 256 bytes and round-trips the training text, 5,000 random bytes and edge strings, and saves and loads exactly. BPE by merge rank equals BPE in training order on every training pre-token. Kneser–Ney equals a brute-force reference at orders 1–3 (to 1e-12), and every distribution sums to 1 (to 1e-9). The ∞-gram equals a scan on 200 random queries.
- **Three planted bugs each fail it:** a merge applied out of order (highest rank first), a dropped byte fallback (byte 255 missing), and a discount above 1 (D1 = 1.2).
- Every eval run checks that the holdout decodes back to its bytes exactly.
- The byte baseline reproduces B4 to the printed digit. No byte-model file changed; the nine CTests of HANDOFF §4.4 pass (listed in the final commit message).

### Limitations

- **Canonical segmentation.** Bits per byte here are the bits of each text's one tokenization, an upper bound on the exact byte-level probability (TOK-B's bridge computes the exact one).
- **Hashed contexts.** Kneser–Ney contexts are 64-bit hashes; a collision over about 10^7 spans has probability about 10^-5.
- **One holdout.** All choices and numbers are on the dev holdout. T4 does the one test read.

## How to reproduce

```bash
cmake --build build --target cyphalm_tok
E=/root/cdata/enwik8
cyphalm_tok baseline $E --out base.bin                     # byte model, 104 s
cyphalm_tok entropy $E ent.f32                             # for ENT, 243 s
for arm in bpe uni wrd spb; do cyphalm_tok train $E --arm $arm --sizes 4096,8192,16384,32768 --out tok/t; done
cyphalm_tok train $E --arm ent --sizes 4096,8192,16384,32768 --entropy ent.f32 --out tok/t
cyphalm_tok train $E --arm bpe --sizes 256,1024,2048 --out tok/t   # amendment 1
cyphalm_tok train $E --arm uni --sizes 1024,2048 --out tok/t
cyphalm_tok eval $E tok/t-bpe-4096.tok --baseline base.bin [--order 5] [--dump d.tsv]
cyphalm_tok eval $E tok/t-bpe-256.tok --baseline base.bin --order 8   # byte Kneser-Ney
```
