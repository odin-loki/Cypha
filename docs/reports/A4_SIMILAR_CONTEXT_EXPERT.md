# A4: the similar-context expert

**Odin Loch** · 2026-10-28 · step A4 (route R2) of [CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §4 · SEMANTICS §5 and S-4

(Draft. The design and the predictions below were written before any run. Results are added under them.)

**Question.** CyphaLM ties a question to a fact only through verbatim bytes ([A2/A3](A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md)). Does an expert that finds stored contexts *near* the current one in vector space, and predicts the next byte from what followed them, help CyphaLM's BPC, its text and its answers?

PLAN rule 10 holds by design. The expert is one more next-byte expert inside CyphaLM's mixture, used on all text and scored on BPC. Each neighbour proposes only the next byte, and only while its bytes agree with what has been written since the last word ended; at the next word the neighbours are found again. Lambdas are fit on held-out wiki bytes, never on questions. Nothing returns a stored string.

---

## The expert (design, fixed before the runs)

`native/include/cypha/cyphalm/similar_context.hpp`, header-only.

- **Keys.** At every word end in the stored text (a word is a run of ASCII letters, digits or apostrophes, as in the class expert), the vector of the previous W words: the SIF-weighted mean (weight a / (a + p(w)), a = 10⁻³, p from the stored text's own word counts) of unit word vectors, centred, projected on principal components (the first one dropped, as SIF drops the common component), unit-normalised and stored as int8. Each key points at the byte after its word.
- **Query.** At the current position: the end of the last completed word (the anchor), and the bytes written since (the tail: separators and the partial word, at most 32). The k nearest keys by cosine, from an inverted-file index over spherical k-means centroids (a few lists probed). One query per word; the result is cached while the tail grows.
- **Proposal.** A neighbour is live while the stored bytes after it equal the tail. Each live neighbour proposes the stored byte after the tail, weighted by exp((cos − cos_max) / T). Their sum, normalised, is the expert's distribution.
- **Mixing.** As InfiniGram and the class expert: P = λ[b] P_sim + (1 − λ[b]) P, one λ per bucket b = (top live cosine, agreement = largest proposal share, live count, inside a word or not), fit on the first half of the holdout.
- **Vectors.** PPMI-SVD (C1), rebuilt with the BPC holdout and the fresh text cut out of enwik8, so no evaluation byte shapes a vector; Cypha's S-1b vectors; and random vectors per word class (k-means classes 256 and 1,024), a class one-hot in disguise.

---

## Predictions (written before any run)

**BPC** (holdout second half, 524,288 bytes, the best variant chosen there):
- **R1.** Over trie + InfiniGram the expert gains 0.005–0.030 BPC, with every block-bootstrap resample below zero (LCI ≥ 0.95).
- **R2.** Over trie + InfiniGram + class (shape:6) the gain is 30–80% of R1's. Both experts predict the next word from word-level context, so they overlap.
- **R3.** On fresh text, read once for the chosen variant, the gain over trie + InfiniGram + class is 0.003–0.025, with the interval excluding 0.

**Variants** (holdout second half, over trie + InfiniGram + class):
- **R4.** W = 8 is best; W = 4 and W = 16 are within 0.005 of it.
- **R5.** k = 32 and k = 128 are within 0.002 of each other; k = 8 is worse by at least 0.002.
- **R6.** PPMI-SVD and Cypha S-1b vectors are within 0.002 of each other (they are level on word similarity); both beat the class vectors by at least 0.002.
- **R7.** SIF weights beat equal weights by at least 0.002.
- **R8.** A finite temperature (0.02–0.1) beats equal neighbour weights.
- **R9.** Keys from wiki + chat against wiki only: within ±0.003 on the wiki holdout.
- **R10.** Vectors built without the evaluation bytes against vectors built on all of enwik8: within 0.002.

**Text.**
- **R11.** Surface S of trie + expert against the trie: +0.00 to +0.10 (it writes real words, like the class expert, but is live on fewer bytes). Trie + class + expert against trie + class: −0.03 to +0.03.

**Answers** (dev 130, chat form, out-of-fold gate at 2% wrong):
- **R12.** Raw right +0 to +3 over the same model without the expert; coverage at 2% wrong +0 to +2 of 130, most likely +0. The neighbours of a question's context come mostly from the subject's article, but the word after a neighbour is rarely the answer's first word, and a neighbour lives through one word only.
- **R13.** With the fact shard, the expert lets some reworded questions reach shard rows: +0 to +2 raw right over the shard model without it.
- **R14.** No added wrong answers at the gate: the Wilson upper bound of the wrong rate does not rise.

**Cost.**
- **R15.** Index build ≤ 120 s and ≤ 300 MB for the wiki keys; a query (one per word) p50 ≤ 0.5 ms, p99 ≤ 3 ms; Q/A stays inside PLAN §10 (p50 ≤ 200 ms a question, the language system ≤ 4 GB).

---

## Setup

| Item | Value |
|---|---|
| Text | enwik8, train `[0, 7340032)`, protocol holdout `[7340032, 8388608)`, fresh text `[8388608, 9437184)` |
| Trie, InfiniGram, class expert | as in B4: the `bpc` model, InfiniGram over the train bytes (cap 8192, 28 lambdas), the class expert with shape classes, tail 6 (12 lambdas). Lambdas on the first holdout half |
| Expert | keys over the train bytes (`cls`) or over InfiniGram's bytes, wiki train + chat (`qa`); 121 lambdas fit on the first holdout half over the distribution it is mixed into |
| Vectors | PPMI-SVD rebuilt by `ppmi_svd` on enwik8 with `[7340032, 9437184)` cut out (WordSim 0.668, SimLex 0.250, MEN 0.626; on all of enwik8 0.672 / 0.255 / 0.637). Also: PPMI-SVD on all of enwik8 (C1), Cypha's S-1b vectors (`sgns_v0p3_e5_wc`, built on all of enwik8), and one random unit vector per word class for the k-means classes 256 and 1,024 |
| Index | 512 lists, 8 probed, 8 k-means iterations on 20,480 sampled keys; recall against an exact scan on 200 holdout positions |
| Intervals | 95% block bootstrap over 4 KiB blocks (2,000 resamples on the holdout, 4,000 on fresh text); prompts for surface S (4,000) |
| Machine | 4 shared cores (three to five other jobs running), `nice -n 5`, g++ 13.3 `-O2 -std=c++20`, clean under `-Wall -Wextra`. Latencies are on this loaded machine |

**Parity.** With `--sim`, `cls` first prints the B4 numbers, and they are equal to the last digit: holdout second half trie 2.278066, trie + InfiniGram 2.072500, + class 1.998603; fresh 2.367826 / 2.166032 / 2.087926; surface S of the trie 0.7612.

**Order of work.** Every setting was chosen on the second half of the protocol holdout (sweeps 1–4 below). The chosen spec was then read once on fresh text, with the lambdas refit on the whole holdout. The Q/A runs use dev only; the frozen test was read once at the end.

---

## Choosing the settings (holdout second half, 524,288 bytes)

trie + InfiniGram 2.072500; trie + InfiniGram + class 1.998603. Each row is the expert added to both. "Live" is the share of bytes on which at least one neighbour proposes a byte.

### Sweep 1: the registered design (W 8, SIF), one setting at a time

| Variant | trie + IG + sim | Gain over trie + IG (95%) | trie + IG + cls + sim | Gain over trie + IG + cls (95%) | Live | Recall | Query p50 / p99 ms |
|---|---:|---|---:|---|---:|---:|---|
| PPMI-SVD, W 8, SIF, k 32 | 2.045978 | −0.0265 [−0.0375, −0.0178] | 1.979071 | −0.0195 [−0.0295, −0.0116] | 0.38 | 0.92 | 0.36 / 0.89 |
| PPMI-SVD, W 8, SIF, k 8 | 2.053914 | −0.0186 [−0.0290, −0.0108] | 1.982520 | −0.0161 [−0.0254, −0.0091] | 0.27 | 0.95 | 0.35 / 4.36 |
| PPMI-SVD, W 8, SIF, k 128 | 2.034537 | −0.0380 [−0.0491, −0.0288] | 1.975431 | −0.0232 [−0.0332, −0.0147] | 0.48 | 0.90 | 0.38 / 0.89 |
| PPMI-SVD, W 8, SIF, k 512 | 2.018693 | −0.0538 [−0.0649, −0.0450] | 1.968641 | −0.0300 [−0.0399, −0.0221] | 0.58 | 0.87 | 0.47 / 1.09 |
| PPMI-SVD, W 8, SIF, k 128, T 0 | 2.035158 | −0.0373 [−0.0470, −0.0293] | 1.977834 | −0.0208 [−0.0292, −0.0135] | 0.48 | 0.90 | 0.38 / 0.90 |
| PPMI-SVD, W 8, SIF, k 128, T 0.02 | 2.042063 | −0.0304 [−0.0415, −0.0215] | 1.977551 | −0.0211 [−0.0310, −0.0131] | 0.48 | 0.90 | 0.38 / 0.88 |
| PPMI-SVD, W 8, SIF, k 128, T 0.1 | 2.033904 | −0.0386 [−0.0493, −0.0297] | 1.976021 | −0.0226 [−0.0322, −0.0145] | 0.48 | 0.90 | 0.39 / 0.99 |
| PPMI-SVD, W 8, SIF, k 128, T 0.2 | 2.034397 | −0.0381 [−0.0483, −0.0296] | 1.976891 | −0.0217 [−0.0308, −0.0140] | 0.48 | 0.90 | 0.39 / 1.03 |
| PPMI-SVD, W 8, SIF, k 128, probe 32 | 2.034064 | −0.0384 [−0.0495, −0.0293] | 1.975113 | −0.0235 [−0.0337, −0.0151] | 0.48 | 0.98 | 1.32 / 3.69 |
| PPMI-SVD, W 4, SIF, k 128 | 2.033251 | −0.0392 [−0.0497, −0.0310] | 1.971450 | −0.0272 [−0.0365, −0.0195] | 0.49 | 0.92 | 0.40 / 1.04 |
| PPMI-SVD, W 16, SIF, k 128 | 2.036791 | −0.0357 [−0.0459, −0.0275] | 1.978192 | −0.0204 [−0.0295, −0.0133] | 0.47 | 0.90 | 0.37 / 0.91 |
| PPMI-SVD, W 8, equal, k 128 | 2.027527 | −0.0450 [−0.0561, −0.0359] | 1.969831 | −0.0288 [−0.0389, −0.0205] | 0.49 | 0.86 | 0.40 / 2.69 |
| PPMI-SVD, W 8, SIF, k 128, dim 32 | 2.036372 | −0.0361 [−0.0467, −0.0273] | 1.977768 | −0.0208 [−0.0303, −0.0129] | 0.48 | 0.91 | 0.26 / 0.67 |
| PPMI-SVD, W 8, SIF, k 128, dim 128 | 2.035333 | −0.0372 [−0.0485, −0.0278] | 1.975307 | −0.0233 [−0.0335, −0.0147] | 0.48 | 0.89 | 0.66 / 2.09 |
| PPMI-SVD, W 8, SIF, k 128, drop 0 | 2.033443 | −0.0391 [−0.0502, −0.0299] | 1.974720 | −0.0239 [−0.0340, −0.0156] | 0.48 | 0.90 | 0.40 / 1.31 |
| PPMI-SVD (all of enwik8), W 8, SIF, k 128 | 2.034308 | −0.0382 [−0.0492, −0.0290] | 1.975445 | −0.0232 [−0.0331, −0.0147] | 0.48 | 0.91 | 0.40 / 1.16 |
| Cypha S-1b, W 8, SIF, k 128 | 2.035897 | −0.0366 [−0.0480, −0.0272] | 1.976032 | −0.0226 [−0.0330, −0.0140] | 0.48 | 0.86 | 0.41 / 0.99 |
| classes 256, W 8, SIF, k 128 | 2.046885 | −0.0256 [−0.0347, −0.0185] | 1.984142 | −0.0145 [−0.0223, −0.0084] | 0.45 | 0.94 | 0.40 / 1.06 |
| classes 1,024, W 8, SIF, k 128 | 2.047232 | −0.0253 [−0.0354, −0.0174] | 1.983893 | −0.0147 [−0.0237, −0.0077] | 0.45 | 0.91 | 0.39 / 1.10 |

- **More neighbours help, all the way.** k = 8, 32, 128, 512 give −0.016, −0.020, −0.023, −0.030 over trie + InfiniGram + class. More neighbours stay live through the word being written (live share 0.27 → 0.58).
- **Shorter windows help.** W = 4 beats W = 8 by 0.004; W = 16 is worse by 0.003.
- **Equal word weights beat SIF**, by 0.006. SIF's weights push the function words out, and those are what predicts the next word.
- **The temperature matters a little.** 0.05 is best; equal neighbour weights (T 0) lose 0.002.
- **Index settings barely matter.** Probing 32 lists (recall 0.98) instead of 8 (recall 0.90) gains 0.0003 for 3.5 times the latency. 32 dimensions lose 0.002; 128 gain nothing; keeping the first component gains 0.0007.
- **Vectors.** PPMI-SVD built without the evaluation bytes and on all of enwik8 are equal (1.975431 / 1.975445). S-1b is level (1.976032). Class vectors are worse by 0.009.

### Sweep 2: short windows, equal word weights, large k

| Variant | trie + IG + sim | Gain over trie + IG (95%) | trie + IG + cls + sim | Gain over trie + IG + cls (95%) | Live | Recall | Query p50 / p99 ms |
|---|---:|---|---:|---|---:|---:|---|
| PPMI-SVD, W 1, equal, k 512 | 2.033854 | −0.0386 [−0.0405, −0.0368] | 1.963879 | −0.0347 [−0.0367, −0.0329] | 0.52 | 0.97 | 0.38 / 2.59 |
| PPMI-SVD, W 2, equal, k 512 | 2.006103 | −0.0664 [−0.0739, −0.0600] | 1.947065 | −0.0515 [−0.0580, −0.0459] | 0.59 | 0.95 | 0.53 / 1.78 |
| PPMI-SVD, W 2, equal, k 2048 | 1.990694 | −0.0818 [−0.0891, −0.0755] | 1.940417 | −0.0582 [−0.0646, −0.0527] | 0.67 | 0.90 | 0.80 / 4.69 |
| PPMI-SVD, W 3, equal, k 512 | 2.004725 | −0.0678 [−0.0782, −0.0592] | 1.951601 | −0.0470 [−0.0564, −0.0393] | 0.60 | 0.94 | 0.52 / 4.72 |
| PPMI-SVD, W 4, equal, k 128 | 2.021384 | −0.0511 [−0.0628, −0.0420] | 1.960925 | −0.0377 [−0.0481, −0.0293] | 0.51 | 0.93 | 0.41 / 4.53 |
| PPMI-SVD, W 4, equal, k 512 | 2.005842 | −0.0667 [−0.0780, −0.0575] | 1.954955 | −0.0436 [−0.0537, −0.0354] | 0.60 | 0.91 | 0.51 / 4.79 |
| PPMI-SVD, W 4, equal, k 2048 | 1.990992 | −0.0815 [−0.0926, −0.0722] | 1.949957 | −0.0486 [−0.0586, −0.0404] | 0.69 | 0.85 | 0.82 / 5.46 |
| PPMI-SVD, W 4, equal, k 512, T 0 | 2.008993 | −0.0635 [−0.0717, −0.0564] | 1.962324 | −0.0363 [−0.0430, −0.0303] | 0.60 | 0.91 | 0.51 / 4.90 |
| PPMI-SVD, W 4, equal, k 512, T 0.02 | 2.027978 | −0.0445 [−0.0554, −0.0358] | 1.966182 | −0.0324 [−0.0418, −0.0246] | 0.60 | 0.91 | 0.54 / 6.00 |
| PPMI-SVD, W 4, equal, k 512, T 0.1 | 2.004899 | −0.0676 [−0.0781, −0.0590] | 1.956959 | −0.0416 [−0.0507, −0.0341] | 0.60 | 0.91 | 0.53 / 5.18 |
| PPMI-SVD, W 4, SIF, k 512 | 2.019957 | −0.0525 [−0.0630, −0.0441] | 1.966455 | −0.0321 [−0.0414, −0.0245] | 0.59 | 0.90 | 0.53 / 4.93 |
| PPMI-SVD, W 8, equal, k 512 | 2.012300 | −0.0602 [−0.0714, −0.0510] | 1.964144 | −0.0345 [−0.0445, −0.0261] | 0.59 | 0.82 | 0.53 / 4.85 |
| PPMI-SVD (all of enwik8), W 4, equal, k 512 | 2.004832 | −0.0677 [−0.0791, −0.0586] | 1.954190 | −0.0444 [−0.0545, −0.0361] | 0.60 | 0.90 | 0.54 / 4.67 |
| Cypha S-1b, W 4, equal, k 512 | 2.005955 | −0.0665 [−0.0781, −0.0571] | 1.954892 | −0.0437 [−0.0541, −0.0351] | 0.60 | 0.88 | 0.54 / 4.92 |
| classes 256, W 4, equal, k 512 | 2.033323 | −0.0392 [−0.0467, −0.0331] | 1.977498 | −0.0211 [−0.0273, −0.0158] | 0.57 | 0.93 | 0.64 / 4.77 |
| classes 1,024, W 4, equal, k 512 | 2.027104 | −0.0454 [−0.0548, −0.0378] | 1.971768 | −0.0268 [−0.0346, −0.0203] | 0.57 | 0.93 | 0.66 / 5.48 |

- **Two words is the best window.** At k = 512 with equal word weights: W = 1, 2, 3, 4, 8 give 1.9639, **1.9471**, 1.9516, 1.9550, 1.9641 over trie + InfiniGram + class.
- **k keeps paying.** W = 2: k 512 → 2,048 gains another 0.0067 (1.9404). W = 4: 128 → 512 → 2,048 gives 1.9609 → 1.9550 → 1.9500.
- **T = 0.05 stays best** (W = 4, k = 512: T 0 1.9623, 0.02 1.9662, 0.05 1.9550, 0.1 1.9570). **SIF loses again**, by 0.0115.
- **Vectors at W = 4.** PPMI-SVD 1.9550, PPMI-SVD on all of enwik8 1.9542, S-1b 1.9549; classes 1.9775 (256) and 1.9718 (1,024).
- **Equal word weights cost latency.** Their context vectors bunch into fewer, larger lists: at W = 8, k = 128 the p99 rises from 0.89 to 2.69 ms and recall falls from 0.90 to 0.86. The p99 column also carries the shared machine's load.

### Sweep 3: around W = 2 (paired against its first row)

The last column is the paired difference from the first row (W 2, equal, k 2,048) over trie + InfiniGram + class: negative is better.

| Variant | trie + IG + sim | Gain over trie + IG (95%) | trie + IG + cls + sim | Gain over trie + IG + cls (95%) | Live | Recall | Query p50 / p99 ms | Against the first row (95%) |
|---|---:|---|---:|---|---:|---:|---|---|
| PPMI-SVD, W 2, equal, k 2048 | 1.990693 | −0.0818 [−0.0891, −0.0755] | 1.940410 | −0.0582 [−0.0646, −0.0527] | 0.67 | 0.90 | 0.88 / 8.80 | |
| PPMI-SVD, W 2, equal, k 8192, probe 16 | 1.975600 | −0.0969 [−0.1042, −0.0909] | 1.934265 | −0.0643 [−0.0708, −0.0588] | 0.74 | 0.89 | 2.24 / 8.21 | −0.0061 [−0.0069, −0.0054] | |
| PPMI-SVD, W 2, equal, k 2048, T 0 | 1.993727 | −0.0788 [−0.0841, −0.0738] | 1.953725 | −0.0449 [−0.0489, −0.0411] | 0.67 | 0.90 | 0.81 / 4.64 | +0.0133 [+0.0108, +0.0161] | |
| PPMI-SVD, W 2, equal, k 2048, T 0.02 | 2.020698 | −0.0518 [−0.0580, −0.0462] | 1.957351 | −0.0413 [−0.0467, −0.0362] | 0.67 | 0.90 | 0.82 / 5.27 | +0.0169 [+0.0157, +0.0182] | |
| PPMI-SVD, W 2, equal, k 2048, T 0.1 | 1.983792 | −0.0887 [−0.0958, −0.0826] | 1.940127 | −0.0585 [−0.0645, −0.0532] | 0.67 | 0.90 | 0.82 / 4.50 | −0.0003 [−0.0011, +0.0006] | |
| PPMI-SVD, W 2, equal, k 2048, min 0.5 | 1.991052 | −0.0814 [−0.0888, −0.0751] | 1.940419 | −0.0582 [−0.0646, −0.0526] | 0.67 | 0.91 | 0.83 / 4.86 | +0.0000 [−0.0001, +0.0002] | |
| PPMI-SVD, W 2, equal, k 2048, drop 0 | 1.988263 | −0.0842 [−0.0916, −0.0780] | 1.939419 | −0.0592 [−0.0655, −0.0538] | 0.67 | 0.92 | 0.86 / 1.96 | −0.0010 [−0.0015, −0.0006] | |
| PPMI-SVD, W 2, equal, k 2048, dim 32 | 1.985331 | −0.0872 [−0.0944, −0.0810] | 1.939253 | −0.0593 [−0.0654, −0.0540] | 0.67 | 0.92 | 0.67 / 1.37 | −0.0012 [−0.0019, −0.0003] | |
| PPMI-SVD, W 2, equal, k 2048, probe 32, lists 2048 | 1.989395 | −0.0831 [−0.0904, −0.0768] | 1.939479 | −0.0591 [−0.0656, −0.0536] | 0.67 | 0.95 | 0.78 / 1.74 | −0.0009 [−0.0012, −0.0007] | |
| PPMI-SVD, W 2, SIF, k 2048 | 2.011276 | −0.0612 [−0.0684, −0.0552] | 1.959209 | −0.0394 [−0.0455, −0.0341] | 0.67 | 0.91 | 0.80 / 2.24 | +0.0188 [+0.0170, +0.0205] | |
| PPMI-SVD (all of enwik8), W 2, equal, k 2048 | 1.989503 | −0.0830 [−0.0904, −0.0767] | 1.939368 | −0.0592 [−0.0657, −0.0538] | 0.67 | 0.91 | 0.83 / 1.95 | −0.0010 [−0.0015, −0.0005] | |
| Cypha S-1b, W 2, equal, k 2048 | 1.996930 | −0.0756 [−0.0831, −0.0692] | 1.944597 | −0.0540 [−0.0603, −0.0484] | 0.67 | 0.88 | 0.86 / 1.77 | +0.0042 [+0.0035, +0.0049] | |
| classes 256, W 2, equal, k 2048 | 2.026617 | −0.0459 [−0.0487, −0.0433] | 1.975278 | −0.0233 [−0.0257, −0.0211] | 0.63 | 0.95 | 0.95 / 5.36 | +0.0349 [+0.0311, +0.0392] | |
| classes 1,024, W 2, equal, k 2048 | 2.021319 | −0.0512 [−0.0557, −0.0472] | 1.969051 | −0.0296 [−0.0332, −0.0262] | 0.64 | 0.90 | 0.84 / 5.65 | +0.0286 [+0.0260, +0.0316] | |

- **k = 8,192 gains 0.0061 more** (16 lists probed), for 2.5 times the query time (p50 2.24 ms) and 465 s to score a megabyte against 245 s.
- **T = 0.05 is the optimum.** Equal neighbour weights lose 0.013, T = 0.02 loses 0.017; T = 0.1 is level.
- **Small wins:** keeping the first component (−0.0010), 32 dimensions (−0.0012, at half the memory), 2,048 lists with 32 probed (−0.0009). A cosine floor of 0.5 changes nothing.
- **SIF loses by 0.019** here.
- **Vectors.** PPMI-SVD on all of enwik8 is better by 0.0010 [0.0005, 0.0015] than the same vectors built without the evaluation bytes: a small effect, inside R10's 0.002. **S-1b is worse by 0.0042** at W = 2, though level at W = 4 and W = 8. Class vectors are worse by 0.029 (1,024) and 0.035 (256).

### Sweep 4: does meaning matter? (random vectors as the control)

Each word gets a random unit vector instead of its PPMI-SVD vector: the same vocabulary, the same pipeline, no meaning. Two contexts are then close only when they share words. Paired against the first row.

| Variant | trie + IG + sim | Gain over trie + IG (95%) | trie + IG + cls + sim | Gain over trie + IG + cls (95%) | Live | Recall | Query p50 / p99 ms | Against the first row (95%) |
|---|---:|---|---:|---|---:|---:|---|---|
| PPMI-SVD, W 2, equal, k 2048 | 1.990693 | −0.0818 [−0.0891, −0.0755] | 1.940410 | −0.0582 [−0.0646, −0.0527] | 0.67 | 0.90 | 0.87 / 5.70 | |
| random per word, W 2, equal, k 2048 | 2.026623 | −0.0459 [−0.0521, −0.0407] | 1.966080 | −0.0325 [−0.0382, −0.0275] | 0.64 | 0.79 | 0.80 / 8.66 | +0.0257 [+0.0239, +0.0274] | |
| PPMI-SVD, W 8, SIF, k 128 | 2.034537 | −0.0380 [−0.0491, −0.0288] | 1.975432 | −0.0232 [−0.0332, −0.0147] | 0.48 | 0.90 | 0.41 / 4.72 | +0.0350 [+0.0302, +0.0395] | |
| random per word, W 8, SIF, k 128 | 2.047675 | −0.0248 [−0.0361, −0.0157] | 1.981722 | −0.0169 [−0.0272, −0.0084] | 0.45 | 0.51 | 0.38 / 4.58 | +0.0413 [+0.0360, +0.0466] | |

- **Meaning carries 44% of the gain at W = 2.** Word identity alone gains 0.0325; PPMI-SVD gains 0.0582. The difference, 0.0257 [0.0239, 0.0274], is what similar (not identical) contexts add.
- **At W = 8 with SIF**, meaning adds 0.0063 (0.0232 against 0.0169).
- **Where the gain falls** (W = 2, k = 2,048, summed bits over all bytes of the second half): 0.0250 at the first byte after a word's end (separators and the next word's first letter) and 0.0332 inside words. The class expert keeps at most 6 bytes since the last word; this expert follows the whole partial word, up to 32 bytes.

### Sweep 5: the small wins together, and the choice

A rule fixed before this sweep ran picks the spec: the lowest trie + InfiniGram + class + sim on the holdout's second half; specs within 0.0005 of it go to the lowest query p50. Paired against the first row.

| Variant | trie + IG + sim | Gain over trie + IG (95%) | trie + IG + cls + sim | Gain over trie + IG + cls (95%) | Live | Recall | Query p50 / p99 ms | Against the first row (95%) |
|---|---:|---|---:|---|---:|---:|---|---|
| PPMI-SVD, W 2, equal, k 2048, drop 0 | 1.988263 | −0.0842 [−0.0916, −0.0780] | 1.939419 | −0.0592 [−0.0655, −0.0538] | 0.67 | 0.92 | 0.86 / 5.21 | |
| PPMI-SVD, W 2, equal, k 2048, dim 32, drop 0 | 1.985172 | −0.0873 [−0.0944, −0.0813] | 1.940079 | −0.0585 [−0.0645, −0.0534] | 0.67 | 0.92 | 0.68 / 4.98 | +0.0007 [−0.0001, +0.0015] | |
| PPMI-SVD, W 2, equal, k 2048, probe 32, dim 32, drop 0, lists 2048 | 1.984401 | −0.0881 [−0.0952, −0.0821] | 1.939666 | −0.0589 [−0.0650, −0.0538] | 0.67 | 0.96 | 0.63 / 4.92 | +0.0002 [−0.0005, +0.0011] | |
| PPMI-SVD, W 2, equal, k 2048, probe 32, drop 0, lists 2048 | 1.987329 | −0.0852 [−0.0925, −0.0789] | 1.938765 | −0.0598 [−0.0662, −0.0544] | 0.67 | 0.96 | 0.80 / 5.31 | −0.0007 [−0.0009, −0.0004] | |

- 32 dimensions lose their edge once the first component is kept (+0.0007, interval spans 0).
- **The choice: PPMI-SVD (no evaluation bytes), W = 2, equal word weights, k = 2,048, T = 0.05, 64 dimensions with the first kept, 2,048 lists with 32 probed.** Spec `vec=ppmi_noeval,sif=0,W=2,k=2048,drop=0,lists=2048,probe=32`. On the holdout's second half: trie + InfiniGram 2.072500 → 1.987329 (−0.0852), trie + InfiniGram + class 1.998603 → 1.938765 (−0.0598 [−0.0662, −0.0544]). Index 85 MB for 1,059,311 keys (a context whose two words are both unknown gets none), built in 65 s.
- **Not chosen: k = 8,192.** It gains about 0.006 more for 2.5 times the query time. It is the dial to turn if BPC matters more than speed.

**How the design moved from the registered one.** The registered expert (W = 8, SIF, k = 32) is a topical expert: it gains 0.020 over trie + InfiniGram + class. The data pulled it to a local one: two words, function words kept, thousands of neighbours. It then works as a smoothed next-word model over contexts of similar meaning, and gains three times as much.
