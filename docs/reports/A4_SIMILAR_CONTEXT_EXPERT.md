# A4: the similar-context expert

**Odin Loch** · 2026-10-28 · step A4 (route R2) of [CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §4 · SEMANTICS §5 and S-4

The design and the predictions were written before any run; the results are under them.

**Question.** CyphaLM ties a question to a fact only through verbatim bytes ([A2/A3](A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md)). Does an expert that finds stored contexts *near* the current one in vector space, and predicts the next byte from what followed them, help CyphaLM's BPC, its text and its answers?

**Answers.**
1. **BPC: yes, by a lot, once the expert is local.** On fresh text, read once, trie + InfiniGram + class goes from 2.0879 to **2.0267** (−0.061 [−0.066, −0.057]), and trie + InfiniGram from 2.1660 to 2.0807 (−0.085). The best expert reads only the last two words, keeps the function words, and pools 2,048 neighbours. Meaning carries 44% of its gain: with random word vectors it gains 0.033 instead of 0.058.
2. **Text: better alone, unchanged on top of the class expert.** Surface S of the trie 0.761 → 0.839; of trie + class 0.905 → 0.910 (the interval spans 0).
3. **Answers: no.** On dev it adds no right answer at a 2% wrong rate (0 → 0). With the fact shard it does harm: look-alike wrong answers pass the gate, or the shard's right answers break. The topical form of the expert finds the right article (*Markov was born in [[Ryazan]]. He ▸ studied…*), but the answer lies before the point it predicts from, and taking it from there would break rule 10. On the frozen test, read once for the recommended configuration, it answers 0 of 200 and the gate keeps nothing (wrong rate 0%, Wilson upper 1.9%).
4. **Cost: inside PLAN §10.** An 85 MB index (wiki keys), built in about a minute; 0.8 ms a query, one query a word; Q/A p50 17 ms a question.
5. **A4b, later the same day: a default.** Gated out of every answer, the expert is now on in `qa`, `sample` and `context8k`, and answering is not worse with it (the trie and trie + InfiniGram answers are byte-identical). It reads a 2.23 MB file that ships (32-d projected PPMI-SVD rows, fresh BPC −0.0002 against the full vectors), and adds 42 MB resident (was 142). `sample` text: SCRIBE v2 0.014 → 0.061, still word salad.
6. **A4c, later the same day: distinct contexts, whole counts.** Each distinct two-word key is now stored once with all its occurrences, and the 8,192 nearest pool them. Fresh text over trie + InfiniGram + class: a further −0.0040 [−0.0048, −0.0032] (2.0275 → 2.0235); surface S and SCRIBE v2 up; answers unchanged. The expert's memory halves (40.0 → 19.3 MB). Pooling whole k-means lists instead (the "counts" form) was 6 MB and very fast but worse, and was removed.

PLAN rule 10 holds by design. The expert is one more next-byte expert inside CyphaLM's mixture, used on all text and scored on BPC. Each neighbour proposes only the next byte, and only while its bytes agree with what has been written since the last word ended; at the next word the neighbours are found again. Lambdas are fit on held-out wiki bytes, never on questions. Nothing returns a stored string.

---

## The expert (design, fixed before the runs)

`native/include/cypha/cyphalm/similar_context.hpp`, header-only.

- **Keys.** At every word end in the stored text (a word is a run of ASCII letters, digits or apostrophes, as in the class expert), the vector of the previous W words: the SIF-weighted mean (weight a / (a + p(w)), a = 10⁻³, p from the stored text's own word counts) of unit word vectors, centred, projected on principal components (the first one dropped, as SIF drops the common component), unit-normalised and stored as int8. Each key points at the byte after its word.
- **Query.** At the current position: the end of the last completed word (the anchor), and the bytes written since (the tail: separators and the partial word, at most 32). The k nearest keys by cosine, from an inverted-file index over spherical k-means centroids (a few lists probed). One query per word; the result is cached while the tail grows.
- **Proposal.** A neighbour is live while the stored bytes after it equal the tail. Each live neighbour proposes the stored byte after the tail, weighted by exp((cos − cos_max) / T). Their sum, normalised, is the expert's distribution.
- **Mixing.** As InfiniGram and the class expert: P = λ[b] P_sim + (1 − λ[b]) P, one λ per bucket b = (top live cosine, agreement = largest proposal share, live count, inside a word or not), fit on the first half of the holdout.
- **Vectors.** PPMI-SVD (C1), rebuilt with the BPC holdout and the fresh text cut out of enwik8, so no evaluation byte shapes a vector; Cypha's S-1b vectors; and random vectors per word class (k-means classes 256 and 1,024), a class one-hot in disguise.
- **Defaults in the header** are this registered design (W 8, SIF, k 32, 64 dimensions with the first dropped, 512 lists, 8 probed). The settings that won on the holdout are given explicitly in the spec (see "The choice" below).

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
| Vectors | PPMI-SVD rebuilt by `ppmi_svd` on enwik8 with `[7340032, 9437184)` cut out (WordSim 0.668, SimLex 0.250, MEN 0.626; on all of enwik8 0.672 / 0.255 / 0.637). Also: PPMI-SVD on all of enwik8 (C1), Cypha's S-1b vectors (`sgns_v0p3_e5_wc`, built on all of enwik8), one random unit vector per word class for the k-means classes 256 and 1,024, and one random unit vector per word (the control in sweep 4) |
| Index | 512 lists, 8 probed (the chosen spec: 2,048 lists, 32 probed); spherical k-means, 8 iterations on 40 sampled keys a list; recall against an exact scan on 200 holdout positions |
| Intervals | 95% block bootstrap over 4 KiB blocks (2,000 resamples on the holdout, 4,000 on fresh text); prompts for surface S (4,000) |
| Machine | 4 shared cores (three to five other jobs running), `nice -n 5`, g++ 13.3 `-O2 -std=c++20`, clean under `-Wall -Wextra`. Latencies are on this loaded machine |

**Parity.** With `--sim`, `cls` first prints the B4 numbers, and they are equal to the last digit: holdout second half trie 2.278066, trie + InfiniGram 2.072500, + class 1.998603; fresh 2.367826 / 2.166032 / 2.087926; surface S of the trie 0.7612. The `qa` runs reproduce the P2 grid's run B (QB below) on every number.

**Order of work.** Every setting was chosen on the second half of the protocol holdout (sweeps 1–5 below). The chosen spec was then read once on fresh text, with the lambdas refit on the whole holdout. The Q/A runs use dev only; the frozen test was read once at the end.

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
| PPMI-SVD, W 2, equal, k 8192, probe 16 | 1.975600 | −0.0969 [−0.1042, −0.0909] | 1.934265 | −0.0643 [−0.0708, −0.0588] | 0.74 | 0.89 | 2.24 / 8.21 | −0.0061 [−0.0069, −0.0054] |
| PPMI-SVD, W 2, equal, k 2048, T 0 | 1.993727 | −0.0788 [−0.0841, −0.0738] | 1.953725 | −0.0449 [−0.0489, −0.0411] | 0.67 | 0.90 | 0.81 / 4.64 | +0.0133 [+0.0108, +0.0161] |
| PPMI-SVD, W 2, equal, k 2048, T 0.02 | 2.020698 | −0.0518 [−0.0580, −0.0462] | 1.957351 | −0.0413 [−0.0467, −0.0362] | 0.67 | 0.90 | 0.82 / 5.27 | +0.0169 [+0.0157, +0.0182] |
| PPMI-SVD, W 2, equal, k 2048, T 0.1 | 1.983792 | −0.0887 [−0.0958, −0.0826] | 1.940127 | −0.0585 [−0.0645, −0.0532] | 0.67 | 0.90 | 0.82 / 4.50 | −0.0003 [−0.0011, +0.0006] |
| PPMI-SVD, W 2, equal, k 2048, min 0.5 | 1.991052 | −0.0814 [−0.0888, −0.0751] | 1.940419 | −0.0582 [−0.0646, −0.0526] | 0.67 | 0.91 | 0.83 / 4.86 | +0.0000 [−0.0001, +0.0002] |
| PPMI-SVD, W 2, equal, k 2048, drop 0 | 1.988263 | −0.0842 [−0.0916, −0.0780] | 1.939419 | −0.0592 [−0.0655, −0.0538] | 0.67 | 0.92 | 0.86 / 1.96 | −0.0010 [−0.0015, −0.0006] |
| PPMI-SVD, W 2, equal, k 2048, dim 32 | 1.985331 | −0.0872 [−0.0944, −0.0810] | 1.939253 | −0.0593 [−0.0654, −0.0540] | 0.67 | 0.92 | 0.67 / 1.37 | −0.0012 [−0.0019, −0.0003] |
| PPMI-SVD, W 2, equal, k 2048, probe 32, lists 2048 | 1.989395 | −0.0831 [−0.0904, −0.0768] | 1.939479 | −0.0591 [−0.0656, −0.0536] | 0.67 | 0.95 | 0.78 / 1.74 | −0.0009 [−0.0012, −0.0007] |
| PPMI-SVD, W 2, SIF, k 2048 | 2.011276 | −0.0612 [−0.0684, −0.0552] | 1.959209 | −0.0394 [−0.0455, −0.0341] | 0.67 | 0.91 | 0.80 / 2.24 | +0.0188 [+0.0170, +0.0205] |
| PPMI-SVD (all of enwik8), W 2, equal, k 2048 | 1.989503 | −0.0830 [−0.0904, −0.0767] | 1.939368 | −0.0592 [−0.0657, −0.0538] | 0.67 | 0.91 | 0.83 / 1.95 | −0.0010 [−0.0015, −0.0005] |
| Cypha S-1b, W 2, equal, k 2048 | 1.996930 | −0.0756 [−0.0831, −0.0692] | 1.944597 | −0.0540 [−0.0603, −0.0484] | 0.67 | 0.88 | 0.86 / 1.77 | +0.0042 [+0.0035, +0.0049] |
| classes 256, W 2, equal, k 2048 | 2.026617 | −0.0459 [−0.0487, −0.0433] | 1.975278 | −0.0233 [−0.0257, −0.0211] | 0.63 | 0.95 | 0.95 / 5.36 | +0.0349 [+0.0311, +0.0392] |
| classes 1,024, W 2, equal, k 2048 | 2.021319 | −0.0512 [−0.0557, −0.0472] | 1.969051 | −0.0296 [−0.0332, −0.0262] | 0.64 | 0.90 | 0.84 / 5.65 | +0.0286 [+0.0260, +0.0316] |

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
| random per word, W 2, equal, k 2048 | 2.026623 | −0.0459 [−0.0521, −0.0407] | 1.966080 | −0.0325 [−0.0382, −0.0275] | 0.64 | 0.79 | 0.80 / 8.66 | +0.0257 [+0.0239, +0.0274] |
| PPMI-SVD, W 8, SIF, k 128 | 2.034537 | −0.0380 [−0.0491, −0.0288] | 1.975432 | −0.0232 [−0.0332, −0.0147] | 0.48 | 0.90 | 0.41 / 4.72 | +0.0350 [+0.0302, +0.0395] |
| random per word, W 8, SIF, k 128 | 2.047675 | −0.0248 [−0.0361, −0.0157] | 1.981722 | −0.0169 [−0.0272, −0.0084] | 0.45 | 0.51 | 0.38 / 4.58 | +0.0413 [+0.0360, +0.0466] |

- **Meaning carries 44% of the gain at W = 2.** Word identity alone gains 0.0325; PPMI-SVD gains 0.0582. The difference, 0.0257 [0.0239, 0.0274], is what similar (not identical) contexts add.
- **At W = 8 with SIF**, meaning adds 0.0063 (0.0232 against 0.0169).
- **Where the gain falls** (W = 2, k = 2,048, summed bits over all bytes of the second half): 0.0250 at the first byte after a word's end (separators and the next word's first letter) and 0.0332 inside words. The class expert keeps at most 6 bytes since the last word; this expert follows the whole partial word, up to 32 bytes.

### Sweep 5: the small wins together, and the choice

A rule fixed before this sweep ran picks the spec: the lowest trie + InfiniGram + class + sim on the holdout's second half; specs within 0.0005 of it go to the lowest query p50. Paired against the first row.

| Variant | trie + IG + sim | Gain over trie + IG (95%) | trie + IG + cls + sim | Gain over trie + IG + cls (95%) | Live | Recall | Query p50 / p99 ms | Against the first row (95%) |
|---|---:|---|---:|---|---:|---:|---|---|
| PPMI-SVD, W 2, equal, k 2048, drop 0 | 1.988263 | −0.0842 [−0.0916, −0.0780] | 1.939419 | −0.0592 [−0.0655, −0.0538] | 0.67 | 0.92 | 0.86 / 5.21 | |
| PPMI-SVD, W 2, equal, k 2048, dim 32, drop 0 | 1.985172 | −0.0873 [−0.0944, −0.0813] | 1.940079 | −0.0585 [−0.0645, −0.0534] | 0.67 | 0.92 | 0.68 / 4.98 | +0.0007 [−0.0001, +0.0015] |
| PPMI-SVD, W 2, equal, k 2048, probe 32, dim 32, drop 0, lists 2048 | 1.984401 | −0.0881 [−0.0952, −0.0821] | 1.939666 | −0.0589 [−0.0650, −0.0538] | 0.67 | 0.96 | 0.63 / 4.92 | +0.0002 [−0.0005, +0.0011] |
| PPMI-SVD, W 2, equal, k 2048, probe 32, drop 0, lists 2048 | 1.987329 | −0.0852 [−0.0925, −0.0789] | 1.938765 | −0.0598 [−0.0662, −0.0544] | 0.67 | 0.96 | 0.80 / 5.31 | −0.0007 [−0.0009, −0.0004] |

- 32 dimensions lose their edge once the first component is kept (+0.0007, interval spans 0).
- **The choice: PPMI-SVD (no evaluation bytes), W = 2, equal word weights, k = 2,048, T = 0.05, 64 dimensions with the first kept, 2,048 lists with 32 probed.** Spec `vec=ppmi_noeval,sif=0,W=2,k=2048,drop=0,lists=2048,probe=32`. On the holdout's second half: trie + InfiniGram 2.072500 → 1.987329 (−0.0852), trie + InfiniGram + class 1.998603 → 1.938765 (−0.0598 [−0.0662, −0.0544]). Index 85 MB for 1,059,311 keys (a context whose two words are both unknown gets none), built in 65 s.
- **Not chosen: k = 8,192.** It gains about 0.006 more for 2.5 times the query time. It is the dial to turn if BPC matters more than speed.

**How the design moved from the registered one.** The registered expert (W = 8, SIF, k = 32) is a topical expert: it gains 0.020 over trie + InfiniGram + class. The data pulled it to a local one: two words, function words kept, thousands of neighbours. It then works as a smoothed next-word model over contexts of similar meaning, and gains three times as much.

---

## Fresh text, read once (1,048,576 bytes), and surface S

The chosen spec, keys over the train bytes, lambdas refit on the whole holdout. Trie 2.367826, trie + InfiniGram 2.166032, + class 2.087926 (B4's numbers, reproduced).

| Model | Fresh BPC | Gain (95% block bootstrap, 4,000 resamples) |
|---|---:|---|
| trie + InfiniGram + sim | 2.080674 | −0.0854 [−0.0905, −0.0807] over trie + InfiniGram |
| trie + InfiniGram + class + sim | **2.026693** | **−0.0612 [−0.0658, −0.0572]** over trie + InfiniGram + class |

Every resample is below zero, so LCI ≥ 0.95 holds for both. On the holdout's second half the same spec gave −0.0852 and −0.0598: the gain carries over to fresh text unchanged. Of the 1,048,576 fresh and 1,048,576 holdout bytes, the expert is live on 67%.

**Surface S** (B4's protocol: 200 holdout prompts, 400 bytes each, T 0.8, nucleus 0.9, no-repeat, the same seed for every model; trie base, no InfiniGram; human text 0.8702):

| Text | Surface S | Change (95% bootstrap over prompts) |
|---|---:|---|
| trie | 0.7612 | |
| trie + sim | 0.8386 | +0.0773 [+0.0658, +0.0879] over the trie |
| trie + class | 0.9052 | |
| trie + class + sim | 0.9095 | +0.0043 [−0.0036, +0.0117] over trie + class |

The expert alone lifts S as the class expert does, a little less. On top of the class expert S does not move: the class expert already brought the samples to real words, which is what S counts.

---

## Answers (dev, 130 questions)

`cyphalm_trie qa` on the 8 MiB protocol (wiki, then two Dolly passes), the class expert (shape:6) in every model, as the P2 grid's run B. "Kept" is the out-of-fold gate at a 2% wrong target (5-fold CV inside dev). Chat form, trie + InfiniGram + class (+ sim); the trie alone and the wiki form answer nothing right in any run. BPC is the chat model's holdout second half.

| Run | Expert keys | Fact shard | BPC trie + cls (+ sim) | BPC trie + IG + cls (+ sim) | Raw right | Kept: right / wrong (coverage) | Wrong rate, Wilson upper | Decode p50 / p99 ms |
|---|---|---|---:|---:|---:|---|---:|---|
| QB (= grid B) | none | no | 2.133872 | 1.995847 | 2 | 0 / 0 (0.0%) | 2.9% | 2.6 / 5.0 |
| QS | wiki + chat | no | 1.985610 | 1.925222 | 1 | 0 / 0 (0.0%) | 2.9% | 31 / 54 |
| QW | wiki only | no | 1.983707 | **1.920433** | 1 | 0 / 0 (0.0%) | 2.9% | 18 / 28 |
| grid B + facts | none | yes | 2.134429 | 1.996357 | 6 | 2 / 1 (2.3%) | 4.2% | |
| QF | wiki + chat | yes | 1.985896 | 1.925547 | 6 | 3 / 3 (4.6%) | 6.6% | 31 / 61 |
| QFW | wiki only | yes | 1.984136 | 1.920864 | 3 | 0 / 0 (0.0%) | 2.9% | 17 / 30 |
| QS8 (topical: W 8, SIF, k 128) | wiki + chat | no | 2.076866 | 1.966519 | 2 | 0 / 2 (1.5%) | 5.4% | 22 / 45 |

**The expert does not answer, and with the fact shard it hurts.** Out of fold, the gate keeps no right answer that it did not keep without the expert.
- **Without the shard** (QS, QW): raw right 2 → 1. The lost one is A0's luck ("In which year did Alexis Carrel win the Nobel Prize?" → *…on 15 April 1912*). Nothing is kept either way.
- **With the shard** (QF): raw right 6 → 6. It loses the Carrel luck and gains "In which Mexican state is Acapulco?" → *Acapulco is in Guerrero.* (the shard's `located_in` row). The gate keeps 3 right / 3 wrong instead of 2 / 1, so it breaks its 2% target: coverage 4.6%, wrong-rate Wilson upper 4.2% → 6.6%.
- **With the shard and wiki-only keys** (QFW): raw right 6 → 3, and the gate keeps nothing (coverage 2.3% → 0%). Wiki neighbours pull the shard's chat sentences toward wiki forms: "When was Alan Kay born?" → *Alan Kay was born in [[Ballston Springfield, Oregon.* (was *…in 1940.*); "When did Alcuin die?" → *Alcuin district council and a discussion…* (was *Alcuin died in 804.*).
- **Why more wrong answers pass with wiki + chat keys.** The two new wrong ones are another person's shard sentence written fluently: "When was Winston Churchill born?" → *Alexander Graham Bell was born in 1847.*; "Where was Napoleon born?" → *Albert Sidney Johnston was born in 1803.* A similar question's answer is exactly what a similar-context expert supplies, and it raises the answer's minimum log-probability, the feature the gate trusts. The gate's threshold fell from −1.42…−1.59 to −2.07.
- **Answers change, content does not.** Only 7–20 of 130 chat answers stay identical, but they stay nonsense: "When was Avicenna born?" → *What the the effects of house the to the United States Supreme Court's case Love…* without the expert, *The day the of the decade.* with it.

**What the neighbours show.** `qa` prints the three nearest stored contexts at each chat prompt (`SIM_NB`).
- **The chosen expert (W = 2)** sees the question's last word and `Assistant`. "Where was Andrey Markov born?" reaches Dolly turns about other people's births (*…\nAssistant: Born ▸ in Detroit, John Baldwin…*, *…▸ and raised in Stevenage…*); "What is the capital of Angola?" reaches *What links Brazil, Uruguay, Mozambique and Angola\nAssistant ▸ : Colonies of Portugal*. It brings the form of an answer, never the entity's fact: two words cannot hold the entity.
- **The topical expert (W = 8, SIF)** finds the right passage. "Where was Andrey Markov born?" → *Markov was born in [[Ryazan]]. He ▸ studied at…* (cosine 0.77); "Who wrote the book Earth in the Balance?" → *Gore started writing ''[[Earth in the Balance]]'', his book ▸ on environmental…* (0.90); "Who wrote the lyrics of the Irish national anthem?" → *The Irish national anthem consists of the ▸ chorus only…* (0.92). **But the answer sits inside the neighbour's context, before the point (▸) whose next bytes it proposes.** A next-byte expert can never deliver it from there.
- **This is where the route stops under rule 10.** Taking "Ryazan" or "Gore" out of a retrieved neighbour's context would be finding the passage and pasting it, which is what REMOVED_NON_CYPHALM_QA records as cheating. So the similar-context expert reaches the right article and cannot use it. I did not build anything past this point.
- **The topical expert drops topic words into its answers.** Its raw right answers: "what is the official language of Angola" → *Colonies of Portuguese and considered as the or a the the…* (its neighbour is the Dolly turn above), and with the trie alone "Who was the Greek hero that is the most central character of Homer's Iliad?" → *The and the the contrary mentions Achilles Achilles…*. These are the subject's words in word salad. The gate keeps neither and keeps two wrong answers instead.

---

## What it costs (PLAN §10: Q/A p50 ≤ 200 ms on one core; the language system ≤ 4 GB)

| Item | Value |
|---|---|
| Code | `similar_context.hpp`, 498 lines, header-only. In `cyphalm_trie`: +352 / −25 lines (`cls --sim`, `--sim-fresh`, `--sim-surface`; `qa --sim`; self-test checks; help text) |
| Index build | 51–65 s for the 1,059,311 wiki keys with 2,048 lists (k-means dominates; 14–23 s with 512 lists); 104 s for the 3,048,121 wiki + chat keys |
| Index in RAM | 85 MB (wiki keys), 220 MB (wiki + chat keys): 64 int8 a key plus its position |
| Query, one per word | p50 0.79 ms, p99 2.8 ms (`cls`); in `qa` with wiki keys p50 0.78–0.80 ms, p99 1.6–1.9 ms; with wiki + chat keys p50 1.6 ms, p99 3.7–4.0 ms. Recall against an exact scan 0.96 |
| Scoring a stream | about 150 s a megabyte, 0.14 ms a byte, on top of trie + InfiniGram + class |
| Q/A decode a question (chat, trie + IG + class) | p50 2.6 → 17 ms (wiki keys) or 31 ms (wiki + chat keys); p99 5 → 28–54 ms. Inside the 200 ms budget |
| Peak RSS | `cls` harness 540 MB (B4's: 257 MB); `qa` 395 → 577 MB (wiki keys) or 888 MB (wiki + chat keys). Inside the 4 GB budget |
| Vectors | PPMI-SVD file 60 MB on disk, 85 s to rebuild without the evaluation bytes (peak 602 MB) |
| Online learning | **None yet.** Keys are batch-built; an inverted file can take appended keys, and the k-means centroids would be kept fixed |

---

## Predictions, scored

| Prediction | Result | Score |
|---|---|---|
| R1. Over trie + InfiniGram, 0.005–0.030 BPC, LCI ≥ 0.95 | −0.0852 on the holdout (chosen spec; −0.097 at k = 8,192); every resample below 0 | **Miss** (far larger); LCI holds |
| R2. Over + class, 30–80% of R1's gain | 70% (0.0598 of 0.0852); 74% for the registered design | **Hit** |
| R3. Fresh text, over + class, 0.003–0.025, interval excluding 0 | −0.0612 [−0.0658, −0.0572] | **Miss** (larger); the interval excludes 0 |
| R4. W = 8 best; W = 4 and 16 within 0.005 | W = 2 best; at the registered setting W = 4 beat W = 8 by 0.004 and W = 16 lost 0.003 | **Miss** |
| R5. k = 32 and 128 within 0.002; k = 8 worse by ≥ 0.002 | 32 → 128 gains 0.0036; k = 8 loses 0.0034 to k = 32; k keeps paying up to 8,192 | **Miss** |
| R6. PPMI-SVD and S-1b within 0.002; both beat class vectors by ≥ 0.002 | W = 8: 0.0006 apart; W = 4: 0.0001; **W = 2: S-1b worse by 0.0042**. Class vectors worse by 0.009–0.035 everywhere | **Hit** at the registered setting, **miss** at the chosen one |
| R7. SIF beats equal weights by ≥ 0.002 | Equal weights win by 0.006 (W 8), 0.012 (W 4), 0.019 (W 2) | **Miss** |
| R8. A finite temperature beats equal neighbour weights | T 0.05 beats T 0 by 0.002 (W 8), 0.007 (W 4), 0.013 (W 2) | **Hit** |
| R9. Wiki + chat keys against wiki only, within ±0.003 on the wiki holdout | Wiki only better by 0.0048 (1.920433 against 1.925222, chat model) | **Miss** |
| R10. Vectors without the evaluation bytes against all of enwik8, within 0.002 | 0.00001 (W 8), 0.0008 (W 4), 0.0010 (W 2) | **Hit** |
| R11. Surface S: trie + expert +0.00 to +0.10; over + class −0.03 to +0.03 | +0.077; +0.004 | **Hit** |
| R12. Raw right +0 to +3; coverage at 2% wrong +0 to +2 | raw right −1 (a lucky answer lost); coverage +0 | **Miss** on raw right, **hit** on coverage |
| R13. With the shard, raw right +0 to +2 | +0 with wiki + chat keys (one gained, one lost); −3 with wiki keys | **Hit** for the keys predicted (wiki + chat) |
| R14. No added wrong answers at the gate | Without the shard: none. With the shard and wiki + chat keys: 1 → 3 wrong, Wilson upper 4.2% → 6.6% | **Miss** |
| R15. Build ≤ 120 s and ≤ 300 MB; query p50 ≤ 0.5 ms, p99 ≤ 3 ms; Q/A inside PLAN §10 | 51–65 s, 85 MB; p50 0.79 ms, p99 1.6–2.8 ms; Q/A p50 17 ms, 577 MB | **Miss** on the query p50 (a cost of k = 2,048); **hit** on the rest |

Five hits, seven misses, three split. The misses run one way: the registered design was a topical expert, and the data wanted a local one. Its BPC gain is three to four times what was predicted. Its effect on answers is worse than predicted.

---

## Frozen test (200 questions, read once)

The configuration recommended in Decisions 2, fixed before this read: the class expert (shape:6) plus this expert with wiki-only keys, no fact shard (`qa --test`, gates chosen on all of dev). The dev lines inside this run equal run QW's.

| Form / model | Raw R / A / W | Gated (dev gate) R / A / W | Coverage | Wrong rate (Wilson upper) |
|---|---|---|---:|---|
| chat / trie + class + sim | 0 / 0 / 200 | 0 / 200 / 0 | 0.0% | 0.0% (1.9%) |
| chat / trie + IG + class + sim | 0 / 0 / 200 | 0 / 200 / 0 | 0.0% | 0.0% (1.9%) |
| wiki / trie + class + sim | 0 / 1 / 199 | 0 / 200 / 0 | 0.0% | 0.0% (1.9%) |
| wiki / trie + IG + class + sim | 0 / 5 / 195 | 0 / 200 / 0 | 0.0% | 0.0% (1.9%) |

- **Nothing is answered, and nothing wrong is let through.** The dev gate (`ig_match0 ≥ 49`) keeps no test answer.
- **A0's two raw right answers on the test** (trie + InfiniGram, no class expert: Alaska, a stored Dolly answer; Apollo 8, luck) are gone. The class expert has never been read on the test, so this loss cannot be pinned on the similar-context expert alone; on dev the expert lost one lucky answer the same way (Carrel).
- The frozen test agrees with dev: this expert does not answer questions.

---

## Decisions

1. **An option: yes.** `cyphalm_trie cls --sim` and `qa --sim` (below), with the expert in `similar_context.hpp`.
2. **For CyphaLM's text: the best measured setting, with wiki-only keys; an option, not a default.** Spec `vec=<PPMI-SVD without the evaluation bytes>,sif=0,W=2,k=2048,drop=0,lists=2048,probe=32,src=wiki`. It passes B4B's four rules for the configuration without the fact shard:
   - fresh BPC −0.0612 [−0.0658, −0.0572] over trie + InfiniGram + class;
   - surface S does not fall (+0.004 [−0.004, +0.012]);
   - Q/A coverage at 2% wrong and the wrong rate's Wilson upper bound unchanged (0 / 0; 2.9%);
   - Q/A p50 17 ms a question, 577 MB peak.

   Wiki-only keys beat wiki + chat keys: 0.005 better on wiki BPC, 85 MB against 220 MB, half the decode time.

   B4B reads rule 3 on the answering configuration, which includes the fact shard, and there rule 3 fails (decision 3). So, like the class and copy experts ([B4B](B4B_LM_DEFAULTS.md)), it stays an option until the expert and the shard can work together.
3. **Not in the answering model while the fact shard (R1) is on.** With the shard it either lets look-alike wrong answers through the gate (wiki + chat keys: 1 → 3 wrong, Wilson upper 4.2% → 6.6%) or breaks the shard's right answers (wiki keys: coverage 2.3% → 0%). B4B's rule 3 fails both ways. Two ways on, for Odin to choose:
   - gate the expert off inside the answer, after `Assistant: ` (a gate, so allowed; not built);
   - or re-choose the answer gate on features the expert cannot inflate, such as A1c's entity count.
4. **Not an answering route.** The topical form finds the right article, but the answer lies inside the neighbour's context, before the point it predicts from. Taking it from there would be pasting from a retrieved passage, which rule 10 forbids, so the design stops here. A4's target (coverage on reworded questions) is not met by this expert. The R1 shards stay the only route that answers.
5. **Vectors.** PPMI-SVD stays the source (S-1b loses 0.004 at W = 2). The rebuild without the evaluation bytes should be the file every BPC experiment uses: the leak is small (0.001) but real.
6. **What to try next.**
   - **Counts instead of neighbours.** k keeps paying (W 2: k 512 → 2,048 → 8,192 gives 1.9471 → 1.9404 → 1.9343). That points to smoothed next-word counts per cluster of two-word contexts (the k-means lists already exist). It would be cheaper than 2,048 neighbours and could learn online.
   - **k = 8,192** where BPC matters more than speed: 0.006 more for 2.5 times the query time.
   - **Online keys.** Append keys as text streams in, with the centroids fixed.
   - **With the copy expert.** Not measured here (copy is not a default); B4B's long-match copy should be read with this expert on.

---

## A4b: the expert as a default (2026-10-28)

Odin, 2026-10-28: "the similar context expert should be used as soon as it is useful." Five steps: a gate that keeps it out of answers; defaults where measurement supports them; a vector file small enough to ship; its RAM; a SCRIBE v2 read. The design, the rules and the predictions below were written before any of these runs.

**The gate.** Inside an answer the expert is off: after the latest `Assistant: ` until the next `User: ` (a chat turn), and in `qa` from the end of the prompt on, which covers the wiki prompt form (`<question> `). Everywhere else, including the question and every BPC read, it stays on. `gate=0` in the spec turns the gate off, for comparison only.

**The shipped vectors.** The keys use 64 principal components of the context means. A word's projection onto them is all a query needs, so the file holds each word's projected row as int8 with one scale per row (`simvec`, 3.8 MB at 64 dimensions, 2.2 MB at 32). A build over the text the projection came from then gives the same keys as the full vectors, up to the rows' int8 rounding. Built from the PPMI-SVD vectors without the evaluation bytes, over wiki train, with the chosen spec.

**Rules (fixed before the runs).**
- **Gate (step 1, B4B rule 3).** In A1d's C1L configuration with the class expert (`shape:6:2:f`, B4's two-order table, as A4 ran it): with the gated expert, raw right and the out-of-fold gate's kept right / wrong are not worse than without the expert, in every model; the holdout lines keep the expert's gain.
- **Vector file (step 3, the lossy rule).** Fresh trie + InfiniGram + class + sim at most +0.002 above the full-precision vectors (2.026693 in A4), and the surface S interval of the change not entirely below 0. Of the files that pass, the smallest ships. If none passes, nothing is committed.
- **Defaults (step 2).** A path gets the expert by default when it improves that path's own reads (BPC with LCI ≥ 0.95, or surface S) and does not worsen its answers.

**Predictions.**

| # | Prediction |
|---|---|
| D1 | With the gate, the trie and trie + InfiniGram answers are byte-identical to the run without the expert (inside an answer the distribution is the same); raw right and the kept right / wrong are equal in those models. The copy models may differ by at most one answer (their lambdas are fit with the expert present). The question's log-probability changes, so a gate on that feature could move |
| D2 | Without the gate (`gate=0`), the C1L shard configuration loses: trie + InfiniGram raw right falls by 1–4, or the out-of-fold gate keeps fewer right or more wrong (as A4's QF and QFW) |
| D3 | The holdout lines are the same with and without the gate, and the expert gains 0.07–0.08 over trie + InfiniGram + class (A4's QW: 0.075) |
| V1 | 64-d int8 file: fresh BPC within +0.0005 of the full vectors; surface S unchanged (interval spans 0) |
| V2 | 32-d int8 file: fresh BPC within +0.002 (A4's holdout: +0.0002 to +0.0007 for 32 dimensions); surface S unchanged. It ships |
| C1 | `context8k`: the expert lowers trie + InfiniGram (cap 8192) on the holdout's second half by 0.06–0.09 |
| C2 | `sample`: trie + sim text scores higher on SCRIBE v2 than trie text, by +0.00 to +0.05 on the mean (word salad stays word salad; trie text scores 0.017 in SCRIBE_V2) |
| M1 | RAM, after the cuts: the default expert adds about 0.04 GB resident in `qa` (32-d keys) and at most 0.1 GB to the peak; before the cuts (A4's code and float vectors) about 0.18 GB to the peak |

### Step 1: the gate

`qa` in A1d's C1L configuration (span candidates, `ent_title`, the low-only token, 20% don't-know turns, D4's fact shard, copy models alongside) with the class expert `shape:6:2:f` (B4's two-order table, as A4 ran it), dev 130, chat form. The expert uses the shipped 32-d file, keys over wiki train.

| Run | Expert | Holdout trie + cls / trie + IG + cls (+ sim) | + copy (trie + IG) | Raw R / A / W: trie | trie + IG | trie + IG + copy | Kept by the out-of-fold gate, trie + IG: right / wrong | Wrong rate, Wilson upper | Decode p50, trie + IG |
|---|---|---|---:|---|---|---|---|---:|---:|
| G0 | none | 2.135181 / 1.996391 | 1.862964 | 0 / 50 / 80 | 6 / 68 / 56 | 3 / 68 / 59 | **2 / 0** (1.5%) | 2.9% | 1.5 ms |
| **G2** | **gated (the default)** | **1.987304 / 1.920925** | **1.799409** | 0 / 50 / 80 | 6 / 68 / 56 | 4 / 68 / 58 | **2 / 0** (1.5%) | 2.9% | 2.0 ms |
| G1 | in answers too (`gate=0`) | 1.987304 / 1.920925 | 1.799409 | 0 / **0** / 130 | **4** / 67 / 59 | **1** / 67 / 62 | **0 / 2** (1.5%) | 5.4% | 17.2 ms |

- **With the gate nothing in answering gets worse.** The trie and trie + InfiniGram answers are byte-identical to G0's in both prompt forms (130 of 130 each), and so are their confidence features; only the question's log-probability moves (the question is not an answer). Raw right, abstentions and the gate's kept right / wrong are the same; the gate picks `min_lp` in all five folds, as without the expert. The copy models keep 107–113 of 130 answers word for word (their lambdas are fit with the expert present); one grade changes, for the better: "When was Alan Kay born?" → *Alan Kay was born in 1940.* (trie + IG + copy, raw right 3 → 4).
- **The holdout keeps the expert's gain**: trie + InfiniGram + class 1.996391 → 1.920925 (−0.0755, LCI P(gain > 0.005) = 1.000 over 256 blocks); with copy, 1.862964 → 1.799409.
- **Without the gate it gets worse, as A4 found.** The trie stops writing its own "I don't know" (50 abstentions → 0); trie + InfiniGram loses two right answers and its gate keeps two wrong ones instead of two right ones.
- **`--no-sim` reproduces the tool as it was**: run G0N (`--no-sim`, the new build) prints G0's lines and rows file (main before the class change, `--cls shape:6`) byte for byte, latencies aside.
- Peak RSS is 284.4 MB in all four runs: the expert's build does not set the `qa` peak in this configuration.

### Step 2: where the expert is a default, and why

| Path | Default? | Evidence |
|---|---|---|
| `qa` | **yes, gated** | Holdout trie + IG + class −0.0755 (LCI 1.000); answers and the gate's kept right / wrong unchanged (step 1); decode p50 1.5 → 2.0 ms a question; peak unchanged in C1L (284.4 MB). In A0 (no class, no copy): holdout trie 2.319989 → 2.106547 (−0.213), trie + IG 2.071574 → 1.968617 (−0.103), LCI 1.000; every printed line equal to `--no-sim` but one risk-coverage curve of the question's log-probability; 447 → 626 s; peak 175.1 → 186.9 MB |
| `sample` | **yes** | Surface S of trie + sim over the trie +0.075 [+0.064, +0.086] (step 3, the 32-d file); SCRIBE v2 0.014 → 0.061 (step 5). The files keep their GEN kinds and gain a `MODEL` line, so `scribe ladder --gen-trie` reads them as before; `--no-sim` writes B1's files byte for byte (sha256 `fcc2b9c4…`, `8592fb7e…`). 76 → 292 s for 1,500 samples; peak 53.6 → 82.7 MB |
| `context8k` | **yes, one line added** | `trie_plus_context_8192_sim_bpc` 1.9686 against trie + InfiniGram 2.0715 (−0.103, holdout second half, chat model). The 221 existing lines, its 50 generated answers included, are byte-identical to M9's output: the answers never see the expert. 645 s; peak 169.3 MB (M9: 165.1–169.0) |
| `cls` | no (an option) | It measures the class expert; `--sim default` adds the default expert's reads when wanted |
| `chat`, `probe` | no | They write only answers, which the gate excludes: the expert would change nothing and cost RAM. Run G1 shows what it does inside answers: the trie stops abstaining and the gate keeps wrong answers |
| `bpc` | no | Its lines are the trie's reference numbers (Odin) |

`--no-sim` turns the expert off in `qa`, `sample` and `context8k`. Without the vector file those subcommands stop with a message (run from the repository root, set `CYPHA_SIMVEC`, or pass `--no-sim`); they never fall back silently.

### Step 3: a vector file that ships

`cls shape:6:2:f` with the chosen spec three times: the full-precision PPMI-SVD vectors (300-d floats, 60 MB, the A4 numbers), and the projected files at 64 and 32 dimensions. Fresh text read for the lossy rule; paired against the full vectors.

| Vectors | File | Index in RAM | Holdout trie + IG + cls + sim (paired change) | Fresh trie + IG + cls + sim (paired change) | Surface S trie + sim (paired change) | Surface S trie + cls + sim (paired change) | Query p50 |
|---|---:|---:|---|---|---|---|---:|
| PPMI-SVD, 300-d float | 60 MB + 0.6 MB | 85.0 MB | 1.938765 | 2.026693 | 0.8386 | 0.9095 | 0.79 ms |
| projected, 64-d int8 | 3.83 MB | 75.6 MB | 1.938865 (+0.0001 [−0.0001, +0.0002]) | 2.026718 (+0.00003 [−0.00008, +0.00013]) | 0.8272 (**−0.0113 [−0.0213, −0.0014]**) | 0.9128 (+0.0032 [−0.0044, +0.0109]) | 0.80 ms |
| **projected, 32-d int8** | **2.23 MB** | **40.0 MB** | 1.939552 (+0.0008 [+0.0000, +0.0017]) | **2.026526 (−0.0002 [−0.0008, +0.0005])** | 0.8365 (−0.0021 [−0.0124, +0.0083]) | 0.9055 (−0.0041 [−0.0116, +0.0037]) | 0.64 ms |

- **The full-precision row reproduces A4 to the last digit** (holdout 1.987329 / 1.938765, fresh 2.080674 / 2.026693, S 0.8386 / 0.9095): the reworked header gives the same expert.
- **The 32-d file passes the lossy rule and ships**: fresh BPC −0.0002 (the rule allows up to +0.002), neither surface interval below 0. It is 2.23 MB (the repository's `.git` is 35 MB) and halves the index.
- **The 64-d file fails the rule** on surface S of trie + sim (−0.011, the interval entirely below 0), though its BPC is level. S is a sampled score: a small change in the distribution changes every sample after its first different byte. The rule was fixed before the run, so the 64-d file does not ship either way.

The file rebuilds byte for byte from enwik8 with the commands in `data/cyphalm/README.md` (PPMI-SVD vectors and file both checked by sha256).

### Step 4: RAM

**The expert alone** (scratch harness `simmem.cpp`: 8 MiB of text read, then the vectors loaded and the expert built over wiki train with the chosen spec; `VmHWM` reset before the load; MB over the 11.6 MB process):

| Version | Resident after the build (`malloc_trim`) | Build peak | Predictions |
|---|---:|---:|---|
| A4's code, 300-d float vectors | +142.2 | +240.9 | (reference) |
| Reworked header, float vectors: the 300-d rows freed after the build, the keys put in list order in place | +85.0 | +106.1 | identical (same checksum over 200,000 reads) |
| The same with the 64-d projected file | +76.0 | +97.1 | lossy (step 3) |
| **The same with the 32-d projected file (the default)** | **+42.1** | **+63.2** | lossy, within the rule (step 3) |

The 42 MB are the index (1,059,311 keys × 32 int8 = 33.9 MB; positions 4.2 MB; centroids and list starts 0.07 MB), the rows (1.8 MB) and the word map (about 2 MB).

**In the subcommands** (peak RSS of the whole run, `--no-sim` → default):

| Subcommand | Peak RSS | Resident the expert adds |
|---|---:|---:|
| `qa` A0 (RAM_PROFILE's configuration) | 175.1 → 186.9 MB | about 42 MB (its index and rows) |
| `qa` C1L + class (step 1) | 284.4 → 284.4 MB | about 42 MB |
| `sample` | 53.6 → 82.7 MB | about 42 MB |
| `context8k` | 165.1–169.0 (M9) → 169.3 MB | about 42 MB |
| `chat`, `probe`, `bpc` | unchanged | 0 (not built) |

Where training or the class and index builds set the peak, the expert's build does not raise it; where the trie's training sets a low peak (`sample`), the expert's build becomes the peak.

**Further cuts, not taken.** Many keys repeat (the same two words give the same key): storing each distinct key once with its positions would save up to about 18 MB, but it changes which tied neighbours fill the k slots, so it is lossy and needs its own read. A sorted word list in place of the hash map would save about 2 MB.

### Step 5: SCRIBE v2 on `sample` text

`sample`'s 750 GEN lines (250 holdout prompts, T 0.6 / 0.8 / 1.0, 600 bytes), with and without the expert, through `scribe ladder --profile v2` on SCRIBE's development ladder A (test [56, 64) MiB, seed 20261027, pack [16, 48), reference [48, 56), PPMI vectors on the pack; the frozen v2 profile). Reported only; not a decision input.

| Text | Kept (≥ 100 B) | SCRIBE v2 (95%) | Surface S | D_L lexical | D_S syntax | D_C coherence | Samples failing syntax |
|---|---:|---|---:|---:|---:|---:|---:|
| human (test) | 500 | 0.923 [0.910, 0.935] | 0.912 | 0.02 | 0.00 | 0.38 | 0% |
| trie | 623 | 0.014 [0.009, 0.021] | 0.734 | 2.29 | 23.64 | 5.20 | 99% |
| **trie + sim** | 605 | **0.061 [0.049, 0.074]** | 0.801 | 0.94 | 18.75 | 4.72 | 94% |
| (sentences shuffled) | 500 | 0.344 | 0.905 | | | | |
| (words shuffled) | 500 | 0.003 | 0.923 | | | | |

By temperature: T 0.6 0.013 → 0.072, T 0.8 0.020 → 0.074, T 1.0 0.010 → 0.033. The expert mostly removes non-words (samples with a lexical penalty 44% → 21%) and some syntax failures. The text is still word salad: it stays far below shuffled sentences, and above shuffled words.

### Predictions, scored (A4b)

| Prediction | Result | Score |
|---|---|---|
| D1. With the gate, trie and trie + IG answers byte-identical; equal raw right and kept; copy models differ by at most one answer | 130 / 130 identical in both forms; equal tallies; copy models: one grade changes (wrong → right), 17–23 answer texts change | **Hit** (the copy texts change more than "one answer", their grades by one) |
| D2. Without the gate, trie + IG loses 1–4 raw right, or the gate keeps fewer right / more wrong | 6 → 4 raw right; kept 2 / 0 → 0 / 2; the trie's abstentions 50 → 0 | **Hit** |
| D3. Holdout lines equal with and without the gate; gain 0.07–0.08 over trie + IG + class | equal; −0.0755 | **Hit** |
| V1. 64-d file: fresh within +0.0005; S unchanged | +0.00003; S of trie + sim −0.011 [−0.021, −0.001] | **Hit** on BPC, **miss** on S |
| V2. 32-d file: fresh within +0.002; S unchanged; it ships | −0.0002; both S intervals span 0; ships | **Hit** |
| C1. `context8k`: −0.06 to −0.09 | −0.103 | **Miss** (larger) |
| C2. SCRIBE v2: +0.00 to +0.05 on the mean | +0.047 pooled (+0.023 to +0.059 by temperature) | **Hit** |
| M1. After the cuts about 0.04 GB resident and at most 0.1 GB peak; before, about 0.18 GB peak | +42 MB resident, +63 MB build peak (no rise in `qa`'s peak); before: +142 MB resident, +241 MB peak | **Hit** after; **miss** before (0.24 GB, larger) |

### Decisions (A4b)

1. **The gate is part of the expert.** It is on wherever the expert runs; `gate=0` exists only to show what it prevents (run G1). It settles A4's decision 3: C1L carries D4's fact shard, and with the gate the shard's answers are untouched (G2 against G0), where A4's ungated runs lost them or let wrong ones through.
2. **A default in `qa` (gated), `sample` and `context8k`; not in `cls`, `chat`, `probe` or `bpc`** (step 2's table). Every default passes its own read: BPC with LCI 1.000 (`qa`, `context8k`), surface S and SCRIBE v2 (`sample`); none worsens an answer.
3. **The shipped vectors are `data/cyphalm/simvec_ppmi32.bin`** (2.23 MB): the 32-d projected file, the only one that passes the lossy rule. It rebuilds byte for byte from enwik8.
4. **RAM: +42 MB resident** where the expert runs, against +142 MB with A4's code; the build peak +63 MB against +241 MB. It does not move `qa`'s peak with the class expert (284.4 MB) and adds 11.8 MB to A0's (175.1 → 186.9 MB).
5. **For Odin.**
   - `sample`'s GEN kinds still read `trie_T0.8` though the text is now trie + sim (a `MODEL` line says so). This keeps `scribe ladder --gen-trie` working; renaming the kinds would need a change in `scribe.cpp`.
   - A CMake build should set `CYPHA_SIMVEC_DEFAULT` to the file's absolute path, so the tools find it from any directory.
6. **Next.** Distinct keys stored once (up to −18 MB, lossy: it needs its own read); next-word counts per cluster of two-word contexts in place of 2,048 neighbours (A4 decision 6); keys appended online.

---

## A4c: counts in place of neighbours (2026-10-28)

HANDOFF §6.2 and A4b's "Next": "k keeps paying" (sweeps 2 and 3), so pool the stored contexts' continuations instead of keeping k live neighbours. The design, the rule and the predictions below were written before any A4c run.

**Why it might pay.** The default takes the k = 2,048 nearest stored *occurrences*. A two-word context like *of the* occurs thousands of times, and its copies all have cosine 1, so they fill the 2,048 slots by themselves: the expert then sees a sample of that exact context's continuations and nothing similar. Rare contexts reach similar ones. Two ways to pool all occurrences instead:

- **`mode=dedupe`.** Each distinct key is stored once (a two-word context with unknown words dropped from its mean), with all its occurrences. The k nearest *distinct* contexts are found as now, and each pools all its occurrences, weighted by exp((cos − cos_max) / T) as a neighbour is now. `norm=1` lets each distinct context vote once (its occurrences' shares) instead of with its count.
- **`mode=counts`.** No neighbours at all: the query's `probe` nearest k-means lists (the inverted file's lists, `lists` of them), each pooling all its occurrences, weighted by exp((cos − cos_max) / T) on the centroid cosine. The keys are dropped after the build; only positions, centroids and the word rows stay.

In both modes, each group's occurrences are sorted by the bytes that follow them, so the occurrences that agree with what has been written since the word ended (the tail) are one contiguous range: a binary search narrows it as the tail grows, and the next bytes are counted by binary search too. An occurrence still lives only through the word being written; nothing returns a stored string (PLAN rule 10). The default mode (`knn`) is unchanged.

**Rules (fixed before the runs).**
1. **Choice, on the holdout's second half** (`cls shape`, the class expert's default table, keys over wiki train, the shipped 32-d vectors; lambdas on the first half). Over trie + InfiniGram + class: the lowest BPC among the candidates below; candidates within 0.0005 of it go to the lowest scoring time per byte.
2. **Default, on fresh text read once** for the chosen candidate, paired against the current default (`default`) in the same run (lambdas on the whole holdout). It replaces the default by either route:
   - **better:** fresh trie + InfiniGram + class + sim lower, with P(paired change < 0) ≥ 0.95 over 4 KiB blocks (LCI); surface S of the change not entirely below 0; SCRIBE v2 of `sample` text not below the current default's 95% interval; inside PLAN §10 (Q/A p50 ≤ 200 ms, the language system ≤ 4 GB);
   - **cheaper (the lossy rule):** fresh within +0.002 of the current default; surface S of the change not entirely below 0; SCRIBE v2 as above; and at least 25% less resident memory or scoring time per byte.
3. **Answers (B4B rule 3):** `qa` in B4c's rule-3 configuration (C1L without `--copy` and `--cls`): the answer rows byte-identical with the new spec and with `default` (the expert is off inside answers, so this must hold by construction).
4. **Code debt.** A mode that is neither the default nor needed by a report's numbers is removed, with its result recorded here.

**Candidates** (all with the default's W = 2, equal word weights, 2,048 lists, the shipped vectors):
- `default` (the reference, the first spec);
- dedupe: k 512, 2,048, 8,192 at T 0.05; T 0.02 and 0.1 at k 2,048; `norm=1` at k 2,048, T 0.05;
- counts: probe 1, 4, 16, 64 at T 0.05; T 0, 0.02, 0.1 at probe 16; and 512 lists with probe 4 and 16 at T 0.05.

**Predictions.**

| # | Prediction |
|---|---|
| N1 | Distinct keys are 35–55% of the 1,059,311 occurrences |
| N2 | dedupe's best candidate is better than `default` on the holdout by 0.000–0.006 over trie + IG + class: frequent contexts get their whole count, rare ones are unchanged |
| N3 | dedupe's best T is lower than 0.05 or `norm=1` wins: with whole counts, frequent similar contexts swamp the exact one at T 0.05 |
| N4 | counts' best candidate is worse than `default` by 0.000–0.010 on the holdout (a list is coarser than 2,048 nearest neighbours), and scores 2–4× faster per byte |
| N5 | Resident memory: dedupe saves 12–20 MB of the expert's 42 MB; counts saves 30–36 MB |
| N6 | The chosen candidate's fresh change against `default` is within ±0.003 of its holdout change |
| N7 | Answers: byte-identical (rule 3) |
| N8 | SCRIBE v2 of `sample` text with the chosen candidate is inside `default`'s interval (B4c: trie + sim + copy 0.056 [0.044, 0.068]) |

### Step 1: the holdout sweep (rule 1)

`cls shape` (the class table `shape:6:3:cp2`), keys over wiki train, the shipped 32-d vectors, one run, the specs in the order below. Over trie + InfiniGram + class (1.995766). The paired change is against `default` in the same run (4 KiB blocks, 2,000 resamples). Memory is the expert's index and rows (`bytes()`); scoring time is the whole holdout (1 MiB) on the shared machine. `default`'s lines equal those of `main` e8748a2's tool (run separately) to the last digit.

| Spec (all `default,` + …) | trie + IG + cls + sim | Against `default` (95%), P(< 0) | Memory | Scoring 1 MiB | Query p50 / p99 | Live |
|---|---:|---|---:|---:|---|---:|
| (`default`: knn, k 2,048) | 1.940055 | | 40.0 MB | 130 s | 0.63 / 4.79 ms | 0.674 |
| dedupe, k 512 | 1.939380 | −0.0007 [−0.0013, −0.0001], 0.99 | 19.3 MB | 49 s | 0.19 / 0.42 ms | 0.681 |
| dedupe, k 2,048 | 1.935892 | −0.0042 [−0.0051, −0.0033], 1.00 | 19.3 MB | 120 s | 0.39 / 1.61 ms | 0.739 |
| **dedupe, k 8,192** | **1.934176** | **−0.0059 [−0.0070, −0.0048], 1.00** | 19.3 MB | 319 s | 0.77 / 4.96 ms | 0.779 |
| dedupe, T 0.02 | 1.945090 | +0.0050 [+0.0038, +0.0062], 0.00 | 19.3 MB | 143 s | 0.41 / 4.49 ms | 0.739 |
| dedupe, T 0.1 | 1.941724 | +0.0017 [+0.0001, +0.0033], 0.02 | 19.3 MB | 116 s | 0.40 / 0.76 ms | 0.739 |
| dedupe, `norm=1` | 1.952130 | +0.0121 [+0.0099, +0.0146], 0.00 | 19.3 MB | 110 s | 0.39 / 0.68 ms | 0.739 |
| counts, probe 1 | 1.956363 | +0.0163 [+0.0150, +0.0177], 0.00 | 6.1 MB | 4 s | 0.015 / 0.059 ms | 0.610 |
| counts, probe 4 | 1.950410 | +0.0104 [+0.0086, +0.0123], 0.00 | 6.1 MB | 6 s | 0.016 / 0.062 ms | 0.690 |
| counts, probe 16 | 1.947715 | +0.0077 [+0.0055, +0.0100], 0.00 | 6.1 MB | 12 s | 0.019 / 0.066 ms | 0.754 |
| counts, probe 64 | 1.945785 | +0.0057 [+0.0032, +0.0083], 0.00 | 6.1 MB | 34 s | 0.031 / 0.086 ms | 0.799 |
| counts, probe 16, T 0 | 1.958960 | +0.0189 [+0.0155, +0.0228], 0.00 | 6.1 MB | 13 s | 0.019 / 0.068 ms | 0.754 |
| counts, probe 16, T 0.02 | 1.952850 | +0.0128 [+0.0109, +0.0149], 0.00 | 6.1 MB | 12 s | 0.019 / 0.068 ms | 0.754 |
| counts, probe 16, T 0.1 | 1.949606 | +0.0096 [+0.0070, +0.0124], 0.00 | 6.1 MB | 12 s | 0.019 / 0.068 ms | 0.754 |
| counts, 512 lists, probe 4 | 1.954302 | +0.0142 [+0.0113, +0.0175], 0.00 | 6.1 MB | 5 s | 0.006 / 0.017 ms | 0.753 |
| counts, 512 lists, probe 16 | 1.952665 | +0.0126 [+0.0094, +0.0162], 0.00 | 6.1 MB | 17 s | 0.009 / 0.044 ms | 0.799 |

- **Distinct keys: 366,725 of 1,059,311 (34.6%).** Two words make few distinct contexts. Storing each once halves the expert (40.0 → 19.3 MB) and keeps every occurrence.
- **Pooling whole counts pays.** At the same k = 2,048, dedupe is better by 0.0042: frequent contexts now count all their occurrences and still reach similar ones. k keeps paying here too (512 → 2,048 → 8,192: −0.0007, −0.0042, −0.0059). At k = 8,192 the 32 probed lists hold about 5,700 distinct keys on average, so the search returns most or all of them, and the index limits it (recall against an exact scan 0.65).
- **T = 0.05 stays best** (0.02 and 0.1 lose), and **letting each context vote once loses** (+0.012): a context's count matters.
- **counts is cheap but coarse.** It needs 6.1 MB and scores 4–38 times faster, but a whole list is too broad a neighbourhood: the best (probe 64) is 0.0057 worse than `default`.
- **Rule 1 picks dedupe with k = 8,192** (1.934176); no other candidate is within 0.0005 of it.

### Step 2: fresh text, read once, surface S and SCRIBE v2 (rule 2)

`cls shape --sim default --sim default,mode=dedupe,k=8192 --sim-fresh --sim-surface`: the fresh text `[8388608, 9437184)` read once for the chosen spec, with `default` beside it for the paired change; lambdas on the whole holdout. Surface S on B4's protocol (200 holdout prompts, 400 bytes, T 0.8, nucleus 0.9, the same seed). Trie + InfiniGram + class on fresh text: 2.085255.

| Read | `default` | dedupe, k 8,192 | Change (95%), P(< 0) |
|---|---:|---:|---|
| Fresh trie + IG + sim | 2.076665 | 2.059846 | |
| **Fresh trie + IG + class + sim** | 2.027493 | **2.023522** | **−0.0040 [−0.0048, −0.0032], 1.00** |
| Surface S, trie + sim | 0.8365 | 0.8828 | +0.0463 [+0.0367, +0.0558] |
| Surface S, trie + class + sim | 0.9111 | 0.9292 | +0.0181 [+0.0107, +0.0252] |

`sample` text (750 GEN lines, the defaults: trie + sim + copy), `scribe ladder --profile v2` as in B4c:

| Text | Kept | SCRIBE v2 (95%) | Surface S | T 0.6 / 0.8 / 1.0 |
|---|---:|---|---:|---|
| `default` (B4c's run) | 613 | 0.056 [0.044, 0.068] | 0.790 | 0.066 / 0.063 / 0.039 |
| dedupe, k 8,192 | 636 | **0.064 [0.052, 0.078]** | 0.845 | 0.083 / 0.076 / 0.033 |

- **The "better" route holds.** Fresh BPC falls by 0.0040 with every resample below zero, surface S rises in both models, SCRIBE v2 is above `default`'s interval's lower end (and its mean higher), and the costs stay inside PLAN §10 (step 3).
- **The gain carries to fresh text at two thirds of its holdout size** (−0.0040 against −0.0059).
- **Text.** Surface S rises more than the BPC suggests (+0.046 over trie + sim): with whole counts the frequent contexts' next words are sampled from their real distribution, not from a 2,048-copy sample. SCRIBE v2 moves within its interval; the text is still word salad.

### Step 3: answers and cost (rule 3, PLAN §10)

`qa` in B4c's rule-3 configuration (C1L without `--copy` and `--cls`: class + sim + copy defaults), dev 130, against B4c's run with `default` (whose rows `main` reproduces):

- **Answers: unchanged.** All 520 rows have the same answer and grade; the four in-answer features are identical; every gate, tally, abstention and risk-coverage line is identical. Only the `question_lp` column moves in every row: the question is outside the answer, so the expert reads it (as in A4b and B4c).
- **Holdout (chat model, second half):** trie + IG + class + sim 1.922375 → **1.914913**; with copy 1.834525 → **1.826934** (−0.0076).
- **Cost:** decode p50 (chat, trie + IG) 1.45 → 1.23 ms; `qa` peak 274.6 → 259.2 MB; time 736 → 765 s on the shared machine.

| Cost | `default` | dedupe, k 8,192 |
|---|---:|---:|
| Expert's memory (index and rows) | 40.0 MB | **19.3 MB** |
| Build (wiki keys) | 24–31 s | 15–18 s |
| Query p50 / p99 (`cls`, fresh read) | 0.61 / 1.18 ms | 0.74 / 1.38 ms |
| Scoring 2 MiB (`cls`, fresh read) | 219 s | 528 s |
| `sample` (1,500 samples), time / peak | 306 s / 82.8 MB | 492 s / **79.1 MB** (558 s / 105.8 MB with the first build; step 4) |
| `context8k`, time / peak | 723 s / 248.8 MB | 504 s / **234.8 MB** |

`context8k` with the new default changes only its two expert lines: trie + IG + class + sim 1.9231 → **1.9159**, with copy 1.8345 → **1.8271**. Every other line, its 50 generated answers included, is identical to B4c's.

### Step 4: the build, and parity

- **The first dedupe build held every key at once** (1,059,311 × 32 bytes) and then copied the distinct ones, so `sample`'s peak, which the expert's build sets, rose 82.8 → 105.8 MB. The final build finds distinct keys with an open-addressing table as they are made, then sorts them by their bytes as the first build did. The index is identical: the new default reprints the sweep's dedupe line (1.964769 / 1.934176) and `sample` writes the same text byte for byte (only its `MODEL` line names the spec). `sample`'s peak is now 79.1 MB, below the old default's.
- **`--no-sim` prints what `main` e8748a2 printed**, byte for byte: `qa` (rows and lines, latencies aside) and `sample` (both files and stdout).
- **The vector file rebuilds byte for byte** with the new default (`cyphalm_trie simvec … --sim default,vec=ppmi_noeval,dim=32`: sha256 `c9fdf9ef…`), so `data/cyphalm/` is unchanged: the file holds the words' projected rows, which do not depend on the mode.
- **`knn` at the same k.** `default,mode=knn` (k 8,192, each occurrence a neighbour) reads 1.936362 on the holdout: dedupe is better by 0.0022 [0.0016, 0.0028] at the same k, with half the query time (p50 0.75 against 1.50 ms). A4b's spec is `default,mode=knn,k=2048`: on the final tool it reprints `main`'s default lines to the last digit (1.984330 / 1.940055, the same table size, recall and live share).
- **Selftest:** with every occurrence pooled at equal weight, knn and dedupe give the same distribution bit for bit at every position of a random text (a planted sort bug fails it).

### Predictions, scored (A4c)

| Prediction | Result | Score |
|---|---|---|
| N1. Distinct keys 35–55% | 366,725 of 1,059,311: 34.6% | **Miss** (just under) |
| N2. dedupe's best better than `default` by 0.000–0.006 on the holdout | −0.0059 (k 8,192) | **Hit** |
| N3. dedupe's best T is lower than 0.05, or `norm=1` wins | T 0.05 best; `norm=1` loses by 0.012 | **Miss** |
| N4. counts' best worse by 0.000–0.010; 2–4× faster | +0.0057 (probe 64), 3.8× faster | **Hit** |
| N5. dedupe saves 12–20 MB; counts 30–36 MB | 20.7 MB; 33.9 MB | **Miss** (dedupe saves a little more); **hit** |
| N6. Fresh change within ±0.003 of the holdout change | −0.0040 against −0.0059 | **Hit** |
| N7. Answers byte-identical | 520 / 520 answers and grades, all in-answer features and every gate line; the question's log-probability moves | **Hit** |
| N8. SCRIBE v2 inside `default`'s interval | 0.064, inside [0.044, 0.068] | **Hit** |

### Decisions (A4c)

1. **The default spec is now dedupe with k = 8,192** (`default` = `W=2,sif=0,k=8192,drop=0,lists=2048,probe=32,src=wiki,mode=dedupe`), by the "better" route of rule 2: fresh text −0.0040 [−0.0048, −0.0032] over `default`, surface S up, SCRIBE v2 up, answers unchanged. `qa`, `sample` and `context8k` use it; the gate and `--no-sim` are as before.
2. **It is also smaller:** the expert holds 19.3 MB instead of 40.0 MB, and no run's peak rises (`qa` 274.6 → 259.2 MB, `context8k` 248.8 → 234.8 MB, `sample` 82.8 → 79.1 MB). It scores more slowly where the expert reads every byte (`cls`: 2.4× per byte; `sample` 306 → 492 s); a query is 0.74 ms at p50, inside PLAN §10.
3. **The counts mode and `norm=1` are removed** (rule 4). They lost on the holdout; their numbers reproduce at commit 2849b32. Whole lists are too coarse a neighbourhood, even at 64 probed.
4. **`mode=knn` stays** for comparison: A4's and A4b's numbers use it (`default,mode=knn,k=2048` is A4b's default).
5. **Next.** At k = 8,192 the 32 probed lists limit the search (recall 0.65): more probed lists is the next dial, at more time per query. Online keys are still open.

## A4d: recall, and keys from the text being read (2026-10-29)

A4c's "Next": the default (dedupe, k 8,192, 2,048 lists, 32 probed) scans about 5,700 distinct keys a query, so it can return at most that many of the 8,192 nearest: its recall against an exact search is 0.65, and k kept paying. Two dials. The design, the rules and the predictions below were written before any A4d run.

**Recall.** More probed lists, other list counts, and a larger k once the search can fill it.

**Online keys (`online=1`).** The contexts of the text already read become keys too.
- At every word end j of the stream being read that lies before the current anchor, the key of the W words ending at j is made as a stored key is, put in its nearest centroid's list, and kept with j.
- A search scans the probed lists' online keys beside their stored ones. An online key is one occurrence: it is live while the stream's bytes [j, j + t) equal the tail (t = the bytes since the anchor), and it proposes the byte at j + t, weighted as a stored key is.
- Every byte an online key reads is already scored: j < anchor and t = n − anchor, so j + t < n, where n is the position being predicted.
- In `cls` the stream is the stored text's own buffer, from the end of the train bytes on; in `qa`, `sample` and `context8k` it is the text being scored or written, from its start. The store starts again when the stream is not the one read so far.
- **Leak check (registered).** A selftest reads a random text through two experts: one sees the whole text, the other the same text with every byte from position m on replaced. Their distributions at m must be identical bit for bit, at many m. A planted leak (keys at j = the anchor itself) must fail it.

**Rules (fixed before the runs).**
1. **Choice, on the holdout's second half** (`cls shape`, keys over wiki train, the shipped vectors; over trie + InfiniGram + class): the lowest BPC; candidates within 0.0005 of it go to the lowest scoring time per byte.
2. **Default, on fresh text read once** for the chosen candidate, paired against the current default in the same run. It replaces the default only if all of these hold:
   - fresh trie + InfiniGram + class + sim lower, with P(paired change < 0) ≥ 0.95 over 4 KiB blocks (LCI);
   - surface S of the change not entirely below 0 (reported alongside, PLAN ledger 64);
   - SCRIBE v2 of `sample` text not below the current default's 95% interval (0.064 [0.052, 0.078]; SCRIBE v2 is the rules' text score, ledger 64);
   - cost: scoring time per byte at most 2× the current default's, and inside PLAN §10 (Q/A p50 ≤ 200 ms on one core; the language system ≤ 4 GB).
3. **Answers (B4B rule 3):** `qa` in B4c's rule-3 configuration: the answer rows byte-identical with the new default and the current one (the expert is off inside answers).
4. **Code debt.** An option that does not become the default and is not needed for a report's numbers is removed; its numbers reproduce at the commit that measured them.

**Candidates** (all `default,` + …): the current default (the reference, first); `probe=64`; `probe=128`; `probe=256`; `k=16384,probe=128`; `lists=512,probe=16`; `lists=8192,probe=128`; `online=1`; `online=1,probe=128`.

**Predictions.**

| # | Prediction |
|---|---|
| D1 | Recall: probe 64 gives 0.85–0.95; probe 128 gives at least 0.95 |
| D2 | The best recall candidate (no online keys) is better than the current default by 0.0005–0.003 on the holdout: k 8,192 now really returns 8,192 contexts |
| D3 | k 16,384 at probe 128 gains at most 0.001 over k 8,192 at probe 128 |
| D4 | Scoring time per byte at probe 128 is 1.3–2.5× the current default's |
| D5 | Online keys gain 0.000–0.004 over trie + InfiniGram + class + sim on the holdout (the copy expert is not in this read, so repeats within the text can show here) |
| D6 | The leak check passes, and the planted leak fails it |
| D7 | The chosen candidate's fresh change is within ±0.002 of its holdout change |
| D8 | Answers byte-identical (rule 3) |
| D9 | SCRIBE v2 of `sample` text with the chosen candidate is inside the current default's interval |

### Step 1: the choice (written before the fresh read)

Rule 1 picks **`online=1,probe=128`**: on the holdout's second half it reads 1.856939 over trie + InfiniGram + class + sim, against 1.934176 for the current default; no other candidate is within 0.0005 of it (`online=1` at probe 32: 1.860386; the best without online keys: probe 128, 1.932203). Its scoring time in that run is 1.98× the current default's (422.7 s against 213.7 s). Rule 2's cost is read in the fresh run, as registered: above 2× there, the candidate fails rule 2 and the default stays. The full holdout table is in step 2.

### Step 2: the holdout sweep (rule 1)

`cls shape`, keys over wiki train, the shipped 32-d vectors; three runs on the shared machine (RA, RB: the tool at `main` e6e9b74; RC: with online keys, whose `default` line equals `main`'s to the last digit). Over trie + InfiniGram + class (1.995766). Each change is paired against `default` in the same run (4 KiB blocks, 2,000 resamples). Memory is the expert's index, rows and online keys (`bytes()`); scoring is the whole holdout (1 MiB); recall is against an exact scan, on 200 holdout positions.

| Spec (`default,` + …) | trie + IG + cls + sim | Against `default` (95%), P(< 0) | Memory | Scoring 1 MiB | Query p50 / p99 | Recall |
|---|---:|---|---:|---:|---|---:|
| (`default`: probe 32) | 1.934176 | | 19.3 MB | 209–214 s | 0.74 / 1.35 ms | 0.652 |
| probe 64 | 1.932624 | −0.0016 [−0.0019, −0.0012], 1.00 | 19.3 MB | 306 s | 1.19 / 2.11 ms | 0.849 |
| probe 128 | 1.932203 | −0.0020 [−0.0023, −0.0016], 1.00 | 19.3 MB | 374 s | 1.46 / 5.59 ms | 0.937 |
| probe 256 | 1.932030 | −0.0021 [−0.0025, −0.0018], 1.00 | 19.3 MB | 532 s | 1.79 / 7.89 ms | 0.979 |
| k 16,384, probe 128 | 1.931675 | −0.0025 [−0.0030, −0.0020], 1.00 | 19.3 MB | 1,082 s | 2.66 / 11.98 ms | 0.834 |
| 512 lists, probe 16 | 1.933235 | −0.0009 [−0.0013, −0.0006], 1.00 | 19.3 MB | 299 s | 1.17 / 2.06 ms | 0.807 |
| 8,192 lists, probe 128 | 1.933680 | −0.0005 [−0.0009, −0.0001], 1.00 | 19.5 MB | 244 s (build 137 s) | 0.80 / 4.89 ms | 0.659 |
| `online=1` (probe 32) | 1.860386 | −0.0738 [−0.0855, −0.0632], 1.00 | 24.6 MB | 252 s | 0.95 / 2.17 ms | 0.652 |
| **`online=1`, probe 128** | **1.856939** | **−0.0772 [−0.0890, −0.0664], 1.00** | 24.6 MB | 423 s | 1.65 / 6.05 ms | 0.937 |

- **Recall pays, a little.** Probing 128 of the 2,048 lists lifts recall from 0.65 to 0.94 and the holdout by 0.0020; 256 adds 0.0002 more. Past that, k 16,384 adds 0.0005 at five times the time. Other list counts are worse at a similar cost.
- **Keys from the text being read pay a lot.** `online=1` gains 0.074 at 1.2× the time. Most of it is inside words (−0.098 of the summed bits per byte there, against −0.037 at word starts): a name or phrase met earlier in the same article is a similar context the stored text does not hold. This read has no copy expert, which is why the gain is so much larger than predicted (step 4 reads it with copy).

### Step 3: fresh text, read once, surface S and SCRIBE v2 (rule 2)

`cls shape --sim default --sim default,online=1,probe=128 --sim-fresh --sim-surface`: the fresh text `[8388608, 9437184)` read once, paired against `default` in the same run; lambdas on the whole holdout. The stream the online keys come from runs from the holdout's start through the fresh text. Surface S on B4's protocol (200 holdout prompts, 400 bytes). `default`'s fresh line equals A4c's read to the last digit (2.023522).

| Read | `default` | `online=1`, probe 128 | Change (95%), P(< 0) |
|---|---:|---:|---|
| Fresh trie + IG + sim | 2.059846 | 1.903987 | |
| **Fresh trie + IG + class + sim** | 2.023522 | **1.878268** | **−0.1453 [−0.1758, −0.1184], 1.00** |
| Surface S, trie + sim | 0.8828 | 0.9010 | +0.0182 [+0.0094, +0.0270] |
| Surface S, trie + class + sim | 0.9292 | 0.9308 | +0.0016 [−0.0049, +0.0079] |
| Scoring 2 MiB (same run) | 650.8 s | 1,215.9 s | 1.87× |

`sample` text with the candidate (750 GEN lines; the defaults: trie + sim + copy), `scribe ladder --profile v2` as in A4c:

| Text | Kept | SCRIBE v2 (95%) | Surface S | T 0.6 / 0.8 / 1.0 |
|---|---:|---|---:|---|
| current default (A4c's run) | 636 | 0.064 [0.052, 0.078] | 0.845 | 0.083 / 0.076 / 0.033 |
| `online=1`, probe 128 | 608 | **0.063 [0.051, 0.078]** | 0.860 | 0.081 / 0.070 / 0.037 |

- **Every part of rule 2 holds.** Fresh BPC falls by 0.145 with every resample below zero; surface S rises over trie + sim and is unchanged over trie + class + sim; SCRIBE v2 is inside the current default's interval; scoring time is 1.87× (under 2×).
- **The fresh gain is about twice the holdout's.** By the fresh text the stream holds the holdout and the fresh text read so far (about 2 MiB of keys, against 0.5–1 MiB in the holdout's second half), so more earlier contexts are found.
- **Text.** SCRIBE v2 does not move; the text is still word salad. Online keys from the model's own sample did not start loops.

### Step 4: answers, the model with copy, and cost (rule 3, PLAN §10)

`qa` in B4c's rule-3 configuration with `--sim default,online=1,probe=128`, against A4c's run with its default (whose rows `main` reproduces):

- **Answers: unchanged.** All 520 rows have the same answer and grade, and every other column is identical except `question_lp` (the question is outside the answer, so the expert reads it).
- **The holdout of the chat model (second half), with the copy expert:** trie + IG + class + sim 1.914913 → **1.839792**; with copy (the default model) 1.826934 → **1.789464** (−0.037). Online keys and the copy expert overlap (the copy expert's own gain falls from 0.088 to 0.050), but half the gain remains on top of copy.
- **Cost:** `qa`'s peak 235.1 MB (unchanged; the online keys of a 1 MiB holdout are 5.3 MB); decode p50 (chat, trie + IG) 1.42 ms; `sample` 890 s against 492 s, at a 79.0 MB peak (A4c: 79.1 MB). The online keys grow by 36 bytes per word of the stream read: 5.3 MB a MiB. A stream much longer than a few MiB would need a cap.
- **`context8k`** with the new default changes only its two expert lines: trie + IG + class + sim 1.9159 → **1.8404**, with copy 1.8271 → **1.7897**. Every other line, its 50 generated answers included, is identical to A4c's run; peak 225.8 MB.

### Step 5: the leak check, and parity

- **Online keys read only bytes already scored.** The selftest reads a random 400-word stream through two experts: one sees the whole stream, the other the same stream with every byte from m on replaced; at every 7th m, in both modes, their distributions at m are identical bit for bit. Online keys are used: the distribution differs from the expert's without them at many positions. A planted leak (keys at j ≤ the anchor instead of j < the anchor) fails the check.
- **With online keys off, nothing changes:** the tool reprints `main`'s `default` lines to the last digit (runs RC and FD).
- **`--no-sim` prints what `main` e6e9b74 printed**, byte for byte: `qa` (rows and lines, latencies aside) and `sample` (both files and stdout).
- **The new default is wired:** `qa` with it gives the explicit spec's rows and lines, and `sample` writes the explicit spec's text byte for byte (only its `MODEL` line names the spec); `sample`'s peak 69.5 MB, 655 s.
- **The selftest and the 9 CTests of HANDOFF §4.4 pass.**

### Predictions, scored (A4d)

| Prediction | Result | Score |
|---|---|---|
| D1. Recall 0.85–0.95 at probe 64; at least 0.95 at probe 128 | 0.849; 0.937 | **Miss** (both just under) |
| D2. The best recall candidate better by 0.0005–0.003 | −0.0021 (probe 256); −0.0025 with k 16,384 | **Hit** |
| D3. k 16,384 gains at most 0.001 over k 8,192 at probe 128 | 0.0005 | **Hit** |
| D4. Scoring time at probe 128 is 1.3–2.5× | 1.78× | **Hit** |
| D5. Online keys gain 0.000–0.004 | −0.074 | **Miss** (far larger: within-article repeats) |
| D6. The leak check passes; the planted leak fails it | Both | **Hit** |
| D7. Fresh change within ±0.002 of the holdout change | −0.145 against −0.077 | **Miss** (the stream is longer by the fresh text) |
| D8. Answers byte-identical | 520 / 520; only `question_lp` moves | **Hit** |
| D9. SCRIBE v2 inside the current default's interval | 0.063 inside [0.052, 0.078] | **Hit** |

### Decisions (A4d)

1. **The default spec is now `online=1` with 128 lists probed** (`default` = `W=2,sif=0,k=8192,drop=0,lists=2048,probe=128,src=wiki,mode=dedupe,online=1`), by the "better" route of rule 2: fresh text −0.1453 [−0.1758, −0.1184], surface S not falling, SCRIBE v2 unchanged, answers unchanged, scoring time 1.87×. `qa`, `sample` and `context8k` use it; the gate and `--no-sim` are as before.
2. **The default model's holdout** (chat model, with copy) improves 1.8269 → **1.7895**.
3. **Nothing is removed** (rule 4): `probe`, `lists` and `k` were already spec settings, and `online` is the default; `online=0` gives A4c's expert exactly.
4. **Next.** A cap or a window on the online keys before a stream grows past a few MiB (they cost 5.3 MB a MiB read).

## A4e: a bound on the online keys (2026-10-29)

**Odin Loch** · 2026-10-29 · follows A4d's "Next"

A4d's online keys cost 36 bytes per word read (a 32-byte key and a 4-byte position): about 5.3 MB per MiB of text, with no end. Every search also scans the probed lists' share of them, so search time grows with the stream too. A4e looks for a bound that stops both without losing the gain. The design, the rules and the predictions below were written before any A4e run.

**The bound (`oncap=N`, 0 = none).** The store holds at most N online keys. Below N nothing changes.
- **`onevict=old`** (the default): the oldest key is dropped, so the store is a window over the last N word ends. A ring of N list numbers (4 bytes each) says which list holds the oldest key. A list frees its dropped front once that front is half the list.
- **`onevict=lru`**: the key found least recently is dropped. A key's stamp is the last search that put it among the k nearest, or its arrival. When the store passes N, it drops to 7N/8 at once (4 bytes of stamp per key).
- **Merging** identical keys (one key, many positions) is measured, not built. It saves the 32 key bytes of a repeated context but keeps 4 bytes per word, so it cannot bound memory by itself.
- **Leak check.** The A4d check (an expert that read a text up to m and one that read the same text with every byte from m on replaced give the same distribution at m, bit for bit) also runs with a bound of 40 keys, for both kinds of drop and both search modes. The selftest also checks that the bound binds, that the store never holds more than the bound, that the window holds exactly the last 40 keys, and that a bound above the stream's keys equals no bound at every position.

**Streams.** `cls` reads the holdout [7, 8) MiB and then a continuation of the corpus (`--sim-stream FROM:LEN`, in place of the fresh MiB). Lambdas are fit on the whole holdout, as for A4d's fresh read. The read is trie + InfiniGram + class + sim over the continuation.
- **Dev:** the continuation is enwik8 [8, 16) MiB (8 MiB; the stream is 9 MiB). [8, 9) MiB is A4d's fresh text, read once there and dev now.
- **Fresh, read once:** the continuation is enwik8 [40, 48) MiB. No CyphaLM expert choice has read it.
- The shipped vectors were built on enwik8 without [7, 9) MiB, so they have seen [9, 16) and [40, 48). That is the same for every candidate, and the comparisons are paired.
- **Growth** (`--sim-growth`): the expert alone reads the stream holdout + [8, 23) MiB. At 1, 2, 4, 8 and 16 MiB of stream it reports keys held, bytes (counted and allocated), resident memory, search p50 and p99, time per byte over the next 64 KiB, and how many held keys repeat a stored or earlier key.

**Candidates** (all `default,` + …): `oncap=0` (A4d's default, the reference); `online=0` (no online keys: the size of the gain); windows `oncap=32768`, `oncap=131072`, `oncap=524288` (about 0.22, 0.9 and 3.6 MiB of text; 1.3, 5.2 and 21 MB); `oncap=131072,onevict=lru`. Each is paired against the reference per byte (`--sim-bits` writes the reference's bits; `--sim-ref` reads them), 4 KiB blocks.

**Rules (fixed before the runs).**
1. **Choice, on dev.** A candidate is eligible if its mean paired change against the reference is at most +0.0005 over the whole continuation and over its last 2 MiB ([14, 16) MiB). Rule 1 picks the eligible candidate with the smallest bound; on a tie, the window. If none is eligible, no bound becomes the default.
2. **Default, on fresh text read once**, the chosen bound paired against the reference in the same run. It becomes the default only if all of these hold:
   - the mean paired change is at most +0.0005, and P(change < +0.001) ≥ 0.95 over 4 KiB blocks (LCI);
   - answers: `qa` in B4c's rule-3 configuration gives byte-identical rows;
   - text: `sample`'s text is byte-identical (the bound cannot bind there), or else SCRIBE v2 of `sample` text is not below the current default's interval (0.063 [0.051, 0.078]);
   - cost: scoring time per byte over the fresh continuation at most the reference's (+5% for a shared machine).
3. **Parity.** With `oncap=0`, every printed number is unchanged: the selftest's earlier lines, `cls`'s `default` lines, `qa` rows and `sample` text, against `main` 7f77615.
4. **Code debt** (A4d's rule 4). An option that does not become the default and is not needed for a report's numbers is removed; its numbers reproduce at the commit that measured them.

**Predictions.**

| # | Prediction |
|---|---|
| E1 | Without a bound the online keys grow 5.0–5.6 MB per MiB of stream, in a straight line (about 85 MB at 16 MiB, counted); search p50 at 16 MiB is at least 2× its value at 1 MiB |
| E2 | At 16 MiB, 30–60% of the held keys repeat a stored key or an earlier online key; merging them would save 25–55% of the bytes and still grow without end |
| E3 | The 2^19 window (3.6 MiB of text) is eligible under rule 1 |
| E4 | The 2^15 window (0.22 MiB) is not eligible, but keeps at least 80% of the online keys' gain over `online=0` on the dev continuation |
| E5 | Least-recently-found at 2^17 is within ±0.0005 of the window at 2^17 |
| E6 | Rule 1 picks a bound of 2^17 or 2^19 keys (5–21 MB) |
| E7 | The chosen bound's fresh change is within ±0.0005 of its dev change |
| E8 | Over the last 2 MiB of the dev continuation, scoring time per byte with the chosen bound is at least 1.3× lower than without |
| E9 | The leak check passes under both kinds of bound; planted bugs (a leak; a window that drops the newest key) fail the selftest |
| E10 | `qa` rows and `sample` text are byte-identical with the chosen bound (it cannot bind in either) |

### Results (A4e)

**Answer: no bound.** Every bound tried loses more than rule 1 allows, and the loss grows with the stream. The online keys stay unbounded, and the bound's code is removed (rule 4). The fresh text [40, 48) MiB was not read, since rule 2 applies only to a chosen bound; it stays unread.

- **Answers: unchanged.** With no bound, `qa` rows are byte-identical to `main`'s (step 3).
- **Text: unchanged.** `sample` text is byte-identical to `main`'s (step 3), so SCRIBE v2 stays 0.063 [0.051, 0.078] and surface S of trie + class + sim stays 0.9308 (A4d).
- **BPC: every bound is worse.** On the 8 MiB dev continuation the best bound (2^19 keys) is +0.0033 [+0.0027, +0.0039] worse over the whole stream and +0.0072 [+0.0055, +0.0091] over the last 2 MiB. Rule 1's limit is +0.0005.
- **Memory: the cost stays.** Without a bound the online keys grow 5.3–5.5 MB per MiB read, in a straight line: 87.6 MB counted (125.8 MB allocated) at 16 MiB.

### Step 1: growth without a bound, and with each bound

The expert alone (no trie, InfiniGram or class) reads the holdout and then enwik8 [8, 23) MiB. At each stream length it reads 64 KiB timed. The index alone is 19.3 MB (69.3 MB resident before the read).

| Stream | Keys held | Online MB (allocated) | Resident MB | Search p50 / p99 µs | µs per byte | Repeat a stored key / an earlier online key |
|---|---|---|---|---|---|---|
| 1 MiB | 147,829 | 5.32 (7.57) | 83.6 | 1,801 / 5,099 | 414 | 71.4% / 3.9% |
| 2 MiB | 298,051 | 10.73 (15.42) | 97.9 | 2,011 / 3,782 | 407 | 71.2% / 4.9% |
| 4 MiB | 604,009 | 21.74 (31.27) | 126.3 | 2,378 / 4,436 | 471 | 69.3% / 6.1% |
| 8 MiB | 1,210,215 | 43.57 (62.58) | 164.0 | 2,778 / 7,022 | 532 | 68.2% / 7.7% |
| 16 MiB | 2,432,200 | 87.56 (125.85) | 218.3 | 3,960 / 9,138 | 716 | 67.5% / 9.8% |

The run's peak is 283.5 MB. An identical earlier run read p50 5,224 µs and 1,036 µs per byte at 16 MiB: on this shared machine the times move by a third, while the counts are exact.

The same read at 16 MiB with each bound:

| Bound | Keys held | Online MB (allocated) | Resident MB | Search p50 / p99 µs | µs per byte | Peak MB |
|---|---|---|---|---|---|---|
| none | 2,432,200 | 87.56 (125.85) | 218.3 | 3,960 / 9,138 | 716 | 283.5 |
| window 2^15 | 32,768 | 1.75 (5.01) | 77.7 | 1,636 / 2,941 | 382 | 80.7 |
| window 2^17 | 131,072 | 7.39 (14.73) | 88.7 | 1,844 / 3,402 | 407 | 88.7 |
| window 2^19 | 524,288 | 30.83 (54.79) | 140.6 | 2,342 / 5,670 | 469 | 140.6 |
| least recently found 2^17 | 121,915 | 4.88 (10.47) | 85.1 | 2,831 / 12,086 | 835 | 85.1 |

- A window holds memory and search time flat from the point it binds.
- Least recently found is slower than the window at every length: it stamps the keys each search finds, and sorts all stamps once per 2^14 keys.
- **Merging repeats** (measured, not built): 77.3% of the held keys repeat a stored key or an earlier online key. Pointing each repeat at the first copy would keep 4 bytes for it instead of 36: 27.4 MB at 16 MiB instead of 87.6 MB (−68.7%). It still grows, about 1.7 MB per MiB.

### Step 2: the choice, on dev (rule 1)

The stream is the holdout and then enwik8 [8, 16) MiB; lambdas are fit on the holdout. Trie + InfiniGram + class + sim over the 8 MiB continuation, each candidate paired with the reference per byte (4 KiB blocks, 4,000 resamples; 2,000 for the last 2 MiB). "Gain kept" is the share of the online keys' gain over `online=0` that the candidate keeps.

| Candidate | BPC | Change, whole | Change, last 2 MiB | Gain kept | Eligible |
|---|---|---|---|---|---|
| `oncap=0` (reference, the default) | 1.942285 | — | — | 100% | — |
| `online=0` | 2.067322 | +0.1250 [+0.1176, +0.1328] | +0.1210 [+0.1094, +0.1346] | 0% | — |
| window 2^15 | 1.956113 | +0.0138 [+0.0127, +0.0150] | +0.0189 [+0.0161, +0.0221] | 88.9% [87.7, 90.2] | no |
| window 2^17 | 1.952505 | +0.0102 [+0.0092, +0.0113] | +0.0170 [+0.0142, +0.0200] | 91.8% [90.7, 92.9] | no |
| window 2^19 | 1.945557 | +0.0033 [+0.0027, +0.0039] | +0.0072 [+0.0055, +0.0091] | 97.4% [96.8, 97.9] | no |
| least recently found 2^17 | 1.958176 | +0.0159 [+0.0149, +0.0169] | +0.0245 [+0.0220, +0.0274] | 87.3% [86.0, 88.5] | no |

Every interval lies above zero (P(change > 0) = 1.00 in every row). No candidate is eligible, so rule 1 picks no bound.

**The loss grows with the stream.** The 2^19 window, by MiB of the continuation: 0, 0, −0.0001, +0.0030, +0.0028, +0.0059, +0.0053, +0.0092. It changes nothing until it binds (in the third MiB), then loses more each MiB. The 2^15 window: +0.0119 in the first MiB, +0.0195 in the last. Keys several MiB back still find the right neighbours: enwik8 repeats link lists, templates and names across articles.

Least recently found against the window at the same size, paired: +0.0057 [+0.0053, +0.0061] worse (last 2 MiB +0.0075). Keeping the keys a search used is worse than keeping the newest.

Scoring time per MiB in these runs is too noisy to read (the reference's last two MiB took 878 s and 899 s; `online=0`'s took 972 s and 423 s). Step 1 gives the times.

### Step 3: the leak check, planted bugs, and parity

- **At a854346 the selftest passes** with the bound's checks: under a bound of 40 keys (both kinds of drop, both search modes) the leak check holds, the bound binds, the store never holds more than 40 keys, the window holds exactly the last 40, and a bound above the stream's keys equals no bound at every position.
- **Planted bugs fail it** at the A4e check: a leak (under a bound, the key at the current word end, which proposes the byte being predicted) and a window that drops the newest key instead of the oldest.
- **The half-list free (a854346) changes no bit.** On a small stream the window's and least recently found's bits are identical before and after it.
- **Parity at a854346, `oncap=0`, against `main` 7f77615** (the same commands, `main`'s binary): `qa` rows byte-identical (B4c's rule-3 configuration) and every stdout line identical except 4 latency maxima and the build time; `sample` text byte-identical (dev 492,257 and fresh 493,903 bytes, the `MODEL` line included); the selftest's lines identical; `cls`'s `default` lines reprint A4d's (holdout 1.856939, the fresh MiB 1.878268).
- **Parity at the final commit (bound removed).** Against `main` 7f77615 the code differs only in comments, three new `cls` flags, the new `simpair` command, `cls` lines printed only with those flags, a range form of `latency_us` (the old call returns the same value) and three read-only diagnostics. `qa` and `sample` run none of it, and `cls` without the new flags runs `main`'s code. Checked by running:
  - the selftest's lines are identical to `main`'s;
  - `sample`'s first 67,881 bytes (the `MODEL` line and the first 104 `GEN` lines) are byte-identical to `main`'s;
  - the growth read's counts at 1 and 2 MiB (keys held, MB counted and allocated, repeats) equal a854346's exactly.
- **The full `qa`, `sample` and `cls` runs at the final commit were stopped.** The shared machine gave each about 6% of a core: 7 minutes of CPU in 2 hours. Their peaks so far were `qa` 184.4 MB, `sample` 69.3 MB and `cls` 305.0 MB (`main`: 184.7 and 69.1 MB). A full `qa` and `sample` parity run belongs in the merge check.
- **The selftest and the 9 CTests of HANDOFF §4.4 pass** at the final commit.

### Predictions, scored (A4e)

| Prediction | Result | Score |
|---|---|---|
| E1. 5.0–5.6 MB per MiB, straight; about 85 MB at 16 MiB; p50 at 16 MiB ≥ 2× at 1 MiB | 5.32, 10.73, 21.74, 43.57, 87.56 MB at 1–16 MiB; p50 3,960 / 1,801 = 2.2× (the earlier run 2.8×) | **Hit** |
| E2. 30–60% of held keys repeat; merging saves 25–55% and still grows | 77.3% repeat; merging saves 68.7%; it still grows (about 1.7 MB per MiB) | **Miss** (more repeats than predicted) |
| E3. The 2^19 window is eligible | +0.0033 whole, +0.0072 last 2 MiB | **Miss** |
| E4. 2^15 not eligible, keeps ≥ 80% of the gain | +0.0138; keeps 88.9% [87.7, 90.2] | **Hit** |
| E5. Least recently found within ±0.0005 of the window at 2^17 | +0.0057 [+0.0053, +0.0061] worse | **Miss** |
| E6. Rule 1 picks 2^17 or 2^19 | No bound eligible | **Miss** |
| E7. Fresh change within ±0.0005 of dev | No bound chosen; fresh text not read | Not scored |
| E8. Time per byte over the last 2 MiB ≥ 1.3× lower with the chosen bound | No bound chosen. In step 1, at 16 MiB: 1.5× lower with 2^19, 1.8× with 2^17 | Not scored |
| E9. Leak check passes under both bounds; planted bugs fail | Both | **Hit** |
| E10. `qa` rows and `sample` text byte-identical with the chosen bound | No bound chosen; with no bound both are byte-identical to `main` | Not scored |

### Decisions (A4e)

1. **No bound becomes the default** (rule 1). The similar-context expert keeps every online key: 5.3–5.5 MB per MiB read, and search time grows with them (p50 2.2× at 16 MiB).
2. **`oncap` and `onevict` are removed** (rule 4), with the bound's selftest checks. Their numbers reproduce at commit a854346 (the dev reads ran on builds before it; a854346 changes memory only, not bits). The measuring tools stay, because they measure the default: `cls --sim-stream FROM:LEN`, `--sim-bits PREFIX`, `--sim-growth MIB,…` and `simpair [--margin M]`.
3. **The fresh text [40, 48) MiB stays unread.**
4. **Where it matters.** `qa` reads a 1 MiB holdout (5.3 MB of keys) and `sample` reads short prompts, so neither grows. Only a long stream does: 100 MiB read would hold about 550 MB of keys.
5. **Next.** Merge repeated online keys (one key, several positions): 68.7% less memory at 16 MiB, keeping every key. It changes which keys fill the k nearest, so it needs its own dev read. A bound for long streams only would cost at least +0.003 BPC (2^19) and is Odin's call.

### Reproduce (A4e)

```bash
# dev (rule 1): the reference and online=0, the windows, then 2^19 and least recently found (bits for simpair)
./cyphalm_trie cls enwik8 shape --sim default --sim default,online=0 --sim-stream 8M:8M --sim-bits A
./cyphalm_trie cls enwik8 shape --sim default,oncap=32768 --sim default,oncap=131072 --sim-stream 8M:8M --sim-bits B
./cyphalm_trie cls enwik8 shape --sim default,oncap=524288 --sim default,oncap=131072,onevict=lru --sim-stream 8M:8M --sim-bits C
./cyphalm_trie simpair A.0.f64 B.0.f64 B.1.f64 C.0.f64 C.1.f64 A.1.f64    # each against the reference
./cyphalm_trie simpair B.1.f64 C.1.f64                                     # least recently found against the window
# growth (step 1), one spec a run
./cyphalm_trie cls enwik8 shape --sim default --sim-stream 8M:15M --sim-growth 1,2,4,8,16
./cyphalm_trie cls enwik8 shape --sim default,oncap=131072 --sim-stream 8M:15M --sim-growth 1,2,4,8,16
```

The bound's specs need commit a854346; the reference, `online=0` and growth without a bound also run at the final commit. Runs and bit files are in `/root/work/a4e/runs/` (`dev/`, `growth/`, `parity/`, `parity2/`).

---

## Reproduce

The numbers come from `cyphalm_trie` at sha256 `8af187a0…b30948` plus this report's changes, built against the repository's headers of 2026-10-28 00:06. The parity lines (B4's holdout, fresh and surface numbers; the P2 grid's run B) reproduce to the last digit. The same changes merged onto the later `cyphalm_trie.cpp` (sha256 `02fc5318…68824b`, after the A1d/A2b, M7 and RAM work) and built against that day's later headers print the same holdout numbers for the chosen spec (1.987329 / 1.938765) and the same parity lines.

```bash
gcc -O2 -c native/third_party/libsais/libsais.c -o libsais.o
g++ -O2 -std=c++20 -I native/include -I native/third_party -I native/third_party/libsais \
    native/tools/cyphalm_trie.cpp native/src/cyphalm/infinigram.cpp libsais.o -o cyphalm_trie
./cyphalm_trie selftest                     # includes the similar-context checks

# PPMI-SVD without the BPC holdout and the fresh text (C1's tool; 85 s, 602 MB)
head -c 7340032 enwik8 > enwik8_noeval && tail -c +9437185 enwik8 >> enwik8_noeval
./ppmi_svd enwik8_noeval EN-WS-353-ALL.txt EN-SIMLEX-999.txt EN-MEN-TR-3k.txt vec/ppmi_noeval --no-ri --no-ablation
# vec/ppmi_all = C1's enwik8.vectors.f32 / .vocab.tsv; vec/s1b = sgns_v0p3_e5_wc.npy and its index written as
# rows of float32 plus "word<TAB>count"; vec/cls256, vec/cls1024 = one random unit 300-d vector per k-means class
# (numpy default_rng(20261028 + k)); vec/rand = one random unit 300-d vector per PPMI word (default_rng(20261028)).

V=vec/ppmi_noeval
# sweeps (holdout second half only); specs that share an index reuse it
./cyphalm_trie cls enwik8 shape:6 --sim vec=$V,k=32 --sim vec=$V,k=8 --sim vec=$V,k=128 ...   # sweep 1 (table order)
./cyphalm_trie cls enwik8 shape:6 --sim vec=$V,sif=0,W=1,k=512 ...                            # sweeps 2-5 likewise
# the decision read: fresh text once, surface S
S="vec=$V,sif=0,W=2,k=2048,drop=0,lists=2048,probe=32"
./cyphalm_trie cls enwik8 shape:6 --sim "$S" --sim-fresh --sim-surface

# answers (dev; the test file only for the shard leakage check)
Q="qa enwik8 dolly.jsonl --dev bench/eval/t_qa_wiki_dev.tsv --leak bench/eval/t_qa_wiki.tsv --cls shape:6"
./cyphalm_trie $Q --out QB.rows.tsv                                     # QB (= the P2 grid's run B)
./cyphalm_trie $Q --out QS.rows.tsv --sim "$S"                          # QS: wiki + chat keys
./cyphalm_trie $Q --out QW.rows.tsv --sim "$S,src=wiki"                 # QW
./cyphalm_trie $Q --out QF.rows.tsv --shard shard_facts.tsv --sim "$S"  # QF (D4's shard, sha256 438aa81c…)
./cyphalm_trie $Q --out QFW.rows.tsv --shard shard_facts.tsv --sim "$S,src=wiki"
./cyphalm_trie $Q --out QS8.rows.tsv --sim "vec=$V,sif=1,W=8,k=128"
# the frozen test, once
./cyphalm_trie qa enwik8 dolly.jsonl --dev bench/eval/t_qa_wiki_dev.tsv --test bench/eval/t_qa_wiki.tsv \
    --cls shape:6 --sim "$S,src=wiki" --out TEST.rows.tsv
# A4c (from the repository root; enwik8 the 100 MB file). Step 1: the sweep (16 specs; the counts mode is at commit 2849b32)
./cyphalm_trie cls enwik8 shape --sim default,mode=knn,k=2048 --sim default,mode=dedupe,k=512 --sim default,mode=dedupe,k=2048 \
  --sim default,mode=dedupe,k=8192 --sim default,mode=dedupe,k=2048,T=0.02 --sim default,mode=dedupe,k=2048,T=0.1
# step 2: fresh text once and surface S (A4c ran it before the default changed, as --sim default --sim default,mode=dedupe,k=8192)
./cyphalm_trie cls enwik8 shape --sim default,mode=knn,k=2048 --sim default --sim-fresh --sim-surface
./cyphalm_trie sample enwik8 dev.txt fresh.txt      # then scribe ladder --gen-trie dev.txt as in B4c
# step 3: answers (B4c's rule-3 configuration), against --sim default,mode=knn,k=2048
./cyphalm_trie qa enwik8 dolly.jsonl --dev bench/eval/t_qa_wiki_dev.tsv --leak bench/eval/t_qa_wiki.tsv --parse parse_dev_span.tsv \
  --dk dk_mention.tsv --dk-ratio 0.20 --fam after --fam-stat ent_title --fam-train low --shard shard_facts.tsv --base a0_rows.tsv --out rows.tsv
# A4d (the tool at main e6e9b74 for RA and RB; with online keys for RC and after). Step 2: the holdout sweep
./cyphalm_trie cls enwik8 shape --sim default,online=0,probe=32 --sim default,online=0,probe=64 --sim default,online=0,probe=128 \
  --sim default,online=0,probe=256 --sim default,online=0,k=16384,probe=128
./cyphalm_trie cls enwik8 shape --sim default,online=0,probe=32 --sim default,online=0,lists=512,probe=16 --sim default,online=0,lists=8192,probe=128
./cyphalm_trie cls enwik8 shape --sim default,online=0,probe=32 --sim default,online=1,probe=32 --sim default
# step 3: fresh text once and surface S (run before the default changed, as --sim default --sim default,online=1,probe=128)
./cyphalm_trie cls enwik8 shape --sim default,online=0,probe=32 --sim default --sim-fresh --sim-surface
./cyphalm_trie sample enwik8 dev.txt fresh.txt      # then scribe ladder --gen-trie dev.txt as in B4c
# step 4: answers, against --sim default,online=0,probe=32 (A4c's default); the same qa command as A4c's step 3
```

Output lines: `cls` prints `sim SPEC keys … recall`, `holdout2 …`, the gains with their intervals, `holdout2_split` (silent / word start / inside a word), `…_vs_first_sim` (paired against the first spec), and with the flags `fresh …` and `surface …`. `qa` adds `sim …` and `sim_lci …` (the holdout read), `SIM_LAT`, `LATENCY` (decode time a question) and `SIM_NB` (the three nearest stored contexts at each dev chat prompt). Run outputs, the sweep scripts and the vector files are in `/root/work/r2/` (`runs/`, `vec/`).

**A4b** (the tool at main `0a90374` plus this step's changes; `$C1L` is A1d's C1L options with `--cls shape:6:2:f`):

```bash
C1L="--dev $DEV --leak $TEST --parse parse_dev_span.tsv --dk dk_mention.tsv --dk-ratio 0.20 --fam after --fam-stat ent_title \
     --fam-train low --shard shard_facts.tsv --copy --base a0_rows.tsv --cls shape:6:2:f"
./cyphalm_trie selftest                                  # the gate's rule, the default spec, a projected file round trip
# step 3: the vector files, and the lossy read (fresh text, surface S, paired against the full vectors)
./cyphalm_trie simvec enwik8 --sim default,vec=vec/ppmi_noeval,dim=64 --out simvec_ppmi64.bin
./cyphalm_trie simvec enwik8 --sim default,vec=vec/ppmi_noeval,dim=32 --out simvec_ppmi32.bin   # = data/cyphalm/
./cyphalm_trie cls enwik8 shape:6:2:f --sim default,vec=vec/ppmi_noeval --sim default,vec=simvec_ppmi64.bin \
    --sim default,vec=simvec_ppmi32.bin --sim-fresh --sim-surface
# step 1 (from the repository root, or with CYPHA_SIMVEC set)
./cyphalm_trie qa enwik8 dolly.jsonl $C1L --no-sim --out G0N.rows.tsv     # without the expert (= G0, main before)
./cyphalm_trie qa enwik8 dolly.jsonl $C1L --out G2.rows.tsv               # the default: gated
./cyphalm_trie qa enwik8 dolly.jsonl $C1L --sim default,gate=0 --out G1.rows.tsv
# step 2 and step 4's subcommand peaks
./cyphalm_trie sample enwik8 trie_dev.txt trie_fresh.txt --no-sim         # B1's files
./cyphalm_trie sample enwik8 sim_dev.txt sim_fresh.txt
./cyphalm_trie context8k enwik8 dolly.jsonl
./cyphalm_trie qa enwik8 dolly.jsonl --dev $DEV --leak $TEST --no-sim; ./cyphalm_trie qa enwik8 dolly.jsonl --dev $DEV --leak $TEST
# step 5
scribe ladder --pack enwik:enwik8:16:48 --ref enwik:enwik8:48:56 --test enwik:enwik8:56:64 --vectors v32 --profile v2 \
    --seed 20261027 --gen-trie trie_dev.txt        # and --gen-trie sim_dev.txt
```

Step 4's harness (`simmem.cpp`), the run outputs and the files are in `/root/work/r2/v2/` (`runs/`, `mem/`, `vec/`, `rebuild/`). Runs G0 (main before the class change, `--cls shape:6`) and G0N print the same lines and rows. The binary used for the runs differs from the final source only in comments, help text, the default file's name (the runs set the path) and an up-front check that the vector file exists.
