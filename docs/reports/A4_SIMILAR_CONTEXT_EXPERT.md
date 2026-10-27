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
