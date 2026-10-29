# E1: a part-of-speech tagger whose decisions the Cypha classifier makes

**Odin Loch** · 2026-10-29 · PLAN Track E, E1 · follows [E1_POS_TAGGER.md](E1_POS_TAGGER.md) (the averaged-perceptron baseline, test UPOS 94.81%) and [E4_QUESTION_CLASSIFIER.md](E4_QUESTION_CLASSIFIER.md) (Cypha's classifier on TREC: 75.0 against the perceptron's 89–91)

(The design, the reads, the rules and the predictions below were written and committed before any run of the new tagger. Results are added under them.)

## Result (test, read once, 2026-10-29)

**The gate is not met.** The Cypha-classifier tagger tags UD English EWT test at **92.06%** UPOS (sentence bootstrap [91.67, 92.47]; Wilson [91.72, 92.39]). The gate is 94%. The perceptron (E1, the same test file) scores 94.81%. The perceptron keeps syntax. Its tagger is unchanged, and so is the question parser that reads it.

| Read (test) | Cypha tagger | Perceptron (E1) |
|---|---:|---:|
| UPOS, all tokens | **92.06%** | 94.81% |
| Known words / unknown words | 94.17% / 71.07% | 96.38% / 79.23% |
| Question sentences | 90.66% | 93.25% |
| Calibrated ECE (temperature from dev) | 0.0415 | 0.0083 |
| Accuracy on the most confident 90% / 80% | 95.55% / 96.94% | 98.05% / 99.03% |
| AUROC of 1 − confidence for "the tag is wrong" | 0.848 | 0.903 |
| OOD flag rate: test, unknown words, random letters, Wikipedia prose | 100% everywhere | — |
| Training / tagging | 252 s (2 passes) / 5,135 tokens/s | 12 s / about 1M tokens/s |

- **What made the classifier work at all:** dense count features (smoothed log P(tag | feature)) and whitening by the pooled within-class covariance. Without whitening the best dev UPOS was 83.8%; with it, 92.0%. Online Cypha then comes within 0.4 points of the exact-means ceiling of its own scoring rule (92.3%).
- **Why it stops at 92%:** Cypha's rule is a shared-variance Gaussian, a linear rule over the input. The perceptron combines many sparse conjunctions (word with previous tag, suffix with previous tag). On test the two disagree on 1,594 tokens: the perceptron alone is right on 1,142, Cypha alone on 452. The largest Cypha confusions are NOUN↔PROPN (429), NOUN↔VERB (226) and SCONJ→ADP (108).
- **The OOD flag cannot be used here.** `Cypha::predict` sets it when the GH gate's inflation exceeds a fixed constant (`kOodThreshold` = 3.0 in `cypha.hpp`); `ood_sigma` does not enter it. On this 279-dimensional input the inflation is above 3 for every token, in or out of domain. A threshold set from the training distribution (for example a quantile of the inflation on training rows) would be needed; that is a library change outside this item.
- **PLAN §11's fallback is not triggered** (test is above 90%), but the confidence half of it does not hold: by rule 5 Cypha's calibrated confidence is not useful here (95.55% against the perceptron's 98.05% at 90% coverage), and the OOD flag is not useful (it fires on 100% of test tokens).

**Question.** The bill of work's E1 item is still open for "the Cypha-classifier tagger". PLAN §3.1 has Core Cypha make the discrete decisions, the part-of-speech tag among them. Can the Cypha classifier (DIF: WorldPrior θ₀ plus ClassDifferential Δk, diagonal Gaussians, natural-gradient updates, NIG/GH gate, OOD flag) tag UD English EWT at the gate, 94% UPOS on the test split? And are its confidence and OOD flag useful?

**Rule 10.** The tagger tags. It answers no question. The question parser (`pos_tagger question`, read by `cyphalm_trie --parse`) keeps the perceptron in this item; swapping it would need its own read of the parser's outputs.

## Why this is hard, and the design that answers it

E4 found that Cypha's classifier learns badly from sparse hashed text features. Its score is a Gaussian log-likelihood ratio with a shared diagonal variance, so it is a linear rule over a dense input. Sparse one-hot features give it little to work with.

So the tagger feeds it **dense count features**. For every feature of a token (its word, its suffixes, its neighbours, the tags before it), the input holds the smoothed distribution of tags that feature had in training, as 17 log-probabilities. The classifier then learns how much to trust each feature. This is stacking, and it is how count statistics become a decision.

## The features (fixed before any run)

For token *i* with word *w*:

| Block | Feature *f* | Dimensions |
|---|---|---|
| B1 | the exact form | 17 |
| B2 | the lowercase form | 17 |
| B3–B6 | the lowercase suffix of 1, 2, 3 and 4 characters | 4 × 17 |
| B7 | the prefix of 2 characters | 17 |
| B8 | the word shape (runs capped at 2: `Xxx`, `dd`, `x-x`) | 17 |
| B9 | the previous lowercase word | 17 |
| B10 | the next lowercase word | 17 |
| B11 | the previous tag | 17 |
| B12 | the previous two tags | 17 |
| B13 | the previous word with this word | 17 |
| B14 | this word with the next word | 17 |
| S | scalars: log(1 + count of the form in training), unseen form, initial capital, all capitals, has digit, has hyphen, first in sentence, sentence ends in `?` | 8 |
| V | word vector of the lowercase form (C track), zeros and a flag when the word has none | k + 1 |

- **Each block** is log P(t | f) for the 17 UPOS tags, with P(t | f) = (c(f, t) + α · P(t)) / (c(f) + α), where P(t) is the training tag prior. α is chosen on dev from {0.5, 2}.
- **Cross-fitting.** On the training split, a token's count features come from the other four of five folds (folds by sentence index mod 5). Otherwise every training token would see its own label inside B1. Dev and test features come from all of train.
- **Previous tags.** In training, B11 and B12 use the gold tags. In tagging, they use the tagger's own predictions, left to right (greedy).
- **Vectors.** Two candidates, chosen on dev: (a) `data/cyphalm/simvec_ppmi32.bin`, committed and reproducible (k = 32); (b) the C1 PPMI-SVD vectors (`enwik8.vectors.f32`, 50,000 × 300; the first 64 dimensions, k = 64).
- **Scaling.** Every input dimension is standardised by its training mean and standard deviation, as E4 did, because the classifier shares one variance per dimension across classes.

## The classifier (fixed before any run)

`cypha::Cypha`, a fresh model from `create_fresh_model_root` (identity `W_enc`, latent dimension = input dimension, field dimension 16), loaded through `Cypha::load`, `train_params().enc_lr = 0` so `W_enc` stays frozen, as E4 and Branch A did. Training is `Cypha::update`, one token at a time, sentences shuffled each pass. Tagging is `Cypha::predict` with the GH path (the library default and the only path that sets the OOD flag). The grid for dev: `delta_lr` ∈ {0.003, 0.01, 0.03}, `world_lr` ∈ {0.002, 0.008}, passes ∈ {1, 2, 3}.

## The comparisons

1. **The perceptron** (`native/tools/pos_tagger.cpp`, E1's baseline), retrained here with its own protocol: dev and test UPOS.
2. **Counts alone, no learning:** argmax over tags of the sum of B1–B14 (uniform weights). It shows what the classifier adds.
3. **Exact class means (diagnostic, not a candidate):** the same standardised input, class means and a shared diagonal variance computed in one batch. It is the ceiling of Cypha's scoring rule, as in E4, and separates the cost of the online updates from the cost of the model family.

## Reads

- **Accuracy:** UPOS on dev and test; known and unknown words (unknown = the form was not in train); question sentences. Intervals: Wilson on tokens, and a bootstrap over sentences (2,000 resamples), which respects that tokens in a sentence are not independent. The gate is read on the bootstrap interval's centre and reported with both.
- **Calibration:** the softmax of Cypha's scores with a temperature fitted on dev (negative log-likelihood); expected calibration error (15 equal-width bins) native and calibrated on test; accuracy on the most confident 90% and 80% of tokens.
- **OOD:** the share of tokens with the OOD flag set on (a) test, (b) test's unknown words, (c) test with every word replaced by random lowercase letters of the same length (seeded), (d) 300 sentences of Wikipedia prose from enwik8 (no gold tags; flag rate only). AUROC of the flag and of 1 − confidence for "the tag is wrong" on test.
- **Cost:** training time, tagging speed (tokens per second, single thread), model size.

## Decision rules (fixed before any run)

1. **Dev choice.** Highest dev UPOS over the grid and the two candidates (α, vectors). Ties within 0.05 points go to fewer passes, then the smaller `delta_lr`.
2. **Design rounds.** PLAN §11 allows two design rounds before its fallback. Round 1 is this design. If round 1's best dev UPOS is under 94%, one amendment may be registered (written here before it runs) and run on dev. Then the test split is read once with the dev choice.
3. **The gate.** E1's Cypha tagger passes if its test UPOS is at least 94%.
4. **The fallback (PLAN §11).** If the test UPOS is under 90%, the perceptron keeps syntax, and the report says whether Cypha's confidence and OOD flag still help (rule 5).
5. **Confidence.** Cypha's calibrated confidence is "useful" if, on test, its accuracy at 90% coverage is at least the temperature-scaled perceptron's. The OOD flag is "useful" if it fires on at least 20% of the random-letter words and on under 5% of test tokens.
6. Nothing changes after the test read.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| P1 | Dev UPOS of the dev choice: 92.0–95.0 (centre 93.5) |
| P2 | Test UPOS: 93.0–94.5; the gate (94%) is met only narrowly or missed narrowly |
| P3 | Cypha is below the perceptron (94.81) by 0.5–2.0 points on test |
| P4 | Counts alone (no learning) score 88–92% on dev; the classifier adds at least 2 points |
| P5 | Exact class means are within 1 point of online Cypha on dev |
| P6 | Unknown-word accuracy is at least the perceptron's (79.2% on test), because of the suffix blocks and vectors |
| P7 | Native ECE above 0.05; calibrated ECE under 0.02 |
| P8 | The OOD flag fires on more random-letter words than test tokens, but meets rule 5's 20% / 5% bar only in part (E4 saw it never fire) |
| P9 | Training under 10 minutes; tagging at least 5,000 tokens per second |

## Data

Universal Dependencies English EWT **v2.18 (2026-05-15)**, fetched again on 2026-10-29 from `raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/` into its own directory. Licence CC BY-SA 4.0 (`LICENSE.txt` in that repository). The three files are byte-identical to E1's copies of 2026-10-27:

| Split | SHA-256 (first 16 hex) |
|---|---|
| train | `d68e06122a702464` |
| dev | `39239e0a60db3ae6` |
| test | `fa024f43dc5da3c5` |

Tokens are the syntactic word lines; multiword-token range lines and empty nodes are skipped, as in E1.

## Amendment 1, round 2 (registered 2026-10-29, before any round-2 run)

**What round 1 showed so far.** The first configuration (α 2, committed vectors, `delta_lr` 0.01, `world_lr` 0.008, one pass) gave on dev: counts alone 88.19%, exact class means 83.80%, Cypha 80.97%. The exact-means number is the ceiling of Cypha's scoring rule on this input, and it is below counts alone. The OOD flag fired on 86% of dev tokens. The rest of the round-1 grid runs as registered; its best dev UPOS will decide whether this amendment is used (rule 2: only if it is under 94%).

**Why.** Cypha's score is a Gaussian log-likelihood ratio with one shared variance per dimension. It treats every dimension as independent. The count blocks are strongly correlated (17 log-probabilities from one feature move together, and B1, B2, B13 and B14 all carry the word), so the rule counts the same evidence many times.

**The amendment.** Before Cypha sees a token, its standardised input is whitened by the pooled within-class covariance of the training rows: Σ_w = (1/N) Σ (z − μ_y)(z − μ_y)ᵀ, eigendecomposed, and x' = Λ^(−1/2) Uᵀ z with each eigenvalue raised by ε times the mean eigenvalue. In this space a shared-variance Gaussian is linear discriminant analysis, the best linear rule when the classes share a covariance. `W_enc` stays frozen (identity); the whitening is fixed preprocessing from training counts, as E4's whitened input was. Nothing else changes.

**Round-2 choices, in two stages (fixed now):**
1. (α, vectors, ε) with α ∈ {0.5, 2}, vectors ∈ {committed, C1}, ε ∈ {0.01, 0.1}: by the dev UPOS of exact class means in the whitened space (LDA), the cheap read of the input's ceiling.
2. On that input, the Cypha grid as in round 1 (`delta_lr` ∈ {0.003, 0.01, 0.03}, `world_lr` ∈ {0.002, 0.008}, passes 1–3), by dev UPOS (rule 1's ties).

The dev choice of the whole item is the better of round 1's best and round 2's best on dev UPOS. Then the test split is read once.

**Predictions for round 2 (registered now):**

| # | Prediction |
|---|---|
| A1 | Whitened exact class means (LDA) on dev: 91–94% |
| A2 | Cypha on the whitened input on dev: within 1.5 points of A1's number |
| A3 | Round 2's best beats round 1's best on dev by at least 4 points |
| A4 | The OOD flag still fires on more than 20% of in-domain tokens (the threshold is not tuned to this input) |

### Round 2, stage 1 (dev, 2026-10-29)

Exact class means in the whitened space (LDA), dev UPOS:

| α | Vectors | ε 0.01 | ε 0.1 |
|---|---|---:|---:|
| 0.5 | committed (simvec) | **92.33%** | 92.33% |
| 0.5 | C1 | 92.30% | 92.30% |
| 2 | committed (simvec) | 92.10% | 92.10% |
| 2 | C1 | 92.19% | 92.19% |

- Whitening lifts the ceiling of Cypha's scoring rule from 84.25% to 92.33% (A1: hit).
- **ε does not change this read.** The exact-means rule re-estimates each dimension's variance, and ε only rescales the whitened dimensions, so the read is blind to ε. Online Cypha is not, because its updates are not scale-invariant. So stage 2 runs both ε values on the chosen input (α 0.5, committed vectors); the stage-2 rule decides between them. This is the only change to the registered plan, and it is made before any stage-2 run.

### Round 1 and round 2, stage 2 (dev, 2026-10-29)

**Round 1 (no whitening).** Twelve of the 24 configurations ran (α 0.5); the best dev UPOS was **83.76%** (committed vectors, `delta_lr` 0.03, `world_lr` 0.002, pass 1), under its own exact-means ceiling (84.25%). The remaining 12 (α 2) were stopped after round 2's first results: α 2's ceiling (exact means, 83.80%) is 8 points under round 2's best, and online Cypha stayed under its ceiling in every run. This is a deviation from the registered grid; it cannot change the dev choice.

**Round 2, stage 2 (whitened input, α 0.5, committed vectors), dev UPOS:**

| ε | `delta_lr` | `world_lr` | Pass 1 | Pass 2 | Pass 3 | Calibrated ECE (after pass 3) | OOD flag rate on dev |
|---|---|---|---:|---:|---:|---:|---:|
| 0.01 | 0.003 | 0.002 | | | 90.10* | 0.052 | 0.84 |
| 0.01 | 0.03 | 0.002 | 91.80 | 91.73 | **91.98** | 0.035 | 0.82 |
| 0.01 | 0.03 | 0.008 | 91.57 | 91.38 | 91.57 | 0.037 | 0.83 |
| 0.01 | 0.01 | 0.002 | 91.03 | 91.35 | 91.71 | 0.045 | 0.82 |
| 0.1 | 0.03 | 0.008 | | **91.96** | | 0.038 | 0.97 |
| 0.1 | 0.03 | 0.002 | | 91.75 | 91.89 | 0.035 | 0.97 |
| 0.1 | 0.01 | 0.002 | 91.30 | 91.53 | 91.74 | 0.036 | 0.98 |

(\* best pass of that row. Full per-pass lines are in the run outputs.)

- **Dev choice (rule 1).** 91.98% (ε 0.01, 3 passes) and 91.96% (ε 0.1, `delta_lr` 0.03, `world_lr` 0.008, 2 passes) are within 0.05 points, so the tie goes to fewer passes: **ε 0.1, `delta_lr` 0.03, `world_lr` 0.008, 2 passes, dev 91.96%**.
- Online Cypha comes within 0.4 points of the whitened exact-means ceiling (92.33%) (A2: hit). Round 2 beats round 1 by 8.2 points (A3: hit).
- **The OOD flag fires on 82–98% of in-domain dev tokens** (A4: hit). Its threshold is not set for this input.
- Dev is under 94% after both design rounds. By rule 2 the test split is now read once with the dev choice.

## The test read (once, 2026-10-29)

Command: `cypha_tagger run … --read-test --wiki wiki300.txt --alpha 0.5 --vec simvec --whiten 0.1 --delta-lr 0.03 --world-lr 0.008 --passes 2 --diag --out final` (binary from commit 9022763's source). Dev after pass 2 reproduced the grid's 91.96% exactly, so the tested model is the chosen one.

```
test, Cypha   all 92.06% (23102/25094)  known 94.17%  unknown 71.07% (2292)  questions 90.66% (1927)
UPOS 92.06%  Wilson [91.72, 92.39]  sentence bootstrap [91.67, 92.47]  gate 94%: NOT MET
unknown words: Wilson [69.18, 72.89]  questions: Wilson [89.28, 91.88]
calibration (T 3.490 from dev): ECE native 0.3730  calibrated 0.0415  acc@90% 95.55%  acc@80% 96.94%
AUROC for 'tag is wrong': 1 - calibrated confidence 0.8475  OOD flag 0.5000
OOD flag rate: test 1.0000  unknown words 1.0000  random letters 1.0000  Wikipedia prose 1.0000 (6,686 tokens, 300 sentences)
cost: training 252.3 s (2 passes), tagging 5135 tokens/s single thread, count tables 279,232 keys
```

**The perceptron's numbers** come from E1's frozen model (`pos_tagger train` reproduces it byte for byte: the same 5,641,653-byte file as on 2026-10-27), evaluated again on the same test file to get per-token scores (`pos_tagger eval MODEL FILE --scores OUT`, new; without `--scores` the output is unchanged). Its temperature was fitted on dev the same way. Tokens decided by its tag dictionary count with confidence 1.

**The Wikipedia prose sample** (for the OOD read only; no gold tags): 300 sentences from enwik8 bytes 90,000,000–95,000,000, markup lines dropped, link labels kept, seeded order (`docs/reports/e1_cypha_tagger/wiki_sents.py`). The calibration comparison is `docs/reports/e1_cypha_tagger/calib.py`.

## Decisions (by the registered rules)

- **Rule 3: the gate is not met** (92.06% < 94%).
- **Rule 4:** test is above 90%, so PLAN §11's fallback is not formally triggered. But the perceptron is better on every read, so it keeps syntax: nothing that uses the tagger changes.
- **Rule 5:** Cypha's confidence is not useful (95.55% < 98.05% at 90% coverage); the OOD flag is not useful (100% on test).
- The tool stays in the tree as the measured Cypha-side tagger, with its CTest, so the next attempt (a non-linear decision such as class-conditional covariances, or a mixture of class components, or an OOD threshold set from training) starts from these numbers.

## Predictions, scored

| # | Prediction | Result |
|---|---|---|
| P1 | Dev UPOS of the dev choice 92.0–95.0 | **Miss, by 0.04:** 91.96 |
| P2 | Test UPOS 93.0–94.5; the gate met or missed narrowly | **Miss:** 92.06, missed by 1.9 points |
| P3 | Cypha below the perceptron by 0.5–2.0 points | **Miss:** 2.75 points below |
| P4 | Counts alone 88–92% on dev; the classifier adds at least 2 points | **Hit:** 89.53% (α 0.5), 88.19% (α 2); the classifier adds 2.43 |
| P5 | Exact class means within 1 point of online Cypha on dev | **Hit:** 0.49 (round 1), 0.37 (round 2) |
| P6 | Unknown-word accuracy at least the perceptron's (79.2%) | **Miss:** 71.07% |
| P7 | Native ECE above 0.05; calibrated under 0.02 | **Partial:** native 0.373 (hit); calibrated 0.0415 (miss) |
| P8 | The OOD flag fires more on random letters than on test, meeting rule 5 only in part | **Miss:** it fires on 100% of both; no separation |
| P9 | Training under 10 minutes; tagging at least 5,000 tokens/s | **Hit:** 252 s; 5,135 tokens/s (on a machine shared with about twenty jobs) |
| A1 | Whitened exact class means 91–94% on dev | **Hit:** 92.33% |
| A2 | Cypha on the whitened input within 1.5 points of A1 | **Hit:** 0.37 below |
| A3 | Round 2 beats round 1 on dev by at least 4 points | **Hit:** +8.2 |
| A4 | The OOD flag still fires on more than 20% of in-domain tokens | **Hit:** 82–100% |

8 hits, 1 partial, 5 misses (P1, P2, P3, P6, P8).

## E2 (entities): not started here

E2 needs, before its own registration: silver entity labels from Wikipedia links typed through Wikidata (PLAN Track E: PERSON, PLACE, ORG, DATE, NUMBER; F2's typed seeds and S15's Wikidata cache can type the links), a BIO chunker over the perceptron's tags (the NP chunker's spans are a start), and a gate (≥ 80% F1 on the silver test). Given this item's finding, E2's decisions should use the perceptron-style learner; Cypha's classifier can be tried as a second arm once its OOD threshold is fixed.

## How to reproduce

```bash
# data: UD_English-EWT v2.18 (2026-05-15), CC BY-SA 4.0
U=https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master
for s in train dev test; do curl -sSO $U/en_ewt-ud-$s.conllu; done
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target cypha_tagger -j4
build/cypha_tagger selftest
# round 2's dev choice and the one test read
build/cypha_tagger run --train en_ewt-ud-train.conllu --dev en_ewt-ud-dev.conllu --test en_ewt-ud-test.conllu --read-test \
  --alpha 0.5 --vec simvec --simvec data/cyphalm/simvec_ppmi32.bin --whiten 0.1 --delta-lr 0.03 --world-lr 0.008 --passes 2 --diag --out final
# the perceptron's per-token scores
g++ -O2 -std=c++20 native/tools/pos_tagger.cpp -o pos_tagger
./pos_tagger train en_ewt-ud-train.conllu en_ewt-ud-dev.conllu model.bin
./pos_tagger eval model.bin en_ewt-ud-test.conllu --scores test_scores.tsv
```

Without `--whiten`, the tool runs round 1's design. The round-1 and round-2 grids are `--whiten 0` / `--whiten 0.01|0.1` over `--delta-lr {0.003, 0.01, 0.03}` × `--world-lr {0.002, 0.008}` × `--passes 3`.
