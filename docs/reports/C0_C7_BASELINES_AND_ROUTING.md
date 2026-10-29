# C0 and C7: fair topic baselines, and PPMI-SVD vectors in Branch A routing

**Odin Loch** · 2026-10-29 · PLAN C0, C7 and T-topic · follows [C1_PPMI_SVD_VECTORS.md](C1_PPMI_SVD_VECTORS.md) and [E4_QUESTION_CLASSIFIER.md](E4_QUESTION_CLASSIFIER.md)

(The design, the rules and the predictions below were written and committed before any model run. Results are added under them.)

**Question.** Core Cypha's text routing scored 16–19% on 20 Newsgroups with hashed features, and 34.0% with TF-IDF/SVD. That 34% ran with `W_enc` learning, the setting that cost 13 points with MiniLM (49.3% against 62.5% frozen). So TF-IDF never had a fair test (C0). And C1's PPMI-SVD word vectors, built from counts alone, have not been tried in the router (C7). With `W_enc` frozen, what do count-based features give the Cypha classifier, against plain TF-IDF with logistic regression and against the hashing router? Do PPMI-SVD vectors reach PLAN's T-topic gate, 55% on 20 Newsgroups at 2k?

**Rule 10.** This is topic routing inside Core Cypha. Nothing here answers a question or reaches `qa` or `chat`. No neural network runs: MiniLM's 62.5% is quoted only as the existing external yardstick.

## Data (fixed before any run)

| Set | Source | Split | Classes |
|---|---|---|---|
| 20 Newsgroups | `20news-bydate.tar.gz` from qwone.com (sha256 `8f1b2514…d95610`), the standard by-date split | train 11,314, test 7,532 | 20 |
| AG News | Hugging Face `fancyzhx/ag_news`, revision `eb185aa`, parquet; the card says licence "unknown" (research use, as PLAN's data table says) | train 120,000, test 7,600 | 4 |

- `scripts/topic_data.py` writes one `label<TAB>text` file per split. The label is the class index (20 Newsgroups: sorted folder order; AG News: World, Sports, Business, Sci/Tech).
- **Two 20 Newsgroups texts.** `full` is the whole message with its headers, as scikit-learn's default gives it (the likely legacy protocol). `strip` removes headers, quoting and the signature, the way scikit-learn's `remove=("headers", "footers", "quotes")` does. `full` is primary; `strip` is reported only.
- File hashes (sha256 of the TSV files): 20ng train full `f51bd80e…`, test full `8ffe0cf7…`, train strip `dda5871e…`, test strip `52d46b62…`; AG train `65eb1b73…`, test `279ee4be…`. AG parquet: train `fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee`, test `71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb`.
- **Word vectors.** C1's PPMI-SVD vectors on all of enwik8 (50,000 words × 300, float32, unit rows; `enwik8.vectors.f32` sha256 `07bad8c6…92171`, `enwik8.vocab.tsv` with counts, sha256 `e6803354…6c74`). The topic data is not in enwik8, so no evaluation text leaks into them.

## Protocols (fixed before any run)

- **2k (PLAN's T-topic protocol).** 2,000 training documents drawn stratified from the training split, seed 20261029 (20 Newsgroups: 100 a class; AG News: 500 a class). 400 of them, stratified, are dev; every choice is made by training on the other 1,600 and scoring dev. Then the chosen setting is refitted on all 2,000 and scored **once** on the whole standard test split.
- **full.** All training documents; 10% stratified held out as dev for choices; refit on all; test once. Reported only.
- **Not the legacy split.** The legacy numbers (16–19%, 34.0%, 62.5%) scored 400 documents from a random split of the same 2,000. The by-date test split is later in time and usually harder than a random split. So the legacy numbers are a guide, not a like-for-like baseline.

## Features (all from counts)

| Id | Feature | Detail |
|---|---|---|
| H | Branch A's hashing router | Its own featuriser (`branch_a_router.cpp`): lower case, white-space split, unigrams and bigrams, MurmurHash3 seed 0, 512 buckets, counts, L2; then PCA to 128 on the training rows (the router's `hashing_svd`) |
| T | TF-IDF, sparse | Lower case; tokens are runs of letters and digits; unigrams and bigrams hashed (MurmurHash3) into 2^18 buckets; tf = 1 + ln(count); idf = ln((1 + N) / (1 + df)) + 1; L2 |
| Td | TF-IDF, dense for Cypha | T through a fixed signed random projection to 1,024 (4 hashed targets a feature, as E4), then z-scored and PCA-whitened on the training rows to k dimensions, k ∈ {128, 256} on dev, shrinkage ε = 0.1 |
| S | SIF of PPMI-SVD vectors | Mean of the in-vocabulary words' vectors weighted a / (a + p(w)), a = 10⁻³, p from the vocabulary counts; the first principal component of the training rows removed (Arora et al. 2017). For Cypha: z-scored and PCA-whitened to 300 (ε = 0.1) |
| S+Td | Both | S and Td concatenated, then whitened together to k ∈ {256, 384} on dev |

## Models

- **Cypha, `W_enc` frozen.** `cypha::Cypha` from `create_fresh_model_root` (identity `W_enc`, `enc_lr` = 0), GH on, field dimension 16, OOD σ 12; online `update`, `predict`. Grid on dev: `delta_lr` ∈ {0.003, 0.01, 0.03} × `world_lr` ∈ {0.002, 0.008}; passes 1–6 (the best pass on dev). Inputs H, Td, S, S+Td. This is E4's recipe.
- **Cypha, `W_enc` learning.** The same with `enc_lr` = 0.002 (the default), on Td and S: the legacy setting.
- **Logistic regression.** Multinomial, on sparse T (C0's "plain TF-IDF + logistic regression") and on S (z-scored); Adagrad; L2 λ ∈ {0, 10⁻⁶, 10⁻⁵}; epochs 1–10 on dev.
- **Averaged perceptron** on sparse T (E4's), epochs on dev.

**Measures.** Accuracy (primary) with a Wilson 95% interval; macro-F1; for Cypha, its confidence's ECE (15 bins) and Brier score. Differences: paired Bayesian bootstrap over test documents (4,000 Dirichlet draws).

## Decision rules (fixed before any run)

1. **Choices.** Every hyperparameter and k by dev accuracy; ties go to the earlier grid item. The C7 arm's input is chosen on dev between S and S+Td (both are "PPMI-SVD vectors in place of the hashing embedder").
2. **The T-topic gate (C7).** The C7 arm (Cypha, `W_enc` frozen, the dev-chosen PPMI-SVD input) on 20 Newsgroups `full`, 2k, test: accuracy ≥ 0.55 meets the gate; ≥ 0.60 meets the stretch. Reported with its Wilson interval.
3. **C7 into Branch A.** LCI (PLAN §8.2) of the C7 arm against the hashing router arm H (both Cypha, frozen) on the same test:
   - P_gain: P(Δ accuracy > 0.005) over the 4,000 draws;
   - n_eff: test documents (7,532);
   - r: canary AG News 2k, P(Δ > −0.01) ≥ 0.95;
   - γ = min(1, (test gain / dev gain) / 0.5), clipped to [0, 1];
   - c: 1 if ECE rises by at most 0.01 and Brier does not rise, else 0.5;
   - σ: three data orders (seeds 1, 2, 3) of the refit give gains of the same sign;
   - κ: featurise and classify within 5 ms a document, peak memory within 2 GB.
   If LCI ≥ 0.95, Branch A's router gains a `ppmi_sif` embedding backend, and hashing stays the fallback when no vector file is configured.
4. **The fair-test question (C0).** Cypha frozen against learning on Td, and logistic regression on T, are reported as baselines. No decision rides on them.
5. **Reads.** The test split is read once per dataset, protocol and arm. `strip` and the full protocols are secondary, reported only. Nothing is changed after a test read.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| P1 | The hashing router arm (H, Cypha frozen) at 20 Newsgroups 2k: 15–25% (legacy 16–19%) |
| P2 | Cypha frozen on Td at 20 Newsgroups 2k: 40–55% (legacy 34.0% with `W_enc` learning) |
| P3 | Cypha with `W_enc` learning on Td is at least 5 points below frozen |
| P4 | Plain TF-IDF + logistic regression at 20 Newsgroups 2k: 55–68% |
| P5 | The C7 arm at 20 Newsgroups 2k: 48–58%; the 55% gate is **not** met (point estimate below 0.55) |
| P6 | Logistic regression on S is within 3 points of Cypha frozen on S |
| P7 | AG News 2k: the C7 arm 78–86%; logistic regression on T 82–88% |
| P8 | 20 Newsgroups full protocol: logistic regression on T 75–83%; the C7 arm 60–72% |
| P9 | `strip` lowers every arm by 5–15 points against `full` |
| P10 | Cypha frozen classifies a featurised document in under 1 ms |

## Results on dev (2026-10-29, before any test read)

Tool: `topic_bench` at commit `fdcd5ce`, and the same computation after `ede6846` (the featuriser moved into `text_features.hpp`; dev lines reproduce to the last digit, checked on H, S and S+Td). Each 2k pool trains on 1,600 documents and scores 400 dev documents. One run per arm, data-order seed 1. Accuracy is the best over the grid and passes.

| Arm | 20 Newsgroups `full` | AG News | 20 Newsgroups `strip` |
|---|---:|---:|---:|
| H: hashing router, Cypha frozen | 0.283 | 0.528 | 0.183 |
| Td: TF-IDF compressed, Cypha frozen | 0.373 | 0.563 | — |
| Td, Cypha with `W_enc` learning | 0.348 | — | — |
| S: SIF of PPMI-SVD, Cypha frozen | 0.643 | 0.805 | 0.540 |
| S, Cypha with `W_enc` learning | 0.618 | — | — |
| S+Td, Cypha frozen | **0.685** | 0.835 | 0.513 |
| Logistic regression on S | 0.623 | 0.810 | — |
| Logistic regression on T | 0.775 | 0.850 | 0.538 |
| Averaged perceptron on T | 0.733 | 0.805 | — |

- **The C7 arm is S+Td** (rule 1: 0.685 against S's 0.643 on 20 Newsgroups `full`). Its setting: `world_lr` 0.008, `delta_lr` 0.01, k = 384, 2 passes.
- **Learning `W_enc` costs points** with count features too: −2.5 on S, −2.5 on Td (dev; one seed).
- **TF-IDF compressed for Cypha is weak** (0.373), while the same TF-IDF under logistic regression is the strongest arm (0.775). The loss is in the compression: 2^18 sparse features through 1,024 random projections and a 256-dimensional whitening.
- **Dev is 400 documents:** one document is 0.25 points, and the 95% half-width near 0.65 is about 4.7 points.
- **Cost on dev.** One PCA fit on the 1,024-dimensional projection took 394–482 s on the shared machine (load around 30); the decomposition is cached by the matrix's bytes, so each later fit of the same rows reads it back (numbers identical).

**Next, the test reads** (registered above): every arm once on its test split, seed 1; for the LCI, the C7 arm and H refitted with data orders 2 and 3 as well (`--refit-seed`); the AG News canary; then the router exported and its test predictions compared with the measured ones.

## Results on test (2026-10-29, each arm read once)

Tool: `topic_bench` at `d1f9a1c`, with `TOPIC_BENCH_CACHE` (numbers identical with and without the cache). Every arm repeats its dev search inside the run (the same choice as above), refits on all 2,000 pool documents, and scores the whole standard test split once. Accuracy with Wilson 95% intervals. ECE and Brier come from the softmax of each model's scores. Time is featurising plus classifying one test document, on the shared machine (load 20–35).

**20 Newsgroups `full`, 2k (7,532 test documents)**

| Arm | Accuracy | Macro-F1 | ECE | Brier | ms / doc |
|---|---|---:|---:|---:|---:|
| H: hashing router, Cypha frozen | 0.203 [0.194, 0.212] | 0.183 | 0.031 | 0.900 | 0.5 |
| Td: TF-IDF compressed, Cypha frozen | 0.285 [0.275, 0.295] | 0.274 | 0.086 | 0.853 | 1.8 |
| Td, `W_enc` learning | 0.244 [0.234, 0.253] | 0.237 | 0.079 | 0.878 | 2.1 |
| S: SIF of PPMI-SVD, Cypha frozen | 0.602 [0.591, 0.613] | 0.594 | 0.406 | 0.755 | 1.0 |
| S, `W_enc` learning | 0.556 [0.545, 0.568] | 0.548 | 0.300 | 0.716 | 1.2 |
| **S+Td: the C7 arm, Cypha frozen** | **0.613 [0.602, 0.624]** | 0.609 | 0.144 | 0.560 | 3.9 |
| Logistic regression on S | 0.576 [0.564, 0.587] | 0.569 | 0.375 | 0.783 | 0.7 |
| Averaged perceptron on T | 0.672 [0.661, 0.682] | 0.668 | 0.612 | 0.933 | 0.9 |
| **Logistic regression on T** | **0.718 [0.708, 0.728]** | 0.713 | 0.162 | 0.438 | 1.0 |

**AG News, 2k (7,600 test documents)**

| Arm | Accuracy | Macro-F1 | ECE | Brier |
|---|---|---:|---:|---:|
| H | 0.521 [0.509, 0.532] | 0.517 | 0.097 | 0.620 |
| Td | 0.539 [0.528, 0.551] | 0.537 | 0.105 | 0.609 |
| S | 0.828 [0.819, 0.836] | 0.826 | 0.223 | 0.340 |
| S+Td (C7) | 0.829 [0.820, 0.837] | 0.829 | 0.241 | 0.347 |
| Logistic regression on S | 0.806 [0.797, 0.815] | 0.805 | 0.177 | 0.366 |
| Averaged perceptron on T | 0.801 [0.792, 0.810] | 0.801 | 0.494 | 0.656 |
| Logistic regression on T | 0.853 [0.845, 0.861] | 0.852 | 0.066 | 0.233 |

**20 Newsgroups `strip`, 2k (secondary)**: H 0.152 [0.144, 0.160]; S 0.499 [0.488, 0.510]; S+Td 0.484 [0.473, 0.496]; logistic regression on T 0.524 [0.513, 0.535].

**20 Newsgroups `full`, full protocol (secondary; 11,314 training documents, 10% dev for choices)**: H 0.232 [0.223, 0.242]; logistic regression on T **0.835 [0.826, 0.843]**; the C7 arm 0.608 [0.597, 0.619] (k = 256, 6 passes; ECE 0.185; 1.5 ms a document; 260 MB peak). With 5.7 times the data, logistic regression gains 11.7 points and Cypha's online classifier gains nothing (0.613 at 2k). AG News's full protocol (120,000 documents) was not run: it is secondary, and the shared machine could not fit it in.

### The gate and the LCI

- **The T-topic gate is met, and so is the stretch.** The C7 arm scores 0.613 [0.602, 0.624] on 20 Newsgroups at 2k: above 0.55, and above 0.60 at the interval's lower bound. Counts alone take Core Cypha from 20% to 61%. MiniLM's legacy 62.5% was on a different, random split, so the two are close but not like for like.
- **The LCI of C7 against the hashing router is 0.5, so C7 is not adopted into Branch A (rule 3).**

| Component | Value |
|---|---|
| P_gain | 1.000: Δ +0.411 [0.397, 0.424] |
| n_eff | 7,532 |
| r (AG News canary) | 1: Δ +0.308 [0.296, 0.322], P(Δ > −0.01) = 1.000 |
| γ | 1: test gain 0.411 against dev gain 0.403 |
| c | **0.5**: ECE rises 0.031 → 0.144 (more than 0.01), while Brier falls 0.900 → 0.560 |
| σ | 1: Δ +0.411, +0.401, +0.409 under refit data orders 1, 2, 3 |
| κ | 1: 2.9 ms a document through the router (3.9 ms in the bench under load); 202 MB peak |
| **LCI** | **0.5** |

- **Why c fails.** The hashing arm's confidence is near uniform over 20 classes, so its ECE is small by default. The C7 arm is right three times as often, and its Brier score is far better, but its softmax is overconfident. The registered rule counts any ECE rise above 0.01. A dev-fitted temperature for Cypha's confidence (as E4 did) would be a new change with its own registered read.

### What else the reads show

- **Plain TF-IDF with logistic regression is the strongest arm**: 0.718 on 20 Newsgroups (0.835 with all the training data), 0.853 on AG News. The C7 arm trails it by 10.5 points on 20 Newsgroups (Δ −0.105 [−0.117, −0.093]) and by 2.4 on AG News. Cypha's input must be dense and as wide as its latent, and compressing TF-IDF to 256 dimensions costs it most of its value: Cypha on Td scores 0.285.
- **`W_enc` learning costs points with count features too:** −4.1 [−5.0, −3.3] on Td, −4.5 [−5.3, −3.8] on S. Frozen is right, as with MiniLM.
- **Cypha beats logistic regression on the same dense S input** by 2.6 points (0.602 against 0.576; Δ [1.6, 3.7]), as it did on MiniLM vectors (62.5% against 60.3%).
- **S+Td against S alone:** +1.1 [+0.2, +2.1] points on 20 Newsgroups; level on AG News; −1.5 on `strip`.
- **Without headers** every arm loses: S+Td −12.9 points, S −10.3, logistic regression on T −19.4, H −5.0. The headers (Subject, Organization) carry much of the topic.

### Branch A: the router backend, built and checked, then reverted by rule 3

- **Built and checked** (commits `ede6846`, `d1f9a1c`): a `count_vectors` backend for `BranchARouter` on the shared featuriser `native/include/cypha/text_features.hpp`, routing with `Cypha::predict` as measured, plus `topic_bench export-router` and `route-eval`.
  - The exported C7 router reproduced the measured test predictions **document for document: 0 of 7,532 differ**, accuracy 0.6131.
  - Routing took 2.9 ms a document at a peak of 97 MB.
  - At the router's default epistemic threshold (entropy > 0.5 nats) it abstains on 92% of documents with 20 classes, so that threshold would need setting for 20-way routing.
- **Reverted** by rule 3 (LCI 0.5). The router is unchanged from main. `text_features.hpp` stays, because `topic_bench` uses it.
- **A core finding on the way.** `Cypha::save` and `Cypha::load` do not round-trip the state `Cypha::predict` reads:
  - `save` writes the root's stale `mahal_ema` (0), not the trained one;
  - `load` resets the GH gate's session state, which `save` did write;
  - the inference model's online statistics are not saved at all: the tiered label context (`t1_counts`, `cooccur`, `mid_trans`, `ctx_last_label`), `llr_scale_*` and `field_step`.
  - In a two-class toy, a saved and reloaded model gave a different label on 45 of 100 inputs; with this state restored, 0.
  - The backend restored it from a sidecar (`online_state.json`). The proper fix belongs in `cypha.cpp`, outside this item.
- **A second finding.** The tiered label-context prior learns the order of the training labels. In the toy, where training alternated labels, it outweighed the input. For routing independent queries that prior is noise, and `topic_bench` shuffles the training order for this reason.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| P1 | H at 20 Newsgroups 2k: 15–25% | **Hit:** 20.3% |
| P2 | Cypha frozen on Td: 40–55% | **Miss:** 28.5% (the 256-dimensional compression loses most of TF-IDF) |
| P3 | `W_enc` learning on Td at least 5 points below frozen | **Miss, narrowly:** −4.1 [−5.0, −3.3] |
| P4 | TF-IDF + logistic regression: 55–68% | **Miss:** 71.8% (above) |
| P5 | The C7 arm 48–58%; gate not met | **Miss:** 61.3%; gate and stretch met |
| P6 | Logistic regression on S within 3 points of Cypha frozen on S | **Hit:** −2.6 |
| P7 | AG News: C7 78–86%; logistic regression on T 82–88% | **Hit:** 82.9%, 85.3% |
| P8 | Full protocol: logistic regression on T 75–83%; C7 60–72% | **Partial:** logistic regression 83.5% (just above); C7 60.8% (inside) |
| P9 | `strip` lowers every arm by 5–15 points | **Partial:** H −5.0, S −10.3, S+Td −12.9 inside; logistic regression on T −19.4 outside |
| P10 | Cypha classifies a featurised document in under 1 ms | **Not scored as registered:** the bench timed featurising and classifying together (0.4–3.9 ms) |

### Decisions

- **C0 is closed.** The fair baselines are recorded: TF-IDF with `W_enc` frozen (Cypha on Td 0.285; the legacy 34% was a different split and protocol), plain TF-IDF with logistic regression 0.718, the hashing router 0.203, MiniLM's 62.5% as the external yardstick.
- **The C7 / T-topic gate is met:** Core Cypha with PPMI-SVD vectors from counts alone scores 0.613 on 20 Newsgroups at 2k.
- **C7 is not adopted into Branch A:** its LCI is 0.5, because of the calibration term. The router stays on hashing until a registered change fixes the confidence (or Odin rules on the c term against a near-uniform baseline).
- **For Odin:**
  - a dev-fitted temperature for Cypha's confidence as the next registered change for Branch A;
  - the router's abstain threshold for 20 classes;
  - `Cypha::save` / `load` not round-tripping the online state `predict` reads (a core fix in `cypha.cpp`);
  - whether the tiered label-context prior should be off for routing independent queries;
  - Cypha's online learner does not gain from more data here, while logistic regression gains 11.7 points.

## How to reproduce

```bash
# data (each into its own directory), then label/text files
curl -LO http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz && tar -xzf 20news-bydate.tar.gz   # sha256 8f1b2514…d95610
# AG News parquet: huggingface.co/datasets/fancyzhx/ag_news, revision eb185aade064a813bc0b7f42de02595523103ca4
python3 -I scripts/topic_data.py 20ng DIR_WITH_BYDATE tsv
python3 -I scripts/topic_data.py ag DIR_WITH_PARQUET tsv
# tool
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target topic_bench
export TOPIC_BENCH_CACHE=$PWD/eigcache   # optional; identical numbers, faster refits
V="--vectors enwik8.vectors.f32 --vocab enwik8.vocab.tsv"   # C1's PPMI-SVD (ppmi_svd.cpp on enwik8)
build/topic_bench run --train tsv/20ng_train_full.tsv --arm cypha_STd --protocol 2k $V                 # dev only
build/topic_bench run --train tsv/20ng_train_full.tsv --test tsv/20ng_test_full.tsv --arm cypha_STd \
  --protocol 2k $V --read-test --out ng_2k_cypha_STd                                                  # one test read
build/topic_bench compare ng_2k_cypha_STd.preds ng_2k_cypha_H.preds --eps 0.005 --draws 4000          # LCI's P_gain
```

Arms: `cypha_H`, `cypha_Td`, `cypha_S`, `cypha_STd`, `cypha_learn_Td`, `cypha_learn_S`, `lr_T`, `lr_S`, `perc_T`. CTest `native_topic_bench_selftest` (17 checks; four planted bugs each fail it: no L2 normalisation, a wrong dev split, a flipped gradient sign, a wrong Brier score).
