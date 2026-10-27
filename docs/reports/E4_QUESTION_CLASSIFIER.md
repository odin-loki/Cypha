# E4: question classifier on TREC, and a test of whether Cypha's classifier can learn language from counts

> **Code removed 2026-10-27** to cut code debt. The finding stands: the averaged perceptron beat the Cypha classifier on TREC, which confirms the PLAN §11 risk. The tool is in git at commit `707629a` (`native/tools/question_classifier.cpp`) if the benchmark has to be rerun. Re-testing Cypha's text learning is part of the tool-profiling plan ([TOOL_PROFILE_PLAN.md](../TOOL_PROFILE_PLAN.md)).

Odin Loch, 2026-10-27

Tool: `native/tools/question_classifier.cpp` (C++20, links `cypha_core`; no other library).
Plan items: Track E row E4 (gate: at least 90% on answer type); the first row of section 11
(Cypha DIF on text features); rule 9 (abstaining is a feature).

## Summary

**Cypha's classifier does not learn this language task well enough from count-based features.
An averaged perceptron on the same questions does.**

| model | grain | set (by dev) | dev acc | test acc | macro-F1 | ECE native | ECE calibrated | acc @90% cov | acc @80% cov |
|---|---|---|---|---|---|---|---|---|---|
| averaged perceptron | coarse | a | 86.2 | **89.0** | 88.8 | 0.365 | 0.029 | 93.8 | 97.2 |
| averaged perceptron | fine | abc | 79.8 | **84.6** | 72.3 | 0.209 | 0.050 | 90.0 | 93.5 |
| batch LDA (diagnostic) | coarse | bc | 75.0 | **84.8** | 84.7 | 0.132 | 0.103 | 88.7 | 91.5 |
| batch LDA (diagnostic) | fine | abc | 65.3 | **74.4** | 63.6 | 0.116 | 0.170 | 80.9 | 86.8 |
| Cypha | coarse | bc | 70.5 | **75.0** | 63.6 | 0.436 | 0.103 | 79.6 | 84.0 |
| Cypha | fine | bc | 59.1 | **62.4** | 36.7 | 0.412 | 0.087 | 68.0 | 73.8 |

Test accuracy in %, 500 TREC test questions. The feature set for each row was chosen on a 545-question
dev split cut from the training data; the test set was scored once per configuration.
"Calibrated" means a softmax temperature fitted on the dev split. Both Cypha rows use `cypha::Cypha`
with `W_enc` frozen (`enc_lr = 0`), trained online. The batch LDA rows are a diagnostic (the same whitened
input as Cypha, exact class means), not a candidate model.

- **Coarse (6 types), the E4 gate.** The perceptron reaches 89.0% with the dev-selected set and 91.6% with the
  full concatenation (`abc`, not the dev choice). Cypha reaches 75.0%, 14 points lower, and misses the
  90% gate by 15. The perceptron sits at the gate within noise (one standard error on 500 questions is
  about 1.4 points); Cypha does not.
- **Fine (50 types).** Perceptron 84.6%, Cypha 62.4% (67.0% in a rerun with a wider grid, see
  "Robustness"), a gap of 17 to 22 points.
- **Selective accuracy (rule 9).** Abstaining on the lowest calibrated confidence, the perceptron answers 90% of coarse
  questions at 93.8% accuracy and 80% at 97.2%. Cypha gets 79.6% and 84.0% at the same coverages. Cypha's
  own confidence is far too low before temperature scaling (ECE 0.18 to 0.44; mean confidence 0.29 for a
  coarse model that is 75% accurate). Its OOD flag fired
  on none of the 500 test questions in any run, and its anomaly score carries no information about errors
  (AUROC 0.44 to 0.61).
- **The risk in section 11 is confirmed for this task**, and it extends from sparse features to dense
  count-based vectors. The fallback written in section 11 applies: the averaged perceptron makes the
  discrete language decision. The second half of that fallback ("Cypha keeps confidence, OOD and routing") is
  not supported by this task: a temperature-scaled perceptron ranks its errors better than Cypha does.

Reference points, stated as approximate. Li and Roth (COLING 2002), trained on 21,500 questions and tested
on 1,000 TREC questions, report 85.1% coarse and 82.6% fine with words only, and 92.5% and 85.0% with
syntactic features (checked against the paper). SVM bag-of-words on the 5,500-question set is reported near
86 to 87% coarse (Zhang and Lee 2003; seen only through a third-party summary, not verified).
The perceptron here is at or slightly above the bag-of-words figures (89.0% with hashed n-grams alone; it
has bigrams and the cue features) and just under the paper's syntactic-feature figure; the 90% roughly
quoted for SNoW-era systems matches that figure only with syntactic features, which this tool does not use.

## Data

| item | value |
|---|---|
| Source | `https://raw.githubusercontent.com/xuanduc24905-beep/Trec_classification/main/data/train_5500.label.txt` and `.../TREC_10.label.txt` (a GitHub copy of the Li and Roth files; found with web search because the UIUC and Hugging Face hosts are blocked here) |
| Training questions | **5,452** (the file is called "5500" after the nominal size) |
| Test questions | **500** (TREC-10) |
| Types | 6 coarse and 50 fine in training; the test set uses all 6 coarse and 42 of the 50 fine, all of which occur in training |
| Coarse counts, train | ENTY 1250, HUM 1223, DESC 1162, NUM 896, LOC 835, ABBR 86 |
| Coarse counts, test | DESC 138, NUM 113, ENTY 94, LOC 81, HUM 65, ABBR 9 (majority-class accuracy 27.6%) |
| Checks | the 5,452 question texts are identical, in the same order, to the `harvardnlp/sent-conv-torch` copy of `TREC.train.all` (numeric coarse labels only), and the coarse label maps one to one onto its numbers. 10 test questions also occur verbatim in training. Files are ISO-8859-1. |

Licence line for the manifest (rule 8): the data come from the UIUC Cognitive Computation Group
(Li and Roth 2002) for research use; they are not part of any shipped pack. I did not check whether the GitHub copy
carries a licence of its own; confirm the UIUC terms before any redistribution.

## Features (counts only, no neural network)

| set | what it is |
|---|---|
| `a` | Hashed word unigrams and bigrams (2^17 buckets), binary term frequency, smooth IDF from the training rows, L2-normalised. |
| `b` | Question cues, hashed one-hots (2^14 buckets), L2-normalised: wh-word (first of what, which, who, whom, whose, when, where, why, how in the first four tokens, or an imperative such as name), the next one and two words, the pairs wh+next and wh+next+head, the first noun-like word (first token after the wh-word that is not an auxiliary, determiner, preposition, pronoun or quantifier such as "many"), whether that word is capitalised, the count of capitalised words, a length bucket, the first word, and a number flag. |
| `c` | Dense: SIF average (Arora et al. 2017) of the C1 PPMI-SVD vectors (enwik8, 300 d, WordSim-353 0.672), weight a/(a+p(w)) with a = 1e-3, first principal component removed, each dimension z-scored on the training rows. Tokens missing from the 50,000-word vocabulary are skipped. |
| `ab`, `bc`, `abc` | Concatenations. |

Featurising one question takes 12 to 22 microseconds; all 11,400 questions (both views) take about 0.7 s.

## Models

**Averaged perceptron** (Collins 2002), multiclass, written in the tool. It scores each class by a
dot product with the sparse or dense concatenation (blocks scaled to about unit norm), updates on mistakes,
and keeps the averaged weights by the lazy time-stamp trick. Epochs (up to 20) are chosen on dev.
Its confidence is the softmax of the scores.

**Cypha classifier** (`cypha::Cypha`). A fresh model from `create_fresh_model_root` (identity `W_enc`,
field dimension 16), loaded through `Cypha::load`, with `train_params().enc_lr = 0` so `W_enc` stays
frozen as Branch A recommends. Training is `Cypha::update` one question at a time, shuffled each pass, for
1 to 5 passes. Scoring is `Cypha::predict`. Both use the GH path (`use_gh = true`), which is the library
default and the only path that sets the OOD flag.

Cypha needs a dense input as wide as its latent, so sets with sparse parts get a **compact projection**:
a fixed signed random projection (each feature goes to 4 of the 256 columns for `a`, or of 128 for `b`),
then every dimension is z-scored on the training rows and the block is whitened by PCA on the training
rows (top 256 components, shrinkage 0.1 of the mean eigenvalue). This is count-based preprocessing fitted
once on the inputs; no label touches it. Hyperparameters tried on dev: `delta_lr` in {0.001, 0.003, 0.01},
`world_lr` in {0.002, 0.008}, passes 1 to 5.

**Batch LDA** (diagnostic, not a product model). Class means and priors on exactly the whitened dense input
Cypha sees, with unit covariance. Cypha's score is a Gaussian log-likelihood ratio with a shared diagonal
variance, so this is the ceiling of its scoring rule when the online prototype updates are replaced by exact
class means. It separates the cost of the online updates from the cost of the model family.

## Protocol

1. 90/10 split of the 5,452 training questions with a fixed seed (4,907 fit, 545 dev).
2. On the fit part: fit the features, train, choose hyperparameters and epochs or passes on dev accuracy,
   fit the softmax temperature on dev negative log-likelihood, and read the abstain threshold for 90%
   coverage from the dev confidences.
3. Refit everything on all 5,452 questions with the chosen settings and score the 500 test questions once.
4. Coarse and fine are separate classifiers. Macro-F1 averages over classes present in the gold or
   predicted labels. ECE uses 15 equal-width bins. The risk-coverage curve ranks test questions by calibrated
   confidence and reports accuracy on the most confident 100, 90, 80, 70 and 50%. For Cypha a second ranking
   puts OOD-flagged questions last.
5. One seed. A 545-question dev split has a standard error of about 1.5 points, and a 500-question test set
   about 1.4 to 2.2 points, so differences under 3 points between sets are not meaningful.

Prediction (rule 3): not recorded in advance. A one-set pilot (set `c`, coarse) had already shown Cypha at
about 60% dev accuracy before the full grid was launched, so no honest advance range exists for this report. The next run
should write its range first (see next steps).

## Results

### Coarse (6 types), all sets

| model | set | dev acc | test acc | macro-F1 | ECE native | ECE calibrated | AUROC conf vs error | acc @90 | acc @80 | AURC | train s | infer us/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| averaged perceptron | a | 86.2 | 89.0 | 88.8 | 0.365 | 0.029 | 0.91 | 93.8 | 97.2 | 0.017 | 0.01 | 1 |
| averaged perceptron | ab | 86.1 | 89.6 | 89.3 | 0.166 | 0.035 | 0.95 | 95.1 | 98.2 | 0.011 | 0.02 | 1 |
| averaged perceptron | abc | 84.6 | 91.6 | 90.0 | 0.036 | 0.036 | 0.90 | 95.1 | 97.5 | 0.014 | 0.41 | 4 |
| averaged perceptron | b | 78.7 | 84.2 | 83.6 | 0.147 | 0.042 | 0.91 | 90.0 | 94.5 | 0.029 | 0.01 | 0 |
| averaged perceptron | bc | 82.8 | 88.6 | 87.4 | 0.033 | 0.041 | 0.90 | 92.9 | 96.5 | 0.020 | 0.25 | 3 |
| averaged perceptron | c | 68.3 | 77.6 | 78.0 | 0.056 | 0.081 | 0.78 | 81.6 | 84.8 | 0.095 | 0.34 | 3 |
| batch LDA (diagnostic) | a | 53.0 | 60.0 | 63.6 | 0.067 | 0.085 | 0.74 | 62.7 | 65.5 | 0.209 | 0.00 | 2 |
| batch LDA (diagnostic) | ab | 63.8 | 72.2 | 73.1 | 0.071 | 0.071 | 0.81 | 76.4 | 79.2 | 0.104 | 0.00 | 2 |
| batch LDA (diagnostic) | abc | 73.0 | 82.0 | 82.2 | 0.117 | 0.088 | 0.89 | 86.9 | 92.0 | 0.043 | 0.00 | 2 |
| batch LDA (diagnostic) | b | 63.1 | 70.4 | 71.3 | 0.097 | 0.097 | 0.82 | 74.4 | 80.5 | 0.116 | 0.00 | 1 |
| batch LDA (diagnostic) | bc | 75.0 | 84.8 | 84.7 | 0.132 | 0.103 | 0.82 | 88.7 | 91.5 | 0.054 | 0.00 | 2 |
| batch LDA (diagnostic) | c | 67.0 | 76.4 | 77.7 | 0.103 | 0.089 | 0.79 | 81.6 | 84.5 | 0.100 | 0.00 | 2 |
| Cypha | abc | 67.2 | 75.6 | 77.3 | 0.419 | 0.064 | 0.82 | 79.6 | 82.2 | 0.086 | 5.89 | 79 |
| Cypha | b | 57.4 | 60.2 | 50.6 | 0.331 | 0.076 | 0.78 | 64.7 | 68.0 | 0.194 | 1.57 | 24 |
| Cypha | bc | 70.5 | 75.0 | 63.6 | 0.436 | 0.103 | 0.83 | 79.6 | 84.0 | 0.089 | 3.73 | 79 |
| Cypha | c | 60.0 | 66.0 | 62.1 | 0.387 | 0.075 | 0.76 | 69.3 | 73.8 | 0.174 | 3.82 | 80 |

### Fine (50 types), all sets

| model | set | dev acc | test acc | macro-F1 | ECE native | ECE calibrated | AUROC conf vs error | acc @90 | acc @80 | AURC | train s | infer us/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| averaged perceptron | a | 76.3 | 78.0 | 63.9 | 0.679 | 0.047 | 0.91 | 84.2 | 90.5 | 0.056 | 0.16 | 6 |
| averaged perceptron | ab | 78.2 | 82.0 | 63.4 | 0.510 | 0.026 | 0.92 | 88.9 | 94.5 | 0.046 | 1.10 | 9 |
| averaged perceptron | abc | 79.8 | 84.6 | 72.3 | 0.209 | 0.050 | 0.88 | 90.0 | 93.5 | 0.040 | 4.31 | 33 |
| averaged perceptron | b | 68.1 | 71.8 | 46.7 | 0.413 | 0.045 | 0.94 | 78.7 | 86.8 | 0.066 | 0.10 | 4 |
| averaged perceptron | bc | 76.5 | 80.6 | 69.0 | 0.228 | 0.043 | 0.90 | 86.7 | 91.8 | 0.048 | 0.96 | 25 |
| averaged perceptron | c | 62.4 | 72.0 | 60.0 | 0.261 | 0.097 | 0.77 | 75.8 | 78.5 | 0.131 | 0.99 | 20 |
| batch LDA (diagnostic) | a | 50.1 | 57.0 | 37.7 | 0.090 | 0.150 | 0.86 | 61.3 | 67.2 | 0.180 | 0.00 | 14 |
| batch LDA (diagnostic) | ab | 58.2 | 66.2 | 45.9 | 0.095 | 0.180 | 0.89 | 71.8 | 79.2 | 0.111 | 0.00 | 14 |
| batch LDA (diagnostic) | abc | 65.3 | 74.4 | 63.6 | 0.116 | 0.170 | 0.84 | 80.9 | 86.8 | 0.099 | 0.00 | 14 |
| batch LDA (diagnostic) | b | 53.6 | 62.8 | 41.0 | 0.098 | 0.203 | 0.90 | 68.4 | 75.5 | 0.127 | 0.00 | 7 |
| batch LDA (diagnostic) | bc | 62.4 | 71.2 | 59.1 | 0.146 | 0.154 | 0.75 | 76.4 | 81.0 | 0.163 | 0.00 | 14 |
| batch LDA (diagnostic) | c | 54.7 | 62.6 | 53.2 | 0.194 | 0.257 | 0.62 | 66.7 | 70.0 | 0.304 | 0.00 | 14 |
| Cypha | abc | 56.7 | 64.0 | 36.5 | 0.389 | 0.051 | 0.87 | 69.1 | 75.2 | 0.126 | 6.87 | 110 |
| Cypha | b | 52.3 | 55.6 | 24.0 | 0.184 | 0.077 | 0.90 | 61.1 | 67.8 | 0.169 | 0.67 | 110 |
| Cypha | bc | 59.1 | 62.4 | 36.7 | 0.412 | 0.087 | 0.85 | 68.0 | 73.8 | 0.154 | 8.57 | 147 |
| Cypha | c | 52.7 | 52.8 | 25.2 | 0.359 | 0.077 | 0.78 | 56.9 | 61.0 | 0.267 | 5.32 | 216 |

`dev acc` is on the 545 held-out training questions; `test acc` is on the 500 TREC test questions.
"ECE native" is the model's own confidence (library confidence for Cypha, softmax of raw scores for the
others); "ECE calibrated" uses the dev temperature. `train s` is the final fit on 5,452 questions (Cypha: the
chosen number of passes; the grid search is not included). `infer us/q` is one question after featurising,
single thread.

Cypha settings chosen on dev:

| grain | set | chosen on dev | softmax T (dev) |
|---|---|---|---|
| coarse | abc | gh=1 world_lr=0.008 delta_lr=0.003 passes=5 | 0.50 |
| coarse | b | gh=1 world_lr=0.002 delta_lr=0.001 passes=4 | 0.27 |
| coarse | bc | gh=1 world_lr=0.008 delta_lr=0.003 passes=3 | 0.55 |
| coarse | c | gh=1 world_lr=0.008 delta_lr=0.003 passes=3 | 0.61 |
| fine | abc | gh=1 world_lr=0.002 delta_lr=0.01 passes=4 | 0.67 |
| fine | b | gh=1 world_lr=0.008 delta_lr=0.01 passes=1 | 0.74 |
| fine | bc | gh=1 world_lr=0.002 delta_lr=0.01 passes=5 | 0.67 |
| fine | c | gh=1 world_lr=0.008 delta_lr=0.01 passes=3 | 0.74 |

### Same input for all three models

Set `c` alone gives every model the same SIF input (the perceptron sees the 300 standardised dimensions,
the LDA and Cypha the 256-component whitened version of them), which removes the sparse-feature and
projection confound:

| model (input: SIF vectors only) | coarse dev | coarse test | fine dev | fine test |
|---|---|---|---|---|
| averaged perceptron | 68.3 | 77.6 | 62.4 | 72.0 |
| batch LDA (diagnostic) | 67.0 | 76.4 | 54.7 | 62.6 |
| Cypha | 60.0 | 66.0 | 52.7 | 52.8 |

On this input the perceptron and the batch LDA agree on coarse (77.6 and 76.4) and online Cypha is
10 points below both. On fine the LDA is 9 points below the perceptron (the shared covariance and the
class means are a weaker model than discriminative training for 50 classes) and Cypha is another 10 below
that.

### Risk-coverage (dev-selected sets)

Accuracy in % on the most confident fraction of the 500 test questions:

| model | grain | set | 100% | 90% | 80% | 70% | 50% | AURC |
|---|---|---|---|---|---|---|---|---|
| averaged perceptron | coarse | a | 89.0 | 93.8 | 97.2 | 99.1 | 99.2 | 0.017 |
| averaged perceptron | fine | abc | 84.6 | 90.0 | 93.5 | 96.3 | 98.4 | 0.040 |
| batch LDA (diagnostic) | coarse | bc | 84.8 | 88.7 | 91.5 | 93.7 | 97.6 | 0.054 |
| batch LDA (diagnostic) | fine | abc | 74.4 | 80.9 | 86.8 | 90.0 | 91.6 | 0.099 |
| Cypha | coarse | bc | 75.0 | 79.6 | 84.0 | 87.7 | 93.2 | 0.089 |
| Cypha | fine | bc | 62.4 | 68.0 | 73.8 | 80.6 | 88.8 | 0.154 |

Reading it: the perceptron's coarse accuracy climbs from 89.0 to 97.2 by giving up 20% of questions. Cypha
climbs from 75.0 to 84.0 over the same cut: it gains about as many points per abstained question, but starts
15 points lower and never reaches the perceptron's unabstained accuracy at 80% coverage. At 90% coverage
Cypha answers coarse questions at 79.6%, so about one answer in five would still be wrong.

### Cypha's OOD and epistemic signals

| grain | set | OOD-flagged of 500 | AUROC of -anomaly for errors | acc @90 (confidence) | acc @90 (flag first) |
|---|---|---|---|---|---|
| coarse | abc | 0 | 0.49 | 79.6 | 79.6 |
| coarse | b | 0 | 0.54 | 64.7 | 64.7 |
| coarse | bc | 0 | 0.53 | 79.6 | 79.6 |
| coarse | c | 0 | 0.61 | 69.3 | 69.3 |
| fine | abc | 0 | 0.47 | 69.1 | 69.1 |
| fine | b | 0 | 0.55 | 61.1 | 61.1 |
| fine | bc | 0 | 0.44 | 68.0 | 68.0 |
| fine | c | 0 | 0.48 | 56.9 | 56.9 |

The OOD flag (the GH anomaly score above 3) never fired on the in-domain test set, so abstaining on it
changes nothing. Its anomaly score also does not predict errors (AUROC about 0.5; 0.5 is chance), so it
cannot stand in for confidence. A five-item probe with `--ask` (below) shows the flag also stays clear for a
declarative sentence and for gibberish. This is a statement about this classifier on this task, not about
Cypha's OOD detection on the Branch A embeddings, where the Gutenberg test did separate (D09).

### Calibration

Cypha's native confidence is far from calibrated, and in the under-confident direction. A re-run of the chosen
coarse `bc` configuration gives mean confidence 0.29 for a model that is 75.4% accurate (ECE 0.46); after
the dev-fitted temperature (0.55, so the scores are sharpened) the mean is 0.67 and the ECE 0.09. Across
all runs the native ECE is 0.33 to 0.44 on coarse and 0.18 to 0.41 on fine, and one dev temperature brings it
to 0.05 to 0.10, in line with the perceptron after the same treatment (0.03 to 0.10). The perceptron's raw
softmax is better placed on the larger sets (native ECE 0.03 to 0.04 on `abc` and `bc` coarse) and off by a
scale on the sparse ones, which the temperature also fixes. The scale does not change the ranking, and the
ranking is what abstention uses. Error-ranking quality differs, as the AUROC columns show: coarse 0.76 to 0.83
for Cypha against 0.78 to 0.95 for the perceptron.

### Time

On a shared 2-core VM, one thread each:

| | train (final fit) | inference per question |
|---|---|---|
| averaged perceptron | 0.01 s (set `a`) to 4.3 s (`abc`, fine) | under 1 to 33 microseconds |
| Cypha (online, 1 to 5 passes) | 0.7 to 8.6 s | 24 to 216 microseconds |
| featurising one question | 12 to 22 microseconds | |

Cypha's per-question cost grows with the number of classes (24 to 80 microseconds at 6 classes, 110 to 216
at 50). All of it is far inside the plan's budgets (section 10), so speed is not the issue. The real training
cost is the dev grid (6 configurations, each a full training on 4,907 questions): 12 to 50 s per set for coarse
and 19 to 65 s for fine, plus 0.3 to 6 s to fit the whitener.

### Robustness

- **Selection noise.** Dev accuracy of an online Cypha run moves by 3 to 5 points between consecutive passes
  (for example 0.58, 0.57, 0.52, 0.59 on set `c`), so the pass picked on dev is partly luck. A rerun for fine
  with `delta_lr` in {0.01, 0.03, 0.1} (higher rates were worse on dev, so the best rate is not at the edge
  of the grid) picked a different trajectory and gave 67.0% on `bc` (dev 61.7) and 63.0% on `abc`
  (dev 58.4), against 62.4% and 64.0% in the main run. The honest Cypha fine figure is 62 to 67%.
- **Set choice.** The dev-selected perceptron coarse set (`a`, 89.0%) is 2.6 points under the best set on test
  (`abc`, 91.6%); the dev split cannot tell them apart.
- **One seed.** Not repeated. The gaps between models (14 and 17 or more points) are well outside the noise
  above, the gaps between sets are not.

### `--ask` probe

`--ask` trains the dev-selected `abc` models on all training data and prints a label with its calibrated
confidence, or ABSTAIN under the dev threshold for 90% coverage or an OOD flag. Five questions, coarse /
fine, perceptron then Cypha (one run with 4 passes, `delta_lr` 0.003, `world_lr` 0.008):

| question | perceptron coarse / fine | Cypha coarse / fine |
|---|---|---|
| Who wrote the novel Moby Dick ? | HUM 1.00 / HUM:ind | HUM 0.81 / HUM:ind |
| What is the boiling point of water in Fahrenheit ? | NUM 0.59 / ABSTAIN (NUM:temp 0.25) | LOC 0.41 (wrong) / LOC:other 0.23 |
| How many calories are in a banana ? | NUM 1.00 / NUM:count | NUM 0.82 / NUM:count |
| The quick brown fox jumps over the lazy dog . | ABSTAIN / ENTY:animal 0.65 | ENTY 0.53 / ABSTAIN |
| asdf qwer zxcv uiop | DESC 0.65 (no abstain) / ABSTAIN | DESC 0.48 (no abstain) / HUM:ind 0.24 (no abstain) |

Anecdotal (n = 5), but it shows the same thing as the test set: no OOD flag on a non-question or on gibberish,
and abstention that rests on a confidence threshold alone. The gibberish activates almost no features (only the
bias, in the perceptron), yet both models give it a coarse label above their abstain threshold.

## Interpretation and verdict on the risk

1. **Can Cypha's classifier learn this language task with these count-based features?** It learns something:
   75.0% coarse against a 27.6% majority baseline, and 62.4 to 67.0% fine. It does not learn it well enough: the
   perceptron is 14 points ahead on coarse and 17 to 22 on fine, Cypha is under the E4 gate by 15 points, and
   its macro-F1 on fine (about 37%) shows it loses most of the rare types. With a frozen `W_enc`, as Branch A
   recommends, the dense route is the only route open to it, and on dense count-based vectors alone it gets
   66.0% coarse and 52.8% fine.
2. **Where the gap comes from, in order of size, on this evidence.**
   - *The online update rule.* On identical whitened input, replacing Cypha's online prototype updates by exact
     class means (batch LDA) gains 9.8 points coarse (set `bc`: 75.0 to 84.8) and 8.8 to 10.4 fine (`bc`: 62.4 to
     71.2; `abc`: 64.0 to 74.4). The dev curves are
     consistent with this being update noise: with a fixed learning rate the class prototypes are exponential
     moving averages over roughly the last 1/`delta_lr` examples of a class, and accuracy swings by several
     points from pass to pass. The lowest rate tried in the pilot (0.0005) was both slow and unstable (29 to 55%
     on dev). I did not test the mechanism directly (iterate averaging is in next steps).
   - *The scoring family against discriminative training.* The perceptron is 4 points ahead of batch LDA on
     coarse and 10 on fine, and on dense input alone it is level on coarse (77.6 against 76.4) and 9 points
     ahead on fine.
   - *Compressing the sparse features.* Not isolated. The perceptron uses the full 2^17 hashed features, Cypha
     and the LDA see 384 random-projection dimensions, so part of the gap may belong to the projection and
     not to the model. The random projection is the weakest part of the setup (set `a` through it gives only
     60% coarse with LDA, against 89% for the perceptron on `a`). LSA would test this.
3. **What the count-based features themselves carry.** For the perceptron, the wh-word and head-word cues
   alone give 84.2% coarse; hashed n-grams alone 89.0%; adding the SIF vectors to both gives 91.6% (+2.0 over
   `ab`, inside the noise on coarse, +2.6 on fine). The PPMI-SVD vectors alone give 77.6% coarse and 72.0% fine.
   This suggests they carry what a question is about more than what it asks for, which would explain why the
   cues matter. So the count-based
   features can support question classification at about the published level, with the perceptron.
4. **Verdict on the section 11 risk.** Confirmed for question classification: Cypha DIF is not competitive on
   text features when it is fed count-based inputs, with a frozen encoder and online training. Use the fallback:
   the averaged perceptron is the question classifier for E4 v0 (and, by the same machinery, the first candidate
   for E1, E2 and E3), with a dev-fitted temperature for confidence and a dev-chosen threshold for abstaining.
   For Cypha, do not yet count on this task as evidence that it adds confidence or OOD value; the evidence here
   is that it adds neither. That does not close the question for the other Cypha roles in the plan (routing on
   sentence embeddings, Branch A; the reward-driven policy in section 3), because they were not tested.
5. **Limits of this verdict.** One task, one seed, one preprocessing design (random sketch plus PCA
   whitening), a six-point grid on three parameters, and features without syntax. Untried Cypha options: the
   Nystrom kernel LLR (`use_kernel_llr`), per-class mixtures (`use_class_gmm`), replay, `enc_lr` above zero
   (Branch A found it hurts), a larger field dimension, and iterate averaging. Any of them could narrow the
   gap, and the batch LDA row shows the headroom its scoring rule has on the same input (about 10 points).

## Build and run

The tool is not registered in `native/CMakeLists.txt` (it was being edited elsewhere). Add next to
`cypha_one_smoke`:

```cmake
add_executable(question_classifier tools/question_classifier.cpp)
target_link_libraries(question_classifier PRIVATE cypha_core)
cypha_mingw_apply_exe_link_options(question_classifier)
```

It needs data paths, so a CTest entry would have to be conditional on them.

Without CMake registration, as done here (core library in its own build directory; the `-D` flags and
include paths are those on the `cypha_core` compile line, taken from `ninja -t commands`):

```bash
cmake -S native -B /root/cb_core -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /root/cb_core --target cypha_core        # about 5 minutes on 2 cores
g++ -O3 -DNDEBUG -std=c++20 -msse4.1 -fopenmp \
    -DCYPHA_ACCEL_GPU_MIN_BATCH_ROWS=1 -DCYPHA_HP_GATE24=1 -DCYPHA_HP_SLOT_MAX=24 \
    -DCYPHA_LLM_ALGORITHM_HP=1 -DHP_SLOT_MAX=24 -DHP_XSIMD=1 \
    -Inative/third_party/hp/include -Inative/third_party/hp/third_party/xsimd/include \
    -Inative/include -Inative/third_party \
    native/tools/question_classifier.cpp /root/cb_core/libcypha_core.a -lpthread -o question_classifier
```

Runs behind the tables (vectors from C1, `enwik8.vectors.f32` and `enwik8.vocab.tsv`):

```bash
Q="question_classifier train_5500.label.txt TREC_10.label.txt enwik8.vectors.f32 enwik8.vocab.tsv"
$Q --models perceptron,ncm --sets a,b,c,ab,bc,abc --grain coarse,fine           # 1 minute
$Q --models cypha --sets b,c,bc,abc --grain coarse --delta-lr 0.001,0.003,0.01 --world-lr 0.002,0.008 --passes 5
$Q --models cypha --sets b,c,bc,abc --grain fine   --delta-lr 0.001,0.003,0.01 --world-lr 0.002,0.008 --passes 5
$Q --models cypha --sets bc,abc --grain fine --delta-lr 0.01,0.03,0.1 --world-lr 0.002,0.008 --passes 5
$Q --models perceptron,cypha --sets abc --delta-lr 0.003 --world-lr 0.008 --passes 4 --ask "Who wrote the novel Moby Dick ?|..."
```

## Next steps

1. **LSA instead of the random sketch** (randomised truncated SVD of the TF-IDF and cue matrices, 256 d), to
   learn how much of Cypha's gap is the projection. If `bc`/`abc` through LSA closes much of it, the projection
   was a confound and the verdict softens to "online updates only".
2. **Cut the online noise.** Average the prototypes over passes (Polyak averaging), use class-mean batches
   per pass, or a decaying `delta_lr`; the batch LDA row suggests up to 10 points are available. Also try `use_kernel_llr` and `use_class_gmm`.
3. **Richer dense cues:** the PPMI-SVD vector of the head word and of the wh-bigram as extra blocks, then
   C6 sentence vectors when they exist.
4. **Seeds and intervals.** Five seeds for the dev split and the shuffles, and a paired bootstrap over the 500
   test questions for each headline difference.
5. **E4 v0 for F1.** Wire the perceptron (fine types plus the relation label) with the dev temperature and the
   90%-coverage threshold into the Q/A path, and train it on the plan's wider sources (SQuAD, Dolly, templated
   and distilled questions). Score it by right / abstained / wrong, as rule 9 asks, not by accuracy alone.
6. **Register** `question_classifier` in `native/CMakeLists.txt` with the snippet above.
7. **Predict first.** Next report writes its range before the run. Suggested for step 1: Cypha through LSA
   reaches 80 to 88% coarse and 66 to 76% fine.
