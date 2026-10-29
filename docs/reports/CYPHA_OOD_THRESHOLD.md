# Cypha's OOD threshold, learned from the training inputs

**Odin Loch** · 2026-10-29 · found by [E1](E1_CYPHA_TAGGER.md) (HANDOFF §5.29) · built on [CSAVE](CYPHA_SAVE_LOAD_FIX.md) (save format 2)

## The problem

`Cypha::predict` sets `PredictOut::is_ood` when the GH gate's inflation is above a fixed constant: `anomaly_score > kOodThreshold`, with `kOodThreshold = 3.0` (`cypha.hpp`). The REST server's own GH path does the same with its own copy of 3.0.

The score is `r_eff / r_base − 1` (`gh_infer_anomaly_score`). Here `r_eff = nig_r_eff_scalar(m, r_base, χ, ψ)`, `m` is the mean per-dimension squared Mahalanobis distance of the latent from the world prior, and `r_base = 1 / mean(1/v)`. Inside, the GIG posterior takes `χ + m / r_base`. `m` has no units, but `r_base` is a variance. So the score grows as the latent variance shrinks: inputs with small per-dimension variance (such as unit-norm vectors, whose per-dimension variance is about 1/d) score high whatever they are. In E1 (279 dimensions) the flag fired on 100% of test tokens, unknown words, random letters and Wikipedia prose alike.

`ood_sigma` does not enter the decision. It is a training-step setting.

## Design (registered before any run)

- **The score stays.** `anomaly_score` is unchanged: same formula, same value. Only the decision changes.
- **The threshold is learned from training.** In `Cypha::update` (classification rows, not the regression head), the input is scored just before it is learned, with the model as it stands: encode, then the same score `predict` would give (a shared helper, so the numbers are bit-identical). The input has not been trained on yet, so each score is a held-out score for the model of that moment.
- **A window of recent scores.** The last W = 1,024 scores are kept in a ring. The threshold is their empirical (1 − α) quantile: sorted scores s, τ = s[⌈(1 − α)·n⌉ − 1], and `is_ood = anomaly_score > τ`. The target α is 0.05 by default (`set_ood_target_rate`).
- **Until 50 scores are held**, the old fixed 3.0 is used. A model never trained through `update` (a fixture, a file from another tool) behaves as before.
- **Saved with the model.** Save format 2's `cypha_save_state` session block gains `ood_target`, `ood_scores` (the ring, in storage order) and `ood_head`. On load, τ is recomputed from them, so `predict` and further training stay bit-identical after a reload. Files without these keys (format 1, or format 2 from CSAVE alone) load with an empty ring: the fixed 3.0 until 50 updates.
- **REST.** The server's own GH predict path uses the model's threshold (`Cypha::ood_threshold()`), not its own 3.0.
- **Cost.** One extra encode and one O(d) score per update, plus a selection over at most 1,024 numbers.

## Callers whose behaviour depends on the OOD decision

| Caller | What it does with `is_ood` |
|---|---|
| `cypha_tagger` (E1) | Reports OOD flag rates and the flag's AUROC for wrong tags |
| `thought/bindings_cypha.hpp` | `classify` primitive's `ood`; `DifApplicability`'s query `ood` (THOUGHT 3.1) |
| `apps/cypha_rest.cpp` | `/predict` returns `is_ood` (via `Cypha::predict`); its view-based GH path computed its own with 3.0; session history and explanations carry it |
| `cypha_save_load_test` (CSAVE) | Requires `is_ood` identical after a reload |

The forecasting node, Branch A router, benches, `topic_bench` and the Qt shell do not read `is_ood` (checked by search). `ood_sigma` (a training setting) is not changed.

## Measures

**M1, synthetic.** Three families × d ∈ {16, 64, 384}, K = 5 classes, 3 seeds each:
- **F1, unit per dimension:** class means N(0, 1.5²·I), noise N(0, I).
- **F2, unit norm:** F1 scaled by 1/√d.
- **F3, correlated:** a latent of d/4 dimensions drawn as F1, mapped by a random d × (d/4) matrix with N(0, 4/d) entries, plus N(0, 0.1²·I) noise.

Each model is a fresh `create_fresh_model_root` model (field 16), trained through `Cypha::update` with the GH gate on 1,500 rows, two passes (3,000 updates; E1 also trained two passes). Then, with `predict` (GH, defaults):
- held-out in-distribution rows (2,000): the false OOD rate;
- planted out-of-distribution rows (1,000 each):
  - P1: class means with the noise doubled (variance × 4);
  - P2: a new cluster whose mean is drawn with 3× the class-mean spread;
  - P3: uniform in [−3s, 3s] per dimension, s the training data's per-dimension standard deviation;
  - P4: the training distribution stretched √10 times about its mean (variance × 10).

Reported for today's fixed 3.0 and for the learned τ: false OOD rate, true OOD rate per planted set (Wilson 95%), τ, and the score's AUROC in-distribution against each planted set (threshold-free, the same for both).

**M2, E1's tagger.** E1's final configuration, rerun with the new code: dev OOD flag rate (dev read), then E1's test read **once for this change**: OOD flag rate on test tokens, unknown words, random letters and Wikipedia prose; the flag's AUROC for wrong tags; and the accuracy lines.

**M3, cost.** Update time per row at d = 384, before and after (a timing loop, reported only).

## Rule

Adopt the learned threshold as the default if all hold:
- **R1, calibrated:** with the learned τ, the held-out in-distribution false OOD rate (pooled over 3 seeds) is within [2.5%, 10%] (0.5× to 2× the 5% target) at all 9 settings.
- **R2, no loss where today works:** at each setting and planted set where today's false OOD rate is at most 5% (today at least as strict as the target), the new true OOD rate is at least today's − 0.03.
- **R3, it detects:** P4's true OOD rate with the learned τ is at least 0.90 at all 9 settings.
- **R4, save and load:** `native_cypha_save_load_test` passes, and the new CTest's round trip (ring, τ, `is_ood` identical after a reload; a file without the keys falls back to 3.0) passes.
- **R5, learning unchanged:** E1's dev and test accuracy lines are identical to E1's recorded run (the threshold changes no label and no training step).

If any fails, the old behaviour stays and the code is kept only if it earns its place (rule 4).

## Predictions (registered before any run)

| # | Prediction | Falsified if |
|---|---|---|
| Q1 | Today's 3.0 is saturated (false OOD rate ≥ 90%) in F2 at d = 64 and 384, and not saturated (≤ 10%) in F1 at d = 16 | Either part fails |
| Q2 | R1 holds at all 9 settings, pooled false rates 3–8% | Any setting outside [2.5%, 10%] |
| Q3 | Two-pass training biases the false rate up a little: the mean pooled false rate over the 9 settings is above 5.0% and under 7% | Mean ≤ 5.0% or ≥ 7% |
| Q4 | R3 holds; P2's true OOD rate is at least 0.5 in F1 at every d | P4 < 0.9 anywhere, or P2 < 0.5 in F1 |
| Q5 | E1 test: the flag fires on 4–8% of test tokens and on at least 20% of random-letter words | Either outside |
| Q6 | E1's accuracy lines are identical | Any digit differs |
| Q7 | Update time per row at d = 384 rises by under 15% | 15% or more |

## Amendment before the registered read (2026-10-29)

Building the CTest showed a flaw in the registered design before M1 ran. Its small calibration check (F2, d = 64, 700 rows, one pass) put 13.4% of held-out rows above the learned threshold, against a 5% target. An exploratory check on the same setting with five seeds (`scratch/diag.cpp`, not a registered read) gave 6.2%, 6.8%, 3.0%, 13.0% and 5.8%.

**Why.** The window held scores taken while the model was still moving. In the 13.0% seed, the window's 95% quantile was 4.04, but the final model scores held-out rows at 4.30 (95% quantile). It scores the last 300 training rows, re-scored, at 4.28. So the bias comes from drift, not from rows being seen in training: re-scoring training rows with the current model matches held-out rows.

**Amended design (the default under test):**
- `update` keeps the latent of each training row (encoded just before the row is learned) in a ring of the last W = 1,024 rows, instead of its score.
- When the threshold is needed after a change, every held latent is scored with the current model (the same helper `predict` uses), and τ is the (1 − α) quantile of those scores. It is cached until the next update.
- Saved in format 2's session block: `ood_target`, `ood_latents` (rows × d), `ood_count`, `ood_head`. Old files load with an empty ring (fixed 3.0 until 50 rows).
- Cost: W·d stored numbers (2.3 MB at d = 279), and a re-score of W·d at the first `predict` after updates.
- Known limit: with a learning encoder (`enc_lr` > 0), held latents come from the encoder of their time.

**R1 and ties at zero (amended with the above, before the registered read).** A smoke run of the measure tool (F1, d = 16, 200 rows; not the registered read) showed that the score is exactly 0 (no inflation) for most in-distribution rows in F1. The threshold is then 0, the lowest possible, and the held-out false rate can be far below 5%. R1's lower bound is there to catch a threshold set too high. So R1 now reads: the false rate is within [2.5%, 10%], **or** it is below 2.5% while τ is 0 (the score's floor). The upper bound is unchanged.

**Arms in M1.** The amended design (B, the default under test) and the registered one (A, a window of scores, computed in the measure tool from `predict` just before each update). The rule R1–R5 applies to B. A is reported only. The measures, rule and predictions above are unchanged.

## Results (2026-10-29)

**Decision: adopted.** R1–R5 all hold for the amended design (B). `predict`'s `is_ood` now compares the unchanged anomaly score with the threshold learned from the model's own training rows. The fixed 3.0 stays only until 50 rows have been learned.

### M1: synthetic (pooled over 3 seeds; 6,000 held-out rows and 3,000 rows per planted set per setting)

Command: `cypha_ood_test measure DIR --seeds 3 --dims 16,64,384 --rows 1500 --passes 2 --held 2000 --planted 1000`. Run twice, on the first build (a second encode in `update`) and on the final one (the latent taken from the training step). The table is identical line for line; only the cost lines differ. A third run printed each seed's τ.

**False OOD rate on held-out in-distribution rows (R1):**

| Setting | Fixed 3.0 | Learned τ (B) [95%] | Score window (A) | τ (B), median of 3 |
|---|---|---|---|---|
| F1, d = 16 | 0.0% | 0.0% [0.0, 0.1] | 0.0% | 0 |
| F1, d = 64 | 0.0% | 0.0% [0.0, 0.1] | 0.0% | 0 |
| F1, d = 384 | 0.0% | 0.0% [0.0, 0.1] | 0.0% | 0 |
| F2, d = 16 | 0.0% | 5.2% [4.7, 5.8] | 5.7% | 1.49 |
| F2, d = 64 | 40.1% | 4.7% [4.2, 5.3] | 3.0% | 3.48 |
| F2, d = 384 | 100.0% | 5.7% [5.2, 6.3] | 8.4% | 11.8 |
| F3, d = 16 | 0.0% | 4.3% [3.8, 4.8] | 10.1% | 0.238 |
| F3, d = 64 | 0.0% | 0.0% [0.0, 0.1] | 0.0% | 0 |
| F3, d = 384 | 0.0% | 0.0% [0.0, 0.1] | 0.0% | 0 |

- Where the score varies (F2; F3 at d = 16), B is within 4.3–5.7% of the 5% target. A, the first registered design, misses twice (10.1% and 8.4%).
- In F1, and in F3 at d = 64 and 384, in-distribution rows all score exactly 0 (no inflation), and τ is 0 in each of the 9 seeds. R1's tie clause applies.

**True OOD rate on planted rows, fixed 3.0 → learned τ (the score's AUROC against held-out rows, the same for both):**

| Setting | P1 noise ×2 | P2 far cluster | P3 uniform | P4 stretched |
|---|---|---|---|---|
| F1, d = 16 | 0.00 → **0.04** (0.52) | 0.00 → **1.00** (1.00) | 0.00 → **0.09** (0.54) | 0.00 → **0.97** (0.98) |
| F1, d = 64 | 0.00 → **0.00** (0.50) | 0.00 → **1.00** (1.00) | 0.00 → **0.00** (0.50) | 0.00 → **1.00** (1.00) |
| F1, d = 384 | 0.00 → **0.00** (0.50) | 0.00 → **1.00** (1.00) | 0.00 → **0.00** (0.50) | 0.00 → **1.00** (1.00) |
| F2, d = 16 | 0.10 → **0.80** (0.95) | 1.00 → **1.00** (1.00) | 0.17 → **0.98** (1.00) | 0.99 → **1.00** (1.00) |
| F2, d = 64 | 1.00 → **1.00** (1.00) | 1.00 → **1.00** (1.00) | 1.00 → **1.00** (1.00) | 1.00 → **1.00** (1.00) |
| F2, d = 384 | 1.00 → **1.00** (1.00) | 1.00 → **1.00** (1.00) | 1.00 → **1.00** (1.00) | 1.00 → **1.00** (1.00) |
| F3, d = 16 | 0.00 → **0.27** (0.71) | 0.00 → **0.79** (0.95) | 0.05 → **0.99** (1.00) | 0.09 → **0.90** (0.98) |
| F3, d = 64 | 0.00 → **0.17** (0.59) | 0.00 → **1.00** (1.00) | 0.00 → **0.37** (0.69) | 0.00 → **0.99** (1.00) |
| F3, d = 384 | 0.00 → **0.00** (0.50) | 0.00 → **1.00** (1.00) | 0.00 → **0.00** (0.50) | 0.00 → **1.00** (1.00) |

- **The fixed 3.0 was wrong both ways.** It fires on everything in F2 at d ≥ 64 (E1's case). In F1, and in F3 at d ≥ 64, it never fires, not even on a far cluster or a variance × 10 stretch. The learned τ catches those (1.00).
- **R2** (where today's false rate is at most 5%: F1, F2 at d = 16, F3): the learned τ never detects less than the fixed 3.0. The smallest change is +0.00.
- **R3:** P4 is caught at 0.90 or more at all 9 settings. The lowest is F3 at d = 16: 0.904 [0.893, 0.914].
- **The score itself has a blind spot (outside this change).** In F1 at d ≥ 64 and F3 at d = 384, the score is 0 for noise-doubled and uniform rows too (AUROC 0.50): the gate's inflation is clipped at 0 when the posterior does not move. No threshold can see those. A better score is a separate question.
- `is_ood` agreed with `anomaly_score > ood_threshold()` on every one of the 162,000 scored rows.

### M2: E1's tagger

E1's final configuration (`--alpha 0.5 --vec simvec --whiten 0.1 --delta-lr 0.03 --world-lr 0.008 --passes 2`), rerun with the final build. The test split was read once for this change.

| OOD flag rate | Fixed 3.0 (E1) | Learned τ [95%] |
|---|---|---|
| Dev tokens | 100% | 6.8% [6.5, 7.2] |
| Test tokens | 100% | 7.5% [7.2, 7.8] (1,882 / 25,094) |
| Test, unknown words | 100% | 13.2% [11.9, 14.7] (303 / 2,292) |
| Random letters | 100% | 15.1% [14.6, 15.5] (3,779 / 25,094) |
| Wikipedia prose | 100% | 5.0% [4.5, 5.5] (332 / 6,686) |
| AUROC of the flag for a wrong tag | 0.500 | 0.534 |

- **The accuracy lines are identical to E1's run, digit for digit** (dev pass 1 91.59%, pass 2 91.96%; test 92.06%, unknown words 71.07%, questions 90.66%; calibration T 3.490, ECE 0.0415). The threshold changes no label and no training step (R5).
- The flag now separates: unknown words and random letters are flagged about twice as often as test tokens. By E1's rule 5 it is still not "useful" (it needs 20% or more of random-letter words and under 5% of test tokens). The score, not the threshold, likely limits that: random-letter words reach E1's count features through the same unknown-word path as real unknown words (13.2% against 15.1%). Not measured further here.
- **A process slip, recorded.** A separate dev-only run was made first (`runs/e1/dev.sh`), but its `cypha_tagger` binary was still the first build's (design A, the score window): after the amendment only the two test binaries were relinked. Its dev rate, 4.1% [3.8, 4.3], is therefore arm A's. Design B's dev rate is the 6.8% printed by the test run, which used the final build (its dev line comes before the test read).

### M3: cost

- **Update.** The first build encoded each row a second time: 31–35% of an update at d = 384 (11–14% at d ≤ 64). The final build takes the latent the training step already computes (`TrainStepExtras::latent_out`), so `update` only copies d numbers: 0.04–5.3 µs a row, 0.03–0.5% of an update (about 1.0–1.4 ms a row at d = 384 on this loaded machine). The ring and τ are bit-identical between the two builds (M1 above).
- **Re-score.** The first `predict` after updates re-scores the held latents: 0.6–7.4 ms in single timings under load (W·d work, at most 1,024 × d).
- **Memory and file size.** W × d numbers: 2.3 MB at d = 279, 3.1 MB at d = 384.

### Predictions scored

| # | Prediction | Result |
|---|---|---|
| Q1 | Fixed 3.0 saturated (≥ 90%) in F2 at d = 64 and 384; not saturated in F1 at d = 16 | **Miss:** d = 384 100% and F1 0%, but d = 64 40.1% |
| Q2 | R1 at all 9 settings, rates 3–8% | **Miss:** R1 holds, but only through the tie clause at 6 settings (0%); the registered falsifier (any setting outside [2.5%, 10%]) is met |
| Q3 | Mean false rate over the 9 settings above 5.0% and under 7% | **Miss:** 2.2% (the six 0% settings) |
| Q4 | R3 holds; P2 at least 0.5 in F1 | **Hit:** P4 ≥ 0.904 everywhere; P2 1.00 |
| Q5 | E1 test: 4–8% of test tokens; at least 20% of random letters | **Miss:** 7.5% (hit), random letters 15.1% (miss) |
| Q6 | E1 accuracy lines identical | **Hit** |
| Q7 | Update time at d = 384 rises under 15% | **Miss** on the first build (31–35%); 0.4–0.5% after taking the training step's latent |

2 hits, 5 misses.

### Checks

- **New CTest `native_cypha_ood_test`** (23 checks): the fixed 3.0 before 50 rows; the ring holds the latents of the last 1,024 rows bit for bit, with the right head; τ equals an independent quantile of the current model's scores (5% and 20% targets); `predict`'s score equals the shared helper's, and `is_ood` follows τ; a changed GH session re-scores; save, load and further learning agree; a file without the keys loads with an empty ring; calibration on F2, d = 64 (held-out 4.6%, P4 100%).
- **Six planted bugs, each fails it:** the quantile index rounded down (3 checks fail); latents not saved (load throws); `predict` still on 3.0 (3); a ring that stops at 1,024 (2); the latent taken after learning (1); a cache not keyed on the GH session (1).
- **`native_cypha_save_load_test`** passes (8 configurations), now also comparing the ring, head, target and threshold after reload.
- **The CI set** (`ctest -R native_ -LE cypha_slow`, full Release build): 144 tests. On the parallel run under load 140 passed and four hp smokes timed out (`hp_fold`, `cyphalm_serve_state`, `hp_predictor_assign_parity`, `hp_undo`, as on main under the same load); run one at a time, all four pass. `native_cypha_tagger_selftest` and `native_thought_bindings_test` pass.

### Callers affected

- `cypha_tagger` (E1): flag rates as above.
- `thought/bindings_cypha.hpp`: the `classify` primitive's `ood` and `DifApplicability`'s query `ood` now use the learned threshold once a process has 50 labels.
- `apps/cypha_rest.cpp`: `/predict`'s `is_ood` (through `Cypha::predict`) and its view-based GH path (`Cypha::ood_threshold()` in place of its own 3.0).
- Models never trained through `update` (fixtures, files from other tools) and files saved before this change keep the fixed 3.0 until they learn 50 rows.
