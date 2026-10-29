# K4 and K5: thought primitives bound to Cypha, first processes on benchmarks, DIF applicability

**Odin Loch** · 2026-10-29 · THOUGHT.md §3.1, §7, §10, §12 (K4, K5) · follows [K2_K3_THOUGHT_INTERPRETER.md](K2_K3_THOUGHT_INTERPRETER.md)

(The design, the reads, the rules and the predictions below were written and committed before any K4 or K5 code or run. Results are added under them.)

**Question.** K2 and K3 built the interpreter and the standard learners, with every primitive stubbed. Can the primitives be bound to Cypha's real libraries? Do the first processes do their job on benchmarks with known answers? And does the Cypha classifier, given only the features of a situation, tell which process applies as well as the Beta stand-in that is handed the task class?

**Rule 10.** A thought process may only answer a question through CyphaLM. Traces are working, not answers (THOUGHT §9). The bindings enforce this: in answer mode, `communicate` sends an answer only when its source is CyphaLM; anything else becomes an "I don't know" card. Retrieved facts may gate an answer, never produce it.

## 1. K4: the bindings (fixed before any code)

One header, `native/include/cypha/thought/bindings.hpp`, binds every primitive the library uses (33 ops over 813 primitive steps in 144 processes) to an implementation. Each binding reads and writes named keys of the working memory, so one binding serves every process that uses the op. A process step can still be bound on its own (`<process>.<step>`), which wins.

| Primitive | Bound to | Working-memory contract (main keys) |
|---|---|---|
| `perceive`, `read` | CyphaLM: bits per byte and per-byte surprise of a text under a packed-trie model (`packed_trie.hpp`) | `text` → `surprise` {bpc, max} |
| `recall_case` | CyphaLM's System 1: its greedy continuation of the prompt and a confidence from its own log-probabilities; the ∞-gram's longest match count when an index is given (`fm_infinigram.hpp`) | `prompt` → `recall` {text, mean_lp, match_n, source = "cyphalm"} |
| `generate` | CyphaLM sampling (`sample_continuation`), seeded | `prompt` → `generated` |
| `retrieve` | Fact store opinions (ECS, `ecs.hpp`); in answer mode the result is marked gate-only | `query` → `opinion` |
| `score`, `evaluate` | ECS fusion of evidence items into an opinion and grade (`ecs.hpp`) | `evidence` → `opinion` |
| `classify` | Epistemic state of an opinion (`epistemic.hpp`); the Cypha DIF classifier when one is attached (`thought/bindings_cypha.hpp`, links `cypha_core`) | `opinion` → `state`; `features` → `label`, `confidence`, `ood` |
| `explain`, `communicate` | Answer and "I don't know" cards with reason codes (`epistemic.hpp`); the rule-10 guard | `answer` {text, source}, `reason` → `card` |
| `test`, `verify`, `falsify` | A hypothesis test with three-way likelihoods and its Bayes factor (`hypothesis.hpp`) | `hypothesis`, `outcome` → `posterior` |
| `update` | Posterior update from a Bayes factor, or a Beta update | `posterior`, `bf` → `posterior` |
| `decide` | `decide` of the hypothesis loop (§7.2), or an LCI promotion decision (`lci.hpp`), or a threshold guard | `decision.kind` → branch |
| `monitor` | CUSUM on a stream, and the e-process (`lci.hpp`) | `obs` → `cusum`, `alarm` |
| `select`, `rank` | Highest value of information among gaps (`gap_map.hpp`), or Thompson draws among options | `options` → `chosen` |
| `estimate` | Beta mean with its Wilson interval; lognormal intervals | `counts` → `estimate` {p, lo, hi} |
| `combine` | Log-space product of interval estimates (Fermi), or opinion fusion | `factors` → `estimate` |
| `transform` | Isotonic calibration (`ecs.hpp` `IsotonicCalibrator`) | `raw` → `calibrated` |
| `compare`, `match` | Cosine of word vectors (SCRIBE's `WordVectors`), numeric difference | `a`, `b` → `similarity` |
| `infer` | Derivation factor of a premise chain with the answer-mode depth cap 3 (`ecs.hpp`) | `premises` → `derived` |
| `simulate`, `measure`, `decompose`, `generate` (fallback), `act`, `schedule`, `wait` | Callbacks the caller supplies; a default that records the step and costs 1 | — |
| `ask` | The Ask-Odin queue: a question appended to a JSONL file, never sent anywhere (no teacher inference in the container) | `question` → queue entry |
| `record` | An append-only JSONL journal (the experiment-DB hook) | any → line |

**Checks (CTest `native_thought_bindings_test`).** A known-answer group for every primitive above, a planted rule-10 bypass, the unbound list of all 144 processes under the default toolkit, and the K2 replay hash with the new interpreter hook unset.

## 2. K5: the first processes (fixed before any run)

THOUGHT §12 names `dual_process` + `confidence_monitor` + `cooperative_reply` on T-QA-wiki, `half_split`, `fermi_estimate`, `sprt_decide` and `ach`. §10 gives their benchmarks. Chosen, because a benchmark with known answers exists in the repo or can be generated exactly:

| # | Processes | Benchmark | Baseline | Score |
|---|---|---|---|---|
| K5a | `confidence_monitor` inside `dual_process`, checked by `cooperative_reply` | T-QA-wiki: CyphaLM's recorded answers (rows of `cyphalm_trie qa`, B4c's run, dev 130 and frozen test 200 questions, chat and wiki forms, trie and trie + IG) | `cyphalm_trie qa`'s own gate (A0) | right / abstained / wrong at a 2% wrong rate; card reason codes |
| K5b | `half_split`, `differential_diagnosis` | planted-fault chains: uniform priors and equal costs; skewed priors (Zipf, s = 1.2) and unequal costs (1–5) | random probing | tests and cost to isolate; fault found |
| K5c | `ach`, `sprt_decide` | synthetic competing-hypothesis cases: 4 hypotheses, 12 evidence items, consistency drawn from the true hypothesis with noise 0.2, and 4 planted items consistent with all | counting consistent items (no diagnosticity) | correct hypothesis; evidence used |
| K5d | `change_detection` (CUSUM), `spc_control` (Shewhart) | simulated Gaussian streams with planted mean shifts of 1σ and 3σ | each other, at matched in-control run length (about 370) | detection delay at the matched false-alarm rate |

- **Not run:** `fermi_estimate` (no Fermi questions with known answers in the repo) and the planning puzzles (no benchmark yet).
- **Standard learners:** on K5b's skewed chains, `solve` chooses among `half_split`, `differential_diagnosis` and random probing by Thompson sampling; regret against the best in hindsight is reported.
- **Splits:** generated cases use dev seeds 1–100 for any choice and test seeds 1001–1200, read once. K5a chooses its gate on dev rows and reads the test rows once (a new change: the gate is now a process).

## 3. DIF applicability (fixed before any run)

THOUGHT §3.1 asks for a Cypha DIF classifier per process, trained online on situation features, with its OOD flag routing to `impasse`. K3's stand-in is a Beta per (process, task class), which needs the task class handed to it.

- **The hook.** `Interpreter::set_applicability(model)` replaces the Beta in `run` and `solve` when set; unset, the interpreter is unchanged (the K2 replay hash must hold).
- **The stream.** Tasks of five types (K5b's chains, K5c's binary evidence streams and competing-hypothesis cases, K5d's monitored streams), each with 8 situation features: 5 informative (structure size, prior skew, cost spread, streaming, number of hypotheses) plus noise, and 3 pure noise. Candidates: the 7 processes of K5b–K5d. A process run on a task without its structure ends in IMPASSE (no chain to split, no stream to watch).
- **Three applicability models:**
  - **Beta-class:** the K3 stand-in, given the true task type as its class (an oracle the real system does not have);
  - **Beta-pooled:** one Beta per process, no class (what the stand-in gives without a hand-made class);
  - **DIF:** one Cypha classifier per process on the 8 features, labels "yes" (R ≥ 0.5) and "no" (IMPASSE), no class given.
- **Reads:** a training stream of 1,500 tasks (dev seeds), then a held-out stream of 1,000 tasks (test seeds) on which every model predicts before each run. Calibration of P(applicable): ECE (10 bins) and Brier against whether the chosen process applied. Selection: the share of runs that end in IMPASSE, and regret in reward against an oracle that runs the best applicable process. OOD: after the stream, 200 tasks of a sixth type (far feature values; no candidate applies): the share flagged.
- **LCI** for the DIF-vs-Beta-pooled comparison, on per-task regret differences of the held-out stream.

## 4. Decision rules

1. **K4:** the bindings are adopted when every check group passes, the planted rule-10 bypass is caught, and the K2 replay hash is unchanged.
2. **K5:** reported against the baselines; nothing in Cypha's defaults changes.
3. **DIF applicability** becomes the default applicability model of the K5 tool (the interpreter itself keeps the hook, unset by default) if on the held-out stream: its ECE ≤ 0.10, its regret ≤ 1.1 × Beta-class's, and LCI ≥ 0.95 against Beta-pooled. Otherwise the Beta stays and the numbers are reported.

## 5. Predictions (registered before any run)

| # | Prediction |
|---|---|
| K4a | With the default toolkit bound, all 144 processes have no unbound op |
| K4b | The planted rule-10 bypass (an answer whose source is the fact store) becomes an "I don't know" card in 100% of cases |
| K4c | The K2 replay hash `18a61550bffc735c` is unchanged |
| K5a1 | Dev: the process answers at most 2 of 520 rows; the gate it picks answers nothing in at least 3 of the 4 configurations (as A0) |
| K5a2 | Test: wrong rate ≤ 2% with a Wilson upper bound ≤ 4.5%; at most 2 right; every "I don't know" card carries a reason code |
| K5b1 | Uniform chains of 64: `half_split` 6.0 tests; `differential_diagnosis` within 0.1 of it; random probing about 31 |
| K5b2 | Skewed chains: `differential_diagnosis` costs at least 15% less than `half_split` |
| K5b3 | Thompson picks `differential_diagnosis` in at least 80% of the second half of skewed episodes |
| K5c1 | `ach` accuracy ≥ 0.85; the counting baseline at least 0.10 lower |
| K5c2 | `sprt_decide` type I and type II ≤ 0.05 |
| K5d1 | 1σ shift: CUSUM's mean delay at most half of Shewhart's |
| K5d2 | 3σ shift: Shewhart's mean delay at most CUSUM's |
| A1 | Beta-class: ECE ≤ 0.05 |
| A2 | Beta-pooled: ECE ≥ 0.20 |
| A3 | DIF: ECE ≤ 0.10, regret ≤ 1.1 × Beta-class's |
| A4 | OOD: DIF flags ≥ 80% of the sixth type and ≤ 10% of known types; the Betas flag none |

## Results (run 2026-10-29)

Code: `native/include/cypha/thought/bindings.hpp`, `bindings_cypha.hpp`, the hook in `interpreter.hpp`, `native/tools/thought_bindings_test.cpp` (CTest `native_thought_bindings_test`) and `native/tools/thought_k5.cpp`. The machine was shared with many other runs.

### K4: the bindings

- **All 33 primitives are bound.** With the default toolkit, none of the 144 processes has an unbound op (K4a). A smoke run of every process with the bound toolkit ends cleanly in 144 of 144 (no step errors).
- **Rule 10.** The planted bypass (100 answers whose source is the fact store, in answer mode) gives 100 "I don't know" cards with reason `NO_EVIDENCE`, and 100 counted refusals (K4b). CyphaLM's own answers pass. In discovery mode, working text is not an answer and passes.
- **The K2 replay** gives hash `18a61550bffc735c` with the hook unset, as in K2 (K4c). `native_thought_test` passes 5,540 checks.
- **`native_thought_bindings_test`: 120 checks**, one group per primitive (CyphaLM's surprise, recall and sampling on a small model, with the ∞-gram match; ECS fusion against `fuse`; epistemic states; cards; Bayes factors, two-way and S7b three-way; Beta and posterior updates; guards, thresholds and an LCI decision; CUSUM; selection by score, Thompson draws and the gap map's VOI; Wilson; the Fermi product; isotonic calibration; word-vector cosine; derivation depth; the Ask-Odin queue; the journal), the whole library bound, `cooperative_reply` and `dual_process` end to end, and the applicability hook (not applicable, OOD, labels, `solve`, back to the Beta).
- **Planted bugs caught:** the rule-10 guard switched off (7 checks fail), CUSUM without its reference value k (1), the Fermi product in log10 (4).
- **One change while building, before any K5 run.** The first test corpus was ambiguous within 4 bytes ("the " was followed by cat, mat, dog and rug), so CyphaLM's greedy recall of "the cat sat on the " gave "rug.": its 4-byte reach again (TOKENS_PLAN §1). The test corpus was made unambiguous within 4 bytes. Recall also reports the prompt's own surprise (`prompt_bpc`) as its familiarity signal, because a greedy continuation soon runs into known text whatever the prompt.

### K5a: `confidence_monitor` inside `dual_process` on T-QA-wiki

- **Rows:** B4c's run of `cyphalm_trie qa` with the defaults (`/root/work/echo/runs/T_on.rows.tsv`, sha256 `f11a5577ce659e62…`): dev 130 and frozen test 200 questions, in 4 configurations. Raw answers on test: chat / trie + IG 2 right, 97 abstained, 101 wrong; the other three 0 right.
- **The process.** `dual_process.fast` is CyphaLM's own recorded answer. Its gate runs `confidence_monitor`: the isotonic map of a feature, fit on dev answers per configuration (`transform`), the threshold chosen by A0's rule on dev (most right with at most 2% wrong), then an answer card or an "I don't know" card (`communicate`). No second CyphaLM route exists, so the slow route abstains.
- **A fix before the test read.** The first dev run dropped 15 rows whose answer was empty (a parsing slip); fixed, then rerun on dev.

| Set | Form | Model | Gate chosen on dev | Right | Abstained | Wrong | Wrong rate (Wilson upper) | "I don't know" cards with a reason |
|---|---|---|---|---:|---:|---:|---|---:|
| dev (in sample) | chat | trie | answer nothing | 0 | 130 | 0 | 0 (0.029) | 130 |
| dev (in sample) | chat | trie + IG | `min_lp`, calibrated ≥ 1.0 | 3 | 127 | 0 | 0 (0.029) | 127 |
| dev (in sample) | wiki | trie, trie + IG | answer nothing | 0 | 130 each | 0 | 0 (0.029) | 130 each |
| **test (read once)** | all four | | as on dev | **0** | **800** | **0** | 0 (0.019 each) | **800** |

- On the test the process answers nothing, as `cyphalm_trie qa`'s own gate does (CURRENT: 0 right kept at 2% wrong). Every abstention carries a reason code. The 3 dev answers were in-sample; the same gate keeps none on the test.

### K5b: planted-fault chains of 64 (test seeds 1001–1200)

| Chains | Process | Tests | Cost | Found | Cost, 95% CI |
|---|---|---:|---:|---:|---|
| uniform | `half_split` | 6.000 | 6.00 | 1.000 | [6.00, 6.00] |
| uniform | `differential_diagnosis` | 6.000 | 6.00 | 1.000 | [6.00, 6.00] |
| uniform | random probing | 31.55 | 31.55 | 1.000 | [29.14, 34.05] |
| skewed | `half_split` | 4.75 | 14.27 | 1.000 | [13.38, 15.14] |
| skewed | `differential_diagnosis` | 5.57 | **11.10** | 1.000 | [10.53, 11.71] |
| skewed | random probing | 34.16 | 101.87 | 1.000 | [93.92, 109.58] |

- **Skewed chains:** `differential_diagnosis` (expected information per unit cost) saves 3.17 cost units [2.50, 3.87] a chain over `half_split`, 22.2%. It runs more tests than `half_split` but cheaper ones.
- **The standard learners:** Thompson sampling among the three on skewed chains (2,000 episodes, G = 100) picks `differential_diagnosis` in 95.8% of the second 1,000 episodes; the 2% floor leaves `half_split` 2.3% and random probing 1.9%.

### K5c: competing hypotheses (test seeds 1001–1200) and SPRT

- **`ach`** picks the true hypothesis in **0.830 [0.772, 0.876]** of 200 cases. The counting baseline (most consistent items, all evidence counted) gets 0.580 [0.511, 0.646]: `ach` is 0.250 [0.180, 0.320] better. `ach` uses 11.2 of 16 items: it drops the 4 planted items and those on which every hypothesis agrees.
- **`sprt_decide`**, 2,000 runs per hypothesis: type I 0.0385, type II 0.0345 (bound 0.05); mean evidence 36.6 and 37.1 observations.

### K5d: monitored streams (h chosen on dev seeds 1–100; test seeds 1001–1200)

The CUSUM limit h = 4.10 gave an in-control run length of about 370 on dev; the Shewhart limit is L = 2.7822 (P(Z > L) = 1/370).

| Shift | Process | Mean run length | 95% CI | Median |
|---|---|---:|---|---:|
| none | `change_detection` (CUSUM) | 333.9 | [285.9, 384.7] | 218 |
| none | `spc_control` (Shewhart) | 432.3 | [371.5, 490.1] | 279 |
| 1σ | `change_detection` | **8.56** | [7.96, 9.20] | 8 |
| 1σ | `spc_control` | 29.08 | [25.23, 33.03] | 21 |
| 3σ | `change_detection` | 2.24 | [2.15, 2.33] | 2 |
| 3σ | `spc_control` | **1.71** | [1.56, 1.85] | 1 |

- Both in-control intervals contain 370. CUSUM catches a 1σ shift 3.4 times sooner; the Shewhart rule is faster for a 3σ jump, as the textbook says.

### Applicability (THOUGHT 3.1)

Training 1,500 tasks, held-out 1,000, then 200 of a new type, on three streams (seeds 1001–1003). ECE of the chosen process is the registered read; all (task, process) pairs are also shown.

| Model | Seed | ECE, chosen (n) | ECE, all pairs | Brier, all | IMPASSE share | Mean regret | 95% CI | OOD flagged: new / known |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Beta-class (given the task type) | 1001 | 0.0051 (958) | 0.0750 | 0.0069 | 0.006 | 0.0091 | [0.0049, 0.0145] | 0 / 0 |
| Beta-class | 1002 | 0.0057 (946) | 0.0752 | 0.0069 | 0.008 | 0.0186 | [0.0112, 0.0265] | 0 / 0 |
| Beta-class | 1003 | 0.0055 (953) | 0.0745 | 0.0069 | 0.008 | 0.0092 | [0.0048, 0.0145] | 0 / 0 |
| Beta-pooled (no class) | 1001 | 0.0146 (968) | 0.0461 | 0.1863 | 0.778 | 0.7047 | [0.6792, 0.7293] | 0 / 0 |
| Beta-pooled | 1002 | 0.0189 (971) | 0.0399 | 0.1912 | 0.801 | 0.7237 | [0.6999, 0.7465] | 0 / 0 |
| Beta-pooled | 1003 | 0.0039 (965) | 0.0598 | 0.1880 | 0.794 | 0.7177 | [0.6942, 0.7429] | 0 / 0 |
| DIF (features only) | 1001 | 0.0818 (912) | 0.0629 | 0.0722 | 0.377 | 0.3395 | [0.3128, 0.3656] | 0.305 / 0 |
| DIF | 1002 | 0.0870 (960) | 0.0839 | 0.0681 | 0.295 | 0.2685 | [0.2433, 0.2934] | 0.562 / 0 |
| DIF | 1003 | 0.0867 (955) | 0.1023 | 0.0619 | 0.304 | 0.2782 | [0.2534, 0.3050] | 0.539 / 0 |

- **Without a hand-given class, the Cypha classifier does most of the Beta's job:** regret 0.27–0.34 against Beta-pooled's 0.70–0.72, Brier 0.06–0.07 against 0.19. The Beta given the true task type is near perfect (regret 0.01–0.02), but no real situation comes with its type attached.
- **ECE misleads here.** Beta-pooled's predictions sit near its base rate, so it is calibrated (ECE 0.004–0.06) and useless (AUROC 0.62, below). Brier and regret show the difference.
- **LCI, DIF against Beta-pooled:** P_gain 1.000, n_eff 1,000, r 1, γ 1, σ 1, κ 1, but c = 0.5 because DIF's all-pairs ECE (0.063) is above Beta-pooled's (0.046). LCI 0.50.
- **Report-only, after the reads:** AUROC of P(applicable) over all pairs, and the mean P where the process applies / does not:

| Model | AUROC (seeds 1001 / 1002 / 1003) | Mean P, applies / does not |
|---|---|---|
| Beta-class | 1.000 / 1.000 / 1.000 | 0.97 / 0.09 |
| Beta-pooled | 0.630 / 0.624 / 0.624 | 0.31–0.35 / 0.24–0.29 |
| DIF | 0.933 / 0.967 / 0.984 | 0.75–0.87 / 0.14–0.17 |

- **Why DIF still hits IMPASSE 30–38% of the time:** it ranks well, but where a process does not apply its P averages 0.14–0.17, above the threshold θ = 0.1, so such processes are still tried. The gap is calibration at the low end, not discrimination. A calibration map of the classifier's P fit on its own labels, or θ from that map, is the next change to try; it needs its own registered read.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| K4a | No unbound op in the 144 processes | **Hit** |
| K4b | The planted bypass becomes "I don't know" in 100% | **Hit:** 100 / 100 |
| K4c | Replay hash unchanged | **Hit:** `18a61550bffc735c` |
| K5a1 | Dev: at most 2 of 520 answered; answer-nothing in ≥ 3 of 4 configurations | **Partial:** 3 answered (all right, in sample); answer-nothing in 3 of 4 |
| K5a2 | Test: wrong ≤ 2%, Wilson upper ≤ 4.5%, at most 2 right, every card with a reason | **Hit:** 0 right, 0 wrong, upper 0.019, 800 / 800 |
| K5b1 | Uniform: 6.0 tests; `differential_diagnosis` within 0.1; random about 31 | **Hit:** 6.000, 6.000, 31.55 |
| K5b2 | Skewed: `differential_diagnosis` ≥ 15% cheaper | **Hit:** 22.2% |
| K5b3 | Thompson ≥ 80% `differential_diagnosis` in the second half | **Hit:** 95.8% |
| K5c1 | `ach` ≥ 0.85; baseline at least 0.10 lower | **Partial:** 0.830; baseline 0.250 lower |
| K5c2 | SPRT errors ≤ 0.05 | **Hit:** 0.0385, 0.0345 |
| K5d1 | 1σ: CUSUM at most half of Shewhart's delay | **Hit:** 8.56 against 29.08 |
| K5d2 | 3σ: Shewhart at most CUSUM's | **Hit:** 1.71 against 2.24 |
| A1 | Beta-class ECE ≤ 0.05 | **Hit:** 0.005–0.006 (chosen) |
| A2 | Beta-pooled ECE ≥ 0.20 | **Miss:** 0.004–0.019; calibrated but uninformative |
| A3 | DIF ECE ≤ 0.10 and regret ≤ 1.1 × Beta-class's | **Miss:** ECE 0.082–0.087, but regret 15–30 × |
| A4 | DIF flags ≥ 80% of the new type, ≤ 10% of known; Betas none | **Partial:** 31–56% new, 0% known; Betas none |

11 hits, 3 partial, 2 misses.

### Decisions (by the registered rules)

1. **K4: the bindings are adopted.** Every check group passes, the planted bypass is caught, and the K2 replay is unchanged.
2. **K5:** reported; nothing in Cypha's defaults changes. `differential_diagnosis` is the better fault finder when faults and test costs are uneven; CUSUM for small shifts, Shewhart for large ones; `ach` beats counting consistent evidence by 0.25.
3. **DIF applicability is not adopted** (rule 3: its regret is far above 1.1 × Beta-class's, and LCI is 0.50). The hook stays in the interpreter, unset by default; the Beta stays as K3 left it. The classifier is the only one of the three that works without a hand-given class, and its miss is low-end calibration (above).

### Checks

- The nine CTests of HANDOFF §4.4 pass (`native_cyphalm_trie_selftest`, `native_fm_infinigram_test`, `native_epistemic_test`, `native_thought_test`, `native_geometry_test`, `native_ecs_lci_test`, `native_scribe_selftest`, `native_cyphalm_neural_smoke`, `native_class_context_test`), and the new `native_thought_bindings_test` (120 checks).
- `bindings_cypha.hpp` uses no POSIX header (`unistd.h` broke the Windows build once before); the K5 tool builds against `cypha_core` with no warnings from these files.

### What is still open

- `fermi_estimate` and the planning processes have no benchmark with known answers in the repo.
- `cooperative_reply`'s relation and manner checks are plain checks for now; SCRIBE's R axis needs a question vector per answer.
- The DIF classifier's calibration map (or θ from it), with a registered read.
- Traces to the experiment DB (the journal binding writes JSONL; `experiment_db.hpp` is not wired).

### How to reproduce

```bash
cmake --build <build> --target thought_bindings_test thought_k5 thought_test
ctest --test-dir <build> -R "native_thought_test|native_thought_bindings_test"
J=docs/thought/processes.json
<build>/thought_k5 qa $J T_on.rows.tsv [--read-test]   # B4c's rows of cyphalm_trie qa
<build>/thought_k5 chains $J
<build>/thought_k5 hyps $J
<build>/thought_k5 monitor $J
<build>/thought_k5 applicability $J                    # 21 s here
```
