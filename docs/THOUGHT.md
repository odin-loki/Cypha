# Cypha thought processes: a learnable library of human thinking and work

**Odin Loch** · 2026-10-27 · Track K of [PLAN.md](PLAN.md)

This gives Cypha a library of the processes people use to think and work, written so Cypha can run them, choose between them, and improve them with the same online statistics it already uses.
- It covers reasoning, problem solving, deciding, planning, learning, metacognition, creativity, science, engineering, quality, fault finding, intelligence analysis, communication, negotiation, software, management, finance, law, teaching, estimation, perception, data science, and AI/ML engineering, including Cypha's own daily upkeep.
- It uses the same statistics as the rest of Cypha: Beta and Dirichlet counts, Welford running moments, the DIF classifier, Thompson sampling, ECS and LCI.

| File | What it is |
|---|---|
| [`thought/processes.yaml`](thought/processes.yaml) | The library: **142 processes in 25 categories, 840 steps**, machine-readable |
| [`thought/INDEX.md`](thought/INDEX.md) | Generated index: every process with its source, when to use it and its core maths |
| [`../scripts/thought_check.py`](../scripts/thought_check.py) | Validator, index builder, and UML export (`--uml <id>` for PlantUML, `--mermaid <id>`) |

---

## 1. What a thought process is

### 1.1 The unit: a TOTE loop

Every process is built from **Test-Operate-Test-Exit** loops (Miller, Galanter & Pribram 1960):
1. Test the gap between state and goal.
2. If it is too big, operate.
3. Test again.
4. Exit when the gap is small enough or further work isn't worth it.

Because all thinking reduces to these loops, a single learning machinery covers all of it.

### 1.2 The structure: an option with a control graph

A process `P` is a tuple `⟨I, S, s₀, E, op, π, β, out⟩`:

| Part | Meaning | In the YAML |
|---|---|---|
| `I` | When it applies (initiation set) | `when`, plus a learned applicability classifier (section 3.1) |
| `S, s₀` | Steps and the start step | `steps`, `start` |
| `E` | Allowed transitions between steps | each step's `next` |
| `op` | What each step does: a primitive (section 7) or `call:<process>` | `op`, `do` |
| `π` | Which transition to take: a guard or a learned policy | learned (section 3.2) |
| `β` | When to stop | `stop`, plus the value-of-computation rule (section 3.7) |
| `out` | What it returns and the feedback it gets | `feedback` |

This is the options framework of hierarchical reinforcement learning (Sutton, Precup & Singh 1999). A process is a temporally extended action with its own start condition, internal policy and stop rule. Processes calling processes give a hierarchy, as in hierarchical task networks.

**UML.** Each `steps` graph is a UML state or activity diagram: steps are actions, `next` lists are control flow, `decide` steps are decision nodes and `return` steps are final nodes. `thought_check.py --uml ach` prints it as PlantUML text, and `--mermaid ach` as Mermaid. These are rendered on demand, not stored as images.

### 1.3 Running a process

```text
run(P, goal, budget):
    if P(applicable | x) < θ: return IMPASSE
    predict p̂ = P(success | P, task class)                 # stored for calibration (3.8)
    s = s₀; trace = []
    while budget > 0:
        result = do(op(s))                                  # primitive, or run(child, ..., share of budget)
        trace.append((s, context, result, cost))
        if s is final: break
        if not VOC_positive(P, s): break                    # 3.7
        s = choose_next(P, s, context)                      # guard if it decides, else learned π (3.2)
    R = feedback(P, outcome)                                # section 8
    learn(P, trace, R, p̂)                                   # section 3
    return outcome, trace
```

A shared **working memory** (the global workspace, process `global_workspace`) holds goals, partial results and the current focus. Every trace is written to the experiment DB, which serves three purposes:
- learning (section 3);
- mining new processes (section 6);
- audit.

---

## 2. The library at a glance

| Category | n | Examples |
|---|---:|---|
| core | 8 | TOTE, System 1 / System 2 arbitration, impasse → subgoal, chunking, global workspace, satisficing, value of computation, predictive perception |
| reason | 8 | deduction, induction, abduction, analogy, causal reasoning, categorisation, Toulmin argument, counterexample search |
| solve | 7 | Polya, means-ends, generate-and-test, divide and conquer, working backwards, local search, constraint search |
| decide | 6 | expected utility, sequential evidence (SPRT), AHP, minimax regret, recognition-primed decision, pre-mortem |
| plan | 4 | HTN planning, critical path and PERT, contingency planning, value-per-time scheduling |
| learn | 6 | spaced repetition, Kolb cycle, deliberate practice, after-action review, reflective cycle, habit formation |
| meta | 4 | confidence monitoring and abstention, self-explanation, goal monitoring, error review |
| create | 5 | diverge and converge, SCAMPER, TRIZ, morphological analysis, random stimulus |
| discover | 2 | gap closing (asking ladder), conjecture and test (the full set is in [DISCOVERY.md](DISCOVERY.md)) |
| science | 5 | scientific method, systematic review, experiment design, peer review, Bayesian update |
| eng | 6 | engineering design, design thinking, V-model, tolerance stack-up, machining process plan, FMEA |
| quality | 7 | job safety analysis, PDCA, DMAIC, statistical process control, reliability-centred maintenance, five whys, fishbone |
| diagnose | 4 | half-split fault finding, differential diagnosis, fault tree, debugging |
| intel | 7 | OODA, intelligence cycle, analysis of competing hypotheses, MDMP, red teaming, situational awareness, signal detection |
| comm | 6 | reading (SQ3R), writing, explaining, clarifying questions, conversation repair, Grice check |
| social | 3 | perspective taking, principled negotiation, ethical check |
| software | 4 | TDD, code review, bisect, incident response |
| manage | 3 | sprint, Kanban flow, risk register |
| finance | 2 | Kelly sizing, cost-benefit with Monte Carlo |
| law | 2 | standard of proof, case-based reasoning |
| teach | 2 | scaffolded teaching, Socratic questioning |
| estimate | 3 | Fermi estimation, forecasting (outside then inside view), reference-class forecasting |
| perceive | 2 | where to look (salience), change detection (CUSUM) |
| data | 17 | CRISP-DM, exploratory analysis, cleaning, quality scorecard, hypothesis tests, A/B tests, regression, time series, clustering, anomaly detection, observational causal effects, honest charts, feature engineering, survey sampling, record linkage, near-duplicates, ETL with reconciliation |
| ai | 19 | ML workflow, cross-validation, hyperparameter search (successive halving), error analysis, model evaluation, ablation, leakage check, drift monitoring (PSI, KS, CUSUM), active learning, Cypha's safe online update, model selection (AIC, BIC, MDL), retrieval pipeline, teacher queries, benchmark hygiene, experiment tracking, slice audit, bandit loop, **daily upkeep**, model card |

Every process carries its source (`ref`) and its core maths (`math`). Some examples:
- RPN = S·O·D (FMEA);
- d′ = z(H) − z(FA) (signal detection);
- t_e = (o + 4m + p)/6 (PERT);
- Wald's SPRT bounds;
- the Kelly fraction;
- CUSUM;
- Weibull R(t);
- AHP's consistency ratio.

The full list is in [`thought/INDEX.md`](thought/INDEX.md).

---

## 3. The standard learners: how every process improves

Every process gets these learners automatically. A process's `learn` field lists only the extras.

### 3.1 When does it apply?

- **Model.** `P(applicable | x)` is a Cypha DIF classifier per process, trained online on situation features x: task type, question type, domain, epistemic state, and features of the input.
- **Labels.** A run that ended in success or a useful partial result is a positive. A run that ended in IMPASSE straight away is a negative.
- **OOD.** The classifier's OOD flag marks situations where no process has been tried before. These go to `impasse`.

### 3.2 Which step comes next?

- **Graph.** The YAML fixes which transitions are allowed.
- **Guards.** Where a step is a `decide` with a measurable guard (a threshold on ECS, VOC or a z-score), the guard decides.
- **Learned choice.** Where several next steps are allowed and no guard decides, a **Dirichlet–multinomial** learns which branch works:
  - Per (step, context bucket): `α_j ← α_j + e_k · R` for the branch j taken, with prior α_j = 1.
  - The policy samples `θ ~ Dir(α)` and takes `argmax θ_j` (Thompson sampling).
  - Where context features matter more than a few buckets can capture, the bucket table is replaced by a contextual DIF classifier over branches.

### 3.3 Did each step help, and what did it cost?

| Quantity | Learner |
|---|---|
| Step success | `Beta(a, b)` per (step, context), updated with credit (section 3.6) |
| Step cost | Welford mean and variance of log cost (time, compute, queries) |
| Step gain | Welford of the improvement in the process's goal measure after the step; used by VOC |

### 3.4 How good is the whole process?

| Quantity | Learner |
|---|---|
| Success | `Beta(a_P, b_P)` per (process, task class) |
| Cost | Welford of log cost per (process, task class) |
| Utility | `U_P = p_P · G − C_P`: ACT-R's expected gain, with G the value of the goal and p, C from the posteriors |
| Utility, incremental form | `U ← U + α (R − U)`: ACT-R's utility learning rule, kept as a fast-adapting second estimate for drifting tasks |

### 3.5 Choosing between processes

For a goal with several applicable processes:

1. **Draw.** For each candidate, sample `p̃ ~ Beta(a_P, b_P)` and `C̃` from the cost posterior, and compute `Ũ_P = p̃ · G − C̃`.
2. **Choose** `argmax Ũ_P`. This is Thompson sampling, so it explores in proportion to its uncertainty.
3. **Floor.** Every applicable process keeps at least ε = 2% of choices, so a process that was unlucky early can still recover.
4. **Regret** is tracked against the best process in hindsight on held-out tasks (section 10).

This is how Cypha learns, per situation, whether Polya or means-ends works better, or whether half-split beats differential diagnosis.

### 3.6 Credit assignment across a trace

A run returns a reward R ∈ [0, 1] at the end (section 8). Each step k in the trace gets credit by an eligibility trace, as in TD(λ):

`e_k = λ^(T − k)`, λ = 0.8 by default. Then `a_k ← a_k + e_k · R` and `b_k ← b_k + e_k · (1 − R)`.

- **Steps with their own check** (a `test` or `verify` step) also get a direct update from that check.
- **Sub-process calls** pass the parent's credit into the child's own Beta, so a sub-process learns from every parent that uses it.

### 3.7 When to stop

Before each step after the first, the run is checked against three rules:
- **Value of computation** (Russell & Wefald 1991): `VOC = E[gain of the next step] − cost`, both from the Welford estimates in 3.3. Stop when VOC ≤ 0.
- **Aspiration**, for search-like processes (Simon's satisficing): stop at the first candidate with value ≥ A, where `A ← A + λ (v − A)` per episode.
- **Budget:** each call carries a budget. A child gets the share `φ = E[child cost] / E[parent remaining cost]`, capped at 0.5 of what the parent has left.

### 3.8 Is it honest about its own chances?

- **Prediction.** Before running, a process predicts `p̂ = E[Beta(a_P, b_P)]` for this task class.
- **Scoring.** After the run, `(p̂, R)` is scored with Brier and ECE per process, and an isotonic map corrects any bias.
- **Use.** The corrected p̂ is what `dual_process` and `confidence_monitor` use. A process that is overconfident about itself gets corrected the same way an answer would.

---

### 3.9 Day-to-day use: the AI and data-science processes

The `ai` and `data` categories are the processes Cypha uses on itself and on its data every day.

- **`daily_upkeep` runs once a day.** In order:
  1. canary suites;
  2. drift checks (`drift_monitoring`: PSI, KS and CUSUM);
  3. an audit sample of new facts and abstentions;
  4. calibration refresh;
  5. the gap map and frontier report;
  6. the curiosity scheduler's discovery budget;
  7. a dated journal for Odin.
- **Admissions pause** if the audit's Wilson lower bound falls below 0.90.
- **Every incoming item** goes through `online_update`. It predicts first (prequential scoring), learns, checks canaries and rolls back on a drop.
- **Every model or default change** runs `ml_workflow` → `model_evaluation` → LCI. It also uses `leakage_check` and `benchmark_hygiene`, so test sets stay honest, and `model_card` so the card stays true.
- **New data** runs `data_quality_check` before it can train anything or support an answer. Its scores update the source's trust Beta.
- **Choices that repeat** (thresholds, scorers, which teacher prompt to use) run as `bandit_loop`s.

## 4. Fast and slow thinking

**System 1** is the fast route: case recall and compiled rules. It covers:
- `recall_case`, including the chat index and the case memory;
- InfiniGram matches in CyphaLM;
- chunked rules.

**System 2** is the slow route: running a deliberate process.

**Arbitration** (`dual_process`) follows Daw, Niv & Dayan (2005): use the route whose predictions are more reliable for the situation.

- Use fast if `Var_fast + bias_fast² < Var_slow + λ · cost_slow`.
- The variances come from the routes' Beta posteriors per competence cell. The bias comes from fast-vs-slow disagreements seen so far.
- When both run and disagree, the fast route's trust for that cell drops. This is the "something feels off" signal.

**Chunking** (Laird, Rosenbloom & Newell 1986) turns repeated deliberation into System 1:
1. After 5 identical System 2 results for a situation class, a rule is compiled. It is tainted until tried.
2. It is used with System 2 still checking.
3. It is kept when the Wilson lower bound of its accuracy is ≥ 0.9 over at least 20 uses, and dropped below 0.8.

Fast answers never bypass the answer-mode rules. A chunked rule's answers carry ECS evidence like any other.

---

## 5. Composition, impasses and hierarchy

- **Calls.** `op: "call:<id>"` runs another process as a step. The library has 43 such calls (for example, `dmaic` calls `ishikawa` and `spc_control`, `ach` is called by the intelligence cycle, `five_whys` is called from five places, and `crisp_dm` chains `eda`, `data_cleaning` and `ml_workflow`).
- **Recursion** is allowed (`means_ends`, `htn_plan`) with a depth cap of 5.
- **Impasse** (Soar): when no process has `P(applicable) ≥ θ`, or all applicable ones fail, `impasse` makes "resolve the impasse" a goal. It then tries, by learned success:
  - decomposing the problem;
  - asking (store, teacher, human);
  - handing over to discovery mode ([DISCOVERY.md](DISCOVERY.md)).
- **Blackboard.** Processes running in parallel post to working memory. `global_workspace` picks what gets broadcast by `relevance × novelty × confidence`.

---

## 6. Learning new processes

The library is a starting point, and processes are also learned.

1. **From traces (process mining).** Count directly-follows pairs `|a > b|` between step types across runs. The heuristics-miner dependency measure (Weijters & van der Aalst 2003) is:

   `a ⇒ b = (|a > b| − |b > a|) / (|a > b| + |b > a| + 1)`

   - Propose an edge when `a ⇒ b ≥ 0.9` and `|a > b| ≥ 10`.
   - Propose removing an edge never taken in 100 runs.
2. **From text.** Procedures written in documents (numbered steps, imperative sentences, manuals, standards) are read into step lists. Verbs map to primitives through a verb lexicon: "measure" → `measure`, "check" → `test`, "if … then" → `decide`. They enter the library tainted, as hypothesis processes.
3. **From demonstrations.** Odin's own work logs (events with timestamps) mined the same way as 1. This captures how he actually works, such as toolmaking or fault finding on the job.
4. **Structural edits:** add or remove an edge, split a step that does two things, merge steps always taken together. Each edit is scored by description length:

   `L(model) + L(traces | model)`, in bits

   - L(model) counts steps and edges.
   - L(traces | model) is the negative log-probability of the logged traces under the learned transition probabilities.
   - An edit that lowers the total explains the logs with a simpler or better model.
5. **Promotion.** Every new process or structural edit is a change like any other. It is promoted only through LCI ([PLAN §8](PLAN.md#8-lci-learning-certainty-index)): it has to beat the current version on held-out tasks without hurting canaries.

---

## 7. Primitives: what the steps call in Cypha

| Primitive | Cypha component |
|---|---|
| `perceive`, `read` | CyphaLM byte input; readers (`qa_infobox` extraction, the tagger E1 and parser E3 when built) |
| `retrieve` | Fact store with ECS opinions; only Known/Believed in answer mode |
| `recall_case` | `chat_index`, case memory, InfiniGram matches, chunked rules (System 1) |
| `match` | Pattern layer; structure mapping |
| `compare`, `measure` | Distances, differences, metrics; SCRIBE axes |
| `classify` | Cypha DIF classifier, with calibrated confidence and the OOD flag |
| `estimate` | DIF regressors with intervals; forecasting module; Fermi combination |
| `decompose`, `combine`, `transform` | Library operators per domain; HTN methods |
| `generate` | DISCOVERY generators G1–G20; CyphaLM sampling |
| `evaluate`, `score`, `rank` | SCRIBE, ECS, EIG, process-specific scores |
| `select`, `decide` | Thompson sampling (3.5); guards over the epistemic state (`epistemic.hpp`) |
| `simulate` | Possible-worlds sandbox ([DISCOVERY §4.4](DISCOVERY.md#44-possible-worlds-the-hypothetical-sandbox)) |
| `test`, `verify`, `falsify` | DISCOVERY tests T1–T10; held-out checks |
| `infer` | Rules with reliabilities; depth cap 3 in answer mode |
| `explain`, `communicate` | Cards (`epistemic.hpp`) with CyphaLM wording |
| `ask` | GGUF teacher (`gguf_mine.py`); the Ask-Odin queue |
| `record`, `update` | Experiment DB, journal; Beta / Dirichlet / Welford / DIF / ECS updates |
| `monitor`, `wait` | CUSUM and SPC rules; triggers |
| `schedule`, `act` | Scheduler; tools and REST actions (after `ethical_check` when the action touches the world) |

---

## 8. Feedback: where the reward R comes from

The reward is always measured, never self-assessed. Every source maps to R ∈ [0, 1] and runs through the same updates.

| Source | R | Example processes |
|---|---|---|
| Verified correctness against a frozen test | 1 right, 0 wrong; abstaining earns the abstention value of the cell (≥ 0 and below 1, so honest abstention beats a wrong answer, PLAN rule 9) | Q/A, categorisation, diagnosis |
| A check inside the process | Pass rate of its `test`/`verify` steps | Polya, TDD, V-model |
| A metric change | Clipped, normalised improvement, promoted only through LCI | PDCA, DMAIC, writing (SCRIBE) |
| Resolution of a forecast | 1 − Brier | forecasting, risk register, PERT |
| Outcome of an action in the world | Measured result | maintenance, scheduling, negotiation |
| Human rating | Odin's 1–5 rating mapped to [0, 1] (R5 in PLAN §3.4); used sparingly | explanations, research journal |
| Calibration | 1 − squared error of the process's own prediction p̂ | every process (3.8) |

---

## 9. Safety and honesty

- **Not answers.** Thought traces are working, not answers. An answer still needs ECS evidence and passes `confidence_monitor` and `cooperative_reply`.
- **Speculation is labelled.** Processes that produce speculation (abduction, analogy, random stimulus, the discovery processes) write to the hypothesis store and are labelled ([DISCOVERY §1](DISCOVERY.md#1-two-modes-and-the-firewall)).
- **Actions in the world** go through `ethical_check` first. Irreversible or high-impact ones also go to a human.
- **Learning that changes the library** (new processes, edits, chunked rules) is promoted only through LCI, and stays tainted until then.

---

## 10. Measuring it

**Per process**
- success rate with a Wilson interval, per task class;
- cost;
- calibration of p̂ (ECE, Brier);
- learning curve;
- how often it is selected;
- regret against the best process in hindsight.

**Library**
- **Coverage:** the share of tasks where some process applies.
- **Impasse rate.**
- **System 1:** hit rate and accuracy.
- **Chunks:** promotion and demotion rates.
- **Mined edits:** precision on held-out traces.
- **Selection regret** over time.

**Benchmarks**, each with known answers:

| Benchmark | Processes tested | Score |
|---|---|---|
| T-QA-wiki (200 blind questions) | `dual_process` over `chat_index` and the fact store, `confidence_monitor`, `cooperative_reply` | right / abstained / wrong (PLAN rule 9) |
| Planted-fault chains (synthetic) | `half_split` vs random testing vs `differential_diagnosis` | tests to isolate; repair success |
| Fermi questions with known answers | `fermi_estimate` | coverage of 90% intervals; log error |
| Resolved forecasting questions | `superforecast`, `reference_class` | Brier; calibration |
| Simulated process streams with planted shifts | `spc_control`, `change_detection` | detection delay at a fixed false-alarm rate |
| Planning puzzles (blocks world, Tower of Hanoi) | `means_ends`, `htn_plan`, `backtracking_search` | solved; nodes expanded |
| Synthetic competing-hypothesis cases | `ach`, `sprt_decide` | correct hypothesis; evidence used |

---

## 11. Worked example: answering a factoid question

1. **Selection.** For the goal "answer: who designed the AK-47?", `dual_process` is selected: P(applicable) = 0.97 for questions.
2. **Fast route.** `recall_case` queries `chat_index`. It returns `ABSTAIN / NO_EVIDENCE`.
3. **Slow route.** The gate goes to `slow`. Thompson sampling picks the fact-store pipeline over decomposition: draws 0.41 against 0.12, given past Beta posteriors per question class.
4. **Gate.** The pipeline returns "Mikhail Kalashnikov", with ECS P 0.93 and grade High. `confidence_monitor` maps it through the cell's isotonic calibration to 0.91, above the cell threshold 0.80, so it answers with an answer card.
5. **Check.** `cooperative_reply` checks quantity, quality (ECS evidence present), relation and manner, then sends.
6. **Feedback.** On a frozen test the answer is verified right, R = 1. With λ = 0.8:
   - `dual_process`'s Beta for person-questions gains 1 success;
   - the `slow` branch's Dirichlet count gains e = 0.8²;
   - the fast route's Beta for this cell gains a miss for abstaining where slow succeeded.
   - Calibration logs (p̂ = 0.91, R = 1).

   Had the slow route also been unsure, the answer would have been an "I don't know" card. R would have been the cell's abstention value, which is higher than the 0 a wrong guess earns.

---

## 12. Track K: work items

| ID | Work | Gate |
|---|---|---|
| K1 | Library, schema and validator (done 2026-10-27: 142 processes, 0 errors) | `thought_check.py` passes |
| K2 | C++ interpreter: steps, guards, calls, budgets, traces to the experiment DB (`native/include/cypha/thought/`) | Replays a trace bit-identically; CTest |
| K3 | Standard learners: Beta, Dirichlet, Welford, DIF applicability, Thompson selection, TD(λ) credit, VOC stop, calibration | Known-answer CTests; regret falls on a synthetic bandit of processes |
| K4 | Primitives bound to existing tools (`chat_index`, `qa_infobox`, ECS, LCI, epistemic, SCRIBE, `gguf_mine.py`) | Each primitive has a test |
| K5 | First executable processes: `dual_process` + `confidence_monitor` + `cooperative_reply` on T-QA-wiki; `half_split` on planted faults; `fermi_estimate`; `sprt_decide`; `ach` | Benchmarks in section 10 |
| K6 | Chunking (fast rules from slow results) | Chunk accuracy ≥ 0.9; System 1 hit rate rises with no rise in wrong answers |
| K7 | Process mining and MDL structural edits; promotion through LCI | Recovers a planted process from synthetic logs; edit precision on held-out traces |
| K8 | Text-to-process reader (procedures in documents → tainted processes) | Precision of extracted steps on 20 hand-checked procedures |
| K9 | Odin's work logs → mined personal processes (when logs exist) | Odin confirms the mined process matches how he works |

**Dependencies.**
- K5 needs S1 (reason codes, done) and the fact store (F1, in progress).
- K7 needs K2's traces.
- The discovery processes in [DISCOVERY.md](DISCOVERY.md) run as library processes once K2 and K3 exist.

---

## 13. References

**Control loops, architectures and arbitration**
- Miller, Galanter & Pribram (1960). *Plans and the Structure of Behavior*. (TOTE.)
- Sutton, Precup & Singh (1999). Between MDPs and semi-MDPs: a framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*.
- Erol, Hendler & Nau (1994). HTN planning: complexity and expressivity. *AAAI*.
- Anderson et al. (2004). An integrated theory of the mind. *Psychological Review*. (ACT-R utility learning.)
- Laird (2012). *The Soar Cognitive Architecture*. MIT Press. (Impasses.)
- Laird, Rosenbloom & Newell (1986). Chunking in Soar. *Machine Learning*.
- Baars (1988). *A Cognitive Theory of Consciousness*. (Global workspace.)
- Kahneman (2011). *Thinking, Fast and Slow*.
- Daw, Niv & Dayan (2005). Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. *Nature Neuroscience*.

**Bounded rationality and sequential decisions**
- Simon (1956). Rational choice and the structure of the environment. *Psychological Review*.
- Russell & Wefald (1991). *Do the Right Thing*. MIT Press. (Value of computation.)
- Wald (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*.
- Thompson (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*.

**Process mining**
- van der Aalst (2016). *Process Mining: Data Science in Action*. Springer.
- Weijters & van der Aalst (2003). Rediscovering workflow models from event-based data using Little Thumb. *Integrated Computer-Aided Engineering*.

The sources of the individual processes are in each `ref` field and in [`thought/INDEX.md`](thought/INDEX.md).
