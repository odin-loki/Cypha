# K2/K3: thought-process interpreter and standard learners

**Date:** 2026-10-27
**Author:** Odin Loch
**Scope:** THOUGHT.md section 12, items K2 (interpreter) and K3 (standard learners), against sections 1.3, 3.1-3.8 and 5. The library is K1 (`docs/thought/processes.yaml`, 142 processes).

## What is implemented

Header-only, namespace `cypha::thought`, standard library plus `third_party/nlohmann/json.hpp`:

| File | Contents |
|---|---|
| `scripts/thought_check.py` | New `--json OUT`: writes the validated library as JSON. Nothing else changed. |
| `docs/thought/processes.json` | The library as JSON (151 KB; 142 processes, 840 steps, 43 calls). Regenerate with `python3 scripts/thought_check.py --json docs/thought/processes.json`. |
| `native/include/cypha/thought/learners.hpp` | Portable random draws; Beta and Dirichlet posteriors; Welford and log-cost Welford; `U = p*G - C`; Thompson selection with the epsilon floor; ACT-R utility; TD(lambda) credits; VOC stop; per-process calibration log (Brier, ECE, isotonic correction); Wilson bounds (from `eval/ecs.hpp`) |
| `native/include/cypha/thought/interpreter.hpp` | `Library` (load and check), `WorkingMemory`, primitive registry, `Interpreter::run` and `solve`, trace records, learning after each run |
| `native/tools/thought_test.cpp` | 5,534 checks, nonzero exit on failure |
| `native/CMakeLists.txt` | One commented block at the end: target `thought_test`, CTest `native_thought_test` (TIMEOUT 300) with the JSON path as its argument |

## API in brief

Learners:

- `uniform01`, `normal01` (Marsaglia polar) and `gamma_sample` (Marsaglia-Tsang, with the U^(1/k) boost for k < 1), all on `std::mt19937_64`.
- `Beta{a,b}`: `update(r, w)` is the fractional update a += w*r, b += w*(1-r); `mean`, `variance`, `sample`, `wilson`.
- `Dirichlet(k)`: `update(j, w)`, `mean`, `sample`, `choose` (argmax of a draw).
- `Welford` (mean, n-1 variance, standard error) and `LogWelford` (Welford of ln cost; `expected` is the lognormal mean, `sample` draws from the cost posterior).
- `utility`, `thompson_select(arms, G, eps, rng)`, `ActRUtility`, `td_credits(T, lambda)`, `voc`, `voc_stop`, `CalibrationLog` (`add`, `brier`, `ece`, `corrected`), `wilson_interval`.

Interpreter:

- `Library::load(path)`, `find`, `check(id)` and `validate()`, using the same rules as `thought_check.py`. `add_process` takes extra (synthetic, later mined) processes.
- `Interpreter(lib, Config, seed)`. `bind(key, Primitive)`, where the key is an op name (`"test"`) or `"<process>.<step>"`; the step binding wins. A `Primitive` takes `(Step, StepContext, WorkingMemory)` and returns `StepResult{ok, value, cost, chosen_next, done, gain, impasse}`.
- `run(pid, task_class, wm, {budget, G, reward})` returns `RunResult{trace, reward}`. `solve(Goal{task_class, candidates}, wm, opt)` selects by Thompson sampling, then runs the choice, or returns IMPASSE.
- `RunTrace` is one record per step: op, ok, value, cost, gain, branch taken and how it was chosen (`final`, `guard`, `single`, `learned` or `stop`), child index, note and credit. Child runs nest inside it. `trace_text` gives an exact form using hex floats, `fnv1a` hashes it, and `set_trace_sink` is the hook for the experiment DB.
- `unbound(pid)` lists the ops a process (and its callees) still has no binding for. This is K4's work list.

## Test results

- Configure 1.2 s. `cmake --build /root/cb_ecs --target thought_test -j1` (Ninja, Release, g++ 13.3) took 13 s. `ctest -R native_thought_test` passed in 0.34 s, with 5,534 checks and 0 failures.
- Warnings: none with `-Wall -Wextra -Wpedantic` on g++ 13.3 or clang 18.1. ASan and UBSan were clean in an earlier build of the same code; I did not rerun them for this report.
- **Learner known answers:**
  - Beta(1,1) plus 3 successes and 1 failure gives mean 2/3. Fractional credit is exact.
  - Beta(2,5) sample moments match.
  - Dirichlet means after updates of 2 and 0.5 are 3/5.5, 1/5.5 and 1.5/5.5. Sample means match.
  - Welford: 5 and 32/7 on the textbook set. On 5,000 points at an offset of 1e6, it agrees with a long-double two-pass computation to a relative 1e-9.
  - TD(0.8) credits on a 4-step trace are 0.512, 0.64, 0.8 and 1.
  - Wilson 8/10 gives (0.490162, 0.943318).
  - VOC is +0.4 (continue) or -0.15 (stop), and the sign flips with G = 10.
  - ACT-R gives 0.1, 0.19, 0.171.
  - The streaming ECE and Brier equal `eval::expected_calibration_error` and `brier_score` to 1e-12.
  - The floor gives the bad arm 2.0% of the choices.
- **Library:** all 142 processes instantiate with no unbound op. A smoke run of each with every primitive stubbed ended 142 done, 0 on budget and 0 in error or impasse. The maximum depth stayed at 5 or less.

## Benchmark numbers

| Benchmark | Result |
|---|---|
| `half_split`, 64-component chain, one planted fault, 200 seeds | Halving: **6.000 tests** on every seed (log2 64), fault found and confirmed every time. Random split point: 7.155. Random probing one component at a time: 31.33. |
| `sprt_decide`, Bernoulli p0 = 0.4 vs p1 = 0.6, alpha = beta = 0.05, 2,000 runs per hypothesis | Type I **0.0275**, type II **0.0380**. The exact value for this walk is 0.0376: each observation moves the LLR by ±ln 1.5, so it needs a net ±8 steps, which is gambler's ruin. Wald's bound is 0.05. Mean evidence: 37.6 observations under H0, 37.1 under H1. |
| `tote` | Closes a gap of 10 in 4-10 operations. With a futile operator (gain 0, cost 1), run 1 stops on budget at cost 50 and run 2 stops on VOC at cost 0. A worthwhile operator never trips VOC. |
| Learned branch (a `decide` with no guard) | Dirichlet mean P(good) = 0.996; good was chosen in 100% of the last 100 runs. |
| Selection: 3 processes, G = 10. Cheap p = 0.60, C = 1 (U = 5.0). Best p = 0.85, C = 2 (U = 6.5). Sure p = 0.95, C = 5 (U = 4.5). 2,000 episodes. | Cumulative regret at 250, 500, 1000, 1500 and 2000 episodes: 138.0, 163.0, 243.0, 289.0, 335.5. **First 1000 episodes: 243.0. Second 1000: 92.5.** First 500: 163.0; last 500: 46.5. Best arm chosen in **89.5%** of all episodes and **94.6%** of the second 1000 (counts 173 / 1789 / 38 for cheap / best / sure). |
| Calibration of p-hat in the selection run | **ECE 0.0154** over all 2,000 predictions (Brier 0.1391), 0.0171 for the best arm, and 0.0314 over the last 1000. |
| Replay | Seed 77, 300 episodes, run twice: 100,827 bytes of trace text, bit-identical, FNV-1a `18a61550bffc735c`. The learner state is identical too, and seed 78 differs. The same hash came from gcc -O2 and clang -O0. |

Regret in the selection run flattens but never reaches zero, because of the epsilon floor. Each of the two worse arms keeps 2% of choices, which costs 0.02 × 1.5 + 0.02 × 2.0 = 0.07 per episode, or 35 per 500 episodes. So most of the 46.5 in the last 500 is the floor, and a best-arm share of 96% is the ceiling.

## Spec ambiguities and what I chose

1. **JSON export.** Key order follows the YAML. The export refuses step ids, `next` targets or `start` that are not strings, because YAML 1.1 reads a bare yes, no, on or off as a boolean. The C++ `Library::check` uses the same rules as the validator, plus "a final step is `return`".
2. **Randomness.** No `std::*_distribution` is used, because they differ across standard libraries. Uniforms come from the top 53 bits of one draw. Beta and Dirichlet are drawn as normalised Gamma draws.
3. **Binding.** The library's ops are generic (`test`, `decide`, `perceive`, ...), so a callback can be bound to one step of one process as well as to an op. `return` is built in and costs nothing.
4. **Guards (3.2).** Any step's callback may set `chosen_next`, not only a `decide` step's. A branch outside `next` is an Error, not a silent fallback. With no guard and one successor, that successor is taken. With several, the Dirichlet for (process, step, task class) is sampled. The task class is the context bucket. The Dirichlet is also credited for branches the guard chose, so it has a prior if a guard is later dropped.
5. **Stopping (1.3, 3.7).**
   - Budget and VOC are checked only before non-final steps, so a run that reaches its final step always ends cleanly. The check is made before a step, so one step can overrun the budget.
   - VOC is checked before each step after the first, from the Welfords of the step about to run.
   - THOUGHT writes VOC = E[gain] - cost. I used **G · E[gain] - E[cost]**, so that both terms are in cost units; gain is in the units of the goal measure.
   - VOC cannot stop a step before it has 5 observations of both gain and cost. A step that never reports a gain is never stopped by VOC.
   - E[cost] is the lognormal mean exp(mu + s^2/2).
   - There is also a cap of 1,000 steps per run (outcome `max_steps`).
6. **Budget share.** phi = E[child cost] / max(E[parent run cost] - spent so far, E[child cost]), capped at 0.5 of what the parent has left. With no cost data the cap applies. A child inherits the task class.
7. **Depth cap 5.** The top level is depth 0. A call that would create depth 6 fails as a step (`ok = false`, note `depth_cap`), and the parent goes on.
8. **Reward (section 8).** The caller passes a `RewardFn`. By default, R is the pass rate of the `test` and `verify` steps anywhere in the run tree. With no such steps, R is 1 if the run reached a final step and 0 otherwise. R is clamped to [0, 1].
9. **Credit (3.6).**
   - Each run's own steps get e_k = lambda^(T-k), with T the length of that run.
   - A child run's credits are scaled by the credit of the call step. The child's process Beta is updated with that credit as its weight; the top-level process is updated with weight 1.
   - `test` and `verify` steps also get a direct update of weight 1 from their own outcome.
   - Steps that failed for a structural reason (`depth_cap`, `no_primitive`) are not used for cost or gain.
10. **ACT-R (3.4).** The update is U <- U + alpha (R·G - C) with alpha = 0.1, so that U is on the same scale as p·G - C. THOUGHT writes R - U. It is updated on top-level runs only and is kept as a second estimate. Selection does not use it.
11. **Thompson floor (3.5).** With probability min(1, n·eps) the pick is uniform over the n arms, so each arm gets at least eps. The cost draw is mu* ~ N(mean of ln C, s^2/n), then C~ = exp(mu* + s^2/2). With one observation, C~ is that value; with none, it is `prior_cost` (0).
12. **Applicability (3.1).**
    - A Beta(1,1) per (process, task class) stands in for the DIF classifier. theta = 0.1, which THOUGHT does not give.
    - A top-level run is a positive label when R >= 0.5; I read that threshold as "success or a useful partial result". A run ending in IMPASSE is a negative label. That covers any IMPASSE, not only one "straight away". Runs with 0 < R < 0.5 give no label.
    - `solve()` returns IMPASSE when no candidate passes theta. It does not yet retry when all applicable processes fail, and it does not run the `impasse` process.
13. **Calibration (3.8).**
    - p-hat is the raw Beta mean before the run, logged per process for top-level runs only.
    - ECE uses 10 equal-width bins.
    - `corrected()` is a PAV fit over the bin accuracies, interpolated linearly. Nothing uses the corrected value yet; `dual_process` and `confidence_monitor` are K5.
14. **Traces to the experiment DB (K2).** Traces go to a `TraceSink` callback. Nothing is written to a DB yet.

## What is left

- **DIF applicability (3.1).** The per-process DIF classifier on situation features, with its OOD flag routing to `impasse`, and a contextual DIF over branches (3.2) in place of the task-class buckets.
- **K4.** Bind primitives to the real tools: CyphaLM, ECS, LCI, epistemic, SCRIBE and `gguf_mine.py`. `Interpreter::unbound(pid)` lists what each process still needs.
- **K5.** Run the first processes on real tasks.
- **K6.** Chunking.
- **K7.** Process mining and MDL edits. This needs the trace sink wired to the experiment DB.
- **Not done in K2/K3:**
  - the aspiration stop rule (3.7); for now a callback ends a run with `done`;
  - impasse resolution (decompose, ask, discovery mode);
  - parallel processes posting to the blackboard, and `global_workspace` broadcast;
  - saving and loading learner state.
- **Platforms.** Not tested on MinGW or Windows; the target uses the standard `cypha_mingw_apply_exe_link_options` call.
