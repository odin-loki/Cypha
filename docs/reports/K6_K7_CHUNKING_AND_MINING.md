# K6 and K7: chunking, process mining with MDL edits, and a calibration map for DIF applicability

**Odin Loch** · 2026-10-29 · THOUGHT.md §3.1, §4, §6, §12 (K6, K7) · follows [K4_K5_THOUGHT_BINDINGS.md](K4_K5_THOUGHT_BINDINGS.md)

(The design, the reads, the rules and the predictions below were written and committed before any K6 or K7 code or run. Results are added under them.)

**Question.**
- **K6:** can repeated System 2 results be compiled into fast rules (THOUGHT §4) that are right at least 90% of the time, so that System 1 answers more often without more wrong answers?
- **K7:** can a process be mined from its own traces (THOUGHT §6.1), and do MDL structural edits (§6.4) improve a process on traces they were not fitted on? Edits are promoted only through LCI (§6.5).
- **Calibration map (HANDOFF §5.23):** K5 found that the Cypha DIF classifier ranks applicability well (AUROC 0.93–0.98) but is poorly calibrated at the low end: where a process does not apply its P averages 0.14–0.17, above θ = 0.1, so it is still tried (IMPASSE 30–38%, regret 0.27–0.34). Does a calibration map of its P fix that?

**Rule 10.** A chunked rule is a learned shortcut, not CyphaLM. In answer mode its answer goes through the rule-10 guard and becomes an "I don't know" card; chunks speed up deliberate work (diagnosis, decisions, discovery), never answering. Mined processes and edits enter the library tainted and are promoted only through LCI.

---

## 1. K6: chunking (fixed before any code)

**Mechanism** (new header `native/include/cypha/thought/chunking.hpp`, THOUGHT §4 as written):
1. **Compile.** When the last 5 System 2 results for a situation class are identical, a rule "class → that result" is compiled. It is tainted.
2. **Trial.** A tainted rule's prediction is scored against System 2, which still runs and gives the answer.
3. **Keep.** The rule joins System 1 (it answers) when the Wilson lower bound of its agreement with System 2 is ≥ 0.9 over at least 20 checked uses.
4. **Audit and drop.** A kept rule is still checked on a random share `a` of its uses (System 2 runs too and gives the answer). It is dropped when its agreement over its last 20 checked uses falls below 0.8. After a drop, the class is either barred or may recompile after 10 identical results in a row (a dev choice, below).
- Chunk answers carry `source = "chunk"`, so the rule-10 guard turns them into "I don't know" cards in answer mode.
- The `dual_process` process of the library runs each task through the interpreter: `fast` (System 1: a kept or tainted chunk), `gate` (a kept chunk answers unless audited), `slow` and `run` (System 2), `compare` and `monitor` (the chunk's agreement).

**Benchmark: a stream of diagnosis tasks with recurring situation classes.**
- 80 situation classes, 16 possible results each. Each task draws a class by a Zipf law (s = 1.1) over a seeded random order of the classes.
- Class kinds (shares fixed): **deterministic** 40 (always result r); **mostly** 16 (r with probability 0.92, else a uniform other result); **ambiguous** 16 (r with 0.6, r′ with 0.4); **shifting** 8 (deterministic r until the stream's midpoint, then r′).
- **System 2** (a deliberate diagnosis) returns the task's true result with probability 0.98, else a uniform wrong one, at cost 10. A System 1 answer costs 1. System 2's noise is drawn from its own stream, so every arm sees the same draws.
- 4,000 tasks per stream. Dev streams: seeds 1–3. Held-out streams: seeds 1001–1003, read once.

**Arms.** `none` (System 2 always; System 1 never answers), `chunk` (the mechanism above), and `naive` (an ablation: compile after 1 result, never drop, no audits).

**Dev choice** (seeds 1–3): audit share `a` ∈ {0.05, 0.10, 0.20} × re-entry after a drop ∈ {barred, 10 identical}. Pick the highest System 1 hit rate whose pooled wrong rate is at most 0.005 above `none`'s; among choices within 0.01 hit rate of the best, the lower wrong rate.

**Reads (held-out, pooled over seeds 1001–1003).**
- **Chunk accuracy:** the share of System 1 answers (given by kept chunks) that match the task's true result, with its Wilson interval.
- **System 1 hit rate:** the share of tasks answered by System 1.
- **Wrong answers:** the share of tasks whose answer is not the true result; the paired difference `chunk − none` per task, with a bootstrap 95% interval (2,000 resamples of tasks).
- Also: cost per task, rules compiled / kept / dropped by class kind, and the delay from a shifting class's switch to the drop of its rule.
- **Answer mode:** one held-out stream in answer mode: every System 1 answer must become an "I don't know" card.

**Gate (THOUGHT §12, K6):** chunk accuracy ≥ 0.9, the System 1 hit rate rises above `none`'s (0), and the upper end of the wrong-answer difference's 95% interval is ≤ +0.005.

## 2. K7: process mining and MDL structural edits (fixed before any code)

**Mechanism** (new header `native/include/cypha/thought/mining.hpp`).
- **Mining (THOUGHT §6.1).** Directly-follows counts `|a > b|` over traces with a virtual start and end. Dependency `a ⇒ b = (|a > b| − |b > a|) / (|a > b| + |b > a| + 1)`. An edge is proposed when `a ⇒ b ≥ 0.9` and `|a > b| ≥ 10`. The heuristics miner's standard loop measures are part of the miner: a self-loop when `|a > a| / (|a > a| + 1) ≥ 0.9` with `|a > a| ≥ 10`; a length-2 loop when `(|aba| + |bab|) / (|aba| + |bab| + 1) ≥ 0.9` with `|aba| + |bab| ≥ 10`, which adds both edges.
- **The model for edits.** A process as a probabilistic automaton over step labels: nodes carry a label (or a short sequence of labels for a merged step), and the successors of a node have distinct first labels, so a trace replays along one path. Transition probabilities are the counts plus 0.5 per allowed edge and per escape. A transition the model does not allow takes the escape: its probability at that node, plus log2(|labels| + 1) bits to name the label, and replay restarts at a node with that label (the first in node order).
- **Description length (THOUGHT §6.4):** `L(model) + L(traces | model)`, in bits.
  - `L(model)`: log2(|labels| + 1) per label in a node, 2 · log2(nodes + 2) per edge, and w · log2(n_v) per free parameter (out-degree, counting the escape) of a node visited n_v times.
  - `L(traces | model)`: the negative log2 probability of the traces under the replay above.
- **Edits.** Remove an edge; add an edge for an escape seen at least 5 times; split a node by dividing its incoming edges into two groups (the two-group split of predecessors that best separates their successor distributions); merge two steps when a is always followed by b and b always preceded by a. Greedy: apply the edit with the largest drop in description length on the training traces, until none lowers it.
- **Promotion (THOUGHT §6.5).** An edited process replaces its start model only if LCI ≥ 0.95 on held-out traces: per-trace log-likelihood gains (ε = 0), the generalisation γ from training against held-out gain, σ from the three thirds of the held-out traces, and c = κ = 1.

**K7a, recovery from synthetic logs.**
- **Planted processes:** 6–12 labels, each used by one node; a backbone with 1–3 successors per non-final node; a back edge (loop) with probability 0.3 per process; branch probabilities from a flat Dirichlet. Traces are random walks of at most 60 events. 5% of traces get one random corruption (an adjacent swap, an inserted random label, or a deleted event).
- **Recovery:** the mined edge set equals the planted edges traversed at least 10 times in the log. 200 traces per process. Dev: processes with seeds 1–50. Held-out: seeds 1001–1100, read once.
- **The library:** each of the 144 processes, 200 random-walk traces with step ids as labels (uniform branch choice, calls as single steps). Recovery as above (report only).

**K7b, edit precision on held-out traces.**
- **Planted processes** as in K7a, but about 20% of the nodes reuse a label already used by another node, so a correct model needs two nodes with the same label.
- **The start model:** the planted process with 2 defects, each one of: an edge removed; a spurious edge added; two nodes with the same label collapsed into one.
- **Edits** are searched on 200 training traces; 300 held-out traces from the planted process score them.
- **Edit precision:** the share of accepted edits whose held-out log-likelihood gain is above 0 (each edit against the model before it), with its Wilson interval. Also: the share of planted defects repaired, and the share of cases promoted by LCI.
- **Dev:** seeds 1–50, for one choice: the parameter weight w ∈ {0.5, 1.0} (higher dev edit precision; tie within 0.01 → more defects repaired). Held-out seeds 1001–1100, read once.

**Gates (THOUGHT §12, K7):** K7a exact recovery in at least 80% of held-out planted processes; K7b edit precision ≥ 0.9; and no LCI-promoted model loses held-out log-likelihood.

**Amendment 1 (2026-10-29, written while coding, before any dev or held-out run).**
- **A repaired defect** is read on structure, not node numbers. The fingerprint of a model is the set of label trigrams along every path p → u → s, with merged steps expanded. A defect is repaired when the final model agrees with the planted one on every trigram that the defect alone changed.
- **Defect order.** A collapse renumbers nodes, so it is applied after the edge defects of the same case. A spurious edge is never a planted edge. A defect with no eligible pick is skipped; removed edges are drawn among planted edges walked at least 20 times in the training traces.
- **Report-only:** the share of the held-out gap closed, (H(start) − H(final)) / (H(start) − H(planted)), with H the held-out bits.

**Amendment 2 (2026-10-29, after K7a's dev read, before any held-out read).** THOUGHT's miner recovers 35 of 50 dev processes (0.70). 17 of its 18 missed edges were walked 11–29 times but met one stray reverse count from a noisy swap, which pulls `a ⇒ b` below 0.9 (for example 26 against 1: 0.893). K7a gains one dev choice among three miners, picked before the held-out read:
- **M0:** THOUGHT's rule as registered (`a ⇒ b ≥ 0.9`, `|a > b| ≥ 10`).
- **M1:** the same with `a ⇒ b ≥ 0.8`.
- **M2:** a noise floor of one: the dependency uses `max(0, |b > a| − 1)` in place of `|b > a|`; threshold 0.9.
- **Rule:** the highest exact recovery on dev seeds 1–50; within 2 processes, fewer spurious edges; then M0, M2, M1 in that order. The held-out read reports M0 too, and K7a's gate is read on the chosen miner. If the choice is not M0, THOUGHT §6.1's thresholds change only with Odin's agreement.

**Amendment 3 (2026-10-29, after K7b's dev read, before any held-out read).** As registered, K7b's edits reach a dev edit precision of 0.597 (w = 0.5) and 0.556 (w = 1.0). Added edges (43 of 44) and splits (20 of 20) help on held-out traces; removed edges help in only 112 of 230. Most removals are of real but rarely walked edges: the two-part code charges each edge 2 log2(N + 2) bits plus its parameter, more than a few escapes cost on the training traces, and held-out traces then walk the edge. K7b gains one dev choice among three edit searches, picked before the held-out read:
- **E0:** as registered, with dev's w = 0.5.
- **E1:** an edge costs log2(N + 2) bits (its target; the source is the node whose list it is in); w = 0.5.
- **E2:** E0, and each edit must also lower the bits of a validation fold: the training traces are split by index into 150 for the search and 50 for validation, with probabilities fitted on the 150. Held-out scoring is unchanged (probabilities fitted on all 200).
- **Rule:** the highest dev edit precision; within 0.02, more planted defects repaired; then E0, E1, E2 in that order. K7b's gate is unchanged and read on the choice.
- **Report-only change:** the gap closed is pooled, Σ(H(start) − H(final)) / Σ(H(start) − H(planted)); the mean of per-case ratios was dominated by cases with tiny gaps.

## 3. Calibration map for DIF applicability (fixed before any run)

- **The map** (`CalibratedApplicability` in `bindings_cypha.hpp`, wrapping K5's `DifApplicability`): at each label (a top-level run's outcome), the classifier's raw P for that process is recorded before the classifier learns from the label. One pooled map over all processes is fitted on these (raw P, label) pairs: from 50 pairs on, refitted every 25 new pairs; before that, the raw P. The OOD flag passes through unchanged. θ stays 0.1, now on the calibrated scale.
- **Two maps:** isotonic (ECS's `IsotonicCalibrator`), or Platt (logistic on logit(raw P), Newton, ridge 1e-3).
- **Streams:** K5's (training 1,500, held-out 1,000, then 200 of a new type). Dev seeds 1–3 choose the map: the lower pooled mean regret; within 0.005, the lower all-pairs ECE.
- **Held-out:** fresh seeds 2001–2003 (K5 read 1001–1003), read once, for Beta-class, Beta-pooled, raw DIF and the chosen calibrated DIF.

**Rules.**
- **C1, replace raw DIF:** the calibrated DIF replaces raw DIF wherever the K5/K6 tools use DIF if, on the held-out seeds, its mean regret is lower than raw DIF's with the paired 95% bootstrap interval of the difference below 0 on seed 2001 and the same sign on 2002 and 2003, and its all-pairs ECE is at most 0.01 above raw DIF's.
- **C2, the applicability default:** K5's rule 3 unchanged. The calibrated DIF becomes the K5 tool's default applicability model if its ECE ≤ 0.10, its regret ≤ 1.1 × Beta-class's, and LCI ≥ 0.95 against Beta-pooled. The interpreter's hook stays unset by default either way.

## 4. Checks

- A new CTest `native_thought_k6_test`: the chunk rules (compile at 5, the keep and drop thresholds, re-entry, audits, rule 10 in answer mode), the miner (dependency, loops, thresholds), the MDL scores against hand-computed values, each edit kind, the LCI promotion, and both calibration maps. Planted bugs must fail it: compile at 4, a keep test on the point estimate instead of the Wilson bound, a dependency without the +1, and a calibration map fitted after the label.
- The 9 CTests of HANDOFF §4.4, `native_thought_test` and `native_thought_bindings_test` pass. K2's replay hash `18a61550bffc735c` holds, and the interpreter is not changed.
- K5's reads reproduce byte for byte with the new code in (`chains`, `hyps`, `monitor`, and `applicability` on its own seeds).

## 5. Predictions (registered before any run)

| # | Prediction |
|---|---|
| P6a | Held-out chunk accuracy ≥ 0.95 |
| P6b | System 1 hit rate 0.35–0.60 |
| P6c | Wrong rate with chunking at most `none`'s (point) |
| P6d | `naive` raises the wrong rate by at least 0.05 |
| P6e | At most 10% of ambiguous classes ever have a kept rule |
| P6f | Every shifting class with a kept rule before the switch has it dropped within 40 of that class's tasks after it |
| P6g | Answer mode: 100% of System 1 answers become "I don't know" cards |
| P7a | Exact recovery in at least 90% of held-out planted processes; pooled edge F1 ≥ 0.98 |
| P7b | The library: exact recovery for at least 95% of the 144 processes |
| P7c | Edit precision ≥ 0.95 |
| P7d | At least 80% of planted defects repaired |
| P7e | LCI promotes at least 70% of edited models; none promoted with a held-out loss |
| PCa | Calibrated DIF: IMPASSE share ≤ 0.15 (raw: 0.30–0.38) |
| PCb | Calibrated DIF: mean regret ≤ 0.15 (raw: 0.27–0.34) |
| PCc | Calibrated DIF: all-pairs ECE ≤ 0.05 |
| PCd | C1 holds; C2 fails on the regret ratio to Beta-class |
| PCe | Isotonic is chosen over Platt on dev |

## Results on dev (2026-10-29)

Tools: `thought_k6` and `thought_k5 calibration` at the commits after amendment 3. The machine was shared with about twenty other runs; times are wall-clock under that load.

### K6 dev (streams 1–3, 4,000 tasks each)

| audit | re-entry | System 1 hit rate | wrong rate | wrong − none | chunk accuracy (Wilson) |
|---|---|---:|---:|---:|---|
| 0.05 | barred | 0.2308 | 0.0246 | +0.0040 | 0.9606 [0.9527, 0.9673] |
| 0.05 | 10 identical | 0.2308 | 0.0246 | +0.0040 | 0.9606 [0.9527, 0.9673] |
| 0.10 | barred | 0.2137 | 0.0209 | +0.0003 | 0.9774 [0.9709, 0.9825] |
| 0.10 | 10 identical | 0.2137 | 0.0209 | +0.0003 | 0.9774 [0.9709, 0.9825] |
| 0.20 | barred | 0.1898 | 0.0205 | −0.0001 | 0.9802 [0.9737, 0.9852] |
| 0.20 | 10 identical | 0.1899 | 0.0205 | −0.0001 | 0.9803 [0.9737, 0.9852] |

- `none` (System 2 always): wrong rate 0.0206.
- **Dev choice (registered rule): audit 0.05, re-entry barred.** It has the highest hit rate within the dev limit on the wrong rate (+0.0040 against +0.005). The held-out gate asks more: the upper end of the interval at most +0.005.
- Re-entry hardly matters: dropped classes rarely produce 10 identical results again.

### K7a dev (planted processes 1–50; amendment 2)

| miner | exact recovery (Wilson) | edge precision | recall (edges walked ≥ 10) | spurious edges |
|---|---|---:|---:|---:|
| M0 (THOUGHT) | 0.70 [0.56, 0.81] (35 / 50) | 1.0000 | 0.9722 | 0 |
| M1 (dependency ≥ 0.8) | 0.92 [0.81, 0.97] (46 / 50) | 1.0000 | 0.9938 | 0 |
| M2 (noise floor 1) | 0.94 [0.84, 0.98] (47 / 50) | 1.0000 | 0.9954 | 0 |

- **Dev choice: M2.**
- **The library (report only):** all 144 processes are recovered exactly from 200 noise-free random walks, with step ids as labels (M0).

### K7b dev (planted processes 1–50; amendment 3)

| search | edit precision (Wilson) | add edge | remove edge | split | merge | defects repaired | gap closed (pooled) | promoted / loss |
|---|---|---|---|---|---|---|---:|---|
| E0, w 0.5 | 0.597 [0.542, 0.651] (184 / 308) | 43 / 44 | 112 / 230 | 20 / 20 | 9 / 14 | 84 / 99 | 0.914 | 40 / 0 |
| E0, w 1.0 | 0.559 [0.504, 0.612] (180 / 322) | 40 / 41 | 113 / 249 | 18 / 18 | 9 / 14 | 83 / 99 | 0.892 | 40 / 0 |
| E1, w 0.5 | 0.645 [0.589, 0.696] (196 / 304) | 44 / 45 | 118 / 220 | 25 / 25 | 9 / 14 | 89 / 99 | 0.931 | 41 / 0 |
| E2, w 0.5 | 0.723 [0.656, 0.781] (141 / 195) | 39 / 39 | 75 / 125 | 16 / 16 | 11 / 15 | 74 / 99 | 0.920 | 42 / 0 |

- **Dev choices:** w = 0.5 (registered), then **E2** (amendment 3).
- No variant reaches 0.9 on dev. Removals of rarely walked real edges remain the weak edit. LCI promotes about 80% of edited models, and never one that loses held-out bits.

### Calibration map dev (streams 1–3)

| model | stream | ECE, chosen | ECE, all pairs | Brier | IMPASSE share | mean regret | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw DIF | 1 / 2 / 3 | 0.079 / 0.081 / 0.090 | 0.086 / 0.113 / 0.039 | 0.091 / 0.108 / 0.087 | 0.339 / 0.346 / 0.465 | 0.311 / 0.299 / 0.432 | 0.911 / 0.831 / 0.925 |
| isotonic | 1 / 2 / 3 | 0.014 / 0.021 / 0.030 | 0.116 / 0.101 / 0.108 | 0.116 / 0.130 / 0.113 | 0.132 / 0.418 / 0.320 | 0.120 / 0.369 / 0.291 | 0.832 / 0.733 / 0.854 |
| Platt | 1 / 2 / 3 | 0.028 / 0.021 / 0.028 | 0.153 / 0.123 / 0.107 | 0.128 / 0.116 / 0.091 | 0.136 / 0.281 / 0.248 | 0.124 / 0.239 / 0.215 | 0.842 / 0.797 / 0.888 |

- Pooled mean regret: raw 0.348, isotonic 0.260, **Platt 0.193**. Mean all-pairs ECE: raw 0.079, isotonic 0.108, Platt 0.128.
- **Dev choice: Platt.**
- The map cuts regret and the chosen process's ECE, but all-pairs ECE rises. The map is fitted on the labels of the processes that were run, then applied to every pair; C1's ECE condition may fail.

## Results on held-out (2026-10-29, each read once)

The dev choices above were committed (9c1df9c) before these reads.

### K6 held-out (streams 1001–1003; audit 0.05, re-entry barred)

| arm | System 1 hit rate | wrong rate | chunk accuracy (Wilson) | cost per task |
|---|---:|---:|---|---:|
| none | 0 | 0.0207 | – | 10.000 |
| **chunk** | **0.3347** | 0.0226 | **0.9691 [0.9633, 0.9740]** (3,893 / 4,017) | 7.417 |
| naive (ablation) | 0.9800 | 0.1172 | 0.8808 [0.8748, 0.8865] (10,358 / 11,760) | 1.180 |

- **Wrong answers, chunk − none** (paired per task): **+0.0019 [−0.0006, +0.0043]**. The upper end is below +0.005.
- Wrong answers, naive − none: +0.0966 [+0.0906, +0.1031].
- Cost per task falls by 26% (10 to 7.42).

| class kind | classes | ever compiled | ever kept | ever dropped |
|---|---:|---:|---:|---:|
| deterministic | 120 | 116 | 20 | 0 |
| mostly | 48 | 47 | 1 | 7 |
| ambiguous | 48 | 21 | 0 | 8 |
| shifting | 24 | 22 | 2 | 10 |

- **Shifting classes:** 2 had a kept rule at the switch. They were dropped after 88 and 42 of their own tasks.
- **Answer mode** (stream 1001): 1,545 System 1 answers, 1,545 "I don't know" cards.
- **K6 gate: MET.** Chunk accuracy 0.969 ≥ 0.9; System 1 hit rate 0.335 > 0; wrong-answer difference upper end +0.0043 ≤ +0.005.

**What the numbers say.**
- Only 20 of 120 deterministic classes ever reach "kept". The keep rule needs a Wilson lower bound of 0.9, so a rule needs about 35 checked uses (more with System 2's 2% noise). Under the Zipf law most classes are too rare for that in 4,000 tasks.
- A kept rule is checked on 5% of its uses. A drop needs 5 disagreements in the last 20 checks, so after a shift it takes about 5 / 0.05 = 100 of the class's uses. The 42 and 88 fit that. A higher audit share drops sooner but answers less often (dev: 0.20 gives hit rate 0.19).

### K7a held-out (planted processes 1001–1100)

| miner | exact recovery (Wilson) | edge precision | recall (edges walked ≥ 10) | F1 | spurious edges |
|---|---|---:|---:|---:|---:|
| M0 (THOUGHT, as registered) | 0.76 [0.668, 0.833] (76 / 100) | 0.9992 | 0.9788 | 0.9889 | 1 |
| **M2 (noise floor 1, dev choice)** | **0.96 [0.902, 0.984]** (96 / 100) | 0.9992 | 0.9969 | 0.9980 | 1 |

- **K7a gate (on the dev choice M2): MET.** THOUGHT's miner as written (M0) would not meet it (0.76).
- **The library (report only, M2):** 144 of 144 processes recovered exactly.

### K7b held-out (planted processes 1001–1100; E2, w = 0.5; 100 cases)

| | share (Wilson) | count |
|---|---|---|
| **Edit precision** | **0.6847 [0.6390, 0.7271]** | 291 / 425 |
| — add edge | 0.9851 [0.9202, 0.9974] | 66 / 67 |
| — split | 1.0000 [0.8957, 1.0000] | 33 / 33 |
| — merge | 0.7692 [0.6166, 0.8735] | 30 / 39 |
| — remove edge | 0.5664 [0.5085, 0.6226] | 162 / 286 |
| Planted defects repaired | 0.7565 [0.6913, 0.8116] | 146 / 193 |
| — removed edge | 0.8615 [0.7573, 0.9254] | 56 / 65 |
| — spurious edge | 0.9000 [0.8077, 0.9507] | 63 / 70 |
| — collapse | 0.4655 [0.3433, 0.5920] | 27 / 58 |

- Held-out gap closed (pooled): 0.9348 (39,821.6 of 42,597.2 bits).
- Cases edited 98; promoted by LCI 73; **promoted with a held-out loss: 0**.
- **K7b gate: NOT MET** (edit precision 0.685 < 0.9). Its second half holds: no promoted model loses held-out bits.

**What the numbers say.** Adding edges and splitting nodes are almost always right. Removing edges is right only 57% of the time; on dev, most wrong removals took out real edges that the training traces walk rarely (amendment 3). The final models still close 93% of the held-out gap, and LCI only promotes models that gain. A collapse is repaired in fewer than half the cases; why the split misses the rest is not measured here.

### Calibration map held-out (seeds 2001–2003; Platt)

| model | stream | ECE, chosen | ECE, all pairs | Brier | IMPASSE share | mean regret [95% CI] | AUROC |
|---|---:|---:|---:|---:|---:|---|---:|
| Beta-class | 2001 / 2002 / 2003 | 0.005 / 0.005 / 0.006 | 0.076 / 0.079 / 0.082 | 0.007 / 0.008 / 0.009 | 0.008 / 0.006 / 0.011 | 0.0174 / 0.0098 / 0.0206 | 1.000 |
| Beta-pooled | 2001 / 2002 / 2003 | 0.035 / 0.009 / 0.005 | 0.062 / 0.034 / 0.054 | 0.187 / 0.186 / 0.187 | 0.607 / 0.803 / 0.622 | 0.531 / 0.728 / 0.555 | 0.62 / 0.64 / 0.61 |
| raw DIF | 2001 | 0.079 | 0.105 | 0.109 | 0.368 | 0.3237 [0.2962, 0.3518] | 0.821 |
| raw DIF | 2002 | 0.079 | 0.077 | 0.088 | 0.360 | 0.3206 [0.2940, 0.3470] | 0.888 |
| raw DIF | 2003 | 0.092 | 0.069 | 0.064 | 0.380 | 0.3403 [0.3130, 0.3681] | 0.942 |
| **Platt DIF** | 2001 | 0.024 | 0.128 | 0.120 | 0.274 | 0.2349 [0.2104, 0.2583] | 0.796 |
| **Platt DIF** | 2002 | 0.021 | 0.123 | 0.119 | 0.267 | 0.2268 [0.2042, 0.2513] | 0.807 |
| **Platt DIF** | 2003 | 0.029 | 0.097 | 0.071 | 0.168 | 0.1578 [0.1364, 0.1796] | 0.975 |

- **C1** (regret saved per task, calibrated against raw, paired): 2001 +0.0888 [+0.0676, +0.1111]; 2002 +0.0938 [+0.0719, +0.1152]; 2003 +0.1825 [+0.1551, +0.2093]. The regret half holds on all three seeds. But pooled all-pairs ECE rises from 0.0836 to 0.1126 (+0.029 > +0.01). **C1 FAILS.**
- **C2:** ECE (chosen) 0.0245 ≤ 0.10; regret 0.2065 against Beta-class 0.0159 (ratio 12.96 > 1.1); LCI against Beta-pooled 0.50 (c = 0.5, since all-pairs ECE is higher than Beta-pooled's). **C2 FAILS.**

**What the numbers say.** The map cuts regret by about a third (0.33 to 0.21) and the chosen process's ECE from 0.08 to 0.02. On seeds 2001 and 2002 it lowers the P of processes that do not apply (mean 0.14 to 0.08), so fewer pass θ = 0.1 and IMPASSE falls. But it is fitted only on the labels of processes that were run, which are not a fair sample of all pairs, and it is then applied to every pair. So the all-pairs ECE gets worse. A map fitted on a fair sample of pairs (for example some runs chosen at random) would be the fair test; that is a new registered read, not this one.

## Predictions scored

| # | Prediction | Result | Score |
|---|---|---|---|
| P6a | Chunk accuracy ≥ 0.95 | 0.969 | hit |
| P6b | System 1 hit rate 0.35–0.60 | 0.335 | miss |
| P6c | Wrong rate with chunking ≤ `none`'s (point) | 0.0226 > 0.0207 | miss |
| P6d | `naive` raises the wrong rate by ≥ 0.05 | +0.097 | hit |
| P6e | ≤ 10% of ambiguous classes ever kept | 0 of 48 | hit |
| P6f | Shifting rules dropped within 40 tasks after the switch | 88 and 42 | miss |
| P6g | Answer mode: 100% "I don't know" cards | 1,545 of 1,545 | hit |
| P7a | ≥ 90% exact recovery; F1 ≥ 0.98 | M0 (registered): 0.76, F1 0.989; M2: 0.96, F1 0.998 | partial (miss as registered; met by the amended miner) |
| P7b | Library ≥ 95% exact | 144 of 144 | hit |
| P7c | Edit precision ≥ 0.95 | 0.685 | miss |
| P7d | ≥ 80% of defects repaired | 0.757 | miss |
| P7e | LCI promotes ≥ 70% of edited models; none with a loss | 73 of 98 (74.5%); 0 | hit |
| PCa | IMPASSE ≤ 0.15 | 0.168–0.274 | miss |
| PCb | Mean regret ≤ 0.15 | 0.2065 (0.158–0.235) | miss |
| PCc | All-pairs ECE ≤ 0.05 | 0.1126 | miss |
| PCd | C1 holds; C2 fails on the regret ratio | C2 fails as said; C1 fails too | partial |
| PCe | Isotonic chosen on dev | Platt | miss |

**6 hits, 2 partial, 9 misses.** The predictions were too hopeful about K7b's edits and about the calibration map, and P6f did not allow for the audit share's delay.

## Decisions (by the registered rules)

1. **K6: gate met.** The chunk store and its `dual_process` binding (`bind_dual_chunks`) are THOUGHT §4 as built. They speed up deliberate work only: in answer mode every chunk answer became an "I don't know" card (rule 10). The binding is opt-in; the default toolkit, the interpreter and Cypha's defaults do not change.
2. **K7a: gate met on M2**, the dev choice. THOUGHT §6.1's miner as written recovers 0.76. Changing §6.1 to a noise floor of one needs Odin's agreement (question below).
3. **K7b: gate not met.** MDL edits stay hypotheses. No mined or edited process enters the library. The LCI guard did its job: 73 promotions, none with a held-out loss.
4. **Calibration map: C1 and C2 fail.** Raw DIF stays wherever the K5 tool uses it, the Beta stays the K5 default, and the interpreter's hook stays unset. `CalibratedApplicability` stays in `bindings_cypha.hpp` for a later read.

**Questions for Odin.**
- Adopt the noise floor of one (M2) in THOUGHT §6.1's miner? Held-out recovery 0.96 against 0.76, with the same edge precision.
- K7b: keep edge removal as an edit (57% right on held-out) or drop it from the search, under a new registered read?
- The calibration map: re-read with a map fitted on a fair sample of pairs (some runs chosen at random), as a new registered read?

## Checks

- **CTests** (in /root/work/k6/build): the nine of HANDOFF §4.4 pass (`native_cyphalm_trie_selftest`, `native_fm_infinigram_test`, `native_epistemic_test`, `native_thought_test`, `native_geometry_test`, `native_ecs_lci_test`, `native_scribe_selftest`, `native_cyphalm_neural_smoke`, `native_class_context_test`), and `native_thought_bindings_test` (120 checks) and the new `native_thought_k6_test` (678 checks).
- **K2's replay hash** `18a61550bffc735c` holds (`native_thought_test`, 5,540 checks). `interpreter.hpp` and `bindings.hpp` are unchanged.
- **Planted bugs** each fail `native_thought_k6_test`: compile at 4 (3 checks fail), keep on the point estimate instead of the Wilson bound (1), a dependency without the +1 (4), a calibration map fitted after the label (3).
- **K5 reproduces byte for byte** with the new code in: `chains`, `hyps`, `monitor` and `applicability` give the same output as K5's published runs.
- Rule 10: nothing in K6 or K7 answers a question. Chunk answers carry `source = "chunk"`, and the guard turns them into cards in answer mode (checked in the CTest and on held-out stream 1001).
- No neural network at runtime; no Qwen inference. Peak memory on the dev runs: 11 MB (`chunk`, `mine`, `edits`) and 19 MB (`calibration`); wall-clock 0–31 s each.

## How to reproduce

```
cmake --build <build> --target thought_k6 thought_k6_test thought_k5
J=docs/thought/processes.json
# dev (choices)
<build>/thought_k6 chunk $J
<build>/thought_k6 mine $J
<build>/thought_k6 edits $J
<build>/thought_k5 calibration $J
# held-out (read once)
<build>/thought_k6 chunk $J --read-test --audit 0.05 --reentry 0
<build>/thought_k6 mine $J --read-test --miner M2
<build>/thought_k6 edits $J --read-test --variant E2 --w 0.5
<build>/thought_k5 calibration $J --read-test platt
ctest --test-dir <build> -R "native_thought_(test|bindings_test|k6_test)"
```

Code: `native/include/cypha/thought/chunking.hpp`, `native/include/cypha/thought/mining.hpp`, `CalibratedApplicability` in `native/include/cypha/thought/bindings_cypha.hpp`, `native/tools/thought_k6.cpp`, `native/tools/thought_k6_test.cpp` (CTest `native_thought_k6_test`), and `thought_k5 calibration` in `native/tools/thought_k5.cpp`.
