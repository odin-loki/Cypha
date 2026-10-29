# Which tools make the cut: profiling Cypha with its support tools

**Odin Loch** · 2026-10-27 · part of PLAN Track P (profiling)

**Aim.** Decide, by measurement, which support tools stay. A tool stays only if it makes **CyphaLM** measurably better at its own tasks, for its cost in code, RAM and time. CyphaLM does every task (PLAN rule 10). A tool may change CyphaLM's training data or its contexts. It never answers in CyphaLM's place.

---

## 1. What is being profiled

| Tool | Lines | How it would feed CyphaLM (never answering for it) |
|---|---:|---|
| `pos_tagger` (E1) | 1,400 | **Class-level context:** a count expert in CyphaLM's mixture keyed on part-of-speech tags and phrase boundaries (PLAN B4), so contexts generalise across words of the same kind |
| `pattern_learn` (F2) | 1,940 | **Facts as training text:** learned facts written back as plain sentences (from their own reversed patterns), added as a fact-text shard that CyphaLM trains on |
| `ppmi_svd` (C1) | 1,370 | **Word classes from vectors** (k-means over PPMI-SVD) as a second class-level count expert. Also soft matching of unseen question words to seen ones inside CyphaLM's context lookup |
| `scripts/gguf_mine.py` (kept by decision) | 2,275 | **Teacher text:** paraphrases and Q/A text from a calibrated teacher, as a separate low-trust shard. Leakage-checked against the test set |
| InfiniGram 8K context (existing CyphaLM expert) | ~350 | Already inside CyphaLM. Measured at −0.26 BPC on 8 MiB. Profiled here for Q/A too |
| Cypha classifier (core) | — | **Abstain gate** on CyphaLM's own answer, from features of CyphaLM's predictive distribution. The answer bytes stay CyphaLM's. **Counts as CyphaLM doing the task for now** (Odin Loch, 2026-10-27). Later, Cypha investigates this through discovery mode: whether CyphaLM can judge its own answers, and what the gate actually contributes ([DISCOVERY §8a](DISCOVERY.md#8a-standing-investigation-who-decides-to-abstain)) |

**Not profiled for removal**, because they are core and kept: ECS, LCI and epistemic (honesty and certainty), SCRIBE (the judge, not a feature), and the thought interpreter. For these, only cost (latency, RAM) is profiled, to keep them inside the PLAN §10 budgets.

---

## 2. First: a CyphaLM question-answering baseline

Nothing can be judged until CyphaLM answers by itself. The baseline harness:

1. **Prompt.** The question in the same form CyphaLM was trained on, e.g. `User: <question>\nAssistant: `, and also a Wikipedia-style form, `<question> `.
2. **Answer.** CyphaLM (trie + InfiniGram mixture) decodes greedily until a newline or sentence end, at most 80 bytes. Its answer may be the "I don't know" message and tag ([DISCOVERY §3.5](DISCOVERY.md#35-the-dont-know-message-and-tag)), which CyphaLM is trained to produce. That counts as abstained, and the tag is logged to the gap map.
3. **Confidence from CyphaLM's own probabilities:**
   - the mean per-byte log-probability of the answer;
   - the minimum;
   - entropy at the first answer byte;
   - InfiniGram match length.
4. **Abstain.** It abstains when confidence is below a threshold chosen on the dev set for a wrong-answer rate ≤ 2% (PLAN rule 9).
5. **Scoring** is right / abstained / wrong on `bench/eval/t_qa_wiki_dev.tsv` (dev, 130 questions), and once at the end on `bench/eval/t_qa_wiki.tsv` (frozen test, 200 questions).

**Known so far:** CyphaLM scored 0/8 on the Wikipedia facts and 1/30 on held-out Dolly questions (2026-10-26). The baseline is expected near zero. That is the point of the profile: it shows how much each tool adds to it.

The harness is the simplest way to let CyphaLM answer and measure it. How CyphaLM should answer in the long run is a separate design, agreed with Odin before it is built: the proposal is [CYPHALM_ANSWERING.md](CYPHALM_ANSWERING.md) (2026-10-28).

---

## 3. Metrics

| Kind | Metric |
|---|---|
| **Task: Q/A** | Right, abstained and wrong on dev. Selective accuracy with a Wilson lower bound. Wrong rate with a Wilson upper bound. Coverage at a fixed wrong rate of 2% (the headline) |
| **Task: language model** | BPC on the enwik 8 MiB holdout (protocol in [CURRENT.md](CURRENT.md)). Per-relation and per-question-type breakdowns |
| **Task: text** | SCRIBE axis profile and surface S, reported only. SCRIBE does not decide anything until it passes its ladder (PLAN §6.4) |
| **Cost** | Lines of tool code, build time, preprocessing time, training time, peak RAM in training and serving, answer latency (p50, p99), data size on disk, external data needed, licence |
| **Upkeep** | Call sites, CTest coverage, number of external data dependencies |

---

## 4. Design

| Configuration | Why |
|---|---|
| **B**: CyphaLM alone | The baseline |
| **B + T** for each tool T | What each tool adds on its own |
| **F**: CyphaLM with every tool | The ceiling |
| **F − T** for each tool T | What each tool adds given the others (catches overlap: the tagger and word classes may do the same job) |

- Every configuration is trained on the same 8 MiB protocol.
- Decoding is deterministic, so one run per configuration. Where sampling is involved, use 3 seeds.
- The grid is about 12 configurations, at about 6–8 minutes each on one core.
- **Leakage check** first (`leakage_check` process): no test question or answer string may appear in any added shard (fact text, teacher text).
  - **For fact shards (decided 2026-10-28, from the D4 and A2 evidence):** a fact shard about in-slice entities must contain answers, because that is route R1. So: no shard question may equal a dev or test question; the shard is built without reading test answers; and the share of test questions it covers by (entity, relation) is reported with every result.
- Every run is recorded in the experiment DB with code and data hashes and its prediction (PLAN rule 3).

---

## 5. What makes the cut

A tool stays only if **all** of these hold:

1. **It helps.** Adding it to B passes LCI (≥ 0.95) on its primary metric. That is coverage at a 2% wrong rate for Q/A tools, or BPC for language-model tools.
2. **It still helps with the others present.** Removing it from F lowers the metric with P ≥ 0.9.
3. **It is honest.** The wrong-answer rate does not rise, with its Wilson upper bound compared (PLAN rule 9).
4. **It is worth its code.** At least +2 right answers per 100 dev questions at matched risk, or at least −0.01 BPC, per 1,000 lines of tool code.
5. **It fits the budgets** in PLAN §10: RAM, latency.

**What happens next**
- **Tools that fail** are deleted. Their result and last commit are recorded, as for the earlier cuts (`docs/reports/REMOVED_NON_CYPHALM_QA.md`, the E4 report).
- **Tools that pass** get a CMake target and a CTest, and their adapter is folded into the CyphaLM library rather than left as a separate program.
- **A tie** goes to fewer lines and fewer dependencies.

---

## 6. Predictions, registered before the runs (PLAN rule 3)

| Tool | Predicted effect | Reasoning |
|---|---|---|
| `pos_tagger` | BPC −0.00 to −0.03; Q/A +0 to +2 per 100 | Tags generalise contexts a little; they add no knowledge |
| `pattern_learn` | Q/A +0 to +4 per 100, almost all on date and birthplace questions; BPC ≈ 0 | Its new facts are mostly dates and places, and only questions about those can gain |
| `ppmi_svd` classes | BPC −0.00 to −0.03; Q/A robustness to rewording slightly up | Word classes overlap with the tagger's job; expect F − T to show they are redundant with it |
| Teacher text | Q/A gain the largest of the shards, +2 to +15 per 100, with the highest contamination risk | It brings world knowledge in, so leakage checks and the low-trust prior matter most |
| InfiniGram | BPC −0.2 to −0.3 (as measured); Q/A gains only where a fact's phrasing matches the question | Long matches copy fact sentences from the training text |
| Cypha abstain gate | Lower wrong rate at the same coverage than CyphaLM's mean log-probability alone, or no better | E4 found Cypha's confidence badly calibrated on text; this may repeat |

---

## 7. Order of work

1. **P0:** the CyphaLM Q/A harness and baseline (section 2). **Done 2026-10-28:** `cyphalm_trie qa`; the baseline is 0.0% coverage at 2% wrong ([report](reports/A0_A1_CYPHALM_QA_AND_DONT_KNOW.md)).
2. **P1:** one adapter per tool, each at most about 200 lines: tag-context expert, class-context expert, fact-text shard writer, teacher-text shard writer, abstain gate.
3. **P2:** the grid (section 4), plus the leakage check.
4. **P3:** decisions (section 5). The frozen test runs once, on the final kept set only. **Done 2026-10-28** ([report](reports/TOOL_PROFILE_P2_GRID.md)).
5. **P4:** delete what failed, fold what passed into the library, and write a dated report with every number and every cut. **Done 2026-10-28:** `cypha_gate` deleted; `pattern_learn` flagged for Odin. **Odin, 2026-10-29: keep `pattern_learn`** as Track S's fact and hypothesis tool (PLAN ledger 65); it never answers.
