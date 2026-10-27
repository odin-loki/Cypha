# How CyphaLM answers questions by itself (proposal)

**Odin Loch** · 2026-10-28 · **Status: proposal, waiting for Odin's agreement before anything is built** (TOOL_PROFILE_PLAN §2)

**In one line.** CyphaLM writes every answer byte. Tools may only add training text, add experts that predict the next byte, or gate (PLAN rule 10). Within that, there are three ways CyphaLM can get from a question to an answer, and they work best together.

---

## 1. Why it scores zero today

CyphaLM predicts the next byte from the bytes it has seen.
- **The knowledge is in prose.** Wikipedia says "Paris is the capital and largest city of France".
- **The question is in another form.** It arrives as "What is the capital of France?".
- **So there is no path.** No context CyphaLM has counted continues from the question's bytes into "Paris", so it backs off to generic text.

That is the whole of the 0/8 on Wikipedia facts and the 1/30 on held-out Dolly ([report](reports/CYPHALM_QA_AND_CONTEXT_TEST.md)). It is a missing path, not a missing fact.

---

## 2. Three ways to make the path, all through CyphaLM

| Route | What changes | CyphaLM's part |
|---|---|---|
| **R1. Question-form training text** | Facts CyphaLM should know are also written as many phrasings of `User: <question>\nAssistant: <answer>`. The facts come from the fact store (F2's learned facts written back as text) and from teacher queries (EMBEDDING_GEOMETRY C1). Each is a low-trust shard, leakage-checked against the test sets | It learns the path by counting, as it learns everything else |
| **R2. Generalising experts** | The class-context expert (tags and word classes, PLAN B4) and the similar-context expert (stored contexts that are near in vector space, SEMANTICS §5 / S-4). InfiniGram long matches are already there | Its mixture predicts from contexts that mean the same thing, not only from the same bytes. An unseen phrasing reaches an R1 path |
| **R3. Answer by checking the statement** | CyphaLM samples k candidate answers. For each, it scores the declarative form of the fact ("The capital of France is ___") by its own bits per byte, and the candidate that makes the statement most probable wins | The prose knowledge it already has becomes usable without any new shard. The question-to-statement frame comes from the question parse (E1) and the learned patterns (F2). Those are data, not answers |

**Not allowed, and why.** Looking up a stored answer string, retrieving a passage and pasting it, or a hand-written fact lookup. All three were built and removed on 2026-10-27 because CyphaLM did not do the task ([record](reports/REMOVED_NON_CYPHALM_QA.md)).

---

## 3. The proposed answer loop

1. **Read.** Question bytes become CyphaLM's context. The question parse (wh-word, verb, entity) and the semantic layer's item vectors are side inputs to the R2 experts.
2. **Generate.** A greedy answer, plus k = 5 samples with fixed seeds.
3. **Check.**
   - **Self-consistency:** the majority fraction across the samples.
   - **R3 statement score:** the bits per byte of each candidate in its declarative frame.
   - **Its own confidence:** the mean and minimum per-byte log-probability, the entropy at the first byte, and the InfiniGram match length.
4. **Decide.**
   - **Answer** when the combined confidence clears the dev-chosen threshold for a 2% wrong rate.
   - **Otherwise, the "don't know" message and tag**, which CyphaLM is trained to write itself (DISCOVERY §3.5). For now, the Cypha classifier gate may trigger the same message on its behalf (DISCOVERY §8a).
5. **Card.** An evidence card with ECS and reason codes (DISCOVERY S1).
6. **Learn.**
   - A user correction becomes training text through the online update.
   - A "don't know" tag opens or updates a gap in the gap map.

---

## 4. Order of work

| Step | Work | Measure |
|---|---|---|
| A0 | TOOL_PROFILE P0 harness: CyphaLM alone, greedy, with its own confidence. **Done 2026-10-28:** coverage at 2% wrong is 0.0% on dev; test 0–2 of 200 raw ([report](reports/A0_A1_CYPHALM_QA_AND_DONT_KNOW.md)) | Right / abstained / wrong on dev 130; coverage at 2% wrong |
| A1 | "Don't know" training text (not-in-slice questions, held-out facts and contested slots, each with message and tag). **Run 2026-10-28:** CyphaLM writes the message itself, but when it writes it is no better than its familiarity signal (AUROC 0.54–0.60). D1 hit, D2 missed, D3 hit | Abstain recall and precision; reason-code accuracy |
| A2 | R1 shards: fact text in question form | Coverage at 2% wrong; BPC unchanged or better |
| A3 | R3 statement scoring | The same, on facts with no R1 shard (it should work there) |
| A4 | R2 experts, after S-1 and the geometry work | Coverage on reworded questions (T-perturb) |
| A5 | Self-consistency and the combined confidence | Risk–coverage curve |

The frozen test (200 questions) runs once, at the end of each step that changes the defaults.

**Predictions** (registered before the runs):
- **A2 helps only where its facts are.** It raises coverage only on questions whose fact is in a shard: 0 to +6 per 100 on dev, almost all on dates and places (same reasoning as the F2 prediction).
- **A3 reaches some of what's already in the prose.** It answers 5–20% of the in-slice questions whose fact appears in the training prose, at a 2% wrong rate.
- **A1 abstains correctly.** CyphaLM's own message and tag reach abstain precision ≥ 0.8 on not-in-slice questions.

---

## 5. What Odin decides

1. **Which routes, and in which order.** The recommendation is R1 + R3 first (A0–A3), then R2 once S-1 and the geometry work report.
2. **Teacher text in R1.** Should Qwen3.5-9B queries be allowed in the R1 shards, as a low-trust shard kept separate so its effect can be measured? Or should the shards come only from Cypha's own fact store?
3. **The R3 frames.** Is it acceptable for the question-to-statement frames to come from the question parse and the learned patterns? They are data that tell CyphaLM what to score, not answers.
4. **The abstain gate.** It stays as "counts as CyphaLM for now" until A1 shows whether CyphaLM's own message is enough.
