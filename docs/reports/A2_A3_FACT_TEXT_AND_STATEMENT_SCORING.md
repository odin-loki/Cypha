# A2 and A3: question-form fact text, and answering by scoring statements

**Odin Loch** · 2026-10-28 · steps A2 (route R1) and A3 (route R3) of [CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §4, approved by Odin on 2026-10-28

**Status: in progress.** Predictions are written first; results follow as each run ends.

---

## Predictions (written before any run)

**A3 (R3 statement scoring).**
- **S1.** R3 runs on 20–30 of the 130 dev questions: only those whose relation has an F2 pattern (birth and death dates and places, capital, language, director). About two thirds of them are in-slice.
- **S2.** Coverage at 2% wrong (5-fold CV in dev): 0–3 of 130. On in-slice questions whose fact is in the training prose: 0–5%. This is **below** the registered CYPHALM_ANSWERING prediction (5–20%). F2's frames are cleaned text, while CyphaLM read raw markup (`'''Name'''`, `[[link]]`). Only InfiniGram's long match can tie a subject to its object, and only where the frame's words occur verbatim.
- **S3.** Raw (ungated) R3 right answers: 2–6 of the questions it runs on.

**A2 (R1 fact shards).**
- **S4 (registered, CYPHALM_ANSWERING §4).** A2 raises coverage only on covered questions: 0 to +6 per 100 dev, almost all dates and places.
- **S5.** The F2 shard covers 3–8 dev questions (entity and relation). Raw right on covered questions: 30–70%. Coverage at 2% wrong: +0 to +4 per 100 dev. No change on questions it does not cover.
- **S6.** Holdout BPC changes by ≤ 0.005.
- **S7.** A2 + A3 together: no better than the better of the two alone, within one question.
