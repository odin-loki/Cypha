# F2R: relation patterns learned over the raw bytes CyphaLM reads

**Odin Loch** · 2026-10-28 · PLAN F2 and D4; CYPHALM_ANSWERING route R3 (frames)

## Predictions (registered before the first raw run)

Written into this file before `pattern_learn --raw` was first run. Scored at the end.

| # | Prediction |
|---|---|
| R1 | Raw mode gives more candidate patterns than the cleaned run (1,318): between 1.2 and 3 times as many, because markup splits one wording into several |
| R2 | Kept raw patterns: 40 to 90 (cleaned: 70). Dates and birth places keep most of them |
| R3 | Of capital, author, director, composer and founder, at most one gets a kept raw pattern. The text, not its form, is the limit (D4) |
| R4 | Admitted raw facts are within 25% of the cleaned run's 1,216. Raw-only admitted facts (not admitted by the cleaned run, by subject, relation and object): 30 to 200 |
| R5 | Raw admitted facts agree with held-out Wikidata at least 0.90 of the time, Wilson lower bound at least 0.85 |
| R6 | Hand check: at least 26 of 30 raw-only new facts are right |
| R7 | Frame reach (dev questions, grading only). Among dev questions with an E1 entity and a template relation that has a kept frame, the share where a kept frame's prefix, with the subject filled, occurs verbatim in the training bytes: at most 10% for the cleaned frames; at least 20% for the raw frames, and at least twice the cleaned share |
| R8 | Decision: raw patterns join the cleaned ones rather than replace them. The union admits more facts than either run, at gold precision within 0.02 of the cleaned run |
