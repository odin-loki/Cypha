# A1c: entity-level familiarity, and a copy path for the tag

**Odin Loch** · 2026-10-28 · follows [A0_A1_CYPHALM_QA_AND_DONT_KNOW.md](A0_A1_CYPHALM_QA_AND_DONT_KNOW.md) "Next steps" 1, 3 and 4 · step A1 of [CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §4

**Status: in progress.** Predictions are written first; results follow as each run ends.

---

## Predictions (written before any run)

**Familiarity statistics** (dev AUROC, not-in-slice against in-slice; 0.5 = chance; A1's question log-probability gave 0.54–0.60):

| Id | Statistic | Predicted dev AUROC | Why |
|---|---|---|---|
| F1 | `ent_bold`: count of `'''entity` (a bold lead) | 0.70–0.80 | An article bolds its subject in the lead. The famous not-in-slice subjects have no article. E1 entity errors on in-slice questions ("Canada", "the world") cost the rest |
| F2 | `ent_title`: count of `<title>entity</title>` | 0.68–0.78 | Same idea, stricter |
| F3 | `ent_count`: count of the entity anywhere | 0.30–0.55 | Famous subjects are mentioned more often than obscure ones that have an article |
| F4 | `ent_link`: count of `[[entity` | 0.35–0.60 | Same reason |
| F5 | `ent_igfrac`: longest piece found / length | 0.45–0.55 | Almost every word occurs somewhere |
| F6 | `ent_lp`: the trie's mean log2 p of the entity | 0.45–0.65 | Familiar bytes, not familiar subjects (A1's lesson) |

**Token run (A1c-after: 20% don't-know turns, token after `Assistant: ` from the best statistic, trie):**
- **F7.** Dev abstain precision on not-in-slice 0.35–0.55, recall 0.75–0.95. D2 (precision ≥ 0.8 and recall ≥ 0.5) **misses** on precision, because E1 misses the entity on many in-slice questions and those get `low`.
- **F8.** Holdout BPC changes by ≤ 0.005 against A0 (as D3).

**Copy expert (a match over the current context, mixed like InfiniGram):**
- **C1.** Holdout BPC (second half): trie + copy beats trie by 0.05–0.15; trie + InfiniGram + copy beats trie + InfiniGram by 0.02–0.08. LCI P(gain > 0.005) ≥ 0.95 for both.
- **C2.** Tags naming the question's entity: 0–10% (A1: 0). The copy expert can continue an entity once its first bytes match the question, but nothing points it at the entity's first byte.
- **C3.** Abstain precision and recall within ±0.05 of the same model without copy; tag completeness within ±10 points.
