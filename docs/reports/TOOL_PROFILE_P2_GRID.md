# Tool profile P2–P4: which support tools make the cut

**Odin Loch** · 2026-10-28 · [TOOL_PROFILE_PLAN.md](../TOOL_PROFILE_PLAN.md) §4–§7

(Draft. The design and the predictions below were written before any grid run. Results are added under them.)

## Design

Every configuration trains CyphaLM on the 8 MiB protocol exactly as `cyphalm_trie qa` does (wiki train, `<|doc|>`, two Dolly passes), answers the 130 dev questions in both prompt forms, and reports, together:
- **Q/A:** right / abstained / wrong, coverage at a 2% wrong rate (5-fold CV inside dev), selective accuracy and wrong rate with Wilson bounds;
- **BPC:** the trie mixture on the whole holdout, and the second half with InfiniGram and with any expert;
- **surface S:** of 400-byte samples from the same 200 holdout prompts (the B4 protocol).

The answering loop is whatever A2/A3 leave as the default (CYPHALM_ANSWERING §4). The frozen test runs once, on the final kept set only (P3).

| Tool T | How it enters CyphaLM | Adapter |
|---|---|---|
| `pos_tagger` (E1) | class-context expert with the E1 tag table | `class_context.hpp` + tag table |
| `ppmi_svd` (C1) and S-1b | class-context expert with k-means word classes | `class_context.hpp` + class table |
| `pattern_learn` (F2/D4) | fact-text shard in the Dolly passes | `pattern_learn --shard` |
| teacher text | low-trust shard, separate | `gguf_mine.py` facts → qa text |
| InfiniGram | the existing expert | already in `cyphalm_trie` |
| Cypha classifier | abstain gate on CyphaLM's features | `cypha_gate` |

**Baselines.** B0 is CyphaLM as it is (trie, then trie + InfiniGram). B is B0 with the class-context expert on shape classes only: the B4 report found that the expert's own mechanics carry most of its gain (−0.078 BPC on fresh text), so a tool is judged by what it adds *beyond* that. Both B0 and B are reported. When the tag and word-class experts are both on, one table maps each word to the pair (tag, class).

**Runs.** B0, B, B + each of {tags, classes, facts, teacher}, F (all), and F − each. InfiniGram and the gate come from the same runs (trie-only rows; gate on or off), so they cost no extra training.

## Predictions (registered before the grid runs)

From TOOL_PROFILE_PLAN §6, plus the ones the B4 result calls for:

| # | Prediction |
|---|---|
| G1 | **B vs B0:** the shape-class expert changes Q/A coverage at 2% wrong by 0 to +1 per 100 (it adds no knowledge), lowers chat-protocol BPC by 0.05–0.09, and raises surface S |
| G2 | **Tags (B + tags vs B):** BPC −0.00 to −0.03; Q/A +0 to +2 per 100 |
| G3 | **Word classes:** BPC −0.00 to −0.03; F − classes shows them redundant with the tags (the difference between F and F − classes is inside its interval) |
| G4 | **Facts:** Q/A +0 to +4 per 100, almost all on date and birthplace questions; BPC ≈ 0 |
| G5 | **Teacher text:** the largest Q/A gain of the shards, +2 to +15 per 100, with the highest contamination risk |
| G6 | **InfiniGram:** BPC −0.2 to −0.3; Q/A gains only where a fact's phrasing matches the question |
| G7 | **Cypha gate:** no lower wrong rate at the same coverage than the best single-feature threshold gate (E4 found Cypha's text confidence badly calibrated) |
| G8 | **Cuts:** the tagger fails §5.4 on BPC alone (it needs −0.014 for its 1,396 lines and gave −0.013 in B4); whether it stays depends on Q/A |
