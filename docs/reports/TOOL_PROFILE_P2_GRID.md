# Tool profile P2–P4: which support tools make the cut

**Odin Loch** · 2026-10-28 · [TOOL_PROFILE_PLAN.md](../TOOL_PROFILE_PLAN.md) §4–§7

(The design and the predictions below were written before any grid run. Results and the P3/P4 decisions are added under them.)

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

---

## Results, first grid (dev, 130 questions; run 2026-10-28)

All runs: `cyphalm_trie qa` on the 8 MiB protocol, wiki then two Dolly passes, chat form shown (the wiki form answers nothing right in any run). "Kept" is the out-of-fold gate at a 2% wrong target (5-fold CV inside dev). BPC is the second half of the holdout, with InfiniGram and, where named, the class-context expert; its lambdas are fit on the first half. B0 and B0 + facts are the A0 and A2 runs (reproduced byte for byte, A2/A3 report).

| Run | Expert | Fact shard | BPC trie | BPC trie + IG | Raw right (trie + IG) | Kept: right / wrong (coverage) | Wrong rate, Wilson upper |
|---|---|---|---:|---:|---:|---|---:|
| B0 | none | no | 2.319989 | 2.071574 | 2 | 0 / 0 (0.0%) | 2.9% |
| B0 + facts | none | yes | 2.321002 | 2.072393 | 7 | 2 / 0 (1.5%) | 2.9% |
| B | shape:6 | no | 2.133872 | 1.995847 | 2 | 0 / 0 (0.0%) | 2.9% |
| B + tags | E1 tags:6 | no | 2.096562 | 1.973015 | 1 | 0 / 0 (0.0%) | 2.9% |
| B + facts | shape:6 | yes | 2.134429 | 1.996357 | 6 | 2 / 1 (2.3%) | 4.2% |
| B + tags + facts | E1 tags:6 | yes | 2.097419 | 1.973696 | 5 | 1 / 1 (1.5%) | 4.2% |

The trie alone (no InfiniGram) gets 0 right raw in every run. The tag table here is E1's tags over wiki train and Dolly (49,536 words), because `qa` trains on both.

**The Cypha gate** (`cypha_gate`, same rows, same 2% target, same folds), against `qa`'s threshold gate:

| Rows | Threshold gate: right / wrong | Cypha gate: right / wrong | Cypha wrong rate (Wilson upper) |
|---|---|---|---|
| B0 + facts | 2 / 0 | 1 / 2 | 1.5% (5.4%) |
| B + facts | 2 / 1 | 3 / 4 | 3.1% (7.6%) |
| B + tags + facts | 1 / 1 | 1 / 3 | 2.3% (6.6%) |
| B | 0 / 0 | 0 / 0 | 0.0% (2.9%) |

### What the first grid says

- **The class-context expert does not help answers.** It lowers BPC by 0.076 (shape) or 0.099 (tags) on this protocol and leaves coverage at 0 without facts. With facts it lets one extra wrong answer through the gate, which is inside the noise of 1–2 answers.
- **The tags add BPC, not answers.** −0.0228 over shape classes on this protocol (−0.0128 on fresh text, B4). Q/A: 0 vs 0 without facts, 1/1 vs 2/1 with them.
- **Facts are the only thing that adds right answers**, and only on the questions they cover (A2).
- **InfiniGram is what makes any answer possible**: every right answer needs it.
- **The Cypha gate is worse than a single-feature threshold** on every set: it keeps more wrong answers at the same target. With at most 7 right answers in 130, any learned gate is underpowered.

### Predictions so far

| # | Result | Score |
|---|---|---|
| G1 | B vs B0: coverage 0 → 0; BPC −0.076; surface S +0.144 (B4) | **Hit** |
| G2 | Tags: BPC −0.0228 (this protocol), −0.0128 (fresh text); Q/A 0 without facts, −1 right with facts | **Hit** on BPC; Q/A inside 0 to +2 without facts, outside it with facts (−1) |
| G4 | Facts: +2 right / +1 wrong over B (+1.5 per 100 at 2.3% coverage), BPC +0.0005 | **Hit** |
| G6 | InfiniGram: BPC −0.248 on the trie; answers only with it | **Hit** |
| G7 | Cypha gate no better than the threshold gate | **Hit** (worse on every set) |
| G3 | Word classes: S-1b's k-means classes (256, 1,024) as the class table: fresh BPC +0.0049 / +0.0032 against shape classes, surface S −0.079 / −0.092 (B4B) | **Miss** (no gain at all) |
| G5 | Teacher text | Not run: inference moved to Odin's PC (PLAN ledger 54) |
| G8 | The tagger fails §5.4 on BPC alone | **Hit**: −0.0128 on fresh text for 1,396 lines, under the −0.014 it needs |

---

## P3: decisions (2026-10-28)

The rule is TOOL_PROFILE_PLAN §5: a tool stays only if (1) adding it to B passes LCI ≥ 0.95 on its primary metric, (2) it still helps with the others present, (3) the wrong rate does not rise, (4) it gives at least +2 right answers per 100 or −0.01 BPC per 1,000 lines of tool code, and (5) it fits the budgets. The evidence is the grid above, B4B ([report](B4B_LM_DEFAULTS.md)) and the answering-variants runs ([A1D/A2B](A1D_A2B_ANSWERING_VARIANTS.md)), which reran the combinations with the familiarity token.

| Tool, in the role profiled | (1) helps | (3) honest | (4) worth its code | Decision |
|---|---|---|---|---|
| `pos_tagger` as the class-expert table | BPC −0.0128 on fresh text, LCI 1.00; Q/A 0 | with facts: 1 right / 1 wrong against shape's 2 / 1 | needs −0.014 for 1,396 lines: **fails** | **Role cut**: the class expert uses shape classes. The tagger **stays as the question parser**: E1's entity and its span candidates drive the familiarity token (A1d: entity right for 76 of 100, token precision 0.600 dev, 0.635 test) |
| `ppmi_svd` / S-1b classes as the class-expert table | **fails**: +0.0049 / +0.0032 BPC against shape classes | | | **Role cut.** The vectors stay a SEMANTICS source; the similar-context expert (A4) is their next use and is being measured |
| `pattern_learn` as the fact-shard writer | +2 right, 0 wrong over B0 (1.5 per 100); sign test on 2 questions, LCI about 0.75–0.87: **fails** | holds (0 wrong) | needs +6 per 100 for 3,026 lines: **fails** | **Fails the cut as a Q/A tool. Not deleted: flagged for Odin.** It is also the fact store behind D4, F2R and Track S's gap map (S3), which this profile does not measure |
| InfiniGram | BPC −0.248 on the trie; every right answer needs it | | far above | **Stays** (core expert) |
| `cypha_gate` (the Cypha classifier as abstain gate) | over ten configurations it keeps 21.7 right / 18.0 wrong (mean of 3 seeds) against the threshold gate's 15 / 5 | **fails**: over the 2% target in 6 of 30 seed runs | | **Deleted** (P4). The threshold gate in `qa` stays |
| Teacher text | not run | | | Waits for Odin's PC |

**Frozen test, read once on the kept set** (A1D/A2B report, configuration C1L: InfiniGram, D4's fact shard, the low-only familiarity token from E1's span candidates, threshold gate): 0 right, 0 wrong kept at 2% wrong (Wilson upper 1.9%); the token's abstain precision 0.635, recall 0.94. R3 with F2R's raw frames: 1 right, 0 wrong.

## P4: cuts (2026-10-28)

- **`native/tools/cypha_gate.cpp`** (224 lines) and its CTest `native_cypha_gate_selftest` are deleted. Its last version is in commit `e82a4a4` (added) as published; the result is the table above and A1D/A2B step 7.
- **Class tables from the tagger and from word vectors** are not used by default: nothing to delete, the class expert simply runs on shape classes.
- **`pattern_learn`** stays until Odin rules on it (above).
