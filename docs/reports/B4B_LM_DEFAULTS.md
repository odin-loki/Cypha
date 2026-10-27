# B4b: which next-byte experts become CyphaLM's defaults

**Odin Loch** · 2026-10-28 · PLAN B4 and B9 · follows [B4_CLASS_CONTEXT_EXPERT.md](B4_CLASS_CONTEXT_EXPERT.md) and [A1C_ENTITY_FAMILIARITY_AND_COPY.md](A1C_ENTITY_FAMILIARITY_AND_COPY.md)

(Draft. The decision rule and the predictions below were written before the runs. Results are added under them.)

**Question.** Two new experts each lower CyphaLM's BPC a lot: the class-context expert (B4: −0.078 on fresh text with shape classes, −0.091 with E1 tags) and the copy expert (A1c: −0.146 on the holdout's second half, chat model). Do they add up, which class table is best, and which of them become defaults?

## Decision rule (fixed before the runs)

An expert, or a setting of one, becomes a default only if all of these hold:
1. **BPC.** On fresh enwik8 text `[8388608, 9437184)`, read once, it lowers trie + InfiniGram BPC with LCI ≥ 0.95 (block bootstrap, 4 KiB blocks).
2. **Text.** Surface S on the 200-prompt protocol (B4) does not fall: the 95% interval of the change is not entirely below zero.
3. **Answers.** In the tool-profile grid, coverage at 2% wrong on dev does not fall, and the wrong rate's Wilson upper bound does not rise.
4. **Cost.** Inside the PLAN §10 budgets (whole language system ≤ 4 GB resident; Q/A p50 ≤ 200 ms on one core), with its build time and RAM recorded.

Settings (tail length, class table) are chosen on the protocol holdout; only the chosen ones are read on fresh text.

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| L1 | The class and copy experts stack. On fresh text, the gain of both together over trie + InfiniGram is 70–100% of the sum of their single gains: the copy expert works on repeats inside the document, the class expert on short within-word and word-class contexts |
| L2 | With both on, trie + InfiniGram on fresh text is 1.95–2.03 BPC (alone: 2.166) |
| L3 | S-1b's k-means word classes (256 or 1,024) beat shape classes by −0.00 to −0.02 BPC on fresh text, no more than the E1 tags do (−0.013) |
| L4 | Tags × word classes (one product table) beat the better single table by less than 0.005 BPC |
| L5 | The copy expert raises surface S less than the class expert does (S counts real words, and copying repeats words the document already has, which the loop term L can penalise) |
