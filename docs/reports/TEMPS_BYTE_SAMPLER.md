# TEMPS: the byte sampler's temperatures

**Odin Loch** · 2026-10-29 · the generation track ([GEN_TRACK](GEN_TRACK.md)), decided by measurement under PLAN ledger 55, open to Odin's veto

## Why

`cyphalm_trie sample` writes every prompt at three fixed temperatures: 0.6, 0.8 and 1.0. With byte Kneser–Ney (BKN2, now the default), SCRIBE v2 rises as the temperature falls. On GEN-A's dev split: T 0.6 0.157 [0.122, 0.196], T 0.8 0.136, T 1.0 0.080. On test (T4d's read): T 0.6 0.147 [0.115, 0.181]. Copying rises too: dev copy share 0.0691 at T 0.6, 0.0383 at T 0.8. T4d proposed letting `sample` take other temperatures. This item measures whether a lower one is better, with the copy rule deciding.

Temperature only reshapes CyphaLM's own next-byte distribution. CyphaLM still writes every byte (rule 10).

## The change (code)

- `cyphalm_trie sample … --temps T[,T][,T]`: one to three temperatures. The default stays 0.6,0.8,1.0. Without the flag the output is byte for byte what it was.
- Each temperature keeps the seed of its position. So `--temps 0.4` draws with the seeds of the default's T 0.6, and the two are paired prompt by prompt.
- GEN lines are labelled `trie_T0.4` and so on (one decimal as before; two if needed). The `MODEL` line gains `temps=…` only when the flag is given. The `GEN_EVAL copy32` line lists the temperatures given.
- `scripts/gen_eval_summary.py` shows its five samples at the run's first temperature when the run has no T 0.8. Its output is unchanged for the default temperatures (checked on BKN2's dev run).
- SCRIBE's ladder scores the pooled rung over every kind. So a run with one temperature is read on the pooled rung.

## Registration (2026-10-29, before any read)

**Arms (dev, GEN-A's frozen dev split, main's default otherwise: byte Kneser–Ney, similar-context, copy):**

| Arm | Flag |
|---|---|
| R06 | `--temps 0.6` (the comparator, and a check) |
| R05 | `--temps 0.5` |
| R04 | `--temps 0.4` |
| R03 | `--temps 0.3` |

Each is read on its pooled SCRIBE v2 rung, with surface S, the 32-byte copy share (the `GEN_EVAL copy32` line), the invented-word rate (`cyphalm_tok_gen words`) and mean length.

**Check.** R06's GEN lines must equal the `trie_T0.6` lines of BKN2's dev file (`/root/work/bkn/runs/bkn2/gen/BKN.gen.txt`) byte for byte, and its copy share must be 0.0691. If not, stop: the pairing is broken.

**Copy rule (dev).** An arm is eligible if its dev copy share is at most the default's dev copy share at T 0.6 plus 0.02: 0.0691 + 0.02 = **0.0891**.

**Dev choice.** Among R05, R04 and R03, the eligible arm with the highest dev SCRIBE. If two are within 0.005, the higher temperature wins (it copies less).

**Test read.** Made only if the chosen arm's dev SCRIBE is above R06's by more than 0.005 (the tie band of T4b and T4c). Then ONE read of GEN-A's test split with `--temps T*`, logged in GEN_TRACK's Reads table.

**Gate (test).** T*'s SCRIBE v2 (pooled rung) above the default's test T 0.6 rung (T4d's BK0: 0.147 [0.115, 0.181]) with disjoint 95% intervals, and its test copy share at most BK0's T 0.6 copy share plus 0.02: 0.0769 + 0.02 = **0.0969**.

**If the gate holds.** `sample`'s default temperatures become {T*, 0.6, 0.8}. The T 0.6 and T 0.8 rows stay comparable with every earlier read; T 1.0 goes (`--temps 0.6,0.8,1.0` restores it). The pooled number then changes meaning, and the docs must say so. `chat` and every other sampler keep their temperatures.

**If it fails.** `--temps` stays as a measuring option. No default changes.

### Predictions

| # | Prediction | Falsified if |
|---|---|---|
| P1 | Dev SCRIBE rises as T falls from 0.6 to 0.5 to 0.4 | any of R05, R04 is below the next higher temperature by more than 0.005 |
| P2 | Copy rises as T falls; R04 is not eligible (copy above 0.0891) | R04's copy is at most 0.0891 |
| P3 | R05 is eligible | R05's copy is above 0.0891 |
| P4 | Surface S rises as T falls (R05 at least 0.930) | R05's surface S is below 0.930 |
| P5 | The invented-word rate falls as T falls | R05's rate is above R06's |
| P6 | R06 reproduces BKN2's T 0.6 lines byte for byte, copy 0.0691 | any GEN line or the copy share differs |
| P7 | If a test read is made, the gate is not met (the intervals overlap) | disjoint intervals on test |
| P8 | Without `--temps`, a 10-prompt `sample` file equals main's byte for byte | any byte differs |

## Results (2026-10-29, dev only)

Tool: `cyphalm_trie` at 918d42c (main 1943ba2 plus `--temps`), SCRIBE `scribe_gena` (GEN-A's binary, as in BKN2 and T4d). Each arm is one temperature on GEN-A's 250 dev prompts, read on SCRIBE's pooled rung.

| Arm | SCRIBE v2 [95%] | Surface S [95%] | Copy share | Invented words | Mean bytes kept | Eligible (copy ≤ 0.0891) |
|---|---|---|---|---|---|---|
| R06, T 0.6 | 0.157 [0.123, 0.196] | 0.927 [0.920, 0.933] | 0.0691 | 0.0141 | 362 | comparator |
| R05, T 0.5 | 0.157 [0.127, 0.191] | 0.899 [0.892, 0.907] | 0.0838 | 0.0141 | 364 | yes |
| R04, T 0.4 | 0.125 [0.101, 0.153] | 0.860 [0.851, 0.870] | 0.1132 | 0.0148 | 397 | no |
| R03, T 0.3 | 0.143 [0.121, 0.169] | 0.833 [0.824, 0.843] | 0.1296 | 0.0182 | 403 | no |

- Continuation bits per byte are the same in every arm (2.0145), as they must be: temperature does not enter the scoring of human continuations.
- **Check (P6).** R06's 250 GEN lines equal the `trie_T0.6` lines of BKN2's dev file byte for byte, and its copy share is 0.0691. The pairing holds, and a one-temperature run read on the pooled rung gives the per-temperature rung's value (0.157 [0.123, 0.196] against BKN2's 0.157 [0.122, 0.196]; the bootstrap draws differ slightly).
- **Check (P8).** Without `--temps`, a 10-prompt `sample` file equals main's byte for byte (19,365 bytes).

**Decision.** Only R05 is eligible, and it ties R06 (0.157 against 0.157, inside the 0.005 band). By the rule there is no test read and no default change. `--temps` stays as a measuring option. `sample`'s temperatures stay 0.6, 0.8 and 1.0.

**What it shows.** Below 0.6 a lower temperature does not help the byte sampler. SCRIBE stops rising at 0.5 and falls at 0.4. Surface S falls steadily (0.927 → 0.833), copying rises (0.069 → 0.130), and invented words rise below 0.5. A likely cause, not measured: at low temperature the distribution is sharp, so the no-repeat rule bans the top choices more often and forces unlikely bytes. T 0.6 is already the byte sampler's best temperature in this grid.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| P1 | Dev SCRIBE rises from 0.6 to 0.5 to 0.4 | **Miss**: flat at 0.5 (0.157), then down at 0.4 (0.125) |
| P2 | Copy rises; R04 not eligible | **Hit**: 0.069 → 0.084 → 0.113 → 0.130; R04 0.1132 |
| P3 | R05 eligible | **Hit**: 0.0838 |
| P4 | Surface S rises (R05 at least 0.930) | **Miss**: it falls (R05 0.899) |
| P5 | Invented-word rate falls as T falls | **Hit, narrowly** at 0.5 (0.01405 against 0.01413); it rises below 0.5 |
| P6 | R06 reproduces BKN2's T 0.6 lines | **Hit** |
| P7 | If a test read is made, the gate is not met | Not scored: no test read |
| P8 | No flag: 10-prompt file equals main's | **Hit** |
