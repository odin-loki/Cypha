# QA recheck: answers, RAM and the report card on today's main

**Odin Loch** · 2026-11-12 · job J5 · PLAN Track P (M18), ledger 118 · branch `wip/qa-recheck`

(Registered before any run. Results go under "Results" below.)

## Why

Many speed merges landed on main today (GENERATION_PLAN §13.50–13.54: O6, the O1 option, M4, O8, the O2 pipeline, O3). The writer changed too (§13.48–13.53: K5 counts from 337 MB, a 16-draft gate, the LDA topic term). Every one of them was checked as exact. None was meant to change an answer. This job checks that on `qa` itself, takes fresh RAM numbers for M18, and makes a new report card. Standing rule: "The report card always uses the latest results."

## What runs

Main `fb79cc93`, built Release (`cmake -DCMAKE_BUILD_TYPE=Release`). Each run is pinned to 4 CPUs (`taskset`) and runs under `nice -n 5`. Peak RAM is VmHWM from `/proc`, polled every second (the card's method).

1. **`qa`, dev only.** The card's C1L protocol, through `scripts/cypha_report_card.sh`. The four C1L inputs come from `bench/eval/qa_inputs.tar.gz`. No frozen test is read: no change here is meant to move an answer, so no test read is due.
2. **`qa` RAM, every configuration that reads only dev:**
   - Q1 built, C1L (the card's run);
   - Q2 built, C1L with `--save DIR` (M18's R1);
   - Q3 served, C1L with `--model DIR` (M18's R2, Q2's store);
   - Q4 built, the `a0` protocol (repository files only).
   - M18's R3 (the frozen-test configuration) is not run. It reads the frozen test.
3. **`context8k`, full default run** (the card's `ctx` part), under `flock ~/.cypha_full.lock`. It also prints the T-perturb tallies (seen, one letter changed, " Thanks." added, paraphrase, held-out Dolly, wiki facts).
4. **The writer** (the card's `gen` part): GEN-A's harness, `scripts/gen_eval.sh dev` with `sample --k5 /mnt/d/cypha_llm/fuzzy/train.txt` and main's defaults (337 MB counts, gate 16, LDA term, planner), from the repository root. It starts only when `free -g` shows at least 14 GB available.

## What it is compared with

- **Answers.** The latest record of `qa`'s answers on main's defaults is M18 step 1's built run R1 (`ac9ac07`, 2026-11-01; `/mnt/d/cypha_llm/profile/m18/R1.out` and `R1.rows.tsv`). Its dev gate (`CV chat/trie_ig all`): **3 right, 2 wrong, 125 abstained** of 130. Before the gate: 6 right, 56 wrong, 68 abstained.
- The last card in REPORT_CARD.md (2026-10-29) reads 2 right, 0 wrong, 128 abstained. That card is from before byte Kneser–Ney became a default. BKN2 (`8e063682`) moved the answers to 3 / 2 / 125 by design (BYTE_KN). So the old card is shown beside, but the rule is read against R1.
- **T-perturb and `context8k`.** Main's last full default run, O5's `F_old.out` (2026-11-06, `/mnt/d/cypha_llm/opt/o5/F_old.out`; the O2 and O3 full runs equal it). `context8k_bpc` 1.4962. Tallies: seen 3/3, one letter changed 3/3, " Thanks." 0/3, paraphrase 0/3, held-out Dolly 0/30, wiki facts 0/8.
- **The writer.** L1's dev GEN file (`bench/k5_coherence/runs/L1.gen.txt`, §13.53), the default since today. Dev SCRIBE v2 0.610 [0.577, 0.645], copy32 0.2416.

## The rule

- **Answers must not change.** Every `RAW`, `CV` and `ABSTAIN` line of `qa`'s output (every form and model), and the rows file, must equal R1's. The only allowed difference is the BPC lines, which are not answers.
- Any difference in right, wrong or abstained is a **regression**. It is reported with the commit that caused it, found by bisecting `ac9ac07..fb79cc93` on `qa` with the same inputs.
- **T-perturb:** every `TALLY` line of `context8k` must equal `F_old.out`. Any difference is a regression, handled the same way.
- **M18's gate:** every `qa` configuration at least 10 MB under 200 MB, so at most 190 MB (VmHWM).
- Nothing is tuned. No expert code is touched.

## Predictions

| # | Prediction |
|---|---|
| P1 | Dev answers equal R1: 3 right, 2 wrong, 125 abstained (gate); 6 / 56 / 68 before it. The rows file is byte-identical to R1's |
| P2 | `context8k`'s whole output equals `F_old.out` under `cmp`: `context8k_bpc` 1.4962, every T-perturb tally the same |
| P3 | `qa` peaks: Q1 and Q2 188–198 MB, Q3 184–194 MB, Q4 150–185 MB. M18 stays open (C1L built and served above 190 MB) |
| P4 | The writer's dev GEN file equals L1's (`cmp`, the MODEL line aside); SCRIBE v2 0.610 |
| P5 | The card's headline lines: `context8k_bpc` for `context8k`, and for `qa` the fullest holdout line, which now includes the doc cache (`holdout2_trie_ig_cls_sim_copy_doc_bpc`, 1.6974 in R1) |

## The headline line (ledger 118)

Ledger 118 left "the report card's headline line" open. The card's reader already takes `context8k_bpc` for `context8k` (2026-11-04). Two things still need a fix, in the docs and the card's own reader only:

- REPORT_CARD.md rule 5 and CURRENT.md still name the old lines.
- The card's `qa` list of lines stops at `holdout2_trie_ig_cls_sim_copy_bpc`. It does not know the doc cache's line (GEN-C2), so it would report 1.7256 as the fullest line instead of 1.6974.

## Results

(To be filled after the runs.)
