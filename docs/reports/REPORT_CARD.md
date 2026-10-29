# The report card: CyphaLM's evaluation contract on one page

**Odin Loch** · 2026-10-29 · bill of work M1–M4 (`cypha_report_card`) · PLAN §1 ("held-out answers + surface S + BPC + SCRIBE + ECS + LCI, reported together")

(What the card measures, its rules and the predictions for today's card were written before the first card run. Results are added under them.)

**What it is.** One command runs CyphaLM's frozen suites for one configuration and writes one card: a dated Markdown table with the git commit and the exact flags, and a TSV with the same numbers. Later reports cite one card per change. Rule 5 of Odin's standing instructions holds on every card: answers, SCRIBE v2 with surface S alongside, and BPC, together.

**Rule 10.** The card measures CyphaLM. It answers nothing itself and changes nothing in the model.

## The parts

| Part | What runs | What the card reads |
|---|---|---|
| `qa` | `cyphalm_trie qa ENWIK8 DOLLY PROTOCOL FLAGS --out L.qa.rows.tsv` | the holdout BPC stack, the experts' paired gain lines, the dev answer tallies, the don't-know (abstain) line, and with `--read-test` the frozen test tallies |
| `ctx` | `cyphalm_trie context8k ENWIK8 DOLLY FLAGS` | the context8k BPC stack |
| `gen` | `cyphalm_trie sample ENWIK8 L.gen_dev.txt L.gen_fresh.txt FLAGS`, then `scribe ladder` (SCRIBE v2) on the dev file with `--export` | the rung means (human, the samples pooled and by temperature, sentences shuffled across documents), surface S, and per-sample scores for paired comparisons |
| `fresh` (only with `--fresh`) | `scribe ladder` on the fresh file with the fresh ladder (`--test enwik:E:64:72 --seed 31415926`) | SCRIBE and surface S on fresh prompts, and LCI's γ |
| every part | VmHWM polled every second (the method of `/root/work/r2/peak.sh`) | peak RSS and wall time |

**Protocols for `qa`.**
- **`c1l`** (the default, the configuration behind CURRENT's numbers): A1d's C1L inputs, `--dev bench/eval/t_qa_wiki_dev.tsv --leak bench/eval/t_qa_wiki.tsv --parse parse_dev_span.tsv --dk dk_mention.tsv --dk-ratio 0.20 --fam after --fam-stat ent_title --fam-train low --shard shard_facts.tsv --base a0_rows.tsv`. The four scratch inputs are named by environment variables (defaults: the container paths of A1d, D4 and qa2).
- **`a0`**: only the repository's files, `--dev bench/eval/t_qa_wiki_dev.tsv --leak bench/eval/t_qa_wiki.tsv`.
- With `--read-test`, `--leak FILE` becomes `--test FILE` and the dev parse becomes `parse_all_span.tsv` (B4c's test step).

**Generation source.** `--gen-source a4d` uses `sample`'s own 250 prompts and the A4d / B4c pipeline. `--gen-source gena` uses GEN-A's frozen generation benchmark through `scripts/gen_eval.sh` once `/root/work/gena/HARNESS_READY` exists and the script is in the tree; `auto` picks `gena` when both hold, else `a4d`. The card states which source it used.

## Registered rules (2026-10-29, before the first card run)

1. **Dev by default.** Every number is dev unless the card says otherwise. The T-QA frozen test is read only with `--read-test`, and each such run appends a line to the Reads table below (date, label, commit, flags). Fresh text is read only with `--fresh`.
2. **Intervals on every number:**
   - **proportions** (answers right, wrong, coverage; don't-know precision and recall): Wilson 95% (`ecs.hpp`, `wilson_lower` / `wilson_upper`);
   - **SCRIBE v2 and surface S**: the 95% bootstrap interval of the mean that `scribe ladder` prints;
   - **BPC**: the point on the fixed holdout, with each expert's paired 90% block-bootstrap gain interval as `cyphalm_trie` prints it (`sim_lci`, `copy_lci`, `cls_given_copy_lci`). `cyphalm_trie` writes no per-block bits, so an absolute-BPC interval is not available; the card says so in the row (a one-flag follow-up in `cyphalm_trie`);
   - **peak RSS and time**: one run, no interval; the card says so.
3. **Comparison with a baseline** (`--baseline-flags` or `--baseline-label`), dev only:
   - **answers**: paired per T-QA dev question (chat form, trie + InfiniGram), Bayesian bootstrap (`lci.hpp`, 4,000 draws, 90%) of the difference in raw right and raw wrong answers;
   - **SCRIBE v2**: paired per (prompt, temperature) present in both runs after SCRIBE's own cleaning and 100-byte rule; Bayesian bootstrap of the difference;
   - **BPC**: the difference of the points only (no paired interval, rule 2);
   - **LCI** (`compute_lci`): P_gain on the paired SCRIBE differences (ε 0); canaries: answers not wrong and answers right, each non-inferior within 1 point (δ 0.01); γ from the fresh SCRIBE gain against the dev gain, **only with `--fresh`** (otherwise γ and LCI are "not read" and the card stops at P_gain, r, σ, κ); c = 1 (generation has no calibration read); σ from the gains at the three sampling temperatures, which stand in for seeds (the card says so); κ = 1 if the candidate's built `qa` peak is within the RAM budget (200 MB, Odin's RAM request), else 0; the n_eff cap of 0.5 below 200 pairs.
4. **ECS on the card** is the abstain side of PLAN §7.4: don't-know precision and recall against questions outside the 8 MiB slice, the in-slice share CyphaLM should have refused, and the `<|unknown …|>` tag counts (complete, reason right), as `cyphalm_trie qa` prints them.
5. **Headline lines.** BPC: `qa`'s fullest holdout stack line present (with the defaults, `holdout2_trie_ig_cls_sim_copy_bpc`, which `qa` scores on the first 512 KiB of the holdout's second half) and context8k's fullest line (`trie_plus_context_8192_cls_sim_copy_bpc`). Answers: `CV chat/trie_ig all` (the out-of-fold gate at 2% wrong, 5-fold inside dev).
6. **Nothing is tuned on the card.** It reports; choices stay in each change's own registered rules.

## Predictions for today's card (main `ba5bef7`, defaults, `c1l`, `a4d` source, dev only)

| # | Prediction |
|---|---|
| R1 | `qa` holdout (defaults, with copy) 1.7895 to the printed digit (CURRENT, A4d) |
| R2 | context8k 1.7897 to the printed digit |
| R3 | Dev answers: 2 right, 0 wrong, 128 abstained out of 130 (A4d's VQ run) |
| R4 | SCRIBE v2 of the samples pooled 0.063 [0.051, 0.078], surface S 0.860, human 0.923, sentences shuffled 0.344 (A4d's SO run; `sample`'s default is SO's spec since A4d) |
| R5 | Built `qa` peak within 185 ± 5 MB (M16; ANS's parity run 184.6 MB) |
| R6 | The selftest catches three planted bugs in the summariser: Wilson with the wrong z, a SCRIBE pairing that ignores dropped samples, and a test tally read into a dev card |

## Today's card (2026-10-29)

**Configuration.** Main's defaults (no flags), protocol `c1l`, text source `a4d`, dev only. No fresh text and no frozen test were read. `qa` and `context8k` ran at `bf0cfec`, `sample` and `scribe` at `78f9a8f`. Both commits carry main `ba5bef7`'s `cyphalm_trie.cpp` and `scribe.cpp`; they add only the card's own files. The same numbers as a TSV: [`report_card/card_20261029.tsv`](report_card/card_20261029.tsv).

**Command.**

```
scripts/cypha_report_card.sh --label card_20261029 --out /root/work/rcard/runs
```

The binaries came from a Release build of the same tree (`CYPHALM_TRIE`, `SCRIBE` and `REPORT_CARD` pointed at it). The background shell's two-hour limit stopped the first run during `sample`. The same command with `--reuse` finished it; `qa` and `context8k` were kept.

### BPC

| Metric | Value | Interval | n | Split | Note |
|---|---:|---|---:|---|---|
| qa holdout, the configuration's fullest stack | 1.7895 | — |  | dev | fixed holdout; no absolute interval (cyphalm_trie writes no per-block bits); line holdout2_trie_ig_cls_sim_copy_bpc (qa scores copy on the first 512 KiB of the holdout's second half) |
| qa holdout, trie + InfiniGram | 2.0729 | — |  | dev | fixed holdout; no absolute interval (cyphalm_trie writes no per-block bits) |
| gain sim_lci trie_cls (paired, 90%) | 0.2404 | [0.2278, 0.2534] |  | dev | cyphalm_trie's block bootstrap |
| gain sim_lci trie_ig_cls (paired, 90%) | 0.1546 | [0.1455, 0.1647] |  | dev | cyphalm_trie's block bootstrap |
| gain copy_lci trie_cls_sim (paired, 90%) | 0.0562 | [0.0498, 0.0636] |  | dev | cyphalm_trie's block bootstrap |
| gain copy_lci trie_ig_cls_sim (paired, 90%) | 0.0503 | [0.0443, 0.0571] |  | dev | cyphalm_trie's block bootstrap |
| gain cls_given_copy_lci trie (paired, 90%) | 0.3113 | [0.3007, 0.3230] |  | dev | cyphalm_trie's block bootstrap |
| gain cls_given_copy_lci trie_ig (paired, 90%) | 0.1788 | [0.1727, 0.1853] |  | dev | cyphalm_trie's block bootstrap |
| context8k, the configuration's fullest stack | 1.7897 | — |  | dev | fixed holdout; no absolute interval (cyphalm_trie writes no per-block bits); line trie_plus_context_8192_cls_sim_copy_bpc |

### Answers (dev)

| Metric | Value | Interval | n | Split | Note |
|---|---:|---|---:|---|---|
| right, kept by the gate (CV chat/trie_ig all) | 0.015 | [0.004, 0.054] | 130 | dev | 2 of 130 |
| wrong, kept by the gate | 0.000 | [0.000, 0.029] | 130 | dev | 0 of 130; the gate's target is 2% (chosen out of fold, 5-fold inside dev) |
| abstained (the gate or CyphaLM's own don't-know) | 0.985 | [0.946, 0.996] | 130 | dev | 128 of 130 |
| coverage (answered) | 0.015 | [0.004, 0.054] | 130 | dev | 2 of 130 |
| right, before the gate | 0.054 | [0.026, 0.107] | 130 | dev | 7 of 130 |
| wrong, before the gate | 0.438 | [0.356, 0.524] | 130 | dev | 57 of 130 |
| abstained, before the gate (CyphaLM's own don't-know) | 0.508 | [0.423, 0.592] | 130 | dev | 66 of 130 |

### ECS: abstaining (dev)

| Metric | Value | Interval | n | Split | Note |
|---|---:|---|---:|---|---|
| don't-know precision (abstained and outside the slice) | 0.455 | [0.340, 0.574] | 66 | dev | 30 of 66 |
| don't-know recall (questions outside the slice refused) | 1.000 | [0.886, 1.000] | 30 | dev | 30 of 30 |
| refused, of the in-slice questions the base rows (--base) get wrong | 0.367 | [0.279, 0.466] | 98 | dev | 36 of 98 |
| <|unknown|> tags complete | 0.429 | [0.300, 0.567] | 49 | dev | 21 of 49 |
| <|unknown|> tags with the right reason | 0.612 | [0.472, 0.736] | 49 | dev | 30 of 49 |

### Text (SCRIBE v2, surface S)

| Metric | Value | Interval | n | Split | Note |
|---|---:|---|---:|---|---|
| CyphaLM samples, pooled: SCRIBE v2 | 0.063 | [0.051, 0.078] | 608 | dev | scribe's 95% bootstrap |
| CyphaLM samples, pooled: surface S | 0.860 | [0.854, 0.867] | 608 | dev | scribe's 95% bootstrap |
| CyphaLM samples, T 0.6: SCRIBE v2 | 0.081 | [0.056, 0.109] | 204 | dev | scribe's 95% bootstrap |
| CyphaLM samples, T 0.6: surface S | 0.905 | [0.898, 0.911] | 204 | dev | scribe's 95% bootstrap |
| CyphaLM samples, T 0.8: SCRIBE v2 | 0.070 | [0.046, 0.096] | 214 | dev | scribe's 95% bootstrap |
| CyphaLM samples, T 0.8: surface S | 0.881 | [0.873, 0.888] | 214 | dev | scribe's 95% bootstrap |
| CyphaLM samples, T 1.0: SCRIBE v2 | 0.037 | [0.023, 0.055] | 190 | dev | scribe's 95% bootstrap |
| CyphaLM samples, T 1.0: surface S | 0.790 | [0.779, 0.801] | 190 | dev | scribe's 95% bootstrap |
| human text (reference rung): SCRIBE v2 | 0.923 | [0.910, 0.935] | 500 | dev | scribe's 95% bootstrap |
| human text (reference rung): surface S | 0.912 | [0.906, 0.918] | 500 | dev | scribe's 95% bootstrap |
| sentences shuffled across documents (control rung): SCRIBE v2 | 0.344 | [0.333, 0.356] | 500 | dev | scribe's 95% bootstrap |
| sentences shuffled across documents (control rung): surface S | 0.905 | [0.901, 0.908] | 500 | dev | scribe's 95% bootstrap |

### Cost

| Metric | Value | Interval | n | Split | Note |
|---|---:|---|---:|---|---|
| qa peak RSS (MB) | 184.7 | — |  |  | one run; no interval |
| qa wall time (s) | 1958 | — |  |  | one run on a shared machine |
| context8k peak RSS (MB) | 202.3 | — |  |  | one run; no interval |
| context8k wall time (s) | 2192 | — |  |  | one run on a shared machine |
| sample peak RSS (MB) | 69.8 | — |  |  | one run; no interval |
| sample wall time (s) | 3556 | — |  |  | one run on a shared machine |
| scribe ladder peak RSS (MB) | 400.2 | — |  |  | one run; no interval |
| scribe ladder wall time (s) | 29 | — |  |  | one run on a shared machine |

**What the card says.**
- **Answers.** The gate keeps 2 right and 0 wrong answers out of 130 dev questions and abstains on 128. Before the gate CyphaLM writes 57 wrong answers (0.438 [0.356, 0.524]); its own don't-know message covers 66.
- **Don't know.** CyphaLM refuses all 30 questions whose subject is outside its 8 MiB slice (recall 1.000 [0.886, 1.000]). Fewer than half its refusals are of that kind (precision 0.455 [0.340, 0.574]).
- **Text.** SCRIBE v2 rates the samples 0.063 [0.051, 0.078] against 0.923 for human text. Sentences shuffled across documents score 0.344, well above CyphaLM: by SCRIBE the samples are still word salad. Surface S (0.860, human 0.912) does not see this.
- **BPC.** 1.7895 on the `qa` holdout and 1.7897 on `context8k`. Every expert's paired gain is positive, with 90% intervals clear of zero.
- **Cost.** Every CyphaLM part peaks under 210 MB. `scribe ladder` peaks at 400 MB (below).

## Predictions scored

| # | Prediction | Result | |
|---|---|---|---|
| R1 | `qa` holdout 1.7895 | 1.7895. Every line of `qa`'s output equals A4d's `VQ` run except timings, and the rows file is byte-identical | **hit** |
| R2 | context8k 1.7897 | 1.7897. The whole output equals A4d's `VK` run | **hit** |
| R3 | Dev answers 2 right, 0 wrong, 128 abstained of 130 | 2, 0, 128 of 130 | **hit** |
| R4 | SCRIBE v2 pooled 0.063 [0.051, 0.078], surface S 0.860, human 0.923, shuffled 0.344 | All four to the printed digit. `sample`'s dev file is byte-identical to A4d's `VS` file; every SCRIBE table equals A4d's `SO` ladder | **hit** |
| R5 | Built `qa` peak 185 ± 5 MB | 184.7 MB | **hit** |
| R6 | The selftest catches three planted bugs | Wilson with z 1.645: 3 checks fail. Pairing that keeps samples under 100 bytes: 4 fail. The test tally read into a dev card: 1 fails. Each exits 1 | **hit** |

## Checks

- **CTests.** The nine of HANDOFF §4.4 named in the brief and the new `native_cypha_report_card_selftest` (26 checks): 10 of 10 pass. The four later additions to §4.4 (`native_cyphalm_tok_selftest`, `native_token_bridge_test`, `native_thought_bindings_test`, `native_thought_k6_test`) also pass.
- **Real outputs.** The summariser read A4d's own output files before the first run and gave CURRENT's numbers. A comparison of today's card with itself pairs all 608 samples and all 130 questions; every difference is 0, both canaries pass, and γ and LCI read "not read" (no `--fresh`).
- **The script's paths** (stub binaries, no corpus): `--read-test` swaps `--leak` for `--test` and the dev parse for `parse_all_span.tsv`; a baseline stays on dev; one line is added to the Reads table; the baseline's card shows no test tally. A part that exits non-zero is not read, and its Cost row says so.

## Limits

- **Time.** Today's card took 2 h 9 min (`qa` 1,958 s, `context8k` 2,192 s, `sample` 3,556 s, `scribe` 29 s) on 4 cores shared with other runs (load 8 to 30). The same parts took about 35 minutes in A4d's runs. PLAN §10 sets 15 minutes for a card run, so the card misses its budget. Two ways to meet it: serve `qa` from a stored model (`qa --model DIR`, 3–5× faster, M10–M14), and run the three parts side by side (each under 210 MB).
- **RAM.** `scribe ladder` peaks at 400 MB for 29 s, over this run's 300 MB budget. That is SCRIBE's Wikipedia pack (SCRIBE v2's report: 394–415 MB), not the card. CyphaLM's parts stay under 210 MB.
- **BPC intervals.** An absolute BPC interval needs per-block bits from `cyphalm_trie`. That is one flag in `cyphalm_trie.cpp`, which this work did not touch.
- **ECS.** The card has the abstain side only. Answer calibration (ECE) needs a confidence per answer in `qa`'s rows file, which it does not write.
- **GEN-A's harness** was not in this checkout, so `auto` chose `a4d`. `--gen-source gena` is untested. The card reads the harness's ladder table; a paired SCRIBE comparison also needs a per-sample export, which `gen_eval.sh` does not write yet.
- **Frozen-test runs** use `parse_all_span.tsv` for the candidate and `parse_dev_span.tsv` for a baseline. The dev questions' parses should be the same in both; the card does not check this.

## How to run it

Build the three tools once:

```
cmake --build native/build --target cyphalm_trie scribe cypha_report_card
```

Then one command makes a card:

```
scripts/cypha_report_card.sh --label my_change -- --sim default,online=0
```

- Everything after `--` goes to every `cyphalm_trie` part (`qa`, `context8k`, `sample`). With no flags, the card reads main's defaults.
- The card is `build/report_card/my_change.card.md`, and the same numbers are in `my_change.card.tsv` (columns: label, commit, date, section, metric, value, lo, hi, n, split, note). `--out DIR` puts them elsewhere.
- **Compare with a baseline:** `--baseline-flags "FLAGS"` also runs the baseline (as `my_change_base`); `--baseline-label card_20261029` compares with a run already in the same folder. The comparison is dev only (rule 3).
- **Fresh prompts:** `--fresh` adds SCRIBE on `sample`'s fresh prompts, and with a baseline it reads LCI's γ. Read fresh text once per change.
- **The frozen test:** `--read-test` reads T-QA's frozen test and appends a line to the Reads table at the end of this file. Commit that line with the change. Read the test once per change.
- **Only some parts:** `--parts qa`, `--parts qa,ctx` or `--parts gen`.
- **Resume:** `--reuse` skips a part that already finished with exit code 0.
- **Protocol:** `--protocol c1l` (the default) or `--protocol a0` (only the repository's files).
- **Text source:** `--gen-source auto|a4d|gena`. Until GEN-A's harness is in the tree, `auto` means `a4d`.
- **Paths:** the binaries are `native/build/…` unless `CYPHALM_TRIE`, `SCRIBE` and `REPORT_CARD` say otherwise. The corpora and C1L inputs are named by `ENWIK8`, `DOLLY`, `SCRIBE_VEC`, `QA_PARSE_DEV`, `QA_PARSE_ALL`, `QA_DK`, `QA_SHARD` and `QA_BASE` (defaults: the container paths). The script's header lists them.
- **The summariser on its own:** `cypha_report_card summarize --dir DIR --label L [--baseline B] [--budget-mb 200]` reads the part outputs and writes the card again. `cypha_report_card selftest` is the CTest `native_cypha_report_card_selftest`.

A part that exits with an error is not read. The card's Cost rows say which part failed.

## Reads

Every `--read-test` run of the T-QA frozen test through the card adds a line here.

| Date (UTC) | Label | Commit | Flags |
|---|---|---|---|
