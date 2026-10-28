# B4c: the class and copy experts gated out of answers

**Odin Loch** · 2026-10-28 · PLAN B4 and B9 · follows [B4B_LM_DEFAULTS.md](B4B_LM_DEFAULTS.md) and [A4_SIMILAR_CONTEXT_EXPERT.md](A4_SIMILAR_CONTEXT_EXPERT.md) (A4b)

(The decision rule and the predictions were written and committed before the runs. Results and the decision are added under them.)

**Question.** B4B kept the class and copy experts out of the defaults. They lower BPC a lot (fresh text 2.166 → 1.912), but with the fact shard they make CyphaLM echo the question or let a wrong answer through (rule 3). The similar-context expert had the same problem. A4b turned it off inside answers and made it a default. Odin approved the same gate for class and copy (2026-10-28). Does the gate fix rule 3? Where do the two experts become defaults?

## What changes

- **The gate.** The class and copy experts are off inside an answer, as the similar-context expert is since A4b. An answer runs after "Assistant: " until the next "User: ", and from the end of qa's prompt (`in_answer`).
- **Defaults** where the rule below holds: `qa`, `sample`, `context8k`. `chat`, `probe` and `bpc` never build them (`chat` and `probe` write only answers; `bpc`'s lines are the trie's reference numbers).
- **Switches.** `--no-cls` and `--no-copy`, like `--no-sim`. With all three, no default expert is built.
- **Settings.** Class: `shape` (= `shape:6:3:cp2`, Odin's default table). Copy: minimum match m = 8 and a 16,384-byte window (B4B's choice; the code had 4,096).
- **qa's `--copy`.** It keeps its two extra models (`trie_cp`, `trie_ig_cp`), whose copy expert is *not* gated: A1c's answer copy, for comparison only, as the similar-context spec's `gate=0`. The default models carry the gated copy. A1c's and A2's `--copy` numbers used m 0 and a 4,096-byte window; they reproduce at their own commits (`--copy --copy-min 0` gives m 0; the window is now 16,384 everywhere).

## Decision rule (fixed before the runs)

B4B's four rules, read per subcommand:

1. **BPC.** Already read in B4B on fresh text: class (shape, 3 orders) and copy (m 8, window 16,384) lower trie + InfiniGram BPC with LCI ≥ 0.95. Not read again. qa's and context8k's holdout lines are printed as a check.
2. **Text.** Already read in B4B: surface S does not fall with either expert (class +0.144; copy m 8 +0.0016 [−0.0035, +0.0069] over class). SCRIBE v2 is also read on `sample`'s text. An expert stays out of `sample`'s default if its SCRIBE v2 change is entirely below zero (95% intervals of the pooled rung, compared as printed).
3. **Answers.** Read here. `qa` in the C1L setting (A1d's options with the D4 fact shard, without `--copy` and `--cls`, so the defaults decide), defaults on against defaults off. Coverage at 2% wrong on dev (5-fold CV; trie and trie + InfiniGram; chat and wiki forms) does not fall, and the wrong rate's Wilson upper bound does not rise. The frozen test is read once, after the dev decision: the chosen setting and its defaults-off reference in one step. Nothing is chosen from it.
4. **Cost.** Peak RSS and wall time of `qa` and `sample` with the defaults, inside PLAN §10 (whole language system ≤ 4 GB resident; Q/A p50 ≤ 200 ms on one core). Recorded.

**Parity (a check, not a rule).** With `--no-cls --no-copy --no-sim`, `qa`'s printed lines and rows file, `sample`'s files and `context8k`'s lines equal the 22ef43b binary's (main 6ad7198) byte for byte. With `--no-cls --no-copy` alone, `sample`'s files and `context8k`'s lines equal 22ef43b's defaults (A4b).

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| E1 | Gated, the trie and trie + IG answers are byte-identical with the defaults on and off, for all 130 dev questions in both forms: the answer text, grade, mean and min log p, first-byte entropy and InfiniGram match. Only `question_lp` changes (the question is outside the answer) |
| E2 | Rule 3 holds. Dev, trie + IG, chat form: the out-of-fold gate keeps 2 right, 0 wrong in both arms |
| E3 | qa holdout (second half, the copy line's first 512 KiB): trie + IG + class + sim + copy 1.82–1.89 BPC (trie + IG 2.0716; A4b's sim alone 1.9686) |
| E4 | `sample` SCRIBE v2 (pooled rung): trie + class + sim 0.08–0.25, its interval above trie + sim's 0.061 [0.049, 0.074]. Adding copy moves it by less than 0.02 |
| E5 | `context8k`: the trie + IG + class + sim + copy line (whole second half) is 1.83–1.91 BPC (trie + IG + sim: 1.9686) |
| E6 | Cost: `qa` peak ≤ 300 MB and ≤ 15 min; `sample` peak ≤ 220 MB and ≤ 8 min; qa decode p50 within ±50% of A4b's (inside answers all three experts are skipped) |

---

## Results (runs 2026-10-28)

The tool at 26a4b28 plus the gates (629c45c); `sample`'s class expert became opt-in after the SCRIBE read (the final commit). Runs one at a time on the shared 4-core machine, so wall times include waiting on the other agents' jobs.

### Parity

| Check | Result |
|---|---|
| `sample --no-cls --no-copy --no-sim` against 22ef43b `sample --no-sim` | both files and stdout byte-identical (and identical to A4b's `trie_dev.txt`) |
| `sample --no-cls --no-copy` against 22ef43b `sample` | both files byte-identical (and identical to A4b's `sim_dev.txt`) |
| `qa` C1L `--no-cls --no-copy --no-sim` against 22ef43b `--no-sim` | rows file byte-identical; stdout identical once the `LATENCY` lines and `build_s` are masked |
| `qa` C1L `--no-cls --no-copy` against 22ef43b's default | rows file byte-identical; stdout differs only in `SIM_LAT` (a timing line; the same 157,028 queries) |
| `context8k --no-cls --no-copy` against 22ef43b's default | stdout byte-identical |
| final `sample` (class opt-in) against `sample --no-cls` of 629c45c | both files byte-identical (trie + sim + copy) |
| final `sample --cls shape` against 629c45c's `sample` | both files byte-identical (trie + class + sim + copy) |
| CTests (HANDOFF §4.4) | 8 of 8 pass; the selftest gains the gate checks |

### Rule 3: answers (qa, C1L setting, dev)

`$C1LD` is A1d's C1L options without `--copy` and `--cls`: the D4 fact shard, 20% don't-know turns, the `low`-only token after "Assistant: ".

- **Answers do not change (E1 hit).** All 520 rows (130 questions × 2 forms × trie and trie + IG) carry the same answer text, grade, mean and min log p, first-byte entropy and InfiniGram match in all three arms: defaults on, similar-context only (the current default), and no expert. Only `question_lp` changes: the question is outside the answer, so the experts read it.
- **The gate does not change (E2 hit).**

| Arm | chat, trie + IG: out of fold right / wrong (coverage) | Wilson upper bound, wrong | Full-dev gate | Other models and forms |
|---|---|---|---|---|
| No expert | 2 / 0 (0.0154) | 0.0287 | `min_lp ≥ −0.8954`, 3 right | 0 / 0 |
| Similar-context only | 2 / 0 (0.0154) | 0.0287 | the same | 0 / 0 |
| **Class + similar-context + copy** | **2 / 0 (0.0154)** | **0.0287** | the same | 0 / 0 |

- The only other stdout change is the risk-coverage curve over `question_lp` (at 30 answered: 5 right → 4; at 50: 6 → 7). The gate never picks `question_lp` here.
- **Rule 3 holds.** The gate is the fix for B4B's echo: with the experts out of the answer, the answer is the one CyphaLM wrote without them.

### The frozen test (read once, after the dev decision)

The chosen setting (class + similar-context + copy) and its reference (similar-context only, the current default), in one step, with `--parse parse_all_span.tsv --test t_qa_wiki.tsv` in place of the dev parse and `--leak`.

- **All 800 test rows (200 questions × 2 forms × trie and trie + IG) carry the same answers and grades in both arms.** The dev rows are again identical too.
- **Every test line is identical between the arms.** The full-dev gate keeps 0 right and 0 wrong on the test (Wilson upper bound of the wrong rate 0.0188), as for C1L before. Raw right: chat, trie + IG, 2 of 200.
- The don't-know token on the test: trie precision 0.635, recall 0.94 (as A1d); trie + IG 0.495 / 0.96.
- Peak RSS of the test step: 298.9 MB (defaults), 229.2 MB (reference).

### BPC (rule 1 check)

qa's chat model (C1L), holdout second half, lambdas on the first half; each expert fit over the distribution it is mixed into:

| Model | BPC | Gain of the last expert, LCI (90%) |
|---|---:|---|
| trie + IG | 2.072881 | |
| + class | 1.994441 | |
| + class + similar-context | 1.922375 | sim given class: −0.0720 [−0.0764, −0.0683], P(gain > 0.005) = 1.000 |
| + class + similar-context + copy (first 512 KiB of the second half) | **1.834525** | copy: −0.0877 [−0.1000, −0.0767], P(gain > 0.005) = 1.000 |
| similar-context alone (A4b default) | 1.968237 | |

`context8k` (its own chat model; 30 Dolly turns held out), whole second half: trie + IG 2.0715 → + class 1.9940 → + similar-context 1.9231 → + copy **1.8345**. Its answers and tallies are byte-identical to 22ef43b's.

The copy lambdas (m 8) are 0 below 8 bytes, then 0.295 / 0.440 / 0.580 / 0.695 (trie + IG) for matches of 8–11, 12–15, 16–31 and 32+ bytes.

### Text: `sample` and SCRIBE v2

A4b's ladder (`scribe ladder … --test enwik:enwik8:56:64 --profile v2`) on `sample`'s dev file. The trie and trie + sim files are byte-identical to A4b's, so their reads are A4b's.

| Text | n | SCRIBE v2 [95%] | Surface S [95%] |
|---|---:|---|---|
| held-out human | 500 | 0.923 [0.910, 0.935] | 0.912 |
| words shuffled within sentences | 500 | 0.003 [0.003, 0.003] | 0.923 |
| trie | 623 | 0.014 [0.009, 0.021] | 0.734 |
| trie + class | 657 | **0.003 [0.002, 0.003]** | 0.855 |
| trie + sim (A4b default) | 605 | 0.061 [0.049, 0.074] | 0.801 |
| trie + class + sim | 623 | **0.025 [0.019, 0.033]** | 0.880 |
| trie + sim + copy | 613 | 0.056 [0.044, 0.068] | 0.790 |
| trie + class + sim + copy | 631 | 0.028 [0.019, 0.037] | 0.877 |

- **SCRIBE v2 says the class expert makes the text worse.** Alone it brings the trie's text down to the "words shuffled within sentences" rung (0.014 → 0.003). On top of sim it halves the score (0.061 → 0.025, disjoint intervals).
- **Surface S says the opposite** (0.734 → 0.855; 0.801 → 0.880), as it did in B4B. B4B's SCRIBE v1 read also rose with the class expert (0.028 → 0.074). The two text scores disagree; which one the rules use is Odin's open decision (HANDOFF §5.6).
- **The copy expert changes nothing measurable** (0.061 → 0.056, overlapping; S 0.801 → 0.790).
- **E4: miss.** Class + sim scored 0.025, not 0.08–0.25 above sim's 0.061.

### Cost (rule 4)

| Run | Peak RSS | Wall time |
|---|---:|---:|
| `qa` C1L, no expert | 204.0 MB | 554 s |
| `qa` C1L, similar-context only (the current default) | 228.4 MB | 682 s |
| **`qa` C1L, class + similar-context + copy** | **274.6 MB** | **736 s** |
| `context8k`, similar-context only | 174.6 MB | 622 s |
| **`context8k`, class + similar-context + copy** | **248.8 MB** | **723 s** |
| `sample`, trie + sim (A4b) | 82.5 MB | 353 s |
| **`sample`, trie + sim + copy** | **82.8 MB** | **321 s** |
| `sample`, trie + class + sim + copy | 99.7 MB | 415 s |

- The class table: 1,596,160 contexts and 19.5 MB in `qa` (wiki train + chat); 885,985 contexts, 10.8 MB and 5 s in `sample` (wiki train).
- qa decode p50: 1.0–1.4 ms in every arm (A4b: 0.9–1.2 ms). Inside answers all three experts are skipped.
- Everything is far inside PLAN §10. `qa`'s peak (275 MB) is above the 200 MB that RAM_PROFILE's next step aims at; the class build adds about 46 MB at its peak.
- **E6: hit** (qa ≤ 300 MB and ≤ 15 min; sample ≤ 220 MB and ≤ 8 min; decode p50 within ±50%).

### Predictions, scored

| # | Result |
|---|---|
| E1 | **Hit.** 520 of 520 answers identical; only `question_lp` changes |
| E2 | **Hit.** 2 right, 0 wrong out of fold in all three arms |
| E3 | **Hit.** 1.8345 (range 1.82–1.89) |
| E4 | **Miss.** Class + sim 0.025, below sim's 0.061; copy moved it by 0.003 (that part hit) |
| E5 | **Hit**, at the edge: 1.8345 (range 1.83–1.91) |
| E6 | **Hit** |

## Decision (2026-10-28)

- **`qa` and `context8k`: class + similar-context + copy are the defaults.** Rules 1, 3 and 4 hold. Rule 2 does not apply, because these subcommands write only answers, and answers are written without the experts. Every answer is byte-identical on dev and on the frozen test, and holdout BPC falls from 2.0729 (trie + IG) to 1.8345.
- **`sample`: similar-context + copy are the defaults; the class expert is not.** By the rule fixed before the runs, an expert stays out of `sample` if its SCRIBE v2 change is entirely below zero. The class expert's is (0.061 → 0.025), so it stays out. The copy expert's is not (0.061 → 0.056, overlapping), so it goes in. `--cls SPEC` still adds the class expert to `sample`.
- **qa's `--copy`** keeps its two models with the copy expert on inside answers (A1c's answer copy), for comparison only. They share the default copy's lambdas, so `--copy-min 0` gives A1c's m 0. No current report's numbers need the old 4,096-byte window; A1c's, A2's and A1d's `--copy` numbers reproduce at their own commits.
- **For Odin:**
  - The two text scores disagree about the class expert. Surface S rises with it; SCRIBE v2 falls to the shuffled-words rung. HANDOFF §5.6 asks which score the rules use. If it is surface S, the class expert would join `sample`'s default too.
  - HANDOFF §5.5 (the same gate for class and copy) is answered by this report: yes, and rule 3 holds with it.

### Reproduce

```bash
# build: cmake -S native -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target cyphalm_trie
B=build/cyphalm_trie; E=enwik8; D=dolly.jsonl   # Dolly 15,015 records
C1LD="--dev bench/eval/t_qa_wiki_dev.tsv --leak bench/eval/t_qa_wiki.tsv --parse parse_dev_span.tsv \
  --dk dk_mention.tsv --dk-ratio 0.20 --fam after --fam-stat ent_title --fam-train low --shard shard_facts.tsv \
  --base a0_rows.tsv"                            # A1d's C1L inputs (A1D_A2B_ANSWERING_VARIANTS.md), without --copy and --cls
$B qa $E $D $C1LD --out E_qa.rows.tsv            # defaults: class + similar-context + copy
$B qa $E $D $C1LD --no-cls --no-copy --out N_qasim.rows.tsv            # the reference (A4b's default)
$B qa $E $D $C1LD --no-cls --no-copy --no-sim --out N_qaoff.rows.tsv   # no expert (= 22ef43b --no-sim)
# the test step: parse_all_span.tsv and --test bench/eval/t_qa_wiki.tsv in place of the dev parse and --leak
$B context8k $E $D                               # and --no-cls --no-copy for the reference
$B sample $E E_s_dev.txt E_s_fresh.txt           # trie + sim + copy; --cls shape adds the class expert
scribe ladder --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors vec32 --profile v2 \
  --seed 20261027 --gen-trie E_s_dev.txt --label trie_sim_copy
```

Scratch results: `/root/work/echo/runs/` (lost with the container; the commands above rebuild them).
