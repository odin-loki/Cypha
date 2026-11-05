# The fact store over enwik9

**Odin Loch** · 2026-11-05 · GENERATION_PLAN §13.38 A

CyphaLM's fact store came from enwik8. I ran `pattern_learn` (F2) over enwik9 past enwik8, with the registered leakage guards, and merged the new facts beside the old ones. Coverage of the 250 GEN-A dev prompts rose from 17.2% to 20.4%. The intervals overlap. The 30% target was missed.

## Commands

The build is plain g++ in WSL (`~/build_fs2`). `pattern_learn` gained two input options and nothing else: `--page-range LO:HI` and `--skip-pages FILE`. The learning rules are unchanged.

```
fs_skip_pages enwik9 100000000 skip_pages.tsv gena_texts.txt scribe_reftest.txt      # the guard's page list
pattern_learn enwik9 e9_raw --cv --page-range 100000000:1000000000 --skip-pages skip_offsets.txt
pattern_learn enwik8 e8_textonly --cv                                                # control, same seeds kind
python3 -I scripts/fs_merge_store.py bench/fact_store/verify_f2r_aug e9_raw bench/fact_store/e9_aug enwik8 enwik9
fs_coverage STORE,bench/fact_store/patterns_raw.tsv bench/eval/gen_prompts_dev.tsv --plan-drop 75497472:100000000
```

The raw outputs are in `/mnt/d/cypha_llm/fact_store_e9/`. The merged store is in git: `bench/fact_store/e9_aug/` (11 MB).

## Guards

- **Overlap pages skipped whole.** `scripts/fs_skip_pages.cpp` cleans each page as SCRIBE `docs` does. It hashes 32-byte windows the way `k5_overlap.cpp` does, against GEN-A's texts and SCRIBE's reference and test texts.
  - Run from 96 MiB, it gives K5_SCALE's counts exactly: 141,633 docs, 4,288 GEN-A hits and 73,802 SCRIBE hits.
  - Run from byte 100,000,000, it flags 74,476 hit pages. 71,676 of them are articles that `pattern_learn` would read, and all of them are skipped. That is about half of enwik9's prose.
- **Plan-time drop.** As before, `--plan-drop 75497472:100000000` removes facts from the prompts' bytes. The default holdout [7340032, 8388608) is dropped too.
- **Nothing from GEN-A test was read.** No dev entity list was used to choose anything.

## Counts

| | enwik8 (verify_f2r_aug) | enwik9 part |
|---|---|---|
| articles read | 8,587 | 77,120 |
| text seeds | 7,484 (+17,907 Wikidata) | 54,629 |
| kept patterns | 70 | 229 |
| prose facts admitted | 1,394 | 10,455 |
| subjects after the drop (merged store) | 3,816 | 25,710 |

The new prose facts are mostly dates: birth_date 6,577, death_date 2,576, birth_year 497. In the merged store, every row of `seeds.tsv` and `facts_f2.tsv` carries a last column, `corpus` (enwik8 or enwik9). Offsets index enwik9, because enwik8 is enwik9's first 1e8 bytes. The frames are enwik8's `patterns.tsv`. The enwik9 patterns are kept beside them as `patterns_add.tsv`.

## Coverage

Coverage is the share of dev prompts that get a plan from the content plan's own entity finder, unchanged. The tool is `scripts/fs_coverage.cpp`. Counting "any fact for a candidate, frames or not" gives the same numbers.

| store | covered | share [Wilson 95%] |
|---|---|---|
| verify_f2r_aug (before) | 43 / 250 | 0.172 [0.130, 0.224] |
| e9_aug (after) | 51 / 250 | 0.204 [0.159, 0.258] |

Why the gain is small:
- The dev prompts come from enwik8 pages, so their entities are enwik8 topics.
- Most enwik9 facts are birth and death dates of other people.
- The guard drops half of the new pages.

**GEN-E's 28.** GEN-E reported 28 of 250 dev prompts planned, using v2_aug. On the current `gen_prompts_dev.tsv` with v2_aug and no drop, I get 45. The subject count matches GEN-E's (3,940), and `content_plan.hpp` has not changed since GEN-E. None of GEN-E's named entities (Agamemnon, Aeneas) occur in my plans. GEN-E probably used a different prompt file. I have not resolved this.

## Precision

**Deviation.** The registered measure is precision against Wikidata (`facts_wd_check`). That needs D4's `wd_values.tsv`, and the file is not on this PC. So the prediction is **not scorable as registered**. The enwik9 run used text seeds only.

**Unregistered, beside it.** Cross-validated precision of the kept patterns is pooled correct over checks on held-out seeds:

| run | kept patterns | checks | precision |
|---|---|---|---|
| enwik8, text seeds only (control) | 44 | 4,912 | 0.971 |
| enwik9 part, text seeds only | 229 | 50,795 | 0.973 |
| enwik8 verify_f2r_aug (with Wikidata) | 70 | 8,628 | 0.958 |

The enwik9 patterns are as precise as the enwik8 control's.

## Time and RAM

These are `/usr/bin/time -v` figures, each run on one core.

| run | time | peak RAM |
|---|---|---|
| `fs_skip_pages` (8 threads) | 1:52 | 1.1 GB |
| `pattern_learn` on enwik9 | 1:01 | 1.04 GB |
| `pattern_learn` on enwik8 control | 0:18 | 112 MB |

## Predictions (§13.38 A)

| prediction | result | score |
|---|---|---|
| before about 30 of 250 (12%) | 43 (17.2% [13.0, 22.4]) | **miss** (more than predicted) |
| after at least 75 of 250 (30%) | 51 (20.4% [15.9, 25.8]) | **miss** |
| prose-fact precision against `facts_wd_check` within 0.05 of enwik8's | no Wikidata file on the PC | **not scorable as registered** (cross-validated, unregistered: 0.973 vs 0.971) |

## For the planner

`--plan bench/fact_store/e9_aug,bench/fact_store/patterns_raw.tsv --plan-drop 75497472:100000000`
