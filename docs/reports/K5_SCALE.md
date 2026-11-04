# K5 on more Wikipedia: clean enwik9 past enwik8

2026-11-04 · Odin Loch

**Odin:** "Let's just go with K5. Do all of it and make sure you document everything we have tried and the results. Do it all on my PC."

K5 is word-level interpolated Kneser–Ney of order 5 (`native/include/cypha/lm/word_kn.hpp`, GENERATION_PLAN §13.11, §13.19). It was trained on about 7 MB of clean enwik8 prose. This report trains it on more clean Wikipedia, from enwik9 past enwik8's bytes, and reads GEN-A dev SCRIBE v2 at each size.

## Registered before the reads (2026-11-04, 19:40, before any K5 read on enwik9 text)

**The question.** Does more in-domain text raise K5's GEN-A dev SCRIBE?

**The arms.** K5, unchanged, trained on nested texts from enwik9 past byte 100,000,000:

| arm | text |
|---|---|
| E8 (the baseline) | the bake-off's `train.txt`: enwik8 [0, 16) MiB, 7,053,155 bytes |
| N7 | 7,053,155 bytes of the new range (a control: same size, new text) |
| N30 | 30 MB |
| N100 | 100 MB |
| N300 | 300 MB |
| NALL | all of it (about 429 MB before the overlap drop) |

- Each arm: one seed (20261101, the default) on GEN-A dev, 250 prompts.
- The best size and E8: 3 more seeds each (20262101, 20263101, 20264101).
- At N300 and NALL: also a bigger vocabulary (`--k5-vmax 65536`, the most 16-bit ids allow).
- Reported for every read: SCRIBE v2 [95%], copy32, the samples with a 13-token training run (must stay 0), build time, peak RAM, vocabulary.

**The rule.**
- The best size (by its first-seed dev SCRIBE) becomes K5's recommended training text if its dev SCRIBE beats E8's 0.353 by at least 0.02, that is, if it reaches 0.373 or more.
- The seed spread of both (4 seeds each) is reported beside it.
- If the rule holds, one GEN-A test read of that size is made (`bench/eval/gen_prompts_test.tsv`), reported against E8's test 0.305 [0.276, 0.338]. The test split is not read otherwise.

**The prediction.**
- P1. N7 scores about E8's level: 0.33–0.36. The new range is the same kind of text.
- P2. SCRIBE rises with size, but slowly: N30 about 0.36, N100 about 0.37, N300 and NALL about 0.37–0.39.
- P3. The rule holds, narrowly (best about 0.38). Probability about 55%.
- P4. copy32 rises with size (more text to match): 0.035 at E8, about 0.05–0.08 at NALL. The 13-token count stays 0 (the cap).
- P5. The bigger vocabulary changes SCRIBE by less than 0.01.
- P6. NALL builds in under 5 minutes and peaks under 12 GB.

## Setup on Odin's PC

- Branch `wip/k5-scale` from `origin/main` (c7dd132), worktree `/home/odin/k5scale_wt`.
- Build: `cmake -S native -B ~/cypha_build/k5scale -DCMAKE_BUILD_TYPE=Release`, then `make -j8 cyphalm_trie scribe` (GCC 13.3). `cyphalm_trie selftest` passes.
- Data: enwik8 and enwik9 at `/mnt/d/cypha_llm/`; SCRIBE's vectors `/mnt/d/cypha_llm/scribe/v32`; the bake-off's `train.txt` at `/mnt/d/cypha_llm/fuzzy/train.txt` (FZ_WORK of `fuzzy_bake.py`).
- enwik9 was already on the PC (1,000,000,000 bytes). It was not downloaded again.
- Two flags were added to `cyphalm_trie sample` (a small separate commit): `--k5-seed S` (default 20261101) and `--k5-vmax V` (default 32,000, at most 65,536). The defaults give byte-identical samples.

## Reproducing 0.353 on this PC

```
cyphalm_trie sample /mnt/d/cypha_llm/enwik8 E8.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv --prompts 250 \
    --k5 /mnt/d/cypha_llm/fuzzy/train.txt
scribe ladder --pack enwik:E8:16:48 --ref enwik:E8:48:56 --test enwik:E8:56:64 --vectors v32 --profile v2 \
    --seed 20261027 --gen-trie E8.gen.txt --label E8
```

| | SCRIBE v2 [95%] | copy32 | 13-token runs | vocabulary | build | peak RAM |
|---|---|---|---|---|---|---|
| §13.19 | 0.353 [0.318, 0.384] | 0.035 | 0 | 32,000 | 1 s | 220 MB |
| this PC | **0.353 [0.318, 0.384]** | 0.0351 | 0 of 250 | 32,000 | 1.1–1.7 s | 218 MB |

The same number to the third digit. With `--k5-seed 20261101` given, the samples are byte-identical to the run without it.


## The new text

**Command** (`scripts/k5_clean_enwik9.sh`; it calls `scripts/k5_overlap.cpp` and `scripts/k5_scale_prep.py`):

```
scripts/k5_clean_enwik9.sh /mnt/d/cypha_llm/enwik9 /mnt/d/cypha_llm/enwik8 ~/cypha_build/k5scale/scribe \
    /mnt/d/cypha_llm/k5scale/clean ~/nn_venv/bin/python
```

1. `scribe docs enwik:enwik9:96:954`. 96 MiB is 100,663,296 bytes, the first whole MiB past enwik8's 100,000,000. The cleaner keeps whole pages only. Result: 141,633 documents, 429,182,099 bytes.
2. The overlap check (below). Result: 100,795 documents, 329,603,263 bytes.
3. A shuffle of the documents (Python `random.Random(20261104)`), then nested prefixes. Each text is the start of the next.

| text | documents | bytes | tokens | SHA-256 (first 12) |
|---|---|---|---|---|
| `train_7M.txt` | 2,239 | 7,051,586 | 1,363,427 | c30f99117f67 |
| `train_30M.txt` | 9,022 | 29,999,652 | 5,788,534 | 18aa8157ac05 |
| `train_100M.txt` | 30,358 | 99,994,296 | 19,314,934 | 21d33851421d |
| `train_300M.txt` | 91,867 | 299,999,917 | 57,915,941 | b1420169efe7 |
| `train_all.txt` | 100,795 | 329,603,262 | 63,623,367 | b7c6ff24f5cb |
| E8: the bake-off's `train.txt` | 1,297 | 7,053,155 | 1,356,468 | |

The texts are on Odin's PC in `/mnt/d/cypha_llm/k5scale/clean/`. They are not in the repo (767 MB).

## The overlap check

**Forbidden** (removed):
- GEN-A's prompts and continuations, dev and test (1,000 texts; 315,799 distinct 32-byte spans). Each is used as written and with white space collapsed.
- SCRIBE's reference and test texts: enwik8 [48, 64) MiB, cleaned by `scribe docs` (1,355 documents; 8,325,008 spans).

**The method.**
- Every 32-byte span is hashed (64-bit FNV-1a). A hit is a 32-byte window of a document whose hash is a forbidden span's.
- If hits cover more than 10% of a document's bytes, the document is dropped. That is GEN_TRACK's 10% rule for near-copies.
- In any other document with a hit, every sentence that overlaps a hit is cut. The document is checked again until it holds none (at most 2 rounds were needed).
- Then a second, separate pass reads the written `train_all.txt` back and counts the documents that hold a forbidden span.

**Results.**

| count | value |
|---|---|
| documents in | 141,633 (429.2 MB) |
| documents with a GEN-A span | 4,288 (28,472 windows) |
| documents with a SCRIBE reference/test span | 73,802 (20,529,256 windows) |
| documents dropped (hits over 10%) | 40,832 (86.6 MB) |
| documents with sentences cut | 33,621 (78,332 sentences, 12.9 MB) |
| documents emptied by the cuts | 6 |
| documents out | 100,795 (329.6 MB) |
| **second pass: documents of `train_all.txt` with a forbidden span** | **0** |

**What was dropped.**
- Most are US census stubs ("As of the census of 2000, there were …"). Their template shares long spans with the census stubs in SCRIBE's reference and test text. 23,318 documents of the new range hold that sentence before the check; 1 does after. Of the 46,134 documents that were dropped or lost their opening sentence, 36,167 mention the census in their first 2,000 characters.
- Most of the rest are near-copies. enwik9 holds some pages twice (for example "Politics of Jamaica" and Gutenberg's page each appear twice; one copy was dropped).

**Counted only** (not removed), on the kept documents:

| text | spans | kept documents holding one | windows |
|---|---|---|---|
| SCRIBE's pack, enwik8 [16, 48) MiB | 17,313,084 | 36,085 | 595,653 |
| GEN-A's whole source region, enwik8 [72 MiB, end) | 12,958,741 | 31,820 | 511,882 |

These were not inspected; they are most likely shared stock phrases. No kept document holds a span of a GEN-A prompt or continuation, or of SCRIBE's reference or test text.

**A change made before the reads.** The first version of the check dropped any document with one hit, as `fuzzy_bake.py` prep does. That would have dropped 52% of the documents, mostly for a single shared phrase. It was replaced by the 10% rule and the sentence cut above. No K5 read on the new text had been made. The numpy version was also too slow on the loaded PC (71 s per 64 MB), so the check is C++.

## The reads

Every read: `scripts/k5_scale_read.sh LABEL TEXT SEED dev [--k5-vmax 65536]`. It runs `cyphalm_trie sample … --prompts 250 --k5 TEXT --k5-seed SEED` under `/usr/bin/time -v`, then SCRIBE v2's ladder (the command above). SCRIBE is the "trie samples T=1.0" row. Seed 20261101 is K5's default.

### By size (GEN-A dev, one seed, vocabulary 32,000)

| arm | text | SCRIBE v2 [95%] | surface S | copy32 | 13-token runs | unknown tokens | KN entries o5 | model | build | peak RAM | wall (build + 250 samples) | cont bits per byte |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E8 | enwik8 7.05 MB | 0.353 [0.318, 0.384] | 0.928 | 0.035 | 0 of 250 | 3.13% | 1,319,212 | 119 MB | 1.1 s | 0.22 GB | 18 s | 1.5500 |
| N7 | 7.05 MB | 0.350 [0.318, 0.382] | 0.923 | 0.030 | 0 of 250 | 3.38% | 1,332,147 | 121 MB | 1.4 s | 0.22 GB | 18 s | 1.5474 |
| N30 | 30 MB | 0.375 [0.346, 0.403] | 0.928 | 0.056 | 0 of 250 | 4.65% | 5,529,307 | 446 MB | 6.3 s | 0.79 GB | 28 s | 1.4815 |
| N100 | 100 MB | 0.363 [0.336, 0.392] | 0.925 | 0.110 | 0 of 250 | 5.19% | 17,847,745 | 1.32 GB | 19.4 s | 2.29 GB | 59 s | 1.4261 |
| N300 | 300 MB | 0.382 [0.353, 0.408] | 0.923 | 0.163 | 0 of 250 | 5.38% | 51,190,920 | 3.53 GB | 62.5 s | 6.20 GB | 2 min 36 s | 1.3767 |
| **NALL** | **329.6 MB** | **0.412 [0.387, 0.439]** | 0.928 | 0.174 | 0 of 250 | 5.39% | 55,988,675 | 3.84 GB | 64.6 s | 6.77 GB | 2 min 42 s | **1.3737** |

- "cont bits per byte": K5's bits per byte on the GEN-A dev continuations (an unknown word costs `<unk>`'s bits).
- copy32 is measured against each arm's own training text. It grows with the text, so it does not compare across sizes. The 12-word cap held at every size (0 samples with a 13-token run).
- Build time is K5's build inside `sample` (tokenising, counting, the cap's windows). The PC was shared (load 10–30) throughout.

### Seeds: the best size (NALL) and E8 (GEN-A dev, vocabulary 32,000)

| seed | E8 | NALL |
|---|---|---|
| 20261101 | 0.353 [0.318, 0.384] | 0.412 [0.387, 0.439] |
| 20262101 | 0.362 [0.328, 0.393] | 0.389 [0.362, 0.416] |
| 20263101 | 0.337 [0.304, 0.365] | 0.378 [0.353, 0.405] |
| 20264101 | 0.323 [0.293, 0.353] | 0.381 [0.352, 0.409] |
| **mean (range)** | **0.344 (0.323–0.362)** | **0.390 (0.378–0.412)** |
| copy32 (range) | 0.032–0.036 | 0.174–0.182 |
| 13-token runs | 0 of 1,000 | 0 of 1,000 |

- NALL's lowest seed (0.378) is above E8's highest (0.362).
- The mean gain is +0.046. The first seed's gain (+0.059) is the luckiest of the four.

### A bigger vocabulary (GEN-A dev, seed 20261101)

| arm | vocabulary | unknown tokens | SCRIBE v2 [95%] | surface S | copy32 | 13-token runs | build | peak RAM |
|---|---|---|---|---|---|---|---|---|
| N300 | 32,000 | 5.38% | 0.382 [0.353, 0.408] | 0.923 | 0.163 | 0 | 62.5 s | 6.20 GB |
| N300 | 65,536 | 2.92% | 0.366 [0.338, 0.394] | 0.919 | 0.147 | 0 | 66.5 s | 6.41 GB |
| NALL | 32,000 | 5.39% | 0.412 [0.387, 0.439] | 0.928 | 0.174 | 0 | 64.6 s | 6.77 GB |
| NALL | 65,536 | 2.93% | 0.402 [0.375, 0.430] | 0.920 | 0.155 | 0 | 71.6 s | 6.99 GB |

The bigger vocabulary halves the unknown tokens but lowers SCRIBE (−0.016 and −0.010, one seed each). 65,536 is the most the 16-bit ids allow; going further needs wider context keys in `word_kn.hpp`, which was not done. K5 stays at 32,000.

## The rule

- The best size by its first-seed dev SCRIBE is **NALL: 0.412**. The bar is 0.353 + 0.02 = 0.373.
- Over 4 seeds NALL scores 0.378–0.412 (mean 0.390). E8 scores 0.323–0.362 (mean 0.344).
- **The rule holds.** `train_all.txt` (329.6 MB of clean enwik9 prose) becomes K5's recommended training text, with the vocabulary at 32,000.
- N300 (0.382, one seed) is within NALL's seed spread. The two differ by 10% of text. The data cannot tell them apart.

## The one GEN-A test read (made once, after the rule held)

`scripts/k5_scale_read.sh TEST_NALL_s20261101 train_all.txt 20261101 test`, that is, `--prompts-file bench/eval/gen_prompts_test.tsv`.

| | SCRIBE v2 [95%] | surface S | copy32 | 13-token runs | cont bits per byte |
|---|---|---|---|---|---|
| E8 (§13.19's test read) | 0.305 [0.276, 0.338] | | | | |
| **NALL** | **0.380 [0.353, 0.406]** | 0.927 | 0.169 | 0 of 250 | 1.3686 |

The test gain is +0.075, and the intervals do not overlap. No other test read was made.

## The predictions, scored

| | prediction | result | |
|---|---|---|---|
| P1 | N7 at E8's level, 0.33–0.36 | 0.350 | hit |
| P2 | rises slowly: N30 ~0.36, N100 ~0.37, N300/NALL 0.37–0.39 | 0.375, 0.363, 0.382, 0.412 (NALL mean 0.390) | partly: N30 higher, N100 lower, not monotone |
| P3 | the rule holds, narrowly (best ~0.38) | holds; best 0.412, mean 0.390 | hit (less narrow) |
| P4 | copy32 0.05–0.08 at NALL; 13-token runs 0 | copy32 0.17; runs 0 | miss on copy32; hit on runs |
| P5 | the bigger vocabulary moves SCRIBE by under 0.01 | −0.016 and −0.010 | miss (it lowers it) |
| P6 | NALL builds in under 5 min, peaks under 12 GB | 65 s; 6.8 GB | hit |

## What this means

- More in-domain text helps K5. The same-size control (N7, 0.350) matches E8 (0.353), so the new range is not easier text. The gain comes from size.
- Most of the gain arrives by 30 MB (0.375). Past that the single-seed reads (0.363, 0.382, 0.412) wander inside about ±0.02 of seed noise.
- Bits keep falling with size (1.550 → 1.374 per byte), and SCRIBE rises less than bits fall.
- **copy32 rises to 0.17.** That is 17% of written bytes lying in some 32-byte span of the 330 MB training text. The 12-word cap held (0 runs over 12 words in 2,250 samples), but the copy share is a cost of the bigger text. Watch it if copying matters beyond the cap.
- Cost: NALL needs about 7 GB of RAM and a 65 s build each time `sample --k5` starts. 250 samples take about 1.5 min on 4 threads.
- Not done: texts beyond enwik9 (the rest of a Wikipedia dump), a vocabulary above 65,536, and more seeds for N300.

## Files

- In the repo: `bench/k5_scale/results_pc_2026-11-04/` holds `READS.txt` (one line per read), and for every read the samples (`*.gen.txt`), K5's log with `/usr/bin/time -v` (`*.sample.log`) and SCRIBE's report (`*.scribe.md`). `clean/` holds the overlap report, the second pass and the texts' SHA-256.
- Scripts: `scripts/k5_clean_enwik9.sh` (the cleaning and the nested sizes), `scripts/k5_overlap.cpp` (the check), `scripts/k5_scale_prep.py` (text plumbing), `scripts/k5_scale_read.sh` (one read).
- On Odin's PC: the texts in `/mnt/d/cypha_llm/k5scale/clean/` (`train_7M.txt` … `train_all.txt`, plus the cleaned dump `clean_e9.txt` and the kept documents `kept_e9.txt`); the reads in `/mnt/d/cypha_llm/k5scale/reads/`; the build in `~/cypha_build/k5scale/`; the worktree `/home/odin/k5scale_wt`.

**To use it:** `cyphalm_trie sample ENWIK8 OUT.gen.txt --prompts-file … --prompts 250 --k5 /mnt/d/cypha_llm/k5scale/clean/train_all.txt`.

## The upgraded K5 at scale (registered 2026-11-05, before the reads)

**The question.** Does the upgraded K5 (the default since K5_UPGRADES.md: nucleus 0.8, 256 exchange classes as the backoff floor, the 8-candidate sentence gate by prompt association) gain from `train_all.txt` as the old K5 did?

**Setup.** `origin/main` 6aed3eb, rebuilt (`cyphalm_trie selftest` ok; `scribe selftest` 90 checks, 0 failed). `K5_THREADS=4`. `scripts/k5_scale_read.sh` now also runs `scribe ladder --profile v3` and reports v2, SURPRISE, MATCH, v3, min3 and the disagree share for the K5 rung. The decision stays on v2.

**Reproduction checks (made before registering, on 6aed3eb).**

| check | expected | got |
|---|---|---|
| `--k5-base k5` on `train_all.txt`, seed 20261101 | NALL 0.412, the same samples | 0.412 [0.387, 0.439]; samples byte-identical to NALL's (MODEL lines aside) |
| the new default on the 7 MB enwik8 `train.txt` (U8), seed 20261101 | 0.498 [0.469, 0.528] (the cloud) | 0.498 [0.469, 0.528] |

`--k5-base k5` on `train_all.txt` now peaks at 8.45 GB (6.77 GB before; the context keys are 128 bits since K5_UPGRADES).

**The arms.** Both on GEN-A dev, 250 prompts, seeds 20261101, 20262101, 20263101 and 20264101:
- **U8**: the new default K5 on the 7 MB enwik8 `train.txt` (the baseline).
- **C1**: the new default K5 on `train_all.txt` (329.6 MB).

**The rule.**
- C1 is adopted as the recommended writer if its 4-seed mean SCRIBE v2 is at least 0.02 above U8's 4-seed mean, with no sample over the 12-word cap.
- copy32, build time and peak RAM are reported. If the class build or the gate's co-occurrence counts are too slow or too big at 330 MB, that is reported with what was measured.
- If the rule holds, one GEN-A test read of C1 (seed 20261101), against U8's test 0.524 [0.493, 0.552]. The test split is not read otherwise.

**The prediction.**
- Q1. U8's 4 seeds: mean about 0.50 (K5_UPGRADES: 0.493–0.505).
- Q2. C1's mean is about 0.03 above U8's (range 0.00 to +0.06). The old K5 gained +0.046; the gate and nucleus 0.8 already take part of what more text gives. The rule holds with probability about 55%.
- Q3. C1's copy32 is 0.18–0.25 (higher than NALL's 0.17: nucleus 0.8 draws more from the top). 0 samples over the cap.
- Q4. C1's build takes 3–10 minutes (the exchange classes over 12× the bigrams) and peaks at 10–14 GB.
- Q5. SCRIBE v3 gains less than v2 between U8 and C1 (under +0.02), and the new judges disagree with v2 about the size of the gain.
