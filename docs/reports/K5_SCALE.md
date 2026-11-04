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

(Results follow below.)
