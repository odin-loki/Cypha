# 1 TB runs: verify the last model, train another, compare

**Odin Loch** · 2026-10-29 · PLAN Track A

This is the runbook for the trie runs on the 1 TB corpus. The live run uses `native/tools/trie_stream_train.cpp` and `scripts/run_train.sh`. Neither file is changed by anything here.

## What is new

| File | What it does |
|---|---|
| `native/include/cypha/cyphalm/stream_trie.hpp` | The live run's model, plus one fix: a 32-bit count about to overflow halves its table instead of wrapping |
| `native/tools/trie_stream_train_v2.cpp` | The second trainer: same corpus files, same model, same checkpoint after each file. Output goes to `<corpus>/trie/v2/` |
| `native/tools/trie_ckpt.cpp` | `verify`, `bpc` and `compare` for both runs' checkpoints; `selftest` (CTest `native_trie_ckpt_selftest`) |
| `scripts/run_train_v2.sh` | Builds and starts the second run on your PC's paths |
| `scripts/verify_compare_1tb.sh` | Verifies both checkpoints and compares them on held-out text |

## A problem to expect in the last model

The live trainer keeps its counts in 32 bits: the unigram, the order-1 table, each context's total and each follower's count. A 32-bit count wraps past 4.29 billion. A space is about one byte in six, so its count wraps after roughly 26 GB of text. Frequent contexts wrap soon after. At 1 TB many counts have wrapped, and their probabilities are wrong.

`trie_ckpt verify` finds this. The trainer also keeps `bytes_seen` in 64 bits, which never wraps. The verifier compares each table's total with it. A short total means a wrap. Expect the last model to show `NOT VERIFIED` with lines like "unigram counts wrapped" and "order-2 totals fall short of the bytes seen". That is the reason for the second run.

The second run halves a table's counts when one would pass 2^32 − 1. The ratios stay right. `verify` reports how many halvings happened.

## Steps at home

1. **Check the live run is finished.** It never exits by itself. It is finished when its log repeats `waiting for a finished file` and no `*.part` file is left in the corpus. Stop it with Ctrl-C. Its checkpoint is safe: each one is written to a temporary file and then renamed.

2. **Verify the last model.**
   ```bash
   CYPHA_ENWIK8=/path/to/enwik8 bash scripts/verify_compare_1tb.sh
   ```
   - The script builds `trie_ckpt`, runs its self-test, then verifies `<corpus>/trie/checkpoint.bin`.
   - It checks the file's size and structure, the count wraps, and that every finished corpus file is in `trained.txt`.
   - `CYPHA_DEEP=1` also re-reads the whole corpus to check the bytes seen. That takes hours.
   - Reports go to `<corpus>/trie/compare_<date>/`.

3. **Start the second run** (after the live run has stopped, so the two do not share RAM):
   ```bash
   bash scripts/run_train_v2.sh
   ```
   - It trains every finished corpus file in the same order as the live run, then stops and writes `done.txt`.
   - Snapshots for the scaling curve (Track A, A1) are kept at 8M, 64M, 1G, 10G and 100G. Each is a full copy of the checkpoint at that point, so the later ones are large. Set `CYPHA_SNAPSHOTS=8M,64M,1G` to keep fewer.
   - It resumes from its own checkpoint if stopped.
   - Settings: `CYPHA_REPO`, `CYPHA_CORPUS`, `CYPHA_CORE`.

4. **Compare the two models.** Run step 2's command again. With `<corpus>/trie/v2/checkpoint.bin` present it verifies both, then writes `compare.txt`. Each held-out file gets one row:
   - A (last run) and B (second run) bits per byte;
   - B − A with a 95% interval over 4 KiB blocks;
   - today's fixed backoff in two forms, full counts and the packed serving form (`synth`).

## Held-out text

- The comparison is fair only on text neither run trained on.
- If enwik8, enwik9 or the Wikipedia dump it came from is in the corpus, enwik8's holdout is text the models have seen. Its bits per byte then measure memory, not prediction.
- Track A's step 1 asks for a frozen slice per category, set aside before training. Put such files in `/mnt/d/cypha_llm/holdout/` (outside the corpus). The script reads every `*.txt` there.
- Or name files directly: `CYPHA_HOLDOUTS="file1.txt file2.txt:0:1048576"`. A spec is `FILE` or `FILE:FROM:LEN`.

## The scaling curve

```bash
/tmp/trie_ckpt bpc <corpus>/trie/v2/checkpoint_8M.bin  <holdouts>
/tmp/trie_ckpt bpc <corpus>/trie/v2/checkpoint_1G.bin  <holdouts>
/tmp/trie_ckpt bpc <corpus>/trie/v2/checkpoint.bin     <holdouts>
```

These give Track A's A1 points: bits per byte against bytes seen. Prediction P-A1 (trie-only BPC at 1 TB on the enwik 8 MiB holdout, 2.00–2.25) can then be scored.

## Memory

- `verify` loads one checkpoint whole, about its file size in RAM.
- `bpc` and `compare` load one model at a time, also about its file size.
- The trainers need about the same RAM as the live run.

## Checks made before this was published (2026-10-29)

- **Self-test:** `trie_ckpt selftest` passes 5,740 checks.
  - Scoring a checkpoint gives the same probabilities as `packed_trie`'s own model trained on the same bytes (largest difference 0).
  - TRI1 and TRI2 files round-trip.
  - With a small cap, no count passes it, and totals stay their followers' sums.
  - A wrapped unigram, a damaged total, a wrong child count, an untrained corpus file and a `*.part` file are each reported.
- **End to end on 24 MiB of enwik8** (three 8 MiB files):
  - The live trainer (built unchanged) and the second trainer built identical models: 754,273 nodes, 2,307,921 followers, 600,243 match keys.
  - Both checkpoints `VERIFIED`, with `--deep` showing 25,165,824 bytes seen.
  - Compared on enwik8's 1 MiB holdout: 2.301584 against 2.301584 bits per byte, difference +0.0000.
  - The 8 MiB snapshot against the final model: −0.0376 [−0.0454, −0.0297].

## Not done yet

- Today's full CyphaLM stack (class, similar-context, copy, byte Kneser–Ney, the document cache) does not run on the 1 TB trie yet. `cyphalm_trie` trains its own trie from a slice. A loader for a 1 TB checkpoint is the next step.
- Byte Kneser–Ney at 1 TB needs its own build at that scale (BYTE_KN.md).
