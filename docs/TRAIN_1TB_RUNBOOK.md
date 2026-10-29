# 1 TB runs: train model N, verify it, compare it

**Odin Loch** · 2026-10-29 · PLAN Track A

Models on the 1 TB corpus are numbered.

- **Model 1** is the live run: `<corpus>/trie/checkpoint.bin`, trained by `native/tools/trie_stream_train.cpp` through `scripts/run_train.sh`. Neither file is changed.
- **Model N** (2 or more) goes to `<corpus>/trie/model<N>/`.

## One script

```bash
bash scripts/train_model.sh          # model 2, compared with model 1
bash scripts/train_model.sh 3        # model 3, compared with model 2
```

It does, in order:

1. Builds the trainer and the checker, and runs the checker's self-test.
2. Verifies model N−1: the file, its structure, whether 32-bit counts wrapped, and that every finished corpus file was trained.
3. Trains model N on the same corpus files. It resumes if stopped. It ends when every finished file is trained, and writes `done.txt`.
4. Verifies model N.
5. Compares model N−1 (A) with model N (B) on held-out text.

Reports go to `<corpus>/trie/model<N>/report/`:
- `verify_model<N-1>.txt` and `verify_model<N>.txt`;
- `compare_model<N-1>_model<N>.txt`;
- `train.log`.

Training 1 TB takes a long time, so run the script where it can keep going, for example in `tmux` or with `nohup bash scripts/train_model.sh > model2.log 2>&1 &`. Start it after model 1's run has stopped, so the two do not share RAM.

Settings, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `CYPHA_REPO` | `/mnt/c/Users/odinl/OneDrive/Desktop/Cypha` | the repository |
| `CYPHA_CORPUS` | `/mnt/d/cypha_llm/corpus` | the corpus |
| `CYPHA_CORE` | `0` | the CPU core |
| `CYPHA_SNAPSHOTS` | `8M,64M,1G,10G,100G` | sizes at which a copy of the checkpoint is kept (the scaling curve); each is a full copy, so the later ones are large |
| `CYPHA_HOLDOUTS` | every `*.txt` in `<corpus>/../holdout` | held-out files, `FILE` or `FILE:FROM:LEN`, space-separated |
| `CYPHA_ENWIK8` | unset | the enwik8 file; adds its 8 MiB-protocol holdout |
| `CYPHA_DEEP` | `0` | `1` re-reads the corpus to check the bytes seen (hours) |

## Before you start: is model 1 finished?

The live trainer never exits by itself. It is finished when its log repeats `waiting for a finished file` and no `*.part` file is left in the corpus. Stop it with Ctrl-C. Its checkpoint is safe: each one is written to a temporary file and then renamed.

## A problem to expect in model 1

The live trainer keeps its counts in 32 bits: the unigram, the order-1 table, each context's total and each follower's count. A 32-bit count wraps past 4.29 billion. A space is about one byte in six, so its count wraps after roughly 26 GB of text. Frequent contexts wrap soon after. At 1 TB many counts have wrapped, and their probabilities are wrong.

The checker finds this. The trainer also keeps `bytes_seen` in 64 bits, which never wraps. The checker compares each table's total with it, and a short total means a wrap. Expect model 1 to show `NOT VERIFIED` with lines like "unigram counts wrapped" and "order-2 totals fall short of the bytes seen".

Model 2 and later use the same model, but a count about to pass 2^32 − 1 halves its table instead of wrapping. The ratios stay right. The checker reports how many halvings happened.

## Held-out text

- The comparison is fair only on text neither model trained on.
- If enwik8, enwik9 or the Wikipedia dump it came from is in the corpus, enwik8's holdout is text the models have seen. Its bits per byte then measure memory, not prediction.
- Track A's step 1 asks for a frozen slice per category, set aside before training. Put such files in `/mnt/d/cypha_llm/holdout/`, outside the corpus. The script reads every `*.txt` there.

Each compare row gives, for one held-out file:
- A's and B's bits per byte, with today's fixed backoff in two forms: full counts, and the packed serving form (`synth`);
- B − A with a 95% interval over 4 KiB blocks. Negative means model N predicts better.

## The scaling curve

```bash
/tmp/cypha_trie_ckpt bpc <corpus>/trie/model2/checkpoint_8M.bin <holdouts>
/tmp/cypha_trie_ckpt bpc <corpus>/trie/model2/checkpoint_1G.bin <holdouts>
/tmp/cypha_trie_ckpt bpc <corpus>/trie/model2/checkpoint.bin    <holdouts>
```

These give Track A's A1 points, bits per byte against bytes seen. Prediction P-A1 (trie only, 2.00–2.25 at 1 TB on the enwik 8 MiB holdout) can then be scored.

## The pieces

| File | What it does |
|---|---|
| `scripts/train_model.sh` | The one script above |
| `native/include/cypha/cyphalm/stream_trie.hpp` | Model 1's model, with counts that halve instead of wrapping; reads both checkpoint formats ("TRI1" for model 1, "TRI2" for later models) |
| `native/tools/trie_stream_train_v2.cpp` | The trainer for model 2 and later |
| `native/tools/trie_ckpt.cpp` | `verify`, `bpc`, `compare`, `selftest` (CTest `native_trie_ckpt_selftest`) |

Memory: the checker loads one model at a time, about its file size in RAM. The trainer needs about the same RAM as model 1's run.

## Checks made before this was published (2026-10-29)

- **Self-test:** `trie_ckpt selftest` passes 5,740 checks.
  - Scoring a checkpoint gives the same probabilities as `packed_trie`'s own model trained on the same bytes (largest difference 0).
  - Both checkpoint formats round-trip.
  - With a small cap, no count passes it, and totals stay their followers' sums.
  - A wrapped unigram, a damaged total, a wrong child count, an untrained corpus file and a `*.part` file are each reported.
- **The script end to end on 24 MiB of enwik8** (three 8 MiB files):
  - Model 1 was built with the live trainer, unchanged. `train_model.sh` verified it, trained model 2, verified model 2 and compared them.
  - The two models are identical: 754,273 nodes, 2,307,921 followers, 600,243 match keys.
  - Both `VERIFIED`. On enwik8's 1 MiB holdout both score 2.301584 bits per byte, difference +0.0000.
  - `train_model.sh 3` verified model 2, trained model 3 and compared the two.
  - The 8 MiB snapshot against the final model: −0.0376 [−0.0454, −0.0297].

## Not done yet

- Today's full CyphaLM stack (class, similar-context, copy, byte Kneser–Ney, the document cache) does not run on a 1 TB trie yet. `cyphalm_trie` trains its own trie from a slice. A loader for a 1 TB checkpoint is the next step.
- Byte Kneser–Ney at 1 TB needs its own build at that scale (BYTE_KN.md).
