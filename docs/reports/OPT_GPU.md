# O4: the overlap check on the GPU

**Odin Loch** · 2026-11-05 · branch `wip/opt-gpu` · GENERATION_PLAN §13.41 O4

**Outcome first.** The overlap check that cleans enwik9 for K5 (`scripts/k5_overlap.cpp`) now runs on the RTX 3090 with `--gpu`. Cleaning all 429 MB took TBD on the CPU and **44 s** on the GPU, started together: **TBD× faster**. The output is byte-identical: the kept text, the report, stdout, the exit code, and the five training texts made from it (sha256). The prediction (at least 5×) holds. The GPU path is off by default: it needs a build with `-DCYPHA_CUDA=ON`.

## The candidates

Wall times on Odin's PC (Xeon Gold 6242, shared with other sessions). Rows marked "log" are from the runs that made today's files; the others were run for this report on the origin/main binaries (`/mnt/d/cypha_llm/opt/o4/prof`).

| Job | Input | Wall | Threads | The parallel work | Code |
|---|---|---:|---:|---|---|
| K5 drafts for the structure ranker (`k5_structure drafts`, M) | 119,850 boundaries × 8 drafts | 9,140 s (log) | 6 | about 960,000 drafts, each a sequential K5 walk over hash tables | `word_kn.hpp` (M1's) |
| **Overlap check, clean** (`k5_overlap clean`) | 429 MB, six passes | **TBD** (2,830 s on 2026-11-04) | 1 | **about 2.5 billion 32-byte windows, each one independent** | `scripts/k5_overlap.cpp` |
| Overlap check, check | 330 MB, one pass | TBD (276 s on 2026-11-04) | 1 | 339 million windows | the same |
| LDA, 50 MB (`k5_structure discourse`) | 3.9 M tokens, 200 sweeps, 100 topics | 614 s (log) | 2 | 79 billion multiply-adds, but in one sequential chain per thread | `lda.hpp` |
| Structure statistics, 210 MB (`k5_structure stats`) | 40 M tokens | 276 s (log) | 1 | string hashing and hash-map updates | `lm/structure/experts.hpp` (M4's) |
| Structure features | 120,000 boundaries | about 60 s (log) | 8 | per candidate, string work | `lm/structure/experts.hpp` |
| Planner skeleton mining (`k5_planner mine`) | 100 MB | 66 s (30 s of it loading K5) | 1 | per sentence, string matching | `planner/skeleton.hpp` |
| Planner gate training (`k5_planner train`) | 3,000 items | 48 s | 6 | K5 fills | `planner/writer.hpp`, K5 |
| SCRIBE v2 on one dev run (`scribe ladder`) | 250 prompts × 3 | 17 s | 1 | mostly setup (pack, reference) | `eval/scribe.hpp` |
| SCRIBE v3.1 on one dev run | the same | 31 s | 1–2 | the same | the same |

## The choice

The overlap check. Why:
- **The most parallel work.** Every 32-byte window of every document is hashed and looked up on its own: 2.5 billion independent window starts in one clean. Nothing depends on a neighbour.
- **The most wall time** of the jobs I may touch: about 47 minutes for a clean, on one thread.
- **Integers only.** Hashes, positions and counts: the GPU can match the CPU exactly, not just closely.
- **The others do not fit:**
  - K5's drafts are the biggest job, but their code is `word_kn.hpp`, which M1 is rewriting now. Each draft is a sequential walk over hash tables.
  - LDA's collapsed Gibbs sampler is one chain: each token's draw depends on every draw before it. A GPU would have to run many chains at once (as AD-LDA does on 2 threads), and that is a different chain, so the topics would differ. It cannot be exact.
  - The structure statistics live in `lm/structure/` (M4's files). They are string and hash-map work, and the saved file follows `unordered_map` order.
  - The planner's steps and SCRIBE take under a minute each; most of it is loading K5 or SCRIBE's setup.

## What the job does

`k5_overlap clean` removes GEN-A's and SCRIBE's reference and test text from clean enwik9 (K5_SCALE.md). It hashes every 32-byte span of the forbidden texts (64-bit FNV-1a) into a sorted set. Then, for each document, it lists the windows whose hash is in the set. A document whose hits cover more than 10% of it is dropped; otherwise the sentences that touch a hit are cut, until none is left. It makes six passes over the text (each forbidden file, all of them, the cut rounds, two counted-only files, the recheck). On the CPU each window costs a hash and a binary search in a set of 8.6 million hashes: about 25 cache misses.

## The kernel

Files: `native/cuda/k5_overlap_gpu.cu` (434 lines), `native/cuda/k5_overlap_gpu.hpp`; the tool's heavy loop in `scripts/k5_overlap.cpp` behind `--gpu`.

- **One thread per window start.** A block of 256 threads loads its 256 + 31 bytes once into shared memory. Each thread hashes its 32 bytes from there, byte by byte, exactly as the CPU does, and notes any newline.
- **Documents are lines**, so the host packs them with a newline between them. A window that holds a newline crosses a document's end and is not a window of any document. This gives the CPU's windows exactly.
- **Look-up.** The sorted set gets a directory on the hash's top bits (2 million buckets for 8.6 million hashes, about 4 a bucket). A thread reads its bucket's two bounds and binary-searches the few hashes inside: two or three memory reads instead of 25.
- **Answers as bits.** Each warp writes its 32 answers as one 32-bit word (`__ballot_sync`). The host reads the bit map and lists each document's hit positions in order.
- **Chunks on two streams.** Text goes over in 64 MB chunks through two pinned buffers. The host packs one chunk while the device works on the other. On a crowded card the chunk shrinks (a sixteenth of free memory).
- **The forbidden sets.** The same hash kernel writes every window's hash and a flag. CUB compacts the flagged hashes, radix-sorts them and removes repeats. If the card has no room for the sort, the host sorts the same hashes (`std::sort`, `std::unique`). Either way it is the same set.
- The cut rounds are batched: every document cut in a round is checked again in one batch.
- **CPU path.** Without `--gpu` the tool runs the old loop. `--threads N` spreads the documents over N threads (same output).
- **CMake.** `option(CYPHA_CUDA ... OFF)`. With it off, `k5_overlap` builds and runs on the CPU only, so CI needs no CUDA. With it on, nvcc 12.4 for sm_86 (the RTX 3090). `scripts/k5_clean_enwik9.sh` still builds the CPU tool with plain g++.
- **Selftest** (`k5_overlap selftest`, a CTest): the CPU's hits against a byte-by-byte check on random text with planted copies; 4 threads against 1; and, when built with CUDA, the GPU against the CPU with 1 MB chunks (many chunks, a document longer than a chunk), 64 MB chunks, a second set (re-upload), and the host-sort fallback.

## Exactness

The outputs are all integers: hashes, positions, counts and the kept text. There is no floating point, so the rule is plain identity. Each run was compared with `cmp` against the old binary's run beside it, and against the pipeline's own files of 2026-11-04 in `/mnt/d/cypha_llm/k5scale/clean/` (made by the same old binary).

| Output | new CPU, 1 thread | new CPU, 5 threads | GPU |
|---|---|---|---|
| `kept_e9.txt` (329,603,263 bytes, sha256 56ea6001…2753) | TBD | identical | identical |
| `overlap_report.txt` (every count of the clean) | TBD | identical | identical |
| stdout and exit code of `clean` | TBD | identical | identical |
| stdout and exit code of `check` on `train_all.txt` | TBD | identical | identical |

- **Downstream.** `k5_scale_prep.py nest` on the GPU's `kept_e9.txt` gives `train_7M`, `train_30M`, `train_100M`, `train_300M` and `train_all.txt` with exactly the sha256s in the pipeline's `train_sha256.txt` (train_all b7c6ff24…d369).
- The committed GPU binary was run once more (with the selftest's host-sort hook added): the same files again.
- No downstream file can differ: every later step (the structure statistics, LDA, the planner's bank and gate, the models, GEN files and SCRIBE tables) reads `train_all.txt` or its prefixes, and those bytes are the same.

## Times and memory

The final round started old and new together at 20:27 (8 threads in all: old 1, new CPU 1, new CPU 5, GPU 1). WSL crashed at about 20:50 and killed every process on it. By then the GPU and 5-thread runs had finished; the two 1-thread runs had not. They were run again together at 21:00 (TBD).

| Run | `clean` wall | hit passes | span sets | reading files | `check` wall | peak RSS (`clean`) |
|---|---:|---:|---:|---:|---:|---:|
| old (origin/main), 1 thread | TBD | | | | TBD | TBD |
| new CPU path, 1 thread | TBD | TBD | TBD | TBD | TBD | TBD |
| new CPU path, `--threads 5` | 520 s | 471 s | 7.9 s | 31.7 s | 101 s | 1.48 GiB |
| **GPU** (`--gpu`) | **44.1 s** | **2.8 s** | **1.0 s** | 30.5 s | **15.6 s** | 1.40 GiB |
| GPU, the committed binary, beside a fresh old `check` (heavier load) | 88.9 s | 4.5 s | 1.6 s | 56.8 s | 19.8 s | 1.40 GiB |
| old `check` beside it | | | | | 440.8 s | |

TBD speedups.

- **Most of the GPU run is reading the D: drive** (30–57 s for 800 MB through WSL's 9P file system). The work the GPU does (hashing, look-ups, sorting) takes 1.3–1.5 s of kernel time for 2.5 billion window starts; copies to and from the card take 0.7–1.0 s.
- **GPU memory:** at most 496 MiB of buffers for `clean` and 285 MiB for `check` (counted by the tool), plus the CUDA context. The card has 24 GiB and is shared: earlier today another job held 24.2 GB of it. So the chunk shrinks to a sixteenth of the free memory, and the span sort falls back to the host if the card is full.
- **CPU RAM:** the GPU path holds every document's hit list at once (20.5 million positions for SCRIBE's set), two 64 MB pinned buffers and the CUDA runtime: TBD than the old tool's TBD.
- Times vary with the load: the PC ran about 10 agents' jobs (load average 25–50 on 64 threads).

## The O4 prediction, scored

TBD

## Commands

```
cmake -S native -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCYPHA_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc
cmake --build build --target k5_overlap
build/k5_overlap selftest
C=/mnt/d/cypha_llm/k5scale/clean
build/k5_overlap clean $C/clean_e9.txt kept_e9.txt overlap_report.txt $C/gena_texts.txt $C/scribe_reftest.txt \
    --info $C/scribe_pack.txt $C/gena_source.txt --gpu
build/k5_overlap check $C/train_all.txt $C/gena_texts.txt $C/scribe_reftest.txt --gpu
```

Outputs and logs: `/mnt/d/cypha_llm/opt/o4/` on Odin's PC (`prof/` the candidates, `final/` the timed round).
