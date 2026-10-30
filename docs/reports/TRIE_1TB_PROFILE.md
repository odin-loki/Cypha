# Byte trie trained on the 1 TB corpus: profile, packed model, tests

**Date:** 2026-10-28 to 2026-10-31
**Trainer:** `native/tools/trie_stream_train.cpp` (8 cores, one pass, checkpoint `TR64`, 64-bit counts)
**Profiler:** `native/tools/trie_stream_profile.cpp`
**Checker and packer:** `native/tools/trie_ckpt.cpp` (`verify`, `bpc`, `pack`) over `cypha/cyphalm/stream_trie.hpp`
**Checksums:** `scripts/verify_corpus_sha.py`
**Packed model in git:** [`models/cyphalm_1tb/`](../../models/cyphalm_1tb/README.md) (`trie_1tb.pk.zst`, Git LFS)
**Raw output:** `D:\cypha_llm\profile\` (profile: `enwik9.txt`, `c4_heldout.out`, `lcet10.out`; packing: `pack.txt`, `verify_tr64.txt`, `bpc_pk.txt`, `bpc_ckpt.txt`; tests: `ctx8k_*.out`, `gen/*.summary.md`)

Short version:

- **Training.** One pass over 918.6 GB of text took about 48 hours. The unpacked model is 12.4 GB.
- **Packing.** The packed model is **992 MB**, or 465 MB compressed. It scores the same as the unpacked model.
- **What does the work.** Order-4 backoff carries the model. The 4-byte match table is worthless at this scale and is not in the packed model.
- **Held-out scores.** **2.17 BPC** on an unseen C4 shard. **3.21 BPC** on the enwik9 holdout, worse than the 8 MiB trie (2.30), because no Wikipedia XML was trained.
- **Against the 8 MiB trie on the same slices.** Off Wikipedia the terabyte is worth a lot: −0.45 BPC on unseen C4 (2.180 against 2.627) and −0.66 on lcet10 (2.328 against 2.984). On Wikipedia XML the 8 MiB wiki trie still wins: 2.298 against 3.207 on the holdout, and 2.440 against 2.880 200 MB later.
- **Model 2.** The runbook's second model is training on 32 threads (`trie_stream_train_v2 --threads 32`), with snapshots for the scaling curve.
- **InfiniGram.** With an 8K InfiniGram over the wiki training slice and Dolly, the 1 TB trie scores 2.276, against 2.068 for the 8 MiB trie.
- **Questions.** Neither model answers an unseen question: 0 of 41.
- **Generation.** SCRIBE v2 rises from 0.020 to 0.028, but the 95% intervals overlap.

## What was trained

Every file below was checked against a published digest before training.

| Category | Files | On disk | Source |
|----------|------:|--------:|--------|
| Articles | 739 | 625.13 GB | RedPajama C4, shards `00000`–`00738` |
| Code | 96 | 224.05 GB | RedPajama GitHub (96 of 98 files) |
| Papers | 100 | 93.79 GB | RedPajama arXiv (all) |
| Structured | 1 | 80.05 GB | RedPajama Stack Exchange |
| Books | 40 | 0.03 GB | Gutenberg `pg1`–`pg39`, `pg1342` |
| **Total** | **976** | **1,023 GB** | **918.6 GB of text after the JSON wrapper is removed** |

Not trained:

- **Wikipedia, Wikisource, and Wikibooks.** They are bz2 XML, and the trainer reads only `.txt` and `.jsonl`. English Wikipedia is also still a 5.9 GB `.part`.
- **RedPajama Common Crawl and the CC WET sample.** The 580 GB articles cap was filled by C4 first.
- **Most of Gutenberg.** The fetch stopped at book 40 when gutenberg.org started answering HTTP 406 to plain requests. `fetch_corpus.py` now asks for the gzip variant.
- **Labelled chat.** Kept out by design (see the plan in [`CURRENT.md`](../CURRENT.md)).
- **Non-ASCII characters inside JSON text.** The trainer's JSON reader drops `\uXXXX` escapes, so accented letters and CJK in C4, GitHub, and arXiv are missing.

Twelve checksum lists under `corpus/sha256/` were also read as text. That is about 0.5 MB of hex digests and paths, roughly 0.00006% of the text. The trainer and `stream_trie.hpp`'s `trainable()` now skip that folder.

## Checksums

RedPajama publishes SHA-256 lists per source (`https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/<source>_SHA256SUMS.txt`). Wikimedia publishes SHA-1 lists per dump.

| Result | Files |
|--------|------:|
| SHA-256 matched on the first pass | 918 |
| SHA-256 failed: download was short | 18 |
| Short files re-fetched and matched | 18 |
| Wikisource and Wikibooks `20261001` dumps, SHA-1 | 2 match |
| No published sum (Gutenberg, labelled, `wet.paths.gz`, catalog) | 50 |

All 18 failures were truncated downloads. In each one the connection closed early, and the old `fetch_corpus.py` renamed the `.part` as finished. Stack Exchange had stopped at 46.2 GB of 80.0 GB. The arXiv, C4, and GitHub failures were missing between 8% and 92% of their bytes. `download()` now compares the bytes received with `Content-Length` (or the total in `Content-Range`) and raises when they differ.

The first training run read those short files. It had another defect too: 32-bit counts, which wrap after 4 GiB. Its unigram total was `bytes_seen` mod 2³². That checkpoint (`TRI1`) was deleted. The retrain wrote 64-bit counts. It first used the tag `TRI2`, which `trie_stream_train_v2` already uses for a different layout. The tag is now `TR64`, and the file's first four bytes were rewritten in place.

## Size

| | Value |
|--|------:|
| Text seen | 918,559,319,258 bytes |
| Context nodes (orders 2–4 plus 163 order-1 roots) | 53,813,513 |
| Order 2 / 3 / 4 contexts | 10,140 / 916,490 / 52,886,719 |
| Follower counts | 574,113,056 |
| Order-4 contexts seen once | 6,607,240 (12%) |
| Follower counts equal to 1 | 184,376,976 (32%) |
| 4-byte match keys | 52,886,719 (46,282,280 seen at least twice) |
| Checkpoint on disk (`TR64`) | **12.38 GB** |
| Resident after load | 12.3 GB |
| Load time from `D:` through WSL | 76–92 s |
| Packed model (`trie_1tb.pk`) | **992,451,375 bytes** |
| Packed, `zstd -19` (in git) | 464,721,876 bytes |

The node count grew with each kind of text: 23.2 million after 401 GB of C4, 46.8 million after code and most of arXiv (836 GB), and 53.8 million at the end. Code and LaTeX bring contexts that web prose does not. The earlier estimate of about 45 MB for a gigabyte of Wikipedia still describes Wikipedia. It does not describe this mix.

`trie_ckpt verify` (4.5 min, 12.1 GB peak) passes every check on the counts and the trie:

- every node total equals its followers' sum;
- the unigram total equals the bytes seen;
- every order-1 row matches its total;
- every link is in range and every depth is right.

Orders 2, 3 and 4 fall 0.65, 0.97 and 1.29 billion bytes short of the bytes seen. Those are the first two to four bytes of each record and file. The one problem it reports is the unfinished English Wikipedia download (`.part`). It also notes that the 12 checksum lists are listed as trained but are no longer trainable files.

## Packed model

`trie_ckpt pack checkpoint.bin trie_1tb.pk --fit enwik9:0:7340032` writes `packed_trie.hpp`'s `PackedModel` (`CYPHAPK` v1). `cyphalm_trie --model` serves this file. The format holds:

- every follower seen at least twice, with its count quantised to one byte;
- the nodes on paths to those followers;
- the unigram and order-1 tables;
- six mixture weights.

The 64-bit counts are scaled on the way in (`stream_trie.hpp`, `to_packed`):

- Each node's followers are shifted right until the node's total is at most 155 million. That is the largest count the one-byte quantiser keeps apart.
- 2,389 nodes were shifted (at most by 8 bits), and the shift took 1,605 followers to 0.
- The unigram was shifted by 8 bits, and 26 order-1 rows were shifted until they fit 32 bits.
- A shift keeps the ratios the backoff reads. The self-test checks that exactly.

The 4-byte match is left out. Its stored follower is an index into its own side array, not into a training text, and the scores below show it hurts. The weights were fit by exponentiated gradient over enwik9's training slice, with the counts frozen. The holdout was never read for fitting.

| | Value |
|--|------:|
| Nodes / followers kept | 41,809,290 / 389,733,888 |
| Weights (KT0, KT1, orders 2–4, match) | 0.000095, 0.003963, 0.001806, 0.052814, **0.941226**, 0.000095 |
| Pack time, peak memory | 5.5 min, 20.3 GB |
| Resident when served (`context8k`, with an InfiniGram over 19 MB) | 1.1 GB |
| Sampling speed | 13.2 µs a byte (8 MiB trie: 15.0) |

| BPC | enwik9 holdout | enwik9 @ 200 MB | C4, unseen shard | lcet10 |
|-----|------:|------:|------:|------:|
| Unpacked, learned mixture (profile below; weights fit on each text) | 3.206 | 2.878 | 2.173 | 2.337 |
| Unpacked, order-4 backoff (profile below) | 3.281 | 2.907 | 2.173 | 2.321 |
| Unpacked, fixed blend with the match (`trie_ckpt bpc`, Full) | 3.418 | 3.041 | 2.297 | 2.424 |
| **Packed, learned mixture** (its stored weights) | **3.207** | **2.880** | **2.180** | **2.328** |
| Packed, order-4 backoff (Synth, no match) | 3.271 | 2.904 | 2.173 | 2.323 |

Packing costs nothing measurable. Packed order-4 backoff is within 0.01 BPC of the unpacked one on every text, and slightly better on enwik9, where dropping followers seen once helps. The packed mixture uses one set of weights fit on enwik9, so it gives up 0.007 on C4 against weights fit on C4 itself.

## Held-out scores (unpacked, `trie_stream_profile`)

Each run fits the six mixture weights on a slice of the scored text and then scores the next slice. The counts stay frozen. The mixture is the same as in `trie_online_mix.cpp`: KT unigram, KT order 1, stupid backoff (alpha 0.4) stopped at orders 2, 3, and 4, plus a one-hot 4-byte match. "Blend" is the fixed rule from that tool: order-4 backoff plus the match, gated by order-1 entropy.

| BPC | enwik9 holdout | enwik9 @ 200 MB | C4, unseen shard | lcet10 |
|-----|------:|------:|------:|------:|
| Fit slice | `[0, 7340032)` | same weights | first 7 MiB | `[0, 200000)` |
| Scored bytes | `[7340032, 8388608)` | `[200000000, 201048576)` | next 1 MiB | `[200000, 426754)` |
| KT unigram | 5.695 | 5.329 | 4.605 | 5.041 |
| KT order 1 | 4.434 | 4.307 | 3.776 | 4.006 |
| Backoff, order 2 | 4.057 | 3.782 | 3.115 | 3.300 |
| Backoff, order 3 | 3.651 | 3.273 | 2.550 | 2.650 |
| Backoff, order 4 | 3.281 | 2.907 | **2.173** | 2.321 |
| Fixed blend with the match | 3.431 | 3.045 | 2.297 | 2.423 |
| **Learned mixture** | **3.206** | **2.878** | **2.173** | **2.337** |
| Mixture entropy, bits | 2.572 | 2.463 | 2.237 | 2.136 |
| Top-1 accuracy | 43.6% | 47.6% | 54.1% | 57.1% |
| Bytes with an order-4 context | 98.6% | 99.4% | 100.0% | 99.8% |
| Match fires | 98.6% | 99.4% | 100.0% | 99.7% |
| Match correct when it fires | 32.5% | 35.6% | 40.8% | 43.2% |

The C4 slice comes from `c4-train.01023-of-01024`, which was never downloaded for training. It was built from the first 25 MB of that file with non-ASCII removed, so it matches what the trainer saw (`scripts/heldout_c4_slice.py`). lcet10 is the Canterbury technical report. It is plain English and is not in the 40 Gutenberg books. Alice is not scored here, because Gutenberg book 11 is *Alice's Adventures in Wonderland*, which was trained.

Fitted weights:

| | KT0 | KT1 | Order 2 | Order 3 | Order 4 | Match |
|--|----:|----:|----:|----:|----:|----:|
| enwik9 | 0.0001 | 0.0036 | 0.0021 | 0.0550 | **0.9391** | 0.0001 |
| C4 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | **0.9995** | 0.0001 |
| lcet10 | 0.0001 | 0.0001 | 0.0095 | 0.0039 | **0.9230** | 0.0635 |

### What the scores show

- **Order-4 backoff is the model.** It takes 92–100% of the weight on every text. On C4 the learned mixture is order-4 backoff to four decimals.
- **The match table no longer works.** It stores the most recent follower of each 4-byte context. After 918 GB of mixed text, the most recent follower is close to arbitrary: it fires on 99% of bytes and is right only 33–43% of the time. The fixed blend is worse than plain order-4 backoff on every text, by 0.10–0.15 BPC. At 8 MiB it took weight 0.06. Here it falls to the 0.0001 floor everywhere except lcet10. Replacing it with the most frequent follower would just repeat order 4's argmax, so the packed model drops it.
- **enwik9 is out of domain.** The 2006 XML dump is mostly wiki markup (`[[`, `{{`, `&quot;`, `<revision>`), and none of it was trained. The same bytes score 2.298 with the 8 MiB trie, which trained on the 7 MiB just before them. Later in the file, at 200 MB, the article text is denser, and the 1 TB trie scores 2.878.
- **Off Wikipedia, the trie is far stronger than the small one.** 2.17 BPC on unseen web text and 2.34 on lcet10, from counts alone, with no adaptation to the scored text. The 8 MiB wiki trie scores 2.63 and 2.98 on the same bytes (next section). On clean Wikipedia prose (GEN-A's human continuations, below) the 1 TB trie wins too: 2.25 against 2.40. The comparison point is hp gate24: about 1.61 BPC on enwik8, at a resident size of about 1.5–2 GB. Those numbers are on different text, so this is not a direct comparison.
- **Wikipedia is the missing domain.** Adding the dumps needs a bz2 reader in the trainer (or a decompressed `.txt` copy), plus the rest of the enwiki download.

## Against the 8 MiB trie on the same slices

`cyphalm_trie sample … --save trie_8m.pk --no-sim --no-copy --no-bkn` stores the 8 MiB trie, trained as usual on enwik9 `[0, 7340032)` with its weights learned on that stream. That gives 0.93 on order 4 and 0.06 on the match, and here the match stays in, because its followers point into the training text. `trie_ckpt bpc` then scores both packed models on the same bytes.

| BPC (learned mixture; packed) | 8 MiB trie | 1 TB trie | Difference |
|-----|------:|------:|------:|
| enwik9 holdout `[7340032, 8388608)` | **2.298** | 3.207 | +0.909 |
| enwik9 `[200000000, 201048576)` | **2.440** | 2.880 | +0.440 |
| C4, unseen shard | 2.627 | **2.180** | −0.447 |
| lcet10 `[200000, 426754)` | 2.984 | **2.328** | −0.656 |
| Model size | 4.1 MB | 992 MB | |

With the fixed order-4 backoff alone (Synth) the gaps are the same within 0.1: 2.386 / 2.536 / 2.734 / 3.069 against 3.271 / 2.904 / 2.173 / 2.323.

**What this means.** The terabyte is worth about half a bit per byte, but only on the kinds of text it holds. Each trie is good on its own domain and poor off it.

- The 8 MiB trie read 7 MiB of wiki XML and wins on wiki XML, even 200 MB further into the file.
- The 1 TB trie read web pages, code, papers and Q&A, and wins on web text and plain English by 0.45–0.66 BPC.
- An order-4 model knows only which byte follows each 4-byte string. Those statistics are not shared across domains: markup, prose, LaTeX and code use different 4-grams.
- Inside one domain, the counts saturate early. The 8 MiB trie on wiki (2.30) and the 1 TB trie on web text (2.18) end up close, though the second read 125,000 times more bytes.

## Tests on the packed model

Both models below are served by the same `cyphalm_trie` build. The 1 TB trie is loaded with `--model trie_1tb.pk`. The 8 MiB trie is trained in the run as usual: wiki `[0, 7340032)` for `sample`, and wiki plus two Dolly passes for `context8k`. Every other expert is off (`--no-bkn --no-cls --no-sim --no-copy --no-doc`), so the trie is the only difference.

### InfiniGram and unseen questions (`context8k`)

InfiniGram is built over the same text for both: wiki train, a `<|doc|>` break, then Dolly without its 30 held-out turns (19.4 MB). Its lambdas are fit on the first half of the enwik holdout and scored on the second half. The 1 TB trie has not read Dolly. The 8 MiB trie has.

| BPC, holdout second half | 8 MiB trie | 1 TB trie |
|-----|------:|------:|
| Trie alone | **2.3161** | 3.1146 |
| + InfiniGram, context cap 4 | 2.2630 | 2.2698 |
| cap 16 | **2.0607** | **2.2548** |
| cap 64 | 2.0681 | 2.2752 |
| cap 8192 | **2.0684** | **2.2757** |

| Questions (greedy, 150 bytes, answer must contain the expected text) | 8 MiB trie | 1 TB trie |
|-----|------:|------:|
| Seen Dolly turns | 3 / 3 | 3 / 3 |
| Seen, first letter changed | 3 / 3 | 3 / 3 |
| Seen, " Thanks." added | 0 / 3 | 0 / 3 |
| Paraphrased | 0 / 3 | 0 / 3 |
| Held-out Dolly turns | 0 / 30 | 0 / 30 |
| Wikipedia facts | 0 / 8 | 0 / 8 |
| **Unseen questions (paraphrased, held out, facts)** | **0 / 41** | **0 / 41** |

InfiniGram closes most of the gap: 0.84 BPC for the 1 TB trie against 0.25 for the 8 MiB trie. Its index holds the in-domain text, and the 1 TB trie's counts do not. The seen questions are answered by InfiniGram copying Dolly, for both models. No trie answers an unseen question. `qa`'s T-QA dev and test sets were not run: `qa --model` serves only a store built by `qa --save`, which trains its own trie.

### Generation (SCRIBE v2, GEN-A dev)

`scripts/gen_eval.sh dev`: 250 prompts of clean prose from enwik8 `[72, 95.37)` MiB, sampled at temperatures 0.6, 0.8 and 1.0 (600 bytes each, `sample`'s seeds). The samples are scored by SCRIBE v2's ladder (pack `[16, 48)` MiB, reference `[48, 56)`, word vectors rebuilt with `ppmi_svd` on the pack as in SCRIBE v1's instructions, seed 20261027).

| | 8 MiB trie | 1 TB trie |
|---|---:|---:|
| SCRIBE v2, pooled [95%] | 0.020 [0.014, 0.027] | 0.028 [0.021, 0.035] |
| T 0.6 / 0.8 / 1.0 | 0.020 / 0.019 / 0.020 | 0.022 / 0.039 / 0.023 |
| Surface S, pooled [95%] | 0.737 [0.729, 0.745] | **0.796 [0.790, 0.801]** |
| Samples SCRIBE keeps (of 750) | 629 | 648 |
| Mean kept length, bytes | 345 | 389 |
| Human continuations, BPC [95%] | 2.404 [2.375, 2.433] | **2.252 [2.224, 2.279]** |
| Copy share (32-byte spans of enwik8's first 7 MiB) | 0.0032 | 0.0000 |
| Integrity distance D_I | 0.00 | 0.00 |

For scale, held-out human text scores 0.923 and sentences shuffled across documents score 0.344. SCRIBE v2 does not separate the two tries, because the intervals overlap. Both are word salad by SCRIBE's measure.

The 1 TB trie's samples carry no wiki markup. They do carry code and LaTeX, as in the profile samples below. Its words are more often real, so surface S rises, with disjoint intervals. It predicts the human continuations 0.15 bits per byte better.

C4 is a 2019 web crawl and may hold copies of some of these 2006 Wikipedia pages. The copy share checks only enwik8's first 7 MiB, so part of that gain may be memory.

## Samples (unpacked profile)

200 bytes each, temperature 0.8, nucleus 0.9, an 11-byte repeat ban, and seed 42. The weights are the ones fitted for each text. Surface `S` uses each text's own fit slice for the vocabulary and the reference statistics, so `S` cannot be compared across columns.

| Prompt | `S` enwik9 | `S` C4 | `S` lcet10 |
|--------|-----:|-----:|-----:|
| `The ` | 0.909 | 0.806 | 0.740 |
| `In 19` | 0.891 | 0.824 | 0.692 |
| `def ` | 0.878 | 0.738 | 0.597 |
| `\n\n` | 0.920 | 0.826 | 0.571 |

With the C4 weights:

> The Count of stuff, so income allow to be able for the money or with the most efficular is a lated the close of relieve that in a process the same time to make a chall probably seconomy of the up of poten

> In 1952.\nBut the used out of the new large amouseholding the of the pression in a serience to ensure to get that will be addition for some of the for work. What I this back to the for and that are that can

The words are real and the phrases run on. Nothing holds together past a few words. Code and LaTeX leak into prose: `$a_n \in \{1,\cdot`, `</message>`, `// Retry.`, `{\n            find`. Four bytes of context cannot tell a Python file from a news article.

## Model 2 (training)

This is the runbook's second model: `trie_stream_train_v2`. It is the same model and the same corpus files as model 1, through `stream_trie.hpp`. It uses 32-bit counts that halve instead of wrapping (`TRI2`), and keeps snapshots at 8M, 64M, 1G, 10G and 100G for the scaling curve. Started 2026-10-31 01:14 with

```bash
CYPHA_THREADS=32 bash scripts/train_model.sh 2      # D:\cypha_llm\run_model2.sh, in its own window
```

into `<corpus>/trie/model2/` (log: `<corpus>/trie/reports/train_model2.log`).

**32 threads.** `--threads N` (`stream_trie.hpp`, `train_file_parallel`) works like this:

1. Each file is cut into N byte ranges. A `.jsonl` cut moves to the next line, so every record is read whole by one thread.
2. Each thread trains its own model on its range.
3. The models merge pairwise in parallel, later data into earlier, then once into the main model.
4. A count that would pass the cap halves its table first, as in one-thread training.

On `.jsonl` files the result is the one-thread model exactly: the self-test (`trie_ckpt selftest`, 22,339 checks) finds the same counts, structure and match entries, and a probability difference of 0. On `.txt` files it differs only by the history reset at each cut.

| One C4 file (766 MB of text), empty model | Wall time | Peak memory |
|-----|------:|------:|
| 8 threads (model 1's setting) | 88 s | 0.97 GB |
| 16 threads | 50 s | 1.2 GB |
| **32 threads** | **39 s** | 2.0 GB |

The serial merge into the main model grows with the trie, so later files take longer. File 2 took 45 s. Model 1's C4 files averaged about 129 s each on 8 threads.

Snapshots are taken at file boundaries, and a file holds about 0.76 GB of text. So the 8M and 64M snapshots were both written after the first file, at 0.77 GB. The small end of the scaling curve comes from the 8 MiB runs instead.

When it finishes, the next steps are:

```bash
bash scripts/verify_model.sh 2
bash scripts/compare_models.sh 1 2
```

Then each snapshot is scored with `trie_ckpt bpc` for A1.

## Track A predictions

[`PLAN.md`](../PLAN.md) registered these predictions on 2026-10-26.

| ID | Prediction | Measured | Result |
|----|-----------|----------|--------|
| P-A1 | Trie-only BPC on the enwik 8 MiB holdout: 2.00–2.25 | 3.206 (packed 3.207; order-4 backoff alone 3.281) | **Missed.** No Wikipedia was trained. This measures transfer from web, code, and papers to 2006 wiki XML, not saturation of the counts |
| P-A2 | Trie + 8K InfiniGram: 1.85–2.05 | 2.2757 (holdout second half; the 8 MiB trie scores 2.0684 the same way) | **Missed**, for the same reason. InfiniGram's text is wiki and Dolly; the trie's counts are not |
| P-A3 | Distinct 4-grams: 50–200 M | 52.9 M order-4 contexts; packed size 992 MB, far above the 45 MB enwik9 estimate | **Held**, near the low end |
| P-A4 | Held-out answers to unseen questions: ≤ 2% | 0 / 41 (30 held-out Dolly turns, 8 Wikipedia facts, 3 paraphrases; the 8 MiB trie also 0 / 41) | **Held.** `qa`'s T-QA sets were not run on the 1 TB trie |
| P-A5 | Surface `S` unchanged within ±0.05 | GEN-A dev, 750 samples: 0.737 → 0.796 (+0.059, intervals disjoint). SCRIBE v2 0.020 → 0.028 (intervals overlap) | **Missed**, just: surface S rose by more than 0.05. The generation score did not move beyond its interval |

The Track A protocol was not followed as written. There was no frozen per-category holdout before training, no 1 GB packed shards, and no checkpoints at 8 MiB, 64 MiB, 1 GB, 10 GB, or 100 GB. So experiment A1 (the scaling curve) cannot be fitted from this run.

## Time

| Step | Time |
|------|-----:|
| Train, one pass, 976 files | about 48 h (2026-10-28 21:31 to 2026-10-30 21:22), about 19 GB of text per hour |
| SHA-256 of 986 GB on `D:` | 2 h at about 130 MB/s |
| Re-fetch of the 18 short files | about 3 h (Stack Exchange alone was 34 GB) |
| Profile on enwik9 (fit 7 MiB, score 2 MiB, 4 samples) | 11.6 min |
| Profile on C4 / lcet10 | 10.5 / 5.0 min |
| `trie_ckpt verify` | 4.5 min, 12.1 GB peak |
| `trie_ckpt pack` (scale, pack, fit weights on 7 MiB, write) | 5.5 min, 20.3 GB peak |
| `trie_ckpt bpc` on the checkpoint, 4 slices | 6.4 min, 20.3 GB peak |
| `zstd -19 -T16` of the packed file | 1.6 min |
| `context8k`, 1 TB trie / 8 MiB trie | 1.1 / 4.7 min |
| `gen_eval.sh` (`sample` + `scribe ladder`), 1 TB trie / 8 MiB trie | 15 s + 10 s / 46 s + 10 s |

Each merge into the shared trie got slower as the trie grew. Stack Exchange, a single 80 GB file split into eight 10 GB shards, was the largest step.

## Reproduce

```bash
# WSL, from the repository root
bash scripts/build_1tb_tools.sh                 # ~/bin: trainer, profiler, trie_ckpt (+ selftest), cyphalm_trie, scribe, ppmi_svd
python3 scripts/verify_corpus_sha.py --root /mnt/d/cypha_llm/corpus
~/bin/trie_stream_train --once /mnt/d/cypha_llm/corpus
T=/mnt/d/cypha_llm/corpus/trie
~/bin/trie_ckpt verify $T/checkpoint.bin --corpus /mnt/d/cypha_llm/corpus
~/bin/trie_stream_profile $T/checkpoint.bin /mnt/d/cypha_llm/enwik9
~/bin/trie_ckpt pack $T/checkpoint.bin $T/trie_1tb.pk --fit /mnt/d/cypha_llm/enwik9:0:7340032
~/bin/trie_ckpt bpc $T/trie_1tb.pk /mnt/d/cypha_llm/enwik9:7340032:1048576 ...
~/bin/cyphalm_trie context8k /mnt/d/cypha_llm/enwik9 /mnt/d/cypha_llm/chat/dolly.jsonl --model $T/trie_1tb.pk \
    --no-bkn --no-cls --no-sim --no-copy --no-doc
CYPHALM_TRIE=~/bin/cyphalm_trie SCRIBE=~/bin/scribe WIKI=/mnt/d/cypha_llm/enwik9 ENWIK8=/mnt/d/cypha_llm/enwik8 \
    SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 scripts/gen_eval.sh dev T1TB OUT --model $T/trie_1tb.pk --no-sim --no-copy --no-bkn
```

- **Trainer.** `--once` stops when no unread file is left. Without it the trainer waits for the fetch to finish more files. It refuses to resume from any checkpoint that is not `TR64`.
- **Paths.** Give `--corpus` as an absolute path, because `trained.txt` holds absolute paths.
- **`/tmp`.** Build into `~/bin`, not `/tmp`: WSL clears `/tmp` when the VM restarts.
- **SCRIBE vectors.** `v32` is `ppmi_svd train32.xml none none - v32 --no-ri --no-ablation` on enwik8 `[16, 48)` MiB, which takes 37 s.

## Next

1. **Train Wikipedia.** Add a bz2 or extracted-text reader and finish the enwiki download. This is what P-A1 and P-A2 are missing.
2. **Keep non-ASCII text.** Fix the `\u` escapes in the JSON reader.
3. **Hold out before training.** Set aside a frozen slice per category, as step 1 of the plan says. Model 2 (running) keeps the snapshots for A1's scaling curve.
4. **Run the T-QA sets.** Let `qa` take a stored trie (`--model FILE`) next to its own InfiniGram and experts, so the T-QA dev and test sets can be read on the 1 TB trie.
