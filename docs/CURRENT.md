# Where the numbers are

The language model being taken forward is the packed byte trie. hp gate24 stays in the tree as a comparison. It scores better on a small slice and it costs too much RAM to train on a terabyte or to keep loaded for day-to-day use.

**The default plan is now [`PLAN.md`](PLAN.md)** (2026-10-26). The 1 TB run below continues inside it as Track A, with its predictions registered there. The question-answering and 8K count-context results are in [`reports/CYPHALM_QA_AND_CONTEXT_TEST.md`](reports/CYPHALM_QA_AND_CONTEXT_TEST.md).

## Packed trie (the model to train)

Tools: `native/tools/trie_fold_lm.cpp`, `trie_online_mix.cpp`, `trie_together.cpp`, `trie_chat.cpp`. Surface score: `native/tools/surface_score.cpp`.

Protocol for the numbers below: enwik9 bytes `[0, 7340032)` train, `[7340032, 8388608)` frozen holdout, unless a row says otherwise.

| Read | BPC | What it means |
|------|----:|---------------|
| Fixed backoff, orders 1–4, 4-byte match | 2.386 | Counts only, alpha 0.4 |
| Learned mixture, weights frozen | **2.298** | About **0.93** on order 4, **0.06** on the match |
| Same mixture, weights still adapting on the holdout | 2.286 | The router is learning the test bytes. The counts are not. |
| PPMD, longer chains, or inheritance | ~2.42 | Worse than backoff |
| GRIA as a probability gate | 2.642 | Worse than backoff |
| Packed-layout accounting | — | **5.4 MB** if singletons are dropped and counts are one byte |
| After 15,011 Dolly turns | 2.341 wiki | Stored accounting about **11 MB** |

Surface scores of trie samples are about **0.74–0.83**. That is readable fragments, not answers.

BPC is held-out `−log2 p`. It is not a generation score.

### What the measurements settled

- Context hashes are spread out. The bytes underneath are not. Order-2 prefixes on 2 MiB had about **6.2** extensions on average, and **47%** of order-4 contexts occurred once.
- Enumerating context masks does not shrink the tables.
- `32/(32+n)` and NIG `r_eu` die after many counts. They are not mix weights. The uncertainty used on the match is `u = H(order-1) / ln(256)`, about **0.49** on this holdout.
- A word-sized trie added as another expert fell to weight **0.0005** and did not change the samples.
- Dolly questions that occur byte-for-byte in the stored chat are copied back. The copy is a search for that prompt, not the order-4 model reading the question. An unseen question stays Wikipedia-shaped. A `<|doc|>` marker is written between the Wikipedia bytes and the chat so those contexts do not glue together.

### What grows, and what must stay separate

Training keeps a node for every context, including ones seen once. The **5.4 MB** figure is the later packed pass (drop counts below 2, one byte per kept count). It is not the resident size during the pass. On the 8 MiB slice the node and follower vectors are about **25 MB**, and the process peaks near **116 MB**.

Counted across all **10⁹** bytes of enwik9, distinct 4-byte contexts do not grow in proportion to the file:

| Bytes | Distinct 4-grams | Seen once | Peak training RAM |
|------:|-----------------:|----------:|------------------:|
| 8 MB | 331,145 | 43% | 116 MB |
| 64 MB | 979,017 | 41% | 128 MB |
| 256 MB | 1,865,580 | 39% | 256 MB |
| 10⁹ | 3,222,263 | 37% | 400 MB |

From 512 MB to 10⁹ bytes the 4-gram count grows like about N^0.47. The packed upper bound at 10⁹ bytes is about **45 MB**. A gigabyte of Wikipedia-like text fits this trie. The match record is a `uint32` offset into the file, so a corpus past 4 GiB needs the next byte stored in the record instead of an index.

Chat must not be poured into those counts. On the 8 MiB split, the 4-byte match fires on **96%** of holdout bytes, and **75.7%** of holdout bytes use a 4-gram that also occurs in Dolly. The match table keeps only the latest follower, so those wiki predictions become the last Dolly byte. Every Dolly prompt also ends in the same 12 bytes, `\nAssistant: `, so a fixed 8-byte or 16-byte key does not tell the questions apart (0 of 15,011 prompts are unique at 12 bytes; **19%** are unique at 16). The chat index has to be the whole prompt pointing at its answer, in its own table, after the wiki counts and the mixture weights are frozen.

## Corpus on D:

`scripts/fetch_corpus.py` writes `D:\cypha_llm\corpus` (`/mnt/d/cypha_llm/corpus` from WSL). Ctrl+C stops it. The same command continues. `scripts/run_fetch.sh` is the visible WSL launcher.

| Category | Cap | What it is |
|----------|----:|------------|
| Books | 40 GB | Gutenberg texts, Wikisource, Wikibooks |
| Code | 200 GB | RedPajama GitHub |
| Articles | 580 GB | C4, RedPajama Common Crawl, English Wikipedia, 50 GB of WET from `CC-MAIN-2026-43` |
| Papers | 90 GB | RedPajama arXiv |
| Structured | 80 GB | Stack Exchange |
| Labelled | 20 GB | Dolly, OpenAssistant, AG News, SQuAD, IMDB |

The RedPajama book files are not downloaded. That slice was removed for copyright. Gutenberg plus Wikisource is the book text, and it is tens of gigabytes, not hundreds. The old PubMed bulk directory and `https://arxiv.org/src/arXiv_src_manifest.xml` return 404. Papers come from the RedPajama arXiv list.

Seed URLs were probed on 2026-10-25 (`python scripts/fetch_corpus.py --check`). Each returned HTTP 206 with a real body.

## hp gate24 (comparison only)

Observe **1.611729 BPC**, archive **1.611759 BPC** on enwik8 (2026-09-19). Lab RSS at mem 22 is about **1.5–2 GB**. A tiny-tier run on the first 8 MiB used about **40 MB** and scored about **1.92 BPC**, with a best sample surface near **0.997**. Those tables are sized in advance. They are not the model trained on this corpus. Serve notes for that binary: [`native/CYPHALM_SERVE.md`](native/CYPHALM_SERVE.md).

The D17 number **2.664 BPC** in `bench/BASELINE_LOCK.json` is a bench lock for the retired hybrid. [`history/LEGACY_LLM.md`](history/LEGACY_LLM.md).

## Plan: train the trie on about 1 TB

This is Track A of [`PLAN.md`](PLAN.md). The experiments A1–A8 and predictions P-A1 to P-A5 are recorded there.

Unstructured text and labelled chat are different jobs.

1. **Hold out text first.** From each category, set aside a frozen slice before any count is updated. Score BPC and surface `S` there. Score the same 8 MiB enwik holdout with the existing gate24 and tiny checkpoints. That is the comparison. Gate24 is not trained on the terabyte.
2. **Shard the unstructured text at about 1 GB.** A full pass of enwik9 (10⁹ bytes) peaked near **400 MB**. One shard at a time stays in that range. Write a `<|doc|>` break between files so order 4 does not glue the end of one file to the start of the next.
3. **Pack the shard, then free it.** Drop counts below 2, store the kept counts in one byte, and drop the fat nodes before the next shard. Merge shards by adding the counts for the same context, then drop counts below 2 again. The mixture weights stay the ones fit on the 8 MiB slice (about **0.93** on order 4 and **0.06** on the match). They are not re-learned across the terabyte.
4. **Match record stores the next byte.** A `uint32` index into the file wraps past 4 GiB. After a shard is packed, the match entry is the predicted byte, not an offset that pins the corpus in RAM.
5. **Labelled chat is a second index.** Freeze the trie and the mixture. Build a map from the whole `User: … Assistant:` prompt to the stored answer. Do not `observe` Dolly into the trie. On the 8 MiB split, **75.7%** of holdout bytes shared a 4-gram with Dolly, and the match keeps only the latest follower, so those wiki predictions became the last Dolly byte. A fixed 8-byte or 16-byte key cannot separate the prompts: every one of the 15,011 ends in `\nAssistant: `.
6. **Ask both ways.** A prompt that is in the chat index returns the stored answer. A prompt that is not is a sample from the packed trie. Report BPC, surface `S`, and the resident size of the packed model next to the gate24 numbers on the small slice.

Order of the unstructured shards: Wikipedia and articles, then books, code, and papers, then Stack Exchange. The labelled index is last.
