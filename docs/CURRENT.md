# Where the numbers are

The language model being taken forward is the packed byte trie. hp gate24 stays in the tree as a comparison. It scores better on a small slice and it costs too much RAM to train on a terabyte or to keep loaded for day-to-day use.

**The default plan is now [`PLAN.md`](PLAN.md)** (2026-10-26). The 1 TB run below continues inside it as Track A, with its predictions registered there. The question-answering and 8K count-context results are in [`reports/CYPHALM_QA_AND_CONTEXT_TEST.md`](reports/CYPHALM_QA_AND_CONTEXT_TEST.md).

## Packed trie (the model to train)

Tools: `native/tools/cyphalm_trie.cpp` (`bpc`, `chat`, `probe`, `context8k`, `sample`, `selftest`) over the one model in `native/include/cypha/cyphalm/packed_trie.hpp`. On 2026-10-27 it replaced the forked `trie_*.cpp` tools and reproduces their numbers to the printed digit. The experiments behind the 2.286, 2.642 and word-trie rows (`trie_together`, `trie_gria_gate`) were retired; their results and the git commits that hold them are in [`reports/B1_TRIE_CONSOLIDATION.md`](reports/B1_TRIE_CONSOLIDATION.md). Surface score: `native/tools/surface_score.cpp`.

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
| + class-context expert, second half of the holdout (B4) | 1.999 / 1.982 | Over trie + InfiniGram (2.0725): shape classes / E1 tags, tail 6. `cyphalm_trie cls`. On fresh text 2.166 → 2.088 / 2.075 |
| + copy expert, second half of the holdout (A1c) | 1.927 | Over trie + InfiniGram (2.072), chat model; `qa --copy`; an option, not a default |

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

## Question answering, CyphaLM by itself

Tool: `cyphalm_trie qa` (dev 130 questions, frozen test 200; coverage at a 2% wrong rate by 5-fold CV inside dev). Reports: [A0/A1](reports/A0_A1_CYPHALM_QA_AND_DONT_KNOW.md), [A1c](reports/A1C_ENTITY_FAMILIARITY_AND_COPY.md), [A2/A3](reports/A2_A3_FACT_TEXT_AND_STATEMENT_SCORING.md).

| Read | Value |
|---|---|
| A0, CyphaLM alone | 0.0% coverage at 2% wrong on dev; raw right 0–2 / 130 |
| A2, + fact shard (R1) | dev 2 right / 0 wrong out of fold (1.5%, Wilson 0.4–5.4%); test +0 |
| A3, statement scoring (R3) | 0% at 2% wrong |
| A2 + A3, frozen test, gated | 1 right / 0 wrong |
| Entity familiarity (`ent_title`) | dev AUROC 0.795 [0.745, 0.847] |
| Copy expert (option) | holdout BPC trie 2.114, trie + InfiniGram 1.926 |

## Word vectors

PPMI-SVD (`native/tools/ppmi_svd.cpp`) stays the source: WordSim-353 0.672, SimLex-999 0.255, MEN 0.637 on all of enwik8 (C1 pairs). Cypha's own learner, `cypha_word_classes train --method sgns --epochs 5 --var0 0.3` ([S-1b](reports/S1B_LEARNED_CONTEXTS.md)): 0.683 / 0.246 / 0.669 on the same pairs, 0.750 / 0.270 / 0.756 on the top-10,000 pairs; no worse on common words, worse on rare ones.

## Fact store and fact text (for CyphaLM to learn from)

Tool: `native/tools/pattern_learn.cpp` ([D4 report](reports/D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md), 2026-10-28). These facts never answer anything. They become training text for CyphaLM (PLAN rule 10).

| Read | Value |
|---|---|
| Text seeds (infobox, table, category) | 7,484 (8,148 before a citation-template seed bug was fixed) |
| Kept patterns, `--cv --wikidata` | 70 (F2 alone: 44) |
| Admitted facts extracted from the text | 1,216 (F2 alone: 832) |
| Agreement with Wikidata on held-out subjects | 0.951, Wilson [0.909, 0.974] |
| Fact shard (`--shard`) | 2,001 question-form rows from 675 facts on pages in the 8 MiB slice; 0 equal to a dev or test question; dev reach 16 / 100 in-slice questions |

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
