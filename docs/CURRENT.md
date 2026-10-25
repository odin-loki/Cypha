# Where the numbers are

Two language-model lines live in this repo. They are not the same model.

## Shipping CyphaLM

hp gate24 context mixer. Quality bar on enwik8: observe **1.611729 BPC**, archive **1.611759 BPC** (measured 2026-09-19). Lab RSS at mem 22 is about **1.5–2 GB**. Lossy tiers are smaller: lean about **1.08 GB** and **1.610 BPC**, tiny about **253 MB** and **1.652 BPC** in the lossy report. A tiny-tier run on the first 8 MiB in October 2026 used about **40 MB** RSS and scored about **1.92 BPC**, with a best sample surface near **0.997**.

Serve and train notes: [`native/CYPHALM_SERVE.md`](native/CYPHALM_SERVE.md). Lossy table: [`reports/CYPHALM_LOSSY_MIXER_REPORT.md`](reports/CYPHALM_LOSSY_MIXER_REPORT.md).

The D17 number **2.664 BPC** in `bench/BASELINE_LOCK.json` is the Hybrid GRIA+LSTM bench lock. That stack is retired as the language model. [`history/LEGACY_LLM.md`](history/LEGACY_LLM.md).

## Small trie

Tools: `native/tools/trie_fold_lm.cpp`, `trie_online_mix.cpp`, `trie_together.cpp`, `trie_chat.cpp`. Surface score: `native/tools/surface_score.cpp`.

Protocol: enwik9 bytes `[0, 7340032)` train, `[7340032, 8388608)` frozen holdout.

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

### Not ready to scale

Training still allocates a node for every context, including the ones seen once. The 5.4 MB number is what `measure()` would keep. It is not the resident size. On the 8 MiB slice the node and follower vectors are already about **25 MB**. At a few gigabytes those vectors are the first thing to run out, and the match table doubles while it grows. The Dolly lookup also builds a second string of the whole corpus.

Do not train this trie on the terabyte until the packed nodes are what the process keeps, the match table compacts without holding two copies, and long context is a walk of a stored position rather than a new node per longer gram.

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

When the bytes are on disk, the train order is unstructured text first, then labelled chat as `User:` / `Assistant:` turns. A question is answerable when that exact turn was stored.
