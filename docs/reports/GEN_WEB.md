# GEN-W: a second benchmark on web prose, and a clean C4 corpus

**Odin Loch** · 2026-11-12 · GENERATION_PLAN §1.3 rule 3, CT0.3 and CT0.4 C1 (§6.2) · Odin's PC

**Why.** GEN-A is Wikipedia, and SCRIBE's pack and reference are Wikipedia too. A writer could learn to please that one genre. §1.3 rule 3 asks for a second benchmark in another genre, scored against its own reference: "a gain must show on both". GEN-W is that benchmark. It is built from C4 web pages. C1 is a clean C4 prose corpus for a later K5 read, disjoint from GEN-W.

This section was committed and pushed before anything below was built.

## Registration (2026-11-12, before any build)

### Which C4 text

The 1 TB trie read all 739 C4 shards on the PC (`00000`–`00738`, TRIE_1TB_PROFILE.md). The only C4 text it never read is `c4-train.01023-of-01024`. On the PC that shard exists only as a 9 MB slice, `D:\cypha_llm\profile\c4_heldout.txt` (4,275 documents). It cannot be the benchmark:

- It was written with every non-ASCII character deleted, to match what the trainer saw. In C4 that mangles words: "Can’t" becomes "Cant", "John’s" becomes "Johns", "café" becomes "caf".
- Measured on the first 20,000 documents of shard `00738`: 48.5% of documents hold a character whose deletion mangles a word (’, ‘, a letter or a symbol other than quotes, dashes and the ellipsis).
- A filter that sees only the stripped text (contractions without an apostrophe, double spaces, a space before punctuation) keeps 51% of documents, and 29% of those are still mangled. Possessives cannot be seen.
- There is no network to fetch the shard again.

So GEN-W is built from trained shards, in full UTF-8. That is safe for the 1 TB trie. It is a count model of orders 2–4 with a 4-byte match table. It cannot hold a 32-byte span of any document, and 2 KB of a document are 2 parts in 10^9 of its 918.6 GB.

**Rule for later arms.** No model that stores spans of text (InfiniGram, copy, similar-context keys, K5's counts) may read GEN-W's four shards. An arm that uses the 1 TB trie says so in its report.

| Use | Shard | What is taken |
|---|---|---|
| GEN-W source (prompts and continuations) | `c4-train.00738-of-01024` | the first 20,000 documents |
| SCRIBE-W pack | `c4-train.00737-of-01024` | documents in file order until 40 MiB of text (WebText's pack size) |
| SCRIBE-W reference | `c4-train.00736-of-01024` | the first 5,000 documents (WebText's reference size) |
| SCRIBE-W rungs (human and control rungs of every `gen_eval` run) | `c4-train.00735-of-01024` | the first 5,000 documents |
| C1 (clean prose for K5) | `00000`, `00001`, `00002` | in order, cut at 1.5 GB |

Nothing in this table is used twice.

### Cleaning (every file above)

1. The JSON is decoded in full (`\uXXXX` included), so accented names survive.
2. Typographic marks become ASCII, because CyphaLM's writer learned Wikipedia's ASCII punctuation: ’ ‘ ʼ → `'`; “ ” „ → `"`; – → `-`; — → ` - `; … → `...`; no-break space → space. Zero-width spaces, soft hyphens and byte-order marks are removed. Every other character is kept as UTF-8.
3. A document whose cleaned text equals an earlier one's (same bytes) is dropped.
4. SCRIBE's plain-text cleaner (`clean_plain_document`, prose lines only) runs on every document. `scribe prompts` and `scribe ladder` apply it to JSONL themselves; C1 uses `scribe docs`.

### GEN-W rows

**Pool.** `scribe prompts SOURCE.jsonl --n 100000 --seed 20261112 --set genw --out pool.jsonl`, SCRIBE's own prompt maker, as GEN-A: one row per document, a prompt of 100–300 bytes and the true continuation of 400–600 bytes with at least 3 sentences, in a seeded order.

**A row is dropped** (checked in pool order) when any holds:

1. The prompt or the continuation holds `http`, `www.` or `@` (links and addresses).
2. More than 10% of the continuation's bytes, or more than 25% of the prompt's, lie in 32-byte spans that occur in the default writer's training text: `fuzzy/train.txt`, K5_SCALE's `train_all.txt` (the 337 MB default) and enwik8's first 8 MiB raw. GEN-A's rule, applied to today's writer.
3. The same, for the SCRIBE-W pack, reference and rungs (SCRIBE-cleaned), and GEN-A's TSV texts. C4 repeats boilerplate across pages.
4. The prompt or continuation shares any 32-byte span with a row already kept (either split).

**Split.** As GEN-A: a row goes to dev when the FNV-1a 64 hash of its prompt's UTF-8 bytes is even, else to test. Each file keeps its first 250 clean rows in pool order.

**Files.** `bench/eval/gen_web_dev.tsv` and `bench/eval/gen_web_test.tsv`, GEN-A's columns (id, doc, prompt, continuation; `\t`, `\n`, `\\` escaped). `doc` is the line number in the source JSONL. Large files stay on the PC under `/mnt/d/cypha_llm/genw/`; `bench/gen_web/MANIFEST.tsv` lists each with bytes and sha256.

### SCRIBE-W

- Vectors: `ppmi_svd` on the pack, wrapped as XML pages (`scripts/scribe_eval/runs/v3_paper_tools/wrap.py`), `--no-ri --no-ablation`, as for WebText.
- `gen_eval.sh` gets `GEN_BENCH=web`: it reads `bench/eval/gen_web_SPLIT.tsv`, and runs `scribe ladder` with the SCRIBE-W pack, reference, rungs and vectors and `--dedup`. Without `GEN_BENCH`, it runs exactly as before.

### The ladder check (before any writer run)

`scribe ladder --profile v2` and `--profile v31`, seed 20261112, `--dedup`, on two test regions:

- **S:** GEN-W's source region, the 20,000 documents without the documents of the 250 test rows. This is the check SCRIBE made on Wikipedia's fresh region.
- **R:** the rungs region (`00735`), which every `gen_eval` run reads.

A pair passes with accuracy at least 0.95 and disjoint 95% intervals. GEN-W is usable when v2 passes 5 of 5 on both regions. v3.1's result is reported; it does not decide.

### C1

- Shards `00000`–`00002`, cleaned as above, one document per line, like `train_all.txt`.
- **Overlap rule** (`scripts/k5_overlap.cpp clean`, K5_SCALE's tool and rule). Forbidden: every 32-byte span of (a) GEN-W's and GEN-A's prompts and continuations, (b) the SCRIBE-W pack, reference and rungs, (c) SCRIBE's Wikipedia pack, reference and test regions (enwik8 [16, 64) MiB, cleaned). A document whose hits cover more than 10% of its bytes is dropped. Otherwise each sentence that touches a hit is cut.
- Then the shards are joined in order and cut at the first document boundary past 1,500,000,000 bytes.
- A second, independent `k5_overlap check` on the written file must find no forbidden span.
- Written to `/mnt/d/cypha_llm/c4prose/c1_train.txt`, with its manifest. K5 is not run on it here.

### The default writer on GEN-W dev (once)

`GEN_BENCH=web scripts/gen_eval.sh dev DEF … --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`: main fb79cc93's default (K5 counting from 337 MB, the gate of 16, the LDA term, the discourse planner), `K5_THREADS=4`, started only when `free -g` shows at least 14 GB available. GEN-W test is not read.

### Predictions

| # | Prediction |
|---|---|
| W1 | The pool holds at least 1,000 rows. Rules 1–4 drop under 15% of the rows read before both files are full |
| W2 | Rule 2 (the writer's training text) drops under 3% of the rows read: web pages rarely copy 2006 Wikipedia |
| W3 | SCRIBE v2 passes 5 of 5 on both S and R (about 3 in 4; WebText passed 5 of 5) |
| W4 | SCRIBE v3.1 passes 5 of 5 on S and R: even odds. On WebText it passed 4 of 5; the weak pair was human > sentences shuffled (0.948) |
| W5 | The human rung's SCRIBE v2 is 0.90–0.94 on S |
| W6 | The default writer's dev SCRIBE v2 on GEN-W is 0.40–0.60, at least 0.05 below its GEN-A dev score (L1, 0.610): a Wikipedia writer judged against web text |
| W7 | Its v3.1 is below its v2, as on GEN-A |
| W8 | Its `k5_cont_bpb` on GEN-W is at least 0.15 above GEN-A dev's (1.357) |
| W9 | Its copy32 is within ±0.05 of GEN-A dev's (0.242); no 13-word copy |
| W10 | Peak RAM 11–12 GB; wall 5–8 minutes |
| W11 | C1 is 1.0–2.0 GB. The overlap rule removes 2–15% of the cleaned bytes; the recheck finds 0 |
