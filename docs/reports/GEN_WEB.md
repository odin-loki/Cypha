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

## Results (2026-11-12)

**Outcome.**
- GEN-W is built: 250 dev and 250 test rows of C4 web prose, frozen.
- SCRIBE v2 passes 4 of 5 control pairs on it, not 5. The failed pair is the bottom one: random lexicon words > random letters (0.886 on S, 0.901 on R). Both rungs score under 0.003. By the registered rule, GEN-W is **not yet cleared** for decisions. Odin's call (below).
- SCRIBE v3.1 passes 5 of 5 on R and 4 of 5 on S (human > sentences shuffled 0.938).
- Today's default writer scores **0.691 [0.665, 0.718]** on GEN-W dev (SCRIBE v2), above its GEN-A dev score (0.610). GEN-W test was not read.
- C1 is written: 1,500,000,876 bytes, 764,046 documents, in `/mnt/d/cypha_llm/c4prose/c1_train.txt`. The overlap rule removed 10.2% of the cleaned bytes. The final check finds no forbidden span.

Tools: main fb79cc93 (branch `wip/gen-web`), `g++ -O2` into `~/genweb_bin` (`scribe selftest`: 120 checks, 0 failed). Build: `scripts/gen_web_build.sh extract clean vectors pool bench ladder c1prep c1 manifest`.

### What changed from the registration

- `gen_eval.sh` was not changed. The harness is a new script, `scripts/gen_eval_web.sh`, with the same three steps and the SCRIBE-W files. GEN-A's harness is untouched.
- Exact repeats were dropped twice: on the decoded text, and again after SCRIBE's cleaner. Neither found a repeat in GEN-W's four regions.
- SCRIBE's cleaner runs once, with `scribe docs`. Every later read uses `raw:` on its output, so line i is document i (the TSVs' `doc` column).

### Regions

| Region | Shard | Documents read | After SCRIBE's cleaner | Text bytes |
|---|---|---:|---:|---:|
| GEN-W source | 00738 | 20,000 | 19,552 | 39,620,548 |
| SCRIBE-W pack | 00737 | 19,226 (40 MiB of text) | 18,830 | 38,951,819 |
| SCRIBE-W reference | 00736 | 5,000 | 4,900 | 10,120,176 |
| SCRIBE-W rungs | 00735 | 5,000 | 4,910 | 10,185,347 |

Vectors: `ppmi_svd` on the wrapped pack: 18,830 pages, 6.42 M tokens, 50,000 words × 300 (83 s, 538 MB).

### GEN-W rows

`scribe prompts` gave a pool of 12,816 rows from 19,552 documents. The first 4,000 were scanned for the span rules. 572 rows were read before both files were full.

| Dropped before the 250 kept | dev | test |
|---|---:|---:|
| 1. a link or address | 3 | 3 |
| 2. the writer's training text (>10% / >25% in 32-byte spans) | 14 | 21 |
| 3. SCRIBE-W or GEN-A text | 17 | 13 |
| 4. a span shared with a kept row | 1 | 0 |

| File | Rows | Documents | Prompt bytes (mean, range) | Continuation bytes (mean, range) | Rows with non-ASCII | sha256 |
|---|---:|---:|---|---|---:|---|
| `bench/eval/gen_web_dev.tsv` | 250 | 250 | 159 (100–294) | 510 (400–599) | 15 | `80f24331e1ca3be2364d2d4ed8f584f783386964e1f84559bd9e398f04375966` |
| `bench/eval/gen_web_test.tsv` | 250 | 250 | 161 (100–296) | 519 (404–600) | 23 | `928fb03c0c52950cffeca30349b0f265a53aa688bf7d6a4b33c33775a495f991` |

No document is in both files. The text is real web prose, spam included. dev/0000 is a spun betting page ("These guys are professionals as well as their job to from all the variables"). C4's filter kept it, and so does GEN-W. The human rungs hold such pages too.

### The ladder check

`scribe ladder … --dedup --seed 20261112`. S is the source region without the 250 test rows' documents (19,302; `--dedup` dropped 192 more). R is the rungs region (4,910; 55 dropped). Reports: `bench/gen_web/ladder/`.

| Pair (accuracy) | v2 S | v2 R | v3.1 S | v3.1 R |
|---|---|---|---|---|
| human > sentences shuffled | 0.958 PASS | 0.967 PASS | 0.938 FAIL | 0.954 PASS |
| sentences shuffled > words shuffled | 1.000 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS |
| words shuffled > random lexicon words | 0.975 PASS | 0.983 PASS | 0.967 PASS | 0.965 PASS |
| random lexicon words > random letters | **0.886 FAIL** | **0.901 FAIL** | 1.000 PASS | 1.000 PASS |
| random letters > one phrase repeated | 1.000 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS |
| **pairs passed** | **4** | **4** | **4** | **5** |

Rung means, SCRIBE v2 on S: human 0.908 [0.891, 0.924], sentences shuffled 0.398, words shuffled 0.008, random lexicon words 0.002, random letters 0.001, one phrase repeated 0.000. On R: human 0.914, sentences shuffled 0.404.

- **Why the bottom pair fails.** Both rungs are junk and score near 0. v2 counts lexical validity only beyond the reference's 99th percentile. Web pages hold more non-words than Wikipedia (brand names, typos, spun text). So the threshold is higher, and random letters' L axis reaches only 26.9 of its cap of 30. On Wikipedia the same pair was 0.998.
- **What it means.** A writer that scores 0.4–0.7 is ranked by the upper pairs, and they pass. But the registered rule asks for 5 of 5, and it is not met. SCRIBE is frozen; nothing was tuned to make the pair pass.

### The default writer on GEN-W dev (run once)

`scripts/gen_eval_web.sh dev DEF ~/genweb_runs --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`: main's default, `K5_THREADS=4`, started with 31 GB available. Files: `bench/gen_web/runs/`.

| Measure | GEN-W dev | GEN-A dev (L1, the same default) |
|---|---|---|
| SCRIBE v2 [95%] | **0.691 [0.665, 0.718]** | 0.610 [0.577, 0.645] |
| SCRIBE v3.1 [95%] | 0.622 [0.605, 0.638] | 0.546 |
| human rung (v2) | 0.914 | 0.923 |
| sentences shuffled (v2) | 0.404 | 0.344 |
| surface S | 0.933 | 0.929 |
| `k5_cont_bpb` | 1.604 | 1.357 |
| copy32 | 0.199 | 0.242 |
| 13-word copies | 0 of 250 | 0 of 250 |
| mean bytes kept | 439 | 450 |
| wall, peak RAM | 10:21, 11.3 GB | 5:24, 11.4 GB |

- **SCRIBE is higher on web text, not lower.** The web reference is looser: shuffled sentences score 0.404 against Wikipedia's 0.344. The writer's Wikipedia-like sentences are closer to web humans than to Wikipedia humans on SCRIBE's axes.
- **The human continuations cost 0.25 more bits a byte under K5** (`k5_cont_bpb`). K5 has read only Wikipedia. Web words and phrasing are new to it.
- **Reading the samples:** the text stays on Wikipedia topics whatever the prompt ("For the 2004 season, the team played…" after a betting prompt). SCRIBE does not see that it is off the prompt's subject.

### C1

Shards `00000`, `00001` and `00002` (official sha256 match, as for GEN-W's shards), 1,068,952 documents read, 2,139,276,176 bytes after SCRIBE's cleaner (1,047,001 documents). No exact repeat inside a shard; 26 across shards, dropped at the join.

| Overlap rule (all three shards) | Value |
|---|---:|
| documents hitting GEN-W or GEN-A spans | 12,289 (1.2%) |
| documents hitting SCRIBE-W spans | 413,982 (39.5%) |
| documents hitting SCRIBE's Wikipedia spans | 140,641 (13.4%) |
| documents dropped (hits over 10% of their bytes) | 68,753 (6.6%), 74,977,047 bytes |
| sentences cut in other documents | 952,690, 142,895,596 bytes |
| bytes removed, in all | 217,872,643 (10.2%) |
| recheck inside `k5_overlap` | 0 documents with a forbidden span |

- **The SCRIBE-W spans are mostly boilerplate.** 40% of C4 documents share a 32-byte span with the 60 MB of SCRIBE-W text. An example: "Your email address will not be published" is twice in the pack, 14 times in shard `00000`, and never in C1. Cutting such sentences is most of the 10%. It also keeps boilerplate out of K5's future counts.
- **Joined** in shard order and cut at the first document boundary past 1.5 GB: all of `00000` and `00001` and the first part of `00002`.
- **The second check** (`k5_overlap check`, run on the written file in three line-split parts at once; `cat` of the parts equals the file): 764,046 documents (254,443 + 255,374 + 254,229), 0 hold a forbidden span (83,468,757 spans in the set).

| File | Bytes | Documents | sha256 |
|---|---:|---:|---|
| `/mnt/d/cypha_llm/c4prose/c1_train.txt` | 1,500,000,876 | 764,046 | `fcabaded2ed9571f5304159d08f7404a7f5be83d7da1241919844f244c44e362` |

One document per line, UTF-8 (14% of documents hold a non-ASCII character; the typographic marks are ASCII). Manifest: `bench/gen_web/C1_MANIFEST.tsv`, and `c4prose/sha256.txt` on the PC. The work files (about 9 GB) are in `c4prose/work/`; they can be deleted. K5 has not read C1.

### Predictions scored

| # | Prediction | Result |
|---|---|---|
| W1 | Pool ≥ 1,000; rules drop under 15% | **Hit** (12,816; 72 of 572 read, 12.6%) |
| W2 | Rule 2 drops under 3% | Miss (35 of 572, 6.1%) |
| W3 | v2 5 of 5 on S and R | **Miss** (4 and 4) |
| W4 | v3.1 5 of 5 on S and R | Miss (4 and 5) |
| W5 | Human v2 0.90–0.94 on S | **Hit** (0.908) |
| W6 | Default 0.40–0.60, at least 0.05 below GEN-A | Miss (0.691, 0.08 above) |
| W7 | v3.1 below v2 | **Hit** (0.622) |
| W8 | `k5_cont_bpb` at least 0.15 above GEN-A | **Hit** (+0.247) |
| W9 | copy32 within ±0.05 of 0.242; no 13-word copy | **Hit** (0.199; 0) |
| W10 | 11–12 GB; 5–8 minutes | Half (11.3 GB; 10:21, on a PC shared with two other jobs and C1's preparation) |
| W11 | C1 1.0–2.0 GB; overlap removes 2–15%; recheck 0 | **Hit** (1.50 GB; 10.2%; 0) |

### For Odin

GEN-W fails its registered gate by one pair, at the bottom of the ladder. The choices:
1. Accept GEN-W with SCRIBE v2 at 4 of 5, because the failed pair is between two kinds of non-text that every writer is far above.
2. Make v3.1 the score of record on GEN-W (5 of 5 on R, the region every run reads; 4 of 5 on S).
3. Keep GEN-W as reported only, until a SCRIBE version passes it 5 of 5.

Until Odin rules, GEN-W numbers are reported beside GEN-A's and decide nothing.

## Reads (GEN-W test)

None yet. One row per read of `bench/eval/gen_web_test.tsv`.
