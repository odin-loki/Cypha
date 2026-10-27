# QE2 probe set

**Odin Loch** · 2026-10-28 · inputs for [EMBEDDING_GEOMETRY.md](../../docs/EMBEDDING_GEOMETRY.md) §7.1 and QE2–QE3

This is the fixed, hashed set of texts that the geometry teacher (Qwen3-Embedding) and Cypha are both measured on. The large files stay on the build machine in `/root/work/teacher/probe/`. The small ones are copied here. Every file is listed with its SHA-256 below.

## Rules it follows

- **No T-QA question is in it.** The 327 distinct questions of `bench/eval/t_qa_wiki*.tsv` are read only to exclude them. An item is dropped if it equals a T-QA question, or contains one longer than 15 characters, after lower-casing and removing punctuation.
- **No benchmark test split is in it.** STS-B, MRPC, PAWS and SNLI enter from their train and validation splits only. Their test splits stay clean for Cypha's later task scores (EMBEDDING_GEOMETRY §7.2).
- **Held-out enwik8 only.** Wiki text comes from enwik8 bytes after 8,388,608. The first 8 MiB is CyphaLM's training slice.
- **Every item survives the index unchanged.** Items have no tab, newline or backslash, so `--key text` matches the `text` column of `embedtexts`' index exactly.
- **Dev and test.** Each item gets a split from its text: `test` when the first 32 bits of SHA-256(text) are 0 mod 5, else `dev`. That gives 27,304 test items (20.1%). A pair is `test` when either side is.

## Sources

| Source | Exact file | Licence | Used for |
|---|---|---|---|
| enwik8 | `/root/cdata/enwik8`, SHA-256 `2b49720e…24a8`, bytes 8,388,608 to the end | Wikipedia text, CC BY-SA | sentences, titles, words, NP chunks, held_out.txt, word counts |
| Dolly 15k | `/root/cdata/dolly.jsonl` (15,014 lines), SHA-256 `4b4981dc…2a9a` | CC BY-SA 3.0 | instructions, response leads, questions.txt |
| STS-B (GLUE) | `huggingface.co/datasets/nyu-mll/glue` at revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`, `stsb/train-00000-of-00001.parquet` (SHA-256 `bbd93bbb…ed1f`) and `stsb/validation-00000-of-00001.parquet` (`152de7cf…0aa9`) | The GLUE card refers to each source's own licence. STS Benchmark: mixed sources (news, captions, forums); research use | similarity classes, positives with score ≥ 4 |
| MRPC (GLUE) | same repository and revision, `mrpc/train-…parquet` (`61fd4130…f0af`) and `mrpc/validation-…parquet` (`33c007db…f50f`) | Microsoft Research Paraphrase Corpus licence (research use) | paraphrase / not paraphrase |
| PAWS labeled_final | `huggingface.co/datasets/google-research-datasets/paws` at `161ece9501cf0a11f3e48bd356eaa82de46d6a09`, `labeled_final/validation-00000-of-00001.parquet` (`7760d829…2f6d`) | "May be freely used for any purpose"; acknowledge Google LLC | adversarial paraphrase / not paraphrase |
| SNLI | `huggingface.co/datasets/stanfordnlp/snli` at `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b`, `plain_text/validation-00000-of-00001.parquet` (`00f5ed8d…ffe`) | CC BY-SA 4.0 | entailment, neutral, contradiction |
| E1 chunker | `/root/work/e1/pos_tagger tag /root/work/e1/pos_model.bin --chunks` | the repository's own | NP chunks for phrases |

The PAWS and SNLI train files were also downloaded but are not used.

## How it was built

The builder is a scratch file outside the repository: `/root/work/teacher/bin/build_probe.py`, SHA-256 `625cc37f…0398`, seed 20261028. It runs in about 2 min 15 s on one core. What it does:

1. **Clean the wiki text.** Split held-out enwik8 into pages and drop redirects and pages with a colon in the title. Unescape the XML twice (so `&amp;mdash;` becomes —). Remove comments, `<ref>` blocks, `<math>`/`<code>`/`<pre>`/`<gallery>` blocks and other tags. Remove `{{templates}}` and `{| tables |}` from the inside out. Replace `[[target|text]]` by its text and drop namespaced links (Image, Category, interwiki). Drop external links, bold and italic quotes, and every line that starts with a heading, list, indent or table mark. This leaves 7,236 articles with prose.
2. **Split sentences.** Split after `.`, `!` or `?` before a capital or digit, except after common abbreviations and single initials. Keep a sentence when it has 6–40 words, starts with a capital, ends with `.`, `!` or `?`, has balanced brackets and quotes, has no markup character (`[]{}|<>=*#_\~^`), and is at least 75% letters. This gives 307,047 sentences.
3. **Wiki items.** Up to 4 sentences per article (30,000 in all). Then, per article, one adjacent pair (3,000), one pair at least 3 sentences apart (3,000), and the title with the first sentence (2,000 titles of 1–6 words). Then 3,000 random pairs of wiki sentences as the unrelated class.
4. **held_out.txt.** 4,000 other sentences of 8–25 words, none of them in the probe.
5. **Phrases.** The E1 chunker reads 120,000 further held-out sentences. NP chunks lose a leading determiner or possessive. Chunks of 2–5 alphabetic words are kept, the 20,000 most frequent (one per lower-cased form).
6. **Words.** The 20,000 most frequent word types in the cleaned held-out text, each in its most common casing. `word_counts.tsv` holds the counts of every type seen at least twice, for the SIF weights of `occlusion --freq`.
7. **Dolly.** questions.txt takes 2,000 instructions from `open_qa`, `general_qa` and `closed_qa` that end in `?` and have 4–30 words. The probe takes 8,000 other records whose instruction has at most 400 characters (longer ones paste a passage in). Each gives its instruction and its response cut to whole sentences within 400 characters. A "wrong response" class pairs an instruction with another record's response from the same category (2,993 pairs).
8. **Pair datasets.** STS-B pairs go into 5 classes by gold score (0–1 … 4–5), and pairs scoring at least 4 are also positives. MRPC and PAWS give paraphrase / not-paraphrase classes; paraphrases are also positives. SNLI gives three classes; entailments are also positives.
9. **Write.** Items are de-duplicated and shuffled with the seed, so the first 1,000 lines of probe.txt are a fair sample. `timing_1000.txt` is exactly those 1,000 lines (what `embedtexts --limit 1000` reads).

## Files

Token counts are Qwen3 tokens, with the appended `<|endoftext|>` counted.

| File | Rows | Size | SHA-256 | Where |
|---|---|---|---|---|
| `probe.txt` | 135,895 items; 2,455,468 tokens; mean 18.1, longest 281 | 10.5 MB | `253aec50d3bc56c162e9369cb9103bb7347faf88cc9f56ba2ba65bdf0653dd10` | build machine |
| `probe_meta.tsv` (row, kind, source, split) | 135,895 | 4.1 MB | `739a2a5a5cc4532fb809bced0a8428d37a39926cdb164d54abca730edb7883e6` | build machine |
| `pos_pairs.tsv` (anchor, positive, source, split) | 24,240 | 5.8 MB | `e456e9e078bd7b45d2bec71e66b6ac72e84446e4f31f0595063da6f3fbb63b66` | build machine |
| `classes.tsv` (a, b, class, split) | 51,073 | 11.3 MB | `58edacdfaeb367df05a80ef1f65e10ae2a478a49b594cfce135391a5ec037769` | build machine |
| `timing_1000.txt` | 1,000; 18,352 tokens | 79 KB | `b09007d4caaf0504549723140f12c903b5d83f5d0d9a4e74e89af887040ab53a` | here and build machine |
| `held_out.txt` | 4,000; 95,008 tokens | 423 KB | `0e19e2576926f1e6725290fe529da3f7ca4b79b694a17f7a9f6c95acb909ae8b` | here and build machine |
| `questions.txt` | 2,000; 24,869 tokens | 104 KB | `84df1fd48338b66e902faffcb0dc6395d254016a765d6072d20a98e0e7f67191` | here and build machine |
| `word_counts.tsv` (word, count) | 94,025 | 1.1 MB | `c7cda4afcf27964e0fd173064e115174e598ca518aab8d1b7d927abc162f3c4d` | here and build machine |
| `build_summary.json` | | 1.3 KB | `0cd5bf9f1dd789227a1a167517348eb949fe8056d019f790f8553f9838bc3b94` | here and build machine |

**probe.txt by kind:**

| Kind | Source | Items |
|---|---|---|
| word | held-out enwik8 frequency | 19,832 |
| phrase | E1 NP chunks | 19,752 |
| title | held-out enwik8 | 2,000 |
| sentence | held-out enwik8 | 38,081 |
| sentence | STS-B | 13,194 |
| sentence | SNLI | 12,917 |
| sentence | MRPC | 7,815 |
| sentence | PAWS | 6,476 |
| instruction | Dolly | 7,924 |
| response lead | Dolly | 7,904 |

Words and phrases fall a little short of 20,000 each because an item that already exists keeps its first kind.

**pos_pairs.tsv by source:** Dolly instruction → response 7,985; PAWS paraphrase 3,527; SNLI entailment 3,323; wiki adjacent 3,000; MRPC paraphrase 2,753; wiki title → lead 2,000; STS-B ≥ 4 1,652.

**classes.tsv by class:** dolly_instruction_response 7,991; paws_not_paraphrase 4,461; paws_paraphrase 3,532; snli_entailment 3,325; snli_contradiction 3,278; snli_neutral 3,235; wiki_adjacent, wiki_same_article and unrelated 3,000 each; dolly_wrong_response 2,993; mrpc_paraphrase 2,753; wiki_title_lead 2,000; stsb_3to4 1,953; stsb_4to5 1,669; mrpc_not_paraphrase 1,323; stsb_2to3 1,255; stsb_0to1 1,235; stsb_1to2 1,135. (Counts before de-duplication of repeated pairs; the file has 51,073 rows.)

## Nested subsets

Runs that cannot take the full set use these. They keep probe.txt's order and text exactly, so `--key text` still matches. Both are made by `/root/work/teacher/bin/pilot_subset.py` (SHA-256 `cb9ad356…aad5`, seed 20261028).

| Subset | Rule | Items | Qwen3 tokens | Pairs |
|---|---|---|---|---|
| `probe_pilot.txt` (0.6B pilot) | every word, phrase and title; both sides of up to 600 seeded pairs per class; the first 6,000 other held-out wiki sentences | 67,379 | 745,914 | `classes_pilot.tsv` 10,800; `pos_pairs_pilot.tsv` 4,895 |
| `probe_s8b.txt` (8B subset, inside the pilot) | the first 3,000 words and phrases and all 2,000 titles of the pilot; 50 pairs per class drawn from the pilot's pairs; 500 more pilot wiki sentences | 10,239 | 82,063 | `classes_s8b.tsv` 900; `pos_pairs_s8b.tsv` 391 |

| File | SHA-256 | Where |
|---|---|---|
| `probe_pilot.txt` | `01c8b137902be15a303baa22fd9227e0f4db3d934df22e768b9ea15f1023429c` | build machine (3.1 MB) |
| `classes_pilot.tsv` | `a8c308c3ec7c727477e3e697156ec552652b502b4ee44c53a0eb1a10dd318c2b` | build machine (2.2 MB) |
| `pos_pairs_pilot.tsv` | `6a6f8ae4d33ac91f99ab398f6c5a9d6f45b3c70697fa4794de170bb981e9d792` | build machine (1.1 MB) |
| `probe_s8b.txt` | `2e530cf876a37faf9316e21901133f17f85a7a67c0e2616cb1eb533dd0ca3237` | here and build machine |
| `classes_s8b.tsv` | `1e297b2d7faa2d23adb4d8ff0653ca4bb154756ab752ec02de91e8a193ad474e` | here and build machine |
| `pos_pairs_s8b.tsv` | `4b722c6a7db9a2799854d64b4fefe3d952b8c6a74a0108f34eb59b3721c4ca2c` | here and build machine |

```bash
python -I /root/work/teacher/bin/pilot_subset.py /root/work/teacher/probe 600 6000                       # the pilot
python -I /root/work/teacher/bin/pilot_subset.py /root/work/teacher/probe 50 500 20261028 3000 s8b pilot  # the 8B subset
```

No teacher vectors exist for any of these yet. Inference moved to Odin's PC on 2026-10-28; the runbook is in [QWEN3_EMBEDDING_DISSECTION.md](../../docs/reports/QWEN3_EMBEDDING_DISSECTION.md) §9.

## Reproduce

```bash
python -I /root/work/teacher/bin/build_probe.py /root/work/teacher/probe /root/work/teacher/data \
    /home/user/Cypha /root/work/e1/pos_tagger /root/work/e1/pos_model.bin
cd /root/work/teacher/probe && sha256sum -c SHA256SUMS
```

The data directory holds the parquet files above, fetched by their revision URLs. The builder needs Python 3.13 with pyarrow (26.0.0 here). A second build on 2026-10-28 gave the same SHA-256 for all nine files.
