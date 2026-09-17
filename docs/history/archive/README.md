# Historical source archive

Preserved source trees for every Cypha version that predates this repository. **Nothing
here is on the product path.** None of it is built, tested, linted or imported; it is kept
as evidence, and it is the only surviving record of the project between February 2025 and
March 2026.

For what these files *mean*, start at [`../README.md`](../README.md). This page only
describes the tree.

> **Read-only by policy.** Do not fix, reformat, modernise or lint anything in this
> directory. Line endings, encodings, typos and dead code are part of the record — including
> the misspelled `cypa-v7-generation`, which is how the directory was named in the source
> archive. Corrections belong in the prose docs one level up.

---

## Layout

| Path | Original name in zip | What it is |
|---|---|---|
| [`prototypes/`](prototypes/) | `Prototypes` | Brain Model and Cell AI v2/v3 — design dialogues and math models, no code |
| [`cypha-v1/`](cypha-v1/) | `Cypha V1` | first use of the name Cypha; HRNA and event-driven mathematics, no code |
| [`cypha-v2/`](cypha-v2/) | `Cypha v2` | the 9,404-line monolith; the only version using `torch`, `ray` and `scipy` |
| [`cypha-v3/`](cypha-v3/) | `Cypha v3` | the great simplification — frameworks dropped, pure `numpy` |
| [`cypha-v4/`](cypha-v4/) | `Cypha v4` | adds `benchmark_suite.py` and `verify_thinking.py` |
| [`cypha-v5/`](cypha-v5/) | `Cypha v5` | real-dataset tooling: `download.py`, `convert.py`, `synthetic_benchmark.py` |
| [`cypha-v6/`](cypha-v6/) | `Cypha v6` | largest tree; game benchmarks and the four profilers under `Profiling/` |
| [`cypa-v7-generation/`](cypa-v7-generation/) | `Cypa v7 Generation` *(sic)* | the generation branch; 55 classes, a different program from v8 |
| [`cypha-v8/`](cypha-v8/) | `Cypha v8` | CyphaDIF: 1,412 lines, 9 classes, plus 17 mathematical papers |
| [`root-monolith/`](root-monolith/) | archive root `Cypha.py` | **the porting source for the native product** — see [`../PYTHON_TO_CPP_BRIDGE.md`](../PYTHON_TO_CPP_BRIDGE.md) |
| [`cypha-vchatgpt/`](cypha-vchatgpt/) | `Cypha vChatGPT` | side branch: the only modular package, only test suite, only GUI, only `transformers` |
| [`cypha-vpattern-matching/`](cypha-vpattern-matching/) | `Cypha vPattern Matching` | side branch: the only one labelled "production" |
| [`cypha-encoder/`](cypha-encoder/) | `Cypha Encoder` | the encoder extracted as a standalone unit |
| [`big-data/`](big-data/) | `Big Data` | dataset acquisition tooling |
| [`datasets/`](datasets/) | `Cypha Classification Settings.jsonl` | 27,524-run sweep, gzipped — analysed in [`../SWEEP_ANALYSIS.md`](../SWEEP_ANALYSIS.md) |
| [`side-quests/retdec-upgrades5/`](side-quests/retdec-upgrades5/) | `retdec_upgrades5.zip` | RetDec decompiler optimiser patches, dated the same day as v8 |

[`MANIFEST.md`](MANIFEST.md) is the byte-exact inventory: every file, its original path, its
size and its `sha256`.

---

## How this was integrated

The source was a 6,108,467-byte zip containing 164 files.

**Excluded (30 files).** Every `__pycache__/*.pyc` — build artifacts, already covered by the
repository `.gitignore`, and reconstructible from the `.py` files beside them.

**Copied verbatim (132 files).** Each was verified `sha256`-identical to its source after
staging; the checksums are recorded in [`MANIFEST.md`](MANIFEST.md).

**Transformed (2 → 10 files).**

- `Cypha Classification Settings.jsonl`, 20,243,385 bytes, is stored as
  `datasets/cypha_classification_settings.jsonl.gz` — 445,917 bytes, a 22.3× reduction.
  Committing 20 MB of JSONL to make it marginally easier to `grep` was not a good trade;
  `zcat` or `gzip.open` recovers the original exactly.
- `retdec_upgrades5.zip` is expanded to its 9 constituent files, so the patches are
  readable and searchable in place rather than opaque inside a nested archive.

**Derived (31 files).** Every `.docx` was converted to Markdown with a stdlib
(`zipfile` + `xml.etree`) extractor and written beside its original. The prose in these
documents — seventeen mathematical papers in `cypha-v8/` alone — is a large part of the
archive's value, and it is useless if it can only be opened by a word processor. **The
`.docx` originals are retained and remain authoritative**; the `.md` files are a convenience
and may drop figures, equation formatting and styling.

Total: **173 files, 12 MB.**

---

## Caveat on timestamps

Per-file modification times in the zip were all rewritten to the moment of zipping. They
are meaningless and **must not be cited as evidence of when anything was written**.

Directory mtimes survived and are the only usable stamps:

| Directory | mtime |
|---|---|
| `Prototypes` | 2025-02-26 |
| `Cypha V1` | 2025-03-04 |
| `Big Data`, `Cypha Encoder`, `Cypha v2`–`v5`, `Cypha vChatGPT`, `Cypha vPattern Matching` | 2026-02-21 20:35 |
| `Cypha v6` | 2026-02-27 |
| `Cypa v7 Generation` | 2026-03-06 |
| `Cypha v8` | 2026-03-11 |
| archive root | 2026-03-14 |

The 2026-02-21 20:35 cluster is a single bulk copy: eight directories sharing one timestamp
to the minute were not authored simultaneously. Their true order is reconstructed from
internal evidence in [`../TIMELINE.md`](../TIMELINE.md).
