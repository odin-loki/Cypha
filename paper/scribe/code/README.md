# SCRIBE: code

**Odin Loch** · 4 November 2026 · version 2.0

This archive holds all the code behind the paper *SCRIBE v3: Three Count-Based Judges of Generated Text* (version 2.0; version 1.0 was *SCRIBE: Scoring Generated Text Against Human Text With Counts Alone*, on SCRIBE v2). The result tables are in the companion archive `SCRIBE_results_v2.0.zip`.

## Contents

| Path | What it is | Paper |
|---|---|---|
| `native/include/cypha/eval/scribe.hpp` | The metric: SCRIBE v1, v2, v3 (SURPRISE, MATCH, their mean) and v3.1 (SURPRISE 3.1, MATCH 3.1, the geometric mean), a header-only C++20 library, standard library only | §4–§7, Appendix A |
| `native/tools/scribe.cpp` | The command-line tool: `docs`, `prompts`, `samples`, `prepare`, `windows`, `ladder`, `attacks` (with `--heldout`), `dedup`, `score`, `selftest` (119 checks); profiles `v1`, `v2`, `v2p`, `v3`, `v31` | §5–§8 |
| `native/tools/ppmi_svd.cpp` | Builds the PPMI-SVD word vectors from a pack (v2's coherence axis) | §4 |
| `scripts/scribe_eval/data.py` | Downloads and converts every dataset; writes a manifest with sources, licences and SHA-256 hashes | §8 |
| `scripts/scribe_eval/corrupt.py`, `attacks.py`, `prompted.py`, `qa_ladder.py` | The held-out corruptions, v2's attack set, the prompted ladder and the Q/A ladder | §9 |
| `scripts/scribe_eval/generate.py` | n-gram model-ladder generation (and the unrun Pythia and GPT-2 ladders) | §9.4 |
| `scripts/scribe_eval/baselines.py`, `judge.py` | Version 1.0's baseline metrics; the LLM judge (written, not run) | version 1.0 |
| `scripts/scribe_eval/meta.py`, `report.py` | Meta-evaluation statistics; version 1.0's results driver | §8 |
| `scripts/scribe_eval/v3_report.py` | Version 2.0's results driver (v2, v3 and v3.1 side by side) | §9 |
| `scripts/scribe_eval/runs/v3_paper.sh` | Every new read of version 2.0, one step per read, as registered | §8, §9 |
| `scripts/scribe_eval/runs/` | The other run scripts as used, including version 1.0's design search | |
| `docs/SCRIBE_PAPER_PLAN.md` | Version 1.0's study plan, with its splits and twenty predictions | |
| `docs/reports/SCRIBE_V0.md` … `SCRIBE_V31.md` | Each version's freeze record, design log and fresh verdict | §3, §5, §6 |
| `docs/reports/SCRIBE_V3_PAPER.md` | Version 2.0's registration (reads and predictions, committed before the reads) and results | §8, §9.12 |
| `docs/reports/K5_UPGRADES.md`, `K5_SCALE.md` | The K5 experiments that raised the co-occurrence question | §3.1, §9.10 |
| `paper/` | The paper's Markdown source, style sheet and build script | |

## Build

```
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

or, without CMake:

```
g++ -O2 -std=c++20 -Inative/include native/tools/scribe.cpp -o scribe
g++ -O2 -std=c++20 native/tools/ppmi_svd.cpp -o ppmi_svd
./scribe selftest        # 119 checks
```

The Python scripts need Python 3.10 or later and the packages in `requirements.txt`. Only `baselines.py`, `judge.py` and `generate.py hf` need PyTorch and Hugging Face models; version 2.0's runs need only numpy, pandas and scipy.

## A first run

```
python -I scripts/scribe_eval/data.py --data DATA --dl DOWNLOADS web
./ppmi_svd pack.xml none none - vec --no-ri --no-ablation          # word vectors from the pack
```

`ppmi_svd` reads MediaWiki XML. For a JSONL pack, wrap each document as `<page><revision><text>…</text></revision></page>` (HTML-escaped) first (`scripts/scribe_eval/runs/v3_paper_tools/wrap.py`).

```
./scribe score --profile v31 --pack DATA/web/pack.jsonl --ref DATA/web/ref.jsonl --vectors vec --in texts.jsonl --out scores.jsonl
```

Each input line is a JSON object with `id` and `text` (and optionally `prompt`: with `--profile v31`, MATCH's echo and the v3.1 prompt statistics use it). Each output line has `scribe` (the profile's headline: `scribe_v31` for `v31`), `scribe_v2`, `scribe_v3`, `surprise`, `match`, `min3`, `disagree`, `surprise31`, `match31`, every judge's deviance terms and every raw statistic. The header of `native/tools/scribe.cpp` documents every command and option.

## Licence

The code is licensed under the GNU Affero General Public License, version 3 (`LICENSE`). The paper is licensed under CC BY 4.0.
