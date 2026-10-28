# SCRIBE: code

**Odin Loch** · 28 October 2026 · version 1.0

This archive holds all the code behind the paper *SCRIBE: Scoring Generated Text Against Human Text With Counts Alone*. The result tables are in the companion archive `SCRIBE_results_v1.0.zip`.

## Contents

| Path | What it is | Paper |
|---|---|---|
| `native/include/cypha/eval/scribe.hpp` | The metric: SCRIBE v2 (and v1), a header-only C++20 library, standard library only | §3, Appendix A |
| `native/tools/scribe.cpp` | The command-line tool: `docs`, `prompts`, `samples`, `prepare`, `windows`, `ladder`, `score`, `selftest` | §3.7 |
| `native/tools/ppmi_svd.cpp` | Builds the PPMI-SVD word vectors from a pack | §3.2 |
| `scripts/scribe_eval/data.py` | Downloads and converts every dataset; writes a manifest with sources, licences and SHA-256 hashes | §5.1 |
| `scripts/scribe_eval/corrupt.py`, `attacks.py`, `prompted.py`, `qa_ladder.py` | Build the held-out corruptions, the attacks, the prompted ladder and the Q/A ladder | §5.2 |
| `scripts/scribe_eval/generate.py` | n-gram model-ladder generation (and the unrun Pythia and GPT-2 ladders) | §5.2 |
| `scripts/scribe_eval/baselines.py`, `judge.py` | Baseline metrics; the LLM judge (written, not run) | §5.3 |
| `scripts/scribe_eval/meta.py`, `report.py` | Meta-evaluation statistics; the driver that writes every table of Section 6 | §5.4, §6 |
| `scripts/scribe_eval/runs/` | The run scripts as used, including the design search on development ladders | §4, §6 |
| `docs/SCRIBE_PAPER_PLAN.md` | The study plan, with the splits and the twenty predictions registered before the first v2 run | §5, §6.9 |
| `docs/reports/SCRIBE_V2.md` | The freeze record, design log, fresh verdict and post-freeze review | §4, §6.10 |
| `paper/` | The paper's Markdown source, style sheet and build script | |

## Build

```
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

or, without CMake:

```
g++ -O2 -std=c++20 -Inative/include native/tools/scribe.cpp -o scribe
g++ -O2 -std=c++20 native/tools/ppmi_svd.cpp -o ppmi_svd
./scribe selftest        # 76 checks
```

The Python scripts need Python 3.10 or later and the packages in `requirements.txt`. Only `baselines.py`, `judge.py` and `generate.py` need PyTorch and Hugging Face models.

## A first run

```
python -I scripts/scribe_eval/data.py --data DATA --dl DOWNLOADS web
./ppmi_svd pack.xml none none - vec --no-ri --no-ablation          # word vectors from the pack
```

`ppmi_svd` reads MediaWiki XML. For a JSONL pack, wrap each document as `<page><revision><text>…</text></revision></page>` (HTML-escaped) first.

```
./scribe score --profile v2 --pack DATA/web/pack.jsonl --ref DATA/web/ref.jsonl --vectors vec --in texts.jsonl --out scores.jsonl
```

Each input line is a JSON object with `id` and `text` (and `prompt` for the prompted profile, which also needs `--ref-prompts`). Each output line has the SCRIBE v2 score, its deviance parts and the z-score of every statistic. The header of `native/tools/scribe.cpp` documents every command and option.

## Licence

The code is licensed under the GNU Affero General Public License, version 3 (`LICENSE`). The paper is licensed under CC BY 4.0.
