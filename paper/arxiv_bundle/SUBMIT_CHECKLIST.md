# CyphaLM arXiv bundle (2026-07-18)

## Contents
- `CyphaLM_paper.md` / `.html` / `.pdf` — camera-ready draft (canonical BPC pin 2.873)
- `abstract.txt` — plain-text abstract for the arXiv form
- `metadata.yaml` — title, author, suggested categories, license note
- `RESULTS_ATTEST.md` — D10/H15/lock provenance snapshot (not in the PDF body)
- `native_fig_*.png/json` — native measurement figures
- `FIGURES_README.md` — figure provenance

## Submit checklist (human)
1. Choose venue (arXiv **cs.LG** primary; optional cs.CL / stat.ML — see `metadata.yaml`).
2. Log in at https://arxiv.org/submit and create a new submission.
3. Upload `CyphaLM_paper.pdf` (or source MD + figures if required). Rebuild PDF if needed: `pandoc CyphaLM_paper.md -o CyphaLM_paper.pdf --pdf-engine=xelatex`.
4. Paste title/authors/abstract from `metadata.yaml` + `abstract.txt`.
5. Confirm license (recommend CC-BY-4.0 or arXiv default).
6. Preview and submit (agent cannot complete the authenticated upload).

## Honest claims to keep
- Hybrid pin **2.873 BPC** @ 300k WikiText-2
- Shared-model CL open; zero forgetting = per-task isolation (D16F)
- GRIA blend currently LSTM-dominated (~99.6%)
