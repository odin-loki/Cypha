# CyphaLM arXiv bundle (2026-08-16)

## Contents
- `CyphaLM_paper.md` / `.html` / `.pdf` — camera-ready draft (living pin **2.664** BPC; L1 2.873 is historical)
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
- Living hybrid pin **2.664 BPC** (L2 + Wave2 BPTT @ 300k WikiText-2). Cite `bench/BASELINE_LOCK.json` + `MODEL_CARD.md` + `RESULTS_ATTEST.md`. Prior L1 **2.873** is historical.
- Shared-model CL open; zero forgetting = per-task isolation (D16F)
- GRIA blend currently LSTM-dominated (~99.6%)
- Training is CPU; optional CUDA is infer-only (GPU train is slower here)
