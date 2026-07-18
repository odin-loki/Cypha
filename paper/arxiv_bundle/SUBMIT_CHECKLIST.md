# CyphaLM arXiv bundle (2026-07-18)

## Contents
- `CyphaLM_paper.md` — camera-ready markdown draft (canonical BPC pin 2.873)
- `native_fig_*.png/json` — native measurement figures
- `FIGURES_README.md` — figure provenance

## Submit checklist (human)
1. Choose venue (arXiv cs.LG / workshop / journal).
2. Convert markdown → PDF if required: `pandoc CyphaLM_paper.md -o CyphaLM_paper.pdf --pdf-engine=pdflatex` (needs a TeX install; HTML already in this bundle).
3. Upload PDF + figures; set metadata from paper title/abstract.
4. Confirm license (recommend CC-BY-4.0 or arXiv default).

## Honest claims to keep
- Hybrid pin **2.873 BPC** @ 300k WikiText-2
- Shared-model CL open; zero forgetting = per-task isolation (D16F)
- GRIA blend currently LSTM-dominated (~99.6%)
