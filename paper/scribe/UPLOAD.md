# Zenodo upload checklist

**Odin Loch** · 2026-10-28

1. Run `paper/scribe/build.sh` (needs pandoc and Playwright's Chromium), then `paper/scribe/bundle.sh`. This gives three files in `paper/scribe/`: `SCRIBE_paper.pdf`, `SCRIBE_code_v1.0.zip` and `SCRIBE_results_v1.0.zip`.
2. On zenodo.org, choose New upload. Upload `SCRIBE_paper.pdf` first, so it is the preview file, then the two zips.
3. Fill the form from `zenodo.json`: upload type Publication, Preprint; title; creator Loch, Odin; description (the abstract); keywords; licence CC BY 4.0 for the record (the code inside the code zip is GNU AGPL v3, as its `LICENSE` and `README.md` say); version 1.0; notes.
4. Publish. The DOI is permanent; later versions get new DOIs under the same concept DOI.

Not in the uploads, on purpose: the evaluated texts themselves (WebText, Gutenberg, WritingPrompts, HANNA, OpenMEVA, MAUVE and Dolly belong to their owners). `scripts/scribe_eval/data.py` in the code zip downloads and prepares every one of them, with checksums.
