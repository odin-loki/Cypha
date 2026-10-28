# Zenodo upload checklist

**Odin Loch** · 2026-10-28

1. Run `paper/scribe/build.sh` (needs pandoc and Playwright's Chromium) and `paper/scribe/bundle.sh`. The bundle is `paper/scribe/SCRIBE_zenodo_v1.0.zip`.
2. On zenodo.org, choose New upload. Upload `SCRIBE_paper.pdf` first, so it is the preview file, then the bundle zip.
3. Fill the form from `zenodo.json`: upload type Publication, Preprint; title; creator Loch, Odin; description (the abstract in the file); keywords; licence CC BY 4.0; version 1.0; related identifier github.com/odin-loki/Cypha, "is supplemented by", software.
4. Optional: link the Cypha GitHub repository to Zenodo and make a release, so the code gets its own DOI; then add it as a related identifier.
5. Publish. The DOI is permanent; later versions get new DOIs under the same concept DOI.

What is not in the bundle, on purpose: the evaluated texts themselves (WebText, Gutenberg, WritingPrompts, HANNA, OpenMEVA, MAUVE and Dolly belong to their owners). `scripts/scribe_eval/data.py` downloads and prepares every one of them, with pinned revisions and checksums.
