# Zenodo upload checklist

**Odin Loch** · 2026-11-04 · version 2.0 (the SCRIBE v3 paper)

Version 2.0 is a **new version of the existing Zenodo record**, not a new upload. That keeps the concept DOI, so links to "SCRIBE, all versions" keep working, and version 1.0 stays online with its own DOI.

1. Run `paper/scribe/build.sh` (needs pandoc and Playwright's Chromium), then `paper/scribe/bundle.sh`. This gives three files in `paper/scribe/`: `SCRIBE_paper.pdf`, `SCRIBE_code_v2.0.zip` and `SCRIBE_results_v2.0.zip`.
2. On zenodo.org, open the version 1.0 record: **DOI `10.5281/zenodo.XXXXXXX` ← Odin: fill in the version 1.0 DOI; it is not in the repository.** Click **"New version"** (top right of the record page). Do not use "New upload".
3. In the new-version draft, remove the three version 1.0 files (`SCRIBE_paper.pdf`, `SCRIBE_code_v1.0.zip`, `SCRIBE_results_v1.0.zip`; Zenodo keeps them on the 1.0 record). Upload `SCRIBE_paper.pdf` first, so it is the preview file, then the two version 2.0 zips.
4. Fill the form from `zenodo.json`: title; description (the abstract); keywords; version **2.0**; publication date; notes. Creator, licence (CC BY 4.0 for the record; the code inside the code zip is GNU AGPL v3, as its `LICENSE` and `README.md` say), upload type (Publication, Preprint) and language carry over from version 1.0; check them.
5. Optional: under "Related works", add "Is new version of" → the version 1.0 DOI.
6. Publish. Version 2.0 gets its own DOI under the same concept DOI.

The version 1.0 files are kept in `paper/scribe/v1.0/` (paper source, PDF, both zips, metadata); `bundle.sh v1.0` rebuilds its zips.

Not in the uploads, on purpose: the evaluated texts themselves (WebText, Gutenberg, WritingPrompts, HANNA, OpenMEVA, MAUVE, Dolly and K5's samples' prompts belong to their owners or to the Cypha repository). `scripts/scribe_eval/data.py` in the code zip downloads and prepares every public dataset, with checksums, and `scripts/scribe_eval/runs/v3_paper.sh` reruns every new read of version 2.0.
