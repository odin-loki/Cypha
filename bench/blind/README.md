# K5 blind reading (2026-11-04)

**Odin Loch** · 2026-11-04

Odin rates 30 continuations from 1 to 10 on a private page (claude.ai artifact "K5 Blind Reading"; ratings saved in its store, collection `ratings`, one document per item, e.g. `p01-A`).

- **What is rated:** 10 GEN-A dev prompts (dev 6, 55, 60, 85, 98, 103, 132, 157, 168, 197; drawn with seed 20261104, leaving out the 8 prompts of the first sheet). Each prompt has three continuations in random order: the real text that followed, the old K5 (nucleus 0.9, no gate) and the new K5 (nucleus 0.8, 256 classes, the 8-candidate prompt-association gate), all cut at a word boundary near 460 characters.
- **Files:**
  - `k5_rating_items.json`: what the page shows.
  - `k5_rating_key.b64`: which continuation is which, base64-encoded so it isn't read by accident before the ratings are in (`base64 -d k5_rating_key.b64`).
- **The question:** does the gate's gain on SCRIBE v2 (+0.145), which SCRIBE v3's new judges did not confirm (+0.011), show in a human reading? Compare new K5 with old K5 per prompt (paired), with the human text as the ceiling.
