# O7: Huffman binarisation in the full mix

**Odin Loch** · 2026-11-12 · branch `wip/opt-huffman` · GENERATION_PLAN §13.41 O7

**Outcome first.** Not kept. The Huffman code makes 5.1 decisions a byte instead of 8 and cuts the full mix's own time 15.6%, but `context8k_bpc` rises from 1.4962 to **1.5069 (+0.0107)** on the full run (+0.018 on the small one). Both parts of the keep rule fail (+0.002 and −20%). `huff=1` stays a setting, off by default; with it off every output equals main's.

## Registered before any run (2026-11-12)

§13.41 registered O7: "code each byte as a path in a Huffman tree of byte frequencies, so the bitwise mix makes about 4.6 binary decisions per text byte instead of 8. This changes the numbers. Kept if `context8k_bpc` is no worse than +0.002 and time falls at least 20%." This section adds the details §13.41 left open. It is committed and pushed before any run of the new code.

**What changes** (`native/include/cypha/online/full_mixer.hpp`, `meta_mixer.hpp`, and 4 lines of `cyphalm_trie.cpp`):
- A new setting `huff` in `--full-mix-params` (0 by default). With `huff=0` nothing changes: the code is the byte's 8 bits, most significant first.
- With `huff=1` the full mix builds a Huffman code from the counts of the training bytes (the bytes before the holdout, which `context8k` already reads). Each count gets +1, so every byte has a code. If a code would be longer than 24 decisions, the counts are halved until none is. Ties go to the lower id, so the tree depends on the counts alone.
- Every bitwise part then walks the true byte's code instead of its 8 bits: the bitwise mixers A and R, the four SSE stages, the AB mixer, the copy input, and the meta mix (M1, M2A, M2G's path probabilities, the masters, M3L and the SSE after them).
- A decision's probability under each input distribution is the mass of its 1-branch over the node's mass, as for the 8 bits.
- **Contexts.** Where the old code used the partial byte c0 (1–255), the new code uses the node's number (1–255, breadth first). Where it used the bit position j (0–7), the new code uses the depth, capped at 7. So every table keeps its size.
- P(byte) is the product of the decisions along its code, so it stays exactly normalised.
- `full-mix-replay` and `meta-mix-replay` keep the 8 bits (the dump does not carry the training counts).
- `context8k` prints one new line with `huff=1`: the decisions made per holdout byte and the code's mean length over the training bytes.

**Arms:** H0, main's default; H1, the same with `--full-mix-params huff=1`.

**Checks first:** with `huff=0` (the default), the new binary's stdout equals main's (`cmp`) on the small `context8k` default, `--no-meta-mix` and `--no-full-mix`.

**Measures:**
- Bits per byte: the small setup (`--train 1048576 --sample 1179648`): `context8k_bpc` (the meta mix's M1Gs), the full mix's ABs and the second-half lines; then **one full default `context8k` with `huff=1`** under `flock ~/.cypha_full.lock`. H0's full value is 1.4962 (main, reproduced by O5).
- **Time:** the full mix's own seconds (the sum of its six parts on the "mixer s:" line; it leaves out waiting for producers), with the meta mix on, as the default runs. H0 and H1 are started together on the small setup, 3 rounds; the median change is scored. Wall time and peak RSS are reported beside it. The full runs cannot start together (the lock), so their times are reported, not scored.
- Decisions per byte.

**Keep rule** (§13.41, read this way): kept if the full run's `context8k_bpc` is at most 1.4982 (1.4962 + 0.002) **and** the full mix's own time falls at least 20% (small runs, median of 3). If kept, `huff=1` becomes the default on a merge the lead does; until then it is a setting, off by default.

**Predictions** (§13.41's stand; mine added now):
- §13.41: about 4.6 decisions per text byte. Mine: 5.0–5.3. The order-0 entropy of enwik8's text with its markup is about 5.1 bits.
- §13.41: kept (bpc within +0.002, time −20%). Mine: the full mix's own time falls only 15–20%. The per-byte parts (the chain, the final layer, the geometric mixers B and M2G) do not shrink; only the bit loop does, by about 5.1/8. And bpc rises by 0.000 to +0.005: a node number mixes bytes that share a code prefix, not a bit pattern, which the order-1 and order-2 SSE contexts were tuned for. So I expect it **not kept**.

**Machine:** threads as O5 ran them (the LSTM 4, the n-gram 4, the defaults); no run starts while `free -g` shows under 10 GB available; the full run takes `flock ~/.cypha_full.lock`.

## Results (2026-11-12)

Old is main 9cbd277 (`~/j3/b_main`); new is 9797e1f (`~/j3/b_o7`). Same compiler and flags (GCC 13.3, CMake Release). The PC's load average was 15–30 (other sessions).

**Checks: hit.** With `huff=0` the new binary's stdout equals main's (`cmp`) on the small default (3 rounds), `--no-meta-mix` and `--no-full-mix`. H1's output is the same in all 3 rounds.

**The code.** The Huffman code of the first 1 MB has a mean length of 5.10 decisions over the training bytes. On the holdout the bitwise mix made **5.12 decisions per byte**, against 8.

**Bits per byte, small setup** (second half; the arms that use the code change, the others do not):

| Arm | H0 (main) | H1 (`huff=1`) | Change |
|---|---:|---:|---:|
| F, Fn (the final layer; no code) | 1.8384, 1.8380 | same | 0 |
| B (the geometric mixer; no code) | 1.8008 | 1.8008 | 0 |
| Fs (Fn, then SSE) | 1.8098 | 1.8202 | +0.0104 |
| A, As (the bitwise mixer) | 1.7835, 1.7753 | 1.8003, 1.7937 | +0.017, +0.018 |
| Bs | 1.7801 | 1.7884 | +0.0083 |
| ABs (the full mix's headline) | 1.7633 | 1.7776 | +0.0143 |
| **M1Gs = `context8k_bpc`** | **1.7501** | **1.7684** | **+0.0183** |

Every arm that walks the code is worse. Even SSE on the final layer's own distribution (Fs) loses 0.010. So the loss is in the contexts, not only in the mixers' learning: a node of the Huffman tree joins bytes by frequency, while a partial byte joins bytes that share their high bits (letters with letters, digits with digits). The SSE tables keyed by the partial byte and the previous bytes lose that.

**Time** (the full mix's own seconds, the sum of its six parts; 3 runs started together each round):

| Round | H0 old | H0 new (`huff=0`) | H1 | H1 vs old |
|---|---:|---:|---:|---:|
| 1 | 59.8 | 59.0 | 50.5 | −15.6% |
| 2 | 91.6 | 92.9 | 76.1 | −16.9% |
| 3 | 66.3 | 65.2 | 56.1 | −15.4% |
| **Median** | | | | **−15.6%** |

- The bit loop fell by a third (28.5 → 18.7 s in round 1). The path probabilities ("bits") rose 2.1 → 2.7 s: each input's distribution is read through the code's leaf order. The per-byte parts (the chain, the final layer, the geometric mixers, learning) did not move.
- Wall time −0.3% to −1.5%; peak RSS the same (1,338–1,341 MB). The small run does not wait on the full mix at the end.

**The full run** (`--full-mix-params huff=1`, default threads, under the lock, 12:02–12:28): **`context8k_bpc` 1.5069**, against main's 1.4962: **+0.0107**. The full mix's ABs 1.5133 against 1.5051 (+0.0082). 5.18 decisions per holdout byte (the code's mean length over the training bytes 5.12). Wall 26:19, peak RSS 1,565 MB. The full mix's own time was 344 s; the O8 full run just before it (main's code, SSE2 meta kernel, a different binary) took 377 s. Those ran one after the other, so they are not scored.

## Predictions, scored

| Prediction | Result | Score |
|---|---|---|
| §13.41: about 4.6 decisions per text byte | 5.12 | **miss** |
| Mine: 5.0–5.3 | 5.12 | **hit** |
| §13.41: `context8k_bpc` no worse than +0.002 | small +0.0183; full +0.0107 (1.5069) | **miss** |
| §13.41: time down at least 20% | the full mix −15.6% (median); wall about −1% | **miss** |
| Mine: time −15% to −20% | −15.6% | **hit** |
| Mine: bpc +0.000 to +0.005 | +0.018 (small), +0.011 (full) | **miss** (worse than I thought) |
| Mine: not kept | not kept | **hit** |

**Not kept.** Both parts of the rule fail. `huff` stays a setting, off by default; with it off every output is main's. Nothing goes to main.

**What might work instead** (not tried): keep the 8-bit code, and skip the decisions an expert is sure of. Or use a code that keeps letters with letters (a tree grown by byte class first, then by frequency). Both keep the SSE contexts' meaning.

## Files

- Code: `native/include/cypha/online/full_mixer.hpp` (the `Code` struct, the bit loop on the code), `meta_mixer.hpp` (the meta mix on the same code), `native/tools/lm/cyphalm_trie.cpp` (4 lines: the training counts when `huff=1`).
- Runs on Odin's PC, `/mnt/d/cypha_llm/opt/o7/`: `H{1,2,3}_{old,h0,h1}` (small rounds; `.out`, `.err`, `.time`), `nmm_*`, `nfm_*` (checks), `small.log`; `full/` (the full run, `full.log`). Scripts: `~/j3/o7small.sh`, `~/j3/full.sh`, `~/j3/k8.sh`.
