# O7: Huffman binarisation in the full mix

**Odin Loch** · 2026-11-12 · branch `wip/opt-huffman` · GENERATION_PLAN §13.41 O7

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
