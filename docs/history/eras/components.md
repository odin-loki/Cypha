# Loose components and side quests

Four items in the archive that are not versions of Cypha: a standalone encoder, a corrupt
dataset tool, a 27,524-run experiment, and a set of decompiler patches.

---

## `cypha-encoder/` — the standalone encoder

**[`../archive/cypha-encoder/Cypha_Encoder.py`](../archive/cypha-encoder/Cypha_Encoder.py)**
· 36,340 bytes, 913 lines, 11 classes

This file is **byte-identical** to `cypha-v3/Cypha_Encoder.py`:

```console
$ md5sum cypha-encoder/Cypha_Encoder.py cypha-v3/Cypha_Encoder.py
31bd0a62020340f37e574e22fc083ca2  cypha-encoder/Cypha_Encoder.py
31bd0a62020340f37e574e22fc083ca2  cypha-v3/Cypha_Encoder.py
```

It is not a separate line of development — it is the v3-era encoder kept aside as its own
unit, presumably because it was the part worth reusing.

The program calls itself **CyphaMicro** and is the small, honest prototype of the v3 era.
Its pipeline is genuinely closed-loop: content-defined chunking → D4 wavelet moments → a
256-bin DAMR reservoir histogram → importance selection to 64 dimensions → `PhaseBridge`
(real → complex) → `ResonanceField` → `Resonator` → `AnchorMemory`, with an
`AdaptiveControlLoop` that reads `FieldStats` and writes back `EncoderParams`
(`chunk_k`, `damr_radius`, `active_scales`).

Unlike the encoder in `cypha-v3/Cypha.py`, this loop actually closes. Encoding the same
input under `EncoderParams(chunk_k=4, damr_radius=3)` and `(chunk_k=8, damr_radius=8)` gives
different vectors — `max|diff| = 2.12`. v3's rewritten `BinaryEncoder.encode(self, data, p)`
never dereferences `p` at all, and returns bitwise-identical output under the same two
settings. And unlike v3's main
program, CyphaMicro stores and queries `AnchorMemory` on the **resonator output** — the
dynamical state — which is what the architecture documents say should happen.

It pays for that honesty. Trained three epochs on its own 14 built-in pairs and then asked
for them back, it recalls **8 of 14**. And because `infer()` does not reset the field, the
answer depends on what was asked before it — eight identical queries for `'cat sound'`
returned `false` once and `bark` seven times:

```
infer('cat sound') x8 -> ['false','bark','bark','bark','bark','bark','bark','bark']
```

The correct answer is `meow`, which it never gives. v3's main program abandoned
resonance-space retrieval for exactly this reason; see
[`03-v3.md`](03-v3.md#the-core-is-computed-and-discarded).

So the two files in `cypha-v3/` are a matched pair: the prototype that does what the
architecture claims and works poorly, and the production file that does something simpler
and works.

---

## `big-data/` — corrupt

**[`../archive/big-data/download_datasets.py`](../archive/big-data/download_datasets.py)**
· 8,755 bytes

The file contains no recoverable content. It is 8,755 `0x00` bytes — one distinct byte
value, zero printable characters:

```console
$ python3 -c "import collections; print(collections.Counter(open('download_datasets.py','rb').read()))"
Counter({0: 8755})
```

The copy inside the source zip is byte-identical, so the content was lost before the archive
was assembled rather than during integration. It is preserved as delivered.

What it was can only be inferred from its neighbours: `cypha-v5/download.py` and
`cypha-v5/convert.py` (both intact, 14,774 and 22,190 bytes) cover dataset acquisition and
format conversion for the v5/v6 benchmarking effort, and `Big Data` was presumably the
larger-corpus version of the same job. Nothing else in the archive references it.

This is the only corrupt file in the archive; a scan of all 142 text files found no other
NUL bytes or missing content.

---

## `datasets/` — the classification settings sweep

**[`../archive/datasets/cypha_classification_settings.jsonl.gz`](../archive/datasets/cypha_classification_settings.jsonl.gz)**
· 445,917 bytes gzipped, 20,243,385 raw

27,524 training runs over 5,397 configurations, 1,070.8 core-hours. It is the largest
research artifact in the archive and the one with the clearest line to the shipping product:
its winning constants appear hard-coded in the root monolith, and the knob it proved inert
is the reason the native code's deliberation band ships disabled.

Analysed in full in [`../SWEEP_ANALYSIS.md`](../SWEEP_ANALYSIS.md).

---

## `side-quests/retdec-upgrades5/` — decompiler patches

**[`../archive/side-quests/retdec-upgrades5/`](../archive/side-quests/retdec-upgrades5/)**
· 9 files, expanded from `retdec_upgrades5.zip`

Nothing to do with Cypha. These are patches for
[RetDec](https://github.com/avast/retdec), Avast's machine-code decompiler — four optimiser
passes plus an installer script, with 2024 copyright headers. The directory name says
"round 5", so rounds 1–4 existed elsewhere and are not in this archive.

**The nested zip is the best-dated object in the whole archive.** Its central directory kept
real per-entry mtimes, which the outer repack destroyed everywhere else:

```
2026-03-11 13:58  patches5/{intrinsic_conv_ext,if_structure_ext,pow2_arithm_ext,
                            var_renamer_ext,intrinsics_opt_ext,strength_reduction}/
2026-03-11 14:00  patches5/if_structure_ext/…      (2 files)
2026-03-11 14:00  patches5/intrinsic_conv_ext/…    (2 files)
2026-03-11 14:02  patches5/pow2_arithm_ext/…       (2 files)
2026-03-11 14:02  patches5/strength_reduction/…    (2 files)
2026-03-11 14:03  patches5/
2026-03-11 14:04  patches5/apply_patches5.sh
```

A **six-minute** authoring window on **2026-03-11** — the same day as the `Cypha v8`
directory.

Two things are visible in that first line. It is a `mkdir` brace expression that **never
expanded** — run under a shell that does not support brace expansion, it created a single
directory with a literal brace in its name, and that directory is still in the zip. And
because it never expanded, it preserves the author's *intended* patch set: six passes, not
four. `var_renamer_ext` and `intrinsics_opt_ext` were planned and never written.

`apply_patches5.sh` (10,102 bytes) is a real installer: it copies sources into a RetDec
checkout, edits the relevant `CMakeLists.txt` with `sed`, and rewrites call sites with an
inline Python script, guarding each step with `grep -q` so it is idempotent.

| Patch | Target | What it does |
|---|---|---|
| 1 | `llvmir2hll/llvm/llvm_intrinsic_converter` | 20+ intrinsic mappings the original lacks — `llvm.ceil.*`→`ceilf/ceil/ceill`, `llvm.round.*`, `llvm.nearbyint.*`, `llvm.trunc.*`, `llvm.rint.*`, `llvm.minnum.*`→`fminf/fmin/fminl`, and more |
| 2 | `llvmir2hll/optimizer/if_structure_optimizer` | patterns 6 and 7: consolidating consecutive `if`s with identical bodies where the second carries an `else` — explicitly picking up a `TODO` in the existing pattern 4, which required neither branch to have an `else` |
| 3 | `llvmir2hll/optimizer/simplify_arithm_expr/pow2_sub_optimizer` | `x*(2^N)→x<<N`, `x/(2^N)→x>>N` and `x%(2^N)→x&(2^N−1)` (both unsigned only), plus double-negation elimination `−(−x)→x`, `~(~x)→x` and `!(!x)→x` for `i1` |
| 4 | `bin2llvmir/optimizations/strength_reduction` | a new LLVM `ModulePass`: `mul x,(2^N)→shl`, `mul x,−(2^N)→neg(shl)`, `udiv x,(2^N)→lshr`, and related integer strength reductions, running after the constants pass |

The work is careful — the unsigned-only restrictions on `/` and `%` are correct (arithmetic
right shift does not implement signed division toward zero), and the `i1` restriction on
`!(!x)` is correct too.

### Why it might sit here

There is no evidence in the archive connecting these patches to Cypha, and this document
does not claim one. Two observations are worth recording, neither conclusive:

- The date is the same day as v8 — the day the project restarted from scratch.
- The subject matter is the same discipline the main line had been practising since v1:
  replacing expensive operations with cheap equivalent ones, on a single core. Patch 3 and
  patch 4 are strength reduction, which is item-by-item what
  `cypha-v1/Cypha Speed Improvements.txt` was about.

The most likely reading is simply that the author worked on both and zipped them together.

---

## `root-monolith/` — not a loose component

`../archive/root-monolith/Cypha.py` sat at the archive root rather than in a version
directory, which makes it look like a stray file. It is the opposite: it is the newest
artifact in the archive and the closest surviving Python ancestor of the native C++ product.

It gets its own document — [`../PYTHON_TO_CPP_BRIDGE.md`](../PYTHON_TO_CPP_BRIDGE.md).
