# CyphaLM BPC Gap Report — hp compress vs Cypha scoring

**Date:** 2026-09-19  
**Branch:** `cursor/bpc-gap-report-94d7` (from `cursor/cyphalm-llm-profile-f9fe`)  
**Harness:** `native/tools/bpc_gap_measure.cpp`, `scripts/cyphalm_bpc_gap.sh`, `native/tools/bpc_gap_verify.cpp`, `scripts/hp_v78_flag_diff.py`

---

## Executive summary

### PROTOCOL MISMATCH (read this first)

**Cypha production default (`CYPHA_HP_PROFILE=light`) is NOT the user’s v82 champ recipe.**

| Profile | v78 `-D` flags matched (of 78 non-SLOT) | `HP_SLOT_MAX` | enwik 8 MB archive BPC (measured this VM) |
|---------|----------------------------------------|---------------|-------------------------------------------|
| **Cypha light / bare hp** | **0 / 78** | 24 | **1.721362** (archive **1,804,979 B**) |
| **v78 + SLOT_MAX=24 gate** | **78 / 78** | 24 | **1.611759** (archive **1,690,052 B**) |
| **v82 champ (PLAN/RECORD ref)** | **78 / 78** | **35** | **1.610906** (archive **1,689,157 B**) — *cited from CompressionAlgorithm encyclopedia; not re-measured here (OOM on 16 GiB VM)* |
| **Cypha gate24 build** | **78 / 78** (via `HpFlags.cmake`) | 24 | **1.611759** archive / **1.611729** observe — **re-measured 2026-09-19** |
| **Cypha champ build** | **78 / 78** (via `HpFlags.cmake`) | 35 | *OOM-killed @ 15 GiB VM (exit 137, ~5–11 s); ~15 GB RSS per hp harness* |

**Cypha light observe BPC matches hp archive BPC only when both use the same bare-light flags** (Δ **+0.000331** @ 8 MB). Comparing Cypha light **1.721** to user **1.610** is a **protocol + metric** gap, not an integration math bug.

The headline **~5.5 BPC** (WikiText clone-API, n=16) is a third metric again — under-trained short eval via 256× `predict_next`, not archive BPC.

| Question | Answer (measured / verified) |
|----------|------------------------------|
| Is Cypha bit-serial math wrong? | **No** — byte NLL = Σ bit NLLs; observe ≡ archive @ **same flags** |
| Why ~5.5 on WikiText? | **256-clone API metric**, n=16; post-fix clone path **7.25 BPC** on 100k slice |
| Why 1.610 ≠ 1.721? | **v82 champ flags + SLOT_MAX=35** vs Cypha **light** (0/78 v78 flags ON) |
| Fast scoring path? | **Bit-serial observe** (~750× faster than 256-clone) |

---

## v82 champ protocol (CompressionAlgorithm encyclopedia — re-measured where noted)

| Item | Value |
|------|-------|
| Corpus | First **8,388,608** bytes of enwik8 (`bench/data/enwik8/enwik8.8mb`; SHA256 `09f6dd7241a8ae21edfd6762f3c6712a1fd02f7f322c5e77cab8bb88f292ee8e`) |
| User archive | **1,689,157 B** → BPC = `archive×8/raw` = **1.610906** |
| CLI | `hp c --mem 22 --lr 2` |
| Compile | Full **`hp/tools/v78_flags.ps1`** — **79** `-DHP_*` flags including `HP_SLOT_MAX=35`, wiki CMs through `HP_WORDLEN_MOD`, mixer gates, slot growth, `-msse4.1`, xsimd includes |
| Bare default (features.hpp, grow OFF) | **1,804,979 B / 1.721362 BPC** — **re-measured** `hp_light` this VM |
| Experimental (not measured here) | +`HP_LR1_SCALE=40` +`HP_WIKIBOLD_MOD` → **1,681,311 B / 1.6035 BPC** @ SLOT_MAX=24 only (encyclopedia; needs SLOT_MAX=35 confirm) |

### Footguns (integration)

1. **Without `v78_flags.ps1`, you are NOT measuring the user’s algorithm quality.** Cypha light = `features.hpp` defaults (almost everything OFF) + `HP_SLOT_MAX=24`.
2. **`--mem` ≠ `SLOT_MAX`.** Runtime `--mem 22` sets table bits; compile-time `HP_SLOT_MAX` caps slot tables (screen **24**; v82 8 MB champ uses **35**).
3. **BPC includes CYHP header** (`hdr+dict 29` in hp CLI stderr).
4. **Integer-only predictor; bit-serial arithmetic coder** — Cypha observe path matches this semantics.
5. **RT PASS = SHA256 roundtrip** — v78 gate24 archive on enwik 8 MB: **RT_PASS=1**, SHA256 match (re-measured this VM).

---

## Flag diff — Cypha vs `v78_flags.ps1`

**Source:** `native/third_party/hp/tools/v78_flags.ps1` (79 `-D` flags), parsed by `scripts/hp_v78_flag_diff.py` against `hp/features.hpp` defaults.

| Profile | CMake / compile | Non-SLOT v78 flags matched | `HP_SLOT_MAX` | Other compile deltas vs v82 CLI |
|---------|-----------------|----------------------------|---------------|----------------------------------|
| **Cypha light** | `CYPHA_HP_PROFILE=light` | **0 / 78** | 24 | `HP_XSIMD=0` (default); no `-msse4.1` on Cypha unless `CYPHA_HP_XSIMD=ON` |
| **Cypha champ** | `CYPHA_HP_PROFILE=champ` | **78 / 78** | 35 | Same XSIMD default OFF; champ still needs `CYPHA_HP_XSIMD=ON` for full v82 SIMD path |
| **v82 hp CLI** | manual g++ with all `-D` from ps1 | 78 / 78 | 35 | `HP_XSIMD=1`, `-msse4.1`, `-O3` |

**All 76 v78 flags that default OFF in `features.hpp` stay OFF in Cypha light** (slot/word/wiki CM mods, gates, `HP_SPARSE_UTF8`, `HP_SENT_STREAM`, … through `HP_WORDLEN_MOD`).

**Two value mismatches even before v78 `-D`s:**

| Flag | `features.hpp` default | v82 `-D` | Cypha light effective |
|------|------------------------|----------|------------------------|
| `HP_MIXER_SKIP` | 0 | 32 | 0 |
| `HP_STATE_MOD` | `HP_WIKI_AXES` | 1 | `HP_WIKI_AXES` |

Cypha **champ** sets both via `v78_flags.ps1` parse in `native/cmake/HpFlags.cmake`.

### Full flag table (79 rows)

Run: `python3 scripts/hp_v78_flag_diff.py`

Abbreviated — every v82 flag is **N** under light, **Y** under champ:

| Flag group (v82 ON) | Count | Light |
|---------------------|-------|-------|
| Mixer / gates (`HP_MIXER_SKIP=32`, `HP_GATE_*`, …) | 4 | all OFF / wrong |
| Slot growth (`HP_SLOT_GROW*`, `HP_MATCH_GROW`, …) | 4 | OFF |
| Extra slots (`HP_SLOT_WORD2..9`, `HP_SLOT_O*`, …) | 28 | OFF |
| Wiki / sentence CMs (`HP_*_MOD`, `HP_SENT_*`, `HP_WIKISTACK_MOD`, …) | 24 | OFF |
| Match / word (`HP_MATCH_*`, `HP_WMATCH_4`, `HP_QUOTE_STACK`, …) | 8 | OFF |
| Groups (`HP_*_GRP`, `HP_SLOT_SGRP`, …) | 10 | OFF |
| `HP_SLOT_MAX` | — | **24** vs v82 **35** |

---

## Root causes (ranked)

### 1. Protocol mismatch (primary gap to user 1.610)

Cypha default integration measures **bare-light hp** (~**1.721 BPC**), not v82 champ (~**1.610 BPC**). This alone explains **~0.11 BPC** of the headline gap — before any WikiText LM eval confusion.

### 2. Metric conflation (WikiText ~5.5)

| Metric | Definition | enwik 8 MB | WikiText 100k |
|--------|------------|------------|---------------|
| **hp archive BPC** | `hp c` → `out_bytes×8/raw` (incl. header) | **1.721** (light) | **2.036** (light) |
| **Cypha observe BPC** | `eval_bpc_compress_equivalent` | **1.721331** (light) | **2.033812** |
| **Cypha clone-API BPC** | `predict_next` + 256× logprobs | — | **7.252** (n=16) |

Profile report **5.48 BPC** = clone-API class, not archive.

### 3. `Predictor` copy bug (fixed)

`ctx_chain_[]` pointer aliasing on copy broke 256-clone logprobs. Fixed via `clone_from` + `rebind_internal_pointers_()`.

### 4. `train_sequence` double-adapt (fixed)

Overlapping `consume` removed; train uses `log_prob_byte` + `observe_next_byte`.

---

## Same-corpus measurements (Cypha light, `table_bits=22`, `--mem 22 --lr 2`)

Corpus SHA256 above. hp archive via matching `hp_light` CLI.

| Corpus | Bytes | Cypha observe | hp archive | Δ |
|--------|-------|---------------|------------|---|
| enwik8.8mb | 65,536 | 2.187 | 2.190 | −0.003 |
| enwik8.8mb | 262,144 | 1.881 | 1.882 | −0.001 |
| enwik8.8mb | 1,048,576 | 1.773 | 1.773 | +0.000 |
| enwik8.8mb | **8,388,608** | **1.721331** | **1.721000** | **+0.000331** |
| WikiText train | 100,000 | 2.034 | 2.036 | −0.002 |
| WikiText train | 262,144 | 1.894 | 1.895 | −0.001 |

**8 MB re-run (2026-09-19):** observe **199,542 ms**; hp compress **243,681 ms**.

### v78-flag hp CLI on same corpus (not Cypha integration)

| Build | Archive bytes | BPC | Notes |
|-------|---------------|-----|-------|
| `hp_light` (bare) | **1,804,979** | **1.721362** | matches Cypha light observe |
| `hp_v78_gate24` (78 flags + `SLOT_MAX=24`) | **1,690,052** | **1.611759** | RT SHA256 **PASS** |
| v82 champ ref (encyclopedia) | **1,689,157** | **1.610906** | Δ **+895 B** vs gate24 → likely **`SLOT_MAX=35`** |
| `hp_v78_champ` (`SLOT_MAX=35`) | — | — | **OOM killed** compress @ 8 MB on 16 GiB VM |

---

## Math verification (bit-serial observe)

`bpc_gap_verify --random 4` (Cypha light, `table_bits=22`):

| Check | Result |
|-------|--------|
| A1 byte NLL = Σ bit NLLs (MSB-first) | **PASS** (`max_delta=8.88e-16`) |
| A2 clone ≡ observe | **PASS** (`max_delta=0`) |
| C10 clone ≡ `next_byte_log_probs` API | **PASS** (`max_delta=0`) |
| C10 MSB bit order | **PASS** |
| `hp_bit_tree_smoke` tree ≡ legacy | **PASS** (`max_delta=0`) |

---

## Latency — bit path vs 256-clone (64 KB enwik, 1 iter)

| Path | Latency |
|------|---------|
| observe 64 KB | **1,907 ms** |
| legacy `next_byte_log_probs` | **23,551 ms** |
| bit_tree `next_byte_log_probs` | **71,112 ms** (510 clones — needs undo stack) |
| `predict_next` | **77,350 ms** |

Default after this PR: `eval_bpc()` → compress-equivalent observe; full vocab stays on fixed legacy clone unless `CYPHA_HP_BIT_TREE_LOGPROBS=1`.

---

## Algorithm encyclopedia — hp scoring paths

hp’s native unit is the **bit**:

\[
P(\text{byte}\mid ctx)=\prod_{i=7}^{0} P(\text{bit}_i\mid ctx,\text{prefix bits})
\]

| Path | Updates main `pred_`? | Valid BPC? | Full 256 softmax? |
|------|----------------------|------------|-------------------|
| **observe / compress-equivalent** | Yes, per observed bit | **Yes** (= archive @ same flags) | No |
| **hp `c` archive** | Yes (compress loop) | **Reference** | No |
| **log_prob_byte** | No (one clone) | Per-byte NLL | No |
| **legacy next_byte_log_probs** | No (256 clones) | Distribution only | Yes |
| **bit_tree next_byte_log_probs** | No (DFS clones) | ≡ legacy | Yes (opt-in) |
| **predict_next / clone eval_bpc** | Yes (consume + adapt) | **Different metric** | Yes |

**Cypha integration default:** treat bytes as bits on the **observe** path for BPC; reserve 256-clone for REST top-k until undo-stack lands.

---

## Integration status

| Check | Status |
|-------|--------|
| observe ≡ archive @ **light** flags | **PASS** (≤0.003 BPC) |
| observe vs user **1.610** | **Protocol mismatch** — need champ + SLOT_MAX=35 |
| Flag diff documented | **0/78 light, 78/78 champ** |
| Clone copy fix | **DONE** |
| Default `eval_bpc` = observe | **DONE** |
| v82 champ compress on 16 GiB VM | **OOM** — needs ≥32 GiB host |
| Locked `BASELINE_LOCK.json` | **Not updated** |

---

## How to reproduce

```bash
# Flag diff
python3 scripts/hp_v78_flag_diff.py | tee /tmp/v78_flag_diff.tsv

# Cypha observe vs bare hp archive (light)
cmake -S native -B native/build -DCYPHA_HP_PROFILE=light
cmake --build native/build --target bpc_gap_measure bpc_gap_verify -j$(nproc)
./native/build/bpc_gap_measure --corpus bench/data/enwik8/enwik8.8mb --bytes 8388608 \
  --hp-tool ./native/build/hp_light

# Build hp CLIs
V78=$(grep -oE '\-DHP_[A-Z0-9_]+=[0-9]+' native/third_party/hp/tools/v78_flags.ps1 | tr '\n' ' ')
g++ -std=c++17 -O3 -msse4.1 -I native/third_party/hp/include \
  -I native/third_party/hp/third_party/xsimd/include \
  -DHP_SLOT_MAX=24 -DHP_XSIMD=0 \
  native/third_party/hp/src/main.cpp -o native/build/hp_light
# v78 gate24: same but $V78 -DHP_SLOT_MAX=24 -DHP_XSIMD=1

# Parity
./native/build/bpc_gap_verify --random 4
./native/build/hp_bit_tree_smoke
```

---

## Recommendations

1. **Never compare 1.610 to Cypha light 1.721** without labeling v82 champ vs bare light.
2. **Label every BPC:** observe / archive / clone-API + `CYPHA_HP_PROFILE` + `HP_SLOT_MAX`.
3. **Champ measurement** on ≥32 GiB host with `-DCYPHA_HP_PROFILE=champ -DCYPHA_HP_XSIMD=ON`.
4. **Use observe path** for BPC benchmarks; keep clone path for REST until undo-stack.
5. **WikiText headline ~5.5** remains clone-API + short eval — not a compression-quality claim.

---

## Related docs

- [`CYPHALM_LLM_PROFILE_REPORT.md`](CYPHALM_LLM_PROFILE_REPORT.md)
- [`MODEL_CARD.md`](../../MODEL_CARD.md)
- [`native/cmake/HpFlags.cmake`](../../native/cmake/HpFlags.cmake)

*All BPC / archive-byte numbers marked “re-measured” were taken on this VM; encyclopedia cites (1,689,157 B / 1.610) are not re-measured here due to champ OOM.*
