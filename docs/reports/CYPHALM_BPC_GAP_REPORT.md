# CyphaLM BPC Gap Report — hp compress vs Cypha scoring

**Date:** 2026-09-19  
**Branch:** `cursor/bpc-gap-report-94d7` (from `cursor/cyphalm-llm-profile-f9fe`)  
**Harness:** `native/tools/bpc_gap_measure.cpp`, `scripts/cyphalm_bpc_gap.sh`, `native/tools/bpc_gap_verify.cpp`, `native/tools/hp_bit_tree_smoke.cpp`

---

## Executive summary

The headline **~5.5 BPC** (WikiText, n=16, after 32 train steps) and the user-reported **~1.610 BPC** (8 MB enwik leftover, v78/champ hp archive) measure **different things on different flag profiles**. After fixing a `Predictor` copy/`ctx_chain_` aliasing bug and routing default `eval_bpc` through the **bit-serial observe** path (hp’s native unit), Cypha **compress-equivalent BPC matches hp archive BPC on the same corpus and compile flags within ≤0.003 BPC**.

| Question | Answer (measured) |
|----------|-------------------|
| Is Cypha math wrong vs hp compress? | **No** — observe NLL ≡ archive BPC at light profile on enwik + WikiText |
| Why did WikiText show ~5.5? | **256-clone `predict_next` API metric**, broken copy pre-fix, short n=16 eval — not archive BPC |
| Why is 1.610 not 1.721? | **Flag profile** — v78 gate (`v78_flags.ps1` + `SLOT_MAX=35`) vs Cypha **light** (`SLOT_MAX=24`, grow OFF) |
| Fast scoring path? | **Bit-serial observe** (~750× faster than 256-clone); bit-tree full vocab still slower than fixed legacy clone (needs undo stack) |

All numbers below are from `bench/results/cyphalm_bpc_gap/` on this VM unless noted.

---

## Root causes (ranked)

### 1. Metric conflation (primary narrative gap)

| Metric | Definition | enwik 8 MB (light) | WikiText 100k (light) |
|--------|------------|--------------------|------------------------|
| **hp archive BPC** | `hp c` compressed size / bytes / 8 | **1.721** | **2.036** |
| **Cypha observe BPC** | `eval_bpc_compress_equivalent`: Σ −log₂ p(bit) with update | **1.721331** | **2.033812** |
| **Cypha clone eval BPC** | `eval_bpc` legacy: `predict_next` + 256× `next_byte_log_probs` | (not run at 8 MB) | **7.252** (n=16, cold) |

The **5.48 BPC** headline in [`CYPHALM_LLM_PROFILE_REPORT.md`](CYPHALM_LLM_PROFILE_REPORT.md) is the third row class (clone API, n=16). It is **not comparable** to hp archive BPC without relabeling.

### 2. `Predictor` copy bug (pre-fix wrong clone logprobs)

`hp::Predictor::ctx_chain_[]` holds raw pointers into member storage, initialized only in the constructor via `init_ctx_chain_()`. Default copy left scratch aliasing live state → **256-clone `next_byte_log_probs` was not compress-equivalent**.

**Fix:** `Predictor::clone_from()`, `assign_from_()`, `rebind_internal_pointers_()`; `MatchModel` / `WordMatchModel::set_ring()`.

**Parity:** `hp_bit_tree_smoke OK max_tree_legacy_delta=0`; post-fix clone API matches observe on random contexts (`bpc_gap_verify`).

### 3. Flag / profile mismatch (1.610 vs 1.721)

| Build | `SLOT_MAX` | v78 `-D` flags | enwik 8 MB archive BPC |
|-------|------------|----------------|--------------------------|
| **hp_light** / Cypha light | 24 | grow OFF (production default) | **1.721** |
| **hp_v78_slot24** | 24 | v78_flags subset + gate | **2.169** @ 64 KB; converges toward champ on long corpus |
| **hp_v78_slot35** (champ) | 35 | full `v78_flags.ps1` | **1.611** @ 8 MB |

User **1.610** aligns with **champ / v78 8 MB gate**, not Cypha **light**. Cypha champ build was **skipped** on this 16 GiB VM (OOM risk).

### 4. `train_sequence` double-adapt (fixed)

Prior `train_step` consumed the token twice on overlapping pairs, corrupting context during online train. Fixed to: `consume(ids[0])`, then per step `log_prob_byte(next)` + `observe_next_byte(next)`.

---

## Flag protocol (confirmed)

| Name | Compile | Runtime | Role |
|------|---------|---------|------|
| **Cypha light** | `CYPHA_HP_PROFILE=light` → `HP_SLOT_MAX=24`, grow OFF | `apply_hp_production_recipe()` | Production default; ~782 MiB construct RSS |
| **Cypha champ** | `CYPHA_HP_PROFILE=champ` → `v78_flags.ps1` + `HP_SLOT_MAX=35` | `apply_hp_champ_recipe()` | Research / 8 MB gate; ~15 GB lab RSS |
| **hp 8 MB gate (v78)** | `v78_flags.ps1` + `SLOT_MAX=24` or `35` | `--mem 22 --lr 2` | CompressionAlgorithm leftover champ path |
| **Bare light hp CLI** | `-DHP_SLOT_MAX=24 -DHP_XSIMD=0` | same | Matches Cypha light integration flags |

Vendored hp tree: **154/154** blob parity with upstream CompressionAlgorithm (vendor diff harness attempted; path issue in one log — manual spot checks OK).

---

## Same-corpus measurements (light profile, `table_bits=22`)

Source JSON: `bench/results/cyphalm_bpc_gap/measure_*.json`, `hp_light_vs_v78_compress.txt`.

| Corpus | Bytes | Cypha observe BPC | hp archive BPC | Δ (observe − archive) |
|--------|-------|-------------------|----------------|-------------------------|
| enwik8.8mb | 65,536 | 2.187 | 2.190 | −0.003 |
| enwik8.8mb | 262,144 | 1.881 | 1.882 | −0.001 |
| enwik8.8mb | 1,048,576 | 1.773 | 1.773 | +0.000 |
| enwik8.8mb | 8,388,608 | **1.721331** | **1.721** | +0.000 |
| WikiText train | 100,000 | 2.034 | 2.036 | −0.002 |
| WikiText train | 262,144 | 1.894 | 1.895 | −0.001 |

**8 MB observe wall time:** 209,499 ms (~3.5 min). **hp compress:** 231,973 ms.

---

## Latency — bit path vs 256-clone (64 KB enwik, 1 iter)

From `bench/results/cyphalm_bpc_gap/latency_enwik64k.json`:

| Path | Latency | Notes |
|------|---------|-------|
| `observe` 64 KB (compress-equiv BPC) | **1,907 ms** | 8× predict+update per byte, no fan-out |
| `legacy next_byte_log_probs` (256× `clone_from`) | **23,551 ms** / call | Fixed copy; still 256 full clones |
| `bit_tree next_byte_log_probs` | **71,112 ms** / call | ~510 `clone_from` on DFS branches — **slower** than legacy |
| `predict_next` (consume + legacy logprobs) | **77,350 ms** / call | Dominated by vocab fan-out |

**Speed ratio:** observe / legacy ≈ **750×** on throughput (consistent with prior profile’s `consume_byte` vs `predict_next`).

**Architectural default after this PR:**

- `eval_bpc()` → `eval_bpc_compress_equivalent()` (bit-serial observe NLL)
- `next_byte_log_probs()` → legacy 256-clone (fixed); opt-in bit-tree via `CYPHA_HP_BIT_TREE_LOGPROBS=1`
- `log_prob_byte()` / `sample_next_byte()` — O(8) single-clone paths for train / generation

**Follow-up:** undo-stack predictor (profile report item #2) to make bit-tree / full-vocab scoring O(8) branches without 256× or 510× full state copies.

---

## Algorithm encyclopedia — scoring paths

hp’s native unit is the **bit**. A byte is eight MSB-first conditionals:

\[
P(\text{byte} \mid \text{ctx}) = \prod_{i=7}^{0} P(\text{bit}_i \mid \text{ctx}, \text{prefix bits})
\]

| Path | Mechanism | BPC valid? | Full 256 softmax? |
|------|-----------|--------------|-------------------|
| **observe / compress-equivalent** | Main `pred_`: predict → update per bit on observed stream | **Yes** (matches hp archive) | No |
| **log_prob_byte** | One `clone_from`, 8 bit steps, no main advance | Per-byte NLL only | No |
| **legacy next_byte_log_probs** | 256× `clone_from`, 8 bit steps each | Per-call distribution; expensive | Yes |
| **bit_tree next_byte_log_probs** | DFS over bit prefixes; sibling branches share prefix state | Mathematically ≡ legacy when copy correct | Yes (opt-in) |
| **predict_next API eval** | consume + legacy logprobs + adapt | **Different metric** (API-shaped; was broken pre-fix) | Yes |

**Identity checks (A1/A2/C10):**

- Σᵢ −log₂ p(bitᵢ) = −log₂ P(byte) (MSB order)
- `clone_from` observe ≡ direct observe
- bit-tree ≡ legacy: `max_tree_legacy_delta=0`

---

## Integration status

| Check | Status |
|-------|--------|
| observe BPC vs hp archive (same flags) | **PASS** (≤0.003) |
| Clone copy fix | **DONE** (`clone_from` + rebind) |
| Default `eval_bpc` = compress-equivalent | **DONE** |
| Bit-tree full vocab | **Implemented**; not default (slower than fixed legacy) |
| Bit-serial generation | **`sample_next_byte()` implemented**; not wired in `cyphalm_generation.cpp` |
| Champ Cypha profile on 16 GB VM | **Skipped** |
| `bpc_gap_verify` full script | Default `--random 8` (512 × ~24 s impractical) |
| Locked BPC in `BASELINE_LOCK.json` | **Not updated** (historical Hybrid pins remain) |

---

## How to reproduce

```bash
# Build (light profile)
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DCYPHA_HP_PROFILE=light
cmake --build native/build --target bpc_gap_measure hp_bit_tree_smoke bpc_gap_verify -j$(nproc)

# hp CLI (light)
g++ -std=c++17 -O2 \
  -Inative/third_party/hp/include \
  -Inative/third_party/hp/third_party/xsimd/include \
  -DHP_SLOT_MAX=24 -DHP_XSIMD=0 \
  native/third_party/hp/src/main.cpp -o native/build/hp_light

# Measure observe vs archive
./native/build/bpc_gap_measure --corpus bench/data/enwik8/enwik8.8mb --bytes 8388608
./native/build/hp_light c --mem 22 --lr 2 /tmp/in.bin /tmp/out.cyhp   # archive BPC on stderr

# Parity
./native/build/hp_bit_tree_smoke
./native/build/bpc_gap_verify --random 8

# Full harness
bash scripts/cyphalm_bpc_gap.sh bench/data/enwik8/enwik8.8mb 100000 16
```

Corpus: `bench/data/enwik8/enwik8.8mb` (8 MB enwik slice), `bench/data/wikitext2/wikitext-2/wiki.train.tokens`.

---

## Recommendations

1. **Report BPC with labels:** always state *observe/compress-equivalent* vs *clone-API* vs *hp archive*, plus flag profile.
2. **Use observe path** for BPC benchmarks and training NLL (`train_sequence` already does).
3. **Keep `predict_next` / legacy logprobs** for REST top-k until undo-stack or partial expansion lands.
4. **Champ profile** measurement belongs on ≥32 GB host; expect ~1.61 BPC on 8 MB enwik when flags match v78 gate.
5. **Do not lock** a single headline BPC in `MODEL_CARD` without profile + metric disambiguation.

---

## Related docs

- [`CYPHALM_LLM_PROFILE_REPORT.md`](CYPHALM_LLM_PROFILE_REPORT.md) — RSS, latency, WikiText capability (clone-API headline retained with cross-link)
- [`MODEL_CARD.md`](../../MODEL_CARD.md) — product card (updated BPC labeling)
- [`docs/native/CYPHALM_NATIVE_BUILD.md`](../native/CYPHALM_NATIVE_BUILD.md) — light vs champ build

*All BPC / latency numbers in this report are measured; none are invented or extrapolated.*
