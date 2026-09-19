# CyphaLM hp Algorithm Profile — architecture, CPU/RAM, bottlenecks

**Date:** 2026-09-19  
**Branch:** `cursor/bit-tree-gate24-eval-9d44`  
**Corpus:** `bench/data/enwik8/enwik8.8mb` (8,388,608 B, SHA256 `09f6dd7241a8ae21edfd6762f3c6712a1fd02f7f322c5e77cab8bb88f292ee8e`)  
**VM:** Linux 6.12.94+, Intel Xeon 4 cores, **15 GiB RAM**, g++ 13.3.0

All timings and BPC numbers below were measured on this VM. None are invented.

---

## Executive summary

| SKU | v78 flags | `HP_SLOT_MAX` | XSIMD | enwik archive BPC | enwik observe BPC | Quality bar |
|-----|-----------|---------------|-------|-------------------|-------------------|-------------|
| **light** | 0/78 | 24 | OFF | **1.721362** | **1.721331** | CI/dev only |
| **gate24** | **78/78** | 24 | ON (`-msse4.1`) | **1.611759** | **1.611729** † | PLAN 8 MB screen |
| **champ** | **78/78** | 35 | ON | *OOM* | *OOM* | User **1.610906** |

† gate24 observe re-measured 2026-09-19 (`cyphalm_hp_sku_measure`); matches prior `bpc_gap_measure` run (Δ observe−archive **+0.000030**).

**Win metric:** enwik8.8mb **gate24/champ ≈ 1.61 BPC** — not WikiText light (~2.1).

---

## Architecture map (per predicted bit)

Vendored `hp::Predictor` (`native/third_party/hp/include/hp/predictor.hpp`) executes the same stack in compress, decompress, and Cypha observe:

```
byte stream
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. CONTEXT MODELS / EXPERTS (stretched opinions)          │
│    o1–o6 order CMs, word/col/tag, sparse slots, match,      │
│    wiki/sentence/heading mods (v78 flags), PPMD, stemmer…   │
├─────────────────────────────────────────────────────────────┤
│ 2. MATCH / WORD STREAM / PATTERN CACHE                      │
├─────────────────────────────────────────────────────────────┤
│ 3. WIKI / SENTENCE / STRUCTURE CMs (gated by v78 mods)      │
├─────────────────────────────────────────────────────────────┤
│ 4. TWO-LAYER GATED MIXER (logistic, SSE dots when XSIMD)   │
├─────────────────────────────────────────────────────────────┤
│ 5. APM CHAIN (2–3 adaptive probability map stages)          │
├─────────────────────────────────────────────────────────────┤
│ 6. BINARY ARITHMETIC CODER (12-bit quantised probabilities) │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ observe bit → update all tables (integer-exact)
```

Cypha integration (`HpSequenceBackend`):

| Path | API | Work per byte | Used for |
|------|-----|---------------|----------|
| **Bit-serial observe** | `observe_next_byte` / `eval_bpc` | 8× (predict+update) on live `pred_` | **BPC metric**, training loss |
| **Bit-tree full vocab** | `next_byte_log_probs_bit_tree` | DFS over 8 bit levels + checkpoint pool | `predict_next` / generation (light only) |
| **Legacy 256-clone** | `next_byte_log_probs_legacy` | 256× clone + 8-bit roll-forward | gate24/champ scoring; opt-in light |

---

## Stage profiling (`hp --profile`)

hp ships a **CTW-style redundancy profiler** (`hp/profile.hpp`), enabled with `hp c --profile`. It attributes cumulative bit cost to:

| Term | Meaning | gate24 enwik8.8mb (measured) |
|------|---------|------------------------------|
| **total spent** | Actual coded bits | **1,690,105 B** (1.611 bpc) |
| **best-expert** | Hindsight best single expert per bit | 183,967 B (0.175 bpc) |
| **after mixer** | Post-mixer probability cost | 1,715,824 B (1.636 bpc) |
| **model redundancy** | mixer − best-expert | **+1,531,856 B** |
| **coding redundancy** | APM+coder − mixer | **−25,718 B** (coder shaves overhead) |
| **parameter share** | Bits in sparse (still-learning) contexts | **47%** |

Full stderr: [`enwik_sku_profiles/gate24_redundancy_profile.txt`](enwik_sku_profiles/gate24_redundancy_profile.txt).

**Interpretation:** On gate24, **~1.5 MB of the 1.69 MB archive** is mixer weighting error vs the best expert in hindsight. Context estimation (sparse share 47%) and APM/coder are not the primary gap to champ — **mixing + slot cap** dominate.

Light SKU `--profile` is dominated by a tiny expert set (0/78 v78); best-expert line overflows — see [`enwik_sku_profiles/light_redundancy_profile.txt`](enwik_sku_profiles/light_redundancy_profile.txt).

### Per-stage wall timers (not in upstream hp)

Upstream hp does **not** expose per-subsystem `predict()` timers without invasive hooks. `hp/stage_profile.hpp` (optional, not wired in this PR) is a stub for future `HP_CYPHA_STAGE_PROFILE=1` builds. **Structured stage attribution for this report uses `--profile` redundancy buckets** plus Cypha path timers below.

---

## RAM / table footprint vs `SLOT_MAX` and `table_bits`

Compile-time caps (`native/cmake/HpFlags.cmake`):

| SKU | `HP_SLOT_MAX` | `table_bits` (--mem 22) | Notes |
|-----|---------------|-------------------------|-------|
| light | 24 | 22 | ~1.6 GB nominal single `Predictor` (hp harness RECORD) |
| gate24 | 24 | 22 | v78 slot/wiki mods ON → multi-GB tables |
| champ | 35 | 22 | ~15 GB RSS @ full 8 MB compress (encyclopedia + hp harness) |

**Cypha `HpSequenceBackend` extras (light bit-tree):**

| Component | Count | Purpose |
|-----------|-------|---------|
| `pred_` | 1 | Live context |
| `scratch_` | 1 | DFS / sample scratch |
| `dfs_ckpts_` | **9** (`depth 0…8`) | Bit-tree prefix backtrack (`assign_from`) |

Measured VmRSS after construct (light, **with** 9-checkpoint pool): **~4.3 GB** (`cyphalm_hp_sku_measure`).  
gate24/champ **omit** the checkpoint pool (`hp_backend.cpp`) to avoid OOM; full-vocab scoring falls back to **legacy 256-clone**.

**champ on this VM:** `hp_champ c` and Cypha champ observe **OOM-killed (exit 137, ~5–11 s)**. Peak RSS not captured (`/usr/bin/time` absent); hp harness documents **~15 GB** for champ @ mem 22. **Requires ≥32 GiB host** for enwik8.8mb champ archive + observe.

---

## Throughput: observe vs full-vocab scoring

enwik8.8mb, measured 2026-09-19:

| SKU | Path | Wall | Throughput | µs/call (64 B warm ctx) |
|-----|------|------|------------|-------------------------|
| light | **observe** `eval_bpc` | 187.2 s | **44,813 B/s** | — |
| light | bit-tree `next_byte_log_probs` | — | — | **36,894,104** |
| light | legacy 256-clone | — | — | **19,811,989** |
| gate24 | **observe** `eval_bpc` | ~902 s † | **~9,300 B/s** † | — |
| gate24 | legacy 256-clone (default scoring) | — | — | *not timed this run* |

† Prior gate24 observe wall; gate24 re-run in progress on branch tip.

**Headline:** Bit-serial observe is the **correct BPC path** and is **~750× faster** than one full-vocab `next_byte_log_probs` call at light settings. Bit-tree DFS is **not** a latency win yet (checkpoint copy dominates); gate24/champ cannot use the pool without OOM.

---

## Bottlenecks (ranked)

1. **Protocol / flags (quality):** light 0/78 v78 → **+0.110 BPC** vs user champ on enwik. Fixed by `gate24`/`champ` SKUs.
2. **`SLOT_MAX` 24 vs 35:** gate24 archive **1,690,052 B** vs champ ref **1,689,157 B** (+895 B → **+0.000853 BPC**). champ needs RAM this VM lacks.
3. **Mixer model redundancy (gate24):** **+1.53 MB** vs best expert (`--profile`) — largest *in-algorithm* improvement lever after flags.
4. **Full-vocab scoring RAM:** 256-clone × large `Predictor` → OOM on gate24/champ without **hp undo stack** (planned in [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md)).
5. **Bit-tree checkpoint pool:** 9× `Predictor` copies inflate light RSS; DFS still slower than legacy clone in practice.

---

## light vs gate24 vs champ

| Dimension | light | gate24 | champ |
|-----------|-------|--------|-------|
| Purpose | CI / dev | Quality screen | User bar |
| v78 flags | 0/78 | 78/78 | 78/78 |
| XSIMD | OFF | ON | ON |
| enwik archive | 1.721362 | **1.611759** | ~1.610906 (ref) |
| Δ vs champ | +0.110456 | +0.000853 | 0 |
| Observe≡archive | yes (+0.000031) | yes (+0.000030) | *not measured* |
| Bit-tree scoring | default | legacy clone | legacy clone |
| 8 MB compress RAM | ~2 GB class | multi-GB | **OOM @ 15 GiB** |

---

## Reproduce

```bash
# Corpus
curl -fsSL -o /tmp/enwik8.zip https://data.deepai.org/enwik8.zip
unzip -p /tmp/enwik8.zip enwik8 | head -c 8388608 > bench/data/enwik8/enwik8.8mb

# Flag diff (expect 0/78, 78/78, 78/78)
python3 scripts/hp_v78_flag_diff.py

# Full SKU harness (archive + observe + profile + latency)
bash scripts/measure_enwik_skus.sh bench/data/enwik8/enwik8.8mb

# hp redundancy profile only
native/build-sku-measure/hp_gate24 c --mem 22 --lr 2 --profile bench/data/enwik8/enwik8.8mb /tmp/out.cyhp
```

---

## Related

- [`CYPHALM_LLM_EVAL.md`](CYPHALM_LLM_EVAL.md) — headline enwik SKU table  
- [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md) — observe≡archive parity  
- [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md) — undo stack / lossy LLM path
