# CyphaLM lossy LLM execution plan

**Date:** 2026-09-20, updated 2026-09-23
**Status:** Phase 0 done; mem tier / serve-compact / cold-slot prune landed 2026-09-20. **2026-09-23:** per-component lossy knobs and measured tiers (`lean` beats gate24 on enwik8: −0.0019 bpc, −30% RAM, faster; `compact` −56% RAM for +0.0057), serve-path leak fixed, bit-tree DFS ~1.85× faster, demand-zero tables. Full write-up: [`CYPHALM_LOSSY_MIXER_REPORT.md`](CYPHALM_LOSSY_MIXER_REPORT.md).  
**Inputs:** [`CYPHALM_LLM_PROFILE_REPORT.md`](CYPHALM_LLM_PROFILE_REPORT.md), [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md), [`CYPHALM_LLM_EVAL.md`](CYPHALM_LLM_EVAL.md), [`CYPHALM_LOSSY_BENCH_RESULTS.json`](CYPHALM_LOSSY_BENCH_RESULTS.json), [`CYPHALM_LOSSY_ENWIK_SCREEN.json`](CYPHALM_LOSSY_ENWIK_SCREEN.json)

---

## Ranked lossy levers (gate24 base, quality bar ~1.61 enwik BPC)

Priority = expected **RAM/speed payoff** vs **quality risk** at gate24 compile profile (`v78_flags.ps1` + `HP_SLOT_MAX=24`). All levers keep gate24 recipe; lossy tiers are runtime/config overlays.

| Rank | Lever | API / flag | RAM | Speed | Quality risk | Status |
|------|-------|------------|-----|-------|--------------|--------|
| **0** | **Per-component tiers** (drop 8 wiki CMs, pool 8 slots, capped match/pool/CM tables) | `apply_hp_lossy_tier(cfg, tier)`, tier = `lean` / `compact` / `small` / `CYPHA_HP_LOSSY_TIER` | **−30%** (lean) … **−56%** (compact) peak RSS | Faster observe | lean **−0.0019 BPC** (better); compact **+0.0057** | **Done** ([report](CYPHALM_LOSSY_MIXER_REPORT.md)) |
| **1** | **Lower `table_bits` (mem tier)** | `apply_hp_lossy_recipe(cfg, mem)` / `CYPHA_HP_LOSSY_MEM=20` | **−38%** RSS @ mem20 enwik | Faster observe + clones | **+0.0063 BPC** on enwik8MB — **fails ~1.612 bar** | **Opt-in only** |
| **2** | **Single-predictor serve** | delta undo on live `pred_` (no `scratch_` twin) | **−~50%** construct RSS (~1.5 GiB vs ~3.1 GiB dual) | Neutral | **None** (identical math) | **Done** (this PR) |
| **3** | **Cold hash-slot prune** | `hp_prune_cold_min_n` / `CYPHA_HP_PRUNE_COLD_MIN_N=4` / `prune_hp_cold_slots()` | No table shrink (fixed arrays) | **~1.7×** faster `predict_next` @ min4 (measured) | **Low** if threshold small; rises with aggressive min | **Implemented** |
| **4** | True undo stack (latency, not lossy) | delta-undo on `hp::Predictor::update` | Drop 9 DFS checkpoints | **~293×** `predict_next` vs pre-undo (measured) | **None** if exact | **Done** (PR #7 + [`GATE24_POST_UNDO_BENCH.md`](GATE24_POST_UNDO_BENCH.md)) |
| **5** | Frozen serve snapshot | train online → export RO `HpSequenceBackend` | One RO predictor + mmap | Faster init; multi-worker share | **Med** (stale vs online) | Planned |
| **6** | Quantized mixer / APM | `HP_MIXER_WT16` (hp compile) | **−50%** mixer RAM | Neutral | **Low** if byte-identical proxy holds | hp upstream option |
| **7** | Top-M partial expansion | expand top-M bytes only | Neutral | **~256/M×** generation | **Med–high** | Planned (phase 4) |
| **8** | Distillation to smaller tables | train gate24 → export mem18 weights | Same as mem tier | Same as mem tier | **Med** | Research |
| **9** | Slot / count floors (compile) | lower `HP_SLOT_MAX` SKU | Large at slot 35 | Slower at high slot | **Med** | gate24 fixed @24 |
| **10** | mmap shared RO tables | post-freeze snapshot | **~N×** workers | Neutral | **Low** | Planned (phase 6) |

**Quality bar:** gate24 enwik8MB observe/archive **~1.612 BPC** must not be destroyed for production lossy tiers. WikiText online observe is a **relative** screen only (absolute BPC ≠ enwik archive).

---

## Measured first cut (2026-09-20, `cyphalm_lossy_bench`)

Corpus: WikiText-2 train slice, 32k warmup + 50k observe eval, 1× `predict_next` latency. Full JSON: [`CYPHALM_LOSSY_BENCH_RESULTS.json`](CYPHALM_LOSSY_BENCH_RESULTS.json).

| Variant | RSS init (MiB) | Δ BPC vs baseline | observe B/s | predict_next (s) |
|---------|----------------|-------------------|-------------|------------------|
| gate24 baseline mem22 | 3053 | — | 14.5k | 175.5 |
| **lossy mem20** | 1400 | +0.00012 | 20.7k | **36.6** |
| serve_compact mem22 | 1538 | 0 | 18.1k | 187.2 |
| prune_cold min4 mem22 | 3053 | 0 | 16.6k | 105.4 |
| combo mem20+compact+prune4 | 1708 | +0.00012 | 21.1k | **63.1** |

**Takeaways (WikiText — relative screen only):**
- **mem20** looked negligible on WikiText (+0.00012 BPC) — **misleading**; see enwik screen below.
- **serve_compact** cuts ~50% construct RSS with **zero** BPC change on WikiText; lazy scratch recreate adds first-call latency.
- **prune_cold_min4** after 32k warmup: no measurable BPC hit on WikiText; speeds `predict_next` ~1.7×.

---

## enwik gate24 screen (2026-09-20, authoritative)

Full **enwik8.8MB** compress-faithful bit-serial observe (`eval_bpc`, 8,388,608 bytes). SHA256 `09f6dd7241a8ae21edfd6762f3c6712a1fd02f7f322c5e77cab8bb88f292ee8e`. JSON: [`CYPHALM_LOSSY_ENWIK_SCREEN.json`](CYPHALM_LOSSY_ENWIK_SCREEN.json).

| Variant | observe BPC | Δ vs mem22 baseline | vs ~1.612 bar | RSS init (MiB) |
|---------|-------------|----------------------|---------------|----------------|
| **gate24 baseline mem22** | **1.611729** | — | on bar | 3083 |
| lossy mem20 | 1.618017 | **+0.00629** | **+0.00602** (fail) | 1913 (−38%) |
| combo mem20+compact+prune4 | 1.618017 | **+0.00629** | fail | 1913 |

**Verdict:** **mem20 does not pass the gate24 quality bar.** Baseline mem22 matches the cited Cypha gate24 observe reference (1.611729). mem20 adds **+0.0063 BPC** (~0.39% relative) — material for compression, not a blow-up, but above the ~1.612 production threshold. compact+prune on mem20 add **no extra** BPC loss beyond mem20 alone.

**Production default:** keep **mem22** gate24. Safe serve levers on mem22:
- `hp_serve_compact` (lossless RAM, −~50% construct RSS on WikiText)
- `hp_prune_cold_min_n` (re-screen per threshold; 0 Δ on WikiText @ min4)

**mem20** remains an **opt-in lossy tier** (`CYPHA_HP_LOSSY_MEM=20`) when −38% RSS is worth +0.0063 BPC.

Reproduce:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build native/build --target cyphalm_lossy_bench
# WikiText relative screen
native/build/cyphalm_lossy_bench --warmup-n 32768 --eval-n 50000 --latency-iters 1
# enwik gate24 screen (full 8MB, ~20 min)
native/build/cyphalm_lossy_bench --enwik-screen
```

---

## API surface (this PR)

| Config field | Env var | Effect |
|--------------|---------|--------|
| `hp_lossy_mem` | `CYPHA_HP_LOSSY_MEM` | Override effective `table_bits` (16–24) |
| `hp_serve_compact` | `CYPHA_HP_SERVE_COMPACT=1` | Drop `scratch_` + DFS checkpoints at init |
| `hp_prune_cold_min_n` | `CYPHA_HP_PRUNE_COLD_MIN_N` | Prune hash slots with state count `< N` after warmup |

```cpp
apply_hp_lossy_recipe(cfg, 20);   // gate24 compile + mem20 tables
cfg.hp_serve_compact = true;
cfg.hp_prune_cold_min_n = 4;
CyphaLMModel model(cfg);
// ... warmup bytes ...
model.prune_hp_cold_slots();      // optional explicit call
model.compact_hp_for_serve();     // idempotent
```

hp internals: `hp::Predictor::prune_cold_hash_slots(min_total)` walks `ctx_chain_` context models + bias counters.

---

## Measured bottlenecks (reference, light profile, `table_bits=22`)

| Bottleneck | Measurement | Implication |
|------------|-------------|-------------|
| **256-clone / full-vocab score (pre-undo)** | `predict_next` **~48 s/call**; legacy 256-clone **~24 s/call** @ 64 KB warm | Superseded by delta undo — see [`GATE24_POST_UNDO_BENCH.md`](GATE24_POST_UNDO_BENCH.md) |
| **Bit-tree + delta undo (current default)** | **~0.16 s/call** `predict_next`; **~0.003 s** greedy | Single `pred_`; parity Δ=0 vs legacy (`hp_bit_tree_smoke`) |
| **Bit-serial observe** | **~30–34k B/s** on WikiText 100k slice | `eval_bpc` / training loss path is fine; this is the compress-equivalent metric |
| **Dual predictor RSS** | Construct **~782 MiB**; peak under load **~1.15 GiB** | Standing `pred_` + `scratch_` + 9 DFS checkpoints adds RAM vs single-predictor serve |
| **Protocol mismatch** | Light observe enwik 8MB **1.721** vs gate24 **~1.612** vs champ **~1.610** | Quality claims require **gate24/champ** SKU, not CI light |

---

## Quality SKUs (compile-time)

| SKU | CMake | v78 flags | `HP_SLOT_MAX` | Role | enwik8MB BPC (measured / cited) |
|-----|-------|-----------|---------------|------|--------------------------------|
| **light** | `-DCYPHA_HP_PROFILE=light` | 0/78 | 24 | **CI default**, fast/dev | **~1.721** observe (gap report) |
| **gate24** | `-DCYPHA_HP_PROFILE=gate24` | 78/78 | 24 | PLAN 8MB quality screen | **~1.612** archive (gap report) |
| **champ** | `-DCYPHA_HP_PROFILE=champ` | 78/78 | 35 | User quality bar | **~1.610** (encyclopedia; needs ≥32 GiB to compress) |

Default stays **light** for CI RAM. Document any quality headline with SKU + metric (observe vs clone-API).

---

## Phase 0 — Done

| Item | Status |
|------|--------|
| `eval_bpc` → bit-serial observe (compress-equivalent) | **Done** |
| `next_byte_log_probs` → MSB bit-tree default (not 256× clone) | **Done** |
| Checkpoint-pool DFS on `scratch_` (no per-fork `clone_from`) | **Done** |
| `sample_next_byte` / generation bit-serial on one scratch | **Done** |
| `gate24` CMake profile + `apply_hp_gate24_recipe()` | **Done** |
| Large-n observe eval harness + report | **Done** |
| Legacy 256-clone opt-in: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1` | **Done** |

---

## Phase 0b — Done (this PR): first lossy levers

| Item | Status |
|------|--------|
| `apply_hp_lossy_recipe` + `CYPHA_HP_LOSSY_MEM` | **Done** |
| `hp_serve_compact` + `compact_for_serve()` | **Done** |
| `prune_cold_hash_slots` on hp context chain + Cypha API | **Done** |
| `cyphalm_lossy_bench` harness + measured JSON | **Done** |
| Ranked lever table (this doc) | **Done** |

---

## Phase 1 — True undo stack (attack #1 latency)

**Goal:** One mutable `scratch_`; record bit-update deltas; `undo()` after each DFS sibling instead of `assign_from` checkpoint copy.

| Field | Detail |
|-------|--------|
| Expected speed | **~10–50×** faster full-vocab score vs checkpoint DFS (profile report estimate) |
| RAM | Drop 9 DFS checkpoint predictors; single scratch only |
| Quality risk | **Low** if undo is exact inverse of `hp::Predictor::update` |
| Experiment | Implement undo in `hp_backend` or thin hp wrapper; parity vs `hp_bit_tree_smoke` on 1k contexts; target `predict_next` < 1 s @ mem 22 |

**Gate:** `hp_bit_tree_smoke` + extended random grid; latency regression doc in `CYPHALM_LLM_EVAL.md`.  
**Coordination:** delta-undo may land on PR #7 — keep branch independent.

---

## Phase 2 — Freeze tables for serve (train / serve split)

**Goal:** Train-online on mutable counts; export immutable snapshot for REST `/predict_next` and `generate_decode`.

| Field | Detail |
|-------|--------|
| Expected RAM | Serve path: one RO predictor; no DFS checkpoint pool |
| Expected speed | Faster init; enables mmap sharing across workers |
| Quality risk | **Medium** — stale snapshot vs online adapter |
| Experiment | `CyphaLMModel::save` → load RO `HpSequenceBackend`; WikiText observe BPC online vs frozen after N train bytes |

**Note:** `hp_serve_compact` (phase 0b) is a RAM subset of this — drops auxiliary predictors but tables still mutate.

---

## Phase 3 — Lower `table_bits` / slot LLM SKU

**Goal:** Product tiers below champ RAM (e.g. `table_bits=18–20`, `slot_max=20`).

| Field | Detail |
|-------|--------|
| Expected RAM | **−4–16×** vs mem 22 (hp harness axis) |
| Quality risk | **Medium** — measure WikiText observe BPC only |
| Experiment | Sweep in `cyphalm_llm_eval`; plot RSS vs BPC |

**Status:** mem20 tier implemented; enwik screen shows +0.0063 BPC — opt-in only, mem22 stays default.

---

## Phase 4 — Top-M partial expansion (lossy infer)

**Goal:** Expand only top-M byte candidates (cheap prior); exact hp score on M ≪ 256.

| Field | Detail |
|-------|--------|
| Expected speed | **~256/M×** generation and REST |
| Quality risk | **Medium–high** — tail bytes missed |
| Experiment | M ∈ {16, 32, 64}; WikiText top-1/top-10 vs full 256; document lossy BPC separately |

Uses bit-tree naturally: stop DFS when outside top-M prefix support.

---

## Phase 5 — Lossy top-M renormalize + sampling

**Goal:** Renormalize distribution over top-M for temperature / top-p sampling.

| Field | Detail |
|-------|--------|
| Quality risk | **High** for calibration (WikiText entropy gap already +0.5–0.7 bits on clone-API) |
| Experiment | Greedy/temp samples M=32 vs full 256; label as **lossy** in API responses |

---

## Phase 6 — mmap shared RO tables

**Goal:** Post-phase-2 snapshot backed by `mmap`; N REST workers share physical pages.

| Field | Detail |
|-------|--------|
| Expected RAM | **~N× reduction** for N workers |
| Quality risk | **Low** (identical math if frozen) |
| Experiment | Two `cypha_rest` workers; `VmRSS` + BPC parity |

---

## Phase 7 — BPE outer / byte inner (API ergonomics)

**Goal:** User-facing BPE tokens; inner loop stays byte-level hp.

| Quality note | Byte BPC ≠ BPE perplexity — report separately |

---

## Evaluation harness follow-ups

| Suite | Target |
|-------|--------|
| CI `native_hp_bit_tree_smoke` | Parity gate (in CTest) |
| `cyphalm_lossy_bench` | RSS / BPC / predict_next deltas per lever |
| `cyphalm_lossy_bench --enwik-screen` | Full enwik8MB gate24 quality gate for mem tiers |
| Nightly `cyphalm_llm_eval` @ n≥100k | Observe BPC drift per SKU |
| Top-k @ n≥64 | After phase 1 undo (currently ~37 s/step) |
| gate24 / champ enwik8MB | Host with corpus + ≥32 GiB for champ compress |
| `BASELINE_LOCK.json` | Optional hp observe pins per SKU (not in this PR) |

---

## API labeling (avoid metric confusion)

| API / metric | Lossless? | Valid compression BPC? |
|--------------|-----------|------------------------|
| `eval_bpc()` / observe | Yes | **Yes** (= hp archive @ same flags) |
| `next_byte_log_probs` bit-tree | Yes (dist only) | No (no main-state update) |
| `predict_next` + top-k | Yes (dist) | No — **clone-API class** |
| `hp_lossy_mem` / mem tier | **Lossy RAM** | Re-measure; online observe may match |
| `prune_cold_hash_slots` | **Lossy** | Re-measure |
| Top-M partial (phase 4+) | **Lossy** | No |
| Frozen serve snapshot | Yes if refreshed | Yes if same flags + fresh enough |

---

## Related

- [`CYPHALM_LOSSY_BENCH_RESULTS.json`](CYPHALM_LOSSY_BENCH_RESULTS.json) — WikiText relative screen  
- [`CYPHALM_LOSSY_ENWIK_SCREEN.json`](CYPHALM_LOSSY_ENWIK_SCREEN.json) — enwik8MB gate24 screen (authoritative)  
- [`CYPHALM_LLM_EVAL.md`](CYPHALM_LLM_EVAL.md) — large-n measured numbers  
- [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md) — protocol / SKU / enwik archive  
- [`native/cmake/HpFlags.cmake`](../../native/cmake/HpFlags.cmake) — gate24  
- [`bench/config/profiles/cyphalm_hp_gate24.json`](../../bench/config/profiles/cyphalm_hp_gate24.json)
