# CyphaLM lossy LLM execution plan

**Date:** 2026-09-19  
**Status:** Phased roadmap from lossless hp mixer → serve-time probability distribution with less RAM  
**Inputs:** [`CYPHALM_LLM_PROFILE_REPORT.md`](CYPHALM_LLM_PROFILE_REPORT.md), [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md), [`CYPHALM_LLM_EVAL.md`](CYPHALM_LLM_EVAL.md)

---

## Measured bottlenecks (this VM, light profile, `table_bits=22`)

| Bottleneck | Measurement | Implication |
|------------|-------------|-------------|
| **256-clone / full-vocab score** | `predict_next` **~37–41 s/call**; legacy 256-clone **~36–47 s/call** | REST `/predict_next` and generation are unusable at scale without trie / undo / partial expansion |
| **Bit-tree checkpoint DFS (new default)** | **~39–40 s/call**; parity Δ=0 vs legacy | Prefix-shared scoring works; `assign_from` backtrack still dominates — needs true undo |
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

## Phase 0 — Done (this PR stack)

| Item | Status |
|------|--------|
| `eval_bpc` → bit-serial observe (compress-equivalent) | **Done** |
| `next_byte_log_probs` → MSB bit-tree default (not 256× clone) | **Done** |
| Checkpoint-pool DFS on `scratch_` (no per-fork `clone_from`) | **Done** |
| `sample_next_byte` / generation bit-serial on one scratch | **Done** (pre-existing) |
| `gate24` CMake profile + `apply_hp_gate24_recipe()` | **Done** |
| Large-n observe eval harness + report | **Done** |
| Legacy 256-clone opt-in: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1` | **Done** |

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

---

## Phase 2 — Freeze tables for serve (train / serve split)

**Goal:** Train-online on mutable counts; export immutable snapshot for REST `/predict_next` and `generate_decode`.

| Field | Detail |
|-------|--------|
| Expected RAM | Serve path: one RO predictor; no DFS checkpoint pool |
| Expected speed | Faster init; enables mmap sharing across workers |
| Quality risk | **Medium** — stale snapshot vs online adapter |
| Experiment | `CyphaLMModel::save` → load RO `HpSequenceBackend`; WikiText observe BPC online vs frozen after N train bytes |

---

## Phase 3 — Lower `table_bits` / slot LLM SKU

**Goal:** Product tiers below champ RAM (e.g. `table_bits=18–20`, `slot_max=20`).

| Field | Detail |
|-------|--------|
| Expected RAM | **−4–16×** vs mem 22 (hp harness axis) |
| Quality risk | **Medium** — measure WikiText observe BPC only |
| Experiment | Sweep in `cyphalm_llm_eval`; plot RSS vs BPC |

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
| CI `native_hp_bit_tree_smoke` | Parity gate (in CTest this PR) |
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
| Top-M partial (phase 4+) | **Lossy** | No |
| Frozen serve snapshot | Yes if refreshed | Yes if same flags + fresh enough |

---

## Related

- [`CYPHALM_LLM_EVAL.md`](CYPHALM_LLM_EVAL.md) — large-n measured numbers  
- [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md) — protocol / SKU / enwik archive  
- [`native/cmake/HpFlags.cmake`](../../native/cmake/HpFlags.cmake) — light / gate24 / champ  
- [`bench/config/profiles/cyphalm_hp_gate24.json`](../../bench/config/profiles/cyphalm_hp_gate24.json)
