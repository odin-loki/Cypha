# CyphaLM holdout shard-merge evaluation

**Date:** 2026-09-20  
**Harness:** `scripts/cyphalm_hp_shard_holdout.sh` → `cyphalm_hp_shard_spike --merge-profile both`  
**Settings:** `table_bits=16`, `holdout_frac=0.2`, `n_shards=2`, `boundary_replay_bytes=4096`

---

## Executive summary

Parallel shard train + weighted table merge is **misleading on in-sample BPC** (~0.69) but **measurably worse than single-stream on holdout** with naive merge (+0.06–0.14 BPC). That gap is **structural**: path-dependent hp state (match rings, wordstream, mixer trajectory) is not mergeable across cold shards.

**What helps on holdout (measured):**

| Technique | alice29 Δ vs single | enwik 1 MiB Δ vs single | Notes |
|-----------|---------------------|-------------------------|-------|
| Baseline weighted merge + boundary replay | **+0.1378** | **+0.0598** | PR #15 reproduction |
| Confidence gating (`min_counter_n=8`) | +0.1373 | +0.0599 | Marginal |
| Distance-weighted multi-pass replay | +0.1381 | +0.0598 | No gain |
| Small bridge fine-tune (4096 B) | +0.1413 | +0.0594 | Hurts alice |
| **Full-train bridge fine-tune** | **+0.0585** | **+0.0221** | Best single lever |
| **Recommended (`holdout` profile)** gated + full-train bridge | **+0.0592** | **+0.0221** | Use for production spike |

**Verdict:** Shard-merge **cannot beat single-stream on holdout** with table merge alone, but **full-train sequential bridge adapt** after merge closes **57–63%** of the holdout gap. Remaining Δ (+0.02–0.06) reflects irreducible path-dependent divergence from parallel cold-shard training.

---

## Honest holdout numbers (2026-09-20, Linux cloud-agent)

| Corpus | single_stream_holdout | merged baseline | Δ baseline | merged improved | Δ improved |
|--------|----------------------|-----------------|------------|-----------------|------------|
| alice29.txt | **1.8072** | 1.9449 | **+0.1378** | **1.8663** | **+0.0592** |
| enwik8.8mb (1 MiB slice) | **1.6349** | 1.6947 | **+0.0598** | **1.6570** | **+0.0221** |

Negative Δ vs single-stream would mean merged beats sequential training on unseen bytes — **not observed**.

---

## Structural limits

1. **Counters / StateMap** — additive and mergeable with byte-weighted averaging, but shards see **disjoint context keys** at boundaries; merged cells are approximate blends.
2. **Hash-slot states** — ContextModel merge already uses **max-evidence** pick (not average); gating low-count slots helps marginally.
3. **Match rings / wordstream / GRIA** — **not mergeable**; only recoverable by **re-consuming** train bytes on the merged predictor.
4. **Mixer / hedge / APM** — online trajectory-dependent; weighted merge is approximate; full-train bridge adapt partially realigns them.

**Implication for N-worker training:** treat shard merge as a **checkpoint**, then run a **sequential bridge pass** over the full train prefix before holdout eval or serve freeze. Pure merge-without-replay will not match single-stream quality.

---

## Recommended API (`hp/shard_merge.hpp`)

```cpp
hp::ShardMergeOptions opts = hp::holdout_merge_options();  // min_counter_n=8, min_statemap_count=4
hp::BoundaryReplayConfig replay = hp::holdout_boundary_replay_config(replay_bytes, train_bytes);
hp::prepare_merged_predictor_for_holdout(pred, train_bytes, train_n, shard_boundaries, replay);
```

`bridge_finetune_bytes = train_bytes` (full train prefix) is the dominant improvement.

---

## Reproduce

```bash
bash scripts/cyphalm_hp_shard_holdout.sh bench/data/canterbury/alice29.txt --boundary-replay-bytes 4096
bash scripts/cyphalm_hp_shard_holdout.sh bench/data/enwik8/enwik8.8mb --max-bytes 1048576 --boundary-replay-bytes 4096

# Component ablation
native/build-shard-holdout/cyphalm_hp_shard_spike --corpus <path> --merge-profile ablation \
  --boundary-replay-bytes 4096 --holdout-frac 0.2 --table-bits 16
```

Artifacts: `docs/reports/CYPHALM_SHARD_HOLDOUT.json`, `CYPHALM_SHARD_HOLDOUT_ENWIK1MB.json`.
