# Multi-View Online Training Plan

**Status:** Phase 0 + 1a implemented (2026-05-31); see [`FINDINGS_CYPHALM_TRAINING.md`](FINDINGS_CYPHALM_TRAINING.md); next **1b** (`schedule_b` hyperparam sweep @ 70k–150k) and **1c** full-corpus eval  
**Last updated:** 2026-06-01  
**Owner track:** CyphaLM (Phase 1) → CyphaDIF (Phase 2)

This document is the canonical plan for **structure-preserving multi-view online training**: presenting the same underlying data in different traversals each epoch, with explicit **view metadata**, so Cypha’s online routing, replay, drift, and fast/slow memory are used deliberately—not as afterthoughts.

Related: [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) (current numbers and priorities), [`cypha_bench/README.md`](../cypha_bench/README.md), [`cypha_lm/README.md`](../cypha_lm/README.md).

---

## Thesis

Cypha’s advantage is **online worldview formation**: routing, replay, drift detection, expert growth, and fast/slow memory. Current training mostly feeds **one static stream** (fixed token order for CyphaLM; round-robin or random interleave for D16 multitask).

**Proposal:** Treat **structure-preserving reorderings** as first-class **views** of the same data. Each training pass uses a different view; the system is told *how* to interpret the stream (`view_id`, memory policy). Local structure (char n-grams within blocks; label–feature pairs for classification) is preserved; **presentation** changes.

**Goal:** Improve LM compression and DIF generalization by exploiting online machinery—not by copying char-LSTM single-pass backprop.

---

## Design principles

| Principle | Meaning |
|-----------|---------|
| **Preserve structure at the right scale** | LM: local char order inside blocks. DIF: never permute labels away from features. |
| **View = (transform, metadata)** | Each pass has a `view_id` and rules the system can learn. |
| **Control memory carry** | Reset fast/local state at boundaries; optionally carry slow/world state across views. |
| **Measure view benefit** | Always compare vs same-order multi-epoch baseline. |
| **Fail loud** | Destructive views (e.g. random token shuffle) should worsen metrics—validates the harness. |

### Data flow (conceptual)

```
Fixed corpus / dataset
        │
        ▼
  View schedule (epoch → ViewSpec[])
        │
        ├── view_id / task signal
        ├── fast memory (SSM / local context) — reset per block?
        ├── slow memory (world prior, GRIA counts, DIF field) — carry?
        └── routing / experts / replay
        │
        ▼
  Predict + online update
```

---

## Phase 0 — Shared infrastructure

**Duration:** ~1–2 weeks  
**Gate:** Unit tests on transforms; block shuffle preserves token multiset.

### 0.1 New module: `cypha_views/`

```
cypha_views/
  __init__.py
  types.py          # ViewSpec, ViewSchedule, MemoryPolicy
  transforms.py     # block_shuffle, rotate_start, reverse, interleave_tasks, ...
  schedule.py       # epoch → list[ViewSpec], seeded RNG
  runner.py         # OnlineViewRunner: stream samples + metadata
```

**Core types:**

- `ViewSpec(name, transform_fn, memory_policy, view_id)`
- `MemoryPolicy`: `{reset_fast, carry_slow, carry_dif, carry_gria_bias}` (names map to LM vs DIF backends)

### 0.2 Benchmark integration

- **Option A:** New domain **D18** — `d18_multi_view_online.py` (cross-cutting view eval)
- **Option B:** Flags on existing domains: `CYPHA_VIEW_SCHEDULE=lm_block_shuffle_v1`

### 0.3 Metrics (all experiments)

| Metric | Purpose |
|--------|---------|
| Primary | BPC (LM) / accuracy / R² (DIF) |
| Per-view ablation | Each view alone vs full schedule |
| Cross-view forgetting | Performance on view 0 after training view *k* |
| Routing diversity | Active experts, routing entropy, `drift_score()` |
| Baselines | n-gram + char-LSTM (LM); SGD/LR (DIF) |

### 0.4 Documentation

- This file (living doc; update **Status** and results tables as phases complete)
- [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) hypothesis ledger + priorities

---

## Phase 1 — CyphaLM multi-view training

**Duration:** ~2–3 weeks  
**Current baseline (40k, `gria_ngram`):** D17 **4.154** BPC (beats trigram **4.398**; +0.24 vs bigram **3.914**). Char-LSTM **3.589**.

### 1.1 View taxonomy (language)

Start conservative; increase aggression only if metrics improve.

| ID | Name | Transform | Local structure preserved | Cypha hook |
|----|------|-----------|-------------------------|------------|
| **LM-V0** | Baseline | Forward contiguous | Full sequence | Current `train_epochs` (same order) |
| **LM-V1** | Block forward | Sentence/paragraph blocks, fixed order | Within-block | `reset_context()` at block boundary |
| **LM-V2** | Block shuffle | Permute blocks (seed per epoch) | Within-block | `view_id` → LM CyphaDIF routing |
| **LM-V3** | Rotated start | Begin at token `k % len` | Contiguous runs | New (context, next) pairs each epoch |
| **LM-V4** | Bidirectional | Even epochs forward; odd reverse | Reversed local chains | `view_id` = `forward` / `backward` |
| **LM-V5** | Staggered windows | Offsets 0, Δ, 2Δ, … | Overlapping windows | Multiple sub-streams |
| **LM-V6** | Curriculum | V1 → V2 → V4 over macro-epochs | Progressive | LR decay per stage |

**Block definition:**

- **Primary:** newline-delimited lines / paragraphs (WikiText, Gutenberg)
- **Fallback:** fixed 256- or 512-char chunks with overlap metadata

**Do not** shuffle individual tokens in early experiments.

### 1.2 Model changes (`cypha_lm`)

| Change | Purpose |
|--------|---------|
| `train_sequence(ids, view_spec=None)` | Accept view transform + metadata |
| `view_id` in config or per-epoch | Signal to `dif.train_step` / optional view embedding |
| `MemoryPolicy` per view | Reset SSM at blocks; optional carry of GRIA Laplace counts |
| `view_schedule` in profile JSON | e.g. `["forward", "block_shuffle", "rotated"]` |
| Optional **view embedding** | Small table concat to GRIA input (extends `gria_ngram` pattern) |

### 1.3 Schedules to benchmark

| Schedule | Composition | Notes |
|----------|-------------|-------|
| **A (minimal)** | `[forward, block_shuffle]` × 2 macro-epochs | First real test |
| **B** | `[forward, block_shuffle, rotated]` × 1 each | Coverage |
| **C (curriculum)** | forward → block_shuffle → bidirectional | Staged |
| **D (ablation)** | forward + **carry slow** (no SSM reset between blocks) | Tests memory policy |

Compare each to current `gria_ngram` + `train_epochs=2` **same order**.

### 1.4 Evaluation (D04 + D17)

| Run | Configuration |
|-----|---------------|
| Fast smoke | `CYPHA_BENCH_FAST=1`, schedules A–B |
| Full 40k | Default limits, schedules A–C |
| Full corpus | `CYPHA_BENCH_FULL_CORPUS=1`, best schedule from 40k |
| Ablations | Per-view BPC; memory carry on/off |

**Success criteria:**

| Tier | Criterion |
|------|-----------|
| **1** | BPC &lt; bigram on D17 or D04 at 40k |
| **2** | Stable trigram margin ≥ 0.1 BPC below trigram |
| **3** | Routing entropy ↑; LM CyphaDIF uses &gt;1 expert meaningfully on block_shuffle |

### 1.5 Implementation order (LM)

1. `cypha_views/transforms.py` — block split, shuffle, rotate (pure `list[int]` functions)
2. `CyphaLM.train_sequence` — loop views from schedule
3. Wire `view_id` into LM `CyphaDIF` during training
4. Profile fields: `view_schedule`, `memory_policy` in `cyphalm_d17_wikitext.json`
5. D17 experiment **17E_multi_view** + learning curve per view
6. Sweep: 3–5 schedules × D17 full 40k

---

## Phase 2 — CyphaDIF multi-view training

**Duration:** ~4–6 weeks  
**Scope:** Main classifier `Cypha.py` `CyphaDIF` / `DIFRegressor`; LM field in `cypha_lm/expert_field/cypha_dif.py` partially wired in Phase 1.

### 2.1 View taxonomy (classification / regression)

| ID | Name | What is reordered | Preserved | Cypha mechanism |
|----|------|-------------------|-----------|-----------------|
| **DIF-V0** | IID stream | Nothing | — | Baseline |
| **DIF-V1** | Class-block shuffle | Order of class blocks | All (x, y) pairs | Expert routing |
| **DIF-V2** | Difficulty curriculum | Easy → hard (margin / loss) | Labels | Replay priority |
| **DIF-V3** | Replay interleave | Live + replay batches | Pair integrity | `PriorityReplayBuffer` |
| **DIF-V4** | Feature presentation | Same x, scaled/noisy/rotated views | Label | Encoder + OOD gate |
| **DIF-V5** | Task permutation | Multitask block order | Per-task pairs | Task ID routing (D16) |
| **DIF-V6** | OOD interleave | Inliers + boundary samples | Labels on inliers | `anomaly_score`, active learning |
| **DIF-V7** | Temporal reversal | Time-series windows reversed | Window labels | D10 / D11 |

### 2.2 Domain mapping

| Bench domain | Primary views | Rationale |
|--------------|---------------|-----------|
| **D01** stat baselines | V1, V2, V4 | Controlled; XOR still needs Kernel LLR |
| **D03** classification | V1, V2, V3 | Tabular online; replay underused |
| **D08** MNIST/HOG | V4, V2 | Presentation + curriculum |
| **D09** documents | V1, V5 | Paragraph/block shuffle |
| **D10** time series | V7, V5 | Temporal views |
| **D11** RL | V7, V3 | Trajectory segments + replay |
| **D16** multitask | V5 (extend `multitask_stream`) | Already has round_robin / random |
| **D02/D05/D07** regression | V1, V3 via `DIFRegressor` | Permuted streams + replay |
| **D12** anomaly | V6 | OOD interleaving |
| **D15** robustness | V4 | Noise as presentation view |

### 2.3 `Cypha.py` changes

| Component | Extension |
|-----------|-----------|
| `train_step(x, label, view_id=None)` | View tag in context prior or namespace |
| `PriorityReplayBuffer` | Tag samples with `view_id`; replay as its own view |
| `WorldPrior.drift_score()` | Log drift per view |
| `multitask_stream()` (D16) | Generalize to `view_stream(dataset, ViewSchedule)` |
| `everyday_profile.json` | Optional `view_schedules` per regime |

**DIFRegressor:** Training labels may use `view_id` prefix during training (`"view2_class3"`), collapsed at eval—test whether this reduces D16B-style forgetting.

### 2.4 Phase 2a — D16 prototype (first)

D16 already has `multitask_stream(interleave=round_robin|random)`. Add:

| Variant | Description |
|---------|-------------|
| **V5a** | Fixed task blocks; **permute task-block order** each macro-epoch |
| **V5b** | Within-task sample shuffle (labels preserved) |
| **V5c** | Stream tuples `(x, y, task_id, view_id)` |

**Success:** Forgetting score ↓ vs D16B **or** per-task accuracy ↑ at equal step count; `drift_score` stable.

### 2.5 Phase 2b — Four representative domains

1. **D03** — tabular (V1, V2, V3)  
2. **D08 HOG** — vision (V4, V2)  
3. **D09** — documents (V1, V5)  
4. **D02 or D05** — regression (V1, V3)

**Success:** ≥2/4 domains improve ≥1 pp accuracy (or equivalent ΔR²).

### 2.6 Phase 2c — Cross-domain report

Artifact: `cypha_bench/report/tables/cross_view_gain_matrix.json`

- Rows: domains  
- Columns: view schedules  
- Cells: Δ vs DIF-V0 baseline  

---

## Phase 3 — Product integration (after Phase 1–2 validate)

| Piece | Description |
|-------|-------------|
| `CYPHA_VIEW_SCHEDULE` env | Path to JSON schedule file |
| Studio / REST | Optional `view_id` on `/update` for deliberate adaptation |
| Checkpointing | Per-view expert banks (D16F-style isolation) |
| Paper | Section: structure-preserving online views (LM + D16 + D03) |

---

## Execution roadmap

| Step | Work | Gate before next |
|------|------|------------------|
| **0** | `cypha_views` + types + tests | Transforms preserve content multiset |
| **1a** | LM-V1, LM-V2 + same-order control | D17 fast: schedule beats same-order `train_epochs=2` |
| **1b** | LM-V3, LM-V4 + `view_id` routing | D17 40k: BPC ↓ ≥0.05 vs **4.154** |
| **1c** | Best schedule + full corpus | BPC &lt; bigram on D17 or documented blocker |
| **2a** | D16 view streams | Forgetting ↓ or ARI ≥ baseline |
| **2b** | D03 + D08 + D09 + regression | ≥2/4 domains improve |
| **2c** | Profiles + D18 summary + CI smoke | `RESEARCH_STATUS` updated |
| **3** | Studio/REST + paper | User-facing story |

### First execution slice (recommended start)

1. Implement `cypha_views/` (`block_shuffle`, `rotate_start`, `ViewSchedule`)
2. Extend `CyphaLM.train_sequence`; add `view_schedule` to D17 profile
3. Add D17 experiment **17E_multi_view** (Schedule A vs baseline)
4. Update results in this doc and `RESEARCH_STATUS.md`
5. Port same runner API to **D16** (V5a/V5b)

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Block shuffle too coarse → no gain | Tune block size; sentence vs paragraph |
| Block shuffle too fine → breaks LM | Never below sentence without explicit rules |
| GRIA overwrite across views | LR decay; Laplace prior; per-view checkpoints |
| View-as-class explodes DIF class count | Collapse at eval; cap view-specific experts |
| False wins on 3k fast mode | Gate promotions on 40k + full corpus |
| Scope creep | Finish Phase 1 before Phase 2b; D16 before D03/D08/D09 |

---

## Out of scope (for now)

- Random **token** permutation (LM)
- PyTorch char-LSTM parity (NumPy baseline stays eval-only)
- Kernel LLR (separate Priority 1—views do not fix XOR alone)
- Full 17-domain re-bench until views validated on D17 + D16

---

## Results log

_Update this section as phases complete._

| Phase | Date | Result | Notes |
|-------|------|--------|-------|
| **0** | 2026-05-31 | ✅ Done | `cypha_views/` package + 6 unit tests |
| **1a** | 2026-05-31 | ✅ Fast | D17 **17E**: `schedule_a` **4.329** vs `same_order` **4.399** BPC (3k, Δ **−0.07**) |
| **1a sweep** | 2026-05-31 | ✅ 32-run grid | **n_train × view** sweep — see [`cyphalm_view_iteration_sweep.json`](../cypha_bench/config/cyphalm_view_iteration_sweep.json) |
| **2a** | 2026-05-31 | ⚠ Mixed | D16 **16G** fast: task-block-shuffle **hurts** accuracy (0.58 mean vs RR 0.81); forgetting 0.0 but wine/digits collapse — needs tuning |
| **1** | 2026-05-31 | ✅ Core | `CyphaLM.train_sequence_views()`, profile `view_schedule=same_order` on D17 |

### Iteration × view sweep (D17 17F, 2026-05-31)

Grid: `n_train` 2k–40k × `same_order_e1`, `same_order_e2`, `schedule_a`, `schedule_b`. Artifact: `cypha_bench/config/cyphalm_view_iteration_sweep.json`.

| Finding | Detail |
|---------|--------|
| Multi-view wins mid-train | `schedule_b` beats `same_order_e2` at every **n_train ≤ 32k** (16k: 4.230 vs 4.357 BPC) |
| 40k same-order global best | `same_order_e2` @ 40k: **4.067** BPC (+0.15 vs bigram) |
| Overtrain on one pass | `same_order_e1` crosses **above** bigram between 12k–16k |
| schedule_b sweet spot | Best at **32k** (4.121 BPC), not 40k |

**Recommendation:** `schedule_b` for training ≤24k tokens; `same_order` + 2 epochs only when committing to full 40k. Early-stop ~12–16k with multi-view when targeting bigram.

Re-run: `python cypha_bench/tuning/cyphalm_view_iteration_sweep.py --write`

### Convergence limit sweep (2026-06-01)

Extended to **250k tokens** (10M-token WikiText cap). Artifact: `cyphalm_convergence_limit.json`. Re-run: `python cypha_bench/tuning/cyphalm_convergence_limit.py --write`

| Mode | Training limit | Best BPC | @ n_train | vs bigram | Status |
|------|----------------|----------|-----------|-----------|--------|
| **same_order_e2** | **~40k** (overtrain by 50k) | **4.094** | 40k | +0.18 | **Converged** — BPC rises after 40k |
| **schedule_b** | **Yes** | **~300k** (worse by 400k) | **3.905** | 300k | +0.34 |

**Prior 40k cap was too low** to see same-order overtrain; it looked like the optimum because the bench default stopped there.

**Recommendation:** `same_order` → **40k max**. `schedule_b` → train to **70k–150k** for best held-out in this run (local min 3.958 @ 150k); full corpus may need **500k+** to plateau.

---

## References

- Online adaptation (existing): D17D ΔBPC ≈ −0.29 on OOD text  
- Multitask stream (existing): `cypha_bench/domains/d16_multitask.py` `multitask_stream()`  
- CyphaLM training (existing): `cypha_lm/model/cypha_lm.py` `train_sequence()`, `train_epochs`  
- Replay (existing): `Cypha.py` `PriorityReplayBuffer`, `replay_ratio`  
- Beat-bigram context: [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) Priority 3  
