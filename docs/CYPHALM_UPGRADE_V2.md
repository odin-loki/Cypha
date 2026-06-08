# CyphaLM Upgrade Track V2 — Learnable Views + Stronger N-Gram Fusion

**Status:** Track A complete (neutral @ 300k); Track B gated fusion implemented  
**Last updated:** 2026-06-07  
**Prerequisite:** Phase 1c full-corpus eval with current profile (`1a16d89`)  
**Related:** [`CYPHALM_ALGORITHM_STUDY.md`](CYPHALM_ALGORITHM_STUDY.md), [`CYPHALM_MODEL_CLASS_RESEARCH.md`](CYPHALM_MODEL_CLASS_RESEARCH.md), [`MULTI_VIEW_TRAINING_PLAN.md`](MULTI_VIEW_TRAINING_PLAN.md)

---

## Context

Sweeps and ablations through **300k tokens** show:

| Finding | Implication |
|---------|-------------|
| Best stack **3.838 BPC** @ 300k | Still **+0.27 vs bigram** |
| `gria_ngram` >> `full` | Keep n-gram path; ignore full DIF+epistemic for char-LM |
| `frozen_alpha` helps | Reduce GRIA capacity; structure matters more than per-token α |
| Fixed `view_id_dim=8` (random table) | Wired but **not learnable** — next lever |
| `ngram_fuse_split` (sum of two linears) | Better than single concat matmul; room for **gated / MLP fusion** |
| Char-LSTM **3.589 BPC** @ 40k | Separate track — see model-class doc |

**Goal:** Close bigram gap with **online-compatible** architectural upgrades before adding a second model class.

---

## Track A — Learnable view embeddings

### Current (V1)

```python
# cypha_lm/model/cypha_lm.py
self._view_embed = fixed random table  # (16 slots, view_id_dim)
v = concat(gria_core, _view_embed[slot])
```

- Slot = `hash(view_spec.view_id) % 16`
- **Not trained** — view signal is static noise unless lucky init
- GRIA `d_input = field_dim + view_id_dim`; BPTT uses `_grad_v_field_core` (view tail stripped)

### Target (V2)

| Component | Design |
|-----------|--------|
| **`ViewEmbedding`** | `cypha_lm/embeddings/view_embed.py` — table `(max_views, d_view)` **online-updated** |
| **Update rule** | After GRIA CE step: `Δe_view = lr_view * grad_v[fd:fd+d_view]` (same lr scale as GRIA or `view_lr`) |
| **Config** | `view_id_dim: 8`, `view_learnable: true`, `view_lr: 0.05`, `max_view_slots: 16` |
| **Schedule mapping** | `forward`→0, `block_shuffle`→1, `rotated`→2, `backward`→3 (stable ids, not hash) |
| **Ablation** | `view_learnable=false` reproduces V1 fixed table |

### Implementation steps

1. Add `ViewEmbedding` with `forward(slot)`, `update(grad_view, lr)`, `get_state` / `set_state`
2. Replace `_view_embed` in `CyphaLM`; use canonical slot ids from `ViewSpec.name`
3. Extend `save`/`load` npz with `view_embed` when learnable
4. D17 experiment **17I_view_learnable** — compare fixed vs learnable @ 40k and 300k under `schedule_b`
5. Success: **≥0.05 BPC drop** vs fixed view @ 300k, or beat bigram

### Risks

- View slots may **overfit** one schedule phase — mitigate with `view_lr_decay` matching `gria_lr_decay`
- Grad through view is tiny if GRIA W down-weights tail — consider **direct auxiliary loss** per view (optional Phase A2)

---

## Track B — Stronger n-gram fusion

### Current (V1)

```text
field_part = W_f @ field_x          # (field_dim, field_dim)
embed_part = W_e @ embed_history    # (field_dim, (1+ngram)*d_embed)
v = field_part + embed_part         # elementwise sum in field space
```

Single linear per branch; no interaction between field and embed streams.

### Target (V2) — options (implement in order)

| Option | Mechanism | Online? | Params |
|--------|-----------|---------|--------|
| **B1 Gated fusion** | `g = σ(W_g @ [field_x; embeds])`; `v = g ⊙ field_part + (1-g) ⊙ embed_part` | ✅ | +1 linear |
| **B2 2-layer MLP** | `v = W2 @ tanh(W1 @ concat(field_x, embeds))` | ✅ | +2 matrices |
| **B3 Position weights** | Learnable `w_0..w_{ngram}` scale each history embed before `W_e` | ✅ | +ngram scalars |
| **B4 Bilinear** | `v = field_part + embed_part + (field_x ⊗ embeds) @ W_b` (low-rank) | ✅ | +low-rank |

**Recommended first:** **B1 gated fusion** + **B3 position weights** (minimal param increase, preserves online updates).

### Config

```python
ngram_fusion: str = "sum"  # sum | gated | mlp2
ngram_position_weights: bool = True
```

### Implementation steps

1. Add `cypha_lm/projection/ngram_fusion.py` — `NgramFusion` module
2. Wire in `_ngram_gria_vector`; CE grads flow to all fusion weights (like GRIA)
3. Unit tests: shape, finite loss, save/load
4. Sweep: `sum` vs `gated` vs `mlp2` @ 40k/300k with best view settings from Track A
5. Success: **≥0.03 BPC** vs `ngram_fuse_split` sum @ 300k

### BPTT note

Backprop through fusion to SSM must use `_grad_v_field_core` then chain through fusion Jacobian (document in `cypha_lm.py`).

---

## Combined experiment matrix

Artifact: `cypha_bench/config/cyphalm_upgrade_v2_sweep.json`  
Script: `cypha_bench/tuning/cyphalm_upgrade_v2_sweep.py` — D17 **17I_view_learnable**

| ID | view | fusion | n_train | BPC | vs fixed |
|----|------|--------|---------|-----|----------|
| baseline | fixed | sum | 40k | **4.023** | — |
| A | learnable | sum | 300k | **3.838** | **−0.00005** (neutral — **keep fixed views**) |

**Track A conclusion:** Learnable view embeddings do not improve BPC at 40k or 300k under `schedule_b`. Keep **`view_learnable=false`**, fixed random `view_id_dim=8`.

| ID | view | fusion | n_train | BPC | vs sum |
|----|------|--------|---------|-----|--------|
| baseline | fixed | sum | 40k | **4.023** | — |
| B gated | fixed | gated | 40k | **4.140** | **+0.116** (worse — keep `ngram_fusion=sum`) |

**Track B @ 40k:** Gated fusion **hurts** vs sum with fixed random fusion weights (GRIA-only learning). **300k gated sweep skipped** unless fusion weights get online updates.

---

## Parallel work

**Model-class track** ([`CYPHALM_MODEL_CLASS_RESEARCH.md`](CYPHALM_MODEL_CLASS_RESEARCH.md)) runs at the same time. Tracks are independent until **ensemble / hybrid** phase.

---

## Commands (after implementation)

```powershell
python cypha_bench/tuning/cyphalm_upgrade_v2_sweep.py --write
python cypha_bench/run_all.py --domain 17
```
