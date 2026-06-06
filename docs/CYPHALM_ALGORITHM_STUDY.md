# CyphaLM Algorithm Study

**Purpose:** Understand the AI-generated baseline, what each subsystem contributes, and which upgrade combinations help — recorded systematically, not by guesswork.

**Artifact:** `cypha_bench/config/cyphalm_component_ablation.json`  
**Runner:** `python cypha_bench/tuning/cyphalm_component_ablation.py --write`  
**Library:** `cypha_bench/adapters/cyphalm_component_study.py`

Related: [`cypha_lm/README.md`](../cypha_lm/README.md), [`FINDINGS_CYPHALM_TRAINING.md`](FINDINGS_CYPHALM_TRAINING.md)

---

## Pipeline (baseline algorithm)

Every token step:

```
token_id
  → IzaacEmbedding (GF(2^n) structured lookup, fixed)
  → CellAISSM (fast/slow exponential memory, optional spectral/multiscale/hebbian)
  → proj_ssm → field_x
  → CyphaDIF.predict(field_x) → mean, epistemic_var, routing, experts
  → _gria_input(field_x, dif_out) → v          ← context_mode selects path
  → GRIAProjection(v) → log_probs
  → CE loss → update GRIA (W, α, bias)
  → optional: DIF.train_step, BPTT→SSM, Hebbian SSM, Laplace bias refresh
```

**Default factory config** (`CyphaLMConfig` dataclass defaults): `context_mode=full`, `train_epochs=1`, `bptt_steps=0`, `view_schedule=same_order`. This is the raw AI-generated baseline.

**Current bench profile** (`cyphalm_d17_wikitext.json`): switched to `gria_ngram` + `bptt_steps=64` + `train_epochs=2` after ablations showed `full` underperforms.

---

## Subsystems and what they do

| Subsystem | Module | Trained? | Role |
|-----------|--------|----------|------|
| **Izaac embed** | `izaac_embed.py` | No | Deterministic structured token vectors |
| **CellAI SSM** | `cellai_ssm.py` | Optional (BPTT / Hebbian) | Temporal context `field_x` from token stream |
| **CyphaDIF** | `cypha_dif.py` | Online NIG updates | Expert routing + field mean + uncertainty |
| **GRIA** | `gria_projection.py` | Yes (always) | Maps context vector → vocab log-probs |
| **Projections** | `cypha_lm.py` | No (fixed random) | Linear maps between subspaces |
| **Multi-view** | `cypha_views/` | Schedule only | Reorders corpus; memory reset policy |

### `context_mode` — architecture switch

| Mode | SSM | DIF in forward | GRIA input |
|------|-----|----------------|------------|
| `full` | ✓ | ✓ + epistemic term | `proj_dif(mean) + u·field_x` |
| `gria_ngram` | ✓ | ✓ (but input uses n-gram path) | `proj_ngram([field_x ‖ embed history])` |
| `ssm_only` | ✓ | ✗ (zeroed) | `field_x` |
| `ablation_no_dif` | ✓ | ✓ (mean only) | `proj_dif(mean)` |
| `ablation_no_ssm` | ✗ | ✗ | `proj_ngram(embed history only)` |

### Training switches (orthogonal to context_mode)

| Field | Effect when changed |
|-------|---------------------|
| `bptt_steps > 0` | Backprop GRIA CE into SSM layer-0 fast weights |
| `train_ssm=True` | Hebbian nudge on SSM (disabled when BPTT active) |
| `online=False` | DIF frozen after init — predict only |
| `laplace_smoothing` | GRIA bias = log-smoothed unigram counts |
| `alpha_learnable=False` | Per-token GRIA blend α fixed |
| `use_multiscale` | Blend fast/slow SSM tracks per layer |
| `use_spectral_pde` | FFT circulant state transition |
| `use_sparse_hebbian` | Sparse outer-product SSM updates |
| `ngram_context` | How many past embeds concat (gria_ngram / no_ssm) |
| `view_schedule` | Multi-view training order |

---

## Ablation study design

Four phases, run via `--phase` or all at once:

### Phase A — Architecture (5 cells)

Each mode in isolation. **Question:** Which pipeline path is viable?

Known prior result @ 40k D17:
- `gria_ngram` **4.154** ≪ `full` **4.725**
- `ablation_no_ssm` **4.165** ≈ gria_ngram → **n-gram embed path carries most gain**
- `full` == `ablation_no_dif` → **epistemic DIF term hurts or is unused in full path**

### Phase B — Single toggles (16 cells)

Start from `gria_ngram` + D17 profile; flip one switch. **Question:** Does each subsystem help or hurt?

| Toggle | Hypothesis |
|--------|------------|
| `no_bptt` | BPTT should help SSM contribute |
| `no_laplace` | Unigram prior anchors early training |
| `dif_offline` | Online expert growth helps/hurts |
| `frozen_alpha` | Learnable α matters |
| `no_multiscale` | Multi-scale memory helps |
| `spectral_pde` / `sparse_hebbian` | v3 SSM innovations |
| `ngram1/3/4` | Optimal history window |
| `tau_tight` | Faster memory may suit char-LM |

### Phase C — SSM combinatorics (8 cells)

All combinations of `{spectral, multiscale, hebbian}` on `gria_ngram`.

### Phase D — Origins + upgrade stacks (8 cells)

| Cell | Meaning |
|------|---------|
| `origin_factory_defaults` | Raw AI baseline (`full`, no BPTT) |
| `origin_bench_default` | Pre-profile bench defaults |
| `upgrade_profile_d17` | Current production profile |
| `upgrade_stack_best` | Combined upgrades from sweeps |

---

## Commands

```powershell
# Fast smoke (~10 cells, 8k tokens)
python cypha_bench/tuning/cyphalm_component_ablation.py --fast --write

# Full study (~37 cells, 40k tokens) — ~4–8 hours
python cypha_bench/tuning/cyphalm_component_ablation.py --write

# Single phase
python cypha_bench/tuning/cyphalm_component_ablation.py --phase architecture --write
python cypha_bench/tuning/cyphalm_component_ablation.py --phase toggle --write
python cypha_bench/tuning/cyphalm_component_ablation.py --phase ssm_combo --write
python cypha_bench/tuning/cyphalm_component_ablation.py --phase upgrade --write

# D17 bench experiment
python cypha_bench/run_all.py --domain 17   # includes 17H when not fast
```

---

## Results log

_Update after each study run._

| Run | n_train | Best cell | BPC | vs bigram | Key finding |
|-----|---------|-----------|-----|-----------|-------------|
| Prior D17 ablations | 40k | `gria_ngram` | 4.154 | +0.24 | `full` pipeline broken for char-LM |
| **Fast component study** | **8k** | **`toggle_frozen_alpha`** | **4.288** | **−0.29** | See below |
| **Full component study** | **40k** | **`toggle_frozen_alpha`** | **4.043** | **+0.13** | D17 — see below |
| **D04 component study** | **40k** | **`toggle_frozen_alpha`** | **4.029** | **+0.07** | Gutenberg — same pattern |

### Fast study (8k, 23 cells, 2026-06-05)

| Finding | Detail |
|---------|--------|
| Architecture | `gria_ngram` **4.367** vs `full` **4.830** (−0.46) — n-gram path essential |
| Factory vs profile | AI defaults **4.865** vs D17 profile **4.367** — profile tuning worth **+0.50 BPC** |
| Epistemic DIF | `full` == `ablation_no_dif` — epistemic term useless |
| **Frozen α** | `alpha_learnable=False` → **4.288** (best); learnable α hurts at short train |
| BPTT / Laplace / online DIF | **No effect** at 8k (identical to baseline) |
| train_ssm | **Catastrophic** — 39.84 BPC @ 40k (+35.9) |
| BPTT @ 40k | **No effect** with schedule_b (axis 0–128 identical) |
| SSM 8-combo | All ~4.094; default multiscale-on wins trivially |
| Stack @ 300k | `axes_frozen_alpha` **3.838** (best stack; still +0.27 vs bigram) |

### Full study (40k, 37 cells, 2026-06-05)

| Finding | Detail |
|---------|--------|
| Architecture | `gria_ngram` **4.094** vs `full` **4.682** (−0.59) |
| Factory vs profile | AI defaults **4.665** vs D17 profile **4.094** |
| **Frozen α** | **4.043** BPC — best @ 40k (+0.05 vs baseline) |
| BPTT / Laplace / online DIF | No measurable effect @ 40k |
| train_ssm | **39.84 BPC** — never enable with BPTT profile |

### Interpretation guide

- **Negative toggle delta** (e.g. `no_bptt` worse than baseline): that component **helps**
- **Positive toggle delta** (removal improves BPC): component **hurts** at this budget
- **`full_minus_gria_ngram_bpc` > 0**: n-gram path is strictly better than full DIF+epistemic
- **`profile_gain_vs_factory_bpc` > 0**: profile tuning improved over raw defaults

---

## Open questions

1. Does BPTT actually move BPC on `gria_ngram`? (toggle test)
2. Which SSM flag combo wins the 8-cell grid?
3. Can `full` pipeline be salvaged with `tau_tight` + BPTT, or is DIF path wrong for char-LM?
4. Does `online=False` DIF beat online (less overfit)?
5. Is factory `full` mode worse than pure `ablation_no_ssm`?
