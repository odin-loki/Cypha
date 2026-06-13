# cypha_som — Optional Self-Organising Upgrades

> **Failed experiment (archived):** [`docs/archive/failed_experiments/cypha_som/README.md`](../docs/archive/failed_experiments/cypha_som/README.md)

`cypha_som` provides six experimental SOM/GNG/GRIA extensions to `CyphaDIF`. All flags are **OFF by default** — the base classifier runs without any of this code.

> **Research status:** Upgrades U2/U1/U4/`"all"` were benchmarked and reverted (performance
> worse than baseline on 9 standard domains). Upgrades U3/U5/U6 are structurally safe but
> target CellAI (SSM) and have minimal effect on the CyphaDIF classifier. See the
> [SOM Upgrade Report](../docs/reports/SOM_UPGRADE_REPORT.md) for the full results.

---

## Upgrade map

| Flag | Module | Class | Description |
|------|--------|-------|-------------|
| U1 | `gng_expert.py` | `GNGExpertManager` | Growing Neural Gas prototypes for auxiliary routing/context |
| U2 | `som_encoder.py` | `OnlineSOMEncoder`, `SOMWrappedEncoder` | 2D grid SOM on encoder features |
| U3 | `gria_controller.py` | `GRIAController` | Entropy-based α control for GNG structural changes |
| U4 | `discriminative_feedback.py` | `DiscriminativeFeedback` | Encoder ΔW modulated by class separation + inverse variance |
| U5 | `hebbian_topology.py` | `DynamicHebbianGraph` | Dynamic edge graph for CellAI state diffusion |
| U6 | `temporal_som.py` | `TemporalSOM` | Autocorrelation-based fast/slow decay scaling for SSM |

Integration entry points: `CyphaSOMHooks` wires U1–U4 into `CyphaDIF`; `wire_cellai(ssm)` attaches U5/U6 to a `CellAISSM` instance from `cypha_lm`.

---

## Enabling upgrades

**Programmatic:**

```python
from cypha_som import set_upgrade_flags, CyphaSOMHooks

# Single upgrade
set_upgrade_flags("U2")

# Multiple
set_upgrade_flags(["U1", "U2"])

# All (not recommended — benchmarked worse than baseline)
set_upgrade_flags("all")

# Reset to all-off
set_upgrade_flags("none")
```

**Environment variables** (evaluated at import time via `apply_env_overrides()`):

```bash
CYPHA_SOM_USE_GNG=1       # U1
CYPHA_SOM_USE_SOM_ENCODER=1  # U2
CYPHA_SOM_USE_GRIA_CONTROLLER=1  # U3
CYPHA_SOM_USE_DISCRIM_FEEDBACK=1  # U4
CYPHA_SOM_USE_DYNAMIC_TOPOLOGY=1  # U5
CYPHA_SOM_USE_TEMPORAL_SOM=1      # U6
```

---

## Wiring into CyphaDIF

```python
from cypha_som import CyphaSOMHooks
from cypha_som.config import set_upgrade_flags

set_upgrade_flags("U2")  # enable SOM encoder only
hooks = CyphaSOMHooks()

# hooks.post_encode(latent, label)   — called after Encoder.project()
# hooks.gng_train_step(latent)       — called during train_step (U1)
# hooks.gria_control(alpha)          — called during GRIA update (U3)
# hooks.discriminative_modulate(dW, labels)  — modulates encoder weight update (U4)
# hooks.merge_context(h, ssm_state)  — merges GNG context into latent (U1+U3)
```

For CellAI SSM integration:

```python
from cypha_som import wire_cellai
from cypha_lm.temporal.cellai_ssm import CellAISSM

ssm = CellAISSM(config)
wire_cellai(ssm)  # attaches U5 (DynamicHebbianGraph) and U6 (TemporalSOM)
```

---

## Running the evaluation

To reproduce the SOM upgrade benchmark:

```bash
python scripts/run_som_upgrade_eval.py
```

This calls `benchmark_baseline.py` with each upgrade flag in sequence and writes results to `artifacts/`. The consolidated report is at [`docs/reports/SOM_UPGRADE_REPORT.md`](../docs/reports/SOM_UPGRADE_REPORT.md).

---

## Tests

```bash
pytest cypha_som/tests/ -q
```

Seven unit tests cover: GNG growth, SOM output dimension, GRIA α bounds, Hebbian spectral radius, temporal SOM decay range, discriminative modulate output, and config flag isolation. **Note:** these tests are not currently in `.github/workflows/ci.yml` — run manually.

---

## Key finding

All six upgrades were evaluated against the 9-domain default benchmark (see `SOM_UPGRADE_REPORT.md`):
- **U2 SOM encoder** degraded accuracy on 7/9 domains.
- **U1 GNG expert** provided minor context benefit but not enough to justify overhead.
- **U4 discriminative feedback** reduced accuracy on structured domains.
- **"all" (U1–U6)**: worst overall — upgrades interact adversely.
- **U3/U5/U6**: structurally safe; U5/U6 are CellAI-only and have no effect on standard CyphaDIF benchmarks.

**Recommended setting:** keep all flags OFF (default) for any production or benchmark use.
