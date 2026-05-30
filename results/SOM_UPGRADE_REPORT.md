# SOM Upgrade Evaluation Report

**Date:** 2026-05-26  
**Spec:** `cypha_som_upgrades.md`  
**Baseline:** `results/baseline.json` (10-class, 50-feat, 10k samples, 3 seeds, RFF+CyphaDIF online)

## Summary

| Upgrade | Accuracy (mean) | Δ vs baseline | Train ms | Verdict |
|---------|-----------------|---------------|----------|---------|
| none | 0.1347 ± 0.0106 | — | ~0.45 | baseline |
| U2 SOM encoder | 0.0997 | **−3.5%** | 0.63 | **REVERT** |
| U1 GNG experts | 0.1337 | −0.1% | 1.09 | **REVERT** (time + variance) |
| U3 GRIA controller | 0.1347 | 0.0 | 0.45 | **KEEP** (no-op without U1; safe) |
| U4 Discriminative feedback | 0.1451 | **+1.0%** | 1.04 | **REVERT** (variance ×1.7, time) |
| U5 Hebbian topology | 0.1347 | 0.0 | 0.45 | **KEEP** (CellAI-only; neutral here) |
| U6 Temporal SOM | 0.1347 | 0.0 | 0.45 | **KEEP** (CellAI-only; neutral here) |
| all combined | 0.0997 | **−3.5%** | 2.80 | **REVERT** |

**Default flags remain OFF** (`cypha_som/config.py`). No upgrade ships enabled.

## What was implemented

- Package `cypha_som/`: GNG, online SOM, GRIA controller, discriminative feedback, Hebbian graph, temporal SOM, hooks.
- `CyphaDIF` hooks: GNG context bias, GRIA control, SOM-on-`h`, modulated contrastive updates (U4).
- `CellAISSM`: optional dynamic diffusion (U5) and temporal decay scaling (U6).
- `benchmark_baseline.py` + `scripts/run_som_upgrade_eval.py`.
- Unit tests: `cypha_som/tests/test_som_upgrades.py` (7 tests); CellAI tests still pass.

## Findings

1. **U2 (SOM on RFF)** hurts accuracy and adds latency — replacing RFF output with SOM prototypes is too lossy for this 10-class online task.
2. **U1 (GNG routing bias)** is neutral on accuracy but **>2× train time** and higher seed variance — not worth default-on.
3. **U3** alone is a no-op unless U1 is enabled; safe to keep code.
4. **U4** shows a small accuracy lift (+1 pp) but fails the protocol on **variance** and **train_step overhead** — left disabled.
5. **U5/U6** do not affect `CyphaDIF` classification benchmarks; enable for CyphaLM / CellAI sequence experiments only.
6. **Combined `all`** inherits U2’s regression — do not enable all flags together without retuning SOM/GNG.

## How to re-run

```bash
python benchmark_baseline.py --dataset classification --seeds 3 --upgrade none --output results/baseline.json
python scripts/run_som_upgrade_eval.py
python -m pytest cypha_som/tests/test_som_upgrades.py cypha_lm/temporal/tests/test_cellai_ssm.py -q
```

Enable one upgrade: `CYPHA_SOM_USE_GNG=1` or `python benchmark_baseline.py --upgrade U1`.

## Recommended next steps

- Tune U2 (`k`, `T_eta`) on **drift** benchmark before re-enabling.
- Pair U3 with U1 and measure drift recovery (per MD §Upgrade 1 Test 3).
- Evaluate U4/U5/U6 on **CyphaLM** WikiText/perplexity and long-context needle tests, not sklearn classification.
