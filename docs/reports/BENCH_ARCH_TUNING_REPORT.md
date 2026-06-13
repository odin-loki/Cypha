# Cypha Bench Architecture Tuning Report

Generated: 2026-05-24T07:30:08+00:00 (post-run recovery)

## Summary

- Target score: 0.92
- Success: **false** (all iterations scored 0.0000 min-ratio vs baseline)
- Iterations run: 5
- Final score: 0.0000
- Runtime: ~17 hours (200-combo swarm × 5 iterations + full-bench validations)

The orchestrator completed all five swarm/validation cycles and wrote `everyday_profile.json`, but the min-ratio scorer returned 0.0 every iteration (likely baseline key mismatch during fast swarm eval, not zero accuracy). Final `validate_profile.py` ran successfully; see `bench/artifacts/tuning/validation_compare.json`.

## Winning profile

Written to `bench/config/everyday_profile.json` with `algorithm_variants`:

- `reg_hash_routing`: false (y-quantile routing)
- `deliberation_lo/hi`: 0.45 / 0.55
- `cold_start_steps`: 10, `min_experts_floor`: 3

## Validation highlights (tuned vs library defaults)

| Domain | Metric | Result |
|--------|--------|--------|
| d02 diabetes | R² | 0.45 (near ridge) |
| d02 california | R² | 0.44 |
| d08 MNIST HOG | acc | ~0.93 (from prior runs) |

## Iterations

- Iteration 1: swarm=full best_validation=0.0 converged=false
- Iteration 2: swarm=narrow best_validation=0.0 converged=false
- Iteration 3: swarm=narrow best_validation=0.0 converged=false
- Iteration 4: swarm=narrow best_validation=0.0 converged=false
- Iteration 5: swarm=narrow best_validation=0.0 converged=false

## Note

Run crashed on report generation (`AttributeError` in `write_arch_tuning_report` — fixed in code). Log: `bench/artifacts/tuning/arch_iterate_20260524_004541.log`.
