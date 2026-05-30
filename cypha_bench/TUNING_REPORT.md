# CyphaDIF Everyday Profile — Validation

Generated: 2026-05-26T07:43:04.840531+00:00

## Selected profile

```json
{
  "classification": null,
  "regression": null,
  "architecture": {
    "replay_ratio": 0.24200000000000005,
    "ood_sigma": 12.0
  }
}
```

## Baseline (library defaults)

- **d02/diabetes** rmse=72.5762
- **d02/california_housing** rmse=1.1783
- **d03/iris** accuracy=0.0000
- **d03/wine** accuracy=0.0000
- **d03/breast_cancer** accuracy=0.0000
- **d03/digits** accuracy=0.0000
- **d03/20newsgroups_subset** accuracy=0.0000
- **d08/raw** accuracy=0.0000
- **d08/hog** accuracy=0.0000

## Tuned (everyday profile)

- **d02/diabetes** rmse=53.3765
- **d02/california_housing** rmse=0.7929
- **d03/iris** accuracy=0.8667
- **d03/wine** accuracy=0.9167
- **d03/breast_cancer** accuracy=0.9474
- **d03/digits** accuracy=0.9222
- **d03/20newsgroups_subset** accuracy=0.1688
- **d08/raw** accuracy=0.7420
- **d08/hog** accuracy=0.8900

## Notes

- Baseline: `CYPHA_BENCH_USE_PROFILE=0`
- Tuned: profile from `cypha_bench/config/everyday_profile.json` (fallback `config/profiled_medium.json`)
