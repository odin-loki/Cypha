# Cypha Bench Baseline Report

Generated: 2026-06-14 10:02 UTC

Default parameters only — no hyperparameter tuning.

## Executive Summary

- Domains run: **13**
- Cross-domain analyses: **4**

## D01

*Timestamp:* 2026-06-13T08:47:54.098+00:00

### backend

- result: cypha_core

### tasks

- result: [{"accuracy":0.9875,"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.8875,"f1_macro":0.8870588235294117}},"cypha_scores":{"accuracy":0.9875},"dataset":"linearly_separable_2class","expert_count":2,"n_test":80,"n_train":320,"task":"linearly_separable_2class"},{"accuracy":0.8875,"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.725,"f1_macro":0.6978021978021978}},"cypha_scores":{"accuracy":0.8875},"dataset":"4_gaussian_blobs","expert_count":2,"n_test":80,"n_train":320,"task":"4_gaussian_blobs"}]

## D03

*Timestamp:* 2026-06-13T06:36:48.964+00:00

### S3_xor_kernel_llr

| Metric | Value |
| --- | --- |
| `accuracy` | 0.5740 |
| `backend` | xor_kernel_bench_native |
| `delta_pp` | 6.8000 |
| `kernel_m` | 256.0000 |

### S3_xor_linear

| Metric | Value |
| --- | --- |
| `accuracy` | 0.5060 |
| `passes` | 2.0000 |
| `seeds` | 1.0000 |

## D04

*Timestamp:* 2026-06-13T06:36:19.858+00:00

### 17B_alpha_spectrum

| Metric | Value |
| --- | --- |
| `fraction_edge_of_chaos` | 0.9961 |
| `mean_alpha` | 0.4981 |
| `n_experts` | 1.0000 |

### backend

- result: cypha_lm_native

### bpc

- result: 4.1584

### corpus

- result: moby_dick.txt

### mode

- result: hybrid

### n_eval

- result: 1000.0000

### n_train

- result: 8000.0000

### profile

- result: d04

### synthetic

- result: false

### vocab_size

- result: 256.0000

## D17

*Timestamp:* 2026-06-13T08:18:10.838+00:00

### 17B_alpha_spectrum

| Metric | Value |
| --- | --- |
| `fraction_edge_of_chaos` | 0.9961 |
| `mean_alpha` | 0.4981 |
| `n_experts` | 1.0000 |

### backend

- result: cypha_lm_native

### bpc

- result: 7.9979

### corpus

- result: synthetic

### mode

- result: hybrid

### n_eval

- result: 500.0000

### n_train

- result: 4000.0000

### profile

- result: d17

### ssm_diagnose

| Metric | Value |
| --- | --- |
| `checks_passed` | true |
| `decay_rates.fast` | 0.3679 |
| `decay_rates.slow` | 0.9512 |
| `domain` | cyphalm |
| `lambda_fast` | 0.3679 |
| `lambda_slow` | 0.9512 |
| `projection.connected_to_routing` | true |
| `projection.context_dim` | 512.0000 |
| `projection.field_dim` | 160.0000 |
| `projection.field_output_mean_rms` | 0.7846 |
| `projection.proj_weight_rms` | 0.0200 |
| `sample_stride` | 8.0000 |
| `state_norms.context_track` | [128 items] |
| `state_norms.fast_track` | [128 items] |
| `state_norms.slow_track` | [128 items] |
| `steps` | 128.0000 |
| `summary.collapsed` | false |
| `summary.context.collapsed` | false |
| `summary.context.exploded` | false |
| `summary.context.final` | 3.4363 |
| `summary.context.max` | 3.8067 |
| `summary.context.min` | 0.8885 |
| `summary.exploded` | false |
| `summary.fast.collapsed` | false |
| `summary.fast.exploded` | false |
| `summary.fast.final` | 2.1179 |
| `summary.fast.max` | 2.8594 |
| `summary.fast.min` | 1.0112 |
| `summary.slow.collapsed` | false |
| `summary.slow.exploded` | false |

### synthetic

- result: true

### vocab_size

- result: 256.0000

## D18

_No experiments recorded._

## D20

*Timestamp:* 2026-06-14T10:02:02.407+00:00

### backend

- result: cypha_cell_hypothesis_sweep --overnight-sweep-smoke

### overnight_sweep_smoke

- result: [{"bench_mode":"hybrid","bpc":7.967382408140876,"id":"B2","n_train":200},{"bench_mode":"hybrid","bpc":7.9673824687505554,"id":"H06","n_train":200},{"bench_mode":"hybrid","bpc":7.967382408140876,"id":"H14","n_train":200}]

### variant_count

- result: 3.0000

## D22

_No experiments recorded._

## D23

_No experiments recorded._

## D24

_No experiments recorded._

## D25

_No experiments recorded._

## D26

_No experiments recorded._

## D27

_No experiments recorded._

## D28

_No experiments recorded._

## Cross-Domain Analyses

### cross_alpha_spectrum_global

| Metric | Value |
| --- | --- |
| `summary.global_mean_alpha` | 0.4981 |
| `summary.global_std_alpha` | 0.0000 |
| `summary.n_measurements` | 2.0000 |
| `summary.within_gul_band_fraction` | 1.0000 |

### cross_forgetting_resistance

| Metric | Value |
| --- | --- |
| `mean_forgetting_score` | — |

### cross_online_adaptation

| Metric | Value |
| --- | --- |
| `n_domains_with_adaptation_signal` | 0.0000 |

### cross_uncertainty_calibration

| Metric | Value |
| --- | --- |
| `summary.mean_ood_auroc` | — |
| `summary.n_experiments` | 5.0000 |

> **Note:** Report figures (JSON + native PNG) live in `bench/report/figures/` (`generate_figure_data`).
