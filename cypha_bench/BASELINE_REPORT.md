# Cypha Bench Baseline Report

Generated: 2026-06-11 09:17 UTC

Default parameters only — no hyperparameter tuning.

## Executive Summary

- Domains run: **17**
- Cross-domain analyses: **4**

## D01

*Timestamp:* 2026-06-11T09:16:30.359+00:00

### backend

- result: cypha_core

### tasks

- result: [{"accuracy":0.9875,"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.8875,"f1_macro":0.8870588235294117}},"cypha_scores":{"accuracy":0.9875},"dataset":"linearly_separable_2class","expert_count":2,"n_test":80,"n_train":320,"task":"linearly_separable_2class"},{"accuracy":0.8875,"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.725,"f1_macro":0.6978021978021978}},"cypha_scores":{"accuracy":0.8875},"dataset":"4_gaussian_blobs","expert_count":2,"n_test":80,"n_train":320,"task":"4_gaussian_blobs"}]

## D02

*Timestamp:* 2026-06-11T08:24:41.180+00:00

### backend

- result: cypha_core

### baselines

| Metric | Value |
| --- | --- |
| `ridge.mae` | 1.3904 |
| `ridge.r2` | 0.1079 |
| `ridge.rmse` | 1.7229 |

### cypha_scores

| Metric | Value |
| --- | --- |
| `mae` | 1.2692 |
| `r2` | 0.2971 |
| `rmse` | 1.5293 |

### dataset

- result: synthetic_regression

### n_experts

- result: 10.0000

## D03

*Timestamp:* 2026-06-11T08:24:41.223+00:00

### backend

- result: cypha_core

### datasets

- result: [{"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.7333333333333333,"f1_macro":0.6666666666666666}},"cypha_scores":{"accuracy":0.9666666666666667},"data_source":"synthetic","dataset":"iris","expert_count":3,"n_test":30,"n_train":120},{"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.8888888888888888,"f1_macro":0.882051282051282}},"cypha_scores":{"accuracy":1.0},"data_source":"synthetic","dataset":"wine","expert_count":3,"n_test":36,"n_train":142}]

### domain

- result: d03_classification

## D04

*Timestamp:* 2026-06-11T09:17:09.979+00:00

### 17B_alpha_spectrum

| Metric | Value |
| --- | --- |
| `fraction_edge_of_chaos` | 0.9922 |
| `mean_alpha` | 0.4961 |
| `n_experts` | 2.0000 |

### backend

- result: cypha_lm_native

### bpc

- result: 4.2233

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

## D05

*Timestamp:* 2026-06-11T08:25:23.185+00:00

### backend

- result: cypha_core

### baselines

| Metric | Value |
| --- | --- |
| `ridge.mae` | 0.9203 |
| `ridge.r2` | 0.0282 |
| `ridge.rmse` | 0.9471 |

### cypha_scores

| Metric | Value |
| --- | --- |
| `expert_count` | 10.0000 |
| `mae` | 0.8844 |
| `mean_epistemic_var` | 0.9142 |
| `r2` | 0.0910 |
| `ridge_rmse` | 0.9471 |
| `rmse` | 0.9159 |

### data_source

- result: synthetic

### domain

- result: d05_chess

### n_samples

- result: 4000.0000

## D06

*Timestamp:* 2026-06-11T08:25:29.314+00:00

### backend

- result: cypha_core

### classification

| Metric | Value |
| --- | --- |
| `cypha_scores.accuracy` | 0.9869 |
| `cypha_scores.expert_count` | 2.0000 |
| `cypha_scores.mean_epistemic_var` | 0.0105 |

### domain

- result: d06_go

### n_samples

- result: 8000.0000

### regression

| Metric | Value |
| --- | --- |
| `cypha_scores.expert_count` | 10.0000 |
| `cypha_scores.mae` | 0.4912 |
| `cypha_scores.mean_epistemic_var` | 0.2516 |
| `cypha_scores.r2` | -0.0272 |
| `cypha_scores.rmse` | 0.5066 |
| `ridge_rmse` | 0.5031 |

## D07

*Timestamp:* 2026-06-11T08:25:30.209+00:00

### backend

- result: cypha_core

### cypha_scores

| Metric | Value |
| --- | --- |
| `accuracy` | 0.9287 |
| `boundary_uncertainty_spearman` | -0.7192 |
| `expert_count` | 3.0000 |
| `mean_epistemic_var` | 0.4599 |

### domain

- result: d07_poker

### n_hands

- result: 12000.0000

## D08

*Timestamp:* 2026-06-11T08:28:59.179+00:00

### backend

- result: cypha_core

### data_source

- result: mnist

### domain

- result: d08_computer_vision

### experiments

- result: [{"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.857,"f1_macro":0.8532772761062708}},"cypha_scores":{"accuracy":0.7675},"dataset":"raw","encoding":"raw","expert_count":10,"n_test":2000,"n_train":10000},{"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.92,"f1_macro":0.9192285196235247}},"cypha_scores":{"accuracy":0.8505},"dataset":"hog","encoding":"hog","expert_count":10,"n_test":2000,"n_train":10000}]

## D09

*Timestamp:* 2026-06-11T09:12:32.287+00:00

### 20news

| Metric | Value |
| --- | --- |
| `cypha_scores.accuracy` | 0.5000 |
| `cypha_scores.expert_count` | 20.0000 |
| `cypha_scores.mean_epistemic_var` | 0.4743 |
| `data_source` | synthetic_newsgroups |
| `n_samples` | 2000.0000 |

### backend

- result: cypha_core

### domain

- result: d09_documents

### gutenberg_book_classification

| Metric | Value |
| --- | --- |
| `cypha_scores.accuracy` | 0.8091 |
| `cypha_scores.expert_count` | 3.0000 |
| `cypha_scores.mean_epistemic_var` | 0.1344 |
| `n_segments` | 1102.0000 |

### gutenberg_ood

| Metric | Value |
| --- | --- |
| `mean_epistemic_in` | 0.5000 |
| `mean_epistemic_ood` | 2.1250 |
| `n_ood` | 1102.0000 |

## D10

*Timestamp:* 2026-06-11T08:29:02.693+00:00

### 10A_ecg_classification

| Metric | Value |
| --- | --- |
| `accuracy` | 0.6067 |
| `data_source` | synthetic |
| `expert_count` | 5.0000 |
| `mean_epistemic_var` | 0.4228 |

### 10B_ecg_sliding_window

| Metric | Value |
| --- | --- |
| `accuracy` | 0.2946 |
| `expert_count` | 5.0000 |
| `mean_epistemic_var` | 1.3953 |

### 10C_ecg_ood_detection

| Metric | Value |
| --- | --- |
| `ood_auroc` | 0.5000 |

### 10D_financial_return_sign

| Metric | Value |
| --- | --- |
| `accuracy` | 0.5304 |
| `expert_count` | 2.0000 |
| `mean_epistemic_var` | — |
| `note` | near_chance_expected |

### backend

- result: cypha_core

## D11

*Timestamp:* 2026-06-11T08:29:04.843+00:00

### 11A_cartpole_value_regression

| Metric | Value |
| --- | --- |
| `expert_count` | 10.0000 |
| `mae` | 7.8564 |
| `mean_epistemic_var` | 73.8513 |
| `r2` | -0.0064 |
| `ridge_rmse` | 9.5931 |
| `rmse` | 9.6205 |

### 11B_gridworld_q_estimation

| Metric | Value |
| --- | --- |
| `n_pairs` | 13.0000 |
| `q_value_mae` | 0.2987 |

### 11C_trajectory_preference

| Metric | Value |
| --- | --- |
| `accuracy` | 1.0000 |
| `expert_count` | 1.0000 |
| `mean_epistemic_var` | 0.0000 |

### backend

- result: cypha_core

## D12

*Timestamp:* 2026-06-11T08:29:26.956+00:00

### 12A_binary_intrusion

| Metric | Value |
| --- | --- |
| `cypha_ood_auroc` | 0.5000 |
| `data_source` | nsl_kdd |

### 12B_attack_types

| Metric | Value |
| --- | --- |
| `mean_epistemic_attack` | 0.3637 |
| `n_test` | 2000.0000 |

### 12C_online_detection

| Metric | Value |
| --- | --- |
| `detection_latency_steps` | 1.0000 |
| `final_attack_acc` | 0.9050 |

### backend

- result: cypha_core

## D13

*Timestamp:* 2026-06-11T08:29:30.169+00:00

### 13A_alpha_vs_compression

| Metric | Value |
| --- | --- |
| `files` | [11 items] |
| `gzip_ratios` | [11 items] |
| `mean_alpha_proxy` | [11 items] |
| `spearman_alpha_vs_gzip` | 0.0000 |

### 13B_binary_vs_text_alpha

| Metric | Value |
| --- | --- |
| `mean_alpha_binary` | 0.0000 |
| `mean_alpha_text` | 0.0000 |

### backend

- result: cypha_core

## D14

*Timestamp:* 2026-06-11T08:29:30.988+00:00

### 14A_feynman_all_equations

| Metric | Value |
| --- | --- |
| `mean_r2` | 0.4444 |
| `mean_rmse` | 34518487335.3894 |
| `per_equation.Stefan_Boltzmann.expert_count` | 10.0000 |
| `per_equation.Stefan_Boltzmann.mae` | 197.2706 |
| `per_equation.Stefan_Boltzmann.mean_epistemic_var` | 127475.0581 |
| `per_equation.Stefan_Boltzmann.r2` | 0.6026 |
| `per_equation.Stefan_Boltzmann.ridge_rmse` | 487.4727 |
| `per_equation.Stefan_Boltzmann.rmse` | 320.6142 |
| `per_equation.bernoulli.expert_count` | 10.0000 |
| `per_equation.bernoulli.mae` | 6.6271 |
| `per_equation.bernoulli.mean_epistemic_var` | 130.4629 |
| `per_equation.bernoulli.r2` | 0.4866 |
| `per_equation.bernoulli.ridge_rmse` | 11.9153 |
| `per_equation.bernoulli.rmse` | 8.6457 |
| `per_equation.capacitor_energy.expert_count` | 10.0000 |
| `per_equation.capacitor_energy.mae` | 3.9218 |
| `per_equation.capacitor_energy.mean_epistemic_var` | 50.1372 |
| `per_equation.capacitor_energy.r2` | 0.7598 |
| `per_equation.capacitor_energy.ridge_rmse` | 11.7080 |
| `per_equation.capacitor_energy.rmse` | 6.0356 |
| `per_equation.centripetal.expert_count` | 10.0000 |
| `per_equation.centripetal.mae` | 12.8252 |
| `per_equation.centripetal.mean_epistemic_var` | 910.0168 |
| `per_equation.centripetal.r2` | 0.2287 |
| `per_equation.centripetal.ridge_rmse` | 46.0741 |
| `per_equation.centripetal.rmse` | 39.8314 |
| `per_equation.coulombs_law.expert_count` | 10.0000 |
| `per_equation.coulombs_law.mae` | 153594101247.0655 |
| `per_equation.coulombs_law.mean_epistemic_var` | 163001015368348594077696.0000 |
| `per_equation.coulombs_law.r2` | 0.0283 |

### 14B_extrapolation_uncertainty

| Metric | Value |
| --- | --- |
| `extrapolation_auroc` | 1.0000 |
| `regressor_uncertainty_auroc` | 0.9992 |

### 14C_noise_vs_aleatoric

| Metric | Value |
| --- | --- |
| `0.000000.mean_epistemic_var` | 20.4592 |
| `0.000000.rmse` | 2.4396 |
| `0.050000.mean_epistemic_var` | 20.3691 |
| `0.050000.rmse` | 2.4376 |
| `0.100000.mean_epistemic_var` | 20.2958 |
| `0.100000.rmse` | 2.4396 |
| `0.200000.mean_epistemic_var` | 20.1995 |
| `0.200000.rmse` | 2.4555 |
| `0.500000.mean_epistemic_var` | 20.3128 |
| `0.500000.rmse` | 2.5937 |

### backend

- result: cypha_core

## D15

*Timestamp:* 2026-06-11T08:29:58.134+00:00

### 15A_gaussian_noise

| Metric | Value |
| --- | --- |
| `0.000000.accuracy` | 0.4723 |
| `0.000000.expert_count` | 10.0000 |
| `0.000000.mean_epistemic_var` | 1.5256 |
| `0.100000.accuracy` | 0.4673 |
| `0.100000.expert_count` | 10.0000 |
| `0.100000.mean_epistemic_var` | 1.5208 |
| `0.200000.accuracy` | 0.4560 |
| `0.200000.expert_count` | 10.0000 |
| `0.200000.mean_epistemic_var` | 1.5120 |
| `0.500000.accuracy` | 0.4093 |
| `0.500000.expert_count` | 10.0000 |
| `0.500000.mean_epistemic_var` | 1.4376 |
| `1.000000.accuracy` | 0.3115 |
| `1.000000.expert_count` | 10.0000 |
| `1.000000.mean_epistemic_var` | 1.2338 |

### 15B_feature_dropout

| Metric | Value |
| --- | --- |
| `0.100000.accuracy` | 0.4470 |
| `0.100000.expert_count` | 10.0000 |
| `0.100000.mean_epistemic_var` | 1.5336 |
| `0.250000.accuracy` | 0.4066 |
| `0.250000.expert_count` | 10.0000 |
| `0.250000.mean_epistemic_var` | 1.6292 |
| `0.500000.accuracy` | 0.3234 |
| `0.500000.expert_count` | 10.0000 |
| `0.500000.mean_epistemic_var` | 1.8124 |
| `0.750000.accuracy` | 0.2125 |
| `0.750000.expert_count` | 10.0000 |
| `0.750000.mean_epistemic_var` | 2.0121 |

### 15C_adversarial_fgsm_proxy

| Metric | Value |
| --- | --- |
| `accuracy_adversarial` | 0.2580 |
| `accuracy_natural` | 0.4540 |
| `mean_epistemic_adversarial` | 1.7652 |
| `mean_epistemic_natural` | 1.5808 |

### backend

- result: cypha_core

## D16

*Timestamp:* 2026-06-11T08:30:10.557+00:00

### 16A_task_discovery

| Metric | Value |
| --- | --- |
| `expert_count` | 16.0000 |
| `per_task_accuracy.digits` | 0.6307 |
| `per_task_accuracy.iris` | 0.9231 |
| `per_task_accuracy.wine` | 0.7111 |
| `routing_ari` | 0.0000 |

### 16B_forgetting_resistance

| Metric | Value |
| --- | --- |
| `forgetting_score` | 0.2703 |
| `task_a_accuracy_after` | 0.6923 |
| `task_a_accuracy_before` | 0.9487 |

### 16D_interleaving_comparison

| Metric | Value |
| --- | --- |
| `block.digits` | 0.6559 |
| `block.iris` | 0.8974 |
| `block.wine` | 0.8222 |
| `random.digits` | 0.6082 |
| `random.iris` | 0.6154 |
| `random.wine` | 0.8000 |
| `round_robin.digits` | 0.6325 |
| `round_robin.iris` | 0.7179 |
| `round_robin.wine` | 0.8667 |

### 16E_save_restore

| Metric | Value |
| --- | --- |
| `retention_ratio` | 1.0000 |
| `task_a_before` | 0.9231 |
| `task_a_corrupted` | 0.6923 |
| `task_a_restored` | 0.9231 |

### 16F_per_task_models

| Metric | Value |
| --- | --- |
| `forgetting_score` | 0.0000 |
| `note` | per-task isolated models — zero forgetting by architecture |
| `per_task_accuracy.digits` | 0.7018 |
| `per_task_accuracy.iris` | 0.9487 |
| `per_task_accuracy.wine` | 1.0000 |

### 16G_view_streams

| Metric | Value |
| --- | --- |
| `forgetting_delta` | -0.0030 |
| `macro_epochs` | 2.0000 |
| `max_steps` | 3000.0000 |
| `round_robin.forgetting_score` | 0.1111 |
| `round_robin.per_task_accuracy.digits` | 0.6332 |
| `round_robin.per_task_accuracy.iris` | 0.8205 |
| `round_robin.per_task_accuracy.wine` | 0.8222 |
| `round_robin.task_a_accuracy_after` | 0.8205 |
| `round_robin.task_a_accuracy_before` | 0.9231 |
| `task_block_shuffle.forgetting_score` | 0.1081 |
| `task_block_shuffle.per_task_accuracy.digits` | 0.6325 |
| `task_block_shuffle.per_task_accuracy.iris` | 0.8462 |
| `task_block_shuffle.per_task_accuracy.wine` | 0.8222 |
| `task_block_shuffle.task_a_accuracy_after` | 0.8462 |
| `task_block_shuffle.task_a_accuracy_before` | 0.9487 |

### backend

- result: cypha_core

## D17

*Timestamp:* 2026-06-11T09:17:28.287+00:00

### 17B_alpha_spectrum

| Metric | Value |
| --- | --- |
| `fraction_edge_of_chaos` | 0.9961 |
| `mean_alpha` | 0.4981 |
| `n_experts` | 1.0000 |

### backend

- result: cypha_lm_native

### bpc

- result: 4.3825

### corpus

- result: wikitext2

### mode

- result: hybrid

### n_eval

- result: 500.0000

### n_train

- result: 4000.0000

### profile

- result: d17

### synthetic

- result: false

### vocab_size

- result: 256.0000

## Cross-Domain Analyses

### cross_alpha_spectrum_global

| Metric | Value |
| --- | --- |
| `summary.global_mean_alpha` | 0.0414 |
| `summary.global_std_alpha` | 0.1374 |
| `summary.n_measurements` | 14.0000 |
| `summary.within_gul_band_fraction` | 0.0833 |

### cross_forgetting_resistance

| Metric | Value |
| --- | --- |
| `mean_forgetting_score` | 0.1351 |

### cross_online_adaptation

| Metric | Value |
| --- | --- |
| `n_domains_with_adaptation_signal` | 1.0000 |

### cross_uncertainty_calibration

| Metric | Value |
| --- | --- |
| `summary.mean_ood_auroc` | 0.6667 |
| `summary.n_experiments` | 36.0000 |

> **Note:** Report figures (JSON + native PNG) live in `cypha_bench/report/figures/` (`generate_figure_data`).
