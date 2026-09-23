# Cypha Bench Baseline Report

Generated: 2026-09-23 08:05 UTC

Default parameters only — no hyperparameter tuning.

## Executive Summary

- Domains run: **25**
- Cross-domain analyses: **4**

## D03

*Timestamp:* 2026-09-23T08:04:34.581+00:00

### backend

- result: cypha_core

### datasets

- result: [{"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.7407407407407407,"f1_macro":0.5802469135802469}},"cypha_scores":{"accuracy":0.37037037037037035,"balanced_accuracy":0.3333333333333333,"ece":0.44433055460056303,"expert_count":3,"generalization_gap":0.010582010582010581,"macro_f1":0.1801801801801802,"margin_mean":0.9539762193688911,"margin_p10":0.6557787724013984,"margin_p50":1.0414995034053467,"mean_confidence":0.6549103555414038,"mean_epistemic_var":0.8011499104126361,"train_accuracy":0.38095238095238093},"data_source":"csv","dataset":"iris","expert_count":3,"kernel_llr":{"enabled":true,"kernel_basis":"rff","kernel_blend":1.0,"kernel_feature_mode":"latent","rff_dim":4096},"n_test":27,"n_train":105,"preprocessor":{"auto_rff_gamma":false,"auto_rff_gamma_cv":true,"input_dim":4,"output_dim":256,"pca_dim":-1,"rff_dim":256,"rff_gamma":0.1,"rff_orf":false,"rff_sorf":true,"scale":true},"view_schedule":"schedule_a"},{"backend":"cypha_core","baselines":{"logistic_regression":{"accuracy":0.8888888888888888,"f1_macro":0.8829629629629631}},"cypha_scores":{"accuracy":0.8055555555555556,"balanced_accuracy":0.8055555555555555,"ece":0.22335537903165967,"expert_count":3,"generalization_gap":-0.07316118935837246,"macro_f1":0.8104454685099847,"margin_mean":0.8359630894408311,"margin_p10":0.05983368100227526,"margin_p50":0.3844775866274861,"mean_confidence":0.6211117847504072,"mean_epistemic_var":0.7950167938082268,"train_accuracy":0.7323943661971831},"data_source":"synthetic","dataset":"wine","expert_count":3,"kernel_llr":{"enabled":true,"kernel_basis":"rff","kernel_blend":1.0,"kernel_feature_mode":"latent","rff_dim":4096},"n_test":36,"n_train":142,"preprocessor":{"auto_rff_gamma":false,"auto_rff_gamma_cv":true,"input_dim":13,"output_dim":256,"pca_dim":-1,"rff_dim":256,"rff_gamma":0.1,"rff_orf":false,"rff_sorf":true,"scale":true},"view_schedule":"schedule_a"}]

### domain

- result: d03_classification

## D14

*Timestamp:* 2026-09-23T08:04:45.223+00:00

### 14A_feynman_all_equations

| Metric | Value |
| --- | --- |
| `kernel_basis` | linear |
| `mean_r2` | -0.2852 |
| `mean_rmse` | 11808469921.1805 |
| `per_equation.Stefan_Boltzmann.crps` | 180.4743 |
| `per_equation.Stefan_Boltzmann.expert_count` | 10.0000 |
| `per_equation.Stefan_Boltzmann.interval_coverage_90` | 0.9100 |
| `per_equation.Stefan_Boltzmann.mae` | 245.4912 |
| `per_equation.Stefan_Boltzmann.mean_epistemic_var` | 68653.0672 |
| `per_equation.Stefan_Boltzmann.r2` | 0.3233 |
| `per_equation.Stefan_Boltzmann.residual_autocorr_lag1` | -0.0054 |
| `per_equation.Stefan_Boltzmann.residual_spectral_flatness` | 0.5662 |
| `per_equation.Stefan_Boltzmann.ridge_rmse` | 530.0391 |
| `per_equation.Stefan_Boltzmann.rmse` | 439.3194 |
| `per_equation.bernoulli.crps` | 5.9225 |
| `per_equation.bernoulli.expert_count` | 10.0000 |
| `per_equation.bernoulli.interval_coverage_90` | 0.8100 |
| `per_equation.bernoulli.mae` | 7.8455 |
| `per_equation.bernoulli.mean_epistemic_var` | 73.5006 |
| `per_equation.bernoulli.r2` | 0.1106 |
| `per_equation.bernoulli.residual_autocorr_lag1` | 0.1460 |
| `per_equation.bernoulli.residual_spectral_flatness` | 0.5671 |
| `per_equation.bernoulli.ridge_rmse` | 12.6637 |
| `per_equation.bernoulli.rmse` | 11.6314 |
| `per_equation.capacitor_energy.crps` | 4.9199 |
| `per_equation.capacitor_energy.expert_count` | 10.0000 |
| `per_equation.capacitor_energy.interval_coverage_90` | 0.7000 |
| `per_equation.capacitor_energy.mae` | 6.7443 |
| `per_equation.capacitor_energy.mean_epistemic_var` | 30.1655 |
| `per_equation.capacitor_energy.r2` | 0.3890 |
| `per_equation.capacitor_energy.residual_autocorr_lag1` | 0.0057 |

### 14B_extrapolation_uncertainty

| Metric | Value |
| --- | --- |
| `extrapolation_auroc` | 1.0000 |
| `regressor_uncertainty_auroc` | 0.9948 |

### 14C_noise_vs_aleatoric

| Metric | Value |
| --- | --- |
| `0.000000.mean_epistemic_var` | 9.8728 |
| `0.000000.rmse` | 3.2632 |
| `0.050000.mean_epistemic_var` | 9.8771 |
| `0.050000.rmse` | 3.2896 |
| `0.100000.mean_epistemic_var` | 9.8946 |
| `0.100000.rmse` | 3.3187 |
| `0.200000.mean_epistemic_var` | 9.9689 |
| `0.200000.rmse` | 3.3848 |
| `0.500000.mean_epistemic_var` | 10.5066 |
| `0.500000.rmse` | 3.6408 |

### backend

- result: cypha_core

## D18

_No experiments recorded._

## D20

*Timestamp:* 2026-09-22T23:54:14.978+00:00

### backend

- result: cypha_cell_hypothesis_sweep --overnight-sweep-smoke --intelligence-profile

### intelligence_profile

- result: true

### kappa_ranked_variants

- result: [{"id":"B2","kappa":0.8971428571428571},{"id":"H06","kappa":0.8971428571428571},{"id":"H14","kappa":0.8971428571428571}]

### n_eval

- result: 32.0000

### n_train

- result: 80.0000

### overnight_sweep_smoke

- result: [{"bench_mode":"hybrid","bpc":6.483989642920271,"id":"B2","kappa":0.8971428571428571,"n_train":80},{"bench_mode":"hybrid","bpc":6.483989642920271,"id":"H06","kappa":0.8971428571428571,"n_train":80},{"bench_mode":"hybrid","bpc":6.483989642920271,"id":"H14","kappa":0.8971428571428571,"n_train":80}]

### pareto_ranked_variants

- result: [{"bpc":6.483989642920271,"id":"B2","kappa":0.8971428571428571,"nondominated":true,"normalized_bpc":0.0,"pareto_score":0.8971428571428571},{"bpc":6.483989642920271,"id":"H06","kappa":0.8971428571428571,"nondominated":true,"normalized_bpc":0.0,"pareto_score":0.8971428571428571},{"bpc":6.483989642920271,"id":"H14","kappa":0.8971428571428571,"nondominated":true,"normalized_bpc":0.0,"pareto_score":0.8971428571428571}]

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

## D29

_No experiments recorded._

## D30

_No experiments recorded._

## D31

_No experiments recorded._

## D32

_No experiments recorded._

## D33

_No experiments recorded._

## D34

_No experiments recorded._

## D35

_No experiments recorded._

## D36

_No experiments recorded._

## D37

_No experiments recorded._

## D38

_No experiments recorded._

## D42

_No experiments recorded._

## D43

_No experiments recorded._

## D56

_No experiments recorded._

## D57

_No experiments recorded._

## Cross-Domain Analyses

### cross_alpha_spectrum_global

| Metric | Value |
| --- | --- |
| `summary.global_mean_alpha` | — |
| `summary.global_std_alpha` | — |
| `summary.n_measurements` | 0.0000 |
| `summary.within_gul_band_fraction` | — |

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
| `summary.mean_ood_auroc` | 1.0000 |
| `summary.n_experiments` | 3.0000 |

> **Note:** Report figures (JSON + native PNG) live in `bench/report/figures/` (`generate_figure_data`).
