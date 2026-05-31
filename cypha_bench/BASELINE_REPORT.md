# Cypha Bench Baseline Report

Generated: 2026-05-31 06:24 UTC

Default parameters only — no hyperparameter tuning.

## Executive Summary

- Domains run: **17**
- Cross-domain analyses: **4**

## D01

*Timestamp:* 2026-05-31T04:22:39.680270+00:00

### linearly_separable_2class

| Metric | Value |
| --- | --- |
| `accuracy` | 0.7825 |
| `cypha_alpha_distribution` | [0.9494238522466439, 0.955311668889827] |
| `cypha_expert_count` | 2 |
| `cypha_fraction_edge_of_chaos` | 0.0000 |
| `cypha_mean_aleatoric_var` | 0.0099 |
| `cypha_mean_alpha` | 0.9524 |
| `cypha_mean_epistemic_var` | 0.0120 |
| `cypha_std_epistemic_var` | 0.0632 |
| `cypha_uncertainty_rank_correlation` | 0.2564 |
| `f1_macro` | 0.7754 |
| `sgd_accuracy` | 0.7925 |
| `sgd_f1_macro` | 0.7922 |

### 4_gaussian_blobs

| Metric | Value |
| --- | --- |
| `accuracy` | 0.9975 |
| `cypha_alpha_distribution` | [0.9551265532326512, 0.9125826618457499, 0.9463752626879252, 0.9369440790234022] |
| `cypha_expert_count` | 4 |
| `cypha_fraction_edge_of_chaos` | 0.0000 |
| `cypha_mean_aleatoric_var` | 0.0083 |
| `cypha_mean_alpha` | 0.9378 |
| `cypha_mean_epistemic_var` | 5.63e-07 |
| `cypha_std_epistemic_var` | 3.05e-11 |
| `cypha_uncertainty_rank_correlation` | 0.1149 |
| `f1_macro` | 0.9975 |
| `sgd_accuracy` | 1.0000 |
| `sgd_f1_macro` | 1.0000 |

### high_dim_noisy

| Metric | Value |
| --- | --- |
| `accuracy` | 0.7850 |
| `cypha_alpha_distribution` | [0.854709318311907, 0.8448657716158512] |
| `cypha_expert_count` | 2 |
| `cypha_fraction_edge_of_chaos` | 0.0000 |
| `cypha_mean_aleatoric_var` | 0.0533 |
| `cypha_mean_alpha` | 0.8498 |
| `cypha_mean_epistemic_var` | 0.1151 |
| `cypha_std_epistemic_var` | 0.2091 |
| `cypha_uncertainty_rank_correlation` | 0.3296 |
| `f1_macro` | 0.7847 |
| `sgd_accuracy` | 0.8325 |
| `sgd_f1_macro` | 0.8322 |

### linear_regression

| Metric | Value |
| --- | --- |
| `cypha_alpha_distribution` | [10 items] |
| `cypha_expert_count` | 10 |
| `cypha_fraction_edge_of_chaos` | 0.0000 |
| `cypha_mean_aleatoric_var` | 2761.8881 |
| `cypha_mean_alpha` | 0.7398 |
| `cypha_mean_epistemic_var` | 11047.5523 |
| `cypha_std_epistemic_var` | 416.5932 |
| `cypha_uncertainty_rank_correlation` | -0.0032 |
| `mae` | 69.2246 |
| `r2` | 0.7563 |
| `rmse` | 86.5928 |
| `sgd_mae` | 0.0923 |
| `sgd_r2` | 1.0000 |
| `sgd_rmse` | 0.1139 |

### nonlinear_regression_sinusoidal

| Metric | Value |
| --- | --- |
| `cypha_alpha_distribution` | [9 items] |
| `cypha_expert_count` | 9 |
| `cypha_fraction_edge_of_chaos` | 0.0000 |
| `cypha_mean_aleatoric_var` | 0.1384 |
| `cypha_mean_alpha` | 0.7303 |
| `cypha_mean_epistemic_var` | 0.5538 |
| `cypha_std_epistemic_var` | 1.11e-16 |
| `cypha_uncertainty_rank_correlation` | 0.0000 |
| `mae` | 0.6515 |
| `r2` | -0.0366 |
| `rmse` | 0.7399 |
| `sgd_mae` | 0.6433 |
| `sgd_r2` | -0.0046 |
| `sgd_rmse` | 0.7284 |

### single_concept_drift

| Metric | Value |
| --- | --- |
| `accuracy` | 0.5550 |
| `cypha_alpha_distribution` | [0.9766947020279926, 0.9891741123369787] |
| `cypha_expert_count` | 2 |
| `cypha_fraction_edge_of_chaos` | 0.0000 |
| `cypha_mean_aleatoric_var` | 0.0168 |
| `cypha_mean_alpha` | 0.9829 |
| `cypha_mean_epistemic_var` | 0.0107 |
| `cypha_std_epistemic_var` | 0.0681 |
| `cypha_uncertainty_rank_correlation` | 0.2394 |
| `f1_macro` | 0.4451 |
| `sgd_accuracy` | 0.7450 |
| `sgd_f1_macro` | 0.7436 |

### pure_noise

| Metric | Value |
| --- | --- |
| `accuracy` | 0.5000 |
| `cypha_alpha_distribution` | [0.9860345208626262, 0.9775134378656982] |
| `cypha_expert_count` | 2 |
| `cypha_fraction_edge_of_chaos` | 0.0000 |
| `cypha_mean_aleatoric_var` | 0.0415 |
| `cypha_mean_alpha` | 0.9818 |
| `cypha_mean_epistemic_var` | 0.0957 |
| `cypha_std_epistemic_var` | 0.1856 |
| `cypha_uncertainty_rank_correlation` | 0.0608 |
| `f1_macro` | 0.4959 |
| `sgd_accuracy` | 0.5000 |
| `sgd_f1_macro` | 0.4968 |

### identical_inputs_different_labels

| Metric | Value |
| --- | --- |
| `accuracy` | 0.4900 |
| `cypha_alpha_distribution` | [0.9885391456919566, 0.9845696231259763] |
| `cypha_expert_count` | 2 |
| `cypha_fraction_edge_of_chaos` | 0.0000 |
| `cypha_mean_aleatoric_var` | 0.0208 |
| `cypha_mean_alpha` | 0.9866 |
| `cypha_mean_epistemic_var` | 0.0456 |
| `cypha_std_epistemic_var` | 0.1351 |
| `cypha_uncertainty_rank_correlation` | 0.0049 |
| `f1_macro` | 0.4783 |
| `sgd_accuracy` | 0.4100 |
| `sgd_f1_macro` | 0.4085 |

### _assertions

| Metric | Value |
| --- | --- |
| `aleatoric_dominates_on_contradictory` | False |
| `blob_expert_count_ok` | True |
| `drift_epistemic_spike` | False |
| `epistemic_higher_on_noise` | True |
| `noise_expert_pathology_flag` | False |

## D02

*Timestamp:* 2026-05-31T04:33:06.100228+00:00

### summary

| Metric | Value |
| --- | --- |
| `domain` | d02_regression |

## D03

*Timestamp:* 2026-05-31T04:34:06.704093+00:00

### summary

| Metric | Value |
| --- | --- |
| `domain` | d03_classification |

## D04

*Timestamp:* 2026-05-31T06:12:43.457927+00:00

### summary

| Metric | Value |
| --- | --- |
| `bigram_bpc` | 3.8411 |
| `char_lm.bigram_bpc` | 3.8411 |
| `char_lm.context_length_bpc.128` | 4.8561 |
| `char_lm.context_length_bpc.16` | 4.7851 |
| `char_lm.context_length_bpc.256` | 4.9045 |
| `char_lm.context_length_bpc.32` | 4.7728 |
| `char_lm.context_length_bpc.64` | 4.7279 |
| `char_lm.context_length_bpc.8` | 4.7961 |
| `char_lm.cypha_dif.lossless_fraction` | 4.57e-05 |
| `char_lm.cypha_dif.mean_aleatoric_var` | 0.9429 |
| `char_lm.cypha_dif.mean_alpha` | 0.0904 |
| `char_lm.cypha_dif.mean_epistemic_var` | 4.31e-05 |
| `char_lm.cypha_dif.n_experts` | 13 |
| `char_lm.delta_vs_bigram` | 1.1595 |
| `char_lm.delta_vs_trigram` | 0.0210 |
| `char_lm.device` | cpu |
| `char_lm.eval_method` | holdout_20pct |
| `char_lm.expert_count` | [20 items] |
| `char_lm.expert_routing.active_experts_per_step` | [120 items] |
| `char_lm.expert_routing.dominant_expert_per_step` | [120 items] |
| `char_lm.expert_routing.epistemic_var_per_step` | [120 items] |
| `char_lm.expert_routing.mean_active_experts` | 1.0000 |
| `char_lm.expert_routing.unique_experts_used` | [12] |
| `char_lm.final_bpc` | 5.0007 |
| `char_lm.final_train_bpc` | 4.5755 |
| `char_lm.generation_preview.greedy` | tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt |
| `char_lm.generation_preview.prompt_text` | exion,
landlord, which is an intimate and confidential one in th |
| `char_lm.generation_preview.top_p` | ttotttot tttt t ttttntettr tlno tltoae tet
ft t ftt t nttttdahacy to ttntt ti tttatittht tottthtte tttttitfttttteohdsttt |
| `char_lm.held_out_bpc` | [20 items] |
| `char_lm.model` | CyphaLM |

## D05

*Timestamp:* 2026-05-31T04:36:41.629777+00:00

### summary

| Metric | Value |
| --- | --- |
| `baselines.dummy_mean.mae` | 0.9475 |
| `baselines.dummy_mean.r2` | -0.0017 |
| `baselines.dummy_mean.rmse` | 0.9621 |
| `baselines.gradient_boosting.mae` | 0.5293 |
| `baselines.gradient_boosting.r2` | 0.5714 |
| `baselines.gradient_boosting.rmse` | 0.6293 |
| `baselines.knn_5.mae` | 0.4807 |
| `baselines.knn_5.r2` | 0.5271 |
| `baselines.knn_5.rmse` | 0.6611 |
| `baselines.random_forest.mae` | 0.5969 |
| `baselines.random_forest.r2` | 0.5018 |
| `baselines.random_forest.rmse` | 0.6785 |
| `baselines.ridge.mae` | 0.4825 |
| `baselines.ridge.r2` | 0.6582 |
| `baselines.ridge.rmse` | 0.5620 |
| `cypha_metrics.alpha_distribution` | [10 items] |
| `cypha_metrics.expert_count` | 10 |
| `cypha_metrics.fraction_edge_of_chaos` | 0.0000 |
| `cypha_metrics.mean_aleatoric_var` | 0.0665 |
| `cypha_metrics.mean_alpha` | 0.7398 |
| `cypha_metrics.mean_epistemic_var` | 0.2661 |
| `cypha_metrics.std_epistemic_var` | 0.1482 |
| `cypha_metrics.uncertainty_rank_correlation` | 0.5226 |
| `cypha_scores.mae` | 0.4926 |
| `cypha_scores.r2` | 0.6471 |
| `cypha_scores.rmse` | 0.5710 |
| `data_source` | synthetic |
| `domain` | d05_chess |
| `n_samples` | 1000 |
| `sgd_online.mae` | 0.4832 |

## D06

*Timestamp:* 2026-05-31T04:36:50.785109+00:00

### summary

| Metric | Value |
| --- | --- |
| `classification.baselines.dummy_majority.accuracy` | 0.5125 |
| `classification.baselines.dummy_majority.f1_macro` | 0.3388 |
| `classification.baselines.gradient_boosting.accuracy` | 1.0000 |
| `classification.baselines.gradient_boosting.f1_macro` | 1.0000 |
| `classification.baselines.knn_5.accuracy` | 0.6675 |
| `classification.baselines.knn_5.f1_macro` | 0.6533 |
| `classification.baselines.logistic_regression.accuracy` | 1.0000 |
| `classification.baselines.logistic_regression.f1_macro` | 1.0000 |
| `classification.baselines.random_forest.accuracy` | 1.0000 |
| `classification.baselines.random_forest.f1_macro` | 1.0000 |
| `classification.cypha_metrics.alpha_distribution` | [0.8765419022643489, 0.8589689350791195] |
| `classification.cypha_metrics.expert_count` | 2 |
| `classification.cypha_metrics.fraction_edge_of_chaos` | 0.0000 |
| `classification.cypha_metrics.mean_aleatoric_var` | 0.0018 |
| `classification.cypha_metrics.mean_alpha` | 0.8678 |
| `classification.cypha_metrics.mean_epistemic_var` | 0.0041 |
| `classification.cypha_metrics.std_epistemic_var` | 0.0434 |
| `classification.cypha_metrics.uncertainty_rank_correlation` | 0.1219 |
| `classification.cypha_scores.accuracy` | 0.9950 |
| `classification.cypha_scores.f1_macro` | 0.9950 |
| `classification.sgd_online.accuracy` | 1.0000 |
| `classification.sgd_online.f1_macro` | 1.0000 |
| `domain` | d06_go |
| `n_samples` | 2000 |
| `regression.baselines.dummy_mean.mae` | 0.4997 |
| `regression.baselines.dummy_mean.r2` | -1.56e-06 |
| `regression.baselines.dummy_mean.rmse` | 0.4998 |
| `regression.baselines.gradient_boosting.mae` | 1.33e-05 |
| `regression.baselines.gradient_boosting.r2` | 1.0000 |
| `regression.baselines.gradient_boosting.rmse` | 1.33e-05 |

## D07

*Timestamp:* 2026-05-31T04:37:00.892185+00:00

### summary

| Metric | Value |
| --- | --- |
| `baselines.dummy_majority.accuracy` | 0.6150 |
| `baselines.dummy_majority.f1_macro` | 0.2539 |
| `baselines.gradient_boosting.accuracy` | 1.0000 |
| `baselines.gradient_boosting.f1_macro` | 1.0000 |
| `baselines.knn_5.accuracy` | 0.8850 |
| `baselines.knn_5.f1_macro` | 0.8508 |
| `baselines.logistic_regression.accuracy` | 0.9833 |
| `baselines.logistic_regression.f1_macro` | 0.9464 |
| `baselines.random_forest.accuracy` | 1.0000 |
| `baselines.random_forest.f1_macro` | 1.0000 |
| `boundary_uncertainty_spearman` | -0.3313 |
| `cypha_metrics.alpha_distribution` | [0.9385387760907788, 0.9466354697247408, 0.9408376372079887] |
| `cypha_metrics.expert_count` | 3 |
| `cypha_metrics.fraction_edge_of_chaos` | 0.0000 |
| `cypha_metrics.mean_aleatoric_var` | 0.0071 |
| `cypha_metrics.mean_alpha` | 0.9420 |
| `cypha_metrics.mean_epistemic_var` | 0.0026 |
| `cypha_metrics.std_epistemic_var` | 0.0340 |
| `cypha_metrics.uncertainty_rank_correlation` | 0.4478 |
| `cypha_scores.accuracy` | 0.9517 |
| `cypha_scores.f1_macro` | 0.9019 |
| `domain` | d07_poker |
| `n_hands` | 3000 |
| `sgd_online.accuracy` | 0.9400 |
| `sgd_online.f1_macro` | 0.8974 |

## D08

*Timestamp:* 2026-05-31T04:40:01.521472+00:00

### raw

| Metric | Value |
| --- | --- |
| `baselines.dummy_majority.accuracy` | 0.1120 |
| `baselines.dummy_majority.f1_macro` | 0.0201 |
| `baselines.gradient_boosting.accuracy` | 0.8840 |
| `baselines.gradient_boosting.f1_macro` | 0.8822 |
| `baselines.knn_5.accuracy` | 0.9100 |
| `baselines.knn_5.f1_macro` | 0.9090 |
| `baselines.logistic_regression.accuracy` | 0.8840 |
| `baselines.logistic_regression.f1_macro` | 0.8827 |
| `baselines.random_forest.accuracy` | 0.9120 |
| `baselines.random_forest.f1_macro` | 0.9098 |
| `cypha_metrics.alpha_distribution` | [10 items] |
| `cypha_metrics.expert_count` | 10 |
| `cypha_metrics.fraction_edge_of_chaos` | 0.0000 |
| `cypha_metrics.mean_aleatoric_var` | 0.0089 |
| `cypha_metrics.mean_alpha` | 0.8480 |
| `cypha_metrics.mean_epistemic_var` | 0.0143 |
| `cypha_metrics.std_epistemic_var` | 0.0805 |
| `cypha_metrics.uncertainty_rank_correlation` | 0.4712 |
| `cypha_scores.accuracy` | 0.7200 |
| `cypha_scores.f1_macro` | 0.7053 |
| `encoding` | raw |
| `n_test` | 500 |
| `n_train` | 2000 |
| `sgd_online.accuracy` | 0.8500 |
| `sgd_online.f1_macro` | 0.8389 |

### hog

| Metric | Value |
| --- | --- |
| `baselines.dummy_majority.accuracy` | 0.1120 |
| `baselines.dummy_majority.f1_macro` | 0.0201 |
| `baselines.gradient_boosting.accuracy` | 0.9220 |
| `baselines.gradient_boosting.f1_macro` | 0.9209 |
| `baselines.knn_5.accuracy` | 0.9340 |
| `baselines.knn_5.f1_macro` | 0.9345 |
| `baselines.logistic_regression.accuracy` | 0.9480 |
| `baselines.logistic_regression.f1_macro` | 0.9489 |
| `baselines.random_forest.accuracy` | 0.9400 |
| `baselines.random_forest.f1_macro` | 0.9411 |
| `cypha_metrics.alpha_distribution` | [10 items] |
| `cypha_metrics.expert_count` | 10 |
| `cypha_metrics.fraction_edge_of_chaos` | 0.0000 |
| `cypha_metrics.mean_aleatoric_var` | 0.0058 |
| `cypha_metrics.mean_alpha` | 0.7704 |
| `cypha_metrics.mean_epistemic_var` | 0.0096 |
| `cypha_metrics.std_epistemic_var` | 0.0653 |
| `cypha_metrics.uncertainty_rank_correlation` | 0.4958 |
| `cypha_scores.accuracy` | 0.8960 |
| `cypha_scores.f1_macro` | 0.8932 |
| `encoding` | hog |
| `n_test` | 500 |
| `n_train` | 2000 |
| `sgd_online.accuracy` | 0.9200 |
| `sgd_online.f1_macro` | 0.9183 |

## D09

*Timestamp:* 2026-05-31T04:40:42.068032+00:00

### summary

| Metric | Value |
| --- | --- |
| `20news.baselines.dummy_majority.accuracy` | 0.0625 |
| `20news.baselines.dummy_majority.f1_macro` | 0.0059 |
| `20news.baselines.gradient_boosting.accuracy` | 0.1875 |
| `20news.baselines.gradient_boosting.f1_macro` | 0.1810 |
| `20news.baselines.knn_5.accuracy` | 0.1500 |
| `20news.baselines.knn_5.f1_macro` | 0.1347 |
| `20news.baselines.logistic_regression.accuracy` | 0.3312 |
| `20news.baselines.logistic_regression.f1_macro` | 0.3343 |
| `20news.baselines.random_forest.accuracy` | 0.3187 |
| `20news.baselines.random_forest.f1_macro` | 0.2944 |
| `20news.cypha_metrics.alpha_distribution` | [20 items] |
| `20news.cypha_metrics.expert_count` | 20 |
| `20news.cypha_metrics.fraction_edge_of_chaos` | 0.0000 |
| `20news.cypha_metrics.mean_aleatoric_var` | 0.3465 |
| `20news.cypha_metrics.mean_alpha` | 0.8796 |
| `20news.cypha_metrics.mean_epistemic_var` | 1.0719 |
| `20news.cypha_metrics.std_epistemic_var` | 0.6871 |
| `20news.cypha_metrics.uncertainty_rank_correlation` | 0.4093 |
| `20news.cypha_scores.accuracy` | 0.3312 |
| `20news.cypha_scores.f1_macro` | 0.3086 |
| `20news.n_samples` | 800 |
| `20news.sgd_online.accuracy` | 0.3563 |
| `20news.sgd_online.f1_macro` | 0.3209 |
| `domain` | d09_documents |
| `gutenberg_book_classification.cypha_scores.accuracy` | 0.6458 |
| `gutenberg_book_classification.cypha_scores.f1_macro` | 0.6431 |
| `gutenberg_book_classification.n_segments` | 240 |
| `gutenberg_book_classification.sgd_online.accuracy` | 0.3333 |
| `gutenberg_book_classification.sgd_online.f1_macro` | 0.1667 |
| `gutenberg_ood.mannwhitney_u` | 36960.0000 |

## D10

*Timestamp:* 2026-05-31T04:41:26.399449+00:00

### 10A_ecg_classification

| Metric | Value |
| --- | --- |
| `accuracy` | 0.2000 |
| `data_source` | synthetic |
| `expert_count` | 5 |
| `f1_macro` | 0.0667 |
| `mean_confidence` | 0.3498 |
| `mean_epistemic_var` | 0.8160 |
| `uncertainty_rank_correlation` | 0.4887 |

### 10B_ecg_sliding_window

| Metric | Value |
| --- | --- |
| `accuracy` | 0.1746 |
| `expert_count` | 5 |
| `f1_macro` | 0.1223 |
| `mean_confidence` | 0.3059 |
| `mean_epistemic_var` | 0.8465 |
| `uncertainty_rank_correlation` | 0.1905 |

### 10C_ecg_ood_detection

| Metric | Value |
| --- | --- |
| `ood_auroc` | 0.6422 |

### 10D_financial_return_sign

| Metric | Value |
| --- | --- |
| `accuracy` | 0.4990 |
| `expert_count` | 2 |
| `f1_macro` | 0.3329 |
| `mean_confidence` | 0.9657 |
| `mean_epistemic_var` | 0.2086 |
| `note` | near_chance_expected |
| `uncertainty_rank_correlation` | -0.0107 |

## D11

*Timestamp:* 2026-05-31T04:42:04.885620+00:00

### 11A_cartpole_value_regression

| Metric | Value |
| --- | --- |
| `expert_count` | 8 |
| `mae` | 8.6827 |
| `mean_epistemic_near_tip` | 107.4492 |
| `mean_epistemic_var` | 102.8552 |
| `r2` | 0.0120 |
| `ridge_r2` | -0.0459 |
| `ridge_rmse` | 11.4404 |
| `rmse` | 11.1192 |
| `uncertainty_rank_correlation` | -0.0670 |

### 11B_gridworld_q_estimation

| Metric | Value |
| --- | --- |
| `n_pairs` | 13 |
| `q_value_mae` | 0.2998 |

### 11C_trajectory_preference

| Metric | Value |
| --- | --- |
| `accuracy` | 0.9200 |
| `expert_count` | 2 |
| `f1_macro` | 0.9184 |
| `mean_confidence` | 0.9154 |
| `mean_epistemic_var` | 0.7372 |
| `uncertainty_rank_correlation` | 0.3205 |

## D12

*Timestamp:* 2026-05-31T04:45:10.626968+00:00

### 12A_binary_intrusion

| Metric | Value |
| --- | --- |
| `cypha_ood_auroc` | 0.8892 |
| `data_source` | nsl_kdd |
| `isolation_forest_auroc` | 0.9397 |

### 12B_attack_types

| Metric | Value |
| --- | --- |
| `mean_epistemic_attack` | 0.6268 |
| `n_test` | 2000 |

### 12C_online_detection

| Metric | Value |
| --- | --- |
| `detection_latency_steps` | 5 |
| `final_attack_acc` | 0.9950 |

## D13

*Timestamp:* 2026-05-31T04:45:26.054188+00:00

### 13A_alpha_vs_compression

| Metric | Value |
| --- | --- |
| `files` | [11 items] |
| `gzip_ratios` | [11 items] |
| `mean_alpha_proxy` | [11 items] |
| `spearman_alpha_vs_gzip` | 0.0727 |

### 13B_binary_vs_text_alpha

| Metric | Value |
| --- | --- |
| `mean_alpha_binary` | 0.6120 |
| `mean_alpha_text` | 0.7717 |

## D14

*Timestamp:* 2026-05-31T04:45:54.918111+00:00

### 14A_feynman_all_equations

| Metric | Value |
| --- | --- |
| `mean_r2` | -0.0103 |
| `mean_rmse` | 25028925797.8449 |
| `per_equation.Stefan_Boltzmann.expert_count` | 8 |
| `per_equation.Stefan_Boltzmann.mae` | 351.9854 |
| `per_equation.Stefan_Boltzmann.mean_epistemic_var` | 352525.3028 |
| `per_equation.Stefan_Boltzmann.r2` | 7.89e-04 |
| `per_equation.Stefan_Boltzmann.ridge_rmse` | 310.1590 |
| `per_equation.Stefan_Boltzmann.rmse` | 492.5692 |
| `per_equation.Stefan_Boltzmann.uncertainty_rank_correlation` | -0.0937 |
| `per_equation.bernoulli.expert_count` | 8 |
| `per_equation.bernoulli.mae` | 8.5845 |
| `per_equation.bernoulli.mean_epistemic_var` | 100.4290 |
| `per_equation.bernoulli.r2` | -0.0189 |
| `per_equation.bernoulli.ridge_rmse` | 5.4996 |
| `per_equation.bernoulli.rmse` | 11.7362 |
| `per_equation.bernoulli.uncertainty_rank_correlation` | 0.2881 |
| `per_equation.capacitor_energy.expert_count` | 8 |
| `per_equation.capacitor_energy.mae` | 9.2912 |
| `per_equation.capacitor_energy.mean_epistemic_var` | 189.3765 |
| `per_equation.capacitor_energy.r2` | -0.0027 |
| `per_equation.capacitor_energy.ridge_rmse` | 5.7637 |
| `per_equation.capacitor_energy.rmse` | 12.1753 |
| `per_equation.capacitor_energy.uncertainty_rank_correlation` | -0.0956 |
| `per_equation.centripetal.expert_count` | 8 |
| `per_equation.centripetal.mae` | 17.1032 |
| `per_equation.centripetal.mean_epistemic_var` | 554.5242 |
| `per_equation.centripetal.r2` | -0.0197 |
| `per_equation.centripetal.ridge_rmse` | 37.0781 |
| `per_equation.centripetal.rmse` | 45.6423 |
| `per_equation.centripetal.uncertainty_rank_correlation` | 0.1399 |

### 14B_extrapolation_uncertainty

| Metric | Value |
| --- | --- |
| `extrapolation_auroc` | 1.0000 |
| `regressor_uncertainty_auroc` | 0.0000 |

### 14C_noise_vs_aleatoric

| Metric | Value |
| --- | --- |
| `0.0.mean_epistemic_var` | 7.2489 |
| `0.0.rmse` | 3.3466 |
| `0.05.mean_epistemic_var` | 7.2848 |
| `0.05.rmse` | 3.3514 |
| `0.1.mean_epistemic_var` | 7.3371 |
| `0.1.rmse` | 3.3599 |
| `0.2.mean_epistemic_var` | 7.4904 |
| `0.2.rmse` | 3.3876 |
| `0.5.mean_epistemic_var` | 8.3414 |
| `0.5.rmse` | 3.5525 |

## D15

*Timestamp:* 2026-05-31T04:46:04.565827+00:00

### 15A_gaussian_noise

| Metric | Value |
| --- | --- |
| `0.0.accuracy` | 0.8467 |
| `0.0.mean_epistemic_var` | 0.4044 |
| `0.1.accuracy` | 0.8289 |
| `0.1.mean_epistemic_var` | 0.4038 |
| `0.2.accuracy` | 0.8156 |
| `0.2.mean_epistemic_var` | 0.3985 |
| `0.5.accuracy` | 0.6222 |
| `0.5.mean_epistemic_var` | 0.3817 |
| `1.0.accuracy` | 0.3711 |
| `1.0.mean_epistemic_var` | 0.4622 |

### 15B_feature_dropout

| Metric | Value |
| --- | --- |
| `0.1.accuracy` | 0.7422 |
| `0.1.mean_epistemic_var` | 0.4440 |
| `0.25.accuracy` | 0.6000 |
| `0.25.mean_epistemic_var` | 0.4849 |
| `0.5.accuracy` | 0.4089 |
| `0.5.mean_epistemic_var` | 0.5477 |
| `0.75.accuracy` | 0.2733 |
| `0.75.mean_epistemic_var` | 0.6152 |

### 15C_adversarial_fgsm_proxy

| Metric | Value |
| --- | --- |
| `accuracy_adversarial` | 0.8667 |
| `accuracy_natural` | 0.8578 |
| `mean_epistemic_adversarial` | 0.4048 |
| `mean_epistemic_natural` | 0.4093 |

## D16

*Timestamp:* 2026-05-31T04:46:23.964314+00:00

### 16A_task_discovery

| Metric | Value |
| --- | --- |
| `expert_count` | 16 |
| `per_task_accuracy.digits` | 0.8111 |
| `per_task_accuracy.iris` | 0.8421 |
| `per_task_accuracy.wine` | 0.8000 |
| `routing_ari` | 1.0000 |

### 16B_forgetting_resistance

| Metric | Value |
| --- | --- |
| `forgetting_score` | 0.8125 |
| `task_a_accuracy_after` | 0.1579 |
| `task_a_accuracy_before` | 0.8421 |

### 16D_interleaving_comparison

| Metric | Value |
| --- | --- |
| `block.digits` | 0.8689 |
| `block.iris` | 0.8421 |
| `block.wine` | 0.8444 |
| `random.digits` | 0.8333 |
| `random.iris` | 0.8158 |
| `random.wine` | 0.8889 |
| `round_robin.digits` | 0.8489 |
| `round_robin.iris` | 0.7895 |
| `round_robin.wine` | 0.8222 |

### 16E_save_restore

| Metric | Value |
| --- | --- |
| `retention_ratio` | 1.0000 |
| `task_a_before` | 0.8421 |
| `task_a_corrupted` | 0.6579 |
| `task_a_restored` | 0.8421 |

### 16F_per_task_models

| Metric | Value |
| --- | --- |
| `forgetting_score` | 0.0000 |
| `note` | per-task isolated models — zero forgetting by architecture |
| `per_task_accuracy.digits` | 0.8533 |
| `per_task_accuracy.iris` | 0.8158 |
| `per_task_accuracy.wine` | 0.9778 |

## D17

*Timestamp:* 2026-05-31T06:24:49.528509+00:00

### 17A_bits_per_character

| Metric | Value |
| --- | --- |
| `bigram_bpc` | 3.9142 |
| `cypha_dif.lossless_fraction` | 4.20e-05 |
| `cypha_dif.mean_aleatoric_var` | 0.9144 |
| `cypha_dif.mean_alpha` | 0.1068 |
| `cypha_dif.mean_epistemic_var` | 3.84e-05 |
| `cypha_dif.n_experts` | 13 |
| `cyphalm_bpc` | 4.6583 |
| `delta_vs_bigram` | 0.7440 |
| `delta_vs_trigram` | 0.2606 |
| `device` | cpu |
| `final_train_bpc` | 4.6673 |
| `learning_curve.expert_count` | [20 items] |
| `learning_curve.final_train_bpc` | 4.6673 |
| `learning_curve.held_out_bpc` | [20 items] |
| `learning_curve.online_train_bpc` | 4.7193 |
| `learning_curve.steps` | [20 items] |
| `learning_curve.trained_steps` | 40000 |
| `online_train_bpc` | 4.7193 |
| `profile.alpha_init` | 0.5000 |
| `profile.context_length` | 256 |
| `profile.d_embed` | 64 |
| `profile.d_state` | 128 |
| `profile.device` | auto |
| `profile.field_dim` | 160 |
| `profile.gria_lr` | 0.0600 |
| `profile.max_experts` | 128 |
| `profile.n_experts` | 4 |
| `profile.online` | True |
| `profile.seed` | 42 |
| `profile.ssm_layers` | 2 |

### 17B_alpha_spectrum

| Metric | Value |
| --- | --- |
| `fraction_edge_of_chaos` | 0.0000 |
| `mean_alpha` | 0.1068 |
| `n_experts` | 11 |

### 17D_online_adaptation

| Metric | Value |
| --- | --- |
| `bpc_improvement` | 0.2870 |
| `bpc_ood_after_adapt` | 4.8186 |
| `bpc_ood_before_adapt` | 5.1057 |

## Cross-Domain Analyses

### cross_alpha_spectrum_global

| Metric | Value |
| --- | --- |
| `summary.global_mean_alpha` | 0.7720 |
| `summary.global_std_alpha` | 0.3022 |
| `summary.n_measurements` | 13 |
| `summary.within_gul_band_fraction` | 0.0000 |

### cross_forgetting_resistance

| Metric | Value |
| --- | --- |
| `mean_forgetting_score` | 0.4062 |

### cross_online_adaptation

| Metric | Value |
| --- | --- |
| `n_domains_with_adaptation_signal` | 2 |

### cross_uncertainty_calibration

| Metric | Value |
| --- | --- |
| `summary.mean_ood_auroc` | 0.8438 |
| `summary.n_experiments` | 43 |
