# CyphaLM Experiment Report

Generated: 2026-05-23 05:49 UTC
Mode: fast (CI)
Total runtime: 211.4s

## Experiment Metrics

### 01_embedding_benchmark

- **knn_accuracy**: {'Izaac (zero-param)': 0.0125, 'Random learned': 0.0025, 'Frequency-init learned': 0.01}
- **collision_rate**: {'Izaac (zero-param)': 0.0, 'Random learned': 0.0, 'Frequency-init learned': 0.0}
- **gf_rank_spearman**: 0.1434
- **vocab_size**: 512

### 02_ssm_sequence_capacity

- **lengths**: 10, 50, 100
- **cellai_lag1_acc**: 0.0625, 0.08163, 0.06566
- **cellai_long_acc**: 0.1, 0.09, 0.07375
- **rnn_lag1_acc**: 0.3333, 0.1339, 0.08965
- **transformer_lag1_acc**: 1, 0.75, 0.875
- **transformer_long_acc**: 1, 1, 0.875
- **cellai_time_ms**: 94.15, 482, 965.8
- **transformer_time_ms**: 67.27, 92.43, 169.6

### 03_expert_self_organisation

- **n_steps**: 1000
- **expert_purity**: 1
- **n_experts**: 1
- **expected_purity_threshold**: 0.85

### 04_alpha_spectrum_emergence

- **edge_fractions**: {'0': 1.0, '200': 0.4263565891472868, '500': 0.4186046511627907, '2000': 0.4883720930232558}
- **final_mean_alpha**: 0.472
- **ood_mean_alpha**: 0.39
- **id_mean_alpha**: 0.472

### 05_lm_training_toy_vocab

- **n_steps**: 5000
- **perplexity**: 12.9
- **bigram_perplexity**: 4.081
- **bits_per_token**: 3.756
- **syntax_valid_rate**: 0
- **lossless_fraction**: 0.0001999

### 06_lm_training_code_corpus

- **n_steps**: 20000
- **bits_per_char**: 0.6146
- **test_bits_per_token**: 4.302
- **n_experts**: 1
- **mean_ood_epistemic**: 4.93e-05
- **generated_sample**: `

_d3c  f:udd tt
dtn)1 r(nuf +tuc:cc: ff`

### 07_uncertainty_calibration

- **ece_nig**: 0
- **ece_dropout_baseline**: 0.3702
- **n_eval_tokens**: 500
- **monotonic_buckets**: 5.124, 5.124, 5.124, 5.124, 5.124...

### 08_online_adaptation

- **checkpoints**: 1, 10, 100
- **ppl_online**: 2300, 864.3, 109.2
- **ppl_frozen**: 2357, 2357, 2357
- **final_expert_count**: 1
- **ppl_improvement**: 2248

### 09_catastrophic_forgetting

- **ppl_a_before**: 57.43
- **ppl_a_after**: 1512
- **ppl_b_after**: 54.25
- **retention_ratio**: 26.33
- **fine_tune_retention_ratio**: 26.33
- **within_10pct**: False

### 10_parameter_efficiency

- **configs**: {'d_embed': 32, 'd_state': 64, 'max_experts': 32}, {'d_embed': 64, 'd_state': 128, 'max_experts': 64}
- **cypha_params**: 69888, 220416
- **cypha_bpc**: 10.05, 10.05
- **cypha_latency_ms**: 90.03, 101.4
- **transformer_params**: 76672, 230464
- **transformer_bpc**: 0.0002079, 0.0001801
- **transformer_latency_ms**: 689.9, 2212

## Benchmarks

### perplexity

- **train_steps**: 500
- **eval_tokens**: 200
- **perplexity**: 171.4
- **train_seconds**: 1.086
- **tokens_per_second**: 460.3

### memory

- **peak_bytes_tracemalloc**: 5057951
- **after_init_bytes**: 4329663
- **static_array_bytes**: 2060288
- **n_experts**: 1
- **n_steps**: 500

### latency

- **predict**:
  - n_tokens: 200
  - mean_ms: 1.03
  - p50_ms: 1.003
  - p95_ms: 1.11
  - tokens_per_sec: 970.9
- **train_step**:
  - n_steps: 100
  - total_seconds: 0.2185
  - steps_per_sec: 457.6

## Summary

- Izaac kNN accuracy: 0.0125
- Expert purity: 1
- Toy LM perplexity: 12.9
- Forgetting retention: 26.33

Figures written to `paper/figures/`.