# cypha_bench — domain modules

Each file under this directory implements one **benchmark domain** (D01–D17). Domains are invoked by `cypha_bench/run_all.py` or directly via `python cypha_bench/domains/dNN_*.py`.

## Language-model domains (CyphaLM)

| ID | Module | Corpus | Profile key |
|----|--------|--------|-------------|
| **D04** | `d04_generation_language.py` | Gutenberg (Moby Dick, etc.) | `d04` → `config/profiles/cyphalm_d04_gutenberg.json` |
| **D17** | `d17_cyphalm_integration.py` | WikiText-2 official splits | `d17` → `config/profiles/cyphalm_d17_wikitext.json` |

Shared adapters: `cypha_bench/adapters/cyphalm_bench.py`, `char_lstm_baseline.py`, `cyphalm_ablations.py`.

### D04 experiments

- Held-out BPC + training curve (`fig04_char_lm_training.png`)
- BPC vs context length (`fig04_context_bpc.png`)
- CyphaDIF expert routing during generation (`fig04_expert_routing.png`)
- Save/restore checkpoint fidelity
- Sampling strategies (greedy, temperature, top-k, top-p, uncertainty-gated)
- **Baselines:** bigram, trigram, 4-gram, 5-gram, char-LSTM (skipped in fast mode)
- **Ablations:** `context_mode` sweep (subset in fast mode — see main bench README)

### D17 experiments

| Key | Experiment |
|-----|------------|
| `17a` | Held-out BPC + learning curve vs n-gram / LSTM baselines |
| `17b` | Alpha spectrum / expert count |
| `17c` | Cross-corpus online adaptation |
| `17d` | OOD adaptation BPC gain |
| `17e` | Multi-view schedule comparison (`same_order` vs `schedule_a` / `schedule_b`) |

Requires `cypha_bench/data/wikitext2/wikitext-2/wiki.{train,valid,test}.tokens` (see `setup/acquire_data.py`). Bench fails on synthetic fallback unless `CYPHA_BENCH_FAST=1` or `CYPHA_BENCH_ALLOW_SYNTHETIC=1`.

### Run a single LM domain

```powershell
pip install -e cypha_lm/
python cypha_bench/domains/d04_generation_language.py
python cypha_bench/domains/d17_cyphalm_integration.py
python cypha_bench/run_all.py --domain 4
python cypha_bench/run_all.py --domain 17
```

### Full-corpus WikiText training

```powershell
$env:CYPHA_BENCH_FULL_CORPUS="1"
python cypha_bench/run_all.py --domain 17
```

Uses effectively all WikiText train tokens and a larger eval window (see `cypha_bench/README.md`).

## Non-LM domains (summary)

| ID | Module | Focus |
|----|--------|-------|
| d01 | `d01_statistical_baselines.py` | Synthetic classification/regression |
| d02 | `d02_regression.py` | Real-world regression |
| d03 | `d03_classification.py` | Tabular classification |
| d05 | `d05_chess.py` | Chess rating / moves |
| d06 | `d06_go.py` | Go board prediction |
| d07 | `d07_poker.py` | Poker hand evaluation |
| d08 | `d08_computer_vision.py` | MNIST / vision |
| d09 | `d09_documents.py` | Document classification |
| d10 | `d10_time_series.py` | Time-series forecasting |
| d11 | `d11_reinforcement_learning.py` | RL environments |
| d12 | `d12_anomaly_detection.py` | Anomaly detection |
| d13 | `d13_compression.py` | Compression metrics |
| d14 | `d14_symbolic_regression.py` | Symbolic regression |
| d15 | `d15_adversarial_robustness.py` | Adversarial robustness |
| d16 | `d16_multitask.py` | Multi-task / continual |

### D16 experiments (selected)

| Key | Experiment |
|-----|------------|
| `16b` | Sequential-task catastrophic forgetting |
| `16g` | Round-robin vs task-block-shuffle view streams (~3k steps) |

Cross-domain analyses live in `cypha_bench/cross_domain/`.

## Environment variables (LM-relevant)

| Variable | Effect |
|----------|--------|
| `CYPHA_BENCH_FAST=1` | Smaller corpora, fewer train steps; LM: skip 4/5-gram, char-LSTM, most ablations |
| `CYPHA_BENCH_FULL_CORPUS=1` | WikiText: train on full `wiki.train.tokens`; larger `n_eval` |
| `CYPHA_BENCH_ALLOW_SYNTHETIC=1` | Allow synthetic text if data missing |
| `CYPHALM_PROFILE` | Override profile id (`d04`, `d17`, `llm`, …) |
| `CYPHA_LM_DEVICE` | `cpu` \| `cuda` \| `auto` for CyphaLM |

Full table: [`../README.md`](../README.md).
