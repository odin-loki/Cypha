# Cypha Bench

Multi-domain evaluation suite for CyphaDIF, **CyphaLM**, and related components. Runs **17 task domains** (d01–d17) plus **4 cross-domain analyses**.

## Domain structure

| ID | Module | Focus |
|----|--------|-------|
| d01 | `d01_statistical_baselines` | Synthetic classification/regression baselines |
| d02 | `d02_regression` | Real-world regression tasks |
| d03 | `d03_classification` | Tabular/image classification |
| **d04** | **`d04_generation_language`** | **CyphaLM char-LM + generation (full LLM stack)** |
| d05 | `d05_chess` | Chess move prediction |
| d06 | `d06_go` | Go board prediction |
| d07 | `d07_poker` | Poker hand evaluation |
| d08 | `d08_computer_vision` | MNIST / vision encoders |
| d09 | `d09_documents` | Document classification |
| d10 | `d10_time_series` | Time-series forecasting |
| d11 | `d11_reinforcement_learning` | RL environments |
| d12 | `d12_anomaly_detection` | Anomaly detection |
| d13 | `d13_compression` | Model compression metrics |
| d14 | `d14_symbolic_regression` | Symbolic regression |
| d15 | `d15_adversarial_robustness` | Adversarial perturbations |
| d16 | `d16_multitask` | Multi-task learning |
| **d17** | **`d17_cyphalm_integration`** | **CyphaLM extended integration (WikiText, alpha, OOD adapt)** |

Cross-domain analyses (`cross_domain/`): uncertainty calibration, online adaptation, forgetting resistance, alpha spectrum.

## D04 — CyphaLM language model domain

**Pipeline:** Izaac → CellAI SSM → CyphaDIF → GRIA

**Experiments reported:**

| Experiment | Output key | Figure |
|------------|------------|--------|
| Held-out BPC + training curve | `final_bpc`, `held_out_bpc` | `fig04_char_lm_training.png` |
| BPC vs context length | `context_length_bpc` | `fig04_context_bpc.png` |
| CyphaDIF expert routing during generation | `expert_routing` | `fig04_expert_routing.png` |
| Save/restore checkpoint fidelity | `save_restore.parity_ok` | — |
| Sampling strategies (greedy, temp, top-k, top-p, uncertainty-gated) | `sampling_strategies` | `fig04_sampling_strategies.png` |

Shared LM helpers: `cypha_bench/adapters/cyphalm_bench.py`

## Install

From the repository root:

```powershell
pip install -r cypha_bench/requirements.txt
pip install -e cypha_lm/                         # required for D04 + D17
pip install -r requirements-verify.txt   # pytest + REST deps for CI parity
```

Optional datasets: `python cypha_bench/setup/acquire_data.py`

## Run the full benchmark

```powershell
python cypha_bench/run_all.py
python cypha_bench/run_all.py --domain 4          # CyphaLM char-LM only
python cypha_bench/run_all.py --from-domain 10    # resume
python cypha_bench/run_all.py --report-only
```

Fast mode (smaller corpora / fewer train steps):

```powershell
$env:CYPHA_BENCH_FAST="1"
python cypha_bench/run_all.py --domain 4
```

## Results

Baseline numbers: [docs/RESEARCH_STATUS.md](../docs/RESEARCH_STATUS.md). Per-domain tables: `cypha_bench/report/tables/`; figures: `cypha_bench/report/figures/`.

## Note on `benchmark_baseline.py`

Root-level `benchmark_baseline.py` is a **separate** SOM upgrade benchmark (`cypha_som`), not the 17-domain suite. Use `cypha_bench/run_all.py` for domain evaluation.
