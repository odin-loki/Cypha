# Cypha Bench

Multi-domain evaluation suite for CyphaDIF and related components. Runs **17 task domains** (d01–d17) plus **4 cross-domain analyses** comparing Cypha against online baselines (SGD, sklearn, etc.).

## Domain structure

| ID | Module | Focus |
|----|--------|-------|
| d01 | `d01_statistical_baselines` | Synthetic classification/regression baselines |
| d02 | `d02_regression` | Real-world regression tasks |
| d03 | `d03_classification` | Tabular/image classification |
| d04 | `d04_generation_language` | Character-level language modelling |
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
| d17 | `d17_cyphalm_integration` | CyphaLM integration |

Cross-domain analyses (`cross_domain/`): uncertainty calibration, online adaptation, forgetting resistance, alpha spectrum.

## Install

From the repository root:

```powershell
pip install -r cypha_bench/requirements.txt
pip install -r requirements-verify.txt   # pytest + REST deps for CI parity
```

Optional datasets: `python cypha_bench/setup/acquire_data.py`

## Run the full benchmark

From the repository root:

```powershell
python cypha_bench/run_all.py
```

This runs all 17 domains, cross-domain analyses, and regenerates the HTML report under `cypha_bench/report/`.

Options:

```powershell
python cypha_bench/run_all.py --domain 4          # single domain (1–17)
python cypha_bench/run_all.py --from-domain 10    # resume from domain 10
python cypha_bench/run_all.py --report-only       # rebuild report from saved tables
```

## Run a single domain

```powershell
python cypha_bench/domains/d04_generation_language.py
python cypha_bench/run_all.py --domain 4
```

## Results

Baseline numbers and tuning reports are summarized in [docs/RESEARCH_STATUS.md](../docs/RESEARCH_STATUS.md). Detailed per-domain tables live in `cypha_bench/report/tables/`; see also `BASELINE_REPORT.md` and related reports in this directory.

## Note on `benchmark_baseline.py`

The root-level `benchmark_baseline.py` script is a **separate** SOM upgrade benchmark (`cypha_som`), not the 17-domain Cypha Bench suite. Use `cypha_bench/run_all.py` for domain evaluation.
