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

Per-domain notes: [`domains/README.md`](domains/README.md).

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
| Sampling strategies | `sampling_strategies` | `fig04_sampling_strategies.png` |
| N-gram + char-LSTM baselines | `bigram_bpc`, …, `char_lstm_bpc` | — |
| Context-mode ablations | `ablations` | — |

Shared LM helpers: `cypha_bench/adapters/cyphalm_bench.py`, `cyphalm_ablations.py`, `char_lstm_baseline.py`.

## Language-model baselines (D04 / D17)

All baselines are scored on the **same held-out split** as CyphaLM, using the training slice seen by the model.

| Baseline | Function | Notes |
|----------|----------|-------|
| Bigram | `bigram_baseline_bpc()` | Character 2-gram (Laplace +1 in `ngram_baseline_bpc`) |
| Trigram | `trigram_baseline_bpc()` | Character 3-gram |
| 4-gram | `fourgram_baseline_bpc()` | Character 4-gram |
| 5-gram | `fivegram_baseline_bpc()` | Character 5-gram |
| Char-LSTM | `char_lstm_baseline_bpc()` | 1-layer NumPy LSTM; no PyTorch |

### Baseline table (held-out BPC, bits/char)

Refresh after a full bench run: `python cypha_bench/run_all.py --domain 4` and `--domain 17`.

| Corpus | CyphaLM | Bigram | Trigram | 4-gram | 5-gram | Char-LSTM |
|--------|---------|--------|---------|--------|--------|-----------|
| D04 Gutenberg (40k train) | **4.122** | 3.931 | **4.522** | 5.592 | 5.949 | 3.505 |
| D17 WikiText valid (40k train) | **4.154** | 3.914 | **4.398** | 5.286 | 5.579 | 3.589 |
| D17 WikiText valid (**full train**, `CYPHA_BENCH_FULL_CORPUS=1`) | *run bench* | *run bench* | *run bench* | *run bench* | *run bench* | *run bench* |

Last pinned numbers (pre-upgrade sweep, 2026-05-31): see [docs/RESEARCH_STATUS.md](../docs/RESEARCH_STATUS.md) and `BASELINE_REPORT.md`.

## CyphaLM ablations

Ablation modes map to `CyphaLMConfig.context_mode` (see [`cypha_lm/README.md`](../cypha_lm/README.md)).

| Mode | What is measured |
|------|------------------|
| `full` | Default stack: SSM + CyphaDIF + GRIA |
| `gria_ngram` | SSM projection + last *K* token embeddings (`ngram_context`) |
| `ssm_only` | SSM path only; DIF contribution zeroed in GRIA input |
| `ablation_no_dif` | Field mean only; no expert routing |
| `ablation_no_ssm` | N-gram embed concat only; SSM state zeroed |

Runner: `cypha_bench/adapters/cyphalm_ablations.py` — `run_lm_ablations(corpus, limits, modes)`.

D04/D17 include an `ablations` block in domain JSON when not in fast mode.

### Run ablations only (dev)

```powershell
pip install -e cypha_lm/
python -c "
from cypha_bench.adapters.cyphalm_bench import cyphalm_bench_limits, prepare_lm_corpus
from cypha_bench.adapters.cyphalm_ablations import run_lm_ablations
corpus = prepare_lm_corpus(prefer_wikitext=True)
limits = cyphalm_bench_limits()
modes = ['full', 'gria_ngram', 'ssm_only', 'ablation_no_dif', 'ablation_no_ssm']
print(run_lm_ablations(corpus, limits, modes, profile='d17'))
"
```

Fast bench (`CYPHA_BENCH_FAST=1`): runs **`full`** and **`gria_ngram`** only; skips 4/5-gram, char-LSTM, and other ablation modes.

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `CYPHA_BENCH_FAST` | `0` | Smaller corpora and train caps; LM skips heavy baselines/ablations |
| `CYPHA_BENCH_FULL_CORPUS` | `0` | WikiText D17: `n_train` ≈ full `wiki.train.tokens`; larger `n_eval` |
| `CYPHA_BENCH_ALLOW_SYNTHETIC` | `0` | Allow synthetic corpus if data files missing |
| `CYPHA_BENCH_USE_PROFILE` | `1` | Use everyday profile for D01–D16 (not CyphaLM profiles) |
| `CYPHA_BENCH_PROFILE_PATH` | — | Override everyday profile JSON path |
| `CYPHALM_PROFILE` | — | CyphaLM profile id: `d04`, `d17`, `llm`, or path via loader |
| `CYPHA_LM_DEVICE` | `auto` | CyphaLM compute device |
| `CYPHA_LM_FAST` | `0` | Shorter `cypha_lm` experiment scripts |

### Examples

```powershell
# Quick smoke (3k train, ablations: full + gria_ngram only)
$env:CYPHA_BENCH_FAST="1"
python cypha_bench/run_all.py --domain 17

# Full WikiText train for beat-bigram experiments
$env:CYPHA_BENCH_FULL_CORPUS="1"
python cypha_bench/run_all.py --domain 17

# Gutenberg CyphaLM only
python cypha_bench/run_all.py --domain 4
```

## Install

From the repository root:

```powershell
pip install -r cypha_bench/requirements.txt
pip install -e cypha_lm/                         # required for D04 + D17
pip install -r requirements-verify.txt   # pytest + REST deps for CI parity
```

Optional datasets: `python cypha_bench/setup/acquire_data.py`

## Domain profiles

Hyperparameters are **domain-specific**, not one global bundle. Registry: `cypha_bench/config/profiles_index.json`.

| Profile | File | Used by |
|---------|------|---------|
| `classification` / `vision` | `everyday_profile.json` | D01–D03, D08–D09 |
| `regression` | `everyday_profile.json` (`regime=regression`) | D02, D05–D07, D10 |
| `llm` (default) | `profiles/cyphalm_llm.json` | CyphaLM CLI, legacy `cyphalm_profile.json` |
| `d04` | `profiles/cyphalm_d04_gutenberg.json` | D04 Gutenberg char-LM |
| `d17` | `profiles/cyphalm_d17_wikitext.json` | D17 WikiText integration |

CyphaLM profile fields (upgrade): `context_mode`, `ngram_context`, `train_epochs`, `bptt_steps`, `laplace_smoothing`, `gria_lr_decay` — see `cypha_lm/README.md`.

Override at runtime: `$env:CYPHALM_PROFILE="d17"` or `make_cyphalm(profile="d04")`.

Tune CyphaLM profiles:

```powershell
# Full grid on both corpuses → writes domain JSON + sweep tables
python cypha_bench/tuning/cyphalm_sweep.py --corpus both --n-train 8000 --write-profile --skip-axis

# Single-axis sweeps (one knob at a time)
python cypha_bench/tuning/cyphalm_sweep.py --corpus d17 --skip-full --axis gria_lr
python cypha_bench/tuning/cyphalm_sweep.py --corpus both --skip-full   # all axes
```

Outputs: `cypha_bench/config/cyphalm_profile_sweep.json`, `cyphalm_profile_axis_sweeps.json`.

### Multi-view online training

Structure-preserving corpus reorderings per macro-epoch (`cypha_views/`). Spec: [`docs/MULTI_VIEW_TRAINING_PLAN.md`](../docs/MULTI_VIEW_TRAINING_PLAN.md).

| Env / profile | Effect |
|---------------|--------|
| `view_schedule` in CyphaLM profile | `same_order`, `schedule_a`, `schedule_b`, `schedule_c` |
| `view_block_size` | Block size for shuffle/segment resets (default 512) |

D17 experiment **17E_multi_view** compares schedules vs `same_order` BPC. D16 **16G_view_streams** compares task-block shuffle vs round-robin.

## Run the full benchmark

```powershell
python cypha_bench/run_all.py
python cypha_bench/run_all.py --domain 4          # CyphaLM char-LM only
python cypha_bench/run_all.py --domain 17         # CyphaLM integration
python cypha_bench/run_all.py --from-domain 10    # resume
python cypha_bench/run_all.py --report-only
```

Regenerate `BASELINE_REPORT.md` after a run:

```powershell
python cypha_bench/report/generate_report.py
```

Committed `cypha_bench/report/` JSON/PNG files are **baseline snapshots** for docs and native `cypha_bench_report` smoke. Re-commit only when intentionally refreshing the published baseline; casual local reruns can leave git dirty — use `git restore cypha_bench/report cypha_bench/BASELINE_REPORT.md` to discard.

## Results

Baseline numbers: [docs/RESEARCH_STATUS.md](../docs/RESEARCH_STATUS.md). Per-domain tables: `cypha_bench/report/tables/`; figures: `cypha_bench/report/figures/`.

## Note on `benchmark_baseline.py`

Root-level `benchmark_baseline.py` is a **separate** SOM upgrade benchmark (`cypha_som`), not the 17-domain suite. Use `cypha_bench/run_all.py` for domain evaluation.
