# CyphaLM

Explicit-mechanism language model: Izaac GF(2^n) embeddings, CellAI SSM, CyphaDIF NIG experts, and GRIA alpha-projection. Core modules use NumPy only; PyTorch is optional for baseline benchmarks.

## Install

From the repository root:

```bash
pip install -e cypha_lm/
pip install -e "cypha_lm/[benchmark]"   # torch + pytest-benchmark
```

## Run tests

```bash
pytest cypha_lm/ -v --tb=short
```

## Experiments

Each script is standalone and writes figures to `paper/figures/`:

```bash
python experiments/01_embedding_benchmark.py
python experiments/05_lm_training_toy_vocab.py
```

Fast CI mode (reduced steps):

```bash
python scripts/run_cypha_lm_report.py
```

Or set `CYPHA_LM_FAST=1` before running a single experiment.

## Benchmarks

```bash
python benchmarks/perplexity_eval.py
python benchmarks/memory_profile.py
python benchmarks/latency_profile.py
pytest benchmarks/ --benchmark-autosave
```

## Package layout

| Module | Role |
|--------|------|
| `embeddings/` | Izaac GF(2^n) token embeddings |
| `temporal/` | CellAI multi-scale SSM |
| `expert_field/` | CyphaDIF routing + NIG experts |
| `projection/` | GRIA alpha-projection |
| `model/` | Full CyphaLM stack and generation |
| `analysis/` | Alpha spectrum and compression profiling |

See [CyphaLM_Plan.md](../CyphaLM_Plan.md) for the full research plan.
