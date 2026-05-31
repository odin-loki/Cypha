# CyphaLM

Explicit-mechanism language model integrated with **CyphaDIF**: Izaac GF(2^n) embeddings → CellAI SSM → **CyphaDIF expert field** → GRIA alpha-projection. Core modules use NumPy only; PyTorch is optional for baseline benchmarks.

## Architecture

```
token_id → IzaacEmbedding → CellAISSM → CyphaDIF.predict(field_x) → GRIAProjection → log_probs
                              ↑                    ↑
                         temporal context    routing_probs, epistemic_var
```

CyphaDIF is not a separate bolt-on — it routes and predicts in the expert field on every forward step. Training updates GRIA (CE gradients) and optionally CyphaDIF experts online (`config.online=True`).

## Install

From the repository root:

```bash
pip install -e cypha_lm/
pip install -e "cypha_lm/[benchmark]"   # torch + pytest-benchmark
pip install -e ".[studio]"            # optional: FastAPI streaming /generate routes
```

## Quick start

```python
from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM

model = CyphaLM(CyphaLMConfig(vocab_size=128))
model.train_sequence([1, 2, 3, 4, 5, 4, 3, 2, 1] * 20)

pred = model.predict_next(3)
print(pred["dominant_expert"], pred["epistemic_var"])

out = model.generate([1, 2, 3], max_tokens=50, strategy="top_p", top_p=0.92)
print(out["generated_ids"])

for chunk in model.stream_generate([1, 2, 3], max_tokens=20, strategy="temperature"):
    if chunk.get("done"):
        break
    print(chunk["token_id"], chunk["dominant_expert"])
```

## Generation strategies

Implemented in `cypha_lm/model/generation.py`:

| Strategy | Function | Notes |
|----------|----------|-------|
| Greedy | `greedy_decode` | Argmax each step |
| Temperature | `temperature_sample` | Full-vocab softmax |
| Top-k | `top_k_sample` | Restrict to k highest log-probs |
| Top-p (nucleus) | `top_p_sample` | Cumulative mass threshold |
| Uncertainty-gated | `uncertainty_gated_sample` | Stop when epistemic_var > threshold |

Unified API: `autoregressive_decode(...)` returns `generated_ids` plus `per_step` trace with **CyphaDIF routing** (`dominant_expert`, `routing_probs`, `active_experts`).

`CyphaLM.generate(..., strategy="top_k", top_k=40)` and `CyphaLM.stream_generate(...)` wrap the same core.

## Save / load

Checkpoints are a pair of files:

- `{base}.json` — config, GRIA, SSM state, CyphaDIF experts, token counts
- `{base}.npz` — projection matrices (`proj_ssm`, `proj_dif`, `proj_embed`)

```python
model.save("/tmp/ckpt")
loaded = CyphaLM.load("/tmp/ckpt")
```

Round-trip fidelity is verified in **cypha_bench D04** (`save_restore.parity_ok`).

## REST API (CyphaStudio)

FastAPI-only routes (not in native `cypha_rest`):

| Route | Purpose |
|-------|---------|
| `POST /lm/load` | Load checkpoint |
| `POST /lm/predict_next` | Single token + routing |
| `POST /generate` | Batch or SSE (`stream=true`) |
| `POST /generate/stream` | SSE token stream |

Set `CYPHA_LM_CHECKPOINT=/path/to/ckpt` at server startup, or load at runtime.

Examples: [`examples/lm_generate_body.json`](../examples/lm_generate_body.json), [`examples/curl_lm_generate_stream.sh`](../examples/curl_lm_generate_stream.sh).

See [`cypha_studio/README.md`](../cypha_studio/README.md) and [`docs/port/PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §4.

## Benchmarks (cypha_bench)

| Domain | Focus |
|--------|-------|
| **D04** | Char-LM: held-out BPC, context-length curve, expert routing trace, save/restore, sampling comparison |
| **D17** | Extended integration: WikiText, alpha spectrum, cross-corpus online adaptation |

Shared helpers: `cypha_bench/adapters/cyphalm_bench.py`

```bash
pip install -e cypha_lm/
python cypha_bench/domains/d04_generation_language.py
python cypha_bench/run_all.py --domain 4
```

## Run tests

```bash
pytest cypha_lm/ -v --tb=short
pytest tests/test_lm_api.py -v    # REST + generation integration
```

## Experiments

Each script is standalone and writes figures to `paper/figures/`:

```bash
python experiments/01_embedding_benchmark.py
python experiments/05_lm_training_toy_vocab.py
python scripts/run_cypha_lm_report.py
```

Set `CYPHA_LM_FAST=1` for reduced steps.

## Package layout

| Module | Role |
|--------|------|
| `embeddings/` | Izaac GF(2^n) token embeddings |
| `temporal/` | CellAI multi-scale SSM |
| `expert_field/` | CyphaDIF routing + NIG experts |
| `projection/` | GRIA alpha-projection |
| `model/` | CyphaLM stack, `generation.py` decoding |
| `analysis/` | Alpha spectrum and compression profiling |

## Empirical results (2026-05-31)

| Metric | Value | Notes |
|--------|-------|-------|
| D04 / D17 held-out BPC | ~4.5–5.2 | Depends on corpus and train steps |
| Bigram baseline | ~3.7–4.2 | Gutenberg / WikiText |
| Save/restore parity | ✅ | log-prob max diff < 1e-9 |

See [docs/RESEARCH_STATUS.md](../docs/RESEARCH_STATUS.md) for full benchmark tables and roadmap.
