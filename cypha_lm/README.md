# CyphaLM

Explicit-mechanism language model integrated with **CyphaDIF**: Izaac GF(2^n) embeddings → CellAI SSM → **CyphaDIF expert field** → GRIA alpha-projection. Core path uses NumPy; optional **CUDA** via CuPy (`device="cuda"` or `CYPHA_LM_DEVICE`). PyTorch is optional for baseline benchmarks.

## Architecture

```
token_id → IzaacEmbedding → CellAISSM → CyphaDIF.predict(field_x) → GRIAProjection → log_probs
                              ↑                    ↑
                         temporal context    routing_probs, epistemic_var
```

CyphaDIF is not a separate bolt-on — it routes and predicts in the expert field on every forward step. Training updates GRIA (CE gradients) and optionally CyphaDIF experts online (`config.online=True`).

## Training and context modes (upgrade)

`CyphaLMConfig` supports architectural ablations and stronger char-LM training:

| Field | Default | Description |
|-------|---------|-------------|
| `context_mode` | `"full"` | How GRIA input `v` is built (see table below) |
| `ngram_context` | `2` | Previous token embeds to concat when `context_mode="gria_ngram"` |
| `train_epochs` | `1` | Full passes over `train_sequence` / bench train loop |
| `gria_lr_decay` | `0.5` | Multiply `gria_lr` after each epoch when `train_epochs > 1` |
| `bptt_steps` | `0` | If &gt; 0, truncated BPTT window for optional SSM fast-weight updates |
| `view_id_dim` | `0` | If &gt; 0, concat per-view embedding into GRIA input |
| `ngram_fuse_split` | `True` | Separate field/embed n-gram projections (sum) vs single concat matmul |

### `context_mode` values

| Mode | GRIA input |
|------|------------|
| `full` | SSM context + CyphaDIF field (default) |
| `gria_ngram` | SSM projection + last *K* Izaac embeddings (n-gram-like shortcut) |
| `hybrid_gria_lstm` | GRIA path + char-LSTM head; online blend (best @ D17 300k) |
| `ssm_only` | SSM path; DIF contribution zeroed |
| `ablation_no_dif` | Field mean only; routing ablated |
| `ablation_no_ssm` | N-gram embed stack only; SSM zeroed |

Bench profiles (`d04`, `d17`) use **`hybrid_gria_lstm`** (D17 @ 300k: **2.87 BPC**) or **`gria_ngram`** on D04 + **`schedule_b`** + frozen α + `gria_lr_decay=0.3` + `ngram_context=3` + `view_id_dim=8`.

### Laplace bias init

When `laplace_smoothing > 0`, token counts from training initialize GRIA bias to log-smoothed unigram probabilities, giving a strong char-LM prior before online CE updates.

```python
from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM

cfg = CyphaLMConfig(
    vocab_size=256,
    context_mode="gria_ngram",
    ngram_context=2,
    train_epochs=2,
    bptt_steps=64,
    laplace_smoothing=1.0,
    gria_lr=0.08,
)
model = CyphaLM(cfg)
model.train_sequence(train_ids)  # respects train_epochs + lr decay
```

## Install

From the repository root:

```bash
pip install -e cypha_lm/
pip install -e "cypha_lm[benchmark]"   # torch + pytest-benchmark
pip install -e "cypha_lm[gpu]"         # CuPy (pick cupy-cuda12x / 11x for your driver)
pip install -e ".[studio]"             # optional: FastAPI streaming /generate routes
```

### GPU (CUDA)

CyphaLM supports optional CuPy for matmul/SSM on GPU (`device="cuda"`). **For sequential char-LM, CPU is faster** — `device="auto"` (default) selects CPU. GPU per-token kernel launch + CPU CyphaDIF sync dominates; use CPU unless you add batched training/eval.

```python
model = CyphaLM(CyphaLMConfig(vocab_size=128, device="auto"))  # CPU
model = CyphaLM(CyphaLMConfig(vocab_size=128, device="cuda"))  # explicit GPU experiment
print(model.device)
```

Benchmark: `python scripts/bench_cyphalm.py --steps 3000 --breakdown`

Environment override: `CYPHA_LM_DEVICE=cpu|cuda|auto`. Performance flags: `use_spectral_pde=False`, `use_sparse_hebbian=False` (defaults).

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
| **D04** | Char-LM: held-out BPC, context-length curve, expert routing, save/restore, sampling, **4/5-gram + char-LSTM baselines**, ablations |
| **D17** | WikiText integration: alpha spectrum, cross-corpus online adaptation, full-corpus mode |

Shared helpers: `cypha_bench/adapters/cyphalm_bench.py`, `cyphalm_ablations.py`, `char_lstm_baseline.py`

```bash
pip install -e cypha_lm/
python cypha_bench/domains/d04_generation_language.py
python cypha_bench/run_all.py --domain 4

# Full WikiText train
CYPHA_BENCH_FULL_CORPUS=1 python cypha_bench/run_all.py --domain 17
```

See [`cypha_bench/README.md`](../cypha_bench/README.md) for env vars and baseline tables.

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

## Empirical results

| Metric | Value | Notes |
|--------|-------|-------|
| D17 held-out BPC @ 300k (`hybrid_gria_lstm`) | **2.873** | Beats bigram **3.478** and char-LSTM bench **2.979** |
| D17 held-out BPC @ 300k (`gria_ngram` stack) | **3.838** | +0.36 vs bigram; hybrid resolves gap |
| D04 Gutenberg @ 300k (`hybrid_gria_lstm`, Moby Dick) | **2.993** | Beats bigram **3.633** and char-LSTM **3.047** (sweep peak **2.859**) |
| D04 Gutenberg @ 40k (prior `gria_ngram`) | **4.122** | Superseded by hybrid @ 300k |
| 4-gram / 5-gram / char-LSTM | see D17 JSON | Baselines in bench tables |
| Ablation `gria_ngram` vs `full` @ 40k | **4.154** vs **4.725** | N-gram embed path carries most gain |
| Save/restore parity | ✅ | log-prob max diff &lt; 1e-9 |

See [docs/RESEARCH_STATUS.md](../docs/RESEARCH_STATUS.md) for beat-bigram roadmap and full benchmark tables.
