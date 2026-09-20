# CyphaLM serve / generation

gate24 CyphaLM is an online hp context mixer. **Train** and **serve** share the same
`hp::Predictor` state advance (predict + update per bit), but serve avoids train-only
bookkeeping and can use faster scoring paths.

## API split

| Path | Entry points | Scoring | Side effects |
|------|--------------|---------|--------------|
| **Train / BPC** | `train_step`, `observe_stream_bits`, `eval_bpc_compress_equivalent` | Bit-serial observe (8 bits/byte) | Increments `train_step_count`; updates tables |
| **Serve / generate** | `serve_advance`, `serve_predict_next`, `serve_greedy_next`, `generate_decode` | Bit-tree joint log probs (default) or O(8) greedy | No `train_step_count`; no `adapt_after_predict` stash |

### HpSequenceBackend

- `serve_advance_byte` — advance live context (alias of `consume_byte`).
- `serve_next_byte_log_probs` — full-vocab log P(next byte). **Default:** MSB bit-tree with delta undo (parity with legacy: `hp_bit_tree_smoke`). Legacy 256-clone path: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`.
- `serve_greedy_next_byte` — O(8) argmax on a scratch fork (no 256-way fan-out).
- `serve_sample_next_byte` — O(8) temperature bit sampling on scratch.

### CyphaLMModel

- `serve_advance(token)` — priming long prompts without vocab fan-out.
- `serve_predict_next(context)` — consume context + bit-tree log probs + top-k fill.
- `serve_greedy_next(context)` — consume context + O(8) greedy byte.
- `predict_next` — legacy wrapper that also records state for `adapt_after_predict` / `train_step_count`.

## Generation

High-level decode lives in `cyphalm_generation.hpp`:

- `generate_decode` — greedy, temperature, top-k, top-p, uncertainty-gated.
- Prompt priming uses `serve_advance`; greedy decode uses `serve_greedy_next`; sampling uses `serve_predict_next` + bit-tree log probs.

CLI example:

```bash
cmake --build native/build --target cyphalm_generate -j$(nproc)
./native/build/cyphalm_generate --prompt "Hello " --max-tokens 32 --strategy temperature
./native/build/cyphalm_generate --strategy greedy --max-tokens 16
```

REST: `POST /cyphalm/generate` (see `cyphalm_rest_routes.cpp`) uses the same `generate_decode` path.

## Quality note

**BPC / compression metrics** use `observe_stream_bits` (compress-faithful bit-serial NLL).
**Generation log probs** use the bit-tree joint path; greedy generation can skip the full tree.
These are the same hp math; greedy/top-k sampling is an inference convenience, not a BPC claim.

## Tests

- `cyphalm_serve_smoke` — serve generation does not bump `train_step_count`; train_step does.
- `hp_bit_tree_smoke` — bit-tree ≡ legacy fork log probs.
- `hp_undo_smoke` — predictor checkpoint round-trip (when undo headers are present).
