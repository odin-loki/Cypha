# Predictive arithmetic coding (living)

**Status:** shipped in native (2026-07-19)
**Pattern:** LLMZip / LMCompress - next-token model probabilities drive an arithmetic (range) coder.

## What landed

| Piece | Location |
|-------|----------|
| 32-bit range coder | `native/include/cypha/cyphalm/arithmetic_coder.hpp` |
| Predictive stream | `native/include/cypha/cyphalm/predictive_codec.hpp` |
| Product API | `Cypha::compress_tokens` / `decompress_tokens` / `generate_via_bits` |
| REST | `POST /sequence/compress`, `POST /sequence/decompress` (bytes hex) |
| Default model | Hybrid GRIA+LSTM via `apply_hybrid_production_recipe` |
| Tests | `native_predictive_codec_smoke`, `native_predictive_codec_bench_smoke` |

Codec path also runs `AdaptivePredictorMixer` (gated n-grams + match + kNN) and optional **online neural adapt** during compress/decompress. See [ADAPTIVE_PREDICTOR_MIXER.md](ADAPTIVE_PREDICTOR_MIXER.md).

## Semantics

1. Token 0 seeds context (not coded).
2. For each next token, `predict_next(ctx)` -> softmax -> integer CDF -> encode/decode.
3. `model_bpc` is mean -log2 P (matches `eval_bpc` when mixer/online_adapt are off).
4. `coded_bpc` is `8 * |bytes| / n_coded` (flush + CDF quantization overhead).
5. `generate_via_bits` samples by decoding uniform random bits through the same coder.

**Encode/decode contract:** if `online_adapt=true`, both sides must start from the same checkpoint (compress mutates weights).

## Reproduce ~2.8 BPC (model rate)

```powershell
$env:CYPHA_BENCH_FULL_CORPUS="1"; $env:CYPHA_BENCH_OVERNIGHT="1"
cyphalm_bench_native --profile d17 --mode hybrid --overnight --n-train 300000 --n-eval 2000 --threads 1 --bench-seed 42
```

Short codec-rate check after a quick train:

```powershell
predictive_codec_bench --n-train 8000 --n-eval 2000 --epochs 1
```

Measured (Hybrid train, mixer+online_adapt on): mix **3.257** @ 8k/2k, **2.903** @ 40k/4k, **2.547** @ 80k/8k.

## Next levers

- Longer train / full D17 checkpoint -> coded BPC approaches ~2.8 on WikiText-2.
- Larger CDF scale / ANS variant if flush overhead matters on short streams.
- Equal-information windows / streaming REST for long documents.