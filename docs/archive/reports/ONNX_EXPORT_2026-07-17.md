# ONNX export — minimal infer subgraph (2026-07-17)

**Build:** `native/build_onnx` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Bill of Work §5 — header-only ModelProto writer + encode→LLR→softmax graph. Did not touch `build_math`, `build_deff`, `BASELINE_*`, overnight processes, GGUF, or parallel/sample-efficiency work.

## Shipped subgraph

Field-conditioned μ₀ and class biases are baked at export time. Kernel LLR blend and deliberation are **not** exported.

| Stage | Nodes | Tensors |
|-------|-------|---------|
| Encoder projection | `Gemm` (`encode_gemm`, `transB=1`) | `x` → `h_raw` via `enc_W` |
| Optional activation | `Tanh` (`encode_tanh`) when `--activation tanh` | `h_raw` → `h` |
| Center / scale | `Sub` (`shift_mu0`), `Mul` (`scale_inv_v`) | `mu0`, `inv_v` → `R` |
| LLR / score | `MatMul` (`score_matmul`), `Add` (`score_bias`) | `D_T`, `llr_bias` → **`llr`** |
| Softmax (opt-in) | `Mul` (`scale_temp`), `Softmax` (`softmax`, `axis=-1`) | `inv_temp` → **`probs`** |

Default CLI write (`--format onnx`) emits `llr` only. Smoke validation uses `with_softmax=true` so the full encode→LLR→softmax chain is exercised.

**I/O:** input `x` `[batch, d]`; outputs `llr` and (when enabled) `probs` `[batch, k]`. Opset 13 / IR 8. Producer `cypha_onnx_export`, graph name `cypha_infer`.

## CLI (unchanged contract)

```text
cypha_onnx_export --cypha PATH --out PATH [--format onnx|json]
                  [--activation tanh|none] [--with-softmax] [--dry-run]
```

JSON intermediate (`--format json`) remains ONNX-ready for external `onnx.helper` conversion.

## Structural validation

`onnx_export_smoke` writes a `.onnx` from `fixtures/reference.cypha` and parses ModelProto with the same header-only wire reader (no onnxruntime dependency). Checks: IR/producer/graph name, input `x`, outputs `llr`/`probs`, initializers `enc_W`/`mu0`/`inv_v`/`D_T`/`llr_bias`/`inv_temp`, and node op sequence `Gemm → Sub → Mul → MatMul → Add → Mul → Softmax`.

## Tests

| CTest | Result |
|-------|--------|
| `native_onnx_export_help` | PASS |
| `native_onnx_export_dry_run` | PASS |
| `native_onnx_export_smoke` | PASS (CLI write) |
| `native_onnx_export_validate` | PASS (structure) |

```powershell
cmake -S native -B native/build_onnx -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/build_onnx --target cypha_onnx_export onnx_export_smoke -j
ctest --test-dir native/build_onnx -R "native_onnx_export" --output-on-failure
```

## Files

- `native/include/cypha/onnx_min_writer.hpp` — writer + `parse_model` / `validate_cypha_infer_model`
- `native/tools/cypha_onnx_export.cpp` — existing CLI (additive; default still LLR-only)
- `native/tools/onnx_export_smoke.cpp` — CTest structural smoke
- `native/CMakeLists.txt` — `onnx_export_smoke` target + `native_onnx_export_validate`

## Follow-up (not in this slice)

- Round-trip numeric parity via onnxruntime when available
- Kernel LLR / deliberation export
- Dynamic field/context priors (currently frozen at export)
