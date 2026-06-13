# CyphaLM Native — Tier 4 Hebbian / Biochemical Hooks

Native C++ mirrors for Cypha Tests **2B** (encoder), **2C** (sparse SSM lateral), and **2D** (associative graph). Python references:

| Hook | Python |
|------|--------|
| Biochemical encoder | `cypha_core` → `EncoderProjection.hebbian_update` |
| Sparse SSM | `cypha_lm/temporal/cellai_ssm.py` → `sparse_hebbian_update` |
| Lateral graph | `native/src/som/` (Hebbian topology smoke) |

## Library

Target: **`cypha_lm_native`** (CMake). Link against `cypha_core` when composing with DIF train steps.

Headers live under `native/include/cypha/cyphalm/`.

## 2B — Pluggable encoder update

```cpp
#include "cypha/cyphalm/hebbian_encoder.hpp"

cypha::cyphalm::HebbianEncoder enc;
enc.d = d;
enc.w = /* row-major d×d */;
enc.class_stats = [&](const std::string& label,
                      std::vector<double>& mu,
                      std::vector<double>& v) -> bool {
  return memory.class_mean_and_variance(label, mu, v);
};

// After a misclassification (true vs predicted):
enc.competitor_label = pred_label;
enc.update(f, h, true_label, enc_lr, 1.0);
```

**Default rule** (`biochemical_hebbian_update`):

```
signal = tanh(r_k[0] - r_j[0])     # Fisher–Rao residuals
ΔW     = lr * weight * signal * outer(h, f)
```

Frobenius cap (8.0) runs every 50 updates, matching `EncoderProjection`.

### Swap your biochemical rule

Assign any callable matching `HebbianEncoderUpdateFn`:

```cpp
enc.update_rule = [](std::vector<double>& w, int d, const double* f, const double* h,
                     const double* mu_k, const double* v_k, const double* mu_j, const double* v_j,
                     double weight, double lr, int& update_count) {
  // Your drop-in biochemical network write rule here.
  // Example: outer(h, f) scaled by a custom metabolite trace.
  cypha::cyphalm::biochemical_hebbian_update(w, d, f, h, mu_k, v_k, mu_j, v_j, weight, lr, update_count);
};
```

Or bind a free function / member via `std::function` or raw function pointer (implicit conversion).

Freeze updates during distillation:

```cpp
enc.frozen = true;
```

## 2C — Sparse Hebbian SSM

```cpp
#include "cypha/cyphalm/hebbian_ssm.hpp"

cypha::cyphalm::HebbianSSMState ssm;
ssm.resize(d_state, n_layers);

// Inside each CellAI layer step, after fast/slow state update:
cypha::cyphalm::sparse_hebbian_update(ssm, h_fast, s_slow, 1e-4, layer);
```

Updates only columns indexed by top-`max(1, d/8)` activations of the fast state (12.5% sparsity).

## 2D — Hebbian associative graph

```cpp
#include "cypha/cyphalm/hebbian_graph.hpp"

cypha::cyphalm::HebbianGraphConfig gc;
gc.n = 2 * d_state;   // matches wire_cellai: d_state * 2
gc.k_neighbors = 4;   // optional: limit diffusion to top-4 edges per node

cypha::cyphalm::HebbianGraph graph(gc);

// Per layer context (after multiscale blend):
std::vector<double> ctx = /* concat fast/slow */;
std::vector<double> ctx_out = graph.diffuse(ctx);
graph.update(ctx_out.data());
```

`diffuse` implements `x' = x + γ (D⁻¹A) x`. `update` applies co-activation Hebbian edge plasticity with prune/form thresholds from Python.

## Composition — `hebbian_stack.hpp`

When `cellai_ssm.cpp` is not yet in the native tree, use `HebbianStack` to wire all three hooks:

```cpp
#include "cypha/cyphalm/hebbian_stack.hpp"

cypha::cyphalm::HebbianStack stack;
cypha::cyphalm::HebbianStackConfig cfg;
cfg.d_state = 64;
cfg.n_layers = 2;
cfg.use_sparse_hebbian = true;
cfg.use_hebb_graph = true;
cfg.graph.n = 128;
stack.configure(cfg);
stack.encoder = enc;

// SSM layer hook:
stack.on_ssm_layer_context(ctx, layer, h_fast, s_slow);

// DIF train hook:
stack.encoder_train_step(f, h, y_label, pred_label, enc_lr);
```

## Parity

```bash
cmake --build build --target cyphalm_hebbian_parity
./build/cyphalm_hebbian_parity
```

CTest name: `native_cyphalm_hebbian`. Compares one encoder update, one sparse SSM write, and one graph diffuse against Python-derived goldens.

## Integration with `train_step_vector`

Encoder hooks are optional alongside existing `contrastive_update_encoder_w`. To enable biochemical mode in native DIF training, set `enc.update_rule` to `biochemical_hebbian_update` and call `encoder.update(...)` from your train loop when `encoder_update_mode == "hebbian"` (mirrors `CyphaDIF._contrastive_with_som`).
