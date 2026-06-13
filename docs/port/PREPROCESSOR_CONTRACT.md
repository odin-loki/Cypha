# Preprocessor JSON contract (`preprocessor.json`)

Saved next to `model.cypha` in the registry. Native loaders must reproduce **`PreprocessorState::transform_one`** behaviour for the same file.

**Not the same as training `sidecar.json`:** parity fixtures under `fixtures/*/sidecar.json` hold numeric goldens for native training/inference harnesses (see [`PORT_CONTRACT.md`](PORT_CONTRACT.md)). This document covers only the registry artifact **`preprocessor.json`**.

**Source of truth:** [`schemas/preprocessor.schema.json`](schemas/preprocessor.schema.json) and **PREPROCESSOR_CONTRACT** field table below.

## Fields (JSON object)

| Key | Type | Meaning |
|-----|------|---------|
| `scale` | bool | If true, apply zero-mean / unit-std using `mean` / `std`. |
| `pca_dim` | int or null | If set, PCA projection dimension after scaling. |
| `rff_dim` | int or null | If set, RFF feature dimension (studio preprocessor path). |
| `rff_gamma` | number | RFF bandwidth scalar. |
| `auto_rff_gamma` | bool (optional) | When true, set `rff_gamma` from median pairwise distance (Python `RFFEncoder.auto_gamma`). |
| `auto_rff_gamma_cv` | bool (optional) | When true, set `rff_gamma` via CV grid search (Python `RFFEncoder.auto_gamma_cv`; overrides `auto_rff_gamma`). |
| `seed` | int | RNG seed used when fitting RFF weights. |
| `mean` | list of float or null | Per-feature mean (length = input dim). |
| `std` | list of float or null | Per-feature std (same length; avoid div-by-zero in reference). |
| `pca_components` | nested list or null | PCA matrix as saved by NumPy `.tolist()`. |
| `pca_mean` | list of float or null | PCA centering vector. |
| `rff_W` | nested list or null | RFF weight matrix. |
| `rff_b` | list of float or null | RFF bias vector. |
| `fitted` | bool | Must be true for inference. |
| `input_dim` | int | Raw feature dimension before transform. |
| `output_dim` | int | Dimension after full pipeline (what the model sees). |

## Pipeline order (must match)

1. **Scale:** `X = (X - mean) / std` when `scale` and `mean` / `std` are set.  
2. **PCA:** `X = (X - pca_mean) @ pca_components.T` when `pca_components` is set.  
3. **RFF:** `X = sqrt(2 / rff_dim) * cos(X @ rff_W.T + rff_b)` when `rff_W` is set (`rff_b` broadcast per row).

## Native notes

- Arrays are **JSON lists**; treat as **float64** row-major when reshaping to matrices.
- **`PreprocessorState::fit_from_design_matrix`** (C++): matches frozen contract for **PCA** with optional **StandardScaler** and optional **RFF**. **`auto_rff_gamma`** (median pairwise) and **`auto_rff_gamma_cv`** (ridge CV on optional targets, else 5-fold reconstruction MSE) select bandwidth before RFF weight init. Parity: **`fixtures/preprocessor_fit_rff/`**, CTest **`native_preprocessor_fit`** + **`native_preprocessor_rff_gamma_cv`**.
- If you add fields, bump a **`preprocessor_schema_version`** in JSON (future) and document here; do not silently rename keys.

## JSON Schema (draft)

Machine-readable shape: [`schemas/preprocessor.schema.json`](schemas/preprocessor.schema.json).
