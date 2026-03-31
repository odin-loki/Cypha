# Preprocessor JSON contract (`preprocessor.json`)

Saved next to `model.cypha` in the registry (`cypha_studio.core.registry`). Native loaders must reproduce **`Preprocessor.transform_one`** / **`transform`** behaviour for the same file.

**Source of truth:** `Preprocessor.save_state` / `load_state` in `cypha_studio/core/dataset.py`.

## Fields (JSON object)

| Key | Type | Meaning |
|-----|------|---------|
| `scale` | bool | If true, apply zero-mean / unit-std using `mean` / `std`. |
| `pca_dim` | int or null | If set, PCA projection dimension after scaling. |
| `rff_dim` | int or null | If set, RFF feature dimension (studio preprocessor path). |
| `rff_gamma` | number | RFF bandwidth scalar. |
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
- **`PreprocessorState::fit_from_design_matrix`** (C++): matches Python **`Preprocessor.fit`** for **PCA** with optional **StandardScaler** (`scale=True` / `False`; PCA always uses a **centered** design matrix). NumPy **`np.linalg.svd`** vs symmetric Jacobi on **`Xc^T Xc`**: same subspace; per-component **sign** may differ and is aligned in **`preprocessor_fit_parity`**. **RFF** weights use NumPy **`default_rng(seed)`** — not reproduced natively; keep **`rff_dim`** unset in native fit or fit in Python and load JSON.
- If you add fields, bump a **`preprocessor_schema_version`** in JSON (future) and document here; do not silently rename keys.

## JSON Schema (draft)

Machine-readable shape: [`schemas/preprocessor.schema.json`](schemas/preprocessor.schema.json).
