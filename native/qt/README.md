# Qt 6 desktop (M5)

The product milestone in [`docs/port/PORT_FULL_STACK.md`](../../docs/port/PORT_FULL_STACK.md) calls for a **Qt 6** shell (dataset → train → registry → predict → explain) talking to native **`cypha_rest`** JSON per **PORT_CONTRACT** §3 (or linking **`cypha_core`** directly).

## Optional build (off by default)

From `native/`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_QT=ON
# Optional: loss/metrics panel uses Qt Charts (needs qt6-charts-dev / Qt6 Charts) instead of painted loss + text log:
#   -DCYPHA_QT_CHARTS=ON
cmake --build build
./build/qt/cypha_qt_stub --help   # Qt Core + cypha_core link check
./build/qt/cypha_qt_stub ../../fixtures/reference.cypha   # buffer load parity

# Widgets shell: CSV inspect, registry scan/load/register, load .cypha, optional sidecars,
# native + REST predict/update, spawn cypha_rest
./build/qt/cypha_qt_shell
# Headless CI smoke (same infer path as cypha_rest classify)
QT_QPA_PLATFORM=offscreen ./build/qt/cypha_qt_shell --smoke ../../fixtures/reference.cypha
```

Targets:

| Binary | Role |
|--------|------|
| **`cypha_qt_stub`** | Qt **Core** only — link + optional `load_cypha_from_buffer` on a file |
| **`cypha_qt_shell`** | Qt **Widgets** + **Network** — CSV + REST/native bulk train + **loss/metrics panel** (REST vs native loss + optional EMA; **Qt Charts** tabbed loss + rolling-accuracy when **`-DCYPHA_QT_CHARTS=ON`**, else painted loss + **text metrics history**); PNG/SVG/CSV export; **Y lock**; **native train log** table; **save `.cypha`** after native train (**`merge_state` + infer patch**, see caveats); **train hparams** UI; registry + **`POST /load`**; **`GET /health`**, **`/ready`**, **`/models`**; **`/predict`** with optional **`return_explanation`**; spawn **`cypha_rest`**; **`--smoke`** |

If Qt6 is missing, CMake prints a warning and skips Qt targets (other native targets still build).

## Dataset panel

The **Dataset** group box is the primary entry point for loading and configuring a CSV dataset.

**Column picker** — after choosing a CSV file the header row is parsed immediately. A **target column** combo lets you pick the prediction target by column name (defaults to the last column, matching `CSVDataset` convention); a **feature columns** checklist shows every other column — uncheck any you want to exclude. Both widgets write through to the underlying *target name* / *feature names* text fields, which remain editable for power-user overrides. **Data preview** shows the first 8 raw CSV rows in a table so you can spot encoding or delimiter issues before running a full load.

**Val split %** (0–40) — hold out the last N% of rows as a read-only validation set. After **Bulk native train** finishes, `best_label_and_conf` is called on every val row; **val accuracy** (`correct/total (%)`) is shown in the stats label and appended to the result line.

**Inspect CSV** — runs `load_csv_dense` with the current column selection, shows row/feature/class counts in the stats label, populates the preview table with the loaded data, and enables **Fit preprocessor…** and **Fill features from row 0**.

## Training CSV (native ingest)

Same contract as **`cypha::load_csv_dense`** / Python **`CSVDataset`**: optional header **target** column by name or index (default index **-1** = last column), optional comma-separated **feature** header names (empty = all columns except target). The **column picker** in the Dataset panel provides a visual alternative to typing these. **Inspect CSV** prints row/feature counts and a truncated first row; **Fill features from row 0** copies the first data row into the feature line edit when its width matches the loaded model (or preprocessor **`input_dim`**). **Bulk REST `/update`:** classification uses each row's string target; check **regression target** for numeric **`y`** and send **`regression_y`** (server must load **`regression_head.json`** with **`mke`** — see **`PORT_CONTRACT.md`**). **MKE `correct_label`** / **`router_train_label`** fields apply to that mode (defaults: first model class). **max rows** `0` = all; per-step **`loss`** plots **REST** (blue) and **native** (orange) on the same chart when you run both bulks — optional **EMA overlay** (α=0.08); **Y lock** (checkbox + min/max spinboxes — pins Y axis to a manual range instead of auto-ranging); **Save chart PNG...**, **Save chart SVG...**, **Clear chart**; **Save loss CSV...** (`step`, `loss_rest`, `loss_rest_ema`, `loss_native`, `loss_native_ema`). Optional **`replay_u01`** JSON array (same file for every request from a fresh offset on the server each call — see native **`cypha_rest`**). **Bulk native train** runs **`dif_train_step_vector`** / **`dif_gh_train_step_vector`** in-process (classification CSV only; mutates the loaded **`CyphaInferModel`** in RAM). **Train 1 row (native)** uses the feature line + **`correct_label`**. Native training needs **`world.F_field`** in the blob or an **`F_field`** JSON path (same as inference). **Native train log** — table showing step #, label, loss, correct (✓/✗) for each native training step (single or bulk); capped at 2000 rows; **Clear log**, **Export CSV...** (step, label, loss, correct).

**Save trained model (.cypha)** — Writes **`CyphaDifMemoryState::merge_state_into_root_for_save`** (**`world`** / **`classes`**) then patches from the live **`CyphaInferModel`** and shell session: **`enc_W`**, **`field_h`**, **`field_step`**, **`field_W_T`** / **`field_a_eff`** / **`w_inject`** (when present), **`temperature`**, **`base_temp`**, **`mahal_*`**, **`llr_scale_*`**, **`llr_ema`**, **`ll_world_ema`** (hardcoded -1.5; same as Python **`save_state`**), **`mid_n`** / **`mid_freq`**, **`mid_trans`**, Tier-1 context keys, **`total_steps`**, **`total_correct`**, **`feat_dim`** (preprocessor **`input_dim`** when loaded, else **`d_latent`**), **`ood_sigma`**, GH session keys. **`field_a_eff`** matches Python **`save_state`** (fp32 **`_A_eff`** promoted to float64 on disk). See **`PORT_CONTRACT.md §1 Qt shell native save parity`** for a full key-level audit. Remaining gaps: key ordering and in-place-only keys (not inserted when absent from the loaded root) — re-export from Python when byte-identical files are needed.

**Native train hyperparameters** — Defaults match **`cypha_rest`**. If **`train_hparams.json`** sits next to the loaded **`.cypha`** (same rule as **`cypha_rest --train-hparams`** auto path), the shell fills the form on load; **Apply** still applies to in-process training. Manual form without that file: **`world_lr`**, **`delta_lr`**, **`ood_sigma`**, **`enc_lr`**, **`replay_ratio`**, **`replay_cap`**, **`align_every`**, **`temp_recalib_every`**. Changing **`replay_cap`** rebuilds an empty replay buffer. **Reset defaults** fills the form only (click **Apply** to use).

## Registry (Python `ModelRegistry` layout)

Pick a **registry root** (`<root>/<name>/<version>/` with **`model.cypha`** + **`card.json`**). **Scan** refreshes the combo; **Load selected bundle** opens that **`model.cypha`** and optional **`preprocessor.json`**. **Register current...** copies the loaded **`.cypha`** + **`card.json`** (+ optional **`preprocessor.json`**) into a new name/version (overwrite optional). **`card.json`** can be set explicitly or defaulted to **`card.json`** beside the current **`.cypha`**.

## F_field JSON

Same row-major 2D layout as **`cypha_rest --f-field-json`**: outer JSON array length **`d_latent`**, each row length **`field_dim`** (from **`field_h`** in the **`.cypha`**). Use **"F field JSON..."** in the GUI when load would otherwise error on missing embedded **`world.F_field`**.

## Preprocessor JSON

Optional **`preprocessor.json`** (same schema as Python / REST). When loaded, the feature text box expects **`input_dim`** comma-separated values; native predict runs **`transform_one`** before encode → LLR → softmax + GH. **Clear preprocessor** returns to latent-size **`d_latent`** features.

## Fit preprocessor (native)

**Fit preprocessor…** button — available after **Inspect CSV** has been run successfully. Opens a dialog to fit a new preprocessor from the loaded feature matrix entirely in-process (no Python required):

- **Scale** (on by default) — z-score normalises each feature column using the training-set mean and stddev.
- **PCA dim** (0 = no PCA) — reduces to the chosen number of principal components using `fit_from_design_matrix`. Max is `n_features`; set to 0 to keep all dimensions after scaling.
- **Output dim** is computed and shown live as you change PCA dim.
- **Fit & use** — fits the preprocessor, installs it in RAM (replaces any loaded `preprocessor.json`), and closes the dialog.
- **Fit & save…** — fits and saves a `preprocessor.json` with the full schema (scale stats, PCA components / mean; RFF fields left empty). The saved file can be loaded directly by `cypha_rest --pre` or from Python.

**RFF note:** RFF weights require Python-side generation (`Preprocessor.fit` with `rff_dim > 0`). The native fitter only covers scale + PCA. After saving, run `native preprocessor_fit_parity + manual RFF sidecar edit` → set `rff_dim` → re-save if you need RFF.

## REST and local server

- **Manual URL:** set the **`cypha_rest`** base URL (no trailing slash required), e.g. `http://127.0.0.1:8099`. **`use_gh`** toggles REST predict/update/bulk and **native** GH train steps. **"Predict (REST)"** / **"Update (REST)"** (optional **`replay_u01`** in body; checkbox **`return_explanation`** → **`POST /predict`** body + full JSON in the log). **"POST /load"** when **`--registry`** is set. **"GET /health"**, **"GET /ready"**, **"GET /models"** append JSON to the log.
- **Spawn server:** startup passes **`--cypha`**, optional **`--f-field-json`**, **`--pre`**, and **`--registry <root>`** when a registry root is configured. On success, the REST base URL is filled from **`http://` + --listen**. **Stdout/stderr** in the **cypha_rest log**; **Clear log** resets it.

The server must have a model loaded for REST predict/update (via **`/load`** or **`--cypha`** at startup). Native predict does not require the server.

## Bulk native training — streaming thread

**Bulk native train** now runs the training loop on a **background `QThread`**. The main thread stays fully responsive during long CSV runs:

- A `QTimer` fires every 80 ms to drain completed steps from a mutex-protected queue.
- The **loss/metrics panel** updates live: loss chart every 200 steps; per-step **rolling accuracy** (window=200) appended on each drained step (Qt Charts **Rolling accuracy** tab when **`-DCYPHA_QT_CHARTS=ON`**, else scrollable **text metrics history** below the painted loss curve).
- The **result label** shows `Training N / total…` progress.
- The **Cancel** button on the progress UI sets a `std::atomic<bool>` flag; the worker exits at the next step boundary — no data corruption.
- Final chart, training log table, val accuracy, and scalar state (EMA loss, rolling accuracy window, GH chi/psi) are all synced back to the main thread in the `on_bulk_finish()` handler.
- Load / Save / Train-one buttons are disabled for the duration of the run and re-enabled on finish.

## Windows packaging (standalone distributable)

Requires **Qt 6 installed natively on Windows** (from [qt.io](https://www.qt.io/download)) and `windeployqt.exe` on PATH.

```powershell
# 1. Build with Qt 6 (MSVC or MinGW toolchain, Qt installed on Windows)
cmake -S native -B native\build `
      -DCMAKE_BUILD_TYPE=Release `
      -DCYPHA_BUILD_QT=ON
cmake --build native\build --target cypha_qt_shell

# 2. Package (copies Qt DLLs alongside the exe)
powershell -ExecutionPolicy Bypass `
  -File native\scripts\package_windows_qt.ps1 `
  -WithFixtures   # optional: copies parity fixtures for a demo

# Output folder: native\dist\cypha_qt_shell_windows\
# Run:
native\dist\cypha_qt_shell_windows\cypha_qt_shell.exe
```

The script (`native/scripts/package_windows_qt.ps1`) will:
1. Copy `cypha_qt_shell.exe` (and `cypha_rest.exe` if present) into the dist folder.
2. Run `windeployqt --no-translations --no-system-d3d-compiler --no-opengl-sw` to pull in all required Qt DLLs.
3. Optionally copy `fixtures/reference.cypha` + `f_field.json` for a headless smoke test.

**Cross-compilation note:** the MinGW cross-build from WSL (`cmake --preset mingw-w64-cross`) does **not** produce a Qt shell — Qt on Windows requires the native Windows Qt DLLs which aren't available in the WSL cross-toolchain. Build natively on Windows for the packaged GUI.

## Roadmap (parity with PySide Studio)

1. **Rich charts** — With **`-DCYPHA_QT_CHARTS=ON`** (Qt6 Charts installed), the **Loss / Metrics** panel uses a tabbed **`QChartView`**: **Loss** tab (`QLineSeries` REST vs native + optional EMA) and **Rolling accuracy** tab (native, window=200). Without Charts (CI default), a painted dual polyline loss curve plus a **text metrics history** (`step | loss | roll_acc | source`) replaces the Charts tabs. **PNG**, hand-written **SVG**, and **CSV** export; **Y lock** for manual Y axis range. **Loss chart interactivity** (both painted and Qt Charts paths): mouse-over tooltip `(step, loss)`, scroll-wheel zoom, click-drag pan (clamped to data extents). Done.
2. **Full native save parity** — all Python **`save_state`** fields used by native reload are patched (incl. **`field_a_eff`**, **`ll_world_ema`=-1.5**, all GH/session keys); remaining deltas are key ordering and in-place-only keys (non-functional for Python-generated roots).
3. ~~**Train hparams UI**~~ — form + Apply + replay cap rebuild in **`cypha_qt_shell`**.
4. ~~**Save `.cypha` after native train**~~ — merge + encoder/field/temperature/scalars (**`save_cypha_file`**).
5. ~~**Native in-process train (v1)**~~ — single row + bulk + **`replay_u01`**.
6. ~~**Bulk REST + regression/MKE + replay_u01 + `/ready`**~~ — in **`cypha_qt_shell`**.
7. ~~**Dataset CSV + registry + `/load`**~~ — in **`cypha_qt_shell`**.
8. ~~**Native train log table**~~ — per-step step #, label, loss, correct + Export CSV.
9. ~~**Streaming bulk training thread**~~ — `QThread` worker, live chart + responsive UI.
10. Keep REST JSON aligned with [`PORT_CONTRACT.md`](../../docs/port/PORT_CONTRACT.md) §3.
