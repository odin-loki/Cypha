# Optional memory profiling and load testing

Cypha native stack does not run long-session profiling in CI by default. Use when you need evidence for huge CSVs or API throughput.

**GUI threading:** see [`STUDIO_THREADING.md`](STUDIO_THREADING.md).

## Memory / diagnostics

- **Native diagnostics orchestrator:**
  ```bash
  cypha_diagnostics_run --fixtures fixtures --phases 1,2,3,4
  ```
- **Valgrind / heap tools** (Linux): wrap **`cypha_qt_shell`** or **`cypha_rest`** for allocation reports.

## Threading sanity

Training uses `QThread`; keep UI updates on the main thread via Qt signals.

## HTTP load (live `cypha_rest`)

After starting **`cypha_rest`** (or your bind host/port):

- **ApacheBench (examples in repo):**  
  - JSON body: [`examples/cypha_predict_body.json`](../examples/cypha_predict_body.json)  
  - **Linux:** `bash scripts/loadtest_ab_predict_example.sh`  
  - **Windows:** `powershell -File scripts/loadtest_ab_predict_example.ps1` (requires `ab` on `PATH`)
- **Manual:** `ab -n 2000 -c 10 -T application/json -p examples/cypha_predict_body.json http://127.0.0.1:7749/predict`

See also [`CYPHA_STUDIO_MASTER_PLAN.md`](CYPHA_STUDIO_MASTER_PLAN.md).
