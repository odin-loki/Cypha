# Optional memory profiling and load testing

CyphaStudio does not run these in CI by default. Use when you need evidence for long sessions, huge CSVs, or API throughput.

**GUI threading:** see [`STUDIO_THREADING.md`](STUDIO_THREADING.md).

## Memory (Python)

- **Studio training widget (tracemalloc wrapper):**
  ```bash
  python scripts/profile_studio_memory.py --steps 800
  ```
- **tracemalloc** — compare snapshots around a workload:
  ```bash
  python -X tracemalloc=25 -c "import tracemalloc; tracemalloc.start(); ... your code ..."
  ```
- **memray** (`pip install memray`) — flamegraph-style allocation reports:
  ```bash
  memray run -o m.bin -m cypha_studio.main
  memray flamegraph m.bin
  ```

## Threading sanity

Training uses `QThread` (`TrainingWorker`); callbacks emit `SignalBus` only. When adding code, avoid touching Qt widgets from the trainer thread — keep UI updates on the main thread via signals.

## HTTP load (live uvicorn)

After `python cypha_studio/main.py --headless` (or your bind host/port):

- **ApacheBench (examples in repo):**  
  - JSON body: [`examples/cypha_predict_body.json`](../examples/cypha_predict_body.json)  
  - **Linux/macOS:** `bash scripts/loadtest_ab_predict_example.sh`  
  - **Windows:** `powershell -File scripts/loadtest_ab_predict_example.ps1` (requires `ab` on `PATH`)
- **Manual:** `ab -n 2000 -c 10 -T application/json -p examples/cypha_predict_body.json http://127.0.0.1:7749/predict`
- **Locust:** write a small `locustfile.py` that POSTs the same JSON to `/predict`; use only on a **trusted** network (no auth in the reference server).

See also [`CYPHA_STUDIO_MASTER_PLAN.md`](CYPHA_STUDIO_MASTER_PLAN.md) Phase 1 optional items.
