# ECG5000 (UCR)

Native D10 loads `ECG5000_TRAIN.txt` / `ECG5000_TEST.txt` (label + 140 samples per line).

```powershell
python scripts/download_ecg5000.py
```

Downloads `.ts` from [Zenodo 11186692](https://zenodo.org/records/11186692) and converts to the `.txt` layout expected by `cypha::bench::load_ecg5000`.

**2026-07-18 measure (FAST D10A):** real `data_source=ecg5000` → **90.11%** default (enriched features, 44 passes). Legacy path: `CYPHA_D10_ECG_ENRICH=0` → **85.96%**. See `docs/reports/D10_ECG5000_GT90_ATTEMPT_2026-07-18.md`.
