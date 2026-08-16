# data/

Committed **historical experiment dumps**. Not the living production lock.

| Path | What it is |
|------|------------|
| [`archive/profiles/`](archive/profiles/) | D17 recipe traces (L1 / L2 / Wave2 BPTT), XOR kernel-LLR sweeps, and July 2026 bench captures. Formerly `artifacts/profiles/`. |
| [`archive/cell_sweep/`](archive/cell_sweep/) | Preserved H15 @ 300k row. Live sweep writes `bench/results/cell_sweep/` (gitignored). |

**Living numbers** stay in [`bench/BASELINE_LOCK.json`](../bench/BASELINE_LOCK.json) (D17 **2.664 BPC**). New local captures go to `artifacts/` (gitignored) — do not add them here unless they document a lock re-pin.

**Not here:** CTest fixtures (`fixtures/`), bench profiles (`bench/config/`), forecast/WikiText corpora (`bench/data/`).

See [`archive/README.md`](archive/README.md) for the file list.
