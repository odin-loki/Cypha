# Run scripts, as used

**Odin Loch** · 2026-10-28

These are the exact scripts that produced the study's runs, kept unchanged as a record. They name the paths of the original working directory, so to rerun one, change the paths at its top.

| File | What it ran |
|---|---|
| `design/` | The v2 design search on development ladders A and B: `lib.py` (z-scores, evidence and ladder accuracy from exported features), `grid*.py` (the combination grids), `v2x.py`, `cand.py`, `comb.py`, `dist.py`, `lex.py`, `tier.py`, and `abl.py` (the ablation table). |
| `profile.sh` | Ablations, seeds, reference sizes, pack sizes, and the timing and memory runs (paper Sections 6.6–6.8). |
| `human_sets.sh` | SCRIBE on the MAUVE human evaluation, HANNA and OpenMEVA, open and prompted profiles. |
| `two_sided.sh` | The exploratory two-sided variant on the same sets. |
| `baselines/make_subsets.py` | The fixed subsets the neural baselines ran on (first N texts per rung, system or source). |
| `baselines/queue.py`, `jobs.txt`, `watchdog.sh` | The baseline job queue, its job list, and the hard time limit. |
