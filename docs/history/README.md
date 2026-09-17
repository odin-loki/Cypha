# Cypha before this repository

This repository's git history begins on **2026-07-18**. `CHANGELOG.md` reaches back to
**2026-03-31**, but the commits it names are not objects here. Everything before that —
about thirteen months of work, eight numbered versions, two side branches and a complete
architectural restart — survived only as a zip of loose directories.

This section is that material: **archived, verified, and written up**.

> **Not the product spine.** Nothing here describes how Cypha works today. For that, start
> at [`docs/README.md`](../README.md). This section explains how it got here, and is kept
> because the native source still refers to Python files that existed nowhere in the
> repository until now.

---

## Start here

| Document | What it answers |
|---|---|
| [`TIMELINE.md`](TIMELINE.md) | What happened when, in what order, and how confident we are of each edge |
| [`PYTHON_TO_CPP_BRIDGE.md`](PYTHON_TO_CPP_BRIDGE.md) | Which archived `Cypha.py` the native C++ was ported from, and how we know |
| [`SWEEP_ANALYSIS.md`](SWEEP_ANALYSIS.md) | The 27,524-run experiment behind a default that still ships |
| [`LINEAGE.md`](LINEAGE.md) | Six threads traced end to end, from the 2025 dialogues to code shipping today |
| [`archive/README.md`](archive/README.md) | The preserved source trees, and how they were integrated |
| [`archive/MANIFEST.md`](archive/MANIFEST.md) | Byte-exact inventory with `sha256` for every file |

---

## The three things worth knowing

**1. Cypha has two unrelated architectures, and the version numbers hide the seam.**
Versions 1 through 7 are *Cypha HRNA* — Harmonic Recursive Neural Architecture, a
resonance-and-oscillator system. Version 8 is *CyphaDIF*, a Bayesian classifier built on
Normal-Inverse-Gamma priors, "derived from first principles". They share **zero class
names**, and the classifier that ships descends from the second. But the break is in the
class structure, not in everything: the causal transition matrix `W_T` crosses it intact,
and `GRIA` leaves the archive entirely and comes back in the C++. See
[`TIMELINE.md`](TIMELINE.md#two-architectural-families) and
[`LINEAGE.md`](LINEAGE.md#7-gria--the-one-family-a-idea-that-came-back).

**2. The native C++ was ported from a file this repository had lost.**
35 files under `native/` cite 79 distinct Python identifiers in their doc comments. 41 of
them resolve against the archive's root `Cypha.py`, and 21 resolve there and **nowhere
else**. `CHANGELOG.md:55` records that same file being deleted at the P7 Python
decommission. See [`PYTHON_TO_CPP_BRIDGE.md`](PYTHON_TO_CPP_BRIDGE.md).

**3. The archive is a record of negative results, and they shaped the defaults.**
A 1,070-core-hour factorial grid found two of its five knobs inert. Three separate v6
profilers found adaptive strategies losing to fixed constants, an adaptation path that
never fired once, and a detector that got *worse* with more evidence. The native product's
deliberation band ships disabled by default, and that is the reason.

---

## Era by era

| Era | Directory | Document |
|---|---|---|
| Prototypes — Brain Model, Cell AI v2/v3 | `prototypes/` | [`eras/00-prototypes.md`](eras/00-prototypes.md) |
| V1 — IRENA → HAEDF → HRNA | `cypha-v1/` | [`eras/01-v1.md`](eras/01-v1.md) |
| v2 — the specification that never ran | `cypha-v2/` | [`eras/02-v2.md`](eras/02-v2.md) |
| v3 — the great simplification | `cypha-v3/` | [`eras/03-v3.md`](eras/03-v3.md) |
| v4 — instrumentation and the thinking demonstration | `cypha-v4/` | [`eras/04-v4.md`](eras/04-v4.md) |
| v5 — Omega-2, real datasets, and a self-audit | `cypha-v5/` | [`eras/05-v5.md`](eras/05-v5.md) |
| v6 — the measurement era, and a missing engine | `cypha-v6/` | [`eras/06-v6.md`](eras/06-v6.md) |
| v7 — Generation, and the moment the architecture mattered | `cypa-v7-generation/` | [`eras/07-v7-generation.md`](eras/07-v7-generation.md) |
| v8 — CyphaDIF, the restart that shipped | `cypha-v8/` | [`eras/08-v8.md`](eras/08-v8.md) |
| The two side branches | `cypha-vchatgpt/`, `cypha-vpattern-matching/` | [`eras/branches.md`](eras/branches.md) |
| Loose components and the RetDec side quest | several | [`eras/components.md`](eras/components.md) |

*Every era in the archive now has a writeup. The sources are inventoried in
[`archive/MANIFEST.md`](archive/MANIFEST.md).*

---

## How to use this section

**Reading the archive as evidence.** Per-file timestamps in the source zip were destroyed —
every file carries the moment of zipping. Only directory mtimes survived, and eight
directories share a single bulk-copy timestamp. Every ordering claim in these documents is
therefore built from internal evidence (module docstrings, class inventories, import sets,
explicit supersession notices) and graded for confidence. Do not cite a file mtime.

**Claims are separated from facts.** The historical documents make a number of assertions —
compression ratios, speedup multipliers, capability claims — that the code beside them does
not support. These writeups quote the claim, then show what the code does, and label the
gap. The point is not to criticise the work; it is to stop the numbers being inherited a
third time.

**The archive is read-only.** Do not fix, reformat or lint anything under
[`archive/`](archive/). Its typos, dead code, line endings and one corrupt file are the
record. Corrections belong in these prose documents.

---

## Where the record still goes dark

The archive ends 2026-03-14. `CHANGELOG.md` begins 2026-03-31. **Seventeen days separate
them and nothing covers the gap** — nor the five months from there to the P7 decommission
on 2026-08-16, during which the Python grew a CSV ingest path, a preprocessor, an experiment
database, a model registry, kernel memory and BPTT. Those subsystems are named by 38 of the
79 Python identifiers the native code cites, and they resolve in no archived version. They
exist today only as C++.
