# Timeline and version DAG

Cypha's development from February 2025 to the native C++ product, reconstructed from the
[archived sources](archive/). Where a claim rests on evidence, the evidence is quoted.

---

## Dating the archive

Every file in the source zip carries the same modification time — `2026-09-17 12:24`,
rewritten at the moment of zipping. **Per-file timestamps are worthless here.** Directory
mtimes survived the repack:

| Directory | mtime |
|---|---|
| `Prototypes` | 2025-02-26 18:34 |
| `Cypha V1` | 2025-03-04 21:26 |
| `Big Data`, `Cypha Encoder`, `Cypha v2`, `v3`, `v4`, `v5`, `Cypha vChatGPT`, `Cypha vPattern Matching` | **2026-02-21 20:35:08** |
| `Cypha v6` | 2026-02-27 17:44 |
| `Cypa v7 Generation` | 2026-03-06 19:19 |
| `Cypha v8` | 2026-03-11 19:43 |
| archive root | 2026-03-14 14:26 |

Eight directories sharing one timestamp *to the second* is a bulk copy, not eight
simultaneous authorships. A directory mtime is therefore an **upper bound** on the work
inside it, and for those eight it is a very loose one.

### Four channels survive inside the files

Repacking rewrites the zip's own metadata. It cannot touch bytes *inside* a file, and four
formats in this archive record a timestamp there:

| Channel | Where | Covers | Files |
|---|---|---|---|
| CPython `.pyc` header | source mtime + size, [PEP 552](https://peps.python.org/pep-0552/) | vChatGPT | 30 |
| OOXML `docProps/core.xml` | `dcterms:created` / `dcterms:modified` | v2, v5, v6, v8 | 31 |
| PDF trailer | `/CreationDate`, `/ModDate` | vPattern Matching | 1 |
| nested zip central directory | per-entry mtimes | each `.docx`, `retdec_upgrades5.zip` | 32 |

Together they date **six** of the fourteen directories, and every recovered date falls
strictly *before* its directory's mtime — exactly as it must if those mtimes are copy times.
That agreement across four independent channels is the strongest evidence in the archive that
the reconstruction below is sound.

| Recovered | Directory | Evidence |
|---|---|---|
| **2025-05-05 07:24** | `Cypha v2` | `Cypha.docx` `dcterms:created` |
| **2026-02-13 05:29–10:43** | `Cypha vChatGPT` | 30 `.pyc` headers |
| **2026-02-13 11:06:27** | `Cypha vPattern Matching` | `Cypha Demo.pdf` `/CreationDate` |
| **2026-02-20 06:48** | `Cypha v6` | `Cypha Encoder Math Proving.docx` |
| **2026-02-21 08:17–08:49** | `Cypha v5` | five `README_*.docx` |
| **2026-02-22 10:08–10:46** | `Cypha v6` | three more `.docx` |
| **2026-03-07 – 2026-03-10** | `Cypha v8` | 17 `.docx` |

All times UTC. The zip's own stamps are local (`+11:00`, AEDT), which is why the bulk copy
reads `20:35` against document stamps of `08:xx` the same day — those are `19:xx` local, some
46 minutes earlier. The `+11'00'` offset written into the PDF confirms the zone, and matches
the `+1000`/`+1100` offsets on this repository's own commits.

### What the documents date, and what they do not

29 of the 31 `.docx` files were produced by a Markdown-to-Word converter, not typed: `cp:revision`
is 1, `dcterms:created` equals `dcterms:modified` to the millisecond, and `docProps/app.xml` has
no editing statistics. Their timestamps date the **conversion**, which is an upper bound on the
Markdown they were made from — but since 28 of the 31 have zip entries stamped within ±1 second
of their own `created` property, that bound is tight and self-consistent.

Three files break the pattern, and each breaks it in an informative way:

- **`Cypha v2/Cypha.docx`** is the only genuinely hand-saved document in the archive —
  `Microsoft Office Word 16.0`, `Normal.dotm`, creator and last-modified-by both `Odin Loch`,
  4,004 words over 35 pages. Word zeroes zip entry times to the DOS epoch (`1980-01-01`), so
  here the *core.xml* stamp is the real one. It dates to **2025-05-05**, nine months before
  anything else in the archive. See [v2](eras/02-v2.md#the-oldest-dated-artifact-in-the-archive).
- **`Cypha v8/cypha_synthesis.docx`** is stamped 977 seconds — 16 minutes 17 seconds — after
  its own `created` property, and is the only one of the fifteen v8 papers whose zip omits the
  explicit directory entries its siblings carry. It was re-saved, by a different code path,
  after the batch finished. It is also the paper that synthesises the other fourteen.
- **`Cypha v8/Cypha_Research_Paper.docx`** claims `2013-12-23T23:15:00Z`. That is not a date:
  it is generator-default metadata left untouched — creator `python-docx`, and an `app.xml`
  reporting **0 words, 1 page, 0 paragraphs** for a document that extracts to 32,395 bytes of
  Markdown. The statistics are demonstrably not this document's, so neither is the timestamp.
  Its zip entries are stamped **2026-03-08 04:12:28**, which sits between `CyphaREADME.docx`
  (03-07) and the paper batch (03-10) whichever zone that stamp is in.

The ordering below rests on this dating where it exists, and elsewhere on internal evidence —
module docstrings, class inventories, import sets and explicit supersession notices. Each edge
is graded for confidence.

---

## The DAG

```
   Prototypes — Brain Model · Cell AI v2 · Cell AI v3/OICFHS      2025-02-26
                       │
                       ▼
   "Cypha V1" — IRENA → HRNA   (the name Cypha does not exist yet) 2025-03-04
                       │
                       ▼
   Cypha v2 — torch + ray + scipy, 9,404 lines ────────────┐  spec *2025-05-05
                       │                                    │
                       │  ◄── frameworks dropped            ▼
                       ▼                              vChatGPT  *2026-02-13
   Cypha v3 — HRNA, BinaryEncoder, pure numpy ──┐     modular HRNA, 38 files
                       │                        │     + torch/transformers/PyQt6
                       ▼                        ▼     + agi/ gui/ security/ tests/
   Cypha v4 — + benchmark_suite,          vPattern Matching *2026-02-13
              verify_thinking             "production" HRNA
                       │                  distillation, 996 lines
                       │  ◄── BinaryEncoder replaced by OmegaEncoder
                       ▼
   Cypha v5 — HRNA "Omega-2" + dataset tooling                   *2026-02-21
                       │
                       ▼
   Cypha v6 — + 4 profilers + game benchmarks              *2026-02-22 … 02-27
                       │
                       ▼
   Cypa v7 Generation — HRNA peak, 55 classes                     2026-03-06
                       ┊
           ╔═══════════┊════════════╗
           ║   CLEAN-SHEET RESTART  ║   family A ends here
           ╚═══════════┊════════════╝
                       ▼
   Cypha v8 — CyphaDIF, 9 classes, 1,412 lines             *2026-03-07 … 03-11
                       │
                       ▼
   root Cypha.py — CyphaDIF "v2", +5 phases                       2026-03-14
                       ┊
                       ┊  ◄── undocumented interval → 2026-08-16
                       ▼
   prior repository — CHANGELOG v0.1.0 (2026-03-31) … v2.3.24
   commits 127bbe9 / 8945c95 / 1dbfa13, none reachable from here
                       │
                       ▼
   this repository — git history from 2026-07-18 (v2.3.25)
   Python decommissioned at P7, v2.4.0, 2026-08-16
                       │
                       ▼
             native C++ `cypha::Cypha`
```

`*` marks a date recovered from *inside* the files, per the table above; unmarked dates are
directory mtimes, which bound the work from above rather than pinning it. Where both exist the
range runs from the earliest recovered date to the directory stamp.

### The class-overlap matrix that fixes the branch points

Counting shared `class` names between every pair of archived trees:

|          | v2 | v3 | v4 | v5 | v6 | v7 | **v8** | **root** | vChatGPT | vPattern |
|---|---|---|---|---|---|---|---|---|---|---|
| v2       | 48 | 10 | 10 | 8 | 8 | 8 | **0** | **0** | **30** | 2 |
| v3       | 10 | 33 | 30 | 24 | 24 | 24 | **0** | **0** | 8 | 5 |
| v4       | 10 | 30 | 39 | 24 | 24 | 24 | **0** | **0** | 8 | 4 |
| v5       | 8 | 24 | 24 | 26 | 26 | 26 | **0** | **0** | 8 | 4 |
| v6       | 8 | 24 | 24 | 26 | 34 | 29 | **0** | **0** | 8 | 4 |
| v7       | 8 | 24 | 24 | 26 | 29 | 53 | **0** | **0** | 8 | 4 |
| **v8**   | **0** | **0** | **0** | **0** | **0** | **0** | 9 | 6 | **0** | **0** |
| **root** | **0** | **0** | **0** | **0** | **0** | **0** | 6 | 19 | **0** | **0** |

Two things fall straight out of it.

**The family break is total.** v8 and the root monolith share **zero** class names with
*every* family-A tree — v2, v3, v4, v5, v6, v7, vChatGPT and vPattern Matching alike — while
sharing 6 with each other. There is no gradual transition to find.

**vChatGPT forks from v2, not v3.** It shares 30 classes with v2 against 8 with each of
v3–v7, and those 30 include classes that exist *only* in v2 on the main line —
`AlternativeFastOperations`, `NaturalMathematicalShortcuts`, `StrategicStochasticNoise`,
`PrecisionControl` (v2's Speed Enhancement Layer), plus `CyphaSystem`, `CyphaMonitor`,
`RecursiveEventCascades`, `SelfGeneratedEventStreams` and `ResonantEventChains`. The 8 it
shares with v3–v7 are just the HRNA level names, which v2 also has. vChatGPT is v2 taken
apart into modules and given a GUI, a trainer and an `agi/` package — not a branch off the
simplified line.

**vPattern Matching forks from the v3 era.** Its `Resonator`, `AnchorMemory`, `MetaLearning`
and `Cypha` are v3-generation names absent from v2; it shares 5 classes with v3 and 4 with
the standalone encoder, against only 2 with v2.


---

## The name "Cypha" is applied retroactively

The directory names in the archive are not contemporaneous labels. Searching the *contents*
of the two oldest directories:

| Directory | occurrences of "Cypha" in file contents |
|---|---|
| `archive/prototypes/` (9 files) | **0** |
| `archive/cypha-v1/` (4 files) | **0** |

The systems in `prototypes/` call themselves Cell AI v2, the Brain Model, and ICFHS /
OICFHS ("Optimized Integrated Cell-Fungal Harmonic System"). The work in `cypha-v1/` calls
itself **IRENA** — "Integrated Recursive Event-Driven Neural Architecture"
(`cypha-v1/Cypha Convo Log.txt:55`) — 11 times, and then **HRNA** 113 times after the
rename. The string "Cypha" first appears inside a file at **v2**, in
`archive/cypha-v2/Cypha.py`.

And it appears there *only* in the code. Within `cypha-v2/` the count is 30 in `Cypha.py`
and **0 in all four documents** — including `Cypha.docx`, whose own title is *"Comprehensive
Mathematical Framework for Optimized Event-Driven HRNA"*. Since that document is
[dated 2025-05-05](eras/02-v2.md#the-oldest-dated-artifact-in-the-archive) and a `.docx`
stores no filename, the boundary is not between directories at all — it runs **through v2**,
between a dated HRNA specification and the undated code that renamed it.

So the naming sequence is:

```
Cell AI / Brain Model / OICFHS  →  IRENA  →  HRNA  →  Cypha (from v2)
```

Directories named `Cypha V1`, and files named `Cypha Convo Log.txt` and `Cypha.docx`, were
titled later, when the archive was assembled. **Any statement that this era "was Cypha" is a retroactive
label, not a contemporaneous fact** — worth keeping in mind when reading the archive's own
filenames as evidence.

---

## Two architectural families

The single most important fact about Cypha's history is that it has **two unrelated
architectures**, and the version numbers conceal the break.

### Family A — Cypha HRNA (v1, v2–v7, and both side branches)

HRNA is **Harmonic Recursive Neural Architecture**, expanded in
[`archive/cypha-vpattern-matching/README.md`](archive/cypha-vpattern-matching/README.md):

> **Harmonic Recursive Neural Architecture** — A resonance-based AGI system for learning
> input-output mappings through quantum-inspired field dynamics.
>
> Cypha is a novel AI architecture that learns mappings through resonance field evolution
> rather than traditional backpropagation.

The acronym originates in v1: `cypha-v1/event-driven-math.md` is titled *"Event-Driven
Asynchronous Mathematical Framework for HRNA"*, and HRNA appears 113 times across
`cypha-v1/`.

Family A splits cleanly into two encoder generations, and the docstrings say so.

**v3 and v4 carry byte-identical headers:**

```
Cypha HRNA — Full Architecture
Binary+Cypha Universal Encoder · HLFC Compression · Resonance Field
Event-Driven Processing · Multi-Level Hierarchy · Adaptive Control Loop
Multicore CPU via concurrent.futures · Pure numpy
```

**v5, v6 and v7 carry a different byte-identical header**, which states the supersession
outright:

```
Cypha HRNA — Universal Encoder + Full Architecture (Omega-2)
Replaces BinaryEncoder with OmegaEncoder implementing the five-operator
Omega information field formula proven in Phase 8 neural network verification.
...
All HRNA layers (ResonanceField, ResonatorLevel, AssemblyLevel, ModuleLevel,
GlobalLevel, RecursiveProcessor, FeedbackController, ThoughtProcessor,
MetaLearning, AnchorMemory) are unchanged.
```

"Replaces BinaryEncoder with OmegaEncoder" is a direct statement that v5 post-dates v3/v4 —
the strongest single ordering edge in the archive. The layer list it declares unchanged
matches v7's class inventory, which places v7 at the end of the HRNA line rather than
anywhere else.

### Family B — CyphaDIF (v8, root monolith, and everything native)

v8 and the root monolith share a header that shares nothing with family A:

```
CyphaDIF  —  Differential Information Field Classifier
Architecture derived from first principles, unifying:
  ─ AIXI / Solomonoff  : MDL prior over class complexity (||Δk||_F ≤ C)
  ─ Information Geometry: natural gradient on Gaussian manifold (Cramér-Rao efficient)
  ─ Active Inference / FEP: world prior θ₀ + differential class offsets Δk
  ─ Information Bottleneck: contrastive encoder feedback (Fisher-Rao residuals)
```

### The break is real, not a refactor

| | v7 (family A) | v8 (family B) |
|---|---|---|
| lines | 5,650 | 1,412 |
| classes | 55 | 9 |
| bytes | 272,560 | 62,385 |
| shared class names | — | **zero** |

v7's 55 classes include `EventScheduler`, `AssemblyLevel`, `AnalogicalReasoner`,
`EpisodicMemory`, `CyphaDecoder`, `CyphaStateful`, `FeedbackController`, `AnchorMemory`.
v8's nine are
`StructuralParser`, `EncoderProjection`, `WorldPrior`, `ClassDifferential`, `DIFMemory`,
`ContextBuffer`, `NIGField`, `ReplayBuffer`, `CyphaDIF`. They have **no name in common**.

A 75% reduction in lines (77% in bytes) with zero shared classes and a docstring claiming
derivation "from first principles" is a restart. The resonance/harmonic programme that ran
from v1 to v7 — roughly thirteen months — was abandoned.

**And it is not an artifact of renaming.** Class names are the cheapest thing to change in a
refactor, so the same break was measured at the line level — counting distinct source lines
v8 holds in common with each other monolith. Because any two Python files share boilerplate
(`return out`, `else:`, `import numpy as np`), the count is meaningless without a length
threshold, so here it is across five:

| min. line length | comments | v5 | v6 | v7 | **root** | root ÷ best family A |
|---:|:--|---:|---:|---:|---:|---:|
| 1 | included | 27 | 29 | 30 | **262** | 8.7× |
| 1 | excluded | 27 | 29 | 30 | **255** | 8.5× |
| 10 | excluded | 13 | 15 | 16 | **240** | 15.0× |
| 25 | excluded | 2 | 3 | 3 | **145** | 48.3× |
| 40 | excluded | 1 | 1 | 1 | **86** | 86.0× |

The ratio is not stable — but its *direction* is, and so is what happens as the threshold
rises. Family-A overlap collapses to nothing (30 → 1) because it was always coincidental
boilerplate; the root-monolith overlap survives (262 → 86) because it is descent. At a
40-character threshold v8 and the root monolith still share 86 lines, and v8 shares **one**
with the entire thirteen months that preceded it.

### But the break is in the class structure, not in everything

"Zero shared class names" is exact, and it is also narrower than it sounds. At least one
*mechanism* crosses the gap intact.

`W_T`, an online-learned linear state-transition matrix, is introduced in v7:

> `W_T` is a dim×dim transition matrix trained online via rank-1 Oja-style update:
> `W_T += lr * (psi - W_T @ psi_prev) ⊗ psi_prev`
> This makes `W_T` approximate the local Jacobian of state dynamics.
> — `archive/cypa-v7-generation/Cypha.py:2148-2150`

and it is in v8 and the root monolith with the same rank-1 outer-product update, written in
the error-descent sign convention:

```python
err   = self._W_T @ h_t - h_target
W_new = self._W_T - lr * np.outer(err, h_t) / self.d
```
— `archive/cypha-v8/Cypha.py:857-858`, identical at `archive/root-monolith/Cypha.py:949-950`

Counting occurrences per archived monolith:

| | v5 | v6 | **v7** | **v8** | **root** |
|---|---|---|---|---|---|
| `W_T` | 0 | 0 | **23** | **6** | **15** |
| `update_causal` | 0 | 0 | 0 | **2** | **2** |
| `field_W_T` | 0 | 0 | 0 | 0 | **5** |

It appears nowhere before v7, survives the restart, gains its `update_causal` method in v8,
gains the `field_W_T` name in the root monolith, and ships today as

```cpp
/// Causal `W_T` SGD step + spectral-radius trim + refresh `a_eff` (Python `update_causal`).
void nig_field_update_causal(...)
```
— `native/include/cypha/nig_field.hpp:26`

So v8 discarded family A's *code* wholesale while keeping at least one of its mechanisms. A
second family-A idea, GRIA, took a stranger route back — see
[`LINEAGE.md`](LINEAGE.md#7-gria--the-one-family-a-idea-that-came-back). The restart is real,
but it was a rewrite with salvage, not a blank page.

---

## The two side branches are both family A

Neither `vChatGPT` nor `vPattern Matching` is on the main line; both are HRNA experiments.

**`cypha-vchatgpt`** is **v2** taken apart into modules (see the overlap matrix above: 30
shared classes with v2, 8 with v3–v7). Its `core/` package re-implements the named HRNA
layers across separate files —
`levels.py` defines `ResonatorLevel`, `AssemblyLevel`, `ModuleLevel`, `GlobalLevel`;
`resonance.py` defines `ResonanceField`; alongside `recursion.py`, `feedback.py`,
`metalearning.py`, `compression.py`, `events.py`, `encoder.py`, `optimization.py`,
`thought.py`. It is the only version that is a package rather than a monolith (38 files),
the only one with a test suite, the only one with a GUI (`PyQt6`), the only one with an
`agi/` package, and the only one to use `transformers` and `datasets`.

**`cypha-vpattern-matching`** is the opposite move: the HRNA core distilled to 996 lines
and 7 classes — `UniversalEncoder`, `ResonanceField`, `Resonator`, `AnchorMemory`,
`MetaLearning`, `Cypha` — and labelled "Production Implementation". It is the only branch
in the archive presented as production-ready, and it ships a 3.5 MB demo PDF. The label does
not survive inspection: its `infer()` answers seen inputs from an exact-string dict and
learns nothing, and its headline separation figure is measured against the wrong baseline —
see [`eras/branches.md`](eras/branches.md#it-does-not-retrieve-from-the-resonator-either).

Both kept a framework (`torch`), and neither continued.

---

## Supporting evidence: the dependency arc

Third-party imports, main line only:

| Version | Numerical stack |
|---|---|
| v2 | `torch`, `ray`, `scipy`, `psutil`, `numpy` |
| **v3** | **`numpy` only** |
| v4 | `numpy`, `sklearn` |
| v5 | `numpy`, `pandas` |
| v6 | `numpy`, `pandas`, `chess`, `treys` |
| v7 | `numpy` |
| v8 | `numpy` |

The v2 → v3 transition drops PyTorch, Ray and SciPy simultaneously, and the main line never
reinstates them. This corroborates v2 → v3 as an ordering edge (a project does not add Ray
*after* committing to `concurrent.futures`, which the v3 docstring advertises as
"Multicore CPU via concurrent.futures · Pure numpy"), and it is the precondition for the
eventual C++ port — see [`PYTHON_TO_CPP_BRIDGE.md`](PYTHON_TO_CPP_BRIDGE.md).

---

## Confidence in each edge

| Edge | Confidence | Basis |
|---|---|---|
| Prototypes → V1 | **high** | directory mtimes eight days apart, in the only period where mtimes are individually meaningful; V1 opens by treating all three prototype threads as finished inputs and fuses them |
| V1 → v2 | medium | V1 is design documents only; v2 is the first implementation |
| v2 → v3 | **high** | v3 drops torch/ray/scipy and advertises "Pure numpy"; frameworks never return |
| v3 → v4 | **high** | byte-identical module docstrings; v4 adds tooling without changing the core |
| v4 → v5 | **high** | v5 docstring: "Replaces BinaryEncoder with OmegaEncoder" |
| v5 → v6 → v7 | **high** | byte-identical Omega-2 docstrings; monotonic growth; v7 matches the declared layer list |
| v7 ⇸ v8 | **high** (as a *break*) | zero shared classes, 75% shrink, "derived from first principles" |
| v8 → root monolith | **high** | shared header plus root's own "Enhancement summary (v2)" listing five phases added |
| **v2 ⇢ vChatGPT** | **high** | shares **30** class names with v2 but only 8 with v3–v7, and the 30 include v2-only Speed-Layer classes (`AlternativeFastOperations`, `NaturalMathematicalShortcuts`, `StrategicStochasticNoise`, `PrecisionControl`) plus `CyphaSystem` and `CyphaMonitor` |
| v3 ⇢ vPattern Matching | medium | its `Resonator`, `AnchorMemory`, `MetaLearning` and `Cypha` are v3-era names absent from v2 (5 shared with v3, 4 with the standalone encoder, only 2 with v2) |
| root monolith → native C++ | **high** | 21 identifiers cited in `native/` resolve there and nowhere else |

The weakest links are the two side branches. They are securely placed in **family A** by
their class vocabulary, but nothing in the archive fixes *where* along v3–v7 they diverge.

---

## Loose components

| Item | Placement |
|---|---|
| `archive/cypha-encoder/Cypha_Encoder.py` | **byte-identical** to `archive/cypha-v3/Cypha_Encoder.py` (both md5 `31bd0a62020340f37e574e22fc083ca2`) — the v3 encoder kept as a standalone unit, not a separate development |
| `archive/big-data/download_datasets.py` | dataset acquisition tooling; belongs with the v5/v6 benchmarking effort |
| `archive/datasets/…jsonl.gz` | the 27,524-run sweep, whose winning constants appear hard-coded in the root monolith — see [`SWEEP_ANALYSIS.md`](SWEEP_ANALYSIS.md) |
| `archive/side-quests/retdec-upgrades5/` | RetDec decompiler optimiser patches dated 2026-03-11, the same day as v8; unrelated to the model |

---

## Two implementations named by the archive but absent from it

The seventeen-day gap at the end is not the only hole. Two implementations are named by
documents inside the archive and are not present in it.

**The first is the v6 engine.** `archive/cypha-v6/Cypha.py` is not the engine that v6's own
files were written against.

`game_benchmark.py:54-58` imports `deliberate_iterative`, `pnq_lookup`, `mcts_search` and
`gria_cascade` from `Cypha`, and none of the four exists in the `Cypha.py` beside it — the
harness cannot import. Those four names appear **zero** times in the `Cypha.py` of v5, v6, v7,
v8 *and* the root monolith. Yet `cypha-v6/Cypha_README.md` mentions `gria` 14 times, `hippo`
25, `pnq` 8, `mcts` 8, `rocchio` 5, `reflexion` 2 and `platt` 2.

The archived v6 engine is essentially v5's Omega-2 code. The engine its README documents and
its benchmark targets — with an ensemble deliberator, hippocampal memory, reflexion and Platt
calibration — was never archived. `GRIA`, which ships in the native product today, was
implemented there.

See [`LINEAGE.md`](LINEAGE.md#7-gria--the-one-family-a-idea-that-came-back).

**The second is CyphaGalois.** The v8 papers name their subject `CyphaDIF` 131 times across
seventeen documents. `cypha_synthesis.md` — and only it — says `CyphaGalois` three times,
including "U4 is now live in CyphaGalois.py. The implementation tracks orbit radius via EMA
in `ClassDifferential.orbit_r`". No such file exists in the archive, and `orbit_r` appears in
no Python file. The synthesis was written against a successor to the archived v8 that was
never preserved.

Both gaps have the same shape: a document survives, names an implementation, and the
implementation does not. See
[`eras/08-v8.md`](eras/08-v8.md#and-the-system-they-analysed-is-not-quite-the-archived-one).

---

## Where the record goes dark

The archive's last artifact is dated 2026-03-14. `CHANGELOG.md`'s oldest release is
`[0.1.0] — 2026-03-31`. **Seventeen days separate them, and nothing covers the gap.**

Worse, the interval from 2026-03-14 to the P7 Python decommission on 2026-08-16 is
undocumented in this repository by anything except its consequences: 38 of the 79 Python
identifiers cited in `native/` resolve in no archived version, naming subsystems — a CSV
ingest path, a preprocessor, an experiment database, a model registry, kernel memory, BPTT
— that were built during those five months and survive only as C++. See
[`PYTHON_TO_CPP_BRIDGE.md`](PYTHON_TO_CPP_BRIDGE.md#what-the-archive-does-not-explain).
