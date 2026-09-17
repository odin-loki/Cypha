# From `Cypha.py` to `cypha::Cypha`

How the Python archive connects to the native C++ product — and why the native source
still contains 79 references to a file that, until this archive was integrated, existed
nowhere in the repository.

---

## The gap this archive fills

Cypha's written record has three segments, and they do not overlap:

| Period | What records it | State |
|---|---|---|
| 2025-02-26 → 2026-03-14 | **this archive** (Prototypes → v8 → root monolith) | recovered here |
| 2026-03-31 → 2026-07-17 | `CHANGELOG.md` only, back to `[0.1.0] — 2026-03-31 · commit 127bbe9` | commits unreachable |
| 2026-07-18 → present | git history, 50 commits from `e8927b9` | intact |

The repository's git history begins on 2026-07-18. `CHANGELOG.md` reaches further back, to
2026-03-31, but the commits it names — `127bbe9`, `8945c95`, `1dbfa13` — are not objects in
this repository:

```console
$ for c in 127bbe9 8945c95 1dbfa13; do git cat-file -t $c; done
fatal: Not a valid object name 127bbe9
fatal: Not a valid object name 8945c95
fatal: Not a valid object name 1dbfa13
```

The archive's newest artifact is dated 2026-03-14. `CHANGELOG.md`'s oldest release is dated
2026-03-31. **The archive ends seventeen days before the changelog begins**, so it covers
precisely the prehistory that no other record in the repository reaches.

---

## `Cypha.py` was a real file, and it was deleted

`CHANGELOG.md:55`, under release **[2.4.0] — 2026-08-16 · Competition lock + forecasting**:

> **Python runtime decommissioned (P7):** `Cypha.py`, `cypha_studio/`, `cypha_core/`,
> `cypha_accel/`, `bench/` (Python package removed), `cypha_lm/`, and related packages
> removed from the product path. Native C++ (`cypha_core` library, `cypha_rest`,
> `cypha_qt_shell`, `cypha_bench_run`) is the sole runtime.

`Cypha.py` is named first. It was never tracked in this repository's git history
(`git log -- Cypha.py` is empty across all 50 commits), because it was removed in the
repository that preceded this one. Five Python files remain in the tree today, all under
`scripts/`, and none is on the product path.

The file the changelog deleted is archived here as
[`archive/root-monolith/Cypha.py`](archive/root-monolith/Cypha.py) — 233,256 bytes,
5,346 lines, 19 classes.

---

## The native source still cites it

35 files under `native/src` and `native/include` mention Python — 56 across `native/` as a whole. Between them they name **79 distinct Python
identifiers** in doc comments — classes, methods and attributes described as the behaviour
the C++ is matching. These are not vague nods; they are precise parity references:

```cpp
// native/include/cypha/nig_field.hpp
/// Diagonal timescales τ (same grouping as Python `NIGField`).
/// `field_h += strength * (‖field_h‖/‖signal‖) * signal` with L2 cap (Python `NIGField.inject`).
/// Causal `W_T` SGD step + spectral-radius trim + refresh `a_eff` (Python `update_causal`).

// native/include/cypha/similarity_index.hpp
/// Mahalanobis similarity index over encoded latents (Python ``SimilarityIndex``).

// native/include/cypha/multilabel_dif.hpp
/// Parameters for ``MultiLabelDIF`` (Python ``MultiLabelDIF`` defaults).
/// One binary ``CyphaDIF`` (pos/neg) per semantic label, sharing ``VectorEncoder`` input dim.

// native/include/cypha/mke_scalar_train_step.hpp
/// One scalar ``MKERegressor.train_step`` composition in native ...

// native/include/cypha/replay_buffer.hpp
/// Priority replay matching `PriorityReplayBuffer` decay + weighted sampling (subset).
```

`NIGField`, `SimilarityIndex`, `MultiLabelDIF`, `CyphaDIF`, `VectorEncoder`,
`MKERegressor`, `PriorityReplayBuffer` — every one of those is a class defined in
`archive/root-monolith/Cypha.py`.

---

## Which archived version is the porting source?

Each of the 79 cited identifiers was resolved against every candidate monolith in the
archive (a class defined, or a method/attribute defined on a class of that name):

| Candidate | Resolved | Share |
|---|---|---|
| **`archive/root-monolith/Cypha.py`** | **41 / 79** | **52%** |
| `archive/cypha-v8/Cypha.py` | 20 / 79 | 25% |
| `archive/cypa-v7-generation/Cypha.py` | 3 / 79 | 4% |
| `archive/cypha-v6/Cypha.py` | 3 / 79 | 4% |
| `archive/cypha-v5/Cypha.py` | 3 / 79 | 4% |
| `archive/cypha-v2/Cypha.py` | 0 / 79 | 0% |

The root monolith wins by better than two to one, and **21 identifiers resolve there and
nowhere else**:

```
CyphaDIF._W_inject          CyphaDIF.adapt_temperature   CyphaDIF.auto_recalibrate
MKERegressor.train_step     MultiLabelDIF                RFFEncoder
RFFEncoder.W                RFFEncoder.auto_gamma        RFFEncoder.auto_gamma_cv
SimilarityIndex             TieredContextBuffer          TieredContextBuffer.predict_next
TieredContextBuffer.record  _batch_logpdf                _mid_freq_total
_to_field_dim               auto_gamma_cv                field_W_T
llr_scale_baseline          load_state                   save_state
```

`TieredContextBuffer` is the cleanest discriminator: v8 has a plain `ContextBuffer`, and
only the root monolith has the tiered form the C++ names.

### The root monolith is not v7

The archive root's `Cypha.py` and `Cypha v7 Generation/Cypha.py` differ in size, in line
count and in md5 — and they share **zero** class names:

| | root monolith | v7 |
|---|---|---|
| md5 | `b91a59b0408854f0ac95eb84aa19eaef` | `174df63c0efe3b9ba2338f19333393e6` |
| lines | 5,346 | 5,650 |
| classes | 19 | 53 |

The root monolith's classes are the v8 DIF lineage extended — `CyphaDIF`, `WorldPrior`,
`ClassDifferential`, `DIFMemory`, `NIGField`, `EncoderProjection` (all present in v8) plus
`RFFEncoder`, `MKERegressor`, `DIFRegressor`, `MultiLabelDIF`, `SimilarityIndex`,
`TieredContextBuffer`, `PriorityReplayBuffer`, `MultiModalCyphaDIF`, `ClassifierDistillation`.
v7's are a different program entirely (`EventScheduler`, `AssemblyLevel`,
`AnalogicalReasoner`, `EpisodicMemory`, `CyphaDecoder`, `CyphaStateful`).

**The root monolith is v8's successor, not v7's copy.** v7 is a branch off the main line,
not a step along it.

---

## Component map

Classes in [`archive/root-monolith/Cypha.py`](archive/root-monolith/Cypha.py) against the
native translation units that carry their behaviour:

| Python class (line) | Native header | Evidence |
|---|---|---|
| `NIGField` (873) | [`nig_field.hpp`](../../native/include/cypha/nig_field.hpp) | named 3× in header comments |
| `PriorityReplayBuffer` (992) | [`replay_buffer.hpp`](../../native/include/cypha/replay_buffer.hpp) | "matching `PriorityReplayBuffer` decay" |
| `SimilarityIndex` (4246) | [`similarity_index.hpp`](../../native/include/cypha/similarity_index.hpp) | "(Python ``SimilarityIndex``)" |
| `MultiLabelDIF` (4138) | [`multilabel_dif.hpp`](../../native/include/cypha/multilabel_dif.hpp) | "(Python ``MultiLabelDIF`` defaults)" |
| `MKERegressor` (3674) | [`mke_scalar_train_step.hpp`](../../native/include/cypha/mke_scalar_train_step.hpp) | "``MKERegressor.train_step`` composition in native" |
| `RFFEncoder` (3511) | [`rff_features.hpp`](../../native/include/cypha/rff_features.hpp), [`orthogonal_rff_encoder.hpp`](../../native/include/cypha/orthogonal_rff_encoder.hpp) | `RFFEncoder.W`, `auto_gamma`, `auto_gamma_cv` cited |
| `EncoderProjection` (215) | [`encoder_contrastive.hpp`](../../native/include/cypha/encoder_contrastive.hpp) | `EncoderProjection.__init__`, `.align_to_offsets` cited |
| `DIFRegressor` (3961) | [`regression.hpp`](../../native/include/cypha/regression.hpp) | shared regression path |
| `CyphaDIF` (1085) | [`cypha.hpp`](../../native/include/cypha/cypha.hpp), [`infer_cpu.hpp`](../../native/include/cypha/infer_cpu.hpp), [`dif_rest.hpp`](../../native/include/cypha/dif_rest.hpp) | 12 distinct `CyphaDIF.*` members cited |
| `DIFMemory` (438) | [`kernel_memory.hpp`](../../native/include/cypha/kernel_memory.hpp), [`memory_train.hpp`](../../native/include/cypha/memory_train.hpp) | `DIFMemory.classify`, `.get_class_params` |
| `TieredContextBuffer` (707) | context path in [`cypha.hpp`](../../native/include/cypha/cypha.hpp) | `.record`, `.predict_next` cited |
| `WorldPrior` (285) | world-prior state in [`cypha.hpp`](../../native/include/cypha/cypha.hpp) | `WorldPrior.F_field` cited |
| `ClassDifferential` (378) | DIF core | `ClassDifferential.fisher_rao_norm` cited |
| `Encoder` / `VectorEncoder` / `ConcatEncoder` (156/171/188) | encoder interfaces | `VectorEncoder` named in `multilabel_dif.hpp` |
| `MultiModalCyphaDIF` (4521) | — | no native counterpart |
| `ClassifierDistillation` (5233) | — | no native counterpart |
| `PerformanceMonitor` (4394) | `bench/` runners | superseded by `cypha_bench_run` |

### Where the port deliberately diverges

The native code is close to a transliteration, but three divergences are documented in its
own comments and are worth knowing before treating it as a faithful copy.

**The production encoder starts as the identity.** The Python ancestor initialises
`EncoderProjection` with a random projection. A freshly created native model does not:

```cpp
// Encoder: identity (VectorEncoder with no learning at start)
root.map.push_back({"enc_W", identity_2d(d)});
```
— `native/src/create_model.cpp:132-133`

`init_encoder_projection_w` — the orthogonalising initialiser that mirrors the Python — exists
in `encoder_contrastive.cpp:254` but is called only from `native/tools/`
(`class_gmm_p3_smoke`, `kernel_cypha_roundtrip`, `encoder_ib_p6_smoke`, `xor_kernel_bench`).
It is reference and bench code, not the product path.

**Deliberation is off, for a portability reason.** `train_step_vector.cpp:24-27` disables the
mid-confidence contrastive nudge because "the old hardcoded 0.25–0.40 band re-enabled
deliberation during train and amplified MSVC/MinGW FP drift" — the Python band, carried
across and then switched off for cross-compiler floating-point reproducibility. See
[`SWEEP_ANALYSIS.md`](SWEEP_ANALYSIS.md#why-it-was-switched-off-is-not-what-you-would-guess).

**Temperature recalibration is off.** `temp_recalib_every{0}`, documented as "default off",
with the note that "Python has no per-step hook" — so this is a native addition that is
disabled rather than a Python behaviour that was dropped
(`native/include/cypha/train_step_vector.hpp:25-26`).

The pattern across all three: the port preserves the mechanism and the parameter names, and
chooses different defaults. Anyone reading the archive to predict native behaviour should
check the defaults rather than assume parity.

### What the archive does *not* explain

38 of the 79 cited identifiers resolve in **no** archived version:

```
CSVDataset.from_file   Preprocessor.fit      Trainer.fit        ExperimentDB / Experiment / Run
ModelRegistry.register create_experiment     list_experiments   compare_runs / leaderboard
CyphaDIF.gh_train_step CyphaDIF.gh_infer     CyphaDIF.retrieve  CyphaDIF._kernel_mem
CyphaDIF.merge_from    CyphaDIF._apply_deliberation             CausalField._step
_bptt_ssm_update       _nig_adapt / _nig_R_eff cypha_save_binary / cypha_load_binary   …
```

These name whole subsystems — a CSV ingest path, a preprocessor, an experiment database, a
model registry, a GH–NIG gate, kernel memory, BPTT — that the archive's last snapshot does
not contain but the native port does. They mark the four and a half months between
2026-03-14 and the P7 decommission on 2026-08-16, during which the Python continued to grow
before being rewritten. That interval remains undocumented by any artifact in this
repository; the archive establishes its starting point, not its contents.

---

## Why a C++ rewrite was available at all

The archive answers this in its import lists. Third-party dependencies, main line only:

| Version | Numerical stack |
|---|---|
| v2 | `torch`, `ray`, `scipy`, `psutil`, `numpy` |
| **v3** | **`numpy` only** — torch, ray and scipy all dropped |
| v4 | `numpy`, `sklearn` |
| v5 | `numpy`, `pandas` (+ archive/format tooling) |
| v6 | `numpy`, `pandas`, `chess`, `treys` (game benchmarks) |
| v7 | `numpy` |
| v8 | `numpy` |

PyTorch was abandoned at v3 and never returned to the main line. From v3 onward every
kernel — the NIG updates, the Fisher-Rao residuals, the field dynamics, the RFF features —
was hand-written against plain arrays. A codebase in that shape has no framework to port,
only arithmetic, which is why the native rewrite could be a faithful translation with
golden-fixture parity rather than a reimplementation.

The two branches that kept a framework are exactly the two that did not continue:
`cypha-vchatgpt` (`torch`, `transformers`, `datasets`, `PyQt6`) and
`cypha-vpattern-matching` (`torch`, `sklearn`).

---

## Related

- [`TIMELINE.md`](TIMELINE.md) — the full chronology and version DAG
- [`SWEEP_ANALYSIS.md`](SWEEP_ANALYSIS.md) — the sweep behind the shipping `deliberation_lo/hi` defaults
- [`archive/MANIFEST.md`](archive/MANIFEST.md) — byte-exact inventory with checksums
- [`docs/native/migration/CPLUSPLUS_2023_MASTER_PLAN.md`](../native/migration/CPLUSPLUS_2023_MASTER_PLAN.md) — the P0–P8 decommission plan
