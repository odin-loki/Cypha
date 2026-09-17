# Threads that run the whole length

Seven ideas that can be traced from the 2025 design dialogues to code shipping
in `native/` today, and one pathology that explains why the project restarted from scratch.

Each thread is stated with its evidence. Where a thread dies, that is said plainly.

---

## 1. Single-core, commodity hardware — the constraint that never moved

The most consequential line in the archive is a question.

Asked about hardware in March 2025, the V1 dialogue produced a cluster specification —
64–256 CPU cores, 4–8 A100/H100 GPUs, 128 GB–1 TB RAM, InfiniBand, a **$1M–$10M** budget
(`archive/cypha-v1/Cypha Convo Log.txt:1505-1580`). The next human turn discarded all of it:

> Is there anyway we can get this running back on a single core?
> — `cypha-v1/Cypha Convo Log.txt:1584`

What followed was a ten-item single-core optimisation programme and a minimum viable
configuration of 16 resonators on commodity hardware. The author thought it important enough
to save separately as `Cypha Speed Improvements.txt`.

The constraint held for the rest of the project's life, with exactly one lapse:

| Version | Posture |
|---|---|
| v1 | single core, decided at `:1584` |
| **v2** | **Ray actors, fractional GPU reservations, `device='cuda'`** — while its own document claims single-core operation |
| v3 | frameworks deleted; pure numpy |
| v4–v8 | pure numpy, single process |
| native | CPU-first; `CHANGELOG.md` records CUDA CI jobs being **removed** |

The v2 lapse is visible in one grep: `ray.remote` matches exactly one file in the whole
archive. And the endpoint is in the changelog, where `windows_cuda_msvc` and `linux_cuda`
are struck from CI and CUDA becomes "an optional local build".

Thirteen months, eight versions, one language change, and the answer to "what hardware does
this need" never changed after the day it was asked.

---

## 2. The dependency floor — why a C++ port was available at all

| Version | Third-party numerical stack |
|---|---|
| v2 | `torch`, `ray`, `scipy`, `psutil`, `numpy` |
| **v3** | **`numpy` only** |
| v4 | `numpy`, `sklearn` |
| v5 | `numpy`, `pandas` |
| v6 | `numpy`, `pandas`, `chess`, `treys` |
| v7 | `numpy` |
| v8 | `numpy` |

v3 dropped PyTorch, Ray and SciPy in a single move and the main line never reinstated them.
From that point every kernel — the NIG updates, the Fisher–Rao residuals, the field
dynamics, the RFF features — is hand-written against plain arrays.

That is the precondition for everything that happened later. A codebase in that shape has no
framework to port, only arithmetic, so the native rewrite could be a faithful translation
with golden-fixture parity rather than a reimplementation. The two branches that kept a
framework — `vchatgpt` (`torch`, `transformers`, `datasets`, `PyQt6`) and
`vpattern-matching` (`torch`, `sklearn`) — are exactly the two that did not continue.

See [`PYTHON_TO_CPP_BRIDGE.md`](PYTHON_TO_CPP_BRIDGE.md#why-a-c-rewrite-was-available-at-all).

---

## 3. Computed and discarded — the pathology that ended the HRNA line

This is the thread that explains the restart.

**v2** computes its four compression stages in `process_input` and never reads the results;
the resonance field is fed the uncompressed tensor. The stages are also applied in parallel
to the same input rather than composed, so the specified operator
`C(Ψ) = Fold(Map(Encode(Extract(Ψ))))` is never actually evaluated
(`archive/cypha-v2/Cypha.py:1216-1230`).

**v3** composes the compression correctly (`:333`) — and then discards *the entire
architecture*. `train_step` stores an anchor built from raw encoder features (`:1225`) and
`infer` queries the same way (`:1291`), so the resonance field, four-level hierarchy, event
system, recursion operators and feedback controllers are computed on every call and
contribute nothing. Verified by ablation: replacing `forward()` with a constant leaves recall
at 200/200 and runs 13× faster.

**v4** leaves the data flow untouched and lowers two thresholds by 7× and 15× so the unused
machinery produces visible activity — a measured showcase emits 11,163 THOUGHT events against
50 RESONANCE.

**v5 finds it.** Not the big one, but the pattern:

```python
# HLFC.compress() removed from hot path (profiled 2025-02):
#   output was never consumed by train_step() or infer()
#   cost: 0.254ms/call = 45% overhead, ~50s wasted per 100k-sample epoch
#   reconstruction cos_sim = 0.48 (lossy on sparse Omega vectors)
```
— `archive/cypha-v5/Cypha.py:2373-2379`

Someone profiled a subsystem, found nothing read its output, measured the waste, checked the
reconstruction quality and deleted it with a note. That is the project's engineering culture
working exactly as it should.

**And v5 keeps the larger instance.** Store and lookup still use the raw Omega vector;
`forward()` still feeds only the loss and the verbose print. The 0.254 ms orphan was removed;
the architectural orphan was not.

**vChatGPT says it out loud.** The one branch that annotates the pathology instead of hiding
it. In both its inference and training paths:

```python
with self._time_block("global"):
    # BYPASS broken layers - use Resonator directly
    globalv = reso[:64]
```
— `archive/cypha-vchatgpt/cypha.py:173-175` and `:219-221`

`AssemblyLevel` and `ModuleLevel` are still called and still timed; their outputs are
discarded and the downstream value is a slice of the resonator output. `GlobalLevel` is
constructed and never invoked. Of ten decomposed layers, four are on the working path.

**v7** grows to 55 classes and 5,650 lines on top of that same core.

**v8 restarts.** 1,412 lines, 9 classes, zero class names in common with v7, and a docstring
claiming derivation "from first principles". Every equation in the new core — world prior,
class differential, MDL decay, Fisher–Rao residual — is on the critical path by construction.

After five versions of accumulating machinery on a core that was never load-bearing, starting
from a small set of equations that are *all* load-bearing was the cheaper move. That is the
best explanation the archive supports for the restart, and every step of it is in the source.

---

## 4. Adaptivity — four negative results and a shipped default

Cypha spent years trying to make its own hyperparameters adapt. It measured the attempts
carefully, and the measurements kept coming back negative. The native product's defaults
record the verdict.

**v6 profiler 3 — adaptive α loses to a fixed constant, in every cell.**

| Modality | n/class | Winner | `Fixed α=0.05` rank | `Adaptive (TP)` rank |
|---|---|---|---|---|
| text | 50 | Fixed α=0.05 (0.990) | **1** | 8 / 9 |
| text | 200 | Switch adaptive→0.05 (0.998) | 2 | 7 / 9 |
| text | 500 | Fixed α=0.05 (0.998) | **1** | 6 / 9 |
| audio | 50 | Anneal 0.40→0.05 (0.905) | 2 | 7 / 9 |
| audio | 200 | Fixed α=0.05 (0.892) | **1** | 6 / 9 |
| audio | 500 | Fixed α=0.15 (0.877) | 4 | 9 / 9 |
| structured | 50 | Fixed α=0.05 (0.653) | **1** | 8 / 9 |
| structured | 200 | Fixed α=0.05 (0.658) | **1** | 6 / 9 |
| structured | 500 | Fixed α=0.05 (0.655) | **1** | 7 / 36 |

`Adaptive (TP)` is never better than sixth, and fixed α = 0.05 beats it in **9 of 9** cells by
between **5.2 and 22.8 percentage points**. The one non-fixed winner, "Switch adaptive→0.05",
ends at 0.05 anyway. The report states its own decision rule verbatim:

> If fixed_low always wins → adaptive mechanism should be disabled entirely.
> — `archive/cypha-v6/Profiling/profiler3_alpha_report.txt`

**v6 profiler 2 — the adaptation path never fires once.** For every distribution shift in
{0.1, 0.2, 0.3, 0.5, 0.7}, the best-threshold row reads `writes=0`, `write_acc=0.000`,
`recovery=+0.000`, `forgetting=+0.000`, `net_gain=+0.000`. A confidence threshold of 1.5
gates out 100% of writes. (Accuracy itself degrades gracefully to shift 0.2, then collapses:
22.4% drop at 0.3, 52.0% at 0.5, 73.6% at 0.7.)

**v6 profiler 4 — the archetype detector gets worse with more evidence.** Overall accuracy
falls 0.333 → 0.200 → 0.167 as the detection step moves 6 → 10 → 20/30/50. Type C is never
detected at any step (F1 = 0.00 throughout); Type B collapses from F1 0.48 to 0.00 by step 20.
The detector converges on predicting Type A for everything.

**The sweep — two of five knobs are inert.** A complete 5,250-cell factorial grid at 1,070.8
core-hours found `deliberate_thresh` and `post_trans_alpha` varying only in the fourth decimal
across their full ranges. See [`SWEEP_ANALYSIS.md`](SWEEP_ANALYSIS.md).

**The outcome, in code shipping today.** The single `deliberate_thresh` was replaced by a
two-sided hysteresis band, ported to C++, kept under regression test — and shipped **off**:

```cpp
constexpr double kDeliberationLoDefault = 1.0;
constexpr double kDeliberationHiDefault = 0.0;
/// Python ``CyphaDIF.deliberation_lo/hi`` defaults (disabled: lo >= hi).
```
— `native/include/cypha/infer_cpu.hpp:19-21`

The Python ancestor shipped it **on** (`_DELIBERATE_LO = 0.25`, `_DELIBERATE_HI = 0.40` in
`archive/root-monolith/Cypha.py`). The native port kept the mechanism and the parameter
names, and inverted the sentinel.

The stated reason is not the sweep, though. `native/src/train_step_vector.cpp:24-27` records
that "the old hardcoded 0.25–0.40 band re-enabled deliberation during train and amplified
MSVC/MinGW FP drift" — it was switched off for cross-compiler floating-point reproducibility.

So an expensive programme of negative results did not delete a feature, and neither did it
disable one: it demoted a formulation, and a portability problem later closed the switch. The
two causes agree in direction and should not be collapsed into one.

---

## 5. The encoder — the one component that was rewritten every time

No part of Cypha changed more often, and it is the only part that survived the restart.

| Era | Encoder | Note |
|---|---|---|
| Prototypes | inherited Cell AI field equation | the engine itself is **lost**; not in the archive |
| v1 | Signal AI complex frame expansion `E(x) = ∑ αᵢ(x)e^{iθᵢ(x)}φᵢ(x)` | paper only |
| v2 | same, plus mantissa/exponent precision and overflow layers | the precision layer adds no significand bits |
| **CyphaMicro** | content-defined chunking → D4 wavelet → DAMR reservoir → 64-D | closes its adaptive loop honestly; retrieves from the dynamical state; 7/14 recall |
| v3 | 4096-D hashed n-grams (1024 uni + 1024 bi + 1024 tri) + 86 structural dims | keeps the name `BinaryEncoder`, discards the wavelet algorithm entirely |
| v4 | unchanged from v3 | |
| **v5** | **OmegaEncoder** — `Ω(x) = [M(x), M(D(x)), M(D²(x)), R(x,K), A(x,L)]` at 3 scales | plus IQ and mel rescues after pure Omega collapsed at cross-class cosine ≈ 0.99 |
| v6, v7 | OmegaEncoder | |
| **v8** | contrastive Fisher–Rao encoder `EncoderProjection` | family B; nothing inherited |
| root | `Encoder` / `VectorEncoder` / `ConcatEncoder` / `RFFEncoder` + `EncoderProjection` | the porting source |
| **native** | `encoder_contrastive.hpp`, `orthogonal_rff_encoder.hpp`, `rff_features.hpp` | cites `EncoderProjection.__init__`, `.align_to_offsets`, `RFFEncoder.W`, `.auto_gamma`, `.auto_gamma_cv` by name |

Two observations.

**The encoder is where the project kept learning.** Each rewrite responds to a measured
failure: CyphaMicro's non-determinism, v3's hash collisions, v5's cross-class collapse on
signal data.

**The restart did not spare it.** v8's contrastive Fisher–Rao encoder shares nothing with
OmegaEncoder. What reaches `native/` descends from v8 and the root monolith, not from the
five encoders that preceded them.

---

## 6. The information field — the only mathematics that survived

Family A's resonance field (`∂R/∂t = −i[H,R] + γ(R² − R)`) is gone. What reached the product
is family B's **Normal-Inverse-Gamma** machinery, and it arrived fully formed in v8:

```
Class k model  =  θ₀ ⊕ Δk          world prior + differential offset (natural parameters)
y* = argmax_k [ log p(h | θ₀ ⊕ Δk) ] + log p(k | context)

Attraction:  Δk += η · residual(h, θk)
Repulsion:   Δj −= η · wj · residual(h, θj)
MDL decay:   Δk *= (1 − λ)
World prior: θ₀ updated via Welford
```
— condensed from the module docstring, `archive/cypha-v8/Cypha.py:3-31`

`NIGField` appears in v8 as an EMA filter bank over multiple timescales, gains a τ = 0.99
group in the root monolith's "Phase 1", and is named three times in the header of the C++
file that implements it:

```cpp
/// Diagonal timescales τ (same grouping as Python `NIGField`).
/// `field_h += strength * (‖field_h‖/‖signal‖) * signal` with L2 cap (Python `NIGField.inject`).
/// Causal `W_T` SGD step + spectral-radius trim + refresh `a_eff` (Python `update_causal`).
```
— `native/include/cypha/nig_field.hpp:9,20,26`

Around it sit `nig_gig_math.hpp`, `nig_gig_score_match.hpp` and `bessel_table.hpp`. This is
the strongest line of continuity in the project — and it is only twelve days long on the
Python side, running from v8 (2026-03-11) to the root monolith (2026-03-14) before the port.

Thirteen months of resonance mathematics produced the engineering culture, the single-core
constraint, the numpy floor and the measurement habits. Almost all of the mathematics that
ships as the *classifier core* was written in the last week and a half.

**Two exceptions.** `W_T`, the online-learned causal transition matrix, is introduced in v7
and crosses the restart intact — same rank-1 outer-product update in v8 and the root
monolith, shipping today as `nig_field_update_causal` with its spectral-radius trim (see
[`TIMELINE.md`](TIMELINE.md#but-the-break-is-in-the-class-structure-not-in-everything)). And
`GRIA` takes a stranger route still — thread 7.

---

## 7. GRIA — the one family-A idea that came back

`GRIA` is **Graded Reversible-Irreversible Algebra**, defined in v6:

> GRIA is a unified mathematical framework for compression and cryptography. The central
> idea is a grade parameter alpha in the range [0, 1] that interpolates smoothly [between
> two] operational modes: reversible (alpha = 0, lossless) and irreversible (alpha = 1,
> destructive). At intermediate values, a GRIA operation is partially reversible — some
> information is preserved, the rest is committed.
> — `archive/cypha-v6/Cypha_README.md`

In v6 it is live code: `gria_cascade(query, adapter, levy, noise, cg, ne_volatility, domain,
grade, candidates)` is one of the deliberation strategies the game benchmark monkey-patches
and profiles, alongside Rocchio, MCTS and PNQ.

Then it disappears. Counting files containing "gria" per archived tree:

| tree | v2 | v3 | v4 | v5 | **v6** | v7 | v8 | root | vChatGPT | vPattern |
|---|---|---|---|---|---|---|---|---|---|---|
| files | 0 | 0 | 0 | 0 | **5** | 0 | 0 | 0 | 0 | 0 |

It is in v6 and **nowhere else in the archive** — not in v7, not in the v8 restart, not in
the root monolith that the native port was written from.

And yet it is in the product today, with the α-grade concept intact:

```cpp
/// GRIA alpha live topology controller (U3; off by default).
class GRIAController {
  /// Entropy-based structural readiness in [0, 1] (0.5 until buffer warm).
  double alpha() const;
  /// Periodic GNG topology action: "skip", "hold", "split", or "merge".
```
— `native/include/cypha/som/gria_controller.hpp`

alongside `native/src/cyphalm/gria_lowrank.cpp`, `gria_projection_alpha` in
`cyphalm_alpha_spectrum.hpp`, and `anchor_gria_alpha_` in the EWC regulariser, whose header
describes "Hybrid EWC: char-LSTM + SSM multiscale ``alpha`` + GRIA per-token ``alpha``".
The living sequence default named in [`docs/README.md`](../README.md) is
**Hybrid GRIA+LSTM L2+Wave2 BPTT**.

### The v6 engine is missing from the archive

The explanation is that `archive/cypha-v6/Cypha.py` is **not the engine v6's own files were
written against.**

`game_benchmark.py` opens with

```python
from Cypha import (CyphaStateful, _build_offset_index, _read_at_offset,
                   deliberate_iterative as _orig_delib,
                   pnq_lookup           as _orig_pnq,
                   mcts_search          as _orig_mcts,
                   gria_cascade         as _orig_gria)
```
— `archive/cypha-v6/game_benchmark.py:54-58`

and **four of those seven names do not exist** in the `Cypha.py` sitting beside it:

| symbol | in `cypha-v6/Cypha.py`? |
|---|---|
| `CyphaStateful`, `_build_offset_index`, `_read_at_offset` | yes — all v5-era |
| `deliberate_iterative`, `pnq_lookup`, `mcts_search`, `gria_cascade` | **no** |

`game_benchmark.py` cannot even import against the archived engine. Counting the terms across
every archived monolith confirms it — `gria`, `pnq`, `mcts` and `deliberate_iterative` appear
**zero** times in the `Cypha.py` of v5, v6, v7, v8 *and* the root monolith:

| `Cypha.py` in | gria | pnq | mcts | deliberate_iterative |
|---|---|---|---|---|
| v5, v6, v7, v8, root | 0 | 0 | 0 | 0 |

Meanwhile `archive/cypha-v6/Cypha_README.md` — the documentation shipped in the same
directory — mentions **gria 14 times, hippo 25, pnq 8, mcts 8, rocchio 5, reflexion 2,
platt 2**.

So the v6 directory holds three things that do not agree: an engine that is essentially v5's
Omega-2 code, a README describing a far more elaborate engine with an ensemble deliberator
(Rocchio + MCTS + PNQ + GRIA), hippocampal memory, reflexion and Platt calibration, and a
benchmark harness written against *that* engine. **The engine the README and the benchmark
describe was never archived.**

That is where `gria_cascade` lived, and it is the most substantial single gap in the record —
larger than the seventeen days between the archive's end and the changelog's start, because
it sits in the middle of the archive rather than after it.

### What this means for the restart

A piece of family-A mathematics — specified in v6, implemented in an engine the archive does
not contain, absent from v7, absent from the v8 restart, absent from the porting source — is
a named component of the current product's default configuration.

So the restart was **not** a clean break in practice. v8 discarded family A's code, but the
ideas survived in documents and in a lost implementation, and at least one was brought back
once the new core worked. The undocumented window from 2026-03-14 to the P7 decommission
should be read as a period of recovery as much as of new work.

---

## Where these threads are documented in detail

| Thread | Primary document |
|---|---|
| Chronology and the family break | [`TIMELINE.md`](TIMELINE.md) |
| The port and the component map | [`PYTHON_TO_CPP_BRIDGE.md`](PYTHON_TO_CPP_BRIDGE.md) |
| The sweep and the deliberation default | [`SWEEP_ANALYSIS.md`](SWEEP_ANALYSIS.md) |
| Per-era detail | [`eras/`](eras/) |
