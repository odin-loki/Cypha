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
machinery produces visible activity. On a fresh model `showcase()` emits **8** events over ten
steps and no cascade ignites; after training on 120 pairs the same call emits **6,340**, almost
all THOUGHT. The absolute count depends on training history — it is driven by
`self._patterns[-5:]` — so it is the ~790× ratio, not either number, that is the finding. See
[`eras/04-v4.md`](eras/04-v4.md#what-the-patch-manufactured-the-thinking).

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

**v7** grows to 55 classes and 5,650 lines on top of that same core — and breaks the pattern
once, in one direction, five days before it was deleted.

`CyphaDecoder` is the only place in thirteen months of HRNA where the resonance field is
load-bearing. Not for classification — v7's `infer()` still queries memory with the pre-field
encoder vector (`archive/cypa-v7-generation/Cypha.py:4205`, `q = res['anchor_q']`), exactly as
v3 did. For **generation**. `generate()` primes its state from the field-derived output of
`forward()`:

```python
prime_out = cypha.forward(prime_text, training=False)
h = prime_out['state'].astype(np.float64)   # 256-dim real state
```
— `archive/cypa-v7-generation/Cypha.py:5376-5377`

and then re-injects the live field into that state on every single step:

```python
cypha.field.evolve(1)
psi_now = cypha.field.psi.real.astype(np.float64)
n_pn = min(len(psi_now), self.state_dim)
h[:n_pn] = 0.95 * h[:n_pn] + 0.05 * psi_now[:n_pn]
```
— `archive/cypa-v7-generation/Cypha.py:5518-5521`

That 5% term is not decorative. `evolve()` renormalises ψ to unit norm every step while `h`
decays, so the measured norm ratio `‖0.05·ψ‖ / ‖0.95·h‖` is **1.69** — the live field is the
larger of the two contributions to the state that produces logits. Removing the blend changes
the top-8 sampling candidates on **48 of 48** steps, a mean of 5.88 tokens swapped, and moves
`argmax(C·h)` on 43 of 48. The field is also prime-dependent rather than free-running:
cos(ψ | *"cat sound"*, ψ | *"capital of Japan"*) averages 0.036 across the run.

**Two of the three advertised field couplings do not survive measurement, though.** The
criticality-scaled temperature at `:5416-5417` (`tau = temperature * (1.0 + kappa_temp_scale *
kappa)`) is genuinely wired and κ is genuinely nonzero (mean 0.194) — but the logits are flat
(full-softmax entropy 5.545177 against ln 256 = 5.545177), so it moves the sampling
distribution by a total variation of 5.2 × 10⁻⁵ and never once changes the candidate set. And
`W_T = cypha.recursive._W_T  # live causal matrix` (`:5404`) is initialised to `eye(dim)*0.01`
and barely moves: measured spectral radius **0.0138**, so `h = W_T @ h` (`:5484`) contracts the
state hundred-fold per step and *erases* the priming rather than propagating it. The comment
at `:5481` claiming `sr≈0.077` did not replicate.

So the exception is real but narrow: one channel of three, in the one capability v8 removed.
It is also never exercised — the file's own `__main__` smoke test never calls `generate()` or
`train_decoder()`, and nothing else in the archive does either.

**v8 restarts.** 1,412 lines, 9 classes, zero class names in common with v7, and a docstring
claiming derivation "from first principles". Every equation in the new core — world prior,
class differential, MDL decay, Fisher–Rao residual — is on the critical path by construction.

After five versions of accumulating machinery on a core that was never load-bearing for
classification — the one exception above being a generation path v8 deleted outright — starting
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

**The sweep — two of five knobs are inert.** A complete 5,250-cell factorial grid — 26,250
runs at **1,013.2 core-hours**, the tier-1 share of the programme's 1,070.8 total — found
`deliberate_thresh` and `post_trans_alpha` varying only in the fourth decimal across their full
ranges. See [`SWEEP_ANALYSIS.md`](SWEEP_ANALYSIS.md).

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

That is the *training*-side switch. The *inference*-side one was closed two months earlier,
and for a different and larger reason — accuracy:

> Bug 1: Deliberation band `[0.4, 0.6]` was masking ~40% of predictions as
> `__unknown__` on binary problems. Fix: `deliberation_lo=1.0, deliberation_hi=0.0`
> (disabled). Effect: +23.5 pp on S1_2class_linear; regression R² −0.007 → 0.756.
> — `CHANGELOG.md:524-526`, under `[1.0.0] — 2026-05-30`

`bench/config/everyday_profile.json` still carries the measurement, as
`diagnostic_upgrade_notes.phase1_deliberation_penalty = {S1_2class: 0.118, R1_iris: 0.079,
S4_multimodal: 0.028}`. Note the band it indicts is `[0.4, 0.6]`, not the archived Python's
`[0.25, 0.40]` — the parameter had already been retuned once by then.

So there are **three** causes, not two, and they arrive in this order: the sweep demoted the
single-threshold formulation (2026-03); an accuracy diagnostic closed the inference-side switch
(2026-05-30); a cross-compiler floating-point concern closed the training-side one. They agree
in direction and should not be collapsed into one — least of all into the portability reason,
which is the smallest of the three.

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
is family B's differential-offset scheme, which arrived fully formed in v8:

```
Class k model  =  θ₀ ⊕ Δk          world prior + differential offset (natural parameters)
y* = argmax_k [ log p(h | θ₀ ⊕ Δk) ] + log p(k | context)

Attraction:  Δk += η · residual(h, θk)
Repulsion:   Δj −= η · wj · residual(h, θj)
MDL decay:   Δk *= (1 − λ)
World prior: θ₀ updated via Welford
```
— condensed from the module docstring, `archive/cypha-v8/Cypha.py:3-31`

**"NIG" names four different things in this project, and they are not even the same
distribution.** The acronym is worth pulling apart before the thread continues, because every
document in the archive uses it loosely and two of v8's own papers expand it differently —
`cypha_synthesis.md:12` as Normal-Inverse-**Gamma**, `cypha_stat_mech.md:140` as
Normal-Inverse-**Gaussian**. Both expansions are in use in the product, for different objects.

| | what it is | where |
|---|---|---|
| `NIGField` | a bank of four exponential decays with a learned `W_T` — **no NIG mathematics at all**; `alpha`, `beta`, `kappa` occur zero times in its 95 lines, and zero times in its C++ port | v8, root, `native/src/nig_field.cpp` |
| the Normal-Inverse-**Gaussian** world gate | a mixing-variable gate over GIG moments. Its `gig_e_inv_v_lam_neg1` evaluates GIG(λ = −1) — which *is* NIG: λ = −½ is the prior index, and the callers form the conjugate update `χ + innovation²/R` before evaluating, which takes λ to −1 | C++ only, no Python ancestor |
| Normal-Inverse-**Gamma**, as `NIGExpert` | the conjugate prior over (μ, σ²), fully parameterised `(κ₀, α₀, β₀)` with per-dimension `kappa_n_`, `alpha_n_`, `beta_n_` | `native/include/cypha/cyphalm/cyphalm_nig_expert.hpp:12,27-33` — the **language model**, not the classifier |
| Normal-Inverse-**Gamma**, as a posterior scale | `nig_delta_posterior_scale`, τ = `v_mean/(n_obs+1)` — the Normal half only, no α or β anywhere — inherited verbatim from v8's `u_k` (`archive/cypha-v8/Cypha.py:631`), but reached only when `CYPHA_USE_NIG_BMA` is set; `use_nig_bma` is `false` by default | `native/src/nig_gig_math.cpp:141-143` |

Only the last is continuous with the archive — and v8 names it: `Cypha.py:410` calls `WorldPrior`
"shared NIG base distribution θ₀". The Gaussian gate is native-only, so "arrived fully formed in v8"
would be false of it. See
[`eras/08-v8.md`](eras/08-v8.md#nigfield-does-not-contain-the-mathematics-its-name-claims).

`NIGField` appears in v8 as an EMA filter bank over multiple timescales, gains a τ = 0.99
group in the root monolith's "Phase 1", and is named three times in the header of the C++
file that implements it:

```cpp
/// Diagonal timescales τ (same grouping as Python `NIGField`).
/// `field_h += strength * (‖field_h‖/‖signal‖) * signal` with L2 cap (Python `NIGField.inject`).
/// Causal `W_T` SGD step + spectral-radius trim + refresh `a_eff` (Python `update_causal`).
```
— `native/include/cypha/nig_field.hpp:9,20,26`

Around it sit `nig_gig_math.hpp`, `nig_gig_score_match.hpp` and `bessel_table.hpp`. Those
three are **not** part of this continuity: across all 175 archived files — including the
`.docx` bodies, the PDF and the gzipped sweep — there is not one occurrence of *Bessel*, *GIG*,
*K₂/K₁*, *score matching*, *Hyvärinen*, *scale mixture*, *variance-mean*, *generalised
hyperbolic*, `r_eff`, `gh_chi` or `gh_psi`. The only "gig" strings anywhere in the archive are
`gigantic_moyo` and `multi-gigabyte`.

What the GIG layer inherits is a **slot, not a method**. The root monolith already computed a
`world_gate` from `mahal_per_dim` and multiplied it into `disc`
(`archive/root-monolith/Cypha.py:548-557`), as a plain sigmoid of a Mahalanobis margin. The C++
keeps all three names and the same composition and replaces only the body — a substitution its
own error message records, throwing `legacy sigmoid gate removed` at
`native/src/infer_cpu.cpp:1158`. The name `world_gate` does not appear in v8 at all; it enters
with the root monolith.

So `NIGField` is the strongest line of continuity in the project — and it is **three days long**
on the Python side, from v8 (2026-03-11) to the root monolith (2026-03-14) before the port;
seven if v8 is dated from its [earliest recovered stamp](TIMELINE.md#dating-the-archive) of
2026-03-07 rather than its directory.

Thirteen months of resonance mathematics produced the engineering culture, the single-core
constraint, the numpy floor and the measurement habits. Almost all of the mathematics that
ships as the *classifier core* was written in the last week and a half.

**Two exceptions.** `W_T`, the online-learned causal transition matrix, is introduced in v7
and crosses the restart intact — same rank-1 outer-product update in v8 and the root
monolith, shipping today as `nig_field_update_causal` with its spectral-radius trim (see
[`TIMELINE.md`](TIMELINE.md#but-the-break-is-in-the-class-structure-not-in-everything)). And
`GRIA` takes a stranger route still — thread 7.

---

### And the restart broke the mechanism it salvaged

`W_T` is the one piece of family A's mathematics that reached the C++ product. What did not
cross with it is the thing that made it meaningful.

**In v7 the target is real.** The update sits inside the resonance step, and regresses `W_T`
onto the state the FFT evolution actually produced:

```python
p = self._prev[:self.dim].real.astype(np.float64)
q = psi[:self.dim].real.astype(np.float64)
causal_pred = self._W_T @ p
causal_err  = q - causal_pred
```
— `archive/cypa-v7-generation/Cypha.py:2190-2193`

`p` is the previous field state, stored at the end of the last call; `q` is the current one,
produced by the split-step integrator. Neither is a function of `W_T`. The residual is a
genuine one-step prediction error, and the docstring's claim that this "makes `W_T` approximate
the local Jacobian of state dynamics" (`:2150`) is defensible.

**In v8 it is not.** The same rule is kept, the call convention is changed, and `W_T` ends up
regressed onto its own output.

The call site injects, snapshots, evolves, and then fits the snapshot to the evolved state:

```python
self.field.inject(self._to_field_dim(h), strength=0.05)
h_old = self.field.h
h_new = self.field.evolve(h_old, update_state=True)
if self._total_steps % 50 == 0:
    self.field.update_causal(h_old, h_new, lr=0.0002)
```
— `archive/root-monolith/Cypha.py:1814-1818`; identical in shape at
`archive/cypha-v8/Cypha.py:1161-1166` (every 10 steps) and
`native/src/train_step_vector.cpp:205-211`

But `evolve` applies `A_eff`, and `A_eff = diag(a) + W_T`. So the residual the update descends
on is

```
err = W_T·h − A_eff·h
    = W_T·h − (diag(a) + W_T)·h
    = −diag(a)·h
```

`W_T` cancels — which is exactly what did not happen in v7, where the target came from a
different operator entirely. Substituting into `W_new = W_T − lr·outer(err, h_t)/d` leaves a
pure Hebbian step, `ΔW_T = +(lr/d)·outer(a⊙h_t, h_t)` — an `a`-weighted autocorrelation of the
field's own state. Reproduced by executing the archived class directly: the actual residual
matches
`−diag(a)·h` to **7.8 × 10⁻⁸** relative in the Python path and **1.8 × 10⁻⁷** in a float32
reimplementation of the C++ arithmetic, and the closed-form ΔW_T matches to 8.9 × 10⁻⁸. The
"loss" the function returns, `err·err/d`, is therefore `‖a⊙h‖²/d` — a measure of field
magnitude, not of prediction error.

Because the target is generated by the very matrix being fitted, it carries no information
about any *future* state. This is a direct consequence of the restart: v8 deleted the FFT
evolution that had been producing v7's targets, and substituted its own linear `evolve` — which
contains `W_T`. The rule survived; the independent signal it was fitting against did not.

**Two things that would be easy to overstate, and are not true.** The update is not
signal-free: `inject()` runs immediately *before* the `h_old` snapshot, so the fresh encoder
latent is about **5%** of `‖h_old‖` at the shipped strength of 0.05, and the update is bilinear
in `h_t`, so it does depend on the data — two different injected signals of equal norm give
updates differing by 10.1%. And the identity `err = −diag(a)·h` is exact only while `evolve`'s
`‖h‖ ≤ 50` cap is slack; across a 3,000-step run the cap engaged on 864 steps and the deviation
reached 6.1 × 10⁻². The extra term is also self-generated, so neither changes the conclusion:
the rule is data-dependent but not causal.

**The matrices around it are never trained at all.** `update_field_map`, which would fit the
field→world coupling `F_field`, is defined at `archive/cypha-v8/Cypha.py:479` and
`archive/root-monolith/Cypha.py:364` and has **zero call sites** in either file or anywhere
under `native/`. The latent→field map `w_inject` has no training write site either: it is
created as zeros (`native/src/create_model.cpp:123`) and read back
(`native/src/infer_cpu.cpp:497-503`), with the only other write being an explicit seed in
`multilabel_dif.cpp:139-140`.

That makes a fresh C++ model's field inert by construction, and the source comment beside it
describes a bootstrap that cannot happen:

```cpp
// Zeros will give zero signal → field stays at rest until training updates W_T.
```
— `native/src/create_model.cpp:125`

With `w_inject` zero the signal is zero, and `nig_field_inject` returns early on `s_sq == 0.0`
(`native/src/nig_field.cpp:100-102`), so `field_h` never leaves zero — and `update_causal`,
bilinear in `h_t`, can never move `W_T` from it. Training cannot start the field, because the
thing that would start it is not what training updates. The path carries signal only for models
loaded from a Python-derived checkpoint or seeded through `initial_w_inject`.

So the full shape of the subsystem is: an untrained input map, feeding a filter bank whose
transition matrix is fitted to its own output, conditioning a prior through an untrained output
map. It is the strongest line of continuity in the project — and the continuity is of the
code, not of the mechanism. What v7 was doing, v8 stopped doing while keeping the lines that
did it.

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
