# v7 — Generation, and the moment the architecture finally mattered

**Directory:** [`../archive/cypa-v7-generation/`](../archive/cypa-v7-generation/) *(the
misspelling is original)* · **mtime** 2026-03-06 ·
**`Cypha.py` 5,650 lines / 272,560 bytes / 53 classes** — the second-largest implementation
in the archive and the last of family A.

Five days before the restart, v7 is the HRNA line at its fullest extent. It is also the one
place in the archive where the dynamical architecture genuinely determines the output.

---

## Pure accretion

v7 is v6 plus a layer. Comparing class sets:

```
v6: 29 classes    v7: 53 classes    shared: 29    dropped: 0
```

**Nothing was removed.** The module docstring is byte-identical to v5's and v6's — the same
"Omega-2" header, the same declaration that all HRNA layers "are unchanged". Every piece of
machinery the previous eras accumulated, including the parts already known to be dead, is
still here.

The 24 additions fall into four groups:

| Group | Classes |
|---|---|
| **Generation** | `CyphaDecoder` |
| **Semantic / cognitive** | `RelationalEdge`, `RelationalEncoder`, `RelationalGraph`, `SchemaLibrary`, `AnalogicalReasoner`, `Episode`, `EpisodicMemory`, `GoalField`, `PredictiveProcessor` |
| **Runtime learning** | `TrainingThought`, `RuntimeThought`, `RuntimeLearner`, `LearningSchedule`, `TemporalConsistencyFilter` |
| **Conditioning & diagnostics** | `InputFilterChain`, `OnlineWhitener`, `GeometryTracker`, `DistributionTracker`, `AnchorAuditor`, `SignalBus`, `StatEngine`, `TemporalFieldHierarchy`, `GeometricUncertainty` |

The second group is a genuine cognitive-architecture layer — a relational graph over
concepts, a library of schemas, episodic memory with replay, a goal field, and an analogical
reasoner. This is the most ambitious the project ever got.

---

## `CyphaDecoder` — the first real gradient learning on the main line

Every earlier version learned by storing prototypes and nudging centroids. v7 adds an
actual trained read-out:

```python
class CyphaDecoder:
    """
    Learned read-out for the Cypha resonance field.

      - C: (vocab_size × state_dim) projection matrix
      - B_embed: (state_dim × vocab_size) token re-injection matrix
      - train_on_sequence(): online SGD on byte sequences via cross-entropy
      - generate(): full cognitive generation loop
    """
```
— `Cypha.py:5214`

It carries its own Adam-lite optimiser — `_mC`, `_vC`, `_mB`, `_vB`, `β₁=0.9`, `β₂=0.999`,
`ε=1e-8` — and a byte-level vocabulary of 256. This is the ancestor of the current product's
CyphaLM, which is still byte-level and still measured in bits per character.

### Why this version is different

Since v3, `forward()`'s output had been computed and thrown away; classification ran on raw
encoder features (see [`03-v3.md`](03-v3.md#the-core-is-computed-and-discarded)). v6 made it
structurally impossible to use, because `forward()` returned a 64- or 256-dimensional state
while `AnchorMemory` lived in 512-dimensional feature space.

`CyphaDecoder.generate()` breaks the pattern:

```python
# ── 1. Prime field via full forward pass ─────────────────────────────
cypha.field.reset(); cypha.res_level.reset()
cypha.assembly.reset(); cypha.module.reset(); cypha.global_l.reset()
prime_out = cypha.forward(prime_text, training=False)
h = prime_out['state'].astype(np.float64)   # 256-dim real state
```

and `h` then flows through the whole loop — biased by a schema attractor, projected by `C`
into token logits, sampled, and re-injected through `B_embed`, with
`W_T = cypha.recursive._W_T  # live causal matrix` stepping the state forward between tokens.

**The field state is the decoder's input.** For the first and only time in the archive, the
resonance field, the level hierarchy and the recursion operators are on the critical path to
the output. The docstring's claim — "All cognitive machinery from the parent Cypha instance
is used live during generation — nothing is bypassed or simplified" — is, for the generation
path, true.

Classification in v7 is unchanged and still bypasses all of it. The architecture became
load-bearing only when the task changed.

The generation loop's own documentation lists eight steps per token: goal-field injection,
κ-scaled temperature, a thought cascade when the top-2 logits are within `cascade_margin`,
schema attractor bias, a planning trajectory to avoid low-probability dead ends, episodic
surprise gating, the `W_T` state transition, and token re-injection. Sampling is top-k
(default 8) plus top-p (0.95); there is no beam search anywhere in the file.

---

## `W_T` — the mechanism that survives the restart

`W_T` is introduced here, and it is the one piece of v7 that reaches the product.

> `W_T` is a dim×dim transition matrix trained online via rank-1 Oja-style update:
> `W_T += lr * (psi - W_T @ psi_prev) ⊗ psi_prev`
> This makes `W_T` approximate the local Jacobian of state dynamics.
> `causal_error = ||psi - W_T @ psi_prev||` measures how surprising the current [state is]
> — `Cypha.py:2148-2151`

initialised as `np.eye(dim) * 0.01` with `lr = 5e-4`.

It appears in no earlier version — zero occurrences in v5 and v6 — and it is in v8 and the
root monolith with the same rank-1 outer-product update, written in error-descent form and
given a spectral-radius bound:

```python
err   = self._W_T @ h_t - h_target
W_new = self._W_T - lr * np.outer(err, h_t) / self.d
```
— `archive/cypha-v8/Cypha.py:857-858`

Today it is `nig_field_update_causal` in
[`native/include/cypha/nig_field.hpp`](../../../native/include/cypha/nig_field.hpp), whose
comment still reads *"Causal `W_T` SGD step + spectral-radius trim + refresh `a_eff` (Python
`update_causal`)"*.

So v8 did not start from nothing. It discarded every class name and kept this. See
[`../TIMELINE.md`](../TIMELINE.md#but-the-break-is-in-the-class-structure-not-in-everything).

---

## What it does not fix

v7 adds 24 classes and removes none, which means it inherits every dead path the line had
accumulated — the `EncoderParams`/`AdaptiveControlLoop` loop that no encoder reads, the
event scheduler that is never scheduled to, the classification route that ignores the
architecture. Five versions of known-inert machinery are still being carried, and a
cognitive layer has been built on top of them.

That is the context for what happens five days later. v7 is 5,650 lines; v8 is 1,412. The
restart is not a change of direction so much as an admission that the accumulated structure
had stopped paying for itself.

---

## The name

The directory is `Cypa v7 Generation` — missing the `h`. It is preserved as
`cypa-v7-generation/` in the archive because renaming it would erase the only evidence that
the typo was in the original. It is the only misspelled directory in the source zip.

---

## How to read this era

Two readings of v7 are both defensible, and the archive supports both.

The unkind one: it is the fifth consecutive version to add machinery to a core that had been
measured inert, culminating in an episodic-memory-and-analogical-reasoning layer on a
classifier that still retrieves by cosine distance over hashed n-grams.

The kind one, and the more interesting: generation is a different task, and for generation
the architecture works. A resonance field with a learned causal transition and a trained
byte-level read-out is a coherent sequence model, and `h` really is the thing being decoded.
v7 found the task the architecture was suited to — and then the project restarted around
classification anyway, keeping `W_T` and eventually reaching back for `GRIA`.

The current product ends up doing both: `cypha::Cypha` classifies, regresses, samples
latents **and** generates, and its living default is a hybrid sequence model measured in BPC.
v7 is where that combination was first attempted.

---

**Previous:** [`06-v6.md`](06-v6.md) · **Next:** `08-v8.md` — the restart.
