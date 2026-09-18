<!--
ARCHIVED. Superseded by repo-root PLAN.md (board) and README.md (how-to).
Do not add new experiments here. Findings may still be true; live queue is PLAN.md.
-->

# hp — Upgrade Paths

Roadmap for continuing this work in Cursor. Every item lists the target file,
the expected gain, and how to verify it. Gains are estimates from the
literature and from what the instrumentation here has measured; treat them as
priors, not promises.

**Ground rules that must not be broken:**

1. No `float`, `double`, `<cmath>` or transcendental calls in `include/` or
   `src/`. Run `test/no_float.sh` after every change.
2. Encoder and decoder must derive every piece of state from data both sides
   already have. Anything else desynchronises the coder.
3. Run `test/roundtrip.sh` and `test/determinism.sh` before accepting any
   change. A compression gain that breaks bit-exactness is worth nothing.

**Current state:** 646,888 B / 1.640 bpc on the 3.1 MB wiki-XML proxy
(dictionary off — it is now off by default, see 4.1).
Effective rank 3.34 of 28 experts. Bit-identical across 8 build
configurations. ~3.7 µs/byte, projecting to a few hours for enwik9 against a
50-hour budget — roughly 50x timing headroom still unspent.

---

## Tier 0 — do these first (highest value per hour)

### 0.1 Benchmark on real enwik8

Everything here is measured on a synthetic proxy. Real enwik8 will move the
numbers and may reverse some findings — the dictionary result in particular
is scale-dependent and was explicitly not validatable at 3 MB.

```bash
wget http://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip
build/hp c --mem 24 enwik8 e8.hp
```

Expect roughly 1.55–1.75 bpc. This is the number that makes the project real;
without it nothing is comparable to published results.

### 0.2 Finish the discovery pool sweep

`include/hp/discover.hpp` — already working (+298 B on a 1.2 MB slice once the
novelty constraint was added), but the parameters are unswept. The sweep was
started and cut short.

Sweep `kSlots` ∈ {12, 16, 24, 32} × `kEvalBytes` ∈ {256, 512, 1024, 2048}.
Note the first version of `fresh_mask()` was net-NEGATIVE because random masks
mostly reproduced contiguous prefixes already covered by orders 1–6. The
rejection rules (no contiguous-from-zero, must reach offset ≥ 4) are what made
it pay. Keep them; make them stricter if wider sweeps disappoint.

**This is the most important item in the file.** It is the mechanism that
replaces hand-tuned context sets with search, and it is the only component
that scales without human effort.

### 0.3 Memory scaling

`--mem 24` was consistently worth ~1.2% over `--mem 22` and the prize allows
10 GB. Current tables are nowhere near that. Raise `table_bits` and
`buf_bits`, measure the curve, find where it flattens. Cheapest gain available.

---

## Tier 1 — diversity (the binding constraint)

Effective rank is 3.34 of 28. Everything that raises it converts directly into
bits. Measure with:

```bash
build/dump_experts <corpus> /tmp/e.i16 37 400000
python3 tools/mp_rank.py /tmp/e.i16 <n_experts>
```

Note: Marchenko-Pastur does **not** apply at γ = p/n ≈ 4e-5 — the noise bulk
collapses and everything reads as signal. Use participation ratio and entropy
effective rank instead. `mp_rank.py` reports all three and flags this.

### 1.1 Negative Correlation Learning — NOT YET BUILT

Liu & Yao 1999. Add a correlation penalty so models actively push apart
instead of converging on the same suffix statistics:

```
err_i  ->  err_i + lambda * (p_i - p_ensemble)
```

In `models.hpp`, `ContextModel::update()` takes the ensemble probability and
modulates its StateMap update. Sweep `lambda` from small — too large and
models degrade individually faster than the ensemble gains.

**Caution:** "Joint Training of Deep Ensembles Fails Due to Learner Collusion"
(arXiv 2301.11323) reports that jointly trained ensembles can game a diversity
metric without gaining real diversity. Validate against effective rank AND
bpc, never rank alone.

This is the single largest untried item. It targets the constraint the
instrumentation identified.

### 1.2 Wiki-aware tag model

`tag:py` is the most decorrelated expert in the ensemble (mean |corr| 0.621),
and it is currently a generic XML depth counter. fx2-cmix uses hardcoded
enwik states — `WIKITABLE`, `SQUAREOPEN`, `HTLINK`, `VERTICALBAR`, and
`linkword = linkword*2104 + j`. Give the most decorrelated expert real
structure. See `src/models/fxcmv1.cpp` in the fx2-cmix repo.

### 1.3 Word-stream models

fx2-cmix runs 4 separate word streams with 10 sparse configurations over them.
`hp` has one word model and one word-bigram. This is a large, well-mapped gap.

### 1.5 Sequence Memoizer / hierarchical Pitman-Yor

`statemap.hpp` implements the PY *discount* only. The full nonparametric
version (Teh's Sequence Memoizer) is infinite-depth with a coagulation
property that collapses inference to a linear number of parameters. Reported
1.89 bpc on Calgary vs PPM 1.93, CTW 1.99. Elegant and unbounded-order, but
note it still loses to paq8 — take the estimator, not the whole architecture.

### 1.4 More match models

The bank of 5 (orders 3, 4, 6, 10, 16) was worth −0.64% and put four match
models in the top-six most-decorrelated. fx2-cmix runs 10, including five
keyed on **word** context rather than byte context. Word-keyed matching is not
implemented here at all.

---

## Tier 2 — mixing (where the measured redundancy is)

`--profile` reports model redundancy of roughly +475 KB against a best-expert
oracle. The oracle is unachievable (selecting it would itself cost bits), so
treat the magnitude as indicative, not as a target.

### 2.1 Per-mixer tuned learning rates

fx2-cmix runs 24 layer-1 mixers with individually tuned rates from 0.0003 to
0.005. `hp` has 6 mixers sharing one rate. Give each its own, sweep
independently. Cheap and mechanical.

### 2.2 More mixer gates

Current gates: partial byte, α bucket, previous byte, match length, source
entropy, Hebbian strength. fx2-cmix combines several signals into single
composite gates, e.g. byte-stream state × first-char index × word count ×
paragraph flag × match flag. Richer gates need the two-layer structure to
avoid dilution — which `hp` already has.

### 2.3 Hedge over layer-1 mixers, not raw experts

`hedge.hpp` currently averages the 28 raw experts, which are ~0.71 correlated,
so the sum-mixture nearly duplicates the product-mixture. The 6 layer-1 mixer
outputs are far more diverse. Expose them from `MixerNet` and hedge over those
instead. Untried, and theoretically the right target.

### 2.6 Rank coding under a model-sorted alphabet

The correct form of the "swap table" idea. Sort all 256 symbols by the model's
current predicted probability and code the RANK rather than the symbol. Rank
distributions are extremely skewed and nearly stationary across contexts —
rank-1 sits at 60-75% almost regardless of context — so one small shared
rank-coder pools statistics that otherwise fragment across millions of
contexts.

**Tension to be aware of:** this is a byte-level technique and `hp` is
binary-decomposed, which is what buys the integer determinism and the speed.
Adopting it means unwinding that. Prototype it separately before committing.

### 2.4 LSTM mixer

cmix's actual edge. Expensive, and the last item to attempt — but the timing
headroom exists.

### 2.5 CTW recursive weighting

A true "double mixture" over all tree sources, complementing the logit
product. `statemap.hpp` already stores the `(n0, n1)` counts a KT estimator
needs, so the recursion is implementable over the existing order chain.

---

## Tier 3 — estimation and memory

### 3.1 Richer bit-history states

`StateTable` caps counts at 20 (882 states, 16-bit). PAQ8/cmix use
hand-designed state machines with different discount schedules. Worth ~2–4%
historically.

### 3.2 Nibble buckets with checksums

Tables are directly indexed and collide freely. PAQ groups slots into
cache-line buckets with checksum bytes and an LRU policy. Worth ~2–3%, and it
improves cache behaviour, which matters at enwik9 scale.

### 3.4 Generalized context modelling

Extends context modelling from suffixes to arbitrary combinations of symbols
in multiple directions, with MDL-based model selection. This is the formal
version of what `discover.hpp` does heuristically — worth reading for the
selection criterion, which is more principled than the loss-ranking currently
used.

### 3.3 PPMD byte model

fx2-cmix runs an order-25 PPMD with a 14 GB budget as one expert. Structurally
different from context mixing, so it should decorrelate well.

---

## Tier 4 — preprocessing

### 4.1 Re-test the dictionary at scale

Measured here: the transform **won 1,498 B (0.23%)** while raw dictionary
storage cost 10,121 B, so it lost overall. At enwik9 scale a dictionary is
~0.3% of output rather than 1.5%. fx2-cmix ships a 412 KB `english.dic` and it
pays for itself.

Two fixes: (a) compress the dictionary before embedding — it is currently
stored raw and English wordlists compress ~2.5x; (b) re-test on enwik8/enwik9
where amortisation is real. `dict.hpp` already keeps the dictionary only when
it measurably pays, so this is safe to leave enabled.

### 4.3 English prior initialisation

Instead of starting every StateMap and APM curve at p = 0.5, initialise them
from English character statistics. Legal — a shipped prior counts in S1 but is
not "outside information" in the prohibited sense, and paq8hp set the
precedent with its custom dictionary.

Honest expectation: **small.** Adaptive models converge within a few hundred
KB, so over 1 GB the cold-start penalty amortises to roughly 0.1-0.3%. A few
KB of table for a fraction of a percent. Worth doing because it is nearly free,
not because it is large. The stronger version of this instinct is 4.1 —
dictionary substitution wins by extending *context reach*, which never
amortises away, rather than by fixing cold start, which does.

### 4.2 Article reordering

The `starlit` preprocessor clusters similar articles before compression so
match and context models see related content adjacently. Offline, two-pass,
and worth several percent. Structurally the largest preprocessing win left.

---

## Tier 5 — research instruments

### 5.1 Run the effective-rank diagnostic on Cypha

`tools/dump_experts.cpp` + `mp_rank.py` work on any expert ensemble. Point
them at Cypha's `AdaptivePredictorMixer`. If Cypha's experts also collapse to
~3 directions, that is a real finding about the flagship framework, and NCL
(1.1) is the fix there too.

### 5.2 Effective rank on fx2-cmix

Instrument their mixer inputs the same way. If their hundreds of models span
rank 30 while `hp` spans 3.34, that quantifies exactly how much
independent-direction headroom exists and where. This is the defensible
version of "reverse engineer the record holder."

### 5.4 Borrow from the MoE literature

Expert collapse is a known, heavily-engineered problem outside compression:

- **MoE load-balancing losses** (Switch Transformer, expert-choice routing) —
  the identical failure mode with years of production engineering behind it.
- **Product-of-Experts theory** (Hinton) — explains *why* logit mixing becomes
  overconfident on correlated experts, which is directly diagnostic of the
  model redundancy `--profile` reports.

### 5.3 Transformer-as-oracle

Run a small char-level LM alongside `hp` over the same bytes. Log positions
where the LM assigns high probability and `hp` assigns low. Cluster them. That
is a ranked list of structures `hp` is blind to — the model telling you what
to build, with no weights shipped. Requires network access to model hosts.

---

---

## Tier 6 — retrofitting to Cypha (LM space)

Compression and language modelling are the same objective: cross-entropy in
nats IS code length. A model at 1.0 bpc has cross-entropy 0.693 nats/char.
No translation step — same number, different units.

**What transfers cleanly:**

| from `hp` | to Cypha / LM |
|---|---|
| APM / SSE (`mixer.hpp`) | probability calibration — APM is contextual Platt scaling |
| effective rank (`tools/`) | MoE expert collapse detection |
| α as adaptive rate (`gria.hpp`) | entropy-driven learning-rate schedules |
| NCL (1.1) | expert diversity in MoE routing |
| redundancy profiler | attributing loss to estimation vs mixing |

**What does not transfer:** the context models themselves. They are hash
tables with zero generalisation to unseen contexts — precisely what a neural
net provides and a lookup table never will.

So the **mixing and calibration layer** ports; the **estimation layer** does
not. That is a good split, because the mixing layer is where the measured
redundancy lives.

Note `hp` already beats Cypha's 2.664 BPC lock by a wide margin, but on a
different task — `hp` is online-only lossless coding, Cypha's lock was a
trained LM benchmark. Do not compare the numbers directly without matching
the protocol.

---

## Findings this session (2026-08-22)

See repo-root `UPGRADES.md` and `RECORD.md` for the full log. Headline:
enwik8 first 1 MB = 236,657 B / 1.806 bpc, round-trip PASS; entropy rank
3.34 → 3.885; CTW rejected (duplicate of PY); B.3 NO-GO; `--mem` desync fixed.

---

## Findings worth preserving

These are measured, and two of them are publishable as negative results.

**α as a statistic beats α as an operator.** As a mixer gate context, GRIA's
α measured +0.08%, stable across four architectural rebuilds. Moved to driving
the fixed-share switching rate (`GriaGate::switch_rate_q16()`), it measured
**+0.25%**. An order parameter should modulate a rate, not select a bucket.

**α is a ratio, so it is partially regime-invariant.** Both terms of
`1 - H(f(X))/H(X)` co-move: markup has low source entropy *and* low achieved
cost. Before rescaling, 96% of bytes fell in 2 of 8 buckets. The components
carry more separable signal than the ratio.

**Sum-mixtures and product-mixtures converge when experts are correlated.**
The Hedge layer was net-negative at effective rank 3.00 and turned positive at
3.34, with identical code and η. Diversity must come before the regret
machinery pays. This was predicted before it was observed.

**Learned beats stored.** The static dictionary lost; the Hebbian associative
model (zero transmitted bytes, synapses potentiated identically on both sides)
landed in the most-decorrelated tier.

**Novelty constraints are what make automated discovery work.** Random context
masks were net-negative until candidates were forced away from contiguous
prefixes already covered by the fixed model set.

---

## Positioning

`hp` is lpaq-class. The Hutter Prize record is ~40% better. The best result
achieved by anyone — nncp v3.2, a 199M-parameter transformer — is
107,261,318 bytes total on enwik9, and is disqualified by the single-core and
50-hour constraints.

Realistic near-term target is enwik8 parity with paq8-class, not the prize.
The instruments here (redundancy profiler, effective-rank diagnostic,
determinism harness, automated context discovery) are the transferable assets
and are independently useful.
