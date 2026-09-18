<!--
ARCHIVED architecture narrative (proxy-era bpc tables). Live champs and
protocol live in root README.md and PLAN.md. Tools: --profile,
dump_experts, rank.cpp.
-->

# hp — an integer-exact context-mixing compressor

A bit-reproducible lossless compressor built as a testbed for Hutter Prize
work, and as the measurement instrument for evaluating modelling ideas
cheaply before committing to a fifty-hour enwik9 run.

Author: Odin Loch

---

## What this is

`hp` is a context-mixing compressor in the PAQ lineage, written from scratch
with two properties that most research compressors lack:

1. **Zero floating point in the coding path.** Every table is built by exact
   integer arithmetic. Archives are byte-identical across compilers and
   optimisation levels — verified, not assumed.
2. **Built-in instrumentation.** A redundancy profiler and a spectral
   effective-rank diagnostic that say *where the bits are going* and *which
   proposed model would actually add information*, before you build it.

The second property is the point. The Hutter Prize field moves roughly 1% per
year; guessing which idea to implement is the expensive part.

---

## Results

Measured on a 3,155,252-byte MediaWiki-XML-like corpus with mixed regimes
(prose, markup, wiki tables, link farms, incompressible blobs).

| build | bytes | bpc |
|---|---:|---:|
| gzip -9 | 1,086,484 | 2.754 |
| xz -9e | 831,904 | 2.109 |
| bzip2 -9 | 798,174 | 2.023 |
| hp v1 (direct counters) | 672,784 | 1.705 |
| hp + indirect + Pitman-Yor | 659,719 | 1.672 |
| hp + decorrelated models | 652,584 | 1.654 |
| **hp + match bank + Hebbian + Hedge** | **646,888** | **1.640** |
| hp, mem 24 | ~639,000 | ~1.62 |

Throughput ≈ 3.7 µs/byte single-core, projecting to a few hours for enwik9
against the prize's 50-hour budget.

---

## Determinism

The Hutter Prize verifies by running *your* decompressor on *their* machine.
A single ULP of difference in a `libm` call changes one probability, changes
one arithmetic-coder interval, and desynchronises the decoder catastrophically.

`test/determinism.sh` builds the same source under eight configurations —
`-O0 -O1 -O2 -O3 -Os`, `-march=native`, `-funroll-loops`, and `-ffast-math` —
and verifies that all eight produce **byte-identical archives** and that every
build can decode every other build's archive.

```
PASS: all builds produce identical archives
PASS: full 8x8 cross-decode matrix
```

`-ffast-math` is included deliberately: it is a no-op here, and demonstrating
that is the point. `test/no_float.sh` enforces the property mechanically by
rejecting any `float`, `double`, `<cmath>` or transcendental call in
`include/` and `src/`.

---

## Architecture

Per bit, eight times per byte:

```
  experts  ->  2-layer gated mixer  ->  3-stage APM chain  ->  binary coder
```

**Experts (28).** Eleven hashed context models each emitting two opinions,
a bias model, a bank of five match models, and a Hebbian associative model.

- Contiguous orders 1, 2, 3, 4, 6
- Word model, two sparse (skip) contexts
- **Column model** — the byte directly above at the same column
- **Tag model** — XML nesting depth and enclosing tag name
- **Word-bigram model** — previous complete word to current word
- **Match bank** — five match models at context orders 3, 4, 6, 10, 16
- **Hebbian associative model** — see below

Everything after the sparse contexts was added because the spectral
diagnostic showed that more byte-suffix contexts would do nothing. Each
addition was chosen to read a different information axis, and each was
verified to raise effective rank before being kept.

**Estimation.** Each model stores a bit-history state (`statemap.hpp`), from
which two estimates are drawn: an *indirect* one via a StateMap shared across
all contexts (so sparse contexts inherit pooled statistics from dense ones),
and a *direct* Pitman-Yor discounted estimate using a backoff chain down the
order hierarchy. Worth −1.94% over plain adaptive counters, and the 2-byte
state replaced a 4-byte counter, so the same RAM buys twice the table.

**Mixing.** Two layers. Five layer-1 mixers each see all experts but are gated
on different contexts (partial byte, GRIA α bucket, previous byte, match
length, source entropy); a layer-2 mixer weights their opinions. Single-layer
gating was tried first and *lost* — a rich gate splits training data N ways
and dilution beat signal. Two layers fixes this because each layer-1 mixer
trains densely on its own small context set.

**Coding.** Binary decomposition, one bit at a time, so the alphabet is {0,1}
and the coder reduces to a 32-bit interval split with no softmax, no CDF scan
and no per-symbol allocation.

---

## Instrumentation

### Redundancy profiler (`--profile`)

Borrows CTW's decomposition of cumulative redundancy into coding, parameter
and model terms. Sample output:

```
total spent      659,719 B   1.672 bpc
  best-expert    194,314 B   0.492 bpc
  after mixer    669,901 B   1.698 bpc
model redundancy (mixer vs best expert):  +475,587 B
coding redundancy (APM+coder vs mixer):    -10,181 B
parameter share (sparse contexts):             45%
```

Caveat: the "best expert in hindsight per bit" oracle is unachievable, since
selecting it would itself cost bits. Treat it as a loose bound whose
*magnitude* is informative, not as a target. The negative coding term is the
APM chain recovering bits the raw mixer left on the table.

### Spectral effective rank (`tools/`)

`dump_experts` writes each expert's stretched opinion to a side file;
`mp_rank.py` analyses the ensemble covariance.

Marchenko-Pastur was the original motivation, but **it does not apply here**
and the tool says so: MP needs `gamma = p/n` bounded away from zero, and with
23 experts over 400k samples `gamma ~ 4e-5`, so the noise bulk collapses and
every eigenvalue registers as signal. Participation ratio and entropy
effective rank are the correct statistics in the `p << n` regime.

| | 17 experts | 23 experts | 28 experts |
|---|---:|---:|---:|
| top-1 variance share | 80.6% | 77.1% | 74.9% |
| participation ratio | 1.53 | 1.67 | 1.76 |
| entropy effective rank | 2.48 | 3.00 | 3.34 |
| mean pairwise \|corr\| | 0.745 | 0.725 | 0.711 |

Most decorrelated experts, current build: `tag:py` (0.621), `m16` (0.641),
`hebb` (0.641), `m10` (0.647), `m6` (0.648). Note that the long-order match
models and the Hebbian model occupy five of the top six slots — both were
added on the diagnostic's prediction and both landed where predicted.

**Seventeen byte-suffix experts spanned 2.5 effective dimensions.** That
diagnosis predicted that more suffix contexts would be worthless and that
structurally different information axes would pay. Adding the column, tag and
word-bigram models raised effective rank to 3.00 and cut output 1.08% —
mechanism identified, mechanism responded.

`tag:py` is the most decorrelated expert in the ensemble (mean |corr| 0.608),
which is where to expand next.

Caveat: participation ratio measures *linear* correlation, while the mixer
operates in stretch domain. Effective rank is a lower bound on true diversity,
so treat it as a compass rather than a proof.

---

## The GRIA ablation (negative result)

GRIA defines `alpha = 1 - H(f(X))/H(X)`. In a compressor this instantiates
exactly: `H(X)` is the order-0 entropy of a sliding window, `H(f(X))` is the
code length actually spent on it. `gria.hpp` computes both in integer
arithmetic and uses the quantised result as a mixer gate.

**As a mixer GATE: +0.08%**, stable across four successive architectural
rebuilds and effectively a dead end.

**As a STATISTIC driving an adaptive rate: +0.25%.** Moving alpha out of the
gate context and into the fixed-share switching rate of the Hedge layer
(`switch_rate_q16()` in `gria.hpp`) tripled its contribution. An order
parameter is a scalar that should modulate a rate, not a categorical that
selects a weight set. This is the single clearest methodological result in
the project.

The reason is structural and worth recording: **alpha is a ratio, and both of
its terms co-move across regimes.** Markup has low source entropy *and* low
achieved cost; prose has more of both. The normalisation that makes alpha a
clean order parameter is what makes it partially blind to regime shifts. The
raw bucket histogram showed 96% of bytes in two of eight buckets before
rescaling.

Rescaling to the observed dynamic range, adding a trajectory (delta-alpha)
term, and giving alpha its own undiluted layer-1 mixer got the sign reliably
positive — but the magnitude is an order of magnitude below the prize's 1%
minimum claim. The components (source entropy and achieved cost separately)
carry more separable signal than the ratio does.

Reproduce with `test/ablation.sh`.

---

## Build and run

```bash
make                       # -> build/hp, build/dump_experts
# or: cmake -S . -B build && cmake --build build

build/hp c input archive
build/hp d archive output
```

Options:

| flag | effect |
|---|---|
| `--mem N` | table bits per model (default 22; 24 is notably better) |
| `--lr N` | mixer learning rate (default 2) |
| `--no-gria` | pin the alpha gate to a constant (ablation baseline) |
| `--dict` | enable dictionary preprocessing (OFF by default — measured negative at 3 MB scale) |
| `--profile` | print the redundancy decomposition |

The archive header carries every flag affecting modelling, so `hp d` takes no
options — the decompressor is standalone, as the rules require.

---

## The Hebbian dictionary

`HebbianModel` in `models.hpp` learns word associations online: when word A is
followed by B, the synapse A->B is potentiated; competing targets decrement
it; a periodic global decay implements synaptic scaling so unreinforced links
fade. The strongest current association, bucketed by synaptic strength,
becomes a prediction context.

This is the direct inversion of the static dictionary below. That version
*built and shipped* a word table: the transform won 1,498 B and the 10,121 B
of stored table destroyed the gain. The Hebbian version costs **zero bytes to
transmit**, because encoder and decoder potentiate identical synapses from
data both sides already hold. It removes the exact term that killed the
original attempt.

It also lands among the most decorrelated experts in the ensemble
(mean |corr| 0.641), which is what a non-suffix information axis should do.

---

## Dictionary preprocessing: a measured failure

`dict.hpp` builds a word-substitution dictionary from the input, embeds it in
the archive, and tokenises. This is the classic Skibinski/paq8hp lever, worth
6–8% historically.

Here it *lost* 1.28%. The breakdown explains why:

```
dict ON : 668,342 total - 10,150 header = 658,192 coded body
dict OFF: 659,719 total -      29 header = 659,690 coded body
```

**The transform itself won 1,498 B (0.23%). Storing the dictionary cost
10,121 B.** On a 3 MB corpus a 10 KB dictionary is 1.5% of output; against
enwik9's ~150 MB output an amortised dictionary is ~0.3%, and the dictionary
is currently stored *raw* (English wordlists compress ~2.5x trivially). The
sign plausibly flips at scale. It cannot be validated on a small proxy, so it
ships disabled-by-default-on-loss: the encoder keeps the dictionary only if it
measurably pays.

---

## Layout

```
include/hp/
  int_math.hpp    squash/stretch, integer log2, no floating point
  coder.hpp       binary arithmetic coder (shared interval split)
  mixer.hpp       2-layer gated mixer network + APM
  statemap.hpp    bit-history states, StateMap, Pitman-Yor estimate
  models.hpp      hashed context models, match model
  gria.hpp        integer alpha order parameter as mixer gate
  predictor.hpp   assembles the stack
  profile.hpp     CTW-style redundancy decomposition
src/main.cpp     CLI, dictionary pipeline, archive format
tools/            dump_experts.cpp, mp_rank.py
test/             no_float.sh, roundtrip.sh, determinism.sh, ablation.sh
```

---

## Honest positioning

`hp` is lpaq-class. The Hutter Prize record is roughly 40% better, and the
best result achieved by anyone — Bellard's nncp v3.2, a 199M-parameter
transformer on a GPU — is better still but disqualified by the single-core and
time constraints.

This is not a prize submission and does not pretend to be. It is a clean,
verifiable, well-instrumented baseline, plus two reusable measurement tools
and one documented negative result.
