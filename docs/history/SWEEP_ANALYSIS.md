# The classification settings sweep (27,524 runs)

Analysis of [`archive/datasets/cypha_classification_settings.jsonl.gz`](archive/datasets/cypha_classification_settings.jsonl.gz)
— the largest single research artifact in the historical archive, and the one that most
directly explains a default still shipping in the native product.

The file arrived as a 20,243,385-byte JSONL. It is stored gzipped (445,917 bytes, 22.3×);
every figure below is reproducible from the compressed file with the
[recipe at the end](#reproducing-this-analysis).

---

## What it is

One JSON object per training run. Each record carries the configuration, the seed, and a
post-hoc diagnostic profile of the run:

| Field group | Keys |
|---|---|
| Identity | `hash` (config digest), `seed`, `tier` |
| Configuration | `cfg` — 5 or 41 knobs, see below |
| Outcome | `final_acc`, `peak_acc`, `fitness`, `step_to_80`, `step_to_90`, `n_steps`, `acc_slope`, `acc_trail` |
| Diagnosis | `archetype`, `arch_conf`, `ece`, `kl_drift`, `gini`, `dead`, `eff_dim` |
| Margin shape | `margin_mean`, `margin_std`, `margin_skew`, `margin_kurt` |
| Cost | `elapsed_s`, `trans_n`, `n_anchors` |

Totals: **27,524 runs**, **5,397 distinct configurations**, **1,070.8 core-hours**,
median run 133 s.

### The archetype labels are worse news than they look

The `archetype` field ties this corpus to the v6 profiling work. It is the taxonomy defined
by [`archive/cypha-v6/Profiling/cypha_profiler4_archetype.py`](archive/cypha-v6/Profiling/cypha_profiler4_archetype.py):

```
IF early_acc > 0.50 AND sigma_mean < 0.02                        → Type A
IF early_acc < 0.35 AND acc_slope > 0.001 AND sigma_slope < 0    → Type B
IF early_acc < 0.35 AND acc_slope < 0.002                        → Type C (slow or stalled)
```
— `cypha_profiler4_archetype.py:41-43`

Type A is the fast learner, Type B the slow-but-improving one, **Type C the stalled one**.
The profiler's ground-truth table assigns audio → A; text, image, video → B; and
`structured` → C with `rf_iq` → **"C (ceiling)"** (`cypha_profiler4_archetype.py:206-212`) —
so the sweep's two labels are already named here, one per Type-C failure mode.

Only two labels occur anywhere in these 27,524 runs: `C_mismatch` (23,582 runs, median
accuracy 0.8438) and `C_ceiling` (3,942 runs, median accuracy 0.8047). **Both are subtypes
of Type C.** Not one run in the corpus was diagnosed as a fast or a still-improving learner;
every configuration was classified as stalled, differing only in whether the diagnosis was
"wrong model for this data" or "hit a capacity ceiling".

That is the context in which the knob results below should be read. A grid search across a
plateau will find the best point on the plateau, and the marginal effects reported here are
consistent with exactly that.

---

## Two designs, not one

The corpus contains exactly two `cfg` shapes, and they are two different experiments.

### Stage 1 — an exhaustive full-factorial grid

26,250 runs over 5 knobs, **5 seeds per configuration**:

| Knob | Levels | Values |
|---|---|---|
| `temperature_init` | 6 | 0.8, 1.0, 1.2, 1.5, 2.0, 2.5 |
| `temperature_decay` | 5 | 0.93, 0.95, 0.97, 0.99, 1.0 |
| `dedup_threshold` | 7 | 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7 |
| `deliberate_thresh` | 5 | 0.2, 0.3, 0.4, 0.5, 0.6 |
| `post_trans_alpha` | 5 | 0.001, 0.005, 0.01, 0.02, 0.05 |

6 × 5 × 7 × 5 × 5 = **5,250 cells, and all 5,250 were visited** — a complete factorial
design with no sampling and no early stopping. 5,250 × 5 seeds = 26,250 runs.

Outcomes: `final_acc` min 0.5859 / median 0.8359 / max 0.9141; median `ece` 0.2964;
18,390 runs reached 80% accuracy, 9,218 reached 90%.

### Stage 2 — a 36-knob coordinate sweep

1,274 runs over 147 configurations with **7 seeds each**, spread across `tier` 2–6, using
a 41-key `cfg` of which 36 actually vary (4–6 levels each). Per-tier medians are identical
(`final_acc` 0.8516, `ece` 0.2436) across all five tiers, which is the signature of a
one-factor-at-a-time design around a fixed baseline rather than a joint search.

The knobs added at this stage name the whole machine, and are worth reading as an
inventory of what the system had grown by this point:

```
GAIN_A GAIN_K GAIN_RHO K_TARGET LAMBDA_LP          field gains and targets
field_dt field_gamma inject_strength res_gamma      NIG field dynamics
lvq_lr lvq_window ema_alpha_store consolidate_every memory / consolidation
anchor_cap cap_update_every dead_staleness          anchor lifecycle
deliberate_lo deliberate_hi                         deliberation, now a band
temp_up_factor temp_down_factor                     temperature, now bidirectional
rl_lr rl_snap_thresh rollback_drop rollback_window  rollback control
alpha_max alpha_vel_scale velocity_ema              adaptation rate control
uncertainty_ema cmap_ema tightness_ema              uncertainty tracking
replay_ratio n_loops_train t2_every acc_window_size training schedule
kl_drift_thresh fb_weight                           drift detection / feedback
```

Stage 2 reached a best mean `fitness` of 0.5803, against 0.6206 for the best stage-1
configuration. Taken at face value the broad five-knob grid found a better operating point
than the 36-knob coordinate sweep; the two stages may not share a fitness normalisation,
so this is reported as recorded rather than interpreted.

---

## Which architecture was swept

This corpus profiles the **family-A (HRNA) line**, not CyphaDIF. The tier-2 knobs give it
away: `K_TARGET`, `LAMBDA_LP`, `GAIN_K`, `GAIN_RHO` and `GAIN_A` are constants defined in
v3's header block —

```python
K_TARGET  = 0.5
LAMBDA_LP = 0.15
GAIN_K    = 0.05
GAIN_RHO  = 0.03
GAIN_A    = 0.04
```
— `archive/cypha-v3/Cypha.py:20-24`

— and they occur in v3, v4, v5, v6 and v7, and **zero times in v8 or the root monolith**.
The same applies to `anchor_cap`, `lvq_lr`, `lvq_window` and `dead_staleness`, which name the
v5/v6 `AnchorMemory` prototype store. The records also carry an `n_anchors` field.

So the sweep measured the predecessor architecture, and the highest `final_acc` it ever
reached anywhere is 0.9219. When v8 cites it — see
[`eras/08-v8.md`](eras/08-v8.md#the-source-is-annotated-with-the-sweep) — it is importing
constants selected against a different system. Five of the knobs it swept do not exist in
the code that quotes it.

---

## The result that mattered: two knobs were inert

Marginal mean `fitness` per level, stage 1 (each column averages 5,250 runs):

| Knob | Marginal means by level | Verdict |
|---|---|---|
| `dedup_threshold` | 0.4 → **0.4899**, 0.45 → 0.5248, 0.5 → 0.5489, 0.55 → 0.5745, 0.6 → **0.6032**, 0.65 → 0.6001, 0.7 → 0.5470 | strong, unimodal, optimum at 0.6 |
| `temperature_init` | 0.8 → 0.5508, 1.0 → 0.5509, 1.2 → 0.5518, 1.5 → 0.5537, 2.0 → 0.5561, 2.5 → 0.5696 | weak, monotone increasing |
| `temperature_decay` | 0.93 → 0.5526, 0.95 → 0.5537, 0.97 → 0.5515, 0.99 → 0.5591, 1.0 → 0.5605 | weak; best is 1.0, i.e. *no decay* |
| `deliberate_thresh` | 0.2 → 0.5555, 0.3 → 0.5555, 0.4 → 0.5555, 0.5 → 0.5556, 0.6 → 0.5553 | **inert** |
| `post_trans_alpha` | 0.001 → 0.5555, 0.005 → 0.5555, 0.01 → 0.5555, 0.02 → 0.5554, 0.05 → 0.5555 | **inert** |

`dedup_threshold` carries essentially the entire signal: moving it from 0.4 to 0.6 is worth
more than every other knob combined. `deliberate_thresh` and `post_trans_alpha` vary only
in the fourth decimal across their full ranges.

The sharpest demonstration is the leaderboard itself. The top five configurations are the
same point in the other four knobs, enumerated across all five `post_trans_alpha` levels,
and they agree to four decimal places on every outcome:

```
fit=0.6206 acc=0.8656 ece=0.1719  n=5  {dedup 0.6, delib 0.2, post_trans_alpha 0.001, decay 1.0, temp 2.5}
fit=0.6206 acc=0.8656 ece=0.1719  n=5  {dedup 0.6, delib 0.2, post_trans_alpha 0.005, decay 1.0, temp 2.5}
fit=0.6206 acc=0.8656 ece=0.1719  n=5  {dedup 0.6, delib 0.2, post_trans_alpha 0.010, decay 1.0, temp 2.5}
fit=0.6206 acc=0.8656 ece=0.1719  n=5  {dedup 0.6, delib 0.2, post_trans_alpha 0.020, decay 1.0, temp 2.5}
fit=0.6206 acc=0.8656 ece=0.1719  n=5  {dedup 0.6, delib 0.2, post_trans_alpha 0.050, decay 1.0, temp 2.5}
```

A parameter whose five settings produce identical accuracy, identical calibration error and
identical fitness across 25 runs is not influencing the model at all.

> **Ranking configurations, not runs.** Every leaderboard here ranks *configurations* by mean
> fitness over their 5 seeds. Ranking the 26,250 individual runs instead gives a different
> answer, and the difference is not cosmetic: the top 100 *configurations* have
> `dedup_threshold` 0.6 in 75 and 0.65 in 25, while the top 100 individual *runs* have 0.55 in
> 75 and 0.65 in 25 — 0.60 does not appear at all. With 5 seeds per cell, the
> configuration-level ranking is the meaningful one, and it is the one that agrees with both
> the marginal means above and the value that shipped. A reader ranking rows will reach the
> opposite conclusion about which threshold won.
>
> The same distinction explains `top_fit = 0.6206`, the figure recorded at
> `archive/cypha-v8/Cypha.py:889`. It is the best configuration mean; the best single run is
> 0.6571, and 3,506 individual runs exceed 0.6206.

---

## What happened to the inert knobs

`deliberate_thresh` — the single scalar gate on whether the classifier should "think
harder" about a sample — does not appear in stage 2. It is replaced there by a two-sided
hysteresis band, `deliberate_lo` and `deliberate_hi`.

That band is what survives. It was carried through the Python line and into the native
port, where it is still present and still named after its Python original:

| Current file | Line | Content |
|---|---|---|
| [`native/include/cypha/cypha.hpp`](../../native/include/cypha/cypha.hpp) | 34–35 | `double deliberation_lo{kDeliberationLoDefault};` / `_hi` |
| [`native/include/cypha/infer_cpu.hpp`](../../native/include/cypha/infer_cpu.hpp) | 19 | `/// Python `` CyphaDIF.deliberation_lo/hi `` defaults (disabled: lo >= hi).` |
| [`native/include/cypha/infer_cpu.hpp`](../../native/include/cypha/infer_cpu.hpp) | 129 | `/// Python `` deliberation_lo `` / `` deliberation_hi `` (defaults disable abstention).` |
| [`native/src/cypha.cpp`](../../native/src/cypha.cpp) | 441–442 | forwards both into the inference options |

`deliberation_lo` appears 43 times and `deliberation_hi` 42 times across `native/`, and the
behaviour is regression-pinned by
[`native/tests/regression/gh_infer_deliberation_golden.cpp`](../../native/tests/regression/gh_infer_deliberation_golden.cpp).

So the honest summary is not "deliberation was measured useless and deleted". It is:

1. the **single-threshold** formulation was measured useless, over a complete factorial grid;
2. it was **reformulated** as a hysteresis band rather than abandoned;
3. the band was **ported to C++** and is still tested;
4. and it ships **disabled by default** — the defaults satisfy `lo >= hi`, which the header
   comment states turns abstention off.

An expensive negative result therefore did not delete a feature; it demoted one from a
default to an opt-in, and that demotion is still visible in the shipping defaults.

---

## Reproducing this analysis

```bash
python3 - <<'PY'
import json, gzip, collections, statistics as st
p = "docs/history/archive/datasets/cypha_classification_settings.jsonl.gz"
recs = [json.loads(l) for l in gzip.open(p, 'rt', encoding='utf-8')]
t1 = [r for r in recs if len(r['cfg']) == 5]

print(len(recs), len({r['hash'] for r in recs}),
      sum(r['elapsed_s'] for r in recs) / 3600)

for k in ('temperature_init', 'temperature_decay', 'dedup_threshold',
          'deliberate_thresh', 'post_trans_alpha'):
    d = collections.defaultdict(list)
    for r in t1:
        d[r['cfg'][k]].append(r['fitness'])
    print(k, {lv: round(st.mean(v), 4) for lv, v in sorted(d.items())})
PY
```

---

## Related

- [`archive/cypha-v6/Profiling/`](archive/cypha-v6/Profiling/) — the four profilers that
  produced the `archetype` vocabulary used here
- [`TIMELINE.md`](TIMELINE.md) — where this sweep sits in the development sequence
- [`PYTHON_TO_CPP_BRIDGE.md`](PYTHON_TO_CPP_BRIDGE.md) — the full set of Python identifiers
  the native code still names
