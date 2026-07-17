# Optimality Phase 8 — Rao-Blackwellise sampling paths (2026-07-17)

**Build:** `native/build_opt_p8` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Audit `generation.hpp` / `replay_buffer.hpp` for MC sample averages replaceable by conditional expectations; did not touch `build_math`, `build_deff`, `BASELINE_*`, overnight.

## Verdict

| Acceptance | Result |
|------------|--------|
| Replace MC sample averages with RB conditional expectations | **NO-GO** — no MC-averaging estimators in scope files |
| Estimator variance does not increase | **N/A** — no production code changed |
| Generation quality / goldens | **Unchanged** — no numerics touched |
| CTest variance smoke | **Pass** — hypothetical MC paths show higher variance than RB baselines |

## Audit summary

Phase 8 targets Monte-Carlo **sample averages** (estimators of the form `(1/n) Σ f(X_i)`) that could be replaced by **conditional expectations given sufficient statistics** already stored in the model (`μ_k`, `v₀`, `inv_v`, `n_obs`, replay priority weights `w_cache_`, co-occurrence / transition counts).

After a full read of `native/include/cypha/generation.hpp`, `native/src/generation.cpp`, `native/include/cypha/replay_buffer.hpp`, and `native/src/replay_buffer.cpp`, **no such MC averaging loops exist**. The paths below are **direct stochastic samplers** (draw `h ~ p(h|·)`), **exact deterministic reductions**, or **exact weighted-selection algorithms** — not MC estimates of expectations built from averaged samples.

## File:line evidence (generation)

| Path | Location | What it does | Why not MC→RB |
|------|----------|--------------|---------------|
| `generate_class_gaussian` (i.i.d.) | `generation.cpp:178-196` | Single draw `h = μ_k + z·σ` | Produces samples; does not average samples to estimate `E[h]` |
| `generate_class_gaussian` (rejection) | `generation.cpp:199-238` | Draw `C` candidates; keep `argmax LLR_k` | Biased **selection**, not `(1/C)Σ h_i` |
| `generate_conditioned` | `generation.cpp:296-310` | Field-shifted Gaussian draw | Direct sample |
| `generate_langevin` | `generation.cpp:362-398` | Langevin MCMC chain | Markov chain sampler, not sample-mean estimator |
| `generate_boundary` | `generation.cpp:457-481` | Interpolated Gaussian + hyperplane projection | Direct sample + deterministic projection |
| `generate_ood` | `generation.cpp:504-561` | Draw candidates; sort by `max LLR` | Rank/select, no averaging |
| `generate_mdl_ball` | `generation.cpp:595-628` | Fisher–Rao ball uniform direction/magnitude | Direct sample on manifold |
| `generate_ancestral` | `generation.cpp:676-695` | `Cat(probs)` then `N(μ_k,v)` | Hierarchical **draw**, not `E[h]` via MC |
| `predict_next_probs` | `generation.cpp:701-734` | Closed form from `cooccur` / `mid_trans` | Already sufficient-stat exact; no sampling |
| `rollout` | `generation.cpp:767-821` | Per-step Gaussian draw + categorical transition | Autoregressive sampler |
| `generate_from_observation` | `generation.cpp:851-871` | Langevin from anchor `h_obs` | MCMC, not MC mean |
| `generate_retrieval_augmented` centroid | `generation.cpp:939-956` | `h_anchor = Σ w_j h_j / Σ w_j` | **Already RB/exact** weighted conditional mean given neighbor latents + softmax weights |
| MDL penalty in OOD scoring | `generation.cpp:537-540` | `u_k = v_mean/(n_obs[k]+1)` from counts | Sufficient-stat closed form, not sampled |

**Note:** `generation.hpp:63` references Langevin MCMC; that is iterative dynamics, not a Monte-Carlo **average** of i.i.d. samples.

## File:line evidence (replay)

| Path | Location | What it does | Why not MC→RB |
|------|----------|--------------|---------------|
| `ReplayBuffer::push` | `replay_buffer.cpp:24-62` | Store `(h,f,label)`; update `w_cache_[i] *= decay`; assign priority `|loss|+ε` | Writes sufficient stats; no averaging |
| `ReplayBuffer::sample` | `replay_buffer.cpp:64-124` | Efraimidis–Spirakis keys `log(u)/w`; `partial_sort` top-`n` | **Exact** weighted sampling without replacement; not an MC estimate of buffer statistics |
| `f_store_` | `replay_buffer.hpp:27`, `replay_buffer.cpp:57-58` | Stored but **never read** in `sample` | No estimator built from feature replay |

There is no KDE path, no `(1/n)Σ` replay-loss estimate, and no priority-replay gradient averaged over random minibatches inside these files (training callers consume sampled rows directly).

## Variance smoke (fixed seed)

`native/tools/rao_blackwell_p8_smoke.cpp` demonstrates that **hypothetical** MC paths would have strictly higher variance than the RB reductions already used (or available analytically):

| Estimator | Variance (smoke) | Notes |
|-----------|------------------|-------|
| MC mean of 32 Gaussian draws / coord | **> 0** | Would be replaced by `μ_k` (var 0 for mean functional) |
| RB `μ_k` given `(μ, v)` | **0** | Tier-1 RB target for class-conditional mean |
| MC “pick one neighbor” RAG proxy | **> 0** | Not in production code |
| RB weighted centroid (`generation.cpp:948-956`) | **0** | Already shipped |
| Production `ReplayBuffer::sample` | unchanged | Exact weighted draw; smoke checks row count only |

## Goldens

**Not regenerated.** No production numerics changed.

## Files touched

- `native/tools/rao_blackwell_p8_smoke.cpp` (new)
- `native/CMakeLists.txt`
- `docs/reports/OPTIMALITY_PHASE8_2026-07-17.md` (this report)

## Build / test

```
cmake -S native -B native/build_opt_p8 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_opt_p8 -j8 --target rao_blackwell_p8_smoke
ctest --test-dir native/build_opt_p8 -R "native_rao_blackwell_p8_smoke|native_generation_golden" --output-on-failure
```

## Follow-up (out of scope for this pass)

If future work adds MC estimates (e.g. averaging Langevin terminal states for `E[h|k]`, or KDE bandwidth from sample counts only), apply RB at insertion time using stored `(μ_k, v₀, n_obs, w_cache_)` before deleting the MC loop.
