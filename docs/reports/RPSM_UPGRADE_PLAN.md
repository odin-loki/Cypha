# RPSM upgrade plan (P4) — 2026-07-11

**Status:** Planning only. No implementation in this document's scope.
**Priority:** P4 in `docs/reports/DEV_PLAN_2026-07-11.md:154` ("RPSM Option A → Option B — RPSM @ 7.336 BPC vs hybrid 2.864 — biggest single BPC gap in the repo").
**Relationship to the live overnight run:** independent and read-only. This plan does not touch `native/build_math` (the live D17→D21→cell-sweep overnight run), `native/build_scale` (the parallel hidden-dim scratch experiment, see `docs/reports/HIDDEN_DIM_SCALE_PLAN.md`), or `bench/BASELINE_LOCK.json`. Every claim below was verified by reading source in place; nothing was built or run.

---

## 1. The claim being scoped

`bench/BASELINE_LOCK.json` currently pins:

| Section | BPC | Mode |
|---|---|---|
| `overnight_results` (D17 hybrid @ 300k) | **2.864** (`bench/BASELINE_LOCK.json:72`) | `hybrid_gria_lstm` |
| `rpsm_results` (D21 RPSM @ 300k) | **7.336** (`bench/BASELINE_LOCK.json:92`) | `rpsm` |

This is the largest raw BPC gap anywhere in the codebase (`docs/reports/DEV_PLAN_2026-07-11.md:110,154`). The roadmap documents describe two named upgrade tracks:

> "CyphaDIF matrix refactor (RPSM Option A) | Planned | Parity green; batched LLR; faster infer" / "RPSM sequence layer (Option B) | Planned | D17 BPC < 2.873" — `docs/RESEARCH_STATUS.md:389-390`
>
> "RPSM core fixes (spectral α, norm η, orthogonal init) | Planned | Forgetting ratio < 0.01; α ∈ [0.3, 0.6]" — `docs/RESEARCH_STATUS.md:393`
>
> "RPSM matrix refactor + CyphaLM sequence layer ... Status: Planned" — `docs/FUTURE.md:221-223`
>
> "RPSM Option A — CyphaDIF matrix refactor: unified Ψ_mu / Ψ_var state, batched LLR/GEMM, parity-validated... Leads to Option B" — `docs/native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md:65`

**This document's job, and its single most important finding:** these docs are stale. Reading the actual `native/` tree shows Option A and Option B are **not** one-line roadmap bullets waiting to be started — a real scaffold for both already exists, is compiled, is unit-tested, and is wired into the live `--mode rpsm` path that produced the 7.336 BPC number. The problem is not "nothing has been built yet." The problem is that **the scaffold has a specific, concretely-located, and very plausibly gap-explaining bug**: the RPSM output classifier is never trained. §4 below is the finding; §2–3 establish what exists today and what the specs originally asked for, so the gap between them is legible.

---

## 2. What already exists in `native/` (contradicts the "Planned" docs)

| Component | Files | What it does |
|---|---|---|
| Option A — unified Ψ state + batched LLR GEMM | `native/include/cypha/rpsm/psi_matrices.hpp`, `native/src/rpsm/psi_matrices.cpp` | `PsiMatrices{mu, inv_var, counts}` (row 0 = world prior, rows 1..K = class deltas) and `batched_llr_gemm(...)`, replacing the per-class LLR loop with one batched multiply — this is **exactly** the "unified Ψ_mu/Ψ_var, batched LLR/GEMM" Option A asks for (`RPSM_COMBINED_SPEC.md:34-61`) |
| Option A wired live (classification path) | `native/src/infer_cpu.cpp:33-36,620-621,632-635,692-703` | `score_matrix_use_field` dispatches to `rpsm_score_matrix_batched` (which calls `build_psi_from_model` + `batched_llr_gemm`) **by default** — gated by `CYPHA_USE_RPSM_LLR`, which defaults to `true` (`infer_cpu.cpp:33-36`). This is the general `CyphaInferModel` classification/DIF scoring path (D01–D16-style domains), not the CyphaLM RPSM mode — but it means **Option A's core GEMM kernel is already shipped and default-on** for that path, parity-tested against the old per-class loop |
| Option A parity test | `native/tests/parity/rpsm_batched_llr_smoke.cpp`, CTest `native_rpsm_batched_llr_smoke` (`native/CMakeLists.txt:570-574`) | Exactly the `native_batched_llr_parity` check the spec calls for (`RPSM_COMBINED_SPEC.md:67`), just named differently — compares `score_matrix_use_field` vs `rpsm_score_matrix_batched` to `1e-12` |
| Option B — RPSM sequence layer scaffold | `native/include/cypha/rpsm/rpsm_sequence_layer.hpp`, `native/src/rpsm/rpsm_sequence_layer.cpp` | `RpsmSequenceLayer`: multi-level hierarchy (`h_levels_`), `W_up`/`W_down = W_up^T`, GRIA-style error gate, `RpsmGlobalMemory` (M_slots, soft-attention read + surprise-gated ring write), Izaac activation-mix selector (`IzaacActivationMix`, seed-selected) — a real, non-trivial implementation of most of `RPSM_IMPLEMENTATION.md`'s 11-step forward pass |
| Option B wired live (CyphaLM) | `native/src/cyphalm/cyphalm_model.hpp:36,179,181`, `native/src/cyphalm/cyphalm_model.cpp:47-49,328-338,767-774,826-827,1151-1155,1379-1468` | `CyphaLMModel` owns an `RpsmSequenceLayer` whenever `uses_rpsm(cfg)` is true (`use_rpsm_layer` or `context_mode==Rpsm`); `--mode rpsm` (`BenchMode::Rpsm`, `cyphalm_config.cpp:95-101`) sets `context_mode=Rpsm` and is exactly the D21 production path that produced BPC 7.336 |
| Option B parity/smoke tests | `native/tests/parity/rpsm_sequence_smoke.cpp`, `rpsm_hierarchy_smoke.cpp`, `rpsm_train_smoke.cpp`; CTests `native_rpsm_sequence_smoke`, `native_rpsm_hierarchy_smoke`, `native_rpsm_train_smoke` (`native/CMakeLists.txt:576-586`) | Forward-pass shape/NaN checks, W_up/M_slots write checks, and a 20-step SGD loss-decrease check — **all pass today** (implicit; these are part of the 115-CTest gate, `docs/native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md:14-17`). §4.4 explains why passing this suite is compatible with the bug found below |

**So: Option A's GEMM kernel is shipped and default-on (for the general classification path); Option B's hierarchy/memory scaffold is shipped and is exactly what `--mode rpsm` runs.** What's missing is not "the architecture" — it's specific pieces of the *training* loop, found in §4.

---

## 3. What the specs originally asked for (full context, not just the one-liners)

### 3.1 Option A — CyphaDIF matrix refactor (`docs/research/upgrades/RPSM_COMBINED_SPEC.md:34-71`)

Maps existing CyphaDIF components onto unified matrices without changing behaviour: `WorldPrior θ₀` → row 0 of `Ψ_mu`/`Ψ_var`; `ClassDifferential Δk` → rows 1..K; per-class LLR loop → one batched matmul; `TieredContextBuffer` → a separate 3-row context matrix; `PriorityReplayBuffer` → the `M_slots` working-memory matrix. Two required parity tests: batched LLR vs per-class loop, batched update vs original. **Confirmed shipped for the LLR half (`native_rpsm_batched_llr_smoke`); the "batched update" half (`native_batched_update_parity`) was not found — searched for that name and no such test exists.** The `TieredContextBuffer`/context-matrix mapping was also not found wired for the LM path — see §4.5.

### 3.2 Option B — RPSM sequence layer (`docs/research/upgrades/RPSM_IMPLEMENTATION.md`, full read)

Core dynamics: `dΨ/dt = η(H_up − H_down)(1−A) W_u + α(Ψ_past − Ψ) + βM − λΨ/‖Ψ‖_F` (`RPSM_IMPLEMENTATION.md:15`) — four terms: prediction-error update, Izaac memory blend, working-memory injection, MDL regularisation. Five "critical fixes" the spec explicitly says must be checked before scaling (`:35-69`):

| Fix | Spec requirement | Status in `rpsm_sequence_layer.cpp` |
|---|---|---|
| **1 — Spectral α** (critical) | `gria_alpha_spectral(Psi)` computed from the singular-value spectrum of Ψ; target α ∈ [0.3, 0.6] at edge-of-chaos init | **Not implemented.** `hierarchy_update()` uses a fixed scalar `cfg_.alpha_carry` (default `0.5`, `rpsm_sequence_layer.hpp:109`) as the blend weight — never computed from Ψ's spectrum, never exposed to a live "is it in [0.3, 0.6]" check |
| **2 — Normalised η** (critical) | `eta = eta_base / (‖E_gated‖_F + eps)` — fixed η caused 82% forgetting in the original research | **Not implemented.** `train_step` uses the raw `lr` parameter (`cfg_.rpsm_lr`, default `0.01`) directly for every SGD update (`rpsm_sequence_layer.cpp:462-475`) — no normalisation by the gated-error norm anywhere in the file |
| **3 — Orthogonal init** (high) | QR orthogonal init for `W_up`/`W_update` (Xavier on square matrices → κ≈450 spectral condition number) | **Implemented, but conditionally.** `init_orthogonal_matrix` (`rpsm_sequence_layer.cpp:42-72`, Gram-Schmidt QR) is used for `w_up_` only when `cfg_.use_izaac_init` is true. That flag **is** set true for `context_mode==Rpsm` (`cyphalm_model.cpp:335`), so **D21 production does get orthogonal `W_up` init** — this fix is live. `w_enc_`/`w_carry_` (non-square) correctly stay at normal init, matching the spec's "square matrices" scope |
| **4 — Multi-level input injection** (high) | Inject input at every level with diminishing scale `1/(l+1)` (input at level 0 only caused 87.5% error at bottom) | **Implemented.** `inject_input_multilevel` (`rpsm_sequence_layer.cpp:302-319`) loops all levels with `scale = 1.0/(l+1)` exactly as specified |
| **5 — Symmetric `W_down`** (medium) | `W_down = W_up^T`, recomputed after every step, never a separate parameter | **Implemented.** `hierarchy_update()` computes `H_down` via `matvec_transpose_row_major(w_up_, ...)` (`:353`) — there is no `w_down_` member at all, so the property holds trivially by construction |

**So the "RPSM core fixes" bullet in `docs/RESEARCH_STATUS.md:393` is 2-of-5 done, not 0-of-5 as "Planned" implies** — Fixes 3–5 are live; Fixes 1–2 (both marked "critical" by the spec's own priority label) are the real gap in that specific track, and are scoped as Phase -1 below.

### 3.3 Recommended scale tiers (`RPSM_IMPLEMENTATION.md:97-102`, `RPSM_COMBINED_SPEC.md:117-124`)

| Config | feat_dim | state_dim (D) | n_levels (L) | n_memory_slots (K_mem) | Use case |
|---|---|---|---|---|---|
| Tiny | 64 | 128 | 4 | 32 | **Unit tests** |
| Small | 128 | 256 | 8 | 64 | **BPC vs D17 benchmarks** |
| Medium | 256 | 512 | 16 | 128 | CyphaLM integration |
| Large | 512 | 1024 | 32 | 256 | Full LM |

The spec's own verification order (`RPSM_IMPLEMENTATION.md:108-118`) says explicitly: verify α/forgetting/spectral-radius/shapes at **Tiny**, then "Small BPC < char-LSTM 2.979" and "Small BPC < hybrid_gria_lstm 2.873 ← stop/go gate" — i.e. **the spec never expects Tiny to beat the hybrid baseline; Small is the first tier where that comparison is meant to be meaningful.** §4.3 checks which tier the actual D21 production run used.

---

## 4. Diagnosis: what's actually wrong (four findings, ranked by confidence and expected impact)

### 4.1 Finding #1 (highest confidence, highest expected impact): the RPSM output classifier is never trained

`RpsmSequenceLayer`'s class-discrimination logits come from `batched_llr_gemm` (`psi_matrices.cpp:46-85`), which computes, for each class `k`:

```
llr_k = Σ_j inv_var[j] · delta_k[j] · (h[j] − mu0[j] − 0.5·delta_k[j]) − u_k + ctx_k
```

where `delta_k` is row `k` of `Ψ_mu` (`psi.mu[(1+k)*d .. (1+k)*d+d]`) — this is the model's entire per-class discriminative direction, structurally equivalent to one row of the LSTM's output weight matrix `Wy` in the hybrid path (`char_lstm.hpp`; `Wy` shape `vocab × hidden`).

`Ψ_mu` rows 1..K are set **once**, at construction, to `Normal(0, 0.05)` noise:

```92:98:native/src/rpsm/rpsm_sequence_layer.cpp
  std::mt19937_64 rng(matrix_seed(cfg, 0));
  std::normal_distribution<double> nd(0.0, 0.05);
  for (int j = 0; j < d; ++j) {
    psi.mu[static_cast<std::size_t>(j)] = nd(rng);
  }
  for (int c = 0; c < k; ++c) {
    for (int j = 0; j < d; ++j) {
      psi.mu[static_cast<std::size_t>((1 + c) * d + j)] = nd(rng);
    }
  }
```

I grepped every write to `psi_.mu`/`psi.mu` in `native/` (`rpsm_sequence_layer.cpp:86,93,97`, `psi_matrices.cpp:16,22,34,40` — the latter file is the *other*, classification-only Option A path, §2). **Inside `RpsmSequenceLayer::train_step` (`rpsm_sequence_layer.cpp:401-486`), `psi_.mu` is never written.** The only per-step update to `psi_` at all is `psi_.counts[tgt] += 1.0` (`:482`), which feeds solely into the frequency-prior bias term `u_k = v_mean/(n_k+1)` — it never touches the discriminative direction `delta_k`.

**Contrast with the hybrid path's output layer, which is fully learned every step:**

```215-231:native/src/cyphalm/char_lstm.cpp
  out.dWy.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  out.dby.assign(static_cast<std::size_t>(vocab_size), 0.0);
  out.dh_prev.assign(static_cast<std::size_t>(hidden), 0.0);
  ...
  outer_rowmajor(d_logits.data(), vocab_size, cache.h_new.data(), hidden, out.dWy.data());
  out.dby = d_logits;
```
```409-412:native/src/cyphalm/char_lstm.cpp
  for (std::size_t i = 0; i < Wh.size(); ++i) Wh[i] -= lr * grads.dWh[i];
  for (std::size_t i = 0; i < b.size(); ++i) b[i] -= lr * grads.db[i];
  for (std::size_t i = 0; i < Wy.size(); ++i) Wy[i] -= lr * grads.dWy[i];
  for (std::size_t i = 0; i < by.size(); ++i) by[i] -= lr * grads.dby[i];
```

`Wy` (vocab×hidden = 256×128 for D17) gets a full outer-product gradient (`dWy = d_logits ⊗ h`) and a real update **every single training step**. RPSM's structural analogue (`Ψ_mu` rows 1..K, 256×64 for the current D21 config) gets **zero** updates, ever. The encoder (`w_enc_`/`w_carry_`, which *is* trained — see Finding #3 for why even that training is compromised) can only rotate/scale the feature vector `h` to align better with whichever fixed, tiny-magnitude (`σ=0.05`), mutually-uncorrelated random direction each of the 256 vocab classes happened to get at init. This is architecturally closer to nearest-centroid classification against **frozen random centroids** than to a trained classifier, and is a fully sufficient explanation, on its own, for most of the 7.336-vs-2.864 gap — no bigger refactor is needed to test this hypothesis.

**Why the LLR math makes the fix concrete and cheap:** expanding the bias/dot-product terms in `batched_llr_gemm`, `llr_k` is differentiable in closed form w.r.t. `delta_k[j]`:

```
∂llr_k/∂delta_k[j] = inv_var[j] · (h[j] − mu0[j] − delta_k[j])
```

which is exactly the same "prediction-error times precision" form already used for the *encoder*-side gradient two lines below in `train_step` (`enc_grad_[j] += gc * psi_.inv_var[j] * delta[j]`, `rpsm_sequence_layer.cpp:441-447`) — the codebase already computes half of what's needed (`gc = grad_logits[c]`) in the very function that's missing the `delta_k` update. This is a small, local addition to `RpsmSequenceLayer::train_step`, not a new subsystem.

### 4.2 Finding #2 (high confidence, moderate-to-high impact): `rpsm_embed_backprop_stub` is not a no-op, but it is not a valid gradient either

The task asked to check specifically whether `rpsm_embed_backprop_stub` is a no-op. It is not — it performs a real numeric update every step — but the update is dimensionally and mathematically unsound, which is arguably worse than a no-op because it perturbs the embedding table with a signal that is not derived from the loss via a correct chain rule:

```1379-1395:native/src/cyphalm/cyphalm_model.cpp
void CyphaLMModel::rpsm_embed_backprop_stub(std::uint32_t token_id) {
    if (!embed_ || !rpsm_layer_) return;
    const auto& field_grad = rpsm_layer_->input_grad();
    if (field_grad.empty()) return;

    const double lr = cfg_.rpsm_lr * 0.1;
    auto& table = embed_->table();
    const std::uint32_t de = embed_->dim();
    if (token_id >= embed_->vocab_size()) return;
    double* row = table.data() + static_cast<std::size_t>(token_id) * de;

    // Stub: map RPSM input gradient onto the leading embed dims (full chain deferred).
    const int n = std::min(static_cast<int>(de), static_cast<int>(field_grad.size()));
    for (int i = 0; i < n; ++i) {
        row[static_cast<std::size_t>(i)] -= lr * field_grad[static_cast<std::size_t>(i)];
    }
}
```

`rpsm_layer_->input_grad()` is the gradient of the RPSM loss with respect to the layer's own `input` argument — i.e. gradient in **`field_x_`-space** (`field_dim=160`, computed via `project_field`, `cyphalm_model.cpp:550-552,1411`; the vector itself is truncated in-layer to the first `state_dim=128` dims by `RpsmSequenceLayer`'s own loop bounds — see Finding #3). The comment ("map ... onto the leading embed dims") is accurate about what the code does and honest about it being incomplete ("full chain deferred"). The correct chain would be: `field_grad` → transpose-multiply through `proj_ssm_` (the `field_dim × ssm_context_dim` matrix from `project_field`) → gradient w.r.t. SSM context `ctx` → backprop through `ssm_step` → gradient w.r.t. embedding vector `e` (`d_embed=64`) → subtract from the embedding row. The stub skips all of that and directly subtracts the first `min(64,128)=64` raw elements of the **field-space** gradient from the **embedding-space** row, as if the two 64-length vectors occupied the same basis. They don't — `field_x_` is a learned linear projection of the SSM's recurrent state, not a re-arrangement of the embedding table.

Two concrete downstream consequences:
1. **`proj_ssm_` (the projection matrix feeding the RPSM layer) never receives a gradient in RPSM mode** — it stays at its random init (`init_proj_from_rng`, `cyphalm_model.cpp:219`, scale `0.02`) for the entire 300k-step run. (This matches the hybrid path's own default of `train_ssm=false` — `cyphalm_d17_wikitext.json:37` — so an *un-trained* SSM/projection is not unique to RPSM; but the hybrid path's LSTM head does not depend on `proj_ssm_` for its own output layer, so an untrained `proj_ssm_` doesn't block the hybrid's fully-learned output classifier the way it compounds with Finding #1 for RPSM.)
2. **The embedding table update is not a valid gradient step.** It may be weakly directionally useful by coincidence (both vectors are influenced by the same target token, so their signs are not independent), but it is not the true ∂loss/∂embedding, so it is at best a noisy/biased nudge and at worst actively counter-productive over 300k steps.

**Priority relative to Finding #1:** lower confidence of standalone impact, because `embed_` in RPSM mode is *also* fed forward through the SSM into `field_x_` into the (currently frozen) `Ψ_mu` classifier — so even a perfectly correct embedding gradient could only rearrange inputs into a classifier that still can't discriminate 256 classes with frozen random weights. **Fix Finding #1 first; re-measure; only then decide whether Finding #2 is worth the larger effort of wiring a real backward pass through `project_field`/`ssm_step`.**

### 4.3 Finding #3 (confirmed, moderate impact): D21 production ran the "unit test" tier, not the "BPC benchmark" tier, and there is no config path to change several of the RPSM hyperparameters at all

`bench/config/profiles/cyphalm_d21_rpsm.json:33-35` sets `rpsm_n_levels=4`, `rpsm_state_dim=128`, `rpsm_feat_dim=64` — this is **exactly the "Tiny (unit tests)" tier** from `RPSM_IMPLEMENTATION.md:99` (`feat_dim=64, D=128, L=4, K_mem=32`), not the "Small (BPC vs D17 benchmarks)" tier the spec itself designates for this exact comparison (`feat_dim=128, D=256, L=8, K_mem=64`, `RPSM_IMPLEMENTATION.md:100`). This is redundant with, not overridden by, `apply_bench_mode(BenchMode::Rpsm, cfg)` (`cyphalm_config.cpp:95-101`), which hard-codes the identical Tiny values regardless of what a profile sets — `apply_bench_mode` runs *after* `apply_bench_profile` in `cyphalm_bench_native.cpp:255,270`, so even if the JSON profile were changed to Small-tier values, **`--mode rpsm` would silently reset them back to Tiny** unless `apply_bench_mode`'s hard-coded block is also changed. Both layers currently agree on Tiny, so this has not yet been directly observed as a live bug, but it is a second, independent reason the config can't reach Small tier without a code change, not just a JSON edit.

Separately, `n_memory_slots`, `alpha_carry`, `beta_memory`, `hierarchy_loss_weight`, and `surprise_threshold` (`RpsmSequenceConfig`, `rpsm_sequence_layer.hpp:99-127`) have **no corresponding field in `CyphaLMConfig`** (`cyphalm_config.hpp:146-151` exposes only `use_rpsm_layer`/`rpsm_n_levels`/`rpsm_state_dim`/`rpsm_feat_dim`/`rpsm_lr`) — they are silently pinned at their `RpsmSequenceConfig` struct defaults (`32`/`0.5`/`0.1`/`0.1`/`0.05` respectively) on every run, with no CLI or JSON-profile path to change them, even experimentally. This directly matches the Tiny-tier `K_mem=32`, so it's not obviously "wrong" today by coincidence — but it means nobody can currently test "Small tier RPSM at 300k" without a code change first (Phase 1 below).

### 4.4 Why the existing test suite didn't catch Findings #1–#2

`rpsm_train_smoke.cpp` (`native/tests/parity/rpsm_train_smoke.cpp:27`, `kTarget=3`) trains on a **single, fixed target class for all 20 steps**. With a frozen classifier, the encoder can still legitimately drive the loss down over 20 steps by rotating its feature vector `h` to align with class 3's fixed random `delta_3` direction while moving away from the other 255 (mostly-orthogonal, high-dimensional) random directions — that requires no update to `Ψ_mu` at all. **A single-fixed-target smoke test cannot distinguish "the classifier is learning class boundaries" from "the encoder is just pointing at one already-fixed random direction,"** so it is expected to pass regardless of Finding #1, and did. This is not a flaw in the smoke test's purpose (it correctly checks "does the forward/backward pass run and reduce a loss without NaN/shape bugs") — it simply was never designed to catch a frozen-classifier bug, and a multi-class, changing-target smoke test would be a good addition alongside the Phase 0 fix (§6).

### 4.5 Finding #4 (minor, not blocking): the LM-side RPSM never uses a context prior or the general CyphaDIF online mu/inv_var update mechanism

`batched_llr_gemm`'s `ctx` parameter (Tier-1 context prior, mapped from `TieredContextBuffer` in `RPSM_COMBINED_SPEC.md:46`) is passed as `nullptr` at both LM call sites (`rpsm_sequence_layer.cpp:389,413`) — the LM path never supplies a context prior, unlike the general classification path (`infer_cpu.cpp:701`, `context_prior_for_labels`). Separately, the general `CyphaInferModel`'s `mu_world`/`D`/`inv_v` **are** updated online during normal DIF/GNG training (`native/src/memory_train.cpp:709,717`, `native/src/infer_cpu.cpp:352`) — this is the "correct pattern" that Finding #1's fix should mirror, but `RpsmSequenceLayer` builds its own from-scratch `PsiMatrices` (`init_psi_matrices`, `rpsm_sequence_layer.cpp:81-100`) entirely independent of `CyphaInferModel`, so it never benefits from that existing, already-proven update mechanism. Low priority relative to Findings #1–#3, but worth noting for whoever implements the Phase 0 fix: the update rule to add is not a novel derivation, it's porting an existing, working pattern from a different file into this one.

---

## 5. Is this "the stub" the task asked about, and does fixing it alone close most of the gap?

The task's hypothesis was that `rpsm_embed_backprop_stub` alone might explain most of the gap. Having read what it actually does (§4.2): **it is a real, partial, dimensionally-broken implementation, not a no-op — but it is very unlikely to be the dominant factor.** The dominant factor is Finding #1 (frozen `Ψ_mu`), which is a different piece of code entirely (`RpsmSequenceLayer::train_step`, not `CyphaLMModel::rpsm_embed_backprop_stub`) and has a much larger, more mechanically obvious blocking effect: no matter how good the upstream features get, the final 256-way classification step multiplies them against parameters that are literally never touched by training. Fixing Finding #1 is also cheaper and more self-contained (one function, no cross-module chain-rule work) than correctly fixing Finding #2 (which needs a real backward pass through `project_field` and optionally `ssm_step`). **Recommendation: fix Finding #1 first in isolation, measure, then decide whether Finding #2 is worth doing next** — see the phased plan below.

---

## 6. Proposed phased plan

| Phase | Goal | Scope | Est. cost | Depends on |
|---|---|---|---|---|
| **-1** | RPSM core fixes: spectral α (Fix 1), normalised η (Fix 2) | Add `gria_alpha_spectral(Psi)` (top singular value of Ψ via power iteration, cheap for `state_dim` ≤ 256) to replace fixed `cfg_.alpha_carry` in `hierarchy_update()`; scale `lr` in `train_step` by `1.0/(‖work_err_‖_F + 1e-8)` per `RPSM_IMPLEMENTATION.md:51-52`. Add both as unit tests (`α ∈ [0.3,0.6]` on random Ψ; forgetting ratio `< 0.01` over 100 steps, per the spec's own verification order) | Small — two self-contained numerical functions, no cross-module wiring | none |
| **0** | Fix the frozen classifier (Finding #1) — **highest expected BPC impact, do first** | Add `Ψ_mu` row-1..K gradient update to `RpsmSequenceLayer::train_step`: `delta_k[j] -= lr_head · grad_logits[k] · psi_.inv_var[j] · (h[j] − mu0[j] − delta_k[j])` for all `k`, mirroring the encoder-side `enc_grad_` computation already present two lines above (`rpsm_sequence_layer.cpp:440-447`). Also consider updating `mu0` (row 0, world prior) with a smaller learning rate, and `inv_var` via an online variance estimate (currently also frozen at `1.0`, `init_psi_matrices:87`) if Phase 0's isolated `Ψ_mu` fix under-delivers. Add a changing-target train smoke test (cycle through several target classes, not `rpsm_train_smoke.cpp`'s fixed `kTarget=3`) to close the Finding #1 blind spot in §4.4 | Small-to-medium — one function, ~15-20 lines of gradient math, mirrors an existing pattern in the same file and an already-proven pattern in `memory_train.cpp` | none — independent of Phase -1, can land first or in parallel |
| **1** | Config/CLI plumbing to reach "Small" tier and beyond | Add `rpsm_n_memory_slots`, `rpsm_alpha_carry` (only if Phase -1's spectral α is *not* taken, else remove the fixed field entirely), `rpsm_beta_memory`, `rpsm_hierarchy_loss_weight`, `rpsm_surprise_threshold` to `CyphaLMConfig` (`cyphalm_config.hpp:146-151`) and thread them into `RpsmSequenceConfig` construction (`cyphalm_model.cpp:329-336`); remove `apply_bench_mode`'s hard override of `rpsm_n_levels`/`rpsm_state_dim`/`rpsm_feat_dim` (`cyphalm_config.cpp:97-100`) so a JSON profile can actually reach Small/Medium tier without a second code path silently resetting it | Small — config plumbing, follows the exact pattern already used for `rpsm_n_levels`/`rpsm_state_dim`/`rpsm_feat_dim` | none — independent of Phase 0/-1 |
| **2** | Sanity sweep at cheap scale: does Phase 0/-1 close the gap, and does tier size matter independently? | New bench domain (see §7) at `n_train=5000` (matching `kD41ScaleNTrain` convention, `bench_domains.cpp:4835`), comparing: (a) current code as-is (frozen `Ψ_mu`, Tiny tier — reproduces the 7.336-class result at small scale as a control), (b) Phase 0 fix only at Tiny tier, (c) Phase 0 + Phase -1 at Tiny tier, (d) Phase 0 + Phase -1 at Small tier (needs Phase 1). This isolates "was it the frozen classifier" from "was it under-tuning" before committing any 300k run | Small — mostly bench-domain plumbing, reusing `run_math_integration_bench_subprocess`-style helpers | Phase 0 (mandatory); Phase -1 and Phase 1 for the (c)/(d) arms |
| **3** | Fix the embed backprop stub properly (Finding #2) — only if Phase 2 shows the gap isn't fully closed | Replace `rpsm_embed_backprop_stub` with a real backward pass: transpose-multiply `rpsm_layer_->input_grad()` through `proj_ssm_` to get `∂loss/∂ctx`, then backprop through `ssm_step` (or, cheaper first cut, stop at `ctx` and only fix the dimensional mismatch by routing through `proj_ssm_`'s transpose rather than truncating field-space onto embed-space directly) | Medium — needs a `ssm_step` backward or a documented, deliberate scope-cut at the projection boundary | Phase 0 (to know if it's even needed), Phase 2 (to confirm) |
| **4** | Option A completion for the LM path | Port the general `CyphaInferModel`'s online `mu_world`/`inv_v` update mechanism (`memory_train.cpp:709,717`) into `RpsmSequenceLayer` in place of Phase 0's from-scratch gradient, if Phase 0's simpler SGD update proves numerically unstable at 300k scale; add the missing `native_batched_update_parity` test (§3.1); wire `TieredContextBuffer`/`ctx` into the LM call sites (Finding #4) | Medium | Phase 0 |
| **5** | Option B remaining pieces | Real Izaac VRF store (currently a documented SHA-256-of-quantised-state stub per `RPSM_IMPLEMENTATION.md:89-91` — not found implemented at all in `native/src/rpsm/`, so this is still fully "Planned," unlike Fixes 3-5 in §3.2), Gaussian-mixture world model injection at the top level (`RPSM_COMBINED_SPEC.md:100-102`) | Large — genuinely new subsystems, matches the original "Planned" status honestly | Phases 0–4 showing RPSM is worth continued investment |
| **6** | Confirm at Small/Medium tier, 300k, single seed, scratch build | Run `--profile d21 --mode rpsm` at Small tier (post-Phase 1) with Phase 0 (+possibly -1/3) fixes in `native/build_scale` (not `native/build_math`), compare BPC against the 7.336 pin and the 2.864/2.873 hybrid pins | — | Phase 2 showing a clearly positive direction |
| **7** | Lock + compare | If Phase 6 shows a meaningful BPC improvement, propose a new, separate `bench/BASELINE_LOCK.json` section (e.g. `rpsm_upgrade_results`) — do not overwrite `rpsm_results`, which is the reference other domains validate against (`baseline_lock_smoke.cpp:189`) | — | Phase 6 |

**Top recommendation: do Phase 0 alone first, in isolation, and re-measure at a cheap scale (Phase 2, n_train=5000) before touching anything else.** It is the cheapest possible change (one function, no new subsystem, no config plumbing needed to test at Tiny tier), it targets the single most mechanically certain defect found (a classifier that is provably never updated), and it is fully decoupled from Phases -1/1/3, so it can be measured on its own before deciding whether the remaining phases are worth the larger effort.

---

## 7. Validation / bench-domain proposal

Following the `d41_math_integration_scale_validation` pattern (`bench_domains.cpp:5073-5177`, `run_math_integration_bench_subprocess` helper at `:4884`): subprocess-drive `cyphalm_bench_native --profile d21 --mode rpsm` at each config-under-test, diff `bpc`/`kappa`/`loss` trajectories, `finalize_domain(...)`. Highest currently-registered domain is `d76` (`bench_domains.cpp:9344`); the hidden-dim scale plan already claims `d77` (`docs/reports/HIDDEN_DIM_SCALE_PLAN.md:198`) — propose **`d78_rpsm_upgrade_validation`**, registered alongside the others at `bench_domains.cpp:9225-9344`, comparing the four arms from Phase 2 above plus the current 7.336 pin as a fixed reference row.

Do not add a differently-named `bench/config/profiles/cyphalm_d21_*.json` variant for the corpus — same reasoning as the hidden-dim plan (`HIDDEN_DIM_SCALE_PLAN.md:200`): `apply_bench_profile` only grants the full WikiText corpus to the literal string `"d21"` (`cyphalm_config.cpp:243-244`, `cyphalm_corpus.cpp:146`). Use `--profile d21 --mode rpsm` plus Phase 1's new CLI/config overrides layered on top, exactly like `--lstm-hidden N` is layered onto `--profile d17` in the hidden-dim plan.

---

## 8. Summary of blockers and risks

1. **The docs (`RESEARCH_STATUS.md`, `FUTURE.md`, `CYPHA_FULL_CPP_FRAMEWORK_PLAN.md`) understate how much of Option A/B is already built.** Anyone picking up this task from the docs alone would likely start re-implementing the batched-LLR GEMM kernel or the hierarchy/memory scaffold from scratch — both already exist, are tested, and are live. The actual gap is training-loop completeness (Findings #1-#3), not architecture.
2. **Finding #1 (frozen `Ψ_mu`) is the highest-confidence, lowest-cost fix and should be tried before anything else**, including before the larger Option A/B phases the docs currently frame as the only path forward. It is plausible, though not certain, that this alone closes most of the 7.336-vs-2.864 gap; §6 Phase 2 is designed specifically to measure this cheaply before committing to anything expensive.
3. **Finding #2 (`rpsm_embed_backprop_stub`) is real but not provably dominant** — it is a second, independent defect that compounds with Finding #1 but is more expensive to fix correctly (needs a real backward pass through `project_field`/`ssm_step`) and should be sequenced after Phase 0/2 tells us whether it's still worth doing.
4. **Finding #3 (Tiny-tier config, no plumbing to Small)** means the current 7.336 number is not even a fair test of the spec's own intended benchmark tier — the spec's own stop/go gate (`RPSM_IMPLEMENTATION.md:116`) is defined at Small tier, and D21 production has never run at Small tier because the code doesn't currently allow it without a change (Phase 1).
5. **No numerical-stability blocker was found in the scaffold itself** — `rpsm_train_smoke`/`rpsm_hierarchy_smoke` pass today (finite loss, finite log-probs, no NaN over 64+ steps at Tiny-equivalent unit-test scale), and orthogonal `W_up` init plus symmetric `W_down` (Fixes 3/5) are already in place, which are exactly the two fixes the original research flagged as most load-bearing for numerical stability (`κ≈450 → 1.0` spectral condition number). This lowers the risk that Phase 0's fix will destabilize training, though it should still be verified at 300k scale (Phase 6), not assumed.
6. **The "RPSM core fixes" (Phase -1) are only 60% blocking-relevant** — Fixes 3–5 are done; only Fixes 1–2 remain, and neither is required for Phase 0 to be testable (Phase 0 can be measured at fixed `alpha_carry=0.5` and fixed `lr` first; Phase -1 is a secondary lever, not a prerequisite).

---

## Appendix: commands reference

All commands are read-only (no writes) or use a scratch build directory (`native/build_scale`, already claimed by the hidden-dim plan as the shared scratch dir — coordinate before both plans build there simultaneously) to avoid touching `native/build_math` (the live overnight run) or `bench/BASELINE_LOCK.json`.

```powershell
# One-time: configure a separate scratch build (does not touch native/build_math)
cmake -S native -B native/build_scale -DCMAKE_BUILD_TYPE=Release
cmake --build native/build_scale --target cyphalm_bench_native --config Release

# Phase 2 sanity sweep (n_train=5000) — after Phase 0 fix lands, before any 300k commitment
native/build_scale/cyphalm_bench_native --profile d21 --mode rpsm --n-train 5000 --n-eval 256 `
    --threads 1 --intelligence-profile
# (repeat post-Phase-0-fix build; compare bpc/kappa against this pre-fix control run)

# Phase 6 production-scale confirmation (only after Phase 2 shows a clear improvement)
native/build_scale/cyphalm_bench_native --profile d21 --mode rpsm --n-train 300000 --n-eval 2000 `
    --threads 1 --intelligence-profile

# Existing RPSM unit/parity tests (read-only verification that current scaffold still passes)
ctest --test-dir native/build_scale -R "rpsm"

# Read-only baseline inspection used to write this document (no writes):
Get-Content bench/BASELINE_LOCK.json
Get-Content bench/config/profiles/cyphalm_d21_rpsm.json
Get-Content bench/config/d21_rpsm_profile.json
```

---

## 9. Phase 0 results (2026-07-11)

**Status: implemented, verified, measured. Finding #1 confirmed and fixed in isolation, per this document's own recommendation (§6, "do Phase 0 alone first").**

### 9.1 What was confirmed

Re-read `RpsmSequenceLayer::train_step` (`native/src/rpsm/rpsm_sequence_layer.cpp:401-486`) firsthand, independent of this doc's own citations, and confirmed the diagnosis exactly as described in §4.1: `psi_.mu` (the real symbol name for `Ψ_mu` in code) rows `1..K` (`delta_k`, the per-class discriminative direction feeding `batched_llr_gemm`) were written exactly once, at construction (`init_psi_matrices`, `:81-100`), and never again — the only per-step write to `psi_` inside `train_step` was `psi_.counts[tgt] += 1.0` (`:482`, unchanged by this fix), which feeds only the frequency-prior bias `u_k`. Grepped every write to `.mu` in `native/src/rpsm/` to confirm no other path touches it. This is the structural analogue of the LSTM hybrid's `Wy` (`native/src/cyphalm/char_lstm.cpp:217,230,411` — full outer-product gradient, updated every step), confirmed by direct comparison of the two files' training loops.

### 9.2 The fix

Added ~15 lines to `RpsmSequenceLayer::train_step` immediately after the existing encoder-gradient loop (`rpsm_sequence_layer.cpp:440-447`, unchanged), following the exact derivation in §4.1:

```
∂llr_k/∂delta_k[j] = inv_var[j] · (feat[j] − mu0[j] − delta_k[j])
delta_k[j] -= lr · grad_logits[k] · inv_var[j] · (feat[j] − mu0[j] − delta_k[j])
```

`grad_logits` (softmax probs minus one-hot target) was already computed two lines above for the encoder gradient — this is the same chain-rule term, applied to the previously-untouched parameter, using the same `lr` (`cfg_.rpsm_lr`) and the same plain-SGD update mechanics already used for every other RPSM-trained parameter (`w_enc_`, `w_carry_`, `w_up_`) and confirmed to match the hybrid path's own convention (`char_lstm.cpp:407-412`: no momentum, no Adam, no weight decay — `param -= lr * grad` for every trained matrix). Row 0 (`mu0`, world prior) was deliberately left untouched, matching the "isolated Ψ_mu fix" scope this document recommended testing first (§6 Phase 0); `inv_var` (frozen at `1.0`) and `alpha_carry`/normalised-η (Phase -1) were also left untouched.

Also added `native/tests/parity/rpsm_train_multiclass_smoke.cpp` (CTest `native_rpsm_train_multiclass_smoke`) to close the blind spot identified in §4.4: it cycles through 4 target classes (not `rpsm_train_smoke`'s fixed single target), asserts `Ψ_mu` rows `1..K` move by a non-trivial norm after training, and asserts a trained layer beats a freshly-initialized one on multi-class held-out NLL — a check a frozen classifier cannot pass.

### 9.3 Build and test results

Built from scratch in a dedicated scratch directory (`native/build_rpsm`, not `native/build_math` or `native/build_scale`). `ctest --test-dir native/build_rpsm -R rpsm` → **7/7 passed** (0.24s–0.93s each): `native_rpsm_batched_llr_smoke`, `native_rpsm_sequence_smoke`, `native_rpsm_hierarchy_smoke`, `native_rpsm_train_multiclass_smoke` (new), `native_rpsm_train_smoke`, `native_cyphalm_bench_rpsm_smoke`, `native_d21_rpsm_smoke` — no regressions in any existing RPSM test.

### 9.4 Before/after BPC, clean scratch comparison at n_train=5000

To get a true apples-to-apples before/after (rather than comparing against the 300k-scale `bench/BASELINE_LOCK.json` pin, which is a different `n_train`), the unpatched `rpsm_sequence_layer.cpp` was built in a second, separate scratch directory (`native/build_rpsm_before`, deleted after use — not committed), and both binaries were run with identical settings: `--profile d21 --mode rpsm --n-train 5000 --n-eval 256 --threads 1`, same seed (42), same Tiny-tier config (`L=4, D=128, feat=64`, unchanged by this phase — see §9.5).

| Run | BPC | Δ vs before |
|---|---|---|
| **Before fix** (frozen `Ψ_mu`, this repo's current committed behavior prior to this change) | **7.4053** | — |
| **After fix** (Phase 0, Tiny tier) | **4.8221** | **−2.583 BPC (−34.9%)** |

For reference, the existing `bench/BASELINE_LOCK.json` 300k-scale pins (different `n_train`, not directly comparable, shown only as context): `rpsm_results` (pre-fix, 300k) = 7.336; `overnight_results` (D17 hybrid, 300k) = 2.864.

**The fix works and the effect size is large and in the predicted direction** — a ~35% relative BPC reduction from a single, self-contained ~15-line change, with zero test regressions. This is a strong confirmation that Finding #1 was a real, significant defect, not a minor one.

**Gap remaining:** even after the fix, RPSM at Tiny tier / n_train=5000 (4.822 BPC) is still well above the hybrid's 300k-scale pin (2.864 BPC). Some of that gap is expected to be an artifact of scale/tier mismatch (5k vs 300k steps; Tiny vs Small tier — see §9.5) rather than a remaining classifier defect, but a same-n_train, same-tier hybrid comparison was out of scope for this task and has not been run; this document does not claim the remaining gap is fully explained.

### 9.5 Quick Small-tier check (not committed, config plumbing still needed for a real run)

Per §4.3 (Finding #3), `apply_bench_mode`'s `BenchMode::Rpsm` case (`native/src/cyphalm/cyphalm_config.cpp:95-101`) hard-codes Tiny-tier dims (`L=4, D=128, feat=64`) after `apply_bench_profile` runs, so no JSON profile or existing CLI flag can reach Small tier (`L=8, D=256, feat=128`) today — confirmed firsthand, matching the doc's claim. As a quick, throwaway experiment (three constants edited locally, rebuilt, run once, then reverted — confirmed via `git status`/`git diff` to leave zero net diff on `cyphalm_config.cpp`), Small tier was tested at the same `n_train=5000`, with the Phase 0 fix applied:

| Tier | BPC (with Phase 0 fix, n_train=5000) |
|---|---|
| Tiny (`L=4, D=128, feat=64`, current production) | 4.8221 |
| Small (`L=8, D=256, feat=128`, spec's benchmark tier) | 4.7819 |

Small tier gives a further **−0.040 BPC (−0.8%)** — real, but an order of magnitude smaller than Phase 0's own **−2.583 BPC**. **Recommendation: tier size is a minor lever here, not a major one; Phase 1 (config plumbing to reach Small/Medium tier for real, committed runs) is not urgent.** This one-off check used a temporary source edit that was reverted and is not part of this commit; a real Small-tier run still requires the Phase 1 plumbing (removing `apply_bench_mode`'s hard override, adding CLI/config fields) to be testable without hand-editing source.

### 9.6 Recommendation on next steps

**Phase 0 alone recovers a large fraction of the frozen-classifier gap (~35% relative BPC reduction) at effectively zero cost and zero risk (one function, no regressions).** Given that:

1. The remaining gap to the hybrid's 2.864 pin is still large (4.822 vs 2.864 at mismatched scale/tier), so **Option A/B's remaining phases are not obsoleted by this fix alone** — there is still real work to do.
2. Tier size (Phase 1, §9.5) is confirmed to be a minor lever (−0.8%), not a major one — **deprioritize Phase 1** relative to where it sat in the original plan.
3. Finding #2 (`rpsm_embed_backprop_stub`'s dimensionally-unsound update, §4.2) and Phase -1 (spectral α / normalised η, §3.2) remain unmeasured and are the next-highest-confidence levers per the original phased plan (§6) — **recommend Finding #2 next**, since an untrained/mis-trained `proj_ssm_`+embedding path plausibly compounds with the classifier fix the same way the original doc predicted, and Phase -1's normalised-η fix directly targets training stability at whatever scale is attempted next.
4. Before committing to Phase 3 (fixing Finding #2's backward pass, non-trivial: needs a real backward through `project_field`/`ssm_step`) or Phase 6 (300k confirmation run), do a same-tier, same-`n_train`, same-scale **hybrid-vs-RPSM** comparison (not attempted here, out of this task's scope) to get a fair apples-to-apples gap measurement before deciding how much further investment is justified.

**Net: this alone does not close enough of the gap to deprioritize Option A/B — it is a strong, cheap first win, but a large gap remains, and the original phased plan's ordering (Phase 0 → measure → decide) is validated: proceed to Finding #2 next, not directly to the large Option A/B refactors (Phases 4-5).**

### Files touched by Phase 0

- `native/src/rpsm/rpsm_sequence_layer.cpp` — the fix (`Ψ_mu` rows `1..K` gradient update in `train_step`)
- `native/tests/parity/rpsm_train_multiclass_smoke.cpp` — new changing-target CTest closing the §4.4 blind spot
- `native/cmake/CyphaParity.cmake`, `native/CMakeLists.txt` — register the new test target/CTest
- `native/src/cyphalm/cyphalm_config.cpp` — **not** modified in the committed diff; the Small-tier check in §9.5 used a local, reverted edit only

---

## 10. Phase 0b results (2026-07-11) — Finding #2 (`rpsm_embed_backprop_stub`) fixed and measured

**Status: implemented, verified (finite-difference + CTest), measured. Finding #2 confirmed exactly as diagnosed in §4.2, fixed with a real chain-rule backward pass, and BPC-measured against the Phase 0 baseline.**

### 10.1 What was confirmed

Re-read `CyphaLMModel::rpsm_embed_backprop_stub` (`native/src/cyphalm/cyphalm_model.cpp:1379-1395` pre-fix) and its call site (`train_step_rpsm`, immediately after `rpsm_layer_->train_step(...)`) firsthand, plus the full forward chain between the embedding table and the RPSM layer's input: `embed_->embed_vec(token_id)` → `ssm_step(e)` (dispatches to `CellAISSM::step`/`HierarchicalSSM::step`; for the D21 RPSM profile — `use_hierarchical_ssm=false`, `use_differential_gate=false`, `use_ca_state_cell=false`, `use_reversible_cell=false` — this reduces to a bare `ssm_->step(e)` call) → `project_field(ctx)` (`field_x_ = proj_ssm_ · ctx`, a plain matvec) → `rpsm_layer_->train_step(field_x_, ...)`. Confirmed §4.2's diagnosis exactly: the stub read `rpsm_layer_->input_grad()` (gradient of loss w.r.t. the RPSM layer's own input, i.e. w.r.t. `field_x_`, restricted to the leading `rpsm_state_dim` entries the layer's forward pass actually reads — see `encode_level0_features`/`inject_input_multilevel` in `rpsm_sequence_layer.cpp`, both loop-bounded by `state_dim`) and subtracted its leading `min(d_embed, field_grad.size())` entries directly from the embedding row, with no `proj_ssm_` or SSM involvement at all — a basis mismatch (field-space vs embed-space), not a valid chain-rule gradient, exactly as diagnosed.

Also read `CellAISSM::step`'s non-spectral leaky-integrator branch (`native/src/cyphalm/cellai_ssm.cpp:203-252`, the branch taken for the D21 RPSM profile since `use_spectral_pde=false`) and its public accessors (`w_fast_layer0()`, `w_slow_layer0()`, `lambda_fast()`, `lambda_slow()`, `multiscale_alpha()`) to determine the true differentiable path: `h[i] = lf·h_prev[i] + (1-lf)·(W_fast·e)[i]`, `s[i] = ls·s_prev[i] + (1-ls)·(W_slow·e)[i]`, and (with `use_multiscale=true`, also the D21 profile's setting) `ctx = [α·h+(1-α)·s ; s]` per layer. Cross-checked the general transpose-multiply convention against two existing precedents in this codebase: `char_lstm.cpp`'s `backward_step` (`dx = Wx^T · dgates`, then `dE[token] = dx` directly, confirming "transpose the forward linear map to backprop through it" is the established pattern) and this same file's `bptt_ssm_update` (`native/src/cyphalm/cyphalm_model.cpp:1185-1291`, the hybrid path's own SSM backprop), which already computes `grad_ctx = matvec_transpose(proj_ssm_, field_dim, ctx_dim, grad_field)` for a different purpose (updating `W_fast`, not embeddings) — reusing that exact transpose convention rather than inventing a new one.

### 10.2 The fix

Renamed `rpsm_embed_backprop_stub` → `rpsm_embed_backprop` (it is no longer a stub) and replaced its body with a real backward pass, implementing exactly the chain the task and §4.2 specified: `loss → field_x_ → ctx → e`.

1. **Zero-pad, don't truncate/misalign**: `rpsm_layer_->input_grad()` (length `rpsm_state_dim`) is copied into a `field_dim`-length buffer with trailing entries left at zero — exact, not an approximation, since `RpsmSequenceLayer::train_step` never reads those trailing entries in the first place (their true gradient is 0).
2. **Transpose through `proj_ssm_`**: `grad_ctx = proj_ssm_^T · grad_field`, reusing the existing `matvec_transpose` helper already defined in this file for `bptt_ssm_update`.
3. **Transpose through the SSM's layer-0 state transition**: split `grad_ctx` into `grad_h`/`grad_s` (accounting for the `use_multiscale` blend: `grad_h = α·grad_ctx_blend`, `grad_s = (1-α)·grad_ctx_blend + grad_ctx_s`), then `grad_e = (1-lf)·W_fast^T·grad_h + (1-ls)·W_slow^T·grad_s`, using `CellAISSM`'s public `w_fast_layer0()`/`w_slow_layer0()`/`lambda_fast()`/`lambda_slow()`/`multiscale_alpha()` accessors — no changes to `CellAISSM` itself were needed.
4. **Apply to the embedding row** with the same `lr = cfg_.rpsm_lr * 0.1` and plain-SGD convention (`row[j] -= lr * grad_e[j]`) the old stub already used, per the task's instruction to keep the established optimizer convention.

**Deliberate, documented scope decision:** only SSM layer 0 is backpropped through (layer 1+, if `ssm_layers>1` as in the D21 profile, takes the previous layer's `ctx` as its own input, not the raw embedding `e`, and `CellAISSM`'s public API only exposes `w_fast_layer0()`/`w_slow_layer0()` for layer 0 in the first place). This exactly mirrors the layer-0-only truncation the neighboring `bptt_ssm_update` already uses for this same SSM (it also only reads `grad_ctx[0:2·d_state]`), so it is consistent with existing codebase precedent rather than a new gap. Practically: for D21 (`ssm_layers=2`), this fix computes the *direct* contribution of layer 0's state to the field vector and does not additionally unroll the indirect contribution that flows through layer 1 back into layer 0 — a real, minor, and explicitly-documented incompleteness, not a basis-mismatch bug like the original stub.

### 10.3 Finite-difference verification (new permanent regression guard)

Added `native/tests/parity/rpsm_embed_grad_finite_diff.cpp` (CTest `native_rpsm_embed_grad_finite_diff`), per the task's request for "the gold-standard way to catch this exact class of bug." Uses a toy configuration with a single SSM layer (`n_layers=1`, so the fix's layer-0-only scope decision introduces *zero* approximation — the analytic formula is exactly complete for this test) and `use_multiscale=true` (matching the D21 profile). Computes:

- **Analytic gradient**: the same transpose-chain formula now in `rpsm_embed_backprop` (kept independently in the test, cross-referenced by comment, so a future accidental edit to one without the other is visible in review) — `field_grad → proj_ssm_^T → layer-0 leaky-integrator transpose → grad_e`.
- **Numerical gradient**: central finite differences (`ε=1e-5`) directly on `nll(e)`, evaluated purely via forward calls into the real `CellAISSM`/`RpsmSequenceLayer` classes (fresh instances per evaluation, same seeds, zero state leakage across evaluations) — no gradient-formula duplication, only forward math.

Result: **max relative error across all 6 embedding dims < 1e-4** (well within finite-difference noise for a smooth `tanh`-activated forward pass). An earlier draft of this test also cross-checked that the *old* stub formula (pasting `field_grad` directly onto embed-space) diverges from the numerical gradient — this was dropped from the committed test because the magnitude of that divergence is scale-dependent on arbitrary toy-matrix choices and isn't a robust, seed-independent property to assert on; the finite-difference-vs-analytic check above is the real, permanent regression guard and needs no such crutch.

### 10.4 CTest results

Built from scratch in `native/build_rpsm` (reused; already had Phase 0's fix). `ctest --test-dir native/build_rpsm -R rpsm` → **8/8 passed**: the 7 tests from Phase 0 (`native_rpsm_batched_llr_smoke`, `native_rpsm_sequence_smoke`, `native_rpsm_hierarchy_smoke`, `native_rpsm_train_multiclass_smoke`, `native_rpsm_train_smoke`, `native_cyphalm_bench_rpsm_smoke`, `native_d21_rpsm_smoke`) plus the new `native_rpsm_embed_grad_finite_diff` — no regressions.

### 10.5 Before/after BPC, clean scratch comparison at n_train=5000

Same methodology as §9.4 (apples-to-apples, same seed/tier/corpus), refined to avoid a pitfall the first attempt hit: building the pre-fix source in a separate `git worktree` initially resolved to the `gutenberg_fallback` corpus instead of `wikitext2` (the worktree checkout didn't have the same corpus data available on that path), which would have made the comparison invalid. Fixed by building the pre-fix binary in a scratch directory *inside* the main checkout instead (`native/build_rpsm_before`, deleted after use — not committed): temporarily `git checkout HEAD --` the two fix files only, configured+built there, ran the bench, then immediately restored the fix files from a backup copy before running the after-fix bench. Both runs confirmed `"corpus": "wikitext2"` in their JSON output.

| Run | BPC | Corpus | Δ vs before |
|---|---|---|---|
| **Before** (Phase 0 fix only — this repo's `HEAD` at commit `788cea0`, i.e. Finding #1 fixed, Finding #2 still the old stub) | **4.822095365884579** | wikitext2 | — |
| **After** (Phase 0 + Phase 0b — Finding #1 and Finding #2 both fixed) | **4.821779758048934** | wikitext2 | **−0.000316 BPC (−0.0066% relative)** |

Both runs used identical settings (`--profile d21 --mode rpsm --n-train 5000 --n-eval 256 --threads 1`, `bench_seed=42`, Tiny tier) and are fully deterministic (re-ran the after-fix bench a second time after rebuilding `native/build_rpsm` from a clean state and got the bit-identical `4.821779758048934`).

**The fix is mathematically necessary and now verified correct (finite-difference match), but its standalone BPC impact at this scale is negligible** — a ~150x smaller effect than Phase 0's −2.583 BPC. This is a real, reproducible, if small, measured improvement in the right direction (not noise — the run is deterministic), not an absence of effect.

### 10.6 Why the BPC impact is so small (root-cause, not just an observation)

Three compounding reasons, found by re-reading the actual magnitudes in the forward chain rather than speculating:

1. **Double small-scale attenuation.** `proj_ssm_` is initialized at scale `0.02` (`init_proj_from_rng(proj_ssm_, ..., kScale=0.02)`, `cyphalm_model.cpp:218`) and `CellAISSM`'s `W_fast`/`W_slow` at scale `0.05` (`cellai_ssm.cpp:92,95`, `v = rng.normal() * 0.05`). The corrected gradient passes through *two* transposed multiplications by matrices at these scales before reaching the embedding, so `grad_e` is systematically much smaller in magnitude than the raw `field_grad` the old stub pasted in directly (which had no such attenuation, just wrong basis). A "more correct but structurally smaller" update and a "wrong but structurally larger" update can end up nudging the embedding by similar net amounts over a small number of steps — which is consistent with, and a plausible mechanistic explanation for, the near-zero measured delta.
2. **Thin per-row update budget.** Each step updates only the *current* token's embedding row (`rpsm_embed_backprop(token_id)`, one row of `embed_->table()`). At `vocab_size=256` and `n_train=5000`, each row is touched only ~20 times on average — not enough steps for even a correctly-signed gradient (right or wrong formula) to move a row far from its random `EmbedTable` init.
3. **The embedding is not the bottleneck this fix targets.** Finding #1 (frozen `Ψ_mu`) directly gated the *entire* 256-way classification decision every step; Finding #2 only affects how much the *input features* to that (now-trained) classifier can be reshaped — a strictly smaller lever, exactly as §4.2's own "priority relative to Finding #1" note anticipated ("lower confidence of standalone impact... fix Finding #1 first").

None of this means the fix was unnecessary: an incorrect gradient is still incorrect regardless of its measured magnitude, and over a much longer run (300k steps, the production tier) accumulated small-but-consistently-*wrong*-direction nudges could compound differently than accumulated small-but-*correct* ones — this was not tested here (out of the cheap-scale scope the task requested) and is flagged in §10.7 as the one open question worth a cheap follow-up if RPSM is revisited.

### 10.7 Updated recommendation and final verdict

**Verdict: RPSM is not yet close enough to the hybrid baseline for Options A/B's remaining phases to be deprioritized, and Phase 0b does not change that conclusion — but Phase 0b was still the correct, necessary next step per the original phased plan (§6), and it can now be considered closed.**

- The two highest-confidence, cheapest fixes identified by this document's original scoping pass (Finding #1, Phase 0; Finding #2, Phase 0b) are both now implemented, both verified (CTest + finite-difference for Phase 0b, CTest + a fresh/trained NLL comparison for Phase 0), and both measured at the same cheap scale (`n_train=5000`, Tiny tier). Combined effect: BPC 7.4053 → 4.8218 (**−2.5835 BPC, −34.9% relative**, almost entirely from Phase 0; Phase 0b's own contribution was real but small, §10.5-10.6).
- **A large, well-measured gap to the hybrid baseline remains**: 4.8218 (RPSM, Tiny tier, 5k steps) vs the existing `bench/BASELINE_LOCK.json` pins of 2.864 (hybrid, 300k steps) / 7.336 (RPSM pre-Phase-0, 300k steps) — not directly comparable scales, but nothing measured in Phase 0, §9.5's tier check (−0.8%), or Phase 0b (−0.0066%) closes anywhere near that gap.
- **Recommendation, in priority order, if RPSM work continues:**
  1. Phase -1 (spectral α, normalised η — §3.2, §6) remains unmeasured and is now the next-highest-confidence unexplored lever from the original plan; it directly targets training stability/dynamics rather than a training-loop completeness bug, so it's a different kind of lever than Findings #1/#2 and hasn't been ruled out by either fix.
  2. If RPSM is to be scaled up at all, a real same-`n_train`, same-tier **hybrid-vs-RPSM** A/B (not yet done in either Phase 0 or Phase 0b — both only compared RPSM against itself, before vs after) is the single most informative next measurement, since it would finally show whether the remaining gap is "RPSM architecture" or "RPSM undertrained relative to a fair comparison."
  3. Phase 1 (config plumbing to Small/Medium tier) remains correctly deprioritized per §9.5/§9.6's own finding (tier size is a minor lever, −0.8%).
  4. The larger Option A/B refactors (Phases 4-5 in §6 — online `mu`/`inv_v` update mechanism, Izaac VRF store, Gaussian-mixture world model) remain the right things to defer until 1-2 above are done, exactly as originally planned — **this fix does not change that ordering, it just confirms Phase 0b was worth doing (correctness) without being the thing that closes the gap (impact).**

### Files touched by Phase 0b

- `native/src/cyphalm/cyphalm_model.cpp` — the fix (`rpsm_embed_backprop_stub` → `rpsm_embed_backprop`, real chain-rule backward pass)
- `native/include/cypha/cyphalm/cyphalm_model.hpp` — declaration rename to match
- `native/tests/parity/rpsm_embed_grad_finite_diff.cpp` — new finite-difference CTest closing the "gold-standard regression guard" request
- `native/cmake/CyphaParity.cmake`, `native/CMakeLists.txt` — register the new test target/CTest
- `docs/reports/RPSM_UPGRADE_PLAN.md` — this section

---

### Files read/cited in this scoping pass

- `docs/reports/DEV_PLAN_2026-07-11.md` (§1–3, full read)
- `docs/reports/HIDDEN_DIM_SCALE_PLAN.md` (full read, style/rigor reference)
- `docs/RESEARCH_STATUS.md:350-430` ("Possible upgrades" table, priorities)
- `docs/FUTURE.md:190-250` (§9-10, RPSM matrix refactor + sequence layer)
- `docs/native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md` (full read)
- `docs/research/upgrades/RPSM_COMBINED_SPEC.md` (full read)
- `docs/research/upgrades/RPSM_IMPLEMENTATION.md` (full read)
- `native/include/cypha/rpsm/psi_matrices.hpp`, `native/src/rpsm/psi_matrices.cpp` (full read)
- `native/include/cypha/rpsm/rpsm_sequence_layer.hpp`, `native/src/rpsm/rpsm_sequence_layer.cpp` (full read)
- `native/include/cypha/cyphalm/cyphalm_config.hpp` (full read)
- `native/src/cyphalm/cyphalm_config.cpp` (`apply_bench_mode`, `apply_bench_profile`, RPSM sections)
- `native/include/cypha/cyphalm/cyphalm_model.hpp` (RPSM member/method declarations)
- `native/src/cyphalm/cyphalm_model.cpp` (`uses_rpsm`, `init_components`, `forward`, `train_step_rpsm`, `rpsm_embed_backprop_stub`, `train_sequence_rpsm`, `project_field`)
- `native/src/cyphalm/char_lstm.cpp` (`CharLSTMGrad`/`backward_step`/`apply_gradients` — `Wy`/`dWy` comparison)
- `native/src/infer_cpu.cpp` (`CYPHA_USE_RPSM_LLR`, `score_matrix_use_field`, `rpsm_score_matrix_batched`)
- `native/src/memory_train.cpp` (existing online `mu`/`D` update pattern, cited as the fix template)
- `native/tests/parity/rpsm_batched_llr_smoke.cpp`, `rpsm_hierarchy_smoke.cpp`, `rpsm_train_smoke.cpp` (full read)
- `native/CMakeLists.txt` (RPSM CTest registrations)
- `bench/BASELINE_LOCK.json` (read-only; `rpsm_results`, `overnight_results` sections)
- `bench/config/profiles/cyphalm_d21_rpsm.json`, `bench/config/d21_rpsm_profile.json`, `bench/config/profiles/cyphalm_d17_wikitext.json` (read-only, config comparison)
- `native/src/bench/bench_domains.cpp` (domain registration table, `d41`/`d76` pattern for the new-domain proposal)
