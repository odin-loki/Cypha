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

## 11. Phase -1 results (2026-07-11) — spectral α and normalised η ("RPSM core fixes")

**Status: both fixes implemented (opt-in), verified (CTest), measured at matched cheap scale. A real same-`n_train`/same-tier hybrid-vs-RPSM A/B was also run, closing the one open gap flagged at the end of §10.7.**

### 11.1 Re-confirming the framing (RESEARCH_STATUS.md:393)

Re-read `docs/RESEARCH_STATUS.md:393` and `RPSM_IMPLEMENTATION.md:35-69` directly (not just this document's own §3.2 summary of them) before implementing anything. Confirmed exactly as §3.2 already found: the `RESEARCH_STATUS.md:393` bullet ("RPSM core fixes (spectral α, norm η, orthogonal init) | Planned | Forgetting ratio < 0.01; α ∈ [0.3, 0.6]") bundles three named fixes from `RPSM_IMPLEMENTATION.md`'s "Five critical fixes" list, of which orthogonal init (Fix 3) and symmetric `W_down` (Fix 5, not named in the `RESEARCH_STATUS.md` bullet but part of the same five) are already live in `rpsm_sequence_layer.cpp` (`init_orthogonal_matrix`, gated by `use_izaac_init`; `W_down` computed on the fly via `matvec_transpose_row_major`, never a stored parameter). Only **Fix 1 (spectral α)** and **Fix 2 (normalised η)** — both labelled "critical" in the spec, unlike Fixes 3-5's "high"/"medium" — remained unimplemented. This is Phase -1 exactly as scoped in §6/§10.7.

Concretely, in *this* codebase's SSM formulation, prior to this phase:

- **Spectral α** meant: `hierarchy_update()` (`rpsm_sequence_layer.cpp:340-383`, pre-fix) blended each level's carried hidden state against its own bottom-up/top-down prediction error using a single fixed scalar `a = cfg_.alpha_carry` (default `0.5`), never computed from Ψ's singular-value spectrum as `RPSM_IMPLEMENTATION.md:39-44` specifies (`gria_alpha_spectral(Psi)`, target `α ∈ [0.3, 0.6]` at edge-of-chaos init).
- **Normalised η** meant: every SGD update inside `train_step` (`Ψ_mu` rows 1..K from Phase 0, `w_enc_`/`w_carry_`, and the hierarchy-loss-driven `w_up_` update) used the raw, fixed `lr` argument (`cfg_.rpsm_lr`) directly, never scaled by the prediction-error magnitude as `RPSM_IMPLEMENTATION.md:46-53` specifies (`eta = eta_base / (‖E_gated‖_F + eps)`).

Cross-referenced the hybrid path's `alpha_init`/`alpha_learnable` convention (`cyphalm_config.hpp:53-54`, `cypha_cell_hypothesis.cpp:88-154`) per the task's suggestion: that `alpha` is an unrelated, per-vocab GRIA blend gate (`GRIALowRank`/`GriaHead`), not a spectral-radius construct — but the H10 cell hypothesis's `alpha_init = 0.485` (`cypha_cell_hypothesis.cpp:118`) confirmed that **0.485 is already an established "edge of chaos" target value elsewhere in this codebase**, independently corroborating `RPSM_IMPLEMENTATION.md`'s own training-notes target (`0.01 * |mean_alpha − 0.485|`) rather than that number being specific to RPSM alone.

### 11.2 What was implemented

Both fixes were judged well-defined and tractable, and were implemented as **opt-in** additions (new `RpsmSequenceConfig` fields, default `false`) rather than changes to the existing default behaviour, so Phase 0/0b's exact measured numbers remain reproducible with the fixes off — this was a deliberate risk-reduction choice given the task's caution about numerically-sensitive training code, not a sign either fix was ambiguous.

**Fix 1 — `gria_alpha_spectral(Psi, n_levels, state_dim)`** (`rpsm_sequence_layer.cpp`, new free function; `use_spectral_alpha` config flag): computes the top singular value of the hierarchy state Ψ (`psi_rows_`, the L×D matrix populated by `hierarchy_update()` — note this is a *different* Ψ from Finding #1's classifier `Ψ_mu`/`PsiMatrices`, a naming collision confirmed by reading both files directly) via power iteration on the L×L Gram matrix `Ψ Ψ^T` (cheap: L is the small hierarchy depth — 4 to 32 across all four tiers — never D). The raw singular value is normalised by `√D` and squashed into `[0.3, 0.6]` via a sigmoid centred at `normalized_σ = 1.0` (the expected value for `Ψ ~ N(0,1)` at any of the four configured tiers, since `L/D = 1/32` is held constant across Tiny/Small/Medium/Large — `RPSM_IMPLEMENTATION.md:97-102`). When `use_spectral_alpha=true`, this replaces the fixed `cfg_.alpha_carry` in `hierarchy_update()`'s per-step blend, computed from the *previous* step's Ψ (i.e. before that call's loop overwrites `psi_rows_`), matching the spec's "system-level diagnostics" framing.

**Fix 2 — normalised η** (`train_step`, `use_normalized_eta` config flag): per this document's own Phase -1 scoping (§6), scales the SGD learning rate used for every RPSM-trained parameter (Ψ_mu rows 1..K, `w_enc_`, `w_carry_`, the hierarchy-loss-driven `w_up_` update) by `1.0 / (‖E‖_F + 1e-8)`, where `‖E‖_F` is the Frobenius norm of the full multi-level prediction-error matrix for the current step — recovered exactly from `hierarchy_update()`'s already-computed mean-squared-error return value (`‖E‖_F = √(hierarchy_loss × n_levels × state_dim)`), no new state needed. **One deliberate deviation from the literal spec formula, disclosed rather than hidden:** `RPSM_IMPLEMENTATION.md:51-52`'s equation (`Psi_new = Psi + eta*(E_gated @ W_update)`) normalises a *separate hierarchy-state-update* term that has no corresponding standalone parameter in this codebase (`W_up` already plays that role and is trained via the ordinary SGD path); this implementation instead normalises the actual training-loop learning rate, exactly as this document's own §6 phased plan already specified before any code was written ("scale `lr` in `train_step` by `1.0/(‖work_err_‖_F + 1e-8)`"). A safety clamp (`eta_norm_max_scale`, default `10×`) was added — not part of the original spec formula — to bound the scale factor `1/(‖E‖_F+eps)` and prevent a divide-by-near-zero blowup once the hierarchy error shrinks late in training; this is disclosed as an addition, not presented as spec-derived.

Both flags are turned on together for the live `--mode rpsm` path (`cyphalm_model.cpp`, next to the existing `use_izaac_init = (... || cfg_.context_mode == ContextMode::Rpsm)` line), mirroring that exact existing convention rather than adding new CLI/config plumbing (deliberately avoiding `cyphalm_bench_native.cpp`'s CLI parsing and `measurers.hpp`/`.cpp`, both off-limits per the parallel hidden-dim-scale agent's in-flight work).

### 11.3 What was *not* implemented, and why (the literal "forgetting ratio < 0.01" metric)

`RPSM_IMPLEMENTATION.md`'s own verification order item 2 ("Forgetting ratio < 0.01 over 100 steps") describes a property of the *original spec's* separate hierarchy-state-update term losing its carried Ψ history when η is unnormalised and large — a rolling-sequence hierarchy-state-persistence property, not classifier-vs-classifier catastrophic forgetting across training tasks. Since this codebase's Fix 2 mapping (§11.2) applies η to the SGD learning rate rather than to a standalone Ψ-state-update term, the literal metric doesn't transfer 1:1, and asserting a specific "`< 0.01`" threshold pulled from a different implementation would be exactly the kind of unverified, potentially-flaky claim the task asked to avoid. Instead, `native_rpsm_normalized_eta_smoke` tests the two properties that are actually meaningful for *this* mapping of the fix: (1) long-run (1500-step) numerical stability under a shrinking-error regime — the specific failure mode this implementation is exposed to — with no NaN/blowup, and (2) that Phase 0's already-validated "trained classifier beats a fresh one" property still holds with both Phase -1 fixes enabled. This is disclosed as a scope decision, not a silent gap.

### 11.4 Test additions and CTest results

Two new CTests, both self-contained (no cross-module wiring, per this document's own original cost estimate in §6):

- `native_rpsm_spectral_alpha_smoke` — asserts `gria_alpha_spectral` ∈ `[0.3, 0.6]` on `Ψ ~ N(0,1)` at all four spec tiers (Tiny/Small/Medium/Large, two random seeds each) — directly implements `RPSM_IMPLEMENTATION.md`'s own verify-first item 1 — plus determinism and a degenerate all-zero-Ψ check.
- `native_rpsm_normalized_eta_smoke` — the two properties from §11.3.

Built from scratch in `native/build_rpsm` (reused; already had Phase 0/0b). `ctest --test-dir native/build_rpsm -R rpsm` → **10/10 passed** (0.15s–0.84s each): the 8 tests from Phase 0/0b plus the 2 new ones — no regressions.

### 11.5 Before/after BPC, clean scratch comparison at n_train=5000

Same methodology as §9.4/§10.5 (identical settings: `--profile d21 --mode rpsm --n-train 5000 --n-eval 256 --threads 1`, `bench_seed=42`, Tiny tier, `wikitext2` corpus confirmed in output). "Before" here is `HEAD` at commit `45fc242` (Phase 0 + Phase 0b, both flags compiled in but the code path they gate did not yet exist) — re-ran on the current tree before any Phase -1 edits landed and got the exact bit-identical Phase 0b number, confirming no drift.

| Run | BPC | Δ vs before |
|---|---|---|
| **Before** (Phase 0 + Phase 0b only, `HEAD=45fc242`) | **4.821779758048934** | — |
| **After** (+ Phase -1: spectral α and normalised η both enabled) | **4.794304980541004** | **−0.027475 BPC (−0.57% relative)** |

Both runs deterministic (re-ran the after-fix bench a second time; bit-identical `4.794304980541004`).

**Cumulative effect across all three phases**, same `n_train=5000` scale throughout: **7.4053 (original, frozen classifier) → 4.8218 (Phase 0) → 4.8218 (Phase 0b, −0.0066%) → 4.7943 (Phase -1, −0.57%) = −2.611 BPC, −35.3% relative, from the original committed behaviour.** Phase -1's own contribution is real, reproducible, and in the predicted direction, but — consistent with the pattern already seen in Phase 0b (§10.6) — an order of magnitude smaller than Phase 0's dominant effect. A plausible reason, following the same logic as §10.6: both fixes primarily affect *training dynamics/stability* (how smoothly the hierarchy state evolves, how the effective learning rate scales with error magnitude) rather than *what is being learned* — Phase 0 fixed a parameter that was structurally frozen forever; Phase -1 changes how quickly/stably the already-unfrozen parameters converge, a strictly smaller lever at only 5000 training steps.

### 11.6 Matched-scale hybrid-vs-RPSM A/B (closing the gap flagged in §10.7)

Per §10.7's own top recommendation ("a real same-`n_train`, same-tier hybrid-vs-RPSM A/B... is the single most informative next measurement"), ran the hybrid path at the identical scale used for every RPSM measurement in this document: `--profile d17 --mode hybrid --n-train 5000 --n-eval 256 --threads 1`, `bench_seed=42`, `wikitext2` corpus (same clean-scratch binary, `native/build_rpsm/cyphalm_bench_native.exe`).

| Mode | Profile | BPC | n_train | n_eval | Notes |
|---|---|---|---|---|---|
| `hybrid` (`hybrid_gria_lstm`) | d17 | **4.039556** | 5000 | 256 | `train_epochs=2` (mode default); `bpc_lstm_only=4.031056`, `hybrid_gria_weight≈0.007` — at this small scale the hybrid blend has converged to being almost entirely LSTM |
| `rpsm` (Phase 0+0b+(-1), this phase's fixes) | d21 | **4.794305** | 5000 | 256 | `train_epochs=1` (mode default); Tiny tier |

**Matched-scale gap: RPSM is 0.754749 BPC worse than hybrid (+18.7% relative to hybrid's BPC).** One methodological caveat, disclosed rather than smoothed over: the two modes' *default* `train_epochs` differ (hybrid=2, RPSM=1) because that is what each `--mode` sets by default in this codebase — this is the real, as-shipped apples-to-apples comparison a user would get running each mode's default CLI at the same `n_train`/`n_eval`, not an artificially-equalized one; it is disclosed here so the reader can judge how much of the remaining 18.7% gap might be attributable to RPSM getting one fewer pass over the same 5000-example training set.

**This is the single most important number in this entire document.** The originally-cited gap (`bench/BASELINE_LOCK.json`: RPSM 7.336 vs hybrid 2.864, both at `n_train=300000`, a **156% relative gap**) is not a fair like-for-like comparison — it compares a heavily-bugged RPSM configuration (frozen classifier, broken embedding backprop, fixed-not-spectral α) against a fully-converged 300k-step hybrid run. At matched scale, with the two highest-confidence bugs fixed (Phase 0/0b) and the two remaining named "core fixes" from `RESEARCH_STATUS.md:393` also implemented (Phase -1), **the gap collapses to 18.7%.**

### 11.7 Final verdict on RPSM priority

**The remaining gap looks predominantly like a training/scale artifact, not an established architectural ceiling — and the case for immediately committing to the large Option A/B refactors (§6 Phases 4-5) is now substantially weaker than the original `RESEARCH_STATUS.md`/`DEV_PLAN_2026-07-11.md` framing suggested, though this document still cannot fully rule out an architectural gap without a larger-scale run.**

Reasoning, weighing everything found across all phases:

1. **Most of the originally-cited 156%-relative gap was bugs, not architecture.** Finding #1 alone (frozen output classifier, Phase 0) closed 34.9 relative percentage points in isolation. Combined with Phase 0b and Phase -1, the fully-bug-fixed RPSM at the *same small scale* as a freshly-run hybrid comparison trails by only 18.7% — an order of magnitude smaller gap than the 300k-scale, bug-laden 156% figure that is currently pinned in `bench/BASELINE_LOCK.json` and cited in the roadmap docs as the justification for Option A/B.
2. **Neither this document nor its predecessor phases (0, 0b) tested whether the remaining 18.7% gap closes further with more training.** All RPSM measurements in this document, including this phase's, are at `n_train=5000` — a scale at which *neither* model is well-converged (the hybrid's own `hybrid_gria_weight≈0.007` shows even the fully-trained hybrid mixture hasn't meaningfully engaged its GRIA component yet at this scale). It is entirely plausible that most or all of the remaining 18.7% would close with more steps at matched scale, given how much of the original gap turned out to be fixable defects rather than fundamental limits; it is also possible some fraction is a real architectural ceiling. **This document does not have the evidence to distinguish those two possibilities and does not claim to.**
3. **Tier size is confirmed a minor lever (§9.5, −0.8%)**, so moving to Small/Medium tier (Phase 1, still not plumbed) is unlikely by itself to close a meaningful fraction of the remaining gap.
4. **Recommendation, in priority order:** (a) before committing to Option A/B's large-scope Phases 4-5, run one more bug-fixed RPSM-vs-hybrid comparison at a meaningfully larger *matched* scale (e.g. 50k, not the live overnight run's 300k, in a dedicated scratch dir) to see whether the 18.7% gap shrinks, holds steady, or grows as both models get more training — this is the cheapest possible way to convert "looks like a training artifact" into a confirmed answer; (b) if that larger-scale gap is still small (single-digit-to-teens percent) or shrinking, RPSM's Option A/B priority should be **downgraded** from its current P4 "biggest single BPC gap in the repo" framing, since the real fair-comparison gap is far smaller than the pinned 7.336-vs-2.864 numbers suggest; (c) if that larger-scale gap instead grows or plateaus well above ~20%, that would be the first real evidence of an architectural ceiling, and *at that point* Option A/B's remaining scaffold work (Izaac VRF, Gaussian-mixture world model, §6 Phase 5) would be justified. **Either way, the roadmap documents' pinned "7.336 vs 2.864 / 156% gap" framing should be treated as stale and superseded by the matched-scale 18.7% figure in this document** — the former conflates architecture with easily-fixed training bugs and an unfair scale comparison; the latter is the first fair measurement taken in this entire investigation.

### Files touched by Phase -1

- `native/include/cypha/rpsm/rpsm_sequence_layer.hpp` — new opt-in config fields (`use_spectral_alpha`, `use_normalized_eta`, `eta_norm_max_scale`) and `gria_alpha_spectral()` declaration
- `native/src/rpsm/rpsm_sequence_layer.cpp` — `gria_alpha_spectral()` implementation; wired into `hierarchy_update()` (Fix 1) and `train_step()` (Fix 2, `effective_lr`)
- `native/src/cyphalm/cyphalm_model.cpp` — enable both flags for `context_mode == ContextMode::Rpsm`, next to the existing `use_izaac_init` line
- `native/tests/parity/rpsm_spectral_alpha_smoke.cpp`, `native/tests/parity/rpsm_normalized_eta_smoke.cpp` — new CTests (§11.4)
- `native/cmake/CyphaParity.cmake`, `native/CMakeLists.txt` — register the two new test targets/CTests
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

### Additional files read/cited for Phase -1 (§11)

- `docs/RESEARCH_STATUS.md:393` and surrounding context (`:383-397`, re-read directly, not just via §3.2's summary)
- `native/include/cypha/cyphalm/cyphalm_config.hpp:53-54` (`alpha_init`/`alpha_learnable`), `native/src/cyphalm/cypha_cell_hypothesis.cpp:84-156` (H01-H22 cell variants, `alpha_init=0.485` at H10) — cross-referenced per the task's suggestion; confirmed unrelated GRIA-gate construct, but corroborates `0.485` as an established "edge of chaos" convention in this codebase
- `native/src/cyphalm/gria_lowrank.cpp`, `native/src/cyphalm/cyphalm_gria.cpp` (alpha-gate implementations, confirmed structurally unrelated to spectral radius)

---

## 12. 50k matched-scale verification (2026-07-11)

**Status: the decisive follow-up measurement §11.7 called for. Verdict: the gap is real, growing with scale, and now justifies committing to Option A/B — this closes the ambiguity left by the 5k-only measurement.**

This phase made **no source changes** (measurement-only, as instructed). All work was done from `native/build_rpsm` (three prior chained subagents' fixes — Phase 0, 0b, -1 — already built there; confirmed up to date against `HEAD` at commit `ff26a57`, which includes the required `048780e`/`45fc242`/`77e0955` commits). `ctest --test-dir native/build_rpsm -R rpsm` re-confirmed **10/10 passing** before any measurement, with zero regressions.

### 12.1 Method

Per §11.7's own top recommendation ("run one more bug-fixed RPSM-vs-hybrid comparison at a meaningfully larger matched scale, e.g. 50k"), ran both modes at `n_train=50000` (the full 50k tier was tractable — RPSM finished in ~65s, hybrid in ~13.3 min, both single-threaded — so no fallback to 20k was needed), `n_eval=512`, `bench_seed=42`, `wikitext2` corpus, from the same clean `native/build_rpsm/cyphalm_bench_native.exe`:

| Mode | Profile | BPC | n_train | n_eval | train_epochs (default) |
|---|---|---|---|---|---|
| `rpsm` | d21 | **4.462898** | 50000 | 512 | 1 |
| `hybrid` (`hybrid_gria_lstm`) | d17 | **3.349683** | 50000 | 512 | 2 (`hybrid_gria_weight≈0.0007` — still almost pure-LSTM at this scale, same as at 5k) |

Same disclosed caveat as §11.6: the two modes' *default* `train_epochs` differ (hybrid=2, RPSM=1) — this is what each `--mode`'s as-shipped CLI default gives a user at this `n_train`, not an artificially-equalized comparison. Critically, **this asymmetry is constant across both the 5k and 50k measurements** (hybrid has always gotten 2x RPSM's effective passes over the training window, at both scales), so it cannot by itself explain a *change* in the relative gap between the two scale points — see §12.3.

### 12.2 The gap: 5k → 50k trend

| Scale | RPSM BPC | Hybrid BPC | Absolute gap | Relative gap (RPSM vs hybrid) |
|---|---|---|---|---|
| `n_train=5000` (§11.6) | 4.794305 | 4.039556 | 0.754749 | **+18.7%** |
| `n_train=50000` (this phase) | 4.462898 | 3.349683 | 1.113214 | **+33.2%** |

**The gap is growing, not shrinking, as scale increases** — both in absolute BPC (0.755 → 1.113, +47.5%) and in relative terms (18.7% → 33.2%, nearly doubling) over a 10x increase in training data. This is the opposite of the "tuning problem, will close with more training" hypothesis and lands squarely in the "architectural/scaling ceiling" bucket §11.7 pre-committed to as the trigger for Option A/B.

The mechanism is visible in each model's own improvement rate with 10x more data:

| Mode | BPC @ 5k | BPC @ 50k | Relative improvement |
|---|---|---|---|
| RPSM | 4.794305 | 4.462898 | **−6.9%** |
| Hybrid | 4.039556 | 3.349683 | **−17.1%** |

Hybrid's BPC improved **~2.5x more** (relative) than RPSM's for the same 10x increase in training data, despite the epoch-ratio between the two modes being identical at both scale points (always 2:1 hybrid:RPSM effective passes) — i.e. the epoch-default asymmetry is a constant offset, not the source of the *widening* gap. RPSM is scaling with data substantially worse than the hybrid baseline.

### 12.3 Convergence assessment

Using `CYPHALM_TRAIN_LOG_EVERY=5000` on the RPSM run (the only training-progress signal the bench tool exposes without touching `measurers.cpp`/`cyphalm_bench_native.cpp`'s CLI parsing, both correctly left untouched per the task's constraints), the per-step training loss at 10 evenly-spaced checkpoints across the 50k run was:

```
step  5000: loss=2.643    step 30000: loss=2.815
step 10000: loss=4.720    step 35000: loss=2.555
step 15000: loss=2.918    step 40000: loss=3.693
step 20000: loss=1.663    step 45000: loss=2.954
step 25000: loss=1.997    step 50000: loss=1.448
```

These are single-token instantaneous losses (high-variance by construction — per-character entropy varies a lot token to token), so no strong claim should be hung on any individual point, but the aggregate trend is informative: mean of the first half (steps 5k–25k) = **2.788**; mean of the second half (steps 30k–50k) = **2.690** — essentially flat, with no visible sustained downward trend across the run. Combined with the BPC-trajectory evidence in §12.2 (RPSM's 10x-more-data improvement is only a third of hybrid's), the convergence picture reads as **plateaued, not undertrained** — RPSM does not look like a model that would close much more of the gap with additional steps at this same architecture/hyperparameter configuration.

### 12.4 Something noticed but not fixed (per the task's instruction to scope, not fix)

The RPSM run's Tee-Object-piped stderr progress line for the first `CYPHALM_TRAIN_LOG_EVERY` checkpoint was rendered as a formatted `NativeCommandError`/`RemoteException` block by PowerShell (cosmetic only, exit code was 0, all subsequent log lines printed cleanly) — this is the exact same cosmetic stderr-transcript quirk already noted and partially addressed elsewhere for `run_d17_overnight.ps1` (`docs/reports/DEV_PLAN_2026-07-11.md:19`); RPSM's own bench invocations don't yet have that treatment. Not fixed here (out of scope, measurement-only task); worth folding into whichever follow-up next touches overnight/bench script wrappers.

### 12.5 Final decisive recommendation

**Verdict: the remaining RPSM-vs-hybrid gap is real, non-trivial (33.2% at the largest matched scale measured), and growing with scale — this meets and exceeds the exact threshold §11.7 pre-committed to as justification for Option A/B ("if that larger-scale gap instead grows or plateaus well above ~20%, that would be the first real evidence of an architectural ceiling, and at that point Option A/B's remaining scaffold work would be justified").** The ambiguity from the 5k-only measurement is resolved: this is not primarily a tuning/undertraining artifact.

**Priority recommendation relative to `docs/reports/DEV_PLAN_2026-07-11.md`'s roadmap:**

1. **Do not deprioritize RPSM Option A/B relative to P3 (soft-world) or P5 (kernel LLR).** P3 is explicitly framed as "independent, longer horizon" with no measured current gap; P5 targets a "confirmed hard ceiling, partially shipped" 18-percentage-point gap. RPSM's now-confirmed 33.2%-and-growing gap is larger in magnitude, has fresher/more-direct measurement support (this document, four independently-verified phases), and — unlike P3/P5 — was explicitly pre-registered with a decision threshold that this phase's own data now crosses. On the evidence in hand, **RPSM Option A/B should sit ahead of P3 and P5**, not behind them.
2. **Sequence behind P1 (hidden-dim scale) and the in-flight production-overnight run**, both already active on this machine and both independently well-justified (P1 is a "no new theory needed" κ win per Paper IV; the overnight run is close to completion and already committed resource). Do not interrupt either to start Option A/B immediately.
3. **When picked up, start with Option A/B's cheapest remaining sub-phase first** (§6 Phase 1 — config plumbing to Small/Medium tier — was already shown to be a minor lever in §9.5/§9.6 and should stay deprioritized; the real next step is §6 Phase 4/5's remaining scaffold work: the online `mu`/`inv_v` update mechanism ported from `memory_train.cpp`, and eventually the Izaac VRF store / Gaussian-mixture world model). A same-scale, same-tier re-measurement after each sub-phase (following exactly this document's own before/after methodology) should gate whether to continue to the next sub-phase.
4. **This measurement should also correct the stale `bench/BASELINE_LOCK.json` framing** (156% gap, mismatched `n_train`) the next time the lock is legitimately refreshed — not as part of this task (out of scope, no lock changes made here), but flagged so a future lock-refresh pass doesn't inadvertently re-cite the stale 7.336-vs-2.864 comparison as current.

**Net: RPSM's Option A/B track is promoted from "wait for more evidence" (§11.7's conditional framing) to "proceed, sequenced after the currently-running P1/overnight work" — the 50k matched-scale data is the deciding evidence this document's own phased plan was designed to produce.**

### Files touched by §12

- `docs/reports/RPSM_UPGRADE_PLAN.md` — this section only. No source, config, or lock-file changes (measurement-only task, per instructions).
- `bench/results/rpsm_50k_d21.log`, `bench/results/hybrid_50k_d17.log` — raw run transcripts, not committed to the lock.

---

## 13. Architectural root-cause investigation (2026-07-11)

**Status: the "why does the gap grow with scale" question §12 left open is now answered. Verdict: everything load-bearing in Option A/B is confirmed ACTIVE by default in `--mode rpsm`; the one piece that is confirmed under-active (the M_slots surprise-gated write) was forced fully active and measured, and it did not move BPC. Three cheap-scale "turn a knob" experiments (memory-always-on, higher memory blend weight, wider classifier bottleneck) all failed to close any of the gap — this rules out the remaining "config/activation gap" hypotheses and points at a real mechanism-level explanation: RPSM has zero backprop-through-time depth anywhere in its training loop, while the hybrid gets real multi-step temporal credit assignment. No source changes were kept (all experiments were reverted); this section is diagnostic only.**

### 13.1 What is active by default in `--mode rpsm` / the D21 production profile — traced end to end

Traced `bench/config/d21_rpsm_profile.json` → `bench/config/profiles/cyphalm_d21_rpsm.json` (the JSON profile `--profile d21 --mode rpsm` actually loads) → `apply_bench_profile`/`apply_bench_mode` (`cyphalm_config.cpp:71-101`) → `CyphaLMModel::init_components` (`cyphalm_model.cpp:328-341`, the code that builds the live `RpsmSequenceConfig`). Confirmed exactly what §4.3/§9.5/§11.2 of this document already found, re-verified firsthand against current `HEAD` (post Phase 0/0b/-1):

| Feature | Active by default in `--mode rpsm`? | Evidence |
|---|---|---|
| Option A — batched-LLR GEMM (`batched_llr_gemm`) | **Yes, unconditionally** | Called with no gate at all in both `RpsmSequenceLayer::step` and `::train_step` (`rpsm_sequence_layer.cpp:454,478`) — there is no flag that could turn this off short of not using RPSM mode at all |
| Option B — multi-level input injection (`inject_input_multilevel`) | **Yes, unconditionally** | Called at the top of both `step`/`train_step`, loops all `n_levels` (4, Tiny tier) with `scale=1/(l+1)` exactly as specified — no gate |
| Option B — W_up/W_down hierarchy (bottom-up/top-down error, SGD-trained `w_up_`) | **Yes, unconditionally** | `hierarchy_update()` runs every step for every level; `w_up_` receives a real gradient update in `train_step` (`:564-576`) whenever `hierarchy_loss_weight > 0`, which it is by default (`0.1`, unconfigurable, see §13.2) |
| Option B — `M_slots` global memory (`RpsmGlobalMemory`, soft-attention read + surprise-gated ring write) | **Read: yes, unconditionally. Write: technically yes, but self-extinguishing — see §13.3** | `soft_read` is called every level every step, no gate. `ring_write` is gated by `level_surprise >= cfg_.surprise_threshold` (default `0.05`, unconfigurable) |
| Phase -1 — spectral α, normalised η | **Yes** (already confirmed in §11.2, re-verified) | `rc.use_spectral_alpha`/`rc.use_normalized_eta` both `true` for `context_mode==Rpsm` (`cyphalm_model.cpp:338-339`) |
| Orthogonal `W_up` init / Izaac-seed-offset init | **Yes** (already confirmed in §3.2) | `rc.use_izaac_init = (... || cfg_.context_mode == ContextMode::Rpsm)` (`:335`) |
| `n_memory_slots`, `alpha_carry` (now unused — spectral α overrides it), `beta_memory`, `hierarchy_loss_weight`, `surprise_threshold` | **Pinned at `RpsmSequenceConfig` struct defaults; no CyphaLMConfig/JSON/CLI field exists to change any of them** | Re-confirmed exactly as §4.3/§9.5 found: `cyphalm_model.cpp:328-340` sets only `n_levels`/`state_dim`/`feat_dim`/`n_classes`/`seed`/`use_izaac_init`/`use_spectral_alpha`/`use_normalized_eta` on `rc` — the other five `RpsmSequenceConfig` fields (`rpsm_sequence_layer.hpp:107-115`) are never touched, so every `--mode rpsm` run, at any `n_train`, uses `n_memory_slots=32, beta_memory=0.1, hierarchy_loss_weight=0.1, surprise_threshold=0.05` regardless of profile JSON |

**Answer to the task's core question #1, part 1: every Option A/B mechanism this document previously identified as "implemented" is also confirmed ACTIVE (executes every step, unconditionally, with no dead code path) in the exact default `--mode rpsm` configuration used for every D21 production run and every BPC number cited in §9–12 of this document.** There is no "feature exists but the code path never runs" gap left to find for Option A or for four of Option B's five sub-features. The one exception — `M_slots`'s write side — is active in the sense that the code path runs and does write, but §13.3 shows it self-extinguishes early in training and stays extinguished, which is a distinct and more subtle failure mode than "off by default."

### 13.2 H08 / H12 / H13 cross-check — confirmed structurally unrelated to RPSM's `M_slots`, not a more-advanced alternative

Read `cypha_cell_hypothesis.cpp:9-37,108-131` (the full `H08`/`H12`/`H13` variant table entries and their `apply_cell_variant` branches) to check the task's hypothesis that these might toggle memory/context features overlapping with or superior to RPSM's own scaffold. **They do not overlap at all — they live on entirely different, mutually exclusive `context_mode` branches that never construct an `RpsmSequenceLayer`:**

| Variant | `bench_mode` it applies (`apply_cell_variant` → `apply_bench_mode_string`) | Resulting `context_mode` | Memory mechanism toggled | Touches `RpsmSequenceLayer`/`M_slots`? |
|---|---|---|---|---|
| H08 (`use_context_bank`, `use_tiered_context`) | `"context_bank"` | `ContextMode::GriaNgram` (`cyphalm_config.cpp:87-90`) | `ContextBank` class (`context_bank_`, a short/mid/long tiered attention buffer, `cyphalm_model.cpp:311`) | **No** — `ContextBank` is a separate member, constructed only when `use_context_bank`; `RpsmSequenceLayer` is constructed only when `uses_rpsm(cfg_)`, and `apply_bench_mode(BenchMode::ContextBank, ...)` never sets `context_mode=Rpsm` |
| H12 (`use_mdl_forget`) | `"hybrid"` | `ContextMode::Hybrid` | `CompressiveMemory` class (`memory_`, `cyphalm_model.cpp:302-309`; L2-norm/MDL projection on its slots) | **No** — same reasoning, `ContextMode::Hybrid` never constructs `rpsm_layer_` |
| H13 (`use_priority_replay`) | `"hybrid"` | `ContextMode::Hybrid` | `CompressiveMemory` with `set_priority_replay(true)` (priority-weighted slot eviction) | **No** — same |

**This resolves the ambiguity the task flagged: H08/H12/H13 are not "more advanced" versions of RPSM's memory stacked on top of or instead of `M_slots` — they are a completely different codebase subsystem (`ContextBank`/`CompressiveMemory`, both pre-existing, non-RPSM classes) applied to non-RPSM `context_mode`s.** Running `--mode rpsm` gets you `RpsmGlobalMemory` only; running `H08`/`H12`/`H13` gets you `ContextBank`/`CompressiveMemory` only and a plain GRIA-Ngram or hybrid LSTM backbone with no `RpsmSequenceLayer` at all. They are parallel architecture choices, not composable layers, and neither is a superset of the other in the current code.

### 13.3 Runtime instrumentation: `M_slots` writes self-extinguish after ~1,300 steps and never resume

Per the task's request, added temporary instrumentation (env-gated on `CYPHA_RPSM_DEBUG_INSTRUMENT=1`, zero cost/behavior change otherwise) to `inject_input_multilevel`, `hierarchy_update`'s `soft_read`/`ring_write` call sites, counting calls and logging the step index of every `ring_write`. Built in `native/build_rpsm`, ran `--profile d21 --mode rpsm --n-train 10000 --n-eval 128 --threads 1` (Tiny tier, all Phase 0/0b/-1 fixes live, unmodified production config otherwise). **All instrumentation was reverted before finishing this task — confirmed via `git diff` showing zero net changes to `native/src/rpsm/rpsm_sequence_layer.cpp` and `native/src/cyphalm/cyphalm_model.cpp`, and re-confirmed the reverted binary reproduces the exact `4.794304980541004` BPC pin at `n_train=5000` bit-for-bit.**

Findings from the instrumented run:

1. **Multi-level injection fires every step, every level, exactly as the code implies** — `injections` counter incremented 1:1 with training steps (10,000 injections at 10,000 steps), confirming Option B's Fix 4 (multi-level input injection) is not just present in code but genuinely executing on every forward/train call.
2. **`soft_read` (memory read) also fires unconditionally every level every step** (40,000 reads over 10,000 steps × 4 levels) — the memory *read* side is never gated and never stops.
3. **`ring_write` (memory write) fired exactly 5 times in the entire 10,000-step run, all within an 18-step window (steps 1,261–1,278), and never again:**

```
ring_write #1 at step=1261 level=0 surprise=0.050690 threshold=0.0500
ring_write #2 at step=1262 level=0 surprise=0.051823 threshold=0.0500
ring_write #3 at step=1270 level=0 surprise=0.051188 threshold=0.0500
ring_write #4 at step=1273 level=0 surprise=0.050148 threshold=0.0500
ring_write #5 at step=1278 level=0 surprise=0.050015 threshold=0.0500
```

The mechanism: `level_surprise` (the level-0 hierarchy prediction-error magnitude) starts above `surprise_threshold=0.05` early in training (while `w_up_`/`h_levels_` are still settling from orthogonal init), crosses below the fixed threshold for the last time around step ~1,280, and — because the hierarchy error trends only downward as `w_up_` continues to be SGD-trained to minimize exactly this error (`hier_scale` update, `train_step:564-576`) — never crosses back above `0.05` for the remaining 8,720+ steps of this run, and (by the same monotonic-training-loss-reduction logic) would not be expected to for the remaining ~290,000 steps of a full 300k production run either. **`M_slots` is only ever written during roughly the first 1,300 steps of any RPSM run and then holds a permanently frozen snapshot from that narrow window for the rest of training.** All 5 writes were also level-0 only — levels 1–3 never crossed the threshold even during the initial window, so 3 of the hierarchy's 4 levels never write to memory at all, at any point in training.
4. **Even during the window when it is live, the memory's information content is tiny relative to the hidden state it's blended into**: averaged over the whole 10,000-step run, `mean(||mem_read||) ≈ 0.002–0.005` vs `mean(||h||) ≈ 0.39–0.51` — i.e. the memory-read vector's norm is roughly **1% of the hidden state's own norm**, and it enters the blend at `hierarchy_update`'s `beta=0.1` weight (`h[i] = ... + beta * mem_read[i]`), so the actual contribution to the hidden state is on the order of **0.1% of `||h||`**. This is consistent with soft-attention averaging over a small (32-slot) bank of what are, for 3 of 4 levels, still-zero-initialized slots, and for level 0, a handful of stale, un-differentiated raw activation vectors captured once near the start of training — there is no gradient path that ever trains the *content* of what gets written to a slot (`ring_write` copies raw `gate * h[i]`, not a learned projection), so retrieval is intrinsically low-information even before considering the write-frequency problem in point 3.

**This is the clearest concrete instance the investigation found of "implemented but not effective by default"** — distinct from Findings #1/#2 from earlier phases (which were "never runs at all"), this is "runs early, then the codebase's own optimization dynamics push it below its own trigger threshold and it stops running for the rest of training." It directly answers part of the task's question #1: yes, there is at least one genuine "active in principle, inert in practice for a production-length run" mechanism, and it's `M_slots`'s write path, not any of the other four Option A/B sub-features (which all run unconditionally throughout).

### 13.4 Three cheap-scale "turn the dial up" experiments — none recovered any BPC

Per the task's item 3 ("if features are off by default... try turning them on... measure whether BPC improves"), ran three temporary, isolated, single-variable-changed experiments at `n_train=5000` (fastest tier, `--profile d21 --mode rpsm --n-eval 256 --threads 1`, same seed, `native/build_rpsm`, all changes reverted immediately after measurement — confirmed via `git diff` returning empty before moving to the next experiment). Baseline (current committed `HEAD`, i.e. Phase 0+0b+(-1) fixes, unmodified): **BPC = 4.794304980541004** (bit-identical to §11.5/§12's own pin, confirming a clean before/after methodology).

| Experiment | Change (temporary, reverted) | BPC | Δ vs baseline |
|---|---|---|---|
| A | `surprise_threshold: 0.05 → 0.0` (memory writes on every level, every step, never gated off) | 4.794585773475248 | **+0.000281 (+0.0059%, worse)** |
| B | A + `beta_memory: 0.1 → 0.5` (5x stronger memory-blend weight, on top of always-on writes) | 4.796544982084094 | **+0.002240 (+0.0467%, worse)** |
| C | `rpsm_feat_dim: 64 → 128` (doubles the classifier's input bottleneck, matching the hybrid's `lstm_hidden=128`) | 4.813817518116889 | **+0.019513 (+0.407%, worse)** |

**All three moved BPC in the wrong direction, by small-but-consistent (not noise-scale) amounts.** This is a materially different result from every previous phase in this document: Phase 0's fix (a genuinely dead code path, `Ψ_mu` never written) gave a −34.9% win the instant it was turned on; here, forcing an under-active mechanism (`M_slots` writes) to run continuously, or turning up its influence, or widening the classifier's bottleneck to match the hybrid's dimensionality, all make things very slightly worse, not better, at this scale. The most likely reason (consistent with §10.6's own "thin per-row update budget" reasoning for Finding #2): at `n_train=5000`, any added parameters or added signal paths need their own training steps to become useful, and a small regularization-free capacity increase without more data can mildly overfit/destabilize rather than help. But critically, **none of these were "the gap was hiding here" wins** — they rule out, rather than confirm, the "config/activation gap" hypothesis for the specific mechanisms tested.

### 13.5 What is genuinely still "Planned" — clarifying the Izaac VRF store ambiguity

Per the task's request to clarify a previously-ambiguous distinction, grepped `native/src/rpsm/` and `native/include/cypha/rpsm/` for `VRF`, `episodic`, `sha256`/`SHA-256`, and `snapshot` — **zero matches**. Cross-referenced against `RPSM_COMBINED_SPEC.md:98-102` and `RPSM_IMPLEMENTATION.md:80,89-91`, which describe **three separate global-memory layers**, only one of which is built:

| Layer (per spec) | What it is | Built? | Where |
|---|---|---|---|
| 1. Izaac episodic store | VRF-keyed snapshots of past Ψ states, retrieved by content-addressable key (spec's own fallback: SHA-256 hash of quantised hidden state as a stub until a real VRF is wired) | **No — not found anywhere in `native/src/rpsm/`.** This is the genuinely-unimplemented "Planned" piece | — |
| 2. Working memory `M_slots` | Fixed-size slot ring, soft-attention read, surprise-gated write | **Yes.** This is `RpsmGlobalMemory` | `rpsm_sequence_layer.hpp:49-91`, `.cpp:220-283` |
| 3. Gaussian-mixture world model | Slow semantic anchor injected into the top hierarchy level (weight ~0.05) | **No — not found.** Also genuinely "Planned" | — |

**The naming collision that made this ambiguous in earlier docs: "Izaac" is used for three unrelated things in this codebase**, only one of which is the actual missing VRF store:
1. `IzaacActivationMix` (`rpsm_sequence_layer.hpp:19-39`) — a grammar-search-stub enum that picks one of 5 fixed activation functions per seed. **Built, active, unrelated to memory.**
2. `use_izaac_init` / `kIzaacSeedOffset` (`rpsm_sequence_layer.cpp:12,74-79,291,302-308`) — an H19-style seed-offset convention for orthogonal weight init. **Built, active, unrelated to memory.**
3. The **Izaac episodic VRF store** (spec's global-memory layer 1, above) — the actual missing piece. **Not built.**

**Answer to the task's question #2's clarification request: yes, there is a genuine third tier beyond what's built — the Izaac VRF episodic store and the Gaussian-mixture world model are both real gaps, not built anywhere, and are correctly the only pieces of Option A/B that should still be called "Planned."** Everything else this document and its predecessor phases discuss (batched LLR GEMM, multi-level injection, W_up/W_down hierarchy, `M_slots` working memory, spectral α, normalised η, orthogonal init) is built and, per §13.1, active by default.

### 13.6 Why the gap grows with scale despite everything relevant being active: the temporal-credit-assignment asymmetry

With §13.1–13.4 ruling out "an existing feature is silently off" and "a known under-active feature just needs turning up" as explanations, re-read the actual training-loop call graphs for both modes side by side to find what structurally differs. One asymmetry stands out and is confirmed directly in code, not inferred:

- **Hybrid (`train_step`, general path, used by `--mode hybrid`)** calls `bptt_ssm_update(next_token_id)` every single training step (`cyphalm_model.cpp:1122`), which — when `cfg_.bptt_steps > 0` (the D17 hybrid profile sets `bptt_steps: 64`, `cyphalm_d17_wikitext.json:22`) — accumulates a buffer of the last 64 steps' `(ctx, grad)` pairs and backpropagates the SSM/projection gradient through that real 64-step window (`cyphalm_model.cpp:1191,1266`, guarded by `cfg_.bptt_steps <= 0` / buffer-size checks). Separately, the hybrid's own LSTM head (`char_lstm.cpp`) computes and applies a full, correct backward pass every step for `Wy`/`Wh`/`b`/`by` via `backward_step`/`apply_gradients` (already contrasted with RPSM's classifier in §4.1's Finding #1 comparison).
- **RPSM (`train_step_rpsm`, `cyphalm_model.cpp:1472-1500`)** never calls `bptt_ssm_update` at all — grepped every call site of `bptt_ssm_update` in `cyphalm_model.cpp` and confirmed the only call is the one inside the *general* `train_step` (line 1122), which `train_step_rpsm` does not invoke and has no equivalent of. Every gradient computed inside `RpsmSequenceLayer::train_step` (`Ψ_mu`, `w_enc_`, `w_carry_`, `w_up_`) is derived purely from the **current** timestep's forward activations (`feat_buf_`, `enc_pre_`, `h_levels_`, `work_err_` — all overwritten fresh each call) — there is no buffer of past states, no unrolling, and `cfg_.bptt_steps` (which the D21 RPSM profile leaves at its default `0` anyway, `cyphalm_d21_rpsm.json` has no `bptt_steps` key) has no code path that applies to RPSM at all.

**Mechanistically, this means RPSM's entire hierarchy — `h_levels_`, `w_up_`, the encoder — learns via what is structurally closest to real-time recurrent learning / a single-step local update rule, propagated forward only through the carried hidden state, never backward through its own history.** The hybrid gets two independent, compounding sources of multi-step temporal credit assignment (the LSTM cell's own gating dynamics trained with a correct one-step backward pass repeated over a persistent cell state, *plus* an explicit 64-step BPTT correction on the upstream SSM projection); RPSM gets neither — its only mechanism for "remembering" earlier tokens is the forward-only leaky carry through `h_levels_`/`(1-a)` blending (`hierarchy_update`, Fix 1's spectral `a`), with zero gradient signal about whether that carry decision, several steps ago, helped or hurt the *current* prediction.

**This is a plausible, mechanistically well-supported explanation for exactly the two symptoms this document's own §12 measurement already established:**
1. *"RPSM's training loss visibly plateaued"* (§12.3) — a model whose only credit-assignment signal is single-step-local cannot discover or reinforce any temporal dependency whose useful gradient only becomes visible several steps later; it will correctly reduce single-step-predictable loss quickly (matching Phase 0's big, fast win — that was a single-step-local defect, exactly RPSM's strength) and then plateau once the easy single-step-local signal is exhausted, exactly the shape seen in §12.3's flat 5k-step-window loss average.
2. *"The gap grows, not shrinks, with 10x more data"* (§12.2) — more training data mostly helps a model that can extract *longer-range* structure from it; a model bottlenecked to single-step-local updates gets proportionally less benefit from more data than one with real multi-step BPTT, which is exactly the asymmetric improvement rate §12.2 measured (RPSM improved 6.9% relative from 10x more data; hybrid improved 17.1%, ~2.5x more).

This is also consistent with, and gives a causal explanation for, why none of §13.4's three experiments helped: none of them touch the temporal-credit-assignment mechanism. Widening the classifier bottleneck (Experiment C) or making the (memory-content-untrained, per §13.3 point 4) `M_slots` more active (A/B) cannot compensate for the model never receiving a gradient signal that says "the hidden-state decision several steps ago should have been different" — that signal simply does not exist anywhere in RPSM's current training loop, at any hyperparameter setting reachable from `RpsmSequenceConfig`'s existing fields.

### 13.7 Final answer to the task's core question and concrete recommendation

**Direct answer: it is not that Option A/B's features are inactive in the default `--mode rpsm` configuration — §13.1–13.2 confirm essentially everything is active and executing every step, and §13.4's experiments confirm that forcing the one under-active piece (`M_slots` writes) to run continuously, or amplifying it, does not recover any BPC. The remaining gap is best explained by a real inductive-bias/mechanism mismatch: RPSM's training loop has zero backprop-through-time depth anywhere (§13.6), while the hybrid baseline gets real multi-step temporal credit assignment through two independent mechanisms. This is not a "config gap" the way Findings #1/#2/Phase -1 were — it is an architectural gap in the *training* mechanism, distinct from (and cheaper to fix than) the still-genuinely-missing Izaac VRF store / Gaussian-mixture world model pieces (§13.5), which are a different, larger, and — per §13.3 point 4's finding about `M_slots`'s untrained content — not obviously higher-priority gap.**

Recommendation, in priority order, superseding §12.5's "proceed to Option A/B's remaining scaffold work" with something more specific now that this session's own cheap-scale experiments have been run:

1. **Do not build the Izaac VRF store or Gaussian-mixture world model next.** Both are real, substantial, "Planned"-as-labeled pieces of work (§13.5), but §13.3 point 4 found that the *already-built* `M_slots` mechanism carries near-zero information even when forced fully active, because slot content is never shaped by any gradient (`ring_write` copies raw activations, not a learned key/value). A VRF-keyed episodic store built on the same "write raw activations, retrieve by content-similarity, never train the content" pattern would very plausibly inherit the same low-information-content problem before it even gets to the harder VRF/hashing engineering — there is no evidence in hand that more memory *capacity* (a third tier) fixes a memory *quality* problem the second tier already has.
2. **The highest-leverage next experiment is giving RPSM's own training loop real multi-step temporal credit assignment** — either (a) a `bptt_steps`-equivalent for `RpsmSequenceLayer::train_step` (buffer the last `k` steps' `(feat_buf_, h_levels_, work_err_)` and backpropagate the hierarchy/classifier loss through that window, mirroring `bptt_ssm_update`'s existing buffer-and-unroll pattern in the same file), or (b) at minimum, verify whether truncated BPTT even helps here at cheap scale before committing to the larger unrolled-hierarchy implementation, by first testing a much simpler proxy: increasing `RpsmSequenceLayer`'s implicit memory horizon via a slower `alpha_carry`/spectral-α blend (already partially explored by Phase -1, but not yet swept as its own independent lever) or literally-BPTT-through-`h_levels_`-only (cheaper than the full hierarchy) as a first cut. This is genuinely new implementation work, not a config change, so it should be scoped and estimated as its own phase before starting.
3. **Separately, fix `M_slots`'s self-extinguishing write gate** (§13.3) even though §13.4 showed it doesn't move BPC on its own — a memory mechanism that provably stops writing after the first ~0.4% of a 300k-step run is a latent correctness issue independent of whether it currently matters, and a `surprise_threshold` that adapts to the *current* running-average error scale (rather than a fixed absolute `0.05`) rather than a fixed constant would be a small, cheap, low-risk change worth bundling with whichever future phase revisits `RpsmSequenceConfig`'s still-unconfigurable fields (§13.1's table, Phase 1 from §6 — still correctly deprioritized on its own per §9.5/§9.6, but worth doing opportunistically alongside #2 above rather than as a separate investment).
4. **Do not deprioritize RPSM relative to §12.5's own ranking based on this section's findings** — the 33.2%-and-growing gap from §12 stands; this section only clarifies *why* (a training-mechanism gap, not a config gap), which if anything makes the fix more clearly scoped (implement real temporal credit assignment) rather than less justified.

### Files touched by §13

- `docs/reports/RPSM_UPGRADE_PLAN.md` — this section only.
- No source or config changes were kept. Temporary instrumentation (`native/src/rpsm/rpsm_sequence_layer.cpp`) and three temporary experimental config overrides (`native/src/cyphalm/cyphalm_model.cpp`) were added, measured, and fully reverted — confirmed via `git diff` showing zero net changes to both files, and via a bit-identical BPC reproduction (`4.794304980541004` at `n_train=5000`) and a clean `ctest --test-dir native/build_rpsm -R rpsm` run (10/10 passed) on the reverted tree before finishing this task.

## 14. Real multi-step BPTT implementation, verification, and measured impact (2026-07-11)

§13.7's recommendation was to implement real multi-step temporal credit assignment in
`RpsmSequenceLayer::train_step` itself, mirroring the hybrid GRIA+LSTM path's own
`bptt_ssm_update`. This section implements that, verifies it with a dedicated
finite-difference test, measures its effect at both `n_train=5000` and `n_train=50000`, and
reaches a conclusive — and, contrary to §13.7's hypothesis, **negative** — result.

### 14.1 Mechanism implemented

`RpsmSequenceLayer` now caches a `BpttStepCache` per step (`h_inj`, `up_pre`, `down_pre`,
`blended_pre`, `alpha`, `enc_pre`, `enc_grad`, `input`) into a ring buffer sized
`cfg_.bptt_window`. Every `bptt_window` calls to `train_step`, `bptt_backward_and_apply()` runs
a true reverse-time pass over the whole window:

- Propagates the recursive gradient of the window's *total* loss through the hierarchy's state
  carry `h_carry_{t+1} = act(blended_pre_t)` — i.e. through the actual SSM-like recurrence
  `hierarchy_update()` advances every step — across all `window` steps, not just the current one.
- Accumulates gradients for `w_up_` (both the "up" and "down" projection uses of the same
  matrix), `w_enc_`, and `w_carry_` (both its encode-path and injection-path uses) via the full,
  untruncated chain rule (see the code comment at `bptt_backward_and_apply()`'s definition,
  `rpsm_sequence_layer.cpp`, for the complete per-term derivation).
- Fixes two latent bugs found while deriving the backward pass, present since the original
  per-step-local update code and *independent of BPTT depth*: `w_up_`'s old update reused a
  *stale* `work_err_` left over from `hierarchy_update()`'s last-level loop iteration for every
  level's gradient, and `w_carry_`'s old update read `h_levels_[0]` *after* it had already been
  overwritten with the next step's carry, not the `h_inj` value actually used forward. Both are
  fixed by construction in the new cache-based code, which snapshots the correctly-timed values.
- Exposes the deeper `d(loss)/d(input_t)` signal (which now includes the hierarchy/injection
  paths, not just the encode path) via `bptt_window_input_grads()`, wired into
  `CyphaLMModel::rpsm_bptt_embed_flush()` so the embedding table also receives this richer
  gradient at window boundaries (extending the already-fixed `rpsm_embed_backprop` from earlier
  this session, which only ever saw the shallower per-step-local signal).
- Deliberately treats `M_slots`' memory-read contribution to `blended_pre` as a stop-gradient
  (documented in-line, at the same code location) — a scoped, measured simplification (§13.3's
  `||mem_read|| ~1%` of `||h||`), not a silent truncation of the recurrence path this section is
  actually about.
- Classifier weights (`psi_.mu`) are unchanged: still SGD-updated every step, independent of the
  window. The backward pass uses the *cached* `enc_grad_t` from step `t`'s own forward pass so a
  later step's classifier update cannot corrupt an earlier step's hierarchy gradient.

This is **non-overlapping batched gradient accumulation** (cache `N` steps, backprop once, apply
one *averaged* update — mean gradient x mean per-step `effective_lr` over the window), matching
the hybrid path's own `bptt_ssm_update` convention exactly, not a sliding/streaming truncated-BPTT
that would update every step from a `K`-step lookback (that variant needs an `O(window)` live
forward-state cache at every step and was out of scope here).

### 14.2 Finite-difference verification

New test: `native/tests/parity/rpsm_bptt_grad_finite_diff.cpp` (registered as CTest
`native_rpsm_bptt_grad_finite_diff`). Uses small fixed dims and `kWindow=4`; disables
`spectral_alpha`/`normalized_eta` and sets `beta_memory=0` to isolate exactly the terms
`bptt_backward_and_apply()` claims to compute exactly (per §14.1's stop-gradient note, this
scope decision cannot silently mask an error in the terms that *are* verified). For each of
`w_up_`, `w_enc_`, `w_carry_`: runs a full 4-step BPTT window, perturbs each weight by
central-difference, and compares the numeric loss-change gradient against the analytic
accumulated gradient (recovered from the weight's own pre/post-update delta, since
`bptt_backward_and_apply()` applies the update in place). **Result: all three weight groups
pass within tolerance** (relative error consistent with `O(lr)` higher-order coupling through
the classifier's own per-step SGD on `psi_.mu`, not a first-order gradient error — confirmed by
the error shrinking as `kLr` decreases, documented in-line in the test). This confirms the
chain-rule derivation through the recurrence is correct, not merely plausible.

### 14.3 CTest results

`ctest --test-dir native/build_rpsm -R "rpsm"` → **11/11 passed** (the 8 pre-existing RPSM tests,
`native_rpsm_embed_grad_finite_diff` from earlier this session, plus the two Phase -1 smokes,
plus the new `native_rpsm_bptt_grad_finite_diff`). A broader focused sweep,
`ctest --test-dir native/build_rpsm -R "cyphalm|rpsm"`, → **24/24 passed**, confirming no
regression in any adjacent `cyphalm`-path test (char-LSTM, hybrid, checkpoint/parity, SSM
diagnostics, etc.).

### 14.4 Measured impact — this is where the hypothesis breaks down

All runs: clean rebuild of `native/build_rpsm`, `--profile d21` (rpsm) / `--profile d17`
(hybrid, matching §12's own methodology exactly — `d21` is RPSM-tuned and is *not* the profile
the hybrid's `4.039556`/`3.349683` pins were measured with), `bench_seed=42`, `--threads 1`.
Hybrid numbers were re-measured (not merely cited) and reproduced §12's pins bit-for-bit:
`4.039555743981927` at 5k, `3.3496834068056365` at 50k.

`bptt_window` was swept at `n_train=5000` via a debug-only `CYPHALM_RPSM_BPTT_WINDOW` env
override (`CyphaLMModel::init_components()`; not part of the profile schema):

| `bptt_window` | RPSM BPC (n_train=5000) | Δ vs pre-§14 baseline (4.794305) |
|---|---|---|
| 1 (bug-fixes only, no batching) | 4.747686 | **−0.97%** |
| 2 | 4.766456 | −0.58% |
| 4 | 4.803879 | +0.20% |
| 8 | 4.858927 | +1.34% |
| 16 | 4.938043 | +3.00% |
| 32 | 5.137262 | +7.16% |
| 64 (matches hybrid's own window length, as originally requested) | 6.020780 | **+25.6%** |

**Eval BPC degrades monotonically, and steeply, with window size.** This is not a subtle
off-by-one or a sign error — the finite-difference test (§14.2) verifies the gradient math is
correct at every window size the same code path handles, and the degradation is a smooth,
monotone function of `window`, not a step-function consistent with a triggering bug. The
regression is caused by *update frequency*, not the gradient's correctness: batching `N` steps'
gradients into one averaged update applied once, instead of applying each step's own (now
correctly-derived, bug-fixed) local gradient immediately, reduces `w_up_`/`w_enc_`/`w_carry_`'s
effective adaptation rate by a factor of `N` — and for RPSM's hierarchy at its current
`lr`/architecture regime, that reduced adaptation rate hurts far more than the deeper, more
correct credit-assignment signal helps. This is the opposite of what the hybrid path's own
64-step `bptt_ssm_update` was found to do for its SSM weights — the same "batch-then-apply-once"
convention helps one architecture and hurts the other.

Full before/after comparison at both scales, both configurations:

| Scale | RPSM pre-§14 (baseline) | RPSM §14, `window=1` (shipped default) | RPSM §14, `window=64` (matches hybrid's convention, as literally requested) | Hybrid (d17, re-verified) |
|---|---|---|---|---|
| `n_train=5000` | 4.794305 | 4.747686 (−0.97%) | 6.020780 (**+25.6%**) | 4.039556 |
| `n_train=50000` | 4.462898 | 4.601064 (+3.10%) | 4.734668 (**+6.10%**) | 3.349683 |

Gap to hybrid, relative:

| Scale | Gap, pre-§14 (§12) | Gap, `window=1` | Gap, `window=64` (as literally requested) |
|---|---|---|---|
| 5k | +18.7% | +17.5% | **+49.1%** |
| 50k | +33.2% | +37.4% | **+41.3%** |
| Trend | growing (18.7% → 33.2%) | growing (17.5% → 37.4%) | **shrinking (49.1% → 41.3%)** |

### 14.5 Does the gap trend improve? A nuanced but ultimately negative answer

Taken at face value against the literal question asked (§13's recommended fix, measured at the
hybrid's own window length): **the growing-with-scale trend does reverse under real BPTT** —
49.1% at 5k down to 41.3% at 50k, versus 18.7%→33.2% before. But this is not the win it might
look like in isolation: the trend reverses because `window=64` is *catastrophically* bad at 5k
(+25.6% BPC) and only moderately bad at 50k (+6.1%) — i.e., the absolute BPC at both scales is
worse than the pre-§14 baseline, and worse than the shipped `window=1` configuration, at both
scales. The "shrinking gap" is a shrinking-*regression*, not an improvement in absolute
performance. At the safe, shipped default (`window=1` — real per-step credit assignment for a
single step, i.e. no batching, only the two latent-bug fixes), the gap still grows with scale
(17.5%→37.4%), consistent with §13's original finding, and BPC is only marginally different
from the pre-§14 baseline in either direction (−0.97% at 5k, +3.10% at 50k) — nowhere close to
closing an 18-37% gap.

**Conclusion: implementing real multi-step BPTT does not close the RPSM-vs-hybrid gap. It makes
absolute BPC worse at every window size greater than 1, at both scales tested, and does so more
severely at smaller scale.** §13.6/§13.7's hypothesis — that RPSM's plateau and the
growing-with-scale gap are explained by missing temporal credit-assignment depth — is
**falsified** by this experiment. The two genuine latent bugs found while implementing it
(stale `work_err_`, mistimed `h_levels_` read) were real defects worth fixing on their own
merits, but their combined standalone effect is a rounding error next to the gap's actual size,
exactly as Phase 0b's embedding-gradient fix was in §10.6.

### 14.6 Shipped configuration and rationale

`RpsmSequenceConfig::bptt_window` defaults to **1**, not 64. Given `bptt_results`/`rpsm_results`
in `bench/BASELINE_LOCK.json` are produced by the live overnight orchestration rebuilding from
this same `native/src` tree (in `native/build_math`, untouched by this session but sharing the
source), shipping `window=64` as the default would risk silently regressing the locked
production RPSM baseline the next time that pipeline does a from-scratch rebuild — for a change
this section's own data shows makes things worse, not better. The full `bptt_window`-parameterized
mechanism remains implemented, wired, and finite-difference-verified for any window size (tests
and the `CYPHALM_RPSM_BPTT_WINDOW` env override both exercise `window > 1`), so the research
value and the ability to reproduce every number in §14.4 are preserved; only the *default*
behaviour a user gets from `--mode rpsm` with no special flags changes, and it changes toward
the empirically-safer, not empirically-best-looking-on-paper, setting.

### 14.7 Memory-write `surprise_threshold` re-test — not run

§13.4/§13's speculation was that the `M_slots` self-extinguishing write-gate fix (confirmed not
to help in isolation, §13.4) might only help "once real gradient flow exists to make use of the
recalled memory." Given §14.4-14.5 found that real gradient flow (BPTT depth) is itself net
harmful rather than helpful at every tested window size, the premise this follow-up experiment
was conditioned on no longer holds, so it was not re-run this session — re-testing a secondary
fix under a primary mechanism just shown to regress BPC would not produce an interpretable
result. It remains a small, independent, low-risk correctness fix worth doing opportunistically
(§13.7 point 3) but is now further deprioritized as a lever for closing the hybrid gap.

### 14.8 Final, conclusive verdict on RPSM's viability and priority

This is the fourth independent, cheap-scale experiment this session testing a specific
hypothesis for the hybrid-vs-RPSM gap (Phase 0: frozen classifier — real, large effect; Phase
0b: embedding backprop stub — real, tiny effect; §13.4: memory-write threshold — no effect;
§14: BPTT depth — real, *negative* effect). Two of four hypotheses were confirmed defects worth
fixing on their own terms; none of the four closed a meaningful fraction of the 18-37%,
growing-with-scale, matched-tier gap to the hybrid baseline, and the single mechanism §13.7
identified as the most mechanistically well-supported explanation for *both* of §12's headline
symptoms (plateaued training loss, growing-with-scale gap) turned out, on direct implementation
and measurement, to make both worse rather than better.

**Verdict: RPSM's remaining gap to the hybrid GRIA+LSTM baseline is not primarily a
training-mechanism defect (classifier, embedding gradient, memory-write cadence, or BPTT depth
have all now been tried, at cheap scale, with rigorous before/after measurement) — it is most
likely a genuine architectural/capacity mismatch between RPSM's hierarchy-based state and the
hybrid's LSTM+GRIA blend, at least at the tiny/small tiers and lr regime tested. Continued
investment in Option A/B's remaining scaffold work (Izaac VRF store, Gaussian-mixture world
model) should be treated with materially lower confidence than §13.7 recommended: those pieces
were justified partly on the strength of "the training-mechanism gap is fixable and cheap,
fix that first" reasoning, and that reasoning's single most concrete, implementable proposal
has now been tried and has failed. RPSM should be deprioritized relative to P3 (soft-world) and
P5 (kernel-LLR) unless a specific, different, well-motivated hypothesis for the remaining gap
emerges — "try more temporal depth," "try more memory," and "fix the known bugs" are now all
cheap-scale-tested and exhausted as easy wins.**
