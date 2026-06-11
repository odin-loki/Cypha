#pragma once

/// Cypha generative sampler — pure-math port of CyphaDIF generation methods.
///
/// All functions are stateless with respect to the model (const CyphaInferModel&)
/// except ``rollout``, which mutates context and field state.  For testing, pass a
/// copy of the model.
///
/// RNG: ``std::mt19937`` seeded externally.  Random draws follow the pre-drawn
/// variates in the generation parity fixture (z_generate / u_transition) so that
/// fixture-based parity tests do not depend on matching ``std::mt19937`` vs
/// ``numpy.random.PCG64`` sequences.

#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cypha/infer_cpu.hpp"

namespace cypha {

/// Constants matching Cypha.py.
constexpr double kGenMinVar = 1e-4;
constexpr double kGenEps    = 1e-8;

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

/// Gaussian class-conditional samples: h ~ N(mu_k, T²·v₀).
///
/// With ``rejection=false`` (or ``temperature ≤ 1``): i.i.d. draws.
/// With ``rejection=true``  (``temperature > 1``): draw ``max_candidates``
/// per output sample; keep the one with the highest raw LLR for ``label``.
///
/// ``z_override`` — if non-null, shape ``(n × d)`` (no-rejection) or
/// ``(n × max_candidates × d)`` (rejection) row-major standard normals
/// provided by the caller (e.g. parity fixture).  If null, draws from
/// ``rng``.  ``rng`` may be nullptr when ``z_override`` is provided.
std::vector<std::vector<double>> generate_class_gaussian(
    const CyphaInferModel& m,
    const std::string&     label,
    int                    n,
    double                 temperature,
    std::mt19937*          rng,
    bool                   rejection        = true,
    int                    max_candidates   = 16,
    const double*          z_override       = nullptr);

/// Field-conditioned generation: mu₀ is shifted by F_field @ field_h before
/// adding the class delta.  Temperature-scaled Gaussian (no rejection branch).
///
/// ``z_override`` — row-major ``(n × d)`` standard normals; nullptr → draw.
std::vector<std::vector<double>> generate_conditioned(
    const CyphaInferModel& m,
    const std::string&     label,
    int                    n,
    double                 temperature,
    std::mt19937*          rng,
    const double*          z_override = nullptr);

/// Langevin MCMC: gradient walk along ∇ log p(h|k) + isotropic prior.
///
/// ``z_init_override``  — ``(n × d)`` standard normals for initialisation.
/// ``z_noise_override`` — ``(n × n_steps × d)`` standard normals for dynamics.
/// Either pointer may be nullptr to draw from ``rng``.
std::vector<std::vector<double>> generate_langevin(
    const CyphaInferModel& m,
    const std::string&     label,
    int                    n,
    int                    n_steps,
    double                 step_size,
    double                 temperature,
    std::mt19937*          rng,
    const double*          z_init_override  = nullptr,
    const double*          z_noise_override = nullptr);

/// Interpolate two classes and snap to the decision hyperplane.
///
/// ``z_override`` — ``(n × d)`` standard normals.
std::vector<std::vector<double>> generate_boundary(
    const CyphaInferModel& m,
    const std::string&     label_a,
    const std::string&     label_b,
    int                    n,
    double                 alpha,
    double                 temperature,
    std::mt19937*          rng,
    const double*          z_override = nullptr);

/// Sample from the out-of-distribution region (max LLR_k < 0).
///
/// ``z_override`` — ``(n_candidates × d)`` standard normals.
std::vector<std::vector<double>> generate_ood(
    const CyphaInferModel& m,
    int                    n,
    int                    n_candidates,
    std::mt19937*          rng,
    const double*          z_override = nullptr);

/// Fisher-Rao ball constrained sampling around class mean.
///
/// ``z_dir_override`` — ``(n × d)``; ``u_mag_override`` — ``(n,)`` uniforms.
std::vector<std::vector<double>> generate_mdl_ball(
    const CyphaInferModel& m,
    const std::string&     label,
    int                    n,
    double                 radius,
    std::mt19937*          rng,
    const double*          z_dir_override = nullptr,
    const double*          u_mag_override = nullptr);

/// Ancestral sampling: k ~ freq^(1/T), h ~ N(mu_k, v₀).
///
/// ``u_class_override`` — ``(n,)`` uniforms for class selection.
/// ``z_override``        — ``(n × d)`` normals for Gaussian sampling.
struct AncestralSample {
    std::string           label;
    std::vector<double>   h;
};

std::vector<AncestralSample> generate_ancestral(
    const CyphaInferModel& m,
    int                    n,
    double                 temperature,
    std::mt19937*          rng,
    const double*          u_class_override = nullptr,
    const double*          z_override       = nullptr);

// ---------------------------------------------------------------------------
// Context & rollout
// ---------------------------------------------------------------------------

/// Next-label probability distribution using Tier-1 co-occurrence + Tier-2
/// EMA transitions (same formula as Python ``TieredContextBuffer.predict_next``).
///
/// Returns a vector of probabilities in the same order as ``m.labels``.
/// Returns a uniform distribution when ``m.ctx_last_label`` is empty or has
/// no recorded context.
std::vector<double> predict_next_probs(
    const CyphaInferModel& m,
    const std::string&     current_label);

/// Autoregressive rollout: generates a sequence of (label, h) pairs.
///
/// At each step:
///   1. Sample h ~ N(mu_current, T²·v₀) using ``z_generate[step]``
///   2. Append (label, h)
///   3. Record label in context buffer + advance field
///   4. Compute predict_next → mix with uniform at ``exploration`` rate
///   5. Sample next class from mixed distribution using ``u_transition[step]``
///
/// ``z_generate``    — ``(n_steps × d)`` row-major normals; nullptr → draw.
/// ``u_transition``  — ``(n_steps,)`` uniforms; nullptr → draw.
///
/// NOTE: This function MUTATES ``m`` (context + field state).  Pass a copy
/// when the original model state must be preserved.
struct RolloutStep {
    std::string           label;
    std::vector<double>   h;
};

std::vector<RolloutStep> rollout(
    CyphaInferModel&   m,
    const std::string& seed_label,
    int                n_steps,
    double             temperature,
    double             exploration,
    std::mt19937*      rng,
    const double*      z_generate   = nullptr,
    const double*      u_transition = nullptr);

// ---------------------------------------------------------------------------
// Observation-anchored & retrieval-augmented generation
// ---------------------------------------------------------------------------

/// Conditional generation warm-started from a latent observation ``h_obs``.
///
/// Langevin walk anchored to ``h_obs``: ``∇_h log p(h|k) = -(h - mu_k) ⊙ inv_v``,
/// step size ``lr = temperature * 0.05``, noise ``√(2·lr)``.
///
/// ``z_noise_override`` — ``(n × n_steps × d)`` standard normals; nullptr → draw.
std::vector<std::vector<double>> generate_from_observation(
    const CyphaInferModel& m,
    const double*          h_obs,
    const std::string&     label,
    int                    n,
    double                 temperature,
    int                    n_steps,
    std::mt19937*          rng,
    const double*          z_noise_override = nullptr);

/// Retrieval-augmented generation (RAG) on raw inputs.
///
/// 1. ``retrieve_from_x`` → top ``k_neighbors`` database rows
/// 2. Distance-weighted latent centroid among hits (Euclidean to query ``h``)
/// 3. ``generate_from_observation`` from the centroid with the nearest-hit label
///
/// Empty database → ``generate_langevin`` on the inferred query class (Python
/// ``generate_real`` fallback).
///
/// ``z_noise_override`` — ``(n × n_steps × d)`` for the observation Langevin;
/// ``z_init_override`` / ``z_langevin_noise_override`` — used only on empty-db
/// fallback (same layout as ``generate_langevin``).
std::vector<std::vector<double>> generate_retrieval_augmented(
    const CyphaInferModel& m,
    const double*          query_x,
    const double*          database_x,
    int                    n_db,
    int                    input_dim,
    int                    k_neighbors,
    int                    n,
    double                 temperature,
    int                    n_steps,
    const CyphaInferOptions& opt,
    std::mt19937*          rng,
    const double*          z_noise_override            = nullptr,
    const double*          z_init_override               = nullptr,
    const double*          z_langevin_noise_override     = nullptr);

}  // namespace cypha
