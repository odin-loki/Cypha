#include "cypha/cyphalm/cyphalm_math_integration.hpp"

#include <algorithm>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_intelligence_hook.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_completeness.hpp"
#include "cypha/intelligence/profile_guided_loss.hpp"

namespace cypha::cyphalm {

namespace {

cypha::intelligence::ProfileObservation mean_profiler_observation(
    const cypha::intelligence::IntelligenceProfiler& profiler) {
    const auto matrix = profiler.get_profile_matrix();
    cypha::intelligence::ProfileObservation obs;
    obs.alpha = matrix[0][0];
    obs.d_eff = matrix[1][0];
    obs.sigma_branch = matrix[2][0];
    obs.tau = matrix[3][0];
    obs.r_eu = matrix[4][0];
    obs.lipschitz = matrix[5][0];
    obs.calibration = matrix[6][0];
    return obs;
}

nlohmann::json profile_guided_lambdas_to_json(
    const cypha::intelligence::ProfileGuidedLossConfig& pg_cfg) {
    return {{"lambda_alpha", pg_cfg.lambda_alpha},
            {"lambda_d_eff", pg_cfg.lambda_d_eff},
            {"lambda_sigma_branch", pg_cfg.lambda_sigma_branch},
            {"lambda_tau", pg_cfg.lambda_tau},
            {"lambda_r_eu", pg_cfg.lambda_r_eu},
            {"lambda_lipschitz", pg_cfg.lambda_lipschitz},
            {"lambda_calibration", pg_cfg.lambda_calibration}};
}

nlohmann::json stat_deltas_to_json(const cypha::intelligence::ProfileStatDeltas& d) {
    return {{"alpha", d.alpha},
            {"d_eff", d.d_eff},
            {"sigma_branch", d.sigma_branch},
            {"tau", d.tau},
            {"r_eu", d.r_eu},
            {"lipschitz", d.lipschitz},
            {"calibration", d.calibration}};
}

cypha::intelligence::AdaptiveNavigationOptions adaptive_opts_from(const CyphaLMConfig& cfg) {
    cypha::intelligence::AdaptiveNavigationOptions opts;
    opts.use_adaptive_lambdas = cfg.use_adaptive_navigation_lambdas;
    opts.use_trajectory_lambdas = cfg.use_kappa_trajectory_lambdas;
    opts.use_per_stat_deviation_lambdas = cfg.use_per_stat_deviation_lambdas;
    opts.use_kappa_ceiling_lambdas = cfg.use_kappa_ceiling_lambdas;
    opts.kappa_ceiling_strength = cfg.kappa_ceiling_strength;
    opts.kappa_ceiling_min_scale = cfg.kappa_ceiling_min_scale;
    opts.use_kappa_trajectory_ceiling = cfg.use_kappa_trajectory_ceiling;
    opts.target_kappa = cfg.kappa_lambda_target;
    opts.trajectory_window = cfg.kappa_trajectory_window;
    opts.deviation_span = cfg.per_stat_deviation_span;
    return opts;
}

cypha::intelligence::ProfileGuidedLossConfig effective_profile_guided_config(
    const cypha::intelligence::IntelligenceProfiler& profiler, const CyphaLMConfig& cfg,
    const MathIntegrationExportOptions& opts) {
    cypha::intelligence::ProfileGuidedLossConfig pg_cfg =
        cypha::intelligence::default_profile_guided_loss_config();
    const auto obs = mean_profiler_observation(profiler);
    cypha::intelligence::KappaTrajectoryState traj_copy;
    const cypha::intelligence::KappaTrajectoryState* traj_ptr = opts.kappa_trajectory;
    if (traj_ptr == nullptr && cfg.use_kappa_trajectory_lambdas) {
        traj_copy.ema_kappa =
            cypha::intelligence::IntelligenceProfiler::criticality_score_for(obs);
        traj_copy.sample_count = 1;
        traj_ptr = &traj_copy;
    }
    cypha::intelligence::KappaTrajectoryState traj_scratch;
    if (traj_ptr != nullptr && cfg.use_kappa_trajectory_lambdas) {
        traj_scratch = *traj_ptr;
        return cypha::intelligence::resolve_adaptive_profile_guided_config(
            pg_cfg, obs, adaptive_opts_from(cfg), &traj_scratch);
    }
    return cypha::intelligence::resolve_adaptive_profile_guided_config(pg_cfg, obs,
                                                                         adaptive_opts_from(cfg),
                                                                         nullptr);
}

double navigation_warmup_factor(const CyphaLMConfig& cfg, std::uint32_t step_count) {
    if (cfg.navigation_loss_warmup_steps <= 0) {
        return 1.0;
    }
    return std::min(1.0, static_cast<double>(step_count) /
                             static_cast<double>(cfg.navigation_loss_warmup_steps));
}

}  // namespace

void apply_math_integration_preset(CyphaLMConfig& cfg) {
    cfg.profile_guided_loss = true;
    cfg.use_full_navigation_loss = true;
    cfg.use_profile_curriculum = true;
    cfg.navigation_loss_warmup_steps = 200;
    cfg.use_adaptive_navigation_lambdas = true;
    cfg.kappa_lambda_target = 0.83;
    cfg.use_kappa_trajectory_lambdas = true;
    cfg.kappa_trajectory_window = 16;
    cfg.use_per_stat_deviation_lambdas = true;
    cfg.per_stat_deviation_span = 1.0;
    cfg.use_kappa_ceiling_lambdas = true;
    cfg.kappa_ceiling_strength = 1.5;
    cfg.kappa_ceiling_min_scale = 0.40;
    cfg.use_kappa_trajectory_ceiling = true;
    cfg.use_kappa_excess_grad_nudge = true;
    cfg.kappa_excess_grad_scale = 0.35;
    cfg.kappa_excess_grad_margin = 0.02;
    cfg.use_kappa_kernel_blend_scale = true;
    cfg.kappa_kernel_blend_floor = 0.08;
    cfg.use_kappa_navigation_warmup_scale = true;
    cfg.kappa_navigation_warmup_strength = 0.35;
    cfg.kappa_navigation_warmup_floor = 0.65;
    cfg.use_lstm_d_eff_hidden_nudge = true;
    cfg.use_eigenvalue_d_eff = false;

    cfg.use_alpha_forget_gate = true;
    cfg.use_tau_forget_gate = true;
    cfg.use_reu_forget_gate = true;
    cfg.reu_forget_gate_blend = 0.25;
    cfg.alpha_init = 0.5;
    cfg.alpha_learnable = true;

    cfg.use_gria_gated_mixture = true;
    cfg.hybrid_blend_logit = 0.5;
    cfg.hybrid_blend_learnable = true;

    cfg.use_ood_branching = true;
    cfg.n_experts = std::max(cfg.n_experts, 8);
    cfg.online = true;

    cfg.use_free_energy_loss = true;
    cfg.free_energy_beta = 0.01;

    cfg.use_mdl_forget = true;
    cfg.mdl_forget_max_norm = 4.0;

    cfg.use_priority_replay = true;
    cfg.max_memory_slots = std::max(cfg.max_memory_slots, 256);
    if (cfg.compress_interval <= 0 || cfg.compress_interval > 16) {
        cfg.compress_interval = 16;
    }

    cfg.use_nig_state_cell = true;
    cfg.n_experts = std::max(cfg.n_experts, 4);

    cfg.use_hebbian_stack = true;
    cfg.use_hebb_graph = true;

    cfg.use_kernel_llr = true;
    cfg.kernel_m = 64;
    cfg.kernel_blend = 0.25;
    cfg.kernel_lr_scale = 1.0;
}

nlohmann::json export_math_integration_report(
    const cypha::intelligence::IntelligenceProfiler& profiler, const CyphaLMConfig& cfg,
    const MathIntegrationExportOptions& opts) {
    const auto completeness = cypha::intelligence::validate_profile_completeness(profiler);
    const auto obs = mean_profiler_observation(profiler);
    const auto effective_cfg = effective_profile_guided_config(profiler, cfg, opts);
    const auto loss_terms =
        cypha::intelligence::compute_profile_guided_loss(obs, effective_cfg);
    const auto deltas = cypha::intelligence::profile_stat_deltas(obs, effective_cfg);
    const double warmup = navigation_warmup_factor(cfg, opts.step_count);

    nlohmann::json out = export_intelligence_monitor_report(profiler);
    out["profile_guided_loss"] = {
        {"alpha_penalty", loss_terms.alpha_penalty},
        {"d_eff_penalty", loss_terms.d_eff_penalty},
        {"sigma_branch_penalty", loss_terms.sigma_branch_penalty},
        {"tau_penalty", loss_terms.tau_penalty},
        {"r_eu_penalty", loss_terms.r_eu_penalty},
        {"lipschitz_penalty", loss_terms.lipschitz_penalty},
        {"calibration_penalty", loss_terms.calibration_penalty},
        {"navigation_loss_total", loss_terms.navigation_loss_total},
        {"total", loss_terms.total},
        {"warmup_factor", warmup},
    };
    out["effective_lambdas"] = profile_guided_lambdas_to_json(effective_cfg);
    out["stat_deltas"] = stat_deltas_to_json(deltas);
    out["navigation_config"] = {
        {"use_adaptive_navigation_lambdas", cfg.use_adaptive_navigation_lambdas},
        {"use_kappa_trajectory_lambdas", cfg.use_kappa_trajectory_lambdas},
        {"use_per_stat_deviation_lambdas", cfg.use_per_stat_deviation_lambdas},
        {"use_kappa_ceiling_lambdas", cfg.use_kappa_ceiling_lambdas},
        {"use_lstm_d_eff_hidden_nudge", cfg.use_lstm_d_eff_hidden_nudge},
        {"use_eigenvalue_d_eff", cfg.use_eigenvalue_d_eff},
        {"use_reu_forget_gate", cfg.use_reu_forget_gate},
        {"use_tau_forget_gate", cfg.use_tau_forget_gate},
        {"use_kernel_llr", cfg.use_kernel_llr},
        {"kappa_lambda_target", cfg.kappa_lambda_target},
        {"kappa_ceiling_strength", cfg.kappa_ceiling_strength},
        {"kappa_ceiling_min_scale", cfg.kappa_ceiling_min_scale},
        {"use_kappa_trajectory_ceiling", cfg.use_kappa_trajectory_ceiling},
        {"use_kappa_excess_grad_nudge", cfg.use_kappa_excess_grad_nudge},
        {"kappa_excess_grad_scale", cfg.kappa_excess_grad_scale},
        {"kappa_excess_grad_margin", cfg.kappa_excess_grad_margin},
        {"use_kappa_kernel_blend_scale", cfg.use_kappa_kernel_blend_scale},
        {"kappa_kernel_blend_floor", cfg.kappa_kernel_blend_floor},
        {"use_kappa_navigation_warmup_scale", cfg.use_kappa_navigation_warmup_scale},
        {"kappa_navigation_warmup_strength", cfg.kappa_navigation_warmup_strength},
        {"kappa_navigation_warmup_floor", cfg.kappa_navigation_warmup_floor},
        {"reu_forget_gate_blend", cfg.reu_forget_gate_blend},
        {"kappa_trajectory_window", cfg.kappa_trajectory_window},
        {"per_stat_deviation_span", cfg.per_stat_deviation_span},
        {"navigation_loss_warmup_steps", cfg.navigation_loss_warmup_steps},
    };
    if (opts.kappa_trajectory != nullptr) {
        const double dkappa =
            opts.kappa_trajectory->sample_count > 1
                ? opts.kappa_trajectory->ema_kappa - opts.kappa_trajectory->prev_ema_kappa
                : 0.0;
        out["kappa_trajectory"] = {{"ema_kappa", opts.kappa_trajectory->ema_kappa},
                                   {"prev_ema_kappa", opts.kappa_trajectory->prev_ema_kappa},
                                   {"sample_count", opts.kappa_trajectory->sample_count},
                                   {"dkappa", dkappa}};
    }
    out["profile_completeness"] = cypha::intelligence::profile_completeness_to_json(completeness);
    out["kappa"] = completeness.kappa;
    out["source"] = "cyphalm_math_integration";
    return out;
}

}  // namespace cypha::cyphalm
