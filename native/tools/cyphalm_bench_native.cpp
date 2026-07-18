// cyphalm_bench_native — char-LM BPC benchmark CLI for native CyphaLM tiers.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"
#include "cypha/cyphalm/cyphalm_alpha_spectrum.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_intelligence_hook.hpp"
#include "cypha/cyphalm/cyphalm_math_integration.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/cyphalm_parallel.hpp"
#include "cypha/bench/bench_paths.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_completeness.hpp"
#include "cypha/intelligence/profile_from_model.hpp"
#include "cypha/intelligence/profile_guided_loss.hpp"

namespace {

struct Args {
    std::string mode = "hybrid";
    std::string profile = "d17";
    std::string cell_variant;
    // Default 40k = same_order_e2 short-budget cap (see docs/reports/D16_MULTIVIEW_POLICY_2026-07-17.md).
    // Override with --n-train; use schedule_b + n_train<=24000 for mid-budget multi-view policy.
    int n_train = 40000;
    int n_eval = 2000;
    int threads = 0;
    bool analysis = false;
    int analysis_steps = 256;
    bool intelligence_profile = false;
    bool math_integration = false;
    bool overnight = false;
    double per_stat_deviation_span = -1.0;
    double kappa_lambda_target = -1.0;
    double kappa_ceiling_strength = -1.0;
    double kappa_ceiling_min_scale = -1.0;
    bool use_eigenvalue_d_eff = false;
    bool use_reu_forget_gate = false;
    bool disable_kappa_excess_grad_nudge = false;
    double kappa_excess_grad_scale = -1.0;
    double kappa_excess_grad_margin = -1.0;
    bool disable_kappa_navigation_warmup = false;
    double kappa_navigation_warmup_strength = -1.0;
    double kappa_navigation_warmup_floor = -1.0;
    bool disable_kappa_kernel_blend_scale = false;
    double kappa_kernel_blend_floor = -1.0;
    double reu_forget_gate_blend = -1.0;
    int kappa_trajectory_window = -1;
    int navigation_loss_warmup_steps = -1;
    double free_energy_beta = -1.0;
    double kernel_blend = -1.0;
    int kernel_m = -1;
    bool hybrid_blend_logit_explicit = false;
    double hybrid_blend_logit = 0.5;
    double mdl_forget_max_norm = -1.0;
    double kernel_lr_scale = -1.0;
    double alpha_init = -1.0;
    double hybrid_blend_lr = -1.0;
    int n_experts = -1;
    int max_memory_slots = -1;
    int compress_interval = -1;
    int lstm_hidden = -1;
    int bptt_lstm = -1;
    std::string lstm_optim;
    double grad_clip = -1.0;
    std::string lstm_init;
    double weight_decay = -1.0;
    double lstm_lr = -1.0;
    int lstm_lr_warmup = -1;
    int lstm_lr_cosine = -1;
    bool use_self_correcting_loop = false;
    bool ngram_position_weights = false;
    bool ngram_bilinear_fusion = false;
    std::int64_t bench_seed = -1;
    bool n_train_explicit = false;
    bool n_eval_explicit = false;
};

void usage() {
    std::cerr
        << "usage: cyphalm_bench_native --mode {char_lstm,ssm,hybrid,ssm_gria,context_bank,spectral,rpsm}\n"
        << "       --cell-variant {B0..H22}  (overrides --mode)\n"
        << "       --profile {d17,d21,d04} --n-train N --n-eval M --threads T\n"
        << "       --overnight  (D17/D21: full WikiText + 300k train budget; or CYPHA_BENCH_OVERNIGHT=1)\n"
        << "       --analysis [--analysis-steps N]\n"
        << "       --intelligence-profile\n"
        << "       --math-integration\n"
        << "       --per-stat-deviation-span S\n"
        << "       --kappa-lambda-target K\n"
        << "       --kappa-ceiling-strength S\n"
        << "       --kappa-ceiling-min-scale S\n"
        << "       --kappa-excess-grad-scale S\n"
        << "       --kappa-excess-grad-margin M\n"
        << "       --disable-kappa-excess-grad-nudge\n"
        << "       --disable-kappa-navigation-warmup\n"
        << "       --kappa-navigation-warmup-strength S\n"
        << "       --kappa-navigation-warmup-floor S\n"
        << "       --disable-kappa-kernel-blend-scale\n"
        << "       --kappa-kernel-blend-floor S\n"
        << "       --reu-forget-gate-blend B\n"
        << "       --kappa-trajectory-window W\n"
        << "       --navigation-loss-warmup-steps N\n"
        << "       --free-energy-beta B\n"
        << "       --kernel-blend B\n"
        << "       --kernel-m M\n"
        << "       --hybrid-blend-logit L\n"
        << "       --mdl-forget-max-norm N\n"
        << "       --kernel-lr-scale S\n"
        << "       --alpha-init A\n"
        << "       --hybrid-blend-lr S\n"
        << "       --n-experts N\n"
        << "       --max-memory-slots N\n"
        << "       --compress-interval N\n"
        << "       --lstm-hidden N  (LSTM head hidden width override; default: profile value, e.g. 128 for d17)\n"
        << "       --bptt-lstm N  (truncated BPTT window; default 1 = pin path; or CYPHA_LSTM_BPTT)\n"
        << "       --optim {sgd,adam}  (LSTM optimizer; default sgd; or CYPHA_LSTM_OPTIM)\n"
        << "       --grad-clip C  (global L2 clip; 0=off; or CYPHA_LSTM_GRAD_CLIP)\n"
        << "       --lstm-init {default,classic}  (or CYPHA_LSTM_INIT)\n"
        << "       --weight-decay W  (AdamW; 0=off; or CYPHA_LSTM_WEIGHT_DECAY)\n"
        << "       --lstm-lr LR  (LSTM head learning rate override; or CYPHA_LSTM_LR)\n"
        << "       --lstm-lr-warmup N  (linear warmup steps; 0=off; or CYPHA_LSTM_LR_WARMUP)\n"
        << "       --lstm-lr-cosine N  (cosine decay steps after warmup; 0=off; or CYPHA_LSTM_LR_COSINE)\n"
        << "       --bench-seed N\n"
        << "       --use-eigenvalue-d-eff\n"
        << "       --use-reu-forget-gate\n"
        << "       --use-self-correcting-loop  (Paper IV epistemic feedback loop in eval/intelligence-profile;\n"
        << "                                    opt-in, default off, hybrid-mode only; see HIDDEN_DIM_SCALE_PLAN.md)\n"
        << "       --ngram-position-weights  (B3: learnable w_0..w_ngram scalars before W_e; default off)\n"
        << "       --ngram-bilinear-fusion  (B4: low-rank field⊗embed bilinear term in sum fusion; default off)\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--mode") a.mode = need("--mode");
        else if (k == "--cell-variant") a.cell_variant = need("--cell-variant");
        else if (k == "--profile") a.profile = need("--profile");
        else if (k == "--n-train") {
            a.n_train = std::stoi(need("--n-train"));
            a.n_train_explicit = true;
        } else if (k == "--n-eval") {
            a.n_eval = std::stoi(need("--n-eval"));
            a.n_eval_explicit = true;
        } else if (k == "--threads") a.threads = std::stoi(need("--threads"));
        else if (k == "--overnight") a.overnight = true;
        else if (k == "--analysis") a.analysis = true;
        else if (k == "--analysis-steps") a.analysis_steps = std::stoi(need("--analysis-steps"));
        else if (k == "--intelligence-profile") a.intelligence_profile = true;
        else if (k == "--math-integration") a.math_integration = true;
        else if (k == "--per-stat-deviation-span") {
            a.per_stat_deviation_span = std::stod(need("--per-stat-deviation-span"));
        }
        else if (k == "--kappa-lambda-target") {
            a.kappa_lambda_target = std::stod(need("--kappa-lambda-target"));
        }
        else if (k == "--kappa-ceiling-strength") {
            a.kappa_ceiling_strength = std::stod(need("--kappa-ceiling-strength"));
        }
        else if (k == "--kappa-ceiling-min-scale") {
            a.kappa_ceiling_min_scale = std::stod(need("--kappa-ceiling-min-scale"));
        }
        else if (k == "--kappa-excess-grad-scale") {
            a.kappa_excess_grad_scale = std::stod(need("--kappa-excess-grad-scale"));
        }
        else if (k == "--kappa-excess-grad-margin") {
            a.kappa_excess_grad_margin = std::stod(need("--kappa-excess-grad-margin"));
        }
        else if (k == "--disable-kappa-excess-grad-nudge") {
            a.disable_kappa_excess_grad_nudge = true;
        }
        else if (k == "--disable-kappa-navigation-warmup") {
            a.disable_kappa_navigation_warmup = true;
        }
        else if (k == "--kappa-navigation-warmup-strength") {
            a.kappa_navigation_warmup_strength = std::stod(need("--kappa-navigation-warmup-strength"));
        }
        else if (k == "--kappa-navigation-warmup-floor") {
            a.kappa_navigation_warmup_floor = std::stod(need("--kappa-navigation-warmup-floor"));
        }
        else if (k == "--disable-kappa-kernel-blend-scale") {
            a.disable_kappa_kernel_blend_scale = true;
        }
        else if (k == "--kappa-kernel-blend-floor") {
            a.kappa_kernel_blend_floor = std::stod(need("--kappa-kernel-blend-floor"));
        }
        else if (k == "--reu-forget-gate-blend") {
            a.reu_forget_gate_blend = std::stod(need("--reu-forget-gate-blend"));
        }
        else if (k == "--kappa-trajectory-window") {
            a.kappa_trajectory_window = std::stoi(need("--kappa-trajectory-window"));
        }
        else if (k == "--navigation-loss-warmup-steps") {
            a.navigation_loss_warmup_steps = std::stoi(need("--navigation-loss-warmup-steps"));
        }
        else if (k == "--free-energy-beta") {
            a.free_energy_beta = std::stod(need("--free-energy-beta"));
        }
        else if (k == "--kernel-blend") {
            a.kernel_blend = std::stod(need("--kernel-blend"));
        }
        else if (k == "--kernel-m") {
            a.kernel_m = std::stoi(need("--kernel-m"));
        }
        else if (k == "--hybrid-blend-logit") {
            a.hybrid_blend_logit = std::stod(need("--hybrid-blend-logit"));
            a.hybrid_blend_logit_explicit = true;
        }
        else if (k == "--mdl-forget-max-norm") {
            a.mdl_forget_max_norm = std::stod(need("--mdl-forget-max-norm"));
        }
        else if (k == "--kernel-lr-scale") {
            a.kernel_lr_scale = std::stod(need("--kernel-lr-scale"));
        }
        else if (k == "--alpha-init") {
            a.alpha_init = std::stod(need("--alpha-init"));
        }
        else if (k == "--hybrid-blend-lr") {
            a.hybrid_blend_lr = std::stod(need("--hybrid-blend-lr"));
        }
        else if (k == "--n-experts") {
            a.n_experts = std::stoi(need("--n-experts"));
        }
        else if (k == "--max-memory-slots") {
            a.max_memory_slots = std::stoi(need("--max-memory-slots"));
        }
        else if (k == "--compress-interval") {
            a.compress_interval = std::stoi(need("--compress-interval"));
        }
        else if (k == "--lstm-hidden") {
            a.lstm_hidden = std::stoi(need("--lstm-hidden"));
        }
        else if (k == "--bptt-lstm") {
            a.bptt_lstm = std::stoi(need("--bptt-lstm"));
        }
        else if (k == "--optim") {
            a.lstm_optim = need("--optim");
        }
        else if (k == "--grad-clip") {
            a.grad_clip = std::stod(need("--grad-clip"));
        }
        else if (k == "--lstm-init") {
            a.lstm_init = need("--lstm-init");
        }
        else if (k == "--weight-decay") {
            a.weight_decay = std::stod(need("--weight-decay"));
        }
        else if (k == "--lstm-lr") {
            a.lstm_lr = std::stod(need("--lstm-lr"));
        }
        else if (k == "--lstm-lr-warmup") {
            a.lstm_lr_warmup = std::stoi(need("--lstm-lr-warmup"));
        }
        else if (k == "--lstm-lr-cosine") {
            a.lstm_lr_cosine = std::stoi(need("--lstm-lr-cosine"));
        }
        else if (k == "--bench-seed") {
            a.bench_seed = std::stoll(need("--bench-seed"));
        }
        else if (k == "--use-eigenvalue-d-eff") a.use_eigenvalue_d_eff = true;
        else if (k == "--use-reu-forget-gate") a.use_reu_forget_gate = true;
        else if (k == "--use-self-correcting-loop") a.use_self_correcting_loop = true;
        else if (k == "--ngram-position-weights") a.ngram_position_weights = true;
        else if (k == "--ngram-bilinear-fusion") a.ngram_bilinear_fusion = true;
        else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        const bool overnight = args.overnight || cypha::bench::bench_overnight_enabled();
        if (overnight) {
            if (!args.n_train_explicit) args.n_train = cypha::bench::bench_full_n_train();
            if (!args.n_eval_explicit) args.n_eval = 2000;
#if defined(_WIN32)
            _putenv_s("CYPHA_BENCH_OVERNIGHT", "1");
            _putenv_s("CYPHA_BENCH_FULL_CORPUS", "1");
#else
            setenv("CYPHA_BENCH_OVERNIGHT", "1", 1);
            setenv("CYPHA_BENCH_FULL_CORPUS", "1", 1);
#endif
        }
        cypha::cyphalm::set_thread_count(args.threads);

        cypha::cyphalm::CyphaLMConfig cfg;
        cypha::cyphalm::apply_bench_profile(args.profile, cfg);
        if (const char* seed_env = std::getenv("CYPHA_BENCH_SEED")) {
            cfg.seed = static_cast<std::uint64_t>(std::stoull(seed_env));
        }
        if (args.bench_seed >= 0) {
            cfg.seed = static_cast<std::uint64_t>(args.bench_seed);
        }
        std::string mode_label = args.mode;
        if (!args.cell_variant.empty()) {
            cypha::cyphalm::apply_cell_variant(args.cell_variant, cfg);
            if (const auto* spec = cypha::cyphalm::find_cell_variant(args.cell_variant)) {
                mode_label = spec->bench_mode;
            }
        } else {
            const auto bench_mode = cypha::cyphalm::parse_bench_mode(args.mode);
            cypha::cyphalm::apply_bench_mode(bench_mode, cfg);
        }
        // Unconditional (not gated behind --math-integration like the grid-search overrides
        // below): hidden-dim needs to be settable in vanilla hybrid mode too, per the
        // hidden-dim scale-up plan (docs/reports/HIDDEN_DIM_SCALE_PLAN.md Phase 1).
        // Default (unset) keeps whatever the profile/mode/cell-variant already configured
        // (128 for the d17 production profile).
        if (args.lstm_hidden > 0) {
            cfg.lstm_hidden = args.lstm_hidden;
        }
        if (args.bptt_lstm > 0) {
            cfg.lstm_bptt_steps = args.bptt_lstm;
        }
        if (!args.lstm_optim.empty()) {
            cfg.lstm_optim = args.lstm_optim;
        }
        if (args.grad_clip >= 0.0) {
            cfg.lstm_grad_clip = args.grad_clip;
        }
        if (!args.lstm_init.empty()) {
            cfg.lstm_init = args.lstm_init;
        }
        if (args.weight_decay >= 0.0) {
            cfg.lstm_weight_decay = args.weight_decay;
        }
        if (args.lstm_lr >= 0.0) {
            cfg.lstm_lr = args.lstm_lr;
        }
        if (args.lstm_lr_warmup >= 0) {
            cfg.lstm_lr_warmup_steps = args.lstm_lr_warmup;
        }
        if (args.lstm_lr_cosine >= 0) {
            cfg.lstm_lr_cosine_steps = args.lstm_lr_cosine;
        }
        // Unconditional for the same reason as --lstm-hidden above: this is a verification flag
        // for the epistemic feedback loop (docs/reports/HIDDEN_DIM_SCALE_PLAN.md's 2026-07-11
        // follow-up section) and should be usable without also requiring --math-integration.
        // Default (false) leaves eval_bpc/accumulate_intelligence_profile behavior identical to
        // today, so the locked D17 baseline is unaffected unless this flag is passed explicitly.
        if (args.use_self_correcting_loop) {
            cfg.use_self_correcting_loop = true;
        }
        // Expert-utilization research knobs (Upgrade wave 2) — env-gated, default off.
        if (const char* soft = std::getenv("CYPHA_LM_SOFT_EXPERT_UPDATES"); soft && soft[0] == '1') {
            cfg.use_soft_expert_updates = true;
        }
        if (const char* ent = std::getenv("CYPHA_LM_ROUTING_ENTROPY_FLOOR"); ent && ent[0] == '1') {
            cfg.use_routing_entropy_floor = true;
        }
        if (const char* ne = std::getenv("CYPHA_LM_N_EXPERTS"); ne && ne[0] != '\0') {
            const int n = std::atoi(ne);
            if (n > 0) cfg.n_experts = n;
        }
        if (args.ngram_position_weights) {
            cfg.ngram_position_weights = true;
        }
        if (args.ngram_bilinear_fusion) {
            cfg.ngram_bilinear_fusion = true;
        }
        if (args.math_integration) {
            cypha::cyphalm::apply_math_integration_preset(cfg);
            args.intelligence_profile = true;
            if (args.per_stat_deviation_span > 0.0) {
                cfg.per_stat_deviation_span = args.per_stat_deviation_span;
            }
            if (args.kappa_lambda_target > 0.0) {
                cfg.kappa_lambda_target = args.kappa_lambda_target;
            }
            if (args.kappa_ceiling_strength > 0.0) {
                cfg.kappa_ceiling_strength = args.kappa_ceiling_strength;
            }
            if (args.kappa_ceiling_min_scale > 0.0) {
                cfg.kappa_ceiling_min_scale = args.kappa_ceiling_min_scale;
            }
            if (args.use_eigenvalue_d_eff) {
                cfg.use_eigenvalue_d_eff = true;
            }
            if (args.use_reu_forget_gate) {
                cfg.use_reu_forget_gate = true;
            }
            if (args.disable_kappa_excess_grad_nudge) {
                cfg.use_kappa_excess_grad_nudge = false;
            }
            if (args.kappa_excess_grad_scale > 0.0) {
                cfg.kappa_excess_grad_scale = args.kappa_excess_grad_scale;
            }
            if (args.kappa_excess_grad_margin >= 0.0) {
                cfg.kappa_excess_grad_margin = args.kappa_excess_grad_margin;
            }
            if (args.disable_kappa_navigation_warmup) {
                cfg.use_kappa_navigation_warmup_scale = false;
            }
            if (args.kappa_navigation_warmup_strength > 0.0) {
                cfg.kappa_navigation_warmup_strength = args.kappa_navigation_warmup_strength;
            }
            if (args.kappa_navigation_warmup_floor > 0.0) {
                cfg.kappa_navigation_warmup_floor = args.kappa_navigation_warmup_floor;
            }
            if (args.disable_kappa_kernel_blend_scale) {
                cfg.use_kappa_kernel_blend_scale = false;
            }
            if (args.kappa_kernel_blend_floor > 0.0) {
                cfg.kappa_kernel_blend_floor = args.kappa_kernel_blend_floor;
            }
            if (args.reu_forget_gate_blend >= 0.0) {
                cfg.reu_forget_gate_blend = args.reu_forget_gate_blend;
            }
            if (args.kappa_trajectory_window > 0) {
                cfg.kappa_trajectory_window = args.kappa_trajectory_window;
            }
            if (args.navigation_loss_warmup_steps >= 0) {
                cfg.navigation_loss_warmup_steps = args.navigation_loss_warmup_steps;
            }
            if (args.free_energy_beta > 0.0) {
                cfg.free_energy_beta = args.free_energy_beta;
            }
            if (args.kernel_blend > 0.0) {
                cfg.kernel_blend = args.kernel_blend;
            }
            if (args.kernel_m > 0) {
                cfg.kernel_m = args.kernel_m;
            }
            if (args.hybrid_blend_logit_explicit) {
                cfg.hybrid_blend_logit = args.hybrid_blend_logit;
            }
            if (args.mdl_forget_max_norm > 0.0) {
                cfg.mdl_forget_max_norm = args.mdl_forget_max_norm;
            }
            if (args.kernel_lr_scale > 0.0) {
                cfg.kernel_lr_scale = args.kernel_lr_scale;
            }
            if (args.alpha_init > 0.0) {
                cfg.alpha_init = args.alpha_init;
            }
            if (args.hybrid_blend_lr > 0.0) {
                cfg.hybrid_blend_lr = args.hybrid_blend_lr;
            }
            if (args.n_experts > 0) {
                cfg.n_experts = args.n_experts;
            }
            if (args.max_memory_slots > 0) {
                cfg.max_memory_slots = args.max_memory_slots;
            }
            if (args.compress_interval > 0) {
                cfg.compress_interval = args.compress_interval;
            }
        }
        if ((args.profile == "d17" || args.profile == "d17_bpe") && cfg.vocab_size < 256)
            cfg.vocab_size = 256;
        if (args.profile == "d21" && cfg.vocab_size < 256) cfg.vocab_size = 256;
        if (args.profile == "d04" && cfg.vocab_size < 128) cfg.vocab_size = 128;

        cypha::cyphalm::LMCorpus corpus;
        bool synthetic = false;
        const bool full_corpus =
            (args.profile == "d17" || args.profile == "d17_bpe" || args.profile == "d21") &&
            (cypha::cyphalm::bench_full_corpus_enabled() || overnight);
        try {
            const int max_chars = full_corpus ? 0 : 10'000'000;
            corpus = cypha::cyphalm::load_bench_corpus(args.profile, max_chars, cfg.vocab_size,
                                                       cfg.bpe_merges_path, cfg.bpe_vocab_path);
        } catch (const std::exception&) {
            if (!cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST")) {
                throw;
            }
            synthetic = true;
            corpus.profile = args.profile;
            corpus.source = "synthetic";
            corpus.vocab_size = cfg.vocab_size;
            corpus.train_ids =
                cypha::cyphalm::synthetic_corpus(args.n_train + args.n_eval + 64, cfg.vocab_size,
                                                 cfg.seed);
            const std::size_t split = static_cast<std::size_t>(args.n_train);
            corpus.eval_ids.assign(corpus.train_ids.begin() + static_cast<std::ptrdiff_t>(split),
                                   corpus.train_ids.end());
            corpus.train_ids.resize(split);
        }

        cfg.vocab_size = corpus.vocab_size;
        if (!cfg.bpe_merges_path.empty() && !cfg.bpe_vocab_path.empty()) {
            corpus.source += "+bpe";
        }

        cypha::cyphalm::CyphaLMModel model(cfg);
        cypha::intelligence::IntelligenceProfiler profiler;
        const bool track_profile = args.math_integration;
        model.train_sequence(corpus.train_ids, args.n_train, cfg.train_epochs,
                             track_profile ? &profiler : nullptr);
        const double bpc = model.eval_bpc(corpus.eval_ids, args.n_eval, nullptr);
        if (args.intelligence_profile) {
            model.accumulate_intelligence_profile(corpus.eval_ids, args.n_eval, profiler);
        }

        nlohmann::json out = {
            {"mode", mode_label},
            {"cell_variant", args.cell_variant.empty() ? nullptr : nlohmann::json(args.cell_variant)},
            {"profile", args.profile},
            {"context_mode", cypha::cyphalm::context_mode_string(cfg.context_mode)},
            {"n_train", args.n_train},
            {"n_eval", args.n_eval},
            {"train_epochs", cfg.train_epochs},
            {"threads", cypha::cyphalm::effective_thread_count()},
            {"corpus", corpus.source},
            {"synthetic", synthetic},
            {"full_corpus", full_corpus},
            {"overnight", overnight},
            {"bench_seed", cfg.seed},
            {"bpc", bpc},
            {"vocab_size", cfg.vocab_size},
            {"lstm_hidden", cfg.lstm_hidden},
            {"use_self_correcting_loop", cfg.use_self_correcting_loop},
            {"ngram_position_weights", cfg.ngram_position_weights},
            {"ngram_bilinear_fusion", cfg.ngram_bilinear_fusion},
            {"ngram_fusion", cfg.ngram_fusion},
            {"n_experts", cfg.n_experts},
            {"use_soft_expert_updates", cfg.use_soft_expert_updates},
            {"use_routing_entropy_floor", cfg.use_routing_entropy_floor},
        };
        if (std::isnan(bpc)) out["bpc"] = nullptr;
        if (args.intelligence_profile) {
            const auto completeness = cypha::intelligence::validate_profile_completeness(profiler);
            if (args.math_integration) {
                cypha::cyphalm::MathIntegrationExportOptions export_opts;
                export_opts.kappa_trajectory = &model.kappa_trajectory_state();
                export_opts.step_count = model.train_step_count();
                out["math_integration"] =
                    cypha::cyphalm::export_math_integration_report(profiler, cfg, export_opts);
            }
            out["intelligence_profile"] = cypha::cyphalm::export_intelligence_monitor_report(profiler);
            // Distinct from intelligence_profile.statistics[d_eff] (measured over the fixed
            // field_dim-wide GRIA field): this is the participation-ratio D_eff measured
            // directly over the actual lstm_hidden-wide LSTM hidden-state history, so it is
            // the statistic that should move when --lstm-hidden changes (see
            // docs/reports/HIDDEN_DIM_SCALE_PLAN.md).
            //
            // Phase 3 follow-up (2026-07-12, "Finding 2" / history-buffer sampling fix):
            // `lstm_hidden_d_eff` alone (the width-normalized ratio) cannot distinguish a
            // genuine representational change from a statistical-power artifact of the
            // history-buffer sample count vs. `lstm_hidden`. Export the raw (unnormalized)
            // effective-dimension count and the actual samples/dims ratio used for *this*
            // measurement alongside it so a future reader can tell at a glance whether a
            // given `lstm_hidden_d_eff` value is well-powered, without re-deriving it from
            // `lstm_hidden` and the (now instance-scaled, not hardcoded) history-buffer cap.
            const auto lstm_hidden_d_eff_detail = model.lstm_hidden_d_eff_detail();
            const bool lstm_hidden_d_eff_available = lstm_hidden_d_eff_detail.normalized >= 0.0;
            out["intelligence_profile"]["lstm_hidden_d_eff"] =
                lstm_hidden_d_eff_available ? nlohmann::json(lstm_hidden_d_eff_detail.normalized)
                                            : nlohmann::json(nullptr);
            out["intelligence_profile"]["lstm_hidden_d_eff_raw"] =
                lstm_hidden_d_eff_available ? nlohmann::json(lstm_hidden_d_eff_detail.raw)
                                            : nlohmann::json(nullptr);
            out["intelligence_profile"]["lstm_hidden_d_eff_sample_ratio"] =
                lstm_hidden_d_eff_available ? nlohmann::json(lstm_hidden_d_eff_detail.sample_ratio)
                                            : nlohmann::json(nullptr);
            out["intelligence_profile"]["lstm_hidden_d_eff_n_samples"] =
                lstm_hidden_d_eff_available ? nlohmann::json(lstm_hidden_d_eff_detail.n_samples)
                                            : nlohmann::json(nullptr);
            out["intelligence_profile"]["lstm_hidden_d_eff_method"] =
                cfg.use_eigenvalue_d_eff ? "covariance_eigenvalue" : "variance_proxy";
            out["profile_completeness"] =
                cypha::intelligence::profile_completeness_to_json(completeness);
            if (!completeness.all_complete) {
                std::cerr << "cyphalm_bench_native: incomplete intelligence profile; missing stats:";
                for (const auto& name : completeness.missing_stats) {
                    std::cerr << ' ' << name;
                }
                std::cerr << '\n';
                return 1;
            }
        }
        if (args.analysis) {
            const auto profile = model.compression_profile();
            out["alpha_spectrum"] = {
                {"mean_alpha", profile.value("mean_alpha", 0.0)},
                {"fraction_edge_of_chaos", profile.value("fraction_near_edge_of_chaos", 0.0)},
                {"n_experts", profile.value("n_experts", 0)},
            };
            const auto track =
                cypha::cyphalm::alpha_spectrum_track(model, args.analysis_steps, corpus.train_ids);
            out["alpha_track_steps"] = track.size();
            if (!track.empty()) out["alpha_track_last"] = track.back();
        }
        if (cfg.context_mode == cypha::cyphalm::ContextMode::Hybrid) {
            out["hybrid_blend_logit"] = model.hybrid_blend_logit();
            out["hybrid_gria_weight"] = model.hybrid_gria_weight();
            const double saved_blend = model.hybrid_blend_logit();
            model.set_hybrid_blend_logit(-40.0);
            out["bpc_lstm_only"] = model.eval_bpc(corpus.eval_ids, args.n_eval);
            model.set_hybrid_blend_logit(saved_blend);
        }
        std::cout << out.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cyphalm_bench_native: " << e.what() << "\n";
        usage();
        return 1;
    }
}
