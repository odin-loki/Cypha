#include "cypha/cyphalm/cyphalm_intelligence_hook.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/cyphalm_profile_curriculum.hpp"
#include "cypha/cyphalm/lm_intelligence_monitor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cellai_ssm.hpp"
#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/hybrid_blend.hpp"
#include "cypha/cyphalm/npz_util.hpp"
#include "cypha/cyphalm/sr_gate_laws.hpp"
#include "cypha/cyphalm/cyphalm_views.hpp"
#include "cypha/cyphalm/ssm_diagnose.hpp"
#include "cypha/bench/bench_paths.hpp"
#include "cypha/intelligence/epistemic_threshold.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/measurers.hpp"
#include "cypha/intelligence/profile_guided_loss.hpp"

namespace cypha {
namespace cyphalm {

namespace {

constexpr double kLog2 = 0.6931471805599453;
constexpr double kLogEps = 1e-12;

bool uses_ssm(ContextMode m) {
    return m != ContextMode::CharLstm;
}

bool uses_gria(ContextMode m) {
    return m != ContextMode::CharLstm;
}

bool uses_rpsm(const CyphaLMConfig& cfg) {
    return cfg.use_rpsm_layer || cfg.context_mode == ContextMode::Rpsm;
}

bool uses_lstm(ContextMode m) {
    return m == ContextMode::Hybrid || m == ContextMode::CharLstm;
}

bool uses_ngram_embed_path(ContextMode mode, const CyphaLMConfig& cfg) {
    if (cfg.ngram_context <= 0) {
        return false;
    }
    return mode == ContextMode::Hybrid || mode == ContextMode::GriaNgram ||
           mode == ContextMode::AblationNoSsm || mode == ContextMode::Rpsm ||
           mode == ContextMode::SsmGria;
}

bool uses_ngram_count_path(const CyphaLMConfig& cfg) {
    return cfg.ngram_context > 0;
}

bool uses_hybrid_ewc(ContextMode mode) {
    return mode == ContextMode::Hybrid || mode == ContextMode::SsmGria ||
           mode == ContextMode::GriaNgram || mode == ContextMode::SsmGriaNoLstm ||
           mode == ContextMode::Full;
}

bool uses_profile_guided_backprop(const CyphaLMConfig& cfg) {
    return cfg.profile_guided_loss || cfg.use_full_navigation_loss || cfg.cell_variant == "H05";
}

// True when CyphaDIF::predict/train_step outputs or side effects feed the hybrid forward path.
// At the locked D17 default (Hybrid + ngram_fuse_split + no forget-gate/OOD/GNG-controller flags),
// GRIA input is built via NgramFusion only -- DIF mean/epistemic are never read -- so both predict
// and online train_step are dead work every character step. Gated conservatively: any config that
// reads DIF output (proj_dif GRIA path, tau/r_eu forget gates, OOD branching, Full/SsmGriaNoLstm
// bptt scaling, discriminative feedback) keeps the subsystem active for bit-identical behavior.
bool dif_subsystem_affects_forward(const CyphaLMConfig& cfg) {
    const auto mode = cfg.context_mode;
    if (mode == ContextMode::CharLstm || mode == ContextMode::SsmGria ||
        mode == ContextMode::AblationNoSsm) {
        return false;
    }
    // Expert-utilization research path: keep DIF alive even on ngram_fuse_split hybrid.
    if (cfg.use_soft_expert_updates || cfg.use_routing_entropy_floor || cfg.n_experts > 0) {
        return true;
    }
    if (cfg.use_discriminative_feedback) {
        return true;
    }
    if (cfg.use_tau_forget_gate || cfg.use_reu_forget_gate || cfg.use_ood_branching) {
        return true;
    }
    if (cfg.use_gng && cfg.use_gria_controller) {
        return true;
    }
    if (mode == ContextMode::Full || mode == ContextMode::SsmGriaNoLstm) {
        return true;
    }
    if (cfg.ngram_fuse_split && uses_ngram_embed_path(mode, cfg)) {
        return false;
    }
    return true;
}

cypha::intelligence::ProfileGuidedLossConfig profile_guided_loss_config_for(const CyphaLMConfig& cfg) {
    if (cfg.use_full_navigation_loss) {
        return cypha::intelligence::default_profile_guided_loss_config();
    }
    cypha::intelligence::ProfileGuidedLossConfig partial;
    const auto targets = cypha::intelligence::IntelligenceProfiler::critical_targets();
    partial.target_r_eu = targets[4];
    partial.target_tau = targets[3];
    partial.lambda_r_eu = 0.1;
    partial.lambda_tau = 0.1;
    return partial;
}

void init_proj(std::vector<double>& proj, int rows, int cols, std::uint64_t seed, double scale) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    proj.assign(static_cast<std::size_t>(rows * cols), 0.0);
    for (auto& v : proj) v = nd(rng);
}

void init_proj_from_rng(std::vector<double>& proj, int rows, int cols, std::mt19937_64& rng,
                        double scale) {
    std::normal_distribution<double> nd(0.0, scale);
    proj.assign(static_cast<std::size_t>(rows * cols), 0.0);
    for (auto& v : proj) v = nd(rng);
}

// CYPHA_PERF_TRACE: opt-in (env-gated), zero-overhead-when-unset phase-timing instrumentation
// for CyphaLMModel::train_step, added for docs/reports/PERFORMANCE_PROFILE_2026-07-12.md. Purely
// diagnostic (stderr summary at process exit) -- never touches training math/state, so it cannot
// affect the D17 BPC pin regardless of whether it is enabled. Safe to keep permanently opt-in.
struct PerfTracePhases {
    static constexpr std::size_t kCount = 8;
    static constexpr std::array<const char*, kCount> kNames = {
        "predict_next (forward: GRIA+LSTM+hybrid blend)",
        "gria_backward (cross_entropy_gradients+update_weights/alpha/bias)",
        "dif_train_step (kernel-LLR memory)",
        "hebbian_stack (encoder_train_step)",
        "bptt_ssm_update",
        "lstm_backward (hybrid path)",
        "rpsm_train_step",
        "tail (ewc/ngram/gng/laplace bookkeeping)",
    };
    bool enabled = false;
    long long calls = 0;
    std::array<double, kCount> totals{};

    PerfTracePhases() { enabled = std::getenv("CYPHA_PERF_TRACE") != nullptr; }
    ~PerfTracePhases() {
        if (!enabled || calls == 0) return;
        double total = 0.0;
        for (double t : totals) total += t;
        std::cerr << "=== CYPHA_PERF_TRACE: train_step phase breakdown over " << calls
                   << " calls (" << total << "s instrumented) ===\n";
        for (std::size_t i = 0; i < kCount; ++i) {
            const double pct = total > 0.0 ? (100.0 * totals[i] / total) : 0.0;
            std::cerr << "  " << kNames[i] << ": " << totals[i] << "s (" << pct << "%)\n";
        }
    }
};
PerfTracePhases g_perf_trace;

// Times `fn()` into g_perf_trace.totals[idx] when CYPHA_PERF_TRACE is set; otherwise calls
// `fn()` directly with no chrono overhead at all.
template <typename Fn>
void perf_trace_scope(std::size_t idx, Fn&& fn) {
    if (!g_perf_trace.enabled) {
        fn();
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    g_perf_trace.totals[idx] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// begin/end pair for wrapping statements that produce a value to bind via `const auto` (where
// perf_trace_scope's capture-by-reference-into-a-lambda pattern would need an extra default
// construction + copy). No-op (single branch, no clock read) when tracing is disabled.
inline std::chrono::steady_clock::time_point perf_trace_begin() {
    return g_perf_trace.enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
}
inline void perf_trace_end(std::size_t idx, std::chrono::steady_clock::time_point t0) {
    if (!g_perf_trace.enabled) return;
    g_perf_trace.totals[idx] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// CYPHA_PERF_TRACE follow-up (2026-07-12, part 3, docs/reports/PERFORMANCE_PROFILE_2026-07-12.md
// "Follow-up (2026-07-12, part 3)"): fine-grained sub-phase breakdown *inside* predict_next
// itself, mirroring char_lstm.cpp's BackwardSubPhaseTrace/BwdScopeTimer pattern used for
// lstm_backward in part 2. Same env var as the top-level trace. No mutex (unlike
// BackwardSubPhaseTrace): predict_next is a CyphaLMModel member function only ever called from
// single-threaded contexts (train_step, generation, ssm_diagnose) -- CyphaLMBatch's
// parallel_batch calls CharLSTMHead::forward_step and GRIALowRank::forward directly on their own
// shared objects, never through CyphaLMModel::predict_next -- so this can use the same
// unsynchronized style as the top-level g_perf_trace above. Purely diagnostic; never touches
// training math/state.
struct PredictNextSubPhaseTrace {
    static constexpr std::size_t kCount = 8;
    static constexpr std::array<const char*, kCount> kNames = {
        "1. embed lookup + history record",
        "2. SSM step + hebbian hooks + field projection",
        "3. DIF (kernel-LLR) predict",
        "4. GRIA input build (memory/context-bank/ngram-fusion + rpsm/gng)",
        "5. GRIA field forward (logits+softmax)",
        "6. LSTM forward step (gates) + hidden-state bookkeeping",
        "7. hybrid blend combination (blend_logit + blend_log_probs)",
        "8. tail (token history + top-k + bookkeeping)",
    };
    bool enabled = false;
    long long calls = 0;
    std::array<double, kCount> totals{};

    PredictNextSubPhaseTrace() { enabled = std::getenv("CYPHA_PERF_TRACE") != nullptr; }
    ~PredictNextSubPhaseTrace() {
        if (!enabled || calls == 0) return;
        double total = 0.0;
        for (double t : totals) total += t;
        std::cerr << "=== CYPHA_PERF_TRACE: predict_next sub-phase breakdown over " << calls
                   << " calls (" << total << "s instrumented) ===\n";
        for (std::size_t i = 0; i < kCount; ++i) {
            const double pct = total > 0.0 ? (100.0 * totals[i] / total) : 0.0;
            std::cerr << "  " << kNames[i] << ": " << totals[i] << "s (" << pct << "%)\n";
        }
    }
};
PredictNextSubPhaseTrace g_predict_trace;

class PredictScopeTimer {
 public:
    explicit PredictScopeTimer(std::size_t idx) : idx_(idx), enabled_(g_predict_trace.enabled) {
        if (enabled_) t0_ = std::chrono::steady_clock::now();
    }
    ~PredictScopeTimer() {
        if (enabled_) {
            g_predict_trace.totals[idx_] +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
        }
    }

 private:
    std::size_t idx_;
    bool enabled_;
    std::chrono::steady_clock::time_point t0_;
};

std::vector<double> resize_to_field(const std::vector<double>& v, int field_dim) {
    std::vector<double> out(static_cast<std::size_t>(field_dim), 0.0);
    const int n = std::min(field_dim, static_cast<int>(v.size()));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = v[static_cast<std::size_t>(i)];
    }
    return out;
}

void fit_sr_gate_laws_on_lstm(CharLSTMHead& lstm, int vocab_size, std::uint64_t seed) {
    std::vector<int> probe_ids;
    const int probe_count = std::min(32, vocab_size);
    for (int i = 0; i < probe_count; ++i) {
        probe_ids.push_back(i);
    }
    if (probe_ids.empty()) {
        probe_ids.push_back(0);
    }
    std::mt19937_64 rng(seed + 811);
    for (int t = 0; t < 8; ++t) {
        probe_ids.push_back(static_cast<int>(rng() % static_cast<std::uint64_t>(std::max(1, vocab_size))));
    }
    const SrGateTrace trace = collect_lstm_gate_trace(lstm, probe_ids, 24);
    SrGateLaws laws = fit_sr_gate_laws(trace, lstm.hidden);
    lstm.set_sr_gate_laws(laws);
}

std::vector<double> matvec(const std::vector<double>& m, int rows, int cols,
                           const std::vector<double>& x) {
    std::vector<double> out(static_cast<std::size_t>(rows), 0.0);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int c = 0; c < cols; ++c) {
            acc += m[static_cast<std::size_t>(r * cols + c)] * x[static_cast<std::size_t>(c)];
        }
        out[static_cast<std::size_t>(r)] = acc;
    }
    return out;
}

// Perf (2026-07-12, part 2, docs/reports/PERFORMANCE_PROFILE_2026-07-12.md "Follow-up"):
// out-param overload so hot-path callers (bptt_ssm_update) can fill a persistent scratch buffer
// in place instead of receiving a freshly heap-allocated vector every call. `.resize()` on an
// already-correctly-sized vector is a no-op on capacity (matches the `.assign()`-reuse pattern
// used throughout char_lstm.cpp's forward_step/backward_step). Numerically identical to the
// value-returning overload below, which now delegates to this one.
//
// Loop order also swapped (r outer, c inner) vs. the original c-outer/r-inner form: `m` is
// stored row-major (`rows x cols`), so the original inner loop over `r` accessed
// `m[r*cols + c]` with stride `cols` doubles between iterations -- a cache-hostile strided
// scan (confirmed as ~29% + ~29% of CharLSTMHead::backward_step's own time for the identical
// access pattern against Wx/Wh, see the char_lstm.cpp fix in the same follow-up section). Row-
// major iteration (r outer, c inner) reads `m` sequentially instead. This is a pure loop-
// interchange, not a summation-order change: for each fixed `c`, the additions into `out[c]`
// still happen in strictly increasing `r` order (0, 1, 2, ..., rows-1) -- the outer loop just
// visits `r` before `c` instead of after -- so the result is bit-for-bit identical to the
// original, not merely "close".
void matvec_transpose(const std::vector<double>& m, int rows, int cols, const std::vector<double>& x,
                      std::vector<double>& out) {
    if (out.size() != static_cast<std::size_t>(cols)) out.resize(static_cast<std::size_t>(cols));
    std::fill(out.begin(), out.end(), 0.0);
    for (int r = 0; r < rows; ++r) {
        const double xr = x[static_cast<std::size_t>(r)];
        const double* row = m.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
        for (int c = 0; c < cols; ++c) {
            out[static_cast<std::size_t>(c)] += row[c] * xr;
        }
    }
}

std::vector<double> matvec_transpose(const std::vector<double>& m, int rows, int cols,
                                     const std::vector<double>& x) {
    std::vector<double> out;
    matvec_transpose(m, rows, cols, x, out);
    return out;
}

}  // namespace

CyphaLMModel::CyphaLMModel(CyphaLMConfig cfg) : cfg_(std::move(cfg)) {
    hybrid_blend_logit_ = cfg_.hybrid_blend_logit;
    // Phase 3 follow-up (docs/reports/HIDDEN_DIM_SCALE_PLAN.md "Finding 2"): scale the
    // LSTM hidden-state history ring buffer with `lstm_hidden` so `lstm_hidden_d_eff()`'s
    // sample/dims ratio doesn't collapse at wide hidden sizes. `2 * lstm_hidden` targets a
    // >=2x samples/dims ratio (the doc's own recommended floor); `std::max(48, ...)` keeps
    // the pre-fix constant as a floor for tiny/degenerate `lstm_hidden` configs rather than
    // shrinking below what was already known to work. At the default `lstm_hidden=128`
    // this intentionally changes behavior (256 rows instead of 48, ratio 2.0 instead of
    // 0.375) -- documented, not a silent regression: 48 rows was never derived from
    // anything about hidden=128 specifically, just a fixed legacy constant, so there is no
    // "correct" old behavior at hidden=128 to preserve bit-for-bit, only a consistent rule
    // applied uniformly across all `lstm_hidden` values.
    lstm_h_history_max_ = std::max(48, 2 * cfg_.lstm_hidden);
    init_components();
}

CyphaLMModel::~CyphaLMModel() = default;

void CyphaLMModel::init_components() {
    const auto mode = cfg_.context_mode;
    if (mode != ContextMode::CharLstm) {
        embed_ = std::make_unique<EmbedTable>(static_cast<std::uint32_t>(cfg_.vocab_size),
                                              static_cast<std::uint32_t>(cfg_.d_embed),
                                              static_cast<std::uint32_t>(cfg_.seed));
    }
    if (uses_ssm(mode)) {
        CellAISSMConfig sc;
        sc.d_input = cfg_.d_embed;
        sc.d_state = cfg_.d_state;
        sc.tau_fast = cfg_.tau_fast;
        sc.tau_slow = cfg_.tau_slow;
        sc.n_layers = cfg_.ssm_layers;
        sc.seed = static_cast<int>(cfg_.seed + 1);
        sc.use_spectral_pde = cfg_.use_spectral_pde;
        sc.use_multiscale = cfg_.use_multiscale;
        sc.use_sparse_hebbian = cfg_.use_sparse_hebbian && !cfg_.use_hebbian_stack;
        if (cfg_.use_hierarchical_ssm) {
            hierarchical_ssm_ = std::make_unique<HierarchicalSSM>(sc, cfg_.compress_interval);
        } else {
            ssm_ = std::make_unique<CellAISSM>(sc);
        }
        if (cfg_.use_hebb_graph && !cfg_.use_hebbian_stack) {
            HebbianGraphConfig gc;
            gc.n = 2 * cfg_.d_state;
            if (auto* active = active_ssm()) {
                active->enable_hebb_graph(gc);
            }
        }
        if (cfg_.use_temporal_som) {
            cypha::som::TemporalSOMConfig tc;
            tc.M = 8;
            tc.L_max = 16;
            if (auto* active = active_ssm()) {
                active->enable_temporal_som(tc);
            }
        }
        if (cfg_.use_hebbian_stack) {
            hebbian_stack_ = std::make_unique<HebbianStack>();
            HebbianStackConfig hcfg;
            hcfg.use_sparse_hebbian = cfg_.use_sparse_hebbian;
            hcfg.use_hebb_graph = cfg_.use_hebb_graph;
            hcfg.ssm_hebb_lr = cfg_.ssm_hebb_lr;
            hcfg.d_state = cfg_.d_state;
            hcfg.n_layers = cfg_.ssm_layers;
            hcfg.graph.n = cfg_.field_dim;
            hebbian_stack_->configure(hcfg);
        }
        {
            std::mt19937_64 rng(cfg_.seed);
            constexpr double kScale = 0.02;
            init_proj_from_rng(proj_ssm_, cfg_.field_dim, ssm_context_dim(), rng, kScale);
            init_proj_from_rng(proj_dif_, cfg_.field_dim, cfg_.field_dim, rng, kScale);
            init_proj_from_rng(proj_embed_, cfg_.field_dim, cfg_.d_embed, rng, kScale);
            const int n_pos = 1 + std::max(0, cfg_.ngram_context);
            const int ngram_in = cfg_.field_dim + n_pos * cfg_.d_embed;
            std::normal_distribution<double> nd(0.0, kScale);
            for (int i = 0; i < cfg_.field_dim * ngram_in; ++i) {
                (void)nd(rng);
            }
        }
    }
    if (cfg_.use_gng) {
        gng_ = std::make_unique<cypha::som::GNGExpertManager>(cfg_.field_dim);
        if (cfg_.use_gria_controller) {
            gria_controller_ = std::make_unique<cypha::som::GRIAController>();
        }
    }
    if (cfg_.use_discriminative_feedback) {
        discriminative_feedback_ = std::make_unique<cypha::som::DiscriminativeFeedback>();
    }
    if (uses_gria(mode)) {
        gria_d_in_ = cfg_.field_dim;
        if (cfg_.view_id_dim > 0) {
            gria_d_in_ += cfg_.view_id_dim;
            view_emb_ = std::make_unique<ViewEmbedding>(
                std::max(1, cfg_.max_view_slots), cfg_.view_id_dim, cfg_.seed + 3,
                cfg_.view_learnable);
        }
        gria_ = std::make_unique<GRIALowRank>(
            gria_d_in_, cfg_.vocab_size, cfg_.gria_rank, cfg_.alpha_init,
            cfg_.alpha_learnable, cfg_.seed + 2);
        gria_in_.assign(static_cast<std::size_t>(gria_d_in_), 0.0);
        token_counts_.assign(static_cast<std::size_t>(cfg_.vocab_size), 1.0);
        refresh_laplace_prior();
    }
    if (mode != ContextMode::CharLstm) {
        dif_ = std::make_unique<CyphaDIF>(cfg_);
    }
    const auto ngram_path = uses_ngram_embed_path(mode, cfg_);
    const int n_pos = 1 + std::max(0, cfg_.ngram_context);
    if (ngram_path) {
        embed_history_.assign(static_cast<std::size_t>(n_pos),
                              std::vector<double>(static_cast<std::size_t>(cfg_.d_embed), 0.0));
    }
    if (uses_ngram_count_path(cfg_)) {
        const int ctx_len = std::max(0, cfg_.ngram_context - 1);
        token_history_.assign(static_cast<std::size_t>(ctx_len), 0);
    }
    if (cfg_.ngram_fuse_split && ngram_path) {
        const int embed_in = n_pos * cfg_.d_embed;
        const int field_in = (mode == ContextMode::AblationNoSsm) ? 0 : cfg_.field_dim;
        if (field_in > 0) {
            ngram_fusion_ = std::make_unique<NgramFusion>(
                cfg_.field_dim, field_in, embed_in, cfg_.ngram_fusion, n_pos,
                cfg_.ngram_position_weights, cfg_.ngram_bilinear_fusion, cfg_.gria_rank,
                cfg_.seed + 4);
        } else {
            std::mt19937_64 rng(cfg_.seed + 4);
            init_proj_from_rng(proj_ngram_embed_, cfg_.field_dim, embed_in, rng, 0.02);
            const int ngram_in = embed_in;
            init_proj_from_rng(proj_ngram_, cfg_.field_dim, ngram_in, rng, 0.02);
        }
    }
    if (uses_lstm(mode)) {
        lstm_ = std::make_unique<CharLSTMHead>(cfg_.vocab_size, cfg_.lstm_hidden, cfg_.seed + 5,
                                               parse_lstm_init_mode(cfg_.lstm_init));
        lstm_->set_bptt_window(cfg_.lstm_bptt_steps);
        lstm_->set_optim(parse_lstm_optim(cfg_.lstm_optim));
        lstm_->set_grad_clip(cfg_.lstm_grad_clip);
        if (cfg_.use_eml_activation) {
            lstm_->set_activation_mode(LSTMActivationMode::Eml);
        } else if (cfg_.use_axiom_activation) {
            lstm_->set_activation_mode(LSTMActivationMode::Axiom);
            lstm_->set_axiom_grammar(axiom_grammar_from_seed(cfg_.seed + 77, cfg_.lstm_hidden));
        }
        lstm_h_.assign(static_cast<std::size_t>(cfg_.lstm_hidden), 0.0);
        lstm_c_.assign(static_cast<std::size_t>(cfg_.lstm_hidden), 0.0);
        if (cfg_.use_sr_gates) {
            lstm_->set_use_sr_gates(true);
            fit_sr_gate_laws_on_lstm(*lstm_, cfg_.vocab_size, cfg_.seed);
        }
    }
    if (mode == ContextMode::SsmGriaNoLstm || mode == ContextMode::Full) {
        selective_ = std::make_unique<SelectiveSSM>(
            static_cast<std::uint32_t>(cfg_.d_embed), static_cast<std::uint32_t>(cfg_.d_state),
            static_cast<std::uint32_t>(cfg_.field_dim), true, 0.9,
            static_cast<std::uint32_t>(cfg_.seed + 13));
    }
    if (cfg_.max_memory_slots > 0 && mode != ContextMode::CharLstm) {
        memory_ = std::make_unique<CompressiveMemory>(
            static_cast<std::uint32_t>(cfg_.field_dim),
            static_cast<std::uint32_t>(cfg_.max_memory_slots), cfg_.nig_kappa0, cfg_.nig_alpha0,
            cfg_.nig_beta0, static_cast<std::uint32_t>(cfg_.seed + 17));
        memory_->set_compress_interval(static_cast<std::uint32_t>(cfg_.compress_interval));
        memory_->set_priority_replay(cfg_.use_priority_replay);
    }
    if (cfg_.use_context_bank) {
        context_bank_ = std::make_unique<ContextBank>(cfg_.d_embed, cfg_.context_bank_slots);
    }
    if (cfg_.use_nig_state_cell) {
        nig_state_cell_ = std::make_unique<NigStateCell>(
            cfg_.nig_kappa0, cfg_.nig_alpha0, cfg_.nig_beta0, cfg_.field_dim);
    }
    if (cfg_.use_reversible_cell) {
        reversible_cell_ = std::make_unique<ReversibleSSMCell>();
    }
    field_x_.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
    if (gria_in_.empty()) {
        gria_in_.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
    }
    if (!cfg_.bpe_merges_path.empty() && !cfg_.bpe_vocab_path.empty()) {
        bpe_ = std::make_unique<BpeTokenizer>(
            BpeTokenizer::load(cfg_.bpe_merges_path, cfg_.bpe_vocab_path));
    }
    if (uses_rpsm(cfg_)) {
        cypha::rpsm::RpsmSequenceConfig rc;
        rc.n_levels = cfg_.rpsm_n_levels;
        rc.state_dim = cfg_.rpsm_state_dim;
        rc.feat_dim = cfg_.rpsm_feat_dim;
        rc.n_classes = cfg_.vocab_size;
        rc.n_memory_slots = cfg_.rpsm_n_memory_slots;
        rc.beta_memory = cfg_.rpsm_beta_memory;
        rc.surprise_threshold = cfg_.rpsm_surprise_threshold;
        rc.hierarchy_loss_weight = cfg_.rpsm_hierarchy_loss_weight;
        rc.bptt_window = std::max(1, cfg_.rpsm_bptt_window);
        rc.seed = cfg_.seed + 29;
        rc.use_izaac_init = (cfg_.cell_variant == "H19" || cfg_.context_mode == ContextMode::Rpsm);
        // Phase -1 (RPSM_UPGRADE_PLAN.md, RESEARCH_STATUS.md:393): spectral alpha + normalised
        // eta, live for the D21 --mode rpsm production path exactly like use_izaac_init above.
        rc.use_spectral_alpha = (cfg_.context_mode == ContextMode::Rpsm);
        rc.use_normalized_eta = (cfg_.context_mode == ContextMode::Rpsm);
        // Research/experiment-only override for the BPTT window length (see
        // RPSM_UPGRADE_PLAN.md sec14 -- default is 1, i.e. off, because window>1 was measured
        // to monotonically *hurt* eval BPC at this layer's lr regime). Env wins over profile.
        if (const char* w = std::getenv("CYPHALM_RPSM_BPTT_WINDOW")) {
            const int wv = std::atoi(w);
            if (wv > 0) rc.bptt_window = wv;
        }
        // Research/experiment-only override for §15's online mu0/inv_var world-stats update
        // (RPSM_UPGRADE_PLAN.md §15) -- default is off (see rpsm_sequence_layer.hpp's
        // `use_online_world_stats` doc comment for why this is opt-in, matching Phase -1's own
        // opt-in-flag convention). Not part of the profile schema; used to reproduce §15's
        // before/after measurement without a rebuild per config change.
        if (const char* ws = std::getenv("CYPHALM_RPSM_WORLD_STATS")) {
            rc.use_online_world_stats = (std::atoi(ws) != 0);
        }
        rpsm_layer_ = std::make_unique<cypha::rpsm::RpsmSequenceLayer>(rc);
        rpsm_log_probs_.assign(static_cast<std::size_t>(cfg_.vocab_size), 0.0);
    }
    if (uses_profile_guided_backprop(cfg_) && !train_profiler_) {
        train_profiler_ = std::make_unique<cypha::intelligence::IntelligenceProfiler>();
    }
}

CyphaLMModel CyphaLMModel::from_json_npz(const std::string& json_path) {
    return load_cyphalm_model(json_path);
}

void CyphaLMModel::save(const std::string& base_path) const {
    save_cyphalm_model(*this, base_path);
}

void CyphaLMModel::reset_context() {
    if (hierarchical_ssm_) hierarchical_ssm_->reset();
    if (ssm_) ssm_->reset();
    if (selective_) selective_->reset();
    if (memory_) memory_->reset();
    if (context_bank_) context_bank_->reset();
    if (nig_state_cell_) nig_state_cell_->reset();
    if (reversible_cell_) reversible_cell_->reset();
    last_mean_alpha_ = 0.5;
    last_train_loss_ = 0.0;
    if (lstm_) {
        // Apply any partial truncated-BPTT window before wiping state (no-op when window empty).
        lstm_->flush_bptt(cfg_.lstm_lr);
        lstm_->reset_state();
        std::fill(lstm_h_.begin(), lstm_h_.end(), 0.0);
        std::fill(lstm_c_.begin(), lstm_c_.end(), 0.0);
    }
    hybrid_lstm_has_cache_ = false;
    last_hybrid_log_g_.clear();
    last_hybrid_log_l_.clear();
    for (auto& row : embed_history_) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    if (!token_history_.empty()) {
        std::fill(token_history_.begin(), token_history_.end(), 0);
    }
    bptt_buffer_.clear();
    last_e_.clear();
    last_ctx_.clear();
    last_ssm_h_fast_.clear();
    if (rpsm_layer_) rpsm_layer_->reset();
    rpsm_bptt_token_ids_.clear();
    // DIF experts persist across view/block resets (Python carry_dif=True).
    std::fill(field_x_.begin(), field_x_.end(), 0.0);
    step_count_ = 0;
}

void CyphaLMModel::record_embedding(const std::vector<double>& e) {
    if (embed_history_.empty()) return;
    for (std::size_t j = embed_history_.size() - 1; j > 0; --j) {
        embed_history_[j] = embed_history_[j - 1];
    }
    embed_history_[0] = e;
}

void CyphaLMModel::ngram_embedding_vector(std::vector<double>& out) const {
    out.clear();
    for (const auto& row : embed_history_) {
        out.insert(out.end(), row.begin(), row.end());
    }
}

std::vector<double> CyphaLMModel::ngram_embedding_vector() const {
    std::vector<double> out;
    ngram_embedding_vector(out);
    return out;
}

void CyphaLMModel::record_token_history(std::uint32_t token_id) {
    if (token_history_.empty()) {
        return;
    }
    for (std::size_t j = token_history_.size() - 1; j > 0; --j) {
        token_history_[j] = token_history_[j - 1];
    }
    token_history_[0] = token_id;
}

std::uint64_t CyphaLMModel::ngram_context_key() const {
    if (token_history_.empty()) {
        return 0;
    }
    std::uint64_t key = 0;
    const std::uint64_t base = static_cast<std::uint64_t>(std::max(1, cfg_.vocab_size));
    for (std::size_t i = token_history_.size(); i > 0; --i) {
        key = key * base + static_cast<std::uint64_t>(token_history_[i - 1]);
    }
    return key;
}

std::vector<double> CyphaLMModel::ngram_count_log_prior() const {
    std::vector<double> prior(static_cast<std::size_t>(cfg_.vocab_size), 0.0);
    if (!uses_ngram_count_path(cfg_) || token_history_.empty()) {
        return prior;
    }
    const auto it = ngram_count_table_.find(ngram_context_key());
    if (it == ngram_count_table_.end()) {
        return prior;
    }
    const auto& counts = it->second;
    double total = 0.0;
    for (double c : counts) {
        total += c;
    }
    const double denom = total + static_cast<double>(cfg_.vocab_size) * cfg_.laplace_smoothing;
    for (int k = 0; k < cfg_.vocab_size; ++k) {
        const double c =
            (k < static_cast<int>(counts.size())) ? counts[static_cast<std::size_t>(k)] : 0.0;
        prior[static_cast<std::size_t>(k)] =
            std::log((c + cfg_.laplace_smoothing) / denom + kLogEps);
    }
    return prior;
}

void CyphaLMModel::observe_ngram_count(std::uint32_t next_token_id) {
    if (!uses_ngram_count_path(cfg_) || token_history_.empty() ||
        next_token_id >= static_cast<std::uint32_t>(cfg_.vocab_size)) {
        return;
    }
    const std::uint64_t key = ngram_context_key();
    auto& counts = ngram_count_table_[key];
    if (counts.size() < static_cast<std::size_t>(cfg_.vocab_size)) {
        counts.assign(static_cast<std::size_t>(cfg_.vocab_size), 0.0);
    }
    counts[static_cast<std::size_t>(next_token_id)] += 1.0;
}

void CyphaLMModel::refresh_laplace_prior() {
    if (gria_ && cfg_.laplace_smoothing > 0.0 && !token_counts_.empty()) {
        gria_->set_laplace_prior(token_counts_.data(), static_cast<int>(token_counts_.size()),
                                 cfg_.laplace_smoothing);
    }
}

std::vector<double> CyphaLMModel::augment_gria_input(const std::vector<double>& v) const {
    if (cfg_.view_id_dim <= 0 || !view_emb_) return v;
    std::vector<double> out = v;
    const auto vv = view_emb_->forward(current_view_slot_);
    out.insert(out.end(), vv.begin(), vv.end());
    return out;
}

int CyphaLMModel::ssm_context_dim() const {
    if (hierarchical_ssm_) return hierarchical_ssm_->fast_tier().context_dim();
    if (ssm_) return ssm_->context_dim();
    return 0;
}

CellAISSM* CyphaLMModel::active_ssm() {
    if (hierarchical_ssm_) {
        return const_cast<CellAISSM*>(&hierarchical_ssm_->fast_tier());
    }
    return ssm_.get();
}

const CellAISSM* CyphaLMModel::active_ssm() const {
    if (hierarchical_ssm_) return &hierarchical_ssm_->fast_tier();
    return ssm_.get();
}

std::vector<double> CyphaLMModel::ssm_step(const std::vector<double>& e) {
    std::vector<double> ctx;
    if (hierarchical_ssm_) {
        ctx = hierarchical_ssm_->step(e);
    } else if (ssm_) {
        ctx = ssm_->step(e);
    }
    if (cfg_.use_differential_gate && !ctx.empty()) {
        constexpr double kTheta0 = 0.75;
        constexpr double kDeltaBlend = 0.25;
        const CellAISSM* active = active_ssm();
        std::vector<double> delta;
        if (active && !active->h_states().empty()) {
            const auto& h0 = active->h_states()[0];
            if (last_ssm_h_fast_.size() == h0.size()) {
                delta.resize(h0.size());
                for (std::size_t i = 0; i < h0.size(); ++i) {
                    delta[i] = h0[i] - last_ssm_h_fast_[i];
                }
            }
            last_ssm_h_fast_ = h0;
        }
        if (!last_ctx_.empty() && last_ctx_.size() == ctx.size()) {
            for (std::size_t i = 0; i < ctx.size(); ++i) {
                const double dh = (i < delta.size()) ? delta[i] : 0.0;
                ctx[i] = kTheta0 * last_ctx_[i] + kDeltaBlend * (ctx[i] + dh);
            }
        }
    } else {
        const CellAISSM* active = active_ssm();
        if (active && !active->h_states().empty()) {
            last_ssm_h_fast_ = active->h_states()[0];
        }
    }
    if (cfg_.use_ca_state_cell && !ctx.empty()) {
        const auto ca_ctx = CAStateCell::step_rule110(ctx, 1.0);
        for (std::size_t i = 0; i < ctx.size() && i < ca_ctx.size(); ++i) {
            ctx[i] = 0.7 * ctx[i] + 0.3 * ca_ctx[i];
        }
    }
    if (cfg_.use_reversible_cell && reversible_cell_ && !ctx.empty()) {
        const std::vector<double> delta = ctx;
        ctx = reversible_cell_->forward(last_ctx_.empty() ? ctx : last_ctx_, delta);
    }
    return ctx;
}

void CyphaLMModel::apply_hebbian_hooks(std::vector<double>& ctx) {
    if (!hebbian_stack_) return;
    const CellAISSM* active = active_ssm();
    if (!active || active->n_layers() < 1) return;
    const auto& h = active->h_states();
    const auto& s = active->s_states();
    hebbian_stack_->on_ssm_layer_context(ctx, 0, h[0].data(), s[0].data());
}

std::vector<double> CyphaLMModel::project_field(const std::vector<double>& ctx) {
    return matvec(proj_ssm_, cfg_.field_dim, ssm_context_dim(), ctx);
}

std::vector<double> CyphaLMModel::gria_input_core(const std::vector<double>& field,
                                                  const DIFPredictOutput* dif_out) const {
    const auto mode = cfg_.context_mode;
    if (ngram_fusion_ &&
        (mode == ContextMode::Hybrid || mode == ContextMode::GriaNgram ||
         mode == ContextMode::AblationNoSsm || mode == ContextMode::Rpsm ||
         mode == ContextMode::SsmGria)) {
        // Perf (2026-07-12, part 3): fills ngram_embed_vec_scratch_ in place via the out-param
        // overload instead of receiving a fresh vector from ngram_embedding_vector() every
        // predict_next call (this is the default D17 hybrid path's ngram-fuse-split branch).
        ngram_embedding_vector(ngram_embed_vec_scratch_);
        auto fused = ngram_fusion_->forward(field, ngram_embed_vec_scratch_);
        // Upgrade wave 2: ngram_fuse_split previously discarded DIF mean entirely. When expert
        // utilization is forced on, blend projected DIF mean into the ngram GRIA input so soft
        // updates / entropy floor can move hybrid logits (default path remains bit-identical).
        if (dif_out != nullptr && !proj_dif_.empty() &&
            (cfg_.use_soft_expert_updates || cfg_.use_routing_entropy_floor || cfg_.n_experts > 0)) {
            const auto mean_resized = resize_to_field(dif_out->mean, cfg_.field_dim);
            const std::vector<double> mean =
                matvec(proj_dif_, cfg_.field_dim, cfg_.field_dim, mean_resized);
            constexpr double kDifBlend = 0.15;
            for (std::size_t i = 0; i < fused.size() && i < mean.size(); ++i) {
                fused[i] = (1.0 - kDifBlend) * fused[i] + kDifBlend * mean[i];
            }
        }
        return fused;
    }
    if (mode == ContextMode::AblationNoSsm && !proj_ngram_.empty()) {
        const auto embeds = ngram_embedding_vector();
        if (!proj_ngram_embed_.empty()) {
            return matvec(proj_ngram_embed_, cfg_.field_dim, static_cast<int>(embeds.size()),
                          embeds);
        }
        return matvec(proj_ngram_, cfg_.field_dim, static_cast<int>(embeds.size()), embeds);
    }
    if (mode == ContextMode::SsmGria && cfg_.ngram_context <= 0) {
        return field;
    }
    if (dif_out != nullptr && !proj_dif_.empty()) {
        const auto mean_resized = resize_to_field(dif_out->mean, cfg_.field_dim);
        std::vector<double> mean = matvec(proj_dif_, cfg_.field_dim, cfg_.field_dim, mean_resized);
        if (mode == ContextMode::Full || mode == ContextMode::SsmGriaNoLstm) {
            const double u = dif_out->epistemic_var;
            for (std::size_t i = 0; i < mean.size() && i < field.size(); ++i) {
                mean[i] += u * field[i] * 0.01;
            }
        }
        return mean;
    }
    return field;
}

std::vector<double> CyphaLMModel::build_gria_input(const std::vector<double>& field,
                                                   const DIFPredictOutput* dif_out) {
    std::vector<double> v = gria_input_core(field, dif_out);
    if (memory_ && !field.empty()) {
        const auto bias = memory_->retrieve(field.data(), static_cast<std::uint32_t>(field.size()));
        for (std::size_t i = 0; i < bias.size() && i < v.size(); ++i) {
            v[i] += bias[i] * 0.01;
        }
    }
    if (context_bank_ && !embed_history_.empty() && !proj_embed_.empty()) {
        const auto& q = embed_history_[0];
        const auto attn = cfg_.use_tiered_context ? context_bank_->tiered_linear_attention(q)
                                                  : context_bank_->linear_attention(q);
        const auto attn_field = matvec(proj_embed_, cfg_.field_dim, cfg_.d_embed, attn);
        for (std::size_t i = 0; i < attn_field.size() && i < v.size(); ++i) {
            v[i] += attn_field[i] * 0.05;
        }
        context_bank_->push(q.data(), cfg_.d_embed);
    }
    if (cfg_.use_algebraic_fingerprint && !field.empty()) {
        mix_algebraic_fingerprint(v, field.data(), static_cast<int>(field.size()));
    }
    return augment_gria_input(v);
}

void CyphaLMModel::fill_top_k(const std::vector<double>& log_probs, PredictNextOutput& out,
                              int k) const {
    const int n = static_cast<int>(log_probs.size());
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
    std::partial_sort(idx.begin(), idx.begin() + std::min(k, n), idx.end(),
                      [&](int a, int b) { return log_probs[static_cast<std::size_t>(a)] >
                                                 log_probs[static_cast<std::size_t>(b)]; });
    const int m = std::min(k, n);
    out.top_k_tokens.resize(static_cast<std::size_t>(m));
    out.top_k_probs.resize(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        const int id = idx[static_cast<std::size_t>(i)];
        out.top_k_tokens[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(id);
        out.top_k_probs[static_cast<std::size_t>(i)] =
            std::exp(log_probs[static_cast<std::size_t>(id)]);
    }
}

void CyphaLMModel::append_lstm_hidden_history(const std::vector<double>& h) {
    if (h.empty()) {
        return;
    }
    if (static_cast<int>(lstm_h_history_rows_.size()) >= lstm_h_history_max_) {
        lstm_h_history_rows_.pop_front();
    }
    lstm_h_history_rows_.push_back(h);
}

LstmHiddenDEffReport CyphaLMModel::lstm_hidden_d_eff_detail() const {
    LstmHiddenDEffReport report;
    if (lstm_h_history_rows_.size() < 4U || !lstm_) {
        return report;
    }
    const int hidden = lstm_->hidden;
    const int rows = static_cast<int>(lstm_h_history_rows_.size());
    std::vector<double> flat(static_cast<std::size_t>(rows * hidden), 0.0);
    for (int r = 0; r < rows; ++r) {
        const auto& row = lstm_h_history_rows_[static_cast<std::size_t>(r)];
        const int use = std::min(hidden, static_cast<int>(row.size()));
        for (int j = 0; j < use; ++j) {
            flat[static_cast<std::size_t>(r * hidden + j)] = row[static_cast<std::size_t>(j)];
        }
    }
    const auto method = cfg_.use_eigenvalue_d_eff
                             ? cypha::intelligence::ParticipationRatioMethod::CovarianceEigenvalue
                             : cypha::intelligence::ParticipationRatioMethod::VarianceProxy;
    report.raw = cypha::intelligence::compute_participation_ratio_raw(flat.data(), rows, hidden, method);
    report.normalized = std::clamp(report.raw / static_cast<double>(hidden), 0.0, 1.0);
    report.n_samples = rows;
    report.n_dims = hidden;
    report.sample_ratio = static_cast<double>(rows) / static_cast<double>(hidden);
    return report;
}

double CyphaLMModel::lstm_hidden_d_eff() const {
    return lstm_hidden_d_eff_detail().normalized;
}

double CyphaLMModel::hybrid_forget_gate_scale(const DIFPredictOutput& dif_out) const {
    double scale = 1.0;
    if (cfg_.use_alpha_forget_gate && gria_ && !gria_->alpha.empty()) {
        double sum = 0.0;
        for (double a : gria_->alpha) {
            sum += a;
        }
        scale = sum / static_cast<double>(gria_->alpha.size());
    }
    if (cfg_.use_tau_forget_gate) {
        double tau_signal = last_profile_tau_;
        if (tau_signal <= 0.0 && dif_out.epistemic_var >= 0.0) {
            tau_signal = cypha::intelligence::compute_epistemic_ratio(dif_out.epistemic_var,
                                                                      dif_out.aleatoric_var);
        }
        tau_signal = std::clamp(tau_signal, 0.0, 1.0);
        const double tau_scale = 0.5 + 0.5 * tau_signal;
        scale = cfg_.use_alpha_forget_gate ? scale * tau_scale : tau_scale;
    }
    if (cfg_.use_reu_forget_gate) {
        double reu_signal = last_profile_r_eu_;
        if (reu_signal <= 0.0 && dif_out.epistemic_var >= 0.0) {
            reu_signal = cypha::intelligence::compute_epistemic_ratio(dif_out.epistemic_var,
                                                                      dif_out.aleatoric_var);
        }
        reu_signal = std::clamp(reu_signal, 0.0, 1.0);
        const double reu_scale = 0.5 + 0.5 * reu_signal;
        const double blend = std::clamp(cfg_.reu_forget_gate_blend, 0.0, 1.0);
        scale *= (1.0 - blend) + blend * reu_scale;
    }
    return std::clamp(scale, 0.1, 1.5);
}

PredictNextOutput CyphaLMModel::predict_next(std::uint32_t token_id) {
    if (g_predict_trace.enabled) ++g_predict_trace.calls;
    PredictNextOutput out;
    const auto mode = cfg_.context_mode;

    if (mode == ContextMode::CharLstm) {
        if (!lstm_) throw std::runtime_error("char_lstm without LSTM head");
        out.log_probs = lstm_->forward(static_cast<int>(token_id));
        fill_top_k(out.log_probs, out);
        return out;
    }

    // Perf (2026-07-12, part 3): log_g_scratch_ replaces a fresh `std::vector<double>
    // log_g(vocab_size)` local allocated every single predict_next call. `.assign(n, 0.0)`
    // reuses the existing heap buffer (no realloc once warmed up) and, same as the original
    // constructor, zero-initializes it -- so callers that skip the block below (e.g. no
    // `embed_`/`active_ssm()`) see the exact same all-zero log_g as before, not stale data from
    // a previous call.
    log_g_scratch_.assign(static_cast<std::size_t>(cfg_.vocab_size), 0.0);
    std::vector<double>& log_g = log_g_scratch_;
    if (embed_ && (active_ssm() || mode == ContextMode::AblationNoSsm)) {
        std::vector<double> e;
        {
            PredictScopeTimer __t(0);  // 1. embed lookup + history record
            e = embed_->embed_vec(token_id);
            record_embedding(e);
            last_e_ = e;
        }
        {
            PredictScopeTimer __t(1);  // 2. SSM step + hebbian hooks + field projection
            if (mode == ContextMode::AblationNoSsm) {
                last_ctx_.assign(static_cast<std::size_t>(ssm_context_dim()), 0.0);
                field_x_.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
            } else {
                auto ctx = ssm_step(e);
                apply_hebbian_hooks(ctx);
                last_ctx_ = ctx;
                field_x_ = project_field(ctx);
                if (cfg_.use_mdl_forget) {
                    mdl_forget_project(field_x_, cfg_.mdl_forget_max_norm);
                }
                if (nig_state_cell_) {
                    const auto nig_h = nig_state_cell_->step(field_x_.data(), cfg_.field_dim);
                    for (std::size_t i = 0; i < field_x_.size() && i < nig_h.size(); ++i) {
                        field_x_[i] = 0.6 * field_x_[i] + 0.4 * nig_h[i];
                    }
                }
            }
        }
        const bool skip_dif =
            mode == ContextMode::SsmGria || mode == ContextMode::AblationNoSsm;
        const bool skip_dif_subsystem = skip_dif || !dif_subsystem_affects_forward(cfg_);
        {
            PredictScopeTimer __t(2);  // 3. DIF (kernel-LLR) predict
            if (dif_ && !skip_dif_subsystem) {
                last_dif_out_ = dif_->predict(field_x_.data(), static_cast<int>(field_x_.size()));
                out.epistemic_var = last_dif_out_.epistemic_var;
                out.aleatoric_var = last_dif_out_.aleatoric_var;
            } else {
                last_dif_out_ = {};
                // Part 7b: when DIF is skipped (D17 ngram-fusion default), mean is never read —
                // avoid field_dim zero-fill every predict_next (same pattern as Part 4 DIF skip).
                if (!skip_dif_subsystem) {
                    last_dif_out_.mean.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
                }
            }
        }
        {
            PredictScopeTimer __t(3);  // 4. GRIA input build (memory/context-bank/ngram-fusion + rpsm/gng)
            if (selective_ && (mode == ContextMode::SsmGriaNoLstm || mode == ContextMode::Full)) {
                const auto sel = selective_->step(e.data(), static_cast<std::uint32_t>(e.size()));
                for (std::size_t i = 0; i < sel.size() && i < field_x_.size(); ++i) {
                    field_x_[i] = 0.5 * field_x_[i] + 0.5 * sel[i];
                }
                if (memory_) {
                    const auto pooled = selective_->pooled_state();
                    if (cfg_.use_priority_replay) {
                        memory_->maybe_store_priority(step_count_, pooled.data(),
                                                        static_cast<std::uint32_t>(pooled.size()),
                                                        std::abs(last_train_loss_) + 1e-3);
                    } else {
                        memory_->maybe_store(step_count_, pooled.data(),
                                             static_cast<std::uint32_t>(pooled.size()));
                    }
                }
            } else if (memory_) {
                if (cfg_.use_priority_replay) {
                    memory_->maybe_store_priority(step_count_, field_x_.data(),
                                                    static_cast<std::uint32_t>(field_x_.size()),
                                                    std::abs(last_train_loss_) + 1e-3);
                } else {
                    memory_->maybe_store(step_count_, field_x_.data(),
                                         static_cast<std::uint32_t>(field_x_.size()));
                }
            }
            gria_in_ = build_gria_input(field_x_, skip_dif_subsystem ? nullptr : &last_dif_out_);
            if (rpsm_layer_ && !field_x_.empty()) {
                (void)rpsm_layer_->step(field_x_.data(), static_cast<int>(field_x_.size()),
                                        rpsm_log_probs_.data());
                const auto& h = rpsm_layer_->hidden();
                for (std::size_t i = 0; i < gria_in_.size() && !h.empty(); ++i) {
                    gria_in_[i] += 0.05 * h[i % h.size()];
                }
            }
            if (gng_ && !field_x_.empty()) {
                last_gng_bmu_ = gng_->step(field_x_);
            }
        }
        {
            PredictScopeTimer __t(4);  // 5. GRIA field forward (logits+softmax)
            if (gria_) {
                gria_->forward(gria_in_.data(), log_g.data());
                if (uses_ngram_count_path(cfg_)) {
                    const auto ngram_prior = ngram_count_log_prior();
                    for (int k = 0; k < cfg_.vocab_size; ++k) {
                        log_g[static_cast<std::size_t>(k)] += ngram_prior[static_cast<std::size_t>(k)];
                    }
                }
            }
        }
    }

    if (mode == ContextMode::Hybrid && lstm_ && gria_) {
        // Perf (2026-07-12, part 3): log_l_scratch_ replaces a fresh `std::vector<double>
        // log_l(vocab_size)` local every call. `.resize()` (not `.assign()`) is enough here --
        // unlike log_g_scratch_ above, log_l is always fully overwritten by forward_step's
        // `log_probs[k] = ...` loop over every vocab index a few lines below, before it is ever
        // read, so no zero-fill is needed to preserve the original semantics.
        log_l_scratch_.resize(static_cast<std::size_t>(cfg_.vocab_size));
        std::vector<double>& log_l = log_l_scratch_;
        double blend_logit = hybrid_blend_logit_;
        {
            PredictScopeTimer __t(5);  // 6. LSTM forward step (gates) + hidden-state bookkeeping
            // Perf (2026-07-12, part 3): predict_lstm_h_scratch_/predict_lstm_c_scratch_ replace
            // predict_next's previous fresh `h_new`/`c_new` locals. forward_step's own h_out/
            // c_out contract starts with `.assign(hidden, 0.0)` (see char_lstm.cpp), which would
            // corrupt `lstm_h_`/`lstm_c_` if they were passed as both the "previous state" input
            // *and* the output buffer in the same call (the zero-fill would wipe the previous
            // state before the per-index loop below it reads it) -- so the output buffers must
            // stay physically distinct from `lstm_h_`/`lstm_c_` for the duration of the call,
            // exactly as the original fresh-local design did. The difference here is *after* the
            // call: swapping (not move-assigning) `lstm_h_`/`predict_lstm_h_scratch_` means the
            // two buffers simply trade roles call-to-call -- `lstm_h_` ends up holding this
            // step's freshly-written new state (bit-identical to before), and
            // `predict_lstm_h_scratch_` ends up holding the *previous* `lstm_h_` buffer, already
            // correctly sized and ready to be overwritten next call with zero heap allocation.
            // A move-assignment (the original pattern) would instead leave the source empty,
            // forcing a fresh allocation via forward_step's `.assign()` on the very next call.
            const double forget_gate_scale =
                (cfg_.use_alpha_forget_gate || cfg_.use_tau_forget_gate || cfg_.use_reu_forget_gate)
                    ? hybrid_forget_gate_scale(last_dif_out_)
                    : 1.0;
            lstm_->forward_step(static_cast<int>(token_id), lstm_h_.data(), lstm_c_.data(),
                                log_l.data(), predict_lstm_h_scratch_, predict_lstm_c_scratch_,
                                &hybrid_lstm_cache_, forget_gate_scale);
            lstm_h_.swap(predict_lstm_h_scratch_);
            lstm_c_.swap(predict_lstm_c_scratch_);
            append_lstm_hidden_history(lstm_h_);
            hybrid_lstm_has_cache_ = true;
            last_hybrid_log_g_ = log_g;
            last_hybrid_log_l_ = log_l;
        }
        {
            PredictScopeTimer __t(6);  // 7. hybrid blend combination (blend_logit + blend_log_probs)
            if (cfg_.use_gria_gated_mixture && gria_) {
                double mean_alpha = 0.5;
                if (!gria_->alpha.empty()) {
                    double sum = 0.0;
                    for (double a : gria_->alpha) {
                        sum += a;
                    }
                    mean_alpha = sum / static_cast<double>(gria_->alpha.size());
                }
                const double delta_alpha = mean_alpha - last_mean_alpha_;
                blend_logit = gria_gated_blend_logit(blend_logit, mean_alpha, delta_alpha);
                last_mean_alpha_ = mean_alpha;
            }
            if (cfg_.use_ood_branching && out.epistemic_var > 0.0) {
                constexpr double kOodThreshold = 0.05;
                if (out.epistemic_var > kOodThreshold) {
                    const double ood_scale = std::min(2.0, out.epistemic_var / kOodThreshold);
                    blend_logit -= 0.35 * ood_scale;
                }
            }
            out.log_probs = blend_log_probs(log_g, log_l, blend_logit);
        }
    } else if (uses_lstm(mode) && lstm_ && !gria_) {
        out.log_probs = lstm_->forward(static_cast<int>(token_id));
    } else if (mode == ContextMode::Rpsm && rpsm_layer_ && !rpsm_log_probs_.empty()) {
        out.log_probs = rpsm_log_probs_;
    } else {
        // Copy (not move) from log_g_scratch_ -- it's a persistent scratch member now (see
        // above), not a one-off local, so it must stay intact for reuse on the next call.
        out.log_probs = log_g;
    }

    {
        PredictScopeTimer __t(7);  // 8. tail (token history + top-k + bookkeeping)
        if (mode != ContextMode::CharLstm) {
            record_token_history(token_id);
        }

        ++step_count_;
        fill_top_k(out.log_probs, out);
    }
    return out;
}

PredictNextOutput CyphaLMModel::repredict_hybrid_blend(double blend_logit) const {
    PredictNextOutput out;
    out.epistemic_var = last_dif_out_.epistemic_var;
    out.aleatoric_var = last_dif_out_.aleatoric_var;
    if (!last_hybrid_log_g_.empty() && !last_hybrid_log_l_.empty()) {
        out.log_probs = blend_log_probs(last_hybrid_log_g_, last_hybrid_log_l_, blend_logit);
        fill_top_k(out.log_probs, out);
    }
    return out;
}

namespace {
double max_prob_confidence(const std::vector<double>& log_probs) {
    if (log_probs.empty()) return 0.0;
    return std::exp(*std::max_element(log_probs.begin(), log_probs.end()));
}
}  // namespace

PredictNextOutput CyphaLMModel::self_correct_if_needed(
    const PredictNextOutput& initial, cypha::intelligence::EpistemicThreshold& threshold) const {
    if (!cfg_.use_self_correcting_loop || cfg_.context_mode != ContextMode::Hybrid) {
        return initial;
    }
    const double r_eu =
        cypha::intelligence::compute_epistemic_ratio(initial.epistemic_var, initial.aleatoric_var);
    if (!threshold.should_correct(r_eu)) {
        return initial;
    }
    // Mirrors the LM-native self-correction algorithm already used on the generation/decode
    // path (`self_correct_predict` in cyphalm_generation.cpp): widen the hybrid blend toward
    // the LSTM head over up to `kMaxSelfCorrectPasses` passes, keeping whichever pass has
    // higher max-softmax confidence. Re-blends cached logits via `repredict_hybrid_blend`
    // (no extra forward pass), so this is cheap relative to `predict_next`.
    constexpr int kMaxSelfCorrectPasses = 3;
    PredictNextOutput best = initial;
    double best_conf = max_prob_confidence(best.log_probs);
    double pass_blend = hybrid_blend_logit_;
    int passes = 1;
    bool corrected = false;
    while (passes < kMaxSelfCorrectPasses && threshold.should_correct(r_eu)) {
        pass_blend = pass_blend * 0.82 + 0.05;
        const PredictNextOutput retry = repredict_hybrid_blend(pass_blend);
        const double retry_conf = max_prob_confidence(retry.log_probs);
        if (retry_conf > best_conf) {
            best = retry;
            best_conf = retry_conf;
            corrected = true;
        }
        ++passes;
    }
    threshold.update(r_eu, corrected);
    return best;
}

void CyphaLMModel::ewc_snapshot() {
    ewc_.snapshot(lstm_.get(), active_ssm(), gria_.get());
}

double CyphaLMModel::ewc_penalty() const {
    return ewc_.penalty(lstm_.get(), active_ssm(), gria_.get());
}

void CyphaLMModel::apply_lstm_ewc(TrainStepMetrics& m, const CharLSTMGrad& grads) {
    HybridEwcGradStub stub;
    stub.has_lstm = true;
    stub.lstm = grads;
    apply_hybrid_ewc(m, stub);
}

void CyphaLMModel::apply_hybrid_ewc(TrainStepMetrics& m, const HybridEwcGradStub& grads) {
    if (cfg_.ewc_lambda <= 0.0) {
        return;
    }
    ewc_.observe_grads(grads);
    if (ewc_.has_snapshot()) {
        const double pen = ewc_.penalty(lstm_.get(), active_ssm(), gria_.get());
        m.loss += cfg_.ewc_lambda * pen;
        m.ewc_penalty = pen;
        ewc_.apply_pull(lstm_.get(), active_ssm(), gria_.get(), cfg_.ewc_lambda, cfg_.lstm_lr,
                        cfg_.gria_lr, cfg_.ssm_lr);
    }
}

TrainStepMetrics CyphaLMModel::train_step(std::uint32_t token_id, std::uint32_t next_token_id,
                                            cypha::intelligence::IntelligenceProfiler* profiler,
                                            LmIntelligenceMonitor* monitor) {
    if (g_perf_trace.enabled) ++g_perf_trace.calls;
    const auto perf_t0_predict = perf_trace_begin();
    const auto pred = predict_next(token_id);
    perf_trace_end(0, perf_t0_predict);
    TrainStepMetrics m;
    m.loss = -pred.log_probs[static_cast<std::size_t>(next_token_id)];
    m.epistemic_var = pred.epistemic_var;
    m.aleatoric_var = pred.aleatoric_var;
    m.active_experts = last_dif_out_.active_experts;
    last_train_loss_ = m.loss;
    if (cfg_.use_gria_gated_mixture) {
        m.alpha_gria = last_mean_alpha_;
    }
    if (cfg_.use_free_energy_loss && m.epistemic_var > 0.0) {
        m.free_energy_penalty = cfg_.free_energy_beta * m.epistemic_var;
        m.loss += m.free_energy_penalty;
    }
    if (cfg_.use_routing_entropy_floor && !last_dif_out_.routing_probs.empty() &&
        last_dif_out_.routing_probs.size() > 1) {
        double ent = 0.0;
        for (double pi : last_dif_out_.routing_probs) {
            if (pi > 1e-12) ent -= pi * std::log(pi);
        }
        const double max_ent = std::log(static_cast<double>(last_dif_out_.routing_probs.size()));
        const double floor = max_ent * cfg_.routing_entropy_floor_frac;
        if (ent < floor) {
            m.loss += cfg_.routing_entropy_lambda * (floor - ent);
        }
    }

    cypha::intelligence::IntelligenceProfiler* active_profiler = profiler;
    if (active_profiler == nullptr && uses_profile_guided_backprop(cfg_) && train_profiler_) {
        active_profiler = train_profiler_.get();
    }
    if (monitor != nullptr) {
        monitor->observe_token(last_e_, field_x_, pred.log_probs, pred.epistemic_var, pred.aleatoric_var,
                               static_cast<std::int64_t>(next_token_id),
                               static_cast<int>(cfg_.vocab_size));
        last_profile_tau_ = monitor->snapshot_observation().tau;
        last_profile_r_eu_ = monitor->snapshot_observation().r_eu;
    }
    if (active_profiler != nullptr && monitor == nullptr) {
        update_profiler_from_lm_token(*active_profiler, last_e_, pred.log_probs, pred.epistemic_var,
                                      pred.aleatoric_var);
    }

    auto apply_profile_guided = [&](GRIALowRankGrad* grad_out, double& blend_nudge_out,
                                    double& logit_nudge_out, double& hidden_nudge_out) {
        blend_nudge_out = 0.0;
        logit_nudge_out = 0.0;
        hidden_nudge_out = 0.0;
        if (!uses_profile_guided_backprop(cfg_)) {
            return;
        }
        auto pg_cfg = profile_guided_loss_config_for(cfg_);
        cypha::intelligence::ProfileObservation monitor_obs{};
        const bool have_monitor_obs = monitor != nullptr;
        if (have_monitor_obs) {
            monitor_obs = monitor->snapshot_observation();
        }
        cypha::intelligence::ProfileObservation scale_obs{};
        bool have_scale_obs = false;
        if (have_monitor_obs) {
            scale_obs = monitor_obs;
            have_scale_obs = true;
        } else if (active_profiler != nullptr) {
            const auto matrix = active_profiler->get_profile_matrix();
            scale_obs.alpha = matrix[0][0];
            scale_obs.d_eff = matrix[1][0];
            scale_obs.sigma_branch = matrix[2][0];
            scale_obs.tau = matrix[3][0];
            scale_obs.r_eu = matrix[4][0];
            scale_obs.lipschitz = matrix[5][0];
            scale_obs.calibration = matrix[6][0];
            have_scale_obs = true;
        }
        if (have_scale_obs && (cfg_.use_adaptive_navigation_lambdas ||
                               cfg_.use_per_stat_deviation_lambdas ||
                               cfg_.use_kappa_ceiling_lambdas)) {
            cypha::intelligence::AdaptiveNavigationOptions nav_opts;
            nav_opts.use_adaptive_lambdas = cfg_.use_adaptive_navigation_lambdas;
            nav_opts.use_trajectory_lambdas =
                cfg_.use_kappa_trajectory_lambdas && have_monitor_obs;
            nav_opts.use_per_stat_deviation_lambdas = cfg_.use_per_stat_deviation_lambdas;
            nav_opts.use_kappa_ceiling_lambdas = cfg_.use_kappa_ceiling_lambdas;
            nav_opts.kappa_ceiling_strength = cfg_.kappa_ceiling_strength;
            nav_opts.kappa_ceiling_min_scale = cfg_.kappa_ceiling_min_scale;
            nav_opts.use_kappa_trajectory_ceiling = cfg_.use_kappa_trajectory_ceiling;
            nav_opts.target_kappa = cfg_.kappa_lambda_target;
            nav_opts.trajectory_window = cfg_.kappa_trajectory_window;
            nav_opts.deviation_span = cfg_.per_stat_deviation_span;
            pg_cfg = cypha::intelligence::resolve_adaptive_profile_guided_config(
                pg_cfg, scale_obs, nav_opts,
                nav_opts.use_trajectory_lambdas ? &kappa_trajectory_state_ : nullptr);
        }
        cypha::intelligence::ProfileGuidedLossTerms pg;
        cypha::intelligence::ProfileGuidedLossGrad pg_grad;
        if (cfg_.use_full_navigation_loss && have_monitor_obs) {
            pg = cypha::intelligence::compute_profile_guided_loss(monitor_obs, pg_cfg);
            pg_grad = cypha::intelligence::compute_profile_guided_loss_grad(
                monitor_obs, pg_cfg, static_cast<int>(cfg_.field_dim));
        } else if (active_profiler != nullptr) {
            pg = cypha::intelligence::compute_profile_guided_loss_from_profiler(*active_profiler, pg_cfg);
            pg_grad = cypha::intelligence::compute_profile_guided_loss_grad_from_profiler(
                *active_profiler, pg_cfg, static_cast<int>(cfg_.field_dim));
        } else {
            return;
        }
        if (cfg_.use_kappa_excess_grad_nudge && have_scale_obs) {
            const double kappa =
                cypha::intelligence::IntelligenceProfiler::criticality_score_for(scale_obs);
            const double nudge_strength =
                cfg_.kappa_ceiling_strength * cfg_.kappa_excess_grad_scale;
            const auto excess_grad = cypha::intelligence::kappa_excess_grad_nudge(
                scale_obs, kappa, cfg_.kappa_lambda_target, nudge_strength,
                cfg_.kappa_excess_grad_margin);
            pg_grad.d_alpha_uniform += excess_grad.d_alpha_uniform;
            pg_grad.d_logit_uniform += excess_grad.d_logit_uniform;
            pg_grad.d_h_hidden_uniform += excess_grad.d_h_hidden_uniform;
        }
        double warmup = 1.0;
        if (cfg_.navigation_loss_warmup_steps > 0) {
            warmup = std::min(
                1.0, static_cast<double>(step_count_ + 1) /
                         static_cast<double>(cfg_.navigation_loss_warmup_steps));
        }
        if (cfg_.use_kappa_navigation_warmup_scale && have_scale_obs) {
            const double kappa =
                cypha::intelligence::IntelligenceProfiler::criticality_score_for(scale_obs);
            warmup = cypha::intelligence::scale_navigation_warmup_from_kappa(
                warmup, kappa, cfg_.kappa_lambda_target, cfg_.kappa_navigation_warmup_strength,
                cfg_.kappa_navigation_warmup_floor);
        }
        pg.total *= warmup;
        pg.navigation_loss_total *= warmup;
        m.profile_guided_loss = pg.total;
        m.loss += pg.total;
        blend_nudge_out = pg_grad.d_alpha_uniform * warmup;
        logit_nudge_out = pg_grad.d_logit_uniform * warmup;
        if (cfg_.use_lstm_d_eff_hidden_nudge && lstm_) {
            double h_d_eff = lstm_hidden_d_eff();
            if (h_d_eff < 0.0 && have_scale_obs) {
                h_d_eff = scale_obs.d_eff;
            }
            if (h_d_eff >= 0.0) {
                const double d_d_eff = h_d_eff - pg_cfg.target_d_eff;
                hidden_nudge_out = 2.0 * pg_cfg.lambda_d_eff * d_d_eff * 0.08 * warmup;
            } else {
                hidden_nudge_out = pg_grad.d_h_hidden_uniform * warmup;
            }
        }
        if (grad_out != nullptr) {
            for (auto& d_alpha : grad_out->d_alpha) {
                d_alpha += pg_grad.d_alpha_uniform * warmup;
            }
            if (!pg_grad.d_gria_input.empty() && grad_out->dv.size() == pg_grad.d_gria_input.size()) {
                for (std::size_t gi = 0; gi < grad_out->dv.size(); ++gi) {
                    grad_out->dv[gi] += pg_grad.d_gria_input[gi] * warmup;
                }
            }
        }
    };

    const auto mode = cfg_.context_mode;
    GRIALowRankGrad gria_grad;
    bool has_gria_grad = false;
    double navigation_blend_nudge = 0.0;
    double navigation_logit_nudge = 0.0;
    double navigation_hidden_nudge = 0.0;
    if (mode == ContextMode::CharLstm && lstm_) {
        apply_profile_guided(nullptr, navigation_blend_nudge, navigation_logit_nudge,
                             navigation_hidden_nudge);
        CharLSTMGrad grads;
        const double logit_nudge =
            cfg_.use_full_navigation_loss ? navigation_logit_nudge : 0.0;
        const double hidden_nudge =
            cfg_.use_full_navigation_loss ? navigation_hidden_nudge : 0.0;
        lstm_->backward(static_cast<int>(next_token_id), cfg_.lstm_lr, &grads, logit_nudge,
                       hidden_nudge);
        if (cfg_.use_full_navigation_loss && navigation_blend_nudge != 0.0 &&
            cfg_.lstm_bptt_steps <= 1) {
            const int hidden = lstm_->hidden;
            const double delta = cfg_.lstm_lr * navigation_blend_nudge * 0.02;
            for (int j = hidden; j < 2 * hidden; ++j) {
                lstm_->b[static_cast<std::size_t>(j)] -= delta;
            }
        }
        if (!grads.dWx.empty()) {
            apply_lstm_ewc(m, grads);
        }
        if (next_token_id < token_counts_.size()) {
            token_counts_[static_cast<std::size_t>(next_token_id)] += 1.0;
        }
        return m;
    }
    perf_trace_scope(1, [&]() {
        if (gria_ && !gria_in_.empty()) {
            gria_grad = gria_->cross_entropy_gradients(gria_in_.data(), static_cast<int>(next_token_id));
            has_gria_grad = true;
            apply_profile_guided(&gria_grad, navigation_blend_nudge, navigation_logit_nudge,
                                 navigation_hidden_nudge);
            gria_->update_weights(gria_grad, cfg_.gria_lr);
            gria_->update_alpha(gria_grad, cfg_.gria_lr);
            gria_->update_bias(gria_grad, cfg_.gria_lr);
            if (ngram_fusion_ && cfg_.ngram_position_weights &&
                !ngram_fusion_->pos_weights().empty() && !gria_grad.dv.empty()) {
                ngram_embedding_vector(ngram_embed_vec_scratch_);
                ngram_fusion_->update_position_weights(gria_grad.dv, ngram_embed_vec_scratch_,
                                                       cfg_.gria_lr);
            }
            if (view_emb_ && cfg_.view_learnable) {
                const auto grad_v = gria_->grad_v_cross_entropy(gria_in_.data(), static_cast<int>(next_token_id));
                view_emb_->update(current_view_slot_, grad_v.data() + cfg_.field_dim, cfg_.view_id_dim,
                                  cfg_.view_lr);
            }
        } else {
            apply_profile_guided(nullptr, navigation_blend_nudge, navigation_logit_nudge,
                                 navigation_hidden_nudge);
        }
    });
    perf_trace_scope(2, [&]() {
        if (cfg_.online && dif_ && embed_ && !proj_embed_.empty() &&
            dif_subsystem_affects_forward(cfg_)) {
            if (cfg_.use_kappa_kernel_blend_scale && cfg_.use_kernel_llr) {
                cypha::intelligence::ProfileObservation scale_obs{};
                bool have_scale_obs = false;
                if (monitor != nullptr) {
                    scale_obs = monitor->snapshot_observation();
                    have_scale_obs = true;
                } else if (active_profiler != nullptr) {
                    const auto matrix = active_profiler->get_profile_matrix();
                    scale_obs.alpha = matrix[0][0];
                    scale_obs.d_eff = matrix[1][0];
                    scale_obs.sigma_branch = matrix[2][0];
                    scale_obs.tau = matrix[3][0];
                    scale_obs.r_eu = matrix[4][0];
                    scale_obs.lipschitz = matrix[5][0];
                    scale_obs.calibration = matrix[6][0];
                    have_scale_obs = true;
                }
                if (have_scale_obs) {
                    const double kappa =
                        cypha::intelligence::IntelligenceProfiler::criticality_score_for(scale_obs);
                    const double effective = cypha::intelligence::scale_kernel_blend_from_kappa(
                        cfg_.kernel_blend, kappa, cfg_.kappa_lambda_target,
                        cfg_.kappa_kernel_blend_floor);
                    dif_->set_runtime_kernel_blend(effective);
                }
            }
            const auto target = matvec(proj_embed_, cfg_.field_dim, cfg_.d_embed,
                                       embed_->embed_vec(next_token_id));
            dif_->train_step(field_x_.data(), static_cast<int>(field_x_.size()), target.data(),
                             static_cast<int>(target.size()));
            if (cfg_.use_kappa_kernel_blend_scale && cfg_.use_kernel_llr) {
                dif_->set_runtime_kernel_blend(cfg_.kernel_blend);
            }
        }
    });
    perf_trace_scope(3, [&]() {
        if (hebbian_stack_ && cfg_.train_ssm && !field_x_.empty() && !last_e_.empty()) {
            const int nxt = static_cast<int>(next_token_id);
            hebbian_stack_->encoder_train_step(field_x_.data(), last_e_.data(), std::to_string(nxt),
                                               std::to_string(nxt), cfg_.ssm_lr);
        }
    });
    perf_trace_scope(4, [&]() { bptt_ssm_update(next_token_id); });
    perf_trace_scope(7, [&]() {
        if (cfg_.use_reversible_cell && reversible_cell_ && reversible_cell_->has_pair()) {
            // NOTE: reconstruct()'s result is intentionally unused today — this call exercises the
            // RevNet-style reconstruct path (H11) but nothing currently consumes x_hat for a
            // memory-savings recompute-on-backward / gradient-checkpointing pattern (the usual
            // reason a RevNet reconstruct exists). See docs/reports/STUB_AUDIT_2026-07-11.md.
            (void)reversible_cell_->reconstruct();
        }
    });
    perf_trace_scope(5, [&]() {
        if (mode == ContextMode::Hybrid && lstm_) {
            if (hybrid_lstm_has_cache_) {
                // BPTT-1 (default): same out-param hot path as before. BPTT>1: window flush via
                // push_bptt_step (opt-in Quality Wave 1; pin path unchanged).
                const double logit_nudge =
                    cfg_.use_full_navigation_loss ? navigation_logit_nudge : 0.0;
                const double hidden_nudge =
                    cfg_.use_full_navigation_loss ? navigation_hidden_nudge : 0.0;
                const bool updated = lstm_->push_bptt_step(
                    hybrid_lstm_cache_, static_cast<int>(next_token_id), cfg_.lstm_lr,
                    &hybrid_lstm_grad_scratch_, logit_nudge, hidden_nudge);
                if (updated) {
                    HybridEwcGradStub stub;
                    stub.has_lstm = true;
                    stub.lstm = hybrid_lstm_grad_scratch_;
                    if (has_gria_grad) {
                        stub.d_gria_alpha = gria_grad.d_alpha;
                        stub.d_gria_U = gria_grad.dU;
                        stub.d_gria_V = gria_grad.dV;
                        stub.d_gria_bias = gria_grad.d_bias;
                    }
                    apply_hybrid_ewc(m, stub);
                } else if (has_gria_grad) {
                    HybridEwcGradStub stub;
                    stub.d_gria_alpha = gria_grad.d_alpha;
                    stub.d_gria_U = gria_grad.dU;
                    stub.d_gria_V = gria_grad.dV;
                    stub.d_gria_bias = gria_grad.d_bias;
                    apply_hybrid_ewc(m, stub);
                }
                hybrid_lstm_has_cache_ = false;
            }
            if (cfg_.hybrid_blend_learnable && !last_hybrid_log_g_.empty() && !last_hybrid_log_l_.empty()) {
                hybrid_blend_logit_ -= cfg_.hybrid_blend_lr *
                                       blend_logit_grad(last_hybrid_log_g_.data(), last_hybrid_log_l_.data(),
                                                        cfg_.vocab_size, hybrid_blend_logit_,
                                                        static_cast<int>(next_token_id));
            }
            if (cfg_.hybrid_blend_learnable && cfg_.use_full_navigation_loss && navigation_blend_nudge != 0.0) {
                hybrid_blend_logit_ -= cfg_.hybrid_blend_lr * navigation_blend_nudge * 0.02;
            }
        }
    });
    perf_trace_scope(6, [&]() {
        if (rpsm_layer_ && mode == ContextMode::Rpsm && !field_x_.empty()) {
            const auto rpsm_m = rpsm_layer_->train_step(field_x_.data(), static_cast<int>(field_x_.size()),
                                                        static_cast<int>(next_token_id), cfg_.rpsm_lr);
            m.loss += rpsm_m.loss;
        }
    });
    perf_trace_scope(7, [&]() {
        if (cfg_.ewc_lambda > 0.0 && uses_hybrid_ewc(mode) && mode != ContextMode::Hybrid &&
            mode != ContextMode::CharLstm && has_gria_grad) {
            HybridEwcGradStub stub;
            stub.d_gria_alpha = gria_grad.d_alpha;
            stub.d_gria_U = gria_grad.dU;
            stub.d_gria_V = gria_grad.dV;
            stub.d_gria_bias = gria_grad.d_bias;
            apply_hybrid_ewc(m, stub);
        }
        observe_ngram_count(next_token_id);
        if (next_token_id < token_counts_.size()) {
            token_counts_[static_cast<std::size_t>(next_token_id)] += 1.0;
        }
        if (gria_controller_ && gng_ && !field_x_.empty()) {
            std::vector<double> act;
            if (!last_dif_out_.mean.empty()) {
                act = last_dif_out_.mean;
            } else if (!pred.log_probs.empty()) {
                act = pred.log_probs;
            } else {
                act = field_x_;
            }
            gria_controller_->push(field_x_, act);
            (void)gria_controller_->act(last_gng_bmu_, *gng_);
        }
        refresh_laplace_prior();
    });
    return m;
}

void CyphaLMModel::bptt_ssm_update(std::uint32_t next_token_id) {
    CellAISSM* active = active_ssm();
    if (cfg_.bptt_steps <= 0 || !active || !gria_ || last_e_.empty() || last_ctx_.empty()) {
        return;
    }
    const auto mode = cfg_.context_mode;
    if (mode == ContextMode::AblationNoSsm || mode == ContextMode::AblationNoDif) {
        return;
    }
    if (mode != ContextMode::Hybrid && mode != ContextMode::GriaNgram &&
        mode != ContextMode::SsmGria && mode != ContextMode::Full &&
        mode != ContextMode::SsmGriaNoLstm && mode != ContextMode::Rpsm) {
        return;
    }
    // Perf (2026-07-12, part 2, docs/reports/PERFORMANCE_PROFILE_2026-07-12.md "Follow-up"):
    // grad_core/grad_field/delta_rows/inv_v/grad_ctx were previously fresh-allocated `std::vector`
    // locals every call (this function runs once per train_step whenever BPTT-on-SSM is active,
    // same hot-loop frequency as CharLSTMHead's forward/backward_step). Converted to persistent
    // CyphaLMModel scratch members (see their declarations for the safety argument, same as
    // hybrid_lstm_grad_scratch_) and filled via `.assign()`/`.resize()`/an out-param
    // matvec_transpose overload so their backing storage is reused call-to-call once warmed up.
    // `grad_v` (from `gria_->grad_v_cross_entropy`) and `grad_field_x`'s return (from
    // `ngram_fusion_`) still allocate inside those other classes' own code -- out of scope here
    // (different files/classes; see docs write-up), but note that even in the common D17
    // (`Hybrid`) config, `grad_core` itself is *not* moved away (it's passed by const-ref into
    // `grad_field_x`), so making it a persistent member is a real, unconditional win there.
    auto grad_v = gria_->grad_v_cross_entropy(gria_in_.data(), static_cast<int>(next_token_id));
    std::vector<double>& grad_core = bptt_grad_core_scratch_;
    grad_core.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
    for (int i = 0; i < cfg_.field_dim && i < static_cast<int>(grad_v.size()); ++i) {
        grad_core[static_cast<std::size_t>(i)] = grad_v[static_cast<std::size_t>(i)];
    }
    std::vector<double>& grad_field = bptt_grad_field_scratch_;
    if (ngram_fusion_ &&
        (mode == ContextMode::Hybrid || mode == ContextMode::GriaNgram ||
         mode == ContextMode::AblationNoSsm || mode == ContextMode::SsmGria)) {
        grad_field = ngram_fusion_->grad_field_x(grad_core);
    } else if (mode == ContextMode::AblationNoSsm && !proj_ngram_embed_.empty()) {
        matvec_transpose(proj_ngram_embed_, cfg_.field_dim,
                        static_cast<int>(proj_ngram_embed_.size()) / cfg_.field_dim, grad_core,
                        grad_field);
    } else if (mode == ContextMode::Full || mode == ContextMode::SsmGriaNoLstm) {
        const double u = last_dif_out_.epistemic_var;
        grad_field.resize(grad_core.size());
        for (std::size_t i = 0; i < grad_core.size(); ++i) {
            grad_field[i] = grad_core[i] * (u * 0.01);
        }
    } else {
        // Not moved (unlike the pre-fix version): grad_core is a persistent member now, so
        // moving it away here would force it to reallocate from scratch on this mode's *next*
        // call. A copy costs one alloc in this (Rpsm-only) branch -- no worse than the pre-fix
        // code's fresh-`grad_core`-every-call baseline, and keeps the member reusable elsewhere.
        grad_field = grad_core;
    }
    if (discriminative_feedback_ && dif_ && !grad_field.empty()) {
        std::vector<double>& delta_rows = bptt_delta_rows_scratch_;
        std::vector<double>& inv_v = bptt_inv_v_scratch_;
        if (dif_->discriminative_state(delta_rows, inv_v)) {
            const int K = static_cast<int>(delta_rows.size()) / cfg_.field_dim;
            const auto d = discriminative_feedback_->compute_d(delta_rows, K, cfg_.field_dim, inv_v);
            discriminative_feedback_->modulate_inplace(grad_field, d);
        }
    }
    const int ctx_dim = active->context_dim();
    std::vector<double>& grad_ctx = bptt_grad_ctx_scratch_;
    matvec_transpose(proj_ssm_, cfg_.field_dim, ctx_dim, grad_field, grad_ctx);
    if (static_cast<int>(grad_ctx.size()) < active->d_state()) return;
    // Perf: grad_h/grad_s used to be full copies of a sub-range of grad_ctx (grad_h =
    // grad_ctx[0:d_state), grad_s = grad_ctx[d_state:2*d_state)) purely so the loops below could
    // write `grad_h[r]`/`grad_s[r]` instead of `grad_ctx[r]`/`grad_ctx[d_state + r]`. Indexing
    // grad_ctx directly is the exact same values, zero arithmetic change, and removes two more
    // fresh heap allocations per call.
    const double lf = active->lambda_fast();
    const double ls = active->lambda_slow();
    std::vector<double> delta(static_cast<std::size_t>(active->d_state() * cfg_.d_embed), 0.0);
    for (int r = 0; r < active->d_state(); ++r) {
        for (int c = 0; c < cfg_.d_embed; ++c) {
            delta[static_cast<std::size_t>(r * cfg_.d_embed + c)] =
                (1.0 - lf) * grad_ctx[static_cast<std::size_t>(r)] * last_e_[static_cast<std::size_t>(c)];
        }
    }
    // Slow tier: mirror outer-product delta for Fisher observe only (W_slow is not BPTT-updated).
    // Perf (2026-07-17, part 6): at the locked D17 default (`ewc_lambda=0`), avg_slow is never
    // consumed — only `ewc_.observe_grads(stub.d_ssm_w_slow=...)` reads it, and that block is
    // gated on `ewc_lambda > 0`. Skip delta_slow / bptt_slow_buffer_ entirely when EWC is off
    // (bit-identical: dead work on the default hybrid path, same pattern as Part 4's DIF skip).
    const bool bptt_slow_for_ewc = cfg_.ewc_lambda > 0.0 && uses_hybrid_ewc(mode);
    std::vector<double> delta_slow;
    if (bptt_slow_for_ewc && static_cast<int>(grad_ctx.size()) >= 2 * active->d_state()) {
        const int d_state = active->d_state();
        delta_slow.assign(static_cast<std::size_t>(d_state * cfg_.d_embed), 0.0);
        for (int r = 0; r < d_state; ++r) {
            for (int c = 0; c < cfg_.d_embed; ++c) {
                delta_slow[static_cast<std::size_t>(r * cfg_.d_embed + c)] =
                    (1.0 - ls) * grad_ctx[static_cast<std::size_t>(d_state + r)] * last_e_[static_cast<std::size_t>(c)];
            }
        }
    }
    bptt_buffer_.push_back(std::move(delta));
    if (!delta_slow.empty()) {
        bptt_slow_buffer_.push_back(std::move(delta_slow));
    }
    if (static_cast<int>(bptt_buffer_.size()) < cfg_.bptt_steps) return;
    std::vector<double> avg = bptt_buffer_.front();
    for (std::size_t i = 1; i < bptt_buffer_.size(); ++i) {
        for (std::size_t j = 0; j < avg.size(); ++j) {
            avg[j] += bptt_buffer_[i][j];
        }
    }
    for (double& v : avg) v /= static_cast<double>(bptt_buffer_.size());
    if (bptt_slow_for_ewc) {
        std::vector<double> avg_slow;
        if (bptt_slow_buffer_.size() == bptt_buffer_.size() && !bptt_slow_buffer_.empty()) {
            avg_slow = bptt_slow_buffer_.front();
            for (std::size_t i = 1; i < bptt_slow_buffer_.size(); ++i) {
                for (std::size_t j = 0; j < avg_slow.size(); ++j) {
                    avg_slow[j] += bptt_slow_buffer_[i][j];
                }
            }
            for (double& v : avg_slow) v /= static_cast<double>(bptt_slow_buffer_.size());
        }
        HybridEwcGradStub stub;
        stub.d_ssm_w_fast = avg;
        if (!avg_slow.empty()) {
            stub.d_ssm_w_slow = avg_slow;
        }
        ewc_.observe_grads(stub);
    }
    active->apply_bptt_delta_avg(avg, cfg_.ssm_lr);
    bptt_buffer_.clear();
    bptt_slow_buffer_.clear();
}

std::vector<double> CyphaLMModel::forward_log_probs(std::uint32_t token_id) {
    return predict_next(token_id).log_probs;
}

namespace {

int view_slot_for_name(const std::string& name) {
    if (name == "forward") return 0;
    if (name == "block_shuffle") return 1;
    if (name == "rotated" || name == "rotate_start") return 2;
    if (name == "backward" || name == "reverse") return 3;
    return 0;
}

}  // namespace

void CyphaLMModel::train_sequence_views(const std::vector<int>& ids,
                                          cypha::intelligence::IntelligenceProfiler* profiler) {
    if (ids.size() < 2) return;
    LmIntelligenceMonitor monitor;
    monitor.set_use_eigenvalue_d_eff(cfg_.use_eigenvalue_d_eff);
    const bool track_profiler = profiler != nullptr || uses_profile_guided_backprop(cfg_);
    cypha::intelligence::IntelligenceProfiler* flush_profiler = profiler;
    if (flush_profiler == nullptr && uses_profile_guided_backprop(cfg_) && train_profiler_) {
        flush_profiler = train_profiler_.get();
    }
    const auto schedule =
        resolve_view_schedule_struct(cfg_.view_schedule, cfg_.seed, std::max(1, cfg_.train_epochs));
    const auto epochs = iter_view_epochs(ids, schedule, std::nullopt, cfg_.view_block_size);
    const double base_gria_lr = cfg_.gria_lr;
    int last_macro = -1;
    int step_i = 0;
    const char* log_env = std::getenv("CYPHALM_TRAIN_LOG_EVERY");
    const int log_every = log_env ? std::max(0, std::atoi(log_env)) : 0;
    for (const auto& item : epochs) {
        if (view_emb_) {
            set_view_slot(view_emb_->slot_for_view(item.view_spec.name));
        } else {
            set_view_slot(view_slot_for_name(item.view_spec.name));
        }
        if (item.epoch_idx != last_macro) {
            reset_context();
            last_macro = item.epoch_idx;
            if (flush_profiler != nullptr) {
                monitor.flush_to_profiler(*flush_profiler);
                monitor.reset();
            }
        } else if (item.reset_before) {
            reset_context();
        }
        cfg_.gria_lr =
            base_gria_lr * std::pow(cfg_.gria_lr_decay, static_cast<double>(item.epoch_idx));
        const int steps = static_cast<int>(item.segment_ids.size()) - 1;
        for (int i = 0; i < steps; ++i) {
            const auto m = train_step(
                static_cast<std::uint32_t>(item.segment_ids[static_cast<std::size_t>(i)]),
                static_cast<std::uint32_t>(item.segment_ids[static_cast<std::size_t>(i + 1)]),
                profiler, track_profiler ? &monitor : nullptr);
            ++step_i;
            if (log_every > 0 && step_i % log_every == 0) {
                std::cerr << "[CyphaLM] train step " << step_i << " view=" << item.view_spec.name
                          << " loss=" << m.loss << std::endl;
            }
        }
    }
    if (flush_profiler != nullptr) {
        monitor.flush_to_profiler(*flush_profiler);
        monitor.reset();
    }
    cfg_.gria_lr = base_gria_lr;
}

std::vector<std::uint32_t> CyphaLMModel::encode_text(const std::string& text) const {
    if (!bpe_) {
        throw std::runtime_error("CyphaLMModel: BPE tokenizer not configured");
    }
    return bpe_->encode(text);
}

std::string CyphaLMModel::decode_tokens(const std::vector<std::uint32_t>& ids) const {
    if (!bpe_) {
        throw std::runtime_error("CyphaLMModel: BPE tokenizer not configured");
    }
    return bpe_->decode(ids);
}

void CyphaLMModel::rpsm_embed_backprop(std::uint32_t token_id) {
    if (!rpsm_layer_) return;
    rpsm_embed_backprop_from_grad(token_id, rpsm_layer_->input_grad());
}

void CyphaLMModel::rpsm_embed_backprop_from_grad(std::uint32_t token_id,
                                                   const std::vector<double>& field_grad) {
    if (!embed_ || !rpsm_layer_) return;
    if (token_id >= embed_->vocab_size()) return;
    if (field_grad.empty()) return;

    // Correct chain rule: loss -> field_x_ (= proj_ssm_ * ctx) -> ctx (SSM layer-0 state
    // transition) -> e (embedding row). `field_grad` is d(loss)/d(field_x_) restricted to the
    // leading `rpsm_state_dim` entries RpsmSequenceLayer::train_step actually reads (see
    // encode_level0_features/inject_input_multilevel, both bounded by `state_dim`); any
    // trailing field_x_ entries never influence the loss, so their gradient is exactly zero,
    // not an approximation -- zero-pad rather than truncate/misalign as the old stub did.
    // §14: this same helper is now also called once per step, in a loop, at each RPSM BPTT
    // window flush -- `field_grad` there is `bptt_window_input_grads()[t]` (the deeper,
    // multi-step-aware gradient), rather than the single-step-local `input_grad()` the
    // immediate per-step caller (`rpsm_embed_backprop`) above passes.
    std::vector<double> grad_field(static_cast<std::size_t>(cfg_.field_dim), 0.0);
    for (std::size_t i = 0; i < field_grad.size() && i < grad_field.size(); ++i) {
        grad_field[i] = field_grad[i];
    }

    CellAISSM* active = active_ssm();
    const int ctx_dim = ssm_context_dim();
    if (!active || ctx_dim <= 0 ||
        proj_ssm_.size() != static_cast<std::size_t>(cfg_.field_dim) * static_cast<std::size_t>(ctx_dim)) {
        return;
    }
    // proj_ssm_ is [field_dim x ctx_dim] row-major (project_field: field_x_ = proj_ssm_ * ctx),
    // so d(loss)/d(ctx) = proj_ssm_^T . grad_field -- same transpose-multiply convention as
    // bptt_ssm_update below and char_lstm.cpp's `dx = Wx^T * dgates`.
    const auto grad_ctx = matvec_transpose(proj_ssm_, cfg_.field_dim, ctx_dim, grad_field);

    const int sd = active->d_state();
    if (sd <= 0 || static_cast<int>(grad_ctx.size()) < 2 * sd) return;

    // Only SSM layer 0 consumes the raw embedding `e` directly (layer 1+, if any, takes the
    // previous layer's ctx as input, not `e`) -- this mirrors the same layer-0-only scope
    // `bptt_ssm_update` already uses for this SSM (it also only reads grad_ctx[0:2*d_state]).
    // We take layer 0's *direct* contribution to the field vector via grad_ctx's first
    // 2*d_state entries; we deliberately do not additionally unroll grad through layer 1+'s
    // state transition back into layer 0's ctx, matching that existing precedent.
    std::vector<double> grad_h(static_cast<std::size_t>(sd));
    std::vector<double> grad_s(static_cast<std::size_t>(sd));
    if (cfg_.use_multiscale && !active->multiscale_alpha().empty()) {
        // ctx = [alpha*h + (1-alpha)*s ; s] per CellAISSM::step's multiscale branch.
        const double alpha = std::clamp(active->multiscale_alpha()[0], 0.0, 1.0);
        for (int i = 0; i < sd; ++i) {
            const double g_blend = grad_ctx[static_cast<std::size_t>(i)];
            const double g_s_direct = grad_ctx[static_cast<std::size_t>(sd + i)];
            grad_h[static_cast<std::size_t>(i)] = alpha * g_blend;
            grad_s[static_cast<std::size_t>(i)] = (1.0 - alpha) * g_blend + g_s_direct;
        }
    } else {
        // ctx = [h ; s].
        for (int i = 0; i < sd; ++i) {
            grad_h[static_cast<std::size_t>(i)] = grad_ctx[static_cast<std::size_t>(i)];
            grad_s[static_cast<std::size_t>(i)] = grad_ctx[static_cast<std::size_t>(sd + i)];
        }
    }

    const auto& w_fast = active->w_fast_layer0();
    const auto& w_slow = active->w_slow_layer0();
    const std::uint32_t de = embed_->dim();
    if (w_fast.size() != static_cast<std::size_t>(sd) * de ||
        w_slow.size() != static_cast<std::size_t>(sd) * de) {
        return;
    }
    const double lf = active->lambda_fast();
    const double ls = active->lambda_slow();

    // CellAISSM::step's non-spectral leaky-integrator update: h[i] = lf*h_prev[i] +
    // (1-lf)*(W_fast . e)[i], s[i] = ls*s_prev[i] + (1-ls)*(W_slow . e)[i], so
    // d(h[i])/d(e[j]) = (1-lf)*W_fast[i,j] and d(s[i])/d(e[j]) = (1-ls)*W_slow[i,j] --
    // grad_e = (1-lf) * W_fast^T . grad_h + (1-ls) * W_slow^T . grad_s.
    std::vector<double> grad_e(de, 0.0);
    for (int i = 0; i < sd; ++i) {
        const double gh = (1.0 - lf) * grad_h[static_cast<std::size_t>(i)];
        const double gs = (1.0 - ls) * grad_s[static_cast<std::size_t>(i)];
        const double* wf_row = w_fast.data() + static_cast<std::size_t>(i) * de;
        const double* ws_row = w_slow.data() + static_cast<std::size_t>(i) * de;
        for (std::uint32_t j = 0; j < de; ++j) {
            grad_e[j] += gh * wf_row[j] + gs * ws_row[j];
        }
    }

    const double lr = cfg_.rpsm_lr * 0.1;
    auto& table = embed_->table();
    double* row = table.data() + static_cast<std::size_t>(token_id) * de;
    for (std::uint32_t j = 0; j < de; ++j) {
        row[j] -= lr * grad_e[j];
    }
}

void CyphaLMModel::rpsm_bptt_embed_flush() {
    // §14: called immediately after an `RpsmSequenceLayer::train_step` call that just flushed a
    // BPTT window (`rpsm_layer_->bptt_window_flushed()`). Applies the deeper, multi-step-aware
    // embedding-gradient correction to every token that appeared in the just-flushed window,
    // on top of (not instead of) the immediate per-step update `rpsm_embed_backprop` already
    // applied for each of those steps as they happened -- these are two distinct gradient
    // signals (single-step-local vs. windowed-recursive), not a double count of the same one;
    // see `bptt_backward_and_apply()` in rpsm_sequence_layer.cpp.
    if (!rpsm_layer_) {
        rpsm_bptt_token_ids_.clear();
        return;
    }
    const auto& grads = rpsm_layer_->bptt_window_input_grads();
    const std::size_t n = std::min(grads.size(), rpsm_bptt_token_ids_.size());
    for (std::size_t t = 0; t < n; ++t) {
        rpsm_embed_backprop_from_grad(rpsm_bptt_token_ids_[t], grads[t]);
    }
    rpsm_bptt_token_ids_.clear();
}

TrainStepMetrics CyphaLMModel::train_step_rpsm(std::uint32_t token_id, std::uint32_t next_token_id,
                                                 cypha::intelligence::IntelligenceProfiler* profiler,
                                                 LmIntelligenceMonitor* monitor) {
    TrainStepMetrics m;
    if (!embed_ || !rpsm_layer_) {
        return train_step(token_id, next_token_id, profiler, monitor);
    }

    const auto e = embed_->embed_vec(token_id);
    record_embedding(e);
    last_e_ = e;
    auto ctx = ssm_step(e);
    apply_hebbian_hooks(ctx);
    last_ctx_ = ctx;
    field_x_ = project_field(ctx);

    rpsm_bptt_token_ids_.push_back(token_id);
    const auto rpsm_m = rpsm_layer_->train_step(field_x_.data(), static_cast<int>(field_x_.size()),
                                                static_cast<int>(next_token_id), cfg_.rpsm_lr);
    m.loss = rpsm_m.loss;
    last_train_loss_ = m.loss;

    rpsm_embed_backprop(token_id);
    if (rpsm_layer_->bptt_window_flushed()) {
        rpsm_bptt_embed_flush();
    }

    if (next_token_id < token_counts_.size()) {
        token_counts_[static_cast<std::size_t>(next_token_id)] += 1.0;
    }
    ++step_count_;
    return m;
}

void CyphaLMModel::train_sequence_rpsm(const std::vector<int>& ids, int n_steps, int epochs,
                                         cypha::intelligence::IntelligenceProfiler* profiler) {
    if (ids.size() < 2 || !rpsm_layer_) return;
    LmIntelligenceMonitor monitor;
    monitor.set_use_eigenvalue_d_eff(cfg_.use_eigenvalue_d_eff);
    const bool track_profiler = profiler != nullptr || uses_profile_guided_backprop(cfg_);
    cypha::intelligence::IntelligenceProfiler* flush_profiler = profiler;
    if (flush_profiler == nullptr && uses_profile_guided_backprop(cfg_) && train_profiler_) {
        flush_profiler = train_profiler_.get();
    }
    const int ep_count = std::max(1, epochs);
    const char* log_env = std::getenv("CYPHALM_TRAIN_LOG_EVERY");
    const int log_every = log_env ? std::max(0, std::atoi(log_env)) : 0;
    int step_i = 0;
    for (int ep = 0; ep < ep_count; ++ep) {
        reset_context();
        const int steps = std::min(n_steps, static_cast<int>(ids.size()) - 1);
        for (int i = 0; i < steps; ++i) {
            const auto m = train_step_rpsm(
                static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]),
                static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i + 1)]), profiler,
                track_profiler ? &monitor : nullptr);
            ++step_i;
            if (log_every > 0 && step_i % log_every == 0) {
                std::cerr << "[CyphaLM] rpsm train step " << step_i << " loss=" << m.loss
                          << std::endl;
            }
        }
        if (flush_profiler != nullptr) {
            monitor.flush_to_profiler(*flush_profiler);
            monitor.reset();
        }
    }
}

void CyphaLMModel::train_sequence(const std::vector<int>& ids, int n_steps, int epochs,
                                    cypha::intelligence::IntelligenceProfiler* profiler) {
    if (ids.size() < 2) return;
    if (cfg_.context_mode == ContextMode::Rpsm && rpsm_layer_) {
        train_sequence_rpsm(ids, n_steps, epochs, profiler);
        return;
    }
    if (cfg_.view_schedule != "same_order") {
        const std::size_t take =
            std::min(ids.size(), static_cast<std::size_t>(std::max(1, n_steps) + 1));
        train_sequence_views(std::vector<int>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(take)),
                             profiler);
        return;
    }
    LmIntelligenceMonitor monitor;
    monitor.set_use_eigenvalue_d_eff(cfg_.use_eigenvalue_d_eff);
    const bool track_profiler = profiler != nullptr || uses_profile_guided_backprop(cfg_) ||
                                cfg_.use_full_navigation_loss;
    cypha::intelligence::IntelligenceProfiler* flush_profiler = profiler;
    if (flush_profiler == nullptr && uses_profile_guided_backprop(cfg_) && train_profiler_) {
        flush_profiler = train_profiler_.get();
    }
    const int ep_count = std::max(1, epochs);
    const double base_gria_lr = cfg_.gria_lr;
    const bool overnight_progress = cypha::bench::bench_overnight_enabled();
    const int steps_per_epoch = std::min(n_steps, static_cast<int>(ids.size()) - 1);
    const int total_steps = steps_per_epoch * ep_count;
    int step_total = 0;

    std::vector<int> curriculum_order;
    if (cfg_.use_profile_curriculum && steps_per_epoch > 0) {
        const int prescan = std::min(512, steps_per_epoch);
        reset_context();
        std::vector<double> prescan_losses(static_cast<std::size_t>(prescan), 0.0);
        for (int i = 0; i < prescan; ++i) {
            const auto pred =
                predict_next(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]));
            const int nxt = ids[static_cast<std::size_t>(i + 1)];
            if (nxt >= 0 && static_cast<std::size_t>(nxt) < pred.log_probs.size()) {
                prescan_losses[static_cast<std::size_t>(i)] =
                    -pred.log_probs[static_cast<std::size_t>(nxt)];
            }
        }
        double max_loss = 0.0;
        for (double loss : prescan_losses) {
            max_loss = std::max(max_loss, loss);
        }
        curriculum_order = profile_curriculum_order(
            ids, steps_per_epoch,
            [&](int step_idx) -> double {
                if (step_idx >= prescan || max_loss <= 1e-12) {
                    return 0.0;
                }
                return prescan_losses[static_cast<std::size_t>(step_idx)] / max_loss;
            });
        reset_context();
    }

    auto train_one_step = [&](int step_idx) {
        reset_context();
        for (int j = 0; j < step_idx; ++j) {
            (void)predict_next(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(j)]));
        }
        train_step(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(step_idx)]),
                   static_cast<std::uint32_t>(ids[static_cast<std::size_t>(step_idx + 1)]), profiler,
                   track_profiler ? &monitor : nullptr);
    };

    for (int ep = 0; ep < ep_count; ++ep) {
        cfg_.gria_lr = base_gria_lr * std::pow(cfg_.gria_lr_decay, static_cast<double>(ep));
        reset_context();
        const int steps = steps_per_epoch;
        if (!curriculum_order.empty()) {
            for (int ord : curriculum_order) {
                if (ord < 0 || ord >= steps) {
                    continue;
                }
                train_one_step(ord);
                ++step_total;
                if (overnight_progress && step_total % 10000 == 0) {
                    std::cerr << "[cyphalm] overnight train progress: step " << step_total << "/"
                              << total_steps << " epoch " << (ep + 1) << "/" << ep_count << std::endl;
                }
            }
        } else {
            for (int i = 0; i < steps; ++i) {
                train_step(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]),
                           static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i + 1)]), profiler,
                           track_profiler ? &monitor : nullptr);
                ++step_total;
                if (overnight_progress && step_total % 10000 == 0) {
                    std::cerr << "[cyphalm] overnight train progress: step " << step_total << "/"
                              << total_steps << " epoch " << (ep + 1) << "/" << ep_count << std::endl;
                }
            }
        }
        if (flush_profiler != nullptr) {
            monitor.flush_to_profiler(*flush_profiler);
            monitor.reset();
        }
    }
    cfg_.gria_lr = base_gria_lr;
    if (lstm_) {
        lstm_->flush_bptt(cfg_.lstm_lr);
    }
    if (cfg_.use_sr_gates && lstm_) {
        fit_sr_gate_laws_on_lstm(*lstm_, cfg_.vocab_size, cfg_.seed);
    }
}

double CyphaLMModel::hybrid_gria_weight() const {
    return sigmoid(hybrid_blend_logit_);
}

AlphaSpectrumSnapshot CyphaLMModel::alpha_spectrum_snapshot() const {
    AlphaSpectrumSnapshot snap;
    if (gria_) snap.gria_projection_alpha = gria_->alpha;
    if (dif_) snap.expert_alpha = dif_->alpha_per_expert();
    if (!snap.expert_alpha.empty()) {
        double sum = 0.0;
        int near = 0;
        for (double a : snap.expert_alpha) {
            sum += a;
            if (std::abs(a - 0.5) < 0.1) ++near;
        }
        snap.mean_expert_alpha = sum / static_cast<double>(snap.expert_alpha.size());
        snap.fraction_experts_near_edge_of_chaos =
            static_cast<double>(near) / static_cast<double>(snap.expert_alpha.size());
    }
    std::vector<double> all = snap.gria_projection_alpha;
    all.insert(all.end(), snap.expert_alpha.begin(), snap.expert_alpha.end());
    if (!all.empty()) {
        double sum = 0.0;
        int near = 0;
        for (double a : all) {
            sum += a;
            if (std::abs(a - 0.5) < 0.1) ++near;
        }
        snap.mean_alpha = sum / static_cast<double>(all.size());
        snap.fraction_near_edge_of_chaos =
            static_cast<double>(near) / static_cast<double>(all.size());
    }
    return snap;
}

nlohmann::json CyphaLMModel::compression_profile() const {
    const auto snap = alpha_spectrum_snapshot();
    nlohmann::json gria_spec = nlohmann::json::object();
    if (gria_) {
        for (const auto& kv : gria_->alpha_spectrum()) {
            gria_spec[kv.first] = kv.second;
        }
    }
    const double total = last_dif_out_.epistemic_var + last_dif_out_.aleatoric_var + 1e-12;
    return {
        {"mean_epistemic_var", last_dif_out_.epistemic_var},
        {"mean_aleatoric_var", last_dif_out_.aleatoric_var},
        {"mean_alpha", snap.mean_alpha},
        {"mean_expert_alpha", snap.mean_expert_alpha},
        {"expert_alpha_spectrum", snap.expert_alpha},
        {"gria_alpha_spectrum", gria_spec},
        {"cfg_n_experts", cfg_.n_experts},
        {"max_experts", cfg_.max_experts},
        {"n_experts", dif_ ? dif_->expert_count() : 0},
        {"lossless_fraction", last_dif_out_.epistemic_var / total},
        {"lossy_fraction", 1.0 - last_dif_out_.epistemic_var / total},
        {"fraction_near_edge_of_chaos", snap.fraction_near_edge_of_chaos},
    };
}

double CyphaLMModel::eval_bpc(const std::vector<int>& ids, int n_eval,
                              cypha::intelligence::IntelligenceProfiler* profiler) {
    reset_context();
    const int n = std::min(n_eval, static_cast<int>(ids.size()) - 1);
    if (n <= 0) return std::numeric_limits<double>::quiet_NaN();
    const int vocab = static_cast<int>(cfg_.vocab_size);
    double bits = 0.0;
    int scored = 0;
    cypha::intelligence::EpistemicThreshold self_correct_threshold(0.5, 5.0);
    for (int i = 0; i < n; ++i) {
        const std::uint32_t tok = static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]);
        std::vector<double> embed;
        if (embed_) {
            try {
                embed = embed_vector(tok);
            } catch (const std::exception&) {
                embed.clear();
            }
        }
        auto pred = predict_next(tok);
        if (cfg_.use_self_correcting_loop) {
            pred = self_correct_if_needed(pred, self_correct_threshold);
        }
        const int nxt = ids[static_cast<std::size_t>(i + 1)];
        if (nxt < 0 || nxt >= vocab ||
            static_cast<std::size_t>(nxt) >= pred.log_probs.size()) {
            continue;
        }
        if (profiler != nullptr) {
            update_profiler_from_lm_token(*profiler, embed, pred.log_probs, pred.epistemic_var,
                                          pred.aleatoric_var);
        }
        bits += -pred.log_probs[static_cast<std::size_t>(nxt)] / kLog2;
        ++scored;
    }
    if (scored <= 0) return std::numeric_limits<double>::quiet_NaN();
    return bits / static_cast<double>(scored);
}

void CyphaLMModel::accumulate_intelligence_profile(const std::vector<int>& ids, int n_steps,
                                                   cypha::intelligence::IntelligenceProfiler& profiler) {
    reset_context();
    const int n = std::min(n_steps, static_cast<int>(ids.size()) - 1);
    if (n <= 0) {
        return;
    }
    const int vocab = static_cast<int>(cfg_.vocab_size);
    LmIntelligenceMonitor monitor;
    monitor.set_use_eigenvalue_d_eff(cfg_.use_eigenvalue_d_eff);
    cypha::intelligence::EpistemicThreshold self_correct_threshold(0.5, 5.0);
    for (int i = 0; i < n; ++i) {
        const std::uint32_t tok = static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]);
        std::vector<double> embed;
        if (embed_) {
            try {
                embed = embed_vector(tok);
            } catch (const std::exception&) {
                embed.clear();
            }
        }
        auto pred = predict_next(tok);
        if (cfg_.use_self_correcting_loop) {
            pred = self_correct_if_needed(pred, self_correct_threshold);
        }
        const int nxt = ids[static_cast<std::size_t>(i + 1)];
        if (nxt < 0 || nxt >= vocab ||
            static_cast<std::size_t>(nxt) >= pred.log_probs.size()) {
            continue;
        }
        monitor.observe_token(embed, field_x_, pred.log_probs, pred.epistemic_var, pred.aleatoric_var,
                              static_cast<std::int64_t>(nxt), vocab);
    }
    monitor.flush_to_profiler(profiler);
}

std::vector<double> CyphaLMModel::embed_vector(std::uint32_t token_id) const {
    if (!embed_) {
        throw std::runtime_error("CyphaLMModel: embed table unavailable");
    }
    return embed_->embed_vec(token_id);
}

double CyphaLMModel::ssm_projection_rms() const {
    if (proj_ssm_.empty()) return 0.0;
    double acc = 0.0;
    for (double v : proj_ssm_) acc += v * v;
    return std::sqrt(acc / static_cast<double>(proj_ssm_.size()));
}

nlohmann::json CyphaLMModel::ssm_diagnostic_report(const std::vector<int>& token_ids, int max_steps) {
    return diagnose_model_tokens(*this, token_ids, max_steps, "cyphalm");
}

}  // namespace cyphalm
}  // namespace cypha
