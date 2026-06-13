#pragma once

/// Unified CyphaLM native config — Tier 0 (speed), Tier 1 (long context),
/// Tier 2 (model class), Tier 4 (Hebbian). Mirrors ``cypha_lm.config.CyphaLMConfig``.

#include <cstdint>
#include <string>

namespace cypha::cyphalm {

enum class ContextMode {
    Full,
    GriaNgram,
    Hybrid,
    CharLstm,
    SsmGria,
    SsmGriaNoLstm,
    AblationNoDif,
    AblationNoSsm,
};

/// Bench / CLI mode aliases (mapped to ``ContextMode`` + tier flags).
enum class BenchMode {
    CharLstm,
    Ssm,
    Hybrid,
    SsmGria,
    ContextBank,
    Spectral,
};

struct CyphaLMConfig {
    int vocab_size = 128;
    int d_embed = 64;

    int d_state = 128;
    double tau_fast = 1.0;
    double tau_slow = 20.0;
    int ssm_layers = 2;
    bool use_spectral_pde = false;
    bool use_multiscale = true;
    bool use_sparse_hebbian = false;

    int n_experts = 0;
    int max_experts = 256;
    int field_dim = 160;
    double nig_kappa0 = 1.0;
    double nig_alpha0 = 2.0;
    double nig_beta0 = 1.0;

    double alpha_init = 0.5;
    bool alpha_learnable = true;
    int gria_rank = 32;

    int context_length = 256;
    ContextMode context_mode = ContextMode::Hybrid;
    int ngram_context = 2;
    int train_epochs = 1;
    std::string view_schedule = "same_order";
    int view_block_size = 512;
    int view_id_dim = 0;
    bool view_learnable = false;
    double view_lr = 0.005;
    int max_view_slots = 16;
    std::string ngram_fusion = "sum";
    bool ngram_position_weights = false;
    bool ngram_fuse_split = true;
    double gria_lr_decay = 0.5;
    int bptt_steps = 0;
    double laplace_smoothing = 1.0;
    bool online = true;
    double gria_lr = 0.05;
    double ssm_lr = 0.001;
    bool train_ssm = false;

    int lstm_hidden = 128;
    double lstm_lr = 0.05;
    double hybrid_blend_logit = 0.0;
    bool hybrid_blend_learnable = true;
    double hybrid_blend_lr = 0.01;

    bool use_context_bank = false;
    int context_bank_slots = 64;
    bool use_hierarchical_ssm = false;
    bool use_hebb_graph = false;
    bool use_hebbian_stack = false;
    /// Optional temporal SOM decay scaling (U6; off by default).
    bool use_temporal_som = false;
    /// Growing Neural Gas auxiliary prototypes (U1; off by default).
    bool use_gng = false;
    /// GRIA alpha live topology controller (U3; requires ``use_gng``).
    bool use_gria_controller = false;
    /// Discriminative feedback on encoder/BPTT grads (U4; off by default).
    bool use_discriminative_feedback = false;
    double ssm_hebb_lr = 1e-4;

    int compress_interval = 64;
    int max_memory_slots = 256;

    std::uint64_t seed = 42;

    /// Optional BPE tokenizer paths (inference encode/decode when both set).
    std::string bpe_merges_path;
    std::string bpe_vocab_path;
};

ContextMode parse_context_mode(const std::string& s);
std::string context_mode_name(ContextMode mode);
std::string context_mode_string(ContextMode mode);

BenchMode parse_bench_mode(const std::string& s);
void apply_bench_mode(BenchMode mode, CyphaLMConfig& cfg);
std::string bench_mode_name(BenchMode mode);

/// Load ``bench/config/profiles/cyphalm_<profile>_wikitext.json`` (or gutenberg for d04).
void apply_bench_profile(const std::string& profile, CyphaLMConfig& cfg);

}  // namespace cypha::cyphalm
