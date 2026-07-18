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
    Rpsm,
    /// Single-context PGM spine: SSM→field→PGM h → Wy logits (U06 tournament winner).
    PgmLogits,
};

/// Bench / CLI mode aliases (mapped to ``ContextMode`` + tier flags).
enum class BenchMode {
    CharLstm,
    Ssm,
    Hybrid,
    SsmGria,
    ContextBank,
    Spectral,
    Rpsm,
    PgmLogits,
};

/// Unified-context tournament (U01–U10): single context carrier (default off = fragmented D17).
enum class UnifiedContextSource {
    None = 0,
    Field,
    Lstm,
    Pgm,
    ContextBank,
    Memory,
    UnifiedBuffer,
};

/// Single readout head when ``use_unified_context`` is set.
enum class UnifiedReadout {
    None = 0,
    Lstm,
    Gria,
    PgmWy,
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
    /// Soft multi-expert NIG updates (vs winner-take-all). Env: CYPHA_LM_SOFT_EXPERT_UPDATES=1.
    bool use_soft_expert_updates = false;
    /// Add routing-entropy floor loss when experts > 1. Env: CYPHA_LM_ROUTING_ENTROPY_FLOOR=1.
    bool use_routing_entropy_floor = false;
    double routing_entropy_lambda = 0.05;
    double routing_entropy_floor_frac = 0.5;
    int field_dim = 160;
    double nig_kappa0 = 1.0;
    double nig_alpha0 = 2.0;
    double nig_beta0 = 1.0;

    double alpha_init = 0.5;
    bool alpha_learnable = true;
    int gria_rank = 32;

    int context_length = 256;
    /// Bare ``CyphaLMConfig`` defaults Hybrid for benches; product/Cypha entry points apply U06 via
    /// ``apply_pgm_logits_recipe`` / ``apply_cell_variant("U06")``.
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
    /// B4: low-rank bilinear term ``U @ ((V_f @ field_x) ⊙ (V_e @ embeds))`` added to sum fusion.
    bool ngram_bilinear_fusion = false;
    bool ngram_fuse_split = false;
    double gria_lr_decay = 0.5;
    int bptt_steps = 0;
    double laplace_smoothing = 1.0;
    bool online = true;
    double gria_lr = 0.05;
    double ssm_lr = 0.001;
    bool train_ssm = false;

    int lstm_hidden = 128;
    double lstm_lr = 0.05;
    /// Truncated BPTT window for char-LSTM (1 = historic BPTT-1). Opt-in; default preserves pin.
    int lstm_bptt_steps = 1;
    /// ``sgd`` (default) or ``adam``. Env: ``CYPHA_LSTM_OPTIM``.
    std::string lstm_optim = "sgd";
    /// Global L2 grad clip; 0 = off. Env: ``CYPHA_LSTM_GRAD_CLIP``.
    double lstm_grad_clip = 0.0;
    /// ``default`` N(0,0.02) or ``classic`` (orthogonal Wh, forget bias +1). Env: ``CYPHA_LSTM_INIT``.
    std::string lstm_init = "default";
    /// AdamW decoupled weight decay; 0 = off (D17 pin). Env: ``CYPHA_LSTM_WEIGHT_DECAY``.
    double lstm_weight_decay = 0.0;
    /// Linear LR warmup steps (0 = no warmup). Env: ``CYPHA_LSTM_LR_WARMUP``.
    int lstm_lr_warmup_steps = 0;
    /// Cosine decay steps after warmup (0 = constant lr / no schedule). Env: ``CYPHA_LSTM_LR_COSINE``.
    int lstm_lr_cosine_steps = 0;
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

    /// Cell hypothesis testbench id (e.g. ``U06``); empty = bare struct defaults (Hybrid / D17).
    std::string cell_variant;
    /// H02/H17: Sheffer ``eml()`` activations in char-LSTM gates.
    bool use_eml_activation = false;
    /// H07: differential gate blends prior SSM context with delta-h.
    bool use_differential_gate = false;
    /// H06: NIG sufficient statistics as recurrent cell state on field path.
    bool use_nig_state_cell = false;
    /// H08: tiered short/mid/long context bank attention.
    bool use_tiered_context = false;
    /// H14: branch hybrid routing when DIF epistemic variance is high (OOD).
    bool use_ood_branching = false;
    /// H09: GRIA α trajectory modulates ordered vs chaotic hybrid blend.
    bool use_gria_gated_mixture = false;
    /// H01: scale char-LSTM forget gate by mean GRIA α.
    bool use_alpha_forget_gate = false;
    /// Paper IV: scale forget gate by τ (monitor) or r_eu (DIF) — ``0.5 + 0.5·signal``.
    bool use_tau_forget_gate = false;
    /// H11: RevNet-style reversible additive coupling on SSM context.
    bool use_reversible_cell = false;
    /// H12: MDL norm projection on recurrent hidden state.
    bool use_mdl_forget = false;
    /// H13: priority-weighted replay slots in compressive memory.
    bool use_priority_replay = false;
    /// H15: seed-evolved eml/sigmoid/tanh gate grammar in char-LSTM.
    bool use_axiom_activation = false;
    /// H16: symbolic-regression gate pre-activation laws fitted on LSTM trace.
    bool use_sr_gates = false;
    /// H18: elementary CA rule 110 on binarized SSM hidden state.
    bool use_ca_state_cell = false;
    /// H21: variational free-energy penalty on epistemic variance in train_step.
    bool use_free_energy_loss = false;
    /// H22: algebraic fingerprint tag mixed into GRIA input.
    bool use_algebraic_fingerprint = false;
    /// H23: Plastic Graph Machine (PGM) hierarchical sparse slot-graph cell.
    bool use_pgm_cell = false;
    /// U01–U10: enforce one context carrier + one readout (off = legacy dual-head D17).
    bool use_unified_context = false;
    UnifiedContextSource unified_context_source = UnifiedContextSource::None;
    UnifiedReadout unified_readout = UnifiedReadout::None;
    /// PGM branching factor b (N ≈ n_sub^levels); hierarchical log-N address.
    int pgm_n_sub = 8;
    /// PGM hierarchy depth L.
    int pgm_levels = 3;
    /// PGM T1 chunk length (consolidate adjacent bindings at boundary).
    int pgm_chunk_len = 16;
    int pgm_topk = 4;
    int pgm_beam = 2;
    int pgm_rehash_t = 16;
    int pgm_hops = 2;

    double mdl_forget_max_norm = 4.0;
    double free_energy_beta = 0.05;

    /// Option B RPSM sequence layer (level-0 CyphaDIF LLR scaffold).
    bool use_rpsm_layer = false;
    int rpsm_n_levels = 4;
    int rpsm_state_dim = 128;
    int rpsm_feat_dim = 64;
    double rpsm_lr = 0.01;
    /// Working-memory ring size / write gate (defaults match ``RpsmSequenceConfig``).
    int rpsm_n_memory_slots = 32;
    double rpsm_beta_memory = 0.1;
    double rpsm_surprise_threshold = 0.05;
    double rpsm_hierarchy_loss_weight = 0.1;
    /// Research BPTT window (1 = local grads; >1 measured negative @ 5k — keep profile default 1).
    int rpsm_bptt_window = 1;

    /// Paper IV: add profile-guided regularizers to per-step train loss.
    bool profile_guided_loss = false;
    /// When true with ``profile_guided_loss``, use all seven statistic lambdas (Paper II navigation loss).
    bool use_full_navigation_loss = false;
    /// Hardest-first block reordering by epistemic uncertainty before training.
    bool use_profile_curriculum = false;
    /// Ramp navigation-loss weight over this many train steps (0 = immediate full weight).
    int navigation_loss_warmup_steps = 200;
    /// Scale profile-guided lambdas from live κ (``clamp(1 - κ/target, 0.1, 1.0)``).
    bool use_adaptive_navigation_lambdas = false;
    /// Target κ for adaptive navigation lambda scaling (Paper III criticality).
    double kappa_lambda_target = 0.89;
    /// EMA κ trajectory modulates adaptive λ (Phase 31; requires ``use_adaptive_navigation_lambdas``).
    bool use_kappa_trajectory_lambdas = false;
    /// EMA window for κ trajectory λ schedule (steps).
    int kappa_trajectory_window = 16;
    /// Per-stat deviation weighting on navigation λ (Phase 32).
    bool use_per_stat_deviation_lambdas = false;
    double per_stat_deviation_span = 0.5;
    /// Weaken navigation λ when κ exceeds ``kappa_lambda_target`` (Phase 34).
    bool use_kappa_ceiling_lambdas = false;
    /// Paper IV: direct D_eff nudge on LSTM hidden state during backprop.
    bool use_lstm_d_eff_hidden_nudge = false;
    /// Covariance eigenvalue participation ratio for D_eff (Phase 35 Paper IV).
    bool use_eigenvalue_d_eff = false;
    /// κ ceiling excess multiplier (Phase 35 joint κ–BPC tuning).
    double kappa_ceiling_strength = 2.5;
    /// Minimum navigation λ scale under κ ceiling (Phase 35).
    double kappa_ceiling_min_scale = 0.35;
    /// Paper IV: scale forget gate by live r_eu (Phase 35 combined τ/r_eu signal).
    bool use_reu_forget_gate = false;
    /// Blend factor for r_eu forget gate (0=off, 1=full multiply; Phase 36 default 0.25).
    double reu_forget_gate_blend = 0.25;
    /// Paper IV §2.3/§4.2 epistemic feedback loop (SelfCorrectingCypha wrapper): during
    /// `eval_bpc`/`accumulate_intelligence_profile`, re-run the hybrid GRIA/LSTM blend at a
    /// wider deliberation setting (up to 3 passes) when live r_eu exceeds a learned
    /// `EpistemicThreshold`, keeping whichever pass has higher confidence. Hybrid-mode only;
    /// no-op otherwise. Opt-in and default-off: does not change the locked D17 BPC/kappa
    /// baseline unless explicitly requested (2026-07-11 follow-up to
    /// docs/reports/HIDDEN_DIM_SCALE_PLAN.md §3's self-correcting-wrapper gap).
    bool use_self_correcting_loop = false;
    /// Damp κ trajectory λ boost when EMA κ exceeds target (Phase 36).
    bool use_kappa_trajectory_ceiling = false;
    /// Direct κ-excess backprop nudge (Phase 37; separate from ceiling λ).
    bool use_kappa_excess_grad_nudge = false;
    /// Scale on κ-excess grad nudge (Phase 37 joint tuning).
    double kappa_excess_grad_scale = 0.35;
    /// κ margin above target before excess grad activates (Phase 37).
    double kappa_excess_grad_margin = 0.02;
    /// Damp kernel LLR blend when κ exceeds target (Phase 38).
    bool use_kappa_kernel_blend_scale = false;
    /// Minimum kernel blend under κ scaling (Phase 38).
    double kappa_kernel_blend_floor = 0.08;
    /// Damp navigation warmup ramp when κ exceeds target (Phase 40).
    bool use_kappa_navigation_warmup_scale = false;
    double kappa_navigation_warmup_strength = 0.35;
    double kappa_navigation_warmup_floor = 0.65;

    /// Elastic weight consolidation on char-LSTM ``Wx``/``Wh`` (0 = off).
    double ewc_lambda = 0.0;

    /// Blend DIF expert LLR softmax with Nyström kernel LLR (H04 / Phase 31).
    bool use_kernel_llr = false;
    double kernel_blend = 0.25;
    int kernel_m = 256;
    double kernel_gamma_scale = 1.0;
    double kernel_lr_scale = 1.0;
};

ContextMode parse_context_mode(const std::string& s);
std::string context_mode_name(ContextMode mode);
std::string context_mode_string(ContextMode mode);

BenchMode parse_bench_mode(const std::string& s);
void apply_bench_mode(BenchMode mode, CyphaLMConfig& cfg);
std::string bench_mode_name(BenchMode mode);

/// Enable the integrated PGM→logits recipe (ContextMode::PgmLogits + H23-ish PGM knobs).
void apply_pgm_logits_recipe(CyphaLMConfig& cfg);

std::string unified_context_source_name(UnifiedContextSource s);
std::string unified_readout_name(UnifiedReadout r);
UnifiedContextSource parse_unified_context_source(const std::string& s);
UnifiedReadout parse_unified_readout(const std::string& s);

/// Load ``bench/config/profiles/cyphalm_<profile>_wikitext.json`` (or gutenberg for d04).
void apply_bench_profile(const std::string& profile, CyphaLMConfig& cfg);

/// Overlay Quality Wave-1/2 LSTM recipe env vars (``CYPHA_LSTM_*``). Safe no-op when unset.
void apply_lstm_recipe_env(CyphaLMConfig& cfg);

/// LSTM LR at training step ``step`` (0-based). When warmup and cosine are both 0, returns
/// ``cfg.lstm_lr`` unchanged. Otherwise: linear warmup 0→``lstm_lr`` over
/// ``lstm_lr_warmup_steps``, then cosine decay to ``0.1 * lstm_lr`` over
/// ``lstm_lr_cosine_steps``, then hold at the floor.
double lstm_lr_at_step(const CyphaLMConfig& cfg, int step);

}  // namespace cypha::cyphalm
