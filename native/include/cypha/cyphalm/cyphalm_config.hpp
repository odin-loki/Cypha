#pragma once

/// Unified CyphaLM native config — Tier 0 (speed), Tier 1 (long context),
/// Tier 2 (model class), Tier 4 (Hebbian). Mirrors ``cypha_lm.config.CyphaLMConfig``.

#include <cstdint>
#include <string>
#include <vector>

namespace cypha::cyphalm {

enum class ContextMode {
    /// hp integer-exact context mixer (odin-loki/CompressionAlgorithm). gate24: v78 + HP_SLOT_MAX=24.
    Hp,
    /// Alias kept for CLI/profile compatibility — maps to ``Hp``.
    Hybrid,
    Full,
    GriaNgram,
    CharLstm,
    SsmGria,
    SsmGriaNoLstm,
    AblationNoDif,
    AblationNoSsm,
    /// Legacy PGM spine (research only; not the production LLM path).
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
    /// Production LLM path: hp context mixer (see ``hp_table_bits``).
    ContextMode context_mode = ContextMode::Hp;
    int ngram_context = 2;
    /// B0: add online n-gram count Laplace log-prior onto GRIA logits. Off by default so
    /// ordinary hybrid (ngram_context>0 for embed fusion only) keeps pre-685dbf2 blend dynamics.
    bool use_ngram_count_prior = false;
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
    /// Stacked residual char-LSTM depth. Production recipe sets **2** (2.664 BPC @300k lock).
    /// Override via profile JSON / ``--lstm-layers`` / ``CYPHA_LSTM_LAYERS`` (1 = historic L1).
    int lstm_layers = 2;
    double lstm_lr = 0.05;
    /// Truncated BPTT window. Production recipe sets **8** via ``apply_wave2_bptt_recipe``.
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
    /// Residual softmax attention from top LSTM ``h`` over memory keys before ``Wy``.
    /// Keys: ContextBank embeds when ``use_context_bank``, else compressive-memory slot means.
    /// KILL @40k (raw/gated both worse than L2); default OFF. CLI: ``--lstm-memory-attn``.
    bool use_lstm_memory_attn = false;
    /// Peak residual scale ``h += scale * Attn`` (ramped by key count / ``2*min_slots``).
    double lstm_memory_attn_scale = 0.10;
    /// Do not apply attn until this many keys are available (cold-slot dilution guard).
    int lstm_memory_attn_min_slots = 16;
    /// Ring of ``(h, next_token)`` for codec-path hidden kNN retrieval (0 = off).
    int hidden_knn_store = 2048;
    int hidden_knn_k = 16;

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

    /// Blend DIF expert LLR softmax with Nyström kernel LLR (H04 / Phase 31). Legacy; hp path ignores.
    bool use_kernel_llr = false;
    double kernel_blend = 0.25;
    int kernel_m = 256;
    double kernel_gamma_scale = 1.0;
    double kernel_lr_scale = 1.0;

    /// hp table bits per model (``--mem`` in hp CLI). Default 22 ≈ 4 MiB tables.
    int hp_table_bits = 22;
    /// Requested hp slot cap (compile-time ``HP_SLOT_MAX`` is authoritative; fixed at 24).
    int hp_slot_max = 24;
    /// hp mixer learning rate (integer, default 2).
    int hp_mixer_lr = 2;
    /// Enable hp GRIA alpha gating.
    bool hp_gria = true;

    /// Lossy RAM: when >0, overrides ``hp_table_bits`` (gate24 compile flags unchanged).
    /// Env: ``CYPHA_HP_LOSSY_MEM``. Typical tiers: 20 (−4× table RAM), 18 (−16×).
    int hp_lossy_mem = 0;
    /// Serve hint: legacy flag when twin scratch existed; now no-op (single-predictor default).
    /// Env: ``CYPHA_HP_SERVE_COMPACT=1``.
    bool hp_serve_compact = false;
    /// Lossy quality: prune hash slots with total state count below this (0=off).
    /// Env: ``CYPHA_HP_PRUNE_COLD_MIN_N``. Applied after warmup via ``prune_cold_slots()``.
    int hp_prune_cold_min_n = 0;

    /// Lossy mixer knobs; each maps 1:1 onto ``hp::Config`` and 0 keeps gate24 exactly.
    /// Set together by ``apply_hp_lossy_tier`` (env ``CYPHA_HP_LOSSY_TIER``). They change
    /// predictions, so they are saved with the model and must match the ``.hpbin`` tables.
    /// Measurements: docs/reports/CYPHALM_LOSSY_MIXER_REPORT.md.
    std::uint64_t hp_cm_drop = 0;    ///< bit i drops context model i (``hp::Predictor::CmId``)
    int hp_cm_bits_cap = 0;          ///< cap every context-model table at this many bits
    std::uint32_t hp_gate_drop = 0;  ///< bit j drops mixer weight set j (``hp::Predictor::Gate``)
    int hp_mixer_skip = 0;           ///< skip mixer update when |err| < this (gate24 = 32)
    /// Upstream mixer gains (hp::Config::lr1_scale / mixer_scale / mixer_skip_l1);
    /// defaults reproduce gate24. Env CYPHA_HP_LR1_SCALE, CYPHA_HP_MIXER_SCALE,
    /// CYPHA_HP_MIXER_SKIP_L1 (and CYPHA_HP_MIXER_SKIP after the tier).
    int hp_lr1_scale = 100;
    int hp_mixer_scale = 0;
    int hp_mixer_skip_l1 = 0;
    /// Optional upstream context models (hp::Config::extra_cms): bit k adds
    /// hp::Predictor::ExtraCm k (0 wikibold, 1 sentpos, 2 cappara, 3 refgroup,
    /// 4 statetrans, 5 cross o2 x sentmem, 6 cross word x brk; 127 = all).
    /// 0 reproduces gate24 bit for bit. Env CYPHA_HP_EXTRA_CMS.
    std::uint32_t hp_extra_cms = 0;
    int hp_match_bits_cap = 0;       ///< cap the 13 byte-match hash tables at this many bits
    int hp_pool_slots = 0;           ///< keep this many discovered-context slots (gate24 = 12)
    int hp_pool_bits_cap = 0;        ///< cap discovered-context tables at this many bits
    int hp_hebb_bits_cap = 0;        ///< cap the Hebbian word-association tables at this many bits
    std::uint32_t hp_match_drop = 0; ///< bit k: drop byte-match model k (hp::Config::match_drop)
    std::string hp_lossy_tier;       ///< name of the applied tier ("" = gate24), informational
    /// Serve-time frozen scoring (``HpSequenceBackend::set_frozen_scoring``): full
    /// next-byte distributions 2.1-2.5x faster for +0.003-0.008 bits/byte held-out
    /// (enwik8 8 MiB pretrain; CYPHALM_LM_QUALITY_REPORT.md). Training / eval_bpc
    /// are unaffected. Env ``CYPHA_HP_FROZEN_SCORING=0`` restores exact hp scoring.
    bool hp_frozen_scoring = true;
    /// Serve-time mixer learning rate as a fraction of the trained rate, applied
    /// while generating (``CyphaLMModel::set_serve_mode``). Adapting half as fast
    /// to a prompt measured -0.005 to -0.007 bits/byte held-out.
    /// Env ``CYPHA_HP_SERVE_MIXER_LR_SCALE``.
    double hp_serve_mixer_lr_scale = 0.5;
    /// Ensembles (``CyphaLMModel::add_ensemble_member``): mixing weights adapt
    /// online at this rate while reading with learning on. Finds the right
    /// split for unequal members (95 MB + 8 MiB: 0.65/0.35, better than any
    /// fixed weight); equal-size shards stay near equal. 0 = fixed weights.
    double hp_ensemble_learning_rate = 0.01;
    /// Serve-time bit-tree pruning (``HpSequenceBackend::set_tree_prune``):
    /// subtrees under this probability are not expanded. 1e-4: 5.5x faster
    /// next-byte distributions, held-out NLL within ±0.0005. 0 = exact.
    /// Env ``CYPHA_HP_TREE_PRUNE``.
    double hp_tree_prune = 1e-4;
};

/// Compile-time ``HP_SLOT_MAX`` baked into this binary (24, gate24).
int hp_compile_slot_max();

/// Clamp ``hp_table_bits`` to ``min(hp_slot_max, hp_compile_slot_max())``.
void normalize_hp_table_bits(CyphaLMConfig& cfg);

ContextMode parse_context_mode(const std::string& s);
std::string context_mode_name(ContextMode mode);
std::string context_mode_string(ContextMode mode);

BenchMode parse_bench_mode(const std::string& s);
void apply_bench_mode(BenchMode mode, CyphaLMConfig& cfg);
std::string bench_mode_name(BenchMode mode);

/// Enable the integrated PGM→logits recipe (ContextMode::PgmLogits + H23-ish PGM knobs).
void apply_pgm_logits_recipe(CyphaLMConfig& cfg);

/// Production CyphaLM default: gate24 hp (v78 flags + table_bits=22, slot_max=24).
void apply_hp_production_recipe(CyphaLMConfig& cfg);

/// Lossy gate24 tier: same v78 compile profile, smaller ``hp_table_bits`` (RAM lever).
/// ``mem_bits`` clamped to [16, 24]. Does not change ``HP_SLOT_MAX``.
void apply_hp_lossy_recipe(CyphaLMConfig& cfg, int mem_bits);

/// Overlay lossy env vars (``CYPHA_HP_LOSSY_MEM``, ``CYPHA_HP_SERVE_COMPACT``,
/// ``CYPHA_HP_PRUNE_COLD_MIN_N``). Safe no-op when unset.
void apply_hp_lossy_env(CyphaLMConfig& cfg);

/// Lossy mixer tier by name. "" / "gate24" is exact. Measured on enwik8 8 MiB (bpc, peak RSS):
/// lean 1.6099 / 1.08 GB (better than gate24's 1.6117 / 1.54 GB), balanced 1.6125 / 0.81 GB,
/// compact 1.6174 / 0.67 GB, small 1.6298 / 0.40 GB, tiny 1.6523 / 0.25 GB.
/// Throws on an unknown name. Keeps ``hp_lossy_mem``.
void apply_hp_lossy_tier(CyphaLMConfig& cfg, const std::string& tier);
std::vector<std::string> hp_lossy_tier_names();

/// Effective table bits after lossy override.
int hp_effective_table_bits(const CyphaLMConfig& cfg);

/// Back-compat alias for ``apply_hp_production_recipe``.
void apply_hybrid_production_recipe(CyphaLMConfig& cfg);

/// Quality Wave-2 LSTM recipe (opt-in; does not flip D17 defaults).
/// Adam + classic init + grad clip + truncated BPTT at ``lstm_lr=0.001``.
/// When ``train_steps > 0`` and ``with_schedule``, also sets linear warmup (~2.5% of steps,
/// capped) then cosine decay over the remaining budget (floor = 0.1×lr).
struct Wave2BpttOptions {
    int bptt_steps = 8;
    double lstm_lr = 0.001;
    double grad_clip = 1.0;
    bool with_schedule = false;
    int train_steps = 0;  // approx token updates (n_train * epochs); used only if with_schedule
};

void apply_wave2_bptt_recipe(CyphaLMConfig& cfg, const Wave2BpttOptions& opt = {});

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
