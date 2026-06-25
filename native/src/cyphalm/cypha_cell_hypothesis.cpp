#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"

#include <stdexcept>

namespace cypha::cyphalm {

namespace {

const std::vector<CellVariantSpec>& variant_table() {
    static const std::vector<CellVariantSpec> kVariants = {
        {"B0", "4-gram", 0, true, "ssm_gria", "SSM+GRIA with ngram_context embed fusion and count prior"},
        {"B1", "char-LSTM", 0, true, "char_lstm", "locked baseline"},
        {"B2", "hybrid_gria_lstm", 0, true, "hybrid", "locked baseline"},
        {"H01", "alpha-gate cell", 1, true, "hybrid", "GRIA mean α scales LSTM forget gate in hybrid forward"},
        {"H02", "EML activation cell", 1, true, "char_lstm", "Sheffer eml() replaces sigmoid/tanh"},
        {"H03", "CausalField cell", 1, true, "ssm", "SSM/SGEMV recurrence primitive"},
        {"H04", "Pure CyphaDIF LM", 1, true, "ssm_gria", "DIF + GRIA without LSTM"},
        {"H05", "alpha-fitness aux loss", 1, true, "hybrid", "hybrid + profile-guided loss in train_step backprop"},
        {"H06", "NIG-state cell", 2, true, "hybrid", "NigStateCell Bayesian update on field path"},
        {"H07", "Differential gate", 2, true, "hybrid", "SSM ctx = theta0 + delta-h blend (last_ctx + dh)"},
        {"H08", "TieredContext cell", 2, true, "context_bank", "short/mid/long tiered context bank attention"},
        {"H09", "GRIA-gated mixture", 2, true, "hybrid", "α trajectory shifts ordered GRIA vs chaotic LSTM blend"},
        {"H10", "NMP regularised", 2, true, "hybrid", "spec_alpha -> 0.485"},
        {"H11", "Reversible cell", 2, true, "ssm", "RevNet additive coupling + backward reconstruct stub on SSM ctx"},
        {"H12", "MDL forget", 2, true, "hybrid", "L2 norm projection on field hidden state"},
        {"H13", "Priority replay recurrence", 2, true, "hybrid", "compressive memory priority replay slots"},
        {"H14", "OOD-branching cell", 2, true, "hybrid", "hybrid blend shifts to LSTM when DIF epistemic high"},
        {"H15", "AXIOM-evolved cell", 3, true, "hybrid", "seed-evolved eml/sigmoid/tanh gate grammar in LSTM"},
        {"H16", "SR on trained LSTM gates", 3, true, "hybrid",
         "fit linear gate laws on LSTM trace; optional SR gate override"},
        {"H17", "Sheffer-only cell", 3, true, "char_lstm", "extreme H02 — EML-only activations"},
        {"H18", "CA state cell", 3, true, "ssm", "elementary CA rule 110 one-step on binarized SSM h"},
        {"H19", "Izaac-seeded init", 3, true, "hybrid", "seed-offset hybrid init (blend logit + GRIA α prior)"},
        {"H20", "Spectral state cell", 3, true, "spectral", "FFT-domain SSM recurrence"},
        {"H21", "Free Energy cell", 3, true, "ssm_gria", "variational β·epistemic_var penalty in train_step"},
        {"H22", "Algebraic fingerprint cell", 3, true, "hybrid", "Izaac algebraic fingerprint tag in GRIA input"},
    };
    return kVariants;
}

void apply_bench_mode_string(const std::string& mode, CyphaLMConfig& cfg) {
    apply_bench_mode(parse_bench_mode(mode), cfg);
}

}  // namespace

const std::vector<CellVariantSpec>& all_cell_variants() { return variant_table(); }

const CellVariantSpec* find_cell_variant(const std::string& id) {
    for (const auto& v : variant_table()) {
        if (v.id == id) {
            return &v;
        }
    }
    return nullptr;
}

void apply_cell_variant(const std::string& id, CyphaLMConfig& cfg) {
    const CellVariantSpec* spec = find_cell_variant(id);
    if (spec == nullptr) {
        throw std::runtime_error("unknown cell variant: " + id);
    }
    apply_bench_mode_string(spec->bench_mode, cfg);
    cfg.cell_variant = id;

    cfg.use_eml_activation = false;
    cfg.use_differential_gate = false;
    cfg.use_nig_state_cell = false;
    cfg.use_tiered_context = false;
    cfg.use_ood_branching = false;
    cfg.profile_guided_loss = false;
    cfg.use_gria_gated_mixture = false;
    cfg.use_reversible_cell = false;
    cfg.use_mdl_forget = false;
    cfg.use_priority_replay = false;
    cfg.use_axiom_activation = false;
    cfg.use_sr_gates = false;
    cfg.use_ca_state_cell = false;
    cfg.use_free_energy_loss = false;
    cfg.use_algebraic_fingerprint = false;
    cfg.use_kernel_llr = false;
    cfg.kernel_blend = 0.25;

    if (id == "B0") {
        cfg.ngram_context = 4;
        cfg.context_length = 128;
    } else if (id == "H01") {
        cfg.alpha_init = 0.5;
        cfg.alpha_learnable = true;
        cfg.use_alpha_forget_gate = true;
    } else if (id == "H02" || id == "H17") {
        cfg.use_eml_activation = true;
    } else if (id == "H04") {
        cfg.use_kernel_llr = true;
        cfg.kernel_m = 64;
        cfg.kernel_lr_scale = 1.0;
    } else if (id == "H05") {
        cfg.alpha_learnable = true;
        cfg.profile_guided_loss = true;
    } else if (id == "H06") {
        cfg.n_experts = 4;
        cfg.online = true;
        cfg.use_nig_state_cell = true;
    } else if (id == "H07") {
        cfg.use_differential_gate = true;
        cfg.bptt_steps = 4;
        cfg.train_ssm = true;
    } else if (id == "H08") {
        cfg.use_context_bank = true;
        cfg.use_tiered_context = true;
        cfg.context_bank_slots = 32;
    } else if (id == "H09") {
        cfg.use_gria_gated_mixture = true;
        cfg.hybrid_blend_logit = 0.5;
        cfg.hybrid_blend_learnable = true;
        cfg.alpha_learnable = true;
    } else if (id == "H10") {
        cfg.alpha_init = 0.485;
        cfg.alpha_learnable = true;
    } else if (id == "H11") {
        cfg.use_reversible_cell = true;
        cfg.use_multiscale = false;
        cfg.ssm_layers = 1;
    } else if (id == "H12") {
        cfg.use_mdl_forget = true;
        cfg.compress_interval = 32;
        cfg.max_memory_slots = 128;
    } else if (id == "H13") {
        cfg.use_priority_replay = true;
        cfg.max_memory_slots = 256;
        cfg.compress_interval = 16;
    } else if (id == "H14") {
        cfg.n_experts = 8;
        cfg.online = true;
        cfg.use_ood_branching = true;
    } else if (id == "H15") {
        cfg.use_axiom_activation = true;
        cfg.alpha_learnable = true;
    } else if (id == "H16") {
        cfg.use_sr_gates = true;
    } else if (id == "H18") {
        cfg.use_ca_state_cell = true;
        cfg.use_multiscale = false;
        cfg.ssm_layers = 1;
    } else if (id == "H19") {
        cfg.seed += 991;
        cfg.hybrid_blend_logit = 0.35;
        cfg.alpha_init = 0.42;
    } else if (id == "H21") {
        cfg.use_free_energy_loss = true;
        cfg.free_energy_beta = 0.08;
    } else if (id == "H22") {
        cfg.use_algebraic_fingerprint = true;
        cfg.alpha_learnable = true;
    }
}

}  // namespace cypha::cyphalm
