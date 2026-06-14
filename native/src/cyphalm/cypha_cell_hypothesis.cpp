#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"

#include <stdexcept>

namespace cypha::cyphalm {

namespace {

const std::vector<CellVariantSpec>& variant_table() {
    static const std::vector<CellVariantSpec> kVariants = {
        {"B0", "4-gram", 0, true, "ssm_gria", "n-gram proxy via GRIA ngram path"},
        {"B1", "char-LSTM", 0, true, "char_lstm", "locked baseline"},
        {"B2", "hybrid_gria_lstm", 0, true, "hybrid", "locked baseline"},
        {"H01", "alpha-gate cell", 1, true, "hybrid", "GRIA alpha as forget gate proxy"},
        {"H02", "EML activation cell", 1, true, "char_lstm", "Sheffer eml() replaces sigmoid/tanh"},
        {"H03", "CausalField cell", 1, true, "ssm", "SSM/SGEMV recurrence primitive"},
        {"H04", "Pure CyphaDIF LM", 1, true, "ssm_gria", "DIF + GRIA without LSTM"},
        {"H05", "alpha-fitness aux loss", 1, true, "hybrid", "hybrid + profile-guided loss tag"},
        {"H06", "NIG-state cell", 2, true, "hybrid", "DIF NIG expert state one-step update"},
        {"H07", "Differential gate", 2, true, "hybrid", "theta0 + delta-h recurrent gate proxy"},
        {"H08", "TieredContext cell", 2, true, "context_bank", "short/mid/long context bank"},
        {"H09", "GRIA-gated mixture", 2, true, "hybrid", "ordered vs chaotic GRIA blend"},
        {"H10", "NMP regularised", 2, true, "hybrid", "spec_alpha -> 0.485"},
        {"H11", "Reversible cell", 2, true, "ssm", "RevNet-style SSM proxy (single-scale)"},
        {"H12", "MDL forget", 2, true, "hybrid", "norm projection via compressive memory"},
        {"H13", "Priority replay recurrence", 2, true, "hybrid", "compressive memory + replay slots"},
        {"H14", "OOD-branching cell", 2, true, "hybrid", "NIG expert branching on hybrid path"},
        {"H15", "AXIOM-evolved cell", 3, true, "hybrid", "proxy — hybrid stack placeholder"},
        {"H16", "SR on trained LSTM gates", 3, true, "hybrid", "proxy — SR placeholder on hybrid"},
        {"H17", "Sheffer-only cell", 3, true, "char_lstm", "extreme H02 — EML-only activations"},
        {"H18", "CA state cell", 3, true, "ssm", "proxy — Wolfram CA via SSM"},
        {"H19", "Izaac-seeded init", 3, true, "hybrid", "proxy — seed-offset hybrid init"},
        {"H20", "Spectral state cell", 3, true, "spectral", "FFT-domain SSM recurrence"},
        {"H21", "Free Energy cell", 3, true, "ssm_gria", "proxy — variational GRIA+SSM"},
        {"H22", "Algebraic fingerprint cell", 3, true, "hybrid", "proxy — hybrid fingerprint tag"},
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

    if (id == "B0") {
        cfg.ngram_context = 4;
        cfg.context_length = 128;
    } else if (id == "H01") {
        cfg.alpha_init = 0.5;
        cfg.alpha_learnable = true;
    } else if (id == "H02" || id == "H17") {
        cfg.use_eml_activation = true;
    } else if (id == "H05") {
        cfg.alpha_learnable = true;
    } else if (id == "H06") {
        cfg.n_experts = 4;
        cfg.online = true;
    } else if (id == "H07") {
        cfg.use_differential_gate = true;
        cfg.bptt_steps = 4;
        cfg.train_ssm = true;
    } else if (id == "H08") {
        cfg.use_context_bank = true;
        cfg.context_bank_slots = 32;
    } else if (id == "H09") {
        cfg.hybrid_blend_logit = 0.5;
        cfg.hybrid_blend_learnable = true;
        cfg.alpha_learnable = true;
    } else if (id == "H10") {
        cfg.alpha_init = 0.485;
        cfg.alpha_learnable = true;
    } else if (id == "H11") {
        cfg.use_multiscale = false;
        cfg.ssm_layers = 1;
    } else if (id == "H12") {
        cfg.compress_interval = 32;
        cfg.max_memory_slots = 128;
    } else if (id == "H13") {
        cfg.max_memory_slots = 256;
        cfg.compress_interval = 16;
    } else if (id == "H14") {
        cfg.n_experts = 8;
        cfg.online = true;
    } else if (id == "H19") {
        cfg.seed += 991;
    }
}

}  // namespace cypha::cyphalm
