// cyphalm_model_parity — 10-token forward-pass scaffold for Tier 2 CyphaLM native.
// Usage:
//   cyphalm_model_parity
//   cyphalm_model_parity <checkpoint.json>
//   cyphalm_model_parity --mode ssm_gria_no_lstm
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

using cypha::cyphalm::ContextMode;
using cypha::cyphalm::CyphaLMConfig;
using cypha::cyphalm::CyphaLMModel;
using cypha::cyphalm::parse_context_mode;

void print_log_probs(const std::vector<double>& lp, std::uint32_t max_show = 5) {
    const std::uint32_t n = std::min(max_show, static_cast<std::uint32_t>(lp.size()));
    for (std::uint32_t i = 0; i < n; ++i) std::cout << "  id=" << i << " log_p=" << lp[i] << "\n";
}

CyphaLMConfig tiny_config(ContextMode mode) {
    CyphaLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.d_embed = 8;
    cfg.d_state = 16;
    cfg.ssm_layers = 1;
    cfg.field_dim = 16;
    cfg.lstm_hidden = 16;
    cfg.ngram_context = 1;
    cfg.context_mode = mode;
    cfg.compress_interval = 4;
    cfg.max_memory_slots = 8;
    cfg.seed = 42;
    cfg.gria_lr = 0.05;
    return cfg;
}

std::vector<std::uint32_t> default_tokens() {
    return {3, 7, 7, 4, 11, 2, 9, 14, 5, 3};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        ContextMode mode = ContextMode::SsmGria;
        std::string json_path;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--mode" && i + 1 < argc) {
                mode = parse_context_mode(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "cyphalm_model_parity [checkpoint.json] [--mode full|gria_ngram|hybrid|char_lstm|ssm_gria|ssm_gria_no_lstm]\n";
                return 0;
            } else if (arg[0] != '-') {
                json_path = arg;
            }
        }

        CyphaLMModel model = json_path.empty() ? CyphaLMModel(tiny_config(mode))
                                               : CyphaLMModel::from_json_npz(json_path);

        const auto tokens = default_tokens();
        std::cout << "cyphalm_model_parity: mode=" << cypha::cyphalm::context_mode_name(mode)
                  << " vocab=" << model.config().vocab_size << " steps=" << tokens.size() << "\n";

        model.reset_context();
        double loss_sum = 0.0;
        for (std::size_t t = 0; t + 1 < tokens.size(); ++t) {
            const std::uint32_t tok = tokens[t];
            const std::uint32_t nxt = tokens[t + 1];
            const auto pred = model.predict_next(tok);
            const auto step = model.train_step(tok, nxt);
            loss_sum += step.loss;
            std::cout << "step " << t << " token=" << tok << " next=" << nxt << " loss=" << step.loss
                      << " epi=" << step.epistemic_var << "\n";
            print_log_probs(pred.log_probs, 3);
        }
        const std::uint32_t last = tokens.back();
        const auto final_pred = model.predict_next(last);
        std::cout << "final token=" << last << " top0=";
        if (!final_pred.top_k_tokens.empty())
            std::cout << final_pred.top_k_tokens[0] << " p=" << final_pred.top_k_probs[0];
        std::cout << "\nmean_loss=" << (loss_sum / static_cast<double>(tokens.size() - 1)) << "\n";
        std::cout << "OK cyphalm_model_parity scaffold (" << tokens.size() << " tokens)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAIL cyphalm_model_parity: " << ex.what() << "\n";
        return 1;
    }
}
