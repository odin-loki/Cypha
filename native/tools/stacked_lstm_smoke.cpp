/// Smoke: hybrid CyphaLM with stacked residual char-LSTM (lstm_layers=2).
#include "cypha/lm/cyphalm_config.hpp"
#include "cypha/lm/cyphalm_model.hpp"
#include "cypha/cyphalm/char_lstm.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

}  // namespace

int main() {
    using cypha::cyphalm::CharLSTMHead;
    using cypha::cyphalm::ContextMode;
    using cypha::cyphalm::CyphaLMConfig;
    using cypha::cyphalm::CyphaLMModel;
    using cypha::cyphalm::apply_hybrid_production_recipe;

    // Unit: stacked head constructs and trains a few steps.
    {
        CharLSTMHead lstm(/*vocab=*/32, /*hidden=*/16, /*seed=*/7,
                          cypha::cyphalm::LSTMInitMode::Default, /*n_layers=*/2);
        if (lstm.n_layers != 2) return fail("n_layers != 2");
        if (lstm.Wx_l.size() != 1 || lstm.Wh_l.size() != 1 || lstm.b_l.size() != 1) {
            return fail("extra layer weight rank");
        }
        std::vector<double> h(16, 0.0);
        std::vector<double> c(16, 0.0);
        double loss0 = 0.0;
        for (int t = 0; t < 8; ++t) {
            const int tok = t % 31;
            const int tgt = (t + 1) % 31;
            loss0 += lstm.train_step(tok, tgt, h, c, 0.05);
        }
        if (!std::isfinite(loss0)) return fail("non-finite stacked train loss");
        const auto lp = lstm.forward(3);
        if (static_cast<int>(lp.size()) != 32) return fail("forward vocab size");
        if (!std::isfinite(lp[0])) return fail("non-finite log_prob");
        std::cout << "stacked_char_lstm_unit OK loss=" << loss0 << "\n";
    }

    // Default layers=1 still constructs single-layer (no extras).
    {
        CharLSTMHead pin(/*vocab=*/16, /*hidden=*/8, /*seed=*/1);
        if (pin.n_layers != 1 || !pin.Wx_l.empty()) return fail("default must be single-layer");
        std::cout << "default_layers1_pin OK\n";
    }

    // Hybrid model with lstm_layers=2: train + predict_next.
    {
        CyphaLMConfig cfg;
        cfg.vocab_size = 64;
        cfg.d_embed = 32;
        cfg.field_dim = 48;
        cfg.d_state = 48;
        cfg.lstm_hidden = 32;
        cfg.lstm_layers = 2;
        cfg.seed = 99;
        apply_hybrid_production_recipe(cfg);
        if (cfg.context_mode != ContextMode::Hybrid) return fail("expected hybrid");
        if (cfg.lstm_layers != 2) return fail("recipe must not clobber lstm_layers");
        if (cfg.lstm_bptt_steps != 8) return fail("production recipe must set bptt=8");
        if (cfg.lstm_optim != "adam") return fail("production recipe must use adam");

        CyphaLMModel model(cfg);
        if (!model.char_lstm() || model.char_lstm()->n_layers != 2) {
            return fail("model LSTM n_layers != 2");
        }

        for (int i = 0; i < 40; ++i) {
            const auto tid = static_cast<std::uint32_t>((i % 17) + 1);
            const auto nxt = static_cast<std::uint32_t>(((i + 1) % 17) + 1);
            (void)model.train_step(tid, nxt);
        }
        const auto pred = model.predict_next(5);
        if (pred.log_probs.size() != static_cast<std::size_t>(cfg.vocab_size)) {
            return fail("predict_next log_probs size");
        }
        if (!std::isfinite(pred.log_probs[0])) return fail("predict_next non-finite");
        std::cout << "hybrid_stacked_lstm2 OK predict_dim=" << pred.log_probs.size() << "\n";
    }

    // Residual LSTM attention-over-memory (opt-in): gated + ramped scale defaults.
    {
        CyphaLMConfig cfg;
        cfg.vocab_size = 64;
        cfg.d_embed = 32;
        cfg.field_dim = 48;
        cfg.d_state = 48;
        cfg.lstm_hidden = 32;
        cfg.lstm_layers = 2;
        cfg.max_memory_slots = 32;
        cfg.compress_interval = 4;
        cfg.use_lstm_memory_attn = true;
        cfg.lstm_memory_attn_scale = 0.10;
        cfg.lstm_memory_attn_min_slots = 8;
        cfg.seed = 101;
        apply_hybrid_production_recipe(cfg);
        if (!cfg.use_lstm_memory_attn) return fail("recipe must not clear memory attn");
        if (cfg.lstm_memory_attn_scale != 0.10) return fail("scale default/override lost");
        if (cfg.lstm_memory_attn_min_slots != 8) return fail("min_slots override lost");

        CyphaLMModel model(cfg);
        // Few steps: below min_slots gate — must still predict finite (attn skipped).
        for (int i = 0; i < 12; ++i) {
            const auto tid = static_cast<std::uint32_t>((i % 11) + 1);
            const auto nxt = static_cast<std::uint32_t>(((i + 3) % 11) + 1);
            (void)model.train_step(tid, nxt);
        }
        {
            const auto pred = model.predict_next(3);
            if (!std::isfinite(pred.log_probs[0])) return fail("gated-off attn non-finite");
        }
        // Fill slots past gate (compress every 4 → ~20 slots after 80 steps).
        for (int i = 12; i < 80; ++i) {
            const auto tid = static_cast<std::uint32_t>((i % 11) + 1);
            const auto nxt = static_cast<std::uint32_t>(((i + 3) % 11) + 1);
            (void)model.train_step(tid, nxt);
        }
        const auto pred = model.predict_next(3);
        if (pred.log_probs.size() != static_cast<std::size_t>(cfg.vocab_size)) {
            return fail("memory-attn predict size");
        }
        if (!std::isfinite(pred.log_probs[0])) return fail("memory-attn non-finite");
        const auto knn = model.hidden_knn_log_probs();
        if (knn.size() != static_cast<std::size_t>(cfg.vocab_size)) {
            return fail("hidden_knn size");
        }
        std::cout << "hybrid_lstm_memory_attn OK\n";
    }

    // Memory-attn with ContextBank keys (prefer bank when use_context_bank).
    {
        CyphaLMConfig cfg;
        cfg.vocab_size = 64;
        cfg.d_embed = 32;
        cfg.field_dim = 48;
        cfg.d_state = 48;
        cfg.lstm_hidden = 32;
        cfg.lstm_layers = 2;
        cfg.max_memory_slots = 32;
        cfg.compress_interval = 4;
        cfg.use_context_bank = true;
        cfg.context_bank_slots = 64;
        cfg.use_lstm_memory_attn = true;
        cfg.lstm_memory_attn_scale = 0.10;
        cfg.lstm_memory_attn_min_slots = 8;
        cfg.seed = 202;
        apply_hybrid_production_recipe(cfg);
        if (!cfg.use_context_bank) return fail("recipe must not clear context bank");

        CyphaLMModel model(cfg);
        for (int i = 0; i < 100; ++i) {
            const auto tid = static_cast<std::uint32_t>((i % 13) + 1);
            const auto nxt = static_cast<std::uint32_t>(((i + 5) % 13) + 1);
            (void)model.train_step(tid, nxt);
        }
        const auto pred = model.predict_next(7);
        if (pred.log_probs.size() != static_cast<std::size_t>(cfg.vocab_size)) {
            return fail("bank-attn predict size");
        }
        if (!std::isfinite(pred.log_probs[0])) return fail("bank-attn non-finite");
        std::cout << "hybrid_lstm_memory_attn_context_bank OK\n";
    }

    // Production defaults: attn remains off.
    {
        CyphaLMConfig cfg;
        apply_hybrid_production_recipe(cfg);
        if (cfg.use_lstm_memory_attn) return fail("production must keep memory attn off");
        if (cfg.lstm_memory_attn_scale != 0.10) return fail("default peak scale should be 0.10");
        if (cfg.lstm_memory_attn_min_slots != 16) return fail("default min_slots should be 16");
        std::cout << "memory_attn_defaults_opt_in OK\n";
    }

    std::cout << "stacked_lstm_smoke PASS\n";
    return 0;
}
