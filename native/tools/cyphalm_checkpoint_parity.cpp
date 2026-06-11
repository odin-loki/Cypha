// cyphalm_checkpoint_parity — save/load roundtrip + optional Python char_lstm checkpoint lock.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace fs = std::filesystem;

namespace {

fs::path resolve_checkpoint_json(const fs::path& sidecar_path, const std::string& ckpt_json) {
    fs::path p(ckpt_json);
    if (p.is_absolute()) {
        return p;
    }
    return (sidecar_path.parent_path() / p).lexically_normal();
}

using cypha::cyphalm::ContextMode;
using cypha::cyphalm::CyphaLMConfig;
using cypha::cyphalm::CyphaLMModel;

std::vector<int> synthetic_ids(int n, int vocab, int seed) {
    std::vector<int> out(static_cast<std::size_t>(n));
    std::uint64_t s = static_cast<std::uint64_t>(seed);
    for (int i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        out[static_cast<std::size_t>(i)] = 1 + static_cast<int>(s % static_cast<std::uint64_t>(vocab - 1));
    }
    return out;
}

double eval_bpc(CyphaLMModel& model, const std::vector<int>& ids, int n_eval) {
    return model.eval_bpc(ids, n_eval);
}

bool near(double a, double b, double atol) { return std::abs(a - b) <= atol; }

int test_native_roundtrip(ContextMode mode, const char* label) {
    CyphaLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.d_embed = 8;
    cfg.d_state = 16;
    cfg.ssm_layers = 1;
    cfg.field_dim = 16;
    cfg.lstm_hidden = 16;
    cfg.ngram_context = 1;
    cfg.ngram_fuse_split = true;
    cfg.context_mode = mode;
    cfg.seed = 42;
    cfg.gria_lr = 0.05;
    cfg.lstm_lr = 0.05;
    cfg.train_epochs = 1;
    cfg.bptt_steps = (mode == ContextMode::Hybrid || mode == ContextMode::GriaNgram) ? 8 : 0;
    if (mode == ContextMode::AblationNoSsm || mode == ContextMode::AblationNoDif) {
        cfg.bptt_steps = 0;
    }

    const auto train_ids = synthetic_ids(220, cfg.vocab_size, 42);
    const auto eval_ids = synthetic_ids(80, cfg.vocab_size, 99);

    CyphaLMModel model(cfg);
    model.train_sequence(train_ids, static_cast<int>(train_ids.size()) - 1, cfg.train_epochs);
    const double bpc_before = eval_bpc(model, eval_ids, static_cast<int>(eval_ids.size()) - 1);

    const std::string base = std::string("C:/Temp/cyphalm_ckpt_parity_") + label;
    model.save(base);
    CyphaLMModel loaded = cypha::cyphalm::load_cyphalm_model(base + ".json");
    const double bpc_after = eval_bpc(loaded, eval_ids, static_cast<int>(eval_ids.size()) - 1);

    if (!near(bpc_before, bpc_after, 1e-9)) {
        std::cerr << "FAIL " << label << " roundtrip bpc before=" << bpc_before << " after=" << bpc_after
                  << "\n";
        return 1;
    }
    std::cout << "OK " << label << " roundtrip bpc=" << bpc_before << "\n";
    return 0;
}

int test_python_fixture(const fs::path& sidecar_path, const char* label) {
    nlohmann::json j;
    {
        std::ifstream in(sidecar_path);
    if (!in) {
        std::cerr << "skip " << label << " fixture (missing " << sidecar_path << ")\n";
        return 0;
    }
        in >> j;
    }
    const std::string ckpt = resolve_checkpoint_json(sidecar_path, j.at("checkpoint_json").get<std::string>())
                                 .string();
    CyphaLMModel model = cypha::cyphalm::load_cyphalm_model(ckpt);
    const auto eval_ids = j.at("eval_ids").get<std::vector<int>>();
    const double expected = j.at("expected_bpc").get<double>();
    const double atol = j.value("atol_bpc", 0.05);
    const double got = eval_bpc(model, eval_ids, static_cast<int>(eval_ids.size()) - 1);
    if (!near(got, expected, atol)) {
        std::cerr << "FAIL " << label << " checkpoint bpc got=" << got << " expected=" << expected
                  << " atol=" << atol << "\n";
        return 1;
    }
    std::cout << "OK " << label << " checkpoint bpc=" << got << " (expected " << expected << ")\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int failures = 0;
    failures += test_native_roundtrip(ContextMode::CharLstm, "char_lstm") != 0 ? 1 : 0;
    failures += test_native_roundtrip(ContextMode::Hybrid, "hybrid") != 0 ? 1 : 0;
    failures += test_native_roundtrip(ContextMode::AblationNoDif, "ablation_no_dif") != 0 ? 1 : 0;
    failures += test_native_roundtrip(ContextMode::AblationNoSsm, "ablation_no_ssm") != 0 ? 1 : 0;

    {
        CyphaLMConfig cfg;
        cfg.vocab_size = 32;
        cfg.d_embed = 8;
        cfg.d_state = 16;
        cfg.ssm_layers = 1;
        cfg.field_dim = 16;
        cfg.context_mode = ContextMode::SsmGria;
        cfg.use_hierarchical_ssm = true;
        cfg.compress_interval = 4;
        cfg.seed = 42;
        const auto train_ids = synthetic_ids(120, cfg.vocab_size, 7);
        const auto eval_ids = synthetic_ids(40, cfg.vocab_size, 11);
        CyphaLMModel model(cfg);
        model.train_sequence(train_ids, static_cast<int>(train_ids.size()) - 1, 1);
        const double bpc_before = eval_bpc(model, eval_ids, static_cast<int>(eval_ids.size()) - 1);
        const std::string base = "C:/Temp/cyphalm_ckpt_parity_hier";
        model.save(base);
        CyphaLMModel loaded = cypha::cyphalm::load_cyphalm_model(base + ".json");
        const double bpc_after = eval_bpc(loaded, eval_ids, static_cast<int>(eval_ids.size()) - 1);
        if (!near(bpc_before, bpc_after, 1e-9)) {
            std::cerr << "FAIL hierarchical_ssm roundtrip bpc before=" << bpc_before
                      << " after=" << bpc_after << "\n";
            ++failures;
        } else {
            std::cout << "OK hierarchical_ssm roundtrip bpc=" << bpc_before << "\n";
        }
    }

    fs::path fixture;
    if (argc >= 2) {
        fixture = fs::path(argv[1]);
    } else {
        fixture = fs::path("parity_fixtures/cyphalm_checkpoint/char_lstm/sidecar.json");
    }
    failures += test_python_fixture(fixture, "python char") != 0 ? 1 : 0;

    const fs::path hybrid_fixture = fs::path("parity_fixtures/cyphalm_checkpoint/hybrid/sidecar.json");
    failures += test_python_fixture(hybrid_fixture, "python hybrid") != 0 ? 1 : 0;

    if (failures == 0) {
        std::cout << "All cyphalm_checkpoint_parity checks PASSED.\n";
        return 0;
    }
    std::cerr << failures << " cyphalm_checkpoint_parity check(s) FAILED.\n";
    return 1;
}
