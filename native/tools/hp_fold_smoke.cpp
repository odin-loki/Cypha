/// Serve-time table folding (CyphaLMModel::fold_hp_tables): a folded model
/// (context, match, pool and Hebbian caps plus a dropped context model) saves
/// and reloads at its new size and predicts identically after reload, and the
/// checkpoint shrinks.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.hp_table_bits = 18;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cypha::cyphalm::CyphaLMModel model(cfg);
    const char* words[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "[[link]] ", "1984 ", "\n"};
    std::mt19937 rng(5);
    std::string text;
    while (text.size() < 30000) text += words[rng() % 9];
    for (char c : text) model.hp_backend().consume_byte(static_cast<std::uint8_t>(c));

    const auto dir = std::filesystem::temp_directory_path() / "hp_fold_smoke";
    std::filesystem::create_directories(dir);
    cypha::cyphalm::save_cyphalm_model(model, (dir / "full").string());
    model.fold_hp_tables(/*cm*/ 17, /*match*/ 16, /*pool*/ 16, /*drop*/ 1ull << 3, /*hebb*/ 16);
    cypha::cyphalm::save_cyphalm_model(model, (dir / "folded").string());
    auto loaded = cypha::cyphalm::load_cyphalm_model((dir / "folded.json").string());

    const auto full_size = std::filesystem::file_size(dir / "full.hpbin");
    const auto folded_size = std::filesystem::file_size(dir / "folded.hpbin");
    for (int i = 0; i < 300; ++i) {
        const auto a = model.hp_backend().serve_next_byte_log_probs(256);
        const auto b = loaded.hp_backend().serve_next_byte_log_probs(256);
        double err = 0.0, z = 0.0;
        for (int k = 0; k < 256; ++k) {
            err = std::max(err, std::abs(a[k] - b[k]));
            z += std::exp(a[k]);
        }
        if (err > 1e-12 || std::abs(z - 1.0) > 1e-9) {
            std::printf("hp_fold_smoke FAIL step %d: reload differs by %.3g (sum %.12f)\n", i, err, z);
            return 1;
        }
        const auto c = static_cast<std::uint8_t>(text[static_cast<std::size_t>(i)]);
        model.hp_backend().consume_byte(c);
        loaded.hp_backend().consume_byte(c);
    }
    std::filesystem::remove_all(dir);
    if (!(folded_size < full_size / 2)) {
        std::printf("hp_fold_smoke FAIL checkpoint %zu -> %zu bytes\n", static_cast<std::size_t>(full_size),
                    static_cast<std::size_t>(folded_size));
        return 1;
    }
    std::printf("hp_fold_smoke OK %zu -> %zu bytes; reload identical over 300 bytes\n",
                static_cast<std::size_t>(full_size), static_cast<std::size_t>(folded_size));
    return 0;
}
