/// Serve-time table folding (CyphaLMModel::fold_hp_tables): a folded model
/// (context, match, pool and Hebbian caps plus a dropped context model) saves
/// and reloads at its new size and predicts identically after reload, and the
/// checkpoint shrinks. Then the same for the per-table occupancy fold
/// (HpSequenceBackend::fold_auto) on a model trained with the upstream mixer
/// settings (layer-1 rate scale, dot scale, layer-1 skip: checkpoint v4).
/// Last, the optional upstream context models (hp_extra_cms): gate24 still
/// writes a v4 checkpoint without the key, each model changes predictions,
/// and a model with all of them saves (v5), reloads and predicts identically,
/// also after fold_auto.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

bool same_after_reload(cypha::cyphalm::CyphaLMModel& model, cypha::cyphalm::CyphaLMModel& loaded,
                       const std::string& text, const char* what) {
    for (int i = 0; i < 300; ++i) {
        const auto a = model.hp_backend().serve_next_byte_log_probs(256);
        const auto b = loaded.hp_backend().serve_next_byte_log_probs(256);
        double err = 0.0, z = 0.0;
        for (int k = 0; k < 256; ++k) {
            err = std::max(err, std::abs(a[k] - b[k]));
            z += std::exp(a[k]);
        }
        if (err > 1e-12 || std::abs(z - 1.0) > 1e-9) {
            std::printf("hp_fold_smoke FAIL %s step %d: reload differs by %.3g (sum %.12f)\n", what, i, err, z);
            return false;
        }
        const auto c = static_cast<std::uint8_t>(text[static_cast<std::size_t>(i)]);
        model.hp_backend().consume_byte(c);
        loaded.hp_backend().consume_byte(c);
    }
    return true;
}

std::uint32_t hpbin_version(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    char head[8] = {};
    in.read(head, 8);
    std::uint32_t v = 0;
    std::memcpy(&v, head + 4, 4);
    return v;
}

bool file_contains(const std::filesystem::path& p, const std::string& needle) {
    std::ifstream in(p);
    const std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return all.find(needle) != std::string::npos;
}

// Optional upstream context models: save/load round trip and identity.
int extra_cms_smoke(const cypha::cyphalm::CyphaLMConfig& base, const std::filesystem::path& dir) {
    const char* words[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "[[link]] ", "1984 ", "\n",
                           "''it'' ", "'''bold''' ", "'''''both''''' ", "<ref name=\"x\">", "<ref>",
                           "<ref group=note>", "</ref> ", "<ref name=y/> ", "Then ", "{{cite}} "};
    std::mt19937 rng(9);
    std::string text;
    while (text.size() < 30000) text += words[rng() % 19];

    // gate24: v4 checkpoint, no hp_extra_cms key.
    cypha::cyphalm::CyphaLMModel g(base);
    for (char c : text) g.hp_backend().consume_byte(static_cast<std::uint8_t>(c));
    cypha::cyphalm::save_cyphalm_model(g, (dir / "x0").string());
    if (hpbin_version(dir / "x0.hpbin") != 4 || file_contains(dir / "x0.json", "hp_extra_cms")) {
        std::printf("hp_fold_smoke FAIL extra_cms=0 must write a v4 checkpoint without hp_extra_cms\n");
        return 1;
    }
    const auto g_next = g.hp_backend().serve_next_byte_log_probs(256);

    // Each model alone is active (changes the next-byte distribution).
    for (int k = 0; k < 7; ++k) {
        cypha::cyphalm::CyphaLMConfig c = base;
        c.hp_extra_cms = 1u << k;
        cypha::cyphalm::CyphaLMModel m(c);
        for (char ch : text) m.hp_backend().consume_byte(static_cast<std::uint8_t>(ch));
        if (m.hp_backend().serve_next_byte_log_probs(256) == g_next) {
            std::printf("hp_fold_smoke FAIL extra context model %d has no effect\n", k);
            return 1;
        }
    }

    cypha::cyphalm::CyphaLMConfig c = base;
    c.hp_extra_cms = 127;
    cypha::cyphalm::CyphaLMModel m(c);
    for (char ch : text) m.hp_backend().consume_byte(static_cast<std::uint8_t>(ch));
    cypha::cyphalm::save_cyphalm_model(m, (dir / "x127").string());
    if (hpbin_version(dir / "x127.hpbin") != 5) {
        std::printf("hp_fold_smoke FAIL extra_cms=127 must write a v5 checkpoint\n");
        return 1;
    }
    auto loaded = cypha::cyphalm::load_cyphalm_model((dir / "x127.json").string());
    if (loaded.config().hp_extra_cms != 127) {
        std::printf("hp_fold_smoke FAIL hp_extra_cms not reloaded\n");
        return 1;
    }
    if (!same_after_reload(m, loaded, text, "extra_cms")) return 1;
    // A second save of the reloaded model is byte-identical to one of the original.
    cypha::cyphalm::save_cyphalm_model(m, (dir / "x127a").string());
    cypha::cyphalm::save_cyphalm_model(loaded, (dir / "x127b").string());
    {
        std::ifstream a(dir / "x127a.hpbin", std::ios::binary), b(dir / "x127b.hpbin", std::ios::binary);
        const std::string sa((std::istreambuf_iterator<char>(a)), std::istreambuf_iterator<char>());
        const std::string sb((std::istreambuf_iterator<char>(b)), std::istreambuf_iterator<char>());
        if (sa != sb) {
            std::printf("hp_fold_smoke FAIL extra_cms re-save differs (%zu vs %zu bytes)\n", sa.size(), sb.size());
            return 1;
        }
    }
    const std::size_t freed = m.hp_backend().fold_auto(0.6);
    cypha::cyphalm::save_cyphalm_model(m, (dir / "x127f").string());
    auto folded = cypha::cyphalm::load_cyphalm_model((dir / "x127f.json").string());
    if (freed == 0 || !same_after_reload(m, folded, text, "extra_cms fold_auto")) {
        if (freed == 0) std::printf("hp_fold_smoke FAIL extra_cms fold_auto freed nothing\n");
        return 1;
    }
    std::printf("hp_fold_smoke OK extra_cms: v4 at 0, all 7 models active, v5 reload identical\n");
    return 0;
}

}  // namespace

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
    if (!same_after_reload(model, loaded, text, "caps")) return 1;
    if (!(folded_size < full_size / 2)) {
        std::printf("hp_fold_smoke FAIL checkpoint %zu -> %zu bytes\n", static_cast<std::size_t>(full_size),
                    static_cast<std::size_t>(folded_size));
        return 1;
    }

    // Occupancy fold on an upstream-mixer model.
    cypha::cyphalm::CyphaLMConfig ucfg = cfg;
    ucfg.hp_lr1_scale = 40;
    ucfg.hp_mixer_scale = 49152;
    ucfg.hp_mixer_skip = 56;
    ucfg.hp_mixer_skip_l1 = 80;
    cypha::cyphalm::CyphaLMModel up(ucfg);
    for (char c : text) up.hp_backend().consume_byte(static_cast<std::uint8_t>(c));
    cypha::cyphalm::save_cyphalm_model(up, (dir / "up_full").string());
    const std::size_t freed = up.hp_backend().fold_auto(0.6);
    cypha::cyphalm::save_cyphalm_model(up, (dir / "up_auto").string());
    auto up_loaded = cypha::cyphalm::load_cyphalm_model((dir / "up_auto.json").string());
    const auto up_full = std::filesystem::file_size(dir / "up_full.hpbin");
    const auto up_auto = std::filesystem::file_size(dir / "up_auto.hpbin");
    if (extra_cms_smoke(cfg, dir) != 0) {
        std::filesystem::remove_all(dir);
        return 1;
    }
    std::filesystem::remove_all(dir);
    if (up_loaded.config().hp_lr1_scale != 40 || up_loaded.config().hp_mixer_scale != 49152 ||
        up_loaded.config().hp_mixer_skip_l1 != 80) {
        std::printf("hp_fold_smoke FAIL upstream mixer settings not reloaded\n");
        return 1;
    }
    if (!same_after_reload(up, up_loaded, text, "fold_auto")) return 1;
    if (freed == 0 || !(up_auto + freed / 2 < up_full)) {
        std::printf("hp_fold_smoke FAIL fold_auto freed %zu, checkpoint %zu -> %zu bytes\n", freed,
                    static_cast<std::size_t>(up_full), static_cast<std::size_t>(up_auto));
        return 1;
    }
    std::printf("hp_fold_smoke OK caps %zu -> %zu bytes, fold_auto %zu -> %zu bytes; reloads identical\n",
                static_cast<std::size_t>(full_size), static_cast<std::size_t>(folded_size),
                static_cast<std::size_t>(up_full), static_cast<std::size_t>(up_auto));
    return 0;
}
