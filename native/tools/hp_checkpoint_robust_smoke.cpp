/// Checkpoint robustness (CYPHALM_LM_QUALITY_REPORT.md, Reference): loads
/// that used to succeed silently or read out of bounds now fail cleanly.
///  - load_cyphalm_model throws on a bad magic, an unknown version, a missing
///    .hpbin and a truncated one; a good checkpoint still loads identically.
///  - A mixer whose saved shape differs from the constructed one fails the
///    stream; so do tables whose mask, bits and size disagree.
///  - A context or match model saved dropped loads dropped into a model built
///    live, and the reverse (off_ / tab_mask_ follow the file).
///  - Merging or copying folded tables (context, Hebbian, pool) is refused and
///    changes nothing; equal shapes still merge.
///  - fold_auto refuses occupancies outside (0, 1).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "hp/shard_merge.hpp"

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using cypha::cyphalm::CyphaLMConfig;
using cypha::cyphalm::CyphaLMModel;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("hp_checkpoint_robust_smoke FAIL %s\n", what);
        ++failures;
    }
}

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void spit(const fs::path& p, const std::string& s) {
    std::ofstream out(p, std::ios::binary);
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

/// load_cyphalm_model(json) throws and the message contains ``needle``.
bool load_throws(const fs::path& json, const std::string& needle) {
    try {
        (void)cypha::cyphalm::load_cyphalm_model(json.string());
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(needle) != std::string::npos) return true;
        std::printf("  unexpected error: %s\n", e.what());
        return false;
    }
    return false;
}

std::string text_of(int seed, std::size_t n) {
    const char* words[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "[[link]] ", "1984 ", "\n"};
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::string t;
    while (t.size() < n) t += words[rng() % 9];
    return t;
}

CyphaLMModel trained(const CyphaLMConfig& cfg, int seed) {
    CyphaLMModel m(cfg);
    for (char c : text_of(seed, 8000)) m.hp_backend().consume_byte(static_cast<std::uint8_t>(c));
    return m;
}

template <typename M>
std::string saved(const M& m) {
    std::ostringstream os;
    m.checkpoint_write(os);
    return os.str();
}

template <typename M>
bool reads_ok(M& m, const std::string& bytes) {
    std::istringstream is(bytes);
    m.checkpoint_read(is);
    return static_cast<bool>(is);
}

void patch_u32(std::string& s, std::size_t at, std::uint32_t v) { std::memcpy(&s[at], &v, 4); }

// C1: bad header, missing / truncated state, mixer shape.
void loader_failures(const CyphaLMConfig& cfg, const fs::path& dir) {
    CyphaLMModel m = trained(cfg, 1);
    cypha::cyphalm::save_cyphalm_model(m, (dir / "good").string());
    const std::string good = slurp(dir / "good.hpbin");
    const std::string json = slurp(dir / "good.json");
    {
        auto loaded = cypha::cyphalm::load_cyphalm_model((dir / "good.json").string());
        check(loaded.hp_backend().predictor().learned_digest() == m.hp_backend().predictor().learned_digest(),
              "a good checkpoint must still load identically");
    }
    {
        // A lossy config must stay dropped after the file restores live tables.
        nlohmann::json meta = nlohmann::json::parse(json);
        meta["config"]["hp_cm_drop"] = std::uint64_t{1} << hp::Predictor::kCmParaMod;
        meta["config"]["hp_pool_slots"] = 8;
        spit(dir / "lossy.json", meta.dump(2) + "\n");
        spit(dir / "lossy.hpbin", good);
        auto lossy = cypha::cyphalm::load_cyphalm_model((dir / "lossy.json").string());
        hp::Predictor& lp = lossy.hp_backend().predictor();
        check(lp.context_model_bits(hp::Predictor::kCmParaMod) == 0,
              "cm_drop must survive loading a checkpoint that has the model");
        check(lp.pool_slot_bits(0) > 0, "an active pool slot must stay live");
        check(lp.pool_slot_bits(8) == 0, "a pool slot past pool_slots must load dropped");
        lossy.hp_backend().consume_byte(static_cast<std::uint8_t>('a'));
        check(lp.pool_slot_bits(8) == 0, "an inactive pool slot must not learn");
    }
    auto variant = [&](const char* name, const std::string& bin) {
        spit(dir / (std::string(name) + ".json"), json);
        spit(dir / (std::string(name) + ".hpbin"), bin);
        return dir / (std::string(name) + ".json");
    };
    std::string bad = good;
    bad[0] = 'X';
    check(load_throws(variant("magic", bad), "not an hp checkpoint"), "bad magic must throw");
    bad = good;
    patch_u32(bad, 4, 99);
    check(load_throws(variant("ver", bad), "unsupported hp checkpoint version 99"), "unknown version must throw");
    check(load_throws(variant("trunc", good.substr(0, good.size() / 2)), "hp checkpoint read failed"),
          "a truncated .hpbin must throw");
    spit(dir / "nobin.json", json);
    check(load_throws(dir / "nobin.json", "hp checkpoint missing"), "a missing .hpbin must throw");

    // Mixer: a saved network of another shape does not load.
    hp::MixerNet a(10, {4, 8}, 16, 2);
    hp::MixerNet same(10, {4, 8}, 16, 2);
    hp::MixerNet wider(12, {4, 8}, 16, 2);
    hp::MixerNet fewer(10, {4}, 16, 2);
    const std::string mix = saved(a);
    check(reads_ok(same, mix), "mixer of the same shape must load");
    check(!reads_ok(wider, mix), "mixer with more inputs must not load");
    check(!reads_ok(fewer, mix), "mixer with fewer weight sets must not load");
}

// C3: off_ / bits_ / tab_mask_ follow the file; inconsistent shapes fail.
void reconcile_on_load() {
    // Context model saved dropped, loaded into one built live: stays off.
    hp::ContextModel live(16, 255), dropped(16, 255);
    dropped.set_context(0x1234u);
    int out[hp::ContextModel::kOutputs] = {};
    dropped.predict(1, 2048, out);
    dropped.update(1);
    dropped.drop();
    check(reads_ok(live, saved(dropped)), "dropped context model must load");
    check(live.table_bits() == 0, "context model saved dropped must load dropped");
    live.set_context(0x1234u);
    out[0] = out[1] = 7;
    live.predict(1, 2048, out);
    check(out[0] == 0 && out[1] == 0, "context model saved dropped must predict nothing");

    // And the reverse: a live table loaded into a model built dropped.
    hp::ContextModel full(16, 255), built_off(0, 255);
    for (std::uint32_t h = 1; h < 500; ++h) {
        full.set_context(h * 0x9E3779B1u);
        full.predict(1, 2048, out);
        full.update(static_cast<int>(h & 1));
    }
    check(reads_ok(built_off, saved(full)), "live context model must load into one built dropped");
    check(built_off.table_bits() == 16, "context model must take the saved table bits");
    int a_out[2] = {}, b_out[2] = {};
    full.set_context(77u * 0x9E3779B1u);
    built_off.set_context(77u * 0x9E3779B1u);
    full.predict(1, 2048, a_out);
    built_off.predict(1, 2048, b_out);
    check(a_out[0] == b_out[0] && a_out[1] == b_out[1], "reloaded context model must predict as saved");

    // Inconsistent context model: bits_ says 16, the table holds 2^14.
    hp::ContextModel small(14, 255), target(16, 255);
    std::string s = saved(small);
    patch_u32(s, 4, 16);  // mask_, bits_, ...
    check(!reads_ok(target, s), "context model bits/table mismatch must fail");
    s = saved(small);
    patch_u32(s, 12, 0xFFFFu);  // mask_, bits_, limit_, then the table's own mask
    check(!reads_ok(target, s), "hash table mask/size mismatch must fail");

    // Match model saved dropped: off after load, no match adopted.
    hp::ByteRing ring(16);
    hp::MatchModel m_dropped(&ring, 16), m_live(&ring, 16), m_full(&ring, 16), m_off(&ring, 0);
    m_dropped.drop();
    check(reads_ok(m_live, saved(m_dropped)), "dropped match model must load");
    check(m_live.table_bits() == 0, "match model saved dropped must load dropped");
    std::uint64_t hist = 0;
    for (char c : text_of(3, 2000)) {
        const auto b = static_cast<std::uint8_t>(c);
        ring.push(b);
        hist = (hist << 8) | b;
        m_live.push_byte(b, hist);
        m_full.push_byte(b, hist);
    }
    check(m_live.match_len() == 0 && m_live.predict(1, 0) == 0, "match model saved dropped must predict nothing");
    check(m_full.match_len() > 0, "live match model must find matches in repeated text");
    check(reads_ok(m_off, saved(m_full)), "live match model must load into one built dropped");
    check(m_off.table_bits() == 16 && m_off.match_len() == m_full.match_len(),
          "match model must take the saved table and state");
    s = saved(m_full);
    patch_u32(s, 8, 0x3FFFu);  // order_, skip_, tab_mask_
    check(!reads_ok(m_off, s), "match model mask/table mismatch must fail");

    // Hebbian: the mask must cover its synapse and bit-history tables.
    hp::HebbianModel h1(14, 255), h2(14, 255);
    s = saved(h1);
    patch_u32(s, 0, 0xFFFFu);
    check(!reads_ok(h2, s), "Hebbian mask/table mismatch must fail");
}

// C2: folded tables are not merged or copied; nothing changes.
void merge_refusal(const CyphaLMConfig& cfg) {
    struct Fold {
        const char* what;
        int cm, pool, hebb;
    };
    const Fold folds[] = {{"context", 15, 0, 0}, {"pool", 0, 14, 0}, {"Hebbian", 0, 0, 14}};
    CyphaLMModel dst = trained(cfg, 4);
    hp::Predictor& d = dst.hp_backend().predictor();
    for (const Fold& f : folds) {
        CyphaLMModel src = trained(cfg, 5);
        src.fold_hp_tables(f.cm, 0, f.pool, 0, f.hebb);
        const hp::Predictor& sp = src.hp_backend().predictor();
        const std::uint64_t before = d.learned_digest();
        char msg[96];
        std::snprintf(msg, sizeof msg, "%s-folded source must not match", f.what);
        check(!d.tables_match(sp), msg);
        std::snprintf(msg, sizeof msg, "%s-folded merge must be refused", f.what);
        check(!d.merge_shard_tables(sp, 1, 1), msg);
        check(!d.transfer_tables_from(sp), msg);
        check(hp::merge_predictor_tables(d, 1, sp, 1) == hp::MergeStatus::ConfigMismatch, msg);
        std::snprintf(msg, sizeof msg, "%s-folded refused merge must change nothing", f.what);
        check(d.learned_digest() == before, msg);
    }
    // Equal shapes still merge.
    CyphaLMModel peer = trained(cfg, 6);
    const std::uint64_t before = d.learned_digest();
    check(d.tables_match(peer.hp_backend().predictor()), "same-config models must match");
    check(d.merge_shard_tables(peer.hp_backend().predictor(), 1, 1) && d.learned_digest() != before,
          "same-config models must merge");

    // The models alone.
    hp::ContextModel c16(16, 255), c14(14, 255);
    check(!c16.merge_tables_from(c14, 1, 1) && !c16.copy_tables_from(c14), "context model size mismatch");
    hp::HebbianModel h16(16, 255), h14(14, 255);
    check(!h16.merge_tables_from(h14, 1, 1) && !h16.copy_tables_from(h14), "Hebbian size mismatch");
    hp::DiscoveryPool p16(16, 1), p14(14, 1), p16_4(16, 1, 4);
    check(!p16.merge_tables_from(p14, 1, 1) && !p16.copy_tables_from(p14), "pool size mismatch");
    check(!p16.tables_match(p16_4), "pool with fewer active slots must not match");
}

// C4: occupancy outside (0, 1) is refused.
void fold_auto_range(const CyphaLMConfig& cfg) {
    CyphaLMModel m = trained(cfg, 7);
    hp::Predictor& p = m.hp_backend().predictor();
    const std::uint64_t before = p.learned_digest();
    check(p.fold_auto(1.0) == 0 && p.fold_auto(1.5) == 0 && p.fold_auto(0.0) == 0 && p.fold_auto(-0.5) == 0,
          "fold_auto outside (0, 1) must fold nothing");
    check(p.learned_digest() == before, "refused fold_auto must change nothing");
    for (double bad : {0.0, 1.0, 2.0}) {
        bool threw = false;
        try {
            (void)m.hp_backend().fold_auto(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "HpSequenceBackend::fold_auto outside (0, 1) must throw");
    }
    check(m.hp_backend().fold_auto(0.5) > 0, "fold_auto(0.5) must still fold");
}

}  // namespace

int main() {
    CyphaLMConfig cfg;
    cfg.hp_table_bits = 16;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    const fs::path dir = fs::temp_directory_path() / "hp_checkpoint_robust_smoke";
    fs::create_directories(dir);
    loader_failures(cfg, dir);
    reconcile_on_load();
    merge_refusal(cfg);
    fold_auto_range(cfg);
    fs::remove_all(dir);
    if (failures != 0) {
        std::printf("hp_checkpoint_robust_smoke: %d check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("hp_checkpoint_robust_smoke OK: bad header / missing / truncated state refused, mixer shape "
                "checked, dropped tables reconciled, folded merges refused, fold_auto range enforced\n");
    return 0;
}
