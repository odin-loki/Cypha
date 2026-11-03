/// Component dumps (ComponentDump; cyphalm_lm_quality --dump-components) of
/// manifest-style composite models (two shards, ∞-gram over a corpus, two
/// small Transformer experts), for bench/lm_compare/mixsim.py:
///  - scored_parts: the first byte's served distribution, rebuilt from this
///    model's and the member's log P, the ∞-gram counts and the experts' log
///    P at the start weights (geometric mix, ∞-gram mix, linear neural mix),
///    equals the served one.
///  - Dumping serves the same distributions, bit for bit, as not dumping.
///  - Each array file holds one row per byte at the dtype's width; float32
///    and float16 rows are the served log P rounded to that precision;
///    meta.json holds the shapes, settings, start weights and the caller's
///    keys.
///  - A session cache and an unknown dtype are refused.
/// The dumps (defaults, the variants, frozen mixing weights, a member-less
/// composite, a plain model; float64 unless named) and the parts, manifests
/// and eval text stay in the directory given as the first argument: the
/// harness test dumps a manifest from there through cyphalm_lm_quality, and
/// the mixsim tests replay every dump against its NLL, and one run's dump
/// with another's settings against that run's NLL.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/component_dump.hpp"
#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

namespace fs = std::filesystem;
using cypha::cyphalm::ComponentDump;
using cypha::cyphalm::CyphaLMModel;
using cypha::cyphalm::HpSequenceBackend;

constexpr double kLn2 = 0.6931471805599453;

int fail(const std::string& what) {
    std::printf("cyphalm_component_dump_smoke FAIL: %s\n", what.c_str());
    return 1;
}

std::string corpus(unsigned seed, int variant, std::size_t n) {
    const char* a[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "Then ", "dog ", "ran ", "\n"};
    const char* b[] = {"insects ", "have ", "six ", "legs ", "and ", "wings. ", "The ", "body ", "\n"};
    std::mt19937 rng(seed);
    std::string t;
    while (t.size() < n) t += variant == 0 ? a[rng() % 10] : b[rng() % 9];
    t.resize(n);
    return t;
}

/// A small random Transformer (BGT1), as in cyphalm_neural_smoke.
void write_bgt(const fs::path& path, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    const std::uint32_t gl = 2, gd = 16, gh = 2, gctx = 8;
    std::ofstream os(path, std::ios::binary);
    os.write("BGT1", 4);
    const std::uint32_t hdr[4] = {gl, gd, gh, gctx};
    os.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    auto put = [&](std::size_t n, float mean) {
        for (std::size_t i = 0; i < n; ++i) {
            const float f = mean + nd(rng);
            os.write(reinterpret_cast<const char*>(&f), sizeof(f));
        }
    };
    put(256 * gd, 0.0f);
    put(gctx * gd, 0.0f);
    for (std::uint32_t l = 0; l < gl; ++l) {
        put(gd, 1.0f);
        put(gd, 0.0f);
        put(3 * gd * gd, 0.0f);
        put(gd * gd, 0.0f);
        put(gd, 1.0f);
        put(gd, 0.0f);
        put(4 * gd * gd, 0.0f);
        put(4 * gd * gd, 0.0f);
    }
    put(gd, 1.0f);
    put(gd, 0.0f);
}

/// Two shards, the ∞-gram over a corpus, two adapting experts.
nlohmann::json base_meta() {
    return {{"cyphalm_ensemble", 1},
            {"members", {{{"checkpoint", "shard_a.json"}}, {{"checkpoint", "shard_b.json"}}}},
            {"learning_rate", 0.05},
            {"infinigram", "corpus.txt"},
            {"neural", {"expert_a.bgt", "expert_b.bgt"}},
            {"neural_learning_rate", 0.1},
            {"neural_adapt", 0.01}};
}

void write_manifest(const fs::path& dir, const std::string& name, const nlohmann::json& extra) {
    nlohmann::json meta = base_meta();
    for (auto it = extra.begin(); it != extra.end(); ++it) meta[it.key()] = it.value();
    std::ofstream(dir / (name + ".json")) << meta.dump(1);
}

void write_parts(const fs::path& dir) {
    fs::create_directories(dir);
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.hp_table_bits = 16;
    const std::string ta = corpus(1, 0, 6000), tb = corpus(2, 1, 6000);
    CyphaLMModel a(cfg), b(cfg);
    for (unsigned char c : ta) a.hp_backend().consume_byte(c);
    for (unsigned char c : tb) b.hp_backend().consume_byte(c);
    cypha::cyphalm::save_cyphalm_model(a, (dir / "shard_a").string());
    cypha::cyphalm::save_cyphalm_model(b, (dir / "shard_b").string());
    {
        std::ofstream os(dir / "corpus.txt", std::ios::binary);
        os << corpus(3, 1, 12000) << corpus(4, 0, 12000);
    }
    write_bgt(dir / "expert_a.bgt", 5);
    write_bgt(dir / "expert_b.bgt", 6);
    write_manifest(dir, "linear", nlohmann::json::object());
    write_manifest(dir, "variants", {{"neural_mix", "log"},
                                     {"infinigram_mode", "longest16"},
                                     {"final_temperature", 0.9},
                                     {"final_temperature_lr", 0.02}});
    write_manifest(dir, "switch_gate", {{"neural_mix", "switch"}, {"ensemble_gate", true}, {"final_temperature", 1.1}});
    write_manifest(dir, "longest16", {{"infinigram_mode", "longest16"}});
    // No members: the ∞-gram and one expert over a lone model.
    write_manifest(dir, "single", {{"members", {{{"checkpoint", "shard_a.json"}}}}, {"neural", {"expert_a.bgt"}}});
    // Both vocabularies, then a stretch neither shard saw.
    std::ofstream(dir / "eval.txt", std::ios::binary)
        << corpus(11, 1, 500) + corpus(12, 0, 500) + "Zebras quietly graze on 42 hills. " + corpus(13, 1, 300);
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

/// The harness's eval loop: score, dump, observe (learning on). Returns the
/// served distributions; meta.json gets the NLL as the harness reports it.
std::vector<std::vector<double>> dump_text(CyphaLMModel& m, const std::string& text, const fs::path& out,
                                           ComponentDump::Dtype dtype) {
    auto& h = m.hp_backend();
    ComponentDump dump(out.string(), h, dtype, h.learning() && h.mixing_learning());
    std::vector<std::vector<double>> served;
    double nll = 0.0;
    for (unsigned char c : text) {
        const auto lp = h.next_byte_log_probs(256);
        dump.write(h, c, lp);
        nll += -lp[c] / kLn2;
        h.observe_next_byte(c);
        served.push_back(lp);
    }
    const double n = static_cast<double>(text.size());
    dump.finish({{"eval", {{"nll_bits_per_byte", nll / n}, {"bytes", text.size()}}}});
    return served;
}

std::vector<std::vector<double>> serve_text(CyphaLMModel& m, const std::string& text) {
    auto& h = m.hp_backend();
    std::vector<std::vector<double>> served;
    for (unsigned char c : text) {
        served.push_back(h.next_byte_log_probs(256));
        h.observe_next_byte(c);
    }
    return served;
}

double half_to_double(std::uint16_t h) {
    const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    const double v = e == 0 ? std::ldexp(m, -24) : e == 31 ? std::numeric_limits<double>::infinity()
                                                            : std::ldexp(1024 + m, e - 25);
    return (h & 0x8000) ? -v : v;
}

/// The first byte rebuilt from its parts at the start weights (every bucket
/// starts alike: 0.8 / 0.1 / 0.1 and 0.7 / 0.15 / 0.15).
int check_first_byte(const fs::path& dir) {
    auto m = cypha::cyphalm::load_cyphalm_model((dir / "linear.json").string());
    auto& h = m.hp_backend();
    const HpSequenceBackend::MixingState st = h.mixing_state();
    const auto lp = h.next_byte_log_probs(256);
    const HpSequenceBackend::ScoredParts parts = h.scored_parts();
    if (parts.models.size() != 2 || parts.neural.size() != 2 || parts.ig_longest == nullptr ||
        parts.ig_reliable == nullptr)
        return fail("scored_parts shape");
    std::vector<double> mix(256, 0.0);
    for (std::size_t i = 0; i < parts.models.size(); ++i)
        for (std::size_t b = 0; b < 256; ++b) mix[b] += st.ensemble[i] * (*parts.models[i])[b];
    double mx = -std::numeric_limits<double>::infinity(), z = 0.0;
    for (double v : mix) mx = std::max(mx, v);
    for (double v : mix) z += std::exp(v - mx);
    for (double& v : mix) v -= mx + std::log(z);
    auto freq = [&](const cypha::cyphalm::InfiniGram::Result& r, std::size_t b, double p0) {
        double tot = 0.0;
        for (std::uint32_t c : r.count) tot += c;
        return tot > 0.0 ? r.count[b] / tot : p0;
    };
    const auto& wi = st.ig.front();
    const std::size_t k = parts.neural.size();
    double worst = 0.0;
    for (std::size_t b = 0; b < 256; ++b) {
        const double p0 = std::exp(mix[b]);
        const double pig = wi[0] * p0 + wi[1] * freq(*parts.ig_longest, b, p0) + wi[2] * freq(*parts.ig_reliable, b, p0);
        double p = st.neural[0] * std::exp(std::log(std::max(pig, 1e-300)));
        for (std::size_t i = 0; i < k; ++i) p += st.neural[i + 1] * std::exp((*parts.neural[i])[b]);
        worst = std::max(worst, std::abs(std::log(std::max(p, 1e-300)) - lp[b]));
    }
    if (worst > 1e-12) return fail("first byte rebuilt from scored_parts differs by " + std::to_string(worst));
    return 0;
}

std::size_t width(const std::string& dtype) { return dtype == "float16" ? 2 : dtype == "float32" ? 4 : 8; }

/// Array sizes against meta.json.
int check_layout(const fs::path& d, std::size_t n, std::size_t models, std::size_t experts, bool ig) {
    nlohmann::json meta;
    std::ifstream(d / "meta.json") >> meta;
    if (meta.at("format") != "cyphalm_components" || meta.at("version") != 1 || meta.at("bytes") != n ||
        meta.at("models") != models || meta.at("neural") != experts || meta.at("infinigram") != ig ||
        !meta.at("eval").contains("nll_bits_per_byte"))
        return fail(d.string() + ": meta.json");
    const std::size_t w = width(meta.at("dtype").get<std::string>());
    auto size_is = [&](const char* name, std::size_t bytes) {
        return bytes == 0 ? !fs::exists(d / name) : fs::exists(d / name) && fs::file_size(d / name) == bytes;
    };
    if (!size_is("truth.bin", n) || !size_is("models.bin", n * models * 256 * w) ||
        !size_is("served.bin", n * 256 * w) || !size_is("neural.bin", n * experts * 256 * w) ||
        !size_is("ig_n.bin", ig ? n * 2 * 4 : 0) || !size_is("ig_total.bin", ig ? n * 2 * 8 : 0) ||
        !size_is("ig_count.bin", ig ? n * 2 * 256 * 4 : 0))
        return fail(d.string() + ": array file sizes");
    if (meta.at("start").at("ensemble").size() != models) return fail(d.string() + ": start ensemble weights");
    return 0;
}

int check_dumps(const fs::path& dir) {
    const std::string text = read_file(dir / "eval.txt");
    const std::size_t n = text.size();
    const fs::path dumps = dir / "dumps";
    using D = ComponentDump::Dtype;
    // Dumping serves what not dumping serves.
    std::vector<std::vector<double>> served;
    {
        auto m = cypha::cyphalm::load_cyphalm_model((dir / "linear.json").string());
        served = dump_text(m, text, dumps / "linear", D::F64);
        auto m2 = cypha::cyphalm::load_cyphalm_model((dir / "linear.json").string());
        if (serve_text(m2, text) != served) return fail("dumping changed the served distributions");
    }
    for (const char* name : {"variants", "switch_gate", "longest16"}) {
        auto m = cypha::cyphalm::load_cyphalm_model((dir / (std::string(name) + ".json")).string());
        (void)dump_text(m, text, dumps / name, D::F64);
    }
    for (auto [name, dtype] : {std::pair{"linear_f32", D::F32}, std::pair{"linear_f16", D::F16}}) {
        auto m = cypha::cyphalm::load_cyphalm_model((dir / "linear.json").string());
        if (dump_text(m, text, dumps / name, dtype) != served) return fail("runs differ across loads");
    }
    {
        auto m = cypha::cyphalm::load_cyphalm_model((dir / "linear.json").string());
        m.hp_backend().set_mixing_learning(false);
        (void)dump_text(m, text, dumps / "frozen_mixing", D::F64);
        nlohmann::json meta;
        std::ifstream(dumps / "frozen_mixing" / "meta.json") >> meta;
        if (meta.at("learn_mix") != false) return fail("frozen mixing weights not recorded");
    }
    {
        auto m = cypha::cyphalm::load_cyphalm_model((dir / "single.json").string());
        if (m.hp_backend().ensemble_size() != 0 || !m.hp_backend().is_composite()) return fail("single manifest shape");
        (void)dump_text(m, text, dumps / "single", D::F64);
    }
    {
        auto m = cypha::cyphalm::load_cyphalm_model((dir / "shard_a.json").string());
        if (m.hp_backend().is_composite()) return fail("plain model composite");
        (void)dump_text(m, text, dumps / "plain", D::F64);
    }
    for (const char* name : {"linear", "variants", "switch_gate", "longest16", "linear_f32", "linear_f16", "frozen_mixing"})
        if (int rc = check_layout(dumps / name, n, 2, 2, true)) return rc;
    if (int rc = check_layout(dumps / "single", n, 1, 1, true)) return rc;
    if (int rc = check_layout(dumps / "plain", n, 1, 0, false)) return rc;
    // Narrow dtypes are the served log P rounded to their precision.
    {
        const std::string f32 = read_file(dumps / "linear_f32" / "served.bin");
        const std::string f16 = read_file(dumps / "linear_f16" / "served.bin");
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t b = 0; b < 256; ++b) {
                const double v = served[r][b];
                float x = 0.0f;
                std::uint16_t hx = 0;
                std::memcpy(&x, f32.data() + (r * 256 + b) * 4, 4);
                std::memcpy(&hx, f16.data() + (r * 256 + b) * 2, 2);
                if (x != static_cast<float>(v)) return fail("float32 row is not the served log P");
                if (std::abs(half_to_double(hx) - v) > std::abs(v) * std::ldexp(1.0, -11) + std::ldexp(1.0, -25))
                    return fail("float16 row off by more than half precision");
            }
        }
    }
    // The plain model's one model is its served distribution; the truth bytes are the text.
    if (read_file(dumps / "plain" / "models.bin") != read_file(dumps / "plain" / "served.bin"))
        return fail("plain model: models.bin differs from served.bin");
    if (read_file(dumps / "linear" / "truth.bin") != text) return fail("truth.bin is not the text");
    return 0;
}

int check_refusals(const fs::path& dir) {
    bool threw = false;
    try {
        (void)ComponentDump::parse_dtype("f8");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) return fail("unknown dtype accepted");
    auto m = cypha::cyphalm::load_cyphalm_model((dir / "linear.json").string());
    m.hp_backend().set_session_cache(true);
    threw = false;
    try {
        ComponentDump d((dir / "dumps" / "refused").string(), m.hp_backend(), ComponentDump::Dtype::F32, true);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) return fail("a session cache was dumped (it is not replayed)");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path dir = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() / "cyphalm_component_dump_smoke";
    fs::remove_all(dir);
    write_parts(dir);
    int rc = check_first_byte(dir);
    if (rc == 0) rc = check_dumps(dir);
    if (rc == 0) rc = check_refusals(dir);
    if (argc <= 1) fs::remove_all(dir);  // kept for the harness and mixsim tests
    if (rc == 0) std::printf("cyphalm_component_dump_smoke OK\n");
    return rc;
}
