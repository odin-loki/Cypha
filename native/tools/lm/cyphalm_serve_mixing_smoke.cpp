/// Serve-time mixing and beam search on a manifest-style composite model (two
/// shard checkpoints, ∞-gram over a corpus, an adapting Transformer expert,
/// session cache):
///  - Prompt scoring (DecodeParams::prompt_score_bytes): without it a request
///    leaves every mixing weight at its start value; with it they adapt to
///    the prompt while the learned tables end exactly as without it; with
///    restore_mixing (default) the weights are back after the request.
///  - set_mixing_learning(false) keeps the weights while the models learn.
///  - generate_beam ranks on the full served distribution: at width 1 its
///    first byte is the argmax of the distribution served after the prompt
///    (for some prompts not the primary's argmax), and it decodes exactly as
///    greedy does, losses included.
///  - Beam search learns nothing but the prompt with learn_from_output off;
///    with it on, the tables learn the output as observing it does.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

namespace fs = std::filesystem;
using cypha::cyphalm::CyphaLMModel;
using cypha::cyphalm::DecodeParams;
using cypha::cyphalm::DecodeStrategy;
using cypha::cyphalm::HpSequenceBackend;

int fail(const char* what) {
    std::printf("cyphalm_serve_mixing_smoke FAIL: %s\n", what);
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

std::vector<int> ids(const std::string& s) {
    std::vector<int> out;
    for (unsigned char c : s) out.push_back(c);
    return out;
}

/// A small random Transformer (BGT1), as in cyphalm_neural_smoke.
void write_bgt(const fs::path& path) {
    std::mt19937 rng(5);
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

/// Two shards (cat text, insect text), an ∞-gram over insect text, an
/// adapting Transformer and the session cache, as a manifest.
fs::path write_manifest(const fs::path& dir) {
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
        os << corpus(3, 1, 20000);
    }
    write_bgt(dir / "expert.bgt");
    const nlohmann::json meta = {{"cyphalm_ensemble", 1},
                                 {"members", {{{"checkpoint", "shard_a.json"}}, {{"checkpoint", "shard_b.json"}}}},
                                 {"learning_rate", 0.05},
                                 {"infinigram", "corpus.txt"},
                                 {"neural", {"expert.bgt"}},
                                 {"neural_learning_rate", 0.1},
                                 {"neural_adapt", 0.01},
                                 {"session_cache", true}};
    const fs::path manifest = dir / "composite.json";
    std::ofstream(manifest) << meta.dump(1);
    return manifest;
}

std::vector<std::uint64_t> digests(CyphaLMModel& m) {
    std::vector<std::uint64_t> out;
    for (auto* p : m.hp_backend().all_predictors()) out.push_back(p->learned_digest());
    return out;
}

/// What generation does before the first byte with the default prompt
/// scoring (prompt <= 512 bytes): new stream, serve rate, every prompt byte
/// but the last scored and learned, the position scored, the last byte
/// learned; then learning off.
void prime_like_generation(CyphaLMModel& m, const std::vector<int>& prompt) {
    auto& h = m.hp_backend();
    m.reset_stream(/*keep_history=*/true);
    m.set_serve_mode(true);
    for (std::size_t i = 0; i + 1 < prompt.size(); ++i) (void)m.serve_observe(static_cast<std::uint32_t>(prompt[i]));
    (void)h.serve_next_byte_log_probs(256);
    m.serve_advance(static_cast<std::uint32_t>(prompt.back()));
    h.set_learning(false);
}

DecodeParams greedy_params() {
    DecodeParams p;
    p.strategy = DecodeStrategy::Greedy;
    p.temperature = 0.0;
    p.word_candidates = 0;
    return p;
}

int check_prompt_scoring(const fs::path& manifest) {
    const std::vector<int> prompt = ids(corpus(7, 1, 700));
    auto run = [&](int score_bytes, bool restore, CyphaLMModel& m) {
        DecodeParams p = greedy_params();
        p.prompt_score_bytes = score_bytes;
        p.restore_mixing = restore;
        return cypha::cyphalm::generate_decode(m, prompt, 8, p);
    };
    auto x0 = cypha::cyphalm::load_cyphalm_model(manifest.string());
    if (!x0.hp_backend().is_composite()) return fail("manifest model is not composite");
    const auto start = x0.hp_backend().mixing_state();
    if (start.ensemble.size() != 2 || start.ig.empty() || start.neural.empty() || start.session.empty())
        return fail("mixing state misses a stage");
    (void)run(0, false, x0);
    if (x0.hp_backend().mixing_state() != start) return fail("without prompt scoring the weights moved");

    auto x1 = cypha::cyphalm::load_cyphalm_model(manifest.string());
    (void)run(512, false, x1);
    const auto adapted = x1.hp_backend().mixing_state();
    if (adapted.ensemble == start.ensemble || adapted.ig == start.ig || adapted.neural == start.neural)
        return fail("prompt scoring did not adapt the ensemble, ∞-gram and neural weights");
    if (digests(x1) != digests(x0)) return fail("prompt scoring changed what the tables learned");

    auto x2 = cypha::cyphalm::load_cyphalm_model(manifest.string());
    (void)run(512, true, x2);
    if (x2.hp_backend().mixing_state() != start) return fail("restore_mixing did not restore the weights");
    x1.hp_backend().set_mixing_state(start);
    if (x1.hp_backend().mixing_state() != start) return fail("set_mixing_state does not round-trip");

    // Frozen mixing: the weights stay, the models still learn.
    auto x3 = cypha::cyphalm::load_cyphalm_model(manifest.string());
    auto& h3 = x3.hp_backend();
    h3.set_mixing_learning(false);
    const auto before = digests(x3);
    for (int b : prompt) (void)h3.observe_next_byte(static_cast<std::uint8_t>(b));
    if (h3.mixing_state() != start) return fail("frozen mixing weights moved");
    if (digests(x3) == before) return fail("models stopped learning with frozen mixing");
    return 0;
}

int check_beam(const fs::path& manifest) {
    // Width 1: the served argmax first, then greedy's exact path.
    const std::string prompts[] = {"insects have six legs and wings. The body ", "the cat sat on a mat. The ",
                                   "Then dog ran on a mat. insects have six ", "insects have six legs. Then dog "};
    int differ = 0;
    for (const std::string& ps : prompts) {
        const std::vector<int> prompt = ids(ps);
        DecodeParams bp = greedy_params();
        bp.strategy = DecodeStrategy::Beam;
        bp.beam_width = 1;
        auto mb = cypha::cyphalm::load_cyphalm_model(manifest.string());
        const auto beam = cypha::cyphalm::generate_beam(mb, prompt, 12, bp);
        auto mg = cypha::cyphalm::load_cyphalm_model(manifest.string());
        const auto greedy = cypha::cyphalm::generate_decode(mg, prompt, 12, greedy_params());
        auto twin = cypha::cyphalm::load_cyphalm_model(manifest.string());
        prime_like_generation(twin, prompt);
        const auto lp = twin.hp_backend().serve_next_byte_log_probs(256);
        const int served = static_cast<int>(std::max_element(lp.begin(), lp.end()) - lp.begin());
        const auto own = HpSequenceBackend::byte_log_probs_bit_tree(twin.hp_backend().predictor(), 256);
        const int primary = static_cast<int>(std::max_element(own.begin(), own.end()) - own.begin());
        if (beam.generated_ids.size() != 12 || beam.per_step.size() != 12) return fail("beam width 1: wrong length");
        if (beam.generated_ids[0] != served) {
            std::printf("  prompt \"%s\": beam %d, served argmax %d, primary %d\n", ps.c_str(),
                        beam.generated_ids[0], served, primary);
            return fail("beam width 1 does not start with the served argmax");
        }
        if (beam.per_step[0].loss != -lp[static_cast<std::size_t>(served)])
            return fail("beam loss is not the served distribution's");
        if (beam.generated_ids != greedy.generated_ids) return fail("beam width 1 differs from greedy");
        for (std::size_t i = 0; i < 12; ++i) {
            if (beam.per_step[i].loss != greedy.per_step[i].loss) return fail("beam width 1 losses differ from greedy");
        }
        if (primary != served) ++differ;
    }
    if (differ == 0) return fail("no prompt where the primary's argmax differs from the served one");

    const std::vector<int> prompt = ids(corpus(9, 1, 300));
    for (bool learn : {false, true}) {
        DecodeParams bp = greedy_params();
        bp.strategy = DecodeStrategy::Beam;
        bp.beam_width = 3;
        bp.learn_from_output = learn;
        auto m = cypha::cyphalm::load_cyphalm_model(manifest.string());
        const auto start = m.hp_backend().mixing_state();
        const auto g = cypha::cyphalm::generate_beam(m, prompt, 20, bp);
        if (g.generated_ids.size() != 20 || g.per_step.size() != 20) return fail("beam width 3: wrong length");
        if (m.hp_backend().mixing_state() != start) return fail("beam request leaked its mixing weights");
        if (!m.hp_backend().predictor().learning()) return fail("beam left learning off");
        // The twin reads the prompt as generation does, then the output:
        // context only, or observed with learning on.
        auto twin = cypha::cyphalm::load_cyphalm_model(manifest.string());
        prime_like_generation(twin, prompt);
        twin.hp_backend().set_learning(learn);
        for (int b : g.generated_ids) {
            if (learn) (void)twin.serve_observe(static_cast<std::uint32_t>(b));
            else twin.serve_advance(static_cast<std::uint32_t>(b));
        }
        if (digests(m) != digests(twin)) {
            return fail(learn ? "beam learn_from_output: tables differ from observing the output"
                              : "beam changed learned tables beyond the prompt");
        }
        if (!learn) {
            // Replay the beam on a fresh model: each recorded loss is the
            // served distribution's, not a later rescore.
            auto replay = cypha::cyphalm::load_cyphalm_model(manifest.string());
            prime_like_generation(replay, prompt);
            for (std::size_t i = 0; i < g.generated_ids.size(); ++i) {
                const auto lp = replay.hp_backend().serve_next_byte_log_probs(256);
                const auto b = static_cast<std::size_t>(g.generated_ids[i]);
                if (g.per_step[i].loss != -lp[b]) return fail("beam replay loss is not the served distribution's");
                replay.serve_advance(static_cast<std::uint32_t>(g.generated_ids[i]));
            }
        }
    }
    return 0;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "cyphalm_serve_mixing_smoke";
    const fs::path manifest = write_manifest(dir);
    const int rc = check_prompt_scoring(manifest) != 0 ? 1 : check_beam(manifest);
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (rc != 0) return rc;
    std::printf("cyphalm_serve_mixing_smoke OK: prompt scoring adapts and restores; beam serves the full mix\n");
    return 0;
}
