/// Serving state:
///  - hp::MixerNet serve-time rate scale runs in 1/16ths: x1 keeps the
///    trained rates exactly, fractions act on rate-1 weight sets (lr1_scale
///    40 shards, where x0.5 used to floor to x1), checkpoints keep the
///    trained rates. On a predictor x1 equals no scaling bit for bit, and
///    x0.5 differs from x0.25 (both were x1 on layer 1 before).
///  - Session cache: the held text is bounded by the window, rebuilds double
///    (1, 2, 4 ... 64 KiB by default), bytes read under hp::StreamRewind
///    (lookahead) never trigger a rebuild and rewind to the same
///    distribution, and reset_stream(false) starts an empty session.
///  - HpSequenceBackend::reset: a model with ∞-gram, an adapting neural
///    expert and the session cache scores after reset exactly like a fresh
///    twin, whatever it read before.
///  - Top-p sampling at a very low temperature picks the argmax, like greedy
///    (exp(lp / T) used to underflow and pick the least likely byte).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "hp/mixer.hpp"
#include "hp/predictor.hpp"

namespace {

int fail(const char* what) {
    std::printf("cyphalm_serve_state_smoke FAIL: %s\n", what);
    return 1;
}

/// Wiki-like text with repeats: words drawn from a small vocabulary.
std::string make_text(std::size_t n, unsigned seed) {
    static const char* words[] = {"the", "cat", "sat", "on", "mat", "[[Link]]", "of", "a", "1984", "river",
                                  "and", "city", "was", "in", "north", "is", "known", "for", "its", "bridge"};
    std::mt19937 rng(seed);
    std::string s;
    while (s.size() < n) {
        s += words[rng() % 20];
        s += (rng() % 9 == 0) ? ".\n" : " ";
    }
    s.resize(n);
    return s;
}

int check_mixer_rates() {
    hp::MixerNet m(4, {2, 2}, 4, 2, {1, 2});
    if (m.rate_q4() != 32 || m.layer1_rate_q4(0) != 16 || m.layer1_rate_q4(1) != 32)
        return fail("trained rates are not 16x in Q4");
    m.set_rate_scale(8, 16, -1);  // x0.5
    if (m.rate_q4() != 16 || m.layer1_rate_q4(0) != 8 || m.layer1_rate_q4(1) != 16)
        return fail("x0.5 does not halve rate-1 sets");
    m.set_rate_scale(1, 64, -1);  // below 1/16: floored there
    if (m.rate_q4() != 1 || m.layer1_rate_q4(0) != 1) return fail("rates below 1/16 not floored at 1/16");
    m.set_rate_scale(8, 16, -1);
    std::stringstream ss;
    m.checkpoint_write(ss);
    hp::MixerNet r(4, {2, 2}, 4, 7, {5, 5});
    r.checkpoint_read(ss);
    if (!ss || r.rate_q4() != 32 || r.layer1_rate_q4(0) != 16 || r.layer1_rate_q4(1) != 32)
        return fail("checkpoint does not carry the trained rates");
    m.set_rate_scale(1, 1, -1);
    if (m.rate_q4() != 32 || m.layer1_rate_q4(0) != 16) return fail("(1, 1) does not restore the trained rates");

    // Predictor with the upstream layer-1 rates (lr1_scale 40: rates 1-2).
    hp::Config pc;
    pc.table_bits = 16;
    pc.lr1_scale = 40;
    pc.normalize();
    const std::string text = make_text(24000, 1);
    auto run = [&](int num, int den) {
        auto p = std::make_unique<hp::Predictor>(pc);
        if (num > 0) p->set_serve_adaptation(num, den, -1);
        std::uint64_t acc = 0;
        for (unsigned char c : text) {
            for (int i = 7; i >= 0; --i) {
                acc = acc * 1000003u + static_cast<std::uint64_t>(p->predict());
                p->update((c >> i) & 1);
            }
        }
        return std::make_pair(acc, p->learned_digest());
    };
    const auto base = run(0, 0), x1 = run(16, 16), half = run(8, 16), quarter = run(4, 16);
    if (base != x1) return fail("serve scale x1 is not bit-identical to the trained rates");
    if (half == quarter) return fail("x0.5 and x0.25 give the same predictor (rates floored)");
    if (half == x1) return fail("x0.5 changes nothing");
    return 0;
}

int check_session_cache() {
    hp::Config pc;
    pc.table_bits = 16;
    pc.normalize();
    const std::string text = make_text(70000, 2);
    {
        // Default window: doubling cadence, 1 KiB ... 64 KiB.
        cypha::cyphalm::HpSequenceBackend hb(pc);
        hb.set_session_cache(true);
        for (unsigned char c : text) hb.consume_byte(c);
        if (hb.session_size() != text.size() || hb.session_held() != text.size() ||
            hb.session_builds() != 7 || hb.session_indexed() != 65536)
            return fail("default session cadence is not 1, 2, 4 ... 64 KiB");
    }
    cypha::cyphalm::HpSequenceBackend hb(pc);
    constexpr std::size_t kWin = 4096;
    hb.set_session_cache(true, 0.02, kWin);
    for (std::size_t i = 0; i < 20000; ++i) hb.consume_byte(static_cast<std::uint8_t>(text[i]));
    // Builds every window / 16 = 256 bytes from 1 KiB on: 1024 + 256 k <= 20000.
    if (hb.session_size() != 20000 || hb.session_held() > kWin + kWin / 16 || hb.session_indexed() != kWin ||
        hb.session_builds() != 75)
        return fail("session text / index not bounded by the window");

    // Lookahead: bytes read under a StreamRewind never rebuild the index,
    // and rewinding returns the same distribution.
    hb.set_learning(false);
    const std::size_t builds0 = hb.session_builds();
    const auto lp0 = hb.next_byte_log_probs(256);
    const std::size_t saved = hb.session_size();
    {
        hp::StreamRewind rw(hb.all_predictors());
        for (std::size_t i = 0; i < 1200; ++i) {
            (void)hb.next_byte_log_probs(256);
            hb.consume_byte(static_cast<std::uint8_t>(text[30000 + i]));
        }
        if (hb.session_builds() != builds0) return fail("session index rebuilt on lookahead bytes");
        rw.rewind();
        hb.truncate_session(saved);
        hb.invalidate_scoring_cache();
    }
    if (hb.session_size() != saved || hb.session_indexed() != kWin || hb.session_builds() != builds0)
        return fail("truncating the lookahead dropped the session index");
    const auto lp1 = hb.next_byte_log_probs(256);
    if (std::memcmp(lp0.data(), lp1.data(), sizeof(double) * 256) != 0)
        return fail("distribution differs after rewinding the lookahead");
    // Committed bytes rebuild again once due.
    for (std::size_t i = 0; i < 300; ++i) hb.consume_byte(static_cast<std::uint8_t>(text[40000 + i]));
    if (hb.session_builds() != builds0 + 1) return fail("committed bytes do not rebuild the session index");

    hb.reset_stream(/*keep_history=*/false);
    if (hb.session_size() != 0 || hb.session_held() != 0 || hb.session_indexed() != 0)
        return fail("reset_stream(false) keeps the session");

    cypha::cyphalm::HpSequenceBackend all(pc);
    all.set_session_cache(true, 0.02, 0);  // 0: no window
    for (std::size_t i = 0; i < 5000; ++i) all.consume_byte(static_cast<std::uint8_t>(text[i]));
    if (all.session_held() != 5000 || all.session_builds() != 3 || all.session_indexed() != 4096)
        return fail("window 0 does not keep the whole session");
    return 0;
}

/// A small random Transformer (BGT1), as in cyphalm_neural_smoke.
void write_bgt(const std::filesystem::path& path) {
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

cypha::cyphalm::CyphaLMConfig small_config() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.hp_table_bits = 16;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    return cfg;
}

int check_reset() {
    const auto tmp = std::filesystem::temp_directory_path();
    const auto bgt = tmp / "cyphalm_serve_state_smoke.bgt";
    const auto corpus = tmp / "cyphalm_serve_state_smoke.txt";
    write_bgt(bgt);
    {
        std::ofstream os(corpus, std::ios::binary);
        os << make_text(30000, 3);
    }
    auto build = [&] {
        auto m = std::make_unique<cypha::cyphalm::CyphaLMModel>(small_config());
        m->attach_infinigram(corpus.string());
        m->attach_neural(bgt.string());
        m->hp_backend().set_neural_adaptation(0.01);
        m->hp_backend().set_session_cache(true, 0.02, 4096);
        return m;
    };
    auto a = build();
    const std::string t1 = make_text(1500, 4), t2 = make_text(1500, 5);
    for (unsigned char c : t1) a->hp_backend().observe_next_byte(c);
    a->reset_context();  // HpSequenceBackend::reset
    auto b = build();
    auto& ha = a->hp_backend();
    auto& hb = b->hp_backend();
    if (ha.session_size() != 0) return fail("reset keeps the session");
    for (unsigned char c : t2) {
        const double la = ha.observe_next_byte(c), lb = hb.observe_next_byte(c);
        if (la != lb) {
            std::printf("  reset model %.12f vs fresh twin %.12f\n", la, lb);
            return fail("reset model does not score like a fresh one");
        }
    }
    if (ha.neural_weights() != hb.neural_weights() || ha.infinigram_weights() != hb.infinigram_weights())
        return fail("mixing weights differ from a fresh twin after reset");
    std::filesystem::remove(bgt);
    std::filesystem::remove(corpus);
    return 0;
}

int check_top_p_low_temperature() {
    const std::string train = make_text(20000, 6);
    auto trained = [&] {
        auto m = std::make_unique<cypha::cyphalm::CyphaLMModel>(small_config());
        for (unsigned char c : train) m->hp_backend().observe_next_byte(c);
        return m;
    };
    auto g = trained();
    auto p = trained();
    std::vector<int> prompt;
    for (unsigned char c : std::string("the cat sat on the ")) prompt.push_back(c);
    cypha::cyphalm::DecodeParams gp;
    gp.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
    gp.temperature = 0.0;
    gp.word_candidates = 0;
    gp.min_p = 0.0;
    cypha::cyphalm::DecodeParams pp = gp;
    pp.strategy = cypha::cyphalm::DecodeStrategy::TopP;
    pp.temperature = 1e-5;
    pp.top_p = 0.9;
    const auto og = cypha::cyphalm::generate_decode(*g, prompt, 40, gp);
    const auto op = cypha::cyphalm::generate_decode(*p, prompt, 40, pp);
    if (og.generated_ids.size() != 40 || og.generated_ids != op.generated_ids)
        return fail("top-p at T 1e-5 does not pick the argmax");
    return 0;
}

}  // namespace

int main() {
    if (check_mixer_rates() != 0) return 1;
    if (check_session_cache() != 0) return 1;
    if (check_reset() != 0) return 1;
    if (check_top_p_low_temperature() != 0) return 1;
    std::printf("cyphalm_serve_state_smoke OK: Q4 serve rates; bounded session cache; reset; low-T top-p\n");
    return 0;
}
