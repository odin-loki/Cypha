/// Speed paths change no output, bit for bit:
///  - the bit tree without the old re-predict schedule (a pruned bit 0 keeps
///    the parent's scratch; with learning off the bit-1 child resumes the
///    saved prediction) equals the old schedule (CYPHA_HP_TREE_REPREDICT=1),
///    frozen and learning, pruned and exact, and the exact frozen tree equals
///    256 independent frozen forks;
///  - a composite model (2 members, ∞-gram, LSTM + Transformer experts with
///    output-layer adaptation, session cache; frozen scoring, pruned tree)
///    on the worker pool serves the same distributions and learns the same
///    weights as the same model on one thread with the old tree schedule,
///    and an hp::StreamRewind lookahead on it (members then read on the
///    calling thread) rewinds exactly;
///  - experts primed on the kept history are not primed again, and the state
///    they keep equals a fresh priming;
///  - the int32-lane mixer dot equals the scalar sum.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "cypha/cyphalm/infinigram.hpp"
#include "cypha/cyphalm/neural_expert.hpp"
#include "hp/predictor.hpp"
#include "hp/simd_dot.hpp"

namespace {

using cypha::cyphalm::HpSequenceBackend;

int fail(const char* what) {
    std::printf("hp_speed_parity_smoke FAIL %s\n", what);
    return 1;
}

void set_reference(bool on) {
#if defined(_WIN32)
    _putenv_s("CYPHA_HP_TREE_REPREDICT", on ? "1" : "");
#else
    if (on) setenv("CYPHA_HP_TREE_REPREDICT", "1", 1);
    else unsetenv("CYPHA_HP_TREE_REPREDICT");
#endif
}

/// Scored with the old tree schedule.
std::vector<double> reference_dist(HpSequenceBackend& hb) {
    set_reference(true);
    auto lp = hb.next_byte_log_probs(256);
    set_reference(false);
    return lp;
}

void write_experts(const std::string& lstm, const std::string& gpt) {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    auto put = [&](std::ofstream& os, std::size_t n, float mean) {
        for (std::size_t i = 0; i < n; ++i) {
            const float f = mean + nd(rng);
            os.write(reinterpret_cast<const char*>(&f), sizeof(f));
        }
    };
    {
        const std::uint32_t layers = 1, d = 16, emb = 8;
        std::ofstream os(lstm, std::ios::binary);
        os.write("BLM1", 4);
        const std::uint32_t hdr[3] = {layers, d, emb};
        os.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        put(os, 256 * emb, 0.0f);
        put(os, 4 * d * emb, 0.0f);
        put(os, 4 * d * d, 0.0f);
        put(os, 4 * d, 0.0f);
        put(os, 4 * d, 0.0f);
        put(os, 256 * d, 0.0f);
        put(os, 256, 0.0f);
    }
    {
        const std::uint32_t gl = 1, gd = 16, gh = 2, gctx = 16;  // re-primes every 8 bytes
        std::ofstream os(gpt, std::ios::binary);
        os.write("BGT1", 4);
        const std::uint32_t hdr[4] = {gl, gd, gh, gctx};
        os.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        put(os, 256 * gd, 0.0f);
        put(os, gctx * gd, 0.0f);
        for (std::uint32_t l = 0; l < gl; ++l) {
            put(os, gd, 1.0f);
            put(os, gd, 0.0f);
            put(os, 3 * gd * gd, 0.0f);
            put(os, gd * gd, 0.0f);
            put(os, gd, 1.0f);
            put(os, gd, 0.0f);
            put(os, 4 * gd * gd, 0.0f);
            put(os, 4 * gd * gd, 0.0f);
        }
        put(os, gd, 1.0f);
        put(os, gd, 0.0f);
    }
}

}  // namespace

int main() {
    // Mixer dot: random and extreme weights / inputs.
    {
        std::mt19937 rng(1);
        for (int t = 0; t < 30000; ++t) {
            const int n = static_cast<int>(rng() % 300);
            std::vector<std::int32_t> w(static_cast<std::size_t>(n));
            std::vector<std::int16_t> st(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                const bool edge = t % 2 == 1;
                w[static_cast<std::size_t>(i)] =
                    edge ? ((rng() & 1) ? hp::kMixerClamp : -hp::kMixerClamp)
                         : static_cast<std::int32_t>(rng() % (2 * hp::kMixerClamp + 1)) - hp::kMixerClamp;
                st[static_cast<std::size_t>(i)] = static_cast<std::int16_t>(
                    edge ? ((t % 4 == 1 || (rng() & 1)) ? 2047 : -2047) : static_cast<int>(rng() % 4095) - 2047);
            }
            std::int64_t ref = 0;
            for (int i = 0; i < n; ++i)
                ref += static_cast<std::int64_t>(w[static_cast<std::size_t>(i)]) * st[static_cast<std::size_t>(i)];
            if (hp::dot_i32_i16(w.data(), st.data(), n) != ref) return fail("mixer dot differs from the scalar sum");
        }
    }

    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.hp_table_bits = 16;
    cfg.hp_cm_bits_cap = 16;
    cfg.hp_match_bits_cap = 16;
    cfg.hp_pool_bits_cap = 16;
    const hp::Config hc = cypha::cyphalm::hp_config_from_cyphalm(cfg);

    std::string text;
    const char* words[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "Then ", "the ", "dog ",
                           "ran ", "[[link]] ", "{{cite}} ", "\n", "1984 ", "Zq "};
    std::mt19937 rng(11);
    while (text.size() < 40000) text += words[rng() % 15];
    auto at = [&](std::size_t i) { return static_cast<std::uint8_t>(text[i]); };

    // 1. Single model: new tree schedule vs old, frozen and learning,
    // pruned and exact; exact frozen tree vs frozen forks.
    {
        HpSequenceBackend hb(hc);
        for (std::size_t i = 0; i < 6000; ++i) hb.consume_byte(at(i));
        int pos = 0;
        for (std::size_t i = 6000; i < 6120; ++i, ++pos) {
            const bool frozen = pos % 2 == 0;
            hb.set_frozen_scoring(frozen);
            hb.set_tree_prune(pos % 3 == 0 ? 0.0 : (pos % 3 == 1 ? 1e-4 : 1e-2));
            const auto lp = hb.next_byte_log_probs(256);
            if (lp != reference_dist(hb)) return fail("bit tree differs from the old schedule");
            if (frozen && pos % 3 == 0 && pos < 12) {
                hb.set_learning(false);
                const auto forks = hb.next_byte_log_probs_assign_reuse(256);
                hb.set_learning(true);
                if (lp != forks) return fail("frozen bit tree differs from 256 frozen forks");
            }
            hb.consume_byte(at(i));
        }
    }

    // 2. Composite model: worker pool + new schedule vs one thread + old.
    const auto dir = std::filesystem::temp_directory_path();
    const std::string lstm = (dir / "hp_speed_parity_smoke.blm").string();
    const std::string gpt = (dir / "hp_speed_parity_smoke.bgt").string();
    write_experts(lstm, gpt);
    const auto ig = std::make_shared<const cypha::cyphalm::InfiniGram>(
        reinterpret_cast<const std::uint8_t*>(text.data()), std::size_t{24000});
    const auto nn_lstm = cypha::cyphalm::load_neural_expert(lstm);
    const auto nn_gpt = cypha::cyphalm::load_neural_expert(gpt);
    auto make = [&](bool parallel) {
        auto hb = std::make_unique<HpSequenceBackend>(hc);
        for (std::size_t i = 0; i < 5000; ++i) hb->consume_byte(at(i));
        for (int k = 0; k < 2; ++k) {
            auto m = std::make_unique<HpSequenceBackend>(hc);
            for (std::size_t i = 0; i < 4000; ++i) m->consume_byte(at(5000 + static_cast<std::size_t>(k) * 4000 + i));
            hb->add_ensemble_member(std::move(m), 0.3);
        }
        hb->set_ensemble_learning_rate(0.01);
        hb->set_infinigram(ig, 0.3);
        hb->set_neural_adaptation(0.002);
        hb->set_neural_learning_rate(0.05);
        hb->add_neural(nn_lstm);
        hb->add_neural(nn_gpt);
        hb->set_session_cache(true, 0.02, 4096);
        hb->set_frozen_scoring(true);
        hb->set_tree_prune(1e-4);
        hb->set_parallel(parallel);
        return hb;
    };
    auto a = make(true);
    auto b = make(false);
    if (!a->parallel() || b->parallel()) return fail("set_parallel");
    for (std::size_t i = 24000; i < 26000; ++i) {
        const bool learn = i % 7 != 0;
        a->set_learning(learn);
        b->set_learning(learn);
        if (i == 25500) {  // a stretch of learning (non-frozen) scoring
            a->set_frozen_scoring(false);
            b->set_frozen_scoring(false);
        }
        const auto la = a->next_byte_log_probs(256);
        const auto lb = reference_dist(*b);
        if (la != lb) return fail("composite distribution: pool + new tree differs from one thread + old tree");
        const double oa = a->observe_next_byte(at(i));
        const double ob = b->observe_next_byte(at(i));
        if (oa != ob) return fail("composite observe");
    }
    a->set_learning(true);
    b->set_learning(true);
    if (a->ensemble_weights() != b->ensemble_weights() || a->infinigram_weights() != b->infinigram_weights() ||
        a->neural_weights() != b->neural_weights() || a->mixing_state() != b->mixing_state() ||
        a->session_size() != b->session_size())
        return fail("composite learned weights differ");
    {
        const auto sa = a->neural_states(), sb = b->neural_states();
        for (std::size_t k = 0; k < sa.size(); ++k)
            if (sa[k].log_p != sb[k].log_p || sa[k].a != sb[k].a || sa[k].ow != sb[k].ow)
                return fail("composite expert states differ");
    }
    // Lookahead under hp::StreamRewind: members read on the calling thread
    // (the recorder is per thread), experts on the pool; exact rewind.
    for (auto* m : {a.get(), b.get()}) m->set_learning(false);
    {
        const auto before = a->next_byte_log_probs(256);
        const auto states = a->neural_states();
        const std::size_t sess = a->session_size();
        const auto mix = a->mixing_state();
        // One rewind at a time: recorders nest per thread, the newest records.
        std::vector<std::vector<double>> ahead;
        {
            hp::StreamRewind rwa(a->all_predictors());
            for (std::size_t i = 26000; i < 26040; ++i) {
                ahead.push_back(a->next_byte_log_probs(256));
                a->serve_advance_byte(at(i));
            }
            rwa.rewind();
        }
        {
            hp::StreamRewind rwb(b->all_predictors());
            for (std::size_t i = 26000; i < 26040; ++i) {
                if (b->next_byte_log_probs(256) != ahead[i - 26000]) return fail("lookahead distribution differs");
                b->serve_advance_byte(at(i));
            }
        }
        a->set_neural_states(states);
        a->truncate_session(sess);
        a->invalidate_scoring_cache();
        if (a->next_byte_log_probs(256) != before || a->mixing_state() != mix)
            return fail("StreamRewind lookahead on the worker pool did not rewind exactly");
    }

    // 3. Priming: kept experts are not primed again; a needed prime equals a fresh one.
    {
        HpSequenceBackend e(hc), fresh(hc);
        for (std::size_t i = 0; i < 3000; ++i) {
            e.consume_byte(at(i));
            fresh.consume_byte(at(i));
        }
        e.set_neural_adaptation(0.002);
        e.add_neural(nn_lstm);
        e.add_neural(nn_gpt);
        if (e.neural_primes() != 2) return fail("each expert should be primed once as it attaches");
        const auto primed = e.neural_states();
        e.reset_stream(/*keep_history=*/true);
        e.set_neural_adaptation(0.002);
        if (e.neural_primes() != 2) return fail("experts holding the kept history were primed again");
        for (std::size_t i = 3000; i < 3100; ++i) e.consume_byte(at(i));
        e.reset_stream(/*keep_history=*/true);  // they read 100 bytes: prime again
        if (e.neural_primes() != 4) return fail("experts that read bytes were not primed again");
        for (std::size_t i = 3000; i < 3100; ++i) fresh.consume_byte(at(i));
        fresh.reset_stream(/*keep_history=*/true);
        fresh.set_neural_adaptation(0.002);
        fresh.add_neural(nn_lstm);
        fresh.add_neural(nn_gpt);
        const auto se = e.neural_states(), sf = fresh.neural_states();
        for (std::size_t k = 0; k < se.size(); ++k)
            if (se[k].log_p != sf[k].log_p || se[k].a != sf[k].a || se[k].b != sf[k].b || se[k].ow != sf[k].ow)
                return fail("re-primed experts differ from freshly primed ones");
        if (primed[0].log_p == se[0].log_p) return fail("control: priming on other bytes changed nothing");
        e.set_neural_adaptation(0.0);  // another setting: primed again
        if (e.neural_primes() != 6) return fail("a new adaptation setting did not re-prime");
    }
    std::filesystem::remove(lstm);
    std::filesystem::remove(gpt);
    std::printf("hp_speed_parity_smoke OK tree schedule, pool, rewind, priming and mixer dot bit-identical\n");
    return 0;
}
