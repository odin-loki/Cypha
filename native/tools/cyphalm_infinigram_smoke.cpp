/// ∞-gram index: the suffix array is sorted and complete, and query() returns
/// the longest context suffix that occurs plus exact next-byte counts
/// (checked against brute force on a small random corpus), with or without
/// a length hint.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "cypha/cyphalm/infinigram.hpp"
#include "hp/checkpoint.hpp"
#include <cmath>
#include <sstream>

int main() {
    std::mt19937 rng(9);
    const char* words[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "the cat ", "abab", "\n"};
    std::string text;
    while (text.size() < 20000) text += words[rng() % 9];
    text += std::string(1, static_cast<char>(0xff));  // a byte seen once, at the end
    const auto path = (std::filesystem::temp_directory_path() / "cyphalm_infinigram_smoke.igr").string();
    cypha::cyphalm::InfiniGram::build(reinterpret_cast<const std::uint8_t*>(text.data()), text.size(), path);
    cypha::cyphalm::InfiniGram ig(path);
    const std::size_t n = text.size();
    // The same index built just in time from the plain corpus (plus a tail
    // that max_bytes must leave out).
    const auto txt = (std::filesystem::temp_directory_path() / "cyphalm_infinigram_smoke.txt").string();
    {
        std::ofstream(txt, std::ios::binary) << text << "HELD-OUT TAIL";
    }
    const auto jit = cypha::cyphalm::InfiniGram::open(txt, n);
    std::filesystem::remove(txt);
    if (jit->size() != n) {
        std::printf("cyphalm_infinigram_smoke FAIL: just-in-time index has %zu bytes\n", jit->size());
        return 1;
    }

    for (int trial = 0; trial < 300; ++trial) {
        // Context: a random corpus slice, sometimes with a novel byte mixed in.
        const std::size_t p = rng() % (n - 40);
        std::string ctx = text.substr(p, 1 + rng() % 30);
        if (trial % 4 == 0) ctx[rng() % ctx.size()] = 'Z';
        const auto r = ig.query(reinterpret_cast<const std::uint8_t*>(ctx.data()), ctx.size(), 64);
        // Brute force: longest suffix with a following byte, and its counts.
        int best = 0;
        std::vector<std::uint32_t> cnt(256, 0);
        for (int m = static_cast<int>(ctx.size()); m >= 0; --m) {
            std::vector<std::uint32_t> c(256, 0);
            std::uint64_t tot = 0;
            const std::string suf = ctx.substr(ctx.size() - static_cast<std::size_t>(m));
            for (std::size_t i = 0; i + static_cast<std::size_t>(m) < n; ++i) {
                if (m == 0 || text.compare(i, static_cast<std::size_t>(m), suf) == 0) {
                    ++c[static_cast<unsigned char>(text[i + static_cast<std::size_t>(m)])];
                    ++tot;
                }
            }
            if (tot > 0) {
                best = m;
                cnt = c;
                break;
            }
        }
        const auto rj = jit->query(reinterpret_cast<const std::uint8_t*>(ctx.data()), ctx.size(), 64);
        bool same = r.n == best && rj.n == best && rj.total == r.total;
        for (int b = 0; b < 256 && same; ++b) same = r.count[b] == cnt[b] && rj.count[b] == cnt[b];
        if (!same) {
            std::printf("cyphalm_infinigram_smoke FAIL trial %d: n %d vs %d\n", trial, r.n, best);
            return 1;
        }
    }
    // Hints give the unhinted result: along a stream (a sliding 64-byte
    // context, novel bytes mixed in) the previous byte's n + 1, and in a
    // backoff a length known to occur.
    {
        std::string stream = text.substr(3000, 3000);
        for (std::size_t i = 97; i < stream.size(); i += 211) stream[i] = 'Z';
        auto same = [](const cypha::cyphalm::InfiniGram::Result& x, const cypha::cyphalm::InfiniGram::Result& y) {
            return x.n == y.n && x.total == y.total && x.count == y.count;
        };
        int prev = 0;  // the empty context matches n = 0
        for (std::size_t t = 1; t <= stream.size(); ++t) {
            const std::size_t len = std::min<std::size_t>(t, 64);
            const auto* c = reinterpret_cast<const std::uint8_t*>(stream.data()) + (t - len);
            const auto plain = ig.query(c, len, 64);
            if (!same(ig.query(c, len, 64, prev + 1), plain)) {
                std::printf("cyphalm_infinigram_smoke FAIL hinted query differs at %zu\n", t);
                return 1;
            }
            for (int m = plain.n / 2; m > 0; m /= 2) {
                if (!same(ig.query(c, len, m, m), ig.query(c, len, m))) {
                    std::printf("cyphalm_infinigram_smoke FAIL backoff query differs at %zu (m %d)\n", t, m);
                    return 1;
                }
            }
            prev = plain.n;
        }
    }
    // As a model expert: normalised, observe == -log p of the served mix,
    // exact rewinds over it, and generation runs.
    {
        cypha::cyphalm::CyphaLMConfig cfg;
        cfg.hp_table_bits = 16;
        cypha::cyphalm::apply_hp_production_recipe(cfg);
        cfg.hp_cm_bits_cap = 16;
        cfg.hp_match_bits_cap = 16;
        cfg.hp_pool_bits_cap = 16;
        cypha::cyphalm::CyphaLMModel model(cfg);
        for (std::size_t i = 0; i < 8000; ++i) model.hp_backend().consume_byte(static_cast<std::uint8_t>(text[i]));
        model.attach_infinigram(path);
        auto& h = model.hp_backend();
        for (std::size_t i = 8000; i < 8400; ++i) {
            const auto lp = h.serve_next_byte_log_probs(256);
            double z = 0.0;
            for (double v : lp) z += std::exp(v);
            const auto c = static_cast<std::uint8_t>(text[i]);
            const double loss = h.observe_next_byte(c);
            if (std::abs(z - 1.0) > 1e-9 || std::abs(loss + lp[c]) > 1e-12) {
                std::printf("cyphalm_infinigram_smoke FAIL expert: sum %.12f loss %.6f vs %.6f\n", z, loss, -lp[c]);
                return 1;
            }
        }
        h.set_learning(false);
        const auto before = h.serve_next_byte_log_probs(256);
        {
            hp::StreamRewind rw(h.all_predictors());
            for (int k = 0; k < 30; ++k) {
                (void)h.serve_next_byte_log_probs(256);
                h.serve_advance_byte(static_cast<std::uint8_t>(text[9000 + static_cast<std::size_t>(k)]));
            }
            rw.rewind();
            h.invalidate_scoring_cache();
        }
        if (h.serve_next_byte_log_probs(256) != before) {
            std::printf("cyphalm_infinigram_smoke FAIL expert distribution changed after rewind\n");
            return 1;
        }
        h.set_learning(true);
        cypha::cyphalm::DecodeParams p;
        p.word_candidates = 4;
        const auto g = cypha::cyphalm::generate_decode(model, {'t', 'h', 'e', ' '}, 40, p);
        if (g.generated_ids.size() != 40) {
            std::printf("cyphalm_infinigram_smoke FAIL generation returned %zu bytes\n", g.generated_ids.size());
            return 1;
        }
    }
    std::filesystem::remove(path);
    std::printf("cyphalm_infinigram_smoke OK 300 queries match brute force; hints exact; expert normalised, rewinds, generates\n");
    return 0;
}
