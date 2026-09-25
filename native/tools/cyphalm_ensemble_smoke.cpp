/// Serve-time ensemble (CyphaLMModel::add_ensemble_member): the mixed
/// distribution is the normalised weighted geometric mean of the members',
/// members advance in step, and word-lookahead generation (exact rewinds over
/// every model) changes nothing learned. log_prob_byte / greedy / sampling
/// read the full mix and reuse the served distribution without changing what
/// is learned.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

cypha::cyphalm::CyphaLMConfig small_cfg() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.hp_table_bits = 16;
    cfg.hp_cm_bits_cap = 16;
    cfg.hp_match_bits_cap = 16;
    cfg.hp_pool_bits_cap = 16;
    return cfg;
}

std::string corpus(unsigned seed, int variant) {
    const char* a[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "Then ", "dog ", "ran ", "\n"};
    const char* b[] = {"insects ", "have ", "six ", "legs ", "and ", "wings. ", "The ", "body ", "\n"};
    std::mt19937 rng(seed);
    std::string t;
    while (t.size() < 6000) t += variant == 0 ? a[rng() % 10] : b[rng() % 9];
    return t;
}

void train(cypha::cyphalm::CyphaLMModel& m, const std::string& t) {
    for (char c : t) m.hp_backend().consume_byte(static_cast<std::uint8_t>(c));
}

double rng_fixed() { return 0.37; }

int argmax(const std::vector<double>& lp) {
    return static_cast<int>(std::max_element(lp.begin(), lp.end()) - lp.begin());
}

/// The byte serve_sample_next_byte draws at temperature 1 for uniform ``r``.
int draw(const std::vector<double>& lp, double r) {
    double mx = -1e300, z = 0.0;
    for (double v : lp) mx = std::max(mx, v);
    std::vector<double> w(lp.size());
    for (std::size_t b = 0; b < lp.size(); ++b) z += (w[b] = std::exp(lp[b] - mx));
    double left = r * z;
    for (std::size_t b = 0; b < w.size(); ++b) {
        left -= w[b];
        if (left <= 0.0) return static_cast<int>(b);
    }
    return static_cast<int>(w.size()) - 1;
}

}  // namespace

int main() {
    const auto cfg = small_cfg();
    const std::string ta = corpus(1, 0), tb = corpus(2, 1);
    const double w = 0.4;

    cypha::cyphalm::CyphaLMModel a(cfg), b(cfg), ens(cfg), mem(cfg);
    train(a, ta);
    train(b, tb);
    train(ens, ta);
    train(mem, tb);
    ens.add_ensemble_member(std::move(mem), w);
    ens.hp_backend().set_ensemble_learning_rate(0.0);  // fixed weights for the exact check

    // Mixed distribution == normalised a^(1-w) b^w, over 300 shared bytes.
    const std::string probe = "the insects sat on a mat. The body has six legs and the dog ran.\n";
    for (int step = 0; step < 300; ++step) {
        const auto la = a.hp_backend().serve_next_byte_log_probs(256);
        const auto lb = b.hp_backend().serve_next_byte_log_probs(256);
        const auto le = ens.hp_backend().serve_next_byte_log_probs(256);
        double z = 0.0, sum_e = 0.0, max_err = 0.0;
        for (int i = 0; i < 256; ++i) z += std::exp((1 - w) * la[i] + w * lb[i]);
        for (int i = 0; i < 256; ++i) {
            const double want = (1 - w) * la[i] + w * lb[i] - std::log(z);
            max_err = std::max(max_err, std::abs(want - le[i]));
            sum_e += std::exp(le[i]);
        }
        if (max_err > 1e-9 || std::abs(sum_e - 1.0) > 1e-9) {
            std::printf("cyphalm_ensemble_smoke FAIL mix at step %d: err %.3g sum %.12f\n", step, max_err, sum_e);
            return 1;
        }
        const auto c = static_cast<std::uint8_t>(probe[static_cast<std::size_t>(step) % probe.size()]);
        a.hp_backend().serve_advance_byte(c);
        b.hp_backend().serve_advance_byte(c);
        ens.hp_backend().serve_advance_byte(c);
    }

    // Word lookahead on the ensemble: every model's learned state unchanged
    // except by the one learned prompt byte, i.e. same as its twins' digests.
    const auto preds = ens.hp_backend().all_predictors();
    if (preds.size() != 2) {
        std::printf("cyphalm_ensemble_smoke FAIL expected 2 predictors, got %zu\n", preds.size());
        return 1;
    }
    cypha::cyphalm::DecodeParams p;
    p.word_candidates = 4;
    const auto g = cypha::cyphalm::generate_decode(ens, {'t'}, 60, p);
    for (auto* twin : {&a, &b}) {
        twin->reset_stream(/*keep_history=*/true);
        twin->set_serve_mode(true);
        twin->serve_advance('t');
    }
    if (g.generated_ids.size() != 60 || preds[0]->learned_digest() != a.hp_backend().predictor().learned_digest() ||
        preds[1]->learned_digest() != b.hp_backend().predictor().learned_digest()) {
        std::printf("cyphalm_ensemble_smoke FAIL word lookahead: %zu bytes or learned state changed\n",
                    g.generated_ids.size());
        return 1;
    }
    // Online weights: observe returns the mix's log p, weights stay a
    // distribution, and on text from corpus B the B-trained member gains weight.
    {
        cypha::cyphalm::CyphaLMModel e2(cfg), m2(cfg);
        train(e2, ta);
        train(m2, tb);
        e2.add_ensemble_member(std::move(m2), 0.5);
        auto& h = e2.hp_backend();
        h.set_ensemble_learning_rate(0.05);
        const std::string tb_more = corpus(3, 1);
        for (int i = 0; i < 400; ++i) {
            const auto c = static_cast<std::uint8_t>(tb_more[static_cast<std::size_t>(i)]);
            const double lp = h.serve_next_byte_log_probs(256)[c];
            const double loss = h.observe_next_byte(c);
            if (std::abs(loss + lp) > 1e-12) {
                std::printf("cyphalm_ensemble_smoke FAIL observe %.12f != -log p %.12f\n", loss, -lp);
                return 1;
            }
        }
        const auto w = h.ensemble_weights();
        if (w.size() != 2 || std::abs(w[0] + w[1] - 1.0) > 1e-9 || !(w[1] > 0.5)) {
            std::printf("cyphalm_ensemble_smoke FAIL learned weights %.4f %.4f\n", w[0], w.size() > 1 ? w[1] : -1.0);
            return 1;
        }
    }
    // Served distribution reuse: log_prob_byte, greedy and sampling read the
    // full mix whether or not it was just scored, observe returns the same
    // log p, and the extra calls change nothing learned (a twin that only
    // scores and observes ends bit-identical).
    {
        cypha::cyphalm::CyphaLMModel e3(cfg), m3(cfg), t3(cfg), tm3(cfg);
        train(e3, ta);
        train(m3, tb);
        train(t3, ta);
        train(tm3, tb);
        e3.add_ensemble_member(std::move(m3), 0.5);
        t3.add_ensemble_member(std::move(tm3), 0.5);
        auto& h = e3.hp_backend();
        auto& twin = t3.hp_backend();
        h.set_ensemble_learning_rate(0.05);
        twin.set_ensemble_learning_rate(0.05);
        const std::string tb_more = corpus(4, 1);
        for (int i = 0; i < 300; ++i) {
            const auto c = static_cast<std::uint8_t>(tb_more[static_cast<std::size_t>(i)]);
            double lp_c = 0.0;
            int greedy = 0, sampled = 0;
            if (i % 3 == 1) h.invalidate_scoring_cache();
            if (i % 2 == 0) {  // cold: these score the mix themselves
                lp_c = h.log_prob_byte(c);
                greedy = h.serve_greedy_next_byte();
                sampled = h.serve_sample_next_byte(1.0, rng_fixed);
            }
            const auto lp = h.serve_next_byte_log_probs(256);
            if (i % 2 == 1) {  // warm: reuse the distribution just served
                lp_c = h.log_prob_byte(c);
                greedy = h.serve_greedy_next_byte();
                sampled = h.serve_sample_next_byte(1.0, rng_fixed);
            }
            if (lp_c != lp[c] || greedy != argmax(lp) || sampled != draw(lp, rng_fixed())) {
                std::printf("cyphalm_ensemble_smoke FAIL served reuse at %d: %.12f vs %.12f, %d/%d, %d/%d\n", i,
                            lp_c, lp[c], greedy, argmax(lp), sampled, draw(lp, rng_fixed()));
                return 1;
            }
            const double loss = h.observe_next_byte(c);
            (void)twin.serve_next_byte_log_probs(256);
            (void)twin.observe_next_byte(c);
            if (loss != -lp[c]) {
                std::printf("cyphalm_ensemble_smoke FAIL observe after reuse %.12f != %.12f\n", loss, -lp[c]);
                return 1;
            }
        }
        const auto pa = h.all_predictors(), pb = twin.all_predictors();
        if (h.ensemble_weights() != twin.ensemble_weights() || pa[0]->learned_digest() != pb[0]->learned_digest() ||
            pa[1]->learned_digest() != pb[1]->learned_digest()) {
            std::printf("cyphalm_ensemble_smoke FAIL served reuse changed learning\n");
            return 1;
        }
        // Non-frozen scoring trains inside the hypothetical byte, so a
        // distribution scored with learning on is stale once learning is off.
        h.set_frozen_scoring(false);
        const auto c = static_cast<std::uint8_t>('s');
        (void)h.serve_next_byte_log_probs(256);
        h.set_learning(false);
        const double lp_off = h.log_prob_byte(c);
        const double want = h.serve_next_byte_log_probs(256)[c];
        h.set_learning(true);
        if (lp_off != want) {
            std::printf("cyphalm_ensemble_smoke FAIL stale served distribution after set_learning\n");
            return 1;
        }
    }
    // Manifest: save two checkpoints + manifest, load it as one model, and get
    // the same distribution as an ensemble built directly.
    {
        const auto dir = std::filesystem::temp_directory_path() / "cyphalm_ensemble_smoke";
        std::filesystem::create_directories(dir);
        cypha::cyphalm::CyphaLMModel p(cfg), q(cfg), direct(cfg), dm(cfg);
        train(p, ta);
        train(q, tb);
        train(direct, ta);
        train(dm, tb);
        cypha::cyphalm::save_cyphalm_model(p, (dir / "a").string());
        cypha::cyphalm::save_cyphalm_model(q, (dir / "b").string());
        cypha::cyphalm::save_cyphalm_ensemble_manifest((dir / "ens.json").string(), {"a.json", "b.json"});
        auto loaded = cypha::cyphalm::load_cyphalm_model((dir / "ens.json").string());
        direct.add_ensemble_member(std::move(dm), 0.5);
        const auto l1 = loaded.hp_backend().serve_next_byte_log_probs(256);
        const auto l2 = direct.hp_backend().serve_next_byte_log_probs(256);
        double err = 0.0;
        for (int i = 0; i < 256; ++i) err = std::max(err, std::abs(l1[i] - l2[i]));
        std::filesystem::remove_all(dir);
        if (loaded.hp_backend().ensemble_size() != 1 || err > 1e-9) {
            std::printf("cyphalm_ensemble_smoke FAIL manifest: %zu members, err %.3g\n",
                        loaded.hp_backend().ensemble_size(), err);
            return 1;
        }
    }
    std::printf("cyphalm_ensemble_smoke OK mix exact over 300 bytes; lookahead kept both models; weights learn; "
                "served reuse exact; manifest\n");
    return 0;
}
