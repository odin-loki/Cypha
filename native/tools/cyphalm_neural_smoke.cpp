/// Neural expert (ByteLstmExpert, BLM1): a random two-layer LSTM written to a
/// file steps exactly like a straightforward double-precision reference, and
/// an hp model with it attached serves normalised distributions whose mixing
/// weights move, and whose state restores after a rewind.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "cypha/cyphalm/neural_expert.hpp"

namespace {

struct Ref {
    int layers, d, emb;
    std::vector<std::vector<double>> w;  // file order
    std::vector<double> h, c;
    std::vector<double> logp;
    const std::vector<double>& at(int i) const { return w[static_cast<std::size_t>(i)]; }
    void step(int byte) {
        std::vector<double> x(at(0).begin() + byte * emb, at(0).begin() + (byte + 1) * emb);
        for (int l = 0; l < layers; ++l) {
            const auto& wih = at(1 + 4 * l);
            const auto& whh = at(2 + 4 * l);
            const auto& bih = at(3 + 4 * l);
            const auto& bhh = at(4 + 4 * l);
            const int in = static_cast<int>(x.size());
            std::vector<double> g(4 * static_cast<std::size_t>(d));
            for (int r = 0; r < 4 * d; ++r) {
                double s = bih[r] + bhh[r];
                for (int k = 0; k < in; ++k) s += wih[static_cast<std::size_t>(r) * in + k] * x[k];
                for (int k = 0; k < d; ++k) s += whh[static_cast<std::size_t>(r) * d + k] * h[l * d + k];
                g[r] = s;
            }
            auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
            for (int k = 0; k < d; ++k) {
                double& cc = c[l * d + k];
                cc = sig(g[d + k]) * cc + sig(g[k]) * std::tanh(g[2 * d + k]);
                h[l * d + k] = sig(g[3 * d + k]) * std::tanh(cc);
            }
            x.assign(h.begin() + l * d, h.begin() + (l + 1) * d);
        }
        const auto& ow = at(1 + 4 * layers);
        const auto& ob = at(2 + 4 * layers);
        std::vector<double> z(256);
        double mx = -1e300;
        for (int b = 0; b < 256; ++b) {
            double s = ob[b];
            for (int k = 0; k < d; ++k) s += ow[static_cast<std::size_t>(b) * d + k] * x[k];
            mx = std::max(mx, z[b] = s);
        }
        double tot = 0.0;
        for (double v : z) tot += std::exp(v - mx);
        logp.resize(256);
        for (int b = 0; b < 256; ++b) logp[b] = z[b] - mx - std::log(tot);
    }
};

}  // namespace

int main() {
    const int layers = 2, d = 24, emb = 8;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    Ref ref{layers, d, emb, {}, std::vector<double>(layers * d, 0.0), std::vector<double>(layers * d, 0.0), {}};
    // Values exactly representable in bf16, the runtime's matrix format.
    auto bf16 = [](float f) {
        std::uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        u &= 0xFFFF0000u;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    };
    auto add = [&](std::size_t n) {
        std::vector<double> v(n);
        for (auto& x : v) x = bf16(nd(rng));
        ref.w.push_back(v);
    };
    add(256 * emb);
    for (int l = 0; l < layers; ++l) {
        add(4 * d * (l == 0 ? emb : d));
        add(4 * d * d);
        add(4 * d);
        add(4 * d);
    }
    add(256 * d);
    add(256);
    const auto path = std::filesystem::temp_directory_path() / "cyphalm_neural_smoke.blm";
    {
        std::ofstream os(path, std::ios::binary);
        os.write("BLM1", 4);
        const std::uint32_t hdr[3] = {layers, d, emb};
        os.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        for (const auto& v : ref.w)
            for (double x : v) {
                const float f = static_cast<float>(x);
                os.write(reinterpret_cast<const char*>(&f), sizeof(f));
            }
    }
    auto nn = cypha::cyphalm::ByteLstmExpert::load(path.string());
    auto st = nn->initial_state();
    ref.step(0);
    const std::string text = "The cat sat on the mat. [[Link]] 1984\n";
    double worst = 0.0;
    for (char ch : text) {
        for (int b = 0; b < 256; ++b) worst = std::max(worst, std::abs(st.log_p[b] - ref.logp[b]));
        nn->step(st, static_cast<std::uint8_t>(ch));
        ref.step(static_cast<unsigned char>(ch));
    }
    if (worst > 1e-4) {
        std::printf("cyphalm_neural_smoke FAIL: step differs from reference by %.3g\n", worst);
        return 1;
    }

    // Attached to an hp model.
    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.hp_table_bits = 16;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cypha::cyphalm::CyphaLMModel model(cfg);
    model.attach_neural(path.string());
    auto& hp = model.hp_backend();
    const auto w0 = hp.neural_weights();
    std::string corpus;
    for (int i = 0; i < 40; ++i) corpus += text;
    for (char ch : corpus) {
        const auto lp = hp.next_byte_log_probs(256);
        double z = 0.0;
        for (double v : lp) z += std::exp(v);
        if (std::abs(z - 1.0) > 1e-6) {
            std::printf("cyphalm_neural_smoke FAIL: mix sums to %.12f\n", z);
            return 1;
        }
        hp.observe_next_byte(static_cast<std::uint8_t>(ch));
    }
    if (hp.neural_weights() == w0) {
        std::printf("cyphalm_neural_smoke FAIL: mixing weights never moved\n");
        return 1;
    }
    const auto saved = hp.neural_state();
    const auto before = hp.next_byte_log_probs(256);
    hp.consume_byte('x');
    hp.set_neural_state(saved);
    // The predictor moved on; only the LSTM part is restored, so compare it.
    if (std::memcmp(hp.neural_state().log_p.data(), saved.log_p.data(), sizeof(double) * 256) != 0) {
        std::printf("cyphalm_neural_smoke FAIL: state restore\n");
        return 1;
    }
    (void)before;
    std::filesystem::remove(path);
    std::printf("cyphalm_neural_smoke OK: max step error %.2g; mix normalised; weights adapt\n", worst);
    return 0;
}
