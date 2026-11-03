// Small byte model. The next-byte distribution is routed by Cypha's own
// math: hp::GriaGate (alpha, entropy bucket, trajectory), predictive
// uncertainty u = H(order-1 KT) / ln(256), a running calibration gap C,
// and a disagreement L between the match and the order-1 model.
// r_epi = 32/(32+count) may die. It only refuses a match on an unseen
// left context. It does not scale the unigram opening.
//
// The large context tables are not allocated. Long context is an index back
// into the training bytes. Held-out counts are frozen.

#include "hp/gria.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSample = 8'388'608;
constexpr std::size_t kTrain = 7'340'032;
constexpr std::uint32_t kEmpty = 0xffffffffu;

std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

int byte_class(unsigned b) {
    if (b == ' ' || b == '\n' || b == '\t' || b == '\r') return 0;
    if (b >= '0' && b <= '9') return 1;
    if (b >= 'a' && b <= 'z') return 2;
    if (b >= 'A' && b <= 'Z') return 3;
    return 4;
}

struct PosMap {
    std::vector<std::uint32_t> keys, vals;
    std::vector<std::uint16_t> seen;
    std::uint32_t mask = 0, filled = 0;

    void reset(std::uint32_t bits) {
        keys.assign(1u << bits, kEmpty);
        vals.assign(1u << bits, 0);
        seen.assign(1u << bits, 0);
        mask = (1u << bits) - 1;
        filled = 0;
    }
    void put_new(std::uint32_t key, std::uint32_t val, std::uint16_t n) {
        std::uint32_t i = mix32(key) & mask;
        while (keys[i] != kEmpty) i = (i + 1) & mask;
        keys[i] = key;
        vals[i] = val;
        seen[i] = n;
        ++filled;
    }
    void grow() {
        auto ok = std::move(keys);
        auto ov = std::move(vals);
        auto os = std::move(seen);
        reset(static_cast<std::uint32_t>(std::bit_width(ok.size())));
        for (std::size_t i = 0; i < ok.size(); ++i)
            if (ok[i] != kEmpty) put_new(ok[i], ov[i], os[i]);
    }
    void put(std::uint32_t key, std::uint32_t val) {
        if ((filled + 1) * 10 > mask * 7) grow();
        std::uint32_t i = mix32(key) & mask;
        for (;;) {
            if (keys[i] == kEmpty) {
                keys[i] = key;
                vals[i] = val;
                seen[i] = 1;
                ++filled;
                return;
            }
            if (keys[i] == key) {
                vals[i] = val;
                if (seen[i] < 65535) ++seen[i];
                return;
            }
            i = (i + 1) & mask;
        }
    }
    bool find(std::uint32_t key, std::uint32_t& follower, int& times) const {
        if (keys.empty()) return false;
        std::uint32_t i = mix32(key) & mask;
        for (int n = 0; n < 64; ++n) {
            if (keys[i] == kEmpty) return false;
            if (keys[i] == key) {
                follower = vals[i];
                times = seen[i];
                return true;
            }
            i = (i + 1) & mask;
        }
        return false;
    }
};

struct Model {
    std::uint32_t uni[256]{};
    std::uint32_t uni_n = 0;
    std::uint32_t o1[256][256]{};
    std::uint32_t o1n[256]{};
    std::uint32_t cls[5][256]{};
    std::uint32_t clsn[5]{};
    PosMap pos;
    hp::GriaGate gria;
    const std::uint8_t* data = nullptr;
    double conf_ema = 0.4;
    double hit_ema = 0.4;
    bool learning = true;

    void init() { pos.reset(20); }

    static std::uint32_t key4(const std::uint8_t* d, std::size_t i) {
        return (std::uint32_t(d[i - 4]) << 24) | (std::uint32_t(d[i - 3]) << 16) |
               (std::uint32_t(d[i - 2]) << 8) | std::uint32_t(d[i - 1]);
    }
    double kt0(unsigned y) const {
        return (uni[y] + 0.5) / (static_cast<double>(uni_n) + 128.0);
    }
    double kt1(unsigned prev, unsigned y) const {
        return (o1[prev][y] + 0.5) / (static_cast<double>(o1n[prev]) + 128.0);
    }
    double ktc(int c, unsigned y) const {
        return (cls[c][y] + 0.5) / (static_cast<double>(clsn[c]) + 128.0);
    }
    // Lifetime count. Goes to zero. Not a mix weight.
    double r_epi(unsigned prev) const {
        return 32.0 / (32.0 + static_cast<double>(o1n[prev]));
    }

    // Entropy of the 256 order-1 KT probabilities, divided by ln(256).
    // Stays in (0, 1) on a common previous byte whose successor is still spread out.
    double predictive_u(unsigned prev) const {
        double h = 0;
        for (int y = 0; y < 256; ++y) {
            const double p = kt1(prev, static_cast<unsigned>(y));
            h -= p * std::log(p);
        }
        return h / std::log(256.0);
    }

    // Fills P[256]. Uses history in `hist` (length i) and match followers in `data`.
    void distribution(const std::uint8_t* hist, std::size_t i, std::size_t follower_limit, double P[256]) const {
        const unsigned prev = i ? hist[i - 1] : 0;
        const int c = byte_class(prev);
        const double r_epi_prev = r_epi(prev);
        const double u = predictive_u(prev);
        const double alpha = gria.alpha_q16() / 65536.0;
        const int traj = (gria.bucket() % 3);  // 0 falling, 1 flat, 2 rising when enabled
        unsigned pred = 0;
        int len = 0;
        int times = 0;
        if (i >= 4) {
            std::uint32_t key = (std::uint32_t(hist[i - 4]) << 24) | (std::uint32_t(hist[i - 3]) << 16) |
                                (std::uint32_t(hist[i - 2]) << 8) | std::uint32_t(hist[i - 1]);
            std::uint32_t follower = 0;
            // r_epi refuses the match only while the left context is still unseen.
            if (pos.find(key, follower, times) && follower < follower_limit && times >= 2 &&
                r_epi_prev < 0.5) {
                pred = data[follower];
                len = 4;
                int back = 5;
                while (len < 24 && i >= static_cast<std::size_t>(back) &&
                       follower >= static_cast<std::size_t>(back) &&
                       hist[i - static_cast<std::size_t>(back)] ==
                           data[follower - static_cast<std::size_t>(back)]) {
                    ++len;
                    ++back;
                }
            }
        }
        const double p_ord = len >= 4 ? kt1(prev, pred) : 0.0;
        const double agree = p_ord / (p_ord + 0.05);
        const double stable = (traj == 1) ? 1.0 : 0.5;
        double match_w = 0;
        if (len >= 4) {
            match_w = stable * (1.0 - u) * (0.35 + 0.65 * agree) * (0.45 + 0.55 * alpha);
            if (match_w < 0) match_w = 0;
            if (match_w > 0.75) match_w = 0.75;
        }
        const bool falling = (traj == 0);
        const double s = std::clamp(0.15 + 0.55 * u + (falling ? 0.2 : 0.0), 0.0, 0.75);
        for (int y = 0; y < 256; ++y) {
            const double base = (1.0 - s) * (0.8 * kt1(prev, y) + 0.2 * ktc(c, y)) + s * kt0(y);
            P[y] = (len >= 4 && static_cast<unsigned>(y) == pred) ? match_w + (1.0 - match_w) * base
                                                                  : (1.0 - match_w) * base;
        }
    }

    double prob_at(std::size_t i, unsigned y) const {
        double P[256];
        distribution(data, i, i, P);
        return std::max(P[y], 1e-12);
    }

    void learn(std::size_t i, double p_true) {
        const unsigned y = data[i];
        const unsigned prev = i ? data[i - 1] : 0;
        double routed[256];
        distribution(data, i, i, routed);
        int arg = 0;
        for (int b = 1; b < 256; ++b)
            if (routed[b] > routed[arg]) arg = b;
        conf_ema = 0.98 * conf_ema + 0.02 * routed[arg];
        hit_ema = 0.98 * hit_ema + 0.02 * (arg == static_cast<int>(y) ? 1.0 : 0.0);
        ++uni[y];
        ++uni_n;
        ++o1[prev][y];
        ++o1n[prev];
        const int c = byte_class(prev);
        ++cls[c][y];
        ++clsn[c];
        if (i >= 4) pos.put(key4(data, i), static_cast<std::uint32_t>(i));
        int p12 = static_cast<int>(p_true * 4096.0);
        if (p12 < 1) p12 = 1;
        if (p12 > 4095) p12 = 4095;
        gria.account_bit(p12);
        gria.account_byte(static_cast<int>(y));
    }
};

double distinct4(const std::string& s) {
    if (s.size() < 4) return 0;
    std::vector<std::uint32_t> g;
    for (std::size_t i = 0; i + 3 < s.size(); ++i) {
        g.push_back((std::uint32_t(std::uint8_t(s[i])) << 24) | (std::uint32_t(std::uint8_t(s[i + 1])) << 16) |
                    (std::uint32_t(std::uint8_t(s[i + 2])) << 8) | std::uint8_t(s[i + 3]));
    }
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return static_cast<double>(g.size()) / static_cast<double>(s.size() - 3);
}

std::string show(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c >= 32 && c < 127) o += static_cast<char>(c);
        else {
            const char* h = "0123456789abcdef";
            o += "\\x";
            o += h[c >> 4];
            o += h[c & 15];
        }
    }
    return o;
}

std::string sample(const Model& model, const std::vector<std::uint8_t>& seed, int ngen, std::mt19937& rng) {
    std::vector<std::uint8_t> seq = seed;
    const double C = 1.0 - std::fabs(model.conf_ema - model.hit_ema);
    const double nucleus = 0.80 + 0.15 * (1.0 - std::clamp(C, 0.0, 1.0));
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::string text;
    text.reserve(static_cast<std::size_t>(ngen));
    for (int t = 0; t < ngen; ++t) {
        const std::size_t i = seq.size();
        const unsigned prev = seq.back();
        const double u = model.predictive_u(prev);
        const double temperature = 0.65 + 0.5 * u;
        double P[256];
        // History for the match is the generated prefix; followers stay in the training buffer.
        model.distribution(seq.data(), i, kTrain, P);
        bool looped = false;
        if (i >= 16) {
            for (std::size_t s = 0; s + 16 < i; ++s) {
                if (std::memcmp(seq.data() + s, seq.data() + (i - 16), 16) == 0) looped = true;
            }
        }
        double raw[256];
        double sum = 0;
        double mx = 0;
        for (int y = 0; y < 256; ++y) mx = std::max(mx, P[y]);
        for (int y = 0; y < 256; ++y) {
            double p = P[y];
            if (looped && p > 0.5 * mx) p = model.kt1(prev, static_cast<unsigned>(y));
            if (i >= 3 && seq[i - 1] == seq[i - 2] && seq[i - 2] == seq[i - 3] &&
                static_cast<unsigned>(y) == seq[i - 1]) {
                p *= 0.02;
            }
            raw[y] = std::pow(std::max(p, 1e-12), 1.0 / temperature);
            sum += raw[y];
        }
        struct Item {
            int y;
            double p;
        } items[256];
        for (int y = 0; y < 256; ++y) items[y] = {y, raw[y] / sum};
        std::sort(items, items + 256, [](const Item& a, const Item& b) { return a.p > b.p; });
        double acc = 0;
        int keep = 1;
        for (int n = 0; n < 256; ++n) {
            acc += items[n].p;
            keep = n + 1;
            if (acc >= nucleus) break;
        }
        double mass = 0;
        for (int n = 0; n < keep; ++n) mass += items[n].p;
        double r = U(rng) * mass;
        int pick = items[0].y;
        for (int n = 0; n < keep; ++n) {
            r -= items[n].p;
            if (r <= 0) {
                pick = items[n].y;
                break;
            }
        }
        seq.push_back(static_cast<std::uint8_t>(pick));
        text.push_back(static_cast<char>(pick));
    }
    return text;
}

}  // namespace

int main() {
    std::ifstream in("/mnt/d/cypha_llm/enwik9", std::ios::binary);
    if (!in) return 1;
    std::vector<std::uint8_t> data(kSample);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(kSample));
    if (static_cast<std::size_t>(in.gcount()) < kSample) return 1;

    Model model;
    model.data = data.data();
    model.init();
    double train_bits = 0;
    for (std::size_t i = 0; i < kTrain; ++i) {
        const double p = model.prob_at(i, data[i]);
        train_bits += -std::log2(p);
        model.learn(i, p);
        if (((i + 1) & 0x1fffff) == 0) std::cerr << "train " << (i + 1) << "\n";
    }
    model.learning = false;
    double held = 0, sum_u = 0, sum_epi = 0;
    for (std::size_t i = kTrain; i < kSample; ++i) {
        const unsigned prev = data[i - 1];
        sum_u += model.predictive_u(prev);
        sum_epi += model.r_epi(prev);
        const double p = model.prob_at(i, data[i]);
        held += -std::log2(p);
        int p12 = static_cast<int>(p * 4096.0);
        if (p12 < 1) p12 = 1;
        if (p12 > 4095) p12 = 4095;
        model.gria.account_bit(p12);
        model.gria.account_byte(data[i]);
    }

    const std::string needle = " is a ";
    std::size_t at = 64;
    for (std::size_t i = 48; i + needle.size() < kTrain; ++i) {
        if (std::memcmp(data.data() + i, needle.data(), needle.size()) == 0) {
            at = i;
            break;
        }
    }
    std::vector<std::uint8_t> seed(data.begin() + static_cast<std::ptrdiff_t>(at - 48),
                                   data.begin() + static_cast<std::ptrdiff_t>(at));
    std::mt19937 rng(42);
    const std::string a = sample(model, seed, 200, rng);
    std::vector<std::uint8_t> seedb = {'T', 'h', 'e', ' '};
    const std::string b = sample(model, seedb, 200, rng);

    const double hn = static_cast<double>(kSample - kTrain);
    const double C = 1.0 - std::fabs(model.conf_ema - model.hit_ema);
    std::cout << "name cypha_routed\n";
    std::cout << "heldout_bpc " << (held / hn) << "\n";
    std::cout << "train_online_bpc " << (train_bits / static_cast<double>(kTrain)) << "\n";
    std::cout << "ram_bytes " << (model.pos.filled * 10ull + sizeof(model.o1) + sizeof(model.cls)) << "\n";
    std::cout << "alpha " << (model.gria.alpha_q16() / 65536.0) << "\n";
    std::cout << "entropy_bucket " << model.gria.entropy_bucket() << "\n";
    std::cout << "calibration_C " << C << "\n";
    std::cout << "mean_r_epi " << (sum_epi / hn) << "\n";
    std::cout << "mean_u " << (sum_u / hn) << "\n";
    std::cout << "distinct4_A " << distinct4(a) << "\n";
    std::cout << "distinct4_B " << distinct4(b) << "\n";
    std::cout << "SAMPLE_A " << show(a) << "\n";
    std::cout << "SAMPLE_B " << show(b) << "\n";
    return 0;
}
