#include "cypha/cyphalm/text_algebraic_fingerprint.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace cypha::cyphalm {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-12;

double clamp01(double x) {
    if (!std::isfinite(x)) {
        return 0.0;
    }
    return std::max(0.0, std::min(1.0, x));
}

double shannon_entropy_bits(const std::vector<double>& probs) {
    double h = 0.0;
    for (double p : probs) {
        if (p > kEps) {
            h -= p * std::log2(p);
        }
    }
    return h;
}

/// Berlekamp-Massey linear complexity on a binary sequence.
int berlekamp_massey_complexity(const std::vector<int>& bits) {
    const int n = static_cast<int>(bits.size());
    if (n <= 0) {
        return 0;
    }
    std::vector<int> c(n + 1, 0);
    std::vector<int> b(n + 1, 0);
    c[0] = 1;
    b[0] = 1;
    int L = 0;
    int m = -1;
    for (int N = 0; N < n; ++N) {
        int d = bits[static_cast<std::size_t>(N)];
        for (int i = 1; i <= L; ++i) {
            d ^= c[i] & bits[static_cast<std::size_t>(N - i)];
        }
        if (d == 1) {
            std::vector<int> t = c;
            const int shift = N - m;
            for (int j = shift; j <= n; ++j) {
                c[j] ^= b[j - shift];
            }
            if (2 * L <= N) {
                L = N + 1 - L;
                m = N;
                b = t;
            }
        }
    }
    return L;
}

double linear_complexity_feature(const std::string& text) {
    if (text.empty()) {
        return 0.0;
    }
    std::vector<int> bits;
    bits.reserve(text.size());
    for (unsigned char ch : text) {
        bits.push_back(static_cast<int>(ch) & 1);
    }
    const int lc = berlekamp_massey_complexity(bits);
    return clamp01(static_cast<double>(lc) / static_cast<double>(bits.size()));
}

void dft_power_spectrum(const std::vector<double>& signal, std::vector<double>& power) {
    const int n = static_cast<int>(signal.size());
    power.clear();
    if (n <= 1) {
        return;
    }
    const int bins = n / 2;
    power.resize(static_cast<std::size_t>(bins));
    for (int k = 1; k <= bins; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (int j = 0; j < n; ++j) {
            const double angle = -2.0 * kPi * static_cast<double>(k) * static_cast<double>(j) /
                                 static_cast<double>(n);
            re += signal[static_cast<std::size_t>(j)] * std::cos(angle);
            im += signal[static_cast<std::size_t>(j)] * std::sin(angle);
        }
        power[static_cast<std::size_t>(k - 1)] = re * re + im * im;
    }
}

double spectral_flatness_feature(const std::string& text) {
    if (text.size() < 4) {
        return 0.0;
    }
    std::vector<double> signal(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        signal[i] = static_cast<double>(static_cast<unsigned char>(text[i])) / 255.0;
    }
    std::vector<double> power;
    dft_power_spectrum(signal, power);
    if (power.empty()) {
        return 0.0;
    }
    double log_sum = 0.0;
    double arith = 0.0;
    int count = 0;
    for (double p : power) {
        const double v = std::max(p, kEps);
        log_sum += std::log(v);
        arith += v;
        ++count;
    }
    if (count <= 0 || arith <= kEps) {
        return 0.0;
    }
    const double geom = std::exp(log_sum / static_cast<double>(count));
    return clamp01(geom / (arith / static_cast<double>(count)));
}

double run_length_entropy_feature(const std::string& text) {
    if (text.empty()) {
        return 0.0;
    }
    std::unordered_map<int, int> hist;
    int run_len = 1;
    for (std::size_t i = 1; i < text.size(); ++i) {
        if (text[i] == text[i - 1]) {
            ++run_len;
        } else {
            ++hist[run_len];
            run_len = 1;
        }
    }
    ++hist[run_len];

    const int n_runs = static_cast<int>(hist.size());
    if (n_runs <= 1) {
        return 0.0;
    }
    std::vector<double> probs;
    probs.reserve(hist.size());
    int total = 0;
    for (const auto& kv : hist) {
        total += kv.second;
    }
    for (const auto& kv : hist) {
        probs.push_back(static_cast<double>(kv.second) / static_cast<double>(total));
    }
    const double h = shannon_entropy_bits(probs);
    const double h_max = std::log2(static_cast<double>(n_runs));
    return h_max > kEps ? clamp01(h / h_max) : 0.0;
}

double ngram_entropy_feature(const std::string& text, int n) {
    if (n <= 0 || static_cast<int>(text.size()) < n) {
        return 0.0;
    }
    std::unordered_map<std::string, int> counts;
    const int total = static_cast<int>(text.size()) - n + 1;
    for (int i = 0; i + n <= static_cast<int>(text.size()); ++i) {
        ++counts[text.substr(static_cast<std::size_t>(i), static_cast<std::size_t>(n))];
    }
    if (counts.empty()) {
        return 0.0;
    }
    std::vector<double> probs;
    probs.reserve(counts.size());
    for (const auto& kv : counts) {
        probs.push_back(static_cast<double>(kv.second) / static_cast<double>(total));
    }
    const double h = shannon_entropy_bits(probs);
    const double h_max = std::log2(static_cast<double>(std::max(1, static_cast<int>(counts.size()))));
    return h_max > kEps ? clamp01(h / h_max) : 0.0;
}

}  // namespace

std::vector<double> TextAlgebraicFingerprint::as_vector() const {
    return {linear_complexity, spectral_flatness, run_length_entropy,
            ngram_entropy_1,     ngram_entropy_2,   ngram_entropy_3};
}

double TextAlgebraicFingerprint::spectrum_position() const {
    const std::vector<double> v = as_vector();
    if (v.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double x : v) {
        sum += clamp01(x);
    }
    return clamp01(sum / static_cast<double>(v.size()));
}

TextAlgebraicFingerprint compute_text_algebraic_fingerprint(const std::string& text) {
    TextAlgebraicFingerprint fp;
    fp.linear_complexity = linear_complexity_feature(text);
    fp.spectral_flatness = spectral_flatness_feature(text);
    fp.run_length_entropy = run_length_entropy_feature(text);
    fp.ngram_entropy_1 = ngram_entropy_feature(text, 1);
    fp.ngram_entropy_2 = ngram_entropy_feature(text, 2);
    fp.ngram_entropy_3 = ngram_entropy_feature(text, 3);
    return fp;
}

nlohmann::json fingerprint_to_json(const TextAlgebraicFingerprint& fp, bool include_spectrum_position) {
    nlohmann::json out;
    out["feature_names"] = {"linear_complexity", "spectral_flatness", "run_length_entropy",
                            "ngram_entropy_1", "ngram_entropy_2", "ngram_entropy_3"};
    out["vector"] = fp.as_vector();
    out["linear_complexity"] = fp.linear_complexity;
    out["spectral_flatness"] = fp.spectral_flatness;
    out["run_length_entropy"] = fp.run_length_entropy;
    out["ngram_entropy_1"] = fp.ngram_entropy_1;
    out["ngram_entropy_2"] = fp.ngram_entropy_2;
    out["ngram_entropy_3"] = fp.ngram_entropy_3;
    if (include_spectrum_position) {
        out["spectrum_position"] = fp.spectrum_position();
    }
    return out;
}

}  // namespace cypha::cyphalm
