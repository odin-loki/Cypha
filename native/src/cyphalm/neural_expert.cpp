#include "cypha/cyphalm/neural_expert.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace cypha::cyphalm {

namespace {

#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__) && !defined(__clang__)
#define CYPHA_NN_CLONES __attribute__((target_clones("avx2", "default")))
#else
#define CYPHA_NN_CLONES
#endif

// out[r] = bias[r] + sum_k w[r][k] * x[k]. Eight independent partial sums
// per row (and two rows at a time) so the compiler vectorises without
// reassociating a single float sum.
CYPHA_NN_CLONES
void matvec(const float* w, const float* x, const float* bias, float* out, int rows, int cols) {
    const int c8 = cols & ~7;
    int r = 0;
    for (; r + 1 < rows; r += 2) {
        const float* w0 = w + static_cast<std::size_t>(r) * cols;
        const float* w1 = w0 + cols;
        float a0[8] = {}, a1[8] = {};
        for (int k = 0; k < c8; k += 8)
            for (int j = 0; j < 8; ++j) {
                a0[j] += w0[k + j] * x[k + j];
                a1[j] += w1[k + j] * x[k + j];
            }
        float s0 = 0.0f, s1 = 0.0f;
        for (int j = 0; j < 8; ++j) {
            s0 += a0[j];
            s1 += a1[j];
        }
        for (int k = c8; k < cols; ++k) {
            s0 += w0[k] * x[k];
            s1 += w1[k] * x[k];
        }
        out[r] = s0 + bias[r];
        out[r + 1] = s1 + bias[r + 1];
    }
    for (; r < rows; ++r) {
        const float* wr = w + static_cast<std::size_t>(r) * cols;
        float acc = 0.0f;
        for (int k = 0; k < cols; ++k) acc += wr[k] * x[k];
        out[r] = acc + bias[r];
    }
}

float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

template <class T>
void read_array(std::istream& is, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

}  // namespace

std::shared_ptr<const ByteLstmExpert> ByteLstmExpert::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("ByteLstmExpert: cannot open " + path);
    char magic[4];
    std::uint32_t hdr[3];
    is.read(magic, 4);
    is.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!is || std::memcmp(magic, "BLM1", 4) != 0) throw std::runtime_error("ByteLstmExpert: bad file " + path);
    auto m = std::make_shared<ByteLstmExpert>();
    m->layers_ = static_cast<int>(hdr[0]);
    m->d_ = static_cast<int>(hdr[1]);
    m->emb_ = static_cast<int>(hdr[2]);
    if (m->layers_ < 1 || m->layers_ > 8 || m->d_ < 1 || m->d_ > 8192 || m->emb_ < 1 || m->emb_ > 8192) {
        throw std::runtime_error("ByteLstmExpert: bad shape in " + path);
    }
    const std::size_t d = static_cast<std::size_t>(m->d_);
    read_array(is, m->emb_w_, 256 * static_cast<std::size_t>(m->emb_));
    for (int l = 0; l < m->layers_; ++l) {
        const std::size_t in = l == 0 ? static_cast<std::size_t>(m->emb_) : d;
        std::vector<float> wih, whh, bih, bhh;
        read_array(is, wih, 4 * d * in);
        read_array(is, whh, 4 * d * d);
        read_array(is, bih, 4 * d);
        read_array(is, bhh, 4 * d);
        std::vector<float> w(4 * d * (in + d));
        for (std::size_t r = 0; r < 4 * d; ++r) {
            std::copy_n(&wih[r * in], in, &w[r * (in + d)]);
            std::copy_n(&whh[r * d], d, &w[r * (in + d) + in]);
        }
        for (std::size_t r = 0; r < 4 * d; ++r) bih[r] += bhh[r];
        m->w_.push_back(std::move(w));
        m->b_.push_back(std::move(bih));
    }
    read_array(is, m->out_w_, 256 * d);
    read_array(is, m->out_b_, 256);
    if (!is) throw std::runtime_error("ByteLstmExpert: truncated " + path);
    return m;
}

std::size_t ByteLstmExpert::parameter_count() const {
    std::size_t n = emb_w_.size() + out_w_.size() + out_b_.size();
    for (std::size_t l = 0; l < w_.size(); ++l) n += w_[l].size() + b_[l].size();
    return n;
}

ByteLstmExpert::State ByteLstmExpert::initial_state() const {
    State s;
    s.h.assign(static_cast<std::size_t>(layers_) * d_, 0.0f);
    s.c.assign(static_cast<std::size_t>(layers_) * d_, 0.0f);
    step(s, 0);
    return s;
}

void ByteLstmExpert::step(State& s, std::uint8_t byte) const {
    const int d = d_;
    std::vector<float> x(emb_w_.begin() + static_cast<std::ptrdiff_t>(byte) * emb_,
                         emb_w_.begin() + static_cast<std::ptrdiff_t>(byte + 1) * emb_);
    std::vector<float> xin, gates(4 * static_cast<std::size_t>(d));
    for (int l = 0; l < layers_; ++l) {
        float* h = &s.h[static_cast<std::size_t>(l) * d];
        float* c = &s.c[static_cast<std::size_t>(l) * d];
        xin.assign(x.begin(), x.end());
        xin.insert(xin.end(), h, h + d);
        matvec(w_[l].data(), xin.data(), b_[l].data(), gates.data(), 4 * d, static_cast<int>(xin.size()));
        for (int k = 0; k < d; ++k) {
            const float i = sigmoid(gates[k]);
            const float f = sigmoid(gates[d + k]);
            const float g = std::tanh(gates[2 * d + k]);
            const float o = sigmoid(gates[3 * d + k]);
            c[k] = f * c[k] + i * g;
            h[k] = o * std::tanh(c[k]);
        }
        x.assign(h, h + d);
    }
    float logits[256];
    matvec(out_w_.data(), x.data(), out_b_.data(), logits, 256, d);
    const float mx = *std::max_element(logits, logits + 256);
    double z = 0.0;
    for (float v : logits) z += std::exp(static_cast<double>(v - mx));
    const double lz = std::log(z) + mx;
    for (int b = 0; b < 256; ++b) s.log_p[static_cast<std::size_t>(b)] = logits[b] - lz;
}

}  // namespace cypha::cyphalm
