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

// out[r] = bias[r] + sum_k w[r][k] * x[k] with bf16 weights (the upper 16
// bits of a float32): half the memory traffic, which is what bounds a single
// stream. Eight partial sums so the compiler vectorises without reassociating.
CYPHA_NN_CLONES
void matvec_bf16(const std::uint16_t* w, const float* x, const float* bias, float* out, int rows, int cols) {
    const int c8 = cols & ~7;
    for (int r = 0; r < rows; ++r) {
        const std::uint16_t* wr = w + static_cast<std::size_t>(r) * cols;
        float a[8] = {};
        for (int k = 0; k < c8; k += 8)
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t u = static_cast<std::uint32_t>(wr[k + j]) << 16;
                float f;
                std::memcpy(&f, &u, sizeof(f));
                a[j] += f * x[k + j];
            }
        float s = 0.0f;
        for (int j = 0; j < 8; ++j) s += a[j];
        for (int k = c8; k < cols; ++k) {
            const std::uint32_t u = static_cast<std::uint32_t>(wr[k]) << 16;
            float f;
            std::memcpy(&f, &u, sizeof(f));
            s += f * x[k];
        }
        out[r] = s + (bias ? bias[r] : 0.0f);
    }
}

CYPHA_NN_CLONES
float dot(const float* a, const float* b, int n) {
    float acc[8] = {};
    const int n8 = n & ~7;
    for (int k = 0; k < n8; k += 8)
        for (int j = 0; j < 8; ++j) acc[j] += a[k + j] * b[k + j];
    float s = 0.0f;
    for (int j = 0; j < 8; ++j) s += acc[j];
    for (int k = n8; k < n; ++k) s += a[k] * b[k];
    return s;
}

std::uint16_t to_bf16(float f) {  // round to nearest even
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<std::uint16_t>(u >> 16);
}

std::vector<std::uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
    return out;
}

float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

template <class T>
void read_array(std::istream& is, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

void log_softmax(const float* logits, std::array<double, 256>& out) {
    const float mx = *std::max_element(logits, logits + 256);
    double z = 0.0;
    for (int b = 0; b < 256; ++b) z += std::exp(static_cast<double>(logits[b] - mx));
    const double lz = std::log(z) + mx;
    for (int b = 0; b < 256; ++b) out[static_cast<std::size_t>(b)] = logits[b] - lz;
}

void layer_norm(const float* x, const float* w, const float* b, float* out, int d) {
    double mean = 0.0, var = 0.0;
    for (int k = 0; k < d; ++k) mean += x[k];
    mean /= d;
    for (int k = 0; k < d; ++k) var += (x[k] - mean) * (x[k] - mean);
    var /= d;
    const double inv = 1.0 / std::sqrt(var + 1e-5);
    for (int k = 0; k < d; ++k) out[k] = static_cast<float>((x[k] - mean) * inv) * w[k] + b[k];
}

}  // namespace

std::shared_ptr<const ByteNeuralExpert> load_neural_expert(const std::string& path) {
    char magic[4] = {};
    {
        std::ifstream is(path, std::ios::binary);
        if (!is) throw std::runtime_error("neural expert: cannot open " + path);
        is.read(magic, 4);
    }
    if (std::memcmp(magic, "BLM1", 4) == 0) return ByteLstmExpert::load(path);
    if (std::memcmp(magic, "BGT1", 4) == 0) return ByteGptExpert::load(path);
    throw std::runtime_error("neural expert: unknown file " + path);
}

// ---------------------------------------------------------------- LSTM

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
        m->w_.push_back(to_bf16(w));
        m->b_.push_back(std::move(bih));
    }
    std::vector<float> ow;
    read_array(is, ow, 256 * d);
    m->out_w_ = to_bf16(ow);
    read_array(is, m->out_b_, 256);
    if (!is) throw std::runtime_error("ByteLstmExpert: truncated " + path);
    return m;
}

std::size_t ByteLstmExpert::parameter_count() const {
    std::size_t n = emb_w_.size() + out_w_.size() + out_b_.size();
    for (std::size_t l = 0; l < w_.size(); ++l) n += w_[l].size() + b_[l].size();
    return n;
}

ByteNeuralExpert::State ByteLstmExpert::initial_state() const {
    State s;
    s.a.assign(static_cast<std::size_t>(layers_) * d_, 0.0f);
    s.b.assign(static_cast<std::size_t>(layers_) * d_, 0.0f);
    step(s, 0);
    return s;
}

void ByteLstmExpert::step(State& s, std::uint8_t byte) const {
    const int d = d_;
    std::vector<float> x(emb_w_.begin() + static_cast<std::ptrdiff_t>(byte) * emb_,
                         emb_w_.begin() + static_cast<std::ptrdiff_t>(byte + 1) * emb_);
    std::vector<float> xin, gates(4 * static_cast<std::size_t>(d));
    for (int l = 0; l < layers_; ++l) {
        float* h = &s.a[static_cast<std::size_t>(l) * d];
        float* c = &s.b[static_cast<std::size_t>(l) * d];
        xin.assign(x.begin(), x.end());
        xin.insert(xin.end(), h, h + d);
        matvec_bf16(w_[l].data(), xin.data(), b_[l].data(), gates.data(), 4 * d, static_cast<int>(xin.size()));
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
    matvec_bf16(out_w_.data(), x.data(), out_b_.data(), logits, 256, d);
    log_softmax(logits, s.log_p);
}

// ---------------------------------------------------------------- Transformer

std::shared_ptr<const ByteGptExpert> ByteGptExpert::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("ByteGptExpert: cannot open " + path);
    char magic[4];
    std::uint32_t hdr[4];
    is.read(magic, 4);
    is.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!is || std::memcmp(magic, "BGT1", 4) != 0) throw std::runtime_error("ByteGptExpert: bad file " + path);
    auto m = std::make_shared<ByteGptExpert>();
    m->layers_ = static_cast<int>(hdr[0]);
    m->d_ = static_cast<int>(hdr[1]);
    m->heads_ = static_cast<int>(hdr[2]);
    m->ctx_ = static_cast<int>(hdr[3]);
    if (m->layers_ < 1 || m->layers_ > 64 || m->d_ < 8 || m->d_ > 8192 || m->heads_ < 1 || m->d_ % m->heads_ != 0 ||
        m->ctx_ < 4 || m->ctx_ > 65536) {
        throw std::runtime_error("ByteGptExpert: bad shape in " + path);
    }
    const std::size_t d = static_cast<std::size_t>(m->d_);
    read_array(is, m->emb_, 256 * d);
    m->emb_bf_ = to_bf16(m->emb_);
    read_array(is, m->pos_, static_cast<std::size_t>(m->ctx_) * d);
    for (int l = 0; l < m->layers_; ++l) {
        Layer L;
        std::vector<float> w;
        read_array(is, L.ln1_w, d);
        read_array(is, L.ln1_b, d);
        read_array(is, w, 3 * d * d);
        L.qkv = to_bf16(w);
        read_array(is, w, d * d);
        L.proj = to_bf16(w);
        read_array(is, L.ln2_w, d);
        read_array(is, L.ln2_b, d);
        read_array(is, w, 4 * d * d);
        L.fc = to_bf16(w);
        read_array(is, w, 4 * d * d);
        L.fc2 = to_bf16(w);
        m->L_.push_back(std::move(L));
    }
    read_array(is, m->lnf_w_, d);
    read_array(is, m->lnf_b_, d);
    if (!is) throw std::runtime_error("ByteGptExpert: truncated " + path);
    return m;
}

std::size_t ByteGptExpert::parameter_count() const {
    std::size_t n = emb_.size() + pos_.size() + lnf_w_.size() + lnf_b_.size();
    for (const auto& L : L_)
        n += L.ln1_w.size() * 4 + L.qkv.size() + L.proj.size() + L.fc.size() + L.fc2.size();
    return n;
}

ByteNeuralExpert::State ByteGptExpert::initial_state() const {
    State s;
    const std::size_t cache = static_cast<std::size_t>(layers_) * ctx_ * d_;
    s.a.assign(cache, 0.0f);
    s.b.assign(cache, 0.0f);
    s.pos = 0;
    step(s, 0);
    return s;
}

void ByteGptExpert::step(State& s, std::uint8_t byte) const {
    if (s.pos >= ctx_) {
        // Window full: re-prime on the last ctx/2 bytes read (learned absolute
        // positions, so the cache cannot slide).
        const std::size_t keep = static_cast<std::size_t>(ctx_ / 2) - 1;
        std::vector<std::uint8_t> tail(s.hist.end() - static_cast<std::ptrdiff_t>(keep), s.hist.end());
        s.pos = 0;
        s.hist.clear();
        for (std::uint8_t b : tail) forward_(s, b, false);
    }
    forward_(s, byte, true);
}

void ByteGptExpert::forward_(State& s, std::uint8_t byte, bool want_logits) const {
    const int d = d_, hd = d_ / heads_, p = s.pos;
    s.hist.push_back(byte);
    std::vector<float> x(static_cast<std::size_t>(d)), h(static_cast<std::size_t>(d)),
        qkv(3 * static_cast<std::size_t>(d)), att(static_cast<std::size_t>(d)), y(static_cast<std::size_t>(d)),
        f(4 * static_cast<std::size_t>(d)), sc(static_cast<std::size_t>(p) + 1);
    for (int k = 0; k < d; ++k) x[k] = emb_[static_cast<std::size_t>(byte) * d + k] + pos_[static_cast<std::size_t>(p) * d + k];
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int l = 0; l < layers_; ++l) {
        const Layer& L = L_[static_cast<std::size_t>(l)];
        layer_norm(x.data(), L.ln1_w.data(), L.ln1_b.data(), h.data(), d);
        matvec_bf16(L.qkv.data(), h.data(), nullptr, qkv.data(), 3 * d, d);
        float* K = &s.a[(static_cast<std::size_t>(l) * ctx_) * d];
        float* V = &s.b[(static_cast<std::size_t>(l) * ctx_) * d];
        std::copy_n(&qkv[static_cast<std::size_t>(d)], d, K + static_cast<std::size_t>(p) * d);
        std::copy_n(&qkv[2 * static_cast<std::size_t>(d)], d, V + static_cast<std::size_t>(p) * d);
        for (int hh = 0; hh < heads_; ++hh) {
            const float* q = &qkv[static_cast<std::size_t>(hh) * hd];
            float mx = -1e30f;
            for (int t = 0; t <= p; ++t) {
                sc[t] = dot(q, K + static_cast<std::size_t>(t) * d + hh * hd, hd) * scale;
                mx = std::max(mx, sc[t]);
            }
            float z = 0.0f;
            for (int t = 0; t <= p; ++t) z += (sc[t] = std::exp(sc[t] - mx));
            float* o = &att[static_cast<std::size_t>(hh) * hd];
            std::fill(o, o + hd, 0.0f);
            for (int t = 0; t <= p; ++t) {
                const float w = sc[t] / z;
                const float* v = V + static_cast<std::size_t>(t) * d + hh * hd;
                for (int k = 0; k < hd; ++k) o[k] += w * v[k];
            }
        }
        matvec_bf16(L.proj.data(), att.data(), nullptr, y.data(), d, d);
        for (int k = 0; k < d; ++k) x[k] += y[k];
        layer_norm(x.data(), L.ln2_w.data(), L.ln2_b.data(), h.data(), d);
        matvec_bf16(L.fc.data(), h.data(), nullptr, f.data(), 4 * d, d);
        for (float& v : f) v = 0.5f * v * (1.0f + std::erf(v * 0.70710678f));  // exact GELU
        matvec_bf16(L.fc2.data(), f.data(), nullptr, y.data(), d, 4 * d);
        for (int k = 0; k < d; ++k) x[k] += y[k];
    }
    s.pos = p + 1;
    if (!want_logits) return;
    layer_norm(x.data(), lnf_w_.data(), lnf_b_.data(), h.data(), d);
    float logits[256];
    matvec_bf16(emb_bf_.data(), h.data(), nullptr, logits, 256, d);
    log_softmax(logits, s.log_p);
}

}  // namespace cypha::cyphalm
