#include "cypha/cyphalm/byte_lstm.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace cypha::cyphalm {

namespace {

inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// y += W x, W is rows x cols row-major. Eight independent partial sums let
// the compiler vectorise the reduction without -ffast-math (deterministic).
inline void gemv_add(const float* W, const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* w = W + static_cast<std::size_t>(r) * cols;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int k = 0;
        for (; k + 8 <= cols; k += 8)
            for (int u = 0; u < 8; ++u) acc[u] += w[k + u] * x[k + u];
        float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        for (; k < cols; ++k) s += w[k] * x[k];
        y[r] += s;
    }
}

// y += W^T x, W is rows x cols row-major, x has rows entries, y has cols.
inline void gemv_t_add(const float* W, const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* w = W + static_cast<std::size_t>(r) * cols;
        const float xr = x[r];
        if (xr == 0.0f) continue;
        for (int k = 0; k < cols; ++k) y[k] += w[k] * xr;
    }
}

// G += a b^T (rows = |a|, cols = |b|).
inline void outer_add(float* G, const float* a, const float* b, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* g = G + static_cast<std::size_t>(r) * cols;
        const float ar = a[r];
        if (ar == 0.0f) continue;
        for (int k = 0; k < cols; ++k) g[k] += ar * b[k];
    }
}

void log_softmax(const std::vector<float>& z, std::vector<float>& out) {
    float mx = z[0];
    for (float v : z) mx = std::max(mx, v);
    double s = 0.0;
    for (float v : z) s += std::exp(static_cast<double>(v - mx));
    const float lse = mx + static_cast<float>(std::log(s));
    out.resize(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) out[i] = z[i] - lse;
}

template <typename T>
void write_vec(std::ostream& os, const std::vector<T>& v) {
    const std::uint64_t n = v.size();
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    os.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

template <typename T>
void read_vec(std::istream& is, std::vector<T>& v) {
    std::uint64_t n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    v.resize(n);
    is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

}  // namespace

ByteLstm::ByteLstm(ByteLstmOptions opt) : opt_(opt), H_(opt.hidden) {
    const int H = H_, G = 4 * H;
    std::mt19937_64 rng(opt.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto init = [&](std::vector<float>& w, std::size_t n, float scale) {
        w.resize(n);
        for (auto& x : w) x = nd(rng) * scale;
    };
    const float s = 1.0f / std::sqrt(static_cast<float>(H));
    init(P_, 256u * G, 0.1f);
    init(Wh_, static_cast<std::size_t>(G) * H, s);
    init(Wy_, 256u * H, s);
    b_.assign(G, 0.0f);
    for (int j = H; j < 2 * H; ++j) b_[j] = 1.0f;  // forget-gate bias
    by_.assign(256, 0.0f);
    for (auto* p : {&mP_, &vP_}) p->assign(P_.size(), 0.0f);
    for (auto* p : {&mWh_, &vWh_}) p->assign(Wh_.size(), 0.0f);
    for (auto* p : {&mb_, &vb_}) p->assign(b_.size(), 0.0f);
    for (auto* p : {&mWy_, &vWy_}) p->assign(Wy_.size(), 0.0f);
    for (auto* p : {&mby_, &vby_}) p->assign(by_.size(), 0.0f);
    h_.assign(H, 0.0f);
    c_.assign(H, 0.0f);
    logits_.assign(256, 0.0f);
    log_softmax(logits_, logp_);
}

void ByteLstm::reset_state() {
    std::fill(h_.begin(), h_.end(), 0.0f);
    std::fill(c_.begin(), c_.end(), 0.0f);
    window_.clear();
    pending_ = -1;
    std::fill(logits_.begin(), logits_.end(), 0.0f);
    for (int k = 0; k < 256; ++k) logits_[k] = by_[k];
    gemv_add(Wy_.data(), h_.data(), logits_.data(), 256, H_);
    log_softmax(logits_, logp_);
}

void ByteLstm::forward_(int x) {
    const int H = H_, G = 4 * H;
    std::vector<float> a(b_);
    const float* px = &P_[static_cast<std::size_t>(x) * G];
    for (int k = 0; k < G; ++k) a[k] += px[k];
    gemv_add(Wh_.data(), h_.data(), a.data(), G, H);
    Step st;
    st.x = x;
    st.h_prev = h_;
    st.c_prev = c_;
    st.i.resize(H);
    st.f.resize(H);
    st.g.resize(H);
    st.o.resize(H);
    st.c.resize(H);
    st.h.resize(H);
    st.tanh_c.resize(H);
    for (int j = 0; j < H; ++j) {
        st.i[j] = sigm(a[j]);
        st.f[j] = sigm(a[H + j]);
        st.g[j] = std::tanh(a[2 * H + j]);
        st.o[j] = sigm(a[3 * H + j]);
        st.c[j] = st.f[j] * c_[j] + st.i[j] * st.g[j];
        st.tanh_c[j] = std::tanh(st.c[j]);
        st.h[j] = st.o[j] * st.tanh_c[j];
    }
    h_ = st.h;
    c_ = st.c;
    for (int k = 0; k < 256; ++k) logits_[k] = by_[k];
    gemv_add(Wy_.data(), h_.data(), logits_.data(), 256, H);
    log_softmax(logits_, logp_);
    window_.push_back(std::move(st));
    pending_ = static_cast<int>(window_.size()) - 1;
}

float ByteLstm::observe(int byte, bool learn) {
    const float loss = -logp_[static_cast<std::size_t>(byte)];
    if (!learn) {
        window_.clear();
        pending_ = -1;
    } else if (pending_ >= 0) {
        Step& st = window_[static_cast<std::size_t>(pending_)];
        st.dlogits.resize(256);
        for (int k = 0; k < 256; ++k) st.dlogits[k] = std::exp(logp_[k]);
        st.dlogits[static_cast<std::size_t>(byte)] -= 1.0f;
        if (static_cast<int>(window_.size()) >= opt_.bptt) {
            backprop_window_();
            window_.clear();
        }
    }
    forward_(byte);
    if (!learn) {
        window_.clear();
        pending_ = -1;
    }
    return loss;
}

void ByteLstm::backprop_window_() {
    const int H = H_, G = 4 * H;
    std::vector<float> gP(P_.size(), 0.0f), gWh(Wh_.size(), 0.0f), gb(b_.size(), 0.0f),
        gWy(Wy_.size(), 0.0f), gby(by_.size(), 0.0f);
    std::vector<float> dh_next(H, 0.0f), dc_next(H, 0.0f), dh(H), dc(H), da(G);
    for (int s = static_cast<int>(window_.size()) - 1; s >= 0; --s) {
        const Step& st = window_[static_cast<std::size_t>(s)];
        if (st.dlogits.empty()) continue;
        outer_add(gWy.data(), st.dlogits.data(), st.h.data(), 256, H);
        for (int k = 0; k < 256; ++k) gby[k] += st.dlogits[k];
        dh = dh_next;
        gemv_t_add(Wy_.data(), st.dlogits.data(), dh.data(), 256, H);
        for (int j = 0; j < H; ++j) {
            dc[j] = dh[j] * st.o[j] * (1.0f - st.tanh_c[j] * st.tanh_c[j]) + dc_next[j];
            const float d_o = dh[j] * st.tanh_c[j];
            const float d_i = dc[j] * st.g[j];
            const float d_g = dc[j] * st.i[j];
            const float d_f = dc[j] * st.c_prev[j];
            da[j] = d_i * st.i[j] * (1.0f - st.i[j]);
            da[H + j] = d_f * st.f[j] * (1.0f - st.f[j]);
            da[2 * H + j] = d_g * (1.0f - st.g[j] * st.g[j]);
            da[3 * H + j] = d_o * st.o[j] * (1.0f - st.o[j]);
            dc_next[j] = dc[j] * st.f[j];
        }
        outer_add(gWh.data(), da.data(), st.h_prev.data(), G, H);
        float* gpx = &gP[static_cast<std::size_t>(st.x) * G];
        for (int k = 0; k < G; ++k) {
            gb[k] += da[k];
            gpx[k] += da[k];
        }
        std::fill(dh_next.begin(), dh_next.end(), 0.0f);
        gemv_t_add(Wh_.data(), da.data(), dh_next.data(), G, H);
    }
    double norm2 = 0.0;
    for (const auto* g : {&gP, &gWh, &gb, &gWy, &gby})
        for (float v : *g) norm2 += static_cast<double>(v) * v;
    const double norm = std::sqrt(norm2);
    if (norm > opt_.clip) {
        const float sc = static_cast<float>(opt_.clip / norm);
        for (auto* g : {&gP, &gWh, &gb, &gWy, &gby})
            for (float& v : *g) v *= sc;
    }
    ++adam_t_;
    adam_(P_, mP_, vP_, gP);
    adam_(Wh_, mWh_, vWh_, gWh);
    adam_(b_, mb_, vb_, gb);
    adam_(Wy_, mWy_, vWy_, gWy);
    adam_(by_, mby_, vby_, gby);
}

void ByteLstm::adam_(std::vector<float>& w, std::vector<float>& m, std::vector<float>& v,
                     const std::vector<float>& g) {
    constexpr float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const float c1 = 1.0f - std::pow(b1, static_cast<float>(adam_t_));
    const float c2 = 1.0f - std::pow(b2, static_cast<float>(adam_t_));
    const float lr = opt_.lr;
    for (std::size_t k = 0; k < w.size(); ++k) {
        if (g[k] == 0.0f && m[k] == 0.0f) continue;
        m[k] = b1 * m[k] + (1.0f - b1) * g[k];
        v[k] = b2 * v[k] + (1.0f - b2) * g[k] * g[k];
        w[k] -= lr * (m[k] / c1) / (std::sqrt(v[k] / c2) + eps);
    }
}

void ByteLstm::write(std::ostream& os) const {
    const std::int32_t hdr[3] = {H_, opt_.bptt, 2};
    os.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    for (const auto* w : {&P_, &Wh_, &b_, &Wy_, &by_, &h_, &c_}) write_vec(os, *w);
}

void ByteLstm::read(std::istream& is) {
    std::int32_t hdr[3] = {};
    is.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (hdr[0] != H_) throw std::runtime_error("ByteLstm::read: hidden size mismatch");
    if (hdr[2] == 1) {
        // v1 stored embedding E (256 x H) and input weights Wx (4H x H); fold them
        // into the per-byte table P[x] = Wx E[x] (exact).
        std::vector<float> E, Wx;
        read_vec(is, E);
        read_vec(is, Wx);
        const int G = 4 * H_;
        for (int x = 0; x < 256; ++x) {
            float* px = &P_[static_cast<std::size_t>(x) * G];
            std::fill(px, px + G, 0.0f);
            gemv_add(Wx.data(), &E[static_cast<std::size_t>(x) * H_], px, G, H_);
        }
        for (auto* w : {&Wh_, &b_, &Wy_, &by_, &h_, &c_}) read_vec(is, *w);
    } else {
        for (auto* w : {&P_, &Wh_, &b_, &Wy_, &by_, &h_, &c_}) read_vec(is, *w);
    }
    window_.clear();
    pending_ = -1;
    for (int k = 0; k < 256; ++k) logits_[k] = by_[k];
    gemv_add(Wy_.data(), h_.data(), logits_.data(), 256, H_);
    log_softmax(logits_, logp_);
}

ByteMixGate::ByteMixGate() {
    theta_[0] = 1.5;  // start trusting hp (w ~ 0.82)
    for (int k = 1; k < kF; ++k) theta_[k] = 0.0;
}

const std::vector<double>& ByteMixGate::mix(const std::vector<double>& log_a,
                                            const std::vector<float>& log_b) {
    const std::size_t n = log_a.size();
    pa_.resize(n);
    pb_.resize(n);
    out_.resize(n);
    double ha = 0.0, hb = 0.0, ma = -1e300, mb = -1e300;
    for (std::size_t k = 0; k < n; ++k) {
        pa_[k] = std::exp(log_a[k]);
        pb_[k] = std::exp(static_cast<double>(log_b[k]));
        if (pa_[k] > 0) ha -= pa_[k] * log_a[k];
        if (pb_[k] > 0) hb -= pb_[k] * static_cast<double>(log_b[k]);
        ma = std::max(ma, log_a[k]);
        mb = std::max(mb, static_cast<double>(log_b[k]));
    }
    constexpr double kLn2 = 0.6931471805599453;
    feat_[0] = 1.0;
    feat_[1] = ha / kLn2 / 8.0;
    feat_[2] = hb / kLn2 / 8.0;
    feat_[3] = ma / 5.0;
    feat_[4] = mb / 5.0;
    double z = 0.0;
    for (int k = 0; k < kF; ++k) z += theta_[k] * feat_[k];
    w_ = 1.0 / (1.0 + std::exp(-z));
    for (std::size_t k = 0; k < n; ++k)
        out_[k] = std::log(std::max(w_ * pa_[k] + (1.0 - w_) * pb_[k], 1e-300));
    return out_;
}

void ByteMixGate::learn(int byte, double lr) {
    const std::size_t y = static_cast<std::size_t>(byte);
    const double pm = std::max(w_ * pa_[y] + (1.0 - w_) * pb_[y], 1e-300);
    const double dz = -(pa_[y] - pb_[y]) * w_ * (1.0 - w_) / pm;
    for (int k = 0; k < kF; ++k) theta_[k] -= lr * dz * feat_[k];
}

void ByteMixGate::write(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(theta_), sizeof(theta_));
}

void ByteMixGate::read(std::istream& is) {
    is.read(reinterpret_cast<char*>(theta_), sizeof(theta_));
}

}  // namespace cypha::cyphalm
