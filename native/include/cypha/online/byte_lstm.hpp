// A byte LSTM learning online (Odin Loch, 2026-11-04; docs/GENERATION_PLAN.md §13.18). Embedding 16, hidden 64, a
// softmax over the 256 next bytes; truncated BPTT over 20 bytes and Adam (step 0.003). It predicts each byte from the
// state that has read every byte before it, then learns it. Used inside the bitwise GLN (bitwise_gln.hpp) and on its
// own as a content-mixer expert (cyphalm_trie --lstm), as PAQ8 and cmix use an LSTM.
// Rule 10 (PLAN): an expert inside CyphaLM. It answers nothing.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cypha::online {

// Binary save and load of vectors and plain values (the GLN's file, "GLN1": a learned model can be trained once, at
// scale, and read by every run, §13.17).
namespace gln_io {
template <class T>
void put(std::ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof v); }
template <class T>
void get(std::istream& i, T& v) { i.read(reinterpret_cast<char*>(&v), sizeof v); }
template <class T>
void put_vec(std::ostream& o, const std::vector<T>& v) {
    put(o, static_cast<std::uint64_t>(v.size()));
    o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}
template <class T>
void get_vec(std::istream& i, std::vector<T>& v) {
    std::uint64_t n = 0;
    get(i, n);
    v.resize(static_cast<std::size_t>(n));
    i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}
}  // namespace gln_io

inline double gln_squash(double x) { return 1.0 / (1.0 + std::exp(-std::clamp(x, -30.0, 30.0))); }
inline double gln_stretch(double p) {
    p = std::clamp(p, 1e-4, 1.0 - 1e-4);
    return std::log(p / (1.0 - p));
}

// A byte LSTM: embedding 16, hidden 64, softmax over 256; learns online by truncated BPTT (window 20) and Adam.
class ByteLSTM {
public:
    static constexpr int H = 64, E = 16, X = E + H, T = 20;

    explicit ByteLSTM(std::uint64_t seed = 0, double lr = 0.003) : lr_(lr) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        auto fill = [&](std::vector<double>& v, std::size_t n, double s) {
            v.resize(n);
            for (auto& x : v) x = nd(rng) * s;
        };
        fill(Em_, 256 * E, 0.1);
        fill(W_, 4 * H * X, 0.1);
        b_.assign(4 * H, 0.0);
        for (int k = H; k < 2 * H; ++k) b_[k] = 1.0;  // forget gate bias
        fill(Wo_, 256 * H, 0.1);
        bo_.assign(256, 0.0);
        for (auto* p : {&Em_, &W_, &b_, &Wo_, &bo_}) {
            m_.emplace_back(p->size(), 0.0);
            v_.emplace_back(p->size(), 0.0);
        }
        h_.fill(0.0);
        c_.fill(0.0);
        cache_ = forward(32);  // the state after reading a space predicts the first byte
    }

    const std::array<double, H>& h() const { return h_; }

    // The distribution of the next byte from the state that has read every byte so far.
    void dist(double* p) const {
        double mx = -1e300;
        for (int y = 0; y < 256; ++y) {
            double z = bo_[y];
            for (int k = 0; k < H; ++k) z += Wo_[static_cast<std::size_t>(y) * H + k] * h_[k];
            p[y] = z;
            mx = std::max(mx, z);
        }
        double s = 0;
        for (int y = 0; y < 256; ++y) s += (p[y] = std::exp(p[y] - mx));
        for (int y = 0; y < 256; ++y) p[y] /= s;
    }

    // y followed the bytes read so far: learn that (when learn), then read y.
    void step(std::uint8_t y, bool learn) {
        if (learn) {
            cache_.y = y;
            buf_.push_back(cache_);
            if (static_cast<int>(buf_.size()) >= T) bptt();
        }
        cache_ = forward(y);
    }

    struct State {
        std::array<double, H> h, c;
    };
    void write(std::ostream& o) const {
        using namespace gln_io;
        for (const auto* v : {&Em_, &W_, &b_, &Wo_, &bo_}) put_vec(o, *v);
        for (const auto& v : m_) put_vec(o, v);
        for (const auto& v : v_) put_vec(o, v);
        put(o, t_);
        put(o, h_);
        put(o, c_);
    }
    void read(std::istream& i) {
        using namespace gln_io;
        for (auto* v : {&Em_, &W_, &b_, &Wo_, &bo_}) get_vec(i, *v);
        for (auto& v : m_) get_vec(i, v);
        for (auto& v : v_) get_vec(i, v);
        get(i, t_);
        get(i, h_);
        get(i, c_);
        buf_.clear();
        cache_ = forward(32);  // a fresh read of a space after the stored state
    }
    // §13.24: the learned LSTM to a file and back ("LST1"), so it can learn a text once (cyphalm_trie lstm-train) and be read
    // by any run.
    bool save_file(const std::string& path) const {
        std::ofstream o(path, std::ios::binary);
        o.write("LST1", 4);
        write(o);
        return static_cast<bool>(o);
    }
    static std::unique_ptr<ByteLSTM> load_file(const std::string& path) {
        std::ifstream i(path, std::ios::binary);
        char magic[4] = {0, 0, 0, 0};
        i.read(magic, 4);
        if (!i || std::memcmp(magic, "LST1", 4) != 0) return nullptr;
        auto L = std::make_unique<ByteLSTM>();
        L->read(i);
        return i ? std::move(L) : nullptr;
    }
    State save() const { return {h_, c_}; }
    // Restores the recurrent state (for writing: no learning happens between save and restore).
    void restore(const State& s) {
        h_ = s.h;
        c_ = s.c;
    }

private:
    struct Step {
        std::uint8_t x = 0, y = 0;
        std::array<double, X> xh;
        std::array<double, H> i, f, o, g, c_prev, c, h;
    };

    Step forward(std::uint8_t xb) {
        Step s;
        s.x = xb;
        for (int k = 0; k < E; ++k) s.xh[k] = Em_[static_cast<std::size_t>(xb) * E + k];
        for (int k = 0; k < H; ++k) s.xh[E + k] = h_[k];
        for (int r = 0; r < 4 * H; ++r) {
            double a = b_[r];
            const double* w = &W_[static_cast<std::size_t>(r) * X];
            for (int k = 0; k < X; ++k) a += w[k] * s.xh[k];
            const int q = r / H, k = r % H;
            if (q == 0) s.i[k] = 1 / (1 + std::exp(-a));
            else if (q == 1) s.f[k] = 1 / (1 + std::exp(-a));
            else if (q == 2) s.o[k] = 1 / (1 + std::exp(-a));
            else s.g[k] = std::tanh(a);
        }
        for (int k = 0; k < H; ++k) {
            s.c_prev[k] = c_[k];
            s.c[k] = s.f[k] * c_[k] + s.i[k] * s.g[k];
            s.h[k] = s.o[k] * std::tanh(s.c[k]);
        }
        h_ = s.h;
        c_ = s.c;
        return s;
    }

    void bptt() {
        std::vector<std::vector<double>> G;
        for (auto* p : {&Em_, &W_, &b_, &Wo_, &bo_}) G.emplace_back(p->size(), 0.0);
        auto &gE = G[0], &gW = G[1], &gb = G[2], &gWo = G[3], &gbo = G[4];
        std::array<double, H> dh_next{}, dc_next{};
        double p[256];
        for (auto it = buf_.rbegin(); it != buf_.rend(); ++it) {
            const Step& s = *it;
            double mx = -1e300;
            for (int y = 0; y < 256; ++y) {
                double z = bo_[y];
                for (int k = 0; k < H; ++k) z += Wo_[static_cast<std::size_t>(y) * H + k] * s.h[k];
                p[y] = z;
                mx = std::max(mx, z);
            }
            double sum = 0;
            for (int y = 0; y < 256; ++y) sum += (p[y] = std::exp(p[y] - mx));
            for (int y = 0; y < 256; ++y) p[y] /= sum;
            p[s.y] -= 1;
            std::array<double, H> dh = dh_next;
            for (int y = 0; y < 256; ++y) {
                gbo[y] += p[y];
                for (int k = 0; k < H; ++k) {
                    gWo[static_cast<std::size_t>(y) * H + k] += p[y] * s.h[k];
                    dh[k] += Wo_[static_cast<std::size_t>(y) * H + k] * p[y];
                }
            }
            std::array<double, 4 * H> da;
            for (int k = 0; k < H; ++k) {
                const double tc = std::tanh(s.c[k]);
                const double d_o = dh[k] * tc * s.o[k] * (1 - s.o[k]);
                const double dc = dh[k] * s.o[k] * (1 - tc * tc) + dc_next[k];
                da[k] = dc * s.g[k] * s.i[k] * (1 - s.i[k]);
                da[H + k] = dc * s.c_prev[k] * s.f[k] * (1 - s.f[k]);
                da[2 * H + k] = d_o;
                da[3 * H + k] = dc * s.i[k] * (1 - s.g[k] * s.g[k]);
                dc_next[k] = dc * s.f[k];
            }
            std::array<double, X> dxh{};
            for (int r = 0; r < 4 * H; ++r) {
                gb[r] += da[r];
                const double* w = &W_[static_cast<std::size_t>(r) * X];
                double* gw = &gW[static_cast<std::size_t>(r) * X];
                for (int k = 0; k < X; ++k) {
                    gw[k] += da[r] * s.xh[k];
                    dxh[k] += w[k] * da[r];
                }
            }
            for (int k = 0; k < E; ++k) gE[static_cast<std::size_t>(s.x) * E + k] += dxh[k];
            for (int k = 0; k < H; ++k) dh_next[k] = dxh[E + k];
        }
        ++t_;
        const double n = static_cast<double>(buf_.size());
        const double c1 = 1 - std::pow(0.9, t_), c2 = 1 - std::pow(0.999, t_);
        std::vector<double>* P[5] = {&Em_, &W_, &b_, &Wo_, &bo_};
        for (int q = 0; q < 5; ++q)
            for (std::size_t j = 0; j < P[q]->size(); ++j) {
                const double g = std::clamp(G[q][j] / n, -5.0, 5.0);
                m_[q][j] = 0.9 * m_[q][j] + 0.1 * g;
                v_[q][j] = 0.999 * v_[q][j] + 0.001 * g * g;
                (*P[q])[j] -= lr_ * (m_[q][j] / c1) / (std::sqrt(v_[q][j] / c2) + 1e-8);
            }
        buf_.clear();
    }

    double lr_;
    int t_ = 0;
    std::vector<double> Em_, W_, b_, Wo_, bo_;
    std::vector<std::vector<double>> m_, v_;
    std::array<double, H> h_, c_;
    Step cache_;
    std::vector<Step> buf_;
};

}  // namespace cypha::online
