#pragma once

/// The decoder's learner (the encoder side of CyphaLM's decoder; docs/DECODER.md, docs/reports/K5_MIXDEC.md).
/// Odin Loch, 2026-11-12.
///
/// A lighter configuration of the context8k stack that gives the WHOLE next-byte distribution at any point of a text
/// and can be copied cheaply, so a decoder can score hypothetical continuations and throw them away. Experts:
///   - the 1 TB trie (PackedModel, Mode::Synth): its five order experts and its mixture with the match (mix_dist);
///   - byte Kneser-Ney 6:4:2:1 (BKN2), counted on K5's clean prose;
///   - the copy expert over the text the learner has read (article, prompt, text so far);
///   - the online n-gram expert (OnlineNgram, orders 1,2,3,4,6), learning only what the learner is primed on;
///   - the byte LSTM (hidden 64), pretrained on K5's clean prose by lstm-train, learning online while primed.
/// Mixer: a geometric mixer over the experts' log distributions (the full mix's arm B, FULL_DISTRIBUTION_MIXING.md),
/// 40 weight sets by copy's match bucket x the previous byte's class, learned online by gradient on log loss.
/// Left out, and why: InfiniGram's caps, the class and similar-context experts, the document cache, the online chain, the
/// final layer, the bitwise mixers and the meta mixer. The bitwise mixers give one byte's probability along one path, so
/// a whole distribution costs 255 node evaluations; the similar-context expert is the serial bottleneck of the full run;
/// none of them has a way to rewind a hypothetical byte. Rule 10: every expert is CyphaLM's own; the byte LSTM is
/// CyphaLM's own online learner, which context8k already runs.
///
/// Learning: the learner learns (mixer weights, n-gram counts, LSTM) only while it is primed: once on a slice of generic
/// clean prose (the base), then per continuation on the retrieved article and the prompt. While writing it is FROZEN: it
/// reads its own text (the experts' contexts move) but learns nothing from it (GEN-C: learning from its own output loops).

#include "cypha/lm/byte_kn.hpp"
#include "cypha/lm/copy_expert.hpp"
#include "cypha/lm/packed_trie.hpp"
#include "cypha/online/byte_lstm.hpp"
#include "cypha/online/mix_core.hpp"
#include "cypha/online/online_ngram.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

namespace cypha::cyphalm::decoder {

/// What the learner reads that does not change: shared by every thread, read-only.
struct Models {
    const trie::PackedModel* trie = nullptr;  // null: no trie experts
    double tw[6] = {};                        // the trie's mixture weights (from its file)
    const bkn::ByteKN* kn = nullptr;          // null: no Kneser-Ney
    bool copy = true, ngram = true, lstm = true;
    double lr = 0.002;    // the geometric mixer's step
    double floor = 1e-6;  // each input distribution's floor before its log
};

class Learner {
 public:
    static constexpr int kIn = 10;   // trie x5, trie mixture, Kneser-Ney, copy, n-gram, LSTM
    static constexpr int kSets = 40;  // copy bucket (8) x previous byte's class (5)
    static constexpr int kTM = 5;     // the input the weights start on (the trie mixture; Kneser-Ney without a trie)

    Learner(const Models& m, const ::cypha::online::ByteLSTM* lstm0) : M_(&m) {
        if (m.lstm && lstm0) L_ = std::make_unique<::cypha::online::ByteLSTM>(*lstm0), L_->set_threads(1);
        if (m.ngram) {
            ::cypha::online::OnlineNgram::Params np;
            np.orders = {1, 2, 3, 4, 6};
            np.tb = 18, np.cap = 60, np.lr = 0.005;
            ng_ = std::make_shared<::cypha::online::OnlineNgram>(np);
            nh_ = ng_->history();
        }
        w_.assign(static_cast<std::size_t>(kSets) * kIn, 0.0);
        const int init = m.trie ? kTM : 6;
        for (int g = 0; g < kSets; ++g) w_[static_cast<std::size_t>(g) * kIn + init] = 1.0;
        buf_.push_back('\n');
    }
    Learner(const Learner& o) { *this = o; }
    Learner& operator=(const Learner& o) {
        if (this == &o) return *this;
        M_ = o.M_;
        buf_ = o.buf_;
        L_ = o.L_ ? std::make_unique<::cypha::online::ByteLSTM>(*o.L_) : nullptr;
        if (L_) L_->set_threads(1);
        ng_ = o.ng_;  // shared: a copy reads the same counts (frozen); own() makes them its own
        nh_ = o.nh_;
        w_ = o.w_;
        learned_ = o.learned_;
        return *this;
    }
    /// The n-gram's counts become this learner's own copy (before it learns, or when another thread reads the original).
    void own() {
        if (ng_) ng_ = std::make_shared<::cypha::online::OnlineNgram>(*ng_);
    }

    /// The next byte's distribution after everything read, learning nothing. The inputs are kept for learn().
    void dist(double* P) {
        const std::uint8_t* h = buf_.data();
        const std::size_t n = buf_.size();
        int k = 0;
        auto put = [&](const double* d) {  // normalised, floored, logged; null or empty: skipped
            double z = 0;
            if (d)
                for (int s = 0; s < 256; ++s) z += d[s];
            if (!d || !(z > 0)) {
                has_[k++] = false;
                return;
            }
            double* L = lin_[k];
            for (int s = 0; s < 256; ++s) L[s] = std::log(std::max(d[s] / z, M_->floor));
            has_[k++] = true;
        };
        if (M_->trie) {
            double e[5][256], TM[256];
            bool em = false;
            unsigned pred = 0;
            M_->trie->fill_experts(h, n, std::numeric_limits<std::size_t>::max(), trie::Mode::Synth, e, em, pred, false);
            trie::mix_dist(M_->tw, e, em, pred, TM);
            for (int j = 0; j < 5; ++j) put(e[j]);
            put(TM);
        } else {
            for (int j = 0; j < 6; ++j) put(nullptr);
        }
        if (M_->kn) {
            double K[256];
            M_->kn->dist(h, n, K);
            put(K);
        } else {
            put(nullptr);
        }
        int cn = 0;
        if (M_->copy) {
            const int cp = copy_pred(h, n, cn);
            if (cp >= 0) {
                double C[256] = {};
                C[cp] = 1.0;
                put(C);
            } else {
                put(nullptr);
            }
        } else {
            put(nullptr);
        }
        if (ng_) {
            double G[256], Go[5 * 256];
            ng_->set_history(nh_);
            ng_->dist(G, Go);
            put(G);
        } else {
            put(nullptr);
        }
        if (L_) {
            double Q[256];
            L_->dist(Q);
            put(Q);
        } else {
            put(nullptr);
        }
        set_ = copy_bucket(cn) * 5 + ::cypha::online::mix::byte_class(n ? h[n - 1] : 32);
        const double* w = &w_[static_cast<std::size_t>(set_) * kIn];
        double z[256] = {};
        for (int j = 0; j < kIn; ++j)
            if (has_[j] && w[j] != 0)
                for (int s = 0; s < 256; ++s) z[s] += w[j] * lin_[j][s];
        double mx = z[0];
        for (int s = 1; s < 256; ++s) mx = std::max(mx, z[s]);
        double t = 0;
        for (int s = 0; s < 256; ++s) t += (P[s] = std::exp(z[s] - mx));
        for (int s = 0; s < 256; ++s) P_[s] = (P[s] /= t);
    }

    /// Reads byte y and learns it (priming only): the mixer's gradient on log loss, the n-gram's counts, the LSTM.
    /// Returns log2 P(y).
    double learn(std::uint8_t y) {
        double P[256];
        dist(P);
        double* w = &w_[static_cast<std::size_t>(set_) * kIn];
        for (int j = 0; j < kIn; ++j) {
            if (!has_[j]) continue;
            double e = 0;
            for (int s = 0; s < 256; ++s) e += P[s] * lin_[j][s];
            w[j] -= M_->lr * (e - lin_[j][y]);
        }
        if (ng_) {
            double po[8];
            ng_->set_history(nh_);
            ng_->code(y, po, true);
            nh_ = ng_->history();
        }
        if (L_) L_->step(y, true);
        buf_.push_back(y);
        ++learned_;
        return std::log2(std::max(P[y], 1e-300));
    }
    double learn(std::string_view s) {
        double b = 0;
        for (const char c : s) b += learn(static_cast<std::uint8_t>(c));
        return b;
    }

    /// Reads byte y without learning (writing, and the decoder's hypothetical bytes).
    void read(std::uint8_t y) {
        if (ng_) {
            nh_.erase(nh_.begin());
            nh_.push_back(y);
        }
        if (L_) L_->step(y, false);
        buf_.push_back(y);
    }
    void read(std::string_view s) {
        for (const char c : s) read(static_cast<std::uint8_t>(c));
    }

    /// A mark to come back to after reading hypothetical bytes (rewind), and the rewind. Exact: the frozen learner's
    /// state is the text read, the LSTM's recurrent state and the n-gram's history.
    struct Mark {
        std::size_t n = 0;
        ::cypha::online::ByteLSTM::State ls{};
        std::vector<std::uint8_t> nh;
    };
    Mark mark() const {
        Mark m;
        m.n = buf_.size();
        if (L_) m.ls = L_->save();
        m.nh = nh_;
        return m;
    }
    void rewind(const Mark& m) {
        buf_.resize(m.n);
        if (L_) L_->restore(m.ls);
        nh_ = m.nh;
    }

    /// Keeps the last 32 KiB of the text read (the copy expert's window is 16 KiB; the trie and Kneser-Ney read a few
    /// bytes), so copies stay small. The predictions do not change. Never between mark() and rewind().
    void trim() {
        if (buf_.size() > 65536) buf_.erase(buf_.begin(), buf_.end() - 32768);
    }
    std::size_t size() const { return buf_.size(); }
    std::uint64_t learned() const { return learned_; }

 private:
    const Models* M_ = nullptr;
    std::vector<std::uint8_t> buf_;
    std::unique_ptr<::cypha::online::ByteLSTM> L_;
    std::shared_ptr<::cypha::online::OnlineNgram> ng_;
    std::vector<std::uint8_t> nh_;
    std::vector<double> w_;
    std::uint64_t learned_ = 0;
    // the last dist()'s inputs, for learn()
    double lin_[kIn][256] = {};
    bool has_[kIn] = {};
    int set_ = 0;
    double P_[256] = {};
};

}  // namespace cypha::cyphalm::decoder
