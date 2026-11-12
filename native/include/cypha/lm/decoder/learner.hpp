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
/// MIXDEC_TUNE.md (2026-11-13), all off by default (the learner is then the same to the last bit):
///   - the class expert (ClassContext, word shapes, counted on K5's prose): a pure function of the text read, so it
///     rewinds for free;
///   - the document cache (DocTopic's cache part, cand 2048 as T1 tuned it): fed while primed, frozen before writing;
///     a frozen stream's state is its partial word and last bytes, which a mark keeps;
///   - the n-gram's orders, table bits, count cap and step (T1's tuned 1..8,12,16, cap 8, step 0.000625), and counts
///     pretrained on generic prose (OnlineNgram::pretrain);
///   - the mixer's floor, step and step decay (T1's tuned arm B: 1.67652e-05, 0.00181686, 0.0001);
///   - a second mixer, its weight sets by the previous byte (mix2=1) or an order-2 hash (mix2=2), joined to the first
///     by a final geometric mixer (5 sets by the previous byte's class) that starts on the first mixer alone.
///
/// Learning: the learner learns (mixer weights, n-gram counts, LSTM) only while it is primed: once on a slice of generic
/// clean prose (the base), then per continuation on the retrieved article and the prompt. While writing it is FROZEN: it
/// reads its own text (the experts' contexts move) but learns nothing from it (GEN-C: learning from its own output loops).

#include "cypha/lm/byte_kn.hpp"
#include "cypha/lm/class_context.hpp"
#include "cypha/lm/copy_expert.hpp"
#include "cypha/lm/doc_cache.hpp"
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
    // MIXDEC_TUNE.md (off by default)
    const ClassContext* cls = nullptr;  // the class expert
    const DocTopic* doc = nullptr;      // the document cache (its cache part)
    std::vector<int> ng_orders{1, 2, 3, 4, 6};
    int ng_tb = 18, ng_cap = 60;
    double ng_lr = 0.005;
    double decay = 0;  // the mixers' step at a weight set's t-th update: lr / (1 + t decay)
    int mix2 = 0;      // 0: one mixer; 1: a second by the previous byte; 2: by an order-2 hash; then a final mixer
};

class Learner {
 public:
    static constexpr int kIn = 12;    // trie x5, trie mixture, Kneser-Ney, copy, n-gram, LSTM, class, document cache
    static constexpr int kSets = 40;  // copy bucket (8) x previous byte's class (5)
    static constexpr int kTM = 5;     // the input the weights start on (the trie mixture; Kneser-Ney without a trie)
    static constexpr int kMaxOrders = 16;

    Learner(const Models& m, const ::cypha::online::ByteLSTM* lstm0) : M_(&m) {
        if (m.lstm && lstm0) L_ = std::make_unique<::cypha::online::ByteLSTM>(*lstm0), L_->set_threads(1);
        if (m.ngram) {
            ::cypha::online::OnlineNgram::Params np;
            np.orders = m.ng_orders;
            np.tb = m.ng_tb, np.cap = m.ng_cap, np.lr = m.ng_lr;
            ng_ = std::make_shared<::cypha::online::OnlineNgram>(np);
            nh_ = ng_->history();
        }
        if (m.doc) ds_ = std::make_shared<DocTopic::Stream>(*m.doc), dp_ = ds_->pos();
        const int init = m.trie ? kTM : 6;
        w_.assign(static_cast<std::size_t>(kSets) * kIn, 0.0);
        for (int g = 0; g < kSets; ++g) w_[static_cast<std::size_t>(g) * kIn + init] = 1.0;
        if (m.decay > 0) n_.assign(kSets, 0);
        if (m.mix2 > 0) {
            const int s2 = sets2();
            w2_.assign(static_cast<std::size_t>(s2) * kIn, 0.0);
            for (int g = 0; g < s2; ++g) w2_[static_cast<std::size_t>(g) * kIn + init] = 1.0;
            wf_.assign(5 * 2, 0.0);
            for (int g = 0; g < 5; ++g) wf_[static_cast<std::size_t>(g) * 2] = 1.0;  // the final mixer starts on mixer 1
            if (m.decay > 0) n2_.assign(static_cast<std::size_t>(s2), 0), nf_.assign(5, 0);
        }
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
        ds_ = o.ds_;  // shared likewise (a frozen stream changes only its position, which each learner keeps)
        dp_ = o.dp_;
        w_ = o.w_;
        w2_ = o.w2_, wf_ = o.wf_;
        n_ = o.n_, n2_ = o.n2_, nf_ = o.nf_;
        learned_ = o.learned_;
        return *this;
    }
    /// The n-gram's counts (and the document stream) become this learner's own copy (before it learns, or when another
    /// thread reads the original).
    void own() {
        if (ng_) ng_ = std::make_shared<::cypha::online::OnlineNgram>(*ng_);
        if (ds_) ds_ = std::make_shared<DocTopic::Stream>(*ds_);
    }
    /// MIXDEC_TUNE.md: the n-gram's counts of generic text, counted alone (no mixer, no other expert).
    void pretrain_ngram(std::string_view s) {
        if (!ng_ || s.empty()) return;
        ng_->set_history(nh_);
        ng_->pretrain(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
        nh_ = ng_->history();
    }
    /// MIXDEC_TUNE.md: the document cache starts a new document (the base's generic prose is not one document).
    void doc_reset() {
        if (!ds_) return;
        ds_ = std::make_shared<DocTopic::Stream>(*M_->doc);
        dp_ = ds_->pos();
    }
    /// MIXDEC_TUNE.md: the document cache stops learning (before writing); its candidates are fixed here.
    void doc_freeze() {
        if (!ds_) return;
        ds_->freeze();
        double D[256];
        ds_->set_pos(dp_);
        ds_->predict(DocTopic::kCache, D);
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
            double G[256], Go[kMaxOrders * 256];
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
        if (M_->cls) {  // MIXDEC_TUNE.md: skipped where no context of it was seen
            double C[256];
            const auto hit = M_->cls->predict(h, n, C);
            put(hit.order > 0 ? C : nullptr);
        } else {
            put(nullptr);
        }
        if (ds_) {  // MIXDEC_TUNE.md: skipped where no candidate word fits (bucket 0)
            double D[256];
            ds_->set_pos(dp_);
            const int b = ds_->predict(DocTopic::kCache, D);
            put(b > 0 ? D : nullptr);
        } else {
            put(nullptr);
        }
        const std::uint8_t pb = n ? h[n - 1] : 32;
        set_ = copy_bucket(cn) * 5 + ::cypha::online::mix::byte_class(pb);
        if (M_->mix2 > 0) {  // MIXDEC_TUNE.md: the second mixer and the final one, on the mixers' log probabilities
            const std::uint8_t pb2 = n > 1 ? h[n - 2] : 32;
            set2_ = M_->mix2 == 1 ? pb : static_cast<int>(((pb2 * 0x9E37u) ^ (pb * 0x85EBu) ^ pb) & 4095u);
            cls_ = ::cypha::online::mix::byte_class(pb);
            mixed_log(&w_[static_cast<std::size_t>(set_) * kIn], P1_);
            mixed_log(&w2_[static_cast<std::size_t>(set2_) * kIn], P2_);
            const double* f = &wf_[static_cast<std::size_t>(cls_) * 2];
            double z[256], mx = -1e300;
            for (int s = 0; s < 256; ++s) {
                z[s] = f[0] * P1_[s] + f[1] * P2_[s];
                mx = std::max(mx, z[s]);
            }
            double t = 0;
            for (int s = 0; s < 256; ++s) t += (P[s] = std::exp(z[s] - mx));
            for (int s = 0; s < 256; ++s) P[s] /= t;
        } else {
            mixed(&w_[static_cast<std::size_t>(set_) * kIn], P);
        }
        for (int s = 0; s < 256; ++s) P_[s] = P[s];
    }

    /// Reads byte y and learns it (priming only): the mixer's gradient on log loss, the n-gram's counts, the LSTM.
    /// Returns log2 P(y).
    double learn(std::uint8_t y) {
        double P[256];
        dist(P);
        if (M_->mix2 > 0) {
            double Q1[256], Q2[256];  // the mixers' log probabilities (P1_, P2_) as probabilities, for their steps
            for (int s = 0; s < 256; ++s) Q1[s] = std::exp(P1_[s]), Q2[s] = std::exp(P2_[s]);
            step(&w_[static_cast<std::size_t>(set_) * kIn], Q1, y, n_.empty() ? nullptr : &n_[static_cast<std::size_t>(set_)]);
            step(&w2_[static_cast<std::size_t>(set2_) * kIn], Q2, y, n2_.empty() ? nullptr : &n2_[static_cast<std::size_t>(set2_)]);
            double* f = &wf_[static_cast<std::size_t>(cls_) * 2];
            const double* in[2] = {P1_, P2_};
            const double lr = rate(nf_.empty() ? nullptr : &nf_[static_cast<std::size_t>(cls_)]);
            for (int j = 0; j < 2; ++j) {
                double e = 0;
                for (int s = 0; s < 256; ++s) e += P[s] * in[j][s];
                f[j] -= lr * (e - in[j][y]);
            }
        } else {
            step(&w_[static_cast<std::size_t>(set_) * kIn], P, y, n_.empty() ? nullptr : &n_[static_cast<std::size_t>(set_)]);
        }
        if (ng_) {
            double po[kMaxOrders];
            ng_->set_history(nh_);
            ng_->code(y, po, true);
            nh_ = ng_->history();
        }
        if (L_) L_->step(y, true);
        if (ds_) {
            ds_->set_pos(dp_);
            ds_->feed(y);
            dp_ = ds_->pos();
        }
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
        if (ds_) {
            ds_->set_pos(dp_);
            ds_->feed(y);
            dp_ = ds_->pos();
        }
        buf_.push_back(y);
    }
    void read(std::string_view s) {
        for (const char c : s) read(static_cast<std::uint8_t>(c));
    }

    /// A mark to come back to after reading hypothetical bytes (rewind), and the rewind. Exact: the frozen learner's
    /// state is the text read, the LSTM's recurrent state, the n-gram's history and the document stream's position.
    struct Mark {
        std::size_t n = 0;
        ::cypha::online::ByteLSTM::State ls{};
        std::vector<std::uint8_t> nh;
        DocTopic::Stream::Pos dp;
    };
    Mark mark() const {
        Mark m;
        m.n = buf_.size();
        if (L_) m.ls = L_->save();
        m.nh = nh_;
        if (ds_) m.dp = dp_;
        return m;
    }
    void rewind(const Mark& m) {
        buf_.resize(m.n);
        if (L_) L_->restore(m.ls);
        nh_ = m.nh;
        if (ds_) dp_ = m.dp;
    }

    /// Keeps the last 32 KiB of the text read (the copy expert's window is 16 KiB; the trie and Kneser-Ney read a few
    /// bytes), so copies stay small. The predictions do not change. Never between mark() and rewind().
    void trim() {
        if (buf_.size() > 65536) buf_.erase(buf_.begin(), buf_.end() - 32768);
    }
    std::size_t size() const { return buf_.size(); }
    std::uint64_t learned() const { return learned_; }

 private:
    int sets2() const { return M_->mix2 == 1 ? 256 : 4096; }
    // One mixer: the geometric mix of the inputs present with the weights w.
    void mixed(const double* w, double* P) const {
        double z[256] = {};
        for (int j = 0; j < kIn; ++j)
            if (has_[j] && w[j] != 0)
                for (int s = 0; s < 256; ++s) z[s] += w[j] * lin_[j][s];
        double mx = z[0];
        for (int s = 1; s < 256; ++s) mx = std::max(mx, z[s]);
        double t = 0;
        for (int s = 0; s < 256; ++s) t += (P[s] = std::exp(z[s] - mx));
        for (int s = 0; s < 256; ++s) P[s] /= t;
    }
    // The same mixer's log probabilities (z - max - log of the sum): no exp-then-log for the final mixer.
    void mixed_log(const double* w, double* LP) const {
        double z[256] = {};
        for (int j = 0; j < kIn; ++j)
            if (has_[j] && w[j] != 0)
                for (int s = 0; s < 256; ++s) z[s] += w[j] * lin_[j][s];
        double mx = z[0];
        for (int s = 1; s < 256; ++s) mx = std::max(mx, z[s]);
        double t = 0;
        for (int s = 0; s < 256; ++s) t += std::exp(z[s] - mx);
        const double lt = mx + std::log(t);
        for (int s = 0; s < 256; ++s) LP[s] = z[s] - lt;
    }
    double rate(std::uint32_t* t) const {
        if (!t) return M_->lr;
        return M_->lr / (1.0 + static_cast<double>((*t)++) * M_->decay);
    }
    // One mixer's gradient step on log loss of y, its output P.
    void step(double* w, const double* P, std::uint8_t y, std::uint32_t* t) {
        const double lr = rate(t);
        for (int j = 0; j < kIn; ++j) {
            if (!has_[j]) continue;
            double e = 0;
            for (int s = 0; s < 256; ++s) e += P[s] * lin_[j][s];
            w[j] -= lr * (e - lin_[j][y]);
        }
    }

    const Models* M_ = nullptr;
    std::vector<std::uint8_t> buf_;
    std::unique_ptr<::cypha::online::ByteLSTM> L_;
    std::shared_ptr<::cypha::online::OnlineNgram> ng_;
    std::vector<std::uint8_t> nh_;
    std::shared_ptr<DocTopic::Stream> ds_;
    DocTopic::Stream::Pos dp_;
    std::vector<double> w_, w2_, wf_;
    std::vector<std::uint32_t> n_, n2_, nf_;
    std::uint64_t learned_ = 0;
    // the last dist()'s inputs, for learn()
    double lin_[kIn][256] = {};
    bool has_[kIn] = {};
    int set_ = 0, set2_ = 0, cls_ = 0;
    double P_[256] = {}, P1_[256] = {}, P2_[256] = {};  // P1_, P2_: the two mixers' log probabilities (mix2)
};

}  // namespace cypha::cyphalm::decoder
