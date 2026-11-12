#pragma once

/// CyphaLM's decoder (docs/DECODER.md, docs/reports/K5_MIXDEC.md). Odin Loch, 2026-11-12.
///
/// The learner (learner.hpp) is primed on the retrieved article and the prompt, then frozen. K5 proposes the words: its
/// nucleus, after the repeat ban and the 12-word copy cap, is untouched. The decoder then
///   - scores each candidate word by the learner's exact probability of its written bytes after the text so far, plus
///     the probability that the word ends there (the next byte is not a letter or digit: the T3 bridge's "cover", so
///     "the" does not also count "then");
///   - combines that with K5's word probability inside the nucleus, in one of two registered ways: the context ratio
///     p_K5 x (p_mix(w | context) / p_mix(w | neutral))^r (classifier-free guidance's form; the neutral learner is the
///     base, primed on generic prose only), or a veto that drops words whose context probability is more than V bits
///     below the best candidate's and samples K5's own distribution on the rest;
///   - draws the word by arithmetic decoding of random bytes through the combined distribution (the reverse of
///     compression: exact sampling, at the nucleus K5 already applied);
///   - and, as a judge in the sentence gate, scores each draft by the learner's log-probability of the draft given the
///     article, the prompt and the text so far, minus the neutral learner's, per byte (a guide toward what the context
///     explains, not toward what is probable: generic text scores near 0 on it).
/// Every choice stays among K5's candidates (rule 10: CyphaLM writes every byte). The planner's fact sentences are
/// untouched; the learner reads them as text.

#include "cypha/lm/decoder/arith.hpp"
#include "cypha/lm/decoder/learner.hpp"
#include "cypha/lm/word_kn.hpp"

#include <atomic>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cypha::cyphalm::decoder {

/// --mixdec SPEC: comma-separated fields. ratio=R (the context ratio, weight R) | veto=V (the veto, V bits); judge=J (the
/// judge's weight in the gate, z-scored as the gate's other terms); ad=0|1 (arithmetic decoding; 1 by default once a
/// word term is on); art=N (article bytes primed; 8192); lstm=0, ngram=0, copy=0, kn=0, trie=0 (leave an expert out).
struct Params {
    int mode = 0;  // 0: no word term; 3: ratio; 4: veto (word_kn.hpp's scorer modes)
    double w = 0;
    double judge = 0;
    int ad = -1;
    std::size_t art = 8192;
    bool lstm = true, ngram = true, copy = true, kn = true, trie = true;
    bool on() const { return mode != 0 || judge != 0 || ad > 0; }
    bool decode() const { return ad > 0 || (ad < 0 && mode != 0); }
    bool parse(const std::string& spec, std::string& err) {
        std::size_t a = 0;
        while (a < spec.size()) {
            std::size_t e = spec.find(',', a);
            if (e == std::string::npos) e = spec.size();
            const std::string f = spec.substr(a, e - a);
            a = e + 1;
            const std::size_t q = f.find('=');
            if (q == std::string::npos) return err = "--mixdec: " + f + " is not KEY=VALUE", false;
            const std::string k = f.substr(0, q);
            const double x = std::atof(f.c_str() + q + 1);
            if (k == "ratio") mode = 3, w = x;
            else if (k == "veto") mode = 4, w = x;
            else if (k == "judge") judge = x;
            else if (k == "ad") ad = x != 0 ? 1 : 0;
            else if (k == "art") art = static_cast<std::size_t>(x);
            else if (k == "lstm") lstm = x != 0;
            else if (k == "ngram") ngram = x != 0;
            else if (k == "copy") copy = x != 0;
            else if (k == "kn") kn = x != 0;
            else if (k == "trie") trie = x != 0;
            else return err = "--mixdec: unknown key " + k, false;
        }
        if (!on()) return err = "--mixdec: nothing on (ratio=, veto=, judge= or ad=1)", false;
        return true;
    }
    std::string describe() const {
        char b[256];
        std::snprintf(b, sizeof b, "mixdec %s%g judge %g ad %d art %zu experts%s%s%s%s%s", mode == 3 ? "ratio " : mode == 4 ? "veto " : "none ",
                      w, judge, decode() ? 1 : 0, art, trie ? " trie" : "", kn ? " kn" : "", copy ? " copy" : "", ngram ? " ngram" : "",
                      lstm ? " lstm" : "");
        return b;
    }
};

/// The models every continuation reads, and the base learner (primed once on generic prose). Read-only once built.
struct Shared {
    Params P;
    Models M;
    std::unique_ptr<::cypha::online::ByteLSTM> lstm0;
    std::unique_ptr<Learner> base;
    mutable std::atomic<std::uint64_t> dists{0}, words{0}, drafts{0}, judged{0}, conts{0};
};

inline bool word_byte(std::uint8_t c) { return c >= 0x80 || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

/// log2 of the probability that the word just read ends: the next byte is not part of a word.
inline double end_lp(const double* P) {
    double e = 0;
    for (int s = 0; s < 256; ++s)
        if (!word_byte(static_cast<std::uint8_t>(s))) e += P[s];
    return std::log2(std::max(e, 1e-300));
}

class Dec;

/// One draft's scorer: a copy of the committed learner that reads the draft's words (frozen), scores K5's candidates
/// with prefix sharing and an exact rewind, and decodes the word by arithmetic decoding.
class Fork : public wkn::ByteScorer {
 public:
    Fork(Dec& d, const Learner& com, std::uint64_t seed);
    void begin(std::string_view) override {}
    double score(std::string_view piece) override;
    double prior(std::string_view piece) override;
    void advance(std::string_view piece) override {
        f_.rewind(root_);
        f_.read(piece);
        root_ = f_.mark();
        memo_.clear();
    }
    bool decode(const double* w, std::size_t k, std::size_t& j) override;

 private:
    struct Node {
        double P[256];
        Learner::Mark m;
    };
    const Node& node(const std::string& key);
    Dec& d_;
    Learner f_;
    Learner::Mark root_;
    std::unordered_map<std::string, Node> memo_;
    std::mt19937_64 src_;
    ArithDecoder ad_;
};

/// One continuation's decoder: the committed learner (primed, then frozen, reading the text as it is committed) and the
/// neutral learner (the base, for the ratio's and the judge's denominators).
class Dec : public wkn::MixDec {
 public:
    Dec(const Shared& S, std::string_view prompt, std::string_view article) : S_(S), com_(*S.base), neu_(*S.base) {
        com_.own();
        neu_.own();
        com_.learn("\n\n");
        com_.learn(article.substr(0, std::min(article.size(), S.P.art)));
        com_.learn("\n\n");
        com_.learn(prompt);
        com_.trim();
        ++S.conts;
    }
    std::unique_ptr<wkn::ByteScorer> fork(std::string_view out, std::uint64_t seed) override {
        if (S_.P.mode == 0 && !S_.P.decode()) return nullptr;
        sync(out);
        ++S_.drafts;
        return std::make_unique<Fork>(*this, com_, seed);
    }
    double judge(std::string_view out, std::string_view draft) override {
        sync(out);
        ++S_.judged;
        if (draft.empty()) return 0;
        const double c = bits(com_, draft), p = bits(neu_, draft);
        return (c - p) / static_cast<double>(draft.size());
    }
    double judge_w() const override { return S_.P.judge; }
    const Shared& shared() const { return S_; }
    Learner& neutral() { return neu_; }

 private:
    friend class Fork;
    // The committed text: read what is new of out (frozen).
    void sync(std::string_view out) {
        if (out.size() > read_) com_.read(out.substr(read_));
        read_ = out.size();
        com_.trim();
    }
    // log2 P(s | learner's text), learning nothing; the learner is rewound after.
    double bits(Learner& L, std::string_view s) {
        const auto m = L.mark();
        double b = 0, P[256];
        for (const char c : s) {
            L.dist(P);
            ++S_.dists;
            b += std::log2(std::max(P[static_cast<std::uint8_t>(c)], 1e-300));
            L.read(static_cast<std::uint8_t>(c));
        }
        L.rewind(m);
        return b;
    }
    const Shared& S_;
    Learner com_, neu_;
    std::size_t read_ = 0;
};

inline Fork::Fork(Dec& d, const Learner& com, std::uint64_t seed)
    : d_(d), f_(com), root_(f_.mark()), src_(seed ^ 0xDEC0DE5EEDull), ad_([this] { return static_cast<std::uint8_t>(src_() >> 56); }) {}

inline const Fork::Node& Fork::node(const std::string& key) {
    auto it = memo_.find(key);
    if (it != memo_.end()) return it->second;
    Node nd;
    if (key.empty()) {
        f_.rewind(root_);
    } else {
        const Node& par = node(key.substr(0, key.size() - 1));
        f_.rewind(par.m);
        f_.read(static_cast<std::uint8_t>(key.back()));
    }
    f_.dist(nd.P);
    ++d_.S_.dists;
    nd.m = f_.mark();
    return memo_.emplace(key, std::move(nd)).first->second;
}

inline double Fork::score(std::string_view piece) {
    ++d_.S_.words;
    std::string key;
    double lb = 0;
    for (const char c : piece) {
        lb += std::log2(std::max(node(key).P[static_cast<std::uint8_t>(c)], 1e-300));
        key.push_back(c);
    }
    if (!piece.empty() && word_byte(static_cast<std::uint8_t>(piece.back()))) lb += end_lp(node(key).P);
    return lb;
}

inline double Fork::prior(std::string_view piece) {
    thread_local std::unordered_map<std::string, double> cache;  // the neutral learner is the same in every continuation
    const std::string k(piece);
    if (auto it = cache.find(k); it != cache.end()) return it->second;
    Learner& L = d_.neutral();
    const auto m = L.mark();
    double lb = 0, P[256];
    for (const char c : piece) {
        L.dist(P);
        ++d_.S_.dists;
        lb += std::log2(std::max(P[static_cast<std::uint8_t>(c)], 1e-300));
        L.read(static_cast<std::uint8_t>(c));
    }
    if (!piece.empty() && word_byte(static_cast<std::uint8_t>(piece.back()))) {
        L.dist(P);
        ++d_.S_.dists;
        lb += end_lp(P);
    }
    L.rewind(m);
    if (cache.size() > 2000000) cache.clear();
    return cache.emplace(k, lb).first->second;
}

inline bool Fork::decode(const double* w, std::size_t k, std::size_t& j) {
    if (!d_.S_.P.decode()) return false;
    j = ad_.decode(w, k);
    return true;
}

/// Builds the shared models: the base learner learns `generic` (priming on generic clean prose).
inline void prime_base(Shared& S, std::string_view generic) {
    S.base = std::make_unique<Learner>(S.M, S.lstm0.get());
    S.base->learn(generic);
    S.base->trim();
}

/// Selftest: the arithmetic coder's round trip, and the learner's exact rewind (a learner with the copy and n-gram
/// experts and a small LSTM learns a text, reads hypothetical bytes, rewinds: the same distribution, bit for bit; a copy
/// gives the same distribution as the original).
inline bool self_test(std::string& err) {
    if (!arith_selftest(err)) return false;
    Models M;
    M.lstm = true;
    M.lr = 0.05;  // no trie or Kneser-Ney here: the weights start on an absent input, so they must move fast
    ::cypha::online::ByteLSTM l0;
    Learner L(M, &l0);
    for (int r = 0; r < 30; ++r) L.learn("The quick brown fox jumps over the lazy dog. ");
    L.learn("The quick ");
    double a[256], b[256], c[256];
    L.dist(a);
    const auto m = L.mark();
    L.read("brown dog sleeps");
    L.dist(b);
    L.rewind(m);
    L.dist(b);
    Learner C(L);
    C.dist(c);
    for (int s = 0; s < 256; ++s)
        if (a[s] != b[s] || a[s] != c[s]) return err = "decoder: rewind or copy changed the distribution", false;
    if (!(a[static_cast<unsigned char>('b')] > a[static_cast<unsigned char>('z')])) return err = "decoder: the learner did not learn", false;
    return true;
}

}  // namespace cypha::cyphalm::decoder
