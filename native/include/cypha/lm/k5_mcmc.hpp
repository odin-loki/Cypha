// Whole-paragraph sampling for K5: Metropolis-Hastings over K5's continuation by small word edits
// (docs/reports/K5_MCMC.md, docs/GENERATION_PLAN.md §13.32 item 3; CGMH: Miao et al., 2019, "Constrained sentence
// generation by Metropolis-Hastings sampling").
//
// Left-to-right sampling cannot fix an earlier word once a later one shows it was wrong. This sampler treats the
// paragraph as one object. It starts from K5's own continuation (whatever generate wrote, gate included) and proposes
// small edits:
//   - replace the word at position i;
//   - insert a word at gap g;
//   - delete the word at position i.
// A new word is drawn from K5 given both sides: q(w) is proportional to 2^(L(w) / T) over the K most probable words of
// K5's left distribution, where L(w) is the log2 probability, under K5, of w and of the order - 1 words after it (all
// the paragraph's probabilities that w changes). A share eps of draws comes from K5's whole left distribution instead,
// so every word can be proposed and every move can be undone. The Metropolis-Hastings ratio uses both proposal
// probabilities, so the chain's target is exactly pi(x) proportional to 2^(E(x) / T).
//
// The energy E (bits), of the paragraph x after the prompt:
//   E = sum_i log2 P_K5(x_i | prompt, x_<i)        (K5's model as it writes: the class floor; no temperature)
//     + lt * topic(x) + le * entity(x)             (the topic and entity terms; see Terms below)
//     - lr * (repeated 4-grams of tokens)          (positions whose 4 tokens ending there occurred earlier, prompt included)
//     + beta * n - ((n - n0) / sigma)^2 / 2        (length, sigma 1.5: beta is the start's mean surprisal per word, so a word of
//                                                   average surprisal costs nothing; the square keeps n near n0)
//   and a paragraph with 13 consecutive training tokens (the 12-word copy cap, prompt included) is never accepted.
// T runs from t0 to t1, linearly over the steps (annealing). The last state is written, cut to the byte budget as
// generate cuts it. The last token of K5's text is held fixed (the byte cut may have split it).
//
// Terms (topic and entity): until wip/k5-discourse's lda.hpp and entity_grid.hpp arrive, the topic term is a stand-in,
// a prompt-word cache: each content word (outside the 100 most frequent) of the paragraph that also occurs in the
// prompt gains log2(1 + 0.1 c(w) / p1(w)) bits (c: its share of the prompt's content words; p1: K5's order-1
// probability), halved for each earlier use of the same word in the paragraph (no stuffing gain).
//
// Rule 10 (PLAN): CyphaLM's own model. No neural network.
#pragma once

#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cypha::cyphalm::wkn::mcmc {

// An extra paragraph term (topic or entity models): begin reads the prompt (K5's ids and tokens); score gives the term
// for the paragraph (its ids, and its tokens: the same words, the held last token as written). One per continuation.
struct ParaTerm {
    virtual ~ParaTerm() = default;
    virtual void begin(const K5& m, const std::vector<std::uint32_t>& ids, const std::vector<std::string>& toks) = 0;
    virtual double score(const K5& m, const std::vector<std::uint32_t>& ids, const std::vector<std::string>& toks) = 0;
};

// wip/k5-discourse's LDA (lda.hpp), given the prompt: the sum over the paragraph's content words of
// log2 (p_topic(w | theta) / p(w)), theta the prompt's topic mixture (lda::LDA::log_ratio(infer(prompt), paragraph)).
struct LdaTerm : ParaTerm {
    std::shared_ptr<const lda::LDA> lda;
    std::vector<double> r;  // per K5 id
    explicit LdaTerm(std::shared_ptr<const lda::LDA> m) : lda(std::move(m)) {}
    void begin(const K5& m, const std::vector<std::uint32_t>&, const std::vector<std::string>& toks) override {
        const auto theta = lda->infer(toks);
        r.assign(m.vocab().size(), 0.0);
        for (std::size_t w = 2; w < r.size(); ++w) {
            const long x = lda->id(m.vocab()[w]);
            if (x >= 0) r[w] = std::log2(lda->p_topic(theta, static_cast<std::size_t>(x)) / lda->unigram(static_cast<std::size_t>(x)));
        }
        th = theta;
    }
    double score(const K5&, const std::vector<std::uint32_t>& ids, const std::vector<std::string>& toks) override {
        double t = 0;
        for (std::size_t i = 0; i + 1 < ids.size(); ++i) t += r[ids[i]];
        if (!toks.empty()) {  // the held last token, as written (it may be outside K5's vocabulary)
            const long x = lda->id(toks.back());
            if (x >= 0) t += std::log2(lda->p_topic(th, static_cast<std::size_t>(x)) / lda->unigram(static_cast<std::size_t>(x)));
        }
        return t;
    }
    std::vector<double> th;
};

// wip/k5-discourse's entity grid (entity_grid.hpp), after the prompt's sentences: full, the change in the grid's log2
// probability when the paragraph's sentences follow the prompt's; cont, the kept entities' term only
// (eg::EntityGrid::score(prompt, paragraph, cont)).
struct GridTerm : ParaTerm {
    std::shared_ptr<const eg::EntityGrid> g;
    bool cont = false;
    eg::EntityGrid::History h0;
    GridTerm(std::shared_ptr<const eg::EntityGrid> m, bool c) : g(std::move(m)), cont(c) {}
    void begin(const K5&, const std::vector<std::uint32_t>&, const std::vector<std::string>& toks) override {
        h0 = {};
        for (const auto& s : eg::split_sentences(toks)) g->append(h0, g->roles(s));
    }
    double score(const K5&, const std::vector<std::uint32_t>&, const std::vector<std::string>& toks) override {
        eg::EntityGrid::History h = h0;
        double d = 0;
        for (const auto& s : eg::split_sentences(toks)) {
            const eg::SentRoles r = g->roles(s);
            d += cont ? g->continuation_score(h, r) : g->append_logp(h, r);
            g->append(h, r);
        }
        return d;
    }
};

struct Params {
    int steps = 0;            // --k5-mcmc STEPS: proposals per paragraph (0: off)
    double t0 = 1.0, t1 = 0.6;  // --k5-mcmc-t0, --k5-mcmc-t1: the temperature, linear from t0 to t1
    double lt = 0, le = 0;    // --k5-mcmc-lt, --k5-mcmc-le: the topic and entity weights
    double lr = 3;            // --k5-mcmc-lr: bits per repeated 4-gram
    double sigma = 1.5;       // --k5-mcmc-len: the length prior's width (words)
    int k = 64;               // --k5-mcmc-k: candidates per proposal
    double eps = 0.1;         // --k5-mcmc-eps: share of draws from K5's whole left distribution
    double p_rep = 0.5, p_ins = 0.25, p_del = 0.25;  // --k5-mcmc-moves R,I,D (normalised)
    std::string topic = "cache";  // --k5-mcmc-topic: cache (the stand-in) or lda (wip/k5-discourse's LDA; --k5-mcmc-lda PATH)
    std::string entity = "full";  // --k5-mcmc-entity: full or cont (wip/k5-discourse's entity grid; --k5-mcmc-eg PATH)
    std::string lda_path, eg_path;  // --k5-mcmc-lda PATH, --k5-mcmc-eg PATH: the models (loaded, or trained on K5's text and saved)
    // Factories for the topic and entity terms (set by the tool once their models are loaded); null: the stand-in
    // cache for the topic, nothing for the entity term.
    std::function<std::unique_ptr<ParaTerm>()> topic_term, entity_term;
    std::string describe() const {
        char b[512];
        std::snprintf(b, sizeof b, "\tmcmc=%d\tmcmc_t=%g-%g\tmcmc_lt=%g(%s)\tmcmc_le=%g(%s)\tmcmc_lr=%g\tmcmc_len=%g\tmcmc_k=%d\tmcmc_eps=%g\tmcmc_moves=%g,%g,%g",
                      steps, t0, t1, lt, topic.c_str(), le, entity.c_str(), lr, sigma, k, eps,
                      p_rep, p_ins, p_del);
        return b;
    }
};

inline bool apply_flag(const std::string& name, const std::string& val, Params& p, std::string& err) {
    const auto num = [&](double& x, double lo, double hi) {
        char* e = nullptr;
        x = std::strtod(val.c_str(), &e);
        if (val.empty() || *e || !(x >= lo && x <= hi)) return err = name + ": bad value " + val, false;
        return true;
    };
    double x = 0;
    if (name == "--k5-mcmc") return num(x, 0, 1e7) && ((p.steps = static_cast<int>(x)), true);
    if (name == "--k5-mcmc-t0") return num(p.t0, 1e-3, 100);
    if (name == "--k5-mcmc-t1") return num(p.t1, 1e-3, 100);
    if (name == "--k5-mcmc-lt") return num(p.lt, 0, 1e6);
    if (name == "--k5-mcmc-le") return num(p.le, 0, 1e6);
    if (name == "--k5-mcmc-lr") return num(p.lr, 0, 1e6);
    if (name == "--k5-mcmc-len") return num(p.sigma, 0.1, 1e6);
    if (name == "--k5-mcmc-k") return num(x, 1, 4096) && ((p.k = static_cast<int>(x)), true);
    if (name == "--k5-mcmc-eps") return num(p.eps, 1e-6, 1);
    if (name == "--k5-mcmc-topic") return val == "cache" || val == "lda" ? (p.topic = val, true) : (err = name + ": cache or lda", false);
    if (name == "--k5-mcmc-entity") return val == "full" || val == "cont" ? (p.entity = val, true) : (err = name + ": full or cont", false);
    if (name == "--k5-mcmc-lda") return p.lda_path = val, true;
    if (name == "--k5-mcmc-eg") return p.eg_path = val, true;
    if (name == "--k5-mcmc-moves") {
        double v[3];
        if (std::sscanf(val.c_str(), "%lf,%lf,%lf", &v[0], &v[1], &v[2]) != 3 || v[0] < 0 || v[1] < 0 || v[2] < 0 || v[0] + v[1] + v[2] <= 0 ||
            (v[1] > 0) != (v[2] > 0))
            return err = name + ": R,I,D (insert and delete both on or both off)", false;
        const double z = v[0] + v[1] + v[2];
        p.p_rep = v[0] / z, p.p_ins = v[1] / z, p.p_del = v[2] / z;
        return true;
    }
    err = "unknown flag " + name;
    return false;
}

struct Stats {
    std::size_t paras = 0, skipped = 0, steps = 0;
    std::size_t prop[3] = {0, 0, 0}, acc[3] = {0, 0, 0};  // replace, insert, delete
    std::size_t changed = 0;                              // proposals accepted that changed the text
    double de = 0, dlm = 0, dn = 0;                       // summed final - start: energy, K5 bits, words
    double secs = 0;
    void add(const Stats& o) {
        paras += o.paras, skipped += o.skipped, steps += o.steps, changed += o.changed;
        for (int i = 0; i < 3; ++i) prop[i] += o.prop[i], acc[i] += o.acc[i];
        de += o.de, dlm += o.dlm, dn += o.dn, secs += o.secs;
    }
};

// One chain. Built per continuation.
class Chain {
 public:
    Chain(const K5& m, const Params& pr) : m_(m), pr_(pr), V_(m.vocab().size()), N_(m.kn().order()) {
        pl_.resize(V_);
    }

    // Tokens of the prompt and of K5's text; false (nothing done) if the text does not re-render to itself.
    bool init(std::string_view prompt, std::string_view gen) {
        S_ = {kBos};
        has_prev_ = false;
        tokenize(prompt, [&](std::string_view t) {
            S_.push_back(m_.id(t));
            prev_.assign(t);
            has_prev_ = true;
        });
        P_ = S_.size();
        std::vector<std::string> gt;
        tokenize(gen, [&](std::string_view t) { gt.emplace_back(t); });
        if (gt.size() < 3) return false;
        for (const auto& t : gt) S_.push_back(m_.id(t));
        last_ = gt.back();
        for (std::size_t j = P_; j + 1 < S_.size(); ++j)
            if (S_[j] == kUnk || S_[j] == kBos || m_.vocab()[S_[j]] != gt[j - P_]) return false;
        std::string_view g = gen;  // the byte cut may end on a space; the render has none there
        while (!g.empty() && g.back() == ' ') g.remove_suffix(1);
        if (render() != g) return false;
        n0_ = static_cast<double>(S_.size() - P_);
        lp_.assign(S_.size(), 0.0);
        double tot = 0;
        for (std::size_t j = P_; j < S_.size(); ++j) tot += (lp_[j] = lpw(S_, j));
        beta_ = -tot / n0_;
        // The topic stand-in: the prompt's content words.
        std::vector<std::uint32_t> pids(S_.begin() + 1, S_.begin() + static_cast<std::ptrdiff_t>(P_));
        const std::vector<std::string> ptoks = tokens(prompt);
        if (pr_.lt > 0) {
            if (pr_.topic_term) {
                topic_ = pr_.topic_term();
                topic_->begin(m_, pids, ptoks);
            } else {
                double nc = 0;
                for (const std::uint32_t w : pids)
                    if (w >= 2 + 100) pc_[w] += 1, nc += 1;
                for (auto& [w, c] : pc_) c = std::log2(1 + 0.1 * (c / nc) / m_.kn().p1()[w]);
            }
        }
        if (pr_.le > 0 && pr_.entity_term) {
            entity_ = pr_.entity_term();
            entity_->begin(m_, pids, ptoks);
        }
        other_ = other(S_);
        capv_ = capv(S_, 1, 0);
        return true;
    }

    double lm() const {
        double t = 0;
        for (std::size_t j = P_; j < S_.size(); ++j) t += lp_[j];
        return t;
    }
    double energy() const { return lm() + other_ - kCapCost * static_cast<double>(capv_); }
    std::size_t words() const { return S_.size() - P_; }
    // Recomputes every term from scratch (the selftest's check of the tracked energy).
    double energy_full() const {
        double t = 0;
        for (std::size_t j = P_; j < S_.size(); ++j) t += lpw(S_, j);
        return t + const_cast<Chain*>(this)->other(S_) - kCapCost * static_cast<double>(capv(S_, 1, 0));
    }

    // One proposal at temperature T.
    void step(std::mt19937_64& rng, double T, Stats& st) {
        step_(rng, T, st);
        tS_ = nullptr;
    }
    void step_(std::mt19937_64& rng, double T, Stats& st) {
        const std::size_t m = S_.size() - P_ - 1;  // editable positions (the last token is held)
        std::uniform_real_distribution<double> U(0, 1);
        double r = U(rng);
        int mv = r < pr_.p_rep ? 0 : r < pr_.p_rep + pr_.p_ins ? 1 : 2;
        if (mv == 2 && m <= 1) mv = pr_.p_rep > 0 ? 0 : 1;
        if (mv == 0 && m == 0) return;
        ++st.steps, ++st.prop[mv];
        if (mv == 0) {  // replace
            const std::size_t a = P_ + static_cast<std::size_t>(rng() % m);
            table(S_, a, false, T);
            const std::uint32_t old = S_[a];
            const std::uint32_t w = draw(rng);
            if (w == old) return;
            const double lw = L(S_, a, false, w), lo = L(S_, a, false, old);
            const double qf = q(w, lw, T), qr = q(old, lo, T);
            std::vector<std::uint32_t> S2 = S_;
            S2[a] = w;
            const double o2 = other(S2);
            const auto A = static_cast<std::ptrdiff_t>(a);
            const std::size_t c2 = capv_ - capv(S_, A, A) + capv(S2, A, A);
            const double la = (lw - lo + o2 - other_ - kCapCost * (static_cast<double>(c2) - static_cast<double>(capv_))) / T + std::log2(qr) - std::log2(qf);
            if (accept(la, rng)) commit(std::move(S2), o2, a, c2), ++st.acc[0], ++st.changed;
        } else if (mv == 1) {  // insert at gap g (before position P_ + g)
            const std::size_t a = P_ + static_cast<std::size_t>(rng() % (m + 1));
            table(S_, a, true, T);
            const std::uint32_t w = draw(rng);
            const double lw = L(S_, a, true, w);
            double lold = 0;
            for (std::size_t j = a; j < std::min(a + static_cast<std::size_t>(N_ - 1), S_.size()); ++j) lold += lp_[j];
            const double qf = q(w, lw, T);
            std::vector<std::uint32_t> S2 = S_;
            S2.insert(S2.begin() + static_cast<std::ptrdiff_t>(a), w);
            const double o2 = other(S2);
            const auto A = static_cast<std::ptrdiff_t>(a);
            const std::size_t c2 = capv_ - capv(S_, A - 1, A) + capv(S2, A, A);
            const double la = (lw - lold + o2 - other_ - kCapCost * (static_cast<double>(c2) - static_cast<double>(capv_))) / T + std::log2(pr_.p_del) -
                              std::log2(pr_.p_ins) - std::log2(qf);
            if (accept(la, rng)) commit(std::move(S2), o2, a, c2), ++st.acc[1], ++st.changed;
        } else {  // delete position a
            const std::size_t a = P_ + static_cast<std::size_t>(rng() % m);
            std::vector<std::uint32_t> S2 = S_;
            const std::uint32_t old = S_[a];
            S2.erase(S2.begin() + static_cast<std::ptrdiff_t>(a));
            double lnew = 0, lold = 0;
            for (std::size_t j = a; j < std::min(a + static_cast<std::size_t>(N_ - 1), S2.size()); ++j) lnew += lpw(S2, j);
            for (std::size_t j = a; j < std::min(a + static_cast<std::size_t>(N_), S_.size()); ++j) lold += lp_[j];
            table(S2, a, true, T);  // the reverse move: inserting old at gap a of S2
            const double qr = q(old, L(S2, a, true, old), T);
            const double o2 = other(S2);
            const auto A = static_cast<std::ptrdiff_t>(a);
            const std::size_t c2 = capv_ - capv(S_, A, A) + capv(S2, A - 1, A);
            const double la = (lnew - lold + o2 - other_ - kCapCost * (static_cast<double>(c2) - static_cast<double>(capv_))) / T + std::log2(pr_.p_ins) +
                              std::log2(qr) - std::log2(pr_.p_del);
            if (accept(la, rng)) commit(std::move(S2), o2, a, c2), ++st.acc[2], ++st.changed;
        }
    }

    std::string render() const {
        std::string o, prev = prev_;
        bool hp = has_prev_;
        for (std::size_t j = P_; j < S_.size(); ++j) {
            const std::string& t = j + 1 == S_.size() ? last_ : m_.vocab()[S_[j]];
            o += piece(t, prev, hp);
            prev = t, hp = true;
        }
        return o;
    }
    const std::vector<std::uint32_t>& seq() const { return S_; }
    std::size_t prompt_len() const { return P_; }

 private:
    // log2 P(S[j] | S[<j]).
    double lpw(const std::vector<std::uint32_t>& S, std::size_t j) const {
        const std::size_t c = std::min<std::size_t>(j, static_cast<std::size_t>(N_ - 1));
        return std::log2(std::max(m_.word_prob(S.data() + j - c, c, S[j]), 1e-300));
    }
    // K5's left distribution at position a of S (its words before a), the top k (cand_), and their L at temperature
    // T: L(w) = log2 p(w | left) + the log2 probabilities of the next N - 1 words with w at a (ins: w inserted before
    // S[a]; else: w replaces S[a]).
    void table(const std::vector<std::uint32_t>& S, std::size_t a, bool ins, double T) {
        tS_ = nullptr;
        inC_.clear();
        st_.ctx.assign(S.begin() + static_cast<std::ptrdiff_t>(a - std::min<std::size_t>(a, static_cast<std::size_t>(N_ + 1))),
                       S.begin() + static_cast<std::ptrdiff_t>(a));
        m_.model_dist(st_, Options{}, pl_.data(), tmp_);
        raw_ = pl_;  // before <unk> and <s> are removed: the energy's own probabilities
        pl_[kUnk] = pl_[kBos] = 0;
        double z = 0;
        for (const double x : pl_) z += x;
        for (double& x : pl_) x /= z;
        // The k most probable (ties: the lower id), by one pass with a min-heap of size k.
        const std::size_t k = std::min<std::size_t>(static_cast<std::size_t>(pr_.k), V_ - 2);
        heap_.clear();
        const auto gt = [](const std::pair<double, std::uint32_t>& x, const std::pair<double, std::uint32_t>& y) {
            return x.first != y.first ? x.first > y.first : x.second < y.second;
        };
        for (std::uint32_t w = 0; w < V_; ++w) {
            if (heap_.size() < k) {
                heap_.emplace_back(pl_[w], w);
                std::push_heap(heap_.begin(), heap_.end(), gt);
            } else if (pl_[w] > heap_.front().first) {
                std::pop_heap(heap_.begin(), heap_.end(), gt);
                heap_.back() = {pl_[w], w};
                std::push_heap(heap_.begin(), heap_.end(), gt);
            }
        }
        cand_.clear();
        for (const auto& [pw, w] : heap_) cand_.push_back(w);
        cL_.resize(k);
        double mx = -1e300;
        for (std::size_t i = 0; i < k; ++i) mx = std::max(mx, cL_[i] = L(S, a, ins, cand_[i]));
        cz_ = 0;
        cw_.resize(k);
        for (std::size_t i = 0; i < k; ++i) cz_ += (cw_[i] = std::exp2((cL_[i] - mx) / T));
        cmx_ = mx;
        for (std::size_t i = 0; i < k; ++i) inC_[cand_[i]] = i;
        tS_ = &S, ta_ = a, tins_ = ins;
    }
    double L(const std::vector<std::uint32_t>& S, std::size_t a, bool ins, std::uint32_t w) {
        if (tS_ == &S && ta_ == a && tins_ == ins) {
            const auto it = inC_.find(w);
            if (it != inC_.end()) return cL_[it->second];
        }
        // The window: N - 1 words before a, w, and the N - 1 words after it.
        const std::size_t lo = a - std::min<std::size_t>(a, static_cast<std::size_t>(N_ - 1));
        buf_.assign(S.begin() + static_cast<std::ptrdiff_t>(lo), S.begin() + static_cast<std::ptrdiff_t>(a));
        const std::size_t at = buf_.size();
        buf_.push_back(w);
        const std::size_t r0 = ins ? a : a + 1;
        for (std::size_t j = r0; j < std::min(r0 + static_cast<std::size_t>(N_ - 1), S.size()); ++j) buf_.push_back(S[j]);
        double t = std::log2(std::max(raw_[w], 1e-300));
        for (std::size_t j = at + 1; j < buf_.size(); ++j) {
            const std::size_t c = std::min<std::size_t>(j, static_cast<std::size_t>(N_ - 1));
            t += std::log2(std::max(m_.word_prob(buf_.data() + j - c, c, buf_[j]), 1e-300));
        }
        return t;
    }
    std::uint32_t draw(std::mt19937_64& rng) {
        std::uniform_real_distribution<double> U(0, 1);
        if (U(rng) < pr_.eps) {
            double u = U(rng), c = 0;
            for (std::uint32_t w = 0; w < V_; ++w)
                if ((c += pl_[w]) > u && pl_[w] > 0) return w;
            for (std::uint32_t w = static_cast<std::uint32_t>(V_); w-- > 0;)
                if (pl_[w] > 0) return w;
        }
        double u = U(rng) * cz_, c = 0;
        for (std::size_t i = 0; i < cand_.size(); ++i)
            if ((c += cw_[i]) > u) return cand_[i];
        return cand_.back();
    }
    // The proposal probability of w (its L given) in the current table.
    double q(std::uint32_t w, double Lw, double T) const {
        double qc = 0;
        const auto it = inC_.find(w);
        if (it != inC_.end()) qc = std::exp2((Lw - cmx_) / T) / cz_;
        return std::max((1 - pr_.eps) * qc + pr_.eps * pl_[w], 1e-300);
    }
    static bool accept(double la, std::mt19937_64& rng) {
        if (la >= 0) return true;
        std::uniform_real_distribution<double> U(0, 1);
        return std::log2(std::max(U(rng), 1e-300)) < la;
    }
    void commit(std::vector<std::uint32_t>&& S2, double o2, std::size_t a, std::size_t c2) {
        capv_ = c2;
        const std::ptrdiff_t d = static_cast<std::ptrdiff_t>(S2.size()) - static_cast<std::ptrdiff_t>(S_.size());
        std::vector<double> lp(S2.size(), 0.0);
        for (std::size_t j = P_; j < a; ++j) lp[j] = lp_[j];
        const std::size_t hi = std::min(a + static_cast<std::size_t>(N_), S2.size());
        for (std::size_t j = hi; j < S2.size(); ++j) lp[j] = lp_[static_cast<std::size_t>(static_cast<std::ptrdiff_t>(j) - d)];
        S_ = std::move(S2);
        for (std::size_t j = a; j < hi; ++j) lp[j] = lpw(S_, j);
        lp_ = std::move(lp);
        other_ = o2;
        tS_ = nullptr;
    }
    // The copy cap: windows of cap + 1 tokens found in the training ids that hold a paragraph token and the positions
    // lo and hi (every such window when lo > hi). An edit changes only the windows over it, so the chain counts those.
    std::size_t capv(const std::vector<std::uint32_t>& S, std::ptrdiff_t lo, std::ptrdiff_t hi) const {
        const NgramSet& cs = m_.cap_set();
        if (!cs.on()) return 0;
        const auto W = static_cast<std::ptrdiff_t>(cs.n()), n = static_cast<std::ptrdiff_t>(S.size()), P = static_cast<std::ptrdiff_t>(P_);
        std::ptrdiff_t s0 = std::max<std::ptrdiff_t>(0, P - W + 1), s1 = n - W;
        if (lo <= hi) s0 = std::max(s0, hi - W + 1), s1 = std::min(s1, lo);
        std::size_t v = 0;
        for (std::ptrdiff_t s = s0; s <= s1; ++s) {
            std::uint64_t h = kFnv0;
            for (std::ptrdiff_t j = 0; j < W; ++j) h = fnv_step(h, S[static_cast<std::size_t>(s + j)]);
            v += cs.has(h);
        }
        return v;
    }
    static constexpr double kCapCost = 1e6;  // bits per window over the cap: never accepted from a state within it
    // Every term but K5's bits and the cap, for the sequence S (prompt + paragraph).
    double other(const std::vector<std::uint32_t>& S) {
        double e = 0;
        // Repeated 4-grams (prompt included), counted where they end in the paragraph.
        seen_.clear();
        std::size_t rep = 0;
        for (std::size_t j = 4; j < S.size(); ++j) {
            std::uint64_t h = kFnv0;
            for (std::size_t i = j - 3; i <= j; ++i) h = fnv_step(h, S[i]);
            if (!seen_.insert(h).second && j >= P_) ++rep;
        }
        e -= pr_.lr * static_cast<double>(rep);
        const double n = static_cast<double>(S.size() - P_);
        e += beta_ * n - 0.5 * ((n - n0_) / pr_.sigma) * ((n - n0_) / pr_.sigma);
        if (pr_.lt > 0 || (pr_.le > 0 && entity_)) {
            para_.assign(S.begin() + static_cast<std::ptrdiff_t>(P_), S.end());
            if (topic_ || entity_) {
                ptok_.resize(para_.size());
                for (std::size_t i = 0; i < para_.size(); ++i) ptok_[i] = i + 1 == para_.size() ? last_ : m_.vocab()[para_[i]];
            }
            if (pr_.lt > 0) e += pr_.lt * topic(para_);
            if (pr_.le > 0 && entity_) e += pr_.le * entity_->score(m_, para_, ptok_);
        }
        return e;
    }
    double topic(const std::vector<std::uint32_t>& x) {
        if (topic_) return topic_->score(m_, x, ptok_);
        double t = 0;
        uses_.clear();
        for (const std::uint32_t w : x) {
            const auto it = pc_.find(w);
            if (it == pc_.end()) continue;
            t += it->second * std::exp2(-static_cast<double>(uses_[w]++));
        }
        return t;
    }

    const K5& m_;
    const Params& pr_;
    std::size_t V_;
    int N_;
    std::vector<std::uint32_t> S_;
    std::size_t P_ = 0;
    std::string prev_, last_;
    bool has_prev_ = false;
    std::vector<double> lp_;
    double n0_ = 0, beta_ = 0, other_ = 0;
    std::size_t capv_ = 0;
    std::unordered_map<std::uint32_t, double> pc_;
    std::unordered_map<std::uint32_t, int> uses_;
    std::unique_ptr<ParaTerm> topic_, entity_;
    // Scratch.
    K5::State st_;
    std::vector<double> pl_, raw_, tmp_, cL_, cw_;
    std::vector<std::uint32_t> cand_, buf_, para_;
    std::vector<std::string> ptok_;
    std::vector<std::pair<double, std::uint32_t>> heap_;
    std::unordered_map<std::uint32_t, std::size_t> inC_;
    std::unordered_set<std::uint64_t> seen_;
    const std::vector<std::uint32_t>* tS_ = nullptr;
    std::size_t ta_ = 0;
    bool tins_ = false;
    double cz_ = 1, cmx_ = 0;
};

// generate's cut to the byte budget (a character the cut splits is dropped).
inline void cut_bytes(std::string& out, std::size_t bytes) {
    if (out.size() <= bytes) return;
    out.resize(bytes);
    std::size_t i = out.size(), back = 0;
    while (i > 0 && back < 4 && (static_cast<unsigned char>(out[i - 1]) & 0xC0) == 0x80) --i, ++back;
    if (i > 0) {
        const auto c = static_cast<unsigned char>(out[i - 1]);
        const std::size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (need > back + 1) out.resize(i - 1);
    }
}

// The sampler on one continuation: K5's text gen (for prompt) in, the chain's last state out.
inline std::string run(const K5& m, std::string_view prompt, std::string gen, std::uint64_t seed, const Options& o, const Params& pr,
                       Stats& st) {
    const auto t0 = std::chrono::steady_clock::now();
    ++st.paras;
    Chain c(m, pr);
    if (pr.steps <= 0 || !K5::word_prob_ok(o) || !c.init(prompt, gen)) {
        ++st.skipped;
        return gen;
    }
    std::mt19937_64 rng(mix64(seed ^ 0x3C3C3C3C5A5A5A5Aull));
    const double e0 = c.energy(), l0 = c.lm(), n0 = static_cast<double>(c.words());
    for (int t = 0; t < pr.steps; ++t) {
        const double T = pr.steps > 1 ? pr.t0 + (pr.t1 - pr.t0) * t / (pr.steps - 1) : pr.t1;
        c.step(rng, T, st);
    }
    st.de += c.energy() - e0, st.dlm += c.lm() - l0, st.dn += static_cast<double>(c.words()) - n0;
    std::string out = c.render();
    cut_bytes(out, o.bytes);
    st.secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// Thread-safe stats for the tool's hook.
struct SharedStats {
    std::mutex mu;
    Stats s;
    void add(const Stats& o) {
        std::lock_guard<std::mutex> g(mu);
        s.add(o);
    }
    void print(std::FILE* f) {
        std::lock_guard<std::mutex> g(mu);
        const auto r = [](std::size_t a, std::size_t b) { return b ? static_cast<double>(a) / static_cast<double>(b) : 0.0; };
        std::fprintf(f, "k5-mcmc: %zu paragraphs (%zu skipped), %zu proposals; accepted replace %.3f insert %.3f delete %.3f; "
                        "mean change per paragraph: energy %+.2f bits, K5 %+.2f bits, words %+.2f; %.1f s of chain time (%.1f ms a paragraph)\n",
                     s.paras, s.skipped, s.steps, r(s.acc[0], s.prop[0]), r(s.acc[1], s.prop[1]), r(s.acc[2], s.prop[2]),
                     s.de / std::max<double>(1, static_cast<double>(s.paras - s.skipped)), s.dlm / std::max<double>(1, static_cast<double>(s.paras - s.skipped)),
                     s.dn / std::max<double>(1, static_cast<double>(s.paras - s.skipped)), s.secs, 1e3 * s.secs / std::max<double>(1, static_cast<double>(s.paras)));
    }
};

// Installs the sampler as o's post hook. stats must outlive the run.
inline void install(Options& o, const Params& pr, SharedStats* stats) {
    o.post = [pr, stats](const K5& m, std::string_view prompt, std::string gen, std::uint64_t seed, const Options& oo) {
        Stats st;
        std::string out = run(m, prompt, std::move(gen), seed, oo, pr, st);
        stats->add(st);
        return out;
    };
    o.post_desc = pr.describe();
}

// ---------------------------------------------------------------- selftest

inline bool self_test(std::string& err) {
    std::string text;
    {
        std::mt19937_64 r(11);
        const char* subj[] = {"The cat", "A dog", "The old man", "Her brother", "The king of France", "It"};
        const char* verb[] = {"saw", "liked", "found", "wrote about", "did not see", "lost"};
        const char* obj[] = {"the river", "a small house", "three apples", "the 1990s", "Paris", "his own book"};
        const char* tail[] = {".", ", and left.", " (twice).", "; it rained.", "!", " in 1,200 days."};
        for (int d = 0; d < 80; ++d) {
            for (int s = 0; s < 8; ++s) {
                text += subj[r() % 6], text += " ", text += verb[r() % 6], text += " ", text += obj[r() % 6], text += tail[r() % 6];
                text += " ";
            }
            if (d % 7 == 0) text += "One two three four five six seven eight nine ten eleven twelve thirteen fourteen. ";
            text += "\n";
        }
    }
    for (const std::size_t classes : {std::size_t{0}, std::size_t{8}}) {
        K5 m;
        m.build(text, 12, kOrder, classes);
        const std::size_t V = m.vocab().size();
        // word_prob equals model_dist's entry.
        {
            Options o;
            std::vector<double> p(V), tmp;
            const char* ctxs[] = {"", "The", "The cat saw", "Her brother lost the river", "One two three four"};
            for (const char* c : ctxs) {
                K5::State s = m.start(c, o);
                m.model_dist(s, o, p.data(), tmp);
                for (std::uint32_t w = 0; w < V; ++w) {
                    const double q = m.word_prob(s.ctx.data(), s.ctx.size(), w);
                    if (std::fabs(q - p[w]) > 1e-12 * std::max(p[w], 1e-300) + 1e-300) {
                        err = "word_prob differs from model_dist after \"" + std::string(c) + "\" (classes " + std::to_string(classes) + ")";
                        return false;
                    }
                }
            }
        }
        // The chain: its tracked energy equals a full recomputation; the text renders and keeps the cap.
        Params pr;
        pr.steps = 400, pr.lt = 1;
        Options o = k5_v1();
        o.bytes = 200;
        for (std::uint64_t sd = 0; sd < 6; ++sd) {
            const std::string prompt = sd % 2 ? "The old man found Paris. Her brother" : "One two three four five six seven eight nine ten eleven";
            const std::string g = m.generate(prompt, sd, o);
            Chain c(m, pr);
            if (!c.init(prompt, g)) {
                err = "chain init on: " + g;
                return false;
            }
            std::mt19937_64 rng(sd);
            Stats st;
            for (int t = 0; t < pr.steps; ++t) {
                c.step(rng, 1.0 - 0.5 * t / pr.steps, st);
                if (t % 50 == 49 && std::fabs(c.energy() - c.energy_full()) > 1e-6 * (1 + std::fabs(c.energy()))) {
                    err = "tracked energy " + std::to_string(c.energy()) + " vs " + std::to_string(c.energy_full());
                    return false;
                }
            }
            if (st.acc[0] + st.acc[1] + st.acc[2] == 0) {
                err = "the chain accepted nothing";
                return false;
            }
            std::string out = c.render();
            cut_bytes(out, o.bytes);
            if (out.empty() || out.find("<unk>") != std::string::npos || m.has_run13(prompt + " " + out)) {
                err = "chain output: " + out;
                return false;
            }
        }
    }
    {  // The discourse terms equal wip/k5-discourse's own calls, and the chain tracks its energy with them on.
        K5 m;
        m.build(text, 12, kOrder, 8);
        const auto tok = [](std::string_view line, auto&& emit) { tokenize(line, emit); };
        auto L = std::make_shared<lda::LDA>();
        lda::Params lp;
        lp.K = 4, lp.iters = 30, lp.min_count = 2;
        L->train_lines(text, tok, lp);
        auto G = std::make_shared<eg::EntityGrid>();
        G->train_lines(text, tok, 2);
        const std::string prompt = "The king of France found Paris. Her brother", para = " lost the river. The king of France saw a small house. It";
        const auto pt = tokens(prompt), ct = tokens(para);
        std::vector<std::uint32_t> pid, cid;
        for (const auto& t : pt) pid.push_back(m.id(t));
        for (const auto& t : ct) cid.push_back(m.id(t));
        LdaTerm lt(L);
        lt.begin(m, pid, pt);
        const double a1 = lt.score(m, cid, ct), b1 = L->log_ratio(L->infer(pt), ct);
        for (const bool cont : {false, true}) {
            GridTerm gt(G, cont);
            gt.begin(m, pid, pt);
            const double a2 = gt.score(m, cid, ct), b2 = G->score(pt, ct, cont);
            if (std::fabs(a2 - b2) > 1e-9) {
                err = "GridTerm " + std::to_string(a2) + " vs EntityGrid::score " + std::to_string(b2);
                return false;
            }
        }
        if (std::fabs(a1 - b1) > 1e-9 * (1 + std::fabs(b1))) {
            err = "LdaTerm " + std::to_string(a1) + " vs log_ratio " + std::to_string(b1);
            return false;
        }
        Params pr;
        pr.steps = 300, pr.lt = 0.5, pr.le = 0.5, pr.topic = "lda";
        pr.topic_term = [L] { return std::make_unique<LdaTerm>(L); };
        pr.entity_term = [G] { return std::make_unique<GridTerm>(G, false); };
        Options o;
        o.bytes = 200;
        const std::string g = m.generate(prompt, 3, o);
        Chain c(m, pr);
        if (!c.init(prompt, g)) {
            err = "chain init (discourse terms): " + g;
            return false;
        }
        std::mt19937_64 rng(5);
        Stats st;
        for (int t = 0; t < pr.steps; ++t) c.step(rng, 0.8, st);
        if (std::fabs(c.energy() - c.energy_full()) > 1e-6 * (1 + std::fabs(c.energy())) || st.acc[0] + st.acc[1] + st.acc[2] == 0) {
            err = "chain with the discourse terms: tracked " + std::to_string(c.energy()) + " vs " + std::to_string(c.energy_full());
            return false;
        }
    }
    {  // Flags.
        Params p;
        std::string e;
        if (!apply_flag("--k5-mcmc", "500", p, e) || p.steps != 500 || !apply_flag("--k5-mcmc-moves", "2,1,1", p, e) || p.p_rep != 0.5 ||
            apply_flag("--k5-mcmc-moves", "1,1,0", p, e) || apply_flag("--k5-mcmc-bogus", "1", p, e)) {
            err = "mcmc apply_flag";
            return false;
        }
    }
    return true;
}

}  // namespace cypha::cyphalm::wkn::mcmc
