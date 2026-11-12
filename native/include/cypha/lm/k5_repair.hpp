// A bidirectional repair pass for K5's sentences (docs/reports/K5_REPAIR.md, job D1).
//
// K5 writes left to right from the last four words, so it cannot see that a clause has gone wrong ("In many as a
// student at a great field."). This pass adds a right-to-left K5: the same Kneser-Ney estimate over the same text,
// each document read backwards by word, with K5's vocabulary and K5's 256 exchange classes (a class floor counted on
// the reversed class stream). Its counts come from one more suffix array over the reversed ids (lm/count_index.hpp).
//
// When K5 finishes a sentence (a word . ! ? or a newline, the byte limit or the word limit), the pass looks at the
// words K5's sampler wrote in it since the last repair. Words the planner wrote (fact sentences) and the prompt are
// never changed, nor is the sentence's last word.
//   - Flag. sf(i) = -log2 Pf(x_i | the 4 words before), sb(i) = -log2 Pb(x_i | the up to 4 words after, inside the
//     sentence). A word is flagged when both directions find it unlikely: min(sf(i), sb(i)) >= tau. A join between two
//     words is broken when the forward model does not expect the right word and the backward model does not expect
//     the left one: min(sf(i + 1), sb(i)) >= tau; both its words are flagged (those that may change).
//   - Resample, Gibbs-style, left to right over the flagged words. Candidates at i: the kf most probable words of
//     forward K5 at i (its distribution as it writes, retrieval overlay and class floor included), the kb most probable
//     of backward K5 at i, and the current word; never <unk>, <s> or a sentence end. A candidate is dropped if it would
//     make a 13-word copy of the training text (the 12-word cap) or break the repeat ban.
//   - Score. Over the window, with w at i: F(w) = sum of log2 Pf over positions i .. i + 4 (inside the sentence),
//     B(w) = sum of log2 Pb over positions i - 4 .. i (the right context cut at the sentence's end). w is drawn with
//     probability proportional to 2^((F(w) + B(w)) / T); T = 2 is the geometric mean of the two directions (each
//     direction alone already gives the conditional of x_i given both sides, so their product counts the evidence
//     twice), T = 1 their plain product, T = 0 the best candidate.
//   - Up to `sweeps` sweeps; the flags are found again before each; a sweep with no flag ends the repair.
// The random stream is seeded from the sentence's ids, so the output does not depend on the threads.
//
// The pass edits K5's state in place (ctx, out, the repeat ban), so the next sentence is drawn after the repaired one.
// It runs on every draft of the sentence gate (each draft is repaired before the gate chooses). The gate's terms that
// are summed while a draft is drawn (fluency, prompt association, topic) and the discourse writer's word lists see the
// draft before its repair; its text (and so MBR and the next sentence) is the repaired one.
//
// Rule 10 (PLAN): CyphaLM's own counts. No neural network.
//
// Author: Odin Loch
#pragma once

#include "cypha/lm/word_kn.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace cypha::cyphalm::wkn::repair {

struct Params {
    double tau = 8;     // tau=B: flag threshold in bits
    double temp = 2;    // t=T: 2 the geometric mean of the directions, 1 their product, 0 the best candidate
    int kf = 16;        // kf=N: forward candidates
    int kb = 16;        // kb=N: backward candidates
    int sweeps = 3;     // sweeps=N
    std::string describe() const {
        char b[160];
        std::snprintf(b, sizeof b, "\trepair_tau=%g\trepair_t=%g\trepair_kf=%d\trepair_kb=%d\trepair_sweeps=%d", tau, temp, kf, kb, sweeps);
        return b;
    }
};

// --k5-repair on | tau=B,t=T,kf=N,kb=N,sweeps=N (fields left out keep their defaults). off: false with no error.
inline bool parse(const std::string& spec, Params& p, std::string& err) {
    err.clear();
    if (spec == "off") return false;
    p = Params{};
    if (spec == "on") return true;
    for (std::size_t a = 0; a <= spec.size();) {
        std::size_t e = spec.find(',', a);
        if (e == std::string::npos) e = spec.size();
        const std::string f = spec.substr(a, e - a);
        const std::size_t q = f.find('=');
        if (q == std::string::npos || q + 1 >= f.size()) return err = "--k5-repair: bad field " + f, false;
        const std::string k = f.substr(0, q);
        char* end = nullptr;
        const double v = std::strtod(f.c_str() + q + 1, &end);
        if (*end) return err = "--k5-repair: bad value in " + f, false;
        if (k == "tau" && v >= 0) p.tau = v;
        else if (k == "t" && v >= 0) p.temp = v;
        else if (k == "kf" && v >= 0 && v <= 1024 && v == std::floor(v)) p.kf = static_cast<int>(v);
        else if (k == "kb" && v >= 0 && v <= 1024 && v == std::floor(v)) p.kb = static_cast<int>(v);
        else if (k == "sweeps" && v >= 1 && v <= 64 && v == std::floor(v)) p.sweeps = static_cast<int>(v);
        else return err = "--k5-repair: fields tau=B, t=T (>= 0), kf=N, kb=N, sweeps=N (1-64); bad " + f, false;
        a = e + 1;
    }
    if (p.kf + p.kb < 1) return err = "--k5-repair: kf + kb must be at least 1", false;
    return true;
}

// The backward class floor: K5's classes, a class trigram on the reversed class stream, and K5's P(w | class), as
// ClassModel gives the forward floor.
class RevClass {
 public:
    void build(const ClassModel& cm, const std::vector<std::uint32_t>& rev) {
        cls_ = cm.classes();
        pwc_ = cm.pwc();
        K2_ = cm.num_classes();
        V_ = cls_.size();
        std::vector<std::uint32_t> cids(rev.size());
        for (std::size_t t = 0; t < rev.size(); ++t) cids[t] = cls_[rev[t]];
        kn_.build(std::move(cids), K2_, 3, 1, false);
        cmass_.assign(K2_, 0.0);
        for (std::size_t w = 0; w < V_; ++w) cmass_[cls_[w]] += pwc_[w];
    }
    bool on() const { return V_ > 0; }
    double prob(const std::uint32_t* ctx, std::size_t n, std::uint32_t w) const {
        double pc[4096 + 2];
        const double z = parts(ctx, n, pc);
        return pc[cls_[w]] * pwc_[w] / z;
    }
    void dist(const std::uint32_t* ctx, std::size_t n, double* p) const {
        double pc[4096 + 2];
        const double z = parts(ctx, n, pc);
        for (std::size_t w = 0; w < V_; ++w) p[w] = pc[cls_[w]] * pwc_[w] / z;
    }

 private:
    double parts(const std::uint32_t* ctx, std::size_t n, double* pc) const {
        std::uint32_t cc[2];
        const std::size_t m = std::min<std::size_t>(n, 2);
        for (std::size_t i = 0; i < m; ++i) cc[i] = cls_[ctx[n - m + i]];
        kn_.dist(cc, m, pc);
        double z = 0;
        for (std::size_t c = 0; c < K2_; ++c) z += pc[c] * cmass_[c];
        return z;
    }
    std::vector<std::uint32_t> cls_;
    std::vector<double> pwc_, cmass_;
    std::size_t K2_ = 0, V_ = 0;
    WordKN kn_;
};

// Counts over a run, printed at the end.
struct Stats {
    std::atomic<std::uint64_t> sentences{0}, words{0}, skipped{0}, flagged{0}, changed{0}, repaired{0}, ns{0};
    void print(std::FILE* f) const {
        std::fprintf(f, "k5-repair: %llu sentences (%llu skipped: banned or not re-rendered), %llu of their words; %llu flags; %llu words "
                        "changed in %llu sentences; %.1f s in the pass (all threads)\n",
                     static_cast<unsigned long long>(sentences.load()), static_cast<unsigned long long>(skipped.load()),
                     static_cast<unsigned long long>(words.load()), static_cast<unsigned long long>(flagged.load()),
                     static_cast<unsigned long long>(changed.load()), static_cast<unsigned long long>(repaired.load()),
                     static_cast<double>(ns.load()) * 1e-9);
    }
};

// One continuation's part: the prompt (the repeat ban's text starts with it) and the backward overlay of the documents
// retrieval read for it (null without retrieval).
struct Cont {
    std::string prompt;
    std::shared_ptr<const Overlay> rov;
};

inline bool sentence_end(const std::string& t) { return t == "." || t == "!" || t == "?" || t == "\n"; }

class Repairer {
 public:
    explicit Repairer(Params p = {}) : p_(p) {}
    const Params& params() const { return p_; }
    const Stats& stats() const { return st_; }
    const WordKN& backward() const { return bk_; }
    const RevClass& backward_classes() const { return bc_; }

    // Options the pass does not model (its forward probabilities are K5's with the retrieval overlay and the class
    // floor, as model_dist gives them with these off).
    static bool supports(const Options& o, std::string& err) {
        if (o.dyn > 0 || o.cache_w > 0 || o.cache_w2 > 0 || o.lda_beta > 0 || o.class_mode != 1 || o.scorer || o.post)
            return err = "--k5-repair does not take --k5-dyn, --k5-cache, --k5-lda-beta, --k5-class-mode mix, --k5-lstm or --k5-mcmc", false;
        if (o.ret.on() && o.ret.grow > 0) return err = "--k5-repair does not take --k5-retrieve grow=", false;
        return true;
    }

    // The backward model over K5's training ids (each document's tokens reversed, <s> first, as forward).
    void build(const K5& m) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<std::uint32_t>& ids = m.train_ids();
        std::vector<std::uint32_t> rev(ids.size());
        for (std::size_t a = 0; a < ids.size();) {
            std::size_t b = a + 1;
            while (b < ids.size() && ids[b] != kBos) ++b;
            rev[a] = ids[a];  // <s> (or the stream's first id)
            for (std::size_t i = a + 1; i < b; ++i) rev[a + 1 + (b - 1 - i)] = ids[i];
            a = b;
        }
        if (m.classes().on()) bc_.build(m.classes(), rev);
        std::uint32_t keep = WordKN::kKeep;
        if (const char* e = std::getenv("CYPHA_K5_KEEP")) keep = static_cast<std::uint32_t>(std::max(1, std::atoi(e)));
        bk_.build(std::move(rev), m.vocab().size(), m.kn().order(), keep);
        std::fprintf(stderr, "k5-repair: backward K5 built in %.1f s (%.1f MB)%s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), static_cast<double>(bk_.bytes()) / 1e6,
                     p_.describe().c_str());
    }

    // start(): the continuation's part.
    void begin(const K5& m, std::string_view prompt, K5::State& s, const Options& o) const {
        auto c = std::make_shared<Cont>();
        c->prompt.assign(prompt);
        if (o.ret.on() && m.retrieval() && s.rov) {  // the same documents, read backwards
            auto ov = std::make_shared<Overlay>();
            ov->L = s.rov->L, ov->rel = s.rov->rel, ov->Dd = s.rov->Dd, ov->order = s.rov->order;
            const std::vector<std::uint32_t>& ids = m.train_ids();
            for (const std::uint32_t d : m.retrieval()->top(s.ctx, o.ret, m.kn().index(), prompt)) {
                ov->cut();
                const auto [a, b] = m.retrieval()->span(d);
                ov->push(ids[a]);
                for (std::size_t i = b; i-- > a + 1;) ov->push(ids[i]);
            }
            c->rov = std::move(ov);
        }
        s.rp = std::move(c);
        s.rp_seg.clear();
    }

    // step_word(): the word just written (its piece pb bytes long); at a sentence's end, the repair.
    void word(const K5& m, K5::State& s, const Options& o, std::size_t pb) const {
        if (!s.rp) return;
        const auto i = static_cast<std::uint32_t>(s.ctx.size() - 1);
        auto& seg = s.rp_seg;
        if (!seg.empty() && seg[seg.size() - 2] + 1 != i) seg.clear();  // not right after the last word K5 drew here
        seg.push_back(i);
        seg.push_back(static_cast<std::uint32_t>(s.out.size() - pb));
        if (!sentence_end(m.vocab()[s.ctx[i]]) && s.out.size() < o.bytes && s.step < o.max_words) return;
        const auto t0 = std::chrono::steady_clock::now();
        fix(m, s, o);
        seg.clear();
        st_.ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }

    // log2 Pf(y[j] | y[0, j)): K5 as it writes (retrieval overlay ov, the class floor).
    double fwd(const K5& m, const std::uint32_t* y, std::size_t j, const Overlay* ov) const {
        const double base = m.classes().on() ? m.classes().prob(y, j, y[j]) : m.kn().p1()[y[j]];
        return std::log2(std::max(m.kn().prob(y, j, y[j], base, ov), 1e-300));
    }
    // log2 Pb(y[j] | y(j, e]): the backward model given the up to order - 1 words after j, up to e.
    double bwd(const std::uint32_t* y, std::size_t j, std::size_t e, const Overlay* ov) const {
        std::uint32_t h[kMaxOrder];
        const std::size_t L = std::min<std::size_t>(static_cast<std::size_t>(bk_.order() - 1), e - j);
        for (std::size_t q = 0; q < L; ++q) h[q] = y[j + L - q];  // farthest first, the next word last
        const double base = bc_.on() ? bc_.prob(h, L, y[j]) : bk_.p1()[y[j]];
        return std::log2(std::max(bk_.prob(h, L, y[j], base, ov), 1e-300));
    }

 private:
    // Renders y[a, e] as step_word writes it (piece after the word before).
    static std::string render(const K5& m, const std::vector<std::uint32_t>& y, std::size_t a, std::size_t e) {
        std::string out;
        for (std::size_t k = a; k <= e; ++k) out += piece(m.vocab()[y[k]], m.vocab()[y[k - 1]], k - 1 > 0 || y[0] != kBos);
        return out;
    }
    // Does y (w at j) hold a training window of cap + 1 ids through j, within y[0, e]?
    static bool capped(const K5& m, const Options& o, const std::vector<std::uint32_t>& y, std::size_t j, std::size_t e) {
        if (!o.cap || m.cap() <= 0) return false;
        const auto n = static_cast<std::size_t>(m.cap() + 1);
        for (std::size_t q = j + 1 >= n ? j + 1 - n : 0; q <= j && q + n <= e + 1; ++q)
            if (m.kn().index().count(y.data() + q, n) > 0) return true;
        return false;
    }

    void fix(const K5& m, K5::State& s, const Options& o) const {
        const auto& seg = s.rp_seg;
        const auto& cont = *static_cast<const Cont*>(s.rp.get());
        const std::size_t a = seg[0], e = s.ctx.size() - 1, out0 = seg[1];
        ++st_.sentences;
        st_.words += e - a + 1;
        if (e < a + 1 || a < 1) return;  // nothing that may change
        std::vector<std::uint32_t> y = s.ctx;
        std::string_view old(s.out);
        old.remove_prefix(out0);
        RepeatBan rb(cont.prompt + s.out.substr(0, out0));
        if (render(m, y, a, e) != old || rb.banned(old)) {  // the sampler lifted the ban here, or the text is not K5's pieces
            ++st_.skipped;
            return;
        }
        const Overlay* fov = s.rov.get();
        const Overlay* bov = cont.rov.get();
        std::uint64_t h = kFnv0;
        for (std::size_t k = a >= 4 ? a - 4 : 0; k <= e; ++k) h = fnv_step(h, y[k]);
        std::mt19937_64 rng(mix64(h ^ 0x4E9A1D5Bull));
        const std::size_t V = m.vocab().size();
        std::vector<double> p(V), base(V);
        std::vector<std::uint32_t> idx, cand;
        std::vector<double> sf(e + 1), sb(e + 1), sc;
        std::vector<char> flag(e + 1);
        std::uint32_t hb[kMaxOrder];
        bool any_change = false;
        const auto topk = [&](int k) {
            idx.clear();
            for (std::uint32_t w = 2; w < V; ++w)
                if (!sentence_end(m.vocab()[w])) idx.push_back(w);
            const auto kk = std::min<std::size_t>(static_cast<std::size_t>(k), idx.size());
            std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(kk), idx.end(),
                              [&](std::uint32_t x, std::uint32_t z) { return p[x] != p[z] ? p[x] > p[z] : x < z; });
            for (std::size_t q = 0; q < kk; ++q)
                if (std::find(cand.begin(), cand.end(), idx[q]) == cand.end()) cand.push_back(idx[q]);
        };
        for (int sw = 0; sw < p_.sweeps; ++sw) {
            for (std::size_t j = a; j <= e; ++j) sf[j] = -fwd(m, y.data(), j, fov);
            for (std::size_t j = a - 1; j < e; ++j) sb[j] = -bwd(y.data(), j, e, bov);
            std::fill(flag.begin(), flag.end(), 0);
            std::size_t nf = 0;
            for (std::size_t j = a; j < e; ++j)
                if (std::min(sf[j], sb[j]) >= p_.tau) flag[j] = 1;
            for (std::size_t j = a - 1; j < e; ++j)
                if (std::min(sf[j + 1], sb[j]) >= p_.tau) {
                    if (j >= a) flag[j] = 1;
                    if (j + 1 < e) flag[j + 1] = 1;
                }
            for (std::size_t j = a; j < e; ++j) nf += flag[j];
            st_.flagged += nf;
            if (nf == 0) break;
            for (std::size_t j = a; j < e; ++j) {
                if (!flag[j]) continue;
                const std::uint32_t cur = y[j];
                cand.assign(1, cur);
                if (p_.kf > 0) {
                    if (m.classes().on()) m.classes().dist(y.data(), j, base.data());
                    m.kn().dist(y.data(), j, p.data(), fov, m.classes().on() ? base.data() : nullptr);
                    topk(p_.kf);
                }
                if (p_.kb > 0) {
                    const std::size_t L = std::min<std::size_t>(static_cast<std::size_t>(bk_.order() - 1), e - j);
                    for (std::size_t q = 0; q < L; ++q) hb[q] = y[j + L - q];
                    if (bc_.on()) bc_.dist(hb, L, base.data());
                    bk_.dist(hb, L, p.data(), bov, bc_.on() ? base.data() : nullptr);
                    topk(p_.kb);
                }
                sc.clear();
                std::size_t kept = 0;
                for (std::size_t c = 0; c < cand.size(); ++c) {
                    const std::uint32_t w = cand[c];
                    y[j] = w;
                    if (w != cur && (capped(m, o, y, j, e) || rb.banned(render(m, y, a, e)))) continue;
                    double f = 0, b = 0;
                    for (std::size_t q = j; q <= e && q <= j + 4; ++q) f += fwd(m, y.data(), q, fov);
                    for (std::size_t q = j >= 4 ? j - 4 : 0; q <= j; ++q)
                        if (q >= 1) b += bwd(y.data(), q, e, bov);
                    cand[kept++] = w;
                    sc.push_back(f + b);
                }
                cand.resize(kept);
                std::size_t pick = 0;
                if (p_.temp <= 0) {
                    for (std::size_t c = 1; c < cand.size(); ++c)
                        if (sc[c] > sc[pick]) pick = c;
                } else {
                    double mx = sc[0];
                    for (const double x : sc) mx = std::max(mx, x);
                    double z = 0;
                    for (double& x : sc) z += (x = std::exp2((x - mx) / p_.temp));
                    double u = std::uniform_real_distribution<double>(0.0, z)(rng);
                    for (pick = 0; pick + 1 < cand.size() && (u -= sc[pick]) > 0; ++pick) {}
                }
                y[j] = cand[pick];
                any_change = any_change || y[j] != cur;
            }
        }
        if (!any_change) return;
        std::size_t nch = 0;
        for (std::size_t j = a; j < e; ++j) nch += y[j] != s.ctx[j];
        if (nch == 0) return;
        st_.changed += nch;
        ++st_.repaired;
        const std::string fresh = render(m, y, a, e);
        std::copy(y.begin() + static_cast<std::ptrdiff_t>(a), y.begin() + static_cast<std::ptrdiff_t>(e), s.ctx.begin() + static_cast<std::ptrdiff_t>(a));
        s.out.resize(out0);
        s.out += fresh;
        rb.add(fresh);
        s.rep = std::move(rb);
    }

    Params p_;
    WordKN bk_;
    RevClass bc_;
    mutable Stats st_;
};

// Selftest: the flag, the forward probability with an overlay against dist, the backward model against dist, and one
// repair (the prompt and the sentence's end kept, the text re-rendered from the ids, the repeat ban rebuilt).
inline bool self_test(std::string& err) {
    Params p;
    if (!parse("on", p, err) || p.tau != 8 || p.temp != 2 || parse("off", p, err) || !err.empty() ||
        !parse("tau=6,t=1,kf=32,kb=0,sweeps=2", p, err) || p.tau != 6 || p.temp != 1 || p.kf != 32 || p.kb != 0 || p.sweeps != 2 ||
        parse("tau=x", p, err) || parse("kf=0,kb=0", p, err) || parse("sweeps=0", p, err) || parse("nope=1", p, err))
        return err = "repair: parse", false;
    if (Options{}.repair) return err = "repair: on by default", false;
    std::string text;
    const char* lines[] = {"The cat sat on the mat . The dog sat on the rug .", "A cat saw the dog . The dog saw a cat on the mat .",
                           "The old man sat on the old rug . A man saw the cat .", "The cat and the dog sat on the mat in the house ."};
    for (int r = 0; r < 6; ++r)
        for (const char* l : lines) text += std::string(l) + "\n";
    K5 m;
    m.build(text, 12, kOrder, 4);
    const std::size_t V = m.vocab().size();
    Overlay ov;
    ov.order = m.kn().order(), ov.rel = 0.5;
    for (const char* t : {"the", "dog", "sat", "on", "the", "mat", "."}) ov.push(m.id(t));
    std::vector<double> pd(V), bs(V);
    const std::vector<std::vector<std::string>> ctxs = {{"<s>", "The"}, {"the", "dog", "sat"}, {"sat", "on", "the"}, {"zz", "the"}, {}};
    for (const auto& c : ctxs) {
        std::vector<std::uint32_t> ids;
        for (const auto& t : c) ids.push_back(t == "<s>" ? kBos : m.id(t));
        for (const Overlay* o : {static_cast<const Overlay*>(nullptr), static_cast<const Overlay*>(&ov)}) {
            m.classes().dist(ids.data(), ids.size(), bs.data());
            m.kn().dist(ids.data(), ids.size(), pd.data(), o, bs.data());
            double z = 0;
            for (std::uint32_t w = 0; w < V; ++w) {
                const double q = m.kn().prob(ids.data(), ids.size(), w, m.classes().prob(ids.data(), ids.size(), w), o);
                if (std::abs(q - pd[w]) > 1e-12 * std::max(1.0, pd[w] * 1e3)) return err = "repair: prob with an overlay differs from dist", false;
                z += pd[w];
            }
            if (std::abs(z - 1) > 1e-9) return err = "repair: dist does not sum to one", false;
        }
    }
    Repairer rp(Params{});
    rp.build(m);
    {  // the backward model: "sat" before "on the mat" (read backwards: "mat the on" -> "sat")
        std::vector<std::uint32_t> y = {kBos, m.id("The"), m.id("cat"), m.id("sat"), m.id("on"), m.id("the"), m.id("mat")};
        const double lb = rp.bwd(y.data(), 3, 6, nullptr);
        std::vector<std::uint32_t> wrong = y;
        wrong[3] = m.id("house");
        if (!(lb > rp.bwd(wrong.data(), 3, 6, nullptr)) || lb > 0) return err = "repair: backward model", false;
        std::uint32_t h[3] = {m.id("mat"), m.id("the"), m.id("on")};
        rp.backward_classes().dist(h, 3, bs.data());
        rp.backward().dist(h, 3, pd.data(), nullptr, bs.data());
        double z = 0;
        for (const double x : pd) z += x;
        if (std::abs(z - 1) > 1e-9 || std::abs(std::log2(pd[m.id("sat")]) - lb) > 1e-9) return err = "repair: backward dist", false;
    }
    {  // one repair by hand: a broken sentence after a prompt
        Options o;
        o.repair = std::make_shared<Repairer>(Params{0.5, 0, 8, 8, 3});
        o.repair->build(m);
        const std::string prompt = "The dog saw";
        K5::State s = m.start(prompt, o);
        if (!s.rp) return err = "repair: begin", false;
        for (const char* t : {"rug", "on", "sat", "cat", "the", "mat", "."}) {
            const std::uint32_t w = m.id(t);
            const std::string pc = piece(m.vocab()[w], s.prev, s.has_prev);
            s.out += pc, s.rep.add(pc);
            K5::learn(s, w, o);
            s.prev = m.vocab()[w], s.has_prev = true, ++s.step;
            o.repair->word(m, s, o, pc.size());
        }
        const auto& st = o.repair->stats();
        if (st.sentences != 1 || st.repaired != 1 || st.changed == 0) return err = "repair: no change made", false;
        const std::vector<std::string> pt = tokens(prompt), ot = tokens(s.out);
        if (s.ctx.size() != 1 + pt.size() + ot.size() || ot.back() != "." || m.id(pt.back()) != s.ctx[pt.size()]) return err = "repair: ids", false;
        for (std::size_t k = 0; k < ot.size(); ++k)
            if (m.id(ot[k]) != s.ctx[1 + pt.size() + k]) return err = "repair: text and ids differ", false;
        RepeatBan ref(prompt + s.out);
        for (const char* probe : {" the mat .", " cat", " on the", " sat on the mat ."})
            if (ref.banned(probe) != s.rep.banned(probe)) return err = "repair: repeat ban not rebuilt", false;
    }
    err.clear();
    return true;
}

}  // namespace cypha::cyphalm::wkn::repair

namespace cypha::cyphalm::wkn {
// K5's hooks (word_kn.hpp declares them; K5_REPAIR.md).
inline void K5::repair_build(const Options& o) { o.repair->build(*this); }
inline void K5::repair_begin(std::string_view prompt, State& s, const Options& o) const { o.repair->begin(*this, prompt, s, o); }
inline void K5::repair_word(State& s, const Options& o, std::size_t pb) const { o.repair->word(*this, s, o, pb); }
}  // namespace cypha::cyphalm::wkn
