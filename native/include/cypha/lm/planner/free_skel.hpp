#pragma once

/// Skeleton-first free sentences (docs/reports/K5_SKELFREE.md, GENERATION_PLAN D3).
///   sample --k5 TEXT ... --k5-skelfree bank=FILE[,n=N][,k5=M][,pool=P][,fills=F]
///
/// Today's free sentences are K5's left-to-right drafts, chosen by the sentence gate. Here some or all of the gate's
/// drafts are written skeleton-first instead:
/// 1. The bank. Whole real sentences of clean enwik9 (train_all.txt), each as its sequence of K5 exchange classes,
///    with its count (k5_skelfree mine). Not tied to a relation; nothing hand-written (rule 10).
/// 2. The choice. Per sentence, `pool` skeletons are drawn from the bank in proportion to their counts (a random real
///    sentence's shape), each is scored by the structure statistics' class 5-gram after the text's last four classes
///    (mean over its codes) plus a quarter of the rhythm expert's length model after the sentence before, and the best
///    `n` not yet used in this continuation are kept.
/// 3. The fill. Each kept skeleton is filled left to right by K5 on word IDs: at each class code, a word of that class
///    drawn from K5's distribution (the retrieval overlay and every online part included, as model_dist gives it)
///    renormalised over the class, nucleus as K5's own (o.nucleus), with the 12-word copy cap and the repeat ban as
///    K5's sampler applies them. With fills > 1 each skeleton is filled F times and the fill with the highest mean
///    log2 P under K5 is kept. A fill that the cap leaves no word for is dropped.
/// 4. The gate. The skeleton drafts join (k5 > 0) or replace (k5 = 0) K5's own drafts in the gate that chooses today
///    (the plain path's z-scored gate, the discourse writer's paragraph gate), unchanged.
/// Skeleton drafts are only made at a sentence start (the text so far ends with . ! ? or a newline); otherwise the
/// gate's drafts are K5's own.
///
/// Off (no flag): nothing here runs, and the output is byte-identical.
///
/// Author: Odin Loch

#include "cypha/lm/structure/experts.hpp"
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace cypha::cyphalm::planner {

struct FreeSkelParams {
    std::string bank;  // the bank file (k5_skelfree mine)
    int n = 16;        // skeleton drafts per sentence
    int k5 = 0;        // K5's own drafts beside them (the first k5 of its streams)
    int pool = 256;    // skeletons drawn from the bank per sentence before scoring
    int fills = 1;     // fills per skeleton (the most fluent kept)
    std::string describe() const {
        char b[256];
        std::snprintf(b, sizeof b, "n=%d,k5=%d,pool=%d,fills=%d", n, k5, pool, fills);
        return b;
    }
};

// "bank=FILE,n=N,k5=M,pool=P,fills=F".
inline bool parse_free_skel(const std::string& spec, FreeSkelParams& p, std::string& err) {
    std::size_t a = 0;
    while (a < spec.size()) {
        std::size_t e = spec.find(',', a);
        if (e == std::string::npos) e = spec.size();
        const std::string f = spec.substr(a, e - a);
        a = e + 1;
        if (f.empty()) continue;
        const std::size_t q = f.find('=');
        if (q == std::string::npos) return err = "--k5-skelfree: " + f + " is not KEY=VALUE", false;
        const std::string k = f.substr(0, q), v = f.substr(q + 1);
        if (k == "bank") {
            p.bank = v;
            continue;
        }
        char* end = nullptr;
        const long x = std::strtol(v.c_str(), &end, 10);
        if (v.empty() || *end || x < 0 || x > 4096) return err = "--k5-skelfree: " + k + " wants a whole number in [0, 4096]", false;
        if (k == "n") p.n = static_cast<int>(x);
        else if (k == "k5") p.k5 = static_cast<int>(x);
        else if (k == "pool") p.pool = static_cast<int>(x);
        else if (k == "fills") p.fills = static_cast<int>(x);
        else return err = "--k5-skelfree: unknown key " + k, false;
    }
    if (p.bank.empty()) return err = "--k5-skelfree: bank=FILE is required", false;
    if (p.n + p.k5 < 1 || p.n + p.k5 > 64) return err = "--k5-skelfree: n + k5 in [1, 64]", false;
    if (p.fills < 1 || p.pool < 1) return err = "--k5-skelfree: fills and pool at least 1", false;
    return true;
}

// The bank: "K5FREE-SKEL 1", then one skeleton a line, "COUNT c1 c2 ..." (K5 class codes), most frequent first.
struct FreeBank {
    std::vector<std::vector<std::uint16_t>> shapes;
    std::vector<std::uint32_t> count;
    std::vector<double> cum;  // cumulative counts (draws in proportion to the count)
    std::uint64_t total = 0;
    bool load(const std::string& path, std::string& err) {
        std::ifstream in(path);
        std::string line;
        if (!std::getline(in, line) || line != "K5FREE-SKEL 1") return err = "cannot read the free skeleton bank " + path, false;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::uint32_t n = 0;
            if (!(ls >> n)) continue;
            std::vector<std::uint16_t> c;
            unsigned x = 0;
            while (ls >> x) c.push_back(static_cast<std::uint16_t>(x));
            if (c.empty() || n == 0) continue;
            shapes.push_back(std::move(c)), count.push_back(n);
            total += n;
            cum.push_back(static_cast<double>(total));
        }
        if (shapes.empty()) return err = "empty free skeleton bank " + path, false;
        return true;
    }
    std::size_t draw(std::mt19937_64& rng) const {
        const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53 * static_cast<double>(total);
        const auto it = std::upper_bound(cum.begin(), cum.end(), u);
        return std::min<std::size_t>(static_cast<std::size_t>(it - cum.begin()), shapes.size() - 1);
    }
};

class FreeSkel {
 public:
    FreeSkel(std::shared_ptr<const FreeBank> bank, FreeSkelParams p) : bank_(std::move(bank)), p_(std::move(p)) {}
    const FreeSkelParams& params() const { return p_; }
    const FreeBank& bank() const { return *bank_; }

    // Does the text so far end a sentence (a skeleton may start)?
    static bool at_start(const wkn::K5& m, const wkn::K5::State& s) {
        if (s.ctx.empty() || s.ctx.back() == wkn::kBos) return true;
        const std::string& t = m.vocab()[s.ctx.back()];
        return t == "." || t == "!" || t == "?" || t == "\n";
    }

    // Step 2: up to n bank indices for the next sentence after s, best first, none in `used`.
    std::vector<std::size_t> choose(const wkn::K5& m, const structure::Stats& stats, const wkn::K5::State& s, std::uint64_t seed, std::uint64_t sent,
                                    const std::set<std::size_t>& used) const {
        namespace st = structure;
        std::mt19937_64 rng(wkn::mix64(seed ^ wkn::mix64(0x5EF0000000ull + sent)));
        std::vector<std::size_t> pool;
        for (int i = 0; i < p_.pool; ++i) {
            const std::size_t b = bank_->draw(rng);
            if (!used.count(b) && std::find(pool.begin(), pool.end(), b) == pool.end()) pool.push_back(b);
        }
        // the context's last four classes and the sentence before (its length bucket)
        std::vector<std::uint32_t> h0;
        for (std::size_t i = s.ctx.size() > 4 ? s.ctx.size() - 4 : 0; i < s.ctx.size(); ++i)
            if (s.ctx[i] != wkn::kBos) h0.push_back(stats.cls(s.ctx[i]));
        const auto last = m.last_sentence(s.ctx);
        int plen = st::kLenNone;
        if (!last.empty() && !(last.size() == 1 && last[0] == wkn::kBos)) {
            st::Sent z;
            z.tok.resize(last.size());
            const std::string& t = m.vocab()[last.back()];
            z.ended = t == "." || t == "!" || t == "?";
            plen = st::len_bucket(z);
        }
        std::vector<std::pair<double, std::size_t>> sc;
        for (const auto b : pool) {
            const auto& cd = bank_->shapes[b];
            std::vector<std::uint32_t> h = h0;
            double lp = 0;
            for (const auto c : cd) {
                lp += stats.class_lp(h.data(), h.size(), c);
                h.push_back(c);
                if (h.size() > st::Stats::kCO - 1) h.erase(h.begin());
            }
            st::Sent z;
            z.tok.resize(cd.size()), z.ended = true;
            sc.push_back({lp / static_cast<double>(cd.size()) + stats.len_lp(plen, st::len_bucket(z)) / 4.0, b});
        }
        std::stable_sort(sc.begin(), sc.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        std::vector<std::size_t> out;
        for (std::size_t i = 0; i < sc.size() && out.size() < static_cast<std::size_t>(p_.n); ++i) out.push_back(sc[i].second);
        return out;
    }

    // Step 3: one fill of class codes after s into c (c = s, then the sentence). words, lp: its K5 ids and their log2
    // probabilities under K5 (the gate's fluency). False if the copy cap leaves no word at a position.
    bool fill(const wkn::K5& m, const wkn::Options& o, const wkn::K5::State& s, const std::vector<std::uint16_t>& codes, std::mt19937_64& rng,
              wkn::K5::State& c, std::vector<std::uint32_t>& words, std::vector<double>& lp, wkn::GenStats& g) const {
        const auto& byc = by_class(m);
        c = s;
        words.clear(), lp.clear();
        thread_local std::vector<double> p, tmp;
        thread_local std::vector<std::pair<double, std::uint32_t>> top;
        thread_local std::vector<std::string> pcs;
        p.resize(m.vocab().size());
        const wkn::NgramSet cs = m.cap_set();
        for (const auto code : codes) {
            if (code >= byc.size() || byc[code].empty()) return false;
            m.model_dist(c, o, p.data(), tmp);
            const double z = std::max(1 - p[wkn::kUnk] - p[wkn::kBos], 1e-300);
            top.clear();
            for (const auto w : byc[code])
                if (p[w] > 0) top.push_back({p[w], w});
            if (top.empty()) return false;
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
            wkn::NgramSet::Index::Range hp;
            const bool capped = o.cap && cs.prefix(c.ctx, hp);
            // bans, as K5's sampler: the cap always; the repeat ban unless it leaves nothing
            std::vector<char> rb(top.size(), 0), cb(top.size(), 0);
            pcs.resize(top.size());
            bool any_ok = false, any_kept = false;
            for (std::size_t i = 0; i < top.size(); ++i) {
                pcs[i] = wkn::piece(m.vocab()[top[i].second], c.prev, c.has_prev);
                cb[i] = capped && cs.completes(hp, top[i].second);
                rb[i] = c.rep.banned(pcs[i]);
                any_ok = any_ok || !cb[i];
                any_kept = any_kept || (!cb[i] && !rb[i]);
            }
            if (!any_ok) {
                ++g.cap_outside;
                return false;
            }
            if (!any_kept) {
                ++g.rep_lifted;
                std::fill(rb.begin(), rb.end(), 0);
            }
            double zk = 0;
            for (std::size_t i = 0; i < top.size(); ++i)
                if (!rb[i] && !cb[i]) zk += top[i].first;
            // nucleus over the kept words of the class, then one draw
            std::vector<std::size_t> kept;
            double c2 = 0;
            for (std::size_t i = 0; i < top.size(); ++i) {
                if (rb[i] || cb[i]) continue;
                kept.push_back(i);
                c2 += top[i].first / zk;
                if (c2 >= o.nucleus) break;
            }
            double zn = 0;
            for (const auto i : kept) zn += top[i].first;
            const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;
            std::size_t j = 0;
            double cdf = 0;
            for (; j + 1 < kept.size(); ++j) {
                cdf += top[kept[j]].first / zn;
                if (cdf > u) break;
            }
            const std::size_t i = kept[j];
            const std::uint32_t w = top[i].second;
            c.out += pcs[i];
            c.rep.add(pcs[i]);
            wkn::K5::learn(c, w, o);
            c.prev = m.vocab()[w], c.has_prev = true;
            ++c.step, ++g.steps;
            words.push_back(w), lp.push_back(std::log2(std::max(top[i].first / z, 1e-300)));
        }
        return true;
    }

    // Steps 2-3: the skeleton drafts for the next sentence after s (stream `sent`), at most n; each a state, its words
    // and their log2 probabilities. Marks the skeletons used.
    struct Draft {
        wkn::K5::State st;
        std::vector<std::uint32_t> words;
        std::vector<double> lp;
        wkn::GenStats g;
        double mean() const {
            double t = 0;
            for (const double x : lp) t += x;
            return lp.empty() ? -1e9 : t / static_cast<double>(lp.size());
        }
    };
    std::vector<Draft> drafts(const wkn::K5& m, const structure::Stats& stats, const wkn::Options& o, const wkn::K5::State& s, std::uint64_t seed,
                              std::uint64_t sent, std::set<std::size_t>& used) const {
        std::vector<Draft> out;
        if (p_.n <= 0 || !at_start(m, s)) return out;
        const auto pick = choose(m, stats, s, seed, sent, used);
        for (std::size_t j = 0; j < pick.size(); ++j) {
            Draft best;
            bool have = false;
            for (int f = 0; f < p_.fills; ++f) {
                std::mt19937_64 rng(wkn::mix64(seed ^ wkn::mix64(0x5F11000000ull + sent * 4096 + j * 64 + static_cast<std::uint64_t>(f))));
                Draft d;
                if (!fill(m, o, s, bank_->shapes[pick[j]], rng, d.st, d.words, d.lp, d.g)) continue;
                if (!have || d.mean() > best.mean()) best = std::move(d), have = true;
            }
            if (!have) continue;
            ++best.g.gate_cands;
            out.push_back(std::move(best));
            used.insert(pick[j]);
        }
        n_drafts_ += out.size(), ++n_sents_;
        return out;
    }
    // (A skeleton offered once is not offered again in the continuation, chosen or not: the bank is large, and it keeps
    // the paragraph's shapes from repeating.)

    std::uint64_t n_drafts() const { return n_drafts_; }
    std::uint64_t n_sents() const { return n_sents_; }
    mutable std::atomic<std::uint64_t> chosen_skel{0}, chosen_k5{0};

 private:
    // K5's words by class (no newline), once per model.
    const std::vector<std::vector<std::uint32_t>>& by_class(const wkn::K5& m) const {
        std::call_once(once_, [&] {
            const auto& cls = m.classes().classes();
            for (std::uint32_t w = 2; w < cls.size(); ++w) {
                if (m.vocab()[w] == "\n") continue;
                if (byc_.size() <= cls[w]) byc_.resize(cls[w] + 1);
                byc_[cls[w]].push_back(w);
            }
        });
        return byc_;
    }
    std::shared_ptr<const FreeBank> bank_;
    FreeSkelParams p_;
    mutable std::once_flag once_;
    mutable std::vector<std::vector<std::uint32_t>> byc_;
    mutable std::atomic<std::uint64_t> n_drafts_{0}, n_sents_{0};
};

// The plain path (no plan) with skeleton drafts: K5's z-scored sentence gate (word_kn.hpp gate_z, the default's: the
// prompt association, the LDA topic term and the markup share; first_k, MBR and the entity grid not taken), its
// drafts the first k5 of K5's own streams (m.draft, the same streams gate_z draws) and the skeleton drafts.
inline std::string generate_free_plain(const wkn::K5& m, const FreeSkel& fk, const structure::Stats& stats, std::string_view prompt, std::uint64_t seed,
                                       const wkn::Options& o, wkn::GenStats* gst) {
    wkn::GenStats g;
    if (m.vocab().size() <= 2) return {};
    wkn::K5::State s = m.start(prompt, o);
    const std::vector<double> assoc = o.gate_wp != 0 ? m.prompt_assoc(s.ctx) : std::vector<double>{};
    const std::vector<double> tlr = o.lda_w != 0 && m.topic_model().trained() ? m.topic_lr(prompt) : std::vector<double>{};
    std::set<std::size_t> used;
    const int nk = fk.params().k5;
    const auto zs = [](std::vector<double>& x) {
        double mu = 0, v = 0;
        for (const double a : x) mu += a;
        mu /= static_cast<double>(x.size());
        for (const double a : x) v += (a - mu) * (a - mu);
        const double sd = std::sqrt(v / static_cast<double>(x.size()));
        for (double& a : x) a = sd > 1e-12 ? (a - mu) / sd : 0.0;
    };
    std::vector<std::uint32_t> words;
    std::vector<double> lps;
    for (std::uint64_t sent = 0; s.step < o.max_words && s.out.size() < o.bytes; ++sent) {
        std::vector<FreeSkel::Draft> ds = fk.drafts(m, stats, o, s, seed, sent, used);
        const std::size_t nsk = ds.size();
        // K5's own drafts: the first k5 of its streams, or all gate_k when no skeleton draft could be made
        const int want = nsk == 0 ? std::max(o.gate_k, 1) : nk;
        for (int k = 0; k < want; ++k) {
            FreeSkel::Draft d;
            m.draft(s, seed, sent, k, o, d.st, d.words, d.lp, d.g);
            ds.push_back(std::move(d));
        }
        if (ds.empty()) break;
        const std::size_t K = ds.size();
        std::vector<double> tf(K), tp(K), tt(K), mk(K);
        for (std::size_t k = 0; k < K; ++k) {
            const auto& d = ds[k];
            const std::size_t b0 = s.out.size();
            double slp = 0, spmi = 0, stop = 0;
            for (std::size_t i = 0; i < d.words.size(); ++i) {
                slp += d.lp[i];
                spmi += assoc.empty() ? 0.0 : assoc[d.words[i]];
                stop += tlr.empty() ? 0.0 : tlr[d.words[i]];
            }
            std::size_t mm = 0;
            for (std::size_t i = b0; i < d.st.out.size(); ++i) mm += std::string_view("[]{}|=<>&*#").find(d.st.out[i]) != std::string_view::npos;
            const double nb = static_cast<double>(std::max<std::size_t>(d.st.out.size() - b0, 1));
            const double n = static_cast<double>(std::max<std::size_t>(d.words.size(), 1));
            tf[k] = slp / n, tp[k] = spmi / n, tt[k] = stop / n, mk[k] = static_cast<double>(mm) / nb;
        }
        zs(tf), zs(tp), zs(tt);
        std::size_t best = 0;
        double bscore = 0;
        for (std::size_t k = 0; k < K; ++k) {
            const double score = o.gate_wf * tf[k] + o.gate_wp * tp[k] + o.lda_w * tt[k] - o.gate_wm * mk[k];
            if (k == 0 || score > bscore) bscore = score, best = k;
        }
        (best < nsk ? fk.chosen_skel : fk.chosen_k5)++;
        s = std::move(ds[best].st);
        g.add(ds[best].g);
        ++g.gate_sents;
    }
    std::string out = std::move(s.out);
    wkn::K5::cut_bytes(out, o.bytes);
    if (gst) gst->add(g);
    return out;
}

// Selftest: the spec parser.
inline bool free_skel_self_test(std::string& err) {
    FreeSkelParams p;
    if (!parse_free_skel("bank=x.txt,n=8,k5=8,pool=100,fills=2", p, err) || p.bank != "x.txt" || p.n != 8 || p.k5 != 8 || p.pool != 100 || p.fills != 2)
        return err = "skelfree: spec", false;
    FreeSkelParams q;
    std::string e2;
    if (parse_free_skel("n=8", q, e2) || parse_free_skel("bank=x,n=-1", q, e2) || parse_free_skel("bank=x,zz=1", q, e2) ||
        parse_free_skel("bank=x,n=0,k5=0", q, e2))
        return err = "skelfree: bad spec accepted", false;
    return true;
}

}  // namespace cypha::cyphalm::planner
