#pragma once

/// CyphaLM's planner: plan -> skeleton -> fill -> gate (docs/reports/K5_PLANNER.md, GENERATION_PLAN §13.37-13.38).
///   sample --k5 TEXT --k5-plan STOREDIR --k5-plan-model MODEL [--k5-plan-gate full|k5] [--plan-drop LO:HI]
///
/// 1. Plan. The prompt's main entity (the content plan's entity finder, unchanged) and its facts in CyphaLM's fact
///    store, as fact IDs. Orders of up to four facts are scored by the bank's relation transitions and by the prompt's
///    association with each fact's value; the best order is written. No facts: the facts whose value is a prompt entity
///    (co-occurring entities). Neither: plain K5 writes the prompt. The path is logged per prompt.
/// 2. Skeleton. The relation's skeletons (class codes with S and O slots, skeleton.hpp), scored by the structure
///    experts' class 5-gram and the rhythm expert's length model, the best kSkel kept.
/// 3. Fill. A beam over K5 word IDs: each class code takes a word of that class (K5's probabilities, the repeat ban and
///    the copy cap as in K5's own sampler), the slots take the fact's tokens. --k5-plan-fill dba: dynamic beam
///    allocation instead (Filler::fill_dba, docs/reports/MERGE_FILL.md); --k5-plan-beam N: the beam (default 4).
/// 4. Gate. Every structure expert (structure/experts.hpp) plus an MBR consensus on word IDs, joined by a meta-mixer
///    trained to rank the real sentence's own fill above the other fills of the same fact (k5_planner train), with a K5
///    fluency floor; or K5 alone (--k5-plan-gate k5, the P2 ablation). Only the winner becomes text.
/// After the plan, K5's own sentence gate writes on to the byte limit.
///
/// The experts read candidates as K5 tokens (vocabulary entries of the IDs); no candidate is rendered to bytes.
///
/// Author: Odin Loch

#include "cypha/lm/planner/fallback.hpp"  // PLAN_FACTS: --k5-plan-fallback pattern
#include "cypha/lm/planner/facts.hpp"
#include "cypha/lm/planner/skeleton.hpp"
#include "cypha/lm/structure/experts.hpp"
#include "cypha/lm/structure/mixer.hpp"
#include "cypha/lm/structure/writer.hpp"
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace cypha::cyphalm::planner {

namespace st = cypha::cyphalm::structure;

inline constexpr int kMbr = st::kNFeat;      // the extra feature: MBR consensus over the step's fills (word-ID bigrams)
inline constexpr int kNF = st::kNFeat + 1;
inline constexpr int kMaxPlan = 4;           // facts in one plan
inline constexpr int kSkel = 8;              // skeletons filled per fact

struct Fill {
    std::vector<std::uint32_t> ids;
    std::vector<std::string> tok;
    double lp = 0;  // sum of log2 P(word) under K5
    bool ok = false;
    double mean() const { return tok.empty() ? -99.0 : lp / static_cast<double>(tok.size()); }
};

// Bank sizes for dynamic beam allocation (Post and Vilar, NAACL 2018): the beam split evenly over banks 0..C (C
// constraint tokens), the remainder to the top bank. Bank adjustment, from the top bank down: a bank with more slots
// than candidates gives the spare slots to the nearest banks that have more candidates than slots (the lower one first
// at equal distance). Sockeye's single leftward pass can leave slots unused; this does not while candidates remain.
inline std::vector<int> bank_sizes(int C, int beam, const std::vector<int>& count) {
    const int nb = C + 1, each = beam / nb;
    std::vector<int> a(static_cast<std::size_t>(nb), each);
    a.back() += beam - each * nb;
    const auto at = [](std::vector<int>& v, int i) -> int& { return v[static_cast<std::size_t>(i)]; };
    std::vector<int> n = count;
    for (int i = nb - 1; i >= 0; --i) {
        int over = at(a, i) - at(n, i);
        if (over <= 0) continue;
        at(a, i) = at(n, i);
        for (int d = 1; d < nb && over > 0; ++d)
            for (const int j : {i - d, i + d})
                if (j >= 0 && j < nb && at(n, j) > at(a, j)) {
                    const int take = std::min(over, at(n, j) - at(a, j));
                    at(a, j) += take, over -= take;
                }
    }
    return a;
}

// The constrained beam (step 3).
class Filler {
 public:
    Filler(const wkn::K5& m, const wkn::Options& o, int beam = 4, int expand = 3, bool dba = false)
        : m_(m), o_(o), beam_(beam), expand_(expand), dba_(dba) {
        const auto& cls = m.classes().classes();
        for (std::uint32_t w = 2; w < cls.size(); ++w) {
            if (m.vocab()[w] == "\n") continue;
            if (by_cls_.size() <= cls[w]) by_cls_.resize(cls[w] + 1);
            by_cls_[cls[w]].push_back(w);
        }
    }
    // K5 distributions computed by this filler (the fill's cost; GENERATION_PLAN §13.40 M3).
    std::uint64_t dists() const { return dists_; }
    Fill fill(const wkn::K5::State& s, const std::vector<std::uint16_t>& codes, const std::vector<std::string>& S,
              const std::vector<std::string>& O) const {
        if (dba_) return fill_dba(s, codes, S, O);
        struct It {
            std::vector<std::uint32_t> ids;
            std::vector<std::string> tok;
            double lp = 0;
            std::string seg, prev;
            bool has_prev = false;
        };
        std::vector<It> cur(1), nxt;
        cur[0].prev = s.prev, cur[0].has_prev = s.has_prev;
        thread_local wkn::K5::State tmp;
        thread_local std::vector<double> p, scratch;
        p.resize(m_.vocab().size());
        const auto& cs = m_.cap_set();
        const auto dist = [&](const It& it) {
            tmp.ctx = s.ctx;
            tmp.ctx.insert(tmp.ctx.end(), it.ids.begin(), it.ids.end());
            m_.model_dist(tmp, o_, p.data(), scratch);
            ++dists_;
            return std::max(1 - p[wkn::kUnk] - p[wkn::kBos], 1e-12);
        };
        for (const auto code : codes) {
            nxt.clear();
            for (const auto& it : cur) {
                if (code == kSlotS || code == kSlotO) {
                    It x = it;
                    for (const auto& t : code == kSlotS ? S : O) {
                        const double z = dist(x);
                        const std::uint32_t id = m_.id(t);
                        x.lp += std::log2(std::max(p[id] / (id == wkn::kUnk ? 1.0 : z), 1e-12));
                        x.ids.push_back(id), x.tok.push_back(t);
                        x.prev = t, x.has_prev = true;
                    }
                    x.seg.clear();
                    nxt.push_back(std::move(x));
                    continue;
                }
                if (code >= by_cls_.size()) continue;
                const double z = dist(it);
                std::uint64_t hp = 0;
                const bool capped = o_.cap && cs.prefix(tmp.ctx, hp);
                std::vector<std::pair<double, std::uint32_t>> top;
                for (const auto w : by_cls_[code])
                    if (p[w] > 0) top.push_back({p[w], w});
                std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
                int took = 0;
                for (const auto& [pw, w] : top) {
                    if (took >= expand_) break;
                    if (capped && cs.has(wkn::fnv_step(hp, w))) continue;
                    const std::string pc = wkn::piece(m_.vocab()[w], it.prev, it.has_prev);
                    if (s.rep.banned(it.seg + pc)) continue;
                    It x = it;
                    x.lp += std::log2(std::max(pw / z, 1e-12));
                    x.ids.push_back(w), x.tok.push_back(m_.vocab()[w]);
                    x.seg += pc, x.prev = m_.vocab()[w], x.has_prev = true;
                    nxt.push_back(std::move(x));
                    ++took;
                }
            }
            if (nxt.empty()) return {};
            std::stable_sort(nxt.begin(), nxt.end(), [](const It& a, const It& b) { return a.lp > b.lp; });
            if (nxt.size() > static_cast<std::size_t>(beam_)) nxt.resize(static_cast<std::size_t>(beam_));
            cur.swap(nxt);
        }
        Fill f;
        f.ids = std::move(cur[0].ids), f.tok = std::move(cur[0].tok), f.lp = cur[0].lp, f.ok = true;
        return f;
    }

    // Dynamic beam allocation (Post and Vilar, NAACL 2018; GENERATION_PLAN §13.40 M3; docs/reports/MERGE_FILL.md).
    // The fact's tokens are lexical constraints: S's and O's, C tokens in all. Before any class position ahead of a
    // name's slot, a hypothesis may start that name; a started name runs to its end; at its slot, a name not yet written
    // starts, and the slot of a written one is skipped. The slot is a deadline, so fill's placements stay in the search.
    // Every step writes one token for every hypothesis (a class word or a name's token), so all finish together with
    // every name written. One beam of beam_ is split over banks 0..C by the constraint tokens met (bank_sizes): a
    // hypothesis that has paid for a name early is not pruned by those that have not. Class words as in fill (the best
    // expand_ of the class by K5, the copy cap, the repeat ban); a name's token costs log2 of its K5 probability, as at
    // a slot. Once started, a name is not broken off (Sockeye may break a phrase and unwind it).
    Fill fill_dba(const wkn::K5::State& s, const std::vector<std::uint16_t>& codes, const std::vector<std::string>& S,
                  const std::vector<std::string>& O) const {
        const std::vector<std::string>* con[2] = {&S, &O};
        bool slot[2] = {false, false};
        std::size_t ncls = 0;
        for (const auto c : codes) {
            slot[0] = slot[0] || c == kSlotS, slot[1] = slot[1] || c == kSlotO;
            ncls += c != kSlotS && c != kSlotO;
        }
        const std::size_t len[2] = {slot[0] ? S.size() : 0, slot[1] ? O.size() : 0};  // a name without a slot is not written (as fill)
        const int C = static_cast<int>(len[0] + len[1]);
        struct It {
            std::vector<std::uint32_t> ids;
            std::vector<std::string> tok;
            double lp = 0;
            std::string seg, prev;
            bool has_prev = false;
            std::size_t j = 0;             // the next skeleton position
            int act = -1;                  // the name being written (0 S, 1 O), or -1
            std::size_t done[2] = {0, 0};  // tokens of S and O written
            int met = 0;                   // constraint tokens written: the bank
        };
        const auto name_of = [](std::uint16_t code) { return code == kSlotS ? 0 : code == kSlotO ? 1 : -1; };
        std::vector<It> cur(1), nxt;
        cur[0].prev = s.prev, cur[0].has_prev = s.has_prev;
        thread_local wkn::K5::State tmp;
        thread_local std::vector<double> p, scratch;
        p.resize(m_.vocab().size());
        const auto& cs = m_.cap_set();
        const auto dist = [&](const It& it) {
            tmp.ctx = s.ctx;
            tmp.ctx.insert(tmp.ctx.end(), it.ids.begin(), it.ids.end());
            m_.model_dist(tmp, o_, p.data(), scratch);
            ++dists_;
            return std::max(1 - p[wkn::kUnk] - p[wkn::kBos], 1e-12);
        };
        const auto settle = [&](It& it) {  // past the slots of written names
            for (int c; it.act < 0 && it.j < codes.size() && (c = name_of(codes[it.j])) >= 0 && it.done[c] == len[c];) ++it.j;
        };
        const auto put = [&](const It& it, int c, double z) {  // it writes name c's next token (p: its distribution)
            It x = it;
            const std::string& t = (*con[c])[x.done[c]];
            const std::uint32_t id = m_.id(t);
            x.lp += std::log2(std::max(p[id] / (id == wkn::kUnk ? 1.0 : z), 1e-12));
            x.ids.push_back(id), x.tok.push_back(t);
            x.prev = t, x.has_prev = true;
            x.seg.clear();
            ++x.done[c], ++x.met;
            x.act = x.done[c] < len[c] ? c : -1;
            return x;
        };
        std::vector<int> cnt(static_cast<std::size_t>(C) + 1);
        std::vector<std::pair<double, std::uint32_t>> top;
        for (std::size_t step = 0; step < ncls + static_cast<std::size_t>(C); ++step) {
            nxt.clear();
            for (auto& it : cur) {
                settle(it);
                if (it.act < 0 && it.j >= codes.size()) continue;  // not reached: every hypothesis writes ncls + C tokens
                if (it.act < 0 && name_of(codes[it.j]) < 0 && codes[it.j] >= by_cls_.size()) continue;
                const double z = dist(it);
                if (it.act >= 0) {  // the name runs on
                    nxt.push_back(put(it, it.act, z));
                    continue;
                }
                const auto code = codes[it.j];
                if (name_of(code) >= 0) {  // its slot: the deadline
                    It x = put(it, name_of(code), z);
                    x.j = it.j + 1;
                    nxt.push_back(std::move(x));
                    continue;
                }
                std::uint64_t hp = 0;
                const bool capped = o_.cap && cs.prefix(tmp.ctx, hp);
                top.clear();
                for (const auto w : by_cls_[code])
                    if (p[w] > 0) top.push_back({p[w], w});
                std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
                int took = 0;
                for (const auto& [pw, w] : top) {
                    if (took >= expand_) break;
                    if (capped && cs.has(wkn::fnv_step(hp, w))) continue;
                    const std::string pc = wkn::piece(m_.vocab()[w], it.prev, it.has_prev);
                    if (s.rep.banned(it.seg + pc)) continue;
                    It x = it;
                    x.lp += std::log2(std::max(pw / z, 1e-12));
                    x.ids.push_back(w), x.tok.push_back(m_.vocab()[w]);
                    x.seg += pc, x.prev = m_.vocab()[w], x.has_prev = true;
                    x.j = it.j + 1;
                    nxt.push_back(std::move(x));
                    ++took;
                }
                for (int c = 0; c < 2; ++c)  // a name early, before this class position (its slot is ahead)
                    if (len[c] > 0 && it.done[c] == 0) nxt.push_back(put(it, c, z));
            }
            if (nxt.empty()) return {};
            std::stable_sort(nxt.begin(), nxt.end(), [](const It& a, const It& b) { return a.lp > b.lp; });
            std::fill(cnt.begin(), cnt.end(), 0);
            for (const auto& x : nxt) ++cnt[static_cast<std::size_t>(x.met)];
            auto room = bank_sizes(C, beam_, cnt);
            cur.clear();
            for (auto& x : nxt)
                if (room[static_cast<std::size_t>(x.met)] > 0) --room[static_cast<std::size_t>(x.met)], cur.push_back(std::move(x));
        }
        const It* best = nullptr;
        for (auto& it : cur) {
            settle(it);
            if (it.met == C && it.act < 0 && it.j >= codes.size() && (!best || it.lp > best->lp)) best = &it;
        }
        if (!best) return {};
        Fill f;
        f.ids = best->ids, f.tok = best->tok, f.lp = best->lp, f.ok = true;
        return f;
    }

 private:
    const wkn::K5& m_;
    const wkn::Options& o_;
    int beam_, expand_;
    bool dba_ = false;
    mutable std::uint64_t dists_ = 0;
    std::vector<std::vector<std::uint32_t>> by_cls_;
};

// MBR consensus on word IDs: each fill's mean bigram F1 against the other fills.
inline std::vector<double> mbr_ids(const std::vector<const std::vector<std::uint32_t>*>& c) {
    const std::size_t n = c.size();
    std::vector<std::vector<std::uint64_t>> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k + 1 < c[i]->size(); ++k) b[i].push_back((static_cast<std::uint64_t>((*c[i])[k]) << 32) | (*c[i])[k + 1]);
        std::sort(b[i].begin(), b[i].end());
    }
    std::vector<double> u(n, 0.0);
    if (n < 2) return u;
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            std::size_t a = 0, d = 0, com = 0;
            while (a < b[i].size() && d < b[j].size()) {
                if (b[i][a] == b[j][d]) ++com, ++a, ++d;
                else if (b[i][a] < b[j][d]) ++a;
                else ++d;
            }
            const double t = static_cast<double>(b[i].size() + b[j].size());
            s += t > 0 ? 2.0 * static_cast<double>(com) / t : 0.0;
        }
        u[i] = s / static_cast<double>(n - 1);
    }
    return u;
}

// One fill of one step with its features.
struct Cand {
    Fill fill;
    std::size_t skel = 0;
    double f[kNF];
};

class Model {
 public:
    FactStore fs;
    SkelBank bank;
    st::Stats stats;
    lda::LDA topics;
    eg::EntityGrid grid;
    st::MetaMixer mx;
    double floor = -1e9;  // K5 fluency floor (mean log2 P per word); fills below it are not chosen unless all are
    bool gate_k5 = false; // --k5-plan-gate k5 (P2): K5 alone chooses
    bool trained = false;
    bool fill_dba = false;  // --k5-plan-fill dba: dynamic beam allocation (Filler::fill_dba)
    int beam = 4;           // --k5-plan-beam N
    std::string stats_path, skel_path, model_path;
    // §13.40 M3: thread-time inside Filler::fill and in the planner's fact steps (ns), fills, K5 distributions, prompts
    // the planner wrote from facts.
    mutable std::atomic<std::uint64_t> fill_ns{0}, fills{0}, dists{0}, steps_ns{0}, planned{0};

    // The model file (k5_planner train's output) without the store; then the store (load_store).
    bool load(const std::string& path, std::string& err) {
        model_path = path;
        std::ifstream in(path);
        std::string tag;
        int ver = 0;
        if (!in || !(in >> tag >> ver) || tag != "K5PLAN" || ver != 1) return err = "cannot read the planner model " + path, false;
        if (!(in >> tag >> stats_path) || tag != "stats") return err = path + ": no stats line", false;
        if (!(in >> tag >> skel_path) || tag != "skel") return err = path + ": no skel line", false;
        if (!(in >> tag >> floor) || tag != "floor") return err = path + ": no floor line", false;
        if (!(in >> tag >> trained) || tag != "trained") return err = path + ": no trained line", false;
        if (trained && !mx.read(in, kNF)) return err = path + ": bad mixer", false;
        return load_parts(err);
    }
    bool load_parts(std::string& err) {
        if (!stats.load(stats_path, err)) return false;
        if (std::ifstream(stats_path + ".lda")) topics.load(stats_path + ".lda");
        if (std::ifstream(stats_path + ".grid")) grid.load(stats_path + ".grid");
        return bank.load(skel_path, err);
    }
    bool save(const std::string& path) const {
        std::ofstream o(path);
        o.precision(17);
        o << "K5PLAN 1\nstats " << stats_path << "\nskel " << skel_path << "\nfloor " << floor << "\ntrained " << (trained ? 1 : 0) << "\n";
        if (trained) mx.write(o);
        return static_cast<bool>(o);
    }
    const st::TrainRuns& runs(const wkn::K5& m) const {
        std::call_once(once_, [&] { tr_.build(m.train_ids()); });
        return tr_;
    }

    // Step 2: the relation's skeletons scored after context c (prepared): class 5-gram mean, the length model, and
    // log2 of the skeleton's count. Returns the best k indices.
    std::vector<std::size_t> best_skeletons(const wkn::K5& m, const st::Context& c, const std::vector<Skel>& sk, const std::vector<std::string>& S,
                                            const std::vector<std::string>& O, std::size_t k, std::size_t exclude = ~std::size_t{0}) const {
        const std::uint32_t cs = stats.cls(m.id(S[0])), co = stats.cls(m.id(O[0]));
        std::vector<std::pair<double, std::size_t>> sc;
        for (std::size_t i = 0; i < sk.size(); ++i) {
            if (i == exclude) continue;
            std::vector<std::uint32_t> h;
            const std::size_t kk = std::min<std::size_t>(c.classes.size(), st::Stats::kCO - 1);
            h.assign(c.classes.end() - static_cast<std::ptrdiff_t>(kk), c.classes.end());
            double lp = 0;
            std::size_t ntok = 0;
            for (const auto code : sk[i].codes) {
                const std::uint32_t cl = code == kSlotS ? cs : code == kSlotO ? co : code;
                lp += stats.class_lp(h.data(), h.size(), cl);
                h.push_back(cl);
                if (h.size() > st::Stats::kCO - 1) h.erase(h.begin());
                ntok += code == kSlotS ? S.size() : code == kSlotO ? O.size() : 1;
            }
            st::Sent shape;
            shape.tok.resize(ntok), shape.ended = true;
            const double score = lp / static_cast<double>(sk[i].codes.size()) + stats.len_lp(c.plen, st::len_bucket(shape)) / 4.0 +
                                 0.25 * std::log2(1.0 + sk[i].count);
            sc.push_back({score, i});
        }
        std::stable_sort(sc.begin(), sc.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::size_t> out;
        for (std::size_t i = 0; i < sc.size() && out.size() < k; ++i) out.push_back(sc[i].second);
        return out;
    }

    // Steps 3-4's candidates: each skeleton filled after state s, with every expert's features (c prepared).
    void candidates(const wkn::K5& m, const st::Experts& ex, const Filler& fl, const st::Context& c, const wkn::K5::State& s,
                    const std::vector<double>& assoc, const std::vector<Skel>& sk, const std::vector<std::size_t>& pick,
                    const std::vector<std::string>& S, const std::vector<std::string>& O, std::vector<Cand>& out) const {
        out.clear();
        for (const auto i : pick) {
            Cand cd;
            const auto t0 = std::chrono::steady_clock::now();
            cd.fill = fl.fill(s, sk[i].codes, S, O);
            fill_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            ++fills;
            if (!cd.fill.ok) continue;
            cd.skel = i;
            st::Sent z;
            z.tok = cd.fill.tok;
            z.ended = true;
            ex.features(c, z, cd.fill.ids, cd.fill.mean(), st::mean_assoc(assoc, cd.fill.ids), cd.f);
            out.push_back(std::move(cd));
        }
        std::vector<const std::vector<std::uint32_t>*> ids;
        for (const auto& cd : out) ids.push_back(&cd.fill.ids);
        const auto u = mbr_ids(ids);
        for (std::size_t i = 0; i < out.size(); ++i) out[i].f[kMbr] = u[i];
        (void)m;
    }
    // The gate's choice among cands (index), after context c.
    std::size_t choose(const st::Context& c, const std::vector<Cand>& cands) const {
        std::size_t best = 0;
        if (gate_k5 || !trained) {
            for (std::size_t i = 1; i < cands.size(); ++i)
                if (cands[i].fill.mean() > cands[best].fill.mean()) best = i;
            return best;
        }
        st::Item it;
        it.n = static_cast<int>(cands.size());
        it.z.resize(cands.size() * kNF);
        for (std::size_t i = 0; i < cands.size(); ++i) row(cands[i].f, it.z.data() + i * kNF);
        st::selectors(c, it.sel);
        std::vector<double> P;
        mx.dist(it, P);
        bool any = false;
        for (const auto& cd : cands) any = any || cd.fill.mean() >= floor;
        double bp = -1;
        for (std::size_t i = 0; i < cands.size(); ++i) {
            if (any && cands[i].fill.mean() < floor) continue;
            if (P[i] > bp) bp = P[i], best = i;
        }
        return best;
    }
    void row(const double* f, float* z) const {
        for (int j = 0; j < kNF; ++j) {
            const double sd = mx.sd.size() > static_cast<std::size_t>(j) && mx.sd[static_cast<std::size_t>(j)] > 0 ? mx.sd[static_cast<std::size_t>(j)] : 1.0;
            z[j] = static_cast<float>(f[j] / sd);
        }
    }

 private:
    mutable std::once_flag once_;
    mutable st::TrainRuns tr_;
};

struct Plan {
    std::string path = "k5";  // main | cooc | k5
    std::uint32_t ent = ~0u;
    std::vector<std::uint32_t> facts;
    std::size_t orders = 0;   // orders scored
};

// Step 1. assoc: K5's prompt association (scores a fact's topic fit by its value's tokens).
inline Plan make_plan(const Model& md, const wkn::K5& m, std::string_view prompt, const std::vector<double>& assoc) {
    Plan pl;
    const auto& fs = md.fs;
    const auto cands = fs.candidates(prompt, &md.stats);  // PLAN_FACTS: CPS::candidates unless --k5-plan-find 2
    const auto fit = [&](std::uint32_t fi) {
        double t = 0;
        const auto& vt = fs.ent_tok(fs.fact(fi).val);
        for (const auto& w : vt) t += assoc.empty() ? 0.0 : assoc[m.id(w)];
        return vt.empty() ? 0.0 : t / static_cast<double>(vt.size());
    };
    // main entity
    for (const auto& c : cands) {
        const std::uint32_t e = fs.find_main(c);  // PLAN_FACTS: fs.find(c) unless --k5-plan-find 2
        if (e == ~0u) continue;
        std::map<std::uint32_t, std::vector<std::uint32_t>> by;
        for (const auto fi : fs.of_subj(e))
            if (md.bank.of(fs.rel(fs.fact(fi).rel))) by[fs.fact(fi).rel].push_back(fi);
        std::vector<std::uint32_t> pick;
        for (const auto& [r, v] : by) {
            std::set<std::uint32_t> vals;
            for (const auto fi : v) vals.insert(fs.fact(fi).val);
            if (vals.size() > 2) continue;
            std::uint32_t best = v[0];
            for (const auto fi : v) {
                const auto& f = fs.fact(fi);
                const auto& b = fs.fact(best);
                if ((f.wd == 1 && b.wd != 1) || (f.wd == b.wd && f.text && !b.text)) best = fi;
            }
            pick.push_back(best);
        }
        if (pick.empty()) continue;
        // keep the kMaxPlan facts that open a document most often (bank transitions from <s>)
        std::stable_sort(pick.begin(), pick.end(), [&](std::uint32_t a, std::uint32_t b) {
            return md.bank.trans_lp("<s>", fs.rel(fs.fact(a).rel)) > md.bank.trans_lp("<s>", fs.rel(fs.fact(b).rel));
        });
        if (pick.size() > static_cast<std::size_t>(kMaxPlan)) pick.resize(kMaxPlan);
        std::sort(pick.begin(), pick.end());
        double bs = -1e300;
        std::vector<std::uint32_t> best;
        do {
            double sc = 0;
            std::string prev = "<s>";
            for (std::size_t i = 0; i < pick.size(); ++i) {
                const std::string& r = fs.rel(fs.fact(pick[i]).rel);
                sc += md.bank.trans_lp(prev, r) + fit(pick[i]) * static_cast<double>(pick.size() - i) / static_cast<double>(pick.size());
                prev = r;
            }
            ++pl.orders;
            if (sc > bs) bs = sc, best = pick;
        } while (std::next_permutation(pick.begin(), pick.end()));
        pl.path = "main", pl.ent = e, pl.facts = best;
        return pl;
    }
    if (fs.options().fallback == "pattern") {  // PLAN_FACTS: the relations-pattern fallback (fallback.hpp)
        const auto fb = typed_fallback(fs, md.bank, md.stats, cands, prompt);
        if (!fb.facts.empty()) {
            pl.path = "pattern", pl.ent = fb.ent, pl.facts = fb.facts, pl.orders = 1;
            return pl;
        }
    }
    // co-occurring entities: facts whose value is a prompt entity, one per subject
    std::set<std::uint32_t> subj;
    for (const auto& c : cands) {
        const std::uint32_t e = fs.find(c);
        if (e == ~0u) continue;
        for (const auto fi : fs.of_val(e)) {
            const auto& f = fs.fact(fi);
            if (pl.facts.size() >= 3) break;
            if (!md.bank.of(fs.rel(f.rel)) || subj.count(f.subj)) continue;
            subj.insert(f.subj);
            pl.facts.push_back(fi);
            if (pl.ent == ~0u) pl.ent = e;
        }
    }
    if (!pl.facts.empty()) pl.path = "cooc", pl.orders = 1;
    return pl;
}

inline std::mutex& log_mutex() {
    static std::mutex mu;
    return mu;
}
inline std::string log_clean(std::string s) {
    for (char& ch : s)
        if (ch == '\t' || ch == '\n') ch = ' ';
    return s;
}

class Writer final : public wkn::StructureWriter {
 public:
    explicit Writer(std::shared_ptr<const Model> md) : md_(std::move(md)) {}
    const std::shared_ptr<const Model>& model() const { return md_; }  // discourse.hpp (PLAN_DISCOURSE.md) writes with it
    // §13.40 M3: the fill's cost, summed over the sampling threads (stderr; the GEN file is not touched).
    ~Writer() {
        const Model& md = *md_;
        if (md.planned == 0) return;
        std::fprintf(stderr, "k5-plan: fill %s (beam %d): %llu fills in %.2f thread-s (%.3f ms each), %llu K5 distributions; "
                     "planner fact steps %.2f thread-s over %llu prompts\n",
                     md.fill_dba ? "dba" : "beam", md.beam, static_cast<unsigned long long>(md.fills.load()), static_cast<double>(md.fill_ns.load()) / 1e9,
                     md.fills ? static_cast<double>(md.fill_ns.load()) / 1e6 / static_cast<double>(md.fills.load()) : 0.0,
                     static_cast<unsigned long long>(md.dists.load()), static_cast<double>(md.steps_ns.load()) / 1e9,
                     static_cast<unsigned long long>(md.planned.load()));
    }
    std::string generate(const wkn::K5& m, std::string_view prompt, std::uint64_t seed, const wkn::Options& o, wkn::GenStats* gst) const override {
        if (m.vocab().size() <= 2) return {};
        const Model& md = *md_;
        wkn::K5::State s = m.start(prompt, o);
        const std::vector<double> assoc = m.prompt_assoc(s.ctx);
        const Plan pl = make_plan(md, m, prompt, assoc);
        if (pl.path == "k5") {
            wkn::Options o2 = o;
            o2.structure = nullptr;
            std::string out = m.generate(prompt, seed, o2, gst);
            log(prompt, pl, {}, {}, out);
            return out;
        }
        wkn::GenStats g;
        const auto t_steps = std::chrono::steady_clock::now();
        const st::Experts ex(md.stats, m, md.runs(m), &md.topics, &md.grid);
        const Filler fl(m, o, md.beam, 3, md.fill_dba);
        st::Context c;
        c.sents = st::split(wkn::tokens(prompt));
        c.n_prompt = c.sents.size();
        c.prompt_chars = prompt.size();
        std::vector<std::size_t> end_at;            // each written fact sentence's end offset
        std::vector<std::uint32_t> written;         // its fact
        std::vector<std::string> sent_tok;
        std::vector<Cand> cands;
        for (const auto fi : pl.facts) {
            if (s.out.size() >= o.bytes || s.step >= o.max_words) break;
            const auto& f = md.fs.fact(fi);
            const auto* sk = md.bank.of(md.fs.rel(f.rel));
            if (!sk) continue;
            const auto& S = md.fs.ent_tok(f.subj);
            const auto& O = md.fs.ent_tok(f.val);
            ex.prepare(c);
            const auto pick = md.best_skeletons(m, c, *sk, S, O, kSkel);
            md.candidates(m, ex, fl, c, s, assoc, *sk, pick, S, O, cands);
            if (cands.empty()) continue;
            const std::size_t b = md.choose(c, cands);
            const auto& win = cands[b].fill;
            for (std::size_t i = 0; i < win.tok.size(); ++i) {  // only the winner becomes text
                const std::string pc = wkn::piece(win.tok[i], s.prev, s.has_prev);
                s.out += pc;
                s.rep.add(pc);
                wkn::K5::learn(s, win.ids[i], o);
                s.prev = win.tok[i], s.has_prev = true;
                ++s.step, ++g.steps;
            }
            g.gate_cands += cands.size(), ++g.gate_sents;
            st::Sent z;
            z.tok = win.tok, z.ended = true;
            c.sents.push_back(std::move(z));
            end_at.push_back(s.out.size());
            written.push_back(fi);
            std::string t;
            for (const auto& x : win.tok) t += (t.empty() ? "" : " ") + x;
            sent_tok.push_back(t);
        }
        md.steps_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_steps).count());
        md.dists += fl.dists();
        ++md.planned;
        // K5's own gate writes on (the same streams as its gate; sentence numbers after the plan's).
        const int K = std::max(o.gate_k, 1);
        std::vector<wkn::K5::State> cs(static_cast<std::size_t>(K));
        std::vector<std::uint32_t> words;
        std::vector<double> lp;
        for (std::uint64_t sent = 1000; s.step < o.max_words && s.out.size() < o.bytes; ++sent) {
            int best = 0;
            double bsc = -1e300;
            for (int k = 0; k < K; ++k) {
                wkn::GenStats gk;
                m.draft(s, seed, sent, k, o, cs[static_cast<std::size_t>(k)], words, lp, gk);
                const auto& out = cs[static_cast<std::size_t>(k)].out;
                std::size_t mk = 0;
                for (std::size_t i = s.out.size(); i < out.size(); ++i) mk += std::string_view("[]{}|=<>&*#").find(out[i]) != std::string_view::npos;
                const double nb = static_cast<double>(std::max<std::size_t>(out.size() - s.out.size(), 1));
                const double sc = o.gate_wp * st::mean_assoc(assoc, words) - o.gate_wm * static_cast<double>(mk) / nb;
                if (k == 0 || sc > bsc) bsc = sc, best = k;
                if (k == 0 || sc >= bsc) g.add(gk);
            }
            s = std::move(cs[static_cast<std::size_t>(best)]);
            ++g.gate_sents;
        }
        std::string out = std::move(s.out);
        wkn::K5::cut_bytes(out, o.bytes);
        std::vector<int> stated;
        for (const auto e : end_at) stated.push_back(e <= out.size() ? 1 : 0);
        log(prompt, pl, written, stated, out, sent_tok);
        if (gst) gst->add(g);
        return out;
    }

 private:
    void log(std::string_view prompt, const Plan& pl, const std::vector<std::uint32_t>& written, const std::vector<int>& stated,
             const std::string& out, const std::vector<std::string>& sents = {}) const {
        const Model& md = *md_;
        std::string line = "PLAN\t" + log_clean(std::string(prompt.substr(0, 60))) + "\t" + pl.path + "\t" +
                           (pl.ent == ~0u ? std::string("-") : log_clean(md.fs.ent(pl.ent))) + "\t" + std::to_string(pl.facts.size()) + "\t" +
                           std::to_string(written.size()) + "\t" + std::to_string(std::accumulate(stated.begin(), stated.end(), 0)) + "\t" +
                           std::to_string(pl.orders);
        for (std::size_t i = 0; i < written.size(); ++i) {
            const auto& f = md.fs.fact(written[i]);
            line += "\t" + log_clean(md.fs.ent(f.subj)) + "|" + md.fs.rel(f.rel) + "|" + log_clean(md.fs.ent(f.val)) + "|" + std::to_string(stated[i]) +
                    "|" + std::to_string(f.wd) + "|" + (f.text ? "text" : "wd") + "|" + log_clean(i < sents.size() ? sents[i] : "");
        }
        (void)out;
        std::lock_guard<std::mutex> lk(log_mutex());
        static std::FILE* fp = [] {
            const char* p = std::getenv("K5_PLAN_LOG");
            std::FILE* f = p ? std::fopen(p, "w") : nullptr;
            return f ? f : stderr;
        }();
        std::fprintf(fp, "%s\n", line.c_str());
        std::fflush(fp);
    }
    std::shared_ptr<const Model> md_;
};

// cyphalm_trie: --k5-plan STOREDIR --k5-plan-model FILE [--k5-plan-gate full|k5] [--k5-plan-fill beam|dba]
// [--k5-plan-beam N]; drops: the evidence ranges to drop. beam 0: the default (4).
inline bool apply_plan_flags(const std::string& store, const std::string& model, const std::string& gate,
                             const std::vector<std::pair<std::size_t, std::size_t>>& drops, wkn::Options& o, std::string& err,
                             const std::string& fill = "beam", int beam = 0) {
    auto md = std::make_shared<Model>();
    if (model.empty()) return err = "--k5-plan needs --k5-plan-model FILE (k5_planner train)", false;
    if (gate != "full" && gate != "k5") return err = "--k5-plan-gate takes full or k5", false;
    if (fill != "beam" && fill != "dba") return err = "--k5-plan-fill takes beam or dba", false;
    if (beam < 0 || beam > 256) return err = "--k5-plan-beam takes 1 to 256", false;
    if (!md->load(model, err)) return false;
    md->gate_k5 = gate == "k5";
    md->fill_dba = fill == "dba";
    if (beam > 0) md->beam = beam;
    md->fs.set_stats(&md->stats);  // PLAN_FACTS: read only by --k5-plan-filter's place rule
    if (!md->fs.load(store, drops, err)) return false;
    std::fprintf(stderr, "k5-plan: store %s: %zu facts (%zu dropped), %zu subjects, %zu relations; bank %zu skeletons over %zu relations; gate %s%s; floor %.3f\n",
                 store.c_str(), md->fs.size(), md->fs.dropped(), md->fs.subjects(), md->fs.rels(), md->bank.skeletons(), md->bank.relations(),
                 gate.c_str(), md->trained ? "" : " (untrained: K5 alone)", md->floor);
    if (md->fill_dba || beam > 0) std::fprintf(stderr, "k5-plan: fill %s, beam %d\n", fill.c_str(), md->beam);
    o.structure = std::make_shared<Writer>(md);
    o.structure_desc = "plan:" + store + "," + model + ",gate=" + gate;
    if (md->fill_dba || beam > 0) o.structure_desc += ",fill=" + fill + ",beam=" + std::to_string(md->beam);  // MERGE_FILL.md
    if (const auto& q = md->fs.options(); q.find != 1 || !q.filter.empty() || q.fallback != "cooc") {  // PLAN_FACTS.md
        std::fprintf(stderr, "k5-plan: find %d (%zu aliases), filter %s (%zu facts left out), fallback %s\n", q.find, md->fs.aliases(),
                     q.filter.empty() ? "off" : q.filter.c_str(), md->fs.filtered(), q.fallback.c_str());
        for (const auto& [r, n] : md->fs.filter_reasons()) std::fprintf(stderr, "k5-plan: filter %s %zu\n", r.c_str(), n);
        o.structure_desc += ",find=" + std::to_string(q.find) + (q.filter.empty() ? "" : ",filter=" + q.filter) + ",fallback=" + q.fallback;
    }
    return true;
}

}  // namespace cypha::cyphalm::planner
