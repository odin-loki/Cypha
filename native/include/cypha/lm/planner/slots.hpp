#pragma once

/// Typed slots and a little smoothing for the planner's fill (docs/reports/PLAN_FILL.md, GENERATION_PLAN §13.42 Q2).
///
/// K5 fills the class codes around a fact's names with its most likely words, so a sentence can put a name where its
/// type does not belong ("He was born in Gothenburg University"). With typed slots, the words K5 may put next to a name
/// must suit the name's type, and a skeleton whose class codes next to a slot do not suit the name is not used.
///
/// The types are learned from CyphaLM's own statistics, with no list of types or of words:
///   - the store's relation signatures: the positions (relation:S, relation:O) an entity fills;
///   - K5's word classes: the exchange class of the name's last token (its head);
///   - the name's shape: digits, capitals, lower-case words, the number of tokens.
/// A latent class model is fitted by EM over the store's entities: each type has a Bernoulli per position, a categorical
/// head class and a categorical shape. Each entity gets a posterior over the types.
///
/// The context model: in clean text, each mention of a store entity adds the entity's posterior to the counts of the
/// words and of the K5 classes at offsets -2, -1, +1, +2 from the name. For a name with posterior pi:
///   fit(pi, k, w) = log2 sum_z pi(z) (n_z(k, w) + a) / (E_z(k, w) + a),   E_z(k, w) = n(k, w) N_z(k) / N(k),
/// the ratio of the word's count next to the type to its count if the type did not matter, smoothed by a pseudo-count a
/// on both sides (so a word seen a few times is no evidence either way). With --k5-plan-typed-fit log the fit is the
/// posterior's mean of the log ratios instead, sum_z pi(z) log2 (n_z + a) / (E_z + a): a name that is most likely an
/// organisation is not let through "born in" because it might be a town.
///
/// The fill (writer.hpp, Filler) refuses a class word next to a slot when its fit for the slot's name is below -tau and
/// goes down the class's list; the skeleton choice drops a skeleton whose class codes next to a slot fit below -tau
/// (unless none would be left).
///
/// The smoothing places (FillOpts, all off by default): S1 the fill's next-word distribution (a share of K5's class
/// model or of K5 at order 3), S2 the gate's softmax temperature, S3 add-k on skeleton counts, S4 the plan's relation
/// transitions interpolated with the relations' marginal.
///
/// File (k5_planner types): "K5PLAN-SLOTS 1", "types K", "alpha A", "positions P NAME...", "prior ...", "theta" rows,
/// "head C" rows, "shape S" rows, "tot k N N_1 ... N_K" per offset, then "w k WORD n n_1 ... n_K" and
/// "c k CLASS n n_1 ... n_K" rows.
///
/// Author: Odin Loch

#include "cypha/lm/planner/facts.hpp"
#include "cypha/lm/planner/skeleton.hpp"
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm::planner {

// ---------------------------------------------------------------- options

// The fill's options (§13.42 Q2). Every field at its default leaves the planner as it was, byte for byte.
struct FillOpts {
    std::string typed;   // --k5-plan-typed FILE: typed slots (k5_planner types)
    double tau = 2.0;    // --k5-plan-typed-tau BITS: a word or skeleton fitting below -tau next to a slot is refused
    int typed_fit = 0;   // --k5-plan-typed-fit mix|log: the fit over the name's types, log2 of the mixed ratio (0) or the mean log ratio (1)
    double s1 = 0;       // --k5-plan-s1 L: blend L of a smoother distribution into the fill's next-word distribution
    int s1_mode = 0;     // --k5-plan-s1-mode class|low: K5's class model (0) or K5 at order 3 (1)
    double s2 = 1;       // --k5-plan-s2 T: the gate's softmax temperature (each ranker's, before the second layer)
    double s3 = 1;       // --k5-plan-s3 K: add-K on skeleton counts (1: today's log2(1 + count))
    double s4 = 0;       // --k5-plan-s4 L: the plan's relation transitions, L of the relations' marginal mixed in
    // --k5-plan-honest: 0 off; 1 on, the fill invents no number, month or name (GENERATION_PLAN §13.48 H1b);
    // 2 dates, no number or month only (HONEST_DATES HD1); 3 nations, HD1 and no nationality or country (HD2);
    // 4 life, HD2 and the subject's life dates from the store (HONEST_DATES2, H3; life_dates.hpp)
    int honest = 0;
    // --k5-plan-skel-add FILE[,FILE]: skeleton banks added to the model's (HONEST_DATES2: k5_dates mine or trim)
    std::string skel_add;
    bool any() const { return !typed.empty() || s1 > 0 || s2 != 1 || s3 != 1 || s4 > 0 || honest || !skel_add.empty(); }
    std::string desc() const {
        std::string d;
        const auto add = [&](const std::string& k, double v) {
            char b[64];
            std::snprintf(b, sizeof b, "%s=%g", k.c_str(), v);
            d += (d.empty() ? "" : ",") + std::string(b);
        };
        if (!typed.empty()) d += "typed=" + typed, add("tau", tau), d += typed_fit ? ",fit=log" : "";
        if (s1 > 0) add(s1_mode ? "s1low" : "s1class", s1);
        if (s2 != 1) add("s2", s2);
        if (s3 != 1) add("s3", s3);
        if (s4 > 0) add("s4", s4);
        if (honest)
            d += (d.empty() ? "" : ",") +
                 std::string(honest == 2 ? "honest=dates" : honest == 3 ? "honest=nations" : honest == 4 ? "honest=life" : "honest");
        if (!skel_add.empty()) d += (d.empty() ? "" : ",") + std::string("skel_add=") + skel_add;
        return d;
    }
};

// cyphalm_trie's --k5-plan-typed... and --k5-plan-s1... flags. Returns false if k is not one of them; err on a bad value.
inline bool fill_flag(const std::string& k, const std::string& v, FillOpts& f, std::string& err) {
    const auto num = [&](double lo, double hi, double& x) {
        char* e = nullptr;
        x = std::strtod(v.c_str(), &e);
        if (v.empty() || *e || !(x >= lo && x <= hi)) err = k + " takes a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]";
    };
    if (k == "--k5-plan-typed") f.typed = v;
    else if (k == "--k5-plan-typed-tau") num(0, 64, f.tau);
    else if (k == "--k5-plan-typed-fit") {
        if (v != "mix" && v != "log") err = k + " takes mix or log";
        f.typed_fit = v == "log";
    } else if (k == "--k5-plan-s1") num(0, 1, f.s1);
    else if (k == "--k5-plan-s1-mode") {
        if (v != "class" && v != "low") err = k + " takes class or low";
        f.s1_mode = v == "low";
    } else if (k == "--k5-plan-s2") num(0.01, 100, f.s2);
    else if (k == "--k5-plan-s3") num(0, 1e6, f.s3);
    else if (k == "--k5-plan-s4") num(0, 1, f.s4);
    else if (k == "--k5-plan-honest") {
        if (v != "on" && v != "off" && v != "dates" && v != "nations" && v != "life") err = k + " takes on, off, dates, nations or life";
        f.honest = v == "on" ? 1 : v == "dates" ? 2 : v == "nations" ? 3 : v == "life" ? 4 : 0;
    } else if (k == "--k5-plan-skel-add") {
        if (v.empty()) err = k + " takes FILE[,FILE]";
        f.skel_add = v;
    } else return false;
    if (k == "--k5-plan-s3" && err.empty() && f.s3 <= 0) err = k + " takes a number above 0";
    return true;
}

// ---------------------------------------------------------------- types

inline constexpr int kSlotOffs = 4;  // context offsets -2, -1, +1, +2
inline int slot_off_index(int off) { return off < 0 ? off + 2 : off + 1; }
inline int slot_off_of(int i) { return i < 2 ? i - 2 : i - 1; }
inline constexpr int kShapes = 12;

// The slots that class position j of a skeleton neighbours within two tokens, with nothing but class codes between:
// out[i] = (0 S or 1 O, offset index). Returns how many (at most 4).
inline int slot_neighbours(const std::vector<std::uint16_t>& codes, std::size_t j, std::pair<int, int>* out) {
    const auto slot = [](std::uint16_t c) { return c == kSlotS ? 0 : c == kSlotO ? 1 : -1; };
    if (slot(codes[j]) >= 0) return 0;
    int n = 0;
    for (std::size_t d = 1; d <= 2; ++d) {
        const int off = static_cast<int>(d);
        if (j + d < codes.size() && slot(codes[j + d]) >= 0 && (d == 1 || slot(codes[j + 1]) < 0)) out[n++] = {slot(codes[j + d]), slot_off_index(-off)};
        if (j >= d && slot(codes[j - d]) >= 0 && (d == 1 || slot(codes[j - 1]) < 0)) out[n++] = {slot(codes[j - d]), slot_off_index(off)};
    }
    return n;
}

// A name's shape, from its characters: 0 one digit token; 1 digit tokens among others; 2 every word token
// capitalised; 3 a lower-case word token; times 1, 2, 3+ tokens.
inline int name_shape(const std::vector<std::string>& t) {
    bool digit = false, lower = false;
    for (const auto& x : t) {
        if (x.empty()) continue;
        if (x[0] >= '0' && x[0] <= '9') digit = true;
        else if (x[0] >= 'a' && x[0] <= 'z') lower = true;
    }
    const int base = t.size() == 1 && digit ? 0 : digit ? 1 : lower ? 3 : 2;
    return base * 3 + (t.size() <= 1 ? 0 : t.size() == 2 ? 1 : 2);
}

class SlotTypes {
 public:
    // One entity: the positions it fills, the K5 class of its last token, its shape.
    struct Obs {
        std::vector<int> p;
        int h = 0, s = 0;
    };

    int K = 0;
    double alpha = 2;                   // the pseudo-count of fit
    int fit_mode = 0;                   // FillOpts::typed_fit: 0 log2 of the mixed ratio, 1 the mean log2 ratio
    std::vector<std::string> pos;       // position names, "relation:S" / "relation:O"
    std::vector<double> prior, theta;   // K; K x P
    std::size_t C = 1;                  // head classes
    std::vector<double> head, shape;    // K x C; K x kShapes

    int pos_id(const std::string& name, bool add) {
        const auto it = pos_id_.find(name);
        if (it != pos_id_.end()) return it->second;
        if (!add) return -1;
        pos_id_[name] = static_cast<int>(pos.size());
        pos.push_back(name);
        return static_cast<int>(pos.size()) - 1;
    }
    int pos_id(const std::string& name) const {
        const auto it = pos_id_.find(name);
        return it == pos_id_.end() ? -1 : it->second;
    }
    // Every entity of fs (index: its entity ID). add: learn new positions (fitting); else positions the model does not
    // know are left out.
    std::vector<Obs> observe(const FactStore& fs, const wkn::K5& m, bool add) {
        std::uint32_t ne = 0;
        for (std::size_t i = 0; i < fs.size(); ++i) ne = std::max({ne, fs.fact(static_cast<std::uint32_t>(i)).subj + 1, fs.fact(static_cast<std::uint32_t>(i)).val + 1});
        std::vector<Obs> ob(ne);
        const auto& cls = m.classes().classes();
        if (add) {
            C = 1;
            for (const auto c : cls) C = std::max<std::size_t>(C, c + 1);
        }
        for (std::size_t i = 0; i < fs.size(); ++i) {
            const auto& f = fs.fact(static_cast<std::uint32_t>(i));
            const std::string& r = fs.rel(f.rel);
            const int ps = add ? pos_id(r + ":S", true) : static_cast<const SlotTypes&>(*this).pos_id(r + ":S");
            const int po = add ? pos_id(r + ":O", true) : static_cast<const SlotTypes&>(*this).pos_id(r + ":O");
            if (ps >= 0) ob[f.subj].p.push_back(ps);
            if (po >= 0) ob[f.val].p.push_back(po);
        }
        for (std::uint32_t e = 0; e < ne; ++e) {
            auto& o = ob[e];
            std::sort(o.p.begin(), o.p.end());
            o.p.erase(std::unique(o.p.begin(), o.p.end()), o.p.end());
            const auto& t = fs.ent_tok(e);
            const std::uint32_t id = t.empty() ? wkn::kUnk : m.id(t.back());
            o.h = cls.empty() ? 0 : static_cast<int>(std::min<std::size_t>(cls[id], C - 1));
            o.s = name_shape(t);
        }
        return ob;
    }

    // EM for the latent class model: restarts from random responsibilities (seeded), the best log-likelihood kept. The
    // types are then ordered by their prior, largest first. Returns the log-likelihood (nats).
    double fit(const std::vector<Obs>& ob, int k, int iters, int restarts, std::uint64_t seed) {
        K = k;
        const std::size_t N = ob.size(), P = pos.size(), Kz = static_cast<std::size_t>(K);
        double best = -1e300;
        std::vector<double> bprior, btheta, bhead, bshape;
        std::vector<double> R(N * Kz);
        for (int rs = 0; rs < restarts; ++rs) {
            std::mt19937_64 rng(seed + 7919ull * static_cast<std::uint64_t>(rs));
            std::uniform_real_distribution<double> u(0.01, 1.0);
            for (std::size_t e = 0; e < N; ++e) {
                double z = 0;
                for (std::size_t c = 0; c < Kz; ++c) z += (R[e * Kz + c] = u(rng));
                for (std::size_t c = 0; c < Kz; ++c) R[e * Kz + c] /= z;
            }
            double ll = 0;
            for (int it = 0; it < iters; ++it) {
                mstep(ob, R);
                ll = estep(ob, R);
            }
            std::fprintf(stderr, "types: restart %d, log-likelihood %.1f\n", rs, ll);
            if (ll > best) best = ll, bprior = prior, btheta = theta, bhead = head, bshape = shape;
        }
        prior = bprior, theta = btheta, head = bhead, shape = bshape;
        std::vector<int> ord(Kz);
        for (int c = 0; c < K; ++c) ord[static_cast<std::size_t>(c)] = c;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return prior[static_cast<std::size_t>(a)] > prior[static_cast<std::size_t>(b)]; });
        const auto perm = [&](std::vector<double>& v, std::size_t w) {
            std::vector<double> o(v.size());
            for (std::size_t c = 0; c < Kz; ++c) std::copy_n(v.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(ord[c]) * w), w, o.begin() + static_cast<std::ptrdiff_t>(c * w));
            v.swap(o);
        };
        perm(prior, 1), perm(theta, P), perm(head, C), perm(shape, kShapes);
        (void)P;
        return best;
    }

    // An entity's posterior over the types into out[K].
    void posterior(const Obs& o, float* out) const { posterior(o, out, lbase()); }
    std::vector<float> posteriors(const std::vector<Obs>& ob) const {
        std::vector<float> p(ob.size() * static_cast<std::size_t>(K));
        const auto lb = lbase();
        for (std::size_t e = 0; e < ob.size(); ++e) posterior(ob[e], p.data() + e * static_cast<std::size_t>(K), lb);
        return p;
    }

    // The context model from clean text: documents (lines) starting in [lo, hi); every mention of a store entity (a
    // token run that starts with a capital or a digit and equals an entity's tokens, longest first) adds the entity's
    // posterior to the counts of the words and classes at offsets -2, -1, +1, +2. Returns the mentions.
    std::size_t count_text(std::string_view text, std::size_t lo, std::size_t hi, const FactStore& fs, const wkn::K5& m, const std::vector<float>& post) {
        const std::size_t Kz = static_cast<std::size_t>(K), ne = post.size() / std::max<std::size_t>(Kz, 1);
        const auto& cls = m.classes().classes();
        for (int k = 0; k < kSlotOffs; ++k) {
            tot_[k].assign(Kz + 1, 0.0), widx_[k].clear(), wrow_[k].clear();
            crow_[k].assign(C * (Kz + 1), 0.0);
        }
        std::size_t mentions = 0;
        std::vector<std::pair<std::uint32_t, std::size_t>> hits;
        const auto add = [&](int k, const std::string& tok, const float* pi) {
            const std::uint32_t id = m.id(tok);
            if (id == wkn::kUnk || id == wkn::kBos) return;
            auto& t = tot_[k];
            t[0] += 1;
            for (std::size_t c = 0; c < Kz; ++c) t[c + 1] += pi[c];
            const auto [it, fresh] = widx_[k].try_emplace(id, static_cast<std::uint32_t>(wrow_[k].size() / (Kz + 1)));
            if (fresh) wrow_[k].resize(wrow_[k].size() + Kz + 1, 0.0);
            double* r = wrow_[k].data() + static_cast<std::size_t>(it->second) * (Kz + 1);
            r[0] += 1;
            for (std::size_t c = 0; c < Kz; ++c) r[c + 1] += pi[c];
            const std::size_t cl = cls.empty() ? 0 : std::min<std::size_t>(cls[id], C - 1);
            double* q = crow_[k].data() + cl * (Kz + 1);
            q[0] += 1;
            for (std::size_t c = 0; c < Kz; ++c) q[c + 1] += pi[c];
        };
        for (std::size_t a = 0; a < text.size();) {
            std::size_t b = text.find('\n', a);
            if (b == std::string_view::npos) b = text.size();
            if (a >= lo && a < hi && b > a) {
                const auto t = wkn::tokens(text.substr(a, b - a));
                for (std::size_t i = 0; i < t.size();) {
                    const char c0 = t[i].empty() ? ' ' : t[i][0];
                    if (!((c0 >= 'A' && c0 <= 'Z') || (c0 >= '0' && c0 <= '9'))) {
                        ++i;
                        continue;
                    }
                    fs.match_at(t, i, hits);
                    std::uint32_t e = ~0u;
                    std::size_t n = 0;
                    for (const auto& [x, len] : hits) {
                        if (x >= ne) continue;
                        const auto& et = fs.ent_tok(x);
                        bool same = et.size() == len;
                        for (std::size_t j = 0; j < len && same; ++j) same = et[j] == t[i + j];
                        if (same) {
                            e = x, n = len;
                            break;
                        }
                    }
                    if (e == ~0u) {
                        ++i;
                        continue;
                    }
                    ++mentions;
                    const float* pi = post.data() + static_cast<std::size_t>(e) * Kz;
                    for (const int off : {-2, -1})
                        if (i >= static_cast<std::size_t>(-off)) add(slot_off_index(off), t[i - static_cast<std::size_t>(-off)], pi);
                    for (const int off : {1, 2})
                        if (i + n - 1 + static_cast<std::size_t>(off) < t.size()) add(slot_off_index(off), t[i + n - 1 + static_cast<std::size_t>(off)], pi);
                    i += n;
                }
            }
            a = b + 1;
            if (a >= hi) break;
        }
        return mentions;
    }

    // fit of word w (K5 ID) at offset index k for a name with posterior pi; 0 for a word with fewer than the file's
    // minimum count there.
    double word_fit(const float* pi, int k, std::uint32_t w) const {
        const auto it = widx_[k].find(w);
        if (it == widx_[k].end()) return 0;
        return ratio(pi, k, wrow_[k].data() + static_cast<std::size_t>(it->second) * static_cast<std::size_t>(K + 1));
    }
    double class_fit(const float* pi, int k, std::uint32_t c) const {
        if (c >= C) return 0;
        return ratio(pi, k, crow_[k].data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(K + 1));
    }
    std::size_t words(int k) const { return widx_[k].size(); }

    // min: rows of words with fewer mentions than this are not written.
    bool save(const std::string& path, const wkn::K5& m, double min) const {
        std::ofstream o(path);
        o.precision(9);
        const std::size_t Kz = static_cast<std::size_t>(K), P = pos.size();
        o << "K5PLAN-SLOTS 1\ntypes " << K << "\nalpha " << alpha << "\npositions " << P;
        for (const auto& p : pos) o << " " << p;
        o << "\nprior";
        for (const double x : prior) o << " " << x;
        o << "\ntheta\n";
        for (std::size_t c = 0; c < Kz; ++c) {
            for (std::size_t p = 0; p < P; ++p) o << (p ? " " : "") << theta[c * P + p];
            o << "\n";
        }
        o << "head " << C << "\n";
        for (std::size_t c = 0; c < Kz; ++c) {
            for (std::size_t h = 0; h < C; ++h) o << (h ? " " : "") << head[c * C + h];
            o << "\n";
        }
        o << "shape " << kShapes << "\n";
        for (std::size_t c = 0; c < Kz; ++c) {
            for (int s = 0; s < kShapes; ++s) o << (s ? " " : "") << shape[c * kShapes + static_cast<std::size_t>(s)];
            o << "\n";
        }
        o.precision(7);  // counts (sums of posteriors): 7 significant digits
        for (int k = 0; k < kSlotOffs; ++k) {
            o << "tot " << k;
            for (const double x : tot_[k]) o << " " << x;
            o << "\n";
        }
        for (int k = 0; k < kSlotOffs; ++k) {
            std::vector<std::pair<std::uint32_t, std::uint32_t>> v(widx_[k].begin(), widx_[k].end());
            std::sort(v.begin(), v.end());
            for (const auto& [w, r] : v) {
                const double* x = wrow_[k].data() + static_cast<std::size_t>(r) * (Kz + 1);
                if (x[0] < min) continue;
                o << "w " << k << " " << m.vocab()[w];
                for (std::size_t j = 0; j <= Kz; ++j) o << " " << x[j];
                o << "\n";
            }
            for (std::size_t c = 0; c < C; ++c) {
                const double* x = crow_[k].data() + c * (Kz + 1);
                if (x[0] <= 0) continue;
                o << "c " << k << " " << c;
                for (std::size_t j = 0; j <= Kz; ++j) o << " " << x[j];
                o << "\n";
            }
        }
        return static_cast<bool>(o);
    }
    // The file without K5: words are bound to K5 IDs by bind.
    bool load(const std::string& path, std::string& err) {
        std::ifstream in(path);
        std::string line, tag;
        if (std::getline(in, line) && !line.empty() && line.back() == '\r') line.pop_back();  // a Windows checkout's CRLF
        if (!in || line != "K5PLAN-SLOTS 1") return err = "cannot read the slot types " + path, false;
        std::size_t P = 0;
        if (!(in >> tag >> K) || tag != "types" || K <= 0 || K > 256) return err = path + ": no types line", false;
        if (!(in >> tag >> alpha) || tag != "alpha") return err = path + ": no alpha line", false;
        if (!(in >> tag >> P) || tag != "positions") return err = path + ": no positions line", false;
        pos.clear(), pos_id_.clear();
        for (std::size_t p = 0; p < P; ++p) {
            std::string s;
            in >> s;
            pos_id(s, true);
        }
        const std::size_t Kz = static_cast<std::size_t>(K);
        const auto rd = [&](std::vector<double>& v, std::size_t n) {
            v.resize(n);
            for (auto& x : v) in >> x;
        };
        if (!(in >> tag) || tag != "prior") return err = path + ": no prior", false;
        rd(prior, Kz);
        if (!(in >> tag) || tag != "theta") return err = path + ": no theta", false;
        rd(theta, Kz * P);
        if (!(in >> tag >> C) || tag != "head" || C == 0) return err = path + ": no head", false;
        rd(head, Kz * C);
        int S = 0;
        if (!(in >> tag >> S) || tag != "shape" || S != kShapes) return err = path + ": no shape", false;
        rd(shape, Kz * kShapes);
        for (int k = 0; k < kSlotOffs; ++k) {
            int kk = -1;
            if (!(in >> tag >> kk) || tag != "tot" || kk != k) return err = path + ": no tot line", false;
            rd(tot_[k], Kz + 1);
            crow_[k].assign(C * (Kz + 1), 0.0);
            widx_[k].clear(), wrow_[k].clear();
            wtok_[k].clear(), wtrow_[k].clear();
        }
        std::getline(in, line);
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            int k = -1;
            if (!(ls >> tag >> k) || k < 0 || k >= kSlotOffs) continue;
            if (tag == "w") {
                std::string w;
                ls >> w;
                wtok_[k].push_back(w);
                for (std::size_t j = 0; j <= Kz; ++j) {
                    double x = 0;
                    ls >> x;
                    wtrow_[k].push_back(x);
                }
            } else if (tag == "c") {
                std::size_t c = 0;
                ls >> c;
                if (c >= C) continue;
                for (std::size_t j = 0; j <= Kz; ++j) ls >> crow_[k][c * (Kz + 1) + j];
            }
        }
        if (!in.eof()) return err = path + ": read error", false;
        return true;
    }
    // The loaded words as K5 IDs (words K5 does not know are dropped).
    void bind(const wkn::K5& m) {
        const std::size_t Kz = static_cast<std::size_t>(K);
        for (int k = 0; k < kSlotOffs; ++k) {
            widx_[k].clear(), wrow_[k].clear();
            for (std::size_t i = 0; i < wtok_[k].size(); ++i) {
                const std::uint32_t id = m.id(wtok_[k][i]);
                if (id == wkn::kUnk) continue;
                widx_[k][id] = static_cast<std::uint32_t>(wrow_[k].size() / (Kz + 1));
                wrow_[k].insert(wrow_[k].end(), wtrow_[k].begin() + static_cast<std::ptrdiff_t>(i * (Kz + 1)),
                                wtrow_[k].begin() + static_cast<std::ptrdiff_t>((i + 1) * (Kz + 1)));
            }
        }
    }

    // The n words that suit type c best (and worst, worst = true) at offset index k, among words with at least min
    // mentions there (for k5_planner types' report).
    std::vector<std::pair<double, std::uint32_t>> top_words(int c, int k, std::size_t n, double min, bool worst) const {
        std::vector<float> pi(static_cast<std::size_t>(K), 0.0f);
        pi[static_cast<std::size_t>(c)] = 1;
        std::vector<std::pair<double, std::uint32_t>> v;
        for (const auto& [w, r] : widx_[k])
            if (wrow_[k][static_cast<std::size_t>(r) * static_cast<std::size_t>(K + 1)] >= min) v.push_back({word_fit(pi.data(), k, w), w});
        std::sort(v.begin(), v.end(), [&](const auto& a, const auto& b) { return worst ? a.first < b.first || (a.first == b.first && a.second < b.second)
                                                                                          : a.first > b.first || (a.first == b.first && a.second < b.second); });
        if (v.size() > n) v.resize(n);
        return v;
    }
    // Type c's description: its largest positions and shapes (for k5_planner types' report).
    std::string describe(int c) const {
        const std::size_t P = pos.size();
        std::vector<std::pair<double, std::size_t>> v;
        for (std::size_t p = 0; p < P; ++p) v.push_back({theta[static_cast<std::size_t>(c) * P + p], p});
        std::sort(v.rbegin(), v.rend());
        std::string d;
        char b[96];
        for (std::size_t i = 0; i < v.size() && i < 5 && v[i].first >= 0.02; ++i) {
            std::snprintf(b, sizeof b, "%s%s %.2f", d.empty() ? "" : ", ", pos[v[i].second].c_str(), v[i].first);
            d += b;
        }
        int bs = 0;
        for (int s = 1; s < kShapes; ++s)
            if (shape[static_cast<std::size_t>(c) * kShapes + static_cast<std::size_t>(s)] > shape[static_cast<std::size_t>(c) * kShapes + static_cast<std::size_t>(bs)]) bs = s;
        std::snprintf(b, sizeof b, "; shape %d (%.2f)", bs, shape[static_cast<std::size_t>(c) * kShapes + static_cast<std::size_t>(bs)]);
        return d + b;
    }

 private:
    // sum_p log(1 - theta(c, p)) per type: an entity's log-likelihood starts from it (no position filled).
    std::vector<double> lbase() const {
        std::vector<double> lb(static_cast<std::size_t>(K), 0.0);
        const std::size_t P = pos.size();
        for (std::size_t z = 0; z < lb.size(); ++z)
            for (std::size_t p = 0; p < P; ++p) lb[z] += std::log1p(-theta[z * P + p]);
        return lb;
    }
    double loglik(const Obs& o, int c, const std::vector<double>& lb) const {
        const std::size_t P = pos.size(), cz = static_cast<std::size_t>(c);
        double l = std::log(prior[cz]) + lb[cz];
        for (const int p : o.p) {
            const double t = theta[cz * P + static_cast<std::size_t>(p)];
            l += std::log(t) - std::log1p(-t);
        }
        return l + std::log(head[cz * C + static_cast<std::size_t>(o.h)]) + std::log(shape[cz * kShapes + static_cast<std::size_t>(o.s)]);
    }
    void posterior(const Obs& o, float* out, const std::vector<double>& lb) const {
        std::vector<double> l(static_cast<std::size_t>(K));
        double mx = -1e300;
        for (int c = 0; c < K; ++c) mx = std::max(mx, l[static_cast<std::size_t>(c)] = loglik(o, c, lb));
        double z = 0;
        for (auto& x : l) z += (x = std::exp(x - mx));
        for (int c = 0; c < K; ++c) out[c] = static_cast<float>(l[static_cast<std::size_t>(c)] / z);
    }
    void mstep(const std::vector<Obs>& ob, const std::vector<double>& R) {
        const std::size_t N = ob.size(), P = pos.size(), Kz = static_cast<std::size_t>(K);
        std::vector<double> nz(Kz, 0.0), cp(Kz * P, 0.0), ch(Kz * C, 0.0), cs(Kz * kShapes, 0.0);
        for (std::size_t e = 0; e < N; ++e)
            for (std::size_t c = 0; c < Kz; ++c) {
                const double w = R[e * Kz + c];
                nz[c] += w;
                for (const int p : ob[e].p) cp[c * P + static_cast<std::size_t>(p)] += w;
                ch[c * C + static_cast<std::size_t>(ob[e].h)] += w;
                cs[c * kShapes + static_cast<std::size_t>(ob[e].s)] += w;
            }
        prior.resize(Kz), theta.resize(Kz * P), head.resize(Kz * C), shape.resize(Kz * kShapes);
        constexpr double a = 0.1, b = 0.1;
        for (std::size_t c = 0; c < Kz; ++c) {
            prior[c] = (nz[c] + 1.0) / (static_cast<double>(N) + static_cast<double>(Kz));
            for (std::size_t p = 0; p < P; ++p) theta[c * P + p] = (cp[c * P + p] + a) / (nz[c] + 2 * a);
            for (std::size_t h = 0; h < C; ++h) head[c * C + h] = (ch[c * C + h] + b) / (nz[c] + b * static_cast<double>(C));
            for (int s = 0; s < kShapes; ++s) shape[c * kShapes + static_cast<std::size_t>(s)] = (cs[c * kShapes + static_cast<std::size_t>(s)] + b) / (nz[c] + b * kShapes);
        }
    }
    double estep(const std::vector<Obs>& ob, std::vector<double>& R) const {
        const std::size_t Kz = static_cast<std::size_t>(K);
        double ll = 0;
        std::vector<double> l(Kz);
        const auto lb = lbase();
        for (std::size_t e = 0; e < ob.size(); ++e) {
            double mx = -1e300;
            for (std::size_t c = 0; c < Kz; ++c) mx = std::max(mx, l[c] = loglik(ob[e], static_cast<int>(c), lb));
            double z = 0;
            for (auto& x : l) z += (x = std::exp(x - mx));
            for (std::size_t c = 0; c < Kz; ++c) R[e * Kz + c] = l[c] / z;
            ll += mx + std::log(z);
        }
        return ll;
    }
    // sum_z pi(z) (n_z + a) / (E_z + a), E_z = n N_z / N; log2. fit_mode 1: sum_z pi(z) log2 (n_z + a) / (E_z + a).
    double ratio(const float* pi, int k, const double* r) const {
        const double N = tot_[k][0];
        if (N <= 0 || r[0] <= 0) return 0;
        double s = 0;
        for (int c = 0; c < K; ++c) {
            if (pi[c] <= 0) continue;
            const double E = r[0] * tot_[k][static_cast<std::size_t>(c) + 1] / N;
            const double q = (r[static_cast<std::size_t>(c) + 1] + alpha) / (E + alpha);
            s += fit_mode ? pi[c] * std::log2(q) : pi[c] * q;
        }
        if (fit_mode) return s;
        return s > 0 ? std::log2(s) : 0;
    }

    std::unordered_map<std::string, int> pos_id_;
    std::vector<double> tot_[kSlotOffs];  // N(k), N_1(k), ..., N_K(k)
    std::unordered_map<std::uint32_t, std::uint32_t> widx_[kSlotOffs];
    std::vector<double> wrow_[kSlotOffs], crow_[kSlotOffs];  // rows: n, n_1, ..., n_K
    std::vector<std::string> wtok_[kSlotOffs];               // load: the words before bind
    std::vector<double> wtrow_[kSlotOffs];
};

}  // namespace cypha::cyphalm::planner
