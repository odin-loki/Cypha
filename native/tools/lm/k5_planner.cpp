// k5_planner: the planner's offline steps (docs/reports/K5_PLANNER.md, GENERATION_PLAN §13.37-13.38).
//
//   k5_planner mine  --k5 TEXT --store DIR --text CLEAN --lo A --hi B --out SKEL [--drop LO:HI]
//       skeletons by distant supervision: sentences of CLEAN's documents starting in [A, B) bytes that hold a store
//       fact's subject and value; and the relation transitions inside documents.
//   k5_planner train --k5 TEXT --store DIR --stats STATS --skel SKEL --text CLEAN --lo A --hi B --out MODEL
//                    [--items N] [--threads T] [--drop LO:HI]
//       the gate's meta-mixer: for each real sentence in [A, B) that states a store fact, the fill of its own skeleton
//       and fact is the target among the fills of the relation's other best skeletons (same fact, same context). The
//       first 80% of items train, the last 20% test. The K5 fluency floor is the 5th percentile of the targets' K5 score.
//       [--typed SLOTS] [--typed-tau B] [--typed-fit mix|log] [--s1 L] [--s1-mode class|low]: the fills typed and
//       smoothed as sample's --k5-plan-typed... and --k5-plan-s1... (docs/reports/PLAN_FILL.md).
//   k5_planner types --k5 TEXT --store DIR --text CLEAN --lo A --hi B --out SLOTS [--drop LO:HI] [--types K]
//                    [--iters N] [--restarts R] [--seed S] [--min N] [--table PREFIX]
//       typed slots (slots.hpp, GENERATION_PLAN §13.42 Q2): the entity types by EM over the store's relation
//       signatures, K5 head classes and name shapes; then the context model from the mentions of store entities in
//       CLEAN's documents starting in [A, B). --table: PREFIX.entities.tsv and PREFIX.positions.tsv (the checker's types).
//   k5_planner selftest
//
// Author: Odin Loch

#include "cypha/lm/planner/writer.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace pl = cypha::cyphalm::planner;
namespace st = cypha::cyphalm::structure;
namespace wkn = cypha::cyphalm::wkn;

namespace {

struct Args {
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& d = "") const {
        const auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
};

std::vector<std::pair<std::size_t, std::size_t>> drops_of(const std::string& s) {
    std::vector<std::pair<std::size_t, std::size_t>> d;
    if (s.empty()) return d;
    const auto c = s.find(':');
    d.push_back({std::strtoull(s.substr(0, c).c_str(), nullptr, 10), std::strtoull(s.substr(c + 1).c_str(), nullptr, 10)});
    return d;
}

bool load_k5(const std::string& path, wkn::K5& m) {
    std::string err;
    const wkn::Options bo;
    if (!m.load(path, 12, err, &bo)) {
        std::cerr << err << "\n";
        return false;
    }
    std::fprintf(stderr, "k5: vocab %zu, %zu tokens\n", m.vocab().size(), m.tokens());
    return true;
}

std::string read_all(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int cmd_mine(const Args& a) {
    wkn::K5 m;
    if (!load_k5(a.get("--k5"), m)) return 1;
    pl::FactStore fs;
    std::string err;
    if (!fs.load(a.get("--store"), drops_of(a.get("--drop")), err)) return std::cerr << err << "\n", 1;
    std::fprintf(stderr, "store: %zu facts, %zu subjects\n", fs.size(), fs.subjects());
    const std::string text = read_all(a.get("--text"));
    const std::size_t lo = std::strtoull(a.get("--lo", "0").c_str(), nullptr, 10), hi = std::strtoull(a.get("--hi", "0").c_str(), nullptr, 10);
    pl::SkelBank bank;
    std::size_t docs = 0, sents = 0, matched = 0, shaped = 0;
    std::map<std::string, std::size_t> per_rel;
    const auto t0 = std::chrono::steady_clock::now();
    pl::for_docs(text, lo, hi ? hi : text.size(), [&](const std::vector<st::Sent>& ss, std::size_t) {
        ++docs;
        std::string prev = "<s>";
        for (const auto& s : ss) {
            ++sents;
            pl::Match mt;
            if (!pl::match_sentence(fs, s.tok, mt)) continue;
            ++matched;
            const std::string& r = fs.rel(fs.fact(mt.fact).rel);
            bank.add_trans(prev, r);
            prev = r;
            std::vector<std::uint16_t> codes;
            if (!pl::shape_of(m, s.tok, mt, codes)) continue;
            ++shaped, ++per_rel[r];
            bank.add(r, codes);
        }
        if (docs % 5000 == 0)
            std::fprintf(stderr, "mine: %zu docs, %zu sentences, %zu matched, %zu skeletons (%.0f s)\n", docs, sents, matched, shaped,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    });
    bank.finish();
    if (!bank.save(a.get("--out"))) return std::cerr << "cannot write " << a.get("--out") << "\n", 1;
    std::printf("mine: %zu docs, %zu sentences, %zu matched a fact, %zu gave a skeleton; %zu relations, %zu skeletons kept\n", docs, sents,
                matched, shaped, bank.relations(), bank.skeletons());
    for (const auto& [r, n] : per_rel) std::printf("  %s %zu\n", r.c_str(), n);
    return 0;
}

// PLAN_FILL.md (§13.42 Q2): the slot types.
int cmd_types(const Args& a) {
    wkn::K5 m;
    if (!load_k5(a.get("--k5"), m)) return 1;
    pl::FactStore fs;
    std::string err;
    if (!fs.load(a.get("--store"), drops_of(a.get("--drop")), err)) return std::cerr << err << "\n", 1;
    std::fprintf(stderr, "store: %zu facts, %zu subjects\n", fs.size(), fs.subjects());
    const int K = std::atoi(a.get("--types", "12").c_str());
    const int iters = std::atoi(a.get("--iters", "100").c_str()), restarts = std::atoi(a.get("--restarts", "4").c_str());
    const std::uint64_t seed = std::strtoull(a.get("--seed", "20261105").c_str(), nullptr, 10);
    const double minc = std::atof(a.get("--min", "10").c_str());
    if (K < 2 || K > 64 || iters < 1 || restarts < 1) return std::cerr << "types: bad --types, --iters or --restarts\n", 2;
    pl::SlotTypes ty;
    const auto t0 = std::chrono::steady_clock::now();
    const auto ob = ty.observe(fs, m, true);
    const double ll = ty.fit(ob, K, iters, restarts, seed);
    const auto post = ty.posteriors(ob);
    const double npar = static_cast<double>(K) * static_cast<double>(ty.pos.size() + ty.C - 1 + pl::kShapes - 1) + K - 1;
    std::printf("types: %zu entities, %zu positions, %zu head classes, %d types; log-likelihood %.1f, BIC %.1f (%.1f s)\n", ob.size(), ty.pos.size(), ty.C, K,
                ll, -2 * ll + npar * std::log(static_cast<double>(ob.size())), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const std::string text = read_all(a.get("--text"));
    const std::size_t lo = std::strtoull(a.get("--lo", "0").c_str(), nullptr, 10), hi = std::strtoull(a.get("--hi", "0").c_str(), nullptr, 10);
    const std::size_t men = ty.count_text(text, lo, hi ? hi : text.size(), fs, m, post);
    std::printf("context: %zu mentions of store entities in [%zu, %zu); words kept per offset (at least %.0f mentions):", men, lo, hi ? hi : text.size(), minc);
    if (!ty.save(a.get("--out"), m, minc)) return std::cerr << "cannot write " << a.get("--out") << "\n", 1;
    pl::SlotTypes back;  // what sample will read
    if (!back.load(a.get("--out"), err)) return std::cerr << err << "\n", 1;
    back.bind(m);
    for (int k = 0; k < pl::kSlotOffs; ++k) std::printf(" %+d %zu", pl::slot_off_of(k), back.words(k));
    std::printf("\n");
    // Each type: its prior, positions and shape, its largest members, the words that suit it least and best at -2, -1, +1.
    std::vector<std::size_t> nf(ob.size(), 0);
    for (std::size_t i = 0; i < fs.size(); ++i) ++nf[fs.fact(static_cast<std::uint32_t>(i)).subj], ++nf[fs.fact(static_cast<std::uint32_t>(i)).val];
    std::vector<int> arg(ob.size(), 0);
    std::vector<std::size_t> size(static_cast<std::size_t>(K), 0);
    for (std::size_t e = 0; e < ob.size(); ++e) {
        const float* p = post.data() + e * static_cast<std::size_t>(K);
        arg[e] = static_cast<int>(std::max_element(p, p + K) - p);
        ++size[static_cast<std::size_t>(arg[e])];
    }
    for (int c = 0; c < K; ++c) {
        std::printf("type %d: prior %.3f, %zu entities; %s\n  members:", c, ty.prior[static_cast<std::size_t>(c)], size[static_cast<std::size_t>(c)], ty.describe(c).c_str());
        std::vector<std::pair<std::size_t, std::size_t>> mem;
        for (std::size_t e = 0; e < ob.size(); ++e)
            if (arg[e] == c) mem.push_back({nf[e], e});
        std::sort(mem.rbegin(), mem.rend());
        for (std::size_t i = 0; i < mem.size() && i < 8; ++i) std::printf(" %s;", fs.ent(static_cast<std::uint32_t>(mem[i].second)).c_str());
        std::printf("\n");
        for (const int off : {-2, -1, 1}) {
            const int k = pl::slot_off_index(off);
            std::printf("  %+d best:", off);
            for (const auto& [f, w] : back.top_words(c, k, 8, 200, false)) std::printf(" %s %.1f", m.vocab()[w].c_str(), f);
            std::printf(" | worst:");
            for (const auto& [f, w] : back.top_words(c, k, 6, 200, true)) std::printf(" %s %.1f", m.vocab()[w].c_str(), f);
            std::printf("\n");
        }
    }
    const std::string tab = a.get("--table");
    if (!tab.empty()) {
        std::ofstream oe(tab + ".entities.tsv"), op(tab + ".positions.tsv");
        oe << "entity\ttype\tposterior\tfacts\tpositions\n";
        for (std::size_t e = 0; e < ob.size(); ++e) {
            oe << fs.ent(static_cast<std::uint32_t>(e)) << "\t" << arg[e] << "\t" << post[e * static_cast<std::size_t>(K) + static_cast<std::size_t>(arg[e])] << "\t" << nf[e] << "\t";
            for (std::size_t i = 0; i < ob[e].p.size(); ++i) oe << (i ? "," : "") << ty.pos[static_cast<std::size_t>(ob[e].p[i])];
            oe << "\n";
        }
        std::vector<std::vector<std::size_t>> pc(ty.pos.size(), std::vector<std::size_t>(static_cast<std::size_t>(K), 0));
        for (std::size_t e = 0; e < ob.size(); ++e)
            for (const int p : ob[e].p) ++pc[static_cast<std::size_t>(p)][static_cast<std::size_t>(arg[e])];
        op << "position\tentities";
        for (int c = 0; c < K; ++c) op << "\tt" << c;
        op << "\n";
        for (std::size_t p = 0; p < ty.pos.size(); ++p) {
            std::size_t n = 0;
            for (const auto x : pc[p]) n += x;
            op << ty.pos[p] << "\t" << n;
            for (const auto x : pc[p]) op << "\t" << x;
            op << "\n";
        }
        if (!oe || !op) return std::cerr << "cannot write " << tab << ".*.tsv\n", 1;
    }
    return 0;
}

struct TItem {
    std::vector<st::Sent> ctx;   // context sentences (first: the prompt)
    std::uint32_t fact = 0;
    std::vector<std::uint16_t> own;
    // computed
    std::vector<std::vector<double>> f;  // candidates' features (row 0: the target)
    std::vector<double> k5;
    bool ok = false;
};

int cmd_train(const Args& a) {
    wkn::K5 m;
    if (!load_k5(a.get("--k5"), m)) return 1;
    auto md = std::make_shared<pl::Model>();
    md->stats_path = a.get("--stats");
    md->skel_path = a.get("--skel");
    std::string err;
    if (!md->load_parts(err) || !md->fs.load(a.get("--store"), drops_of(a.get("--drop")), err)) return std::cerr << err << "\n", 1;
    {
        std::size_t diff = 0;
        const auto& cls = m.classes().classes();
        for (std::uint32_t w = 0; w < cls.size(); ++w) diff += md->stats.cls(w) != cls[w];
        std::fprintf(stderr, "train: K5 classes and the statistics' classes differ on %zu of %zu words\n", diff, cls.size());
    }
    const std::string text = read_all(a.get("--text"));
    const std::size_t lo = std::strtoull(a.get("--lo", "0").c_str(), nullptr, 10), hi = std::strtoull(a.get("--hi", "0").c_str(), nullptr, 10);
    const std::size_t N = std::strtoull(a.get("--items", "3000").c_str(), nullptr, 10);
    std::vector<TItem> items;
    std::size_t seen = 0;
    pl::for_docs(text, lo, hi ? hi : text.size(), [&](const std::vector<st::Sent>& ss, std::size_t) {
        for (std::size_t j = 1; j < ss.size(); ++j) {
            pl::Match mt;
            if (!pl::match_sentence(md->fs, ss[j].tok, mt)) continue;
            const auto* sk = md->bank.of(md->fs.rel(md->fs.fact(mt.fact).rel));
            if (!sk || sk->size() < 2) continue;
            TItem it;
            if (!pl::shape_of(m, ss[j].tok, mt, it.own)) continue;
            ++seen;
            it.fact = mt.fact;
            it.ctx.assign(ss.begin() + static_cast<std::ptrdiff_t>(j >= 3 ? j - 3 : 0), ss.begin() + static_cast<std::ptrdiff_t>(j));
            items.push_back(std::move(it));
        }
    });
    // An even spread of N items over the range.
    if (items.size() > N) {
        std::vector<TItem> keep;
        for (std::size_t i = 0; i < N; ++i) keep.push_back(std::move(items[i * items.size() / N]));
        items.swap(keep);
    }
    std::fprintf(stderr, "train: %zu real fact sentences with a skeleton, %zu used\n", seen, items.size());
    const wkn::Options o;
    const st::Experts ex(md->stats, m, md->runs(m), &md->topics, &md->grid);
    // PLAN_FILL.md: the fills typed and smoothed as sample's (--typed, --s1, --s1-mode, --typed-tau).
    for (const char* k : {"--typed", "--typed-tau", "--typed-fit", "--s1", "--s1-mode"})
        if (!a.get(k).empty() && (!pl::fill_flag(std::string("--k5-plan-") + (k + 2), a.get(k), md->fo, err) || !err.empty()))
            return std::cerr << (err.empty() ? std::string("bad ") + k : err) << "\n", 2;
    if (!md->fo.typed.empty()) {
        md->types = std::make_shared<pl::SlotTypes>();
        if (!md->types->load(md->fo.typed, err)) return std::cerr << err << "\n", 1;
        md->types->fit_mode = md->fo.typed_fit;
        std::fprintf(stderr, "train: fills %s\n", md->fo.desc().c_str());
    }
    const pl::Filler fl(m, o, 4, 3, false, &md->fo, md->bind_types(m));
    std::atomic<std::size_t> next{0}, done{0};
    const auto t0 = std::chrono::steady_clock::now();
    const auto work = [&] {
        for (std::size_t i; (i = next++) < items.size();) {
            TItem& it = items[i];
            st::Context c;
            c.sents = it.ctx;
            c.n_prompt = 1;
            const std::string ctext = st::render(it.ctx);
            c.prompt_chars = st::render({it.ctx[0]}).size();
            ex.prepare(c);
            const wkn::K5::State s = m.start(ctext, o);
            const auto assoc = m.prompt_assoc(s.ctx);
            const auto& f = md->fs.fact(it.fact);
            const auto& S = md->fs.ent_tok(f.subj);
            const auto& O = md->fs.ent_tok(f.val);
            const auto& sk = *md->bank.of(md->fs.rel(f.rel));
            std::vector<pl::Skel> pool{pl::Skel{it.own, 1}};
            // The others: random real skeletons of the relation (not the best-scored ones: those would differ from the
            // target by being typical, and the mixer would learn "pick the odd one").
            std::vector<std::size_t> idx(sk.size());
            for (std::size_t k = 0; k < idx.size(); ++k) idx[k] = k;
            std::mt19937_64 r(0x9E3779B9ull * (i + 1));
            std::shuffle(idx.begin(), idx.end(), r);
            for (const auto k : idx)
                if (sk[k].codes != it.own && pool.size() < static_cast<std::size_t>(pl::kSkel)) pool.push_back(sk[k]);
            std::vector<std::size_t> pick(pool.size());
            for (std::size_t k = 0; k < pick.size(); ++k) pick[k] = k;
            std::vector<pl::Cand> cands;
            md->candidates(m, ex, fl, c, s, assoc, pool, pick, S, O, cands, f.subj, f.val);
            if (cands.size() >= 2 && cands[0].skel == 0) {
                for (const auto& cd : cands) it.f.emplace_back(cd.f, cd.f + pl::kNF), it.k5.push_back(cd.fill.mean());
                it.ok = true;
            }
            const std::size_t d = ++done;
            if (d % 200 == 0)
                std::fprintf(stderr, "train: %zu of %zu items (%.0f s)\n", d, items.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    };
    const unsigned T = static_cast<unsigned>(std::max(1, std::atoi(a.get("--threads", "8").c_str())));
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < T; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    std::vector<TItem*> ok;
    for (auto& it : items)
        if (it.ok) ok.push_back(&it);
    if (ok.size() < 50) return std::cerr << "train: too few items (" << ok.size() << ")\n", 1;
    const std::size_t ntr = ok.size() * 4 / 5;
    // sd over the training rows
    std::vector<double> sd(pl::kNF, 1.0);
    {
        std::vector<double> s1(pl::kNF, 0), s2(pl::kNF, 0);
        double n = 0;
        for (std::size_t i = 0; i < ntr; ++i)
            for (const auto& r : ok[i]->f) {
                for (int j = 0; j < pl::kNF; ++j) s1[j] += r[j], s2[j] += r[j] * r[j];
                n += 1;
            }
        for (int j = 0; j < pl::kNF; ++j) {
            const double v = s2[j] / n - (s1[j] / n) * (s1[j] / n);
            sd[j] = v > 1e-12 ? std::sqrt(v) : 1.0;
        }
    }
    // Items: the target's row first in the data; shuffle candidate order per item (fixed seed) so position carries nothing.
    std::mt19937_64 rng(20261105);
    std::vector<st::Item> its;
    std::vector<std::vector<double>> k5s;
    for (auto* p : ok) {
        st::Item it;
        it.n = static_cast<int>(p->f.size());
        std::vector<int> perm(static_cast<std::size_t>(it.n));
        for (int i = 0; i < it.n; ++i) perm[static_cast<std::size_t>(i)] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        it.z.resize(static_cast<std::size_t>(it.n * pl::kNF));
        std::vector<double> k5(static_cast<std::size_t>(it.n));
        for (int i = 0; i < it.n; ++i) {
            const auto src = static_cast<std::size_t>(perm[static_cast<std::size_t>(i)]);
            if (src == 0) it.target = i;
            for (int j = 0; j < pl::kNF; ++j) it.z[static_cast<std::size_t>(i * pl::kNF + j)] = static_cast<float>(p->f[src][static_cast<std::size_t>(j)] / sd[static_cast<std::size_t>(j)]);
            k5[static_cast<std::size_t>(i)] = p->k5[src];
        }
        st::Context c;
        c.sents = p->ctx;
        c.n_prompt = 1;
        c.prompt_chars = st::render({p->ctx[0]}).size();
        st::Experts(md->stats, m, md->runs(m)).prepare(c);
        st::selectors(c, it.sel);
        its.push_back(std::move(it));
        k5s.push_back(std::move(k5));
    }
    const std::vector<char> all(pl::kNF, 1);
    st::MetaMixer best;
    double bestl = 1e300;
    for (const double U : {2.0, 4.0, 8.0}) {
        st::MetaMixer mx;
        for (int sel = 0; sel < 3; ++sel)
            for (const double eta : {0.003, 0.01, 0.03}) mx.r.emplace_back(pl::kNF, sel, U, eta, all);
        mx.eta2 = 0.02;
        mx.init_layer2();
        double l = 0;
        for (int pass = 0; pass < 2; ++pass) {
            l = 0;
            for (std::size_t i = 0; i < ntr; ++i) l += mx.learn(its[i]);
        }
        l /= static_cast<double>(ntr);
        std::fprintf(stderr, "train: U %.0f, second-pass loss %.4f bits\n", U, l);
        if (l < bestl) bestl = l, best = mx;
    }
    best.sd = sd;
    // Test: top-1 and log loss of the mixer, K5 alone, MBR alone, chance.
    double acc_mx = 0, acc_k5 = 0, acc_mbr = 0, loss_mx = 0, chance = 0, nt = 0;
    for (std::size_t i = ntr; i < its.size(); ++i) {
        const auto& it = its[i];
        std::vector<double> P;
        best.dist(it, P);
        int bm = 0, bk = 0, bb = 0;
        for (int k = 1; k < it.n; ++k) {
            if (P[static_cast<std::size_t>(k)] > P[static_cast<std::size_t>(bm)]) bm = k;
            if (k5s[i][static_cast<std::size_t>(k)] > k5s[i][static_cast<std::size_t>(bk)]) bk = k;
            if (it.z[static_cast<std::size_t>(k * pl::kNF + pl::kMbr)] > it.z[static_cast<std::size_t>(bb * pl::kNF + pl::kMbr)]) bb = k;
        }
        acc_mx += bm == it.target, acc_k5 += bk == it.target, acc_mbr += bb == it.target;
        loss_mx += -std::log2(std::max(P[static_cast<std::size_t>(it.target)], 1e-300));
        chance += 1.0 / it.n, nt += 1;
    }
    std::vector<double> tk5;
    for (std::size_t i = 0; i < ntr; ++i) tk5.push_back(k5s[i][static_cast<std::size_t>(its[i].target)]);
    std::sort(tk5.begin(), tk5.end());
    md->floor = tk5[tk5.size() / 20];
    md->mx = best;
    md->trained = true;
    if (!md->save(a.get("--out"))) return std::cerr << "cannot write " << a.get("--out") << "\n", 1;
    std::printf("train: %zu items (%zu train, %zu test), mean %.1f candidates\n", its.size(), ntr, its.size() - ntr,
                [&] { double t = 0; for (const auto& x : its) t += x.n; return t / static_cast<double>(its.size()); }());
    std::printf("test top-1: mixer %.3f, K5 alone %.3f, MBR alone %.3f, chance %.3f; mixer log loss %.3f bits\n", acc_mx / nt, acc_k5 / nt, acc_mbr / nt,
                chance / nt, loss_mx / nt);
    std::printf("floor (5th percentile of the targets' K5 mean log2 P): %.3f\n", md->floor);
    std::printf("weights (ranker 0, set 0):");
    for (int j = 0; j < pl::kNF; ++j) std::printf(" %s=%.3f", j == pl::kMbr ? "mbr" : st::feat_name(j), best.r[0].w(0, j));
    std::printf("\n");
    return 0;
}

// PLAN_FILL.md (§13.42 Q2): the options' defaults change nothing; shapes; EM types on a small store; the context model's
// signs; a typed fill refuses a word that does not suit its slot; S2 at temperature 1 equals the mixer; S4 at 0 equals
// the old transitions.
int selftest_q2(const wkn::K5& m, const std::string& txt, const std::vector<std::uint16_t>& codes, const wkn::K5::State& s0, const pl::Fill& f) {
    const auto fail = [](const char* what) { return std::fprintf(stderr, "selftest q2: %s\n", what), 1; };
    pl::FillOpts fo;
    std::string err;
    if (fo.any() || !fo.desc().empty()) return fail("default options");
    if (!pl::fill_flag("--k5-plan-s1", "0.05", fo, err) || !err.empty() || fo.s1 != 0.05 || !fo.any()) return fail("--k5-plan-s1");
    if (!pl::fill_flag("--k5-plan-s3", "0", fo, err) || err.empty()) return fail("--k5-plan-s3 0 must be refused");
    if (pl::fill_flag("--k5-plan-beam", "4", fo, err)) return fail("fill_flag took another flag");
    if (pl::name_shape(wkn::tokens("1954")) != 0 || pl::name_shape(wkn::tokens("Paris")) != 6 || pl::name_shape(wkn::tokens("Gothenburg University")) != 7 ||
        pl::name_shape(wkn::tokens("Military personnel")) != 10 || pl::name_shape(wkn::tokens("May 4 , 1950")) != 5)
        return fail("name shapes");
    const std::string dir = "/tmp/k5_planner_selftest_q2";
    if (std::system(("mkdir -p " + dir).c_str()) != 0) return fail("mkdir");
    {
        std::ofstream s(dir + "/seeds.tsv");
        s << "subject\trelation\tobject\tsource\ttemplate\tbyte_offset\n";
        const char* ppl[][3] = {{"Alan Smith", "Paris", "1901"}, {"Bob Jones", "Rome", "1910"}, {"Carl Brown", "Paris", "1920"},
                                {"Dan White", "Berlin", "1930"}, {"Eve Black", "Rome", "1940"}, {"Fay Green", "Berlin", "1950"}};
        for (const auto& p : ppl)
            s << p[0] << "\tbirth_place\t" << p[1] << "\tinfobox\tinfobox\t5\n" << p[0] << "\tbirth_year\t" << p[2] << "\tinfobox\tinfobox\t5\n";
        s << "Paris\tcountry\tFrance\tinfobox\tinfobox\t5\nRome\tcountry\tItaly\tinfobox\tinfobox\t5\nBerlin\tcountry\tGermany\tinfobox\tinfobox\t5\n";
    }
    pl::FactStore fs;
    if (!fs.load(dir, {}, err)) return fail("store");
    pl::SlotTypes ty;
    const auto ob = ty.observe(fs, m, true);
    ty.fit(ob, 4, 60, 4, 7);
    const auto post = ty.posteriors(ob);
    const auto type = [&](const std::string& e) {
        const float* p = post.data() + static_cast<std::size_t>(fs.find(e)) * 4;
        return static_cast<int>(std::max_element(p, p + 4) - p);
    };
    if (type("Paris") != type("Rome") || type("Rome") != type("Berlin") || type("1901") != type("1950") || type("Alan Smith") != type("Fay Green") ||
        type("Paris") == type("1901") || type("Paris") == type("Alan Smith") || type("1901") == type("Alan Smith"))
        return fail("EM types");
    ty.count_text(txt, 0, txt.size(), fs, m, post);
    if (!ty.save(dir + "/slots.txt", m, 1)) return fail("save");
    pl::SlotTypes t2;
    if (!t2.load(dir + "/slots.txt", err)) return fail(err.c_str());
    t2.bind(m);
    const auto P = [&](const std::string& e) { return post.data() + static_cast<std::size_t>(fs.find(e)) * 4; };
    const int m2 = pl::slot_off_index(-2), p2 = pl::slot_off_index(2);
    const double fc = t2.word_fit(P("Paris"), m2, m.id("born")), fp = t2.word_fit(P("Alan Smith"), m2, m.id("born"));
    const double fq = t2.word_fit(P("Alan Smith"), p2, m.id("born")), fr = t2.word_fit(P("Paris"), p2, m.id("born"));
    if (!(fc > 0 && fp < -2 && fq > 0 && fr < -2)) return std::fprintf(stderr, "selftest q2: context fits %.2f %.2f %.2f %.2f\n", fc, fp, fq, fr), 1;
    if (std::fabs(t2.word_fit(P("Paris"), m2, m.id("born")) - ty.word_fit(P("Paris"), m2, m.id("born"))) > 1e-3) return fail("slots file round trip");
    {  // --k5-plan-typed-fit log: the mean log ratio is never above log2 of the mixed ratio (Jensen), equal for a sure type
        float mixp[4] = {0, 0, 0, 0};
        mixp[type("Paris")] = 0.5f, mixp[type("Alan Smith")] = 0.5f;
        const double mx = t2.word_fit(mixp, m2, m.id("born"));
        t2.fit_mode = 1;
        const double lg = t2.word_fit(mixp, m2, m.id("born")), sure = t2.word_fit(P("Paris"), m2, m.id("born"));
        t2.fit_mode = 0;
        if (!(lg <= mx + 1e-12 && lg < mx - 0.1 && std::fabs(sure - fc) < 0.05)) return std::fprintf(stderr, "selftest q2: log fit %.3f %.3f %.3f\n", mx, lg, sure), 1;
        pl::FillOpts fl2;
        std::string e2;
        if (!pl::fill_flag("--k5-plan-typed-fit", "log", fl2, e2) || !e2.empty() || fl2.typed_fit != 1) return fail("--k5-plan-typed-fit log");
        if (!pl::fill_flag("--k5-plan-typed-fit", "geo", fl2, e2) || e2.empty()) return fail("--k5-plan-typed-fit geo must be refused");
    }
    // Typed fills: the right types keep the untyped fill; the names' types swapped refuse "born" after the city.
    const wkn::Options o;
    pl::FillOpts ft;
    ft.typed = dir + "/slots.txt";
    const pl::Filler fl(m, o, 4, 3, false, &ft, &t2);
    const auto S = wkn::tokens("Alan Smith"), O = wkn::tokens("Paris");
    const auto g = fl.fill(s0, codes, S, O, P("Alan Smith"), P("Paris"));
    if (!g.ok || g.tok != f.tok) return fail("typed fill with the right types");
    const auto h = fl.fill(s0, codes, S, O, P("Paris"), P("Alan Smith"));
    if (fl.refused() == 0 || (h.ok && h.tok.size() > 3 && h.tok[3] == "born")) return fail("typed fill with swapped types");
    // S1 changes the fill's probabilities, not its tokens' places.
    pl::FillOpts f1;
    f1.s1 = 0.1;
    const pl::Filler fs1(m, o, 4, 3, false, &f1);
    const auto g1 = fs1.fill(s0, codes, S, O);
    if (!g1.ok || g1.tok.size() != f.tok.size() || g1.tok[0] != "Alan" || g1.tok[5] != "Paris" || g1.lp == f.lp) return fail("S1 fill");
    // S2: temperature 1 is the mixer itself; a higher one is flatter.
    pl::Model md;
    md.trained = true;
    const std::vector<char> all(pl::kNF, 1);
    md.mx.r.emplace_back(pl::kNF, 0, 4.0, 0.05, all), md.mx.r.emplace_back(pl::kNF, 2, 4.0, 0.2, all);
    md.mx.init_layer2();
    st::Item it;
    it.n = 3;
    it.z.resize(3 * pl::kNF);
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<float> uz(-1, 1);
    for (auto& x : it.z) x = uz(rng);
    for (int r = 0; r < 30; ++r) md.mx.learn(it);
    std::vector<double> P1, PT, P2;
    md.mx.dist(it, P1);
    md.dist_temp(it, 1.0, PT);
    md.dist_temp(it, 3.0, P2);
    for (int i = 0; i < 3; ++i)
        if (std::fabs(P1[static_cast<std::size_t>(i)] - PT[static_cast<std::size_t>(i)]) > 1e-12) return fail("S2 at temperature 1");
    if (!(*std::max_element(P2.begin(), P2.end()) < *std::max_element(P1.begin(), P1.end()))) return fail("S2 flatter");
    // S4: 0 is the old add-half transitions; above 0 it moves towards the marginal.
    pl::SkelBank b;
    b.add("birth_place", codes), b.add("founded", codes), b.add_trans("<s>", "birth_place"), b.add_trans("<s>", "birth_place"), b.add_trans("birth_place", "founded");
    b.finish();
    if (!b.save(dir + "/skel.txt")) return fail("bank save");
    pl::SkelBank b2;
    if (!b2.load(dir + "/skel.txt", err)) return fail("bank load");
    const double old = std::log2((0 + 0.5) / (2 + 0.5 * 3));
    if (b2.trans_lp("<s>", "founded") != old) return fail("S4 at 0");
    b2.s4 = 0.1;
    if (!(b2.trans_lp("<s>", "founded") > old)) return fail("S4 above 0");
    std::printf("k5_planner selftest q2: ok (typed fill with swapped types: %s; refused %llu)\n",
                h.ok ? [&] { std::string t; for (const auto& x : h.tok) t += x + " "; return t; }().c_str() : "no fill", static_cast<unsigned long long>(fl.refused()));
    return 0;
}

int cmd_selftest() {
    // A tiny K5 and store: a mined skeleton, a fill that keeps the fact's tokens and the class codes.
    std::string txt;
    for (int i = 0; i < 40; ++i)
        txt += "Alan Smith was born in Paris . He was a painter . Bob Jones was born in Rome . The city is old . Alan Smith lived in Paris .\n";
    wkn::K5 m;
    wkn::Options bo;
    bo.classes = 8;
    m.build(txt, 12, 5, bo.classes);
    const std::string dir = "/tmp/k5_planner_selftest";
    std::system(("mkdir -p " + dir).c_str());
    {
        std::ofstream s(dir + "/seeds.tsv");
        s << "subject\trelation\tobject\tsource\ttemplate\tbyte_offset\nAlan Smith\tbirth_place\tParis\tinfobox\tinfobox\t5\n"
          << "Bob Jones\tbirth_place\tRome\tinfobox\tinfobox\t500\n";
    }
    pl::FactStore fs;
    std::string err;
    if (!fs.load(dir, {{400, 600}}, err) || fs.size() != 1 || fs.dropped() != 1) return std::fprintf(stderr, "selftest: store load/drop failed\n"), 1;
    const auto t = wkn::tokens("Alan Smith was born in Paris .");
    pl::Match mt;
    if (!pl::match_sentence(fs, t, mt) || mt.sa != 0 || mt.sb != 2 || mt.oa != 5) return std::fprintf(stderr, "selftest: match failed\n"), 1;
    std::vector<std::uint16_t> codes;
    if (!pl::shape_of(m, t, mt, codes) || codes.size() != 6 || codes[0] != pl::kSlotS || codes[4] != pl::kSlotO)
        return std::fprintf(stderr, "selftest: shape failed\n"), 1;
    const wkn::Options o;
    const pl::Filler fl(m, o);
    const auto s0 = m.start("The city is old .", o);
    const auto f = fl.fill(s0, codes, fs.ent_tok(fs.fact(0).subj), fs.ent_tok(fs.fact(0).val));
    if (!f.ok || f.tok.size() != 7 || f.tok[0] != "Alan" || f.tok[1] != "Smith" || f.tok[5] != "Paris") return std::fprintf(stderr, "selftest: fill failed\n"), 1;
    const auto& cls = m.classes().classes();
    for (const std::size_t i : {2u, 3u, 4u, 6u})
        if (cls[f.ids[i]] != codes[i - 1]) return std::fprintf(stderr, "selftest: fill broke a class code\n"), 1;
    // MERGE_FILL.md: dynamic beam allocation. Bank sizes (no slot left unused while candidates remain); the fill writes
    // each name once, whole, and the skeleton's class codes in order.
    if (pl::bank_sizes(3, 4, {3, 2, 0, 0}) != std::vector<int>{2, 2, 0, 0} || pl::bank_sizes(5, 4, {3, 2, 0, 0, 0, 0}) != std::vector<int>{2, 2, 0, 0, 0, 0} ||
        pl::bank_sizes(2, 4, {0, 0, 5}) != std::vector<int>{0, 0, 4} || pl::bank_sizes(3, 4, {0, 0, 3, 1}) != std::vector<int>{0, 0, 3, 1} ||
        pl::bank_sizes(3, 8, {5, 1, 1, 9}) != std::vector<int>{3, 1, 1, 3} || pl::bank_sizes(1, 4, {1, 1}) != std::vector<int>{1, 1})
        return std::fprintf(stderr, "selftest: bank sizes\n"), 1;
    for (const int beam : {1, 4, 8}) {
        const pl::Filler fd(m, o, beam, 3, true);
        const auto g = fd.fill(s0, codes, fs.ent_tok(fs.fact(0).subj), fs.ent_tok(fs.fact(0).val));
        if (!g.ok || g.tok.size() != 7) return std::fprintf(stderr, "selftest: dba fill failed (beam %d)\n", beam), 1;
        std::vector<std::uint16_t> rest;
        int alan = -1, paris = -1;
        for (std::size_t i = 0; i < g.tok.size(); ++i) {
            if (g.tok[i] == "Alan" && i + 1 < g.tok.size() && g.tok[i + 1] == "Smith" && alan < 0) alan = static_cast<int>(i), ++i;
            else if (g.tok[i] == "Paris" && paris < 0) paris = static_cast<int>(i);
            else rest.push_back(static_cast<std::uint16_t>(cls[g.ids[i]]));
        }
        std::vector<std::uint16_t> want;
        for (const auto c : codes)
            if (c != pl::kSlotS && c != pl::kSlotO) want.push_back(c);
        if (alan < 0 || paris < 0 || rest != want) return std::fprintf(stderr, "selftest: dba fill broke a name or a class code (beam %d)\n", beam), 1;
        if (fd.dists() == 0 || fd.dists() > static_cast<std::uint64_t>(beam) * g.tok.size())  // one K5 distribution per hypothesis and step
            return std::fprintf(stderr, "selftest: dba distributions %llu (beam %d)\n", static_cast<unsigned long long>(fd.dists()), beam), 1;
        if (beam == 4) {
            std::string t;
            for (const auto& x : g.tok) t += x + " ";
            std::printf("k5_planner selftest: dba fill (beam 4): %s\n", t.c_str());
        }
    }
    pl::SkelBank b;
    b.add("birth_place", codes), b.add_trans("<s>", "birth_place");
    b.finish();
    if (!b.save(dir + "/skel.txt")) return 1;
    pl::SkelBank b2;
    if (!b2.load(dir + "/skel.txt", err) || !b2.of("birth_place") || b2.of("birth_place")->at(0).codes != codes) return std::fprintf(stderr, "selftest: bank round trip\n"), 1;
    const auto u = pl::mbr_ids({&f.ids, &f.ids});
    if (std::fabs(u[0] - 1.0) > 1e-12) return std::fprintf(stderr, "selftest: mbr\n"), 1;
    if (const int r = selftest_q2(m, txt, codes, s0, f)) return r;
    std::string fill;
    for (const auto& x : f.tok) fill += x + " ";
    std::printf("k5_planner selftest: ok (fill: %s)\n", fill.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: k5_planner mine|train|types|selftest [--key value ...]\n");
        return 2;
    }
    const std::string cmd = argv[1];
    Args a;
    for (int i = 2; i + 1 < argc; i += 2) a.kv[argv[i]] = argv[i + 1];
    if (cmd == "mine") return cmd_mine(a);
    if (cmd == "train") return cmd_train(a);
    if (cmd == "types") return cmd_types(a);
    if (cmd == "selftest") return cmd_selftest();
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
