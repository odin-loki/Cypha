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
    const pl::Filler fl(m, o);
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
            md->candidates(m, ex, fl, c, s, assoc, pool, pick, S, O, cands);
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
    pl::SkelBank b;
    b.add("birth_place", codes), b.add_trans("<s>", "birth_place");
    b.finish();
    if (!b.save(dir + "/skel.txt")) return 1;
    pl::SkelBank b2;
    if (!b2.load(dir + "/skel.txt", err) || !b2.of("birth_place") || b2.of("birth_place")->at(0).codes != codes) return std::fprintf(stderr, "selftest: bank round trip\n"), 1;
    const auto u = pl::mbr_ids({&f.ids, &f.ids});
    if (std::fabs(u[0] - 1.0) > 1e-12) return std::fprintf(stderr, "selftest: mbr\n"), 1;
    std::string fill;
    for (const auto& x : f.tok) fill += x + " ";
    std::printf("k5_planner selftest: ok (fill: %s)\n", fill.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: k5_planner mine|train|selftest [--key value ...]\n");
        return 2;
    }
    const std::string cmd = argv[1];
    Args a;
    for (int i = 2; i + 1 < argc; i += 2) a.kv[argv[i]] = argv[i + 1];
    if (cmd == "mine") return cmd_mine(a);
    if (cmd == "train") return cmd_train(a);
    if (cmd == "selftest") return cmd_selftest();
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
