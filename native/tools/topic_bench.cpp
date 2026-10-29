// C0 / C7 (PLAN Track C; docs/reports/C0_C7_BASELINES_AND_ROUTING.md): topic classification with
// count-based features. Fair baselines for Core Cypha's text routing (C0), and C1's PPMI-SVD vectors in
// place of Branch A's hashing embedder (C7), on 20 Newsgroups and AG News.
//
// Features (all from counts; no neural network):
//   H    Branch A's own hashing featuriser (branch_a_router.cpp: lower case, white-space split, unigrams and
//        bigrams, MurmurHash3 seed 0, 512 buckets, counts, L2), then PCA to 128 on the training rows and
//        z-scored components (the router's hashing_svd)
//   T    TF-IDF, sparse: lower case, runs of letters and digits, unigrams + bigrams hashed (MurmurHash3) into
//        2^18 buckets, tf = 1 + ln(count), idf = ln((1 + N) / (1 + df)) + 1, L2
//   Td   T through a fixed signed random projection to 1,024 (E4's), then z-scored and PCA-whitened to k
//   S    SIF mean of PPMI-SVD word vectors (a / (a + p(w)), a = 1e-3), first principal component removed;
//        for Cypha whitened to its full width
//   STd  S and Td concatenated, whitened together to k
// Models: Cypha (cypha::Cypha from create_fresh_model_root; W_enc frozen = enc_lr 0, or learning =
// enc_lr 0.002), multinomial logistic regression (Adagrad) on T or S, averaged perceptron on T.
//
// Usage:
//   topic_bench run --train TRAIN.tsv --test TEST.tsv --arm ARM --protocol 2k|full [--n-per-class N]
//                   [--vectors V.f32 --vocab V.tsv] [--seed S] [--refit-seed S2] [--read-test] [--out PREFIX]
//     ARM: cypha_H cypha_Td cypha_S cypha_STd cypha_learn_Td cypha_learn_S lr_T lr_S perc_T
//     Prints DEV lines for every setting tried, the CHOICE, and with --read-test refits on all training
//     rows and scores the test file once: RESULT line, PREFIX.preds (index gold pred conf brier).
//   topic_bench compare A.preds B.preds [--eps 0.005] [--draws 4000] [--seed 1]
//     Paired Bayesian bootstrap (Dirichlet weights) of accuracy(A) - accuracy(B): mean, 95% interval,
//     P(delta > eps), P(delta > -1 point).
//   topic_bench selftest
// Input files: "label<TAB>text" per line (scripts/topic_data.py).
#include "cypha/create_model.hpp"
#include "cypha/cypha.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/text_features.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace {

using namespace cypha::textfeat;
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// ------------------------------------------------------------------------------------------------ data
struct Doc {
    int label{0};
    std::string text;
};

std::vector<Doc> read_tsv(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<Doc> out;
    std::string line;
    while (std::getline(in, line)) {
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        out.push_back({std::atoi(line.substr(0, tab).c_str()), line.substr(tab + 1)});
    }
    return out;
}

int n_classes(const std::vector<Doc>& d) {
    int k = 0;
    for (const auto& x : d) k = std::max(k, x.label + 1);
    return k;
}

// Stratified pick: per class, shuffle that class's indices with the seed and take the first n.
std::vector<int> stratified_sample(const std::vector<Doc>& d, int K, int n_per_class, std::uint32_t seed) {
    std::vector<std::vector<int>> by(K);
    for (int i = 0; i < static_cast<int>(d.size()); ++i) by[d[i].label].push_back(i);
    std::vector<int> out;
    for (int k = 0; k < K; ++k) {
        std::mt19937 rng(seed + 7919u * static_cast<std::uint32_t>(k));
        std::shuffle(by[k].begin(), by[k].end(), rng);
        const int n = n_per_class < 0 ? static_cast<int>(by[k].size()) : std::min<int>(n_per_class, by[k].size());
        out.insert(out.end(), by[k].begin(), by[k].begin() + n);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Stratified dev split of a set of indices: per class, the first ceil(frac * n) after a seeded shuffle.
void stratified_dev(const std::vector<Doc>& d, const std::vector<int>& idx, int K, double frac, std::uint32_t seed,
                    std::vector<int>& tr, std::vector<int>& dv) {
    std::vector<std::vector<int>> by(K);
    for (int i : idx) by[d[i].label].push_back(i);
    tr.clear();
    dv.clear();
    for (int k = 0; k < K; ++k) {
        std::mt19937 rng(seed * 31u + 104729u * static_cast<std::uint32_t>(k));
        std::shuffle(by[k].begin(), by[k].end(), rng);
        const int nd = static_cast<int>(std::ceil(frac * static_cast<double>(by[k].size()) - 1e-9));
        for (int j = 0; j < static_cast<int>(by[k].size()); ++j) (j < nd ? dv : tr).push_back(by[k][j]);
    }
    std::sort(tr.begin(), tr.end());
    std::sort(dv.begin(), dv.end());
}

// ------------------------------------------------------------------------------------------------ models
struct Scores {
    std::vector<int> gold, pred;
    std::vector<double> conf, brier;  // conf: max softmax probability; brier: multiclass Brier score
    double train_s{0}, us_per_doc{0};
};

double softmax_into(const std::vector<double>& s, std::vector<double>& p) {
    p.resize(s.size());
    const double mx = *std::max_element(s.begin(), s.end());
    double z = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        p[i] = std::isfinite(s[i]) ? std::exp(s[i] - mx) : 0.0;
        z += p[i];
    }
    for (double& x : p) x /= z;
    return z;
}

void score_row(Scores& sc, const std::vector<double>& logits, int gold) {
    std::vector<double> p;
    softmax_into(logits, p);
    const int pr = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
    double b = 0;
    for (std::size_t k = 0; k < p.size(); ++k) {
        const double y = static_cast<int>(k) == gold ? 1.0 : 0.0;
        b += (p[k] - y) * (p[k] - y);
    }
    sc.gold.push_back(gold);
    sc.pred.push_back(pr);
    sc.conf.push_back(p[pr]);
    sc.brier.push_back(b);
}

double accuracy(const Scores& s) {
    int ok = 0;
    for (std::size_t i = 0; i < s.gold.size(); ++i) ok += s.gold[i] == s.pred[i];
    return s.gold.empty() ? 0.0 : static_cast<double>(ok) / static_cast<double>(s.gold.size());
}

double macro_f1(const Scores& s, int K) {
    std::vector<int> tp(K, 0), fp(K, 0), fn(K, 0);
    for (std::size_t i = 0; i < s.gold.size(); ++i) {
        if (s.gold[i] == s.pred[i]) {
            tp[s.gold[i]]++;
        } else {
            fp[s.pred[i]]++;
            fn[s.gold[i]]++;
        }
    }
    double f = 0;
    int m = 0;
    for (int k = 0; k < K; ++k) {
        if (tp[k] + fp[k] + fn[k] == 0) continue;
        f += 2.0 * tp[k] / (2.0 * tp[k] + fp[k] + fn[k]);
        ++m;
    }
    return m ? f / m : 0.0;
}

double ece(const Scores& s, int bins = 15) {
    std::vector<double> sc(bins, 0), sa(bins, 0);
    std::vector<int> n(bins, 0);
    for (std::size_t i = 0; i < s.conf.size(); ++i) {
        const int b = std::min(bins - 1, static_cast<int>(s.conf[i] * bins));
        n[b]++;
        sc[b] += s.conf[i];
        sa[b] += s.gold[i] == s.pred[i] ? 1.0 : 0.0;
    }
    double e = 0;
    for (int b = 0; b < bins; ++b)
        if (n[b]) e += static_cast<double>(n[b]) / s.conf.size() * std::fabs(sa[b] / n[b] - sc[b] / n[b]);
    return e;
}

double mean_of(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / v.size();
}

void wilson(double k, double n, double& lo, double& hi) {
    const double z = 1.959963984540054, p = n > 0 ? k / n : 0.0;
    const double den = 1 + z * z / n, cen = (p + z * z / (2 * n)) / den;
    const double half = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den;
    lo = cen - half;
    hi = cen + half;
}

// Cypha with W_enc frozen (enc_lr 0) or learning (enc_lr 0.002), E4's wrapper.
struct CyphaClf {
    cypha::Cypha model;
    bool init(int dim, double world_lr, double delta_lr, bool learn_enc) {
        static int counter = 0;
        const auto dir = std::filesystem::temp_directory_path() /
                         ("topic_bench_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(dir);
        cypha::FreshModelParams fp;
        fp.input_dim = dim;
        fp.field_dim = 16;
        fp.temperature = 1.0;
        fp.world_lr = world_lr;
        fp.delta_lr = delta_lr;
        const auto path = (dir / "fresh.cypha").string();
        cypha::save_cypha_file(path.c_str(), cypha::create_fresh_model_root(fp));
        const bool ok = model.load(path);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        if (!ok) return false;
        model.world_lr() = world_lr;
        model.delta_lr() = delta_lr;
        model.ood_sigma() = 12.0;
        model.train_params().enc_lr = learn_enc ? 0.002 : 0.0;
        model.train_params().replay_ratio = 0.0;
        return true;
    }
    void learn(const std::vector<double>& x, int label) { learn_named(x, std::to_string(label)); }
    void learn_named(const std::vector<double>& x, const std::string& lab) {
        cypha::UpdateOpts uo;
        uo.use_gh = true;
        const auto r = model.update(x.data(), static_cast<int>(x.size()), &lab, nullptr, uo);
        if (!r.detail.empty()) {
            std::fprintf(stderr, "cypha update: %s\n", r.detail.c_str());
            std::exit(3);
        }
    }
    std::vector<double> logits(const std::vector<double>& x, int K) {
        cypha::PredictOpts po;
        po.use_gh = true;
        const auto p = model.predict(x.data(), static_cast<int>(x.size()), po);
        std::vector<double> lg(K, -std::numeric_limits<double>::infinity());
        for (std::size_t j = 0; j < p.labels.size(); ++j) {
            const int k = std::atoi(p.labels[j].c_str());
            if (k >= 0 && k < K && j < p.all_scores.size()) lg[k] = p.all_scores[j];
        }
        return lg;
    }
};

// Multinomial logistic regression with Adagrad, on sparse rows or dense rows.
struct LogReg {
    int K{0};
    std::size_t F{0};
    double lambda{0}, eta{0.5};
    std::vector<float> w, g2, b, gb2;
    void init(int k, std::size_t f, double lam) {
        K = k;
        F = f;
        lambda = lam;
        w.assign(static_cast<std::size_t>(K) * F, 0.f);
        g2.assign(w.size(), 0.f);
        b.assign(K, 0.f);
        gb2.assign(K, 0.f);
    }
    std::vector<double> logits(const SparseRow& x) const {
        std::vector<double> s(K);
        for (int k = 0; k < K; ++k) {
            const float* wk = w.data() + static_cast<std::size_t>(k) * F;
            double acc = b[k];
            for (std::size_t j = 0; j < x.idx.size(); ++j) acc += static_cast<double>(wk[x.idx[j]]) * x.val[j];
            s[k] = acc;
        }
        return s;
    }
    void step(const SparseRow& x, int y) {
        std::vector<double> p;
        softmax_into(logits(x), p);
        for (int k = 0; k < K; ++k) {
            const double gk = p[k] - (k == y ? 1.0 : 0.0);
            float* wk = w.data() + static_cast<std::size_t>(k) * F;
            float* gk2 = g2.data() + static_cast<std::size_t>(k) * F;
            for (std::size_t j = 0; j < x.idx.size(); ++j) {
                const std::uint32_t f = x.idx[j];
                const double g = gk * x.val[j] + lambda * wk[f];
                gk2[f] += static_cast<float>(g * g);
                wk[f] -= static_cast<float>(eta * g / (std::sqrt(gk2[f]) + 1e-8));
            }
            gb2[k] += static_cast<float>(gk * gk);
            b[k] -= static_cast<float>(eta * gk / (std::sqrt(gb2[k]) + 1e-8));
        }
    }
};

SparseRow dense_as_sparse(const std::vector<double>& x) {
    SparseRow r;
    for (std::size_t j = 0; j < x.size(); ++j) {
        r.idx.push_back(static_cast<std::uint32_t>(j));
        r.val.push_back(static_cast<float>(x[j]));
    }
    return r;
}

// Averaged perceptron (Collins 2002), as E4.
struct Perceptron {
    int K{0};
    std::size_t F{0};
    std::vector<float> w, u;
    double c{1};
    void init(int k, std::size_t f) {
        K = k;
        F = f;
        w.assign(static_cast<std::size_t>(K) * F, 0.f);
        u.assign(w.size(), 0.f);
        c = 1;
    }
    std::vector<double> scores(const SparseRow& x, const std::vector<float>& wt) const {
        std::vector<double> s(K, 0.0);
        for (int k = 0; k < K; ++k) {
            const float* wk = wt.data() + static_cast<std::size_t>(k) * F;
            double acc = 0;
            for (std::size_t j = 0; j < x.idx.size(); ++j) acc += static_cast<double>(wk[x.idx[j]]) * x.val[j];
            s[k] = acc;
        }
        return s;
    }
    void step(const SparseRow& x, int y) {
        const auto s = scores(x, w);
        const int p = static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin());
        if (p != y) {
            float* wy = w.data() + static_cast<std::size_t>(y) * F;
            float* wp = w.data() + static_cast<std::size_t>(p) * F;
            float* uy = u.data() + static_cast<std::size_t>(y) * F;
            float* up = u.data() + static_cast<std::size_t>(p) * F;
            const float cc = static_cast<float>(c);
            for (std::size_t j = 0; j < x.idx.size(); ++j) {
                wy[x.idx[j]] += x.val[j];
                wp[x.idx[j]] -= x.val[j];
                uy[x.idx[j]] += cc * x.val[j];
                up[x.idx[j]] -= cc * x.val[j];
            }
        }
        c += 1;
    }
    std::vector<float> averaged() const {
        std::vector<float> a(w.size());
        const float inv = static_cast<float>(1.0 / c);
        for (std::size_t i = 0; i < w.size(); ++i) a[i] = w[i] - u[i] * inv;
        return a;
    }
};

// ------------------------------------------------------------------------------------------------ features per arm
using Featurizer = TextFeaturizer;

void fit_on(Featurizer& fz, const std::vector<Doc>& d, const std::vector<int>& idx) {
    std::vector<const std::string*> docs;
    for (int i : idx) docs.push_back(&d[i].text);
    fz.fit(docs);
}

// ------------------------------------------------------------------------------------------------ train / eval
struct Setting {
    double world_lr{0}, delta_lr{0}, lambda{0};
    int k{0};
    int passes{1};
};

std::string setting_str(const std::string& model, const Setting& s) {
    char b[160];
    if (model == "cypha")
        std::snprintf(b, sizeof b, "world_lr=%g delta_lr=%g k=%d passes=%d", s.world_lr, s.delta_lr, s.k, s.passes);
    else if (model == "lr")
        std::snprintf(b, sizeof b, "lambda=%g k=%d epochs=%d", s.lambda, s.k, s.passes);
    else
        std::snprintf(b, sizeof b, "k=%d epochs=%d", s.k, s.passes);
    return b;
}

// Train on `tr`, evaluate `ev` after each pass 1..max_passes. Returns per-pass scores.
std::vector<Scores> train_eval(const std::string& model, bool learn_enc, const Featurizer& fz, const std::vector<Doc>& trd,
                               const std::vector<int>& tr, const std::vector<Doc>& evd, const std::vector<int>& ev, int K,
                               const Setting& st, int max_passes, std::uint32_t order_seed) {
    std::vector<Scores> out;
    const auto t0 = Clock::now();
    // featurise once
    std::vector<std::vector<double>> Xd_tr, Xd_ev;
    std::vector<SparseRow> Xs_tr, Xs_ev;
    if (fz.sparse()) {
        for (int i : tr) Xs_tr.push_back(fz.tfidf.row(trd[i].text));
    } else {
        for (int i : tr) Xd_tr.push_back(fz.dense(trd[i].text));
    }
    const auto tf0 = Clock::now();
    if (fz.sparse()) {
        for (int i : ev) Xs_ev.push_back(fz.tfidf.row(evd[i].text));
    } else {
        for (int i : ev) Xd_ev.push_back(fz.dense(evd[i].text));
    }
    const double feat_ev_s = secs(tf0);
    std::vector<int> order(tr.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(order_seed);
    if (model == "cypha") {
        CyphaClf c;
        if (!c.init(static_cast<int>(Xd_tr[0].size()), st.world_lr, st.delta_lr, learn_enc)) {
            std::fprintf(stderr, "cypha init failed\n");
            std::exit(3);
        }
        for (int p = 1; p <= max_passes; ++p) {
            std::shuffle(order.begin(), order.end(), rng);
            for (int i : order) c.learn(Xd_tr[i], trd[tr[i]].label);
            Scores sc;
            sc.train_s = secs(t0);
            const auto ti = Clock::now();
            for (std::size_t j = 0; j < ev.size(); ++j) score_row(sc, c.logits(Xd_ev[j], K), evd[ev[j]].label);
            sc.us_per_doc = (secs(ti) + feat_ev_s) * 1e6 / std::max<std::size_t>(1, ev.size());
            out.push_back(std::move(sc));
        }
    } else if (model == "lr") {
        LogReg m;
        const std::size_t F = fz.sparse() ? (1u << kTBits) : Xd_tr[0].size();
        m.init(K, F, st.lambda);
        if (!fz.sparse()) {
            for (auto& x : Xd_tr) Xs_tr.push_back(dense_as_sparse(x));
            for (auto& x : Xd_ev) Xs_ev.push_back(dense_as_sparse(x));
        }
        for (int p = 1; p <= max_passes; ++p) {
            std::shuffle(order.begin(), order.end(), rng);
            for (int i : order) m.step(Xs_tr[i], trd[tr[i]].label);
            Scores sc;
            sc.train_s = secs(t0);
            const auto ti = Clock::now();
            for (std::size_t j = 0; j < ev.size(); ++j) score_row(sc, m.logits(Xs_ev[j]), evd[ev[j]].label);
            sc.us_per_doc = (secs(ti) + feat_ev_s) * 1e6 / std::max<std::size_t>(1, ev.size());
            out.push_back(std::move(sc));
        }
    } else {  // perceptron
        Perceptron m;
        m.init(K, 1u << kTBits);
        for (int p = 1; p <= max_passes; ++p) {
            std::shuffle(order.begin(), order.end(), rng);
            for (int i : order) m.step(Xs_tr[i], trd[tr[i]].label);
            const auto avg = m.averaged();
            Scores sc;
            sc.train_s = secs(t0);
            const auto ti = Clock::now();
            for (std::size_t j = 0; j < ev.size(); ++j) score_row(sc, m.scores(Xs_ev[j], avg), evd[ev[j]].label);
            sc.us_per_doc = (secs(ti) + feat_ev_s) * 1e6 / std::max<std::size_t>(1, ev.size());
            out.push_back(std::move(sc));
        }
    }
    return out;
}

struct ArmSpec {
    std::string model, feat;
    bool learn_enc{false};
    std::vector<int> ks;
};

ArmSpec arm_spec(const std::string& arm) {
    if (arm == "cypha_H") return {"cypha", "H", false, {128}};
    if (arm == "cypha_Td") return {"cypha", "Td", false, {128, 256}};
    if (arm == "cypha_S") return {"cypha", "S", false, {300}};
    if (arm == "cypha_STd") return {"cypha", "STd", false, {256, 384}};
    if (arm == "cypha_learn_Td") return {"cypha", "Td", true, {128, 256}};
    if (arm == "cypha_learn_S") return {"cypha", "S", true, {300}};
    if (arm == "lr_T") return {"lr", "T", false, {0}};
    if (arm == "lr_S") return {"lr", "Sz", false, {300}};
    if (arm == "perc_T") return {"perc", "T", false, {0}};
    std::fprintf(stderr, "unknown arm %s\n", arm.c_str());
    std::exit(2);
}

int run(int argc, char** argv) {
    std::string train_path, test_path, arm, protocol = "2k", vec_path, vocab_path, out_prefix;
    int n_per_class = -1;
    std::uint32_t seed = 1, refit_seed = 0;  // refit_seed 0: the refit uses `seed`
    bool read_test = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--train") train_path = next();
        else if (a == "--test") test_path = next();
        else if (a == "--arm") arm = next();
        else if (a == "--protocol") protocol = next();
        else if (a == "--n-per-class") n_per_class = std::atoi(next().c_str());
        else if (a == "--vectors") vec_path = next();
        else if (a == "--vocab") vocab_path = next();
        else if (a == "--seed") seed = static_cast<std::uint32_t>(std::atoi(next().c_str()));
        else if (a == "--refit-seed") refit_seed = static_cast<std::uint32_t>(std::atoi(next().c_str()));
        else if (a == "--read-test") read_test = true;
        else if (a == "--out") out_prefix = next();
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    const ArmSpec sp = arm_spec(arm);
    const auto train = read_tsv(train_path);
    const int K = n_classes(train);
    WordVectors wv;
    if (sp.feat == "S" || sp.feat == "STd" || sp.feat == "Sz") {
        if (vec_path.empty() || vocab_path.empty()) {
            std::fprintf(stderr, "arm %s needs --vectors and --vocab\n", arm.c_str());
            return 2;
        }
        wv = load_vectors(vec_path, vocab_path);
    }
    // The pool: 2k = n_per_class a class (stratified, seed 20261029); full = all. Dev: 20% (2k) or 10% (full).
    if (protocol == "2k" && n_per_class < 0) n_per_class = K == 20 ? 100 : 500;
    const std::vector<int> pool = stratified_sample(train, K, protocol == "2k" ? n_per_class : -1, 20261029u);
    std::vector<int> tr, dv;
    stratified_dev(train, pool, K, protocol == "2k" ? 0.2 : 0.1, 20261029u, tr, dv);
    std::printf("DATA arm=%s protocol=%s classes=%d pool=%zu train=%zu dev=%zu feat=%s model=%s learn_enc=%d\n", arm.c_str(),
                protocol.c_str(), K, pool.size(), tr.size(), dv.size(), sp.feat.c_str(), sp.model.c_str(), sp.learn_enc ? 1 : 0);
    std::fflush(stdout);

    // Grid on dev.
    std::vector<Setting> grid;
    const int max_passes = sp.model == "cypha" ? 6 : 10;
    for (int k : sp.ks) {
        if (sp.model == "cypha") {
            for (double dl : {0.003, 0.01, 0.03})
                for (double wl : {0.002, 0.008}) {
                    Setting s;
                    s.world_lr = wl;
                    s.delta_lr = dl;
                    s.k = k;
                    grid.push_back(s);
                }
        } else if (sp.model == "lr") {
            for (double lam : {0.0, 1e-6, 1e-5}) {
                Setting s;
                s.lambda = lam;
                s.k = k;
                grid.push_back(s);
            }
        } else {
            Setting s;
            s.k = k;
            grid.push_back(s);
        }
    }
    Setting best;
    double best_acc = -1;
    int last_k = -1;
    Featurizer fz;
    fz.wv = &wv;
    fz.kind = sp.feat;
    for (const Setting& g : grid) {
        if (g.k != last_k) {
            fz.k = g.k;
            const auto tf = Clock::now();
            fit_on(fz, train, tr);
            std::printf("FIT feat=%s k=%d on %zu rows in %.1f s\n", sp.feat.c_str(), g.k, tr.size(), secs(tf));
            last_k = g.k;
        }
        const auto per_pass = train_eval(sp.model, sp.learn_enc, fz, train, tr, train, dv, K, g, max_passes, seed);
        std::printf("DEV %s :", setting_str(sp.model, g).c_str());
        for (std::size_t p = 0; p < per_pass.size(); ++p) {
            const double acc = accuracy(per_pass[p]);
            std::printf(" %.4f", acc);
            if (acc > best_acc + 1e-12) {
                best_acc = acc;
                best = g;
                best.passes = static_cast<int>(p) + 1;
            }
        }
        std::printf("  (train %.1f s)\n", per_pass.back().train_s);
        std::fflush(stdout);
    }
    std::printf("CHOICE arm=%s %s dev_acc=%.4f\n", arm.c_str(), setting_str(sp.model, best).c_str(), best_acc);
    std::fflush(stdout);
    if (!read_test) return 0;

    // Refit on the whole pool and score the test file once.
    const auto test = read_tsv(test_path);
    std::vector<int> te(test.size());
    std::iota(te.begin(), te.end(), 0);
    fz.k = best.k;
    fit_on(fz, train, pool);
    if (refit_seed == 0) refit_seed = seed;
    const auto res = train_eval(sp.model, sp.learn_enc, fz, train, pool, test, te, K, best, best.passes, refit_seed);
    const Scores& sc = res.back();
    const double acc = accuracy(sc);
    double lo, hi;
    wilson(acc * sc.gold.size(), static_cast<double>(sc.gold.size()), lo, hi);
    std::printf("RESULT arm=%s protocol=%s seed=%u refit_seed=%u n=%zu acc=%.4f [%.4f, %.4f] macro_f1=%.4f ece=%.4f brier=%.4f "
                "train_s=%.1f us_per_doc=%.0f %s\n",
                arm.c_str(), protocol.c_str(), seed, refit_seed, sc.gold.size(), acc, lo, hi, macro_f1(sc, K), ece(sc), mean_of(sc.brier),
                sc.train_s, sc.us_per_doc, setting_str(sp.model, best).c_str());
    if (!out_prefix.empty()) {
        std::ofstream o(out_prefix + ".preds");
        for (std::size_t i = 0; i < sc.gold.size(); ++i)
            o << i << '\t' << sc.gold[i] << '\t' << sc.pred[i] << '\t' << sc.conf[i] << '\t' << sc.brier[i] << '\n';
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ compare
struct Preds {
    std::vector<int> ok;
    std::vector<double> conf, brier;
};

Preds read_preds(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    Preds p;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        long idx;
        int g, pr;
        double c, b;
        if (!(ss >> idx >> g >> pr >> c >> b)) continue;
        p.ok.push_back(g == pr ? 1 : 0);
        p.conf.push_back(c);
        p.brier.push_back(b);
    }
    return p;
}

int compare(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: compare A.preds B.preds [--eps E] [--draws N] [--seed S]\n");
        return 2;
    }
    const Preds A = read_preds(argv[2]), B = read_preds(argv[3]);
    double eps = 0.005;
    int draws = 4000;
    std::uint32_t seed = 1;
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--eps") eps = std::atof(argv[i + 1]);
        else if (a == "--draws") draws = std::atoi(argv[i + 1]);
        else if (a == "--seed") seed = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    }
    if (A.ok.size() != B.ok.size() || A.ok.empty()) {
        std::fprintf(stderr, "prediction files differ in length (%zu vs %zu)\n", A.ok.size(), B.ok.size());
        return 2;
    }
    const std::size_t n = A.ok.size();
    std::mt19937_64 rng(seed);
    std::exponential_distribution<double> ex(1.0);
    std::vector<double> deltas;
    deltas.reserve(draws);
    int gt_eps = 0, gt_m1 = 0;
    for (int d = 0; d < draws; ++d) {
        double sw = 0, s = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double w = ex(rng);
            sw += w;
            s += w * (A.ok[i] - B.ok[i]);
        }
        const double delta = s / sw;
        deltas.push_back(delta);
        gt_eps += delta > eps;
        gt_m1 += delta > -0.01;
    }
    std::sort(deltas.begin(), deltas.end());
    double acc_a = 0, acc_b = 0;
    for (std::size_t i = 0; i < n; ++i) {
        acc_a += A.ok[i];
        acc_b += B.ok[i];
    }
    std::printf("COMPARE n=%zu accA=%.4f accB=%.4f delta=%.4f [%.4f, %.4f] P(delta>%.3f)=%.4f P(delta>-0.01)=%.4f "
                "brierA=%.4f brierB=%.4f\n",
                n, acc_a / n, acc_b / n, (acc_a - acc_b) / n, deltas[static_cast<std::size_t>(0.025 * draws)],
                deltas[static_cast<std::size_t>(0.975 * draws) - 1], eps, static_cast<double>(gt_eps) / draws,
                static_cast<double>(gt_m1) / draws, mean_of(A.brier), mean_of(B.brier));
    return 0;
}

// ------------------------------------------------------------------------------------------------ selftest
int g_fail = 0, g_checks = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(c)) {                                                           \
            ++g_fail;                                                         \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); \
        }                                                                     \
    } while (0)

int selftest() {
    // MurmurHash3 x86_32 known values.
    CHECK(murmur3("", 0) == 0u);
    CHECK(murmur3("", 1) == 0x514E28B7u);
    CHECK(murmur3("hello", 0) == 0x248bfa47u);
    CHECK(murmur3("Hello, world!", 1234) == 0xfaf6cdb3u);
    // Router featuriser: unit length, unigrams + bigrams counted.
    {
        const auto v = router_hash("a b a", 512);
        double s = 0, nz = 0;
        for (double x : v) {
            s += x * x;
            nz += x > 0;
        }
        CHECK(std::fabs(s - 1.0) < 1e-12);
        CHECK(nz >= 2 && nz <= 4);  // "a"(2), "b", "a b", "b a" (collisions allowed)
    }
    // Tokenisers.
    {
        const auto t = alnum_tokens("Hello, World 42-x!");
        CHECK(t.size() == 4 && t[0] == "hello" && t[2] == "42" && t[3] == "x");
        const auto v = vec_tokens("Don't stop-now 3rd");
        CHECK(v.size() == 3 && v[0] == "don't" && v[1] == "stop-now" && v[2] == "<num>");
    }
    // TF-IDF rows are L2 normalised; a word in every document has the minimum idf.
    {
        std::vector<std::string> docs = {"apple banana", "apple cherry", "apple apple date"};
        std::vector<const std::string*> p;
        for (auto& d : docs) p.push_back(&d);
        TfIdf t;
        t.fit(p);
        const auto r = t.row(docs[2]);
        double s = 0;
        for (float x : r.val) s += static_cast<double>(x) * x;
        CHECK(std::fabs(s - 1.0) < 1e-6);
        const std::uint32_t mask = (1u << kTBits) - 1u;
        CHECK(std::fabs(t.idf[murmur3("u apple", 0) & mask] - 1.0f) < 1e-6);
    }
    // Wilson interval contains p and is inside [0, 1].
    {
        double lo, hi;
        wilson(55, 100, lo, hi);
        CHECK(lo < 0.55 && hi > 0.55 && lo > 0.44 && hi < 0.65);
    }
    // Stratified sampling: exact per-class counts; dev split per class.
    {
        std::vector<Doc> d;
        for (int i = 0; i < 300; ++i) d.push_back({i % 3, "x"});
        const auto s = stratified_sample(d, 3, 40, 7);
        CHECK(s.size() == 120);
        std::vector<int> tr, dv;
        stratified_dev(d, s, 3, 0.2, 7, tr, dv);
        CHECK(dv.size() == 24 && tr.size() == 96);
        int c0 = 0;
        for (int i : dv) c0 += d[i].label == 0;
        CHECK(c0 == 8);
    }
    // Logistic regression separates two easy classes.
    {
        LogReg m;
        m.init(2, 4, 0.0);
        SparseRow a, b;
        a.idx = {0};
        a.val = {1.f};
        b.idx = {1};
        b.val = {1.f};
        for (int e = 0; e < 20; ++e) {
            m.step(a, 0);
            m.step(b, 1);
        }
        const auto la = m.logits(a), lb = m.logits(b);
        CHECK(la[0] > la[1] && lb[1] > lb[0]);
    }
    // Brier of a perfect one-hot is 0; of uniform over K is (K-1)/K.
    {
        Scores s;
        score_row(s, {50.0, 0.0, 0.0}, 0);
        CHECK(s.brier[0] < 1e-12 && s.pred[0] == 0);
        score_row(s, {0.0, 0.0, 0.0, 0.0}, 2);
        CHECK(std::fabs(s.brier[1] - 0.75) < 1e-12);
    }
    std::printf("topic_bench selftest: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "run") return run(argc, argv);
    if (argc >= 2 && std::string(argv[1]) == "compare") return compare(argc, argv);
    if (argc >= 2 && std::string(argv[1]) == "selftest") return selftest();
    std::fprintf(stderr,
                 "usage: topic_bench run --train TRAIN.tsv --test TEST.tsv --arm ARM --protocol 2k|full [--vectors V --vocab V]\n"
                 "                       [--seed S] [--read-test] [--out PREFIX]\n"
                 "       topic_bench compare A.preds B.preds [--eps 0.005] [--draws 4000]\n"
                 "       topic_bench selftest\n");
    return 2;
}
