// cypha_gate: the Core Cypha classifier as CyphaLM's abstain gate (TOOL_PROFILE_PLAN P1; DISCOVERY §8a).
//
// CyphaLM writes every answer byte. This tool only decides, from CyphaLM's own confidence features, whether
// an answer it already wrote is kept or replaced by "I don't know" (the gate that "counts as CyphaLM for now").
// It reads the per-question rows of `cyphalm_trie qa --out`: one row per (question, prompt form, model) with
// the grade (0 right, 1 abstained, 2 wrong) and the feature columns.
//
// - Rows are filtered by --where (column=value pairs, e.g. form=chat,lm=trie_ig) and split by the `set` column.
// - Dev: 5-fold cross-validation with fold = row mod 5 (as cyphalm_trie qa's threshold gate). On the four
//   training folds the features are z-scored, a fresh Cypha classifier (W_enc frozen) learns "right" vs
//   "wrong" online over --passes shuffled passes, and the gate statistic is its LLR(right) - LLR(wrong).
//   The threshold is the one that keeps the most right answers with wrong <= 2% of the training rows (ties:
//   fewer wrong, then fewer answered); it is applied to the held-out fold. Abstained rows stay abstained.
// - Test: the gate learnt on all of dev, applied once.
// With no right answer in the training rows the gate answers nothing.
//
// Usage: cypha_gate <rows.tsv> --features mean_lp,min_lp,entropy0,ig_n0,question_lp [--where form=chat,lm=trie_ig]
//                   [--passes 4] [--seed 1]
//        cypha_gate --selftest      (synthetic rows; CTest native_cypha_gate_selftest)
// Prints CV and TEST lines: n, right, abstained, wrong, coverage, wrong rate and its Wilson upper bound.

#include "cypha/create_model.hpp"
#include "cypha/cypha.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

struct Row {
    std::string set;
    int row = 0, grade = 0;
    std::vector<double> x;
};

double wilson_hi(double k, double n) {
    if (n <= 0) return 0;
    const double z = 1.96, p = k / n, d = 1 + z * z / n;
    return (p + z * z / (2 * n) + z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d;
}

// A fresh classifier, trained on rows `tr`; returns the gate statistic for every row of `all`.
std::vector<double> fit_score(const std::vector<const Row*>& tr, const std::vector<const Row*>& all, int passes,
                              unsigned seed) {
    const std::size_t d = tr.empty() ? 0 : tr[0]->x.size();
    std::vector<double> mu(d, 0), sd(d, 0);
    for (auto* r : tr)
        for (std::size_t j = 0; j < d; ++j) mu[j] += r->x[j] / tr.size();
    for (auto* r : tr)
        for (std::size_t j = 0; j < d; ++j) sd[j] += (r->x[j] - mu[j]) * (r->x[j] - mu[j]) / tr.size();
    for (auto& s : sd) s = s > 1e-12 ? std::sqrt(s) : 1;
    auto z = [&](const Row* r) {
        std::vector<double> v(d);
        for (std::size_t j = 0; j < d; ++j) v[j] = (r->x[j] - mu[j]) / sd[j];
        return v;
    };
    const auto path = std::filesystem::temp_directory_path() / ("cypha_gate_" + std::to_string(::getpid()) + ".cypha");
    cypha::FreshModelParams fp;
    fp.input_dim = static_cast<int>(d);
    fp.field_dim = 16;
    fp.world_lr = 0.008;
    fp.delta_lr = 0.01;
    cypha::save_cypha_file(path.c_str(), cypha::create_fresh_model_root(fp));
    cypha::Cypha m;
    const bool ok = m.load(path.string());
    std::filesystem::remove(path);
    if (!ok) throw std::runtime_error("Cypha::load failed");
    m.train_params().enc_lr = 0.0;
    std::vector<const Row*> order = tr;
    std::mt19937 rng(seed);
    const std::string lab[2] = {"right", "wrong"};
    for (int p = 0; p < passes; ++p) {
        std::shuffle(order.begin(), order.end(), rng);
        for (auto* r : order) {
            const auto v = z(r);
            m.update(v.data(), static_cast<int>(d), &lab[r->grade == 0 ? 0 : 1], nullptr);
        }
    }
    std::vector<double> out;
    for (auto* r : all) {
        const auto v = z(r);
        const auto po = m.predict(v.data(), static_cast<int>(d));
        double sr = -1e9, sw = -1e9;
        for (std::size_t k = 0; k < po.labels.size(); ++k)
            (po.labels[k] == "right" ? sr : sw) = po.all_scores[k];
        out.push_back(sr - sw);
    }
    return out;
}

// Threshold on (score, grade) of the training rows: most right with wrong <= 2% of n.
double choose_tau(std::vector<std::pair<double, int>> s, std::size_t n) {
    std::sort(s.begin(), s.end(), [](auto& a, auto& b) { return a.first > b.first; });
    double tau = 1e300;
    int r = 0, w = 0, br = 0, bw = 0;
    for (std::size_t k = 0; k < s.size(); ++k) {
        (s[k].second == 0 ? r : w)++;
        if (k + 1 < s.size() && s[k + 1].first == s[k].first) continue;
        if (w > 0.02 * n) break;
        if (r > br || (r == br && r > 0 && w < bw)) br = r, bw = w, tau = s[k].first;
    }
    return tau;
}

struct Tally {
    int n = 0, right = 0, abst = 0, wrong = 0;
    void add(int g) { ++n, (g == 0 ? right : g == 1 ? abst : wrong)++; }
    void print(const char* tag) const {
        std::printf("%s n %d right %d abstained %d wrong %d coverage %.4f wrong_rate %.4f wrong_rate_wilson_hi %.4f\n", tag, n,
                    right, abst, wrong, n ? double(right + wrong) / n : 0.0, n ? double(wrong) / n : 0.0, wilson_hi(wrong, n));
    }
};

// Trains on `train` (dev rows of the training folds) and grades `apply`; abstained rows stay abstained.
void gate(const std::vector<const Row*>& train, const std::vector<const Row*>& apply, int passes, unsigned seed, Tally& t) {
    std::vector<const Row*> tr;
    for (auto* r : train)
        if (r->grade != 1) tr.push_back(r);
    const bool any_right = std::any_of(tr.begin(), tr.end(), [](const Row* r) { return r->grade == 0; });
    std::vector<double> sa(apply.size(), -1e300);
    double tau = 1e300;
    if (any_right && tr.size() > 1) {
        std::vector<const Row*> both = tr;
        both.insert(both.end(), apply.begin(), apply.end());
        const auto sc = fit_score(tr, both, passes, seed);
        std::vector<std::pair<double, int>> s;
        for (std::size_t k = 0; k < tr.size(); ++k) s.push_back({sc[k], tr[k]->grade});
        tau = choose_tau(s, train.size());
        std::copy(sc.begin() + static_cast<std::ptrdiff_t>(tr.size()), sc.end(), sa.begin());
    }
    for (std::size_t k = 0; k < apply.size(); ++k) t.add(apply[k]->grade != 1 && sa[k] >= tau ? apply[k]->grade : 1);
}

Tally run(const std::vector<Row>& rows, int passes, unsigned seed) {
    std::vector<const Row*> dev, test;
    for (const auto& r : rows) (r.set == "test" ? test : dev).push_back(&r);
    Tally cv;
    for (int f = 0; f < 5; ++f) {
        std::vector<const Row*> tr, ap;
        for (auto* r : dev) (r->row % 5 == f ? ap : tr).push_back(r);
        gate(tr, ap, passes, seed + f, cv);
    }
    cv.print("CV");
    if (!test.empty()) {
        Tally t;
        gate(dev, test, passes, seed, t);
        t.print("TEST");
    }
    return cv;
}

int selftest() {
    std::mt19937 rng(7);
    std::normal_distribution<double> N(0, 1);
    std::uniform_real_distribution<double> U(0, 1);
    std::vector<Row> rows;
    for (int i = 0; i < 400; ++i) {
        const double u = U(rng);
        const int g = u < 0.3 ? 0 : u < 0.4 ? 1 : 2;  // 30% right, 10% abstained, 60% wrong
        rows.push_back({i < 300 ? "dev" : "test", i, g, {(g == 0 ? 2.5 : -1.0) + N(rng), 0.5 * N(rng)}});
    }
    const Tally cv = run(rows, 4, 1);
    const double cov = double(cv.right + cv.wrong) / cv.n, wr = double(cv.wrong) / cv.n;
    const bool ok = cov > 0.10 && wr <= 0.04;
    std::printf("selftest %s (CV coverage %.3f > 0.10, wrong rate %.3f <= 0.04)\n", ok ? "PASS" : "FAIL", cov, wr);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--selftest") return selftest();
    if (argc < 4) return std::fprintf(stderr, "usage: cypha_gate <rows.tsv> --features a,b,... [--where k=v,...] [--passes N]\n"), 2;
    std::vector<std::string> feats;
    std::map<std::string, std::string> where;
    int passes = 4;
    unsigned seed = 1;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        std::stringstream ss(v);
        std::string item;
        if (k == "--features")
            while (std::getline(ss, item, ',')) feats.push_back(item);
        else if (k == "--where")
            while (std::getline(ss, item, ',')) where[item.substr(0, item.find('='))] = item.substr(item.find('=') + 1);
        else if (k == "--passes") passes = std::atoi(v.c_str());
        else if (k == "--seed") seed = static_cast<unsigned>(std::atoi(v.c_str()));
    }
    std::ifstream in(argv[1]);
    std::string line;
    std::getline(in, line);
    std::map<std::string, std::size_t> col;
    {
        std::stringstream ss(line);
        std::string c;
        for (std::size_t j = 0; std::getline(ss, c, '\t'); ++j) col[c] = j;
    }
    for (const auto& f : feats)
        if (!col.count(f)) return std::fprintf(stderr, "no column %s\n", f.c_str()), 1;
    std::vector<Row> rows;
    while (std::getline(in, line)) {
        std::vector<std::string> f;
        std::stringstream ss(line);
        for (std::string c; std::getline(ss, c, '\t');) f.push_back(c);
        bool keep = true;
        for (auto& [k, v] : where) keep = keep && col.count(k) && f[col[k]] == v;
        if (!keep) continue;
        Row r{f[col["set"]], std::atoi(f[col["row"]].c_str()), std::atoi(f[col["grade"]].c_str()), {}};
        for (const auto& name : feats) r.x.push_back(std::atof(f[col[name]].c_str()));
        rows.push_back(std::move(r));
    }
    std::printf("rows %zu features %zu\n", rows.size(), feats.size());
    run(rows, passes, seed);
    return 0;
}
