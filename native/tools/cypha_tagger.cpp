// E1: a part-of-speech tagger whose decisions the Cypha classifier makes (docs/reports/E1_CYPHA_TAGGER.md).
//
// The Cypha classifier (cypha::Cypha: DIF world prior + class deltas, diagonal Gaussians, GH gate, OOD flag) scores
// a shared-variance Gaussian log-likelihood ratio, so it is a linear rule over a dense input. Sparse hashed features
// give it little to work with (E4). So each token's input is dense count statistics: for every feature f of the
// token (its form, suffixes, prefix, shape, neighbours, the tags before it), log P(t | f) for the 17 UPOS tags,
// smoothed toward the tag prior; plus 8 scalars and a word vector. On the training split the count features are
// cross-fitted (a sentence's features come from the other four of five folds); dev and test use all of train.
// Every dimension is standardised by its training mean and standard deviation. Training is Cypha::update one token
// at a time (W_enc frozen, identity); tagging is greedy left to right with Cypha::predict (GH path).
//
// Also here, for comparison: counts alone (argmax of the summed log-probabilities, no learning) and exact class
// means with a shared diagonal variance (the ceiling of Cypha's scoring rule, as in E4).
//
// --whiten EPS (amendment 1): the standardised input is whitened by the pooled within-class covariance of the training
// rows (eigenvalues raised by EPS x their mean), so that a shared-variance Gaussian becomes linear discriminant analysis.
//
// Measured 2026-10-29 (UD English EWT v2.18): dev choice alpha 0.5, simvec vectors, --whiten 0.1, delta_lr 0.03,
// world_lr 0.008, 2 passes: dev 91.96%, test 92.06% (sentence bootstrap [91.67, 92.47]); gate 94% not met; the
// averaged perceptron (pos_tagger) scores 94.81%. Without whitening the best dev was 83.76%. The OOD flag fires on every
// token (kOodThreshold is fixed at 3.0 and the GH inflation exceeds it on this 279-dim input). docs/reports/E1_CYPHA_TAGGER.md
//
// Usage:
//   cypha_tagger run --train T.conllu --dev D.conllu [--test X.conllu --read-test] [--alpha A] [--vec simvec|c1|none]
//                    [--simvec FILE] [--c1 VEC.f32 VOCAB.tsv] [--whiten EPS] [--delta-lr X] [--world-lr X] [--passes P] [--seed S]
//                    [--diag] [--wiki TOKENS.txt] [--out PREFIX]
//   cypha_tagger selftest
// With --read-test the test split is tagged once and every read of the report is printed. Without it, test is not
// opened. --out PREFIX writes per-token rows (PREFIX.dev.tsv, PREFIX.test.tsv: gold, predicted, confidence, ood,
// the 17 scores) for calibration and comparison.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "cypha/create_model.hpp"
#include "cypha/cypha.hpp"
#include "cypha/load_cypha.hpp"

namespace {

using u64 = std::uint64_t;
using Clock = std::chrono::steady_clock;
constexpr int K = 17;
const std::array<const char*, K> kTags = {"ADJ", "ADP",   "ADV",  "AUX",   "CCONJ", "DET",  "INTJ", "NOUN", "NUM",
                                          "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"};

int tag_id(const std::string& t) {
    for (int k = 0; k < K; ++k)
        if (t == kTags[static_cast<std::size_t>(k)]) return k;
    return K - 1;  // X
}

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// ---------------------------------------------------------------- data
struct Sent {
    std::vector<std::string> w;
    std::vector<int> t;
    bool q = false;
};

bool read_conllu(const std::string& path, std::vector<Sent>& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line, text;
    Sent cur;
    auto flush = [&] {
        if (!cur.w.empty()) {
            cur.q = !text.empty() && text.back() == '?';
            out.push_back(std::move(cur));
        }
        cur = Sent();
        text.clear();
    };
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) {
            flush();
            continue;
        }
        if (line[0] == '#') {
            if (line.rfind("# text = ", 0) == 0) text = line.substr(9);
            continue;
        }
        std::vector<std::string> fld;
        std::size_t b = 0;
        while (true) {
            const std::size_t e = line.find('\t', b);
            if (e == std::string::npos) {
                fld.push_back(line.substr(b));
                break;
            }
            fld.push_back(line.substr(b, e - b));
            b = e + 1;
        }
        if (fld.size() < 4) continue;
        if (fld[0].find('-') != std::string::npos || fld[0].find('.') != std::string::npos) continue;
        cur.w.push_back(fld[1]);
        cur.t.push_back(tag_id(fld[3]));
    }
    flush();
    return true;
}

// Plain tokens, one per line, blank line between sentences (no gold tags).
bool read_tokens(const std::string& path, std::vector<Sent>& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    Sent cur;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) {
            if (!cur.w.empty()) {
                cur.q = cur.w.back() == "?";
                out.push_back(std::move(cur));
            }
            cur = Sent();
            continue;
        }
        cur.w.push_back(line);
        cur.t.push_back(K - 1);
    }
    if (!cur.w.empty()) out.push_back(std::move(cur));
    return true;
}

// ---------------------------------------------------------------- word helpers (UTF-8 aware affixes)
std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}
std::vector<std::size_t> cp_starts(const std::string& s) {
    std::vector<std::size_t> st;
    for (std::size_t i = 0; i < s.size(); ++i)
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) st.push_back(i);
    return st;
}
std::string suffix_cp(const std::string& s, int k) {
    const auto st = cp_starts(s);
    if (static_cast<int>(st.size()) <= k) return s;
    return s.substr(st[st.size() - static_cast<std::size_t>(k)]);
}
std::string prefix_cp(const std::string& s, int k) {
    const auto st = cp_starts(s);
    if (static_cast<int>(st.size()) <= k) return s;
    return s.substr(0, st[static_cast<std::size_t>(k)]);
}
std::string shape(const std::string& s) {
    std::string r;
    char last = 0;
    int run = 0;
    for (unsigned char c : s) {
        char m;
        if (c >= 'A' && c <= 'Z') m = 'X';
        else if (c >= 'a' && c <= 'z') m = 'x';
        else if (c >= '0' && c <= '9') m = 'd';
        else if (c >= 0x80) m = 'u';
        else m = static_cast<char>(c);
        if (m == last) {
            if (++run > 2) continue;
        } else {
            run = 1;
            last = m;
        }
        r.push_back(m);
    }
    return r;
}
bool has_digit(const std::string& s) {
    for (unsigned char c : s)
        if (c >= '0' && c <= '9') return true;
    return false;
}
bool init_cap(const std::string& s) { return !s.empty() && s[0] >= 'A' && s[0] <= 'Z'; }
bool all_caps(const std::string& s) {
    bool any = false;
    for (unsigned char c : s) {
        if (c >= 'a' && c <= 'z') return false;
        if (c >= 'A' && c <= 'Z') any = true;
    }
    return any;
}

u64 fnv(const std::string& s, u64 h = 1469598103934665603ULL) {
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}
u64 key(int block, const std::string& s) { return fnv(s, 1469598103934665603ULL ^ (static_cast<u64>(block + 1) * 0x9E3779B97F4A7C15ULL)); }

// ---------------------------------------------------------------- count features
constexpr int kBlocks = 14;  // B1..B14
constexpr int kScalars = 8;
const char* kStart = "<s>";
const char* kEnd = "</s>";

// The feature strings of token i, blocks B1..B10, B13, B14 (static); B11 / B12 depend on previous tags.
struct TokFeat {
    std::array<u64, kBlocks> k{};
};

void static_keys(const Sent& s, std::size_t i, TokFeat& f) {
    const std::string& w = s.w[i];
    const std::string lw = lower(w);
    const std::string pw = i > 0 ? lower(s.w[i - 1]) : kStart;
    const std::string nw = i + 1 < s.w.size() ? lower(s.w[i + 1]) : kEnd;
    f.k[0] = key(0, w);
    f.k[1] = key(1, lw);
    f.k[2] = key(2, suffix_cp(lw, 1));
    f.k[3] = key(3, suffix_cp(lw, 2));
    f.k[4] = key(4, suffix_cp(lw, 3));
    f.k[5] = key(5, suffix_cp(lw, 4));
    f.k[6] = key(6, prefix_cp(lw, 2));
    f.k[7] = key(7, shape(w));
    f.k[8] = key(8, pw);
    f.k[9] = key(9, nw);
    f.k[12] = key(12, pw + "\x1f" + lw);
    f.k[13] = key(13, lw + "\x1f" + nw);
}
u64 tag1_key(int t1) { return key(10, std::to_string(t1)); }
u64 tag2_key(int t2, int t1) { return key(11, std::to_string(t2) + "," + std::to_string(t1)); }

using Counts = std::array<std::uint32_t, K>;
using Table = std::unordered_map<u64, Counts>;

struct CountModel {
    Table total;
    std::vector<Table> fold;  // per-fold counts (cross-fitting subtracts them)
    std::array<double, K> prior{};
    std::unordered_map<std::string, std::uint32_t> form_count;
    double alpha = 2.0;

    void build(const std::vector<Sent>& S, int nfold) {
        fold.assign(static_cast<std::size_t>(nfold), Table());
        std::array<double, K> pc{};
        double n = 0;
        for (std::size_t si = 0; si < S.size(); ++si) {
            const Sent& s = S[si];
            Table& F = fold[si % static_cast<std::size_t>(nfold)];
            TokFeat f;
            for (std::size_t i = 0; i < s.w.size(); ++i) {
                static_keys(s, i, f);
                const int t1 = i > 0 ? s.t[i - 1] : K;
                const int t2 = i > 1 ? s.t[i - 2] : K;
                f.k[10] = tag1_key(t1);
                f.k[11] = tag2_key(t2, t1);
                const int t = s.t[i];
                for (int b = 0; b < kBlocks; ++b) {
                    total[f.k[static_cast<std::size_t>(b)]][static_cast<std::size_t>(t)]++;
                    F[f.k[static_cast<std::size_t>(b)]][static_cast<std::size_t>(t)]++;
                }
                pc[static_cast<std::size_t>(t)] += 1;
                n += 1;
                form_count[s.w[i]]++;
            }
        }
        for (int k = 0; k < K; ++k) prior[static_cast<std::size_t>(k)] = (pc[static_cast<std::size_t>(k)] + 1.0) / (n + K);
    }
    // log P(t | f) into out[0..K), from total counts minus fold `ex` (ex < 0: none).
    void logp(u64 kk, int ex, double* out) const {
        Counts c{};
        auto it = total.find(kk);
        if (it != total.end()) c = it->second;
        if (ex >= 0) {
            auto jt = fold[static_cast<std::size_t>(ex)].find(kk);
            if (jt != fold[static_cast<std::size_t>(ex)].end())
                for (int k = 0; k < K; ++k) c[static_cast<std::size_t>(k)] -= jt->second[static_cast<std::size_t>(k)];
        }
        double n = 0;
        for (int k = 0; k < K; ++k) n += c[static_cast<std::size_t>(k)];
        for (int k = 0; k < K; ++k)
            out[k] = std::log((c[static_cast<std::size_t>(k)] + alpha * prior[static_cast<std::size_t>(k)]) / (n + alpha));
    }
    double form_freq(const std::string& w, int ex_fold_count) const {
        (void)ex_fold_count;
        auto it = form_count.find(w);
        return it == form_count.end() ? 0.0 : static_cast<double>(it->second);
    }
};

// ---------------------------------------------------------------- word vectors
struct Vectors {
    int dim = 0;
    std::unordered_map<std::string, std::vector<float>> v;
    bool load_simvec(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (b.size() < 24 || std::memcmp(b.data(), "CYSIMV1", 8) != 0) return false;
        std::size_t p = 8;
        auto u32 = [&](std::uint32_t& x) {
            std::memcpy(&x, b.data() + p, 4);
            p += 4;
        };
        std::uint32_t rows = 0, dims = 0;
        u32(rows);
        u32(dims);
        std::vector<std::string> words(rows);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto len = static_cast<unsigned char>(b[p++]);
            words[r].assign(b.data() + p, len);
            p += len;
        }
        std::vector<float> scale(rows);
        std::memcpy(scale.data(), b.data() + p, 4ULL * rows);
        p += 4ULL * rows;
        dim = static_cast<int>(dims);
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<float> x(dims);
            for (std::uint32_t d = 0; d < dims; ++d) x[d] = static_cast<float>(static_cast<signed char>(b[p + r * dims + d])) * scale[r];
            v[words[r]] = std::move(x);
        }
        return true;
    }
    bool load_c1(const std::string& vec, const std::string& vocab, int keep) {
        std::ifstream fv(vocab);
        if (!fv) return false;
        std::vector<std::string> words;
        std::string line;
        while (std::getline(fv, line)) words.push_back(line.substr(0, line.find('\t')));
        std::ifstream f(vec, std::ios::binary);
        if (!f) return false;
        f.seekg(0, std::ios::end);
        const auto bytes = static_cast<std::size_t>(f.tellg());
        f.seekg(0);
        const std::size_t full = bytes / 4 / words.size();
        dim = keep;
        std::vector<float> row(full);
        for (const auto& w : words) {
            f.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(4 * full));
            double nn = 0;
            for (int d = 0; d < keep; ++d) nn += static_cast<double>(row[static_cast<std::size_t>(d)]) * row[static_cast<std::size_t>(d)];
            nn = nn > 0 ? 1.0 / std::sqrt(nn) : 0.0;
            std::vector<float> x(static_cast<std::size_t>(keep));
            for (int d = 0; d < keep; ++d) x[static_cast<std::size_t>(d)] = static_cast<float>(row[static_cast<std::size_t>(d)] * nn);
            v[w] = std::move(x);
        }
        return true;
    }
};

// ---------------------------------------------------------------- the input vector
struct Featurizer {
    const CountModel* cm = nullptr;
    const Vectors* vec = nullptr;
    int D() const { return kBlocks * K + kScalars + (vec ? vec->dim + 1 : 0); }
    // Static part (all but B11 / B12) of token i; ex = fold to exclude (training) or -1.
    void fill_static(const Sent& s, std::size_t i, int ex, double* x) const {
        TokFeat f;
        static_keys(s, i, f);
        for (int b = 0; b < kBlocks; ++b) {
            if (b == 10 || b == 11) continue;
            cm->logp(f.k[static_cast<std::size_t>(b)], ex, x + b * K);
        }
        double* sc = x + kBlocks * K;
        const std::string& w = s.w[i];
        double fc = cm->form_freq(w, ex);
        sc[0] = std::log1p(fc);
        sc[1] = fc == 0 ? 1.0 : 0.0;
        sc[2] = init_cap(w) ? 1.0 : 0.0;
        sc[3] = all_caps(w) ? 1.0 : 0.0;
        sc[4] = has_digit(w) ? 1.0 : 0.0;
        sc[5] = w.find('-') != std::string::npos ? 1.0 : 0.0;
        sc[6] = i == 0 ? 1.0 : 0.0;
        sc[7] = s.q ? 1.0 : 0.0;
        if (vec) {
            double* vv = sc + kScalars;
            auto it = vec->v.find(lower(w));
            if (it == vec->v.end()) {
                for (int d = 0; d < vec->dim; ++d) vv[d] = 0;
                vv[vec->dim] = 1.0;
            } else {
                for (int d = 0; d < vec->dim; ++d) vv[d] = it->second[static_cast<std::size_t>(d)];
                vv[vec->dim] = 0.0;
            }
        }
    }
    void fill_tags(int t2, int t1, int ex, double* x) const {
        cm->logp(tag1_key(t1), ex, x + 10 * K);
        cm->logp(tag2_key(t2, t1), ex, x + 11 * K);
    }
};

// ---------------------------------------------------------------- the Cypha classifier
struct CyphaClf {
    cypha::Cypha model;
    bool init(int dim, double delta_lr, double world_lr, double ood_sigma, int field_dim) {
        static int counter = 0;
        const auto dir = std::filesystem::temp_directory_path() /
                         ("e1_cypha_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(dir);
        cypha::FreshModelParams fp;
        fp.input_dim = dim;
        fp.field_dim = field_dim;
        fp.temperature = 1.0;
        fp.world_lr = world_lr;
        fp.delta_lr = delta_lr;
        const cypha::CNode root = cypha::create_fresh_model_root(fp);
        const auto path = (dir / "fresh.cypha").string();
        cypha::save_cypha_file(path.c_str(), root);
        const bool ok = model.load(path);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        if (!ok) return false;
        model.world_lr() = world_lr;
        model.delta_lr() = delta_lr;
        model.ood_sigma() = ood_sigma;
        model.train_params().enc_lr = 0.0;  // W_enc frozen (identity)
        return true;
    }
    void learn(const std::vector<double>& x, const std::string& label) {
        cypha::UpdateOpts uo;
        uo.use_gh = true;
        const auto r = model.update(x.data(), static_cast<int>(x.size()), &label, nullptr, uo);
        if (!r.detail.empty()) {
            std::fprintf(stderr, "cypha update: %s\n", r.detail.c_str());
            std::exit(3);
        }
    }
    cypha::PredictOut ask(const std::vector<double>& x) {
        cypha::PredictOpts po;
        po.use_gh = true;
        return model.predict(x.data(), static_cast<int>(x.size()), po);
    }
};

// ---------------------------------------------------------------- evaluation helpers
struct TokOut {
    int gold = 0, pred = 0;
    double conf = 0;   // Cypha's native confidence
    bool ood = false;
    std::array<double, K> score{};  // scores in tag order (Cypha LLRs)
    bool known = false, q = false;
    std::size_t sent = 0;
};

double wilson_lo(double k, double n) {
    if (n <= 0) return 0;
    const double z = 1.959964, p = k / n, d = 1 + z * z / n;
    return (p + z * z / (2 * n) - z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d;
}
double wilson_hi(double k, double n) {
    if (n <= 0) return 0;
    const double z = 1.959964, p = k / n, d = 1 + z * z / n;
    return (p + z * z / (2 * n) + z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d;
}

std::array<double, K> softmax_t(const std::array<double, K>& s, double T) {
    std::array<double, K> p{};
    double m = *std::max_element(s.begin(), s.end());
    double z = 0;
    for (int k = 0; k < K; ++k) z += p[static_cast<std::size_t>(k)] = std::exp((s[static_cast<std::size_t>(k)] - m) / T);
    for (auto& x : p) x /= z;
    return p;
}
double fit_temperature(const std::vector<TokOut>& R) {
    double bestT = 1, best = 1e300;
    for (double lt = -4; lt <= 6.0001; lt += 0.05) {
        const double T = std::exp(lt);
        double nll = 0;
        for (const auto& r : R) nll -= std::log(std::max(1e-300, softmax_t(r.score, T)[static_cast<std::size_t>(r.gold)]));
        if (nll < best) best = nll, bestT = T;
    }
    return bestT;
}
double ece(const std::vector<double>& conf, const std::vector<int>& ok) {
    const int B = 15;
    std::vector<double> sc(B, 0), sa(B, 0), n(B, 0);
    for (std::size_t i = 0; i < conf.size(); ++i) {
        int b = std::min(B - 1, static_cast<int>(conf[i] * B));
        sc[static_cast<std::size_t>(b)] += conf[i];
        sa[static_cast<std::size_t>(b)] += ok[i];
        n[static_cast<std::size_t>(b)] += 1;
    }
    double e = 0;
    for (int b = 0; b < B; ++b)
        if (n[static_cast<std::size_t>(b)] > 0) e += std::fabs(sa[static_cast<std::size_t>(b)] - sc[static_cast<std::size_t>(b)]);
    return e / static_cast<double>(conf.size());
}
double acc_at_coverage(const std::vector<double>& conf, const std::vector<int>& ok, double cov) {
    std::vector<std::size_t> idx(conf.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return conf[a] > conf[b]; });
    const std::size_t m = static_cast<std::size_t>(std::llround(cov * static_cast<double>(conf.size())));
    double c = 0;
    for (std::size_t i = 0; i < m; ++i) c += ok[idx[i]];
    return m ? c / static_cast<double>(m) : 0;
}
// AUROC of score s for the positive class (pos[i] = 1); ties count 1/2.
double auroc(const std::vector<double>& s, const std::vector<int>& pos) {
    std::vector<std::size_t> idx(s.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return s[a] < s[b]; });
    double np = 0, nn = 0, rank_sum = 0;
    std::size_t i = 0;
    while (i < idx.size()) {
        std::size_t j = i;
        while (j < idx.size() && s[idx[j]] == s[idx[i]]) ++j;
        const double r = (static_cast<double>(i) + static_cast<double>(j) + 1) / 2.0;  // 1-based mean rank
        for (std::size_t k = i; k < j; ++k)
            if (pos[idx[k]]) rank_sum += r, np += 1;
            else nn += 1;
        i = j;
    }
    if (np == 0 || nn == 0) return 0.5;
    return (rank_sum - np * (np + 1) / 2) / (np * nn);
}

struct AccRow {
    double all = 0, known = 0, unknown = 0, ques = 0;
    std::size_t n = 0, nk = 0, nu = 0, nq = 0;
    double c = 0, ck = 0, cu = 0, cq = 0;
};
AccRow accuracy(const std::vector<TokOut>& R) {
    AccRow a;
    for (const auto& r : R) {
        const double ok = r.gold == r.pred ? 1.0 : 0.0;
        a.c += ok, a.n++;
        if (r.known) a.ck += ok, a.nk++;
        else a.cu += ok, a.nu++;
        if (r.q) a.cq += ok, a.nq++;
    }
    a.all = a.n ? a.c / static_cast<double>(a.n) : 0;
    a.known = a.nk ? a.ck / static_cast<double>(a.nk) : 0;
    a.unknown = a.nu ? a.cu / static_cast<double>(a.nu) : 0;
    a.ques = a.nq ? a.cq / static_cast<double>(a.nq) : 0;
    return a;
}
// Bootstrap over sentences: 95% interval of token accuracy.
std::pair<double, double> boot_sent(const std::vector<TokOut>& R, std::size_t nsent, int reps, std::uint64_t seed) {
    std::vector<double> c(nsent, 0), n(nsent, 0);
    for (const auto& r : R) c[r.sent] += r.gold == r.pred, n[r.sent] += 1;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> U(0, nsent - 1);
    std::vector<double> acc;
    for (int b = 0; b < reps; ++b) {
        double cc = 0, nn = 0;
        for (std::size_t i = 0; i < nsent; ++i) {
            const std::size_t j = U(rng);
            cc += c[j], nn += n[j];
        }
        acc.push_back(cc / nn);
    }
    std::sort(acc.begin(), acc.end());
    return {acc[static_cast<std::size_t>(0.025 * reps)], acc[static_cast<std::size_t>(0.975 * reps)]};
}

// ---------------------------------------------------------------- tagging with a scorer
struct Standard {
    std::vector<double> mu, inv;  // per-dimension mean and 1 / std
    std::vector<double> W;        // amendment 1: whitening by the pooled within-class covariance (D x D, row-major), or empty
    void apply(std::vector<double>& x) const {
        for (std::size_t d = 0; d < x.size(); ++d) x[d] = (x[d] - mu[d]) * inv[d];
        if (W.empty()) return;
        const std::size_t D = x.size();
        thread_local std::vector<double> y;
        y.assign(D, 0.0);
        for (std::size_t i = 0; i < D; ++i) {
            const double* w = W.data() + i * D;
            double s = 0;
            for (std::size_t j = 0; j < D; ++j) s += w[j] * x[j];
            y[i] = s;
        }
        x.swap(y);
    }
};

// Eigendecomposition of a symmetric matrix (cyclic Jacobi). A is n x n row-major and is destroyed; on return
// evals[i] and the column i of V (row-major n x n) are the i-th eigenpair.
void jacobi_eigen(std::vector<double>& A, int n, std::vector<double>& evals, std::vector<double>& V) {
    V.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) V[static_cast<std::size_t>(i * n + i)] = 1.0;
    auto a = [&](int i, int j) -> double& { return A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]; };
    auto v = [&](int i, int j) -> double& { return V[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]; };
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0, diag = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) (i == j ? diag : off) += a(i, j) * a(i, j);
        if (off <= 1e-22 * std::max(1.0, diag)) break;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q) {
                const double apq = a(p, q);
                if (std::fabs(apq) < 1e-300) continue;
                const double theta = (a(q, q) - a(p, p)) / (2 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < n; ++k) {  // columns p, q
                    const double akp = a(k, p), akq = a(k, q);
                    a(k, p) = c * akp - s * akq;
                    a(k, q) = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {  // rows p, q
                    const double apk = a(p, k), aqk = a(q, k);
                    a(p, k) = c * apk - s * aqk;
                    a(q, k) = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = v(k, p), vkq = v(k, q);
                    v(k, p) = c * vkp - s * vkq;
                    v(k, q) = s * vkp + c * vkq;
                }
            }
    }
    evals.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) evals[static_cast<std::size_t>(i)] = a(i, i);
}

template <class Scorer>
std::vector<TokOut> tag_all(const std::vector<Sent>& S, const Featurizer& F, const Standard& Z, const CountModel& cm,
                            Scorer&& score) {
    std::vector<TokOut> R;
    std::vector<double> x(static_cast<std::size_t>(F.D()));
    for (std::size_t si = 0; si < S.size(); ++si) {
        const Sent& s = S[si];
        int t1 = K, t2 = K;
        for (std::size_t i = 0; i < s.w.size(); ++i) {
            F.fill_static(s, i, -1, x.data());
            F.fill_tags(t2, t1, -1, x.data());
            TokOut o;
            o.gold = s.t[i];
            o.known = cm.form_count.count(s.w[i]) > 0;
            o.q = s.q;
            o.sent = si;
            score(x, Z, o);
            R.push_back(o);
            t2 = t1;
            t1 = o.pred;
        }
    }
    return R;
}

struct Opt {
    std::string train, dev, test, wiki, simvec, c1vec, c1vocab, out, vec = "simvec";
    double alpha = 2.0, delta_lr = 0.01, world_lr = 0.008, ood_sigma = 12.0, whiten = 0.0;
    int passes = 3, seed = 1, field_dim = 16, c1keep = 64;
    bool read_test = false, diag = false;
};

int cmd_run(const Opt& o) {
    std::vector<Sent> tr, dv, te, wk;
    if (!read_conllu(o.train, tr) || !read_conllu(o.dev, dv)) return std::fprintf(stderr, "cannot read train/dev\n"), 1;
    if (o.read_test && !read_conllu(o.test, te)) return std::fprintf(stderr, "cannot read test\n"), 1;
    if (!o.wiki.empty() && !read_tokens(o.wiki, wk)) return std::fprintf(stderr, "cannot read wiki tokens\n"), 1;
    const auto t0 = Clock::now();
    CountModel cm;
    cm.alpha = o.alpha;
    cm.build(tr, 5);
    Vectors V;
    const Vectors* vp = nullptr;
    if (o.vec == "simvec") {
        if (!V.load_simvec(o.simvec)) return std::fprintf(stderr, "cannot load simvec %s\n", o.simvec.c_str()), 1;
        vp = &V;
    } else if (o.vec == "c1") {
        if (!V.load_c1(o.c1vec, o.c1vocab, o.c1keep)) return std::fprintf(stderr, "cannot load c1 vectors\n"), 1;
        vp = &V;
    }
    Featurizer F{&cm, vp};
    const int D = F.D();
    // Training rows: cross-fitted, gold previous tags.
    std::size_t ntr = 0;
    for (const auto& s : tr) ntr += s.w.size();
    std::vector<float> X(ntr * static_cast<std::size_t>(D));
    std::vector<int> Y(ntr);
    std::vector<std::size_t> sent_first(tr.size());
    {
        std::vector<double> x(static_cast<std::size_t>(D));
        std::size_t r = 0;
        for (std::size_t si = 0; si < tr.size(); ++si) {
            sent_first[si] = r;
            const Sent& s = tr[si];
            const int ex = static_cast<int>(si % 5);
            for (std::size_t i = 0; i < s.w.size(); ++i, ++r) {
                F.fill_static(s, i, ex, x.data());
                const int t1 = i > 0 ? s.t[i - 1] : K, t2 = i > 1 ? s.t[i - 2] : K;
                F.fill_tags(t2, t1, ex, x.data());
                for (int d = 0; d < D; ++d) X[r * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)] = static_cast<float>(x[static_cast<std::size_t>(d)]);
                Y[r] = s.t[i];
            }
        }
    }
    Standard Z;
    Z.mu.assign(static_cast<std::size_t>(D), 0);
    Z.inv.assign(static_cast<std::size_t>(D), 0);
    for (std::size_t r = 0; r < ntr; ++r)
        for (int d = 0; d < D; ++d) Z.mu[static_cast<std::size_t>(d)] += X[r * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)];
    for (auto& m : Z.mu) m /= static_cast<double>(ntr);
    {
        std::vector<double> var(static_cast<std::size_t>(D), 0);
        for (std::size_t r = 0; r < ntr; ++r)
            for (int d = 0; d < D; ++d) {
                const double e = X[r * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)] - Z.mu[static_cast<std::size_t>(d)];
                var[static_cast<std::size_t>(d)] += e * e;
            }
        for (int d = 0; d < D; ++d) {
            const double sd = std::sqrt(var[static_cast<std::size_t>(d)] / static_cast<double>(ntr));
            Z.inv[static_cast<std::size_t>(d)] = sd > 1e-9 ? 1.0 / sd : 0.0;
        }
    }
    if (o.whiten > 0) {
        // Amendment 1: pooled within-class covariance of the standardised rows, eigendecomposed; x' = L^-1/2 U^T z.
        const std::size_t Du = static_cast<std::size_t>(D);
        std::vector<double> cm_(static_cast<std::size_t>(K) * Du, 0.0), cnt(K, 0.0), z(Du);
        auto zrow = [&](std::size_t r) {
            for (std::size_t d = 0; d < Du; ++d) z[d] = (X[r * Du + d] - Z.mu[d]) * Z.inv[d];
        };
        for (std::size_t r = 0; r < ntr; ++r) {
            zrow(r);
            cnt[static_cast<std::size_t>(Y[r])] += 1;
            for (std::size_t d = 0; d < Du; ++d) cm_[static_cast<std::size_t>(Y[r]) * Du + d] += z[d];
        }
        for (int k = 0; k < K; ++k)
            for (std::size_t d = 0; d < Du; ++d) cm_[static_cast<std::size_t>(k) * Du + d] /= std::max(1.0, cnt[static_cast<std::size_t>(k)]);
        std::vector<double> S(Du * Du, 0.0);
        for (std::size_t r = 0; r < ntr; ++r) {
            zrow(r);
            const double* m = cm_.data() + static_cast<std::size_t>(Y[r]) * Du;
            for (std::size_t d = 0; d < Du; ++d) z[d] -= m[d];
            for (std::size_t i = 0; i < Du; ++i) {
                const double zi = z[i];
                if (zi == 0) continue;
                double* row = S.data() + i * Du;
                for (std::size_t j = i; j < Du; ++j) row[j] += zi * z[j];
            }
        }
        for (std::size_t i = 0; i < Du; ++i)
            for (std::size_t j = i; j < Du; ++j) {
                S[i * Du + j] /= static_cast<double>(ntr);
                S[j * Du + i] = S[i * Du + j];
            }
        std::vector<double> ev, U;
        jacobi_eigen(S, D, ev, U);
        double mean_ev = 0;
        for (double e : ev) mean_ev += std::max(0.0, e);
        mean_ev /= static_cast<double>(D);
        Z.W.assign(Du * Du, 0.0);
        for (std::size_t i = 0; i < Du; ++i) {
            const double sc = 1.0 / std::sqrt(std::max(0.0, ev[i]) + o.whiten * mean_ev);
            for (std::size_t j = 0; j < Du; ++j) Z.W[i * Du + j] = sc * U[j * Du + i];  // row i = (column i of U) / sqrt(l_i + eps)
        }
        double emin = 1e300, emax = 0;
        for (double e : ev) emin = std::min(emin, e), emax = std::max(emax, e);
        std::printf("whitening: eps %.3f x mean eigenvalue %.4g; eigenvalues %.3g .. %.3g\n", o.whiten, mean_ev, emin, emax);
    }
    const auto t1c = Clock::now();
    std::printf("E1 config: alpha %.2f vec %s D %d whiten %.3f delta_lr %.4f world_lr %.4f passes %d seed %d | train %zu tokens, features %.1f s\n",
                o.alpha, o.vec.c_str(), D, o.whiten, o.delta_lr, o.world_lr, o.passes, o.seed, ntr, secs(t0, t1c));
    std::fflush(stdout);

    auto print_acc = [](const char* nm, const std::vector<TokOut>& R) {
        const AccRow a = accuracy(R);
        std::printf("  %-34s all %6.2f%% (%.0f/%zu)  known %6.2f%%  unknown %6.2f%% (%zu)  questions %6.2f%% (%zu)\n", nm,
                    100 * a.all, a.c, a.n, 100 * a.known, 100 * a.unknown, a.nu, 100 * a.ques, a.nq);
    };

    // Counts alone and exact class means (diagnostics).
    if (o.diag) {
        auto counts_alone = [&](const std::vector<double>& x, const Standard&, TokOut& out) {
            std::array<double, K> s{};
            for (int b = 0; b < kBlocks; ++b)
                for (int k = 0; k < K; ++k) s[static_cast<std::size_t>(k)] += x[static_cast<std::size_t>(b * K + k)];
            out.score = s;
            out.pred = static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin());
        };
        print_acc("dev, counts alone (no learning)", tag_all(dv, F, Z, cm, counts_alone));
        // Exact class means on standardised rows, pooled within-class diagonal variance.
        std::vector<double> mu(static_cast<std::size_t>(K * D), 0), cnt(K, 0), var(static_cast<std::size_t>(D), 0);
        std::vector<double> zr(static_cast<std::size_t>(D));
        auto trow = [&](std::size_t r) {  // the training row as the classifier sees it (standardised, whitened if on)
            for (int d = 0; d < D; ++d) zr[static_cast<std::size_t>(d)] = X[r * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)];
            Z.apply(zr);
        };
        for (std::size_t r = 0; r < ntr; ++r) {
            cnt[static_cast<std::size_t>(Y[r])] += 1;
            trow(r);
            for (int d = 0; d < D; ++d) mu[static_cast<std::size_t>(Y[r] * D + d)] += zr[static_cast<std::size_t>(d)];
        }
        for (int k = 0; k < K; ++k)
            for (int d = 0; d < D; ++d) mu[static_cast<std::size_t>(k * D + d)] /= std::max(1.0, cnt[static_cast<std::size_t>(k)]);
        for (std::size_t r = 0; r < ntr; ++r) {
            trow(r);
            for (int d = 0; d < D; ++d) {
                const double e = zr[static_cast<std::size_t>(d)] - mu[static_cast<std::size_t>(Y[r] * D + d)];
                var[static_cast<std::size_t>(d)] += e * e;
            }
        }
        for (auto& v : var) v = std::max(1e-6, v / static_cast<double>(ntr));
        auto exact = [&](const std::vector<double>& x0, const Standard& z, TokOut& out) {
            std::vector<double> x = x0;
            z.apply(x);
            std::array<double, K> s{};
            for (int k = 0; k < K; ++k) {
                double l = std::log(std::max(1.0, cnt[static_cast<std::size_t>(k)]));
                for (int d = 0; d < D; ++d) {
                    const double e = x[static_cast<std::size_t>(d)] - mu[static_cast<std::size_t>(k * D + d)];
                    l -= e * e / (2 * var[static_cast<std::size_t>(d)]);
                }
                s[static_cast<std::size_t>(k)] = l;
            }
            out.score = s;
            out.pred = static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin());
        };
        print_acc("dev, exact class means (diagnostic)", tag_all(dv, F, Z, cm, exact));
        std::fflush(stdout);
    }

    // The Cypha classifier, online.
    CyphaClf clf;
    if (!clf.init(D, o.delta_lr, o.world_lr, o.ood_sigma, o.field_dim)) return std::fprintf(stderr, "Cypha init failed\n"), 1;
    auto cypha_score = [&](const std::vector<double>& x0, const Standard& z, TokOut& out) {
        std::vector<double> x = x0;
        z.apply(x);
        const auto p = clf.ask(x);
        out.score.fill(-1e30);
        for (std::size_t j = 0; j < p.labels.size() && j < p.all_scores.size(); ++j)
            out.score[static_cast<std::size_t>(tag_id(p.labels[j]))] = p.all_scores[j];
        out.pred = tag_id(p.label);
        out.conf = p.confidence;
        out.ood = p.is_ood;
    };
    std::mt19937_64 rng(static_cast<std::uint64_t>(o.seed));
    std::vector<std::size_t> order(tr.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> x(static_cast<std::size_t>(D));
    double train_s = 0;
    std::vector<TokOut> dev_last;
    for (int pass = 1; pass <= o.passes; ++pass) {
        std::shuffle(order.begin(), order.end(), rng);
        const auto a = Clock::now();
        for (std::size_t si : order) {
            const std::size_t r0 = sent_first[si];
            for (std::size_t i = 0; i < tr[si].w.size(); ++i) {
                const std::size_t r = r0 + i;
                x.assign(static_cast<std::size_t>(D), 0.0);
                for (int d = 0; d < D; ++d) x[static_cast<std::size_t>(d)] = X[r * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)];
                Z.apply(x);
                clf.learn(x, kTags[static_cast<std::size_t>(Y[r])]);
            }
        }
        train_s += secs(a, Clock::now());
        dev_last = tag_all(dv, F, Z, cm, cypha_score);
        char nm[64];
        std::snprintf(nm, sizeof nm, "dev, Cypha, pass %d", pass);
        print_acc(nm, dev_last);
        std::printf("    train %.1f s so far\n", train_s);
        std::fflush(stdout);
    }
    const double T = fit_temperature(dev_last);
    {
        std::vector<double> cn, cc;
        std::vector<int> ok;
        for (const auto& r : dev_last) {
            ok.push_back(r.gold == r.pred);
            cn.push_back(r.conf);
            const auto p = softmax_t(r.score, T);
            cc.push_back(p[static_cast<std::size_t>(r.pred)]);
        }
        double nood = 0;
        for (const auto& r : dev_last) nood += r.ood;
        std::printf("  dev calibration: T %.3f  ECE native %.4f  calibrated %.4f  acc@90%% %.2f%%  acc@80%% %.2f%%  OOD flag rate %.4f\n", T,
                    ece(cn, ok), ece(cc, ok), 100 * acc_at_coverage(cc, ok, 0.9), 100 * acc_at_coverage(cc, ok, 0.8),
                    nood / static_cast<double>(dev_last.size()));
    }
    auto dump = [&](const std::string& path, const std::vector<TokOut>& R) {
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "gold\tpred\tconf\tood\tknown\tq\tsent");
        for (int k = 0; k < K; ++k) std::fprintf(f, "\t%s", kTags[static_cast<std::size_t>(k)]);
        std::fprintf(f, "\n");
        for (const auto& r : R) {
            std::fprintf(f, "%s\t%s\t%.6f\t%d\t%d\t%d\t%zu", kTags[static_cast<std::size_t>(r.gold)], kTags[static_cast<std::size_t>(r.pred)], r.conf,
                         r.ood ? 1 : 0, r.known ? 1 : 0, r.q ? 1 : 0, r.sent);
            for (int k = 0; k < K; ++k) std::fprintf(f, "\t%.5g", r.score[static_cast<std::size_t>(k)]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
    };
    if (!o.out.empty()) dump(o.out + ".dev.tsv", dev_last);
    if (!o.read_test) return 0;

    // ---------------- the one test read
    const auto a = Clock::now();
    const auto R = tag_all(te, F, Z, cm, cypha_score);
    const double tag_s = secs(a, Clock::now());
    if (!o.out.empty()) dump(o.out + ".test.tsv", R);
    std::printf("\nTEST READ (once)\n");
    print_acc("test, Cypha", R);
    const AccRow ar = accuracy(R);
    const auto bs = boot_sent(R, te.size(), 2000, 20261029);
    std::printf("  UPOS %.2f%%  Wilson [%.2f, %.2f]  sentence bootstrap [%.2f, %.2f]  gate 94%%: %s\n", 100 * ar.all, 100 * wilson_lo(ar.c, static_cast<double>(ar.n)),
                100 * wilson_hi(ar.c, static_cast<double>(ar.n)), 100 * bs.first, 100 * bs.second, ar.all >= 0.94 ? "MET" : "NOT MET");
    std::printf("  unknown words: Wilson [%.2f, %.2f]  questions: Wilson [%.2f, %.2f]\n", 100 * wilson_lo(ar.cu, static_cast<double>(ar.nu)),
                100 * wilson_hi(ar.cu, static_cast<double>(ar.nu)), 100 * wilson_lo(ar.cq, static_cast<double>(ar.nq)),
                100 * wilson_hi(ar.cq, static_cast<double>(ar.nq)));
    std::vector<double> cn, cc, err_s, ood_s;
    std::vector<int> ok, err;
    double nood = 0, nood_unk = 0, nunk = 0;
    for (const auto& r : R) {
        ok.push_back(r.gold == r.pred);
        err.push_back(r.gold != r.pred);
        cn.push_back(r.conf);
        const auto p = softmax_t(r.score, T);
        cc.push_back(p[static_cast<std::size_t>(r.pred)]);
        err_s.push_back(1 - p[static_cast<std::size_t>(r.pred)]);
        ood_s.push_back(r.ood ? 1.0 : 0.0);
        nood += r.ood;
        if (!r.known) nunk += 1, nood_unk += r.ood;
    }
    std::printf("  calibration (T %.3f from dev): ECE native %.4f  calibrated %.4f  acc@90%% %.2f%%  acc@80%% %.2f%%\n", T, ece(cn, ok), ece(cc, ok),
                100 * acc_at_coverage(cc, ok, 0.9), 100 * acc_at_coverage(cc, ok, 0.8));
    std::printf("  AUROC for 'tag is wrong': 1 - calibrated confidence %.4f  OOD flag %.4f\n", auroc(err_s, err), auroc(ood_s, err));
    std::printf("  OOD flag rate: test %.4f (%0.f/%zu)  unknown words %.4f (%0.f/%0.f)\n", nood / static_cast<double>(R.size()), nood, R.size(),
                nunk > 0 ? nood_unk / nunk : 0.0, nood_unk, nunk);
    // Random letters of the same length, same sentences.
    {
        std::vector<Sent> rnd = te;
        std::mt19937_64 g(20261029);
        std::uniform_int_distribution<int> L(0, 25);
        for (auto& s : rnd)
            for (auto& w : s.w)
                for (auto& c : w) c = static_cast<char>('a' + L(g));
        const auto RR = tag_all(rnd, F, Z, cm, cypha_score);
        double n = 0;
        for (const auto& r : RR) n += r.ood;
        std::printf("  OOD flag rate: random letters %.4f (%0.f/%zu)\n", n / static_cast<double>(RR.size()), n, RR.size());
    }
    if (!wk.empty()) {
        const auto RW = tag_all(wk, F, Z, cm, cypha_score);
        double n = 0;
        for (const auto& r : RW) n += r.ood;
        std::printf("  OOD flag rate: Wikipedia prose %.4f (%0.f/%zu, %zu sentences)\n", n / static_cast<double>(RW.size()), n, RW.size(), wk.size());
    }
    std::printf("  cost: training %.1f s (%d passes), tagging %.0f tokens/s single thread, count tables %zu keys\n", train_s, o.passes,
                static_cast<double>(R.size()) / tag_s, cm.total.size());
    return 0;
}

int cmd_selftest() {
    int fails = 0;
    auto check = [&](bool c, const char* what) {
        if (!c) std::fprintf(stderr, "FAIL: %s\n", what), ++fails;
    };
    // Affixes and shapes.
    check(suffix_cp("running", 3) == "ing", "suffix 3");
    check(prefix_cp("running", 2) == "ru", "prefix 2");
    check(suffix_cp("caf\xc3\xa9", 1) == "\xc3\xa9", "UTF-8 suffix");
    check(shape("Hello") == "Xxx", "shape Xxx");
    check(shape("1999") == "dd", "shape dd");
    check(shape("e-mail") == "x-xx", "shape x-xx");
    // Counts and cross-fitting: a toy corpus where fold 0 holds the only "zzz" (NOUN).
    std::vector<Sent> S(5);
    for (int i = 0; i < 5; ++i) {
        S[static_cast<std::size_t>(i)].w = {"the", i == 0 ? "zzz" : "dog"};
        S[static_cast<std::size_t>(i)].t = {tag_id("DET"), tag_id("NOUN")};
    }
    CountModel cm;
    cm.alpha = 2.0;
    cm.build(S, 5);
    double lp_all[K], lp_ex[K];
    cm.logp(key(0, "zzz"), -1, lp_all);
    cm.logp(key(0, "zzz"), 0, lp_ex);
    check(lp_all[tag_id("NOUN")] > lp_all[tag_id("DET")], "seen form favours its tag");
    // Excluding fold 0 removes zzz's only count: back to the prior.
    check(std::fabs(lp_ex[tag_id("NOUN")] - std::log(cm.prior[static_cast<std::size_t>(tag_id("NOUN"))])) < 1e-12, "cross-fit leaves prior");
    double sum = 0;
    for (int k = 0; k < K; ++k) sum += std::exp(lp_all[k]);
    check(std::fabs(sum - 1.0) < 1e-9, "log-probabilities normalised");
    // Metrics.
    check(std::fabs(auroc({0.1, 0.2, 0.9, 0.8}, {0, 0, 1, 1}) - 1.0) < 1e-12, "auroc perfect");
    check(std::fabs(auroc({0.5, 0.5}, {0, 1}) - 0.5) < 1e-12, "auroc ties");
    check(std::fabs(ece({1.0, 1.0}, {1, 1})) < 1e-12, "ece perfect");
    check(std::fabs(wilson_lo(50, 100) - 0.4038) < 1e-3, "wilson lo");
    // Jacobi eigendecomposition: [[2,1],[1,2]] has eigenvalues 1 and 3; A = V diag(e) V^T on a random 6 x 6.
    {
        std::vector<double> A = {2, 1, 1, 2}, ev, V;
        jacobi_eigen(A, 2, ev, V);
        std::sort(ev.begin(), ev.end());
        check(std::fabs(ev[0] - 1) < 1e-12 && std::fabs(ev[1] - 3) < 1e-12, "jacobi 2x2");
        const int n = 6;
        std::mt19937_64 g(7);
        std::normal_distribution<double> N01;
        std::vector<double> B(n * n), M(n * n, 0.0);
        for (auto& b : B) b = N01(g);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k) M[static_cast<std::size_t>(i * n + j)] += B[static_cast<std::size_t>(i * n + k)] * B[static_cast<std::size_t>(j * n + k)];
        std::vector<double> A2 = M;
        jacobi_eigen(A2, n, ev, V);
        double err = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double r = 0;
                for (int k = 0; k < n; ++k) r += V[static_cast<std::size_t>(i * n + k)] * ev[static_cast<std::size_t>(k)] * V[static_cast<std::size_t>(j * n + k)];
                err = std::max(err, std::fabs(r - M[static_cast<std::size_t>(i * n + j)]));
            }
        check(err < 1e-9, "jacobi reconstructs A");
    }
    std::array<double, K> s{};
    s[3] = 2.0;
    const auto p = softmax_t(s, 1.0);
    double ps = 0;
    for (double v : p) ps += v;
    check(std::fabs(ps - 1) < 1e-12 && p[3] > p[0], "softmax");
    std::printf("cypha_tagger selftest: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "selftest") return cmd_selftest();
    if (mode != "run") {
        std::fprintf(stderr, "usage: cypha_tagger run --train T --dev D [--test X --read-test] [options] | selftest\n");
        return 2;
    }
    Opt o;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--train") o.train = nx();
        else if (a == "--dev") o.dev = nx();
        else if (a == "--test") o.test = nx();
        else if (a == "--read-test") o.read_test = true;
        else if (a == "--wiki") o.wiki = nx();
        else if (a == "--alpha") o.alpha = std::atof(nx().c_str());
        else if (a == "--vec") o.vec = nx();
        else if (a == "--simvec") o.simvec = nx();
        else if (a == "--c1") o.c1vec = nx(), o.c1vocab = nx();
        else if (a == "--delta-lr") o.delta_lr = std::atof(nx().c_str());
        else if (a == "--world-lr") o.world_lr = std::atof(nx().c_str());
        else if (a == "--ood-sigma") o.ood_sigma = std::atof(nx().c_str());
        else if (a == "--whiten") o.whiten = std::atof(nx().c_str());
        else if (a == "--passes") o.passes = std::atoi(nx().c_str());
        else if (a == "--seed") o.seed = std::atoi(nx().c_str());
        else if (a == "--diag") o.diag = true;
        else if (a == "--out") o.out = nx();
        else return std::fprintf(stderr, "unknown option %s\n", a.c_str()), 2;
    }
    if (o.read_test && o.test.empty()) return std::fprintf(stderr, "--read-test needs --test\n"), 2;
    return cmd_run(o);
}
