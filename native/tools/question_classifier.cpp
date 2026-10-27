// E4: question classifier on TREC (Li & Roth 2002), and the plan's main-risk test: can the Cypha
// classifier (cypha::Cypha, VectorEncoder with W_enc frozen) learn a language task from count-based
// features? Compared against an averaged perceptron (Collins 2002) written here.
//
// What it does, in order:
//   1. Reads the TREC question set: "COARSE:fine question text" per line (6 coarse, 50 fine types).
//   2. Builds three feature families from counts only (no neural net):
//        a  hashed word unigrams + bigrams, TF-IDF (binary tf, smooth idf), L2-normalised
//        b  question cues: wh-word, the next two words, a first noun-like word (head), capitalised-word
//           count, length bucket, first word -- each as hashed one-hots (and the pairs wh+next, wh+head)
//        c  dense SIF average (Arora et al. 2017) of the PPMI-SVD word vectors, first principal
//           component removed, weight a / (a + p(w)), a = 1e-3
//      and any concatenation ("ab", "bc", "abc", ...).
//   3. Trains two model families on each feature set, separately for the 6 coarse and the 50 fine labels:
//        perceptron  averaged multiclass perceptron on the sparse/dense concatenation (epochs chosen on dev)
//        cypha       cypha::Cypha loaded from create_fresh_model_root (identity W_enc, enc_lr = 0 so W_enc
//                    stays frozen), trained online with Cypha::update, scored with Cypha::predict.
//                    Its input must be dense and as wide as the latent, so the sparse families are first
//                    compressed by a fixed signed random projection (4 hashed targets per feature), every
//                    dimension is z-scored on the training rows, and the block is whitened by PCA on the
//                    training rows (shrinkage eps, p components). Nothing here learns through the label.
//   4. Protocol: 90/10 split of the training questions (fixed seed) picks hyperparameters, fits the
//      softmax temperature for calibration, and picks the abstain threshold. Then everything is refitted
//      on all training questions and scored ONCE on the test questions. Test labels never touch any choice.
//   5. Reports per model and feature set: accuracy, macro-F1, ECE (15 bins; native confidence and
//      dev-temperature-scaled), AUROC of confidence for error detection, risk-coverage (accuracy at
//      100/90/80/70/50% coverage, abstaining on the lowest confidence; for Cypha also with its
//      OOD flag abstaining first), AURC, train time, and inference time per question.
//   6. --ask "q1|q2": trains the best (by dev) coarse and fine model of each family on all training
//      data and prints label + confidence, or ABSTAIN when the calibrated confidence is under the dev
//      threshold for 90% coverage or Cypha flags the item as OOD.
//
// Measured 2026-10-27 (TREC: 5,452 train, 500 test; seed 1; one run on a shared 2-core VM; feature
// set picked on the 545-question dev split, test scored once; accuracy %, coarse / fine):
//   averaged perceptron   89.0 / 84.6   (sets a / abc; across sets 77.6-91.6 coarse, 71.8-84.6 fine)
//   Cypha (frozen W_enc)  75.0 / 62.4   (set bc for both; across sets 60.2-75.6 coarse, 52.8-64.0 fine)
//   batch LDA (diagnostic) 84.8 / 74.4  (same whitened dense input as Cypha; bc / abc)
//   Selective accuracy at 90% / 80% coverage (dev-fitted temperature, abstain on lowest confidence):
//     perceptron 93.8 / 97.2 coarse, 90.0 / 93.5 fine;  Cypha 79.6 / 84.0 coarse, 68.0 / 73.8 fine.
//   Cypha's own confidence is far too low (mean 0.29 at 75% accuracy; ECE 0.18-0.44, 0.05-0.10 after temperature scaling) and
//   its OOD flag fired on 0 of 500 test questions in every run and its anomaly score does not track errors
//   (AUROC 0.44-0.61). Train: perceptron 0.01-4.3 s, Cypha 0.7-8.6 s. Inference per question: perceptron
//   <1-33 us, Cypha 24-216 us; featurising one question 12-22 us.
//   Reference (approximate, not re-run): Li & Roth 2002 report 85.1 coarse / 82.6 fine with words only and
//   92.5 / 85.0 with syntactic features, trained on 21,500 questions; SVM bag-of-words on the 5,500 set
//   is reported near 86-87 coarse. Report: docs/reports/E4_QUESTION_CLASSIFIER.md
//
// Usage (build against a separate cypha_core build; the -D flags and include paths are those of the
// cypha_core compile line, see `ninja -t commands`; with CMake use the snippet in the report instead):
//   cmake -S native -B /root/cb_core -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build /root/cb_core --target cypha_core
//   g++ -O3 -DNDEBUG -std=c++20 -msse4.1 -fopenmp -DCYPHA_ACCEL_GPU_MIN_BATCH_ROWS=1 -DCYPHA_HP_GATE24=1
//       -DCYPHA_HP_SLOT_MAX=24 -DCYPHA_LLM_ALGORITHM_HP=1 -DHP_SLOT_MAX=24 -DHP_XSIMD=1
//       -Inative/third_party/hp/include -Inative/third_party/hp/third_party/xsimd/include -Inative/include
//       -Inative/third_party native/tools/question_classifier.cpp /root/cb_core/libcypha_core.a -lpthread
//       -o question_classifier
//   ./question_classifier <train_5500.label> <TREC_10.label> <enwik8.vectors.f32> <enwik8.vocab.tsv>
//        [--sets a,b,c,ab,bc,abc] [--models perceptron,ncm,cypha] [--grain coarse,fine]
//        [--hash-bits 17] [--proj-a 256] [--proj-b 128] [--whiten-p 256] [--whiten-eps 0.1]
//        [--sif-a 0.001] [--epochs 20] [--passes 6] [--dev-frac 0.1] [--seed 1]
//        [--delta-lr 0.003,0.01,0.03] [--world-lr 0.002,0.008] [--gh 0|1|0,1] [--field-dim 16]
//        [--ask "question one|question two"] [--quiet]
//   Prints DATA / SPLIT / DEV / CFG / RESULT lines and a final table; --ask prints, per model and grain,
//   the label with its calibrated confidence, or ABSTAIN.
//   Data (GitHub-hosted copy of the Li & Roth files): raw.githubusercontent.com/xuanduc24905-beep/
//   Trec_classification/main/data/train_5500.label.txt (5,452 questions) and TREC_10.label.txt (500).
//   Vectors: PPMI-SVD from native/tools/ppmi_svd.cpp (50,000 x 300 float32, unit rows) and its vocab.tsv.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

#include "cypha/create_model.hpp"
#include "cypha/cypha.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

bool g_quiet = false;
void note(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void note(const char* fmt, ...) {
    if (g_quiet) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// ------------------------------------------------------------------------------------------------
// Data
// ------------------------------------------------------------------------------------------------

struct Question {
    std::string text;
    std::string coarse;  // "NUM"
    std::string fine;    // "NUM:dist"
};

std::vector<Question> read_trec(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<Question> out;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        const auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        const std::string lab = line.substr(0, sp);
        const auto colon = lab.find(':');
        if (colon == std::string::npos) continue;
        out.push_back({line.substr(sp + 1), lab.substr(0, colon), lab});
    }
    return out;
}

// ------------------------------------------------------------------------------------------------
// Text: tokens, cues
// ------------------------------------------------------------------------------------------------

struct Tok {
    std::string lc;
    bool cap{false};
};

bool is_alnum_c(unsigned char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

// Same convention as ppmi_svd.cpp: lowercase runs of ASCII letters with single internal apostrophes or
// hyphens; a token touching a digit becomes "<num>" (the vectors have no digit tokens).
std::vector<Tok> tokenize(const std::string& s) {
    std::vector<Tok> out;
    const std::size_t n = s.size();
    std::size_t i = 0;
    while (i < n) {
        if (!is_alnum_c(static_cast<unsigned char>(s[i]))) {
            ++i;
            continue;
        }
        std::size_t j = i;
        bool digit = false;
        while (j < n) {
            const unsigned char c = static_cast<unsigned char>(s[j]);
            if (is_alnum_c(c)) {
                if (c >= '0' && c <= '9') digit = true;
                ++j;
            } else if ((c == '\'' || c == '-') && j + 1 < n && is_alnum_c(static_cast<unsigned char>(s[j + 1]))) {
                ++j;
            } else {
                break;
            }
        }
        Tok t;
        t.cap = (s[i] >= 'A' && s[i] <= 'Z');
        if (digit) {
            t.lc = "<num>";
        } else {
            t.lc.assign(s, i, j - i);
            for (char& ch : t.lc) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        out.push_back(std::move(t));
        i = j;
    }
    return out;
}

const std::unordered_set<std::string>& wh_words() {
    static const std::unordered_set<std::string> s = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"};
    return s;
}
const std::unordered_set<std::string>& imperative_words() {
    static const std::unordered_set<std::string> s = {"name", "define", "list", "give", "tell", "describe", "explain", "stand"};
    return s;
}
// Words skipped when looking for the first noun-like word after the wh-word.
const std::unordered_set<std::string>& skip_words() {
    static const std::unordered_set<std::string> s = {
        "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "done", "has", "have", "had",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must", "the", "a", "an", "of", "in",
        "on", "at", "to", "for", "from", "by", "with", "as", "and", "or", "that", "this", "these", "those", "it",
        "its", "s", "there", "he", "she", "they", "you", "i", "we", "his", "her", "their", "your", "my", "our",
        "many", "much", "kind", "type", "sort", "some", "any", "one", "about", "after", "before", "during",
        "into", "than", "then", "not", "no", "so", "if", "also", "most", "more", "very", "n't", "t", "m", "ll", "re", "ve", "d"};
    return s;
}

std::uint64_t fnv1a(const std::string& s, std::uint64_t seed = 1469598103934665603ULL) {
    std::uint64_t h = seed;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    // final mix so low bits are good
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

struct Cues {
    std::vector<std::string> feats;   // names of cue features
    std::string head;                 // first noun-like word (lowercase) or ""
    std::string next;                 // word right after the wh-word or ""
};

Cues extract_cues(const std::vector<Tok>& t) {
    Cues c;
    const int n = static_cast<int>(t.size());
    int wh_pos = -1;
    for (int i = 0; i < std::min(n, 4); ++i) {
        if (wh_words().count(t[i].lc)) {
            wh_pos = i;
            break;
        }
    }
    if (wh_pos < 0 && n > 0 && imperative_words().count(t[0].lc)) wh_pos = 0;
    const std::string wh = wh_pos >= 0 ? t[wh_pos].lc : std::string("none");
    auto at = [&](int i) -> std::string { return (i >= 0 && i < n) ? t[i].lc : std::string("<e>"); };
    const int p0 = wh_pos >= 0 ? wh_pos : -1;
    const std::string n1 = at(p0 + 1), n2 = at(p0 + 2);
    c.next = n1;
    // first noun-like word after the wh-word
    int head_pos = -1;
    for (int i = p0 + 1; i < n; ++i) {
        if (skip_words().count(t[i].lc)) continue;
        head_pos = i;
        break;
    }
    c.head = head_pos >= 0 ? t[head_pos].lc : std::string();
    int ncap = 0;
    for (int i = 1; i < n; ++i) ncap += t[i].cap ? 1 : 0;
    bool has_num = false;
    for (const auto& tk : t) has_num = has_num || tk.lc == "<num>";
    const std::string head = c.head.empty() ? std::string("<none>") : c.head;
    c.feats = {
        "wh=" + wh,
        "wh+n1=" + wh + "|" + n1,
        "wh+n1+n2=" + wh + "|" + n1 + "|" + n2,
        "n1=" + n1,
        "head=" + head,
        "wh+head=" + wh + "|" + head,
        "wh+n1+head=" + wh + "|" + n1 + "|" + head,
        "first=" + at(0),
        "len=" + std::to_string(std::min(n / 3, 8)),
        "ncap=" + std::to_string(std::min(ncap, 3)),
        "headcap=" + std::string(head_pos >= 0 && t[head_pos].cap && head_pos > 0 ? "1" : "0"),
    };
    if (has_num) c.feats.push_back("has_num=1");
    return c;
}

// ------------------------------------------------------------------------------------------------
// Word vectors, SIF
// ------------------------------------------------------------------------------------------------

struct WordVectors {
    int dim{0};
    std::unordered_map<std::string, int> index;
    std::vector<float> vec;       // V x dim
    std::vector<double> prob;     // unigram probability from the vocab counts
    const float* row(int i) const { return vec.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(dim); }
    int find(const std::string& w) const {
        auto it = index.find(w);
        if (it != index.end()) return it->second;
        const auto dash = w.find('-');
        if (dash != std::string::npos) {
            auto it2 = index.find(w.substr(0, dash));
            if (it2 != index.end()) return it2->second;
        }
        return -1;
    }
};

WordVectors load_vectors(const std::string& vec_path, const std::string& vocab_path) {
    WordVectors wv;
    std::ifstream vf(vocab_path);
    if (!vf) {
        std::fprintf(stderr, "cannot open %s\n", vocab_path.c_str());
        std::exit(2);
    }
    std::string line;
    std::vector<double> counts;
    while (std::getline(vf, line)) {
        const auto tab = line.find('\t');
        const std::string w = line.substr(0, tab);
        const double c = tab == std::string::npos ? 1.0 : std::atof(line.c_str() + tab + 1);
        wv.index.emplace(w, static_cast<int>(counts.size()));
        counts.push_back(c);
    }
    std::ifstream in(vec_path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", vec_path.c_str());
        std::exit(2);
    }
    const std::size_t bytes = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    const std::size_t V = counts.size();
    wv.dim = static_cast<int>(bytes / (V * sizeof(float)));
    if (static_cast<std::size_t>(wv.dim) * V * sizeof(float) != bytes) {
        std::fprintf(stderr, "vector file size %zu is not V(%zu) x dim x 4\n", bytes, V);
        std::exit(2);
    }
    wv.vec.resize(V * static_cast<std::size_t>(wv.dim));
    in.read(reinterpret_cast<char*>(wv.vec.data()), static_cast<std::streamsize>(bytes));
    double total = 0;
    for (double c : counts) total += c;
    wv.prob.resize(V);
    for (std::size_t i = 0; i < V; ++i) wv.prob[i] = counts[i] / total;
    return wv;
}

// ------------------------------------------------------------------------------------------------
// Sparse rows, projection, dense preprocessing
// ------------------------------------------------------------------------------------------------

struct SparseRow {
    std::vector<std::uint32_t> idx;
    std::vector<float> val;
};

void l2_normalise(SparseRow& r) {
    double s = 0;
    for (float v : r.val) s += static_cast<double>(v) * v;
    if (s <= 0) return;
    const float inv = static_cast<float>(1.0 / std::sqrt(s));
    for (float& v : r.val) v *= inv;
}

// Fixed signed random projection (count-sketch with 4 targets per feature). Deterministic from the index.
void project_sparse(const SparseRow& r, int P, std::vector<double>& out, std::size_t out_off) {
    for (std::size_t k = 0; k < r.idx.size(); ++k) {
        const std::uint64_t h = fnv1a(std::to_string(r.idx[k]), 0x9e3779b97f4a7c15ULL);
        for (int t = 0; t < 4; ++t) {
            const std::uint64_t g = (h >> (t * 14)) ^ (h * (static_cast<std::uint64_t>(t) + 1) >> 7);
            const std::size_t col = static_cast<std::size_t>(g % static_cast<std::uint64_t>(P));
            const double sign = ((h >> (60 - t)) & 1ULL) ? 1.0 : -1.0;
            out[out_off + col] += sign * 0.5 * r.val[k];
        }
    }
}

// Cyclic Jacobi for a symmetric matrix. A is n x n row-major (destroyed). Eigenvectors returned as rows of Vt.
void jacobi_eigen(std::vector<double>& A, int n, std::vector<double>& evals, std::vector<double>& Vt) {
    Vt.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) Vt[static_cast<std::size_t>(i) * n + i] = 1.0;
    auto a = [&](int i, int j) -> double& { return A[static_cast<std::size_t>(i) * n + j]; };
    for (int sweep = 0; sweep < 40; ++sweep) {
        double off = 0, diag = 0;
        for (int p = 0; p < n; ++p) {
            diag += a(p, p) * a(p, p);
            for (int q = p + 1; q < n; ++q) off += a(p, q) * a(p, q);
        }
        if (off <= 1e-22 * (diag + 1e-300)) break;
        const double thr = sweep < 3 ? 0.2 * off / (static_cast<double>(n) * n) : 0.0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a(p, q);
                if (std::fabs(apq) <= 1e-300 || apq * apq < thr) continue;
                const double theta = (a(q, q) - a(p, p)) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {  // rows p,q
                    const double x = a(p, k), y = a(q, k);
                    a(p, k) = c * x - s * y;
                    a(q, k) = s * x + c * y;
                }
                for (int k = 0; k < n; ++k) {  // columns p,q
                    const double x = a(k, p), y = a(k, q);
                    a(k, p) = c * x - s * y;
                    a(k, q) = s * x + c * y;
                }
                double* vp = &Vt[static_cast<std::size_t>(p) * n];
                double* vq = &Vt[static_cast<std::size_t>(q) * n];
                for (int k = 0; k < n; ++k) {
                    const double x = vp[k], y = vq[k];
                    vp[k] = c * x - s * y;
                    vq[k] = s * x + c * y;
                }
            }
        }
    }
    evals.resize(n);
    for (int i = 0; i < n; ++i) evals[i] = a(i, i);
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int x, int y) { return evals[x] > evals[y]; });
    std::vector<double> e2(n), V2(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i) {
        e2[i] = evals[ord[i]];
        std::memcpy(&V2[static_cast<std::size_t>(i) * n], &Vt[static_cast<std::size_t>(ord[i]) * n], sizeof(double) * n);
    }
    evals = std::move(e2);
    Vt = std::move(V2);
}

// z-score then PCA-whiten (top p components, shrinkage eps on the mean eigenvalue). Fitted on training rows.
struct Whitener {
    int in_dim{0}, out_dim{0};
    std::vector<double> mean, sd;
    std::vector<double> W;  // out_dim x in_dim
    void fit(const std::vector<std::vector<double>>& X, int p, double eps) {
        in_dim = static_cast<int>(X[0].size());
        const std::size_t n = X.size();
        mean.assign(in_dim, 0.0);
        sd.assign(in_dim, 0.0);
        for (const auto& r : X)
            for (int j = 0; j < in_dim; ++j) mean[j] += r[j];
        for (double& m : mean) m /= static_cast<double>(n);
        for (const auto& r : X)
            for (int j = 0; j < in_dim; ++j) sd[j] += (r[j] - mean[j]) * (r[j] - mean[j]);
        for (double& s : sd) s = std::sqrt(s / static_cast<double>(n)) + 1e-8;
        if (p <= 0 || p >= in_dim) p = in_dim;
        out_dim = p;
        std::vector<double> C(static_cast<std::size_t>(in_dim) * in_dim, 0.0);
        std::vector<double> z(in_dim);
        for (const auto& r : X) {
            for (int j = 0; j < in_dim; ++j) z[j] = (r[j] - mean[j]) / sd[j];
            for (int i = 0; i < in_dim; ++i) {
                const double zi = z[i];
                double* row = &C[static_cast<std::size_t>(i) * in_dim];
                for (int j = i; j < in_dim; ++j) row[j] += zi * z[j];
            }
        }
        for (int i = 0; i < in_dim; ++i)
            for (int j = i; j < in_dim; ++j) {
                C[static_cast<std::size_t>(i) * in_dim + j] /= static_cast<double>(n);
                C[static_cast<std::size_t>(j) * in_dim + i] = C[static_cast<std::size_t>(i) * in_dim + j];
            }
        std::vector<double> ev, Vt;
        jacobi_eigen(C, in_dim, ev, Vt);
        double mean_ev = 0;
        for (double e : ev) mean_ev += std::max(e, 0.0);
        mean_ev /= in_dim;
        W.assign(static_cast<std::size_t>(out_dim) * in_dim, 0.0);
        for (int i = 0; i < out_dim; ++i) {
            const double s = 1.0 / std::sqrt(std::max(ev[i], 0.0) + eps * mean_ev + 1e-12);
            for (int j = 0; j < in_dim; ++j) W[static_cast<std::size_t>(i) * in_dim + j] = s * Vt[static_cast<std::size_t>(i) * in_dim + j];
        }
    }
    std::vector<double> apply(const std::vector<double>& x) const {
        std::vector<double> z(in_dim), o(out_dim, 0.0);
        for (int j = 0; j < in_dim; ++j) z[j] = (x[j] - mean[j]) / sd[j];
        for (int i = 0; i < out_dim; ++i) {
            const double* w = &W[static_cast<std::size_t>(i) * in_dim];
            double acc = 0;
            for (int j = 0; j < in_dim; ++j) acc += w[j] * z[j];
            o[i] = acc;
        }
        return o;
    }
};

// ------------------------------------------------------------------------------------------------
// Feature extractor (fitted on a set of training questions)
// ------------------------------------------------------------------------------------------------

struct RawFeatures {
    SparseRow a;                // hashed tf-idf (index in [0, 2^bits))
    SparseRow b;                // hashed cues   (index in [0, 2^cbits))
    std::vector<float> c;       // SIF after PC removal, standardised, scaled by 1/sqrt(dim)
    std::vector<double> c_std;  // SIF standardised (unit variance per dim) for the dense route
};

struct Extractor {
    int bits{17}, cbits{14};
    double sif_a{1e-3};
    const WordVectors* wv{nullptr};
    std::vector<float> idf;       // size 2^bits
    std::vector<double> pc;       // first principal component of SIF vectors
    std::vector<double> c_mean, c_sd;

    static void add_feat(std::unordered_map<std::uint32_t, float>& m, const std::string& f, int bits) {
        m[static_cast<std::uint32_t>(fnv1a(f) & ((1ULL << bits) - 1))] += 1.0f;
    }
    void word_feats(const std::vector<Tok>& t, std::unordered_map<std::uint32_t, float>& m) const {
        for (std::size_t i = 0; i < t.size(); ++i) {
            add_feat(m, "u=" + t[i].lc, bits);
            if (i + 1 < t.size()) add_feat(m, "b=" + t[i].lc + "_" + t[i + 1].lc, bits);
        }
        add_feat(m, "bos=" + (t.empty() ? std::string("<e>") : t[0].lc), bits);
    }
    std::vector<double> sif_raw(const std::vector<Tok>& t) const {
        std::vector<double> v(wv->dim, 0.0);
        int cnt = 0;
        for (const auto& tk : t) {
            if (tk.lc == "<num>") continue;
            const int id = wv->find(tk.lc);
            if (id < 0) continue;
            const double w = sif_a / (sif_a + wv->prob[id]);
            const float* r = wv->row(id);
            for (int j = 0; j < wv->dim; ++j) v[j] += w * r[j];
            ++cnt;
        }
        if (cnt > 0)
            for (double& x : v) x /= cnt;
        return v;
    }

    void fit(const std::vector<Question>& train) {
        const std::size_t N = train.size(), B = 1ULL << bits;
        std::vector<int> df(B, 0);
        std::vector<std::vector<double>> sif;
        sif.reserve(N);
        for (const auto& q : train) {
            const auto t = tokenize(q.text);
            std::unordered_map<std::uint32_t, float> m;
            word_feats(t, m);
            for (const auto& kv : m) df[kv.first] += 1;
            sif.push_back(sif_raw(t));
        }
        idf.assign(B, 0.f);
        for (std::size_t i = 0; i < B; ++i) idf[i] = static_cast<float>(std::log((1.0 + N) / (1.0 + df[i])) + 1.0);
        // first principal component of the SIF matrix (uncentred, as in Arora et al.): power iteration on X'X
        const int d = wv->dim;
        std::vector<double> u(d, 1.0 / std::sqrt(static_cast<double>(d))), nu(d);
        for (int it = 0; it < 60; ++it) {
            std::fill(nu.begin(), nu.end(), 0.0);
            for (const auto& r : sif) {
                double dot = 0;
                for (int j = 0; j < d; ++j) dot += r[j] * u[j];
                for (int j = 0; j < d; ++j) nu[j] += dot * r[j];
            }
            double nrm = 0;
            for (double x : nu) nrm += x * x;
            nrm = std::sqrt(nrm) + 1e-30;
            for (int j = 0; j < d; ++j) u[j] = nu[j] / nrm;
        }
        pc = u;
        c_mean.assign(d, 0.0);
        c_sd.assign(d, 0.0);
        for (auto& r : sif) {
            double dot = 0;
            for (int j = 0; j < d; ++j) dot += r[j] * pc[j];
            for (int j = 0; j < d; ++j) r[j] -= dot * pc[j];
            for (int j = 0; j < d; ++j) c_mean[j] += r[j];
        }
        for (double& m : c_mean) m /= static_cast<double>(N);
        for (const auto& r : sif)
            for (int j = 0; j < d; ++j) c_sd[j] += (r[j] - c_mean[j]) * (r[j] - c_mean[j]);
        for (double& s : c_sd) s = std::sqrt(s / static_cast<double>(N)) + 1e-8;
    }

    RawFeatures featurize(const std::string& text) const {
        RawFeatures f;
        const auto t = tokenize(text);
        {
            std::unordered_map<std::uint32_t, float> m;
            word_feats(t, m);
            std::vector<std::pair<std::uint32_t, float>> v(m.begin(), m.end());
            std::sort(v.begin(), v.end());
            for (const auto& kv : v) {
                f.a.idx.push_back(kv.first);
                f.a.val.push_back(kv.second * idf[kv.first]);
            }
            l2_normalise(f.a);
        }
        {
            const Cues cu = extract_cues(t);
            std::unordered_map<std::uint32_t, float> m;
            for (const auto& s : cu.feats) add_feat(m, s, cbits);
            std::vector<std::pair<std::uint32_t, float>> v(m.begin(), m.end());
            std::sort(v.begin(), v.end());
            for (const auto& kv : v) {
                f.b.idx.push_back(kv.first);
                f.b.val.push_back(kv.second);
            }
            l2_normalise(f.b);
        }
        {
            const int d = wv->dim;
            std::vector<double> r = sif_raw(t);
            double dot = 0;
            for (int j = 0; j < d; ++j) dot += r[j] * pc[j];
            f.c.resize(d);
            f.c_std.resize(d);
            const double sc = 1.0 / std::sqrt(static_cast<double>(d));
            for (int j = 0; j < d; ++j) {
                const double z = (r[j] - dot * pc[j] - c_mean[j]) / c_sd[j];
                f.c_std[j] = z;
                f.c[j] = static_cast<float>(z * sc);
            }
        }
        return f;
    }
};

// ------------------------------------------------------------------------------------------------
// Metrics
// ------------------------------------------------------------------------------------------------

void softmax_T(const std::vector<double>& s, double T, std::vector<double>& p) {
    p.resize(s.size());
    double mx = *std::max_element(s.begin(), s.end());
    double z = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        p[i] = std::exp((s[i] - mx) / T);
        z += p[i];
    }
    for (double& x : p) x /= z;
}

struct Scored {
    std::vector<int> gold, pred;
    std::vector<std::vector<double>> logits;  // n x K in the global label order (-inf for unseen)
    std::vector<double> native_conf;          // library / raw-softmax confidence
    std::vector<char> flagged;                // Cypha OOD flag (empty for perceptron)
    std::vector<double> anomaly;              // Cypha anomaly score (empty for perceptron)
    double train_s{0}, infer_us{0}, feat_us{0};
};

double accuracy(const std::vector<int>& g, const std::vector<int>& p) {
    int ok = 0;
    for (std::size_t i = 0; i < g.size(); ++i) ok += g[i] == p[i];
    return g.empty() ? 0.0 : static_cast<double>(ok) / static_cast<double>(g.size());
}

double macro_f1(const std::vector<int>& g, const std::vector<int>& p, int K) {
    std::vector<int> tp(K, 0), fp(K, 0), fn(K, 0);
    for (std::size_t i = 0; i < g.size(); ++i) {
        if (g[i] == p[i]) tp[g[i]]++;
        else {
            fp[p[i]]++;
            fn[g[i]]++;
        }
    }
    double s = 0;
    int m = 0;
    for (int k = 0; k < K; ++k) {
        if (tp[k] + fp[k] + fn[k] == 0) continue;  // class in neither gold nor pred
        const double f = (2.0 * tp[k]) / (2.0 * tp[k] + fp[k] + fn[k]);
        s += f;
        ++m;
    }
    return m ? s / m : 0.0;
}

double ece(const std::vector<double>& conf, const std::vector<char>& ok, int bins = 15) {
    std::vector<double> sc(bins, 0), sa(bins, 0);
    std::vector<int> n(bins, 0);
    for (std::size_t i = 0; i < conf.size(); ++i) {
        int b = std::min(bins - 1, static_cast<int>(conf[i] * bins));
        n[b]++;
        sc[b] += conf[i];
        sa[b] += ok[i] ? 1.0 : 0.0;
    }
    double e = 0;
    for (int b = 0; b < bins; ++b)
        if (n[b]) e += (static_cast<double>(n[b]) / static_cast<double>(conf.size())) * std::fabs(sa[b] / n[b] - sc[b] / n[b]);
    return e;
}

double nll_at_T(const std::vector<std::vector<double>>& logits, const std::vector<int>& gold, double T) {
    double s = 0;
    std::vector<double> p;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        softmax_T(logits[i], T, p);
        s -= std::log(std::max(p[gold[i]], 1e-12));
    }
    return s / static_cast<double>(logits.size());
}

double fit_temperature(const std::vector<std::vector<double>>& logits, const std::vector<int>& gold) {
    double best_T = 1, best = 1e300;
    for (double lt = -6.0; lt <= 6.0; lt += 0.1) {  // T = e^lt, geometric grid
        const double T = std::exp(lt);
        const double v = nll_at_T(logits, gold, T);
        if (v < best) {
            best = v;
            best_T = T;
        }
    }
    return best_T;
}

double auroc(const std::vector<double>& score, const std::vector<char>& positive) {  // P(score_pos > score_neg)
    std::vector<int> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return score[a] < score[b]; });
    double rank_sum = 0;
    long npos = 0, nneg = 0;
    for (std::size_t i = 0; i < idx.size();) {
        std::size_t j = i;
        while (j < idx.size() && score[idx[j]] == score[idx[i]]) ++j;
        const double avg_rank = 0.5 * (static_cast<double>(i) + static_cast<double>(j - 1)) + 1.0;
        for (std::size_t k = i; k < j; ++k)
            if (positive[idx[k]]) {
                rank_sum += avg_rank;
                ++npos;
            } else {
                ++nneg;
            }
        i = j;
    }
    if (!npos || !nneg) return 0.5;
    return (rank_sum - static_cast<double>(npos) * (static_cast<double>(npos) + 1) / 2.0) / (static_cast<double>(npos) * static_cast<double>(nneg));
}

struct RiskCoverage {
    std::vector<double> acc_at;  // for the coverages in kCov
    double aurc{0};
};
const std::array<double, 5> kCov = {1.0, 0.9, 0.8, 0.7, 0.5};

// rank key: higher = answered first. ok[i] = prediction correct.
RiskCoverage risk_coverage(const std::vector<double>& key, const std::vector<char>& ok) {
    const std::size_t n = key.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return key[a] > key[b]; });
    std::vector<double> cum(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) cum[i + 1] = cum[i] + (ok[idx[i]] ? 1.0 : 0.0);
    RiskCoverage rc;
    for (double c : kCov) {
        const std::size_t k = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(c * static_cast<double>(n) - 1e-9)));
        rc.acc_at.push_back(cum[k] / static_cast<double>(k));
    }
    double a = 0;
    for (std::size_t k = 1; k <= n; ++k) a += 1.0 - cum[k] / static_cast<double>(k);
    rc.aurc = a / static_cast<double>(n);
    return rc;
}

// ------------------------------------------------------------------------------------------------
// Averaged perceptron (Collins 2002), multiclass, sparse input
// ------------------------------------------------------------------------------------------------

struct Perceptron {
    int K{0};
    std::size_t F{0};
    std::vector<float> w, u;  // averaged weights = w - u / c
    double c{1};
    void init(int k, std::size_t f) {
        K = k;
        F = f;
        w.assign(static_cast<std::size_t>(K) * F, 0.f);
        u.assign(static_cast<std::size_t>(K) * F, 0.f);
        c = 1;
    }
    template <class W>
    void scores(const SparseRow& x, const W& wt, std::vector<double>& s) const {
        s.assign(K, 0.0);
        for (int k = 0; k < K; ++k) {
            const float* wk = wt.data() + static_cast<std::size_t>(k) * F;
            double acc = 0;
            for (std::size_t j = 0; j < x.idx.size(); ++j) acc += static_cast<double>(wk[x.idx[j]]) * x.val[j];
            s[k] = acc;
        }
    }
    void epoch(const std::vector<SparseRow>& X, const std::vector<int>& y, const std::vector<int>& order) {
        std::vector<double> s;
        for (int i : order) {
            scores(X[i], w, s);
            int p = 0;
            for (int k = 1; k < K; ++k)
                if (s[k] > s[p]) p = k;
            if (p != y[i]) {
                float* wy = w.data() + static_cast<std::size_t>(y[i]) * F;
                float* wp = w.data() + static_cast<std::size_t>(p) * F;
                float* uy = u.data() + static_cast<std::size_t>(y[i]) * F;
                float* up = u.data() + static_cast<std::size_t>(p) * F;
                const float cc = static_cast<float>(c);
                for (std::size_t j = 0; j < X[i].idx.size(); ++j) {
                    const std::uint32_t f = X[i].idx[j];
                    const float v = X[i].val[j];
                    wy[f] += v;
                    wp[f] -= v;
                    uy[f] += cc * v;
                    up[f] -= cc * v;
                }
            }
            c += 1;
        }
    }
    std::vector<float> averaged() const {
        std::vector<float> a(w.size());
        const float inv = static_cast<float>(1.0 / c);
        for (std::size_t i = 0; i < w.size(); ++i) a[i] = w[i] - u[i] * inv;
        return a;
    }
};

// ------------------------------------------------------------------------------------------------
// Cypha wrapper
// ------------------------------------------------------------------------------------------------

struct CyphaParams {
    double delta_lr{0.01}, world_lr{0.008};
    bool gh{true};
    int passes{3};
    int field_dim{16};
    double ood_sigma{12.0};
    double replay_ratio{0.0};
};

struct CyphaClf {
    cypha::Cypha model;
    int d{0};
    bool gh{true};
    bool init(int dim, const CyphaParams& p) {
        d = dim;
        gh = p.gh;
        static int counter = 0;
        const auto dir = std::filesystem::temp_directory_path() /
                         ("qc_cypha_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(dir);
        cypha::FreshModelParams fp;
        fp.input_dim = dim;
        fp.field_dim = p.field_dim;
        fp.temperature = 1.0;
        fp.world_lr = p.world_lr;
        fp.delta_lr = p.delta_lr;
        const cypha::CNode root = cypha::create_fresh_model_root(fp);
        const auto path = (dir / "fresh.cypha").string();
        cypha::save_cypha_file(path.c_str(), root);
        const bool ok = model.load(path);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        if (!ok) return false;
        model.world_lr() = p.world_lr;
        model.delta_lr() = p.delta_lr;
        model.ood_sigma() = p.ood_sigma;
        model.train_params().enc_lr = 0.0;  // W_enc frozen (identity)
        model.train_params().replay_ratio = p.replay_ratio;
        return true;
    }
    void learn(const std::vector<double>& x, const std::string& label) {
        cypha::UpdateOpts uo;
        uo.use_gh = gh;
        const auto r = model.update(x.data(), static_cast<int>(x.size()), &label, nullptr, uo);
        if (!r.detail.empty()) {
            std::fprintf(stderr, "cypha update: %s\n", r.detail.c_str());
            std::exit(3);
        }
    }
    cypha::PredictOut ask(const std::vector<double>& x) {
        cypha::PredictOpts po;
        po.use_gh = gh;
        return model.predict(x.data(), static_cast<int>(x.size()), po);
    }
};

// ------------------------------------------------------------------------------------------------
// Experiment plumbing
// ------------------------------------------------------------------------------------------------

struct Options {
    std::string train_path, test_path, vec_path, vocab_path;
    std::vector<std::string> sets{"a", "b", "c", "ab", "bc", "abc"};
    std::vector<std::string> models{"perceptron", "cypha"};
    std::vector<std::string> grains{"coarse", "fine"};
    int hash_bits{17}, proj_a{256}, proj_b{128}, whiten_p{256};
    double whiten_eps{0.1}, sif_a{1e-3}, dev_frac{0.1};
    int epochs{20}, passes{6};
    int seed{1};
    std::vector<double> delta_lrs{0.003, 0.01, 0.03};
    std::vector<double> world_lrs{0.002, 0.008};
    std::vector<int> ghs{1};
    int field_dim{16};
    std::string ask;
    std::string dump;  // optional per-item dump of the best configs (path)
};

std::vector<std::string> split(const std::string& s, char sep = ',') {
    std::vector<std::string> o;
    std::string cur;
    for (char c : s) {
        if (c == sep) {
            o.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    o.push_back(cur);
    return o;
}

// One view = an extractor fitted on `fit` plus features for `fit` and `eval`.
struct View {
    std::vector<Question> fit, eval;
    Extractor ex;
    std::vector<RawFeatures> F_fit, F_eval;
    // dense (Cypha) inputs per feature set
    std::unordered_map<std::string, std::vector<std::vector<double>>> dense_fit, dense_eval;
    std::unordered_map<std::string, std::shared_ptr<Whitener>> whiteners;
};

void build_view(View& v, const Options& o, const WordVectors& wv) {
    v.ex.bits = o.hash_bits;
    v.ex.sif_a = o.sif_a;
    v.ex.wv = &wv;
    v.ex.fit(v.fit);
    v.F_fit.reserve(v.fit.size());
    for (const auto& q : v.fit) v.F_fit.push_back(v.ex.featurize(q.text));
    v.F_eval.reserve(v.eval.size());
    for (const auto& q : v.eval) v.F_eval.push_back(v.ex.featurize(q.text));
}

bool has(const std::string& set, char c) { return set.find(c) != std::string::npos; }

// Sparse input for the perceptron
SparseRow perceptron_row(const RawFeatures& f, const std::string& set, const Options& o) {
    SparseRow r;
    const std::uint32_t offA = 0, offB = 1u << o.hash_bits, offC = offB + (1u << 14);
    if (has(set, 'a'))
        for (std::size_t k = 0; k < f.a.idx.size(); ++k) {
            r.idx.push_back(offA + f.a.idx[k]);
            r.val.push_back(f.a.val[k]);
        }
    if (has(set, 'b'))
        for (std::size_t k = 0; k < f.b.idx.size(); ++k) {
            r.idx.push_back(offB + f.b.idx[k]);
            r.val.push_back(f.b.val[k]);
        }
    if (has(set, 'c'))
        for (std::size_t k = 0; k < f.c.size(); ++k) {
            r.idx.push_back(offC + static_cast<std::uint32_t>(k));
            r.val.push_back(f.c[k]);
        }
    r.idx.push_back(offC + static_cast<std::uint32_t>(f.c.size()));  // bias
    r.val.push_back(1.0f);
    return r;
}
std::size_t perceptron_dim(const Options& o, int dense_dim) { return (1u << o.hash_bits) + (1u << 14) + dense_dim + 1; }

// Dense input for Cypha: [c z-scored | projected a | projected b] -> z-score + PCA whiten (fitted on fit rows).
std::vector<double> raw_dense(const RawFeatures& f, const std::string& set, const Options& o) {
    std::vector<double> x;
    const int dc = static_cast<int>(f.c_std.size());
    const int total = (has(set, 'c') ? dc : 0) + (has(set, 'a') ? o.proj_a : 0) + (has(set, 'b') ? o.proj_b : 0);
    x.assign(total, 0.0);
    std::size_t off = 0;
    if (has(set, 'c')) {
        for (int j = 0; j < dc; ++j) x[off + j] = f.c_std[j];
        off += dc;
    }
    if (has(set, 'a')) {
        project_sparse(f.a, o.proj_a, x, off);
        off += o.proj_a;
    }
    if (has(set, 'b')) {
        project_sparse(f.b, o.proj_b, x, off);
        off += o.proj_b;
    }
    return x;
}

void ensure_dense(View& v, const std::string& set, const Options& o) {
    if (v.dense_fit.count(set)) return;
    std::vector<std::vector<double>> raw;
    raw.reserve(v.F_fit.size());
    for (const auto& f : v.F_fit) raw.push_back(raw_dense(f, set, o));
    auto wh = std::make_shared<Whitener>();
    const auto t0 = Clock::now();
    wh->fit(raw, o.whiten_p, o.whiten_eps);
    note("    whitener[%s]: %d -> %d dims in %.1fs\n", set.c_str(), wh->in_dim, wh->out_dim, seconds_since(t0));
    std::vector<std::vector<double>> df, de;
    df.reserve(raw.size());
    for (const auto& r : raw) df.push_back(wh->apply(r));
    de.reserve(v.F_eval.size());
    for (const auto& f : v.F_eval) de.push_back(wh->apply(raw_dense(f, set, o)));
    v.dense_fit[set] = std::move(df);
    v.dense_eval[set] = std::move(de);
    v.whiteners[set] = wh;
}

struct Labels {
    std::vector<std::string> names;
    std::unordered_map<std::string, int> idx;
    int add(const std::string& s) {
        auto it = idx.find(s);
        if (it != idx.end()) return it->second;
        const int k = static_cast<int>(names.size());
        names.push_back(s);
        idx[s] = k;
        return k;
    }
};

std::vector<int> gold_ids(const std::vector<Question>& qs, bool fine, Labels& L) {
    std::vector<int> y;
    for (const auto& q : qs) y.push_back(L.add(fine ? q.fine : q.coarse));
    return y;
}

// ---- model runs: train on view.fit, score view.eval ------------------------------------------------

struct PercRun {
    Scored sc;
    int best_epoch{0};
    std::vector<float> weights;  // averaged, for ask
};

// Trains up to `epochs`; if `pick_epoch` <= 0 the best epoch on the eval set is chosen (dev use), else exactly that many.
PercRun run_perceptron(const View& v, const std::string& set, const Options& o, const std::vector<int>& y_fit,
                       const std::vector<int>& y_eval, int K, int pick_epoch, int seed_off) {
    PercRun out;
    std::vector<SparseRow> Xf, Xe;
    const auto tf = Clock::now();
    for (const auto& f : v.F_fit) Xf.push_back(perceptron_row(f, set, o));
    for (const auto& f : v.F_eval) Xe.push_back(perceptron_row(f, set, o));
    (void)tf;
    Perceptron P;
    P.init(K, perceptron_dim(o, v.ex.wv->dim));
    std::mt19937_64 rng(static_cast<std::uint64_t>(o.seed) * 7919 + seed_off);
    std::vector<int> order(Xf.size());
    std::iota(order.begin(), order.end(), 0);
    const int E = pick_epoch > 0 ? pick_epoch : o.epochs;
    double best_acc = -1;
    std::vector<float> best_w;
    int best_e = 0;
    double train_s = 0;
    for (int e = 1; e <= E; ++e) {
        std::shuffle(order.begin(), order.end(), rng);
        const auto t0 = Clock::now();
        P.epoch(Xf, y_fit, order);
        train_s += seconds_since(t0);
        if (pick_epoch <= 0) {
            const auto avg = P.averaged();
            std::vector<double> s;
            int ok = 0;
            for (std::size_t i = 0; i < Xe.size(); ++i) {
                P.scores(Xe[i], avg, s);
                int p = static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin());
                ok += p == y_eval[i];
            }
            const double acc = static_cast<double>(ok) / static_cast<double>(Xe.size());
            if (acc > best_acc + 1e-12) {
                best_acc = acc;
                best_e = e;
            }
        }
    }
    out.best_epoch = pick_epoch > 0 ? pick_epoch : best_e;
    if (pick_epoch > 0) {
        // retrain exactly pick_epoch epochs was done above; averaged weights are the final ones
    } else {
        // dev mode: re-run to the best epoch so the returned scores correspond to it
        Perceptron P2;
        P2.init(K, perceptron_dim(o, v.ex.wv->dim));
        std::mt19937_64 rng2(static_cast<std::uint64_t>(o.seed) * 7919 + seed_off);
        std::vector<int> ord2(Xf.size());
        std::iota(ord2.begin(), ord2.end(), 0);
        for (int e = 1; e <= best_e; ++e) {
            std::shuffle(ord2.begin(), ord2.end(), rng2);
            P2.epoch(Xf, y_fit, ord2);
        }
        P = std::move(P2);
    }
    out.weights = P.averaged();
    out.sc.train_s = train_s * (pick_epoch > 0 ? 1.0 : static_cast<double>(out.best_epoch) / E);
    const auto t1 = Clock::now();
    std::vector<double> s, p;
    out.sc.gold = y_eval;
    for (std::size_t i = 0; i < Xe.size(); ++i) {
        P.scores(Xe[i], out.weights, s);
        out.sc.logits.push_back(s);
        out.sc.pred.push_back(static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin()));
        softmax_T(s, 1.0, p);
        out.sc.native_conf.push_back(*std::max_element(p.begin(), p.end()));
    }
    out.sc.infer_us = 1e6 * seconds_since(t1) / static_cast<double>(Xe.size());
    return out;
}

struct CyphaRun {
    Scored sc;
    std::vector<double> acc_per_pass;
    int best_pass{0};
    std::unique_ptr<CyphaClf> clf;
    std::vector<std::string> names;
};

// Trains `passes` passes online (shuffled each pass); evaluates on eval after each pass when per_pass.
CyphaRun run_cypha(const View& v, const std::string& set, const Options& o, const CyphaParams& cp,
                   const std::vector<int>& y_fit, const std::vector<int>& y_eval, const Labels& L, int K,
                   bool per_pass, int seed_off) {
    CyphaRun out;
    const auto& Xf = v.dense_fit.at(set);
    const auto& Xe = v.dense_eval.at(set);
    out.clf = std::make_unique<CyphaClf>();
    if (!out.clf->init(static_cast<int>(Xf[0].size()), cp)) {
        std::fprintf(stderr, "Cypha::load failed\n");
        std::exit(3);
    }
    std::mt19937_64 rng(static_cast<std::uint64_t>(o.seed) * 104729 + seed_off);
    std::vector<int> order(Xf.size());
    std::iota(order.begin(), order.end(), 0);
    double train_s = 0;
    auto eval_now = [&](Scored* sc) -> double {
        int ok = 0;
        const auto t1 = Clock::now();
        for (std::size_t i = 0; i < Xe.size(); ++i) {
            const auto po = out.clf->ask(Xe[i]);
            // map to global label order
            std::vector<double> lg(K, -1e30);
            for (std::size_t j = 0; j < po.labels.size(); ++j) {
                auto it = L.idx.find(po.labels[j]);
                if (it != L.idx.end()) lg[it->second] = po.all_scores[j];
            }
            const int p = static_cast<int>(std::max_element(lg.begin(), lg.end()) - lg.begin());
            ok += p == y_eval[i];
            if (sc) {
                sc->logits.push_back(std::move(lg));
                sc->pred.push_back(p);
                sc->native_conf.push_back(po.confidence);
                sc->flagged.push_back(po.is_ood ? 1 : 0);
                sc->anomaly.push_back(po.anomaly_score);
            }
        }
        if (sc) sc->infer_us = 1e6 * seconds_since(t1) / static_cast<double>(Xe.size());
        return static_cast<double>(ok) / static_cast<double>(Xe.size());
    };
    for (int pass = 1; pass <= cp.passes; ++pass) {
        std::shuffle(order.begin(), order.end(), rng);
        const auto t0 = Clock::now();
        for (int i : order) out.clf->learn(Xf[i], L.names[y_fit[i]]);
        train_s += seconds_since(t0);
        if (per_pass && pass < cp.passes) out.acc_per_pass.push_back(eval_now(nullptr));
    }
    out.sc.gold = y_eval;
    const double final_acc = eval_now(&out.sc);
    out.acc_per_pass.push_back(final_acc);
    out.sc.train_s = train_s;
    out.names = L.names;
    return out;
}

// ---- reporting ---------------------------------------------------------------------------------------

struct Report {
    std::string grain, set, model, cfg;
    double acc{0}, f1{0}, ece_native{0}, ece_cal{0}, auroc_conf{0}, aurc{0}, T{1}, mean_native{0}, mean_cal{0};
    std::vector<double> acc_at;       // confidence ranking
    std::vector<double> acc_at_ood;   // Cypha: OOD-flagged abstain first
    double flagged_frac{0}, flagged_acc_rest{0}, auroc_anom{0};
    double train_s{0}, infer_us{0};
    double thr90{0};
};

Report make_report(const std::string& grain, const std::string& set, const std::string& model, const std::string& cfg,
                   const Scored& sc, double T, int K) {
    Report r;
    r.grain = grain;
    r.set = set;
    r.model = model;
    r.cfg = cfg;
    r.T = T;
    r.acc = accuracy(sc.gold, sc.pred);
    r.f1 = macro_f1(sc.gold, sc.pred, K);
    const std::size_t n = sc.gold.size();
    std::vector<char> ok(n);
    for (std::size_t i = 0; i < n; ++i) ok[i] = sc.gold[i] == sc.pred[i];
    std::vector<double> cal(n), p;
    for (std::size_t i = 0; i < n; ++i) {
        softmax_T(sc.logits[i], T, p);
        cal[i] = *std::max_element(p.begin(), p.end());
    }
    r.ece_native = ece(sc.native_conf, ok);
    for (std::size_t i = 0; i < n; ++i) {
        r.mean_native += sc.native_conf[i] / static_cast<double>(n);
        r.mean_cal += cal[i] / static_cast<double>(n);
    }
    r.ece_cal = ece(cal, ok);
    r.auroc_conf = auroc(cal, ok);
    const auto rc = risk_coverage(cal, ok);
    r.acc_at = rc.acc_at;
    r.aurc = rc.aurc;
    if (!sc.flagged.empty()) {
        std::vector<double> key(n);
        int fl = 0, fl_ok_rest = 0, rest = 0;
        for (std::size_t i = 0; i < n; ++i) {
            key[i] = sc.flagged[i] ? cal[i] - 10.0 : cal[i];
            fl += sc.flagged[i];
            if (!sc.flagged[i]) {
                ++rest;
                fl_ok_rest += ok[i];
            }
        }
        r.acc_at_ood = risk_coverage(key, ok).acc_at;
        r.flagged_frac = static_cast<double>(fl) / static_cast<double>(n);
        r.flagged_acc_rest = rest ? static_cast<double>(fl_ok_rest) / rest : 0.0;
        std::vector<double> neg_anom(n);
        for (std::size_t i = 0; i < n; ++i) neg_anom[i] = -sc.anomaly[i];
        r.auroc_anom = auroc(neg_anom, ok);
    }
    r.train_s = sc.train_s;
    r.infer_us = sc.infer_us;
    return r;
}

double quantile_threshold(std::vector<double> conf, double coverage) {  // answer items with conf >= thr
    std::sort(conf.begin(), conf.end(), std::greater<double>());
    const std::size_t k = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(coverage * static_cast<double>(conf.size()) - 1e-9)));
    return conf[k - 1];
}

void print_report(const Report& r) {
    std::printf("RESULT grain=%s set=%s model=%s acc=%.4f f1=%.4f ece_native=%.4f ece_cal=%.4f T=%.3f auroc=%.3f aurc=%.4f mean_conf_native=%.3f mean_conf_cal=%.3f", r.grain.c_str(),
                r.set.c_str(), r.model.c_str(), r.acc, r.f1, r.ece_native, r.ece_cal, r.T, r.auroc_conf, r.aurc, r.mean_native, r.mean_cal);
    for (std::size_t i = 0; i < kCov.size(); ++i) std::printf(" acc@%d=%.4f", static_cast<int>(kCov[i] * 100), r.acc_at[i]);
    if (!r.acc_at_ood.empty()) {
        for (std::size_t i = 0; i < kCov.size(); ++i) std::printf(" oodacc@%d=%.4f", static_cast<int>(kCov[i] * 100), r.acc_at_ood[i]);
        std::printf(" flagged=%.3f acc_unflagged=%.4f auroc_anom=%.3f", r.flagged_frac, r.flagged_acc_rest, r.auroc_anom);
    }
    std::printf(" train_s=%.2f infer_us=%.1f cfg=\"%s\"\n", r.train_s, r.infer_us, r.cfg.c_str());
    std::fflush(stdout);
}

// ---- the closure kept for --ask -----------------------------------------------------------------------

struct AskModel {
    std::string label;  // "perceptron coarse set=abc"
    std::function<void(const std::string&)> run;
};

}  // namespace

int main(int argc, char** argv) {
    Options o;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--sets") o.sets = split(next());
        else if (a == "--models") o.models = split(next());
        else if (a == "--grain") o.grains = split(next());
        else if (a == "--hash-bits") o.hash_bits = std::atoi(next().c_str());
        else if (a == "--proj-a") o.proj_a = std::atoi(next().c_str());
        else if (a == "--proj-b") o.proj_b = std::atoi(next().c_str());
        else if (a == "--whiten-p") o.whiten_p = std::atoi(next().c_str());
        else if (a == "--whiten-eps") o.whiten_eps = std::atof(next().c_str());
        else if (a == "--sif-a") o.sif_a = std::atof(next().c_str());
        else if (a == "--epochs") o.epochs = std::atoi(next().c_str());
        else if (a == "--passes") o.passes = std::atoi(next().c_str());
        else if (a == "--dev-frac") o.dev_frac = std::atof(next().c_str());
        else if (a == "--seed") o.seed = std::atoi(next().c_str());
        else if (a == "--field-dim") o.field_dim = std::atoi(next().c_str());
        else if (a == "--delta-lr") {
            o.delta_lrs.clear();
            for (auto& s : split(next())) o.delta_lrs.push_back(std::atof(s.c_str()));
        } else if (a == "--world-lr") {
            o.world_lrs.clear();
            for (auto& s : split(next())) o.world_lrs.push_back(std::atof(s.c_str()));
        } else if (a == "--gh") {
            o.ghs.clear();
            for (auto& s : split(next())) o.ghs.push_back(std::atoi(s.c_str()));
        } else if (a == "--ask") o.ask = next();
        else if (a == "--quiet") g_quiet = true;
        else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 1;
        } else pos.push_back(a);
    }
    if (pos.size() != 4) {
        std::fprintf(stderr,
                     "usage: %s <train.label> <test.label> <vectors.f32> <vocab.tsv> [options]\n"
                     "       see the header comment of native/tools/question_classifier.cpp\n",
                     argv[0]);
        return 1;
    }
    o.train_path = pos[0];
    o.test_path = pos[1];
    o.vec_path = pos[2];
    o.vocab_path = pos[3];

    const auto T_start = Clock::now();
    const auto train = read_trec(o.train_path);
    const auto test = read_trec(o.test_path);
    {
        std::unordered_set<std::string> c, f;
        for (const auto& q : train) {
            c.insert(q.coarse);
            f.insert(q.fine);
        }
        std::unordered_set<std::string> tc, tf;
        for (const auto& q : test) {
            tc.insert(q.coarse);
            tf.insert(q.fine);
        }
        std::printf("DATA train=%zu test=%zu coarse_train=%zu fine_train=%zu coarse_test=%zu fine_test=%zu\n", train.size(), test.size(), c.size(),
                    f.size(), tc.size(), tf.size());
    }
    const WordVectors wv = load_vectors(o.vec_path, o.vocab_path);
    std::printf("VECTORS words=%zu dim=%d\n", wv.index.size(), wv.dim);

    // 90/10 split of the training questions (fixed seed)
    std::vector<int> perm(train.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937_64 srng(static_cast<std::uint64_t>(o.seed) * 31337 + 7);
    std::shuffle(perm.begin(), perm.end(), srng);
    const std::size_t ndev = static_cast<std::size_t>(o.dev_frac * static_cast<double>(train.size()));
    View vdev, vfin;
    for (std::size_t i = 0; i < perm.size(); ++i) (i < ndev ? vdev.eval : vdev.fit).push_back(train[perm[i]]);
    vfin.fit = train;
    vfin.eval = test;
    std::printf("SPLIT fit=%zu dev=%zu final_train=%zu test=%zu seed=%d\n", vdev.fit.size(), vdev.eval.size(), vfin.fit.size(), vfin.eval.size(), o.seed);
    {
        const auto t0 = Clock::now();
        build_view(vdev, o, wv);
        build_view(vfin, o, wv);
        std::printf("FEATURES built in %.1fs; featurisation %.1f us per question\n", seconds_since(t0),
                    1e6 * seconds_since(t0) / static_cast<double>(vdev.fit.size() + vdev.eval.size() + vfin.fit.size() + vfin.eval.size()));
    }
    const double feat_us = [&] {  // timing of a single featurise call, for the report
        const auto t0 = Clock::now();
        for (int i = 0; i < 200; ++i) (void)vfin.ex.featurize(test[static_cast<std::size_t>(i) % test.size()].text);
        return 1e6 * seconds_since(t0) / 200.0;
    }();
    std::printf("FEAT_US featurise one question: %.1f us (hash tf-idf + cues + SIF)\n", feat_us);

    std::vector<Report> reports;
    struct Best {
        double dev_acc{-1};
        std::function<void(const std::string&)> ask;
    };
    std::unordered_map<std::string, Best> best_for_ask;  // key: model|grain

    for (const std::string& grain : o.grains) {
        const bool fine = grain == "fine";
        Labels L;
        const auto y_fit_dev = gold_ids(vdev.fit, fine, L);
        const auto y_dev = gold_ids(vdev.eval, fine, L);
        const auto y_fin = gold_ids(vfin.fit, fine, L);
        const auto y_test = gold_ids(vfin.eval, fine, L);
        const int K = static_cast<int>(L.names.size());
        std::printf("GRAIN %s: %d labels\n", grain.c_str(), K);

        for (const std::string& set : o.sets) {
            for (const std::string& model : o.models) {
                if (model == "perceptron") {
                    const auto t0 = Clock::now();
                    PercRun dev = run_perceptron(vdev, set, o, y_fit_dev, y_dev, K, 0, 1);
                    const double T = fit_temperature(dev.sc.logits, dev.sc.gold);
                    std::vector<double> dev_cal(dev.sc.gold.size()), p;
                    for (std::size_t i = 0; i < dev_cal.size(); ++i) {
                        softmax_T(dev.sc.logits[i], T, p);
                        dev_cal[i] = *std::max_element(p.begin(), p.end());
                    }
                    const double thr = quantile_threshold(dev_cal, 0.9);
                    const double dev_acc = accuracy(dev.sc.gold, dev.sc.pred);
                    PercRun fin = run_perceptron(vfin, set, o, y_fin, y_test, K, dev.best_epoch, 2);
                    Report r = make_report(grain, set, "perceptron", "epochs=" + std::to_string(dev.best_epoch), fin.sc, T, K);
                    r.thr90 = thr;
                    reports.push_back(r);
                    std::printf("DEV grain=%s set=%s model=perceptron dev_acc=%.4f epochs=%d T=%.3f thr90=%.3f\n", grain.c_str(), set.c_str(), dev_acc,
                                dev.best_epoch, T, thr);
                    print_report(r);
                    note("  [%s %s perceptron] %.1fs\n", grain.c_str(), set.c_str(), seconds_since(t0));
                    if (!o.ask.empty() && dev_acc > best_for_ask["perceptron|" + grain].dev_acc) {
                        auto& slot = best_for_ask["perceptron|" + grain];
                        slot.dev_acc = dev_acc;
                        auto weights = std::make_shared<std::vector<float>>(fin.weights);
                        slot.ask = [&, weights, set, T, thr, K, grain, L](const std::string& text) {
                            const RawFeatures f = vfin.ex.featurize(text);
                            const SparseRow x = perceptron_row(f, set, o);
                            Perceptron P;
                            P.init(K, perceptron_dim(o, wv.dim));
                            std::vector<double> s, pr;
                            P.scores(x, *weights, s);
                            softmax_T(s, T, pr);
                            const int b = static_cast<int>(std::max_element(pr.begin(), pr.end()) - pr.begin());
                            const bool abstain = pr[b] < thr;
                            std::printf("ASK perceptron %-6s set=%-3s : %s\n", grain.c_str(), set.c_str(),
                                        abstain ? "ABSTAIN" : L.names[b].c_str());
                            std::printf("    top: %s %.3f (abstain threshold %.3f for 90%% coverage on dev)\n", L.names[b].c_str(), pr[b], thr);
                        };
                    }
                } else if (model == "ncm") {
                    // Diagnostic, not a product model: batch LDA on the same whitened dense input Cypha sees
                    // (class means + log prior, unit covariance). It is the ceiling of Cypha's scoring family
                    // when its online prototype updates are replaced by exact class means.
                    ensure_dense(vdev, set, o);
                    ensure_dense(vfin, set, o);
                    auto run_ncm = [&](const View& v, const std::vector<int>& yf, const std::vector<int>& ye) {
                        Scored sc;
                        const auto& Xf = v.dense_fit.at(set);
                        const auto& Xe = v.dense_eval.at(set);
                        const std::size_t d = Xf[0].size();
                        const auto t0 = Clock::now();
                        std::vector<double> mu(static_cast<std::size_t>(K) * d, 0.0), cnt(K, 0.0), bias(K, 0.0);
                        for (std::size_t i = 0; i < Xf.size(); ++i) {
                            cnt[yf[i]] += 1;
                            for (std::size_t j = 0; j < d; ++j) mu[static_cast<std::size_t>(yf[i]) * d + j] += Xf[i][j];
                        }
                        for (int k = 0; k < K; ++k) {
                            double n2 = 0;
                            for (std::size_t j = 0; j < d; ++j) {
                                mu[static_cast<std::size_t>(k) * d + j] /= std::max(cnt[k], 1.0);
                                n2 += mu[static_cast<std::size_t>(k) * d + j] * mu[static_cast<std::size_t>(k) * d + j];
                            }
                            bias[k] = -0.5 * n2 + std::log(std::max(cnt[k], 1.0) / static_cast<double>(Xf.size()));
                        }
                        sc.train_s = seconds_since(t0);
                        const auto t1 = Clock::now();
                        sc.gold = ye;
                        std::vector<double> pr;
                        for (std::size_t i = 0; i < Xe.size(); ++i) {
                            std::vector<double> lg(K, 0.0);
                            for (int k = 0; k < K; ++k) {
                                double a = bias[k];
                                for (std::size_t j = 0; j < d; ++j) a += mu[static_cast<std::size_t>(k) * d + j] * Xe[i][j];
                                lg[k] = a;
                            }
                            softmax_T(lg, 1.0, pr);
                            sc.native_conf.push_back(*std::max_element(pr.begin(), pr.end()));
                            sc.pred.push_back(static_cast<int>(std::max_element(lg.begin(), lg.end()) - lg.begin()));
                            sc.logits.push_back(std::move(lg));
                        }
                        sc.infer_us = 1e6 * seconds_since(t1) / static_cast<double>(Xe.size());
                        return sc;
                    };
                    Scored dev = run_ncm(vdev, y_fit_dev, y_dev);
                    const double T = fit_temperature(dev.logits, dev.gold);
                    Scored fin = run_ncm(vfin, y_fin, y_test);
                    Report r = make_report(grain, set, "ncm-lda", "batch", fin, T, K);
                    reports.push_back(r);
                    std::printf("DEV grain=%s set=%s model=ncm-lda dev_acc=%.4f T=%.3f\n", grain.c_str(), set.c_str(), accuracy(dev.gold, dev.pred), T);
                    print_report(r);
                } else if (model == "cypha") {
                    ensure_dense(vdev, set, o);
                    ensure_dense(vfin, set, o);
                    // grid on dev
                    double best_acc = -1;
                    CyphaParams best_cp;
                    int cfg_i = 0;
                    for (int gh : o.ghs)
                        for (double wl : o.world_lrs)
                            for (double dl : o.delta_lrs) {
                                CyphaParams cp;
                                cp.delta_lr = dl;
                                cp.world_lr = wl;
                                cp.gh = gh != 0;
                                cp.passes = o.passes;
                                cp.field_dim = o.field_dim;
                                const auto t0 = Clock::now();
                                CyphaRun run = run_cypha(vdev, set, o, cp, y_fit_dev, y_dev, L, K, true, 10 + cfg_i++);
                                int bp = 1;
                                for (int p = 1; p <= static_cast<int>(run.acc_per_pass.size()); ++p)
                                    if (run.acc_per_pass[p - 1] > run.acc_per_pass[bp - 1] + 1e-12) bp = p;
                                const double a = run.acc_per_pass[bp - 1];
                                std::printf("CFG grain=%s set=%s cypha gh=%d world_lr=%g delta_lr=%g dev_acc_by_pass=[", grain.c_str(), set.c_str(), gh, wl, dl);
                                for (std::size_t p = 0; p < run.acc_per_pass.size(); ++p) std::printf("%s%.3f", p ? " " : "", run.acc_per_pass[p]);
                                std::printf("] best_pass=%d (%.1fs)\n", bp, seconds_since(t0));
                                std::fflush(stdout);
                                if (a > best_acc) {
                                    best_acc = a;
                                    best_cp = cp;
                                    best_cp.passes = bp;
                                }
                            }
                    // dev scores of the best config, to fit T and the threshold (rerun with the chosen passes)
                    CyphaRun devbest = run_cypha(vdev, set, o, best_cp, y_fit_dev, y_dev, L, K, false, 99);
                    const double T = fit_temperature(devbest.sc.logits, devbest.sc.gold);
                    std::vector<double> dev_cal(devbest.sc.gold.size()), p;
                    for (std::size_t i = 0; i < dev_cal.size(); ++i) {
                        softmax_T(devbest.sc.logits[i], T, p);
                        dev_cal[i] = *std::max_element(p.begin(), p.end());
                    }
                    const double thr = quantile_threshold(dev_cal, 0.9);
                    const double dev_acc = accuracy(devbest.sc.gold, devbest.sc.pred);
                    CyphaRun fin = run_cypha(vfin, set, o, best_cp, y_fin, y_test, L, K, false, 100);
                    char cfg[160];
                    std::snprintf(cfg, sizeof cfg, "gh=%d world_lr=%g delta_lr=%g passes=%d", best_cp.gh ? 1 : 0, best_cp.world_lr, best_cp.delta_lr,
                                  best_cp.passes);
                    Report r = make_report(grain, set, "cypha", cfg, fin.sc, T, K);
                    r.thr90 = thr;
                    reports.push_back(r);
                    std::printf("DEV grain=%s set=%s model=cypha dev_acc=%.4f cfg=\"%s\" T=%.3f thr90=%.3f\n", grain.c_str(), set.c_str(), dev_acc, cfg, T, thr);
                    print_report(r);
                    if (!o.ask.empty() && dev_acc > best_for_ask["cypha|" + grain].dev_acc) {
                        auto& slot = best_for_ask["cypha|" + grain];
                        slot.dev_acc = dev_acc;
                        auto clf = std::shared_ptr<CyphaClf>(std::move(fin.clf));
                        auto wh = vfin.whiteners.at(set);
                        slot.ask = [&, clf, wh, set, T, thr, K, grain, L](const std::string& text) {
                            const RawFeatures f = vfin.ex.featurize(text);
                            const std::vector<double> x = wh->apply(raw_dense(f, set, o));
                            const auto po = clf->ask(x);
                            std::vector<double> s(po.all_scores), pr;
                            softmax_T(s, T, pr);
                            const int b = static_cast<int>(std::max_element(pr.begin(), pr.end()) - pr.begin());
                            const bool abstain = pr[b] < thr || po.is_ood;
                            std::printf("ASK cypha      %-6s set=%-3s : %s\n", grain.c_str(), set.c_str(), abstain ? "ABSTAIN" : po.labels[b].c_str());
                            std::printf("    top: %s %.3f (abstain threshold %.3f for 90%% coverage on dev; OOD flag %s, anomaly %.2f)\n", po.labels[b].c_str(),
                                        pr[b], thr, po.is_ood ? "set" : "clear", po.anomaly_score);
                        };
                    }
                }
            }
        }
    }

    // markdown-ish summary
    std::printf("\nSUMMARY (test set; acc, macro-F1, ECE native -> temperature-scaled, acc at 90%% / 80%% coverage, train s, inference us per question)\n");
    for (const auto& r : reports) {
        std::printf("| %-6s | %-3s | %-10s | %.1f | %.1f | %.3f -> %.3f | %.1f | %.1f | %.2f | %.0f |\n", r.grain.c_str(), r.set.c_str(), r.model.c_str(),
                    100 * r.acc, 100 * r.f1, r.ece_native, r.ece_cal, 100 * r.acc_at[1], 100 * r.acc_at[2], r.train_s, r.infer_us);
    }
    if (!o.ask.empty()) {
        for (const std::string& q : split(o.ask, '|')) {  // several questions may be given, separated by '|'
            std::printf("\nQUESTION: %s\n", q.c_str());
            for (const char* m : {"perceptron", "cypha"})
                for (const char* g : {"coarse", "fine"}) {
                    auto it = best_for_ask.find(std::string(m) + "|" + g);
                    if (it != best_for_ask.end() && it->second.ask) it->second.ask(q);
                }
        }
    }
    std::printf("TOTAL wall %.1fs\n", seconds_since(T_start));
    return 0;
}
