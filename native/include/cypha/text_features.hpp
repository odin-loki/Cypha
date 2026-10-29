#pragma once
// Count-based text features for Core Cypha's topic routing (PLAN C0/C7; docs/reports/C0_C7_BASELINES_AND_ROUTING.md).
// Used by native/tools/topic_bench.cpp. Kept as a header with a checked file format so Branch A's router can
// compute exactly the measured features: a `count_vectors` router backend built on it reproduced the measured
// test predictions document for document (commit ede6846 on the C0/C7 work), but it is not adopted, because
// C7's registered LCI came out at 0.5 (the calibration term; see the report). No neural network: hashing,
// TF-IDF, a fixed signed random projection, PCA whitening, and SIF means of PPMI-SVD word vectors (C1).
//
// Feature kinds (TextFeaturizer::kind):
//   H    Branch A's hashing featuriser (512 buckets, unigrams + bigrams), PCA to 128, z-scored components
//   T    TF-IDF, sparse (2^18 hashed unigrams + bigrams, tf = 1 + ln c, smooth idf, L2)
//   Td   T projected to 1,024 (4 signed hashed targets a feature), z-scored and PCA-whitened to k
//   S    SIF mean of the word vectors (a / (a + p(w)), a = 1e-3), first principal component removed, whitened
//   Sz   S z-scored, not rotated (for logistic regression)
//   STd  S and Td (at 256) concatenated, whitened together to k
// A fitted featuriser saves to one binary file ("CYTFEAT1"); load() checks the word-vector files against the
// FNV-1a fingerprint taken when it was fitted, and refuses a mismatch.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::textfeat {

// ------------------------------------------------------------------------------------------------ hashing
// MurmurHash3 x86 32-bit, as branch_a_router.cpp.
inline std::uint32_t murmur3(const std::string& s, std::uint32_t seed) {
    const auto* data = reinterpret_cast<const std::uint8_t*>(s.data());
    const std::size_t len = s.size();
    const std::uint32_t c1 = 0xcc9e2d51, c2 = 0x1b873593;
    std::uint32_t h1 = seed;
    const int nblocks = static_cast<int>(len / 4);
    for (int i = 0; i < nblocks; ++i) {
        std::uint32_t k1;
        std::memcpy(&k1, data + 4 * i, 4);
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64;
    }
    const std::uint8_t* tail = data + nblocks * 4;
    std::uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= static_cast<std::uint32_t>(tail[2]) << 16; [[fallthrough]];
        case 2: k1 ^= static_cast<std::uint32_t>(tail[1]) << 8; [[fallthrough]];
        case 1:
            k1 ^= static_cast<std::uint32_t>(tail[0]);
            k1 *= c1;
            k1 = (k1 << 15) | (k1 >> 17);
            k1 *= c2;
            h1 ^= k1;
    }
    h1 ^= static_cast<std::uint32_t>(len);
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;
    return h1;
}

inline std::uint64_t fnv1a(const std::string& s, std::uint64_t seed = 1469598103934665603ULL) {
    std::uint64_t h = seed;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

// ------------------------------------------------------------------------------------------------ text
// Branch A router featuriser (branch_a_router.cpp tokenize_ngrams + hashing_features).
inline std::vector<double> router_hash(const std::string& text, int n_features) {
    std::string lower;
    lower.reserve(text.size());
    for (unsigned char c : text) lower.push_back(static_cast<char>(std::tolower(c)));
    std::istringstream iss(lower);
    std::vector<std::string> words;
    std::string w;
    while (iss >> w) words.push_back(w);
    std::vector<double> acc(static_cast<std::size_t>(n_features), 0.0);
    auto add = [&](const std::string& t) { acc[murmur3(t, 0) % static_cast<std::uint32_t>(n_features)] += 1.0; };
    for (const auto& x : words) add(x);
    for (std::size_t i = 0; i + 1 < words.size(); ++i) add(words[i] + " " + words[i + 1]);
    double s = 0;
    for (double v : acc) s += v * v;
    s = std::sqrt(s);
    if (s > 1e-8)
        for (double& v : acc) v /= s;
    return acc;
}

// T's tokens: lower-case runs of ASCII letters and digits.
inline std::vector<std::string> alnum_tokens(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char c : s) {
        if (std::isalnum(c) && c < 128) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (!cur.empty()) {
            out.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

inline bool is_alnum_c(unsigned char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

// S's tokens: ppmi_svd.cpp's convention (lower-case letter runs with single internal apostrophes or hyphens;
// a token touching a digit is "<num>", which the vectors do not have). As E4.
inline std::vector<std::string> vec_tokens(const std::string& s) {
    std::vector<std::string> out;
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
        if (digit) {
            out.emplace_back("<num>");
        } else {
            std::string t(s, i, j - i);
            for (char& ch : t) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            out.push_back(std::move(t));
        }
        i = j;
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ vectors
struct WordVectors {
    int dim{0};
    std::unordered_map<std::string, int> index;
    std::vector<float> vec;
    std::vector<double> prob;
    const float* row(int i) const { return vec.data() + static_cast<std::size_t>(i) * dim; }
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

inline WordVectors load_vectors(const std::string& vec_path, const std::string& vocab_path) {
    WordVectors wv;
    std::ifstream vf(vocab_path);
    if (!vf) throw std::runtime_error("cannot open " + vocab_path);
    std::string line;
    std::vector<double> counts;
    while (std::getline(vf, line)) {
        const auto tab = line.find('\t');
        wv.index.emplace(line.substr(0, tab), static_cast<int>(counts.size()));
        counts.push_back(tab == std::string::npos ? 1.0 : std::atof(line.c_str() + tab + 1));
    }
    std::ifstream in(vec_path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + vec_path);
    const std::size_t bytes = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    const std::size_t V = counts.size();
    wv.dim = static_cast<int>(bytes / (V * sizeof(float)));
    if (V == 0 || static_cast<std::size_t>(wv.dim) * V * sizeof(float) != bytes)
        throw std::runtime_error("vector file " + vec_path + " is not V x dim x 4 bytes");
    wv.vec.resize(V * static_cast<std::size_t>(wv.dim));
    in.read(reinterpret_cast<char*>(wv.vec.data()), static_cast<std::streamsize>(bytes));
    double total = 0;
    for (double c : counts) total += c;
    wv.prob.resize(V);
    for (std::size_t i = 0; i < V; ++i) wv.prob[i] = counts[i] / total;
    return wv;
}

// ------------------------------------------------------------------------------------------------ linear algebra
// Cyclic Jacobi (E4's): eigenvalues descending, eigenvectors as rows of Vt.
inline void jacobi_eigen(std::vector<double>& A, int n, std::vector<double>& evals, std::vector<double>& Vt) {
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
                for (int k = 0; k < n; ++k) {
                    const double x = a(p, k), y = a(q, k);
                    a(p, k) = c * x - s * y;
                    a(q, k) = s * x + c * y;
                }
                for (int k = 0; k < n; ++k) {
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

// jacobi_eigen with a disk cache keyed by the matrix's bytes (env TOPIC_BENCH_CACHE = a directory). The
// same training rows give the same covariance, so a second fit (another k, another arm) reads the
// decomposition back instead of recomputing it. Numbers are identical either way.
inline void eigen_cached(std::vector<double>& A, int n, std::vector<double>& evals, std::vector<double>& Vt) {
    const char* dir = std::getenv("TOPIC_BENCH_CACHE");
    std::string path;
    if (dir && *dir) {
        std::uint64_t h = 1469598103934665603ULL;
        const auto* b = reinterpret_cast<const unsigned char*>(A.data());
        for (std::size_t i = 0; i < A.size() * sizeof(double); ++i) {
            h ^= b[i];
            h *= 1099511628211ULL;
        }
        char name[64];
        std::snprintf(name, sizeof name, "/eig_%d_%016llx.bin", n, static_cast<unsigned long long>(h));
        path = std::string(dir) + name;
        std::ifstream in(path, std::ios::binary);
        if (in) {
            evals.resize(n);
            Vt.resize(static_cast<std::size_t>(n) * n);
            in.read(reinterpret_cast<char*>(evals.data()), static_cast<std::streamsize>(sizeof(double) * n));
            in.read(reinterpret_cast<char*>(Vt.data()), static_cast<std::streamsize>(sizeof(double) * Vt.size()));
            if (in) return;
        }
    }
    jacobi_eigen(A, n, evals, Vt);
    if (!path.empty()) {
        std::ofstream out(path + ".tmp", std::ios::binary);
        out.write(reinterpret_cast<const char*>(evals.data()), static_cast<std::streamsize>(sizeof(double) * n));
        out.write(reinterpret_cast<const char*>(Vt.data()), static_cast<std::streamsize>(sizeof(double) * Vt.size()));
        out.close();
        std::filesystem::rename(path + ".tmp", path);
    }
}

// PCA projection fitted on training rows. zscore_in: z-score inputs first (E4's whitener); else centre only
// (the router's hashing_svd). Outputs are z-scored components (whitened, with shrinkage eps on the mean
// eigenvalue; eps = 0 gives plain z-scored principal components).
struct Pca {
    int in_dim{0}, out_dim{0};
    bool zin{true};
    std::vector<double> mean, sd, W;
    void fit(const std::vector<std::vector<double>>& X, int p, double eps, bool zscore_in) {
        zin = zscore_in;
        in_dim = static_cast<int>(X[0].size());
        const std::size_t n = X.size();
        mean.assign(in_dim, 0.0);
        sd.assign(in_dim, 1.0);
        for (const auto& r : X)
            for (int j = 0; j < in_dim; ++j) mean[j] += r[j];
        for (double& m : mean) m /= static_cast<double>(n);
        if (zin) {
            std::fill(sd.begin(), sd.end(), 0.0);
            for (const auto& r : X)
                for (int j = 0; j < in_dim; ++j) sd[j] += (r[j] - mean[j]) * (r[j] - mean[j]);
            for (double& s : sd) s = std::sqrt(s / static_cast<double>(n)) + 1e-8;
        }
        if (p <= 0 || p >= in_dim) p = in_dim;
        out_dim = p;
        std::vector<double> C(static_cast<std::size_t>(in_dim) * in_dim, 0.0), z(in_dim);
        for (const auto& r : X) {
            for (int j = 0; j < in_dim; ++j) z[j] = (r[j] - mean[j]) / sd[j];
            for (int i = 0; i < in_dim; ++i) {
                const double zi = z[i];
                if (zi == 0.0) continue;
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
        eigen_cached(C, in_dim, ev, Vt);
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

// ------------------------------------------------------------------------------------------------ sparse TF-IDF
struct SparseRow {
    std::vector<std::uint32_t> idx;
    std::vector<float> val;
};

constexpr int kTBits = 18;
constexpr int kProj = 1024;

inline std::unordered_map<std::uint32_t, float> t_counts(const std::string& text) {
    const auto t = alnum_tokens(text);
    std::unordered_map<std::uint32_t, float> m;
    const std::uint32_t mask = (1u << kTBits) - 1u;
    for (std::size_t i = 0; i < t.size(); ++i) {
        m[murmur3("u " + t[i], 0) & mask] += 1.f;
        if (i + 1 < t.size()) m[murmur3("b " + t[i] + " " + t[i + 1], 0) & mask] += 1.f;
    }
    return m;
}

struct TfIdf {
    std::vector<float> idf;
    void fit(const std::vector<const std::string*>& docs) {
        const std::size_t B = 1u << kTBits, N = docs.size();
        std::vector<int> df(B, 0);
        for (const auto* d : docs)
            for (const auto& kv : t_counts(*d)) df[kv.first]++;
        idf.assign(B, 0.f);
        for (std::size_t i = 0; i < B; ++i) idf[i] = static_cast<float>(std::log((1.0 + N) / (1.0 + df[i])) + 1.0);
    }
    SparseRow row(const std::string& text) const {
        auto m = t_counts(text);
        std::vector<std::pair<std::uint32_t, float>> v(m.begin(), m.end());
        std::sort(v.begin(), v.end());
        SparseRow r;
        double s = 0;
        for (const auto& kv : v) {
            const float x = static_cast<float>((1.0 + std::log(kv.second)) * idf[kv.first]);
            r.idx.push_back(kv.first);
            r.val.push_back(x);
            s += static_cast<double>(x) * x;
        }
        if (s > 0) {
            const float inv = static_cast<float>(1.0 / std::sqrt(s));
            for (float& x : r.val) x *= inv;
        }
        return r;
    }
};

// E4's fixed signed random projection (4 hashed targets a feature).
inline std::vector<double> project_sparse(const SparseRow& r, int P) {
    std::vector<double> out(P, 0.0);
    for (std::size_t k = 0; k < r.idx.size(); ++k) {
        const std::uint64_t h = fnv1a(std::to_string(r.idx[k]), 0x9e3779b97f4a7c15ULL);
        for (int t = 0; t < 4; ++t) {
            const std::uint64_t g = (h >> (t * 14)) ^ (h * (static_cast<std::uint64_t>(t) + 1) >> 7);
            const std::size_t col = static_cast<std::size_t>(g % static_cast<std::uint64_t>(P));
            const double sign = ((h >> (60 - t)) & 1ULL) ? 1.0 : -1.0;
            out[col] += sign * 0.5 * r.val[k];
        }
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ SIF
struct Sif {
    const WordVectors* wv{nullptr};
    double a{1e-3};
    std::vector<double> pc;
    std::vector<double> raw(const std::string& text) const {
        std::vector<double> v(wv->dim, 0.0);
        int cnt = 0;
        for (const auto& t : vec_tokens(text)) {
            if (t == "<num>") continue;
            const int id = wv->find(t);
            if (id < 0) continue;
            const double w = a / (a + wv->prob[id]);
            const float* r = wv->row(id);
            for (int j = 0; j < wv->dim; ++j) v[j] += w * r[j];
            ++cnt;
        }
        if (cnt > 0)
            for (double& x : v) x /= cnt;
        return v;
    }
    // First principal component of the training rows (uncentred, Arora et al.), by power iteration.
    void fit(const std::vector<std::vector<double>>& rows) {
        const int d = wv->dim;
        std::vector<double> u(d, 1.0 / std::sqrt(static_cast<double>(d))), nu(d);
        for (int it = 0; it < 60; ++it) {
            std::fill(nu.begin(), nu.end(), 0.0);
            for (const auto& r : rows) {
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
    }
    std::vector<double> remove_pc(std::vector<double> r) const {
        double dot = 0;
        for (std::size_t j = 0; j < r.size(); ++j) dot += r[j] * pc[j];
        for (std::size_t j = 0; j < r.size(); ++j) r[j] -= dot * pc[j];
        return r;
    }
};


// ------------------------------------------------------------------------------------------------ featuriser
inline std::uint64_t fnv1a_file(const std::string& path, std::uint64_t h = 1469598103934665603ULL) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<char> buf(1 << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize n = in.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            h ^= static_cast<unsigned char>(buf[static_cast<std::size_t>(i)]);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

struct TextFeaturizer {
    std::string kind;  // H, T, Td, S, Sz, STd
    int k{0};
    const WordVectors* wv{nullptr};
    std::string vectors_path, vocab_path;  // recorded by save(); used by load()
    std::uint64_t vectors_fp{0};           // FNV-1a over the vector file, then the vocab file
    TfIdf tfidf;
    Sif sif;
    Pca pca_h, pca_t, pca_s, pca_st;
    std::vector<double> zmean, zsd;  // Sz only
    WordVectors owned_wv;            // filled by load()

    bool sparse() const { return kind == "T"; }
    bool uses_vectors() const { return kind == "S" || kind == "Sz" || kind == "STd"; }
    bool uses_tfidf() const { return kind == "T" || kind == "Td" || kind == "STd"; }

    void fit(const std::vector<const std::string*>& docs) {
        if (kind == "H") {
            std::vector<std::vector<double>> X;
            for (const auto* s : docs) X.push_back(router_hash(*s, 512));
            pca_h.fit(X, 128, 0.0, false);
            return;
        }
        if (uses_tfidf()) tfidf.fit(docs);
        if (uses_vectors()) {
            sif.wv = wv;
            std::vector<std::vector<double>> R;
            for (const auto* s : docs) R.push_back(sif.raw(*s));
            sif.fit(R);
            for (auto& r : R) r = sif.remove_pc(r);
            if (kind == "S") pca_s.fit(R, 0, 0.1, true);
            if (kind == "Sz") {
                const std::size_t d = R[0].size();
                zmean.assign(d, 0.0);
                zsd.assign(d, 0.0);
                for (const auto& r : R)
                    for (std::size_t j = 0; j < d; ++j) zmean[j] += r[j];
                for (double& m : zmean) m /= static_cast<double>(R.size());
                for (const auto& r : R)
                    for (std::size_t j = 0; j < d; ++j) zsd[j] += (r[j] - zmean[j]) * (r[j] - zmean[j]);
                for (double& v : zsd) v = std::sqrt(v / static_cast<double>(R.size())) + 1e-8;
            }
        }
        if (kind == "Td" || kind == "STd") {
            std::vector<std::vector<double>> P;
            for (const auto* s : docs) P.push_back(project_sparse(tfidf.row(*s), kProj));
            pca_t.fit(P, kind == "Td" ? k : 256, 0.1, true);
            if (kind == "STd") {
                std::vector<std::vector<double>> C;
                for (std::size_t i = 0; i < docs.size(); ++i) {
                    auto s = sif.remove_pc(sif.raw(*docs[i]));
                    auto t = pca_t.apply(P[i]);
                    s.insert(s.end(), t.begin(), t.end());
                    C.push_back(std::move(s));
                }
                pca_st.fit(C, k, 0.1, true);
            }
        }
    }

    std::vector<double> dense(const std::string& text) const {
        if (kind == "H") return pca_h.apply(router_hash(text, 512));
        if (kind == "Td") return pca_t.apply(project_sparse(tfidf.row(text), kProj));
        if (kind == "S") return pca_s.apply(sif.remove_pc(sif.raw(text)));
        if (kind == "Sz") {
            auto r = sif.remove_pc(sif.raw(text));
            for (std::size_t j = 0; j < r.size(); ++j) r[j] = (r[j] - zmean[j]) / zsd[j];
            return r;
        }
        if (kind == "STd") {
            auto s = sif.remove_pc(sif.raw(text));
            auto t = pca_t.apply(project_sparse(tfidf.row(text), kProj));
            s.insert(s.end(), t.begin(), t.end());
            return pca_st.apply(s);
        }
        throw std::runtime_error("dense() is not defined for feature kind " + kind);
    }

    int dim() const {
        if (kind == "H") return pca_h.out_dim;
        if (kind == "Td") return pca_t.out_dim;
        if (kind == "S") return pca_s.out_dim;
        if (kind == "Sz") return static_cast<int>(zmean.size());
        if (kind == "STd") return pca_st.out_dim;
        return 0;
    }

    // ---- binary file
    struct Writer {
        std::string b;
        void u64(std::uint64_t v) { b.append(reinterpret_cast<const char*>(&v), 8); }
        void i32(std::int32_t v) { b.append(reinterpret_cast<const char*>(&v), 4); }
        void f64(double v) { b.append(reinterpret_cast<const char*>(&v), 8); }
        void str(const std::string& s) {
            u64(s.size());
            b += s;
        }
        void vd(const std::vector<double>& v) {
            u64(v.size());
            b.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double));
        }
        void vf(const std::vector<float>& v) {
            u64(v.size());
            b.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
        }
        void pca(const Pca& p) {
            i32(p.in_dim);
            i32(p.out_dim);
            i32(p.zin ? 1 : 0);
            vd(p.mean);
            vd(p.sd);
            vd(p.W);
        }
    };
    struct Reader {
        const std::string& b;
        std::size_t pos{0};
        void need(std::size_t n) {
            if (pos + n > b.size()) throw std::runtime_error("featuriser file is truncated");
        }
        std::uint64_t u64() {
            need(8);
            std::uint64_t v;
            std::memcpy(&v, b.data() + pos, 8);
            pos += 8;
            return v;
        }
        std::int32_t i32() {
            need(4);
            std::int32_t v;
            std::memcpy(&v, b.data() + pos, 4);
            pos += 4;
            return v;
        }
        double f64() {
            need(8);
            double v;
            std::memcpy(&v, b.data() + pos, 8);
            pos += 8;
            return v;
        }
        std::string str() {
            const std::size_t n = u64();
            need(n);
            std::string s = b.substr(pos, n);
            pos += n;
            return s;
        }
        std::vector<double> vd() {
            const std::size_t n = u64();
            need(n * sizeof(double));
            std::vector<double> v(n);
            std::memcpy(v.data(), b.data() + pos, n * sizeof(double));
            pos += n * sizeof(double);
            return v;
        }
        std::vector<float> vf() {
            const std::size_t n = u64();
            need(n * sizeof(float));
            std::vector<float> v(n);
            std::memcpy(v.data(), b.data() + pos, n * sizeof(float));
            pos += n * sizeof(float);
            return v;
        }
        Pca pca() {
            Pca p;
            p.in_dim = i32();
            p.out_dim = i32();
            p.zin = i32() != 0;
            p.mean = vd();
            p.sd = vd();
            p.W = vd();
            if (p.mean.size() != static_cast<std::size_t>(p.in_dim) || p.sd.size() != static_cast<std::size_t>(p.in_dim) ||
                p.W.size() != static_cast<std::size_t>(p.in_dim) * static_cast<std::size_t>(p.out_dim))
                throw std::runtime_error("featuriser file has a malformed PCA block");
            return p;
        }
    };

    // vectors_path / vocab_path / vectors_fp must be set when the kind uses word vectors.
    void save(const std::string& path) const {
        Writer w;
        w.b.assign("CYTFEAT1", 8);
        w.str(kind);
        w.i32(k);
        w.f64(sif.a);
        w.str(vectors_path);
        w.str(vocab_path);
        w.u64(vectors_fp);
        w.vf(tfidf.idf);
        w.vd(sif.pc);
        w.vd(zmean);
        w.vd(zsd);
        w.pca(pca_h);
        w.pca(pca_t);
        w.pca(pca_s);
        w.pca(pca_st);
        std::uint64_t h = 1469598103934665603ULL;
        for (unsigned char c : w.b) {
            h ^= c;
            h *= 1099511628211ULL;
        }
        w.u64(h);
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write " + path);
        out.write(w.b.data(), static_cast<std::streamsize>(w.b.size()));
    }

    // Loads the file; the word vectors come from the recorded paths unless vectors_override / vocab_override
    // are given (a relative recorded path is resolved against the featuriser file's directory).
    void load(const std::string& path, const std::string& vectors_override = {}, const std::string& vocab_override = {}) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open featuriser file " + path);
        std::ostringstream ss;
        ss << in.rdbuf();
        const std::string b = ss.str();
        if (b.size() < 16 || b.compare(0, 8, "CYTFEAT1") != 0) throw std::runtime_error("not a featuriser file: " + path);
        std::uint64_t h = 1469598103934665603ULL;
        for (std::size_t i = 0; i < b.size() - 8; ++i) {
            h ^= static_cast<unsigned char>(b[i]);
            h *= 1099511628211ULL;
        }
        std::uint64_t stored;
        std::memcpy(&stored, b.data() + b.size() - 8, 8);
        if (h != stored) throw std::runtime_error("featuriser file checksum mismatch: " + path);
        Reader r{b, 8};
        kind = r.str();
        k = r.i32();
        sif.a = r.f64();
        vectors_path = r.str();
        vocab_path = r.str();
        vectors_fp = r.u64();
        tfidf.idf = r.vf();
        sif.pc = r.vd();
        zmean = r.vd();
        zsd = r.vd();
        pca_h = r.pca();
        pca_t = r.pca();
        pca_s = r.pca();
        pca_st = r.pca();
        if (uses_tfidf() && tfidf.idf.size() != (1u << kTBits)) throw std::runtime_error("featuriser file: idf size");
        if (uses_vectors()) {
            const auto base = std::filesystem::path(path).parent_path();
            auto resolve = [&](const std::string& over, const std::string& rec) {
                if (!over.empty()) return over;
                const std::filesystem::path p(rec);
                return (p.is_absolute() ? p : base / p).string();
            };
            const std::string vp = resolve(vectors_override, vectors_path), vo = resolve(vocab_override, vocab_path);
            const std::uint64_t fp = fnv1a_file(vo, fnv1a_file(vp));
            if (fp != vectors_fp) throw std::runtime_error("word vectors do not match the featuriser: " + vp);
            owned_wv = load_vectors(vp, vo);
            wv = &owned_wv;
            sif.wv = wv;
            if (sif.pc.size() != static_cast<std::size_t>(wv->dim)) throw std::runtime_error("featuriser file: SIF size");
        }
    }
};

}  // namespace cypha::textfeat
