#pragma once

/// Similar-context expert for CyphaLM (SEMANTICS §5 and S-4; CYPHALM_ANSWERING route R2, step A4): the count-model
/// form of a kNN-LM (Khandelwal et al. 2020). It predicts the next byte from what followed stored contexts that are
/// *near* the current one in vector space, not identical to it. Report: docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md.
///
/// - Keys. At every word end in the stored text (a word is a run of ASCII letters, digits or apostrophes, as in
///   class_context.hpp), the vector of the previous W words: the SIF-weighted mean (a / (a + p(w)), p from the stored
///   text's own counts) of unit word vectors, centred, projected on principal components (the first `drop` removed),
///   unit-normalised and stored as int8. Each key points at the byte after its word.
/// - Query. At the end of the last completed word (the anchor), the k nearest keys by cosine, from an inverted file
///   over spherical k-means centroids (`probe` of `lists` scanned). One query per word, cached while the tail (the
///   bytes since the anchor: separators and the partial word, at most `max_tail`) grows.
/// - Proposal. A neighbour is live while its stored bytes after the anchor equal the tail; it proposes the next one,
///   weighted by exp((cos - cos_max) / temp) (equal weights at temp 0). The caller mixes the normalised sum into
///   CyphaLM's distribution with one lambda per bucket(), as cyphalm_trie mixes InfiniGram. A neighbour never lives
///   past the word being written, so the expert proposes bytes, never a stored passage (PLAN rule 10).
/// - Batch-built over a text that must outlive the expert. RAM: about (dim rounded up to 16) + 4 bytes a key.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cypha::cyphalm {

class SimilarContext {
 public:
    struct Params {
        int W = 8;              ///< words in a context
        int k = 32;             ///< neighbours per query
        double temp = 0.05;     ///< softmax temperature over cosine; 0 = equal weights
        double min_sim = -1;    ///< neighbours below this cosine are dropped
        bool sif = true;        ///< SIF weights; false = equal word weights
        double sif_a = 1e-3;
        int dim = 64;           ///< principal components kept (0 = all after `drop`)
        int drop = 1;           ///< leading principal components removed
        int lists = 512;        ///< inverted-file lists
        int probe = 8;          ///< lists scanned per query (lists = exact search)
        int max_tail = 32;      ///< bytes after the anchor that a neighbour is followed through
        int iters = 8;          ///< k-means iterations
        std::uint64_t seed = 20261028;
    };
    struct Vectors {  ///< unit word vectors, row r for word id r
        int d = 0;
        std::vector<float> v;
        std::unordered_map<std::string, int> id;
        std::size_t rows() const { return d ? v.size() / static_cast<std::size_t>(d) : 0; }
    };
    struct Hit {
        int live = 0;        ///< neighbours whose bytes agree with the tail
        int word = 0;        ///< bytes of the partial word typed since the separators
        double top = 0;      ///< best live cosine
        double agree = 0;    ///< largest share of one proposed byte
    };
    static constexpr int kBuckets = 1 + 5 * 4 * 3 * 2;

    /// `prefix.vectors.f32` (rows x d float32, no header) and `prefix.vocab.tsv` (word TAB count, one line a row).
    static Vectors load(const std::string& f32, const std::string& vocab, std::string& err) {
        Vectors V;
        std::ifstream vf(vocab);
        std::string line;
        std::vector<std::string> words;
        while (std::getline(vf, line)) words.push_back(line.substr(0, line.find('\t')));
        std::ifstream in(f32, std::ios::binary | std::ios::ate);
        if (words.empty() || !in) return err = "cannot read " + f32 + " / " + vocab, V;
        const std::size_t bytes = static_cast<std::size_t>(in.tellg());
        if (bytes % (4 * words.size())) return err = "size of " + f32 + " is not rows x d x 4", V;
        V.d = static_cast<int>(bytes / 4 / words.size());
        V.v.resize(bytes / 4);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(V.v.data()), static_cast<std::streamsize>(bytes));
        for (std::size_t r = 0; r < words.size(); ++r) {
            float* x = &V.v[r * static_cast<std::size_t>(V.d)];
            double s = 0;
            for (int j = 0; j < V.d; ++j) s += double(x[j]) * x[j];
            for (int j = 0; s > 0 && j < V.d; ++j) x[j] = static_cast<float>(x[j] / std::sqrt(s));
            V.id.emplace(lower(words[r]), static_cast<int>(r));
        }
        return V;
    }

    SimilarContext(Vectors vec, Params p) : vec_(std::move(vec)), p_(p) {
        p_.W = std::max(1, p_.W);
        p_.k = std::max(1, p_.k);
        p_.drop = std::clamp(p_.drop, 0, vec_.d - 1);
        dim_ = p_.dim <= 0 || p_.dim > vec_.d - p_.drop ? vec_.d - p_.drop : p_.dim;
        stride_ = (dim_ + 15) / 16 * 16;
    }

    /// Take the query-time settings (k, temp, min_sim, probe, max_tail) from q; the keys and the index stay.
    void set_query(const Params& q) {
        p_.k = std::max(1, q.k);
        p_.temp = q.temp;
        p_.min_sim = q.min_sim;
        p_.probe = q.probe;
        p_.max_tail = q.max_tail;
        cache_ = ~std::uint64_t{0};
        lat_us_.clear();
    }

    /// Keys at every word end of text[0..n), the projection and the inverted file.
    void build(const std::uint8_t* text, std::size_t n) {
        text_ = text;
        n_ = n;
        std::vector<std::pair<std::uint32_t, int>> words;  // (end, word id or -1)
        std::vector<double> count(vec_.rows(), 0);
        for (std::size_t i = 0; i < n;) {
            if (!is_word(text[i])) { ++i; continue; }
            std::size_t j = i;
            while (j < n && is_word(text[j])) ++j;
            const int id = lookup(text + i, j - i);
            words.push_back({static_cast<std::uint32_t>(j), id});
            if (id >= 0) ++count[static_cast<std::size_t>(id)];
            i = j;
        }
        double total = 0;
        for (double c : count) total += c;
        weight_.assign(vec_.rows(), 1.0f);
        for (std::size_t r = 0; p_.sif && r < weight_.size(); ++r)
            weight_[r] = static_cast<float>(p_.sif_a / (p_.sif_a + count[r] / std::max(total, 1.0)));
        // Principal components of the weighted means (in the vectors' own space) on a sample of contexts.
        const int D = vec_.d;
        const std::size_t step = std::max<std::size_t>(1, words.size() / 40000);
        std::vector<double> mu(static_cast<std::size_t>(D), 0), C(static_cast<std::size_t>(D) * D, 0), x(static_cast<std::size_t>(D));
        std::size_t m = 0;
        for (std::size_t w = 0; w < words.size(); w += step) {
            if (!mean_of(words, w, x)) continue;
            ++m;
            for (int a = 0; a < D; ++a) {
                mu[static_cast<std::size_t>(a)] += x[static_cast<std::size_t>(a)];
                for (int b = a; b < D; ++b) C[static_cast<std::size_t>(a) * D + b] += x[static_cast<std::size_t>(a)] * x[static_cast<std::size_t>(b)];
            }
        }
        if (m == 0) return;
        for (auto& v : mu) v /= static_cast<double>(m);
        for (int a = 0; a < D; ++a)
            for (int b = a; b < D; ++b)
                C[static_cast<std::size_t>(b) * D + a] = C[static_cast<std::size_t>(a) * D + b] =
                    C[static_cast<std::size_t>(a) * D + b] / static_cast<double>(m) - mu[static_cast<std::size_t>(a)] * mu[static_cast<std::size_t>(b)];
        std::vector<double> evec;
        eigen_sym(C, D, evec);  // columns, by falling eigenvalue
        proj_.assign(vec_.rows() * static_cast<std::size_t>(dim_), 0.0f);
        mu_proj_.assign(static_cast<std::size_t>(dim_), 0.0f);
        for (int c = 0; c < dim_; ++c) {
            const std::size_t col = static_cast<std::size_t>(c + p_.drop);
            double s = 0;
            for (int a = 0; a < D; ++a) s += evec[static_cast<std::size_t>(a) * D + col] * mu[static_cast<std::size_t>(a)];
            mu_proj_[static_cast<std::size_t>(c)] = static_cast<float>(s);
            for (std::size_t r = 0; r < vec_.rows(); ++r) {
                double t = 0;
                for (int a = 0; a < D; ++a) t += evec[static_cast<std::size_t>(a) * D + col] * vec_.v[r * static_cast<std::size_t>(D) + static_cast<std::size_t>(a)];
                proj_[r * static_cast<std::size_t>(dim_) + static_cast<std::size_t>(c)] = static_cast<float>(t);
            }
        }
        // Keys.
        std::vector<std::int8_t> keys;
        std::vector<std::uint32_t> pos;
        std::vector<std::int8_t> q(static_cast<std::size_t>(stride_));
        for (std::size_t w = 0; w < words.size(); ++w) {
            std::vector<int> ids;
            for (std::size_t j = w + 1; j-- > 0 && w - j < static_cast<std::size_t>(p_.W);) ids.push_back(words[j].second);
            if (!key_of(ids, q.data())) continue;
            keys.insert(keys.end(), q.begin(), q.end());
            pos.push_back(words[w].first);
        }
        // Inverted file: spherical k-means on a sample, then every key to its nearest centroid.
        const std::size_t N = pos.size();
        if (N == 0) return;
        const int L =static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(static_cast<std::size_t>(std::max(1, p_.lists)), N)));
        std::mt19937_64 rng(p_.seed);
        std::vector<std::size_t> sample(N);
        for (std::size_t i = 0; i < N; ++i) sample[i] = i;
        std::shuffle(sample.begin(), sample.end(), rng);
        sample.resize(std::min<std::size_t>(N, 40 * static_cast<std::size_t>(L)));
        cent_.assign(static_cast<std::size_t>(L) * stride_, 0);
        for (int c = 0; c < L; ++c) std::memcpy(&cent_[static_cast<std::size_t>(c) * stride_], &keys[sample[static_cast<std::size_t>(c)] * stride_], static_cast<std::size_t>(stride_));
        std::vector<int> as(sample.size());
        for (int it = 0; it < p_.iters; ++it) {
            std::vector<double> sum(static_cast<std::size_t>(L) * stride_, 0);
            std::vector<std::size_t> cnt(static_cast<std::size_t>(L), 0);
            for (std::size_t s = 0; s < sample.size(); ++s) {
                as[s] = nearest(&keys[sample[s] * stride_], L);
                ++cnt[static_cast<std::size_t>(as[s])];
                for (int j = 0; j < stride_; ++j) sum[static_cast<std::size_t>(as[s]) * stride_ + static_cast<std::size_t>(j)] += keys[sample[s] * stride_ + static_cast<std::size_t>(j)];
            }
            for (int c = 0; c < L; ++c) {
                std::vector<float> f(static_cast<std::size_t>(stride_));
                if (cnt[static_cast<std::size_t>(c)] == 0) {  // empty: reseed from a random sampled key
                    const std::size_t s = sample[rng() % sample.size()];
                    for (int j = 0; j < stride_; ++j) f[static_cast<std::size_t>(j)] = keys[s * stride_ + static_cast<std::size_t>(j)];
                } else
                    for (int j = 0; j < stride_; ++j) f[static_cast<std::size_t>(j)] = static_cast<float>(sum[static_cast<std::size_t>(c) * stride_ + static_cast<std::size_t>(j)]);
                quantize(f.data(), &cent_[static_cast<std::size_t>(c) * stride_]);
            }
        }
        std::vector<int> list(N);
        start_.assign(static_cast<std::size_t>(L) + 1, 0);
        for (std::size_t i = 0; i < N; ++i) ++start_[static_cast<std::size_t>(list[i] = nearest(&keys[i * stride_], L)) + 1];
        for (int c = 0; c < L; ++c) start_[static_cast<std::size_t>(c) + 1] += start_[static_cast<std::size_t>(c)];
        keys_.assign(N * stride_, 0);
        pos_.assign(N, 0);
        std::vector<std::uint32_t> fill(start_.begin(), start_.end() - 1);
        for (std::size_t i = 0; i < N; ++i) {
            const std::uint32_t at = fill[static_cast<std::size_t>(list[i])]++;
            std::memcpy(&keys_[static_cast<std::size_t>(at) * stride_], &keys[i * stride_], static_cast<std::size_t>(stride_));
            pos_[at] = pos[i];
        }
        lists_ = L;
        cache_ = ~std::uint64_t{0};
    }

    /// The next-byte distribution after h[0..n) (oldest byte first): the live neighbours' proposals, summing to 1, or
    /// all zero when no neighbour is live (live = 0, bucket 0).
    Hit predict(const std::uint8_t* h, std::size_t n, double P[256]) const {
        std::fill(P, P + 256, 0.0);
        Hit hit;
        std::size_t anchor = 0, ctx = 0;
        if (!anchor_of(h, n, anchor, ctx, hit.word)) return hit;
        search_cached(h, ctx, anchor);
        const std::size_t t = n - anchor;
        double wsum = 0, best = -2;
        for (std::size_t j = 0; j < nb_pos_.size(); ++j)
            if (alive(nb_pos_[j], h + anchor, t)) best = std::max(best, static_cast<double>(nb_sim_[j]));
        for (std::size_t j = 0; j < nb_pos_.size(); ++j) {
            if (!alive(nb_pos_[j], h + anchor, t)) continue;
            const double w = p_.temp > 0 ? std::exp((nb_sim_[j] - best) / p_.temp) : 1.0;
            P[text_[nb_pos_[j] + t]] += w;
            wsum += w;
            ++hit.live;
        }
        if (hit.live == 0) return hit;
        double mx = 0;
        for (int s = 0; s < 256; ++s) mx = std::max(mx, P[s] /= wsum);
        hit.top = best;
        hit.agree = mx;
        return hit;
    }

    static int bucket(const Hit& h) {
        if (h.live == 0) return 0;
        const int sb = h.top < 0.5 ? 0 : h.top < 0.65 ? 1 : h.top < 0.8 ? 2 : h.top < 0.9 ? 3 : 4;
        const int ab = h.agree < 0.5 ? 0 : h.agree < 0.8 ? 1 : h.agree < 0.99 ? 2 : 3;
        const int lb = h.live == 1 ? 0 : h.live < 8 ? 1 : 2;
        return 1 + ((sb * 4 + ab) * 3 + lb) * 2 + (h.word > 0);
    }

    /// The k nearest stored contexts of the context ending at the anchor of h[0..n): (position after the key's word,
    /// cosine), best first. `exact` scans every key (the index's recall check). Diagnostics only.
    std::vector<std::pair<std::uint32_t, float>> neighbours(const std::uint8_t* h, std::size_t n, bool exact = false) const {
        std::size_t anchor = 0, ctx = 0;
        int word = 0;
        std::vector<std::pair<std::uint32_t, float>> out;
        if (!anchor_of(h, n, anchor, ctx, word)) return out;
        std::vector<std::int8_t> q(static_cast<std::size_t>(stride_));
        if (!query_key(h, ctx, anchor, q.data())) return out;
        std::vector<std::uint32_t> p;
        std::vector<float> s;
        search(q.data(), exact ? lists_ : p_.probe, p, s);
        for (std::size_t j = 0; j < p.size(); ++j) out.push_back({p[j], s[j]});
        return out;
    }

    std::size_t keys() const { return pos_.size(); }
    std::size_t bytes() const { return keys_.size() + pos_.size() * 4 + cent_.size() + proj_.size() * 4; }
    int dim() const { return dim_; }
    int lists() const { return lists_; }
    std::size_t queries() const { return lat_us_.size(); }
    /// Search latency quantile q in [0, 1], microseconds (every search of this expert so far).
    double latency_us(double q) const {
        if (lat_us_.empty()) return 0;
        std::vector<float> v = lat_us_;
        const std::size_t i = std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())));
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
        return v[i];
    }

    /// Eigenvectors of the symmetric D x D matrix A (cyclic Jacobi), as the columns of V, by falling eigenvalue.
    static void eigen_sym(std::vector<double> A, int D, std::vector<double>& V) {
        const auto at = [D](std::vector<double>& M, int r, int c) -> double& { return M[static_cast<std::size_t>(r) * D + c]; };
        V.assign(static_cast<std::size_t>(D) * D, 0);
        for (int i = 0; i < D; ++i) at(V, i, i) = 1;
        for (int sweep = 0; sweep < 60; ++sweep) {
            double off = 0, all = 0;
            for (int p = 0; p < D; ++p)
                for (int q = 0; q < D; ++q) (p == q ? all : off) += at(A, p, q) * at(A, p, q);
            if (off <= 1e-24 * (all + off) || off == 0) break;
            for (int p = 0; p < D; ++p)
                for (int q = p + 1; q < D; ++q) {
                    const double apq = at(A, p, q);
                    if (apq == 0) continue;
                    const double th = (at(A, q, q) - at(A, p, p)) / (2 * apq);
                    const double t = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th * th + 1));
                    const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                    for (int k = 0; k < D; ++k) {
                        const double kp = at(A, k, p), kq = at(A, k, q);
                        at(A, k, p) = c * kp - s * kq;
                        at(A, k, q) = s * kp + c * kq;
                    }
                    for (int k = 0; k < D; ++k) {
                        const double pk = at(A, p, k), qk = at(A, q, k);
                        at(A, p, k) = c * pk - s * qk;
                        at(A, q, k) = s * pk + c * qk;
                    }
                    for (int k = 0; k < D; ++k) {
                        const double kp = at(V, k, p), kq = at(V, k, q);
                        at(V, k, p) = c * kp - s * kq;
                        at(V, k, q) = s * kp + c * kq;
                    }
                }
        }
        std::vector<int> order(static_cast<std::size_t>(D));
        for (int i = 0; i < D; ++i) order[static_cast<std::size_t>(i)] = i;
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return at(A, x, x) > at(A, y, y); });
        std::vector<double> S(V);
        for (int r = 0; r < D; ++r)
            for (int c = 0; c < D; ++c) at(V, r, c) = S[static_cast<std::size_t>(r) * D + order[static_cast<std::size_t>(c)]];
    }

 private:
    static bool is_word(std::uint8_t b) { return std::isalnum(b) || b == '\''; }
    static std::string lower(std::string s) {
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }
    int lookup(const std::uint8_t* w, std::size_t len) const {
        const auto it = vec_.id.find(lower(std::string(reinterpret_cast<const char*>(w), len)));
        return it == vec_.id.end() ? -1 : it->second;
    }

    // The weighted mean of the words w - W + 1 .. w in the vectors' own space (the PCA sample). False if none known.
    bool mean_of(const std::vector<std::pair<std::uint32_t, int>>& words, std::size_t w, std::vector<double>& x) const {
        std::fill(x.begin(), x.end(), 0.0);
        double m = 0;
        for (std::size_t j = w + 1; j-- > 0 && w - j < static_cast<std::size_t>(p_.W);) {
            const int id = words[j].second;
            if (id < 0) continue;
            const float a = weight_[static_cast<std::size_t>(id)];
            for (int d = 0; d < vec_.d; ++d) x[static_cast<std::size_t>(d)] += a * vec_.v[static_cast<std::size_t>(id) * vec_.d + d];
            ++m;
        }
        if (m == 0) return false;
        for (auto& v : x) v /= m;
        return true;
    }

    // The key of a context from its word ids (unknown = -1): the projected weighted mean, centred, unit, as int8.
    bool key_of(const std::vector<int>& ids, std::int8_t* q) const {
        std::vector<float> x(static_cast<std::size_t>(stride_), 0.0f);
        double m = 0;
        for (int id : ids) {
            if (id < 0) continue;
            const float a = weight_[static_cast<std::size_t>(id)];
            for (int d = 0; d < dim_; ++d) x[static_cast<std::size_t>(d)] += a * proj_[static_cast<std::size_t>(id) * dim_ + d];
            ++m;
        }
        if (m == 0) return false;
        for (int d = 0; d < dim_; ++d) x[static_cast<std::size_t>(d)] = static_cast<float>(x[static_cast<std::size_t>(d)] / m) - mu_proj_[static_cast<std::size_t>(d)];
        return quantize(x.data(), q);
    }

    bool quantize(const float* x, std::int8_t* q) const {
        double s = 0;
        for (int d = 0; d < stride_; ++d) s += double(x[d]) * x[d];
        const double f = s > 1e-20 ? 127.0 / std::sqrt(s) : 0.0;
        for (int d = 0; d < stride_; ++d) q[d] = static_cast<std::int8_t>(std::lround(std::clamp(x[d] * f, -127.0, 127.0)));
        return f > 0;
    }

    int dot(const std::int8_t* a, const std::int8_t* b) const {
        int s = 0;
        for (int i = 0; i < stride_; i += 16)
            for (int j = 0; j < 16; ++j) s += a[i + j] * b[i + j];
        return s;
    }
    int nearest(const std::int8_t* x, int L) const {
        int best = 0, bs = -1 << 30;
        for (int c = 0; c < L; ++c) {
            const int s = dot(x, &cent_[static_cast<std::size_t>(c) * stride_]);
            if (s > bs) bs = s, best = c;
        }
        return best;
    }

    // h[ctx..anchor) holds the context's words, h[anchor..n) the tail. False if the tail is too long or no word ended.
    bool anchor_of(const std::uint8_t* h, std::size_t n, std::size_t& anchor, std::size_t& ctx, int& word) const {
        const std::size_t stop = n > static_cast<std::size_t>(p_.max_tail) + 1 ? n - p_.max_tail - 1 : 0;
        std::size_t i = n;
        while (i > stop && is_word(h[i - 1])) --i;
        word = static_cast<int>(n - i);
        while (i > stop && !is_word(h[i - 1])) --i;
        if (i == stop || n - i > static_cast<std::size_t>(p_.max_tail)) return false;
        anchor = i;
        const std::size_t lim = anchor > 1024 ? anchor - 1024 : 0;
        int w = 0;
        while (i > lim && w < p_.W) {
            while (i > lim && is_word(h[i - 1])) --i;
            ++w;
            while (i > lim && w < p_.W && !is_word(h[i - 1])) --i;
        }
        ctx = i;
        return true;
    }

    bool query_key(const std::uint8_t* h, std::size_t ctx, std::size_t anchor, std::int8_t* q) const {
        std::vector<int> ids;
        for (std::size_t i = anchor; i > ctx;) {  // newest word first
            std::size_t j = i;
            while (j > ctx && is_word(h[j - 1])) --j;
            if (j < i) ids.push_back(lookup(h + j, i - j));
            i = j;
            while (i > ctx && !is_word(h[i - 1])) --i;
        }
        return key_of(ids, q);
    }

    void search(const std::int8_t* q, int probe, std::vector<std::uint32_t>& p, std::vector<float>& s) const {
        p.clear();
        s.clear();
        if (lists_ == 0) return;
        std::vector<std::pair<int, int>> cs(static_cast<std::size_t>(lists_));
        for (int c = 0; c < lists_; ++c) cs[static_cast<std::size_t>(c)] = {dot(q, &cent_[static_cast<std::size_t>(c) * stride_]), c};
        const int np = std::clamp(probe, 1, lists_);
        std::partial_sort(cs.begin(), cs.begin() + np, cs.end(), [](auto& x, auto& y) { return x.first > y.first; });
        std::vector<std::pair<int, std::uint32_t>> cand;
        for (int l = 0; l < np; ++l) {
            const int c = cs[static_cast<std::size_t>(l)].second;
            for (std::uint32_t i = start_[static_cast<std::size_t>(c)]; i < start_[static_cast<std::size_t>(c) + 1]; ++i)
                cand.push_back({dot(q, &keys_[static_cast<std::size_t>(i) * stride_]), i});
        }
        const std::size_t k = std::min<std::size_t>(cand.size(), static_cast<std::size_t>(p_.k));
        std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(k), cand.end(),
                          [](auto& x, auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
        for (std::size_t j = 0; j < k; ++j) {
            const float sim = static_cast<float>(cand[j].first / (127.0 * 127.0));
            if (sim < p_.min_sim) break;
            p.push_back(pos_[cand[j].second]);
            s.push_back(sim);
        }
    }

    void search_cached(const std::uint8_t* h, std::size_t ctx, std::size_t anchor) const {
        std::uint64_t key = 0xcbf29ce484222325ull ^ (anchor - ctx);
        for (std::size_t i = ctx; i < anchor; ++i) key = (key ^ h[i]) * 0x100000001b3ull;
        if (key == cache_) return;
        cache_ = key;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::int8_t> q(static_cast<std::size_t>(stride_));
        nb_pos_.clear();
        nb_sim_.clear();
        if (query_key(h, ctx, anchor, q.data())) search(q.data(), p_.probe, nb_pos_, nb_sim_);
        lat_us_.push_back(static_cast<float>(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()));
    }

    bool alive(std::uint32_t pos, const std::uint8_t* tail, std::size_t t) const {
        return pos + t < n_ && std::memcmp(text_ + pos, tail, t) == 0;
    }

    Vectors vec_;
    Params p_;
    int dim_ = 0, stride_ = 16, lists_ = 0;
    const std::uint8_t* text_ = nullptr;
    std::size_t n_ = 0;
    std::vector<float> weight_, proj_, mu_proj_;
    std::vector<std::int8_t> keys_, cent_;  // keys grouped by list; centroids
    std::vector<std::uint32_t> pos_, start_;
    mutable std::uint64_t cache_ = ~std::uint64_t{0};  // the last searched context, and its neighbours
    mutable std::vector<std::uint32_t> nb_pos_;
    mutable std::vector<float> nb_sim_, lat_us_;
};

}  // namespace cypha::cyphalm
