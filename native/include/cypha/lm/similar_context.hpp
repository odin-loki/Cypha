#pragma once

/// Similar-context expert for CyphaLM (SEMANTICS §5 and S-4; CYPHALM_ANSWERING route R2, step A4): the count-model
/// form of a kNN-LM (Khandelwal et al. 2020). It predicts the next byte from what followed stored contexts that are
/// *near* the current one in vector space, not identical to it. Report: docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md.
///
/// - Keys. At every word end in the stored text (a word is a run of ASCII letters, digits or apostrophes, as in
///   class_context.hpp), the vector of the previous W words: the SIF-weighted (a / (a + p(w)), p from the stored
///   text's own counts) or equal-weighted mean of unit word vectors, centred, projected on principal components (the
///   first `drop` removed), unit-normalised and stored as int8. Each key points at the byte after its word.
/// - Query. At the end of the last completed word (the anchor), the k nearest keys by cosine, from an inverted file
///   over spherical k-means centroids (`probe` of `lists` scanned). One query per word, cached while the tail (the
///   bytes since the anchor: separators and the partial word, at most `max_tail`) grows.
/// - Proposal. A neighbour is live while its stored bytes after the anchor equal the tail; it proposes the next one,
///   weighted by exp((cos - cos_max) / temp) (equal weights at temp 0). The caller mixes the normalised sum into
///   CyphaLM's distribution with one lambda per bucket(), as cyphalm_trie mixes InfiniGram. A neighbour never lives
///   past the word being written, so the expert proposes bytes, never a stored passage (PLAN rule 10).
/// - Batch-built over a text that must outlive the expert. RAM: about (dim rounded up to 16) + 4 bytes a key.
/// - Vectors come as unit rows (`load`: ppmi_svd's .vectors.f32 and .vocab.tsv; the build finds the principal
///   components) or already projected (`load_projected`: int8 rows with a scale each, as `save_projected` writes them
///   after a build; the build then only centres them). A projected file from a build over a text gives that text the
///   same keys as the full vectors, up to the int8 rounding of the rows.
/// - Params' defaults are the design registered before A4's runs. The settings chosen on the holdout are W 2, equal
///   weights, k 2048, drop 0, 2048 lists with 32 probed (PPMI-SVD vectors): fresh text -0.061 BPC over trie +
///   InfiniGram + class. A4c's default adds mode 1 (dedupe) with k 8192: fresh text a further -0.0040. A4d's adds 128
///   probed lists (recall 0.65 -> 0.94) and online keys: fresh text a further -0.145.
/// - A4c, `mode` 1 (dedupe, the default spec's): each distinct key is stored once with all its occurrences; the k
///   nearest distinct keys each pool their occurrences, weighted as a neighbour is. A key's occurrences are sorted by
///   the bytes after them, so the ones that agree with the tail are one range, narrowed by binary search. Mode 0 (knn,
///   each occurrence a neighbour) is A4's. A4c's counts mode (whole lists pooled) lost and was removed (commit 2849b32).
/// - A4d, `online` (keys from the text being read): at every word end j of the stream that lies before the current
///   anchor, the key of the W words ending at j joins its nearest list; a search scans the probed lists' online keys
///   beside the stored ones. An online key is one occurrence: live while the stream's bytes [j, j + t) equal the tail,
///   it proposes the byte at j + t. As j < anchor and t = n - anchor, j + t < n: it reads only bytes already scored.
///   The stream is the stored text's own buffer from its end on (h == the stored text), or the query text from its
///   start; the online keys start again when the stream is not the one read so far.
/// - A4e: each online key costs (dim rounded up to 16) + 4 bytes, 5.3 to 5.5 MB a MiB of text read, with no end, and
///   every search scans the probed lists' share of them. Every bound tried (2^15 to 2^19 keys; the oldest dropped, or
///   the least recently found) lost more than 0.0005 bits per byte over 8 MiB, so none was kept
///   (docs/reports/A4_SIMILAR_CONTEXT_EXPERT.md, "A4e"; the bound's code is at commit a854346).
/// - A4f, `merge` (dedupe mode, with `online`): an online key equal (all its int8 bytes) to a stored key of its list or
///   to an earlier online key is not stored again; its position joins that key's. A list's stored keys lie sorted by
///   their bytes, so a binary search finds an equal one; the list's online keys keep a sorted index. Each list holds its
///   repeats in one array grouped by owner (its stored keys, then its online keys) with a start per owner. The k nearest
///   are k distinct keys; a key found brings every position it has, each weighted as the key.

#include <algorithm>
#include <cctype>
#include <istream>
#include <limits>
#include <ostream>
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
        int mode = 0;           ///< A4c: 0 knn (each occurrence a neighbour), 1 dedupe (see above)
        bool online = false;    ///< A4d: keys from the text being read too (see above)
        bool merge = false;     ///< A4f: online keys equal to one held are stored once, with every position (dedupe)
    };
    struct Vectors {  ///< word vectors, row r for word id r
        int d = 0;
        std::vector<float> v;            ///< unit rows (load); empty for projected vectors
        std::vector<std::int8_t> q;      ///< projected rows (load_projected): row r is scale[r] * q[r * d .. r * d + d)
        std::vector<float> scale;
        std::unordered_map<std::string, int> id;
        bool projected() const { return !scale.empty(); }
        std::size_t rows() const {
            return projected() ? scale.size() : d ? v.size() / static_cast<std::size_t>(d) : 0;
        }
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

    /// A projected file: "CYSIMV1" and a zero byte, rows and dim (uint32), each row's word (a length byte and its
    /// bytes), the rows' scales (float32), the rows (int8, rows x dim), then the FNV-1a hash of every byte before it.
    /// Little-endian. Returns empty vectors and sets err on a bad file.
    static Vectors load_projected(std::istream& in, std::string& err) {
        Vectors V;
        std::uint64_t h = 0xcbf29ce484222325ull;
        auto get = [&](void* dst, std::size_t n) {
            in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
            const auto* b = static_cast<const unsigned char*>(dst);
            for (std::size_t i = 0; in && i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
            return static_cast<bool>(in);
        };
        auto u32 = [&](std::uint32_t& x) {
            unsigned char b[4];
            if (!get(b, 4)) return false;
            x = std::uint32_t{b[0]} | std::uint32_t{b[1]} << 8 | std::uint32_t{b[2]} << 16 | std::uint32_t{b[3]} << 24;
            return true;
        };
        char magic[8];
        std::uint32_t rows = 0, dim = 0;
        if (!get(magic, 8) || std::memcmp(magic, "CYSIMV1", 8) != 0 || !u32(rows) || !u32(dim) || dim == 0 || dim > 4096 ||
            rows == 0 || rows > (1u << 26))
            return err = "not a projected vector file", Vectors{};
        V.d = static_cast<int>(dim);
        for (std::uint32_t r = 0; r < rows; ++r) {
            unsigned char len = 0;
            std::string w(255, '\0');
            if (!get(&len, 1) || (len && !get(&w[0], len))) return err = "projected vector file cut short", Vectors{};
            w.resize(len);
            if (len) V.id.emplace(lower(w), static_cast<int>(r));
        }
        V.scale.resize(rows);
        for (auto& f : V.scale) {
            std::uint32_t x = 0;
            if (!u32(x)) return err = "projected vector file cut short", Vectors{};
            std::memcpy(&f, &x, 4);
        }
        V.q.resize(std::size_t{rows} * dim);
        if (!get(V.q.data(), V.q.size())) return err = "projected vector file cut short", Vectors{};
        const std::uint64_t want = h;
        unsigned char b[8];
        if (!in.read(reinterpret_cast<char*>(b), 8)) return err = "projected vector file cut short", Vectors{};
        std::uint64_t got = 0;
        for (int i = 7; i >= 0; --i) got = got << 8 | b[i];
        if (got != want) return err = "projected vector file checksum mismatch", Vectors{};
        return V;
    }
    static Vectors load_projected(const std::string& path, std::string& err) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return err = "cannot read " + path, Vectors{};
        return load_projected(in, err);
    }

    /// Write this expert's projected rows (its keys' coordinates for every word) after build(), int8 with a scale a row.
    bool save_projected(std::ostream& out) const {
        if (!rows_ || !dim_) return false;
        std::uint64_t h = 0xcbf29ce484222325ull;
        auto put = [&](const void* src, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(src);
            for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
            out.write(static_cast<const char*>(src), static_cast<std::streamsize>(n));
        };
        auto u32 = [&](std::uint32_t x) {
            const unsigned char b[4] = {static_cast<unsigned char>(x), static_cast<unsigned char>(x >> 8),
                                        static_cast<unsigned char>(x >> 16), static_cast<unsigned char>(x >> 24)};
            put(b, 4);
        };
        std::vector<std::string> word(rows_);
        for (const auto& [w, r] : vec_.id)
            if (static_cast<std::size_t>(r) < rows_ && w.size() < 256) word[static_cast<std::size_t>(r)] = w;
        std::vector<float> sc(rows_);
        std::vector<std::int8_t> qv(rows_ * static_cast<std::size_t>(dim_));
        for (std::size_t r = 0; r < rows_; ++r) {
            float mx = 0;
            for (int d = 0; d < dim_; ++d) mx = std::max(mx, std::fabs(proj(r, d)));
            sc[r] = mx > 0 ? mx / 127.0f : 0.0f;
            for (int d = 0; d < dim_; ++d)
                qv[r * static_cast<std::size_t>(dim_) + static_cast<std::size_t>(d)] =
                    sc[r] > 0 ? static_cast<std::int8_t>(std::lround(std::clamp(proj(r, d) / sc[r], -127.0f, 127.0f))) : 0;
        }
        put("CYSIMV1", 8);
        u32(static_cast<std::uint32_t>(rows_));
        u32(static_cast<std::uint32_t>(dim_));
        for (const auto& w : word) {
            const unsigned char len = static_cast<unsigned char>(w.size());
            put(&len, 1);
            put(w.data(), w.size());
        }
        for (float f : sc) {
            std::uint32_t x = 0;
            std::memcpy(&x, &f, 4);
            u32(x);
        }
        put(qv.data(), qv.size());
        unsigned char b[8];
        for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(h >> (8 * i));
        out.write(reinterpret_cast<const char*>(b), 8);
        return static_cast<bool>(out);
    }

    SimilarContext(Vectors vec, Params p) : vec_(std::move(vec)), p_(p) {
        p_.W = std::max(1, p_.W);
        p_.k = std::max(1, p_.k);
        p_.merge = p_.merge && p_.mode == 1;  // A4f: dedupe mode only
        if (vec_.projected()) {  // already the coordinates: no components dropped; dim keeps the leading ones
            p_.drop = 0;
            dim_ = p_.dim <= 0 || p_.dim > vec_.d ? vec_.d : p_.dim;
        } else {
            p_.drop = std::clamp(p_.drop, 0, vec_.d - 1);
            dim_ = p_.dim <= 0 || p_.dim > vec_.d - p_.drop ? vec_.d - p_.drop : p_.dim;
        }
        stride_ = (dim_ + 15) / 16 * 16;
    }

    /// Take the query-time settings (k, temp, min_sim, probe, max_tail) from q; the keys and the index stay.
    void set_query(const Params& q) {
        p_.k = std::max(1, q.k);
        p_.temp = q.temp;
        p_.min_sim = q.min_sim;
        p_.probe = q.probe;
        p_.max_tail = p_.mode ? std::min(q.max_tail, sort_tail_) : q.max_tail;  // A4c: groups are sorted that deep
        p_.online = q.online;  // A4d
        p_.merge = q.merge && p_.mode == 1;  // A4f: dedupe mode only
        reset_online(nullptr, 0);
        cache_ = ~std::uint64_t{0};
        restart_ = true;
        lat_us_.clear();
    }

    /// Keys at every word end of text[0..n), the projection and the inverted file.
    void build(const std::uint8_t* text, std::size_t n) {
        text_ = text;
        n_ = n;
        rows_ = vec_.rows();
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
        const std::size_t step = std::max<std::size_t>(1, words.size() / 40000);  // the sample of contexts
        if (vec_.projected()) {  // the rows are the coordinates already: only the centre is needed
            std::vector<double> mu(static_cast<std::size_t>(dim_), 0);
            std::vector<float> x(static_cast<std::size_t>(dim_));
            std::size_t m = 0;
            for (std::size_t w = 0; w < words.size(); w += step)
                if (proj_mean(window(words, w), x.data())) {
                    ++m;
                    for (int d = 0; d < dim_; ++d) mu[static_cast<std::size_t>(d)] += x[static_cast<std::size_t>(d)];
                }
            if (m == 0) return;
            mu_proj_.assign(static_cast<std::size_t>(dim_), 0.0f);
            for (int d = 0; d < dim_; ++d) mu_proj_[static_cast<std::size_t>(d)] = static_cast<float>(mu[static_cast<std::size_t>(d)] / static_cast<double>(m));
        } else if (!pca(words, step)) {
            return;
        }
        build_index(words);
    }

 private:
    // Principal components of the weighted means (in the vectors' own space) on a sample of contexts; the rows'
    // projections and the projected centre. The unit rows are then freed: queries read only the projections.
    bool pca(const std::vector<std::pair<std::uint32_t, int>>& words, std::size_t step) {
        const int D = vec_.d;
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
        if (m == 0) return false;
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
        std::vector<float>().swap(vec_.v);
        return true;
    }

    // The ids of the words w - W + 1 .. w, newest first.
    std::vector<int> window(const std::vector<std::pair<std::uint32_t, int>>& words, std::size_t w) const {
        std::vector<int> ids;
        for (std::size_t j = w + 1; j-- > 0 && w - j < static_cast<std::size_t>(p_.W);) ids.push_back(words[j].second);
        return ids;
    }

    // Keys at every word end, then the inverted file. The keys are put in list order in place.
    void build_index(const std::vector<std::pair<std::uint32_t, int>>& words) {
        std::vector<std::int8_t> keys;
        std::vector<std::uint32_t> pos;
        std::vector<std::uint32_t> occ, grp;  // A4c dedupe: the occurrences, grouped by distinct key
        if (p_.mode == 1) {
            distinct_keys(words, keys, pos, occ, grp);
        } else {
            keys.reserve(words.size() * static_cast<std::size_t>(stride_));  // at most one key a word: no reallocation
            pos.reserve(words.size());
            std::vector<std::int8_t> q(static_cast<std::size_t>(stride_));
            for (std::size_t w = 0; w < words.size(); ++w) {
                if (!key_of(window(words, w), q.data())) continue;
                keys.insert(keys.end(), q.begin(), q.end());
                pos.push_back(words[w].first);
            }
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
        std::vector<std::uint32_t> slot(N);  // each key's list, then its place in list order
        start_.assign(static_cast<std::size_t>(L) + 1, 0);
        for (std::size_t i = 0; i < N; ++i)
            ++start_[(slot[i] = static_cast<std::uint32_t>(nearest(&keys[i * stride_], L))) + std::size_t{1}];
        for (int c = 0; c < L; ++c) start_[static_cast<std::size_t>(c) + 1] += start_[static_cast<std::size_t>(c)];
        std::vector<std::uint32_t> fill(start_.begin(), start_.end() - 1);
        for (std::size_t i = 0; i < N; ++i) slot[i] = fill[slot[i]]++;
        std::vector<std::int8_t> tmp(static_cast<std::size_t>(stride_));
        for (std::size_t i = 0; i < N; ++i)  // follow each cycle of the permutation
            while (slot[i] != i) {
                const std::size_t j = slot[i];
                std::memcpy(tmp.data(), &keys[i * stride_], static_cast<std::size_t>(stride_));
                std::memcpy(&keys[i * stride_], &keys[j * stride_], static_cast<std::size_t>(stride_));
                std::memcpy(&keys[j * stride_], tmp.data(), static_cast<std::size_t>(stride_));
                std::swap(pos[i], pos[j]);
                std::swap(slot[i], slot[j]);
            }
        lists_ = L;
        if (p_.mode == 1) group_occurrences(pos, occ, grp);  // A4c
        keys_ = std::move(keys);
        pos_ = std::move(pos);
        cache_ = ~std::uint64_t{0};
        restart_ = true;
    }

    // A4c dedupe: the distinct keys, found with an open-addressing table as they are made (the N keys are never held
    // at once), then sorted by their bytes. keys: the distinct rows in that order; pos: their ids 0..U-1; occ: the
    // occurrences grouped by row, each group in text order; grp: each group's start (U + 1 entries).
    void distinct_keys(const std::vector<std::pair<std::uint32_t, int>>& words, std::vector<std::int8_t>& keys,
                       std::vector<std::uint32_t>& pos, std::vector<std::uint32_t>& occ, std::vector<std::uint32_t>& grp) const {
        const std::size_t S = static_cast<std::size_t>(stride_);
        constexpr std::uint32_t kNone = ~std::uint32_t{0};
        std::vector<std::int8_t> rows, q(S);
        std::vector<std::uint32_t> table(std::size_t{1} << 16, kNone), gid, at;
        const auto hash = [&](const std::int8_t* x) {
            std::uint64_t h = 0xcbf29ce484222325ull;
            for (std::size_t d = 0; d < S; ++d) h = (h ^ static_cast<std::uint8_t>(x[d])) * 0x100000001b3ull;
            return h;
        };
        const auto place = [&](std::uint32_t g) {  // the empty slot for row g
            std::size_t i = hash(&rows[g * S]) & (table.size() - 1);
            while (table[i] != kNone) i = (i + 1) & (table.size() - 1);
            table[i] = g;
        };
        for (std::size_t w = 0; w < words.size(); ++w) {
            if (!key_of(window(words, w), q.data())) continue;
            std::size_t i = hash(q.data()) & (table.size() - 1);
            while (table[i] != kNone && std::memcmp(&rows[table[i] * S], q.data(), S) != 0) i = (i + 1) & (table.size() - 1);
            if (table[i] == kNone) {
                table[i] = static_cast<std::uint32_t>(rows.size() / S);
                rows.insert(rows.end(), q.begin(), q.end());
                if (rows.size() / S * 2 > table.size()) {  // keep the load under one half
                    table.assign(table.size() * 2, kNone);
                    for (std::uint32_t g = 0; g < rows.size() / S; ++g) place(g);
                }
                gid.push_back(static_cast<std::uint32_t>(rows.size() / S - 1));
            } else {
                gid.push_back(table[i]);
            }
            at.push_back(words[w].first);
        }
        std::vector<std::uint32_t>().swap(table);
        const std::size_t U = rows.size() / S;
        std::vector<std::uint32_t> order(U), rank(U);
        for (std::uint32_t g = 0; g < U; ++g) order[g] = g;
        std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
            const int c = std::memcmp(&rows[a * S], &rows[b * S], S);
            return c ? c < 0 : a < b;
        });
        keys.resize(U * S);
        for (std::size_t r = 0; r < U; ++r) {
            rank[order[r]] = static_cast<std::uint32_t>(r);
            std::memcpy(&keys[r * S], &rows[order[r] * S], S);
        }
        std::vector<std::int8_t>().swap(rows);
        grp.assign(U + 1, 0);  // counting sort of the occurrences by their row's rank, stable in text order
        for (const std::uint32_t g : gid) ++grp[rank[g] + std::size_t{1}];
        for (std::size_t r = 0; r < U; ++r) grp[r + 1] += grp[r];
        occ.resize(at.size());
        std::vector<std::uint32_t> fill(grp.begin(), grp.end() - 1);
        for (std::size_t i = 0; i < at.size(); ++i) occ[fill[rank[gid[i]]]++] = at[i];
        pos.resize(U);
        for (std::size_t g = 0; g < U; ++g) pos[g] = static_cast<std::uint32_t>(g);
    }

    // A4c, after the lists are made: the distinct keys' occurrences in list order (grp_), each key's sorted by the bytes
    // after it (max_tail + 1 of them; the shorter first at the text's end).
    void group_occurrences(std::vector<std::uint32_t>& pos, const std::vector<std::uint32_t>& occ, const std::vector<std::uint32_t>& grp) {
        sort_tail_ = p_.max_tail;
        const std::size_t L = static_cast<std::size_t>(sort_tail_) + 1;
        const auto less = [&](std::uint32_t a, std::uint32_t b) {
            const std::size_t la = std::min(L, n_ - a), lb = std::min(L, n_ - b);
            const int c = std::memcmp(text_ + a, text_ + b, std::min(la, lb));
            return c ? c < 0 : la != lb ? la < lb : a < b;
        };
        std::vector<std::uint32_t> o;
        o.reserve(occ.size());
        grp_.assign(pos.size() + 1, 0);
        for (std::size_t s = 0; s < pos.size(); ++s) {
            grp_[s] = static_cast<std::uint32_t>(o.size());
            o.insert(o.end(), occ.begin() + grp[pos[s]], occ.begin() + grp[pos[s] + 1]);
        }
        grp_[pos.size()] = static_cast<std::uint32_t>(o.size());
        pos = std::move(o);
        for (std::size_t g = 0; g + 1 < grp_.size(); ++g) std::sort(pos.begin() + grp_[g], pos.begin() + grp_[g + 1], less);
    }

 public:
    /// The next-byte distribution after h[0..n) (oldest byte first): the live neighbours' proposals, summing to 1, or
    /// all zero when no neighbour is live (live = 0, bucket 0).
    Hit predict(const std::uint8_t* h, std::size_t n, double P[256]) const {
        if (p_.mode) return predict_pooled(h, n, P);  // A4c
        std::fill(P, P + 256, 0.0);
        Hit hit;
        std::size_t anchor = 0, ctx = 0;
        if (!anchor_of(h, n, anchor, ctx, hit.word)) return hit;
        ingest(h, anchor);  // A4d
        search_cached(h, ctx, anchor);
        // The live neighbours: when this tail extends the last call's, only that call's survivors are checked.
        const std::size_t t = n - anchor;
        const std::uint8_t* tail = h + anchor;
        if (restart_ || t < tail_.size() || !std::equal(tail_.begin(), tail_.end(), tail)) {
            live_.resize(nb_pos_.size());
            for (std::size_t j = 0; j < live_.size(); ++j) live_[j] = static_cast<std::uint32_t>(j);
            tail_.clear();
            restart_ = false;
        }
        const std::size_t from = tail_.size();
        std::size_t m = 0;
        for (const std::uint32_t j : live_) {
            const std::uint32_t e = nb_pos_[j], pos = e & ~kOnline;
            const std::uint8_t* src = e & kOnline ? h : text_;  // A4d: an online key reads the stream (pos + t < n)
            if ((e & kOnline || pos + t < n_) && std::memcmp(src + pos + from, tail + from, t - from) == 0) live_[m++] = j;
        }
        live_.resize(m);
        tail_.assign(tail, tail + t);
        if (m == 0) return hit;
        double wsum = 0, mx = 0;
        for (const std::uint32_t j : live_) {
            const std::uint32_t e = nb_pos_[j];
            P[(e & kOnline ? h : text_)[(e & ~kOnline) + t]] += nb_w_[j];
            wsum += nb_w_[j];
        }
        for (int s = 0; s < 256; ++s) mx = std::max(mx, P[s] /= wsum);
        hit.live = static_cast<int>(m);
        hit.top = nb_sim_[live_[0]];  // neighbours are kept best first
        hit.agree = mx;
        return hit;
    }

    /// A4c (dedupe): the distinct keys found for the anchor, each pooling its occurrences that agree with the tail,
    /// weighted as a neighbour is. Live = live occurrences.
    Hit predict_pooled(const std::uint8_t* h, std::size_t n, double P[256]) const {
        std::fill(P, P + 256, 0.0);
        Hit hit;
        std::size_t anchor = 0, ctx = 0;
        if (!anchor_of(h, n, anchor, ctx, hit.word)) return hit;
        ingest(h, anchor);  // A4d
        search_cached(h, ctx, anchor);
        const std::size_t t = n - anchor;
        const std::uint8_t* tail = h + anchor;
        if (restart_ || t < tail_.size() || !std::equal(tail_.begin(), tail_.end(), tail)) {
            live_.resize(nb_pos_.size());
            glo_.resize(nb_pos_.size());
            ghi_.resize(nb_pos_.size());
            for (std::size_t j = 0; j < live_.size(); ++j) {
                live_[j] = static_cast<std::uint32_t>(j);
                if (nb_pos_[j] & kOnline) continue;  // A4d: an online key is one occurrence, read from the stream
                glo_[j] = grp_[nb_pos_[j]];
                ghi_[j] = grp_[nb_pos_[j] + std::size_t{1}];
            }
            tail_.clear();
            restart_ = false;
        }
        const std::size_t from = tail_.size();
        const auto cmp = [&](std::uint32_t pos) {  // the bytes [from, t) after pos against the tail's, shorter first
            const std::size_t want = t - from, avail = pos + from < n_ ? std::min(want, n_ - pos - from) : 0;
            const int c = avail ? std::memcmp(text_ + pos + from, tail + from, avail) : 0;
            return c ? c : avail < want ? -1 : 0;
        };
        std::size_t m = 0;
        for (const std::uint32_t j : live_) {
            if (nb_pos_[j] & kOnline) {  // A4d: live while the stream after it agrees with the tail
                if (std::memcmp(h + (nb_pos_[j] & ~kOnline) + from, tail + from, t - from) == 0) live_[m++] = j;
                continue;
            }
            auto lo = pos_.begin() + glo_[j], hi = pos_.begin() + ghi_[j];
            if (t > from) {
                lo = std::partition_point(lo, hi, [&](std::uint32_t p) { return cmp(p) < 0; });
                hi = std::partition_point(lo, hi, [&](std::uint32_t p) { return cmp(p) <= 0; });
            }
            while (lo < hi && *lo + t >= n_) ++lo;  // no next byte: these sort first
            if (lo == hi) continue;
            glo_[j] = static_cast<std::uint32_t>(lo - pos_.begin());
            ghi_[j] = static_cast<std::uint32_t>(hi - pos_.begin());
            live_[m++] = j;
        }
        live_.resize(m);
        tail_.assign(tail, tail + t);
        if (m == 0) return hit;
        double wsum = 0, mx = 0;
        std::size_t occ = 0;
        for (const std::uint32_t j : live_) {
            const double w = nb_w_[j];
            if (nb_pos_[j] & kOnline) {  // A4d
                P[h[(nb_pos_[j] & ~kOnline) + t]] += w;
                wsum += w;
                ++occ;
                continue;
            }
            for (auto i = pos_.begin() + glo_[j], e = pos_.begin() + ghi_[j]; i < e;) {
                const std::uint8_t b = text_[*i + t];
                const auto f = std::partition_point(i, e, [&](std::uint32_t p) { return text_[p + t] <= b; });
                P[b] += w * static_cast<double>(f - i);
                i = f;
            }
            wsum += w * static_cast<double>(ghi_[j] - glo_[j]);
            occ += ghi_[j] - glo_[j];
        }
        for (int s = 0; s < 256; ++s) mx = std::max(mx, P[s] /= wsum);
        hit.live = static_cast<int>(std::min<std::size_t>(occ, std::numeric_limits<int>::max()));
        hit.top = nb_sim_[live_[0]];  // groups are kept best first
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
    /// A4c: the keys stored: distinct keys (dedupe), all keys (knn).
    std::size_t contexts() const { return keys_.size() / static_cast<std::size_t>(stride_); }
    /// The index and the word coordinates (keys, positions, centroids, list starts, rows); the word map not counted.
    std::size_t bytes() const {
        return keys_.size() + pos_.size() * 4 + grp_.size() * 4 + cent_.size() + start_.size() * 4 + proj_.size() * 4 + vec_.q.size() +
               vec_.scale.size() * 4 + vec_.v.size() * 4 + online_bytes();  // A4d: the online keys

    }
    int dim() const { return dim_; }
    int lists() const { return lists_; }
    std::size_t queries() const { return lat_us_.size(); }
    /// Search latency quantile q in [0, 1], microseconds (every search of this expert so far, or searches [from, to)).
    double latency_us(double q) const { return latency_us(q, 0, lat_us_.size()); }
    double latency_us(double q, std::size_t from, std::size_t to) const {
        to = std::min(to, lat_us_.size());
        if (from >= to) return 0;
        std::vector<float> v(lat_us_.begin() + static_cast<std::ptrdiff_t>(from), lat_us_.begin() + static_cast<std::ptrdiff_t>(to));
        const std::size_t i = std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())));
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
        return v[i];
    }
    /// A4e diagnostics: the online keys held now (every word end of the stream so far) and the bytes their arrays hold
    /// (capacity, as allocated).
    std::size_t online_keys() const { return on_count_; }
    /// A4f: the online keys stored (distinct under merge; every one without it).
    std::size_t online_distinct() const {
        std::size_t d = 0;
        for (const auto& v : on_pos_) d += v.size();
        return d;
    }
    /// The bytes the online keys hold: a row and a position per key; under merge also the sorted index, the repeats
    /// and their owners' starts (A4f).
    std::size_t online_bytes() const {
        if (!p_.merge) return on_count_ * (static_cast<std::size_t>(stride_) + 4);
        std::size_t b = 0;
        for (const auto& v : on_keys_) b += v.size();
        for (const auto& v : on_pos_) b += v.size() * 4;
        for (const auto& v : on_ord_) b += v.size() * 4;
        for (const auto& v : rep_) b += v.size() * 4;
        for (const auto& v : rstart_) b += v.size() * 4;
        return b;
    }
    std::size_t online_alloc_bytes() const {
        std::size_t b = 0;
        for (const auto& v : on_keys_) b += v.capacity();
        for (const auto& v : on_pos_) b += v.capacity() * 4;
        for (const auto& v : on_ord_) b += v.capacity() * 4;  // A4f (empty without merge)
        for (const auto& v : rep_) b += v.capacity() * 4;
        for (const auto& v : rstart_) b += v.capacity() * 4;
        return b;
    }
    /// A4e diagnostics, for merging: of the online keys held, how many equal a stored key (dedupe: a distinct key) and
    /// how many equal an earlier held online key but no stored one.
    std::pair<std::size_t, std::size_t> online_duplicates() const {
        if (p_.merge) return {rep_stored_, rep_online_};  // A4f: the repeats merged, counted as they come
        const std::size_t S = static_cast<std::size_t>(stride_);
        const auto hash = [S](const std::int8_t* x) {
            std::uint64_t h = 0xcbf29ce484222325ull;
            for (std::size_t d = 0; d < S; ++d) h = (h ^ static_cast<std::uint8_t>(x[d])) * 0x100000001b3ull;
            return h;
        };
        std::unordered_map<std::uint64_t, char> seen;  // 1 stored, 2 online
        for (std::size_t r = 0; r * S < keys_.size(); ++r) seen.emplace(hash(&keys_[r * S]), 1);
        std::vector<std::pair<std::uint32_t, const std::int8_t*>> held;  // in stream order
        for (std::size_t c = 0; c < on_pos_.size(); ++c)
            for (std::size_t i = 0; i < on_pos_[c].size(); ++i) held.push_back({on_pos_[c][i], &on_keys_[c][i * S]});
        std::sort(held.begin(), held.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        std::size_t stored = 0, online = 0;
        for (const auto& [p, x] : held) {
            const auto [it, fresh] = seen.emplace(hash(x), 2);
            if (!fresh) ++(it->second == 1 ? stored : online);
        }
        return {stored, online};
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

    // Row r's coordinate d: the PCA projection (unit rows) or the stored int8 row times its scale (projected rows).
    float proj(std::size_t r, int d) const {
        return vec_.projected() ? vec_.scale[r] * vec_.q[r * static_cast<std::size_t>(vec_.d) + static_cast<std::size_t>(d)]
                                : proj_[r * static_cast<std::size_t>(dim_) + static_cast<std::size_t>(d)];
    }

    // The weighted mean of the known words' coordinates (x[0..dim)). False if none is known.
    bool proj_mean(const std::vector<int>& ids, float* x) const {
        std::fill(x, x + dim_, 0.0f);
        double m = 0;
        for (int id : ids) {
            if (id < 0) continue;
            const float a = weight_[static_cast<std::size_t>(id)];
            for (int d = 0; d < dim_; ++d) x[d] += a * proj(static_cast<std::size_t>(id), d);
            ++m;
        }
        if (m == 0) return false;
        for (int d = 0; d < dim_; ++d) x[d] = static_cast<float>(x[d] / m);
        return true;
    }

    // The key of a context from its word ids (unknown = -1): the projected weighted mean, centred, unit, as int8.
    bool key_of(const std::vector<int>& ids, std::int8_t* q) const {
        std::vector<float> x(static_cast<std::size_t>(stride_), 0.0f);
        if (!proj_mean(ids, x.data())) return false;
        for (int d = 0; d < dim_; ++d) x[static_cast<std::size_t>(d)] -= mu_proj_[static_cast<std::size_t>(d)];
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
        int best = 0, bs = std::numeric_limits<int>::min();
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

    void search(const std::int8_t* q, int probe, std::vector<std::uint32_t>& p, std::vector<float>& s, bool online = false) const {
        p.clear();
        s.clear();
        if (lists_ == 0) return;
        std::vector<std::pair<int, int>> cs(static_cast<std::size_t>(lists_));
        for (int c = 0; c < lists_; ++c) cs[static_cast<std::size_t>(c)] = {dot(q, &cent_[static_cast<std::size_t>(c) * stride_]), c};
        const int np = std::clamp(probe, 1, lists_);
        std::partial_sort(cs.begin(), cs.begin() + np, cs.end(), [](auto& x, auto& y) { return x.first > y.first; });
        std::vector<std::pair<int, std::uint32_t>> cand;
        const bool merged = online && p_.merge && !on_pos_.empty();  // A4f
        if (merged) on_found_.clear();
        for (int l = 0; l < np; ++l) {
            const int c = cs[static_cast<std::size_t>(l)].second;
            for (std::uint32_t i = start_[static_cast<std::size_t>(c)]; i < start_[static_cast<std::size_t>(c) + 1]; ++i)
                cand.push_back({dot(q, &keys_[static_cast<std::size_t>(i) * stride_]), i});
            if (online && !on_pos_.empty())  // A4d: this list's online keys, their stream positions marked kOnline
                for (std::size_t i = 0; i < on_pos_[static_cast<std::size_t>(c)].size(); ++i) {
                    const int d = dot(q, &on_keys_[static_cast<std::size_t>(c)][i * static_cast<std::size_t>(stride_)]);
                    if (!merged) {
                        cand.push_back({d, on_pos_[static_cast<std::size_t>(c)][i]});
                        continue;
                    }
                    cand.push_back({d, kOnline | static_cast<std::uint32_t>(on_found_.size())});  // A4f: (list, key)
                    on_found_.push_back({static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(i)});
                }
        }
        const std::size_t k = std::min<std::size_t>(cand.size(), static_cast<std::size_t>(p_.k));
        std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(k), cand.end(),
                          [](auto& x, auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
        for (std::size_t j = 0; j < k; ++j) {
            const float sim = static_cast<float>(cand[j].first / (127.0 * 127.0));
            if (sim < p_.min_sim) break;
            if (merged) {  // A4f: the key, then every repeat merged into it, each a position with the key's cosine
                std::size_t L = 0, owner = 0;
                if (cand[j].second & kOnline) {
                    const auto [c, i] = on_found_[cand[j].second & ~kOnline];
                    L = c;
                    owner = static_cast<std::size_t>(start_[L + 1] - start_[L]) + i;
                    p.push_back(on_pos_[L][i]);
                } else {
                    L = static_cast<std::size_t>(std::upper_bound(start_.begin(), start_.end(), cand[j].second) - start_.begin()) - 1;
                    owner = cand[j].second - start_[L];
                    p.push_back(cand[j].second);
                }
                s.push_back(sim);
                const auto& r = rstart_[L];
                for (std::uint32_t x = r[owner]; x < r[owner + 1]; ++x) {
                    p.push_back(rep_[L][x]);
                    s.push_back(sim);
                }
                continue;
            }
            p.push_back(cand[j].second & kOnline ? cand[j].second  // A4d: an online key's stream position
                        : p_.mode == 1 ? cand[j].second : pos_[cand[j].second]);  // A4c dedupe: the distinct key's slot
            s.push_back(sim);
        }
    }

    // A4d: the online keys start again for stream h (nullptr: none yet).
    void reset_online(const std::uint8_t* h, std::size_t from) const {
        // Fresh vectors, so the memory of the keys held before is given back (assign would keep each one's capacity).
        const std::size_t nl = p_.online ? static_cast<std::size_t>(lists_) : 0;
        std::vector<std::vector<std::int8_t>>(nl).swap(on_keys_);
        std::vector<std::vector<std::uint32_t>>(nl).swap(on_pos_);
        on_count_ = 0;
        const bool m = p_.online && p_.merge;  // A4f: each list's stored keys own an empty run of repeats
        std::vector<std::vector<std::uint32_t>>(m ? nl : 0).swap(on_ord_);
        std::vector<std::vector<std::uint32_t>>(m ? nl : 0).swap(rep_);
        std::vector<std::vector<std::uint32_t>>(m ? nl : 0).swap(rstart_);
        for (std::size_t L = 0; L < rstart_.size(); ++L) rstart_[L].assign(static_cast<std::size_t>(start_[L + 1] - start_[L]) + 1, 0);
        rep_stored_ = rep_online_ = 0;
        on_h_ = h;
        on_upto_ = from;
        on_sig_.clear();
        cache_ = ~std::uint64_t{0};
    }

    // A4f: key q of list L at stream position pos (kOnline set). If q equals a stored key of L or an earlier online key
    // of L (every byte), pos joins that key's repeats and true is returned. Otherwise q is a new key: its place in L's
    // sorted index and its owner's empty run are made, and false is returned (the caller stores its row and position).
    bool merge_online(std::size_t L, const std::int8_t* q, std::uint32_t pos) const {
        const std::size_t S = static_cast<std::size_t>(stride_);
        const auto add = [&](std::size_t owner) {  // pos at the end of owner's run; every later run moves one on
            auto& r = rstart_[L];
            rep_[L].insert(rep_[L].begin() + r[owner + 1], pos);
            for (std::size_t x = owner + 1; x < r.size(); ++x) ++r[x];
        };
        std::uint32_t lo = start_[L], hi = start_[L + 1];  // the stored keys of L, sorted by their bytes (dedupe)
        while (lo < hi) {
            const std::uint32_t mid = lo + (hi - lo) / 2;
            if (std::memcmp(&keys_[static_cast<std::size_t>(mid) * S], q, S) < 0) lo = mid + 1;
            else hi = mid;
        }
        if (lo < start_[L + 1] && std::memcmp(&keys_[static_cast<std::size_t>(lo) * S], q, S) == 0) {
            add(lo - start_[L]);
            ++rep_stored_;
            return true;
        }
        auto& ord = on_ord_[L];
        const auto& rows = on_keys_[L];
        const auto it = std::lower_bound(ord.begin(), ord.end(), q, [&](std::uint32_t i, const std::int8_t* x) {
            return std::memcmp(&rows[static_cast<std::size_t>(i) * S], x, S) < 0;
        });
        if (it != ord.end() && std::memcmp(&rows[static_cast<std::size_t>(*it) * S], q, S) == 0) {
            add(static_cast<std::size_t>(start_[L + 1] - start_[L]) + *it);
            ++rep_online_;
            return true;
        }
        ord.insert(it, static_cast<std::uint32_t>(on_pos_[L].size()));
        rstart_[L].push_back(rstart_[L].back());
        return false;
    }

    // A4d: the keys at the stream's word ends j with on_upto_ <= j < anchor, each in its nearest list. The stream is the
    // stored text's buffer from its end on, or h from its start; another buffer, a shorter stream or other bytes just
    // before on_upto_ start the keys again.
    void ingest(const std::uint8_t* h, std::size_t anchor) const {
        if (!p_.online || lists_ == 0) return;
        const std::size_t from = h == text_ ? n_ : 0;
        const std::size_t sig = std::min<std::size_t>(on_upto_ - std::min(on_upto_, from), 32);
        if (h != on_h_ || anchor < on_upto_ || on_upto_ < from || !std::equal(on_sig_.begin(), on_sig_.end(), h + on_upto_ - sig))
            reset_online(h, from);
        std::vector<std::int8_t> q(static_cast<std::size_t>(stride_));
        for (std::size_t j = std::max<std::size_t>(on_upto_, 1); j < anchor; ++j) {
            if (!is_word(h[j - 1]) || is_word(h[j])) continue;  // j is a word end
            std::size_t i = j, w = 0;
            const std::size_t lim = j > 1024 ? j - 1024 : 0;
            while (i > lim && w < static_cast<std::size_t>(p_.W)) {  // back W words, as anchor_of
                while (i > lim && is_word(h[i - 1])) --i;
                ++w;
                while (i > lim && w < static_cast<std::size_t>(p_.W) && !is_word(h[i - 1])) --i;
            }
            if (!query_key(h, i, j, q.data())) continue;
            const std::size_t L = static_cast<std::size_t>(nearest(q.data(), lists_));
            ++on_count_;
            if (p_.merge && merge_online(L, q.data(), static_cast<std::uint32_t>(j) | kOnline)) continue;  // A4f
            on_keys_[L].insert(on_keys_[L].end(), q.begin(), q.end());
            on_pos_[L].push_back(static_cast<std::uint32_t>(j) | kOnline);
        }
        on_upto_ = std::max(on_upto_, anchor);
        const std::size_t s2 = std::min<std::size_t>(on_upto_ - from, 32);
        on_sig_.assign(h + on_upto_ - s2, h + on_upto_);
    }

    void search_cached(const std::uint8_t* h, std::size_t ctx, std::size_t anchor) const {
        std::uint64_t key = 0xcbf29ce484222325ull ^ (anchor - ctx);
        for (std::size_t i = ctx; i < anchor; ++i) key = (key ^ h[i]) * 0x100000001b3ull;
        if (p_.online) key = (key ^ on_count_) * 0x100000001b3ull;  // A4d: new online keys mean a new search
        if (key == cache_) return;
        cache_ = key;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::int8_t> q(static_cast<std::size_t>(stride_));
        nb_pos_.clear();
        nb_sim_.clear();
        if (query_key(h, ctx, anchor, q.data())) search(q.data(), p_.probe, nb_pos_, nb_sim_, p_.online);
        nb_w_.resize(nb_sim_.size());  // exp((cos - cos_best) / temp); the shift cancels when the live ones are normalised
        for (std::size_t j = 0; j < nb_w_.size(); ++j) nb_w_[j] = p_.temp > 0 ? std::exp((double{nb_sim_[j]} - double{nb_sim_[0]}) / p_.temp) : 1.0;
        restart_ = true;  // predict() starts again from every neighbour
        lat_us_.push_back(static_cast<float>(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()));
    }

    Vectors vec_;
    Params p_;
    int dim_ = 0, stride_ = 16, lists_ = 0;
    const std::uint8_t* text_ = nullptr;
    std::size_t n_ = 0;
    std::size_t rows_ = 0;
    std::vector<float> weight_, proj_, mu_proj_;  // proj_: the unit rows' projections (empty for projected rows)
    std::vector<std::int8_t> keys_, cent_;  // keys grouped by list; centroids
    std::vector<std::uint32_t> pos_, start_;
    std::vector<std::uint32_t> grp_;  // A4c dedupe: each distinct key's occurrences in pos_, U + 1 starts
    int sort_tail_ = 0;               // A4c: the tail depth each key's occurrences are sorted by
    mutable std::uint64_t cache_ = ~std::uint64_t{0};  // the last searched context, and its neighbours
    mutable std::vector<std::uint32_t> nb_pos_, live_;  // live_: the neighbours that agree with tail_
    mutable std::vector<std::uint32_t> glo_, ghi_;      // A4c: each key's range of occurrences that agrees with tail_
    mutable std::vector<float> nb_sim_, lat_us_;
    mutable std::vector<double> nb_w_;
    mutable std::vector<std::uint8_t> tail_;
    mutable bool restart_ = true;
    static constexpr std::uint32_t kOnline = 0x80000000u;  // A4d: marks an online key's stream position
    mutable std::vector<std::vector<std::int8_t>> on_keys_;  // A4d: online keys by list
    mutable std::vector<std::vector<std::uint32_t>> on_pos_;
    mutable std::size_t on_count_ = 0, on_upto_ = 0;
    mutable const std::uint8_t* on_h_ = nullptr;
    mutable std::vector<std::uint8_t> on_sig_;  // the stream's bytes just before on_upto_
    mutable std::vector<std::vector<std::uint32_t>> on_ord_;  // A4f: each list's online keys, sorted by their bytes
    mutable std::vector<std::vector<std::uint32_t>> rep_;     // A4f: each list's repeats, grouped by owner
    mutable std::vector<std::vector<std::uint32_t>> rstart_;  // A4f: each owner's run in rep_ (stored keys, then online)
    mutable std::vector<std::pair<std::uint32_t, std::uint32_t>> on_found_;  // A4f: a search's online keys (list, key)
    mutable std::size_t rep_stored_ = 0, rep_online_ = 0;     // A4f: repeats merged into stored / online keys
};

}  // namespace cypha::cyphalm
