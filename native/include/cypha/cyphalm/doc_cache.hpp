#pragma once

/// Long-range context without a network (PLAN B5; generation track step 3, GEN-C): next-byte experts that keep
/// CyphaLM on the document's subject across sentences. Report: docs/reports/GEN_TRACK.md, "GEN-C".
///
/// Three parts, each a next-byte expert that the caller mixes in with one lambda a bucket, like InfiniGram:
/// - **Triggers** (Rosenfeld 1996): a content word earlier in the document raises the odds of the words it triggers.
///   Trigger pairs come from the training bytes only: two content words that share at least `min_co` training
///   documents, scored by local mutual information co * ln(co N / (df_a df_b)), the `per_source` best kept per word.
/// - **Cache** (Kuhn & De Mori 1990): content words already in the document are likely to come again.
/// - **Topic** (optional): the mean of the document's word vectors (the similar-context expert's projected PPMI-SVD
///   rows, data/cyphalm/simvec_ppmi32.bin) picks the `topic_k` nearest content words.
///
/// Each part keeps a mass over candidate words. At a position, the bytes since the last non-letter are the partial
/// word; the part predicts the next byte from the candidates that begin with it (compared without case), each one
/// proposing its next letter in its most frequent training spelling, or, when the candidate is the whole partial
/// word, a word-ending byte from the training distribution of bytes after a word. The part never writes a word or a
/// passage: it proposes one byte of a word, and the mixture decides (PLAN rule 10).
///
/// Words are runs of ASCII letters. Content words have at least `min_len` letters and are not among the `stop_top`
/// words that most training documents contain (a stop list from counts, not a hand list). A document is the text
/// from the last "<page>"; a stream is reset there. Nothing is fetched: everything comes from the training bytes and
/// the vector file.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cypha::cyphalm {

class DocTopic {
 public:
    struct Params {
        int min_len = 3;          ///< content words: at least this many letters
        int stop_top = 200;       ///< the words in the most training documents are not content words
        int min_df = 3;           ///< trigger sources and targets: in at least this many training documents
        int min_co = 3;           ///< a trigger pair: in at least this many training documents together
        int per_source = 32;      ///< triggers kept a source word
        int max_doc_words = 100;  ///< per training document, the content words (best tf-idf) that form pairs
        int cand = 1024;          ///< candidate words a part keeps (the largest masses)
        bool trig = true, cache = true, topic = false;
        int topic_k = 512;        ///< topic candidates
        double topic_temp = 0.05; ///< topic mass exp((cos - cos_max) / temp)
        int topic_every = 4;      ///< topic candidates recomputed every this many content words
        bool frozen = false;      ///< sample: the document state comes from the prompt only (Stream::freeze)
    };
    enum Part { kTrig = 0, kCache = 1, kTopic = 2, kParts = 3 };
    static constexpr int kBuckets = 1 + 5 * 4;  ///< 0: no candidate; then kind (5) x confidence (4)

    DocTopic() = default;
    explicit DocTopic(Params p) : p_(p) {}
    const Params& params() const { return p_; }

    /// Word vectors for the topic part: rows by lowercase word, unit or not (they are normalised here).
    void set_vectors(int dim, std::unordered_map<std::string, std::vector<float>> rows) {
        vdim_ = dim;
        vrows_ = std::move(rows);
        for (auto& [w, r] : vrows_) {
            double nn = 0;
            for (float x : r) nn += static_cast<double>(x) * x;
            if (nn > 0)
                for (float& x : r) x = static_cast<float>(x / std::sqrt(nn));
        }
    }

    /// Learn the vocabulary, the stop list, the spellings, the word-ending bytes and the trigger pairs from text[0, n).
    void build(const std::uint8_t* text, std::size_t n) {
        // pass 1: documents, words, document frequency, spellings, bytes after a word
        std::vector<std::pair<std::size_t, std::size_t>> docs = split_docs(text, n);
        std::unordered_map<std::string, int> wid;
        std::vector<std::string> words;
        std::vector<int> df;
        std::vector<std::unordered_map<std::string, int>> spell;
        double after[256] = {0};
        std::vector<int> seen;  // last doc a word was seen in
        for (std::size_t d = 0; d < docs.size(); ++d) {
            std::size_t i = docs[d].first;
            const std::size_t e = docs[d].second;
            while (i < e) {
                if (!alpha(text[i])) {
                    ++i;
                    continue;
                }
                std::size_t j = i;
                while (j < e && alpha(text[j])) ++j;
                if (j - i <= 40) {
                    std::string surf(reinterpret_cast<const char*>(text + i), j - i), low = lower(surf);
                    auto it = wid.find(low);
                    int id;
                    if (it == wid.end()) {
                        id = static_cast<int>(words.size());
                        wid.emplace(low, id);
                        words.push_back(low);
                        df.push_back(0);
                        seen.push_back(-1);
                        spell.emplace_back();
                    } else id = it->second;
                    if (seen[static_cast<std::size_t>(id)] != static_cast<int>(d)) {
                        seen[static_cast<std::size_t>(id)] = static_cast<int>(d);
                        ++df[static_cast<std::size_t>(id)];
                    }
                    ++spell[static_cast<std::size_t>(id)][surf];
                    if (j < e) after[text[j]] += 1;
                }
                i = j;
            }
        }
        const std::size_t V = words.size();
        ndocs_ = docs.size();
        // the stop list: the stop_top words with the largest document frequency
        std::vector<int> order(V);
        for (std::size_t k = 0; k < V; ++k) order[k] = static_cast<int>(k);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return df[static_cast<std::size_t>(a)] != df[static_cast<std::size_t>(b)]
                       ? df[static_cast<std::size_t>(a)] > df[static_cast<std::size_t>(b)]
                       : words[static_cast<std::size_t>(a)] < words[static_cast<std::size_t>(b)];
        });
        content_.assign(V, 1);
        for (std::size_t k = 0; k < V && static_cast<int>(k) < p_.stop_top; ++k) content_[static_cast<std::size_t>(order[k])] = 0;
        for (std::size_t k = 0; k < V; ++k)
            if (static_cast<int>(words[k].size()) < p_.min_len) content_[k] = 0;
        // spellings: the most frequent surface form (ties: the smaller string)
        surf_.assign(V, std::string());
        for (std::size_t k = 0; k < V; ++k) {
            int best = -1;
            for (const auto& [s, c] : spell[k])
                if (c > best || (c == best && s < surf_[k])) best = c, surf_[k] = s;
        }
        words_ = std::move(words);
        wid_ = std::move(wid);
        df_ = std::move(df);
        double tot = 0;
        for (int b = 0; b < 256; ++b) tot += after[b];
        for (int b = 0; b < 256; ++b) end_[b] = tot > 0 ? after[b] / tot : 0;
        // pass 2: trigger pairs from each document's best content words
        trig_.assign(V, {});
        if (p_.trig) build_triggers(text, docs);
        // topic: the content words that have a vector, normalised
        topic_ids_.clear();
        topic_rows_.clear();
        if (p_.topic && vdim_ > 0) {
            for (std::size_t k = 0; k < V; ++k) {
                if (!content_[k] || df_[k] < p_.min_df) continue;
                auto it = vrows_.find(words_[k]);
                if (it == vrows_.end()) continue;
                double nn = 0;
                for (float x : it->second) nn += static_cast<double>(x) * x;
                if (nn <= 0) continue;
                const float inv = static_cast<float>(1.0 / std::sqrt(nn));
                topic_ids_.push_back(static_cast<int>(k));
                for (float x : it->second) topic_rows_.push_back(x * inv);
            }
        }
    }

    std::size_t vocab() const { return words_.size(); }
    std::size_t docs() const { return ndocs_; }
    std::size_t trigger_pairs() const {
        std::size_t s = 0;
        for (const auto& t : trig_) s += t.size();
        return s;
    }
    std::size_t topic_words() const { return topic_ids_.size(); }
    int id(const std::string& lower_word) const {
        auto it = wid_.find(lower_word);
        return it == wid_.end() ? -1 : it->second;
    }
    bool is_content(int id) const { return id >= 0 && content_[static_cast<std::size_t>(id)]; }
    /// The triggered words of a word (lowercase), best first, with their weights (they sum to 1).
    std::vector<std::pair<std::string, double>> triggers(const std::string& lower_word) const {
        std::vector<std::pair<std::string, double>> out;
        const int i = id(lower_word);
        if (i < 0) return out;
        for (const auto& [t, w] : trig_[static_cast<std::size_t>(i)]) out.emplace_back(words_[static_cast<std::size_t>(t)], w);
        return out;
    }

    /// One reader of one text: feed the bytes in order, ask for the next byte's distribution in between.
    class Stream {
     public:
        explicit Stream(const DocTopic& m) : m_(m) { reset(); }
        void reset() {
            for (auto& q : q_) q.clear();
            for (auto& c : cand_) c.clear();
            for (bool& d : dirty_) d = true;
            partial_.clear();
            prev_ = ' ';
            tail_.clear();
            nwords_ = since_topic_ = 0;
            topic_sum_.assign(static_cast<std::size_t>(m_.vdim_), 0.0f);
            seen_.clear();
            frozen_ = false;
        }
        /// From here on the words read no longer change the parts' masses (the partial word is still followed): in
        /// sample, after the prompt, so the parts cannot feed on what CyphaLM itself writes.
        void freeze() { frozen_ = true; }
        /// After `b`: the partial word grows or ends; a document restarts after "<page>".
        void feed(std::uint8_t b) {
            if (alpha(b)) {
                if (partial_.size() < 40) partial_.push_back(static_cast<char>(std::tolower(b)));
            } else {
                if (!partial_.empty()) on_word(partial_);
                partial_.clear();
            }
            prev_ = b;
            tail_.push_back(static_cast<char>(b));
            if (tail_.size() > 6) tail_.erase(tail_.begin());
            if (tail_ == "<page>") reset();
        }
        void feed(const std::uint8_t* s, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) feed(s[i]);
        }
        /// The part's next-byte distribution in P (sums to 1) and its bucket; bucket 0 (no candidate) gives uniform.
        int predict(int part, double P[256]) {
            refresh(part);
            const auto& C = cand_[static_cast<std::size_t>(part)];
            const std::string& pre = partial_;
            auto lo = std::lower_bound(C.begin(), C.end(), pre, [](const Cand& c, const std::string& s) { return c.low < s; });
            double m = 0, tot = total_[part];
            std::fill(P, P + 256, 0.0);
            for (auto it = lo; it != C.end() && it->low.compare(0, pre.size(), pre) == 0; ++it) {
                m += it->mass;
                if (it->low.size() > pre.size()) {
                    const std::string& sf = m_.surf_[static_cast<std::size_t>(it->id)];
                    std::uint8_t c = static_cast<std::uint8_t>(sf.size() > pre.size() ? sf[pre.size()] : it->low[pre.size()]);
                    P[c] += it->mass;
                } else
                    for (int b = 0; b < 256; ++b) P[b] += it->mass * m_.end_[b];
            }
            if (!(m > 0)) {
                for (int b = 0; b < 256; ++b) P[b] = 1.0 / 256;
                return 0;
            }
            double s = 0;
            for (int b = 0; b < 256; ++b) s += P[b];
            for (int b = 0; b < 256; ++b) P[b] /= s;
            int kind;
            if (pre.empty()) kind = prev_ == ' ' ? 0 : 1;
            else kind = pre.size() == 1 ? 2 : pre.size() == 2 ? 3 : 4;
            int conf;
            if (pre.empty()) {
                const std::size_t w = seen_.size();
                conf = w <= 1 ? 0 : w <= 4 ? 1 : w <= 16 ? 2 : 3;
            } else {
                const double f = tot > 0 ? m / tot : 0;
                conf = f < 0.01 ? 0 : f < 0.05 ? 1 : f < 0.2 ? 2 : 3;
            }
            return 1 + kind * 4 + conf;
        }
        std::size_t candidates(int part) {
            refresh(part);
            return cand_[static_cast<std::size_t>(part)].size();
        }

     private:
        struct Cand {
            std::string low;
            int id;
            double mass;
        };
        void on_word(const std::string& low) {
            if (frozen_) return;
            const int id = m_.id(low);
            if (!m_.is_content(id)) return;
            ++nwords_;
            const bool first = seen_.emplace(id, 1).second;
            if (!first) ++seen_[id];
            if (m_.p_.cache) {
                q_[kCache][id] += 1;
                dirty_[kCache] = true;
            }
            if (m_.p_.trig && first) {
                for (const auto& [t, w] : m_.trig_[static_cast<std::size_t>(id)]) q_[kTrig][t] += w;
                if (!m_.trig_[static_cast<std::size_t>(id)].empty()) dirty_[kTrig] = true;
            }
            if (m_.p_.topic && m_.vdim_ > 0) {
                auto it = m_.vrows_.find(low);
                if (it != m_.vrows_.end()) {
                    for (int d = 0; d < m_.vdim_; ++d) topic_sum_[static_cast<std::size_t>(d)] += it->second[static_cast<std::size_t>(d)];
                    if (++since_topic_ >= m_.p_.topic_every || cand_[kTopic].empty()) dirty_[kTopic] = true, since_topic_ = 0;
                }
            }
        }
        void refresh(int part) {
            if (!dirty_[part]) return;
            dirty_[part] = false;
            auto& C = cand_[static_cast<std::size_t>(part)];
            C.clear();
            total_[part] = 0;
            std::vector<std::pair<double, int>> all;
            if (part == kTopic) {
                if (!m_.p_.topic || m_.topic_ids_.empty()) return;
                double nn = 0;
                for (float x : topic_sum_) nn += static_cast<double>(x) * x;
                if (nn <= 0) return;
                const double inv = 1.0 / std::sqrt(nn);
                const std::size_t D = static_cast<std::size_t>(m_.vdim_);
                std::vector<std::pair<double, int>> cs;
                cs.reserve(m_.topic_ids_.size());
                for (std::size_t r = 0; r < m_.topic_ids_.size(); ++r) {
                    const float* row = m_.topic_rows_.data() + r * D;
                    double dot = 0;
                    for (std::size_t d = 0; d < D; ++d) dot += static_cast<double>(row[d]) * topic_sum_[d];
                    cs.emplace_back(dot * inv, m_.topic_ids_[r]);
                }
                const std::size_t k = std::min<std::size_t>(cs.size(), static_cast<std::size_t>(m_.p_.topic_k));
                std::partial_sort(cs.begin(), cs.begin() + static_cast<std::ptrdiff_t>(k), cs.end(),
                                  [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
                const double top = k ? cs[0].first : 0;
                for (std::size_t i = 0; i < k; ++i) all.emplace_back(std::exp((cs[i].first - top) / m_.p_.topic_temp), cs[i].second);
            } else {
                for (const auto& [id, q] : q_[static_cast<std::size_t>(part)]) all.emplace_back(q, id);
                const std::size_t k = std::min<std::size_t>(all.size(), static_cast<std::size_t>(m_.p_.cand));
                std::nth_element(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k), all.end(),
                                 [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
                all.resize(k);
            }
            for (const auto& [q, id] : all) {
                C.push_back({m_.words_[static_cast<std::size_t>(id)], id, q});
                total_[part] += q;
            }
            std::sort(C.begin(), C.end(), [](const Cand& a, const Cand& b) { return a.low < b.low; });
        }

        const DocTopic& m_;
        std::unordered_map<int, double> q_[kParts];
        std::vector<Cand> cand_[kParts];
        bool dirty_[kParts];
        double total_[kParts] = {0, 0, 0};
        std::string partial_, tail_;
        std::uint8_t prev_ = ' ';
        int nwords_ = 0, since_topic_ = 0;
        bool frozen_ = false;
        std::vector<float> topic_sum_;
        std::unordered_map<int, int> seen_;
    };

    static bool alpha(std::uint8_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
    static std::string lower(const std::string& s) {
        std::string o(s);
        for (char& c : o) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return o;
    }
    /// Documents of text[0, n): from each "<page>" to the next (the bytes before the first one are a document too).
    static std::vector<std::pair<std::size_t, std::size_t>> split_docs(const std::uint8_t* text, std::size_t n) {
        std::vector<std::pair<std::size_t, std::size_t>> d;
        static const char key[] = "<page>";
        std::size_t start = 0;
        for (std::size_t i = 0; i + 6 <= n; ++i)
            if (text[i] == '<' && std::memcmp(text + i, key, 6) == 0) {
                if (i > start) d.emplace_back(start, i);
                start = i;
            }
        if (n > start) d.emplace_back(start, n);
        return d;
    }

 private:
    void build_triggers(const std::uint8_t* text, const std::vector<std::pair<std::size_t, std::size_t>>& docs) {
        const std::size_t V = words_.size();
        const double N = static_cast<double>(docs.size());
        std::vector<std::uint64_t> pairs;
        std::unordered_map<int, int> tf;
        std::vector<std::pair<double, int>> best;
        for (const auto& [s, e] : docs) {
            tf.clear();
            std::size_t i = s;
            while (i < e) {
                if (!alpha(text[i])) {
                    ++i;
                    continue;
                }
                std::size_t j = i;
                while (j < e && alpha(text[j])) ++j;
                if (j - i <= 40) {
                    const int w = id(lower(std::string(reinterpret_cast<const char*>(text + i), j - i)));
                    if (w >= 0 && content_[static_cast<std::size_t>(w)] && df_[static_cast<std::size_t>(w)] >= p_.min_df) ++tf[w];
                }
                i = j;
            }
            best.clear();
            for (const auto& [w, c] : tf) best.emplace_back(c * std::log(N / df_[static_cast<std::size_t>(w)]), w);
            const std::size_t k = std::min<std::size_t>(best.size(), static_cast<std::size_t>(p_.max_doc_words));
            std::partial_sort(best.begin(), best.begin() + static_cast<std::ptrdiff_t>(k), best.end(),
                              [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
            for (std::size_t a = 0; a < k; ++a)
                for (std::size_t b = a + 1; b < k; ++b) {
                    std::uint64_t x = static_cast<std::uint64_t>(best[a].second), y = static_cast<std::uint64_t>(best[b].second);
                    if (x > y) std::swap(x, y);
                    pairs.push_back(x * V + y);
                }
        }
        std::sort(pairs.begin(), pairs.end());
        std::vector<std::vector<std::pair<double, int>>> cand(V);
        for (std::size_t i = 0; i < pairs.size();) {
            std::size_t j = i;
            while (j < pairs.size() && pairs[j] == pairs[i]) ++j;
            const double co = static_cast<double>(j - i);
            if (co >= p_.min_co) {
                const int a = static_cast<int>(pairs[i] / V), b = static_cast<int>(pairs[i] % V);
                const double pmi = std::log(co * N / (static_cast<double>(df_[static_cast<std::size_t>(a)]) * df_[static_cast<std::size_t>(b)]));
                if (pmi > 0) {
                    cand[static_cast<std::size_t>(a)].emplace_back(co * pmi, b);
                    cand[static_cast<std::size_t>(b)].emplace_back(co * pmi, a);
                }
            }
            i = j;
        }
        std::vector<std::uint64_t>().swap(pairs);
        for (std::size_t a = 0; a < V; ++a) {
            auto& c = cand[a];
            const std::size_t k = std::min<std::size_t>(c.size(), static_cast<std::size_t>(p_.per_source));
            std::partial_sort(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(k), c.end(),
                              [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
            double s = 0;
            for (std::size_t t = 0; t < k; ++t) s += c[t].first;
            for (std::size_t t = 0; t < k; ++t) trig_[a].emplace_back(c[t].second, c[t].first / s);
            std::vector<std::pair<double, int>>().swap(c);
        }
    }

    Params p_;
    std::vector<std::string> words_, surf_;
    std::unordered_map<std::string, int> wid_;
    std::vector<int> df_;
    std::vector<char> content_;
    double end_[256] = {0};
    std::vector<std::vector<std::pair<int, double>>> trig_;
    std::size_t ndocs_ = 0;
    int vdim_ = 0;
    std::unordered_map<std::string, std::vector<float>> vrows_;
    std::vector<int> topic_ids_;
    std::vector<float> topic_rows_;
};

}  // namespace cypha::cyphalm
