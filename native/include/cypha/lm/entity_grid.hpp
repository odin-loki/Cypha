// An entity grid for K5 (Barzilay and Lapata, 2008; docs/reports/K5_DISCOURSE.md, GENERATION_PLAN §13.32).
//
// A paragraph is a grid: one row per sentence, one column per entity, each cell the entity's role in that sentence:
// S (subject), O (object), X (other) or - (absent). Each column is a Markov chain of order 2 over the four roles. Its
// transition probabilities P(r | r2, r1) are counted on K5's clean training text, one grid per document, add-0.5
// smoothed. A paragraph's log probability is the sum over its columns of the chain's log probabilities, each column
// starting from (-, -).
//
// No parser. Entities and roles come from simple rules on K5's tokens:
//   - entities: maximal runs of capitalised words (a sentence's first word only if it is not a function word), and the
//     last word of a run of content words after a determiner ("the national institute" -> "institute"); the key is
//     the run's last word, lower case, a plural -s removed;
//   - roles by position: a mention right after a preposition (or a preposition and a determiner) is X; of the others,
//     the sentence's first is S and the rest are O; an entity's role is its strongest mention's (S > O > X).
//
// Interface for other sessions (the MCMC sampler of wip/k5-mcmc): tokens are K5's (wkn::tokens).
//   EntityGrid g; g.train(docs);                         // docs: one token vector per document
//   auto sents = split_sentences(tokens);                // tokens -> sentences at . ! ? and line breaks
//   double lp = g.paragraph_logp(sents, &ntrans);        // log2 P of the paragraph's grid; lp / ntrans = coherence
//   EntityGrid::History h; g.append(h, g.roles(s));      // incremental: the grid so far
//   double d = g.append_logp(h, g.roles(s));             // = paragraph_logp(... + s) - paragraph_logp(...)
//   double c = g.continuation_score(h, g.roles(s));      // sum over kept entities of log2 P(r | h) - log2 P(- | h)
//   double x = g.score(context_tokens, candidate_tokens); // the stable call: append_logp of the candidate's sentences
//                                                         // after the context's (cont = true: continuation_score)
//   g.train_lines(text, tokenizer, threads); g.save(path); g.load(path)
// word_kn.hpp's wkn::eg_score(grid, context_text, candidate_text) does the tokenising.
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <thread>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cypha/lm/stopwords.hpp"

namespace cypha::cyphalm::eg {

enum Role : std::uint8_t { kAbsent = 0, kX = 1, kO = 2, kS = 3 };
inline char role_char(Role r) { return "-XOS"[r]; }

// Splits K5's tokens into sentences at ".", "!", "?" and "\n" (the end token stays with its sentence).
inline std::vector<std::vector<std::string>> split_sentences(const std::vector<std::string>& toks) {
    std::vector<std::vector<std::string>> out(1);
    for (const auto& t : toks) {
        out.back().push_back(t);
        if (words::sentence_end(t)) out.emplace_back();
    }
    if (out.back().empty()) out.pop_back();
    return out;
}

// One sentence's entities and their roles (sorted by key).
using SentRoles = std::vector<std::pair<std::string, Role>>;

inline SentRoles sentence_roles(const std::vector<std::string>& s) {
    const auto& stop = words::stop();
    const auto& det = words::determiners();
    const auto& prep = words::prepositions();
    struct Mention {
        std::size_t start;
        std::string key;
    };
    std::vector<Mention> ms;
    // The first word of the sentence, not counting opening punctuation.
    std::size_t first = 0;
    while (first < s.size() && !words::alpha(s[first])) ++first;
    std::vector<char> used(s.size(), 0);
    // Capitalised runs.
    for (std::size_t i = 0; i < s.size();) {
        if (!words::alpha(s[i]) || !words::capital(s[i])) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < s.size() && words::alpha(s[j]) && words::capital(s[j])) ++j;
        std::size_t a = i;
        if (a == first && stop.count(words::lower(s[a]))) ++a;  // "The United States" -> "United States"; "It" -> none
        if (a < j) {
            const std::string k = words::stem(s[j - 1]);
            if (!stop.count(k) && k.size() > 1) {
                ms.push_back({a, k});
                for (std::size_t t = a; t < j; ++t) used[t] = 1;
            }
        }
        i = j;
    }
    // Content-word runs after a determiner.
    for (std::size_t i = 0; i + 1 < s.size(); ++i) {
        if (!det.count(words::lower(s[i]))) continue;
        std::size_t j = i + 1, last = s.size();
        while (j < s.size() && j <= i + 4 && words::alpha(s[j]) && !words::capital(s[j]) && !used[j] && !stop.count(words::lower(s[j])))
            last = j, ++j;
        if (last == s.size()) continue;
        const std::string k = words::stem(s[last]);
        if (k.size() > 2) {
            ms.push_back({i, k});
            for (std::size_t t = i + 1; t <= last; ++t) used[t] = 1;
        }
    }
    std::sort(ms.begin(), ms.end(), [](const Mention& a, const Mention& b) { return a.start < b.start; });
    SentRoles r;
    bool subj = false;
    for (const auto& m : ms) {
        // X when a preposition precedes the mention (directly, or before its determiner).
        bool x = false;
        if (m.start > 0) {
            const std::string p1 = words::lower(s[m.start - 1]);
            if (prep.count(p1)) x = true;
            else if (det.count(p1) && m.start > 1 && prep.count(words::lower(s[m.start - 2]))) x = true;
            if (!x && det.count(words::lower(s[m.start])) && prep.count(p1)) x = true;
        }
        Role role = x ? kX : (subj ? kO : kS);
        if (!x) subj = true;
        auto it = std::find_if(r.begin(), r.end(), [&](const auto& e) { return e.first == m.key; });
        if (it == r.end()) r.emplace_back(m.key, role);
        else it->second = std::max(it->second, role);
    }
    std::sort(r.begin(), r.end());
    return r;
}

class EntityGrid {
 public:
    // The grid so far: each entity's last two roles, and the number of sentences.
    struct History {
        std::unordered_map<std::string, std::pair<Role, Role>> e;  // (r2, r1): two sentences back, one back
        std::size_t n = 0;
    };

    // The transition counts of one or more documents.
    struct Counts {
        std::array<double, 64> c{};
        std::size_t docs = 0, sents = 0, cols = 0;
        void add(const Counts& o) {
            for (int i = 0; i < 64; ++i) c[i] += o.c[i];
            docs += o.docs, sents += o.sents, cols += o.cols;
        }
    };
    // Adds one document (K5's tokens) to cnt: every column's transitions, each column from (-, -).
    static void count_doc(const std::vector<std::string>& d, Counts& cnt) {
        const auto sents = split_sentences(d);
        std::unordered_map<std::string, std::vector<Role>> col;
        for (std::size_t i = 0; i < sents.size(); ++i)
            for (const auto& [k, r] : sentence_roles(sents[i])) {
                auto& v = col[k];
                v.resize(sents.size(), kAbsent);
                v[i] = r;
            }
        for (auto& [k, v] : col) {
            v.resize(sents.size(), kAbsent);
            Role r2 = kAbsent, r1 = kAbsent;
            for (const Role r : v) {
                cnt.c[idx(r2, r1, r)] += 1;
                r2 = r1, r1 = r;
            }
        }
        ++cnt.docs;
        cnt.sents += sents.size();
        cnt.cols += col.size();
    }
    // docs: one token vector per document (K5's tokens). Counts every column's transitions, add-alpha smoothed.
    void train(const std::vector<std::vector<std::string>>& docs, double alpha = 0.5) {
        Counts c;
        for (const auto& d : docs) count_doc(d, c);
        set_counts(c, alpha);
    }
    // text: one document per line; tokenize(line, emit) calls emit(token) (e.g. wkn::tokenize); on `threads` threads.
    template <class Tok>
    void train_lines(std::string_view text, Tok&& tokenize, unsigned threads = 8, double alpha = 0.5) {
        std::vector<std::string_view> lines;
        for (std::size_t a = 0; a < text.size();) {
            std::size_t b = text.find('\n', a);
            if (b == std::string_view::npos) b = text.size();
            if (b > a) lines.push_back(text.substr(a, b - a));
            a = b + 1;
        }
        const unsigned T = std::max(1u, threads);
        std::vector<Counts> part(T);
        std::vector<std::thread> th;
        for (unsigned t = 0; t < T; ++t)
            th.emplace_back([&, t] {
                std::vector<std::string> d;
                for (std::size_t j = lines.size() * t / T; j < lines.size() * (t + 1) / T; ++j) {
                    d.clear();
                    tokenize(lines[j], [&](std::string_view x) { d.emplace_back(x); });
                    count_doc(d, part[t]);
                }
            });
        for (auto& x : th) x.join();
        Counts c;
        for (const auto& x : part) c.add(x);
        set_counts(c, alpha);
    }
    void set_counts(const Counts& cnt, double alpha = 0.5) {
        const auto& c = cnt.c;
        docs_ = cnt.docs, sents_ = cnt.sents, cols_ = cnt.cols;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                double z = 0;
                for (int r = 0; r < 4; ++r) z += c[idx(Role(a), Role(b), Role(r))] + alpha;
                for (int r = 0; r < 4; ++r) lp_[idx(Role(a), Role(b), Role(r))] = std::log2((c[idx(Role(a), Role(b), Role(r))] + alpha) / z);
            }
        counts_ = c;
        alpha_ = alpha;
        trained_ = true;
    }
    bool save(const std::string& path) const {
        std::ofstream o(path, std::ios::binary);
        const std::uint64_t d = docs_, s = sents_, c = cols_;
        o.write("EGR1", 4);
        o.write(reinterpret_cast<const char*>(counts_.data()), 64 * 8);
        o.write(reinterpret_cast<const char*>(&d), 8), o.write(reinterpret_cast<const char*>(&s), 8), o.write(reinterpret_cast<const char*>(&c), 8);
        o.write(reinterpret_cast<const char*>(&alpha_), 8);
        return static_cast<bool>(o);
    }
    bool load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        char m[4];
        Counts c;
        std::uint64_t d = 0, s = 0, n = 0;
        double alpha = 0.5;
        if (!in.read(m, 4) || std::string_view(m, 4) != "EGR1") return false;
        in.read(reinterpret_cast<char*>(c.c.data()), 64 * 8);
        in.read(reinterpret_cast<char*>(&d), 8), in.read(reinterpret_cast<char*>(&s), 8), in.read(reinterpret_cast<char*>(&n), 8);
        in.read(reinterpret_cast<char*>(&alpha), 8);
        if (!in) return false;
        c.docs = d, c.sents = s, c.cols = n;
        set_counts(c, alpha);
        return true;
    }
    bool trained() const { return trained_; }
    std::size_t docs() const { return docs_; }
    std::size_t sentences() const { return sents_; }
    std::size_t columns() const { return cols_; }
    // log2 P(r | r2, r1), and the training count of the transition.
    double logp(Role r2, Role r1, Role r) const { return lp_[idx(r2, r1, r)]; }
    double count(Role r2, Role r1, Role r) const { return counts_[idx(r2, r1, r)]; }

    SentRoles roles(const std::vector<std::string>& sentence) const { return sentence_roles(sentence); }

    // The change in the paragraph's log2 probability when sentence s (its roles) is appended to h: every entity in h
    // or s makes one transition; an entity new in s also pays its h.n leading absences from (-, -).
    double append_logp(const History& h, const SentRoles& s) const {
        double d = 0;
        for (const auto& [k, rr] : h.e) {
            const auto it = std::lower_bound(s.begin(), s.end(), k, [](const auto& e, const std::string& key) { return e.first < key; });
            const Role r = it != s.end() && it->first == k ? it->second : kAbsent;
            d += logp(rr.first, rr.second, r);
        }
        const double lead = h.n * logp(kAbsent, kAbsent, kAbsent);
        for (const auto& [k, r] : s)
            if (!h.e.count(k)) d += lead + logp(kAbsent, kAbsent, r);
        return d;
    }
    // How well s keeps the paragraph's entities: for each entity of h that s mentions, log2 P(r | h) - log2 P(- | h).
    // New entities and dropped ones count 0.
    double continuation_score(const History& h, const SentRoles& s) const {
        double d = 0;
        for (const auto& [k, r] : s) {
            const auto it = h.e.find(k);
            if (it == h.e.end()) continue;
            d += logp(it->second.first, it->second.second, r) - logp(it->second.first, it->second.second, kAbsent);
        }
        return d;
    }
    void append(History& h, const SentRoles& s) const {
        for (auto& [k, rr] : h.e) {
            const auto it = std::lower_bound(s.begin(), s.end(), k, [](const auto& e, const std::string& key) { return e.first < key; });
            const Role r = it != s.end() && it->first == k ? it->second : kAbsent;
            rr = {rr.second, r};
        }
        for (const auto& [k, r] : s)
            if (!h.e.count(k)) h.e.emplace(k, std::make_pair(kAbsent, r));
        ++h.n;
    }
    // The stable scoring call: log2 score of the candidate (one or more sentences of K5's tokens) given the context
    // (prompt + paragraph so far, K5's tokens). full: the change in the grid's log2 probability (append_logp, summed
    // over the candidate's sentences); cont: continuation_score (kept entities only).
    double score(const std::vector<std::string>& context, const std::vector<std::string>& candidate, bool cont = false) const {
        History h;
        for (const auto& s : split_sentences(context)) append(h, roles(s));
        double d = 0;
        for (const auto& s : split_sentences(candidate)) {
            const SentRoles r = roles(s);
            d += cont ? continuation_score(h, r) : append_logp(h, r);
            append(h, r);
        }
        return d;
    }
    // log2 P of a paragraph's whole grid (sentences of K5's tokens); ntrans: the number of transitions (columns x rows).
    double paragraph_logp(const std::vector<std::vector<std::string>>& sents, std::size_t* ntrans = nullptr) const {
        History h;
        double lp = 0;
        for (const auto& s : sents) {
            const SentRoles r = sentence_roles(s);
            lp += append_logp(h, r);
            append(h, r);
        }
        if (ntrans) *ntrans = h.e.size() * h.n;
        return lp;
    }

 private:
    static int idx(Role a, Role b, Role r) { return (a * 4 + b) * 4 + r; }
    std::array<double, 64> lp_{}, counts_{};
    double alpha_ = 0.5;
    std::size_t docs_ = 0, sents_ = 0, cols_ = 0;
    bool trained_ = false;
};

// Selftest: roles on a hand-made sentence, and append_logp adding up to paragraph_logp.
inline bool self_test(std::string& err) {
    const std::vector<std::string> s = {"The", "national", "institute", "of", "Barack", "Obama", "gave", "the", "estimate", "."};
    const SentRoles r = sentence_roles(s);
    const auto role = [&](const char* k) {
        for (const auto& [x, y] : r)
            if (x == k) return y;
        return kAbsent;
    };
    if (role("institute") != kS || role("obama") != kX || role("estimate") != kO) {
        err = "entity_grid: roles";
        for (const auto& [x, y] : r) err += " " + x + ":" + role_char(y);
        return false;
    }
    std::vector<std::vector<std::string>> docs = {
        {"Obama", "won", ".", "Obama", "spoke", "to", "the", "press", ".", "The", "press", "liked", "Obama", "."},
        {"The", "cat", "sat", ".", "The", "cat", "ate", "the", "fish", ".", "A", "dog", "barked", "."}};
    EntityGrid g;
    g.train(docs);
    const auto sents = split_sentences(docs[0]);
    std::size_t nt = 0;
    const double lp = g.paragraph_logp(sents, &nt);
    // Direct sum over the grid.
    double direct = 0;
    std::unordered_map<std::string, std::vector<Role>> col;
    for (std::size_t i = 0; i < sents.size(); ++i)
        for (const auto& [k, x] : sentence_roles(sents[i])) {
            auto& v = col[k];
            v.resize(sents.size(), kAbsent);
            v[i] = x;
        }
    for (auto& [k, v] : col) {
        v.resize(sents.size(), kAbsent);
        Role a = kAbsent, b = kAbsent;
        for (const Role x : v) direct += g.logp(a, b, x), a = b, b = x;
    }
    if (std::abs(lp - direct) > 1e-9 || nt != col.size() * sents.size()) return err = "entity_grid: paragraph_logp", false;
    {  // score(context, candidate) = paragraph_logp(context + candidate) - paragraph_logp(context)
        const std::vector<std::string> ctx(docs[0].begin(), docs[0].begin() + 9), cand(docs[0].begin() + 9, docs[0].end());
        if (std::abs(g.score(ctx, cand) - (lp - g.paragraph_logp(split_sentences(ctx)))) > 1e-9) return err = "entity_grid: score", false;
    }
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
            double z = 0;
            for (int x = 0; x < 4; ++x) z += std::exp2(g.logp(Role(a), Role(b), Role(x)));
            if (std::abs(z - 1) > 1e-9) return err = "entity_grid: not normalised", false;
        }
    return true;
}

}  // namespace cypha::cyphalm::eg
