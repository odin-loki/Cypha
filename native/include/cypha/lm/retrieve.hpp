// docs/reports/K5_RETRIEVE.md (GENERATION_PLAN §13.65, G2): read up on the subject before writing.
//
// For each prompt, K5 finds the documents of its own training text most related to the prompt and adds their n-grams
// to its counts for that continuation only (word_kn.hpp: an Overlay, as --k5-dyn adds the prompt's own n-grams).
//
// The method: BM25 (Robertson and Zaragoza 2009; k1 1.2, b 0.75) over K5's word ids. The query is the prompt's distinct
// words outside K5's skip_top most frequent words (the gate's prompt-association rule); a word in more than max_df of
// the documents is left out. A word's postings are its suffix-array interval in K5's count index (lm/count_index.hpp),
// so the index adds only the documents' start positions and lengths. Only documents of the --k5-more text are searched
// (K5_SCALE's train_all.txt: enwik9 past enwik8, GEN-A and SCRIBE overlap removed); K5's own text is never retrieved.
// Ties go to the earlier document. Everything is K5's own counts (rule 10); nothing is read from GEN-A or SCRIBE.
//
// The weight (Params): w=L is --k5-dyn's absolute weight (the documents' counts times L beside K5's counts). rel=R is
// relative: in a context the documents saw, their counts get R times K5's own count of the context, so the documents
// hold about R / (1 + R) of that order's mass, whatever the context's frequency.
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cypha/lm/count_index.hpp"

namespace cypha::cyphalm::retrieve {

struct Params {
    int k = 0;             // documents retrieved per prompt (0: off, the default)
    double rel = 0;        // relative weight R (see above); 0: absolute
    double w = 0;          // absolute weight L (--k5-dyn's); used when rel is 0
    int skip_top = 100;    // query words: outside K5's skip_top most frequent words
    double max_df = 0.05;  // and in at most this share of the documents
    bool on() const { return k > 0; }
    std::string describe() const {
        char b[200];
        std::snprintf(b, sizeof b, "\tretrieve_k=%d\tretrieve_rel=%g\tretrieve_w=%g\tretrieve_top=%d\tretrieve_maxdf=%g", k, rel, w, skip_top, max_df);
        return b;
    }
};

// --k5-retrieve k=N,rel=R | k=N,w=L [,top=T][,maxdf=F]
inline bool parse(const std::string& val, Params& p, std::string& err) {
    Params q;
    for (std::size_t a = 0; a <= val.size();) {
        std::size_t b = val.find(',', a);
        if (b == std::string::npos) b = val.size();
        const std::string f = val.substr(a, b - a);
        const std::size_t e = f.find('=');
        char* end = nullptr;
        const double v = e == std::string::npos ? 0 : std::strtod(f.c_str() + e + 1, &end);
        if (e == std::string::npos || e + 1 >= f.size() || *end) return err = "--k5-retrieve: bad field " + f, false;
        const std::string key = f.substr(0, e);
        if (key == "k" && v >= 0 && v <= 256 && v == std::floor(v)) q.k = static_cast<int>(v);
        else if (key == "rel" && v >= 0 && v <= 100) q.rel = v;
        else if (key == "w" && v >= 0 && v <= 1e6) q.w = v;
        else if (key == "top" && v >= 0 && v <= 60000 && v == std::floor(v)) q.skip_top = static_cast<int>(v);
        else if (key == "maxdf" && v > 0 && v <= 1) q.max_df = v;
        else return err = "--k5-retrieve: fields k=N (0-256), rel=R, w=L, top=T, maxdf=F; bad " + f, false;
        a = b + 1;
    }
    if (q.k > 0 && q.rel <= 0 && q.w <= 0) return err = "--k5-retrieve: give rel=R or w=L", false;
    p = q;
    return true;
}

class Index {
 public:
    using CI = cidx::CountIndex<std::uint32_t>;
    // ids: K5's id stream (each document starts with bos); from: the first id of the searched text (the --k5-more
    // text's first document). log: a file that gets each new prompt's documents (empty: none).
    void build(const std::vector<std::uint32_t>& ids, std::uint32_t bos, std::size_t from, std::string log = {}) {
        start_.clear();
        for (std::size_t i = from; i < ids.size(); ++i)
            if (ids[i] == bos) start_.push_back(static_cast<std::uint32_t>(i));
        start_.push_back(static_cast<std::uint32_t>(ids.size()));
        double tot = 0;
        len_.assign(docs(), 0.0);
        for (std::size_t d = 0; d < docs(); ++d) tot += len_[d] = static_cast<double>(start_[d + 1] - start_[d] - 1);
        avg_ = docs() ? tot / static_cast<double>(docs()) : 1.0;
        from_ = from;
        log_ = std::move(log);
    }
    std::size_t docs() const { return start_.empty() ? 0 : start_.size() - 1; }
    // Document d's ids: [first, second), its bos first. d counts from the searched text's first document, so it is
    // the line number (from 0) of a document of that text.
    std::pair<std::size_t, std::size_t> span(std::uint32_t d) const { return {start_[d], start_[d + 1]}; }

    // The k best documents for the query ids (a prompt's tokens), best first. prompt: for the log only.
    std::vector<std::uint32_t> top(const std::vector<std::uint32_t>& q, const Params& p, const CI& ci, std::string_view prompt) const {
        std::vector<std::uint32_t> words;
        const std::uint32_t lo = 2 + static_cast<std::uint32_t>(p.skip_top);
        for (const std::uint32_t v : q)
            if (v >= lo && v < ci.sigma() && std::find(words.begin(), words.end(), v) == words.end()) words.push_back(v);
        std::sort(words.begin(), words.end());
        const double N = static_cast<double>(docs());
        std::map<std::uint32_t, double> score;
        std::vector<std::uint32_t> ds;
        constexpr double k1 = 1.2, b = 0.75;
        for (const std::uint32_t v : words) {
            const CI::Range r = ci.first(v);
            ds.clear();
            for (std::size_t i = r.lo; i < r.hi; ++i) {
                const std::size_t pos = ci.sa(i);
                if (pos < from_) continue;
                ds.push_back(static_cast<std::uint32_t>(std::upper_bound(start_.begin(), start_.end(), static_cast<std::uint32_t>(pos)) - start_.begin() - 1));
            }
            std::sort(ds.begin(), ds.end());
            std::size_t df = 0;
            for (std::size_t i = 0; i < ds.size(); ++i) df += i == 0 || ds[i] != ds[i - 1];
            if (df == 0 || static_cast<double>(df) > p.max_df * N) continue;
            const double idf = std::log(1.0 + (N - static_cast<double>(df) + 0.5) / (static_cast<double>(df) + 0.5));
            for (std::size_t i = 0; i < ds.size();) {
                std::size_t j = i + 1;
                while (j < ds.size() && ds[j] == ds[i]) ++j;
                const double tf = static_cast<double>(j - i), L = len_[ds[i]] / avg_;
                score[ds[i]] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * L));
                i = j;
            }
        }
        std::vector<std::pair<double, std::uint32_t>> all;
        all.reserve(score.size());
        for (const auto& [d, s] : score) all.emplace_back(s, d);
        const std::size_t k = std::min<std::size_t>(static_cast<std::size_t>(std::max(p.k, 0)), all.size());
        std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k), all.end(), [](const auto& x, const auto& y) {
            return x.first != y.first ? x.first > y.first : x.second < y.second;
        });
        std::vector<std::uint32_t> out;
        for (std::size_t i = 0; i < k; ++i) out.push_back(all[i].second);
        if (!log_.empty()) write_log(prompt, words.size(), all, k);
        return out;
    }

 private:
    void write_log(std::string_view prompt, std::size_t nq, const std::vector<std::pair<double, std::uint32_t>>& all, std::size_t k) const {
        std::string key(prompt);
        std::lock_guard<std::mutex> g(*mu_);
        if (!logged_->insert(key).second) return;
        for (char& c : key)
            if (c == '\t' || c == '\n' || c == '\r') c = ' ';
        if (FILE* f = std::fopen(log_.c_str(), "a")) {
            std::fprintf(f, "%s\t%zu\t", key.c_str(), nq);
            for (std::size_t i = 0; i < k; ++i) std::fprintf(f, "%s%u:%.3f", i ? "," : "", all[i].second, all[i].first);
            std::fprintf(f, "\n");
            std::fclose(f);
        }
    }
    std::vector<std::uint32_t> start_;  // document d: ids [start_[d], start_[d + 1]); the last entry is the stream's end
    std::vector<double> len_;           // its tokens (bos aside)
    double avg_ = 1;
    std::size_t from_ = 0;
    std::string log_;
    std::shared_ptr<std::mutex> mu_ = std::make_shared<std::mutex>();
    std::shared_ptr<std::set<std::string>> logged_ = std::make_shared<std::set<std::string>>();
};

}  // namespace cypha::cyphalm::retrieve
