#pragma once

/// Sentence skeletons for the planner (docs/reports/K5_PLANNER.md, GENERATION_PLAN §13.37 step 2).
///
/// A skeleton is a sentence shape over K5's 256 exchange classes with two slots, S (the fact's subject) and O (its
/// value). Skeletons are mined offline by distant supervision: a real sentence of clean enwik9 that holds a store
/// fact's subject and value (as K5 tokens) gives that fact's relation one skeleton, its other tokens replaced by their
/// K5 classes. The bank also counts which relation follows which inside a document (the plan's transitions).
///
/// File: "K5PLAN-SKEL 1", then "skel REL COUNT CODES" (codes: class numbers, S, O), "trans R1 R2 COUNT" (R1 "<s>": the
/// first fact sentence of a document).
///
/// Author: Odin Loch

#include "cypha/lm/planner/facts.hpp"
#include "cypha/lm/structure/experts.hpp"
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm::planner {

inline constexpr std::uint16_t kSlotS = 0xFFF0, kSlotO = 0xFFF1;
inline constexpr std::size_t kMinCodes = 5, kMaxCodes = 32;

struct Skel {
    std::vector<std::uint16_t> codes;
    std::uint32_t count = 0;
};

// One fact matched in a real sentence: the fact, the S and O spans [a, b) in the sentence's tokens.
struct Match {
    std::uint32_t fact = ~0u;
    std::size_t sa = 0, sb = 0, oa = 0, ob = 0;
};

inline bool tok_eq_ci(const std::string& a, const std::string& b) { return a.size() == b.size() && lower_s(a) == lower_s(b); }

// The first store fact whose subject and value both occur in tokens t (subject longest first, value not inside it).
inline bool match_sentence(const FactStore& fs, const std::vector<std::string>& t, Match& out,
                           const std::vector<std::uint32_t>* only_rels = nullptr) {
    std::vector<std::pair<std::uint32_t, std::size_t>> hits;
    for (std::size_t a = 0; a < t.size(); ++a) {
        if (t[a].empty() || !(t[a][0] >= 'A' && t[a][0] <= 'Z')) continue;  // subjects are names
        fs.match_at(t, a, hits);
        for (const auto& [e, n] : hits) {
            for (const auto fi : fs.of_subj(e)) {
                const auto& f = fs.fact(fi);
                if (only_rels && std::find(only_rels->begin(), only_rels->end(), f.rel) == only_rels->end()) continue;
                const auto& vt = fs.ent_tok(f.val);
                for (std::size_t b = 0; b + vt.size() <= t.size(); ++b) {
                    if (b < a + n && b + vt.size() > a) continue;
                    bool ok = true;
                    for (std::size_t k = 0; k < vt.size() && ok; ++k) ok = tok_eq_ci(t[b + k], vt[k]);
                    if (!ok) continue;
                    out = {fi, a, a + n, b, b + vt.size()};
                    return true;
                }
            }
        }
    }
    return false;
}

// The skeleton of a matched sentence: classes of K5 ids, the spans as slots. False if a token outside the slots is
// unknown to K5, if it is too short or long, or if it does not end with ". ! ?".
inline bool shape_of(const wkn::K5& m, const std::vector<std::string>& t, const Match& mt, std::vector<std::uint16_t>& codes) {
    codes.clear();
    const auto& cls = m.classes().classes();
    if (t.empty() || !(t.back() == "." || t.back() == "!" || t.back() == "?")) return false;
    for (std::size_t i = 0; i < t.size();) {
        if (i == mt.sa) {
            codes.push_back(kSlotS), i = mt.sb;
            continue;
        }
        if (i == mt.oa) {
            codes.push_back(kSlotO), i = mt.ob;
            continue;
        }
        const std::uint32_t id = m.id(t[i]);
        if (id == wkn::kUnk || t[i] == "\n") return false;
        codes.push_back(static_cast<std::uint16_t>(cls[id]));
        ++i;
    }
    return codes.size() >= kMinCodes && codes.size() <= kMaxCodes;
}

class SkelBank {
 public:
    static constexpr std::size_t kKeep = 300;  // skeletons kept per relation (the most frequent)

    void add(const std::string& rel, const std::vector<std::uint16_t>& codes) { ++raw_[rel][codes]; }
    void add_trans(const std::string& a, const std::string& b) { ++trans_[a][b]; }
    // Keeps each relation's kKeep most frequent skeletons.
    void finish() {
        by_rel_.clear();
        for (auto& [r, mp] : raw_) {
            auto& v = by_rel_[r];
            for (const auto& [c, n] : mp) v.push_back({c, n});
            std::stable_sort(v.begin(), v.end(), [](const Skel& a, const Skel& b) { return a.count > b.count; });
            if (v.size() > kKeep) v.resize(kKeep);
        }
    }
    bool save(const std::string& path) const {
        std::ofstream o(path);
        o << "K5PLAN-SKEL 1\n";
        for (const auto& [r, v] : by_rel_)
            for (const auto& s : v) {
                o << "skel " << r << " " << s.count;
                for (const auto c : s.codes) o << " " << (c == kSlotS ? std::string("S") : c == kSlotO ? std::string("O") : std::to_string(c));
                o << "\n";
            }
        for (const auto& [a, mp] : trans_)
            for (const auto& [b, n] : mp) o << "trans " << a << " " << b << " " << n << "\n";
        return static_cast<bool>(o);
    }
    bool load(const std::string& path, std::string& err) {
        std::ifstream in(path);
        std::string line, tag;
        if (!std::getline(in, line) || line != "K5PLAN-SKEL 1") return err = "cannot read the skeleton bank " + path, false;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            ls >> tag;
            if (tag == "skel") {
                std::string r, c;
                Skel s;
                ls >> r >> s.count;
                while (ls >> c) s.codes.push_back(c == "S" ? kSlotS : c == "O" ? kSlotO : static_cast<std::uint16_t>(std::stoi(c)));
                by_rel_[r].push_back(std::move(s));
            } else if (tag == "trans") {
                std::string a, b;
                std::uint64_t n = 0;
                ls >> a >> b >> n;
                trans_[a][b] += n;
            }
        }
        for (const auto& [a, mp] : trans_)
            for (const auto& [b, n] : mp) out_[a] += n, in_[b] += n, all_ += n;
        return true;
    }
    // HONEST_DATES2 (--k5-plan-skel-add): the skeletons of another bank file appended to their relations, after the
    // bank's own (so its indices stay), those already in the relation left out; its transitions are not read.
    // Returns the number added, or -1 (err) if the file cannot be read.
    long extend(const std::string& path, std::string& err) {
        std::ifstream in(path);
        std::string line, tag;
        if (!std::getline(in, line) || line != "K5PLAN-SKEL 1") return err = "cannot read the skeleton bank " + path, -1;
        long added = 0;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            ls >> tag;
            if (tag != "skel") continue;
            std::string r, c;
            Skel s;
            ls >> r >> s.count;
            while (ls >> c) s.codes.push_back(c == "S" ? kSlotS : c == "O" ? kSlotO : static_cast<std::uint16_t>(std::stoi(c)));
            auto& v = by_rel_[r];
            if (std::any_of(v.begin(), v.end(), [&](const Skel& x) { return x.codes == s.codes; })) continue;
            v.push_back(std::move(s));
            ++added;
        }
        return added;
    }
    const std::vector<Skel>* of(const std::string& rel) const {
        const auto it = by_rel_.find(rel);
        return it == by_rel_.end() || it->second.empty() ? nullptr : &it->second;
    }
    // log2 P(b | a) over relations, add-half smoothed over the relations with skeletons. s4 > 0 (§13.42 Q2, S4: the
    // plan's fact choice): (1 - s4) of it and s4 of b's marginal over all transitions, add-half smoothed alike.
    double trans_lp(const std::string& a, const std::string& b) const {
        const double R = static_cast<double>(by_rel_.size()) + 1;
        double n = 0, tot = 0;
        if (const auto it = trans_.find(a); it != trans_.end()) {
            if (const auto jt = it->second.find(b); jt != it->second.end()) n = static_cast<double>(jt->second);
            tot = static_cast<double>(out_.at(a));
        }
        if (s4 <= 0) return std::log2((n + 0.5) / (tot + 0.5 * R));
        const auto ib = in_.find(b);
        const double nb = ib == in_.end() ? 0.0 : static_cast<double>(ib->second);
        return std::log2((1 - s4) * (n + 0.5) / (tot + 0.5 * R) + s4 * (nb + 0.5) / (static_cast<double>(all_) + 0.5 * R));
    }
    double s4 = 0;  // --k5-plan-s4 (slots.hpp FillOpts)
    std::size_t relations() const { return by_rel_.size(); }
    std::size_t skeletons() const {
        std::size_t n = 0;
        for (const auto& [r, v] : by_rel_) n += v.size();
        return n;
    }
    const std::map<std::string, std::vector<Skel>>& all() const { return by_rel_; }

 private:
    std::map<std::string, std::map<std::vector<std::uint16_t>, std::uint32_t>> raw_;
    std::map<std::string, std::vector<Skel>> by_rel_;
    std::map<std::string, std::map<std::string, std::uint64_t>> trans_;
    std::map<std::string, std::uint64_t> out_, in_;  // transitions from a relation; into one (S4)
    std::uint64_t all_ = 0;
};

// Walks the documents (lines) of text that start in [lo, hi): each document's sentences (K5's tokens, cut as K5's
// gate cuts) go to f(doc_sentences, doc_index).
template <class F>
void for_docs(std::string_view text, std::size_t lo, std::size_t hi, F&& f) {
    std::size_t doc = 0;
    for (std::size_t a = 0; a < text.size();) {
        std::size_t b = text.find('\n', a);
        if (b == std::string_view::npos) b = text.size();
        if (a >= lo && a < hi && b > a) f(structure::split(wkn::tokens(text.substr(a, b - a))), doc);
        ++doc;
        a = b + 1;
        if (a >= hi) break;
    }
}

}  // namespace cypha::cyphalm::planner
