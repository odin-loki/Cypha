#pragma once

/// K5's structure experts (docs/reports/K5_STRUCTURE.md, GENERATION_PLAN §13.33). Each scores one candidate sentence
/// given the prompt and the paragraph so far. K5 writes the candidates; these experts never touch its word
/// probabilities. Their statistics (Stats) are counted once from clean text K5 never saw and saved to a file.
///
/// Sentences are K5's tokens, cut as K5's gate cuts a candidate (".", "!" or "?" once 24 bytes are written, or at 240
/// bytes), and rendered with K5's spacing, so real and written sentences differ only in their words.
///
/// Features (higher is not always better; the meta-mixer learns each sign):
///   k5          mean log2 P(word) under K5 (given, from the draft)          pmi        the gate's prompt association (given)
///   class       mean log2 P(class) under a class 5-gram counted in S
///   ent_old     log(1 + names that the context has)                         ent_new    log(1 + names the context lacks)
///   ent_pmi     mean best document PPMI of a new name with a context name   slot_fit   mean log2 P(type | 2 words) / P(type)
///   pron_agree  mean log2 P(pronoun | main entity) / P(pronoun)              pron_orphan he/she with no person before it
///   year_fit    mean log2 P(gap bucket to the context's last year), centred n_years   log(1 + years)
///   open        log2 P(opening type | two previous types)
///   len         log2 P(length bucket | previous bucket)  balanced  brackets and quotes balanced  markup  the gate's share
///   match_train longest run shared with K5's training text (tokens, at most 13)   match_ctx  longest run shared with context
///   topic       LDA (wip/k5-discourse lda.hpp): mean log2 p_topic(w | the context's topics) / p(w) over content words
///   grid_full   the entity grid (entity_grid.hpp): the change in the grid's log2 probability when the candidate joins
///   grid_cont   ... its continuation score (kept entities only)
///
/// Experts that read the same statistics share one pass (docs/reports/MERGE_GATE.md, GENERATION_PLAN §13.40 M4): the
/// class n-gram's contexts are read off its n-grams; reference and time, and topic's count and ratio, share a pass
/// over the tokens; a name's statistics serve its co-occurrence and its slot type; counting and scoring share one slot walk.
///
/// Author: Odin Loch

#include "cypha/lm/entity_grid.hpp"  // wip/k5-discourse
#include "cypha/lm/lda.hpp"          // wip/k5-discourse
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cypha::cyphalm::structure {

namespace wkn = cypha::cyphalm::wkn;

// ---------------------------------------------------------------- features

enum Feat : int {
    kK5, kPmi, kClass, kEntOld, kEntNew, kEntPmi, kSlotFit, kPronAgree, kPronOrphan, kYearFit, kNYears, kOpen, kLen,
    kBalanced, kMarkup, kMatchTrain, kMatchCtx, kTopic, kGridFull, kGridCont, kNFeat
};
inline const char* feat_name(int j) {
    static const char* n[kNFeat] = {"k5", "pmi", "class", "ent_old", "ent_new", "ent_pmi", "slot_fit", "pron_agree", "pron_orphan",
                                    "year_fit", "n_years", "open", "len", "balanced", "markup", "match_train", "match_ctx",
                                    "topic", "grid_full", "grid_cont"};
    return n[j];
}
// The experts and their features (for "each expert alone" and the ablations).
struct ExpertDef {
    const char* name;
    std::vector<int> feats;
};
inline const std::vector<ExpertDef>& experts() {
    static const std::vector<ExpertDef> e = {
        {"k5", {kK5}},
        {"pmi", {kPmi}},
        {"class", {kClass}},
        {"relations", {kEntOld, kEntNew, kEntPmi, kSlotFit}},
        {"reference", {kPronAgree, kPronOrphan}},
        {"time", {kYearFit, kNYears}},
        {"transitions", {kOpen}},
        {"rhythm", {kLen, kBalanced, kMarkup}},
        {"novelty", {kMatchTrain, kMatchCtx}},
        {"topic", {kTopic}},
        {"grid", {kGridFull, kGridCont}},
    };
    return e;
}
// The natural direction of each feature (+1: more like a real, connected sentence), for the uniform baseline.
inline double natural_sign(int j) {
    switch (j) {
        case kEntNew: case kPronOrphan: case kMarkup: case kMatchTrain: case kMatchCtx: return -1;
        default: return 1;
    }
}

// ---------------------------------------------------------------- sentences

struct Sent {
    std::vector<std::string> tok;
    bool ended = false;  // ended by . ! ? (otherwise cut at 240 bytes or at the end of its document)
};

inline std::string lower(std::string_view s) {
    std::string o(s);
    for (char& c : o)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return o;
}
inline bool is_cap(std::string_view t) { return !t.empty() && t[0] >= 'A' && t[0] <= 'Z'; }
inline bool is_digits(std::string_view t) {
    if (t.empty()) return false;
    for (const char c : t)
        if (!(c >= '0' && c <= '9')) return false;
    return true;
}
inline int year_of(std::string_view t) {
    if (t.size() != 4 || !is_digits(t)) return 0;
    const int y = std::atoi(std::string(t).c_str());
    return y >= 1000 && y <= 2099 ? y : 0;
}
inline bool is_number(std::string_view t) { return !t.empty() && t[0] >= '0' && t[0] <= '9' && !year_of(t); }
inline std::uint64_t h64(std::string_view s) {
    std::uint64_t h = wkn::kFnv0;
    for (const char c : s) h = wkn::fnv_step(h, static_cast<unsigned char>(c));
    return wkn::mix64(h);
}

// Splits a token stream into sentences as K5's gate cuts a candidate. first_has_prev: the first token follows text.
inline std::vector<Sent> split(const std::vector<std::string>& toks, bool first_has_prev = false, std::size_t min_b = 24, std::size_t max_b = 240) {
    std::vector<Sent> out;
    Sent cur;
    std::size_t nb = 0;
    std::string prev;
    bool has_prev = first_has_prev;
    for (const auto& t : toks) {
        nb += wkn::piece(t, prev, has_prev).size();
        cur.tok.push_back(t);
        prev = t, has_prev = true;
        const bool end = t == "." || t == "!" || t == "?";
        if (nb >= max_b || (nb >= min_b && (end || t == "\n"))) {
            cur.ended = end;
            out.push_back(std::move(cur));
            cur = Sent{};
            nb = 0;
        }
    }
    if (!cur.tok.empty()) out.push_back(std::move(cur));
    return out;
}
inline std::string render(const std::vector<Sent>& ss, bool has_prev = false) {
    std::string o, prev;
    for (const auto& s : ss)
        for (const auto& t : s.tok) o += wkn::piece(t, prev, has_prev), prev = t, has_prev = true;
    return o;
}

// Personal pronouns: 0 masculine, 1 feminine, 2 neuter, 3 plural; -1 none.
inline int pron_class(std::string_view t) {
    static const std::unordered_map<std::string, int> m = {
        {"he", 0}, {"him", 0}, {"his", 0}, {"himself", 0}, {"she", 1}, {"her", 1}, {"hers", 1}, {"herself", 1},
        {"it", 2}, {"its", 2}, {"itself", 2}, {"they", 3}, {"them", 3}, {"their", 3}, {"theirs", 3}, {"themselves", 3}};
    if (t.empty() || t.size() > 10) return -1;
    const auto it = m.find(lower(t));
    return it == m.end() ? -1 : it->second;
}
inline bool is_month(std::string_view t) {
    static const std::unordered_set<std::string> m = {"January", "February", "March", "April", "May", "June", "July",
                                                      "August", "September", "October", "November", "December"};
    return m.count(std::string(t)) > 0;
}

// ---------------------------------------------------------------- statistics (counted from S)

// Opening types: 0 connective, 1 pronoun subject, 2 "The ...", 3 date or time phrase, 4 named subject, 5 other; 6 none.
inline constexpr int kOpenTypes = 6, kOpenNone = 6;
// Length buckets (tokens): <=8, <=12, <=16, <=20, <=25, <=30, <=40, >40, 8 cut (no end mark); 9 none.
inline constexpr int kLenB = 9, kLenNone = 9;
inline int len_bucket(const Sent& s) {
    if (!s.ended) return 8;
    const std::size_t n = s.tok.size();
    return n <= 8 ? 0 : n <= 12 ? 1 : n <= 16 ? 2 : n <= 20 ? 3 : n <= 25 ? 4 : n <= 30 ? 5 : n <= 40 ? 6 : 7;
}
// Year gap buckets (year - the context's last year): <-100, [-100,-26], [-25,-6], [-5,-1], 0, [1,5], [6,25], [26,100], >100;
// 9 the context has no year.
inline constexpr int kGapB = 10;
inline int gap_bucket(int y, int last) {
    if (!last) return 9;
    const int d = y - last;
    return d < -100 ? 0 : d < -25 ? 1 : d < -5 ? 2 : d < 0 ? 3 : d == 0 ? 4 : d <= 5 ? 5 : d <= 25 ? 6 : d <= 100 ? 7 : 8;
}
// Slot types: 0 word, 1 person, 2 place, 3 year, 4 number, 5 other name.
inline constexpr int kSlotT = 6;

struct NameStat {
    std::uint32_t n = 0, docs = 0, after_loc = 0;
    std::uint32_t pron[4] = {0, 0, 0, 0};
    std::int32_t id = -1;  // its id for document co-occurrence (names in at least 3 documents)
    int type() const {     // 1 person, 2 place, 5 other name
        const std::uint32_t np = pron[0] + pron[1] + pron[2] + pron[3];
        if (n >= 2 && after_loc * 10 >= n * 3) return 2;
        if (np >= 3 && (pron[0] + pron[1]) * 2 > np) return 1;
        return 5;
    }
};

class Stats {
 public:
    // The class n-gram's order and discount.
    static constexpr int kCO = 5;
    static constexpr double kD = 0.75;

    // Counts the statistics from docs (one document per line) with K5's classes (cls: class per K5 id; ncls classes).
    void build(std::string_view text, const wkn::K5& m, std::FILE* log = stderr) {
        cls_ = m.classes().classes();
        ncls_ = 0;
        for (const auto c : cls_) ncls_ = std::max<std::uint32_t>(ncls_, c + 1);
        if (cls_.empty()) ncls_ = 1;
        // Documents as sentences of tokens.
        std::vector<std::vector<Sent>> docs;
        std::size_t ntok = 0;
        for (std::size_t a = 0; a < text.size();) {
            std::size_t b = text.find('\n', a);
            if (b == std::string_view::npos) b = text.size();
            if (b > a) {
                auto t = wkn::tokens(text.substr(a, b - a));
                ntok += t.size();
                docs.push_back(split(t));
            }
            a = b + 1;
        }
        docs_ = docs.size();
        if (log) std::fprintf(log, "stats: %zu docs, %zu tokens\n", docs.size(), ntok);
        // Pass 0: capitalisation of every word away from a sentence's start.
        for (const auto& d : docs)
            for (const auto& s : d)
                for (std::size_t i = 1; i < s.tok.size(); ++i) {
                    const auto& t = s.tok[i];
                    if (t.empty() || !wkn::az(t[0])) continue;
                    auto& c = caps_[h64(lower(t))];
                    (is_cap(t) ? c.first : c.second)++;
                }
        // Pass 1: names, pronouns after names, document co-occurrence, years, openings, lengths, classes.
        std::vector<std::uint64_t> doc_names;
        std::vector<std::vector<std::uint64_t>> doc_name_list(docs.size());
        std::vector<std::uint32_t> cseq;
        cseq.reserve(ntok + docs.size());
        for (std::size_t di = 0; di < docs.size(); ++di) {
            const auto& d = docs[di];
            doc_names.clear();
            int po1 = kOpenNone, po2 = kOpenNone, pl = kLenNone;
            std::vector<int> recent_years;  // per sentence: its last year (0 none), the previous 6 sentences
            std::vector<std::pair<std::uint64_t, std::size_t>> open_names;  // names waiting for a pronoun (hash, token age)
            std::size_t age = 0;
            cseq.push_back(cls_.empty() ? 0 : cls_[wkn::kBos]);
            for (const auto& s : d) {
                const auto runs = name_runs(s);
                for (const auto& [a, b] : runs) {
                    const std::string nm = join(s, a, b);
                    const std::uint64_t h = h64(nm);
                    auto& ns = names_[h];
                    ++ns.n;
                    if (a > 0) {
                        const std::string p = lower(s.tok[a - 1]);
                        if (p == "in" || p == "at" || p == "from" || p == "near" || p == "of") ++ns.after_loc;
                    }
                    if (std::find(doc_names.begin(), doc_names.end(), h) == doc_names.end()) doc_names.push_back(h);
                }
                // The first pronoun after names counts for the earliest of them within 20 tokens.
                std::size_t ri = 0;
                for (std::size_t i = 0; i < s.tok.size();) {
                    if (ri < runs.size() && runs[ri].first == i) {
                        age += runs[ri].second - runs[ri].first;
                        open_names.emplace_back(h64(join(s, runs[ri].first, runs[ri].second)), age);
                        i = runs[ri].second;
                        ++ri;
                        continue;
                    }
                    const int pc = pron_class(s.tok[i]);
                    if (pc >= 0) {  // credits the earliest name still within 20 tokens (the subject, most often)
                        for (const auto& [h, a0] : open_names)
                            if (age - a0 <= 20) {
                                ++names_[h].pron[pc];
                                break;
                            }
                        open_names.clear();
                        ++pron_all_[pc];
                    }
                    ++age, ++i;
                }
                // Years against the last year of the previous 6 sentences.
                int last = 0;
                for (std::size_t k = recent_years.size(); k-- > 0 && recent_years.size() - k <= 6;)
                    if (recent_years[k]) {
                        last = recent_years[k];
                        break;
                    }
                int sy = 0;
                for (const auto& t : s.tok)
                    if (const int y = year_of(t)) ++gap_[gap_bucket(y, last)], sy = y;
                recent_years.push_back(sy);
                // Openings and lengths.
                const int ot = open_type(s);
                ++open_[(po2 * (kOpenTypes + 1) + po1) * kOpenTypes + ot];
                po2 = po1, po1 = ot;
                const int lb = len_bucket(s);
                ++len_[pl * kLenB + lb];
                pl = lb;
                for (const auto& t : s.tok) cseq.push_back(cls_.empty() ? 0 : cls_[m.id(t)]);
            }
            doc_name_list[di] = doc_names;
            for (const auto h : doc_names) ++names_[h].docs;
        }
        // Name ids and document co-occurrence (each document's 30 most frequent names).
        std::int32_t nid = 0;
        for (auto& [h, ns] : names_)
            if (ns.docs >= 3) ns.id = nid++;
        nnames_ = static_cast<std::size_t>(nid);
        std::vector<std::uint64_t> pairs;
        for (auto& dn : doc_name_list) {
            std::vector<std::int32_t> ids;
            for (const auto h : dn) {
                const auto it = names_.find(h);
                if (it != names_.end() && it->second.id >= 0) ids.push_back(it->second.id);
            }
            if (ids.size() > 30) ids.resize(30);
            std::sort(ids.begin(), ids.end());
            for (std::size_t i = 0; i < ids.size(); ++i)
                for (std::size_t j = i + 1; j < ids.size(); ++j)
                    pairs.push_back((static_cast<std::uint64_t>(ids[i]) << 32) | static_cast<std::uint32_t>(ids[j]));
            dn.clear(), dn.shrink_to_fit();
        }
        count_sorted(pairs, cooc_);
        // Pass 2: slot types after one and two words.
        slot1_.assign(kSlotH * kSlotT, 0);
        slot2_.assign(kSlotH * kSlotT, 0);
        for (const auto& d : docs) {
            std::string p1 = "<s>", p2 = "<s>";
            for (const auto& s : d) {
                const auto runs = name_runs(s);
                slot_walk(s, runs, [&](std::size_t r) { return type_of_name(join(s, runs[r].first, runs[r].second)); }, p2, p1,
                          [&](int ty, const std::string& q2, const std::string& q1) {
                              ++slot1_[slot_h(q1) * kSlotT + ty];
                              ++slot2_[slot_h(q2 + " " + q1) * kSlotT + ty];
                              ++slot_all_[ty];
                          });
            }
        }
        docs.clear();
        // The class n-gram.
        build_classes(cseq);
        if (log)
            std::fprintf(log, "stats: %zu names (%zu with ids), %zu co-occurring pairs, %zu caps words, class n-grams %zu\n", names_.size(),
                         nnames_, cooc_.size(), caps_.size(), cng_.size());
    }

    // ---- the pieces the experts read

    // Name runs of s: maximal runs of capitalised tokens; a sentence's first token joins only if the word is mostly
    // capitalised away from a sentence's start. [a, b) token ranges.
    std::vector<std::pair<std::size_t, std::size_t>> name_runs(const Sent& s) const {
        std::vector<std::pair<std::size_t, std::size_t>> r;
        for (std::size_t i = 0; i < s.tok.size();) {
            const bool next_cap = i + 1 < s.tok.size() && cap_name(s.tok[i + 1], false);
            if (!cap_name(s.tok[i], i == 0, next_cap)) {
                ++i;
                continue;
            }
            std::size_t j = i + 1;
            while (j < s.tok.size() && cap_name(s.tok[j], false)) ++j;
            r.emplace_back(i, j);
            i = j;
        }
        return r;
    }
    // A sentence's first token is a name if it is mostly capitalised away from a sentence's start, or never seen there
    // and followed by a capitalised token ("John Smith was ...").
    bool cap_name(std::string_view t, bool first, bool next_cap = false) const {
        if (!is_cap(t) || t == "I") return false;
        if (pron_class(t) >= 0) return false;
        if (!first) return true;
        const auto it = caps_.find(h64(lower(t)));
        if (it == caps_.end()) return next_cap;
        return it->second.first > it->second.second;
    }
    // The relations expert's slot walk over s, for its counts (build) and its scores (features) alike: each slot's type
    // (0 a word, run_type(r) for name run r, which is one slot, 3 a year, 4 a number) with the two words before it as
    // shapes, visit(type, p2, p1); then p2, p1 move on.
    template <class RunType, class Visit>
    static void slot_walk(const Sent& s, const std::vector<std::pair<std::size_t, std::size_t>>& runs, RunType&& run_type, std::string& p2, std::string& p1, Visit&& visit) {
        for (std::size_t i = 0, ri = 0; i < s.tok.size(); ++i) {
            int ty = year_of(s.tok[i]) ? 3 : is_number(s.tok[i]) ? 4 : 0;
            std::size_t skip = 0;
            if (ri < runs.size() && runs[ri].first == i) ty = run_type(ri), skip = runs[ri].second - runs[ri].first - 1, ++ri;
            visit(ty, p2, p1);
            p2 = p1, p1 = ty ? std::string(ty == 3 ? "<year>" : ty == 4 ? "<num>" : "<name>") : lower(s.tok[i]);
            i += skip;
        }
    }
    static std::string join(const Sent& s, std::size_t a, std::size_t b) {
        std::string o;
        for (std::size_t i = a; i < b; ++i) {
            if (i > a) o += ' ';
            o += s.tok[i];
        }
        return o;
    }
    const NameStat* name(std::string_view nm) const {
        const auto it = names_.find(h64(nm));
        return it == names_.end() ? nullptr : &it->second;
    }
    int type_of_name(std::string_view nm) const {
        const NameStat* n = name(nm);
        return n ? n->type() : 5;
    }
    int open_type(const Sent& s) const {
        if (s.tok.empty()) return 5;
        const std::string& a = s.tok[0];
        static const std::unordered_set<std::string> conn = {
            "However", "But", "And", "Also", "Thus", "Therefore", "Moreover", "Furthermore", "Meanwhile", "Later", "Then",
            "Although", "While", "As", "Because", "Despite", "Instead", "Nevertheless", "Finally", "Additionally", "Similarly",
            "Consequently", "Hence", "Still", "Yet", "Eventually", "Initially", "Originally", "Subsequently", "Afterwards",
            "Indeed", "Nonetheless", "Likewise", "Accordingly", "Otherwise", "So", "Or", "Unlike", "Like", "Besides", "Overall"};
        static const std::unordered_set<std::string> pron = {"He", "She", "It", "They", "This", "These", "That", "Those",
                                                             "His", "Her", "Its", "Their", "We", "I", "There", "Such"};
        static const std::unordered_set<std::string> prep = {"In", "On", "During", "After", "Before", "By", "Since", "From",
                                                             "Until", "Between", "At", "Around", "Following", "Circa"};
        static const std::unordered_set<std::string> in_conn = {"addition", "contrast", "particular", "fact", "general",
                                                                "practice", "turn", "short", "response", "comparison"};
        if (year_of(a) || is_month(a)) return 3;
        if (prep.count(a)) {
            for (std::size_t i = 1; i < s.tok.size() && i <= 3; ++i)
                if (year_of(s.tok[i]) || is_month(s.tok[i])) return 3;
            if (a == "In" && s.tok.size() > 1 && in_conn.count(s.tok[1])) return 0;
        }
        if (conn.count(a)) return 0;
        if (pron.count(a)) return 1;
        if (a == "The" || a == "A" || a == "An") return 2;
        if (cap_name(a, true, s.tok.size() > 1 && cap_name(s.tok[1], false))) return 4;
        return 5;
    }
    double open_lp(int p2, int p1, int t) const {
        const std::size_t b = static_cast<std::size_t>(p2 * (kOpenTypes + 1) + p1) * kOpenTypes;
        double z = 0;
        for (int k = 0; k < kOpenTypes; ++k) z += open_[b + k] + 1.0;
        return std::log2((open_[b + static_cast<std::size_t>(t)] + 1.0) / z);
    }
    double len_lp(int prev, int b) const {
        double z = 0;
        for (int k = 0; k < kLenB; ++k) z += len_[static_cast<std::size_t>(prev * kLenB + k)] + 1.0;
        return std::log2((len_[static_cast<std::size_t>(prev * kLenB + b)] + 1.0) / z);
    }
    // log2 P(gap bucket), centred so that its mean under S's distribution is 0.
    double gap_score(int b) const {
        double z = 0, e = 0;
        for (int k = 0; k < kGapB; ++k) z += gap_[k] + 1.0;
        for (int k = 0; k < kGapB; ++k) {
            const double p = (gap_[k] + 1.0) / z;
            e += p * std::log2(p);
        }
        return std::log2((gap_[b] + 1.0) / z) - e;
    }
    // log2 P(pronoun class | name) - log2 P(pronoun class), the name's counts smoothed toward its type's.
    double pron_score(const NameStat* n, int pc) const {
        double z = 0;
        for (int k = 0; k < 4; ++k) z += pron_all_[k] + 1.0;
        const double pg = (pron_all_[pc] + 1.0) / z;
        if (!n) return 0;
        double zn = 0;
        for (int k = 0; k < 4; ++k) zn += n->pron[k];
        const double a = 2.0;
        const double p = (n->pron[pc] + a * pg) / (zn + a);
        return std::log2(p / pg);
    }
    // log2 P(type | the two words before) - log2 P(type); backs off to one word, then 0.
    double slot_score(const std::string& p2, const std::string& p1, int ty) const {
        double za = 0;
        for (int k = 0; k < kSlotT; ++k) za += slot_all_[k] + 1.0;
        const double pa = (slot_all_[ty] + 1.0) / za;
        for (int lev = 0; lev < 2; ++lev) {
            const auto& tab = lev == 0 ? slot2_ : slot1_;
            const std::size_t b = (lev == 0 ? slot_h(p2 + " " + p1) : slot_h(p1)) * kSlotT;
            double z = 0;
            for (int k = 0; k < kSlotT; ++k) z += tab[b + static_cast<std::size_t>(k)];
            if (z >= 5) return std::log2((tab[b + static_cast<std::size_t>(ty)] + 0.5) / (z + 0.5 * kSlotT) / pa);
        }
        return 0;
    }
    // Document PPMI of two names (ids), 0 if they never share a document.
    double name_ppmi(std::int32_t a, std::int32_t b, std::uint32_t da, std::uint32_t db) const {
        if (a == b) return 0;
        if (a > b) std::swap(a, b), std::swap(da, db);
        const std::uint64_t k = (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint32_t>(b);
        const auto it = std::lower_bound(cooc_.begin(), cooc_.end(), std::make_pair(k, std::uint32_t{0}));
        if (it == cooc_.end() || it->first != k) return 0;
        return std::max(0.0, std::log2(static_cast<double>(it->second) * static_cast<double>(docs_) / (static_cast<double>(da) * db)));
    }
    std::uint32_t cls(std::uint32_t id) const { return cls_.empty() ? 0 : cls_[id]; }
    // log2 P(c | the classes before it) under the class n-gram (interpolated absolute discounting; add-one unigram).
    // A context's n-grams lie together in cng_ (its followers): their number and their sum are what the context table
    // held. Contexts of one or two classes are indexed densely (low_); a longer one is found from its first two classes
    // by stepping down to each next class's followers (kid_), a short search each time.
    double class_lp(const std::uint32_t* h, std::size_t n, std::uint32_t c) const {
        double p = (uni_[c] + 1.0) / (uni_tot_ + ncls_);
        for (std::size_t m = 1; m < static_cast<std::size_t>(kCO) && m <= n; ++m) {  // the context: the last m classes
            const std::uint32_t* x = h + n - m;
            const Low& l = low_[m == 1 ? x[0] : 512 + (x[0] << 9 | x[1])];
            std::size_t a = l.first, b = a + l.types;
            double tot = l.n;
            if (m > 2) {
                for (std::size_t i = 2; i < m && a < b; ++i) {
                    const std::size_t j = follower(a, b, x[i]);
                    a = j == b ? b : kid_[j - start_[3]];
                    b = j == b ? b : j + 1 < start_[i + 2] ? kid_[j + 1 - start_[3]] : start_[i + 3];
                }
                std::uint64_t t = 0;
                for (std::size_t q = a; q < b; ++q) t += cng_[q].second;
                tot = static_cast<double>(t);
            }
            if (a == b) break;
            const std::size_t e = follower(a, b, c);
            const double cnt = e != b ? cng_[e].second : 0.0;
            p = std::max(cnt - kD, 0.0) / tot + kD * static_cast<double>(b - a) / tot * p;
        }
        return std::log2(p);
    }
    std::size_t docs() const { return docs_; }
    std::size_t bytes() const { return caps_.size() * 40 + names_.size() * 64 + cooc_.size() * 16 + (slot1_.size() + slot2_.size()) * 4 + cng_.size() * 16; }

    // ---- save / load

    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        const auto w = [&](const void* p, std::size_t n) { f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n)); };
        const auto wv = [&](const auto& v) {
            const std::uint64_t n = v.size();
            w(&n, 8);
            if (n) w(v.data(), n * sizeof(v[0]));
        };
        w("K5ST0001", 8);
        const std::uint64_t hd[3] = {docs_, nnames_, ncls_};
        w(hd, sizeof hd);
        wv(cls_);
        std::vector<std::pair<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>>> caps;
        for (const auto& [h, c] : caps_)
            if (c.first + c.second >= 2) caps.emplace_back(h, c);
        wv(caps);
        std::vector<std::pair<std::uint64_t, NameStat>> nm(names_.begin(), names_.end());
        wv(nm);
        wv(cooc_);
        wv(slot1_), wv(slot2_);
        w(slot_all_.data(), sizeof(slot_all_));
        w(gap_.data(), sizeof(gap_)), w(open_.data(), sizeof(open_)), w(len_.data(), sizeof(len_)), w(pron_all_.data(), sizeof(pron_all_));
        wv(uni_);
        w(&uni_tot_, sizeof uni_tot_);
        wv(cng_), wv(contexts());
        return static_cast<bool>(f);
    }
    bool load(const std::string& path, std::string& err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return err = "cannot read " + path, false;
        const auto r = [&](void* p, std::size_t n) { f.read(static_cast<char*>(p), static_cast<std::streamsize>(n)); };
        const auto rv = [&](auto& v) {
            std::uint64_t n = 0;
            r(&n, 8);
            v.resize(n);
            if (n) r(v.data(), n * sizeof(v[0]));
        };
        char mg[8];
        r(mg, 8);
        if (std::string(mg, 8) != "K5ST0001") return err = path + " is not a K5 structure statistics file", false;
        std::uint64_t hd[3];
        r(hd, sizeof hd);
        docs_ = hd[0], nnames_ = hd[1], ncls_ = static_cast<std::uint32_t>(hd[2]);
        rv(cls_);
        std::vector<std::pair<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>>> caps;
        rv(caps);
        caps_.clear();
        caps_.reserve(caps.size());
        for (const auto& [h, c] : caps) caps_.emplace(h, c);
        std::vector<std::pair<std::uint64_t, NameStat>> nm;
        rv(nm);
        names_.clear();
        names_.reserve(nm.size());
        for (const auto& [h, n] : nm) names_.emplace(h, n);
        rv(cooc_);
        rv(slot1_), rv(slot2_);
        r(slot_all_.data(), sizeof(slot_all_));
        r(gap_.data(), sizeof(gap_)), r(open_.data(), sizeof(open_)), r(len_.data(), sizeof(len_)), r(pron_all_.data(), sizeof(pron_all_));
        rv(uni_);
        r(&uni_tot_, sizeof uni_tot_);
        rv(cng_);
        std::uint64_t nctx = 0;  // the contexts: class_lp reads them off cng_, so they are not held
        r(&nctx, 8);
        const auto at = f.tellg();
        if (!f.seekg(0, std::ios::end) || f.tellg() - at != static_cast<std::streamoff>(nctx * sizeof(CtxE))) return err = path + ": truncated", false;
        index_classes();
        return true;
    }

 private:
    static constexpr std::size_t kSlotH = std::size_t{1} << 22;
    static std::size_t slot_h(const std::string& s) { return static_cast<std::size_t>(h64(s) & (kSlotH - 1)); }
    template <class K>
    static void count_sorted(std::vector<K>& v, std::vector<std::pair<K, std::uint32_t>>& out) {
        std::sort(v.begin(), v.end());
        out.clear();
        for (std::size_t i = 0; i < v.size();) {
            std::size_t j = i;
            while (j < v.size() && v[j] == v[i]) ++j;
            out.emplace_back(v[i], static_cast<std::uint32_t>(j - i));
            i = j;
        }
        v.clear(), v.shrink_to_fit();
    }
    using NGram = std::pair<std::uint64_t, std::uint32_t>;  // (order << 58 | classes, 9 bits each), count
    struct CtxE {  // a context as the file holds it: the n-grams' first k - 1 classes, their total count and followers
        std::uint64_t k;
        double n, types;
    };
    struct Low {  // a context of one or two classes: its first n-gram in cng_, their number and their total count
        double n = 0;
        std::uint32_t first = 0, types = 0;
    };
    // start_, low_ and kid_ from cng_ (sorted by order, then context, then last class). Every n-gram of order k + 1
    // follows an n-gram of order k (its first k classes), so one pass per order finds where each one's followers start.
    void index_classes() {
        constexpr std::uint64_t mask = (std::uint64_t{1} << 58) - 1;
        for (int k = 0; k <= kCO; ++k)
            start_[static_cast<std::size_t>(k)] = static_cast<std::size_t>(std::lower_bound(cng_.begin(), cng_.end(), NGram{static_cast<std::uint64_t>(k) << 58, 0}) - cng_.begin());
        start_[kCO + 1] = cng_.size();
        low_.assign(512 + (std::size_t{1} << 18), Low{});
        for (std::size_t i = start_[2]; i < start_[4]; ++i) {
            Low& x = low_[(cng_[i].first >> 58 == 2 ? 0 : 512) + ((cng_[i].first & mask) >> 9)];
            if (!x.types) x.first = static_cast<std::uint32_t>(i);
            x.n += cng_[i].second, ++x.types;
        }
        kid_.resize(start_[kCO] - start_[3]);
        for (std::size_t k = 3; k < kCO; ++k)
            for (std::size_t j = start_[k], q = start_[k + 1]; j < start_[k + 1]; ++j) {
                while (q < start_[k + 2] && (cng_[q].first >> 9) < ((static_cast<std::uint64_t>(k + 1) << 49) | (cng_[j].first & mask))) ++q;
                kid_[j - start_[3]] = static_cast<std::uint32_t>(q);
            }
    }
    // The n-gram in [a, b) (one context's followers) whose last class is c, or b.
    std::size_t follower(std::size_t a, std::size_t b, std::uint32_t c) const {
        const auto it = std::lower_bound(cng_.begin() + static_cast<std::ptrdiff_t>(a), cng_.begin() + static_cast<std::ptrdiff_t>(b), c,
                                         [](const NGram& e, std::uint32_t v) { return (e.first & 511) < v; });
        return it != cng_.begin() + static_cast<std::ptrdiff_t>(b) && (it->first & 511) == c ? static_cast<std::size_t>(it - cng_.begin()) : b;
    }
    std::vector<CtxE> contexts() const {
        std::vector<CtxE> ctx;
        for (const auto& [key, c] : cng_) {
            const std::uint64_t ck = (key >> 58 << 58) | ((key & ((std::uint64_t{1} << 58) - 1)) >> 9);
            if (!ctx.empty() && ctx.back().k == ck) ctx.back().n += c, ctx.back().types += 1;
            else ctx.push_back({ck, static_cast<double>(c), 1});
        }
        return ctx;
    }
    void build_classes(const std::vector<std::uint32_t>& cs) {
        const std::uint32_t bos = cls_.empty() ? 0 : cls_[wkn::kBos];
        uni_.assign(ncls_, 0);
        uni_tot_ = 0;
        for (const auto c : cs)
            if (c != bos) ++uni_[c], ++uni_tot_;
        cng_.clear();
        for (int k = 2; k <= kCO; ++k) {
            std::vector<std::uint64_t> keys;
            keys.reserve(cs.size());
            std::uint64_t win = 0;
            std::size_t run = 0;  // tokens since the last document start (bos)
            for (const auto c : cs) {
                if (c == bos) run = 0, win = 0;
                win = ((win << 9) | c) & ((std::uint64_t{1} << (9 * k)) - 1);
                ++run;
                if (c != bos && run >= static_cast<std::size_t>(k)) keys.push_back((static_cast<std::uint64_t>(k) << 58) | win);
            }
            std::vector<std::pair<std::uint64_t, std::uint32_t>> cnt;
            count_sorted(keys, cnt);
            cng_.insert(cng_.end(), cnt.begin(), cnt.end());
        }
        // Keys of order k sort below order k + 1 (the order is in the top bits), so cng_ stays sorted.
        index_classes();
    }

    std::vector<std::uint32_t> cls_;
    std::uint32_t ncls_ = 1;
    std::size_t docs_ = 0, nnames_ = 0;
    std::unordered_map<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>> caps_;  // (capitalised, lower case) mid-sentence
    std::unordered_map<std::uint64_t, NameStat> names_;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> cooc_;
    std::vector<std::uint32_t> slot1_, slot2_;
    std::array<double, kSlotT> slot_all_{};
    std::array<double, kGapB> gap_{};
    std::array<double, (kOpenTypes + 1) * (kOpenTypes + 1) * kOpenTypes> open_{};
    std::array<double, (kLenB + 1) * kLenB> len_{};
    std::array<double, 4> pron_all_{};
    std::vector<double> uni_;
    double uni_tot_ = 0;
    std::vector<NGram> cng_;
    std::array<std::size_t, kCO + 2> start_{};
    std::vector<Low> low_;           // [0, 512): one class; then two classes (first << 9 | second)
    std::vector<std::uint32_t> kid_;  // for each n-gram of orders 3 and 4 (index - start_[3]): where its followers start
};

// ---------------------------------------------------------------- the experts

// The longest-run sets of K5's training text: hashes of every n-gram of its ids, n = 2..13.
class TrainRuns {
 public:
    static constexpr int kMax = 13;
    void build(const std::vector<std::uint32_t>& ids) {
        for (int n = 2; n <= kMax; ++n) {
            auto& v = h_[static_cast<std::size_t>(n)];
            v.clear();
            v.reserve(ids.size());
            for (std::size_t s = 0; s + static_cast<std::size_t>(n) <= ids.size(); ++s) v.push_back(hash(ids.data() + s, static_cast<std::size_t>(n)));
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            auto& x = at_[static_cast<std::size_t>(n)];  // where each value of the top 16 bits begins: a lookup searches one bucket
            x.assign(kB + 1, 0);
            for (std::size_t b = 0, i = 0; b <= kB; ++b) {
                while (i < v.size() && (v[i] >> 48) < b) ++i;
                x[b] = static_cast<std::uint32_t>(i);
            }
        }
        built_ = true;
    }
    bool built() const { return built_; }
    // The longest run of ids (at most kMax) that is an n-gram of the training text (1 if any id; 0 if empty).
    int longest(const std::vector<std::uint32_t>& ids) const {
        int best = ids.empty() ? 0 : 1;
        for (std::size_t s = 0; s < ids.size(); ++s) {
            int n = best + 1;
            while (n <= kMax && s + static_cast<std::size_t>(n) <= ids.size() && has(static_cast<std::size_t>(n), hash(ids.data() + s, static_cast<std::size_t>(n)))) best = n++;
        }
        return best;
    }

 private:
    static constexpr std::size_t kB = std::size_t{1} << 16;
    bool has(std::size_t n, std::uint64_t h) const {
        if (at_[n].empty()) return false;
        const auto b = static_cast<std::size_t>(h >> 48);
        return std::binary_search(h_[n].begin() + at_[n][b], h_[n].begin() + at_[n][b + 1], h);
    }
    static std::uint64_t hash(const std::uint32_t* p, std::size_t n) {
        std::uint64_t h = wkn::kFnv0;
        for (std::size_t i = 0; i < n; ++i) h = wkn::fnv_step(h, p[i]);
        return h;
    }
    std::array<std::vector<std::uint64_t>, kMax + 1> h_;
    std::array<std::vector<std::uint32_t>, kMax + 1> at_;
    bool built_ = false;
};

// The context of a candidate: the prompt's sentences and the paragraph so far, with what the experts read from it.
struct Context {
    std::vector<Sent> sents;  // prompt sentences, then the paragraph's
    std::size_t n_prompt = 1;
    std::size_t prompt_chars = 0;
    // derived (prepare)
    std::vector<const NameStat*> name_st;  // each context name's statistics, in order of first mention
    const NameStat* main = nullptr;
    bool has_person = false;
    int last_year = 0;
    int po1 = kOpenNone, po2 = kOpenNone, plen = kLenNone;
    std::vector<std::uint32_t> ids, classes;  // every context token's K5 id and class
    std::unordered_set<std::string> name_toks;
    std::string p1 = "<s>", p2 = "<s>";
    std::vector<double> theta;     // LDA's topic mixture of the context
    eg::EntityGrid::History grid;  // the entity grid of the context's sentences
    std::size_t index() const { return sents.size() - n_prompt; }  // the candidate's index in the paragraph
    int prompt_bucket() const { return prompt_chars <= 130 ? 0 : prompt_chars <= 200 ? 1 : 2; }
};

class Experts {
 public:
    Experts(const Stats& st, const wkn::K5& m, const TrainRuns& tr, const lda::LDA* topics = nullptr, const eg::EntityGrid* grid = nullptr)
        : st_(st), m_(m), tr_(tr), lda_(topics && topics->trained() ? topics : nullptr), eg_(grid && grid->trained() ? grid : nullptr) {}

    void prepare(Context& c) const {
        c.name_st.clear(), c.name_toks.clear(), c.ids.clear(), c.classes.clear();
        std::unordered_map<std::string, int> freq;
        std::vector<std::string> order;
        c.last_year = 0, c.has_person = false;
        for (const auto& s : c.sents) {
            for (const auto& [a, b] : st_.name_runs(s)) {
                const std::string nm = Stats::join(s, a, b);
                if (!freq.count(nm)) order.push_back(nm);
                ++freq[nm];
                for (std::size_t i = a; i < b; ++i) c.name_toks.insert(s.tok[i]);
            }
            for (const auto& t : s.tok) {
                if (const int y = year_of(t)) c.last_year = y;
                const std::uint32_t id = m_.id(t);
                c.ids.push_back(id), c.classes.push_back(st_.cls(id));
            }
        }
        if (lda_) {
            std::vector<std::string> all;
            for (const auto& s : c.sents) all.insert(all.end(), s.tok.begin(), s.tok.end());
            c.theta = lda_->infer(all);
        }
        c.grid = eg::EntityGrid::History{};
        if (eg_)
            for (const auto& s : c.sents) eg_->append(c.grid, eg_->roles(s.tok));
        int best = 0;
        c.main = nullptr;
        for (const auto& nm : order) {
            const NameStat* ns = st_.name(nm);
            c.name_st.push_back(ns);
            if (ns && ns->type() == 1) c.has_person = true;
            if (freq[nm] > best) best = freq[nm], c.main = ns;
        }
        const std::size_t n = c.sents.size();
        c.po1 = n >= 1 ? st_.open_type(c.sents[n - 1]) : kOpenNone;
        c.po2 = n >= 2 ? st_.open_type(c.sents[n - 2]) : kOpenNone;
        c.plen = n >= 1 ? len_bucket(c.sents[n - 1]) : kLenNone;
        c.p1 = c.p2 = "<s>";
        if (n >= 1) {
            const auto& s = c.sents[n - 1];
            const auto w = [&](std::size_t i) { return year_of(s.tok[i]) ? std::string("<year>") : is_number(s.tok[i]) ? std::string("<num>") : is_cap(s.tok[i]) && i > 0 ? std::string("<name>") : lower(s.tok[i]); };
            if (!s.tok.empty()) c.p1 = w(s.tok.size() - 1);
            if (s.tok.size() >= 2) c.p2 = w(s.tok.size() - 2);
        }
    }

    // The features of candidate s (its K5 ids) after context c; k5lp and pmi come from the draft (K5's own numbers).
    void features(const Context& c, const Sent& s, const std::vector<std::uint32_t>& ids, double k5lp, double pmi, double* f) const {
        for (int j = 0; j < kNFeat; ++j) f[j] = 0;
        f[kK5] = std::max(k5lp, -25.0), f[kPmi] = pmi;  // a few real words have almost no K5 mass: clipped at 2^-25
        // class: the class n-gram over the context's last 4 classes, then the candidate's, each after the 4 before it
        {
            constexpr std::size_t H = Stats::kCO - 1;
            std::vector<std::uint32_t> h(c.classes.end() - static_cast<std::ptrdiff_t>(std::min(c.classes.size(), H)), c.classes.end());
            const std::size_t k0 = h.size();
            for (const auto id : ids) h.push_back(st_.cls(id));
            double lp = 0;
            for (std::size_t p = k0; p < h.size(); ++p) lp += st_.class_lp(h.data() + (p > H ? p - H : 0), std::min(p, H), h[p]);
            f[kClass] = ids.empty() ? 0 : lp / static_cast<double>(ids.size());
        }
        // relations: each name run's statistics are looked up once, for its co-occurrence and for its slot type
        const auto runs = st_.name_runs(s);
        {
            int old = 0, nw = 0, nslot = 0;
            double pm = 0, slot = 0;
            int npm = 0;
            std::vector<const NameStat*> run_st;
            for (const auto& [a, b] : runs) {
                const NameStat* ns = run_st.emplace_back(st_.name(Stats::join(s, a, b)));
                bool seen = false;
                for (std::size_t i = a; i < b && !seen; ++i) seen = c.name_toks.count(s.tok[i]) > 0;
                if (seen) {
                    ++old;
                } else {
                    ++nw;
                    double bp = 0;
                    if (ns && ns->id >= 0)
                        for (const NameStat* cs : c.name_st)
                            if (cs && cs->id >= 0) bp = std::max(bp, st_.name_ppmi(ns->id, cs->id, ns->docs, cs->docs));
                    pm += bp, ++npm;
                }
            }
            f[kEntOld] = std::log1p(old), f[kEntNew] = std::log1p(nw);
            f[kEntPmi] = npm ? pm / npm : 0;
            // slots: name-run starts, years, numbers
            std::string p1 = c.p1, p2 = c.p2;
            Stats::slot_walk(s, runs, [&](std::size_t r) { return run_st[r] ? run_st[r]->type() : 5; }, p2, p1, [&](int ty, const std::string& q2, const std::string& q1) {
                if (ty) slot += st_.slot_score(q2, q1, ty), ++nslot;
            });
            f[kSlotFit] = nslot ? slot / nslot : 0;
        }
        // reference and time, in one pass over the tokens
        {
            double ag = 0, g = 0;
            int np = 0, ny = 0;
            std::size_t first_g = s.tok.size();  // the first he or she
            for (std::size_t i = 0; i < s.tok.size(); ++i) {
                if (const int y = year_of(s.tok[i])) g += st_.gap_score(gap_bucket(y, c.last_year)), ++ny;
                const int pc = pron_class(s.tok[i]);
                if (pc < 0) continue;
                if (pc <= 1 && first_g == s.tok.size()) first_g = i;
                if (c.main) ag += st_.pron_score(c.main, pc), ++np;
            }
            f[kPronAgree] = np ? ag / np : 0;
            // he/she with no person in the context and no name before it in the sentence
            if (first_g < s.tok.size() && !c.has_person) f[kPronOrphan] = !runs.empty() && runs[0].first < first_g ? 0 : 1;
            f[kYearFit] = ny ? g / ny : 0;
            f[kNYears] = std::log1p(ny);
        }
        // transitions
        f[kOpen] = st_.open_lp(c.po2, c.po1, st_.open_type(s));
        // rhythm
        {
            f[kLen] = st_.len_lp(c.plen, len_bucket(s));
            int par = 0, br = 0, q = 0;
            std::size_t mk = 0, nb = 0;
            std::string prev = c.sents.empty() || c.sents.back().tok.empty() ? std::string() : c.sents.back().tok.back();
            bool hp = !prev.empty();
            for (const auto& t : s.tok) {
                const std::string pc = wkn::piece(t, prev, hp);
                prev = t, hp = true;
                nb += pc.size();
                for (const char ch : pc) {
                    par += ch == '(' ? 1 : ch == ')' ? -1 : 0;
                    br += ch == '[' ? 1 : ch == ']' ? -1 : 0;
                    q += ch == '"';
                    mk += std::string_view("[]{}|=<>&*#").find(ch) != std::string_view::npos;
                }
            }
            f[kBalanced] = par == 0 && br == 0 && q % 2 == 0 ? 1 : 0;
            f[kMarkup] = nb ? static_cast<double>(mk) / static_cast<double>(nb) : 0;
        }
        // novelty
        f[kMatchTrain] = tr_.longest(ids);
        {
            int best = 0;
            const auto& a = c.ids;
            std::vector<int> prev(ids.size() + 1, 0), cur(ids.size() + 1, 0);
            for (std::size_t i = 0; i < a.size(); ++i) {
                for (std::size_t j = 0; j < ids.size(); ++j) {
                    cur[j + 1] = a[i] == ids[j] ? prev[j] + 1 : 0;
                    best = std::max(best, cur[j + 1]);
                }
                std::swap(prev, cur);
            }
            f[kMatchCtx] = best;
        }
        // topic and grid (wip/k5-discourse); topic counts the content words and sums their log ratio in one pass
        if (lda_ && !c.theta.empty()) {
            double r = 0;
            std::size_t n = 0;
            for (const auto& t : s.tok)
                if (const long x = lda_->id(t); x >= 0) r += std::log2(lda_->p_topic(c.theta, static_cast<std::size_t>(x)) / lda_->unigram(static_cast<std::size_t>(x))), ++n;
            f[kTopic] = n ? r / static_cast<double>(n) : 0;
        }
        if (eg_) {
            const auto r = eg_->roles(s.tok);
            f[kGridFull] = eg_->append_logp(c.grid, r);
            f[kGridCont] = eg_->continuation_score(c.grid, r);
        }
    }

    const Stats& stats() const { return st_; }
    const wkn::K5& k5() const { return m_; }

 private:
    const Stats& st_;
    const wkn::K5& m_;
    const TrainRuns& tr_;
    const lda::LDA* lda_;
    const eg::EntityGrid* eg_;
};

}  // namespace cypha::cyphalm::structure
