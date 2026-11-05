#pragma once

/// The gibberish filter for the planner's fact store (docs/reports/PLAN_FACTS.md, GENERATION_PLAN §13.42 Q1).
///   sample --k5 ... --k5-plan STORE --k5-plan-filter PARAMS|on
///
/// Every rule reads CyphaLM's own store and its own structure statistics only (no list of facts, no outside source):
///   shape      the object holds markup ([ ] { } | < > = * # _ '' http &x;), starts or ends with stray punctuation, has
///              unbalanced brackets or quotes, or starts or ends with a function word (a fragment: "of the", "and"); or
///              the subject is a list or disambiguation page, or holds markup;
///   self       the object is the subject, a part of it ("Horned owl" order "owl"; a place or lexicon value may sit
///              inside the subject: "Berkeley, California" located_in "California"), or holds the subject; not for
///              developer, publisher and owner (a firm names its products: "Microsoft Word" developer "Microsoft");
///   year       a year relation's object is not a year (1-4 digits, "BC" allowed, 3500 BC to 2006), or the subject's
///              birth and death years are more than 125 years apart, or death comes before birth (except for years up
///              to 1000, where "BC" is often missing: Empedocles 490-430);
///   place      a place relation's object is not known as a place: neither a subject of a place relation in the store
///              (located_in, country, capital, largest_city, location), nor the object of place relations of at least
///              place_support subjects, nor (with the structure statistics, structure/experts.hpp) a name that mostly
///              follows "in", "at", "near" ... in running text (the relations expert's place type); or it is known as a
///              person in the store (a subject with a birth or death year);
///   person     a person relation's object is not a capitalised name of at most 5 words (particles such as "de", "van"
///              allowed), or is known as a place; with person >= 2, also not seen as a subject in the store;
///   name       a name relation's object (developer, publisher, owner) does not start with a capital or a digit;
///   lex        a lexicon relation's object (occupation, nationality, genre, language, ...) is given to fewer than
///              lex_support subjects, holds a digit, or (nationality, language) is known as a place;
///   taxon      family, order or class: the object holds a digit, has more than 2 words, or does not start with a letter;
///   agree      a functional relation (one true value: capital, birth_year, family, ...) with another object for the same
///              subject that more rows of the store support;
///   xrel       a lexicon object (not is_a) read from a category, a lead or prose (not an infobox or table field, which
///              names its relation) that the store gives to more subjects under another lexicon relation ("Blues" as a
///              nationality, more often a genre), or that is known as a place; nationality and language share values
///              ("French") and are not compared;
///   spread     a nationality read together with fewer than spread_min distinct occupations over the store (the
///              occupation read at the same place: one category, "English poets"); a real nationality goes with many
///              ("English" poets, actors, ...), a category word read as one with few ("Baroque" composers, "Film" actors);
///   p          an F2 prose fact with P below min_p.
/// Which rules apply and their thresholds are FilterParams, read from a file of "key value" lines (or "on": the
/// defaults below). They were chosen on half of a blind-labelled sample of 300 store facts (bench/fact_filter/).
///
/// Author: Odin Loch

#include "cypha/lm/structure/experts.hpp"  // read only: the relations expert's name types (place rule)

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cypha::cyphalm::planner {

// One fact as the planner holds it (canonical relation and value, markup stripped), with where it came from.
struct FactRow {
    std::string subj, rel, val;  // rel and val canonical (birth_date -> birth_year with the year)
    std::string src;             // infobox | table | category | lead | wikidata | f2
    std::string tpl, pattern;    // the seed's template, or F2's best pattern
    double p = 1.0;              // F2's P (1 for a seed)
    int n_ev = 1;                // F2's evidence count
    int n_rows = 1;              // raw rows of the store behind this (subject, relation, value)
    std::size_t off = 0;         // byte offset of the first evidence
};

struct FilterParams {
    int shape = 1, self = 1, year = 1, place = 1, person = 1, name = 1, lex = 1, taxon = 1, agree = 0, xrel = 1, spread = 0;
    int place_support = 2;  // a value of place relations for at least this many subjects counts as a known place
    int lex_support = 2;    // a lexicon value must be given to at least this many subjects
    int spread_min = 2;     // a nationality's subjects must have at least this many distinct occupations
    double min_p = 0.0;     // F2 facts below this P are dropped
    bool read(const std::string& path, std::string& err) {
        if (path == "on") return true;
        std::ifstream in(path);
        if (!in) return err = "cannot read the filter parameters " + path, false;
        std::string k;
        double v;
        while (in >> k >> v) {
            int* const ints[] = {&shape, &self, &year, &place, &person, &name, &lex, &taxon, &agree, &xrel, &spread, &place_support, &lex_support, &spread_min};
            const char* const names[] = {"shape", "self", "year", "place", "person", "name", "lex", "taxon", "agree", "xrel", "spread", "place_support",
                                         "lex_support", "spread_min"};
            bool ok = false;
            for (std::size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
                if (k == names[i]) *ints[i] = static_cast<int>(v), ok = true;
            if (k == "min_p") min_p = v, ok = true;
            if (!ok) return err = path + ": unknown filter key " + k, false;
        }
        return true;
    }
    std::string str() const {
        std::ostringstream o;
        o << "shape " << shape << "\nself " << self << "\nyear " << year << "\nplace " << place << "\nperson " << person << "\nname " << name
          << "\nlex " << lex << "\ntaxon " << taxon << "\nagree " << agree << "\nxrel " << xrel << "\nspread " << spread << "\nplace_support "
          << place_support << "\nlex_support " << lex_support << "\nspread_min " << spread_min << "\nmin_p " << min_p << "\n";
        return o.str();
    }
};

class FactFilter {
 public:
    enum Reason : std::uint32_t {
        kShape = 1u << 0, kSelf = 1u << 1, kYear = 1u << 2, kPlace = 1u << 3, kPerson = 1u << 4, kName = 1u << 5,
        kLex = 1u << 6, kTaxon = 1u << 7, kAgree = 1u << 8, kLowP = 1u << 9, kXrel = 1u << 10, kSpread = 1u << 11, kNReason = 12
    };
    static const char* reason_name(int b) {
        static const char* n[kNReason] = {"shape", "self", "year", "place", "person", "name", "lex", "taxon", "agree", "p", "xrel", "spread"};
        return b >= 0 && b < kNReason ? n[b] : "?";
    }
    static std::string lower(std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }
    static std::vector<std::string> words(const std::string& s) {
        std::vector<std::string> w;
        std::istringstream is(s);
        for (std::string x; is >> x;) w.push_back(x);
        return w;
    }
    // The relation kinds (pattern_learn's relation types).
    static int kind(const std::string& r) {  // 0 other, 1 year, 2 place, 3 person, 4 name, 5 lexicon, 6 taxon
        static const std::unordered_map<std::string, int> K = {
            {"birth_year", 1}, {"death_year", 1}, {"founded", 1}, {"capital", 2}, {"largest_city", 2}, {"country", 2}, {"location", 2},
            {"located_in", 2}, {"birth_place", 2}, {"death_place", 2}, {"author", 3}, {"director", 3}, {"composer", 3}, {"founder", 3},
            {"creator", 3}, {"spouse", 3}, {"predecessor", 3}, {"successor", 3}, {"inventor", 3}, {"developer", 4}, {"publisher", 4},
            {"owner", 4}, {"genre", 5}, {"language", 5}, {"currency", 5}, {"occupation", 5}, {"nationality", 5}, {"party", 5},
            {"is_a", 5}, {"family", 6}, {"order", 6}, {"class", 6}};
        const auto it = K.find(r);
        return it == K.end() ? 0 : it->second;
    }
    static bool functional(const std::string& r) {
        static const std::unordered_set<std::string> F = {"capital", "largest_city", "birth_place", "death_place", "birth_year", "death_year",
                                                          "founded", "currency", "family", "order", "class"};
        return F.count(r) > 0;
    }
    static bool is_year(const std::string& v, int& y) {
        std::string s = v;
        bool bc = false;
        if (s.size() > 3 && s.compare(s.size() - 3, 3, " BC") == 0) bc = true, s = s.substr(0, s.size() - 3);
        if (s.empty() || s.size() > 4) return false;
        for (const char c : s)
            if (!std::isdigit(static_cast<unsigned char>(c))) return false;
        y = std::atoi(s.c_str()) * (bc ? -1 : 1);
        return y >= -3500 && y <= 2006 && !(y == 0 && !bc);
    }

    explicit FactFilter(FilterParams p = {}, const structure::Stats* st = nullptr) : p_(p), st_(st) {}
    const FilterParams& params() const { return p_; }
    // The relations expert's place type for a value (the whole value, or its part before a comma: "Amherst, Nova Scotia").
    bool typed_place(const std::string& v) const {
        if (!st_) return false;
        const auto key = [](const std::string& x) {
            std::string o;
            for (const auto& t : wkn::tokens(x)) o += (o.empty() ? "" : " ") + t;
            return o;
        };
        if (st_->type_of_name(key(v)) == 2) return true;
        const std::size_t c = v.find(',');
        return c != std::string::npos && c > 0 && st_->type_of_name(key(v.substr(0, c))) == 2;
    }

    // Statistics over the rows the planner would load (after the drops).
    void prepare(const std::vector<FactRow>& rows) {
        rows_ = &rows;
        place_subj_.clear(), place_val_.clear(), person_.clear(), subj_.clear(), lex_n_.clear(), years_.clear(), best_.clear(), spread_n_.clear();
        std::unordered_map<std::string, std::set<std::string>> pv;   // place value -> subjects
        std::unordered_map<std::string, std::set<std::string>> lv;   // rel \x1f value -> subjects
        // occupations read at the same place as a nationality (one category, "English poets": the same subject and byte offset)
        const auto at = [](const std::string& s, std::size_t off) { return s + '\x1f' + std::to_string(off); };
        std::unordered_map<std::string, std::set<std::string>> occ;  // subject \x1f offset -> occupations read there
        for (const auto& r : rows)
            if (r.rel == "occupation") occ[at(lower(r.subj), r.off)].insert(lower(r.val));
        std::unordered_map<std::string, std::set<std::string>> nat_occ;  // nationality -> the occupations read with it
        for (const auto& r : rows) {
            const std::string s = lower(r.subj), v = lower(r.val);
            subj_.insert(s);
            if (r.rel == "nationality") {
                auto& no = nat_occ[v];
                if (const auto it = occ.find(at(s, r.off)); it != occ.end()) no.insert(it->second.begin(), it->second.end());
            }
            const int k = kind(r.rel);
            if (k == 2 && r.rel != "birth_place" && r.rel != "death_place") place_subj_.insert(s);
            if (k == 2) pv[v].insert(s);
            if (r.rel == "birth_year" || r.rel == "death_year") person_.insert(s);
            if (k == 5 || k == 6) lv[r.rel + '\x1f' + v].insert(s);
            int y;
            if ((r.rel == "birth_year" || r.rel == "death_year") && is_year(r.val, y)) years_[s + '\x1f' + r.rel].push_back(y);
            if (functional(r.rel)) {
                auto& b = best_[s + '\x1f' + r.rel];
                b = std::max(b, r.n_rows);
            }
        }
        for (const auto& [v, ss] : pv) place_val_[v] = static_cast<int>(ss.size());
        for (const auto& [v, ss] : nat_occ) spread_n_[v] = static_cast<int>(ss.size());
        val_rels_.clear();
        for (const auto& [k, ss] : lv) {
            lex_n_[k] = static_cast<int>(ss.size());
            const std::size_t c = k.find('\x1f');
            val_rels_[k.substr(c + 1)][k.substr(0, c)] = static_cast<int>(ss.size());
        }
    }

    // The reasons row i is gibberish (0: keep).
    std::uint32_t flags(std::size_t i) const { return flags_of((*rows_)[i]); }
    bool keep(std::size_t i) const { return flags(i) == 0; }

    std::uint32_t flags_of(const FactRow& r) const {
        std::uint32_t f = 0;
        const std::string& v = r.val;
        const std::string lv = lower(v), ls = lower(r.subj);
        const auto w = words(v);
        const int k = kind(r.rel);
        if (p_.shape && (bad_shape(v, w) || ls.compare(0, 8, "list of ") == 0 || ls.compare(0, 9, "lists of ") == 0 ||
                         ls.find("disambiguation") != std::string::npos || ls.find_first_of("[]{}|<>=") != std::string::npos))
            f |= kShape;  // a list page or a disambiguation page is not a subject
        // "Berkeley, California" located_in "California" and "Horned owl" is_a "owl" are fine: the value may sit inside the
        // subject for places and lexicon values only; a firm names its products ("Microsoft Word" developer "Microsoft")
        const bool in_subj = (k == 0 || k == 3 || k == 6) && contains_words(ls, lv);
        const bool has_subj = k != 1 && k != 4 && k != 5 && contains_words(lv, ls);
        if (p_.self && (lv == ls || in_subj || has_subj)) f |= kSelf;
        if (p_.year && k == 1) {
            int y;
            if (!is_year(v, y)) f |= kYear;
            else if (r.rel != "founded") {
                const std::string other = r.rel == "birth_year" ? "death_year" : "birth_year";
                const auto it = years_.find(ls + '\x1f' + other);
                if (it != years_.end()) {
                    bool any_ok = false;
                    for (const int z : it->second) {  // ancient years may run backwards: "BC" is often missing (Empedocles 490-430)
                        const int life = r.rel == "birth_year" ? z - y : y - z;
                        any_ok = any_ok || (life >= 0 && life <= 125) || (life < 0 && life >= -125 && std::max(y, z) <= 1000);
                    }
                    if (!any_ok) f |= kYear;
                }
            }
        }
        if (p_.place && k == 2) {
            const auto it = place_val_.find(lv);
            const bool known = place_subj_.count(lv) || (it != place_val_.end() && it->second >= p_.place_support) || typed_place(v);
            if (!known || (person_.count(lv) && !place_subj_.count(lv))) f |= kPlace;
        }
        if (p_.person && k == 3) {
            if (!name_shape(w, 5) || place_subj_.count(lv) || (p_.person >= 2 && !subj_.count(lv))) f |= kPerson;
        }
        if (p_.name && k == 4) {
            const unsigned char c = v.empty() ? 0 : static_cast<unsigned char>(v[0]);
            if (!(std::isupper(c) || std::isdigit(c) || c >= 0x80)) f |= kName;
        }
        if (p_.lex && k == 5) {
            const auto it = lex_n_.find(r.rel + '\x1f' + lv);
            const int n = it == lex_n_.end() ? 0 : it->second;
            bool digit = false;
            for (const char c : v) digit = digit || std::isdigit(static_cast<unsigned char>(c));
            if (n < p_.lex_support || digit || ((r.rel == "nationality" || r.rel == "language") && place_subj_.count(lv))) f |= kLex;
        }
        if (p_.taxon && k == 6) {
            const unsigned char c = v.empty() ? 0 : static_cast<unsigned char>(v[0]);
            bool digit = false;
            for (const char x : v) digit = digit || std::isdigit(static_cast<unsigned char>(x));
            if (w.empty() || w.size() > 2 || digit || !(std::isalpha(c) || c >= 0x80)) f |= kTaxon;
        }
        if (p_.agree && functional(r.rel)) {
            const auto it = best_.find(ls + '\x1f' + r.rel);
            if (it != best_.end() && it->second > r.n_rows) f |= kAgree;
        }
        if (r.src == "f2" && r.p < p_.min_p) f |= kLowP;
        // the value is given more often under another lexicon relation; an infobox or table field names its relation itself;
        // nationality and language share values ("French")
        if (p_.xrel && (k == 5 || k == 6) && r.rel != "is_a" && r.src != "infobox" && r.src != "table") {
            const auto it = lex_n_.find(r.rel + '\x1f' + lv);
            const int n = it == lex_n_.end() ? 0 : it->second;
            const auto jt = val_rels_.find(lv);
            const bool natlang = r.rel == "nationality" || r.rel == "language";
            int other = 0;
            if (jt != val_rels_.end())
                for (const auto& [rr, m] : jt->second)
                    if (rr != r.rel && rr != "is_a" && !(natlang && (rr == "nationality" || rr == "language"))) other = std::max(other, m);
            if (other > n || place_subj_.count(lv)) f |= kXrel;
        }
        if (p_.spread && r.rel == "nationality") {  // a nationality goes with many occupations
            const auto it = spread_n_.find(lv);
            if ((it == spread_n_.end() ? 0 : it->second) < p_.spread_min) f |= kSpread;
        }
        return f;
    }

 private:
    static bool contains_words(const std::string& hay, const std::string& needle) {  // needle's words, in order, inside hay
        if (needle.empty() || needle.size() >= hay.size() + 1) return needle == hay;
        const std::string h = " " + hay + " ", n = " " + needle + " ";
        return h.find(n) != std::string::npos;
    }
    static bool fn_word(const std::string& x) {
        static const std::unordered_set<std::string> F = {"of", "the", "and", "or", "in", "on", "at", "by", "for", "from", "to", "with", "a",
                                                          "an", "as", "is", "was", "that", "which", "who", "but", "its", "his", "her",
                                                          "their", "this", "these", "than"};
        return F.count(x) > 0;
    }
    static bool bad_shape(const std::string& v, const std::vector<std::string>& w) {
        if (v.empty() || w.empty()) return true;
        for (const char* m : {"[", "]", "{", "}", "|", "<", ">", "=", "*", "#", "_", "''", "http", "www.", "\\"})
            if (v.find(m) != std::string::npos) return true;
        for (std::size_t i = 0; i + 1 < v.size(); ++i)  // an XML or HTML entity (&nbsp;)
            if (v[i] == '&' && std::isalpha(static_cast<unsigned char>(v[i + 1]))) {
                std::size_t j = i + 1;
                while (j < v.size() && std::isalnum(static_cast<unsigned char>(v[j]))) ++j;
                if (j < v.size() && v[j] == ';') return true;
            }
        const unsigned char a = static_cast<unsigned char>(v.front()), z = static_cast<unsigned char>(v.back());
        if (std::ispunct(a) && a != '\'' && a != '"' && a != '(') return true;
        if (std::ispunct(z) && z != ')' && z != '.' && z != '\'' && z != '"' && z != '!' && z != '?') return true;
        int par = 0, quo = 0;
        for (const char c : v) {
            par += c == '(' ? 1 : c == ')' ? -1 : 0;
            if (par < 0) return true;
            quo += c == '"';
        }
        if (par != 0 || quo % 2) return true;
        if (v.size() == 1 && !std::isdigit(a)) return true;
        if (fn_word(lower(w.front())) && w.size() > 1) return true;
        if (fn_word(lower(w.back())) && w.size() > 1) return true;
        return false;
    }
    static bool name_shape(const std::vector<std::string>& w, std::size_t max_words) {
        if (w.empty() || w.size() > max_words) return false;
        static const std::unordered_set<std::string> particle = {"de", "da", "di", "del", "della", "der", "den", "van", "von", "la", "le",
                                                                 "du", "dos", "das", "of", "the", "y", "e", "al", "bin", "ibn", "el", "zu",
                                                                 "ter", "ten", "af", "and"};
        bool cap0 = false;
        for (std::size_t i = 0; i < w.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(w[i][0]);
            const bool cap = std::isupper(c) || c >= 0x80;
            if (i == 0) cap0 = cap;
            if (!cap && !(i > 0 && particle.count(w[i]))) return false;
        }
        return cap0;
    }
    FilterParams p_;
    const structure::Stats* st_ = nullptr;
    const std::vector<FactRow>* rows_ = nullptr;
    std::unordered_set<std::string> place_subj_, person_, subj_;
    std::unordered_map<std::string, int> place_val_, lex_n_, best_, spread_n_;
    std::unordered_map<std::string, std::vector<int>> years_;
    std::unordered_map<std::string, std::map<std::string, int>> val_rels_;  // lexicon value -> relation -> subjects
};

}  // namespace cypha::cyphalm::planner
