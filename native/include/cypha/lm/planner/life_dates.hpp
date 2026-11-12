#pragma once

/// Life dates for the planner's honest fill (docs/reports/HONEST_DATES2.md, H3; --k5-plan-honest life).
///
/// H2 (HONEST_DATES.md) let the fill write a number or a month only if it is one of the fact's own words. The store's
/// facts hold years, so every date skeleton ("S (born [month] [day], O)") failed and its sentence was dropped. H3 gives
/// the fill the dates the store holds and the skeletons that ask only for those:
///
/// - DateClasses: K5's month and number classes, read from K5 itself. A class is a month class if month names hold at
///   least half its mass under P(w | class), a number class if words with a digit do.
/// - LifeBook: each store subject's birth and death dates as the store's own rows give them (facts_f2.tsv and
///   seeds.tsv, the planner's drops applied). FactStore keeps only the year of a full date (CPS::canon); the book keeps
///   the month and day too, but only for a year FactStore holds as the subject's one birth or death year.
/// - trim_dates: the trim rule. A mined skeleton of birth_year or death_year whose class positions hold a date gives
///   variants without the month and day ("(born [month] [day], O)" -> "(born O)") and without the second date
///   ("([month] [day], [year] - [month] [day], O)" -> "([month] [day], O)").
/// - skel_fit: a skeleton asks for more months or numbers than the book holds for the fact's subject.
///
/// Author: Odin Loch

#include "cypha/lm/content_plan.hpp"
#include "cypha/lm/planner/facts.hpp"
#include "cypha/lm/planner/skeleton.hpp"
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm::planner {

inline constexpr int kOtherClass = 0, kMonthClass = 1, kNumberClass = 2;

inline const char* const* month_names() {
    static const char* const m[] = {"January", "February", "March", "April", "May", "June", "July",
                                    "August", "September", "October", "November", "December"};
    return m;
}
// 1..12 for a month name, else 0.
inline int month_no(const std::string& t) {
    for (int i = 0; i < 12; ++i)
        if (t == month_names()[i]) return i + 1;
    return 0;
}
inline bool month_word(const std::string& t) { return month_no(t) != 0; }
inline bool has_digit(const std::string& t) {
    for (const char ch : t)
        if (ch >= '0' && ch <= '9') return true;
    return false;
}
// A year as a sentence writes it: three or four digits.
inline bool year_word(const std::string& t) {
    if (t.size() < 3 || t.size() > 4) return false;
    for (const char ch : t)
        if (ch < '0' || ch > '9') return false;
    return true;
}
inline bool life_rel(const std::string& r) { return r == "birth_year" || r == "death_year"; }

class DateClasses {
 public:
    DateClasses() = default;
    explicit DateClasses(const wkn::K5& m) {
        const auto& cls = m.classes().classes();
        const auto& pwc = m.classes().pwc();
        std::size_t K = 0;
        for (std::uint32_t w = 2; w < cls.size(); ++w) K = std::max<std::size_t>(K, cls[w] + 1);
        mon_.assign(K, 0.0), num_.assign(K, 0.0), tot_.assign(K, 0.0);
        for (std::uint32_t w = 2; w < cls.size() && w < pwc.size(); ++w) {
            const std::string& t = m.vocab()[w];
            tot_[cls[w]] += pwc[w];
            if (month_word(t)) mon_[cls[w]] += pwc[w];
            else if (has_digit(t)) num_[cls[w]] += pwc[w];
        }
        const auto cl = [&](const char* t) {
            const std::uint32_t id = m.id(t);
            return id == wkn::kUnk || id >= cls.size() ? 0xFFFFu : static_cast<std::uint16_t>(cls[id]);
        };
        open_ = cl("("), close_ = cl(")"), comma_ = cl(",");
        kinds_.resize(K);
        for (std::size_t c = 0; c < K; ++c) kinds_[c] = month_mass(static_cast<std::uint32_t>(c)) >= 0.5 ? kMonthClass : number_mass(static_cast<std::uint32_t>(c)) >= 0.5 ? kNumberClass : kOtherClass;
    }
    // For tests: the kinds of classes 0..n-1 and the classes of "(", ")" and ",".
    DateClasses(std::vector<int> kinds, std::uint16_t open, std::uint16_t close, std::uint16_t comma)
        : open_(open), close_(close), comma_(comma), kinds_(std::move(kinds)) {}
    double month_mass(std::uint32_t c) const { return c < tot_.size() && tot_[c] > 0 ? mon_[c] / tot_[c] : 0.0; }
    double number_mass(std::uint32_t c) const { return c < tot_.size() && tot_[c] > 0 ? num_[c] / tot_[c] : 0.0; }
    int kind(std::uint32_t c) const { return c < kinds_.size() ? kinds_[c] : kOtherClass; }
    // The classes of "(", ")" and "," (0xFFFF if K5 does not know the word).
    std::uint16_t open() const { return open_; }
    std::uint16_t close() const { return close_; }
    std::uint16_t comma() const { return comma_; }
    // For a skeleton (class codes): its month positions and its number positions.
    std::pair<int, int> asks(const std::vector<std::uint16_t>& codes) const {
        int mo = 0, nu = 0;
        for (const auto c : codes) {
            if (c == kSlotS || c == kSlotO) continue;
            const int k = kind(c);
            mo += k == kMonthClass, nu += k == kNumberClass;
        }
        return {mo, nu};
    }

 private:
    std::vector<double> mon_, num_, tot_;
    std::uint16_t open_ = 0xFFFF, close_ = 0xFFFF, comma_ = 0xFFFF;
    std::vector<int> kinds_;
};

// ---------------------------------------------------------------- the trim rule

// A skeleton's date units: maximal runs of month, number and O positions with commas only inside a run (for a life
// relation, O is a year). Each unit is [a, b) in codes.
inline std::vector<std::pair<std::size_t, std::size_t>> date_units(const DateClasses& dc, const std::vector<std::uint16_t>& codes) {
    std::vector<std::pair<std::size_t, std::size_t>> u;
    const auto dated = [&](std::size_t i) { return codes[i] == kSlotO || (codes[i] != kSlotS && dc.kind(codes[i]) != kOtherClass); };
    for (std::size_t i = 0; i < codes.size();) {
        if (!dated(i)) {
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        for (;;) {
            if (j < codes.size() && dated(j)) ++j;
            else if (j + 1 < codes.size() && codes[j] == dc.comma() && dated(j + 1)) j += 2;
            else break;
        }
        u.push_back({i, j});
        i = j;
    }
    return u;
}

// The trim rule (HONEST_DATES2.md). For a life relation's skeleton whose O sits in a date unit inside brackets
// ("(" ... O ... ")", no bracket between), only the units inside those brackets change:
//   A, year only: a unit holding O keeps only O; another unit keeps only its last position if that is a number (its
//      year), else it goes.
//   B, no second date: the units after O's unit go, with the one position between (the dash of "(O - date)"). A unit
//      before O's is kept: without it, "(date - O)" would read as a birth date.
//   AB: A, then B.
// Variants equal to the skeleton itself, variants where O is left alone in brackets ("( O )"), and variants shorter
// than the bank's minimum length are not kept.
inline std::vector<std::vector<std::uint16_t>> trim_dates(const DateClasses& dc, const std::vector<std::uint16_t>& codes, bool life) {
    std::vector<std::vector<std::uint16_t>> out;
    if (!life) return out;
    const auto units = date_units(dc, codes);
    std::ptrdiff_t ou = -1;
    for (std::size_t k = 0; k < units.size(); ++k)
        for (std::size_t i = units[k].first; i < units[k].second; ++i)
            if (codes[i] == kSlotO) ou = static_cast<std::ptrdiff_t>(k);
    if (ou < 0) return out;
    const auto [oa, ob] = units[static_cast<std::size_t>(ou)];
    std::ptrdiff_t pa = -1, pb = -1;  // the brackets around O's unit
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(oa) - 1; i >= 0 && pa < 0; --i) {
        if (codes[static_cast<std::size_t>(i)] == dc.close()) break;
        if (codes[static_cast<std::size_t>(i)] == dc.open()) pa = i;
    }
    for (std::size_t i = ob; i < codes.size() && pb < 0; ++i) {
        if (codes[i] == dc.open()) break;
        if (codes[i] == dc.close()) pb = static_cast<std::ptrdiff_t>(i);
    }
    if (pa < 0 || pb < 0) return out;
    const auto variant = [&](bool A, bool B) {
        std::vector<char> keep(codes.size(), 1);
        for (std::size_t k = 0; k < units.size(); ++k) {
            const auto [a, b] = units[k];
            if (static_cast<std::ptrdiff_t>(a) <= pa || static_cast<std::ptrdiff_t>(b) > pb) continue;  // outside the brackets
            const bool has_o = static_cast<std::ptrdiff_t>(k) == ou;
            if (B && a >= ob) {
                for (std::size_t i = a; i < b; ++i) keep[i] = 0;
                if (ob + 1 == a) keep[ob] = 0;  // "O - date"
                continue;
            }
            if (!A) continue;
            if (has_o) {
                for (std::size_t i = a; i < b; ++i) keep[i] = codes[i] == kSlotO;
            } else {
                const bool year = dc.kind(codes[b - 1]) == kNumberClass;
                for (std::size_t i = a; i < b; ++i) keep[i] = year && i + 1 == b;
            }
        }
        std::vector<std::uint16_t> v;
        for (std::size_t i = 0; i < codes.size(); ++i)
            if (keep[i]) v.push_back(codes[i]);
        return v;
    };
    std::set<std::vector<std::uint16_t>> seen{codes};
    for (const auto& [A, B] : {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}}) {
        auto v = variant(A, B);
        bool alone = false;
        for (std::size_t i = 1; i + 1 < v.size(); ++i) alone = alone || (v[i] == kSlotO && v[i - 1] == dc.open() && v[i + 1] == dc.close());
        if (alone || v.size() < kMinCodes || !seen.insert(v).second) continue;
        out.push_back(std::move(v));
    }
    return out;
}

// ---------------------------------------------------------------- the book

// One word of a subject's life dates: its K5 id, the date (0 birth, 1 death) and its part (0 month, 1 day, 2 year).
struct LifeWord {
    std::uint32_t id = 0;
    int date = 0, part = 0;
};

// The name as discourse.hpp's write_name writes it (the store's title without a trailing "( ... )" and without a
// ", Region" after a word), so that the book finds a subject by the tokens either writer fills its slot with.
inline std::vector<std::string> life_write_name(const std::vector<std::string>& t) {
    std::vector<std::string> o = t;
    if (o.size() > 2 && o.back() == ")") {
        const auto it = std::find(o.begin() + 1, o.end(), std::string("("));
        if (it != o.end()) o.erase(it, o.end());
    }
    const auto letter = [](const std::string& w) {
        const auto c = w.empty() ? 0 : static_cast<unsigned char>(w[0]);
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0xC0;
    };
    const auto upper = [](const std::string& w) {
        const auto c = w.empty() ? 0 : static_cast<unsigned char>(w[0]);
        return (c >= 'A' && c <= 'Z') || c >= 0xC0;
    };
    for (std::size_t i = 1; i + 1 < o.size(); ++i)
        if (o[i] == "," && letter(o[i - 1]) && upper(o[i + 1])) {
            o.resize(i);
            break;
        }
    return o.empty() ? t : o;
}

class LifeBook {
 public:
    // dir: the store; drops: the planner's evidence drops; fs: the planner's store (loaded); m: K5 (the words' ids).
    bool build(const std::string& dir, const std::vector<std::pair<std::size_t, std::size_t>>& drops, const FactStore& fs, const wkn::K5& m,
               std::string& err) {
        // the full dates of the store's rows
        std::map<std::pair<std::uint32_t, int>, std::set<std::string>> full;
        for (const char* name : {"/facts_f2.tsv", "/seeds.tsv"}) {
            const bool f2 = name[1] == 'f';
            std::ifstream in(dir + name);
            if (!in) {
                if (f2) continue;
                return err = "life dates: cannot read " + dir + name, false;
            }
            std::string line;
            std::getline(in, line);
            while (std::getline(in, line)) {
                std::vector<std::string> f;
                for (std::size_t a = 0;;) {
                    const std::size_t b = line.find('\t', a);
                    f.push_back(line.substr(a, b == std::string::npos ? std::string::npos : b - a));
                    if (b == std::string::npos) break;
                    a = b + 1;
                }
                if (f2 ? f.size() < 9 : f.size() < 6) continue;
                if (f[1] != "birth_date" && f[1] != "death_date") continue;
                const std::size_t off = static_cast<std::size_t>(std::strtoull((f2 ? f[8] : f[5]).c_str(), nullptr, 10));
                bool drop = false;
                for (const auto& [lo, hi] : drops) drop = drop || (off >= lo && off < hi);
                if (drop) continue;
                const std::string& v = f[2];  // YYYY-MM-DD
                if (v.size() != 10 || v[4] != '-' || v[7] != '-') continue;
                const std::string subj = ContentPlanStore::strip(f[0]);
                const std::uint32_t e = fs.find(subj);
                if (e == ~0u || lower_s(fs.ent(e)) != lower_s(subj)) continue;
                full[{e, f[1][0] == 'b' ? 0 : 1}].insert(v);
            }
        }
        const std::uint32_t rb = fs.rel_id("birth_year"), rd = fs.rel_id("death_year");
        for (std::uint32_t e = 0; e < fs_ents(fs); ++e) {
            const auto& of = fs.of_subj(e);
            if (of.empty()) continue;
            std::vector<LifeWord> w;
            for (int d = 0; d < 2; ++d) {
                std::set<std::uint32_t> ys;
                for (const auto fi : of)
                    if (fs.fact(fi).rel == (d == 0 ? rb : rd)) ys.insert(fs.fact(fi).val);
                if (ys.size() != 1) continue;  // none, or two years: no date
                const auto& yt = fs.ent_tok(*ys.begin());
                if (yt.size() != 1 || !year_word(yt[0])) continue;
                const std::uint32_t yid = m.id(yt[0]);
                if (yid == wkn::kUnk) continue;
                const auto it = full.find({e, d});
                if (it != full.end() && it->second.size() == 1) {  // one full date, of that year
                    const std::string& v = *it->second.begin();
                    const std::string y = ContentPlanStore::year_of(v);
                    const int mo = std::atoi(v.substr(5, 2).c_str()), dy = std::atoi(v.substr(8, 2).c_str());
                    if (y == yt[0] && mo >= 1 && mo <= 12 && dy >= 1 && dy <= 31) {
                        const std::uint32_t mid = m.id(month_names()[mo - 1]), did = m.id(std::to_string(dy));
                        if (mid != wkn::kUnk && did != wkn::kUnk) {
                            w.push_back({mid, d, 0}), w.push_back({did, d, 1});
                            ++full_[d];
                        }
                    }
                }
                w.push_back({yid, d, 2});
                ++years_[d];
            }
            if (w.empty()) continue;
            const auto add = [&](const std::vector<std::string>& t) {
                const auto [it, fresh] = by_key_.try_emplace(key(t), static_cast<std::uint32_t>(words_.size()));
                if (!fresh && it->second != words_.size()) it->second = ~0u;  // two subjects write the same name: neither
            };
            add(fs.ent_tok(e));
            if (life_write_name(fs.ent_tok(e)) != fs.ent_tok(e)) add(life_write_name(fs.ent_tok(e)));
            words_.push_back(std::move(w));
        }
        return true;
    }
    // The life words of the subject a slot is filled with (null: none).
    const std::vector<LifeWord>* of(const std::vector<std::string>& S) const {
        const auto it = by_key_.find(key(S));
        return it == by_key_.end() || it->second == ~0u ? nullptr : &words_[it->second];
    }
    std::size_t subjects() const { return words_.size(); }
    std::size_t years(int d) const { return years_[d]; }
    std::size_t full_dates(int d) const { return full_[d]; }

 private:
    static std::uint32_t fs_ents(const FactStore& fs) {  // one past the highest subject ID
        std::uint32_t n = 0;
        for (std::uint32_t i = 0; i < fs.size(); ++i) n = std::max(n, fs.fact(i).subj + 1);
        return n;
    }
    static std::string key(const std::vector<std::string>& t) {
        std::string k;
        for (const auto& x : t) k += lower_s(x) + '\x1f';
        return k;
    }
    std::unordered_map<std::string, std::uint32_t> by_key_;
    std::vector<std::vector<LifeWord>> words_;
    std::size_t years_[2] = {0, 0}, full_[2] = {0, 0};
};

// Where the book is read from: set by apply_plan_flags under --k5-plan-honest life (one planner per process).
struct LifeSpec {
    std::string dir;
    std::vector<std::pair<std::size_t, std::size_t>> drops;
    const FactStore* fs = nullptr;
};
inline LifeSpec& life_spec() {
    static LifeSpec s;
    return s;
}
// The book, read once K5 is known (the first fill or skeleton choice); null without --k5-plan-honest life.
inline const LifeBook* life_book(const wkn::K5& m) {
    static std::once_flag once;
    static LifeBook book;
    static bool ok = false;
    if (!life_spec().fs) return nullptr;
    std::call_once(once, [&] {
        std::string err;
        ok = book.build(life_spec().dir, life_spec().drops, *life_spec().fs, m, err);
        if (!ok) std::fprintf(stderr, "k5-plan: life book: %s\n", err.c_str());
        else
            std::fprintf(stderr, "k5-plan: life book: %zu subjects; birth %zu years (%zu with month and day), death %zu years (%zu with month and day)\n",
                         book.subjects(), book.years(0), book.full_dates(0), book.years(1), book.full_dates(1));
    });
    return ok ? &book : nullptr;
}

// skel_fit: whether a skeleton's month and number positions (DateClasses::asks) can be filled from the subject's life
// words, O's own year aside (O fills its slot).
inline bool skel_fit(const DateClasses& dc, const std::vector<std::uint16_t>& codes, const std::vector<LifeWord>* lw, const std::vector<std::uint32_t>& o_ids) {
    const auto [mo, nu] = dc.asks(codes);
    int hm = 0, hn = 0;
    bool o_taken = false;
    if (lw)
        for (const auto& w : *lw) {
            if (w.part == 2 && !o_taken && std::find(o_ids.begin(), o_ids.end(), w.id) != o_ids.end()) {
                o_taken = true;
                continue;
            }
            hm += w.part == 0, hn += w.part != 0;
        }
    return mo <= hm && nu <= hn;
}

// ---------------------------------------------------------------- selftest

inline bool life_selftest(std::string& err) {
    if (month_no("March") != 3 || month_word("march") || !year_word("1926") || year_word("12") || year_word("19a6") || !has_digit("3rd"))
        return err = "life: words", false;
    if (life_write_name({"Alcaeus", "(", "poet", ")"}) != std::vector<std::string>{"Alcaeus"} ||
        life_write_name({"Balclutha", ",", "New", "Zealand"}) != std::vector<std::string>{"Balclutha"})
        return err = "life: write_name", false;
    // classes: 1 "(", 2 ")", 3 ",", 4 month, 5 number, 6 dash, 7 other
    const DateClasses dc({0, 0, 0, 0, kMonthClass, kNumberClass, 0, 0}, 1, 2, 3);
    using V = std::vector<std::uint16_t>;
    const std::uint16_t S = kSlotS, O = kSlotO;
    const V death = {S, 1, 4, 5, 3, 5, 6, 4, 5, 3, O, 2, 7, 7, 7};  // S (April 1, 2005 - April 1, O) is a poet
    const auto td = trim_dates(dc, death, true);
    if (td != std::vector<V>{V{S, 1, 5, 6, O, 2, 7, 7, 7}}) return err = "life: trim (date - O)", false;
    const V born = {S, 1, 7, 4, 5, 3, O, 2, 7, 7, 7};  // S (born April 1, O) is a poet
    if (trim_dates(dc, born, true) != std::vector<V>{V{S, 1, 7, O, 2, 7, 7, 7}}) return err = "life: trim (born)", false;
    const V birth = {S, 1, O, 6, 4, 5, 3, 5, 2, 7, 7, 7};  // S (O - April 1, 2005) is a poet
    if (trim_dates(dc, birth, true) != std::vector<V>{V{S, 1, O, 6, 5, 2, 7, 7, 7}}) return err = "life: trim (O - date)", false;
    const V two = {S, 1, O, 6, 4, 5, 3, 5, 2, 7, 1, 5, 6, 5, 2};  // S (O - April 1, 2005) is (2005 - 2005)
    if (trim_dates(dc, two, true) != std::vector<V>{V{S, 1, O, 6, 5, 2, 7, 1, 5, 6, 5, 2}})
        return err = "life: trim (other brackets)", false;
    if (!trim_dates(dc, death, false).empty() || !trim_dates(dc, V{S, 7, 7, 7, 2}, true).empty()) return err = "life: trim (none)", false;
    if (dc.asks(death) != std::pair<int, int>{2, 3}) return err = "life: asks", false;
    const std::vector<LifeWord> lw = {{10, 0, 0}, {11, 0, 1}, {12, 0, 2}, {13, 1, 2}};  // a full birth date, a death year
    if (!skel_fit(dc, born, &lw, {12}) || skel_fit(dc, death, &lw, {13}) || !skel_fit(dc, V{S, 1, O, 6, 5, 2, 7, 7}, &lw, {12}) ||
        skel_fit(dc, born, nullptr, {12}))
        return err = "life: fit", false;
    return true;
}

}  // namespace cypha::cyphalm::planner
