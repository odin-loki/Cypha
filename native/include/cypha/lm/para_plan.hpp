// The paragraph-plan term of K5's sentence gate (docs/reports/PARA_PLAN.md, J2 / CT3.3; GENERATION_PLAN §13.59).
// Odin Loch, 2026-11-12.
//
// Real paragraphs move in a pattern: a sentence's function (a definition, a birth, a career step, a date ...) and the
// way it refers to the paragraph's subject (by name, by pronoun, by "the society", or not at all) follow the sentence
// before. scripts/para_plan_mine.py counted those steps in 329.6 MB of clean Wikipedia (bench/para_plan/tables/
// state1.tsv). This file ports its labels to C++ and scores a gate draft by
//
//   term = log2 P(state(draft) | state(previous sentence))   add-0.5 smoothed over the 40 states
//          - orph   if the draft's ref is PRON and the previous sentence's ref is NONE   (an orphan pronoun)
//          - lead   if the draft is DEF or BIRTH, its ref is NONE and it holds a name    (a lead sentence about another name)
//
// A state is (func, ref). For the first draft the previous state is START:func(the prompt's last sentence); the
// orphan penalty needs a previous sentence of the continuation, so it never applies to the first draft.
// K5's gate z-scores the term over its K drafts and adds w times it (word_kn.hpp's gate_z, discourse.hpp's free
// sentences). Flag: --k5-para-plan TABLEDIR[,w=W,orph=W,lead=W]; off by default, and with it off nothing here runs.
//
// The labels are the script's (open_type, func_type, ref_type, subject_of, content, split_sents, TOK). Tokens are
// [^\W_]+(?:['’-][^\W_]+)* | [^\w\s] on code points; Python's \w is exact in ASCII and approximated outside it
// (letters and digits are word characters; the punctuation, symbol, mark and space blocks are not). Case outside
// ASCII: Latin-1, Latin Extended-A, Greek and Cyrillic. The capitalisation table (a word's capitalised and lower-case
// counts away from a sentence's start) is the script's caps.json: TABLEDIR/caps.json, else TABLEDIR/../caps.json.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cypha::cyphalm::pp {

// ---------------------------------------------------------------- code points

inline std::size_t u8_at(std::string_view s, std::size_t i, char32_t& cp) {
    const auto c = static_cast<unsigned char>(s[i]);
    const std::size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) {
        cp = 0xFFFD;
        return 1;
    }
    cp = n == 1 ? c : n == 2 ? (c & 0x1Fu) : n == 3 ? (c & 0x0Fu) : (c & 0x07u);
    for (std::size_t k = 1; k < n; ++k) {
        const auto d = static_cast<unsigned char>(s[i + k]);
        if ((d & 0xC0) != 0x80) {
            cp = 0xFFFD;
            return 1;
        }
        cp = (cp << 6) | (d & 0x3Fu);
    }
    return n;
}
inline void u8_put(std::string& o, char32_t c) {
    if (c < 0x80) {
        o += static_cast<char>(c);
    } else if (c < 0x800) {
        o += static_cast<char>(0xC0 | (c >> 6)), o += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        o += static_cast<char>(0xE0 | (c >> 12)), o += static_cast<char>(0x80 | ((c >> 6) & 0x3F)), o += static_cast<char>(0x80 | (c & 0x3F));
    } else {
        o += static_cast<char>(0xF0 | (c >> 18)), o += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        o += static_cast<char>(0x80 | ((c >> 6) & 0x3F)), o += static_cast<char>(0x80 | (c & 0x3F));
    }
}
// Python's str.isspace / re \s.
inline bool is_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
// Python's re \w (letters, digits, numerals and _).
inline bool is_word(char32_t c) {
    if (c < 0x80) return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    if (c <= 0xBF) return c == 0xAA || c == 0xB2 || c == 0xB3 || c == 0xB5 || c == 0xB9 || c == 0xBA || (c >= 0xBC && c <= 0xBE);
    if (c == 0xD7 || c == 0xF7) return false;
    if (c >= 0x300 && c <= 0x36F) return false;
    if (c >= 0x2000 && c <= 0x20CF) return false;
    if (c >= 0x2190 && c <= 0x245F) return false;
    if (c >= 0x2500 && c <= 0x2BFF) return false;
    if (c >= 0x2E00 && c <= 0x2E7F) return false;
    if (c >= 0x3000 && c <= 0x303F) return false;
    if ((c >= 0xFE30 && c <= 0xFE6F) || (c >= 0xFF01 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) ||
        (c >= 0xFF5B && c <= 0xFF65))
        return false;
    if (c == 0xFFFD || (c >= 0x1F000 && c <= 0x1FAFF)) return false;
    return !is_space(c);
}
inline bool is_alnum(char32_t c) { return c != '_' && is_word(c); }
inline bool is_digit(char32_t c) { return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0x2070 && c <= 0x2079); }
inline bool is_alpha(char32_t c) { return is_alnum(c) && !is_digit(c) && !(c >= 0xBC && c <= 0xBE); }
inline char32_t to_lower(char32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c < 0xC0) return c;
    if (c <= 0xDE && c != 0xD7) return c + 32;
    if (c >= 0x100 && c <= 0x137) return (c % 2 == 0) ? c + 1 : c;
    if (c >= 0x139 && c <= 0x148) return (c % 2 == 1) ? c + 1 : c;
    if (c >= 0x14A && c <= 0x177) return (c % 2 == 0) ? c + 1 : c;
    if (c == 0x178) return 0xFF;
    if (c >= 0x179 && c <= 0x17E) return (c % 2 == 1) ? c + 1 : c;
    if (c >= 0x391 && c <= 0x3A9 && c != 0x3A2) return c + 32;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    return c;
}
inline bool is_upper(char32_t c) {
    if (c >= 'A' && c <= 'Z') return true;
    if (c < 0xC0) return false;
    return to_lower(c) != c;
}
inline std::string lower(std::string_view s) {
    std::string o;
    o.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        char32_t c = 0;
        const std::size_t n = u8_at(s, i, c);
        if (c < 0x80) o += static_cast<char>(to_lower(c));
        else if (c == 0xFFFD && n == 1) o += s[i];
        else u8_put(o, to_lower(c));
        i += n;
    }
    return o;
}
inline std::size_t nchars(std::string_view s) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < s.size(); ++n) {
        char32_t c = 0;
        i += u8_at(s, i, c);
    }
    return n;
}
inline char32_t first_cp(std::string_view s) {
    char32_t c = 0;
    if (s.empty()) return 0;
    u8_at(s, 0, c);
    return c;
}
inline bool all_digits(std::string_view s) {  // Python's str.isdigit
    if (s.empty()) return false;
    for (std::size_t i = 0; i < s.size();) {
        char32_t c = 0;
        i += u8_at(s, i, c);
        if (!is_digit(c)) return false;
    }
    return true;
}
// " ".join(t.split())
inline std::string normalize(std::string_view t) {
    std::string o;
    o.reserve(t.size());
    bool sp = false;
    for (std::size_t i = 0; i < t.size();) {
        char32_t c = 0;
        const std::size_t n = u8_at(t, i, c);
        if (is_space(c)) {
            sp = true;
        } else {
            if (sp && !o.empty()) o += ' ';
            sp = false;
            o.append(t.substr(i, n));
        }
        i += n;
    }
    return o;
}

// ---------------------------------------------------------------- word lists (the script's)

using WordSet = std::unordered_set<std::string>;
inline WordSet word_set(const char* s) {
    WordSet w;
    std::istringstream in(s);
    for (std::string x; in >> x;) w.insert(x);
    return w;
}
inline const WordSet& kStop() {
    static const WordSet w = word_set(
        "that this with from were have which their there they been when also into than then them these those other about after "
        "before would could should while where being because some such more most many much over under only very what will your "
        "first second three during between within without through against among upon each both either well just like even");
    return w;
}
inline const WordSet& kAbbr() {
    static const WordSet w = word_set("mr mrs ms dr st jr sr vs inc co ltd no gen col lt sgt prof rev mt ft ca c e.g i.e etc u.s u.k fig vol pp ed");
    return w;
}
inline const WordSet& kMonths() {
    static const WordSet w = word_set("January February March April May June July August September October November December");
    return w;
}
inline const WordSet& kConn() {
    static const WordSet w = word_set(
        "However But And Also Thus Therefore Moreover Furthermore Meanwhile Later Then Although While As Because Despite "
        "Instead Nevertheless Finally Additionally Similarly Consequently Hence Still Yet Eventually Initially Originally Subsequently "
        "Afterwards Indeed Nonetheless Likewise Accordingly Otherwise So Or Unlike Like Besides Overall");
    return w;
}
inline const WordSet& kPronOpen() {
    static const WordSet w = word_set("He She It They This These That Those His Her Its Their We I There Such");
    return w;
}
inline const WordSet& kPrep() {
    static const WordSet w = word_set("In On During After Before By Since From Until Between At Around Following Circa");
    return w;
}
inline const WordSet& kInConn() {
    static const WordSet w = word_set("addition contrast particular fact general practice turn short response comparison");
    return w;
}
inline const WordSet& kPron3() {
    static const WordSet w = word_set("he him his himself she her hers herself it its itself they them their theirs themselves");
    return w;
}
inline const WordSet& kDetAna() {
    static const WordSet w = word_set("The This These Such That Those");
    return w;
}
inline const WordSet& kGlue() {
    static const WordSet w = word_set("of de del della der van von la le du da di y and the");
    return w;
}

enum Func { kDEF, kDEATH, kBIRTH, kEDUC, kAWARD, kWORK, kNAME, kPLACE, kEVENT, kOTHER };
enum Ref { kRefName, kRefPron, kRefDefnp, kRefNone };
inline constexpr int kStates = 40;  // func * 4 + ref

// ---------------------------------------------------------------- tokens and sentences

// TOK.findall
inline std::vector<std::string> toks(std::string_view s) {
    std::vector<std::string> v;
    const std::size_t n = s.size();
    const auto run = [&](std::size_t k) {
        while (k < n) {
            char32_t d = 0;
            const std::size_t l = u8_at(s, k, d);
            if (!is_alnum(d)) break;
            k += l;
        }
        return k;
    };
    for (std::size_t i = 0; i < n;) {
        char32_t c = 0;
        const std::size_t len = u8_at(s, i, c);
        if (is_alnum(c)) {
            std::size_t j = run(i + len);
            while (j < n) {
                char32_t d = 0;
                const std::size_t l = u8_at(s, j, d);
                if ((d != '\'' && d != 0x2019 && d != '-') || j + l >= n) break;
                char32_t e = 0;
                const std::size_t l2 = u8_at(s, j + l, e);
                if (!is_alnum(e)) break;
                j = run(j + l + l2);
            }
            v.emplace_back(s.substr(i, j - i));
            i = j;
        } else if (!is_word(c) && !is_space(c)) {
            v.emplace_back(s.substr(i, len));
            i += len;
        } else {
            i += len;
        }
    }
    return v;
}

// content(s): [A-Za-z]+ runs of s.lower(), 4+ letters, not a function word.
inline std::vector<std::string> content(std::string_view s) {
    const std::string l = lower(s);
    std::vector<std::string> v;
    const auto az = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    for (std::size_t i = 0; i < l.size();) {
        if (!az(l[i])) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < l.size() && az(l[j])) ++j;
        std::string w = l.substr(i, j - i);
        if (w.size() >= 4 && !kStop().count(w)) v.push_back(std::move(w));
        i = j;
    }
    return v;
}
inline WordSet content_set(std::string_view s) {
    WordSet w;
    for (auto& x : content(s)) w.insert(std::move(x));
    return w;
}

inline bool closer(char c) { return c == '"' || c == '\'' || c == ')' || c == ']'; }

// split_sents (on " ".join(t.split())): BOUND = [.!?]["')\]]*\s+(?=["'(\[]?[A-Z0-9]).
inline std::vector<std::string> split_sents(std::string_view raw) {
    const std::string t = normalize(raw);
    const std::size_t n = t.size();
    std::vector<std::string> out;
    std::size_t a = 0;
    std::string pend;
    const auto strip = [](std::string_view x) {
        while (!x.empty() && x.front() == ' ') x.remove_prefix(1);
        while (!x.empty() && x.back() == ' ') x.remove_suffix(1);
        return std::string(x);
    };
    for (std::size_t i = 0; i < n;) {
        const char ch = t[i];
        if (ch != '.' && ch != '!' && ch != '?') {
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        while (j < n && closer(t[j])) ++j;
        std::size_t k = j;
        while (k < n && t[k] == ' ') ++k;
        std::size_t m = k;
        if (m < n && (t[m] == '"' || t[m] == '\'' || t[m] == '(' || t[m] == '[')) ++m;
        if (k == j || !(m < n && ((t[m] >= 'A' && t[m] <= 'Z') || (t[m] >= '0' && t[m] <= '9')))) {
            ++i;
            continue;
        }
        // a boundary match [i, k): w = t[a:i + 1].rstrip(".!?\"')]").split(); last = w[-1].lower()
        std::string_view head(t.data() + a, i + 1 - a);
        while (!head.empty() && (head.back() == '.' || head.back() == '!' || head.back() == '?' || closer(head.back()))) head.remove_suffix(1);
        while (!head.empty() && head.back() == ' ') head.remove_suffix(1);
        const std::size_t sp = head.rfind(' ');
        const std::string last = lower(sp == std::string_view::npos ? head : head.substr(sp + 1));
        bool skip = kAbbr().count(last) > 0;
        if (!skip && nchars(last) == 1 && is_alpha(first_cp(last))) skip = true;
        if (!skip && last.size() >= 3 && last.size() % 2 == 1) {  // ([a-z]\.)+[a-z]
            bool ok = true;
            for (std::size_t q = 0; q < last.size() && ok; ++q) ok = q % 2 == 0 ? (last[q] >= 'a' && last[q] <= 'z') : last[q] == '.';
            skip = ok;
        }
        if (skip) {
            i = k;
            continue;
        }
        const std::string body = strip(std::string_view(t.data() + a, k - a));
        std::string piece = pend.empty() ? body : pend + " " + body;
        if (piece.size() < 24) {
            pend = std::move(piece);
        } else {
            out.push_back(std::move(piece));
            pend.clear();
        }
        a = k;
        i = k;
    }
    const std::string rest = strip(std::string_view(t.data() + a, n - a));
    if (!rest.empty() || !pend.empty()) out.push_back(pend.empty() ? rest : strip(pend + " " + rest));
    std::vector<std::string> o;
    for (auto& x : out)
        if (!x.empty()) o.push_back(std::move(x));
    return o;
}

inline bool year_of(std::string_view t) {
    if (t.size() != 4) return false;
    for (const char c : t)
        if (c < '0' || c > '9') return false;
    const int y = (t[0] - '0') * 1000 + (t[1] - '0') * 100 + (t[2] - '0') * 10 + (t[3] - '0');
    return y >= 1000 && y <= 2099;
}

// bare(t): re.sub(r"['’]s?$", "", t).
inline std::string bare(std::string_view t) {
    static constexpr std::string_view q = "\xE2\x80\x99";
    if (t.size() >= 2 && t.back() == 's' && t[t.size() - 2] == '\'') return std::string(t.substr(0, t.size() - 2));
    if (t.size() >= 4 && t.back() == 's' && t.substr(t.size() - 4, 3) == q) return std::string(t.substr(0, t.size() - 4));
    if (!t.empty() && t.back() == '\'') return std::string(t.substr(0, t.size() - 1));
    if (t.size() >= 3 && t.substr(t.size() - 3) == q) return std::string(t.substr(0, t.size() - 3));
    return std::string(t);
}

// Maximal runs of \w and what lies between them, for the script's \b(...)\b regexes.
struct Runs {
    std::vector<std::string> run;
    std::vector<std::string> sep;  // sep[i]: between run[i] and run[i + 1]
};
inline Runs runs_of(std::string_view s) {
    Runs r;
    std::size_t i = 0, last_end = std::string::npos;
    const std::size_t n = s.size();
    while (i < n) {
        char32_t c = 0;
        const std::size_t l = u8_at(s, i, c);
        if (!is_word(c)) {
            i += l;
            continue;
        }
        std::size_t j = i;
        while (j < n) {
            char32_t d = 0;
            const std::size_t l2 = u8_at(s, j, d);
            if (!is_word(d)) break;
            j += l2;
        }
        if (last_end != std::string::npos) r.sep.emplace_back(s.substr(last_end, i - last_end));
        r.run.emplace_back(s.substr(i, j - i));
        last_end = j;
        i = j;
    }
    return r;
}
inline bool all_space(std::string_view s) {
    if (s.empty()) return false;
    for (std::size_t i = 0; i < s.size();) {
        char32_t c = 0;
        i += u8_at(s, i, c);
        if (!is_space(c)) return false;
    }
    return true;
}

// The func cues (IGNORECASE), each a list of alternatives: one word, or two words with one space between.
struct Cue {
    Func f;
    std::vector<std::pair<std::string, std::string>> alt;
};
inline const std::vector<Cue>& kCues() {
    static const std::vector<Cue> c = [] {
        const auto mk = [](Func f, const std::string& s) {
            Cue q{f, {}};
            for (std::size_t a = 0; a <= s.size();) {
                std::size_t b = s.find('|', a);
                if (b == std::string::npos) b = s.size();
                const std::string x = s.substr(a, b - a);
                const std::size_t sp = x.find(' ');
                q.alt.emplace_back(x.substr(0, sp), sp == std::string::npos ? std::string() : x.substr(sp + 1));
                a = b + 1;
            }
            return q;
        };
        return std::vector<Cue>{
            mk(kDEATH, "died|dies|death|killed|buried|murdered|executed|assassinated|funeral|passed away"),
            mk(kBIRTH, "born|birth|birthplace|childhood|grew up|early life|his parents|her parents"),
            mk(kEDUC, "educated|studied|graduated|graduate|degree|attended|student|doctorate|alumnus"),
            mk(kAWARD, "award|awarded|awards|prize|prizes|medal|medals|honour|honor|honours|honors|won|wins|nominated|nomination"),
            mk(kWORK, "worked|works|joined|became|appointed|elected|served|career|founded|wrote|written|published|released|"
                      "recorded|directed|played|signed|produced|composed|built|designed|developed|invented|starred|commanded|ruled|reigned"),
            mk(kNAME, "named|renamed|called|known as|derives|derived|meaning|etymology|the name|its name|his name|her name|name of"),
            mk(kPLACE, "located|situated|lies|population|inhabitants|area of|km|kilometres|kilometers|miles|border|borders|"
                       "northern|southern|eastern|western|river|coast|capital|census"),
        };
    }();
    return c;
}

// ---------------------------------------------------------------- the tables, the labels and the term

class ParaPlan {
 public:
    // Loads dir/state1.tsv and the capitalisation table. False and err on failure.
    bool load(const std::string& dir, std::string& err) {
        dir_ = dir;
        std::ifstream in(dir + "/state1.tsv");
        if (!in) return err = "--k5-para-plan: cannot read " + dir + "/state1.tsv", false;
        std::string line;
        std::size_t n = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (!add_line(line)) return err = "--k5-para-plan: bad line in state1.tsv: " + line, false;
            ++n;
        }
        if (n == 0) return err = "--k5-para-plan: state1.tsv holds no counts", false;
        std::string cj = dir + "/caps.json";
        if (!std::ifstream(cj)) cj = dir + "/../caps.json";
        std::ifstream cf(cj, std::ios::binary);
        if (!cf) return err = "--k5-para-plan: cannot read " + dir + "/caps.json or " + dir + "/../caps.json", false;
        std::stringstream ss;
        ss << cf.rdbuf();
        if (!parse_caps(ss.str())) return err = "--k5-para-plan: cannot parse " + cj, false;
        std::fprintf(stderr, "k5: para plan %s: %zu state transitions (%.0f sentences), %zu capitalised words (%s)\n", dir.c_str(), n, total_,
                     caps_.size(), cj.c_str());
        return true;
    }
    // One "prev\tnext\tcount" line ("f:r" or "START:f", then "f:r").
    bool add_line(const std::string& line) {
        const std::size_t t1 = line.find('\t'), t2 = t1 == std::string::npos ? t1 : line.find('\t', t1 + 1);
        if (t2 == std::string::npos) return false;
        const int p = state_of(line.substr(0, t1), true), q = state_of(line.substr(t1 + 1, t2 - t1 - 1), false);
        char* e = nullptr;
        const double c = std::strtod(line.c_str() + t2 + 1, &e);
        if (p < 0 || q < 0 || q >= kStates || *e || t2 + 1 >= line.size() || c < 0) return false;
        cnt_[static_cast<std::size_t>(p) * kStates + static_cast<std::size_t>(q)] += c;
        row_[static_cast<std::size_t>(p)] += c;
        total_ += c;
        return true;
    }
    // caps.json: {"word": [capitalised, lower], ...} (json.dump's escapes).
    bool parse_caps(const std::string& s) {
        std::size_t i = s.find('{');
        if (i == std::string::npos) return false;
        ++i;
        const auto ws = [&] {
            while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
        };
        const auto hex4 = [&](std::size_t at) { return static_cast<char32_t>(std::strtoul(s.substr(at, 4).c_str(), nullptr, 16)); };
        const auto str = [&](std::string& o) {
            if (i >= s.size() || s[i] != '"') return false;
            ++i;
            o.clear();
            while (i < s.size() && s[i] != '"') {
                if (s[i] != '\\') {
                    o += s[i++];
                    continue;
                }
                if (++i >= s.size()) return false;
                const char c = s[i++];
                if (c == 'u') {
                    if (i + 4 > s.size()) return false;
                    char32_t u = hex4(i);
                    i += 4;
                    if (u >= 0xD800 && u <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        const char32_t lo = hex4(i + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00), i += 6;
                    }
                    u8_put(o, u);
                } else {
                    o += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c == 'b' ? '\b' : c == 'f' ? '\f' : c;
                }
            }
            if (i >= s.size()) return false;
            ++i;
            return true;
        };
        const auto num = [&](std::uint32_t& x) {
            ws();
            char* e = nullptr;
            x = static_cast<std::uint32_t>(std::strtoul(s.c_str() + i, &e, 10));
            if (e == s.c_str() + i) return false;
            i = static_cast<std::size_t>(e - s.c_str());
            return true;
        };
        std::string k;
        for (;;) {
            ws();
            if (i < s.size() && s[i] == '}') return true;
            if (!str(k)) return false;
            ws();
            if (i >= s.size() || s[i++] != ':') return false;
            ws();
            if (i >= s.size() || s[i++] != '[') return false;
            std::uint32_t a = 0, b = 0;
            if (!num(a)) return false;
            ws();
            if (i >= s.size() || s[i++] != ',') return false;
            if (!num(b)) return false;
            ws();
            if (i >= s.size() || s[i++] != ']') return false;
            caps_[k] = {a, b};
            ws();
            if (i < s.size() && s[i] == ',') ++i;
        }
    }
    void set_caps(const std::string& w, std::uint32_t cap, std::uint32_t low) { caps_[w] = {cap, low}; }
    const std::string& dir() const { return dir_; }

    // Caps.name_word (structure/experts.hpp cap_name).
    bool name_word(std::string_view t, bool first, bool next_cap = false) const {
        if (!is_upper(first_cp(t)) || t == "I") return false;
        const std::string l = lower(t);
        if (kPron3().count(l)) return false;
        if (!first) return true;
        const auto it = caps_.find(l);
        if (it == caps_.end()) return next_cap;
        return it->second.first > it->second.second;
    }
    int open_type(const std::vector<std::string>& tt) const {
        if (tt.empty()) return 5;
        const std::string& a = tt[0];
        if (year_of(a) || kMonths().count(a)) return 3;
        if (kPrep().count(a)) {
            for (std::size_t i = 1; i < 4 && i < tt.size(); ++i)
                if (year_of(tt[i]) || kMonths().count(tt[i])) return 3;
            if (a == "In" && tt.size() > 1 && kInConn().count(tt[1])) return 0;
        }
        if (kConn().count(a)) return 0;
        if (kPronOpen().count(a)) return 1;
        if (a == "The" || a == "A" || a == "An") return 2;
        if (name_word(a, true, tt.size() > 1 && name_word(tt[1], false))) return 4;
        return 5;
    }
    int func_type(std::string_view s, const std::vector<std::string>& tt) const {
        {  // DEF_RE on " ".join(tt[:14]): \b(is|was|are|were)\s+(a|an|the|one of)\b, case-sensitive
            std::string head;
            for (std::size_t i = 0; i < tt.size() && i < 14; ++i) head += (i ? " " : "") + tt[i];
            const Runs r = runs_of(head);
            bool def = false;
            for (std::size_t i = 0; i + 1 < r.run.size() && !def; ++i) {
                const std::string& a = r.run[i];
                if ((a != "is" && a != "was" && a != "are" && a != "were") || !all_space(r.sep[i])) continue;
                const std::string& b = r.run[i + 1];
                def = b == "a" || b == "an" || b == "the" || (b == "one" && i + 2 < r.run.size() && r.sep[i + 1] == " " && r.run[i + 2] == "of");
            }
            if (def) {
                const int o = open_type(tt);
                if (o == 2 || o == 4 || o == 5) return kDEF;
            }
        }
        const Runs r = runs_of(s);
        std::vector<std::string> low(r.run.size());
        for (std::size_t i = 0; i < r.run.size(); ++i) low[i] = lower(r.run[i]);
        for (const Cue& c : kCues())
            for (std::size_t i = 0; i < low.size(); ++i)
                for (const auto& [w1, w2] : c.alt) {
                    if (low[i] != w1) continue;
                    if (w2.empty() || (i + 1 < low.size() && r.sep[i] == " " && low[i + 1] == w2)) return c.f;
                }
        for (const auto& t : tt)
            if (year_of(t) || kMonths().count(t)) return kEVENT;
        return kOTHER;
    }
    std::vector<std::vector<std::string>> name_runs(const std::vector<std::string>& tt) const {
        std::vector<std::vector<std::string>> runs;
        std::vector<std::string> cur;
        for (std::size_t i = 0; i < tt.size(); ++i) {
            const std::string& t = tt[i];
            const std::string nxt = i + 1 < tt.size() ? tt[i + 1] : std::string();
            const bool opener = i == 0 && (kPronOpen().count(t) || kConn().count(t) || kPrep().count(t) || t == "The" || t == "A" || t == "An");
            if (name_word(t, i == 0, nxt.empty() ? false : name_word(nxt, false)) && !opener) {
                cur.push_back(t);
            } else if (!cur.empty() && kGlue().count(lower(t)) && is_upper(first_cp(nxt))) {
                cur.push_back(t);
            } else {
                if (!cur.empty()) runs.push_back(cur);
                cur.clear();
            }
        }
        if (!cur.empty()) runs.push_back(cur);
        std::vector<std::vector<std::string>> out;
        for (auto r : runs) {
            while (!r.empty() && kGlue().count(lower(r.back()))) r.pop_back();
            if (!r.empty()) out.push_back(std::move(r));
        }
        return out;
    }
    static std::string head_of(const std::vector<std::string>& run) {
        for (auto it = run.rbegin(); it != run.rend(); ++it)
            if (nchars(*it) >= 3 && !kGlue().count(lower(*it)) && !all_digits(*it)) return *it;
        return run.back();
    }
    struct Subject {
        int kind = 2;  // 0 NAME, 1 TERM, 2 NONE
        std::string h;
        WordSet ws;
    };
    Subject subject_of(const std::vector<std::string>& ps) const {
        std::unordered_map<std::string, std::pair<int, int>> cnt;  // head -> (count, order of first sight)
        std::unordered_map<std::string, WordSet> words;
        for (const auto& s : ps)
            for (const auto& r : name_runs(toks(s))) {
                const std::string h = bare(head_of(r));
                auto it = cnt.find(h);
                if (it == cnt.end()) it = cnt.emplace(h, std::make_pair(0, static_cast<int>(cnt.size()))).first;
                ++it->second.first;
                WordSet& w = words[h];
                for (const auto& t : r) {
                    const std::string b = bare(t);
                    if (nchars(b) >= 4 && !kGlue().count(lower(t))) w.insert(b);
                }
            }
        Subject sj;
        if (!cnt.empty()) {
            const std::pair<int, int>* best = nullptr;
            for (const auto& [h, v] : cnt)
                if (!best || v.first > best->first || (v.first == best->first && v.second < best->second)) best = &v, sj.h = h;
            sj.kind = 0;
            sj.ws = words[sj.h];
            sj.ws.insert(sj.h);
            const auto t0 = ps.empty() ? std::vector<std::string>{} : toks(ps[0]);
            bool opens = false;
            for (std::size_t i = 0; i < t0.size() && i < 3; ++i) opens = opens || sj.ws.count(bare(t0[i]));
            if (opens)
                for (std::size_t i = 0; i < t0.size() && i < 8; ++i) {
                    const std::string& t = t0[i];
                    if (t == "(" || t == "is" || t == "was" || t == "are" || t == "were" || t == ",") break;
                    const std::string b = bare(t);
                    if (nchars(b) >= 4 && name_word(t, false)) sj.ws.insert(b);
                }
            return sj;
        }
        std::unordered_map<std::string, std::pair<int, int>> cw;
        for (const auto& s : ps)
            for (const auto& w : content(s)) {
                auto it = cw.find(w);
                if (it == cw.end()) it = cw.emplace(w, std::make_pair(0, static_cast<int>(cw.size()))).first;
                ++it->second.first;
            }
        if (cw.empty()) return sj;
        const std::pair<int, int>* best = nullptr;
        for (const auto& [w, v] : cw)
            if (!best || v.first > best->first || (v.first == best->first && v.second < best->second)) best = &v, sj.h = w;
        sj.kind = 1;
        sj.ws = {sj.h};
        return sj;
    }
    static int ref_type(const Subject& sj, const std::vector<std::string>& tt0, const WordSet& prev_content) {
        std::vector<std::string> tt;
        tt.reserve(tt0.size());
        for (const auto& t : tt0) tt.push_back(bare(t));
        if (sj.kind == 0) {
            for (const auto& t : tt)
                if (sj.ws.count(t)) return kRefName;
            WordSet low;
            for (const auto& w : sj.ws) low.insert(lower(w));
            for (const auto& t : tt)
                if (low.count(t)) return kRefDefnp;
        } else if (sj.kind == 1) {
            for (const auto& t : tt) {
                const std::string l = lower(t);
                if (l == sj.h || l == sj.h + "s" || (!l.empty() && l.back() == 's' && l.substr(0, l.size() - 1) == sj.h)) return kRefName;
            }
        }
        if (sj.kind != 2) {
            for (std::size_t i = 0; i < tt.size() && i < 6; ++i)
                if (kPron3().count(lower(tt[i]))) return kRefPron;
            for (std::size_t i = 0; i < tt.size() && i < 4; ++i) {
                const std::string l = lower(tt[i]);
                if (kDetAna().count(tt[i]) || (i > 0 && (l == "the" || l == "this" || l == "these")))
                    for (std::size_t j = i + 1; j < tt.size() && j < i + 3; ++j)
                        if (prev_content.count(lower(tt[j]))) return kRefDefnp;
            }
        }
        return kRefNone;
    }

    // What one continuation needs from its prompt, made once.
    struct Track {
        Subject subj;
        WordSet pw;     // the prompt's content words
        WordSet plast;  // its last sentence's
        int p_func = kOTHER;
    };
    Track begin(std::string_view prompt) const {
        Track tr;
        const auto ps = split_sents(prompt);
        tr.subj = subject_of(ps);
        for (const auto& s : ps)
            for (auto& w : content(s)) tr.pw.insert(std::move(w));
        if (!ps.empty()) {
            tr.plast = content_set(ps.back());
            tr.p_func = func_type(ps.back(), toks(ps.back()));
        }
        return tr;
    }
    // The last sentence of the text written so far (out, split as the script splits): its state, ref and content
    // words. Nothing written yet: START:func(the prompt's last sentence), no ref, that sentence's content words.
    struct Prev {
        int state = 0;  // 0..39, or 40 + func for START
        int ref = -1;
        WordSet content;
    };
    Prev prev(const Track& tr, std::string_view out) const {
        Prev p;
        const auto ss = split_sents(out);
        if (ss.empty()) {
            p.state = kStates + tr.p_func;
            p.content = tr.plast;
            return p;
        }
        const std::string& last = ss.back();
        WordSet anchor = tr.pw;
        if (ss.size() >= 2)
            for (auto& w : content(ss[ss.size() - 2])) anchor.insert(std::move(w));
        else
            anchor.insert(tr.plast.begin(), tr.plast.end());
        const auto tt = toks(last);
        p.ref = ref_type(tr.subj, tt, anchor);
        p.state = func_type(last, tt) * 4 + p.ref;
        p.content = content_set(last);
        return p;
    }
    struct Label {
        int func = kOTHER, ref = kRefNone;
        bool orphan = false, lead = false;
        double lp = 0;  // log2 P(state | previous state)
    };
    // A draft, read as one sentence after pv.
    Label label(const Track& tr, const Prev& pv, std::string_view draft) const {
        Label L;
        const std::string d = normalize(draft);
        const auto tt = toks(d);
        L.func = func_type(d, tt);
        WordSet anchor = tr.pw;
        anchor.insert(pv.content.begin(), pv.content.end());
        L.ref = ref_type(tr.subj, tt, anchor);
        L.orphan = pv.ref == kRefNone && L.ref == kRefPron;
        L.lead = (L.func == kDEF || L.func == kBIRTH) && L.ref == kRefNone && !name_runs(tt).empty();
        L.lp = logp(pv.state, L.func * 4 + L.ref);
        return L;
    }
    // The gate's term for a draft (before z-scoring).
    double term(const Track& tr, const Prev& pv, std::string_view draft, double orph, double lead) const {
        const Label L = label(tr, pv, draft);
        return L.lp - (L.orphan ? orph : 0.0) - (L.lead ? lead : 0.0);
    }
    double logp(int prev, int next) const {
        const auto p = static_cast<std::size_t>(prev);
        return std::log2((cnt_[p * kStates + static_cast<std::size_t>(next)] + 0.5) / (row_[p] + 0.5 * kStates));
    }

 private:
    static int state_of(const std::string& s, bool allow_start) {
        const std::size_t q = s.find(':');
        if (q == std::string::npos) return -1;
        const std::string a = s.substr(0, q), b = s.substr(q + 1);
        char* e = nullptr;
        const long y = std::strtol(b.c_str(), &e, 10);
        if (b.empty() || *e) return -1;
        if (a == "START") return allow_start && y >= 0 && y < 10 ? kStates + static_cast<int>(y) : -1;
        const long x = std::strtol(a.c_str(), &e, 10);
        if (a.empty() || *e || x < 0 || x >= 10 || y < 0 || y >= 4) return -1;
        return static_cast<int>(x * 4 + y);
    }
    std::string dir_;
    std::vector<double> cnt_ = std::vector<double>(static_cast<std::size_t>(kStates + 10) * kStates, 0.0);
    std::vector<double> row_ = std::vector<double>(static_cast<std::size_t>(kStates + 10), 0.0);
    double total_ = 0;
    std::unordered_map<std::string, std::pair<std::uint32_t, std::uint32_t>> caps_;
};

// The flag's settings (word_kn.hpp's Options holds one; plan null: off).
struct Spec {
    std::shared_ptr<const ParaPlan> plan;
    double w = 1, orph = 0, lead = 0;
};
// --k5-para-plan TABLEDIR[,w=W,orph=W,lead=W] ("off": none). Loads the tables.
inline bool parse_flag(const std::string& val, Spec& sp, std::string& err) {
    if (val == "off") {
        sp = Spec{};
        return true;
    }
    std::vector<std::string> f;
    for (std::size_t a = 0; a <= val.size();) {
        std::size_t b = val.find(',', a);
        if (b == std::string::npos) b = val.size();
        f.push_back(val.substr(a, b - a));
        a = b + 1;
    }
    if (f.empty() || f[0].empty()) return err = "--k5-para-plan: TABLEDIR[,w=W,orph=W,lead=W]", false;
    Spec n;
    for (std::size_t i = 1; i < f.size(); ++i) {
        const std::size_t q = f[i].find('=');
        if (q == std::string::npos || q + 1 >= f[i].size()) return err = "--k5-para-plan: bad field " + f[i], false;
        const std::string k = f[i].substr(0, q);
        char* e = nullptr;
        const double v = std::strtod(f[i].c_str() + q + 1, &e);
        if (*e || !(v >= -100 && v <= 100)) return err = "--k5-para-plan: bad field " + f[i], false;
        if (k == "w") n.w = v;
        else if (k == "orph") n.orph = v;
        else if (k == "lead") n.lead = v;
        else return err = "--k5-para-plan: unknown field " + k, false;
    }
    auto p = std::make_shared<ParaPlan>();
    if (!p->load(f[0], err)) return false;
    n.plan = std::move(p);
    sp = std::move(n);
    return true;
}

// The labels on fixed sentences equal para_plan_mine.py's (expected values printed by the script with the same small
// capitalisation table); table lines parse; the term reads them; the flag parses. No data files.
inline bool self_test(std::string& err) {
    ParaPlan p;
    p.set_caps("barbara", 50, 1), p.set_caps("anderson", 40, 2), p.set_caps("new", 30, 900);
    p.set_caps("zealand", 70, 0), p.set_caps("society", 3, 80), p.set_caps("smith", 20, 1);
    const std::string prompt = "Barbara Anderson (born 1926) is a New Zealand playwright. She wrote for the Wellington Society.";
    if (split_sents(prompt) != std::vector<std::string>{"Barbara Anderson (born 1926) is a New Zealand playwright.", "She wrote for the Wellington Society."} ||
        split_sents("Dr. Smith went home. He slept. Then the U.S. Army came at 5 p.m. It was late, said Mr. Jones in 1990.") !=
            std::vector<std::string>{"Dr. Smith went home. He slept.", "Then the U.S. Army came at 5 p.m. It was late, said Mr. Jones in 1990."})
        return err = "para_plan: split_sents", false;
    if (toks("Smith's co-founded well\xE2\x80\x93known \xE2\x80\x9CHi\xE2\x80\x9D 10km x_y") !=
        std::vector<std::string>{"Smith's", "co-founded", "well", "\xE2\x80\x93", "known", "\xE2\x80\x9C", "Hi", "\xE2\x80\x9D", "10km", "x", "y"})
        return err = "para_plan: toks", false;
    const ParaPlan::Track tr = p.begin(prompt);
    if (tr.subj.kind != 0 || tr.subj.h != "Anderson" || tr.subj.ws != WordSet{"Barbara", "Anderson"} || tr.p_func != kWORK)
        return err = "para_plan: subject", false;
    struct Row {
        const char* s;
        int open, func, ref;
    };
    const Row rows[] = {  // window(prompt sentences, these) in para_plan_mine.py
        {"She was born in Auckland and grew up there.", 1, 2, 1},
        {"In 1950 the society moved to a new hall.", 3, 8, 2},
        {"John Smith was a British actor who starred in films.", 4, 0, 3},
        {"He died in 1980.", 1, 1, 1},
        {"However, Anderson's plays won several awards.", 0, 4, 0},
        {"The river flows north through the town.", 2, 7, 3},
        {"These plays are known as the Wellington cycle.", 1, 6, 3},
        {"It is one of the oldest theatres in the country.", 1, 9, 1},
        {"Many critics praised her early work for its humour.", 5, 9, 1},
        {"Smith studied law at the university of Otago.", 4, 3, 3},
    };
    WordSet prev_c = tr.plast;
    for (const auto& r : rows) {
        const auto tt = toks(r.s);
        WordSet anchor = tr.pw;
        anchor.insert(prev_c.begin(), prev_c.end());
        const int o = p.open_type(tt), f = p.func_type(r.s, tt), q = ParaPlan::ref_type(tr.subj, tt, anchor);
        if (o != r.open || f != r.func || q != r.ref) {
            char b[256];
            std::snprintf(b, sizeof b, "para_plan: labels of \"%s\": %d %d %d, want %d %d %d", r.s, o, f, q, r.open, r.func, r.ref);
            return err = b, false;
        }
        prev_c = content_set(r.s);
    }
    if (p.name_runs(toks("John Smith was a British actor.")) != std::vector<std::vector<std::string>>{{"John", "Smith"}, {"British"}})
        return err = "para_plan: name runs", false;
    // the table and the term
    if (!p.add_line("START:5\t2:1\t30") || !p.add_line("START:5\t9:0\t10") || !p.add_line("7:3\t2:1\t5") || p.add_line("x\t1:1\t3") ||
        p.add_line("START:5\t2:7\t3") || p.add_line("1:1\t2:1\t"))
        return err = "para_plan: table lines", false;
    const ParaPlan::Prev p0 = p.prev(tr, "");
    if (p0.state != kStates + kWORK || p0.ref != -1 || std::fabs(p.logp(p0.state, 2 * 4 + 1) - std::log2(30.5 / 60.0)) > 1e-12)
        return err = "para_plan: the first previous state", false;
    const ParaPlan::Label L0 = p.label(tr, p0, " He was born in Paris.");
    if (L0.orphan || L0.func != kBIRTH || L0.ref != kRefPron) return err = "para_plan: no orphan on the first draft", false;
    const ParaPlan::Prev p1 = p.prev(tr, " She was born in Auckland. The river flows north through the town.");
    const ParaPlan::Label L = p.label(tr, p1, " He was born in Paris.");
    if (p1.state != kPLACE * 4 + kRefNone || L.func != kBIRTH || L.ref != kRefPron || !L.orphan || L.lead ||
        std::fabs(L.lp - std::log2(5.5 / 25.0)) > 1e-12 || std::fabs(p.term(tr, p1, " He was born in Paris.", 2, 3) - (L.lp - 2)) > 1e-12)
        return err = "para_plan: orphan pronoun", false;
    const ParaPlan::Label L2 = p.label(tr, p1, " John Smith was a British actor.");
    if (!L2.lead || L2.orphan || std::fabs(p.term(tr, p1, " John Smith was a British actor.", 2, 3) - (L2.lp - 3)) > 1e-12)
        return err = "para_plan: lead penalty", false;
    if (p.label(tr, p1, " Anderson was a British actor.").lead) return err = "para_plan: a lead about the subject is not penalised", false;
    Spec sp;
    std::string e2;
    if (parse_flag("/nonexistent/para_plan_dir,w=2", sp, e2) || sp.plan || parse_flag("", sp, e2) || !parse_flag("off", sp, e2) || sp.plan ||
        Spec{}.plan)
        return err = "para_plan: flag", false;
    return true;
}

}  // namespace cypha::cyphalm::pp
