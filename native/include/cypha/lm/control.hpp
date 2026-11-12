// SPDX-License-Identifier: Apache-2.0
// docs/reports/K5_CONTROL.md (GENERATION_PLAN §13.76): a decoding controller that steers K5's text toward human text.
// Odin Loch, 2026-11-13.
//
// While K5 writes, the controller reads each finished sentence and keeps two running measures:
//   link:  does the sentence share a content word with the sentence before (the prompt's last sentence for the first)?
//          Content words as scripts/para_plan_mine.py's content(): runs of ASCII letters, lower-cased, 4+ letters,
//          not on its stop list. The share over sentences 2+ is hook_screen.py's on_prev.
//   words: words a sentence (runs of letters or digits, ' or - inside a word).
// Each is an exponentially weighted mean over the sentences so far (weight lam on the newest), started at its human
// target, so the first sentence is written with the run's own knobs. The error is the target minus the mean, less a
// dead band (no action inside the human range); a leaky integral (leak, clamped to [-2, 2]) adds the error that
// stays. The control signal u = clamp(kp e + ki I, -1, 1). Before each free sentence the knobs are set:
//   link  -> temperature T0 - tspan u, and the retrieved documents' weight rel0 2^(relspan u);
//   words -> wl u, the weight of a gate term: z(words in the draft) over the drafts (a draft that did not end with
//            . ! or ? counts as the shortest).
// It learns nothing from the text and changes no count: it only moves knobs. Off by default (--k5-control off).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace cypha::cyphalm::wkn::ctl {

struct Params {
    bool link = false, words = false;
    // targets (human GEN-A dev continuations, K5_CONTROL.md) and the loops' settings
    double link_t = 0.508, link_db = 0.15, link_kp = 4, link_ki = 1;
    double words_t = 20.4, words_db = 0.3, words_kp = 2, words_ki = 0.5;  // words: errors in log2 units
    double lam = 0.5, leak = 0.7;
    // the knobs: temperature T0 - tspan u (0: not moved), rel0 2^(relspan u) (0: not moved), the gate's length weight
    double tspan = 0.1, relspan = 1.0, wl = 1.0;
    bool on() const { return link || words; }
    std::string describe() const {
        if (!on()) return {};
        char b[320];
        std::snprintf(b, sizeof b, "\tcontrol=link:%d,words:%d,lt:%g,ldb:%g,lkp:%g,lki:%g,wt:%g,wdb:%g,wkp:%g,wki:%g,lam:%g,leak:%g,tspan:%g,relspan:%g,wl:%g",
                      link ? 1 : 0, words ? 1 : 0, link_t, link_db, link_kp, link_ki, words_t, words_db, words_kp, words_ki, lam, leak, tspan,
                      relspan, wl);
        return b;
    }
};

// --k5-control SPEC: off, or comma-separated fields link=0|1, words=0|1, lt, ldb, lkp, lki, wt, wdb, wkp, wki, lam,
// leak, tspan, relspan, wl.
inline bool parse(std::string_view spec, Params& p, std::string& err) {
    p = Params{};
    if (spec == "off") return true;
    std::size_t a = 0;
    while (a <= spec.size()) {
        std::size_t e = spec.find(',', a);
        if (e == std::string_view::npos) e = spec.size();
        const std::string f(spec.substr(a, e - a));
        const std::size_t q = f.find('=');
        if (q == std::string::npos) return err = "--k5-control: field without a value: " + f, false;
        const std::string k = f.substr(0, q), v = f.substr(q + 1);
        char* end = nullptr;
        const double x = std::strtod(v.c_str(), &end);
        if (v.empty() || *end || !std::isfinite(x)) return err = "--k5-control: bad value " + f, false;
        const auto in = [&](double& d, double lo, double hi) {
            if (!(x >= lo && x <= hi)) return err = "--k5-control: out of range " + f, false;
            d = x;
            return true;
        };
        bool ok = true;
        if (k == "link") p.link = x != 0;
        else if (k == "words") p.words = x != 0;
        else if (k == "lt") ok = in(p.link_t, 0, 1);
        else if (k == "ldb") ok = in(p.link_db, 0, 1);
        else if (k == "lkp") ok = in(p.link_kp, 0, 100);
        else if (k == "lki") ok = in(p.link_ki, 0, 100);
        else if (k == "wt") ok = in(p.words_t, 1, 200);
        else if (k == "wdb") ok = in(p.words_db, 0, 4);
        else if (k == "wkp") ok = in(p.words_kp, 0, 100);
        else if (k == "wki") ok = in(p.words_ki, 0, 100);
        else if (k == "lam") ok = in(p.lam, 0.01, 1);
        else if (k == "leak") ok = in(p.leak, 0, 1);
        else if (k == "tspan") ok = in(p.tspan, 0, 0.5);
        else if (k == "relspan") ok = in(p.relspan, 0, 4);
        else if (k == "wl") ok = in(p.wl, 0, 4);
        else return err = "--k5-control: unknown field " + k, false;
        if (!ok) return false;
        a = e + 1;
    }
    if (!p.on()) return err = "--k5-control: neither link=1 nor words=1 (use off)", false;
    return true;
}

// The knobs for the next sentence.
struct Knobs {
    double temperature = 1, rel = 0, wl = 0;
};

inline bool is_end_mark(char c) { return c == '.' || c == '!' || c == '?'; }
inline bool is_alnum(unsigned char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
inline bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
inline bool is_space(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

// para_plan_mine.py's content(): runs of ASCII letters, lower-cased, 4+ letters, not on its stop list. Sorted, unique.
inline std::vector<std::string> content_words(std::string_view s) {
    static const char* const kStop[] = {"that", "this", "with", "from", "were", "have", "which", "their", "there", "they", "been", "when",
        "also", "into", "than", "then", "them", "these", "those", "other", "about", "after", "before", "would", "could", "should", "while",
        "where", "being", "because", "some", "such", "more", "most", "many", "much", "over", "under", "only", "very", "what", "will", "your",
        "first", "second", "three", "during", "between", "within", "without", "through", "against", "among", "upon", "each", "both",
        "either", "well", "just", "like", "even"};
    std::vector<std::string> out;
    for (std::size_t i = 0; i < s.size();) {
        if (!is_alpha(s[i])) {
            ++i;
            continue;
        }
        std::string w;
        while (i < s.size() && is_alpha(s[i])) w += static_cast<char>(s[i] | 0x20), ++i;
        if (w.size() < 4) continue;
        bool stop = false;
        for (const char* t : kStop) stop = stop || w == t;
        if (!stop) out.push_back(std::move(w));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
// Words: runs of letters or digits (UTF-8 bytes count as letters), with ' or - inside a word.
inline int count_words(std::string_view s) {
    int n = 0;
    for (std::size_t i = 0; i < s.size();) {
        if (!is_alnum(static_cast<unsigned char>(s[i]))) {
            ++i;
            continue;
        }
        ++n;
        while (i < s.size()) {
            if (is_alnum(static_cast<unsigned char>(s[i]))) ++i;
            else if ((s[i] == '\'' || s[i] == '-') && i + 1 < s.size() && is_alnum(static_cast<unsigned char>(s[i + 1]))) ++i;
            else break;
        }
    }
    return n;
}
// Sentences of t, as para_plan_mine.py's split_sents cuts them (near enough): after . ! ? (and closing quotes or
// brackets) followed by white space and a capital, a digit or an opening mark, not after a one-letter word or a listed
// abbreviation; a piece under 24 bytes joins the next. The last piece is kept whether it ends or not.
inline std::vector<std::string> split_sentences(std::string_view t) {
    static const char* const kAbbr[] = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "inc", "co", "ltd", "no", "gen", "col", "lt", "sgt",
                                        "prof", "rev", "mt", "ft", "ca", "c", "e.g", "i.e", "etc", "u.s", "u.k", "fig", "vol", "pp", "ed"};
    std::vector<std::string> out;
    std::string pend;
    const auto push = [&](std::string_view piece) {
        const std::size_t x = piece.find_first_not_of(" \t\n\r"), y = piece.find_last_not_of(" \t\n\r");
        if (x == std::string_view::npos) return;
        std::string p(piece.substr(x, y - x + 1));
        std::string joined = pend.empty() ? std::move(p) : pend + " " + p;
        pend.clear();
        if (joined.size() < 24) pend = std::move(joined);
        else out.push_back(std::move(joined));
    };
    std::size_t a = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (!is_end_mark(t[i])) continue;
        std::size_t j = i + 1;
        while (j < t.size() && (t[j] == '"' || t[j] == '\'' || t[j] == ')' || t[j] == ']')) ++j;
        std::size_t k = j;
        while (k < t.size() && is_space(t[k])) ++k;
        if (k == j || k >= t.size()) continue;
        const char n = t[k];
        if (!((n >= 'A' && n <= 'Z') || (n >= '0' && n <= '9') || n == '"' || n == '\'' || n == '(' || n == '[')) continue;
        std::size_t w1 = i;  // the last word before the mark
        while (w1 > a && !is_space(t[w1 - 1])) --w1;
        std::string last(t.substr(w1, i - w1));
        while (!last.empty() && (is_end_mark(last.back()) || last.back() == '"' || last.back() == '\'' || last.back() == ')' || last.back() == ']'))
            last.pop_back();
        for (char& c : last) c = static_cast<char>(c >= 'A' && c <= 'Z' ? c | 0x20 : c);
        bool abbr = last.size() == 1 && is_alpha(last[0]);
        for (const char* x : kAbbr) abbr = abbr || last == x;
        if (abbr) continue;
        push(t.substr(a, j - a));
        a = k;
        i = k - 1;
    }
    push(t.substr(a));
    if (!pend.empty()) out.push_back(std::move(pend));
    return out;
}

// One continuation's controller: what it has read and its loops' state.
struct Track {
    std::size_t pos = 0;             // bytes of the text read so far
    std::vector<std::string> prev;   // the content words of the last sentence read (the prompt's last at the start)
    double x_link = 0, x_words = 0;  // the running means (words in log2)
    double i_link = 0, i_words = 0;  // the leaky integrals
    int n = 0;                       // sentences read

    void begin(std::string_view prompt, const Params& p) {
        const auto ps = split_sentences(prompt);
        prev = ps.empty() ? std::vector<std::string>{} : content_words(ps.back());
        x_link = p.link_t, x_words = std::log2(p.words_t);
    }
    static double dead(double e, double db) { return e > db ? e - db : e < -db ? e + db : 0.0; }
    // Reads one finished sentence.
    void read(std::string_view s, const Params& p) {
        const auto cw = content_words(s);
        bool hit = false;
        for (std::size_t i = 0, j = 0; i < cw.size() && j < prev.size();) {
            if (cw[i] == prev[j]) {
                hit = true;
                break;
            }
            if (cw[i] < prev[j]) ++i;
            else ++j;
        }
        const int w = count_words(s);
        x_link = (1 - p.lam) * x_link + p.lam * (hit ? 1.0 : 0.0);
        x_words = (1 - p.lam) * x_words + p.lam * std::log2(static_cast<double>(std::max(w, 1)));
        i_link = std::clamp(p.leak * i_link + dead(p.link_t - x_link, p.link_db), -2.0, 2.0);
        i_words = std::clamp(p.leak * i_words + dead(std::log2(p.words_t) - x_words, p.words_db), -2.0, 2.0);
        prev = cw;
        ++n;
    }
    // Reads the sentences written since the last call (out[pos, end); K5 calls it at a sentence boundary), then gives
    // the knobs for the next sentence. t0 and rel0: the run's own.
    Knobs step(const Params& p, std::string_view out, double t0, double rel0) {
        if (out.size() > pos) {
            for (const auto& s : split_sentences(out.substr(pos))) read(s, p);
            pos = out.size();
        }
        Knobs k;
        k.temperature = t0, k.rel = rel0;
        if (p.link) {
            const double u = std::clamp(p.link_kp * dead(p.link_t - x_link, p.link_db) + p.link_ki * i_link, -1.0, 1.0);
            if (p.tspan > 0) k.temperature = t0 - p.tspan * u;
            if (p.relspan > 0 && rel0 > 0) k.rel = rel0 * std::exp2(p.relspan * u);
        }
        if (p.words) {
            const double u = std::clamp(p.words_kp * dead(std::log2(p.words_t) - x_words, p.words_db) + p.words_ki * i_words, -1.0, 1.0);
            k.wl = p.wl * u;
        }
        return k;
    }
};

// The length term's raw value for a draft: its words, or -1 if it did not end with . ! or ? (scored as the shortest).
inline double draft_words(std::string_view text) {
    const std::size_t e = text.find_last_not_of(" \t\n\r\"')]");
    if (e == std::string_view::npos || !is_end_mark(text[e])) return -1;
    return count_words(text);
}
// z-scores of the drafts' length values; an unended draft (-1) takes the lowest ended value (0 if none ended).
inline std::vector<double> length_z(std::vector<double> v) {
    double lo = 1e300;
    for (const double x : v)
        if (x >= 0) lo = std::min(lo, x);
    if (lo == 1e300) lo = 0;
    for (double& x : v)
        if (x < 0) x = lo;
    const double n = static_cast<double>(std::max<std::size_t>(v.size(), 1));
    double m = 0, s = 0;
    for (const double x : v) m += x;
    m /= n;
    for (const double x : v) s += (x - m) * (x - m);
    const double sd = std::sqrt(s / n);
    for (double& x : v) x = sd > 1e-12 ? (x - m) / sd : 0.0;
    return v;
}

}  // namespace cypha::cyphalm::wkn::ctl
