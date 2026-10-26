// B7: an honest, fuzzy chat index. Retrieval, not understanding.
//
// Replaces copied_answer() in trie_chat.cpp (a std::string::find of the whole
// prompt in the training bytes: fires on 975/975 verbatim Dolly prompts and
// 0/975 with the first letter lowercased). This tool indexes the QUESTION of
// every Dolly record (instruction, plus context when the record has one),
// returns the stored response of the best match, and ABSTAINS when the match
// is not good enough. It says "retrieval" because that is all it is.
//
// Index        normalise (lowercase, Unicode -> ASCII where trivial, punctuation
//              stripped, whitespace collapsed; optional light suffix stemming;
//              optional removal of leading/trailing courtesy words such as
//              "please" / "thanks"), then two scorers over the same text:
//                bm25   BM25, k1 = 1.2, b = 0.75, over word tokens
//                cos    TF-IDF cosine over character 3-grams
//                sum    mean of bm25 / self-score and cos
//              bm25 is normalised by the self-score (the score the query would
//              get from a stored question identical to it), so it lies in [0, 1].
// Abstain      answer only if score >= ts and (best - runner-up) >= tm, where the
//              runner-up is the best match with a different question text. (ts, tm)
//              are chosen per scorer on the dev set D only, at two points:
//                G (headline)  every changed form keeps >= 0.96 of verbatim hits on D,
//                              precision >= 95%, >= 95% of verbatim hits still answered,
//                              then the most leave-one-out abstention
//                B             precision >= 95%, then the best mean of (perturbed answered
//                              correctly, leave-one-out abstained)
//
// Protocol     (records are numbered in file order after dropping empty
//              instruction/response, as load_dolly() does)
//   H  index % 10 == 3   held out, never indexed (T-QA-dolly). Records whose
//                        normalised instruction equals an indexed one are removed
//                        and counted (near-duplicate leak).
//   D  index % 10 == 7   indexed. Dev only: the five perturbed forms of every D
//                        record (should answer correctly) and the leave-one-out
//                        verbatim query (own entry excluded; should abstain) pick
//                        (ts, tm) at the points G and B described above.
//   P  1000 indexed records not in D, fixed seed (T-perturb), five forms:
//        verbatim | first letter lowercased | " Thanks." appended to the
//        instruction | punctuation deleted (instruction and context) | one word
//        of the instruction swapped for a synonym from a built-in table.
//        Accuracy = the returned response is the target's response. Gate: every
//        changed form >= 95% of verbatim accuracy.
//   H  abstain rate; token-F1 of the answered ones against the reference.
//   Speed: build time, query p50/p99, serialised index bytes, peak RSS.
//   Four configurations are run: plain, +stem, +filler, +stem+filler. The
//   headline is the one with the best dev score (chosen on D, not on H or P).
//
// Measured 2026-10-27 (Dolly 15,015 records; 13,513 indexed; H 1,453 after dropping 49
//   near-duplicates; 2 cores, g++ 13.3 -O2). Headline by dev score: plain normalisation,
//   "sum" scorer, point G (ts 0.730, tm 0.100):
//   T-perturb (1000 records): top-1 accuracy 98.7% on every form except synonym 97.8%
//     (ratio >= 0.992, gate PASS with no abstain). With abstain at G: verbatim 93.9%,
//     changed forms 0.957..1.000 of verbatim, gate PASS (narrowly; synonym is n = 362 and
//     fails at B and on 1 of 3 other seeds). 18 of the 1000 targets share their question
//     text with another record that has a different response, so some misses cannot be fixed.
//   H: abstains on 92.6%; of the 107 answered the mean F1 is 0.276 (13% above 0.5);
//     if forced to answer all, mean F1 is 0.175. A genuinely new question gets an
//     unrelated stored answer, or none.
//   Speed: build 1.0 s, index 28.0 MB + responses 4.9 MB, query p50 0.34 ms, p99 3.2 ms.
//   Report: docs/reports/B7_CHAT_INDEX.md
//
// Fast path (S1, 2026-10-27). --ask "q" and --ask-file questions.tsv no longer rerun the sweep (about 52 s per
//   question). They load Dolly, build the headline index once (about 1 s) and read the headline operating point
//   (configuration, scorer, ts, tm at point G) from a small thresholds file next to the data
//   (<dolly.jsonl>.chat_index_op, or --thresholds PATH). The file holds ts and tm as %.17g, so they are
//   bit-identical to the sweep's. The sweep writes it at the end of every run; if it is missing, or was written
//   for other data or settings (file size, record count, --irep, --configs), the fast path runs the dev stage of
//   every configuration once (about 27 s, no T-perturb, H or speed runs), writes the file, and carries on.
//   --sweep with --ask / --ask-file restores the old behaviour (sweep first, then answer); --recalibrate forces
//   the one-off dev stage. Cached: one question 1.3 s end to end, 200 questions 1.3 s.
//
// Answers and abstentions (DISCOVERY 3.1 and 10). A question is answered only if score >= ts and
//   (best - runner-up) >= tm. Otherwise it abstains with a reason code:
//     NO_EVIDENCE         the best stored question scores below ts
//     AMBIGUOUS_QUESTION  the best scores >= ts but a stored question with a different text is within tm of it
//   (if both tests fail the code is NO_EVIDENCE: a margin between two weak matches is not ambiguity).
//   Every answer prints an evidence card (matched stored question, score, margin, record index, category);
//   every abstention prints an "I don't know" card with the nearest stored question (and the runner-up when
//   the reason is AMBIGUOUS_QUESTION). Each question also prints one machine-readable line,
//   "RESULT_JSON {...}" (decision, reason, score, margin, thresholds, nearest and runner-up question, record
//   index, category, answer); --quiet prints only those lines (timing goes to stderr).
//
// Usage: g++ -O2 -std=c++20 -I native/third_party native/tools/chat_index.cpp -o chat_index
//        chat_index dolly.jsonl --ask "a question"            fast path, one card + one RESULT_JSON line
//        chat_index dolly.jsonl --ask-file questions.tsv      fast path, first tab-separated column = question
//                                                             (a first line "question" is a header)
//        chat_index dolly.jsonl [--seed N] [--irep N] [--configs plain,stem,filler,both]
//                                                             the full evaluation sweep
//   --irep N        repeat the instruction N times in the indexed text (default 1)
//   --sweep         with --ask / --ask-file: run the full sweep first (old behaviour), then answer
//   --recalibrate   ignore the thresholds file and recompute it (dev stage only)
//   --thresholds P  where the thresholds file lives (default <dolly.jsonl>.chat_index_op)
//   --quiet         (fast path) print only the RESULT_JSON lines; the timing line goes to stderr

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kK1 = 1.2, kB = 0.75;
constexpr int kScorers = 3;
const char* const kScorerName[kScorers] = {"bm25", "cos3", "sum"};

double secs_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// ------------------------------------------------------------ normalisation

const char kExtA[] = "aaaaaa" "cccccccc" "dddd" "eeeeeeeeee" "gggggggg" "hhhh" "iiiiiiiiii" "ii" "jj" "kkk"
                     "llllllllll" "nnnnnnnnn" "oooooooo" "rrrrrr" "ssssssss" "tttttt" "uuuuuuuuuuuu" "ww" "yyy"
                     "zzzzzz" "s";
static_assert(sizeof(kExtA) == 129, "Latin Extended-A table must cover U+0100..U+017F");
const char kLat1[] = "aaaaaa" "a" "c" "eeee" "iiii" "d" "n" "ooooo" " " "o" "uuuu" "y" "t" "s"   // U+00C0..U+00DF
                     "aaaaaa" "a" "c" "eeee" "iiii" "d" "n" "ooooo" " " "o" "uuuu" "y" "t" "y";  // U+00E0..U+00FF
static_assert(sizeof(kLat1) == 65, "Latin-1 table must cover U+00C0..U+00FF");

void append_folded(std::uint32_t cp, const char* raw, std::size_t rawlen, std::string& out) {
    if (cp < 0x80) { out += static_cast<char>(cp); return; }
    if (cp == 0xC6 || cp == 0xE6) { out += "ae"; return; }
    if (cp == 0xDF) { out += "ss"; return; }
    if (cp == 0x152 || cp == 0x153) { out += "oe"; return; }
    if (cp >= 0xA1 && cp <= 0xBF) { out += ' '; return; }
    if (cp == 0xA0) { out += ' '; return; }
    if (cp >= 0xC0 && cp <= 0xFF) { out += kLat1[cp - 0xC0]; return; }
    if (cp >= 0x100 && cp <= 0x17F) { out += kExtA[cp - 0x100]; return; }
    if (cp == 0x2018 || cp == 0x2019 || cp == 0x201A || cp == 0x201B || cp == 0x2032 || cp == 0x2BC) { out += '\''; return; }
    if ((cp >= 0x201C && cp <= 0x201F) || cp == 0x2033) { out += '"'; return; }
    if ((cp >= 0x2010 && cp <= 0x2015) || cp == 0x2212) { out += '-'; return; }
    if (cp == 0x2026 || cp == 0x2022) { out += ' '; return; }
    out.append(raw, rawlen);  // anything else keeps its UTF-8 bytes and stays inside a token
}

std::string fold_ascii(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) { out += static_cast<char>(c); ++i; continue; }
        const int n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
        if (n == 0 || i + static_cast<std::size_t>(n) > s.size()) { out += static_cast<char>(c); ++i; continue; }
        std::uint32_t cp = c & (0xFFu >> (n + 1));
        bool ok = true;
        for (int k = 1; k < n; ++k) {
            const unsigned char d = static_cast<unsigned char>(s[i + k]);
            if ((d & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (d & 0x3F);
        }
        if (!ok) { out += static_cast<char>(c); ++i; continue; }
        append_folded(cp, s.data() + i, static_cast<std::size_t>(n), out);
        i += static_cast<std::size_t>(n);
    }
    return out;
}

// lowercase, fold, apostrophes deleted, every other non-alphanumeric byte is a space, spaces collapsed
std::string norm(const std::string& s) {
    const std::string f = fold_ascii(s);
    std::string out;
    out.reserve(f.size());
    bool pend = false;
    for (const unsigned char c : f) {
        if (c == '\'') continue;
        char lc;
        if (c >= 'A' && c <= 'Z') lc = static_cast<char>(c + 32);
        else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c >= 0x80) lc = static_cast<char>(c);
        else { pend = !out.empty(); continue; }
        if (pend) { out += ' '; pend = false; }
        out += lc;
    }
    return out;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> t;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && s[i] == ' ') ++i;
        std::size_t j = i;
        while (j < s.size() && s[j] != ' ') ++j;
        if (j > i) t.push_back(s.substr(i, j - i));
        i = j;
    }
    return t;
}

std::string stem(std::string w) {
    for (const char c : w)
        if (c < 'a' || c > 'z') return w;  // numbers and non-ASCII tokens are left alone
    const std::size_t L = w.size();
    if (L <= 3) return w;
    auto ends = [&](const char* s) {
        const std::size_t n = std::strlen(s);
        return w.size() >= n && w.compare(w.size() - n, n, s) == 0;
    };
    auto undouble = [&]() {
        const std::size_t n = w.size();
        if (n >= 2 && w[n - 1] == w[n - 2] && w[n - 1] != 'l' && w[n - 1] != 's' && w[n - 1] != 'z') w.pop_back();
    };
    if (ends("ies") && L > 4) { w.resize(L - 3); w += 'y'; }
    else if (ends("sses")) w.resize(L - 2);
    else if (ends("xes") || ends("zes") || ends("ches") || ends("shes")) w.resize(L - 2);
    else if (ends("ing") && L > 5) { w.resize(L - 3); undouble(); }
    else if (ends("ed") && L > 4) { w.resize(L - 2); undouble(); }
    else if (ends("s") && !ends("ss") && !ends("us") && !ends("is")) w.resize(L - 1);
    if (w.size() > 3 && w.back() == 'e') w.pop_back();
    return w;
}

// leading "hi / hello / hey / please", trailing "thanks / thank you / please / thx ... (in advance)"
void strip_filler(std::vector<std::string>& t) {
    static const std::unordered_set<std::string> lead = {"hi", "hello", "hey", "please", "pls", "kindly"};
    static const std::unordered_set<std::string> trig = {"thanks", "thank", "thx", "ty", "please", "pls"};
    static const std::unordered_set<std::string> tail = {"thanks", "thank", "thx", "ty", "please", "pls", "you", "in",
                                                         "advance", "a", "lot", "so", "much", "very", "again", "many"};
    std::size_t b = 0;
    while (b + 1 < t.size() && lead.count(t[b])) ++b;
    if (b) t.erase(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(b));
    std::size_t j = t.size();
    while (j > 0 && tail.count(t[j - 1])) --j;
    std::size_t k = t.size();
    for (std::size_t i = j; i < t.size(); ++i)
        if (trig.count(t[i])) { k = i; break; }
    if (k == t.size()) return;
    if (k > j && t[k - 1] == "many") --k;
    if (k >= 1) t.resize(k);
}

struct Cfg {
    bool stem = false, filler = false;
    std::string name;
};

struct QDoc {
    std::vector<std::string> toks;
    std::string tri;  // " tok tok tok " for the character 3-gram scorer
};

QDoc make_q(const std::string& instr, const std::string& ctx, const Cfg& cfg, int irep) {
    std::vector<std::string> a = split(norm(instr));
    if (cfg.filler) strip_filler(a);
    const std::vector<std::string> c = split(norm(ctx));
    QDoc q;
    for (int r = 0; r < irep; ++r) q.toks.insert(q.toks.end(), a.begin(), a.end());
    q.toks.insert(q.toks.end(), c.begin(), c.end());
    if (cfg.stem)
        for (auto& w : q.toks) w = stem(w);
    q.tri = " ";
    for (const auto& w : q.toks) { q.tri += w; q.tri += ' '; }
    return q;
}

// ------------------------------------------------------------------ data

struct Rec {
    std::string instr, ctx, resp, cat;
    std::string ikey;  // norm(instr)
    int keyid = 0;     // id of norm(instr + " " + ctx): records with the same question text share it
};

std::vector<Rec> load_dolly(const std::string& path) {
    std::vector<Rec> recs;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto j = nlohmann::json::parse(line);
        Rec r;
        r.instr = j.value("instruction", "");
        r.ctx = j.value("context", "");
        r.resp = j.value("response", "");
        r.cat = j.value("category", "");
        if (r.instr.empty() || r.resp.empty()) continue;
        recs.push_back(std::move(r));
    }
    return recs;
}

double token_f1(const std::string& a, const std::string& b) {
    const auto ta = split(norm(a)), tb = split(norm(b));
    if (ta.empty() && tb.empty()) return 1.0;
    if (ta.empty() || tb.empty()) return 0.0;
    std::unordered_map<std::string, int> cnt;
    for (const auto& w : tb) ++cnt[w];
    int common = 0;
    for (const auto& w : ta) {
        auto it = cnt.find(w);
        if (it != cnt.end() && it->second > 0) { --it->second; ++common; }
    }
    if (common == 0) return 0.0;
    const double p = static_cast<double>(common) / static_cast<double>(ta.size());
    const double r = static_cast<double>(common) / static_cast<double>(tb.size());
    return 2 * p * r / (p + r);
}

// ------------------------------------------------------------ perturbations

enum Form { kVerb = 0, kLower, kThanks, kNoPunct, kSyn, kForms };
const char* const kFormName[kForms] = {"verbatim", "lower-first", "+ Thanks.", "no punctuation", "synonym"};

const std::vector<std::pair<std::string, std::string>>& syn_pairs() {
    static const std::vector<std::pair<std::string, std::string>> p = {
        {"big", "large"}, {"small", "little"}, {"buy", "purchase"}, {"start", "begin"}, {"end", "finish"},
        {"show", "display"}, {"help", "assist"}, {"make", "create"}, {"get", "obtain"}, {"use", "utilize"},
        {"fast", "quick"}, {"smart", "intelligent"}, {"happy", "glad"}, {"rich", "wealthy"}, {"hard", "difficult"},
        {"easy", "simple"}, {"choose", "select"}, {"tell", "inform"}, {"say", "state"}, {"ask", "inquire"},
        {"car", "automobile"}, {"house", "home"}, {"kid", "child"}, {"doctor", "physician"}, {"job", "career"},
        {"movie", "film"}, {"photo", "picture"}, {"pretty", "beautiful"}, {"huge", "enormous"},
        {"important", "significant"}, {"need", "require"}, {"find", "locate"}, {"build", "construct"},
        {"fix", "repair"}, {"stop", "halt"}, {"keep", "retain"}, {"give", "provide"}, {"list", "enumerate"},
        {"purpose", "goal"}, {"method", "approach"}, {"main", "primary"}, {"many", "numerous"}, {"best", "finest"},
        {"difference", "distinction"}, {"ocean", "sea"}, {"speed", "velocity"}, {"explain", "describe"},
        {"mistake", "error"}, {"idea", "concept"}, {"trip", "journey"}, {"strange", "odd"}, {"answer", "reply"},
        {"problem", "issue"}, {"talk", "speak"}, {"cheap", "inexpensive"}, {"change", "alter"}};
    return p;
}

const std::unordered_map<std::string, std::string>& syn_map() {
    static const std::unordered_map<std::string, std::string> m = [] {
        std::unordered_map<std::string, std::string> r;
        for (const auto& [a, b] : syn_pairs()) {
            r.emplace(a, b);
            r.emplace(b, a);
        }
        return r;
    }();
    return m;
}

std::string delete_punct(const std::string& s) {
    std::string o;
    for (const unsigned char c : s)
        if (!(c < 0x80 && std::ispunct(c))) o += static_cast<char>(c);
    return o;
}

// returns false when the form does not apply to (or does not change) this record
bool make_form(Form f, const Rec& r, std::string& instr, std::string& ctx) {
    instr = r.instr;
    ctx = r.ctx;
    switch (f) {
        case kVerb: return true;
        case kLower:
            for (char& c : instr) {
                if (std::isalpha(static_cast<unsigned char>(c))) {
                    if (c >= 'A' && c <= 'Z') { c = static_cast<char>(c + 32); return true; }
                    return false;
                }
            }
            return false;
        case kThanks: instr += " Thanks."; return true;
        case kNoPunct:
            instr = delete_punct(r.instr);
            ctx = delete_punct(r.ctx);
            return instr != r.instr || ctx != r.ctx;
        case kSyn: {
            const auto& m = syn_map();
            std::size_t i = 0;
            while (i < instr.size()) {
                while (i < instr.size() && !std::isalpha(static_cast<unsigned char>(instr[i]))) ++i;
                std::size_t j = i;
                while (j < instr.size() && std::isalpha(static_cast<unsigned char>(instr[j]))) ++j;
                if (j > i) {
                    std::string w = instr.substr(i, j - i), lw = w;
                    for (char& c : lw) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    auto it = m.find(lw);
                    if (it != m.end()) {
                        std::string rep = it->second;
                        if (std::isupper(static_cast<unsigned char>(w[0]))) rep[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(rep[0])));
                        instr = instr.substr(0, i) + rep + instr.substr(j);
                        return true;
                    }
                }
                i = j;
            }
            return false;
        }
        default: return false;
    }
}

// ------------------------------------------------------------------ index

struct Res {
    int doc[kScorers] = {-1, -1, -1};
    float s1[kScorers] = {0, 0, 0};  // normalised best score
    float s2[kScorers] = {0, 0, 0};  // best score of a different question text
    int doc2[kScorers] = {-1, -1, -1};  // the document behind s2 (-1: none)
};

struct Index {
    std::size_t N = 0;
    double avgdl = 1;
    std::vector<int> rec_of, keyid_of, doc_of;
    std::vector<float> dl, K;
    // word BM25
    std::unordered_map<std::string, std::uint32_t> vocab;
    std::vector<std::uint32_t> w_off, w_doc, w_tf;
    std::vector<float> w_idf;
    std::size_t vocab_bytes = 0;
    // character 3-gram TF-IDF
    std::vector<std::uint32_t> t_key, t_off, t_doc;
    std::vector<float> t_w, t_idf;
    // build timing
    double t_bm25 = 0, t_tri = 0;
    // scratch
    std::vector<double> ab;
    std::vector<float> ac;

    void build(const std::vector<Rec>& recs, const std::vector<int>& ids, const Cfg& cfg, int irep) {
        N = ids.size();
        rec_of = ids;
        doc_of.assign(recs.size(), -1);
        keyid_of.resize(N);
        std::vector<QDoc> docs(N);
        dl.resize(N);
        double tot = 0;
        for (std::size_t d = 0; d < N; ++d) {
            doc_of[static_cast<std::size_t>(ids[d])] = static_cast<int>(d);
            keyid_of[d] = recs[static_cast<std::size_t>(ids[d])].keyid;
            docs[d] = make_q(recs[static_cast<std::size_t>(ids[d])].instr, recs[static_cast<std::size_t>(ids[d])].ctx, cfg, irep);
            dl[d] = static_cast<float>(docs[d].toks.size());
            tot += dl[d];
        }
        avgdl = std::max(1.0, tot / static_cast<double>(N));
        K.resize(N);
        for (std::size_t d = 0; d < N; ++d) K[d] = static_cast<float>(kK1 * (1 - kB + kB * dl[d] / avgdl));

        auto t0 = Clock::now();
        std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> post;
        for (std::size_t d = 0; d < N; ++d) {
            std::vector<std::string> toks = docs[d].toks;
            std::sort(toks.begin(), toks.end());
            for (std::size_t i = 0; i < toks.size();) {
                std::size_t j = i;
                while (j < toks.size() && toks[j] == toks[i]) ++j;
                auto it = vocab.find(toks[i]);
                if (it == vocab.end()) {
                    it = vocab.emplace(toks[i], static_cast<std::uint32_t>(post.size())).first;
                    post.emplace_back();
                    vocab_bytes += toks[i].size() + 1;
                }
                post[it->second].emplace_back(static_cast<std::uint32_t>(d), static_cast<std::uint32_t>(j - i));
                i = j;
            }
        }
        w_off.assign(post.size() + 1, 0);
        w_idf.resize(post.size());
        for (std::size_t t = 0; t < post.size(); ++t) {
            w_off[t + 1] = w_off[t] + static_cast<std::uint32_t>(post[t].size());
            const double df = static_cast<double>(post[t].size());
            w_idf[t] = static_cast<float>(std::log(1.0 + (static_cast<double>(N) - df + 0.5) / (df + 0.5)));
        }
        w_doc.resize(w_off.back());
        w_tf.resize(w_off.back());
        for (std::size_t t = 0; t < post.size(); ++t)
            for (std::size_t p = 0; p < post[t].size(); ++p) {
                w_doc[w_off[t] + p] = post[t][p].first;
                w_tf[w_off[t] + p] = post[t][p].second;
            }
        t_bm25 = secs_since(t0);

        t0 = Clock::now();
        struct Tri { std::uint32_t key, doc, cnt; };
        std::vector<Tri> tri;
        std::vector<std::uint32_t> keys;
        for (std::size_t d = 0; d < N; ++d) {
            keys.clear();
            const std::string& s = docs[d].tri;
            for (std::size_t i = 0; i + 3 <= s.size(); ++i)
                keys.push_back((static_cast<std::uint32_t>(static_cast<unsigned char>(s[i])) << 16) |
                               (static_cast<std::uint32_t>(static_cast<unsigned char>(s[i + 1])) << 8) |
                               static_cast<std::uint32_t>(static_cast<unsigned char>(s[i + 2])));
            std::sort(keys.begin(), keys.end());
            for (std::size_t i = 0; i < keys.size();) {
                std::size_t j = i;
                while (j < keys.size() && keys[j] == keys[i]) ++j;
                tri.push_back({keys[i], static_cast<std::uint32_t>(d), static_cast<std::uint32_t>(j - i)});
                i = j;
            }
        }
        std::sort(tri.begin(), tri.end(), [](const Tri& a, const Tri& b) { return a.key != b.key ? a.key < b.key : a.doc < b.doc; });
        std::vector<double> norm2(N, 0.0);
        std::vector<float> idf_of(tri.size());
        for (std::size_t i = 0; i < tri.size();) {
            std::size_t j = i;
            while (j < tri.size() && tri[j].key == tri[i].key) ++j;
            const double df = static_cast<double>(j - i);
            const double idf = std::log((static_cast<double>(N) + 1.0) / (df + 1.0)) + 1.0;
            t_key.push_back(tri[i].key);
            t_off.push_back(static_cast<std::uint32_t>(i));
            t_idf.push_back(static_cast<float>(idf));
            for (std::size_t p = i; p < j; ++p) {
                const double w = (1.0 + std::log(static_cast<double>(tri[p].cnt))) * idf;
                norm2[tri[p].doc] += w * w;
                idf_of[p] = static_cast<float>(idf);
            }
            i = j;
        }
        t_off.push_back(static_cast<std::uint32_t>(tri.size()));
        t_doc.resize(tri.size());
        t_w.resize(tri.size());
        for (std::size_t p = 0; p < tri.size(); ++p) {
            const double w = (1.0 + std::log(static_cast<double>(tri[p].cnt))) * idf_of[p];
            t_doc[p] = tri[p].doc;
            t_w[p] = static_cast<float>(w / std::sqrt(norm2[tri[p].doc]));
        }
        t_tri = secs_since(t0);
        ab.assign(N, 0.0);
        ac.assign(N, 0.0f);
    }

    // mask: bit 0 = bm25, bit 1 = cos3, 3 = both (and sum). exclude_rec: leave-one-out for the dev set.
    void search(const QDoc& q, int exclude_rec, int mask, Res& out) {
        out = Res{};
        const double idf_unseen = std::log(1.0 + (static_cast<double>(N) + 0.5) / 0.5);
        double self = 0;
        if (mask & 1) {
            std::fill(ab.begin(), ab.end(), 0.0);
            const double qlen = static_cast<double>(q.toks.size());
            const double qK = kK1 * (1 - kB + kB * qlen / avgdl);
            std::unordered_map<std::string, int> qt;
            for (const auto& w : q.toks) ++qt[w];
            for (const auto& [w, qtf] : qt) {
                const auto it = vocab.find(w);
                const double idf = it == vocab.end() ? idf_unseen : w_idf[it->second];
                self += idf * qtf * (kK1 + 1) / (qtf + qK);
                if (it == vocab.end()) continue;
                const std::uint32_t t = it->second;
                for (std::uint32_t p = w_off[t]; p < w_off[t + 1]; ++p) {
                    const std::uint32_t d = w_doc[p];
                    const double tf = w_tf[p];
                    ab[d] += idf * tf * (kK1 + 1) / (tf + K[d]);
                }
            }
        }
        double qden = 0;
        if (mask & 2) {
            std::fill(ac.begin(), ac.end(), 0.0f);
            std::vector<std::uint32_t> keys;
            for (std::size_t i = 0; i + 3 <= q.tri.size(); ++i)
                keys.push_back((static_cast<std::uint32_t>(static_cast<unsigned char>(q.tri[i])) << 16) |
                               (static_cast<std::uint32_t>(static_cast<unsigned char>(q.tri[i + 1])) << 8) |
                               static_cast<std::uint32_t>(static_cast<unsigned char>(q.tri[i + 2])));
            std::sort(keys.begin(), keys.end());
            double qn2 = 0;
            const double idf_unseen_t = std::log(static_cast<double>(N) + 1.0) + 1.0;
            for (std::size_t i = 0; i < keys.size();) {
                std::size_t j = i;
                while (j < keys.size() && keys[j] == keys[i]) ++j;
                const auto lb = std::lower_bound(t_key.begin(), t_key.end(), keys[i]);
                const bool present = lb != t_key.end() && *lb == keys[i];
                const std::size_t pos = static_cast<std::size_t>(lb - t_key.begin());
                const double idf = present ? t_idf[pos] : idf_unseen_t;
                const double w = (1.0 + std::log(static_cast<double>(j - i))) * idf;
                qn2 += w * w;
                if (present)
                    for (std::uint32_t p = t_off[pos]; p < t_off[pos + 1]; ++p) ac[t_doc[p]] += static_cast<float>(w * t_w[p]);
                i = j;
            }
            qden = std::sqrt(qn2);
        }
        const int ex = exclude_rec >= 0 ? doc_of[static_cast<std::size_t>(exclude_rec)] : -1;
        const double bs = self > 0 ? 1.0 / self : 0.0;
        const double cs = qden > 0 ? 1.0 / qden : 0.0;
        for (int k = 0; k < kScorers; ++k) {
            if (k == 0 && !(mask & 1)) continue;
            if (k == 1 && !(mask & 2)) continue;
            if (k == 2 && mask != 3) continue;
            int d1 = -1, d2 = -1;
            double v1 = -1, v2 = -1;
            for (std::size_t d = 0; d < N; ++d) {
                if (static_cast<int>(d) == ex) continue;
                const double v = k == 0 ? ab[d] * bs : k == 1 ? ac[d] * cs : 0.5 * (ab[d] * bs + ac[d] * cs);
                if (d1 < 0) { d1 = static_cast<int>(d); v1 = v; continue; }
                if (keyid_of[d] == keyid_of[static_cast<std::size_t>(d1)]) {
                    if (v > v1) { d1 = static_cast<int>(d); v1 = v; }
                } else if (v > v1) {
                    d2 = d1; v2 = v1; d1 = static_cast<int>(d); v1 = v;
                } else if (v > v2) {
                    d2 = static_cast<int>(d); v2 = v;
                }
            }
            out.doc2[k] = d2;
            out.doc[k] = d1;
            out.s1[k] = static_cast<float>(std::max(0.0, v1));
            out.s2[k] = static_cast<float>(std::max(0.0, v2));
        }
    }

    // serialised-equivalent size (what a saved index would occupy), in bytes
    struct Size { std::size_t bm25, tri, docs, total; };
    Size bytes() const {
        Size s{};
        s.bm25 = 8 * w_doc.size() + 4 * w_off.size() + vocab_bytes;
        s.tri = 8 * t_doc.size() + 8 * t_key.size() + 4;
        s.docs = 8 * N + 4 * N;  // rec id + question key + doc length
        s.total = s.bm25 + s.tri + s.docs;
        return s;
    }
};

// ------------------------------------------------------------ evaluation

struct Item {
    Res r;
    bool ok[kScorers] = {false, false, false};
    double f1[kScorers] = {0, 0, 0};
    int form = 0, rid = -1;  // dev set only: which perturbation, which record
};

struct Op {
    double ts = 0, tm = 0, prec = 0, rec = 0, neg_abs = 0, obj = -1, min_ratio = 0, vkeep = 0;
    bool met = false;
};

bool answers(const Item& it, int k, const Op& op) {
    return it.r.s1[k] >= op.ts && (it.r.s1[k] - it.r.s2[k]) >= op.tm;
}

Op select_op(const std::vector<Item>& pos, const std::vector<Item>& neg, int k) {
    static const double tms[] = {0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2};
    Op best;
    double best_prec = -1;
    Op fallback;
    for (double tm : tms) {
        for (int i = 1; i <= 199; ++i) {
            const double ts = 0.005 * i;
            Op c{ts, tm};
            std::size_t ans = 0, cor = 0, nab = 0;
            for (const auto& it : pos)
                if (answers(it, k, c)) { ++ans; cor += it.ok[k]; }
            for (const auto& it : neg)
                if (!answers(it, k, c)) ++nab;
            c.prec = ans ? static_cast<double>(cor) / static_cast<double>(ans) : 1.0;
            c.rec = pos.empty() ? 0 : static_cast<double>(cor) / static_cast<double>(pos.size());
            c.neg_abs = neg.empty() ? 0 : static_cast<double>(nab) / static_cast<double>(neg.size());
            c.obj = 0.5 * (c.rec + c.neg_abs);
            c.met = c.prec >= 0.95;
            if (c.met && (!best.met || c.obj > best.obj)) best = c;
            if (!best.met && c.prec > best_prec) { best_prec = c.prec; fallback = c; }
        }
    }
    return best.met ? best : fallback;
}

// Gate operating point (headline). The 95% precision floor alone is vacuous here (top-1 is ~98% right at any
// threshold), so the dev criterion is the T-perturb gate itself, measured on D: every changed form keeps
// >= kDevRatio of the verbatim answered-and-correct count on the same records, precision >= 0.95, verbatim
// questions are still answered (>= kVerbKeep of the always-answer verbatim hits; the ratio gate alone could be
// met by abstaining on everything), and among the settings that do, abstain on as many leave-one-out dev
// questions as possible.
constexpr double kVerbKeep = 0.95;
constexpr double kDevRatio = 0.96;  // 0.95 gate plus 0.01 slack for sampling noise between D and P

Op select_gate_op(const std::vector<Item>& pos, const std::vector<Item>& neg, int k) {
    static const double tms[] = {0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2};
    std::unordered_map<int, const Item*> verb;
    for (const auto& it : pos)
        if (it.form == kVerb) verb[it.rid] = &it;
    std::vector<const Item*> vp(pos.size(), nullptr);
    for (std::size_t i = 0; i < pos.size(); ++i) vp[i] = verb[pos[i].rid];
    Op best, fallback;
    double best_ratio = -1;
    for (double tm : tms) {
        for (int i = 1; i <= 199; ++i) {
            Op c{0.005 * i, tm};
            std::array<double, kForms> fa{}, fv{};
            std::size_t ans = 0, cor = 0, nab = 0, vall = 0, vans = 0;
            for (std::size_t pi = 0; pi < pos.size(); ++pi) {
                const Item& it = pos[pi];
                const bool an = answers(it, k, c);
                if (an) { ++ans; cor += it.ok[k]; }
                if (it.form == kVerb) { vall += it.ok[k]; vans += an && it.ok[k]; }
                if (it.form != kVerb) {
                    fa[static_cast<std::size_t>(it.form)] += an && it.ok[k];
                    fv[static_cast<std::size_t>(it.form)] += answers(*vp[pi], k, c) && vp[pi]->ok[k];
                }
            }
            for (const auto& it : neg)
                if (!answers(it, k, c)) ++nab;
            double mr = 9;
            for (int f = 1; f < kForms; ++f)
                if (fv[static_cast<std::size_t>(f)] > 0) mr = std::min(mr, fa[static_cast<std::size_t>(f)] / fv[static_cast<std::size_t>(f)]);
            c.min_ratio = mr;
            c.prec = ans ? static_cast<double>(cor) / static_cast<double>(ans) : 1.0;
            c.rec = pos.empty() ? 0 : static_cast<double>(cor) / static_cast<double>(pos.size());
            c.neg_abs = neg.empty() ? 0 : static_cast<double>(nab) / static_cast<double>(neg.size());
            c.obj = c.neg_abs;
            c.vkeep = vall ? static_cast<double>(vans) / static_cast<double>(vall) : 0;
            c.met = mr >= kDevRatio && c.prec >= 0.95 && c.vkeep >= kVerbKeep;
            if (c.met && (!best.met || c.obj > best.obj)) best = c;
            if (!best.met && mr > best_ratio) { best_ratio = mr; fallback = c; }
        }
    }
    return best.met ? best : fallback;
}

double pct(double x) { return 100.0 * x; }

double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))];
}

long rss_kb(const char* key) {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind(key, 0) == 0) return std::atol(line.c_str() + std::strlen(key) + 1);
    return -1;
}

struct Summary {
    Cfg cfg;
    Op op[kScorers];   // G: gate operating point (headline)
    Op opb[kScorers];  // B: balanced operating point
    double dev_obj[kScorers] = {0, 0, 0};
    double min_ratio_abs[kScorers] = {0, 0, 0};
    double min_ratio_all[kScorers] = {0, 0, 0};
    double min_ratio_b[kScorers] = {0, 0, 0};
    double verb_acc_abs[kScorers] = {0, 0, 0};
    double verb_acc_all[kScorers] = {0, 0, 0};
    double h_abstain[kScorers] = {0, 0, 0};
    double h_abstain_b[kScorers] = {0, 0, 0};
    double h_f1_ans[kScorers] = {0, 0, 0};
    double h_f1_gt[kScorers] = {0, 0, 0};
};

struct World {
    std::vector<Rec> recs;
    std::vector<int> indexed, dev, held, held_removed, pset;
    std::vector<int> dev_neg;
    std::size_t leak_removed = 0, ambiguous = 0;
};

// One query against an index: the top-1 result of every scorer, scored against `target` when given.
Item query_item(const std::vector<Rec>& recs, Index& ix, const Cfg& cfg, int irep, const std::string& instr, const std::string& ctx, int exclude,
                const std::string& target, bool with_f1) {
    Item it;
    const QDoc q = make_q(instr, ctx, cfg, irep);
    ix.search(q, exclude, 3, it.r);
    for (int k = 0; k < kScorers; ++k) {
        if (it.r.doc[k] < 0) continue;
        const Rec& got = recs[static_cast<std::size_t>(ix.rec_of[static_cast<std::size_t>(it.r.doc[k])])];
        it.ok[k] = got.resp == target;
        if (with_f1) it.f1[k] = token_f1(got.resp, target);
    }
    return it;
}

// The dev stage, shared by the sweep and the fast path: the five perturbed forms of every D record (positives) and
// the leave-one-out verbatim queries (negatives) pick the operating points G and B of every scorer, on D only.
// Fills S.op, S.opb and S.dev_obj. `ix` must already be built from W.indexed with `cfg`.
struct DevCounts { std::size_t pos = 0, neg = 0; };

DevCounts dev_operating_points(const World& W, const Cfg& cfg, int irep, Index& ix, Summary& S) {
    const auto& recs = W.recs;
    std::vector<Item> dev_pos, dev_neg;
    for (int id : W.dev) {
        const Rec& r = recs[static_cast<std::size_t>(id)];
        for (int f = 0; f < kForms; ++f) {
            std::string i2, c2;
            if (!make_form(static_cast<Form>(f), r, i2, c2)) continue;
            Item it = query_item(recs, ix, cfg, irep, i2, c2, -1, r.resp, false);
            it.form = f;
            it.rid = id;
            dev_pos.push_back(it);
        }
    }
    for (int id : W.dev_neg) {
        const Rec& r = recs[static_cast<std::size_t>(id)];
        dev_neg.push_back(query_item(recs, ix, cfg, irep, r.instr, r.ctx, id, r.resp, false));
    }
    for (int k = 0; k < kScorers; ++k) {
        S.op[k] = select_gate_op(dev_pos, dev_neg, k);
        S.opb[k] = select_op(dev_pos, dev_neg, k);
        S.dev_obj[k] = S.op[k].met ? S.op[k].obj : -1;
    }
    return {dev_pos.size(), dev_neg.size()};
}

// Headline = the (configuration, scorer) with the best dev score; a strictly better score wins, so ties keep the
// earlier configuration and scorer. P and H play no part. Returns false when there is nothing to choose from.
bool pick_headline(const std::vector<Summary>& sums, std::size_t& bi, int& bk) {
    double best = -2;
    bool any = false;
    for (std::size_t i = 0; i < sums.size(); ++i)
        for (int k = 0; k < kScorers; ++k)
            if (sums[i].dev_obj[k] > best) { best = sums[i].dev_obj[k]; bi = i; bk = k; any = true; }
    return any;
}

Summary run_config(const World& W, const Cfg& cfg, int irep) {
    Summary S;
    S.cfg = cfg;
    const auto& recs = W.recs;
    std::printf("\n## Configuration: %s (stem=%d, filler=%d, irep=%d)\n\n", cfg.name.c_str(), cfg.stem, cfg.filler, irep);
    Index ix;
    const auto tb = Clock::now();
    ix.build(recs, W.indexed, cfg, irep);
    const double build_s = secs_since(tb);
    const auto sz = ix.bytes();
    std::size_t resp_bytes = 0;
    for (int id : W.indexed) resp_bytes += recs[static_cast<std::size_t>(id)].resp.size() + 4;
    std::printf("Index: %zu questions, %zu word terms, %zu word postings, %zu trigram keys, %zu trigram postings\n", ix.N,
                ix.vocab.size(), ix.w_doc.size(), ix.t_key.size(), ix.t_doc.size());
    std::printf("Build: %.2f s total (bm25 postings %.2f s, 3-gram postings %.2f s, the rest tokenising)\n", build_s, ix.t_bm25, ix.t_tri);
    std::printf("Size (serialised equivalent): bm25 %.2f MB, 3-gram %.2f MB, doc tables %.2f MB, index %.2f MB, response store %.2f MB\n",
                sz.bm25 / 1e6, sz.tri / 1e6, sz.docs / 1e6, sz.total / 1e6, resp_bytes / 1e6);

    auto run_query = [&](const std::string& instr, const std::string& ctx, int exclude, const std::string& target, bool with_f1) {
        return query_item(recs, ix, cfg, irep, instr, ctx, exclude, target, with_f1);
    };

    // ---- dev set: perturbed forms (positives) and leave-one-out verbatim (negatives)
    const DevCounts dc = dev_operating_points(W, cfg, irep, ix, S);
    std::printf("\n### Dev set D (%zu records, %zu perturbed queries, %zu leave-one-out queries): operating points, chosen on D only\n\n", W.dev.size(), dc.pos, dc.neg);
    std::printf("G = gate point: every changed form keeps >= %.2f of verbatim answered-and-correct on D, precision >= 0.95, >= 0.95 of verbatim hits still answered, then maximise leave-one-out abstention.\n"
                "B = balanced point: precision >= 0.95, then maximise the mean of (D perturbed answered correctly) and (D leave-one-out abstained).\n\n", kDevRatio);
    std::printf("| scorer | point | ts | tm | precision on D perturbed | D perturbed answered correctly | min changed-form ratio on D | verbatim kept on D | D leave-one-out abstained | criterion met |\n|---|---|---|---|---|---|---|---|---|---|\n");
    for (int k = 0; k < kScorers; ++k) {
        std::printf("| %s | G | %.3f | %.3f | %.1f%% | %.1f%% | %.3f | %.1f%% | %.1f%% | %s |\n", kScorerName[k], S.op[k].ts, S.op[k].tm, pct(S.op[k].prec), pct(S.op[k].rec),
                    S.op[k].min_ratio, pct(S.op[k].vkeep), pct(S.op[k].neg_abs), S.op[k].met ? "yes" : "NO");
        std::printf("| %s | B | %.3f | %.3f | %.1f%% | %.1f%% | n/a | n/a | %.1f%% | %s |\n", kScorerName[k], S.opb[k].ts, S.opb[k].tm, pct(S.opb[k].prec), pct(S.opb[k].rec),
                    pct(S.opb[k].neg_abs), S.opb[k].met ? "yes" : "NO");
    }

    // ---- T-perturb
    std::vector<Item> verb(W.pset.size());
    std::vector<std::vector<int>> fpos(kForms);
    std::vector<std::vector<Item>> fit(kForms);
    for (std::size_t pi = 0; pi < W.pset.size(); ++pi) {
        const Rec& r = recs[static_cast<std::size_t>(W.pset[pi])];
        for (int f = 0; f < kForms; ++f) {
            std::string i2, c2;
            if (!make_form(static_cast<Form>(f), r, i2, c2)) continue;
            Item it = run_query(i2, c2, -1, r.resp, false);
            if (f == kVerb) verb[pi] = it;
            fpos[static_cast<std::size_t>(f)].push_back(static_cast<int>(pi));
            fit[static_cast<std::size_t>(f)].push_back(it);
        }
    }
    std::printf("\n### T-perturb (%zu indexed records not in D, %zu with a duplicated question text and different response)\n", W.pset.size(), W.ambiguous);
    std::printf("\nacc = returned the target's response. 'always' = top-1, no abstain. G / B = (answered and correct) / n at the dev operating point G (gate) or B (balanced).\n"
                "ratio = form / verbatim on the same records. Gate: every changed form ratio >= 0.95.\n");
    for (int k = 0; k < kScorers; ++k) {
        std::printf("\n#### %s (G: ts %.3f tm %.3f; B: ts %.3f tm %.3f)\n\n| form | n | acc always | ratio | acc G | ratio | coverage G | precision G | acc B | ratio |\n|---|---|---|---|---|---|---|---|---|---|\n", kScorerName[k],
                    S.op[k].ts, S.op[k].tm, S.opb[k].ts, S.opb[k].tm);
        double min_all = 9, min_abs = 9, min_b = 9;
        std::string worst_all, worst_abs, worst_b;
        for (int f = 0; f < kForms; ++f) {
            const auto& its = fit[static_cast<std::size_t>(f)];
            const auto& pp = fpos[static_cast<std::size_t>(f)];
            const double n = static_cast<double>(its.size());
            double a_all = 0, a_abs = 0, a_b = 0, ans = 0, v_all = 0, v_abs = 0, v_b = 0;
            for (std::size_t i = 0; i < its.size(); ++i) {
                a_all += its[i].ok[k];
                const bool an = answers(its[i], k, S.op[k]);
                ans += an;
                a_abs += an && its[i].ok[k];
                a_b += answers(its[i], k, S.opb[k]) && its[i].ok[k];
                const Item& v = verb[static_cast<std::size_t>(pp[i])];
                v_all += v.ok[k];
                v_abs += answers(v, k, S.op[k]) && v.ok[k];
                v_b += answers(v, k, S.opb[k]) && v.ok[k];
            }
            const double r_all = v_all > 0 ? a_all / v_all : 0, r_abs = v_abs > 0 ? a_abs / v_abs : 0, r_b = v_b > 0 ? a_b / v_b : 0;
            if (f != kVerb) {
                if (r_all < min_all) { min_all = r_all; worst_all = kFormName[f]; }
                if (r_abs < min_abs) { min_abs = r_abs; worst_abs = kFormName[f]; }
                if (r_b < min_b) { min_b = r_b; worst_b = kFormName[f]; }
            } else {
                S.verb_acc_all[k] = a_all / n;
                S.verb_acc_abs[k] = a_abs / n;
            }
            std::printf("| %s | %zu | %.1f%% | %.3f | %.1f%% | %.3f | %.1f%% | %.1f%% | %.1f%% | %.3f |\n", kFormName[f], its.size(), pct(a_all / n), r_all, pct(a_abs / n), r_abs,
                        pct(ans / n), ans > 0 ? pct(a_abs / ans) : 0.0, pct(a_b / n), r_b);
        }
        S.min_ratio_all[k] = min_all;
        S.min_ratio_abs[k] = min_abs;
        S.min_ratio_b[k] = min_b;
        std::printf("\nGate, always answer: %s (min ratio %.3f, %s). Gate at G: %s (%.3f, %s). Gate at B: %s (%.3f, %s).\n", min_all >= 0.95 ? "PASS" : "FAIL", min_all,
                    worst_all.c_str(), min_abs >= 0.95 ? "PASS" : "FAIL", min_abs, worst_abs.c_str(), min_b >= 0.95 ? "PASS" : "FAIL", min_b, worst_b.c_str());
    }

    // ---- held-out H
    std::vector<Item> hit;
    for (int id : W.held) {
        const Rec& r = recs[static_cast<std::size_t>(id)];
        hit.push_back(run_query(r.instr, r.ctx, -1, r.resp, true));
    }
    std::printf("\n### T-QA-dolly held-out H (%zu records after removing %zu near-duplicates)\n\n", hit.size(), W.leak_removed);
    std::printf("Correct behaviour for a new question is to abstain. 'always' = answer every question.\n\n"
                "| scorer | point | abstain rate | answered | mean F1 of answered | frac F1 > 0.5 (answered) | answered with F1 > 0.5 / all H | mean F1 always | frac F1 > 0.5 always |\n|---|---|---|---|---|---|---|---|---|\n");
    for (int k = 0; k < kScorers; ++k) {
        for (int pt = 0; pt < 2; ++pt) {
            const Op& o = pt == 0 ? S.op[k] : S.opb[k];
            double ans = 0, f1a = 0, g5 = 0, f1all = 0, g5all = 0;
            for (const auto& it : hit) {
                f1all += it.f1[k];
                g5all += it.f1[k] > 0.5;
                if (answers(it, k, o)) { ++ans; f1a += it.f1[k]; g5 += it.f1[k] > 0.5; }
            }
            const double n = static_cast<double>(hit.size());
            if (pt == 0) {
                S.h_abstain[k] = 1.0 - ans / n;
                S.h_f1_ans[k] = ans > 0 ? f1a / ans : 0;
                S.h_f1_gt[k] = ans > 0 ? g5 / ans : 0;
            } else {
                S.h_abstain_b[k] = 1.0 - ans / n;
            }
            std::printf("| %s | %s | %.1f%% | %.0f | %.3f | %.1f%% | %.1f%% | %.3f | %.1f%% |\n", kScorerName[k], pt == 0 ? "G" : "B", pct(1.0 - ans / n), ans,
                        ans > 0 ? f1a / ans : 0.0, ans > 0 ? pct(g5 / ans) : 0.0, pct(g5 / n), f1all / n, pct(g5all / n));
        }
    }

    // ---- risk-coverage
    std::vector<const Item*> pool;
    for (int f = 0; f < kForms; ++f)
        for (const auto& it : fit[static_cast<std::size_t>(f)]) pool.push_back(&it);
    std::printf("\n### Risk-coverage on the T-perturb test forms (all %zu form queries pooled; margin gate fixed at each scorer's G tm)\n", pool.size());
    static const double grid[] = {0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99};
    for (int k = 0; k < kScorers; ++k) {
        std::printf("\n#### %s\n\n| ts | coverage | accuracy of answered | H abstain | H answered: mean F1 |\n|---|---|---|---|---|\n", kScorerName[k]);
        std::vector<double> ts_list(std::begin(grid), std::end(grid));
        ts_list.push_back(S.op[k].ts);
        std::sort(ts_list.begin(), ts_list.end());
        for (double ts : ts_list) {
            Op o{ts, S.op[k].tm};
            double ans = 0, cor = 0;
            for (const Item* it : pool)
                if (answers(*it, k, o)) { ++ans; cor += it->ok[k]; }
            double hans = 0, hf = 0;
            for (const auto& it : hit)
                if (answers(it, k, o)) { ++hans; hf += it.f1[k]; }
            std::printf("| %.3f%s | %.1f%% | %.1f%% | %.1f%% | %.3f |\n", ts, ts == S.op[k].ts ? " (G)" : "", pct(ans / static_cast<double>(pool.size())),
                        ans > 0 ? pct(cor / ans) : 100.0, pct(1.0 - hans / static_cast<double>(hit.size())), hans > 0 ? hf / hans : 0.0);
        }
    }

    // ---- speed
    std::vector<QDoc> sq;
    std::vector<bool> has_ctx;
    for (int id : W.pset) { sq.push_back(make_q(recs[static_cast<std::size_t>(id)].instr, recs[static_cast<std::size_t>(id)].ctx, cfg, irep)); has_ctx.push_back(!recs[static_cast<std::size_t>(id)].ctx.empty()); }
    for (int id : W.held) { sq.push_back(make_q(recs[static_cast<std::size_t>(id)].instr, recs[static_cast<std::size_t>(id)].ctx, cfg, irep)); has_ctx.push_back(!recs[static_cast<std::size_t>(id)].ctx.empty()); }
    std::printf("\n### Speed (%zu queries: T-perturb verbatim + H; single thread; latency includes scoring all %zu stored questions)\n\n", sq.size(), ix.N);
    std::printf("| scorer | p50 all | p99 all | p50 instruction-only | p99 instruction-only | p50 with context | p99 with context |\n|---|---|---|---|---|---|---|\n");
    const int masks[kScorers] = {1, 2, 3};
    for (int k = 0; k < kScorers; ++k) {
        std::vector<double> all, noctx, withctx;
        Res res;
        for (std::size_t i = 0; i < sq.size(); ++i) {
            const auto t0 = Clock::now();
            ix.search(sq[i], -1, masks[k], res);
            const double ms = secs_since(t0) * 1e3;
            all.push_back(ms);
            (has_ctx[i] ? withctx : noctx).push_back(ms);
        }
        std::printf("| %s | %.2f ms | %.2f ms | %.2f ms | %.2f ms | %.2f ms | %.2f ms |\n", kScorerName[k], quantile(all, 0.5), quantile(all, 0.99),
                    quantile(noctx, 0.5), quantile(noctx, 0.99), quantile(withctx, 0.5), quantile(withctx, 0.99));
    }
    std::printf("\nProcess memory so far: peak RSS %.1f MB, current %.1f MB (includes the loaded Dolly records and all earlier configurations).\n", rss_kb("VmHWM:") / 1024.0,
                rss_kb("VmRSS:") / 1024.0);
    return S;
}

// ------------------------------------------------------------ the fast path

// Everything the fast path needs from the sweep is the headline: which configuration, which scorer, and the gate
// point (ts, tm) on D. It is written next to the data file by the sweep and by the one-off calibration, and read
// back by --ask / --ask-file, so a question costs one index build (about 1 s) instead of the 52 s sweep.
struct Headline {
    Cfg cfg;
    int scorer = 2;
    Op op;  // gate point G
};

struct Fingerprint {
    std::uint64_t bytes = 0, records = 0, indexed = 0;
    int irep = 1;
    std::string configs;
    bool operator==(const Fingerprint& o) const {
        return bytes == o.bytes && records == o.records && indexed == o.indexed && irep == o.irep && configs == o.configs;
    }
};

std::uint64_t file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    return in ? static_cast<std::uint64_t>(in.tellg()) : 0;
}

std::vector<Cfg> parse_configs(const std::string& configs) {
    std::vector<Cfg> cfgs;
    std::stringstream ss(configs);
    for (std::string tok; std::getline(ss, tok, ',');) {
        if (tok == "plain") cfgs.push_back({false, false, "plain"});
        else if (tok == "stem") cfgs.push_back({true, false, "+stem"});
        else if (tok == "filler") cfgs.push_back({false, true, "+filler"});
        else if (tok == "both") cfgs.push_back({true, true, "+stem +filler"});
    }
    return cfgs;
}

Fingerprint fingerprint_of(const World& W, const std::string& path, int irep, const std::string& configs) {
    Fingerprint f;
    f.bytes = file_bytes(path);
    f.records = W.recs.size();
    f.indexed = W.indexed.size();
    f.irep = irep;
    f.configs = configs;
    return f;
}

// %.17g round-trips a double exactly, so the stored ts and tm are bit-identical to the sweep's.
bool write_headline(const std::string& path, const Fingerprint& f, const Headline& h) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out) return false;
        char buf[64];
        out << "chat_index_op 1\n";
        out << "dolly_bytes " << f.bytes << "\ndolly_records " << f.records << "\nindexed " << f.indexed << "\nirep " << f.irep << "\nconfigs " << f.configs << "\n";
        out << "stem " << h.cfg.stem << "\nfiller " << h.cfg.filler << "\nname " << h.cfg.name << "\nscorer " << h.scorer << "\n";
        std::snprintf(buf, sizeof buf, "%.17g", h.op.ts);
        out << "ts " << buf << "\n";
        std::snprintf(buf, sizeof buf, "%.17g", h.op.tm);
        out << "tm " << buf << "\n";
        if (!out) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

// Returns false (and says why) when the file is missing, malformed, or was written for other data or settings.
bool read_headline(const std::string& path, const Fingerprint& want, Headline& h, std::string& why) {
    std::ifstream in(path);
    if (!in) { why = "no thresholds file"; return false; }
    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        const auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        kv[line.substr(0, sp)] = line.substr(sp + 1);
    }
    for (const char* key : {"chat_index_op", "dolly_bytes", "dolly_records", "indexed", "irep", "configs", "stem", "filler", "name", "scorer", "ts", "tm"})
        if (!kv.count(key)) { why = std::string("thresholds file lacks ") + key; return false; }
    if (kv["chat_index_op"] != "1") { why = "thresholds file version"; return false; }
    Fingerprint got;
    got.bytes = std::strtoull(kv["dolly_bytes"].c_str(), nullptr, 10);
    got.records = std::strtoull(kv["dolly_records"].c_str(), nullptr, 10);
    got.indexed = std::strtoull(kv["indexed"].c_str(), nullptr, 10);
    got.irep = std::atoi(kv["irep"].c_str());
    got.configs = kv["configs"];
    if (!(got == want)) { why = "thresholds file is for other data or settings (size, records, irep or --configs differ)"; return false; }
    h.cfg.stem = kv["stem"] == "1";
    h.cfg.filler = kv["filler"] == "1";
    h.cfg.name = kv["name"];
    h.scorer = std::atoi(kv["scorer"].c_str());
    h.op.ts = std::strtod(kv["ts"].c_str(), nullptr);
    h.op.tm = std::strtod(kv["tm"].c_str(), nullptr);
    if (h.scorer < 0 || h.scorer >= kScorers) { why = "thresholds file scorer"; return false; }
    return true;
}

// The one-off calibration: only the dev stage of every configuration (no T-perturb, H, risk-coverage or speed).
// Same code as the sweep's dev stage, so the headline is identical to the sweep's.
bool calibrate(const World& W, const std::vector<Cfg>& cfgs, int irep, Headline& h) {
    std::vector<Summary> sums;
    for (const auto& cfg : cfgs) {
        Index ix;
        ix.build(W.recs, W.indexed, cfg, irep);
        Summary S;
        S.cfg = cfg;
        dev_operating_points(W, cfg, irep, ix, S);
        sums.push_back(S);
    }
    std::size_t bi = 0;
    int bk = 2;
    if (!pick_headline(sums, bi, bk)) return false;
    h.cfg = sums[bi].cfg;
    h.scorer = bk;
    h.op = sums[bi].op[bk];
    return true;
}

// ------------------------------------------------------------ answer or abstain, with a reason and a card

// Reason codes (docs/DISCOVERY.md 3.1) the retrieval index can give:
//   NO_EVIDENCE         the best stored question scores below ts: nothing stored is close to this question.
//   AMBIGUOUS_QUESTION  the best scores >= ts, but a stored question with a different text is within tm of it.
// When both tests fail, NO_EVIDENCE wins: a margin between two weak matches says nothing about ambiguity
// (3.1 defines the code as "two or more fit the question"). Both flags are kept in the machine line.
struct Verdict {
    bool answer = false;
    const char* reason = nullptr;
    double score = 0, runner = 0, margin = 0;
    int doc = -1, doc2 = -1;
    bool score_low = false, margin_low = false;
};

Verdict decide(const Res& r, int k, const Op& op) {
    Verdict v;
    v.doc = r.doc[k];
    v.doc2 = r.doc2[k];
    v.score = r.s1[k];
    v.runner = r.s2[k];
    v.margin = r.s1[k] - r.s2[k];
    v.score_low = !(r.s1[k] >= op.ts);
    v.margin_low = !((r.s1[k] - r.s2[k]) >= op.tm);  // the exact expressions answers() uses
    v.answer = !v.score_low && !v.margin_low;
    if (!v.answer) v.reason = (v.score_low || v.doc < 0) ? "NO_EVIDENCE" : "AMBIGUOUS_QUESTION";
    return v;
}

double r6(double x) { return std::round(x * 1e6) / 1e6; }

std::string with_ctx(const Rec& r) { return r.ctx.empty() ? r.instr : r.instr + "  [+ context]"; }

void print_card(const World& W, const Index& ix, const Headline& h, int n, const std::string& q, const Verdict& v) {
    const auto rec_of = [&](int doc) -> const Rec& { return W.recs[static_cast<std::size_t>(ix.rec_of[static_cast<std::size_t>(doc)])]; };
    const auto rec_no = [&](int doc) { return ix.rec_of[static_cast<std::size_t>(doc)]; };
    std::printf("\nQ%d: %s\n", n, q.c_str());
    if (v.answer) {
        const Rec& m = rec_of(v.doc);
        std::printf("%s\n", m.resp.c_str());
        std::printf("[RETRIEVED · score %.3f (needs >= %.3f) · margin %.3f (needs >= %.3f)]\n", v.score, h.op.ts, v.margin, h.op.tm);
        std::printf("Matched stored question: \"%s\" (record %d · category %s)\n", with_ctx(m).c_str(), rec_no(v.doc), m.cat.c_str());
        std::printf("Because: the closest of %zu stored Dolly questions. This is retrieval, not understanding; the answer is the stored one.\n", ix.N);
        return;
    }
    std::printf("I don't know.                    [%s]\n", v.reason);
    if (v.doc < 0) { std::printf("Why: the index is empty.\n"); return; }
    const Rec& m = rec_of(v.doc);
    if (v.score_low)
        std::printf("Why: the closest stored question scores %.3f, below the %.3f needed to answer. Nothing stored matches it closely enough to be sure it is the same question.\n", v.score, h.op.ts);
    else
        std::printf("Why: two stored questions with different texts match about equally (%.3f and %.3f, margin %.3f, under the %.3f needed), so I cannot tell which one is meant.\n",
                    v.score, v.runner, v.margin, h.op.tm);
    std::printf("Nearest stored question: \"%s\" (record %d · category %s · score %.3f)\n", with_ctx(m).c_str(), rec_no(v.doc), m.cat.c_str(), v.score);
    if (!v.score_low && v.doc2 >= 0) {
        const Rec& m2 = rec_of(v.doc2);
        std::printf("Runner-up: \"%s\" (record %d · category %s · score %.3f)\n", with_ctx(m2).c_str(), rec_no(v.doc2), m2.cat.c_str(), v.runner);
    }
    std::printf("What would let me answer: a stored question that matches this one more closely%s.\n", v.score_low ? "" : ", or one clearly closer than the runner-up");
}

// One machine-readable line per question: "RESULT_JSON " + one JSON object.
std::string result_json(const World& W, const Index& ix, const Headline& h, int n, const std::string& q, const Verdict& v) {
    nlohmann::json j;
    j["n"] = n;
    j["question"] = q;
    j["decision"] = v.answer ? "answer" : "abstain";
    j["reason"] = v.answer ? nlohmann::json(nullptr) : nlohmann::json(v.reason);
    j["score"] = r6(v.score);
    j["margin"] = r6(v.margin);
    j["runner_up_score"] = r6(v.runner);
    j["ts"] = h.op.ts;
    j["tm"] = h.op.tm;
    j["score_below_ts"] = v.score_low;
    j["margin_below_tm"] = v.margin_low;
    j["scorer"] = kScorerName[h.scorer];
    j["config"] = h.cfg.name;
    const auto put = [&](const char* pfx, int doc) {
        const std::string p = pfx;
        if (doc < 0) { j[p + "question"] = nullptr; j[p + "record"] = nullptr; j[p + "category"] = nullptr; return; }
        const int rn = ix.rec_of[static_cast<std::size_t>(doc)];
        const Rec& m = W.recs[static_cast<std::size_t>(rn)];
        j[p + "question"] = m.instr;
        j[p + "has_context"] = !m.ctx.empty();
        j[p + "record"] = rn;
        j[p + "category"] = m.cat;
    };
    put("nearest_", v.doc);
    put("runner_up_", v.doc2);
    j["answer"] = (v.answer && v.doc >= 0) ? nlohmann::json(W.recs[static_cast<std::size_t>(ix.rec_of[static_cast<std::size_t>(v.doc)])].resp) : nlohmann::json(nullptr);
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

// Questions: the first tab-separated column of each line. A first line whose first cell is "question" is a header.
std::vector<std::string> read_questions(const std::string& path) {
    std::vector<std::string> qs;
    std::ifstream in(path);
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string cell = line.substr(0, line.find('\t'));
        const bool header = first && cell == "question";
        first = false;
        if (cell.empty() || header) continue;
        qs.push_back(cell);
    }
    return qs;
}

}  // namespace

namespace {

// Loads Dolly and lays out the protocol sets (H, D, P) exactly as the sweep always has.
bool build_world(const std::string& path, std::uint64_t seed, World& W) {
    W.recs = load_dolly(path);
    if (W.recs.empty()) return false;
    {
        std::unordered_map<std::string, int> keys;
        for (auto& r : W.recs) {
            r.ikey = norm(r.instr);
            auto it = keys.emplace(norm(r.instr + " " + r.ctx), static_cast<int>(keys.size())).first;
            r.keyid = it->second;
        }
    }
    const int R = static_cast<int>(W.recs.size());
    for (int i = 0; i < R; ++i) {
        if (i % 10 == 3) W.held_removed.push_back(i);  // temporarily: all of H
        else W.indexed.push_back(i);
        if (i % 10 == 7) W.dev.push_back(i);
    }
    std::unordered_map<std::string, int> ikey_count;
    for (int i : W.indexed) ++ikey_count[W.recs[static_cast<std::size_t>(i)].ikey];
    std::vector<int> removed;
    for (int i : W.held_removed) {
        if (ikey_count.count(W.recs[static_cast<std::size_t>(i)].ikey)) removed.push_back(i);
        else W.held.push_back(i);
    }
    W.leak_removed = removed.size();
    for (int i : W.dev)
        if (ikey_count[W.recs[static_cast<std::size_t>(i)].ikey] == 1) W.dev_neg.push_back(i);  // only itself has this instruction
    {
        std::vector<int> cand;
        for (int i : W.indexed)
            if (i % 10 != 7) cand.push_back(i);
        std::mt19937_64 rng(seed);
        for (std::size_t i = cand.size(); i > 1; --i) std::swap(cand[i - 1], cand[static_cast<std::size_t>(rng() % i)]);
        cand.resize(std::min<std::size_t>(1000, cand.size()));
        std::sort(cand.begin(), cand.end());
        W.pset = cand;
        std::unordered_map<int, std::set<std::string>> by_key;
        for (int i : W.indexed) by_key[W.recs[static_cast<std::size_t>(i)].keyid].insert(W.recs[static_cast<std::size_t>(i)].resp);
        for (int i : W.pset)
            if (by_key[W.recs[static_cast<std::size_t>(i)].keyid].size() > 1) ++W.ambiguous;
    }
    return true;
}

void answer_questions(const World& W, const Headline& h, int irep, const std::vector<std::string>& qs, bool quiet, double* build_s, double* ask_s) {
    Index ix;
    auto t0 = Clock::now();
    ix.build(W.recs, W.indexed, h.cfg, irep);
    *build_s = secs_since(t0);
    t0 = Clock::now();
    int n = 0;
    for (const auto& q : qs) {
        Res r;
        ix.search(make_q(q, "", h.cfg, irep), -1, 3, r);
        const Verdict v = decide(r, h.scorer, h.op);
        ++n;
        if (!quiet) print_card(W, ix, h, n, q, v);
        std::printf("RESULT_JSON %s\n", result_json(W, ix, h, n, q, v).c_str());
    }
    *ask_s = secs_since(t0);
}

}  // namespace

int main(int argc, char** argv) {
    const auto t_start = Clock::now();
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s dolly.jsonl [--ask \"question\"] [--ask-file questions.tsv] [--quiet] [--recalibrate] [--thresholds PATH]\n"
                     "       %s dolly.jsonl [--sweep] [--seed N] [--irep N] [--configs plain,stem,filler,both] [--ask ...]\n",
                     argv[0], argv[0]);
        return 2;
    }
    std::string path = argv[1], configs = "plain,stem,filler,both", ask, ask_file, op_path;
    std::uint64_t seed = 20261027;
    int irep = 1;
    bool sweep = false, quiet = false, recalibrate = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--sweep") { sweep = true; continue; }
        if (a == "--quiet") { quiet = true; continue; }
        if (a == "--recalibrate") { recalibrate = true; continue; }
        if (i + 1 >= argc) { std::fprintf(stderr, "option %s needs a value\n", a.c_str()); return 2; }
        const char* val = argv[++i];
        if (a == "--seed") seed = std::strtoull(val, nullptr, 10);
        else if (a == "--irep") irep = std::atoi(val);
        else if (a == "--configs") configs = val;
        else if (a == "--ask") ask = val;
        else if (a == "--ask-file") ask_file = val;
        else if (a == "--thresholds") op_path = val;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (op_path.empty()) op_path = path + ".chat_index_op";
    const bool asking = !ask.empty() || !ask_file.empty();
    // --ask / --ask-file alone take the fast path; the full sweep runs only for a bare run or with --sweep.
    const bool fast = asking && !sweep;

    std::vector<std::string> qs;
    if (!ask.empty()) qs.push_back(ask);
    if (!ask_file.empty()) {
        const auto more = read_questions(ask_file);
        if (more.empty()) { std::fprintf(stderr, "no questions in %s\n", ask_file.c_str()); return 1; }
        qs.insert(qs.end(), more.begin(), more.end());
    }

    World W;
    if (!build_world(path, seed, W)) { std::fprintf(stderr, "no records in %s\n", path.c_str()); return 1; }
    const double load_s = secs_since(t_start);
    const std::vector<Cfg> cfgs = parse_configs(configs);
    const Fingerprint fp = fingerprint_of(W, path, irep, configs);
    const int R = static_cast<int>(W.recs.size());

    if (fast) {
        Headline h;
        std::string why, source = "file";
        double cal_s = 0;
        if (recalibrate || !read_headline(op_path, fp, h, why)) {
            if (recalibrate) why = "--recalibrate";
            std::fprintf(quiet ? stderr : stdout, "# calibrating the operating point (%s); dev stage only, one-off\n", why.c_str());
            const auto tc = Clock::now();
            if (!calibrate(W, cfgs, irep, h)) { std::fprintf(stderr, "calibration failed\n"); return 1; }
            cal_s = secs_since(tc);
            source = "calibrated";
            if (!write_headline(op_path, fp, h)) std::fprintf(stderr, "warning: could not write %s\n", op_path.c_str());
        }
        double build_s = 0, ask_s = 0;
        answer_questions(W, h, irep, qs, quiet, &build_s, &ask_s);
        std::fprintf(quiet ? stderr : stdout, "\n# fast path: thresholds %s (%s) -> config %s, scorer %s, ts %.3f tm %.3f. load %.2f s, calibrate %.2f s, index build %.2f s, %zu question(s) in %.3f s, total %.2f s\n",
                     source.c_str(), op_path.c_str(), h.cfg.name.c_str(), kScorerName[h.scorer], h.op.ts, h.op.tm, load_s, cal_s, build_s, qs.size(), ask_s, secs_since(t_start));
        return 0;
    }

    std::printf("# chat_index run (seed %llu, irep %d)\n\n", static_cast<unsigned long long>(seed), irep);
    std::printf("Records %d. Indexed %zu (train %zu + D %zu). Held-out H candidates %zu; removed as near-duplicates of an indexed instruction: %zu; H used: %zu.\n",
                R, W.indexed.size(), W.indexed.size() - W.dev.size(), W.dev.size(), W.held_removed.size(), W.leak_removed, W.held.size());
    std::printf("Dev negatives (D leave-one-out, instruction unique in the index): %zu. T-perturb sample: %zu.\n", W.dev_neg.size(), W.pset.size());
    {
        std::set<std::string> cats;
        for (const auto& r : W.recs) cats.insert(r.cat);
        std::printf("\n| category | all | H used | H share |\n|---|---|---|---|\n");
        for (const auto& c : cats) {
            std::size_t a = 0, h = 0;
            for (const auto& r : W.recs) a += r.cat == c;
            for (int i : W.held) h += W.recs[static_cast<std::size_t>(i)].cat == c;
            std::printf("| %s | %zu | %zu | %.1f%% |\n", c.c_str(), a, h, pct(static_cast<double>(h) / static_cast<double>(a)));
        }
    }
    std::printf("\nSynonym table: %zu pairs (both directions).\n", syn_pairs().size());
    {
        std::array<std::size_t, kForms> app{};
        for (int id : W.pset)
            for (int f = 0; f < kForms; ++f) {
                std::string i2, c2;
                app[static_cast<std::size_t>(f)] += make_form(static_cast<Form>(f), W.recs[static_cast<std::size_t>(id)], i2, c2);
            }
        std::printf("T-perturb forms that apply to the %zu sampled records:", W.pset.size());
        for (int f = 0; f < kForms; ++f) std::printf(" %s %zu%s", kFormName[f], app[static_cast<std::size_t>(f)], f + 1 < kForms ? "," : "\n");
    }

    std::vector<Summary> sums;
    for (const auto& c : cfgs) sums.push_back(run_config(W, c, irep));

    std::printf("\n## Summary across configurations (dev score picks the headline; P and H play no part in the choice)\n\n");
    std::printf("| config | scorer | dev score (D leave-one-out abstain at G) | verbatim acc at G | min changed ratio always | min ratio at G | min ratio at B | H abstain at G | H abstain at B | H answered mean F1 at G |\n|---|---|---|---|---|---|---|---|---|---|\n");
    for (std::size_t i = 0; i < sums.size(); ++i)
        for (int k = 0; k < kScorers; ++k) {
            const auto& s = sums[i];
            std::printf("| %s | %s | %.3f | %.1f%% | %.3f | %.3f | %.3f | %.1f%% | %.1f%% | %.3f |\n", s.cfg.name.c_str(), kScorerName[k], s.dev_obj[k], pct(s.verb_acc_abs[k]),
                        s.min_ratio_all[k], s.min_ratio_abs[k], s.min_ratio_b[k], pct(s.h_abstain[k]), pct(s.h_abstain_b[k]), s.h_f1_ans[k]);
        }
    std::size_t bi = 0;
    int bk = 2;
    const bool have_headline = pick_headline(sums, bi, bk);
    if (!sums.empty()) std::printf("\nHeadline by dev score: %s / %s.\n", sums[bi].cfg.name.c_str(), kScorerName[bk]);
    if (have_headline) {
        Headline h;
        h.cfg = sums[bi].cfg;
        h.scorer = bk;
        h.op = sums[bi].op[bk];
        if (write_headline(op_path, fp, h)) std::printf("Headline operating point (G: ts %.3f tm %.3f) written to %s for --ask / --ask-file.\n", h.op.ts, h.op.tm, op_path.c_str());
        else std::fprintf(stderr, "warning: could not write %s\n", op_path.c_str());
        if (asking) {
            double build_s = 0, ask_s = 0;
            answer_questions(W, h, irep, qs, quiet, &build_s, &ask_s);
        }
    }
    return 0;
}
