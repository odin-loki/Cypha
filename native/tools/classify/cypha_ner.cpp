// E2: an entity recogniser whose decisions the Cypha classifier makes (docs/reports/E2_ENTITIES.md).
//
// BIO tagging over six types (PERSON, PLACE, ORG, DATE, NUMBER, MISC; 13 tags), with E1's machinery
// (native/tools/classify/cypha_tagger.cpp, docs/reports/E1_CYPHA_TAGGER.md): every token's input is dense count
// statistics, log P(tag | f) for each feature f (form, lowercase form, suffixes, prefix, shape, neighbours, the tags
// before it, the shapes around it), smoothed toward the tag prior; plus scalars and a word vector. On the training
// split the count features are cross-fitted (a sentence's features come from the other four of five folds). Every
// dimension is standardised, then whitened by the pooled within-class covariance (E1's amendment 1). Training is
// Cypha::update one token at a time (W_enc frozen, identity); tagging is greedy left to right with Cypha::predict.
//
// The data are silver sentences from Wikipedia link anchors (scripts/e2_silver.py): "title \t text \t spans", the
// spans "start:end:TYPE;..." in UTF-8 byte offsets of the text.
//
// Usage:
//   cypha_ner run --train T.tsv --dev D.tsv [--test X.tsv --read-test] [--hand H.tsv] [--alpha A] [--whiten EPS]
//                 [--simvec FILE] [--delta-lr X] [--world-lr X] [--passes P] [--cypha-sents N] [--seed S] [--diag] [--o-keep P]
//                 [--prompts GEN.tsv --prompt-out SPANS.tsv] [--pred-out PREFIX]
//   cypha_ner selftest
// --o-keep P (amendment 1): Cypha learns from each O token with probability P (seeded); entity tokens always.
// --passes 0 runs only the diagnostics (counts alone, exact class means). Without --read-test the test split is not
// opened. --prompts tags the prompt column (3rd) of a GEN-A prompts file and writes one row per span:
// id, byte start, byte end, type, surface (in the prompt's own bytes, after TSV unescaping).
//
// Author: Odin Loch

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "cypha/classify/create_model.hpp"
#include "cypha/classify/load_cypha.hpp"
#include "cypha/cypha.hpp"

namespace {

using u64 = std::uint64_t;
using Clock = std::chrono::steady_clock;
constexpr int K = 13;
constexpr int NT = 6;  // entity types
const std::array<const char*, K> kTags = {"O",      "B-PERSON", "I-PERSON", "B-PLACE",  "I-PLACE", "B-ORG", "I-ORG",
                                          "B-DATE", "I-DATE",   "B-NUMBER", "I-NUMBER", "B-MISC",  "I-MISC"};
const std::array<const char*, NT> kTypes = {"PERSON", "PLACE", "ORG", "DATE", "NUMBER", "MISC"};
constexpr int kCoreTypes = 5;  // the gate's types: all but MISC

int type_id(const std::string& t) {
    for (int k = 0; k < NT; ++k)
        if (t == kTypes[static_cast<std::size_t>(k)]) return k;
    return -1;
}
int tag_id(const std::string& t) {
    for (int k = 0; k < K; ++k)
        if (t == kTags[static_cast<std::size_t>(k)]) return k;
    return 0;
}
int b_tag(int type) { return 1 + 2 * type; }
int i_tag(int type) { return 2 + 2 * type; }
int tag_type(int tag) { return tag == 0 ? -1 : (tag - 1) / 2; }
bool tag_is_b(int tag) { return tag > 0 && (tag - 1) % 2 == 0; }

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// ---------------------------------------------------------------- tokens
bool wordc(unsigned char c) { return std::isalnum(c) || c >= 0x80; }
const std::unordered_set<std::string>& abbrev() {
    static const std::unordered_set<std::string> A = {"Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr",  "Prof", "Gen", "Col", "Lt", "Sgt",
                                                      "Capt", "Rev", "Mt", "Ft", "vs", "etc", "Inc", "Ltd", "Co",  "Corp", "No"};
    return A;
}
// Words: runs of letters, digits and UTF-8 bytes, joined by - ' . & (and , between digits) when a word character follows.
// A final period stays on an abbreviation ("U.S.", "J.", "Dr."). A possessive "'s" is its own token. Any other
// non-space byte is a token of its own.
void tokenize(const std::string& s, std::vector<std::string>& w, std::vector<std::pair<std::size_t, std::size_t>>& off) {
    w.clear(), off.clear();
    const std::size_t n = s.size();
    std::size_t i = 0;
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ++i;
            continue;
        }
        if (!wordc(c)) {
            w.push_back(s.substr(i, 1)), off.push_back({i, i + 1});
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        while (j < n) {
            const unsigned char d = static_cast<unsigned char>(s[j]);
            if (wordc(d)) {
                ++j;
                continue;
            }
            const bool joiner = d == '-' || d == '\'' || d == '.' || d == '&' ||
                                (d == ',' && std::isdigit(static_cast<unsigned char>(s[j - 1])) && j + 1 < n && std::isdigit(static_cast<unsigned char>(s[j + 1])));
            if (joiner && j + 1 < n && wordc(static_cast<unsigned char>(s[j + 1]))) {
                j += 2;
                continue;
            }
            break;
        }
        std::string tok = s.substr(i, j - i);
        if (j < n && s[j] == '.') {
            const bool ab = tok.find('.') != std::string::npos || (tok.size() == 1 && std::isupper(static_cast<unsigned char>(tok[0]))) ||
                            abbrev().count(tok) > 0;
            if (ab) tok += '.', ++j;
        }
        if (tok.size() > 2 && tok[tok.size() - 2] == '\'' && (tok.back() == 's' || tok.back() == 'S')) {
            w.push_back(tok.substr(0, tok.size() - 2)), off.push_back({i, j - 2});
            w.push_back(tok.substr(tok.size() - 2)), off.push_back({j - 2, j});
        } else {
            w.push_back(tok), off.push_back({i, j});
        }
        i = j;
    }
}

// ---------------------------------------------------------------- data
struct Span {
    int a = 0, b = 0, type = 0;  // tokens [a, b)
    bool operator<(const Span& o) const { return a != o.a ? a < o.a : b != o.b ? b < o.b : type < o.type; }
    bool operator==(const Span& o) const { return a == o.a && b == o.b && type == o.type; }
};
struct Sent {
    std::string title, text;
    std::size_t base = 0;  // byte offset of text inside its prompt (prompts only)
    std::string id;        // prompt id (prompts only)
    std::vector<std::string> w;
    std::vector<std::pair<std::size_t, std::size_t>> off;
    std::vector<int> t;  // gold tags (0 when none)
};

// BIO tags -> spans; an I- after O or after another type opens a span (the usual repair).
std::vector<Span> spans_of(const std::vector<int>& t) {
    std::vector<Span> out;
    int cur = -1, a = 0;
    for (int i = 0; i <= static_cast<int>(t.size()); ++i) {
        const int tag = i < static_cast<int>(t.size()) ? t[static_cast<std::size_t>(i)] : 0;
        const int ty = tag_type(tag);
        const bool cont = ty >= 0 && ty == cur && !tag_is_b(tag);
        if (!cont && cur >= 0) out.push_back({a, i, cur}), cur = -1;
        if (ty >= 0 && !cont) cur = ty, a = i;
    }
    return out;
}

// Tags from byte spans: tokens overlapping a span take its type (B on the first); a later overlapping span is skipped.
void tags_from_bytes(Sent& s, const std::vector<std::array<std::size_t, 3>>& bs) {
    s.t.assign(s.w.size(), 0);
    for (const auto& [x, y, ty] : bs) {
        std::vector<std::size_t> in;
        for (std::size_t i = 0; i < s.w.size(); ++i)
            if (s.off[i].first < y && s.off[i].second > x) in.push_back(i);
        if (in.empty()) continue;
        bool clash = false;
        for (auto i : in) clash = clash || s.t[i] != 0;
        if (clash) continue;
        for (std::size_t k = 0; k < in.size(); ++k) s.t[in[k]] = k == 0 ? b_tag(static_cast<int>(ty)) : i_tag(static_cast<int>(ty));
    }
}

std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> f;
    std::size_t a = 0;
    for (;;) {
        const std::size_t b = s.find(d, a);
        f.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) return f;
        a = b + 1;
    }
}

bool read_silver(const std::string& path, std::vector<Sent>& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && line.back() == '\r') line.pop_back();
        const auto fld = split(line, '\t');
        if (fld.size() < 2) continue;
        Sent s;
        s.title = fld[0], s.text = fld[1];
        tokenize(s.text, s.w, s.off);
        if (s.w.empty()) continue;
        std::vector<std::array<std::size_t, 3>> bs;
        if (fld.size() > 2 && !fld[2].empty())
            for (const auto& sp : split(fld[2], ';')) {
                const auto p = split(sp, ':');
                if (p.size() != 3) continue;
                const int ty = type_id(p[2]);
                if (ty < 0) continue;
                bs.push_back({std::strtoull(p[0].c_str(), nullptr, 10), std::strtoull(p[1].c_str(), nullptr, 10), static_cast<std::size_t>(ty)});
            }
        tags_from_bytes(s, bs);
        out.push_back(std::move(s));
    }
    return true;
}

std::string unescape_tsv(const std::string& s) {
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            o += c == 't' ? '\t' : c == 'n' ? '\n' : c;
        } else o += s[i];
    }
    return o;
}

// A prompt's sentences: tokens of the whole prompt, cut after . ! ? when the next token starts with a capital, a
// digit or a quote (the same rule as the silver sentences).
void prompt_sents(const std::string& id, const std::string& p, std::vector<Sent>& out) {
    std::vector<std::string> w;
    std::vector<std::pair<std::size_t, std::size_t>> off;
    tokenize(p, w, off);
    std::size_t a = 0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        const bool end = (w[i] == "." || w[i] == "!" || w[i] == "?") &&
                         (i + 1 == w.size() || std::isupper(static_cast<unsigned char>(w[i + 1][0])) || std::isdigit(static_cast<unsigned char>(w[i + 1][0])) ||
                          w[i + 1] == "\"" || w[i + 1] == "'");
        if (!end && i + 1 < w.size()) continue;
        Sent s;
        s.id = id;
        s.base = off[a].first;
        s.text = p.substr(off[a].first, off[i].second - off[a].first);
        tokenize(s.text, s.w, s.off);
        s.t.assign(s.w.size(), 0);
        if (!s.w.empty()) out.push_back(std::move(s));
        a = i + 1;
    }
}

// ---------------------------------------------------------------- word helpers (as E1)
std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}
std::vector<std::size_t> cp_starts(const std::string& s) {
    std::vector<std::size_t> st;
    for (std::size_t i = 0; i < s.size(); ++i)
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) st.push_back(i);
    return st;
}
std::string suffix_cp(const std::string& s, int k) {
    const auto st = cp_starts(s);
    if (static_cast<int>(st.size()) <= k) return s;
    return s.substr(st[st.size() - static_cast<std::size_t>(k)]);
}
std::string prefix_cp(const std::string& s, int k) {
    const auto st = cp_starts(s);
    if (static_cast<int>(st.size()) <= k) return s;
    return s.substr(0, st[static_cast<std::size_t>(k)]);
}
std::string shape(const std::string& s) {
    std::string r;
    char last = 0;
    int run = 0;
    for (unsigned char c : s) {
        char m;
        if (c >= 'A' && c <= 'Z') m = 'X';
        else if (c >= 'a' && c <= 'z') m = 'x';
        else if (c >= '0' && c <= '9') m = 'd';
        else if (c >= 0x80) m = 'u';
        else m = static_cast<char>(c);
        if (m == last) {
            if (++run > 2) continue;
        } else {
            run = 1;
            last = m;
        }
        r.push_back(m);
    }
    return r;
}
bool has_digit(const std::string& s) {
    for (unsigned char c : s)
        if (c >= '0' && c <= '9') return true;
    return false;
}
bool init_cap(const std::string& s) { return !s.empty() && s[0] >= 'A' && s[0] <= 'Z'; }
bool all_caps(const std::string& s) {
    bool any = false;
    for (unsigned char c : s) {
        if (c >= 'a' && c <= 'z') return false;
        if (c >= 'A' && c <= 'Z') any = true;
    }
    return any;
}
bool is_punct(const std::string& s) { return s.size() == 1 && !wordc(static_cast<unsigned char>(s[0])); }

u64 fnv(const std::string& s, u64 h = 1469598103934665603ULL) {
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}
u64 key(int block, const std::string& s) { return fnv(s, 1469598103934665603ULL ^ (static_cast<u64>(block + 1) * 0x9E3779B97F4A7C15ULL)); }

// ---------------------------------------------------------------- count features
constexpr int kBlocks = 15;  // B1..B15 (B11, B12: previous tags)
constexpr int kScalars = 10;
const char* kStart = "<s>";
const char* kEnd = "</s>";

struct TokFeat {
    std::array<u64, kBlocks> k{};
};

void static_keys(const Sent& s, std::size_t i, TokFeat& f) {
    const std::string& w = s.w[i];
    const std::string lw = lower(w);
    const std::string pw = i > 0 ? lower(s.w[i - 1]) : kStart;
    const std::string nw = i + 1 < s.w.size() ? lower(s.w[i + 1]) : kEnd;
    const std::string ps = i > 0 ? shape(s.w[i - 1]) : kStart;
    const std::string ns = i + 1 < s.w.size() ? shape(s.w[i + 1]) : kEnd;
    f.k[0] = key(0, w);
    f.k[1] = key(1, lw);
    f.k[2] = key(2, suffix_cp(lw, 1));
    f.k[3] = key(3, suffix_cp(lw, 2));
    f.k[4] = key(4, suffix_cp(lw, 3));
    f.k[5] = key(5, suffix_cp(lw, 4));
    f.k[6] = key(6, prefix_cp(lw, 2));
    f.k[7] = key(7, shape(w));
    f.k[8] = key(8, pw);
    f.k[9] = key(9, nw);
    f.k[12] = key(12, pw + "\x1f" + lw);
    f.k[13] = key(13, lw + "\x1f" + nw);
    f.k[14] = key(14, ps + "\x1f" + shape(w) + "\x1f" + ns + (i == 0 ? "\x1f^" : ""));
}
u64 tag1_key(int t1) { return key(10, std::to_string(t1)); }
u64 tag2_key(int t2, int t1) { return key(11, std::to_string(t2) + "," + std::to_string(t1)); }

using Counts = std::array<std::uint32_t, K>;
using Table = std::unordered_map<u64, Counts>;

struct CountModel {
    Table total;
    std::vector<Table> fold;
    std::array<double, K> prior{};
    std::unordered_map<std::string, std::uint32_t> form_count;
    std::unordered_map<std::string, std::pair<std::uint32_t, std::uint32_t>> lower_count;  // lowercase -> (all, capitalised)
    double alpha = 2.0;

    void build(const std::vector<Sent>& S, int nfold) {
        fold.assign(static_cast<std::size_t>(nfold), Table());
        std::array<double, K> pc{};
        double n = 0;
        for (std::size_t si = 0; si < S.size(); ++si) {
            const Sent& s = S[si];
            Table& F = fold[si % static_cast<std::size_t>(nfold)];
            TokFeat f;
            for (std::size_t i = 0; i < s.w.size(); ++i) {
                static_keys(s, i, f);
                const int t1 = i > 0 ? s.t[i - 1] : K;
                const int t2 = i > 1 ? s.t[i - 2] : K;
                f.k[10] = tag1_key(t1);
                f.k[11] = tag2_key(t2, t1);
                const int t = s.t[i];
                for (int b = 0; b < kBlocks; ++b) {
                    total[f.k[static_cast<std::size_t>(b)]][static_cast<std::size_t>(t)]++;
                    F[f.k[static_cast<std::size_t>(b)]][static_cast<std::size_t>(t)]++;
                }
                pc[static_cast<std::size_t>(t)] += 1;
                n += 1;
                form_count[s.w[i]]++;
                auto& lc = lower_count[lower(s.w[i])];
                lc.first++;
                if (i > 0 && init_cap(s.w[i])) lc.second++;
            }
        }
        for (int k = 0; k < K; ++k) prior[static_cast<std::size_t>(k)] = (pc[static_cast<std::size_t>(k)] + 1.0) / (n + K);
    }
    void logp(u64 kk, int ex, double* out) const {
        Counts c{};
        auto it = total.find(kk);
        if (it != total.end()) c = it->second;
        if (ex >= 0) {
            auto jt = fold[static_cast<std::size_t>(ex)].find(kk);
            if (jt != fold[static_cast<std::size_t>(ex)].end())
                for (int k = 0; k < K; ++k) c[static_cast<std::size_t>(k)] -= jt->second[static_cast<std::size_t>(k)];
        }
        double n = 0;
        for (int k = 0; k < K; ++k) n += c[static_cast<std::size_t>(k)];
        for (int k = 0; k < K; ++k)
            out[k] = std::log((c[static_cast<std::size_t>(k)] + alpha * prior[static_cast<std::size_t>(k)]) / (n + alpha));
    }
    double form_freq(const std::string& w) const {
        auto it = form_count.find(w);
        return it == form_count.end() ? 0.0 : static_cast<double>(it->second);
    }
};

// ---------------------------------------------------------------- word vectors (E1's committed simvec)
struct Vectors {
    int dim = 0;
    std::unordered_map<std::string, std::vector<float>> v;
    bool load_simvec(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (b.size() < 24 || std::memcmp(b.data(), "CYSIMV1", 8) != 0) return false;
        std::size_t p = 8;
        auto u32 = [&](std::uint32_t& x) {
            std::memcpy(&x, b.data() + p, 4);
            p += 4;
        };
        std::uint32_t rows = 0, dims = 0;
        u32(rows);
        u32(dims);
        std::vector<std::string> words(rows);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto len = static_cast<unsigned char>(b[p++]);
            words[r].assign(b.data() + p, len);
            p += len;
        }
        std::vector<float> scale(rows);
        std::memcpy(scale.data(), b.data() + p, 4ULL * rows);
        p += 4ULL * rows;
        dim = static_cast<int>(dims);
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<float> x(dims);
            for (std::uint32_t d = 0; d < dims; ++d) x[d] = static_cast<float>(static_cast<signed char>(b[p + r * dims + d])) * scale[r];
            v[words[r]] = std::move(x);
        }
        return true;
    }
};

struct Featurizer {
    const CountModel* cm = nullptr;
    const Vectors* vec = nullptr;
    int D() const { return kBlocks * K + kScalars + (vec ? vec->dim + 1 : 0); }
    void fill_static(const Sent& s, std::size_t i, int ex, double* x) const {
        TokFeat f;
        static_keys(s, i, f);
        for (int b = 0; b < kBlocks; ++b) {
            if (b == 10 || b == 11) continue;
            cm->logp(f.k[static_cast<std::size_t>(b)], ex, x + b * K);
        }
        double* sc = x + kBlocks * K;
        const std::string& w = s.w[i];
        const double fc = cm->form_freq(w);
        const auto lc = cm->lower_count.find(lower(w));
        const double la = lc == cm->lower_count.end() ? 0.0 : lc->second.first, lcap = lc == cm->lower_count.end() ? 0.0 : lc->second.second;
        sc[0] = std::log1p(fc);
        sc[1] = fc == 0 ? 1.0 : 0.0;
        sc[2] = init_cap(w) ? 1.0 : 0.0;
        sc[3] = all_caps(w) ? 1.0 : 0.0;
        sc[4] = has_digit(w) ? 1.0 : 0.0;
        sc[5] = w.find('-') != std::string::npos ? 1.0 : 0.0;
        sc[6] = i == 0 ? 1.0 : 0.0;
        sc[7] = (lcap + 0.5) / (la + 1.0);  // share of the lowercase form's mid-sentence uses that are capitalised
        sc[8] = std::log1p(la);
        sc[9] = is_punct(w) ? 1.0 : 0.0;
        if (vec) {
            double* vv = sc + kScalars;
            auto it = vec->v.find(lower(w));
            if (it == vec->v.end()) {
                for (int d = 0; d < vec->dim; ++d) vv[d] = 0;
                vv[vec->dim] = 1.0;
            } else {
                for (int d = 0; d < vec->dim; ++d) vv[d] = it->second[static_cast<std::size_t>(d)];
                vv[vec->dim] = 0.0;
            }
        }
    }
    void fill_tags(int t2, int t1, int ex, double* x) const {
        cm->logp(tag1_key(t1), ex, x + 10 * K);
        cm->logp(tag2_key(t2, t1), ex, x + 11 * K);
    }
};

// ---------------------------------------------------------------- the Cypha classifier (as E1)
struct CyphaClf {
    cypha::Cypha model;
    bool init(int dim, double delta_lr, double world_lr, double ood_sigma, int field_dim) {
        static int counter = 0;
        const auto dir = std::filesystem::temp_directory_path() / ("e2_cypha_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(dir);
        cypha::FreshModelParams fp;
        fp.input_dim = dim;
        fp.field_dim = field_dim;
        fp.temperature = 1.0;
        fp.world_lr = world_lr;
        fp.delta_lr = delta_lr;
        const cypha::CNode root = cypha::create_fresh_model_root(fp);
        const auto path = (dir / "fresh.cypha").string();
        cypha::save_cypha_file(path.c_str(), root);
        const bool ok = model.load(path);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        if (!ok) return false;
        model.world_lr() = world_lr;
        model.delta_lr() = delta_lr;
        model.ood_sigma() = ood_sigma;
        model.train_params().enc_lr = 0.0;  // W_enc frozen (identity)
        return true;
    }
    void learn(const std::vector<double>& x, const std::string& label) {
        cypha::UpdateOpts uo;
        uo.use_gh = true;
        const auto r = model.update(x.data(), static_cast<int>(x.size()), &label, nullptr, uo);
        if (!r.detail.empty()) {
            std::fprintf(stderr, "cypha update: %s\n", r.detail.c_str());
            std::exit(3);
        }
    }
    cypha::PredictOut ask(const std::vector<double>& x) {
        cypha::PredictOpts po;
        po.use_gh = true;
        return model.predict(x.data(), static_cast<int>(x.size()), po);
    }
};

// ---------------------------------------------------------------- standardising and whitening (as E1)
struct Standard {
    std::vector<double> mu, inv, W;
    void apply(std::vector<double>& x) const {
        for (std::size_t d = 0; d < x.size(); ++d) x[d] = (x[d] - mu[d]) * inv[d];
        if (W.empty()) return;
        const std::size_t D = x.size();
        thread_local std::vector<double> y;
        y.assign(D, 0.0);
        for (std::size_t i = 0; i < D; ++i) {
            const double* w = W.data() + i * D;
            double s = 0;
            for (std::size_t j = 0; j < D; ++j) s += w[j] * x[j];
            y[i] = s;
        }
        x.swap(y);
    }
};

void jacobi_eigen(std::vector<double>& A, int n, std::vector<double>& evals, std::vector<double>& V) {
    V.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) V[static_cast<std::size_t>(i * n + i)] = 1.0;
    auto a = [&](int i, int j) -> double& { return A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]; };
    auto v = [&](int i, int j) -> double& { return V[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]; };
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0, diag = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) (i == j ? diag : off) += a(i, j) * a(i, j);
        if (off <= 1e-22 * std::max(1.0, diag)) break;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q) {
                const double apq = a(p, q);
                if (std::fabs(apq) < 1e-300) continue;
                const double theta = (a(q, q) - a(p, p)) / (2 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = a(k, p), akq = a(k, q);
                    a(k, p) = c * akp - s * akq;
                    a(k, q) = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = a(p, k), aqk = a(q, k);
                    a(p, k) = c * apk - s * aqk;
                    a(q, k) = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = v(k, p), vkq = v(k, q);
                    v(k, p) = c * vkp - s * vkq;
                    v(k, q) = s * vkp + c * vkq;
                }
            }
    }
    evals.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) evals[static_cast<std::size_t>(i)] = a(i, i);
}

// ---------------------------------------------------------------- span F1 with a sentence bootstrap
struct PRF {
    std::array<double, NT> tp{}, fp{}, fn{};
};
void add_counts(const std::vector<Span>& gold, const std::vector<Span>& pred, PRF& c) {
    std::vector<Span> g = gold, p = pred;
    std::sort(g.begin(), g.end());
    std::sort(p.begin(), p.end());
    for (const auto& x : p) {
        if (std::binary_search(g.begin(), g.end(), x)) c.tp[static_cast<std::size_t>(x.type)] += 1;
        else c.fp[static_cast<std::size_t>(x.type)] += 1;
    }
    for (const auto& x : g)
        if (!std::binary_search(p.begin(), p.end(), x)) c.fn[static_cast<std::size_t>(x.type)] += 1;
}
double f1(double tp, double fp, double fn) { return tp + fp + fn > 0 ? 2 * tp / (2 * tp + fp + fn) : 0.0; }
// micro F1 over types [0, nt)
double micro(const PRF& c, int nt) {
    double tp = 0, fp = 0, fn = 0;
    for (int k = 0; k < nt; ++k) tp += c.tp[static_cast<std::size_t>(k)], fp += c.fp[static_cast<std::size_t>(k)], fn += c.fn[static_cast<std::size_t>(k)];
    return f1(tp, fp, fn);
}

struct EvalOut {
    PRF all;
    std::vector<PRF> per_sent;
    double tok_ok = 0, tok_n = 0;
};
EvalOut evaluate(const std::vector<Sent>& S, const std::vector<std::vector<int>>& pred) {
    EvalOut e;
    e.per_sent.resize(S.size());
    for (std::size_t i = 0; i < S.size(); ++i) {
        add_counts(spans_of(S[i].t), spans_of(pred[i]), e.per_sent[i]);
        for (std::size_t j = 0; j < S[i].t.size(); ++j) e.tok_ok += S[i].t[j] == pred[i][j], e.tok_n += 1;
    }
    for (const auto& c : e.per_sent)
        for (int k = 0; k < NT; ++k)
            e.all.tp[static_cast<std::size_t>(k)] += c.tp[static_cast<std::size_t>(k)], e.all.fp[static_cast<std::size_t>(k)] += c.fp[static_cast<std::size_t>(k)],
                e.all.fn[static_cast<std::size_t>(k)] += c.fn[static_cast<std::size_t>(k)];
    return e;
}
// 95% bootstrap interval over sentences of the F1 of type ty (-1: micro over the 5 core types, -2: micro over all 6)
std::pair<double, double> boot_f1(const EvalOut& e, int ty, int reps, std::uint64_t seed) {
    const std::size_t n = e.per_sent.size();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> U(0, n - 1);
    std::vector<double> v;
    for (int b = 0; b < reps; ++b) {
        PRF c;
        for (std::size_t i = 0; i < n; ++i) {
            const auto& x = e.per_sent[U(rng)];
            for (int k = 0; k < NT; ++k)
                c.tp[static_cast<std::size_t>(k)] += x.tp[static_cast<std::size_t>(k)], c.fp[static_cast<std::size_t>(k)] += x.fp[static_cast<std::size_t>(k)],
                    c.fn[static_cast<std::size_t>(k)] += x.fn[static_cast<std::size_t>(k)];
        }
        v.push_back(ty == -1 ? micro(c, kCoreTypes) : ty == -2 ? micro(c, NT)
                                                                : f1(c.tp[static_cast<std::size_t>(ty)], c.fp[static_cast<std::size_t>(ty)], c.fn[static_cast<std::size_t>(ty)]));
    }
    std::sort(v.begin(), v.end());
    return {v[static_cast<std::size_t>(0.025 * reps)], v[static_cast<std::size_t>(0.975 * reps)]};
}
void print_eval(const char* name, const EvalOut& e, bool intervals) {
    std::printf("  %-40s micro F1 (5 types) %.4f", name, micro(e.all, kCoreTypes));
    if (intervals) {
        const auto b = boot_f1(e, -1, 2000, 20261112);
        std::printf(" [%.4f, %.4f]", b.first, b.second);
    }
    std::printf("  with MISC %.4f  token acc %.4f\n", micro(e.all, NT), e.tok_n > 0 ? e.tok_ok / e.tok_n : 0.0);
    for (int k = 0; k < NT; ++k) {
        const auto& c = e.all;
        const double tp = c.tp[static_cast<std::size_t>(k)], fp = c.fp[static_cast<std::size_t>(k)], fn = c.fn[static_cast<std::size_t>(k)];
        std::printf("    %-7s P %.4f R %.4f F1 %.4f", kTypes[static_cast<std::size_t>(k)], tp + fp > 0 ? tp / (tp + fp) : 0.0, tp + fn > 0 ? tp / (tp + fn) : 0.0,
                    f1(tp, fp, fn));
        if (intervals) {
            const auto b = boot_f1(e, k, 2000, 20261112 + static_cast<std::uint64_t>(k));
            std::printf(" [%.4f, %.4f]", b.first, b.second);
        }
        std::printf("  (gold %.0f)\n", tp + fn);
    }
    std::fflush(stdout);
}

// ---------------------------------------------------------------- tagging
template <class Scorer>
std::vector<std::vector<int>> tag_all(const std::vector<Sent>& S, const Featurizer& F, const Standard& Z, Scorer&& score) {
    std::vector<std::vector<int>> out(S.size());
    std::vector<double> x(static_cast<std::size_t>(F.D()));
    for (std::size_t si = 0; si < S.size(); ++si) {
        const Sent& s = S[si];
        int t1 = K, t2 = K;
        out[si].resize(s.w.size());
        for (std::size_t i = 0; i < s.w.size(); ++i) {
            F.fill_static(s, i, -1, x.data());
            F.fill_tags(t2, t1, -1, x.data());
            const int p = score(x, Z);
            out[si][i] = p;
            t2 = t1;
            t1 = p;
        }
    }
    return out;
}

void write_pred(const std::string& path, const std::vector<Sent>& S, const std::vector<std::vector<int>>& pred) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    for (std::size_t i = 0; i < S.size(); ++i) {
        std::string sp;
        for (const auto& x : spans_of(pred[i])) {
            if (!sp.empty()) sp += ';';
            sp += std::to_string(S[i].off[static_cast<std::size_t>(x.a)].first) + ":" + std::to_string(S[i].off[static_cast<std::size_t>(x.b - 1)].second) + ":" +
                  kTypes[static_cast<std::size_t>(x.type)];
        }
        std::fprintf(f, "%s\t%s\t%s\n", S[i].title.c_str(), S[i].text.c_str(), sp.c_str());
    }
    std::fclose(f);
}

struct Opt {
    std::string train, dev, test, hand, simvec = "data/cyphalm/simvec_ppmi32.bin", prompts, prompt_out, pred_out;
    double alpha = 2.0, delta_lr = 0.01, world_lr = 0.008, ood_sigma = 12.0, whiten = 0.1, o_keep = 1.0;
    int passes = 3, seed = 1, field_dim = 16;
    std::size_t cypha_sents = 15000;
    bool read_test = false, diag = false;
};

int cmd_run(const Opt& o) {
    const auto t0 = Clock::now();
    std::vector<Sent> tr, dv, te, hd, pr;
    if (!read_silver(o.train, tr) || !read_silver(o.dev, dv)) return std::fprintf(stderr, "cannot read train/dev\n"), 1;
    if (o.read_test && !read_silver(o.test, te)) return std::fprintf(stderr, "cannot read test\n"), 1;
    if (o.read_test && !o.hand.empty() && !read_silver(o.hand, hd)) return std::fprintf(stderr, "cannot read hand set\n"), 1;
    if (!o.prompts.empty()) {
        std::ifstream in(o.prompts);
        if (!in) return std::fprintf(stderr, "cannot read prompts\n"), 1;
        std::string line;
        std::getline(in, line);  // header
        while (std::getline(in, line)) {
            while (!line.empty() && line.back() == '\r') line.pop_back();
            const auto f = split(line, '\t');
            if (f.size() < 3) continue;
            prompt_sents(f[0], unescape_tsv(f[2]), pr);
        }
    }
    CountModel cm;
    cm.alpha = o.alpha;
    cm.build(tr, 5);
    Vectors V;
    if (!V.load_simvec(o.simvec)) return std::fprintf(stderr, "cannot load simvec %s\n", o.simvec.c_str()), 1;
    Featurizer F{&cm, &V};
    const int D = F.D();
    const std::size_t Du = static_cast<std::size_t>(D);
    // Cypha's training rows: a seeded sample of training sentences (all of them build the counts), cross-fitted.
    std::vector<std::size_t> pick(tr.size());
    std::iota(pick.begin(), pick.end(), 0);
    {
        std::mt19937_64 g(static_cast<std::uint64_t>(o.seed) * 7919ULL + 17);
        std::shuffle(pick.begin(), pick.end(), g);
        if (pick.size() > o.cypha_sents) pick.resize(o.cypha_sents);
    }
    std::size_t ntr = 0;
    for (auto si : pick) ntr += tr[si].w.size();
    std::vector<float> X(ntr * Du);
    std::vector<int> Y(ntr);
    std::vector<std::size_t> first(pick.size());
    {
        std::vector<double> x(Du);
        std::size_t r = 0;
        for (std::size_t q = 0; q < pick.size(); ++q) {
            first[q] = r;
            const Sent& s = tr[pick[q]];
            const int ex = static_cast<int>(pick[q] % 5);
            for (std::size_t i = 0; i < s.w.size(); ++i, ++r) {
                F.fill_static(s, i, ex, x.data());
                const int t1 = i > 0 ? s.t[i - 1] : K, t2 = i > 1 ? s.t[i - 2] : K;
                F.fill_tags(t2, t1, ex, x.data());
                for (std::size_t d = 0; d < Du; ++d) X[r * Du + d] = static_cast<float>(x[d]);
                Y[r] = s.t[i];
            }
        }
    }
    Standard Z;
    Z.mu.assign(Du, 0);
    Z.inv.assign(Du, 0);
    for (std::size_t r = 0; r < ntr; ++r)
        for (std::size_t d = 0; d < Du; ++d) Z.mu[d] += X[r * Du + d];
    for (auto& m : Z.mu) m /= static_cast<double>(ntr);
    {
        std::vector<double> var(Du, 0);
        for (std::size_t r = 0; r < ntr; ++r)
            for (std::size_t d = 0; d < Du; ++d) {
                const double e = X[r * Du + d] - Z.mu[d];
                var[d] += e * e;
            }
        for (std::size_t d = 0; d < Du; ++d) {
            const double sd = std::sqrt(var[d] / static_cast<double>(ntr));
            Z.inv[d] = sd > 1e-9 ? 1.0 / sd : 0.0;
        }
    }
    if (o.whiten > 0) {
        std::vector<double> cmu(static_cast<std::size_t>(K) * Du, 0.0), cnt(K, 0.0), z(Du);
        auto zrow = [&](std::size_t r) {
            for (std::size_t d = 0; d < Du; ++d) z[d] = (X[r * Du + d] - Z.mu[d]) * Z.inv[d];
        };
        for (std::size_t r = 0; r < ntr; ++r) {
            zrow(r);
            cnt[static_cast<std::size_t>(Y[r])] += 1;
            for (std::size_t d = 0; d < Du; ++d) cmu[static_cast<std::size_t>(Y[r]) * Du + d] += z[d];
        }
        for (int k = 0; k < K; ++k)
            for (std::size_t d = 0; d < Du; ++d) cmu[static_cast<std::size_t>(k) * Du + d] /= std::max(1.0, cnt[static_cast<std::size_t>(k)]);
        std::vector<double> S(Du * Du, 0.0);
        for (std::size_t r = 0; r < ntr; ++r) {
            zrow(r);
            const double* m = cmu.data() + static_cast<std::size_t>(Y[r]) * Du;
            for (std::size_t d = 0; d < Du; ++d) z[d] -= m[d];
            for (std::size_t i = 0; i < Du; ++i) {
                const double zi = z[i];
                if (zi == 0) continue;
                double* row = S.data() + i * Du;
                for (std::size_t j = i; j < Du; ++j) row[j] += zi * z[j];
            }
        }
        for (std::size_t i = 0; i < Du; ++i)
            for (std::size_t j = i; j < Du; ++j) {
                S[i * Du + j] /= static_cast<double>(ntr);
                S[j * Du + i] = S[i * Du + j];
            }
        std::vector<double> ev, U;
        jacobi_eigen(S, D, ev, U);
        double mean_ev = 0;
        for (double e : ev) mean_ev += std::max(0.0, e);
        mean_ev /= static_cast<double>(D);
        Z.W.assign(Du * Du, 0.0);
        for (std::size_t i = 0; i < Du; ++i) {
            const double sc = 1.0 / std::sqrt(std::max(0.0, ev[i]) + o.whiten * mean_ev);
            for (std::size_t j = 0; j < Du; ++j) Z.W[i * Du + j] = sc * U[j * Du + i];
        }
    }
    std::size_t ntok_all = 0;
    for (const auto& s : tr) ntok_all += s.w.size();
    std::printf("E2 config: alpha %.2f D %d whiten %.3f delta_lr %.4f world_lr %.4f passes %d seed %d | counts from %zu sentences (%zu tokens, %zu keys); "
                "Cypha rows %zu sentences (%zu tokens); dev %zu sentences; features %.1f s\n",
                o.alpha, D, o.whiten, o.delta_lr, o.world_lr, o.passes, o.seed, tr.size(), ntok_all, cm.total.size(), pick.size(), ntr, dv.size(),
                secs(t0, Clock::now()));
    std::fflush(stdout);

    if (o.diag) {
        auto counts_alone = [&](const std::vector<double>& x, const Standard&) {
            std::array<double, K> s{};
            for (int b = 0; b < kBlocks; ++b)
                for (int k = 0; k < K; ++k) s[static_cast<std::size_t>(k)] += x[static_cast<std::size_t>(b * K + k)];
            return static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin());
        };
        print_eval("dev, counts alone (no learning)", evaluate(dv, tag_all(dv, F, Z, counts_alone)), false);
        std::vector<double> mu(static_cast<std::size_t>(K) * Du, 0), cnt(K, 0), var(Du, 0), zr(Du);
        auto trow = [&](std::size_t r) {
            for (std::size_t d = 0; d < Du; ++d) zr[d] = X[r * Du + d];
            Z.apply(zr);
        };
        for (std::size_t r = 0; r < ntr; ++r) {
            cnt[static_cast<std::size_t>(Y[r])] += 1;
            trow(r);
            for (std::size_t d = 0; d < Du; ++d) mu[static_cast<std::size_t>(Y[r]) * Du + d] += zr[d];
        }
        for (int k = 0; k < K; ++k)
            for (std::size_t d = 0; d < Du; ++d) mu[static_cast<std::size_t>(k) * Du + d] /= std::max(1.0, cnt[static_cast<std::size_t>(k)]);
        for (std::size_t r = 0; r < ntr; ++r) {
            trow(r);
            for (std::size_t d = 0; d < Du; ++d) {
                const double e = zr[d] - mu[static_cast<std::size_t>(Y[r]) * Du + d];
                var[d] += e * e;
            }
        }
        for (auto& v : var) v = std::max(1e-6, v / static_cast<double>(ntr));
        auto exact = [&](const std::vector<double>& x0, const Standard& z) {
            std::vector<double> x = x0;
            z.apply(x);
            int best = 0;
            double bl = -1e300;
            for (int k = 0; k < K; ++k) {
                double l = std::log(std::max(1.0, cnt[static_cast<std::size_t>(k)]));
                for (std::size_t d = 0; d < Du; ++d) {
                    const double e = x[d] - mu[static_cast<std::size_t>(k) * Du + d];
                    l -= e * e / (2 * var[d]);
                }
                if (l > bl) bl = l, best = k;
            }
            return best;
        };
        print_eval("dev, exact class means (diagnostic)", evaluate(dv, tag_all(dv, F, Z, exact)), false);
    }
    if (o.passes <= 0) return 0;

    CyphaClf clf;
    if (!clf.init(D, o.delta_lr, o.world_lr, o.ood_sigma, o.field_dim)) return std::fprintf(stderr, "Cypha init failed\n"), 1;
    auto cypha_score = [&](const std::vector<double>& x0, const Standard& z) {
        std::vector<double> x = x0;
        z.apply(x);
        return tag_id(clf.ask(x).label);
    };
    std::mt19937_64 rng(static_cast<std::uint64_t>(o.seed));
    std::vector<std::size_t> order(pick.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> x(Du);
    std::mt19937_64 keep_rng(static_cast<std::uint64_t>(o.seed) * 104729ULL + 3);
    std::uniform_real_distribution<double> keep_u(0.0, 1.0);
    double train_s = 0;
    for (int pass = 1; pass <= o.passes; ++pass) {
        std::shuffle(order.begin(), order.end(), rng);
        const auto a = Clock::now();
        for (std::size_t q : order) {
            const std::size_t r0 = first[q];
            for (std::size_t i = 0; i < tr[pick[q]].w.size(); ++i) {
                const std::size_t r = r0 + i;
                if (Y[r] == 0 && o.o_keep < 1.0 && keep_u(keep_rng) >= o.o_keep) continue;  // amendment 1: O tokens subsampled
                for (std::size_t d = 0; d < Du; ++d) x[d] = X[r * Du + d];
                Z.apply(x);
                clf.learn(x, kTags[static_cast<std::size_t>(Y[r])]);
            }
        }
        train_s += secs(a, Clock::now());
        const auto pd = tag_all(dv, F, Z, cypha_score);
        char nm[64];
        std::snprintf(nm, sizeof nm, "dev, Cypha, pass %d", pass);
        print_eval(nm, evaluate(dv, pd), pass == o.passes);
        std::printf("    train %.1f s so far\n", train_s);
        std::fflush(stdout);
        if (pass == o.passes && !o.pred_out.empty()) write_pred(o.pred_out + ".dev.tsv", dv, pd);
    }
    if (!pr.empty()) {
        const auto a = Clock::now();
        const auto pp = tag_all(pr, F, Z, cypha_score);
        std::FILE* f = std::fopen(o.prompt_out.c_str(), "w");
        if (!f) return std::fprintf(stderr, "cannot write %s\n", o.prompt_out.c_str()), 1;
        std::fprintf(f, "id\tstart\tend\ttype\tsurface\n");
        std::size_t nsp = 0;
        for (std::size_t i = 0; i < pr.size(); ++i)
            for (const auto& sp : spans_of(pp[i])) {
                const std::size_t s0 = pr[i].off[static_cast<std::size_t>(sp.a)].first, s1 = pr[i].off[static_cast<std::size_t>(sp.b - 1)].second;
                std::fprintf(f, "%s\t%zu\t%zu\t%s\t%s\n", pr[i].id.c_str(), pr[i].base + s0, pr[i].base + s1, kTypes[static_cast<std::size_t>(sp.type)],
                             pr[i].text.substr(s0, s1 - s0).c_str());
                ++nsp;
            }
        std::fclose(f);
        std::printf("prompts: %zu sentences tagged, %zu spans, %.1f s -> %s\n", pr.size(), nsp, secs(a, Clock::now()), o.prompt_out.c_str());
    }
    if (!o.read_test) return 0;
    std::printf("\nTEST READ (once)\n");
    const auto a = Clock::now();
    const auto pt = tag_all(te, F, Z, cypha_score);
    const double tag_s = secs(a, Clock::now());
    const auto et = evaluate(te, pt);
    print_eval("silver test, Cypha", et, true);
    const double m = micro(et.all, kCoreTypes);
    std::printf("  gate (micro F1 over the 5 types >= 0.80 on the silver test): %s\n", m >= 0.80 ? "MET" : "NOT MET");
    if (!o.pred_out.empty()) write_pred(o.pred_out + ".test.tsv", te, pt);
    if (!hd.empty()) {
        const auto ph = tag_all(hd, F, Z, cypha_score);
        print_eval("hand-checked set, Cypha", evaluate(hd, ph), true);
        if (!o.pred_out.empty()) write_pred(o.pred_out + ".hand.tsv", hd, ph);
    }
    std::size_t nt = 0;
    for (const auto& s : te) nt += s.w.size();
    std::printf("  cost: training %.1f s (%d passes), tagging %.0f tokens/s single thread, count tables %zu keys\n", train_s, o.passes,
                static_cast<double>(nt) / tag_s, cm.total.size());
    return 0;
}

int cmd_selftest() {
    int fails = 0;
    auto check = [&](bool c, const char* what) {
        if (!c) std::fprintf(stderr, "FAIL: %s\n", what), ++fails;
    };
    std::vector<std::string> w;
    std::vector<std::pair<std::size_t, std::size_t>> off;
    tokenize("In 1954, Holden's U.S. plant (40,100,000 tons) closed. Dr. Smith left.", w, off);
    const std::vector<std::string> want = {"In", "1954", ",", "Holden", "'s", "U.S.", "plant", "(", "40,100,000", "tons", ")", "closed", ".", "Dr.", "Smith", "left", "."};
    check(w == want, "tokenize");
    check(off[3].first == 9 && off[3].second == 15 && off[4].first == 15, "offsets and possessive");
    // spans from bytes and back
    Sent s;
    s.text = "General Motors New Zealand opened in Petone in 1954.";
    tokenize(s.text, s.w, s.off);
    tags_from_bytes(s, {{0, 26, 2}, {37, 43, 1}, {47, 51, 3}});
    const auto sp = spans_of(s.t);
    check(sp.size() == 3 && sp[0].a == 0 && sp[0].b == 4 && sp[0].type == 2 && sp[1].type == 1 && sp[2].type == 3, "BIO from byte spans");
    check(spans_of({0, 2, 2, 0}).size() == 1 && spans_of({0, 2, 2, 0})[0].a == 1, "I- after O opens a span");
    check(spans_of({1, 1}).size() == 2, "B B is two spans");
    // F1
    PRF c;
    add_counts({{0, 2, 0}, {3, 4, 1}}, {{0, 2, 0}, {3, 5, 1}}, c);
    check(c.tp[0] == 1 && c.fp[1] == 1 && c.fn[1] == 1, "span counts exact match");
    check(std::fabs(micro(c, kCoreTypes) - 0.5) < 1e-12, "micro F1");
    // prompt sentences
    std::vector<Sent> ps;
    prompt_sents("x", "Black powder burns. It was used in China. ok", ps);
    check(ps.size() == 2 && ps[1].w[0] == "It" && ps[1].base == 20, "prompt sentences");
    // cross-fitting
    std::vector<Sent> S(5);
    for (int i = 0; i < 5; ++i) {
        S[static_cast<std::size_t>(i)].w = {"in", i == 0 ? "Zzz" : "Paris"};
        S[static_cast<std::size_t>(i)].t = {0, b_tag(1)};
    }
    CountModel cm;
    cm.build(S, 5);
    double la[K], le[K];
    cm.logp(key(0, "Zzz"), -1, la);
    cm.logp(key(0, "Zzz"), 0, le);
    check(la[b_tag(1)] > la[0], "seen form favours its tag");
    check(std::fabs(le[b_tag(1)] - std::log(cm.prior[static_cast<std::size_t>(b_tag(1))])) < 1e-12, "cross-fit leaves prior");
    std::printf("cypha_ner selftest: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "selftest") return cmd_selftest();
    if (mode != "run") {
        std::fprintf(stderr, "usage: cypha_ner run --train T --dev D [--test X --read-test] [options] | selftest\n");
        return 2;
    }
    Opt o;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--train") o.train = nx();
        else if (a == "--dev") o.dev = nx();
        else if (a == "--test") o.test = nx();
        else if (a == "--hand") o.hand = nx();
        else if (a == "--read-test") o.read_test = true;
        else if (a == "--alpha") o.alpha = std::atof(nx().c_str());
        else if (a == "--simvec") o.simvec = nx();
        else if (a == "--delta-lr") o.delta_lr = std::atof(nx().c_str());
        else if (a == "--world-lr") o.world_lr = std::atof(nx().c_str());
        else if (a == "--whiten") o.whiten = std::atof(nx().c_str());
        else if (a == "--o-keep") o.o_keep = std::atof(nx().c_str());
        else if (a == "--passes") o.passes = std::atoi(nx().c_str());
        else if (a == "--cypha-sents") o.cypha_sents = static_cast<std::size_t>(std::atol(nx().c_str()));
        else if (a == "--seed") o.seed = std::atoi(nx().c_str());
        else if (a == "--diag") o.diag = true;
        else if (a == "--prompts") o.prompts = nx();
        else if (a == "--prompt-out") o.prompt_out = nx();
        else if (a == "--pred-out") o.pred_out = nx();
        else return std::fprintf(stderr, "unknown option %s\n", a.c_str()), 2;
    }
    if (o.read_test && o.test.empty()) return std::fprintf(stderr, "--read-test needs --test\n"), 2;
    if (!o.prompts.empty() && o.prompt_out.empty()) return std::fprintf(stderr, "--prompts needs --prompt-out\n"), 2;
    return cmd_run(o);
}
