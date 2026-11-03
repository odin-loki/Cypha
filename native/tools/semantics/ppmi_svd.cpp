// C1: distributional word vectors from counts only (PPMI + randomised SVD) and a random-indexing
// comparison. No neural net, no external library.
//
// What it does, in order:
//   1. Cleans a MediaWiki XML dump (enwik8): entities decoded, <ref>/<math>/other tags, comments,
//      {{templates}}, {| tables |}, [[a|b]] -> b, [[a]] -> a, [[Category:..]]/[[Image:..]]/[[xx:..]]
//      dropped, '' and ''' dropped, URLs dropped, #REDIRECT pages skipped. Lowercase; a token is a run of
//      letters with single internal apostrophes/hyphens; any token touching a digit is dropped.
//      A newline ends a context unit (windows never cross a line or an article).
//   2. Vocabulary: the top V words (default 50,000) by frequency, used for targets and contexts.
//      Out-of-vocabulary tokens are deleted from the stream before windowing (word2vec convention).
//   3. Co-occurrence: symmetric window +-5, weight 1/distance. Counted memory-safely: per chunk of
//      the stream the (target, context, distance) triples are packed into uint64 keys, sorted, reduced
//      to runs, and merged into one sorted run. Final store is CSR with uint16 columns, float weight,
//      and a saturating uint8 raw count (so pairs seen once can be dropped).
//   4. PPMI with context-distribution smoothing (alpha = 0.75). Pairs with raw count < min_count are
//      dropped; min_count 1 (keep everything) is the headline setting and min_count 2 (drop singletons,
//      as plan row C1 said) is run as the ablation. Marginals come from the retained pairs.
//   5. Randomised truncated SVD (Halko, Martinsson, Tropp 2011), written here: Gaussian test matrix
//      with k + 10 columns, 3 power iterations with modified Gram-Schmidt (twice) between products,
//      small matrix B = Q^T A, exact eigendecomposition of B B^T by cyclic Jacobi in double.
//      Vectors = U * S^p, rows L2-normalised, for p = 0, 0.5, 1. The top triplets are checked by the
//      residual |A v - s u| / s.
//   6. Random indexing (Sahlgren 2005): 2,000 dimensions, 10 nonzero +-1 index entries per word,
//      same window and 1/distance weight, fixed memory (V x 2000 floats). Four context weightings
//      (none, log(N/f), f^-0.5, f^-1); each scored with and without mean-centring of the unit rows.
//   7. Spearman rho on WordSim-353, SimLex-999 and MEN (pairs with a word outside the vocabulary
//      are excluded and counted). Nearest neighbours for 15 probe words. Build time and peak RAM.
//
// Measured 2026-10-27 on enwik8 (100 MB XML: 12,343 pages, 11.23M tokens, 265,565 types; the top 50,000
// words cover 96.56% of tokens). One thread on a shared 2-core VM: 152 s for the whole run (cleaning 4 s,
// counting 9 s, SVD 55 s), peak RAM 640 MB (reached while counting).
//   Window +-5: 97.1M ordered updates -> 19.6M distinct ordered pairs (9.8M unordered); 6.67M have count >= 2.
//   Spearman rho, WS353 (350 of 353 pairs) | SimLex-999 (982 of 999) | MEN (2800 of 3000); the rest have a
//   word outside the vocabulary:
//     PPMI-SVD 300 d, min_count 1, p = 0.5 : 0.672 | 0.255 | 0.637   (p = 0: 0.681 0.262 0.633; p = 1: 0.592 0.243 0.581)
//     PPMI-SVD 300 d, min_count 2, p = 0.5 : 0.589 | 0.220 | 0.548
//     random indexing 2000 d, best variant : 0.404 | 0.131 | 0.374   (plain: 0.156 0.089 0.153)
//   Published PPMI-SVD on billion-word corpora is about 0.65-0.70 (WS353) and 0.3-0.4 (SimLex);
//   this corpus is about 90x smaller. Report: docs/reports/C1_PPMI_SVD_VECTORS.md
//
// Usage:
//   g++ -O2 -std=c++20 native/tools/ppmi_svd.cpp -o ppmi_svd
//   ./ppmi_svd <enwik8> <EN-WS-353-ALL.txt> <EN-SIMLEX-999.txt> <EN-MEN-TR-3k.txt|-> <out_prefix>
//              [--vocab 50000] [--dim 300] [--window 5] [--alpha 0.75] [--power 3] [--oversample 10]
//              [--ri-dim 2000] [--ri-nnz 10] [--max-bytes N] [--min-count 1] [--no-ri] [--no-ablation]
//              [--dump-clean file]
//   Writes <out_prefix>.vectors.f32 (V x dim float32, row-major, p = 0.5, min_count 1 by default, unit rows) and
//   <out_prefix>.vocab.tsv (word, count; row i of the matrix = line i). Keep the prefix outside the repo.
//   Evaluation files (tab-separated: word1 word2 score) are the copies in
//   github.com/mfaruqui/eval-word-vectors, data/word-sim/.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <strings.h>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma GCC optimize("O3")

namespace {

using Clock = std::chrono::steady_clock;
const Clock::time_point g_t0 = Clock::now();

double elapsed() { return std::chrono::duration<double>(Clock::now() - g_t0).count(); }

long peak_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmHWM:", 0) == 0) return std::atol(line.c_str() + 6) / 1024;
    return -1;
}

void say(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::printf("[%7.1fs %5ldMB] %s\n", elapsed(), peak_rss_mb(), buf);
    std::fflush(stdout);
}

constexpr uint16_t kSep = 0xFFFF;
constexpr uint32_t kSep32 = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------------------------
// 1. Cleaning and tokenising
// ---------------------------------------------------------------------------------------------

std::string lower_ascii(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// &amp; &lt; &gt; &quot; decoded; every other entity becomes a space. Two rounds undo double escaping.
void decode_entities(std::string& s) {
    for (int pass = 0; pass < 2; ++pass) {
        std::string o;
        o.reserve(s.size());
        const size_t n = s.size();
        for (size_t i = 0; i < n;) {
            if (s[i] != '&') {
                o.push_back(s[i++]);
                continue;
            }
            if (s.compare(i, 5, "&amp;") == 0) {
                o.push_back('&');
                i += 5;
            } else if (s.compare(i, 4, "&lt;") == 0) {
                o.push_back('<');
                i += 4;
            } else if (s.compare(i, 4, "&gt;") == 0) {
                o.push_back('>');
                i += 4;
            } else if (s.compare(i, 6, "&quot;") == 0) {
                o.push_back('"');
                i += 6;
            } else {
                size_t j = i + 1;
                while (j < n && j < i + 10 && (std::isalnum((unsigned char)s[j]) || s[j] == '#')) ++j;
                o.push_back(' ');
                i = (j < n && s[j] == ';' && j > i + 1) ? j + 1 : i + 1;
            }
        }
        s.swap(o);
    }
}

bool drops_content(const std::string& name) {
    return name == "ref" || name == "math" || name == "gallery" || name == "timeline" || name == "imagemap";
}

// Removes comments and tags; for ref/math/gallery/timeline/imagemap removes the content too.
std::string strip_tags(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    const size_t n = s.size();
    for (size_t i = 0; i < n;) {
        if (s[i] != '<') {
            o.push_back(s[i++]);
            continue;
        }
        if (s.compare(i, 4, "<!--") == 0) {
            size_t e = s.find("-->", i + 4);
            o.push_back(' ');
            i = (e == std::string::npos) ? n : e + 3;
            continue;
        }
        size_t j = i + 1;
        const bool closing = (j < n && s[j] == '/');
        if (closing) ++j;
        if (j >= n || !std::isalpha((unsigned char)s[j])) {
            o.push_back(' ');
            ++i;
            continue;
        }
        const size_t name_start = j;
        while (j < n && std::isalnum((unsigned char)s[j])) ++j;
        const std::string name = lower_ascii(s.substr(name_start, j - name_start));
        size_t e = s.find('>', j);
        if (e == std::string::npos || e - i > 600) {
            o.push_back(' ');
            i = j;
            continue;
        }
        const bool selfclose = s[e - 1] == '/';
        i = e + 1;
        o.push_back(' ');
        if (!closing && !selfclose && drops_content(name)) {
            size_t c = s.find("</" + name, i);
            if (c != std::string::npos) {
                size_t g = s.find('>', c);
                i = (g == std::string::npos) ? n : g + 1;
            }
        }
    }
    return o;
}

// Removes {{...}} (nested) and {| ... |} (nested).
std::string strip_templates(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    const size_t n = s.size();
    int dt = 0, dtab = 0;
    for (size_t i = 0; i < n;) {
        const char a = s[i], b = (i + 1 < n) ? s[i + 1] : 0;
        if (a == '{' && b == '{') {
            ++dt;
            i += 2;
        } else if (a == '}' && b == '}' && dt > 0) {
            --dt;
            i += 2;
        } else if (a == '{' && b == '|') {
            ++dtab;
            i += 2;
        } else if (a == '|' && b == '}' && dtab > 0) {
            --dtab;
            i += 2;
        } else if (dt > 0 || dtab > 0) {
            ++i;
        } else {
            o.push_back(a);
            ++i;
        }
    }
    return o;
}

bool is_dropped_namespace(const std::string& inner) {
    if (inner.empty() || inner[0] == ':') return false;
    size_t p = inner.find(':');
    if (p == std::string::npos || p > 12) return false;
    const std::string pre = lower_ascii(inner.substr(0, p));
    static const char* ns[] = {"category", "image", "file", "media", "wikipedia", "template", "portal", "help",
                               "user", "talk", "wikt", "wiktionary", "commons", "meta", "mediawiki"};
    for (const char* x : ns)
        if (pre == x) return true;
    // interlanguage link: 2-3 letters, optional -xx part, e.g. de, fr, zh-min-nan
    size_t k = 0;
    while (k < pre.size() && std::isalpha((unsigned char)pre[k])) ++k;
    if (k >= 2 && k <= 3 && (k == pre.size() || pre[k] == '-')) return true;
    return false;
}

// [[a|b]] -> b, [[a]] -> a, namespace links dropped with their caption; nested links handled by recursion.
void strip_links(const std::string& s, std::string& out) {
    const size_t n = s.size();
    for (size_t i = 0; i < n;) {
        if (s[i] == '[' && i + 1 < n && s[i + 1] == '[') {
            int d = 1;
            size_t j = i + 2;
            while (j + 1 < n && d > 0) {
                if (s[j] == '[' && s[j + 1] == '[') {
                    ++d;
                    j += 2;
                } else if (s[j] == ']' && s[j + 1] == ']') {
                    --d;
                    j += 2;
                } else {
                    ++j;
                }
            }
            if (d > 0) {
                out.push_back(' ');
                i += 2;
                continue;
            }
            const std::string inner = s.substr(i + 2, j - 2 - (i + 2));
            i = j;
            if (is_dropped_namespace(inner)) {
                out.push_back(' ');
                continue;
            }
            // first '|' at nesting depth 0
            size_t bar = std::string::npos;
            int dd = 0;
            for (size_t k = 0; k < inner.size(); ++k) {
                if (inner[k] == '[' && k + 1 < inner.size() && inner[k + 1] == '[') {
                    ++dd;
                    ++k;
                } else if (inner[k] == ']' && k + 1 < inner.size() && inner[k + 1] == ']') {
                    --dd;
                    ++k;
                } else if (inner[k] == '|' && dd == 0) {
                    bar = k;
                    break;
                }
            }
            std::string disp = (bar == std::string::npos) ? inner : inner.substr(bar + 1);
            if (disp.empty()) disp = (bar == std::string::npos) ? inner : inner.substr(0, bar);
            strip_links(disp, out);  // no spaces added: [[radical]]s must stay "radicals" 
        } else {
            out.push_back(s[i++]);
        }
    }
}

// Length in bytes of a letter at s[i]: ASCII letters and the 2-byte Latin-1 / Latin Extended-A letters.
inline int letter_len(const std::string& s, size_t i) {
    const unsigned char c = (unsigned char)s[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return 1;
    if (i + 1 >= s.size()) return 0;
    const unsigned char d = (unsigned char)s[i + 1];
    if (c == 0xC3 && d >= 0x80 && d <= 0xBF && d != 0x97 && d != 0xB7) return 2;
    if ((c == 0xC4 || c == 0xC5) && (d & 0xC0) == 0x80) return 2;
    return 0;
}

inline bool url_start(const std::string& s, size_t i) {
    const size_t n = s.size();
    if (i + 6 >= n) return false;
    if (strncasecmp(&s[i], "http://", 7) == 0 || strncasecmp(&s[i], "https:/", 7) == 0) return true;
    if (strncasecmp(&s[i], "ftp://", 6) == 0 || strncasecmp(&s[i], "www.", 4) == 0) return true;
    return false;
}

// Calls emit(word) for each token and emit_break() at each newline.
template <class Emit, class Brk>
void tokenize(const std::string& s, Emit&& emit, Brk&& emit_break) {
    const size_t n = s.size();
    std::string w;
    bool has_digit = false;
    auto flush = [&]() {
        if (!w.empty() && !has_digit) emit(w);
        w.clear();
        has_digit = false;
    };
    for (size_t i = 0; i < n;) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\n') {
            flush();
            emit_break();
            ++i;
            continue;
        }
        if (w.empty() && url_start(s, i)) {
            while (i < n && !std::isspace((unsigned char)s[i]) && s[i] != ']' && s[i] != '|' && s[i] != '<') ++i;
            continue;
        }
        const int L = letter_len(s, i);
        if (L) {
            if (L == 1) {
                w.push_back((char)std::tolower(c));
            } else {
                unsigned char d = (unsigned char)s[i + 1];
                if (c == 0xC3 && d >= 0x80 && d <= 0x9E) d = (unsigned char)(d + 0x20);
                w.push_back((char)c);
                w.push_back((char)d);
            }
            i += L;
            continue;
        }
        if (c >= '0' && c <= '9') {
            w.push_back((char)c);
            has_digit = true;
            ++i;
            continue;
        }
        if ((c == '\'' || c == '-') && !w.empty() && i + 1 < n && letter_len(s, i + 1) > 0) {
            w.push_back((char)c);
            ++i;
            continue;
        }
        flush();
        ++i;
    }
    flush();
    emit_break();
}

struct Corpus {
    std::vector<uint32_t> toks;  // ids in order of first appearance, kSep32 = line/article break
    std::vector<std::string> words;
    std::vector<uint64_t> freq;
    size_t pages = 0, redirects = 0;
};

Corpus clean_and_tokenize(const std::string& raw, const std::string& dump_path) {
    Corpus C;
    std::unordered_map<std::string, uint32_t> ids;
    ids.reserve(1 << 20);
    C.toks.reserve(raw.size() / 6);
    std::ofstream dump;
    size_t dumped = 0;
    if (!dump_path.empty()) dump.open(dump_path);
    auto emit = [&](const std::string& w) {
        auto it = ids.find(w);
        uint32_t id;
        if (it == ids.end()) {
            id = (uint32_t)C.words.size();
            ids.emplace(w, id);
            C.words.push_back(w);
            C.freq.push_back(0);
        } else {
            id = it->second;
        }
        ++C.freq[id];
        C.toks.push_back(id);
        if (dump.is_open() && dumped < 300000) {
            dump << w << ' ';
            dumped += w.size() + 1;
        }
    };
    auto brk = [&]() {
        if (!C.toks.empty() && C.toks.back() != kSep32) C.toks.push_back(kSep32);
        if (dump.is_open() && dumped < 300000) dump << '\n';
    };
    size_t pos = 0;
    while (true) {
        size_t a = raw.find("<text", pos);
        if (a == std::string::npos) break;
        size_t gt = raw.find('>', a);
        if (gt == std::string::npos) break;
        size_t b = raw.find("</text>", gt);
        if (b == std::string::npos) b = raw.size();
        pos = b;
        if (raw[gt - 1] == '/') continue;  // <text ... /> empty
        std::string t = raw.substr(gt + 1, b - gt - 1);
        ++C.pages;
        decode_entities(t);
        {
            size_t k = 0;
            while (k < t.size() && std::isspace((unsigned char)t[k])) ++k;
            if (strncasecmp(t.c_str() + k, "#redirect", 9) == 0) {
                ++C.redirects;
                continue;
            }
        }
        t = strip_tags(t);
        t = strip_templates(t);
        std::string linked;
        linked.reserve(t.size());
        strip_links(t, linked);
        tokenize(linked, emit, brk);
        if (dump.is_open() && dumped < 300000) dump << "\n=====\n";
    }
    return C;
}

// ---------------------------------------------------------------------------------------------
// 2. Co-occurrence counts
// ---------------------------------------------------------------------------------------------

struct Run {
    std::vector<uint32_t> k;
    std::vector<float> w;
    std::vector<uint8_t> n;
    size_t size() const { return k.size(); }
};

void merge_runs(Run& a, Run& b, Run& out) {
    out.k.clear();
    out.w.clear();
    out.n.clear();
    out.k.reserve(a.size() + b.size());
    out.w.reserve(a.size() + b.size());
    out.n.reserve(a.size() + b.size());
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a.k[i] < b.k[j]) {
            out.k.push_back(a.k[i]);
            out.w.push_back(a.w[i]);
            out.n.push_back(a.n[i]);
            ++i;
        } else if (a.k[i] > b.k[j]) {
            out.k.push_back(b.k[j]);
            out.w.push_back(b.w[j]);
            out.n.push_back(b.n[j]);
            ++j;
        } else {
            out.k.push_back(a.k[i]);
            out.w.push_back(a.w[i] + b.w[j]);
            out.n.push_back((uint8_t)std::min(255, (int)a.n[i] + (int)b.n[j]));
            ++i;
            ++j;
        }
    }
    for (; i < a.size(); ++i) {
        out.k.push_back(a.k[i]);
        out.w.push_back(a.w[i]);
        out.n.push_back(a.n[i]);
    }
    for (; j < b.size(); ++j) {
        out.k.push_back(b.k[j]);
        out.w.push_back(b.w[j]);
        out.n.push_back(b.n[j]);
    }
}

struct Counts {
    uint32_t V = 0;
    std::vector<uint64_t> rp;
    std::vector<uint16_t> col;
    std::vector<float> w;
    std::vector<uint8_t> n;
    double total_weight = 0;
    uint64_t updates = 0;  // ordered (target, context) updates emitted
};

Counts build_counts(const std::vector<uint16_t>& corp, uint32_t V, int W) {
    Counts C;
    C.V = V;
    const size_t N = corp.size();
    const size_t chunk = 1500000;
    double invd[8] = {0};
    for (int d = 1; d <= W && d < 8; ++d) invd[d] = 1.0 / d;
    Run global, tmp;
    std::vector<uint64_t> buf;
    buf.reserve(chunk * (size_t)W * 2 + 16);
    size_t done_chunks = 0;
    for (size_t start = 0; start < N; start += chunk) {
        const size_t end = std::min(N, start + chunk);
        buf.clear();
        for (size_t i = start; i < end; ++i) {
            const uint32_t t = corp[i];
            if (t == kSep) continue;
            for (int d = 1; d <= W; ++d) {
                const size_t j = i + d;
                if (j >= N) break;
                const uint32_t c = corp[j];
                if (c == kSep) break;
                buf.push_back(((uint64_t)(t * V + c) << 3) | (uint64_t)d);
                buf.push_back(((uint64_t)(c * V + t) << 3) | (uint64_t)d);
            }
        }
        C.updates += buf.size();
        std::sort(buf.begin(), buf.end());
        Run r;
        r.k.reserve(buf.size() / 2);
        r.w.reserve(buf.size() / 2);
        r.n.reserve(buf.size() / 2);
        for (size_t i = 0; i < buf.size();) {
            const uint64_t key = buf[i] >> 3;
            double w = 0;
            int cnt = 0;
            size_t j = i;
            while (j < buf.size() && (buf[j] >> 3) == key) {
                w += invd[buf[j] & 7];
                ++cnt;
                ++j;
            }
            r.k.push_back((uint32_t)key);
            r.w.push_back((float)w);
            r.n.push_back((uint8_t)std::min(cnt, 255));
            i = j;
        }
        if (global.size() == 0) {
            global = std::move(r);
        } else {
            merge_runs(global, r, tmp);
            std::swap(global, tmp);
            std::vector<uint32_t>().swap(tmp.k);
            std::vector<float>().swap(tmp.w);
            std::vector<uint8_t>().swap(tmp.n);
        }
        ++done_chunks;
        if (done_chunks % 4 == 0) say("counts: %zu/%zu tokens, %zu distinct ordered pairs so far", end, N, global.size());
    }
    std::vector<uint64_t>().swap(buf);
    const size_t nnz = global.size();
    C.rp.assign((size_t)V + 1, 0);
    C.col.resize(nnz);
    for (size_t p = 0; p < nnz; ++p) {
        const uint32_t row = global.k[p] / V;
        ++C.rp[row + 1];
        C.col[p] = (uint16_t)(global.k[p] - row * V);
    }
    for (uint32_t r = 0; r < V; ++r) C.rp[r + 1] += C.rp[r];
    std::vector<uint32_t>().swap(global.k);
    C.w = std::move(global.w);
    C.n = std::move(global.n);
    for (size_t p = 0; p < nnz; ++p) C.total_weight += C.w[p];
    return C;
}

// ---------------------------------------------------------------------------------------------
// 3. PPMI
// ---------------------------------------------------------------------------------------------

struct CSR {
    uint32_t n = 0;
    std::vector<uint64_t> rp;
    std::vector<uint16_t> col;
    std::vector<float> val;
    size_t nnz() const { return col.size(); }
};

struct PpmiStats {
    size_t kept_pairs = 0, positive_pairs = 0;
};

CSR build_ppmi(const Counts& C, int min_count, double alpha, PpmiStats& st) {
    const uint32_t V = C.V;
    std::vector<double> row(V, 0.0), colsum(V, 0.0);
    st.kept_pairs = 0;
    for (uint32_t i = 0; i < V; ++i)
        for (uint64_t p = C.rp[i]; p < C.rp[i + 1]; ++p) {
            if (C.n[p] < min_count) continue;
            row[i] += C.w[p];
            colsum[C.col[p]] += C.w[p];
            ++st.kept_pairs;
        }
    double Z = 0;
    std::vector<double> lca(V, 0.0);
    for (uint32_t c = 0; c < V; ++c)
        if (colsum[c] > 0) Z += std::pow(colsum[c], alpha);
    for (uint32_t c = 0; c < V; ++c) lca[c] = colsum[c] > 0 ? alpha * std::log(colsum[c]) : 0.0;
    const double logZ = std::log(Z);
    CSR A;
    A.n = V;
    A.rp.assign((size_t)V + 1, 0);
    A.col.reserve(st.kept_pairs / 2);
    A.val.reserve(st.kept_pairs / 2);
    for (uint32_t i = 0; i < V; ++i) {
        const double lr = row[i] > 0 ? std::log(row[i]) : 0.0;
        for (uint64_t p = C.rp[i]; p < C.rp[i + 1]; ++p) {
            if (C.n[p] < min_count) continue;
            const uint16_t c = C.col[p];
            const double pmi = std::log((double)C.w[p]) + logZ - lr - lca[c];
            if (pmi > 0) {
                A.col.push_back(c);
                A.val.push_back((float)pmi);
            }
        }
        A.rp[i + 1] = A.col.size();
    }
    st.positive_pairs = A.col.size();
    A.col.shrink_to_fit();
    A.val.shrink_to_fit();
    return A;
}

// ---------------------------------------------------------------------------------------------
// 4. Randomised SVD
// ---------------------------------------------------------------------------------------------

struct Mat {  // column-major: c columns of length n
    size_t n = 0, c = 0;
    std::vector<float> d;
    void init(size_t n_, size_t c_) {
        n = n_;
        c = c_;
        d.assign(n * c, 0.0f);
    }
    float* col(size_t j) { return d.data() + j * n; }
    const float* col(size_t j) const { return d.data() + j * n; }
};

template <int B>
void mul_blk(const CSR& A, const float* const* x, float* const* y) {
    const uint64_t* rp = A.rp.data();
    const uint16_t* col = A.col.data();
    const float* val = A.val.data();
    const float* xs[B];
    float* ys[B];
    for (int b = 0; b < B; ++b) {
        xs[b] = x[b];
        ys[b] = y[b];
    }
    for (uint32_t i = 0; i < A.n; ++i) {
        float acc[B];
        for (int b = 0; b < B; ++b) acc[b] = 0.f;
        for (uint64_t p = rp[i]; p < rp[i + 1]; ++p) {
            const float v = val[p];
            const unsigned c = col[p];
            for (int b = 0; b < B; ++b) acc[b] += v * xs[b][c];
        }
        for (int b = 0; b < B; ++b) ys[b][i] = acc[b];
    }
}

template <int B>
void mul_t_blk(const CSR& A, const float* const* x, float* const* y) {
    const uint64_t* rp = A.rp.data();
    const uint16_t* col = A.col.data();
    const float* val = A.val.data();
    const float* xs[B];
    float* ys[B];
    for (int b = 0; b < B; ++b) {
        xs[b] = x[b];
        ys[b] = y[b];
        std::fill(ys[b], ys[b] + A.n, 0.0f);
    }
    for (uint32_t i = 0; i < A.n; ++i) {
        float xi[B];
        for (int b = 0; b < B; ++b) xi[b] = xs[b][i];
        for (uint64_t p = rp[i]; p < rp[i + 1]; ++p) {
            const float v = val[p];
            const unsigned c = col[p];
            for (int b = 0; b < B; ++b) ys[b][c] += v * xi[b];
        }
    }
}

// Y = A X (transpose = false) or Y = A^T X (transpose = true); A is square here.
void spmm(const CSR& A, const Mat& X, Mat& Y, bool transpose) {
    Y.init(A.n, X.c);
    size_t j = 0;
    for (; j + 4 <= X.c; j += 4) {
        const float* xs[4] = {X.col(j), X.col(j + 1), X.col(j + 2), X.col(j + 3)};
        float* ys[4] = {Y.col(j), Y.col(j + 1), Y.col(j + 2), Y.col(j + 3)};
        if (transpose) mul_t_blk<4>(A, xs, ys); else mul_blk<4>(A, xs, ys);
    }
    for (; j < X.c; ++j) {
        const float* xs[1] = {X.col(j)};
        float* ys[1] = {Y.col(j)};
        if (transpose) mul_t_blk<1>(A, xs, ys); else mul_blk<1>(A, xs, ys);
    }
}

float dotf(const float* a, const float* b, size_t n) {
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; ++k) s[k] += a[i + k] * b[i + k];
    float t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (; i < n; ++i) t += a[i] * b[i];
    return t;
}

void axpyf(float* __restrict y, const float* __restrict x, float a, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] += a * x[i];
}

// Modified Gram-Schmidt with one re-orthogonalisation ("twice is enough"). Y is overwritten by Q.
void orthonormalize(Mat& Y) {
    const size_t n = Y.n;
    for (size_t j = 0; j < Y.c; ++j) {
        float* yj = Y.col(j);
        for (int pass = 0; pass < 2; ++pass)
            for (size_t i = 0; i < j; ++i) {
                const float r = dotf(Y.col(i), yj, n);
                axpyf(yj, Y.col(i), -r, n);
            }
        const double nrm = std::sqrt((double)dotf(yj, yj, n));
        if (nrm > 1e-30) {
            const float inv = (float)(1.0 / nrm);
            for (size_t i = 0; i < n; ++i) yj[i] *= inv;
        } else {
            std::fill(yj, yj + n, 0.0f);
        }
    }
}

// Cyclic Jacobi for a symmetric n x n matrix (row-major, destroyed). Eigenvectors in the columns of Vm.
void jacobi_eig(std::vector<double>& A, int n, std::vector<double>& Vm, std::vector<double>& ev) {
    Vm.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) Vm[(size_t)i * n + i] = 1.0;
    double fro = 0;
    for (double x : A) fro += x * x;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += A[(size_t)p * n + q] * A[(size_t)p * n + q];
        if (off <= 1e-26 * fro) break;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[(size_t)p * n + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double theta = (A[(size_t)q * n + q] - A[(size_t)p * n + p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = A[(size_t)k * n + p], akq = A[(size_t)k * n + q];
                    A[(size_t)k * n + p] = c * akp - s * akq;
                    A[(size_t)k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = A[(size_t)p * n + k], aqk = A[(size_t)q * n + k];
                    A[(size_t)p * n + k] = c * apk - s * aqk;
                    A[(size_t)q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = Vm[(size_t)k * n + p], vkq = Vm[(size_t)k * n + q];
                    Vm[(size_t)k * n + p] = c * vkp - s * vkq;
                    Vm[(size_t)k * n + q] = s * vkp + c * vkq;
                }
            }
    }
    ev.resize(n);
    for (int i = 0; i < n; ++i) ev[i] = A[(size_t)i * n + i];
}

struct SvdResult {
    int k = 0;
    std::vector<double> S;
    Mat U;                       // V x k, columns are left singular vectors
    std::vector<double> resid;   // relative residual |A v - s u| / s for sampled triplets
    std::vector<int> resid_idx;
};

SvdResult randomized_svd(const CSR& A, int k, int oversample, int power_iters, uint64_t seed) {
    const int l = k + oversample;
    const size_t V = A.n;
    SvdResult R;
    R.k = k;
    Mat X, Y, Z;
    X.init(V, l);
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> g(0.f, 1.f);
        for (float& v : X.d) v = g(rng);
    }
    spmm(A, X, Y, false);  // Y = A Omega
    say("svd: Y = A*Omega done");
    for (int it = 0; it < power_iters; ++it) {
        orthonormalize(Y);
        spmm(A, Y, Z, true);  // Z = A^T Q
        orthonormalize(Z);
        spmm(A, Z, Y, false);  // Y = A Z
        say("svd: power iteration %d/%d done", it + 1, power_iters);
    }
    orthonormalize(Y);  // Y is now Q (V x l)
    Mat& Q = Y;
    Mat Bt;
    spmm(A, Q, Bt, true);  // Bt = A^T Q  (= B^T, B = Q^T A is l x V)
    // G = B B^T = Bt^T Bt  (l x l)
    std::vector<double> G((size_t)l * l, 0.0);
    for (int i = 0; i < l; ++i)
        for (int j = i; j < l; ++j) {
            const double d = dotf(Bt.col(i), Bt.col(j), V);
            G[(size_t)i * l + j] = G[(size_t)j * l + i] = d;
        }
    std::vector<double> W, ev;
    jacobi_eig(G, l, W, ev);
    std::vector<int> order(l);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return ev[a] > ev[b]; });
    say("svd: small eigenproblem (%dx%d) done", l, l);
    R.S.resize(k);
    R.U.init(V, k);
    std::vector<double> Wk((size_t)l * k);
    for (int j = 0; j < k; ++j) {
        R.S[j] = std::sqrt(std::max(ev[order[j]], 0.0));
        for (int i = 0; i < l; ++i) Wk[(size_t)i * k + j] = W[(size_t)i * l + order[j]];
    }
    for (int j = 0; j < k; ++j) {
        float* uj = R.U.col(j);
        for (int i = 0; i < l; ++i) axpyf(uj, Q.col(i), (float)Wk[(size_t)i * k + j], V);
    }
    // residual check: v_j = Bt W_j / s_j ; r = |A v_j - s_j u_j| / s_j
    for (int j : {0, 9, 99, k - 1}) {
        if (j < 0 || j >= k || R.S[j] <= 0) continue;
        Mat v, av;
        v.init(V, 1);
        for (int i = 0; i < l; ++i) axpyf(v.col(0), Bt.col(i), (float)(Wk[(size_t)i * k + j] / R.S[j]), V);
        spmm(A, v, av, false);
        double num = 0;
        for (size_t r = 0; r < V; ++r) {
            const double d = (double)av.col(0)[r] - R.S[j] * (double)R.U.col(j)[r];
            num += d * d;
        }
        R.resid.push_back(std::sqrt(num) / R.S[j]);
        R.resid_idx.push_back(j);
    }
    return R;
}

// ---------------------------------------------------------------------------------------------
// 5. Random indexing
// ---------------------------------------------------------------------------------------------

struct RiModel {
    int D = 0;
    uint32_t V = 0;
    std::vector<float> R;     // V x D raw context vectors
    std::vector<float> mean;  // mean of unit rows
    std::vector<float> rnorm; // |R_w|
    std::vector<float> cnorm; // |u_w - mean|
};

void build_ri(const std::vector<uint16_t>& corp, uint32_t V, int W, int D, int nnz, const std::vector<float>& cw,
              uint64_t seed, RiModel& M) {
    M.D = D;
    M.V = V;
    M.R.assign((size_t)V * D, 0.0f);
    std::vector<int16_t> iv((size_t)V * nnz);
    std::mt19937_64 rng(seed);
    for (uint32_t w = 0; w < V; ++w) {
        int chosen = 0;
        while (chosen < nnz) {
            const int pos = (int)(rng() % (uint64_t)D);
            bool dup = false;
            for (int q = 0; q < chosen; ++q)
                if (std::abs((int)iv[(size_t)w * nnz + q]) - 1 == pos) dup = true;
            if (dup) continue;
            iv[(size_t)w * nnz + chosen] = (int16_t)((chosen % 2 == 0 ? 1 : -1) * (pos + 1));
            ++chosen;
        }
    }
    const size_t N = corp.size();
    for (size_t i = 0; i < N; ++i) {
        const uint32_t t = corp[i];
        if (t == kSep) continue;
        for (int d = 1; d <= W; ++d) {
            const size_t j = i + d;
            if (j >= N) break;
            const uint32_t c = corp[j];
            if (c == kSep) break;
            const float wt = 1.0f / (float)d;
            float* rt = &M.R[(size_t)t * D];
            float* rc = &M.R[(size_t)c * D];
            const float a = wt * cw[c], b = wt * cw[t];
            const int16_t* ic = &iv[(size_t)c * nnz];
            const int16_t* it = &iv[(size_t)t * nnz];
            for (int q = 0; q < nnz; ++q) {
                const int sc = ic[q], st = it[q];
                rt[std::abs(sc) - 1] += sc > 0 ? a : -a;
                rc[std::abs(st) - 1] += st > 0 ? b : -b;
            }
        }
    }
    // unit-row mean, norms
    M.rnorm.assign(V, 0.f);
    std::vector<double> mu(D, 0.0);
    for (uint32_t w = 0; w < V; ++w) {
        const float* r = &M.R[(size_t)w * D];
        double s = 0;
        for (int j = 0; j < D; ++j) s += (double)r[j] * r[j];
        M.rnorm[w] = (float)std::sqrt(s);
        if (s > 0) {
            const double inv = 1.0 / std::sqrt(s);
            for (int j = 0; j < D; ++j) mu[j] += r[j] * inv;
        }
    }
    M.mean.resize(D);
    double mm = 0;
    for (int j = 0; j < D; ++j) {
        mu[j] /= V;
        M.mean[j] = (float)mu[j];
        mm += mu[j] * mu[j];
    }
    M.cnorm.assign(V, 0.f);
    for (uint32_t w = 0; w < V; ++w) {
        const float* r = &M.R[(size_t)w * D];
        double dm = 0;
        if (M.rnorm[w] > 0)
            for (int j = 0; j < D; ++j) dm += (double)r[j] * mu[j];
        dm = M.rnorm[w] > 0 ? dm / M.rnorm[w] : 0;
        M.cnorm[w] = (float)std::sqrt(std::max(1.0 - 2.0 * dm + mm, 1e-12));
    }
}

// ---------------------------------------------------------------------------------------------
// 6. Evaluation
// ---------------------------------------------------------------------------------------------

struct EvalSet {
    std::string name;
    std::vector<std::pair<int, int>> pairs;  // covered pairs (vocabulary ranks)
    std::vector<double> gold;
    size_t total = 0, oov_pairs = 0;
    std::vector<std::string> missing;
};

EvalSet load_eval(const std::string& name, const std::string& path, const std::unordered_map<std::string, int>& vocab) {
    EvalSet E;
    E.name = name;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string a, b;
        double g;
        if (!(ss >> a >> b >> g)) continue;
        a = lower_ascii(a);
        b = lower_ascii(b);
        ++E.total;
        auto ia = vocab.find(a), ib = vocab.find(b);
        if (ia == vocab.end() || ib == vocab.end()) {
            ++E.oov_pairs;
            if (ia == vocab.end()) E.missing.push_back(a);
            if (ib == vocab.end()) E.missing.push_back(b);
            continue;
        }
        E.pairs.push_back({ia->second, ib->second});
        E.gold.push_back(g);
    }
    return E;
}

std::vector<double> rank_avg(const std::vector<double>& v) {
    const size_t n = v.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
    std::vector<double> r(n);
    for (size_t i = 0; i < n;) {
        size_t j = i;
        while (j < n && v[idx[j]] == v[idx[i]]) ++j;
        const double avg = 0.5 * (double)(i + j - 1) + 1.0;
        for (size_t k = i; k < j; ++k) r[idx[k]] = avg;
        i = j;
    }
    return r;
}

double spearman(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() < 3) return 0;
    const std::vector<double> ra = rank_avg(a), rb = rank_avg(b);
    const size_t n = a.size();
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) {
        ma += ra[i];
        mb += rb[i];
    }
    ma /= n;
    mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i) {
        sab += (ra[i] - ma) * (rb[i] - mb);
        saa += (ra[i] - ma) * (ra[i] - ma);
        sbb += (rb[i] - mb) * (rb[i] - mb);
    }
    return sab / std::sqrt(saa * sbb);
}

// get(w, out) writes the unit vector of word w (dim floats).
double eval_rho(const EvalSet& E, int dim, const std::function<void(int, float*)>& get) {
    std::unordered_map<int, std::vector<float>> cache;
    auto vec = [&](int w) -> const std::vector<float>& {
        auto it = cache.find(w);
        if (it != cache.end()) return it->second;
        std::vector<float> v(dim);
        get(w, v.data());
        return cache.emplace(w, std::move(v)).first->second;
    };
    std::vector<double> sims, gold;
    for (size_t i = 0; i < E.pairs.size(); ++i) {
        const auto& a = vec(E.pairs[i].first);
        const auto& b = vec(E.pairs[i].second);
        double s = 0;
        for (int j = 0; j < dim; ++j) s += (double)a[j] * b[j];
        sims.push_back(s);
        gold.push_back(E.gold[i]);
    }
    return spearman(sims, gold);
}

struct ResultRow {
    std::string label;
    std::vector<double> rho;  // per eval set
};

std::vector<float> make_vectors(const SvdResult& S, double p) {
    const size_t V = S.U.n;
    const int k = S.k;
    std::vector<float> E(V * (size_t)k);
    std::vector<float> sc(k);
    for (int j = 0; j < k; ++j) sc[j] = (float)std::pow(S.S[j], p);
    for (int j = 0; j < k; ++j) {
        const float* u = S.U.col(j);
        for (size_t w = 0; w < V; ++w) E[w * k + j] = u[w] * sc[j];
    }
    for (size_t w = 0; w < V; ++w) {
        double s = 0;
        for (int j = 0; j < k; ++j) s += (double)E[w * k + j] * E[w * k + j];
        const float inv = s > 0 ? (float)(1.0 / std::sqrt(s)) : 0.f;
        for (int j = 0; j < k; ++j) E[w * k + j] *= inv;
    }
    return E;
}

void print_nn(const std::string& label, const std::vector<std::string>& probes,
              const std::unordered_map<std::string, int>& vocab, const std::vector<std::string>& itos, uint32_t V,
              const std::function<void(int, std::vector<float>&)>& sims, int topn) {
    std::printf("\nNearest neighbours: %s\n", label.c_str());
    std::vector<float> s(V);
    for (const auto& p : probes) {
        auto it = vocab.find(p);
        if (it == vocab.end()) {
            std::printf("  %-9s (not in vocabulary)\n", p.c_str());
            continue;
        }
        sims(it->second, s);
        s[it->second] = -1e30f;
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + topn, idx.end(), [&](int a, int b) { return s[a] > s[b]; });
        std::printf("  %-9s", p.c_str());
        for (int i = 0; i < topn; ++i) std::printf(" %s", itos[idx[i]].c_str());
        std::printf("\n");
    }
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <corpus> <wordsim353> <simlex999> <men|-> <out_prefix> [--vocab N] [--dim K] "
                     "[--window W] [--alpha A] [--power Q] [--oversample P] [--ri-dim D] [--ri-nnz Z] "
                     "[--max-bytes N] [--min-count M] [--no-ri] [--no-ablation] [--dump-clean file]\n",
                     argv[0]);
        return 2;
    }
    const std::string corpus_path = argv[1], ws_path = argv[2], sl_path = argv[3], men_path = argv[4],
                      out_prefix = argv[5];
    uint32_t Vmax = 50000;
    int K = 300, W = 5, Q = 3, P = 10, RID = 2000, RIZ = 10;
    double alpha = 0.75;
    size_t max_bytes = 0;
    bool do_ri = true, do_ablation = true;
    int primary_mc = 1;  // min_count of the setting that is written to disk and shown as neighbours
    std::string dump_path;
    for (int i = 6; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : "0"; };
        if (a == "--vocab") Vmax = (uint32_t)std::atoi(next());
        else if (a == "--dim") K = std::atoi(next());
        else if (a == "--window") W = std::atoi(next());
        else if (a == "--alpha") alpha = std::atof(next());
        else if (a == "--power") Q = std::atoi(next());
        else if (a == "--oversample") P = std::atoi(next());
        else if (a == "--ri-dim") RID = std::atoi(next());
        else if (a == "--ri-nnz") RIZ = std::atoi(next());
        else if (a == "--max-bytes") max_bytes = (size_t)std::atoll(next());
        else if (a == "--min-count") primary_mc = std::atoi(next());
        else if (a == "--no-ri") do_ri = false;
        else if (a == "--no-ablation") do_ablation = false;
        else if (a == "--dump-clean") dump_path = next();
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (W < 1 || W > 7 || Vmax > 65000) {
        std::fprintf(stderr, "window must be 1..7 and vocab <= 65000\n");
        return 2;
    }

    // --- read and clean
    std::string raw;
    {
        std::ifstream f(corpus_path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "cannot open %s\n", corpus_path.c_str());
            return 1;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        raw = ss.str();
        if (max_bytes && raw.size() > max_bytes) raw.resize(max_bytes);
    }
    say("read %zu bytes", raw.size());
    Corpus C = clean_and_tokenize(raw, dump_path);
    const size_t raw_bytes = raw.size();
    std::string().swap(raw);
    uint64_t total_tokens = 0;
    for (uint64_t f : C.freq) total_tokens += f;
    say("cleaned: %zu pages (%zu redirects skipped), %llu tokens, %zu types", C.pages, C.redirects,
         (unsigned long long)total_tokens, C.words.size());

    // --- vocabulary
    std::vector<uint32_t> order(C.words.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return C.freq[a] != C.freq[b] ? C.freq[a] > C.freq[b] : C.words[a] < C.words[b];
    });
    const uint32_t V = (uint32_t)std::min<size_t>(Vmax, order.size());
    std::vector<uint32_t> rank_of(C.words.size(), kSep32);
    std::vector<std::string> itos(V);
    std::vector<uint64_t> vfreq(V);
    std::unordered_map<std::string, int> vocab;
    uint64_t covered_tokens = 0;
    for (uint32_t r = 0; r < V; ++r) {
        rank_of[order[r]] = r;
        itos[r] = C.words[order[r]];
        vfreq[r] = C.freq[order[r]];
        vocab.emplace(itos[r], (int)r);
        covered_tokens += vfreq[r];
    }
    std::vector<uint16_t> corp;
    corp.reserve(covered_tokens + 1000000);
    for (uint32_t id : C.toks) {
        if (id == kSep32) {
            if (!corp.empty() && corp.back() != kSep) corp.push_back(kSep);
        } else if (rank_of[id] != kSep32) {
            corp.push_back((uint16_t)rank_of[id]);
        }
    }
    {
        std::vector<uint32_t>().swap(C.toks);
        std::vector<std::string>().swap(C.words);
    }
    say("vocabulary: top %u words, count of rank V = %llu; covers %llu of %llu tokens (%.2f%%)", V,
         (unsigned long long)vfreq[V - 1], (unsigned long long)covered_tokens, (unsigned long long)total_tokens,
         100.0 * covered_tokens / (double)total_tokens);

    // --- evaluation sets
    std::vector<EvalSet> sets;
    sets.push_back(load_eval("WS353", ws_path, vocab));
    sets.push_back(load_eval("SimLex999", sl_path, vocab));
    if (men_path != "-") sets.push_back(load_eval("MEN", men_path, vocab));
    for (const auto& s : sets) {
        std::string miss;
        for (size_t i = 0; i < s.missing.size() && i < 12; ++i) miss += s.missing[i] + " ";
        say("eval %s: %zu pairs, %zu covered, %zu excluded (out of vocabulary). Missing words: %s", s.name.c_str(),
             s.total, s.pairs.size(), s.oov_pairs, miss.c_str());
    }
    std::vector<ResultRow> rows;
    auto score_all = [&](const std::string& label, int dim, const std::function<void(int, float*)>& get) {
        ResultRow r;
        r.label = label;
        for (const auto& s : sets) r.rho.push_back(eval_rho(s, dim, get));
        std::string line = label + ":";
        for (size_t i = 0; i < sets.size(); ++i) {
            char b[64];
            std::snprintf(b, sizeof b, " %s=%.4f", sets[i].name.c_str(), r.rho[i]);
            line += b;
        }
        say("%s", line.c_str());
        rows.push_back(r);
    };
    const std::vector<std::string> probes = {"king", "paris", "computer", "music", "river", "war", "january", "red",
                                             "run", "good", "france", "physics", "dog", "money", "bank"};

    // --- co-occurrence
    Counts counts = build_counts(corp, V, W);
    const size_t nnz_counts = counts.col.size();
    size_t nnz_sym = 0;  // unordered pairs {t, c}, t <= c (matrix is symmetric)
    for (uint32_t i = 0; i < V; ++i)
        for (uint64_t p = counts.rp[i]; p < counts.rp[i + 1]; ++p)
            if (counts.col[p] >= i) ++nnz_sym;
    size_t nnz_ge2 = 0;
    for (size_t p = 0; p < nnz_counts; ++p)
        if (counts.n[p] >= 2) ++nnz_ge2;
    const double t_counts = elapsed();
    say("co-occurrence: window +-%d, weight 1/d; %llu ordered updates; %zu nonzero ordered pairs (%zu unordered); "
         "%zu with raw count >= 2; total weight %.4g",
         W, (unsigned long long)counts.updates, nnz_counts, nnz_sym, nnz_ge2, counts.total_weight);

    // --- PPMI + SVD: the headline min_count first, then the other of {1, 2} as the ablation
    std::vector<float> E_best;  // p = 0.5, min_count = primary_mc
    std::string svd_notes;
    std::vector<int> mcs = {primary_mc};
    if (do_ablation) mcs.push_back(primary_mc == 1 ? 2 : 1);
    for (int mc : mcs) {
        PpmiStats st;
        CSR A = build_ppmi(counts, mc, alpha, st);
        say("PPMI (alpha %.2f, min_count %d): %zu pairs kept, %zu positive (matrix %ux%u)", alpha, mc, st.kept_pairs,
             st.positive_pairs, V, V);
        const double t_svd0 = elapsed();
        SvdResult S = randomized_svd(A, K, P, Q, 12345);
        const double t_svd = elapsed() - t_svd0;
        char nb[512];
        std::snprintf(nb, sizeof nb,
                      "min_count %d: SVD %.0fs; sigma_1 %.2f, sigma_100 %.2f, sigma_%d %.2f; residual |Av-su|/s at j=",
                      mc, t_svd, S.S[0], S.S[std::min(99, K - 1)], K, S.S[K - 1]);
        svd_notes += nb;
        for (size_t i = 0; i < S.resid.size(); ++i) {
            std::snprintf(nb, sizeof nb, "%d:%.3g ", S.resid_idx[i] + 1, S.resid[i]);
            svd_notes += nb;
        }
        svd_notes += "\n";
        say("%s", svd_notes.substr(svd_notes.rfind("min_count")).c_str());
        for (double p : {0.0, 0.5, 1.0}) {
            std::vector<float> E = make_vectors(S, p);
            char lab[96];
            std::snprintf(lab, sizeof lab, "PPMI-SVD min_count=%d p=%.1f", mc, p);
            score_all(lab, K, [&](int w, float* out) { std::memcpy(out, &E[(size_t)w * K], sizeof(float) * K); });
            if (mc == primary_mc && p == 0.5) E_best = std::move(E);
        }
        if (mc == primary_mc) {
            print_nn("PPMI-SVD (min_count " + std::to_string(mc) + ", p = 0.5, " + std::to_string(K) + " d)", probes, vocab, itos, V,
                     [&](int w, std::vector<float>& s) {
                         const float* q = &E_best[(size_t)w * K];
                         for (uint32_t r = 0; r < V; ++r) {
                             const float* v = &E_best[(size_t)r * K];
                             float d = 0;
                             for (int j = 0; j < K; ++j) d += q[j] * v[j];
                             s[r] = d;
                         }
                     },
                     8);
        }
    }
    const double t_ppmi_done = elapsed();

    // --- write vectors + vocabulary
    if (!E_best.empty()) {
        std::ofstream vf(out_prefix + ".vectors.f32", std::ios::binary);
        vf.write(reinterpret_cast<const char*>(E_best.data()), (std::streamsize)(E_best.size() * sizeof(float)));
        std::ofstream tf(out_prefix + ".vocab.tsv");
        for (uint32_t r = 0; r < V; ++r) tf << itos[r] << '\t' << vfreq[r] << '\n';
        say("wrote %s.vectors.f32 (%u x %d float32) and %s.vocab.tsv", out_prefix.c_str(), V, K, out_prefix.c_str());
    }
    std::vector<float>().swap(E_best);
    // free the count matrix before random indexing
    counts = Counts();

    // --- random indexing
    if (do_ri) {
        struct RiVariant {
            const char* name;
            int mode;
        };
        const RiVariant variants[] = {{"RI plain", 0}, {"RI log(N/f)", 1}, {"RI f^-0.5", 2}, {"RI f^-1", 3}};
        for (const RiVariant& var : variants) {
            std::vector<float> cw(V, 1.0f);
            for (uint32_t r = 0; r < V; ++r) {
                const double f = (double)vfreq[r];
                if (var.mode == 1) cw[r] = (float)std::max(0.0, std::log((double)covered_tokens / f));
                if (var.mode == 2) cw[r] = (float)(1.0 / std::sqrt(f));
                if (var.mode == 3) cw[r] = (float)(1.0 / f);
            }
            if (var.mode >= 2) {  // rescale so the mean weight over tokens is 1 (keeps floats in a sane range)
                double m = 0;
                for (uint32_t r = 0; r < V; ++r) m += cw[r] * (double)vfreq[r];
                m /= (double)covered_tokens;
                for (uint32_t r = 0; r < V; ++r) cw[r] = (float)(cw[r] / m);
            }
            const double t_ri0 = elapsed();
            RiModel M;
            build_ri(corp, V, W, RID, RIZ, cw, 777, M);
            say("random indexing (%s): %u x %d floats built in %.0fs", var.name, V, RID, elapsed() - t_ri0);
            score_all(std::string(var.name) + " cosine", RID, [&](int w, float* out) {
                const float* r = &M.R[(size_t)w * RID];
                const float inv = M.rnorm[w] > 0 ? 1.0f / M.rnorm[w] : 0.f;
                for (int j = 0; j < RID; ++j) out[j] = r[j] * inv;
            });
            score_all(std::string(var.name) + " centred", RID, [&](int w, float* out) {
                const float* r = &M.R[(size_t)w * RID];
                const float inv = M.rnorm[w] > 0 ? 1.0f / M.rnorm[w] : 0.f;
                const float ic = 1.0f / M.cnorm[w];
                for (int j = 0; j < RID; ++j) out[j] = (r[j] * inv - M.mean[j]) * ic;
            });
            if (var.mode == 2) {
                print_nn("random indexing (context weight f^-0.5, centred, " + std::to_string(RID) + " d)", probes, vocab, itos, V,
                         [&](int w, std::vector<float>& s) {
                             std::vector<float> q(RID);
                             const float inv = 1.0f / M.rnorm[w], ic = 1.0f / M.cnorm[w];
                             double mq = 0;
                             for (int j = 0; j < RID; ++j) {
                                 q[j] = (M.R[(size_t)w * RID + j] * inv - M.mean[j]) * ic;
                                 mq += (double)M.mean[j] * q[j];
                             }
                             for (uint32_t r = 0; r < V; ++r) {
                                 const float* v = &M.R[(size_t)r * RID];
                                 double d = 0;
                                 for (int j = 0; j < RID; ++j) d += (double)v[j] * q[j];
                                 s[r] = M.rnorm[r] > 0 ? (float)((d / M.rnorm[r] - mq) / M.cnorm[r]) : -1e30f;
                             }
                         },
                         8);
            }
        }
    }

    // --- summary
    std::printf("\n==== SUMMARY ====\n");
    std::printf("corpus: %zu bytes, %zu pages, %llu tokens, vocabulary %u (coverage %.2f%% of tokens)\n", raw_bytes,
                C.pages, (unsigned long long)total_tokens, V, 100.0 * covered_tokens / (double)total_tokens);
    std::printf("tokens used after dropping out-of-vocabulary: %zu (incl. line breaks)\n", corp.size());
    std::printf("co-occurrence nonzero pairs: %zu ordered (%zu unordered); raw count >= 2: %zu\n", nnz_counts, nnz_sym,
                nnz_ge2);
    std::printf("%s", svd_notes.c_str());
    std::printf("\n| method |");
    for (const auto& s : sets) std::printf(" %s (n=%zu) |", s.name.c_str(), s.pairs.size());
    std::printf("\n|---|");
    for (size_t i = 0; i < sets.size(); ++i) std::printf("---|");
    std::printf("\n");
    for (const auto& r : rows) {
        std::printf("| %s |", r.label.c_str());
        for (double x : r.rho) std::printf(" %.3f |", x);
        std::printf("\n");
    }
    for (const auto& s : sets)
        std::printf("%s: %zu pairs in file, %zu covered, %zu excluded\n", s.name.c_str(), s.total, s.pairs.size(),
                    s.oov_pairs);
    std::printf("time: counts done at %.0fs, PPMI+SVD done at %.0fs, total %.0fs; peak RAM %ld MB\n", t_counts,
                t_ppmi_done, elapsed(), peak_rss_mb());
    return 0;
}
