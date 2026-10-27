// S-1 "words as classes": Core Cypha's classifier fitted with one class per word (docs/SEMANTICS.md §2-3, §7).
//
// Each of the top 10,000 enwik8 words is a Cypha class. Each occurrence is one observation: the sum of fixed
// sparse +-1 random index vectors (d = 256, 8 nonzeros) of the words within +-5 on the same line, weighted
// 1/distance (random indexing). One online pass. Per token, the true class and 5 competitors drawn by
// unigram^0.75 are swapped into a 6-slot CyphaDifMemoryState, and Cypha's own memory_train() scores them,
// pulls the true class differential toward h - theta0, repels the competitors (softmax weighted, capped),
// applies MDL shrinkage and updates the WorldPrior. With --enc-lr > 0, Cypha's batch_encode() gives h = W x and
// contrastive_update_encoder_w() runs on every miss. The public Cypha::update() scores every class, so it is
// too slow here (probe at the end: about 70-120 hours for 10,000 classes x 10M tokens).
//   Cypha parts used: create_fresh_model_root / Cypha::load, CyphaDifMemoryState::memory_train,
//   class_mean_and_variance, batch_encode, contrastive_update_encoder_w, regression::ridge_fit_bias (a check).
//   Written here: the slot swapping and negative-sampling table, the context vectors, a per-class diagonal
//   variance (EMA of the squared residual; the core keeps only the shared v0), the optional 1/(n+1) step
//   schedule (--sched inv), KL / Fisher-Rao distances, the vector-output Bayesian ridge map, the WordNet reader,
//   and the count baselines accumulated from the same observations.
//
// Evaluation, all on the same pairs (both words among the 10,000 classes and in the PPMI-SVD vocabulary):
//   word similarity   WordSim-353, SimLex-999, MEN; Spearman rho
//   hypernymy         Open English WordNet direct noun hypernym pairs; direction by KL asymmetry, total
//                     variance and |delta|, against "more frequent word = hypernym"
//   relations         capital-of, country -> language; leave-one-out ridge map and mean offset, @1 / @10,
//                     abstention on the ridge predictive variance
//   baselines         random indexing (plain, centred, f^-1/2 weighted, v0-whitened), a count Gaussian of
//                     the same observations, and the C1 PPMI-SVD vectors (native/tools/ppmi_svd.cpp)
//
// Measured 2026-10-27 (enwik8, 9.87M class tokens, one pass). Report: docs/reports/C5_WORDS_AS_CLASSES.md
//   WordSim / SimLex / MEN on 259 / 651 / 1316 pairs:
//     Cypha defaults, frozen encoder     cos(mu) 0.017 / 0.116 / 0.137
//     Cypha, 1/(n+1) step, world_lr 5e-5 cos(mu) 0.141 / -0.048 / 0.136; whitened cos(delta) 0.240 / 0.134 / 0.372
//     Cypha defaults + contrastive enc.  cos(mu) 0.093 / -0.004 / 0.103
//     random indexing, centred           0.208 / 0.066 / 0.223 (whitened by the same v0: 0.256 / 0.101 / 0.358)
//     PPMI-SVD 300 d                     0.723 / 0.265 / 0.732
//   Hypernym direction, 13,722 pairs: frequency 60.6%; Cypha KL 46.9-53.8%, variance 50.5-53.8%.
//   Capital-of (26 pairs), ridge @10: Cypha 0-11.5%, random indexing 7.7%, PPMI-SVD 80.8%.
//   Cost: 82 s training (8.3 us/step, frozen encoder) or 806 s (contrastive encoder); peak RAM 343 MB;
//   model 20.5 MB (float32 differentials + variances).
//
// Subcommands: only "s1" (this experiment) exists, and it is also the default when the first argument is a
// path. train / distill / eval / export (docs/EMBEDDING_GEOMETRY.md §9.3) are to be added to the table in main().
//
// Usage:
//   cypha_word_classes [s1] <enwik8> <EN-WS-353-ALL.txt> <EN-SIMLEX-999.txt> <EN-MEN-TR-3k.txt>
//       <ppmi.vectors.f32> <ppmi.vocab.tsv> <wordnet_yaml_dir> [options] > report.md 2> log
//   <wordnet_yaml_dir> holds src/yaml/noun.*.yaml from github.com/globalwordnet/english-wordnet.
//   options: --classes 10000 --ctx-vocab 50000 --dim 256 --nnz 8 --window 5 --neg 5 --passes 1 --seed 1
//            --delta-lr 0.05 --world-lr 0.008 --enc-lr 0.002 (0 freezes the encoder) --sched fixed|inv
//            --lr-floor 0.002 --lr-cap 0.1 --obs plain|weighted --max-tokens 0 --no-bench --tag NAME
//   Runs in the report: defaults with --enc-lr 0; --enc-lr 0 --sched inv --world-lr 0.00005 --no-bench;
//   defaults --no-bench (contrastive encoder on).
//
// Build (from the repo root, against the already built core library):
//   g++ -O3 -DNDEBUG -std=c++20 -msse4.1 -fopenmp -DCYPHA_ACCEL_GPU_MIN_BATCH_ROWS=1 -DCYPHA_HP_GATE24=1
//       -DCYPHA_HP_SLOT_MAX=24 -DCYPHA_LLM_ALGORITHM_HP=1 -DHP_SLOT_MAX=24 -DHP_XSIMD=1
//       -Inative/third_party/hp/include -Inative/third_party/hp/third_party/xsimd/include -Inative/include
//       -Inative/third_party native/tools/cypha_word_classes.cpp /root/cb_core/libcypha_core.a -lpthread
//       -o cypha_word_classes


#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <strings.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

#include "cypha/create_model.hpp"
#include "cypha/cypha.hpp"
#include "cypha/encoder_contrastive.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/regression.hpp"

namespace {

using Clock = std::chrono::steady_clock;
const Clock::time_point g_t0 = Clock::now();
double elapsed() { return std::chrono::duration<double>(Clock::now() - g_t0).count(); }
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

long peak_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmHWM:", 0) == 0) return std::atol(line.c_str() + 6) / 1024;
    return -1;
}

void say(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void say(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[%7.1fs %5ldMB] %s\n", elapsed(), peak_rss_mb(), buf);
}

constexpr uint16_t kSep = 0xFFFF;
constexpr uint32_t kSep32 = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------------------------
// 1. Cleaning and tokenising: copied verbatim from native/tools/ppmi_svd.cpp (lines 107-437), so the
//    token stream is the one the PPMI-SVD vectors were built on.
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



// Spearman rank correlation, copied verbatim from native/tools/ppmi_svd.cpp (lines 980-1014).
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


// ---------------------------------------------------------------------------------------------
// 2. Options, vocabulary, stream, index vectors
// ---------------------------------------------------------------------------------------------

struct Options {
    std::string enwik8, ws, simlex, men, ppmi_vec, ppmi_vocab, wn_dir;
    int classes = 10000;     // words that are classes (top by frequency)
    int ctx_vocab = 50000;   // words kept in the stream (the rest are deleted before windowing, as in C1)
    int dim = 256;
    int nnz = 8;             // nonzero +-1 entries per index vector
    int window = 5;
    int neg = 5;             // sampled competitor classes per observation
    int passes = 1;
    int seed = 1;
    double delta_lr = 0.05;  // Cypha library default for the class differential
    double world_lr = 0.008; // Cypha library default
    double enc_lr = 0.002;   // Cypha library default; 0 freezes the encoder
    std::string sched = "fixed";  // fixed | inv   (inv: delta_lr = clamp(1/(n_k+1), lr_floor, lr_cap))
    std::string obs = "plain";    // plain | weighted   (weighted: each context index vector scaled by f^-1/2; an ablation)
    double lr_floor = 0.002, lr_cap = 0.1;
    long long max_tokens = 0;
    bool bench_full = true;
    bool quiet_probe = false;
    std::string tag = "run";
};

struct Vocab {
    std::vector<std::string> words;  // by rank
    std::vector<uint64_t> freq;      // by rank
    std::unordered_map<std::string, int> rank;
};

std::vector<uint16_t> build_stream(const Corpus& C, int V, Vocab& voc, uint64_t& total, uint64_t& kept) {
    std::vector<uint32_t> ord(C.words.size());
    std::iota(ord.begin(), ord.end(), 0u);
    std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
        if (C.freq[a] != C.freq[b]) return C.freq[a] > C.freq[b];
        return C.words[a] < C.words[b];
    });
    const int Vn = std::min<int>(V, (int)ord.size());
    std::vector<int32_t> rank_of(C.words.size(), -1);
    for (int r = 0; r < Vn; ++r) {
        rank_of[ord[r]] = r;
        voc.words.push_back(C.words[ord[r]]);
        voc.freq.push_back(C.freq[ord[r]]);
        voc.rank.emplace(C.words[ord[r]], r);
    }
    std::vector<uint16_t> tok;
    tok.reserve(C.toks.size());
    total = kept = 0;
    for (uint32_t t : C.toks) {
        if (t == kSep32) {
            if (!tok.empty() && tok.back() != kSep) tok.push_back(kSep);
            continue;
        }
        ++total;
        const int r = rank_of[t];
        if (r >= 0) {
            tok.push_back((uint16_t)r);
            ++kept;
        }
    }
    return tok;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL) {}
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double u01() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

struct IndexVecs {
    int nnz = 0;
    std::vector<uint8_t> pos;
    std::vector<int8_t> sgn;
};

IndexVecs make_index(int V, int d, int nnz, uint64_t seed) {
    IndexVecs iv;
    iv.nnz = nnz;
    iv.pos.resize((size_t)V * nnz);
    iv.sgn.resize((size_t)V * nnz);
    for (int r = 0; r < V; ++r) {
        Rng g(seed * 1000003ULL + (uint64_t)r);
        for (int k = 0; k < nnz; ++k) {
            uint8_t p;
            bool dup;
            do {
                p = (uint8_t)(g.next() % (uint64_t)d);
                dup = false;
                for (int q = 0; q < k; ++q) dup = dup || iv.pos[(size_t)r * nnz + q] == p;
            } while (dup);
            iv.pos[(size_t)r * nnz + k] = p;
            iv.sgn[(size_t)r * nnz + k] = (g.next() & 1) ? 1 : -1;
        }
    }
    return iv;
}

// Observation for the occurrence at tok[i]: sum over words within +-W (same line) of
// (1/distance) * indexvector(word) [* scale[word] when scale != nullptr]. The word itself is excluded.
void build_ctx(const std::vector<uint16_t>& tok, size_t i, const IndexVecs& iv, int d, int W, const float* scale,
               double* x) {
    std::fill(x, x + d, 0.0);
    auto add = [&](uint16_t c, double w) {
        if (scale) w *= scale[c];
        const uint8_t* p = &iv.pos[(size_t)c * iv.nnz];
        const int8_t* s = &iv.sgn[(size_t)c * iv.nnz];
        for (int t = 0; t < iv.nnz; ++t) x[p[t]] += w * s[t];
    };
    for (int k = 1; k <= W; ++k) {
        if (i < (size_t)k) break;
        const uint16_t c = tok[i - k];
        if (c == kSep) break;
        add(c, 1.0 / k);
    }
    for (int k = 1; k <= W; ++k) {
        if (i + k >= tok.size()) break;
        const uint16_t c = tok[i + k];
        if (c == kSep) break;
        add(c, 1.0 / k);
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Cypha training with sampled competitors
// ---------------------------------------------------------------------------------------------

struct Model {
    int d = 0, K = 0;
    // Cypha state
    std::vector<double> D;          // K x d   ClassDifferential (the master copy; Cypha's own update writes it)
    std::vector<double> nobs;       // K       Cypha n_obs
    std::vector<double> world_mu, world_v;  // WorldPrior after training
    std::vector<double> enc_w;      // d x d final encoder
    // written here (not Cypha): per-class diagonal variance, EMA of squared residual about the Cypha mean
    std::vector<float> pcvar;       // K x d
    // count baselines accumulated from the same observations
    std::vector<double> S, Sw, S2;  // K x d: sum of obs, sum of f^-1/2-weighted obs, sum of squares
    std::vector<double> cnt;        // K
    // run statistics
    double train_s = 0;
    long long steps = 0, correct = 0, enc_updates = 0;
    double enc_drift = 0;
};

Model train_cypha(const std::vector<uint16_t>& tok, const Vocab& voc, const IndexVecs& iv, const Options& O) {
    const int d = O.dim, K = O.classes, Kp = 1 + O.neg;
    Model M;
    M.d = d;
    M.K = K;
    M.D.assign((size_t)K * d, 0.0);
    M.nobs.assign(K, 0.0);
    M.pcvar.assign((size_t)K * d, 0.f);
    M.S.assign((size_t)K * d, 0.0);
    M.Sw.assign((size_t)K * d, 0.0);
    M.S2.assign((size_t)K * d, 0.0);
    M.cnt.assign(K, 0.0);

    // ---- one Cypha object: a fresh .cypha, identity encoder, d-dimensional latent -----------------
    cypha::Cypha cy;
    {
        cypha::FreshModelParams fp;
        fp.input_dim = d;
        fp.field_dim = 16;
        fp.temperature = 1.0;
        fp.world_lr = O.world_lr;
        fp.delta_lr = O.delta_lr;
        const cypha::CNode root = cypha::create_fresh_model_root(fp);
        const auto path = (std::filesystem::temp_directory_path() / ("cwc_fresh_" + std::to_string(::getpid()) + ".cypha")).string();
        cypha::save_cypha_file(path.c_str(), root);
        const bool ok = cy.load(path);
        std::error_code ec;
        std::filesystem::remove(path, ec);
        if (!ok) {
            std::fprintf(stderr, "cannot load fresh model\n");
            std::exit(3);
        }
    }
    cypha::CyphaDifMemoryState* mem = cy.mem();
    cypha::CyphaInferModel* infer = cy.infer();
    // The scratch memory holds the true class plus `neg` sampled competitors in Kp slots. Cypha's memory_train
    // scores, repels and shrinks exactly these slots; rows are swapped in from / out to the master copy.
    mem->labels.clear();
    mem->label_index.clear();
    for (int s = 0; s < Kp; ++s) {
        mem->labels.push_back("S" + std::to_string(s));
        mem->label_index["S" + std::to_string(s)] = s;
    }
    mem->D.assign((size_t)Kp * d, 0.0);
    mem->n_obs_buf.assign(Kp, 0.0);
    mem->n_correct.assign(Kp, 0);
    mem->use_class_gmm = false;

    // ---- negative-sampling table, unigram^0.75 over the classes ------------------------------------
    const size_t TB = 1u << 22;
    std::vector<uint16_t> table(TB);
    {
        double z = 0;
        for (int k = 0; k < K; ++k) z += std::pow((double)voc.freq[k], 0.75);
        double acc = std::pow((double)voc.freq[0], 0.75) / z;
        int k = 0;
        for (size_t a = 0; a < TB; ++a) {
            while ((double)a / (double)TB > acc && k < K - 1) {
                ++k;
                acc += std::pow((double)voc.freq[k], 0.75) / z;
            }
            table[a] = (uint16_t)k;
        }
    }
    // f^-1/2 context scale for the weighted random-indexing baseline
    std::vector<float> wscale(voc.words.size());
    for (size_t r = 0; r < voc.words.size(); ++r) wscale[r] = (float)(1.0 / std::sqrt((double)voc.freq[r]) * 100.0);

    Rng rng((uint64_t)O.seed);
    std::vector<double> x(d), xw(d), H(d), mu_pre(d);
    std::vector<double> mu_k, v_k, mu_j, v_j;
    const std::unordered_map<std::string, double> no_ctx;
    std::vector<int> slot_cls(Kp);
    const size_t N = (O.max_tokens > 0) ? std::min<size_t>(tok.size(), (size_t)O.max_tokens) : tok.size();
    const double temp = infer->temperature;
    const bool wobs = (O.obs == "weighted");
    int enc_count = 0;
    const auto t_train = Clock::now();

    for (int pass = 0; pass < O.passes; ++pass) {
        for (size_t i = 0; i < N; ++i) {
            const uint16_t t = tok[i];
            if (t == kSep || t >= K) continue;
            build_ctx(tok, i, iv, d, O.window, nullptr, x.data());
            if (pass == 0 || wobs) build_ctx(tok, i, iv, d, O.window, wscale.data(), xw.data());
            if (pass == 0) {  // count baselines see each observation once
                double* s = &M.S[(size_t)t * d];
                double* sw = &M.Sw[(size_t)t * d];
                double* s2 = &M.S2[(size_t)t * d];
                for (int j = 0; j < d; ++j) {
                    s[j] += x[j];
                    sw[j] += xw[j];
                    s2[j] += x[j] * x[j];
                }
                M.cnt[t] += 1;
            }
            // slots: true class in a random slot, competitors from unigram^0.75
            const int st = (int)(rng.next() % (uint64_t)Kp);
            slot_cls[st] = t;
            for (int s = 0; s < Kp; ++s) {
                if (s == st) continue;
                int c;
                bool dup;
                do {
                    c = table[rng.next() & (TB - 1)];
                    dup = (c == t);
                    for (int q = 0; q < s && !dup; ++q) dup = (q != st && slot_cls[q] == c);
                } while (dup);
                slot_cls[s] = c;
            }
            for (int s = 0; s < Kp; ++s) {
                std::memcpy(&mem->D[(size_t)s * d], &M.D[(size_t)slot_cls[s] * d], sizeof(double) * d);
                mem->n_obs_buf[s] = M.nobs[slot_cls[s]];
            }
            // latent: h = W x (Cypha's VectorEncoder). With enc_lr = 0 the encoder stays the identity.
            if (O.enc_lr > 0) {
                cypha::batch_encode(*infer, wobs ? xw.data() : x.data(), 1, H);
            } else {
                H = wobs ? xw : x;
            }
            // residual about the pre-update class mean, for the per-class variance written here
            const double* Dt = &mem->D[(size_t)st * d];
            for (int j = 0; j < d; ++j) mu_pre[j] = mem->world_mu[j] + Dt[j];
            const double n_pre = M.nobs[t];
            double dlr = O.delta_lr;
            if (O.sched == "inv") dlr = std::min(O.lr_cap, std::max(O.lr_floor, 1.0 / (n_pre + 1.0)));

            cypha::MemoryTrainMeta meta;
            // Cypha: score the slots, natural-gradient update of the true class's differential, repulsion of the
            // competitors (softmax-weighted, capped), MDL shrinkage of every slot, WorldPrior update.
            (void)mem->memory_train(H.data(), mem->labels[st], nullptr, no_ctx, temp, 15.0, O.world_lr, dlr, &meta);
            ++M.steps;
            M.correct += meta.correct ? 1 : 0;
            if (O.enc_lr > 0 && !meta.correct && meta.pred_label != mem->labels[st]) {
                if (mem->class_mean_and_variance(mem->labels[st], mu_k, v_k) &&
                    mem->class_mean_and_variance(meta.pred_label, mu_j, v_j)) {
                    // Cypha: Fisher-Rao contrastive encoder update (true class vs. the confusing competitor)
                    cypha::contrastive_update_encoder_w(infer->enc_w, d, wobs ? xw.data() : x.data(), H.data(), mu_k.data(), v_k.data(),
                                                        mu_j.data(), v_j.data(), 1.0, O.enc_lr, enc_count);
                    ++M.enc_updates;
                }
            }
            // write the slots back
            for (int s = 0; s < Kp; ++s) {
                std::memcpy(&M.D[(size_t)slot_cls[s] * d], &mem->D[(size_t)s * d], sizeof(double) * d);
                M.nobs[slot_cls[s]] = mem->n_obs_buf[s];
            }
            // per-class variance (written here): v <- (1-eta) v + eta * (h - mu_pre)^2, initialised at the WorldPrior v
            float* pv = &M.pcvar[(size_t)t * d];
            if (n_pre == 0) {
                for (int j = 0; j < d; ++j) pv[j] = (float)mem->world_v[j];
            }
            const double eta = (O.sched == "inv") ? std::max(dlr, 1.0 / (n_pre + 1.0)) : dlr;
            for (int j = 0; j < d; ++j) {
                const double r = H[j] - mu_pre[j];
                pv[j] = (float)((1.0 - eta) * pv[j] + eta * r * r);
            }
            if ((M.steps & 0x1FFFFF) == 0) {
                say("pass %d step %lld (%.1f%% of pass): train acc over %d slots %.3f, enc updates %lld, world_v mean %.4f",
                    pass, M.steps, 100.0 * (double)i / (double)N, Kp, (double)M.correct / (double)M.steps,
                    M.enc_updates, mem->world_v_mean);
            }
        }
    }
    M.train_s = since(t_train);
    M.world_mu = mem->world_mu;
    M.world_v = mem->world_v;
    M.enc_w = infer->enc_w;
    double fr = 0;
    for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) {
            const double e = M.enc_w[(size_t)a * d + b] - (a == b ? 1.0 : 0.0);
            fr += e * e;
        }
    M.enc_drift = std::sqrt(fr);
    return M;
}

// ---------------------------------------------------------------------------------------------
// 4. Full public-API cost probe (why sampled competitors are needed)
// ---------------------------------------------------------------------------------------------

void bench_full_api(int d) {
    std::printf("\n## Public API cost: Cypha::update scores every class\n\n| classes K | us per update (public Cypha::update) | extrapolated to 10,000 classes x 10M tokens |\n|---|---|---|\n");
    for (int K : {100, 400, 1600}) {
        cypha::Cypha cy;
        cypha::FreshModelParams fp;
        fp.input_dim = d;
        fp.field_dim = 16;
        const cypha::CNode root = cypha::create_fresh_model_root(fp);
        const auto path = (std::filesystem::temp_directory_path() / ("cwc_bench_" + std::to_string(::getpid()) + ".cypha")).string();
        cypha::save_cypha_file(path.c_str(), root);
        if (!cy.load(path)) continue;
        std::error_code ec;
        std::filesystem::remove(path, ec);
        cy.train_params().enc_lr = 0.0;
        Rng g(7);
        std::vector<double> x(d);
        auto rnd = [&]() {
            for (double& v : x) v = (g.u01() - 0.5);
        };
        for (int k = 0; k < K; ++k) {
            rnd();
            const std::string lb = "w" + std::to_string(k);
            cy.update(x.data(), d, &lb, nullptr, {});
        }
        const int reps = K >= 1600 ? 60 : 200;
        const auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            rnd();
            const std::string lb = "w" + std::to_string((int)(g.next() % (uint64_t)K));
            cy.update(x.data(), d, &lb, nullptr, {});
        }
        const double us = since(t0) / reps * 1e6;
        const double ext_h = us * (10000.0 / K) * 1e7 / 1e6 / 3600.0;
        std::printf("| %d | %.0f | %.0f hours |\n", K, us, ext_h);
    }
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------------
// 5. Evaluation helpers
// ---------------------------------------------------------------------------------------------

struct PairSet {
    std::string name;
    std::vector<std::array<int, 2>> pr;  // ranks (covered pairs only)
    std::vector<double> gold;
    int total = 0;
};

PairSet read_pairs(const std::string& path, const std::string& name, const Vocab& voc, int K,
                   const std::vector<int>& ppmi_row) {
    PairSet P;
    P.name = name;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string a, b;
        double g;
        if (!(ss >> a >> b >> g)) continue;
        ++P.total;
        for (char& c : a) c = (char)std::tolower((unsigned char)c);
        for (char& c : b) c = (char)std::tolower((unsigned char)c);
        auto ia = voc.rank.find(a), ib = voc.rank.find(b);
        if (ia == voc.rank.end() || ib == voc.rank.end()) continue;
        if (ia->second >= K || ib->second >= K) continue;
        if (ppmi_row[ia->second] < 0 || ppmi_row[ib->second] < 0) continue;
        P.pr.push_back({ia->second, ib->second});
        P.gold.push_back(g);
    }
    return P;
}

double cosine(const float* a, const float* b, int d) {
    double ab = 0, aa = 0, bb = 0;
    for (int j = 0; j < d; ++j) {
        ab += (double)a[j] * b[j];
        aa += (double)a[j] * a[j];
        bb += (double)b[j] * b[j];
    }
    return ab / (std::sqrt(aa * bb) + 1e-30);
}

double rho_of(const PairSet& P, const std::function<double(int, int)>& sim) {
    std::vector<double> s(P.pr.size());
    for (size_t i = 0; i < P.pr.size(); ++i) s[i] = sim(P.pr[i][0], P.pr[i][1]);
    return spearman(s, P.gold);
}

// Diagonal-Gaussian divergences. v is a float variance vector, m a double mean.
double kl_diag(const double* ma, const float* va, const double* mb, const float* vb, int d) {
    double s = 0;
    for (int j = 0; j < d; ++j) {
        const double A = std::max((double)va[j], 1e-4), B = std::max((double)vb[j], 1e-4);
        const double dm = ma[j] - mb[j];
        s += std::log(B / A) + (A + dm * dm) / B - 1.0;
    }
    return 0.5 * s;
}

// Fisher-Rao distance between diagonal Gaussians: product of hyperbolic planes (metric (dmu^2 + 2 dsigma^2)/sigma^2):
// d_j = sqrt(2) * arccosh(1 + (dmu^2/2 + (s1-s2)^2) / (2 s1 s2)), d = sqrt(sum_j d_j^2).
double fr_diag(const double* ma, const float* va, const double* mb, const float* vb, int d) {
    double s = 0;
    for (int j = 0; j < d; ++j) {
        const double s1 = std::sqrt(std::max((double)va[j], 1e-4)), s2 = std::sqrt(std::max((double)vb[j], 1e-4));
        const double dm = ma[j] - mb[j];
        const double arg = 1.0 + (0.5 * dm * dm + (s1 - s2) * (s1 - s2)) / (2.0 * s1 * s2);
        const double a = std::acosh(std::max(arg, 1.0));
        s += 2.0 * a * a;
    }
    return std::sqrt(s);
}

double auroc(const std::vector<double>& score, const std::vector<char>& pos) {
    const size_t n = score.size();
    const std::vector<double> r = rank_avg(score);
    double np = 0, nn = 0, sr = 0;
    for (size_t i = 0; i < n; ++i) {
        if (pos[i]) {
            ++np;
            sr += r[i];
        } else {
            ++nn;
        }
    }
    if (np == 0 || nn == 0) return std::nan("");
    return (sr - np * (np + 1) / 2.0) / (np * nn);
}

// ---- small dense linear algebra for the relation maps (n <= ~100) ---------------------------------

bool chol_inplace(std::vector<double>& A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[(size_t)i * n + j];
            for (int k = 0; k < j; ++k) s -= A[(size_t)i * n + k] * A[(size_t)j * n + k];
            if (i == j) {
                if (s <= 1e-300) return false;
                A[(size_t)i * n + i] = std::sqrt(s);
            } else {
                A[(size_t)i * n + j] = s / A[(size_t)j * n + j];
            }
        }
    }
    return true;
}

// Solves L L^T X = B for X (n x m, row-major, in place). L from chol_inplace.
void chol_solve(const std::vector<double>& L, int n, std::vector<double>& B, int m) {
    for (int c = 0; c < m; ++c) {
        for (int i = 0; i < n; ++i) {
            double s = B[(size_t)i * m + c];
            for (int k = 0; k < i; ++k) s -= L[(size_t)i * n + k] * B[(size_t)k * m + c];
            B[(size_t)i * m + c] = s / L[(size_t)i * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = B[(size_t)i * m + c];
            for (int k = i + 1; k < n; ++k) s -= L[(size_t)k * n + i] * B[(size_t)k * m + c];
            B[(size_t)i * m + c] = s / L[(size_t)i * n + i];
        }
    }
}

// Bayesian ridge in the dual (n << d): y = A x + b + noise, A ~ N(0, tau^2), noise ~ N(0, sigma^2), lambda = sigma^2/tau^2.
// Inputs and outputs are centred on the training means, so b is unpenalised. lambda is chosen from a grid by the closed-form
// leave-one-out error of kernel ridge (inner loop). Predictive variance factor for x*:
//   u(x*) = 1 + 1/n + (|xc|^2 - k*^T (K + lambda I)^-1 k*) / lambda      (predictive variance = sigma^2 * u).
struct RidgeMap {
    int n = 0, d = 0, dout = 0;
    double lambda = 0;
    std::vector<double> xbar, ybar, Xc, alpha, L;  // Xc n x d, alpha n x dout, L chol of K + lambda I
    bool fit(const std::vector<const float*>& X, const std::vector<const float*>& Y, int din, int dy) {
        n = (int)X.size();
        d = din;
        dout = dy;
        xbar.assign(d, 0.0);
        ybar.assign(dout, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) xbar[j] += X[i][j] / n;
            for (int j = 0; j < dout; ++j) ybar[j] += Y[i][j] / n;
        }
        Xc.assign((size_t)n * d, 0.0);
        std::vector<double> Yc((size_t)n * dout);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) Xc[(size_t)i * d + j] = X[i][j] - xbar[j];
            for (int j = 0; j < dout; ++j) Yc[(size_t)i * dout + j] = Y[i][j] - ybar[j];
        }
        std::vector<double> Kmat((size_t)n * n, 0.0);
        double tr = 0;
        for (int a = 0; a < n; ++a)
            for (int b = 0; b <= a; ++b) {
                double s = 0;
                for (int j = 0; j < d; ++j) s += Xc[(size_t)a * d + j] * Xc[(size_t)b * d + j];
                Kmat[(size_t)a * n + b] = Kmat[(size_t)b * n + a] = s;
                if (a == b) tr += s;
            }
        const double scale = tr / n + 1e-12;
        double best = 1e300;
        for (double g : {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0}) {
            const double lam = g * scale;
            std::vector<double> A = Kmat;
            for (int i = 0; i < n; ++i) A[(size_t)i * n + i] += lam;
            if (!chol_inplace(A, n)) continue;
            std::vector<double> Minv((size_t)n * n, 0.0);
            for (int i = 0; i < n; ++i) Minv[(size_t)i * n + i] = 1.0;
            chol_solve(A, n, Minv, n);
            // hat = K Minv ; fitted = hat Yc ; leave-one-out residual e_i / (1 - hat_ii)
            std::vector<double> Hat((size_t)n * n, 0.0);
            for (int i = 0; i < n; ++i)
                for (int m = 0; m < n; ++m) {
                    const double kim = Kmat[(size_t)i * n + m];
                    for (int l = 0; l < n; ++l) Hat[(size_t)i * n + l] += kim * Minv[(size_t)m * n + l];
                }
            double err = 0;
            for (int i = 0; i < n; ++i) {
                const double h = Hat[(size_t)i * n + i];
                for (int c = 0; c < dout; ++c) {
                    double f = 0;
                    for (int l = 0; l < n; ++l) f += Hat[(size_t)i * n + l] * Yc[(size_t)l * dout + c];
                    const double e = (Yc[(size_t)i * dout + c] - f) / std::max(1.0 - h, 1e-6);
                    err += e * e;
                }
            }
            if (err < best) {
                best = err;
                lambda = lam;
            }
        }
        std::vector<double> A = Kmat;
        for (int i = 0; i < n; ++i) A[(size_t)i * n + i] += lambda;
        if (!chol_inplace(A, n)) return false;
        L = A;
        alpha = Yc;
        chol_solve(L, n, alpha, dout);
        return true;
    }
    void predict(const float* x, std::vector<double>& out, double& u) const {
        std::vector<double> xc(d), k(n);
        double xx = 0;
        for (int j = 0; j < d; ++j) {
            xc[j] = x[j] - xbar[j];
            xx += xc[j] * xc[j];
        }
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int j = 0; j < d; ++j) s += Xc[(size_t)i * d + j] * xc[j];
            k[i] = s;
        }
        out.assign(ybar.begin(), ybar.end());
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < dout; ++c) out[c] += k[i] * alpha[(size_t)i * dout + c];
        std::vector<double> kk = k;
        chol_solve(L, n, kk, 1);
        double q = 0;
        for (int i = 0; i < n; ++i) q += k[i] * kk[i];
        u = 1.0 + 1.0 / n + std::max(xx - q, 0.0) / lambda;
    }
};

// ---- relations --------------------------------------------------------------------------------------

struct RelItem {
    int subj, obj;
    int rank_ridge, rank_off;
    double u;          // predictive variance factor of the ridge map
    double margin;     // cosine of the top candidate (retrieval confidence, an alternative abstention score)
};

int rank_of_target(const float* q, const std::vector<float>& E, int dim, int K, int target, int exclude) {
    // 1-based rank of `target` by cosine to q among candidates 0..K-1 except `exclude`
    const double st = cosine(q, &E[(size_t)target * dim], dim);
    int r = 1;
    for (int c = 0; c < K; ++c) {
        if (c == exclude || c == target) continue;
        if (cosine(q, &E[(size_t)c * dim], dim) > st) ++r;
    }
    return r;
}

// E: K x dim, rows already unit length.
std::vector<RelItem> run_relation(const std::vector<std::pair<int, int>>& P, const std::vector<float>& E, int dim, int K,
                                  double* lam_mean) {
    std::vector<RelItem> out;
    double lsum = 0;
    for (size_t j = 0; j < P.size(); ++j) {
        std::vector<const float*> X, Y;
        std::vector<double> off(dim, 0.0);
        int ntr = 0;
        for (size_t i = 0; i < P.size(); ++i) {
            if (i == j) continue;
            X.push_back(&E[(size_t)P[i].first * dim]);
            Y.push_back(&E[(size_t)P[i].second * dim]);
            ++ntr;
        }
        for (int i = 0; i < ntr; ++i)
            for (int c = 0; c < dim; ++c) off[c] += ((double)Y[i][c] - X[i][c]) / ntr;
        RelItem it{};
        it.subj = P[j].first;
        it.obj = P[j].second;
        const float* xs = &E[(size_t)it.subj * dim];
        // mean offset (as scripts/gguf_mine.py): q = unit(x + mean(y - x))
        std::vector<float> q(dim);
        for (int c = 0; c < dim; ++c) q[c] = (float)(xs[c] + off[c]);
        it.rank_off = rank_of_target(q.data(), E, dim, K, it.obj, it.subj);
        // Bayesian ridge map
        RidgeMap R;
        if (R.fit(X, Y, dim, dim)) {
            std::vector<double> yp;
            R.predict(xs, yp, it.u);
            for (int c = 0; c < dim; ++c) q[c] = (float)yp[c];
            it.rank_ridge = rank_of_target(q.data(), E, dim, K, it.obj, it.subj);
            // confidence = cosine of the top candidate
            double top = -2;
            for (int c = 0; c < K; ++c) {
                if (c == it.subj) continue;
                top = std::max(top, cosine(q.data(), &E[(size_t)c * dim], dim));
            }
            it.margin = top;
            lsum += R.lambda;
        } else {
            it.rank_ridge = K;
            it.u = 1e9;
            it.margin = -1;
        }
        out.push_back(it);
    }
    if (lam_mean) *lam_mean = lsum / std::max<size_t>(1, P.size());
    return out;
}

// ---------------------------------------------------------------------------------------------
// 6. WordNet hypernym pairs from Open English WordNet YAML (globalwordnet/english-wordnet, src/yaml/noun.*.yaml)
// ---------------------------------------------------------------------------------------------

struct Syn {
    std::vector<std::string> members, hyper;
};

std::string strip_yaml(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ')) s.pop_back();
    if (s.size() >= 2 && (s.front() == '\'' || s.front() == '"') && s.back() == s.front()) s = s.substr(1, s.size() - 2);
    return s;
}

std::unordered_map<std::string, Syn> load_wordnet(const std::string& dir) {
    std::unordered_map<std::string, Syn> W;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        const std::string fn = e.path().filename().string();
        if (fn.rfind("noun.", 0) != 0 || fn.size() < 5 || fn.substr(fn.size() - 5) != ".yaml") continue;
        std::ifstream in(e.path());
        std::string line, cur;
        int mode = 0;  // 1 members, 2 hypernym
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (line[0] != ' ') {  // synset id line "01234567-n:"
                if (line.size() > 3 && line.back() == ':') {
                    cur = line.substr(0, line.size() - 1);
                    W[cur];
                    mode = 0;
                }
                continue;
            }
            if (line.rfind("  members:", 0) == 0) { mode = 1; continue; }
            if (line.rfind("  hypernym:", 0) == 0) { mode = 2; continue; }
            if (line.rfind("  - ", 0) == 0) {
                if (mode == 1) W[cur].members.push_back(strip_yaml(line.substr(4)));
                else if (mode == 2) W[cur].hyper.push_back(strip_yaml(line.substr(4)));
                continue;
            }
            if (line.size() > 2 && line[2] != ' ' && line[2] != '-') mode = 0;  // another key
        }
    }
    return W;
}

bool plain_word(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!(c >= 'a' && c <= 'z')) return false;
    return true;
}

// ---------------------------------------------------------------------------------------------
// 7. Relation word lists (single-token words; pairs with a word outside the top classes are dropped at run time)
// ---------------------------------------------------------------------------------------------

const char* kCapitalOf[][2] = {
    {"france", "paris"}, {"germany", "berlin"}, {"italy", "rome"}, {"spain", "madrid"}, {"japan", "tokyo"},
    {"china", "beijing"}, {"russia", "moscow"}, {"egypt", "cairo"}, {"greece", "athens"}, {"canada", "ottawa"},
    {"poland", "warsaw"}, {"austria", "vienna"}, {"portugal", "lisbon"}, {"ireland", "dublin"}, {"sweden", "stockholm"},
    {"norway", "oslo"}, {"denmark", "copenhagen"}, {"finland", "helsinki"}, {"hungary", "budapest"}, {"turkey", "ankara"},
    {"iran", "tehran"}, {"iraq", "baghdad"}, {"india", "delhi"}, {"thailand", "bangkok"}, {"indonesia", "jakarta"},
    {"peru", "lima"}, {"cuba", "havana"}, {"kenya", "nairobi"}, {"libya", "tripoli"}, {"syria", "damascus"},
    {"lebanon", "beirut"}, {"jordan", "amman"}, {"afghanistan", "kabul"}, {"pakistan", "islamabad"},
    {"bangladesh", "dhaka"}, {"vietnam", "hanoi"}, {"philippines", "manila"}, {"australia", "canberra"},
    {"venezuela", "caracas"}, {"chile", "santiago"}, {"colombia", "bogota"}, {"switzerland", "bern"},
    {"belgium", "brussels"}, {"netherlands", "amsterdam"}, {"romania", "bucharest"}, {"bulgaria", "sofia"},
    {"serbia", "belgrade"}, {"croatia", "zagreb"}, {"ukraine", "kiev"}, {"scotland", "edinburgh"}, {"wales", "cardiff"},
    {"iceland", "reykjavik"}, {"morocco", "rabat"}, {"algeria", "algiers"}, {"tunisia", "tunis"}, {"nepal", "kathmandu"},
    {"mongolia", "ulaanbaatar"}, {"laos", "vientiane"}, {"burma", "rangoon"}, {"korea", "seoul"},
    {"taiwan", "taipei"}, {"israel", "jerusalem"}, {"albania", "tirana"}, {"latvia", "riga"}, {"lithuania", "vilnius"},
    {"estonia", "tallinn"}, {"slovakia", "bratislava"}, {"slovenia", "ljubljana"}, {"ecuador", "quito"},
    {"uruguay", "montevideo"}, {"paraguay", "asuncion"}, {"nicaragua", "managua"}, {"jamaica", "kingston"},
    {"ghana", "accra"}, {"nigeria", "abuja"}, {"senegal", "dakar"}, {"sudan", "khartoum"},
    {"somalia", "mogadishu"}, {"zimbabwe", "harare"}, {"zambia", "lusaka"}, {"angola", "luanda"}, {"uganda", "kampala"},
    {"tanzania", "dodoma"}, {"mali", "bamako"}, {"yemen", "sanaa"}, {"oman", "muscat"}, 
    {"qatar", "doha"}, {"cyprus", "nicosia"}, {"malta", "valletta"}, {"georgia", "tbilisi"},
    {"armenia", "yerevan"}, {"azerbaijan", "baku"}, {"belarus", "minsk"}, {"moldova", "chisinau"}, {"macedonia", "skopje"},
};

const char* kCountryLanguage[][2] = {
    {"france", "french"}, {"germany", "german"}, {"italy", "italian"}, {"spain", "spanish"}, {"japan", "japanese"},
    {"china", "chinese"}, {"russia", "russian"}, {"poland", "polish"}, {"greece", "greek"}, {"portugal", "portuguese"},
    {"netherlands", "dutch"}, {"sweden", "swedish"}, {"denmark", "danish"}, {"norway", "norwegian"},
    {"finland", "finnish"}, {"hungary", "hungarian"}, {"turkey", "turkish"}, {"iran", "persian"}, {"israel", "hebrew"},
    {"india", "hindi"}, {"korea", "korean"}, {"vietnam", "vietnamese"}, {"thailand", "thai"}, {"wales", "welsh"},
    {"ireland", "irish"}, {"iceland", "icelandic"}, {"romania", "romanian"}, {"bulgaria", "bulgarian"},
    {"croatia", "croatian"}, {"serbia", "serbian"}, {"ukraine", "ukrainian"}, {"albania", "albanian"},
    {"lithuania", "lithuanian"}, {"latvia", "latvian"}, {"estonia", "estonian"}, {"brazil", "portuguese"},
    {"mexico", "spanish"}, {"argentina", "spanish"}, {"egypt", "arabic"}, {"pakistan", "urdu"},
    {"indonesia", "indonesian"}, {"philippines", "tagalog"}, {"scotland", "gaelic"}, {"slovakia", "slovak"},
    {"slovenia", "slovene"}, {"austria", "german"}, {"belgium", "dutch"}, {"switzerland", "german"},
    {"cuba", "spanish"}, {"peru", "spanish"}, {"chile", "spanish"}, {"colombia", "spanish"}, {"syria", "arabic"},
    {"iraq", "arabic"}, {"libya", "arabic"}, {"jordan", "arabic"}, {"lebanon", "arabic"}, {"algeria", "arabic"},
    {"morocco", "arabic"}, {"tunisia", "arabic"}, {"armenia", "armenian"}, {"georgia", "georgian"},
    {"mongolia", "mongolian"}, {"nepal", "nepali"}, {"malta", "maltese"},
};

}  // namespace

// ---------------------------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------------------------

// s1: the words-as-classes experiment. argv[0] is the program or subcommand name.
int cmd_s1(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
                     "usage: %s [s1] <enwik8> <EN-WS-353-ALL.txt> <EN-SIMLEX-999.txt> <EN-MEN-TR-3k.txt> <ppmi.vectors.f32> "
                     "<ppmi.vocab.tsv> <wordnet_yaml_dir> [options; see header]\n",
                     argv[0]);
        return 2;
    }
    Options O;
    O.enwik8 = argv[1];
    O.ws = argv[2];
    O.simlex = argv[3];
    O.men = argv[4];
    O.ppmi_vec = argv[5];
    O.ppmi_vocab = argv[6];
    O.wn_dir = argv[7];
    for (int a = 8; a < argc; ++a) {
        const std::string k = argv[a];
        auto val = [&]() -> std::string { return a + 1 < argc ? std::string(argv[++a]) : std::string(); };
        if (k == "--classes") O.classes = std::atoi(val().c_str());
        else if (k == "--ctx-vocab") O.ctx_vocab = std::atoi(val().c_str());
        else if (k == "--dim") O.dim = std::atoi(val().c_str());
        else if (k == "--nnz") O.nnz = std::atoi(val().c_str());
        else if (k == "--window") O.window = std::atoi(val().c_str());
        else if (k == "--neg") O.neg = std::atoi(val().c_str());
        else if (k == "--passes") O.passes = std::atoi(val().c_str());
        else if (k == "--seed") O.seed = std::atoi(val().c_str());
        else if (k == "--delta-lr") O.delta_lr = std::atof(val().c_str());
        else if (k == "--world-lr") O.world_lr = std::atof(val().c_str());
        else if (k == "--enc-lr") O.enc_lr = std::atof(val().c_str());
        else if (k == "--sched") O.sched = val();
        else if (k == "--obs") O.obs = val();
        else if (k == "--lr-floor") O.lr_floor = std::atof(val().c_str());
        else if (k == "--lr-cap") O.lr_cap = std::atof(val().c_str());
        else if (k == "--max-tokens") O.max_tokens = std::atoll(val().c_str());
        else if (k == "--no-bench") O.bench_full = false;
        else if (k == "--tag") O.tag = val();
        else {
            std::fprintf(stderr, "unknown option %s\n", k.c_str());
            return 2;
        }
    }
    const int d = O.dim, K = O.classes;
    if (O.dim > 256) {
        std::fprintf(stderr, "dim > 256 is not supported (uint8 index positions)\n");
        return 2;
    }

    // ---- corpus ------------------------------------------------------------------------------------
    say("reading %s", O.enwik8.c_str());
    std::string raw;
    {
        std::ifstream in(O.enwik8, std::ios::binary | std::ios::ate);
        if (!in) {
            std::fprintf(stderr, "cannot open %s\n", O.enwik8.c_str());
            return 2;
        }
        raw.resize((size_t)in.tellg());
        in.seekg(0);
        in.read(&raw[0], (std::streamsize)raw.size());
    }
    const auto t_clean = Clock::now();
    Corpus C = clean_and_tokenize(raw, "");
    raw.clear();
    raw.shrink_to_fit();
    Vocab voc;
    uint64_t total = 0, kept = 0;
    std::vector<uint16_t> tok = build_stream(C, O.ctx_vocab, voc, total, kept);
    const size_t n_types = C.words.size(), n_pages = C.pages;
    C = Corpus();
    uint64_t class_tokens = 0;
    for (int r = 0; r < K && r < (int)voc.freq.size(); ++r) class_tokens += voc.freq[r];
    const double clean_s = since(t_clean);
    say("cleaned: %zu pages, %llu tokens, %zu types; kept %llu tokens in top %d words; class tokens (top %d) %llu (%.2f%%); %.1fs",
        n_pages, (unsigned long long)total, n_types, (unsigned long long)kept, O.ctx_vocab, K,
        (unsigned long long)class_tokens, 100.0 * (double)class_tokens / (double)total, clean_s);
    if ((int)voc.words.size() < K) {
        std::fprintf(stderr, "vocabulary smaller than --classes\n");
        return 2;
    }

    // ---- PPMI-SVD vectors, matched by word ---------------------------------------------------------
    std::vector<int> ppmi_row(voc.words.size(), -1);
    int ppmi_dim = 0;
    std::vector<float> ppmi;
    {
        std::ifstream vf(O.ppmi_vocab);
        std::string line;
        std::unordered_map<std::string, int> idx;
        int n = 0;
        while (std::getline(vf, line)) {
            idx.emplace(line.substr(0, line.find('\t')), n++);
        }
        std::ifstream in(O.ppmi_vec, std::ios::binary | std::ios::ate);
        if (!in || n == 0) {
            std::fprintf(stderr, "cannot read PPMI vectors\n");
            return 2;
        }
        const size_t bytes = (size_t)in.tellg();
        in.seekg(0);
        ppmi_dim = (int)(bytes / ((size_t)n * sizeof(float)));
        ppmi.resize((size_t)n * ppmi_dim);
        in.read(reinterpret_cast<char*>(ppmi.data()), (std::streamsize)bytes);
        for (size_t r = 0; r < voc.words.size(); ++r) {
            auto it = idx.find(voc.words[r]);
            if (it != idx.end()) ppmi_row[r] = it->second;
        }
        int top_match = 0;
        for (int r = 0; r < K; ++r) top_match += (ppmi_row[r] >= 0 && ppmi_row[r] < K) ? 1 : 0;
        say("PPMI-SVD: %d rows x %d dims; of my top %d words %d have a row, %d of them in the PPMI top %d", n, ppmi_dim, K,
            (int)std::count_if(ppmi_row.begin(), ppmi_row.begin() + K, [](int v) { return v >= 0; }), top_match, K);
    }

    // ---- training ----------------------------------------------------------------------------------
    const IndexVecs iv = make_index((int)voc.words.size(), d, O.nnz, (uint64_t)O.seed);
    say("training: d=%d nnz=%d window=%d classes=%d competitors=%d passes=%d delta_lr=%g(%s) world_lr=%g enc_lr=%g", d, O.nnz,
        O.window, K, O.neg, O.passes, O.delta_lr, O.sched.c_str(), O.world_lr, O.enc_lr);
    Model M = train_cypha(tok, voc, iv, O);
    say("trained: %lld steps in %.1fs (%.1f us/step), sampled-slot train accuracy %.3f, encoder updates %lld, ||W-I||_F = %.3f",
        M.steps, M.train_s, 1e6 * M.train_s / std::max<long long>(1, M.steps),
        (double)M.correct / std::max<long long>(1, M.steps), M.enc_updates, M.enc_drift);

    // ---- representations ---------------------------------------------------------------------------
    const size_t KD = (size_t)K * d;
    std::vector<float> mu(KD), dl(KD), ri_plain(KD), ri_cent(KD), ri_w(KD);  // Cypha mean, Cypha differential, RI variants
    {
        std::vector<double> m1(d, 0.0), m2(d, 0.0);
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < d; ++j) {
                const double n = std::max(1.0, M.cnt[k]);
                m1[j] += M.S[(size_t)k * d + j] / n / K;
                m2[j] += M.Sw[(size_t)k * d + j] / n / K;
            }
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < d; ++j) {
                const size_t o = (size_t)k * d + j;
                const double n = std::max(1.0, M.cnt[k]);
                dl[o] = (float)M.D[o];
                mu[o] = (float)(M.world_mu[j] + M.D[o]);
                ri_plain[o] = (float)M.S[o];
                ri_cent[o] = (float)(M.S[o] / n - m1[j]);
                ri_w[o] = (float)(M.Sw[o] / n - m2[j]);
            }
    }
    // class-conditional Gaussian parameters
    std::vector<double> mu_d(KD);
    for (size_t o = 0; o < KD; ++o) mu_d[o] = mu[o];
    std::vector<double> cm(KD);   // count-Gaussian mean
    std::vector<float> cv(KD);    // count-Gaussian variance (exact moments of the same observations)
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < d; ++j) {
            const size_t o = (size_t)k * d + j;
            const double n = std::max(1.0, M.cnt[k]);
            cm[o] = M.S[o] / n;
            cv[o] = (float)std::max(M.S2[o] / n - cm[o] * cm[o], 1e-4);
        }
    for (size_t o = 0; o < KD; ++o) M.pcvar[o] = std::max(M.pcvar[o], 1e-4f);

    // ---- word similarity ---------------------------------------------------------------------------
    std::vector<PairSet> sets;
    sets.push_back(read_pairs(O.ws, "WordSim-353", voc, K, ppmi_row));
    sets.push_back(read_pairs(O.simlex, "SimLex-999", voc, K, ppmi_row));
    sets.push_back(read_pairs(O.men, "MEN", voc, K, ppmi_row));
    std::vector<double> inv_v0(d);
    for (int j = 0; j < d; ++j) inv_v0[j] = 1.0 / std::max(M.world_v[j], 1e-4);

    struct Method {
        std::string name;
        std::function<double(int, int)> sim;
    };
    std::vector<Method> methods;
    methods.push_back({"Cypha cos(mu)  mu = theta0 + delta_k", [&](int a, int b) { return cosine(&mu[(size_t)a * d], &mu[(size_t)b * d], d); }});
    methods.push_back({"Cypha cos(delta_k)", [&](int a, int b) { return cosine(&dl[(size_t)a * d], &dl[(size_t)b * d], d); }});
    methods.push_back({"Cypha cos(delta_k) in v0-whitened coordinates (Fisher-Rao metric of the shared v0)", [&](int a, int b) {
        double ab = 0, aa = 0, bb = 0;
        for (int j = 0; j < d; ++j) {
            const double x = dl[(size_t)a * d + j], y = dl[(size_t)b * d + j];
            ab += x * y * inv_v0[j];
            aa += x * x * inv_v0[j];
            bb += y * y * inv_v0[j];
        }
        return ab / (std::sqrt(aa * bb) + 1e-30);
    }});
    methods.push_back({"Cypha -Mahalanobis(mu_a, mu_b; shared v0)  [= -2 KL under the core's shared variance]", [&](int a, int b) {
        double s = 0;
        for (int j = 0; j < d; ++j) {
            const double x = mu_d[(size_t)a * d + j] - mu_d[(size_t)b * d + j];
            s += x * x * inv_v0[j];
        }
        return -s;
    }});
    methods.push_back({"Cypha -symmetric KL (per-class variance, written here)", [&](int a, int b) {
        return -(kl_diag(&mu_d[(size_t)a * d], &M.pcvar[(size_t)a * d], &mu_d[(size_t)b * d], &M.pcvar[(size_t)b * d], d) +
                 kl_diag(&mu_d[(size_t)b * d], &M.pcvar[(size_t)b * d], &mu_d[(size_t)a * d], &M.pcvar[(size_t)a * d], d));
    }});
    methods.push_back({"Cypha -Fisher-Rao distance (per-class variance, written here)", [&](int a, int b) {
        return -fr_diag(&mu_d[(size_t)a * d], &M.pcvar[(size_t)a * d], &mu_d[(size_t)b * d], &M.pcvar[(size_t)b * d], d);
    }});
    methods.push_back({"(a) random indexing, plain sum, cosine", [&](int a, int b) { return cosine(&ri_plain[(size_t)a * d], &ri_plain[(size_t)b * d], d); }});
    methods.push_back({"(a') random indexing, mean-centred", [&](int a, int b) { return cosine(&ri_cent[(size_t)a * d], &ri_cent[(size_t)b * d], d); }});
    methods.push_back({"(a'') random indexing, f^-1/2 context weights, mean-centred", [&](int a, int b) { return cosine(&ri_w[(size_t)a * d], &ri_w[(size_t)b * d], d); }});
    methods.push_back({"(a''') random indexing, mean-centred, whitened by the trained v0 (fair counterpart of the whitened Cypha row; exact when the encoder is frozen and the observation plain)", [&](int a, int b) {
        double ab = 0, aa = 0, bb = 0;
        for (int j = 0; j < d; ++j) {
            const double x = ri_cent[(size_t)a * d + j], y = ri_cent[(size_t)b * d + j];
            ab += x * y * inv_v0[j];
            aa += x * x * inv_v0[j];
            bb += y * y * inv_v0[j];
        }
        return ab / (std::sqrt(aa * bb) + 1e-30);
    }});
    methods.push_back({"Count-Gaussian -symmetric KL (exact per-word mean and variance of the same observations)", [&](int a, int b) {
        return -(kl_diag(&cm[(size_t)a * d], &cv[(size_t)a * d], &cm[(size_t)b * d], &cv[(size_t)b * d], d) +
                 kl_diag(&cm[(size_t)b * d], &cv[(size_t)b * d], &cm[(size_t)a * d], &cv[(size_t)a * d], d));
    }});
    methods.push_back({"(b) PPMI-SVD 300 d, cosine", [&](int a, int b) {
        return cosine(&ppmi[(size_t)ppmi_row[a] * ppmi_dim], &ppmi[(size_t)ppmi_row[b] * ppmi_dim], ppmi_dim);
    }});

    std::printf("# cypha_word_classes [%s]\n\n", O.tag.c_str());
    std::printf("config: d=%d nnz=%d window=%d classes=%d competitors=%d passes=%d delta_lr=%g (%s) world_lr=%g enc_lr=%g obs=%s seed=%d max_tokens=%lld\n\n",
                d, O.nnz, O.window, K, O.neg, O.passes, O.delta_lr, O.sched.c_str(), O.world_lr, O.enc_lr, O.obs.c_str(), O.seed, O.max_tokens);
    std::printf("corpus: %llu tokens, %zu types, %llu tokens in the top %d words, %llu observations per pass (top-%d targets)\n\n",
                (unsigned long long)total, n_types, (unsigned long long)kept, O.ctx_vocab, (unsigned long long)class_tokens, K);
    std::printf("## Word similarity, Spearman rho on pairs with both words among the %d classes\n\n| method |", K);
    for (const auto& P : sets) std::printf(" %s (%zu of %d pairs) |", P.name.c_str(), P.pr.size(), P.total);
    std::printf("\n|---|");
    for (size_t s = 0; s < sets.size(); ++s) std::printf("---|");
    std::printf("\n");
    for (const auto& m : methods) {
        std::printf("| %s |", m.name.c_str());
        for (const auto& P : sets) std::printf(" %.3f |", rho_of(P, m.sim));
        std::printf("\n");
    }
    std::fflush(stdout);

    // ---- nearest neighbours (qualitative) ----------------------------------------------------------
    std::printf("\n## Nearest neighbours (cosine), top 8\n\n");
    for (const char* w : {"france", "king", "run", "red", "water", "computer", "music", "war"}) {
        auto it = voc.rank.find(w);
        if (it == voc.rank.end() || it->second >= K) continue;
        const int a = it->second;
        auto nn = [&](const std::vector<float>& E, int dim) {
            std::vector<std::pair<double, int>> v;
            for (int b = 0; b < K; ++b)
                if (b != a) v.push_back({cosine(&E[(size_t)a * dim], &E[(size_t)b * dim], dim), b});
            std::partial_sort(v.begin(), v.begin() + 8, v.end(), [](auto& x, auto& y) { return x.first > y.first; });
            std::string s;
            for (int i = 0; i < 8; ++i) s += voc.words[v[i].second] + (i < 7 ? " " : "");
            return s;
        };
        std::vector<float> pp((size_t)K * ppmi_dim, 0.f);
        for (int r = 0; r < K; ++r) std::memcpy(&pp[(size_t)r * ppmi_dim], &ppmi[(size_t)ppmi_row[r] * ppmi_dim], sizeof(float) * ppmi_dim);
        std::printf("- **%s** Cypha(delta): %s\n  - RI centred: %s\n  - PPMI-SVD: %s\n", w, nn(dl, d).c_str(), nn(ri_cent, d).c_str(),
                    nn(pp, ppmi_dim).c_str());
    }

    // ---- hypernymy ---------------------------------------------------------------------------------
    {
        auto W = load_wordnet(O.wn_dir);
        std::set<uint64_t> pairs;
        size_t raw_pairs = 0;
        for (const auto& kv : W) {
            for (const auto& hid : kv.second.hyper) {
                auto hi = W.find(hid);
                if (hi == W.end()) continue;
                for (const auto& a : kv.second.members) {
                    if (!plain_word(a)) continue;
                    for (const auto& b : hi->second.members) {
                        if (!plain_word(b) || a == b) continue;
                        ++raw_pairs;
                        auto ra = voc.rank.find(a), rb = voc.rank.find(b);
                        if (ra == voc.rank.end() || rb == voc.rank.end()) continue;
                        if (ra->second >= K || rb->second >= K) continue;
                        pairs.insert(((uint64_t)ra->second << 32) | (uint64_t)rb->second);
                    }
                }
            }
        }
        std::vector<std::array<int, 2>> H;  // (hypo, hyper)
        size_t dropped_rev = 0;
        for (uint64_t p : pairs) {
            const int a = (int)(p >> 32), b = (int)(p & 0xFFFFFFFFu);
            if (pairs.count(((uint64_t)b << 32) | (uint64_t)a)) {
                ++dropped_rev;
                continue;
            }
            H.push_back({a, b});
        }
        std::printf("\n## Hypernymy direction (Open English WordNet, direct noun hypernym pairs, single-word lemmas)\n\n");
        std::printf("%zu synsets read; %zu lemma pairs before the vocabulary filter; %zu distinct pairs with both words among the classes; "
                    "%zu dropped because the reverse pair also exists (cross-sense); **%zu pairs scored**.\n\n",
                    W.size(), raw_pairs, pairs.size(), dropped_rev, H.size());
        std::vector<double> trace_pc(K, 0.0), trace_c(K, 0.0), nrm(K, 0.0), epi(K, 0.0), lf(K);
        double vmean0 = 0;
        for (int j = 0; j < d; ++j) vmean0 += M.world_v[j] / d;
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < d; ++j) {
                trace_pc[k] += M.pcvar[(size_t)k * d + j];
                trace_c[k] += cv[(size_t)k * d + j];
                nrm[k] += (double)dl[(size_t)k * d + j] * dl[(size_t)k * d + j];
            }
            nrm[k] = std::sqrt(nrm[k]);
            epi[k] = vmean0 / (M.nobs[k] + 1.0);
            lf[k] = std::log((double)voc.freq[k]);
        }
        struct Dir {
            std::string name;
            std::function<double(int, int)> margin;  // >0: predicts the right direction for (hypo, hyper)
        };
        std::vector<Dir> dirs;
        dirs.push_back({"frequency baseline: more frequent word = hypernym", [&](int a, int b) { return (double)voc.freq[b] - (double)voc.freq[a]; }});
        dirs.push_back({"Cypha: KL(hypo||hyper) < KL(hyper||hypo), per-class variance (written here)", [&](int a, int b) {
            return kl_diag(&mu_d[(size_t)b * d], &M.pcvar[(size_t)b * d], &mu_d[(size_t)a * d], &M.pcvar[(size_t)a * d], d) -
                   kl_diag(&mu_d[(size_t)a * d], &M.pcvar[(size_t)a * d], &mu_d[(size_t)b * d], &M.pcvar[(size_t)b * d], d);
        }});
        dirs.push_back({"Cypha: total variance, var(hyper) > var(hypo), per-class variance (written here)", [&](int a, int b) { return trace_pc[b] - trace_pc[a]; }});
        dirs.push_back({"Cypha: smaller mean-differential norm = hypernym, |delta(hyper)| < |delta(hypo)|", [&](int a, int b) { return nrm[a] - nrm[b]; }});
        dirs.push_back({"Cypha core: epistemic variance v0/(n+1), larger = hypernym (the core's only per-class variance)", [&](int a, int b) { return epi[b] - epi[a]; }});
        dirs.push_back({"Cypha core: KL under the shared v0 (symmetric by construction)", [&](int, int) { return 0.0; }});
        dirs.push_back({"Count-Gaussian: KL(hypo||hyper) < KL(hyper||hypo), exact per-word moments", [&](int a, int b) {
            return kl_diag(&cm[(size_t)b * d], &cv[(size_t)b * d], &cm[(size_t)a * d], &cv[(size_t)a * d], d) -
                   kl_diag(&cm[(size_t)a * d], &cv[(size_t)a * d], &cm[(size_t)b * d], &cv[(size_t)b * d], d);
        }});
        dirs.push_back({"Count-Gaussian: total variance, var(hyper) > var(hypo)", [&](int a, int b) { return trace_c[b] - trace_c[a]; }});
        std::vector<size_t> matched;
        for (size_t i = 0; i < H.size(); ++i)
            if (std::fabs(lf[H[i][1]] - lf[H[i][0]]) < std::log(2.0)) matched.push_back(i);
        std::printf("| direction rule | accuracy, all %zu pairs (95%% CI half-width) | accuracy, %zu frequency-matched pairs (within a factor 2) |\n|---|---|---|\n",
                    H.size(), matched.size());
        for (const auto& dr : dirs) {
            auto acc = [&](const std::vector<size_t>* sub, double* hw) {
                double w = 0, n = 0;
                auto one = [&](size_t i) {
                    const double m = dr.margin(H[i][0], H[i][1]);
                    w += m > 0 ? 1.0 : (m == 0 ? 0.5 : 0.0);
                    n += 1;
                };
                if (sub) for (size_t i : *sub) one(i);
                else for (size_t i = 0; i < H.size(); ++i) one(i);
                const double p = n > 0 ? w / n : std::nan("");
                if (hw) *hw = n > 0 ? 1.96 * std::sqrt(p * (1 - p) / n) : 0;
                return p;
            };
            double hw1 = 0, hw2 = 0;
            const double a1 = acc(nullptr, &hw1), a2 = acc(&matched, &hw2);
            std::printf("| %s | %.1f%% (+-%.1f) | %.1f%% (+-%.1f) |\n", dr.name.c_str(), 100 * a1, 100 * hw1, 100 * a2, 100 * hw2);
        }
        std::vector<double> t1(trace_pc.begin(), trace_pc.end()), t2(trace_c.begin(), trace_c.end()), t3(nrm.begin(), nrm.end());
        std::printf("\nSpearman across the %d classes: per-class variance trace vs log frequency %.3f; count-Gaussian variance trace vs log frequency %.3f; "
                    "|delta| vs log frequency %.3f\n", K, spearman(t1, lf), spearman(t2, lf), spearman(t3, lf));
    }
    std::fflush(stdout);

    // ---- relations as maps -------------------------------------------------------------------------
    {
        auto build = [&](const char* (*lst)[2], size_t n) {
            std::vector<std::pair<int, int>> out;
            std::set<std::pair<int, int>> seen;
            for (size_t i = 0; i < n; ++i) {
                auto a = voc.rank.find(lst[i][0]), b = voc.rank.find(lst[i][1]);
                if (a == voc.rank.end() || b == voc.rank.end() || a->second >= K || b->second >= K) continue;
                if (a->second == b->second) continue;  // subject = object cannot be scored
                if (seen.insert({a->second, b->second}).second) out.push_back({a->second, b->second});
            }
            return out;
        };
        const auto cap = build(kCapitalOf, sizeof(kCapitalOf) / sizeof(kCapitalOf[0]));
        const auto lang = build(kCountryLanguage, sizeof(kCountryLanguage) / sizeof(kCountryLanguage[0]));

        // unit-length versions
        auto unit = [&](const std::vector<float>& E, int dim) {
            std::vector<float> U(E.begin(), E.begin() + (size_t)K * dim);
            for (int k = 0; k < K; ++k) {
                double s = 0;
                for (int j = 0; j < dim; ++j) s += (double)U[(size_t)k * dim + j] * U[(size_t)k * dim + j];
                const float inv = (float)(1.0 / (std::sqrt(s) + 1e-30));
                for (int j = 0; j < dim; ++j) U[(size_t)k * dim + j] *= inv;
            }
            return U;
        };
        std::vector<float> pp((size_t)K * ppmi_dim);
        for (int r = 0; r < K; ++r) std::memcpy(&pp[(size_t)r * ppmi_dim], &ppmi[(size_t)ppmi_row[r] * ppmi_dim], sizeof(float) * ppmi_dim);
        struct Rep {
            std::string name;
            std::vector<float> E;
            int dim;
        };
        std::vector<Rep> reps;
        reps.push_back({"Cypha (delta_k, unit)", unit(dl, d), d});
        reps.push_back({"(a') random indexing, mean-centred", unit(ri_cent, d), d});
        reps.push_back({"(b) PPMI-SVD", unit(pp, ppmi_dim), ppmi_dim});

        std::printf("\n## Relations as maps, object ~ A subject + b, leave-one-out, nearest word among the %d classes (subject excluded)\n\n", K);
        std::printf("Map: Bayesian ridge in the dual form written here (Cypha's regression module has scalar ridge/MKE experts only, no vector-output map); "
                    "lambda per held-out fold from closed-form inner leave-one-out. Offset: x + mean(y - x) on the training pairs.\n\n");
        for (int rel = 0; rel < 2; ++rel) {
            const auto& P = rel == 0 ? cap : lang;
            std::printf("### %s: %zu single-word pairs in the classes\n\n", rel == 0 ? "capital-of" : "country -> language", P.size());
            std::printf("| representation | ridge map @1 | ridge map @10 | mean offset @1 | mean offset @10 | median rank (ridge / offset) | mean lambda |\n|---|---|---|---|---|---|---|\n");
            std::vector<std::vector<RelItem>> store;
            for (const auto& R : reps) {
                double lam = 0;
                auto res = run_relation(P, R.E, R.dim, K, &lam);
                int h1 = 0, h10 = 0, o1 = 0, o10 = 0;
                std::vector<double> rr, ro;
                for (const auto& it : res) {
                    h1 += it.rank_ridge == 1;
                    h10 += it.rank_ridge <= 10;
                    o1 += it.rank_off == 1;
                    o10 += it.rank_off <= 10;
                    rr.push_back(it.rank_ridge);
                    ro.push_back(it.rank_off);
                }
                std::sort(rr.begin(), rr.end());
                std::sort(ro.begin(), ro.end());
                const double n = (double)res.size();
                std::printf("| %s | %.1f%% (%d/%zu) | %.1f%% (%d/%zu) | %.1f%% (%d/%zu) | %.1f%% (%d/%zu) | %.0f / %.0f | %.3g |\n", R.name.c_str(),
                            100.0 * h1 / n, h1, res.size(), 100.0 * h10 / n, h10, res.size(), 100.0 * o1 / n, o1, res.size(),
                            100.0 * o10 / n, o10, res.size(), rr.empty() ? 0.0 : rr[rr.size() / 2], ro.empty() ? 0.0 : ro[ro.size() / 2], lam);
                store.push_back(std::move(res));
            }
            // abstention with the predictive variance
            std::printf("\nAbstaining on the predictive variance of the ridge map (answer the lowest-variance fraction), accuracy at @10 / @1; in brackets the same with the top-candidate cosine as the confidence:\n\n");
            std::printf("| representation | coverage 100%% | 75%% | 50%% | 25%% | AUROC(-variance -> hit@10) | AUROC(cosine -> hit@10) |\n|---|---|---|---|---|---|---|\n");
            for (size_t ri = 0; ri < reps.size(); ++ri) {
                auto& res = store[ri];
                std::vector<size_t> byu(res.size()), byc(res.size());
                std::iota(byu.begin(), byu.end(), 0);
                byc = byu;
                std::sort(byu.begin(), byu.end(), [&](size_t a, size_t b) { return res[a].u < res[b].u; });
                std::sort(byc.begin(), byc.end(), [&](size_t a, size_t b) { return res[a].margin > res[b].margin; });
                std::printf("| %s |", reps[ri].name.c_str());
                for (double cov : {1.0, 0.75, 0.5, 0.25}) {
                    const size_t m = std::max<size_t>(1, (size_t)std::llround(cov * res.size()));
                    int a10 = 0, a1 = 0, c10 = 0, c1 = 0;
                    for (size_t i = 0; i < m; ++i) {
                        a10 += res[byu[i]].rank_ridge <= 10;
                        a1 += res[byu[i]].rank_ridge == 1;
                        c10 += res[byc[i]].rank_ridge <= 10;
                        c1 += res[byc[i]].rank_ridge == 1;
                    }
                    std::printf(" %.0f%% / %.0f%% (%.0f%% / %.0f%%) [n=%zu] |", 100.0 * a10 / m, 100.0 * a1 / m, 100.0 * c10 / m, 100.0 * c1 / m, m);
                }
                std::vector<double> su, sc;
                std::vector<char> pos;
                for (const auto& it : res) {
                    su.push_back(-it.u);
                    sc.push_back(it.margin);
                    pos.push_back(it.rank_ridge <= 10);
                }
                std::printf(" %.3f | %.3f |\n", auroc(su, pos), auroc(sc, pos));
            }
            std::printf("\n");
            if (rel == 0) {
                // list the Cypha predictions for the capital-of relation
                std::printf("Cypha ridge-map misses (capital-of, rank > 10): ");
                for (const auto& it : store[0])
                    if (it.rank_ridge > 10) std::printf("%s->%s(rank %d) ", voc.words[it.subj].c_str(), voc.words[it.obj].c_str(), it.rank_ridge);
                std::printf("\n\n");
            }
        }

        // Cross-check the dual-form ridge against Cypha's own scalar ridge on output dimension 0 (capital-of, Cypha representation)
        {
            const auto& R = reps[0];
            std::vector<const float*> X, Y;
            for (const auto& pr : cap) {
                X.push_back(&R.E[(size_t)pr.first * d]);
                Y.push_back(&R.E[(size_t)pr.second * d]);
            }
            RidgeMap rm;
            if (X.size() > 3 && rm.fit(X, Y, d, d)) {
                const int n = (int)X.size();
                // primal design with centred inputs; Cypha's ridge_fit_bias leaves the bias unpenalised
                std::vector<double> Phi((size_t)n * d), y0(n), coef(d + 1);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < d; ++j) Phi[(size_t)i * d + j] = X[i][j];
                    y0[i] = Y[i][0];
                }
                // ridge_fit_bias does not centre, so subtract the training means to match the dual (centred) fit
                std::vector<double> xb(d, 0.0);
                double yb = 0;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < d; ++j) xb[j] += Phi[(size_t)i * d + j] / n;
                    yb += y0[i] / n;
                }
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < d; ++j) Phi[(size_t)i * d + j] -= xb[j];
                    y0[i] -= yb;
                }
                if (cypha::regression::ridge_fit_bias(Phi.data(), n, d, rm.lambda, y0.data(), coef.data())) {
                    double maxdiff = 0, maxabs = 0;
                    for (int i = 0; i < n; ++i) {
                        double a = coef[d];
                        for (int j = 0; j < d; ++j) a += Phi[(size_t)i * d + j] * coef[j];
                        std::vector<double> yp;
                        double u;
                        rm.predict(X[i], yp, u);
                        maxdiff = std::max(maxdiff, std::fabs(a + yb - yp[0]));
                        maxabs = std::max(maxabs, std::fabs(yp[0]));
                    }
                    std::printf("Cross-check: cypha::regression::ridge_fit_bias vs the dual-form ridge, output dimension 0, all %d capital pairs, same lambda: "
                                "max |difference| of the fitted values %.2e (max |value| %.3f).\n", n, maxdiff, maxabs);
                } else {
                    std::printf("Cross-check: ridge_fit_bias reported a non-SPD normal matrix.\n");
                }
            }
        }
    }

    // ---- cost --------------------------------------------------------------------------------------
    const double model_mb_f32 = (double)K * d * 4 * 2 / 1e6;
    std::printf("\n## Cost\n\n");
    std::printf("| item | value |\n|---|---|\n");
    std::printf("| cleaning + tokenising + vocabulary | %.1f s |\n", clean_s);
    std::printf("| Cypha training (%d pass%s, %lld steps, single thread) | %.1f s (%.1f us per step with %d slots) |\n", O.passes,
                O.passes == 1 ? "" : "es", M.steps, M.train_s, 1e6 * M.train_s / std::max<long long>(1, M.steps), 1 + O.neg);
    std::printf("| whole run wall time (excluding the public-API probe) | %.1f s |\n", elapsed());
    std::printf("| peak RAM | %ld MB |\n", peak_rss_mb());
    std::printf("| model: class differentials (%d x %d float32) + per-class variances (same) | %.1f MB (in memory as double: %.1f MB for the differentials) |\n", K, d,
                model_mb_f32, (double)K * d * 8 / 1e6);
    std::printf("| model: WorldPrior (mu, v) + encoder W (%d x %d double) | %.2f MB |\n", d, d, ((double)d * d * 8 + 2.0 * d * 8) / 1e6);
    std::printf("| sampled-slot training accuracy (true class ranked first among %d slots) | %.3f |\n", 1 + O.neg, (double)M.correct / std::max<long long>(1, M.steps));
    std::printf("| encoder updates / ||W - I||_F | %lld / %.3f |\n", M.enc_updates, M.enc_drift);
    std::fflush(stdout);

    if (O.bench_full) bench_full_api(d);
    return 0;
}

int main(int argc, char** argv) {
    // Subcommand table. train / distill / eval / export (docs/EMBEDDING_GEOMETRY.md §9.3) are added here later.
    struct Sub {
        const char* name;
        int (*fn)(int, char**);
    };
    static const Sub subs[] = {{"s1", cmd_s1}};
    if (argc >= 2)
        for (const Sub& s : subs)
            if (std::strcmp(argv[1], s.name) == 0) return s.fn(argc - 1, argv + 1);
    return cmd_s1(argc, argv);  // no subcommand: the S-1 run, as in the report's command lines
}
