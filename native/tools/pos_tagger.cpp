// E1 (and the rule part of E2): part-of-speech tagger, phrase chunker, question parse helper.
//
// Tagger: averaged perceptron (Collins 2002), greedy left to right, for UPOS (17 tags) and XPOS (49 tags),
// trained separately on Universal Dependencies English EWT. Features per token: word, lowercase word,
// prefixes and suffixes of 1-3 characters, word shape (Xxx, dd, ...), hyphen / digit / capital / all-caps
// flags, previous two and next two words, suffixes of the previous and next word, two word bigrams, shapes of
// the neighbours, "sentence ends in ?" x word, and the previous one and two predicted tags (also with the word
// and its suffix). A tag dictionary (words seen >= 20 times with one tag >= 97% of the time: 685 UPOS words,
// 47% of training tokens) bypasses the scorer and is not trained on. No beam; greedy only.
//
// Chunker: rules over the predicted UPOS tags. NP = DET? DET? (NUM | ADJ | ADV* ADJ | participle)* (NOUN | PROPN)+
// (a possessive PRON counts as DET; "," between adjectives and "-" inside a modifier chain are allowed); lone PRON
// is a pronoun NP. Verb group = AUX/VERB run with ADV / not between (a "to" starts one). PP = ADP + NP.
// Evaluation: exact span match of the nominal NPs against NPs read off the gold dependency tree: a NOUN/PROPN
// head plus its det, predet, amod, nummod, possessive-pronoun, compound and flat dependents (and adverbs of an
// included adjective), punctuation inside the span filled in and at the edges trimmed.
//
// Question parse: for a question, the wh-word, the wh-phrase (with a leading preposition, "how many"), the head
// noun of the wh-phrase, the auxiliary, the main verb (the copula when no VERB follows), a focus noun (the wh head,
// else the predicate noun after a copula), and the main entity (the NP holding the longest PROPN run outside the
// wh-phrase, ties to the later one; else the last nominal NP).
//
// Protocol (docs/reports/E1_POS_TAGGER.md): tokens are the UD syntactic words (multiword-token lines and empty
// nodes skipped), gold tokenisation. Train on train, up to 40 epochs with the training order reshuffled each
// epoch, patience 6; after every epoch the averaged weights are scored on dev and the best epoch is kept (UPOS
// 13, XPOS 9). Test was run once, after the code and the chunker rules were frozen on dev. Known word = exact
// form seen in train. Question = sentence whose "# text" ends in "?". Speed = single thread, best of 3,
// pre-tokenised input, token hashing included. The model is int16-quantised hashed features (no strings); the
// train command reloads the file and re-scores dev to prove the saved model is the measured one.
//
// Measured 2026-10-27 (UD English EWT v2.18; train 12,544 sentences / 204,578 tokens; dev 2,001 / 25,148;
// test 2,077 / 25,094):
//   test UPOS 94.81% (gate >= 94%: pass); known words 96.38%, unknown words 79.23% (9.1% of tokens);
//     question sentences 93.25% (170 sentences); XPOS 94.13%. Dev: UPOS 94.81%, XPOS 94.29%.
//   NP chunker F1 (exact span, predicted UPOS): dev 85.67, test 83.78; with gold UPOS (rule ceiling) 90.14 / 88.09.
//   speed: about 1.0M tokens/s hashing + UPOS, about 0.5M tokens/s with XPOS as well (a model trained with
//     --no-dict: about 0.65M / 0.28M, dev UPOS 94.85%). Training: 12 s for both taggers.
//   model: 5.6 MB for UPOS + XPOS (2.8 MB UPOS only). Averaged-perceptron taggers score approximately 94-96%
//     UPOS on EWT in the literature (stated, not verified here).
//   Report: docs/reports/E1_POS_TAGGER.md
//
// Build: g++ -O2 -std=c++20 native/tools/pos_tagger.cpp -o pos_tagger
// Usage: pos_tagger train <train.conllu> <dev.conllu> <model_out> [--epochs N] [--patience N] [--prune X]
//                          [--dict-min N] [--dict-ratio R] [--no-dict] [--no-xpos]
//        pos_tagger eval <model> <file.conllu>   (accuracy, known / unknown / questions, speed, NP F1)
//        pos_tagger tag <model> [--xpos] [--chunks]   (stdin lines -> word/UPOS[/XPOS] [+ bracketed chunks])
//        pos_tagger question <model> [--tsv]   (stdin questions -> wh / aux / verb / head noun / entity)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr int kMaxTags = 128;
constexpr u16 kNone = 0xFFFF;

// ---------------------------------------------------------------- hashing
inline u64 fin(u64 x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x ? x : 1;
}
inline u64 hs(const std::string& s) {
    u64 h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return fin(h);
}
inline u64 comb(u64 a, u64 b) { return fin(a ^ (b + 0x9E3779B97F4A7C15ULL + (a << 6) + (a >> 2))); }

// Open-addressing table, key 0 = empty.
struct HashTab {
    std::vector<u64> keys;
    std::vector<u32> vals;
    u64 mask = 0;
    void init(std::size_t expected) {
        std::size_t cap = 16;
        while (cap < expected * 2 + 2) cap <<= 1;
        keys.assign(cap, 0);
        vals.assign(cap, 0);
        mask = cap - 1;
    }
    void put(u64 k, u32 v) {
        std::size_t i = k & mask;
        while (keys[i] && keys[i] != k) i = (i + 1) & mask;
        keys[i] = k;
        vals[i] = v;
    }
    bool get(u64 k, u32& v) const {
        std::size_t i = k & mask;
        while (keys[i]) {
            if (keys[i] == k) {
                v = vals[i];
                return true;
            }
            i = (i + 1) & mask;
        }
        return false;
    }
};

// ---------------------------------------------------------------- strings
inline int cplen(unsigned char c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1; }
std::string prefix_cp(const std::string& s, int k) {
    std::size_t i = 0;
    int c = 0;
    while (i < s.size() && c < k) {
        i += cplen(static_cast<unsigned char>(s[i]));
        ++c;
    }
    return s.substr(0, std::min(i, s.size()));
}
std::string suffix_cp(const std::string& s, int k) {
    std::size_t i = s.size();
    int c = 0;
    while (i > 0 && c < k) {
        --i;
        while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i;
        ++c;
    }
    return s.substr(i);
}
std::string lower(std::string s) {
    for (auto& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return s;
}
// Word shape: X upper, x lower, d digit, other ASCII kept; runs capped at 2 (Xxx, dd, ...).
std::string shape(const std::string& s) {
    std::string o;
    char last = 0;
    int run = 0;
    std::size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        char cl;
        if (c < 0x80) {
            if (c >= 'A' && c <= 'Z') cl = 'X';
            else if (c >= 'a' && c <= 'z') cl = 'x';
            else if (c >= '0' && c <= '9') cl = 'd';
            else cl = static_cast<char>(c);
        } else if (c == 0xE2 && i + 1 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0x80) {
            cl = '*';  // general punctuation (curly quotes, dashes, ellipsis)
        } else {
            cl = 'x';
        }
        if (cl == last) {
            if (++run <= 2) o += cl;
        } else {
            o += cl;
            last = cl;
            run = 1;
        }
        i += cplen(c);
    }
    return o;
}

// ---------------------------------------------------------------- data
struct Sent {
    std::vector<std::string> w, upos, xpos, deprel;
    std::vector<int> head;  // 1-based, 0 = root
    std::string text;
    bool question() const { return !text.empty() && text.back() == '?'; }
};

bool read_conllu(const std::string& path, std::vector<Sent>& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    Sent cur;
    auto flush = [&] {
        if (!cur.w.empty()) out.push_back(std::move(cur));
        cur = Sent();
    };
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) {
            flush();
            continue;
        }
        if (line[0] == '#') {
            if (line.rfind("# text = ", 0) == 0) cur.text = line.substr(9);
            continue;
        }
        std::vector<std::string> fld;
        std::size_t b = 0;
        while (true) {
            std::size_t e = line.find('\t', b);
            if (e == std::string::npos) {
                fld.push_back(line.substr(b));
                break;
            }
            fld.push_back(line.substr(b, e - b));
            b = e + 1;
        }
        if (fld.size() < 8) continue;
        if (fld[0].find('-') != std::string::npos || fld[0].find('.') != std::string::npos) continue;  // MWT range / empty node
        cur.w.push_back(fld[1]);
        cur.upos.push_back(fld[3]);
        cur.xpos.push_back(fld[4]);
        cur.head.push_back(std::atoi(fld[6].c_str()));
        cur.deprel.push_back(fld[7]);
    }
    flush();
    return true;
}

// ---------------------------------------------------------------- features
// Per-sentence cache of token hashes. Arrays are padded by 2 each side (index i+2).
struct Ctx {
    int n = 0;
    bool q = false;
    std::vector<u64> w, lw, p1, p2, p3, s1, s2, s3, shp;
    std::vector<u8> fl;  // 1 hyphen, 2 digit, 4 capital, 8 all-caps
    void build(const std::vector<std::string>& words) {
        n = static_cast<int>(words.size());
        const std::size_t N = static_cast<std::size_t>(n) + 4;
        for (auto* v : {&w, &lw, &p1, &p2, &p3, &s1, &s2, &s3, &shp}) v->assign(N, 0);
        fl.assign(N, 0);
        static const u64 pad[4] = {hs("<B2>"), hs("<B1>"), hs("<E1>"), hs("<E2>")};
        auto set_pad = [&](std::size_t k, u64 p) {
            for (auto* v : {&w, &lw, &p1, &p2, &p3, &s1, &s2, &s3, &shp}) (*v)[k] = p;
        };
        set_pad(0, pad[0]);
        set_pad(1, pad[1]);
        set_pad(N - 2, pad[2]);
        set_pad(N - 1, pad[3]);
        q = n > 0 && (words[n - 1] == "?" || (n > 1 && words[n - 2] == "?"));
        for (int i = 0; i < n; ++i) {
            const std::string& s = words[i];
            const std::string l = lower(s);
            const std::size_t k = static_cast<std::size_t>(i) + 2;
            w[k] = hs(s);
            lw[k] = hs(l);
            p1[k] = hs(prefix_cp(l, 1));
            p2[k] = hs(prefix_cp(l, 2));
            p3[k] = hs(prefix_cp(l, 3));
            s1[k] = hs(suffix_cp(l, 1));
            s2[k] = hs(suffix_cp(l, 2));
            s3[k] = hs(suffix_cp(l, 3));
            shp[k] = hs(shape(s));
            u8 f = 0;
            bool up = false, lo = false;
            for (unsigned char c : s) {
                if (c == '-') f |= 1;
                if (c >= '0' && c <= '9') f |= 2;
                if (c >= 'A' && c <= 'Z') up = true;
                if (c >= 'a' && c <= 'z') lo = true;
            }
            if (up) f |= 4;
            if (up && !lo && s.size() > 1) f |= 8;
            fl[k] = f;
        }
    }
};

// Static (history-free) features of token i. Cached across epochs in training.
void static_feats(const Ctx& c, int i, std::vector<u64>& f) {
    f.clear();
    const std::size_t k = static_cast<std::size_t>(i) + 2;
    auto add = [&](u64 id, u64 a) { f.push_back(comb(id, a)); };
    auto add2 = [&](u64 id, u64 a, u64 b) { f.push_back(comb(comb(id, a), b)); };
    add(1, 0);  // bias
    add(2, c.w[k]);
    add(3, c.lw[k]);
    add(4, c.p1[k]);
    add(5, c.p2[k]);
    add(6, c.p3[k]);
    add(7, c.s1[k]);
    add(8, c.s2[k]);
    add(9, c.s3[k]);
    add(10, c.shp[k]);
    const u8 fl = c.fl[k];
    if (fl & 1) add(11, 1);
    if (fl & 2) add(12, 1);
    if (fl & 4) add(13, i > 0 ? 2 : 1);  // capital, mid-sentence vs initial
    if (fl & 8) add(14, 1);
    add(15, c.lw[k - 1]);
    add(16, c.s3[k - 1]);
    add(17, c.lw[k + 1]);
    add(18, c.s3[k + 1]);
    add(19, c.lw[k - 2]);
    add(20, c.lw[k + 2]);
    add2(21, c.lw[k - 1], c.lw[k]);
    add2(22, c.lw[k], c.lw[k + 1]);
    add(23, c.shp[k - 1]);
    add(24, c.shp[k + 1]);
    if (c.q) add(25, c.lw[k]);  // sentence ends in '?': word-specific question evidence
}
constexpr int kDyn = 5;
// History features: previous one and two predicted tags (id n_tags = sentence start).
void dyn_feats(const Ctx& c, int i, u16 t1, u16 t2, u64* d) {
    const std::size_t k = static_cast<std::size_t>(i) + 2;
    d[0] = comb(40, t1);
    d[1] = comb(comb(41, t2), t1);
    d[2] = comb(comb(42, t1), c.lw[k]);
    d[3] = comb(comb(43, t1), c.s3[k]);
    d[4] = comb(comb(44, t1), c.lw[k + 1]);
}

}  // namespace

namespace {

// ---------------------------------------------------------------- inference tagger
struct Tagger {
    std::vector<std::string> tags;
    HashTab feat;           // feature hash -> row
    std::vector<u32> off;   // CSR offsets, rows+1
    std::vector<u16> etag;
    std::vector<float> ew;
    HashTab dict;           // word hash -> tag (frequent unambiguous words)
    bool use_dict = true;

    std::size_t T() const { return tags.size(); }

    void tag(const Ctx& c, std::vector<u16>& out) const {
        const int n = c.n;
        out.assign(static_cast<std::size_t>(n), 0);
        const u16 START = static_cast<u16>(T());
        u16 t1 = START, t2 = START;
        thread_local std::vector<u64> f;
        float sc[kMaxTags];
        const int nt = static_cast<int>(T());
        for (int i = 0; i < n; ++i) {
            u32 dv;
            if (use_dict && dict.get(c.w[static_cast<std::size_t>(i) + 2], dv)) {
                out[i] = static_cast<u16>(dv);
            } else {
                for (int t = 0; t < nt; ++t) sc[t] = 0;
                static_feats(c, i, f);
                u64 d[kDyn];
                dyn_feats(c, i, t1, t2, d);
                for (u64 h : f) add_feat(h, sc);
                for (int j = 0; j < kDyn; ++j) add_feat(d[j], sc);
                int best = 0;
                for (int t = 1; t < nt; ++t)
                    if (sc[t] > sc[best]) best = t;
                out[i] = static_cast<u16>(best);
            }
            t2 = t1;
            t1 = out[i];
        }
    }
    void add_feat(u64 h, float* sc) const {
        u32 r;
        if (!feat.get(h, r)) return;
        for (u32 e = off[r]; e < off[r + 1]; ++e) sc[etag[e]] += ew[e];
    }
};

struct Model {
    Tagger up, xp;
    bool has_xp = false;
    HashTab vocab;  // hashes of the exact word forms seen in training
};

// ---------------------------------------------------------------- model file
// "CYPOS001", vocab, then 1 or 2 tagger blocks. Weights are int16 with one scale per tagger.
template <class T> void put(std::string& b, T v) { b.append(reinterpret_cast<const char*>(&v), sizeof(T)); }
struct Rd {
    const char* p;
    const char* e;
    template <class T> T get() {
        T v;
        if (p + sizeof(T) > e) {
            std::fprintf(stderr, "model file truncated\n");
            std::exit(1);
        }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
};

void save_tagger(std::string& b, const Tagger& m) {
    put<u32>(b, static_cast<u32>(m.T()));
    for (auto& t : m.tags) {
        put<u8>(b, static_cast<u8>(t.size()));
        b.append(t);
    }
    u32 nd = 0;
    for (std::size_t i = 0; i < m.dict.keys.size(); ++i) nd += m.dict.keys[i] != 0;
    put<u32>(b, nd);
    for (std::size_t i = 0; i < m.dict.keys.size(); ++i)
        if (m.dict.keys[i]) {
            put<u64>(b, m.dict.keys[i]);
            put<u16>(b, static_cast<u16>(m.dict.vals[i]));
        }
    float mx = 1e-9f;
    for (float w : m.ew) mx = std::max(mx, std::abs(w));
    const float scale = mx / 32767.0f;
    const u32 rows = static_cast<u32>(m.off.size() - 1);
    put<u32>(b, rows);
    put<float>(b, scale);
    // Recover each row's key from the table.
    std::vector<u64> key(rows, 0);
    for (std::size_t i = 0; i < m.feat.keys.size(); ++i)
        if (m.feat.keys[i]) key[m.feat.vals[i]] = m.feat.keys[i];
    for (u32 r = 0; r < rows; ++r) {
        put<u64>(b, key[r]);
        put<u8>(b, static_cast<u8>(m.off[r + 1] - m.off[r]));
        for (u32 e = m.off[r]; e < m.off[r + 1]; ++e) {
            put<u8>(b, static_cast<u8>(m.etag[e]));
            put<std::int16_t>(b, static_cast<std::int16_t>(std::lround(m.ew[e] / scale)));
        }
    }
}
void load_tagger(Rd& r, Tagger& m) {
    const u32 nt = r.get<u32>();
    m.tags.resize(nt);
    for (auto& t : m.tags) {
        const u8 l = r.get<u8>();
        t.assign(r.p, l);
        r.p += l;
    }
    const u32 nd = r.get<u32>();
    m.dict.init(nd);
    for (u32 i = 0; i < nd; ++i) {
        const u64 k = r.get<u64>();
        m.dict.put(k, r.get<u16>());
    }
    const u32 rows = r.get<u32>();
    const float scale = r.get<float>();
    m.feat.init(rows);
    m.off.assign(1, 0);
    for (u32 i = 0; i < rows; ++i) {
        const u64 k = r.get<u64>();
        m.feat.put(k, i);
        const u8 ne = r.get<u8>();
        for (u8 j = 0; j < ne; ++j) {
            m.etag.push_back(r.get<u8>());
            m.ew.push_back(static_cast<float>(r.get<std::int16_t>()) * scale);
        }
        m.off.push_back(static_cast<u32>(m.etag.size()));
    }
}
bool save_model(const std::string& path, const Model& M, const std::vector<u64>& vocab) {
    std::string b = "CYPOS001";
    put<u32>(b, static_cast<u32>(vocab.size()));
    for (u64 k : vocab) put<u64>(b, k);
    put<u8>(b, M.has_xp ? 2 : 1);
    save_tagger(b, M.up);
    if (M.has_xp) save_tagger(b, M.xp);
    std::ofstream f(path, std::ios::binary);
    f.write(b.data(), static_cast<std::streamsize>(b.size()));
    std::printf("model written: %s (%zu bytes)\n", path.c_str(), b.size());
    return static_cast<bool>(f);
}
bool load_model(const std::string& path, Model& M) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() < 8 || b.compare(0, 8, "CYPOS001") != 0) return false;
    Rd r{b.data() + 8, b.data() + b.size()};
    const u32 nv = r.get<u32>();
    M.vocab.init(nv);
    for (u32 i = 0; i < nv; ++i) M.vocab.put(r.get<u64>(), 1);
    const u8 nt = r.get<u8>();
    load_tagger(r, M.up);
    M.has_xp = nt == 2;
    if (M.has_xp) load_tagger(r, M.xp);
    return true;
}

// ---------------------------------------------------------------- averaged-perceptron trainer
struct TEntry {
    u16 tag;
    float w;
    double tot;
    u32 ts;
};
struct TrainSet {
    std::vector<Ctx> ctx;
    std::vector<std::vector<u16>> gold, dtag;
    std::vector<std::vector<u32>> srow, soff;
};

struct Trainer {
    std::vector<std::string> tags;
    std::unordered_map<u64, u32> idx;
    std::vector<std::vector<TEntry>> rows;
    std::vector<u64> rowkey;
    HashTab dict;
    u32 clock = 0;

    u32 row(u64 h) {
        auto it = idx.try_emplace(h, static_cast<u32>(rows.size()));
        if (it.second) {
            rows.emplace_back();
            rowkey.push_back(h);
        }
        return it.first->second;
    }
    void upd(u32 r, u16 tag, float delta) {
        auto& v = rows[r];
        TEntry* e = nullptr;
        for (auto& x : v)
            if (x.tag == tag) {
                e = &x;
                break;
            }
        if (!e) {
            v.push_back({tag, 0.f, 0.0, clock});
            e = &v.back();
        }
        e->tot += static_cast<double>(e->w) * (clock - e->ts);
        e->ts = clock;
        e->w += delta;
    }
    // Cache the static feature rows of every training token (created on first sight).
    void prepare(TrainSet& D) {
        std::vector<u64> f;
        for (std::size_t s = 0; s < D.ctx.size(); ++s) {
            auto& sr = D.srow[s];
            auto& so = D.soff[s];
            so.assign(1, 0);
            for (int i = 0; i < D.ctx[s].n; ++i) {
                static_feats(D.ctx[s], i, f);
                for (u64 h : f) sr.push_back(row(h));
                so.push_back(static_cast<u32>(sr.size()));
            }
        }
    }
    // One greedy left-to-right pass over a sentence; returns (non-dict tokens, correct).
    void train_sentence(const TrainSet& D, std::size_t s, std::size_t& n, std::size_t& ok) {
        const Ctx& c = D.ctx[s];
        const auto& gold = D.gold[s];
        const auto& dt = D.dtag[s];
        const auto& sr = D.srow[s];
        const auto& so = D.soff[s];
        const int nt = static_cast<int>(tags.size());
        const u16 START = static_cast<u16>(nt);
        u16 t1 = START, t2 = START;
        float sc[kMaxTags];
        for (int i = 0; i < c.n; ++i) {
            u16 pred;
            if (dt[i] != kNone) {
                pred = dt[i];
            } else {
                for (int t = 0; t < nt; ++t) sc[t] = 0;
                for (u32 k = so[i]; k < so[i + 1]; ++k)
                    for (auto& e : rows[sr[k]]) sc[e.tag] += e.w;
                u64 d[kDyn];
                dyn_feats(c, i, t1, t2, d);
                for (int j = 0; j < kDyn; ++j) {
                    auto it = idx.find(d[j]);
                    if (it != idx.end())
                        for (auto& e : rows[it->second]) sc[e.tag] += e.w;
                }
                int best = 0;
                for (int t = 1; t < nt; ++t)
                    if (sc[t] > sc[best]) best = t;
                pred = static_cast<u16>(best);
                ++clock;
                ++n;
                if (pred == gold[i]) {
                    ++ok;
                } else {
                    for (u32 k = so[i]; k < so[i + 1]; ++k) {
                        upd(sr[k], gold[i], +1.f);
                        upd(sr[k], pred, -1.f);
                    }
                    for (int j = 0; j < kDyn; ++j) {
                        const u32 r = row(d[j]);
                        upd(r, gold[i], +1.f);
                        upd(r, pred, -1.f);
                    }
                }
            }
            t2 = t1;
            t1 = pred;
        }
    }
    // Averaged weights -> inference tagger. Entries with |w| <= prune are dropped.
    Tagger snapshot(float prune) const {
        Tagger m;
        m.tags = tags;
        m.dict = dict;
        std::vector<u32> keep;
        std::vector<std::vector<std::pair<u16, float>>> avg(rows.size());
        std::size_t nrows = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            for (auto& e : rows[r]) {
                const float a = static_cast<float>((e.tot + static_cast<double>(e.w) * (clock - e.ts)) / clock);
                if (std::abs(a) > prune) avg[r].push_back({e.tag, a});
            }
            nrows += !avg[r].empty();
        }
        m.feat.init(nrows);
        m.off.assign(1, 0);
        u32 out = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            if (avg[r].empty()) continue;
            m.feat.put(rowkey[r], out++);
            for (auto& p : avg[r]) {
                m.etag.push_back(p.first);
                m.ew.push_back(p.second);
            }
            m.off.push_back(static_cast<u32>(m.etag.size()));
        }
        return m;
    }
};

double accuracy(const Tagger& m, const std::vector<Ctx>& ctx, const std::vector<std::vector<u16>>& gold) {
    std::size_t n = 0, ok = 0;
    std::vector<u16> out;
    for (std::size_t s = 0; s < ctx.size(); ++s) {
        m.tag(ctx[s], out);
        for (std::size_t i = 0; i < out.size(); ++i) {
            ++n;
            ok += out[i] == gold[s][i];
        }
    }
    return n ? static_cast<double>(ok) / static_cast<double>(n) : 0.0;
}

}  // namespace

namespace {

// ---------------------------------------------------------------- chunker (rules over UPOS)
// kind: 'N' nominal NP (head NOUN/PROPN), 'R' pronoun NP, 'V' verb group, 'P' prepositional phrase.
struct Chunk {
    int b, e, head;  // [b, e), head token index
    char kind;
};
using Strs = std::vector<std::string>;

bool is_poss_pron(const std::string& w) {
    static const std::set<std::string> s = {"my", "your", "his", "her", "its", "our", "their", "whose"};
    return s.count(lower(w)) > 0;
}
bool participle(const std::string& w) {
    const std::string l = lower(w);
    return l.size() > 4 && (l.compare(l.size() - 3, 3, "ing") == 0 || l.compare(l.size() - 2, 2, "ed") == 0 ||
                            l.compare(l.size() - 2, 2, "en") == 0);
}

// DET? DET? (NUM|ADJ|ADV* ADJ|participle)* (NOUN|PROPN)+   (hyphen allowed between nouns); lone PRON.
std::vector<Chunk> np_chunks(const Strs& w, const Strs& t) {
    const int n = static_cast<int>(w.size());
    std::vector<Chunk> out;
    auto nounish = [&](int k) { return k < n && (t[k] == "NOUN" || t[k] == "PROPN"); };
    int i = 0;
    while (i < n) {
        int j = i;
        if (t[j] == "DET" || (t[j] == "PRON" && is_poss_pron(w[j]))) {
            ++j;
            if (j < n && t[j] == "DET") ++j;  // "all the", "both these"
        }
        auto modifier = [&](int x) { return x < n && (t[x] == "ADJ" || t[x] == "NUM" || t[x] == "ADV"); };
        auto headish = [&](int x) { return x < n && (nounish(x) || t[x] == "ADJ" || t[x] == "NUM"); };
        while (j < n) {
            if (t[j] == "NUM" || t[j] == "ADJ") {
                ++j;
            } else if (t[j] == "ADV") {
                int k = j;
                while (k < n && t[k] == "ADV") ++k;
                if (k < n && (t[k] == "ADJ" || t[k] == "NUM")) j = k;
                else break;
            } else if (t[j] == "PUNCT" && j > i && j + 1 < n && modifier(j - 1) &&
                       ((w[j] == "," && t[j - 1] == "ADJ" && t[j + 1] == "ADJ") ||
                        (w[j] == "-" && (headish(j + 1) || (t[j + 1] == "VERB" && participle(w[j + 1])))))) {
                ++j;  // "a bad , bad tradeoff", "a 15 - year term"
            } else if (t[j] == "VERB" && j + 1 < n && participle(w[j]) &&
                       ((j > i && headish(j + 1)) ||
                        (j == i && headish(j + 1) && (j == 0 || t[j - 1] == "ADP" || t[j - 1] == "VERB" || t[j - 1] == "CCONJ" ||
                                                      t[j - 1] == "SCONJ" || t[j - 1] == "PUNCT")))) {
                ++j;  // "the running water", "seized suspected terrorists"
            } else {
                break;
            }
        }
        int k = j;
        while (k < n && (nounish(k) || (k > j && t[k] == "PUNCT" && w[k] == "-" && nounish(k + 1)))) ++k;
        if (k > j) {
            out.push_back({i, k, k - 1, 'N'});
            i = k;
        } else if (t[i] == "PRON") {
            out.push_back({i, i + 1, i, 'R'});
            ++i;
        } else {
            ++i;
        }
    }
    return out;
}

std::vector<Chunk> all_chunks(const Strs& w, const Strs& t) {
    const int n = static_cast<int>(w.size());
    std::vector<Chunk> out = np_chunks(w, t);
    const std::vector<Chunk> nps = out;
    auto verbish = [&](int k) { return k < n && (t[k] == "AUX" || t[k] == "VERB"); };
    // Verb groups: AUX/VERB runs; ADV/not between auxiliaries; "to" starts a group.
    int i = 0;
    while (i < n) {
        int j = i;
        bool start = verbish(j);
        if (!start && t[j] == "PART" && lower(w[j]) == "to") {
            int k = j + 1;
            while (k < n && t[k] == "ADV") ++k;
            if (verbish(k)) {
                start = true;
                j = k;
            }
        }
        if (!start) {
            ++i;
            continue;
        }
        int e = j + 1, last = j;
        while (e < n) {
            if (verbish(e)) {
                last = e;
                ++e;
            } else if (t[e] == "ADV" || (t[e] == "PART" && (lower(w[e]) == "n't" || lower(w[e]) == "not"))) {
                int k = e;
                while (k < n && (t[k] == "ADV" || (t[k] == "PART" && (lower(w[k]) == "n't" || lower(w[k]) == "not")))) ++k;
                if (verbish(k)) e = k;
                else break;
            } else {
                break;
            }
        }
        out.push_back({i, last + 1, last, 'V'});
        i = last + 1;
    }
    // Prepositional phrases: ADP + NP.
    for (const auto& c : nps)
        if (c.b > 0 && t[c.b - 1] == "ADP") out.push_back({c.b - 1, c.e, c.head, 'P'});
    std::sort(out.begin(), out.end(), [](const Chunk& a, const Chunk& b) {
        return a.b != b.b ? a.b < b.b : a.e > b.e;
    });
    return out;
}

std::string join(const Strs& w, int b, int e) {
    std::string s;
    for (int i = b; i < e; ++i) {
        if (i > b) s += ' ';
        s += w[i];
    }
    return s;
}
// "[PP in [NP what year]] did [NP the Anschluss] ..."
std::string bracket(const Strs& w, const std::vector<Chunk>& ch) {
    auto name = [](char k) { return k == 'N' ? "NP" : k == 'R' ? "NP" : k == 'V' ? "VG" : "PP"; };
    std::string s;
    const int n = static_cast<int>(w.size());
    for (int i = 0; i < n; ++i) {
        for (const auto& c : ch)
            if (c.b == i) s += std::string("[") + name(c.kind) + " ";
        if (i) {
            // keep spacing sane: no space directly after an opening bracket
        }
        s += w[i];
        std::vector<const Chunk*> closing;
        for (const auto& c : ch)
            if (c.e == i + 1) closing.push_back(&c);
        std::sort(closing.begin(), closing.end(), [](const Chunk* a, const Chunk* b) { return a->b > b->b; });
        for (std::size_t k = 0; k < closing.size(); ++k) s += "]";
        if (i + 1 < n) s += ' ';
    }
    return s;
}

// ---------------------------------------------------------------- gold NPs from dependency trees
// Head NOUN/PROPN plus its det / predet / amod / nummod / possessive-pronoun / compound / flat dependents
// (and adverbs of an included adjective); punctuation inside the span (e.g. a hyphen) is filled in; the
// span is the contiguous run that contains the head. Nouns that are compound/flat parts of a noun are
// absorbed into the parent's span. With_pron also returns single-token pronoun NPs.
std::vector<std::pair<int, int>> gold_nps(const Sent& s, bool with_pron) {
    const int n = static_cast<int>(s.w.size());
    std::vector<std::vector<int>> ch(n);
    for (int i = 0; i < n; ++i)
        if (s.head[i] > 0) ch[s.head[i] - 1].push_back(i);
    auto nounish = [&](int i) { return s.upos[i] == "NOUN" || s.upos[i] == "PROPN"; };
    std::vector<std::pair<int, int>> out;
    for (int h = 0; h < n; ++h) {
        if (with_pron && s.upos[h] == "PRON" && s.deprel[h] != "nmod:poss") out.push_back({h, h + 1});
        if (!nounish(h)) continue;
        const std::string& dr = s.deprel[h];
        if ((dr == "compound" || dr == "flat" || dr == "flat:name") && s.head[h] > 0 && nounish(s.head[h] - 1)) continue;
        std::vector<char> in(n, 0);
        in[h] = 1;
        std::function<void(int)> grow = [&](int m) {
            for (int d : ch[m]) {
                const std::string& r = s.deprel[d];
                bool take = false;
                if (nounish(m)) {
                    if (d < m && (r == "det" || r == "det:predet" || r == "amod" || r == "nummod")) take = true;
                    else if (d < m && r == "nmod:poss" && s.upos[d] == "PRON") take = true;
                    else if (r == "compound" || r == "flat" || r == "flat:name") take = true;
                } else if (s.upos[m] == "ADJ" || s.upos[m] == "NUM") {
                    if (d < m && r == "advmod" && s.upos[d] == "ADV") take = true;
                }
                if (take) {
                    in[d] = 1;
                    grow(d);
                }
            }
        };
        grow(h);
        int lo = n, hi = -1;
        for (int i = 0; i < n; ++i)
            if (in[i]) {
                lo = std::min(lo, i);
                hi = std::max(hi, i);
            }
        for (int p = lo; p <= hi; ++p)
            if (!in[p] && s.upos[p] == "PUNCT") in[p] = 1;
        int l = h, r = h;
        while (l > 0 && in[l - 1]) --l;
        while (r + 1 < n && in[r + 1]) ++r;
        while (l < h && s.upos[l] == "PUNCT") ++l;
        while (r > h && s.upos[r] == "PUNCT") --r;
        out.push_back({l, r + 1});
    }
    return out;
}

// ---------------------------------------------------------------- question parse helper
struct QParse {
    std::string wh, wh_phrase, wh_head, aux, verb, focus, entity, entity_run, entity_via;
};
bool is_wh(const std::string& l) {
    static const std::set<std::string> s = {"who", "whom", "whose", "what", "which", "when", "where", "why", "how"};
    return s.count(l) > 0;
}

QParse parse_question(const Strs& w, const Strs& t, const std::vector<Chunk>& ch) {
    QParse q;
    const int n = static_cast<int>(w.size());
    int wh = -1;
    for (int i = 0; i < n; ++i)
        if (is_wh(lower(w[i]))) {
            wh = i;
            break;
        }
    int wb = 0, we = 0;  // wh-phrase [wb, we)
    if (wh >= 0) {
        q.wh = w[wh];
        wb = wh;
        we = wh + 1;
        if (wb > 0 && t[wb - 1] == "ADP") --wb;  // "In what year"
        const std::string lw = lower(w[wh]);
        if (lw == "how" && wh + 1 < n && (t[wh + 1] == "ADJ" || t[wh + 1] == "ADV")) we = wh + 2;  // how many / long
        for (const auto& c : ch) {
            if (c.kind != 'N') continue;
            const bool starts_here = c.b == wh || (lw == "how" && c.b == wh + 1);
            if (starts_here && c.head >= wh + 1) {
                q.wh_head = w[c.head];
                we = std::max(we, c.e);
                break;
            }
        }
        q.wh_phrase = join(w, wb, we);
    }
    // Auxiliary and main verb: first VERB after the wh-phrase, before a clause boundary.
    std::vector<int> auxs;
    int verb = -1;
    for (int k = we; k < n; ++k) {
        const std::string lw = lower(w[k]);
        if (t[k] == "PUNCT" || t[k] == "SCONJ" || t[k] == "CCONJ") break;
        if (is_wh(lw) || lw == "that") break;
        if (t[k] == "AUX") auxs.push_back(k);
        else if (t[k] == "VERB") {
            verb = k;
            break;
        }
    }
    if (verb < 0 && !auxs.empty()) {  // copular question: "What is the capital of Algeria?"
        verb = auxs.back();
        auxs.pop_back();
    }
    for (int a : auxs) q.aux += (q.aux.empty() ? "" : " ") + w[a];
    if (verb >= 0) q.verb = w[verb];
    // Focus noun: wh_head, else the predicate noun after a copula.
    if (!q.wh_head.empty()) q.focus = q.wh_head;
    else if (verb >= 0 && t[verb] == "AUX" && wh >= 0)
        for (const auto& c : ch)
            if (c.kind == 'N' && c.b > verb && (c.b >= we)) {
                q.focus = w[c.head];
                break;
            }
    // Main entity: longest PROPN run outside the wh-phrase (ties: the later one), else the last nominal NP.
    int bestb = -1, beste = -1;
    for (int i = 0; i < n;) {
        if (t[i] == "PROPN" && !(i >= wb && i < we)) {
            int j = i;
            while (j < n && t[j] == "PROPN" && !(j >= wb && j < we)) ++j;
            if (bestb < 0 || j - i >= beste - bestb) {
                bestb = i;
                beste = j;
            }
            i = j;
        } else {
            ++i;
        }
    }
    if (bestb >= 0) {
        q.entity_via = "propn-run";
        q.entity_run = join(w, bestb, beste);
        q.entity = q.entity_run;
        for (const auto& c : ch)
            if (c.kind == 'N' && c.b <= bestb && c.e >= beste) q.entity = join(w, c.b, c.e);
    } else {
        for (const auto& c : ch)
            if (c.kind == 'N' && !(c.b < we && c.e > wb && wh >= 0)) {
                q.entity = join(w, c.b, c.e);
                q.entity_via = "last-np";
            }
    }
    return q;
}

// ---------------------------------------------------------------- tokenizer for `tag` / `question` input
std::size_t lead_punct(const std::string& s) {
    if (s.empty()) return 0;
    if (std::strchr("([{\"`$#", s[0])) return 1;
    static const char* u[] = {"\xE2\x80\x9C", "\xE2\x80\x98", "\xC2\xAB", "\xC2\xA1", "\xC2\xBF"};
    for (auto p : u)
        if (s.compare(0, std::strlen(p), p) == 0) return std::strlen(p);
    return 0;
}
std::size_t trail_punct(const std::string& s) {
    if (s.empty()) return 0;
    if (std::strchr(".,;:!?)]}\"'%", s.back())) return 1;
    static const char* u[] = {"\xE2\x80\x9D", "\xE2\x80\x99", "\xC2\xBB", "\xE2\x80\xA6"};
    for (auto p : u) {
        const std::size_t l = std::strlen(p);
        if (s.size() >= l && s.compare(s.size() - l, l, p) == 0) return l;
    }
    return 0;
}
Strs tokenize(const std::string& line) {
    static const std::set<std::string> abbr = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "jr.", "sr.", "u.s.", "no.", "inc.", "co."};
    Strs out;
    std::istringstream is(line);
    std::string chunk;
    while (is >> chunk) {
        Strs lead, trail;
        std::string core = chunk;
        while (std::size_t l = lead_punct(core)) {
            lead.push_back(core.substr(0, l));
            core.erase(0, l);
        }
        while (!core.empty()) {
            if (core.size() >= 3 && core.compare(core.size() - 3, 3, "...") == 0) {
                trail.push_back("...");
                core.erase(core.size() - 3);
                continue;
            }
            const std::size_t l = trail_punct(core);
            if (!l) break;
            if (core.back() == '.') {
                const std::string lc = lower(core);
                bool dotted = lc.size() >= 2;  // U.S. style: letters separated by periods
                for (std::size_t i = 0; i < lc.size(); ++i)
                    if ((i % 2 == 0) != (lc[i] != '.')) dotted = false;
                if (abbr.count(lc) || dotted) break;
            }
            trail.insert(trail.begin(), core.substr(core.size() - l));
            core.erase(core.size() - l);
        }
        for (auto& x : lead) out.push_back(x);
        if (!core.empty()) {
            const std::string lc = lower(core);
            auto ends = [&](const char* suf) {
                const std::size_t l = std::strlen(suf);
                return lc.size() > l && lc.compare(lc.size() - l, l, suf) == 0;
            };
            if (lc == "cannot") {
                out.push_back(core.substr(0, 3));
                out.push_back(core.substr(3));
            } else if (lc == "can't" || lc == "won't" || lc == "shan't") {
                out.push_back(core.substr(0, lc.size() - 3));
                out.push_back(core.substr(lc.size() - 3));
            } else if (ends("n't")) {
                out.push_back(core.substr(0, core.size() - 3));
                out.push_back(core.substr(core.size() - 3));
            } else {
                static const char* suf[] = {"'s", "'re", "'ve", "'ll", "'d", "'m"};
                bool done = false;
                for (auto sx : suf)
                    if (ends(sx) && core.find('-') == std::string::npos) {
                        const std::size_t l = std::strlen(sx);
                        out.push_back(core.substr(0, core.size() - l));
                        out.push_back(core.substr(core.size() - l));
                        done = true;
                        break;
                    }
                if (!done) out.push_back(core);
            }
        }
        for (auto& x : trail) out.push_back(x);
    }
    return out;
}

}  // namespace

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Opts {
    int epochs = 15, patience = 3, dict_min = 20;
    double dict_ratio = 0.97;
    float prune = 0.f;
    bool use_dict = true, xpos = true;
};

struct Acc {
    std::size_t n = 0, c = 0;
    void add(bool ok) {
        ++n;
        c += ok;
    }
    double pct() const { return n ? 100.0 * static_cast<double>(c) / static_cast<double>(n) : 0.0; }
};

std::vector<Ctx> build_ctx(const std::vector<Sent>& S) {
    std::vector<Ctx> c(S.size());
    for (std::size_t i = 0; i < S.size(); ++i) c[i].build(S[i].w);
    return c;
}
const Strs& tagcol(const Sent& s, bool xpos) { return xpos ? s.xpos : s.upos; }
std::vector<std::vector<u16>> gold_ids(const std::vector<Sent>& S, bool xpos, const std::vector<std::string>& tags) {
    std::unordered_map<std::string, u16> id;
    for (std::size_t i = 0; i < tags.size(); ++i) id[tags[i]] = static_cast<u16>(i);
    std::vector<std::vector<u16>> g(S.size());
    for (std::size_t s = 0; s < S.size(); ++s)
        for (auto& t : tagcol(S[s], xpos)) {
            auto it = id.find(t);
            g[s].push_back(it == id.end() ? kNone : it->second);
        }
    return g;
}

Tagger train_one(const char* name, const std::vector<Sent>& tr, const std::vector<Sent>& dv, const std::vector<Ctx>& trc,
                 const std::vector<Ctx>& dvc, bool xpos, const Opts& o) {
    std::set<std::string> ts;
    for (auto& s : tr)
        for (auto& t : tagcol(s, xpos)) ts.insert(t);
    Trainer T;
    T.tags.assign(ts.begin(), ts.end());
    if (T.tags.size() >= static_cast<std::size_t>(kMaxTags)) {
        std::fprintf(stderr, "too many tags\n");
        std::exit(1);
    }
    TrainSet D;
    D.ctx = trc;
    D.gold = gold_ids(tr, xpos, T.tags);
    const auto dgold = gold_ids(dv, xpos, T.tags);
    // Tag dictionary: frequent words with one dominant tag.
    std::unordered_map<u64, std::map<u16, int>> cnt;
    for (std::size_t s = 0; s < tr.size(); ++s)
        for (std::size_t i = 0; i < tr[s].w.size(); ++i) cnt[hs(tr[s].w[i])][D.gold[s][i]]++;
    std::vector<std::pair<u64, u16>> dent;
    for (auto& kv : cnt) {
        int tot = 0, top = 0;
        u16 tt = 0;
        for (auto& p : kv.second) {
            tot += p.second;
            if (p.second > top) {
                top = p.second;
                tt = p.first;
            }
        }
        if (o.use_dict && tot >= o.dict_min && top >= o.dict_ratio * tot) dent.push_back({kv.first, tt});
    }
    T.dict.init(dent.size());
    for (auto& p : dent) T.dict.put(p.first, p.second);
    D.dtag.resize(tr.size());
    std::size_t dtok = 0, ntok = 0;
    for (std::size_t s = 0; s < tr.size(); ++s)
        for (std::size_t i = 0; i < tr[s].w.size(); ++i) {
            u32 v;
            const bool hit = T.dict.get(hs(tr[s].w[i]), v);
            D.dtag[s].push_back(hit ? static_cast<u16>(v) : kNone);
            dtok += hit;
            ++ntok;
        }
    D.srow.resize(tr.size());
    D.soff.resize(tr.size());
    T.prepare(D);
    std::printf("[%s] %zu tags, tag dictionary %zu words covering %.1f%% of train tokens, %zu feature rows\n", name,
                T.tags.size(), dent.size(), 100.0 * static_cast<double>(dtok) / static_cast<double>(ntok), T.rows.size());
    std::vector<std::size_t> order(tr.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    Tagger best;
    double bestacc = -1;
    int bestep = 0, since = 0;
    for (int ep = 1; ep <= o.epochs; ++ep) {
        const auto t0 = Clock::now();
        std::mt19937 rng(1234 + static_cast<unsigned>(ep));
        std::shuffle(order.begin(), order.end(), rng);
        std::size_t n = 0, ok = 0;
        for (std::size_t s : order) T.train_sentence(D, s, n, ok);
        Tagger m = T.snapshot(o.prune);
        m.use_dict = o.use_dict;
        const double acc = accuracy(m, dvc, dgold);
        std::printf("[%s] epoch %2d  online-train %.2f%%  dev %.2f%%  (%.1fs)\n", name, ep,
                    n ? 100.0 * static_cast<double>(ok) / static_cast<double>(n) : 0.0, 100 * acc, secs(t0, Clock::now()));
        std::fflush(stdout);
        if (acc > bestacc) {
            bestacc = acc;
            bestep = ep;
            best = std::move(m);
            since = 0;
        } else if (++since >= o.patience) {
            break;
        }
    }
    std::printf("[%s] best epoch %d, dev %.2f%%\n", name, bestep, 100 * bestacc);
    return best;
}

int cmd_train(int argc, char** argv) {
    if (argc < 5) return std::fprintf(stderr, "usage: pos_tagger train <train.conllu> <dev.conllu> <model_out> [options]\n"), 2;
    Opts o;
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        auto nextv = [&]() { return i + 1 < argc ? argv[++i] : "0"; };
        if (a == "--epochs") o.epochs = std::atoi(nextv());
        else if (a == "--patience") o.patience = std::atoi(nextv());
        else if (a == "--prune") o.prune = static_cast<float>(std::atof(nextv()));
        else if (a == "--dict-min") o.dict_min = std::atoi(nextv());
        else if (a == "--dict-ratio") o.dict_ratio = std::atof(nextv());
        else if (a == "--no-dict") o.use_dict = false;
        else if (a == "--no-xpos") o.xpos = false;
    }
    std::vector<Sent> tr, dv;
    if (!read_conllu(argv[2], tr) || !read_conllu(argv[3], dv)) return std::fprintf(stderr, "cannot read data\n"), 1;
    std::size_t ntok = 0;
    for (auto& s : tr) ntok += s.w.size();
    std::printf("train: %zu sentences, %zu tokens; dev: %zu sentences\n", tr.size(), ntok, dv.size());
    const auto t0 = Clock::now();
    const auto trc = build_ctx(tr), dvc = build_ctx(dv);
    Model M;
    M.up = train_one("UPOS", tr, dv, trc, dvc, false, o);
    if (o.xpos) {
        M.xp = train_one("XPOS", tr, dv, trc, dvc, true, o);
        M.has_xp = true;
    }
    std::set<u64> vs;
    for (auto& s : tr)
        for (auto& w : s.w) vs.insert(hs(w));
    const std::vector<u64> vocab(vs.begin(), vs.end());
    if (!save_model(argv[4], M, vocab)) return 1;
    Model R;
    if (!load_model(argv[4], R)) return std::fprintf(stderr, "reload failed\n"), 1;
    R.up.use_dict = M.up.use_dict;
    std::printf("reloaded (int16-quantised) model, dev UPOS %.2f%%", 100 * accuracy(R.up, dvc, gold_ids(dv, false, R.up.tags)));
    if (R.has_xp) std::printf(", dev XPOS %.2f%%", 100 * accuracy(R.xp, dvc, gold_ids(dv, true, R.xp.tags)));
    std::printf("\ntotal training time %.1fs\n", secs(t0, Clock::now()));
    return 0;
}

struct PRF {
    std::size_t gold = 0, pred = 0, ok = 0;
    void add(const std::set<std::pair<int, int>>& g, const std::set<std::pair<int, int>>& p) {
        gold += g.size();
        pred += p.size();
        for (auto& x : p) ok += g.count(x);
    }
    double P() const { return pred ? 100.0 * static_cast<double>(ok) / static_cast<double>(pred) : 0; }
    double R() const { return gold ? 100.0 * static_cast<double>(ok) / static_cast<double>(gold) : 0; }
    double F() const { return P() + R() > 0 ? 2 * P() * R() / (P() + R()) : 0; }
};

int cmd_eval(int argc, char** argv) {
    if (argc < 4) return std::fprintf(stderr, "usage: pos_tagger eval <model> <file.conllu>\n"), 2;
    Model M;
    if (!load_model(argv[2], M)) return std::fprintf(stderr, "cannot load model\n"), 1;
    std::vector<Sent> S;
    if (!read_conllu(argv[3], S)) return std::fprintf(stderr, "cannot read data\n"), 1;
    std::size_t ntok = 0, nq = 0;
    for (auto& s : S) {
        ntok += s.w.size();
        nq += s.question();
    }
    std::size_t ndict = 0;
    for (u64 k : M.up.dict.keys) ndict += k != 0;
    std::printf("file: %s\nsentences: %zu  tokens: %zu  question sentences: %zu  UPOS tag-dictionary words: %zu\n", argv[3], S.size(),
                ntok, nq, ndict);
    // Timing: best of 3, single thread. Context building (token hashing) is included in the UPOS figure.
    double t_ctx = 1e9, t_up = 1e9, t_xp = 1e9;
    std::vector<Ctx> ctx;
    std::vector<std::vector<u16>> pu(S.size()), px(S.size());
    for (int rep = 0; rep < 3; ++rep) {
        auto a = Clock::now();
        ctx = build_ctx(S);
        auto b = Clock::now();
        for (std::size_t s = 0; s < S.size(); ++s) M.up.tag(ctx[s], pu[s]);
        auto c = Clock::now();
        if (M.has_xp)
            for (std::size_t s = 0; s < S.size(); ++s) M.xp.tag(ctx[s], px[s]);
        auto d = Clock::now();
        t_ctx = std::min(t_ctx, secs(a, b));
        t_up = std::min(t_up, secs(b, c));
        t_xp = std::min(t_xp, secs(c, d));
    }
    const auto gu = gold_ids(S, false, M.up.tags);
    Acc all, known, unk, ques, xall, xknown, xunk, xques;
    std::map<std::pair<std::string, std::string>, int> conf;
    for (std::size_t s = 0; s < S.size(); ++s) {
        const bool q = S[s].question();
        for (std::size_t i = 0; i < S[s].w.size(); ++i) {
            u32 dummy;
            const bool kn = M.vocab.get(hs(S[s].w[i]), dummy);
            const bool ok = pu[s][i] == gu[s][i];
            all.add(ok);
            (kn ? known : unk).add(ok);
            if (q) ques.add(ok);
            if (!ok) conf[{S[s].upos[i], M.up.tags[pu[s][i]]}]++;
            if (M.has_xp) {
                const bool okx = M.xp.tags[px[s][i]] == S[s].xpos[i];
                xall.add(okx);
                (kn ? xknown : xunk).add(okx);
                if (q) xques.add(okx);
            }
        }
    }
    auto line = [](const char* nm, const Acc& a) { std::printf("  %-22s %6.2f%%  (%zu/%zu)\n", nm, a.pct(), a.c, a.n); };
    std::printf("UPOS accuracy\n");
    line("all tokens", all);
    line("known words", known);
    line("unknown words", unk);
    line("question sentences", ques);
    if (M.has_xp) {
        std::printf("XPOS accuracy\n");
        line("all tokens", xall);
        line("known words", xknown);
        line("unknown words", xunk);
        line("question sentences", xques);
    }
    const double tk = static_cast<double>(ntok);
    std::printf("speed (single thread, best of 3, %zu tokens)\n", ntok);
    std::printf("  token hashing          %8.0f tokens/s\n", tk / t_ctx);
    std::printf("  UPOS tag only          %8.0f tokens/s\n", tk / t_up);
    std::printf("  hashing + UPOS         %8.0f tokens/s\n", tk / (t_ctx + t_up));
    if (M.has_xp) std::printf("  hashing + UPOS + XPOS  %8.0f tokens/s\n", tk / (t_ctx + t_up + t_xp));
    std::vector<std::pair<int, std::pair<std::string, std::string>>> cv;
    for (auto& kv : conf) cv.push_back({kv.second, kv.first});
    std::sort(cv.rbegin(), cv.rend());
    std::printf("top UPOS confusions (gold -> predicted)\n");
    for (std::size_t i = 0; i < cv.size() && i < 10; ++i)
        std::printf("  %-6s -> %-6s %d\n", cv[i].second.first.c_str(), cv[i].second.second.c_str(), cv[i].first);
    // NP chunker against NPs from the gold dependency trees.
    PRF gold_tags, pred_tags, pred_pron;
    for (std::size_t s = 0; s < S.size(); ++s) {
        Strs pt;
        for (u16 t : pu[s]) pt.push_back(M.up.tags[t]);
        auto toset = [](const std::vector<Chunk>& ch, bool pron) {
            std::set<std::pair<int, int>> r;
            for (auto& c : ch)
                if (c.kind == 'N' || (pron && c.kind == 'R')) r.insert({c.b, c.e});
            return r;
        };
        auto gs = gold_nps(S[s], false);
        auto gp = gold_nps(S[s], true);
        const std::set<std::pair<int, int>> G(gs.begin(), gs.end()), GP(gp.begin(), gp.end());
        gold_tags.add(G, toset(np_chunks(S[s].w, S[s].upos), false));
        pred_tags.add(G, toset(np_chunks(S[s].w, pt), false));
        pred_pron.add(GP, toset(np_chunks(S[s].w, pt), true));
    }
    std::printf("NP chunker (exact span match against NPs derived from gold dependency trees)\n");
    auto pl = [](const char* nm, const PRF& p) {
        std::printf("  %-34s gold %5zu pred %5zu  P %5.2f  R %5.2f  F1 %5.2f\n", nm, p.gold, p.pred, p.P(), p.R(), p.F());
    };
    pl("nominal NPs, gold UPOS (ceiling)", gold_tags);
    pl("nominal NPs, predicted UPOS", pred_tags);
    pl("nominal + pronoun NPs, predicted", pred_pron);
    return 0;
}

struct Tagged {
    Strs w, up, xp;
};
Tagged run_tagger(const Model& M, const std::string& line) {
    Tagged r;
    r.w = tokenize(line);
    if (r.w.empty()) return r;
    Ctx c;
    c.build(r.w);
    std::vector<u16> o;
    M.up.tag(c, o);
    for (u16 t : o) r.up.push_back(M.up.tags[t]);
    if (M.has_xp) {
        M.xp.tag(c, o);
        for (u16 t : o) r.xp.push_back(M.xp.tags[t]);
    }
    return r;
}

int cmd_tag(int argc, char** argv) {
    if (argc < 3) return std::fprintf(stderr, "usage: pos_tagger tag <model> [--xpos] [--chunks]  < lines\n"), 2;
    Model M;
    if (!load_model(argv[2], M)) return std::fprintf(stderr, "cannot load model\n"), 1;
    bool xp = false, chunks = false;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--xpos") xp = M.has_xp;
        if (std::string(argv[i]) == "--chunks") chunks = true;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        Tagged r = run_tagger(M, line);
        if (r.w.empty()) {
            std::printf("\n");
            continue;
        }
        for (std::size_t i = 0; i < r.w.size(); ++i)
            std::printf("%s%s/%s%s%s", i ? " " : "", r.w[i].c_str(), r.up[i].c_str(), xp ? "/" : "", xp ? r.xp[i].c_str() : "");
        std::printf("\n");
        if (chunks) std::printf("  %s\n", bracket(r.w, all_chunks(r.w, r.up)).c_str());
    }
    return 0;
}

int cmd_question(int argc, char** argv) {
    if (argc < 3) return std::fprintf(stderr, "usage: pos_tagger question <model> [--tsv]  < questions\n"), 2;
    Model M;
    if (!load_model(argv[2], M)) return std::fprintf(stderr, "cannot load model\n"), 1;
    const bool tsv = argc > 3 && std::string(argv[3]) == "--tsv";
    auto dash = [](const std::string& s) { return s.empty() ? std::string("-") : s; };
    std::string line;
    if (tsv) std::printf("question\twh\twh_phrase\twh_head\taux\tverb\tfocus\tentity\tentity_via\n");
    while (std::getline(std::cin, line)) {
        Tagged r = run_tagger(M, line);
        if (r.w.empty()) continue;
        const auto ch = all_chunks(r.w, r.up);
        const QParse q = parse_question(r.w, r.up, ch);
        if (tsv) {
            std::printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", line.c_str(), dash(q.wh).c_str(), dash(q.wh_phrase).c_str(),
                        dash(q.wh_head).c_str(), dash(q.aux).c_str(), dash(q.verb).c_str(), dash(q.focus).c_str(),
                        dash(q.entity).c_str(), dash(q.entity_via).c_str());
            continue;
        }
        std::printf("Q: %s\n  tags      :", line.c_str());
        for (std::size_t i = 0; i < r.w.size(); ++i) std::printf(" %s/%s", r.w[i].c_str(), r.up[i].c_str());
        std::printf("\n  chunks    : %s\n", bracket(r.w, ch).c_str());
        std::printf("  wh        : %s\n  wh_phrase : %s\n  wh_head   : %s\n  aux       : %s\n  verb      : %s\n  focus     : %s\n",
                    dash(q.wh).c_str(), dash(q.wh_phrase).c_str(), dash(q.wh_head).c_str(), dash(q.aux).c_str(),
                    dash(q.verb).c_str(), dash(q.focus).c_str());
        std::printf("  entity    : %s  [%s]\n\n", dash(q.entity).c_str(), dash(q.entity_via).c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "train") return cmd_train(argc, argv);
    if (mode == "eval") return cmd_eval(argc, argv);
    if (mode == "tag") return cmd_tag(argc, argv);
    if (mode == "question") return cmd_question(argc, argv);
    std::fprintf(stderr,
                 "usage: pos_tagger train <train.conllu> <dev.conllu> <model_out> [--epochs N] [--prune X] [--no-dict] [--no-xpos]\n"
                 "       pos_tagger eval <model> <file.conllu>\n"
                 "       pos_tagger tag <model> [--xpos] [--chunks]   (stdin lines)\n"
                 "       pos_tagger question <model> [--tsv]          (stdin questions)\n");
    return 2;
}
