// cyphalm_tok_gen: TOK-A's token-level CyphaLM read through the exact byte-token bridge
// (docs/TOKENS_PLAN.md T3 and T4; docs/reports/TOKENS_T3_T4.md).
//
//   contract <enwik8> <TOK> [--n 150] [--len 48]
//       checks the tokenizer against the bridge's contract on snippets of the dev holdout
//   bias <enwik8> <TOK> [--bytes 65536] [--positions 400] [--order 5] [--cap 64]
//       on the dev slice (the holdout's second half): canonical-segmentation bits per byte (TOK-A's measure),
//       the exact marginal through the bridge (raw and renormalised), and the naive next-byte estimate
//       against the exact one at sampled positions
//   gen <enwik8> <TOK> <OUT> --prompts-file bench/eval/gen_prompts_{dev,test}.tsv [--temps 0.6,0.8,1.0] [--bytes 600]
//       GEN-A's harness for the token model: samples whole tokens after each prompt (healed by the bridge) at each
//       temperature; writes OUT (PROMPT and GEN lines for `scribe ladder --gen-trie`), OUT.cont.tsv (each human
//       continuation's bits through the bridge) and GEN_EVAL lines (continuation bits per byte, copy share)
//   words <enwik8> <SAMPLE_FILE>...
//       the invented-word rate of each sample file: the share of generated words (letter runs, lower case)
//       that never occur in the training slice
//
// The token model is TOK-A's: interpolated modified Kneser-Ney (orders 1-5) mixed with the ∞-gram over
// token IDs, λ per (match length x count) bucket fitted on the holdout's first half, as cyphalm_tok eval.
// Nothing here answers a question (PLAN rule 10).
#include "cypha/cyphalm/token_bridge.hpp"
#include "cypha/cyphalm/token_model.hpp"
#include "cypha/cyphalm/tokenizer.hpp"
#include "cypha/eval/scribe.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_set>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace br = cypha::cyphalm::bridge;
namespace tk = cypha::cyphalm::tok;

namespace {

constexpr std::size_t kTrain = 7340032, kSample = 8388608, kHalf = kTrain + (kSample - kTrain) / 2;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::string arg(const std::vector<std::string>& a, const std::string& k, const std::string& d = "") {
    for (std::size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] == k) return a[i + 1];
    return d;
}
bool read_prefix(const std::string& path, std::size_t n, std::string& out) {
    std::ifstream is(path, std::ios::binary);
    out.resize(n);
    is.read(out.data(), static_cast<std::streamsize>(n));
    return static_cast<std::size_t>(is.gcount()) == n;
}

// TOK-A's tokenizer as the bridge needs it. The stable position is the start of the last pretoken (BPE,
// Unigram, Words: GPT-2 pretokens; SpaceByte: chunks), moved back to the start of a whitespace run just
// before it (that run gives its last space to the pretoken after it, so its split depends on what follows).
// Entropy tokens are longest matches over the raw text: fixed once max_len bytes past their start are known.
// Words (a whole pretoken or its bytes) is not prefix-closed, so the bridge is not exact for it.
class TokaAdapter : public br::Tokenizer {
public:
    explicit TokaAdapter(const tk::Tokenizer& t) : t_(t) {}
    int vocab_size() const override { return static_cast<int>(t_.size()); }
    std::string_view token(int id) const override { return t_.pieces[static_cast<std::size_t>(id)]; }
    std::vector<int> encode(std::string_view s) const override {
        const std::vector<std::uint32_t> v = t_.encode(s);
        return std::vector<int>(v.begin(), v.end());
    }
    std::size_t stable_prefix(std::string_view s, std::size_t hint) const override {
        if (hint >= s.size()) return hint;
        const auto* u = reinterpret_cast<const std::uint8_t*>(s.data()) + hint;
        const std::size_t n = s.size() - hint;
        switch (t_.kind) {
            case tk::Kind::BPE:
            case tk::Kind::Unigram:
            case tk::Kind::Words: {
                std::size_t last = 0, prev = 0;
                bool any = false;
                tk::pretokens(u, n, [&](std::size_t a, std::size_t) {
                    if (any) prev = last;
                    last = a;
                    any = true;
                });
                if (last > 0 && tk::byte_class(u[last - 1]) >= 3) last = prev;
                return hint + last;
            }
            case tk::Kind::SpaceByte: {
                // A chunk ends after a space-like byte not preceded by another; encoded on its own, a substring
                // has no byte before it, so a start between two space-like bytes is not stable.
                std::size_t last = 0;
                tk::spacebyte_chunks(u, n, [&](std::size_t a, std::size_t) {
                    if (a == 0 || !(tk::spacelike(u[a]) && tk::spacelike(u[a - 1]))) last = a;
                });
                return hint + last;
            }
            case tk::Kind::Entropy: {
                std::size_t q = 0;
                while (q < n && q + t_.max_len <= n) {
                    const std::vector<std::uint32_t> e = t_.encode(std::string_view(s.data() + hint + q, t_.max_len));
                    q += t_.pieces[e[0]].size();
                }
                return hint + q;
            }
        }
        return hint;
    }

private:
    const tk::Tokenizer& t_;
};

// TOK-A's model: KN (orders 1..K) mixed with the token ∞-gram, λ per bucket.
class TokaModel : public br::Model {
public:
    TokaModel(const tk::KneserNey& kn, const tk::TokenInfini& inf, const double* lam, std::size_t cap)
        : kn_(kn), inf_(inf), cap_(cap) {
        std::copy(lam, lam + tk::kTokBuckets, lam_);
    }
    void next(std::span<const int> h, std::vector<double>& out) const override {
        const std::size_t keep = std::min(h.size(), cap_);
        buf_.assign(h.end() - static_cast<std::ptrdiff_t>(keep), h.end());
        kn_.dist(buf_.data(), buf_.size(), out);
        // The longest match and its count by binary search only; the followers of a frequent short context
        // (many occurrences to scan) are cached by that context, so the result is the same, only faster.
        const auto r = inf_.query(buf_.data(), buf_.size(), 0, cap_);
        if (r.n <= 0 || r.total == 0) return;  // no match: bucket 0, λ fitted to 0
        const std::vector<std::pair<std::uint32_t, std::uint64_t>>* fol = &fol_;
        if (r.total > 2000) {
            std::vector<std::uint32_t> key(buf_.end() - r.n, buf_.end());
            auto it = cache_.find(key);
            if (it == cache_.end()) {
                inf_.followers(buf_.data(), buf_.size(), cap_, -1, std::size_t{1} << 40, fol_);
                if (cache_.size() > 50000) cache_.clear();
                it = cache_.emplace(std::move(key), fol_).first;
            }
            fol = &it->second;
        } else {
            inf_.followers(buf_.data(), buf_.size(), cap_, -1, std::size_t{1} << 40, fol_);
        }
        const double l = lam_[tk::tok_bucket(r.n, r.total)];
        for (double& x : out) x *= (1 - l);
        for (const auto& [w, c] : *fol) out[w] += l * static_cast<double>(c) / static_cast<double>(r.total);
    }

private:
    const tk::KneserNey& kn_;
    const tk::TokenInfini& inf_;
    double lam_[tk::kTokBuckets];
    std::size_t cap_;
    mutable std::vector<std::uint32_t> buf_;
    mutable std::vector<std::pair<std::uint32_t, std::uint64_t>> fol_;
    mutable std::map<std::vector<std::uint32_t>, std::vector<std::pair<std::uint32_t, std::uint64_t>>> cache_;
};

// Builds KN and the ∞-gram on the training slice and fits λ on the holdout's first half (cyphalm_tok eval).
struct Built {
    tk::Tokenizer tok;
    std::vector<std::uint32_t> toks;
    std::size_t ntrain = 0;
    std::unique_ptr<tk::KneserNey> kn;
    std::unique_ptr<tk::TokenInfini> inf;
    double lam[tk::kTokBuckets];
};
std::unique_ptr<Built> build(const std::string& data, const std::string& tokpath, int K, std::size_t cap) {
    auto B = std::make_unique<Built>();
    B->tok = tk::Tokenizer::load(tokpath);
    const auto V = static_cast<std::uint32_t>(B->tok.size());
    const auto* d = reinterpret_cast<const std::uint8_t*>(data.data());
    B->tok.encode(d, kTrain, B->toks);
    B->ntrain = B->toks.size();
    std::vector<std::uint32_t> first;
    B->tok.encode(d + kTrain, kHalf - kTrain, first);
    B->kn = std::make_unique<tk::KneserNey>(B->toks.data(), B->ntrain, V, K);
    B->inf = std::make_unique<tk::TokenInfini>(B->toks.data(), B->ntrain, V);
    // λ on the first half, continuing the training stream (as cyphalm_tok eval).
    std::vector<std::uint32_t> all(B->toks);
    all.insert(all.end(), first.begin(), first.end());
    std::vector<double> pk, pi;
    std::vector<int> bk;
    int prev = -1;
    for (std::size_t i = B->ntrain; i < all.size(); ++i) {
        pk.push_back(B->kn->prob(all.data(), i, all[i]));
        const auto r = B->inf->query(all.data(), i, all[i], cap, prev < 0 ? -1 : prev + 1);
        prev = r.n;
        pi.push_back(r.total ? static_cast<double>(r.count) / static_cast<double>(r.total) : 0.0);
        bk.push_back(tk::tok_bucket(r.n, r.total));
    }
    tk::fit_lambdas(pi, pk, bk, pk.size(), tk::kTokBuckets, B->lam);
    return B;
}

int cmd_contract(const std::vector<std::string>& a) {
    std::string data;
    if (a.size() < 3 || !read_prefix(a[1], kSample, data)) {
        std::cerr << "usage: cyphalm_tok_gen contract <enwik8> <TOK> [--n 150] [--len 48]\n";
        return 2;
    }
    const tk::Tokenizer t = tk::Tokenizer::load(a[2]);
    TokaAdapter ad(t);
    const int n = std::stoi(arg(a, "--n", "150"));
    const std::size_t len = std::stoull(arg(a, "--len", "48"));
    std::mt19937_64 r(20261029);
    std::vector<std::string> samples;
    for (int k = 0; k < n; ++k) samples.push_back(data.substr(kHalf + r() % (kSample - kHalf - len), len));
    std::string why;
    const int bad = br::check_contract(ad, samples, &why);
    std::printf("contract %s: %d samples of %zu bytes, %d violations%s%s\n", a[2].c_str(), n, len, bad,
                bad ? "; first: " : "", why.c_str());
    return bad == 0 ? 0 : 1;
}

int cmd_bias(const std::vector<std::string>& a) {
    std::string data;
    if (a.size() < 3 || !read_prefix(a[1], kSample, data)) {
        std::cerr << "usage: cyphalm_tok_gen bias <enwik8> <TOK> [--bytes 65536] [--positions 400]\n";
        return 2;
    }
    const std::size_t nbytes = std::stoull(arg(a, "--bytes", "65536"));
    const int npos = std::stoi(arg(a, "--positions", "400"));
    const int K = std::stoi(arg(a, "--order", "5"));
    const std::size_t cap = std::stoull(arg(a, "--cap", "64"));
    const double t0 = now_s();
    const auto B = build(data, a[2], K, cap);
    const double t_build = now_s() - t0;
    TokaAdapter ad(B->tok);
    TokaModel model(*B->kn, *B->inf, B->lam, cap);
    br::Options raw_opt;
    raw_opt.renormalize = false;
    raw_opt.valid_cache = 20000;
    br::Options ren_opt;
    ren_opt.valid_cache = 20000;
    br::Bridge braw(ad, model, raw_opt), bren(ad, model, ren_opt);
    const std::string slice = data.substr(kHalf, nbytes);
    // Sampled positions for the next-byte comparison (fixed seed).
    std::mt19937_64 r(7);
    std::vector<char> at(nbytes + 1, 0);
    for (int k = 0; k < npos; ++k) at[1 + r() % (nbytes - 1)] = 1;
    br::Stream sr(braw), sn(bren);
    double nv_bits = 0, ex_bits = 0, tv = 0;
    int counted = 0;
    std::vector<double> q;
    for (std::size_t i = 0; i < nbytes; ++i) {
        if (at[i]) {
            // exact next byte (renormalised bridge) against the naive estimate: the first byte of the next
            // token after the canonical encoding of the bytes so far (raw model)
            const std::array<double, 256> ex = sn.next_byte();
            const std::vector<int> e = ad.encode(std::string_view(slice).substr(0, i));
            model.next(e, q);
            std::array<double, 256> nv{};
            for (int v = 0; v < ad.vocab_size(); ++v) nv[static_cast<unsigned char>(ad.token(v)[0])] += q[static_cast<std::size_t>(v)];
            const auto c = static_cast<unsigned char>(slice[i]);
            nv_bits += -std::log2(std::max(nv[c], 1e-300));
            ex_bits += -std::log2(std::max(ex[c], 1e-300));
            double d = 0;
            for (int b = 0; b < 256; ++b) d += std::abs(nv[static_cast<std::size_t>(b)] - ex[static_cast<std::size_t>(b)]);
            tv += d / 2;
            ++counted;
        }
        sr.push(std::string_view(slice).substr(i, 1));
        sn.push(std::string_view(slice).substr(i, 1));
    }
    const double can = -sr.log2_canonical() / static_cast<double>(nbytes);
    const double mraw = -sr.log2_prefix() / static_cast<double>(nbytes);
    const double mren = -sn.log2_prefix() / static_cast<double>(nbytes);
    const double cren = -sn.log2_canonical() / static_cast<double>(nbytes);
    std::printf("bias %s: %zu bytes of the dev slice from %zu; build %.0f s, read %.0f s\n", a[2].c_str(), nbytes, kHalf,
                t_build, now_s() - t0 - t_build);
    std::printf("  bits per byte: canonical (raw model) %.4f | marginal over covers, raw %.4f | renormalised: canonical %.4f, marginal %.4f\n",
                can, mraw, cren, mren);
    std::printf("  next byte at %d sampled positions: naive %.4f bits, exact %.4f bits (difference %.4f); mean total-variation distance %.4f\n",
                counted, nv_bits / counted, ex_bits / counted, nv_bits / counted - ex_bits / counted, tv / counted);
    return 0;
}

// cyphalm_trie's printable() and scribe's unescape_gen().
std::string printable(const std::string& s) {
    std::string o;
    const char* hex = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c >= 32 && c < 127) o += static_cast<char>(c);
        else {
            o += "\\x";
            o += hex[c >> 4];
            o += hex[c & 15];
        }
    }
    return o;
}
std::string unescape(const std::string& in) {
    std::string o;
    auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : 0); };
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\\' && i + 1 < in.size() && in[i + 1] == 'n') {
            o.push_back('\n');
            ++i;
        } else if (in[i] == '\\' && i + 3 < in.size() && in[i + 1] == 'x') {
            o.push_back(static_cast<char>((hex(in[i + 2]) << 4) | hex(in[i + 3])));
            i += 3;
        } else {
            o.push_back(in[i]);
        }
    }
    return o;
}

// GEN-A's harness, for the token model: prompts and human continuations from bench/eval/gen_prompts_{dev,test}.tsv
// (header, then id, doc, prompt, continuation; tab, newline and backslash escaped), three temperatures, 600 bytes,
// the same seeds as `cyphalm_trie sample`. Writes OUT (MODEL, PROMPT and GEN lines for `scribe ladder --gen-trie`;
// the kinds are trie_T0.6 / 0.8 / 1.0, the names SCRIBE's per-temperature rungs read), OUT.cont.tsv (each human
// continuation's bits, through the bridge: log2 P(prompt) - log2 P(prompt + continuation)) and GEN_EVAL lines.
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

// GEN-A's copy share: generated bytes inside 32-byte spans that occur in the training bytes [0, 7 MiB), raw or cleaned
// by SCRIBE (clean_wiki_documents), the same reference as cyphalm_trie sample --prompts-file.
struct Spans32 {
    static constexpr std::size_t K = 32;
    std::vector<std::uint64_t> h;
    static std::uint64_t hash(const std::uint8_t* p) {
        std::uint64_t x = 0xcbf29ce484222325ull;
        for (std::size_t i = 0; i < K; ++i) x = (x ^ p[i]) * 0x100000001b3ull;
        return x;
    }
    void add(const std::uint8_t* t, std::size_t n) {
        for (std::size_t i = 0; i + K <= n; ++i) h.push_back(hash(t + i));
    }
    Spans32(const std::uint8_t* t, std::size_t n) {
        add(t, n);
        for (const auto& d : cypha::eval::clean_wiki_documents(reinterpret_cast<const char*>(t), n))
            add(reinterpret_cast<const std::uint8_t*>(d.data()), d.size());
        std::sort(h.begin(), h.end());
        h.erase(std::unique(h.begin(), h.end()), h.end());
    }
    std::pair<std::size_t, std::size_t> covered(const std::string& s) const {
        const auto* b = reinterpret_cast<const std::uint8_t*>(s.data());
        std::vector<char> cov(s.size(), 0);
        for (std::size_t i = 0; i + K <= s.size(); ++i)
            if (std::binary_search(h.begin(), h.end(), hash(b + i)))
                std::fill(cov.begin() + static_cast<std::ptrdiff_t>(i), cov.begin() + static_cast<std::ptrdiff_t>(i + K), 1);
        return {static_cast<std::size_t>(std::count(cov.begin(), cov.end(), 1)), s.size()};
    }
};

int cmd_gen(const std::vector<std::string>& a) {
    std::string data;
    const std::string pf = arg(a, "--prompts-file");
    if (a.size() < 4 || pf.empty() || !read_prefix(a[1], kSample, data)) {
        std::cerr << "usage: cyphalm_tok_gen gen <enwik8> <TOK> <OUT> --prompts-file F [--temps 0.6,0.8,1.0] [--bytes 600] [--limit N]\n";
        return 2;
    }
    std::vector<double> temps;
    {
        std::stringstream ss(arg(a, "--temps", "0.6,0.8,1.0"));
        std::string t;
        while (std::getline(ss, t, ',')) temps.push_back(std::stod(t));
    }
    const std::size_t nbytes = std::stoull(arg(a, "--bytes", "600"));
    const long limit = std::stol(arg(a, "--limit", "-1"));
    std::vector<std::string> prompts, conts;
    {
        std::ifstream in(pf);
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) {
            std::vector<std::string> f;
            std::stringstream ss(line);
            std::string x;
            while (std::getline(ss, x, '\t')) f.push_back(x);
            if (f.size() < 4) continue;
            prompts.push_back(unescape_tsv(f[2]));
            conts.push_back(" " + unescape_tsv(f[3]));
        }
    }
    const std::size_t np = limit >= 0 ? std::min<std::size_t>(prompts.size(), static_cast<std::size_t>(limit)) : prompts.size();
    const double t0 = now_s();
    const auto B = build(data, a[2], 5, 64);
    TokaAdapter ad(B->tok);
    TokaModel model(*B->kn, *B->inf, B->lam, 64);
    br::Options opt;
    opt.valid_cache = 20000;
    br::Bridge b(ad, model, opt);
    std::ofstream out(a[3]), ct(a[3] + ".cont.tsv");
    if (!out || !ct) {
        std::cerr << "cannot write " << a[3] << "\n";
        return 1;
    }
    out << "MODEL\ttok\t" << a[2] << "\tKN5+INF\n";
    ct << "row\tbytes\tbits\n";
    std::vector<std::vector<std::string>> gens(temps.size());
    std::vector<double> bits(np), nb(np);
    for (std::size_t k = 0; k < np; ++k) {
        out << "PROMPT\t" << k << "\t" << printable(prompts[k]) << "\n";
        for (std::size_t ti = 0; ti < temps.size(); ++ti) {
            std::mt19937_64 rng(1000003ull * (k + 1) + ti * 7919ull);
            const std::string text = br::sample_after(b, prompts[k], nbytes, temps[ti], rng);
            char kind[32];
            std::snprintf(kind, sizeof kind, "trie_T%.1f", temps[ti]);
            out << "GEN\t" << kind << "\t" << printable(text) << "\n";
            gens[ti].push_back(text);
        }
        br::Stream st(b);
        st.push(prompts[k]);
        const double l0 = st.log2_prefix();
        st.push(conts[k]);
        bits[k] = l0 - st.log2_prefix();
        nb[k] = static_cast<double>(conts[k].size());
        ct << k << "\t" << conts[k].size() << "\t" << bits[k] << "\n";
        out.flush();
        if ((k + 1) % 25 == 0) std::cerr << "prompt " << (k + 1) << "/" << np << " (" << static_cast<int>(now_s() - t0) << " s)\n";
    }
    double sb = 0, sn = 0;
    for (std::size_t k = 0; k < np; ++k) {
        sb += bits[k];
        sn += nb[k];
    }
    std::vector<double> boot;
    std::mt19937_64 brng(20261029);
    std::uniform_int_distribution<std::size_t> pick(0, np - 1);
    for (int r = 0; r < 2000; ++r) {
        double x = 0, y = 0;
        for (std::size_t k = 0; k < np; ++k) {
            const std::size_t j = pick(brng);
            x += bits[j];
            y += nb[j];
        }
        boot.push_back(x / y);
    }
    std::sort(boot.begin(), boot.end());
    std::printf("GEN_EVAL cont_bpc %.4f ci95 %.4f %.4f bytes %.0f prompts %zu\n", sb / sn, boot[49], boot[1949], sn, np);
    const Spans32 sp(reinterpret_cast<const std::uint8_t*>(data.data()), kTrain);
    std::size_t all_c = 0, all_n = 0;
    std::printf("GEN_EVAL copy32");
    for (std::size_t ti = 0; ti < temps.size(); ++ti) {
        std::size_t cv = 0, n = 0;
        for (const auto& g : gens[ti]) {
            const auto [x, y] = sp.covered(g);
            cv += x;
            n += y;
        }
        all_c += cv;
        all_n += n;
        std::printf(" T%.1f %.4f", temps[ti], n ? static_cast<double>(cv) / static_cast<double>(n) : 0.0);
    }
    std::printf(" pooled %.4f\n", all_n ? static_cast<double>(all_c) / static_cast<double>(all_n) : 0.0);
    std::cerr << "gen done: " << np << " prompts, " << static_cast<int>(now_s() - t0) << " s\n";
    return 0;
}

int cmd_words(const std::vector<std::string>& a) {
    std::string data;
    if (a.size() < 3 || !read_prefix(a[1], kTrain, data)) {
        std::cerr << "usage: cyphalm_tok_gen words <enwik8> <SAMPLE_FILE>...\n";
        return 2;
    }
    auto words = [](const std::string& s, auto&& f) {
        std::string w;
        for (unsigned char c : s) {
            if (tk::is_letter(c) && c < 0x80) w.push_back(static_cast<char>(c | 32));
            else if (!w.empty()) {
                f(w);
                w.clear();
            }
        }
        if (!w.empty()) f(w);
    };
    std::unordered_set<std::string> seen;
    words(data, [&](const std::string& w) { seen.insert(w); });
    for (std::size_t k = 2; k < a.size(); ++k) {
        std::ifstream in(a[k]);
        std::string line;
        std::map<std::string, std::pair<long, long>> by;  // kind -> (words, invented)
        while (std::getline(in, line)) {
            if (line.compare(0, 4, "GEN\t") != 0) continue;
            const std::size_t t2 = line.find('\t', 4);
            const std::string kind = line.substr(4, t2 - 4);
            words(unescape(line.substr(t2 + 1)), [&](const std::string& w) {
                auto& e = by[kind];
                ++e.first;
                if (!seen.count(w)) ++e.second;
            });
        }
        long tw = 0, ti = 0;
        for (const auto& [kind, e] : by) {
            std::printf("%s %s: words %ld, invented %ld (%.4f)\n", a[k].c_str(), kind.c_str(), e.first, e.second,
                        e.first ? static_cast<double>(e.second) / static_cast<double>(e.first) : 0.0);
            tw += e.first;
            ti += e.second;
        }
        std::printf("%s all: words %ld, invented %ld (%.4f)\n", a[k].c_str(), tw, ti, tw ? static_cast<double>(ti) / static_cast<double>(tw) : 0.0);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> a(argv + 1, argv + argc);
    if (a.empty()) {
        std::cerr << "usage: cyphalm_tok_gen contract | bias | gen (see the file header)\n";
        return 2;
    }
    try {
        if (a[0] == "contract") return cmd_contract(a);
        if (a[0] == "bias") return cmd_bias(a);
        if (a[0] == "gen") return cmd_gen(a);
        if (a[0] == "words") return cmd_words(a);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "unknown command " << a[0] << "\n";
    return 2;
}
