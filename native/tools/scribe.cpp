// SCRIBE: the generated-text metric (docs/PLAN.md section 6). One tool for every use; it replaces
// scribe_ladder.cpp (v0 and v1). Library: native/include/cypha/eval/scribe.hpp (header-only, no neural network).
// Reports: docs/reports/SCRIBE_V0.md, SCRIBE_V1.md, SCRIBE_V2.md. Plan of the v2 study: docs/SCRIBE_PAPER_PLAN.md.
//
// Subcommands
//   docs     SRC --out F                                    cleaned documents, one JSON object per line
//   prompts  SRC --n 250 --seed S --set NAME --out F        prompt (100-300 B) + true human continuation (400-600 B,
//                                                           >= 3 sentences), one per document
//   samples  SRC --n 500 --seed S --set NAME [--short] [--system NAME] --out F
//                                                           samples cut from documents the way human samples are
//                                                           (sentence windows of 400-600 B with >= 3 sentences, or 60-260 B)
//   prepare  --in F --out G [--mode continuation|wikiraw|plain] [--min-bytes 100]
//                                                           model output -> the text that is scored (and that every
//                                                           baseline scores): continuation = whitespace collapsed and the
//                                                           unterminated tail cut; wikiraw = v1's clean_generated_text
//   ladder   --pack SRC --ref SRC --test SRC [--vectors P] [--profile v1|v2] [--seed S] [--chain-rng]
//            [--gen-trie F] [--model F]... [--export DIR] [--label L]
//                                                           the PLAN 6.4 V1 control ladder (and model rungs), report on stdout
//   score    --pack SRC --ref SRC [--vectors P] [--profile v1|v2] --in F --out G
//                                                           every text of F scored: axes, sub-scores, raw statistics
//   selftest                                                known-answer checks (CTest native_scribe_selftest)
//
// SRC is enwik:PATH:LO:HI (bytes [LO, HI) MiB of a MediaWiki XML dump such as enwik8/enwik9, cleaned page by page
// with clean_wiki_documents, as v0 and v1 did) or a JSONL file whose "text" fields are documents (cleaned with
// clean_plain_document: prose lines only). JSONL lines in and out: {"id", "set", "system", "text", "prompt"?, "ref"?}.
//
// v1 parity: `ladder --pack enwik:E:16:48 --ref enwik:E:48:56 --test enwik:E:56:64 --vectors v32 --profile v1
// --seed 20261027 --chain-rng --gen-trie gen_dev.txt` prints the v1 development ladder's rung means and pair verdicts to
// the printed digit, and `--test enwik:E:64:72 --seed 31415926 --gen-trie gen_fresh.txt` the v1 fresh ladder.
//
// Author: Odin Loch

#include "cypha/eval/scribe.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using namespace cypha::eval;
using cypha::eval::scribe_detail::is_alpha_a;
using cypha::eval::scribe_detail::is_digit_a;
using cypha::eval::scribe_detail::is_space_a;
using cypha::eval::scribe_detail::is_upper_a;

namespace {

constexpr std::size_t MiB = 1048576;

double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
double g_t0 = now_s();

// ===========================================================================
// JSON lines: a small reader for flat objects (nested values are skipped) and a writer
// ===========================================================================

std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 16);
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c >= 0x80) {
      const std::size_t L = scribe_detail::utf8_seq_len(s, i);
      if (L == 0) {
        o += "\\ufffd";  // invalid byte: replacement character, so every reader accepts the line
        ++i;
      } else {
        o.append(s, i, L);
        i += L;
      }
      continue;
    }
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          o += b;
        } else {
          o.push_back(static_cast<char>(c));
        }
    }
    ++i;
  }
  return o;
}

struct JsonObj {
  std::map<std::string, std::string> str;  // top-level string values
  std::map<std::string, double> num;       // top-level numbers
  std::string get(const std::string& k, const std::string& def = "") const {
    const auto it = str.find(k);
    return it == str.end() ? def : it->second;
  }
};

class JsonReader {
 public:
  explicit JsonReader(const std::string& s) : s_(s) {}
  bool object(JsonObj& o) {
    ws();
    if (!eat('{')) return false;
    ws();
    if (eat('}')) return true;
    while (true) {
      ws();
      std::string k;
      if (!string(k)) return false;
      ws();
      if (!eat(':')) return false;
      ws();
      if (peek() == '"') {
        std::string v;
        if (!string(v)) return false;
        o.str[k] = std::move(v);
      } else if (peek() == '{' || peek() == '[') {
        if (!skip_nested()) return false;
      } else {
        const std::size_t b = p_;
        while (p_ < s_.size() && s_[p_] != ',' && s_[p_] != '}' && !is_space_a(static_cast<unsigned char>(s_[p_]))) ++p_;
        const std::string tok = s_.substr(b, p_ - b);
        char* end = nullptr;
        const double v = std::strtod(tok.c_str(), &end);
        if (end && *end == '\0' && !tok.empty()) o.num[k] = v;
      }
      ws();
      if (eat(',')) continue;
      if (eat('}')) return true;
      return false;
    }
  }

 private:
  char peek() const { return p_ < s_.size() ? s_[p_] : '\0'; }
  bool eat(char c) {
    if (peek() != c) return false;
    ++p_;
    return true;
  }
  void ws() {
    while (p_ < s_.size() && is_space_a(static_cast<unsigned char>(s_[p_]))) ++p_;
  }
  static int hexv(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  }
  bool hex4(unsigned& v) {
    if (p_ + 4 > s_.size()) return false;
    v = 0;
    for (int k = 0; k < 4; ++k) {
      const int h = hexv(s_[p_ + static_cast<std::size_t>(k)]);
      if (h < 0) return false;
      v = v * 16 + static_cast<unsigned>(h);
    }
    p_ += 4;
    return true;
  }
  bool string(std::string& out) {
    if (!eat('"')) return false;
    while (p_ < s_.size()) {
      const char c = s_[p_++];
      if (c == '"') return true;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (p_ >= s_.size()) return false;
      const char e = s_[p_++];
      switch (e) {
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'u': {
          unsigned cp = 0;
          if (!hex4(cp)) return false;
          if (cp >= 0xD800 && cp <= 0xDBFF && p_ + 6 <= s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
            p_ += 2;
            unsigned lo = 0;
            if (!hex4(lo)) return false;
            if (lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            else cp = 0xFFFD;
          } else if (cp >= 0xD800 && cp <= 0xDFFF) {
            cp = 0xFFFD;
          }
          scribe_detail::append_utf8(out, cp);
          break;
        }
        default: out.push_back(e);  // \" \\ \/
      }
    }
    return false;
  }
  bool skip_nested() {
    int depth = 0;
    while (p_ < s_.size()) {
      const char c = s_[p_];
      if (c == '"') {
        std::string tmp;
        if (!string(tmp)) return false;
        continue;
      }
      ++p_;
      if (c == '{' || c == '[') ++depth;
      else if (c == '}' || c == ']') {
        if (--depth == 0) return true;
      }
    }
    return false;
  }
  const std::string& s_;
  std::size_t p_ = 0;
};

std::vector<JsonObj> read_jsonl(const std::string& path) {
  std::vector<JsonObj> v;
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::string line;
  std::size_t ln = 0, bad = 0;
  while (std::getline(in, line)) {
    ++ln;
    if (line.empty()) continue;
    JsonObj o;
    if (JsonReader(line).object(o)) v.push_back(std::move(o));
    else ++bad;
  }
  if (bad) std::fprintf(stderr, "%s: %zu of %zu lines are not JSON objects (skipped)\n", path.c_str(), bad, ln);
  return v;
}

// One output line: {"id":..,"set":..,"system":..,"text":..[,"prompt":..][,"ref":..]} plus raw extra JSON members.
std::string text_line(const std::string& id, const std::string& set, const std::string& system, const std::string& text,
                      const std::string& prompt = "", const std::string& ref = "", const std::string& extra = "") {
  std::string o = "{\"id\":\"" + json_escape(id) + "\",\"set\":\"" + json_escape(set) + "\",\"system\":\"" + json_escape(system) +
                  "\",\"text\":\"" + json_escape(text) + "\"";
  if (!prompt.empty()) o += ",\"prompt\":\"" + json_escape(prompt) + "\"";
  if (!ref.empty()) o += ",\"ref\":\"" + json_escape(ref) + "\"";
  if (!extra.empty()) o += "," + extra;
  o += "}";
  return o;
}

std::string num(double v, int prec = 6) {
  if (!std::isfinite(v)) return "null";
  char b[64];
  std::snprintf(b, sizeof b, "%.*g", prec, v);
  return b;
}

// ===========================================================================
// Sources and samples
// ===========================================================================

// enwik:PATH:LO:HI -> cleaned pages of bytes [LO, HI) MiB; anything else -> JSONL documents (clean_plain_document).
std::vector<std::string> load_docs(const std::string& spec) {
  if (spec.rfind("enwik:", 0) == 0) {
    const std::size_t c2 = spec.rfind(':'), c1 = spec.rfind(':', c2 - 1);
    if (c1 == std::string::npos || c1 < 6) {
      std::fprintf(stderr, "bad source %s (want enwik:PATH:LO:HI)\n", spec.c_str());
      std::exit(2);
    }
    const std::string path = spec.substr(6, c1 - 6);
    const double lo = std::strtod(spec.c_str() + c1 + 1, nullptr), hi = std::strtod(spec.c_str() + c2 + 1, nullptr);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      std::fprintf(stderr, "cannot open %s\n", path.c_str());
      std::exit(1);
    }
    in.seekg(0, std::ios::end);
    const std::size_t size = static_cast<std::size_t>(in.tellg());
    const std::size_t b = std::min(size, static_cast<std::size_t>(lo * MiB)), e = std::min(size, static_cast<std::size_t>(hi * MiB));
    std::vector<char> raw(e > b ? e - b : 0);
    in.seekg(static_cast<std::streamoff>(b));
    in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    return clean_wiki_documents(raw.data(), raw.size());
  }
  std::vector<std::string> docs;
  for (const JsonObj& o : read_jsonl(spec)) {
    std::string d = clean_plain_document(o.get("text"));
    if (!d.empty()) docs.push_back(std::move(d));
  }
  return docs;
}

struct Sample {
  std::string text;
  int doc = 0;
};

// Consecutive, non-overlapping windows of sentences with total length in [lo, hi] and at least
// min_sent sentences; the target length is drawn uniformly in [lo, hi] per window (v0 and v1's rule).
std::vector<Sample> make_samples(const std::vector<std::string>& docs, std::size_t lo, std::size_t hi, int min_sent, int max_per_doc,
                                 std::mt19937_64& rng) {
  std::vector<Sample> out;
  std::uniform_real_distribution<double> U(0.0, 1.0);
  for (std::size_t d = 0; d < docs.size(); ++d) {
    const std::vector<std::string> s = split_sentences(docs[d]);
    int got = 0;
    for (std::size_t i = 0; i < s.size() && got < max_per_doc;) {
      const double target = static_cast<double>(lo) + U(rng) * static_cast<double>(hi - lo);
      std::size_t j = i, total = 0;
      while (j < s.size() && static_cast<double>(total) < target) total += s[j++].size() + (total ? 1 : 0);
      if (total >= lo && total <= hi && static_cast<int>(j - i) >= min_sent) {
        std::string t;
        for (std::size_t k = i; k < j; ++k) {
          if (k > i) t.push_back(' ');
          t += s[k];
        }
        out.push_back({std::move(t), static_cast<int>(d)});
        ++got;
        i = j;
      } else {
        ++i;
      }
    }
  }
  return out;
}

std::vector<Sample> pick(std::vector<Sample> v, std::size_t n, std::mt19937_64& rng) {
  std::shuffle(v.begin(), v.end(), rng);
  if (v.size() > n) v.resize(n);
  return v;
}

std::string unescape_gen(const std::string& in) {
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

std::vector<std::string> split_ws(const std::string& s) {
  std::vector<std::string> v;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && is_space_a(static_cast<unsigned char>(s[i]))) ++i;
    std::size_t j = i;
    while (j < s.size() && !is_space_a(static_cast<unsigned char>(s[j]))) ++j;
    if (j > i) v.push_back(s.substr(i, j - i));
    i = j;
  }
  return v;
}

std::string join(const std::vector<std::string>& v, const char* sep = " ") {
  std::string o;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) o += sep;
    o += v[i];
  }
  return o;
}

// ===========================================================================
// Controls (unchanged from v0 and v1; each keeps sentence lengths, final punctuation and a capital first letter)
// ===========================================================================

std::string ctl_shuffle_sentences(const Sample& s, const std::vector<std::pair<int, std::string>>& pool, std::mt19937_64& rng) {
  const std::vector<std::string> own = split_sentences(s.text);
  std::string out;
  std::uniform_int_distribution<std::size_t> pickd(0, pool.size() - 1);
  for (std::size_t i = 0; i < own.size(); ++i) {
    const std::string* chosen = nullptr;
    for (int tries = 0; tries < 60; ++tries) {
      const auto& c = pool[pickd(rng)];
      if (c.first == s.doc) continue;
      chosen = &c.second;
      const double r = static_cast<double>(c.second.size()) / static_cast<double>(own[i].size());
      if (r >= 0.6 && r <= 1.7) break;
    }
    if (!chosen) chosen = &pool[pickd(rng)].second;
    if (i) out.push_back(' ');
    out += *chosen;
  }
  return out;
}

std::string ctl_shuffle_words(const std::string& text, std::mt19937_64& rng) {
  std::vector<std::string> sents;
  for (const std::string& s : split_sentences(text)) {
    std::vector<std::string> w = split_ws(s);
    if (w.empty()) continue;
    std::string tail;
    while (!w.back().empty() && std::string(".!?\"')").find(w.back().back()) != std::string::npos) {
      tail.insert(tail.begin(), w.back().back());
      w.back().pop_back();
    }
    if (w.back().empty()) w.pop_back();
    if (w.empty()) {
      sents.push_back(tail);
      continue;
    }
    std::shuffle(w.begin(), w.end(), rng);
    for (char& c : w[0]) {
      if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
      if (is_alpha_a(static_cast<unsigned char>(c))) break;
    }
    sents.push_back(join(w) + tail);
  }
  return join(sents);
}

template <class Fn>
std::string replace_cores(const std::string& text, Fn make_core) {
  std::vector<std::string> w = split_ws(text);
  for (std::string& t : w) {
    bool digit = false;
    for (char c : t) digit = digit || is_digit_a(static_cast<unsigned char>(c));
    if (digit) continue;
    std::size_t b = 0;
    while (b < t.size() && !is_alpha_a(static_cast<unsigned char>(t[b]))) ++b;
    if (b >= t.size()) continue;
    std::size_t e = b;
    while (e < t.size() && (is_alpha_a(static_cast<unsigned char>(t[e])) || t[e] == '\'' || t[e] == '-' || static_cast<unsigned char>(t[e]) >= 0x80))
      ++e;
    std::string core = make_core(t.substr(b, e - b));
    if (is_upper_a(static_cast<unsigned char>(t[b])) && !core.empty() && core[0] >= 'a' && core[0] <= 'z') core[0] = static_cast<char>(core[0] - 32);
    t = t.substr(0, b) + core + t.substr(e);
  }
  return join(w);
}

std::string ctl_random_lexicon(const std::string& text, const ScribeModel& m, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> U(0.0, 1.0);
  return replace_cores(text, [&](const std::string&) {
    for (int k = 0; k < 20; ++k) {
      const std::string& w = m.draw_word(U(rng));
      if (!w.empty() && w[0] != '\'') return w;
    }
    return std::string("the");
  });
}

std::string ctl_random_letters(const std::string& text, std::mt19937_64& rng) {
  return replace_cores(text, [&](const std::string& core) {
    std::string r(core.size(), 'a');
    for (char& c : r) c = static_cast<char>('a' + static_cast<int>(rng() % 26));
    return r;
  });
}

std::string ctl_repeat_phrase(const std::string& text) {
  std::vector<std::string> w = split_ws(text);
  if (w.size() > 6) w.resize(6);
  while (!w.empty()) {
    std::string& last = w.back();
    while (!last.empty() && !is_alpha_a(static_cast<unsigned char>(last.back())) && !is_digit_a(static_cast<unsigned char>(last.back())) &&
           static_cast<unsigned char>(last.back()) < 0x80)
      last.pop_back();
    if (last.empty()) w.pop_back();
    else break;
  }
  std::string phrase = join(w) + ".";
  std::string out;
  while (out.size() < text.size()) {
    if (!out.empty()) out.push_back(' ');
    out += phrase;
  }
  return out;
}

// ===========================================================================
// The pack (everything trained on the pack text) and the human reference
// ===========================================================================

struct Ctx {
  ScribeModel model;
  WordVectors wv;
  SentenceEmbedder emb;
  SurfaceRef surf;
  ScribeScorer scorer;   // v1
  ScribeV2Scorer v2;     // v2
  WordDict dict;
  bool use_vectors = false;
  std::string profile = "v2";
  std::size_t n_pack_docs = 0, pack_bytes = 0, n_ref_std = 0, n_ref_short = 0;
  std::vector<Features> ref_feats;  // reference samples, both buckets
  const SentenceEmbedder* embedder() const { return use_vectors ? &emb : nullptr; }
};

// --v2 "k=v,..." (V2Params::parse) must be set before the reference is built: the thresholds depend on it.
void set_v2_params(Ctx& c, const std::string& spec) {
  if (spec.empty()) return;
  if (!c.v2.params.parse(spec)) {
    std::fprintf(stderr, "bad --v2 parameters: %s\n", spec.c_str());
    std::exit(2);
  }
}

void build_pack(Ctx& c, const std::vector<std::string>& pack_docs, const std::string& vec_prefix) {
  for (const auto& d : pack_docs) c.pack_bytes += d.size();
  c.n_pack_docs = pack_docs.size();
  c.model.train(pack_docs);
  c.v2.train_pack(pack_docs);
  for (const auto& d : pack_docs) surface_add_vocab(c.surf, d);
  std::fprintf(stderr, "[%.0fs] pack: %zu docs, %zu bytes; vocab %zu, %zu trigrams\n", now_s() - g_t0, c.n_pack_docs, c.pack_bytes,
               c.model.vocab_size(), c.model.n_trigrams());
  if (vec_prefix.empty()) {
    std::fprintf(stderr, "no --vectors: C falls back to TF-IDF cosines (C0 absent)\n");
    return;
  }
  if (!c.wv.load(vec_prefix)) {
    std::fprintf(stderr, "cannot load word vectors %s.{vectors.f32,vocab.tsv} (300 dimensions)\n", vec_prefix.c_str());
    std::exit(1);
  }
  c.emb = SentenceEmbedder(&c.wv, 1e-3);
  std::vector<std::vector<float>> X;
  std::mt19937_64 prng(5);
  std::vector<std::string> toks;
  for (std::size_t d = 0; d < pack_docs.size() && X.size() < 60000; ++d) {
    for (const std::string& s : split_sentences(pack_docs[d])) {
      if (prng() % 8 != 0) continue;
      toks.clear();
      tokenize_words(s, toks);
      std::vector<float> v;
      if (toks.size() >= 3 && c.emb.raw(toks, v)) X.push_back(std::move(v));
    }
  }
  c.emb.fit_pc(X);
  c.use_vectors = true;
  std::fprintf(stderr, "[%.0fs] word vectors: %zu words x %zu; principal component from %zu pack sentences\n", now_s() - g_t0, c.wv.size(), c.wv.dim(),
               X.size());
}

Features featurize(Ctx& c, const std::string& text, const std::string& prompt = "") {
  Features f = extract_features(c.model, c.dict, text, c.embedder());
  c.v2.extend(f, text, prompt, c.embedder());
  return f;
}

// Reference samples from ref_docs with rng (v1's draw order: standard first, then short).
void build_reference(Ctx& c, const std::vector<std::string>& ref_docs, std::mt19937_64& rng) {
  const std::vector<Sample> ref_std = make_samples(ref_docs, 400, 600, 3, 4, rng);
  const std::vector<Sample> ref_short = make_samples(ref_docs, 60, 260, 1, 3, rng);
  c.n_ref_std = ref_std.size();
  c.n_ref_short = ref_short.size();
  c.ref_feats.reserve(ref_std.size() + ref_short.size());
  for (const auto* v : {&ref_std, &ref_short})
    for (const Sample& s : *v) c.ref_feats.push_back(featurize(c, s.text));
  for (const Features& f : c.ref_feats) {
    c.scorer.add_reference(f);
    c.v2.add_reference(f);
  }
  c.scorer.finish();
  c.v2.finish();
  double sd[2] = {0, 0}, sl[2] = {0, 0}, sn[2] = {0, 0};
  for (std::size_t i = 0; i < c.ref_feats.size(); ++i) {
    const std::string& t = i < ref_std.size() ? ref_std[i].text : ref_short[i - ref_std.size()].text;
    const int b = length_bucket(t.size());
    sd[b] += scribe_detail::distinct4_bytes(t);
    sl[b] += c.ref_feats[i].loop8;
    sn[b] += 1;
  }
  for (int b = 0; b < 2; ++b) {
    c.surf.ref_d4[b] = sn[b] ? sd[b] / sn[b] : 0.9;
    c.surf.ref_loop[b] = sn[b] ? sl[b] / sn[b] : 0;
  }
  std::fprintf(stderr, "[%.0fs] reference: %zu standard + %zu short samples\n", now_s() - g_t0, c.n_ref_std, c.n_ref_short);
}

// The per-sample result in both profiles.
struct Result {
  Scored v1;
  ScoredV2 v2;
  double surface = 0;
  double headline(const std::string& profile) const { return profile == "v1" ? v1.scribe : v2.scribe; }
};

Result score_one(const Ctx& c, const Features& f, const std::string& text) {
  Result r;
  r.v1 = c.scorer.score(f);
  r.v2 = c.v2.score(f);
  r.surface = surface_S(c.surf, text);
  return r;
}

// The JSON members of one scored sample: headline, both profiles, raw statistics.
std::string result_json(const Features& f, const Result& r, const std::string& profile) {
  std::string o = "\"scribe\":" + num(r.headline(profile)) + ",\"scribe_v1\":" + num(r.v1.scribe) + ",\"scribe_v2\":" + num(r.v2.scribe) +
                  ",\"surface_s\":" + num(r.surface);
  o += ",\"v1\":{\"S\":" + num(r.v1.S) + ",\"C\":" + num(f.has_c ? r.v1.C : NAN) + ",\"I\":" + num(r.v1.I) + ",\"s_gate\":" + (r.v1.s_gate ? "1" : "0") +
       ",\"i_gate\":" + (r.v1.i_gate ? "1" : "0") + "}";
  o += ",\"v2\":" + v2_json(r.v2);
  o += ",\"raw\":" + features_json(f);
  return o;
}

// ===========================================================================
// docs / prompts / samples / prepare
// ===========================================================================

struct Args {
  std::map<std::string, std::string> kv;
  std::vector<std::string> multi_model;
  std::vector<std::string> pos;
  std::map<std::string, bool> flag;
  std::string get(const std::string& k, const std::string& d = "") const {
    const auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  bool has(const std::string& k) const { return flag.count(k) > 0; }
};

Args parse_args(int argc, char** argv, int from) {
  static const std::vector<std::string> flags = {"--short", "--chain-rng", "--quiet"};
  Args a;
  for (int i = from; i < argc; ++i) {
    const std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      if (std::find(flags.begin(), flags.end(), s) != flags.end()) {
        a.flag[s] = true;
      } else if (i + 1 < argc) {
        if (s == "--model") a.multi_model.push_back(argv[++i]);
        else a.kv[s] = argv[++i];
      } else {
        std::fprintf(stderr, "missing value for %s\n", s.c_str());
        std::exit(2);
      }
    } else {
      a.pos.push_back(s);
    }
  }
  return a;
}

std::ofstream open_out(const std::string& path) {
  if (path.empty()) {
    std::fprintf(stderr, "need --out\n");
    std::exit(2);
  }
  std::ofstream o(path);
  if (!o) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(1);
  }
  return o;
}

int cmd_docs(const Args& a) {
  if (a.pos.empty()) return 2;
  const std::vector<std::string> docs = load_docs(a.pos[0]);
  std::ofstream o = open_out(a.get("--out"));
  const std::string set = a.get("--set", "docs");
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < docs.size(); ++i) {
    char id[64];
    std::snprintf(id, sizeof id, "%s/%06zu", set.c_str(), i);
    o << text_line(id, set, "human", docs[i]) << "\n";
    bytes += docs[i].size();
  }
  std::fprintf(stderr, "%zu documents, %zu bytes\n", docs.size(), bytes);
  return 0;
}

// One prompt per document: a run of whole sentences of 100-300 bytes from a random sentence of the document, then the
// true continuation: the next whole sentences, 400-600 bytes (target drawn uniformly), at least 3 sentences.
int cmd_prompts(const Args& a) {
  if (a.pos.empty()) return 2;
  const std::vector<std::string> docs = load_docs(a.pos[0]);
  const std::size_t n = std::strtoull(a.get("--n", "250").c_str(), nullptr, 10);
  std::mt19937_64 rng(std::strtoull(a.get("--seed", "1").c_str(), nullptr, 10));
  const std::string set = a.get("--set", "prompts");
  std::vector<std::size_t> order(docs.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::shuffle(order.begin(), order.end(), rng);
  std::ofstream o = open_out(a.get("--out"));
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::size_t made = 0;
  for (std::size_t d : order) {
    if (made >= n) break;
    const std::vector<std::string> s = split_sentences(docs[d]);
    if (s.size() < 5) continue;
    bool done = false;
    for (int tries = 0; tries < 6 && !done; ++tries) {
      const std::size_t i0 = static_cast<std::size_t>(U(rng) * static_cast<double>(s.size() - 4));
      std::size_t j = i0, pb = 0;
      while (j < s.size() && pb < 100) pb += s[j++].size() + (pb ? 1 : 0);
      if (pb < 100 || pb > 300) continue;
      const double target = 400.0 + U(rng) * 200.0;
      std::size_t k = j, cb = 0;
      while (k < s.size() && static_cast<double>(cb) < target) cb += s[k++].size() + (cb ? 1 : 0);
      if (cb < 400 || cb > 600 || k - j < 3) continue;
      std::vector<std::string> ps(s.begin() + static_cast<std::ptrdiff_t>(i0), s.begin() + static_cast<std::ptrdiff_t>(j));
      std::vector<std::string> cs(s.begin() + static_cast<std::ptrdiff_t>(j), s.begin() + static_cast<std::ptrdiff_t>(k));
      const std::string ref = join(cs);
      char id[64];
      std::snprintf(id, sizeof id, "%s/%04zu", set.c_str(), made);
      o << text_line(id, set, "human", ref, join(ps), ref, "\"meta\":{\"doc\":" + std::to_string(d) + "}") << "\n";
      ++made;
      done = true;
    }
  }
  std::fprintf(stderr, "%zu prompts from %zu documents\n", made, docs.size());
  return made ? 0 : 1;
}

int cmd_samples(const Args& a) {
  if (a.pos.empty()) return 2;
  const std::vector<std::string> docs = load_docs(a.pos[0]);
  const std::size_t n = std::strtoull(a.get("--n", "500").c_str(), nullptr, 10);
  std::mt19937_64 rng(std::strtoull(a.get("--seed", "1").c_str(), nullptr, 10));
  const bool shrt = a.has("--short");
  std::vector<Sample> all = shrt ? make_samples(docs, 60, 260, 1, 3, rng) : make_samples(docs, 400, 600, 3, 4, rng);
  const std::size_t cand = all.size();
  const std::vector<Sample> got = pick(std::move(all), n, rng);
  std::ofstream o = open_out(a.get("--out"));
  const std::string set = a.get("--set", "samples"), system = a.get("--system", "human");
  for (std::size_t i = 0; i < got.size(); ++i) {
    char id[96];
    std::snprintf(id, sizeof id, "%s/%s/%04zu", set.c_str(), system.c_str(), i);
    o << text_line(id, set, system, got[i].text, "", "", "\"meta\":{\"doc\":" + std::to_string(got[i].doc) + "}") << "\n";
  }
  std::fprintf(stderr, "%zu samples (of %zu candidates) from %zu documents\n", got.size(), cand, docs.size());
  return 0;
}

int cmd_prepare(const Args& a) {
  const std::string mode = a.get("--mode", "continuation");
  const std::size_t min_bytes = std::strtoull(a.get("--min-bytes", "100").c_str(), nullptr, 10);
  std::ofstream o = open_out(a.get("--out"));
  std::size_t n = 0, kept = 0, raw_b = 0, kept_b = 0;
  for (const JsonObj& j : read_jsonl(a.get("--in"))) {
    ++n;
    const std::string raw = j.get("text");
    raw_b += raw.size();
    std::string t;
    if (mode == "wikiraw") t = clean_generated_text(raw);
    else if (mode == "plain") t = scribe_detail::collapse_ws(collapse_newlines(raw));
    else t = prepare_continuation(raw);
    if (t.size() < min_bytes) continue;
    ++kept;
    kept_b += t.size();
    o << text_line(j.get("id"), j.get("set"), j.get("system"), t, j.get("prompt"), j.get("ref")) << "\n";
  }
  std::fprintf(stderr, "prepare (%s): %zu texts, %zu kept (>= %zu B); mean bytes raw %.0f, kept %.0f\n", mode.c_str(), n, kept, min_bytes,
               n ? double(raw_b) / double(n) : 0.0, kept ? double(kept_b) / double(kept) : 0.0);
  return 0;
}

// ===========================================================================
// ladder
// ===========================================================================

struct Rung {
  std::string key, name;
  std::vector<std::string> texts, ids, kinds;
  std::vector<Features> feats;
  std::vector<Result> res;
  std::vector<double> scribe, surf;
  std::pair<double, double> ci_scribe{0, 0}, ci_surf{0, 0};
};

std::string fmt(double v, int prec = 3) {
  char b[64];
  std::snprintf(b, sizeof b, "%.*f", prec, v);
  return b;
}
std::string fmt_ci(double m, std::pair<double, double> ci, int prec = 3) {
  return fmt(m, prec) + " [" + fmt(ci.first, prec) + ", " + fmt(ci.second, prec) + "]";
}

struct GenSet {
  std::vector<std::string> texts, kinds;
  std::size_t lines = 0, dropped = 0;
};

// GEN lines of a cyphalm_trie sample file -> clean_generated_text; under 100 bytes dropped (v1's rule).
GenSet load_gen_trie(const std::string& path) {
  GenSet g;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.compare(0, 4, "GEN\t") != 0) continue;
    const std::size_t t2 = line.find('\t', 4);
    if (t2 == std::string::npos) continue;
    ++g.lines;
    const std::string text = clean_generated_text(unescape_gen(line.substr(t2 + 1)));
    if (text.size() < 100) {
      ++g.dropped;
      continue;
    }
    g.texts.push_back(text);
    g.kinds.push_back(line.substr(4, t2 - 4));
  }
  return g;
}

int cmd_ladder(const Args& a) {
  Ctx c;
  c.profile = a.get("--profile", "v2");
  const std::string label = a.get("--label", "ladder");
  const std::uint64_t ref_seed = std::strtoull(a.get("--ref-seed", "20261027").c_str(), nullptr, 10);
  const std::uint64_t seed = std::strtoull(a.get("--seed", "31415926").c_str(), nullptr, 10);
  const std::vector<std::string> pack_docs = load_docs(a.get("--pack"));
  const std::vector<std::string> ref_docs = load_docs(a.get("--ref"));
  const std::vector<std::string> test_docs = load_docs(a.get("--test"));
  std::fprintf(stderr, "[%.0fs] docs: pack %zu, ref %zu, test %zu\n", now_s() - g_t0, pack_docs.size(), ref_docs.size(), test_docs.size());
  set_v2_params(c, a.get("--v2"));
  build_pack(c, pack_docs, a.get("--vectors"));
  std::mt19937_64 rng(ref_seed);
  build_reference(c, ref_docs, rng);

  // ---- samples: --chain-rng continues the reference draws (v1's development ladder), else a fresh generator.
  std::mt19937_64 trng(seed);
  std::mt19937_64& srng = a.has("--chain-rng") ? rng : trng;
  std::vector<Sample> test_std_all = make_samples(test_docs, 400, 600, 3, 4, srng);
  const std::size_t n_cand = test_std_all.size();
  const std::size_t n_take = std::strtoull(a.get("--n", "500").c_str(), nullptr, 10);
  const std::vector<Sample> test_std = pick(test_std_all, n_take, srng);
  const std::vector<Sample> test_short = pick(make_samples(test_docs, 60, 260, 1, 3, srng), n_take, srng);
  if (test_std.size() < 100) {
    std::fprintf(stderr, "too few test samples (%zu)\n", test_std.size());
    return 1;
  }
  std::mt19937_64 crng(seed + 1);
  std::vector<std::pair<int, std::string>> sent_pool;
  for (const Sample& s : test_std)
    for (const std::string& x : split_sentences(s.text)) sent_pool.emplace_back(s.doc, x);

  // Slot order is v1's (it fixes the bootstrap seeds); model rungs from --model files follow.
  enum { R_HUMAN, R_TRIE, R_SS, R_WS, R_RL, R_RLET, R_REP, R_T06, R_T08, R_T10, R_SHORT, R_N };
  std::vector<Rung> rungs(R_N);
  const char* keys[R_N] = {"human", "trie", "sent_shuffled", "word_shuffled", "random_lexicon", "random_letters", "repeated_phrase",
                           "trie_T0.6", "trie_T0.8", "trie_T1.0", "human_short"};
  const char* names[R_N] = {"held-out human", "trie samples (pooled)", "sentences shuffled across docs", "words shuffled within sentences",
                            "random lexicon words", "random letters", "one phrase repeated", "trie samples T=0.6", "trie samples T=0.8",
                            "trie samples T=1.0", "held-out human, short (60-260 B)"};
  for (int i = 0; i < R_N; ++i) {
    rungs[static_cast<std::size_t>(i)].key = keys[i];
    rungs[static_cast<std::size_t>(i)].name = names[i];
  }
  for (const Sample& s : test_std) {
    rungs[R_HUMAN].texts.push_back(s.text);
    rungs[R_SS].texts.push_back(ctl_shuffle_sentences(s, sent_pool, crng));
    rungs[R_WS].texts.push_back(ctl_shuffle_words(s.text, crng));
    rungs[R_RL].texts.push_back(ctl_random_lexicon(s.text, c.model, crng));
    rungs[R_RLET].texts.push_back(ctl_random_letters(s.text, crng));
    rungs[R_REP].texts.push_back(ctl_repeat_phrase(s.text));
  }
  for (const Sample& s : test_short) rungs[R_SHORT].texts.push_back(s.text);
  GenSet gen;
  if (!a.get("--gen-trie").empty()) gen = load_gen_trie(a.get("--gen-trie"));
  for (std::size_t i = 0; i < gen.texts.size(); ++i) {
    rungs[R_TRIE].texts.push_back(gen.texts[i]);
    rungs[R_TRIE].kinds.push_back(gen.kinds[i]);
    const int slot = gen.kinds[i] == "trie_T0.6" ? R_T06 : (gen.kinds[i] == "trie_T0.8" ? R_T08 : (gen.kinds[i] == "trie_T1.0" ? R_T10 : -1));
    if (slot >= 0) rungs[static_cast<std::size_t>(slot)].texts.push_back(gen.texts[i]);
  }
  // Model rungs: prepared JSONL files, one rung per "system".
  std::vector<std::size_t> model_rungs;
  for (const std::string& mf : a.multi_model) {
    std::map<std::string, std::size_t> slot;
    for (const JsonObj& j : read_jsonl(mf)) {
      const std::string sys = j.get("system", "model");
      auto it = slot.find(sys);
      if (it == slot.end()) {
        Rung r;
        r.key = sys;
        r.name = sys;
        rungs.push_back(std::move(r));
        it = slot.emplace(sys, rungs.size() - 1).first;
        model_rungs.push_back(it->second);
      }
      rungs[it->second].texts.push_back(j.get("text"));
      rungs[it->second].ids.push_back(j.get("id"));
    }
  }
  for (Rung& r : rungs) {
    if (r.ids.empty())
      for (std::size_t i = 0; i < r.texts.size(); ++i) {
        char id[128];
        std::snprintf(id, sizeof id, "%s/%s/%04zu", label.c_str(), r.key.c_str(), i);
        r.ids.push_back(id);
      }
    r.feats.reserve(r.texts.size());
    for (const std::string& t : r.texts) r.feats.push_back(featurize(c, t));
  }
  std::fprintf(stderr, "[%.0fs] %s: features extracted\n", now_s() - g_t0, label.c_str());

  std::uint64_t bseed = 100 + seed % 1000;
  for (Rung& r : rungs) {
    for (std::size_t i = 0; i < r.feats.size(); ++i) {
      r.res.push_back(score_one(c, r.feats[i], r.texts[i]));
      r.scribe.push_back(r.res.back().headline(c.profile));
      r.surf.push_back(r.res.back().surface);
    }
    r.ci_scribe = bootstrap_mean_ci(r.scribe, 1000, ++bseed);
    bseed += 2;  // v1 drew two more intervals per rung here
    r.ci_surf = bootstrap_mean_ci(r.surf, 1000, ++bseed);
  }

  // ---- report
  std::printf("# SCRIBE %s ladder: %s (profile %s)\n\n", label.c_str(), a.get("--test").c_str(), c.profile.c_str());
  std::printf("Pack %s: %zu docs, %zu bytes, lexicon %zu words, %zu trigrams; vectors: %s. Reference %s: %zu standard + %zu short samples (seed %llu). "
              "Test: %zu standard samples (of %zu candidates), %zu short (seed %llu%s).\n\n",
              a.get("--pack").c_str(), c.n_pack_docs, c.pack_bytes, c.model.vocab_size(), c.model.n_trigrams(),
              c.use_vectors ? a.get("--vectors").c_str() : "none", a.get("--ref").c_str(), c.n_ref_std, c.n_ref_short,
              static_cast<unsigned long long>(ref_seed), test_std.size(), n_cand, test_short.size(), static_cast<unsigned long long>(seed),
              a.has("--chain-rng") ? ", reference generator continued" : "");
  if (gen.lines)
    std::printf("Trie rung: %zu GEN lines, %zu kept after cleaning (%zu under 100 B dropped).\n\n", gen.lines, gen.texts.size(), gen.dropped);
  const double human_mean = mean_of(rungs[R_HUMAN].scribe);
  std::printf("## Rung means (95%% bootstrap interval of the mean, 1000 resamples)\n\n| rung | n | SCRIBE | parity | surface S | mean bytes |\n|---|---|---|---|---|---|\n");
  auto row = [&](const Rung& r) {
    if (r.feats.empty()) return;
    double by = 0;
    for (const Features& f : r.feats) by += static_cast<double>(f.bytes);
    std::printf("| %s | %zu | %s | %.2f | %s | %.0f |\n", r.name.c_str(), r.feats.size(), fmt_ci(mean_of(r.scribe), r.ci_scribe).c_str(),
                mean_of(r.scribe) / human_mean, fmt_ci(mean_of(r.surf), r.ci_surf).c_str(), by / static_cast<double>(r.feats.size()));
  };
  for (std::size_t i = 0; i < rungs.size(); ++i) row(rungs[i]);

  struct Chain {
    std::string name;
    std::vector<std::size_t> idx;
  };
  std::vector<Chain> chains;
  if (!rungs[R_TRIE].feats.empty()) {
    chains.push_back({"trie pooled", {R_HUMAN, R_TRIE, R_SS, R_WS, R_RL, R_RLET, R_REP}});
    for (int t : {R_T06, R_T08, R_T10})
      if (!rungs[static_cast<std::size_t>(t)].feats.empty())
        chains.push_back({rungs[static_cast<std::size_t>(t)].name, {R_HUMAN, static_cast<std::size_t>(t), R_SS, R_WS, R_RL, R_RLET, R_REP}});
  }
  chains.push_back({"control ladder (no model rung)", {R_HUMAN, R_SS, R_WS, R_RL, R_RLET, R_REP}});
  for (std::size_t m : model_rungs) chains.push_back({"model " + rungs[m].name, {R_HUMAN, m, R_SS, R_WS, R_RL, R_RLET, R_REP}});
  std::printf("\n## Ladders: pairwise ordering accuracy over all (upper, lower) pairs (ties = 1/2) and disjoint 95%% intervals; PASS needs both (accuracy >= 0.95)\n");
  for (const Chain& ch : chains) {
    std::printf("\n### %s\n\n| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |\n|---|---|---|---|---|---|---|\n", ch.name.c_str());
    int ps = 0, pu = 0, np = 0;
    for (std::size_t k = 0; k + 1 < ch.idx.size(); ++k) {
      const Rung &x = rungs[ch.idx[k]], &y = rungs[ch.idx[k + 1]];
      const double a1 = pairwise_accuracy(x.scribe, y.scribe), a2 = pairwise_accuracy(x.surf, y.surf);
      const bool d1 = x.ci_scribe.first > y.ci_scribe.second, d2 = x.ci_surf.first > y.ci_surf.second;
      const bool p1 = a1 >= 0.95 && d1, p2 = a2 >= 0.95 && d2;
      ps += p1;
      pu += p2;
      ++np;
      std::printf("| %s > %s | %.3f | %s | %s | %.3f | %s | %s |\n", x.name.c_str(), y.name.c_str(), a1, d1 ? "yes" : "no", p1 ? "PASS" : "FAIL", a2,
                  d2 ? "yes" : "no", p2 ? "PASS" : "FAIL");
    }
    std::printf("\nVerdict (%s): SCRIBE %s (%d of %d pairs), surface S %s (%d of %d pairs)\n", ch.name.c_str(), ps == np ? "PASS" : "FAIL", ps, np,
                pu == np ? "PASS" : "FAIL", pu, np);
  }
  // Axis profile of the headline profile.
  std::printf("\n## Axis profile (mean over samples)\n\n");
  if (c.profile == "v1") {
    std::printf("| rung | S | C (n) | I | S gate | I gate |\n|---|---|---|---|---|---|\n");
    for (const Rung& r : rungs) {
      if (r.feats.empty()) continue;
      double S = 0, C = 0, I = 0, g1 = 0, g2 = 0;
      int nc = 0;
      for (const Result& x : r.res) {
        S += x.v1.S;
        I += x.v1.I;
        g1 += x.v1.s_gate;
        g2 += x.v1.i_gate;
        if (x.v1.has_c) {
          C += x.v1.C;
          ++nc;
        }
      }
      const double n = static_cast<double>(r.res.size());
      std::printf("| %s | %.2f | %.2f (%d) | %.2f | %.0f%% | %.0f%% |\n", r.name.c_str(), S / n, nc ? C / nc : 0.0, nc, I / n, 100 * g1 / n, 100 * g2 / n);
    }
  } else {
    std::printf("%s", v2_profile_header().c_str());
    for (const Rung& r : rungs) {
      if (r.feats.empty()) continue;
      std::vector<ScoredV2> v;
      for (const Result& x : r.res) v.push_back(x.v2);
      std::printf("%s", v2_profile_row(r.name, v).c_str());
    }
  }
  // V4: correlation with length on human text.
  {
    auto by = [](const Rung& r) {
      std::vector<double> b;
      for (const Features& f : r.feats) b.push_back(static_cast<double>(f.bytes));
      return b;
    };
    std::vector<double> sc_all = rungs[R_HUMAN].scribe, by_all = by(rungs[R_HUMAN]);
    sc_all.insert(sc_all.end(), rungs[R_SHORT].scribe.begin(), rungs[R_SHORT].scribe.end());
    const std::vector<double> bs = by(rungs[R_SHORT]);
    by_all.insert(by_all.end(), bs.begin(), bs.end());
    std::printf("\n## V4: Pearson (Spearman) of SCRIBE with bytes on held-out human text\n\nstandard %.3f (%.3f); short %.3f (%.3f); pooled %.3f (%.3f). "
                "Mean SCRIBE standard %.3f, short %.3f.\n",
                pearson(rungs[R_HUMAN].scribe, by(rungs[R_HUMAN])), spearman(rungs[R_HUMAN].scribe, by(rungs[R_HUMAN])),
                pearson(rungs[R_SHORT].scribe, bs), spearman(rungs[R_SHORT].scribe, bs), pearson(sc_all, by_all), spearman(sc_all, by_all),
                mean_of(rungs[R_HUMAN].scribe), mean_of(rungs[R_SHORT].scribe));
  }
  // ---- export: every rung's texts and per-sample scores, and the reference statistics.
  const std::string dir = a.get("--export");
  if (!dir.empty()) {
    std::filesystem::create_directories(dir);
    std::ofstream ot(dir + "/texts.jsonl"), os(dir + "/scores.jsonl"), orf(dir + "/reference.jsonl");
    for (const Rung& r : rungs)
      for (std::size_t i = 0; i < r.texts.size(); ++i) {
        ot << text_line(r.ids[i], label, r.key, r.texts[i], "", "", r.kinds.empty() ? "" : "\"meta\":{\"kind\":\"" + json_escape(r.kinds[i]) + "\"}") << "\n";
        os << "{\"id\":\"" << json_escape(r.ids[i]) << "\",\"system\":\"" << json_escape(r.key) << "\"," << result_json(r.feats[i], r.res[i], c.profile)
           << "}\n";
      }
    for (std::size_t i = 0; i < c.ref_feats.size(); ++i)
      orf << "{\"id\":\"ref/" << i << "\",\"raw\":" << features_json(c.ref_feats[i]) << ",\"v2\":" << v2_json(c.v2.score(c.ref_feats[i])) << "}\n";
    std::fprintf(stderr, "exported to %s\n", dir.c_str());
  }
  std::fprintf(stderr, "[%.0fs] done\n", now_s() - g_t0);
  return 0;
}

// ===========================================================================
// score
// ===========================================================================

int cmd_score(const Args& a) {
  Ctx c;
  c.profile = a.get("--profile", "v2");
  const std::vector<std::string> pack_docs = load_docs(a.get("--pack"));
  const std::vector<std::string> ref_docs = load_docs(a.get("--ref"));
  set_v2_params(c, a.get("--v2"));
  build_pack(c, pack_docs, a.get("--vectors"));
  std::mt19937_64 rng(std::strtoull(a.get("--ref-seed", "20261027").c_str(), nullptr, 10));
  build_reference(c, ref_docs, rng);
  const std::vector<JsonObj> in = read_jsonl(a.get("--in"));
  std::ofstream o = open_out(a.get("--out"));
  const double t1 = now_s();
  for (const JsonObj& j : in) {
    const std::string text = j.get("text");
    const Features f = featurize(c, text, j.get("prompt"));
    const Result r = score_one(c, f, text);
    o << "{\"id\":\"" << json_escape(j.get("id")) << "\",\"system\":\"" << json_escape(j.get("system")) << "\"," << result_json(f, r, c.profile) << "}\n";
  }
  const double dt = now_s() - t1;
  std::fprintf(stderr, "[%.0fs] scored %zu texts in %.2f s (%.0f texts/s)\n", now_s() - g_t0, in.size(), dt, dt > 0 ? double(in.size()) / dt : 0.0);
  return 0;
}

// ===========================================================================
// selftest
// ===========================================================================

int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    ++g_checks;                                                   \
    if (!(cond)) {                                                \
      ++g_fail;                                                   \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    }                                                             \
  } while (0)
#define CHECK_NEAR(x, y, tol)                                                                  \
  do {                                                                                         \
    ++g_checks;                                                                                \
    const double _x = (x), _y = (y);                                                           \
    if (!(std::fabs(_x - _y) <= (tol))) {                                                      \
      ++g_fail;                                                                                \
      std::printf("FAIL %s:%d  %s = %.10g, expected %.10g\n", __FILE__, __LINE__, #x, _x, _y); \
    }                                                                                          \
  } while (0)

int selftest() {
  // ---- v0 and v1 checks (unchanged) -----------------------------------------------------
  const std::string raw =
      "'''Rome''' is the [[capital city|capital]] of [[Italy]] and a major European city.&lt;ref&gt;Some ref&lt;/ref&gt; "
      "{{cite web|url=x}} It has [[Image:Foo.jpg|thumb|A [[nested]] caption]] many old buildings and wide squares.\n"
      "== History ==\n* a list item that is long enough to be a line but starts with a star sign okay\n"
      "{| class=\"wikitable\"\n| cell\n|}\n"
      "Another paragraph about the city of Rome with enough words to count as prose here.&amp;nbsp;";
  CHECK(clean_wiki_body(raw) ==
        "Rome is the capital of Italy and a major European city. It has many old buildings and wide squares. "
        "Another paragraph about the city of Rome with enough words to count as prose here.");
  const std::string page = "<page><title>X</title><revision><text xml:space=\"preserve\">" + raw + "</text></revision></page>"
                           "<page><title>R</title><revision><text>#REDIRECT [[X]]</text></revision></page>"
                           "<page><title>cut</title><revision><text>Unfinished";
  CHECK(clean_wiki_documents(page.data(), page.size()).size() == 1);
  const std::vector<std::string> ss = split_sentences("Mr. Smith went to Washington. He liked it! Did he? J. R. Tolkien wrote books.");
  CHECK(ss.size() == 4 && ss[0] == "Mr. Smith went to Washington." && ss[3] == "J. R. Tolkien wrote books.");
  CHECK(split_sentences("It ended. then lower case began.").size() == 2);
  std::vector<std::string> tk;
  tokenize_words("Rose's 1,000 don't U.S. cars-and-trucks", tk);
  CHECK((tk == std::vector<std::string>{"rose", "'s", "1,000", "don't", "u", "s", "cars", "and", "trucks"}));
  CHECK(numeric_token("1,000") && numeric_token("1980s") && !numeric_token("abc") && !numeric_token("3d"));
  CHECK_NEAR(jsd_bits({1, 0}, {0, 1}), 1.0, 1e-12);
  CHECK_NEAR(jsd_bits({0.5, 0.5}, {0.5, 0.5}), 0.0, 1e-12);
  CHECK_NEAR(jsd_bits({0.5, 0.5}, {1, 0}), 0.311278, 1e-5);
  Ecdf e;
  for (int i = 1; i <= 100; ++i) e.add(i);
  e.finish();
  CHECK_NEAR(e.F(50.5), 0.5, 1e-12);
  CHECK_NEAR(e.one_sided(50.5), 1.0, 1e-12);
  CHECK_NEAR(e.two_sided(0.0), 1.0 / 101.0, 1e-12);
  CHECK_NEAR(e.one_sided(1000.0), 1.0, 1e-12);
  CHECK_NEAR(pairwise_accuracy({3}, {1, 2, 3}), 2.5 / 3.0, 1e-12);
  CHECK_NEAR(pearson({1, 2, 3, 4}, {2, 4, 6, 8}), 1.0, 1e-12);
  CHECK_NEAR(spearman({1, 2, 3, 4}, {1, 8, 27, 64}), 1.0, 1e-12);
  CHECK_NEAR(scribe_detail::loop8_bytes("abababababababababab"), 11.0 / 13.0, 1e-12);
  CHECK(scribe_detail::utf8_valid("caf\xC3\xA9") && !scribe_detail::utf8_valid("caf\xC3") && !scribe_detail::utf8_valid("\xC0\x80"));
  ScribeModel m;
  m.train({"The cat sat on the mat. The dog sat on the log. A cat saw a dog.", "The dog saw the cat. The mat was red. A log was on the mat."}, 1);
  const std::uint32_t the = m.word_id("the"), cat = m.word_id("cat"), dog = m.word_id("dog");
  CHECK(the != ScribeModel::kUnk && cat != ScribeModel::kUnk && m.word_id("zebra") == ScribeModel::kUnk);
  const std::uint32_t ctxs[][2] = {{ScribeModel::kBos, ScribeModel::kBos}, {the, cat}, {cat, dog}, {the, the}, {dog, ScribeModel::kUnk}};
  for (const auto& cx : ctxs) {
    double s = 0;
    for (std::uint32_t w = 0; w < m.vocab_size(); ++w) s += std::exp(m.logp(cx[0], cx[1], w));
    CHECK_NEAR(s, 1.0, 1e-9);
  }
  CHECK(m.logp(the, cat, m.word_id("sat")) > m.logp(the, cat, m.word_id("red")));
  WordDict dict;
  const Features ok = extract_features(m, dict, "The cat sat on the mat. The dog saw the cat.");
  const Features bad = extract_features(m, dict, "the cat (sat on");
  CHECK_NEAR(ok.s1, 1.0, 1e-12);
  CHECK_NEAR(ok.s4, 1.0, 1e-12);
  CHECK(ok.has_c && ok.n_sent == 2 && ok.n_words == 11);
  CHECK_NEAR(bad.s4, 2.0 / 6.0, 1e-12);
  CHECK_NEAR(ok.d[0], 7.0 / 11.0, 1e-12);
  const Features rep = extract_features(m, dict, "The cat sat. The cat sat. The cat sat.");
  CHECK_NEAR(rep.rep, 2.0 / 3.0, 1e-12);
  SurfaceRef sr;
  surface_add_vocab(sr, "the quick brown fox jumps over the lazy dog and the cat");
  CHECK(surface_S(sr, "the quick brown fox jumps over the lazy dog and the cat") > surface_S(sr, "zzz qqq xxx zzz qqq xxx zzz qqq xxx zzz qqq xxx"));
  {
    const std::vector<std::vector<float>> U = {{1, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    CHECK_NEAR(scribe_detail::mean_pairwise_cosine(U), 1.0 / 3.0, 1e-9);
    CHECK_NEAR(scribe_detail::mean_pairwise_cosine({{1, 0}, {0, 1}}), 0.0, 1e-9);
  }
  {
    WordVectors wv;
    wv.set({"aa", "bb", "cc", "dd"}, {1000, 100, 10, 1}, {1, 0.2f, 0, 1, 0.1f, 0, 0.9f, 0.3f, 0, 0, 1, 0.2f}, 3);
    for (int i = 0; i < 4; ++i) {
      double n2 = 0;
      for (int k = 0; k < 3; ++k) n2 += double(wv.row(i)[k]) * wv.row(i)[k];
      CHECK_NEAR(n2, 1.0, 1e-5);
    }
    CHECK(wv.index("cc") == 2 && wv.index("zz") == -1 && wv.dim() == 3 && wv.size() == 4);
    std::stringstream vs(std::ios::in | std::ios::out | std::ios::binary), ts;
    const float rw[6] = {1, 0, 0, 0, 1, 0};
    vs.write(reinterpret_cast<const char*>(rw), sizeof rw);
    ts << "the\t100\nof\t50\n";
    WordVectors w2;
    CHECK(w2.load_streams(vs, ts, 3) && w2.size() == 2 && w2.index("of") == 1);
    SentenceEmbedder se(&wv);
    std::vector<float> va, vb;
    CHECK(se.raw({"aa", "dd"}, va) && !se.raw({"aa", "zz"}, vb, 2) && se.raw({"aa", "zz"}, vb, 1));
  }
  {
    Features f;
    f.sent_len = {5, 5, 5, 5};
    std::vector<double> pmf(kMaxLen + 1, 0.0);
    pmf[5] = 0.5;
    pmf[10] = 0.5;
    CHECK_NEAR(ks_lengths({&f}, pmf), 0.5, 1e-12);
  }
  {
    const std::string g = clean_generated_text(
        " of the [[city]] was large. It has ''many'' old [[Rome|buildings]] and wide squares.\n\nAnother sentence here is fine. The next one is cu");
    CHECK(g == "It has many old buildings and wide squares. Another sentence here is fine.");
    CHECK(clean_generated_text("no terminator at all here") == "terminator at all");
  }
  // ---- tool-level checks ----------------------------------------------------------------
  {
    const std::string line = "{\"id\":\"a\\\"b\",\"n\":3.5,\"meta\":{\"x\":[1,{\"y\":\"}\"}]},\"text\":\"caf\\u00e9 \\ud83d\\ude00\\nz\",\"t\":true}";
    JsonObj o;
    CHECK(JsonReader(line).object(o));
    CHECK(o.get("id") == "a\"b" && o.num.count("n") && o.num.at("n") == 3.5);
    CHECK(o.get("text") == "caf\xC3\xA9 \xF0\x9F\x98\x80\nz");
    JsonObj o2;
    CHECK(JsonReader("{\"text\":\"" + json_escape(std::string("q\"\\\n\x01 \xC3")) + "\"}").object(o2));
    CHECK(o2.get("text") == std::string("q\"\\\n\x01 \xEF\xBF\xBD"));  // the invalid byte became U+FFFD
    JsonObj o3;
    CHECK(!JsonReader("{\"a\":\"unterminated}").object(o3));
  }
  {
    CHECK(prepare_continuation("  It was a dark night.\nThe rain fell on the  town. And then the") == "It was a dark night. The rain fell on the town.");
    CHECK(prepare_continuation("no boundary at all") == "no boundary at all");
    CHECK(prepare_continuation("He said \"stop.\" Then he left") == "He said \"stop.\"");
  }
  {
    std::mt19937_64 r(1);
    const std::string t = "The cat sat on the mat. The dog saw the cat. A red mat was on the log.";
    CHECK(split_sentences(ctl_shuffle_words(t, r)).size() == 3);
    CHECK(ctl_repeat_phrase(t).size() >= t.size());
  }
  scribe_v2_selftest(g_checks, g_fail);
  std::printf("scribe selftest: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: scribe <docs|prompts|samples|prepare|ladder|score|selftest> ...  (see the header of native/tools/scribe.cpp)\n");
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "selftest" || cmd == "--selftest") return selftest();
  const Args a = parse_args(argc, argv, 2);
  if (cmd == "docs") return cmd_docs(a);
  if (cmd == "prompts") return cmd_prompts(a);
  if (cmd == "samples") return cmd_samples(a);
  if (cmd == "prepare") return cmd_prepare(a);
  if (cmd == "ladder") return cmd_ladder(a);
  if (cmd == "score") return cmd_score(a);
  std::fprintf(stderr, "unknown subcommand %s\n", cmd.c_str());
  return 2;
}
