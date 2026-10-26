// SCRIBE v1 control ladder (docs/PLAN.md section 6.4 V1, V4): does the metric order obvious controls and
// real model output, and does the old surface score S = H*D*L? Report: docs/reports/SCRIBE_V1.md.
//
// Data (enwik8 = 2006 Wikipedia XML; every range is disjoint from the trie's training text [0, 7 MiB)):
//   [16, 48) MiB  cleaned; trains the lexicon (words seen >= 2 times), the Kneser-Ney word trigram, the IDF,
//                 the PPMI-SVD word vectors (ppmi_svd on exactly these bytes) and the sentence-vector
//                 principal component; also 20 stretches for the E null
//   [48, 56) MiB  human REFERENCE samples (empirical CDFs, per length bucket: short 60-260 B, standard 400-600 B)
//                 and the sentence-length pmf for E3
//   [56, 64) MiB  DEVELOPMENT test: 500 held-out human samples (400-600 B, >= 3 sentences) + 500 short; the
//                 design of v1 was chosen on this ladder (it is v0's ladder, with v0's seed 20261027)
//   [64, 72) MiB  FRESH test, same recipe, seed 31415926 by default. Never tuned on: the verdict.
// Rungs, best to worst: held-out human > trie samples > sentences shuffled across documents > words shuffled
// within sentences > random lexicon words > random letters > one phrase repeated. The controls keep the
// surface form (sentence lengths, final punctuation, capital first letter). The trie rung is real model
// output from trie_sample (GEN<TAB>kind<TAB>text lines, 600-byte continuations of paragraph starts of the
// trie's holdout at T = 0.6, 0.8, 1.0, nucleus 0.9), prepared by clean_generated_text (markup stripped, the
// cut first and last word and the prompt's sentence dropped). Reported pooled and per temperature.
// PASS for a pair (PLAN 6.4): pairwise ordering accuracy >= 0.95 over all (upper, lower) pairs (ties 1/2) AND
// non-overlapping 95% bootstrap intervals of the rung means (1000 resamples). Run for SCRIBE and for surface S.
//
// What changed from v0 (details and numbers in the report):
//   C6 sentence vectors: PPMI-SVD word vectors, SIF weights (a = 1e-3), first principal component removed
//   C1 = mean cosine over ALL sentence pairs; C0 = word-level cohesion (all pairs of distinct content words)
//   C is defined for short samples: a sample with fewer than two embeddable sentences is cut into two halves
//   I gate: I < 0.2 caps the sample at I (the S gate is unchanged)
//   E3 = Kolmogorov distance of sentence lengths; the E null is drawn from 20 stretches, and the reference
//     length pmf comes from the reference range, not from the stretches
//
// Measured 2026-10-27 (fresh ladder, seed 31415926; 500 human samples; trie rung 623 of 750 continuations):
//   Mean SCRIBE / surface S: human 0.557 / 0.919; trie 0.035 / 0.734; sentences shuffled 0.227 / 0.902;
//   words shuffled 0.120 / 0.926; random lexicon 0.067 / 0.899; random letters 0.006 / 0.034; repeated 0.001 / 0.029.
//   Pair accuracies, SCRIBE / surface S (ladder with the pooled trie rung):
//     human > trie 0.993 PASS / 0.963 PASS      trie > sentences shuffled 0.035 / 0.039
//     sentences shuffled > words shuffled 0.829 / 0.348      words shuffled > random lexicon 0.722 / 0.727
//     random lexicon > random letters 1.000 PASS / 1.000 PASS      random letters > repeated 0.999 PASS / 0.556
//   SCRIBE passes 3 of 6 adjacent pairs, surface S 2 of 6 (v0: 1 of 6 each). Same three pairs pass at T = 0.6, 0.8,
//   1.0 and on two more seeds (27182818, 16180339); accuracies move by under 0.03.
//   V4: Pearson(SCRIBE, bytes) on human text 0.023 (standard), 0.107 (short), -0.009 pooled; short/standard
//   mean SCRIBE 1.02 (v0: 0.659 against 0.558). E: the 500-sample human set scores E = 0.65 on the fresh ladder
//   (0.11 on the development ladder): the null is still too narrow for the Zipf slope on some regions.
//
// Usage:
//   scribe_ladder <enwik8> --vectors PREFIX [--gen-dev FILE] [--gen-fresh FILE] [--fresh-seed N]
//                 [--only-dev | --only-fresh] [--c-one-sided] [--soft-gate]
//   scribe_ladder --selftest        (known-answer checks, no corpus; CTest native_scribe_selftest)
//   PREFIX: ppmi_svd output for enwik8 [16, 48) MiB only, e.g.
//     dd if=enwik8 of=train32.xml bs=1048576 skip=16 count=32
//     ppmi_svd train32.xml none none - PREFIX --no-ri --no-ablation        (the two similarity files may be absent)
//   FILE:   trie_sample output (dev and fresh files use disjoint prompts):
//     trie_sample enwik8 gen_dev.txt gen_fresh.txt
//   --c-one-sided and --soft-gate switch on the variants that were tried and not adopted (see the report).
//   Prints Markdown on stdout, progress on stderr. About 10 s and 210 MB peak for both ladders on one core; the output is deterministic.

#include "cypha/eval/scribe.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
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

struct Sample {
  std::string text;
  int doc = 0;
};

double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Consecutive, non-overlapping windows of sentences with total length in [lo, hi] and at least
// min_sent sentences; the target length is drawn uniformly in [lo, hi] per window.
std::vector<Sample> make_samples(const std::vector<std::string>& docs, std::size_t lo, std::size_t hi, int min_sent,
                                 int max_per_doc, std::mt19937_64& rng) {
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

// ---- controls -----------------------------------------------------------------------------

std::string ctl_shuffle_sentences(const Sample& s, const std::vector<std::pair<int, std::string>>& pool, std::mt19937_64& rng) {
  const std::vector<std::string> own = split_sentences(s.text);
  std::string out;
  std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
  for (std::size_t i = 0; i < own.size(); ++i) {
    const std::string* chosen = nullptr;
    for (int tries = 0; tries < 60; ++tries) {
      const auto& c = pool[pick(rng)];
      if (c.first == s.doc) continue;
      chosen = &c.second;
      const double r = static_cast<double>(c.second.size()) / static_cast<double>(own[i].size());
      if (r >= 0.6 && r <= 1.7) break;
    }
    if (!chosen) chosen = &pool[pick(rng)].second;
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
    std::string tail;  // terminal punctuation and closers stay at the end
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

// Replaces the alphabetic core of every whitespace token (tokens containing a digit stay).
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
    while (e < t.size() && (is_alpha_a(static_cast<unsigned char>(t[e])) || t[e] == '\'' || t[e] == '-' ||
                            static_cast<unsigned char>(t[e]) >= 0x80))
      ++e;
    std::string core = make_core(t.substr(b, e - b));
    if (is_upper_a(static_cast<unsigned char>(t[b])) && !core.empty() && core[0] >= 'a' && core[0] <= 'z')
      core[0] = static_cast<char>(core[0] - 32);
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
    while (!last.empty() && !is_alpha_a(static_cast<unsigned char>(last.back())) &&
           !is_digit_a(static_cast<unsigned char>(last.back())) && static_cast<unsigned char>(last.back()) < 0x80)
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


// ---- scoring a rung -----------------------------------------------------------------------

struct Rung {
  std::string name;
  std::vector<std::string> texts;
  std::vector<std::string> kinds;  // model rungs: the GEN kind of each text
  std::vector<Features> feats;
  std::vector<Scored> sc;
  std::vector<double> scribe, scribe_cap, scribe_soft, scribe_s, scribe_raw, surf;
  double E = 0, e1s = 0, e3s = 0, set_scribe = 0, zipf = 0, ks = 0;
  std::pair<double, double> ci_scribe{0, 0}, ci_s{0, 0}, ci_raw{0, 0}, ci_surf{0, 0}, ci_set{0, 0};
};

double mean_field(const Rung& r, double Scored::*f, bool only_c = false) {
  double s = 0;
  int n = 0;
  for (const Scored& x : r.sc) {
    if (only_c && !x.has_c) continue;
    s += x.*f;
    ++n;
  }
  return n ? s / n : 0.0;
}

// Mean of a sub-score over the samples that have it (c0, c1, c3 are absent for very short samples).
double mean_sub(const Rung& r, double Scored::*f, bool Features::*has) {
  double s = 0;
  int n = 0;
  for (std::size_t i = 0; i < r.sc.size(); ++i) {
    if (!(r.feats[i].*has)) continue;
    s += r.sc[i].*f;
    ++n;
  }
  return n ? s / n : 0.0;
}

std::string fmt(double v, int prec = 3) {
  char b[64];
  std::snprintf(b, sizeof b, "%.*f", prec, v);
  return b;
}

std::string fmt_ci(double m, std::pair<double, double> ci, int prec = 3) {
  return fmt(m, prec) + " [" + fmt(ci.first, prec) + ", " + fmt(ci.second, prec) + "]";
}

// GEN lines of a trie_sample file -> prepared texts (clean_generated_text), with their kinds.
// Texts under 100 bytes after preparation are dropped (counted); a text with no sentence boundary is kept.
struct GenSet {
  std::vector<std::string> texts, kinds;
  std::size_t lines = 0, dropped = 0;
  double raw_bytes = 0;
};

GenSet load_gen(const std::string& path) {
  GenSet g;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.compare(0, 4, "GEN\t") != 0) continue;
    const std::size_t t2 = line.find('\t', 4);
    if (t2 == std::string::npos) continue;
    ++g.lines;
    const std::string raw = unescape(line.substr(t2 + 1));
    g.raw_bytes += static_cast<double>(raw.size());
    const std::string text = clean_generated_text(raw);
    if (text.size() < 100) {
      ++g.dropped;
      continue;
    }
    g.texts.push_back(text);
    g.kinds.push_back(line.substr(4, t2 - 4));
  }
  return g;
}


// ---- selftest -----------------------------------------------------------------------------

int g_fail = 0, g_checks = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      ++g_fail;                                                           \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);         \
    }                                                                     \
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
  // Cleaner.
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
  // Sentences and tokens.
  const std::vector<std::string> ss = split_sentences("Mr. Smith went to Washington. He liked it! Did he? J. R. Tolkien wrote books.");
  CHECK(ss.size() == 4);
  CHECK(ss.size() == 4 && ss[0] == "Mr. Smith went to Washington." && ss[3] == "J. R. Tolkien wrote books.");
  CHECK(split_sentences("It ended. then lower case began.").size() == 2);
  std::vector<std::string> tk;
  tokenize_words("Rose's 1,000 don't U.S. cars-and-trucks", tk);
  CHECK((tk == std::vector<std::string>{"rose", "'s", "1,000", "don't", "u", "s", "cars", "and", "trucks"}));
  CHECK(numeric_token("1,000") && numeric_token("1980s") && !numeric_token("abc") && !numeric_token("3d"));
  // Statistics.
  CHECK_NEAR(jsd_bits({1, 0}, {0, 1}), 1.0, 1e-12);
  CHECK_NEAR(jsd_bits({0.5, 0.5}, {0.5, 0.5}), 0.0, 1e-12);
  CHECK_NEAR(jsd_bits({0.5, 0.5}, {1, 0}), 0.311278, 1e-5);
  Ecdf e;
  for (int i = 1; i <= 100; ++i) e.add(i);
  e.finish();
  CHECK_NEAR(e.F(50.5), 0.5, 1e-12);
  CHECK_NEAR(e.one_sided(50.5), 1.0, 1e-12);
  CHECK_NEAR(e.two_sided(50.5), 1.0, 1e-12);
  CHECK_NEAR(e.two_sided(0.0), 1.0 / 101.0, 1e-12);
  CHECK_NEAR(e.one_sided(0.0), 1.0 / 101.0, 1e-12);
  CHECK_NEAR(e.one_sided(1000.0), 1.0, 1e-12);
  CHECK_NEAR(pairwise_accuracy({3}, {1, 2, 3}), 2.5 / 3.0, 1e-12);
  CHECK_NEAR(pearson({1, 2, 3, 4}, {2, 4, 6, 8}), 1.0, 1e-12);
  CHECK_NEAR(spearman({1, 2, 3, 4}, {1, 8, 27, 64}), 1.0, 1e-12);
  CHECK_NEAR(scribe_detail::loop8_bytes("abababababababababab"), 11.0 / 13.0, 1e-12);
  CHECK(scribe_detail::utf8_valid("caf\xC3\xA9") && !scribe_detail::utf8_valid("caf\xC3") && !scribe_detail::utf8_valid("\xC0\x80"));
  // Kneser-Ney: probabilities sum to one over the vocabulary in every context.
  ScribeModel m;
  m.train({"The cat sat on the mat. The dog sat on the log. A cat saw a dog.", "The dog saw the cat. The mat was red. A log was on the mat."}, 1);
  const std::uint32_t the = m.word_id("the"), cat = m.word_id("cat"), dog = m.word_id("dog");
  CHECK(the != ScribeModel::kUnk && cat != ScribeModel::kUnk && m.word_id("zebra") == ScribeModel::kUnk);
  const std::uint32_t ctxs[][2] = {{ScribeModel::kBos, ScribeModel::kBos}, {the, cat}, {cat, dog}, {the, the}, {dog, ScribeModel::kUnk}};
  for (const auto& c : ctxs) {
    double s = 0;
    for (std::uint32_t w = 0; w < m.vocab_size(); ++w) s += std::exp(m.logp(c[0], c[1], w));
    CHECK_NEAR(s, 1.0, 1e-9);
  }
  CHECK(m.logp(the, cat, m.word_id("sat")) > m.logp(the, cat, m.word_id("red")));  // seen continuation beats unseen
  // Features on tiny inputs.
  WordDict dict;
  const Features ok = extract_features(m, dict, "The cat sat on the mat. The dog saw the cat.");
  const Features bad = extract_features(m, dict, "the cat (sat on");
  CHECK_NEAR(ok.s1, 1.0, 1e-12);
  CHECK_NEAR(ok.s4, 1.0, 1e-12);
  CHECK(ok.has_c && ok.n_sent == 2 && ok.n_words == 11);
  CHECK_NEAR(bad.s4, 2.0 / 6.0, 1e-12);  // only UTF-8 and even quotes pass
  CHECK_NEAR(ok.d[0], 7.0 / 11.0, 1e-12);  // the cat sat on mat dog saw
  const Features rep = extract_features(m, dict, "The cat sat. The cat sat. The cat sat.");
  CHECK_NEAR(rep.rep, 2.0 / 3.0, 1e-12);
  // Surface score port: a loop scores far below its source.
  SurfaceRef sr;
  surface_add_vocab(sr, "the quick brown fox jumps over the lazy dog and the cat");
  CHECK(surface_S(sr, "the quick brown fox jumps over the lazy dog and the cat") >
        surface_S(sr, "zzz qqq xxx zzz qqq xxx zzz qqq xxx zzz qqq xxx"));
  // ---- v1 additions -----------------------------------------------------------------------
  // Mean cosine over all pairs: {e1, e1, e2} -> (1 + 0 + 0) / 3.
  {
    const std::vector<std::vector<float>> U = {{1, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    CHECK_NEAR(scribe_detail::mean_pairwise_cosine(U), 1.0 / 3.0, 1e-9);
    CHECK_NEAR(scribe_detail::mean_pairwise_cosine({{1, 0}, {0, 1}}), 0.0, 1e-9);
  }
  // Word vectors: centring gives unit rows; a binary round trip through streams; SIF weights.
  {
    WordVectors wv;
    wv.set({"aa", "bb", "cc", "dd"}, {1000, 100, 10, 1}, {1, 0.2f, 0, 1, 0.1f, 0, 0.9f, 0.3f, 0, 0, 1, 0.2f}, 3);
    for (int i = 0; i < 4; ++i) {
      double n2 = 0;
      for (int k = 0; k < 3; ++k) n2 += double(wv.row(i)[k]) * wv.row(i)[k];
      CHECK_NEAR(n2, 1.0, 1e-5);
    }
    CHECK(wv.index("cc") == 2 && wv.index("zz") == -1 && wv.dim() == 3 && wv.size() == 4);
    CHECK(wv.prob(0) > wv.prob(1) && wv.prob(1) > wv.prob(3));
    std::stringstream vs(std::ios::in | std::ios::out | std::ios::binary), ts;
    const float raw[6] = {1, 0, 0, 0, 1, 0};
    vs.write(reinterpret_cast<const char*>(raw), sizeof raw);
    ts << "the\t100\nof\t50\n";
    WordVectors w2;
    CHECK(w2.load_streams(vs, ts, 3) && w2.size() == 2 && w2.index("of") == 1);
    std::stringstream vshort(std::ios::in | std::ios::out | std::ios::binary), t2;
    vshort.write(reinterpret_cast<const char*>(raw), 8);
    t2 << "the\t100\nof\t50\n";
    WordVectors w3;
    CHECK(!w3.load_streams(vshort, t2, 3));  // file shorter than the vocabulary says
    // SIF: the rare word dominates the sentence vector; PC removal leaves a unit vector orthogonal to the PC.
    SentenceEmbedder se(&wv);
    std::vector<float> a, b;
    CHECK(se.raw({"aa", "dd"}, a) && !se.raw({"aa", "zz"}, b, 2) && se.raw({"aa", "zz"}, b, 1));
    std::vector<std::vector<float>> X;
    std::mt19937 r(5);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    for (int i = 0; i < 200; ++i) X.push_back({3.0f + nd(r), nd(r), nd(r)});
    se.fit_pc(X);
    CHECK(se.pc().size() == 3 && std::fabs(se.pc()[0]) > 0.99);
    std::vector<float> e;
    CHECK(se.embed({"aa", "bb", "dd"}, e));
    double n2 = 0, dot = 0;
    for (int k = 0; k < 3; ++k) {
      n2 += double(e[k]) * e[k];
      dot += double(e[k]) * se.pc()[static_cast<std::size_t>(k)];
    }
    CHECK_NEAR(n2, 1.0, 1e-5);
    CHECK_NEAR(dot, 0.0, 1e-5);
  }
  // C is defined for short samples (two halves), C1 uses all pairs, the sample's content words give C0.
  {
    ScribeModel m2;
    std::vector<std::string> docs;
    for (int i = 0; i < 30; ++i) docs.push_back("The cat sat on the mat today. The dog saw the cat and the log. A red mat was on the log.");
    m2.train(docs, 1);
    std::vector<std::string> ws;
    std::vector<double> cn;
    std::vector<float> vv;
    std::mt19937 r(9);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const std::string& w : m2.words()) {
      ws.push_back(w);
      cn.push_back(10);
      for (int k = 0; k < 8; ++k) vv.push_back(nd(r));
    }
    WordVectors wv;
    wv.set(ws, cn, vv, 8);
    SentenceEmbedder se(&wv, 1e-3);
    WordDict dict;
    const Features one = extract_features(m2, dict, "The cat sat on the mat today and the dog saw the red log.", &se);
    CHECK(one.n_sent == 1 && one.has_c1 && one.has_c);  // one sentence: halves
    const Features tiny = extract_features(m2, dict, "The cat sat.", &se);
    CHECK(!tiny.has_c1 && !tiny.has_c0);
    const Features three = extract_features(m2, dict, "The cat sat on the mat. The dog saw the cat. A red mat was on the log.", &se);
    CHECK(three.n_sent == 3 && three.has_c1);
    // Same text with the TF-IDF fallback (no embedder): all-pairs cosine, no C0.
    const Features fb = extract_features(m2, dict, "The cat sat on the mat. The dog saw the cat. A red mat was on the log.", nullptr);
    CHECK(fb.has_c1 && !fb.has_c0);
  }
  // E3 Kolmogorov distance: all lengths 5 against a pmf of half 5s and half 10s -> 0.5.
  {
    Features f;
    f.sent_len = {5, 5, 5, 5};
    std::vector<double> pmf(kMaxLen + 1, 0.0);
    pmf[5] = 0.5;
    pmf[10] = 0.5;
    CHECK_NEAR(ks_lengths({&f}, pmf), 0.5, 1e-12);
    pmf.assign(kMaxLen + 1, 0.0);
    pmf[5] = 1.0;
    CHECK_NEAR(ks_lengths({&f}, pmf), 0.0, 1e-12);
  }
  // I gate: a sample with perfect form but a heavy loop is capped at I; without the gate it is not.
  {
    ScribeScorer sc;
    std::mt19937 r(3);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int i = 0; i < 300; ++i) {
      Features f;
      f.bytes = 450;
      f.n_words = 80;
      f.s1 = 0.9 + 0.1 * U(r);
      f.s3 = -6 + U(r);
      f.s4 = 0.8 + 0.2 * U(r);
      f.loop8 = 0.1 * U(r);
      f.rep = 0.0;
      f.d[0] = 0.55 + 0.1 * U(r);
      f.d[1] = 0.8 + 0.1 * U(r);
      f.d[2] = 0.9 + 0.05 * U(r);
      f.d[3] = 0.95 + 0.04 * U(r);
      f.has_c = f.has_c0 = f.has_c1 = f.has_c3 = true;
      f.c0 = 0.1 + 0.05 * U(r);
      f.c1 = 0.2 + 0.1 * U(r);
      f.c3 = 0.5 + 0.2 * U(r);
      sc.add_reference(f);
    }
    sc.finish();
    Features good = {};
    good.bytes = 450;
    good.n_words = 80;
    good.s1 = 1.0;
    good.s3 = -5.5;
    good.s4 = 1.0;
    good.loop8 = 0.05;
    good.d[0] = 0.6;
    good.d[1] = 0.85;
    good.d[2] = 0.92;
    good.d[3] = 0.97;
    good.has_c = good.has_c0 = good.has_c1 = good.has_c3 = true;
    good.c0 = 0.12;
    good.c1 = 0.25;
    good.c3 = 0.6;
    Features loop = good;
    loop.loop8 = 0.9;
    loop.rep = 0.8;
    loop.d[0] = 0.1;
    loop.d[1] = 0.1;
    loop.d[2] = 0.1;
    loop.d[3] = 0.1;
    const Scored a = sc.score(good), b = sc.score(loop);
    CHECK(!a.i_gate && !a.s_gate && a.scribe > 0.3);
    CHECK(b.i_gate && !b.s_gate && b.scribe <= b.I + 1e-12 && b.scribe < b.scribe_s);
    CHECK(b.scribe_s > b.scribe);  // the PLAN's S gate alone lets the loop through
  }
  // Model-output cleaning: markup stripped, first (prompt) sentence and the cut-off last sentence dropped.
  {
    const std::string g = clean_generated_text(
        " of the [[city]] was large. It has ''many'' old [[Rome|buildings]] and wide squares.\n\nAnother sentence here is fine. The next one is cu");
    CHECK(g == "It has many old buildings and wide squares. Another sentence here is fine.");
    CHECK(clean_generated_text("no terminator at all here") == "terminator at all");  // cut first and last word, no boundary: kept whole
    CHECK(clean_generated_text(" alpha beta. gamma delta eps. zeta eta") == "gamma delta eps.");  // prompt remnant and unterminated last sentence dropped
  }
  std::printf("scribe selftest: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}

}  // namespace

namespace {

// Everything that does not depend on the ladder being run: models, references, pools for the E null.
struct Ctx {
  ScribeModel model;
  WordVectors wv;
  SentenceEmbedder emb;
  SurfaceRef surf;
  ScribeScorer scorer;
  WordDict dict;
  std::vector<Features> pool_feats;
  std::vector<std::vector<const Features*>> stretches[2];  // per length bucket: about two dozen disjoint stretches
  std::map<std::pair<int, std::size_t>, SetReference> set_refs;
  std::vector<double> ref_pmf[2];  // sentence-length pmf of the reference SAMPLES of each length bucket (not of the training text)
  double t0 = 0;
  std::size_t n_train_docs = 0, train_bytes = 0, n_ref_std = 0, n_ref_short = 0, n_stretches = 0;
  bool use_vectors = false;
  const SentenceEmbedder* embedder() const { return use_vectors ? &emb : nullptr; }
};

const SetReference& set_ref_for(Ctx& c, int bucket, std::size_t n) {
  const auto key = std::make_pair(bucket, n);
  auto it = c.set_refs.find(key);
  if (it == c.set_refs.end())
    it = c.set_refs.emplace(key, build_set_reference(c.stretches[bucket], c.ref_pmf[bucket], n, c.dict.size(), 300,
                                                     11 + static_cast<std::uint64_t>(bucket) * 1000 + n)).first;
  return it->second;
}

struct PairRes {
  double acc;
  bool disjoint, pass;
};
PairRes pair_res(const Rung& hi, const Rung& lo, const std::vector<double> Rung::*vals, const std::pair<double, double> Rung::*ci) {
  PairRes r;
  r.acc = pairwise_accuracy(hi.*vals, lo.*vals);
  r.disjoint = (hi.*ci).first > (lo.*ci).second;
  r.pass = r.acc >= 0.95 && r.disjoint;
  return r;
}

// Builds one ladder from `test_docs`, scores it and prints the report. dev_label only names it.
void run_ladder(Ctx& c, const char* label, const std::vector<std::string>& test_docs, std::mt19937_64& rng,
                std::uint64_t ctl_seed, const std::string& gen_path, std::uint64_t seed, const char* range_name) {
  const double t0 = c.t0;
  // ---- samples. The order of the draws from rng is the order v0 used, so the development ladder keeps v0's human samples.
  std::vector<Sample> test_std_all = make_samples(test_docs, 400, 600, 3, 4, rng);
  const std::vector<Sample> test_std = pick(test_std_all, 500, rng);
  const std::vector<Sample> test_short = pick(make_samples(test_docs, 60, 260, 1, 3, rng), 500, rng);
  std::fprintf(stderr, "[%.0fs] %s samples: std %zu (of %zu candidates), short %zu\n", now_s() - t0, label, test_std.size(),
               test_std_all.size(), test_short.size());
  if (test_std.size() < 100) {
    std::fprintf(stderr, "too few test samples\n");
    return;
  }
  std::mt19937_64 crng(ctl_seed);
  std::vector<std::pair<int, std::string>> sent_pool;
  for (const Sample& s : test_std)
    for (const std::string& x : split_sentences(s.text)) sent_pool.emplace_back(s.doc, x);

  enum { R_HUMAN, R_TRIE, R_SS, R_WS, R_RL, R_RLET, R_REP, R_T06, R_T08, R_T10, R_SHORT, R_N };
  std::vector<Rung> rungs(R_N);
  const char* names[R_N] = {"held-out human",          "trie samples (pooled)",   "sentences shuffled across docs",
                            "words shuffled within sentences", "random lexicon words", "random letters",
                            "one phrase repeated",      "trie samples T=0.6",      "trie samples T=0.8",
                            "trie samples T=1.0",       "held-out human, short (60-260 B)"};
  for (int i = 0; i < R_N; ++i) rungs[static_cast<std::size_t>(i)].name = names[i];
  for (const Sample& s : test_std) {
    rungs[R_HUMAN].texts.push_back(s.text);
    rungs[R_SS].texts.push_back(ctl_shuffle_sentences(s, sent_pool, crng));
    rungs[R_WS].texts.push_back(ctl_shuffle_words(s.text, crng));
    rungs[R_RL].texts.push_back(ctl_random_lexicon(s.text, c.model, crng));
    rungs[R_RLET].texts.push_back(ctl_random_letters(s.text, crng));
    rungs[R_REP].texts.push_back(ctl_repeat_phrase(s.text));
  }
  for (const Sample& s : test_short) rungs[R_SHORT].texts.push_back(s.text);
  const GenSet gen = gen_path.empty() ? GenSet{} : load_gen(gen_path);
  for (std::size_t i = 0; i < gen.texts.size(); ++i) {
    rungs[R_TRIE].texts.push_back(gen.texts[i]);
    rungs[R_TRIE].kinds.push_back(gen.kinds[i]);
    const int slot = gen.kinds[i] == "trie_T0.6" ? R_T06 : (gen.kinds[i] == "trie_T0.8" ? R_T08 : (gen.kinds[i] == "trie_T1.0" ? R_T10 : -1));
    if (slot >= 0) rungs[static_cast<std::size_t>(slot)].texts.push_back(gen.texts[i]);
  }
  std::fprintf(stderr, "[%.0fs] %s ladder built; trie GEN lines %zu, kept %zu\n", now_s() - t0, label, gen.lines, gen.texts.size());
  for (Rung& r : rungs) {
    r.feats.reserve(r.texts.size());
    for (const std::string& t : r.texts) r.feats.push_back(extract_features(c.model, c.dict, t, c.embedder()));
  }
  // Human test pool for the E calibration check.
  std::vector<Features> calib_feats;
  calib_feats.reserve(test_std_all.size());
  for (const Sample& s : test_std_all) calib_feats.push_back(extract_set_features(c.dict, s.text));
  std::fprintf(stderr, "[%.0fs] %s features extracted (dict %zu)\n", now_s() - t0, label, c.dict.size());

  // ---- scoring
  std::uint64_t bseed = 100 + seed % 1000;
  for (Rung& r : rungs) {
    for (std::size_t i = 0; i < r.feats.size(); ++i) {
      const Scored s = c.scorer.score(r.feats[i]);
      r.sc.push_back(s);
      r.scribe.push_back(s.scribe);
      r.scribe_cap.push_back(s.scribe_cap);
      r.scribe_soft.push_back(s.scribe_soft);
      r.scribe_s.push_back(s.scribe_s);
      r.scribe_raw.push_back(s.scribe_raw);
      r.surf.push_back(surface_S(c.surf, r.texts[i]));
    }
    r.ci_scribe = bootstrap_mean_ci(r.scribe, 1000, ++bseed);
    r.ci_s = bootstrap_mean_ci(r.scribe_s, 1000, ++bseed);
    r.ci_raw = bootstrap_mean_ci(r.scribe_raw, 1000, ++bseed);
    r.ci_surf = bootstrap_mean_ci(r.surf, 1000, ++bseed);
  }
  std::vector<std::uint32_t> scratch(c.dict.size(), 0);
  for (Rung& r : rungs) {
    if (r.feats.size() < 50) continue;
    std::vector<std::size_t> bytes;
    for (const Features& f : r.feats) bytes.push_back(f.bytes);
    std::nth_element(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(bytes.size() / 2), bytes.end());
    const int b = length_bucket(bytes[bytes.size() / 2]);
    const std::size_t n = std::min<std::size_t>(r.feats.size(), 500);
    std::vector<const Features*> set;
    for (std::size_t i = 0; i < n; ++i) set.push_back(&r.feats[i]);
    const SetReference& sref = set_ref_for(c, b, n);
    const SetValues v = set_values(set, sref.pmf, scratch);
    r.zipf = v.zipf;
    r.ks = v.ks;
    r.E = sref.E(v, &r.e1s, &r.e3s);
    double ms = 0;
    for (std::size_t i = 0; i < n; ++i) ms += r.scribe[i];
    r.set_scribe = ms / static_cast<double>(n) * std::pow(r.E, c.scorer.profile.wE);
  }
  std::fprintf(stderr, "[%.0fs] %s scored\n", now_s() - t0, label);

  // ---- report
  std::printf("\n# %s ladder: human samples from enwik8 %s (seed %llu)\n\n", label, range_name, static_cast<unsigned long long>(seed));
  std::printf("%zu standard human samples (400-600 B, >= 3 sentences) chosen from %zu candidates, %zu short samples (60-260 B). Trie rung: %zu GEN lines in %s, %zu kept after cleaning "
              "(%zu dropped for under 100 B after cleaning); mean raw length %.0f B.\n\n",
              test_std.size(), test_std_all.size(), test_short.size(), gen.lines, gen_path.empty() ? "(none)" : gen_path.c_str(),
              gen.texts.size(), gen.dropped, gen.lines ? gen.raw_bytes / static_cast<double>(gen.lines) : 0.0);

  const double human_mean = mean_of(rungs[R_HUMAN].scribe);
  std::printf("## Rung means (95%% bootstrap interval of the mean, 1000 resamples)\n\n");
  std::printf("| rung | n | SCRIBE | parity | surface S | mean bytes | E | set-level SCRIBE |\n|---|---|---|---|---|---|---|---|\n");
  auto row = [&](const Rung& r) {
    if (r.feats.empty()) return;
    double by = 0;
    for (const Features& f : r.feats) by += static_cast<double>(f.bytes);
    by /= static_cast<double>(r.feats.size());
    std::printf("| %s | %zu | %s | %.2f | %s | %.0f | %s | %s |\n", r.name.c_str(), r.feats.size(),
                fmt_ci(mean_of(r.scribe), r.ci_scribe).c_str(), mean_of(r.scribe) / human_mean,
                fmt_ci(mean_of(r.surf), r.ci_surf).c_str(), by, r.feats.size() >= 50 ? fmt(r.E, 2).c_str() : "-",
                r.feats.size() >= 50 ? fmt(r.set_scribe, 3).c_str() : "-");
  };
  for (int i : {R_HUMAN, R_TRIE, R_SS, R_WS, R_RL, R_RLET, R_REP}) row(rungs[static_cast<std::size_t>(i)]);
  std::printf("\nExtra rows (not on the ladder):\n\n| rung | n | SCRIBE | parity | surface S | mean bytes | E | set-level SCRIBE |\n|---|---|---|---|---|---|---|---|\n");
  for (int i : {R_T06, R_T08, R_T10, R_SHORT}) row(rungs[static_cast<std::size_t>(i)]);

  // Ladders: the pooled trie rung, each temperature, and the ladder without a model rung.
  struct Chain {
    std::string name;
    std::vector<int> idx;
  };
  const std::vector<Chain> chains = {{"trie pooled", {R_HUMAN, R_TRIE, R_SS, R_WS, R_RL, R_RLET, R_REP}},
                                     {"trie T=0.6", {R_HUMAN, R_T06, R_SS, R_WS, R_RL, R_RLET, R_REP}},
                                     {"trie T=0.8", {R_HUMAN, R_T08, R_SS, R_WS, R_RL, R_RLET, R_REP}},
                                     {"trie T=1.0", {R_HUMAN, R_T10, R_SS, R_WS, R_RL, R_RLET, R_REP}},
                                     {"no model rung", {R_HUMAN, R_SS, R_WS, R_RL, R_RLET, R_REP}}};
  std::printf("\n## V1 ladder: pairwise ordering accuracy over all (upper, lower) pairs (ties = 1/2) and disjoint 95%% intervals; PASS needs both (accuracy >= 0.95)\n");
  for (const Chain& ch : chains) {
    if (rungs[static_cast<std::size_t>(ch.idx[1])].feats.empty()) continue;
    std::printf("\n### Ladder with: %s\n\n| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |\n|---|---|---|---|---|---|---|\n", ch.name.c_str());
    int ps = 0, pu = 0, np = 0;
    for (std::size_t k = 0; k + 1 < ch.idx.size(); ++k) {
      const Rung &a = rungs[static_cast<std::size_t>(ch.idx[k])], &b = rungs[static_cast<std::size_t>(ch.idx[k + 1])];
      const PairRes x = pair_res(a, b, &Rung::scribe, &Rung::ci_scribe), y = pair_res(a, b, &Rung::surf, &Rung::ci_surf);
      ps += x.pass;
      pu += y.pass;
      ++np;
      std::printf("| %s > %s | %.3f | %s | %s | %.3f | %s | %s |\n", a.name.c_str(), b.name.c_str(), x.acc, x.disjoint ? "yes" : "no",
                  x.pass ? "PASS" : "FAIL", y.acc, y.disjoint ? "yes" : "no", y.pass ? "PASS" : "FAIL");
    }
    std::printf("\nVerdict (%s): SCRIBE %s (%d of %d pairs), surface S %s (%d of %d pairs)\n", ch.name.c_str(), ps == np ? "PASS" : "FAIL", ps, np,
                pu == np ? "PASS" : "FAIL", pu, np);
  }

  std::printf("\n## Gate variants on the pooled ladder: pairwise accuracy (headline = the configured gate; caps = PLAN 6.3 S cap plus the I cap; scales = raw * min(1,S/0.2) * min(1,I/0.2); S cap only = v0; raw = no gate)\n\n");
  std::printf("| upper > lower | headline | caps | scales | S cap only | raw |\n|---|---|---|---|---|---|\n");
  {
    const Chain& ch = chains[0];
    if (!rungs[R_TRIE].feats.empty())
      for (std::size_t k = 0; k + 1 < ch.idx.size(); ++k) {
        const Rung &a = rungs[static_cast<std::size_t>(ch.idx[k])], &b = rungs[static_cast<std::size_t>(ch.idx[k + 1])];
        std::printf("| %s > %s | %.3f | %.3f | %.3f | %.3f | %.3f |\n", a.name.c_str(), b.name.c_str(), pairwise_accuracy(a.scribe, b.scribe),
                    pairwise_accuracy(a.scribe_cap, b.scribe_cap), pairwise_accuracy(a.scribe_soft, b.scribe_soft),
                    pairwise_accuracy(a.scribe_s, b.scribe_s), pairwise_accuracy(a.scribe_raw, b.scribe_raw));
      }
  }

  // Which score separates which pair: accuracy of upper > lower for every sub-score and axis (samples without the sub-score are left out).
  std::printf("\n## Which score separates each pair (pairwise accuracy of upper > lower, pooled ladder; 0.50 = no information, below 0.50 = backwards)\n\n");
  std::printf("| upper > lower | S1 | S3 | S4 | S | C0 | C1 | C3 | C | I2 | I3 | I |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n");
  {
    auto vec = [](const Rung& r, double Scored::*f, bool Features::*has) {
      std::vector<double> v;
      for (std::size_t i = 0; i < r.sc.size(); ++i)
        if (!has || r.feats[i].*has) v.push_back(r.sc[i].*f);
      return v;
    };
    const Chain& ch = chains[0];
    if (!rungs[R_TRIE].feats.empty())
      for (std::size_t k = 0; k + 1 < ch.idx.size(); ++k) {
        const Rung &a = rungs[static_cast<std::size_t>(ch.idx[k])], &b = rungs[static_cast<std::size_t>(ch.idx[k + 1])];
        auto acc = [&](double Scored::*f, bool Features::*has = nullptr) { return pairwise_accuracy(vec(a, f, has), vec(b, f, has)); };
        std::printf("| %s > %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n", a.name.c_str(), b.name.c_str(),
                    acc(&Scored::s1), acc(&Scored::s3), acc(&Scored::s4), acc(&Scored::S), acc(&Scored::c0, &Features::has_c0),
                    acc(&Scored::c1, &Features::has_c1), acc(&Scored::c3, &Features::has_c3), acc(&Scored::C, &Features::has_c), acc(&Scored::i2),
                    acc(&Scored::i3), acc(&Scored::I));
      }
  }
  std::printf("\n## Where the pooled trie samples fall: share of (trie, control) pairs in which the trie scores higher\n\n| control rung | SCRIBE | surface S |\n|---|---|---|\n");
  if (!rungs[R_TRIE].feats.empty())
    for (int i : {R_HUMAN, R_SS, R_WS, R_RL, R_RLET, R_REP}) {
      const Rung& o = rungs[static_cast<std::size_t>(i)];
      std::printf("| %s | %.3f | %.3f |\n", o.name.c_str(), pairwise_accuracy(rungs[R_TRIE].scribe, o.scribe), pairwise_accuracy(rungs[R_TRIE].surf, o.surf));
    }

  std::printf("\n## Axis profile (mean over samples; C over samples where it is defined)\n\n");
  std::printf("| rung | S | C (n) | I | S gate | I gate | E1 (Zipf) | E3 (lengths, KS) | E |\n|---|---|---|---|---|---|---|---|---|\n");
  for (int i : {R_HUMAN, R_TRIE, R_T06, R_T08, R_T10, R_SS, R_WS, R_RL, R_RLET, R_REP, R_SHORT}) {
    const Rung& r = rungs[static_cast<std::size_t>(i)];
    if (r.feats.empty()) continue;
    int nc = 0, g1 = 0, g2 = 0;
    for (const Scored& s : r.sc) {
      nc += s.has_c;
      g1 += s.s_gate;
      g2 += s.i_gate;
    }
    const double n = static_cast<double>(std::max<std::size_t>(1, r.sc.size()));
    std::printf("| %s | %s | %s (%d) | %s | %.0f%% | %.0f%% | %s | %s | %s |\n", r.name.c_str(), fmt(mean_field(r, &Scored::S), 2).c_str(),
                fmt(mean_field(r, &Scored::C, true), 2).c_str(), nc, fmt(mean_field(r, &Scored::I), 2).c_str(), 100.0 * g1 / n,
                100.0 * g2 / n, r.feats.size() >= 50 ? fmt(r.e1s, 2).c_str() : "-", r.feats.size() >= 50 ? fmt(r.e3s, 2).c_str() : "-",
                r.feats.size() >= 50 ? fmt(r.E, 2).c_str() : "-");
  }
  std::printf("\n## Sub-scores (mean; each over the samples where it is defined)\n\n");
  std::printf("| rung | S1 | S3 | S4 | C0 | C1 | C3 | I2 | I3 |\n|---|---|---|---|---|---|---|---|---|\n");
  for (int i : {R_HUMAN, R_TRIE, R_T06, R_T08, R_T10, R_SS, R_WS, R_RL, R_RLET, R_REP, R_SHORT}) {
    const Rung& r = rungs[static_cast<std::size_t>(i)];
    if (r.feats.empty()) continue;
    std::printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", r.name.c_str(), fmt(mean_field(r, &Scored::s1), 2).c_str(),
                fmt(mean_field(r, &Scored::s3), 2).c_str(), fmt(mean_field(r, &Scored::s4), 2).c_str(),
                fmt(mean_sub(r, &Scored::c0, &Features::has_c0), 2).c_str(), fmt(mean_sub(r, &Scored::c1, &Features::has_c1), 2).c_str(),
                fmt(mean_sub(r, &Scored::c3, &Features::has_c3), 2).c_str(), fmt(mean_field(r, &Scored::i2), 2).c_str(),
                fmt(mean_field(r, &Scored::i3), 2).c_str());
  }
  std::printf("\nRaw means: ");
  for (int i : {R_HUMAN, R_TRIE, R_SS, R_WS, R_RL, R_RLET, R_REP}) {
    const Rung& r = rungs[static_cast<std::size_t>(i)];
    if (r.feats.empty()) continue;
    double s3 = 0, c0 = 0, c1 = 0, n0 = 0, n1 = 0;
    for (const Features& f : r.feats) {
      s3 += f.s3;
      if (f.has_c0) {
        c0 += f.c0;
        n0 += 1;
      }
      if (f.has_c1) {
        c1 += f.c1;
        n1 += 1;
      }
    }
    std::printf("%s: S3 %.2f nats/token, C0 %.3f, C1 %.3f; ", r.name.c_str(), s3 / static_cast<double>(r.feats.size()), n0 ? c0 / n0 : 0.0, n1 ? c1 / n1 : 0.0);
  }
  std::printf("\n");

  std::printf("\n## Trie rung by temperature (SCRIBE mean, surface S mean, gate rates)\n\n| kind | n | SCRIBE | surface S | S gate | I gate | mean bytes |\n|---|---|---|---|---|---|---|\n");
  {
    std::map<std::string, std::vector<std::size_t>> by;
    for (std::size_t i = 0; i < rungs[R_TRIE].kinds.size(); ++i) by[rungs[R_TRIE].kinds[i]].push_back(i);
    for (const auto& kv : by) {
      double a = 0, b = 0, g1 = 0, g2 = 0, by_ = 0;
      for (std::size_t i : kv.second) {
        a += rungs[R_TRIE].scribe[i];
        b += rungs[R_TRIE].surf[i];
        g1 += rungs[R_TRIE].sc[i].s_gate;
        g2 += rungs[R_TRIE].sc[i].i_gate;
        by_ += static_cast<double>(rungs[R_TRIE].feats[i].bytes);
      }
      const double n = static_cast<double>(kv.second.size());
      std::printf("| %s | %zu | %.3f | %.3f | %.0f%% | %.0f%% | %.0f |\n", kv.first.c_str(), kv.second.size(), a / n, b / n, 100 * g1 / n, 100 * g2 / n, by_ / n);
    }
  }

  std::printf("\n## V4 stability: correlation of the score with sample length on held-out human text\n\n");
  std::printf("| set | n | metric | Pearson vs bytes | Spearman vs bytes | Pearson vs words |\n|---|---|---|---|---|---|\n");
  auto corr_row = [&](const char* nm, const Rung& r) {
    std::vector<double> by, wd;
    for (const Features& f : r.feats) {
      by.push_back(static_cast<double>(f.bytes));
      wd.push_back(static_cast<double>(f.n_words));
    }
    std::printf("| %s | %zu | SCRIBE | %.3f | %.3f | %.3f |\n", nm, r.feats.size(), pearson(r.scribe, by), spearman(r.scribe, by), pearson(r.scribe, wd));
    std::printf("| %s | %zu | surface S | %.3f | %.3f | %.3f |\n", nm, r.feats.size(), pearson(r.surf, by), spearman(r.surf, by), pearson(r.surf, wd));
  };
  corr_row("standard 400-600 B", rungs[R_HUMAN]);
  corr_row("short 60-260 B", rungs[R_SHORT]);
  {
    std::vector<double> sc_all = rungs[R_HUMAN].scribe, su_all = rungs[R_HUMAN].surf, by_all;
    for (const Features& f : rungs[R_HUMAN].feats) by_all.push_back(static_cast<double>(f.bytes));
    for (std::size_t i = 0; i < rungs[R_SHORT].feats.size(); ++i) {
      sc_all.push_back(rungs[R_SHORT].scribe[i]);
      su_all.push_back(rungs[R_SHORT].surf[i]);
      by_all.push_back(static_cast<double>(rungs[R_SHORT].feats[i].bytes));
    }
    std::printf("| both buckets pooled | %zu | SCRIBE | %.3f | %.3f | - |\n", sc_all.size(), pearson(sc_all, by_all), spearman(sc_all, by_all));
    std::printf("| both buckets pooled | %zu | surface S | %.3f | %.3f | - |\n", su_all.size(), pearson(su_all, by_all), spearman(su_all, by_all));
  }
  {
    std::size_t one = 0;
    for (const Features& f : rungs[R_SHORT].feats) one += (f.n_sent < 2);
    std::printf("\nMean SCRIBE of held-out human: standard %.3f, short %.3f (bucket parity %.2f; %zu of %zu short samples have one sentence). Mean surface S: standard %.3f, short %.3f.\n",
                mean_of(rungs[R_HUMAN].scribe), mean_of(rungs[R_SHORT].scribe),
                mean_of(rungs[R_SHORT].scribe) / std::max(1e-9, mean_of(rungs[R_HUMAN].scribe)), one, rungs[R_SHORT].feats.size(),
                mean_of(rungs[R_HUMAN].surf), mean_of(rungs[R_SHORT].surf));
  }

  // E calibration: random sets of 250 held-out human samples against the null for sets of 250. A calibrated
  // null gives a two-sided Zipf score of mean 0.50, a one-sided length score of mean 0.75, and 10% of sets under 0.1 on E1.
  {
    const SetReference& sref = set_ref_for(c, 1, 250);
    std::mt19937_64 grng(seed + 17);
    std::vector<std::uint32_t> sc2(c.dict.size(), 0);
    double m1 = 0, m3 = 0, me = 0, zs = 0, zq = 0, ks_ = 0, kq = 0;
    int low1 = 0, low3 = 0;
    const int R = 100;
    for (int r = 0; r < R; ++r) {
      std::vector<std::size_t> idx(calib_feats.size());
      std::iota(idx.begin(), idx.end(), std::size_t{0});
      std::shuffle(idx.begin(), idx.end(), grng);
      std::vector<const Features*> set;
      for (std::size_t i = 0; i < std::min<std::size_t>(250, idx.size()); ++i) set.push_back(&calib_feats[idx[i]]);
      const SetValues v = set_values(set, sref.pmf, sc2);
      double a, b;
      zs += v.zipf;
      zq += v.zipf * v.zipf;
      ks_ += v.ks;
      kq += v.ks * v.ks;
      me += sref.E(v, &a, &b);
      m1 += a;
      m3 += b;
      low1 += (a < 0.1);
      low3 += (b < 0.1);
    }
    std::printf("\n## E null check: %d random sets of 250 held-out human samples (pool of %zu), null of 300 human sets from %zu stretches of [16,48) MiB\n\n",
                R, calib_feats.size(), c.n_stretches);
    std::printf("| statistic | mean score | expected if calibrated | share of sets under 0.1 | expected |\n|---|---|---|---|---|\n");
    std::printf("| E1 Zipf slope (two-sided) | %.3f | 0.50 | %.0f%% | 10%% |\n", m1 / R, 100.0 * low1 / R);
    std::printf("| E3 sentence lengths, KS (one-sided) | %.3f | 0.75 | %.0f%% | 5%% |\n", m3 / R, 100.0 * low3 / R);
    std::printf("| E (geometric mean) | %.3f | - | - | - |\n", me / R);
    {
      auto msd = [](const Ecdf& e, double sign, double* m, double* sd) {
        double a = 0, b = 0;
        for (double x : e.v) {
          a += sign * x;
          b += x * x;
        }
        const double n = static_cast<double>(e.v.size());
        *m = a / n;
        *sd = std::sqrt(std::max(0.0, b / n - (a / n) * (a / n)));
      };
      double nzm, nzs, nkm, nks;
      msd(sref.zipf, 1.0, &nzm, &nzs);
      msd(sref.ks, -1.0, &nkm, &nks);
      std::printf("\nRaw values, mean (sd): Zipf slope, null %.3f (%.3f) against held-out sets %.3f (%.3f); KS distance, null %.4f (%.4f) against held-out sets %.4f (%.4f).\n",
                  nzm, nzs, zs / R, std::sqrt(std::max(0.0, zq / R - (zs / R) * (zs / R))), nkm, nks, ks_ / R,
                  std::sqrt(std::max(0.0, kq / R - (ks_ / R) * (ks_ / R))));
    }
    std::printf("\nThe 500-sample held-out human set of this ladder: E1 %.2f (slope %.3f), E3 %.2f (KS distance %.4f), E %.2f.\n", rungs[R_HUMAN].e1s,
                rungs[R_HUMAN].zipf, rungs[R_HUMAN].e3s, rungs[R_HUMAN].ks, rungs[R_HUMAN].E);
  }
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  std::string enwik, vec_prefix, gen_dev, gen_fresh;
  std::uint64_t fresh_seed = 31415926;
  bool only_dev = false, only_fresh = false, c_one_sided = false, soft_gate = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--selftest") return selftest();
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--vectors") vec_prefix = next();
    else if (a == "--gen-dev") gen_dev = next();
    else if (a == "--gen-fresh") gen_fresh = next();
    else if (a == "--fresh-seed") fresh_seed = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--c-one-sided") c_one_sided = true;
    else if (a == "--soft-gate") soft_gate = true;
    else if (a == "--only-dev") only_dev = true;
    else if (a == "--only-fresh") only_fresh = true;
    else if (!a.empty() && a[0] != '-' && enwik.empty()) enwik = a;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (enwik.empty()) {
    std::fprintf(stderr,
                 "usage: scribe_ladder <enwik8> [--vectors PREFIX] [--gen-dev FILE] [--gen-fresh FILE] [--fresh-seed N] [--only-dev|--only-fresh] [--c-one-sided] [--soft-gate]\n"
                 "       scribe_ladder --selftest\n"
                 "  PREFIX: ppmi_svd output (PREFIX.vectors.f32 + PREFIX.vocab.tsv), trained on enwik8 [16,48) MiB only.\n"
                 "  FILE:   GEN<TAB>kind<TAB>text lines from trie_sample (dev and fresh prompts are disjoint).\n");
    return 2;
  }
  const std::uint64_t kDevSeed = 20261027;
  std::mt19937_64 rng(kDevSeed);
  Ctx c;
  c.t0 = now_s();
  const double t0 = c.t0;

  // ---- data: enwik8 [16, 72) MiB; [16,48) trains, [48,56) is the human reference, [56,64) the development test, [64,72) the fresh test
  const std::size_t n_raw = only_dev ? 48 * MiB : 56 * MiB;
  std::vector<char> raw(n_raw);
  {
    std::ifstream in(enwik, std::ios::binary);
    if (!in) {
      std::fprintf(stderr, "cannot open %s\n", enwik.c_str());
      return 1;
    }
    in.seekg(static_cast<std::streamoff>(16 * MiB));
    in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    if (static_cast<std::size_t>(in.gcount()) != raw.size()) {
      std::fprintf(stderr, "enwik8 shorter than %zu MiB\n", 16 + n_raw / MiB);
      return 1;
    }
  }
  std::vector<std::string> train_docs = clean_wiki_documents(raw.data(), 32 * MiB);
  const std::vector<std::string> ref_docs = clean_wiki_documents(raw.data() + 32 * MiB, 8 * MiB);
  const std::vector<std::string> dev_docs = clean_wiki_documents(raw.data() + 40 * MiB, 8 * MiB);
  std::vector<std::string> fresh_docs;
  if (!only_dev) fresh_docs = clean_wiki_documents(raw.data() + 48 * MiB, 8 * MiB);
  std::vector<char>().swap(raw);
  for (const auto& d : train_docs) c.train_bytes += d.size();
  c.n_train_docs = train_docs.size();
  std::fprintf(stderr, "[%.0fs] cleaned: train %zu docs %zu bytes, ref %zu docs, dev %zu docs, fresh %zu docs\n", now_s() - t0,
               train_docs.size(), c.train_bytes, ref_docs.size(), dev_docs.size(), fresh_docs.size());

  c.scorer.profile.c_one_sided = c_one_sided;
  c.scorer.profile.gate_mode = soft_gate ? 1 : 0;
  c.model.train(train_docs);
  std::fprintf(stderr, "[%.0fs] model: vocab %zu, %zu trigrams, %llu sentences\n", now_s() - t0, c.model.vocab_size(), c.model.n_trigrams(),
               static_cast<unsigned long long>(c.model.n_train_sentences()));
  for (const auto& d : train_docs) surface_add_vocab(c.surf, d);

  // ---- C6 sentence vectors: word vectors (PPMI-SVD on the training range), SIF weights, first principal component removed
  if (!vec_prefix.empty()) {
    if (!c.wv.load(vec_prefix)) {
      std::fprintf(stderr, "cannot load word vectors %s.{vectors.f32,vocab.tsv} (300 dimensions)\n", vec_prefix.c_str());
      return 1;
    }
    c.emb = SentenceEmbedder(&c.wv, 1e-3);
    std::vector<std::vector<float>> X;
    std::mt19937_64 prng(5);
    std::vector<std::string> toks;
    for (std::size_t d = 0; d < train_docs.size() && X.size() < 60000; ++d) {
      for (const std::string& s : split_sentences(train_docs[d])) {
        if (prng() % 8 != 0) continue;
        toks.clear();
        tokenize_words(s, toks);
        std::vector<float> v;
        if (toks.size() >= 3 && c.emb.raw(toks, v)) X.push_back(std::move(v));
      }
    }
    c.emb.fit_pc(X);
    c.use_vectors = true;
    std::fprintf(stderr, "[%.0fs] word vectors: %zu words x %zu; principal component from %zu training sentences\n", now_s() - t0, c.wv.size(),
                 c.wv.dim(), X.size());
  } else {
    std::fprintf(stderr, "no --vectors: C falls back to TF-IDF cosines (C0 absent)\n");
  }

  // ---- human reference samples (the empirical CDFs) from [48,56) MiB
  const std::vector<Sample> ref_std = make_samples(ref_docs, 400, 600, 3, 4, rng);
  const std::vector<Sample> ref_short = make_samples(ref_docs, 60, 260, 1, 3, rng);
  c.n_ref_std = ref_std.size();
  c.n_ref_short = ref_short.size();
  {
    std::vector<Features> rf;
    rf.reserve(ref_std.size() + ref_short.size());
    for (const auto* v : {&ref_std, &ref_short})
      for (const Sample& s : *v) rf.push_back(extract_features(c.model, c.dict, s.text, c.embedder()));
    for (const Features& f : rf) c.scorer.add_reference(f);
    c.scorer.finish();
    for (int b = 0; b < 2; ++b) c.ref_pmf[b].assign(kMaxLen + 1, 0.0);
    {
      double cnt[2] = {0, 0};
      for (const Features& f : rf)
        for (std::uint16_t l : f.sent_len) {
          c.ref_pmf[length_bucket(f.bytes)][std::min<std::size_t>(l, kMaxLen)] += 1;
          cnt[length_bucket(f.bytes)] += 1;
        }
      for (int b = 0; b < 2; ++b)
        for (double& x : c.ref_pmf[b]) x /= std::max(1.0, cnt[b]);
    }
    double sd[2] = {0, 0}, sl[2] = {0, 0}, sn[2] = {0, 0};
    for (std::size_t i = 0; i < rf.size(); ++i) {
      const std::string& t = i < ref_std.size() ? ref_std[i].text : ref_short[i - ref_std.size()].text;
      const int b = length_bucket(t.size());
      sd[b] += scribe_detail::distinct4_bytes(t);
      sl[b] += rf[i].loop8;
      sn[b] += 1;
    }
    for (int b = 0; b < 2; ++b) {
      c.surf.ref_d4[b] = sn[b] ? sd[b] / sn[b] : 0.9;
      c.surf.ref_loop[b] = sn[b] ? sl[b] / sn[b] : 0;
    }
  }
  // ---- pools for the E null: 20 disjoint stretches of the training range [16,48) MiB. E1 and E3 use only token counts and sentence
  // lengths, never the trained model; the length pmf they are compared with comes from the reference range [48,56), not from these stretches.
  {
    std::mt19937_64 prng(555);
    std::vector<std::vector<Sample>> parts;
    auto add_parts = [&](const std::vector<std::string>& docs, int G) {
      for (int q = 0; q < G; ++q) {
        const std::vector<std::string> chunk(docs.begin() + static_cast<std::ptrdiff_t>(docs.size() * static_cast<std::size_t>(q) / static_cast<std::size_t>(G)),
                                             docs.begin() + static_cast<std::ptrdiff_t>(docs.size() * static_cast<std::size_t>(q + 1) / static_cast<std::size_t>(G)));
        std::vector<Sample> a = make_samples(chunk, 400, 600, 3, 4, prng);
        const std::vector<Sample> b = make_samples(chunk, 60, 260, 1, 3, prng);
        const std::size_t na = a.size();
        a.insert(a.end(), b.begin(), b.end());
        parts.push_back(std::move(a));
        (void)na;
      }
    };
    add_parts(train_docs, 20);  // the reference range [48,56) supplies the length pmf, so it stays out of the null
    std::size_t total = 0;
    for (const auto& pt : parts) total += pt.size();
    c.pool_feats.reserve(total);
    for (int b = 0; b < 2; ++b) c.stretches[b].resize(parts.size());
    for (std::size_t q = 0; q < parts.size(); ++q)
      for (const Sample& s : parts[q]) {
        c.pool_feats.push_back(extract_set_features(c.dict, s.text));
        c.stretches[length_bucket(c.pool_feats.back().bytes)][q].push_back(&c.pool_feats.back());
      }
    c.n_stretches = parts.size();
  }
  std::fprintf(stderr, "[%.0fs] reference features done (%zu standard + %zu short reference samples, %zu null-pool samples in %zu stretches)\n",
               now_s() - t0, ref_std.size(), ref_short.size(), c.pool_feats.size(), c.n_stretches);

  std::printf("# SCRIBE v1 control ladder\n\nConfiguration: C sub-scores %s, headline gate = %s.\n\nData: enwik8 cleaned; train [16,48) MiB = %zu docs, %zu bytes, lexicon %zu words, %zu trigrams; C6 vectors: %s; "
              "reference [48,56) MiB: %zu standard + %zu short samples; E null: sets of human samples from %zu stretches of [16,48) MiB.\n",
              c.scorer.profile.c_one_sided ? "one-sided" : "two-sided (PLAN 6.2)", c.scorer.profile.gate_mode == 1 ? "scales (raw * min(1,S/0.2) * min(1,I/0.2))" : "PLAN 6.3 caps at S and I",
              c.n_train_docs, c.train_bytes, c.model.vocab_size(), c.model.n_trigrams(),
              c.use_vectors ? (std::to_string(c.wv.size()) + " words x " + std::to_string(c.wv.dim()) + ", SIF a = 1e-3, first principal component removed").c_str() : "none (TF-IDF fallback)",
              c.n_ref_std, c.n_ref_short, c.n_stretches);

  if (!only_fresh) run_ladder(c, "Development", dev_docs, rng, kDevSeed + 1, gen_dev, kDevSeed, "[56, 64) MiB");
  if (!only_dev) {
    std::mt19937_64 frng(fresh_seed);
    run_ladder(c, "Fresh", fresh_docs, frng, fresh_seed + 1, gen_fresh, fresh_seed, "[64, 72) MiB");
  }
  std::fprintf(stderr, "[%.0fs] done\n", now_s() - t0);
  return 0;
}
