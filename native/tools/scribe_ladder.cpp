// SCRIBE v0 control ladder (docs/PLAN.md section 6.4 V1 and V4): does the metric order obvious
// controls correctly, and does the old surface score S = H*D*L?
//
// What it does (enwik8 = 2006 Wikipedia XML; all ranges are disjoint byte ranges of the file):
//   [16 MiB, 48 MiB)  cleaned with the markup cleaner, then trains the lexicon (words seen >= 2
//                     times), the Kneser-Ney word trigram and the IDF (cypha/eval/scribe.hpp)
//   [48 MiB, 56 MiB)  human REFERENCE samples: the empirical CDFs that normalise every statistic
//                     (two length buckets: short 60-260 bytes, standard 400-600 bytes)
//   [56 MiB, 64 MiB)  held-out human TEST samples (400-600 bytes, >= 3 sentences, cut at
//                     sentence boundaries); every control below is built from them
// Ladder (best to worst), 500 samples per rung except the model rung:
//   human > model > sentences shuffled across documents > words shuffled within sentences
//         > random lexicon words > random letters > one phrase repeated
// The controls keep the surface shape on purpose: shuffles keep each sentence's length, final
// punctuation and a capital first letter; "random lexicon words" and "random letters" replace
// each alphabetic word in place (frequency-weighted lexicon draws, or letters of the same
// length) and keep punctuation and capitalisation; the repeated phrase is the first six words
// of the sample as a sentence, repeated to the sample's length. So S4 and E3 cannot catch them.
// Model rung: the GEN<TAB>kind<TAB>text lines of ctx.txt (Cypha trie + context outputs),
// de-duplicated and cut after their last sentence terminator (the logger cut them at 150 bytes).
// Pass condition (PLAN 6.4): for every adjacent pair, pairwise ordering accuracy >= 0.95 over all
// (upper, lower) pairs (ties count one half) AND non-overlapping 95% bootstrap intervals
// (1000 resamples) of the rung means. Run for SCRIBE (PLAN 6.3, S gate only) and for the old
// surface S, whose reference d4 and loop8 are the human reference means of the same bucket.
// V4: Pearson and Spearman correlation of the score with sample length on human text.
//
// Measured 2026-10-27 (seed 20261027; report: docs/reports/SCRIBE_V0.md):
//   Training text 17.5 MB cleaned prose (2600 docs, lexicon 47,691 words, 2.05 M trigrams);
//   1909 standard + 1729 short reference samples; 500 held-out human samples (400-600 B).
//   Model rung: 50 GEN lines in ctx.txt, 45 unique after trimming (qa.txt has no GEN lines).
//   Mean SCRIBE [95% CI] / mean surface S per rung:
//     held-out human 0.558 [0.540,0.577] / 0.909      model (ctx GEN) 0.469 [0.369,0.566] / 0.882
//     sentences shuffled 0.308 / 0.895                 words shuffled 0.112 / 0.920
//     random lexicon words 0.120 / 0.894               random letters 0.006 / 0.034
//     one phrase repeated 0.051 / 0.029
//   Ladder (PLAN 6.4, accuracy >= 0.95 and disjoint intervals on all 6 adjacent pairs):
//     SCRIBE FAIL, 1 of 6 pairs (only random lexicon words > random letters, 1.000). Accuracies:
//       human>model 0.559, model>sentences shuffled 0.613, sentences>words shuffled 0.931,
//       words shuffled>random lexicon 0.414, random lexicon>random letters 1.000,
//       random letters>repeated phrase 0.030 (the S gate leaves a repeated phrase at 0.05 and
//       caps random letters at 0.006; the PLAN has no gate on I).
//     surface S FAIL, 1 of 6 pairs; it rates random lexicon words 0.894 and words shuffled
//       0.920 about as high as held-out human 0.909 and above the model rung 0.882.
//   Caveats: the ctx.txt GEN lines are mostly verbatim Dolly answers copied by the trie, i.e.
//   human text of another genre, so the human>model pair measures a genre gap, not a model gap.
//   The two real generated texts (qa.txt greedy and sample): SCRIBE 0.041 and 0.023 (S gate
//   fires), surface S 0.037 and 0.815.
//   V4: Pearson(SCRIBE, bytes) on human 400-600 B = 0.048 (Spearman 0.029); short 60-260 B
//   -0.109; both buckets pooled -0.221 (short human 0.659 vs standard 0.558: imperfect parity).
//   Set level E (Zipf, sentence lengths) is reported but not used for the verdict; its null is
//   too narrow (held-out human set E = 0.20).
//
// Usage: scribe_ladder <enwik8> [gen_lines_a.txt] [gen_lines_b.txt]   |   scribe_ladder --selftest
//        scribe_ladder --selftest      (known-answer checks, no corpus; CTest native_scribe_selftest)
// About 8 s and 150 MB peak on one core. Prints Markdown tables on stdout, progress on stderr.

#include "cypha/eval/scribe.hpp"

#include <chrono>
#include <cstdio>
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

// Cuts after the last sentence terminator (at or after 40 bytes); returns the text unchanged if
// there is none.
std::string trim_to_sentence_end(const std::string& t) {
  std::size_t best = std::string::npos;
  for (std::size_t i = 0; i < t.size(); ++i) {
    if ((t[i] == '.' || t[i] == '!' || t[i] == '?') && (i + 1 == t.size() || is_space_a(static_cast<unsigned char>(t[i + 1]))))
      best = i;
  }
  if (best == std::string::npos || best < 40) return t;
  return t.substr(0, best + 1);
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
  std::vector<Features> feats;
  std::vector<Scored> sc;
  std::vector<double> scribe, scribe_i, scribe_soft, surf;
  double E = 0, e1s = 0, e3s = 0, set_scribe = 0;
  double zipf = 0, e3raw = 0;
  std::pair<double, double> ci_scribe{0, 0}, ci_surf{0, 0}, ci_i{0, 0}, ci_soft{0, 0}, ci_set{0, 0};
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

std::string fmt(double v, int prec = 3) {
  char b[64];
  std::snprintf(b, sizeof b, "%.*f", prec, v);
  return b;
}

std::string fmt_ci(double m, std::pair<double, double> ci, int prec = 3) {
  return fmt(m, prec) + " [" + fmt(ci.first, prec) + ", " + fmt(ci.second, prec) + "]";
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
  std::printf("scribe selftest: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--selftest") return selftest();
  if (argc < 2) {
    std::fprintf(stderr, "usage: scribe_ladder enwik8 [ctx.txt] [qa.txt] | --selftest\n");
    return 2;
  }
  const std::string enwik = argv[1];
  // Model-output files hold "GEN<TAB>kind<TAB>text" lines (trie_context_8k / trie_qa_probe output).
  // Both are optional; a missing file just leaves the model rung smaller or empty.
  const std::string ctx_path = argc > 2 ? argv[2] : "";
  const std::string qa_path = argc > 3 ? argv[3] : "";
  const std::uint64_t kSeed = 20261027;
  std::mt19937_64 rng(kSeed);
  const double t0 = now_s();

  // ---- data ------------------------------------------------------------------------------
  std::vector<char> raw(48 * MiB);
  {
    std::ifstream in(enwik, std::ios::binary);
    if (!in) {
      std::fprintf(stderr, "cannot open %s\n", enwik.c_str());
      return 1;
    }
    in.seekg(static_cast<std::streamoff>(16 * MiB));
    in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    if (static_cast<std::size_t>(in.gcount()) != raw.size()) {
      std::fprintf(stderr, "enwik8 shorter than 64 MiB\n");
      return 1;
    }
  }
  std::vector<std::string> train_docs = clean_wiki_documents(raw.data(), 32 * MiB);
  const std::vector<std::string> ref_docs = clean_wiki_documents(raw.data() + 32 * MiB, 8 * MiB);
  const std::vector<std::string> test_docs = clean_wiki_documents(raw.data() + 40 * MiB, 8 * MiB);
  std::vector<char>().swap(raw);
  std::size_t train_bytes = 0;
  for (const auto& d : train_docs) train_bytes += d.size();
  std::fprintf(stderr, "[%.0fs] cleaned: train %zu docs %zu bytes, ref %zu docs, test %zu docs\n", now_s() - t0,
               train_docs.size(), train_bytes, ref_docs.size(), test_docs.size());

  ScribeModel model;
  model.train(train_docs);
  std::fprintf(stderr, "[%.0fs] model: vocab %zu, %zu trigrams, %llu sentences\n", now_s() - t0, model.vocab_size(),
               model.n_trigrams(), static_cast<unsigned long long>(model.n_train_sentences()));

  SurfaceRef surf;
  for (const auto& d : train_docs) surface_add_vocab(surf, d);
  const std::size_t n_train_docs = train_docs.size();

  // ---- samples ---------------------------------------------------------------------------
  const std::vector<Sample> ref_std = make_samples(ref_docs, 400, 600, 3, 4, rng);
  const std::vector<Sample> ref_short = make_samples(ref_docs, 60, 260, 1, 3, rng);
  const std::vector<Sample> test_std = pick(make_samples(test_docs, 400, 600, 3, 4, rng), 500, rng);
  const std::vector<Sample> test_short = pick(make_samples(test_docs, 60, 260, 1, 3, rng), 500, rng);
  std::fprintf(stderr, "[%.0fs] samples: ref std %zu short %zu; test std %zu short %zu\n", now_s() - t0, ref_std.size(),
               ref_short.size(), test_std.size(), test_short.size());
  if (test_std.size() < 100) {
    std::fprintf(stderr, "too few test samples\n");
    return 1;
  }

  WordDict dict;
  ScribeScorer scorer;
  std::vector<Features> ref_feats;
  std::vector<std::vector<const Features*>> slices[2];  // per bucket: reference range + 4 stretches of the training range
  {
    ref_feats.reserve(ref_std.size() + ref_short.size());
    for (const auto* v : {&ref_std, &ref_short})
      for (const Sample& s : *v) ref_feats.push_back(extract_features(model, dict, s.text));
    for (const Features& f : ref_feats) {
      scorer.add_reference(f);
    }
    scorer.finish();
    // Surface-score reference: mean distinct4 and loop8 of the human reference samples per bucket.
    double sd[2] = {0, 0}, sl[2] = {0, 0}, sn[2] = {0, 0};
    for (std::size_t i = 0; i < ref_feats.size(); ++i) {
      const std::string& t = i < ref_std.size() ? ref_std[i].text : ref_short[i - ref_std.size()].text;
      const int b = length_bucket(t.size());
      sd[b] += scribe_detail::distinct4_bytes(t);
      sl[b] += ref_feats[i].loop8;
      sn[b] += 1;
    }
    for (int b = 0; b < 2; ++b) {
      surf.ref_d4[b] = sn[b] ? sd[b] / sn[b] : 0.9;
      surf.ref_loop[b] = sn[b] ? sl[b] / sn[b] : 0;
    }
  }
  // Null for the set-level statistics: human sets from five stretches of the corpus (the reference
  // range and four equal parts of the training range). E1 and E3 use only token counts and
  // sentence lengths, never the trained model, so using training text here leaks nothing.
  std::vector<Features> slice_feats;
  {
    std::vector<std::vector<Sample>> parts;
    parts.push_back({});
    for (int q = 0; q < 4; ++q) {
      const std::vector<std::string> chunk(train_docs.begin() + static_cast<std::ptrdiff_t>(train_docs.size() * q / 4),
                                           train_docs.begin() + static_cast<std::ptrdiff_t>(train_docs.size() * (q + 1) / 4));
      std::vector<Sample> a = make_samples(chunk, 400, 600, 3, 4, rng), b = make_samples(chunk, 60, 260, 1, 3, rng);
      a.insert(a.end(), b.begin(), b.end());
      parts.push_back(std::move(a));
    }
    std::size_t total = ref_feats.size();
    for (const auto& pt : parts) total += pt.size();
    slice_feats.reserve(total);
    for (int b = 0; b < 2; ++b) slices[b].resize(5);
    for (const Features& f : ref_feats) slices[length_bucket(f.bytes)][0].push_back(&f);
    for (std::size_t q = 1; q < parts.size(); ++q)
      for (const Sample& s : parts[q]) {
        slice_feats.push_back(extract_features(model, dict, s.text));
        slices[length_bucket(slice_feats.back().bytes)][q].push_back(&slice_feats.back());
      }
  }
  std::fprintf(stderr, "[%.0fs] reference features done\n", now_s() - t0);

  // ---- ladder texts ------------------------------------------------------------------------
  std::vector<std::pair<int, std::string>> sent_pool;
  for (const Sample& s : test_std)
    for (const std::string& x : split_sentences(s.text)) sent_pool.emplace_back(s.doc, x);

  std::vector<Rung> rungs(7);
  const char* names[7] = {"held-out human", "model output (ctx.txt GEN)", "sentences shuffled across docs",
                          "words shuffled within sentences", "random lexicon words", "random letters",
                          "one phrase repeated"};
  for (int i = 0; i < 7; ++i) rungs[static_cast<std::size_t>(i)].name = names[i];
  for (const Sample& s : test_std) {
    rungs[0].texts.push_back(s.text);
    rungs[2].texts.push_back(ctl_shuffle_sentences(s, sent_pool, rng));
    rungs[3].texts.push_back(ctl_shuffle_words(s.text, rng));
    rungs[4].texts.push_back(ctl_random_lexicon(s.text, model, rng));
    rungs[5].texts.push_back(ctl_random_letters(s.text, rng));
    rungs[6].texts.push_back(ctl_repeat_phrase(s.text));
  }
  // Model rung from the GEN lines.
  std::size_t gen_lines = 0, gen_qa = 0, untrimmed = 0;
  std::vector<std::string> model_kind;
  std::vector<std::string> model_raw;  // as logged (150-byte cut), for the extra row
  {
    std::ifstream in(ctx_path);
    std::string line;
    std::unordered_set<std::string> seen;
    while (std::getline(in, line)) {
      if (line.compare(0, 4, "GEN\t") != 0) continue;
      ++gen_lines;
      const std::size_t t2 = line.find('\t', 4);
      if (t2 == std::string::npos) continue;
      const std::string kind = line.substr(4, t2 - 4);
      const std::string text = unescape(line.substr(t2 + 1));
      const std::string trimmed = trim_to_sentence_end(text);
      if (trimmed == text) ++untrimmed;
      if (!seen.insert(trimmed).second) continue;
      rungs[1].texts.push_back(trimmed);
      model_kind.push_back(kind);
      model_raw.push_back(text);
    }
    std::ifstream in2(qa_path);
    while (std::getline(in2, line))
      if (line.compare(0, 4, "GEN\t") == 0) ++gen_qa;
  }
  // qa.txt carries no GEN lines; its "greedy:" and "sample:" lines are the order-4 mixture outputs.
  std::vector<std::pair<std::string, std::string>> qa_texts;
  {
    std::ifstream in(qa_path);
    std::string line;
    std::unordered_set<std::string> seen;
    while (std::getline(in, line)) {
      for (const char* tag : {"   greedy: ", "   sample: "}) {
        if (line.compare(0, std::strlen(tag), tag) == 0) {
          const std::string t = unescape(line.substr(std::strlen(tag)));
          if (seen.insert(t).second) qa_texts.emplace_back(std::string(tag + 3, std::strlen(tag) - 5), t);
        }
      }
    }
  }
  std::fprintf(stderr, "[%.0fs] ladder built; model rung n=%zu (GEN lines %zu, qa.txt GEN lines %zu)\n", now_s() - t0,
               rungs[1].texts.size(), gen_lines, gen_qa);

  // Extra rows (not part of the ladder): untrimmed model lines, qa.txt outputs, short human.
  Rung extra_raw, extra_qa, human_short;
  extra_raw.name = "model output, as logged (150-byte cut)";
  extra_raw.texts = model_raw;
  extra_qa.name = "qa.txt greedy/sample (order-4 mixture)";
  for (const auto& p : qa_texts) extra_qa.texts.push_back(p.second);
  human_short.name = "held-out human, short bucket (60-260 B)";
  for (const Sample& s : test_short) human_short.texts.push_back(s.text);

  std::vector<Rung*> all;
  for (Rung& r : rungs) all.push_back(&r);
  all.push_back(&extra_raw);
  all.push_back(&extra_qa);
  all.push_back(&human_short);
  for (Rung* r : all) {
    r->feats.reserve(r->texts.size());
    for (const std::string& t : r->texts) r->feats.push_back(extract_features(model, dict, t));
  }
  std::fprintf(stderr, "[%.0fs] features extracted (dict %zu)\n", now_s() - t0, dict.size());

  // ---- set references and scoring ------------------------------------------------------------
  std::map<std::pair<int, std::size_t>, SetReference> set_refs;
  auto set_ref_for = [&](int bucket, std::size_t n) -> const SetReference& {
    const auto key = std::make_pair(bucket, n);
    auto it = set_refs.find(key);
    if (it == set_refs.end())
      it = set_refs.emplace(key, build_set_reference(slices[bucket], model.length_histogram(), n, dict.size(), 200, 11 + static_cast<std::uint64_t>(bucket))).first;
    return it->second;
  };
  std::vector<std::uint32_t> scratch(dict.size(), 0);
  std::uint64_t bseed = 100;
  for (Rung* r : all) {
    for (std::size_t i = 0; i < r->feats.size(); ++i) {
      const Scored s = scorer.score(r->feats[i]);
      r->sc.push_back(s);
      r->scribe.push_back(s.scribe);
      r->scribe_i.push_back(s.scribe_igate);
      r->scribe_soft.push_back(s.scribe_soft);
      r->surf.push_back(surface_S(surf, r->texts[i]));
    }
    r->ci_scribe = bootstrap_mean_ci(r->scribe, 1000, ++bseed);
    r->ci_i = bootstrap_mean_ci(r->scribe_i, 1000, ++bseed);
    r->ci_surf = bootstrap_mean_ci(r->surf, 1000, ++bseed);
    r->ci_soft = bootstrap_mean_ci(r->scribe_soft, 1000, ++bseed);
    if (r->feats.size() >= 10) {
      std::vector<std::size_t> bytes;
      for (const Features& f : r->feats) bytes.push_back(f.bytes);
      std::nth_element(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(bytes.size() / 2), bytes.end());
      const int b = length_bucket(bytes[bytes.size() / 2]);
      std::vector<const Features*> set;
      for (const Features& f : r->feats) set.push_back(&f);
      const SetReference& sref = set_ref_for(b, set.size());
      const SetValues v = set_values(set, sref.hist, scratch);
      r->zipf = v.zipf;
      r->e3raw = v.e3;
      r->E = sref.E(v, &r->e1s, &r->e3s);
      r->set_scribe = mean_of(r->scribe) * std::pow(r->E, scorer.profile.wE);
      r->ci_set = {r->ci_scribe.first * std::pow(r->E, scorer.profile.wE), r->ci_scribe.second * std::pow(r->E, scorer.profile.wE)};
    }
  }
  std::fprintf(stderr, "[%.0fs] scored\n", now_s() - t0);

  // ---- report ----------------------------------------------------------------------------------
  std::printf("# SCRIBE v0 control ladder (seed %llu)\n\n", static_cast<unsigned long long>(kSeed));
  std::printf("Data: enwik8 cleaned; train [16,48) MiB = %zu docs, %zu bytes, vocab %zu, %zu trigrams; reference [48,56) MiB: %zu standard + %zu short samples; "
              "test [56,64) MiB: %zu standard samples. Set-level null: human sets from five stretches of the corpus.\n",
              n_train_docs, train_bytes, model.vocab_size(), model.n_trigrams(), ref_std.size(),
              ref_short.size(), test_std.size());
  std::printf("Model rung: %zu GEN lines in ctx.txt (%zu without a sentence terminator to cut at), %zu unique after trimming; qa.txt has %zu GEN lines.\n\n",
              gen_lines, untrimmed, rungs[1].texts.size(), gen_qa);

  std::printf("## Rung means (95%% bootstrap interval of the mean, 1000 resamples)\n\n");
  std::printf("| rung | n | SCRIBE | parity | surface S | mean bytes | E | set-level SCRIBE |\n|---|---|---|---|---|---|---|---|\n");
  const double human_mean = mean_of(rungs[0].scribe);
  auto row = [&](const Rung& r) {
    double by = 0;
    for (const Features& f : r.feats) by += static_cast<double>(f.bytes);
    by /= std::max<std::size_t>(1, r.feats.size());
    std::printf("| %s | %zu | %s | %.2f | %s | %.0f | %s | %s |\n", r.name.c_str(), r.feats.size(),
                fmt_ci(mean_of(r.scribe), r.ci_scribe).c_str(), mean_of(r.scribe) / human_mean, fmt_ci(mean_of(r.surf), r.ci_surf).c_str(), by,
                r.feats.size() >= 10 ? fmt(r.E, 2).c_str() : "-",
                r.feats.size() >= 10 ? fmt_ci(r.set_scribe, r.ci_set).c_str() : "-");
  };
  for (const Rung& r : rungs) row(r);
  std::printf("\nExtra rows (not on the ladder):\n\n| rung | n | SCRIBE | parity | surface S | mean bytes | E | set-level SCRIBE |\n|---|---|---|---|---|---|---|---|\n");
  row(extra_raw);
  row(extra_qa);
  row(human_short);

  struct PairRes {
    double acc;
    bool disjoint, pass;
  };
  auto pair_res = [&](const Rung& hi, const Rung& lo, const std::vector<double> Rung::*vals,
                      const std::pair<double, double> Rung::*ci) {
    PairRes r;
    r.acc = pairwise_accuracy(hi.*vals, lo.*vals);
    r.disjoint = (hi.*ci).first > (lo.*ci).second;
    r.pass = r.acc >= 0.95 && r.disjoint;
    return r;
  };
  // Pairs: the six adjacent ones, plus human > sentences shuffled with the model rung skipped.
  const std::vector<std::pair<int, int>> pairs = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {0, 2}};
  std::printf("\n## V1 ladder: pairwise ordering accuracy (all upper/lower pairs, ties = 1/2) and disjoint intervals\n\n");
  std::printf("| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |\n|---|---|---|---|---|---|---|\n");
  int pass_scribe = 0, pass_surf = 0;
  for (const auto& pr : pairs) {
    const Rung &a = rungs[static_cast<std::size_t>(pr.first)], &b = rungs[static_cast<std::size_t>(pr.second)];
    const PairRes x = pair_res(a, b, &Rung::scribe, &Rung::ci_scribe), y = pair_res(a, b, &Rung::surf, &Rung::ci_surf);
    if (pr.second == pr.first + 1) {
      pass_scribe += x.pass;
      pass_surf += y.pass;
    }
    std::printf("| %s > %s%s | %.3f | %s | %s | %.3f | %s | %s |\n", a.name.c_str(), b.name.c_str(),
                pr.second == pr.first + 1 ? "" : " (model rung skipped)", x.acc, x.disjoint ? "yes" : "no", x.pass ? "PASS" : "FAIL",
                y.acc, y.disjoint ? "yes" : "no", y.pass ? "PASS" : "FAIL");
  }
  std::printf("\nLadder verdict (6 adjacent pairs must all pass): SCRIBE %s (%d of 6), surface S %s (%d of 6)\n",
              pass_scribe == 6 ? "PASS" : "FAIL", pass_scribe, pass_surf == 6 ? "PASS" : "FAIL", pass_surf);

  std::printf("\n## Proposals tried after the first run (not in the PLAN; tuned on this ladder, so not evidence)\n\n");
  std::printf("A: PLAN gate plus a cap at I when I < 0.2. B: SCRIBE_raw * min(1, S/0.2) * min(1, I/0.2) (scales instead of capping).\n\n");
  std::printf("| upper > lower | A acc | | B acc | |\n|---|---|---|---|---|\n");
  int pass_a = 0, pass_b = 0;
  for (const auto& pr : pairs) {
    const Rung &a = rungs[static_cast<std::size_t>(pr.first)], &b = rungs[static_cast<std::size_t>(pr.second)];
    const PairRes x = pair_res(a, b, &Rung::scribe_i, &Rung::ci_i), y = pair_res(a, b, &Rung::scribe_soft, &Rung::ci_soft);
    if (pr.second == pr.first + 1) {
      pass_a += x.pass;
      pass_b += y.pass;
    }
    std::printf("| %s > %s%s | %.3f | %s | %.3f | %s |\n", a.name.c_str(), b.name.c_str(),
                pr.second == pr.first + 1 ? "" : " (model rung skipped)", x.acc, x.pass ? "PASS" : "FAIL", y.acc,
                y.pass ? "PASS" : "FAIL");
  }
  std::printf("\nA passes %d of 6 adjacent pairs, B passes %d of 6.\n", pass_a, pass_b);
  {
    std::printf("\nMeans: ");
    for (const Rung& r : rungs) std::printf("%s A %.3f B %.3f; ", r.name.c_str(), mean_of(r.scribe_i), mean_of(r.scribe_soft));
    std::printf("\n");
  }

  std::printf("\n## Axis profile (mean over samples; C over samples with >= 2 sentences)\n\n");
  std::printf("| rung | S | C (n) | I | S gate fired | E1 (Zipf) | E3 (lengths) | E |\n|---|---|---|---|---|---|---|---|\n");
  auto prow = [&](const Rung& r) {
    int nc = 0, gate = 0;
    for (const Scored& s : r.sc) {
      nc += s.has_c;
      gate += s.s_gate;
    }
    std::printf("| %s | %s | %s (%d) | %s | %.0f%% | %s | %s | %s |\n", r.name.c_str(), fmt(mean_field(r, &Scored::S), 2).c_str(),
                fmt(mean_field(r, &Scored::C, true), 2).c_str(), nc, fmt(mean_field(r, &Scored::I), 2).c_str(),
                100.0 * gate / std::max<std::size_t>(1, r.sc.size()), r.feats.size() >= 10 ? fmt(r.e1s, 2).c_str() : "-",
                r.feats.size() >= 10 ? fmt(r.e3s, 2).c_str() : "-", r.feats.size() >= 10 ? fmt(r.E, 2).c_str() : "-");
  };
  for (const Rung& r : rungs) prow(r);
  prow(extra_raw);
  prow(human_short);

  std::printf("\n## Sub-scores (mean; C-side over samples with >= 2 sentences)\n\n");
  std::printf("| rung | S1 | S3 | S4 | C1 | C3 | I2 | I3 |\n|---|---|---|---|---|---|---|---|\n");
  auto srow = [&](const Rung& r) {
    std::printf("| %s | %s | %s | %s | %s | %s | %s | %s |\n", r.name.c_str(), fmt(mean_field(r, &Scored::s1), 2).c_str(),
                fmt(mean_field(r, &Scored::s3), 2).c_str(), fmt(mean_field(r, &Scored::s4), 2).c_str(),
                fmt(mean_field(r, &Scored::c1, true), 2).c_str(), fmt(mean_field(r, &Scored::c3, true), 2).c_str(),
                fmt(mean_field(r, &Scored::i2), 2).c_str(), fmt(mean_field(r, &Scored::i3), 2).c_str());
  };
  for (const Rung& r : rungs) srow(r);
  srow(extra_raw);
  srow(human_short);

  std::printf("\n## Raw statistics on held-out human text (mean) and E raw values\n\n");
  {
    double s1 = 0, s3 = 0, s4 = 0, c1 = 0, c3 = 0, l8 = 0, rp = 0, d1 = 0, d4 = 0;
    int nc = 0;
    for (const Features& f : rungs[0].feats) {
      s1 += f.s1;
      s3 += f.s3;
      s4 += f.s4;
      l8 += f.loop8;
      rp += f.rep;
      d1 += f.d[0];
      d4 += f.d[3];
      if (f.has_c) {
        c1 += f.c1;
        c3 += f.c3;
        ++nc;
      }
    }
    const double n = static_cast<double>(rungs[0].feats.size());
    std::printf("S1 %.3f, S3 %.3f nats/token, S4 %.3f, C1 %.3f, C3 %.3f bits, loop8 %.3f, repeated-sentence rate %.3f, distinct-1 %.3f, distinct-4 %.3f; "
                "Zipf slope of the 500-sample set %.3f, E3 raw %.3f\n",
                s1 / n, s3 / n, s4 / n, nc ? c1 / nc : 0.0, nc ? c3 / nc : 0.0, l8 / n, rp / n, d1 / n, d4 / n, rungs[0].zipf, rungs[0].e3raw);
  }

  std::printf("\n## Model rung by kind (SCRIBE mean, surface S mean)\n\n| kind | n | SCRIBE | surface S |\n|---|---|---|---|\n");
  {
    std::map<std::string, std::vector<std::size_t>> by;
    for (std::size_t i = 0; i < model_kind.size(); ++i) by[model_kind[i]].push_back(i);
    for (const auto& kv : by) {
      double a = 0, b = 0;
      for (std::size_t i : kv.second) {
        a += rungs[1].scribe[i];
        b += rungs[1].surf[i];
      }
      std::printf("| %s | %zu | %.3f | %.3f |\n", kv.first.c_str(), kv.second.size(), a / kv.second.size(), b / kv.second.size());
    }
  }
  std::printf("\nqa.txt outputs scored individually:\n\n| source | SCRIBE | S gate | surface S | text |\n|---|---|---|---|---|\n");
  for (std::size_t i = 0; i < extra_qa.texts.size(); ++i)
    std::printf("| %s | %.3f | %s | %.3f | %.60s |\n", qa_texts[i].first.c_str(), extra_qa.scribe[i], extra_qa.sc[i].s_gate ? "yes" : "no",
                extra_qa.surf[i], extra_qa.texts[i].c_str());

  std::printf("\n## V4 stability: correlation of the score with sample length on held-out human text\n\n");
  std::printf("| set | n | metric | Pearson vs bytes | Spearman vs bytes | Pearson vs words |\n|---|---|---|---|---|---|\n");
  auto corr_row = [&](const char* nm, const Rung& r) {
    std::vector<double> by, wd;
    for (const Features& f : r.feats) {
      by.push_back(static_cast<double>(f.bytes));
      wd.push_back(static_cast<double>(f.n_words));
    }
    std::printf("| %s | %zu | SCRIBE | %.3f | %.3f | %.3f |\n", nm, r.feats.size(), pearson(r.scribe, by), spearman(r.scribe, by),
                pearson(r.scribe, wd));
    std::printf("| %s | %zu | surface S | %.3f | %.3f | %.3f |\n", nm, r.feats.size(), pearson(r.surf, by), spearman(r.surf, by),
                pearson(r.surf, wd));
  };
  corr_row("standard 400-600 B", rungs[0]);
  corr_row("short 60-260 B", human_short);
  {
    std::vector<double> sc_all = rungs[0].scribe, by_all;
    for (const Features& f : rungs[0].feats) by_all.push_back(static_cast<double>(f.bytes));
    for (std::size_t i = 0; i < human_short.feats.size(); ++i) {
      sc_all.push_back(human_short.scribe[i]);
      by_all.push_back(static_cast<double>(human_short.feats[i].bytes));
    }
    std::printf("| both buckets pooled | %zu | SCRIBE | %.3f | %.3f | - |\n", sc_all.size(), pearson(sc_all, by_all), spearman(sc_all, by_all));
    std::vector<double> su_all = rungs[0].surf;
    for (double v : human_short.surf) su_all.push_back(v);
    std::printf("| both buckets pooled | %zu | surface S | %.3f | %.3f | - |\n", su_all.size(), pearson(su_all, by_all), spearman(su_all, by_all));
  }
  std::printf("\nMean SCRIBE of held-out human: standard %.3f, short %.3f (bucket parity). Mean surface S: standard %.3f, short %.3f.\n",
              mean_of(rungs[0].scribe), mean_of(human_short.scribe), mean_of(rungs[0].surf), mean_of(human_short.surf));
  std::fprintf(stderr, "[%.0fs] done\n", now_s() - t0);
  return 0;
}
