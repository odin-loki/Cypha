// SCRIBE v1: the generated-text quality metric (docs/PLAN.md section 6).
// (v2 and v3 are further down: ScribeV2Scorer, and SurpriseModel / PackIndex / V3Scorer, docs/reports/SCRIBE_V3.md.)
//
// Header-only, standard library only, no neural network. Scope is the OPEN-CONTINUATION profile with
// axes S (syntax), C (coherence), I (integrity) and E (expression, set level). R (relevance) and B (belief)
// need question and fact machinery that does not exist yet. Results: docs/reports/SCRIBE_V1.md.
//
// Sub-scores (all raw statistics are normalised by the empirical CDF of human reference
// samples of the same length bucket, PLAN 6.1; one-sided s = min(1, 2F), two-sided
// s = 1 - |2F - 1|, with a mid-rank CDF smoothed by +0.5/(N+1) so ties and tails stay finite):
//   S1 lexical validity   fraction of word tokens in the lexicon (seen >= 2 times in the
//                         reference-training text) or numeric                       one-sided
//   S3 word fluency       mean log-probability per predicted token (words and </s>) under an
//                         interpolated Kneser-Ney word trigram; every out-of-vocabulary word
//                         pays a flat extra 8 nats                                    one-sided
//   S4 well-formedness    pass rate of six checks: valid UTF-8, sentence-initial capital (fraction
//                         of sentences), terminal punctuation at the end, balanced brackets, even
//                         number of double quotes, no mid-word cut at the end
//   C0 word cohesion      mean cosine over all pairs of distinct content words (idf >= 3 and known
//                         to the word vectors; >= 4 of them)                          two-sided
//   C1 sentence cohesion  mean cosine over ALL pairs of sentence vectors (C6: PPMI-SVD word vectors,
//                         SIF weights a/(a+p(w)) with a = 1e-3, first principal component removed).
//                         A sample with fewer than two embeddable sentences is cut into two halves of
//                         its tokens (>= 8 tokens), so short samples have the axis    two-sided
//   C3 topic drift        Jensen-Shannon divergence (bits) of the content-word distribution of the first
//                         half against the second half (>= 4 content words)           two-sided
//                         The C axis is the geometric mean of the C sub-scores that exist.
//   I2 self-repetition    geometric mean of the scores of loop8 and of the repeated-sentence
//                         rate (lower is better)                                      one-sided
//   I3 diversity          geometric mean of distinct-1..4 (word n-grams)              two-sided
//   E1 Zipf slope         rank-frequency slope of a SET of samples against human sets of
//                         the same size and bucket                                    two-sided
//   E3 sentence lengths   Kolmogorov distance between the set's sentence-length distribution and the
//                         human one, against the same distance for human sets         one-sided
//
// Combining (6.3): axis = geometric mean of its sub-scores; per-sample SCRIBE = weighted geometric mean
// of S, C, I with eps = 0.01 (open continuation weights S 2, C 2, I 1, E 1). Gates: S < 0.2 caps the sample
// at S, and (new in v1) I < 0.2 caps it at I. The set-level value is the mean of SCRIBE_x times E^w_E.
// ScribeProfile also carries two variants that were tried and not adopted (one-sided C scores; scaling
// instead of capping); see the report.
//
// E null (build_set_reference): human sets drawn from a third of ~20 stretches of the human corpus, so
// that it carries between-stretch variation; the length distribution they are compared with comes from
// text outside those stretches.
//
// Also here: a Wikipedia markup cleaner (human text), clean_generated_text (model continuations), a
// sentence splitter, a word tokenizer, the old surface score S = H*D*L (ported from
// native/tools/eval/surface_score.cpp), and the small statistics helpers the ladder needs (bootstrap
// interval, pairwise ordering accuracy, Pearson and Spearman correlation).
//
// Independence (PLAN V5): the lexicon, the trigram, the IDF and the word vectors come from text the caller
// passes in; the ladder uses text disjoint from everything it scores and from the trie's training text.
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cypha::eval {

// ===========================================================================
// Text utilities: UTF-8, markup cleaner, sentence splitter, tokenizer
// ===========================================================================
namespace scribe_detail {

inline bool is_alnum_a(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
inline bool is_digit_a(unsigned char c) { return c >= '0' && c <= '9'; }
inline bool is_upper_a(unsigned char c) { return c >= 'A' && c <= 'Z'; }
inline bool is_alpha_a(unsigned char c) { return (c >= 'a' && c <= 'z') || is_upper_a(c); }
inline bool is_space_a(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
inline unsigned char lower_a(unsigned char c) { return is_upper_a(c) ? static_cast<unsigned char>(c + 32) : c; }

inline std::string lower_ascii(std::string s) {
  for (char& c : s) c = static_cast<char>(lower_a(static_cast<unsigned char>(c)));
  return s;
}

inline void append_utf8(std::string& o, unsigned cp) {
  if (cp < 0x80) {
    o.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    o.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    o.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x110000) {
    o.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Length of the UTF-8 sequence starting at s[i], or 0 if it is invalid. Rejects overlong forms,
// surrogates and code points above U+10FFFF.
inline size_t utf8_seq_len(const std::string& s, size_t i, unsigned* cp_out = nullptr) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  size_t L = 0;
  unsigned cp = 0;
  if (c < 0x80) {
    if (cp_out) *cp_out = c;
    return 1;
  } else if (c >= 0xC2 && c <= 0xDF) {
    L = 2;
    cp = c & 0x1F;
  } else if (c >= 0xE0 && c <= 0xEF) {
    L = 3;
    cp = c & 0x0F;
  } else if (c >= 0xF0 && c <= 0xF4) {
    L = 4;
    cp = c & 0x07;
  } else {
    return 0;
  }
  if (i + L > s.size()) return 0;
  for (size_t k = 1; k < L; ++k) {
    const unsigned char d = static_cast<unsigned char>(s[i + k]);
    if ((d & 0xC0) != 0x80) return 0;
    cp = (cp << 6) | (d & 0x3F);
  }
  if ((L == 3 && cp < 0x800) || (L == 4 && cp < 0x10000) || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
  if (cp_out) *cp_out = cp;
  return L;
}

inline bool utf8_valid(const std::string& s) {
  for (size_t i = 0; i < s.size();) {
    const size_t L = utf8_seq_len(s, i);
    if (L == 0) return false;
    i += L;
  }
  return true;
}

// Byte length of the letter-like character at s[i] (ASCII alphanumeric, or a non-ASCII letter),
// or 0 when s[i] starts punctuation, a symbol, whitespace or an invalid sequence.
inline size_t word_char_len(const std::string& s, size_t i) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) return is_alnum_a(c) ? 1 : 0;
  unsigned cp = 0;
  const size_t L = utf8_seq_len(s, i, &cp);
  if (L == 0) return 0;
  const bool letter = (cp >= 0xC0 && cp <= 0x24F && cp != 0xD7 && cp != 0xF7) || cp == 0xAA || cp == 0xB5 || cp == 0xBA ||
                      (cp >= 0x370 && cp <= 0x52F) || (cp >= 0x1E00 && cp <= 0x1FFF) || (cp >= 0x3040 && cp <= 0x9FFF) ||
                      (cp >= 0xAC00 && cp <= 0xD7AF);
  return letter ? L : 0;
}

// ---------------------------------------------------------------------------
// Wikipedia markup cleaner
// ---------------------------------------------------------------------------

inline std::string decode_entities(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '&') {
      const size_t j = s.find(';', i + 1);
      if (j != std::string::npos && j - i <= 10 && j > i + 1) {
        const std::string name = s.substr(i + 1, j - i - 1);
        bool consume = true;  // false: leave the '&' as text
        if (name == "amp") o.push_back('&');
        else if (name == "lt") o.push_back('<');
        else if (name == "gt") o.push_back('>');
        else if (name == "quot") o.push_back('"');
        else if (name == "apos") o.push_back('\'');
        else if (name == "nbsp") o.push_back(' ');
        else if (name == "ndash") o += "\xE2\x80\x93";
        else if (name == "mdash") o += "\xE2\x80\x94";
        else if (name == "minus") o.push_back('-');
        else if (name == "hellip") o += "...";
        else if (name[0] == '#') {
          unsigned cp = 0;
          bool okn = name.size() > 1;
          if (okn && (name[1] == 'x' || name[1] == 'X')) {
            for (size_t k = 2; k < name.size(); ++k) {
              const char h = name[k];
              if (h >= '0' && h <= '9') cp = cp * 16 + static_cast<unsigned>(h - '0');
              else if (h >= 'a' && h <= 'f') cp = cp * 16 + static_cast<unsigned>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') cp = cp * 16 + static_cast<unsigned>(h - 'A' + 10);
              else okn = false;
            }
          } else {
            for (size_t k = 1; k < name.size(); ++k) {
              if (is_digit_a(static_cast<unsigned char>(name[k]))) cp = cp * 10 + static_cast<unsigned>(name[k] - '0');
              else okn = false;
            }
          }
          if (!okn) consume = false;
          else if (cp >= 32 && cp < 0x110000) append_utf8(o, cp);
        } else {
          for (char ch : name)
            if (!is_alnum_a(static_cast<unsigned char>(ch))) consume = false;
          // unknown alphanumeric named entities (&eacute; ...) are dropped
        }
        if (consume) {
          i = j + 1;
          continue;
        }
      }
    }
    o.push_back(s[i++]);
  }
  return o;
}

inline bool block_tag(const std::string& name) {
  return name == "ref" || name == "math" || name == "gallery" || name == "timeline" || name == "source" ||
         name == "pre" || name == "imagemap" || name == "noinclude" || name == "score";
}

// Removes comments, HTML tags, and the whole body of <ref>, <math>, <gallery> and similar blocks.
inline std::string strip_tags(const std::string& t) {
  const std::string low = lower_ascii(t);
  const size_t n = t.size();
  std::string o;
  o.reserve(n);
  for (size_t i = 0; i < n;) {
    if (t[i] != '<') {
      o.push_back(t[i++]);
      continue;
    }
    if (low.compare(i, 4, "<!--") == 0) {
      const size_t e = low.find("-->", i + 4);
      if (e == std::string::npos) break;
      i = e + 3;
      continue;
    }
    size_t j = i + 1;
    const bool closing = j < n && t[j] == '/';
    if (closing) ++j;
    size_t k = j;
    while (k < n && is_alpha_a(static_cast<unsigned char>(low[k]))) ++k;
    const size_t gt = (k > j && k - j <= 12) ? low.find('>', k) : std::string::npos;
    if (gt == std::string::npos || gt - i > 600) {
      o.push_back(t[i++]);
      continue;
    }
    const std::string name = low.substr(j, k - j);
    const bool self_close = low[gt - 1] == '/';
    if (!closing && !self_close && block_tag(name)) {
      const size_t e = low.find("</" + name, gt);
      if (e == std::string::npos) {
        i = gt + 1;
      } else {
        const size_t e2 = low.find('>', e);
        i = (e2 == std::string::npos) ? n : e2 + 1;
      }
      continue;
    }
    if (name == "br") o.push_back(' ');
    i = gt + 1;
  }
  return o;
}

// Index just past the closer matching the opener at s[i..i+len), counting nesting; npos if none
// within 60000 bytes.
inline size_t match_close(const std::string& s, size_t i, const char* open, const char* close) {
  const size_t lo = std::strlen(open), lc = std::strlen(close);
  int depth = 0;
  const size_t lim = std::min(s.size(), i + 60000);
  for (size_t p = i; p < lim;) {
    if (s.compare(p, lo, open) == 0) {
      ++depth;
      p += lo;
    } else if (s.compare(p, lc, close) == 0) {
      --depth;
      p += lc;
      if (depth == 0) return p;
    } else {
      ++p;
    }
  }
  return std::string::npos;
}

// Drops {| ... |} tables (line based) and {{ ... }} templates (nested).
inline std::string strip_tables_and_templates(const std::string& t) {
  std::string a;
  a.reserve(t.size());
  int depth = 0;
  size_t pos = 0;
  while (pos < t.size()) {
    size_t e = t.find('\n', pos);
    if (e == std::string::npos) e = t.size();
    size_t b = pos;
    while (b < e && (t[b] == ' ' || t[b] == '\t')) ++b;
    const bool open = t.compare(b, 2, "{|") == 0;
    const bool close = t.compare(b, 2, "|}") == 0;
    if (open) ++depth;
    if (depth == 0) {
      a.append(t, pos, e - pos);
      a.push_back('\n');
    }
    if (close && depth > 0) --depth;
    pos = e + 1;
  }
  std::string o;
  o.reserve(a.size());
  for (size_t i = 0; i < a.size();) {
    if (a.compare(i, 2, "{{") == 0) {
      const size_t e = match_close(a, i, "{{", "}}");
      if (e != std::string::npos) {
        i = e;
        continue;
      }
    }
    o.push_back(a[i++]);
  }
  return o;
}

inline bool lang_prefix(const std::string& p) {
  if (p.size() < 2 || p.size() > 12) return false;
  if (p.size() > 3 && p.find('-') == std::string::npos) return false;
  for (char c : p)
    if (!(c >= 'a' && c <= 'z') && c != '-') return false;
  return true;
}

// [[a|b]] -> b, [[a]] -> a; Image:/File:/Category:/Media: and interlanguage links are dropped.
inline std::string strip_wiki_links(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    if (s.compare(i, 2, "[[") == 0) {
      const size_t e = match_close(s, i, "[[", "]]");
      if (e == std::string::npos) {
        o.push_back(s[i++]);
        continue;
      }
      std::string inner = s.substr(i + 2, e - i - 4);
      i = e;
      if (!inner.empty() && inner[0] == ':') inner.erase(0, 1);
      const size_t colon = inner.find(':');
      if (colon != std::string::npos && colon > 0 && colon <= 12) {
        const std::string pre = lower_ascii(inner.substr(0, colon));
        const size_t bar0 = inner.find('|');
        if ((bar0 == std::string::npos || colon < bar0) &&
            (pre == "image" || pre == "file" || pre == "category" || pre == "media" || lang_prefix(pre)))
          continue;
      }
      int depth = 0;
      size_t last_bar = std::string::npos;
      for (size_t p = 0; p < inner.size(); ++p) {
        if (inner.compare(p, 2, "[[") == 0) {
          ++depth;
          ++p;
        } else if (inner.compare(p, 2, "]]") == 0) {
          --depth;
          ++p;
        } else if (inner[p] == '|' && depth == 0) {
          last_bar = p;
        }
      }
      std::string disp = (last_bar == std::string::npos) ? inner : inner.substr(last_bar + 1);
      if (disp.empty() && last_bar != std::string::npos) disp = inner.substr(0, last_bar);
      o += strip_wiki_links(disp);
    } else {
      o.push_back(s[i++]);
    }
  }
  return o;
}

// [http://x text] -> text, [http://x] -> nothing.
inline std::string strip_external_links(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '[' && (s.compare(i + 1, 7, "http://") == 0 || s.compare(i + 1, 8, "https://") == 0 ||
                        s.compare(i + 1, 6, "ftp://") == 0 || s.compare(i + 1, 7, "mailto:") == 0)) {
      const size_t e = s.find(']', i);
      if (e != std::string::npos && e - i < 1500) {
        const std::string inner = s.substr(i + 1, e - i - 1);
        const size_t sp = inner.find(' ');
        if (sp != std::string::npos) o += inner.substr(sp + 1);
        i = e + 1;
        continue;
      }
    }
    o.push_back(s[i++]);
  }
  return o;
}

inline std::string collapse_ws(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  bool sp = true;
  for (char c : s) {
    if (c == ' ' || c == '\t' || c == '\r') {
      if (!sp) o.push_back(' ');
      sp = true;
    } else {
      o.push_back(c);
      sp = false;
    }
  }
  while (!o.empty() && o.back() == ' ') o.pop_back();
  return o;
}

// A kept line is a prose paragraph: long enough, starts like a sentence, ends like one, and has
// no leftover markup.
inline bool prose_line(const std::string& l) {
  if (l.size() < 60) return false;
  const unsigned char f = static_cast<unsigned char>(l[0]);
  if (!(is_upper_a(f) || is_digit_a(f) || f == '"' || f == '(' || f >= 0xC0)) return false;
  size_t e = l.size();
  while (e > 0 && (l[e - 1] == '"' || l[e - 1] == ')' || l[e - 1] == '\'')) --e;
  if (e == 0 || !(l[e - 1] == '.' || l[e - 1] == '!' || l[e - 1] == '?')) return false;
  for (char c : l)
    if (c == '{' || c == '}' || c == '[' || c == ']' || c == '|' || c == '<' || c == '>' || c == '=' || c == '*' ||
        c == '#' || c == '_' || c == '\\' || c == '~')
      return false;
  if (l.find("http") != std::string::npos || l.find("www.") != std::string::npos) return false;
  return true;
}

}  // namespace scribe_detail

// The markup-stripping steps shared by the human-text cleaner and the model-output cleaner:
// entities, tags, tables, templates, links, bold and italic quotes.
inline std::string strip_wiki_markup(const std::string& body) {
  using namespace scribe_detail;
  std::string t = decode_entities(body);
  t = strip_tags(t);
  t = strip_tables_and_templates(t);
  t = strip_wiki_links(t);
  t = strip_external_links(t);
  t = decode_entities(t);  // second pass: "&amp;nbsp;" in 2006 dumps
  std::string o;
  o.reserve(t.size());
  for (size_t i = 0; i < t.size();) {  // '' and ''' -> nothing
    if (t[i] == '\'' && t.compare(i, 2, "''") == 0) {
      while (i < t.size() && t[i] == '\'') ++i;
    } else {
      o.push_back(t[i++]);
    }
  }
  return o;
}

// Cleans one page body (the text between <text ...> and </text>, still XML-escaped) to prose.
inline std::string clean_wiki_body(const std::string& body) {
  using namespace scribe_detail;
  const std::string t = strip_wiki_markup(body);
  std::string doc;
  size_t pos = 0;
  while (pos < t.size()) {
    size_t e = t.find('\n', pos);
    if (e == std::string::npos) e = t.size();
    const std::string line = collapse_ws(t.substr(pos, e - pos));
    if (!(pos < t.size() && (t[pos] == ' ')) && prose_line(line)) {
      if (!doc.empty()) doc.push_back(' ');
      doc += line;
    }
    pos = e + 1;
  }
  return doc;
}

// Cleans every complete <page> in [data, data+n) and returns one prose string per page. A page
// cut by either end of the range is skipped.
inline std::vector<std::string> clean_wiki_documents(const char* data, size_t n) {
  std::vector<std::string> docs;
  const std::string_view sv(data, n);
  size_t pos = 0;
  while (true) {
    const size_t p = sv.find("<page>", pos);
    if (p == std::string_view::npos) break;
    const size_t e = sv.find("</page>", p);
    if (e == std::string_view::npos) break;
    pos = e + 7;
    const std::string_view page = sv.substr(p, e - p);
    const size_t t0 = page.find("<text");
    if (t0 == std::string_view::npos) continue;
    const size_t gt = page.find('>', t0);
    if (gt == std::string_view::npos || page[gt - 1] == '/') continue;
    const size_t t1 = page.find("</text>", gt);
    if (t1 == std::string_view::npos) continue;
    const std::string body(page.substr(gt + 1, t1 - gt - 1));
    size_t b = 0;
    while (b < body.size() && scribe_detail::is_space_a(static_cast<unsigned char>(body[b]))) ++b;
    if (scribe_detail::lower_ascii(body.substr(b, 9)) == "#redirect") continue;
    std::string d = clean_wiki_body(body);
    if (!d.empty()) docs.push_back(std::move(d));
  }
  return docs;
}

// Sentence splitter. A boundary is . ! or ? (with trailing closing quotes/brackets) followed by
// whitespace or the end, except after an initial, a known abbreviation, or an inner-dot token
// such as "U.S." followed by a lowercase or digit start. A lowercase start after an ordinary
// full stop IS a boundary, so that S4 can see the missing capital.
inline std::vector<std::string> split_sentences(const std::string& text) {
  using namespace scribe_detail;
  static const std::unordered_set<std::string> abbr = {
      "mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "inc", "co", "ltd", "corp", "mt", "ft", "gen", "col", "lt",
      "capt", "sgt", "rev", "gov", "sen", "rep", "prof", "ca", "approx", "ed", "vol", "pp", "est", "fig", "no", "jan",
      "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "cf", "al", "ie", "eg"};
  std::vector<std::string> out;
  const size_t n = text.size();
  size_t start = 0;
  auto flush = [&](size_t endi) {
    size_t b = start, e = endi;
    while (b < e && is_space_a(static_cast<unsigned char>(text[b]))) ++b;
    while (e > b && is_space_a(static_cast<unsigned char>(text[e - 1]))) --e;
    if (e > b) out.emplace_back(text, b, e - b);
  };
  for (size_t i = 0; i < n; ++i) {
    const char c = text[i];
    if (c != '.' && c != '!' && c != '?') continue;
    size_t j = i + 1;
    while (j < n && (text[j] == '.' || text[j] == '!' || text[j] == '?')) ++j;
    size_t k = j;
    while (k < n) {
      if (text[k] == '"' || text[k] == '\'' || text[k] == ')' || text[k] == ']') ++k;
      else if (text.compare(k, 3, "\xE2\x80\x9D") == 0 || text.compare(k, 3, "\xE2\x80\x99") == 0) k += 3;
      else break;
    }
    if (!(k >= n || is_space_a(static_cast<unsigned char>(text[k])))) {
      i = j - 1;
      continue;
    }
    if (c == '.' && j == i + 1) {
      size_t b = i;
      while (b > 0 && (is_alnum_a(static_cast<unsigned char>(text[b - 1])) || text[b - 1] == '.')) --b;
      const std::string tok = lower_ascii(text.substr(b, i - b));
      size_t nx = k;
      while (nx < n && is_space_a(static_cast<unsigned char>(text[nx]))) ++nx;
      const unsigned char nc = nx < n ? static_cast<unsigned char>(text[nx]) : 0;
      if (tok.size() == 1 && is_alpha_a(static_cast<unsigned char>(tok[0]))) {
        i = j - 1;
        continue;
      }
      if (abbr.count(tok)) {
        i = j - 1;
        continue;
      }
      if (tok.find('.') != std::string::npos && !(is_upper_a(nc) && tok.size() > 4)) {
        i = j - 1;
        continue;
      }
    }
    flush(k);
    start = k;
    i = k > 0 ? k - 1 : 0;
  }
  flush(n);
  return out;
}

// Prepares a CONTINUATION generated by a model that was trained on raw wiki text, so that it can be
// scored next to cleaned human prose. Wiki markup is stripped with the same steps as for human text;
// then leftover markup characters ({ } [ ] | < > = * # _ \ ~) become spaces and every run of white
// space (newlines included) one space. A continuation starts where the prompt was cut and stops where
// the length limit cut it, so: a first word that starts without a space (the prompt cut it) and a
// last word not followed by a sentence terminator are dropped; then, if the text has two or more
// sentences, the first one (the rest of the prompt's sentence) is dropped, and so is a last sentence
// without a terminator if one or more sentences remain before it. Text without any sentence boundary
// is kept whole: a model that writes 600 bytes without one has that scored by S4, not hidden by
// dropping the sample. Unlike the human cleaner it never drops whole lines. Returns "" if nothing is left.
inline std::string clean_generated_text(const std::string& raw) {
  using namespace scribe_detail;
  const bool starts_clean = !raw.empty() && is_space_a(static_cast<unsigned char>(raw[0]));
  std::string t = strip_wiki_markup(raw);
  for (char& c : t) {
    if (c == '{' || c == '}' || c == '[' || c == ']' || c == '|' || c == '<' || c == '>' || c == '=' || c == '*' || c == '#' ||
        c == '_' || c == '\\' || c == '~' || c == '\n' || c == '\r' || c == '\t')
      c = ' ';
  }
  t = collapse_ws(t);
  if (!starts_clean) {
    const size_t sp = t.find(' ');
    t = sp == std::string::npos ? std::string() : t.substr(sp + 1);
  }
  auto terminated = [](const std::string& x) {
    size_t e = x.size();
    while (e > 0 && (x[e - 1] == '"' || x[e - 1] == ')' || x[e - 1] == '\'')) --e;
    return e > 0 && (x[e - 1] == '.' || x[e - 1] == '!' || x[e - 1] == '?');
  };
  if (!terminated(t)) {
    const size_t sp = t.rfind(' ');
    t = sp == std::string::npos ? std::string() : t.substr(0, sp);
  }
  std::vector<std::string> s = split_sentences(t);
  if (s.size() >= 2) s.erase(s.begin());
  if (s.size() >= 2 && !terminated(s.back())) s.pop_back();
  std::string o;
  for (const std::string& x : s) {
    if (!o.empty()) o.push_back(' ');
    o += x;
  }
  return o;
}

inline bool numeric_token(const std::string& t) {
  using namespace scribe_detail;
  if (t.empty() || !is_digit_a(static_cast<unsigned char>(t[0]))) return false;
  size_t p = 0;
  while (p < t.size() && (is_digit_a(static_cast<unsigned char>(t[p])) || t[p] == '.' || t[p] == ',')) ++p;
  if (p == t.size()) return true;
  const std::string r = t.substr(p);
  return r == "s" || r == "st" || r == "nd" || r == "rd" || r == "th";
}

// Lowercased word tokens. Hyphens and other punctuation separate; an apostrophe between letters
// stays inside; "x's" splits into "x" and "'s"; digit groups keep inner . and ,.
inline void tokenize_words(const std::string& s, std::vector<std::string>& out) {
  using namespace scribe_detail;
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    size_t L = word_char_len(s, i);
    if (L == 0) {
      ++i;
      continue;
    }
    std::string tok;
    size_t j = i;
    while (j < n) {
      L = word_char_len(s, j);
      if (L > 0) {
        if (L == 1) tok.push_back(static_cast<char>(lower_a(static_cast<unsigned char>(s[j]))));
        else tok.append(s, j, L);
        j += L;
      } else if (s[j] == '\'' && !tok.empty() && j + 1 < n && word_char_len(s, j + 1) > 0) {
        tok.push_back('\'');
        ++j;
      } else if (s.compare(j, 3, "\xE2\x80\x99") == 0 && !tok.empty() && j + 3 < n && word_char_len(s, j + 3) > 0) {
        tok.push_back('\'');
        j += 3;
      } else if ((s[j] == '.' || s[j] == ',') && !tok.empty() && is_digit_a(static_cast<unsigned char>(tok.back())) &&
                 j + 1 < n && is_digit_a(static_cast<unsigned char>(s[j + 1]))) {
        tok.push_back(s[j]);
        ++j;
      } else {
        break;
      }
    }
    i = j;
    if (tok.size() > 2 && tok.compare(tok.size() - 2, 2, "'s") == 0) {
      out.push_back(tok.substr(0, tok.size() - 2));
      out.push_back("'s");
    } else {
      out.push_back(std::move(tok));
    }
  }
}

// ===========================================================================
// Statistics helpers
// ===========================================================================

// Empirical CDF with mid-rank ties and +0.5/(N+1) smoothing: F in (0, 1), never exactly 0 or 1.
struct Ecdf {
  std::vector<double> v;  // sorted ascending
  void add(double x) { v.push_back(x); }
  void finish() { std::sort(v.begin(), v.end()); }
  double F(double x) const {
    if (v.empty()) return 0.5;
    const auto lo = std::lower_bound(v.begin(), v.end(), x);
    const auto hi = std::upper_bound(v.begin(), v.end(), x);
    return (static_cast<double>(lo - v.begin()) + 0.5 * static_cast<double>(hi - lo) + 0.5) /
           (static_cast<double>(v.size()) + 1.0);
  }
  // PLAN 6.1: one-sided (higher is better) and two-sided (typical is best).
  double one_sided(double x) const { return std::min(1.0, 2.0 * F(x)); }
  double two_sided(double x) const { return 1.0 - std::fabs(2.0 * F(x) - 1.0); }
  // One-sided with lower being better: min(1, 2 (1 - F)).
  double one_sided_low(double x) const { return std::min(1.0, 2.0 * (1.0 - F(x))); }
};

inline double jsd_bits(const std::vector<double>& p, const std::vector<double>& q) {
  double j = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const double m = 0.5 * (p[i] + q[i]);
    if (p[i] > 0) j += 0.5 * p[i] * std::log2(p[i] / m);
    if (q[i] > 0) j += 0.5 * q[i] * std::log2(q[i] / m);
  }
  return std::min(1.0, std::max(0.0, j));
}

inline double geo_mean(const std::vector<double>& x, double floor_v = 1e-6) {
  if (x.empty()) return 0;
  double s = 0;
  for (double v : x) s += std::log(std::max(floor_v, v));
  return std::exp(s / static_cast<double>(x.size()));
}

inline double mean_of(const std::vector<double>& x) {
  if (x.empty()) return 0;
  return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

// 95% percentile bootstrap interval of the mean (B resamples over samples).
inline std::pair<double, double> bootstrap_mean_ci(const std::vector<double>& x, int B = 1000, std::uint64_t seed = 1) {
  if (x.empty()) return {0, 0};
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, x.size() - 1);
  std::vector<double> means(static_cast<size_t>(B));
  for (int b = 0; b < B; ++b) {
    double s = 0;
    for (size_t i = 0; i < x.size(); ++i) s += x[pick(rng)];
    means[static_cast<size_t>(b)] = s / static_cast<double>(x.size());
  }
  std::sort(means.begin(), means.end());
  return {means[static_cast<size_t>(0.025 * (B - 1))], means[static_cast<size_t>(std::ceil(0.975 * (B - 1)))]};
}

// P(a > b) over all pairs (a from hi, b from lo), ties count one half.
inline double pairwise_accuracy(const std::vector<double>& hi, const std::vector<double>& lo) {
  if (hi.empty() || lo.empty()) return 0;
  std::vector<double> sl = lo;
  std::sort(sl.begin(), sl.end());
  double wins = 0;
  for (double a : hi) {
    const double below = static_cast<double>(std::lower_bound(sl.begin(), sl.end(), a) - sl.begin());
    const double upto = static_cast<double>(std::upper_bound(sl.begin(), sl.end(), a) - sl.begin());
    wins += below + 0.5 * (upto - below);
  }
  return wins / (static_cast<double>(hi.size()) * static_cast<double>(lo.size()));
}

inline double pearson(const std::vector<double>& x, const std::vector<double>& y) {
  const size_t n = std::min(x.size(), y.size());
  if (n < 3) return 0;
  const double mx = std::accumulate(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n), 0.0) / static_cast<double>(n);
  const double my = std::accumulate(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(n), 0.0) / static_cast<double>(n);
  double sxy = 0, sxx = 0, syy = 0;
  for (size_t i = 0; i < n; ++i) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
    syy += (y[i] - my) * (y[i] - my);
  }
  return (sxx > 0 && syy > 0) ? sxy / std::sqrt(sxx * syy) : 0;
}

inline std::vector<double> ranks_avg(const std::vector<double>& x) {
  std::vector<size_t> idx(x.size());
  std::iota(idx.begin(), idx.end(), size_t{0});
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return x[a] < x[b]; });
  std::vector<double> r(x.size());
  for (size_t i = 0; i < idx.size();) {
    size_t j = i;
    while (j < idx.size() && x[idx[j]] == x[idx[i]]) ++j;
    for (size_t k = i; k < j; ++k) r[idx[k]] = 0.5 * static_cast<double>(i + j - 1);
    i = j;
  }
  return r;
}

inline double spearman(const std::vector<double>& x, const std::vector<double>& y) {
  return pearson(ranks_avg(x), ranks_avg(y));
}

// ===========================================================================
// The reference-trained model: lexicon, Kneser-Ney word trigram, IDF
// ===========================================================================

// Sentence length histogram: 11 bins of 4 tokens (1-4, 5-8, ..., 37-40), the last bin is >= 41.
constexpr int kLenBins = 11;
constexpr size_t kMaxLen = 120;  // sentence lengths above this share one entry of the pmf
inline size_t len_bin(size_t L) { return std::min<size_t>(kLenBins - 1, (std::max<size_t>(L, 1) - 1) / 4); }

class ScribeModel {
 public:
  static constexpr std::uint32_t kUnk = 0, kBos = 1, kEos = 2, kNum = 3;
  static constexpr double kOovPenalty = 8.0;  // extra nats per out-of-vocabulary word in S3

  // Trains on cleaned documents (one string per document). Words seen at least min_count times
  // form the lexicon; everything else is <unk> in the trigram.
  void train(const std::vector<std::string>& docs, std::uint32_t min_count = 2) {
    std::unordered_map<std::string, std::uint32_t> prov;
    prov.reserve(1u << 20);
    std::vector<std::string> prov_words(4);
    std::vector<std::uint64_t> prov_count(4, 0);
    std::vector<std::uint32_t> flat, sent_off;
    std::vector<std::string> toks;
    for (const std::string& d : docs) {
      for (const std::string& s : split_sentences(d)) {
        toks.clear();
        tokenize_words(s, toks);
        if (toks.empty()) continue;
        sent_off.push_back(static_cast<std::uint32_t>(flat.size()));
        for (const std::string& t : toks) {
          std::uint32_t id;
          if (numeric_token(t)) {
            id = kNum;
          } else {
            auto it = prov.find(t);
            if (it == prov.end()) {
              id = static_cast<std::uint32_t>(prov_words.size());
              prov.emplace(t, id);
              prov_words.push_back(t);
              prov_count.push_back(0);
            } else {
              id = it->second;
            }
          }
          ++prov_count[id];
          flat.push_back(id);
        }
      }
    }
    sent_off.push_back(static_cast<std::uint32_t>(flat.size()));
    n_sent_ = sent_off.size() - 1;
    len_hist_.assign(kLenBins, 0.0);
    len_pmf_.assign(kMaxLen + 1, 0.0);
    for (size_t s = 0; s + 1 < sent_off.size(); ++s) {
      const std::uint32_t L = sent_off[s + 1] - sent_off[s];
      len_hist_[len_bin(L)] += 1.0;
      len_pmf_[std::min<size_t>(L, kMaxLen)] += 1.0;
    }
    for (double& x : len_hist_) x /= static_cast<double>(std::max<std::uint64_t>(1, n_sent_));
    for (double& x : len_pmf_) x /= static_cast<double>(std::max<std::uint64_t>(1, n_sent_));
    // Lexicon: provisional ids with count >= min_count, ranked by count (ties by string).
    std::vector<std::uint32_t> keep;
    for (std::uint32_t p = 4; p < prov_words.size(); ++p)
      if (prov_count[p] >= min_count) keep.push_back(p);
    std::sort(keep.begin(), keep.end(), [&](std::uint32_t a, std::uint32_t b) {
      return prov_count[a] != prov_count[b] ? prov_count[a] > prov_count[b] : prov_words[a] < prov_words[b];
    });
    if (keep.size() > 2000000) keep.resize(2000000);
    std::vector<std::uint32_t> remap(prov_words.size(), kUnk);
    remap[kNum] = kNum;
    words_ = {"<unk>", "<s>", "</s>", "<num>"};
    counts_.assign(4, 0);
    for (std::uint32_t p : keep) {
      remap[p] = static_cast<std::uint32_t>(words_.size());
      words_.push_back(prov_words[p]);
      counts_.push_back(prov_count[p]);
    }
    lex_.reserve(words_.size() * 2);
    for (std::uint32_t i = 4; i < words_.size(); ++i) lex_.emplace(words_[i], i);
    V_ = words_.size();
    cum_.assign(V_, 0);  // cumulative counts over lexicon ids (>= 4) for frequency-weighted draws
    for (std::uint32_t i = 4; i < V_; ++i) cum_[i] = cum_[i - 1] + counts_[i];
    // Trigram events and IDF document frequencies from the flat stream.
    std::vector<std::uint64_t> tri;
    tri.reserve(flat.size() + sent_off.size());
    df_.assign(V_, 0);
    std::vector<std::uint32_t> ids, uniq;
    for (size_t s = 0; s + 1 < sent_off.size(); ++s) {
      ids.assign(2, kBos);
      uniq.clear();
      for (std::uint32_t p = sent_off[s]; p < sent_off[s + 1]; ++p) {
        const std::uint32_t id = remap[flat[p]];
        ids.push_back(id);
        if (id >= 4) uniq.push_back(id);
      }
      ids.push_back(kEos);
      std::sort(uniq.begin(), uniq.end());
      uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
      for (std::uint32_t id : uniq) ++df_[id];
      for (size_t t = 2; t < ids.size(); ++t)
        tri.push_back((std::uint64_t(ids[t - 2]) << 42) | (std::uint64_t(ids[t - 1]) << 21) | ids[t]);
    }
    std::vector<std::uint32_t>().swap(flat);
    std::sort(tri.begin(), tri.end());
    // Unique trigrams with counts.
    for (size_t i = 0; i < tri.size();) {
      size_t j = i;
      while (j < tri.size() && tri[j] == tri[i]) ++j;
      tri_.k.push_back(tri[i]);
      tri_.v.push_back(static_cast<std::uint32_t>(j - i));
      i = j;
    }
    std::vector<std::uint64_t>().swap(tri);
    D3_ = discount(tri_.v);
    // Trigram context (u,v): total and number of distinct followers.
    group(tri_.k, tri_.v, 21, c3_);
    // Bigram continuation counts N1+(. v w) = number of distinct u.
    std::vector<std::uint64_t> bi;
    bi.reserve(tri_.k.size());
    const std::uint64_t m42 = (std::uint64_t(1) << 42) - 1;
    for (std::uint64_t k : tri_.k) bi.push_back(k & m42);
    std::sort(bi.begin(), bi.end());
    for (size_t i = 0; i < bi.size();) {
      size_t j = i;
      while (j < bi.size() && bi[j] == bi[i]) ++j;
      bi_.k.push_back(bi[i]);
      bi_.v.push_back(static_cast<std::uint32_t>(j - i));
      i = j;
    }
    std::vector<std::uint64_t>().swap(bi);
    D2_ = discount(bi_.v);
    group(bi_.k, bi_.v, 21, c2_);
    // Unigram continuation counts N1+(. w) = number of distinct v.
    cc1_.assign(V_, 0);
    for (std::uint64_t k : bi_.k) ++cc1_[k & ((std::uint64_t(1) << 21) - 1)];
    tot1_ = static_cast<double>(bi_.k.size());
    distinct1_ = 0;
    for (std::uint32_t c : cc1_) distinct1_ += (c > 0);
    D1_ = discount(cc1_);
    trained_ = true;
  }

  bool trained() const { return trained_; }
  // Natural-log unigram probability of a lexicon id (ids >= 4) among lexicon tokens; 0 for other ids.
  double log_unigram(std::uint32_t id) const {
    if (id < 4 || id >= V_ || cum_.empty() || cum_.back() == 0) return 0.0;
    return std::log(static_cast<double>(counts_[id]) / static_cast<double>(cum_.back()));
  }
  size_t vocab_size() const { return V_; }
  std::uint64_t n_train_sentences() const { return n_sent_; }
  size_t n_trigrams() const { return tri_.k.size(); }
  const std::vector<std::string>& words() const { return words_; }
  // Sentence-length histogram (kLenBins bins) of the training text: the human reference for E3.
  const std::vector<double>& length_histogram() const { return len_hist_; }
  // Sentence-length pmf of the training text, index = tokens per sentence (0..kMaxLen, the last
  // entry holds every longer sentence): the human reference for the E3 Kolmogorov distance.
  const std::vector<double>& length_pmf() const { return len_pmf_; }

  // Id of a token: numeric -> kNum, lexicon word -> its id, otherwise kUnk.
  std::uint32_t word_id(const std::string& tok) const {
    if (numeric_token(tok)) return kNum;
    auto it = lex_.find(tok);
    return it == lex_.end() ? kUnk : it->second;
  }
  bool in_lexicon_or_numeric(const std::string& tok) const { return word_id(tok) != kUnk; }

  // Frequency-weighted draw from the lexicon (ids >= 4); u uniform in [0,1).
  const std::string& draw_word(double u) const {
    const std::uint64_t total = cum_.back();
    const std::uint64_t r = static_cast<std::uint64_t>(u * static_cast<double>(total));
    auto it = std::upper_bound(cum_.begin() + 4, cum_.end(), r);
    size_t id = static_cast<size_t>(it - cum_.begin());
    if (id >= V_) id = V_ - 1;
    return words_[id];
  }

  // Sentence-level IDF (natural log) of a lexicon id; unknown words get the maximum.
  double idf(std::uint32_t id) const {
    const double N = static_cast<double>(n_sent_);
    if (id < 4 || id >= V_) return std::log(N + 1.0);
    return std::log((N + 1.0) / (static_cast<double>(df_[id]) + 1.0));
  }

  // Interpolated Kneser-Ney trigram, natural log. Sums to one over all ids (tested).
  double logp(std::uint32_t u, std::uint32_t v, std::uint32_t w) const {
    const double V = static_cast<double>(V_);
    double p = (std::max(0.0, static_cast<double>(cc1_[w]) - D1_) + D1_ * distinct1_ / V) / tot1_;
    const std::uint64_t k2 = (std::uint64_t(v) << 21) | w;
    long ci = c2_.find(std::uint64_t(v));
    if (ci >= 0) {
      const double tot = c2_.total[static_cast<size_t>(ci)];
      const double cc = bi_.get(k2);
      p = std::max(0.0, cc - D2_) / tot + D2_ * c2_.distinct[static_cast<size_t>(ci)] / tot * p;
    }
    ci = c3_.find((std::uint64_t(u) << 21) | v);
    if (ci >= 0) {
      const double tot = c3_.total[static_cast<size_t>(ci)];
      const double c = tri_.get((std::uint64_t(u) << 42) | k2);
      p = std::max(0.0, c - D3_) / tot + D3_ * c3_.distinct[static_cast<size_t>(ci)] / tot * p;
    }
    return std::log(p);
  }

  // Sum over the in-lexicon words of a sentence of log p(w | u, v) - log p_freq(w), and their number (v2 s3g).
  double sentence_gain(const std::vector<std::uint32_t>& ids, size_t& n) const {
    std::uint32_t u = kBos, v = kBos;
    double g = 0;
    for (std::uint32_t w : ids) {
      if (w >= 4) {
        g += logp(u, v, w) - log_unigram(w);
        ++n;
      }
      u = v;
      v = w;
    }
    return g;
  }
  // Log-prob of one sentence (ids of its tokens) including </s>; adds the OOV penalty.
  // Returns the number of predicted tokens (words + </s>) through n_pred.
  double sentence_logp(const std::vector<std::uint32_t>& ids, size_t& n_pred) const {
    std::uint32_t u = kBos, v = kBos;
    double lp = 0;
    for (std::uint32_t w : ids) {
      lp += logp(u, v, w);
      if (w == kUnk) lp -= kOovPenalty;
      u = v;
      v = w;
    }
    lp += logp(u, v, kEos);
    n_pred += ids.size() + 1;
    return lp;
  }

 private:
  struct KV {
    std::vector<std::uint64_t> k;
    std::vector<std::uint32_t> v;
    std::uint32_t get(std::uint64_t key) const {
      const auto it = std::lower_bound(k.begin(), k.end(), key);
      return (it != k.end() && *it == key) ? v[static_cast<size_t>(it - k.begin())] : 0;
    }
  };
  struct Ctx {
    std::vector<std::uint64_t> k;
    std::vector<std::uint32_t> total, distinct;
    long find(std::uint64_t key) const {
      const auto it = std::lower_bound(k.begin(), k.end(), key);
      return (it != k.end() && *it == key) ? static_cast<long>(it - k.begin()) : -1;
    }
  };
  static double discount(const std::vector<std::uint32_t>& counts) {
    double n1 = 0, n2 = 0;
    for (std::uint32_t c : counts) {
      n1 += (c == 1);
      n2 += (c == 2);
    }
    return (n1 + 2 * n2) > 0 ? n1 / (n1 + 2 * n2) : 0.5;
  }
  // Groups sorted keys by key >> shift: sum of counts and number of keys.
  static void group(const std::vector<std::uint64_t>& keys, const std::vector<std::uint32_t>& cnt, int shift, Ctx& out) {
    for (size_t i = 0; i < keys.size();) {
      const std::uint64_t g = keys[i] >> shift;
      size_t j = i;
      std::uint32_t tot = 0;
      while (j < keys.size() && (keys[j] >> shift) == g) tot += cnt[j++];
      out.k.push_back(g);
      out.total.push_back(tot);
      out.distinct.push_back(static_cast<std::uint32_t>(j - i));
      i = j;
    }
  }

  bool trained_ = false;
  size_t V_ = 0;
  std::uint64_t n_sent_ = 0;
  std::vector<std::string> words_;
  std::vector<std::uint64_t> counts_, cum_;
  std::unordered_map<std::string, std::uint32_t> lex_;
  std::vector<std::uint32_t> df_, cc1_;
  std::vector<double> len_hist_, len_pmf_;
  KV tri_, bi_;
  Ctx c3_, c2_;
  double D1_ = 0.5, D2_ = 0.5, D3_ = 0.5, tot1_ = 1, distinct1_ = 1;
};

// ===========================================================================
// C6: word vectors (PPMI-SVD, native/tools/semantics/ppmi_svd.cpp) and SIF sentence vectors
// ===========================================================================

// Word vectors from counts only. File format = ppmi_svd's output: <prefix>.vectors.f32 is V x dim
// float32, row-major, unit rows; <prefix>.vocab.tsv has "word<TAB>count" per line, row i = line i.
// On load the count-weighted mean vector is subtracted and the rows are re-normalised, which
// removes the direction every word shares; word-level cohesion needs that.
class WordVectors {
 public:
  void set(std::vector<std::string> words, std::vector<double> counts, std::vector<float> vecs, size_t dim, bool center = true) {
    words_ = std::move(words);
    dim_ = dim;
    v_ = std::move(vecs);
    const size_t V = words_.size();
    total_ = 0;
    for (double c : counts) total_ += c;
    p_.assign(V, 0.0);
    for (size_t i = 0; i < V; ++i) p_[i] = total_ > 0 ? counts[i] / total_ : 0.0;
    idx_.clear();
    idx_.reserve(V * 2);
    for (size_t i = 0; i < V; ++i) idx_.emplace(words_[i], static_cast<int>(i));
    if (center && V > 0) {
      std::vector<double> m(dim_, 0.0);
      for (size_t i = 0; i < V; ++i)
        for (size_t k = 0; k < dim_; ++k) m[k] += p_[i] * v_[i * dim_ + k];
      for (size_t i = 0; i < V; ++i) {
        double n2 = 0;
        for (size_t k = 0; k < dim_; ++k) {
          const double x = v_[i * dim_ + k] - m[k];
          v_[i * dim_ + k] = static_cast<float>(x);
          n2 += x * x;
        }
        const double inv = n2 > 0 ? 1.0 / std::sqrt(n2) : 0.0;
        for (size_t k = 0; k < dim_; ++k) v_[i * dim_ + k] = static_cast<float>(v_[i * dim_ + k] * inv);
      }
    }
  }
  // False if the files are unreadable or their sizes disagree.
  bool load_streams(std::istream& vec, std::istream& vocab, size_t dim) {
    std::vector<std::string> words;
    std::vector<double> counts;
    std::string line;
    while (std::getline(vocab, line)) {
      const size_t t = line.find('\t');
      if (t == std::string::npos) continue;
      words.push_back(line.substr(0, t));
      counts.push_back(std::strtod(line.c_str() + t + 1, nullptr));
    }
    if (words.empty() || dim == 0) return false;
    std::vector<float> v(words.size() * dim);
    vec.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    if (static_cast<size_t>(vec.gcount()) != v.size() * sizeof(float)) return false;
    set(std::move(words), std::move(counts), std::move(v), dim);
    return true;
  }
  bool load(const std::string& prefix, size_t dim = 300) {
    std::ifstream a(prefix + ".vectors.f32", std::ios::binary), b(prefix + ".vocab.tsv");
    return a && b && load_streams(a, b, dim);
  }
  size_t dim() const { return dim_; }
  size_t size() const { return words_.size(); }
  int index(const std::string& w) const {
    const auto it = idx_.find(w);
    return it == idx_.end() ? -1 : it->second;
  }
  const float* row(int i) const { return v_.data() + static_cast<size_t>(i) * dim_; }
  double prob(int i) const { return p_[static_cast<size_t>(i)]; }

 private:
  size_t dim_ = 0;
  double total_ = 0;
  std::vector<std::string> words_;
  std::vector<float> v_;
  std::vector<double> p_;
  std::unordered_map<std::string, int> idx_;
};

// Arora, Liang and Ma (2017): sentence vector = mean of a/(a+p(w)) * v_w, minus its projection on
// the first principal direction of a sample of such vectors (fit_pc), then L2-normalised.
class SentenceEmbedder {
 public:
  explicit SentenceEmbedder(const WordVectors* wv = nullptr, double a = 1e-3) : wv_(wv), a_(a) {}
  bool ready() const { return wv_ != nullptr && wv_->size() > 0; }
  const WordVectors& vectors() const { return *wv_; }
  size_t dim() const { return wv_ ? wv_->dim() : 0; }

  // Weighted mean of the word vectors of the known tokens; false if fewer than min_known are known.
  bool raw(const std::vector<std::string>& toks, std::vector<float>& out, int min_known = 2) const {
    const size_t d = dim();
    out.assign(d, 0.0f);
    int known = 0;
    for (const std::string& t : toks) {
      const int i = wv_->index(t);
      if (i < 0) continue;
      const float w = static_cast<float>(a_ / (a_ + wv_->prob(i)));
      const float* r = wv_->row(i);
      for (size_t k = 0; k < d; ++k) out[k] += w * r[k];
      ++known;
    }
    if (known < min_known || known == 0) return false;
    for (float& x : out) x /= static_cast<float>(known);
    return true;
  }
  // First right singular vector of the (uncentred) matrix of raw sentence vectors, by power iteration.
  void fit_pc(const std::vector<std::vector<float>>& X, int iters = 60) {
    const size_t d = dim();
    pc_.assign(d, 0.0f);
    if (X.empty() || d == 0) return;
    std::vector<double> v(d), nv(d);
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> N(0.0, 1.0);
    for (double& x : v) x = N(rng);
    for (int it = 0; it < iters; ++it) {
      std::fill(nv.begin(), nv.end(), 0.0);
      for (const auto& x : X) {
        double dot = 0;
        for (size_t k = 0; k < d; ++k) dot += x[k] * v[k];
        for (size_t k = 0; k < d; ++k) nv[k] += dot * x[k];
      }
      double n2 = 0;
      for (double x : nv) n2 += x * x;
      if (!(n2 > 0)) return;
      const double inv = 1.0 / std::sqrt(n2);
      for (size_t k = 0; k < d; ++k) v[k] = nv[k] * inv;
    }
    for (size_t k = 0; k < d; ++k) pc_[k] = static_cast<float>(v[k]);
  }
  const std::vector<float>& pc() const { return pc_; }
  // Unit sentence vector with the first principal component removed; false if there is nothing to embed.
  bool embed(const std::vector<std::string>& toks, std::vector<float>& out, int min_known = 2) const {
    if (!raw(toks, out, min_known)) return false;
    const size_t d = dim();
    if (pc_.size() == d) {
      double dot = 0;
      for (size_t k = 0; k < d; ++k) dot += out[k] * pc_[k];
      for (size_t k = 0; k < d; ++k) out[k] -= static_cast<float>(dot) * pc_[k];
    }
    double n2 = 0;
    for (float x : out) n2 += double(x) * x;
    if (!(n2 > 1e-18)) return false;
    const float inv = static_cast<float>(1.0 / std::sqrt(n2));
    for (float& x : out) x *= inv;
    return true;
  }

 private:
  const WordVectors* wv_;
  double a_;
  std::vector<float> pc_;
};

// ===========================================================================
// Per-sample raw statistics
// ===========================================================================

// Dense ids for words, shared by all samples of a run; used by the set-level statistics.
class WordDict {
 public:
  std::uint32_t id(const std::string& w) {
    auto it = m_.find(w);
    if (it != m_.end()) return it->second;
    const std::uint32_t i = static_cast<std::uint32_t>(m_.size());
    m_.emplace(w, i);
    return i;
  }
  size_t size() const { return m_.size(); }

 private:
  std::unordered_map<std::string, std::uint32_t> m_;
};

struct Features {
  size_t bytes = 0;
  int n_sent = 0;   // sentences with at least one token
  int n_words = 0;  // word tokens
  bool has_c = false;                     // at least one coherence statistic is defined
  bool has_c0 = false, has_c1 = false, has_c3 = false;
  // raw statistics
  double s1 = 0, s3 = -30, s4 = 0, c0 = 0, c1 = 0, c3 = 1, loop8 = 0, rep = 0, d[4] = {1, 1, 1, 1};
  // v2 additions from extract_features (v1 never reads them)
  bool has_c1a = false;   // two or more embeddable sentences in a row
  double c1a = 0;         // mean cosine of ADJACENT sentence vectors
  bool has_c2 = false;    // two or more segments with content words
  double c2 = 0;          // lexical cohesion: mean IDF-weighted overlap of content words over all pairs of segments
  double s3min = -30;     // worst sentence: lowest per-token log-probability of a sentence with >= 3 tokens
  double s3g = 0;         // context gain: mean of log p(w | u, v) - log p_freq(w) over in-lexicon words (order information)
  bool has_s3g = false;
  double sl_mean = 0;     // mean sentence length (tokens)
  double sl_max = 0;      // longest sentence (tokens)
  // v2 additions from ScribeV2Scorer::extend
  bool has_v2 = false, has_prompt = false, has_r = false;
  double s2 = -30;        // skeleton trigram: per-token log-probability of the function-word / word-class sequence
  double s2g = 0;         // skeleton context gain over the skeleton unigram
  double wfreq = 0;       // mean unigram log-probability of the lexicon words (word rarity)
  double i1 = 0;          // share of tokens inside a word 8-gram that occurs verbatim in the pack
  double r1 = 0, r4 = 0, r5 = 0, i4 = 0;  // prompted: prompt cosine, key-term coverage, informativeness, prompt echo
  // set level
  std::vector<std::uint32_t> ids;       // WordDict ids of the tokens
  std::vector<std::uint16_t> sent_len;  // tokens per sentence
};

namespace scribe_detail {

inline std::uint64_t fnv1a(const std::string& s) {
  std::uint64_t h = 14695981039346656037ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

// Fraction of 8-byte windows that already occurred earlier (ported from surface_score.cpp).
inline double loop8_bytes(const std::string& s) {
  if (s.size() < 16) return 0;
  std::unordered_set<std::uint64_t> seen;
  int loops = 0, n = 0;
  for (size_t i = 0; i + 8 <= s.size(); ++i) {
    std::uint64_t h = 14695981039346656037ull;
    for (int k = 0; k < 8; ++k) {
      h ^= static_cast<unsigned char>(s[i + static_cast<size_t>(k)]);
      h *= 1099511628211ull;
    }
    if (!seen.insert(h).second) ++loops;
    ++n;
  }
  return static_cast<double>(loops) / static_cast<double>(n);
}

inline double distinct4_bytes(const std::string& s) {
  if (s.size() < 4) return 0;
  std::vector<std::uint32_t> g;
  g.reserve(s.size());
  for (size_t i = 0; i + 3 < s.size(); ++i)
    g.push_back((std::uint32_t(std::uint8_t(s[i])) << 24) | (std::uint32_t(std::uint8_t(s[i + 1])) << 16) |
                (std::uint32_t(std::uint8_t(s[i + 2])) << 8) | std::uint32_t(std::uint8_t(s[i + 3])));
  std::sort(g.begin(), g.end());
  g.erase(std::unique(g.begin(), g.end()), g.end());
  return static_cast<double>(g.size()) / static_cast<double>(s.size() - 3);
}

inline bool brackets_balanced(const std::string& s) {
  std::string st;
  for (char c : s) {
    if (c == '(' || c == '[' || c == '{') {
      st.push_back(c);
    } else if (c == ')' || c == ']' || c == '}') {
      const char o = c == ')' ? '(' : (c == ']' ? '[' : '{');
      if (st.empty() || st.back() != o) return false;
      st.pop_back();
    }
  }
  return st.empty();
}

// S4: six checks; the capital check is the fraction of sentences that begin with a capital.
inline double well_formedness(const std::string& text, const std::vector<std::string>& sents) {
  double pass = 0;
  pass += utf8_valid(text) ? 1 : 0;
  {
    int ok = 0, tot = 0;
    for (const std::string& s : sents) {
      size_t i = 0;
      while (i < s.size() && !is_alpha_a(static_cast<unsigned char>(s[i])) && static_cast<unsigned char>(s[i]) < 0x80) ++i;
      if (i >= s.size()) continue;  // no letters: nothing to check
      ++tot;
      const unsigned char c = static_cast<unsigned char>(s[i]);
      ok += (is_upper_a(c) || c >= 0x80 || is_digit_a(c)) ? 1 : 0;
    }
    pass += tot ? static_cast<double>(ok) / tot : 1.0;
  }
  size_t e = text.size();
  while (e > 0 && is_space_a(static_cast<unsigned char>(text[e - 1]))) --e;
  size_t q = e;
  while (q > 0 && (text[q - 1] == '"' || text[q - 1] == ')' || text[q - 1] == '\'' || text[q - 1] == ']')) --q;
  pass += (q > 0 && (text[q - 1] == '.' || text[q - 1] == '!' || text[q - 1] == '?')) ? 1 : 0;
  pass += brackets_balanced(text) ? 1 : 0;
  pass += (std::count(text.begin(), text.end(), '"') % 2 == 0) ? 1 : 0;
  pass += (e > 0 && !is_alnum_a(static_cast<unsigned char>(text[e - 1])) &&
           static_cast<unsigned char>(text[e - 1]) < 0x80)
              ? 1
              : 0;
  return pass / 6.0;
}

using SparseVec = std::vector<std::pair<std::uint64_t, float>>;

inline double cosine(const SparseVec& a, const SparseVec& b) {
  double dot = 0, na = 0, nb = 0;
  for (const auto& x : a) na += double(x.second) * x.second;
  for (const auto& x : b) nb += double(x.second) * x.second;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i].first == b[j].first) dot += double(a[i++].second) * b[j++].second;
    else if (a[i].first < b[j].first) ++i;
    else ++j;
  }
  return (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
}

// Mean cosine over ALL pairs i < j of unit vectors (dense, dim d): (|sum|^2 - n) / (n (n - 1)).
inline double mean_pairwise_cosine(const std::vector<std::vector<float>>& U) {
  const size_t n = U.size();
  if (n < 2) return 0;
  const size_t d = U[0].size();
  std::vector<double> sum(d, 0.0);
  double self = 0;
  for (const auto& u : U)
    for (size_t k = 0; k < d; ++k) {
      sum[k] += u[k];
      self += double(u[k]) * u[k];
    }
  double s2 = 0;
  for (double x : sum) s2 += x * x;
  return (s2 - self) / (static_cast<double>(n) * static_cast<double>(n - 1));
}

}  // namespace scribe_detail

// Content words for C3: sentence-df fraction below exp(-kContentIdf), i.e. not a function word.
constexpr double kContentIdf = 3.0;

// Tokens of the text, one list per sentence (sentences without tokens are dropped).
inline std::vector<std::vector<std::string>> sentence_tokens(const std::string& text, std::vector<std::string>* sents_out = nullptr) {
  std::vector<std::string> sents = split_sentences(text);
  std::vector<std::vector<std::string>> toks;
  for (const std::string& s : sents) {
    std::vector<std::string> t;
    tokenize_words(s, t);
    if (!t.empty()) toks.push_back(std::move(t));
  }
  if (sents_out) *sents_out = std::move(sents);
  return toks;
}

// Cheap features for the set-level (E) pools: only the dense word ids and the sentence lengths.
inline Features extract_set_features(WordDict& dict, const std::string& text) {
  Features f;
  f.bytes = text.size();
  for (const auto& t : sentence_tokens(text)) {
    for (const std::string& w : t) f.ids.push_back(dict.id(w));
    f.sent_len.push_back(static_cast<std::uint16_t>(std::min<size_t>(t.size(), 65535)));
    f.n_words += static_cast<int>(t.size());
  }
  f.n_sent = static_cast<int>(f.sent_len.size());
  return f;
}

// emb: the C6 sentence embedder; with nullptr (or an embedder that is not ready) C1 and C0 fall
// back to TF-IDF bag-of-words cosines (C0 is then absent).
inline Features extract_features(const ScribeModel& m, WordDict& dict, const std::string& text,
                                 const SentenceEmbedder* emb = nullptr) {
  using namespace scribe_detail;
  Features f;
  f.bytes = text.size();
  std::vector<std::string> sents;
  const std::vector<std::vector<std::string>> toks = sentence_tokens(text, &sents);
  f.n_sent = static_cast<int>(toks.size());
  const bool use_vec = emb != nullptr && emb->ready();
  std::vector<std::string> flat;
  int hit = 0;
  double lp = 0;
  size_t npred = 0;
  std::vector<SparseVec> vecs;
  std::vector<std::pair<std::uint64_t, double>> content;  // key, idf in order
  std::vector<std::vector<float>> sentence_vecs;
  std::vector<std::vector<float>> content_word_vecs;  // unique content words known to the vectors
  std::unordered_set<int> content_seen;
  std::vector<std::unordered_map<std::uint64_t, double>> seg_content;  // C2: content words of each sentence
  double gain_sum = 0;
  size_t gain_n = 0;
  for (const auto& t : toks) {
    seg_content.emplace_back();
    std::vector<std::uint32_t> ids;
    std::unordered_map<std::uint64_t, float> tf;
    for (const std::string& w : t) {
      const std::uint32_t id = m.word_id(w);
      ids.push_back(id);
      hit += (id != ScribeModel::kUnk) ? 1 : 0;
      flat.push_back(w);
      f.ids.push_back(dict.id(w));
      if (id != ScribeModel::kNum) {
        const std::uint64_t key = (id != ScribeModel::kUnk) ? id : ((std::uint64_t(1) << 63) | (fnv1a(w) >> 1));
        const double idf = m.idf(id);
        tf[key] += 1.0f;
        if (idf >= kContentIdf) {
          content.emplace_back(key, idf);
          seg_content.back()[key] = idf;
          if (use_vec) {
            const int vi = emb->vectors().index(w);
            if (vi >= 0 && content_seen.insert(vi).second) {
              const float* r = emb->vectors().row(vi);
              content_word_vecs.emplace_back(r, r + emb->dim());
            }
          }
        }
      }
    }
    f.sent_len.push_back(static_cast<std::uint16_t>(std::min<size_t>(t.size(), 65535)));
    gain_sum += m.sentence_gain(ids, gain_n);
    {
      size_t np_s = 0;
      const double lps = m.sentence_logp(ids, np_s);
      lp += lps;
      npred += np_s;
      if (t.size() >= 3) {
        const double per = lps / static_cast<double>(np_s);
        if (f.s3min == -30 || per < f.s3min) f.s3min = per;
      }
    }
    SparseVec v;
    v.reserve(tf.size());
    for (const auto& kv : tf) {
      const std::uint32_t id = (kv.first >> 63) ? ScribeModel::kUnk : static_cast<std::uint32_t>(kv.first);
      v.emplace_back(kv.first, static_cast<float>(kv.second * m.idf(id)));
    }
    std::sort(v.begin(), v.end());
    vecs.push_back(std::move(v));
    if (use_vec) {
      std::vector<float> sv;
      if (emb->embed(t, sv)) sentence_vecs.push_back(std::move(sv));
    }
  }
  f.n_words = static_cast<int>(flat.size());
  if (f.s3min == -30 && npred) f.s3min = lp / static_cast<double>(npred);  // no sentence of 3 tokens: the sample's mean
  if (gain_n >= 3) {
    f.s3g = gain_sum / static_cast<double>(gain_n);
    f.has_s3g = true;
  }
  for (std::uint16_t L : f.sent_len) {
    f.sl_mean += L;
    f.sl_max = std::max<double>(f.sl_max, L);
  }
  if (!f.sent_len.empty()) f.sl_mean /= static_cast<double>(f.sent_len.size());
  if (sentence_vecs.size() >= 2) {
    double sa = 0;
    for (size_t k = 0; k + 1 < sentence_vecs.size(); ++k) {
      double dot = 0;
      for (size_t q = 0; q < sentence_vecs[k].size(); ++q) dot += double(sentence_vecs[k][q]) * sentence_vecs[k + 1][q];
      sa += dot;
    }
    f.c1a = sa / static_cast<double>(sentence_vecs.size() - 1);
    f.has_c1a = true;
  }
  f.s1 = f.n_words ? static_cast<double>(hit) / f.n_words : 0.0;
  f.s3 = npred ? lp / static_cast<double>(npred) : -30.0;
  f.s4 = well_formedness(text, sents);
  // C1: mean cosine over ALL pairs of segments. Segments are the sentences; a sample with fewer than
  // two embeddable sentences is cut into two halves of its tokens instead, so that short samples
  // have the axis too.
  if (use_vec) {
    if (sentence_vecs.size() >= 2) {
      f.c1 = mean_pairwise_cosine(sentence_vecs);
      f.has_c1 = true;
    } else if (flat.size() >= 8) {
      const std::vector<std::string> a(flat.begin(), flat.begin() + static_cast<std::ptrdiff_t>(flat.size() / 2));
      const std::vector<std::string> b(flat.begin() + static_cast<std::ptrdiff_t>(flat.size() / 2), flat.end());
      std::vector<std::vector<float>> two(2);
      if (emb->embed(a, two[0]) && emb->embed(b, two[1])) {
        f.c1 = mean_pairwise_cosine(two);
        f.has_c1 = true;
      }
    }
    // C0: mean cosine over all pairs of distinct content words (word-level cohesion).
    if (content_word_vecs.size() >= 4) {
      f.c0 = mean_pairwise_cosine(content_word_vecs);
      f.has_c0 = true;
    }
  } else if (vecs.size() >= 2) {
    double s = 0;
    size_t np = 0;
    for (size_t i = 0; i < vecs.size(); ++i)
      for (size_t j = i + 1; j < vecs.size(); ++j) {
        s += cosine(vecs[i], vecs[j]);
        ++np;
      }
    f.c1 = s / static_cast<double>(np);
    f.has_c1 = true;
  }
  // C3: JSD of first-half vs second-half content-word distributions (halves of the content words,
  // so it is defined for one sentence too).
  {
    const size_t mc = content.size();
    if (mc >= 4) {
      std::unordered_map<std::uint64_t, std::pair<double, double>> cnt;
      for (size_t i = 0; i < mc; ++i) {
        auto& e = cnt[content[i].first];
        (i < mc / 2 ? e.first : e.second) += 1.0;
      }
      const double na = static_cast<double>(mc / 2), nb = static_cast<double>(mc - mc / 2);
      std::vector<double> p, q;
      for (const auto& kv : cnt) {
        p.push_back(kv.second.first / na);
        q.push_back(kv.second.second / nb);
      }
      f.c3 = jsd_bits(p, q);
      f.has_c3 = true;
    }
  }
  // C2: lexical cohesion (the entity-grid idea with counts only). Segments are the sentences that have content
  // words; with fewer than two, the two halves of the sample's content words. For each pair of segments, the IDF
  // mass of the shared content words over the smaller segment's IDF mass; C2 is the mean over all pairs.
  {
    std::vector<std::unordered_map<std::uint64_t, double>> segs;
    for (auto& sc : seg_content)
      if (!sc.empty()) segs.push_back(std::move(sc));
    if (segs.size() < 2 && content.size() >= 4) {
      segs.assign(2, {});
      for (size_t i = 0; i < content.size(); ++i) segs[i < content.size() / 2 ? 0 : 1][content[i].first] = content[i].second;
    }
    if (segs.size() >= 2) {
      std::vector<double> mass(segs.size(), 0.0);
      for (size_t i = 0; i < segs.size(); ++i)
        for (const auto& kv : segs[i]) mass[i] += kv.second;
      double tot = 0;
      size_t np = 0;
      for (size_t i = 0; i < segs.size(); ++i)
        for (size_t j = i + 1; j < segs.size(); ++j) {
          const auto& a = segs[i].size() <= segs[j].size() ? segs[i] : segs[j];
          const auto& b = segs[i].size() <= segs[j].size() ? segs[j] : segs[i];
          double sh = 0;
          for (const auto& kv : a)
            if (b.count(kv.first)) sh += kv.second;
          const double den = std::min(mass[i], mass[j]);
          tot += den > 0 ? sh / den : 0.0;
          ++np;
        }
      f.c2 = tot / static_cast<double>(np);
      f.has_c2 = true;
    }
  }
  f.has_c = f.has_c0 || f.has_c1 || f.has_c3;
  // I2: loop8 and repeated-sentence rate.
  f.loop8 = loop8_bytes(text);
  if (toks.size() >= 2) {
    std::unordered_set<std::string> seen;
    int dup = 0;
    for (const auto& t : toks) {
      std::string key;
      for (const std::string& w : t) {
        key += w;
        key.push_back(' ');
      }
      if (!seen.insert(key).second) ++dup;
    }
    f.rep = static_cast<double>(dup) / static_cast<double>(toks.size());
  }
  // I3: distinct-1..4 over word n-grams.
  for (int n = 1; n <= 4; ++n) {
    const size_t total = flat.size() >= static_cast<size_t>(n) ? flat.size() - static_cast<size_t>(n) + 1 : 0;
    if (total == 0) {
      f.d[n - 1] = 1.0;
      continue;
    }
    std::unordered_set<std::uint64_t> u;
    for (size_t i = 0; i < total; ++i) {
      std::uint64_t h = 1469598103934665603ull;
      for (int k = 0; k < n; ++k) {
        h ^= fnv1a(flat[i + static_cast<size_t>(k)]) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 1099511628211ull;
      }
      u.insert(h);
    }
    f.d[n - 1] = static_cast<double>(u.size()) / static_cast<double>(total);
  }
  return f;
}

// ===========================================================================
// Normalisation against human reference samples, axes, per-sample SCRIBE
// ===========================================================================

inline int length_bucket(size_t bytes) { return bytes < 300 ? 0 : 1; }  // 0: short, 1: standard

struct BucketReference {
  Ecdf s1, s3, s4, c0, c1, c3, nloop, nrep, d1, d2, d3, d4;
  size_t n = 0;
  void add(const Features& f) {
    ++n;
    s1.add(f.s1);
    s3.add(f.s3);
    s4.add(f.s4);
    nloop.add(-f.loop8);
    nrep.add(-f.rep);
    d1.add(f.d[0]);
    d2.add(f.d[1]);
    d3.add(f.d[2]);
    d4.add(f.d[3]);
    if (f.has_c0) c0.add(f.c0);
    if (f.has_c1) c1.add(f.c1);
    if (f.has_c3) c3.add(f.c3);
  }
  void finish() {
    for (Ecdf* e : {&s1, &s3, &s4, &c0, &c1, &c3, &nloop, &nrep, &d1, &d2, &d3, &d4}) e->finish();
  }
};

struct ScribeProfile {
  double wS = 2, wC = 2, wI = 1, wE = 1;  // open continuation (R and B are 0 in v0/v1)
  double eps = 0.01;
  double gate = 0.2;
  // false (default) = PLAN 6.2: C1 and C3 two-sided, and C0 too. true = one-sided (more cohesion and less drift are
  // better; too much cohesion is the I axis's job). Tried on the development ladder: +0.02 on human > sentences shuffled,
  // nothing elsewhere, so the PLAN's scoring stays.
  bool c_one_sided = false;
  // Which gate the headline `scribe` uses: 0 = PLAN 6.3 caps (S < gate caps at S, I < gate caps at I),
  // 1 = scales (raw * min(1, S/gate) * min(1, I/gate)).
  int gate_mode = 0;
};

struct Scored {
  double s1 = 0, s3 = 0, s4 = 0, c0 = 0, c1 = 0, c3 = 0, i2 = 0, i3 = 0;  // sub-scores in [0,1]
  double S = 0, C = 0, I = 0;                                           // axes
  bool has_c = false;
  double scribe = 0;       // the headline score: gate_mode of the profile
  bool s_gate = false;     // S gate fired
  bool i_gate = false;     // I gate fired
  double scribe_cap = 0;   // PLAN 6.3 caps: S < 0.2 caps at S, and (v1) I < 0.2 caps at I
  double scribe_soft = 0;  // raw * min(1, S/0.2) * min(1, I/0.2)
  double scribe_s = 0;     // the PLAN 6.3 gate alone (S only), as in v0
  double scribe_raw = 0;   // weighted geometric mean before any gate
};

class ScribeScorer {
 public:
  BucketReference ref[2];
  ScribeProfile profile;
  void add_reference(const Features& f) { ref[length_bucket(f.bytes)].add(f); }
  void finish() {
    ref[0].finish();
    ref[1].finish();
  }
  Scored score(const Features& f) const {
    const int b0 = length_bucket(f.bytes);
    const BucketReference& r = ref[ref[b0].n < 20 && ref[1 - b0].n >= 20 ? 1 - b0 : b0];  // too few in the bucket: the other
    Scored o;
    o.s1 = r.s1.one_sided(f.s1);
    o.s3 = r.s3.one_sided(f.s3);
    o.s4 = r.s4.one_sided(f.s4);
    const double nl = r.nloop.one_sided(-f.loop8), nr = r.nrep.one_sided(-f.rep);
    o.i2 = geo_mean({nl, nr});
    o.i3 = geo_mean({r.d1.two_sided(f.d[0]), r.d2.two_sided(f.d[1]), r.d3.two_sided(f.d[2]), r.d4.two_sided(f.d[3])});
    o.has_c = f.has_c;
    if (f.has_c) {
      std::vector<double> cs;
      const bool one = profile.c_one_sided;
      if (f.has_c0) cs.push_back(o.c0 = one ? r.c0.one_sided(f.c0) : r.c0.two_sided(f.c0));
      if (f.has_c1) cs.push_back(o.c1 = one ? r.c1.one_sided(f.c1) : r.c1.two_sided(f.c1));
      if (f.has_c3) cs.push_back(o.c3 = one ? r.c3.one_sided_low(f.c3) : r.c3.two_sided(f.c3));
      o.C = geo_mean(cs);
    }
    o.S = geo_mean({o.s1, o.s3, o.s4});
    o.I = geo_mean({o.i2, o.i3});
    const ScribeProfile& p = profile;
    double num = p.wS * std::log(std::max(p.eps, o.S)) + p.wI * std::log(std::max(p.eps, o.I));
    double den = p.wS + p.wI;
    if (f.has_c) {
      num += p.wC * std::log(std::max(p.eps, o.C));
      den += p.wC;
    }
    o.scribe_raw = std::exp(num / den);
    o.scribe_s = o.scribe_raw;
    if (o.S < p.gate) {
      o.scribe_s = std::min(o.scribe_s, o.S);
      o.s_gate = true;
    }
    o.scribe_cap = o.scribe_s;
    if (o.I < p.gate) {
      o.scribe_cap = std::min(o.scribe_cap, o.I);
      o.i_gate = true;
    }
    o.scribe_soft = o.scribe_raw * std::min(1.0, o.S / p.gate) * std::min(1.0, o.I / p.gate);
    o.scribe = p.gate_mode == 1 ? o.scribe_soft : o.scribe_cap;
    return o;
  }
};

// ===========================================================================
// Set level: E1 Zipf slope, E3 Kolmogorov distance of the sentence-length distribution
// ===========================================================================

struct SetValues {
  double zipf = 0;  // rank-frequency slope
  double ks = 1;    // Kolmogorov distance of the sentence lengths from the human length pmf
};

// sup_L |F_set(L) - F_ref(L)| over sentence lengths (tokens), ref_pmf from ScribeModel::length_pmf.
inline double ks_lengths(const std::vector<const Features*>& set, const std::vector<double>& ref_pmf) {
  std::vector<double> c(kMaxLen + 1, 0.0);
  double n = 0;
  for (const Features* f : set)
    for (std::uint16_t l : f->sent_len) {
      c[std::min<size_t>(l, kMaxLen)] += 1;
      n += 1;
    }
  if (n <= 0) return 1.0;
  double a = 0, b = 0, d = 0;
  for (size_t L = 0; L <= kMaxLen && L < ref_pmf.size(); ++L) {
    a += c[L] / n;
    b += ref_pmf[L];
    d = std::max(d, std::fabs(a - b));
  }
  return d;
}

// scratch must have size >= number of WordDict entries and be all zero on entry (left all zero).
inline SetValues set_values(const std::vector<const Features*>& set, const std::vector<double>& ref_pmf,
                            std::vector<std::uint32_t>& scratch) {
  SetValues o;
  std::vector<std::uint32_t> touched;
  for (const Features* f : set)
    for (std::uint32_t id : f->ids)
      if (scratch[id]++ == 0) touched.push_back(id);
  std::vector<double> cs;
  for (std::uint32_t id : touched) {
    if (scratch[id] >= 2) cs.push_back(scratch[id]);
    scratch[id] = 0;
  }
  std::sort(cs.begin(), cs.end(), std::greater<double>());
  const size_t m = std::min<size_t>(cs.size(), 1000);
  if (m >= 10) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t r = 0; r < m; ++r) {
      const double x = std::log(static_cast<double>(r + 1)), y = std::log(cs[r]);
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
    }
    const double dm = static_cast<double>(m);
    o.zipf = (dm * sxy - sx * sy) / (dm * sxx - sx * sx);
  }
  o.ks = ks_lengths(set, ref_pmf);
  return o;
}

struct SetReference {
  std::vector<double> pmf;  // human sentence-length pmf
  Ecdf zipf, ks;            // human sets of the same size; ks holds -distance, so higher is better
  // E in [0,1]: geometric mean of the Zipf score (two-sided) and the length score (one-sided).
  double E(const SetValues& v, double* e1s = nullptr, double* e3s = nullptr) const {
    const double a = zipf.two_sided(v.zipf), b = ks.one_sided(-v.ks);
    if (e1s) *e1s = a;
    if (e3s) *e3s = b;
    return geo_mean({a, b});
  }
};

// Null for E: n_sets human sets of set_size samples. `stretches` are many disjoint stretches of
// the human corpus (the caller cuts [16, 56) MiB of enwik8 into about two dozen), each a pool of
// samples of the length bucket. A null set picks about a third of the stretches at random, pools
// them and draws set_size samples from the pool without replacement. So the null carries the
// variation between stretches and, because the pool is much larger than a set, almost none of the
// finite-population shrinkage that made v0's null too narrow (it drew 500 of ~1,900 samples).
inline SetReference build_set_reference(const std::vector<std::vector<const Features*>>& stretches,
                                        const std::vector<double>& ref_pmf, size_t set_size, size_t dict_size,
                                        int n_sets = 300, std::uint64_t seed = 7) {
  SetReference r;
  r.pmf = ref_pmf;
  std::mt19937_64 rng(seed);
  std::vector<std::uint32_t> scratch(dict_size, 0);
  const size_t G = stretches.size();
  const size_t take = std::max<size_t>(1, G / 3);
  std::vector<size_t> order(G);
  for (int s = 0; s < n_sets; ++s) {
    std::iota(order.begin(), order.end(), size_t{0});
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<const Features*> pool;
    for (size_t g = 0; g < take && g < G; ++g) pool.insert(pool.end(), stretches[order[g]].begin(), stretches[order[g]].end());
    std::shuffle(pool.begin(), pool.end(), rng);
    if (pool.size() > set_size) pool.resize(set_size);
    const SetValues v = set_values(pool, r.pmf, scratch);
    r.zipf.add(v.zipf);
    r.ks.add(-v.ks);
  }
  r.zipf.finish();
  r.ks.finish();
  return r;
}

// ===========================================================================
// v2: plain-text documents, continuations, and the human-typicality deviance
// ===========================================================================
//
// v2 keeps v1's statistics, adds a few (s2, s3min, sentence lengths, word rarity, adjacent cohesion, verbatim copy,
// and the prompted ones r1, r4, r5, i4), and replaces v1's percentile scores, geometric means and caps by one rule:
//   1. Every raw statistic x becomes a signed z-score against the human reference samples of the same length bucket:
//      inside the 2nd-98th percentile z = Phi^-1(F(x)) with F the mid-rank ECDF; beyond it the tail is extended
//      linearly with a scale taken from the outer quantiles (so a value far outside the human range keeps its distance
//      instead of saturating at the ECDF's last step). A statistic that human samples almost always have (>= 95%) and
//      this sample lacks counts as z = 4 on its bad side.
//   2. Only the bad side counts: u = max(0, -z) for higher-is-better statistics, max(0, z) for lower-is-better, |z| for
//      two-sided ones. A dead zone d (1; 0.5 for relevance) forgives the ordinary human spread: e = max(0, u - d).
//   3. The evidence of each axis is summed (L, S, C, I, R), and
//        D = capS*sat(L - thL) + capS*sat(S - thS) + wI*max(0, I - thI) + wC*C + wR*R,  sat(x) = 1 - exp(-max(0, x)/kapS),
//      with thL, thS, thI the qS/qI quantiles of the reference samples' own L, S, I in the bucket. Form failures are
//      categorical and saturate; degeneration counts beyond the human quantile; coherence is graded. No gates: a
//      deviance can only add, so no axis can rescue another (PLAN 6.3's intent).
//   4. SCRIBE_x = exp(-D / tau): 1 for a sample that is human-typical on every statistic, falling with the evidence
//      that it is not. Pairwise orderings depend only on D.

inline std::string collapse_newlines(std::string s) {
  for (char& c : s)
    if (c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v') c = ' ';
  return s;
}

namespace scribe_detail {

// True if x ends with . ! or ? followed only by closing quotes or brackets (ASCII or the UTF-8 right quotes).
inline bool ends_terminal(const std::string& x) {
  size_t e = x.size();
  while (e > 0) {
    if (x[e - 1] == '"' || x[e - 1] == '\'' || x[e - 1] == ')' || x[e - 1] == ']') {
      --e;
    } else if (e >= 3 && (x.compare(e - 3, 3, "\xE2\x80\x9D") == 0 || x.compare(e - 3, 3, "\xE2\x80\x99") == 0)) {
      e -= 3;
    } else {
      break;
    }
  }
  return e > 0 && (x[e - 1] == '.' || x[e - 1] == '!' || x[e - 1] == '?');
}

// A prose line of a plain-text document: at least 20 bytes, starts like a sentence, ends like one, no markup.
inline bool prose_plain_line(const std::string& l) {
  if (l.size() < 20) return false;
  size_t b = 0;
  while (b < l.size() && (l[b] == '"' || l[b] == '\'' || l[b] == '(')) ++b;
  if (b >= l.size()) return false;
  const unsigned char f = static_cast<unsigned char>(l[b]);
  if (!(is_upper_a(f) || is_digit_a(f) || f >= 0xC0)) return false;
  if (!ends_terminal(l)) return false;
  for (char c : l)
    if (c == '{' || c == '}' || c == '[' || c == ']' || c == '|' || c == '<' || c == '>' || c == '=' || c == '*' || c == '#' || c == '_' ||
        c == '\\' || c == '~')
      return false;
  if (l.find("http") != std::string::npos || l.find("www.") != std::string::npos) return false;
  return true;
}

// Inverse standard normal CDF (Acklam's rational approximation, relative error < 1.2e-9).
inline double inv_phi(double p) {
  if (p <= 0) return -40;
  if (p >= 1) return 40;
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
                             -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
  const double plow = 0.02425;
  if (p < plow) {
    const double q = std::sqrt(-2 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - plow) {
    const double q = std::sqrt(-2 * std::log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const double q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

inline std::uint64_t mix64(std::uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return x;
}

}  // namespace scribe_detail

// A plain-text document (web page, book chunk, story) to prose: each line is collapsed and kept only if it is a
// prose line (prose_plain_line); kept lines are joined with spaces. The counterpart of clean_wiki_body.
inline std::string clean_plain_document(const std::string& text) {
  using namespace scribe_detail;
  std::string doc;
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t e = text.find('\n', pos);
    if (e == std::string::npos) e = text.size();
    const std::string line = collapse_ws(collapse_newlines(text.substr(pos, e - pos)));
    size_t b = 0;
    while (b < line.size() && line[b] == ' ') ++b;
    const std::string l = line.substr(b);
    if (prose_plain_line(l)) {
      if (!doc.empty()) doc.push_back(' ');
      doc += l;
    }
    if (e == text.size()) break;
    pos = e + 1;
  }
  return doc;
}

// A model's continuation of a prompt (the prompt itself excluded), prepared for scoring: white space collapsed,
// and an unterminated last sentence dropped when at least one whole sentence comes before it. A text without any
// sentence boundary is kept whole, so the missing boundary is scored rather than hidden.
inline std::string prepare_continuation(const std::string& raw) {
  using namespace scribe_detail;
  std::string t = collapse_ws(collapse_newlines(raw));
  size_t b = 0;
  while (b < t.size() && t[b] == ' ') ++b;
  t = t.substr(b);
  std::vector<std::string> s = split_sentences(t);
  if (s.size() >= 2 && !ends_terminal(s.back())) s.pop_back();
  std::string o;
  for (const std::string& x : s) {
    if (!o.empty()) o.push_back(' ');
    o += x;
  }
  return o;
}

// ---------------------------------------------------------------------------
// The v2 statistics
// ---------------------------------------------------------------------------

enum V2Stat {
  V_S1, V_S2, V_S2G, V_S3, V_S3G, V_S3MIN, V_S4, V_SLMEAN, V_SLMAX, V_WFREQ,  // S: form
  V_C0, V_C1, V_C1A, V_C2, V_C3,                                // C: coherence
  V_LOOP, V_REP, V_D1, V_D2, V_D3, V_D4, V_I1,                  // I: integrity
  V_R1, V_R4, V_R5, V_I4,                                       // R: relevance (prompted only)
  V_N
};

struct V2StatSpec {
  const char* name;
  char axis;      // 'L' lexical validity, 'S' syntax, 'C' coherence, 'I' integrity, 'R' relevance
  int side;       // +1 higher is better, -1 lower is better, 0 two-sided
  double min_sd;  // floor of the tail scale, in the statistic's units
  bool used;      // counts in the deviance (false: computed and reported only)
};

inline const V2StatSpec& v2_spec(int k) {
  // Coherence is one-sided in v2 (more cohesion and less drift are better): too much cohesion from repetition is
  // the I axis's job, and two-sided C cost 0.02-0.03 of ordering accuracy on the development ladders. s4's floor is
  // 0.15 (about one failed check of six), so one unbalanced quote is moderate evidence, not z = -5.
  static const V2StatSpec specs[V_N] = {
      {"s1", 'L', +1, 0.01, true},      {"s2", 'S', +1, 0.05, false},    {"s2g", 'S', +1, 0.05, true},
      {"s3", 'S', +1, 0.05, false},     {"s3g", 'S', +1, 0.05, true},
      {"s3min", 'S', +1, 0.10, false},  {"s4", 'S', +1, 0.15, true},     {"sl_mean", 'S', 0, 0.5, false},
      {"sl_max", 'S', -1, 1.0, false},  {"wfreq", 'S', 0, 0.05, false},  {"c0", 'C', +1, 0.005, true},
      {"c1", 'C', +1, 0.01, true},      {"c1a", 'C', +1, 0.01, true},    {"c2", 'C', +1, 0.005, true},
      {"c3", 'C', -1, 0.01, true},      {"loop8", 'I', -1, 0.01, true},  {"rep", 'I', -1, 0.05, true},
      {"d1", 'I', 0, 0.01, true},       {"d2", 'I', 0, 0.01, true},      {"d3", 'I', 0, 0.005, true},
      {"d4", 'I', 0, 0.005, true},      {"i1", 'I', 0, 0.01, true},      {"r1", 'R', +1, 0.01, true},
      {"r4", 'R', +1, 0.02, true},      {"r5", 'R', 0, 0.02, true},      {"i4", 'R', -1, 0.01, true}};
  return specs[k];
}

// The raw value of each v2 statistic and whether the sample has it.
inline void v2_values(const Features& f, double* x, bool* has) {
  for (int k = 0; k < V_N; ++k) has[k] = true;
  x[V_S1] = f.s1;
  x[V_S2] = f.s2;
  x[V_S3] = f.s3;
  x[V_S2G] = f.s2g;
  x[V_S3G] = f.s3g;
  has[V_S3G] = f.has_s3g;
  x[V_S3MIN] = f.s3min;
  x[V_S4] = f.s4;
  x[V_SLMEAN] = f.sl_mean;
  x[V_SLMAX] = f.sl_max;
  x[V_WFREQ] = f.wfreq;
  x[V_C0] = f.c0;
  has[V_C0] = f.has_c0;
  x[V_C1] = f.c1;
  has[V_C1] = f.has_c1;
  x[V_C1A] = f.c1a;
  has[V_C1A] = f.has_c1a;
  x[V_C2] = f.c2;
  has[V_C2] = f.has_c2;
  x[V_C3] = f.c3;
  has[V_C3] = f.has_c3;
  x[V_LOOP] = f.loop8;
  x[V_REP] = f.rep;
  for (int n = 0; n < 4; ++n) x[V_D1 + n] = f.d[n];
  x[V_I1] = f.i1;
  x[V_R1] = f.r1;
  x[V_R4] = f.r4;
  x[V_R5] = f.r5;
  x[V_I4] = f.i4;
  for (int k : {V_R1, V_R4, V_R5, V_I4}) has[k] = f.has_r;
  if (!f.has_v2)
    for (int k : {V_S2, V_S2G, V_WFREQ, V_I1}) has[k] = false;
  if (f.n_words == 0)
    for (int k = 0; k < V_N; ++k) has[k] = has[k] && (k == V_S4);
}

// Signed robust z-score of x against a human sample (see the v2 rule above).
class QuantileZ {
 public:
  void add(double x) { v_.push_back(x); }
  size_t size() const { return v_.size(); }
  void finish(double min_sd) {
    std::sort(v_.begin(), v_.end());
    if (v_.empty()) return;
    q02_ = quantile(0.02);
    q16_ = quantile(0.16);
    q84_ = quantile(0.84);
    q98_ = quantile(0.98);
    const double span = 2.053748910631823 - 0.994457883209753;
    sl_ = std::max(min_sd, (q16_ - q02_) / span);
    su_ = std::max(min_sd, (q98_ - q84_) / span);
  }
  double z(double x) const {
    if (v_.empty()) return 0;
    if (x < q02_) return -2.053748910631823 - (q02_ - x) / sl_;
    if (x > q98_) return 2.053748910631823 + (x - q98_) / su_;
    const auto lo = std::lower_bound(v_.begin(), v_.end(), x);
    const auto hi = std::upper_bound(v_.begin(), v_.end(), x);
    const double F = (static_cast<double>(lo - v_.begin()) + 0.5 * static_cast<double>(hi - lo) + 0.5) / (static_cast<double>(v_.size()) + 1.0);
    return std::max(-2.053748910631823, std::min(2.053748910631823, scribe_detail::inv_phi(F)));
  }

 private:
  double quantile(double p) const {
    const double h = p * static_cast<double>(v_.size() - 1);
    const size_t i = static_cast<size_t>(h);
    const double fr = h - static_cast<double>(i);
    return i + 1 < v_.size() ? v_[i] * (1 - fr) + v_[i + 1] * fr : v_.back();
  }
  std::vector<double> v_;
  double q02_ = 0, q16_ = 0, q84_ = 0, q98_ = 0, sl_ = 1, su_ = 1;
};

// The v2 combination (chosen on the development ladders A and B only; docs/reports/SCRIBE_V2.md):
//   u_k = bad-side |z_k|, e_k = max(0, u_k - dead) for every used statistic k
//   L, S, C, I, R = sums of e_k over the lexical (s1), syntax (s2g, s3g, s4), coherence (c0, c1, c1a, c2, c3),
//                   integrity (loop8, rep, d1-d4, i1) and relevance (r1, r4, r5, i4; prompted only) statistics
//   D = cap * sat(L - thL) + cap * sat(S - thS) + wI * max(0, I - thI) + wC * C + wR * R,  sat(x) = 1 - exp(-max(0, x) / kap)
//   thL, thS, thI = the q quantiles of L, S and I over the human reference samples of the length bucket
//   SCRIBE = exp(-D / tau)
// Prompted profile (v2p): wR = 8, with relevance dead zone 0.5 (rdead); chosen on development prompts only.
// Not being made of words (L) and not being in an order people write (S) are two separate, categorical form
// failures: each counts only beyond what 99% of human samples show, and each saturates at cap, so samples that fail
// in the same way are ordered by their coherence. Degeneration (I) is the worst failure and does not saturate.
// Coherence is graded: less cohesive is worse at every level, so it counts from the dead zone on.
struct V2Params {
  double dead = 1.0;
  double qS = 0.99, qI = 0.99;  // qS is used for both L and S
  double capS = 30, kapS = 4;
  double wC = 1, wI = 3, wR = 0;  // wR = 8 in the prompted profile (v2p)
  double rdead = 0.5;            // relevance dead zone (chosen on the development prompts)
  bool two_sided = false;        // exploratory, post hoc (not v2): syntax gains and coherence also penalised when higher than human
  double tau = 5;
  double missing_z = 4;
  std::vector<int> off;  // statistics switched off (ablations)
  std::vector<int> on;   // reported-only statistics switched on (experiments)
  // "dead=1,qS=0.95,qI=0.98,capS=30,kapS=8,wC=1,wI=1,wR=0,tau=5,off=s2+i1,on=wfreq"
  bool parse(const std::string& spec) {
    size_t p = 0;
    while (p < spec.size()) {
      size_t e = spec.find(',', p);
      if (e == std::string::npos) e = spec.size();
      const std::string kv = spec.substr(p, e - p);
      const size_t eq = kv.find('=');
      if (eq == std::string::npos) return false;
      const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
      const double x = std::strtod(v.c_str(), nullptr);
      if (k == "dead") dead = x;
      else if (k == "qS") qS = x;
      else if (k == "qI") qI = x;
      else if (k == "capS") capS = x;
      else if (k == "kapS") kapS = x;
      else if (k == "wC") wC = x;
      else if (k == "wI") wI = x;
      else if (k == "wR") wR = x;
      else if (k == "rdead") rdead = x;
      else if (k == "twosided") two_sided = x != 0;
      else if (k == "tau") tau = x;
      else if (k == "missing") missing_z = x;
      else if (k == "off" || k == "on") {
        std::vector<int>& dst = k == "off" ? off : on;
        size_t q = 0;
        while (q < v.size()) {
          size_t r = v.find('+', q);
          if (r == std::string::npos) r = v.size();
          const std::string nm = v.substr(q, r - q);
          bool found = false;
          for (int st = 0; st < V_N; ++st)
            if (nm == v2_spec(st).name) {
              dst.push_back(st);
              found = true;
            }
          if (!found) return false;
          q = r + 1;
        }
      } else {
        return false;
      }
      p = e + 1;
    }
    return true;
  }
  bool counts(int k) const {
    if (std::find(off.begin(), off.end(), k) != off.end()) return false;
    return v2_spec(k).used || std::find(on.begin(), on.end(), k) != on.end();
  }
};

struct ScoredV2 {
  double z[V_N] = {};
  bool has[V_N] = {};
  bool imputed[V_N] = {};  // missing where humans almost always have it: counted at missing_z
  double L = 0, S = 0, C = 0, I = 0, R = 0;      // raw axis evidence (sums of e_k)
  double D = 0, DL = 0, DS = 0, DC = 0, DI = 0, DR = 0;  // the axis terms of D, and D
  double scribe = 0;
};

class ScribeV2Scorer {
 public:
  V2Params params;

  // Trains what v2 adds on top of ScribeModel: the skeleton trigram (function words kept, every other word mapped to
  // a suffix class), unigram counts for word rarity, the function-word list, and the set of word 8-grams for I1.
  void train_pack(const std::vector<std::string>& docs, size_t n_function_words = 150) {
    std::unordered_map<std::string, std::uint64_t> cnt;
    cnt.reserve(1u << 20);
    std::vector<std::string> toks;
    std::vector<std::vector<std::string>> sents;
    total_ = 0;
    for (const std::string& d : docs)
      for (const std::string& s : split_sentences(d)) {
        toks.clear();
        tokenize_words(s, toks);
        if (toks.empty()) continue;
        for (const std::string& t : toks)
          if (!numeric_token(t)) {
            ++cnt[t];
            ++total_;
          }
        sents.push_back(toks);
      }
    std::vector<std::pair<std::uint64_t, std::string>> byc;
    byc.reserve(cnt.size());
    for (auto& kv : cnt) byc.emplace_back(kv.second, kv.first);
    std::sort(byc.begin(), byc.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    fw_.clear();
    for (size_t i = 0; i < byc.size() && fw_.size() < n_function_words; ++i) fw_.insert(byc[i].second);
    uni_.clear();
    uni_.reserve(cnt.size());
    for (auto& kv : cnt)
      if (kv.second >= 2) uni_.emplace(kv.first, std::log(static_cast<double>(kv.second) / static_cast<double>(total_)));
    std::vector<std::string> mapped;
    mapped.reserve(sents.size());
    grams8_.clear();
    for (const auto& t : sents) {
      std::string m;
      for (const std::string& w : t) {
        if (!m.empty()) m.push_back(' ');
        m += skel_token(w);
      }
      mapped.push_back(std::move(m));
    }
    std::vector<std::string> all;
    for (const auto& t : sents) all.insert(all.end(), t.begin(), t.end());
    std::vector<std::vector<std::string>>().swap(sents);
    const std::vector<std::uint64_t> h = token_hashes(all);
    std::vector<std::string>().swap(all);
    grams8_.reserve(h.size());
    for (size_t i = 0; i + 8 <= h.size(); ++i) grams8_.push_back(gram_key(h, i));
    std::sort(grams8_.begin(), grams8_.end());
    grams8_.erase(std::unique(grams8_.begin(), grams8_.end()), grams8_.end());
    skel_.train(mapped, 1);
    trained_ = true;
  }
  bool trained() const { return trained_; }
  size_t n_grams8() const { return grams8_.size(); }

  // Adds the v2 statistics that need the pack's v2 models (and the prompt, when there is one) to f.
  void extend(Features& f, const std::string& text, const std::string& prompt, const SentenceEmbedder* emb) const {
    if (!trained_) return;
    f.has_v2 = true;
    std::vector<std::vector<std::string>> st = sentence_tokens(text);
    std::vector<std::string> flat;
    double lp = 0, wsum = 0, sg = 0;
    size_t npred = 0, nw = 0, sgn = 0;
    for (const auto& t : st) {
      std::vector<std::uint32_t> ids;
      ids.reserve(t.size());
      for (const std::string& w : t) {
        ids.push_back(skel_.word_id(skel_token(w)));
        flat.push_back(w);
        const auto it = uni_.find(w);
        if (it != uni_.end()) {
          wsum += it->second;
          ++nw;
        }
      }
      lp += skel_.sentence_logp(ids, npred);
      sg += skel_.sentence_gain(ids, sgn);
    }
    f.s2 = npred ? lp / static_cast<double>(npred) : -30.0;
    f.s2g = sgn ? sg / static_cast<double>(sgn) : 0.0;
    f.wfreq = nw ? wsum / static_cast<double>(nw) : -20.0;
    // I1: share of tokens covered by a word 8-gram that occurs in the pack.
    if (flat.size() >= 8) {
      const std::vector<std::uint64_t> h = token_hashes(flat);
      std::vector<char> cov(flat.size(), 0);
      for (size_t i = 0; i + 8 <= h.size(); ++i)
        if (std::binary_search(grams8_.begin(), grams8_.end(), gram_key(h, i)))
          for (size_t k = i; k < i + 8; ++k) cov[k] = 1;
      size_t c = 0;
      for (char x : cov) c += static_cast<size_t>(x);
      f.i1 = static_cast<double>(c) / static_cast<double>(flat.size());
    }
    // Prompted statistics.
    if (!prompt.empty()) {
      std::vector<std::string> ptoks;
      for (const auto& t : sentence_tokens(prompt)) ptoks.insert(ptoks.end(), t.begin(), t.end());
      if (!ptoks.empty() && !flat.empty()) {
        f.has_prompt = true;
        f.has_r = true;
        std::unordered_set<std::string> pset(ptoks.begin(), ptoks.end()), tset(flat.begin(), flat.end());
        // R4: rarity-weighted share of the prompt's content words that the text uses.
        double num = 0, den = 0;
        std::unordered_set<std::string> seenp;
        for (const std::string& w : ptoks) {
          if (!content_word(w) || !seenp.insert(w).second) continue;
          const double wt = rarity(w);
          den += wt;
          if (tset.count(w)) num += wt;
        }
        f.r4 = den > 0 ? num / den : 0.0;
        // R5: share of the text's content tokens that are not in the prompt.
        size_t ct = 0, nov = 0;
        for (const std::string& w : flat)
          if (content_word(w)) {
            ++ct;
            nov += pset.count(w) ? 0 : 1;
          }
        f.r5 = ct ? static_cast<double>(nov) / static_cast<double>(ct) : 1.0;
        // I4: share of the text's word 4-grams that occur in the prompt.
        if (flat.size() >= 4 && ptoks.size() >= 4) {
          std::unordered_set<std::string> p4;
          for (size_t i = 0; i + 4 <= ptoks.size(); ++i) p4.insert(ptoks[i] + " " + ptoks[i + 1] + " " + ptoks[i + 2] + " " + ptoks[i + 3]);
          size_t hit = 0, tot = 0;
          for (size_t i = 0; i + 4 <= flat.size(); ++i, ++tot)
            hit += p4.count(flat[i] + " " + flat[i + 1] + " " + flat[i + 2] + " " + flat[i + 3]);
          f.i4 = tot ? static_cast<double>(hit) / static_cast<double>(tot) : 0.0;
        }
        // R1: cosine of the SIF vectors of the prompt and the text.
        if (emb != nullptr && emb->ready()) {
          std::vector<float> a, b;
          if (emb->embed(ptoks, a, 1) && emb->embed(flat, b, 1)) {
            double dot = 0;
            for (size_t k = 0; k < a.size(); ++k) dot += double(a[k]) * b[k];
            f.r1 = dot;
          }
        }
      }
    }
  }

  void add_reference(const Features& f) {
    const int b = length_bucket(f.bytes);
    double x[V_N];
    bool has[V_N];
    v2_values(f, x, has);
    ++nref_[b];
    for (int k = 0; k < V_N; ++k) {
      if (!has[k]) continue;
      ref_[b][k].add(x[k]);
      ++present_[b][k];
    }
    ref_feats_[b].push_back(f);
  }
  // A human continuation of a prompt: adds only the relevance statistics (r1, r4, r5, i4) to the reference.
  void add_reference_prompted(const Features& f) {
    if (!f.has_r) return;
    const int b = length_bucket(f.bytes);
    double x[V_N];
    bool has[V_N];
    v2_values(f, x, has);
    for (int k : {V_R1, V_R4, V_R5, V_I4})
      if (has[k]) ref_[b][k].add(x[k]);
    ++nref_prompted_[b];
  }
  size_t n_reference_prompted(int b) const { return nref_prompted_[b]; }
  // Finishes the z-transforms, then the thresholds thS and thI from the reference samples' own S and I.
  void finish() {
    for (int b = 0; b < 2; ++b)
      for (int k = 0; k < V_N; ++k) ref_[b][k].finish(v2_spec(k).min_sd);
    for (int b = 0; b < 2; ++b) {
      std::vector<double> L, S, I;
      for (const Features& f : ref_feats_[b]) {
        const ScoredV2 o = evidence(f);
        L.push_back(o.L);
        S.push_back(o.S);
        I.push_back(o.I);
      }
      thL_[b] = quantile_of(L, params.qS);
      thS_[b] = quantile_of(S, params.qS);
      thI_[b] = quantile_of(I, params.qI);
    }
  }
  double threshold_L(int b) const { return thL_[b]; }
  double threshold_S(int b) const { return thS_[b]; }
  double threshold_I(int b) const { return thI_[b]; }

  // The reference bucket for a sample: its length bucket, or the other one when that bucket has under 20 samples.
  int bucket(const Features& f) const {
    const int b = length_bucket(f.bytes);
    return nref_[b] < 20 && nref_[1 - b] >= 20 ? 1 - b : b;
  }

  ScoredV2 score(const Features& f) const {
    ScoredV2 o = evidence(f);
    const int b = bucket(f);
    o.DL = params.capS * (1.0 - std::exp(-std::max(0.0, o.L - thL_[b]) / params.kapS));
    o.DS = params.capS * (1.0 - std::exp(-std::max(0.0, o.S - thS_[b]) / params.kapS));
    o.DI = params.wI * std::max(0.0, o.I - thI_[b]);
    o.DC = params.wC * o.C;
    o.DR = params.wR * o.R;
    o.D = o.DL + o.DS + o.DI + o.DC + o.DR;
    o.scribe = std::exp(-o.D / params.tau);
    return o;
  }

  // z-scores and raw axis evidence (no thresholds).
  ScoredV2 evidence(const Features& f) const {
    ScoredV2 o;
    const int b = bucket(f);
    double x[V_N];
    bool has[V_N];
    v2_values(f, x, has);
    double ax[5] = {0, 0, 0, 0, 0};
    for (int k = 0; k < V_N; ++k) {
      const V2StatSpec& sp = v2_spec(k);
      const int a = sp.axis == 'S' ? 0 : sp.axis == 'C' ? 1 : sp.axis == 'I' ? 2 : sp.axis == 'R' ? 3 : 4;
      if (a == 3 && !f.has_prompt) continue;
      if (ref_[b][k].size() < 20) continue;  // the reference does not have it: not scored
      double u;
      if (has[k]) {
        o.has[k] = true;
        o.z[k] = ref_[b][k].z(x[k]);
        const bool ts = params.two_sided && (k == V_S2G || k == V_S3G || a == 1);
        u = ts ? std::fabs(o.z[k]) : sp.side > 0 ? std::max(0.0, -o.z[k]) : sp.side < 0 ? std::max(0.0, o.z[k]) : std::fabs(o.z[k]);
      } else if (a != 3 && present_[b][k] >= 0.95 * static_cast<double>(nref_[b])) {
        o.imputed[k] = true;
        o.z[k] = sp.side > 0 ? -params.missing_z : params.missing_z;
        u = params.missing_z;
      } else {
        continue;
      }
      if (params.counts(k)) ax[a] += std::max(0.0, u - (a == 3 ? params.rdead : params.dead));
    }
    o.S = ax[0];
    o.C = ax[1];
    o.I = ax[2];
    o.R = ax[3];
    o.L = ax[4];
    return o;
  }

  // Skeleton token: a function word stays itself; any other word becomes its class.
  std::string skel_token(const std::string& w) const {
    if (numeric_token(w)) return "zznum";
    if (fw_.count(w)) return w;
    for (unsigned char c : w)
      if (!(c >= 'a' && c <= 'z') && c != '\'') return "zzother";
    auto ends = [&](const char* suf) {
      const size_t L = std::strlen(suf);
      return w.size() > L + 1 && w.compare(w.size() - L, L, suf) == 0;
    };
    if (ends("ing")) return "zzing";
    if (ends("ed")) return "zzed";
    if (ends("ly")) return "zzly";
    if (ends("tion") || ends("sion") || ends("ment") || ends("ness") || ends("ity") || ends("ism")) return "zznom";
    if (ends("al") || ends("ive") || ends("ous") || ends("ful") || ends("able") || ends("ible") || ends("ic") || ends("ish")) return "zzadj";
    if (ends("er") || ends("or")) return "zzer";
    if (ends("s") && !ends("ss")) return "zzs";
    return "zzw";
  }
  bool content_word(const std::string& w) const {
    if (w.size() < 2 || fw_.count(w) || numeric_token(w)) return false;
    return true;
  }
  double rarity(const std::string& w) const {
    const auto it = uni_.find(w);
    return it == uni_.end() ? std::log(static_cast<double>(std::max<std::uint64_t>(2, total_))) : -it->second;
  }

 private:
  static std::vector<std::uint64_t> token_hashes(const std::vector<std::string>& t) {
    std::vector<std::uint64_t> h(t.size());
    for (size_t i = 0; i < t.size(); ++i) h[i] = scribe_detail::fnv1a(t[i]);
    return h;
  }
  static std::uint64_t gram_key(const std::vector<std::uint64_t>& h, size_t i) {
    std::uint64_t k = 0x9e3779b97f4a7c15ull;
    for (size_t j = 0; j < 8; ++j) k = scribe_detail::mix64(k ^ (h[i + j] + 0x9e3779b97f4a7c15ull * (j + 1)));
    return k;
  }
  bool trained_ = false;
  ScribeModel skel_;
  std::unordered_set<std::string> fw_;
  std::unordered_map<std::string, double> uni_;
  std::uint64_t total_ = 0;
  std::vector<std::uint64_t> grams8_;
  QuantileZ ref_[2][V_N];
  double present_[2][V_N] = {};
  size_t nref_[2] = {0, 0}, nref_prompted_[2] = {0, 0};
  std::vector<Features> ref_feats_[2];
  double thL_[2] = {0, 0}, thS_[2] = {0, 0}, thI_[2] = {0, 0};
  static double quantile_of(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const double h = p * static_cast<double>(v.size() - 1);
    const size_t i = static_cast<size_t>(h);
    const double fr = h - static_cast<double>(i);
    return i + 1 < v.size() ? v[i] * (1 - fr) + v[i + 1] * fr : v.back();
  }
};

// ---------------------------------------------------------------------------
// JSON and Markdown helpers for the tool
// ---------------------------------------------------------------------------

inline std::string scribe_num(double v, int prec = 6) {
  if (!std::isfinite(v)) return "null";
  char b[64];
  std::snprintf(b, sizeof b, "%.*g", prec, v);
  return b;
}

inline std::string features_json(const Features& f) {
  double x[V_N];
  bool has[V_N];
  v2_values(f, x, has);
  std::string o = "{\"bytes\":" + std::to_string(f.bytes) + ",\"n_sent\":" + std::to_string(f.n_sent) + ",\"n_words\":" + std::to_string(f.n_words);
  for (int k = 0; k < V_N; ++k) o += ",\"" + std::string(v2_spec(k).name) + "\":" + (has[k] ? scribe_num(x[k]) : std::string("null"));
  o += ",\"has_prompt\":" + std::string(f.has_prompt ? "1" : "0") + "}";
  return o;
}

inline std::string v2_json(const ScoredV2& s) {
  std::string o = "{\"scribe\":" + scribe_num(s.scribe) + ",\"D\":" + scribe_num(s.D) + ",\"DL\":" + scribe_num(s.DL) + ",\"DS\":" + scribe_num(s.DS) + ",\"DC\":" + scribe_num(s.DC) +
                  ",\"DI\":" + scribe_num(s.DI) + ",\"DR\":" + scribe_num(s.DR) + ",\"L\":" + scribe_num(s.L) + ",\"S\":" + scribe_num(s.S) + ",\"C\":" + scribe_num(s.C) +
                  ",\"I\":" + scribe_num(s.I) + ",\"R\":" + scribe_num(s.R) + ",\"z\":{";
  bool first = true;
  for (int k = 0; k < V_N; ++k) {
    if (!s.has[k] && !s.imputed[k]) continue;
    if (!first) o += ",";
    first = false;
    o += "\"" + std::string(v2_spec(k).name) + "\":" + scribe_num(s.z[k], 4);
  }
  o += "}}";
  return o;
}

inline std::string v2_profile_header() {
  return "| rung | n | SCRIBE | mean D | D_L | D_S | D_C | D_I | D_R | share D_L > 0 | share D_S > 0 | share D_I > 0 |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n";
}
inline std::string v2_profile_row(const std::string& name, const std::vector<ScoredV2>& v) {
  if (v.empty()) return "";
  double sc = 0, D = 0, l = 0, a = 0, b = 0, c = 0, d = 0, pl = 0, pa = 0, pc = 0;
  for (const ScoredV2& s : v) {
    sc += s.scribe;
    D += s.D;
    l += s.DL;
    a += s.DS;
    b += s.DC;
    c += s.DI;
    d += s.DR;
    pl += s.DL > 0;
    pa += s.DS > 0;
    pc += s.DI > 0;
  }
  const double n = static_cast<double>(v.size());
  char buf[512];
  std::snprintf(buf, sizeof buf, "| %s | %zu | %.3f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.0f%% | %.0f%% | %.0f%% |\n", name.c_str(), v.size(), sc / n, D / n,
                l / n, a / n, b / n, c / n, d / n, 100 * pl / n, 100 * pa / n, 100 * pc / n);
  return buf;
}

// Known-answer checks of the v2 parts (called by the tool's selftest).
inline void scribe_v2_selftest(int& checks, int& fails) {
  auto check = [&](bool ok, const char* what) {
    ++checks;
    if (!ok) {
      ++fails;
      std::printf("FAIL v2: %s\n", what);
    }
  };
  using scribe_detail::inv_phi;
  check(std::fabs(inv_phi(0.5)) < 1e-9, "inv_phi(0.5) = 0");
  check(std::fabs(inv_phi(0.975) - 1.959963985) < 1e-6, "inv_phi(0.975)");
  check(std::fabs(inv_phi(0.001) + 3.090232306) < 1e-6, "inv_phi(0.001)");
  // QuantileZ on a standard normal sample: z(x) close to x inside, linear outside.
  {
    QuantileZ q;
    std::mt19937_64 r(7);
    std::normal_distribution<double> N(0.0, 1.0);
    for (int i = 0; i < 20000; ++i) q.add(N(r));
    q.finish(0.01);
    check(std::fabs(q.z(0.0)) < 0.05 && std::fabs(q.z(1.0) - 1.0) < 0.06, "QuantileZ centre");
    check(std::fabs(q.z(-5.0) + 5.0) < 0.25 && std::fabs(q.z(6.0) - 6.0) < 0.3, "QuantileZ tails extend linearly");
    check(q.z(-50) < q.z(-10) && q.z(50) > q.z(10), "QuantileZ monotone far out");
  }
  // A constant human statistic: anything else is far out, on the right side.
  {
    QuantileZ q;
    for (int i = 0; i < 100; ++i) q.add(1.0);
    q.finish(0.05);
    check(q.z(0.5) < -9 && q.z(1.0) > -2.1 && q.z(1.0) < 2.1, "QuantileZ with ties");
  }
  check(prepare_continuation("A big cat sat. Dogs ran to") == "A big cat sat.", "prepare_continuation drops the cut tail");
  check(clean_plain_document("Title line\nThis is a real sentence of prose here.\n* list item that is long enough.\nAnother full sentence follows it.") ==
            "This is a real sentence of prose here. Another full sentence follows it.",
        "clean_plain_document keeps prose lines");
  check(scribe_detail::ends_terminal("He left.\xE2\x80\x9D") && !scribe_detail::ends_terminal("He left"), "ends_terminal");
  // Skeleton, rarity, copy detection and prompted statistics on a toy pack.
  {
    std::vector<std::string> docs;
    for (int i = 0; i < 40; ++i)
      docs.push_back("The quick brown fox jumps over the lazy dog near the river bank today. The old man was walking slowly along the road. "
                     "Green trees grow beside the quiet water.");
    ScribeV2Scorer v;
    v.train_pack(docs, 10);
    check(v.trained() && v.n_grams8() > 10, "v2 pack trained");
    check(v.skel_token("walking") == "zzing" && v.skel_token("1990") == "zznum" && v.skel_token("the") == "the", "skeleton tokens");
    Features f;
    v.extend(f, "The quick brown fox jumps over the lazy dog near the river bank today.", "", nullptr);
    check(f.has_v2 && f.i1 > 0.99, "copied pack text has i1 = 1");
    Features g;
    v.extend(g, "Purple elephants calculate distant galaxies with patient enthusiasm forever.", "", nullptr);
    check(g.i1 < 0.01 && g.s2 < f.s2, "new text: no copy, less typical skeleton");
    Features h;
    v.extend(h, "The old man was walking slowly along the road near the river.", "The old man near the river.", nullptr);
    check(h.has_r && h.r4 > 0.99 && h.r5 < 0.8, "prompted coverage and informativeness");
  }
  // The deviance: a human-typical sample scores near 1, one far-out statistic lowers it, ablation removes it.
  {
    ScribeV2Scorer v;
    std::mt19937_64 r(3);
    std::normal_distribution<double> N(0.0, 1.0);
    auto mk = [&](double shift) {
      Features f;
      f.bytes = 500;
      f.n_words = 90;
      f.n_sent = 4;
      f.s1 = 0.95 + 0.01 * N(r);
      f.s3 = -6 + 0.3 * N(r);
      f.s3min = -7 + 0.4 * N(r);
      f.s4 = 1.0;
      f.sl_mean = 22 + 3 * N(r);
      f.sl_max = 30 + 4 * N(r);
      f.has_c = f.has_c0 = f.has_c1 = f.has_c1a = f.has_c3 = true;
      f.c0 = 0.1 + 0.02 * N(r);
      f.c1 = 0.3 + 0.1 * N(r) + shift;
      f.c1a = 0.3 + 0.1 * N(r);
      f.c3 = 0.6 + 0.05 * N(r);
      f.loop8 = 0.05 + 0.01 * N(r);
      f.d[0] = 0.7 + 0.03 * N(r);
      f.d[1] = 0.95;
      f.d[2] = 0.99;
      f.d[3] = 1.0;
      return f;
    };
    for (int i = 0; i < 1000; ++i) v.add_reference(mk(0));
    v.finish();
    const ScoredV2 a = v.score(mk(0)), b = v.score(mk(-1.0));
    check(a.scribe > 0.5 && b.scribe < a.scribe && b.DC > 5, "a far-out coherence statistic adds deviance");
    v.params.off = {V_C1};
    check(v.score(mk(-1.0)).DC < b.DC, "an ablated statistic stops counting");
    V2Params p;
    check(p.parse("dead=0.5,capS=20,wC=2,off=s2+c1,on=wfreq") && p.dead == 0.5 && p.capS == 20 && p.wC == 2 && p.off.size() == 2 &&
              p.on.size() == 1 && !p.counts(V_S2) && p.counts(V_WFREQ) && p.counts(V_S3G) && !p.counts(V_S3) && !p.counts(V_SLMAX),
          "V2Params parse");
    check(!p.parse("bogus=1"), "V2Params rejects unknown keys");
  }
}

// ===========================================================================
// SCRIBE v3: two more judges with other maths, and the mean of three (docs/reports/SCRIBE_V3.md)
// ===========================================================================
//
// v2 rests partly on co-occurrence (word vectors for coherence). v3 adds two judges that read none:
//   SURPRISE (information theory): the per-word surprisal, in bits, of the text under a model trained on the pack
//            only, and how it is spread. Too predictable (copied, generic, looping) is as bad as too surprising.
//   MATCH    (string matching): how the text's bytes match the pack (a suffix array over the pack), the text's own
//            Lempel-Ziv complexity, and, with a prompt, how much of the text is the prompt again.
// Each statistic is compared with the human reference samples of its length bucket by v2's ECDF z-score (QuantileZ),
// two-sided; evidence and score follow v2's form. SCRIBE v3 = mean(v2, SURPRISE, MATCH).

// Character trigram over the bytes of word types, Witten-Bell interpolated down to an add-one unigram over the 256
// bytes and the end of the word: a proper distribution over all finite byte strings. SURPRISE spells the words the
// pack's trigram calls <unk> with one trained on the pack's once-seen word types, and numbers with one trained on its
// number types.
class SpellModel {
 public:
  static constexpr int kSym = 257, kEow = 256, kBow = 257, kCtx = 258;
  void train(const std::vector<std::string>& types) {
    n1_.assign(kSym, 0);
    n2_.assign(static_cast<size_t>(kCtx) * kSym, 0);
    N2_.assign(kCtx, 0);
    T2_.assign(kCtx, 0);
    N3_.assign(static_cast<size_t>(kCtx) * kCtx, 0);
    T3_.assign(static_cast<size_t>(kCtx) * kCtx, 0);
    n3_.clear();
    N1_ = 0;
    for (const std::string& w : types) {
      int a = kBow, b = kBow;
      for (size_t i = 0; i <= w.size(); ++i) {
        const int c = i < w.size() ? static_cast<unsigned char>(w[i]) : kEow;
        ++n1_[static_cast<size_t>(c)];
        ++N1_;
        if (n2_[static_cast<size_t>(b) * kSym + static_cast<size_t>(c)]++ == 0) ++T2_[static_cast<size_t>(b)];
        ++N2_[static_cast<size_t>(b)];
        const size_t ab = static_cast<size_t>(a) * kCtx + static_cast<size_t>(b);
        if (n3_[static_cast<std::uint32_t>(ab * kSym + static_cast<size_t>(c))]++ == 0) ++T3_[ab];
        ++N3_[ab];
        a = b;
        b = c;
      }
    }
    trained_ = !types.empty();
  }
  bool trained() const { return trained_; }
  // Bits to spell w and end it.
  double bits(const std::string& w) const {
    if (!trained_) return 8.0 * static_cast<double>(w.size() + 1);
    double s = 0;
    int a = kBow, b = kBow;
    for (size_t i = 0; i <= w.size(); ++i) {
      const int c = i < w.size() ? static_cast<unsigned char>(w[i]) : kEow;
      double p = (static_cast<double>(n1_[static_cast<size_t>(c)]) + 1.0) / (static_cast<double>(N1_) + kSym);
      const size_t bi = static_cast<size_t>(b);
      if (N2_[bi]) p = (static_cast<double>(n2_[bi * kSym + static_cast<size_t>(c)]) + T2_[bi] * p) / (static_cast<double>(N2_[bi]) + T2_[bi]);
      const size_t ab = static_cast<size_t>(a) * kCtx + bi;
      if (N3_[ab]) {
        const auto it = n3_.find(static_cast<std::uint32_t>(ab * kSym + static_cast<size_t>(c)));
        const double n = it == n3_.end() ? 0.0 : static_cast<double>(it->second);
        p = (n + T3_[ab] * p) / (static_cast<double>(N3_[ab]) + T3_[ab]);
      }
      s -= std::log2(p);
      a = b;
      b = c;
    }
    return s;
  }

 private:
  bool trained_ = false;
  std::uint64_t N1_ = 0;
  std::vector<std::uint64_t> n1_;
  std::vector<std::uint32_t> n2_, N2_, T2_, N3_, T3_;
  std::unordered_map<std::uint32_t, std::uint32_t> n3_;
};

// ---------------------------------------------------------------------------
// SURPRISE: the surprisal sequence
// ---------------------------------------------------------------------------

// The model whose code length SURPRISE reads. Static part: the pack's interpolated Kneser-Ney word trigram
// (ScribeModel, sentence-internal context, proper over its vocabulary), with <unk> and <num> spelled by SpellModel, so
// every string has a probability. Dynamic part (the text's own history, as a compressor would use it):
//   p1(w) = (1 - lambda) p_static(w) + lambda c_H(w) / |H|             unigram cache (Kuhn and De Mori 1990)
//   p(w)  = (1 - k / (k + beta)) p1(w) + k / (k + beta) [w = w*]       match model: the last k >= kmin words also
//                                                                      occur earlier, followed there by w*
// s(w) = -log2 p(w) bits; s0(w) = -log2 p_static(w) (no history). Both are proper code lengths.
struct SurpriseParams {
  double lambda = 0.1;  // unigram cache weight
  double beta = 1.0;    // match model: confidence k / (k + beta) after a match of k words
  int kmin = 2;         // shortest word match the match model uses
  double coh_min_bits = 8;  // cohesion gain: only words whose unigram surprisal is at least this (content words)
};

class SurpriseModel {
 public:
  SurpriseParams params;
  // The spelling models from the pack's word types (the same tokenisation as ScribeModel::train).
  void train(const ScribeModel& m, const std::vector<std::string>& pack_docs) {
    m_ = &m;
    std::unordered_map<std::string, std::uint32_t> cnt;
    cnt.reserve(1u << 20);
    std::vector<std::string> toks;
    for (const std::string& d : pack_docs)
      for (const std::string& s : split_sentences(d)) {
        toks.clear();
        tokenize_words(s, toks);
        for (const std::string& t : toks) ++cnt[t];
      }
    std::vector<std::string> once, nums;
    uni_.assign(m.vocab_size(), 0.0);
    double total = 0;
    for (const auto& kv : cnt) {
      if (numeric_token(kv.first)) nums.push_back(kv.first);
      else if (kv.second == 1) once.push_back(kv.first);
      uni_[m.word_id(kv.first)] += kv.second;  // <unk> and <num> collect their words' counts
      total += kv.second;
    }
    for (double& x : uni_) x = -std::log2((x + 0.5) / (total + 0.5 * static_cast<double>(uni_.size())));
    std::sort(once.begin(), once.end());
    std::sort(nums.begin(), nums.end());
    unk_.train(once);
    num_.train(nums);
    n_once_ = once.size();
    n_nums_ = nums.size();
  }
  bool trained() const { return m_ != nullptr && m_->trained(); }  // an untrained spelling model spells 8 bits a byte
  size_t n_once_types() const { return n_once_; }
  size_t n_number_types() const { return n_nums_; }
  double spell_bits(const std::string& w) const { return numeric_token(w) ? num_.bits(w) : unk_.bits(w); }

  // What sequence() computes for one text.
  struct Seq {
    std::vector<float> s, s0, su, sp;  // per word: surprisal with the history, without it, under the unigram; spelling bits
    std::vector<std::string> toks;     // the words
    std::vector<std::uint16_t> sent;   // the sentence of each word
    double coh = std::numeric_limits<double>::quiet_NaN();  // cohesion gain (NaN: under two sentences)
  };
  // Surprisal of every word token of the text (sentence order). su: the pack unigram's bits plus the same spelling
  // bits, so su - s0 is the information the trigram context gives about the word.
  void sequence(const std::string& text, Seq& q) const {
    q = Seq();
    if (!trained()) return;
    const std::vector<std::vector<std::string>> st = sentence_tokens(text);
    std::vector<std::uint64_t> h;
    std::unordered_map<std::string, int> seen;
    size_t H = 0;
    const double ln2 = std::log(2.0);
    for (size_t si = 0; si < st.size(); ++si) {
      const auto& t = st[si];
      std::uint32_t u = ScribeModel::kBos, v = ScribeModel::kBos;
      for (const std::string& w : t) {
        const std::uint32_t id = m_->word_id(w);
        const double spell = id == ScribeModel::kUnk ? unk_.bits(w) : (id == ScribeModel::kNum ? num_.bits(w) : 0.0);
        const double b0 = -m_->logp(u, v, id) / ln2 + spell;
        double p = std::exp2(-b0);
        if (H > 0) {
          const auto it = seen.find(w);
          const double c = it == seen.end() ? 0.0 : static_cast<double>(it->second);
          p = (1.0 - params.lambda) * p + params.lambda * c / static_cast<double>(H);
        }
        // Match model: the longest k >= kmin such that the last k words occurred earlier (most recent on ties).
        const std::uint64_t hw = scribe_detail::fnv1a(w);
        const size_t i = h.size();
        int best_k = 0;
        std::uint64_t pred = 0;
        for (size_t j = i; j-- > 1;) {  // the earlier word h[j] followed context h[j-k..j)
          int k = 0;
          while (k < 32 && static_cast<size_t>(k) < j && h[j - 1 - static_cast<size_t>(k)] == h[i - 1 - static_cast<size_t>(k)]) ++k;
          if (k > best_k) {
            best_k = k;
            pred = h[j];
          }
        }
        if (best_k >= params.kmin) {
          const double conf = best_k / (best_k + params.beta);
          p = (1.0 - conf) * p + (hw == pred ? conf : 0.0);
        }
        q.s.push_back(static_cast<float>(-std::log2(std::max(p, 1e-300))));
        q.s0.push_back(static_cast<float>(b0));
        q.su.push_back(static_cast<float>(uni_[id] + spell));
        q.sp.push_back(static_cast<float>(spell));
        q.toks.push_back(w);
        q.sent.push_back(static_cast<std::uint16_t>(std::min<size_t>(si, 65535)));
        h.push_back(hw);
        ++seen[w];
        ++H;
        u = v;
        v = id;
      }
    }
    // Cohesion as code length: each sentence coded with a unigram cache of the text's OTHER sentences,
    // p = (1 - lambda) p_static + lambda c_other(w) / |other|; the bits it saves per word, averaged over the sentences.
    if (st.size() >= 2) {
      std::unordered_map<std::string, int> all;
      for (const auto& t : st)
        for (const std::string& w : t) ++all[w];
      size_t k = 0;
      const size_t Hall = q.s0.size();
      double tot = 0;
      int nsent = 0;
      for (const auto& t : st) {
        std::unordered_map<std::string, int> own;
        for (const std::string& w : t) ++own[w];
        const size_t Ho = Hall - t.size();
        double g = 0;
        int m = 0;
        for (const std::string& w : t) {
          const double b0 = q.s0[k];
          const double bu = q.su[k++];
          if (Ho == 0 || bu < params.coh_min_bits) continue;  // content words only
          const double c = static_cast<double>(all[w] - own[w]);
          const double p = (1.0 - params.lambda) * std::exp2(-b0) + params.lambda * c / static_cast<double>(Ho);
          g += b0 + std::log2(std::max(p, 1e-300));
          ++m;
        }
        if (m) {
          tot += g / m;
          ++nsent;
        }
      }
      if (nsent) q.coh = tot / nsent;
    }
  }

  // SCRIBE v3.1 (docs/reports/SCRIBE_V31.md). Per word, how badly it fits BOTH its neighbours under the pack's
  // static trigram (no history):
  //   fit(w_i) = [s0(w_i | w_{i-2} w_{i-1}) - su(w_i)] + [s0(R | w_{i-1} w_i) - s0(R | w_{i-2} w_{i-1})]
  // where R is the right context (the next two words, or </s>) and the second bracket codes R once with w_i and once
  // with w_i deleted. So fit = the sentence's code length with w_i, minus its code length without w_i, minus w_i's
  // own unigram code length: minus the pointwise mutual information of w_i with its two-sided context, in bits.
  // Negative: the word belongs; positive: the sentence reads better without it (an inserted word). Spelling bits cancel.
  // With a prompt, also the bits a unigram cache of the prompt's words saves on the text's content words (unigram
  // surprisal >= coh_min_bits), p = (1 - lambda) p_static + lambda c_prompt(w) / |prompt|, per word of the text; and
  // per sentence (lead: the first sentence's gain per word minus the mean of the other sentences').
  struct Extra31 {
    std::vector<float> fit;  // per word, sentence order
    bool has_pgain = false, has_lead = false;
    double pgain = 0, lead = 0;
  };
  void extra31(const std::string& text, const std::string& prompt, Extra31& x) const {
    x = Extra31();
    if (!trained()) return;
    const double ln2 = std::log(2.0);
    const std::vector<std::vector<std::string>> st = sentence_tokens(text);
    std::unordered_map<std::string, int> pc;
    size_t P = 0;
    if (!prompt.empty())
      for (const auto& t : sentence_tokens(prompt))
        for (const std::string& w : t) {
          ++pc[w];
          ++P;
        }
    std::vector<std::uint32_t> ids;
    std::vector<double> sg;  // prompt gain per word, per sentence
    double gain = 0;
    size_t nw = 0;
    for (const auto& t : st) {
      ids.clear();
      for (const std::string& w : t) ids.push_back(m_->word_id(w));
      const long n = static_cast<long>(ids.size());
      auto at = [&](long j) { return j < 0 ? ScribeModel::kBos : (j >= n ? ScribeModel::kEos : ids[static_cast<size_t>(j)]); };
      double g = 0;
      for (long i = 0; i < n; ++i) {
        const std::uint32_t u = at(i - 2), v = at(i - 1), w = at(i), r1 = at(i + 1);
        const double lw = m_->logp(u, v, w);
        double with = m_->logp(v, w, r1), without = m_->logp(u, v, r1);
        if (i + 1 < n) {
          const std::uint32_t r2 = at(i + 2);
          with += m_->logp(w, r1, r2);
          without += m_->logp(v, r1, r2);
        }
        x.fit.push_back(static_cast<float>(-lw / ln2 - uni_[w] - (with - without) / ln2));
        if (P > 0) {
          const std::string& tok = t[static_cast<size_t>(i)];
          const double spell = w == ScribeModel::kUnk ? unk_.bits(tok) : (w == ScribeModel::kNum ? num_.bits(tok) : 0.0);
          if (uni_[w] + spell >= params.coh_min_bits) {
            const double b0 = -lw / ln2 + spell;
            const auto it = pc.find(tok);
            const double c = it == pc.end() ? 0.0 : static_cast<double>(it->second);
            g += b0 + std::log2(std::max((1.0 - params.lambda) * std::exp2(-b0) + params.lambda * c / static_cast<double>(P), 1e-300));
          }
        }
      }
      if (n > 0) {
        gain += g;
        nw += static_cast<size_t>(n);
        sg.push_back(g / static_cast<double>(n));
      }
    }
    if (P > 0 && nw > 0) {
      x.has_pgain = true;
      x.pgain = gain / static_cast<double>(nw);
      if (sg.size() >= 2) {
        double o = 0;
        for (size_t i = 1; i < sg.size(); ++i) o += sg[i];
        x.lead = sg[0] - o / static_cast<double>(sg.size() - 1);
        x.has_lead = true;
      }
    }
  }

 private:
  const ScribeModel* m_ = nullptr;
  SpellModel unk_, num_;
  std::vector<double> uni_;  // unigram surprisal (bits) of each id of m_, <unk> and <num> as classes
  size_t n_once_ = 0, n_nums_ = 0;
};

// ---------------------------------------------------------------------------
// MATCH: a suffix array over the pack, matching statistics, Lempel-Ziv complexity
// ---------------------------------------------------------------------------

// The pack's documents joined by '\n' and a suffix array sorted on the first kCap bytes of each suffix (matches are
// counted up to kCap bytes; a longer one counts as kCap). The texts scored never hold '\n', so no match crosses a
// document boundary. The longest prefix of a pattern that occurs in the pack is its longest common prefix with one of
// its two neighbours in the sorted order: one binary search inside the bucket of its first two bytes.
class PackIndex {
 public:
  static constexpr size_t kCap = 64;
  void build(const std::vector<std::string>& docs, unsigned threads = 4) {
    text_.clear();
    size_t tot = 0;
    for (const std::string& d : docs) tot += d.size() + 1;
    text_.reserve(tot);
    for (const std::string& d : docs) {
      text_ += d;
      text_.push_back('\n');
    }
    for (char& c : text_)
      if (c == '\0') c = ' ';
    const size_t N = text_.size();
    const unsigned char* t = reinterpret_cast<const unsigned char*>(text_.data());
    for (size_t i = 0; i < N; ++i) {
      has1_[t[i]] = true;
      ++n1_[t[i]];
    }
    auto key2 = [&](size_t i) { return (static_cast<std::uint32_t>(t[i]) << 8) | (i + 1 < N ? t[i + 1] : 0u); };
    b2_.assign(65537, 0);
    for (size_t i = 0; i < N; ++i)
      if (t[i] != '\n') ++b2_[key2(i) + 1];
    for (size_t k = 1; k <= 65536; ++k) b2_[k] += b2_[k - 1];
    sa_.assign(b2_[65536], 0);
    {
      std::vector<std::uint32_t> fill(b2_.begin(), b2_.end() - 1);
      for (size_t i = 0; i < N; ++i)
        if (t[i] != '\n') sa_[fill[key2(i)]++] = static_cast<std::uint32_t>(i);
    }
    std::atomic<size_t> next{0};
    auto work = [&]() {
      std::vector<std::pair<std::uint64_t, std::uint32_t>> buf;
      for (size_t b; (b = next++) < 65536;) {
        const size_t lo = b2_[b], hi = b2_[b + 1];
        if (hi - lo < 2) continue;
        buf.clear();
        for (size_t r = lo; r < hi; ++r) buf.emplace_back(key8(sa_[r] + 2), sa_[r]);
        std::sort(buf.begin(), buf.end(), [&](const auto& x, const auto& y) {
          if (x.first != y.first) return x.first < y.first;
          return cmp_text(x.second + 10, y.second + 10, kCap - 10) < 0;
        });
        for (size_t r = lo; r < hi; ++r) sa_[r] = buf[r - lo].second;
      }
    };
    std::vector<std::thread> th;
    for (unsigned k = 1; k < std::max(1u, threads); ++k) th.emplace_back(work);
    work();
    for (std::thread& x : th) x.join();
    built_ = true;
  }
  bool built() const { return built_; }
  size_t size() const { return text_.size(); }
  size_t bytes() const { return text_.size() + sa_.size() * 4 + b2_.size() * 4; }

  // The longest prefix of p[0..len), at most kCap bytes, that occurs in the pack.
  size_t match(const unsigned char* p, size_t len) const {
    len = std::min(len, kCap);
    if (len == 0 || !has1_[p[0]]) return 0;
    if (len == 1) return 1;
    const std::uint32_t k = (static_cast<std::uint32_t>(p[0]) << 8) | p[1];
    const size_t lo = b2_[k], hi = b2_[k + 1];
    if (lo == hi) return 1;
    size_t a = lo, b = hi;
    while (a < b) {
      const size_t mid = a + (b - a) / 2;
      if (cmp_pat(sa_[mid] + 2, p + 2, len - 2) < 0) a = mid + 1;
      else b = mid;
    }
    size_t best = 2;
    if (a < hi) best = std::max(best, 2 + lcp(sa_[a] + 2, p + 2, len - 2));
    if (a > lo) best = std::max(best, 2 + lcp(sa_[a - 1] + 2, p + 2, len - 2));
    return best;
  }
  // How often p[0..len) occurs in the pack (len <= kCap; a longer string is counted by its first kCap bytes).
  size_t count(const unsigned char* p, size_t len) const {
    len = std::min(len, kCap);
    if (len == 0) return text_.size();
    if (len == 1) return n1_[p[0]];
    const std::uint32_t k = (static_cast<std::uint32_t>(p[0]) << 8) | p[1];
    const size_t lo = b2_[k], hi = b2_[k + 1];
    if (len == 2) return hi - lo;
    size_t a = lo, b = hi;
    while (a < b) {  // first suffix >= p
      const size_t mid = a + (b - a) / 2;
      if (cmp_pat(sa_[mid] + 2, p + 2, len - 2) < 0) a = mid + 1;
      else b = mid;
    }
    size_t c = a, e = hi;
    while (c < e) {  // first suffix that does not start with p
      const size_t mid = c + (e - c) / 2;
      if (lcp(sa_[mid] + 2, p + 2, len - 2) == len - 2) c = mid + 1;
      else e = mid;
    }
    return c - a;
  }

 private:
  std::uint64_t key8(size_t i) const {
    std::uint64_t k = 0;
    for (size_t j = 0; j < 8; ++j) k = (k << 8) | (i + j < text_.size() ? static_cast<unsigned char>(text_[i + j]) : 0u);
    return k;
  }
  int cmp_text(size_t i, size_t j, size_t len) const {
    const size_t N = text_.size();
    const size_t li = i < N ? std::min(len, N - i) : 0, lj = j < N ? std::min(len, N - j) : 0;
    const int r = std::memcmp(text_.data() + std::min(i, N), text_.data() + std::min(j, N), std::min(li, lj));
    if (r != 0) return r;
    return li < lj ? -1 : (li > lj ? 1 : 0);
  }
  int cmp_pat(size_t i, const unsigned char* q, size_t m) const {
    const size_t N = text_.size();
    const size_t li = i < N ? std::min(m, N - i) : 0;
    const int r = std::memcmp(text_.data() + std::min(i, N), q, li);
    if (r != 0) return r;
    return li < m ? -1 : 0;
  }
  size_t lcp(size_t i, const unsigned char* q, size_t m) const {
    const size_t N = text_.size();
    const size_t li = i < N ? std::min(m, N - i) : 0;
    size_t k = 0;
    while (k < li && static_cast<unsigned char>(text_[i + k]) == q[k]) ++k;
    return k;
  }
  bool built_ = false;
  std::string text_;
  std::vector<std::uint32_t> sa_, b2_;
  bool has1_[256] = {};
  size_t n1_[256] = {};
};

namespace scribe_detail {

// Lempel-Ziv (1976) complexity: the number of phrases of the LZ76 parsing (Kaspar and Schuster 1987).
inline size_t lz76_phrases(const std::string& s) {
  const size_t n = s.size();
  if (n == 0) return 0;
  if (n == 1) return 1;
  size_t c = 1, l = 1, i = 0, k = 1, kmax = 1;
  while (true) {
    if (s[i + k - 1] == s[l + k - 1]) {
      ++k;
      if (l + k > n) {
        ++c;
        break;
      }
    } else {
      if (k > kmax) kmax = k;
      ++i;
      if (i == l) {
        ++c;
        l += kmax;
        if (l + 1 > n) break;
        i = 0;
        k = 1;
        kmax = 1;
      } else {
        k = 1;
      }
    }
  }
  return c;
}

// For each position i of a, the longest common prefix of a[i..] with any b[j..] (at most cap): a naive scan, fine for
// texts of a few hundred bytes. With self = true, b is a itself and only j < i counts (the text's own past; the
// source may run on into i, as in LZ77).
inline std::vector<std::uint16_t> lcp_scan(const std::string& a, const std::string& b, bool self, size_t cap = 64) {
  std::vector<std::uint16_t> m(a.size(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    size_t best = 0;
    const size_t jend = self ? i : b.size();
    for (size_t j = 0; j < jend; ++j) {
      if (b[j] != a[i]) continue;
      size_t k = 1;
      const size_t lim = std::min(cap, std::min(a.size() - i, b.size() - j));
      while (k < lim && a[i + k] == b[j + k]) ++k;
      if (k > best) {
        best = k;
        if (best >= cap) break;
      }
    }
    m[i] = static_cast<std::uint16_t>(best);
  }
  return m;
}

// Coverage: for each byte j, the longest match m[i] with i <= j < i + m[i].
inline std::vector<std::uint16_t> match_cover(const std::vector<std::uint16_t>& m) {
  std::vector<std::uint16_t> c(m.size(), 0);
  for (size_t i = 0; i < m.size(); ++i)
    for (size_t j = i; j < std::min(m.size(), i + m[i]); ++j) c[j] = std::max(c[j], m[i]);
  return c;
}

inline double quantile_sorted(const std::vector<double>& v, double p) {
  if (v.empty()) return 0;
  const double h = p * static_cast<double>(v.size() - 1);
  const size_t i = static_cast<size_t>(h);
  const double fr = h - static_cast<double>(i);
  return i + 1 < v.size() ? v[i] * (1 - fr) + v[i + 1] * fr : v.back();
}

}  // namespace scribe_detail

// ---------------------------------------------------------------------------
// The v3 statistics
// ---------------------------------------------------------------------------

enum V3Stat {
  // SURPRISE
  W_MEAN, W_SD, W_AC1, W_LO, W_HI, W_CTX, W_ANTI, W_SPELL, W_COH,
  // MATCH
  M_MED, M_P90, M_LONG, M_SHORT, M_LZ, M_TRI, M_COH, M_ECHO, M_ECHO32,
  // v3.1 only (computed and scored in profile v31; docs/reports/SCRIBE_V31.md)
  W_FIT, W_FITHI, W_PGAIN, W_LEAD, M_BRIDGE, M_PCOH,
  W3_N
};
constexpr int W3_N_V3 = W_FIT;  // the statistics of profile v3

// Each statistic is two-sided: its z-score is compared with the human reference on both sides, and each side's
// evidence goes to one group of its metric's deviance (0: that side is not counted, because another statistic of the
// same group already measures it):
//   'L' lexical failure (not words)          saturating
//   'S' order failure (not word order)       saturating
//   'C' cohesion failure (sentences unrelated, or off the prompt)   graded
//   'P' too predictable (copied, looping, echoing)                  linear beyond the human threshold
struct V3StatSpec {
  const char* name;
  char metric;      // 'W' SURPRISE, 'M' MATCH
  double min_sd;    // floor of the tail scale (QuantileZ)
  char plus, minus; // group of the high side and of the low side
  char missing;     // side counted (at missing_z) when the statistic is absent where >= 95% of human samples have it
};

inline const V3StatSpec& v3_spec(int k) {
  static const V3StatSpec specs[W3_N] = {
      {"w_mean", 'W', 0.10, 'L', 'P', '+'},   // mean surprisal (bits per word)
      {"w_sd", 'W', 0.10, 'L', 'P', '+'},     // its standard deviation
      {"w_ac1", 'W', 0.02, 'P', 'S', '-'},    // its lag-1 autocorrelation (burstiness)
      {"w_lo", 'W', 0.01, 'P', 'S', '-'},     // share of words below the low threshold (too predictable)
      {"w_hi", 'W', 0.01, 'L', 'P', '+'},     // share above the high threshold (salad)
      {"w_ctx", 'W', 0.05, 'P', 'S', '-'},    // information the trigram context gives (unigram bits - trigram bits)
      {"w_anti", 'W', 0.01, 0, 0, 0},        // share of words the context makes > 2 bits LESS likely (reported only)
      {"w_spell", 'W', 0.05, 'L', 'P', '+'},  // bits spent spelling words the pack's trigram does not know (per word)
      {"w_coh", 'W', 0.02, 'P', 'C', '-'},    // bits a sentence's content words save from the other sentences
      {"m_med", 'M', 0.25, 'P', 0, '-'},      // median longest pack match from each byte
      {"m_p90", 'M', 0.5, 'P', 'S', '-'},     // its 90th percentile
      {"m_long", 'M', 0.005, 'P', 0, '-'},    // share of bytes inside pack matches of >= 32 bytes (copying)
      {"m_short", 'M', 0.005, 'L', 'P', '+'}, // share of bytes only inside pack matches of <= 4 bytes (novel junk)
      {"m_lz", 'M', 0.01, 'C', 'P', '+'},     // LZ76 complexity, phrases x log2(n) / n
      {"m_tri", 'M', 0.01, 'P', 'S', '-'},    // share of word starts whose pack match covers >= 3 whole words
      {"m_coh", 'M', 0.005, 'P', 'C', '-'},   // bytes shared with the other sentences, weighted by their pack information
      {"m_echo", 'M', 0.005, 'P', 'C', 0},    // share of bytes inside >= 8-byte matches with the prompt (prompted only)
      {"m_echo32", 'M', 0.005, 'P', 0, 0},   // share inside >= 32-byte matches with the prompt (copied prompt sentences)
      // v3.1 (their groups here are the defaults of profile v31; profile v3 never computes them)
      {"w_fit", 'W', 0.05, 'S', 'P', '+'},    // mean fit: minus the PMI of each word with its two-sided context (bits)
      {"w_fithi", 'W', 0.01, 'S', 'P', '+'},  // share of words whose fit is above the reference words' fit_q quantile
      {"w_pgain", 'W', 0.01, 'P', 'C', 0},    // bits per word a unigram cache of the prompt saves (prompted only)
      {"w_lead", 'W', 0.01, 0, 0, 0},         // first sentence's prompt gain minus the others' (reported only)
      {"m_bridge", 'M', 0.005, 'S', 0, '+'},  // share of words whose two pack bigrams are absent while the neighbours' is present
      {"m_pcoh", 'M', 0.005, 'P', 'C', 0}};   // bytes shared with the prompt from word starts, weighted by pack information
  return specs[k];
}
inline bool v3_prompted(int k) { return k == M_ECHO || k == M_ECHO32 || k == W_PGAIN || k == W_LEAD || k == M_PCOH; }

// Raw per-text material of v3. w_lo and w_hi need the reference's thresholds and are finished by V3Scorer::values.
struct V3Features {
  size_t bytes = 0;
  int n_words = 0;
  std::vector<float> s;  // SURPRISE: per-word surprisal with the history (bits)
  bool has_w = false, has_w_coh = false;
  double w_mean = 0, w_sd = 0, w_ac1 = 0, w_ctx = 0, w_anti = 0, w_spell = 0, w_coh = 0;
  bool has_m = false, has_m_coh = false, has_echo = false;
  double m_med = 0, m_p90 = 0, m_long = 0, m_short = 0, m_lz = 0, m_tri = 0, m_coh = 0, m_echo = 0, m_echo32 = 0;
  double m_words = 0;  // reported only: mean number of whole words a pack match covers from a word start
  SurpriseModel::Seq seq;  // kept only on request (exploration dumps)
  // v3.1 (only with v31 = true)
  bool has_31 = false, has_pgain = false, has_lead = false, has_bridge = false, has_pcoh = false;
  std::vector<float> fit;  // per word: minus the PMI with the two-sided context (bits)
  double w_fit = 0, w_pgain = 0, w_lead = 0, m_bridge = 0, m_pcoh = 0;
};

inline V3Features extract_v3(const SurpriseModel& sm, const PackIndex& px, const std::string& text, const std::string& prompt = "",
                             bool keep_seq = false, bool v31 = false) {
  using namespace scribe_detail;
  V3Features f;
  f.bytes = text.size();
  SurpriseModel::Seq q;
  sm.sequence(text, q);
  f.n_words = static_cast<int>(q.s.size());
  f.s = q.s;
  if (f.n_words > 0) {
    const double n = static_cast<double>(f.n_words);
    double m = 0, ctx = 0, sp = 0, anti = 0;
    for (size_t i = 0; i < q.s.size(); ++i) {
      m += q.s[i];
      ctx += q.su[i] - q.s0[i];
      sp += q.sp[i];
      anti += q.s0[i] - q.su[i] > 2.0;
    }
    m /= n;
    double v = 0, c1 = 0;
    for (size_t i = 0; i < q.s.size(); ++i) {
      v += (q.s[i] - m) * (q.s[i] - m);
      if (i + 1 < q.s.size()) c1 += (q.s[i] - m) * (q.s[i + 1] - m);
    }
    f.w_mean = m;
    f.w_sd = std::sqrt(v / n);
    f.w_ac1 = v > 0 ? c1 / v : 0.0;
    f.w_ctx = ctx / n;
    f.w_spell = sp / n;
    f.w_anti = anti / n;
    f.has_w = true;
    f.w_coh = q.coh;
    f.has_w_coh = std::isfinite(q.coh);
  }
  if (keep_seq) f.seq = std::move(q);
  if (v31 && f.n_words > 0) {
    SurpriseModel::Extra31 x;
    sm.extra31(text, prompt, x);
    if (!x.fit.empty()) {
      f.has_31 = true;
      f.fit = std::move(x.fit);
      double s = 0;
      for (float v : f.fit) s += v;
      f.w_fit = s / static_cast<double>(f.fit.size());
    }
    f.has_pgain = x.has_pgain;
    f.w_pgain = x.pgain;
    f.has_lead = x.has_lead;
    f.w_lead = x.lead;
  }
  const size_t n = text.size();
  if (!px.built() || n < 8) return f;
  // v3.1, MATCH: a word that fits neither neighbour as strings. The bigrams it forms with its left and with its right
  // neighbour both are absent from the pack, while the bigram its two neighbours form without it is present.
  if (v31) {
    std::vector<std::pair<size_t, size_t>> tk;
    for (size_t p = 0; p < n;) {
      while (p < n && text[p] == ' ') ++p;
      size_t e = p;
      while (e < n && text[e] != ' ') ++e;
      if (e > p) tk.emplace_back(p, e);
      p = e;
    }
    auto occurs = [&](const std::string& s) {
      const size_t L = std::min(s.size(), PackIndex::kCap);
      return px.match(reinterpret_cast<const unsigned char*>(s.data()), s.size()) >= L;
    };
    if (tk.size() >= 3) {
      size_t nb = 0;
      for (size_t k = 1; k + 1 < tk.size(); ++k) {
        const std::string left = text.substr(tk[k - 1].first, tk[k].second - tk[k - 1].first);
        const std::string right = text.substr(tk[k].first, tk[k + 1].second - tk[k].first);
        if (occurs(left) || occurs(right)) continue;
        const std::string bridge = text.substr(tk[k - 1].first, tk[k - 1].second - tk[k - 1].first) + " " +
                                   text.substr(tk[k + 1].first, tk[k + 1].second - tk[k + 1].first);
        nb += occurs(bridge);
      }
      f.m_bridge = static_cast<double>(nb) / static_cast<double>(tk.size() - 2);
      f.has_bridge = true;
    }
  }
  f.has_m = true;
  const unsigned char* t = reinterpret_cast<const unsigned char*>(text.data());
  // Matching statistics: from each byte, the longest prefix of the rest of the text that occurs in the pack.
  std::vector<std::uint16_t> mp(n);
  for (size_t i = 0; i < n; ++i) mp[i] = static_cast<std::uint16_t>(px.match(t + i, n - i));
  {
    std::vector<double> v(mp.begin(), mp.end());
    std::sort(v.begin(), v.end());
    f.m_med = quantile_sorted(v, 0.5);
    f.m_p90 = quantile_sorted(v, 0.9);
  }
  {
    const std::vector<std::uint16_t> cov = match_cover(mp);
    size_t lg = 0, sh = 0;
    for (std::uint16_t c : cov) {
      lg += c >= 32;
      sh += c <= 4;
    }
    f.m_long = static_cast<double>(lg) / static_cast<double>(n);
    f.m_short = static_cast<double>(sh) / static_cast<double>(n);
  }
  f.m_lz = static_cast<double>(lz76_phrases(text)) * std::log2(static_cast<double>(n)) / static_cast<double>(n);
  // From each word start, how many whole space-separated words the pack match covers.
  {
    double tot = 0;
    size_t starts = 0, k3 = 0;
    for (size_t p = 0; p < n; ++p) {
      if ((p > 0 && text[p - 1] != ' ') || text[p] == ' ') continue;
      const size_t end = p + mp[p];
      size_t k = 0, e = p;
      while (e < n) {
        size_t q2 = e;
        while (q2 < n && text[q2] != ' ') ++q2;
        if (q2 > end) break;
        ++k;
        e = q2 + 1;
      }
      tot += static_cast<double>(k);
      k3 += k >= 3;
      ++starts;
    }
    f.m_words = starts ? tot / static_cast<double>(starts) : 0.0;
    f.m_tri = starts ? static_cast<double>(k3) / static_cast<double>(starts) : 0.0;
  }
  // Cohesion: each sentence against the text's other sentences. From each word start, the longest match (>= 5 bytes) in
  // the other sentences; each byte it covers counts the matched string's information in the pack,
  // w = log2((N + 1) / (count + 1)) / log2(N + 1) in [0, 1], above a floor of 0.45 (strings the pack holds more than
  // about N^0.55 times, such as " the ", count nothing): max(0, w - 0.45) / 0.55. Mean over the sentence's bytes, then
  // over the sentences.
  {
    const std::vector<std::string> ss = split_sentences(text);
    if (ss.size() >= 2) {
      double sw = 0;
      const double logN = std::log2(static_cast<double>(px.size()) + 1.0);
      for (size_t j = 0; j < ss.size(); ++j) {
        std::string other;
        for (size_t i = 0; i < ss.size(); ++i)
          if (i != j) {
            if (!other.empty()) other.push_back('\n');
            other += ss[i];
          }
        const std::string& s = ss[j];
        std::vector<float> cw(s.size(), 0.0f);
        for (size_t p = 0; p < s.size(); ++p) {
          if (p > 0 && s[p - 1] != ' ') continue;
          size_t L = 0;
          for (size_t q2 = 0; q2 < other.size(); ++q2) {
            if (other[q2] != s[p]) continue;
            size_t k = 1;
            const size_t lim = std::min(PackIndex::kCap, std::min(s.size() - p, other.size() - q2));
            while (k < lim && s[p + k] == other[q2 + k]) ++k;
            L = std::max(L, k);
          }
          if (L < 5) continue;
          const size_t c = px.count(reinterpret_cast<const unsigned char*>(s.data()) + p, L);
          const double w = std::log2((static_cast<double>(px.size()) + 1.0) / (static_cast<double>(c) + 1.0)) / logN;
          const float wf = static_cast<float>(std::max(0.0, w - 0.45) / 0.55);
          for (size_t b = p; b < p + L; ++b) cw[b] = std::max(cw[b], wf);
        }
        double bw = 0;
        for (float x : cw) bw += x;
        sw += bw / static_cast<double>(s.size());
      }
      f.m_coh = sw / static_cast<double>(ss.size());
      f.has_m_coh = true;
    }
  }
  // Echo: the share of the text's bytes inside matches of >= 8 bytes with the prompt.
  if (!prompt.empty()) {
    const std::vector<std::uint16_t> me = lcp_scan(text, prompt, false);
    std::vector<std::uint16_t> e8(n, 0), e32(n, 0);
    for (size_t i = 0; i < n; ++i) {
      if (me[i] >= 8) e8[i] = me[i];
      if (me[i] >= 32) e32[i] = me[i];
    }
    const std::vector<std::uint16_t> ce = match_cover(e8), ce32 = match_cover(e32);
    size_t ae = 0, ae32 = 0;
    for (size_t i = 0; i < n; ++i) {
      ae += ce[i] > 0;
      ae32 += ce32[i] > 0;
    }
    f.m_echo = static_cast<double>(ae) / static_cast<double>(n);
    f.m_echo32 = static_cast<double>(ae32) / static_cast<double>(n);
    f.has_echo = true;
    // v3.1: cohesion with the prompt, m_coh's weighting. From each word start, the longest match (>= 4 bytes) with
    // the prompt; each byte it covers counts the matched string's information in the pack above the 0.45 floor.
    if (v31) {
      const double logN = std::log2(static_cast<double>(px.size()) + 1.0);
      std::vector<float> cw(n, 0.0f);
      for (size_t p = 0; p < n; ++p) {
        if ((p > 0 && text[p - 1] != ' ') || text[p] == ' ' || me[p] < 4) continue;
        const size_t L = me[p];
        const size_t c = px.count(t + p, L);
        const double w = std::log2((static_cast<double>(px.size()) + 1.0) / (static_cast<double>(c) + 1.0)) / logN;
        const float wf = static_cast<float>(std::max(0.0, w - 0.45) / 0.55);
        for (size_t b = p; b < std::min(n, p + L); ++b) cw[b] = std::max(cw[b], wf);
      }
      double s = 0;
      for (float x : cw) s += x;
      f.m_pcoh = s / static_cast<double>(n);
      f.has_pcoh = true;
    }
  }
  return f;
}

// extract_v3 over many texts on several threads (the models are only read). prompts: empty, or one per text.
inline std::vector<V3Features> extract_v3_all(const SurpriseModel& sm, const PackIndex& px, const std::vector<std::string>& texts,
                                              const std::vector<std::string>& prompts = {}, unsigned threads = 4, bool keep_seq = false,
                                              bool v31 = false) {
  std::vector<V3Features> out(texts.size());
  std::atomic<size_t> next{0};
  auto work = [&]() {
    for (size_t i; (i = next++) < texts.size();)
      out[i] = extract_v3(sm, px, texts[i], i < prompts.size() ? prompts[i] : std::string(), keep_seq, v31);
  };
  std::vector<std::thread> th;
  for (unsigned k = 1; k < std::max(1u, threads); ++k) th.emplace_back(work);
  work();
  for (std::thread& x : th) x.join();
  return out;
}

// ---------------------------------------------------------------------------
// The v3 scores
// ---------------------------------------------------------------------------

// One new metric's deviance, v2's form with two-sided evidence split by side (u = the bad side's |z|):
//   L = sum of max(0, u - deadL) over the lexical sides, S = the same over the order sides (deadS)
//   C = sum of max(0, u - deadC) over the cohesion sides (m_lz's high side: lz_w max(0, u - lz_dead))
//   G = sum of max(0, u - deadG) over the too-predictable sides (graded)
//   X = sum over the too-predictable sides of max(0, e_k - th_k), e_k = max(0, u_k - dead), th_k = the q quantile of
//       e_k over the human reference samples of the bucket (for the echo, over the prompted reference continuations)
//   D = cap sat(L - thL) + cap sat(S - thS) + wC C + wG G + wP X,   sat(x) = 1 - exp(-max(0, x) / kap)
//   thL, thS = the q quantiles of L and S over the human reference samples of the length bucket
//   score = exp(-D / tau)
// Chosen on development ladder A and its attack battery (docs/reports/SCRIBE_V3.md); tau puts the mean score of dev A's
// human samples at v2's (0.909).
struct V3MetricParams {
  double dead = 1, deadL = 4, deadS = 1, deadC = 0, deadG = 1.5;
  double cap = 30, kap = 2, wC = 4, wG = 1, wP = 8, q = 0.99, tau = 20;
};

struct V3Params {
  V3MetricParams w, m;    // SURPRISE, MATCH
  double w_qlo = 0.25;    // w_lo: words below this quantile of the reference words' surprisal
  double w_qhi = 0.90;    // w_hi: words above this one
  double lz_w = 0.5, lz_dead = 2;  // MATCH: m_lz's high side counts in C with this weight and dead zone
  double missing_z = 4;
  double disagree = 0.5;  // flag a text when max - min of the three scores exceeds this
  // v3.1 (profile v31, docs/reports/SCRIBE_V31.md). Profile v3 leaves all of this at its v3 value.
  bool v31 = false;     // compute and score the v3.1 statistics
  char grp[W3_N][2];    // group of each statistic's high side and low side ('L', 'S', 'C', 'P' or 0: not counted)
  double swt[W3_N];     // weight of each statistic's L, S and C evidence
  double fit_q = 0.9;   // w_fithi: share of words whose fit is above this quantile of the reference words' fit
  double cp = 0;        // the combination of v2, SURPRISE and MATCH: power mean with this exponent (0: geometric)
  V3Params() {
    m.kap = 0.25;
    m.wC = 6;
    m.deadG = 2;
    m.wP = 20;
    m.tau = 30;
    for (int k = 0; k < W3_N; ++k) {
      grp[k][0] = v3_spec(k).plus;
      grp[k][1] = v3_spec(k).minus;
      swt[k] = 1;
    }
  }
  // Profile v31's defaults (frozen on development data; docs/reports/SCRIBE_V31.md).
  void set_v31() { v31 = true; }
  static int stat_index(const std::string& name) {
    for (int k = 0; k < W3_N; ++k)
      if (name == v3_spec(k).name) return k;
    return -1;
  }
  // "w.tau=6,m.wC=4,lz_w=0.5,..." (w. SURPRISE, m. MATCH)
  bool parse(const std::string& spec) {
    size_t p = 0;
    while (p < spec.size()) {
      size_t e = spec.find(',', p);
      if (e == std::string::npos) e = spec.size();
      const std::string kv = spec.substr(p, e - p);
      p = e + 1;
      const size_t eq = kv.find('=');
      if (eq == std::string::npos) return false;
      std::string k = kv.substr(0, eq);
      const double x = std::strtod(kv.c_str() + eq + 1, nullptr);
      if (k == "w_qlo") w_qlo = x;
      else if (k == "w_qhi") w_qhi = x;
      else if (k == "lz_w") lz_w = x;
      else if (k == "lz_dead") lz_dead = x;
      else if (k == "missing") missing_z = x;
      else if (k == "disagree") disagree = x;
      else if (k == "fit_q") fit_q = x;
      else if (k == "cp") cp = x;
      else if (k.rfind("g.", 0) == 0 || k.rfind("wt.", 0) == 0) {  // g.<stat>=XY (groups of the high, low side; 0 = none), wt.<stat>=x
        const bool g = k[0] == 'g';
        const int s = stat_index(k.substr(g ? 2 : 3));
        if (s < 0) return false;
        if (g) {
          const std::string v = kv.substr(eq + 1);
          if (v.size() != 2) return false;
          for (int side = 0; side < 2; ++side) {
            const char c = v[static_cast<size_t>(side)];
            if (c != '0' && c != 'L' && c != 'S' && c != 'C' && c != 'P') return false;
            grp[s][side] = c == '0' ? 0 : c;
          }
        } else {
          swt[s] = x;
        }
      } else if (k.size() > 2 && (k[0] == 'w' || k[0] == 'm') && k[1] == '.') {
        V3MetricParams& mp = k[0] == 'w' ? w : m;
        k = k.substr(2);
        if (k == "dead") mp.dead = x;
        else if (k == "deadL") mp.deadL = x;
        else if (k == "deadS") mp.deadS = x;
        else if (k == "deadC") mp.deadC = x;
        else if (k == "deadG") mp.deadG = x;
        else if (k == "wG") mp.wG = x;
        else if (k == "cap") mp.cap = x;
        else if (k == "kap") mp.kap = x;
        else if (k == "wC") mp.wC = x;
        else if (k == "wP") mp.wP = x;
        else if (k == "q") mp.q = x;
        else if (k == "tau") mp.tau = x;
        else return false;
      } else {
        return false;
      }
    }
    return true;
  }
};

struct ScoredV3 {
  double z[W3_N] = {};
  bool has[W3_N] = {};
  bool imputed[W3_N] = {};
  double E[2][4] = {};   // [SURPRISE, MATCH][L, S, C, G] evidence sums (G: the too-predictable sides, graded)
  double eP[W3_N] = {};  // each statistic's evidence on its too-predictable side
  double Dg[2][4] = {};  // their terms of D
  double D[2] = {};
  double surprise = 0, match = 0;
};

class V3Scorer {
 public:
  V3Params params;

  // A human reference sample (open continuation).
  void add_reference(const V3Features& f) { ref_[length_bucket(f.bytes)].push_back(f); }
  // A human continuation of a prompt: only its prompt statistics count (the prompted reference of m_echo, m_echo32
  // and, in v3.1, w_pgain, w_lead and m_pcoh).
  void add_reference_echo(const V3Features& f) {
    if (!f.has_echo && !f.has_pgain) return;
    const int b = length_bucket(f.bytes);
    double x[W3_N];
    bool has[W3_N];
    values(f, x, has);
    for (int k = 0; k < W3_N; ++k)
      if (is_echo(k) && has[k]) {
        pq_[b][k].add(x[k]);
        pvals_[b][k].push_back(x[k]);
      }
  }
  size_t n_reference(int b) const { return ref_[b].size(); }
  size_t n_reference_echo(int b) const { return pq_[b][M_ECHO].size(); }
  double threshold_lo() const { return th_lo_; }
  double threshold_hi() const { return th_hi_; }
  double threshold_fit() const { return th_fit_; }
  double group_threshold(int metric, int b, int g) const { return th_[metric][b][g]; }
  double p_threshold(int b, int k) const { return thP_[b][k]; }
  const std::vector<V3Features>& reference(int b) const { return ref_[b]; }
  // z of each prompted reference continuation's echo (bucket b; which: 0 m_echo, 1 m_echo32, or a statistic index).
  std::vector<double> echo_reference_z(int b, int which = 0) const {
    const int k = which == 0 ? M_ECHO : which == 1 ? M_ECHO32 : which;
    std::vector<double> z;
    for (double v : pvals_[b][k]) z.push_back(pq_[b][k].z(v));
    return z;
  }

  // Word thresholds from the pooled reference words, then the z-transforms, then the group thresholds.
  void finish() {
    std::vector<double> all;
    for (int b = 0; b < 2; ++b)
      for (const V3Features& f : ref_[b]) all.insert(all.end(), f.s.begin(), f.s.end());
    std::sort(all.begin(), all.end());
    th_lo_ = scribe_detail::quantile_sorted(all, params.w_qlo);
    th_hi_ = scribe_detail::quantile_sorted(all, params.w_qhi);
    if (params.v31) {
      std::vector<double> fa;
      for (int b = 0; b < 2; ++b)
        for (const V3Features& f : ref_[b]) fa.insert(fa.end(), f.fit.begin(), f.fit.end());
      std::sort(fa.begin(), fa.end());
      th_fit_ = scribe_detail::quantile_sorted(fa, params.fit_q);
    }
    for (int b = 0; b < 2; ++b) {
      nref_[b] = ref_[b].size();
      for (int k = 0; k < W3_N; ++k) {
        q_[b][k] = QuantileZ();
        present_[b][k] = 0;
      }
      for (const V3Features& f : ref_[b]) {
        double x[W3_N];
        bool has[W3_N];
        values(f, x, has);
        for (int k = 0; k < W3_N; ++k)
          if (has[k] && !is_echo(k)) {
            q_[b][k].add(x[k]);
            ++present_[b][k];
          }
      }
      for (int k = 0; k < W3_N; ++k)
        if (!is_echo(k)) q_[b][k].finish(v3_spec(k).min_sd);
      for (int k = 0; k < W3_N; ++k)
        if (is_echo(k)) pq_[b][k].finish(v3_spec(k).min_sd);
    }
    for (int b = 0; b < 2; ++b) {
      std::vector<double> E[2][4];
      for (const V3Features& f : ref_[b]) {
        const ScoredV3 o = evidence(f);
        for (int m = 0; m < 2; ++m)
          for (int g = 0; g < 4; ++g) E[m][g].push_back(o.E[m][g]);
      }
      for (int m = 0; m < 2; ++m)
        for (int g = 0; g < 4; ++g) {
          std::sort(E[m][g].begin(), E[m][g].end());
          th_[m][b][g] = scribe_detail::quantile_sorted(E[m][g], m == 0 ? params.w.q : params.m.q);
        }
      // Too predictable: each statistic counts only beyond what the q share of human samples show on it.
      std::vector<double> eP[W3_N];
      for (const V3Features& f : ref_[b]) {
        const ScoredV3 o = evidence(f);
        for (int k = 0; k < W3_N; ++k) eP[k].push_back(o.eP[k]);
      }
      for (int k = 0; k < W3_N; ++k) {
        std::sort(eP[k].begin(), eP[k].end());
        thP_[b][k] = scribe_detail::quantile_sorted(eP[k], v3_spec(k).metric == 'W' ? params.w.q : params.m.q);
      }
    }
    for (int b = 0; b < 2; ++b)  // the prompt statistics, from the prompted continuations
      for (int k = 0; k < W3_N; ++k) {
        if (!is_echo(k)) continue;
        const int eb = pq_[b][k].size() >= 20 ? b : 1 - b;
        const V3MetricParams& mp = v3_spec(k).metric == 'W' ? params.w : params.m;
        const double side = params.grp[k][0] == 'P' ? 1.0 : params.grp[k][1] == 'P' ? -1.0 : 0.0;
        std::vector<double> e;
        for (double v : pvals_[eb][k]) e.push_back(std::max(0.0, side * pq_[eb][k].z(v) - mp.dead));
        std::sort(e.begin(), e.end());
        thP_[b][k] = scribe_detail::quantile_sorted(e, mp.q);
      }
  }

  // The raw value of each statistic and whether the text has it.
  void values(const V3Features& f, double* x, bool* has) const {
    for (int k = 0; k < W3_N; ++k) has[k] = false;
    if (f.has_w) {
      size_t lo = 0, hi = 0;
      for (float v : f.s) {
        lo += v < th_lo_;
        hi += v > th_hi_;
      }
      const double n = static_cast<double>(f.s.size());
      x[W_MEAN] = f.w_mean;
      x[W_SD] = f.w_sd;
      x[W_AC1] = f.w_ac1;
      x[W_LO] = static_cast<double>(lo) / n;
      x[W_HI] = static_cast<double>(hi) / n;
      x[W_CTX] = f.w_ctx;
      x[W_SPELL] = f.w_spell;
      x[W_ANTI] = f.w_anti;
      for (int k : {W_MEAN, W_SD, W_AC1, W_LO, W_HI, W_CTX, W_ANTI, W_SPELL}) has[k] = true;
    }
    x[W_COH] = f.w_coh;
    has[W_COH] = f.has_w_coh;
    if (f.has_m) {
      x[M_MED] = f.m_med;
      x[M_P90] = f.m_p90;
      x[M_LONG] = f.m_long;
      x[M_SHORT] = f.m_short;
      x[M_LZ] = f.m_lz;
      x[M_TRI] = f.m_tri;
      for (int k : {M_MED, M_P90, M_LONG, M_SHORT, M_LZ, M_TRI}) has[k] = true;
    }
    x[M_COH] = f.m_coh;
    has[M_COH] = f.has_m_coh;
    x[M_ECHO] = f.m_echo;
    has[M_ECHO] = f.has_echo;
    x[M_ECHO32] = f.m_echo32;
    has[M_ECHO32] = f.has_echo;
    if (!params.v31) return;
    if (f.has_31) {
      size_t hi = 0;
      for (float v : f.fit) hi += v > th_fit_;
      x[W_FIT] = f.w_fit;
      x[W_FITHI] = static_cast<double>(hi) / static_cast<double>(f.fit.size());
      has[W_FIT] = has[W_FITHI] = true;
    }
    x[W_PGAIN] = f.w_pgain;
    has[W_PGAIN] = f.has_pgain;
    x[W_LEAD] = f.w_lead;
    has[W_LEAD] = f.has_lead;
    x[M_BRIDGE] = f.m_bridge;
    has[M_BRIDGE] = f.has_bridge;
    x[M_PCOH] = f.m_pcoh;
    has[M_PCOH] = f.has_pcoh;
  }

  int bucket(const V3Features& f) const {
    const int b = length_bucket(f.bytes);
    return nref_[b] < 20 && nref_[1 - b] >= 20 ? 1 - b : b;
  }

  // z-scores and group evidence (no thresholds).
  ScoredV3 evidence(const V3Features& f) const {
    ScoredV3 o;
    const int b = bucket(f);
    double x[W3_N];
    bool has[W3_N];
    values(f, x, has);
    for (int k = 0; k < W3_N; ++k) {
      const V3StatSpec& sp = v3_spec(k);
      const int m = sp.metric == 'W' ? 0 : 1;
      const V3MetricParams& mp = m == 0 ? params.w : params.m;
      const QuantileZ& qz = !is_echo(k) ? q_[b][k] : (pq_[b][k].size() >= 20 ? pq_[b][k] : pq_[1 - b][k]);
      if (qz.size() < 20) continue;
      double z;
      if (has[k]) {
        z = qz.z(x[k]);
        o.has[k] = true;
      } else if (sp.missing && !is_echo(k) && present_[b][k] >= 0.95 * static_cast<double>(nref_[b])) {
        z = sp.missing == '+' ? params.missing_z : -params.missing_z;
        o.imputed[k] = true;
      } else {
        continue;
      }
      o.z[k] = z;
      for (int side = 0; side < 2; ++side) {
        const char g = params.grp[k][side];
        if (!g) continue;
        const double u = side == 0 ? z : -z;
        const int gi = g == 'L' ? 0 : g == 'S' ? 1 : g == 'C' ? 2 : 3;
        double dz = gi == 0 ? mp.deadL : gi == 1 ? mp.deadS : gi == 2 ? mp.deadC : mp.dead;
        double wt = params.swt[k];
        if (k == M_LZ && gi == 2) {
          dz = params.lz_dead;
          wt = params.lz_w;
        }
        if (gi == 3) {
          o.E[m][3] += std::max(0.0, u - mp.deadG);
          o.eP[k] = std::max(0.0, u - mp.dead);
        } else {
          o.E[m][gi] += wt * std::max(0.0, u - dz);
        }
      }
    }
    return o;
  }

  ScoredV3 score(const V3Features& f) const {
    ScoredV3 o = evidence(f);
    const int b = bucket(f);
    for (int m = 0; m < 2; ++m) {
      const V3MetricParams& mp = m == 0 ? params.w : params.m;
      auto sat = [&](double x) { return mp.cap * (1.0 - std::exp(-std::max(0.0, x) / mp.kap)); };
      o.Dg[m][0] = sat(o.E[m][0] - th_[m][b][0]);
      o.Dg[m][1] = sat(o.E[m][1] - th_[m][b][1]);
      o.Dg[m][2] = mp.wC * o.E[m][2];
      double ex = 0;
      for (int k = 0; k < W3_N; ++k)
        if ((v3_spec(k).metric == 'W') == (m == 0)) ex += std::max(0.0, o.eP[k] - thP_[b][k]);
      o.Dg[m][3] = mp.wG * o.E[m][3] + mp.wP * ex;
      o.D[m] = o.Dg[m][0] + o.Dg[m][1] + o.Dg[m][2] + o.Dg[m][3];
    }
    o.surprise = std::exp(-o.D[0] / params.w.tau);
    o.match = std::exp(-o.D[1] / params.m.tau);
    return o;
  }

 private:
  std::vector<V3Features> ref_[2];
  static bool is_echo(int k) { return v3_prompted(k); }  // scored against the prompted continuations
  QuantileZ q_[2][W3_N], pq_[2][W3_N];
  std::vector<double> pvals_[2][W3_N];
  double present_[2][W3_N] = {};
  double thP_[2][W3_N] = {};  // [bucket][statistic]: the too-predictable side's threshold
  size_t nref_[2] = {0, 0};
  double th_lo_ = 0, th_hi_ = 0, th_fit_ = 0;
  double th_[2][2][4] = {};  // [metric][bucket][group]
};

// SCRIBE v3 = the mean of v2, SURPRISE and MATCH; also their minimum, and a flag when they disagree.
struct V3Combined {
  double v3 = 0, min3 = 0, spread = 0;
  bool disagree = false;
};
inline V3Combined combine_v3(double v2, double surprise, double match, double disagree_at = 0.5) {
  V3Combined c;
  c.v3 = (v2 + surprise + match) / 3.0;
  c.min3 = std::min(v2, std::min(surprise, match));
  c.spread = std::max(v2, std::max(surprise, match)) - c.min3;
  c.disagree = c.spread > disagree_at;
  return c;
}

// SCRIBE v3.1 = the power mean of v2, SURPRISE and MATCH with exponent cp. At cp = 0 it is the geometric mean,
// exp(-(D_v2 / tau_v2 + D_S / tau_S + D_M / tau_M) / 3): the three deviances add, so a confident catch by one judge
// (a deviance far beyond any human text's) is not diluted by two judges that see nothing.
inline double combine_v31(double v2, double surprise, double match, double cp) {
  const double f = 1e-300, a = std::max(v2, f), b = std::max(surprise, f), c = std::max(match, f);
  if (cp == 0) return std::exp((std::log(a) + std::log(b) + std::log(c)) / 3.0);
  return std::pow((std::pow(a, cp) + std::pow(b, cp) + std::pow(c, cp)) / 3.0, 1.0 / cp);
}

inline std::string v3_json(const ScoredV3& s, const V3Combined& c) {
  std::string o = "{\"surprise\":" + scribe_num(s.surprise) + ",\"match\":" + scribe_num(s.match) + ",\"min3\":" + scribe_num(c.min3) +
                  ",\"spread\":" + scribe_num(c.spread) + ",\"disagree\":" + (c.disagree ? "1" : "0");
  const char* G = "LSCP";
  for (int m = 0; m < 2; ++m) {
    o += m == 0 ? ",\"surprise_D\":{" : ",\"match_D\":{";
    o += "\"D\":" + scribe_num(s.D[m]);
    for (int g = 0; g < 4; ++g) o += ",\"D" + std::string(1, G[g]) + "\":" + scribe_num(s.Dg[m][g]) + ",\"" + std::string(1, G[g]) + "\":" + scribe_num(s.E[m][g]);
    o += "}";
  }
  o += ",\"z\":{";
  bool first = true;
  for (int k = 0; k < W3_N; ++k) {
    if (!s.has[k] && !s.imputed[k]) continue;
    if (!first) o += ",";
    first = false;
    o += "\"" + std::string(v3_spec(k).name) + "\":" + scribe_num(s.z[k], 4);
  }
  o += "}}";
  return o;
}

inline std::string v3_raw_json(const V3Features& f) {
  auto opt = [](bool h, double x) { return h ? scribe_num(x) : std::string("null"); };
  return "{\"n_words\":" + std::to_string(f.n_words) + ",\"w_mean\":" + opt(f.has_w, f.w_mean) + ",\"w_sd\":" + opt(f.has_w, f.w_sd) +
         ",\"w_ac1\":" + opt(f.has_w, f.w_ac1) + ",\"w_ctx\":" + opt(f.has_w, f.w_ctx) + ",\"w_anti\":" + opt(f.has_w, f.w_anti) + ",\"w_spell\":" + opt(f.has_w, f.w_spell) + ",\"w_coh\":" + opt(f.has_w_coh, f.w_coh) +
         ",\"m_med\":" + opt(f.has_m, f.m_med) + ",\"m_p90\":" + opt(f.has_m, f.m_p90) + ",\"m_long\":" + opt(f.has_m, f.m_long) +
         ",\"m_short\":" + opt(f.has_m, f.m_short) + ",\"m_lz\":" + opt(f.has_m, f.m_lz) + ",\"m_tri\":" + opt(f.has_m, f.m_tri) +
         ",\"m_words\":" + opt(f.has_m, f.m_words) + ",\"m_coh\":" + opt(f.has_m_coh, f.m_coh) + ",\"m_echo\":" + opt(f.has_echo, f.m_echo) + ",\"m_echo32\":" + opt(f.has_echo, f.m_echo32) +
         (f.has_31 || f.has_bridge || f.has_pcoh || f.has_pgain
              ? ",\"w_fit\":" + opt(f.has_31, f.w_fit) + ",\"w_pgain\":" + opt(f.has_pgain, f.w_pgain) + ",\"w_lead\":" + opt(f.has_lead, f.w_lead) +
                    ",\"m_bridge\":" + opt(f.has_bridge, f.m_bridge) + ",\"m_pcoh\":" + opt(f.has_pcoh, f.m_pcoh)
              : std::string()) +
         "}";
}

// Known-answer checks of the v3 parts (called by the tool's selftest).
inline void scribe_v3_selftest(int& checks, int& fails) {
  auto check = [&](bool ok, const char* what) {
    ++checks;
    if (!ok) {
      ++fails;
      std::printf("FAIL v3: %s\n", what);
    }
  };
  // LZ76: the classic example 0001101001000101 parses 0 | 001 | 10 | 100 | 1000 | 101.
  check(scribe_detail::lz76_phrases("0001101001000101") == 6, "lz76 of the Lempel-Ziv example");
  check(scribe_detail::lz76_phrases("aaaaaaaa") == 2 && scribe_detail::lz76_phrases("") == 0, "lz76 of a run");
  // Spelling model: proper (the single-symbol words and the empty word take at most all the mass) and it prefers words
  // like its training types.
  {
    SpellModel sp;
    sp.train({"cat", "car", "cart", "dog", "door", "dot"});
    double mass = std::exp2(-sp.bits(""));
    for (int c = 1; c < 256; ++c) mass += std::exp2(-sp.bits(std::string(1, static_cast<char>(c))));
    check(mass > 0 && mass <= 1.0 + 1e-9, "spelling model is sub-normalised over short strings");
    check(sp.bits("cat") < sp.bits("xqz") && sp.bits("cart") < sp.bits("zzzz"), "spelling model prefers its own spellings");
  }
  // Suffix array: longest match and counts against a brute-force scan.
  {
    const std::vector<std::string> docs = {"the cat sat on the mat", "a cat and a hat", "mathematics"};
    PackIndex px;
    px.build(docs, 2);
    std::string all;
    for (const std::string& d : docs) all += d + "\n";
    bool ok = true;
    for (const std::string q : {"the cat", "cat sat on the hat", "mat", "hatx", "zebra", "a", "at", "matrix", "e mat"}) {
      size_t best = 0;
      for (size_t i = 0; i < all.size(); ++i) {
        size_t k = 0;
        while (k < q.size() && i + k < all.size() && all[i + k] == q[k]) ++k;
        best = std::max(best, k);
      }
      size_t cnt = 0;
      for (size_t i = 0; i + q.size() <= all.size(); ++i) cnt += all.compare(i, q.size(), q) == 0;
      ok = ok && px.match(reinterpret_cast<const unsigned char*>(q.data()), q.size()) == best &&
           px.count(reinterpret_cast<const unsigned char*>(q.data()), q.size()) == cnt;
    }
    check(ok, "suffix array match and count equal a brute-force scan");
  }
  // SURPRISE: the history makes a repeated sentence cheap; MATCH: copying the pack and the prompt are seen.
  {
    std::vector<std::string> docs;
    for (int i = 0; i < 30; ++i)
      docs.push_back("The quick brown fox jumps over the lazy dog near the river bank today. The old man was walking slowly along the road. "
                     "Green trees grow beside the quiet water. A small boat " + std::to_string(i) + " drifted past the mill.");
    ScribeModel m;
    m.train(docs, 2);
    SurpriseModel sm;
    sm.train(m, docs);
    sm.params.coh_min_bits = 0;  // the toy vocabulary has no rare words
    PackIndex px;
    px.build(docs, 2);
    SurpriseModel::Seq q;
    sm.sequence("The green boat was near the old dog. The green boat was near the old dog.", q);
    double a = 0, b = 0;
    const size_t h = q.s.size() / 2;
    for (size_t i = 0; i < h; ++i) a += q.s[i];
    for (size_t i = h; i < q.s.size(); ++i) b += q.s[i];
    check(q.s.size() == 16 && b < 0.5 * a, "the repeat of a sentence costs under half the bits");
    check(std::isfinite(q.coh) && q.coh > 0, "a repeated sentence has a cohesion gain");
    SurpriseModel::Seq r;
    sm.sequence("Zxqvbk plorf wumbatz.", r);
    check(r.s.size() == 3 && r.s[0] > 20 && r.sp[0] > 10, "non-words are spelled at a high cost");
    const V3Features cp = extract_v3(sm, px, "The quick brown fox jumps over the lazy dog near the river bank today. The old man was walking slowly.",
                                     "Green trees grow beside the quiet water.");
    check(cp.has_m && cp.m_long > 0.9 && cp.m_echo32 == 0, "copied pack text: nearly all bytes inside long pack matches");
    const V3Features ec = extract_v3(sm, px, "Green trees grow beside the quiet water. Then it rained on the hills.", "Green trees grow beside the quiet water.");
    check(ec.has_echo && ec.m_echo32 > 0.5 && ec.m_echo >= ec.m_echo32, "a copied prompt sentence is echo");
    // v3.1: an inserted word fits neither neighbour (fit > 0, above every other word), in code length and as strings.
    {
      SurpriseModel::Extra31 x0, x1;
      sm.extra31("The old man was walking slowly along the road.", "", x0);
      sm.extra31("The old man was walking river slowly along the road.", "", x1);
      double mx0 = -1e9, others = -1e9;
      for (float v : x0.fit) mx0 = std::max(mx0, static_cast<double>(v));
      for (size_t i = 0; i < x1.fit.size(); ++i)
        if (i != 5) others = std::max(others, static_cast<double>(x1.fit[i]));
      check(x0.fit.size() == 9 && x1.fit.size() == 10 && x1.fit[5] > 0 && x1.fit[5] > others && x1.fit[5] > mx0 && !x0.has_pgain,
            "an inserted word has a positive fit, above every word of the sentence");
      const V3Features b0 = extract_v3(sm, px, "The old man was walking slowly along the road today.", "", false, true);
      const V3Features b1 = extract_v3(sm, px, "The old man was walking zebra slowly along the road today.", "", false, true);
      check(b0.has_bridge && b1.has_bridge && b0.m_bridge == 0 && std::fabs(b1.m_bridge - 1.0 / 9.0) < 1e-12 && b1.has_31 && b1.w_fit > b0.w_fit,
            "m_bridge counts the inserted word (and only it); w_fit rises");
      const V3Features e0 = extract_v3(sm, px, "v3.1 is off by default: no fit here.", "a prompt");
      check(!e0.has_31 && !e0.has_bridge && !e0.has_pcoh && !e0.has_pgain && e0.fit.empty(), "extract_v3 without v31 computes no v3.1 statistic");
    }
    // v3.1: the prompt's words save bits (more when they are stuffed in), and are shared strings with the prompt.
    {
      const std::string a = "The old man was walking slowly along the road. Green trees grow beside the water.";
      const std::string a_stuffed = "The old man was walking slowly mill along the road mill. Green trees grow mill beside the water.";
      const V3Features p0 = extract_v3(sm, px, a, "Green trees grow beside the quiet water.", false, true);
      const V3Features p1 = extract_v3(sm, px, a, "The old man was walking.", false, true);
      const V3Features q0 = extract_v3(sm, px, a, "The quiet mill.", false, true);
      const V3Features q1 = extract_v3(sm, px, a_stuffed, "The quiet mill.", false, true);
      check(p0.has_pgain && p1.has_pgain && q0.has_pgain && q1.has_pgain && q0.w_pgain < q1.w_pgain && p0.w_pgain > q0.w_pgain,
            "w_pgain rises with the prompt's words in the text (and stuffed in)");
      check(q0.has_pcoh && q1.has_pcoh && q0.m_pcoh < q1.m_pcoh && p0.m_pcoh > q0.m_pcoh, "m_pcoh rises with the prompt's strings");
      check(p0.has_lead && p1.has_lead && p1.w_lead > p0.w_lead, "w_lead: the first sentence carries the prompt's words");
    }
  }
  // v3.1: the combination does not dilute one judge's confident catch.
  {
    check(std::fabs(combine_v31(0.9, 0.8, 0.7, 1) - 0.8) < 1e-12 && std::fabs(combine_v31(0.5, 0.5, 0.5, 0) - 0.5) < 1e-12 &&
              std::fabs(combine_v31(0.9, 0.4, 0.6, 0) - std::cbrt(0.9 * 0.4 * 0.6)) < 1e-12,
          "power mean: cp = 1 is the mean, cp = 0 the geometric mean");
    check(combine_v31(0.95, 0.95, 1e-9, 0) < 0.01 && combine_v31(0.95, 0.95, 0.0, 0) < 1e-90 && combine_v31(1, 1, 1, -2) == 1.0,
          "one judge's confident catch is not diluted");
    V3Params p;
    p.set_v31();
    check(p.parse("g.w_fit=S0,wt.m_bridge=0.5,cp=-1,fit_q=0.8") && p.grp[W_FIT][0] == 'S' && p.grp[W_FIT][1] == 0 && p.swt[M_BRIDGE] == 0.5 &&
              p.cp == -1 && p.fit_q == 0.8 && !p.parse("g.w_fit=X0") && !p.parse("wt.nosuch=1") && p.v31,
          "V3Params parses the v3.1 keys");
  }
  // The score: a human-typical sample scores high, a far-out one low; the mean of three, the minimum and the flag.
  {
    V3Scorer v;
    std::mt19937_64 r(5);
    std::normal_distribution<double> N(0.0, 1.0);
    auto mk = [&](double shift) {
      V3Features f;
      f.bytes = 500;
      f.n_words = 90;
      f.has_w = f.has_w_coh = f.has_m = f.has_m_coh = true;
      for (int i = 0; i < 90; ++i) f.s.push_back(static_cast<float>(std::max(0.0, 9 + 5 * N(r))));
      f.w_mean = 9.5 + 0.8 * N(r) + shift;
      f.w_sd = 5 + 0.5 * N(r);
      f.w_ac1 = 0.05 * N(r);
      f.w_ctx = 3 + 0.4 * N(r);
      f.w_spell = 1 + 0.3 * N(r);
      f.w_coh = 0.7 + 0.3 * N(r);
      f.m_med = 11 + N(r);
      f.m_p90 = 18 + N(r);
      f.m_long = 0.0;
      f.m_short = 0.01 + 0.002 * N(r);
      f.m_lz = 3 + 0.1 * N(r);
      f.m_tri = 0.25 + 0.05 * N(r);
      f.m_coh = 0.15 + 0.05 * N(r);
      return f;
    };
    for (int i = 0; i < 1000; ++i) v.add_reference(mk(0));
    v.finish();
    const ScoredV3 a = v.score(mk(0)), b = v.score(mk(-12)), c = v.score(mk(30));
    check(a.surprise > 0.5 && b.surprise < 0.5 * a.surprise && c.surprise < 0.5 * a.surprise, "too predictable and too surprising both lower SURPRISE");
    check(a.match > 0.5, "a human-typical sample keeps MATCH high");
    const V3Combined k = combine_v3(0.9, 0.3, 0.6);
    check(std::fabs(k.v3 - 0.6) < 1e-12 && std::fabs(k.min3 - 0.3) < 1e-12 && k.disagree && !combine_v3(0.9, 0.8, 0.7).disagree, "mean, minimum, disagreement");
    V3Params p;
    check(p.parse("w.tau=10,m.wC=2,lz_w=0.25") && p.w.tau == 10 && p.m.wC == 2 && p.lz_w == 0.25 && !p.parse("w.bogus=1"), "V3Params parse");
  }
}

// ===========================================================================
// Old surface score S = H*D*L (native/tools/eval/surface_score.cpp), for comparison
// ===========================================================================

struct SurfaceRef {
  std::unordered_set<std::string> vocab;  // lowercase ASCII-letter words of length 3..32 in the training text
  double ref_d4[2] = {0.9, 0.9}, ref_loop[2] = {0, 0};  // per length bucket
};

inline void surface_add_vocab(SurfaceRef& r, const std::string& text) {
  using namespace scribe_detail;
  for (size_t i = 0; i < text.size();) {
    if (!is_alpha_a(static_cast<unsigned char>(text[i]))) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < text.size() && is_alpha_a(static_cast<unsigned char>(text[j]))) ++j;
    if (j - i >= 3 && j - i <= 32) r.vocab.insert(lower_ascii(text.substr(i, j - i)));
    i = j;
  }
}

inline double surface_S(const SurfaceRef& r, const std::string& s) {
  using namespace scribe_detail;
  int words = 0, hit = 0;
  for (size_t i = 0; i < s.size();) {
    if (!is_alpha_a(static_cast<unsigned char>(s[i]))) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < s.size() && is_alpha_a(static_cast<unsigned char>(s[j]))) ++j;
    if (j - i >= 3) {
      ++words;
      if (r.vocab.count(lower_ascii(s.substr(i, j - i)))) ++hit;
    }
    i = j;
  }
  const double H = words ? static_cast<double>(hit) / words : 0.0;
  const int b = length_bucket(s.size());
  const double D = std::max(0.0, 1.0 - std::fabs(distinct4_bytes(s) - r.ref_d4[b]));
  const double L = std::max(0.0, 1.0 - std::max(0.0, loop8_bytes(s) - r.ref_loop[b]));
  return H * D * L;
}

}  // namespace cypha::eval
