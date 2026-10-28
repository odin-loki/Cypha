// SCRIBE v1: the generated-text quality metric (docs/PLAN.md section 6).
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
// native/tools/surface_score.cpp), and the small statistics helpers the ladder needs (bootstrap
// interval, pairwise ordering accuracy, Pearson and Spearman correlation).
//
// Independence (PLAN V5): the lexicon, the trigram, the IDF and the word vectors come from text the caller
// passes in; the ladder uses text disjoint from everything it scores and from the trie's training text.
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <istream>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
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
// C6: word vectors (PPMI-SVD, native/tools/ppmi_svd.cpp) and SIF sentence vectors
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
// Old surface score S = H*D*L (native/tools/surface_score.cpp), for comparison
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
