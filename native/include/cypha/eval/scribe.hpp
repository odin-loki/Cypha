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
  for (const auto& t : toks) {
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
    lp += m.sentence_logp(ids, npred);
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
    const BucketReference& r = ref[length_bucket(f.bytes)];
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
