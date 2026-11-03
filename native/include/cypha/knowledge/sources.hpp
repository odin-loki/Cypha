#pragma once
// Evidence sources other than Wikidata (S8c, docs/reports/S8C_EVIDENCE_SOURCES.md; Odin, 2026-10-29: "Get another
// evidence source ... datamine and assess data sources reliability"). Each source answers a hypothesis (subject,
// relation, value) with a three-way outcome (S7b): positive (it holds that value), negative (it holds other values of a
// single-valued relation), inconclusive (it does not cover the slot). The answers are evidence for Cypha's own store and
// hypotheses, never an answer for CyphaLM (PLAN rule 10).
//
//   Value / same_value   values matched by Wikidata item when both carry one (an identity link, not a copied fact),
//                        else by year (year relations), by day (date relations) or by a normalised name
//   Table                a source's facts by (normalised subject, relation), with subject aliases (GeoNames)
//   LeadText             a subject's current lead paragraph: positive when the value is named in it, never negative
//   dawid_skene          each source's three-way likelihoods and the share of true items, fitted by EM with no reference,
//                        assuming the sources are independent given the truth. When two sources copy each other the fit
//                        follows the pair and overrates both (a known-answer check plants this): its estimates are an
//                        upper bound for correlated sources, and it cannot say which sources copy

//
// Author: Odin Loch

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cypha/eval/hypothesis.hpp"

namespace cypha::eval::sources {

// ---------------------------------------------------------------------------
// Names, dates and relations
// ---------------------------------------------------------------------------

/// Lower case, with Latin-1 / Latin Extended-A accented letters folded to ASCII (UTF-8 input). Other bytes kept.
inline std::string fold_lower(const std::string& in) {
  std::string o;
  o.reserve(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(in[i]);
    if (c < 0x80) {
      o += static_cast<char>(std::tolower(c));
      continue;
    }
    if ((c == 0xC3 || c == 0xC4 || c == 0xC5) && i + 1 < in.size()) {
      const unsigned cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(in[i + 1]) & 0x3Fu);
      // U+00C0..U+017F
      static const char* const k =
          "aaaaaaaceeeeiiiidnooooo/ouuuuyts"   // U+00C0..U+00DF
          "aaaaaaaceeeeiiiidnooooo/ouuuuyty"   // U+00E0..U+00FF
          "aaaaaaccccccccddddeeeeeeeeeegggg"   // U+0100..U+011F
          "gggghhhhiiiiiiiiiijjjjkkklllllll"   // U+0120..U+013F
          "lllnnnnnnnnnoooooooorrrrrrssssss"   // U+0140..U+015F
          "ssttttttuuuuuuuuuuuuwwyyyzzzzzzs";  // U+0160..U+017F
      if (cp >= 0xC0 && cp <= 0x17F) {
        const char r = k[cp - 0xC0];
        if (r != '/') o += r;
        ++i;
        continue;
      }
    }
    o += static_cast<char>(c);
  }
  return o;
}

/// Normal form of a name: folded, alphanumeric words; no "(...)", no tail after the first comma, no leading "the", and
/// no trailing " language" or " music".
inline std::string norm_name(const std::string& in) {
  std::string s = in;
  if (const std::size_t p = s.find(" ("); p != std::string::npos && p > 0) s.resize(p);
  if (const std::size_t p = s.find(", "); p != std::string::npos && p > 0) s.resize(p);
  const std::string f = fold_lower(s);
  std::string o;
  bool sp = true;
  for (unsigned char c : f) {
    if (std::isalnum(c) || c >= 0x80) o += static_cast<char>(c), sp = false;
    else if (!sp) o += ' ', sp = true;
  }
  while (!o.empty() && o.back() == ' ') o.pop_back();
  if (o.compare(0, 4, "the ") == 0) o.erase(0, 4);
  for (const char* suf : {" language", " music"}) {
    const std::string t = suf;
    if (o.size() > t.size() && o.compare(o.size() - t.size(), t.size(), t) == 0) o.resize(o.size() - t.size());
  }
  return o;
}

/// Normal form of a subject: folded, alphanumeric words, nothing dropped. A subject's title keeps its "(...)" and comma
/// parts, which tell homonyms apart ("Georgia (country)", "Georgia (U.S. state)", "Hanover, New Hampshire").
inline std::string norm_subject(const std::string& in) {
  std::string o;
  bool sp = true;
  for (unsigned char c : fold_lower(in)) {
    if (std::isalnum(c) || c >= 0x80) o += static_cast<char>(c), sp = false;
    else if (!sp) o += ' ', sp = true;
  }
  while (!o.empty() && o.back() == ' ') o.pop_back();
  return o;
}

inline bool year_relation(const std::string& r) {
  return r == "birth_year" || r == "death_year" || r == "founded" || r == "dissolved";
}
inline bool date_relation(const std::string& r) { return r == "birth_date" || r == "death_date"; }

/// Single-valued relations: a source holding other values for the slot counts against a value (a negative result).
inline bool single_valued(const std::string& r) {
  static const std::unordered_set<std::string> k = {"birth_date", "birth_year", "death_date", "death_year", "birth_place",
                                                    "death_place", "founded", "dissolved", "capital", "largest_city",
                                                    "currency", "country"};
  return k.count(r) > 0;
}

struct Value {
  std::string raw, name, qid, day;  // day: "YYYY-MM-DD" when the value is a full date
  int year = -100000;               // a year when the value starts with one (or is one)
};

/// A value from its text and (optional) Wikidata item. Years and days are read from "YYYY", "YYYY-MM-DD" or "-YYYY".
inline Value make_value(const std::string& text, const std::string& qid = "") {
  Value v;
  v.raw = text;
  v.name = norm_name(text);
  v.qid = qid;
  std::size_t i = 0;
  bool neg = false;
  if (i < text.size() && text[i] == '-') neg = true, ++i;
  std::size_t j = i;
  while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
  if (j - i >= 3 && j - i <= 4 && (j == text.size() || text[j] == '-')) {
    v.year = std::stoi(text.substr(i, j - i)) * (neg ? -1 : 1);
    if (!neg && j - i == 4 && text.size() >= j + 6 && text[j] == '-' && std::isdigit(static_cast<unsigned char>(text[j + 1])) &&
        std::isdigit(static_cast<unsigned char>(text[j + 2])) && text[j + 3] == '-' &&
        std::isdigit(static_cast<unsigned char>(text[j + 4])) && std::isdigit(static_cast<unsigned char>(text[j + 5])))
      v.day = text.substr(i, 10);
  }
  return v;
}

/// Do two values of relation r name the same thing? By item when both carry one, by year (year relations), by day (date
/// relations; a year-only side matches on its year) or by normalised name.
inline bool same_value(const std::string& r, const Value& a, const Value& b) {
  if (year_relation(r)) return a.year > -100000 && a.year == b.year;
  if (date_relation(r)) {
    if (!a.day.empty() && !b.day.empty()) return a.day == b.day;
    return a.year > -100000 && a.year == b.year;
  }
  if (!a.qid.empty() && !b.qid.empty()) return a.qid == b.qid;
  return !a.name.empty() && a.name == b.name;
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/// A source's facts, keyed by subject (norm_subject) and relation.
struct Table {
  std::string name;
  std::unordered_map<std::string, std::vector<Value>> slots;  // norm(subject) + '\x1f' + relation
  std::unordered_map<std::string, std::string> alias;         // norm(alias) -> norm(subject) (only for unknown subjects)
  std::unordered_set<std::string> subjects;                   // norm(subject)

  void add(const std::string& subject, const std::string& rel, const std::string& value, const std::string& qid = "") {
    const std::string s = norm_subject(subject);
    if (s.empty()) return;
    subjects.insert(s);
    slots[s + '\x1f' + rel].push_back(make_value(value, qid));
  }
  void add_alias(const std::string& a, const std::string& subject) {
    const std::string k = norm_subject(a), s = norm_subject(subject);
    if (!k.empty() && !s.empty() && k != s) alias.emplace(k, s);
  }
  /// The slot's values, or nullptr when the source does not cover it.
  const std::vector<Value>* find(const std::string& subject, const std::string& rel) const {
    std::string s = norm_subject(subject);
    auto it = slots.find(s + '\x1f' + rel);
    if (it == slots.end() && !subjects.count(s)) {
      const auto a = alias.find(s);
      if (a != alias.end()) it = slots.find(a->second + '\x1f' + rel);
    }
    return it == slots.end() ? nullptr : &it->second;
  }
};

/// A table source's answer for (s, r, v).
inline TestOutcome answer(const Table& t, const std::string& s, const std::string& r, const Value& v) {
  const std::vector<Value>* vs = t.find(s, r);
  if (!vs || vs->empty()) return TestOutcome::Inconclusive;
  for (const Value& x : *vs)
    if (same_value(r, x, v)) return TestOutcome::Positive;
  return single_valued(r) ? TestOutcome::Negative : TestOutcome::Inconclusive;
}

/// A subject's current lead paragraph. Positive when the value's name (or, for year and date relations, its year) occurs
/// on word boundaries; inconclusive otherwise. Never negative: a lead paragraph leaves most facts out.
struct LeadText {
  std::unordered_map<std::string, std::string> text;  // norm_subject(subject) -> " " + normalised words + " "

  void add(const std::string& subject, const std::string& extract) {
    const std::string s = norm_subject(subject);
    std::string w;
    bool sp = true;
    for (unsigned char c : fold_lower(extract)) {
      if (std::isalnum(c) || c >= 0x80) w += static_cast<char>(c), sp = false;
      else if (!sp) w += ' ', sp = true;
    }
    if (!s.empty()) text[s] = " " + w + " ";
  }
  TestOutcome answer(const std::string& s, const std::string& r, const Value& v) const {
    const auto it = text.find(norm_subject(s));
    if (it == text.end()) return TestOutcome::Inconclusive;
    std::string needle;
    if (year_relation(r) || date_relation(r)) {
      if (v.year <= -100000) return TestOutcome::Inconclusive;
      needle = std::to_string(v.year < 0 ? -v.year : v.year);
    } else {
      needle = v.name;
    }
    if (needle.size() < 3) return TestOutcome::Inconclusive;
    return it->second.find(" " + needle + " ") != std::string::npos ? TestOutcome::Positive : TestOutcome::Inconclusive;
  }
};

inline int outcome_index(TestOutcome o) { return o == TestOutcome::Positive ? 0 : o == TestOutcome::Negative ? 1 : 2; }

// ---------------------------------------------------------------------------
// S8d: calibration on the generators' own graded hypotheses (docs/reports/S8D_PROMOTION.md)
// ---------------------------------------------------------------------------

/// P(positive | true) / P(positive | false); 0 without likelihoods.
inline double positive_lr(const OutcomeLikelihoods& L) { return L.set() && L.f[0] > 0.0 ? L.t[0] / L.f[0] : 0.0; }

/// A (source, group) cell's likelihoods (add-one): its own tally when it holds at least `min_n` graded true and `min_n`
/// graded false items, else the source's tally pooled over its groups. `own` says which was used.
inline OutcomeLikelihoods calibrated_likelihoods(const TestTally& cell, const TestTally& pooled, double min_n = 10.0, bool* own = nullptr) {
  const bool use = cell.t_pos + cell.t_neg + cell.t_inc >= min_n && cell.f_pos + cell.f_neg + cell.f_inc >= min_n;
  if (own) *own = use;
  return likelihoods_from(use ? cell : pooled);
}

/// A source may carry a promotion (give the positive result `decide` needs) only where its positive likelihood ratio on
/// the generators' graded hypotheses is at least `min_lr` (S8c's bar, measured on the errors that matter).
inline bool may_carry_promotion(const OutcomeLikelihoods& L, double min_lr = 10.0) { return positive_lr(L) >= min_lr; }

// ---------------------------------------------------------------------------
// Reliability without a reference: Dawid-Skene
// ---------------------------------------------------------------------------

/// Each source's P(outcome | true) and P(outcome | false) (positive, negative, inconclusive) and the share of true items.
struct DSFit {
  double prior = 0.5;
  std::vector<std::array<double, 3>> t, f;
  int iterations = 0;
  double loglik = 0.0;
};

/// P(item true | its outcomes) under a fit.
inline std::vector<double> ds_posterior(const DSFit& m, const std::vector<std::vector<int>>& y) {
  std::vector<double> post(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) {
    double lt = std::log(m.prior), lf = std::log(1.0 - m.prior);
    for (std::size_t j = 0; j < y[i].size(); ++j) lt += std::log(m.t[j][y[i][j]]), lf += std::log(m.f[j][y[i][j]]);
    post[i] = 1.0 / (1.0 + std::exp(lf - lt));
  }
  return post;
}

/// Dawid-Skene by EM. y[i][j] is source j's outcome on item i (0 positive, 1 negative, 2 inconclusive). Started from the
/// items' share of positive outcomes among their conclusive ones; add-a smoothing in every M step. Stops when the
/// log-likelihood moves by less than 1e-9 per item, or after max_iter steps.
inline DSFit dawid_skene(const std::vector<std::vector<int>>& y, std::size_t sources, int max_iter = 1000, double a = 1.0) {
  DSFit m;
  m.t.assign(sources, {1.0 / 3, 1.0 / 3, 1.0 / 3});
  m.f.assign(sources, {1.0 / 3, 1.0 / 3, 1.0 / 3});
  std::vector<double> post(y.size(), 0.5);
  for (std::size_t i = 0; i < y.size(); ++i) {
    double p = 0, n = 0;
    for (int o : y[i]) p += o == 0, n += o == 1;
    post[i] = (p + 0.5) / (p + n + 1.0);
  }
  double last = -1e300;
  for (int it = 0; it < max_iter; ++it) {
    // M step
    double sp = 0;
    for (double q : post) sp += q;
    m.prior = std::min(1.0 - 1e-6, std::max(1e-6, (sp + a) / (static_cast<double>(y.size()) + 2 * a)));
    for (std::size_t j = 0; j < sources; ++j) {
      std::array<double, 3> ct{a, a, a}, cf{a, a, a};
      for (std::size_t i = 0; i < y.size(); ++i) ct[y[i][j]] += post[i], cf[y[i][j]] += 1.0 - post[i];
      const double nt = ct[0] + ct[1] + ct[2], nf = cf[0] + cf[1] + cf[2];
      for (int k = 0; k < 3; ++k) m.t[j][k] = ct[k] / nt, m.f[j][k] = cf[k] / nf;
    }
    // E step
    double ll = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      double lt = std::log(m.prior), lf = std::log(1.0 - m.prior);
      for (std::size_t j = 0; j < sources; ++j) lt += std::log(m.t[j][y[i][j]]), lf += std::log(m.f[j][y[i][j]]);
      const double mx = std::max(lt, lf);
      ll += mx + std::log(std::exp(lt - mx) + std::exp(lf - mx));
      post[i] = 1.0 / (1.0 + std::exp(lf - lt));
    }
    m.iterations = it + 1;
    m.loglik = ll;
    if (std::fabs(ll - last) < 1e-9 * std::max<std::size_t>(1, y.size())) break;
    last = ll;
  }
  // Label switching: the "true" class is the one in which sources say positive more often than in the other.
  double pt = 0, pf = 0;
  for (std::size_t j = 0; j < sources; ++j) pt += m.t[j][0], pf += m.f[j][0];
  if (pt < pf) std::swap(m.t, m.f), m.prior = 1.0 - m.prior;
  return m;
}

}  // namespace cypha::eval::sources
