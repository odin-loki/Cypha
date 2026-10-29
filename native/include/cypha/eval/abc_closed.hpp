#pragma once
// G12's closed form (Swanson's ABC, given A and C), kept after S11b removed G12's open form
// (docs/reports/S11B_ABC_DISCOVERY.md; S6c in docs/reports/S6_HYPOTHESIS_SCORING.md). A and C are linked through B when
// B shares a document with A and another with C: score(A, C) = sum over B of idf(B) * PPMI(A, B) * PPMI(B, C), hub
// B left out. The documents that hold both A and C are left out before counting, so the score is indirect evidence
// only and never reads a document where the two appear together. On a pair that never shares a document it is S11b's
// closed form exactly. It scores Cypha's own hypotheses (S6's P0, S7's T7) and never answers a question (PLAN rule 10).
//
//   Cooc      documents as sets of entity ids, and each entity's documents (its postings).
//   Closed    the leave-out closed form for one pair, with its strongest B.
//   Helpers   wiki link targets, redirect resolution, titles that are dates or lists.
//
// Author: Odin Loch

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cypha::eval::abc {

// Positive pointwise mutual information over n documents: max(0, ln(n * nxy / (nx * ny))).
inline double ppmi(double n, double nxy, double nx, double ny) {
  if (n <= 0 || nxy <= 0 || nx <= 0 || ny <= 0) return 0.0;
  const double v = std::log(n * nxy / (nx * ny));
  return v > 0 ? v : 0.0;
}

struct Cooc {
  std::vector<std::vector<std::uint32_t>> docs;      // each document's entities, sorted, unique
  std::vector<std::vector<std::uint32_t>> postings;  // each entity's documents, sorted

  std::size_t n_docs() const { return docs.size(); }
  std::size_t n_entities() const { return postings.size(); }
  std::uint32_t df(std::uint32_t e) const { return static_cast<std::uint32_t>(postings[e].size()); }

  // Documents given as entity id lists (ids below n_ent; duplicates allowed). Each document becomes a sorted set.
  static Cooc build(std::vector<std::vector<std::uint32_t>> ds, std::size_t n_ent) {
    Cooc c;
    c.postings.assign(n_ent, {});
    for (auto& d : ds) {
      std::sort(d.begin(), d.end());
      d.erase(std::unique(d.begin(), d.end()), d.end());
    }
    c.docs = std::move(ds);
    for (std::size_t i = 0; i < c.docs.size(); ++i)
      for (std::uint32_t e : c.docs[i]) c.postings[e].push_back(static_cast<std::uint32_t>(i));
    return c;
  }

  // The documents holding both x and y (a merge of the two postings).
  std::vector<std::uint32_t> shared(std::uint32_t x, std::uint32_t y) const {
    std::vector<std::uint32_t> out;
    const auto& a = postings[x];
    const auto& b = postings[y];
    for (std::size_t i = 0, j = 0; i < a.size() && j < b.size();) {
      if (a[i] < b[j]) ++i;
      else if (b[j] < a[i]) ++j;
      else out.push_back(a[i]), ++i, ++j;
    }
    return out;
  }
};

struct ClosedResult {
  double score = 0.0;
  std::size_t links = 0;                               // the B with a positive term
  std::size_t left_out = 0;                            // documents holding both A and C
  std::vector<std::pair<std::uint32_t, double>> top;   // the strongest B, largest first
};

// The leave-out closed form. usable[e] == 0 keeps e out of the B role (dates and lists); a B needs min_df..hub_cap
// documents once the shared ones are gone.
class Closed {
 public:
  Closed(const Cooc& c, std::vector<char> usable, std::uint32_t hub_cap = 200, std::uint32_t min_df = 2)
      : c_(c), usable_(std::move(usable)), hub_(hub_cap), min_df_(min_df), ca_(c.n_entities(), 0), cc_(c.n_entities(), 0),
        dec_(c.n_entities(), 0), skip_(c.n_docs(), 0) {}

  ClosedResult score(std::uint32_t a, std::uint32_t c, std::size_t top = 3) const {
    ClosedResult out;
    if (a == c) return out;
    const std::vector<std::uint32_t> both = c_.shared(a, c);
    out.left_out = both.size();
    const double n = static_cast<double>(c_.n_docs() - both.size());
    const double na = static_cast<double>(c_.df(a) - both.size()), nc = static_cast<double>(c_.df(c) - both.size());
    if (na <= 0 || nc <= 0 || n <= 0) return out;
    std::vector<std::uint32_t> touched_dec, touched_a, touched_c;
    for (std::uint32_t d : both) {
      skip_[d] = 1;
      for (std::uint32_t y : c_.docs[d])
        if (dec_[y]++ == 0) touched_dec.push_back(y);
    }
    auto count = [&](std::uint32_t x, std::vector<std::uint32_t>& cnt, std::vector<std::uint32_t>& touched) {
      for (std::uint32_t d : c_.postings[x]) {
        if (skip_[d]) continue;
        for (std::uint32_t y : c_.docs[d])
          if (cnt[y]++ == 0) touched.push_back(y);
      }
    };
    count(a, ca_, touched_a);
    count(c, cc_, touched_c);
    std::vector<std::pair<std::uint32_t, double>> terms;
    for (std::uint32_t b : touched_a) {
      if (b == a || b == c || !usable_[b] || cc_[b] == 0) continue;
      const std::uint32_t nb = c_.df(b) - dec_[b];
      if (nb < min_df_ || nb > hub_) continue;
      const double t = std::log(n / nb) * ppmi(n, ca_[b], na, nb) * ppmi(n, cc_[b], nb, nc);
      if (t > 0) terms.push_back({b, t}), out.score += t;
    }
    out.links = terms.size();
    std::sort(terms.begin(), terms.end(), [](const auto& x, const auto& y) { return x.second != y.second ? x.second > y.second : x.first < y.first; });
    if (terms.size() > top) terms.resize(top);
    out.top = std::move(terms);
    for (std::uint32_t y : touched_a) ca_[y] = 0;
    for (std::uint32_t y : touched_c) cc_[y] = 0;
    for (std::uint32_t y : touched_dec) dec_[y] = 0;
    for (std::uint32_t d : both) skip_[d] = 0;
    return out;
  }

 private:
  const Cooc& c_;
  std::vector<char> usable_;
  std::uint32_t hub_, min_df_;
  mutable std::vector<std::uint32_t> ca_, cc_, dec_;
  mutable std::vector<char> skip_;
};

// ---------------------------------------------------------------------------
// Wiki text helpers (as S11b's)
// ---------------------------------------------------------------------------

// The targets of every [[target]] and [[target|label]] in wiki text, "#section" dropped, "_" made a space, the first
// letter capitalised. A target whose colon is followed by a non-space is another namespace or a language and is
// dropped. A leading ":" is removed.
inline std::vector<std::string> link_targets(const std::string& text) {
  std::vector<std::string> out;
  for (std::size_t i = 0; (i = text.find("[[", i)) != std::string::npos;) {
    i += 2;
    std::size_t e = i;
    while (e < text.size() && text[e] != ']' && text[e] != '|' && text[e] != '[' && text[e] != '\n') ++e;
    if (e >= text.size() || text[e] == '[' || text[e] == '\n') continue;
    std::string t = text.substr(i, e - i);
    const std::size_t hash = t.find('#');
    if (hash != std::string::npos) t.resize(hash);
    if (!t.empty() && t[0] == ':') t.erase(0, 1);
    for (char& ch : t)
      if (ch == '_') ch = ' ';
    const std::size_t b0 = t.find_first_not_of(' '), b1 = t.find_last_not_of(' ');
    if (b0 == std::string::npos) continue;
    t = t.substr(b0, b1 - b0 + 1);
    const std::size_t colon = t.find(':');
    if (colon != std::string::npos && colon + 1 < t.size() && t[colon + 1] != ' ') continue;
    if (t[0] >= 'a' && t[0] <= 'z') t[0] = static_cast<char>(t[0] - 'a' + 'A');
    out.push_back(std::move(t));
  }
  return out;
}

// Follows a redirect map (title -> target) at most `hops` times.
inline std::string resolve(const std::string& t, const std::unordered_map<std::string, std::string>& redirects, int hops = 3) {
  std::string cur = t;
  for (int i = 0; i < hops; ++i) {
    const auto it = redirects.find(cur);
    if (it == redirects.end() || it->second == cur) break;
    cur = it->second;
  }
  return cur;
}

// Titles that are dates or lists: a year ("1952", "300 BC", "AD 79"), a decade ("1990s"), a century ("19th century"),
// a day of the year ("March 4", "4 March"), a month name, or "List of ...". They are left out of every role.
inline bool date_or_list_title(const std::string& t) {
  static const char* const kMonths[12] = {"January", "February", "March", "April", "May", "June",
                                          "July", "August", "September", "October", "November", "December"};
  auto digits = [](const std::string& s) { return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; }); };
  auto is_month = [&](const std::string& s) {
    for (const char* m : kMonths)
      if (s == m) return true;
    return false;
  };
  if (t.rfind("List of ", 0) == 0 || t.rfind("Lists of ", 0) == 0) return true;
  if (digits(t)) return true;
  if (t.size() > 3 && t.compare(t.size() - 3, 3, " BC") == 0 && digits(t.substr(0, t.size() - 3))) return true;
  if (t.size() > 4 && t.compare(t.size() - 4, 4, " BCE") == 0 && digits(t.substr(0, t.size() - 4))) return true;
  if (t.rfind("AD ", 0) == 0 && digits(t.substr(3))) return true;
  if (t.size() > 1 && t.back() == 's' && digits(t.substr(0, t.size() - 1))) return true;  // 1990s
  const std::size_t sp = t.find(' ');
  if (is_month(t)) return true;
  if (sp != std::string::npos) {
    const std::string a = t.substr(0, sp), b = t.substr(sp + 1);
    if ((is_month(a) && digits(b) && b.size() <= 2) || (digits(a) && a.size() <= 2 && is_month(b))) return true;
    if (b == "century" || b == "century BC" || b == "millennium" || b == "millennium BC") {
      if (a.size() >= 3 && digits(a.substr(0, a.size() - 2))) return true;
    }
  }
  return false;
}

}  // namespace cypha::eval::abc
