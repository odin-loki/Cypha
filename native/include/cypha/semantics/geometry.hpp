// Cypha semantics geometry: the Cypha side of docs/EMBEDDING_GEOMETRY.md (sections 4.2, 4.5, 5.1, 7.1, 9.3).
//
// Header-only, standard library plus cypha/eval/ecs.hpp (the mask's "ECS >= moderate" grade rule).
// Deterministic: every random choice goes through a seeded std::mt19937_64, every container that is
// iterated is ordered, and every update is computed from the pre-update state and then applied.
//
//   ItemStore           per item: mean mu, diagonal variance Sigma, evidence count n, world prior theta0;
//                       fp32 in memory, fp16 export / import, FNV-1a state hash
//   ParaphraseSet       known synonym / alias / paraphrase pairs, admitted at ECS grade >= Moderate
//   pull_push_step      the section 4.2 rule: masked softmax at tau, tangent cosine gradients,
//                       natural-gradient step K = Sigma / (n + kappa0), mirror updates, optional
//                       query-query / document-document uniformity terms, MDL shrinkage, online variance;
//                       optional evidence count weighted by information (StepOptions::info_count, S-1c)
//                       pull_push_step and distill_step reuse one scratch buffer per thread (no heap
//                       allocation per call once warm; S-1c), with the arithmetic unchanged
//   distill_step        section 5.1 relational softmax distillation, force (p_T - p_S) / tau_S per pair,
//                       and the combined loss alpha * KL + (1 - alpha) * InfoNCE
//   MRL                 PCA rotation that orders dimensions by variance (cyclic Jacobi), nested loss
//                       over the prefixes {32, 64, 128, 256} capped at d
//   post-processing     centring, PCA / ZCA whitening, CSLS neighbour scoring
//   metrics (7.1)       pair Spearman (average ranks for ties), stratified pair sampling, kNN@k Jaccard,
//                       recall of the teacher's top 10 in the student's top 50, linear CKA,
//                       k-occurrence skewness (hubness), force-match KL, the confidence check
//
// Spec choices where the doc leaves room are listed in docs/reports/QE_GEOMETRY_LIBRARY.md.
//
// Author: Odin Loch
#pragma once

#include "cypha/eval/ecs.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cypha::semantics {

// ---------------------------------------------------------------------------------------------
// fp16 (IEEE binary16). to-half is the round-to-nearest-even routine of
// src/cyphalm/component_dump.cpp (internal there, so it is restated here for a header-only build).
// ---------------------------------------------------------------------------------------------

inline std::uint16_t fp32_to_fp16(float f) {
  std::uint32_t x = 0;
  std::memcpy(&x, &f, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  x &= 0x7fffffffu;
  if (x >= 0x7f800000u) return static_cast<std::uint16_t>(sign | (x > 0x7f800000u ? 0x7e00u : 0x7c00u));
  if (x >= 0x477ff000u) return static_cast<std::uint16_t>(sign | 0x7c00u);  // rounds past 65504
  if (x < 0x38800000u) {                                                    // below 2^-14: subnormal or 0
    if (x < 0x33000000u) return static_cast<std::uint16_t>(sign);           // below 2^-25
    const std::uint32_t e = x >> 23;
    const std::uint32_t m = (x & 0x7fffffu) | 0x800000u;
    const std::uint32_t shift = 126u - e;
    std::uint32_t h = m >> shift;
    const std::uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(sign | h);
  }
  std::uint32_t h = (x >> 13) - (112u << 10);
  const std::uint32_t rem = x & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return static_cast<std::uint16_t>(sign | h);
}

inline float fp16_to_fp32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1fu;
  std::uint32_t mant = h & 0x3ffu;
  std::uint32_t x = 0;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {  // subnormal: normalise
      int e = -1;
      do {
        ++e;
        mant <<= 1;
      } while ((mant & 0x400u) == 0);
      x = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((mant & 0x3ffu) << 13);
    }
  } else if (exp == 0x1fu) {
    x = sign | 0x7f800000u | (mant << 13);
  } else {
    x = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float f = 0.0f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

inline std::vector<std::uint16_t> to_fp16(const std::vector<float>& v) {
  std::vector<std::uint16_t> o(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) o[i] = fp32_to_fp16(v[i]);
  return o;
}

inline std::vector<float> from_fp16(const std::vector<std::uint16_t>& v) {
  std::vector<float> o(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) o[i] = fp16_to_fp32(v[i]);
  return o;
}

// ---------------------------------------------------------------------------------------------
// Small vector helpers
// ---------------------------------------------------------------------------------------------

namespace detail {

inline double dot(const float* a, const float* b, int m) {
  double s = 0.0;
  for (int j = 0; j < m; ++j) s += static_cast<double>(a[j]) * static_cast<double>(b[j]);
  return s;
}

inline double norm(const float* a, int m) { return std::sqrt(dot(a, a, m)); }

inline double cosine(const float* a, const float* b, int m) {
  const double na = norm(a, m), nb = norm(b, m);
  if (na <= 0.0 || nb <= 0.0) return 0.0;
  return dot(a, b, m) / (na * nb);
}

inline void normalise(float* a, int d) {
  const double n = norm(a, d);
  if (n <= 0.0) return;
  for (int j = 0; j < d; ++j) a[j] = static_cast<float>(a[j] / n);
}

inline std::uint64_t fnv1a(std::uint64_t h, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

inline double normal01(std::mt19937_64& eng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  return nd(eng);
}

/// Sparse per-item accumulator for one step (<= a few hundred touched items); ordered by first touch.
/// g[k] belongs to ids[k] for k < ids.size(); rows past that are kept for reuse (reset() keeps them).
struct DeltaAcc {
  int d = 0;
  std::vector<int> ids;
  std::vector<std::vector<double>> g;
  void reset(int dim) {
    d = dim;
    ids.clear();
  }
  std::vector<double>& at(int id) {
    for (std::size_t k = 0; k < ids.size(); ++k)
      if (ids[k] == id) return g[k];
    const std::size_t k = ids.size();
    ids.push_back(id);
    if (k < g.size())
      g[k].assign(static_cast<std::size_t>(d), 0.0);
    else
      g.emplace_back(static_cast<std::size_t>(d), 0.0);
    return g[k];
  }
};

}  // namespace detail

/// Cosine similarity on the first m coordinates.
inline double cosine(const float* a, const float* b, int m) { return detail::cosine(a, b, m); }

/// Tangent cosine gradient: d cos(a, b) / d a on the first m coordinates,
/// = (b/|b| - cos * a/|a|) / |a|, which is perpendicular to a. Writes m values to out.
inline void cosine_grad(const float* a, const float* b, int m, double* out) {
  const double na = detail::norm(a, m), nb = detail::norm(b, m);
  if (na <= 0.0 || nb <= 0.0) {
    for (int j = 0; j < m; ++j) out[j] = 0.0;
    return;
  }
  const double c = detail::dot(a, b, m) / (na * nb);
  for (int j = 0; j < m; ++j) out[j] = (b[j] / nb - c * a[j] / na) / na;
}

/// masked_softmax written into p (resized to s.size()); used by the steps so that p's buffer is reused.
inline void masked_softmax_into(const std::vector<double>& s, double tau, const std::vector<char>& mask,
                                std::vector<double>& p) {
  p.assign(s.size(), 0.0);
  double mx = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < s.size(); ++i)
    if (mask.empty() || mask[i]) mx = std::max(mx, s[i] / tau);
  if (!std::isfinite(mx)) return;
  double z = 0.0;
  for (std::size_t i = 0; i < s.size(); ++i)
    if (mask.empty() || mask[i]) z += (p[i] = std::exp(s[i] / tau - mx));
  for (double& v : p) v /= z;
}

/// Softmax of logits s / tau over the entries with mask != 0 (masked entries get weight 0).
/// An empty mask means every entry is kept.
inline std::vector<double> masked_softmax(const std::vector<double>& s, double tau,
                                          const std::vector<char>& mask = {}) {
  std::vector<double> p;
  masked_softmax_into(s, tau, mask, p);
  return p;
}

/// The Matryoshka prefixes of section 4.5: {32, 64, 128, 256} below d, then d itself.
inline std::vector<int> mrl_prefixes(int d) {
  std::vector<int> out;
  for (int m : {32, 64, 128, 256})
    if (m < d) out.push_back(m);
  out.push_back(d);
  return out;
}

// ---------------------------------------------------------------------------------------------
// Known paraphrase / synonym / alias pairs (the second half of the section 4.2 mask)
// ---------------------------------------------------------------------------------------------

class ParaphraseSet {
 public:
  /// Admits the pair only when its ECS grade is at least Moderate (section 4.2: "lexical pack, ECS >= moderate").
  bool add(int i, int j, cypha::eval::Grade g = cypha::eval::Grade::Moderate) {
    if (i == j || !cypha::eval::grade_at_least(g, cypha::eval::Grade::Moderate)) return false;
    return pairs_.insert(key(i, j)).second;
  }
  bool contains(int i, int j) const { return pairs_.count(key(i, j)) != 0; }
  std::size_t size() const { return pairs_.size(); }

 private:
  static std::uint64_t key(int i, int j) {
    const auto a = static_cast<std::uint32_t>(std::min(i, j)), b = static_cast<std::uint32_t>(std::max(i, j));
    return (static_cast<std::uint64_t>(a) << 32) | b;
  }
  std::set<std::uint64_t> pairs_;
};

// ---------------------------------------------------------------------------------------------
// Item store (section 4.1)
// ---------------------------------------------------------------------------------------------

struct Fp16Snapshot {
  int n_items = 0;
  int d = 0;
  std::vector<std::uint16_t> mu, var, world_prior, item_prior;  // item_prior empty when not set
  std::vector<float> n;                                          // counts stay fp32 (fp16 tops out at 65504)
};

struct ItemStore {
  int n_items = 0;
  int d = 0;
  std::vector<float> mu;           // n_items * d, kept on the unit sphere by the steps (retraction)
  std::vector<float> var;          // n_items * d, diagonal Sigma
  std::vector<float> n;            // evidence count per item
  std::vector<float> world_prior;  // theta0, d; shared by every item without its own prior
  std::vector<float> item_prior;   // optional n_items * d per-item theta0 (e.g. teacher-initialised); empty = shared

  ItemStore() = default;
  ItemStore(int items, int dim, float var0)
      : n_items(items), d(dim),
        mu(static_cast<std::size_t>(items) * static_cast<std::size_t>(dim), 0.0f),
        var(static_cast<std::size_t>(items) * static_cast<std::size_t>(dim), var0),
        n(static_cast<std::size_t>(items), 0.0f), world_prior(static_cast<std::size_t>(dim), 0.0f) {}

  /// Random unit means, variance var0 (default 1/d), n = 0, theta0 = 0.
  static ItemStore random(int items, int dim, std::uint64_t seed, float var0 = -1.0f) {
    ItemStore s(items, dim, var0 > 0.0f ? var0 : 1.0f / static_cast<float>(dim));
    std::mt19937_64 eng(seed);
    for (int i = 0; i < items; ++i) {
      float* m = s.mean(i);
      for (int j = 0; j < dim; ++j) m[j] = static_cast<float>(detail::normal01(eng));
      detail::normalise(m, dim);
    }
    return s;
  }

  float* mean(int i) { return mu.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d); }
  const float* mean(int i) const { return mu.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d); }
  float* variance(int i) { return var.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d); }
  const float* variance(int i) const { return var.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d); }
  const float* prior(int i) const {
    return item_prior.empty() ? world_prior.data()
                              : item_prior.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
  }

  void set_world_prior(const float* p) { std::copy(p, p + d, world_prior.begin()); }
  /// theta0 = the normalised mean of all item means (SEMANTICS 2: the WorldPrior).
  void set_world_prior_to_mean() {
    std::vector<double> m(static_cast<std::size_t>(d), 0.0);
    for (int i = 0; i < n_items; ++i)
      for (int j = 0; j < d; ++j) m[static_cast<std::size_t>(j)] += mean(i)[j];
    for (int j = 0; j < d; ++j) world_prior[static_cast<std::size_t>(j)] = static_cast<float>(m[static_cast<std::size_t>(j)] / std::max(1, n_items));
    detail::normalise(world_prior.data(), d);
  }
  void set_item_prior(int i, const float* p) {
    if (item_prior.empty()) {
      item_prior.resize(mu.size());
      for (int k = 0; k < n_items; ++k)
        std::copy(world_prior.begin(), world_prior.end(), item_prior.begin() + static_cast<std::ptrdiff_t>(k) * d);
    }
    std::copy(p, p + d, item_prior.begin() + static_cast<std::ptrdiff_t>(i) * d);
  }

  /// Mean of Sigma over the dimensions: the item's scalar uncertainty.
  double mean_variance(int i) const {
    double s = 0.0;
    for (int j = 0; j < d; ++j) s += variance(i)[j];
    return s / d;
  }

  Fp16Snapshot export_fp16() const {
    Fp16Snapshot s;
    s.n_items = n_items;
    s.d = d;
    s.mu = to_fp16(mu);
    s.var = to_fp16(var);
    s.world_prior = to_fp16(world_prior);
    s.item_prior = to_fp16(item_prior);
    s.n = n;
    return s;
  }
  static ItemStore import_fp16(const Fp16Snapshot& s) {
    const std::size_t nd = static_cast<std::size_t>(s.n_items) * static_cast<std::size_t>(s.d);
    if (s.mu.size() != nd || s.var.size() != nd || s.n.size() != static_cast<std::size_t>(s.n_items) ||
        s.world_prior.size() != static_cast<std::size_t>(s.d) || (!s.item_prior.empty() && s.item_prior.size() != nd))
      throw std::invalid_argument("geometry: fp16 snapshot has inconsistent sizes");
    ItemStore o;
    o.n_items = s.n_items;
    o.d = s.d;
    o.mu = from_fp16(s.mu);
    o.var = from_fp16(s.var);
    o.world_prior = from_fp16(s.world_prior);
    o.item_prior = from_fp16(s.item_prior);
    o.n = s.n;
    return o;
  }

  /// FNV-1a over every stored byte: bit-identical states give identical hashes.
  std::uint64_t state_hash() const {
    std::uint64_t h = 1469598103934665603ull;
    h = detail::fnv1a(h, &n_items, sizeof(n_items));
    h = detail::fnv1a(h, &d, sizeof(d));
    h = detail::fnv1a(h, mu.data(), mu.size() * sizeof(float));
    h = detail::fnv1a(h, var.data(), var.size() * sizeof(float));
    h = detail::fnv1a(h, n.data(), n.size() * sizeof(float));
    h = detail::fnv1a(h, world_prior.data(), world_prior.size() * sizeof(float));
    h = detail::fnv1a(h, item_prior.data(), item_prior.size() * sizeof(float));
    return h;
  }
};

// ---------------------------------------------------------------------------------------------
// Step options and statistics
// ---------------------------------------------------------------------------------------------

struct StepOptions {
  // section 4.2
  double tau = 0.05;          // InfoNCE temperature
  double delta = 0.1;         // false-negative margin, copied from Qwen3-Embedding
  bool use_mask = true;       // both mask rules (margin and known paraphrase)
  bool uniformity = false;    // add the query-query and document-document terms of the examples
  double eta = 1.0;           // step size; the force carries 1/tau as in section 3.2
  double kappa0 = 10.0;       // prior evidence: K = Sigma / (n + kappa0); also the prior weight of Sigma0
  double lambda_mdl = 0.0;    // MDL shrinkage: mu <- mu + lambda_mdl / (n + 1) * (theta0 - mu)
  double var_rate_floor = 0.0;  // lower bound on the online variance rate (0: pure running mean)
  float var_floor = 1e-6f;
  bool renormalise = true;    // retraction onto the unit sphere after each step
  bool update_variance = true;
  // S-1c: pull_push_step adds w = 1 - H(p) / ln k to the evidence count n (anchor and positive) instead of 1,
  // where p is the anchor's masked InfoNCE softmax at the largest prefix and k its unmasked entries (w = 1 for
  // k < 2). A flat softmax (random partners) adds little evidence, so K = Sigma / (n + kappa0) does not decay
  // before structure forms. Off by default: every default output is unchanged.
  bool info_count = false;
  // section 5.1
  double tau_T = 0.05;
  double tau_S = 0.05;
  double alpha = 1.0;         // alpha * KL + (1 - alpha) * InfoNCE (InfoNCE needs a positive)
  // section 4.5: Matryoshka prefixes; empty = the full d only. mrl_prefixes(d) gives {32, 64, 128, 256} capped at d.
  std::vector<int> prefixes;
};

struct StepStats {
  double loss = 0.0;   // mean over prefixes
  double p_pos = 0.0;  // InfoNCE positive weight at the last prefix
  int masked = 0;      // candidates masked, summed over prefixes
  double info = 1.0;   // 1 - H(p) / ln k of the InfoNCE softmax at the last prefix (1 when k < 2)
};

struct PullPushExample {
  int a = -1;               // anchor (query side)
  int b = -1;               // positive (document side)
  std::vector<int> cands;   // candidate negatives C for the anchor: confusions, unigram^0.75, mined
  std::vector<int> qq;      // other queries: s(a, q_j) terms (uniformity, used when opt.uniformity)
  std::vector<int> dd;      // other documents: s(b, d_j) terms (uniformity, used when opt.uniformity)
};

struct DistillExample {
  int a = -1;
  std::vector<int> cands;
  std::vector<float> teacher_sims;  // s_T(a, c) for each candidate
  int positive = -1;                // the mined positive for the InfoNCE half (alpha < 1); -1 = none
};

namespace detail {

inline void resolve_prefixes_into(const StepOptions& o, int d, std::vector<int>& p) {
  p.clear();
  for (int m : o.prefixes)
    if (m > 0) p.push_back(std::min(m, d));
  if (p.empty()) p.push_back(d);
  std::sort(p.begin(), p.end());
  p.erase(std::unique(p.begin(), p.end()), p.end());
}

inline std::vector<int> resolve_prefixes(const StepOptions& o, int d) {
  std::vector<int> p;
  resolve_prefixes_into(o, d, p);
  return p;
}

/// Buffers reused across calls of pull_push_step and distill_step (one per thread): no heap allocation per call
/// once they have grown. The arithmetic is the same as with fresh vectors (S-1c).
struct StepScratch {
  DeltaAcc acc;
  std::vector<int> pref, ti, tj;
  std::vector<char> mask;
  std::vector<double> tmp, s, p, ra, rb, r, rc, step, sT, pT, sS, pS, tni, tnj;
};

inline StepScratch& step_scratch() {
  thread_local StepScratch sc;
  return sc;
}

/// Adds f * grad_i s_m(i, j) to acc[i] and f * grad_j s_m(i, j) to acc[j] (the mirror force).
inline void add_pair_force(const ItemStore& st, int i, int j, double f, int m, DeltaAcc& acc, std::vector<double>& tmp) {
  if (f == 0.0) return;
  tmp.resize(static_cast<std::size_t>(m));
  cosine_grad(st.mean(i), st.mean(j), m, tmp.data());
  {
    std::vector<double>& gi = acc.at(i);
    for (int k = 0; k < m; ++k) gi[static_cast<std::size_t>(k)] += f * tmp[static_cast<std::size_t>(k)];
  }
  cosine_grad(st.mean(j), st.mean(i), m, tmp.data());
  std::vector<double>& gj = acc.at(j);
  for (int k = 0; k < m; ++k) gj[static_cast<std::size_t>(k)] += f * tmp[static_cast<std::size_t>(k)];
}

/// add_pair_force with |i|, |j| and cos(i, j) on the first m coordinates already known (S-1c). They are the values
/// cosine_grad would compute again (the same norm and dot functions on the same data; dot(j, i) = dot(i, j) and
/// |j||i| = |i||j| exactly), so the forces are bit-identical, without three dot products per pair.
inline void add_pair_force_known(const ItemStore& st, int i, int j, double f, int m, double ni, double nj, double c,
                                 DeltaAcc& acc) {
  if (f == 0.0) return;
  const float* a = st.mean(i);
  const float* b = st.mean(j);
  const bool ok = ni > 0.0 && nj > 0.0;
  {
    std::vector<double>& gi = acc.at(i);
    for (int k = 0; k < m; ++k) gi[static_cast<std::size_t>(k)] += f * (ok ? (b[k] / nj - c * a[k] / ni) / ni : 0.0);
  }
  std::vector<double>& gj = acc.at(j);
  for (int k = 0; k < m; ++k) gj[static_cast<std::size_t>(k)] += f * (ok ? (a[k] / ni - c * b[k] / nj) / nj : 0.0);
}

/// One InfoNCE term set (section 4.2) at prefix m: pushes the forces into acc, returns -log p_b.
inline double infonce_forces(const ItemStore& st, const PullPushExample& ex, const StepOptions& o,
                             const ParaphraseSet* para, int m, double weight, DeltaAcc& acc,
                             std::vector<double>& tmp, StepStats& stats, StepScratch& sc) {
  (void)tmp;
  std::vector<int>& ti = sc.ti;
  std::vector<int>& tj = sc.tj;
  std::vector<double>& s = sc.s;
  std::vector<double>& tni = sc.tni;
  std::vector<double>& tnj = sc.tnj;
  std::vector<char>& mask = sc.mask;
  ti.clear();
  tj.clear();
  s.clear();
  tni.clear();
  tnj.clear();
  mask.clear();
  // cosine(x, y) = dot / (|x||y|), or 0 when a norm is 0: computed once per term, with each norm computed once
  const double n_a = norm(st.mean(ex.a), m), n_b = norm(st.mean(ex.b), m);
  auto cos_known = [&](int x, double nx, int y, double ny) {
    if (nx <= 0.0 || ny <= 0.0) return 0.0;
    return dot(st.mean(x), st.mean(y), m) / (nx * ny);
  };
  const double s_ab = cos_known(ex.a, n_a, ex.b, n_b);
  ti.push_back(ex.a);
  tj.push_back(ex.b);
  s.push_back(s_ab);
  tni.push_back(n_a);
  tnj.push_back(n_b);
  mask.push_back(1);
  auto masked_out = [&](int other, double sim) {
    if (other == ex.a || other == ex.b) return true;  // the positive itself, or the anchor
    if (!o.use_mask) return false;
    if (sim > s_ab + o.delta) return true;            // assumed unlabelled paraphrase
    return para != nullptr && (para->contains(ex.b, other) || para->contains(ex.a, other));
  };
  auto add_side = [&](int from, double n_from, const std::vector<int>& others) {
    for (int c : others) {
      const double n_c = norm(st.mean(c), m);
      const double sim = cos_known(from, n_from, c, n_c);
      ti.push_back(from);
      tj.push_back(c);
      s.push_back(sim);
      tni.push_back(n_from);
      tnj.push_back(n_c);
      const bool out = masked_out(c, sim);
      mask.push_back(out ? 0 : 1);
      if (out) ++stats.masked;
    }
  };
  add_side(ex.a, n_a, ex.cands);
  if (o.uniformity) {
    add_side(ex.a, n_a, ex.qq);
    add_side(ex.b, n_b, ex.dd);
  }
  std::vector<double>& p = sc.p;
  masked_softmax_into(s, o.tau, mask, p);
  stats.p_pos = p[0];
  if (o.info_count) {
    // information share of this softmax, 1 - H(p) / ln k over the unmasked entries (StepOptions::info_count)
    std::size_t k = 0;
    double h = 0.0;
    for (std::size_t t = 0; t < p.size(); ++t)
      if (mask[t]) {
        ++k;
        if (p[t] > 0.0) h -= p[t] * std::log(p[t]);
      }
    stats.info = k < 2 ? 1.0 : std::max(0.0, 1.0 - h / std::log(static_cast<double>(k)));
  }
  add_pair_force_known(st, ex.a, ex.b, weight * (1.0 - p[0]) / o.tau, m, n_a, n_b, s_ab, acc);
  for (std::size_t t = 1; t < ti.size(); ++t)
    if (mask[t]) add_pair_force_known(st, ti[t], tj[t], -weight * p[t] / o.tau, m, tni[t], tnj[t], s[t], acc);
  return -std::log(std::max(p[0], 1e-300));
}

/// Applies the accumulated forces through the natural-gradient step, then MDL shrinkage.
/// step is a d-long work buffer (resized here); every entry is written before it is read.
inline void apply_natural_step(ItemStore& st, const DeltaAcc& acc, const StepOptions& o, std::vector<double>& step) {
  const int d = st.d;
  step.resize(static_cast<std::size_t>(d));
  for (std::size_t k = 0; k < acc.ids.size(); ++k) {
    const int i = acc.ids[k];
    float* m = st.mean(i);
    const float* v = st.variance(i);
    const double denom = static_cast<double>(st.n[static_cast<std::size_t>(i)]) + o.kappa0;
    for (int j = 0; j < d; ++j) step[static_cast<std::size_t>(j)] = o.eta * (v[j] / denom) * acc.g[k][static_cast<std::size_t>(j)];
    // the diagonal preconditioner is not isotropic, so the step is projected back onto the tangent
    const double mm = dot(m, m, d);
    if (mm > 0.0) {
      double sm = 0.0;
      for (int j = 0; j < d; ++j) sm += step[static_cast<std::size_t>(j)] * m[j];
      const double c = sm / mm;
      for (int j = 0; j < d; ++j) step[static_cast<std::size_t>(j)] -= c * m[j];
    }
    for (int j = 0; j < d; ++j) m[j] = static_cast<float>(m[j] + step[static_cast<std::size_t>(j)]);
  }
  for (int i : acc.ids) {
    float* m = st.mean(i);
    if (o.lambda_mdl > 0.0) {
      const float* th = st.prior(i);
      const double lam = std::min(1.0, o.lambda_mdl / (static_cast<double>(st.n[static_cast<std::size_t>(i)]) + 1.0));
      for (int j = 0; j < d; ++j) m[j] = static_cast<float>(m[j] + lam * (th[j] - m[j]));
    }
    if (o.renormalise) normalise(m, d);
  }
}

inline void apply_natural_step(ItemStore& st, const DeltaAcc& acc, const StepOptions& o) {
  std::vector<double> step;
  apply_natural_step(st, acc, o, step);
}

/// Online diagonal variance about the pre-update mean, Sigma <- Sigma + rate (r^2 - Sigma),
/// rate = max(floor, w / (n + kappa0)) after n += w: the Bayesian running mean with Sigma0 at weight kappa0.
inline void observe(ItemStore& st, int i, const std::vector<double>& r2, double w, const StepOptions& o) {
  st.n[static_cast<std::size_t>(i)] += static_cast<float>(w);
  if (!o.update_variance) return;
  const double rate = std::max(o.var_rate_floor, w / (static_cast<double>(st.n[static_cast<std::size_t>(i)]) + o.kappa0));
  float* v = st.variance(i);
  for (int j = 0; j < st.d; ++j)
    v[j] = std::max(o.var_floor, static_cast<float>(v[j] + rate * (r2[static_cast<std::size_t>(j)] - v[j])));
}

inline void residual2_into(const ItemStore& st, int i, int target, std::vector<double>& r) {
  r.resize(static_cast<std::size_t>(st.d));
  const float* a = st.mean(i);
  const float* b = st.mean(target);
  const double nb = norm(b, st.d);
  for (int j = 0; j < st.d; ++j) {
    const double e = (nb > 0.0 ? b[j] / nb : 0.0) - a[j];
    r[static_cast<std::size_t>(j)] = e * e;
  }
}

inline std::vector<double> residual2(const ItemStore& st, int i, int target) {
  std::vector<double> r;
  residual2_into(st, i, target, r);
  return r;
}

}  // namespace detail

// ---------------------------------------------------------------------------------------------
// Section 4.2: the Cypha pull-push rule
// ---------------------------------------------------------------------------------------------

/// One update on (anchor a, positive b, candidates C). The loss is averaged over the MRL prefixes.
/// Evidence: a and b each gain one observation (a observes b's direction, b observes a's).
inline StepStats pull_push_step(ItemStore& st, const PullPushExample& ex, const StepOptions& o,
                                const ParaphraseSet* para = nullptr) {
  StepStats stats;
  detail::StepScratch& sc = detail::step_scratch();
  detail::resolve_prefixes_into(o, st.d, sc.pref);
  const double w = 1.0 / static_cast<double>(sc.pref.size());
  detail::DeltaAcc& acc = sc.acc;
  acc.reset(st.d);
  for (int m : sc.pref) stats.loss += w * detail::infonce_forces(st, ex, o, para, m, w, acc, sc.tmp, stats, sc);
  detail::residual2_into(st, ex.a, ex.b, sc.ra);
  detail::residual2_into(st, ex.b, ex.a, sc.rb);
  detail::apply_natural_step(st, acc, o, sc.step);
  const double ev = o.info_count ? stats.info : 1.0;
  detail::observe(st, ex.a, sc.ra, ev, o);
  detail::observe(st, ex.b, sc.rb, ev, o);
  return stats;
}

// ---------------------------------------------------------------------------------------------
// Section 5.1: relational softmax distillation, and alpha * KL + (1 - alpha) * InfoNCE
// ---------------------------------------------------------------------------------------------

inline double kl_divergence(const std::vector<double>& p, const std::vector<double>& q) {
  double s = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i)
    if (p[i] > 0.0) s += p[i] * (std::log(p[i]) - std::log(std::max(q[i], 1e-300)));
  return s;
}

/// Moves s_S(a, c) by (p_T(c) - p_S(c)) / tau_S through the natural-gradient step.
/// The anchor gains one observation; its residual is the p_T-weighted spread of the candidates about it.
inline StepStats distill_step(ItemStore& st, const DistillExample& ex, const StepOptions& o,
                              const ParaphraseSet* para = nullptr) {
  if (ex.teacher_sims.size() != ex.cands.size()) throw std::invalid_argument("distill_step: one teacher similarity per candidate");
  StepStats stats;
  detail::StepScratch& sc = detail::step_scratch();
  detail::resolve_prefixes_into(o, st.d, sc.pref);
  const double w = 1.0 / static_cast<double>(sc.pref.size());
  const bool nce = o.alpha < 1.0 && ex.positive >= 0;
  const double alpha = nce ? o.alpha : 1.0;
  std::vector<double>& sT = sc.sT;
  sT.assign(ex.teacher_sims.begin(), ex.teacher_sims.end());
  std::vector<double>& pT = sc.pT;
  masked_softmax_into(sT, o.tau_T, {}, pT);
  detail::DeltaAcc& acc = sc.acc;
  acc.reset(st.d);
  std::vector<double>& sS = sc.sS;
  std::vector<double>& pS = sc.pS;
  for (int m : sc.pref) {
    sS.resize(ex.cands.size());
    for (std::size_t c = 0; c < ex.cands.size(); ++c) sS[c] = cosine(st.mean(ex.a), st.mean(ex.cands[c]), m);
    masked_softmax_into(sS, o.tau_S, {}, pS);
    stats.loss += w * alpha * kl_divergence(pT, pS);
    for (std::size_t c = 0; c < ex.cands.size(); ++c)
      if (ex.cands[c] != ex.a)
        detail::add_pair_force(st, ex.a, ex.cands[c], w * alpha * (pT[c] - pS[c]) / o.tau_S, m, acc, sc.tmp);
    if (nce) {
      PullPushExample pp;
      pp.a = ex.a;
      pp.b = ex.positive;
      for (int c : ex.cands)
        if (c != ex.positive) pp.cands.push_back(c);
      stats.loss += w * (1.0 - alpha) * detail::infonce_forces(st, pp, o, para, m, w * (1.0 - alpha), acc, sc.tmp, stats, sc);
    }
  }
  std::vector<double>& r = sc.r;
  r.assign(static_cast<std::size_t>(st.d), 0.0);
  for (std::size_t c = 0; c < ex.cands.size(); ++c) {
    if (pT[c] <= 0.0 || ex.cands[c] == ex.a) continue;
    detail::residual2_into(st, ex.a, ex.cands[c], sc.rc);
    for (int j = 0; j < st.d; ++j) r[static_cast<std::size_t>(j)] += pT[c] * sc.rc[static_cast<std::size_t>(j)];
  }
  detail::apply_natural_step(st, acc, o, sc.step);
  detail::observe(st, ex.a, r, 1.0, o);
  return stats;
}

/// Applies an externally computed force (one d-vector per item) through the natural-gradient step.
/// Used for tests and for callers that build their own losses.
inline void natural_gradient_step(ItemStore& st, int item, const std::vector<double>& grad, const StepOptions& o) {
  detail::DeltaAcc acc;
  acc.d = st.d;
  std::vector<double>& g = acc.at(item);
  for (int j = 0; j < st.d && j < static_cast<int>(grad.size()); ++j) g[static_cast<std::size_t>(j)] = grad[static_cast<std::size_t>(j)];
  detail::apply_natural_step(st, acc, o);
}

/// MDL shrinkage alone: mu <- mu + lambda / (n + 1) * (theta0 - mu), then retraction.
inline void mdl_shrink(ItemStore& st, int i, double lambda, bool renormalise = true) {
  float* m = st.mean(i);
  const float* th = st.prior(i);
  const double lam = std::min(1.0, lambda / (static_cast<double>(st.n[static_cast<std::size_t>(i)]) + 1.0));
  for (int j = 0; j < st.d; ++j) m[j] = static_cast<float>(m[j] + lam * (th[j] - m[j]));
  if (renormalise) detail::normalise(m, st.d);
}

// ---------------------------------------------------------------------------------------------
// Candidate sources (section 4.2): unigram^0.75 sampling and the current nearest neighbours
// ---------------------------------------------------------------------------------------------

class UnigramSampler {
 public:
  UnigramSampler(const std::vector<double>& counts, double power = 0.75) {
    cdf_.resize(counts.size());
    double s = 0.0;
    for (std::size_t i = 0; i < counts.size(); ++i) cdf_[i] = (s += std::pow(std::max(0.0, counts[i]), power));
    if (s <= 0.0) throw std::invalid_argument("UnigramSampler: all counts are zero");
    for (double& c : cdf_) c /= s;
  }
  int sample(std::mt19937_64& eng) const {
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(eng);
    const auto it = std::upper_bound(cdf_.begin(), cdf_.end(), u);
    return static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(it - cdf_.begin()), cdf_.size() - 1));
  }

 private:
  std::vector<double> cdf_;
};

// ---------------------------------------------------------------------------------------------
// Linear algebra: cyclic Jacobi for small symmetric matrices, PCA basis, rotation
// ---------------------------------------------------------------------------------------------

/// Symmetric n x n (row-major) -> eigenvalues in descending order and eigenvectors as the columns of V
/// (V[i * n + k] = component i of eigenvector k).
inline void jacobi_eigen(std::vector<double> A, int n, std::vector<double>& evals, std::vector<double>& V,
                         int max_sweeps = 100, double tol = 1e-13) {
  const std::size_t N = static_cast<std::size_t>(n);
  V.assign(N * N, 0.0);
  for (std::size_t i = 0; i < N; ++i) V[i * N + i] = 1.0;
  double frob = 0.0;
  for (double x : A) frob += x * x;
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    double off = 0.0;
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = i + 1; j < N; ++j) off += A[i * N + j] * A[i * N + j];
    if (off <= tol * tol * frob || off == 0.0) break;
    for (std::size_t p = 0; p < N; ++p) {
      for (std::size_t q = p + 1; q < N; ++q) {
        const double apq = A[p * N + q];
        if (apq == 0.0) continue;
        const double theta = (A[q * N + q] - A[p * N + p]) / (2.0 * apq);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (std::size_t k = 0; k < N; ++k) {  // columns p, q
          const double akp = A[k * N + p], akq = A[k * N + q];
          A[k * N + p] = c * akp - s * akq;
          A[k * N + q] = s * akp + c * akq;
        }
        for (std::size_t k = 0; k < N; ++k) {  // rows p, q
          const double apk = A[p * N + k], aqk = A[q * N + k];
          A[p * N + k] = c * apk - s * aqk;
          A[q * N + k] = s * apk + c * aqk;
        }
        for (std::size_t k = 0; k < N; ++k) {
          const double vkp = V[k * N + p], vkq = V[k * N + q];
          V[k * N + p] = c * vkp - s * vkq;
          V[k * N + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  std::vector<std::size_t> order(N);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return A[x * N + x] > A[y * N + y]; });
  evals.resize(N);
  std::vector<double> Vs(N * N);
  for (std::size_t k = 0; k < N; ++k) {
    evals[k] = A[order[k] * N + order[k]];
    for (std::size_t i = 0; i < N; ++i) Vs[i * N + k] = V[i * N + order[k]];
  }
  V.swap(Vs);
}

/// Column means of X (N x d).
inline std::vector<double> column_mean(const float* X, int N, int d) {
  std::vector<double> m(static_cast<std::size_t>(d), 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < d; ++j) m[static_cast<std::size_t>(j)] += X[static_cast<std::size_t>(i) * d + j];
  for (double& v : m) v /= std::max(1, N);
  return m;
}

/// Covariance (centred, divided by N) of X (N x d), row-major d x d.
inline std::vector<double> covariance(const float* X, int N, int d) {
  const std::vector<double> m = column_mean(X, N, d);
  const std::size_t D = static_cast<std::size_t>(d);
  std::vector<double> C(D * D, 0.0), row(D);
  for (int i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < D; ++j) row[j] = X[static_cast<std::size_t>(i) * D + j] - m[j];
    for (std::size_t a = 0; a < D; ++a) {
      const double ra = row[a];
      double* Ca = C.data() + a * D;
      for (std::size_t b = a; b < D; ++b) Ca[b] += ra * row[b];
    }
  }
  for (std::size_t a = 0; a < D; ++a)
    for (std::size_t b = a; b < D; ++b) {
      C[a * D + b] /= std::max(1, N);
      C[b * D + a] = C[a * D + b];
    }
  return C;
}

/// X <- X V for an orthogonal d x d V (row-major, eigenvectors as columns).
inline void rotate_rows(float* X, int N, int d, const std::vector<double>& V) {
  const std::size_t D = static_cast<std::size_t>(d);
  std::vector<double> out(D);
  for (int i = 0; i < N; ++i) {
    float* x = X + static_cast<std::size_t>(i) * D;
    std::fill(out.begin(), out.end(), 0.0);
    for (std::size_t a = 0; a < D; ++a) {
      const double xa = x[a];
      const double* Va = V.data() + a * D;
      for (std::size_t k = 0; k < D; ++k) out[k] += xa * Va[k];
    }
    for (std::size_t k = 0; k < D; ++k) x[k] = static_cast<float>(out[k]);
  }
}

/// MRL ordering (section 4.5): rotates every mean (and theta0) into the PCA basis of the means, so that
/// dimension 0 carries the most variance. Cosines are unchanged (orthogonal rotation). The diagonal Sigma
/// is carried as diag(V^T Sigma V). Returns the eigenvalues, descending; V receives the rotation.
inline std::vector<double> mrl_order(ItemStore& st, std::vector<double>* V_out = nullptr) {
  std::vector<double> evals, V;
  jacobi_eigen(covariance(st.mu.data(), st.n_items, st.d), st.d, evals, V);
  rotate_rows(st.mu.data(), st.n_items, st.d, V);
  rotate_rows(st.world_prior.data(), 1, st.d, V);
  if (!st.item_prior.empty()) rotate_rows(st.item_prior.data(), st.n_items, st.d, V);
  const std::size_t D = static_cast<std::size_t>(st.d);
  std::vector<float> nv(D);
  for (int i = 0; i < st.n_items; ++i) {
    float* v = st.variance(i);
    for (std::size_t k = 0; k < D; ++k) {
      double s = 0.0;
      for (std::size_t a = 0; a < D; ++a) s += V[a * D + k] * V[a * D + k] * v[a];
      nv[k] = static_cast<float>(s);
    }
    std::copy(nv.begin(), nv.end(), v);
  }
  if (V_out) *V_out = V;
  return evals;
}

// ---------------------------------------------------------------------------------------------
// Post-processing: centring, PCA / ZCA whitening, CSLS
// ---------------------------------------------------------------------------------------------

/// Subtracts the column mean; returns it.
inline std::vector<double> centre(float* X, int N, int d) {
  const std::vector<double> m = column_mean(X, N, d);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < d; ++j) X[static_cast<std::size_t>(i) * d + j] = static_cast<float>(X[static_cast<std::size_t>(i) * d + j] - m[static_cast<std::size_t>(j)]);
  return m;
}

/// Normalises each row to unit length.
inline void normalise_rows(float* X, int N, int d) {
  for (int i = 0; i < N; ++i) detail::normalise(X + static_cast<std::size_t>(i) * d, d);
}

enum class WhitenMode { PCA, ZCA };

struct Whitener {
  int d_in = 0;
  int d_out = 0;
  std::vector<double> mean;  // d_in
  std::vector<double> W;     // d_out x d_in, row-major: y = W (x - mean)

  std::vector<float> apply(const float* X, int N) const {
    std::vector<float> Y(static_cast<std::size_t>(N) * static_cast<std::size_t>(d_out));
    std::vector<double> xc(static_cast<std::size_t>(d_in));
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < d_in; ++j) xc[static_cast<std::size_t>(j)] = X[static_cast<std::size_t>(i) * d_in + j] - mean[static_cast<std::size_t>(j)];
      for (int k = 0; k < d_out; ++k) {
        double s = 0.0;
        const double* Wk = W.data() + static_cast<std::size_t>(k) * d_in;
        for (int j = 0; j < d_in; ++j) s += Wk[j] * xc[static_cast<std::size_t>(j)];
        Y[static_cast<std::size_t>(i) * d_out + k] = static_cast<float>(s);
      }
    }
    return Y;
  }
};

/// PCA whitening y = Lambda^{-1/2} U^T (x - m), optionally truncated to d_out leading components;
/// ZCA y = U Lambda^{-1/2} U^T (x - m) (full rank). eps regularises small eigenvalues.
inline Whitener fit_whitener(const float* X, int N, int d, WhitenMode mode, double eps = 1e-6, int d_out = -1) {
  Whitener w;
  w.d_in = d;
  w.mean = column_mean(X, N, d);
  std::vector<double> evals, U;
  jacobi_eigen(covariance(X, N, d), d, evals, U);
  const std::size_t D = static_cast<std::size_t>(d);
  if (mode == WhitenMode::PCA) {
    w.d_out = (d_out > 0 && d_out < d) ? d_out : d;
    w.W.assign(static_cast<std::size_t>(w.d_out) * D, 0.0);
    for (int k = 0; k < w.d_out; ++k) {
      const double sc = 1.0 / std::sqrt(std::max(evals[static_cast<std::size_t>(k)], 0.0) + eps);
      for (std::size_t j = 0; j < D; ++j) w.W[static_cast<std::size_t>(k) * D + j] = sc * U[j * D + static_cast<std::size_t>(k)];
    }
  } else {
    w.d_out = d;
    w.W.assign(D * D, 0.0);
    for (std::size_t k = 0; k < D; ++k) {
      const double sc = 1.0 / std::sqrt(std::max(evals[k], 0.0) + eps);
      for (std::size_t a = 0; a < D; ++a) {
        const double ua = U[a * D + k] * sc;
        for (std::size_t b = 0; b < D; ++b) w.W[a * D + b] += ua * U[b * D + k];
      }
    }
  }
  return w;
}

/// Full N x N cosine matrix on the first m coordinates of X (row stride d).
inline std::vector<double> cosine_matrix(const float* X, int N, int d, int m = -1) {
  if (m <= 0 || m > d) m = d;
  std::vector<double> inv(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    const double nn = detail::norm(X + static_cast<std::size_t>(i) * d, m);
    inv[static_cast<std::size_t>(i)] = nn > 0.0 ? 1.0 / nn : 0.0;
  }
  std::vector<double> S(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 0.0);
  for (int i = 0; i < N; ++i) {
    S[static_cast<std::size_t>(i) * N + i] = 1.0;
    for (int j = i + 1; j < N; ++j) {
      const double c = detail::dot(X + static_cast<std::size_t>(i) * d, X + static_cast<std::size_t>(j) * d, m) *
                       inv[static_cast<std::size_t>(i)] * inv[static_cast<std::size_t>(j)];
      S[static_cast<std::size_t>(i) * N + j] = S[static_cast<std::size_t>(j) * N + i] = c;
    }
  }
  return S;
}

using KnnLists = std::vector<std::vector<int>>;

/// Top-k (self excluded) of each row of a similarity matrix; ties go to the lower index.
inline KnnLists knn_from_matrix(const std::vector<double>& S, int N, int k) {
  KnnLists out(static_cast<std::size_t>(N));
  k = std::min(k, N - 1);
  std::vector<int> idx;
  for (int i = 0; i < N; ++i) {
    idx.clear();
    for (int j = 0; j < N; ++j)
      if (j != i) idx.push_back(j);
    const double* row = S.data() + static_cast<std::size_t>(i) * N;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int x, int y) {
      return row[x] > row[y] || (row[x] == row[y] && x < y);
    });
    out[static_cast<std::size_t>(i)].assign(idx.begin(), idx.begin() + k);
  }
  return out;
}

/// Cosine kNN (self excluded) on the first m coordinates.
inline KnnLists knn_cosine(const float* X, int N, int d, int k, int m = -1) {
  return knn_from_matrix(cosine_matrix(X, N, d, m), N, k);
}

/// CSLS radius r(x): the mean cosine of x to its k nearest neighbours (self excluded).
inline std::vector<double> csls_radius(const std::vector<double>& S, int N, int k) {
  const KnnLists nn = knn_from_matrix(S, N, k);
  std::vector<double> r(static_cast<std::size_t>(N), 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j : nn[static_cast<std::size_t>(i)]) r[static_cast<std::size_t>(i)] += S[static_cast<std::size_t>(i) * N + j];
    if (!nn[static_cast<std::size_t>(i)].empty()) r[static_cast<std::size_t>(i)] /= static_cast<double>(nn[static_cast<std::size_t>(i)].size());
  }
  return r;
}

/// CSLS(x, y) = 2 cos(x, y) - r(x) - r(y) (Conneau et al. 2018), within one set.
inline std::vector<double> csls_matrix(const std::vector<double>& S, int N, int k_csls) {
  const std::vector<double> r = csls_radius(S, N, k_csls);
  std::vector<double> C(S.size());
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      C[static_cast<std::size_t>(i) * N + j] = 2.0 * S[static_cast<std::size_t>(i) * N + j] - r[static_cast<std::size_t>(i)] - r[static_cast<std::size_t>(j)];
  return C;
}

inline KnnLists knn_csls(const float* X, int N, int d, int k, int k_csls = 10, int m = -1) {
  return knn_from_matrix(csls_matrix(cosine_matrix(X, N, d, m), N, k_csls), N, k);
}

// ---------------------------------------------------------------------------------------------
// Metrics (section 7.1)
// ---------------------------------------------------------------------------------------------

/// Average ranks (1-based); tied values share the mean of their ranks.
inline std::vector<double> average_ranks(const std::vector<double>& x) {
  const std::size_t n = x.size();
  std::vector<std::size_t> o(n);
  std::iota(o.begin(), o.end(), std::size_t{0});
  std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n;) {
    std::size_t j = i;
    while (j + 1 < n && x[o[j + 1]] == x[o[i]]) ++j;
    const double avg = 0.5 * static_cast<double>(i + j) + 1.0;
    for (std::size_t k = i; k <= j; ++k) r[o[k]] = avg;
    i = j + 1;
  }
  return r;
}

inline double pearson(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  if (n < 2) return 0.0;
  double ma = 0.0, mb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ma += a[i];
    mb += b[i];
  }
  ma /= static_cast<double>(n);
  mb /= static_cast<double>(n);
  double sab = 0.0, saa = 0.0, sbb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) * (a[i] - ma);
    sbb += (b[i] - mb) * (b[i] - mb);
  }
  return (saa > 0.0 && sbb > 0.0) ? sab / std::sqrt(saa * sbb) : 0.0;
}

/// Spearman rho: Pearson of average ranks (ties handled).
inline double spearman(const std::vector<double>& a, const std::vector<double>& b) {
  return pearson(average_ranks(a), average_ranks(b));
}

struct PairSample {
  std::vector<std::pair<int, int>> pairs;
  std::vector<int> stratum;  // 0 near, 1 middle, 2 random
};

/// Pairs stratified by the teacher (section 7.1): per item, n_near from its teacher top k_near,
/// n_mid from teacher ranks k_near..k_mid, n_rand uniform. Pairs are unordered and deduplicated.
inline PairSample stratified_pairs(const float* T, int N, int dT, int n_near, int n_mid, int n_rand,
                                   std::uint64_t seed, int k_near = 10, int k_mid = 100) {
  PairSample out;
  const std::vector<double> S = cosine_matrix(T, N, dT);
  k_near = std::min(k_near, N - 1);
  k_mid = std::min(std::max(k_mid, k_near), N - 1);
  const KnnLists nn = knn_from_matrix(S, N, k_mid);
  std::mt19937_64 eng(seed);
  std::set<std::uint64_t> seen;
  auto push = [&](int i, int j, int s) {
    if (i == j) return;
    const auto a = static_cast<std::uint32_t>(std::min(i, j)), b = static_cast<std::uint32_t>(std::max(i, j));
    if (!seen.insert((static_cast<std::uint64_t>(a) << 32) | b).second) return;
    out.pairs.emplace_back(static_cast<int>(a), static_cast<int>(b));
    out.stratum.push_back(s);
  };
  for (int i = 0; i < N; ++i) {
    const std::vector<int>& L = nn[static_cast<std::size_t>(i)];
    for (int t = 0; t < n_near && k_near > 0; ++t)
      push(i, L[static_cast<std::size_t>(std::uniform_int_distribution<int>(0, k_near - 1)(eng))], 0);
    for (int t = 0; t < n_mid && k_mid > k_near; ++t)
      push(i, L[static_cast<std::size_t>(std::uniform_int_distribution<int>(k_near, k_mid - 1)(eng))], 1);
    for (int t = 0; t < n_rand; ++t) push(i, std::uniform_int_distribution<int>(0, N - 1)(eng), 2);
  }
  return out;
}

/// Pair Spearman of teacher and student cosines over the pairs (student on its first mS coordinates).
inline double pair_spearman(const float* T, int dT, const float* S, int dS, const std::vector<std::pair<int, int>>& pairs,
                            int mS = -1) {
  if (mS <= 0 || mS > dS) mS = dS;
  std::vector<double> a, b;
  a.reserve(pairs.size());
  b.reserve(pairs.size());
  for (const auto& [i, j] : pairs) {
    a.push_back(detail::cosine(T + static_cast<std::size_t>(i) * dT, T + static_cast<std::size_t>(j) * dT, dT));
    b.push_back(detail::cosine(S + static_cast<std::size_t>(i) * dS, S + static_cast<std::size_t>(j) * dS, mS));
  }
  return spearman(a, b);
}

/// Mean Jaccard overlap of two neighbour-list sets.
inline double knn_jaccard(const KnnLists& A, const KnnLists& B) {
  double s = 0.0;
  const std::size_t n = std::min(A.size(), B.size());
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<int> a = A[i], b = B[i];
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::vector<int> in;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(in));
    const double u = static_cast<double>(a.size() + b.size() - in.size());
    s += u > 0.0 ? static_cast<double>(in.size()) / u : 1.0;
  }
  return n ? s / static_cast<double>(n) : 0.0;
}

/// Mean fraction of each teacher top-kT list found in the student's top-kS list (kT = 10, kS = 50 in 7.1).
inline double recall_in_topk(const KnnLists& teacher_k, const KnnLists& student_k) {
  double s = 0.0;
  const std::size_t n = std::min(teacher_k.size(), student_k.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (teacher_k[i].empty()) continue;
    std::set<int> st(student_k[i].begin(), student_k[i].end());
    int hit = 0;
    for (int j : teacher_k[i]) hit += st.count(j) ? 1 : 0;
    s += static_cast<double>(hit) / static_cast<double>(teacher_k[i].size());
  }
  return n ? s / static_cast<double>(n) : 0.0;
}

/// Linear CKA (Kornblith et al. 2019): ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F), columns centred.
inline double linear_cka(const float* X, int dx, const float* Y, int dy, int N) {
  const std::vector<double> mx = column_mean(X, N, dx), my = column_mean(Y, N, dy);
  const std::size_t DX = static_cast<std::size_t>(dx), DY = static_cast<std::size_t>(dy);
  std::vector<double> XX(DX * DX, 0.0), YY(DY * DY, 0.0), YX(DY * DX, 0.0), xr(DX), yr(DY);
  for (int i = 0; i < N; ++i) {
    for (std::size_t a = 0; a < DX; ++a) xr[a] = X[static_cast<std::size_t>(i) * DX + a] - mx[a];
    for (std::size_t a = 0; a < DY; ++a) yr[a] = Y[static_cast<std::size_t>(i) * DY + a] - my[a];
    for (std::size_t a = 0; a < DX; ++a)
      for (std::size_t b = 0; b < DX; ++b) XX[a * DX + b] += xr[a] * xr[b];
    for (std::size_t a = 0; a < DY; ++a)
      for (std::size_t b = 0; b < DY; ++b) YY[a * DY + b] += yr[a] * yr[b];
    for (std::size_t a = 0; a < DY; ++a)
      for (std::size_t b = 0; b < DX; ++b) YX[a * DX + b] += yr[a] * xr[b];
  }
  auto f2 = [](const std::vector<double>& M) {
    double s = 0.0;
    for (double v : M) s += v * v;
    return s;
  };
  const double den = std::sqrt(f2(XX)) * std::sqrt(f2(YY));
  return den > 0.0 ? f2(YX) / den : 0.0;
}

/// k-occurrence N_k(x): how often x appears in the other items' kNN lists.
inline std::vector<double> k_occurrence(const KnnLists& knn, int N) {
  std::vector<double> c(static_cast<std::size_t>(N), 0.0);
  for (const auto& L : knn)
    for (int j : L) c[static_cast<std::size_t>(j)] += 1.0;
  return c;
}

/// Skewness of the k-occurrence distribution (hubness; Radovanovic et al. 2010).
inline double k_occurrence_skewness(const KnnLists& knn, int N) {
  const std::vector<double> c = k_occurrence(knn, N);
  double m = 0.0;
  for (double v : c) m += v;
  m /= std::max(1, N);
  double m2 = 0.0, m3 = 0.0;
  for (double v : c) {
    m2 += (v - m) * (v - m);
    m3 += (v - m) * (v - m) * (v - m);
  }
  m2 /= std::max(1, N);
  m3 /= std::max(1, N);
  return m2 > 0.0 ? m3 / std::pow(m2, 1.5) : 0.0;
}

/// Force match: mean over anchors of KL(p_T || p_S), both at the teacher's tau, over each anchor's candidates.
inline double force_match_kl(const float* T, int dT, const float* S, int dS, const std::vector<int>& anchors,
                             const std::vector<std::vector<int>>& cands, double tau) {
  double s = 0.0;
  std::size_t used = 0;
  for (std::size_t k = 0; k < anchors.size() && k < cands.size(); ++k) {
    const int a = anchors[k];
    std::vector<double> sT, sS;
    for (int c : cands[k]) {
      sT.push_back(detail::cosine(T + static_cast<std::size_t>(a) * dT, T + static_cast<std::size_t>(c) * dT, dT));
      sS.push_back(detail::cosine(S + static_cast<std::size_t>(a) * dS, S + static_cast<std::size_t>(c) * dS, dS));
    }
    if (sT.empty()) continue;
    s += kl_divergence(masked_softmax(sT, tau), masked_softmax(sS, tau));
    ++used;
  }
  return used ? s / static_cast<double>(used) : 0.0;
}

/// Per-item disagreement with the teacher: 1 - Spearman of the item's teacher and student similarity rows
/// (to every other item when N <= max_partners, else to max_partners seeded random partners).
inline std::vector<double> item_disagreement(const float* T, int dT, const float* S, int dS, int N,
                                             int max_partners = 1000, std::uint64_t seed = 1) {
  std::vector<double> out(static_cast<std::size_t>(N), 0.0);
  std::mt19937_64 eng(seed);
  std::vector<int> partners;
  for (int i = 0; i < N; ++i) {
    partners.clear();
    if (N - 1 <= max_partners) {
      for (int j = 0; j < N; ++j)
        if (j != i) partners.push_back(j);
    } else {
      for (int t = 0; t < max_partners; ++t) {
        int j = std::uniform_int_distribution<int>(0, N - 2)(eng);
        partners.push_back(j >= i ? j + 1 : j);
      }
    }
    std::vector<double> a, b;
    for (int j : partners) {
      a.push_back(detail::cosine(T + static_cast<std::size_t>(i) * dT, T + static_cast<std::size_t>(j) * dT, dT));
      b.push_back(detail::cosine(S + static_cast<std::size_t>(i) * dS, S + static_cast<std::size_t>(j) * dS, dS));
    }
    out[static_cast<std::size_t>(i)] = 1.0 - spearman(a, b);
  }
  return out;
}

/// The section 7.1 confidence check: Spearman of the per-item mean variance against per-item disagreement.
inline double confidence_check(const ItemStore& st, const float* T, int dT, int max_partners = 1000, std::uint64_t seed = 1) {
  std::vector<double> v(static_cast<std::size_t>(st.n_items));
  for (int i = 0; i < st.n_items; ++i) v[static_cast<std::size_t>(i)] = st.mean_variance(i);
  return spearman(v, item_disagreement(T, dT, st.mu.data(), st.d, st.n_items, max_partners, seed));
}

}  // namespace cypha::semantics
