// Tests for cypha/semantics/geometry.hpp (docs/EMBEDDING_GEOMETRY.md sections 4.2, 4.5, 5.1, 7.1).
// Synthetic data only; no corpus. Exits nonzero if any check fails.
//
//   (a) known answers: Spearman with ties, CKA invariances, tangent gradient, softmax weights, fp16, whitening
//   (b) planted teacher (400 items, 64 d, 20 clusters, paraphrases, hard negatives), 32-d student from random:
//       InfoNCE with the 4.2 rule, relational distillation, distillation beats InfoNCE-only
//   (c) the false-negative mask keeps planted paraphrases together
//   (d) confidence-scaled steps and MDL shrinkage
//   (e) CSLS lowers hubness
//   (f) MRL ordering: prefix agreement rises with prefix length; the nested loss helps the short prefix
//   (g) confidence check: noisier training gives higher variance; variance-disagreement Spearman > 0
//   (h) determinism: same seed, bit-identical state hash
//
// Author: Odin Loch
#include "cypha/semantics/geometry.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cypha::semantics;

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond)                                                \
  do {                                                             \
    ++g_checks;                                                    \
    if (!(cond)) {                                                 \
      ++g_failed;                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);  \
    }                                                              \
  } while (0)

#define CHECK_NEAR(x, y, tol)                                                                          \
  do {                                                                                                 \
    ++g_checks;                                                                                        \
    const double _x = (x), _y = (y);                                                                   \
    if (!(std::fabs(_x - _y) <= (tol))) {                                                              \
      ++g_failed;                                                                                      \
      std::printf("FAIL %s:%d  %s = %.9g, expected %.9g (tol %g)\n", __FILE__, __LINE__, #x, _x, _y, \
                  (double)(tol));                                                                      \
    }                                                                                                  \
  } while (0)

namespace {

double gauss(std::mt19937_64& e) { return std::normal_distribution<double>(0.0, 1.0)(e); }
int uni(std::mt19937_64& e, int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(e); }

void normalise(std::vector<float>& v) {
  double s = 0;
  for (float x : v) s += (double)x * x;
  s = std::sqrt(s);
  for (float& x : v) x = (float)(x / s);
}

// ---------------------------------------------------------------------------------------------
// The planted teacher
// ---------------------------------------------------------------------------------------------

struct Teacher {
  int N = 0, D = 0, K = 0;
  std::vector<float> X;                      // N x D unit rows
  std::vector<int> cluster;                  // per item
  std::vector<std::pair<int, int>> para;     // planted paraphrase pairs
  std::vector<std::pair<int, int>> hardneg;  // planted hard negatives: same cluster, far apart
  std::vector<double> S;                     // cosine matrix
  KnnLists nn;                               // teacher neighbours by rank (all N - 1)
  const float* row(int i) const { return X.data() + (size_t)i * D; }
};

Teacher make_teacher(int K, int per, int D, int sub, double r, std::uint64_t seed) {
  Teacher t;
  t.N = K * per;
  t.D = D;
  t.K = K;
  t.X.assign((size_t)t.N * D, 0.0f);
  t.cluster.assign(t.N, 0);
  std::mt19937_64 e(seed);
  for (int k = 0; k < K; ++k) {
    std::vector<float> c(D);
    for (float& v : c) v = (float)gauss(e);
    normalise(c);
    // random orthonormal basis of a sub-dimensional local subspace
    std::vector<std::vector<double>> B(sub, std::vector<double>(D));
    for (int s = 0; s < sub; ++s) {
      for (double& v : B[s]) v = gauss(e);
      for (int q = 0; q < s; ++q) {
        double dp = 0;
        for (int j = 0; j < D; ++j) dp += B[s][j] * B[q][j];
        for (int j = 0; j < D; ++j) B[s][j] -= dp * B[q][j];
      }
      double nn = 0;
      for (double v : B[s]) nn += v * v;
      nn = std::sqrt(nn);
      for (double& v : B[s]) v /= nn;
    }
    std::vector<std::vector<double>> z(per, std::vector<double>(sub));
    for (int i = 0; i < per; ++i)
      for (double& v : z[i]) v = gauss(e) / std::sqrt((double)sub);
    // plants: item 1 paraphrases item 0, item 3 paraphrases item 2; item 5 is item 4 reflected (hard negative)
    for (int s = 0; s < sub; ++s) {
      z[1][s] = z[0][s] + 0.05 * gauss(e) / std::sqrt((double)sub);
      z[3][s] = z[2][s] + 0.05 * gauss(e) / std::sqrt((double)sub);
      z[5][s] = -z[4][s];
    }
    for (int i = 0; i < per; ++i) {
      const int id = k * per + i;
      t.cluster[id] = k;
      std::vector<float> x(D);
      for (int j = 0; j < D; ++j) {
        double v = c[j] + 0.03 * gauss(e);
        for (int s = 0; s < sub; ++s) v += r * z[i][s] * B[s][j];
        x[j] = (float)v;
      }
      normalise(x);
      std::copy(x.begin(), x.end(), t.X.begin() + (size_t)id * D);
    }
    t.para.push_back({k * per + 0, k * per + 1});
    t.para.push_back({k * per + 2, k * per + 3});
    t.hardneg.push_back({k * per + 4, k * per + 5});
  }
  t.S = cosine_matrix(t.X.data(), t.N, D);
  t.nn = knn_from_matrix(t.S, t.N, t.N - 1);
  return t;
}

struct Eval {
  double spearman = 0, jacc10 = 0, recall = 0, cka = 0, fkl = 0;
};

Eval evaluate(const Teacher& t, const ItemStore& s, const PairSample& ps, int m = -1) {
  Eval ev;
  ev.spearman = pair_spearman(t.X.data(), t.D, s.mu.data(), s.d, ps.pairs, m);
  const KnnLists tk10 = knn_from_matrix(t.S, t.N, 10);
  const KnnLists sk10 = knn_cosine(s.mu.data(), s.n_items, s.d, 10, m);
  const KnnLists sk50 = knn_cosine(s.mu.data(), s.n_items, s.d, 50, m);
  ev.jacc10 = knn_jaccard(tk10, sk10);
  ev.recall = recall_in_topk(tk10, sk50);
  ev.cka = linear_cka(t.X.data(), t.D, s.mu.data(), s.d, t.N);
  std::vector<int> anchors(t.N);
  std::vector<std::vector<int>> cands(t.N);
  for (int i = 0; i < t.N; ++i) {
    anchors[i] = i;
    cands[i].assign(t.nn[i].begin(), t.nn[i].begin() + 50);
  }
  ev.fkl = force_match_kl(t.X.data(), t.D, s.mu.data(), s.d, anchors, cands, 0.05);
  return ev;
}

void print_eval(const char* name, const Eval& e) {
  std::printf("  %-28s pair Spearman %.4f | kNN@10 Jaccard %.4f | top10-in-top50 %.4f | CKA %.4f | force KL %.4f\n",
              name, e.spearman, e.jacc10, e.recall, e.cka, e.fkl);
}

// InfoNCE example: positive from the teacher top 3; candidates = 4 mined (teacher ranks 4..30) + 8 random.
PullPushExample nce_example(const Teacher& t, int a, std::mt19937_64& e) {
  PullPushExample ex;
  ex.a = a;
  ex.b = t.nn[a][uni(e, 0, 2)];
  for (int k = 0; k < 4; ++k) ex.cands.push_back(t.nn[a][uni(e, 3, 29)]);
  for (int k = 0; k < 8; ++k) ex.cands.push_back(uni(e, 0, t.N - 1));
  return ex;
}

// Distillation examples for one anchor, two candidate sets (each its own softmax, so each stratum gets force):
//   local:  teacher top 10 + 10 from ranks 10..59 + 4 random
//   global: 24 random items (the softmax then ranks the far field, i.e. the between-cluster geometry)
std::vector<DistillExample> distill_examples(const Teacher& t, int a, std::mt19937_64& e) {
  auto rnd = [&]() {
    const int c = uni(e, 0, t.N - 2);
    return c >= a ? c + 1 : c;
  };
  std::vector<DistillExample> out(2);
  DistillExample& lo = out[0];
  lo.a = a;
  for (int k = 0; k < 10; ++k) lo.cands.push_back(t.nn[a][k]);
  for (int k = 0; k < 10; ++k) lo.cands.push_back(t.nn[a][uni(e, 10, 59)]);
  for (int k = 0; k < 4; ++k) lo.cands.push_back(rnd());
  lo.positive = t.nn[a][0];
  DistillExample& gl = out[1];
  gl.a = a;
  for (int k = 0; k < 24; ++k) gl.cands.push_back(rnd());
  for (DistillExample& ex : out)
    for (int c : ex.cands) ex.teacher_sims.push_back((float)t.S[(size_t)a * t.N + c]);
  return out;
}

std::vector<int> shuffled(int N, std::mt19937_64& e) {
  std::vector<int> o(N);
  for (int i = 0; i < N; ++i) o[i] = i;
  for (int i = N - 1; i > 0; --i) std::swap(o[i], o[uni(e, 0, i)]);
  return o;
}

void train_nce(const Teacher& t, ItemStore& s, const StepOptions& o, int epochs, std::uint64_t seed,
               const ParaphraseSet* para = nullptr) {
  std::mt19937_64 e(seed);
  for (int ep = 0; ep < epochs; ++ep)
    for (int a : shuffled(t.N, e)) pull_push_step(s, nce_example(t, a, e), o, para);
}

void train_distill(const Teacher& t, ItemStore& s, const StepOptions& o, int epochs, std::uint64_t seed) {
  std::mt19937_64 e(seed);
  for (int ep = 0; ep < epochs; ++ep)
    for (int a : shuffled(t.N, e))
      for (const DistillExample& ex : distill_examples(t, a, e)) distill_step(s, ex, o);
}

// ---------------------------------------------------------------------------------------------
// (a) known answers
// ---------------------------------------------------------------------------------------------

void test_known_answers() {
  std::printf("(a) known answers\n");
  // Spearman with ties, worked by hand:
  //   x = {1, 2, 2, 3} -> ranks {1, 2.5, 2.5, 4}; y = {1, 3, 2, 4} -> ranks {1, 3, 2, 4}
  //   deviations from 2.5: dx = {-1.5, 0, 0, 1.5}, dy = {-1.5, 0.5, -0.5, 1.5}
  //   sum dx dy = 4.5, sum dx^2 = 4.5, sum dy^2 = 5  ->  rho = 4.5 / sqrt(22.5) = sqrt(0.9)
  CHECK_NEAR(spearman({1, 2, 2, 3}, {1, 3, 2, 4}), std::sqrt(0.9), 1e-12);
  //   x = {10, 20, 20, 20, 30} -> ranks {1, 3, 3, 3, 5}; y = 1..5; rho = 8 / sqrt(8 * 10) = 2 / sqrt(5)
  CHECK_NEAR(spearman({10, 20, 20, 20, 30}, {1, 2, 3, 4, 5}), 2.0 / std::sqrt(5.0), 1e-12);
  CHECK_NEAR(spearman({1, 2, 3, 4}, {1, 8, 27, 64}), 1.0, 1e-12);
  CHECK_NEAR(spearman({1, 2, 3, 4}, {4, 3, 2, 1}), -1.0, 1e-12);
  const std::vector<double> rk = average_ranks({5, 1, 5, 3});
  CHECK_NEAR(rk[0], 3.5, 0);
  CHECK_NEAR(rk[1], 1.0, 0);
  CHECK_NEAR(rk[2], 3.5, 0);
  CHECK_NEAR(rk[3], 2.0, 0);

  // CKA(X, X R) = 1 for orthogonal R; invariant to isotropic scale; < 1 against unrelated Y
  {
    const int N = 200, d = 16;
    std::mt19937_64 e(11);
    std::vector<float> X((size_t)N * d), Y((size_t)N * d);
    for (float& v : X) v = (float)gauss(e);
    for (float& v : Y) v = (float)gauss(e);
    std::vector<double> A((size_t)d * d);
    for (double& v : A) v = gauss(e);
    std::vector<double> Sym((size_t)d * d);
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < d; ++j) Sym[(size_t)i * d + j] = A[(size_t)i * d + j] + A[(size_t)j * d + i];
    std::vector<double> ev, R;
    jacobi_eigen(Sym, d, ev, R);  // eigenvectors of a symmetric matrix: an orthogonal R
    double orth = 0;
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < d; ++j) {
        double s = 0;
        for (int k = 0; k < d; ++k) s += R[(size_t)k * d + i] * R[(size_t)k * d + j];
        orth = std::max(orth, std::fabs(s - (i == j ? 1.0 : 0.0)));
      }
    CHECK(orth < 1e-10);
    // eigen-decomposition reconstructs Sym
    double rec = 0;
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < d; ++j) {
        double s = 0;
        for (int k = 0; k < d; ++k) s += R[(size_t)i * d + k] * ev[k] * R[(size_t)j * d + k];
        rec = std::max(rec, std::fabs(s - Sym[(size_t)i * d + j]));
      }
    CHECK(rec < 1e-9);
    for (int k = 1; k < d; ++k) CHECK(ev[k - 1] >= ev[k]);
    std::vector<float> XR = X;
    rotate_rows(XR.data(), N, d, R);
    std::vector<float> Xs = X;
    for (float& v : Xs) v *= 3.7f;
    CHECK_NEAR(linear_cka(X.data(), d, XR.data(), d, N), 1.0, 1e-6);
    CHECK_NEAR(linear_cka(X.data(), d, Xs.data(), d, N), 1.0, 1e-6);
    CHECK_NEAR(linear_cka(X.data(), d, X.data(), d, N), 1.0, 1e-9);
    const double cxy = linear_cka(X.data(), d, Y.data(), d, N);
    CHECK(cxy < 0.3);
    std::printf("  CKA(X,XR) %.8f  CKA(X,3.7X) %.8f  CKA(X,Y random) %.4f\n",
                linear_cka(X.data(), d, XR.data(), d, N), linear_cka(X.data(), d, Xs.data(), d, N), cxy);

    // whitening: PCA and ZCA outputs have identity covariance; ZCA's W is symmetric
    for (WhitenMode mode : {WhitenMode::PCA, WhitenMode::ZCA}) {
      std::vector<float> Z((size_t)N * d);
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < d; ++j) Z[(size_t)i * d + j] = (float)(X[(size_t)i * d + j] * (1.0 + j) + 0.5 * X[(size_t)i * d + (j + 1) % d]);
      const Whitener w = fit_whitener(Z.data(), N, d, mode, 1e-9);
      const std::vector<float> Wz = w.apply(Z.data(), N);
      const std::vector<double> C = covariance(Wz.data(), N, w.d_out);
      double err = 0;
      for (int i = 0; i < w.d_out; ++i)
        for (int j = 0; j < w.d_out; ++j) err = std::max(err, std::fabs(C[(size_t)i * w.d_out + j] - (i == j ? 1.0 : 0.0)));
      CHECK(err < 1e-4);
      if (mode == WhitenMode::ZCA) {
        double asym = 0;
        for (int i = 0; i < d; ++i)
          for (int j = 0; j < d; ++j) asym = std::max(asym, std::fabs(w.W[(size_t)i * d + j] - w.W[(size_t)j * d + i]));
        CHECK(asym < 1e-9);
      }
    }
    std::vector<float> Xc = X;
    centre(Xc.data(), N, d);
    const std::vector<double> mc = column_mean(Xc.data(), N, d);
    double mx = 0;
    for (double v : mc) mx = std::max(mx, std::fabs(v));
    CHECK(mx < 1e-6);
  }

  // the cosine gradient is orthogonal to the vector (full and prefix), and matches a finite difference
  {
    std::mt19937_64 e(5);
    const int d = 24;
    std::vector<float> a(d), b(d);
    for (float& v : a) v = (float)gauss(e);
    for (float& v : b) v = (float)gauss(e);
    for (int m : {d, 8}) {
      std::vector<double> g(m);
      cosine_grad(a.data(), b.data(), m, g.data());
      double dp = 0, gn = 0;
      for (int j = 0; j < m; ++j) {
        dp += g[j] * a[j];
        gn += g[j] * g[j];
      }
      CHECK(std::fabs(dp) < 1e-9 * std::max(1.0, std::sqrt(gn)));
      // finite difference on coordinate 3
      std::vector<float> ap = a, am = a;
      const float h = 1e-3f;
      ap[3] += h;
      am[3] -= h;
      const double fd = (cosine(ap.data(), b.data(), m) - cosine(am.data(), b.data(), m)) / (2.0 * h);
      CHECK_NEAR(g[3], fd, 1e-4);
    }
  }

  // softmax weights: s = {0.5, 0.3, 0.1} at tau 0.1 -> e^5, e^3, e^1 over their sum; mask zeroes an entry
  {
    const std::vector<double> p = masked_softmax({0.5, 0.3, 0.1}, 0.1);
    const double z = std::exp(5.0) + std::exp(3.0) + std::exp(1.0);
    CHECK_NEAR(p[0], std::exp(5.0) / z, 1e-12);
    CHECK_NEAR(p[1], std::exp(3.0) / z, 1e-12);
    CHECK_NEAR(p[2], std::exp(1.0) / z, 1e-12);
    const std::vector<double> q = masked_softmax({0.5, 0.3, 0.1}, 0.1, {1, 0, 1});
    CHECK_NEAR(q[1], 0.0, 0);
    CHECK_NEAR(q[0], std::exp(5.0) / (std::exp(5.0) + std::exp(1.0)), 1e-12);
  }

  // the 4.2 margin mask: a candidate more similar than the positive + delta gets no push;
  // and one step lowers the InfoNCE loss and the distillation KL
  {
    ItemStore s(4, 3, 0.3f);
    const float va[3] = {1, 0, 0}, vb[3] = {0.6f, 0.8f, 0}, vc[3] = {0.95f, 0.31225f, 0}, vd[3] = {0, 0, 1};
    std::copy(va, va + 3, s.mean(0));
    std::copy(vb, vb + 3, s.mean(1));
    std::copy(vc, vc + 3, s.mean(2));
    std::copy(vd, vd + 3, s.mean(3));
    PullPushExample ex;
    ex.a = 0;
    ex.b = 1;
    ex.cands = {2, 3};  // s(a,c=2) = 0.95 > s(a,b) + 0.1 = 0.7 -> masked; c=3 kept
    StepOptions o;
    o.tau = 0.1;
    o.eta = 0.5;
    ItemStore s2 = s;
    const StepStats st = pull_push_step(s2, ex, o);
    CHECK(st.masked == 1);
    const double z = std::exp(0.6 / 0.1) + std::exp(0.0);
    CHECK_NEAR(st.p_pos, std::exp(0.6 / 0.1) / z, 1e-9);
    CHECK_NEAR(st.loss, -std::log(std::exp(0.6 / 0.1) / z), 1e-9);
    // the masked candidate is not moved at all
    for (int j = 0; j < 3; ++j) CHECK(s2.mean(2)[j] == s.mean(2)[j]);
    ItemStore s3 = s2;
    const StepStats st2 = pull_push_step(s3, ex, o);
    CHECK(st2.loss < st.loss);
    StepOptions no = o;
    no.use_mask = false;
    ItemStore s4 = s;
    CHECK(pull_push_step(s4, ex, no).masked == 0);
    // known paraphrase set: (b, d) admitted at Moderate, rejected at Low
    ParaphraseSet ps;
    CHECK(!ps.add(1, 3, cypha::eval::Grade::Low));
    CHECK(ps.add(1, 3, cypha::eval::Grade::Moderate));
    CHECK(ps.contains(3, 1));
    ItemStore s5 = s;
    CHECK(pull_push_step(s5, ex, o, &ps).masked == 2);

    DistillExample dx;
    dx.a = 0;
    dx.cands = {1, 2, 3};
    dx.teacher_sims = {0.9f, 0.2f, 0.1f};
    ItemStore s6 = s;
    StepOptions od;
    od.tau_T = od.tau_S = 0.1;
    od.eta = 0.5;
    const double l0 = distill_step(s6, dx, od).loss;
    const double l1 = distill_step(s6, dx, od).loss;
    CHECK(l1 < l0);
    // the force on each pair is (p_T - p_S) / tau_S: the mirror update moves b (teacher closer) toward the
    // anchor and c (student closer) away from it
    ItemStore s7 = s;
    distill_step(s7, dx, od);
    CHECK(cosine(s.mean(0), s7.mean(1), 3) > cosine(s.mean(0), s.mean(1), 3));
    CHECK(cosine(s.mean(0), s7.mean(2), 3) < cosine(s.mean(0), s.mean(2), 3));
  }

  // fp16 round trip
  {
    CHECK(fp32_to_fp16(1.0f) == 0x3c00u);
    CHECK(fp32_to_fp16(-2.0f) == 0xc000u);
    CHECK(fp32_to_fp16(65504.0f) == 0x7bffu);
    CHECK(fp32_to_fp16(1e6f) == 0x7c00u);
    CHECK(fp16_to_fp32(0x3555u) == 0.333251953125f);
    CHECK(fp16_to_fp32(0x0001u) == std::ldexp(1.0f, -24));
    CHECK(fp16_to_fp32(fp32_to_fp16(std::ldexp(1.0f, -20))) == std::ldexp(1.0f, -20));
    for (std::uint32_t h = 0; h < 0x7c00u; ++h) CHECK_NEAR(fp32_to_fp16(fp16_to_fp32((std::uint16_t)h)), h, 0);
    ItemStore s = ItemStore::random(50, 32, 3);
    s.set_world_prior_to_mean();
    const ItemStore r = ItemStore::import_fp16(s.export_fp16());
    double err = 0;
    for (size_t i = 0; i < s.mu.size(); ++i) err = std::max(err, (double)std::fabs(s.mu[i] - r.mu[i]));
    CHECK(err < 1e-3);
    CHECK(ItemStore::import_fp16(r.export_fp16()).state_hash() == r.state_hash());
  }
}

// ---------------------------------------------------------------------------------------------
// (b) planted teacher; (c) mask
// ---------------------------------------------------------------------------------------------

struct BResults {
  Eval start, nce, distill, combined;
};

StepOptions nce_options() {
  StepOptions o;
  o.tau = 0.05;
  o.eta = 20.0;
  o.kappa0 = 10.0;
  return o;
}

StepOptions distill_options() {
  StepOptions o;
  o.tau_T = 0.05;
  o.tau_S = 0.05;
  o.eta = 20.0;
  o.kappa0 = 10.0;
  return o;
}

constexpr int kEpochs = 60;

BResults test_planted(const Teacher& t, const PairSample& ps) {
  std::printf("(b) planted teacher: %d items, %d d, %d clusters, %zu paraphrase pairs, %zu hard negatives; student 32 d\n",
              t.N, t.D, t.K, t.para.size(), t.hardneg.size());
  BResults r;
  const ItemStore s0 = ItemStore::random(t.N, 32, 101);
  r.start = evaluate(t, s0, ps);
  print_eval("start (random)", r.start);

  ItemStore sn = s0;
  train_nce(t, sn, nce_options(), kEpochs, 7);
  r.nce = evaluate(t, sn, ps);
  print_eval("InfoNCE only (4.2 rule)", r.nce);

  ItemStore sd = s0;
  train_distill(t, sd, distill_options(), kEpochs, 7);
  r.distill = evaluate(t, sd, ps);
  print_eval("relational distillation", r.distill);

  ItemStore sc = s0;
  StepOptions oc = distill_options();
  oc.alpha = 0.7;
  oc.tau = 0.05;
  train_distill(t, sc, oc, kEpochs, 7);
  r.combined = evaluate(t, sc, ps);
  print_eval("0.7 KL + 0.3 InfoNCE", r.combined);

  // hard negatives: the planted pairs sit below the cluster's paraphrases after distillation
  double hn = 0, pp = 0;
  for (auto [i, j] : t.hardneg) hn += cosine(sd.mean(i), sd.mean(j), sd.d);
  for (auto [i, j] : t.para) pp += cosine(sd.mean(i), sd.mean(j), sd.d);
  hn /= t.hardneg.size();
  pp /= t.para.size();
  std::printf("  distilled: mean cos paraphrase pairs %.4f, planted hard negatives %.4f\n", pp, hn);
  CHECK(pp > hn + 0.2);

  CHECK(r.nce.spearman > r.start.spearman + 0.3);
  CHECK(r.distill.spearman >= 0.85);
  CHECK(r.distill.jacc10 >= 0.5);
  CHECK(r.distill.spearman > r.nce.spearman);
  CHECK(r.distill.jacc10 > r.nce.jacc10);
  CHECK(r.distill.fkl < r.nce.fkl);
  CHECK(r.combined.spearman >= 0.8);
  return r;
}

struct MaskResult {
  double on = 0, off = 0;
};

MaskResult test_mask(const Teacher& t) {
  std::printf("(c) false-negative mask\n");
  // anchors with a planted paraphrase partner p of their positive b in the candidate set
  ParaphraseSet known;
  for (auto [i, j] : t.para) known.add(i, j, cypha::eval::Grade::High);
  // Each planted pair (i, j): j is a near-duplicate of i. j is never another anchor's positive; its only pull is
  // as an anchor toward i. Whenever some anchor's positive is i, j is also sampled among its negatives: the case
  // the mask exists for ("without this, near-duplicates get torn apart").
  std::vector<int> dup_of(t.N, -1), dup(t.N, -1);
  for (auto [i, j] : t.para) {
    dup_of[j] = i;
    dup[i] = j;
  }
  // The student starts teacher-initialised (section 5.3): a seeded Gaussian projection of the teacher to 32 d,
  // so the near-duplicates start close, which is when an unmasked push can tear them apart.
  ItemStore s0(t.N, 32, 1.0f / 32);
  {
    std::mt19937_64 e(202);
    std::vector<double> P((size_t)t.D * 32);
    for (double& v : P) v = gauss(e);
    for (int i = 0; i < t.N; ++i) {
      float* m = s0.mean(i);
      for (int k = 0; k < 32; ++k) {
        double v = 0;
        for (int j = 0; j < t.D; ++j) v += t.row(i)[j] * P[(size_t)j * 32 + k];
        m[k] = (float)v;
      }
    }
    normalise_rows(s0.mu.data(), t.N, 32);
  }
  double start = 0;
  for (auto [i, j] : t.para) start += cosine(s0.mean(i), s0.mean(j), 32);
  start /= t.para.size();
  auto run = [&](bool mask) {
    ItemStore s = s0;
    StepOptions o = nce_options();
    o.use_mask = mask;
    std::mt19937_64 e(9);
    for (int ep = 0; ep < kEpochs; ++ep) {
      for (int a : shuffled(t.N, e)) {
        PullPushExample ex = nce_example(t, a, e);
        if (dup_of[a] >= 0) {
          ex.b = dup_of[a];
        } else {
          while (dup_of[ex.b] >= 0 || ex.b == a) ex.b = t.nn[a][uni(e, 0, 3)];
        }
        if (dup[ex.b] >= 0 && dup[ex.b] != a) ex.cands.push_back(dup[ex.b]);
        pull_push_step(s, ex, o, mask ? &known : nullptr);
      }
      // plus, per pair, 3 examples from i's teacher neighbours with positive i and j among the negatives
      for (auto [i, j] : t.para) {
        for (int r = 0; r < 3; ++r) {
          int a = t.nn[i][uni(e, 0, 5)];
          if (a == j || dup_of[a] >= 0) continue;
          PullPushExample ex = nce_example(t, a, e);
          ex.b = i;
          ex.cands.push_back(j);
          pull_push_step(s, ex, o, mask ? &known : nullptr);
        }
      }
    }
    double m = 0;
    for (auto [i, j] : t.para) m += cosine(s.mean(i), s.mean(j), s.d);
    return m / t.para.size();
  };
  MaskResult r;
  r.on = run(true);
  r.off = run(false);
  std::printf("  mean cos of planted paraphrase pairs: start %.4f; after InfoNCE mask on %.4f, mask off %.4f (teacher %.4f)\n",
              start, r.on, r.off,
              [&] {
                double m = 0;
                for (auto [i, j] : t.para) m += t.S[(size_t)i * t.N + j];
                return m / t.para.size();
              }());
  CHECK(r.on > r.off + 0.05);
  return r;
}

// ---------------------------------------------------------------------------------------------
// (d) confidence-scaled steps and MDL
// ---------------------------------------------------------------------------------------------

void test_confidence_steps() {
  std::printf("(d) confidence-scaled steps and MDL\n");
  const int d = 16;
  ItemStore s = ItemStore::random(2, d, 4, 0.05f);
  std::copy(s.mean(0), s.mean(0) + d, s.mean(1));  // identical starting means and variances
  s.n[0] = 1000.0f;                                // well evidenced
  s.n[1] = 1.0f;                                   // seen once
  std::vector<double> g(d);
  std::mt19937_64 e(8);
  for (double& v : g) v = gauss(e);
  StepOptions o;
  o.eta = 1.0;
  ItemStore before = s;
  natural_gradient_step(s, 0, g, o);
  natural_gradient_step(s, 1, g, o);
  const double move0 = std::acos(std::min(1.0, cosine(before.mean(0), s.mean(0), d)));
  const double move1 = std::acos(std::min(1.0, cosine(before.mean(1), s.mean(1), d)));
  std::printf("  angle moved: n = 1000 -> %.6f rad, n = 1 -> %.6f rad\n", move0, move1);
  CHECK(move0 < move1);
  CHECK(move1 > 20.0 * move0);

  // MDL: an item seen once is pulled toward theta0; a well-evidenced one barely moves
  ItemStore m = ItemStore::random(2, d, 12);
  std::vector<float> th(d);
  for (float& v : th) v = (float)gauss(e);
  normalise(th);
  m.set_world_prior(th.data());
  m.n[0] = 1.0f;
  m.n[1] = 1000.0f;
  const double c0 = cosine(m.mean(0), th.data(), d), c1 = cosine(m.mean(1), th.data(), d);
  mdl_shrink(m, 0, 0.5);
  mdl_shrink(m, 1, 0.5);
  const double c0b = cosine(m.mean(0), th.data(), d), c1b = cosine(m.mean(1), th.data(), d);
  std::printf("  MDL lambda 0.5: cos to theta0, n = 1: %.4f -> %.4f; n = 1000: %.4f -> %.4f\n", c0, c0b, c1, c1b);
  CHECK(c0b > c0 + 0.1);
  CHECK(std::fabs(c1b - c1) < 0.01);
  // and inside the step: same example, lambda_mdl > 0 moves a once-seen anchor toward theta0
  ItemStore p = ItemStore::random(6, d, 13);
  p.set_world_prior(th.data());
  PullPushExample ex;
  ex.a = 0;
  ex.b = 1;
  ex.cands = {2, 3, 4, 5};
  StepOptions om;
  om.lambda_mdl = 0.5;
  StepOptions on;
  ItemStore pa = p, pb = p;
  pull_push_step(pa, ex, om);
  pull_push_step(pb, ex, on);
  CHECK(cosine(pa.mean(0), th.data(), d) > cosine(pb.mean(0), th.data(), d));
}

// ---------------------------------------------------------------------------------------------
// (e) CSLS and hubness
// ---------------------------------------------------------------------------------------------

struct CslsResult {
  double cos_skew = 0, csls_skew = 0;
};

CslsResult test_csls() {
  std::printf("(e) CSLS and hubness\n");
  const int N = 600, d = 32, H = 12;
  std::mt19937_64 e(31);
  std::vector<float> common(d);
  for (float& v : common) v = (float)gauss(e);
  normalise(common);
  std::vector<float> X((size_t)N * d);
  for (int i = 0; i < N; ++i) {
    std::vector<float> x(d);
    const double w = i < H ? 1.0 : 0.25;  // planted hubs sit near the common direction
    const double nz = i < H ? 0.08 : 0.2;
    for (int j = 0; j < d; ++j) x[j] = (float)(w * common[j] + nz * gauss(e));
    normalise(x);
    std::copy(x.begin(), x.end(), X.begin() + (size_t)i * d);
  }
  const KnnLists kc = knn_cosine(X.data(), N, d, 10);
  const KnnLists ks = knn_csls(X.data(), N, d, 10, 10);
  CslsResult r;
  r.cos_skew = k_occurrence_skewness(kc, N);
  r.csls_skew = k_occurrence_skewness(ks, N);
  const std::vector<double> oc = k_occurrence(kc, N), os = k_occurrence(ks, N);
  double hc = 0, hs = 0;
  for (int i = 0; i < H; ++i) {
    hc += oc[i];
    hs += os[i];
  }
  std::printf("  k-occurrence skewness (k = 10): cosine %.3f, CSLS %.3f; mean N_10 of the %d hubs: %.1f -> %.1f\n",
              r.cos_skew, r.csls_skew, H, hc / H, hs / H);
  CHECK(r.cos_skew > 1.0);
  CHECK(r.csls_skew < r.cos_skew - 0.5);
  CHECK(hs < hc);
  return r;
}

// ---------------------------------------------------------------------------------------------
// (f) MRL
// ---------------------------------------------------------------------------------------------

struct MrlResult {
  std::vector<int> prefixes;
  std::vector<double> before, after;
  double nested32 = 0, flat32 = 0;
};

MrlResult test_mrl(const Teacher& t, const PairSample& tps) {
  std::printf("(f) MRL ordering and nested loss\n");
  MrlResult r;
  const int N = 800, d = 256;
  std::mt19937_64 e(41);
  // a decaying spectrum lambda_j ~ j^-1.2 in a random basis
  std::vector<double> A((size_t)d * d);
  for (double& v : A) v = gauss(e);
  std::vector<double> Sym((size_t)d * d);
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) Sym[(size_t)i * d + j] = A[(size_t)i * d + j] + A[(size_t)j * d + i];
  std::vector<double> ev, R;
  jacobi_eigen(Sym, d, ev, R);
  ItemStore s(N, d, 1.0f / d);
  for (int i = 0; i < N; ++i) {
    std::vector<float> z(d);
    for (int j = 0; j < d; ++j) z[j] = (float)(gauss(e) * std::pow(1.0 + j, -0.6));
    float* m = s.mean(i);
    for (int k = 0; k < d; ++k) {
      double v = 0;
      for (int j = 0; j < d; ++j) v += R[(size_t)k * d + j] * z[j];
      m[k] = (float)v;
    }
  }
  const std::vector<float> full = s.mu;
  const PairSample ps = stratified_pairs(full.data(), N, d, 4, 4, 8, 3);
  r.prefixes = mrl_prefixes(d);
  for (int m : r.prefixes) r.before.push_back(pair_spearman(full.data(), d, s.mu.data(), d, ps.pairs, m));
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<double> lam = mrl_order(s);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  for (int m : r.prefixes) r.after.push_back(pair_spearman(full.data(), d, s.mu.data(), d, ps.pairs, m));
  std::printf("  d = %d, prefixes", d);
  for (int m : r.prefixes) std::printf(" %d", m);
  std::printf("; PCA ordering %.0f ms\n  prefix agreement (pair Spearman vs full) before ordering:", ms);
  for (double v : r.before) std::printf(" %.4f", v);
  std::printf("\n  after ordering:");
  for (double v : r.after) std::printf(" %.4f", v);
  std::printf("\n");
  for (size_t k = 1; k < r.after.size(); ++k) CHECK(r.after[k] > r.after[k - 1]);
  CHECK_NEAR(r.after.back(), 1.0, 1e-6);
  CHECK(r.after[0] > r.before[0] + 0.1);
  for (size_t k = 1; k < lam.size(); ++k) CHECK(lam[k - 1] >= lam[k]);
  // cosines unchanged by the rotation
  double dc = 0;
  for (int k = 0; k < 50; ++k) {
    const auto [i, j] = ps.pairs[k];
    dc = std::max(dc, std::fabs(cosine(full.data() + (size_t)i * d, full.data() + (size_t)j * d, d) -
                                cosine(s.mean(i), s.mean(j), d)));
  }
  CHECK(dc < 1e-5);

  // nested loss: a 64-d student distilled with prefixes {32, 64} agrees better at 32 than a flat one
  ItemStore a = ItemStore::random(t.N, 64, 303), b = a;
  StepOptions on = distill_options();
  on.prefixes = mrl_prefixes(64);
  train_distill(t, a, on, kEpochs / 2, 17);
  train_distill(t, b, distill_options(), kEpochs / 2, 17);
  r.nested32 = pair_spearman(t.X.data(), t.D, a.mu.data(), 64, tps.pairs, 32);
  r.flat32 = pair_spearman(t.X.data(), t.D, b.mu.data(), 64, tps.pairs, 32);
  const double n64 = pair_spearman(t.X.data(), t.D, a.mu.data(), 64, tps.pairs, 64);
  const double f64 = pair_spearman(t.X.data(), t.D, b.mu.data(), 64, tps.pairs, 64);
  std::printf("  64-d student, pair Spearman with teacher at prefix 32 / 64: nested %.4f / %.4f, flat %.4f / %.4f\n",
              r.nested32, n64, r.flat32, f64);
  CHECK(r.nested32 > r.flat32);
  return r;
}

// ---------------------------------------------------------------------------------------------
// (g) confidence check
// ---------------------------------------------------------------------------------------------

struct ConfResult {
  double var_noisy = 0, var_clean = 0, dis_noisy = 0, dis_clean = 0, rho = 0;
};

ConfResult conf_summary(const ItemStore& s, const Teacher& t, const std::vector<char>& noisy) {
  ConfResult r;
  const std::vector<double> dis = item_disagreement(t.X.data(), t.D, s.mu.data(), s.d, t.N);
  int nn = 0, nc = 0;
  for (int i = 0; i < t.N; ++i) {
    if (noisy[i]) {
      r.var_noisy += s.mean_variance(i);
      r.dis_noisy += dis[i];
      ++nn;
    } else {
      r.var_clean += s.mean_variance(i);
      r.dis_clean += dis[i];
      ++nc;
    }
  }
  r.var_noisy /= nn;
  r.var_clean /= nc;
  r.dis_noisy /= nn;
  r.dis_clean /= nc;
  r.rho = confidence_check(s, t.X.data(), t.D);
  return r;
}

std::pair<ConfResult, ConfResult> test_confidence_check(const Teacher& t) {
  std::printf("(g) confidence check (variance vs disagreement); a quarter of the items get noisier training\n");
  std::vector<char> noisy(t.N, 0);
  std::mt19937_64 e(51);
  for (int i : shuffled(t.N, e))
    if (std::count(noisy.begin(), noisy.end(), 1) < t.N / 4) noisy[i] = 1;
  // distillation (the main route): the noisy anchors' teacher similarities carry N(0, 0.3^2) noise
  ItemStore sd = ItemStore::random(t.N, 32, 404);
  {
    const StepOptions o = distill_options();
    for (int ep = 0; ep < kEpochs; ++ep)
      for (int a : shuffled(t.N, e))
        for (DistillExample ex : distill_examples(t, a, e)) {
          if (noisy[a])
            for (float& v : ex.teacher_sims) v += (float)(0.3 * gauss(e));
          distill_step(sd, ex, o);
        }
  }
  // InfoNCE: half of the noisy anchors' positives are random items
  ItemStore sn = ItemStore::random(t.N, 32, 405);
  {
    const StepOptions o = nce_options();
    for (int ep = 0; ep < kEpochs; ++ep) {
      for (int a : shuffled(t.N, e)) {
        PullPushExample ex = nce_example(t, a, e);
        if (noisy[a] && uni(e, 0, 1) == 0) {
          const int c = uni(e, 0, t.N - 2);
          ex.b = c >= a ? c + 1 : c;
        }
        pull_push_step(sn, ex, o);
      }
    }
  }
  const ConfResult rd = conf_summary(sd, t, noisy), rn = conf_summary(sn, t, noisy);
  for (const auto& [name, r] : {std::pair<const char*, ConfResult>{"distillation", rd}, {"InfoNCE", rn}})
    std::printf("  %-12s mean variance noisy %.5f, clean %.5f | disagreement noisy %.4f, clean %.4f | "
                "Spearman(variance, disagreement) %.4f\n",
                name, r.var_noisy, r.var_clean, r.dis_noisy, r.dis_clean, r.rho);
  CHECK(rd.var_noisy > rd.var_clean);
  CHECK(rd.dis_noisy > rd.dis_clean);
  CHECK(rd.rho > 0.15);
  CHECK(rn.var_noisy > rn.var_clean);
  CHECK(rn.rho > 0.0);
  return {rd, rn};
}

// ---------------------------------------------------------------------------------------------
// (h) determinism
// ---------------------------------------------------------------------------------------------

void test_determinism(const Teacher& t) {
  std::printf("(h) determinism\n");
  auto run = [&](std::uint64_t seed) {
    ItemStore s = ItemStore::random(t.N, 32, seed);
    StepOptions o = distill_options();
    o.alpha = 0.7;
    o.lambda_mdl = 0.1;
    s.set_world_prior_to_mean();
    train_distill(t, s, o, 3, seed);
    train_nce(t, s, nce_options(), 2, seed);
    mrl_order(s);
    return s.state_hash();
  };
  const std::uint64_t h1 = run(77), h2 = run(77), h3 = run(78);
  std::printf("  state hash seed 77: %016llx, again: %016llx, seed 78: %016llx\n", (unsigned long long)h1,
              (unsigned long long)h2, (unsigned long long)h3);
  CHECK(h1 == h2);
  CHECK(h1 != h3);
}

}  // namespace

int main() {
  const auto t0 = std::chrono::steady_clock::now();
  test_known_answers();
  const Teacher t = make_teacher(20, 20, 64, 8, 0.6, 2026);
  const PairSample ps = stratified_pairs(t.X.data(), t.N, t.D, 5, 5, 10, 99);
  test_planted(t, ps);
  test_mask(t);
  test_confidence_steps();
  test_csls();
  test_mrl(t, ps);
  test_confidence_check(t);
  test_determinism(t);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("\n%d checks, %d failed (%.1f s)\n", g_checks, g_failed, s);
  return g_failed == 0 ? 0 : 1;
}
