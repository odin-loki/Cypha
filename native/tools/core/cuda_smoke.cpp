/// cuda_smoke — verify cypha::accel (CUDA GPU or parallel CPU) vs reference.
///
/// Exit codes:
///   0  correctness checks passed
///   2  --bench requested but no CUDA GPU (skip)
///   1  correctness failure
///
/// Usage: cuda_smoke [--bench]

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cypha/accel_backend.hpp"
#include "cypha/nig_gig_math.hpp"

namespace {

double max_abs_diff(const double* a, const double* b, int n) {
  double m = 0.0;
  for (int i = 0; i < n; i++) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

void ref_batch_encode(const double* x, int n, int d, const double* w, double* h) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < d; j++) {
      double acc = 0.0;
      for (int k = 0; k < d; k++) acc += x[i * d + k] * w[j * d + k];
      h[i * d + j] = acc;
    }
}

void ref_score_matrix(const double* H, int n, int d, int K, const double* mu0, const double* inv_v,
                      const double* D, const double* Dsq, const double* uk, const double* ctx,
                      double* llr) {
  for (int i = 0; i < n; i++)
    for (int k = 0; k < K; k++) {
      double c = 0;
      for (int j = 0; j < d; j++) c += (H[i * d + j] - mu0[j]) * inv_v[j] * D[k * d + j];
      llr[i * K + k] = c - 0.5 * Dsq[k] - uk[k] + ctx[k];
    }
}

void ref_softmax(const double* logits, int n, int K, double T, double* probs) {
  double invT = 1.0 / std::max(T, 1e-8);
  for (int i = 0; i < n; i++) {
    const double* row = logits + i * K;
    double mx = row[0];
    for (int k = 1; k < K; k++)
      if (row[k] > mx) mx = row[k];
    double s = 0.0;
    double* out = probs + i * K;
    for (int k = 0; k < K; k++) {
      out[k] = std::exp((row[k] - mx) * invT);
      s += out[k];
    }
    s = std::max(s, 1e-300);
    for (int k = 0; k < K; k++) out[k] /= s;
  }
}

void ref_world_gate(const double* H, int n, int d, const double* psi, double chi, double* g) {
  for (int i = 0; i < n; i++) {
    double dot = 0;
    for (int j = 0; j < d; j++) dot += H[i * d + j] * psi[j];
    g[i] = std::tanh(chi * dot);
  }
}

void ref_world_gate_nig(const double* H, int n, int d, const double* mu0, const double* inv_v,
                        double r_base, double gh_chi, double gh_psi, double* g) {
  const double denom_d = static_cast<double>(std::max(d, 1));
  for (int i = 0; i < n; i++) {
    double mahal = 0.0;
    for (int j = 0; j < d; j++) {
      double diff = H[i * d + j] - mu0[j];
      double rj = diff * inv_v[j];
      mahal += diff * rj;
    }
    mahal /= denom_d;
    double mp = std::max(mahal, 0.0);
    double r_eff = cypha::nig_r_eff_scalar(mp, r_base, gh_chi, gh_psi);
    g[i] = r_base / std::max(r_eff, r_base);
  }
}

using Clock = std::chrono::high_resolution_clock;
double ms_since(std::chrono::time_point<Clock> t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  bool bench = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--bench") bench = true;
  }

  cypha::accel::init();
  std::cout << "Backend: " << cypha::accel::device_info() << "\n";
  std::cout << "CUDA active: " << (cypha::accel::is_available() ? "yes" : "no (parallel CPU)") << "\n";

  const int N = 64, d = 128, K = 16;
  std::mt19937 rng(12345);
  std::normal_distribution<double> nd;
  std::uniform_real_distribution<double> ud(0.01, 2.0);

  std::vector<double> X(N * d), W(d * d), H_ref(N * d), H_acc(N * d);
  std::vector<double> mu0(d), inv_v(d), D(K * d), Dsq(K), uk(K), ctx(K);
  std::vector<double> LLR_ref(N * K), LLR_acc(N * K);
  std::vector<double> logits(N * K), P_ref(N * K), P_acc(N * K);
  std::vector<double> psi(d), G_ref(N), G_acc(N);
  std::vector<double> Gn_ref(N), Gn_acc(N);
  double chi = 1.5;
  const double r_base = 0.35;
  const double gh_chi = 1.0;
  const double gh_psi = 1.0;

  for (auto& v : X) v = nd(rng);
  for (auto& v : W) v = nd(rng) * 0.1;
  for (auto& v : mu0) v = nd(rng) * 0.5;
  for (auto& v : inv_v) v = ud(rng);
  for (auto& v : D) v = nd(rng) * 0.3;
  for (int k = 0; k < K; k++) {
    double s = 0;
    for (int j = 0; j < d; j++) s += D[k * d + j] * D[k * d + j] * inv_v[j];
    Dsq[k] = s;
    uk[k] = 0.01;
    ctx[k] = nd(rng) * 0.1;
  }
  for (auto& v : logits) v = nd(rng) * 2.0;
  for (auto& v : psi) v = nd(rng) * 0.2;

  ref_batch_encode(X.data(), N, d, W.data(), H_ref.data());
  ref_score_matrix(H_ref.data(), N, d, K, mu0.data(), inv_v.data(), D.data(), Dsq.data(), uk.data(),
                   ctx.data(), LLR_ref.data());
  ref_softmax(logits.data(), N, K, 1.0, P_ref.data());
  ref_world_gate(H_ref.data(), N, d, psi.data(), chi, G_ref.data());
  ref_world_gate_nig(H_ref.data(), N, d, mu0.data(), inv_v.data(), r_base, gh_chi, gh_psi, Gn_ref.data());

  cypha::accel::batch_encode(X.data(), N, d, W.data(), H_acc.data());
  cypha::accel::score_matrix(H_acc.data(), N, d, K, mu0.data(), inv_v.data(), D.data(), Dsq.data(),
                             uk.data(), ctx.data(), LLR_acc.data());
  cypha::accel::softmax_rows(logits.data(), N, K, 1.0, P_acc.data());
  cypha::accel::world_gate_batch(H_acc.data(), N, d, psi.data(), chi, G_acc.data());
  cypha::accel::world_gate_nig_field_batch(H_acc.data(), N, d, mu0.data(), inv_v.data(), r_base, gh_chi,
                                           gh_psi, Gn_acc.data());

  constexpr double kAtol = 1e-9;
  bool pass = true;
  auto check = [&](const char* name, const double* got, const double* exp, int n) {
    double err = max_abs_diff(got, exp, n);
    bool ok2 = err <= kAtol;
    std::cout << (ok2 ? "PASS " : "FAIL ") << name << "  max_err=" << err << "  atol=" << kAtol << "\n";
    if (!ok2) pass = false;
  };

  check("batch_encode", H_acc.data(), H_ref.data(), N * d);
  check("score_matrix", LLR_acc.data(), LLR_ref.data(), N * K);
  check("softmax_rows", P_acc.data(), P_ref.data(), N * K);
  check("world_gate", G_acc.data(), G_ref.data(), N);
  check("world_gate_nig", Gn_acc.data(), Gn_ref.data(), N);

  if (bench) {
    if (!cypha::accel::is_available()) {
      std::cout << "cuda_smoke: --bench requires a CUDA GPU (build with -DCYPHA_ENABLE_CUDA=ON) -- SKIP\n";
      cypha::accel::shutdown();
      return 2;
    }
    const int REPS = 200;
    auto t0 = Clock::now();
    for (int r = 0; r < REPS; r++) ref_batch_encode(X.data(), N, d, W.data(), H_ref.data());
    double cpu_enc_ms = ms_since(t0) / REPS;

    t0 = Clock::now();
    for (int r = 0; r < REPS; r++) cypha::accel::batch_encode(X.data(), N, d, W.data(), H_acc.data());
    double gpu_enc_ms = ms_since(t0) / REPS;

    t0 = Clock::now();
    for (int r = 0; r < REPS; r++)
      ref_score_matrix(H_ref.data(), N, d, K, mu0.data(), inv_v.data(), D.data(), Dsq.data(), uk.data(),
                       ctx.data(), LLR_ref.data());
    double cpu_sm_ms = ms_since(t0) / REPS;

    t0 = Clock::now();
    for (int r = 0; r < REPS; r++)
      cypha::accel::score_matrix(H_acc.data(), N, d, K, mu0.data(), inv_v.data(), D.data(), Dsq.data(),
                                 uk.data(), ctx.data(), LLR_acc.data());
    double gpu_sm_ms = ms_since(t0) / REPS;

    std::cout << "\n--- Benchmark (N=" << N << " d=" << d << " K=" << K << " avg over " << REPS << " reps) ---\n";
    std::cout << "batch_encode  CPU(ref)=" << cpu_enc_ms << "ms  CUDA=" << gpu_enc_ms
              << "ms  speedup=" << cpu_enc_ms / std::max(gpu_enc_ms, 0.001) << "x\n";
    std::cout << "score_matrix  CPU(ref)=" << cpu_sm_ms << "ms  CUDA=" << gpu_sm_ms
              << "ms  speedup=" << cpu_sm_ms / std::max(gpu_sm_ms, 0.001) << "x\n";
  }

  cypha::accel::shutdown();

  if (pass) {
    std::cout << "\nAll accel correctness checks PASSED.\n";
    return 0;
  }
  std::cerr << "\nSome accel checks FAILED.\n";
  return 1;
}
