#include "cypha/accel_backend.hpp"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cypha/nig_gig_math.hpp"

#ifdef CYPHA_ENABLE_CUDA
extern "C" {
int cypha_accel_cuda_try_init(char* name_out, int name_cap);
void cypha_accel_cuda_shutdown();
int cypha_accel_cuda_ready();
const char* cypha_accel_cuda_device_name();
void cypha_accel_cuda_batch_encode(const double* x, int n, int d, const double* w, double* h);
void cypha_accel_cuda_score_matrix(const double* H, int n, int d, int K, const double* mu0,
                                   const double* inv_v, const double* D, const double* D_sq,
                                   const double* u_k, const double* ctx, double* llr);
void cypha_accel_cuda_softmax_rows(const double* logits, int n, int K, double inv_T, double* probs);
void cypha_accel_cuda_world_gate(const double* H, int n, int d, const double* psi, double chi,
                                 double* gates);
void cypha_accel_cuda_world_gate_nig(const double* H, int n, int d, const double* mu0,
                                     const double* inv_v, double r_base, double gh_chi, double gh_psi,
                                     double* gates);
}
#endif

#ifndef CYPHA_ACCEL_GPU_MIN_BATCH_ROWS
#define CYPHA_ACCEL_GPU_MIN_BATCH_ROWS 1
#endif

namespace cypha {
namespace accel {

namespace {

std::mutex g_mu;
std::atomic<bool> g_inited{false};
std::atomic<bool> g_gpu{false};
std::string g_info;

/// When CUDA is ready, use the GPU for all batch sizes ``n >= kGpuMinBatchRows`` (default 1).
/// Override via ``-DCYPHA_ACCEL_GPU_MIN_BATCH_ROWS=N`` at CMake configure.
static constexpr int kGpuMinBatchRows = CYPHA_ACCEL_GPU_MIN_BATCH_ROWS;

#if defined(CYPHA_ENABLE_CUDA)
static bool cuda_path_for_batch(int n) {
  return g_gpu.load(std::memory_order_acquire) && cypha_accel_cuda_ready() && n >= kGpuMinBatchRows;
}
#endif

static int thread_workers() {
  unsigned n = std::thread::hardware_concurrency();
  return n ? static_cast<int>(n) : 4;
}

template <class F>
static void parallel_rows(int begin, int end, F&& f) {
  const int n = end - begin;
  if (n <= 0) return;
  int nt = std::min(thread_workers(), n);
  if (nt <= 1) {
    for (int i = begin; i < end; ++i) f(i);
    return;
  }
  const int chunk = (n + nt - 1) / nt;
  std::vector<std::thread> th;
  th.reserve(static_cast<size_t>(nt));
  for (int t = 0; t < nt; ++t) {
    int lo = begin + t * chunk;
    int hi = std::min(end, lo + chunk);
    if (lo >= hi) break;
    th.emplace_back([lo, hi, &f]() {
      for (int i = lo; i < hi; ++i) f(i);
    });
  }
  for (auto& x : th) x.join();
}

// ── CPU parallel (ISO C++ threads) ───────────────────────────────────────

static void cpu_parallel_batch_encode(const double* x, int n, int d, const double* w, double* h) {
  parallel_rows(0, n, [&](int i) {
    for (int j = 0; j < d; ++j) {
      double acc = 0.0;
      for (int k = 0; k < d; ++k) acc += x[i * d + k] * w[j * d + k];
      h[i * d + j] = acc;
    }
  });
}

static void cpu_parallel_score_matrix(const double* H, int n, int d, int K, const double* mu0,
                                      const double* inv_v, const double* D, const double* D_sq,
                                      const double* u_k, const double* ctx, double* llr) {
  parallel_rows(0, n, [&](int i) {
    for (int k = 0; k < K; ++k) {
      double cross = 0.0;
      for (int j = 0; j < d; ++j)
        cross += (H[i * d + j] - mu0[j]) * inv_v[j] * D[k * d + j];
      llr[i * K + k] = cross - 0.5 * D_sq[k] - u_k[k] + ctx[k];
    }
  });
}

static void cpu_parallel_softmax(const double* logits, int n, int K, double inv_T, double* probs) {
  parallel_rows(0, n, [&](int i) {
    const double* row = logits + i * K;
    double mx = row[0];
    for (int k = 1; k < K; ++k)
      if (row[k] > mx) mx = row[k];
    double s = 0.0;
    double* out = probs + i * K;
    for (int k = 0; k < K; ++k) {
      out[k] = std::exp((row[k] - mx) * inv_T);
      s += out[k];
    }
    s = std::max(s, 1e-300);
    for (int k = 0; k < K; ++k) out[k] /= s;
  });
}

static void cpu_parallel_world_gate(const double* H, int n, int d, const double* psi, double chi,
                                    double* g) {
  parallel_rows(0, n, [&](int i) {
    double dot = 0.0;
    for (int j = 0; j < d; ++j) dot += H[i * d + j] * psi[j];
    g[i] = std::tanh(chi * dot);
  });
}

static void cpu_parallel_world_gate_nig(const double* H, int n, int d, const double* mu0,
                                        const double* inv_v, double r_base, double gh_chi, double gh_psi,
                                        double* g) {
  const double denom_d = std::max(d, 1);
  parallel_rows(0, n, [&](int i) {
    double mahal_per_dim = 0.0;
    for (int j = 0; j < d; ++j) {
      double diff = H[i * d + j] - mu0[j];
      double rj = diff * inv_v[j];
      mahal_per_dim += diff * rj;
    }
    mahal_per_dim /= denom_d;
    double mp = std::max(mahal_per_dim, 0.0);
    double r_eff = cypha::nig_r_eff_scalar(mp, r_base, gh_chi, gh_psi);
    g[i] = r_base / std::max(r_eff, r_base);
  });
}

/// Ensure one-time init; returns whether GPU path is active.
static bool ensure_inited() {
  if (g_inited.load(std::memory_order_acquire)) {
    return g_gpu.load(std::memory_order_acquire);
  }
  return init();
}

}  // namespace

bool init() {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_inited.load(std::memory_order_relaxed)) {
    return g_gpu.load(std::memory_order_relaxed);
  }
  bool gpu = false;
#ifdef CYPHA_ENABLE_CUDA
  char buf[512]{};
  if (cypha_accel_cuda_try_init(buf, static_cast<int>(sizeof(buf)))) {
    gpu = true;
    g_info = buf[0] ? std::string(buf) : std::string(cypha_accel_cuda_device_name());
  } else
#endif
  {
    g_info = std::string("CPU (std::thread, ") + std::to_string(thread_workers()) + " workers)";
  }
  g_gpu.store(gpu, std::memory_order_release);
  g_inited.store(true, std::memory_order_release);
  return gpu;
}

bool is_available() {
  ensure_inited();
  return g_gpu.load(std::memory_order_acquire);
}

std::string device_info() {
  ensure_inited();
  // g_info is written once under g_mu during init; immutable afterwards until shutdown.
  std::lock_guard<std::mutex> lk(g_mu);
  return g_info.empty() ? std::string("not initialised") : g_info;
}

void shutdown() {
  std::lock_guard<std::mutex> lk(g_mu);
#ifdef CYPHA_ENABLE_CUDA
  cypha_accel_cuda_shutdown();
#endif
  g_gpu.store(false, std::memory_order_release);
  g_inited.store(false, std::memory_order_release);
  g_info.clear();
}

void batch_encode(const double* x_row, int n, int d, const double* w_row, double* h_out) {
  if (n <= 0 || d <= 0) {
    return;
  }
  (void)ensure_inited();
#if defined(CYPHA_ENABLE_CUDA)
  if (cuda_path_for_batch(n)) {
    std::lock_guard<std::mutex> lk(g_mu);
    cypha_accel_cuda_batch_encode(x_row, n, d, w_row, h_out);
    return;
  }
#endif
  cpu_parallel_batch_encode(x_row, n, d, w_row, h_out);
}

void score_matrix(const double* h_row, int n, int d, int K, const double* mu0, const double* inv_v,
                  const double* D_row, const double* D_sq, const double* u_k, const double* ctx,
                  double* llr_out) {
  if (n <= 0 || K <= 0 || d <= 0) {
    return;
  }
  (void)ensure_inited();
#if defined(CYPHA_ENABLE_CUDA)
  if (cuda_path_for_batch(n)) {
    std::lock_guard<std::mutex> lk(g_mu);
    cypha_accel_cuda_score_matrix(h_row, n, d, K, mu0, inv_v, D_row, D_sq, u_k, ctx, llr_out);
    return;
  }
#endif
  cpu_parallel_score_matrix(h_row, n, d, K, mu0, inv_v, D_row, D_sq, u_k, ctx, llr_out);
}

void softmax_rows(const double* logits, int n, int K, double temperature, double* probs_out) {
  if (n <= 0 || K <= 0) {
    return;
  }
  double inv_T = 1.0 / std::max(temperature, 1e-8);
  (void)ensure_inited();
#if defined(CYPHA_ENABLE_CUDA)
  if (cuda_path_for_batch(n)) {
    std::lock_guard<std::mutex> lk(g_mu);
    cypha_accel_cuda_softmax_rows(logits, n, K, inv_T, probs_out);
    return;
  }
#endif
  cpu_parallel_softmax(logits, n, K, inv_T, probs_out);
}

void world_gate_batch(const double* h_row, int n, int d, const double* psi_vec, double chi,
                      double* gates_out) {
  if (n <= 0 || d <= 0) {
    return;
  }
  (void)ensure_inited();
#if defined(CYPHA_ENABLE_CUDA)
  if (cuda_path_for_batch(n)) {
    std::lock_guard<std::mutex> lk(g_mu);
    cypha_accel_cuda_world_gate(h_row, n, d, psi_vec, chi, gates_out);
    return;
  }
#endif
  cpu_parallel_world_gate(h_row, n, d, psi_vec, chi, gates_out);
}

void world_gate_nig_field_batch(const double* h_row, int n, int d, const double* mu0,
                                const double* inv_v, double r_base, double gh_chi, double gh_psi,
                                double* gates_out) {
  if (n <= 0 || d <= 0) {
    return;
  }
  (void)ensure_inited();
#if defined(CYPHA_ENABLE_CUDA)
  if (cuda_path_for_batch(n)) {
    std::lock_guard<std::mutex> lk(g_mu);
    cypha_accel_cuda_world_gate_nig(h_row, n, d, mu0, inv_v, r_base, gh_chi, gh_psi, gates_out);
    return;
  }
#endif
  cpu_parallel_world_gate_nig(h_row, n, d, mu0, inv_v, r_base, gh_chi, gh_psi, gates_out);
}

}  // namespace accel
}  // namespace cypha
