// CUDA implementation for cypha::accel (optional; CYPHA_ENABLE_CUDA).
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "cypha/bessel_table.hpp"

namespace {

bool g_ok = false;
char g_name[256]{};

double* g_pool = nullptr;
std::size_t g_pool_doubles = 0;

double* d_bessel_k2k1 = nullptr;

void chk(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[cypha_accel_cuda] %s: %s\n", where, cudaGetErrorString(e));
  }
}

cudaError_t pool_ensure(std::size_t nd) {
  if (nd <= g_pool_doubles) {
    return cudaSuccess;
  }
  cudaFree(g_pool);
  g_pool = nullptr;
  std::size_t n = nd;
  if (g_pool_doubles > 0 && n < g_pool_doubles * 2) {
    n = g_pool_doubles * 2;
  }
  if (n < 4096) {
    n = 4096;
  }
  cudaError_t e = cudaMalloc(&g_pool, n * sizeof(double));
  if (e == cudaSuccess) {
    g_pool_doubles = n;
  }
  return e;
}

void pool_clear() {
  cudaFree(g_pool);
  g_pool = nullptr;
  g_pool_doubles = 0;
}

cudaError_t ensure_bessel_table() {
  if (d_bessel_k2k1 != nullptr) {
    return cudaSuccess;
  }
  const std::size_t nb = cypha::detail::kBesselN * sizeof(double);
  cudaError_t e = cudaMalloc(&d_bessel_k2k1, nb);
  if (e != cudaSuccess) {
    return e;
  }
  return cudaMemcpy(d_bessel_k2k1, cypha::detail::kBesselK2K1, nb, cudaMemcpyHostToDevice);
}

void bessel_clear() {
  cudaFree(d_bessel_k2k1);
  d_bessel_k2k1 = nullptr;
}

__global__ void k_batch_encode(const double* X, const double* W, int n, int d, double* H) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * d;
  if (tid >= total) return;
  int i = tid / d;
  int j = tid % d;
  double acc = 0.0;
  int bx = i * d;
  int bw = j * d;
  for (int k = 0; k < d; ++k) acc += X[bx + k] * W[bw + k];
  H[i * d + j] = acc;
}

__global__ void k_score_matrix(const double* H, int n, int d, int K, const double* mu0,
                               const double* inv_v, const double* D, const double* D_sq,
                               const double* u_k, const double* ctx, double* LLR) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * K;
  if (tid >= total) return;
  int i = tid / K;
  int k = tid % K;
  double cross = 0.0;
  int bh = i * d;
  int bd = k * d;
  for (int j = 0; j < d; ++j)
    cross += (H[bh + j] - mu0[j]) * inv_v[j] * D[bd + j];
  LLR[i * K + k] = cross - 0.5 * D_sq[k] - u_k[k] + ctx[k];
}

__global__ void k_softmax_rows(const double* logits, int N, int K, double inv_T, double* probs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const double* row = logits + i * K;
  double* out = probs + i * K;
  double mx = row[0];
  for (int k = 1; k < K; ++k)
    if (row[k] > mx) mx = row[k];
  double s = 0.0;
  for (int k = 0; k < K; ++k) {
    double e = exp((row[k] - mx) * inv_T);
    out[k] = e;
    s += e;
  }
  s = fmax(s, 1e-300);
  for (int k = 0; k < K; ++k) out[k] /= s;
}

__global__ void k_world_gate(const double* H, int n, int d, const double* psi, double chi,
                             double* gates) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double dot = 0.0;
  int b = i * d;
  for (int j = 0; j < d; ++j) dot += H[b + j] * psi[j];
  gates[i] = tanh(chi * dot);
}

__device__ double d_np_interp_k2k1(const double* tab, double x) {
  const int N = 16384;
  const double x0 = 1e-6;
  const double x1 = 120.0;
  if (x <= x0) return tab[0];
  if (x >= x1) return tab[N - 1];
  const double step = (x1 - x0) / static_cast<double>(N - 1);
  double pos = (x - x0) / step;
  int i = static_cast<int>(pos);
  if (i >= N - 1) i = N - 2;
  double t = pos - static_cast<double>(i);
  return tab[i] * (1.0 - t) + tab[i + 1] * t;
}

__device__ double d_gig_e_inv_v(const double* bessel_tab, double chi0, double psi) {
  const double eps = 1e-8;
  if (chi0 < eps || psi < eps) {
    return psi / fmax(chi0, eps);
  }
  double chi_g = fmax(chi0, eps);
  double xv = sqrt(chi_g * psi);
  if (xv < 1e-6) {
    return psi / chi_g;
  }
  double chi_b = chi_g;
  double x_b = xv;
  if (x_b <= 120.0) {
    const double lo = 1e-6;
    const double hi = 120.0;
    double xt = fmin(fmax(x_b, lo), hi);
    double ratio = d_np_interp_k2k1(bessel_tab, xt);
    return sqrt(psi / chi_b) * ratio;
  }
  return psi / chi_b;
}

__global__ void k_world_gate_nig(const double* H, int n, int d, const double* mu0, const double* inv_v,
                                 double r_base, double gh_chi, double gh_psi, const double* bessel_tab,
                                 double* gates) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double eps = 1e-8;
  double denom_d = static_cast<double>(d > 0 ? d : 1);
  double mahal = 0.0;
  int b = i * d;
  for (int j = 0; j < d; ++j) {
    double diff = H[b + j] - mu0[j];
    double rj = diff * inv_v[j];
    mahal += diff * rj;
  }
  mahal /= denom_d;
  double mp = fmax(mahal, 0.0);
  double chi_post = gh_chi + mp / fmax(r_base, eps);
  double e_inv = d_gig_e_inv_v(bessel_tab, chi_post, gh_psi);
  double r_eff = r_base / fmax(e_inv, eps);
  gates[i] = r_base / fmax(r_eff, r_base);
}

static int threads_for(int n) {
  int t = 256;
  if (n < 64) t = 64;
  int blocks = (n + t - 1) / t;
  return blocks;
}

}  // namespace

extern "C" {

int cypha_accel_cuda_try_init(char* name_out, int name_cap) {
  g_ok = false;
  if (name_out && name_cap > 0) name_out[0] = '\0';
  cudaError_t e = cudaSetDevice(0);
  if (e != cudaSuccess) return 0;
  cudaDeviceProp prop{};
  e = cudaGetDeviceProperties(&prop, 0);
  if (e != cudaSuccess) return 0;
  std::snprintf(g_name, sizeof(g_name), "%s (CUDA)", prop.name);
  if (name_out && name_cap > 0) std::strncpy(name_out, g_name, static_cast<size_t>(name_cap - 1));
  g_ok = true;
  return 1;
}

void cypha_accel_cuda_shutdown() {
  if (g_ok) cudaDeviceSynchronize();
  pool_clear();
  bessel_clear();
  g_ok = false;
}

int cypha_accel_cuda_ready() { return g_ok ? 1 : 0; }

const char* cypha_accel_cuda_device_name() { return g_name; }

void cypha_accel_cuda_batch_encode(const double* x, int n, int d, const double* w, double* h) {
  if (!g_ok) return;
  const std::size_t nx = static_cast<std::size_t>(n) * static_cast<std::size_t>(d);
  const std::size_t nw = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
  const std::size_t nh = nx;
  const std::size_t need = nx + nw + nh;
  if (pool_ensure(need) != cudaSuccess) return;
  double* d_x = g_pool;
  double* d_w = g_pool + nx;
  double* d_h = g_pool + nx + nw;
  cudaMemcpy(d_x, x, nx * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, nw * sizeof(double), cudaMemcpyHostToDevice);
  int total = n * d;
  int block = 256;
  int grid = (total + block - 1) / block;
  k_batch_encode<<<grid, block>>>(d_x, d_w, n, d, d_h);
  chk(cudaGetLastError(), "k_batch_encode");
  cudaMemcpy(h, d_h, nh * sizeof(double), cudaMemcpyDeviceToHost);
}

void cypha_accel_cuda_score_matrix(const double* H, int n, int d, int K, const double* mu0,
                                   const double* inv_v, const double* D, const double* D_sq,
                                   const double* u_k, const double* ctx, double* llr) {
  if (!g_ok) return;
  const std::size_t nh = static_cast<std::size_t>(n) * static_cast<std::size_t>(d);
  const std::size_t nd = static_cast<std::size_t>(d);
  const std::size_t nKd = static_cast<std::size_t>(K) * static_cast<std::size_t>(d);
  const std::size_t nK = static_cast<std::size_t>(K);
  const std::size_t nllr = static_cast<std::size_t>(n) * nK;
  std::size_t off = 0;
  const std::size_t need = nh + nd + nd + nKd + nK + nK + nK + nllr;
  if (pool_ensure(need) != cudaSuccess) return;
  double* d_H = g_pool + off;
  off += nh;
  double* d_mu = g_pool + off;
  off += nd;
  double* d_inv = g_pool + off;
  off += nd;
  double* d_D = g_pool + off;
  off += nKd;
  double* d_Dsq = g_pool + off;
  off += nK;
  double* d_uk = g_pool + off;
  off += nK;
  double* d_ctx = g_pool + off;
  off += nK;
  double* d_llr = g_pool + off;
  cudaMemcpy(d_H, H, nh * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mu, mu0, nd * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_inv, inv_v, nd * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, D, nKd * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Dsq, D_sq, nK * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_uk, u_k, nK * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ctx, ctx, nK * sizeof(double), cudaMemcpyHostToDevice);
  {
    int total = n * K;
    int block = 256;
    int grid = (total + block - 1) / block;
    k_score_matrix<<<grid, block>>>(d_H, n, d, K, d_mu, d_inv, d_D, d_Dsq, d_uk, d_ctx, d_llr);
    chk(cudaGetLastError(), "k_score_matrix");
  }
  cudaMemcpy(llr, d_llr, nllr * sizeof(double), cudaMemcpyDeviceToHost);
}

void cypha_accel_cuda_softmax_rows(const double* logits, int n, int K, double inv_T, double* probs) {
  if (!g_ok) return;
  const std::size_t nlk = static_cast<std::size_t>(n) * static_cast<std::size_t>(K);
  const std::size_t need = nlk + nlk;
  if (pool_ensure(need) != cudaSuccess) return;
  double* d_l = g_pool;
  double* d_p = g_pool + nlk;
  cudaMemcpy(d_l, logits, nlk * sizeof(double), cudaMemcpyHostToDevice);
  int block = 256;
  int grid = threads_for(n);
  k_softmax_rows<<<grid, block>>>(d_l, n, K, inv_T, d_p);
  chk(cudaGetLastError(), "k_softmax_rows");
  cudaMemcpy(probs, d_p, nlk * sizeof(double), cudaMemcpyDeviceToHost);
}

void cypha_accel_cuda_world_gate(const double* H, int n, int d, const double* psi, double chi,
                                 double* gates) {
  if (!g_ok) return;
  const std::size_t nh = static_cast<std::size_t>(n) * static_cast<std::size_t>(d);
  const std::size_t nd = static_cast<std::size_t>(d);
  const std::size_t ng = static_cast<std::size_t>(n);
  const std::size_t need = nh + nd + ng;
  if (pool_ensure(need) != cudaSuccess) return;
  double* d_H = g_pool;
  double* d_psi = g_pool + nh;
  double* d_g = g_pool + nh + nd;
  cudaMemcpy(d_H, H, nh * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_psi, psi, nd * sizeof(double), cudaMemcpyHostToDevice);
  int block = 256;
  int grid = threads_for(n);
  k_world_gate<<<grid, block>>>(d_H, n, d, d_psi, chi, d_g);
  chk(cudaGetLastError(), "k_world_gate");
  cudaMemcpy(gates, d_g, ng * sizeof(double), cudaMemcpyDeviceToHost);
}

void cypha_accel_cuda_world_gate_nig(const double* H, int n, int d, const double* mu0,
                                     const double* inv_v, double r_base, double gh_chi, double gh_psi,
                                     double* gates) {
  if (!g_ok) return;
  if (ensure_bessel_table() != cudaSuccess) return;
  const std::size_t nh = static_cast<std::size_t>(n) * static_cast<std::size_t>(d);
  const std::size_t nd = static_cast<std::size_t>(d);
  const std::size_t ng = static_cast<std::size_t>(n);
  const std::size_t need = nh + nd + nd + ng;
  if (pool_ensure(need) != cudaSuccess) return;
  double* d_H = g_pool;
  double* d_mu = g_pool + nh;
  double* d_inv = g_pool + nh + nd;
  double* d_g = g_pool + nh + nd + nd;
  cudaMemcpy(d_H, H, nh * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mu, mu0, nd * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_inv, inv_v, nd * sizeof(double), cudaMemcpyHostToDevice);
  int block = 256;
  int grid = threads_for(n);
  k_world_gate_nig<<<grid, block>>>(d_H, n, d, d_mu, d_inv, r_base, gh_chi, gh_psi, d_bessel_k2k1, d_g);
  chk(cudaGetLastError(), "k_world_gate_nig");
  cudaMemcpy(gates, d_g, ng * sizeof(double), cudaMemcpyDeviceToHost);
}

}  // extern "C"
