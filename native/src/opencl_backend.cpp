#include "cypha/opencl_backend.hpp"

#ifdef CYPHA_ENABLE_OPENCL

// Pin the API to OpenCL 3.0 so cl_version.h doesn't emit the "not defined" warning.
#ifndef CL_TARGET_OPENCL_VERSION
#  define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace cypha {
namespace ocl {

// ── OpenCL kernel sources ──────────────────────────────────────────────────
// All kernels use double precision (cl_khr_fp64 / fp64 extension).
// They are compiled at runtime so any fp64-capable device works without
// pre-compiled binaries.

static const char* kKernelSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// ── batch_encode ────────────────────────────────────────────────────────────
// H[i,j] = sum_k X[i,k] * W[j,k]   (H = X @ W^T, W stored row-major)
// Grid: (N, d) work-items
__kernel void batch_encode(
    __global const double* X,   // N*d
    __global const double* W,   // d*d  (row-major: W[j,k] = W[j*d+k])
    int d,
    __global double* H          // N*d
) {
    int i = get_global_id(0);   // sample index
    int j = get_global_id(1);   // output feature index
    double acc = 0.0;
    int base_x = i * d;
    int base_w = j * d;
    for (int k = 0; k < d; k++) {
        acc += X[base_x + k] * W[base_w + k];
    }
    H[i * d + j] = acc;
}

// ── score_matrix ─────────────────────────────────────────────────────────────
// LLR[i,k] = sum_j (H[i,j]-mu0[j])*inv_v[j]*D[k,j] - 0.5*D_sq[k] - u_k[k] + ctx[k]
// Grid: (N, K) work-items
__kernel void score_matrix(
    __global const double* H,       // N*d
    int N, int d, int K,
    __global const double* mu0,     // d
    __global const double* inv_v,   // d
    __global const double* D,       // K*d (row-major)
    __global const double* D_sq,    // K
    __global const double* u_k,     // K
    __global const double* ctx,     // K
    __global double* LLR            // N*K
) {
    int i = get_global_id(0);   // sample
    int k = get_global_id(1);   // class
    if (i >= N || k >= K) return;
    double cross = 0.0;
    int base_h = i * d;
    int base_d = k * d;
    for (int j = 0; j < d; j++) {
        cross += (H[base_h + j] - mu0[j]) * inv_v[j] * D[base_d + j];
    }
    LLR[i * K + k] = cross - 0.5 * D_sq[k] - u_k[k] + ctx[k];
}

// ── softmax_rows ─────────────────────────────────────────────────────────────
// probs[i,k] = exp((logits[i,k]-max_i)/T) / sum_k
// One work-item per row (serial loop over K columns for correctness).
// Grid: (N,) global work-items.
__kernel void softmax_rows(
    __global const double* logits,  // N*K
    int N, int K,
    double inv_temperature,         // 1/T (already clamped to safe value)
    __global double* probs          // N*K
) {
    int i = get_global_id(0);
    if (i >= N) return;
    __global const double* row = logits + i * K;
    __global double*       out = probs  + i * K;

    // Find max for numerical stability
    double mx = row[0];
    for (int k = 1; k < K; k++) {
        if (row[k] > mx) mx = row[k];
    }
    // Compute exp and sum
    double s = 0.0;
    for (int k = 0; k < K; k++) {
        double e = exp((row[k] - mx) * inv_temperature);
        out[k] = e;
        s += e;
    }
    s = max(s, 1e-300);
    for (int k = 0; k < K; k++) {
        out[k] /= s;
    }
}

// ── world_gate_batch ─────────────────────────────────────────────────────────
// gate[i] = tanh(chi * dot(h[i,:], psi_vec))
// Grid: (N,) work-items
__kernel void world_gate_batch(
    __global const double* H,       // N*d
    int d,
    __global const double* psi_vec, // d  (= world direction * inv_v, precomputed)
    double chi,
    __global double* gates           // N
) {
    int i = get_global_id(0);
    double dot = 0.0;
    int base = i * d;
    for (int j = 0; j < d; j++) {
        dot += H[base + j] * psi_vec[j];
    }
    gates[i] = tanh(chi * dot);
}
)CL";

// ── Backend state ──────────────────────────────────────────────────────────

namespace {

struct OclState {
    cl_platform_id   platform{nullptr};
    cl_device_id     device{nullptr};
    cl_context       ctx{nullptr};
    cl_command_queue queue{nullptr};
    cl_program       program{nullptr};
    cl_kernel        k_batch_encode{nullptr};
    cl_kernel        k_score_matrix{nullptr};
    cl_kernel        k_softmax_rows{nullptr};
    cl_kernel        k_world_gate{nullptr};
    std::string      dev_info;
    bool             ready{false};
};

static OclState g_ocl;
static std::mutex g_mutex;

static void ocl_check(cl_int err, const char* where) {
    if (err != CL_SUCCESS) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "OpenCL error %d at %s", (int)err, where);
        throw std::runtime_error(msg);
    }
}

static bool device_has_fp64(cl_device_id dev) {
    char ext[4096] = {};
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(ext) - 1, ext, nullptr);
    return std::strstr(ext, "cl_khr_fp64") != nullptr ||
           std::strstr(ext, "cl_amd_fp64") != nullptr;
}

/// Score a device: prefer GPU, then NVIDIA/AMD > Intel > POCL.
static int device_score(cl_device_id dev) {
    cl_device_type dtype = CL_DEVICE_TYPE_CPU;
    clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
    char vendor[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor) - 1, vendor, nullptr);
    std::string v(vendor);
    int score = 0;
    if (dtype == CL_DEVICE_TYPE_GPU) score += 1000;
    if (v.find("NVIDIA") != std::string::npos) score += 100;
    if (v.find("AMD") != std::string::npos || v.find("Advanced Micro") != std::string::npos) score += 80;
    if (v.find("Intel") != std::string::npos) score += 50;
    return score;
}

}  // namespace

// ── Public API ─────────────────────────────────────────────────────────────

bool init() {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_ocl.ready) return true;

    // Enumerate all platforms and devices
    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0)
        return false;

    std::vector<cl_platform_id> platforms(n_platforms);
    clGetPlatformIDs(n_platforms, platforms.data(), nullptr);

    cl_device_id best_dev = nullptr;
    cl_platform_id best_plat = nullptr;
    int best_score = -1;

    for (auto plat : platforms) {
        cl_uint n_devs = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &n_devs) != CL_SUCCESS)
            continue;
        std::vector<cl_device_id> devs(n_devs);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, n_devs, devs.data(), nullptr);
        for (auto dev : devs) {
            if (!device_has_fp64(dev)) continue;
            int s = device_score(dev);
            if (s > best_score) {
                best_score = s;
                best_dev = dev;
                best_plat = plat;
            }
        }
    }
    if (!best_dev) return false;

    // Build device info string
    {
        char name[256] = {};
        clGetDeviceInfo(best_dev, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
        g_ocl.dev_info = std::string(name) + " (fp64 OK)";
    }

    cl_int err;
    g_ocl.platform = best_plat;
    g_ocl.device   = best_dev;
    g_ocl.ctx      = clCreateContext(nullptr, 1, &best_dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;

#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {0};
    g_ocl.queue = clCreateCommandQueueWithProperties(g_ocl.ctx, best_dev, props, &err);
#else
    g_ocl.queue = clCreateCommandQueue(g_ocl.ctx, best_dev, 0, &err);
#endif
    if (err != CL_SUCCESS) { clReleaseContext(g_ocl.ctx); return false; }

    // Compile kernels
    const char* src = kKernelSrc;
    g_ocl.program = clCreateProgramWithSource(g_ocl.ctx, 1, &src, nullptr, &err);
    if (err != CL_SUCCESS) goto fail;

    err = clBuildProgram(g_ocl.program, 1, &best_dev, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Print build log for diagnosis
        std::size_t logsize = 0;
        clGetProgramBuildInfo(g_ocl.program, best_dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsize);
        std::vector<char> log(logsize + 1, '\0');
        clGetProgramBuildInfo(g_ocl.program, best_dev, CL_PROGRAM_BUILD_LOG, logsize, log.data(), nullptr);
        std::fprintf(stderr, "[cypha_ocl] kernel build error:\n%s\n", log.data());
        goto fail;
    }

    g_ocl.k_batch_encode  = clCreateKernel(g_ocl.program, "batch_encode",    &err); if (err != CL_SUCCESS) goto fail;
    g_ocl.k_score_matrix  = clCreateKernel(g_ocl.program, "score_matrix",    &err); if (err != CL_SUCCESS) goto fail;
    g_ocl.k_softmax_rows  = clCreateKernel(g_ocl.program, "softmax_rows",    &err); if (err != CL_SUCCESS) goto fail;
    g_ocl.k_world_gate    = clCreateKernel(g_ocl.program, "world_gate_batch",&err); if (err != CL_SUCCESS) goto fail;

    g_ocl.ready = true;
    return true;

fail:
    if (g_ocl.k_batch_encode) { clReleaseKernel(g_ocl.k_batch_encode); g_ocl.k_batch_encode = nullptr; }
    if (g_ocl.k_score_matrix) { clReleaseKernel(g_ocl.k_score_matrix); g_ocl.k_score_matrix = nullptr; }
    if (g_ocl.k_softmax_rows) { clReleaseKernel(g_ocl.k_softmax_rows); g_ocl.k_softmax_rows = nullptr; }
    if (g_ocl.k_world_gate)   { clReleaseKernel(g_ocl.k_world_gate);   g_ocl.k_world_gate   = nullptr; }
    if (g_ocl.program) { clReleaseProgram(g_ocl.program); g_ocl.program = nullptr; }
    if (g_ocl.queue)   { clReleaseCommandQueue(g_ocl.queue); g_ocl.queue = nullptr; }
    if (g_ocl.ctx)     { clReleaseContext(g_ocl.ctx);       g_ocl.ctx   = nullptr; }
    return false;
}

bool is_available() {
    std::lock_guard<std::mutex> lk(g_mutex);
    return g_ocl.ready;
}

std::string device_info() {
    std::lock_guard<std::mutex> lk(g_mutex);
    return g_ocl.dev_info;
}

void shutdown() {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_ocl.k_batch_encode) clReleaseKernel(g_ocl.k_batch_encode);
    if (g_ocl.k_score_matrix) clReleaseKernel(g_ocl.k_score_matrix);
    if (g_ocl.k_softmax_rows) clReleaseKernel(g_ocl.k_softmax_rows);
    if (g_ocl.k_world_gate)   clReleaseKernel(g_ocl.k_world_gate);
    if (g_ocl.program)        clReleaseProgram(g_ocl.program);
    if (g_ocl.queue)          clReleaseCommandQueue(g_ocl.queue);
    if (g_ocl.ctx)            clReleaseContext(g_ocl.ctx);
    g_ocl = OclState{};
}

// ── Helper: allocate, upload, dispatch, download, free ────────────────────

namespace {

// Simple RAII wrapper for cl_mem so we don't leak on early return.
struct ClMem {
    cl_mem buf{nullptr};
    explicit ClMem(cl_mem b) : buf(b) {}
    ~ClMem() { if (buf) clReleaseMemObject(buf); }
    ClMem(const ClMem&) = delete;
    ClMem& operator=(const ClMem&) = delete;
    cl_mem get() const { return buf; }
};

ClMem make_ro(cl_context ctx, const double* data, std::size_t n_elem) {
    cl_int err;
    auto b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            n_elem * sizeof(double),
                            const_cast<double*>(data), &err);
    ocl_check(err, "make_ro");
    return ClMem(b);
}

ClMem make_wo(cl_context ctx, std::size_t n_elem) {
    cl_int err;
    auto b = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                            n_elem * sizeof(double), nullptr, &err);
    ocl_check(err, "make_wo");
    return ClMem(b);
}

void read_buf(cl_command_queue q, cl_mem buf, double* dst, std::size_t n_elem) {
    ocl_check(clEnqueueReadBuffer(q, buf, CL_TRUE, 0,
                                  n_elem * sizeof(double), dst, 0, nullptr, nullptr),
              "read_buf");
}

}  // namespace

// ── batch_encode ─────────────────────────────────────────────────────────────

// CPU fallback used by all paths when OpenCL unavailable.
namespace {
void cpu_batch_encode(const double* x, int n, int d, const double* w, double* h) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) {
            double acc = 0.0;
            for (int k = 0; k < d; k++) acc += x[i*d+k] * w[j*d+k];
            h[i*d+j] = acc;
        }
}
void cpu_score_matrix(const double* H, int n, int d, int K,
                      const double* mu0, const double* inv_v,
                      const double* D, const double* D_sq,
                      const double* u_k, const double* ctx,
                      double* llr) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++) {
            double cross = 0.0;
            for (int j = 0; j < d; j++)
                cross += (H[i*d+j] - mu0[j]) * inv_v[j] * D[k*d+j];
            llr[i*K+k] = cross - 0.5*D_sq[k] - u_k[k] + ctx[k];
        }
}
void cpu_softmax_rows(const double* logits, int n, int K, double inv_T, double* probs) {
    for (int i = 0; i < n; i++) {
        const double* row = logits + i*K;
        double mx = row[0];
        for (int k = 1; k < K; k++) if (row[k] > mx) mx = row[k];
        double s = 0.0;
        double* out = probs + i*K;
        for (int k = 0; k < K; k++) { out[k] = std::exp((row[k]-mx)*inv_T); s += out[k]; }
        s = std::max(s, 1e-300);
        for (int k = 0; k < K; k++) out[k] /= s;
    }
}
void cpu_world_gate(const double* H, int n, int d, const double* psi, double chi, double* g) {
    for (int i = 0; i < n; i++) {
        double dot = 0.0;
        for (int j = 0; j < d; j++) dot += H[i*d+j] * psi[j];
        g[i] = std::tanh(chi * dot);
    }
}
}  // namespace

void batch_encode(const double* x_row, int n, int d,
                  const double* w_row,
                  double* h_out) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_ocl.ready) {
        cpu_batch_encode(x_row, n, d, w_row, h_out);
        return;
    }
    try {
        auto bX = make_ro(g_ocl.ctx, x_row,  static_cast<std::size_t>(n * d));
        auto bW = make_ro(g_ocl.ctx, w_row,  static_cast<std::size_t>(d * d));
        auto bH = make_wo(g_ocl.ctx,         static_cast<std::size_t>(n * d));
        cl_mem mx = bX.get(), mw = bW.get(), mh = bH.get();
        cl_int di = d;
        ocl_check(clSetKernelArg(g_ocl.k_batch_encode, 0, sizeof(cl_mem), &mx), "ke arg0");
        ocl_check(clSetKernelArg(g_ocl.k_batch_encode, 1, sizeof(cl_mem), &mw), "ke arg1");
        ocl_check(clSetKernelArg(g_ocl.k_batch_encode, 2, sizeof(cl_int), &di), "ke arg2");
        ocl_check(clSetKernelArg(g_ocl.k_batch_encode, 3, sizeof(cl_mem), &mh), "ke arg3");
        std::size_t gws[2] = {static_cast<std::size_t>(n), static_cast<std::size_t>(d)};
        ocl_check(clEnqueueNDRangeKernel(g_ocl.queue, g_ocl.k_batch_encode,
                                         2, nullptr, gws, nullptr, 0, nullptr, nullptr),
                  "ke enqueue");
        read_buf(g_ocl.queue, bH.get(), h_out, static_cast<std::size_t>(n * d));
    } catch (...) {
        cpu_batch_encode(x_row, n, d, w_row, h_out);
    }
}

void score_matrix(const double* h_row, int n, int d, int K,
                  const double* mu0, const double* inv_v,
                  const double* D_row,
                  const double* D_sq,
                  const double* u_k,
                  const double* ctx,
                  double* llr_out) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_ocl.ready) {
        cpu_score_matrix(h_row, n, d, K, mu0, inv_v, D_row, D_sq, u_k, ctx, llr_out);
        return;
    }
    try {
        auto bH   = make_ro(g_ocl.ctx, h_row,  static_cast<std::size_t>(n * d));
        auto bMu  = make_ro(g_ocl.ctx, mu0,    static_cast<std::size_t>(d));
        auto bInv = make_ro(g_ocl.ctx, inv_v,  static_cast<std::size_t>(d));
        auto bD   = make_ro(g_ocl.ctx, D_row,  static_cast<std::size_t>(K * d));
        auto bDsq = make_ro(g_ocl.ctx, D_sq,   static_cast<std::size_t>(K));
        auto bUk  = make_ro(g_ocl.ctx, u_k,    static_cast<std::size_t>(K));
        auto bCtx = make_ro(g_ocl.ctx, ctx,    static_cast<std::size_t>(K));
        auto bLLR = make_wo(g_ocl.ctx,         static_cast<std::size_t>(n * K));

        cl_int cn = n, cd = d, cK = K;
        cl_mem mh   = bH.get(),  mmu = bMu.get(), minv = bInv.get();
        cl_mem md   = bD.get(),  mds = bDsq.get(), muk = bUk.get();
        cl_mem mctx = bCtx.get(), mllr = bLLR.get();
        auto& kk = g_ocl.k_score_matrix;
        ocl_check(clSetKernelArg(kk, 0, sizeof(cl_mem), &mh),   "sm 0");
        ocl_check(clSetKernelArg(kk, 1, sizeof(cl_int), &cn),   "sm 1");
        ocl_check(clSetKernelArg(kk, 2, sizeof(cl_int), &cd),   "sm 2");
        ocl_check(clSetKernelArg(kk, 3, sizeof(cl_int), &cK),   "sm 3");
        ocl_check(clSetKernelArg(kk, 4, sizeof(cl_mem), &mmu),  "sm 4");
        ocl_check(clSetKernelArg(kk, 5, sizeof(cl_mem), &minv), "sm 5");
        ocl_check(clSetKernelArg(kk, 6, sizeof(cl_mem), &md),   "sm 6");
        ocl_check(clSetKernelArg(kk, 7, sizeof(cl_mem), &mds),  "sm 7");
        ocl_check(clSetKernelArg(kk, 8, sizeof(cl_mem), &muk),  "sm 8");
        ocl_check(clSetKernelArg(kk, 9, sizeof(cl_mem), &mctx), "sm 9");
        ocl_check(clSetKernelArg(kk,10, sizeof(cl_mem), &mllr), "sm 10");
        std::size_t gws[2] = {static_cast<std::size_t>(n), static_cast<std::size_t>(K)};
        ocl_check(clEnqueueNDRangeKernel(g_ocl.queue, kk, 2, nullptr, gws,
                                         nullptr, 0, nullptr, nullptr), "sm enqueue");
        read_buf(g_ocl.queue, bLLR.get(), llr_out, static_cast<std::size_t>(n * K));
    } catch (...) {
        cpu_score_matrix(h_row, n, d, K, mu0, inv_v, D_row, D_sq, u_k, ctx, llr_out);
    }
}

void softmax_rows(const double* logits, int n, int K,
                  double temperature,
                  double* probs_out) {
    double inv_T = 1.0 / std::max(temperature, 1e-8);
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_ocl.ready) {
        cpu_softmax_rows(logits, n, K, inv_T, probs_out);
        return;
    }
    try {
        auto bL = make_ro(g_ocl.ctx, logits,   static_cast<std::size_t>(n * K));
        auto bP = make_wo(g_ocl.ctx,           static_cast<std::size_t>(n * K));
        cl_int cn = n, cK = K;
        cl_double cInvT = static_cast<cl_double>(inv_T);
        cl_mem ml = bL.get(), mp = bP.get();
        auto& kk = g_ocl.k_softmax_rows;
        ocl_check(clSetKernelArg(kk, 0, sizeof(cl_mem),    &ml),    "sf 0");
        ocl_check(clSetKernelArg(kk, 1, sizeof(cl_int),    &cn),    "sf 1");
        ocl_check(clSetKernelArg(kk, 2, sizeof(cl_int),    &cK),    "sf 2");
        ocl_check(clSetKernelArg(kk, 3, sizeof(cl_double), &cInvT), "sf 3");
        ocl_check(clSetKernelArg(kk, 4, sizeof(cl_mem),    &mp),    "sf 4");
        std::size_t gws[1] = {static_cast<std::size_t>(n)};
        ocl_check(clEnqueueNDRangeKernel(g_ocl.queue, kk, 1, nullptr, gws,
                                         nullptr, 0, nullptr, nullptr), "sf enqueue");
        read_buf(g_ocl.queue, bP.get(), probs_out, static_cast<std::size_t>(n * K));
    } catch (...) {
        cpu_softmax_rows(logits, n, K, inv_T, probs_out);
    }
}

void world_gate_batch(const double* h_row, int n, int d,
                      const double* psi_vec,
                      double chi,
                      double* gates_out) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_ocl.ready) {
        cpu_world_gate(h_row, n, d, psi_vec, chi, gates_out);
        return;
    }
    try {
        auto bH   = make_ro(g_ocl.ctx, h_row,   static_cast<std::size_t>(n * d));
        auto bPsi = make_ro(g_ocl.ctx, psi_vec, static_cast<std::size_t>(d));
        auto bG   = make_wo(g_ocl.ctx,          static_cast<std::size_t>(n));
        cl_int cd = d;
        cl_double cchi = static_cast<cl_double>(chi);
        cl_mem mh = bH.get(), mp = bPsi.get(), mg = bG.get();
        auto& kk = g_ocl.k_world_gate;
        ocl_check(clSetKernelArg(kk, 0, sizeof(cl_mem),    &mh),   "wg 0");
        ocl_check(clSetKernelArg(kk, 1, sizeof(cl_int),    &cd),   "wg 1");
        ocl_check(clSetKernelArg(kk, 2, sizeof(cl_mem),    &mp),   "wg 2");
        ocl_check(clSetKernelArg(kk, 3, sizeof(cl_double), &cchi), "wg 3");
        ocl_check(clSetKernelArg(kk, 4, sizeof(cl_mem),    &mg),   "wg 4");
        std::size_t gws[1] = {static_cast<std::size_t>(n)};
        ocl_check(clEnqueueNDRangeKernel(g_ocl.queue, kk, 1, nullptr, gws,
                                         nullptr, 0, nullptr, nullptr), "wg enqueue");
        read_buf(g_ocl.queue, bG.get(), gates_out, static_cast<std::size_t>(n));
    } catch (...) {
        cpu_world_gate(h_row, n, d, psi_vec, chi, gates_out);
    }
}

}  // namespace ocl
}  // namespace cypha

#else  // CYPHA_ENABLE_OPENCL not defined — stub out everything

#include <cmath>
#include <string>

namespace cypha {
namespace ocl {

bool        init()        { return false; }
bool        is_available(){ return false; }
std::string device_info() { return "OpenCL not compiled in (build with -DCYPHA_ENABLE_OPENCL=ON)"; }
void        shutdown()    {}

static void cpu_be(const double* x, int n, int d, const double* w, double* h) {
    for (int i=0;i<n;i++) for (int j=0;j<d;j++) {
        double a=0; for (int k=0;k<d;k++) a+=x[i*d+k]*w[j*d+k]; h[i*d+j]=a; } }
static void cpu_sm(const double* H, int n, int d, int K, const double* mu0, const double* inv_v,
                   const double* D, const double* Dsq, const double* uk, const double* ctx, double* llr) {
    for (int i=0;i<n;i++) for (int k=0;k<K;k++) {
        double c=0; for (int j=0;j<d;j++) c+=(H[i*d+j]-mu0[j])*inv_v[j]*D[k*d+j];
        llr[i*K+k]=c-0.5*Dsq[k]-uk[k]+ctx[k]; } }
static void cpu_sf(const double* l, int n, int K, double iT, double* p) {
    for (int i=0;i<n;i++){
        const double* r=l+i*K; double mx=r[0]; for(int k=1;k<K;k++) if(r[k]>mx) mx=r[k];
        double s=0; double* o=p+i*K; for(int k=0;k<K;k++){o[k]=std::exp((r[k]-mx)*iT);s+=o[k];}
        s=std::max(s,1e-300); for(int k=0;k<K;k++) o[k]/=s; } }

void batch_encode(const double* x, int n, int d, const double* w, double* h) { cpu_be(x,n,d,w,h); }
void score_matrix(const double* H, int n, int d, int K, const double* mu0, const double* inv_v,
                  const double* D, const double* Dsq, const double* uk, const double* ctx, double* llr) {
    cpu_sm(H,n,d,K,mu0,inv_v,D,Dsq,uk,ctx,llr); }
void softmax_rows(const double* l, int n, int K, double T, double* p) {
    cpu_sf(l,n,K,1.0/std::max(T,1e-8),p); }
void world_gate_batch(const double* H, int n, int d, const double* psi, double chi, double* g) {
    for(int i=0;i<n;i++){double dot=0;for(int j=0;j<d;j++)dot+=H[i*d+j]*psi[j];g[i]=std::tanh(chi*dot);} }

}  // namespace ocl
}  // namespace cypha

#endif  // CYPHA_ENABLE_OPENCL
