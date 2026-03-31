#pragma once
/// OpenCL GPU backend for Cypha hot-path operations.
///
/// Build with ``-DCYPHA_ENABLE_OPENCL=ON``.  When OpenCL is absent or no
/// fp64-capable device is found, every function falls back silently to the
/// CPU path in infer_cpu.cpp.
///
/// Thread safety: ``init()`` / ``shutdown()`` are NOT thread-safe.  Inference
/// calls (``batch_encode``, ``score_matrix``, ``softmax_rows``) use the same
/// command queue and ARE safe for one-thread-at-a-time use; wrap with a mutex
/// for multi-threaded call sites.

#include <string>
#include <vector>

namespace cypha {
namespace ocl {

// ---------------------------------------------------------------------------
// Device query / lifecycle
// ---------------------------------------------------------------------------

/// Initialise the OpenCL backend: enumerate platforms/devices, prefer
/// NVIDIA/AMD GPU, fall back to POCL CPU device.  Compiles kernels on first
/// call.  Returns ``true`` if a usable fp64 device was found and kernels
/// compiled successfully.  Safe to call multiple times (no-op after first
/// successful init).
bool init();

/// Returns ``true`` if ``init()`` succeeded and the backend is ready.
bool is_available();

/// Human-readable description of the chosen device (e.g. "NVIDIA GeForce
/// RTX 3090 (fp64 OK)").  Empty if ``!is_available()``.
std::string device_info();

/// Release all OpenCL resources.  ``is_available()`` returns false afterwards.
void shutdown();

// ---------------------------------------------------------------------------
// Accelerated compute
// ---------------------------------------------------------------------------

/// Dense GEMM: H (n×d) = X (n×d) @ W^T (d×d), all row-major float64.
/// Equivalent to ``batch_encode`` in ``infer_cpu.cpp``.
/// Falls back to CPU triple-loop if ``!is_available()``.
void batch_encode(const double* x_row, int n, int d,
                  const double* w_row,
                  double* h_out);

/// LLR matrix (n×K) from latent batch H (n×d).
///
///   R[i,:]    = (H[i,:] − μ₀) ⊙ inv_v
///   LLR[i,k]  = R[i,:] · D[k,:] − 0.5·D_sq[k] − u_k[k] + ctx[k]
///
/// D (K×d), D_sq (K,), u_k (K,), ctx (K,) are all row-major float64.
/// Falls back to CPU if ``!is_available()``.
void score_matrix(const double* h_row, int n, int d, int K,
                  const double* mu0, const double* inv_v,
                  const double* D_row,
                  const double* D_sq,
                  const double* u_k,
                  const double* ctx,
                  double* llr_out);

/// Row-wise softmax: probs (n×K) = softmax(logits (n×K), temperature).
/// ``temperature`` scales the logits before softmax (≥ kGenEps applied).
/// Falls back to CPU if ``!is_available()``.
void softmax_rows(const double* logits, int n, int K,
                  double temperature,
                  double* probs_out);

/// World-gate vector (GH path) for a batch of latents h (n×d).
/// gate[i] = tanh(chi * h[i,:] · (inv_v ⊙ psi_vec)) — scalar per sample.
/// ``psi_vec`` (d,) = world direction psi * inv_v (precomputed outside).
/// Falls back to CPU if ``!is_available()``.
void world_gate_batch(const double* h_row, int n, int d,
                      const double* psi_vec,
                      double chi,
                      double* gates_out);

}  // namespace ocl
}  // namespace cypha
