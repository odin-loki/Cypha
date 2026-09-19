// GPU mixer precision + estimation sweep. Diagnostic only. Not the codec.
//
// Question: how many experts, and how many bits of x and W, before the mix
// loses a sizeable amount vs a full-precision linear mixer?
//
//   nvcc -O3 -std=c++17 -arch=sm_86 hp/tools/gpu_precision_search.cu -lcublas
//        -o hp/build/gpu_precision_search.exe
//   hp/build/gpu_precision_search.exe [dump.i16] [n_experts]
//
// Frozen dump_experts logits (stride 37). Compass, not an accept gate.
// Cannot measure bit-to-bit MIXER_SKIP / stale-reuse on this dump.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_OK(expr)                                                          \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(_e));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
#define CUBLAS_OK(expr)                                                        \
    do {                                                                       \
        cublasStatus_t _s = (expr);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS %s:%d %d\n", __FILE__, __LINE__,      \
                         (int)_s);                                             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static constexpr float kScale = 256.f;
static constexpr float kLn2 = 0.693147182f;
static constexpr int kMaxD = 96;

enum Kind : int {
    K_FULL = 0,
    K_Q_X,          // quantize logits to n bits, full W
    K_Q_W,          // full logits, snap W to 2^-n
    K_Q_XW,         // both
    K_STATIC_K,     // keep univariate top-k, zero the rest
    K_DYN_K,        // keep |x| top-k this row
    K_MAG,          // zero |x| < theta
    K_EST_MEAN,     // equal-weight all
    K_EST_TOPMEAN,  // equal-weight univariate top-k
    K_EST_BEST,     // best univariate expert only
    K_NOISE         // add uniform noise of amplitude a to x
};

struct Spec {
    int kind;
    int p;  // bits, k, or theta
    float amp;
};

__device__ __forceinline__ float d_sig(float z) {
    z = fminf(fmaxf(z, -40.f), 40.f);
    return 1.f / (1.f + expf(-z));
}
__device__ __forceinline__ float d_nats(float logit, float y) {
    const float z = logit / kScale;
    return fmaxf(z, 0.f) - z * y + log1pf(expf(-fabsf(z)));
}
__device__ __forceinline__ float quant_nbit(float x, int bits) {
    if (bits >= 16) return x;
    if (bits < 1) bits = 1;
    const float levels = (float)(1 << (bits - 1));
    const float step = 2048.f / levels;
    return rintf(x / step) * step;
}
__device__ __forceinline__ float snap_pow2(float w, int bits) {
    if (bits >= 24) return w;
    const float s = (float)(1 << bits);
    return rintf(w * s) / s;
}

__global__ void k_univariate(const float* X, const float* y, int n, int d,
                             float* bits) {
    const int j = blockIdx.x;
    if (j >= d) return;
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += d_nats(X[(size_t)i * d + j], y[i]);
    __shared__ float sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) bits[j] = (sh[0] / (float)n) / kLn2;
}

__global__ void k_append_bias(const float* X, float* Xb, int n, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int j = 0; j < d; ++j) Xb[(size_t)i * (d + 1) + j] = X[(size_t)i * d + j];
    Xb[(size_t)i * (d + 1) + d] = 1.f;
}

__global__ void k_err(const float* logits, const float* y, float* err, int n,
                      int nm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (i >= n || m >= nm) return;
    err[(size_t)i * nm + m] =
        (d_sig(logits[(size_t)i * nm + m] / kScale) - y[i]) / kScale;
}

__global__ void k_ce(const float* logits, const float* y, float* ce, int n,
                     int nm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (i >= n || m >= nm) return;
    atomicAdd(&ce[m], d_nats(logits[(size_t)i * nm + m], y[i]));
}

__global__ void k_scale(float* gW, int n, float invn) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    gW[t] *= invn;
}

__global__ void k_sgd(float* W, const float* gW, float* mom, int n, float lr) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float m = 0.9f * mom[t] + gW[t];
    mom[t] = m;
    W[t] = W[t] - lr * m;
}

__global__ void k_eval_specs(const float* X, const float* y, const float* W,
                             const int* order, const Spec* specs, int n, int d,
                             int nm, float* ce) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (i >= n || m >= nm) return;
    const Spec s = specs[m];
    const float* row = X + (size_t)i * d;
    const float bias = W[d];
    float logit = 0.f;

    if (s.kind == K_EST_BEST) {
        logit = row[order[0]];
    } else if (s.kind == K_EST_MEAN) {
        float a = 0.f;
        for (int j = 0; j < d; ++j) a += row[j];
        logit = a / (float)d;
    } else if (s.kind == K_EST_TOPMEAN) {
        const int k = s.p < d ? s.p : d;
        float a = 0.f;
        for (int t = 0; t < k; ++t) a += row[order[t]];
        logit = a / (float)k;
    } else if (s.kind == K_DYN_K) {
        float xv[kMaxD];
        float av[kMaxD];
        for (int j = 0; j < d; ++j) {
            xv[j] = row[j];
            av[j] = fabsf(row[j]);
        }
        const int k = s.p < d ? s.p : d;
        for (int t = 0; t < k; ++t) {
            int best = t;
            for (int j = t + 1; j < d; ++j)
                if (av[j] > av[best]) best = j;
            float tmpa = av[t];
            av[t] = av[best];
            av[best] = tmpa;
            const float tmpx = xv[t];
            xv[t] = xv[best];
            xv[best] = tmpx;
        }
        // After partial sort, first k of xv are the kept values but we lost
        // original indices. Re-select by threshold of kth |x|.
        const float thr = av[k - 1];
        logit = bias;
        int kept = 0;
        for (int j = 0; j < d; ++j) {
            const float x = row[j];
            if (fabsf(x) >= thr && kept < k) {
                logit += W[j] * x;
                ++kept;
            }
        }
    } else {
        logit = bias;
        const int kstat = s.p;
        for (int j = 0; j < d; ++j) {
            float x = row[j];
            float w = W[j];
            if (s.kind == K_NOISE) {
                unsigned rng = 0x9E3779B9u * (unsigned)(i * 131 + j * 17 + 3);
                float u = ((int)(rng & 1023) - 512) / 512.f;
                x += s.amp * u;
            }
            if (s.kind == K_Q_X || s.kind == K_Q_XW) x = quant_nbit(x, s.p);
            if (s.kind == K_Q_W || s.kind == K_Q_XW) w = snap_pow2(w, s.p);
            if (s.kind == K_MAG && fabsf(x) < (float)s.p) continue;
            if (s.kind == K_STATIC_K) {
                int keep = 0;
                for (int t = 0; t < kstat && t < d; ++t)
                    if (order[t] == j) {
                        keep = 1;
                        break;
                    }
                if (!keep) continue;
            }
            logit += w * x;
        }
    }
    atomicAdd(&ce[m], d_nats(logit, y[i]));
}

static void* dalloc(size_t bytes) {
    void* p = nullptr;
    CUDA_OK(cudaMalloc(&p, bytes));
    CUDA_OK(cudaMemset(p, 0, bytes));
    return p;
}

static void gemm_logits(cublasHandle_t h, const float* Xb, const float* W,
                        float* logits, int n, int d1, int nm) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, nm, n, d1, &alpha, W, d1,
                          Xb, d1, &beta, logits, nm));
}
static void gemm_grad(cublasHandle_t h, const float* Xb, const float* err,
                      float* gW, int n, int d1, int nm) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, d1, nm, n, &alpha, Xb, d1,
                          err, nm, &beta, gW, d1));
}

static void train_linear(cublasHandle_t h, const float* Xb, const float* y,
                         int n, int d1, float* W, int steps, float lr,
                         const float* Xbva, const float* yva, int nva,
                         float* val_bits) {
    float* logits = (float*)dalloc((size_t)n * 4);
    float* err = (float*)dalloc((size_t)n * 4);
    float* gW = (float*)dalloc(d1 * 4);
    float* mom = (float*)dalloc(d1 * 4);
    float* ce = (float*)dalloc(4);
    dim3 block(128);
    dim3 grid((n + 127) / 128, 1);
    for (int s = 0; s < steps; ++s) {
        gemm_logits(h, Xb, W, logits, n, d1, 1);
        k_err<<<grid, block>>>(logits, y, err, n, 1);
        gemm_grad(h, Xb, err, gW, n, d1, 1);
        k_scale<<<(d1 + 255) / 256, 256>>>(gW, d1, 1.f / (float)n);
        k_sgd<<<(d1 + 255) / 256, 256>>>(W, gW, mom, d1, lr);
    }
    float* logits_va = (float*)dalloc((size_t)nva * 4);
    dim3 gridva((nva + 127) / 128, 1);
    CUDA_OK(cudaMemset(ce, 0, 4));
    gemm_logits(h, Xbva, W, logits_va, nva, d1, 1);
    k_ce<<<gridva, block>>>(logits_va, yva, ce, nva, 1);
    CUDA_OK(cudaDeviceSynchronize());
    float hce = 0.f;
    CUDA_OK(cudaMemcpy(&hce, ce, 4, cudaMemcpyDeviceToHost));
    *val_bits = (hce / (float)nva) / kLn2;
    CUDA_OK(cudaFree(logits));
    CUDA_OK(cudaFree(err));
    CUDA_OK(cudaFree(gW));
    CUDA_OK(cudaFree(mom));
    CUDA_OK(cudaFree(ce));
    CUDA_OK(cudaFree(logits_va));
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dump = argc > 1 ? argv[1] : nullptr;
    const int d = argc > 2 ? std::atoi(argv[2]) : 77;
    std::string dump_s;
    if (!dump) {
        const char* la = std::getenv("LOCALAPPDATA");
        dump_s = std::string(la ? la : ".") + "\\hp_lab\\v57.i16";
        dump = dump_s.c_str();
    }
    if (d > kMaxD) {
        std::fprintf(stderr, "n_experts %d > kMaxD %d\n", d, kMaxD);
        return 2;
    }
    std::FILE* f = std::fopen(dump, "rb");
    if (!f) {
        std::perror(dump);
        return 1;
    }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const int wrec = d + 1;
    const int n_all = (int)(bytes / (wrec * 2));
    std::vector<std::int16_t> raw((size_t)n_all * wrec);
    if (std::fread(raw.data(), 2, (size_t)n_all * wrec, f) !=
        (size_t)n_all * wrec) {
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    int ntr = 0, nva = 0;
    for (int i = 0; i < n_all; ++i) ((i % 5) == 4 ? nva : ntr)++;
    std::vector<float> Xtr((size_t)ntr * d), ytr(ntr), Xva((size_t)nva * d),
        yva(nva);
    int it = 0, iv = 0;
    for (int i = 0; i < n_all; ++i) {
        float* dst = ((i % 5) == 4) ? Xva.data() + (size_t)iv * d
                                    : Xtr.data() + (size_t)it * d;
        float* ys = ((i % 5) == 4) ? &yva[iv++] : &ytr[it++];
        for (int j = 0; j < d; ++j) dst[j] = (float)raw[(size_t)i * wrec + j];
        *ys = (float)raw[(size_t)i * wrec + d];
    }

    cublasHandle_t handle;
    CUBLAS_OK(cublasCreate(&handle));
    cudaEvent_t t0, t1;
    CUDA_OK(cudaEventCreate(&t0));
    CUDA_OK(cudaEventCreate(&t1));
    CUDA_OK(cudaEventRecord(t0));

    float *dX, *dy, *dXva, *dyva, *duni;
    CUDA_OK(cudaMalloc(&dX, Xtr.size() * 4));
    CUDA_OK(cudaMalloc(&dy, ytr.size() * 4));
    CUDA_OK(cudaMalloc(&dXva, Xva.size() * 4));
    CUDA_OK(cudaMalloc(&dyva, yva.size() * 4));
    CUDA_OK(cudaMalloc(&duni, d * 4));
    CUDA_OK(cudaMemcpy(dX, Xtr.data(), Xtr.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dy, ytr.data(), ytr.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dXva, Xva.data(), Xva.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dyva, yva.data(), yva.size() * 4, cudaMemcpyHostToDevice));

    k_univariate<<<d, 256>>>(dXva, dyva, nva, d, duni);
    std::vector<float> uni(d);
    CUDA_OK(cudaMemcpy(uni.data(), duni, d * 4, cudaMemcpyDeviceToHost));
    std::vector<int> order(d);
    for (int i = 0; i < d; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return uni[a] < uni[b]; });

    const int d1 = d + 1;
    float* dXb = (float*)dalloc((size_t)ntr * d1 * 4);
    float* dXbva = (float*)dalloc((size_t)nva * d1 * 4);
    k_append_bias<<<(ntr + 255) / 256, 256>>>(dX, dXb, ntr, d);
    k_append_bias<<<(nva + 255) / 256, 256>>>(dXva, dXbva, nva, d);
    float* dW = (float*)dalloc(d1 * 4);
    float linear_bits = 0.f;
    train_linear(handle, dXb, dy, ntr, d1, dW, 80, 2e-3f, dXbva, dyva, nva,
                 &linear_bits);

    std::vector<Spec> specs;
    auto add = [&](int kind, int p, float amp = 0.f) {
        specs.push_back(Spec{kind, p, amp});
    };
    add(K_FULL, 0);
    const int qbits[] = {16, 12, 10, 8, 6, 4, 3, 2};
    for (int b : qbits) add(K_Q_X, b);
    for (int b : qbits) add(K_Q_W, b);
    for (int b : qbits) add(K_Q_XW, b);
    const int ks[] = {1, 2, 4, 8, 12, 16, 24, 32, 48, 77};
    for (int k : ks) add(K_STATIC_K, k);
    const int dks[] = {4, 8, 12, 16, 24, 32};
    for (int k : dks) add(K_DYN_K, k);
    const int th[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    for (int t : th) add(K_MAG, t);
    add(K_EST_MEAN, 0);
    add(K_EST_TOPMEAN, 4);
    add(K_EST_TOPMEAN, 8);
    add(K_EST_TOPMEAN, 16);
    add(K_EST_BEST, 0);
    const float amps[] = {1.f, 4.f, 16.f, 64.f, 256.f};
    for (float a : amps) add(K_NOISE, 0, a);

    const int nm = (int)specs.size();
    Spec* dspecs = (Spec*)dalloc(specs.size() * sizeof(Spec));
    CUDA_OK(cudaMemcpy(dspecs, specs.data(), specs.size() * sizeof(Spec),
                       cudaMemcpyHostToDevice));
    int* dorder = (int*)dalloc(d * 4);
    CUDA_OK(cudaMemcpy(dorder, order.data(), d * 4, cudaMemcpyHostToDevice));
    float* dce = (float*)dalloc(nm * 4);
    dim3 block(128);
    dim3 grid((nva + 127) / 128, nm);
    k_eval_specs<<<grid, block>>>(dXva, dyva, dW, dorder, dspecs, nva, d, nm,
                                  dce);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> ce(nm);
    CUDA_OK(cudaMemcpy(ce.data(), dce, nm * 4, cudaMemcpyDeviceToHost));
    for (float& v : ce) v = (v / (float)nva) / kLn2;

    CUDA_OK(cudaEventRecord(t1));
    CUDA_OK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));

    auto kind_name = [](int k) -> const char* {
        switch (k) {
        case K_FULL: return "full";
        case K_Q_X: return "q_x";
        case K_Q_W: return "q_w";
        case K_Q_XW: return "q_xw";
        case K_STATIC_K: return "static_k";
        case K_DYN_K: return "dyn_k";
        case K_MAG: return "mag";
        case K_EST_MEAN: return "est_mean";
        case K_EST_TOPMEAN: return "est_topmean";
        case K_EST_BEST: return "est_best";
        case K_NOISE: return "noise";
        default: return "?";
        }
    };

    char outpath[512];
    const char* la = std::getenv("LOCALAPPDATA");
    std::snprintf(outpath, sizeof(outpath), "%s\\hp_lab\\gpu_precision_search.json",
                  la ? la : ".");
    std::FILE* jo = std::fopen(outpath, "wb");
    auto emit = [&](const char* s) {
        std::fputs(s, stdout);
        if (jo) std::fputs(s, jo);
    };

    char buf[2048];
    std::snprintf(
        buf, sizeof(buf),
        "{\n  \"backend\": \"cuda\",\n  \"device\": \"%s\",\n  \"wall_ms\": %.2f,\n"
        "  \"n_train\": %d,\n  \"n_val\": %d,\n  \"n_experts\": %d,\n"
        "  \"linear_train_val_bits\": %.6f,\n"
        "  \"oracle_best_expert_val_bits\": %.6f,\n"
        "  \"visible_loss\": 0.001,\n  \"sizeable_loss\": 0.010,\n"
        "  \"caveat\": \"Frozen v57 logits, stride 37. Offline linear mix, not "
        "the live 10-gate MixerNet. Deltas are compass, not leftover bytes. "
        "Cannot measure MIXER_SKIP on this dump.\",\n  \"runs\": [",
        prop.name, ms, ntr, nva, d, linear_bits, uni[order[0]]);
    emit(buf);
    for (int m = 0; m < nm; ++m) {
        const float delta = ce[m] - linear_bits;
        const char* call = delta < 0.001f ? "free" : (delta < 0.010f ? "visible" : "sizeable");
        std::snprintf(buf, sizeof(buf),
                      "%s{\"kind\":\"%s\",\"p\":%d,\"amp\":%.4f,\"val_bits\":%.6f,"
                      "\"delta\":%.6f,\"call\":\"%s\"}",
                      m ? "," : "", kind_name(specs[m].kind), specs[m].p,
                      specs[m].amp, ce[m], delta, call);
        emit(buf);
    }
    emit("]\n}\n");
    if (jo) std::fclose(jo);
    std::fprintf(stderr, "wrote %s (%.1f ms on %s)\n", outpath, ms, prop.name);
    cublasDestroy(handle);
    return 0;
}
