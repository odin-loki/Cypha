// GPU cluster-MoE compass. Diagnostic only. Not the codec.
//
// Cluster 77 frozen experts, mix inside each cluster (real W·x), squash,
// then mix cluster opinions with uneven v. Does not replace the mix with
// a mean. Hard MoE still mixes inside the chosen cluster.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 hp/tools/gpu_moe_cluster.cu -lcublas
//        -o hp/build/gpu_moe_cluster.exe
//   hp/build/gpu_moe_cluster.exe [dump.i16] [n_experts]

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
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
static constexpr int kMaxK = 32;
static constexpr int kMaxD = 96;

__device__ __forceinline__ float d_sig(float z) {
    z = fminf(fmaxf(z, -40.f), 40.f);
    return 1.f / (1.f + expf(-z));
}
__device__ __forceinline__ float d_nats(float logit, float y) {
    const float z = logit / kScale;
    return fmaxf(z, 0.f) - z * y + log1pf(expf(-fabsf(z)));
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

__global__ void k_scale(float* g, int n, float invn) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    g[t] *= invn;
}

__global__ void k_sgd(float* W, const float* g, float* mom, int n, float lr) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float m = 0.9f * mom[t] + g[t];
    mom[t] = m;
    W[t] = W[t] - lr * m;
}

__global__ void k_mask_w(float* W, const float* mask, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    if (mask) W[t] *= mask[t];
}

__global__ void k_q_xw(float* Xb, float* W, int n, int d1, int xbits, int wbits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float xstep = 2048.f / (float)(1 << (xbits - 1));
        for (int j = 0; j < d1 - 1; ++j) {
            float x = Xb[(size_t)i * d1 + j];
            Xb[(size_t)i * d1 + j] = rintf(x / xstep) * xstep;
        }
    }
    if (i < d1) {
        const float s = (float)(1 << wbits);
        W[i] = rintf(W[i] * s) / s;
    }
}

// Hierarchical: z_c = stretch(squash(w_c · x_c)) = kScale*(2*sig(dot/kScale)-1)
// logit = v · z + b
// assign[j] in 0..K-1
__global__ void k_moe_train(const float* X, const float* y, const int* assign,
                            float* W, float* b1, float* v, float* b2, float* mW,
                            float* mb1, float* mv, float* mb2, int n, int d,
                            int K, float lr, int hard_top, float* ce_out) {
    // One block per row. Not the 3090's favorite, but n=64k * K<=32 is tiny.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = X + (size_t)i * d;
    float z[kMaxK];
    float dots[kMaxK];
    float energy[kMaxK];
    for (int c = 0; c < K; ++c) {
        float dot = b1[c];
        float en = 0.f;
        for (int j = 0; j < d; ++j) {
            if (assign[j] != c) continue;
            dot += W[j] * row[j];
            en += row[j] * row[j];
        }
        dots[c] = dot;
        energy[c] = en;
        const float s = d_sig(dot / kScale);
        z[c] = kScale * (2.f * s - 1.f);
    }
    int use[kMaxK];
    for (int c = 0; c < K; ++c) use[c] = 1;
    if (hard_top > 0 && hard_top < K) {
        for (int c = 0; c < K; ++c) use[c] = 0;
        for (int t = 0; t < hard_top; ++t) {
            int best = -1;
            float be = -1.f;
            for (int c = 0; c < K; ++c) {
                if (use[c]) continue;
                if (energy[c] > be) {
                    be = energy[c];
                    best = c;
                }
            }
            if (best >= 0) use[best] = 1;
        }
    }
    float logit = b2[0];
    float vsum = 0.f;
    for (int c = 0; c < K; ++c)
        if (use[c]) {
            logit += v[c] * z[c];
            vsum += 1.f;
        }
    (void)vsum;
    const float p = d_sig(logit / kScale);
    const float dlogit = (p - y[i]) / kScale;
    atomicAdd(ce_out, d_nats(logit, y[i]));

    if (lr == 0.f) return;
    atomicAdd(b2, -lr * dlogit);
    atomicAdd(mb2, dlogit);
    for (int c = 0; c < K; ++c) {
        if (!use[c]) continue;
        atomicAdd(&v[c], -lr * dlogit * z[c]);
        atomicAdd(&mv[c], dlogit * z[c]);
        const float s = d_sig(dots[c] / kScale);
        const float dz_ddot = 2.f * s * (1.f - s);
        const float ddot = dlogit * v[c] * dz_ddot;
        atomicAdd(&b1[c], -lr * ddot);
        atomicAdd(&mb1[c], ddot);
        for (int j = 0; j < d; ++j) {
            if (assign[j] != c) continue;
            const float g = ddot * row[j];
            atomicAdd(&W[j], -lr * g);
            atomicAdd(&mW[j], g);
        }
    }
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

static float train_linear(cublasHandle_t h, const float* Xb, const float* y,
                          int n, int d1, float* W, const float* mask, int steps,
                          float lr, const float* Xbva, const float* yva, int nva) {
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
        if (mask) k_mask_w<<<(d1 + 255) / 256, 256>>>(W, mask, d1);
    }
    float* logits_va = (float*)dalloc((size_t)nva * 4);
    dim3 gridva((nva + 127) / 128, 1);
    CUDA_OK(cudaMemset(ce, 0, 4));
    gemm_logits(h, Xbva, W, logits_va, nva, d1, 1);
    k_ce<<<gridva, block>>>(logits_va, yva, ce, nva, 1);
    CUDA_OK(cudaDeviceSynchronize());
    float hce = 0.f;
    CUDA_OK(cudaMemcpy(&hce, ce, 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(logits));
    CUDA_OK(cudaFree(err));
    CUDA_OK(cudaFree(gW));
    CUDA_OK(cudaFree(mom));
    CUDA_OK(cudaFree(ce));
    CUDA_OK(cudaFree(logits_va));
    return (hce / (float)nva) / kLn2;
}

static std::vector<int> kmeans_experts(const float* X, int n, int d, int K,
                                       int seed) {
    std::vector<double> mean(d, 0), var(d, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j) mean[j] += X[(size_t)i * d + j];
    for (int j = 0; j < d; ++j) mean[j] /= n;
    std::vector<float> Z((size_t)d * n);
    for (int j = 0; j < d; ++j) {
        double v = 0;
        for (int i = 0; i < n; ++i) {
            const double z = X[(size_t)i * d + j] - mean[j];
            Z[(size_t)j * n + i] = (float)z;
            v += z * z;
        }
        var[j] = v > 0 ? std::sqrt(v) : 1.0;
        const float inv = 1.f / (float)var[j];
        for (int i = 0; i < n; ++i) Z[(size_t)j * n + i] *= inv;
    }
    auto dist2 = [&](int a, int b) {
        double s = 0;
        const float* pa = Z.data() + (size_t)a * n;
        const float* pb = Z.data() + (size_t)b * n;
        for (int i = 0; i < n; ++i) {
            const double t = (double)pa[i] - (double)pb[i];
            s += t * t;
        }
        return s;
    };
    std::mt19937 rng(seed);
    std::vector<int> cent;
    cent.push_back(0);
    std::vector<double> nearest(d, 1e100);
    for (int k = 1; k < K; ++k) {
        double tot = 0;
        for (int j = 0; j < d; ++j) {
            double best = 1e100;
            for (int c : cent) best = std::min(best, dist2(j, c));
            nearest[j] = best;
            tot += best;
        }
        std::uniform_real_distribution<double> u(0, tot);
        double r = u(rng);
        int pick = d - 1;
        for (int j = 0; j < d; ++j) {
            r -= nearest[j];
            if (r <= 0) {
                pick = j;
                break;
            }
        }
        cent.push_back(pick);
    }
    std::vector<int> assign(d, 0);
    for (int it = 0; it < 25; ++it) {
        for (int j = 0; j < d; ++j) {
            int best = 0;
            double bd = 1e100;
            for (int c = 0; c < K; ++c) {
                const double dd = dist2(j, cent[c]);
                if (dd < bd) {
                    bd = dd;
                    best = c;
                }
            }
            assign[j] = best;
        }
        std::vector<int> count(K, 0);
        std::vector<double> acc((size_t)K * n, 0);
        for (int j = 0; j < d; ++j) {
            const int c = assign[j];
            count[c]++;
            const float* pj = Z.data() + (size_t)j * n;
            for (int i = 0; i < n; ++i) acc[(size_t)c * n + i] += pj[i];
        }
        for (int c = 0; c < K; ++c) {
            if (!count[c]) continue;
            const double inv = 1.0 / count[c];
            for (int i = 0; i < n; ++i) acc[(size_t)c * n + i] *= inv;
        }
        for (int c = 0; c < K; ++c) {
            if (!count[c]) continue;
            int best = 0;
            double bd = 1e100;
            for (int j = 0; j < d; ++j) {
                if (assign[j] != c) continue;
                double s = 0;
                const float* pj = Z.data() + (size_t)j * n;
                for (int i = 0; i < n; ++i) {
                    const double t = pj[i] - acc[(size_t)c * n + i];
                    s += t * t;
                }
                if (s < bd) {
                    bd = s;
                    best = j;
                }
            }
            cent[c] = best;
        }
    }
    return assign;
}

static float eval_moe(const float* dX, const float* dy, const int* dassign,
                      float* dW, float* db1, float* dv, float* db2, int n, int d,
                      int K, int hard_top) {
    float* dce = (float*)dalloc(4);
    k_moe_train<<<(n + 127) / 128, 128>>>(dX, dy, dassign, dW, db1, dv, db2,
                                          nullptr, nullptr, nullptr, nullptr, n,
                                          d, K, 0.f, hard_top, dce);
    CUDA_OK(cudaDeviceSynchronize());
    float hce = 0.f;
    CUDA_OK(cudaMemcpy(&hce, dce, 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(dce));
    return (hce / (float)n) / kLn2;
}

static void train_moe(const float* dX, const float* dy, const int* dassign,
                      int n, int d, int K, int steps, float lr, int hard_top,
                      float* dW, float* db1, float* dv, float* db2) {
    CUDA_OK(cudaMemset(db1, 0, K * 4));
    CUDA_OK(cudaMemset(db2, 0, 4));
    const float w0 = 1.f / (float)d;
    const float v0 = 1.f / (float)K;
    CUDA_OK(cudaMemcpy(dW, std::vector<float>(d, w0).data(), d * 4,
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dv, std::vector<float>(K, v0).data(), K * 4,
                       cudaMemcpyHostToDevice));
    float* mW = (float*)dalloc(d * 4);
    float* mb1 = (float*)dalloc(K * 4);
    float* mv = (float*)dalloc(K * 4);
    float* mb2 = (float*)dalloc(4);
    float* dce = (float*)dalloc(4);
    const float step = lr / (float)n;
    for (int s = 0; s < steps; ++s) {
        CUDA_OK(cudaMemset(dce, 0, 4));
        k_moe_train<<<(n + 127) / 128, 128>>>(dX, dy, dassign, dW, db1, dv, db2,
                                              mW, mb1, mv, mb2, n, d, K, step,
                                              hard_top, dce);
    }
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaFree(mW));
    CUDA_OK(cudaFree(mb1));
    CUDA_OK(cudaFree(mv));
    CUDA_OK(cudaFree(mb2));
    CUDA_OK(cudaFree(dce));
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
        std::fprintf(stderr, "n_experts %d > kMaxD\n", d);
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
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return uni[a] < uni[b]; });

    const int d1 = d + 1;
    float* dXb = (float*)dalloc((size_t)ntr * d1 * 4);
    float* dXbva = (float*)dalloc((size_t)nva * d1 * 4);
    k_append_bias<<<(ntr + 255) / 256, 256>>>(dX, dXb, ntr, d);
    k_append_bias<<<(nva + 255) / 256, 256>>>(dXva, dXbva, nva, d);

    float* dWlin = (float*)dalloc(d1 * 4);
    const float lin = train_linear(handle, dXb, dy, ntr, d1, dWlin, nullptr, 80,
                                   2e-3f, dXbva, dyva, nva);

    auto prefix_bits = [&](int k) {
        std::vector<float> mask(d1, 0.f);
        for (int t = 0; t < k && t < d; ++t) mask[order[t]] = 1.f;
        mask[d] = 1.f;
        float* dm = (float*)dalloc(d1 * 4);
        CUDA_OK(cudaMemcpy(dm, mask.data(), d1 * 4, cudaMemcpyHostToDevice));
        float* dW = (float*)dalloc(d1 * 4);
        const float b =
            train_linear(handle, dXb, dy, ntr, d1, dW, dm, 80, 2e-3f, dXbva,
                         dyva, nva);
        CUDA_OK(cudaFree(dm));
        CUDA_OK(cudaFree(dW));
        return b;
    };
    const float p32 = prefix_bits(32);
    const float p48 = prefix_bits(48);

    float* dXbq = (float*)dalloc((size_t)nva * d1 * 4);
    float* dWq = (float*)dalloc(d1 * 4);
    CUDA_OK(cudaMemcpy(dXbq, dXbva, (size_t)nva * d1 * 4, cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaMemcpy(dWq, dWlin, d1 * 4, cudaMemcpyDeviceToDevice));
    k_q_xw<<<(nva + 255) / 256, 256>>>(dXbq, dWq, nva, d1, 4, 6);
    float* dlogq = (float*)dalloc((size_t)nva * 4);
    float* dceq = (float*)dalloc(4);
    gemm_logits(handle, dXbq, dWq, dlogq, nva, d1, 1);
    k_ce<<<dim3((nva + 127) / 128, 1), 128>>>(dlogq, dyva, dceq, nva, 1);
    CUDA_OK(cudaDeviceSynchronize());
    float qce = 0.f;
    CUDA_OK(cudaMemcpy(&qce, dceq, 4, cudaMemcpyDeviceToHost));
    const float q46 = (qce / (float)nva) / kLn2;

    struct Row {
        int K;
        const char* mode;
        float bits;
        int hard;
    };
    std::vector<Row> rows;
    const int Ks[] = {4, 8, 12, 16, 24};
    float* dW = (float*)dalloc(d * 4);
    float* db1 = (float*)dalloc(kMaxK * 4);
    float* dv = (float*)dalloc(kMaxK * 4);
    float* db2 = (float*)dalloc(4);
    int* dassign = (int*)dalloc(d * 4);

    for (int K : Ks) {
        auto assign = kmeans_experts(Xtr.data(), ntr, d, K, 42 + K);
        std::vector<int> hist(K, 0);
        for (int a : assign) hist[a]++;
        CUDA_OK(cudaMemcpy(dassign, assign.data(), d * 4, cudaMemcpyHostToDevice));
        train_moe(dX, dy, dassign, ntr, d, K, 60, 3e-3f, 0, dW, db1, dv, db2);
        const float soft = eval_moe(dXva, dyva, dassign, dW, db1, dv, db2, nva, d,
                                    K, 0);
        const float h1 = eval_moe(dXva, dyva, dassign, dW, db1, dv, db2, nva, d, K,
                                  1);
        const float h2 = eval_moe(dXva, dyva, dassign, dW, db1, dv, db2, nva, d, K,
                                  2);
        rows.push_back({K, "soft_all", soft, 0});
        rows.push_back({K, "hard_top1", h1, 1});
        rows.push_back({K, "hard_top2", h2, 2});
        train_moe(dX, dy, dassign, ntr, d, K, 60, 3e-3f, 1, dW, db1, dv, db2);
        const float h1t = eval_moe(dXva, dyva, dassign, dW, db1, dv, db2, nva, d,
                                   K, 1);
        rows.push_back({K, "hard_trained_top1", h1t, 1});
        std::fprintf(stderr, "K=%d sizes", K);
        for (int c = 0; c < K; ++c) std::fprintf(stderr, " %d", hist[c]);
        std::fprintf(stderr, " soft=%.4f h1=%.4f h2=%.4f h1t=%.4f\n", soft, h1,
                     h2, h1t);
    }

    CUDA_OK(cudaEventRecord(t1));
    CUDA_OK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));

    auto call_of = [&](float b) {
        const float delta = b - lin;
        if (delta < 0.001f) return "free";
        if (delta < 0.010f) return "visible";
        return "sizeable";
    };

    char outpath[512];
    const char* la = std::getenv("LOCALAPPDATA");
    std::snprintf(outpath, sizeof(outpath), "%s\\hp_lab\\gpu_moe_cluster.json",
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
        "  \"linear_val_bits\": %.6f,\n  \"prefix32_val_bits\": %.6f,\n"
        "  \"prefix48_val_bits\": %.6f,\n  \"q4x_q6w_val_bits\": %.6f,\n"
        "  \"oracle_best_expert_val_bits\": %.6f,\n"
        "  \"visible_loss\": 0.001,\n  \"sizeable_loss\": 0.010,\n"
        "  \"caveat\": \"Frozen v57 logits. Cluster-MoE still does W·x inside "
        "each cluster; squash then uneven v. Not an accept gate. No LSTM.\",\n"
        "  \"runs\": [",
        prop.name, ms, ntr, nva, d, lin, p32, p48, q46, uni[order[0]]);
    emit(buf);
    int first = 1;
    auto emit_row = [&](int K, const char* mode, float bits, int hard) {
        std::snprintf(buf, sizeof(buf),
                      "%s{\"K\":%d,\"mode\":\"%s\",\"hard_top\":%d,\"val_bits\":%.6f,"
                      "\"delta\":%.6f,\"call\":\"%s\"}",
                      first ? "" : ",", K, mode, hard, bits, bits - lin,
                      call_of(bits));
        first = 0;
        emit(buf);
    };
    emit_row(1, "linear_all", lin, 0);
    emit_row(1, "prefix32", p32, 0);
    emit_row(1, "prefix48", p48, 0);
    emit_row(1, "q4x_q6w", q46, 0);
    for (const auto& r : rows) emit_row(r.K, r.mode, r.bits, r.hard);
    emit("]\n}\n");
    if (jo) std::fclose(jo);
    std::fprintf(stderr, "wrote %s (%.1f ms on %s)\n", outpath, ms, prop.name);
    cublasDestroy(handle);
    return 0;
}
