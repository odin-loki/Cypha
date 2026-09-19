// GPU expert + layer-1 mixer importance + G1 trials.
//   nvcc -O3 -std=c++17 -arch=sm_86 hp/tools/gpu_importance.cu -lcublas
//        -o hp/build/gpu_importance.exe
//   hp/build/gpu_importance.exe dump.i16 n_exp n_gates [max_use]

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
static constexpr int kMaxD = 160;

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

__global__ void k_scale_mask(float* gW, const float* mask, int n, int d1,
                             float invn) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * d1) return;
    float g = gW[t] * invn;
    if (mask) g *= mask[t];
    gW[t] = g;
}

__global__ void k_sgd(float* W, const float* gW, float* mom, const float* mask,
                      int n, float lr) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float m = 0.9f * mom[t] + gW[t];
    mom[t] = m;
    float w = W[t] - lr * m;
    if (mask) w *= mask[t];
    W[t] = w;
}

__global__ void k_mute_eval(const float* X, const float* y, const float* W,
                            int n, int d, int nm, float* ce) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (i >= n || m >= nm) return;
    const float* row = X + (size_t)i * d;
    float logit = W[d];
    for (int j = 0; j < d; ++j) {
        if (j == m) continue;
        logit += W[j] * row[j];
    }
    atomicAdd(&ce[m], d_nats(logit, y[i]));
}

__global__ void k_residual(const float* X, const float* y, const int* best,
                           const float* W, int n, int d, float* ce) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = X + (size_t)i * d;
    const int b = best[0];
    float logit = row[b] + W[d];
    for (int j = 0; j < d; ++j) {
        if (j == b) continue;
        logit += W[j] * (row[j] - row[b]);
    }
    atomicAdd(ce, d_nats(logit, y[i]));
}

__global__ void k_qeval(const float* X, const float* y, const float* W, int n,
                        int d, int xbits, int wbits, float* ce) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xstep = 2048.f / (float)(1 << (xbits - 1));
    const float ws = (float)(1 << wbits);
    float logit = rintf(W[d] * ws) / ws;
    const float* row = X + (size_t)i * d;
    for (int j = 0; j < d; ++j) {
        float x = rintf(row[j] / xstep) * xstep;
        float w = rintf(W[j] * ws) / ws;
        logit += w * x;
    }
    atomicAdd(ce, d_nats(logit, y[i]));
}

__global__ void k_prec_moe(const float* X, const float* y, const float* W, int n,
                           int d, float theta, float* ce) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xstep = 2048.f / 8.f;
    float logit = W[d];
    const float* row = X + (size_t)i * d;
    for (int j = 0; j < d; ++j) {
        float x = row[j];
        if (fabsf(x) < theta) x = rintf(x / xstep) * xstep;
        logit += W[j] * x;
    }
    atomicAdd(ce, d_nats(logit, y[i]));
}

__global__ void k_softmax_mix(const float* X, const float* y, const float* W,
                              int n, int d, float* ce) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = X + (size_t)i * d;
    float m = -1e9f;
    for (int j = 0; j < d; ++j) m = fmaxf(m, W[j]);
    float z = 0.f, acc = 0.f;
    for (int j = 0; j < d; ++j) {
        const float e = expf((W[j] - m) * 8.f);
        z += e;
        acc += e * row[j];
    }
    atomicAdd(ce, d_nats(acc / fmaxf(z, 1e-8f), y[i]));
}

static void* dalloc(size_t b) {
    void* p = nullptr;
    CUDA_OK(cudaMalloc(&p, b));
    CUDA_OK(cudaMemset(p, 0, b));
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
        k_scale_mask<<<(d1 + 255) / 256, 256>>>(gW, mask, 1, d1, 1.f / (float)n);
        k_sgd<<<(d1 + 255) / 256, 256>>>(W, gW, mom, mask, d1, lr);
    }
    float* lva = (float*)dalloc((size_t)nva * 4);
    CUDA_OK(cudaMemset(ce, 0, 4));
    gemm_logits(h, Xbva, W, lva, nva, d1, 1);
    k_ce<<<dim3((nva + 127) / 128, 1), 128>>>(lva, yva, ce, nva, 1);
    CUDA_OK(cudaDeviceSynchronize());
    float hce = 0.f;
    CUDA_OK(cudaMemcpy(&hce, ce, 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(logits));
    CUDA_OK(cudaFree(err));
    CUDA_OK(cudaFree(gW));
    CUDA_OK(cudaFree(mom));
    CUDA_OK(cudaFree(ce));
    CUDA_OK(cudaFree(lva));
    return (hce / (float)nva) / kLn2;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        std::fprintf(stderr, "usage: gpu_importance dump.i16 n_exp n_gates [maxn]\n");
        return 2;
    }
    const char* dump = argv[1];
    const int d = std::atoi(argv[2]);
    const int ng = std::atoi(argv[3]);
    const int maxn = argc > 4 ? std::atoi(argv[4]) : 250000;
    if (d > kMaxD) {
        std::fprintf(stderr, "d too big\n");
        return 2;
    }
    const int wrec = d + ng + 5;
    std::FILE* f = std::fopen(dump, "rb");
    if (!f) {
        std::perror(dump);
        return 1;
    }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    int n_all = (int)(bytes / (wrec * 2));
    if (n_all > maxn) n_all = maxn;
    std::vector<std::int16_t> raw((size_t)n_all * wrec);
    if (std::fread(raw.data(), 2, (size_t)n_all * wrec, f) !=
        (size_t)n_all * wrec) {
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    int ntr = 0, nva = 0;
    for (int i = 0; i < n_all; ++i) ((i % 5) == 4 ? nva : ntr)++;
    std::vector<float> Xtr((size_t)ntr * d), ytr(ntr), Xva((size_t)nva * d), yva(nva);
    std::vector<float> Gtr((size_t)ntr * ng), Gva((size_t)nva * ng);
    int it = 0, iv = 0;
    for (int i = 0; i < n_all; ++i) {
        const std::int16_t* r = raw.data() + (size_t)i * wrec;
        const int va = ((i % 5) == 4);
        float* x = va ? Xva.data() + (size_t)iv * d : Xtr.data() + (size_t)it * d;
        float* g = va ? Gva.data() + (size_t)iv * ng : Gtr.data() + (size_t)it * ng;
        float* ys = va ? &yva[iv] : &ytr[it];
        for (int j = 0; j < d; ++j) x[j] = (float)r[j];
        for (int j = 0; j < ng; ++j) g[j] = (float)r[d + j];
        *ys = (float)r[wrec - 1];
        va ? ++iv : ++it;
    }

    cublasHandle_t h;
    CUBLAS_OK(cublasCreate(&h));
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
    const float lin = train_linear(h, dXb, dy, ntr, d1, dW, nullptr, 80, 2e-3f,
                                   dXbva, dyva, nva);

    float* dce = (float*)dalloc(d * 4);
    k_mute_eval<<<dim3((nva + 127) / 128, d), 128>>>(dXva, dyva, dW, nva, d, d,
                                                     dce);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> mute(d);
    CUDA_OK(cudaMemcpy(mute.data(), dce, d * 4, cudaMemcpyDeviceToHost));
    for (float& v : mute) v = (v / (float)nva) / kLn2;

    auto prefix = [&](int k) {
        std::vector<float> mask(d1, 0.f);
        for (int t = 0; t < k && t < d; ++t) mask[order[t]] = 1.f;
        mask[d] = 1.f;
        float* dm = (float*)dalloc(d1 * 4);
        CUDA_OK(cudaMemcpy(dm, mask.data(), d1 * 4, cudaMemcpyHostToDevice));
        float* dWp = (float*)dalloc(d1 * 4);
        const float b = train_linear(h, dXb, dy, ntr, d1, dWp, dm, 60, 2e-3f,
                                     dXbva, dyva, nva);
        CUDA_OK(cudaFree(dm));
        CUDA_OK(cudaFree(dWp));
        return b;
    };
    const int pks[] = {4, 8, 16, 24, 32, 48};
    float pbit[6];
    for (int i = 0; i < 6; ++i) pbit[i] = prefix(pks[i]);

    int* dbest = (int*)dalloc(4);
    CUDA_OK(cudaMemcpy(dbest, order.data(), 4, cudaMemcpyHostToDevice));
    float* dWr = (float*)dalloc(d1 * 4);
    CUDA_OK(cudaMemcpy(dWr, dW, d1 * 4, cudaMemcpyDeviceToDevice));
    float* dcer = (float*)dalloc(4);
    k_residual<<<(nva + 127) / 128, 128>>>(dXva, dyva, dbest, dWr, nva, d, dcer);
    CUDA_OK(cudaDeviceSynchronize());
    float rce = 0.f;
    CUDA_OK(cudaMemcpy(&rce, dcer, 4, cudaMemcpyDeviceToHost));
    const float residual = (rce / (float)nva) / kLn2;

    float* dceq = (float*)dalloc(4);
    k_qeval<<<(nva + 127) / 128, 128>>>(dXva, dyva, dW, nva, d, 4, 6, dceq);
    CUDA_OK(cudaDeviceSynchronize());
    float qce = 0.f;
    CUDA_OK(cudaMemcpy(&qce, dceq, 4, cudaMemcpyDeviceToHost));
    const float q46 = (qce / (float)nva) / kLn2;

    CUDA_OK(cudaMemset(dceq, 0, 4));
    k_prec_moe<<<(nva + 127) / 128, 128>>>(dXva, dyva, dW, nva, d, 256.f, dceq);
    CUDA_OK(cudaDeviceSynchronize());
    float pmoe = 0.f;
    CUDA_OK(cudaMemcpy(&pmoe, dceq, 4, cudaMemcpyDeviceToHost));
    pmoe = (pmoe / (float)nva) / kLn2;

    CUDA_OK(cudaMemset(dceq, 0, 4));
    k_softmax_mix<<<(nva + 127) / 128, 128>>>(dXva, dyva, dW, nva, d, dceq);
    CUDA_OK(cudaDeviceSynchronize());
    float smx = 0.f;
    CUDA_OK(cudaMemcpy(&smx, dceq, 4, cudaMemcpyDeviceToHost));
    smx = (smx / (float)nva) / kLn2;

    float* dG = nullptr, *dGva = nullptr, *dGb = nullptr, *dGbva = nullptr,
           *dWg = nullptr;
    float glin = 0.f;
    std::vector<float> gmute;
    if (ng > 0) {
        CUDA_OK(cudaMalloc(&dG, Gtr.size() * 4));
        CUDA_OK(cudaMalloc(&dGva, Gva.size() * 4));
        CUDA_OK(cudaMemcpy(dG, Gtr.data(), Gtr.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dGva, Gva.data(), Gva.size() * 4, cudaMemcpyHostToDevice));
        const int g1 = ng + 1;
        dGb = (float*)dalloc((size_t)ntr * g1 * 4);
        dGbva = (float*)dalloc((size_t)nva * g1 * 4);
        k_append_bias<<<(ntr + 255) / 256, 256>>>(dG, dGb, ntr, ng);
        k_append_bias<<<(nva + 255) / 256, 256>>>(dGva, dGbva, nva, ng);
        dWg = (float*)dalloc(g1 * 4);
        glin = train_linear(h, dGb, dy, ntr, g1, dWg, nullptr, 80, 2e-3f, dGbva,
                            dyva, nva);
        float* dceg = (float*)dalloc(ng * 4);
        k_mute_eval<<<dim3((nva + 127) / 128, ng), 128>>>(dGva, dyva, dWg, nva, ng,
                                                          ng, dceg);
        CUDA_OK(cudaDeviceSynchronize());
        gmute.assign(ng, 0.f);
        CUDA_OK(cudaMemcpy(gmute.data(), dceg, ng * 4, cudaMemcpyDeviceToHost));
        for (float& v : gmute) v = (v / (float)nva) / kLn2;
        CUDA_OK(cudaFree(dceg));
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
    std::snprintf(outpath, sizeof(outpath), "%s\\hp_lab\\gpu_importance.json",
                  la ? la : ".");
    std::FILE* jo = std::fopen(outpath, "wb");
    auto emit = [&](const char* s) {
        std::fputs(s, stdout);
        if (jo) std::fputs(s, jo);
    };
    char buf[4096];
    std::snprintf(buf, sizeof(buf),
                  "{\n  \"device\": \"%s\",\n  \"wall_ms\": %.2f,\n  \"n_train\": %d,\n"
                  "  \"n_val\": %d,\n  \"n_exp\": %d,\n  \"n_gates\": %d,\n"
                  "  \"linear_val_bits\": %.6f,\n  \"residual_val_bits\": %.6f,\n"
                  "  \"q4x_q6w_val_bits\": %.6f,\n  \"prec_moe_256_val_bits\": %.6f,\n"
                  "  \"softmax_val_bits\": %.6f,\n  \"layer1_linear_val_bits\": %.6f,\n"
                  "  \"best_expert_val_bits\": %.6f,\n  \"prefix\": [",
                  prop.name, ms, ntr, nva, d, ng, lin, residual, q46, pmoe, smx, glin,
                  uni[order[0]]);
    emit(buf);
    for (int i = 0; i < 6; ++i) {
        std::snprintf(buf, sizeof(buf),
                      "%s{\"k\":%d,\"val_bits\":%.6f,\"delta\":%.6f,\"call\":\"%s\"}",
                      i ? "," : "", pks[i], pbit[i], pbit[i] - lin, call_of(pbit[i]));
        emit(buf);
    }
    emit("],\n  \"expert_mute\": [");
    std::vector<int> mi(d);
    for (int i = 0; i < d; ++i) mi[i] = i;
    std::sort(mi.begin(), mi.end(),
              [&](int a, int b) { return (mute[a] - lin) > (mute[b] - lin); });
    for (int t = 0; t < d; ++t) {
        const int j = mi[t];
        std::snprintf(buf, sizeof(buf),
                      "%s{\"i\":%d,\"uni\":%.6f,\"mute_bits\":%.6f,\"delta\":%.6f}",
                      t ? "," : "", j, uni[j], mute[j], mute[j] - lin);
        emit(buf);
    }
    emit("],\n  \"gate_mute\": [");
    for (int j = 0; j < ng; ++j) {
        std::snprintf(buf, sizeof(buf),
                      "%s{\"i\":%d,\"mute_bits\":%.6f,\"delta\":%.6f}",
                      j ? "," : "", j, gmute[j], gmute[j] - glin);
        emit(buf);
    }
    emit("]\n}\n");
    if (jo) std::fclose(jo);
    std::fprintf(stderr, "wrote %s (%.1f ms)\n", outpath, ms);
    cublasDestroy(h);
    return 0;
}
