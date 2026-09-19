// GPU mixer search v2 — cuBLAS SGEMM, diagnostic only.
//   nvcc -O3 -std=c++17 -arch=sm_86 hp/tools/gpu_mixer_search.cu -lcublas -o hp/build/gpu_mixer_search.exe

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
            std::fprintf(stderr, "cuBLAS %s:%d status %d\n", __FILE__,         \
                         __LINE__, (int)_s);                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static constexpr float kScale = 256.f;
static constexpr float kLn2 = 0.693147182f;

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
    const float logit = logits[(size_t)i * nm + m];
    err[(size_t)i * nm + m] = (d_sig(logit / kScale) - y[i]) / kScale;
}

__global__ void k_ce(const float* logits, const float* y, float* ce, int n,
                     int nm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (i >= n || m >= nm) return;
    atomicAdd(&ce[m], d_nats(logits[(size_t)i * nm + m], y[i]));
}

__global__ void k_scale_mask(float* gW, const float* mask, int n, int d1, float invn) {
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

__global__ void k_q16(float* W, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    W[t] = rintf(W[t] * 65536.f) / 65536.f;
}

__global__ void k_pairwise(const float* X, const int* top, int ntop, float* Xp,
                           int n, int d, int dp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = X + (size_t)i * d;
    float* out = Xp + (size_t)i * dp;
    int c = 0;
    for (int a = 0; a < ntop; ++a) out[c++] = row[top[a]];
    for (int a = 0; a < ntop; ++a)
        for (int b = a + 1; b < ntop; ++b)
            out[c++] = row[top[a]] * row[top[b]] / kScale;
}

__global__ void k_symbolic(const float* X, const float* y, int n, int d,
                           const int* top, int ntop, int n_pop, float* bits,
                           int* op_out, int* ijk) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_pop) return;
    unsigned rng = 0x9E3779B9u * (unsigned)(m + 1);
    auto rnd = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return rng;
    };
    const int op = (int)(rnd() % 8u);
    const int ei = top[rnd() % (unsigned)ntop];
    const int ej = top[rnd() % (unsigned)ntop];
    const int ek = top[rnd() % (unsigned)ntop];
    const float a = ((int)(rnd() & 1023) - 512) / 256.f;
    const float c = ((int)(rnd() & 1023) - 512) / 256.f;
    const float bias = ((int)(rnd() & 1023) - 512) / 16.f;
    float sum = 0.f;
    for (int t = 0; t < n; ++t) {
        const float* row = X + (size_t)t * d;
        const float xi = row[ei], xj = row[ej], xk = row[ek];
        float f;
        switch (op) {
        case 0: f = xi + xj; break;
        case 1: f = xi - xj; break;
        case 2: f = xi * xj / kScale; break;
        case 3: f = fabsf(xi - xj); break;
        case 4: f = fminf(xi, xj); break;
        case 5: f = fmaxf(xi, xj); break;
        case 6: f = 0.5f * (xi + xj); break;
        default: f = xi; break;
        }
        sum += d_nats(a * f + c * xk + bias, y[t]);
    }
    bits[m] = (sum / (float)n) / kLn2;
    op_out[m] = op;
    ijk[m * 3] = ei;
    ijk[m * 3 + 1] = ej;
    ijk[m * 3 + 2] = ek;
}

static void* dalloc(size_t bytes) {
    void* p = nullptr;
    CUDA_OK(cudaMalloc(&p, bytes));
    CUDA_OK(cudaMemset(p, 0, bytes));
    return p;
}

// logits (n x nm row-major) = Xb (n x d1) @ W^T (d1 x nm)
static void gemm_logits(cublasHandle_t h, const float* Xb, const float* W,
                        float* logits, int n, int d1, int nm) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, nm, n, d1, &alpha, W, d1,
                          Xb, d1, &beta, logits, nm));
}

// gW (nm x d1 row-major) = err^T @ Xb   then scaled in k_scale_mask
static void gemm_grad(cublasHandle_t h, const float* Xb, const float* err,
                      float* gW, int n, int d1, int nm) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, d1, nm, n, &alpha, Xb, d1,
                          err, nm, &beta, gW, d1));
}

static void train_bank(cublasHandle_t h, const float* Xb, const float* y, int n,
                       int d1, int nm, float* W, const float* mask, int steps,
                       float lr, const float* Xbva, const float* yva, int nva,
                       std::vector<float>* bits) {
    float* logits = (float*)dalloc((size_t)n * nm * 4);
    float* err = (float*)dalloc((size_t)n * nm * 4);
    float* gW = (float*)dalloc((size_t)nm * d1 * 4);
    float* mom = (float*)dalloc((size_t)nm * d1 * 4);
    float* ce = (float*)dalloc(nm * 4);
    dim3 block(128);
    dim3 grid((n + 127) / 128, nm);
    const int nW = nm * d1;
    for (int s = 0; s < steps; ++s) {
        gemm_logits(h, Xb, W, logits, n, d1, nm);
        k_err<<<grid, block>>>(logits, y, err, n, nm);
        gemm_grad(h, Xb, err, gW, n, d1, nm);
        k_scale_mask<<<(nW + 255) / 256, 256>>>(gW, mask, nm, d1, 1.f / (float)n);
        k_sgd<<<(nW + 255) / 256, 256>>>(W, gW, mom, mask, nW, lr);
    }
    CUDA_OK(cudaGetLastError());
    float* logits_va = (float*)dalloc((size_t)nva * nm * 4);
    dim3 gridva((nva + 127) / 128, nm);
    CUDA_OK(cudaMemset(ce, 0, nm * 4));
    gemm_logits(h, Xbva, W, logits_va, nva, d1, nm);
    k_ce<<<gridva, block>>>(logits_va, yva, ce, nva, nm);
    CUDA_OK(cudaDeviceSynchronize());
    bits->assign(nm, 0.f);
    CUDA_OK(cudaMemcpy(bits->data(), ce, nm * 4, cudaMemcpyDeviceToHost));
    for (float& v : *bits) v = (v / (float)nva) / kLn2;
    CUDA_OK(cudaFree(logits));
    CUDA_OK(cudaFree(err));
    CUDA_OK(cudaFree(gW));
    CUDA_OK(cudaFree(mom));
    CUDA_OK(cudaFree(ce));
    CUDA_OK(cudaFree(logits_va));
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    const char* dump = argc > 1 ? argv[1] : nullptr;
    const int d = argc > 2 ? std::atoi(argv[2]) : 77;
    std::string dump_s;
    if (!dump) {
        const char* la = std::getenv("LOCALAPPDATA");
        dump_s = std::string(la ? la : ".") + "\\hp_lab\\v57.i16";
        dump = dump_s.c_str();
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
    if (std::fread(raw.data(), 2, (size_t)n_all * wrec, f) != (size_t)n_all * wrec) {
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    int ntr = 0, nva = 0;
    for (int i = 0; i < n_all; ++i) ((i % 5) == 4 ? nva : ntr)++;
    std::vector<float> Xtr((size_t)ntr * d), ytr(ntr), Xva((size_t)nva * d), yva(nva);
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

    const int ks[] = {1, 2, 4, 8, 12, 16, 24, 32, 48, 77};
    constexpr int n_prefix = 10;
    constexpr int n_sparse = 8192;
    constexpr int ksp = 8;
    const int nm = 1 + n_prefix + n_sparse;
    std::vector<float> mask((size_t)nm * d1, 0.f);
    for (int j = 0; j < d1; ++j) mask[j] = 1.f;
    for (int p = 0; p < n_prefix; ++p) {
        const int k = std::min(ks[p], d);
        for (int t = 0; t < k; ++t) mask[(size_t)(1 + p) * d1 + order[t]] = 1.f;
        mask[(size_t)(1 + p) * d1 + d] = 1.f;
    }
    for (int m = 0; m < n_sparse; ++m) {
        for (int t = 0; t < ksp; ++t)
            mask[(size_t)(1 + n_prefix + m) * d1 + ((m * 17 + t * 31 + 3) % d)] = 1.f;
        mask[(size_t)(1 + n_prefix + m) * d1 + d] = 1.f;
    }
    float* dW = (float*)dalloc((size_t)nm * d1 * 4);
    float* dmask = (float*)dalloc((size_t)nm * d1 * 4);
    CUDA_OK(cudaMemcpy(dmask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice));

    size_t free0 = 0, tot = 0;
    CUDA_OK(cudaMemGetInfo(&free0, &tot));

    std::vector<float> bits;
    train_bank(handle, dXb, dy, ntr, d1, nm, dW, dmask, 80, 2e-3f, dXbva, dyva,
               nva, &bits);

    // Q16 snap of dense linear (model 0)
    float* dWq = (float*)dalloc(d1 * 4);
    CUDA_OK(cudaMemcpy(dWq, dW, d1 * 4, cudaMemcpyDeviceToDevice));
    k_q16<<<(d1 + 255) / 256, 256>>>(dWq, d1);
    std::vector<float> q16bits;
    train_bank(handle, dXb, dy, ntr, d1, 1, dWq, nullptr, 0, 0.f, dXbva, dyva, nva,
               &q16bits);
    // 0 steps: just eval. train_bank with 0 steps still evals. Good.

    // Pairwise products of top-12
    constexpr int ntop = 12;
    const int npair = ntop * (ntop - 1) / 2;
    const int dp = ntop + npair;
    const int dp1 = dp + 1;
    int* dtop = (int*)dalloc(ntop * 4);
    CUDA_OK(cudaMemcpy(dtop, order.data(), ntop * 4, cudaMemcpyHostToDevice));
    float* dXp = (float*)dalloc((size_t)ntr * dp * 4);
    float* dXpva = (float*)dalloc((size_t)nva * dp * 4);
    k_pairwise<<<(ntr + 255) / 256, 256>>>(dX, dtop, ntop, dXp, ntr, d, dp);
    k_pairwise<<<(nva + 255) / 256, 256>>>(dXva, dtop, ntop, dXpva, nva, d, dp);
    float* dXpb = (float*)dalloc((size_t)ntr * dp1 * 4);
    float* dXpbva = (float*)dalloc((size_t)nva * dp1 * 4);
    k_append_bias<<<(ntr + 255) / 256, 256>>>(dXp, dXpb, ntr, dp);
    k_append_bias<<<(nva + 255) / 256, 256>>>(dXpva, dXpbva, nva, dp);
    float* dWp = (float*)dalloc(dp1 * 4);
    std::vector<float> pairbits;
    train_bank(handle, dXpb, dy, ntr, dp1, 1, dWp, nullptr, 80, 2e-3f, dXpbva,
               dyva, nva, &pairbits);

    constexpr int n_pop = 16384;
    float* dsb = (float*)dalloc(n_pop * 4);
    int* dop = (int*)dalloc(n_pop * 4);
    int* dijk = (int*)dalloc(n_pop * 12);
    k_symbolic<<<(n_pop + 255) / 256, 256>>>(dXva, dyva, nva, d, dtop, ntop, n_pop,
                                             dsb, dop, dijk);
    std::vector<float> sb(n_pop);
    std::vector<int> sop(n_pop), sijk(n_pop * 3);
    CUDA_OK(cudaMemcpy(sb.data(), dsb, n_pop * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(sop.data(), dop, n_pop * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(sijk.data(), dijk, n_pop * 12, cudaMemcpyDeviceToHost));
    int sbest = 0;
    for (int i = 1; i < n_pop; ++i)
        if (sb[i] < sb[sbest]) sbest = i;
    static const char* opname[] = {"add", "sub", "mul/256", "absdiff",
                                   "min", "max", "avg",     "id"};

    CUDA_OK(cudaEventRecord(t1));
    CUDA_OK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));
    size_t free1 = 0;
    CUDA_OK(cudaMemGetInfo(&free1, &tot));

    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));

    float sparse_best = 1e9f;
    std::vector<float> sparse(bits.begin() + 1 + n_prefix, bits.end());
    for (float v : sparse) sparse_best = std::min(sparse_best, v);
    std::nth_element(sparse.begin(), sparse.begin() + sparse.size() / 2, sparse.end());
    const float sparse_med = sparse[sparse.size() / 2];

    char outpath[512];
    const char* la = std::getenv("LOCALAPPDATA");
    std::snprintf(outpath, sizeof(outpath), "%s\\hp_lab\\gpu_mixer_search.json",
                  la ? la : ".");
    std::FILE* jo = std::fopen(outpath, "wb");
    auto emit = [&](const char* s) {
        std::fputs(s, stdout);
        if (jo) std::fputs(s, jo);
    };
    char buf[4096];
    std::snprintf(
        buf, sizeof(buf),
        "{\n  \"backend\": \"cuda-cublas\",\n  \"device\": \"%s\",\n"
        "  \"sm\": %d%d,\n  \"wall_ms\": %.2f,\n  \"vram_total_mib\": %zu,\n"
        "  \"vram_used_mib\": %zu,\n  \"n_train\": %d,\n  \"n_val\": %d,\n"
        "  \"n_experts\": %d,\n  \"n_models\": %d,\n"
        "  \"oracle_best_expert_val_bits\": %.6f,\n"
        "  \"linear_all_val_bits\": %.6f,\n  \"linear_q16_val_bits\": %.6f,\n"
        "  \"pairwise_top12_val_bits\": %.6f,\n  \"prefix\": [",
        prop.name, prop.major, prop.minor, ms, tot >> 20, (tot - free1) >> 20, ntr,
        nva, d, nm, uni[order[0]], bits[0], q16bits[0], pairbits[0]);
    emit(buf);
    for (int p = 0; p < n_prefix; ++p) {
        std::snprintf(buf, sizeof(buf), "%s{\"k\":%d,\"val_bits\":%.6f}", p ? "," : "",
                      std::min(ks[p], d), bits[1 + p]);
        emit(buf);
    }
    std::snprintf(
        buf, sizeof(buf),
        "],\n  \"sparse_random\": {\"n_models\":%d,\"k\":%d,\"best_val_bits\":%.6f,"
        "\"median_val_bits\":%.6f},\n"
        "  \"symbolic\": {\"n_pop\":%d,\"val_bits\":%.6f,\"op\":\"%s\",\"i\":%d,"
        "\"j\":%d,\"k\":%d},\n"
        "  \"best_univariate_experts\": [%d,%d,%d,%d,%d,%d,%d,%d],\n"
        "  \"caveat\": \"Offline frozen v57 logits. Compass only. cuBLAS SGEMM.\"\n}\n",
        n_sparse, ksp, sparse_best, sparse_med, n_pop, sb[sbest],
        opname[sop[sbest] & 7], sijk[sbest * 3], sijk[sbest * 3 + 1],
        sijk[sbest * 3 + 2], order[0], order[1], order[2], order[3], order[4],
        order[5], order[6], order[7]);
    emit(buf);
    if (jo) std::fclose(jo);
    std::fprintf(stderr, "wrote %s (%.1f ms, %zu MiB used on %s)\n", outpath, ms,
                 (tot - free1) >> 20, prop.name);
    cublasDestroy(handle);
    return 0;
}
