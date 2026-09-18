// Exhaustive subset + tiny-formula search on the GPU.
// Question: is there a simpler mixer than 77-expert logistic?
//   nvcc -O3 -std=c++17 -arch=sm_86 hp/tools/gpu_simple_search.cu -lcublas -o hp/build/gpu_simple_search.exe

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

// Equal-weight / integer-coeff formulas: logit = (a*xi + b*xj + c*xk)/den + bias
__global__ void k_int_formula(const float* X, const float* y, int n, int d,
                              const int* top, int ntop, float* bits, int* meta) {
    // grid: one thread per (i,j,k,a,b,c) with a,b,c in {-2,-1,0,1,2}, i<=j<=k
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int ncoeff = 5;  // -2..2
    const int ntrip = ntop * ntop * ntop;
    const int nform = ntrip * ncoeff * ncoeff * ncoeff;
    if (m >= nform) return;
    int t = m;
    const int c = t % ncoeff - 2;
    t /= ncoeff;
    const int b = t % ncoeff - 2;
    t /= ncoeff;
    const int a = t % ncoeff - 2;
    t /= ncoeff;
    const int k = t % ntop;
    t /= ntop;
    const int j = t % ntop;
    t /= ntop;
    const int i = t % ntop;
    if (a == 0 && b == 0 && c == 0) {
        bits[m] = 9.f;
        return;
    }
    const int ei = top[i], ej = top[j], ek = top[k];
    const int den = (a ? 1 : 0) + (b ? 1 : 0) + (c ? 1 : 0);
    float sum = 0.f;
    for (int s = 0; s < n; ++s) {
        const float* row = X + (size_t)s * d;
        const float logit =
            (a * row[ei] + b * row[ej] + c * row[ek]) / (float)den;
        sum += d_nats(logit, y[s]);
    }
    bits[m] = (sum / (float)n) / kLn2;
    meta[m * 6] = ei;
    meta[m * 6 + 1] = ej;
    meta[m * 6 + 2] = ek;
    meta[m * 6 + 3] = a;
    meta[m * 6 + 4] = b;
    meta[m * 6 + 5] = c;
}

static void* dalloc(size_t n) {
    void* p = nullptr;
    CUDA_OK(cudaMalloc(&p, n));
    CUDA_OK(cudaMemset(p, 0, n));
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

    constexpr int ntop = 12;
    const int d1 = d + 1;
    float* dXb = (float*)dalloc((size_t)ntr * d1 * 4);
    float* dXbva = (float*)dalloc((size_t)nva * d1 * 4);
    k_append_bias<<<(ntr + 255) / 256, 256>>>(dX, dXb, ntr, d);
    k_append_bias<<<(nva + 255) / 256, 256>>>(dXva, dXbva, nva, d);

    const int nm = (1 << ntop) - 1;  // all nonempty subsets of top-12
    std::vector<float> mask((size_t)nm * d1, 0.f);
    std::vector<int> popcnt(nm);
    for (int m = 0; m < nm; ++m) {
        const int bits = m + 1;
        int pc = 0;
        for (int t = 0; t < ntop; ++t)
            if (bits & (1 << t)) {
                mask[(size_t)m * d1 + order[t]] = 1.f;
                ++pc;
            }
        mask[(size_t)m * d1 + d] = 1.f;
        popcnt[m] = pc;
    }
    float* dW = (float*)dalloc((size_t)nm * d1 * 4);
    float* dmask = (float*)dalloc((size_t)nm * d1 * 4);
    CUDA_OK(cudaMemcpy(dmask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice));

    float* logits = (float*)dalloc((size_t)ntr * nm * 4);
    float* err = (float*)dalloc((size_t)ntr * nm * 4);
    float* gW = (float*)dalloc((size_t)nm * d1 * 4);
    float* mom = (float*)dalloc((size_t)nm * d1 * 4);
    float* ce = (float*)dalloc(nm * 4);
    dim3 block(128);
    dim3 grid((ntr + 127) / 128, nm);
    const int nW = nm * d1;
    const int steps = 60;
    for (int s = 0; s < steps; ++s) {
        gemm_logits(handle, dXb, dW, logits, ntr, d1, nm);
        k_err<<<grid, block>>>(logits, dy, err, ntr, nm);
        gemm_grad(handle, dXb, err, gW, ntr, d1, nm);
        k_scale_mask<<<(nW + 255) / 256, 256>>>(gW, dmask, nm, d1, 1.f / (float)ntr);
        k_sgd<<<(nW + 255) / 256, 256>>>(dW, gW, mom, dmask, nW, 2e-3f);
    }
    float* logits_va = (float*)dalloc((size_t)nva * nm * 4);
    dim3 gridva((nva + 127) / 128, nm);
    CUDA_OK(cudaMemset(ce, 0, nm * 4));
    gemm_logits(handle, dXbva, dW, logits_va, nva, d1, nm);
    k_ce<<<gridva, block>>>(logits_va, dyva, ce, nva, nm);
    std::vector<float> vbits(nm);
    CUDA_OK(cudaMemcpy(vbits.data(), ce, nm * 4, cudaMemcpyDeviceToHost));
    for (float& v : vbits) v = (v / (float)nva) / kLn2;

    float best_at_k[13];
    int best_m_at_k[13];
    for (int k = 0; k <= ntop; ++k) {
        best_at_k[k] = 9.f;
        best_m_at_k[k] = -1;
    }
    for (int m = 0; m < nm; ++m) {
        const int k = popcnt[m];
        if (vbits[m] < best_at_k[k]) {
            best_at_k[k] = vbits[m];
            best_m_at_k[k] = m;
        }
    }

    // Integer 3-term formulas on top-8 (8^3 * 5^3 = 64k * 125 = 8M — too many
    // if each loops nva). Use top-6: 216*125=27000 formulas * 16k val = 4.3e8.
    constexpr int ftop = 6;
    int* dtop = (int*)dalloc(ftop * 4);
    CUDA_OK(cudaMemcpy(dtop, order.data(), ftop * 4, cudaMemcpyHostToDevice));
    const int nform = ftop * ftop * ftop * 5 * 5 * 5;
    float* dform = (float*)dalloc(nform * 4);
    int* dmeta = (int*)dalloc(nform * 6 * 4);
    k_int_formula<<<(nform + 255) / 256, 256>>>(dXva, dyva, nva, d, dtop, ftop,
                                                dform, dmeta);
    std::vector<float> form(nform);
    std::vector<int> meta(nform * 6);
    CUDA_OK(cudaMemcpy(form.data(), dform, nform * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(meta.data(), dmeta, nform * 24, cudaMemcpyDeviceToHost));
    int fbest = 0;
    for (int i = 1; i < nform; ++i)
        if (form[i] < form[fbest]) fbest = i;

    CUDA_OK(cudaEventRecord(t1));
    CUDA_OK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));

    char outpath[512];
    const char* la = std::getenv("LOCALAPPDATA");
    std::snprintf(outpath, sizeof(outpath), "%s\\hp_lab\\gpu_simple_search.json",
                  la ? la : ".");
    std::FILE* jo = std::fopen(outpath, "wb");
    auto emit = [&](const char* s) {
        std::fputs(s, stdout);
        if (jo) std::fputs(s, jo);
    };
    char buf[2048];
    std::snprintf(buf, sizeof(buf),
                  "{\n  \"backend\": \"cuda-cublas-subset\",\n  \"device\": \"%s\",\n"
                  "  \"wall_ms\": %.2f,\n  \"n_subsets\": %d,\n"
                  "  \"oracle_expert0\": %.6f,\n  \"best_by_k\": [",
                  prop.name, ms, nm, uni[order[0]]);
    emit(buf);
    for (int k = 1; k <= ntop; ++k) {
        int m = best_m_at_k[k];
        int setbits = m + 1;
        std::snprintf(buf, sizeof(buf),
                      "%s{\"k\":%d,\"val_bits\":%.6f,\"subset_mask\":%d}",
                      k > 1 ? "," : "", k, best_at_k[k], setbits);
        emit(buf);
    }
    std::snprintf(buf, sizeof(buf),
                  "],\n  \"int_formula\": {\"val_bits\":%.6f,\"i\":%d,\"j\":%d,\"k\":%d,"
                  "\"a\":%d,\"b\":%d,\"c\":%d,\"n_tried\":%d},\n"
                  "  \"top12_experts\": [%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n}\n",
                  form[fbest], meta[fbest * 6], meta[fbest * 6 + 1],
                  meta[fbest * 6 + 2], meta[fbest * 6 + 3], meta[fbest * 6 + 4],
                  meta[fbest * 6 + 5], nform, order[0], order[1], order[2],
                  order[3], order[4], order[5], order[6], order[7], order[8],
                  order[9], order[10], order[11]);
    emit(buf);
    if (jo) std::fclose(jo);
    std::fprintf(stderr, "wrote %s (%.1f ms)\n", outpath, ms);
    cublasDestroy(handle);
    return 0;
}
