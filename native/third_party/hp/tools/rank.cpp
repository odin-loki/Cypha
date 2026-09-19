// tools/rank.cpp
//
// C++ stand-in for mp_rank.py. Diagnostic only: not linked into the codec.
// Floats are allowed here (same rule as dump_experts.cpp). No CUDA, no Qt,
// no numpy. Wall time of THIS tool is not the codec runtime; use hp --profile
// for that.
//
//   rank <dump.i16> <n_experts>
//
// Input: little-endian int16 records of (n_experts + 1); last value is the bit.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double mean_col(const std::vector<double>& z, int n, int p, int j) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += z[i * p + j];
    return s / static_cast<double>(n);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: rank <dump.i16> <n_experts>\n");
        return 2;
    }
    const int p = std::atoi(argv[2]);
    if (p < 2 || p > 512) {
        std::fprintf(stderr, "bad n_experts\n");
        return 2;
    }

    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror(argv[1]); return 1; }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const int w = p + 1;
    const int n = static_cast<int>(bytes / (static_cast<long>(w) * 2));
    if (n < p + 2) {
        std::fprintf(stderr, "not enough samples n=%d\n", n);
        std::fclose(f);
        return 1;
    }

    std::vector<std::int16_t> rec(static_cast<std::size_t>(w));
    std::vector<double> X(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (std::fread(rec.data(), 2, static_cast<std::size_t>(w), f) !=
            static_cast<std::size_t>(w))
            break;
        for (int j = 0; j < p; ++j)
            X[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(j)] = rec[j];
        y[static_cast<std::size_t>(i)] = rec[p];
    }
    std::fclose(f);

    const double gamma = static_cast<double>(p) / static_cast<double>(n);
    std::printf("samples n=%d  experts p=%d  gamma=%.6f\n", n, p, gamma);
    if (gamma < 0.01)
        std::printf("NOTE: Marchenko-Pastur does not apply at this gamma. "
                    "Use PR and entropy rank.\n\n");

    std::vector<double> mu(static_cast<std::size_t>(p)),
        sd(static_cast<std::size_t>(p));
    for (int j = 0; j < p; ++j) {
        mu[static_cast<std::size_t>(j)] = mean_col(X, n, p, j);
        double v = 0;
        for (int i = 0; i < n; ++i) {
            const double d =
                X[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) +
                  static_cast<std::size_t>(j)] -
                mu[static_cast<std::size_t>(j)];
            v += d * d;
        }
        v = std::sqrt(v / static_cast<double>(n));
        sd[static_cast<std::size_t>(j)] = v > 0 ? v : 1.0;
    }

    std::vector<double> C(static_cast<std::size_t>(p) * static_cast<std::size_t>(p), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < p; ++a) {
            const double za =
                (X[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) +
                   static_cast<std::size_t>(a)] -
                 mu[static_cast<std::size_t>(a)]) /
                sd[static_cast<std::size_t>(a)];
            for (int b = a; b < p; ++b) {
                const double zb =
                    (X[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) +
                       static_cast<std::size_t>(b)] -
                     mu[static_cast<std::size_t>(b)]) /
                    sd[static_cast<std::size_t>(b)];
                C[static_cast<std::size_t>(a) * static_cast<std::size_t>(p) +
                  static_cast<std::size_t>(b)] += za * zb;
            }
        }
    }
    for (int a = 0; a < p; ++a) {
        for (int b = a; b < p; ++b) {
            C[static_cast<std::size_t>(a) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(b)] /= static_cast<double>(n);
            C[static_cast<std::size_t>(b) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(a)] =
                C[static_cast<std::size_t>(a) * static_cast<std::size_t>(p) +
                  static_cast<std::size_t>(b)];
        }
    }

    // Jacobi eigenvalue decomposition, descending.
    std::vector<double> A = C;
    std::vector<double> eigs(static_cast<std::size_t>(p));
    for (int iter = 0; iter < 64; ++iter) {
        int p_i = 0, p_j = 1;
        double mx = 0;
        for (int i = 0; i < p; ++i)
            for (int j = i + 1; j < p; ++j) {
                const double aij = std::fabs(
                    A[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) +
                      static_cast<std::size_t>(j)]);
                if (aij > mx) { mx = aij; p_i = i; p_j = j; }
            }
        if (mx < 1e-12) break;
        const double app =
            A[static_cast<std::size_t>(p_i) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(p_i)];
        const double aqq =
            A[static_cast<std::size_t>(p_j) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(p_j)];
        const double apq =
            A[static_cast<std::size_t>(p_i) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(p_j)];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0 ? 1.0 : -1.0) /
                         (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        for (int k = 0; k < p; ++k) {
            if (k == p_i || k == p_j) continue;
            const double aik =
                A[static_cast<std::size_t>(p_i) * static_cast<std::size_t>(p) +
                  static_cast<std::size_t>(k)];
            const double ajk =
                A[static_cast<std::size_t>(p_j) * static_cast<std::size_t>(p) +
                  static_cast<std::size_t>(k)];
            A[static_cast<std::size_t>(p_i) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(k)] = c * aik - s * ajk;
            A[static_cast<std::size_t>(k) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(p_i)] =
                A[static_cast<std::size_t>(p_i) * static_cast<std::size_t>(p) +
                  static_cast<std::size_t>(k)];
            A[static_cast<std::size_t>(p_j) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(k)] = s * aik + c * ajk;
            A[static_cast<std::size_t>(k) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(p_j)] =
                A[static_cast<std::size_t>(p_j) * static_cast<std::size_t>(p) +
                  static_cast<std::size_t>(k)];
        }
        A[static_cast<std::size_t>(p_i) * static_cast<std::size_t>(p) +
          static_cast<std::size_t>(p_i)] = app - t * apq;
        A[static_cast<std::size_t>(p_j) * static_cast<std::size_t>(p) +
          static_cast<std::size_t>(p_j)] = aqq + t * apq;
        A[static_cast<std::size_t>(p_i) * static_cast<std::size_t>(p) +
          static_cast<std::size_t>(p_j)] = 0;
        A[static_cast<std::size_t>(p_j) * static_cast<std::size_t>(p) +
          static_cast<std::size_t>(p_i)] = 0;
    }
    for (int i = 0; i < p; ++i)
        eigs[static_cast<std::size_t>(i)] =
            A[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) +
              static_cast<std::size_t>(i)];
    for (int i = 0; i < p; ++i)
        for (int j = i + 1; j < p; ++j)
            if (eigs[static_cast<std::size_t>(j)] > eigs[static_cast<std::size_t>(i)]) {
                const double t = eigs[static_cast<std::size_t>(i)];
                eigs[static_cast<std::size_t>(i)] = eigs[static_cast<std::size_t>(j)];
                eigs[static_cast<std::size_t>(j)] = t;
            }

    double sum = 0, sum2 = 0, h = 0;
    for (int i = 0; i < p; ++i) {
        if (eigs[static_cast<std::size_t>(i)] < 0) eigs[static_cast<std::size_t>(i)] = 0;
        sum += eigs[static_cast<std::size_t>(i)];
        sum2 += eigs[static_cast<std::size_t>(i)] * eigs[static_cast<std::size_t>(i)];
    }
    const double pr = sum2 > 0 ? (sum * sum) / sum2 : 0;
    for (int i = 0; i < p; ++i) {
        if (eigs[static_cast<std::size_t>(i)] <= 0 || sum <= 0) continue;
        const double pk = eigs[static_cast<std::size_t>(i)] / sum;
        h -= pk * std::log(pk);
    }
    const double er = std::exp(h);

    std::printf("participation ratio    = %.3f of %d\n", pr, p);
    std::printf("entropy effective rank = %.3f of %d\n", er, p);
    std::printf("top-1 variance share   = %.1f%%\n\n",
                sum > 0 ? 100.0 * eigs[0] / sum : 0.0);
    std::printf("eigenvalue spectrum:\n");
    for (int i = 0; i < p; ++i)
        std::printf("  %2d  %8.4f\n", i, eigs[static_cast<std::size_t>(i)]);
    return 0;
}
