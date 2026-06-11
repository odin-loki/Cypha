#include "cypha/generation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "cypha/nig_field.hpp"

namespace cypha {

namespace {

// ── helpers ──────────────────────────────────────────────────────────────────

/// Diagonal-Gaussian log pdf for a batch H (n×d), returns shape (n,).
/// Python ``_batch_logpdf``.
static void batch_logpdf(const double* H_row, int n, int d,
                         const double* mu, const double* v,
                         std::vector<double>& out) {
    out.resize(static_cast<std::size_t>(n));
    double log_norm = 0.0;
    for (int i = 0; i < d; i++) {
        double vs = std::max(v[i], kGenMinVar);
        log_norm += 0.5 * std::log(vs);
    }
    for (int i = 0; i < n; i++) {
        double maha = 0.0;
        const double* row = H_row + static_cast<std::ptrdiff_t>(i) * d;
        for (int j = 0; j < d; j++) {
            double vs = std::max(v[j], kGenMinVar);
            double diff = row[j] - mu[j];
            maha += 0.5 * diff * diff / vs;
        }
        out[static_cast<std::size_t>(i)] = -log_norm - maha;
    }
}

/// Compute raw LLR for candidate matrix H (n×d) against all classes.
/// LLR[i,k] = sum_j R[i,j]*D[k,j] - 0.5*D_sq[k],  R = (H - mu0)*inv_v.
/// No MDL penalty, no context — generation path.
static void raw_llr_matrix(const double* H_row, int n, int d,
                            const double* mu0, const double* inv_v,
                            const double* D_row, int K,
                            std::vector<double>& llr_out) {
    llr_out.assign(static_cast<std::size_t>(n * K), 0.0);
    // Precompute D_sq[k] = D[k,:] . (D[k,:] * inv_v)
    std::vector<double> D_sq(static_cast<std::size_t>(K), 0.0);
    for (int k = 0; k < K; k++) {
        const double* dk = D_row + static_cast<std::ptrdiff_t>(k) * d;
        double s = 0.0;
        for (int j = 0; j < d; j++) {
            s += dk[j] * dk[j] * inv_v[j];
        }
        D_sq[static_cast<std::size_t>(k)] = s;
    }
    for (int i = 0; i < n; i++) {
        const double* h = H_row + static_cast<std::ptrdiff_t>(i) * d;
        double* row_out = llr_out.data() + static_cast<std::ptrdiff_t>(i) * K;
        for (int k = 0; k < K; k++) {
            const double* dk = D_row + static_cast<std::ptrdiff_t>(k) * d;
            double cross = 0.0;
            for (int j = 0; j < d; j++) {
                cross += (h[j] - mu0[j]) * inv_v[j] * dk[j];
            }
            row_out[k] = cross - 0.5 * D_sq[static_cast<std::size_t>(k)];
        }
    }
}

/// Inverse-CDF categorical: given cumulative-probability compatible uniform u ∈ [0,1)
/// and normalised probs, return the chosen index.
static int categorical_icdf(double u, const double* probs, int K) {
    double cum = 0.0;
    for (int i = 0; i < K; i++) {
        cum += probs[i];
        if (u < cum) return i;
    }
    return K - 1;
}

/// Draw d standard-normal samples into buf using rng (non-null).
static void draw_normal(std::mt19937& rng, int d, double* buf) {
    std::normal_distribution<double> nd;
    for (int i = 0; i < d; i++) {
        buf[i] = nd(rng);
    }
}


/// Find class index in m.labels; returns -1 if not found.
static int label_index(const CyphaInferModel& m, const std::string& label) {
    for (int i = 0; i < static_cast<int>(m.labels.size()); i++) {
        if (m.labels[static_cast<std::size_t>(i)] == label) return i;
    }
    return -1;
}

/// mu_k for class k: mu_world + D[k,:].
static void mu_k_for(const CyphaInferModel& m, int k, std::vector<double>& mu_k_out) {
    int d = m.d_latent;
    mu_k_out.resize(static_cast<std::size_t>(d));
    const double* dk = m.D.data() + static_cast<std::ptrdiff_t>(k) * d;
    for (int i = 0; i < d; i++) {
        mu_k_out[static_cast<std::size_t>(i)] = m.mu_world[static_cast<std::size_t>(i)] + dk[i];
    }
}

/// v0 from inv_v (element-wise reciprocal, clamped to kGenMinVar).
static std::vector<double> v0_from_inv_v(const CyphaInferModel& m) {
    int d = m.d_latent;
    std::vector<double> v(static_cast<std::size_t>(d));
    for (int i = 0; i < d; i++) {
        double iv = m.inv_v[static_cast<std::size_t>(i)];
        v[static_cast<std::size_t>(i)] = (iv > 0.0) ? 1.0 / iv : kGenMinVar;
    }
    return v;
}

/// std[i] = sqrt(max(v0[i], kGenMinVar)) * temperature.
static std::vector<double> gen_std(const std::vector<double>& v0, double temperature) {
    int d = static_cast<int>(v0.size());
    std::vector<double> s(static_cast<std::size_t>(d));
    for (int i = 0; i < d; i++) {
        s[static_cast<std::size_t>(i)] =
            std::sqrt(std::max(v0[static_cast<std::size_t>(i)], kGenMinVar)) * temperature;
    }
    return s;
}

/// Latent→field projection matching Python ``_to_field_dim``.
/// Writes fd-dim signal into sig_out.
static void to_field_dim(const CyphaInferModel& m, const double* h,
                         int d, int fd, std::vector<double>& sig_out) {
    bool ok = latent_to_field_signal(h, d, m.w_inject, fd, sig_out);
    if (!ok) {
        sig_out.assign(static_cast<std::size_t>(fd), 0.0);
        double norm_sq = 0.0;
        for (int i = 0; i < d; i++) norm_sq += h[i] * h[i];
        double inv_norm = 1.0 / (std::sqrt(norm_sq) + kGenEps);
        for (int i = 0; i < fd && i < d; i++) {
            sig_out[static_cast<std::size_t>(i)] = h[i] * inv_norm;
        }
    }
}

}  // namespace

// ── generate_class_gaussian ─────────────────────────────────────────────────

std::vector<std::vector<double>> generate_class_gaussian(
    const CyphaInferModel& m,
    const std::string&     label,
    int                    n,
    double                 temperature,
    std::mt19937*          rng,
    bool                   rejection,
    int                    max_candidates,
    const double*          z_override) {
    int d = m.d_latent;
    int K = static_cast<int>(m.labels.size());
    int k_idx = label_index(m, label);
    if (k_idx < 0)
        throw std::runtime_error("generate_class_gaussian: unknown label '" + label + "'");

    std::vector<double> v0 = v0_from_inv_v(m);
    std::vector<double> mu_k;
    mu_k_for(m, k_idx, mu_k);
    std::vector<double> std_vec = gen_std(v0, temperature);

    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(n));

    const bool do_rejection = (rejection && temperature > 1.0);

    if (!do_rejection) {
        // i.i.d. Gaussian
        std::vector<double> z_buf(static_cast<std::size_t>(d));
        for (int i = 0; i < n; i++) {
            const double* z_row = nullptr;
            if (z_override) {
                z_row = z_override + static_cast<std::ptrdiff_t>(i) * d;
            } else {
                draw_normal(*rng, d, z_buf.data());
                z_row = z_buf.data();
            }
            std::vector<double> h(static_cast<std::size_t>(d));
            for (int j = 0; j < d; j++) {
                h[static_cast<std::size_t>(j)] = mu_k[static_cast<std::size_t>(j)]
                    + z_row[j] * std_vec[static_cast<std::size_t>(j)];
            }
            result.push_back(std::move(h));
        }
        return result;
    }

    // Rejection path: draw max_candidates per sample, keep best LLR_k
    int C = max_candidates;
    std::vector<double> cands(static_cast<std::size_t>(C * d));
    std::vector<double> llr_mat;
    std::vector<double> z_buf_all(static_cast<std::size_t>(C * d));

    for (int i = 0; i < n; i++) {
        // Fill candidates
        const double* z_cands = nullptr;
        if (z_override) {
            z_cands = z_override + static_cast<std::ptrdiff_t>(i * C) * d;
        } else {
            draw_normal(*rng, C * d, z_buf_all.data());
            z_cands = z_buf_all.data();
        }
        for (int ci = 0; ci < C; ci++) {
            const double* zr = z_cands + static_cast<std::ptrdiff_t>(ci) * d;
            double* cand_row = cands.data() + static_cast<std::ptrdiff_t>(ci) * d;
            for (int j = 0; j < d; j++) {
                cand_row[j] = mu_k[static_cast<std::size_t>(j)]
                    + zr[j] * std_vec[static_cast<std::size_t>(j)];
            }
        }
        // Compute raw LLR
        raw_llr_matrix(cands.data(), C, d,
                       m.mu_world.data(), m.inv_v.data(),
                       m.D.data(), K, llr_mat);
        // Best candidate for label k_idx
        int best_ci = 0;
        double best_llr = -std::numeric_limits<double>::infinity();
        for (int ci = 0; ci < C; ci++) {
            double v = llr_mat[static_cast<std::size_t>(ci * K + k_idx)];
            if (v > best_llr) {
                best_llr = v;
                best_ci = ci;
            }
        }
        const double* best_row = cands.data() + static_cast<std::ptrdiff_t>(best_ci) * d;
        result.push_back(std::vector<double>(best_row, best_row + d));
    }
    return result;
}

// ── generate_conditioned ─────────────────────────────────────────────────────

std::vector<std::vector<double>> generate_conditioned(
    const CyphaInferModel& m,
    const std::string&     label,
    int                    n,
    double                 temperature,
    std::mt19937*          rng,
    const double*          z_override) {
    int d = m.d_latent;
    int fd = m.field_dim;
    int k_idx = label_index(m, label);
    if (k_idx < 0)
        throw std::runtime_error("generate_conditioned: unknown label '" + label + "'");

    // mu0_cond = mu_world + f_field_matrix @ field_h
    // f_field is (d × fd) row-major: mu0_cond[i] = mu_world[i] + f_field[i*fd..] . field_h
    std::vector<double> mu0_cond(static_cast<std::size_t>(d));
    if (!m.f_field.empty() && !m.field_h.empty() && fd > 0 &&
        static_cast<int>(m.f_field.size()) == d * fd &&
        static_cast<int>(m.field_h.size()) == fd) {
        // Guard: skip if field state norm is too large or non-finite
        double h_sq = 0.0;
        for (int i = 0; i < fd; i++) {
            h_sq += m.field_h[static_cast<std::size_t>(i)] * m.field_h[static_cast<std::size_t>(i)];
        }
        if (std::isfinite(h_sq) && h_sq <= 1e8) {
            for (int i = 0; i < d; i++) {
                double s = m.mu_world[static_cast<std::size_t>(i)];
                const double* frow = m.f_field.data() + static_cast<std::ptrdiff_t>(i) * fd;
                for (int fi = 0; fi < fd; fi++) {
                    s += frow[fi] * m.field_h[static_cast<std::size_t>(fi)];
                }
                mu0_cond[static_cast<std::size_t>(i)] = s;
            }
        } else {
            mu0_cond = m.mu_world;
        }
    } else {
        mu0_cond = m.mu_world;
    }

    // mu_k = mu0_cond + delta_k
    const double* dk = m.D.data() + static_cast<std::ptrdiff_t>(k_idx) * d;
    std::vector<double> v0 = v0_from_inv_v(m);
    std::vector<double> mu_k(static_cast<std::size_t>(d));
    for (int i = 0; i < d; i++) {
        mu_k[static_cast<std::size_t>(i)] = mu0_cond[static_cast<std::size_t>(i)] + dk[i];
    }
    std::vector<double> std_vec = gen_std(v0, temperature);

    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(n));
    std::vector<double> z_buf(static_cast<std::size_t>(d));
    for (int i = 0; i < n; i++) {
        const double* z_row = nullptr;
        if (z_override) {
            z_row = z_override + static_cast<std::ptrdiff_t>(i) * d;
        } else {
            draw_normal(*rng, d, z_buf.data());
            z_row = z_buf.data();
        }
        std::vector<double> h(static_cast<std::size_t>(d));
        for (int j = 0; j < d; j++) {
            h[static_cast<std::size_t>(j)] = mu_k[static_cast<std::size_t>(j)]
                + z_row[j] * std_vec[static_cast<std::size_t>(j)];
        }
        result.push_back(std::move(h));
    }
    return result;
}

// ── generate_langevin ────────────────────────────────────────────────────────

std::vector<std::vector<double>> generate_langevin(
    const CyphaInferModel& m,
    const std::string&     label,
    int                    n,
    int                    n_steps,
    double                 step_size,
    double                 temperature,
    std::mt19937*          rng,
    const double*          z_init_override,
    const double*          z_noise_override) {
    int d = m.d_latent;
    int k_idx = label_index(m, label);
    if (k_idx < 0)
        throw std::runtime_error("generate_langevin: unknown label '" + label + "'");

    std::vector<double> v0 = v0_from_inv_v(m);
    std::vector<double> mu_k;
    mu_k_for(m, k_idx, mu_k);

    const double* dk = m.D.data() + static_cast<std::ptrdiff_t>(k_idx) * d;
    // grad_llr_k = inv_v * delta_k
    std::vector<double> grad_llr_k(static_cast<std::size_t>(d));
    for (int i = 0; i < d; i++) {
        grad_llr_k[static_cast<std::size_t>(i)] =
            m.inv_v[static_cast<std::size_t>(i)] * dk[i];
    }
    // v_prior = v0 * max(T², 0.1)
    double T2 = temperature * temperature;
    double v_scale = std::max(T2, 0.1);
    std::vector<double> v_prior(static_cast<std::size_t>(d));
    for (int i = 0; i < d; i++) {
        v_prior[static_cast<std::size_t>(i)] =
            v0[static_cast<std::size_t>(i)] * v_scale;
    }
    double sqrt_2step_T = std::sqrt(2.0 * step_size * std::max(T2, kGenMinVar));
    // sqrt(v0) for initialisation
    std::vector<double> sqrt_v0(static_cast<std::size_t>(d));
    for (int i = 0; i < d; i++) {
        sqrt_v0[static_cast<std::size_t>(i)] = std::sqrt(v0[static_cast<std::size_t>(i)]);
    }

    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(n));
    std::vector<double> z_init_buf(static_cast<std::size_t>(d));
    std::vector<double> z_noise_buf(static_cast<std::size_t>(d));

    for (int i = 0; i < n; i++) {
        // Init: h = mu_k + z_init * sqrt(v0) * 0.5
        const double* z_init = nullptr;
        if (z_init_override) {
            z_init = z_init_override + static_cast<std::ptrdiff_t>(i) * d;
        } else {
            draw_normal(*rng, d, z_init_buf.data());
            z_init = z_init_buf.data();
        }
        std::vector<double> h(static_cast<std::size_t>(d));
        for (int j = 0; j < d; j++) {
            h[static_cast<std::size_t>(j)] = mu_k[static_cast<std::size_t>(j)]
                + z_init[j] * sqrt_v0[static_cast<std::size_t>(j)] * 0.5;
        }

        for (int s = 0; s < n_steps; s++) {
            const double* z_noise = nullptr;
            if (z_noise_override) {
                z_noise = z_noise_override
                    + static_cast<std::ptrdiff_t>(i) * n_steps * d
                    + static_cast<std::ptrdiff_t>(s) * d;
            } else {
                draw_normal(*rng, d, z_noise_buf.data());
                z_noise = z_noise_buf.data();
            }
            for (int j = 0; j < d; j++) {
                double vp = v_prior[static_cast<std::size_t>(j)] + kGenEps;
                double grad = grad_llr_k[static_cast<std::size_t>(j)]
                    - (h[static_cast<std::size_t>(j)] - mu_k[static_cast<std::size_t>(j)]) / vp;
                h[static_cast<std::size_t>(j)] =
                    h[static_cast<std::size_t>(j)]
                    + step_size * grad
                    + sqrt_2step_T * z_noise[j];
            }
        }
        result.push_back(std::move(h));
    }
    return result;
}

// ── generate_boundary ────────────────────────────────────────────────────────

std::vector<std::vector<double>> generate_boundary(
    const CyphaInferModel& m,
    const std::string&     label_a,
    const std::string&     label_b,
    int                    n,
    double                 alpha,
    double                 temperature,
    std::mt19937*          rng,
    const double*          z_override) {
    int d = m.d_latent;
    int ka = label_index(m, label_a);
    int kb = label_index(m, label_b);
    if (ka < 0) throw std::runtime_error("generate_boundary: unknown label '" + label_a + "'");
    if (kb < 0) throw std::runtime_error("generate_boundary: unknown label '" + label_b + "'");

    std::vector<double> v0 = v0_from_inv_v(m);
    std::vector<double> mu_a, mu_b;
    mu_k_for(m, ka, mu_a);
    mu_k_for(m, kb, mu_b);
    const double* dma = m.D.data() + static_cast<std::ptrdiff_t>(ka) * d;
    const double* dmb = m.D.data() + static_cast<std::ptrdiff_t>(kb) * d;

    // normal = (dm_a - dm_b) * inv_v
    std::vector<double> normal(static_cast<std::size_t>(d));
    double n_sq = 0.0;
    for (int i = 0; i < d; i++) {
        normal[static_cast<std::size_t>(i)] =
            (dma[i] - dmb[i]) * m.inv_v[static_cast<std::size_t>(i)];
        n_sq += normal[static_cast<std::size_t>(i)] * normal[static_cast<std::size_t>(i)];
    }
    n_sq += kGenEps;

    // target_dot = 0.5 * (||dm_a||²_inv_v - ||dm_b||²_inv_v)
    double dot_a = 0.0, dot_b = 0.0;
    for (int i = 0; i < d; i++) {
        dot_a += dma[i] * dma[i] * m.inv_v[static_cast<std::size_t>(i)];
        dot_b += dmb[i] * dmb[i] * m.inv_v[static_cast<std::size_t>(i)];
    }
    double target_dot = 0.5 * (dot_a - dot_b);

    // mu_interp
    std::vector<double> mu_interp(static_cast<std::size_t>(d));
    double eff_temp = std::max(temperature, kGenEps);
    std::vector<double> std_vec = gen_std(v0, eff_temp);
    for (int i = 0; i < d; i++) {
        mu_interp[static_cast<std::size_t>(i)] =
            (1.0 - alpha) * mu_a[static_cast<std::size_t>(i)]
            + alpha * mu_b[static_cast<std::size_t>(i)];
    }

    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(n));
    std::vector<double> z_buf(static_cast<std::size_t>(d));
    for (int i = 0; i < n; i++) {
        const double* z_row = nullptr;
        if (z_override) {
            z_row = z_override + static_cast<std::ptrdiff_t>(i) * d;
        } else {
            draw_normal(*rng, d, z_buf.data());
            z_row = z_buf.data();
        }
        std::vector<double> h(static_cast<std::size_t>(d));
        for (int j = 0; j < d; j++) {
            h[static_cast<std::size_t>(j)] = mu_interp[static_cast<std::size_t>(j)]
                + z_row[j] * std_vec[static_cast<std::size_t>(j)];
        }
        // Project onto decision hyperplane
        double curr = 0.0;
        for (int j = 0; j < d; j++) {
            curr += (h[static_cast<std::size_t>(j)] - m.mu_world[static_cast<std::size_t>(j)])
                * normal[static_cast<std::size_t>(j)];
        }
        double t = (curr - target_dot) / n_sq;
        for (int j = 0; j < d; j++) {
            h[static_cast<std::size_t>(j)] -= t * normal[static_cast<std::size_t>(j)];
        }
        result.push_back(std::move(h));
    }
    return result;
}

// ── generate_ood ─────────────────────────────────────────────────────────────

std::vector<std::vector<double>> generate_ood(
    const CyphaInferModel& m,
    int                    n,
    int                    n_candidates,
    std::mt19937*          rng,
    const double*          z_override) {
    int d = m.d_latent;
    int K = static_cast<int>(m.labels.size());
    std::vector<double> v0 = v0_from_inv_v(m);

    // std = sqrt(max(v0, MIN_VAR)) * 2.0
    std::vector<double> std_vec(static_cast<std::size_t>(d));
    for (int i = 0; i < d; i++) {
        std_vec[static_cast<std::size_t>(i)] =
            std::sqrt(std::max(v0[static_cast<std::size_t>(i)], kGenMinVar)) * 2.0;
    }

    // H = mu0 + z * std  (n_candidates × d)
    std::vector<double> H(static_cast<std::size_t>(n_candidates * d));
    std::vector<double> z_buf;
    const double* z_row = nullptr;
    if (z_override) {
        z_row = z_override;
    } else {
        z_buf.resize(static_cast<std::size_t>(n_candidates * d));
        draw_normal(*rng, n_candidates * d, z_buf.data());
        z_row = z_buf.data();
    }
    for (int i = 0; i < n_candidates; i++) {
        for (int j = 0; j < d; j++) {
            H[static_cast<std::size_t>(i * d + j)] =
                m.mu_world[static_cast<std::size_t>(j)]
                + z_row[static_cast<std::ptrdiff_t>(i) * d + j]
                  * std_vec[static_cast<std::size_t>(j)];
        }
    }

    // ll_world = _batch_logpdf(H, mu0, v0)
    std::vector<double> ll_world;
    batch_logpdf(H.data(), n_candidates, d, m.mu_world.data(), v0.data(), ll_world);

    // max_llr[i] over all classes
    std::vector<double> max_llr(static_cast<std::size_t>(n_candidates),
                                -std::numeric_limits<double>::infinity());

    std::vector<double> mu_k_buf(static_cast<std::size_t>(d));
    std::vector<double> ll_k;
    for (int k = 0; k < K; k++) {
        mu_k_for(m, k, mu_k_buf);
        batch_logpdf(H.data(), n_candidates, d, mu_k_buf.data(), v0.data(), ll_k);
        double u_k = m.v_mean / (m.n_obs[static_cast<std::size_t>(k)] + 1.0);
        for (int i = 0; i < n_candidates; i++) {
            double llr_ki = ll_k[static_cast<std::size_t>(i)]
                - ll_world[static_cast<std::size_t>(i)] - u_k;
            if (llr_ki > max_llr[static_cast<std::size_t>(i)])
                max_llr[static_cast<std::size_t>(i)] = llr_ki;
        }
    }

    // Sort by max_llr ascending, take first n
    std::vector<int> order(static_cast<std::size_t>(n_candidates));
    for (int i = 0; i < n_candidates; i++) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(),
              [&max_llr](int a, int b) {
                  return max_llr[static_cast<std::size_t>(a)]
                       < max_llr[static_cast<std::size_t>(b)];
              });

    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n && i < n_candidates; i++) {
        int ci = order[static_cast<std::size_t>(i)];
        const double* row = H.data() + static_cast<std::ptrdiff_t>(ci) * d;
        result.push_back(std::vector<double>(row, row + d));
    }
    return result;
}

// ── generate_mdl_ball ────────────────────────────────────────────────────────

std::vector<std::vector<double>> generate_mdl_ball(
    const CyphaInferModel& m,
    const std::string&     label,
    int                    n,
    double                 radius,
    std::mt19937*          rng,
    const double*          z_dir_override,
    const double*          u_mag_override) {
    int d = m.d_latent;
    int k_idx = label_index(m, label);
    if (k_idx < 0)
        throw std::runtime_error("generate_mdl_ball: unknown label '" + label + "'");

    std::vector<double> v0 = v0_from_inv_v(m);
    std::vector<double> mu_k;
    mu_k_for(m, k_idx, mu_k);
    std::vector<double> std_vec(static_cast<std::size_t>(d));
    for (int i = 0; i < d; i++) {
        std_vec[static_cast<std::size_t>(i)] =
            std::sqrt(std::max(v0[static_cast<std::size_t>(i)], kGenMinVar));
    }

    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(n));
    std::vector<double> z_buf(static_cast<std::size_t>(d));
    std::vector<double> u_buf(1);
    std::uniform_real_distribution<double> ud(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        const double* z_dir = nullptr;
        double u_mag = 0.0;
        if (z_dir_override) {
            z_dir = z_dir_override + static_cast<std::ptrdiff_t>(i) * d;
        } else {
            draw_normal(*rng, d, z_buf.data());
            z_dir = z_buf.data();
        }
        if (u_mag_override) {
            u_mag = u_mag_override[i];
        } else {
            u_mag = ud(*rng);
        }

        // raw_fr = z_dir / std_vec;  dir_fr = raw_fr / ||raw_fr||
        std::vector<double> raw_fr(static_cast<std::size_t>(d));
        double fr_norm_sq = 0.0;
        for (int j = 0; j < d; j++) {
            raw_fr[static_cast<std::size_t>(j)] =
                z_dir[j] / std_vec[static_cast<std::size_t>(j)];
            fr_norm_sq += raw_fr[static_cast<std::size_t>(j)] * raw_fr[static_cast<std::size_t>(j)];
        }
        double fr_norm = std::sqrt(fr_norm_sq) + kGenEps;
        // r = radius * u_mag^(1/d)
        double r = radius * std::pow(u_mag, 1.0 / static_cast<double>(d));
        std::vector<double> h(static_cast<std::size_t>(d));
        for (int j = 0; j < d; j++) {
            double dir_fr_j = raw_fr[static_cast<std::size_t>(j)] / fr_norm;
            double delta_j  = dir_fr_j * r * std_vec[static_cast<std::size_t>(j)];
            h[static_cast<std::size_t>(j)] = mu_k[static_cast<std::size_t>(j)] + delta_j;
        }
        result.push_back(std::move(h));
    }
    return result;
}

// ── generate_ancestral ───────────────────────────────────────────────────────

std::vector<AncestralSample> generate_ancestral(
    const CyphaInferModel& m,
    int                    n,
    double                 temperature,
    std::mt19937*          rng,
    const double*          u_class_override,
    const double*          z_override) {
    int K = static_cast<int>(m.labels.size());
    int d = m.d_latent;
    if (K == 0) throw std::runtime_error("generate_ancestral: no classes in model");

    // freq = max(n_obs, 1) / sum
    std::vector<double> freq(static_cast<std::size_t>(K));
    double freq_sum = 0.0;
    for (int k = 0; k < K; k++) {
        freq[static_cast<std::size_t>(k)] = std::max(m.n_obs[static_cast<std::size_t>(k)], 1.0);
        freq_sum += freq[static_cast<std::size_t>(k)];
    }
    double eff_T = temperature + kGenEps;
    std::vector<double> probs(static_cast<std::size_t>(K));
    double prob_sum = 0.0;
    for (int k = 0; k < K; k++) {
        probs[static_cast<std::size_t>(k)] =
            std::pow(freq[static_cast<std::size_t>(k)] / freq_sum, 1.0 / eff_T);
        prob_sum += probs[static_cast<std::size_t>(k)];
    }
    for (int k = 0; k < K; k++) {
        probs[static_cast<std::size_t>(k)] /= prob_sum;
    }

    std::vector<double> v0 = v0_from_inv_v(m);
    std::vector<double> std_vec(static_cast<std::size_t>(d));
    for (int i = 0; i < d; i++) {
        std_vec[static_cast<std::size_t>(i)] =
            std::sqrt(std::max(v0[static_cast<std::size_t>(i)], kGenMinVar));
    }

    std::vector<AncestralSample> result;
    result.reserve(static_cast<std::size_t>(n));
    std::vector<double> z_buf(static_cast<std::size_t>(d));
    std::uniform_real_distribution<double> ud(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        double u_c = u_class_override ? u_class_override[i] : ud(*rng);
        int ki = categorical_icdf(u_c, probs.data(), K);

        const double* z_row = nullptr;
        if (z_override) {
            z_row = z_override + static_cast<std::ptrdiff_t>(i) * d;
        } else {
            draw_normal(*rng, d, z_buf.data());
            z_row = z_buf.data();
        }
        std::vector<double> mu_k_buf;
        mu_k_for(m, ki, mu_k_buf);
        std::vector<double> h(static_cast<std::size_t>(d));
        for (int j = 0; j < d; j++) {
            h[static_cast<std::size_t>(j)] = mu_k_buf[static_cast<std::size_t>(j)]
                + z_row[j] * std_vec[static_cast<std::size_t>(j)];
        }
        result.push_back({m.labels[static_cast<std::size_t>(ki)], std::move(h)});
    }
    return result;
}

// ── predict_next_probs ───────────────────────────────────────────────────────

std::vector<double> predict_next_probs(const CyphaInferModel& m,
                                       const std::string& /*current_label*/) {
    int K = static_cast<int>(m.labels.size());
    std::vector<double> probs(static_cast<std::size_t>(K));
    if (K == 0) return probs;

    const std::string& last = m.ctx_last_label;
    if (last.empty()) {
        double inv_k = 1.0 / static_cast<double>(K);
        std::fill(probs.begin(), probs.end(), inv_k);
        return probs;
    }

    // scores[k] = cooccur[last][k] + 5.0 * mid_trans[last][k] + 1e-3
    auto it_co = m.cooccur.find(last);
    auto it_tr = m.mid_trans.find(last);
    double total = 0.0;
    for (int k = 0; k < K; k++) {
        double co_val = 0.0, tr_val = 0.0;
        if (it_co != m.cooccur.end()) {
            auto jt = it_co->second.find(m.labels[static_cast<std::size_t>(k)]);
            if (jt != it_co->second.end()) co_val = jt->second;
        }
        if (it_tr != m.mid_trans.end()) {
            auto jt = it_tr->second.find(m.labels[static_cast<std::size_t>(k)]);
            if (jt != it_tr->second.end()) tr_val = jt->second;
        }
        probs[static_cast<std::size_t>(k)] = co_val + 5.0 * tr_val + 1e-3;
        total += probs[static_cast<std::size_t>(k)];
    }
    if (total > 0.0) {
        for (auto& p : probs) p /= total;
    }
    return probs;
}

// ── rollout ──────────────────────────────────────────────────────────────────

std::vector<RolloutStep> rollout(
    CyphaInferModel&   m,
    const std::string& seed_label,
    int                n_steps,
    double             temperature,
    double             exploration,
    std::mt19937*      rng,
    const double*      z_generate,
    const double*      u_transition) {
    int d = m.d_latent;
    int fd = m.field_dim;
    int K = static_cast<int>(m.labels.size());
    if (K == 0) throw std::runtime_error("rollout: no classes in model");

    std::vector<double> v0 = v0_from_inv_v(m);
    std::vector<double> std_vec = gen_std(v0, temperature);
    std::vector<double> uniform_p(static_cast<std::size_t>(K), 1.0 / static_cast<double>(K));

    std::vector<double> z_buf(static_cast<std::size_t>(d));
    std::vector<double> u_buf(1);
    std::vector<double> field_sig;
    std::vector<double> mixed(static_cast<std::size_t>(K));
    std::uniform_real_distribution<double> ud(0.0, 1.0);

    std::string current = seed_label;
    std::vector<RolloutStep> result;
    result.reserve(static_cast<std::size_t>(n_steps));

    for (int step = 0; step < n_steps; step++) {
        int k_cur = label_index(m, current);
        if (k_cur < 0) k_cur = 0;  // fallback

        // Sample h from current class (no rejection — temperature typically ≤ 1)
        const double* z_row = nullptr;
        if (z_generate) {
            z_row = z_generate + static_cast<std::ptrdiff_t>(step) * d;
        } else {
            draw_normal(*rng, d, z_buf.data());
            z_row = z_buf.data();
        }
        std::vector<double> mu_k_buf;
        mu_k_for(m, k_cur, mu_k_buf);
        std::vector<double> h(static_cast<std::size_t>(d));
        for (int j = 0; j < d; j++) {
            h[static_cast<std::size_t>(j)] = mu_k_buf[static_cast<std::size_t>(j)]
                + z_row[j] * std_vec[static_cast<std::size_t>(j)];
        }
        result.push_back({current, h});

        // Update context
        context_record_step(m, current, /*correct=*/true);

        // Field inject + evolve (matches train_step_vector pattern: copy h before evolve)
        if (!m.field_w_t.empty() &&
            static_cast<int>(m.field_a_eff.size()) == fd * fd &&
            fd > 0 && !m.field_h.empty()) {
            to_field_dim(m, h.data(), d, fd, field_sig);
            nig_field_inject(m.field_h, field_sig.data(), fd, 0.05);
            std::vector<double> h_old = m.field_h;
            nig_field_evolve(m.field_a_eff, fd, h_old.data(), m.field_h);
            m.field_step += 1;
        }

        // predict_next: set ctx_last_label to current (context_record_step already did this)
        std::vector<double> raw_probs = predict_next_probs(m, current);
        double raw_sum = kGenEps;
        for (double p : raw_probs) raw_sum += p;
        double mixed_sum = 0.0;
        for (int k = 0; k < K; k++) {
            mixed[static_cast<std::size_t>(k)] =
                (1.0 - exploration) * raw_probs[static_cast<std::size_t>(k)] / raw_sum
                + exploration * uniform_p[static_cast<std::size_t>(k)];
            mixed_sum += mixed[static_cast<std::size_t>(k)];
        }
        for (int k = 0; k < K; k++) {
            mixed[static_cast<std::size_t>(k)] /= mixed_sum;
        }

        // Sample next class
        double u_t = u_transition ? u_transition[step] : ud(*rng);
        int next_k = categorical_icdf(u_t, mixed.data(), K);
        current = m.labels[static_cast<std::size_t>(next_k)];
    }
    return result;
}

// ── generate_from_observation ────────────────────────────────────────────────

std::vector<std::vector<double>> generate_from_observation(
    const CyphaInferModel& m,
    const double*          h_obs,
    const std::string&     label,
    int                    n,
    double                 temperature,
    int                    n_steps,
    std::mt19937*          rng,
    const double*          z_noise_override) {
    int d = m.d_latent;
    int k_idx = label_index(m, label);
    if (k_idx < 0)
        throw std::runtime_error("generate_from_observation: unknown label '" + label + "'");

    std::vector<double> mu_k;
    mu_k_for(m, k_idx, mu_k);

    const double lr = temperature * 0.05;
    const double noise_scale = std::sqrt(2.0 * lr);

    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(n));
    std::vector<double> z_noise_buf(static_cast<std::size_t>(d));

    for (int i = 0; i < n; i++) {
        std::vector<double> h(h_obs, h_obs + d);
        for (int s = 0; s < n_steps; s++) {
            const double* z_noise = nullptr;
            if (z_noise_override) {
                z_noise = z_noise_override
                    + static_cast<std::ptrdiff_t>(i) * n_steps * d
                    + static_cast<std::ptrdiff_t>(s) * d;
            } else {
                draw_normal(*rng, d, z_noise_buf.data());
                z_noise = z_noise_buf.data();
            }
            for (int j = 0; j < d; j++) {
                double grad = -(h[static_cast<std::size_t>(j)] - mu_k[static_cast<std::size_t>(j)])
                    * m.inv_v[static_cast<std::size_t>(j)];
                h[static_cast<std::size_t>(j)] =
                    h[static_cast<std::size_t>(j)] + lr * grad + noise_scale * z_noise[j];
            }
        }
        result.push_back(std::move(h));
    }
    return result;
}

// ── generate_retrieval_augmented ─────────────────────────────────────────────

std::vector<std::vector<double>> generate_retrieval_augmented(
    const CyphaInferModel& m,
    const double*          query_x,
    const double*          database_x,
    int                    n_db,
    int                    input_dim,
    int                    k_neighbors,
    int                    n,
    double                 temperature,
    int                    n_steps,
    const CyphaInferOptions& opt,
    std::mt19937*          rng,
    const double*          z_noise_override,
    const double*          z_init_override,
    const double*          z_langevin_noise_override) {
    const int d = m.d_latent;

    std::vector<RetrieveHit> hits =
        retrieve_from_x(m, query_x, database_x, n_db, input_dim, k_neighbors, opt);

    if (hits.empty()) {
        std::vector<double> h_q;
        batch_encode(m, query_x, 1, h_q);
        const InferAtHResult inf = infer_at_h(m, h_q.data(), opt);
        if (inf.label == kUnknownLabel || label_index(m, inf.label) < 0) {
            std::vector<std::vector<double>> fallback;
            fallback.reserve(static_cast<std::size_t>(n));
            for (int i = 0; i < n; i++) {
                fallback.push_back(std::vector<double>(h_q.begin(), h_q.end()));
            }
            return fallback;
        }
        constexpr double kFallbackStep = 0.05;
        return generate_langevin(m, inf.label, n, n_steps, kFallbackStep, temperature, rng,
                                 z_init_override, z_langevin_noise_override);
    }

    std::vector<double> h_query;
    batch_encode(m, query_x, 1, h_query);

    std::vector<double> h_db;
    batch_encode(m, database_x, n_db, h_db);

    const int k_actual = static_cast<int>(hits.size());
    std::vector<double> dists2(static_cast<std::size_t>(k_actual), 0.0);
    for (int j = 0; j < k_actual; j++) {
        const int idx = hits[static_cast<std::size_t>(j)].index;
        const double* h_i = h_db.data() + static_cast<std::ptrdiff_t>(idx) * d;
        double s = 0.0;
        for (int t = 0; t < d; t++) {
            double diff = h_i[t] - h_query[static_cast<std::size_t>(t)];
            s += diff * diff;
        }
        dists2[static_cast<std::size_t>(j)] = s;
    }

    double d_max = 0.0;
    for (int j = 0; j < k_actual; j++) {
        d_max = std::max(d_max, dists2[static_cast<std::size_t>(j)]);
    }
    const double denom = std::max(d_max * 0.1, kGenEps);

    std::vector<double> weights(static_cast<std::size_t>(k_actual), 0.0);
    double w_sum = 0.0;
    for (int j = 0; j < k_actual; j++) {
        weights[static_cast<std::size_t>(j)] =
            std::exp(-dists2[static_cast<std::size_t>(j)] / denom);
        w_sum += weights[static_cast<std::size_t>(j)];
    }
    w_sum += kGenEps;

    std::vector<double> h_anchor(static_cast<std::size_t>(d), 0.0);
    for (int j = 0; j < k_actual; j++) {
        const int idx = hits[static_cast<std::size_t>(j)].index;
        const double* h_i = h_db.data() + static_cast<std::ptrdiff_t>(idx) * d;
        const double w = weights[static_cast<std::size_t>(j)] / w_sum;
        for (int t = 0; t < d; t++) {
            h_anchor[static_cast<std::size_t>(t)] += w * h_i[t];
        }
    }

    int nearest_j = 0;
    double best_d = dists2[0];
    for (int j = 1; j < k_actual; j++) {
        if (dists2[static_cast<std::size_t>(j)] < best_d) {
            best_d = dists2[static_cast<std::size_t>(j)];
            nearest_j = j;
        }
    }
    const std::string& label = hits[static_cast<std::size_t>(nearest_j)].predicted_label;

    return generate_from_observation(m, h_anchor.data(), label, n, temperature, n_steps, rng,
                                     z_noise_override);
}

}  // namespace cypha
