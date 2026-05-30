#include "cypha/nig_gig_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "cypha/bessel_table.hpp"

namespace cypha {

namespace {

constexpr double kEps = 1e-8;

double np_interp_k2k1(double x) {
  using cypha::detail::kBesselK2K1;
  using cypha::detail::kBesselN;
  using cypha::detail::kBesselX0;
  using cypha::detail::kBesselX1;
  if (x <= kBesselX0) {
    return kBesselK2K1[0];
  }
  if (x >= kBesselX1) {
    return kBesselK2K1[kBesselN - 1];
  }
  const double step = (kBesselX1 - kBesselX0) / static_cast<double>(kBesselN - 1);
  double pos = (x - kBesselX0) / step;
  std::size_t i = static_cast<std::size_t>(pos);
  if (i >= kBesselN - 1) {
    i = kBesselN - 2;
  }
  double t = pos - static_cast<double>(i);
  return kBesselK2K1[i] * (1.0 - t) + kBesselK2K1[i + 1] * t;
}

double np_interp_k0k1(double x) {
  using cypha::detail::kBesselK0K1;
  using cypha::detail::kBesselN;
  using cypha::detail::kBesselX0;
  using cypha::detail::kBesselX1;
  if (x <= kBesselX0) {
    return kBesselK0K1[0];
  }
  if (x >= kBesselX1) {
    return kBesselK0K1[kBesselN - 1];
  }
  const double step = (kBesselX1 - kBesselX0) / static_cast<double>(kBesselN - 1);
  double pos = (x - kBesselX0) / step;
  std::size_t i = static_cast<std::size_t>(pos);
  if (i >= kBesselN - 1) {
    i = kBesselN - 2;
  }
  double t = pos - static_cast<double>(i);
  return kBesselK0K1[i] * (1.0 - t) + kBesselK0K1[i + 1] * t;
}

}  // namespace

double gig_e_inv_v_lam_neg1(double chi0, double psi) {
  if (chi0 < kEps || psi < kEps) {
    return psi / std::max(chi0, kEps);
  }
  double chi_g = std::max(chi0, kEps);
  double x = std::sqrt(chi_g * psi);
  if (x < 1e-6) {
    return psi / chi_g;
  }
  double chi_b = chi_g;
  double x_b = x;
  if (x_b <= 120.0) {
    double xt = std::clamp(x_b, cypha::detail::kBesselX0, cypha::detail::kBesselX1);
    double ratio = np_interp_k2k1(xt);
    return std::sqrt(psi / chi_b) * ratio;
  }
  return psi / chi_b;
}

double gig_e_v_lam_neg1(double chi0, double psi) {
  if (chi0 < kEps || psi < kEps) {
    return chi0 / std::max(psi, kEps);
  }
  double chi_g = std::max(chi0, kEps);
  double x = std::sqrt(chi_g * psi);
  if (x < 1e-6) {
    return chi_g / std::max(psi, kEps);
  }
  double chi_b = chi_g;
  double x_b = x;
  if (x_b <= 120.0) {
    double xt = std::clamp(x_b, cypha::detail::kBesselX0, cypha::detail::kBesselX1);
    double ratio = np_interp_k0k1(xt);
    return std::sqrt(chi_b / psi) * ratio;
  }
  return std::sqrt(chi_b / std::max(psi, kEps));
}

double nig_adapt_chi_impl(double chi, double psi, double innovation_sq, double R, double alpha) {
  double chi_post = chi + innovation_sq / std::max(R, kEps);
  double ev = gig_e_v_lam_neg1(chi_post, psi);
  return std::clamp(alpha * ev, 1e-4, 1e3);
}

double nig_r_eff_scalar(double mp, double r, double chi, double psi) {
  mp = std::max(mp, 0.0);
  double chi_post = chi + mp / std::max(r, kEps);
  double e_inv = gig_e_inv_v_lam_neg1(chi_post, psi);
  return r / std::max(e_inv, kEps);
}

}  // namespace cypha
