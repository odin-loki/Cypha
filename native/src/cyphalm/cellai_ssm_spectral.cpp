#include "cypha/cyphalm/cellai_ssm.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace cypha::cyphalm {
namespace {

constexpr double kPi = 3.14159265358979323846;

int next_pow2(int n) {
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

void fft_inplace(std::vector<std::complex<double>>& a, bool inverse) {
  const int n = static_cast<int>(a.size());
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(a[static_cast<std::size_t>(i)], a[static_cast<std::size_t>(j)]);
    }
  }

  for (int len = 2; len <= n; len <<= 1) {
    const double ang = (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    for (int i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (int j = 0; j < len / 2; ++j) {
        auto& u = a[static_cast<std::size_t>(i + j)];
        auto& v = a[static_cast<std::size_t>(i + j + len / 2)];
        const std::complex<double> t = w * v;
        v = u - t;
        u = u + t;
        w *= wlen;
      }
    }
  }

  if (inverse) {
    for (auto& v : a) {
      v /= static_cast<double>(n);
    }
  }
}

std::vector<std::complex<double>> rfft(const std::vector<double>& x) {
  const int n = static_cast<int>(x.size());
  const int nfft = next_pow2(n);
  std::vector<std::complex<double>> buf(static_cast<std::size_t>(nfft), std::complex<double>(0.0, 0.0));
  for (int i = 0; i < n; ++i) {
    buf[static_cast<std::size_t>(i)] = std::complex<double>(x[static_cast<std::size_t>(i)], 0.0);
  }
  fft_inplace(buf, false);

  const int out_len = n / 2 + 1;
  std::vector<std::complex<double>> out(static_cast<std::size_t>(out_len));
  for (int k = 0; k < out_len; ++k) {
    if (k < nfft) {
      out[static_cast<std::size_t>(k)] = buf[static_cast<std::size_t>(k)];
    } else {
      out[static_cast<std::size_t>(k)] = std::complex<double>(0.0, 0.0);
    }
  }
  return out;
}

std::vector<double> irfft(const std::vector<std::complex<double>>& spec, int n) {
  const int nfft = next_pow2(n);
  std::vector<std::complex<double>> buf(static_cast<std::size_t>(nfft), std::complex<double>(0.0, 0.0));

  const int spec_len = static_cast<int>(spec.size());
  for (int k = 0; k < spec_len && k < nfft; ++k) {
    buf[static_cast<std::size_t>(k)] = spec[static_cast<std::size_t>(k)];
  }
  for (int k = spec_len; k < nfft; ++k) {
    const int mirror = nfft - k;
    if (mirror > 0 && mirror < spec_len) {
      buf[static_cast<std::size_t>(k)] = std::conj(spec[static_cast<std::size_t>(mirror)]);
    }
  }

  fft_inplace(buf, true);

  std::vector<double> out(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = buf[static_cast<std::size_t>(i)].real();
  }
  return out;
}

}  // namespace

std::vector<double> spectral_step(const std::vector<double>& state,
                                  const std::vector<double>& kernel) {
  if (state.size() != kernel.size()) {
    throw std::invalid_argument("spectral_step: state and kernel length mismatch");
  }
  if (state.empty()) {
    return {};
  }
  if (static_cast<int>(state.size()) > 512) {
    throw std::invalid_argument("spectral_step: D must be <= 512");
  }

  const auto s_fft = rfft(state);
  const auto k_fft = rfft(kernel);
  const int out_len = static_cast<int>(s_fft.size());
  std::vector<std::complex<double>> prod(static_cast<std::size_t>(out_len));
  for (int i = 0; i < out_len; ++i) {
    prod[static_cast<std::size_t>(i)] = s_fft[static_cast<std::size_t>(i)] * k_fft[static_cast<std::size_t>(i)];
  }
  return irfft(prod, static_cast<int>(state.size()));
}

}  // namespace cypha::cyphalm
