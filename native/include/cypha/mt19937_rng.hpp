#pragma once

#include <vector>

/// NumPy ``np.random.default_rng(seed)`` (PCG64 + SeedSequence) for parity with Python.
namespace cypha {

class NumpyDefaultRng {
 public:
  explicit NumpyDefaultRng(int seed);
  ~NumpyDefaultRng();

  NumpyDefaultRng(const NumpyDefaultRng&) = delete;
  NumpyDefaultRng& operator=(const NumpyDefaultRng&) = delete;

  double normal(double loc, double scale);
  double uniform(double low, double high);
  /// NumPy ``Generator.integers(low, high)`` — half-open ``[low, high)``.
  int integers(int low, int high);
  /// NumPy ``Generator.permutation(n)`` — Fisher-Yates with ``integers(0, i+1)``.
  std::vector<int> permutation(int n);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace cypha
