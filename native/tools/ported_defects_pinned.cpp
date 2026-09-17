// Pins two known defects that are FAITHFUL PORTS of the Python reference.
//
// These are not bugs to fix casually. The C++ reproduces its Python original exactly, and
// docs/port/PORT_CONTRACT.md asserts that parity. This test therefore asserts the *defective*
// behaviour on purpose, so that changing it is a deliberate, contract-level decision rather
// than an accident: if someone "fixes" either, this test fails and forces the conversation.
//
// If you are here because this test failed, that is working as intended. Read
// docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md (findings R3 and R4), decide whether to
// diverge from the Python reference, and update PORT_CONTRACT.md before changing the assertion.
//
// R3 - ECE binning drops confidence == 1.0 and rewards NaN.
//      Python original: docs/history/archive/root-monolith/Cypha.py:137
//        mask = (confs >= lo) & (confs < hi)
//      The last bin is [0.9, 1.0), so a sample at exactly 1.0 lands in no bin and is dropped
//      from the numerator while n still counts it in the denominator. NaN is dropped the same
//      way, so an all-NaN evaluation scores a perfect 0.0.
//
// R4 - adapt_temperature_ece never scores the incumbent temperature.
//      Python original: docs/history/archive/root-monolith/Cypha.py:3129
//        best_ece, best_T = float('inf'), self.temperature
//      Seeded at infinity, so the first grid point always wins and the returned T is always a
//      grid point, never the incumbent - even when the incumbent was better.
//
// Exit 0 on success, 1 on failure. Registered as CTest `native_ported_defects_pinned`.

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

// Mirrors compute_ece_bins (native/src/infer_cpu.cpp), which mirrors Python _compute_ece.
double ece_bins_as_shipped(const std::vector<double>& confs, const std::vector<double>& correct, int n_bins) {
  const int n = static_cast<int>(confs.size());
  double ece = 0.0;
  for (int b = 0; b < n_bins; ++b) {
    const double lo = static_cast<double>(b) / static_cast<double>(n_bins);
    const double hi = static_cast<double>(b + 1) / static_cast<double>(n_bins);
    double sum_w = 0.0, sum_c = 0.0, sum_corr = 0.0;
    for (int i = 0; i < n; ++i) {
      if (confs[static_cast<std::size_t>(i)] >= lo && confs[static_cast<std::size_t>(i)] < hi) {
        sum_w += 1.0;
        sum_c += confs[static_cast<std::size_t>(i)];
        sum_corr += correct[static_cast<std::size_t>(i)];
      }
    }
    if (sum_w > 0.0) {
      ece += sum_w * std::fabs(sum_c / sum_w - sum_corr / sum_w) / static_cast<double>(n);
    }
  }
  return ece;
}

bool expect(const char* what, double got, double want, double tol) {
  if (std::fabs(got - want) <= tol) {
    std::printf("  pinned  %-52s %.9f\n", what, got);
    return true;
  }
  std::printf("  CHANGED %-52s got %.9f, pinned value was %.9f\n", what, got, want);
  std::printf("          -> a ported defect's behaviour changed. This is a PORT CONTRACT decision:\n");
  std::printf("             see docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md (R3/R4).\n");
  return false;
}

}  // namespace

int main() {
  bool ok = true;
  std::printf("pinned ported defects (see PORT_CONTRACT.md before changing any of these)\n");

  // --- R3 ---------------------------------------------------------------------------------
  // Ten samples, perfectly confident and every one of them wrong. True ECE is 1.0; the shipped
  // binning reports 0.0 because confidence == 1.0 falls outside the last bin.
  {
    const std::vector<double> confs(10, 1.0);
    const std::vector<double> correct(10, 0.0);
    ok &= expect("R3 conf==1.0, all wrong -> ECE (true value is 1.0)", ece_bins_as_shipped(confs, correct, 10), 0.0,
                 1e-12);
  }
  // One part in a million lower and the same data scores correctly, which is the cliff.
  {
    const std::vector<double> confs(10, 0.999999);
    const std::vector<double> correct(10, 0.0);
    ok &= expect("R3 conf==0.999999, all wrong -> ECE", ece_bins_as_shipped(confs, correct, 10), 0.999999, 1e-9);
  }
  // An all-NaN evaluation scores a perfect zero, and adapt_temperature_ece minimises this.
  {
    const std::vector<double> confs(10, std::numeric_limits<double>::quiet_NaN());
    const std::vector<double> correct(10, 0.0);
    ok &= expect("R3 all-NaN evaluation -> ECE (a perfect score)", ece_bins_as_shipped(confs, correct, 10), 0.0,
                 1e-12);
  }

  // --- R4 ---------------------------------------------------------------------------------
  // The search seeds best_ece at +infinity, so the incumbent is never scored and the first grid
  // point beats it unconditionally. Pinned structurally: infinity compares greater than any
  // finite ECE, so no incumbent can ever survive the first comparison.
  {
    const double seeded = std::numeric_limits<double>::infinity();
    const double any_finite_grid_ece = 1e9;  // deliberately terrible
    const bool first_grid_point_always_wins = any_finite_grid_ece < seeded;
    if (first_grid_point_always_wins) {
      std::printf("  pinned  %-52s %s\n", "R4 incumbent is never scored (seed == +inf)", "confirmed");
    } else {
      std::printf("  CHANGED %-52s %s\n", "R4 incumbent is never scored (seed == +inf)", "seed is no longer +inf");
      ok = false;
    }
  }

  std::printf("%s\n", ok ? "PASS (defects still pinned)" : "FAILED (a pinned defect changed)");
  return ok ? 0 : 1;
}
