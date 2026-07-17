// CTest: DIFRegressor predict_batch + expert EMA vs fixtures/regression_m4/sidecar.json
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/regression_stub.hpp"

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

std::string read_all(const char* path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  std::stringstream b;
  b << f.rdbuf();
  return b.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: regression_m4_golden <fixtures/regression_m4/sidecar.json>\n";
      return 2;
    }
    if (cypha::regression::native_regression_milestone() < 5) {
      std::cerr << "native_regression_milestone expected >= 5 (MKE route softmax + RLS + two-stage)\n";
      return 1;
    }
    auto j = nlohmann::json::parse(read_all(argv[1]));
    constexpr double kTol = 1e-12;
    constexpr double kTolRls = 1e-9;

    const auto& batch = j.at("batch");
    const int n = batch.at("n").get<int>();
    const int k = batch.at("k").get<int>();
    const int d = batch.at("d").get<int>();
    std::vector<double> probs = batch.at("probs").get<std::vector<double>>();
    std::vector<double> mu_mat = batch.at("mu_mat").get<std::vector<double>>();
    std::vector<double> var_vec = batch.at("var_vec").get<std::vector<double>>();
    std::vector<double> y_out(static_cast<std::size_t>(n * d));
    std::vector<double> unc_out(static_cast<std::size_t>(n));
    cypha::regression::predict_mixture_batch(probs.data(), n, k, d, mu_mat.data(), var_vec.data(), y_out.data(),
                                             unc_out.data());
    std::vector<double> exp_y = batch.at("expected_y").get<std::vector<double>>();
    std::vector<double> exp_unc = batch.at("expected_unc").get<std::vector<double>>();
    for (std::size_t i = 0; i < exp_y.size(); ++i) {
      if (!near_eq(y_out[i], exp_y[i], kTol)) {
        std::cerr << "batch y mismatch at " << i << " got " << y_out[i] << " expected " << exp_y[i] << "\n";
        return 1;
      }
    }
    for (std::size_t i = 0; i < exp_unc.size(); ++i) {
      if (!near_eq(unc_out[i], exp_unc[i], kTol)) {
        std::cerr << "batch unc mismatch at " << i << " got " << unc_out[i] << " expected " << exp_unc[i] << "\n";
        return 1;
      }
    }

    const auto& ema = j.at("ema");
    const int ed = ema.at("d").get<int>();
    const double lr = ema.at("lr").get<double>();
    std::vector<double> mu = ema.at("mu_before").get<std::vector<double>>();
    double var_ema = ema.at("var_before").get<double>();
    int n_updates = ema.at("n_before").get<int>();
    std::vector<double> y = ema.at("y").get<std::vector<double>>();
    cypha::regression::expert_target_ema_step(mu, var_ema, n_updates, y.data(), ed, lr);
    if (static_cast<int>(mu.size()) != ed) {
      std::cerr << "ema: bad mu size\n";
      return 1;
    }
    std::vector<double> mu_exp = ema.at("mu_after").get<std::vector<double>>();
    for (std::size_t i = 0; i < mu_exp.size(); ++i) {
      if (!near_eq(mu[i], mu_exp[i], kTol)) {
        std::cerr << "ema mu mismatch at " << i << "\n";
        return 1;
      }
    }
    if (!near_eq(var_ema, ema.at("var_after").get<double>(), kTol)) {
      std::cerr << "ema var mismatch\n";
      return 1;
    }
    if (n_updates != ema.at("n_after").get<int>()) {
      std::cerr << "ema n_updates mismatch\n";
      return 1;
    }

    const auto& ein = j.at("ema_init");
    const int id = ein.at("d").get<int>();
    const double ilr = ein.at("lr").get<double>();
    std::vector<double> imu;
    double ivar = 0.0;
    int inu = 0;
    std::vector<double> iy = ein.at("y").get<std::vector<double>>();
    cypha::regression::expert_target_ema_step(imu, ivar, inu, iy.data(), id, ilr);
    std::vector<double> imu_exp = ein.at("mu_after").get<std::vector<double>>();
    for (std::size_t i = 0; i < imu_exp.size(); ++i) {
      if (!near_eq(imu[i], imu_exp[i], kTol)) {
        std::cerr << "ema_init mu mismatch\n";
        return 1;
      }
    }
    if (!near_eq(ivar, ein.at("var_after").get<double>(), kTol)) {
      std::cerr << "ema_init var mismatch\n";
      return 1;
    }
    if (inu != ein.at("n_after").get<int>()) {
      std::cerr << "ema_init n_updates mismatch\n";
      return 1;
    }

    const auto& rr = j.at("rff_rls");
    const int Dr = rr.at("D").get<int>();
    std::vector<double> phi_r = rr.at("phi").get<std::vector<double>>();
    std::vector<double> w_r = rr.at("w_before").get<std::vector<double>>();
    double b_r = rr.at("b_before").get<double>();
    std::vector<double> P_r = rr.at("P_before").get<std::vector<double>>();
    const double y_raw_r = rr.at("y_raw").get<double>();
    const double y_mean_r = rr.at("y_mean").get<double>();
    const double y_std_r = rr.at("y_std").get<double>();
    if (static_cast<int>(phi_r.size()) != Dr || static_cast<int>(w_r.size()) != Dr ||
        static_cast<int>(P_r.size()) != (Dr + 1) * (Dr + 1)) {
      std::cerr << "rff_rls: bad sizes\n";
      return 1;
    }
    const double loss = cypha::regression::rff_rls_train_step(phi_r.data(), Dr, w_r.data(), &b_r, P_r.data(),
                                                              y_raw_r, y_mean_r, y_std_r);
    if (!near_eq(loss, rr.at("expected_loss").get<double>(), kTolRls)) {
      std::cerr << "rff_rls loss mismatch\n";
      return 1;
    }
    std::vector<double> w_exp = rr.at("w_after").get<std::vector<double>>();
    std::vector<double> P_exp = rr.at("P_after").get<std::vector<double>>();
    for (std::size_t i = 0; i < w_exp.size(); ++i) {
      if (!near_eq(w_r[i], w_exp[i], kTolRls)) {
        std::cerr << "rff_rls w mismatch at " << i << "\n";
        return 1;
      }
    }
    if (!near_eq(b_r, rr.at("b_after").get<double>(), kTolRls)) {
      std::cerr << "rff_rls b mismatch\n";
      return 1;
    }
    for (std::size_t i = 0; i < P_exp.size(); ++i) {
      if (!near_eq(P_r[i], P_exp[i], kTolRls)) {
        std::cerr << "rff_rls P mismatch at " << i << "\n";
        return 1;
      }
    }

    const auto& mk = j.at("mke_rls");
    const int Dm = mk.at("D").get<int>();
    std::vector<double> phi_m = mk.at("phi").get<std::vector<double>>();
    const double pi_m = mk.at("pi").get<double>();
    const double gh_m = mk.at("gh_scale").get<double>();
    const double err_m = mk.at("err").get<double>();
    const double ff_m = mk.at("forgetting_factor").get<double>();
    std::vector<double> w_m = mk.at("w_before").get<std::vector<double>>();
    std::vector<double> P_m = mk.at("P_before").get<std::vector<double>>();
    if (static_cast<int>(phi_m.size()) != Dm || static_cast<int>(w_m.size()) != Dm ||
        static_cast<int>(P_m.size()) != Dm * Dm) {
      std::cerr << "mke_rls: bad sizes\n";
      return 1;
    }
    cypha::regression::mke_expert_rls_scalar_step(phi_m.data(), Dm, pi_m, gh_m, err_m, ff_m, w_m.data(),
                                                  P_m.data());
    std::vector<double> w_ma = mk.at("w_after").get<std::vector<double>>();
    std::vector<double> P_ma = mk.at("P_after").get<std::vector<double>>();
    for (std::size_t i = 0; i < w_ma.size(); ++i) {
      if (!near_eq(w_m[i], w_ma[i], kTolRls)) {
        std::cerr << "mke_rls w mismatch at " << i << "\n";
        return 1;
      }
    }
    for (std::size_t i = 0; i < P_ma.size(); ++i) {
      if (!near_eq(P_m[i], P_ma[i], kTolRls)) {
        std::cerr << "mke_rls P mismatch at " << i << "\n";
        return 1;
      }
    }

    const auto& mff = mk.at("forgetting_case");
    const double ff2 = mff.at("forgetting_factor").get<double>();
    std::vector<double> w_m2 = mff.at("w_before").get<std::vector<double>>();
    std::vector<double> P_m2 = mff.at("P_before").get<std::vector<double>>();
    cypha::regression::mke_expert_rls_scalar_step(phi_m.data(), Dm, pi_m, 1.0, err_m, ff2, w_m2.data(),
                                                  P_m2.data());
    std::vector<double> w_m2e = mff.at("w_after").get<std::vector<double>>();
    std::vector<double> P_m2e = mff.at("P_after").get<std::vector<double>>();
    for (std::size_t i = 0; i < w_m2e.size(); ++i) {
      if (!near_eq(w_m2[i], w_m2e[i], kTolRls)) {
        std::cerr << "mke_rls ff w mismatch at " << i << "\n";
        return 1;
      }
    }
    for (std::size_t i = 0; i < P_m2e.size(); ++i) {
      if (!near_eq(P_m2[i], P_m2e[i], kTolRls)) {
        std::cerr << "mke_rls ff P mismatch at " << i << "\n";
        return 1;
      }
    }

    const auto& mskip = mk.at("low_pi_noop");
    const int Ds = mskip.at("D").get<int>();
    std::vector<double> phi_s = mskip.at("phi").get<std::vector<double>>();
    std::vector<double> w_s = mskip.at("w_before").get<std::vector<double>>();
    std::vector<double> P_s = mskip.at("P_before").get<std::vector<double>>();
    std::vector<double> w_sb = w_s;
    std::vector<double> P_sb = P_s;
    cypha::regression::mke_expert_rls_scalar_step(phi_s.data(), Ds, mskip.at("pi").get<double>(),
                                                  mskip.at("gh_scale").get<double>(), mskip.at("err").get<double>(),
                                                  mskip.at("forgetting_factor").get<double>(), w_s.data(),
                                                  P_s.data());
    for (std::size_t i = 0; i < w_sb.size(); ++i) {
      if (!near_eq(w_s[i], w_sb[i], kTol)) {
        std::cerr << "mke_rls low_pi should no-op w\n";
        return 1;
      }
    }
    for (std::size_t i = 0; i < P_sb.size(); ++i) {
      if (!near_eq(P_s[i], P_sb[i], kTol)) {
        std::cerr << "mke_rls low_pi should no-op P\n";
        return 1;
      }
    }

    const auto& ts = j.at("two_stage");
    const int Kts = ts.at("K").get<int>();
    const int dts = ts.at("d_in").get<int>();
    const int D2 = ts.at("D2").get<int>();
    std::vector<double> llr_ts = ts.at("llr").get<std::vector<double>>();
    std::vector<double> x_ts = ts.at("x").get<std::vector<double>>();
    std::vector<double> w1_ts = ts.at("w1").get<std::vector<double>>();
    std::vector<double> phi2_ts = ts.at("phi2").get<std::vector<double>>();
    std::vector<double> w2_ts = ts.at("w2").get<std::vector<double>>();
    if (static_cast<int>(llr_ts.size()) != Kts || static_cast<int>(x_ts.size()) != dts ||
        static_cast<int>(w1_ts.size()) != Kts + dts || static_cast<int>(phi2_ts.size()) != D2 ||
        static_cast<int>(w2_ts.size()) != D2) {
      std::cerr << "two_stage: bad sizes\n";
      return 1;
    }
    const double yp = cypha::regression::two_stage_dif_predict(
        llr_ts.data(), Kts, x_ts.data(), dts, w1_ts.data(), ts.at("b1").get<double>(), phi2_ts.data(), D2,
        w2_ts.data(), ts.at("b2").get<double>(), ts.at("y_mean").get<double>(), ts.at("y_std").get<double>());
    if (!near_eq(yp, ts.at("expected_y").get<double>(), kTolRls)) {
      std::cerr << "two_stage y mismatch\n";
      return 1;
    }

    const auto& rt = j.at("mke_route");
    const int Kmr = rt.at("K").get<int>();
    const double Tmr = rt.at("temperature").get<double>();
    const double eps_rt = rt.at("eps").get<double>();
    std::vector<double> llr_mr = rt.at("llr").get<std::vector<double>>();
    std::vector<double> mu_mr = rt.at("expert_mu").get<std::vector<double>>();
    std::vector<double> pr(static_cast<std::size_t>(Kmr));
    cypha::regression::router_softmax_from_llr(llr_mr.data(), Kmr, Tmr, eps_rt, pr.data());
    std::vector<double> p_exp = rt.at("expected_probs").get<std::vector<double>>();
    for (int i = 0; i < Kmr; ++i) {
      if (!near_eq(pr[static_cast<std::size_t>(i)], p_exp[static_cast<std::size_t>(i)], kTolRls)) {
        std::cerr << "mke_route prob mismatch at " << i << "\n";
        return 1;
      }
    }
    double entr = 0.0;
    const double yhat =
        cypha::regression::mke_scalar_predict_from_llr(llr_mr.data(), Kmr, Tmr, eps_rt, mu_mr.data(), &entr);
    if (!near_eq(yhat, rt.at("expected_y_hat").get<double>(), kTolRls)) {
      std::cerr << "mke_route y_hat mismatch\n";
      return 1;
    }
    if (!near_eq(entr, rt.at("expected_entropy").get<double>(), kTolRls)) {
      std::cerr << "mke_route entropy mismatch\n";
      return 1;
    }

    const auto& r10 = rt.at("k_gt_8");
    const int K10 = r10.at("K").get<int>();
    std::vector<double> llr10 = r10.at("llr").get<std::vector<double>>();
    std::vector<double> mu10 = r10.at("expert_mu").get<std::vector<double>>();
    std::vector<double> p10(static_cast<std::size_t>(K10));
    cypha::regression::router_softmax_from_llr(llr10.data(), K10, Tmr, eps_rt, p10.data());
    std::vector<double> p10e = r10.at("expected_probs").get<std::vector<double>>();
    for (int i = 0; i < K10; ++i) {
      if (!near_eq(p10[static_cast<std::size_t>(i)], p10e[static_cast<std::size_t>(i)], kTolRls)) {
        std::cerr << "mke_route k>8 prob mismatch at " << i << "\n";
        return 1;
      }
    }
    double e10 = 0.0;
    const double y10 =
        cypha::regression::mke_scalar_predict_from_llr(llr10.data(), K10, Tmr, eps_rt, mu10.data(), &e10);
    if (!near_eq(y10, r10.at("expected_y_hat").get<double>(), kTolRls)) {
      std::cerr << "mke_route k>8 y_hat mismatch\n";
      return 1;
    }
    if (!near_eq(e10, r10.at("expected_entropy").get<double>(), kTolRls)) {
      std::cerr << "mke_route k>8 entropy mismatch\n";
      return 1;
    }

    std::cout << "regression M4 parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "regression_m4_golden: " << e.what() << "\n";
    return 1;
  }
}
