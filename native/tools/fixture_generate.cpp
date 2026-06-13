// cypha_fixture_gen — native parity fixture regeneration (replaces removed Python generators).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/accel_backend.hpp"
#include "cypha/csv_ingest.hpp"
#include "cypha/cyphalm/cellai_ssm.hpp"
#include "cypha/cyphalm/char_lstm.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/embed_table.hpp"
#include "cypha/cyphalm/gria_lowrank.hpp"
#include "cypha/cyphalm/hybrid_blend.hpp"
#include "cypha/generation.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/mke_scalar_train_step.hpp"
#include "cypha/multilabel_dif.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/similarity_index.hpp"
#include "cypha/train_step_vector.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

constexpr const char* kUsage =
    "usage: cypha_fixture_gen --list\n"
    "       cypha_fixture_gen --fixture NAME --out DIR [--fixtures-root DIR]\n";

struct NativeParityBin {
  std::uint32_t n{0};
  std::uint32_t d{0};
  std::uint32_t k{0};
  std::vector<double> x_rowmajor;
  std::vector<double> llr_rowmajor;
};

struct Args {
  bool list_only{false};
  std::string fixture;
  fs::path out;
  fs::path fixtures_root;
};

void usage() { std::cerr << kUsage; }

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (k == "--list") {
      a.list_only = true;
    } else if (k == "--fixture") {
      a.fixture = need("--fixture");
    } else if (k == "--out") {
      a.out = need("--out");
    } else if (k == "--fixtures-root") {
      a.fixtures_root = need("--fixtures-root");
    } else if (k == "--help" || k == "-h") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + k);
    }
  }
  return a;
}

std::vector<std::uint8_t> read_all_bytes(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open " + path.string());
  }
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::string read_text(const fs::path& path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("cannot open " + path.string());
  }
  std::stringstream b;
  b << f.rdbuf();
  return b.str();
}

Json read_json(const fs::path& path) { return Json::parse(read_text(path)); }

void copy_fixture_file(const fs::path& src, const fs::path& dst) {
  if (!fs::exists(src)) {
    throw std::runtime_error("missing companion file " + src.string());
  }
  fs::create_directories(dst.parent_path());
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
}

NativeParityBin load_native_parity_bin(const fs::path& path) {
  const std::vector<std::uint8_t> sidecar = read_all_bytes(path);
  if (sidecar.size() < 8 + 4 * 5 + 8 * 2) {
    throw std::runtime_error("native_parity.bin too small");
  }
  if (std::memcmp(sidecar.data(), "CYPHNP01", 8) != 0) {
    throw std::runtime_error("native_parity.bin: bad magic (expected CYPHNP01)");
  }
  std::uint32_t ver = 0;
  std::uint32_t n = 0;
  std::uint32_t d = 0;
  std::uint32_t k = 0;
  std::uint32_t field_dim = 0;
  std::size_t o = 8;
  std::memcpy(&ver, sidecar.data() + o, 4);
  o += 4;
  std::memcpy(&n, sidecar.data() + o, 4);
  o += 4;
  std::memcpy(&d, sidecar.data() + o, 4);
  o += 4;
  std::memcpy(&k, sidecar.data() + o, 4);
  o += 4;
  std::memcpy(&field_dim, sidecar.data() + o, 4);
  o += 4;
  o += 16;  // temperature, eps
  if (ver != 1u && ver != 2u) {
    throw std::runtime_error("native_parity.bin: unsupported version");
  }
  const std::size_t core = o + static_cast<std::size_t>(d) * field_dim * 8u +
                           static_cast<std::size_t>(n) * d * 8u +
                           static_cast<std::size_t>(n) * k * 8u * 2u + static_cast<std::size_t>(n) * 8u;
  const std::size_t need = (ver == 2u) ? core + static_cast<std::size_t>(n) * 8u * 2u : core;
  if (sidecar.size() < need) {
    throw std::runtime_error("native_parity.bin: truncated payload");
  }
  o += static_cast<std::size_t>(d) * field_dim * 8u;  // F_field
  const double* x_in = reinterpret_cast<const double*>(sidecar.data() + o);
  o += static_cast<std::size_t>(n) * d * 8u;
  const double* exp_llr = reinterpret_cast<const double*>(sidecar.data() + o);

  NativeParityBin out;
  out.n = n;
  out.d = d;
  out.k = k;
  out.x_rowmajor.assign(x_in, x_in + static_cast<std::size_t>(n) * d);
  out.llr_rowmajor.assign(exp_llr, exp_llr + static_cast<std::size_t>(n) * k);
  return out;
}

std::vector<double> flatten_f_field(const Json& j) {
  std::vector<double> o;
  for (const auto& row : j) {
    for (const auto& v : row) {
      o.push_back(v.get<double>());
    }
  }
  return o;
}

cypha::CyphaInferModel load_infer_model(const fs::path& fixtures_root) {
  const fs::path cypha_path = fixtures_root / "reference.cypha";
  const fs::path ff_path = fixtures_root / "f_field.json";
  cypha::CNode root_node = cypha::load_cypha_file(cypha_path.string().c_str());
  const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
  const int fd = static_cast<int>(fh.shape[0]);
  const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
  const int d = static_cast<int>(enc.shape[0]);

  std::ifstream jf(ff_path);
  if (!jf) {
    throw std::runtime_error("cannot open f_field.json");
  }
  std::stringstream fj;
  fj << jf.rdbuf();
  std::vector<double> fflat = flatten_f_field(Json::parse(fj.str()));
  if (static_cast<int>(fflat.size()) != d * fd) {
    throw std::runtime_error("f_field size mismatch");
  }
  return cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);
}

std::pair<cypha::CyphaInferModel, cypha::CyphaDifMemoryState> load_infer_and_mem(const fs::path& cypha_path,
                                                                                  const fs::path& ff_path) {
  cypha::CNode root_node = cypha::load_cypha_file(cypha_path.string().c_str());
  const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
  const int fd = static_cast<int>(fh.shape[0]);
  const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
  const int d = static_cast<int>(enc.shape[0]);
  std::vector<double> fflat = flatten_f_field(read_json(ff_path));
  if (static_cast<int>(fflat.size()) != d * fd) {
    throw std::runtime_error("f_field size mismatch");
  }
  return {cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd),
          cypha::CyphaDifMemoryState::from_cypha_root(root_node, fflat.data(), fd)};
}

void write_json_pretty(const fs::path& path, const nlohmann::ordered_json& j) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write " + path.string());
  }
  out << j.dump(2) << '\n';
}

void generate_batch_llr(const fs::path& out_dir, const fs::path& fixtures_root) {
  const NativeParityBin bin = load_native_parity_bin(fixtures_root / "native_parity.bin");
  const cypha::CyphaInferModel infer = load_infer_model(fixtures_root);
  if (static_cast<int>(infer.labels.size()) != static_cast<int>(bin.k)) {
    throw std::runtime_error("label count mismatch between model and native_parity.bin");
  }
  if (infer.d_latent != static_cast<int>(bin.d)) {
    throw std::runtime_error("d_latent mismatch between model and native_parity.bin");
  }

  std::vector<double> llr;
  cypha::batch_llr_from_x(infer, bin.x_rowmajor.data(), static_cast<int>(bin.n), llr);
  constexpr double kTol = 1e-9;
  for (std::size_t i = 0; i < llr.size(); ++i) {
    if (std::abs(llr[i] - bin.llr_rowmajor[i]) > kTol) {
      throw std::runtime_error("batch_llr_from_x disagrees with native_parity.bin at index " +
                               std::to_string(i));
    }
  }

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  sidecar["fixture_schema"] = 1;
  sidecar["source"] = "fixtures/expected.npz (x_input, llr)";
  sidecar["n"] = bin.n;
  sidecar["d_in"] = bin.d;
  sidecar["K"] = bin.k;
  sidecar["x_rowmajor"] = bin.x_rowmajor;
  sidecar["expected_llr_rowmajor"] = bin.llr_rowmajor;
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (n=" << bin.n << " d_in=" << bin.d
            << " K=" << bin.k << ")\n";
}

void generate_memory_train(const fs::path& out_dir, const fs::path& fixtures_root) {
  const fs::path src_dir = fixtures_root / "memory_train";
  const Json src = read_json(src_dir / "sidecar.json");

  std::vector<double> h;
  for (const auto& v : src.at("h")) {
    h.push_back(v.get<double>());
  }
  std::vector<double> h_field;
  for (const auto& v : src.at("h_field")) {
    h_field.push_back(v.get<double>());
  }
  const std::string label = src.at("label").get<std::string>();
  const double temperature = src.at("temperature").get<double>();
  const double ood_sigma = src.at("ood_sigma").get<double>();
  const double world_lr = src.at("world_lr").get<double>();
  const double delta_lr = src.at("delta_lr").get<double>();
  const int field_dim = src.at("field_dim").get<int>();
  const std::vector<double> f_flat = flatten_f_field(src.at("f_field"));

  std::unordered_map<std::string, double> ctx;
  for (auto it = src.at("context_prior").begin(); it != src.at("context_prior").end(); ++it) {
    ctx[it.key()] = it.value().get<double>();
  }

  cypha::CNode before = cypha::load_cypha_file((src_dir / "before.cypha").string().c_str());
  cypha::CyphaDifMemoryState st =
      cypha::CyphaDifMemoryState::from_cypha_root(before, f_flat.data(), field_dim);
  const double loss = st.memory_train(h.data(), label, h_field.data(), ctx, temperature, ood_sigma, world_lr,
                                      delta_lr, nullptr);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  sidecar["h"] = src.at("h");
  sidecar["h_field"] = src.at("h_field");
  sidecar["label"] = label;
  sidecar["temperature"] = temperature;
  sidecar["ood_sigma"] = ood_sigma;
  sidecar["world_lr"] = world_lr;
  sidecar["delta_lr"] = delta_lr;
  sidecar["context_prior"] = src.at("context_prior");
  sidecar["expected_loss"] = loss;
  sidecar["f_field"] = src.at("f_field");
  sidecar["field_dim"] = field_dim;

  copy_fixture_file(src_dir / "before.cypha", out_dir / "before.cypha");
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (expected_loss=" << loss << ")\n";
}

void generate_preprocessor(const fs::path& out_dir, const fs::path& fixtures_root) {
  const fs::path src_dir = fixtures_root / "preprocessor";
  copy_fixture_file(src_dir / "preprocessor.json", out_dir / "preprocessor.json");

  const cypha::PreprocessorState pre =
      cypha::PreprocessorState::from_json_file((out_dir / "preprocessor.json").string().c_str());
  const Json src = read_json(src_dir / "sidecar.json");
  std::vector<double> x;
  for (const auto& v : src.at("x")) {
    x.push_back(v.get<double>());
  }
  const std::vector<double> expected = pre.transform_one(x);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  sidecar["x"] = src.at("x");
  sidecar["expected"] = expected;
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (output_dim=" << expected.size() << ")\n";
}

void generate_train_step_vector(const fs::path& out_dir, const fs::path& fixtures_root) {
  const fs::path src_dir = fixtures_root / "train_step_vector";
  const Json src = read_json(src_dir / "sidecar.json");

  auto [infer, mem] = load_infer_and_mem(fixtures_root / "reference.cypha", fixtures_root / "f_field.json");

  std::vector<double> x;
  for (const auto& v : src.at("x")) {
    x.push_back(v.get<double>());
  }
  const int d = infer.d_latent;
  if (static_cast<int>(x.size()) != d) {
    throw std::runtime_error("x dim mismatch");
  }
  const std::string label = src.at("label").get<std::string>();

  cypha::TrainStepParams tsp;
  tsp.enc_lr = src.value("enc_lr", 0.002);
  tsp.replay_ratio = src.value("replay_ratio", 0.30);
  tsp.replay_cap = src.value("replay_cap", 10000);
  tsp.align_every = src.value("align_every", 500);
  tsp.temp_recalib_every = src.value("temp_recalib_every", 0);

  const double world_lr = src.value("world_lr", 0.008);
  const double delta_lr = src.value("delta_lr", 0.05);
  const double ood_sigma = src.value("ood_sigma", 15.0);
  int total_steps_before = src.at("total_steps_before").get<int>();

  cypha::ReplayBuffer replay(tsp.replay_cap);
  std::mt19937 rng{424242};
  int enc_updates = 0;
  int step_count = total_steps_before;
  cypha::TrainStepExtras extras{};
  extras.total_steps = &step_count;
  extras.ood_sigma = nullptr;
  extras.llr_ema = nullptr;

  const double loss = cypha::dif_train_step_vector(infer, mem, replay, x.data(), d, label, world_lr, delta_lr,
                                                   world_lr, delta_lr, ood_sigma, tsp, rng, enc_updates, nullptr,
                                                   &extras);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  sidecar["x"] = src.at("x");
  sidecar["label"] = label;
  sidecar["expected_loss"] = loss;
  sidecar["total_steps_before"] = total_steps_before;
  sidecar["world_lr"] = world_lr;
  sidecar["delta_lr"] = delta_lr;
  sidecar["ood_sigma"] = ood_sigma;
  sidecar["enc_lr"] = tsp.enc_lr;
  sidecar["replay_ratio"] = tsp.replay_ratio;
  sidecar["replay_cap"] = tsp.replay_cap;
  sidecar["align_every"] = tsp.align_every;
  sidecar["temp_recalib_every"] = tsp.temp_recalib_every;
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (expected_loss=" << loss << ")\n";
}

void generate_regression_m4(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "regression_m4" / "sidecar.json");
  nlohmann::ordered_json out = nlohmann::ordered_json::object();

  const auto& batch = src.at("batch");
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
  nlohmann::ordered_json batch_out = nlohmann::ordered_json::object();
  batch_out["n"] = n;
  batch_out["k"] = k;
  batch_out["d"] = d;
  batch_out["probs"] = probs;
  batch_out["mu_mat"] = mu_mat;
  batch_out["var_vec"] = var_vec;
  batch_out["expected_y"] = y_out;
  batch_out["expected_unc"] = unc_out;
  out["batch"] = std::move(batch_out);

  const auto& ema = src.at("ema");
  const int ed = ema.at("d").get<int>();
  const double lr = ema.at("lr").get<double>();
  std::vector<double> mu = ema.at("mu_before").get<std::vector<double>>();
  double var_ema = ema.at("var_before").get<double>();
  int n_updates = ema.at("n_before").get<int>();
  std::vector<double> y = ema.at("y").get<std::vector<double>>();
  cypha::regression::expert_target_ema_step(mu, var_ema, n_updates, y.data(), ed, lr);
  nlohmann::ordered_json ema_out = nlohmann::ordered_json::object();
  ema_out["d"] = ed;
  ema_out["lr"] = lr;
  ema_out["mu_before"] = ema.at("mu_before");
  ema_out["var_before"] = ema.at("var_before");
  ema_out["n_before"] = ema.at("n_before");
  ema_out["y"] = y;
  ema_out["mu_after"] = mu;
  ema_out["var_after"] = var_ema;
  ema_out["n_after"] = n_updates;
  out["ema"] = std::move(ema_out);

  const auto& ein = src.at("ema_init");
  const int id = ein.at("d").get<int>();
  const double ilr = ein.at("lr").get<double>();
  std::vector<double> imu;
  double ivar = 0.0;
  int inu = 0;
  std::vector<double> iy = ein.at("y").get<std::vector<double>>();
  cypha::regression::expert_target_ema_step(imu, ivar, inu, iy.data(), id, ilr);
  nlohmann::ordered_json ein_out = nlohmann::ordered_json::object();
  ein_out["d"] = id;
  ein_out["lr"] = ilr;
  ein_out["y"] = iy;
  ein_out["mu_after"] = imu;
  ein_out["var_after"] = ivar;
  ein_out["n_after"] = inu;
  out["ema_init"] = std::move(ein_out);

  const auto& rr = src.at("rff_rls");
  const int Dr = rr.at("D").get<int>();
  std::vector<double> phi_r = rr.at("phi").get<std::vector<double>>();
  std::vector<double> w_r = rr.at("w_before").get<std::vector<double>>();
  double b_r = rr.at("b_before").get<double>();
  std::vector<double> P_r = rr.at("P_before").get<std::vector<double>>();
  const double loss_r = cypha::regression::rff_rls_train_step(phi_r.data(), Dr, w_r.data(), &b_r, P_r.data(),
                                                              rr.at("y_raw").get<double>(),
                                                              rr.at("y_mean").get<double>(),
                                                              rr.at("y_std").get<double>());
  nlohmann::ordered_json rr_out = nlohmann::ordered_json::object();
  rr_out["D"] = Dr;
  rr_out["phi"] = phi_r;
  rr_out["w_before"] = rr.at("w_before");
  rr_out["b_before"] = rr.at("b_before");
  rr_out["P_before"] = rr.at("P_before");
  rr_out["y_raw"] = rr.at("y_raw");
  rr_out["y_mean"] = rr.at("y_mean");
  rr_out["y_std"] = rr.at("y_std");
  rr_out["expected_loss"] = loss_r;
  rr_out["w_after"] = w_r;
  rr_out["b_after"] = b_r;
  rr_out["P_after"] = P_r;
  out["rff_rls"] = std::move(rr_out);

  const auto& mk = src.at("mke_rls");
  const int Dm = mk.at("D").get<int>();
  std::vector<double> phi_m = mk.at("phi").get<std::vector<double>>();
  const double pi_m = mk.at("pi").get<double>();
  const double gh_m = mk.at("gh_scale").get<double>();
  const double err_m = mk.at("err").get<double>();
  const double ff_m = mk.at("forgetting_factor").get<double>();
  std::vector<double> w_m = mk.at("w_before").get<std::vector<double>>();
  std::vector<double> P_m = mk.at("P_before").get<std::vector<double>>();
  cypha::regression::mke_expert_rls_scalar_step(phi_m.data(), Dm, pi_m, gh_m, err_m, ff_m, w_m.data(),
                                                P_m.data());
  nlohmann::ordered_json mk_out = nlohmann::ordered_json::object();
  mk_out["D"] = Dm;
  mk_out["phi"] = phi_m;
  mk_out["pi"] = pi_m;
  mk_out["gh_scale"] = gh_m;
  mk_out["err"] = err_m;
  mk_out["forgetting_factor"] = ff_m;
  mk_out["w_before"] = mk.at("w_before");
  mk_out["P_before"] = mk.at("P_before");
  mk_out["w_after"] = w_m;
  mk_out["P_after"] = P_m;

  const auto& mff = mk.at("forgetting_case");
  std::vector<double> w_m2 = mff.at("w_before").get<std::vector<double>>();
  std::vector<double> P_m2 = mff.at("P_before").get<std::vector<double>>();
  cypha::regression::mke_expert_rls_scalar_step(phi_m.data(), Dm, pi_m, 1.0, err_m,
                                                mff.at("forgetting_factor").get<double>(), w_m2.data(),
                                                P_m2.data());
  nlohmann::ordered_json mff_out = nlohmann::ordered_json::object();
  mff_out["forgetting_factor"] = mff.at("forgetting_factor");
  mff_out["w_before"] = mff.at("w_before");
  mff_out["P_before"] = mff.at("P_before");
  mff_out["w_after"] = w_m2;
  mff_out["P_after"] = P_m2;
  mk_out["forgetting_case"] = std::move(mff_out);
  mk_out["low_pi_noop"] = mk.at("low_pi_noop");
  out["mke_rls"] = std::move(mk_out);

  const auto& ts = src.at("two_stage");
  const int Kts = ts.at("K").get<int>();
  const int dts = ts.at("d_in").get<int>();
  const int D2 = ts.at("D2").get<int>();
  std::vector<double> llr_ts = ts.at("llr").get<std::vector<double>>();
  std::vector<double> x_ts = ts.at("x").get<std::vector<double>>();
  std::vector<double> w1_ts = ts.at("w1").get<std::vector<double>>();
  std::vector<double> phi2_ts = ts.at("phi2").get<std::vector<double>>();
  std::vector<double> w2_ts = ts.at("w2").get<std::vector<double>>();
  const double yp = cypha::regression::two_stage_dif_predict(
      llr_ts.data(), Kts, x_ts.data(), dts, w1_ts.data(), ts.at("b1").get<double>(), phi2_ts.data(), D2,
      w2_ts.data(), ts.at("b2").get<double>(), ts.at("y_mean").get<double>(), ts.at("y_std").get<double>());
  nlohmann::ordered_json ts_out = nlohmann::ordered_json::object();
  ts_out["K"] = Kts;
  ts_out["d_in"] = dts;
  ts_out["D2"] = D2;
  ts_out["llr"] = llr_ts;
  ts_out["x"] = x_ts;
  ts_out["w1"] = w1_ts;
  ts_out["b1"] = ts.at("b1");
  ts_out["phi2"] = phi2_ts;
  ts_out["w2"] = w2_ts;
  ts_out["b2"] = ts.at("b2");
  ts_out["y_mean"] = ts.at("y_mean");
  ts_out["y_std"] = ts.at("y_std");
  ts_out["expected_y"] = yp;
  out["two_stage"] = std::move(ts_out);

  const auto& rt = src.at("mke_route");
  const int Kmr = rt.at("K").get<int>();
  const double Tmr = rt.at("temperature").get<double>();
  const double eps_rt = rt.at("eps").get<double>();
  std::vector<double> llr_mr = rt.at("llr").get<std::vector<double>>();
  std::vector<double> mu_mr = rt.at("expert_mu").get<std::vector<double>>();
  std::vector<double> pr(static_cast<std::size_t>(Kmr));
  cypha::regression::router_softmax_from_llr(llr_mr.data(), Kmr, Tmr, eps_rt, pr.data());
  double entr = 0.0;
  const double yhat =
      cypha::regression::mke_scalar_predict_from_llr(llr_mr.data(), Kmr, Tmr, eps_rt, mu_mr.data(), &entr);
  nlohmann::ordered_json rt_out = nlohmann::ordered_json::object();
  rt_out["K"] = Kmr;
  rt_out["llr"] = llr_mr;
  rt_out["temperature"] = Tmr;
  rt_out["eps"] = eps_rt;
  rt_out["expert_mu"] = mu_mr;
  rt_out["expected_probs"] = pr;
  rt_out["expected_y_hat"] = yhat;
  rt_out["expected_entropy"] = entr;

  const auto& r10 = rt.at("k_gt_8");
  const int K10 = r10.at("K").get<int>();
  std::vector<double> llr10 = r10.at("llr").get<std::vector<double>>();
  std::vector<double> mu10 = r10.at("expert_mu").get<std::vector<double>>();
  std::vector<double> p10(static_cast<std::size_t>(K10));
  cypha::regression::router_softmax_from_llr(llr10.data(), K10, Tmr, eps_rt, p10.data());
  double e10 = 0.0;
  const double y10 =
      cypha::regression::mke_scalar_predict_from_llr(llr10.data(), K10, Tmr, eps_rt, mu10.data(), &e10);
  nlohmann::ordered_json r10_out = nlohmann::ordered_json::object();
  r10_out["K"] = K10;
  r10_out["llr"] = llr10;
  r10_out["expected_probs"] = p10;
  r10_out["expected_y_hat"] = y10;
  r10_out["expected_entropy"] = e10;
  r10_out["expert_mu"] = mu10;
  rt_out["k_gt_8"] = std::move(r10_out);
  out["mke_route"] = std::move(rt_out);

  write_json_pretty(out_dir / "sidecar.json", out);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (regression_m4 blocks)\n";
}

std::vector<double> flatten_json_mat(const Json& j) {
  std::vector<double> out;
  for (const auto& row : j) {
    for (const auto& v : row) {
      out.push_back(v.get<double>());
    }
  }
  return out;
}

std::vector<double> flatten_json_3d(const Json& j) {
  std::vector<double> out;
  for (const auto& block : j) {
    for (const auto& row : block) {
      for (const auto& v : row) {
        out.push_back(v.get<double>());
      }
    }
  }
  return out;
}

std::vector<std::vector<double>> json_to_matrix(const Json& j) {
  std::vector<std::vector<double>> m;
  for (const auto& row : j) {
    m.push_back(row.get<std::vector<double>>());
  }
  return m;
}

void generate_score_batch(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "score_batch" / "sidecar.json");
  const int n = src.at("n").get<int>();
  const int d = src.at("d").get<int>();
  const int K = src.at("K").get<int>();
  std::vector<double> F = src.at("F_rowmajor").get<std::vector<double>>();
  std::vector<double> W = src.at("W_enc_rowmajor").get<std::vector<double>>();
  std::vector<double> mu0 = src.at("mu0").get<std::vector<double>>();
  std::vector<double> inv_v = src.at("inv_v").get<std::vector<double>>();
  std::vector<double> D = src.at("D_rowmajor").get<std::vector<double>>();
  std::vector<double> D_sq = src.at("D_sq").get<std::vector<double>>();
  std::vector<double> u_k = src.at("u_k").get<std::vector<double>>();
  std::vector<double> ctx = src.at("ctx").get<std::vector<double>>();

  cypha::accel::init();
  std::vector<double> H(static_cast<std::size_t>(n * d));
  std::vector<double> llr(static_cast<std::size_t>(n * K));
  cypha::accel::batch_encode(F.data(), n, d, W.data(), H.data());
  cypha::accel::score_matrix(H.data(), n, d, K, mu0.data(), inv_v.data(), D.data(), D_sq.data(), u_k.data(),
                             ctx.data(), llr.data());
  cypha::accel::shutdown();

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  if (src.contains("source")) {
    sidecar["source"] = src.at("source");
  }
  sidecar["n"] = n;
  sidecar["d"] = d;
  sidecar["K"] = K;
  sidecar["F_rowmajor"] = F;
  sidecar["W_enc_rowmajor"] = W;
  sidecar["mu0"] = mu0;
  sidecar["inv_v"] = inv_v;
  sidecar["D_rowmajor"] = D;
  sidecar["D_sq"] = D_sq;
  sidecar["u_k"] = u_k;
  sidecar["ctx"] = ctx;
  sidecar["expected_H_rowmajor"] = H;
  sidecar["expected_LLR_rowmajor"] = llr;
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (score_batch n=" << n << ")\n";
}

cypha::KernelMemory load_kernel_from_json(const Json& st) {
  const int feat_dim = st.at("feat_dim").get<int>();
  const int M = st.at("M").get<int>();
  cypha::KernelMemory km(feat_dim, M, 0);
  std::map<std::string, std::vector<double>> weights;
  for (const auto& pr : st.at("weights").items()) {
    weights[pr.key()] = pr.value().get<std::vector<double>>();
  }
  std::vector<double> basis = st.at("basis_rowmajor").get<std::vector<double>>();
  km.load_state(st.at("n_basis").get<int>(), st.at("n_seen").get<int>(), basis.data(), M, weights);
  return km;
}

void generate_kernel_llr(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "kernel_llr" / "sidecar.json");
  const auto& st = src.at("kernel_state");
  cypha::KernelMemory km = load_kernel_from_json(st);
  const int feat_dim = st.at("feat_dim").get<int>();
  const int M = st.at("M").get<int>();
  const int n_test = src.at("n_test").get<int>();
  const int K = src.at("K").get<int>();
  const double blend = src.at("blend").get<double>();
  std::vector<std::string> labels = src.at("labels").get<std::vector<std::string>>();
  std::vector<double> h = src.at("h_test_rowmajor").get<std::vector<double>>();
  std::vector<double> linear = src.at("linear_llr_rowmajor").get<std::vector<double>>();

  std::vector<double> phi(M);
  std::vector<double> got_phi;
  std::vector<double> got_kernel;
  std::vector<double> kernel_row(static_cast<std::size_t>(K));
  for (int i = 0; i < n_test; ++i) {
    const double* row = h.data() + static_cast<std::size_t>(i * feat_dim);
    km.phi(row, phi);
    got_phi.insert(got_phi.end(), phi.begin(), phi.end());
    km.score_all(row, labels, kernel_row);
    got_kernel.insert(got_kernel.end(), kernel_row.begin(), kernel_row.end());
  }
  std::vector<double> blended = linear;
  for (int i = 0; i < n_test; ++i) {
    for (int k = 0; k < K; ++k) {
      const std::size_t idx = static_cast<std::size_t>(i * K + k);
      blended[idx] = (1.0 - blend) * linear[idx] + blend * got_kernel[idx];
    }
  }

  const auto& up = src.at("update_step");
  cypha::KernelMemory km_up = load_kernel_from_json(st);
  std::vector<double> h_up = up.at("h").get<std::vector<double>>();
  const std::string label = up.at("label").get<std::string>();
  std::vector<std::string> all_labels = up.at("all_labels").get<std::vector<std::string>>();
  const double lr = up.at("lr").get<double>();
  km_up.update(h_up.data(), label, all_labels, lr);
  nlohmann::ordered_json weights_after = nlohmann::ordered_json::object();
  for (const auto& pr : km_up.weights()) {
    weights_after[pr.first] = pr.second;
  }

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  if (src.contains("source")) {
    sidecar["source"] = src.at("source");
  }
  sidecar["labels"] = labels;
  sidecar["blend"] = blend;
  sidecar["kernel_state"] = st;
  sidecar["n_test"] = n_test;
  sidecar["K"] = K;
  sidecar["h_test_rowmajor"] = h;
  sidecar["linear_llr_rowmajor"] = linear;
  sidecar["expected_phi_rowmajor"] = got_phi;
  sidecar["expected_kernel_scores_rowmajor"] = got_kernel;
  sidecar["expected_blended_rowmajor"] = blended;
  nlohmann::ordered_json up_out = nlohmann::ordered_json::object();
  up_out["h"] = h_up;
  up_out["label"] = label;
  up_out["all_labels"] = all_labels;
  up_out["lr"] = lr;
  up_out["n_basis_before"] = up.at("n_basis_before");
  up_out["n_basis_after"] = km_up.n_basis();
  up_out["weights_after"] = weights_after;
  sidecar["update_step"] = std::move(up_out);
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (kernel_llr)\n";
}

void generate_gh_infer_deliberation(const fs::path& out_dir, const fs::path& fixtures_root) {
  const fs::path src_dir = fixtures_root / "gh_infer_deliberation";
  copy_fixture_file(src_dir / "reference.cypha", out_dir / "reference.cypha");
  copy_fixture_file(src_dir / "f_field.json", out_dir / "f_field.json");
  const Json src = read_json(src_dir / "sidecar.json");

  cypha::CNode root = cypha::load_cypha_file((out_dir / "reference.cypha").string().c_str());
  const cypha::CNode& fh = cypha::map_get_required(root, "field_h");
  const int fd = static_cast<int>(fh.shape[0]);
  const cypha::CNode& enc = cypha::map_get_required(root, "enc_W");
  const int d = static_cast<int>(enc.shape[0]);
  std::vector<double> fflat = flatten_f_field(read_json(out_dir / "f_field.json"));
  cypha::CyphaInferModel model = cypha::CyphaInferModel::from_root(root, fflat.data(), fd);

  const double alpha = src.value("nig_alpha", 0.98);
  double chi = 1.0;
  double psi = 1.0;
  nlohmann::ordered_json cases = nlohmann::ordered_json::array();
  for (const auto& c : src.at("cases")) {
    std::vector<double> x = c.at("x").get<std::vector<double>>();
    std::vector<double> H;
    cypha::batch_encode(model, x.data(), 1, H);
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    if (c.contains("name")) {
      oc["name"] = c.at("name");
    }
    oc["x"] = c.at("x");
    const bool use_gh = c.value("use_gh", false);
    oc["use_gh"] = use_gh;
    if (use_gh) {
      chi = c.value("chi", chi);
      psi = c.value("psi", psi);
      oc["chi"] = chi;
      oc["psi"] = psi;
      cypha::GhInferAtHResult gh = cypha::gh_infer_at_h(model, H.data(), chi, psi, alpha);
      oc["expected_label"] = gh.label;
      oc["expected_confidence"] = gh.confidence;
      oc["expected_r_eff"] = gh.r_eff;
      oc["expected_chi_new"] = gh.chi_new;
      oc["expected_psi_new"] = gh.psi_new;
      chi = gh.chi_new;
      psi = gh.psi_new;
    } else {
      cypha::CyphaInferOptions opt{};
      opt.deliberation_lo = c.value("deliberation_lo", model.deliberation_lo);
      opt.deliberation_hi = c.value("deliberation_hi", model.deliberation_hi);
      opt.use_field = true;
      cypha::InferAtHResult inf = cypha::infer_at_h(model, H.data(), opt);
      oc["expected_label"] = inf.label;
      oc["expected_confidence"] = inf.confidence;
      if (c.contains("deliberation_lo")) {
        oc["deliberation_lo"] = c.at("deliberation_lo");
      }
      if (c.contains("deliberation_hi")) {
        oc["deliberation_hi"] = c.at("deliberation_hi");
      }
    }
    cases.push_back(std::move(oc));
  }

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  if (src.contains("description")) {
    sidecar["description"] = src.at("description");
  }
  sidecar["reference_cypha"] = src.value("reference_cypha", "reference.cypha");
  sidecar["f_field_json"] = src.value("f_field_json", "f_field.json");
  if (src.contains("input_dim")) {
    sidecar["input_dim"] = src.at("input_dim");
  }
  if (src.contains("field_dim")) {
    sidecar["field_dim"] = src.at("field_dim");
  }
  if (src.contains("labels")) {
    sidecar["labels"] = src.at("labels");
  }
  sidecar["nig_alpha"] = alpha;
  sidecar["cases"] = std::move(cases);
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (gh_infer_deliberation)\n";
}

void generate_retrieval(const fs::path& out_dir, const fs::path& fixtures_root) {
  const fs::path src_dir = fixtures_root / "retrieval";
  copy_fixture_file(src_dir / "reference.cypha", out_dir / "reference.cypha");
  copy_fixture_file(src_dir / "f_field.json", out_dir / "f_field.json");
  const Json src = read_json(src_dir / "sidecar.json");

  cypha::CNode root = cypha::load_cypha_file((out_dir / "reference.cypha").string().c_str());
  const cypha::CNode& fh = cypha::map_get_required(root, "field_h");
  const int fd = static_cast<int>(fh.shape[0]);
  const cypha::CNode& enc = cypha::map_get_required(root, "enc_W");
  const int d = static_cast<int>(enc.shape[0]);
  const int input_dim = src.at("input_dim").get<int>();
  std::vector<double> fflat = flatten_f_field(read_json(out_dir / "f_field.json"));
  cypha::CyphaInferModel model = cypha::CyphaInferModel::from_root(root, fflat.data(), fd);
  cypha::CyphaInferOptions opt;
  opt.use_field = src.value("use_field", true);
  opt.deliberation_lo = src.value("deliberation_lo", cypha::kDeliberationLoDefault);
  opt.deliberation_hi = src.value("deliberation_hi", cypha::kDeliberationHiDefault);

  nlohmann::ordered_json cases = nlohmann::ordered_json::array();
  for (const auto& c : src.at("cases")) {
    std::vector<double> query = c.at("query_x").get<std::vector<double>>();
    std::vector<double> db_flat;
    for (const auto& row : c.at("database_x")) {
      auto rv = row.get<std::vector<double>>();
      db_flat.insert(db_flat.end(), rv.begin(), rv.end());
    }
    const int n_db = static_cast<int>(c.at("database_x").size());
    const int top_k = c.at("top_k").get<int>();
    std::optional<std::string> label_opt;
    if (c.contains("label") && !c.at("label").is_null()) {
      label_opt = c.at("label").get<std::string>();
    }
    auto hits = cypha::retrieve_from_x(model, query.data(), db_flat.data(), n_db, input_dim, top_k, opt, label_opt);
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    if (c.contains("name")) {
      oc["name"] = c.at("name");
    }
    oc["query_x"] = c.at("query_x");
    oc["database_x"] = c.at("database_x");
    oc["top_k"] = top_k;
    if (c.contains("label")) {
      oc["label"] = c.at("label");
    }
    nlohmann::ordered_json exp = nlohmann::ordered_json::array();
    for (const auto& h : hits) {
      nlohmann::ordered_json eh = nlohmann::ordered_json::object();
      eh["index"] = h.index;
      eh["log_likelihood"] = h.log_likelihood;
      eh["predicted_label"] = h.predicted_label;
      exp.push_back(std::move(eh));
    }
    oc["expected"] = std::move(exp);
    cases.push_back(std::move(oc));
  }

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  if (src.contains("description")) {
    sidecar["description"] = src.at("description");
  }
  sidecar["reference_cypha"] = src.value("reference_cypha", "reference.cypha");
  sidecar["f_field_json"] = src.value("f_field_json", "f_field.json");
  sidecar["input_dim"] = input_dim;
  sidecar["use_field"] = src.value("use_field", true);
  if (src.contains("deliberation_lo")) {
    sidecar["deliberation_lo"] = src.at("deliberation_lo");
  }
  if (src.contains("deliberation_hi")) {
    sidecar["deliberation_hi"] = src.at("deliberation_hi");
  }
  sidecar["cases"] = std::move(cases);
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (retrieval)\n";
}

void generate_embed_table(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "embed_table" / "sidecar.json");
  const auto run_case = [](const Json& cfg) -> nlohmann::ordered_json {
    const std::uint32_t vocab = cfg.at("vocab_size").get<std::uint32_t>();
    const std::uint32_t d_embed = cfg.at("d_embed").get<std::uint32_t>();
    const std::uint32_t seed = cfg.at("seed").get<std::uint32_t>();
    cypha::cyphalm::EmbedTable table(vocab, d_embed, seed);
    double table_sum = 0.0;
    for (double v : table.table()) {
      table_sum += v;
    }
    nlohmann::ordered_json out = nlohmann::ordered_json::object();
    out["vocab_size"] = vocab;
    out["d_embed"] = d_embed;
    out["seed"] = seed;
    if (cfg.contains("n")) {
      out["n"] = cfg.at("n");
    }
    if (cfg.contains("k")) {
      out["k"] = cfg.at("k");
    }
    if (cfg.contains("a")) {
      out["a"] = cfg.at("a");
    }
    if (cfg.contains("b")) {
      out["b"] = cfg.at("b");
    }
    out["table_sum"] = table_sum;
    nlohmann::ordered_json tokens = nlohmann::ordered_json::object();
    for (const auto& item : cfg.at("tokens").items()) {
      const std::uint32_t tid = static_cast<std::uint32_t>(std::stoul(item.key()));
      tokens[item.key()] = table.embed_vec(tid);
    }
    out["tokens"] = std::move(tokens);
    return out;
  };

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  if (src.contains("description")) {
    sidecar["description"] = src.at("description");
  }
  if (src.contains("cases")) {
    nlohmann::ordered_json cases = nlohmann::ordered_json::array();
    for (const auto& cfg : src.at("cases")) {
      cases.push_back(run_case(cfg));
    }
    sidecar["cases"] = std::move(cases);
  } else {
    sidecar = run_case(src);
    if (src.contains("fixture_schema")) {
      sidecar["fixture_schema"] = src.at("fixture_schema");
    }
    if (src.contains("description")) {
      sidecar["description"] = src.at("description");
    }
  }
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (embed_table)\n";
}

void generate_similarity_index(const fs::path& out_dir, const fs::path& fixtures_root) {
  const fs::path src_dir = fixtures_root / "similarity_index";
  copy_fixture_file(src_dir / "reference.cypha", out_dir / "reference.cypha");
  copy_fixture_file(src_dir / "f_field.json", out_dir / "f_field.json");
  const Json src = read_json(src_dir / "sidecar.json");

  cypha::CNode root_node = cypha::load_cypha_file((out_dir / "reference.cypha").string().c_str());
  const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
  const int fd = static_cast<int>(fh.shape[0]);
  const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
  const int d = static_cast<int>(enc.shape[0]);
  std::vector<double> fflat = flatten_f_field(read_json(out_dir / "f_field.json"));
  cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);
  cypha::SimilarityIndex idx(infer);
  for (const auto& ex : src.at("add_examples")) {
    std::vector<double> x = ex.at("x").get<std::vector<double>>();
    nlohmann::json md = ex.value("metadata", nlohmann::json(nullptr));
    idx.add(x.data(), d, md);
  }
  const double sim = idx.similarity(src.at("sim_x1").get<std::vector<double>>().data(), d,
                                    src.at("sim_x2").get<std::vector<double>>().data());
  auto hits = idx.query(src.at("query_x").get<std::vector<double>>().data(), d, src.at("query_k").get<int>(), true);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  sidecar["add_examples"] = src.at("add_examples");
  sidecar["sim_x1"] = src.at("sim_x1");
  sidecar["sim_x2"] = src.at("sim_x2");
  sidecar["expected_similarity"] = sim;
  sidecar["query_x"] = src.at("query_x");
  sidecar["query_k"] = src.at("query_k");
  nlohmann::ordered_json exp_q = nlohmann::ordered_json::array();
  for (const auto& h : hits) {
    nlohmann::ordered_json eh = nlohmann::ordered_json::object();
    eh["index"] = h.index;
    eh["similarity"] = h.similarity;
    exp_q.push_back(std::move(eh));
  }
  sidecar["expected_query"] = std::move(exp_q);
  if (src.contains("batch_query")) {
    const auto& bq = src.at("batch_query");
    const int n = bq.at("n").get<int>();
    std::vector<double> xs = bq.at("x_rowmajor").get<std::vector<double>>();
    const int k = bq.at("k").get<int>();
    auto batch = idx.query_batch(xs.data(), n, d, k);
    nlohmann::ordered_json exp_b = nlohmann::ordered_json::array();
    for (const auto& row : batch) {
      nlohmann::ordered_json er = nlohmann::ordered_json::array();
      for (const auto& h : row) {
        nlohmann::ordered_json eh = nlohmann::ordered_json::object();
        eh["index"] = h.index;
        eh["similarity"] = h.similarity;
        er.push_back(std::move(eh));
      }
      exp_b.push_back(std::move(er));
    }
    sidecar["batch_query"] = nlohmann::ordered_json::object();
    sidecar["batch_query"]["n"] = n;
    sidecar["batch_query"]["k"] = k;
    sidecar["batch_query"]["x_rowmajor"] = xs;
    sidecar["batch_query"]["expected"] = std::move(exp_b);
  }
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (similarity_index)\n";
}

void generate_multilabel_dif(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "multilabel_dif" / "sidecar.json");
  cypha::MultiLabelDifParams p;
  p.input_dim = src.at("d_in").get<int>();
  p.field_dim = src.at("field_dim").get<int>();
  p.world_lr = src.at("world_lr").get<double>();
  p.delta_lr = src.at("delta_lr").get<double>();
  p.ood_sigma = src.at("ood_sigma").get<double>();
  p.train.enc_lr = src.value("enc_lr", 0.0);
  p.train.replay_ratio = src.value("replay_ratio", 0.0);
  p.train.replay_cap = src.value("replay_cap", 10000);
  p.train.align_every = src.value("align_every", 500);
  if (src.contains("label_rng_seeds")) {
    for (auto it = src.at("label_rng_seeds").begin(); it != src.at("label_rng_seeds").end(); ++it) {
      p.label_rng_seeds[it.key()] = static_cast<std::uint32_t>(it.value().get<std::uint64_t>());
    }
  }
  const int d = p.input_dim;
  const int fd = p.field_dim;
  auto load_matrix_map = [&](const char* key, std::unordered_map<std::string, std::vector<double>>& out, int rows,
                             int cols) {
    if (!src.contains(key)) {
      return;
    }
    for (auto it = src.at(key).begin(); it != src.at(key).end(); ++it) {
      out[it.key()] = flatten_json_mat(it.value());
      if (static_cast<int>(out[it.key()].size()) != rows * cols) {
        throw std::runtime_error(std::string(key) + " size mismatch");
      }
    }
  };
  auto load_vec_map = [&](const char* key, std::unordered_map<std::string, std::vector<double>>& out, int n) {
    if (!src.contains(key)) {
      return;
    }
    for (auto it = src.at(key).begin(); it != src.at(key).end(); ++it) {
      out[it.key()] = it.value().get<std::vector<double>>();
      if (static_cast<int>(out[it.key()].size()) != n) {
        throw std::runtime_error(std::string(key) + " size mismatch");
      }
    }
  };
  load_matrix_map("initial_w_inject", p.initial_w_inject, fd, d);
  load_matrix_map("initial_field_w_t", p.initial_field_w_t, fd, fd);
  load_vec_map("initial_field_sr_vec", p.initial_field_sr_vec, fd);
  if (src.contains("initial_enc_w")) {
    for (auto it = src.at("initial_enc_w").begin(); it != src.at("initial_enc_w").end(); ++it) {
      p.initial_enc_w[it.key()] = flatten_json_mat(it.value());
    }
  }
  cypha::MultiLabelDif mlf(p);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  sidecar["d_in"] = p.input_dim;
  sidecar["field_dim"] = p.field_dim;
  sidecar["world_lr"] = p.world_lr;
  sidecar["delta_lr"] = p.delta_lr;
  sidecar["ood_sigma"] = p.ood_sigma;
  sidecar["enc_lr"] = p.train.enc_lr;
  sidecar["replay_ratio"] = p.train.replay_ratio;
  sidecar["replay_cap"] = p.train.replay_cap;
  sidecar["align_every"] = p.train.align_every;
  if (src.contains("label_rng_seeds")) {
    sidecar["label_rng_seeds"] = src.at("label_rng_seeds");
  }
  if (src.contains("initial_w_inject")) {
    sidecar["initial_w_inject"] = src.at("initial_w_inject");
  }
  if (src.contains("initial_field_w_t")) {
    sidecar["initial_field_w_t"] = src.at("initial_field_w_t");
  }
  if (src.contains("initial_field_sr_vec")) {
    sidecar["initial_field_sr_vec"] = src.at("initial_field_sr_vec");
  }
  if (src.contains("initial_enc_w")) {
    sidecar["initial_enc_w"] = src.at("initial_enc_w");
  }

  nlohmann::ordered_json steps_out = nlohmann::ordered_json::array();
  for (const auto& step : src.at("train_steps")) {
    std::vector<double> x = step.at("x").get<std::vector<double>>();
    std::unordered_map<std::string, bool> labels;
    for (auto it = step.at("labels").begin(); it != step.at("labels").end(); ++it) {
      labels[it.key()] = it.value().get<bool>();
    }
    auto losses = mlf.train_step(x.data(), p.input_dim, labels);
    nlohmann::ordered_json so = nlohmann::ordered_json::object();
    so["x"] = step.at("x");
    so["labels"] = step.at("labels");
    nlohmann::ordered_json el = nlohmann::ordered_json::object();
    for (const auto& pr : losses) {
      el[pr.first] = pr.second;
    }
    so["expected_losses"] = std::move(el);
    steps_out.push_back(std::move(so));
  }
  sidecar["train_steps"] = std::move(steps_out);

  const int n = src.at("batch_n").get<int>();
  std::vector<double> xs = src.at("batch_x_rowmajor").get<std::vector<double>>();
  auto got_batch = mlf.predict_batch(xs.data(), n, p.input_dim);
  nlohmann::ordered_json eb = nlohmann::ordered_json::object();
  for (const auto& pr : got_batch) {
    eb[pr.first] = pr.second;
  }
  sidecar["batch_n"] = n;
  sidecar["batch_x_rowmajor"] = xs;
  sidecar["expected_batch"] = std::move(eb);

  std::vector<double> xq = src.at("predict_x").get<std::vector<double>>();
  auto got_pred = mlf.predict(xq.data(), p.input_dim);
  nlohmann::ordered_json ep = nlohmann::ordered_json::object();
  for (const auto& pr : got_pred) {
    ep[pr.first] = pr.second;
  }
  sidecar["predict_x"] = xq;
  sidecar["expected_predict"] = std::move(ep);
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (multilabel_dif)\n";
}

void generate_merge_from(const fs::path& out_dir, const fs::path& fixtures_root) {
  const fs::path src_dir = fixtures_root / "merge_from";
  copy_fixture_file(src_dir / "self_before.cypha", out_dir / "self_before.cypha");
  copy_fixture_file(src_dir / "other.cypha", out_dir / "other.cypha");
  const Json src = read_json(src_dir / "sidecar.json");
  const int field_dim = src.at("field_dim").get<int>();
  std::vector<double> f_self = flatten_f_field(src.at("f_field_self"));
  std::vector<double> f_other = flatten_f_field(src.at("f_field_other"));
  cypha::CNode self_before = cypha::load_cypha_file((out_dir / "self_before.cypha").string().c_str());
  cypha::CNode other = cypha::load_cypha_file((out_dir / "other.cypha").string().c_str());
  cypha::CyphaDifMemoryState self =
      cypha::CyphaDifMemoryState::from_cypha_root(self_before, f_self.data(), field_dim);
  cypha::CyphaDifMemoryState other_mem =
      cypha::CyphaDifMemoryState::from_cypha_root(other, f_other.data(), field_dim);
  const double w_self = src.value("weight_self", 0.5);
  const double w_other = src.value("weight_other", 0.5);
  auto new_labels = cypha::memory_merge_from(self, other_mem, w_self, w_other);
  cypha::CNode merged = cypha::CyphaDifMemoryState::merge_state_into_root_for_save(self_before, self);
  cypha::save_cypha_file((out_dir / "self_after.cypha").string().c_str(), merged);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  sidecar["field_dim"] = field_dim;
  sidecar["weight_self"] = w_self;
  sidecar["weight_other"] = w_other;
  sidecar["expected_new_labels"] = new_labels;
  sidecar["f_field_self"] = src.at("f_field_self");
  sidecar["f_field_other"] = src.at("f_field_other");
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (merge_from)\n";
}

std::string pick_dif_regressor_expert(int python_step, int n_existing, int k_target, cypha::CyphaInferModel& infer,
                                     const double* x, int /*d*/) {
  if (n_existing < k_target && python_step <= k_target * 20) {
    return "_e" + std::to_string(python_step % k_target);
  }
  if (n_existing == 0) {
    return "_e0";
  }
  std::vector<double> H;
  cypha::batch_encode(infer, x, 1, H);
  std::vector<double> llr;
  cypha::score_matrix_use_field(infer, H.data(), 1, llr);
  const int K = static_cast<int>(infer.labels.size());
  if (K == 0) {
    return "_e0";
  }
  int bi = 0;
  for (int k = 1; k < K; ++k) {
    if (llr[static_cast<std::size_t>(k)] > llr[static_cast<std::size_t>(bi)]) {
      bi = k;
    }
  }
  return infer.labels[static_cast<std::size_t>(bi)];
}

void generate_dif_regressor_train_step(const fs::path& out_dir, const fs::path& fixtures_root) {
  const fs::path src_dir = fixtures_root / "dif_regressor_train_step";
  copy_fixture_file(src_dir / "before.cypha", out_dir / "before.cypha");
  copy_fixture_file(src_dir / "f_field.json", out_dir / "f_field.json");
  const Json src = read_json(src_dir / "sidecar.json");
  auto [infer, mem] = load_infer_and_mem(out_dir / "before.cypha", out_dir / "f_field.json");
  const int d = infer.d_latent;

  cypha::TrainStepParams tsp;
  tsp.enc_lr = src.value("enc_lr", 0.002);
  tsp.replay_ratio = src.value("replay_ratio", 0.0);
  tsp.replay_cap = src.value("replay_cap", 10000);
  tsp.align_every = src.value("align_every", 500);
  tsp.temp_recalib_every = src.value("temp_recalib_every", 0);
  const double world_lr = src.at("world_lr").get<double>();
  const double delta_lr = src.at("delta_lr").get<double>();
  const double ood_sigma = src.at("ood_sigma").get<double>();
  const int n_experts_cap = src.at("n_experts").get<int>();
  const double target_lr = src.at("target_lr").get<double>();
  const int target_dim = src.at("target_dim").get<int>();

  cypha::ReplayBuffer replay(tsp.replay_cap);
  std::vector<double> replay_u01_storage;
  if (src.contains("replay_u01")) {
    replay_u01_storage = src.at("replay_u01").get<std::vector<double>>();
  }
  std::size_t replay_u01_pos = 0;
  std::mt19937 rng{424242};
  int enc_updates = 0;
  int total_steps = 0;
  cypha::TrainStepExtras extras{};
  extras.total_steps = &total_steps;
  if (!replay_u01_storage.empty()) {
    extras.replay_u01 = replay_u01_storage.data();
    extras.replay_u01_len = replay_u01_storage.size();
    extras.replay_u01_pos = &replay_u01_pos;
  }

  struct ExpertStat {
    std::vector<double> mu;
    double var_ema{0.0};
    int n_updates{0};
  };
  std::unordered_map<std::string, ExpertStat> experts;
  nlohmann::ordered_json steps_out = nlohmann::ordered_json::array();
  for (const auto& st : src.at("steps")) {
    std::vector<double> x = st.at("x").get<std::vector<double>>();
    const double y = st.at("y").get<double>();
    const int k_target = std::max(n_experts_cap, 4);
    const int python_step = total_steps + 1;
    const int n_existing = static_cast<int>(mem.labels.size());
    const std::string expert = pick_dif_regressor_expert(python_step, n_existing, k_target, infer, x.data(), d);
    const double loss = cypha::dif_train_step_vector(infer, mem, replay, x.data(), d, expert, world_lr, delta_lr,
                                                     world_lr, delta_lr, ood_sigma, tsp, rng, enc_updates, nullptr,
                                                     &extras);
    ExpertStat& es = experts[expert];
    cypha::regression::expert_target_ema_step(es.mu, es.var_ema, es.n_updates, &y, target_dim, target_lr);
    nlohmann::ordered_json so = nlohmann::ordered_json::object();
    so["x"] = st.at("x");
    so["y"] = y;
    so["expected_loss"] = loss;
    so["expected_expert"] = expert;
    steps_out.push_back(std::move(so));
  }

  nlohmann::ordered_json mu_out = nlohmann::ordered_json::object();
  nlohmann::ordered_json var_out = nlohmann::ordered_json::object();
  nlohmann::ordered_json n_out = nlohmann::ordered_json::object();
  for (const auto& pr : experts) {
    nlohmann::ordered_json mu_arr = nlohmann::ordered_json::array();
    for (double v : pr.second.mu) {
      mu_arr.push_back(v);
    }
    mu_out[pr.first] = std::move(mu_arr);
    var_out[pr.first] = pr.second.var_ema;
    n_out[pr.first] = pr.second.n_updates;
  }

  std::vector<double> qx = src.at("predict_x").get<std::vector<double>>();
  std::vector<double> Hq;
  cypha::batch_encode(infer, qx.data(), 1, Hq);
  std::vector<double> llr;
  cypha::score_matrix_use_field(infer, Hq.data(), 1, llr);
  const int K = static_cast<int>(infer.labels.size());
  std::vector<double> z(static_cast<std::size_t>(K));
  for (int i = 0; i < K; ++i) {
    z[static_cast<std::size_t>(i)] = llr[static_cast<std::size_t>(i)] / (infer.temperature + 1e-8);
  }
  std::vector<double> probs;
  cypha::softmax_batch_reference(z.data(), 1, K, 1e-8, probs);
  std::vector<double> mu_k(static_cast<std::size_t>(K), 0.0);
  std::vector<double> var_k(static_cast<std::size_t>(K), 0.0);
  for (int i = 0; i < K; ++i) {
    const std::string& lb = infer.labels[static_cast<std::size_t>(i)];
    auto it = experts.find(lb);
    if (it != experts.end() && !it->second.mu.empty()) {
      mu_k[static_cast<std::size_t>(i)] = it->second.mu[0];
      var_k[static_cast<std::size_t>(i)] = it->second.var_ema;
    }
  }
  double yhat = 0.0;
  double unc = 0.0;
  cypha::regression::predict_mixture_scalar(probs.data(), mu_k.data(), var_k.data(), static_cast<std::size_t>(K),
                                            yhat, unc);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  sidecar["d_latent"] = d;
  sidecar["field_dim"] = src.at("field_dim");
  sidecar["world_lr"] = world_lr;
  sidecar["delta_lr"] = delta_lr;
  sidecar["ood_sigma"] = ood_sigma;
  sidecar["enc_lr"] = tsp.enc_lr;
  sidecar["replay_ratio"] = tsp.replay_ratio;
  sidecar["replay_cap"] = tsp.replay_cap;
  sidecar["align_every"] = tsp.align_every;
  sidecar["temp_recalib_every"] = tsp.temp_recalib_every;
  sidecar["n_experts"] = n_experts_cap;
  sidecar["target_lr"] = target_lr;
  sidecar["target_dim"] = target_dim;
  if (src.contains("replay_u01")) {
    sidecar["replay_u01"] = src.at("replay_u01");
  }
  sidecar["steps"] = std::move(steps_out);
  sidecar["final_expert_mu"] = std::move(mu_out);
  sidecar["final_expert_var"] = std::move(var_out);
  sidecar["final_expert_n"] = std::move(n_out);
  sidecar["predict_x"] = qx;
  sidecar["expected_y_pred"] = yhat;
  sidecar["expected_uncertainty"] = unc;
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (dif_regressor_train_step)\n";
}

void generate_dif_train_sequence_dir(const fs::path& out_dir, const fs::path& src_dir) {
  copy_fixture_file(src_dir / "before.cypha", out_dir / "before.cypha");
  copy_fixture_file(src_dir / "f_field.json", out_dir / "f_field.json");
  const Json src = read_json(src_dir / "sidecar.json");
  auto [infer, mem] = load_infer_and_mem(out_dir / "before.cypha", out_dir / "f_field.json");
  const int d = infer.d_latent;
  const int d_in = src.at("d_in").get<int>();
  const int n = src.at("n").get<int>();
  const int K = src.at("K").get<int>();

  cypha::TrainStepParams tsp;
  tsp.enc_lr = src.value("enc_lr", 0.002);
  tsp.replay_ratio = src.value("replay_ratio", 0.30);
  tsp.replay_cap = src.value("replay_cap", 10000);
  tsp.align_every = src.value("align_every", 500);
  tsp.temp_recalib_every = src.value("temp_recalib_every", 0);
  const double world_lr = src.value("world_lr", 0.008);
  const double delta_lr = src.value("delta_lr", 0.05);
  const double ood_sigma = src.value("ood_sigma", 15.0);

  cypha::ReplayBuffer replay(tsp.replay_cap);
  std::vector<double> replay_u01_storage;
  if (src.contains("replay_u01")) {
    replay_u01_storage = src.at("replay_u01").get<std::vector<double>>();
  }
  std::size_t replay_u01_pos = 0;
  const unsigned rseed = static_cast<unsigned>(src.value("rng_seed", 7755));
  std::mt19937 rng{rseed};
  int enc_updates = 0;
  int total_steps = 0;
  cypha::TrainStepExtras extras{};
  extras.total_steps = &total_steps;
  if (!replay_u01_storage.empty()) {
    extras.replay_u01 = replay_u01_storage.data();
    extras.replay_u01_len = replay_u01_storage.size();
    extras.replay_u01_pos = &replay_u01_pos;
  }

  std::vector<std::vector<double>> xs;
  std::vector<std::string> labels;
  for (const auto& step : src.at("steps")) {
    xs.push_back(step.at("x").get<std::vector<double>>());
    labels.push_back(step.at("label").get<std::string>());
  }

  const bool use_gh = src.value("use_gh", false);
  std::vector<double> got_losses;
  double chi = src.value("chi_start", 1.0);
  double psi = src.value("psi_start", 1.0);
  if (use_gh) {
    std::vector<double> gh_inv = src.at("gh_inv_v_clean").get<std::vector<double>>();
    const double gh_r_base = src.at("gh_r_base").get<double>();
    const double nig_alpha = src.value("nig_alpha", 0.98);
    got_losses = cypha::dif_gh_train_classify_sequence(infer, mem, replay, xs, labels, gh_inv, gh_r_base, chi, psi,
                                                       nig_alpha, world_lr, delta_lr, ood_sigma, tsp, rng, enc_updates,
                                                       &extras);
  } else {
    got_losses = cypha::dif_train_classify_sequence(infer, mem, replay, xs, labels, world_lr, delta_lr, world_lr,
                                                    delta_lr, ood_sigma, tsp, rng, enc_updates, &extras);
  }

  const std::vector<double> x_all = src.at("x_rowmajor").get<std::vector<double>>();
  std::vector<double> llr;
  cypha::batch_llr_from_x(infer, x_all.data(), n, llr);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  for (const char* key :
       {"fixture_schema", "description", "n_epochs", "trainer_seed", "temperature", "replay_u01"}) {
    if (src.contains(key)) {
      sidecar[key] = src.at(key);
    }
  }
  if (use_gh) {
    sidecar["use_gh"] = true;
    sidecar["gh_inv_v_clean"] = src.at("gh_inv_v_clean");
    sidecar["gh_r_base"] = src.at("gh_r_base");
    sidecar["chi_start"] = src.value("chi_start", 1.0);
    sidecar["psi_start"] = src.value("psi_start", 1.0);
    sidecar["nig_alpha"] = src.value("nig_alpha", 0.98);
    sidecar["expected_chi_end"] = chi;
    sidecar["expected_psi_end"] = psi;
  }
  sidecar["d_in"] = d_in;
  sidecar["field_dim"] = src.at("field_dim");
  sidecar["n"] = n;
  sidecar["n_steps"] = src.at("n_steps");
  sidecar["K"] = K;
  sidecar["label_order"] = src.at("label_order");
  sidecar["world_lr"] = world_lr;
  sidecar["delta_lr"] = delta_lr;
  sidecar["enc_lr"] = tsp.enc_lr;
  sidecar["ood_sigma"] = ood_sigma;
  sidecar["replay_ratio"] = tsp.replay_ratio;
  sidecar["replay_cap"] = tsp.replay_cap;
  sidecar["align_every"] = tsp.align_every;
  sidecar["temp_recalib_every"] = tsp.temp_recalib_every;
  sidecar["rng_seed"] = rseed;
  sidecar["expected_step_losses"] = got_losses;
  sidecar["steps"] = src.at("steps");
  sidecar["x_rowmajor"] = x_all;
  sidecar["expected_llr_rowmajor"] = llr;
  write_json_pretty(out_dir / "sidecar.json", sidecar);
}

void generate_quantile_dif_train(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_dif_train_sequence_dir(out_dir, fixtures_root / "quantile_dif_train");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (quantile_dif_train)\n";
}

void generate_dif_train_replay(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_dif_train_sequence_dir(out_dir, fixtures_root / "dif_train_replay");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (dif_train_replay)\n";
}

void generate_studio_trainer_classify_hotpath(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_dif_train_sequence_dir(out_dir, fixtures_root / "studio_trainer_classify_hotpath");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (studio_trainer_classify_hotpath)\n";
}

void generate_studio_trainer_gh_classify_hotpath(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_dif_train_sequence_dir(out_dir, fixtures_root / "studio_trainer_gh_classify_hotpath");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (studio_trainer_gh_classify_hotpath)\n";
}

std::unordered_map<std::string, std::vector<double>> load_vec_map_json(const Json& j) {
  std::unordered_map<std::string, std::vector<double>> m;
  for (const auto& [k, v] : j.items()) {
    m[k] = v.get<std::vector<double>>();
  }
  return m;
}

std::unordered_map<std::string, std::vector<double>> load_p_map_json(const Json& j, int d) {
  std::unordered_map<std::string, std::vector<double>> m;
  const std::size_t expect = static_cast<std::size_t>(d * d);
  for (const auto& [k, arr] : j.items()) {
    auto row = arr.get<std::vector<double>>();
    if (row.size() != expect) {
      throw std::runtime_error("P size mismatch for " + k);
    }
    m[k] = std::move(row);
  }
  return m;
}

void generate_mke_train_dir(const fs::path& out_dir, const fs::path& src_dir) {
  copy_fixture_file(src_dir / "before.cypha", out_dir / "before.cypha");
  copy_fixture_file(src_dir / "f_field.json", out_dir / "f_field.json");
  const Json src = read_json(src_dir / "sidecar.json");
  auto [infer, mem] = load_infer_and_mem(out_dir / "before.cypha", out_dir / "f_field.json");
  const int d_latent = infer.d_latent;
  const int d_in = src.at("d_in").get<int>();
  const int d_rff = src.at("D_rff").get<int>();
  std::vector<double> W_rff = src.at("rff_W_rowmajor").get<std::vector<double>>();
  std::vector<double> b_rff = src.at("rff_b").get<std::vector<double>>();

  cypha::TrainStepParams tsp;
  tsp.enc_lr = src.value("enc_lr", 0.002);
  tsp.replay_ratio = src.value("replay_ratio", 0.30);
  tsp.replay_cap = src.value("replay_cap", 10000);
  tsp.align_every = src.value("align_every", 500);
  tsp.temp_recalib_every = src.value("temp_recalib_every", 0);
  const double world_lr = src.value("world_lr", 0.008);
  const double delta_lr = src.value("delta_lr", 0.05);
  const double ood_sigma = src.value("ood_sigma", 15.0);
  const double temperature = src.at("temperature").get<double>();
  const double ff = src.at("forgetting_factor").get<double>();
  constexpr double kPiFloor = 0.02;
  constexpr double kSoftmaxEps = 1e-8;

  cypha::ReplayBuffer replay(tsp.replay_cap);
  if (src.contains("replay_warmup")) {
    for (const auto& e : src.at("replay_warmup")) {
      std::vector<double> h = e.at("h").get<std::vector<double>>();
      std::vector<double> f = e.at("f").get<std::vector<double>>();
      std::string label = e.at("label").get<std::string>();
      double loss_v = e.at("loss_v").get<double>();
      double loss_arg = std::max(0.0, loss_v - 1e-3);
      replay.push(h.data(), f.data(), d_latent, label, loss_arg);
    }
  }
  std::vector<double> replay_u01_storage;
  if (src.contains("replay_u01") && src.at("replay_u01").is_array() && !src.at("replay_u01").empty()) {
    replay_u01_storage = src.at("replay_u01").get<std::vector<double>>();
  }
  std::size_t replay_u01_pos = 0;
  unsigned rseed = static_cast<unsigned>(src.value("rng_seed", 42));
  std::mt19937 rng{rseed};
  int enc_updates = src.value("enc_update_count_start", 0);
  int total_steps = src.value("total_steps_start", 0);
  cypha::TrainStepExtras extras{};
  extras.total_steps = &total_steps;
  if (!replay_u01_storage.empty()) {
    extras.replay_u01 = replay_u01_storage.data();
    extras.replay_u01_len = replay_u01_storage.size();
    extras.replay_u01_pos = &replay_u01_pos;
  }

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  for (const char* key :
       {"fixture_schema", "description", "replay_warmup", "replay_u01", "enc_update_count_start", "total_steps_start"}) {
    if (src.contains(key)) {
      sidecar[key] = src.at(key);
    }
  }
  sidecar["d_in"] = d_in;
  sidecar["D_rff"] = d_rff;
  sidecar["rff_W_rowmajor"] = W_rff;
  sidecar["rff_b"] = b_rff;
  sidecar["temperature"] = temperature;
  sidecar["forgetting_factor"] = ff;
  sidecar["world_lr"] = world_lr;
  sidecar["delta_lr"] = delta_lr;
  sidecar["ood_sigma"] = ood_sigma;
  sidecar["enc_lr"] = tsp.enc_lr;
  sidecar["replay_ratio"] = tsp.replay_ratio;
  sidecar["replay_cap"] = tsp.replay_cap;
  sidecar["align_every"] = tsp.align_every;
  sidecar["temp_recalib_every"] = tsp.temp_recalib_every;
  sidecar["rng_seed"] = rseed;

  const bool extended = src.contains("steps") && src.at("steps").is_array() && !src.at("steps").empty();
  if (!extended) {
    std::vector<double> x = src.at("x").get<std::vector<double>>();
    std::vector<double> phi(static_cast<std::size_t>(d_rff));
    cypha::regression::rff_encode_batch_rowmajor(x.data(), 1, d_in, W_rff.data(), b_rff.data(), d_rff, phi.data());
    auto w_work = load_vec_map_json(src.at("w_before"));
    auto p_work = load_p_map_json(src.at("P_before"), d_rff);
    const double y = src.at("y").get<double>();
    const int K = static_cast<int>(infer.labels.size());
    std::vector<double> gh_k(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) {
      gh_k[static_cast<std::size_t>(i)] = src.at("gh_scales")[i].get<double>();
    }
    std::string router_label = src.at("router_train_label").get<std::string>();
    cypha::regression::MkeScalarTrainStepOutputs step_out{};
    cypha::regression::mke_scalar_train_step_from_phi(infer, mem, replay, phi.data(), d_rff, y, w_work, p_work,
                                                      gh_k.data(), temperature, ff, kPiFloor, tsp, world_lr,
                                                      delta_lr, ood_sigma, rng, enc_updates, &extras, &router_label,
                                                      kSoftmaxEps, &step_out);
    sidecar["x"] = x;
    sidecar["y"] = y;
    sidecar["routing_labs"] = src.at("routing_labs");
    sidecar["gh_scales"] = src.at("gh_scales");
    sidecar["w_before"] = src.at("w_before");
    sidecar["P_before"] = src.at("P_before");
    sidecar["router_train_label"] = router_label;
    nlohmann::ordered_json phi_j = nlohmann::ordered_json::array();
    for (double v : phi) {
      phi_j.push_back(v);
    }
    sidecar["expected_phi"] = std::move(phi_j);
    sidecar["expected_err_sq"] = step_out.err_sq;
    sidecar["expected_router_loss"] = step_out.router_loss;
    nlohmann::ordered_json w_after = nlohmann::ordered_json::object();
    nlohmann::ordered_json p_after = nlohmann::ordered_json::object();
    for (const auto& pr : w_work) {
      w_after[pr.first] = pr.second;
    }
    for (const auto& pr : p_work) {
      p_after[pr.first] = pr.second;
    }
    sidecar["w_after"] = std::move(w_after);
    sidecar["P_after"] = std::move(p_after);
  } else {
    nlohmann::ordered_json steps_out = nlohmann::ordered_json::array();
    for (const auto& st : src.at("steps")) {
      std::vector<double> x = st.at("x").get<std::vector<double>>();
      std::vector<double> phi(static_cast<std::size_t>(d_rff));
      cypha::regression::rff_encode_batch_rowmajor(x.data(), 1, d_in, W_rff.data(), b_rff.data(), d_rff, phi.data());
      auto w_work = load_vec_map_json(st.at("w_before"));
      auto p_work = load_p_map_json(st.at("P_before"), d_rff);
      const double y = st.at("y").get<double>();
      const int K = static_cast<int>(infer.labels.size());
      std::vector<double> gh_k(static_cast<std::size_t>(K));
      for (int i = 0; i < K; ++i) {
        gh_k[static_cast<std::size_t>(i)] = st.at("gh_scales")[i].get<double>();
      }
      std::string router_label = st.at("router_train_label").get<std::string>();
      cypha::regression::MkeScalarTrainStepOutputs step_out{};
      cypha::regression::mke_scalar_train_step_from_phi(infer, mem, replay, phi.data(), d_rff, y, w_work, p_work,
                                                        gh_k.data(), temperature, ff, kPiFloor, tsp, world_lr,
                                                        delta_lr, ood_sigma, rng, enc_updates, &extras, &router_label,
                                                        kSoftmaxEps, &step_out);
      nlohmann::ordered_json so = nlohmann::ordered_json::object();
      so["x"] = st.at("x");
      so["y"] = y;
      so["routing_labs"] = st.at("routing_labs");
      so["gh_scales"] = st.at("gh_scales");
      so["w_before"] = st.at("w_before");
      so["P_before"] = st.at("P_before");
      so["router_train_label"] = router_label;
      nlohmann::ordered_json phi_j = nlohmann::ordered_json::array();
      for (double v : phi) {
        phi_j.push_back(v);
      }
      so["expected_phi"] = std::move(phi_j);
      so["expected_err_sq"] = step_out.err_sq;
      so["expected_router_loss"] = step_out.router_loss;
      nlohmann::ordered_json w_after = nlohmann::ordered_json::object();
      nlohmann::ordered_json p_after = nlohmann::ordered_json::object();
      for (const auto& pr : w_work) {
        w_after[pr.first] = pr.second;
      }
      for (const auto& pr : p_work) {
        p_after[pr.first] = pr.second;
      }
      so["w_after"] = std::move(w_after);
      so["P_after"] = std::move(p_after);
      so["enc_w_rowmajor"] = infer.enc_w;
      steps_out.push_back(std::move(so));
    }
    sidecar["steps"] = std::move(steps_out);
  }
  write_json_pretty(out_dir / "sidecar.json", sidecar);
}

void generate_mke_train_step(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_mke_train_dir(out_dir, fixtures_root / "mke_train_step");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (mke_train_step)\n";
}

void generate_mke_train_extended(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_mke_train_dir(out_dir, fixtures_root / "mke_train_extended");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (mke_train_extended)\n";
}

void generate_rff_regression(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "rff_regression" / "sidecar.json");
  const auto& rr = src.at("rff_ridge");
  const int n = rr.at("n").get<int>();
  const int d_in = rr.at("d_in").get<int>();
  const int D = rr.at("D").get<int>();
  const double lam = rr.at("lam").get<double>();
  const double y_mean = rr.at("y_mean").get<double>();
  const double y_std = rr.at("y_std").get<double>();
  std::vector<double> X = rr.at("X").get<std::vector<double>>();
  std::vector<double> W = rr.at("W").get<std::vector<double>>();
  std::vector<double> b = rr.at("b").get<std::vector<double>>();
  std::vector<double> y_raw = rr.at("y_raw").get<std::vector<double>>();
  std::vector<double> phi(static_cast<std::size_t>(n * D));
  cypha::regression::rff_encode_batch_rowmajor(X.data(), n, d_in, W.data(), b.data(), D, phi.data());
  std::vector<double> yn(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    yn[static_cast<std::size_t>(i)] = (y_raw[static_cast<std::size_t>(i)] - y_mean) / y_std;
  }
  std::vector<double> coef(static_cast<std::size_t>(D + 1));
  cypha::regression::ridge_fit_bias(phi.data(), n, D, lam, yn.data(), coef.data());
  std::vector<double> pred_norm(static_cast<std::size_t>(n));
  cypha::regression::linear_predict_with_bias(phi.data(), n, D, coef.data(), pred_norm.data());
  std::vector<double> y_pred(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    y_pred[static_cast<std::size_t>(i)] = pred_norm[static_cast<std::size_t>(i)] * y_std + y_mean;
  }

  const auto& md = src.at("mke_dots");
  const int d_feat = md.at("d_feat").get<int>();
  const int K = md.at("K").get<int>();
  std::vector<double> ph1 = md.at("phi").get<std::vector<double>>();
  std::vector<double> wexp = md.at("W_experts_rowmajor").get<std::vector<double>>();
  std::vector<double> dots(static_cast<std::size_t>(K));
  cypha::regression::mke_expert_linear_dots(ph1.data(), d_feat, K, wexp.data(), dots.data());

  nlohmann::ordered_json out = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    out["fixture_schema"] = src.at("fixture_schema");
  }
  nlohmann::ordered_json rr_out = nlohmann::ordered_json::object();
  rr_out["n"] = n;
  rr_out["d_in"] = d_in;
  rr_out["D"] = D;
  rr_out["lam"] = lam;
  rr_out["y_mean"] = y_mean;
  rr_out["y_std"] = y_std;
  rr_out["X"] = X;
  rr_out["W"] = W;
  rr_out["b"] = b;
  rr_out["y_raw"] = y_raw;
  rr_out["expected_phi_rowmajor"] = phi;
  rr_out["expected_coef"] = coef;
  rr_out["expected_y_pred"] = y_pred;
  out["rff_ridge"] = std::move(rr_out);
  nlohmann::ordered_json md_out = nlohmann::ordered_json::object();
  md_out["d_feat"] = d_feat;
  md_out["K"] = K;
  md_out["phi"] = ph1;
  md_out["W_experts_rowmajor"] = wexp;
  md_out["expected_dots"] = dots;
  out["mke_dots"] = std::move(md_out);
  write_json_pretty(out_dir / "sidecar.json", out);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (rff_regression)\n";
}

void generate_two_stage_ridge_dir(const fs::path& out_dir, const fs::path& src_dir) {
  const Json src = read_json(src_dir / "sidecar.json");
  const int n = src.at("n").get<int>();
  const int K = src.at("K").get<int>();
  const int d_in = src.at("d_in").get<int>();
  const int D2 = src.at("D2").get<int>();
  std::vector<double> llr = src.at("llr_rowmajor").get<std::vector<double>>();
  std::vector<double> X = src.at("X_rowmajor").get<std::vector<double>>();
  std::vector<double> y_raw = src.at("y_raw").get<std::vector<double>>();
  std::vector<double> encW = src.at("enc2_W").get<std::vector<double>>();
  std::vector<double> encb = src.at("enc2_b").get<std::vector<double>>();
  std::vector<double> w1(static_cast<std::size_t>(K + d_in));
  double b1 = 0.0;
  std::vector<double> w2(static_cast<std::size_t>(D2));
  double b2 = 0.0;
  cypha::regression::two_stage_dif_ridge_fit_from_llr(
      llr.data(), n, K, X.data(), d_in, y_raw.data(), src.at("y_mean").get<double>(), src.at("y_std").get<double>(),
      src.at("lam1").get<double>(), src.at("lam2").get<double>(), encW.data(), encb.data(), D2, w1.data(), &b1,
      w2.data(), &b2);
  std::vector<double> phi(static_cast<std::size_t>(n * D2));
  cypha::regression::rff_encode_batch_rowmajor(X.data(), n, d_in, encW.data(), encb.data(), D2, phi.data());
  std::vector<double> yhat_batch(static_cast<std::size_t>(n));
  cypha::regression::two_stage_dif_predict_batch(llr.data(), n, K, X.data(), d_in, w1.data(), b1, phi.data(), D2,
                                                 w2.data(), b2, 0.0, 1.0, yhat_batch.data());

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  for (const char* key :
       {"fixture_schema", "description", "n", "K", "d_in", "D2", "lam1", "lam2", "llr_rowmajor", "X_rowmajor",
        "y_raw", "y_mean", "y_std", "enc2_W", "enc2_b"}) {
    sidecar[key] = src.at(key);
  }
  sidecar["expected_w1"] = w1;
  sidecar["expected_b1"] = b1;
  sidecar["expected_w2"] = w2;
  sidecar["expected_b2"] = b2;
  sidecar["expected_yn_hat"] = yhat_batch;
  write_json_pretty(out_dir / "sidecar.json", sidecar);
}

void generate_two_stage_ridge_fit(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_two_stage_ridge_dir(out_dir, fixtures_root / "two_stage_ridge_fit");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (two_stage_ridge_fit)\n";
}

void generate_two_stage_e2e_ridge(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_two_stage_ridge_dir(out_dir, fixtures_root / "two_stage_e2e_ridge");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (two_stage_e2e_ridge)\n";
}

void generate_two_stage_pipeline(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "two_stage_pipeline" / "sidecar.json");
  cypha::CNode root_node = cypha::load_cypha_file((fixtures_root / "reference.cypha").string().c_str());
  const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
  const int fd = static_cast<int>(fh.shape[0]);
  const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
  const int d = static_cast<int>(enc.shape[0]);
  std::vector<double> fflat = flatten_f_field(read_json(fixtures_root / "f_field.json"));
  cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);
  const int d_in = src.at("d_in").get<int>();
  const int D2 = src.at("D2").get<int>();
  std::vector<double> x = src.at("x").get<std::vector<double>>();
  std::vector<double> encW = src.at("enc2_W").get<std::vector<double>>();
  std::vector<double> encb = src.at("enc2_b").get<std::vector<double>>();
  std::vector<double> w1 = src.at("w1").get<std::vector<double>>();
  std::vector<double> w2 = src.at("w2").get<std::vector<double>>();
  std::vector<double> h;
  cypha::batch_encode(infer, x.data(), 1, h);
  std::vector<double> llr;
  cypha::score_matrix_use_field(infer, h.data(), 1, llr);
  const double yp = cypha::regression::two_stage_dif_predict_with_clf(
      infer, x.data(), d_in, encW.data(), encb.data(), D2, w1.data(), src.at("b1").get<double>(), w2.data(),
      src.at("b2").get<double>(), src.at("y_mean").get<double>(), src.at("y_std").get<double>());

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  sidecar["d_in"] = d_in;
  sidecar["D2"] = D2;
  sidecar["K"] = src.at("K");
  sidecar["x"] = x;
  sidecar["enc2_W"] = encW;
  sidecar["enc2_b"] = encb;
  sidecar["w1"] = w1;
  sidecar["b1"] = src.at("b1");
  sidecar["w2"] = w2;
  sidecar["b2"] = src.at("b2");
  sidecar["y_mean"] = src.at("y_mean");
  sidecar["y_std"] = src.at("y_std");
  sidecar["expected_llr"] = llr;
  sidecar["expected_y"] = yp;
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (two_stage_pipeline)\n";
}

void generate_preprocess_train_classify_dir(const fs::path& out_dir, const fs::path& src_dir) {
  copy_fixture_file(src_dir / "before.cypha", out_dir / "before.cypha");
  copy_fixture_file(src_dir / "f_field.json", out_dir / "f_field.json");
  copy_fixture_file(src_dir / "preprocessor.json", out_dir / "preprocessor.json");
  if (fs::exists(src_dir / "train.csv")) {
    copy_fixture_file(src_dir / "train.csv", out_dir / "train.csv");
  }
  const Json src = read_json(src_dir / "sidecar.json");
  cypha::PreprocessorState pre =
      cypha::PreprocessorState::from_json_file((out_dir / "preprocessor.json").string().c_str());
  auto [infer, mem] = load_infer_and_mem(out_dir / "before.cypha", out_dir / "f_field.json");
  const int d = infer.d_latent;
  const int d_raw = src.at("d_raw").get<int>();
  const int n = src.at("n").get<int>();
  const int K = src.at("K").get<int>();

  cypha::TrainStepParams tsp;
  tsp.enc_lr = src.value("enc_lr", 0.002);
  tsp.replay_ratio = src.value("replay_ratio", 0.30);
  tsp.replay_cap = src.value("replay_cap", 10000);
  tsp.align_every = src.value("align_every", 500);
  tsp.temp_recalib_every = src.value("temp_recalib_every", 0);
  const double world_lr = src.value("world_lr", 0.008);
  const double delta_lr = src.value("delta_lr", 0.05);
  const double ood_sigma = src.value("ood_sigma", 15.0);

  cypha::ReplayBuffer replay(tsp.replay_cap);
  std::vector<double> replay_u01_storage;
  if (src.contains("replay_u01")) {
    replay_u01_storage = src.at("replay_u01").get<std::vector<double>>();
  }
  std::size_t replay_u01_pos = 0;
  unsigned rseed = static_cast<unsigned>(src.value("rng_seed", 7755));
  std::mt19937 rng{rseed};
  int enc_updates = 0;
  int total_steps = 0;
  cypha::TrainStepExtras extras{};
  extras.total_steps = &total_steps;
  if (!replay_u01_storage.empty()) {
    extras.replay_u01 = replay_u01_storage.data();
    extras.replay_u01_len = replay_u01_storage.size();
    extras.replay_u01_pos = &replay_u01_pos;
  }

  std::vector<std::vector<double>> xs;
  std::vector<std::string> labels;
  if (src.contains("csv") && !src["csv"].is_null()) {
    const std::string csv_name = src.at("csv").get<std::string>();
    const auto& cs = src.at("csv_spec");
    cypha::CsvDenseSpec spec;
    spec.has_header = cs.value("has_header", true);
    std::string delim = cs.value("delimiter", ",");
    spec.delimiter = delim[0];
    spec.target_col_name = cs.value("target_col_name", std::string{});
    if (spec.target_col_name.empty()) {
      spec.target_col_index = cs.at("target_col_index").get<int>();
    }
    if (cs.contains("feature_col_names") && cs["feature_col_names"].is_array()) {
      for (const auto& v : cs.at("feature_col_names")) {
        spec.feature_col_names.push_back(v.get<std::string>());
      }
    }
    cypha::CsvDenseResult csv = cypha::load_csv_dense(out_dir / csv_name, spec);
    for (int r = 0; r < csv.n_rows; ++r) {
      std::vector<double> xraw(static_cast<std::size_t>(d_raw));
      const std::size_t base = static_cast<std::size_t>(r * d_raw);
      for (int c = 0; c < d_raw; ++c) {
        xraw[static_cast<std::size_t>(c)] = csv.x_rowmajor[base + static_cast<std::size_t>(c)];
      }
      xs.push_back(pre.transform_one(xraw));
      labels.push_back(csv.y_class[static_cast<std::size_t>(r)]);
    }
  } else {
    for (const auto& step : src.at("steps")) {
      std::vector<double> xraw = step.at("x_raw").get<std::vector<double>>();
      xs.push_back(pre.transform_one(xraw));
      labels.push_back(step.at("label").get<std::string>());
    }
  }

  const bool use_gh = src.value("use_gh", false);
  double chi = src.value("chi_start", 1.0);
  double psi = src.value("psi_start", 1.0);
  std::vector<double> got_losses;
  if (use_gh) {
    std::vector<double> gh_inv = src.at("gh_inv_v_clean").get<std::vector<double>>();
    got_losses = cypha::dif_gh_train_classify_sequence(
        infer, mem, replay, xs, labels, gh_inv, src.at("gh_r_base").get<double>(), chi, psi,
        src.value("nig_alpha", 0.98), world_lr, delta_lr, ood_sigma, tsp, rng, enc_updates, &extras);
  } else {
    got_losses = cypha::dif_train_classify_sequence(infer, mem, replay, xs, labels, world_lr, delta_lr, world_lr,
                                                    delta_lr, ood_sigma, tsp, rng, enc_updates, &extras);
  }
  std::vector<double> x_all = src.at("x_rowmajor").get<std::vector<double>>();
  std::vector<double> llr;
  cypha::batch_llr_from_x(infer, x_all.data(), n, llr);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  for (const char* key :
       {"fixture_schema", "description", "csv", "csv_spec", "use_gh", "gh_inv_v_clean", "gh_r_base", "chi_start",
        "psi_start", "nig_alpha", "n_epochs", "trainer_seed", "temperature", "replay_u01", "steps"}) {
    if (src.contains(key)) {
      sidecar[key] = src.at(key);
    }
  }
  if (use_gh) {
    sidecar["expected_chi_end"] = chi;
    sidecar["expected_psi_end"] = psi;
  }
  sidecar["d_in"] = src.at("d_in");
  sidecar["d_raw"] = d_raw;
  sidecar["field_dim"] = src.at("field_dim");
  sidecar["n"] = n;
  sidecar["n_steps"] = src.at("n_steps");
  sidecar["K"] = K;
  sidecar["label_order"] = src.at("label_order");
  sidecar["world_lr"] = world_lr;
  sidecar["delta_lr"] = delta_lr;
  sidecar["enc_lr"] = tsp.enc_lr;
  sidecar["ood_sigma"] = ood_sigma;
  sidecar["replay_ratio"] = tsp.replay_ratio;
  sidecar["replay_cap"] = tsp.replay_cap;
  sidecar["align_every"] = tsp.align_every;
  sidecar["temp_recalib_every"] = tsp.temp_recalib_every;
  sidecar["rng_seed"] = rseed;
  sidecar["expected_step_losses"] = got_losses;
  sidecar["x_rowmajor"] = x_all;
  sidecar["expected_llr_rowmajor"] = llr;
  write_json_pretty(out_dir / "sidecar.json", sidecar);
}

void generate_csv_preprocess_classify_hotpath(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_preprocess_train_classify_dir(out_dir, fixtures_root / "csv_preprocess_classify_hotpath");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (csv_preprocess_classify_hotpath)\n";
}

void generate_studio_trainer_preprocess_classify_hotpath(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_preprocess_train_classify_dir(out_dir, fixtures_root / "studio_trainer_preprocess_classify_hotpath");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (studio_trainer_preprocess_classify_hotpath)\n";
}

void generate_studio_trainer_preprocess_gh_classify_hotpath(const fs::path& out_dir,
                                                            const fs::path& fixtures_root) {
  generate_preprocess_train_classify_dir(out_dir, fixtures_root / "studio_trainer_preprocess_gh_classify_hotpath");
  std::cout << "wrote " << (out_dir / "sidecar.json").string()
            << " (studio_trainer_preprocess_gh_classify_hotpath)\n";
}

cypha::CyphaInferModel load_generation_model(const fs::path& fixtures_root) {
  cypha::CNode root_node = cypha::load_cypha_file((fixtures_root / "reference.cypha").string().c_str());
  const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
  const int fd = static_cast<int>(fh.shape[0]);
  std::vector<double> fflat = flatten_f_field(read_json(fixtures_root / "f_field.json"));
  return cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);
}

void generate_generation(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "generation" / "sidecar.json");
  cypha::CyphaInferModel m = load_generation_model(fixtures_root);
  const int d = src.at("d_latent").get<int>();
  const double atol = src.at("atol").get<double>();
  nlohmann::ordered_json cases = nlohmann::ordered_json::object();

  const auto& sc = src.at("cases");
  {
    const auto& c = sc.at("generate_gaussian_no_rejection");
    auto got = cypha::generate_class_gaussian(m, c.at("label").get<std::string>(), c.at("n").get<int>(),
                                              c.at("temperature").get<double>(), nullptr, false, 16,
                                              flatten_json_mat(c.at("z")).data());
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["label"] = c.at("label");
    oc["n"] = c.at("n");
    oc["temperature"] = c.at("temperature");
    oc["z"] = c.at("z");
    oc["expected_h"] = got;
    cases["generate_gaussian_no_rejection"] = std::move(oc);
  }
  {
    const auto& c = sc.at("generate_gaussian_rejection");
    auto got = cypha::generate_class_gaussian(m, c.at("label").get<std::string>(), c.at("n").get<int>(),
                                              c.at("temperature").get<double>(), nullptr, true,
                                              c.at("max_candidates").get<int>(),
                                              flatten_json_mat(c.at("z_candidates")).data());
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["label"] = c.at("label");
    oc["n"] = c.at("n");
    oc["temperature"] = c.at("temperature");
    oc["max_candidates"] = c.at("max_candidates");
    oc["z_candidates"] = c.at("z_candidates");
    oc["expected_h"] = got;
    cases["generate_gaussian_rejection"] = std::move(oc);
  }
  {
    const auto& c = sc.at("generate_conditioned");
    auto got = cypha::generate_conditioned(m, c.at("label").get<std::string>(), c.at("n").get<int>(),
                                         c.at("temperature").get<double>(), nullptr,
                                         flatten_json_mat(c.at("z")).data());
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["label"] = c.at("label");
    oc["n"] = c.at("n");
    oc["temperature"] = c.at("temperature");
    oc["z"] = c.at("z");
    oc["expected_h"] = got;
    cases["generate_conditioned"] = std::move(oc);
  }
  {
    const auto& c = sc.at("generate_langevin");
    auto got = cypha::generate_langevin(m, c.at("label").get<std::string>(), c.at("n").get<int>(),
                                        c.at("n_steps").get<int>(), c.at("step_size").get<double>(),
                                        c.at("temperature").get<double>(), nullptr,
                                        flatten_json_mat(c.at("z_init")).data(),
                                        flatten_json_3d(c.at("z_noise")).data());
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["label"] = c.at("label");
    oc["n"] = c.at("n");
    oc["n_steps"] = c.at("n_steps");
    oc["step_size"] = c.at("step_size");
    oc["temperature"] = c.at("temperature");
    oc["z_init"] = c.at("z_init");
    oc["z_noise"] = c.at("z_noise");
    oc["expected_h"] = got;
    cases["generate_langevin"] = std::move(oc);
  }
  {
    const auto& c = sc.at("predict_next");
    cypha::CyphaInferModel m2 = m;
    m2.ctx_last_label = c.at("last_label").get<std::string>();
    auto got = cypha::predict_next_probs(m2, m2.ctx_last_label);
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["last_label"] = c.at("last_label");
    oc["expected_probs"] = got;
    cases["predict_next"] = std::move(oc);
  }
  if (sc.contains("generate_boundary")) {
    const auto& c = sc.at("generate_boundary");
    auto got = cypha::generate_boundary(m, c.at("label_a").get<std::string>(), c.at("label_b").get<std::string>(),
                                        c.at("n").get<int>(), c.at("alpha").get<double>(),
                                        c.at("temperature").get<double>(), nullptr,
                                        flatten_json_mat(c.at("z")).data());
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["label_a"] = c.at("label_a");
    oc["label_b"] = c.at("label_b");
    oc["n"] = c.at("n");
    oc["alpha"] = c.at("alpha");
    oc["temperature"] = c.at("temperature");
    oc["z"] = c.at("z");
    oc["expected_h"] = got;
    cases["generate_boundary"] = std::move(oc);
  }
  if (sc.contains("generate_ood")) {
    const auto& c = sc.at("generate_ood");
    auto got = cypha::generate_ood(m, c.at("n").get<int>(), c.at("n_candidates").get<int>(), nullptr,
                                   flatten_json_mat(c.at("z_candidates")).data());
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["n"] = c.at("n");
    oc["n_candidates"] = c.at("n_candidates");
    oc["z_candidates"] = c.at("z_candidates");
    oc["expected_h"] = got;
    cases["generate_ood"] = std::move(oc);
  }
  if (sc.contains("generate_mdl_ball")) {
    const auto& c = sc.at("generate_mdl_ball");
    auto got = cypha::generate_mdl_ball(m, c.at("label").get<std::string>(), c.at("n").get<int>(),
                                        c.at("radius").get<double>(), nullptr, flatten_json_mat(c.at("z_dir")).data(),
                                        c.at("u_mag").get<std::vector<double>>().data());
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["label"] = c.at("label");
    oc["n"] = c.at("n");
    oc["radius"] = c.at("radius");
    oc["z_dir"] = c.at("z_dir");
    oc["u_mag"] = c.at("u_mag");
    oc["expected_h"] = got;
    cases["generate_mdl_ball"] = std::move(oc);
  }
  if (sc.contains("generate_ancestral")) {
    const auto& c = sc.at("generate_ancestral");
    auto got = cypha::generate_ancestral(m, c.at("n").get<int>(), c.at("temperature").get<double>(), nullptr,
                                         c.at("u_class").get<std::vector<double>>().data(),
                                         flatten_json_mat(c.at("z")).data());
    nlohmann::ordered_json labels = nlohmann::ordered_json::array();
    nlohmann::ordered_json eh = nlohmann::ordered_json::array();
    for (const auto& s : got) {
      labels.push_back(s.label);
      eh.push_back(s.h);
    }
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["n"] = c.at("n");
    oc["temperature"] = c.at("temperature");
    oc["u_class"] = c.at("u_class");
    oc["z"] = c.at("z");
    oc["expected_labels"] = std::move(labels);
    oc["expected_h"] = std::move(eh);
    cases["generate_ancestral"] = std::move(oc);
  }
  if (sc.contains("rollout")) {
    const auto& c = sc.at("rollout");
    cypha::CyphaInferModel m2 = m;
    auto got = cypha::rollout(m2, c.at("seed_label").get<std::string>(), c.at("n_steps").get<int>(),
                              c.at("temperature").get<double>(), c.at("exploration").get<double>(), nullptr,
                              flatten_json_mat(c.at("z_generate")).data(),
                              c.at("u_transition").get<std::vector<double>>().data());
    nlohmann::ordered_json labels = nlohmann::ordered_json::array();
    nlohmann::ordered_json eh = nlohmann::ordered_json::array();
    for (const auto& s : got) {
      labels.push_back(s.label);
      eh.push_back(s.h);
    }
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["seed_label"] = c.at("seed_label");
    oc["n_steps"] = c.at("n_steps");
    oc["temperature"] = c.at("temperature");
    oc["exploration"] = c.at("exploration");
    oc["z_generate"] = c.at("z_generate");
    oc["u_transition"] = c.at("u_transition");
    oc["expected_labels"] = std::move(labels);
    oc["expected_h"] = std::move(eh);
    cases["rollout"] = std::move(oc);
  }
  if (sc.contains("generate_from_observation")) {
    const auto& c = sc.at("generate_from_observation");
    auto got = cypha::generate_from_observation(m, c.at("h_obs").get<std::vector<double>>().data(),
                                                c.at("label").get<std::string>(), c.at("n").get<int>(),
                                                c.at("temperature").get<double>(), c.at("n_steps").get<int>(), nullptr,
                                                flatten_json_3d(c.at("z_noise")).data());
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["label"] = c.at("label");
    oc["n"] = c.at("n");
    oc["temperature"] = c.at("temperature");
    oc["n_steps"] = c.at("n_steps");
    oc["h_obs"] = c.at("h_obs");
    oc["z_noise"] = c.at("z_noise");
    oc["expected_h"] = got;
    cases["generate_from_observation"] = std::move(oc);
  }
  if (sc.contains("generate_retrieval_augmented")) {
    const auto& c = sc.at("generate_retrieval_augmented");
    std::vector<double> db_flat;
    for (const auto& row : c.at("database_x")) {
      auto rv = row.get<std::vector<double>>();
      db_flat.insert(db_flat.end(), rv.begin(), rv.end());
    }
    const int n_db = static_cast<int>(c.at("database_x").size());
    const int input_dim = src.value("input_dim", d);
    cypha::CyphaInferOptions opt;
    opt.use_field = true;
    auto got = cypha::generate_retrieval_augmented(
        m, c.at("query_x").get<std::vector<double>>().data(), db_flat.data(), n_db, input_dim,
        c.at("k_neighbors").get<int>(), c.at("n").get<int>(), c.at("temperature").get<double>(),
        c.at("n_steps").get<int>(), opt, nullptr, flatten_json_3d(c.at("z_noise")).data());
    nlohmann::ordered_json oc = nlohmann::ordered_json::object();
    oc["n"] = c.at("n");
    oc["k_neighbors"] = c.at("k_neighbors");
    oc["temperature"] = c.at("temperature");
    oc["n_steps"] = c.at("n_steps");
    oc["query_x"] = c.at("query_x");
    oc["database_x"] = c.at("database_x");
    oc["z_noise"] = c.at("z_noise");
    oc["expected_h"] = got;
    if (c.contains("expected_label")) {
      oc["expected_label"] = c.at("expected_label");
    }
    cases["generate_retrieval_augmented"] = std::move(oc);
  }

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  sidecar["fixture_schema"] = src.value("fixture_schema", 1);
  if (src.contains("generator")) {
    sidecar["generator"] = src.at("generator");
  }
  if (src.contains("seed")) {
    sidecar["seed"] = src.at("seed");
  }
  sidecar["d_latent"] = d;
  sidecar["field_dim"] = src.at("field_dim");
  sidecar["input_dim"] = src.value("input_dim", d);
  if (src.contains("labels")) {
    sidecar["labels"] = src.at("labels");
  }
  sidecar["atol"] = atol;
  sidecar["cases"] = std::move(cases);
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (generation)\n";
}

void generate_cyphalm_char_lstm(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "cyphalm_char_lstm" / "sidecar.json");
  cypha::cyphalm::CharLSTMHead lstm;
  lstm.vocab_size = src.at("vocab_size").get<int>();
  lstm.hidden = src.at("hidden").get<int>();
  const auto& w = src.at("char_lstm");
  lstm.E = w.at("E").get<std::vector<double>>();
  lstm.Wx = w.at("Wx").get<std::vector<double>>();
  lstm.Wh = w.at("Wh").get<std::vector<double>>();
  lstm.b = w.at("b").get<std::vector<double>>();
  lstm.Wy = w.at("Wy").get<std::vector<double>>();
  lstm.by = w.at("by").get<std::vector<double>>();
  cypha::cyphalm::GRIALowRank gria;
  gria.field_dim = src.at("field_dim").get<int>();
  gria.vocab_size = src.at("vocab_size").get<int>();
  gria.rank = src.at("rank").get<int>();
  const auto& gw = src.at("gria");
  gria.U = gw.at("U").get<std::vector<double>>();
  gria.V = gw.at("V").get<std::vector<double>>();
  gria.alpha = gw.at("alpha").get<std::vector<double>>();
  gria.bias = gw.at("bias").get<std::vector<double>>();

  const int token_id = src.at("token_id").get<int>();
  const int target_id = src.at("target_id").get<int>();
  const int gria_target = src.at("gria_target_id").get<int>();
  const double blend_logit = src.at("blend_logit").get<double>();
  const int vocab = src.at("vocab_size").get<int>();
  std::vector<double> h = src.at("h_init").get<std::vector<double>>();
  std::vector<double> c = src.at("c_init").get<std::vector<double>>();
  std::vector<double> v = src.at("v_field").get<std::vector<double>>();

  std::vector<double> log_probs(static_cast<std::size_t>(vocab));
  std::vector<double> h_new;
  std::vector<double> c_new;
  cypha::cyphalm::CharLSTMCache cache;
  lstm.forward_step(token_id, h.data(), c.data(), log_probs.data(), h_new, c_new, &cache);
  const double loss = -log_probs[static_cast<std::size_t>(target_id)];
  cypha::cyphalm::CharLSTMGrad grads = lstm.backward_step(cache, target_id);
  std::vector<double> log_g(static_cast<std::size_t>(vocab));
  gria.forward(v.data(), log_g.data());
  cypha::cyphalm::GRIALowRankGrad gg = gria.cross_entropy_gradients(v.data(), gria_target);
  std::vector<double> log_blend(static_cast<std::size_t>(vocab));
  cypha::cyphalm::blend_log_probs(log_g.data(), log_probs.data(), vocab, blend_logit, log_blend.data());
  const double bg = cypha::cyphalm::blend_logit_grad(log_g.data(), log_probs.data(), vocab, blend_logit, gria_target);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  for (const char* key :
       {"fixture_schema", "vocab_size", "hidden", "field_dim", "rank", "char_lstm", "gria", "token_id", "target_id",
        "gria_target_id", "blend_logit", "h_init", "c_init", "v_field"}) {
    sidecar[key] = src.at(key);
  }
  nlohmann::ordered_json exp = nlohmann::ordered_json::object();
  exp["log_probs"] = log_probs;
  exp["loss"] = loss;
  exp["h_new"] = h_new;
  exp["c_new"] = c_new;
  exp["dWy_norm"] = std::sqrt(std::inner_product(grads.dWy.begin(), grads.dWy.end(), grads.dWy.begin(), 0.0));
  exp["dWx_sum"] = std::accumulate(grads.dWx.begin(), grads.dWx.end(), 0.0);
  exp["dh_prev"] = grads.dh_prev;
  exp["log_probs_gria"] = log_g;
  exp["dU_sum"] = std::accumulate(gg.dU.begin(), gg.dU.end(), 0.0);
  exp["dV_sum"] = std::accumulate(gg.dV.begin(), gg.dV.end(), 0.0);
  exp["log_blend"] = log_blend;
  exp["blend_logit_grad"] = bg;
  sidecar["expected"] = std::move(exp);
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (cyphalm_char_lstm)\n";
}

void generate_cyphalm_checkpoint_dir(const fs::path& out_dir, const fs::path& src_dir) {
  const Json src = read_json(src_dir / "sidecar.json");
  copy_fixture_file(src_dir / src.at("checkpoint_json").get<std::string>(),
                    out_dir / src.at("checkpoint_json").get<std::string>());
  cypha::cyphalm::CyphaLMModel model =
      cypha::cyphalm::load_cyphalm_model((out_dir / src.at("checkpoint_json").get<std::string>()).string());
  const auto eval_ids = src.at("eval_ids").get<std::vector<int>>();
  const double bpc = model.eval_bpc(eval_ids, static_cast<int>(eval_ids.size()) - 1);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  if (src.contains("name")) {
    sidecar["name"] = src.at("name");
  }
  sidecar["checkpoint_json"] = src.at("checkpoint_json");
  sidecar["eval_ids"] = eval_ids;
  sidecar["expected_bpc"] = bpc;
  if (src.contains("atol_bpc")) {
    sidecar["atol_bpc"] = src.at("atol_bpc");
  }
  write_json_pretty(out_dir / "sidecar.json", sidecar);
}

void generate_cyphalm_checkpoint_char_lstm(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_cyphalm_checkpoint_dir(out_dir, fixtures_root / "cyphalm_checkpoint" / "char_lstm");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (cyphalm_checkpoint_char_lstm)\n";
}

void generate_cyphalm_checkpoint_hybrid(const fs::path& out_dir, const fs::path& fixtures_root) {
  generate_cyphalm_checkpoint_dir(out_dir, fixtures_root / "cyphalm_checkpoint" / "hybrid");
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (cyphalm_checkpoint_hybrid)\n";
}

void generate_cyphalm_ssm(const fs::path& out_dir, const fs::path& fixtures_root) {
  const Json src = read_json(fixtures_root / "cyphalm_ssm" / "sidecar.json");
  cypha::cyphalm::CellAISSMConfig cfg;
  const auto& c = src.at("config");
  cfg.d_input = c.at("d_input").get<int>();
  cfg.d_state = c.at("d_state").get<int>();
  cfg.tau_fast = c.at("tau_fast").get<double>();
  cfg.tau_slow = c.at("tau_slow").get<double>();
  cfg.n_layers = c.at("n_layers").get<int>();
  cfg.seed = c.at("seed").get<int>();
  cfg.use_spectral_pde = c.at("use_spectral_pde").get<bool>();
  cfg.use_multiscale = c.at("use_multiscale").get<bool>();
  cfg.use_sparse_hebbian = c.at("use_sparse_hebbian").get<bool>();
  cypha::cyphalm::CellAISSM ssm(cfg);
  const auto& wfast = src.at("W_fast");
  const auto& wslow = src.at("W_slow");
  for (int layer = 0; layer < cfg.n_layers; ++layer) {
    ssm.set_projection_weights(layer, flatten_json_mat(wfast[layer]), flatten_json_mat(wslow[layer]));
  }
  std::vector<double> e_t = src.at("e_t").get<std::vector<double>>();
  const auto ctx = ssm.step(e_t);

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  sidecar["config"] = src.at("config");
  sidecar["e_t"] = e_t;
  sidecar["context"] = ctx;
  sidecar["context_dim"] = src.at("context_dim");
  sidecar["W_fast"] = src.at("W_fast");
  sidecar["W_slow"] = src.at("W_slow");
  write_json_pretty(out_dir / "sidecar.json", sidecar);
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (cyphalm_ssm)\n";
}

struct FixtureSpec {
  const char* name;
  const char* description;
  void (*generate)(const fs::path& out_dir, const fs::path& fixtures_root);
};

const FixtureSpec kFixtures[] = {
    {"batch_llr", "batch LLR from raw X (sidecar.json for batch_llr_parity)", generate_batch_llr},
    {"memory_train", "one DIFMemory.train step (sidecar.json + before.cypha)", generate_memory_train},
    {"preprocessor", "Preprocessor transform (sidecar.json + preprocessor.json)", generate_preprocessor},
    {"train_step_vector", "one dif_train_step_vector loss (sidecar.json)", generate_train_step_vector},
    {"quantile_dif_train",
     "quantile DIF train replay + batch LLR (sidecar.json + before.cypha + f_field.json)",
     generate_quantile_dif_train},
    {"regression_m4", "M4 regression golden vectors (sidecar.json)", generate_regression_m4},
    {"score_batch", "accel batch_encode + score_matrix (sidecar.json)", generate_score_batch},
    {"kernel_llr", "KernelMemory phi/score/blend (sidecar.json)", generate_kernel_llr},
    {"gh_infer_deliberation", "gh_infer + deliberation cases (sidecar.json + reference.cypha)", generate_gh_infer_deliberation},
    {"generation", "native generation math (sidecar.json)", generate_generation},
    {"embed_table", "Izaac EmbedTable tokens (sidecar.json)", generate_embed_table},
    {"retrieval", "retrieve_from_x cases (sidecar.json + reference.cypha)", generate_retrieval},
    {"multilabel_dif", "MultiLabelDif train/predict (sidecar.json)", generate_multilabel_dif},
    {"merge_from", "memory_merge_from (sidecar.json + cypha files)", generate_merge_from},
    {"similarity_index", "SimilarityIndex query (sidecar.json + reference.cypha)", generate_similarity_index},
    {"dif_regressor_train_step", "DIFRegressor online step (sidecar.json + before.cypha)", generate_dif_regressor_train_step},
    {"mke_train_step", "MKERegressor single train step (sidecar.json + before.cypha)", generate_mke_train_step},
    {"mke_train_extended", "MKERegressor multi-step extended (sidecar.json + before.cypha)", generate_mke_train_extended},
    {"dif_train_replay", "DIF train replay quantile path (sidecar.json + before.cypha)", generate_dif_train_replay},
    {"studio_trainer_classify_hotpath", "Trainer classify hotpath (sidecar.json + before.cypha)", generate_studio_trainer_classify_hotpath},
    {"studio_trainer_gh_classify_hotpath", "Trainer GH classify hotpath (sidecar.json + before.cypha)", generate_studio_trainer_gh_classify_hotpath},
    {"rff_regression", "RFF ridge + MKE dots (sidecar.json)", generate_rff_regression},
    {"two_stage_pipeline", "two_stage_dif_predict_with_clf (sidecar.json)", generate_two_stage_pipeline},
    {"two_stage_ridge_fit", "two_stage_dif_ridge_fit_from_llr (sidecar.json)", generate_two_stage_ridge_fit},
    {"two_stage_e2e_ridge", "two_stage e2e ridge fit (sidecar.json)", generate_two_stage_e2e_ridge},
    {"csv_preprocess_classify_hotpath", "CSV ingest + preprocess train (sidecar.json)", generate_csv_preprocess_classify_hotpath},
    {"studio_trainer_preprocess_classify_hotpath", "preprocess + classify hotpath (sidecar.json)", generate_studio_trainer_preprocess_classify_hotpath},
    {"studio_trainer_preprocess_gh_classify_hotpath", "preprocess + GH classify hotpath (sidecar.json)", generate_studio_trainer_preprocess_gh_classify_hotpath},
    {"cyphalm_char_lstm", "Char-LSTM + GRIA + blend (sidecar.json)", generate_cyphalm_char_lstm},
    {"cyphalm_checkpoint_char_lstm", "CyphaLM checkpoint BPC char_lstm (sidecar.json + checkpoint.json)", generate_cyphalm_checkpoint_char_lstm},
    {"cyphalm_checkpoint_hybrid", "CyphaLM checkpoint BPC hybrid (sidecar.json + checkpoint.json)", generate_cyphalm_checkpoint_hybrid},
    {"cyphalm_ssm", "CellAISSM step context (sidecar.json)", generate_cyphalm_ssm},
};

const FixtureSpec* find_fixture(const std::string& name) {
  for (const FixtureSpec& f : kFixtures) {
    if (name == f.name) {
      return &f;
    }
  }
  return nullptr;
}

void list_fixtures() {
  for (const FixtureSpec& f : kFixtures) {
    std::cout << f.name << "\t" << f.description << '\n';
  }
}

fs::path resolve_fixtures_root(const Args& a) {
  if (!a.fixtures_root.empty()) {
    return fs::absolute(a.fixtures_root);
  }
  if (a.out.empty()) {
    throw std::runtime_error("--fixtures-root required when --out is omitted");
  }
  return fs::absolute(a.out).parent_path();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (args.list_only) {
      list_fixtures();
      return 0;
    }
    if (args.fixture.empty() || args.out.empty()) {
      usage();
      return 2;
    }
    const FixtureSpec* spec = find_fixture(args.fixture);
    if (!spec) {
      std::cerr << "unknown fixture: " << args.fixture << " (try --list)\n";
      return 2;
    }
    const fs::path fixtures_root = resolve_fixtures_root(args);
    const fs::path out_dir = fs::absolute(args.out);
    spec->generate(out_dir, fixtures_root);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cypha_fixture_gen: " << e.what() << '\n';
    return 1;
  }
}
