// cypha_fixture_gen — native parity fixture regeneration (replaces removed Python generators).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
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

void generate_quantile_dif_train(const fs::path& out_dir, const fs::path& fixtures_root) {
  const fs::path src_dir = fixtures_root / "quantile_dif_train";
  const Json src = read_json(src_dir / "sidecar.json");

  copy_fixture_file(src_dir / "before.cypha", out_dir / "before.cypha");
  copy_fixture_file(src_dir / "f_field.json", out_dir / "f_field.json");

  auto [infer, mem] = load_infer_and_mem(out_dir / "before.cypha", out_dir / "f_field.json");
  const int d = infer.d_latent;
  const int d_in = src.at("d_in").get<int>();
  const int n = src.at("n").get<int>();
  const int K = src.at("K").get<int>();
  if (d_in != d) {
    throw std::runtime_error("d_in mismatch enc_W");
  }

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
  const unsigned rseed = static_cast<unsigned>(src.value("rng_seed", 7755));
  std::mt19937 rng{rseed};
  int enc_updates = 0;
  int total_steps = 0;
  cypha::TrainStepExtras extras{};
  extras.total_steps = &total_steps;
  extras.ood_sigma = nullptr;
  extras.llr_ema = nullptr;

  const auto& steps = src.at("steps");
  std::vector<std::vector<double>> xs;
  std::vector<std::string> labels;
  xs.reserve(steps.size());
  labels.reserve(steps.size());
  for (const auto& step : steps) {
    std::vector<double> x;
    for (const auto& v : step.at("x")) {
      x.push_back(v.get<double>());
    }
    if (static_cast<int>(x.size()) != d) {
      throw std::runtime_error("step x dim mismatch");
    }
    xs.push_back(std::move(x));
    labels.push_back(step.at("label").get<std::string>());
  }

  const std::vector<double> got_losses = cypha::dif_train_classify_sequence(
      infer, mem, replay, xs, labels, world_lr, delta_lr, world_lr, delta_lr, ood_sigma, tsp, rng, enc_updates,
      &extras);

  const std::vector<double> x_all = src.at("x_rowmajor").get<std::vector<double>>();
  if (static_cast<int>(x_all.size()) != n * d) {
    throw std::runtime_error("x_rowmajor size mismatch");
  }
  std::vector<double> llr;
  cypha::batch_llr_from_x(infer, x_all.data(), n, llr);
  if (static_cast<int>(llr.size()) != n * K) {
    throw std::runtime_error("llr size mismatch");
  }

  nlohmann::ordered_json sidecar = nlohmann::ordered_json::object();
  if (src.contains("fixture_schema")) {
    sidecar["fixture_schema"] = src.at("fixture_schema");
  }
  if (src.contains("description")) {
    sidecar["description"] = src.at("description");
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
  std::cout << "wrote " << (out_dir / "sidecar.json").string() << " (n_steps=" << steps.size() << " n=" << n
            << ")\n";
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
