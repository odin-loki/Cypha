// cypha_fixture_gen — native parity fixture regeneration (replaces removed Python generators).
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"

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

struct FixtureSpec {
  const char* name;
  const char* description;
  void (*generate)(const fs::path& out_dir, const fs::path& fixtures_root);
};

const FixtureSpec kFixtures[] = {
    {"batch_llr", "batch LLR from raw X (sidecar.json for batch_llr_parity)", generate_batch_llr},
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
