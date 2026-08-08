// One Cypha smoke: load → classify → (optional regress) → sample latents → tokens → generate.
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cypha/cypha.hpp"
#include "cypha/cyphalm/predictive_codec.hpp"

int main(int argc, char** argv) {
  std::string cypha_path;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--cypha" && i + 1 < argc) {
      cypha_path = argv[++i];
    }
  }
  if (cypha_path.empty()) {
    const char* candidates[] = {
        "fixtures/reference.cypha",
        "../fixtures/reference.cypha",
        "../../fixtures/reference.cypha",
    };
    for (const char* c : candidates) {
      if (std::filesystem::exists(c)) {
        cypha_path = c;
        break;
      }
    }
  }
  if (cypha_path.empty()) {
    std::cerr << "usage: cypha_one_smoke --cypha path/to/model.cypha\n";
    return 1;
  }

  std::string regression_json_path;
  {
    const std::filesystem::path beside =
        std::filesystem::path(cypha_path).parent_path() / "regression_head.json";
    if (std::filesystem::exists(beside)) {
      regression_json_path = beside.string();
    }
  }
  if (regression_json_path.empty()) {
    const char* candidates[] = {
        "fixtures/regression_head.json",
        "../fixtures/regression_head.json",
        "../../fixtures/regression_head.json",
    };
    for (const char* c : candidates) {
      if (std::filesystem::exists(c)) {
        regression_json_path = c;
        break;
      }
    }
  }
  const bool regression_expected = !regression_json_path.empty();

  cypha::Cypha model;
  if (!model.load(cypha_path, {}, {}, regression_json_path)) {
    std::cerr << "load failed: " << cypha_path << "\n";
    return 1;
  }
  const int d = model.infer()->d_latent;
  std::vector<double> x(static_cast<std::size_t>(d), 0.1);
  auto pred = model.predict(x.data(), d);
  if (!pred.detail.empty()) {
    std::cerr << "predict: " << pred.detail << "\n";
    return 2;
  }
  std::cout << "predict label=" << pred.label << " conf=" << pred.confidence << "\n";
  if (regression_expected) {
    if (!pred.y.has_value()) {
      std::cerr << "regress: regression_head.json present but predict y is null\n";
      return 10;
    }
    std::cout << "predict regress y=" << *pred.y << " uncertainty=" << pred.uncertainty << "\n";
  } else {
    std::cout << "regress: skip (no regression_head.json)\n";
  }

  std::string lbl = pred.label;
  auto upd = model.update(x.data(), d, &lbl, nullptr);
  if (!upd.detail.empty()) {
    std::cerr << "update: " << upd.detail << "\n";
    return 3;
  }
  std::cout << "update loss=" << upd.loss << "\n";

  const auto save_path =
      (std::filesystem::temp_directory_path() / "cypha_one_smoke_save.cypha").string();
  try {
    model.save(save_path);
  } catch (const std::exception& ex) {
    std::cerr << "save: " << ex.what() << "\n";
    return 7;
  }
  cypha::Cypha reloaded;
  if (!reloaded.load(save_path)) {
    std::cerr << "reload after save failed: " << save_path << "\n";
    return 8;
  }
  auto pred2 = reloaded.predict(x.data(), d);
  if (!pred2.detail.empty() || pred2.label.empty()) {
    std::cerr << "predict after save/load failed\n";
    return 9;
  }
  std::cout << "save/load label=" << pred2.label << "\n";

  cypha::SampleOpts sopt;
  sopt.mode = cypha::SampleMode::Langevin;
  sopt.x = x.data();
  sopt.x_dim = d;
  sopt.n_samples = 2;
  sopt.n_steps = 4;
  auto samp = model.sample(sopt);
  if (!samp.detail.empty()) {
    std::cerr << "sample: " << samp.detail << "\n";
    return 4;
  }
  std::cout << "sample n=" << samp.h.size() << " label=" << samp.label << "\n";

  if (!model.init_default_sequence(64, 32)) {
    std::cerr << "init_default_sequence failed\n";
    return 5;
  }
  if (!model.sequence() ||
      model.sequence()->config().context_mode != cypha::cyphalm::ContextMode::Hybrid) {
    std::cerr << "init_default_sequence should use Hybrid production recipe\n";
    return 10;
  }
  (void)model.train_token(1, 2);
  auto tok = model.predict_next(1);
  if (!tok.detail.empty()) {
    std::cerr << "predict_next: " << tok.detail << "\n";
    return 6;
  }
  std::cout << "predict_next log_probs=" << tok.log_probs.size()
            << " mode=" << cypha::cyphalm::context_mode_name(model.sequence()->config().context_mode)
            << "\n";

  cypha::GenerateTokenOpts gopt;
  gopt.max_tokens = 4;
  gopt.decode.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
  const std::string text = model.generate({1, 2, 3}, gopt);
  std::cout << "generate chars=" << text.size() << "\n";

  const std::vector<std::uint32_t> stream = {1, 2, 3, 4, 5, 2, 3, 4};
  for (std::size_t i = 0; i + 1 < stream.size(); ++i) {
    (void)model.train_token(stream[i], stream[i + 1]);
  }
  const auto seq_ckpt = std::filesystem::temp_directory_path() / "cypha_one_smoke_seq";
  model.save_sequence(seq_ckpt.string());
  const std::string ckpt_json = seq_ckpt.string() + ".json";

  cypha::cyphalm::PredictiveCodecOptions codec_opt;
  codec_opt.online_adapt = false;
  cypha::Cypha enc;
  cypha::Cypha dec;
  if (!enc.load_sequence(ckpt_json) || !dec.load_sequence(ckpt_json)) {
    std::cerr << "reload sequence for codec failed\n";
    return 11;
  }
  auto packed = enc.compress_tokens(stream, codec_opt);
  if (!packed.detail.empty() || packed.bytes.empty()) {
    std::cerr << "compress_tokens: " << packed.detail << "\n";
    return 11;
  }
  std::string detail;
  auto round = dec.decompress_tokens(packed.bytes, stream.front(), stream.size(), &detail, codec_opt);
  if (!detail.empty() || round != stream) {
    std::cerr << "compress/decompress roundtrip failed: " << detail << "\n";
    return 12;
  }
  std::cout << "predictive_codec model_bpc=" << packed.model_bpc << " coded_bpc=" << packed.coded_bpc
            << "\n";
  std::cout << "one_cypha_smoke OK\n";
  return 0;
}
