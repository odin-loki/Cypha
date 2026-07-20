#pragma once

/// Single public Cypha type: classify + regress + latent sample + tokens.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/predictive_codec.hpp"
#include "cypha/ewc_regularizer.hpp"
#include "cypha/generation.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/intelligence/epistemic_threshold.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/retrieval.hpp"
#include "cypha/train_step_vector.hpp"

namespace cypha {

struct PredictOpts {
  bool use_gh{true};
  bool use_field{true};
  double deliberation_lo{kDeliberationLoDefault};
  double deliberation_hi{kDeliberationHiDefault};
  bool self_correct{false};
  int self_correct_max_passes{3};
};

struct PredictOut {
  std::string label;
  double confidence{0.0};
  std::vector<double> all_scores;  // LLR per label order
  std::vector<std::string> labels;
  std::optional<double> y;
  double uncertainty{0.0};
  double anomaly_score{0.0};
  bool is_ood{false};
  double r_eff{0.0};
  bool self_corrected{false};
  int correction_passes{0};
  double r_eu_proxy{0.0};
  std::string detail;  // non-empty => error
};

struct UpdateOpts {
  bool use_gh{true};
  const std::string* router_train_label{nullptr};
  /// Optional deterministic replay stream (lifetime must cover the update call).
  const double* replay_u01{nullptr};
  std::size_t replay_u01_len{0};
  bool ewc_snapshot{false};
  /// NaN = use Cypha's stored ``ewc_lambda_``; else override for this step (and store).
  double ewc_lambda{std::numeric_limits<double>::quiet_NaN()};
};

struct UpdateOut {
  double loss{0.0};
  std::string detail;
};

enum class SampleMode {
  Langevin,
  FromObservation,
  RetrievalAugmented,
  ClassGaussian,
  Conditioned,
};

struct SampleOpts {
  SampleMode mode{SampleMode::Langevin};
  std::string label;  // empty => infer
  int n_samples{10};
  int n_steps{30};
  double temperature{1.0};
  double step_size{0.05};
  int k_neighbors{5};
  /// Raw query features (preprocessed by caller if needed). Required.
  const double* x{nullptr};
  int x_dim{0};
  /// Flat database rows for RAG (n_db * input_dim).
  const double* database_x{nullptr};
  int n_db{0};
  int database_input_dim{0};
  CyphaInferOptions infer_opt{};
};

struct SampleOut {
  std::vector<std::vector<double>> h;
  std::string label;
  std::string mode;
  std::string detail;
};

struct RetrieveOpts {
  int top_k{5};
  CyphaInferOptions infer_opt{};
  std::optional<std::string> label;
};

struct RetrieveOut {
  std::vector<RetrieveHit> hits;
  std::string detail;
};

struct TokenOut {
  std::vector<double> log_probs;
  std::vector<std::uint32_t> top_k_tokens;
  std::vector<double> top_k_probs;
  std::string detail;
};

struct GenerateTokenOpts {
  cyphalm::DecodeParams decode{};
  int max_tokens{32};
};

/// One Cypha instance: tabular classify/regress/latent-sample + sequence tokens.
class Cypha {
 public:
  Cypha();
  ~Cypha();

  Cypha(const Cypha&) = delete;
  Cypha& operator=(const Cypha&) = delete;
  Cypha(Cypha&&) noexcept;
  Cypha& operator=(Cypha&&) noexcept;

  /// Load `.cypha` bundle. Optional preprocessor / F_field JSON / regression_head.json.
  bool load(const std::string& cypha_path, const std::string& preprocessor_path = {},
            const std::string& f_field_json_path = {}, const std::string& regression_json_path = {},
            const std::string& train_hparams_path = {});

  /// Load sequence checkpoint (JSON config path).
  bool load_sequence(const std::string& json_path);
  /// Fresh sequence model: Hybrid GRIA+LSTM production recipe (~2.8 BPC after ~300k train).
  bool init_default_sequence(int vocab_size = 256, int d_model = 64);

  void save(const std::string& cypha_path) const;
  void save_sequence(const std::string& base_path) const;

  bool loaded() const { return infer_ != nullptr; }
  bool sequence_loaded() const { return seq_ != nullptr; }

  PredictOut predict(const double* x, int n, const PredictOpts& o = {});
  UpdateOut update(const double* x, int n, const std::string* label, const double* y,
                   const UpdateOpts& o = {});

  SampleOut sample(const SampleOpts& o);
  RetrieveOut retrieve(const double* x, int n, const double* database_x, int n_db, int input_dim,
                       const RetrieveOpts& o = {});

  TokenOut predict_next(std::uint32_t token);
  double train_token(std::uint32_t token, std::uint32_t next);
  std::string generate(const std::vector<int>& prompt_ids, const GenerateTokenOpts& o = {});

  /// Predictive arithmetic coding (LLMZip-style): model probs → entropy-coded bitstream.
  /// Default options enable the adaptive predictor mixer (neural + online n-grams).
  cyphalm::PredictiveCodecResult compress_tokens(
      const std::vector<std::uint32_t>& tokens, const cyphalm::PredictiveCodecOptions& opt = {});
  std::vector<std::uint32_t> decompress_tokens(const std::vector<std::uint8_t>& bytes,
                                               std::uint32_t seed, std::size_t n_tokens,
                                               std::string* detail = nullptr,
                                               const cyphalm::PredictiveCodecOptions& opt = {});
  std::vector<std::uint32_t> generate_via_bits(const std::vector<std::uint32_t>& prefix,
                                               std::size_t n_new, std::uint64_t rng_seed,
                                               std::string* detail = nullptr,
                                               const cyphalm::PredictiveCodecOptions& opt = {});

  // Accessors for REST / tools during cutover (non-owning).
  CyphaInferModel* infer() { return infer_.get(); }
  const CyphaInferModel* infer() const { return infer_.get(); }
  CyphaDifMemoryState* mem() { return mem_.get(); }
  PreprocessorState* preprocessor() { return pre_.get(); }
  ReplayBuffer* replay() { return replay_.get(); }
  cyphalm::CyphaLMModel* sequence() { return seq_.get(); }
  std::mt19937& rng() { return rng_; }

  /// Owned pointers for REST ModelView wiring (primary process instance).
  std::unique_ptr<CyphaInferModel>& infer_owned() { return infer_; }
  std::unique_ptr<CyphaDifMemoryState>& mem_owned() { return mem_; }
  std::unique_ptr<PreprocessorState>& preprocessor_owned() { return pre_; }
  std::unique_ptr<ReplayBuffer>& replay_owned() { return replay_; }
  std::unique_ptr<KernelMemory>& kernel_mem_owned() { return kernel_mem_; }

  std::vector<double>& reg_mu() { return reg_mu_; }
  std::vector<double>& reg_var() { return reg_var_; }
  bool mke_active() const { return mke_active_; }
  bool& mke_active_ref() { return mke_active_; }
  int mke_d_in() const { return mke_d_in_; }
  int& mke_d_in_ref() { return mke_d_in_; }
  std::vector<double>& mke_W() { return mke_W_; }
  std::vector<double>& mke_b() { return mke_b_; }
  double& mke_temperature() { return mke_temperature_; }
  double& mke_forgetting() { return mke_forgetting_; }
  double& mke_pi_floor() { return mke_pi_floor_; }
  std::vector<double>& mke_gh_scales() { return mke_gh_scales_; }
  std::unordered_map<std::string, std::vector<double>>& mke_w() { return mke_w_; }
  std::unordered_map<std::string, std::vector<double>>& mke_p() { return mke_p_; }

  double& world_lr() { return world_lr_; }
  double& delta_lr() { return delta_lr_; }
  double& ood_sigma() { return ood_sigma_; }
  TrainStepParams& train_params() { return tsp_; }
  int& enc_updates() { return enc_updates_; }
  int& total_steps() { return total_steps_; }
  double& llr_ema() { return llr_ema_; }
  std::vector<double>& gh_inv_v_clean() { return gh_inv_v_clean_; }
  double& gh_R_base() { return gh_R_base_; }
  double& gh_chi() { return gh_chi_; }
  double& gh_psi() { return gh_psi_; }
  KernelMemory* kernel_mem() { return kernel_mem_.get(); }
  bool& use_kernel_llr() { return use_kernel_llr_; }
  double& kernel_blend() { return kernel_blend_; }
  intelligence::EpistemicThreshold& epistemic_threshold() { return epistemic_threshold_; }
  const intelligence::EpistemicThreshold& epistemic_threshold() const { return epistemic_threshold_; }

  EwcRegularizer* ewc() { return ewc_.get(); }
  const EwcRegularizer* ewc() const { return ewc_.get(); }
  double& ewc_lambda() { return ewc_lambda_; }
  const double& ewc_lambda() const { return ewc_lambda_; }

  bool load_regression_json(const std::string& path);

 private:
  void clear_mke();
  void apply_default_hparams();
  bool load_ff_json(const std::string& path, int d, int fd, std::vector<double>& out) const;
  static std::string mode_name(SampleMode m);

  std::unique_ptr<CyphaInferModel> infer_;
  std::unique_ptr<CyphaDifMemoryState> mem_;
  std::unique_ptr<PreprocessorState> pre_;
  std::unique_ptr<ReplayBuffer> replay_;
  std::unique_ptr<KernelMemory> kernel_mem_;
  std::unique_ptr<cyphalm::CyphaLMModel> seq_;
  /// Template `.cypha` root retained for ``save`` (world/classes merged from ``mem_``).
  CNode root_;

  std::vector<double> reg_mu_;
  std::vector<double> reg_var_;
  bool mke_active_{false};
  int mke_d_in_{0};
  std::vector<double> mke_W_;
  std::vector<double> mke_b_;
  double mke_temperature_{1.0};
  double mke_forgetting_{1.0};
  double mke_pi_floor_{0.02};
  std::vector<double> mke_gh_scales_;
  std::unordered_map<std::string, std::vector<double>> mke_w_;
  std::unordered_map<std::string, std::vector<double>> mke_p_;

  double world_lr_{0.008};
  double delta_lr_{0.05};
  double ood_sigma_{15.0};
  TrainStepParams tsp_{};
  int enc_updates_{0};
  int total_steps_{0};
  double llr_ema_{0.0};
  std::vector<double> gh_inv_v_clean_;
  double gh_R_base_{1.0};
  double gh_chi_{1.0};
  double gh_psi_{1.0};
  bool use_kernel_llr_{false};
  double kernel_blend_{0.5};
  intelligence::EpistemicThreshold epistemic_threshold_{0.5, 5.0};
  std::unique_ptr<EwcRegularizer> ewc_;
  double ewc_lambda_{0.0};

  std::mt19937 rng_{424242};
  static constexpr double kOodThreshold = 3.0;
  static constexpr double kGhNigAdaptAlpha = 0.98;
};

}  // namespace cypha
