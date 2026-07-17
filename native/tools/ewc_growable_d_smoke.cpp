/// Regression + parity test for the `EwcRegularizer` growable-`D` prefix fix and the new opt-in
/// NIG world-field (`world_mu`) protection. See docs/reports/EWC_D16B_SCOPING_2026-07-12.md.
///
/// Before 2026-07-12, `EwcRegularizer::penalty()`/`apply_pull()` required an *exact* size match
/// between the live parameter vector and the snapshot-time anchor. `D` (the class-delta score
/// matrix) grows append-only every time a brand-new class is trained after `snapshot()` -- exactly
/// what happens in any multi-task continual-learning run once task B introduces classes task A
/// never saw (bench D16B). That made the `D` term of the penalty/pull silently degrade to a no-op
/// the instant task B's first new class was trained, even though `ewc_lambda > 0` and a snapshot
/// existed. This test pins down: (1) the fixed behavior computes a real, hand-verifiable
/// prefix-only penalty after `D` grows, (2) it is deterministic, and (3) `set_protect_world_field`
/// engages an additional, independently-verifiable term.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/ewc_regularizer.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

namespace {

struct Fixture {
  cypha::CyphaInferModel infer;
  cypha::CyphaDifMemoryState mem;
  cypha::ReplayBuffer replay{100};
  cypha::TrainStepParams tsp{};
  std::mt19937 rng{7};
  int enc_updates{0};
};

Fixture make_fixture() {
  cypha::FreshModelParams fp;
  fp.input_dim = 5;
  fp.field_dim = 16;
  const cypha::CNode root = cypha::create_fresh_model_root(fp);
  Fixture f{cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim),
            cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim)};
  return f;
}

void train_one(Fixture& f, const std::vector<double>& x, const std::string& label,
               cypha::TrainStepExtras* extras) {
  (void)cypha::dif_train_step_vector(f.infer, f.mem, f.replay, x.data(), static_cast<int>(x.size()), label, 0.008,
                                     0.03, 0.008, 0.03, 12.0, f.tsp, f.rng, f.enc_updates, nullptr, extras);
}

}  // namespace

int main() {
  const std::vector<double> xa = {0.1, 0.2, 0.3, 0.4, 0.5};
  const std::vector<double> xb = {0.9, 0.8, 0.1, 0.05, 0.4};

  // --- 1. Growable-D prefix regression check -------------------------------------------------
  {
    Fixture f = make_fixture();
    cypha::EwcRegularizer ewc;
    cypha::TrainStepExtras extras{};
    extras.ewc = &ewc;
    extras.ewc_lambda = 0.5;

    train_one(f, xa, "class_a", nullptr);  // task A: only "class_a" exists at snapshot time.
    ewc.snapshot(f.mem, f.infer);
    const std::size_t anchor_d_size = f.mem.D.size();  // == 1 * d_latent here.
    assert(anchor_d_size > 0);

    // Task B introduces a brand-new class -> mem.D grows (append-only) past anchor_d_size.
    train_one(f, xb, "class_b", &extras);
    assert(f.mem.D.size() > anchor_d_size);

    const double penalty_after_growth = ewc.penalty(f.mem, f.infer);
    if (!(penalty_after_growth > 0.0)) {
      std::fprintf(stderr,
                   "ewc_growable_d_smoke: FAIL penalty==%.6g after D grew past the anchor size -- "
                   "the growable-D prefix fix has regressed back to the pre-2026-07-12 silent no-op\n",
                   penalty_after_growth);
      return 1;
    }

    // Parity: hand-compute the same anchor-prefix squared penalty independently (this is exactly
    // what `snapshot()`'s crude anchor-squared Fisher proxy + prefix-only comparison should give:
    // 0.5 * sum_{i<anchor_size}( (anchor_i^2+eps) * (D_i - anchor_i)^2 ) for the D term, plus the
    // enc_w term at full size since enc_w never grows).
    double expected = 0.0;
    {
      // Re-derive the anchor by re-running task A alone on a fresh fixture (deterministic).
      Fixture anchor_fixture = make_fixture();
      train_one(anchor_fixture, xa, "class_a", nullptr);
      const auto& anchor_D = anchor_fixture.mem.D;
      const auto& anchor_enc_w = anchor_fixture.infer.enc_w;
      constexpr double kFisherEps = 1e-8;
      for (std::size_t i = 0; i < anchor_D.size(); ++i) {
        const double fisher = anchor_D[i] * anchor_D[i] + kFisherEps;
        const double d = f.mem.D[i] - anchor_D[i];
        expected += fisher * d * d;
      }
      assert(anchor_enc_w.size() == f.infer.enc_w.size());
      for (std::size_t i = 0; i < anchor_enc_w.size(); ++i) {
        const double fisher = anchor_enc_w[i] * anchor_enc_w[i] + kFisherEps;
        const double d = f.infer.enc_w[i] - anchor_enc_w[i];
        expected += fisher * d * d;
      }
      expected *= 0.5;
    }
    const double rel_err = std::abs(expected - penalty_after_growth) / std::max(std::abs(expected), 1e-12);
    if (rel_err > 1e-9) {
      std::fprintf(stderr,
                   "ewc_growable_d_smoke: FAIL penalty parity mismatch: hand-computed=%.10g engine=%.10g "
                   "rel_err=%.3g\n",
                   expected, penalty_after_growth, rel_err);
      return 1;
    }

    // Determinism: an identical fixture, replayed identically, must reproduce the same penalty
    // bit-for-bit (same seed, same steps, same lambda).
    Fixture f2 = make_fixture();
    cypha::EwcRegularizer ewc2;
    cypha::TrainStepExtras extras2{};
    extras2.ewc = &ewc2;
    extras2.ewc_lambda = 0.5;
    train_one(f2, xa, "class_a", nullptr);
    ewc2.snapshot(f2.mem, f2.infer);
    train_one(f2, xb, "class_b", &extras2);
    const double penalty_repeat = ewc2.penalty(f2.mem, f2.infer);
    if (penalty_repeat != penalty_after_growth) {
      std::fprintf(stderr,
                   "ewc_growable_d_smoke: FAIL nondeterministic: run1=%.17g run2=%.17g\n",
                   penalty_after_growth, penalty_repeat);
      return 1;
    }
    std::printf("ewc_growable_d_smoke: growth penalty=%.6g (parity + determinism OK)\n", penalty_after_growth);
  }

  // --- 2. NIG world-field (world_mu) opt-in protection ----------------------------------------
  {
    Fixture f_off = make_fixture();
    Fixture f_on = make_fixture();
    cypha::EwcRegularizer ewc_off;
    cypha::EwcRegularizer ewc_on;
    ewc_on.set_protect_world_field(true);
    assert(!ewc_off.protect_world_field());
    assert(ewc_on.protect_world_field());

    cypha::TrainStepExtras extras_off{};
    extras_off.ewc = &ewc_off;
    extras_off.ewc_lambda = 0.5;
    cypha::TrainStepExtras extras_on{};
    extras_on.ewc = &ewc_on;
    extras_on.ewc_lambda = 0.5;

    train_one(f_off, xa, "class_a", nullptr);
    train_one(f_on, xa, "class_a", nullptr);
    ewc_off.snapshot(f_off.mem, f_off.infer);
    ewc_on.snapshot(f_on.mem, f_on.infer);

    // world_mu is fixed-size (d_latent) and identical between the two anchors at this point
    // (both fixtures trained identically up to here) -- protect_world_field must not change D/
    // enc_w anchoring, only add the extra world_mu term.
    train_one(f_off, xb, "class_b", &extras_off);
    train_one(f_on, xb, "class_b", &extras_on);

    const double penalty_off = ewc_off.penalty(f_off.mem, f_off.infer);
    const double penalty_on = ewc_on.penalty(f_on.mem, f_on.infer);
    // D/enc_w drift should match between the two runs (protect_world_field only adds a pull term
    // on world_mu, and world_mu's own pull does not feed back into D/enc_w's update equations),
    // so `penalty_on` must be strictly larger by exactly the (non-negative) world_mu term.
    if (!(penalty_on >= penalty_off)) {
      std::fprintf(stderr,
                   "ewc_growable_d_smoke: FAIL protect_world_field made penalty smaller (off=%.6g on=%.6g)\n",
                   penalty_off, penalty_on);
      return 1;
    }
    if (!(penalty_on > penalty_off)) {
      std::fprintf(stderr,
                   "ewc_growable_d_smoke: FAIL protect_world_field added no measurable penalty "
                   "(off=%.6g on=%.6g) -- world_mu term looks inert\n",
                   penalty_off, penalty_on);
      return 1;
    }
    std::printf("ewc_growable_d_smoke: world-field penalty off=%.6g on=%.6g PASS\n", penalty_off, penalty_on);
  }

  std::printf("ewc_growable_d_smoke: PASS\n");
  return 0;
}
