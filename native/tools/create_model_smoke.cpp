/// Smoke test for create_fresh_model_root: creates an empty model, verifies
/// CyphaInferModel and CyphaDifMemoryState load from it, and checks save/reload
/// roundtrip.  Exit 0 on success.
#include <cassert>
#include <cstdio>

#include "cypha/create_model.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"

int main() {
    cypha::FreshModelParams p;
    p.input_dim = 8;
    p.field_dim = 16;
    p.temperature = 1.0;

    // Build root
    cypha::CNode root = cypha::create_fresh_model_root(p);

    // Load infer model
    auto m = cypha::CyphaInferModel::from_root(root, nullptr, p.field_dim);
    assert(m.d_latent == p.input_dim);
    assert(m.labels.empty());
    assert(m.field_dim == p.field_dim);
    assert(m.temperature > 0.0);

    // Load train state
    auto s = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, p.field_dim);
    assert(s.labels.empty());
    assert(static_cast<int>(m.enc_w.size()) == p.input_dim * p.input_dim);

    // Save + reload roundtrip
    const char* tmp_path = "/tmp/cypha_create_model_smoke.cypha";
    cypha::save_cypha_file(tmp_path, root);
    cypha::CNode root2 = cypha::load_cypha_file(tmp_path);
    auto m2 = cypha::CyphaInferModel::from_root(root2, nullptr, p.field_dim);
    assert(m2.d_latent == p.input_dim);
    assert(m2.labels.empty());

    // Different dims
    cypha::FreshModelParams p2;
    p2.input_dim = 32;
    p2.field_dim = 64;
    cypha::CNode root3 = cypha::create_fresh_model_root(p2);
    auto m3 = cypha::CyphaInferModel::from_root(root3, nullptr, p2.field_dim);
    assert(m3.d_latent == 32);
    assert(m3.field_dim == 64);

    std::puts("create_model_smoke: PASS");
    return 0;
}
