// generation_parity — verify native generation math against Python fixture.
// Usage: generation_parity <parity_fixtures/generation/sidecar.json>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/generation.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

bool near_all(const std::vector<double>& got, const std::vector<double>& exp,
              double atol) {
    if (got.size() != exp.size()) return false;
    for (std::size_t i = 0; i < got.size(); i++) {
        if (std::abs(got[i] - exp[i]) > atol) return false;
    }
    return true;
}

std::vector<double> read_vec(const json& j) {
    return j.get<std::vector<double>>();
}

std::vector<std::vector<double>> read_matrix(const json& j) {
    std::vector<std::vector<double>> m;
    for (const auto& row : j) {
        m.push_back(read_vec(row));
    }
    return m;
}

/// Flatten list-of-lists to row-major double vector.
std::vector<double> flatten_mat(const json& j) {
    std::vector<double> out;
    for (const auto& row : j) {
        for (const auto& v : row) out.push_back(v.get<double>());
    }
    return out;
}

/// Flatten 3-D list to row-major double vector.
std::vector<double> flatten_3d(const json& j) {
    std::vector<double> out;
    for (const auto& block : j)
        for (const auto& row : block)
            for (const auto& v : row) out.push_back(v.get<double>());
    return out;
}

std::vector<double> flatten_f_field(const json& j) {
    return flatten_mat(j);
}

bool check_case(const std::string& name,
                const std::vector<std::vector<double>>& got,
                const json& exp_j,
                double atol,
                int d) {
    auto exp_mat = read_matrix(exp_j);
    if (got.size() != exp_mat.size()) {
        std::cerr << "FAIL " << name << ": n mismatch got=" << got.size()
                  << " exp=" << exp_mat.size() << "\n";
        return false;
    }
    for (std::size_t i = 0; i < got.size(); i++) {
        if (!near_all(got[i], exp_mat[i], atol)) {
            std::cerr << "FAIL " << name << "[" << i << "]: max_err=";
            double max_e = 0.0;
            for (int j = 0; j < d; j++) {
                max_e = std::max(max_e, std::abs(got[i][static_cast<std::size_t>(j)]
                                                 - exp_mat[i][static_cast<std::size_t>(j)]));
            }
            std::cerr << max_e << " atol=" << atol << "\n";
            return false;
        }
    }
    std::cout << "PASS " << name << " (" << got.size() << " samples)\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: generation_parity <parity_fixtures/generation/sidecar.json>\n";
            return 2;
        }

        fs::path side = fs::path(argv[1]);
        fs::path fix_root = side.parent_path().parent_path();

        std::ifstream sf(side);
        if (!sf) throw std::runtime_error("cannot open sidecar: " + side.string());
        std::stringstream buf;
        buf << sf.rdbuf();
        auto j = json::parse(buf.str());

        const double atol = j.at("atol").get<double>();
        const int d = j.at("d_latent").get<int>();
        const int fd = j.at("field_dim").get<int>();
        const int input_dim = j.value("input_dim", d);

        // Load reference.cypha
        fs::path cypha_path = fix_root / "reference.cypha";
        cypha::CNode root_node = cypha::load_cypha_file(cypha_path.string().c_str());

        // Load f_field.json (F_field matrix for conditioned generation)
        fs::path ff_path = fix_root / "f_field.json";
        std::vector<double> f_field_flat;
        {
            std::ifstream jf(ff_path);
            if (!jf) throw std::runtime_error("cannot open f_field.json");
            std::stringstream fj;
            fj << jf.rdbuf();
            f_field_flat = flatten_f_field(json::parse(fj.str()));
        }

        cypha::CyphaInferModel m = cypha::CyphaInferModel::from_root(
            root_node, f_field_flat.data(), fd);

        const auto& cases = j.at("cases");
        bool all_pass = true;

        // ── 1. generate_gaussian_no_rejection ────────────────────────────────
        {
            const auto& c = cases.at("generate_gaussian_no_rejection");
            std::string lbl = c.at("label").get<std::string>();
            int n = c.at("n").get<int>();
            double T = c.at("temperature").get<double>();
            std::vector<double> z = flatten_mat(c.at("z"));
            auto got = cypha::generate_class_gaussian(
                m, lbl, n, T, nullptr, /*rejection=*/false, 16, z.data());
            all_pass &= check_case("generate_gaussian_no_rejection", got,
                                   c.at("expected_h"), atol, d);
        }

        // ── 2. generate_gaussian_rejection ───────────────────────────────────
        {
            const auto& c = cases.at("generate_gaussian_rejection");
            std::string lbl = c.at("label").get<std::string>();
            int n = c.at("n").get<int>();
            double T = c.at("temperature").get<double>();
            int mc = c.at("max_candidates").get<int>();
            std::vector<double> z = flatten_mat(c.at("z_candidates"));
            auto got = cypha::generate_class_gaussian(
                m, lbl, n, T, nullptr, /*rejection=*/true, mc, z.data());
            all_pass &= check_case("generate_gaussian_rejection", got,
                                   c.at("expected_h"), atol, d);
        }

        // ── 3. generate_conditioned ───────────────────────────────────────────
        {
            const auto& c = cases.at("generate_conditioned");
            std::string lbl = c.at("label").get<std::string>();
            int n = c.at("n").get<int>();
            double T = c.at("temperature").get<double>();
            std::vector<double> z = flatten_mat(c.at("z"));
            auto got = cypha::generate_conditioned(m, lbl, n, T, nullptr, z.data());
            all_pass &= check_case("generate_conditioned", got,
                                   c.at("expected_h"), atol, d);
        }

        // ── 4. generate_langevin ──────────────────────────────────────────────
        {
            const auto& c = cases.at("generate_langevin");
            std::string lbl = c.at("label").get<std::string>();
            int n = c.at("n").get<int>();
            int n_steps = c.at("n_steps").get<int>();
            double step_size = c.at("step_size").get<double>();
            double T = c.at("temperature").get<double>();
            std::vector<double> z_init = flatten_mat(c.at("z_init"));
            std::vector<double> z_noise = flatten_3d(c.at("z_noise"));
            auto got = cypha::generate_langevin(
                m, lbl, n, n_steps, step_size, T,
                nullptr, z_init.data(), z_noise.data());
            all_pass &= check_case("generate_langevin", got,
                                   c.at("expected_h"), atol, d);
        }

        // ── 5. generate_boundary ──────────────────────────────────────────────
        {
            const auto& c = cases.at("generate_boundary");
            std::string la = c.at("label_a").get<std::string>();
            std::string lb = c.at("label_b").get<std::string>();
            int n = c.at("n").get<int>();
            double alpha = c.at("alpha").get<double>();
            double T = c.at("temperature").get<double>();
            std::vector<double> z = flatten_mat(c.at("z"));
            auto got = cypha::generate_boundary(
                m, la, lb, n, alpha, T, nullptr, z.data());
            all_pass &= check_case("generate_boundary", got,
                                   c.at("expected_h"), atol, d);
        }

        // ── 6. generate_ood ───────────────────────────────────────────────────
        {
            const auto& c = cases.at("generate_ood");
            int n = c.at("n").get<int>();
            int n_cands = c.at("n_candidates").get<int>();
            std::vector<double> z = flatten_mat(c.at("z_candidates"));
            auto got = cypha::generate_ood(m, n, n_cands, nullptr, z.data());
            all_pass &= check_case("generate_ood", got,
                                   c.at("expected_h"), atol, d);
        }

        // ── 7. generate_mdl_ball ──────────────────────────────────────────────
        {
            const auto& c = cases.at("generate_mdl_ball");
            std::string lbl = c.at("label").get<std::string>();
            int n = c.at("n").get<int>();
            double radius = c.at("radius").get<double>();
            std::vector<double> z_dir = flatten_mat(c.at("z_dir"));
            std::vector<double> u_mag = c.at("u_mag").get<std::vector<double>>();
            auto got = cypha::generate_mdl_ball(
                m, lbl, n, radius, nullptr, z_dir.data(), u_mag.data());
            all_pass &= check_case("generate_mdl_ball", got,
                                   c.at("expected_h"), atol, d);
        }

        // ── 8. generate_ancestral ─────────────────────────────────────────────
        {
            const auto& c = cases.at("generate_ancestral");
            int n = c.at("n").get<int>();
            double T = c.at("temperature").get<double>();
            std::vector<double> u_class = c.at("u_class").get<std::vector<double>>();
            std::vector<double> z = flatten_mat(c.at("z"));
            auto exp_labels = c.at("expected_labels").get<std::vector<std::string>>();
            auto exp_h = read_matrix(c.at("expected_h"));
            auto got = cypha::generate_ancestral(
                m, n, T, nullptr, u_class.data(), z.data());
            bool ok = true;
            if (static_cast<int>(got.size()) != n) {
                std::cerr << "FAIL generate_ancestral: n mismatch\n"; ok = false;
            } else {
                for (int i = 0; i < n; i++) {
                    if (got[static_cast<std::size_t>(i)].label !=
                        exp_labels[static_cast<std::size_t>(i)]) {
                        std::cerr << "FAIL generate_ancestral[" << i << "]: label "
                                  << got[static_cast<std::size_t>(i)].label
                                  << " vs " << exp_labels[static_cast<std::size_t>(i)] << "\n";
                        ok = false;
                    }
                    if (!near_all(got[static_cast<std::size_t>(i)].h,
                                  exp_h[static_cast<std::size_t>(i)], atol)) {
                        std::cerr << "FAIL generate_ancestral[" << i << "]: h mismatch\n";
                        ok = false;
                    }
                }
            }
            if (ok) std::cout << "PASS generate_ancestral (" << n << " samples)\n";
            all_pass &= ok;
        }

        // ── 9. predict_next ───────────────────────────────────────────────────
        {
            const auto& c = cases.at("predict_next");
            std::string last = c.at("last_label").get<std::string>();
            auto exp_probs = c.at("expected_probs").get<std::vector<double>>();
            // Set ctx_last_label on a copy
            cypha::CyphaInferModel m2 = m;
            m2.ctx_last_label = last;
            auto got = cypha::predict_next_probs(m2, last);
            bool ok = near_all(got, exp_probs, atol);
            if (ok) std::cout << "PASS predict_next (" << got.size() << " classes)\n";
            else {
                std::cerr << "FAIL predict_next: max_err=";
                double max_e = 0.0;
                for (std::size_t i = 0; i < got.size(); i++)
                    max_e = std::max(max_e, std::abs(got[i] - exp_probs[i]));
                std::cerr << max_e << " atol=" << atol << "\n";
            }
            all_pass &= ok;
        }

        // ── 10. rollout ───────────────────────────────────────────────────────
        {
            const auto& c = cases.at("rollout");
            std::string seed = c.at("seed_label").get<std::string>();
            int n_steps = c.at("n_steps").get<int>();
            double T = c.at("temperature").get<double>();
            double expl = c.at("exploration").get<double>();
            std::vector<double> z_gen = flatten_mat(c.at("z_generate"));
            std::vector<double> u_tr = c.at("u_transition").get<std::vector<double>>();
            auto exp_labels = c.at("expected_labels").get<std::vector<std::string>>();
            auto exp_h = read_matrix(c.at("expected_h"));

            cypha::CyphaInferModel m2 = m;  // fresh copy so context starts same
            auto got = cypha::rollout(m2, seed, n_steps, T, expl, nullptr,
                                      z_gen.data(), u_tr.data());
            bool ok = true;
            if (static_cast<int>(got.size()) != n_steps) {
                std::cerr << "FAIL rollout: n_steps mismatch\n"; ok = false;
            } else {
                for (int i = 0; i < n_steps; i++) {
                    std::size_t si = static_cast<std::size_t>(i);
                    if (got[si].label != exp_labels[si]) {
                        std::cerr << "FAIL rollout[" << i << "]: label "
                                  << got[si].label << " vs " << exp_labels[si] << "\n";
                        ok = false;
                    }
                    if (!near_all(got[si].h, exp_h[si], atol)) {
                        std::cerr << "FAIL rollout[" << i << "]: h mismatch max_err=";
                        double me = 0.0;
                        for (int j = 0; j < d; j++)
                            me = std::max(me, std::abs(
                                got[si].h[static_cast<std::size_t>(j)] -
                                exp_h[si][static_cast<std::size_t>(j)]));
                        std::cerr << me << "\n";
                        ok = false;
                    }
                }
            }
            if (ok) std::cout << "PASS rollout (" << n_steps << " steps)\n";
            all_pass &= ok;
        }

        // ── 11. generate_from_observation ─────────────────────────────────────
        {
            const auto& c = cases.at("generate_from_observation");
            std::string lbl = c.at("label").get<std::string>();
            int n = c.at("n").get<int>();
            double T = c.at("temperature").get<double>();
            int n_steps = c.at("n_steps").get<int>();
            std::vector<double> h_obs = c.at("h_obs").get<std::vector<double>>();
            std::vector<double> z_noise = flatten_3d(c.at("z_noise"));
            auto got = cypha::generate_from_observation(
                m, h_obs.data(), lbl, n, T, n_steps, nullptr, z_noise.data());
            all_pass &= check_case("generate_from_observation", got,
                                   c.at("expected_h"), atol, d);
        }

        // ── 12. generate_retrieval_augmented ──────────────────────────────────
        {
            const auto& c = cases.at("generate_retrieval_augmented");
            int n = c.at("n").get<int>();
            int k_neighbors = c.at("k_neighbors").get<int>();
            double T = c.at("temperature").get<double>();
            int n_steps = c.at("n_steps").get<int>();
            std::vector<double> query = c.at("query_x").get<std::vector<double>>();
            std::vector<double> z_noise = flatten_3d(c.at("z_noise"));
            std::vector<double> db_flat;
            for (const auto& row : c.at("database_x")) {
                auto rv = row.get<std::vector<double>>();
                db_flat.insert(db_flat.end(), rv.begin(), rv.end());
            }
            const int n_db = static_cast<int>(c.at("database_x").size());
            cypha::CyphaInferOptions opt;
            opt.use_field = true;
            auto got = cypha::generate_retrieval_augmented(
                m, query.data(), db_flat.data(), n_db, input_dim, k_neighbors,
                n, T, n_steps, opt, nullptr, z_noise.data());
            all_pass &= check_case("generate_retrieval_augmented", got,
                                   c.at("expected_h"), atol, d);
            if (c.contains("expected_label")) {
                std::string exp_lbl = c.at("expected_label").get<std::string>();
                if (!exp_lbl.empty()) {
                    std::cout << "  retrieval label=" << exp_lbl << "\n";
                }
            }
        }

        if (all_pass) {
            std::cout << "\nAll generation parity checks PASSED.\n";
            return 0;
        }
        std::cerr << "\nSome generation parity checks FAILED.\n";
        return 1;

    } catch (const std::exception& e) {
        std::cerr << "generation_parity: exception: " << e.what() << "\n";
        return 1;
    }
}
