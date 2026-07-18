#include "cypha/bench/bench_figures.hpp"

#include "cypha/bench/bench_figure_render.hpp"
#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_report_json.hpp"

#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace cypha::bench {

namespace fs = std::filesystem;

fs::path figures_dir() { return bench_root() / "report" / "figures"; }

namespace {

void write_json_file(const fs::path& path, const ProfileJson& payload) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << payload.dump(2);
}

void write_figure_png(const ProfileJson& fig, std::vector<fs::path>& written) {
    const std::string name = fig.value("figure", "");
    if (name.empty()) return;
    const fs::path png_path = figures_dir() / (name + ".png");
    if (const auto rendered = render_figure_png(fig, png_path)) {
        written.push_back(*rendered);
    }
}

void write_csv_file(const fs::path& path, const std::vector<std::string>& header,
                    const std::vector<std::vector<std::string>>& rows) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (i > 0) out << ',';
        out << header[i];
    }
    out << '\n';
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            if (i > 0) out << ',';
            out << row[i];
        }
        out << '\n';
    }
}

ProfileJson grouped_bar_figure(const std::string& name, const std::string& title, const std::string& y_label,
                               const std::vector<std::string>& categories,
                               const std::vector<std::pair<std::string, std::vector<double>>>& series) {
    ProfileJson payload = ProfileJson{
        {"figure", name},
        {"format", "figure_data_v1"},
        {"chart_type", "grouped_bar"},
        {"title", title},
        {"y_label", y_label},
        {"categories", categories},
        {"series", ProfileJson::array()},
    };
    for (const auto& s : series) {
        payload["series"].push_back(ProfileJson{{"name", s.first}, {"values", s.second}});
    }
    return payload;
}

std::optional<ProfileJson> table_for_domain(const std::unordered_map<std::string, ProfileJson>& tables,
                                            const std::string& domain_id) {
    const auto it = tables.find(domain_id);
    if (it == tables.end()) return std::nullopt;
    return it->second;
}

double json_number(const ProfileJson& j, double fallback = 0.0) {
    if (j.is_null() || !j.is_number()) return fallback;
    return j.get<double>();
}

double nested_number(const ProfileJson& root, std::initializer_list<const char*> keys, double fallback = 0.0) {
    ProfileJson cur = root;
    for (const char* key : keys) {
        if (!cur.is_object() || !cur.contains(key)) return fallback;
        cur = cur[key];
    }
    return json_number(cur, fallback);
}

ProfileJson experiment_root(const ProfileJson& domain_table) {
    const ProfileJson& exp = domain_table.value("experiments", ProfileJson::object());
    if (exp.contains("summary") && exp["summary"].is_object()) return exp["summary"];
    return exp;
}

void emit_figure(const ProfileJson& fig, ProfileJson& manifest, std::vector<fs::path>& written) {
    const std::string name = fig.value("figure", "");
    if (name.empty()) return;
    const fs::path path = figures_dir() / (name + ".json");
    write_json_file(path, fig);
    written.push_back(path);
    write_figure_png(fig, written);
    manifest["figures"].push_back(fig["figure"]);
}

}  // namespace

std::vector<fs::path> generate_figure_data() {
    const auto tables = load_all_domain_tables();
    std::vector<fs::path> written;
    ProfileJson manifest = ProfileJson{
        {"format", "figures_manifest_v2"},
        {"render", "png"},
        {"figures", ProfileJson::array()},
    };

    if (const auto d01 = table_for_domain(tables, "d01")) {
        const ProfileJson& exp = d01->value("experiments", ProfileJson::object());
        if (exp.contains("tasks") && exp["tasks"].is_array()) {
            std::vector<std::string> categories;
            std::vector<double> cypha_acc;
            std::vector<double> lr_acc;
            for (const auto& task : exp["tasks"]) {
                if (!task.is_object()) continue;
                categories.push_back(task.value("task", task.value("dataset", "task")));
                cypha_acc.push_back(nested_number(task, {"cypha_scores", "accuracy"},
                                                  nested_number(task, {"accuracy"})));
                lr_acc.push_back(nested_number(task, {"baselines", "logistic_regression", "accuracy"}));
            }
            if (!categories.empty()) {
                const ProfileJson fig = grouped_bar_figure(
                    "fig01_sanity_overview", "Classification accuracy (D01 subset)", "accuracy", categories,
                    {{"Cypha", cypha_acc}, {"Logistic Regression", lr_acc}});
                const fs::path path = figures_dir() / "fig01_sanity_overview.json";
                write_json_file(path, fig);
                written.push_back(path);
                write_figure_png(fig, written);

                std::vector<std::vector<std::string>> rows;
                for (std::size_t i = 0; i < categories.size(); ++i) {
                    rows.push_back({categories[i], std::to_string(cypha_acc[i]), std::to_string(lr_acc[i])});
                }
                const fs::path csv = figures_dir() / "fig01_sanity_overview.csv";
                write_csv_file(csv, {"task", "cypha_accuracy", "logistic_regression_accuracy"}, rows);
                written.push_back(csv);
                manifest["figures"].push_back(fig["figure"]);
            }
        }
    }

    if (const auto d02 = table_for_domain(tables, "d02")) {
        const ProfileJson& exp = d02->value("experiments", ProfileJson::object());
        const std::string dataset = exp.value("dataset", "dataset");
        const double cypha_rmse = nested_number(exp, {"cypha_scores", "rmse"});
        const double ridge_rmse = nested_number(exp, {"baselines", "ridge", "rmse"}, nested_number(exp, {"ridge_rmse"}));
        const ProfileJson fig = grouped_bar_figure(
            "fig02_regression_rmse_comparison", "Regression RMSE comparison", "rmse", {dataset},
            {{"Cypha", {cypha_rmse}}, {"Ridge", {ridge_rmse}}});
        const fs::path path = figures_dir() / "fig02_regression_rmse_comparison.json";
        write_json_file(path, fig);
        written.push_back(path);
        write_figure_png(fig, written);
        manifest["figures"].push_back(fig["figure"]);
    }

    if (const auto d03 = table_for_domain(tables, "d03")) {
        const ProfileJson& exp = d03->value("experiments", ProfileJson::object());
        if (exp.contains("datasets") && exp["datasets"].is_array()) {
            std::vector<std::string> categories;
            std::vector<double> cypha_acc;
            std::vector<double> lr_acc;
            for (const auto& ds : exp["datasets"]) {
                if (!ds.is_object()) continue;
                categories.push_back(ds.value("dataset", "dataset"));
                cypha_acc.push_back(nested_number(ds, {"cypha_scores", "accuracy"}));
                lr_acc.push_back(nested_number(ds, {"baselines", "logistic_regression", "accuracy"}));
            }
            if (!categories.empty()) {
                const ProfileJson fig = grouped_bar_figure(
                    "fig03_classification_accuracy", "Classification accuracy by dataset", "accuracy", categories,
                    {{"Cypha", cypha_acc}, {"Logistic Regression", lr_acc}});
                const fs::path path = figures_dir() / "fig03_classification_accuracy.json";
                write_json_file(path, fig);
                written.push_back(path);
                write_figure_png(fig, written);
                manifest["figures"].push_back(fig["figure"]);
            }
        }
    }

    if (const auto d05 = table_for_domain(tables, "d05")) {
        const ProfileJson& exp = d05->value("experiments", ProfileJson::object());
        const std::string source = exp.value("data_source", "chess");
        const double cypha_rmse = nested_number(exp, {"cypha_scores", "rmse"});
        const double ridge_rmse =
            nested_number(exp, {"baselines", "ridge", "rmse"}, nested_number(exp, {"cypha_scores", "ridge_rmse"}));
        ProfileJson fig = ProfileJson{
            {"figure", "fig05_chess_rmse_vs_baselines"},
            {"format", "figure_data_v1"},
            {"chart_type", "bar"},
            {"title", "Chess outcome regression (" + source + ")"},
            {"y_label", "rmse"},
            {"categories", ProfileJson::array({"Cypha", "Ridge"})},
            {"series", ProfileJson::array({ProfileJson{{"name", "RMSE"}, {"values", ProfileJson::array({cypha_rmse, ridge_rmse})}}})},
        };
        const fs::path path = figures_dir() / "fig05_chess_rmse_vs_baselines.json";
        write_json_file(path, fig);
        written.push_back(path);
        write_figure_png(fig, written);
        manifest["figures"].push_back(fig["figure"]);
    }

    if (const auto d04 = table_for_domain(tables, "d04")) {
        const ProfileJson exp = experiment_root(*d04);
        const ProfileJson char_lm = exp.contains("char_lm") && exp["char_lm"].is_object()
                                        ? exp["char_lm"]
                                        : exp;
        const std::string source = char_lm.value("corpus_source", exp.value("corpus", "corpus"));
        const double cypha_bpc = nested_number(char_lm, {"final_bpc"}, nested_number(exp, {"bpc"}));
        const double bigram_bpc = nested_number(char_lm, {"bigram_bpc"});
        const double trigram_bpc = nested_number(char_lm, {"trigram_bpc"});

        std::vector<std::string> categories = {"Cypha (sequence)"};
        std::vector<double> bpc_values = {cypha_bpc};
        if (bigram_bpc > 0.0) {
            categories.push_back("Bigram");
            bpc_values.push_back(bigram_bpc);
        }
        if (trigram_bpc > 0.0) {
            categories.push_back("Trigram");
            bpc_values.push_back(trigram_bpc);
        }

        ProfileJson fig = ProfileJson{
            {"figure", "fig04_char_lm_training"},
            {"format", "figure_data_v1"},
            {"chart_type", "bar"},
            {"title", "Cypha char LM (" + source + ")"},
            {"y_label", "bits per character"},
            {"categories", categories},
            {"series", ProfileJson::array({ProfileJson{{"name", "BPC"}, {"values", bpc_values}}})},
        };
        emit_figure(fig, manifest, written);
    }

    if (const auto d06 = table_for_domain(tables, "d06")) {
        const ProfileJson exp = experiment_root(*d06);
        const ProfileJson& reg = exp.value("regression", ProfileJson::object());
        const ProfileJson& cls = exp.value("classification", ProfileJson::object());
        const double cypha_rmse = nested_number(reg, {"cypha_scores", "rmse"});
        const double sgd_rmse = nested_number(reg, {"sgd_online", "rmse"});
        const double rf_rmse = nested_number(reg, {"baselines", "random_forest", "rmse"},
                                             nested_number(reg, {"baselines", "ridge", "rmse"},
                                                           nested_number(reg, {"ridge_rmse"})));

        const double cypha_acc = nested_number(cls, {"cypha_scores", "accuracy"});
        const double sgd_acc = nested_number(cls, {"sgd_online", "accuracy"});
        const double logreg_acc = nested_number(cls, {"baselines", "logistic_regression", "accuracy"});

        const ProfileJson fig = grouped_bar_figure(
            "fig06_go_territory_regression", "Go territory regression and outcome classification", "score",
            {"Territory RMSE", "Outcome accuracy"},
            {{"Cypha", {cypha_rmse, cypha_acc}},
             {"SGD", {sgd_rmse, sgd_acc}},
             {"RF/LogReg", {rf_rmse, logreg_acc}}});
        emit_figure(fig, manifest, written);
    }

    if (const auto d07 = table_for_domain(tables, "d07")) {
        const ProfileJson exp = experiment_root(*d07);
        const double cypha_acc = nested_number(exp, {"cypha_scores", "accuracy"});
        const double sgd_acc = nested_number(exp, {"sgd_online", "accuracy"});
        const double rf_acc = nested_number(exp, {"baselines", "random_forest", "accuracy"});

        ProfileJson fig = ProfileJson{
            {"figure", "fig07_poker_decision_accuracy"},
            {"format", "figure_data_v1"},
            {"chart_type", "bar"},
            {"title", "Poker decision accuracy"},
            {"y_label", "accuracy"},
            {"categories", ProfileJson::array({"Cypha", "SGD", "RF"})},
            {"series", ProfileJson::array({ProfileJson{
                              {"name", "accuracy"},
                              {"values", ProfileJson::array({cypha_acc, sgd_acc, rf_acc})}}})},
        };
        emit_figure(fig, manifest, written);
    }

    if (const auto d08 = table_for_domain(tables, "d08")) {
        const ProfileJson exp = experiment_root(*d08);
        const std::string source = exp.value("data_source", "vision");
        std::vector<std::string> categories;
        std::vector<double> cypha_acc;
        std::vector<double> baseline_acc;

        auto ingest_encoding = [&](const ProfileJson& run, const std::string& fallback_label) {
            if (!run.is_object()) return;
            const std::string label = run.value("encoding", run.value("dataset", fallback_label));
            categories.push_back(label);
            cypha_acc.push_back(nested_number(run, {"cypha_scores", "accuracy"}));
            baseline_acc.push_back(nested_number(run, {"baselines", "logistic_regression", "accuracy"},
                                                 nested_number(run, {"sgd_online", "accuracy"})));
        };

        if (exp.contains("experiments") && exp["experiments"].is_array()) {
            for (const auto& run : exp["experiments"]) ingest_encoding(run, "encoding");
        } else {
            if (exp.contains("raw")) ingest_encoding(exp["raw"], "raw");
            if (exp.contains("hog")) ingest_encoding(exp["hog"], "hog");
        }

        if (!categories.empty()) {
            const ProfileJson fig = grouped_bar_figure(
                "fig08_mnist_accuracy_by_encoding", "Vision classification (" + source + ")", "accuracy",
                categories, {{"Cypha", cypha_acc}, {"LogReg/SGD", baseline_acc}});
            emit_figure(fig, manifest, written);
        }
    }

    if (const auto d09 = table_for_domain(tables, "d09")) {
        const ProfileJson exp = experiment_root(*d09);
        const ProfileJson& news = exp.value("20news", ProfileJson::object());
        const ProfileJson& ood = exp.value("gutenberg_ood", ProfileJson::object());

        const double cypha_acc = nested_number(news, {"cypha_scores", "accuracy"});
        const double sgd_acc = nested_number(news, {"sgd_online", "accuracy"});
        const double logreg_acc = nested_number(news, {"baselines", "logistic_regression", "accuracy"});
        const double ep_in = nested_number(ood, {"mean_epistemic_in"});
        const double ep_ood = nested_number(ood, {"mean_epistemic_ood"});

        ProfileJson fig = ProfileJson{
            {"figure", "fig09_documents_overview"},
            {"format", "figure_data_v1"},
            {"chart_type", "bar"},
            {"title", "Documents overview"},
            {"y_label", "score"},
            {"categories", ProfileJson::array({"20news Cypha", "20news SGD", "20news LogReg", "Epistemic in",
                                                "Epistemic OOD"})},
            {"series", ProfileJson::array({ProfileJson{
                              {"name", "score"},
                              {"values", ProfileJson::array({cypha_acc, sgd_acc, logreg_acc, ep_in, ep_ood})}}})},
        };
        emit_figure(fig, manifest, written);
    }

    const fs::path manifest_path = figures_dir() / "figures_manifest.json";
    write_json_file(manifest_path, manifest);
    written.push_back(manifest_path);
    return written;
}

}  // namespace cypha::bench
