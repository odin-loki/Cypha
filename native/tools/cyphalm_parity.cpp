// cyphalm_parity — meta-runner for native CyphaLM parity tools / fixtures.
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path exe_dir(int argc, char** argv) {
    if (argc >= 1 && argv[0] != nullptr) {
        std::error_code ec;
        const fs::path p = fs::absolute(fs::path(argv[0]), ec);
        if (!ec) return p.parent_path();
    }
    return fs::current_path();
}

fs::path find_repo_root(const fs::path& start) {
    fs::path cur = start;
    for (int i = 0; i < 8 && !cur.empty(); ++i) {
        if (fs::is_directory(cur / "parity_fixtures")) return cur;
        cur = cur.parent_path();
    }
    return start;
}

std::string sibling_exe(const fs::path& dir, const char* stem) {
#if defined(_WIN32)
    const std::string name = std::string(stem) + ".exe";
#else
    const std::string name = stem;
#endif
    const fs::path candidate = dir / name;
    if (fs::is_regular_file(candidate)) return candidate.string();
    return name;
}

int run_cmd(const std::string& cmd) {
    std::cerr << "cyphalm_parity: run " << cmd << "\n";
    return std::system(cmd.c_str());
}

int run_tool(const fs::path& dir, const char* stem, const std::vector<std::string>& args = {}) {
    std::string cmd = "\"" + sibling_exe(dir, stem) + "\"";
    for (const auto& a : args) cmd += " \"" + a + "\"";
    return run_cmd(cmd);
}

std::vector<fs::path> discover_sidecars(int argc, char** argv, const fs::path& repo_root) {
    std::vector<fs::path> out;
    if (argc >= 2) {
        fs::path p(argv[1]);
        if (p.is_relative()) p = repo_root / p;
        out.push_back(fs::absolute(p));
        return out;
    }
    const fs::path root = repo_root / "parity_fixtures";
    if (!fs::is_directory(root)) return out;
    for (const auto& ent : fs::directory_iterator(root)) {
        if (!ent.is_directory()) continue;
        const fs::path side = ent.path() / "sidecar.json";
        if (fs::is_regular_file(side) && ent.path().filename().string().rfind("cyphalm_", 0) == 0) {
            out.push_back(fs::absolute(side));
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path tool_dir = exe_dir(argc, argv);
    const fs::path repo_root = find_repo_root(tool_dir);
    int failures = 0;

    failures += run_tool(tool_dir, "cyphalm_ssm_parity") != 0 ? 1 : 0;
    failures += run_tool(tool_dir, "cyphalm_model_parity") != 0 ? 1 : 0;
    failures += run_tool(tool_dir, "cyphalm_hebbian_parity") != 0 ? 1 : 0;

    const auto sidecars = discover_sidecars(argc, argv, repo_root);
    for (const auto& side : sidecars) {
        const std::string dir_name = side.parent_path().filename().string();
        if (dir_name.find("char_lstm") != std::string::npos) {
            failures += run_tool(tool_dir, "cyphalm_char_lstm_parity", {side.string()}) != 0 ? 1 : 0;
        }
    }

    if (failures == 0) {
        std::cout << "All CyphaLM native parity checks PASSED.\n";
        return 0;
    }
    std::cerr << "cyphalm_parity: " << failures << " tool(s) failed\n";
    return 1;
}
