// cyphalm_parity — meta-runner for native CyphaLM parity tools / fixtures.
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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
        if (fs::is_directory(cur / "fixtures")) return cur;
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

std::string quote_arg(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

int run_process(const fs::path& exe, const std::vector<std::string>& args) {
#ifdef _WIN32
    std::string cmd = quote_arg(exe.string());
    for (const auto& a : args) {
        cmd += ' ';
        cmd += quote_arg(a);
    }
    std::vector<char> buf(cmd.begin(), cmd.end());
    buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        return 127;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return static_cast<int>(code);
#else
    std::string cmd = quote_arg(exe.string());
    for (const auto& a : args) {
        cmd += ' ';
        cmd += quote_arg(a);
    }
    return std::system(cmd.c_str());
#endif
}

int run_tool(const fs::path& dir, const char* stem, const std::vector<std::string>& args = {}) {
    const fs::path exe = fs::path(sibling_exe(dir, stem));
    std::cerr << "cyphalm_parity: run " << quote_arg(exe.string());
    for (const auto& a : args) {
        std::cerr << ' ' << quote_arg(a);
    }
    std::cerr << "\n";
    return run_process(exe, args);
}

std::vector<fs::path> discover_sidecars(int argc, char** argv, const fs::path& repo_root) {
    std::vector<fs::path> out;
    if (argc >= 2) {
        fs::path p(argv[1]);
        if (p.is_relative()) p = repo_root / p;
        out.push_back(fs::absolute(p));
        return out;
    }
    const fs::path root = repo_root / "fixtures";
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
