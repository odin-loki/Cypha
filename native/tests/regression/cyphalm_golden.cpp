// cyphalm_golden — meta-runner for native CyphaLM parity tools (hp path).
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
    std::cerr << "cyphalm_golden: run " << quote_arg(exe.string());
    for (const auto& a : args) {
        std::cerr << ' ' << quote_arg(a);
    }
    std::cerr << "\n";
    return run_process(exe, args);
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path tool_dir = exe_dir(argc, argv);
    int failures = 0;

    failures += run_tool(tool_dir, "cyphalm_model_golden") != 0 ? 1 : 0;
    failures += run_tool(tool_dir, "hp_roundtrip_smoke") != 0 ? 1 : 0;

    if (failures == 0) {
        std::cout << "All CyphaLM native parity checks PASSED.\n";
        return 0;
    }
    std::cerr << "cyphalm_golden: " << failures << " tool(s) failed\n";
    return 1;
}
