#include "cypha_rest_static_ui.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef CYPHA_EMBED_STATIC_UI
#include "cypha_rest_static_embed.hpp"
#endif

namespace fs = std::filesystem;

namespace cypha_rest_ui {
namespace {

std::string resolve_static_dir(const char* argv0) {
  if (const char* env = std::getenv("CYPHA_REST_STATIC_DIR")) {
    if (*env != '\0') {
      return env;
    }
  }
  std::vector<fs::path> candidates;
  if (argv0 != nullptr && *argv0 != '\0') {
    try {
      const fs::path exe = fs::absolute(fs::path(argv0));
      candidates.push_back(exe.parent_path() / "static");
      candidates.push_back(exe.parent_path() / "tools" / "static");
      candidates.push_back(exe.parent_path().parent_path() / "tools" / "static");
    } catch (...) {
    }
  }
  candidates.push_back(fs::current_path() / "native" / "tools" / "static");
  candidates.push_back(fs::current_path() / "tools" / "static");
  for (const auto& p : candidates) {
    std::error_code ec;
    if (fs::is_directory(p, ec) && fs::exists(p / "index.html", ec)) {
      return p.generic_string();
    }
  }
  return {};
}

}  // namespace

void configure_static_ui(httplib::Server& svr, const char* argv0) {
#ifdef CYPHA_EMBED_STATIC_UI
  (void)argv0;
  const auto& files = embedded_files();
  svr.Get("/", [&files](const httplib::Request&, httplib::Response& res) {
    const auto it = files.find("/ui/index.html");
    if (it == files.end()) {
      res.status = 503;
      res.set_content("embedded UI missing index.html", "text/plain");
      return;
    }
    res.set_content(it->second.first, it->second.second);
  });
  for (const auto& kv : files) {
    const std::string path = kv.first;
    if (path == "/ui/index.html") {
      continue;
    }
    const char* content = kv.second.first;
    const char* mime = kv.second.second;
    svr.Get(path, [content, mime](const httplib::Request&, httplib::Response& res) {
      res.set_content(content, mime);
    });
  }
  std::cout << "Studio Web UI: embedded (CYPHA_EMBED_STATIC_UI)\n";
  return;
#endif

  const std::string static_dir = resolve_static_dir(argv0);
  if (static_dir.empty()) {
    std::cerr << "warning: Studio Web UI static/ not found (set CYPHA_REST_STATIC_DIR)\n";
    return;
  }

  if (!svr.set_mount_point("/ui", static_dir)) {
    std::cerr << "warning: failed to mount /ui from " << static_dir << "\n";
    return;
  }

  svr.Get("/", [static_dir](const httplib::Request&, httplib::Response& res) {
    const fs::path index_path = fs::path(static_dir) / "index.html";
    std::ifstream f(index_path, std::ios::binary);
    if (!f) {
      res.status = 503;
      res.set_content("index.html missing", "text/plain");
      return;
    }
    std::stringstream buf;
    buf << f.rdbuf();
    res.set_content(buf.str(), "text/html; charset=utf-8");
  });

  std::cout << "Studio Web UI: serving / and /ui/* from " << static_dir << "\n";
}

}  // namespace cypha_rest_ui
