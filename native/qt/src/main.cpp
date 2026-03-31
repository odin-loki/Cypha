// Headless Qt Core entry — verifies Qt + cypha_core link (including mke_scalar_train_step TU). Replace with Widgets shell per qt/README.md.
#include <QByteArray>
#include <QCoreApplication>
#include <QFile>
#include <QStringList>
#include <QtGlobal>

#include <cstdio>

#include "cypha/load_cypha.hpp"

extern "C" int cypha_core_mke_scalar_train_step_link_touch();

int main(int argc, char** argv) {
  QCoreApplication app(argc, argv);
  const QStringList args = QCoreApplication::arguments();

  QString cypha_path;
  for (int i = 1; i < args.size(); ++i) {
    const QString& a = args.at(i);
    if (a == QStringLiteral("--help") || a == QStringLiteral("-h")) {
      std::printf(
          "cypha_qt_stub [--help] [.cypha]\n"
          "  Optional path: read file into memory and parse with load_cypha_from_buffer\n"
          "  (same v3 layout as Python cypha_load_binary_from_bytes).\n");
      return 0;
    }
    if (!a.startsWith(QLatin1Char('-'))) {
      cypha_path = a;
      break;
    }
  }

  if (cypha_core_mke_scalar_train_step_link_touch() != 1) {
    return 2;
  }

  if (!cypha_path.isEmpty()) {
    QFile f(cypha_path);
    if (!f.open(QIODevice::ReadOnly)) {
      std::fprintf(stderr, "cypha_qt_stub: cannot open %s\n", cypha_path.toUtf8().constData());
      return 3;
    }
    const QByteArray blob = f.readAll();
    cypha::CNode root = cypha::load_cypha_from_buffer(
        reinterpret_cast<const std::uint8_t*>(blob.constData()), static_cast<std::size_t>(blob.size()));
    if (root.kind != cypha::CNode::Map) {
      std::fprintf(stderr, "cypha_qt_stub: root is not a map\n");
      return 4;
    }
    if (!cypha::map_get(root, "world") || !cypha::map_get(root, "classes")) {
      std::fprintf(stderr, "cypha_qt_stub: missing world/classes keys\n");
      return 5;
    }
    std::printf("cypha_qt_stub OK (Qt %s); load_cypha_from_buffer %zu bytes\n", QT_VERSION_STR,
                static_cast<std::size_t>(blob.size()));
    return 0;
  }

  std::printf("cypha_qt_stub OK (Qt %s)\n", QT_VERSION_STR);
  return 0;
}
