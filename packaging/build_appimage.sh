#!/usr/bin/env bash
# Build cypha_qt_shell + cypha_rest (Release) and package a self-contained AppImage.
#
# Prerequisites (Ubuntu/Debian example):
#   sudo apt-get install -y cmake ninja-build g++ libsqlite3-dev \
#     qt6-base-dev libxkbcommon-x11-0 libxcb-cursor0 libegl1 libgl1-mesa-dri libglib2.0-0
#
# linuxdeploy + Qt plugin (one-time; versions pinned for reproducibility):
#   export LINUXDEPLOY_URL="https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
#   export LINUXDEPLOY_QT_URL="https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-x86_64.AppImage"
#   wget -O /tmp/linuxdeploy-x86_64.AppImage "$LINUXDEPLOY_URL"
#   wget -O /tmp/linuxdeploy-plugin-qt-x86_64.AppImage "$LINUXDEPLOY_QT_URL"
#   chmod +x /tmp/linuxdeploy-*.AppImage
#   export PATH="/tmp:$PATH"
#   export LINUXDEPLOY="/tmp/linuxdeploy-x86_64.AppImage"
#   export LINUXDEPLOY_QT="/tmp/linuxdeploy-plugin-qt-x86_64.AppImage"
#
# Usage:
#   bash packaging/build_appimage.sh [VERSION] [BUILD_DIR] [OUT_DIR]
#
# Examples:
#   bash packaging/build_appimage.sh 2.2.8
#   bash packaging/build_appimage.sh 2.2.8 native/build-appimage dist
#
# Output:
#   dist/cypha-<VERSION>-linux-x86_64.AppImage
#
# The AppImage bundles Qt shared libraries via linuxdeploy-plugin-qt. cypha_rest is
# included as a sidecar in usr/bin/ (spawned by the shell or run standalone).

set -euo pipefail

VERSION="${1:-dev}"
BUILD_DIR="${2:-native/build-appimage}"
OUT_DIR="${3:-dist}"
SKIP_BUILD="${SKIP_BUILD:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR_ABS="$REPO_ROOT/$BUILD_DIR"
OUT_DIR_ABS="$REPO_ROOT/$OUT_DIR"
APPDIR="$OUT_DIR_ABS/Cypha.AppDir"
APPIMAGE="$OUT_DIR_ABS/cypha-${VERSION}-linux-x86_64.AppImage"

LINUXDEPLOY="${LINUXDEPLOY:-linuxdeploy-x86_64.AppImage}"
LINUXDEPLOY_QT="${LINUXDEPLOY_QT:-linuxdeploy-plugin-qt-x86_64.AppImage}"

find_built_exe() {
  local name="$1"
  local candidate
  for candidate in \
    "$BUILD_DIR_ABS/$name" \
    "$BUILD_DIR_ABS/qt/$name"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

require_linuxdeploy() {
  if [[ -x "$LINUXDEPLOY" ]]; then
    return 0
  fi
  if command -v "$LINUXDEPLOY" >/dev/null 2>&1; then
    LINUXDEPLOY="$(command -v "$LINUXDEPLOY")"
    return 0
  fi
  cat >&2 <<'EOF'
error: linuxdeploy not found.

Install the continuous AppImages (see script header) and export:
  export LINUXDEPLOY=/path/to/linuxdeploy-x86_64.AppImage
  export LINUXDEPLOY_QT=/path/to/linuxdeploy-plugin-qt-x86_64.AppImage
  export PATH="$(dirname "$LINUXDEPLOY"):$PATH"
  export APPIMAGE_EXTRACT_AND_RUN=1

Docs: https://docs.appimage.org/packaging-guide/from-source/native-binaries.html
EOF
  return 1
}

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "==> Configure Release build (CYPHA_BUILD_QT=ON)"
  cmake -S "$REPO_ROOT/native" -B "$BUILD_DIR_ABS" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCYPHA_BUILD_QT=ON \
    -G Ninja

  echo "==> Build cypha_qt_shell + cypha_rest"
  cmake --build "$BUILD_DIR_ABS" --parallel --target cypha_qt_shell cypha_rest
else
  echo "==> SKIP_BUILD=1 — packaging from existing $BUILD_DIR_ABS"
fi

SHELL_EXE="$(find_built_exe cypha_qt_shell)"
REST_EXE="$(find_built_exe cypha_rest)"

echo "  cypha_qt_shell: $SHELL_EXE"
echo "  cypha_rest:     $REST_EXE"

echo "==> Stage AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"

install -m 755 "$SHELL_EXE" "$APPDIR/usr/bin/cypha_qt_shell"
install -m 755 "$REST_EXE" "$APPDIR/usr/bin/cypha_rest"

cat >"$APPDIR/cypha_qt_shell.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Cypha Studio
Comment=Cypha native Qt shell (train, infer, registry)
Exec=cypha_qt_shell
Icon=cypha
Categories=Development;Science;
Terminal=false
EOF

# Icon is required by linuxdeploy when desktop Icon=cypha is set.
if [[ -f "$REPO_ROOT/packaging/cypha.png" ]]; then
  install -m 644 "$REPO_ROOT/packaging/cypha.png" "$APPDIR/cypha.png"
else
  echo "error: missing packaging/cypha.png (required for AppImage Icon=cypha)" >&2
  exit 1
fi
# Also install under hicolor so Icon lookup finds it after desktop deploy.
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
install -m 644 "$APPDIR/cypha.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/cypha.png"

cat >"$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
exec "$HERE/usr/bin/cypha_qt_shell" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# Demo fixtures (optional sidecar for smoke / first launch)
if [[ -d "$REPO_ROOT/fixtures" ]]; then
  mkdir -p "$APPDIR/usr/share/cypha/demo_fixtures"
  for f in reference.cypha f_field.json train_hparams.json; do
    if [[ -f "$REPO_ROOT/fixtures/$f" ]]; then
      cp "$REPO_ROOT/fixtures/$f" "$APPDIR/usr/share/cypha/demo_fixtures/"
    fi
  done
fi

echo "$VERSION" >"$APPDIR/usr/share/cypha/VERSION"

require_linuxdeploy

export ARCH="${ARCH:-x86_64}"
export APPIMAGE_EXTRACT_AND_RUN=1
export PATH="$(dirname "$(readlink -f "$LINUXDEPLOY" 2>/dev/null || echo "$LINUXDEPLOY")"):$PATH"

PLUGIN_ARGS=()
if [[ -n "${LINUXDEPLOY_QT:-}" ]]; then
  if [[ -x "$LINUXDEPLOY_QT" ]] || command -v "$LINUXDEPLOY_QT" >/dev/null 2>&1; then
    PLUGIN_ARGS=(--plugin "qt")
    if [[ -x "$LINUXDEPLOY_QT" ]]; then
      export LINUXDEPLOY_PLUGIN_QT="$LINUXDEPLOY_QT"
    fi
  else
    echo "warning: LINUXDEPLOY_QT not found — continuing without --plugin qt (AppImage may miss Qt libs)" >&2
  fi
fi

ICON_ARG=()
if [[ -f "$APPDIR/cypha.png" ]]; then
  ICON_ARG=(--icon-file "$APPDIR/cypha.png")
fi

echo "==> linuxdeploy (bundle Qt .so + create AppImage)"
# linuxdeploy steps:
#   1. Copy cypha_qt_shell into AppDir/usr/bin (done above)
#   2. --executable triggers dependency scan; --plugin qt pulls Qt platform/plugins
#   3. --output appimage writes a single relocatable file
(
  cd "$OUT_DIR_ABS"
  "$LINUXDEPLOY" \
    --appdir "$APPDIR" \
    --executable "$APPDIR/usr/bin/cypha_qt_shell" \
    --desktop-file "$APPDIR/cypha_qt_shell.desktop" \
    "${ICON_ARG[@]}" \
    "${PLUGIN_ARGS[@]}" \
    --output appimage
)

# linuxdeploy names output after the desktop file by default.
BUILT_APPIMAGE="$(find "$OUT_DIR_ABS" -maxdepth 1 -name '*.AppImage' -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
if [[ -z "${BUILT_APPIMAGE:-}" ]]; then
  BUILT_APPIMAGE="$(find "$OUT_DIR_ABS" -maxdepth 1 -name 'cypha_qt_shell*.AppImage' -type f | head -1)"
fi
if [[ -z "${BUILT_APPIMAGE:-}" ]]; then
  BUILT_APPIMAGE="$(find "$OUT_DIR_ABS" -maxdepth 1 -name '*.AppImage' -type f | head -1)"
fi
if [[ -z "${BUILT_APPIMAGE:-}" || ! -f "$BUILT_APPIMAGE" ]]; then
  echo "error: linuxdeploy did not produce an AppImage under $OUT_DIR_ABS" >&2
  exit 1
fi

mv -f "$BUILT_APPIMAGE" "$APPIMAGE"
chmod +x "$APPIMAGE"

echo ""
echo "Created $APPIMAGE ($(du -h "$APPIMAGE" | cut -f1))"
echo ""
echo "Run:"
echo "  ./$APPIMAGE"
echo "  ./$APPIMAGE --smoke usr/share/cypha/demo_fixtures/reference.cypha"
echo ""
echo "Sidecar REST server (bundled):"
echo "  ./$APPIMAGE --appimage-extract-and-run  # if inspecting extracted tree"
echo "  # or extract: ./$APPIMAGE --appimage-extract"
echo "  # then: Cypha.AppDir/usr/bin/cypha_rest --listen 127.0.0.1:8765 --cypha ..."
