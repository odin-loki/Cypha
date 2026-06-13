#!/usr/bin/env bash
# Build cypha_qt_shell + cypha_rest (Release) and package a macOS .app bundle via macdeployqt.
#
# Prerequisites (Homebrew example):
#   brew install cmake ninja qt@6 sqlite3
#   export PATH="$(brew --prefix qt@6)/bin:$PATH"
#
# Usage:
#   bash packaging/macos_bundle.sh [VERSION] [BUILD_DIR] [OUT_DIR]
#
# Examples:
#   bash packaging/macos_bundle.sh 2.3.1
#   bash packaging/macos_bundle.sh 2.3.1 native/build-qt-release dist
#   SKIP_BUILD=1 bash packaging/macos_bundle.sh 2.3.1 native/build-release dist
#
# Output:
#   dist/cypha-<VERSION>-macos/Cypha.app
#   dist/cypha-<VERSION>-macos/cypha_rest          (sidecar binary)
#   dist/cypha-<VERSION>-macos/share/demo_fixtures/ (when fixtures/ exists)

set -euo pipefail

VERSION="${1:-dev}"
BUILD_DIR="${2:-native/build-qt-release}"
OUT_DIR="${3:-dist/cypha-${VERSION}-macos}"
SKIP_BUILD="${SKIP_BUILD:-0}"
WITH_FIXTURES="${WITH_FIXTURES:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR_ABS="$REPO_ROOT/$BUILD_DIR"
OUT_DIR_ABS="$REPO_ROOT/$OUT_DIR"
APP_NAME="Cypha.app"
APP_DIR="$OUT_DIR_ABS/$APP_NAME"
MACDEPLOYQT="${MACDEPLOYQT:-macdeployqt}"

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

require_macdeployqt() {
  if [[ -x "$MACDEPLOYQT" ]]; then
    return 0
  fi
  if command -v "$MACDEPLOYQT" >/dev/null 2>&1; then
    MACDEPLOYQT="$(command -v "$MACDEPLOYQT")"
    return 0
  fi
  cat >&2 <<'EOF'
error: macdeployqt not found.

Install Qt 6 (Homebrew: brew install qt@6) and ensure macdeployqt is on PATH:
  export PATH="$(brew --prefix qt@6)/bin:$PATH"

Or pass MACDEPLOYQT=/path/to/macdeployqt
EOF
  return 1
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: macos_bundle.sh must run on macOS (macdeployqt is unavailable elsewhere)" >&2
  exit 1
fi

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

require_macdeployqt

echo "==> Stage .app skeleton"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

install -m 755 "$SHELL_EXE" "$APP_DIR/Contents/MacOS/cypha_qt_shell"

cat >"$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>cypha_qt_shell</string>
  <key>CFBundleIdentifier</key>
  <string>com.cypha.studio</string>
  <key>CFBundleName</key>
  <string>Cypha Studio</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${VERSION}</string>
  <key>CFBundleVersion</key>
  <string>${VERSION}</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

if [[ -f "$REPO_ROOT/packaging/cypha.png" ]]; then
  install -m 644 "$REPO_ROOT/packaging/cypha.png" "$APP_DIR/Contents/Resources/cypha.png"
fi

mkdir -p "$OUT_DIR_ABS"
if [[ -n "$REST_EXE" ]]; then
  install -m 755 "$REST_EXE" "$OUT_DIR_ABS/cypha_rest"
fi

echo "$VERSION" >"$OUT_DIR_ABS/VERSION"

if [[ "$WITH_FIXTURES" == "1" && -d "$REPO_ROOT/fixtures" ]]; then
  mkdir -p "$OUT_DIR_ABS/share/demo_fixtures"
  for f in reference.cypha f_field.json train_hparams.json; do
    if [[ -f "$REPO_ROOT/fixtures/$f" ]]; then
      cp "$REPO_ROOT/fixtures/$f" "$OUT_DIR_ABS/share/demo_fixtures/"
    fi
  done
fi

echo "==> macdeployqt (bundle Qt frameworks into $APP_NAME)"
"$MACDEPLOYQT" "$APP_DIR" -verbose=1

echo ""
echo "Created $APP_DIR"
if [[ -n "$REST_EXE" ]]; then
  echo "Sidecar REST: $OUT_DIR_ABS/cypha_rest"
fi
echo ""
echo "Run:"
echo "  open \"$APP_DIR\""
echo "  \"$APP_DIR/Contents/MacOS/cypha_qt_shell\" --smoke share/demo_fixtures/reference.cypha"
echo ""
