# Packaging

Standalone **native C++** distributables for Cypha — no Python runtime or venv. See [`docs/FUTURE.md`](../docs/FUTURE.md) §3.

| Script | Platform | Output |
|--------|----------|--------|
| [`install_release_linux.sh`](install_release_linux.sh) | Linux tarball | Install prebuilt `bin/` from a GitHub Release |
| [`install_release_windows.ps1`](install_release_windows.ps1) | Windows zip | Install prebuilt `bin/` from a GitHub Release |
| [`build_appimage.sh`](build_appimage.sh) | Linux (local/CI) | Self-contained **`cypha_qt_shell`** AppImage + **`cypha_rest`** sidecar |
| [`build_windows_bundle.ps1`](build_windows_bundle.ps1) | Windows (native) | **`windeployqt`** folder with Qt DLLs |

Release archives are also produced by [`scripts/package_release_linux.sh`](../scripts/package_release_linux.sh) and [`scripts/package_release_windows.ps1`](../scripts/package_release_windows.ps1) in [`.github/workflows/release.yml`](../.github/workflows/release.yml).

---

## GitHub Releases (prebuilt native)

Tagged releases (`v*`) publish **native-only** assets (no Python venv). Maintainer helpers:

```powershell
# One-time: authenticate gh CLI for local release prep / draft notes
gh auth login

# Optional: emit markdown notes for a tag (stub — append to release body locally)
pwsh -File scripts/create_release_notes.ps1 -Tag v2.3.24
```

CI workflow [`.github/workflows/release.yml`](../.github/workflows/release.yml) uploads tarballs/AppImage/zip on tag push; `generate_release_notes: true` fills the GitHub UI body. Use `create_release_notes.ps1` for a git-log supplement before tagging.

| Asset | Platform | Contents |
|-------|----------|----------|
| `cypha-<ver>-linux-x86_64.tar.gz` | Linux x86_64 | CLI tools (`cypha_rest`, bench, diagnostics, …) |
| `cypha-<ver>-linux-x86_64.AppImage` | Linux x86_64 | Standalone Qt shell + bundled `cypha_rest` |
| `cypha-<ver>-windows-x86_64.zip` | Windows x86_64 | Native **MSVC** CLI tools |
| `cypha-<ver>-arxiv-bundle.zip` | All | Paper figures bundle (no runtime) |

Prebuilt installers: **[GitHub Releases — latest (`v2.3.24`)](https://github.com/odin-loki/Cypha/releases/tag/v2.3.24)**.

```bash
# Linux CLI tarball
tar xzf cypha-2.3.24-linux-x86_64.tar.gz && cd cypha-2.3.24-linux-x86_64 && bash install.sh

# Linux Qt AppImage (no install step)
chmod +x cypha-2.3.24-linux-x86_64.AppImage
./cypha-2.3.24-linux-x86_64.AppImage
```

```powershell
# Windows CLI zip (MSVC)
Expand-Archive cypha-2.3.24-windows-x86_64.zip
cd cypha-2.3.24-windows-x86_64
powershell -ExecutionPolicy Bypass -File install.ps1
```

---

## Linux AppImage (standalone Qt shell)

Build a single-file **`cypha_qt_shell`** with Qt libraries bundled via **linuxdeploy** + **linuxdeploy-plugin-qt**. **`cypha_rest`** is included as a sidecar in `usr/bin/`.

### Prerequisites

```bash
sudo apt-get install -y cmake ninja-build g++ libsqlite3-dev \
  qt6-base-dev libxkbcommon-x11-0 libxcb-cursor0 libegl1 libgl1-mesa-dri libglib2.0-0
```

### One-time: linuxdeploy tools

```bash
export LINUXDEPLOY_URL="https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
export LINUXDEPLOY_QT_URL="https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-x86_64.AppImage"
wget -O /tmp/linuxdeploy-x86_64.AppImage "$LINUXDEPLOY_URL"
wget -O /tmp/linuxdeploy-plugin-qt-x86_64.AppImage "$LINUXDEPLOY_QT_URL"
chmod +x /tmp/linuxdeploy-*.AppImage
export LINUXDEPLOY=/tmp/linuxdeploy-x86_64.AppImage
export LINUXDEPLOY_QT=/tmp/linuxdeploy-plugin-qt-x86_64.AppImage
export PATH="/tmp:$PATH"
export APPIMAGE_EXTRACT_AND_RUN=1
```

### Build

```bash
# From repo root
bash packaging/build_appimage.sh 2.3.24

# Re-package from an existing CMake tree (e.g. CI build-release):
SKIP_BUILD=1 bash packaging/build_appimage.sh 2.3.24 native/build-release dist
```

Output: `dist/cypha-<ver>-linux-x86_64.AppImage`

### Run

```bash
chmod +x dist/cypha-2.3.24-linux-x86_64.AppImage
./dist/cypha-2.3.24-linux-x86_64.AppImage

# Headless smoke (demo fixtures bundled when built from a full tree)
./dist/cypha-2.3.24-linux-x86_64.AppImage --appimage-extract-and-run --smoke \
  usr/share/cypha/demo_fixtures/reference.cypha
```

**linuxdeploy steps** (automated by `build_appimage.sh`):

1. `cmake -DCYPHA_BUILD_QT=ON -DCMAKE_BUILD_TYPE=Release` → `cypha_qt_shell` + `cypha_rest`
2. Stage `Cypha.AppDir/usr/bin/` with both binaries + `.desktop` + `AppRun`
3. `linuxdeploy --appdir … --executable … --plugin qt --output appimage`

---

## Windows standalone folder (windeployqt)

Requires **Qt 6 installed natively on Windows** ([qt.io](https://www.qt.io/download)). MinGW **cross-builds from WSL** cannot produce a Qt GUI bundle — build on Windows.

```powershell
# From repo root — configures, builds, and runs windeployqt
powershell -ExecutionPolicy Bypass -File packaging\build_windows_bundle.ps1 `
  -Version 2.3.24 `
  -WithFixtures

# Package from an existing build:
powershell -ExecutionPolicy Bypass -File packaging\build_windows_bundle.ps1 `
  -BuildDir native\build `
  -QtBinDir "C:\Qt\6.11.0\msvc2022_64\bin" `
  -SkipBuild `
  -WithFixtures
```

Output: `dist\cypha-<ver>-windows-qt\` containing `cypha_qt_shell.exe`, Qt DLLs, and optionally `cypha_rest.exe`.

```powershell
dist\cypha-2.3.24-windows-qt\cypha_qt_shell.exe
dist\cypha-2.3.24-windows-qt\cypha_qt_shell.exe --smoke share\demo_fixtures\reference.cypha
```

Lower-level helper (same windeployqt pattern): [`native/scripts/package_windows_qt.ps1`](../native/scripts/package_windows_qt.ps1).

---

## After installation

Release `bin/` includes production tools: **`cypha_rest`**, **`cypha_bench_run`**, **`cypha_bench_report`**, **`cypha_tune_run`**, **`cypha_diagnostics_run`**, and dev parity tools under **`bin/dev/`**. The AppImage / Windows Qt folder adds **`cypha_qt_shell`** as a self-contained GUI.

```bash
# Verify (from a source build tree)
ctest --test-dir native/build -R native_ --output-on-failure

# Run REST
cypha_rest --listen 127.0.0.1:8099 \
  --cypha share/demo_fixtures/reference.cypha \
  --f-field-json share/demo_fixtures/f_field.json
```

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full validation checklist and [`native/README.md`](../native/README.md) for CUDA, MinGW cross-compile, and CMake presets.
