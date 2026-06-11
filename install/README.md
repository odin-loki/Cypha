# Installers

Platform-specific setup scripts for Cypha.

| Script | Platform | What it sets up |
|--------|----------|-----------------|
| [`install_windows.ps1`](install_windows.ps1) | Windows (PowerShell) | Python venv, dependencies, optional MSVC native build + windeployqt |
| [`install_linux.sh`](install_linux.sh) | Linux / WSL (bash) | Python venv, dependencies, optional GCC/CMake native build + CTest gate |

| [`install_release_linux.sh`](install_release_linux.sh) | Linux tarball | Install prebuilt `bin/` from a GitHub Release bundle |
| [`install_release_windows.ps1`](install_release_windows.ps1) | Windows zip | Install prebuilt `bin/` from a GitHub Release bundle |

---

## GitHub Releases (prebuilt native)

Tagged releases (`v*`) build **Linux** and **Windows** installer archives via [`.github/workflows/release.yml`](../.github/workflows/release.yml):

| Asset | Platform |
|-------|----------|
| `cypha-<ver>-linux-x86_64.tar.gz` | Linux x86_64 (glibc + OpenMP) |
| `cypha-<ver>-windows-x86_64.zip` | Windows x86_64 (MinGW PE, static libgcc/libstdc++) |

Prebuilt native installers: **[GitHub Releases — latest `v2.2.6`](https://github.com/odin-loki/Cypha/releases/latest)** (`cypha-*-linux-x86_64.tar.gz`, `cypha-*-windows-x86_64.zip`).

```bash
# Linux (example)
tar xzf cypha-2.2.6-linux-x86_64.tar.gz && cd cypha-2.2.6-linux-x86_64 && bash install.sh
```

```powershell
# Windows (example)
Expand-Archive cypha-2.2.6-windows-x86_64.zip
cd cypha-2.2.6-windows-x86_64
powershell -ExecutionPolicy Bypass -File install.ps1
```

Full Python Studio: use the source installers below (clone repo).

---

## Quick start

**Windows (Python only, headless):**
```powershell
cd C:\path\to\Cypha
powershell -ExecutionPolicy Bypass -File install\install_windows.ps1
```

**Windows (full Studio + native C++ Qt shell):**
```powershell
$env:QT_PREFIX = "C:\Qt\6.7.0\msvc2022_64"
powershell -ExecutionPolicy Bypass -File install\install_windows.ps1 -Studio -Native -Qt
```

**Linux / WSL (Python only):**
```bash
bash install/install_linux.sh
```

**Linux / WSL (full — Python + native + Qt):**
```bash
sudo apt-get install -y cmake ninja-build build-essential qt6-base-dev libsqlite3-dev
bash install/install_linux.sh --studio --native --qt
```

**Linux with CUDA:**
```bash
bash install/install_linux.sh --native --cuda
```

---

## After installation

Activate the Python environment, then:

```bash
# Verify everything is healthy
pytest tests/ -m "not slow"

# Run the Studio GUI
python cypha_studio/main.py

# Run the native Qt shell (if built)
./native/build-install/qt/cypha_qt_shell        # Linux
.\native\build-windows-msvc\Release\cypha_qt_shell.exe   # Windows (after windeployqt)

# Run the multi-domain benchmark
python benchmark.py
```

See the root [CONTRIBUTING.md](../CONTRIBUTING.md) for the full test checklist and
[native/README.md](../native/README.md) for advanced native build options (CUDA
arch selection, MinGW cross-compile, CMake presets).
