# Installers

Platform-specific setup scripts for Cypha.

| Script | Platform | What it sets up |
|--------|----------|-----------------|
| [`install_windows.ps1`](install_windows.ps1) | Windows (PowerShell) | Python venv, dependencies, optional MSVC native build + windeployqt |
| [`install_linux.sh`](install_linux.sh) | Linux / WSL (bash) | Python venv, dependencies, optional GCC/CMake native build + CTest gate |

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
