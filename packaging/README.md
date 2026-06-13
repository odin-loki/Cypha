# Installers

Prebuilt **native** release bundles from GitHub Releases. For building from source, see [`docs/native/NATIVE_QUICKSTART.md`](../docs/native/NATIVE_QUICKSTART.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md).

| Script | Platform | What it sets up |
|--------|----------|-----------------|
| [`install_release_linux.sh`](install_release_linux.sh) | Linux tarball | Install prebuilt `bin/` from a GitHub Release bundle |
| [`install_release_windows.ps1`](install_release_windows.ps1) | Windows zip | Install prebuilt `bin/` from a GitHub Release bundle |

---

## GitHub Releases (prebuilt native)

Tagged releases (`v*`) build **Linux** and **Windows** installer archives via [`.github/workflows/release.yml`](../.github/workflows/release.yml):

| Asset | Platform |
|-------|----------|
| `cypha-<ver>-linux-x86_64.tar.gz` | Linux x86_64 (glibc + OpenMP) |
| `cypha-<ver>-windows-x86_64.zip` | Windows x86_64 (MinGW PE, static libgcc/libstdc++) |

Prebuilt native installers: **[GitHub Releases — latest `v2.2.8`](https://github.com/odin-loki/Cypha/releases/latest)** (`cypha-*-linux-x86_64.tar.gz`, `cypha-*-windows-x86_64.zip`).

```bash
# Linux (example)
tar xzf cypha-2.2.7-linux-x86_64.tar.gz && cd cypha-2.2.7-linux-x86_64 && bash install.sh
```

```powershell
# Windows (example)
Expand-Archive cypha-2.2.7-windows-x86_64.zip
cd cypha-2.2.7-windows-x86_64
powershell -ExecutionPolicy Bypass -File install.ps1
```

---

## After installation

Release `bin/` includes production tools: **`cypha_rest`**, **`cypha_bench_run`**, **`cypha_bench_report`**, **`cypha_tune_run`**, **`cypha_diagnostics_run`**, **`cypha_qt_shell`** (when packaged), and dev parity tools under **`bin/dev/`**.

```bash
# Verify (from a source build tree)
ctest --test-dir native/build -R native_ --output-on-failure

# Full Windows gate (rebuild + CTest + bench + REST smoke)
powershell -File scripts/cypha_native_validate_all.ps1

# Run the native Qt shell (if built or packaged)
./native/build/qt/cypha_qt_shell        # Linux
.\native\build\cypha_qt_shell.exe     # Windows

# Multi-domain benchmark
cypha_bench_run --from-domain 1
```

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full validation checklist and [`native/README.md`](../native/README.md) for advanced build options (CUDA, MinGW cross-compile, CMake presets).
