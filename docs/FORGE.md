# FORGE in Cypha

**Odin Loch** · 2026-11-03 · Odin: "Put forge in Cypha."

FORGE is Odin's other project. It now sits in Cypha under [`forge/`](../forge/README.md). This page covers what it is, where it sits, how to build, test and update it, what passed here, how it would join Cypha's build, and where it could help Cypha.

Nothing in Cypha uses FORGE yet. Cypha's build and CI are unchanged.

---

## 1. What FORGE is

FORGE is a C++23 library and toolset for running AI models on CPUs and GPUs ([its README](../forge/README.md)). One codebase covers:

| Layer | What it gives |
|---|---|
| `lane` | SIMD kernels picked at run time for the CPU in use: scalar, AVX2, AVX-512, NEON, SVE, RVV; AMX and SME tile kernels |
| `grid` | GPU compute behind one API. Backends load at run time: CPU, Vulkan, CUDA, HIP, Level Zero, OpenCL, Metal |
| `flow` | Graph IR, optimisation passes, memory planning, weight and KV codecs |
| `pack` | The `.fpk` model container (BLAKE3, Ed25519 signing, XChaCha20-Poly1305 encryption), a BPE tokenizer, importers for safetensors, GGUF and ONNX |
| `mind` | The inference engine: sessions, samplers, batching, speculative decoding, grammars, LoRA, the C ABI |
| `balance`, `forge-serve` | An OpenAI-compatible HTTP server with a load-balancing proxy |
| `os` | A portable OS layer for Linux, Android, Windows, macOS, iOS and FreeBSD |

An `.fpk` file carries its own computation graph. So a new architecture means importing a file, not writing engine code.

FORGE is version 0.1.0 and in development. In its own words: CUDA, HIP, OpenCL, Level Zero and Metal have only run against mock drivers, never real GPUs. Vulkan has run on software devices. See [`forge/docs/STATUS.md`](../forge/docs/STATUS.md).

FORGE has the same licence as Cypha: AGPL-3.0 or later, or commercial. Its `LICENSE`, `COMMERCIAL-LICENCE.md` and `NOTICE` stay in `forge/`. Its third-party code is Monocypher and the MIT-licensed Level Zero header.

## 2. Where it is

- `forge/` at the repository root. FORGE's main branch at `5a943fb`, taken 2026-11-03: 458 files, 11 MB.
- It came in with `git subtree add --prefix=forge https://github.com/odin-loki/forge main --squash`. Cypha's history gains one squashed commit of FORGE's tree and one merge commit. FORGE's own history is not copied.
- FORGE has no Git LFS files.
- FORGE's repository stays the source. Make FORGE changes there, then pull them in (section 5).
- FORGE's own CI (`forge/.github/workflows/ci.yml`) does not run in Cypha. GitHub only reads workflows in the root `.github/`.
- Cypha's `.gitignore` and `.gitattributes` also apply inside `forge/`:
  - `*.bin` matches one FORGE file, `forge/fuzz/corpus/safetensors/regress_hdrlen_overflow.bin`. It is tracked, so it stays. `git subtree pull` brings in new files whatever the ignore rules say. Copying files in by hand and running `git add` would silently drop new `.bin`, `.so`, `.html`, `.key`, `.pem` or `models/` files. So always use `git subtree pull`.
  - `*.cpp`, `*.hpp`, `*.md` and `*.json` get `text working-tree-encoding=UTF-8`. FORGE's files are UTF-8 with LF line ends, so nothing changes. The checkout is clean.

## 3. Build and test

FORGE builds on its own, from its own CMake. You need CMake 3.25 or newer, Ninja, Python 3 and a C++23 compiler.

```sh
cmake -S forge -B build-forge -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-forge -j4
(cd build-forge && ctest -j4)
./build-forge/forge info
```

Or use FORGE's preset from inside `forge/`:

```sh
cd forge
cmake --preset dev            # build/dev, RelWithDebInfo
cmake --build build/dev -j4
(cd build/dev && ctest -j4)
./build/dev/forge info
```

- The preset builds in `forge/build/dev`. Both `forge/.gitignore` and Cypha's `.gitignore` ignore `build/`.
- Configure and the script tests run Python from the source tree. They leave `__pycache__/` folders in `forge/spec/gen/`, `forge/python/forge/` and `forge/verify/golden/`, which are ignored.
- Some script tests need `numpy`, `safetensors`, `onnx` and `cryptography` in the Python that CMake finds. Without them ctest reports those tests as "Not Run (Disabled)", not as failures.
- FORGE's own guides: [`forge/docs/building.md`](../forge/docs/building.md) and [`forge/docs/testing.md`](../forge/docs/testing.md).

## 4. Results here (2026-11-03)

The container: g++ 13.3, CMake 3.28, Ninja, Python 3.13 with numpy and cryptography but not safetensors or onnx. 4 cores, shared with other jobs (load average 9 to 10 during the build). No GPU, no Vulkan SDK, no Android NDK, no Xcode.

Nothing in FORGE was changed. Living in a subdirectory needed no change.

The full build and test ran on FORGE `692738e`. FORGE's main moved to `5a943fb` during the work (5 files: a benchmark harness and GPU driver call counters). It was pulled with `git subtree pull`, which tested the update path. The incremental rebuild took 29 steps (7 min), and both ctest runs below gave the same counts on `5a943fb`. Cypha's history keeps one add, at `5a943fb`.

- **Configure:** passed (19 s).
- **Build:** passed, 238 of 238 steps, 55 min on the shared cores (RelWithDebInfo). 380 compiler warnings (259 of them `-Wmisleading-indentation`), no errors.
- **`./forge info`:** runs. lane picks `avx512`. grid finds the CPU and the Vulkan loader; there is no CUDA, HIP or Level Zero runtime.
- **`forge run golden/tiny_f32.fpk --prompt "Hello" -n 8`:** runs. The golden model has random weights, so its text is noise.

**`ctest -j4`: 108 tests. 75 passed, 16 skipped, 16 not run, 1 failed.**

| Result | Tests | Why |
|---|---|---|
| Failed (1) | `test_script` (its cases `shell_backends_golden` and `emit_golden_all_shells`) | This container. Its `rc` on `PATH` is the Node.js `rc` config tool (`/opt/node22/bin/rc`), not the Plan 9 rc shell, and FORGE takes it for the shell. With `/opt/node22/bin` off `PATH`, `test_script` passes. |
| Skipped (16) | `test_vulkan`, `test_vulkan_align256` | Vulkan backend not built (no headers, no `glslangValidator`) |
| | `test_gpu_cuda`, `test_gpu_hip`, `test_gpu_opencl`, `test_gpu_ze`, `test_gpu_metal`, and the four `test_gpu_features_*` | No GPU drivers or devices; Metal needs Apple |
| | `mobile_android_arm64-v8a`, `mobile_android_x86_64`, `mobile_android_device`, `mobile_ios` | No Android NDK, no adb device, no Xcode |
| | `test_text_conformance` | No Unicode test data (`FORGE_FETCH_UCD` is off) |
| Not run (16) | the nine `golden_*` tests, `doc_example_mind_generate`, `doc_example_capi_generate`, `capi_smoke`, `python_bindings`, `serve_http`, `pack_signing`, `onnx_import` | Disabled at configure: Python lacks `safetensors` (and `onnx` for `onnx_import`) |

**Second run, with the Python modules.** safetensors 0.8.0 and onnx 1.23.2 were installed into a side folder on `PYTHONPATH`, not into the system Python. Then: 108 tests, 91 passed, 16 skipped, 1 failed (`test_script`, the same `rc` cause). All 16 Python tests passed, including the golden models, the four mock GPU drivers, `serve_http` and `onnx_import`. FORGE's own Cypha example, `example_cypha_classifier`, passed in both runs.

## 5. Updating from Odin's repository

```sh
git subtree pull --prefix=forge https://github.com/odin-loki/forge main --squash
```

- It needs a clean working tree.
- It adds one squashed commit and one merge commit. `git subtree` finds the last pull from the `git-subtree-split:` line in the previous squashed commit.
- After a pull, build FORGE and run its tests again (section 3).
- A fix made in Cypha's `forge/` can go back with `git subtree push --prefix=forge <your fork> <branch>` and a pull request to FORGE. It is simpler to make the change in FORGE first and pull it.

One effect of living in a subdirectory: `forge/tools/sbom.py` records the commit from `git rev-parse HEAD`. Inside Cypha that is Cypha's commit, not FORGE's. Its test (`sbom_spdx`) still passes. Make FORGE's SBOMs and releases (`tools/release.py`, which also tags) from FORGE's own repository, not from Cypha.

## 6. Cypha's build is unaffected

Checked on 2026-11-03:

- CI (`.github/workflows/ci.yml`) only configures `-S native`. `release.yml` packages `native/` builds.
- `native/CMakeLists.txt` and `native/cmake/*.cmake` have no `file(GLOB …)`. The only `add_subdirectory` is `qt`. Their `../` paths name fixed files in `fixtures/`, `bench/`, `docs/` and `data/`.
- The CTest scripts (`scripts/ci_native_*.sh`) filter with `-R native_`.
- No script or packaging step walks the repository root. The `os.walk` and `rglob` calls start in corpus folders.
- The root `Makefile` has four named targets and no globs. `make help` still works.
- No FORGE target name equals a Cypha target name.

## 7. Wiring FORGE into Cypha's build (not done)

This waits for `native/` to be reorganised into modules. The plan is an option, off by default:

```cmake
option(CYPHA_WITH_FORGE "Build FORGE from ../forge so Cypha targets can link it" OFF)
if(CYPHA_WITH_FORGE)
  set(FORGE_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
  set(FORGE_BUILD_APPS     OFF CACHE BOOL "" FORCE)
  set(FORGE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../forge ${CMAKE_BINARY_DIR}/forge EXCLUDE_FROM_ALL)
  # per Cypha module that uses it, for example:
  # target_link_libraries(<module> PRIVATE forge_lane_mv)
  # target_compile_definitions(<module> PRIVATE CYPHA_WITH_FORGE=1)
endif()
```

- `forge/` is outside `native/`, so `add_subdirectory` needs the explicit binary directory.
- `EXCLUDE_FROM_ALL` means only the FORGE targets that Cypha links get built.
- With FORGE's tests off, Cypha's CTest list does not change. The CI scripts filter `-R native_` in any case.
- FORGE needs CMake 3.25. Cypha asks for 3.19. With the option on, the build needs 3.25.
- Both use C++23. FORGE's `add_compile_options` and `CMAKE_CXX_STANDARD` stay in its own directory.
- FORGE needs Python 3 at configure time for its code generators.
- FORGE uses `PROJECT_SOURCE_DIR`, not `CMAKE_SOURCE_DIR`, so it works as a subdirectory.
- FORGE has `install()` rules. Cypha does not use `cmake --install` today. If it starts to, FORGE's rules would join it and could fail on FORGE targets that `EXCLUDE_FROM_ALL` never built.
- FORGE's own CI already builds `forge_c` alone with tests, apps and examples off (its iOS job), so a library-only build is a supported shape.
- MSVC is not checked here. FORGE's own CI has an MSVC job, but Cypha with FORGE under MSVC is untried. Keep the option off in Cypha's `windows_msvc` job until it is checked.
- CI: add one optional Linux job with `-DCYPHA_WITH_FORGE=ON` first. FORGE's own test suite stays FORGE's job.

**What it costs.** Measured here, Release, from a clean build directory, with FORGE's tests, apps and examples off. 4 shared cores, load average 6 to 9:

| What Cypha builds | Steps | Time |
|---|---|---|
| FORGE's configure (Python code generators) | | 5–20 s |
| `forge_lane_mv`: lane's kernel tables (scalar, AVX2, AVX-512, AMX tiles) | 8 | 20 s |
| then `forge_c`: the C ABI, with mind, flow, pack and grid in one library | 13 more | 80 s |
| for comparison, all of FORGE with tests, apps and examples (RelWithDebInfo) | 238 | 55 min |

So a Cypha module that links only lane adds about 20 s to a clean build. Serving through mind adds under 2 minutes more. A repeat build with nothing changed does no FORGE work. Building FORGE's whole test suite is what costs time; keep it in FORGE's own CI. Every checkout of Cypha is 11 MB larger either way.

## 8. Where FORGE could serve Cypha (possibilities, not done)

None of these is built or measured. Each would be its own registered experiment, with a measured gain before any default changes.

- **lane for CyphaLM's mixers.** The content mixer (`native/tools/cyphalm_trie.cpp`) and the adaptive mixer (`adaptive_predictor_mixer.hpp`) do dot products, sigmoids and softmaxes for every byte. lane has `dot`, `saxpy`, `sigmoid`, `tanh`, `exp`, `log`, `softmax_row` and `logsumexp`, picked at run time. Cypha has its own AVX2 and AVX-512 kernels only in the neural expert (`neural_expert.cpp`). Release builds do not use `-march=native` (`CYPHA_NATIVE_ARCH` is off), so the mixers compile for baseline x86-64. lane would give them AVX2 or AVX-512, and NEON, SVE or RVV on Arm and RISC-V, without Cypha writing code per target. Caveat: the mixers' vectors are short, and a call through a function table may cost more than it saves.
- **lane for a GLN.** GLNB ([GENERATION_PLAN §13.13](GENERATION_PLAN.md)) is Python (`bench/fuzzy/glnb.py`). A C++ GLN picks weights by a context hash, then does a dot product and a sigmoid per neuron for every bit: lane's `dot` and `sigmoid`, or `gemm_nt` for a layer at once. GLNB was not adopted (0.30 bits per byte behind the content mixer), so this matters only if a GLN comes back.
- **lane and grid for the LSTM.** `char_lstm.hpp`, and the LSTM inside GLNB, are matrix-vector products with sigmoid and tanh gates. lane has `gemm_nn`, `gemm_nt`, `sigmoid` and `tanh`, and AMX (bf16, int8) and SME tile kernels for larger hidden sizes. grid's GPU op set has `matmul_t` but no LSTM cell op. One would be added as a spec op.
- **grid for GPUs.** Cypha's GPU path is CUDA only (`accel_backend.hpp`, `CYPHA_ENABLE_CUDA`). grid loads CUDA, HIP, Vulkan, Level Zero, OpenCL and Metal at run time and links no vendor SDK. Its GPU backends have not yet run on real GPUs, so this waits on FORGE. grid's parallel CPU primitives (`radix_sort`, `scan`, `histogram`, `top_k`) might also serve index building.
- **pack for Cypha's files.** An `.fpk` holds named sections with BLAKE3 hashes, Ed25519 signatures and optional encryption. Packed tries, checkpoints and data packs could ship that way, signed. FORGE's own example [`forge/examples/integrations/cypha_classifier.cpp`](../forge/examples/integrations/cypha_classifier.cpp) already stores a toy Cypha classifier's statistics as `.fpk` sections.
- **mind and forge-serve for serving.** mind has a `custom` session kind (`forge/mind/include/forge/mind/plugin.hpp`) for engines that are not neural networks; its header names Cypha as the example. A CyphaLM service registered there would be reachable through FORGE's C ABI and its Python, Kotlin and Swift bindings. Whether `forge-serve`'s OpenAI-compatible endpoints can drive a custom session is not checked. `balance` could sit in front of several `cypha_rest` servers as a load-balancing proxy (also not checked).
- **os for large files.** FORGE's OS layer has memory-mapped files, huge pages, NUMA and CPU topology on every platform. These could help load and read the packed 1 TB trie.
