# Pull Request

## Summary

<!-- What does this PR do? One paragraph. -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature / architectural upgrade
- [ ] Native C++ port / parity update
- [ ] Documentation / benchmark only
- [ ] Cleanup / refactor

## Checklist

### Native C++ (required)

- [ ] `cmake -S native -B native/build && cmake --build native/build -j4` succeeds
- [ ] `ctest --test-dir native/build --output-on-failure` passes
- [ ] `ctest --test-dir native/build -R native_ --output-on-failure` passes (parity + smoke subset)
- [ ] Windows: `powershell -File scripts\cypha_native_validate_all.ps1` (or `-SkipBuild` after rebuild)
- [ ] CUDA changes: local `-DCYPHA_ENABLE_CUDA=ON` build and **`native_cuda_smoke`** / **`native_score_batch`** (CUDA is not in CI)

### If changing `cypha_core` math or `.cypha` format

- [ ] Parity fixtures updated (regenerate sidecars if numerics changed)
- [ ] `ctest --test-dir native/build -R native_parity --output-on-failure` passes
- [ ] `docs/port/PORT_CONTRACT.md` updated if API shape changed

### Documentation

- [ ] `CHANGELOG.md` [Unreleased] section updated
- [ ] `docs/RESEARCH_STATUS.md` updated if benchmark numbers changed

## Benchmark impact

<!-- If this touches core math, include before/after on at least D01/D06/D08:
     cypha_bench_run -->

| Domain | Before | After |
|--------|--------|-------|
| D01 4-blobs | | |
| D06 Go | | |
| D08 MNIST HOG | | |
