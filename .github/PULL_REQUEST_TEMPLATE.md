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

### Python
- [ ] `python test_cypha.py` passes (54 deterministic checks)
- [ ] `python cypha_studio/test_cypha_studio.py` passes
- [ ] `pytest tests/ -m "not slow"` passes

### If changing `Cypha.py` core math or `.cypha` format
- [ ] Parity fixtures regenerated: `python scripts/generate_parity_fixtures.py`
- [ ] `pytest tests/test_parity_fixtures.py -v` passes
- [ ] `docs/port/PORT_CONTRACT.md` updated if API shape changed

### If changing native C++
- [ ] `cmake -S native -B native/build && cmake --build native/build -j4` succeeds
- [ ] `ctest --test-dir native/build --output-on-failure` passes
- [ ] Subprocess parity tests pass: `pytest tests/test_*_native*.py -v`
- [ ] Windows: `powershell -File scripts\cypha_native_validate_all.ps1` (or `-SkipBuild` after rebuild)
- [ ] CUDA changes: local `-DCYPHA_ENABLE_CUDA=ON` build or rely on **`windows_cuda_msvc`** + **`linux_cuda`** CI jobs

### Documentation
- [ ] `CHANGELOG.md` [Unreleased] section updated
- [ ] `docs/RESEARCH_STATUS.md` updated if benchmark numbers changed

## Benchmark impact

<!-- If this touches core math, include before/after on at least D01/D06/D08:
     python benchmark_baseline.py -->

| Domain | Before | After |
|--------|--------|-------|
| D01 4-blobs | | |
| D06 Go | | |
| D08 MNIST HOG | | |
