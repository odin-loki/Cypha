# Cypha root Makefile — native convenience wrappers (see `make help`).

.PHONY: help test native-test native-bench-gpu regen-parity

help:
	@echo "Cypha Makefile targets (native-only; Python runtime removed in P7):"
	@echo "  help              — list targets (this message)"
	@echo "  native-test       — configure Release native build + run CTest (native_*)"
	@echo "  native-bench-gpu  — WSL/Linux CUDA smoke + cyphalm_bench_native (see scripts/wsl_bench_gpu.sh)"
	@echo "  regen-parity      — regenerate fixtures/ (native parity tools; see fixtures/README.md)"

native-test:
	bash scripts/ci_native_linux.sh

native-bench-gpu:
	bash scripts/wsl_bench_gpu.sh

regen-parity:
	@echo "Regenerate fixtures with native parity tools documented in fixtures/README.md"
