# Cypha root Makefile — convenience wrappers (see `make help`).

.PHONY: help test regen-parity experiment-ddl bench profile-fast cov gpu-bench gpu-fullbench e2e-profile e2e-profile-fast tune-coarse tune-medium tune-gpu-heavy bench-gpu-prod

help:
	@echo "Cypha Makefile targets:"
	@echo "  help              — list targets (this message)"
	@echo "  test              — pytest tests/ + legacy test_cypha.py + cypha_studio tests"
	@echo "  regen-parity      — regenerate parity_fixtures/ (generate_parity_fixtures.py)"
	@echo "  experiment-ddl    — export ExperimentDB DDL to artifacts/experiment_schema.sql"
	@echo "  bench             — run benchmark.py"
	@echo "  profile-fast      — quick real-dataset profile (profile_real_datasets.py --fast)"
	@echo "  cov               — pytest with coverage (cypha_studio, cypha_accel)"
	@echo "  gpu-bench         — GPU microbench (scripts/gpu_microbench.py)"
	@echo "  gpu-fullbench     — full GPU bench (scripts/gpu_fullbench.py)"
	@echo "  e2e-profile       — download + profile e2e datasets"
	@echo "  e2e-profile-fast  — e2e profile with --fast"
	@echo "  tune-coarse       — coarse quality/performance tune (includes generation)"
	@echo "  tune-medium       — medium preset tune (max 150 combos)"
	@echo "  tune-gpu-heavy    — coarse tune with heavy GPU burn settings"
	@echo "  bench-gpu-prod    — microbench + fullbench + coarse tune (needs CuPy)"

# Run Python + Studio test suites (headless Qt).
test:
	QT_QPA_PLATFORM=offscreen pytest tests/ -v --tb=short
	python test_cypha.py
	python cypha_studio/test_cypha_studio.py

# Regenerate core parity fixtures under parity_fixtures/.
regen-parity:
	python scripts/generate_parity_fixtures.py

# M6: emit ExperimentDB DDL to artifacts/ (canonical source is cypha_studio.core.experiment._SCHEMA).
experiment-ddl:
	python scripts/export_experiment_schema_sql.py -o artifacts/experiment_schema.sql

bench:
	python benchmark.py

profile-fast:
	python scripts/profile_real_datasets.py --fast

cov:
	QT_QPA_PLATFORM=offscreen pytest tests/ -q --cov=cypha_studio --cov=cypha_accel --cov-report=term-missing

gpu-bench:
	python scripts/gpu_microbench.py

gpu-fullbench:
	python scripts/gpu_fullbench.py

e2e-profile:
	python scripts/download_profile_e2e.py

e2e-profile-fast:
	python scripts/download_profile_e2e.py --fast

tune-coarse:
	python scripts/tune_quality_performance.py --preset coarse --include-generation

tune-medium:
	python scripts/tune_quality_performance.py --preset medium --max-combos 150 --include-generation --jobs 2

tune-gpu-heavy:
	python scripts/tune_quality_performance.py --preset coarse --include-generation --jobs 1 --gpu-burn-passes 64 --gpu-batch-n 12288 --gpu-stress-repeats 24

# Full production GPU path: microbench + fullbench + coarse tune (needs CuPy in this python; use 3.10–3.12).
bench-gpu-prod:
	python scripts/bench_gpu_production.py
