.PHONY: all install initial-times numba-parallel mpi-split mpi-full clean

UV ?= uv
MPIEXEC ?= mpirun
UV_CACHE_DIR ?= .uv-cache
XDG_CACHE_HOME ?= .cache
MPLCONFIGDIR ?= .mplconfig
PYTHON ?= XDG_CACHE_HOME=$(XDG_CACHE_HOME) MPLCONFIGDIR=$(MPLCONFIGDIR) UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python
DATA ?= data/galaxy_1000
DT ?= 0.001
NI ?= 20
NJ ?= 20
NK ?= 1
FRAMES ?= 30
WARMUP ?= 3
CSV ?= outputs/initial_times.csv
PLOT ?= outputs/initial_times.png
THREADS ?= 1 2 4 8
STEPS ?= 10
NUMBA_CSV ?= outputs/numba_parallel_benchmark.csv
NUMBA_PLOT ?= outputs/numba_parallel_benchmark.png
BASELINE_COMPUTE_MS ?=
BASELINE_FRAME_MS ?=
MPI_CSV ?= outputs/mpi_split_benchmark.csv
MPI_PLOT ?= outputs/mpi_split_benchmark.png
FULL_MPI_CSV ?= outputs/mpi_full_benchmark.csv
FULL_MPI_PLOT ?= outputs/mpi_full_benchmark.png
PROCESSES ?= 2 3 4 5 6 7 8

BASELINE_ARGS :=
ifneq ($(strip $(BASELINE_COMPUTE_MS)),)
BASELINE_ARGS += --baseline-compute-ms $(BASELINE_COMPUTE_MS)
endif
ifneq ($(strip $(BASELINE_FRAME_MS)),)
BASELINE_ARGS += --baseline-frame-ms $(BASELINE_FRAME_MS)
endif

all: install

install:
	XDG_CACHE_HOME=$(XDG_CACHE_HOME) MPLCONFIGDIR=$(MPLCONFIGDIR) UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync

initial-times: install
	$(PYTHON) measure_initial_times.py --mode headless --data $(DATA) --dt $(DT) --grid $(NI) $(NJ) $(NK) --warmup-steps $(WARMUP) --frames $(FRAMES) --csv $(CSV)
	$(PYTHON) plot_initial_times.py --csv $(CSV) --output $(PLOT)

numba-parallel: install
	$(PYTHON) benchmark_numba_parallel.py --data $(DATA) --dt $(DT) --grid $(NI) $(NJ) $(NK) --warmup-steps $(WARMUP) --steps $(STEPS) --threads $(THREADS) $(BASELINE_ARGS) --csv $(NUMBA_CSV)
	$(PYTHON) plot_numba_parallel.py --csv $(NUMBA_CSV) --output $(NUMBA_PLOT)

mpi-split: install
	$(MPIEXEC) -n 2 /bin/zsh -lc 'XDG_CACHE_HOME=$(XDG_CACHE_HOME) MPLCONFIGDIR=$(MPLCONFIGDIR) UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python benchmark_mpi_split.py --data $(DATA) --dt $(DT) --grid $(NI) $(NJ) $(NK) --warmup-steps $(WARMUP) --steps $(STEPS) --threads $(THREADS) --csv $(MPI_CSV)'
	$(PYTHON) plot_mpi_split.py --csv $(MPI_CSV) --output $(MPI_PLOT)

mpi-full: install
	$(PYTHON) benchmark_mpi_full.py --data $(DATA) --dt $(DT) --grid $(NI) $(NJ) $(NK) --warmup-steps $(WARMUP) --steps $(STEPS) --processes $(PROCESSES) --threads $(THREADS) $(BASELINE_ARGS) --csv $(FULL_MPI_CSV)
	$(PYTHON) plot_mpi_full.py --csv $(FULL_MPI_CSV) --output $(FULL_MPI_PLOT)

clean:
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name "build" -o -name "dist" \) -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
