# Ant Simulation Project

This project includes a C++ simulation program (baseline / vectorized / OpenMP / MPI) and Python plotting scripts.

## 0. Project Structure

- `src/`: C++ source code and `Makefile`
- `output/`: executables and benchmark CSV outputs
- `plots/`: Python plotting scripts, `pyproject.toml`, and generated figures
- `README.md`: usage guide

## 1. Prerequisites

### 1.1 System Requirements

At minimum, you need:

- `g++` (C++17 support, GCC 13+ recommended)
- `make`
- `SDL2` (for rendering linkage)
- `OpenMP` (parallel mode)
- `mpic++` + `mpirun` (for MPI targets)

Current defaults in `src/Makefile`:

- `CXX = g++-15`
- `-fopenmp`
- `-lSDL2`

If `g++-15` is not available on your machine, override the compiler:

```bash
cd src
make all CXX=g++
```

### 1.2 Python Dependencies (for `plots`)

Plot dependencies are defined in `plots/pyproject.toml`:

- Python version requirement: `>=3.13`

- `matplotlib`
- `pandas`
- `scienceplots`

Recommended with `uv`:

```bash
cd plots
uv sync
```

Or with `pip`:

```bash
cd plots
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib pandas scienceplots
```

## 2. Build and Basic Run

```bash
cd src
make all
```

Generated executables:

- `../output/ant_simu.exe`
- `../output/ant_simu_mpi.exe` (after running `make mpi`)

## 3. Run Benchmarks (Make Targets)

Run all commands below from `src/`.

### 3.1 Basic Benchmark

```bash
make benchmark
```

Output: `../output/timing_results.csv`

### 3.2 Mode Comparison (non-vectorized / vectorized / vectorized+OpenMP)

```bash
make benchmark_non_vectorized
make benchmark_vectorized
make benchmark_vectorized_openmp
# or run all three at once
make benchmark_modes
```

Outputs:

- `../output/timing_non_vectorized.csv`
- `../output/timing_vectorized.csv`
- `../output/timing_vectorized_openmp.csv`

### 3.3 Second Optimization (submap)

```bash
make benchmark_vectorized_openmp_submap
# equivalent alias
make benchmark_second_amelioration
```

Output: `../output/timing_vectorized_openmp_submap.csv`

### 3.4 Compute-Size Scalability (shared-memory path)

```bash
make benchmark_scalability
```

Output: `../output/timing_scalability.csv`

Custom example:

```bash
make benchmark_scalability SCALABILITY_ANTS="1000 3000 5000 10000" SCALABILITY_ITER=2000 SCALABILITY_RUNS=5
```

### 3.5 MPI Benchmarks (first distributed-memory approach)

First compile the MPI executable:

```bash
make mpi
```

Then run:

```bash
make benchmark_mpi
make benchmark_mpi_scalability
make benchmark_mpi_compute_scalability
```

Outputs:

- `../output/timing_mpi.csv`
- `../output/timing_mpi_speedup.csv`
- `../output/timing_mpi_compute_scalability.csv`

Custom examples:

```bash
make benchmark_mpi_scalability MPI_PROCS="1 2 4 8" MPI_RUNS=5 MPI_ITER=1000 MPI_ANTS=5000
make benchmark_mpi_compute_scalability MPI_COMP_NP=4 MPI_COMP_ANTS="1000 2500 5000 10000" MPI_COMP_RUNS=3
```

## 4. Generate Plots with Python

Run the following commands from `plots/`.

```bash
cd plots
```

### 4.1 Baseline Workflow Plot

```bash
python baseline.py
```

Input: `../output/timing_results.csv`  
Output: `workflow_runtime.png/.pdf`

### 4.2 Shared-Memory Mode Comparison Plot

```bash
python mode_comparison.py
```

Inputs:

- `../output/timing_non_vectorized.csv`
- `../output/timing_vectorized.csv`
- `../output/timing_vectorized_openmp.csv`

Output: `mode_comparison_scienceplots.png/.pdf`

### 4.3 Shared-Memory Scalability Plot

```bash
python scalability.py
```

Input: `../output/timing_scalability.csv`  
Output: `scalability_runtime.png/.pdf`

### 4.4 Submap (second optimization) Comparison Plot

```bash
python second_amelioration_submap_scienceplots.py
```

Inputs:

- `../output/timing_vectorized_openmp.csv`
- `../output/timing_vectorized_openmp_submap.csv`

Output: `second_amelioration_submap_scienceplots.png/.pdf`

### 4.5 Baseline vs Submap Thread-Scaling Plot

```bash
python submap_thread_scaling_scienceplots.py --csv ../output/timing_cmp_summary.csv
```

Input: `../output/timing_cmp_summary.csv`  
Output: `submap_thread_scaling_scienceplots.png/.pdf`

### 4.6 MPI Scalability Plot

`mpi_scalability_scienceplots.py` defaults to `timing_mpi_speedup_fair.csv`.  
If you generated `timing_mpi_speedup.csv` with `make benchmark_mpi_scalability`, pass it explicitly:

```bash
python mpi_scalability_scienceplots.py \
  --speedup-csv ../output/timing_mpi_speedup.csv \
  --baseline-csv ../output/timing_vectorized_openmp.csv
```

Output: `mpi_scalability_scienceplots.png/.pdf`

### 4.7 MPI Compute-Size Scalability Plot

```bash
python mpi_compute_scalability_scienceplots.py
```

Input: `../output/timing_mpi_compute_scalability.csv`  
Output: `mpi_compute_scalability_scienceplots.png/.pdf`

### 4.8 MPI vs Shared-Memory Comparison Plot

Again, it is recommended to pass the speedup CSV explicitly:

```bash
python mpi_vs_shared_memory_scienceplots.py \
  --mpi-csv ../output/timing_mpi_speedup.csv \
  --baseline-csv ../output/timing_vectorized_openmp.csv
```

Output: `mpi_vs_shared_memory_scienceplots.png/.pdf`

### 4.9 Premiere MPI vs Baseline Summary Plot

```bash
python premiere_mpi_vs_baseline_scaling_scienceplots.py \
  --csv ../output/timing_cmp_premiere_vs_baseline_summary.csv
```

Output: `premiere_mpi_vs_baseline_scaling_scienceplots.png/.pdf`

## 5. Typical End-to-End Workflow

```bash
# 1) Build and run benchmarks
cd src
make benchmark_modes
make benchmark_vectorized_openmp_submap
make benchmark_scalability
make benchmark_mpi_scalability
make benchmark_mpi_compute_scalability

# 2) Generate plots
cd ../plots
uv sync  # or use your pip virtual environment
python mode_comparison.py
python second_amelioration_submap_scienceplots.py
python scalability.py
python mpi_scalability_scienceplots.py --speedup-csv ../output/timing_mpi_speedup.csv --baseline-csv ../output/timing_vectorized_openmp.csv
python mpi_compute_scalability_scienceplots.py
python mpi_vs_shared_memory_scienceplots.py --mpi-csv ../output/timing_mpi_speedup.csv --baseline-csv ../output/timing_vectorized_openmp.csv
```
