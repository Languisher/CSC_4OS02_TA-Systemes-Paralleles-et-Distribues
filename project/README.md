## Timer


```
cd project/src
make all
../output/ant_simu.exe --benchmark --runs 10 --iterations 2000 --render 0 --csv ../output/timing_results.csv
```

### Detailed component timing + scalability

`ant_simu.exe` now supports:
- `--ants <N>`: control workload size (number of ants)
- CSV metrics for:
  - `ant_select_move_ms`
  - `ant_terrain_cost_ms`
  - `ant_mark_pheromone_ms`
  - `evaporation_ms`
  - `pheromone_update_ms`
  - normalized per-work metrics (`*_per_move`, `*_per_cell`, `*_per_iter`)

Run default benchmark:
```
cd project/src
make benchmark
```

Run scalability sweep (different ant counts) and aggregate `avg` rows:
```
cd project/src
make benchmark_scalability
```

Customize sweep:
```
cd project/src
make benchmark_scalability SCALABILITY_ANTS="1000 3000 5000 10000" SCALABILITY_ITER=2000 SCALABILITY_RUNS=5
```

## 
