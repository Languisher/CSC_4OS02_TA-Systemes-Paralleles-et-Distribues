import argparse
import csv
import time
from pathlib import Path

import numpy as np
from numba import get_num_threads, set_num_threads

from nbodies_grid_numba_parallel import NBodySystem as ParallelNBodySystem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the numba-parallel version as a function of the thread count."
    )
    parser.add_argument("--data", default="data/galaxy_1000", help="Input galaxy dataset.")
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation time step.")
    parser.add_argument(
        "--grid",
        nargs=3,
        type=int,
        metavar=("NI", "NJ", "NK"),
        default=(20, 20, 1),
        help="Cartesian grid dimensions.",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Thread counts to benchmark.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps before timing.")
    parser.add_argument("--steps", type=int, default=10, help="Timed steps per thread count.")
    parser.add_argument(
        "--csv",
        default="outputs/numba_parallel_benchmark.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def simulate_display_step(positions, colors, luminosities):
    frame_points = np.asarray(positions, dtype=np.float32).copy()
    frame_colors = (
        np.asarray(colors, dtype=np.float32)
        * np.asarray(luminosities, dtype=np.float32)[:, np.newaxis]
        / 255.0
    )
    return frame_points, frame_colors


def benchmark_system(system_cls, args):
    system = system_cls(args.data, ncells_per_dir=tuple(args.grid))
    intensities = np.clip(system.masses / system.max_mass, 0.5, 1.0).astype(np.float32)
    colors = np.asarray(system.colors, dtype=np.float32)

    for _ in range(args.warmup_steps):
        system.update_positions(args.dt)
        simulate_display_step(system.positions, colors, intensities)

    compute_samples = []
    total_samples = []
    for _ in range(args.steps):
        total_start = time.perf_counter()
        simulate_display_step(system.positions, colors, intensities)
        compute_start = time.perf_counter()
        system.update_positions(args.dt)
        total_end = time.perf_counter()

        compute_samples.append((total_end - compute_start) * 1000.0)
        total_samples.append((total_end - total_start) * 1000.0)

    return {
        "compute_ms": sum(compute_samples) / len(compute_samples),
        "frame_ms": sum(total_samples) / len(total_samples),
    }


def benchmark_thread_count(args, thread_count):
    set_num_threads(thread_count)
    actual_threads = get_num_threads()
    result = benchmark_system(ParallelNBodySystem, args)
    result["threads_requested"] = thread_count
    result["threads_used"] = actual_threads
    return result


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for thread_count in args.threads:
        result = benchmark_thread_count(args, thread_count)
        results.append(result)
        print(
            f"threads={result['threads_used']} | "
            f"compute={result['compute_ms']:.3f} ms | frame={result['frame_ms']:.3f} ms"
        )

    baseline_compute = results[0]["compute_ms"]
    baseline_frame = results[0]["frame_ms"]
    for result in results:
        result["compute_speedup"] = baseline_compute / result["compute_ms"]
        result["frame_speedup"] = baseline_frame / result["frame_ms"]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threads_requested",
                "threads_used",
                "compute_ms",
                "frame_ms",
                "compute_speedup",
                "frame_speedup",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV écrit dans {csv_path}")


if __name__ == "__main__":
    main()
