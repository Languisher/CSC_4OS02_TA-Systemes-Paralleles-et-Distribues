import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep MPI process counts and numba thread counts for the full MPI version."
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
        "--processes",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8],
        help="Total MPI process counts to benchmark. Rank 0 is display; others compute.",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Numba thread counts on compute ranks.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps before timing.")
    parser.add_argument("--steps", type=int, default=10, help="Timed steps.")
    parser.add_argument(
        "--baseline-compute-ms",
        type=float,
        default=54.41,
        help="Reference initial compute time in ms used for speedup annotations.",
    )
    parser.add_argument(
        "--baseline-frame-ms",
        type=float,
        default=54.41,
        help="Reference initial frame time in ms used for speedup annotations.",
    )
    parser.add_argument("--mpiexec", default="mpirun", help="MPI launcher executable.")
    parser.add_argument("--uv", default="uv", help="uv executable.")
    parser.add_argument("--uv-cache-dir", default=".uv-cache", help="uv cache dir.")
    parser.add_argument("--xdg-cache-home", default=".cache", help="XDG cache dir.")
    parser.add_argument("--mplconfigdir", default=".mplconfig", help="Matplotlib config dir.")
    parser.add_argument(
        "--csv",
        default="outputs/mpi_full_benchmark.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def run_one_case(args, process_count, thread_count):
    shell_cmd = (
        f"XDG_CACHE_HOME={args.xdg_cache_home} "
        f"MPLCONFIGDIR={args.mplconfigdir} "
        f"UV_CACHE_DIR={args.uv_cache_dir} "
        f"{args.uv} run python nbodies_mpi_full.py "
        f"--data {args.data} --dt {args.dt} "
        f"--grid {args.grid[0]} {args.grid[1]} {args.grid[2]} "
        f"--threads {thread_count} --warmup-steps {args.warmup_steps} --steps {args.steps} --json"
    )
    cmd = [args.mpiexec, "-n", str(process_count), "/bin/zsh", "-lc", shell_cmd]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No output returned for processes={process_count}, threads={thread_count}")
    result = json.loads(lines[-1])
    result["baseline_compute_ms"] = args.baseline_compute_ms
    result["baseline_frame_ms"] = args.baseline_frame_ms
    result["compute_speedup"] = args.baseline_compute_ms / result["compute_ms"]
    result["frame_speedup"] = args.baseline_frame_ms / result["frame_ms"]
    return result


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for process_count in args.processes:
        for thread_count in args.threads:
            result = run_one_case(args, process_count, thread_count)
            rows.append(result)
            print(
                f"processes={result['mpi_processes']} | compute_ranks={result['compute_ranks']} | "
                f"threads={result['threads_used']} | compute={result['compute_ms']:.3f} ms | "
                f"frame={result['frame_ms']:.3f} ms"
            )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mpi_processes",
                "compute_ranks",
                "threads_used",
                "compute_ms",
                "frame_ms",
                "baseline_compute_ms",
                "baseline_frame_ms",
                "compute_speedup",
                "frame_speedup",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV écrit dans {csv_path}")


if __name__ == "__main__":
    main()
