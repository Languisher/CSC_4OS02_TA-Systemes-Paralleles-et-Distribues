import argparse
import csv
from pathlib import Path

from mpi4py import MPI

from nbodies_mpi_split import run_split_measurement


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the MPI split version as a function of the numba thread count."
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
        help="Thread counts to benchmark on rank 1.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps before timing.")
    parser.add_argument("--steps", type=int, default=10, help="Timed steps per thread count.")
    parser.add_argument(
        "--csv",
        default="outputs/mpi_split_benchmark.csv",
        help="CSV output path. Rank 0 writes it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    results = []
    for thread_count in args.threads:
        single_run_args = argparse.Namespace(
            data=args.data,
            dt=args.dt,
            grid=tuple(args.grid),
            threads=thread_count,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
        )
        result = run_split_measurement(single_run_args)
        if rank == 0:
            result["threads_requested"] = thread_count
            results.append(result)
            print(
                f"threads={result['threads_used']} | "
                f"compute={result['compute_ms']:.3f} ms | frame={result['frame_ms']:.3f} ms"
            )
        comm.Barrier()

    if rank == 0:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "threads_requested",
                    "threads_used",
                    "compute_ms",
                    "frame_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV écrit dans {csv_path}")


if __name__ == "__main__":
    main()
