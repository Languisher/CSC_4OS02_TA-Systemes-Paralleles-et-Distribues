"""Monte-Carlo approximation of pi (sequential + MPI).

Examples:
  uv run python compute_pi.py --samples 40000000
  mpiexec -n 4 uv run python compute_pi.py --samples 40000000 --mpi
"""

from __future__ import annotations

import argparse
import time
import numpy as np

try:
    from mpi4py import MPI  # type: ignore
except Exception:
    MPI = None


def count_hits(nb_samples: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    x = 2.0 * rng.random(nb_samples) - 1.0
    y = 2.0 * rng.random(nb_samples) - 1.0
    return int(np.add.reduce((x * x + y * y) < 1.0, 0))


def run_sequential(nb_samples: int) -> tuple[float, float]:
    start = time.time()
    hits = count_hits(nb_samples, seed=1234567)
    elapsed = time.time() - start
    return 4.0 * hits / nb_samples, elapsed


def run_mpi(nb_samples: int) -> tuple[float, float, int]:
    if MPI is None:
        raise RuntimeError("mpi4py is not available in this environment.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    base = nb_samples // size
    rem = nb_samples % size
    local_samples = base + (1 if rank < rem else 0)

    comm.Barrier()
    start = time.time()
    local_hits = count_hits(local_samples, seed=1234567 + 97 * rank)
    total_hits = comm.reduce(local_hits, op=MPI.SUM, root=0)
    comm.Barrier()
    elapsed = time.time() - start

    if rank == 0:
        return 4.0 * total_hits / nb_samples, elapsed, size
    return 0.0, elapsed, size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=40_000_000)
    parser.add_argument("--mpi", action="store_true")
    args = parser.parse_args()

    if args.mpi:
        pi_val, elapsed, size = run_mpi(args.samples)
        if MPI is not None and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"[MPI] procs={size} time={elapsed:.6f} s pi={pi_val:.12f}")
    else:
        pi_val, elapsed = run_sequential(args.samples)
        print(f"[SEQ] time={elapsed:.6f} s pi={pi_val:.12f}")


if __name__ == "__main__":
    main()
