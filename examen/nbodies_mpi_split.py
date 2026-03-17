# Étape "Séparation de l'affichage et du calcul"
# Processus 0 : affichage / frame
# Processus 1 : calcul des trajectoires
import argparse
import time

import numpy as np
from mpi4py import MPI
from numba import get_num_threads, set_num_threads

from nbodies_grid_numba_parallel import NBodySystem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split display and compute across two MPI processes."
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
    parser.add_argument("--threads", type=int, default=1, help="Numba threads on rank 1.")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps before timing.")
    parser.add_argument("--steps", type=int, default=10, help="Timed steps.")
    return parser.parse_args()


def simulate_display_step(positions, colors, luminosities):
    frame_points = np.asarray(positions, dtype=np.float32).copy()
    frame_colors = (
        np.asarray(colors, dtype=np.float32)
        * np.asarray(luminosities, dtype=np.float32)[:, np.newaxis]
        / 255.0
    )
    return frame_points, frame_colors


def run_split_measurement(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        raise RuntimeError("nbodies_mpi_split.py requires exactly 2 MPI processes.")

    if rank == 1:
        set_num_threads(args.threads)
        system = NBodySystem(args.data, ncells_per_dir=tuple(args.grid))
        positions = np.ascontiguousarray(system.positions.astype(np.float32))
        meta = {
            "n_bodies": positions.shape[0],
            "colors": np.asarray(system.colors, dtype=np.float32),
            "intensities": np.clip(system.masses / system.max_mass, 0.5, 1.0).astype(np.float32),
            "threads_used": get_num_threads(),
        }
        comm.send(meta, dest=0, tag=0)

        for _ in range(args.warmup_steps):
            system.update_positions(args.dt)
            positions[:] = system.positions
            comm.Send([positions, MPI.FLOAT], dest=0, tag=1)

        compute_samples = []
        for _ in range(args.steps):
            compute_start = time.perf_counter()
            system.update_positions(args.dt)
            compute_end = time.perf_counter()
            positions[:] = system.positions
            comm.Send([positions, MPI.FLOAT], dest=0, tag=2)
            compute_samples.append((compute_end - compute_start) * 1000.0)

        comm.send(
            {
                "threads_used": get_num_threads(),
                "compute_ms": sum(compute_samples) / len(compute_samples),
            },
            dest=0,
            tag=3,
        )
        return None

    meta = comm.recv(source=1, tag=0)
    recv_positions = np.empty((meta["n_bodies"], 3), dtype=np.float32)

    for _ in range(args.warmup_steps):
        comm.Recv([recv_positions, MPI.FLOAT], source=1, tag=1)
        simulate_display_step(recv_positions, meta["colors"], meta["intensities"])

    frame_samples = []
    for _ in range(args.steps):
        frame_start = time.perf_counter()
        comm.Recv([recv_positions, MPI.FLOAT], source=1, tag=2)
        simulate_display_step(recv_positions, meta["colors"], meta["intensities"])
        frame_end = time.perf_counter()
        frame_samples.append((frame_end - frame_start) * 1000.0)

    compute_result = comm.recv(source=1, tag=3)
    return {
        "threads_used": compute_result["threads_used"],
        "compute_ms": compute_result["compute_ms"],
        "frame_ms": sum(frame_samples) / len(frame_samples),
    }


def main():
    args = parse_args()
    result = run_split_measurement(args)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(
            f"threads={result['threads_used']} | "
            f"compute={result['compute_ms']:.3f} ms | frame={result['frame_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
