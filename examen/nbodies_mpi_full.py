import argparse
import json
import time

import numpy as np
from mpi4py import MPI
from numba import get_num_threads, njit, prange, set_num_threads

G = 1.560339e-13


def generate_star_color(mass: float) -> tuple[int, int, int]:
    if mass > 5.0:
        return (150, 180, 255)
    elif mass > 2.0:
        return (255, 255, 255)
    elif mass >= 1.0:
        return (255, 255, 200)
    else:
        return (255, 150, 100)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full MPI parallelization of the trajectory computation with rank 0 dedicated to display."
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
    parser.add_argument("--threads", type=int, default=1, help="Numba threads for compute ranks.")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps before timing.")
    parser.add_argument("--steps", type=int, default=10, help="Timed steps.")
    parser.add_argument("--json", action="store_true", help="Print a JSON result on rank 0.")
    return parser.parse_args()


def load_dataset(filename):
    positions = []
    velocities = []
    masses = []
    max_mass = 0.0
    box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)
    with open(filename, "r", encoding="utf-8") as handle:
        for line in handle:
            data = line.split()
            if not data:
                continue
            masses.append(float(data[0]))
            positions.append([float(data[1]), float(data[2]), float(data[3])])
            velocities.append([float(data[4]), float(data[5]), float(data[6])])
            max_mass = max(max_mass, masses[-1])
            for i in range(3):
                box[0][i] = min(box[0][i], positions[-1][i] - 1.0e-6)
                box[1][i] = max(box[1][i], positions[-1][i] + 1.0e-6)

    positions = np.array(positions, dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)
    masses = np.array(masses, dtype=np.float32)
    colors = np.array([generate_star_color(m) for m in masses], dtype=np.float32)
    intensities = np.clip(masses / max_mass, 0.5, 1.0).astype(np.float32)
    ids = np.arange(positions.shape[0], dtype=np.int64)
    return ids, positions, velocities, masses, colors, intensities, box


def simulate_display_step(positions, colors, luminosities):
    frame_points = np.asarray(positions, dtype=np.float32).copy()
    frame_colors = (
        np.asarray(colors, dtype=np.float32)
        * np.asarray(luminosities, dtype=np.float32)[:, np.newaxis]
        / 255.0
    )
    return frame_points, frame_colors


def partition_axis(n_cells_x, n_compute_ranks):
    counts = np.full(n_compute_ranks, n_cells_x // n_compute_ranks, dtype=np.int64)
    counts[: n_cells_x % n_compute_ranks] += 1
    starts = np.zeros(n_compute_ranks, dtype=np.int64)
    starts[1:] = np.cumsum(counts[:-1])
    ends = starts + counts
    return starts, ends


def compute_cell_indices_np(positions, grid_min, cell_size, n_cells):
    idx = np.floor((positions - grid_min) / cell_size).astype(np.int64)
    np.clip(idx, 0, n_cells - 1, out=idx)
    return idx


def owner_of_bodies(cell_x_indices, cell_partition_ends):
    return np.searchsorted(cell_partition_ends, cell_x_indices, side="right")


def compute_global_cell_stats(positions, masses, grid_min, cell_size, n_cells):
    cell_indices = compute_cell_indices_np(positions, grid_min, cell_size, n_cells)
    morse = (
        cell_indices[:, 0]
        + cell_indices[:, 1] * n_cells[0]
        + cell_indices[:, 2] * n_cells[0] * n_cells[1]
    )
    n_total = int(np.prod(n_cells))
    cell_masses = np.bincount(morse, weights=masses, minlength=n_total).astype(np.float32)
    weighted_x = np.bincount(morse, weights=masses * positions[:, 0], minlength=n_total).astype(np.float32)
    weighted_y = np.bincount(morse, weights=masses * positions[:, 1], minlength=n_total).astype(np.float32)
    weighted_z = np.bincount(morse, weights=masses * positions[:, 2], minlength=n_total).astype(np.float32)
    cell_com = np.zeros((n_total, 3), dtype=np.float32)
    valid = cell_masses > 0.0
    cell_com[valid, 0] = weighted_x[valid] / cell_masses[valid]
    cell_com[valid, 1] = weighted_y[valid] / cell_masses[valid]
    cell_com[valid, 2] = weighted_z[valid] / cell_masses[valid]
    return cell_indices, cell_masses, cell_com


def build_extended_structure(all_ids, all_positions, all_masses, all_cell_indices, ghost_start, ghost_end, n_cells):
    mask = (all_cell_indices[:, 0] >= ghost_start) & (all_cell_indices[:, 0] < ghost_end)
    ext_ids = all_ids[mask]
    ext_positions = all_positions[mask]
    ext_masses = all_masses[mask]
    ext_indices = all_cell_indices[mask].copy()
    ext_nx = ghost_end - ghost_start
    ext_indices[:, 0] -= ghost_start
    local_morse = (
        ext_indices[:, 0]
        + ext_indices[:, 1] * ext_nx
        + ext_indices[:, 2] * ext_nx * n_cells[1]
    )
    order = np.argsort(local_morse, kind="stable")
    ext_ids = np.ascontiguousarray(ext_ids[order], dtype=np.int64)
    ext_positions = np.ascontiguousarray(ext_positions[order], dtype=np.float32)
    ext_masses = np.ascontiguousarray(ext_masses[order], dtype=np.float32)
    local_morse = np.ascontiguousarray(local_morse[order], dtype=np.int64)
    counts = np.bincount(local_morse, minlength=int(ext_nx * n_cells[1] * n_cells[2]))
    starts = np.empty(len(counts) + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.cumsum(counts)
    return ext_ids, ext_positions, ext_masses, starts, ext_nx


def allgather_1d(comm, local_array, mpi_type):
    counts = np.array(comm.allgather(local_array.shape[0]), dtype=np.int32)
    displs = np.zeros_like(counts)
    if len(counts) > 1:
        displs[1:] = np.cumsum(counts[:-1])
    total = int(np.sum(counts))
    gathered = np.empty(total, dtype=local_array.dtype)
    comm.Allgatherv(local_array, [gathered, counts, displs, mpi_type])
    return gathered


def allgather_2d(comm, local_array, mpi_type):
    counts = np.array(comm.allgather(local_array.shape[0]), dtype=np.int32)
    displs = np.zeros_like(counts)
    if len(counts) > 1:
        displs[1:] = np.cumsum(counts[:-1])
    total = int(np.sum(counts))
    width = local_array.shape[1]
    gathered = np.empty((total, width), dtype=local_array.dtype)
    comm.Allgatherv(local_array, [gathered, counts * width, displs * width, mpi_type])
    return gathered


@njit(parallel=True)
def compute_acceleration_targets(
    target_positions: np.ndarray,
    target_ids: np.ndarray,
    ext_positions: np.ndarray,
    ext_masses: np.ndarray,
    ext_ids: np.ndarray,
    ext_cell_starts: np.ndarray,
    global_cell_masses: np.ndarray,
    global_cell_com: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
    ghost_start: int,
    ext_nx: int,
):
    n_targets = target_positions.shape[0]
    accelerations = np.zeros((n_targets, 3), dtype=np.float32)
    for itarget in prange(n_targets):
        pos = target_positions[itarget]
        target_id = target_ids[itarget]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    global_morse = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    if (abs(ix - cell_idx[0]) <= 2) and (abs(iy - cell_idx[1]) <= 2) and (abs(iz - cell_idx[2]) <= 2):
                        local_x = ix - ghost_start
                        if 0 <= local_x < ext_nx:
                            local_morse = local_x + iy * ext_nx + iz * ext_nx * n_cells[1]
                            start_idx = ext_cell_starts[local_morse]
                            end_idx = ext_cell_starts[local_morse + 1]
                            for j in range(start_idx, end_idx):
                                if ext_ids[j] != target_id:
                                    direction = ext_positions[j] - pos
                                    distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                                    if distance > 1.0e-10:
                                        inv_dist3 = 1.0 / (distance ** 3)
                                        accelerations[itarget, :] += G * direction[:] * inv_dist3 * ext_masses[j]
                    else:
                        cell_mass = global_cell_masses[global_morse]
                        if cell_mass > 0.0:
                            direction = global_cell_com[global_morse] - pos
                            distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                accelerations[itarget, :] += G * direction[:] * inv_dist3 * cell_mass
    return accelerations


def run_full_parallel_measurement(args):
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size < 2:
        raise RuntimeError("nbodies_mpi_full.py requires at least 2 MPI processes.")

    is_display_rank = world_rank == 0
    compute_comm = comm.Split(color=0 if is_display_rank else 1, key=world_rank)

    if world_rank == 0:
        ids, positions, velocities, masses, colors, intensities, box = load_dataset(args.data)
        initial_state = {
            "ids": ids,
            "positions": positions,
            "velocities": velocities,
            "masses": masses,
            "colors": colors,
            "intensities": intensities,
            "box": box,
        }
    else:
        initial_state = None

    initial_state = comm.bcast(initial_state, root=0)

    ids = initial_state["ids"]
    all_masses_by_id = initial_state["masses"]
    colors = initial_state["colors"]
    intensities = initial_state["intensities"]
    grid_min = initial_state["box"][0].astype(np.float32)
    grid_max = initial_state["box"][1].astype(np.float32)
    n_cells = np.array(args.grid, dtype=np.int64)
    cell_size = ((grid_max - grid_min) / n_cells).astype(np.float32)
    n_bodies = ids.shape[0]

    n_compute_ranks = world_size - 1
    partition_starts, partition_ends = partition_axis(n_cells[0], n_compute_ranks)

    if is_display_rank:
        frame_positions = np.empty((n_bodies, 3), dtype=np.float32)
        compute_samples = []
        frame_samples = []

        for _ in range(args.warmup_steps):
            comm.Recv([frame_positions, MPI.FLOAT], source=1, tag=10)
            _ = comm.recv(source=1, tag=11)
            simulate_display_step(frame_positions, colors, intensities)

        for _ in range(args.steps):
            frame_start = time.perf_counter()
            comm.Recv([frame_positions, MPI.FLOAT], source=1, tag=20)
            compute_ms = comm.recv(source=1, tag=21)
            simulate_display_step(frame_positions, colors, intensities)
            frame_end = time.perf_counter()
            compute_samples.append(compute_ms)
            frame_samples.append((frame_end - frame_start) * 1000.0)

        return {
            "mpi_processes": world_size,
            "compute_ranks": n_compute_ranks,
            "threads_used": None,
            "compute_ms": sum(compute_samples) / len(compute_samples),
            "frame_ms": sum(frame_samples) / len(frame_samples),
        }

    compute_rank = compute_comm.Get_rank()
    set_num_threads(args.threads)

    initial_cell_indices = compute_cell_indices_np(initial_state["positions"], grid_min, cell_size, n_cells)
    initial_owners = owner_of_bodies(initial_cell_indices[:, 0], partition_ends)
    owned_mask = initial_owners == compute_rank
    local_ids = np.ascontiguousarray(ids[owned_mask], dtype=np.int64)
    local_positions = np.ascontiguousarray(initial_state["positions"][owned_mask], dtype=np.float32)
    local_velocities = np.ascontiguousarray(initial_state["velocities"][owned_mask], dtype=np.float32)
    local_masses = np.ascontiguousarray(all_masses_by_id[local_ids], dtype=np.float32)

    for step_idx in range(args.warmup_steps + args.steps):
        step_start = time.perf_counter()

        all_ids = allgather_1d(compute_comm, local_ids, MPI.LONG)
        all_positions = allgather_2d(compute_comm, local_positions, MPI.FLOAT)
        all_velocities = allgather_2d(compute_comm, local_velocities, MPI.FLOAT)
        all_cell_indices, global_cell_masses, global_cell_com = compute_global_cell_stats(
            all_positions,
            all_masses_by_id[all_ids],
            grid_min,
            cell_size,
            n_cells,
        )

        owners = owner_of_bodies(all_cell_indices[:, 0], partition_ends)
        current_owned_mask = owners == compute_rank
        local_ids = np.ascontiguousarray(all_ids[current_owned_mask], dtype=np.int64)
        local_positions = np.ascontiguousarray(all_positions[current_owned_mask], dtype=np.float32)
        local_velocities = np.ascontiguousarray(all_velocities[current_owned_mask], dtype=np.float32)
        local_masses = np.ascontiguousarray(all_masses_by_id[local_ids], dtype=np.float32)

        ghost_start = max(0, int(partition_starts[compute_rank]) - 2)
        ghost_end = min(int(n_cells[0]), int(partition_ends[compute_rank]) + 2)
        ext_ids, ext_positions, ext_masses, ext_cell_starts, ext_nx = build_extended_structure(
            all_ids,
            all_positions,
            all_masses_by_id[all_ids],
            all_cell_indices,
            ghost_start,
            ghost_end,
            n_cells,
        )

        a1 = compute_acceleration_targets(
            local_positions,
            local_ids,
            ext_positions,
            ext_masses,
            ext_ids,
            ext_cell_starts,
            global_cell_masses,
            global_cell_com,
            grid_min,
            cell_size,
            n_cells,
            ghost_start,
            ext_nx,
        )

        predicted_positions = np.ascontiguousarray(
            local_positions + local_velocities * args.dt + 0.5 * a1 * args.dt * args.dt,
            dtype=np.float32,
        )

        pred_all_ids = allgather_1d(compute_comm, local_ids, MPI.LONG)
        pred_all_positions = allgather_2d(compute_comm, predicted_positions, MPI.FLOAT)
        pred_cell_indices, pred_global_cell_masses, pred_global_cell_com = compute_global_cell_stats(
            pred_all_positions,
            all_masses_by_id[pred_all_ids],
            grid_min,
            cell_size,
            n_cells,
        )
        pred_ext_ids, pred_ext_positions, pred_ext_masses, pred_ext_cell_starts, pred_ext_nx = build_extended_structure(
            pred_all_ids,
            pred_all_positions,
            all_masses_by_id[pred_all_ids],
            pred_cell_indices,
            ghost_start,
            ghost_end,
            n_cells,
        )

        a2 = compute_acceleration_targets(
            predicted_positions,
            local_ids,
            pred_ext_positions,
            pred_ext_masses,
            pred_ext_ids,
            pred_ext_cell_starts,
            pred_global_cell_masses,
            pred_global_cell_com,
            grid_min,
            cell_size,
            n_cells,
            ghost_start,
            pred_ext_nx,
        )

        local_positions = predicted_positions
        local_velocities = np.ascontiguousarray(
            local_velocities + 0.5 * (a1 + a2) * args.dt,
            dtype=np.float32,
        )

        local_compute_ms = (time.perf_counter() - step_start) * 1000.0
        step_compute_ms = compute_comm.allreduce(local_compute_ms, op=MPI.MAX)

        if compute_rank == 0:
            positions_by_id = np.empty((n_bodies, 3), dtype=np.float32)
            positions_by_id[pred_all_ids] = pred_all_positions
            if step_idx < args.warmup_steps:
                comm.Send([positions_by_id, MPI.FLOAT], dest=0, tag=10)
                comm.send(step_compute_ms, dest=0, tag=11)
            else:
                comm.Send([positions_by_id, MPI.FLOAT], dest=0, tag=20)
                comm.send(step_compute_ms, dest=0, tag=21)

    return None


def main():
    args = parse_args()
    result = run_full_parallel_measurement(args)
    if MPI.COMM_WORLD.Get_rank() == 0:
        result["threads_used"] = args.threads
        if args.json:
            print(json.dumps(result))
        else:
            print(
                f"processes={result['mpi_processes']} | threads={result['threads_used']} | "
                f"compute={result['compute_ms']:.3f} ms | frame={result['frame_ms']:.3f} ms"
            )


if __name__ == "__main__":
    main()
