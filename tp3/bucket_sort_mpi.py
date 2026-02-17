from __future__ import annotations

import argparse

import numpy as np
from mpi4py import MPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel bucket sort with MPI (root generates data)."
    )
    parser.add_argument(
        "-n",
        "--size",
        type=int,
        default=40,
        help="Total number of integers to sort.",
    )
    parser.add_argument(
        "--min-value",
        type=int,
        default=0,
        help="Minimum random integer value (inclusive).",
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=999,
        help="Maximum random integer value (inclusive).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by rank 0 to generate data.",
    )
    parser.add_argument(
        "--print-array",
        action="store_true",
        help="Print input and sorted arrays on rank 0.",
    )
    return parser.parse_args()


def split_counts(total: int, nprocs: int) -> np.ndarray:
    counts = np.full(nprocs, total // nprocs, dtype=np.int32)
    counts[: total % nprocs] += 1
    return counts


def displacements(counts: np.ndarray) -> np.ndarray:
    displs = np.zeros_like(counts)
    displs[1:] = np.cumsum(counts[:-1])
    return displs


def compute_bucket_ids(values: np.ndarray, min_value: int, max_value: int, nprocs: int) -> np.ndarray:
    value_range = max_value - min_value + 1
    bucket_width = max(1, (value_range + nprocs - 1) // nprocs)
    bucket_ids = (values - min_value) // bucket_width
    return np.minimum(bucket_ids, nprocs - 1).astype(np.int32, copy=False)


def main() -> None:
    args = parse_args()
    if args.size < 0:
        raise ValueError("--size must be non-negative")
    if args.max_value < args.min_value:
        raise ValueError("--max-value must be >= --min-value")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    local_counts = None
    send_displs = None
    global_data = None

    if rank == 0:
        # Rank 0 creates the global unsorted input.
        rng = np.random.default_rng(args.seed)
        global_data = rng.integers(
            args.min_value,
            args.max_value + 1,
            size=args.size,
            dtype=np.int32,
        )
        if args.print_array:
            print(f"Input array ({args.size} elements): {global_data.tolist()}")
        local_counts = split_counts(args.size, nprocs)
        send_displs = displacements(local_counts)

    local_n = np.array(0, dtype=np.int32)
    if rank == 0:
        local_n[...] = local_counts[rank]
    # Share how many elements each rank will receive.
    comm.Scatter(
        [local_counts, MPI.INT] if rank == 0 else None,
        [local_n, MPI.INT],
        root=0,
    )

    local_chunk = np.empty(int(local_n), dtype=np.int32)
    # Distribute the initial data chunks from rank 0.
    comm.Scatterv(
        [global_data, local_counts, send_displs, MPI.INT] if rank == 0 else None,
        [local_chunk, MPI.INT],
        root=0,
    )

    if local_chunk.size > 0:
        bucket_ids = compute_bucket_ids(local_chunk, args.min_value, args.max_value, nprocs)
        send_counts = np.bincount(bucket_ids, minlength=nprocs).astype(np.int32)
    else:
        send_counts = np.zeros(nprocs, dtype=np.int32)

    recv_counts = np.empty(nprocs, dtype=np.int32)
    comm.Alltoall([send_counts, MPI.INT], [recv_counts, MPI.INT])

    send_displs2 = displacements(send_counts)
    recv_displs2 = displacements(recv_counts)

    if local_chunk.size > 0:
        order = np.argsort(bucket_ids, kind="stable")
        send_buffer = local_chunk[order]
    else:
        send_buffer = np.empty(0, dtype=np.int32)

    recv_total = int(np.sum(recv_counts))
    recv_buffer = np.empty(recv_total, dtype=np.int32)

    # Exchange buckets so each rank owns one value interval.
    comm.Alltoallv(
        [send_buffer, send_counts, send_displs2, MPI.INT],
        [recv_buffer, recv_counts, recv_displs2, MPI.INT],
    )

    local_sorted = np.sort(recv_buffer, kind="quicksort")

    local_sorted_n = np.array(local_sorted.size, dtype=np.int32)
    gathered_counts = np.empty(nprocs, dtype=np.int32) if rank == 0 else None
    comm.Gather([local_sorted_n, MPI.INT], [gathered_counts, MPI.INT] if rank == 0 else None, root=0)

    if rank == 0:
        gathered_displs = displacements(gathered_counts)
        sorted_global = np.empty(int(np.sum(gathered_counts)), dtype=np.int32)
    else:
        gathered_displs = None
        sorted_global = None

    # Gather sorted intervals back to rank 0.
    comm.Gatherv(
        [local_sorted, MPI.INT],
        [sorted_global, gathered_counts, gathered_displs, MPI.INT] if rank == 0 else None,
        root=0,
    )

    if rank == 0:
        ok = bool(np.all(sorted_global[:-1] <= sorted_global[1:])) if sorted_global.size > 1 else True
        print(f"Sorted correctly: {ok}")
        if args.print_array:
            print(f"Sorted array ({sorted_global.size} elements): {sorted_global.tolist()}")


if __name__ == "__main__":
    main()
