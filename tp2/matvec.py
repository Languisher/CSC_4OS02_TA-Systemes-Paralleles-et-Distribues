import argparse

import numpy as np
from mpi4py import MPI


def build_vector(dim: int) -> np.ndarray:
    return np.arange(1, dim + 1, dtype=np.float64)


def matvec_sequential(dim: int, u: np.ndarray) -> np.ndarray:
    rows = np.arange(dim, dtype=np.int64)[:, None]
    cols = np.arange(dim, dtype=np.int64)[None, :]
    a = ((rows + cols) % dim).astype(np.float64) + 1.0
    return a @ u


def local_block_bounds(dim: int, size: int, rank: int) -> tuple[int, int]:
    nloc = dim // size
    start = rank * nloc
    return start, start + nloc


def matvec_columns(comm: MPI.Comm, dim: int, u: np.ndarray) -> np.ndarray:
    rank = comm.Get_rank()
    size = comm.Get_size()
    start, end = local_block_bounds(dim, size, rank)

    cols = np.arange(start, end, dtype=np.int64)
    u_loc = u[start:end]
    rows = np.arange(dim, dtype=np.int64)[:, None]
    a_loc = ((rows + cols[None, :]) % dim).astype(np.float64) + 1.0
    v_partial = a_loc @ u_loc

    v = np.empty(dim, dtype=np.float64)
    comm.Allreduce(v_partial, v, op=MPI.SUM)
    return v


def matvec_rows(comm: MPI.Comm, dim: int, u: np.ndarray) -> np.ndarray:
    rank = comm.Get_rank()
    size = comm.Get_size()
    start, end = local_block_bounds(dim, size, rank)

    rows = np.arange(start, end, dtype=np.int64)[:, None]
    cols = np.arange(dim, dtype=np.int64)[None, :]
    a_loc = ((rows + cols) % dim).astype(np.float64) + 1.0
    v_loc = a_loc @ u

    counts = np.full(size, dim // size, dtype=np.int32)
    displs = np.arange(size, dtype=np.int32) * (dim // size)
    v = np.empty(dim, dtype=np.float64)
    comm.Allgatherv(v_loc, [v, counts, displs, MPI.DOUBLE])
    return v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPI matrix-vector product")
    parser.add_argument("--mode", choices=["seq", "cols", "rows"], default="seq")
    parser.add_argument("--dim", type=int, default=1200)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--speedup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.dim % size != 0:
        if rank == 0:
            raise ValueError("dim must be divisible by number of processes")
        return

    u = build_vector(args.dim)

    seq_time = None
    if args.speedup and rank == 0:
        t0 = MPI.Wtime()
        v_seq = matvec_sequential(args.dim, u)
        t1 = MPI.Wtime()
        seq_time = t1 - t0
    else:
        v_seq = None

    times = []
    v = None
    for _ in range(args.repeat):
        comm.Barrier()
        t0 = MPI.Wtime()
        if args.mode == "seq":
            if rank == 0:
                v = matvec_sequential(args.dim, u)
            else:
                v = np.empty(args.dim, dtype=np.float64)
            comm.Bcast(v, root=0)
        elif args.mode == "cols":
            v = matvec_columns(comm, args.dim, u)
        else:
            v = matvec_rows(comm, args.dim, u)
        comm.Barrier()
        t1 = MPI.Wtime()
        times.append(t1 - t0)

    t_avg = float(np.mean(times))

    if args.check:
        if rank == 0 and v_seq is None:
            v_seq = matvec_sequential(args.dim, u)
        if rank != 0:
            v_seq = np.empty(args.dim, dtype=np.float64)
        comm.Bcast(v_seq, root=0)
        ok = np.allclose(v, v_seq, rtol=1e-12, atol=1e-12)
    else:
        ok = True

    if rank == 0:
        print(f"mode={args.mode} processes={size} dim={args.dim} nloc={args.dim // size}")
        print(f"avg_time={t_avg:.6f} s")
        if args.speedup and seq_time is not None:
            print(f"seq_time={seq_time:.6f} s")
            print(f"speedup={seq_time / t_avg:.6f}")
        print(f"check={'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
