import argparse
from dataclasses import dataclass
from math import log

import numpy as np
from mpi4py import MPI

try:
    import matplotlib.cm as mpl_cm
except ModuleNotFoundError:
    mpl_cm = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


WORK_TAG = 1
STOP_TAG = 2
RESULT_TAG = 3


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth: bool = False, clamp: bool = True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        if clamp:
            return max(0.0, min(value, 1.0))
        return value

    def count_iterations(self, c: complex, smooth: bool = False) -> int | float:
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1.0) * (c.real + 1.0) + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if -0.75 < c.real < 0.5:
            ct = c.real - 0.25 + 1j * c.imag
            ctnrm = abs(ct)
            if ctnrm < 0.5 * (1.0 - ct.real / max(ctnrm, 1e-14)):
                return self.max_iterations

        z = 0j
        for it in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return it + 1 - log(log(abs(z))) / log(2.0)
                return it
        return self.max_iterations


def compute_row(y: int, width: int, scale_x: float, scale_y: float, ms: MandelbrotSet) -> np.ndarray:
    row = np.empty(width, dtype=np.float64)
    imag = -1.125 + scale_y * y
    for x in range(width):
        c = complex(-2.0 + scale_x * x, imag)
        row[x] = ms.convergence(c, smooth=True)
    return row


def block_row_range(height: int, size: int, rank: int) -> tuple[int, int]:
    base = height // size
    rem = height % size
    start = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    return start, start + count


def static_balanced_rows(height: int) -> list[int]:
    rows = []
    for i in range((height + 1) // 2):
        rows.append(i)
        j = height - 1 - i
        if j != i:
            rows.append(j)
    return rows


def assemble_image(width: int, height: int, rows: list[int], values: list[np.ndarray]) -> np.ndarray:
    image = np.empty((height, width), dtype=np.float64)
    for y, row in zip(rows, values, strict=True):
        image[y, :] = row
    return image


def run_block(comm: MPI.Comm, width: int, height: int, ms: MandelbrotSet) -> np.ndarray | None:
    rank = comm.Get_rank()
    size = comm.Get_size()
    scale_x = 3.0 / width
    scale_y = 2.25 / height

    y0, y1 = block_row_range(height, size, rank)
    local_rows = list(range(y0, y1))
    local_values = [compute_row(y, width, scale_x, scale_y, ms) for y in local_rows]

    all_rows = comm.gather(local_rows, root=0)
    all_values = comm.gather(local_values, root=0)

    if rank != 0:
        return None

    rows = [y for part in all_rows for y in part]
    values = [row for part in all_values for row in part]
    return assemble_image(width, height, rows, values)


def run_static_balanced(comm: MPI.Comm, width: int, height: int, ms: MandelbrotSet) -> np.ndarray | None:
    rank = comm.Get_rank()
    size = comm.Get_size()
    scale_x = 3.0 / width
    scale_y = 2.25 / height

    rows_order = static_balanced_rows(height)
    local_rows = rows_order[rank::size]
    local_values = [compute_row(y, width, scale_x, scale_y, ms) for y in local_rows]

    all_rows = comm.gather(local_rows, root=0)
    all_values = comm.gather(local_values, root=0)

    if rank != 0:
        return None

    rows = [y for part in all_rows for y in part]
    values = [row for part in all_values for row in part]
    return assemble_image(width, height, rows, values)


def run_master_worker(comm: MPI.Comm, width: int, height: int, ms: MandelbrotSet) -> np.ndarray | None:
    rank = comm.Get_rank()
    size = comm.Get_size()
    scale_x = 3.0 / width
    scale_y = 2.25 / height

    if size == 1:
        rows = list(range(height))
        values = [compute_row(y, width, scale_x, scale_y, ms) for y in rows]
        return assemble_image(width, height, rows, values)

    if rank == 0:
        image = np.empty((height, width), dtype=np.float64)
        next_row = 0
        active = 0

        for worker in range(1, size):
            if next_row < height:
                comm.send(next_row, dest=worker, tag=WORK_TAG)
                next_row += 1
                active += 1
            else:
                comm.send(-1, dest=worker, tag=STOP_TAG)

        status = MPI.Status()
        while active > 0:
            y, row = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
            worker = status.Get_source()
            image[y, :] = row

            if next_row < height:
                comm.send(next_row, dest=worker, tag=WORK_TAG)
                next_row += 1
            else:
                comm.send(-1, dest=worker, tag=STOP_TAG)
                active -= 1
        return image

    status = MPI.Status()
    while True:
        y = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == STOP_TAG:
            break
        row = compute_row(y, width, scale_x, scale_y, ms)
        comm.send((y, row), dest=0, tag=RESULT_TAG)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPI Mandelbrot")
    parser.add_argument("--mode", choices=["block", "static", "master"], default="block")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--iter", type=int, default=200)
    parser.add_argument("--escape", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ms = MandelbrotSet(max_iterations=args.iter, escape_radius=args.escape)

    comm.Barrier()
    t0 = MPI.Wtime()
    if args.mode == "block":
        convergence = run_block(comm, args.width, args.height, ms)
    elif args.mode == "static":
        convergence = run_static_balanced(comm, args.width, args.height, ms)
    else:
        convergence = run_master_worker(comm, args.width, args.height, ms)
    comm.Barrier()
    t1 = MPI.Wtime()

    if rank == 0:
        out = args.output or f"mandelbrot_{args.mode}_p{comm.Get_size()}.png"
        t_img0 = MPI.Wtime()
        if Image is not None and mpl_cm is not None:
            image = Image.fromarray(np.uint8(mpl_cm.plasma(convergence) * 255))
            image.save(out)
            saved = True
        else:
            saved = False
        t_img1 = MPI.Wtime()

        print(f"mode={args.mode} processes={comm.Get_size()} width={args.width} height={args.height}")
        print(f"compute_time={t1 - t0:.6f} s")
        print(f"image_time={t_img1 - t_img0:.6f} s")
        if saved:
            print(f"output={out}")
        else:
            print("output=skipped (missing matplotlib or Pillow)")


if __name__ == "__main__":
    main()
