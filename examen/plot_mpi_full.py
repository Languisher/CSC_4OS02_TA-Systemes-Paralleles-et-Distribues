import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the full MPI benchmark as two heatmaps."
    )
    parser.add_argument(
        "--csv",
        default="outputs/mpi_full_benchmark.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/mpi_full_benchmark.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No benchmark rows found in {csv_path}")
    return rows


def build_matrix(rows, value_key, process_values, thread_values):
    matrix = np.zeros((len(process_values), len(thread_values)), dtype=np.float32)
    for row in rows:
        i = process_values.index(int(row["mpi_processes"]))
        j = thread_values.index(int(row["threads_used"]))
        matrix[i, j] = float(row[value_key])
    return matrix


def annotate_heatmap(ax, data, fmt):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", color="black", fontweight="bold")


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    process_values = sorted({int(row["mpi_processes"]) for row in rows})
    thread_values = sorted({int(row["threads_used"]) for row in rows})
    baseline_compute_ms = float(rows[0]["baseline_compute_ms"])
    baseline_frame_ms = float(rows[0]["baseline_frame_ms"])

    compute_speedup = build_matrix(rows, "compute_speedup", process_values, thread_values)
    frame_speedup = build_matrix(rows, "frame_speedup", process_values, thread_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(compute_speedup, cmap="YlGnBu", aspect="auto")
    axes[0].set_title(f"Accélération calcul (baseline {baseline_compute_ms:.2f} ms)")
    axes[0].set_xlabel("Threads numba")
    axes[0].set_ylabel("Processus MPI totaux")
    axes[0].set_xticks(range(len(thread_values)), [str(v) for v in thread_values])
    axes[0].set_yticks(range(len(process_values)), [str(v) for v in process_values])
    annotate_heatmap(axes[0], compute_speedup, "x{:.2f}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(frame_speedup, cmap="OrRd", aspect="auto")
    axes[1].set_title(f"Accélération frame (baseline {baseline_frame_ms:.2f} ms)")
    axes[1].set_xlabel("Threads numba")
    axes[1].set_ylabel("Processus MPI totaux")
    axes[1].set_xticks(range(len(thread_values)), [str(v) for v in thread_values])
    axes[1].set_yticks(range(len(process_values)), [str(v) for v in process_values])
    annotate_heatmap(axes[1], frame_speedup, "x{:.2f}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Parallélisation MPI complète du calcul des trajectoires")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Figure écrite dans {output_path}")


if __name__ == "__main__":
    main()
