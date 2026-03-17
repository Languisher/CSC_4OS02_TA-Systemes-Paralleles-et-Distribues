import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot timings and speedups for the MPI split version."
    )
    parser.add_argument(
        "--csv",
        default="outputs/mpi_split_benchmark.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/mpi_split_benchmark.png",
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


def annotate_speedups(ax, bars, speedups):
    ymax = ax.get_ylim()[1]
    for bar, speedup in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02 * ymax,
            f"x{speedup:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    thread_labels = [str(int(row["threads_used"])) for row in rows]
    compute_ms = [float(row["compute_ms"]) for row in rows]
    frame_ms = [float(row["frame_ms"]) for row in rows]
    ref_compute_ms = compute_ms[0]
    ref_frame_ms = frame_ms[0]
    compute_speedup = [ref_compute_ms / value if value > 0.0 else 0.0 for value in compute_ms]
    frame_speedup = [ref_frame_ms / value if value > 0.0 else 0.0 for value in frame_ms]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    bars_compute = axes[0].bar(thread_labels, compute_ms, color="#2ca02c", alpha=0.9)
    axes[0].set_title(f"Calcul par pas (MPI split, ref 1 thread = {ref_compute_ms:.2f} ms)")
    axes[0].set_xlabel("Nombre de threads")
    axes[0].set_ylabel("Temps moyen par pas (ms)")
    axes[0].grid(axis="y", alpha=0.25)
    annotate_speedups(axes[0], bars_compute, compute_speedup)

    bars_frame = axes[1].bar(thread_labels, frame_ms, color="#d62728", alpha=0.9)
    axes[1].set_title(f"Total par frame (MPI split, ref 1 thread = {ref_frame_ms:.2f} ms)")
    axes[1].set_xlabel("Nombre de threads")
    axes[1].set_ylabel("Temps moyen par frame (ms)")
    axes[1].grid(axis="y", alpha=0.25)
    annotate_speedups(axes[1], bars_frame, frame_speedup)

    fig.suptitle("Séparation MPI de l'affichage et du calcul")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Figure écrite dans {output_path}")


if __name__ == "__main__":
    main()
