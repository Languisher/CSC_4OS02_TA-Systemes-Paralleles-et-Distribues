import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot numba-parallel timings and speedups as a function of the thread count."
    )
    parser.add_argument(
        "--csv",
        default="outputs/numba_parallel_benchmark.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/numba_parallel_benchmark.png",
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


def annotate_frame_breakdown(ax, bars, compute_ms, render_ms, total_ms):
    for bar, compute, render, total in zip(bars, compute_ms, render_ms, total_ms):
        compute_ratio = 100.0 * compute / total if total > 0.0 else 0.0
        render_ratio = 100.0 * render / total if total > 0.0 else 0.0
        x = bar.get_x() + bar.get_width() / 2.0
        ax.text(
            x,
            compute / 2.0,
            f"{compute_ratio:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
        if render > 0.02:
            ax.text(
                x,
                compute + render / 2.0,
                f"{render_ratio:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="#1f1f1f",
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
    render_ms = [max(total - compute, 0.0) for compute, total in zip(compute_ms, frame_ms)]
    numba1_compute = compute_ms[0]
    numba1_frame = frame_ms[0]
    compute_speedup = [numba1_compute / value if value > 0.0 else 0.0 for value in compute_ms]
    frame_speedup = [numba1_frame / value if value > 0.0 else 0.0 for value in frame_ms]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    bars_compute = axes[0].bar(thread_labels, compute_ms, color="#1f77b4", alpha=0.9)
    axes[0].set_title(f"Calcul par pas (ref numba 1 thread = {numba1_compute:.2f} ms)")
    axes[0].set_xlabel("Nombre de threads")
    axes[0].set_ylabel("Temps moyen par pas (ms)")
    axes[0].grid(axis="y", alpha=0.25)
    annotate_speedups(axes[0], bars_compute, compute_speedup)

    bars_compute_part = axes[1].bar(
        thread_labels,
        compute_ms,
        color="#dd8452",
        alpha=0.95,
        label="Calcul",
    )
    bars_render_part = axes[1].bar(
        thread_labels,
        render_ms,
        bottom=compute_ms,
        color="#55a868",
        alpha=0.95,
        label="Affichage",
    )
    axes[1].set_title(f"Total par frame (ref numba 1 thread = {numba1_frame:.2f} ms)")
    axes[1].set_xlabel("Nombre de threads")
    axes[1].set_ylabel("Temps moyen par frame (ms)")
    axes[1].grid(axis="y", alpha=0.25)
    annotate_speedups(axes[1], bars_compute_part, frame_speedup)
    annotate_frame_breakdown(axes[1], bars_compute_part, compute_ms, render_ms, frame_ms)
    axes[1].legend()

    fig.suptitle("Performances du code numba parallèle selon le nombre de threads")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Figure écrite dans {output_path}")


if __name__ == "__main__":
    main()
