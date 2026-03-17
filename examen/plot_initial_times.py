import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot render/update timings from a CSV file."
    )
    parser.add_argument(
        "--csv",
        default="outputs/initial_times.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/initial_times.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No timing rows found in {csv_path}")
    return rows


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    frames = [int(row["frame"]) for row in rows]
    render_ms = [float(row["render_ms"]) for row in rows]
    update_ms = [float(row["update_ms"]) for row in rows]

    mean_render = sum(render_ms) / len(render_ms)
    mean_update = sum(update_ms) / len(update_ms)

    plt.figure(figsize=(10, 6))
    plt.plot(frames, render_ms, marker="o", linewidth=1.5, label="Affichage (render)")
    plt.plot(frames, update_ms, marker="s", linewidth=1.5, label="Calcul (update)")
    plt.axhline(mean_render, color="#1f77b4", linestyle="--", linewidth=1, label=f"Moyenne affichage: {mean_render:.2f} ms")
    plt.axhline(mean_update, color="#ff7f0e", linestyle="--", linewidth=1, label=f"Moyenne calcul: {mean_update:.2f} ms")
    plt.xlabel("Frame")
    plt.ylabel("Temps (ms)")
    plt.title("Temps initiaux par frame")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Figure écrite dans {output_path}")


if __name__ == "__main__":
    main()
