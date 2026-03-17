import argparse
import csv
import time
from pathlib import Path

import numpy as np
from nbodies_grid_numba_timing import NBodySystem, run_simulation

def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure initial render/update times and export them to CSV."
    )
    parser.add_argument(
        "--data",
        default="data/galaxy_1000",
        help="Input galaxy dataset.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Simulation time step.",
    )
    parser.add_argument(
        "--grid",
        nargs=3,
        type=int,
        metavar=("NI", "NJ", "NK"),
        default=(20, 20, 1),
        help="Cartesian grid dimensions.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Number of compute-only warmup steps before measuring.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of measured frames.",
    )
    parser.add_argument(
        "--csv",
        default="outputs/initial_times.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--mode",
        choices=("headless", "opengl"),
        default="headless",
        help="Measurement mode. 'headless' is robust in non-GUI environments.",
    )
    parser.add_argument(
        "--hidden-window",
        action="store_true",
        help="Create a hidden SDL window while still measuring rendering.",
    )
    parser.add_argument(
        "--disable-vsync",
        action="store_true",
        help="Disable VSync to avoid display refresh throttling in timings.",
    )
    return parser.parse_args()


def simulate_display_step(positions, colors, luminosities):
    frame_points = np.asarray(positions, dtype=np.float32).copy()
    frame_colors = (
        np.asarray(colors, dtype=np.float32)
        * np.asarray(luminosities, dtype=np.float32)[:, np.newaxis]
        / 255.0
    )
    return frame_points, frame_colors


def run_headless_measurement(args, rows):
    system = NBodySystem(args.data, ncells_per_dir=tuple(args.grid))
    intensities = np.clip(system.masses / system.max_mass, 0.5, 1.0).astype(np.float32)
    colors = np.asarray(system.colors, dtype=np.float32)

    for _ in range(args.warmup_steps):
        system.update_positions(args.dt)
        simulate_display_step(system.positions, colors, intensities)

    for frame in range(args.frames):
        render_start = time.perf_counter()
        simulate_display_step(system.positions, colors, intensities)
        render_end = time.perf_counter()

        update_start = time.perf_counter()
        system.update_positions(args.dt)
        frame_end = time.perf_counter()

        rows.append(
            {
                "frame": frame,
                "render_ms": (render_end - render_start) * 1000.0,
                "update_ms": (frame_end - update_start) * 1000.0,
                "total_ms": (frame_end - render_start) * 1000.0,
                "mode": args.mode,
            }
        )


def run_opengl_measurement(args, rows):
    system = NBodySystem(args.data, ncells_per_dir=tuple(args.grid))
    for _ in range(args.warmup_steps):
        system.update_positions(args.dt)

    def record_metrics(frame, render_ms, update_ms, total_ms):
        rows.append(
            {
                "frame": frame,
                "render_ms": render_ms,
                "update_ms": update_ms,
                "total_ms": total_ms,
                "mode": args.mode,
            }
        )

    run_simulation(
        args.data,
        ncells_per_dir=tuple(args.grid),
        dt=args.dt,
        max_frames=args.frames,
        metrics_callback=record_metrics,
        visible=not args.hidden_window,
        vsync=not args.disable_vsync,
    )


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.mode == "headless":
        run_headless_measurement(args, rows)
    else:
        run_opengl_measurement(args, rows)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "render_ms", "update_ms", "total_ms", "mode"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        raise RuntimeError("No timing samples were recorded.")

    mean_render = sum(row["render_ms"] for row in rows) / len(rows)
    mean_update = sum(row["update_ms"] for row in rows) / len(rows)
    mean_total = sum(row["total_ms"] for row in rows) / len(rows)

    print(f"CSV écrit dans {csv_path}")
    print(f"Temps moyen affichage : {mean_render:.3f} ms")
    print(f"Temps moyen calcul     : {mean_update:.3f} ms")
    print(f"Temps moyen total      : {mean_total:.3f} ms")


if __name__ == "__main__":
    main()
