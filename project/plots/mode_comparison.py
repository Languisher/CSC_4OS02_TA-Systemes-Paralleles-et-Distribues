import argparse
import os
from pathlib import Path

_MPL_CONFIG = Path(__file__).resolve().parent / ".mplconfig"
_MPL_CONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    plt.style.use("default")


def _load_mode_stats(csv_path: Path, mode_name: str) -> dict:
    df = pd.read_csv(csv_path)
    if "run" not in df.columns:
        raise ValueError(f"{csv_path} is missing required 'run' column")

    run_rows = df[df["run"].astype(str) != "avg"].copy()
    numeric_cols = [c for c in run_rows.columns if c != "run"]
    for c in numeric_cols:
        run_rows[c] = pd.to_numeric(run_rows[c], errors="coerce")

    if len(run_rows) > 0:
        # Median is more robust to occasional outlier runs.
        row = run_rows[numeric_cols].median(numeric_only=True)
    else:
        avg_rows = df[df["run"].astype(str) == "avg"].copy()
        if len(avg_rows) == 0:
            raise ValueError(f"{csv_path} has neither run rows nor avg row")
        row = avg_rows.iloc[0]
        for c in avg_rows.columns:
            if c != "run":
                row[c] = pd.to_numeric(row[c], errors="coerce")

    return {
        "mode": mode_name,
        "run_total_ms": float(row["run_total_ms"]),
        "ants_ms": float(row["ants_ms"]),
        "evaporation_ms": float(row["evaporation_ms"]),
        "pheromone_update_ms": float(row["pheromone_update_ms"]),
        "ant_select_ns_per_move": float(row["ant_select_ns_per_move"]),
        "ant_terrain_ns_per_move": float(row["ant_terrain_ns_per_move"]),
        "ant_mark_ns_per_move": float(row["ant_mark_ns_per_move"]),
        "threads": int(row["threads"]) if "threads" in row else 1,
    }


def make_figure(nonvec: Path, vec: Path, vec_omp: Path, output_prefix: Path) -> None:
    records = [
        _load_mode_stats(nonvec, "Non-vectorized"),
        _load_mode_stats(vec, "Vectorized"),
        _load_mode_stats(vec_omp, "Vectorized + Shared Memory"),
    ]
    plot_df = pd.DataFrame(records)

    baseline_total = plot_df.loc[plot_df["mode"] == "Non-vectorized", "run_total_ms"].iloc[0]
    baseline_ants = plot_df.loc[plot_df["mode"] == "Non-vectorized", "ants_ms"].iloc[0]
    plot_df["speedup_total"] = baseline_total / plot_df["run_total_ms"]
    plot_df["speedup_ants"] = baseline_ants / plot_df["ants_ms"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    modes = plot_df["mode"].tolist()
    x = range(len(modes))

    ax = axes[0, 0]
    bars = ax.bar(x, plot_df["run_total_ms"], color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title("Run Total Time")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(list(x), modes, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.35)
    for b, v in zip(bars, plot_df["run_total_ms"]):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    width = 0.24
    ax.bar([i - width for i in x], plot_df["ants_ms"], width=width, label="ants_ms")
    ax.bar(x, plot_df["evaporation_ms"], width=width, label="evaporation_ms")
    ax.bar([i + width for i in x], plot_df["pheromone_update_ms"], width=width, label="pheromone_update_ms")
    ax.set_title("Core Stage Breakdown")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(list(x), modes, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    width = 0.24
    ax.bar([i - width for i in x], plot_df["ant_select_ns_per_move"], width=width, label="select")
    ax.bar(x, plot_df["ant_terrain_ns_per_move"], width=width, label="terrain")
    ax.bar([i + width for i in x], plot_df["ant_mark_ns_per_move"], width=width, label="mark")
    ax.set_title("Normalized Per-Move Cost")
    ax.set_ylabel("ns / move")
    ax.set_xticks(list(x), modes, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    width = 0.3
    bars1 = ax.bar([i - width / 2 for i in x], plot_df["speedup_total"], width=width, label="Total speedup")
    bars2 = ax.bar([i + width / 2 for i in x], plot_df["speedup_ants"], width=width, label="Ant-step speedup")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Speedup vs Non-vectorized")
    ax.set_ylabel("x speedup")
    ax.set_xticks(list(x), modes, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    for b, v in zip(bars1, plot_df["speedup_total"]):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for b, v in zip(bars2, plot_df["speedup_ants"]):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Mode Comparison", y=0.99)
    fig.tight_layout()

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot non-vectorized/vectorized/OpenMP comparison with scienceplots.")
    parser.add_argument("--nonvec", type=Path, default=Path("../output/timing_non_vectorized.csv"))
    parser.add_argument("--vec", type=Path, default=Path("../output/timing_vectorized.csv"))
    parser.add_argument("--vec-omp", type=Path, default=Path("../output/timing_vectorized_openmp.csv"))
    parser.add_argument("--out", type=Path, default=Path("mode_comparison_scienceplots"))
    args = parser.parse_args()

    make_figure(args.nonvec, args.vec, args.vec_omp, args.out)


if __name__ == "__main__":
    main()
