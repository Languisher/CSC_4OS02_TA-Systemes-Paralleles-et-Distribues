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


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required = [
        "nb_ants",
        "run_total_ms",
        "speedup_vs_first",
        "advance_time_ms",
        "ants_ms",
        "evaporation_ms",
        "mpi_mark_sync_ms",
        "mpi_evap_sync_ms",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=required).sort_values("nb_ants")


def plot_compute_scaling(df: pd.DataFrame, out_prefix: Path) -> None:
    x = df["nb_ants"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    ax = axes[0]
    ax.plot(x, df["run_total_ms"], marker="o", linewidth=1.5, label="run_total_ms")
    ax.plot(x, df["advance_time_ms"], marker="s", linewidth=1.5, label="advance_time_ms")
    ax.plot(x, df["ants_ms"], marker="^", linewidth=1.5, label="ants_ms")
    ax.plot(x, df["evaporation_ms"], marker="D", linewidth=1.5, label="evaporation_ms")
    ax.set_title("MPI Compute-Size Scaling")
    ax.set_xlabel("Number of ants")
    ax.set_ylabel("Time (ms)")
    ax.grid(axis="both", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(x, df["speedup_vs_first"], marker="o", linewidth=1.5, color="#55A868", label="speedup_vs_first")
    ax.plot(x, df["mpi_mark_sync_ms"], marker="s", linewidth=1.5, color="#C44E52", label="mpi_mark_sync_ms")
    ax.plot(x, df["mpi_evap_sync_ms"], marker="^", linewidth=1.5, color="#8172B3", label="mpi_evap_sync_ms")
    ax.set_title("Relative Speedup and Sync Cost")
    ax.set_xlabel("Number of ants")
    ax.set_ylabel("Speedup / Time (ms)")
    ax.grid(axis="both", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("MPI Compute-Dimension Scalability (fixed process count)", y=1.02)
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MPI compute-size scalability with scienceplots.")
    parser.add_argument("--csv", type=Path, default=Path("../output/timing_mpi_compute_scalability.csv"))
    parser.add_argument("--out", type=Path, default=Path("mpi_compute_scalability_scienceplots"))
    args = parser.parse_args()

    df = load_csv(args.csv)
    plot_compute_scaling(df, args.out)


if __name__ == "__main__":
    main()
