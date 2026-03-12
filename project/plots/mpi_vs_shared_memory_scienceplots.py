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


def load_mpi_scaling(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing MPI scaling CSV: {path}")
    df = pd.read_csv(path)
    required = ["processes", "run_total_ms", "speedup"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=required).sort_values("processes")


def load_shared_memory_baseline(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline CSV: {path}")
    df = pd.read_csv(path)
    if "run" not in df.columns:
        raise ValueError(f"{path} missing 'run' column")
    avg = df[df["run"].astype(str) == "avg"]
    if avg.empty:
        raise ValueError(f"{path} has no avg row")
    run_total = pd.to_numeric(avg.iloc[0]["run_total_ms"], errors="coerce")
    if pd.isna(run_total):
        raise ValueError(f"{path} avg run_total_ms is not numeric")
    return float(run_total)


def plot_compare(mpi_df: pd.DataFrame, baseline_ms: float, out_prefix: Path) -> None:
    procs = mpi_df["processes"].astype(int).tolist()
    x = range(len(procs))

    mpi_df = mpi_df.copy()
    mpi_df["relative_to_shared"] = baseline_ms / mpi_df["run_total_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    ax = axes[0]
    bars = ax.bar(x, mpi_df["run_total_ms"], color="#4C72B0", alpha=0.9, label="MPI runtime")
    ax.axhline(baseline_ms, color="#C44E52", linestyle="--", linewidth=1.2, label="Shared-memory baseline")
    ax.set_xticks(list(x), procs)
    ax.set_xlabel("MPI Processes")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Runtime: MPI vs Shared-Memory")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    for b, v in zip(bars, mpi_df["run_total_ms"]):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    ax.plot(procs, mpi_df["speedup"], marker="o", linewidth=1.5, color="#55A868", label="MPI speedup vs MPI(1)")
    ax.plot(
        procs,
        mpi_df["relative_to_shared"],
        marker="s",
        linewidth=1.5,
        color="#8172B3",
        label="Shared-memory / MPI",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("MPI Processes")
    ax.set_ylabel("Speedup / Relative Performance (x)")
    ax.set_title("Scaling and Relative Performance")
    ax.set_xticks(procs)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("MPI(1/2/4/8) vs Baseline Shared-Memory (Vectorized+OpenMP)", y=1.02)
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MPI 1/2/4/8 against shared-memory baseline with scienceplots.")
    parser.add_argument("--mpi-csv", type=Path, default=Path("../output/timing_mpi_speedup_fair.csv"))
    parser.add_argument("--baseline-csv", type=Path, default=Path("../output/timing_vectorized_openmp.csv"))
    parser.add_argument("--out", type=Path, default=Path("mpi_vs_shared_memory_scienceplots"))
    args = parser.parse_args()

    mpi_df = load_mpi_scaling(args.mpi_csv)
    baseline_ms = load_shared_memory_baseline(args.baseline_csv)
    plot_compare(mpi_df, baseline_ms, args.out)


if __name__ == "__main__":
    main()
