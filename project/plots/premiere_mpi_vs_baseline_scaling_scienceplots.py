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


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    df = pd.read_csv(path)
    required = [
        "workers",
        "mode",
        "run_total_ms",
        "ants_ms",
        "advance_time_ms",
        "evaporation_ms",
        "mpi_mark_sync_ms",
        "mpi_evap_sync_ms",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing column: {c}")
        if c != "mode":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=required).sort_values(["workers", "mode"])


def plot_compare(df: pd.DataFrame, out_prefix: Path) -> None:
    b = df[df["mode"] == "baseline"].sort_values("workers").reset_index(drop=True)
    m = df[df["mode"] == "premiere_mpi"].sort_values("workers").reset_index(drop=True)
    if b.empty or m.empty:
        raise ValueError("CSV must contain both mode=baseline and mode=premiere_mpi")

    workers = b["workers"].astype(int).tolist()
    b1 = float(b.loc[b["workers"] == 1, "run_total_ms"].iloc[0])
    b["speedup_vs_b1"] = b1 / b["run_total_ms"]
    m["speedup_vs_b1"] = b1 / m["run_total_ms"]
    m["relative_to_baseline"] = b["run_total_ms"].values / m["run_total_ms"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    ax = axes[0][0]
    ax.plot(workers, b["run_total_ms"], marker="o", linewidth=1.5, label="Baseline (OpenMP) run_total_ms")
    ax.plot(workers, m["run_total_ms"], marker="s", linewidth=1.5, label="Premiere facon (MPI) run_total_ms")
    ax.set_xlabel("Workers (threads or MPI ranks)")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Total Runtime")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0][1]
    ax.plot(workers, b["ants_ms"], marker="o", linewidth=1.5, label="Baseline ants_ms")
    ax.plot(workers, m["ants_ms"], marker="s", linewidth=1.5, label="Premiere facon ants_ms")
    ax.plot(workers, m["evaporation_ms"], marker="^", linewidth=1.2, label="Premiere facon evaporation_ms")
    ax.set_xlabel("Workers")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Compute Components")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][0]
    ax.plot(workers, b["speedup_vs_b1"], marker="o", linewidth=1.5, label="Baseline speedup vs baseline(1)")
    ax.plot(workers, m["speedup_vs_b1"], marker="s", linewidth=1.5, label="Premiere facon speedup vs baseline(1)")
    ax.plot(workers, m["relative_to_baseline"], marker="d", linewidth=1.5, label="Baseline / Premiere facon")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Workers")
    ax.set_ylabel("Speedup / Relative (x)")
    ax.set_title("Scaling Comparison")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][1]
    ax.plot(workers, m["mpi_mark_sync_ms"], marker="o", linewidth=1.5, label="mpi_mark_sync_ms")
    ax.plot(workers, m["mpi_evap_sync_ms"], marker="s", linewidth=1.5, label="mpi_evap_sync_ms")
    ax.plot(workers, m["advance_time_ms"], marker="^", linewidth=1.2, label="advance_time_ms")
    ax.set_xlabel("MPI Processes")
    ax.set_ylabel("Time (ms)")
    ax.set_title("MPI Synchronization Cost")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Premiere facon (MPI distributed-memory) vs Baseline OpenMP", y=1.01)
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline threads 1/2/4/8 vs premiere facon MPI 1/2/4/8.")
    parser.add_argument("--csv", type=Path, default=Path("../output/timing_cmp_premiere_vs_baseline_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("premiere_mpi_vs_baseline_scaling_scienceplots"))
    args = parser.parse_args()

    df = load_summary(args.csv)
    plot_compare(df, args.out)


if __name__ == "__main__":
    main()
