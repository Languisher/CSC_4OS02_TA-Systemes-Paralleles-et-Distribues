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


def load_speedup_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required = [
        "processes",
        "run_total_ms",
        "speedup",
        "advance_time_ms",
        "ants_ms",
        "evaporation_ms",
        "mpi_mark_sync_ms",
        "mpi_evap_sync_ms",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["processes", "run_total_ms", "speedup"]).sort_values("processes")
    return df


def load_baseline_avg(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "run" not in df.columns:
        raise ValueError(f"{path} missing required 'run' column")
    avg_rows = df[df["run"].astype(str) == "avg"]
    if avg_rows.empty:
        raise ValueError(f"{path} has no avg row")
    row = avg_rows.iloc[0]
    needed = ["run_total_ms", "advance_time_ms", "ants_ms", "evaporation_ms"]
    out = {}
    for c in needed:
        if c not in row.index:
            raise ValueError(f"{path} missing required column: {c}")
        out[c] = float(pd.to_numeric(row[c], errors="coerce"))
    return out


def build_compare_df(mpi_df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    records = [
        {
            "label": "baseline",
            "processes": 0,
            "run_total_ms": baseline["run_total_ms"],
            "advance_time_ms": baseline["advance_time_ms"],
            "ants_ms": baseline["ants_ms"],
            "evaporation_ms": baseline["evaporation_ms"],
            "mpi_mark_sync_ms": 0.0,
            "mpi_evap_sync_ms": 0.0,
            "speedup_vs_mpi1": None,
            "speedup_vs_baseline": 1.0,
        }
    ]
    mpi1 = float(mpi_df.loc[mpi_df["processes"] == 1, "run_total_ms"].iloc[0])
    for _, r in mpi_df.iterrows():
        records.append(
            {
                "label": f"MPI-{int(r['processes'])}",
                "processes": int(r["processes"]),
                "run_total_ms": float(r["run_total_ms"]),
                "advance_time_ms": float(r["advance_time_ms"]),
                "ants_ms": float(r["ants_ms"]),
                "evaporation_ms": float(r["evaporation_ms"]),
                "mpi_mark_sync_ms": float(r["mpi_mark_sync_ms"]),
                "mpi_evap_sync_ms": float(r["mpi_evap_sync_ms"]),
                "speedup_vs_mpi1": float(r["speedup"]),
                "speedup_vs_baseline": baseline["run_total_ms"] / float(r["run_total_ms"]),
            }
        )
    return pd.DataFrame(records)


def plot_mpi_scalability(df: pd.DataFrame, out_prefix: Path) -> None:
    labels = df["label"].tolist()
    x = range(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    ax = axes[0]
    colors = ["#C44E52"] + ["#4C72B0"] * (len(labels) - 1)
    bars = ax.bar(x, df["run_total_ms"], color=colors, alpha=0.9, label="run_total_ms")
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Baseline + MPI(1/2/4/8) Runtime")
    ax.grid(axis="y", alpha=0.35)
    for b, v in zip(bars, df["run_total_ms"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax2 = ax.twinx()
    mpi_x = [i for i, p in enumerate(df["processes"]) if p > 0]
    speedup_mpi = [df.iloc[i]["speedup_vs_mpi1"] for i in mpi_x]
    speedup_base = [df.iloc[i]["speedup_vs_baseline"] for i in range(len(df))]
    ax2.plot(mpi_x, speedup_mpi, marker="o", linewidth=1.5, color="#55A868", label="MPI speedup vs MPI-1")
    ax2.plot(list(x), speedup_base, marker="s", linewidth=1.5, color="#8172B3", label="speedup vs baseline")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_ylim(bottom=0.0)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    width = 0.75
    ax.bar(x, df["ants_ms"], width=width, label="ants_ms", color="#55A868")
    ax.bar(x, df["evaporation_ms"], width=width, bottom=df["ants_ms"], label="evaporation_ms", color="#8172B3")
    base_sync = df["ants_ms"] + df["evaporation_ms"]
    ax.bar(x, df["mpi_mark_sync_ms"], width=width, bottom=base_sync, label="mpi_mark_sync_ms", color="#C44E52")
    ax.bar(
        x,
        df["mpi_evap_sync_ms"],
        width=width,
        bottom=base_sync + df["mpi_mark_sync_ms"],
        label="mpi_evap_sync_ms",
        color="#CCB974",
    )
    ax.plot(list(x), df["advance_time_ms"], color="black", marker="s", linewidth=1.2, label="advance_time_ms")
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Stage Breakdown (Baseline + MPI)")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Premiere facon: Baseline + MPI(1/2/4/8)", y=1.02)
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline + MPI(1/2/4/8) scalability and speedup with scienceplots.")
    parser.add_argument("--speedup-csv", type=Path, default=Path("../output/timing_mpi_speedup_fair.csv"))
    parser.add_argument("--baseline-csv", type=Path, default=Path("../output/timing_vectorized_openmp.csv"))
    parser.add_argument("--out", type=Path, default=Path("mpi_scalability_scienceplots"))
    args = parser.parse_args()

    mpi_df = load_speedup_csv(args.speedup_csv)
    baseline = load_baseline_avg(args.baseline_csv)
    compare_df = build_compare_df(mpi_df, baseline)
    plot_mpi_scalability(compare_df, args.out)


if __name__ == "__main__":
    main()
