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


def load_avg(path: Path, name: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    if "run" not in df.columns:
        raise ValueError(f"{path} missing 'run' column")
    avg = df[df["run"].astype(str) == "avg"]
    row = avg.iloc[0] if not avg.empty else df.iloc[-1]
    fields = [
        "run_total_ms",
        "advance_time_ms",
        "ants_ms",
        "evaporation_ms",
        "pheromone_update_ms",
        "submap_reassign_ms",
        "submap_border_crossings",
        "submap_border_marks_exchanged",
    ]
    out = {"mode": name}
    for f in fields:
        out[f] = float(pd.to_numeric(row.get(f, 0.0), errors="coerce"))
    return out


def plot_compare(base: dict, submap: dict, out_prefix: Path) -> None:
    df = pd.DataFrame([base, submap])
    baseline = float(df.loc[df["mode"] == base["mode"], "run_total_ms"].iloc[0])
    df["speedup_vs_base"] = baseline / df["run_total_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    x = range(len(df))
    modes = df["mode"].tolist()

    ax = axes[0]
    width = 0.22
    ax.bar([i - width for i in x], df["ants_ms"], width=width, label="ants_ms")
    ax.bar(x, df["evaporation_ms"], width=width, label="evaporation_ms")
    ax.bar([i + width for i in x], df["pheromone_update_ms"], width=width, label="pheromone_update_ms")
    ax.plot(list(x), df["run_total_ms"], marker="o", linewidth=1.5, color="black", label="run_total_ms")
    ax.set_xticks(list(x), modes)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Second Amelioration Runtime Breakdown")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    bars = ax.bar(list(x), df["speedup_vs_base"], color=["#4C72B0", "#55A868"])
    ax2 = ax.twinx()
    ax2.plot(list(x), df["submap_reassign_ms"], marker="s", color="#C44E52", linewidth=1.5, label="submap_reassign_ms")
    ax2.plot(
        list(x),
        df["submap_border_crossings"],
        marker="^",
        color="#8172B3",
        linewidth=1.5,
        label="submap_border_crossings",
    )
    ax2.plot(
        list(x),
        df["submap_border_marks_exchanged"],
        marker="D",
        color="#64B5CD",
        linewidth=1.5,
        label="submap_border_marks_exchanged",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(list(x), modes)
    ax.set_ylabel("Speedup vs Vec+OpenMP (x)")
    ax2.set_ylabel("Reassign ms / Border crossings")
    ax.set_title("Speedup and Border-Exchange Cost")
    ax.grid(axis="y", alpha=0.35)
    for b, v in zip(bars, df["speedup_vs_base"]):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("Vectorized+OpenMP Baseline vs Seconde Façon (Sub-map Ownership)", y=1.02)
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot second amelioration (sub-map ownership) against vec+openmp baseline.")
    parser.add_argument("--base-csv", type=Path, default=Path("../output/timing_vectorized_openmp.csv"))
    parser.add_argument("--submap-csv", type=Path, default=Path("../output/timing_vectorized_openmp_submap.csv"))
    parser.add_argument("--out", type=Path, default=Path("second_amelioration_submap_scienceplots"))
    args = parser.parse_args()

    base = load_avg(args.base_csv, "Vec+OpenMP")
    submap = load_avg(args.submap_csv, "Seconde facon")
    plot_compare(base, submap, args.out)


if __name__ == "__main__":
    main()
