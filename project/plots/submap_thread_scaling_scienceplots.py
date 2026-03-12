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
        "threads",
        "mode",
        "run_total_ms",
        "ants_ms",
        "submap_reassign_ms",
        "border_crossings",
        "border_marks_exchanged",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing column: {c}")
    for c in ["threads", "run_total_ms", "ants_ms", "submap_reassign_ms", "border_crossings", "border_marks_exchanged"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    return df.sort_values(["threads", "mode"])


def plot(df: pd.DataFrame, out_prefix: Path) -> None:
    b = df[df["mode"] == "baseline"].sort_values("threads").reset_index(drop=True)
    s = df[df["mode"] == "submap"].sort_values("threads").reset_index(drop=True)
    if b.empty or s.empty:
        raise ValueError("Need both modes in CSV: baseline and submap")

    threads = b["threads"].astype(int).tolist()
    b1 = float(b.loc[b["threads"] == 1, "run_total_ms"].iloc[0])
    b_speedup = b1 / b["run_total_ms"]
    s_speedup = b1 / s["run_total_ms"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    ax = axes[0][0]
    ax.plot(threads, b["run_total_ms"], marker="o", linewidth=1.5, label="baseline run_total_ms")
    ax.plot(threads, s["run_total_ms"], marker="s", linewidth=1.5, label="submap run_total_ms")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Total Runtime vs Threads")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0][1]
    ax.plot(threads, b["ants_ms"], marker="o", linewidth=1.5, label="baseline ants_ms")
    ax.plot(threads, s["ants_ms"], marker="s", linewidth=1.5, label="submap ants_ms")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Ant Update Time (ms)")
    ax.set_title("Ant Kernel Time vs Threads")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][0]
    ax.plot(threads, b_speedup, marker="o", linewidth=1.5, label="baseline speedup (vs baseline@1)")
    ax.plot(threads, s_speedup, marker="s", linewidth=1.5, label="submap speedup (vs baseline@1)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup (x)")
    ax.set_title("Speedup Comparison")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][1]
    ax.plot(threads, s["submap_reassign_ms"], marker="o", linewidth=1.5, label="submap_reassign_ms")
    ax.plot(threads, s["border_crossings"], marker="s", linewidth=1.5, label="border_crossings")
    ax.plot(threads, s["border_marks_exchanged"], marker="^", linewidth=1.5, label="border_marks_exchanged")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Cost / Count")
    ax.set_title("Border-Exchange Metrics (Submap)")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Baseline vs Seconde facon (Submap) Thread Scaling", y=1.01)
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline vs submap thread scaling from timing_cmp_summary.csv.")
    parser.add_argument("--csv", type=Path, default=Path("../output/timing_cmp_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("submap_thread_scaling_scienceplots"))
    args = parser.parse_args()

    df = load_summary(args.csv)
    plot(df, args.out)


if __name__ == "__main__":
    main()
