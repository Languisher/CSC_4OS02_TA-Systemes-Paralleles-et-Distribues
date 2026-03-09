import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Optional: use SciencePlots style when available
try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    plt.style.use("default")

# Global scientific-looking defaults (work even without scienceplots)
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.8,
    }
)

# 读取 CSV
df = pd.read_csv("../output/timing_results.csv")

# 使用 avg 行
avg = df[df["run"] == "avg"].iloc[0]

# 按 workflow 顺序排列
labels = [
    "SDL init",
    "Land generation",
    "Land normalization",
    "Ant initialization",
    "Pheromone initialization",
    "Ant updates",
    "Evaporation",
    "Pheromone update"
]

times = [
    avg["sdl_init_ms"],
    avg["land_generation_ms"],
    avg["land_normalization_ms"],
    avg["ant_init_ms"],
    avg["pheromone_init_ms"],
    avg["ants_ms"],
    avg["evaporation_ms"],
    avg["pheromone_update_ms"]
]

plt.figure(figsize=(10,5), dpi=300)

ax = plt.gca()
bars = ax.bar(
    labels,
    times,
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.8,
    alpha=0.9,
)

# 在柱顶标注时间
for bar, t in zip(bars, times):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + max(times) * 0.01,
        f"{t:.2f} ms",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax.set_ylabel("Execution Time (ms)")
ax.set_title("Execution Time of Simulation Workflow")
ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
ax.grid(axis="y")
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.grid(axis="y", which="minor", linestyle=":", alpha=0.22)
ax.margins(y=0.12)

plt.tight_layout()

plt.savefig("workflow_runtime.pdf")
plt.show()
