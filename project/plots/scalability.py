import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    plt.style.use("default")

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
    }
)

df = pd.read_csv("../output/timing_scalability.csv")
df = df[df["run"] == "avg"].copy()
df["nb_ants"] = pd.to_numeric(df["nb_ants"], errors="coerce")
df = df.sort_values("nb_ants")

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

# 1) Overall stage time (ms)
ax = axes[0, 0]
ax.plot(df["nb_ants"], df["ants_ms"], marker="o", label="Ant updates")
ax.plot(df["nb_ants"], df["evaporation_ms"], marker="s", label="Evaporation")
ax.plot(df["nb_ants"], df["pheromone_update_ms"], marker="^", label="Pheromone update")
ax.plot(df["nb_ants"], df["run_total_ms"], marker="d", label="Run total")
ax.set_title("Scalability: Stage Time")
ax.set_xlabel("Number of ants")
ax.set_ylabel("Time (ms)")
ax.grid(True)
ax.legend(frameon=False, fontsize=8)

# 2) Ant internal split (ms)
ax = axes[0, 1]
ax.plot(df["nb_ants"], df["ant_select_move_ms"], marker="o", label="Select/move")
ax.plot(df["nb_ants"], df["ant_terrain_cost_ms"], marker="s", label="Terrain cost")
ax.plot(df["nb_ants"], df["ant_mark_pheromone_ms"], marker="^", label="mark_pheronome")
ax.set_title("Ant::advance Internal Breakdown")
ax.set_xlabel("Number of ants")
ax.set_ylabel("Time (ms)")
ax.grid(True)
ax.legend(frameon=False, fontsize=8)

# 3) Normalized ant-step cost (ns)
ax = axes[1, 0]
ax.plot(df["nb_ants"], df["ant_select_ns_per_move"], marker="o", label="Select/move ns per move")
ax.plot(df["nb_ants"], df["ant_terrain_ns_per_move"], marker="s", label="Terrain ns per move")
ax.plot(df["nb_ants"], df["ant_mark_ns_per_move"], marker="^", label="Mark ns per move")
ax.set_title("Normalized Cost Per Ant Move")
ax.set_xlabel("Number of ants")
ax.set_ylabel("ns / move")
ax.grid(True)
ax.legend(frameon=False, fontsize=8)

# 4) Other normalized metrics + outcome
ax = axes[1, 1]
ax.plot(df["nb_ants"], df["ants_us_per_ant_call"] * 1000.0, marker="o", label="Ant updates ns per ant-call")
ax.plot(df["nb_ants"], df["evaporation_ns_per_cell"], marker="s", label="Evaporation ns per cell")
ax.plot(df["nb_ants"], df["update_us_per_iter"] * 1000.0, marker="^", label="Update ns per iter")
ax2 = ax.twinx()
ax2.plot(df["nb_ants"], df["delivered_food"], color="black", linestyle=":", marker="d", label="Delivered food")
ax.set_title("Normalized Metrics and Delivered Food")
ax.set_xlabel("Number of ants")
ax.set_ylabel("Time (ns)")
ax2.set_ylabel("Delivered food")
ax.grid(True)

lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, fontsize=8, loc="upper left")

fig.suptitle("Ant Simulation Scalability (from timing_scalability.csv)", y=0.995)
plt.tight_layout()
plt.savefig("scalability_runtime.png")
plt.savefig("scalability_runtime.pdf")
plt.close(fig)
