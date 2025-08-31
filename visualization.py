

# import block 

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import gaussian_kde


# load data
with open("output.pkl", "rb") as f:
    data = pickle.load(f)


expressed_over_time = data["expressed_over_time"]
variance_over_time = data["variance_over_time"]
disutility_over_time = data["disutility_over_time"]
expressed_snapshots = data["expressed_snapshots"]
swapless_cycles = data["swapless_cycles"]
snapshot_cycles = data["snapshot_cycles"]
final_cycle = data["final_cycle"]

print(expressed_snapshots.keys())


#########################################################
# HEATMAPS
#########################################################

cycles_to_plot = [0, 200, 400, 764]

# get values for color bar limits
all_vals_global = np.concatenate([np.array(s) for s in expressed_snapshots.values()])
vmin, vmax = all_vals_global.min(), all_vals_global.max()

# plot
fig, axs = plt.subplots(2, 2, figsize=(7, 7))
axs = axs.flatten()

for i, cycle in enumerate(cycles_to_plot):
    grid = np.array(expressed_snapshots[cycle]).reshape(30, 30)
    im = axs[i].imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[i].axis("off")
    axs[i].set_title(f"Cycle {cycle}", fontsize=12, fontweight='light', loc='center', pad=10)

plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.2)

# colorbar
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Expressed attitudes', rotation=270, labelpad=15, fontsize=10)


plt.show()


#########################################################
# POLARIZATION AND VARIANCE
#########################################################

# setup
percentiles = [5, 20, 40]
cycles = expressed_over_time.keys()
left_tail_means = {p: [] for p in percentiles}
right_tail_means = {p: [] for p in percentiles}

# collect tail means
for cycle in cycles:
    attitudes = np.array(expressed_over_time[cycle])
    sorted_attitudes = np.sort(attitudes)
    n = len(attitudes)

    for p in percentiles:
        k = int(np.floor(n * p / 100))
        left_avg = sorted_attitudes[:k].mean() if k > 0 else np.nan
        right_avg = sorted_attitudes[-k:].mean() if k > 0 else np.nan
        left_tail_means[p].append(left_avg)
        right_tail_means[p].append(right_avg)

# use stored variance
variance = [variance_over_time[cycle] for cycle in cycles]
scaled_variance = np.array(variance) * 1000  # rescale for ×10³

# plot layout
fig, axs = plt.subplots(1, 2, figsize=(5, 4), sharex=True)

# polarisation plot
for p in percentiles:
    axs[0].plot(cycles, left_tail_means[p], linestyle="dashed", color="black", linewidth=1)
    axs[0].plot(cycles, right_tail_means[p], linestyle="solid", color="black", linewidth=1)
axs[0].set_title("Polarisation", fontsize=10)
axs[0].set_xlabel("Cycle", fontsize=9)
axs[0].set_ylabel("Expressed Attitudes", fontsize=9)
axs[0].grid(True, linewidth=0.3)

# variance plot
axs[1].plot(cycles, scaled_variance, color="black", linewidth=1)
axs[1].set_title("Attitude Variance", fontsize=10)
axs[1].set_xlabel("Cycle", fontsize=9)
axs[1].set_ylabel("Variance ×10³", fontsize=9)
axs[1].grid(True, linewidth=0.3)

plt.tight_layout()
plt.show()


#########################################################
# EXPRESSED ATTITUDES DISTRIBUTION FIRST VS. LAST CYCLE
#########################################################


start_cycle = 0
end_cycle = max(expressed_over_time.keys())


start_vals = np.array(expressed_over_time[start_cycle])
end_vals = np.array(expressed_over_time[end_cycle])

# KDE for smooth curve
kde_start = gaussian_kde(start_vals)
kde_end = gaussian_kde(end_vals)
x_vals = np.linspace(0, 1, 100)

# plot
plt.figure(figsize=(4, 3))
plt.plot(x_vals, kde_start(x_vals), linestyle="solid", color="black", linewidth=1, label=f"Cycle {start_cycle}")
plt.plot(x_vals, kde_end(x_vals), linestyle="dashed", color="black", linewidth=1, label=f"Cycle {end_cycle}")

plt.xlabel("Attitudes", fontsize=8)
plt.ylabel("Density", fontsize=8)
plt.title("Distribution of expressed attitudes", fontsize=9, pad=18)
plt.text(0.5, 1.02, "w = 0.5", ha='center', va='bottom', fontsize=8, transform=plt.gca().transAxes)
plt.ylim(0, 10)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.show()


#########################################################
# NETWORK DISUTILITY OVER TIME
#########################################################

cycles = disutility_over_time.keys()
disutility = [disutility_over_time[cycle] for cycle in cycles]

# plot
plt.figure(figsize=(3, 4))
plt.plot(cycles, disutility, color="black", linewidth=1)
plt.title("Network disutility over time", fontsize=10)
plt.xlabel("Cycle", fontsize=9)
plt.ylabel("Total disutility", fontsize=9)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.show()


#########################################################
# CDF SWAPLESS CYCLES
#########################################################

cdf_values = np.arange(1, len(swapless_cycles) + 1)

plt.figure(figsize=(5, 4))
plt.plot(swapless_cycles, cdf_values, color="black", linewidth=1)
plt.title("CDF of swapless cycles", fontsize=11)
plt.xlabel("Cycle", fontsize=10)
plt.ylabel("Count", fontsize=10)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.show()