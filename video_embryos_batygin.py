import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

import rebound_sims as reb_sims
from helpers import plot_prettier
plot_prettier(dpi=500)

# === PARAMETERS ===
dataset_id = 2
sim_id = 138
t_units = 'kyr'
fps = 20
pause_frames = 10

# === LOAD SIM ===
base_dir = Path.cwd()
file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"
saved_sim = reb_sims.load_simulation_run(file_path)
sim_data, metadata = saved_sim

rock_names = list(sim_data.keys())
num_pl, num_em, num_ptsml = metadata['num_pl'], metadata['num_em'], metadata['num_ptsml']
ref_name = rock_names[num_pl - 1]
embryo_names = rock_names[num_pl:num_pl + num_em]

times = sim_data[rock_names[0]]['time']
time_factor = {'yr': 1, 'kyr': 1e3, 'Myr': 1e6}[t_units]
t_plot = (times / time_factor).to_numpy()

a_ref = sim_data[ref_name]["a"].to_numpy()

# === COMPUTE TRAJECTORIES ===
def compute_traj(name):
    tau_a_ref = -sim_data[ref_name]["tau_a"]
    tau_a_em = -sim_data[name]["tau_a"]
    tau_a = (1 / tau_a_ref - 1 / tau_a_em) ** -1
    Omega = 2 * np.pi / sim_data[name]["P"]
    log_tau_a_Omega = np.log10(tau_a * Omega).to_numpy()
    tau_e_ref = -sim_data[ref_name]["tau_e"]
    log_K2 = np.log10(tau_a / tau_e_ref).to_numpy()
    a_em = sim_data[name]["a"].to_numpy()
    return log_tau_a_Omega, log_K2, a_em

trajectories = {name: compute_traj(name) for name in embryo_names}

all_x = np.concatenate([np.asarray(v[0])[np.isfinite(v[0])] for v in trajectories.values()])
all_y = np.concatenate([np.asarray(v[1])[np.isfinite(v[1])] for v in trajectories.values()])
xpad = 0.05 * (all_x.max() - all_x.min())
ypad = 0.05 * (all_y.max() - all_y.min())

fig, ax = plt.subplots(figsize=(5, 4))
ax.set_xlim(all_x.min() - xpad, all_x.max() + xpad)
ax.set_ylim(all_y.min() - ypad, all_y.max() + ypad)
ax.set_xlabel(r"$\log_{10}(\tau_a \Omega)$")
ax.set_ylabel(r"$\log \mathcal{K}_2$")
ax.grid(True)

em_style = {'c': 'navy', 'alpha': 0.85, 'markersize': 6}
label_style = {'fontsize': 5, 'color': 'k', 'ha': 'left', 'va': 'bottom'}

points = {name: ax.plot([], [], 'o', **em_style)[0] for name in embryo_names}
labels = {
    name: ax.annotate(str(i), xy=(0, 0), xytext=(3, 3), textcoords='offset points', **label_style)
    for i, name in enumerate(embryo_names)
}
for lbl in labels.values():
    lbl.set_visible(False)

collided = {name: False for name in embryo_names}
crossed_a = {name: False for name in embryo_names}   # a_embryo > a_ref, permanent square
last_valid = {name: None for name in embryo_names}
title = ax.set_title("")

def update(frame_idx):
    for name in embryo_names:
        x_arr, y_arr, a_arr = trajectories[name]
        x, y, a_em = x_arr[frame_idx], y_arr[frame_idx], a_arr[frame_idx]

        if np.isfinite(x) and np.isfinite(y) and not crossed_a[name]:
            points[name].set_data([x], [y])
            labels[name].xy = (x, y)
            labels[name].set_visible(True)
            last_valid[name] = (x, y)

            if not crossed_a[name] and np.isfinite(a_em) and a_em > a_ref[frame_idx]:
                ax.plot(x, y, marker='s', color='darkorange', ms=6, mew=1,
                         mec='k', zorder=5)
                crossed_a[name] = True
        else:
            points[name].set_data([], [])
            labels[name].set_visible(False)
            if not collided[name] and last_valid[name] is not None and not crossed_a[name]:
                lx, ly = last_valid[name]
                ax.plot(lx, ly, marker='x', color='red', ms=7, mew=2, zorder=5)
                collided[name] = True
                if name == 'embryo 0' or name == 'embryo 1':
                    tau_a_ref = -sim_data['planet c']["tau_a"]
                    tau_a_em = -sim_data[name]["tau_a"]
                    tau_a = (1 / tau_a_ref - 1 / tau_a_em) ** -1
                    Omega = 2 * np.pi / sim_data[name]["P"]
                    log_tau_a_Omega = np.log10(tau_a * Omega).to_numpy()
                    tau_e_ref = -sim_data['planet c']["tau_e"]
                    log_K2 = np.log10(tau_a / tau_e_ref).to_numpy()
                    print("COLLISION", frame_idx, name, tau_a_ref[frame_idx], tau_a_em[frame_idx], tau_a[frame_idx], Omega[frame_idx], log_tau_a_Omega[frame_idx], tau_e_ref[frame_idx], log_K2[frame_idx])

    title.set_text(f"Dataset {dataset_id}, Sim {sim_id}, t = {t_plot[frame_idx]:.1f} {t_units}")
    return list(points.values()) + list(labels.values()) + [title]

frames = (
    [0] * pause_frames +
    list(range(len(t_plot))) +
    [len(t_plot) - 1] * pause_frames
)

ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True, blit=False)

ani.save(
    f"embryo_trajectories_dataset{dataset_id}_sim{sim_id}.mp4",
    writer=FFMpegWriter(fps=fps)
)

print("Animation saved.")