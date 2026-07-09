import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.animation import FuncAnimation, FFMpegWriter

from astropy import units as u
from pathlib import Path
from collections import defaultdict
from fractions import Fraction

import mmr_id
import rebound_sims as reb_sims

import warnings
warnings.filterwarnings('ignore')

from helpers import plot_prettier
plot_prettier(dpi=500)

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm)    
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# Get planet outcomes for all sims in one dataset
def classify_resonance(P_ratio, librates):
    """Classify first-order resonances of simulations up to 7:6. 
    If planets are not librating, the result will be "not in resonance".
    If planets are in higher-order or higher-index resonance, the result
    will be "other".


    Args:
        P_ratio (float): (Assigned) period ratio.
        librates (bool): Whether resonant angle librates.

    Returns:
        Classification of the resonance.
    """    


    if not librates or np.isnan(P_ratio):
        return "not in resonance"

    resonances = {
        "2:1": 2/1,
        "3:2": 3/2,
        "4:3": 4/3,
        "5:4": 5/4,
        "6:5": 6/5,
        "7:6": 7/6
    }

    for name, ratio in resonances.items():
        if P_ratio == ratio:
            return name

    return "other"

def planet_embryo(dataset_id, num_sims, pomega='mixed', amp_threshold=90, snapshot=-1):
    rows = []
    base_dir = Path.cwd()
    for sim_id in range(num_sims):
        file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"
        
        try:
            saved_sim = reb_sims.load_simulation_run(file_path)
        except FileNotFoundError:
            continue

        sim_data, metadata = saved_sim

        b = 'embryo 0'
        c = 'planet b'
            
        # Get P_ratios by direct division
        if not np.isnan(sim_data[b]['P'].iloc[snapshot]):
            true_inner_P_ratio = sim_data[c]['P'].iloc[snapshot]/sim_data[b]['P'].iloc[snapshot]
            print(true_inner_P_ratio)
            # Swap planets if inner became outer
            if true_inner_P_ratio < 1:
                b, c = c, b
                true_inner_P_ratio = 1/true_inner_P_ratio
                
            # Check Delta (find p, q such that Delta is closest to 0)
            inner_best = mmr_id.find_best_twoBR_pq(metadata['m_star'], sim_data[b], sim_data[c], snapshot=snapshot)
            inner_P_ratio = inner_best[0]/inner_best[1]
            inner_Delta = true_inner_P_ratio/inner_P_ratio - 1
            
            if inner_Delta > 3: # percent threshold
                inner_P_ratio = true_inner_P_ratio
            
            # Check libration of 2BR angle
            inner_librates = (mmr_id.check_resonance(metadata['m_star'], sim_data[b], sim_data[c], pomega=pomega, amp_threshold=amp_threshold) != (0,0))
            
        # Get tau_a_Omega and K2
        m_pl = metadata['m_vals'][0]
        m_em = metadata['m_em'] * m_earth
        a1 = 0.35
        P1 = a1**(3/2) # for M = 1
        P2 = P1#*true_inner_P_ratio
        a2 = P2**(2/3)
        
        Sigma1 = metadata['Sigma_1au']*(AU**2 / Msun) * a1**-metadata['alpha']
        h1 = metadata['h_1au'] * a1**metadata['beta']
        Sigma2 = metadata['Sigma_1au']*(AU**2 / Msun) * a2**-metadata['alpha']
        h2 = metadata['h_1au'] * a2**metadata['beta']

        tau_a1 = 1/(2.7+1.1*metadata['alpha']) / m_em / (Sigma1 * a1**2) * h1**2 / (2*np.pi / P1) 
        tau_a2 = 1/(2.7+1.1*metadata['alpha']) / m_pl / (Sigma2 * a2**2) * h2**2 / (2*np.pi / P2) 
        tau_a = (tau_a2**(-1)-tau_a1**(-1))**(-1)
        
        log_tau_Omega = np.log10(tau_a * 2*np.pi / (a1**(3/2))) 
        
        chi_a = 1 / (2.7 + 1.1 * metadata['alpha'])
        chi_e = 1 / 0.780
        tau_e2 = chi_e/chi_a * h2**2 * tau_a2
        log_K2 = np.log10(tau_a / tau_e2)

        # Create df to store data
        Sigma_1au = metadata['Sigma_1au']
        h_1au = metadata['h_1au']
        
        rows.append({
            "sim_id": sim_id,
            "Sigma_1au": Sigma_1au,
            "h_1au": h_1au,
            "inner_P_ratio": inner_P_ratio,
            "true_inner_P_ratio": round(true_inner_P_ratio, 3),
            "inner_P_Delta (%)": round(inner_Delta*100, 1),
            "inner_librates": inner_librates,
            "inner_res_class": classify_resonance(inner_P_ratio, inner_librates),
            "tau_a (kyr)": tau_a/1000,
            "log_tau_Omega": round(log_tau_Omega, 2), # prevent errors in plotting
            "log_K2": round(log_K2, 2)
        })
        
    outcomes = pd.DataFrame(rows)
    return outcomes

# Precompute snapshots
dataset_id = 10
snapshots = np.arange(0, 1000, 5).astype(int)
# for snapshot in np.arange(0, 1000, 5).astype(int):
#     planet_outcomes = planet_embryo(dataset_id=dataset_id, num_sims=400, pomega='mixed', amp_threshold=90, snapshot=snapshot)
#     planet_outcomes.to_hdf(f"dfs/planet_outcomes{dataset_id}_{snapshot}.h5", key="df", mode="w") # Save to disk
    
# Plotting code
def plot_param_grid_map(outcomes, value_col, label, x_col="Sigma_1au", y_col="h_1au", cmap='viridis', vmin=None, vmax=None, exp=False, log_cmap=False, show_text=True, show_libration=False, show_Delta=False, show_box=False, black_threshold=0.5, h_cut=None, ax=None, create_colorbar=True):
    # Bin first
    if x_col == "log_tau_Omega":

        outcomes = outcomes.copy()

        bins = np.linspace(
            outcomes[x_col].min(),
            outcomes[x_col].max(),
            25
        )

        outcomes["log_tau_Omega_bin"] = pd.cut(
            outcomes[x_col],
            bins=bins,
            labels=False,
            include_lowest=True
        )

        x_labels = (
            outcomes
            .groupby("log_tau_Omega_bin", observed=True)["log_tau_Omega"]
            .mean()
            .values
        )

        x_col = "log_tau_Omega_bin"


    # Now compute unique grid values AFTER binning
    x_vals = np.sort(outcomes[x_col].dropna().unique())
    y_vals = np.sort(outcomes[y_col].unique())

    if h_cut:
        y_vals = y_vals[y_vals <= h_cut]
    
    # Grid
    grid = (
        outcomes
        .pivot_table(
            index=x_col,
            columns=y_col,
            values=value_col,
            aggfunc=lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan
        )
        .reindex(index=x_vals,
                 columns=y_vals)
    )

    values = grid.values

    # Libration and Delta grid
    if value_col == "inner_P_ratio":
        if show_libration:
            libration_grid = (
                outcomes
                .pivot_table(
                    index=x_col,
                    columns=y_col,
                    values="inner_librates",
                    aggfunc="mean"
                )
                .reindex(index=x_vals, columns=y_vals)
            )
        else:
            libration_grid = None
        if show_Delta:
            Delta_grid = (
                outcomes
                .pivot_table(
                    index=x_col,
                    columns=y_col,
                    values="inner_P_Delta (%)",
                    aggfunc="mean"
                )
                .reindex(index=x_vals, columns=y_vals)
            )
        else:
            Delta_grid = None
    elif value_col == "outer_P_ratio":
        if show_libration:
            libration_grid = (
                outcomes
                .pivot(index=x_col, columns=y_col, values="outer_librates")
                .reindex(index=x_vals, columns=y_vals)
            )
        else:
            libration_grid = None
        if show_Delta:
            Delta_grid = (
                outcomes
                .pivot(index=x_col, columns=y_col, values="outer_P_Delta (%)")
                .reindex(index=x_vals, columns=y_vals)
            )
        else:
            Delta_grid = None
    else:
        libration_grid = None
        Delta_grid = None

    # Edges 
    x_edges = np.empty(len(x_vals)+1)
    if x_col == 'Sigma_1au':
        x_edges[1:-1] = np.sqrt(x_vals[:-1] * x_vals[1:])
        x_edges[0] = x_vals[0]**2 / x_edges[1]
        x_edges[-1] = x_vals[-1]**2 / x_edges[-2]
    elif x_col == "log_tau_Omega_bin":

        x_edges[1:-1] = (
            np.arange(len(x_vals)-1) + 0.5
        )

        x_edges[0] = -0.5
        x_edges[-1] = len(x_vals)-0.5
    
    y_edges = np.empty(len(y_vals)+1)
    y_edges[1:-1] = np.sqrt(y_vals[:-1] * y_vals[1:])
    y_edges[0] = y_vals[0]**2 / y_edges[1]
    y_edges[-1] = y_vals[-1]**2 / y_edges[-2]
    
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))
    else:
        fig = ax.figure
    
    if 'survived' in value_col:
        cmap = mcolors.ListedColormap(["#482173", "#2E6F8E", "#29AF7F"])
        norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    else:
        cmap = plt.get_cmap(cmap).copy() # make a copy so we don't modify the global colormap
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if log_cmap else None

    cmap.set_bad("lightgray")
    
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        values.T,
        cmap=cmap,
        norm=norm,
        shading="flat",
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
    )

    if x_col == "Sigma_1au":
        ax.set_xlabel(r"$\Sigma_{1\,\rm AU}\;(\mathrm{g\,cm^{-2}})$")
        ax.set_xscale("log")
        ax.set_xticklabels([f"{x:.0f}" for x in x_vals][1::3]) 
        ax.set_xticks(x_vals[1::3])
    elif x_col == "log_tau_Omega_bin":

        ax.set_xlabel(r"$\log_{10}(\tau_a\Omega)$")
        ax.set_xscale("linear")

        tick_idx = np.arange(len(x_labels))[::3]

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(
            [f"{x:.1f}" for x in x_labels[::3]]
        )
    
    if y_col == "h_1au":
        ax.set_ylabel(r"$h_{1\,\rm AU}$")
        ax.set_yscale("log")
        ax.set_yticks(y_vals[1::3])
        ax.set_yticklabels([f"{y:.3f}" for y in y_vals][1::3])
    elif y_col == "log_K2":
        ax.set_ylabel(r"$\log \mathcal{K_2}$")
        ax.set_yscale("linear")    

    ax.minorticks_off()

    # Cell text
    ratios = [2/1, 5/3, 3/2, 4/3, 5/4, 6/5, 7/6, 1, 8/7, 11/9, 11/8, 9/7, 7/5, 8/5, 7/4, 9/8, 10/9, 11/10, 12/11, 13/12, 14/13, 13/11, 13/10]
    
    if x_col == "log_tau_Omega_bin":
        x_plot_vals = np.arange(len(x_vals))
    else:
        x_plot_vals = x_vals

    for i, x in enumerate(x_plot_vals):
        for j, y in enumerate(y_vals):

            value = values[i, j]
            if np.isnan(value):
                continue

            color = "white"
            if value > black_threshold:
                color = "k"
                
            if exp:
                disp_value = f"{value:.1e}"
            elif 'survived' in value_col:
                disp_value = f"{value:.2g}"  
            elif "%" in value_col:
                disp_value = f"{round(value)}%"
            elif "ratio" in value_col:
                frac = Fraction(value).limit_denominator()
                disp_value = f"{frac.numerator}:{frac.denominator}" if value in ratios else f"{value:.3f}" 
            elif "kyr" in value_col:
                disp_value = int(value)
            elif "id" in value_col:
                disp_value = int(value)
            else:
                disp_value = f"{value:.2f}"  
                
            if show_text:
                text = ax.text(
                    x, y,
                    disp_value,
                    ha="center",
                    va="center",
                    fontsize=4,
                    weight='bold',
                    color=color,
                    zorder=5
                )
                
                text.set_path_effects([
                    path_effects.withStroke(linewidth=0.4, foreground="black")
                ])
                if value > black_threshold:
                    text.set_path_effects([
                        path_effects.withStroke(linewidth=0.4, foreground="white")
                    ])

            if libration_grid is not None:
                if libration_grid.values[i, j]:
                    ax.annotate(
                        "L",
                        xy=(x, y),
                        xytext=(0, 1),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=5,
                        color="C3",
                        fontweight="bold"
                    )
                    
            if Delta_grid is not None:
                ax.annotate(
                    rf"$\Delta = {Delta_grid.values[i, j]}$%",
                    xy=(x, y),
                    xytext=(0, -5),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=4,
                    color=color,
                    fontweight="bold"
                )
                
    if show_box:                # Bottom left corner
        rect = patches.Rectangle((1352, 0.0311), width=2325, height=0.022, edgecolor='r', facecolor='none', linewidth=1, alpha=0.7, linestyle='dashed')
        ax.add_patch(rect)
            
    if create_colorbar:
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(label)
        cbar.ax.tick_params(direction='inout')

        if 'ratio' in value_col:
            cbar.set_ticks(ratios[:8], labels=['2:1', '5:3', '3:2', '4:3', '5:4', '6:5', '7:6', '1:1'])
            cbar.ax.minorticks_off()
        elif 'survived' in value_col:
            cbar.set_ticks([1,2,3])

    return mesh

fig, ax = plt.subplots(figsize=(5,4))

first = pd.read_hdf(f"dfs/planet_outcomes{dataset_id}_{snapshots[0]}.h5", key="df")

plot_param_grid_map(
    first,
    "inner_P_ratio",
    r"Inner pair period ratio ($P_c/P_b$)",
    x_col="log_tau_Omega", y_col="log_K2",
    ax=ax,
    create_colorbar=True,
    show_text=True,
    vmin=1,
    vmax=2,
    log_cmap=True,
    black_threshold=1.6
)

def update(snapshot):

    ax.clear()

    outcomes = pd.read_hdf(
        f"dfs/planet_outcomes{dataset_id}_{int(snapshot)}.h5",
        key="df"
    )

    plot_param_grid_map(
        outcomes,
        "inner_P_ratio",
        r"Inner pair period ratio ($P_c/P_b$)",
        x_col="log_tau_Omega", y_col="log_K2",
        ax=ax,
        create_colorbar=False,
        show_text=True,
        show_libration=False,
        show_Delta=False,
        show_box=False,
        black_threshold=1.6,
        vmin=1,
        vmax=2,
        log_cmap=True,
    )

    ax.set_title(f"Snapshot {snapshot}")

    return ax.get_children()

fps = 20
pause = 10

frames = (
    [snapshots[0]] * pause +
    list(snapshots) +
    [snapshots[-1]] * pause
)

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=1000/fps,
    repeat=True,
)


ani.save(
    "P_ratio_snapshots_K2.mp4",
    writer=FFMpegWriter(fps=fps)
)

print("Animation saved.")