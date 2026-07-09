import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import rebound_sims as reb_sims

def animate_cumulative_mass_distribution(sim_id, dataset_id=7, n_frames=100,
                                         fps=15, save_path=None):
    """
    Animate the cumulative planetesimal mass distribution over snapshots
    for a single simulation.

    Planet b/c/d locations are shown as vertical dashed lines.
    
    Parameters
    ----------
    sim_id : int
        Simulation ID.
    dataset_id : int
        Dataset number.
    n_frames : int
        Number of snapshots/frames to animate.
    fps : int
        Frames per second.
    save_path : str or Path or None
        If given, saves animation. Use .gif or .mp4 extension.
    """

    def get_ptsml_distribution(sim_data, metadata, snapshot_idx):
        a = []
        m = []

        ptsml_names = [
            name for name in sim_data.keys()
            if name.startswith("ptsml")
        ]

        for name in ptsml_names:
            df = sim_data[name]

            if snapshot_idx >= len(df):
                continue

            aval = df["a"].iloc[snapshot_idx]
            if np.isnan(aval):
                continue

            a.append(aval)
            m.append(metadata["m_ptsml"])

        a = np.asarray(a)
        m = np.asarray(m)

        if len(a) == 0:
            return np.asarray([]), np.asarray([])

        order = np.argsort(a)
        return a[order], np.cumsum(m[order])

    def get_planet_locations(sim_data, metadata, snapshot_idx):
        locations = {}

        for i in range(metadata["num_pl"]):
            name = f"planet {chr(ord('b') + i)}"

            if name not in sim_data:
                continue

            df = sim_data[name]

            if snapshot_idx >= len(df):
                continue

            aval = df["a"].iloc[snapshot_idx]
            if np.isnan(aval):
                continue

            locations[name] = aval

        return locations

    def get_interior_resonance_locations(planet_locations):
        resonance_locations = {}
        resonances = {
            "2:1": 2 / 1,
            "3:2": 3 / 2,
            "4:3": 4 / 3,
            "5:4": 5 / 4,
            "6:5": 6 / 5,
        }

        for planet_name, planet_a in planet_locations.items():
            resonance_locations[planet_name] = {
                res_name: planet_a * period_ratio**(-2 / 3)
                for res_name, period_ratio in resonances.items()
            }

        return resonance_locations

    base_dir = Path.cwd()
    file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"
    saved_sim = reb_sims.load_simulation_run(file_path)
    sim_data, metadata = saved_sim

    # Use available snapshots if fewer than requested
    first_name = list(sim_data.keys())[0]
    max_snapshots = len(sim_data[first_name])
    n_frames = min(n_frames, max_snapshots)

    # Precompute distributions and planet positions
    distributions = [
        get_ptsml_distribution(sim_data, metadata, idx)
        for idx in range(n_frames)
    ]

    planet_locations_by_frame = [
        get_planet_locations(sim_data, metadata, idx)
        for idx in range(n_frames)
    ]

    resonance_locations_by_frame = [
        get_interior_resonance_locations(locations)
        for locations in planet_locations_by_frame
    ]

    # Fixed plot limits to avoid jitter
    all_a = np.concatenate([
        a for a, M in distributions
        if len(a) > 0
    ])

    all_M = np.concatenate([
        M for a, M in distributions
        if len(M) > 0
    ])

    all_planet_a = np.asarray([
        aval
        for locations in planet_locations_by_frame
        for aval in locations.values()
    ])

    all_resonance_a = np.asarray([
        aval
        for planet_locations in resonance_locations_by_frame
        for resonance_locations in planet_locations.values()
        for aval in resonance_locations.values()
    ])

    xvals = np.concatenate([all_a, all_planet_a, all_resonance_a])
    xmin = max(0.1, np.nanmin(xvals) * 0.8)
    xmax = np.nanmax(xvals) * 1.2
    ymax = np.nanmax(all_M) * 1.05

    fig, ax = plt.subplots(figsize=(6, 4))

    line, = ax.step([], [], where="post", lw=2, color="k")

    planet_lines = {}
    planet_texts = {}
    resonance_lines = {}
    resonance_texts = {}

    planet_colors = {
        "planet b": "C0",
        "planet c": "C1",
        "planet d": "C2",
    }

    for name in ["planet b", "planet c", "planet d"]:
        planet_lines[name] = ax.axvline(
            np.nan,
            color=planet_colors[name],
            ls="--",
            lw=1.5,
            alpha=0.85
        )

        planet_texts[name] = ax.text(
            1,
            ymax * 0.96,
            name[-1],
            color=planet_colors[name],
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold"
        )

        resonance_lines[name] = {}
        resonance_texts[name] = {}

        for res_name in ["2:1", "3:2", "4:3", "5:4", "6:5"]:
            resonance_lines[name][res_name] = ax.axvline(
                np.nan,
                color=planet_colors[name],
                ls=":",
                lw=0.9,
                alpha=0.45
            )

            resonance_texts[name][res_name] = ax.text(
                1,
                ymax * 0.88,
                res_name,
                color=planet_colors[name],
                ha="center",
                va="top",
                fontsize=6,
                alpha=0.75,
                rotation=90
            )

    time_text = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xscale("log")
    ax.set_xlabel("Semi-major axis (AU)")
    ax.set_ylabel("Cumulative mass of planetesimals ($M_\\oplus$)")
    ax.grid(True)

    def update(frame):
        a, M = distributions[frame]

        line.set_data(a, M)

        locations = planet_locations_by_frame[frame]
        resonance_locations = resonance_locations_by_frame[frame]

        for name in ["planet b", "planet c", "planet d"]:
            if name in locations:
                aval = locations[name]
                planet_lines[name].set_xdata([aval, aval])
                planet_texts[name].set_position((aval, ymax * 0.96))
                planet_texts[name].set_visible(True)

                for res_name, res_aval in resonance_locations[name].items():
                    resonance_lines[name][res_name].set_xdata([res_aval, res_aval])
                    resonance_texts[name][res_name].set_position((res_aval, ymax * 0.88))
                    resonance_texts[name][res_name].set_visible(True)
            else:
                planet_lines[name].set_xdata([np.nan, np.nan])
                planet_texts[name].set_visible(False)

                for res_name in ["2:1", "3:2", "4:3", "5:4", "6:5"]:
                    resonance_lines[name][res_name].set_xdata([np.nan, np.nan])
                    resonance_texts[name][res_name].set_visible(False)

        try:
            t = sim_data[first_name]["time"].iloc[frame]
            time_text.set_text(f"sim {sim_id}, snapshot {frame}, t = {t/1e3:.1f} kyr")
        except Exception:
            time_text.set_text(f"sim {sim_id}, snapshot {frame}")

        resonance_line_artists = [
            res_line
            for planet_res_lines in resonance_lines.values()
            for res_line in planet_res_lines.values()
        ]
        resonance_text_artists = [
            res_text
            for planet_res_texts in resonance_texts.values()
            for res_text in planet_res_texts.values()
        ]

        return [
            line,
            time_text,
            *planet_lines.values(),
            *planet_texts.values(),
            *resonance_line_artists,
            *resonance_text_artists,
        ]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)

        if save_path.suffix == ".gif":
            anim.save(save_path, writer=PillowWriter(fps=fps))
        elif save_path.suffix == ".mp4":
            anim.save(save_path, writer=FFMpegWriter(fps=fps))
        else:
            raise ValueError("save_path must end in .gif or .mp4")

    return anim

assert len(sys.argv) == 3

anim = animate_cumulative_mass_distribution(
    sim_id=sys.argv[2],
    dataset_id=sys.argv[1],
    n_frames=1000,
    fps=60,
    save_path=f"Figures/mass_cdf_ds{sys.argv[1]}_sim{sys.argv[2]}.mp4"
)

# Usage: python3 plot_cdf_video.py <dataset_id> <sim_id>
