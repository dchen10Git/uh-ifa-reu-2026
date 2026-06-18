"""
plot_orbital_video.py
---------------------
Generate a top-down animation of orbital evolution from rebound simulation HDF5 files.

Usage:
    python plot_orbital_video.py --dataset 8 --sim 54
    python plot_orbital_video.py --dataset 8 --sim 54 --fps 30 --output my_video.mp4
    python plot_orbital_video.py --dataset 8 --sim 54 --gif   # output as GIF instead
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from pathlib import Path

import rebound_sims as reb_sims


# ─────────────────────────────────────────────────────────────────────────────
# Orbital mechanics helpers
# ─────────────────────────────────────────────────────────────────────────────

def mean_anomaly_to_true_anomaly(M, e, n_iter=50):
    """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson, then convert to f."""
    M = np.atleast_1d(np.asarray(M, dtype=float)) % (2 * np.pi)
    E = M.copy()
    for _ in range(n_iter):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E += dE
        if np.all(np.abs(dE) < 1e-10):
            break
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                       np.sqrt(1 - e) * np.cos(E / 2))
    return f[0] if f.size == 1 else f


def l_to_xy(a, e, pomega, l):
    """Convert mean longitude l to (x, y) position [AU].

    l = M + pomega  where  pomega = omega + Omega  (longitude of periapsis)
    So M = l - pomega, then solve Kepler's equation for true anomaly f,
    then rotate by pomega to get inertial coordinates.
    """
    M = (l - pomega) % (2 * np.pi)
    f = mean_anomaly_to_true_anomaly(M, e)

    # Distance from focus
    r = a * (1 - e**2) / (1 + e * np.cos(f))

    # Position in orbital plane (periapsis along +x), then rotate by pomega
    x = r * np.cos(f + pomega)
    y = r * np.sin(f + pomega)
    return x, y


def orbit_ellipse_xy(a, e, pomega, n_pts=200):
    """Return (x, y) arrays tracing the full orbit ellipse (face-on).
    pomega = omega + Omega orients the ellipse in the inertial frame.
    """
    f = np.linspace(0, 2 * np.pi, n_pts)
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    x = r * np.cos(f + pomega)
    y = r * np.sin(f + pomega)
    return x, y


def df_col(df, col, idx):
    """Safely get df[col].iloc[idx], returning NaN if missing."""
    if col not in df.columns:
        return np.nan
    v = df[col].iloc[idx]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sim(dataset_id, sim_id, base_dir=None):
    if base_dir is None:
        base_dir = Path.cwd()
    file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"

    if not file_path.exists():
        # Try relative to cwd as fallback
        file_path = base_dir / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find simulation file at {file_path}")

    if reb_sims is None:
        raise ImportError("reb_sims module not found. Make sure it's on your PYTHONPATH.")

    print(f"Loading dataset {dataset_id}, sim {sim_id} from {file_path} ...")
    saved_sim = reb_sims.load_simulation_run(file_path)
    sim_data, metadata = saved_sim

    rock_names = list(sim_data.keys())
    num_pl     = metadata["num_pl"]
    num_em     = metadata["num_em"]
    num_ptsml  = metadata["num_ptsml"]

    times = sim_data[rock_names[0]]["time"].values
    print(f"  {len(times)} timesteps, t_end = {times[-1]/1e3:.1f} kyr")
    print(f"  planets: {num_pl}, embryos: {num_em}, planetesimals: {num_ptsml}")

    return sim_data, metadata, rock_names, num_pl, num_em, num_ptsml, times


# ─────────────────────────────────────────────────────────────────────────────
# Animation builder
# ─────────────────────────────────────────────────────────────────────────────

# Visual config by body type
STYLE = {
    "planet":       dict(s=120,  color=None,    alpha=1.0,  zorder=5,  lw=1.2),
    "embryo":       dict(s=25,   color="skyblue",  alpha=0.75, zorder=3,  lw=0.5),
    "planetesimal": dict(s=5,    color="white",  alpha=0.6,  zorder=2,  lw=0.3),
}

PLANET_COLORS = [f"C{i}" for i in range(10)]

ORBIT_ALPHA = {
    "planet":       0.55,
    "embryo":       0.20,
    "planetesimal": 0.08,
}


def build_animation(
    sim_data, metadata, rock_names,
    num_pl, num_em, num_ptsml, times,
    n_frames=300,
    fps=24,
    draw_orbits=True,
    xlim=(-2, 2),
    ylim=(-2, 2),
    t_units="kyr",
):
    time_factor = {"yr": 1, "kyr": 1e3, "Myr": 1e6}[t_units]

    # Subsample timesteps evenly
    frame_idxs = np.linspace(0, len(times) - 1, n_frames, dtype=int)
    frame_idxs = np.arange(0, 300) # <- to see first 300 frames

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="k")
    ax.set_facecolor("k")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)", color="white")
    ax.set_ylabel("y (AU)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Central star
    ax.scatter([0], [0], s=200, color="yellow", zorder=10, marker="*")

    time_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        color="white", fontsize=9, va="top", fontfamily="monospace"
    )

    # ── Pre-build artist lists ────────────────────────────────────────────────

    planet_dots   = []
    planet_orbits = []  # list of Line2D

    embryo_dots   = []
    embryo_orbits = []

    ptsml_xs = np.full(num_ptsml, np.nan)
    ptsml_ys = np.full(num_ptsml, np.nan)
    ptsml_sc = ax.scatter(ptsml_xs, ptsml_ys, **{**STYLE["planetesimal"], "color": "white"})

    # Planets
    for i in range(num_pl):
        color = PLANET_COLORS[i % len(PLANET_COLORS)]
        sc = ax.scatter([], [], **{**STYLE["planet"], "color": color, "label": rock_names[i]})
        planet_dots.append(sc)
        if draw_orbits:
            line, = ax.plot([], [], color=color, lw=STYLE["planet"]["lw"],
                            alpha=ORBIT_ALPHA["planet"], zorder=STYLE["planet"]["zorder"] - 1)
            planet_orbits.append(line)

    # Embryos
    for i in range(num_em):
        sc = ax.scatter([], [], **STYLE["embryo"])
        embryo_dots.append(sc)
        if draw_orbits:
            line, = ax.plot([], [], color="skyblue", lw=STYLE["embryo"]["lw"],
                            alpha=ORBIT_ALPHA["embryo"], zorder=STYLE["embryo"]["zorder"] - 1)
            embryo_orbits.append(line)

    ax.legend(loc="upper right", fontsize=7, framealpha=0.2,
              labelcolor="white", facecolor="black")

    # ── Update function ───────────────────────────────────────────────────────

    def get_pos(df, idx):
        """Extract (x, y) and pomega from a body's dataframe at timestep idx.
        Uses mean longitude l to place the body on its orbit.
        pomega = omega + Omega (longitude of periapsis) orients the ellipse.
        Falls back gracefully if columns are missing.
        """
        a      = df_col(df, "a",      idx)
        e      = df_col(df, "e",      idx)
        l      = df_col(df, "l",      idx)   # mean longitude (required)

        # pomega = omega + Omega.  Try stored pomega first, then sum parts, then 0.
        pomega = df_col(df, "pomega", idx)
        if not np.isfinite(pomega):
            omega = df_col(df, "omega", idx)
            Omega = df_col(df, "Omega", idx)
            pomega = (0.0 if not np.isfinite(omega) else omega) + \
                     (0.0 if not np.isfinite(Omega) else Omega)

        if not np.isfinite(a) or not np.isfinite(e) or e >= 1.0 or a <= 0:
            return np.nan, np.nan, pomega

        if not np.isfinite(l):
            return np.nan, np.nan, pomega

        x, y = l_to_xy(a, e, pomega, l)
        return x, y, pomega

    def update(frame_num):
        idx = frame_idxs[frame_num]
        t = times[idx]
        time_text.set_text(f"t = {t / time_factor:.2f} {t_units}")

        # Planets
        for i in range(num_pl):
            name = rock_names[i]
            df   = sim_data[name]
            x, y, pomega = get_pos(df, idx)
            a = df_col(df, "a", idx)
            e = df_col(df, "e", idx)

            if not np.isfinite(x):
                planet_dots[i].set_offsets(np.empty((0, 2)))
                if draw_orbits:
                    planet_orbits[i].set_data([], [])
            else:
                planet_dots[i].set_offsets([[x, y]])
                if draw_orbits:
                    ox, oy = orbit_ellipse_xy(a, e, pomega)
                    planet_orbits[i].set_data(ox, oy)

        # Embryos
        for i in range(num_em):
            name = rock_names[num_pl + i]
            df   = sim_data[name]
            x, y, pomega = get_pos(df, idx)
            a = df_col(df, "a", idx)
            e = df_col(df, "e", idx)

            if not np.isfinite(x):
                embryo_dots[i].set_offsets(np.empty((0, 2)))
                if draw_orbits:
                    embryo_orbits[i].set_data([], [])
            else:
                embryo_dots[i].set_offsets([[x, y]])
                if draw_orbits:
                    ox, oy = orbit_ellipse_xy(a, e, pomega)
                    embryo_orbits[i].set_data(ox, oy)

        # Planetesimals (batch update via scatter)
        for i in range(num_ptsml):
            name = rock_names[num_pl + num_em + i]
            df   = sim_data[name]
            x, y, _ = get_pos(df, idx)
            ptsml_xs[i] = x
            ptsml_ys[i] = y

        ptsml_sc.set_offsets(np.column_stack([ptsml_xs, ptsml_ys]))

        artists = [time_text, ptsml_sc] + planet_dots + embryo_dots
        if draw_orbits:
            artists += planet_orbits + embryo_orbits
        return artists

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 / fps,
        blit=True,
    )

    return fig, ani


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate top-down orbital evolution video.")
    parser.add_argument("--dataset",    type=int, default=0,      help="dataset_id")
    parser.add_argument("--sim",        type=int, default=54,     help="sim_id")
    parser.add_argument("--frames",     type=int, default=300,    help="Number of animation frames")
    parser.add_argument("--fps",        type=int, default=24,     help="Frames per second")
    parser.add_argument("--no-orbits",  action="store_true",      help="Skip drawing orbit ellipses")
    parser.add_argument("--xlim",       type=float, default=2,  help="Half-width of x axis (AU)")
    parser.add_argument("--tunits",     default="kyr",            choices=["yr","kyr","Myr"])
    parser.add_argument("--output",     default=None,             help="Output filename (auto if not set)")
    parser.add_argument("--gif",        action="store_true",      help="Output GIF instead of MP4")
    parser.add_argument("--basedir",    default=None,             help="Base directory (default: cwd)")
    args = parser.parse_args()

    base_dir = Path(args.basedir) if args.basedir else Path.cwd()

    sim_data, metadata, rock_names, num_pl, num_em, num_ptsml, times = \
        load_sim(args.dataset, args.sim, base_dir)

    lim = args.xlim
    fig, ani = build_animation(
        sim_data, metadata, rock_names,
        num_pl, num_em, num_ptsml, times,
        n_frames=args.frames,
        fps=args.fps,
        draw_orbits=not args.no_orbits,
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        t_units=args.tunits,
    )

    if args.output:
        out_path = Path(args.output)
    elif args.gif:
        out_path = Path(f"orbital_evo_ds{args.dataset}_sim{args.sim}.gif")
    else:
        out_path = Path(f"orbital_evo_ds{args.dataset}_sim{args.sim}.mp4")

    print(f"Rendering {args.frames} frames → {out_path} ...")

    if args.gif:
        writer = animation.PillowWriter(fps=args.fps)
    else:
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800,
                                        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])

    ani.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()