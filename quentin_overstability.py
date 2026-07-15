import celmech
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import astropy.units as u
import astropy.constants as constants

import warnings
warnings.filterwarnings('ignore')


from astropy import units as u
from astropy import constants as const
import cmath

RSUN = const.R_sun.to('km').value
RJUP = const.R_jup.to('km').value
REAR = const.R_earth.to('km').value
MJUP = const.M_jup.to('kg').value
MEAR = const.M_earth.to('kg').value
MSUN = const.M_sun.to('kg').value
AU = const.au.to('km').value

from matplotlib import rc
linewidths = 2
axislinewidths = 3
lenticks = 8
fontsize = 20
rc('font', family='serif', size=fontsize)
rc('xtick.major', size=lenticks)
rc('xtick', direction='in',top=True)
rc('ytick', direction='in',right=True)
rc('xtick.major', size=lenticks * 4 / 3,width=2.5)
rc('xtick.minor', size=lenticks * 2 / 3,visible=True,width=1)
rc('ytick.major', size=lenticks * 4 / 3,width=2.5)
rc('ytick.minor', size=lenticks * 2 / 3,visible=True,width=1)
rc('lines', linewidth=linewidths)
rc('axes', linewidth=axislinewidths)
rc('figure',facecolor='w',figsize=(8,4))
plt.rcParams['text.usetex'] = False
pallet = ['#'+s for s in ["03045e","023e8a","0077b6","0096c7","00b4d8","48cae4","90e0ef","ade8f4","caf0f8"]]

# ---- h(q) and f(q) from your provided equations ----
def h_q(q, alpha, f_d_i, f_d_o, j):
    """Compute h(q) as defined in Eq. 31"""
    numerator = f_d_i**2 + f_d_o**2 * q**2 * alpha
    denominator = j * f_d_i**2 + (j - 1) * f_d_o**2 * q * np.sqrt(alpha)
    return (1 / (1 + q * np.sqrt(alpha))) * (numerator / denominator)

def f_q(q, alpha, f_d_i, f_d_o):
    """Compute f(q) as defined in Eq. 32"""
    ratio_squared = (f_d_o / f_d_i)**2
    return 1 - ratio_squared * q**2 * alpha

# ---- Main resonance criteria functions ----
def tau_m_over_tau_e_overstab(mu_o, q, alpha, j, f_d_i, f_d_o):
    """Stability criterion"""
    h = h_q(q, alpha, f_d_i, f_d_o, j)
    f = f_q(q, alpha, f_d_i, f_d_o)
    factor = (3 / mu_o)**(2 / 3)
    return factor * h * ((j - 1) / (f_d_i * alpha))**(2 / 3) * f

def tau_m_over_tau_e_escape(mu_o, q, alpha, j, f_d_i, f_d_o):
    """Escape criterion"""
    h = h_q(q, alpha, f_d_i, f_d_o, j)
    factor = ((3 / mu_o)**(2 / 3)) * h / 4
    numerator = (j - 1)**2 + j**2 * q
    denominator = f_d_i * alpha + f_d_o * q**2
    return factor * (numerator / denominator)**(2 / 3)

# ---- Optional diagnostic expressions ----
def tau_e_tau_m_weak_e_damping(q, f_d_i, f_d_o, alpha, mu_o, n_o, j):
    """Weak eccentricity damping condition"""
    n_i = n_o * (j / (j - 1)) ** (3 / 2)
    term1 = j * f_d_i ** 2
    term2 = (j - 1) * f_d_o ** 2 * q * np.sqrt(alpha)
    return 1 / ((1 + q * np.sqrt(alpha)) * (term1 + term2)  * (mu_o * n_i * alpha)**2)

def tau_m_slow_migration(j, mu_o, mu_i, n_o, alpha, f_d_i, f_d_o):
    """Slow migration condition"""
    n_i = n_o * (j / (j - 1)) ** (3 / 2)
    A = (j - 1) * mu_o * n_i * alpha + j * mu_i * n_o
    B = ((j - 1)**2 * mu_o * n_i**2 * alpha + j**2 * mu_i * n_o**2) ** (1 / 3)
    C = (mu_o * n_i * alpha * f_d_i**2 + mu_i * n_o * f_d_o**2) ** (2 / 3)

    return 1 / A * B / (3**(1/3) * C)

def tau_a_from_sd_and_h(SF,H,a,mp,s,beta):
    G = const.G.to('AU 3 Msun -1 yr -2')
    SF = SF * u.g /u.cm**2 * (a)**(-s)
    H = H * a ** beta
    Ms = 1 * const.M_sun
    a *= u.au
    n = np.sqrt(G*Ms/a**3)
    mp *= const.M_earth
    tau_a =  Ms*Ms * H*H / (2.7+1.1*s) / (mp * SF * a*a * n)
    K = 0.78 / (2.7 + 1.1 * s) * H ** (-2)
    return tau_a.to('yr').value, K


# Redefine plot_capture_escape_normed within this cell to control its behavior
def old_plot_capture_escape_normed(ax,mu_i, mu_o, q, alpha, j, f_d_i, f_d_o, n_o, c,horiz_ls='--', vert_ls='-', diag_ls='-',horiz_lw=2, vert_lw=2, diag_lw=2,horiz_x_range=None, vert_y_range=None):

    K_over = tau_m_over_tau_e_overstab(mu_o, q, alpha, j, f_d_i, f_d_o)
    K_esca = tau_m_over_tau_e_escape(mu_o, q, alpha, j, f_d_i, f_d_o)
    tauae_weak = tau_e_tau_m_weak_e_damping(q, f_d_i, f_d_o, alpha, mu_o, n_o, j)

    tauae_weak_normed = tauae_weak * n_o  ## (taua * n_o) * tau_e
    tau_a_slow = tau_m_slow_migration(j, mu_o, mu_i, n_o, alpha, f_d_i, f_d_o)
    tau_a_slow_normed = tau_a_slow * n_o  ## (taua * n_o)

    # Determine x-range for horizontal and diagonal lines
    if horiz_x_range is None:
        plot_x_min, plot_x_max = ax.get_xlim() # Default to full plot x-range
    else:
        plot_x_min, plot_x_max = horiz_x_range

    taua_plot_normed_lines = np.logspace(np.log10(plot_x_min), np.log10(plot_x_max), 1000)
    taue_plot = tauae_weak_normed/taua_plot_normed_lines

    Kmin = K_esca
    if K_over > K_esca:
        ax.plot(taua_plot_normed_lines,np.ones_like(taua_plot_normed_lines) * K_esca,ls=horiz_ls,lw=horiz_lw,c=c)
        Kmin = K_esca
    elif K_over > 0:
        ax.plot(taua_plot_normed_lines,np.ones_like(taua_plot_normed_lines) * K_over,ls=horiz_ls,lw=horiz_lw,c=c)
        Kmin = K_over
    else:
        Kmin = 10 # Default Kmin if no overstable regime

    # Determine vertical line y-range
    if vert_y_range is not None:
        vert_line_y_min, vert_line_y_max = vert_y_range
    else:
        vert_line_y_min, vert_line_y_max = ax.get_ylim()[0], Kmin # Default: from bottom of current plot to Kmin

    # Modified vertical line to start from the bottom y-axis limit and go up to Kmin
    ax.plot([tau_a_slow_normed,tau_a_slow_normed],[vert_line_y_min, vert_line_y_max],
            ls=vert_ls,c=c,lw=vert_lw)

    ax.text(tau_a_slow_normed,50,'{0:d}:{1:d}'.format(j,j-1),color=c,ha='right',rotation=90,fontsize=13)
    ax.loglog(taua_plot_normed_lines,taua_plot_normed_lines/taue_plot/n_o,ls=diag_ls,c=c,lw=diag_lw)

    ax.set_xlabel(r'$au_a/P_o$'  ,fontsize=20)
    ax.set_ylabel(r'$K \equiv \tau_a/\tau_e$',fontsize=20)

def not_corrected_lines_migration():

    fig, ax = plt.subplots(figsize=(8, 7))


    # -------------------------------------------------------------------
    # Resonance / stability benchmark curves
    # -------------------------------------------------------------------
    # Outer embryo mass: all models use a 5 Earth-mass outer embryo.
    # In solar-mass units, M_earth/M_sun \simeq 3e-6.
    mu_o = 5 * 3e-6

    # Representative inner/outer mass ratio used for the criterion curves.
    # This keeps the benchmark biased toward q < 1, consistent with the
    # intended long overstability escape-chain setup.
    q = 0.5
    mu_i = q * mu_o

    per_o = 10.0  # outer period in days
    n_o = 2 * np.pi / (per_o / 365.25)

    # Set x and y limits before plotting the lines so they can be retrieved by get_xlim/get_ylim
    ax.set_xlim(1e3, 5e8)
    ax.set_ylim(1, 2e4)

    # Define adjustable ranges for the plotted lines for each MMR
    # These dictionaries can be modified by the user to control line extents for specific MMRs.
    # By default, they are initialized to cover the entire plot range.
    horiz_x_ranges = {}
    vert_y_ranges = {}

    default_xlim = ax.get_xlim()
    default_ylim = ax.get_ylim()

    for j_val in [2, 3,4]:
        horiz_x_ranges[j_val] = default_xlim
        # Note: vert_y_ranges[j_val] is (ymin, ymax). The Kmin in plot_capture_escape_normed will ensure it doesn't go above the relevant K value.
        vert_y_ranges[j_val] = (default_ylim[0], default_ylim[1])

    # Example of how to adjust the ranges for a specific MMR (e.g., j=3):
    # Uncomment and modify these lines to customize ranges
    #horiz_x_ranges[2] = (1e6, 1e9) # Set a specific x-range for horizontal lines for j=2
    #vert_y_ranges[2] = (1, 1000)  # Set a specific y-range for vertical lines for j=2

    #horiz_x_ranges[3] = (2e5, 1e9) # Set a specific x-range for horizontal lines for j=3
    #vert_y_ranges[3] = (1, 650) # Set a specific y-range for vertical lines for j=3

    for i, j in enumerate([2, 3,4]):
        k = 1
        alpha_res = ((j - k) / j)**(2 / 3)

        # Laplace/disturbing-function coefficients for j:(j-k) resonance
        f_d_i, f_d_o = np.abs(celmech.disturbing_function.get_fg_coefficients(j, k))

        old_plot_capture_escape_normed(
            ax, mu_i, mu_o, q, alpha_res, j, f_d_i, f_d_o, n_o,
            c=pallet[i],
            horiz_x_range=horiz_x_ranges[j],
            vert_y_range=vert_y_ranges[j]
        )

    # -------------------------------------------------------------------
    # Disk-model tracks: tau_a/P versus K
    # -------------------------------------------------------------------
    # Disk power-law parameters used in tau_a_from_sd_and_h:
    # Sigma(a) = Sigma0 * (a/AU)^(-s)
    # h(a)     = h0 * (a/AU)^beta
    s = 1.5
    beta = 0.0

    # Wide semimajor-axis grid appropriate for the k ~ 10--25 initial spacings.
    a_grid = np.logspace(-1, 1, 5)  # 0.1--10 AU

    P_grid = (
        2 * np.pi / np.sqrt(const.G * const.M_sun)
        * (a_grid * u.au)**(3/2)
    ).to('yr').value

    model_specs = {
        "Model 1: compact": {
            "Sigma0_range": (1700, 2000),
            "h0_range": (0.02, 0.22),
            "marker": "s",
            "note": r"$k\sim5$--$10$",
        },
        "Model 2: wide fiducial": {
            "Sigma0_range": (1700, 2000),
            "h0_range": (0.03, 0.30),
            "marker": "o",
            "note": r"$k\sim10$--$25$",
        },
        "Model 3: cold disk": {
            "Sigma0_range": (1700, 2000),
            "h0_range": (0.015, 0.025),
            "marker": "^",
            "note": r"$h_0=0.01$--$0.03$",
        },
        "Model 4: hot disk": {
            "Sigma0_range": (1700, 2000),
            "h0_range": (0.10, 0.28),
            "marker": "D",
            "note": r"$h_0=0.10$--$0.30$",
        },
        "Model 5: broad sweep": {
            "Sigma0_range": (170, 17000),
            "h0_range": (0.01, 0.25),
            "marker": "P",
            "note": r"broad $\Sigma_0$",
        },
    }

    def representative_values(vmin, vmax):
        """
        Return deterministic min, median, and max representative values.

        For strictly positive quantities spanning a range, the median is chosen
        as the geometric midpoint. This is especially important for Sigma0 in
        Model 5, which spans two orders of magnitude.
        """
        vmed = np.sqrt(vmin * vmax)
        return np.array([vmin,vmax])

    def plot_model_representatives(ax,label,spec,a_grid,P_grid,cmap="plasma",norm=None,s_powerlaw=1.0,beta_powerlaw=0.0):
        """
        Plot three representative model tracks:
            min:    (Sigma0_min, Sigma0_min)
            median: (Sigma0_med, h0_med)
            max:    (Sigma0_max, h0_max)

        Marker shape distinguishes the model.
        Color distinguishes semimajor axis.
        """
        sf_min, sf_max = spec["Sigma0_range"]
        h_min, h_max = spec["h0_range"]

        sigma_reps = representative_values(sf_min, sf_max)
        h_reps = representative_values(h_min, h_max)
        rep_names = ["min","max"]

        scatters = []

        for rep_name, sf, h0 in zip(rep_names, sigma_reps, h_reps):
            tau_a, K = tau_a_from_sd_and_h(
                SF=sf,
                H=h0,
                a=a_grid.copy(),
                mp=mu_o / 3e-6,  # convert solar mass ratio to Earth masses
                s=s_powerlaw,
                beta=beta_powerlaw,
            )

            tau_a_normed = tau_a * 2 * np.pi / P_grid

            sc = ax.scatter(
                tau_a_normed,
                K,
                c=a_grid,
                cmap=cmap,
                norm=norm,
                marker=spec["marker"],
                s=50,
                edgecolor="k",
                linewidth=0.35,
                alpha=0.85,
                zorder=20,
            )
            scatters.append(sc)

            # Connect the points lightly so that the trajectory direction in
            # parameter space remains visible while the marker colors encode a.
            ax.plot(
                tau_a_normed,
                K,
                color="0.35",
                lw=0.8,
                alpha=0.35,
                zorder=10,
            )

            # Annotate only the end points for the three representative tracks.
            ax.text(
                tau_a_normed[-1],
                K[-1],
                f" {rep_name}",
                fontsize=7,
                alpha=0.75,
                zorder=25,
            )

        return scatters[-1]

    # Shared color normalization: semimajor axis is the same for all models.
    a_norm = LogNorm(vmin=a_grid.min(), vmax=a_grid.max())
    cmap = "plasma"

    last_scatter = None
    for label, spec in model_specs.items():
        last_scatter = plot_model_representatives(
            ax,
            label,
            spec,
            a_grid=a_grid,
            P_grid=P_grid,
            cmap=cmap,
            norm=a_norm,
            s_powerlaw=s,
            beta_powerlaw=beta,
        )

    # -------------------------------------------------------------------
    # Legends and colorbar
    # -------------------------------------------------------------------
    model_handles = [
        Line2D(
            [0], [0],
            marker=spec["marker"],
            color="w",
            label=label,
            markerfacecolor="0.65",
            markeredgecolor="k",
            markersize=8,
            linestyle="None",
        )
        for label, spec in model_specs.items()
    ]

    legend1 = ax.legend(
        handles=model_handles,
        fontsize=8,
        frameon=False,
        loc="upper right",

    )
    ax.add_artist(legend1)

    #cbar = fig.colorbar(last_scatter, ax=ax, pad=0.02)
    #cbar.set_label(r"$a\ \mathrm{[AU]}$", fontsize=14)

    # -------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------
    ax.text(0.05, 0.70, 'dissipation stability', color='k', rotation=62,
            transform=ax.transAxes, fontsize=12)
    ax.text(0.12, 0.20, 'adiabaticity', color='k', rotation=90,
            transform=ax.transAxes, fontsize=12)
    ax.text(0.75, 0.33, 'overstability', color='k',
            transform=ax.transAxes, fontsize=12)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$\tau_a/P_o$', fontsize=20)
    ax.set_ylabel(r'$K \equiv \tau_a/\tau_e$', fontsize=20)

    ax.set_title(
        r'$M_{\rm in}/M_{\rm out}=0.5$, '
        r'$M_{\rm out}=5\,M_\oplus$'
    )
    # ============================================================
    # Hot-Jupiter core formation highlight region
    # ============================================================

    # Adjust these boundaries as needed
    hj_xmin, hj_xmax = 1e4, 3e6       # tau_a / P_o
    hj_ymin, hj_ymax = 1.5, 30        # K = tau_a / tau_e

    ax.fill_between(
        [hj_xmin, hj_xmax],
        hj_ymin,
        hj_ymax,
        color="gold",
        alpha=0.18,
        zorder=0,
    )

    ax.plot(
        [hj_xmin, hj_xmax, hj_xmax, hj_xmin, hj_xmin],
        [hj_ymin, hj_ymin, hj_ymax, hj_ymax, hj_ymin],
        color="goldenrod",
        lw=1.5,
        ls="--",
        alpha=0.9,
        zorder=1,
    )

    ax.text(
        np.sqrt(hj_xmin * hj_xmax),
        np.sqrt(hj_ymin * hj_ymax),
        "hot-Jupiter\ncore growth",
        ha="center",
        va="center",
        fontsize=11,
        color="darkgoldenrod",
        zorder=30,
    )

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_capture_escape_normed(ax,mu_i,mu_o,q,alpha,j,f_d_i,f_d_o,n_o,c,horiz_ls='--',vert_ls='-',diag_ls='-',horiz_lw=2,vert_lw=2,diag_lw=2,horiz_x_range=None,vert_y_range=None,diag_x_range=None):


    K_over = tau_m_over_tau_e_overstab(
        mu_o, q, alpha, j, f_d_i, f_d_o
    )

    K_esca = tau_m_over_tau_e_escape(
        mu_o, q, alpha, j, f_d_i, f_d_o
    )

    tauae_weak = tau_e_tau_m_weak_e_damping(
        q, f_d_i, f_d_o, alpha, mu_o, n_o, j
    )

    tauae_weak_normed = tauae_weak * n_o

    tau_a_slow = tau_m_slow_migration(
        j, mu_o, mu_i, n_o, alpha, f_d_i, f_d_o
    )

    tau_a_slow_normed = tau_a_slow * n_o
    
    if K_over > K_esca:
        Kmin = K_esca
    elif K_over > 0:
        Kmin = K_over
    else:
        Kmin = 10

    # Horizontal dashed overstability line
    if horiz_x_range is None:
        hx_min, hx_max = ax.get_xlim()
    else:
        hx_min, hx_max = horiz_x_range

    x_horiz = np.logspace(np.log10(hx_min), np.log10(hx_max), 1000)

    ax.plot(
        x_horiz,
        np.ones_like(x_horiz) * Kmin,
        ls=horiz_ls,
        lw=horiz_lw,
        c=c,
    )

    # Vertical slow-migration line
    if vert_y_range is None:
        vy_min, vy_max = ax.get_ylim()[0], Kmin
    else:
        vy_min, vy_max = vert_y_range

    ax.plot(
        [tau_a_slow_normed, tau_a_slow_normed],
        [vy_min, vy_max],
        ls=vert_ls,
        lw=vert_lw,
        c=c,
    )

    # Diagonal separatrix: original notebook expression
    if diag_x_range is None:
        dx_min, dx_max = ax.get_xlim()
    else:
        dx_min, dx_max = diag_x_range

    x_diag = np.logspace(np.log10(dx_min), np.log10(dx_max), 1000)

    taue_plot = tauae_weak_normed / x_diag
    K_diag = x_diag / taue_plot / n_o

    ax.loglog(
        x_diag,
        K_diag,
        ls=diag_ls,
        lw=diag_lw,
        c=c,
    )

    ax.text(
        tau_a_slow_normed,
        50,
        f'{j}:{j-1}',
        color=c,
        ha='right',
        rotation=90,
        fontsize=13,
    )
def add_system_to_capture_diagram(ax,sigma0,h0,a,mp,s=1.5,beta=0.0,label=None,marker='*',size=180,color='red',edgecolor='k',annotate=True):
    """
    Add a single system to the tau_a/P vs K diagram.

    Parameters
    ----------
    sigma0 : float
        Surface density normalization [g/cm^2]

    h0 : float
        Aspect ratio normalization

    a : float
        Semimajor axis [AU]

    mp : float
        Planet mass [Earth masses]

    label : str
        Optional system label
    """

    tau_a, K = tau_a_from_sd_and_h(
        SF=sigma0,
        H=h0,
        a=np.array([a]),
        mp=mp,
        s=s,
        beta=beta,
    )

    P = np.sqrt(a**3)

    tau_a_over_P = tau_a[0] / P

    ax.scatter(
        tau_a_over_P,
        K[0],
        marker=marker,
        s=size,
        color=color,
        edgecolor=edgecolor,
        zorder=100,
    )

    if annotate and label is not None:
        ax.annotate(
            label,
            (tau_a_over_P, K[0]),
            xytext=(6,6),
            textcoords='offset points',
            fontsize=10,
        )

    return tau_a_over_P, K[0]
def add_system_track_to_capture_diagram(ax,sigma0,h0,a_grid,a_norm,s=1.5,beta=0.0,mp=1.0,label=None,marker='*',size=180,edgecolor='k',lw=1.2,alpha=1.0,annotate=True,):

    P_grid = np.sqrt(a_grid**3)

    tau_a, K = tau_a_from_sd_and_h(
        SF=sigma0,
        H=h0,
        a=a_grid.copy(),
        mp=mp,
        s=s,
        beta=beta,
    )

    tau_a_over_P = tau_a / P_grid

    # grey connecting line
    ax.plot(
        tau_a_over_P,
        K,
        color='0.3',
        lw=lw,
        alpha=0.5,
        zorder=70,
    )

    # colored markers
    sc = ax.scatter(
        tau_a_over_P,
        K,
        c=a_grid,
        cmap='plasma',
        norm=a_norm,
        marker=marker,
        s=size,
        edgecolor=edgecolor,
        linewidth=0.7,
        zorder=90,
    )

    if annotate and label is not None:

        mid = len(a_grid)//2

        ax.annotate(
            label,
            (tau_a_over_P[mid], K[mid]),
            xytext=(8,8),
            textcoords='offset points',
            fontsize=10,
            weight='bold',
            zorder=100,
        )

    return sc
# ============================================================
# Figure setup
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

ax.set_xlim(1e3, 5e8)
ax.set_ylim(1, 2e4)

# ============================================================
# Resonance benchmark parameters
# ============================================================

mu_o = 5 * 3e-6
q = 0.1
mu_i = q * mu_o

per_o = 10.0  # days
n_o = 2 * np.pi / (per_o / 365.25) # outer Omega (mean motion)

# Independent line ranges

horiz_x_ranges = {
    2: (7e5, 5e8),
    3: (2.5e5, 5e8),
    4: (1.5e5, 5e8),
    5: (1e5, 5e8),
    6: (8e4, 5e8),
    7: (6.5e4,5e8),
}

vert_y_ranges = {
    2: (1, 1.3e3),
    3: (1, 7e2),
    4: (1, 6e2),
    5: (1, 4.7e2),
    6: (1, 4.7e2),
    7: (1,4.7e2),
}

diag_x_ranges = {
    2: (6.8e5, 1e9),
    3: (2.5e5, 1e9),
    4: (1.5e5, 1e9),
    5: (1e5, 1e9),
    6: (7.7e4, 1e9),
    7: (6e4,1e9),
}

# Plot resonance boundaries
import pickle as pkl
with open("fg_library.pkl", "rb") as f:
    fg_lib = pkl.load(f)


for i, j in enumerate([2,3,4,5,6,7]):

    k = 1 # order
    alpha_res = ((j - k) / j)**(2 / 3)
    
    f_d_i, f_d_o = np.abs(fg_lib[(j, k)])    

    plot_capture_escape_normed(
        ax,
        mu_i,
        mu_o,
        q,
        alpha_res,
        j,
        f_d_i,
        f_d_o,
        n_o,
        c=pallet[i],
        # horiz_x_range=horiz_x_ranges[j],
        # vert_y_range=vert_y_ranges[j],
        # diag_x_range=diag_x_ranges[j],
    )


# ============================================================
# Disk model tracks
# ============================================================

s = 1.
beta = 0.

a_grid = np.logspace(-1, 1, 5)
cmap = "plasma"
P_grid = (
    2 * np.pi / np.sqrt(const.G * const.M_sun)
    * (a_grid * u.au)**(3 / 2)
).to('yr').value
"""
model_specs = {
    "Model 1: compact": {
        "Sigma0_range": (1700, 2000),
        "h0_range": (0.02, 0.22),
        "marker": "s",
    },
    "Model 2: wide fiducial": {
        "Sigma0_range": (1700, 2000),
        "h0_range": (0.03, 0.30),
        "marker": "o",
    },
    "Model 3: cold disk": {
        "Sigma0_range": (1700, 2000),
        "h0_range": (0.015, 0.025),
        "marker": "^",
    },
    "Model 4: hot disk": {
        "Sigma0_range": (1700, 2000),
        "h0_range": (0.10, 0.28),
        "marker": "D",
    },
    "Model 5: broad sweep": {
        "Sigma0_range": (170, 17000),
        "h0_range": (0.01, 0.25),
        "marker": "P",
    },
}
"""
"""
model_specs = {
   

    "Model 3: cold disk": {
        "Sigma0_range": (1700, 2000),
        "h0_range": (0.01, 0.03),
        "marker": "^",
    },
    "Model 4: hot disk": {
        "Sigma0_range": (1700, 2000),
        "h0_range": (0.10, 0.25),
        "marker": "D",
    },
    
}
"""
model_specs = {} # choose models above

def representative_values(vmin, vmax):
    return np.array([vmin, vmax])


def plot_model_representatives(ax,label,spec,a_grid,P_grid,cmap="plasma",norm=None,s_powerlaw=1.5,beta_powerlaw=0.0,):

    sf_min, sf_max = spec["Sigma0_range"]
    h_min, h_max = spec["h0_range"]

    sigma_reps = representative_values(sf_min, sf_max)
    h_reps = representative_values(h_min, h_max)
    rep_names = ["min", "max"]

    last_scatter = None

    for rep_name, sf, h0 in zip(rep_names, sigma_reps, h_reps):

        tau_a, K = tau_a_from_sd_and_h(
            SF=sf,
            H=h0,
            a=a_grid.copy(),
            mp=mu_o / 3e-6,
            s=s_powerlaw,
            beta=beta_powerlaw,
        )

        tau_a_normed = tau_a * 2 * np.pi / P_grid

        last_scatter = ax.scatter(
            tau_a_normed,
            K,
            c=a_grid,
            cmap=cmap,
            norm=norm,
            marker=spec["marker"],
            s=50,
            edgecolor="k",
            linewidth=0.35,
            alpha=0.85,
            zorder=20,
        )

        ax.plot(
            tau_a_normed,
            K,
            color="0.35",
            lw=0.8,
            alpha=0.35,
            zorder=10,
        )

        ax.text(
            tau_a_normed[-1],
            K[-1],
            f" {rep_name}",
            fontsize=7,
            alpha=0.75,
            zorder=25,
        )

    return last_scatter


a_norm = LogNorm(vmin=a_grid.min(), vmax=a_grid.max())
cmap = "inferno"

last_scatter = None

for label, spec in model_specs.items():
    last_scatter = plot_model_representatives(
        ax,
        label,
        spec,
        a_grid=a_grid,
        P_grid=P_grid,
        cmap=cmap,
        norm=a_norm,
        s_powerlaw=s,
        beta_powerlaw=beta,
    )


# ============================================================
# Legend and colorbar
# ============================================================

model_handles = [
    Line2D(
        [0],
        [0],
        marker=spec["marker"],
        color="w",
        label=label,
        markerfacecolor="0.65",
        markeredgecolor="k",
        markersize=8,
        linestyle="None",
    )
    for label, spec in model_specs.items()
]

legend1 = ax.legend(
    handles=model_handles,
    fontsize=16,
    frameon=False,
    loc="upper right",
)

ax.add_artist(legend1)

#cbar = fig.colorbar(last_scatter, ax=ax, pad=0.02)
#cbar.set_label(r"$a\ \mathrm{[AU]}$", fontsize=14)


# ============================================================
# Formatting
# ============================================================

ax.text(
    0.05,
    0.70,
    "dissipative stability",
    color="k",
    rotation=62,
    transform=ax.transAxes,
    fontsize=12,
)

ax.text(
    0.12,
    0.20,
    "adiabaticity",
    color="k",
    rotation=90,
    transform=ax.transAxes,
    fontsize=12,
)

ax.text(
    0.75,
    0.33,
    "overstability",
    color="k",
    transform=ax.transAxes,
    fontsize=12,
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$\tau_a/P_o$", fontsize=20)
ax.set_ylabel(r"$K \equiv \tau_a/\tau_e$", fontsize=20)

ax.set_title(
    r"$M_{\rm in}/M_{\rm out}=0.5$, "
    r"$M_{\rm out}=5\,M_\oplus$",
    fontsize=18,
)

ax.tick_params(
    which="both",
    direction="in",
    top=True,
    right=True,
    labelsize=11,
)
# ============================================================
# Hot-Jupiter core formation highlight region
# ============================================================
"""
# Adjust these boundaries as needed
hj_xmin, hj_xmax = 2e5, 1e7       # tau_a / P_o
hj_ymin, hj_ymax = 5, 20        # K = tau_a / tau_e

ax.fill_between(
    [hj_xmin, hj_xmax],
    hj_ymin,
    hj_ymax,
    color="gold",
    alpha=0.18,
    zorder=0,
)

ax.plot(
    [hj_xmin, hj_xmax, hj_xmax, hj_xmin, hj_xmin],
    [hj_ymin, hj_ymin, hj_ymax, hj_ymax, hj_ymin],
    color="goldenrod",
    lw=1.5,
    ls="--",
    alpha=0.9,
    zorder=1,
)

ax.text(
    np.sqrt(hj_xmin * hj_xmax),
    np.sqrt(hj_ymin * hj_ymax),
    "merger formed via overstable escape",
    ha="center",
    va="center",
    fontsize=11,
    color="darkgoldenrod",
    zorder=30,
)





# Adjust these boundaries as needed
hj_xmin, hj_xmax = 5e3, 5e4       # tau_a / P_o
hj_ymin, hj_ymax = 70, 9e2        # K = tau_a / tau_e

ax.fill_between(
    [hj_xmin, hj_xmax],
    hj_ymin,
    hj_ymax,
    color="skyblue",
    alpha=0.18,
    zorder=0,
)

ax.plot(
    [hj_xmin, hj_xmax, hj_xmax, hj_xmin, hj_xmin],
    [hj_ymin, hj_ymin, hj_ymax, hj_ymax, hj_ymin],
    color="blue",
    lw=1.5,
    ls="--",
    alpha=0.9,
    zorder=1,
)

ax.text(
    np.sqrt(hj_xmin * hj_xmax),
    np.sqrt(hj_ymin * hj_ymax),
    "merger formed via MMR crossing",
    ha="center",
    va="center",
    fontsize=11,
    color="darkblue",
    zorder=30,
)
"""


def add_system_track_to_capture_diagram(ax,sigma0,h0,a_grid,a_norm,s=1.5,beta=0.0,mp=1.0,label=None,marker='*',size=180,edgecolor='k',lw=1.2,alpha=1.0,annotate=True,):

    P_grid = np.sqrt(a_grid**3)

    tau_a, K = tau_a_from_sd_and_h(
        SF=sigma0,
        H=h0,
        a=a_grid.copy(),
        mp=mp,
        s=s,
        beta=beta,
    )

    tau_a_over_P = tau_a / P_grid

    # grey connecting line
    ax.plot(
        tau_a_over_P,
        K,
        color='0.3',
        lw=lw,
        alpha=0.5,
        zorder=70,
    )

    # colored markers
    sc = ax.scatter(
        tau_a_over_P,
        K,
        c=a_grid,
        cmap='plasma',
        norm=a_norm,
        marker=marker,
        s=size,
        edgecolor=edgecolor,
        linewidth=0.7,
        zorder=90,
    )

    if annotate and label is not None:

        mid = len(a_grid)//2

        ax.annotate(
            label,
            (tau_a_over_P[mid], K[mid]),
            xytext=(8,8),
            textcoords='offset points',
            fontsize=10,
            weight='bold',
            zorder=100,
        )

    return sc


def add_sigma_h_region_to_capture_diagram(ax,sigma0_range,h0_range,a_grid,mp=1.0,s=1.5,beta=0.0,n_edge=80,facecolor="gold",edgecolor="darkgoldenrod",alpha=0.18,label=None,zorder=5):
    """
    Transform a rectangular region in (Sigma0, h0)
    into the (tau_a/P, K) diagram.

    sigma0_range : tuple
        (sigma0_min, sigma0_max) in g/cm^2

    h0_range : tuple
        (h0_min, h0_max)

    a_grid : array
        Same AU grid used in the main plot.
    """

    sig_min, sig_max = sigma0_range
    h_min, h_max = h0_range

    sigma_vals = np.linspace(sig_min, sig_max, n_edge)
    h_vals = np.linspace(h_min, h_max, n_edge)

    boundary_sigma = np.concatenate([
        sigma_vals,
        np.full_like(h_vals, sig_max),
        sigma_vals[::-1],
        np.full_like(h_vals, sig_min),
    ])

    boundary_h = np.concatenate([
        np.full_like(sigma_vals, h_min),
        h_vals,
        np.full_like(sigma_vals, h_max),
        h_vals[::-1],
    ])

    P_grid = np.sqrt(a_grid**3)

    for ia, a in enumerate(a_grid):

        tau_a_list = []
        K_list = []

        for sigma0, h0 in zip(boundary_sigma, boundary_h):

            tau_a, K = tau_a_from_sd_and_h(
                SF=sigma0,
                H=h0,
                a=np.array([a]),
                mp=mp,
                s=s,
                beta=beta,
            )

            tau_a_over_P = tau_a[0] / P_grid[ia]

            tau_a_list.append(tau_a_over_P)
            K_list.append(K[0])

        ax.fill(
            tau_a_list,
            K_list,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            lw=1.2,
            zorder=zorder,
        )

        if label is not None and ia == len(a_grid)//2:
            ax.text(
                np.median(tau_a_list),
                np.median(K_list),
                label,
                ha="center",
                va="center",
                fontsize=10,
                color=edgecolor,
                zorder=zorder + 10,
            )






"""
add_system_track_to_capture_diagram(
    ax,
    sigma0=1857,
    h0=0.1104,
    a_grid=a_grid,
    a_norm=a_norm,
    
    mp=1,
    label=r"Hot disk 1$M_\oplus$",
    marker='*',
    size=220,
)
"""


"""
add_system_track_to_capture_diagram(
    ax,
    sigma0=1957,
    h0=0.0236,
    a_grid=a_grid,
    a_norm=a_norm,
    
    mp=1,
    label=r"Cold disk 1$M_\oplus$",
    marker='*',
    size=220,
)
"""


"""
add_sigma_h_region_to_capture_diagram(
    ax,
    sigma0_range=(1700, 2000),
    h0_range=(0.1, 0.25),
    a_grid=a_grid,
    mp=1,
    facecolor="red",
    edgecolor="darkred",
    alpha=0.5,
    label="Merger via overstable escape",
)



add_sigma_h_region_to_capture_diagram(
    ax,
    sigma0_range=(1700, 2000),
    h0_range=(0.01, 0.03),
    a_grid=a_grid,
    mp=1,
    facecolor="darkcyan",
    edgecolor="darkblue",
    alpha=0.5,
    label="Merger via fast migration",
)
"""


add_sigma_h_region_to_capture_diagram(
    ax,
    sigma0_range=(100, 10000),
    h0_range=(0.01, 0.10),
    a_grid=np.logspace(-1,0,50),
    mp=1,
    facecolor="darkred",
    edgecolor="black",
    alpha=0.01,
    label="",
)


# add_sigma_h_region_to_capture_diagram(
#     ax,
#     sigma0_range=(1700, 2000),
#     h0_range=(0.01, 0.03),
#     a_grid=np.logspace(-1,1,50),
#     mp=1,
#     facecolor="darkblue",
#     edgecolor="black",
#     alpha=0.07,
#     label="Cold disk ",
# )


# add_sigma_h_region_to_capture_diagram(
#     ax,
#     sigma0_range=(1700, 2000),
#     h0_range=(0.1, 0.3),
#     a_grid=np.logspace(-1,1,50),
#     mp=1,
#     facecolor="darkred",
#     edgecolor="black",
#     alpha=0.07,
#     label="Hot disk",
# )
"""
add_sigma_h_region_to_capture_diagram(
    ax,
    sigma0_range=(288, 16161),
    h0_range=(0.01, 0.182),
    a_grid=a_grid,
    mp=1,
    facecolor="gold",
    edgecolor="darkgoldenrod",
    alpha=0.5,
    label="Hot jupiter growth",
)
"""


# conversion: M_sun / AU^2 -> g / cm^2
MSUN_AU2_TO_CGS = 1.98847e33 / (1.495978707e13)**2
"""
push_regions = {
    "Model 3 cold disk": {
        "sigma_sim": (0.0001942508444198, 0.0002230481697545),
        "h": (0.0100223245098489, 0.0293384666769982),
        "N": 45,
    },
    "Model 4 hot disk": {
        "sigma_sim": (0.0001946273475703, 0.0002230481697545),
        "h": (0.1037296389555114, 0.1290548455858325),
        "N": 10,
    },

    "Model 5 broad sweep": {
        "sigma_sim": (3.250048448572509e-05, 0.001818878666772),
        "h": (0.0106414034395569, 0.1820121812634587),
        "N": 25,
    },
    "Model 2 wide fluidical": {
        "sigma_sim": (0.0001943459504662, 0.0002214059361542),
        "h": (0.0100609926133451, 0.053367422452935),
        "N": 13,
    },
    "Model 1 compact fluidical": {
        "sigma_sim": (0.0001964556814947, 0.0002239143374248),
        "h": (0.0110850926629089, 0.0959384200310377),
        "N": 14,
    },
}
"""
"""
push_regions = {
    "Model 3 cold disk": {
        "sigma_sim": (3.250048448572509e-05, 0.001818878666772),
        "h": (0.0100223245098489, 0.0293384666769982),
        "N": 45,
    },
    "Model 4 hot disk": {
        "sigma_sim": (3.250048448572509e-05, 0.001818878666772),
        "h": (0.1037296389555114, 0.1290548455858325),
        "N": 10,
    },
}
"""
"""
push_regions = {
    "Model 5 broad sweep": {
        "sigma_sim": (3.250048448572509e-05, 0.001818878666772),
        "h": (0.0106414034395569, 0.1820121812634587),
        "N": 25,
    },
}

for name, reg in push_regions.items():

    sigma_cgs = (
        reg["sigma_sim"][0] * MSUN_AU2_TO_CGS,
        reg["sigma_sim"][1] * MSUN_AU2_TO_CGS,
    )

    add_sigma_h_region_to_capture_diagram(
        ax,
        sigma0_range=sigma_cgs,
        h0_range=reg["h"],
        a_grid=np.logspace(-1,1,50),
        mp=1,
        facecolor="cyan",
        edgecolor="cyan",
        alpha=0.1,
        zorder=3,
        label='',
    )

    print(
        name,
        f"N={reg['N']}",
        f"Sigma_cgs=({sigma_cgs[0]:.1f}, {sigma_cgs[1]:.1f}) g/cm^2",
        f"h=({reg['h'][0]:.3f}, {reg['h'][1]:.3f})",
    )
"""

"""
push_regions = {
    "Model 3 cold disk": {
        "sigma_sim": (3.250048448572509e-05, 0.001818878666772),
        "h": (0.0100223245098489, 0.0293384666769982),
        "N": 45,
    },
    "Model 4 hot disk": {
        "sigma_sim": (3.250048448572509e-05, 0.001818878666772),
        "h": (0.1037296389555114, 0.1290548455858325),
        "N": 10,
    },
}


for name, reg in push_regions.items():

    sigma_cgs = (
        reg["sigma_sim"][0] * MSUN_AU2_TO_CGS,
        reg["sigma_sim"][1] * MSUN_AU2_TO_CGS,
    )

    add_sigma_h_region_to_capture_diagram(
        ax,
        sigma0_range=sigma_cgs,
        h0_range=reg["h"],
        a_grid=np.logspace(-1,1,50),
        mp=1,
        facecolor="gold",
        edgecolor="gold",
        alpha=0.3,
        zorder=3,
        label='',
    )

    print(
        name,
        f"N={reg['N']}",
        f"Sigma_cgs=({sigma_cgs[0]:.1f}, {sigma_cgs[1]:.1f}) g/cm^2",
        f"h=({reg['h'][0]:.3f}, {reg['h'][1]:.3f})",
    )
"""
from matplotlib.cm import ScalarMappable

a_norm = LogNorm(0.1,1)

sm = ScalarMappable(
    norm=a_norm,
    cmap='plasma'
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    pad=0.02
)

cbar.set_ticks([0.1,0.3,1])
cbar.set_ticklabels(['0.1','0.3','1'])
cbar.ax.minorticks_off()
cbar.set_label(
    r"$a\,[{\rm AU}]$",
    fontsize=14
)

##################TIME EVOLUTION TRACK ############################

MEARTH_IN_MSUN = 3.003e-6

def add_planet_evolution_track(ax,csv_file,planet_id,label=None,downsample=200,cmap="plasma",norm=None,marker_size=20,lw=1.2,collision_marker=True,runaway_marker=True):
    """
    Add time evolution of one planet in tau_a/P vs K space.

    Uses simulation columns:
        tau_a{i} (yr), tau_e{i} (yr), P{i} (yr),
        a{i} (AU), mass{i} (M_sun)
    """

    df = pd.read_csv(csv_file)

    i = planet_id

    cols = {
        "time": "Time (yr)",
        "a": f"a{i} (AU)",
        "P": f"P{i} (yr)",
        "tau_a": f"tau_a{i} (yr)",
        "tau_e": f"tau_e{i} (yr)",
        "mass": f"mass{i} (M_sun)",
    }

    for c in cols.values():
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    valid = (
        np.isfinite(df[cols["P"]])
        & np.isfinite(df[cols["tau_a"]])
        & np.isfinite(df[cols["tau_e"]])
        & np.isfinite(df[cols["a"]])
        & np.isfinite(df[cols["mass"]])
        & (df[cols["P"]] > 0)
        & (df[cols["tau_e"]] > 0)
    )

    d = df.loc[valid].copy()

    if len(d) == 0:
        print(f"No valid data for planet {planet_id}")
        return None

    # coordinates in capture diagram
    x = d[cols["tau_a"]].values / d[cols["P"]].values
    y = d[cols["tau_a"]].values / d[cols["tau_e"]].values
    a = d[cols["a"]].values
    m_earth = d[cols["mass"]].values / MEARTH_IN_MSUN

    # downsample for plotting
    step = max(1, len(d) // downsample)

    x_plot = x[::step]
    y_plot = y[::step]
    a_plot = a[::step]

    # grey trajectory line
    ax.plot(
        x,
        y,
        color="0.25",
        lw=lw,
        alpha=0.45,
        zorder=75,
    )

    # colored by semimajor axis
    sc = ax.scatter(
        x_plot,
        y_plot,
        c=a_plot,
        cmap=cmap,
        norm=norm,
        s=marker_size,
        edgecolor="none",
        alpha=0.9,
        zorder=90,
    )

    # mark start
    ax.scatter(
        x[0],
        y[0],
        marker="o",
        s=80,
        facecolor="white",
        edgecolor="k",
        linewidth=1.2,
        zorder=110,
    )

    # mark final valid point / collision disappearance
    if collision_marker:
        ax.scatter(
            x[-1],
            y[-1],
            marker="x",
            s=120,
            color="red",
            linewidth=2.2,
            zorder=120,
        )

    # mark first time reaching 10 Earth masses
    if runaway_marker and np.any(m_earth >= 10):

        idx10 = np.argmax(m_earth >= 10)

        ax.scatter(
            x[idx10],
            y[idx10],
            marker="*",
            s=260,
            facecolor="gold",
            edgecolor="k",
            linewidth=1.0,
            zorder=130,
        )

    if label is not None:
        ax.annotate(
            label,
            (x_plot[len(x_plot)//2], y_plot[len(y_plot)//2]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            zorder=140,
        )

    return sc

"""
evo_sc = add_planet_evolution_track(ax,
    csv_file="simulation_outputs_10-25/system_0197.csv",
    planet_id=7,
    label="System 197, planet 7",
    downsample=300,
    cmap="plasma",
    norm=a_norm,
)
"""



def add_mass_lineage_track(ax,csv_file,start_planet_id=None,final_planet_id=None,mp_threshold=10.0,label=None,cmap="plasma",norm=None,downsample=300,marker_size=18):
    df = pd.read_csv(csv_file)

    planet_ids = []
    for c in df.columns:
        if c.startswith("mass") and c.endswith("(M_sun)"):
            pid = int(c.split("mass")[1].split(" ")[0])
            planet_ids.append(pid)
    planet_ids = sorted(planet_ids)

    def mass_col(i): return f"mass{i} (M_sun)"
    def a_col(i): return f"a{i} (AU)"
    def P_col(i): return f"P{i} (yr)"
    def ta_col(i): return f"tau_a{i} (yr)"
    def te_col(i): return f"tau_e{i} (yr)"

    masses = {
        i: df[mass_col(i)].values
        for i in planet_ids
        if mass_col(i) in df.columns
    }

    if start_planet_id is None:
        if final_planet_id is not None:
            start_planet_id = final_planet_id
        else:
            final_masses = np.array([masses[i][-1] for i in planet_ids])
            start_planet_id = planet_ids[np.nanargmax(final_masses)]

    current = start_planet_id
    lineage = [current]

    x_track, y_track, a_track = [], [], []
    t_track, id_track, m_track = [], [], []
    collision_points = []

    has_started = False

    for k in range(len(df) - 1):

        m_now = masses[current][k]
        m_next = masses[current][k + 1]

        # store current point only if physically valid
        if np.isfinite(m_now) and m_now > 0:

            P = df[P_col(current)].iloc[k]
            ta = df[ta_col(current)].iloc[k]
            te = df[te_col(current)].iloc[k]
            a = df[a_col(current)].iloc[k]

            valid_orbit = (
                np.isfinite(P)
                and np.isfinite(ta)
                and np.isfinite(te)
                and np.isfinite(a)
                and P > 0
                and te > 0
            )

            if valid_orbit:
                x_track.append(ta / P)
                y_track.append(ta / te)
                a_track.append(a)
                t_track.append(df["Time (yr)"].iloc[k])
                id_track.append(current)
                m_track.append(m_now / MEARTH_IN_MSUN)

                has_started = True

        # only search for handoff after the lineage has truly started
        disappeared = (
            has_started
            and np.isfinite(m_now)
            and m_now > 0
            and (
                (not np.isfinite(m_next))
                or (m_next <= 0)
                or (m_next < 0.1 * m_now)
            )
        )

        if disappeared:

            best_j = None
            best_score = np.inf

            for j in planet_ids:

                if j == current:
                    continue

                mj_now = masses[j][k]
                mj_next = masses[j][k + 1]

                if not np.isfinite(mj_next) or mj_next <= 0:
                    continue

                if not np.isfinite(mj_now):
                    mj_now = 0.0

                gain = mj_next - mj_now

                if gain <= 0:
                    continue

                score = abs(gain - m_now)

                if score < best_score:
                    best_score = score
                    best_j = j

            if best_j is not None:

                tc = df["Time (yr)"].iloc[k + 1]

                P = df[P_col(best_j)].iloc[k + 1]
                ta = df[ta_col(best_j)].iloc[k + 1]
                te = df[te_col(best_j)].iloc[k + 1]

                if (
                    len(t_track) > 0
                    and tc > t_track[0]
                    and np.isfinite(P)
                    and np.isfinite(ta)
                    and np.isfinite(te)
                    and P > 0
                    and te > 0
                ):
                    collision_points.append((ta / P, ta / te, tc))

                current = best_j
                lineage.append(current)

    x_track = np.array(x_track)
    y_track = np.array(y_track)
    a_track = np.array(a_track)
    t_track = np.array(t_track)
    m_track = np.array(m_track)

    if len(x_track) == 0:
        print("No valid lineage track found.")
        return None, lineage

    # remove any collision markers at or before the first valid point
    collision_points = [
        p for p in collision_points
        if p[2] > t_track[0]
    ]

    step = max(1, len(x_track) // downsample)

    ax.plot(
        x_track,
        y_track,
        color="0.2",
        lw=1.4,
        alpha=0.55,
        zorder=80,
    )

    sc = ax.scatter(
        x_track[::step],
        y_track[::step],
        c=a_track[::step],
        cmap=cmap,
        norm=norm,
        s=marker_size,
        edgecolor="none",
        alpha=0.95,
        zorder=90,
    )

    # start marker
    ax.scatter(
        x_track[0],
        y_track[0],
        marker="o",
        s=90,
        facecolor="white",
        edgecolor="k",
        linewidth=1.2,
        zorder=120,
    )

    # collision markers
    for xc, yc, tc in collision_points:
        ax.scatter(
            xc,
            yc,
            marker="x",
            s=140,
            color="red",
            linewidth=2.3,
            zorder=130,
        )

    # first time above threshold
    if np.any(m_track >= mp_threshold):
        idx = np.argmax(m_track >= mp_threshold)

        ax.scatter(
            x_track[idx],
            y_track[idx],
            marker="*",
            s=300,
            facecolor="gold",
            edgecolor="k",
            linewidth=1.0,
            zorder=140,
        )

    if label is not None:
        mid = len(x_track) // 2
        ax.annotate(
            label,
            (x_track[mid], y_track[mid]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            zorder=150,
        )

    print("Mass lineage:", lineage)

    return sc, lineage
"""
#hot disk run 140:
sc, lineage = add_mass_lineage_track(
    ax,
    csv_file="simulation_outputs_h_0.1/system_0140.csv",
    start_planet_id=9,
    final_planet_id=None,
    label="System 140 mass lineage",
    cmap="plasma",
    norm=a_norm,
)



sc, lineage = add_mass_lineage_track(
    ax,
    csv_file="simulation_outputs_h_0.1/system_0140.csv",
    start_planet_id=20,
    final_planet_id=None,
    label="",
    cmap="plasma",
    norm=a_norm,
)
#hot disk run 137:
sc, lineage = add_mass_lineage_track(
    ax,
    csv_file="simulation_outputs_h_0.1/system_0137.csv",
    start_planet_id=20,
    final_planet_id=None,
    label="System 137 mass lineage",
    cmap="plasma",
    norm=a_norm,
)
#hot disk run 137:
sc, lineage = add_mass_lineage_track(
    ax,
    csv_file="simulation_outputs_h_0.1/system_0137.csv",
    start_planet_id=3,
    final_planet_id=None,
    label="",
    cmap="plasma",
    norm=a_norm,
)


#compact system 105:
sc, lineage = add_mass_lineage_track(
    ax,
    csv_file="simulation_outputs_5-10_h-0.1-0.25/system_0105.csv",
    start_planet_id=20,
    final_planet_id=None,
    label="System 105 mass lineage",
    cmap="plasma",
    norm=a_norm,
)


#cold disk run 197:

sc, lineage = add_mass_lineage_track(
    ax,
    csv_file="simulation_outputs_10-25/system_0197.csv",
    start_planet_id=20,
    final_planet_id=None,
    label="System 197 mass lineage",
    cmap="plasma",
    norm=a_norm,
)

sc, lineage = add_mass_lineage_track(
    ax,
    csv_file="simulation_outputs_10-25/system_0197.csv",
    start_planet_id=16,
    final_planet_id=None,
    label="",
    cmap="plasma",
    norm=a_norm,
)

sc, lineage = add_mass_lineage_track(
    ax,
    csv_file="simulation_outputs_10-25/system_0197.csv",
    start_planet_id=6,
    final_planet_id=None,
    label="",
    cmap="plasma",
    norm=a_norm,
)
"""
"""
event_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="white", markeredgecolor="k",
           label="track start", markersize=8),

    Line2D([0], [0], marker="x", color="red",
           label="collision / disappearance", markersize=9,
           linestyle="None"),

    Line2D([0], [0], marker="*", color="w",
           markerfacecolor="gold", markeredgecolor="k",
           label=r"$M \geq 10\,M_\oplus$", markersize=12,
           linestyle="None"),
]
"""
#ax.legend(handles=event_handles, loc="lower left", fontsize=9, frameon=False)
plt.tight_layout()
plt.show()


# # ============================================================
# # Resonance benchmark parameters
# # ============================================================

# mu_o = 5 * 3e-6
# q = 0.5
# mu_i = q * mu_o

# per_o = 10.0  # days
# n_o = 2 * np.pi / (per_o / 365.25)

# # Independent line ranges

# horiz_x_ranges = {
#     2: (7e5, 5e8),
#     3: (2.5e5, 5e8),
#     4: (1.5e5, 5e8),
#     5: (1e5, 5e8),
#     6: (8e4, 5e8),
#     7:(6.5e4,5e8),


# }

# vert_y_ranges = {
#     2: (1, 1.3e3),
#     3: (1, 7e2),
#     4: (1, 6e2),
#     5: (1, 4.7e2),
#     6: (1, 4.7e2),
#     7: (1,4.7e2),

# }

# diag_x_ranges = {
#     2: (6.8e5, 1e9),
#     3: (2.5e5, 1e9),
#     4: (1.5e5, 1e9),
#     5: (1e5, 1e9),
#     6: (7.7e4, 1e9),
#     7:(6e4,1e9),


# }

# # Plot resonance boundaries

# for i, j in enumerate([2, 3, 4, 5,6,7]):

def Lin2025_overstability_region(j= 3,k=2,mu_o=5.9e-5,q=0.3):

    alpha = ((j - k) / j)**(2/3)

    ## 2. get the Laplacian coeffifcients for j:(j-k) MMR
    f_d_i,f_d_o = np.abs(celmech.disturbing_function.get_fg_coefficients(j,k))


    ## 4. inner-to-outer mass ratio

    mu_i = q * mu_o

    ## 5. period of outer planet
    per_o = 5.8 ## in unit of days
    n_o = 2 * np.pi/(per_o/365.25)
    n_i = n_o * (j / (j - 1)) ** (3 / 2)


    # numerator and denominator of the fraction
    num = f_d_i**2 + (f_d_o**2) * (q**2) * alpha
    den = j * f_d_i**2 + (j - 1) * (f_d_o**2) * q * np.sqrt(alpha)

    # Compute tau_m (slow migration)
    numerator = ((j-1)**2 * mu_o * n_i**2 * alpha + j**2 * mu_i * n_o**2)**(1/3)
    denominator = (3**(1/3)) * (mu_o * n_i * alpha * f_d_i**2 + mu_i * n_o * f_d_o**2)**(2/3)
    prefactor = 1 / ((j-1) * mu_o * n_i * alpha + j * mu_i * n_o)



    h_q = (1 / (1 + q * np.sqrt(alpha))) * (num / den)
    f_q = 1 - ((f_d_o / f_d_i)**2) * (q**2) * alpha

    ratio_tau_a_e_overstable=((3/mu_o)**(2/3))*h_q*((j-1)/(f_d_i*alpha))**(2/3)*f_q

    ratio_tau_a_e_esc=((3/mu_o)**(2/3))*h_q/8*(((j-1)**2+j**2*q)/(f_d_i*alpha+f_d_o*q**2))**(2/3)

    tau_a_slow_mig = 0.5*prefactor * (numerator / denominator)

    # Compute (tau_e * tau_m)_weak,e
    term1 = 1 / (1 + q * np.sqrt(alpha))
    term2 = 1 / (j * f_d_i**2 + (j - 1) * f_d_o**2 * q * np.sqrt(alpha))
    term3 = 1 / ((mu_o * n_i * alpha)**2)

    tau_e_tau_m_weak_e = term1 * term2 * term3
    slope= 0.5*tau_e_tau_m_weak_e
    tau_a_synthetic=np.linspace(0,5e6,1000)
    K_synthetic=slope*tau_a_synthetic
    return ratio_tau_a_e_esc,ratio_tau_a_e_overstable,tau_a_slow_mig,tau_a_synthetic,K_synthetic

# Overstability
print(Lin2025_overstability_region(3, 2))