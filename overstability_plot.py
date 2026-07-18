# NOTE: Code is a bit broken.

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import astropy.units as u
from resonance_criteria import *
from helpers import get_omega, plot_prettier_lite
plot_prettier_lite(dpi=600)

# Unit conversions
AU = u.AU.to(u.cm)
Msun = u.Msun.to(u.g)
m_earth = u.Mearth.to(u.Msun)
r_earth = u.earthRad.to(u.AU)
r_sun = u.Rsun.to(u.AU)
G = 4 * np.pi ** 2  # AU, yr, Msun units

# fixed parameters
p_coupling = 2    # p = 2 roughly matches B&M26
Sigma_1au = 1700  # g/cm^2 (this doesn't matter)
m2 = 5 * m_earth     # outer planet mass
m_star = 1.0      # Msun
r_star = 1.5 * r_sun
r2_earth = 10.0 * r_earth   # outer planet radius, r_earth
a1_fixed = 0.5    # AU, matches the a1 used inside get_ta_te

# "overstability": original plot (background = largest overstable-safe k, solid boundaries)
# "adiabaticity":  background = largest k for which adiabatic capture holds, dashed boundaries only
# "both":          overstability background, both solid and dashed boundaries overlaid
criterion_mode = "overstability"

with open("fg_library.pkl", "rb") as fpkl:
    fg_lib = pkl.load(fpkl)

# === PARAMETERS ===
# grid: m1 on x, h_1au on y
n_m1, n_h = 100, 100
m1_grid = np.logspace(np.log10(1e-12), np.log10(4), n_m1) * m_earth # inner planet mass
h_1au_grid = np.logspace(np.log10(0.01), np.log10(0.11), n_h) # aspect ratio

klist = [1, 2, 3, 4, 5, 6, 7]  # 1 indicates overstable for ALL resonances

diffs = {}
eps_crit = {}
adiab_diffs = {}  # RHS - LHS of the adiabaticity criterion; > 0 means adiabatic capture holds
tau_Omega = {}

for k in klist[1:]:
    alpha_res, m_order, B, R = get_k_params(k)

    diff_k = np.full((n_h, n_m1), np.nan)
    crit_k = np.full((n_h, n_m1), np.nan)
    adiab_diff_k = np.full((n_h, n_m1), np.nan)
    tau_Omega_k = np.full((n_h, n_m1), np.nan)

    parameters = {
        "Sigma_1au": Sigma_1au,
        "alpha": 1,
        "beta": 0,
        "ide_position": 0.1,
    }
    
    for i, h_1au in enumerate(h_1au_grid):
        
        parameters['h_1au'] = h_1au
        parameters['ide_width'] = h_1au
        
        for j, m1 in enumerate(m1_grid):
            r1 = (m1 / m_earth) ** (1/3) * r_earth  # inner planet radius, r_earth
            r2 = r2_earth
            a1 = a1_fixed
            a2 = a1 * (k / (k - 1)) ** (2 / 3)  # exact resonance
            
            try:
                eps_p, eps_p_crit, ta = eps_p_and_crit(k, parameters, m_star, m1, m2, r1, r2, a1, a2)
                diff_k[i, j] = eps_p - eps_p_crit
                crit_k[i, j] = eps_p_crit
            
                omega1 = get_omega(m_star, m1, a1_fixed)
                lhs = ta * omega1
                rhs = adiabaticity_crit(k, m_star, m1, m2) ** -1
                tau_Omega_k[i,j] = lhs
                adiab_diff_k[i, j] = lhs - rhs # if positive, then stable
            except Exception:
                pass

    diffs[k] = diff_k
    eps_crit[k] = crit_k
    adiab_diffs[k] = adiab_diff_k
    tau_Omega[k] = tau_Omega_k

# background grid:
# gray = no data at all
# otherwise color corresponding to largest k satisfying whichever criterion is selected

def k_background(grids, criterion='largest'):
    """grids: dict k -> 2D array where >0 means the criterion is satisfied at that k.
    Returns a background array of the largest such k per cell (1 if none, NaN if no data)."""
    background = np.full((n_h, n_m1), np.nan)
    for i in range(n_h):
        for j in range(n_m1):
            if all(np.isnan(grids[k][i, j]) for k in klist[1:]):
                continue
            chosen = np.nan
            if criterion == "largest":
                for k in reversed(klist[1:]):
                    if grids[k][i, j] > 0:
                        chosen = k
                        break

            elif criterion == "smallest":
                for k in klist[1:]:
                    if grids[k][i, j] > 0:
                        chosen = k
                        break
                
            if np.isnan(chosen):
                chosen = 1
            background[i, j] = chosen
    return background

if criterion_mode == "adiabaticity":
    k_bg = k_background(adiab_diffs, criterion='smallest')
    background_label = "Non-adiabatic"
    colorbar_title = r'Smallest $k$ with adiabatic capture'
else:
    k_bg = k_background(diffs, criterion='largest')
    background_label = "Overstable"
    colorbar_title = r'Resonance index $k$'

fig, ax = plt.subplots(figsize=(7, 6))

cmap = plt.cm.get_cmap('inferno', len(klist)).copy()
cmap.set_bad('lightgray')     # NaN -> gray

im = ax.pcolormesh(
    m1_grid,
    h_1au_grid,
    k_bg,
    shading='nearest',
    cmap=cmap,
    vmin=min(klist) - 0.5,
    vmax=max(klist) + 0.5,
)

# Overlay boundaries
colors = cmap(np.linspace(0, 1, len(klist) + 1))[1:]

legend_handles = []
for k, color in zip(klist[1:], colors):
    if criterion_mode in ("overstability", "both"):
        ax.contour(
            m1_grid,
            h_1au_grid,
            diffs[k],
            levels=[0],
            colors=[color],
            linewidths=1.5
        )
        legend_handles.append(
            plt.Line2D([0], [0], color=color, lw=1.5,
                       label=f'{k}:{k-1}')
        )

    if criterion_mode in ("adiabaticity", "both"):
        ax.contour(
            m1_grid,
            h_1au_grid,
            adiab_diffs[k],
            levels=[0],
            colors=[color],
            linewidths=1.2,
            linestyles='dashed' if criterion_mode == "both" else 'solid',
        )
        if criterion_mode == "adiabaticity":
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=1.2, label=f'{k}:{k-1}')
            )

if criterion_mode == "both":
    legend_handles.append(
        plt.Line2D([0], [0], color='k', lw=1.2, linestyle='dashed',
                   label='adiabatic capture boundary')
    )

cbar = plt.colorbar(im, ax=ax)
cbar.set_ticks(klist)
cbar.set_ticklabels([background_label] + klist[1:])
cbar.set_label(colorbar_title)

legend_title = {
    "overstability": "MMR boundary (overstability)",
    "adiabaticity": "MMR boundary (adiabatic capture)",
    "both": "MMR boundary (solid: overstability, dashed: adiabaticity)",
}[criterion_mode]

ax.legend(handles=legend_handles,
          title=legend_title,
          loc='upper left',
          fontsize=7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$m_1$ [$m_\oplus$]')
ax.set_ylabel(r'$h/r$')
# ax.set_xticks([3,4,5])
# ax.set_xticklabels(['3', '4', '5'])
ax.set_yticks([0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20])
ax.set_yticklabels(['0.01', '0.02', '0.04', '0.06', '0.08', '0.10', '0.15', '0.20'])
ax.set_ylim(h_1au_grid[0], h_1au_grid[-1])
ax.minorticks_off()
ax.tick_params(axis='y', right=True)
title_prefix = {
    "overstability": "Overstability boundaries",
    "adiabaticity": "Adiabatic capture boundaries",
    "both": "Overstability & adiabaticity boundaries",
}[criterion_mode]
ax.set_title(rf'{title_prefix} | $p = {p_coupling}$ | $m_2 ={m2} M_\oplus$')

plt.show()

# See min and max tau_a_Omega values
# vals = []

# for k in klist[1:]:
#     mask = adiab_diffs[k] > 0
#     vals.extend(tau_Omega[k][mask])

# vals = np.log10(np.asarray(vals))

# print(vals.min(), vals.max())