import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import astropy.units as u
import rebound
from timescales import tau_t1_mig, tau_gas
from helpers import plot_prettier_lite
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
m2_earth = 5      # outer planet mass, m_earth
m_star = 1.0      # Msun
r_star = 1.5 * r_sun
r2_earth = 10.0   # outer planet radius, r_earth
a1_fixed = 0.5    # AU, matches the a1 used inside get_ta_te

# "overstability": original plot (background = largest overstable-safe k, solid boundaries)
# "adiabaticity":  background = largest k for which adiabatic capture holds, dashed boundaries only
# "both":          overstability background, both solid and dashed boundaries overlaid
criterion_mode = "adiabaticity"

with open("fg_library.pkl", "rb") as fpkl:
    fg_lib = pkl.load(fpkl)

def get_k_params(k):
    """Return the k-dependent constants (alpha_res, m_order, B, R)."""
    alpha_res = ((k - 1) / k) ** (2 / 3)
    m_order = k - 1
    B = 0.8 * m_order
    f_coef, g_coef = fg_lib[(k, 1)]  # first-order resonance
    # delta_{m,1} triggers at k = 2
    if k != 2:
        R = abs(f_coef) / g_coef
    else:
        R = abs(f_coef) / (g_coef - 2 * alpha_res)
    return alpha_res, m_order, B, R

def get_ta_te(alpha_res, m1_earth, h_1au):
    """Build a minimal two-body system (embryo inner, planet outer) at the
    k:k-1 resonance, a1 fixed, and return (ta1, te1, ta2, te2) from tau_t1_mig."""

    ide_position = 0.1
    ide_width = h_1au

    m1 = m1_earth * m_earth
    m2 = m2_earth * m_earth
    a1 = a1_fixed
    a2 = a1 / alpha_res
    r1 = (m1_earth ** (1 / 3)) * r_earth  # embryo radius scaling from fake_sim
    r2 = r2_earth * r_earth

    parameters = {
        "m_vals": np.array([m1, m2]),
        "m_star": m_star,
        "r_vals": np.array([r1, r2]),
        "r_star": r_star,
        "Sigma_1au": Sigma_1au,  # should be given in g/cm^2
        "h_1au": h_1au,
        "alpha": 1,
        "beta": 0,
        "ide_position": ide_position,
        "ide_width": ide_width,
    }

    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(m=m_star, r=r_star, hash='star')
    sim.add(m=m1, r=r1, a=a1, hash='embryo')
    sim.add(m=m2, r=r2, a=a2, hash='planet')

    # Both gas drag and t1mig on embryo
    ta1_mig, te1_mig = -np.array(tau_t1_mig(sim.particles[1], parameters))[0:2]
    ta1_gas, te1_gas = -np.array(tau_gas(sim.particles[1], parameters))[0:2]
    ta1 = 1 / (1 / ta1_mig + 1 / ta1_gas)
    te1 = 1 / (1 / te1_mig + 1 / te1_gas)
    
    # Only t1mig on outer planet
    ta2, te2 = -np.array(tau_t1_mig(sim.particles[2], parameters))[0:2]
    return ta1, te1, ta2, te2

def get_omega1(m1_earth, a1=a1_fixed):
    """Mean motion of the inner body, treating it as a two-body orbit
    around the star (embryo mass included, though it's usually negligible
    next to m_star)."""
    m1 = m1_earth * m_earth
    P1 = np.sqrt(a1 ** 3 / (m_star + m1))  # yr, since G = 4 pi^2 here
    return 2 * np.pi / P1

def adiabaticity_rhs(k, m1_earth, m2_earth):
    """RHS of the compact-orbit adiabatic-capture criterion:
    (tau_a Omega_1) > RHS(k, m1, m2, M)."""
    m1 = m1_earth * m_earth
    m2 = m2_earth * m_earth
    return ((5 * np.pi / 8) * (5 / (36 * k ** 5 * (k - 1) ** (2 / 3)))**(1/3) 
            * (m_star / (m1 + m2))**(4/3)) ** -1

def eps_p_and_crit(alpha_res, m_order, B, R, m1_earth, h_1au):
    ta1, te1, ta2, te2 = get_ta_te(alpha_res, m1_earth, h_1au)
    zeta = m1_earth / m2_earth
    ta = 1 / (1 / ta2 - 1 / ta1)
    te = 1 / (1 / te1 + zeta / te2)
    tae = 1 / (1 / te1 - zeta ** 2 * alpha_res / (R ** 2 * te2))
    eps_p = (m1_earth + m2_earth) * m_earth / m_star
    ratio_e_ae = te / tae
    C = (
        (3 * p_coupling * m_order) / (B * 2 ** 1.5)
        * ratio_e_ae
        * (1 + zeta) ** 2
        / (m_order * (zeta + 1) + p_coupling * ratio_e_ae) ** 1.5
    )
    eps_p_crit = C * (te / ta) ** 1.5  # this might be nan if ta is negative:
                                       # that indicates the inner planet migrates faster, which is possible
    return eps_p, eps_p_crit, ta

# === PARAMETERS ===
# grid: m1 on x, h_1au on y
n_m1, n_h = 100, 100
m1_grid = np.logspace(np.log10(1e-12), np.log10(4), n_m1)  # m_earth
h_1au_grid = np.logspace(np.log10(0.01), np.log10(0.11), n_h)  # aspect ratio

klist = [1, 2, 3, 4, 5, 6, 7]  # 1 indicates overstable for ALL resonances

diffs = {}
eps_crit = {}
adiab_diffs = {}  # RHS - LHS of the adiabaticity criterion; > 0 means adiabatic capture holds
tauomega = {}

for k in klist[1:]:
    alpha_res, m_order, B, R = get_k_params(k)

    diff_k = np.full((n_h, n_m1), np.nan)
    crit_k = np.full((n_h, n_m1), np.nan)
    adiab_diff_k = np.full((n_h, n_m1), np.nan)
    tauomega_k = np.full((n_h, n_m1), np.nan)

    for i, h_1au in enumerate(h_1au_grid):
        for j, m1_earth in enumerate(m1_grid):
            try:
                eps_p, eps_p_crit, ta = eps_p_and_crit(
                    alpha_res, m_order, B, R,
                    m1_earth, h_1au
                )
                diff_k[i, j] = eps_p - eps_p_crit
                crit_k[i, j] = eps_p_crit
            
                omega1 = get_omega1(m1_earth)
                lhs = ta * omega1
                # if np.log10(lhs) <5:
                #     print(m1_earth)
                rhs = adiabaticity_rhs(k, m1_earth, m2_earth)
                tauomega_k[i,j] = lhs
                adiab_diff_k[i, j] = lhs - rhs # if positive, then stable
            except Exception:
                pass

    diffs[k] = diff_k
    eps_crit[k] = crit_k
    adiab_diffs[k] = adiab_diff_k
    tauomega[k] = tauomega_k

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
ax.set_title(rf'{title_prefix} | $p = {p_coupling}$ | $m_2 ={m2_earth} M_\oplus$')

plt.show()

# See min and max tau_a_Omega values
# vals = []

# for k in klist[1:]:
#     mask = adiab_diffs[k] > 0
#     vals.extend(tauomega[k][mask])

# vals = np.log10(np.asarray(vals))

# print(vals.min(), vals.max())