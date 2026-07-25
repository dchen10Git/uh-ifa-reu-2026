import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
import rebound
from timescales import get_ta_te
import warnings
warnings.filterwarnings('ignore')

from helpers import plot_prettier_lite
plot_prettier_lite(save_dpi=600, fontsize=9)

AU = u.AU.to(u.cm)    
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# fixed disk / star parameters for this study
m_star = 1.0
r_star = 1.5 * r_sun
Sigma_1au = 1700
h_1au = 0.03
alpha = 1
beta = 0

parameters = {
    "m_star": m_star,
    "r_star": r_star,
    "Sigma_1au": Sigma_1au,
    "h_1au": h_1au,
    "alpha": alpha,
    "beta": beta,
    "ide_position": 0.1,
    "ide_width": h_1au
}

# grid
n_a, n_m = 500, 100
a_grid = np.logspace(np.log10(0.10), np.log10(1), n_a)   # [AU]
m_grid = np.logspace(-8, -0.5, n_m)                       # [m_earth]

tau_a_mig = np.full((n_m, n_a), np.nan)
tau_a_drag = np.full((n_m, n_a), np.nan)

tau_e_mig = np.full((n_m, n_a), np.nan)
tau_e_drag = np.full((n_m, n_a), np.nan)

for i, m_e in enumerate(m_grid):
    r_e = (m_e)**(1/3) * r_earth # [AU]

    for j, a in enumerate(a_grid):
        ts_mig = np.array(get_ta_te(parameters, m_star, m_e*m_earth, r_e, a, tau='mig', ide=False))
        ts_gas = np.array(get_ta_te(parameters, m_star, m_e*m_earth, r_e, a, tau='gas', ide=False))
        
        tau_a_mig[i, j] = abs(ts_mig[0])
        tau_a_drag[i, j] = abs(ts_gas[0])
        
        tau_e_mig[i, j] = abs(ts_mig[1])
        tau_e_drag[i, j] = abs(ts_gas[1])
        
combined_a = 1/(1/tau_a_mig + 1/tau_a_drag)
combined_e = 1/(1/tau_e_mig + 1/tau_e_drag)

# ratio > 0 (log10) means drag timescale longer -> migration dominates
ratio = np.log10(tau_a_drag / tau_a_mig)

fig, ax = plt.subplots(figsize=(5, 3))
c = ax.pcolormesh(a_grid, m_grid, ratio, cmap="RdBu_r", shading="auto",
                   vmin=-3, vmax=3)
ax.contour(a_grid, m_grid, ratio, levels=[0], colors="k", linewidths=2)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("a [AU]")
ax.set_ylabel(r"m [$M_\oplus$]")
cb = fig.colorbar(c, ax=ax)
cb.set_label(r"$\log_{10}(\tau_{a, \rm gas}/\tau_{a, \rm mig})$")
ax.set_title("Gas drag (blue) vs Type I migration (red) dominant regime")

fig.tight_layout()
plt.show()

# # === crossover mass as a function of a ===
# m_crit = np.full(n_a, np.nan)
# for j in range(n_a):
#     col = ratio[:, j]
#     sign_change = np.where(np.diff(np.sign(col)))[0]
#     if len(sign_change):
#         k = sign_change[0]
#         # linear interp in log-log space
#         x0, x1 = col[k], col[k+1]
#         y0, y1 = np.log10(m_grid[k]), np.log10(m_grid[k+1])
#         m_crit[j] = 10**(y0 + (0 - x0) * (y1 - y0) / (x1 - x0))

# fig2, ax2 = plt.subplots(figsize=(7, 5))
# ax2.plot(a_grid, m_crit, 'k-')
# ax2.set_xscale("log")
# ax2.set_yscale("log")
# ax2.set_xlabel("a [AU]")
# ax2.set_ylabel(r"$m_{\rm crit}$ [$M_\oplus$]")
# ax2.set_title("Transition mass: drag vs Type I")
# fig2.tight_layout()
# plt.show()

# === Colormaps of individual timescales ===


vmin = np.nanmin([np.log10(tau_a_mig), np.log10(tau_a_drag), np.log10(combined_a)])
vmax = np.nanmax([np.log10(tau_a_mig), np.log10(tau_a_drag), np.log10(combined_a)])

# Contours
contour_levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1)

fig3, axes = plt.subplots(1, 3, figsize=(8, 3.5), sharey=True)

titles = [r"$\tau_{a, \rm Type \,I}$ (yr)", r"$\tau_{a, \rm gas \,drag}$ (yr)", "Combined (yr)"]

for ax, data, title in zip(axes, [tau_a_mig, tau_a_drag, combined_a], titles):
    c = ax.pcolormesh(a_grid, m_grid, np.log10(data), cmap="viridis",
                       shading="auto", vmin=vmin, vmax=vmax)
    
    cs = ax.contour(
        a_grid, m_grid, np.log10(data),
        levels=contour_levels,
        colors="white",
        linewidths=0.8,
        alpha=0.8
    )

    ax.clabel(cs, fmt=r"$10^{%d}$", fontsize=8)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("a [AU]")
    ax.set_title(title)
    
axes[0].set_ylabel(r"m [$M_\oplus$]")

fig3.subplots_adjust(right=0.88)
fig3.suptitle(f"$\Sigma_0$: {Sigma_1au:.0f} g/cm$^2$, $h_0$: {h_1au:.3f}")

cbar_ax = fig3.add_axes([0.90, 0.115, 0.02, 0.765])
cb = fig3.colorbar(c, cax=cbar_ax)
cb.set_label(r"$\log_{10}(\tau)$ [yr]")

plt.show()


# === Combined a ===
vmin = np.nanmin(np.log10(combined_a))
vmax = np.nanmax(np.log10(combined_a))
contour_levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1)

fig4, ax = plt.subplots(figsize=(3.6, 3.2))

title = rf"Combined $\tau_a$ ($\Sigma_0={Sigma_1au:.0f}$ g/cm$^2$, $h_0={h_1au:.3f}$)"

c = ax.pcolormesh(a_grid, m_grid, np.log10(combined_a), cmap="viridis", shading="auto", vmin=vmin, vmax=vmax,)

cs = ax.contour(a_grid, m_grid, np.log10(combined_a), levels=contour_levels, colors="white", linewidths=0.8, alpha=0.8,)

ax.clabel(cs, fmt=r"$10^{%d}$", fontsize=8)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("a [AU]")
ax.set_ylabel(r"m [$M_\oplus$]")
ax.set_title(title)

# Create a colorbar axis attached to the main axes
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.08)

cb = fig4.colorbar(c, cax=cax)
cb.set_label(r"$\log_{10}(\tau)$ [yr]")
plt.tight_layout()
plt.show()

# === Combined e ===
vmin = np.nanmin(np.log10(combined_e))
vmax = np.nanmax(np.log10(combined_e))
contour_levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1)

fig4, ax = plt.subplots(figsize=(3.6, 3.2))

title = rf"Combined $\tau_e$ ($\Sigma_0={Sigma_1au:.0f}$ g/cm$^2$, $h_0={h_1au:.3f}$)"

c = ax.pcolormesh(a_grid, m_grid, np.log10(combined_e), cmap="viridis", shading="auto", vmin=vmin, vmax=vmax,)

cs = ax.contour(a_grid, m_grid, np.log10(combined_e), levels=contour_levels, colors="white", linewidths=0.8, alpha=0.8,)

ax.clabel(cs, fmt=r"$10^{%d}$", fontsize=8)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("a [AU]")
ax.set_ylabel(r"m [$M_\oplus$]")
ax.set_title(title)

# Create a colorbar axis attached to the main axes
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.08)

cb = fig4.colorbar(c, cax=cax)
cb.set_label(r"$\log_{10}(\tau)$ [yr]")
plt.tight_layout()
plt.show()

# Line (fixed AU)

a_fix = 0.5  # AU
j_fix = np.argmin(np.abs(a_grid - a_fix))

fig5, ax = plt.subplots(figsize=(4, 3.2))

ax.plot(m_grid, tau_a_mig[:, j_fix], label=r"$\tau_{a,\rm mig}$", color="tab:red")
ax.plot(m_grid, tau_a_drag[:, j_fix], label=r"$\tau_{a,\rm gas}$", color="tab:blue")
ax.plot(m_grid, combined_a[:, j_fix], label="Combined", color="k", ls="--")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_{\rm em}$ [$M_\oplus$]")
ax.set_ylabel(r"$\tau_a$ [yr]")
# ax.set_title(f"a = {a_grid[j_fix]:.2f} AU")
ax.legend(frameon=False)

fig5.tight_layout()
plt.show()

# Line (fixed AU)

a_fix = 0.5  # AU
j_fix = np.argmin(np.abs(a_grid - a_fix))

fig5, ax = plt.subplots(figsize=(4, 3.2))

ax.plot(m_grid, tau_e_mig[:, j_fix], label=r"$\tau_{e,\rm mig}$", color="tab:red")
ax.plot(m_grid, tau_e_drag[:, j_fix], label=r"$\tau_{e,\rm gas}$", color="tab:blue")
ax.plot(m_grid, combined_e[:, j_fix], label="Combined", color="k", ls="--")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_{\rm em}$ [$M_\oplus$]")
ax.set_ylabel(r"$\tau_e$ [yr]")
ax.legend(frameon=False)

fig5.tight_layout()
plt.show()