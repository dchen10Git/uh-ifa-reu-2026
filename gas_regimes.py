import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import rebound
from timescales import tau_t1_mig, tau_gas


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
Sigma_1au = 3400
h_1au = 0.047
alpha = 1
beta = 0

base_params = {
    "m_star": m_star,
    "r_star": r_star,
    "Sigma_1au": Sigma_1au,
    "h_1au": h_1au,
    "alpha": alpha,
    "beta": beta,
    "ide_position": 0.1,
    "ide_width": 0.1 * h_1au**beta,
}

# grid
n_a, n_m = 100, 100
a_grid = np.logspace(np.log10(0.1), np.log10(1), n_a)   # AU
m_grid = np.logspace(-10, -1, n_m)                        # m_earth

tau_mig = np.full((n_m, n_a), np.nan)
tau_drag = np.full((n_m, n_a), np.nan)

def make_particle(m_e, a):
    r = m_e**(1/3) * r_earth   # earth-like bulk density
    m = m_e * m_earth
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(m=m_star, r=r_star, hash='star')
    sim.add(m=m, r=r, a=a)
    return sim.particles[1], m, r

for i, m_e in enumerate(m_grid):
    for j, a in enumerate(a_grid):
        p, m, r = make_particle(m_e, a)
        params = dict(base_params)
        params["m_vals"] = np.array([m])
        params["a_vals"] = np.array([a])
        params["r_vals"] = np.array([r])

        ta_mig = np.array(tau_t1_mig(p, params))[0]
        ta_drag = np.array(tau_gas(p, params))[0]

        tau_mig[i, j] = abs(ta_mig)
        tau_drag[i, j] = abs(ta_drag)
        
combined = 1/(1/tau_mig + 1/tau_drag)

# ratio > 0 (log10) means drag timescale longer -> migration dominates
ratio = np.log10(tau_drag / tau_mig)

fig, ax = plt.subplots(figsize=(8, 6))
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

vmin = np.nanmin([np.log10(tau_mig), np.log10(tau_drag), np.log10(combined)])
vmax = np.nanmax([np.log10(tau_mig), np.log10(tau_drag), np.log10(combined)])

# Contours
contour_levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1)

fig3, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

titles = [r"$\tau_{a, \rm Type \,I}$ [yr]", r"$\tau_{a, \rm gas \,drag}$ [yr]", "Combined"]

for ax, data, title in zip(axes, [tau_mig, tau_drag, combined], titles):
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
cbar_ax = fig3.add_axes([0.90, 0.115, 0.02, 0.765])
cb = fig3.colorbar(c, cax=cbar_ax)
cb.set_label(r"$\log_{10}(\tau)$ [yr]")

plt.show()