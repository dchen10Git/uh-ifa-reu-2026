import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import astropy.units as u
import rebound

from timescales import tau_t1_mig, tau_gas

# Unit conversions
AU = u.AU.to(u.cm)
Msun = u.Msun.to(u.g)
m_earth = u.Mearth.to(u.Msun)
r_earth = u.earthRad.to(u.AU)
r_sun = u.Rsun.to(u.AU)

# fixed parameters
p_coupling = 2    # p = 2 roughly matches B&M26
Sigma_1au = 621  # g/cm^2
a1 = 0.10366          # AU (does not significantly affect the plot)
m2_earth = 5      # outer planet mass, m_earth
m_star = 1.0      # Msun
r_star = 1.5 * r_sun
r2_earth = 10.0   # outer planet radius, r_earth

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
    m1 = m1_earth * m_earth
    m2 = m2_earth * m_earth
    a2 = a1 / alpha_res

    r1 = (m1_earth ** (1 / 3)) * r_earth  # embryo radius scaling from fake_sim
    r2 = r2_earth * r_earth

    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(m=m_star, r=r_star, hash='star')
    sim.add(m=m1, r=r1, a=a1, hash='embryo')
    sim.add(m=m2, r=r2, a=a2, hash='planet')

    parameters = {
        "m_vals": np.array([m1, m2]),
        "m_star": m_star,
        "r_vals": np.array([r1, r2]),
        "r_star": r_star,
        "a_vals": np.array([a1, a2]),
        "Sigma_1au": Sigma_1au,  # should be given in g/cm^2
        "h_1au": h_1au,
        "alpha": 1,
        "beta": 0,
        "ide_position": 0.1,
        "ide_width": h_1au,
    }

    # Both gas drag and t1mig on embryo
    ta1_mig, te1_mig = -np.array(tau_t1_mig(sim.particles[1], parameters))[0:2]
    ta1_gas, te1_gas = -np.array(tau_gas(sim.particles[1], parameters))[0:2]
    ta1 = 1/(1/ta1_mig + 1/ta1_gas)
    print(ta1_mig, te1_gas,ta1)
    
    te1 = 1/(1/te1_mig + 1/te1_gas)
    ta2, te2 = -np.array(tau_t1_mig(sim.particles[2], parameters))[0:2]
    return ta1, te1, ta2, te2

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
    eps_p_crit = C * (te / ta) ** 1.5
    return eps_p, eps_p_crit

# grid: m1 on x, h_1au on y
n_m1, n_h = 80, 80
# m1_grid = np.logspace(np.log10(1e-13), np.log10(1e-9), n_m1) # m_earth
# h_1au_grid = np.logspace(np.log10(0.06), np.log10(0.11), n_h) # aspect ratio

m1_grid = np.logspace(np.log10(3), np.log10(6), n_m1) # m_earth
h_1au_grid = np.logspace(np.log10(0.01), np.log10(0.11), n_h) # aspect ratio

klist = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]

diffs = {}
eps_crit = {}

for k in klist[1:]:
    alpha_res, m_order, B, R = get_k_params(k)

    diff_k = np.full((n_h, n_m1), np.nan)
    crit_k = np.full((n_h, n_m1), np.nan)

    for i, h_1au in enumerate(h_1au_grid):
        for j, m1_earth in enumerate(m1_grid):
            try:
                eps_p, eps_p_crit = eps_p_and_crit(
                    alpha_res, m_order, B, R,
                    m1_earth, h_1au
                )
                diff_k[i, j] = eps_p - eps_p_crit
                crit_k[i, j] = eps_p_crit
            except Exception:
                pass

    diffs[k] = diff_k
    eps_crit[k] = crit_k
    
# background grid:
# gray = no eps_p_crit
# otherwise color corresponding to largest stable resonance

k_background = np.full((n_h, n_m1), np.nan)

for i in range(n_h):
    for j in range(n_m1):

        # If every resonance failed, leave gray
        if all(np.isnan(eps_crit[k][i, j]) for k in klist[1:]):
            continue

        # Choose which resonance to display.
        # Here: largest k satisfying eps_p > eps_p_crit.
        chosen = np.nan
        for k in klist[::-1]:
            if k == 0:
                continue
            if diffs[k][i, j] > 0:
                chosen = k

        # If none are stable, use the smallest resonance
        if np.isnan(chosen):
            chosen = 0

        k_background[i, j] = chosen
        
fig, ax = plt.subplots(figsize=(7, 6))

cmap = plt.cm.get_cmap('inferno', len(klist)).copy()
cmap.set_bad('lightgray')     # NaN -> gray

im = ax.pcolormesh(
    m1_grid,
    h_1au_grid,
    k_background,
    shading='nearest',
    cmap=cmap,
    vmin=min(klist)-0.5,
    vmax=max(klist)+0.5,
)

# Overlay boundaries
colors = cmap(np.linspace(0, 1, len(klist)+1))[1:]

legend_handles = []
for k, color in zip(klist[1:], colors):
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

cbar = plt.colorbar(im, ax=ax)
cbar.set_ticks(klist)
cbar.set_label(r'Resonance index $k$')

ax.legend(handles=legend_handles,
          title='MMR boundary',
          loc='upper left',
          fontsize=8)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$m_1$ [$m_\oplus$]')
ax.set_ylabel(r'$h/r$')
ax.set_yticks([0.01, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.set_yticklabels(['0.01', '0.02', '0.04', '0.06', '0.08', '0.10'])
ax.minorticks_off()
ax.set_ylim(0.06, 0.11)
ax.tick_params(axis='y', right=True)
ax.set_title(rf'Overstability boundaries | $a_1$={a1} AU | $p$={p_coupling} | $m_2$={m2_earth} $M_\oplus$')

plt.show()