import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from helpers import plot_prettier
plot_prettier()

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm)    
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# Constants that never change
m2 = 5 * m_earth
M = 1
s = 1

chi_a = 1/(2.7 + 1.1*s)
chi_e = 1/0.780

x_min, x_max = 4.2, 6.5
y_min, y_max = 1.2, 7

logK2 = np.linspace(y_min, y_max, 600)
K2 = 10**logK2

# Figure
fig, ax = plt.subplots(figsize=(6,4.5))

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel(r"$\log_{10}(\tau_a\Omega_1)$")
ax.set_ylabel(r"$\log_{10}(\mathcal{K}_2)$")
ax.grid(alpha=0.3)

# Animation
m1_values = np.logspace(np.log10(0.005), np.log10(1.0), 120) * m_earth

def update(frame):

    ax.clear()

    m1 = m1_values[frame]
    zeta = m1/m2
    
    A = (chi_a / chi_e) / (1 - zeta)

    def logK2_to_hr(logK2):
        return np.sqrt(A / 10**logK2)

    def hr_to_logK2(hr):
        return np.log10(A / hr**2)

    ax2 = ax.secondary_yaxis(
        'right',
        functions=(logK2_to_hr, hr_to_logK2)
    )

    ax2.set_ylabel(r"$h/r$")
    ax2.set_yticks(np.round(np.logspace(-3, -1, 8), 3))

    K1 = K2 * zeta

    def adiabaticity_crit(k):
        return (
            (5*np.pi/8)
            * (5/(36*k**5*(k-1)**(2/3)))**(1/3)
            * (M/(m1+m2))**(4/3)
        )

    def dissipative_crit(k):

        return (
            (m2/M)
            * np.sqrt(
                32*k**3*(1+zeta)*(zeta*K1 + K2)
                /(25*K1*K2)
            )
        )

    def overstability_K2_crit(k):

        lhs_base = (
            zeta
            * (k*(1+zeta)-2*zeta)
            /(1-zeta)
            * (chi_a/chi_e)
        )

        A = lhs_base**1.5
        B = (15/32)*(1-zeta**2)/m2

        hcrit = (A/B)**(1/3)

        return chi_a/chi_e/(1-zeta)*hcrit**-2

    # Plot all resonances
    x_adia_prev = x_max
    for k in range(3,8):

        color = f"C{k-3}"

        C_adia = adiabaticity_crit(k)
        C_diss = dissipative_crit(k)

        x_adia = np.log10(C_adia)
        x_diss = -np.log10(C_diss)

        transition_idx = np.argmax(x_diss >= x_adia)

        if x_diss[transition_idx] < x_adia:
            y_transition = y_max
        else:
            y_transition = logK2[transition_idx]

        y_over = np.log10(overstability_K2_crit(k))

        y_adia_start = max(y_min, y_over)

        if y_adia_start < y_transition:
            ax.vlines(
                x=x_adia,
                ymin=y_adia_start,
                ymax=min(y_transition, y_max),
                lw=2,
                color=color,
                label=rf"{k}:{k-1}"
            )
            ax.vlines(
                x=x_adia,
                ymin=y_min,
                ymax=y_adia_start,
                lw=2,
                color=color,
                ls=":"
            )
    
        mask = logK2 >= max(y_transition, y_over)

        ax.plot(
            x_diss[mask],
            logK2[mask],
            lw=2,
            color=color,
        )

        if y_min <= y_over <= y_max:
            ax.hlines(
                y=y_over,
                xmin=x_adia,
                xmax=x_adia_prev,
                lw=2,
                color=color,
                ls=":",
            )
    
            x_adia_prev = x_adia

            ax.plot(
                x_adia,
                y_over,
                "o",
                color=color,
                ms=4
            )

    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)

    ax.set_xlabel(r"$\log_{10}(\tau_a\Omega_1)$")
    ax.set_ylabel(r"$\log_{10}(\mathcal{K}_2)$")

    ax.legend(title="MMR")
    ax.grid(True)

    ax.set_title(
        rf"$m_1={m1/m_earth:.3f}\ M_\oplus,\ "
        rf"m_2={m2/m_earth:.1g}\ M_\oplus$"
    )

fps = 20
pause = 10 # 1 second
frames = (
    [0] * pause +
    list(range(len(m1_values))) +
    [len(m1_values)-1] * pause
) # allow for pausing one second before and after

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=50,
    repeat=True,
)

# Save as gif
writer = PillowWriter(fps=20)
ani.save("m1_animation.gif", writer=writer)

plt.show()
