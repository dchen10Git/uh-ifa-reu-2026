import numpy as np
import matplotlib.pyplot as plt

E = 1.211


def drag_coefficient(e, h):
    """Return C_d using tau_gas()'s formula with inc = 0."""
    eta = (11 / 16) * h**2
    mach = ((2 * E / np.pi) * e + eta) / h
    return np.minimum(0.44 + 0.21 * mach**2, 2.0)


# Plot ranges
h_values = np.linspace(0.01, 0.1, 500)
e_values = np.linspace(0.0, 0.4, 500)

h_grid, e_grid = np.meshgrid(h_values, e_values)
cd_grid = drag_coefficient(e_grid, h_grid)

fig, ax = plt.subplots(figsize=(8, 6))

image = ax.pcolormesh(
    h_grid,
    e_grid,
    cd_grid,
    shading="auto",
    cmap="viridis",
    vmin=0.44,
    vmax=2.0,
)

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label(r"$C_d$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"Eccentricity $e$")
ax.set_title(r"Drag coefficient $C_d$ ($\mathrm{inc}=0$)")
ax.set_xlim(h_values[0], h_values[-1])
ax.set_ylim(e_values[0], e_values[-1])

fig.tight_layout()
plt.show()