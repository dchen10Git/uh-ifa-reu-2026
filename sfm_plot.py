"""
Plot the Second Fundamental Model (SFM) resonance diagram for two chosen bodies
(e.g. 'embryo 0' and 'planet b') from a reb_sims simulation output.

Why this bypasses plot_ell2SFM(df):
plot_ell2SFM() auto-detects resonant pairs via get_p_by_pair(), which reads a
DataFrame under a fixed 8-columns-per-planet layout (mean longitude, period, k, h,
mass at fixed offsets 1+8*I, 2+8*I, 3+8*I, 4+8*I, 7+8*I -- see samples2ell_twoplanets'
int-indexed branch). That layout comes from the DACE/GRSW export format and isn't
documented elsewhere, so faking it for reb_sims output means guessing undocumented
column positions.

plot_ell() is the function plot_ell2SFM calls once it has found a pair, and it takes
raw orbital element arrays directly -- no schema to match. It also prints the
delta-criterion resonance fraction (% of samples with IsResonant flag set) alongside
the plot, using the same topology/delta logic as plot_ell2SFM.

Units, confirmed against rebound_sims.py / run_sim.py:
- sim_data[name] columns 'e', 'P' (years), 'l' (mean longitude, rad), 'pomega'
  (longitude of periapsis, rad) come straight from rebound's orb = rock.orbit(...),
  so no unit conversion or reconstruction from Omega/omega/M is needed.
- metadata['m_vals'] is already in solar masses -- it's built as
  `np.array([...]) * m_earth` before being stored in `parameters`.
- metadata['m_em'] and metadata['m_ptsml'] are stored RAW, in Earth masses
  (used unconverted elsewhere, e.g. for r_em/r_ptsml), unlike m_vals. They need
  `* m_earth` to become solar masses -- easy to get this backwards since m_vals
  and m_em look like they should be in the same units but aren't.
- metadata['num_pl'] gives how many of ['planet b','planet c','planet d'] are
  active in this run, used to index into m_vals.
- Absolute period units don't matter (only P2/P1 and the resonance ratio do),
  since ell2SFM normalizes T1 to 1 internally.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from resonantstate.ell2SFM import plot_ell, plot_topology
from resonantstate.simulations_resonance_analysis import get_nearest_resonance
import rebound_sims as reb_sims
import mmr_id

import warnings
warnings.filterwarnings('ignore')

m_earth = u.Mearth.to(u.Msun)  # Earth mass in solar masses

def get_body_mass_ratio(name, metadata):
    """Mass ratio m_body / m_star."""
    m_star = metadata['m_star']  # already in Msun
    if name.startswith('planet'):
        planet_names = ['planet b', 'planet c', 'planet d'][:metadata['num_pl']]
        idx = planet_names.index(name)
        m_body = metadata['m_vals'][idx]  # already in Msun
    elif name.startswith('embryo'):
        m_body = metadata['m_em'] * m_earth  # stored in Earth masses -> convert
    elif name.startswith('ptsml'):
        m_body = metadata['m_ptsml'] * m_earth  # stored in Earth masses -> convert
    else:
        raise ValueError(f"Unrecognized body name: {name}")
    return m_body / m_star

def get_orbital_elements(df):
    """Extract (e, pomega, mean longitude, period) time series from a sim_data[name] DataFrame."""
    return df['e'].values, df['pomega'].values, df['l'].values, df['P'].values

def build_sfm_inputs(dataset_id, sim_id, name1, name2, base_dir=None, snapshot=None):
    """
    snapshot: None to keep the full time series (default), or a row index
    (e.g. -1 for the last saved output) to plot a single state.
    """
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    sim_data, metadata = reb_sims.load_simulation_run(file_path)

    e1, vp1, lbd1, P1 = get_orbital_elements(sim_data[name1])
    e2, vp2, lbd2, P2 = get_orbital_elements(sim_data[name2])

    if snapshot is not None:
        e1, vp1, lbd1, P1 = (np.atleast_1d(x[snapshot]) for x in (e1, vp1, lbd1, P1))
        e2, vp2, lbd2, P2 = (np.atleast_1d(x[snapshot]) for x in (e2, vp2, lbd2, P2))

    # Order inner/outer by period so the p:p+1 convention (name1=inner, name2=outer) holds
    if np.nanmedian(P1) > np.nanmedian(P2):
        name1, name2 = name2, name1
        e1, e2 = e2, e1
        vp1, vp2 = vp2, vp1
        lbd1, lbd2 = lbd2, lbd1
        P1, P2 = P2, P1

    m1 = get_body_mass_ratio(name1, metadata)
    m2 = get_body_mass_ratio(name2, metadata)
    
    # variables needed for mmr_id.find_best_twoBR_pq
    m_star = metadata['m_star'] 
    b = sim_data[name1]
    c = sim_data[name2]

    return dict(m_star=m_star, b=b, c=c, e1=e1, e2=e2, vp1=vp1, vp2=vp2, m1=m1, m2=m2, P1=P1, P2=P2,
                lbd1=lbd1, lbd2=lbd2, name1=name1, name2=name2)


def plot_delta_criterion(dataset_id, sim_id, name1, name2, base_dir=None, snapshot=None):
    inp = build_sfm_inputs(dataset_id, sim_id, name1, name2, base_dir, snapshot=snapshot)

    # NOTE: Only first-order MMRs are considered for this implementation. Higher-orders MMRs will not be plotted correctly and will raise an error.
    P_ratio = np.nanmedian(inp['P2']) / np.nanmedian(inp['P1'])
    p, order, dist = get_nearest_resonance(P_ratio, second_order=False, kmax=12, difference_order=0.2)
    print(f"{inp['name1']} / {inp['name2']}: P2/P1 = {P_ratio:.4f}, "
          f"nearest first-order resonance p = {p} ({p+1}:{p}), distance = {dist:.4f}")
    
    p_actual, q_actual = mmr_id.find_best_twoBR_pq(inp['m_star'], inp['b'], inp['c'], snapshot=-1)
    if p_actual - q_actual != 1:
        raise ValueError(f"Detected resonance ({p_actual}:{q_actual}), order {p_actual - q_actual} != 1; only first-order MMRs are supported for this plotting code.")

    fig, ax = plt.subplots(figsize=(9, 9))
    plot_ell(fig, ax, inp['e1'], inp['e2'], inp['vp1'], inp['vp2'],
             inp['m1'], inp['m2'], inp['P1'], inp['P2'], inp['lbd1'], inp['lbd2'],
             pair=(inp['name1'], inp['name2']), p=p, colors='tab:blue', color_lim=None,
             label=f"{inp['name1']}-{inp['name2']}")
    plot_topology(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax

if __name__ == '__main__':
    assert len(sys.argv) == 3
    dataset_id = sys.argv[1]
    sim_id = sys.argv[2]
    embryo_name = 'embryo 9'
    
    # plot_delta_criterion(dataset_id, sim_id, name1='planet b', name2='planet c', snapshot=np.arange(-5, 0)) 
    plot_delta_criterion(dataset_id, sim_id, name1='planet b', name2=embryo_name, snapshot=np.arange(-3, 0)) 
    plot_delta_criterion(dataset_id, sim_id, name1=embryo_name, name2='planet c', snapshot=np.arange(-3, 0))