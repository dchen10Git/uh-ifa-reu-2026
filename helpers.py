import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm)    
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU)

def plot_prettier(save_dpi=600, fig_dpi=200, fontsize=8, usetex=False): 
    '''
    Make plots look nicer compared to Matplotlib defaults
    Parameters: 
        dpi - int, "dots per inch" - controls resolution of PNG images that are produced
                by Matplotlib
        fontsize - int, font size to use overall
        usetex - bool, whether to use LaTeX to render fonds of axes labels 
                use False if you don't have LaTeX installed on your system
    '''
    plt.rcParams['figure.dpi']= fig_dpi
    plt.rc("savefig", dpi=save_dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    if usetex:
        plt.rc('text', usetex=usetex)
    else:
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        
def plot_prettier_lite(fig_dpi=200, save_dpi=500, fontsize=8):
    plt.rcParams['figure.dpi']= fig_dpi
    plt.rc("savefig", dpi=save_dpi)
    plt.rc('font', size=fontsize)
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def get_omega(m_star, m, a):
    """Mean motion of the inner body, treating it as a two-body orbit
    around the star (embryo mass included, though it's usually negligible
    next to m_star). Units should be in AU, yr, Msun. Returns Omega in 1/yr."""
    P1 = np.sqrt(a ** 3 / (m_star + m))  # yr, as long as G = 4 pi^2
    return 2 * np.pi / P1