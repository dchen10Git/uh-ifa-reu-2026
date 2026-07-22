# === IMPORTS ===
import numpy as np
import pandas as pd
from astropy import units as u
from collections import defaultdict

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm)    
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# Obtain and clean up collision log
def get_collision_log(metadata, max_time=None):
    """Collects collision log and converts to pd.DataFrame for later use.

    Args:
        metadata (dict): Metadata from simulation; typically comes from saved_sim[1].
        max_time (float, optional)

    Returns:
        pd.DataFrame: collision_df
    """    
    collision_df = pd.DataFrame(metadata["collision_log"])
    collision_df["time"] = collision_df["time"] / 1000  # kyr

    if max_time is not None:
        collision_df = collision_df[collision_df["time"] <= max_time]

    collision_df["time"] = collision_df["time"].round(2)
    collision_df = collision_df.rename(columns={"time": "time (kyr)"})

    collision_df["victim mass"] = (collision_df["victim mass"] / m_earth).round(4)
    collision_df = collision_df.rename(columns={"victim mass": "victim mass (m_earth)"})

    return collision_df.sort_values("time (kyr)")

# Obtain and clean up particle fates
def get_particle_fates(metadata, max_time=None):
    collision_df = get_collision_log(metadata, max_time)

    fates = {name: "alive" for name in metadata["particle_fate"]}

    for _, row in collision_df.iterrows():
        fates[row["victim"]] = "collided"

    return pd.Series(fates, name="outcome").sort_index().to_frame()

# Summarize accretion data
def summarize_accretion(metadata, collision_df, fates_df, percent=True):
    """Summarizes accretion information.

    Args:
        collision_df (pandas.DataFrame): DataFrame of collision log.
        percent (bool, optional): Whether to show fraction as percentage. 
            Defaults to True.

    Returns:
        pandas.DataFrame: Dataframe containing accretion summary.
    """    
    children = defaultdict(list)

    for _, row in collision_df.iterrows():
        children[row["survivor"]].append(row["victim"])

    def get_all_accreted(body):
        """Returns every particle ultimately accreted by body."""
        accreted = []

        for child in children.get(body, []):
            accreted.append(child)
            accreted.extend(get_all_accreted(child))

        return accreted

    alive_particles = [
        name
        for name, outcome in fates_df["outcome"].items()
        if outcome == "alive"
    ]

    rows = []

    for particle in alive_particles:
        accreted = get_all_accreted(particle)

        n_embryos = sum("embryo" in p for p in accreted)
        n_ptsmls = sum("ptsml" in p for p in accreted)

        rows.append({
            "particle": particle,
            "n_ems": n_embryos,
            "n_ptsmls": n_ptsmls,
            "embryo %": n_embryos / metadata['num_em'] if metadata["num_em"] != 0 else 0,
            "ptsml %": n_ptsmls / metadata['num_ptsml'] if metadata["num_ptsml"] != 0 else 0,
        })

    alive_accretion_df = (pd.DataFrame(rows).set_index("particle").sort_index())

    # Keep only particles that actually accreted something
    alive_accretion_df = alive_accretion_df[
        alive_accretion_df["n_ems"] + alive_accretion_df["n_ptsmls"] > 0
    ].sort_values('n_ptsmls', ascending=False) # & sort by ptsmls accreted

    # Remaining unaccreted bodies
    n_alive_ems = metadata['num_em'] - alive_accretion_df["n_ems"].sum()
    n_alive_ptsmls = metadata['num_ptsml'] - alive_accretion_df["n_ptsmls"].sum()

    alive_accretion_df.loc["Survived"] = {
        "n_ems": n_alive_ems,
        "n_ptsmls": n_alive_ptsmls,
        "embryo %": n_alive_ems / metadata['num_em'] if metadata["num_em"] != 0 else 0,
        "ptsml %": n_alive_ptsmls / metadata['num_ptsml']if metadata["num_ptsml"] != 0 else 0,
    }

    if percent:
        # Convert to %
        alive_accretion_df["embryo %"] = (alive_accretion_df["embryo %"]*100).round(2)
        alive_accretion_df["ptsml %"] = (alive_accretion_df["ptsml %"]*100).round(2)

    return alive_accretion_df
