# === IMPORTS ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

from astropy import units as u
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
from functools import reduce

import mmr_id
import rebound_sims as reb_sims
from param_grid_plots import plot_param_grid_multi, plot_param_grid_map
from accretion import *

import sys
import warnings
warnings.filterwarnings('ignore')

from helpers import *
plot_prettier_lite(save_dpi=600, fig_dpi=200)

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm)    
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# First-order resonance classes (up to 7:6)
def classify_resonance(P_ratio, librates):
    """Classify first-order resonances of simulations up to 7:6. 
    If planets are not librating, the result will be "not in resonance".
    If planets are in higher-order or higher-index resonance, the result
    will be "other".


    Args:
        P_ratio (float): (Assigned) period ratio.
        librates (bool): Whether resonant angle librates.

    Returns:
        Classification of the resonance.
    """    


    if not librates or np.isnan(P_ratio):
        return "not in resonance"

    resonances = {
        "2:1": 2/1,
        "3:2": 3/2,
        "4:3": 4/3,
        "5:4": 5/4,
        "6:5": 6/5,
        "7:6": 7/6
    }

    for name, ratio in resonances.items():
        if P_ratio == ratio:
            return name

    return "other"

# P ratios, libration, and Delta between planets
def get_planet_outcomes(dataset_id, sim_ids, pomega='mixed', amp_threshold=90):
    rows = []
    base_dir = Path.cwd()
    for sim_id in sim_ids:
        file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"
        
        try:
            saved_sim = reb_sims.load_simulation_run(file_path)
        except FileNotFoundError:
            continue

        sim_data, metadata = saved_sim
        
        # Check planet survival
        alive_planets = []
        for name in ['planet b', 'planet c', 'planet d'][:metadata['num_pl']]:
            if not np.isnan(sim_data[name]['a'].iloc[-1]):
                alive_planets.append(name)

        # Check both pairs if all planets are alive
        if len(alive_planets) == 3:
            c = alive_planets[1]; d = alive_planets[2]
            
            # Get P_ratios by direct division
            true_outer_P_ratio = sim_data[d]['P'].iloc[-1]/sim_data[c]['P'].iloc[-1]
            
            # Swap planets if inner became outer
            if true_outer_P_ratio < 1:
                c, d = d, c
                true_outer_P_ratio = 1/true_outer_P_ratio
            
            # Check Delta (find p, q such that Delta is closest to 0)
            outer_best = mmr_id.find_best_twoBR_pq(metadata['m_star'], sim_data[c], sim_data[d])
            outer_P_ratio = outer_best[0]/outer_best[1]
            outer_Delta = true_outer_P_ratio/outer_P_ratio - 1
            
            # Check libration of 2BR angle
            outer_librates = (mmr_id.check_resonance(metadata['m_star'], sim_data[c], sim_data[d], pomega=pomega, N=25, amp_threshold=amp_threshold) != (0,0))
        
        else:
            outer_P_ratio = np.nan
            true_outer_P_ratio = np.nan
            outer_Delta = np.nan
            outer_librates = False
        
        # Check ratio of remaining pair, call it inner pair
        if len(alive_planets) >=2:
            b = alive_planets[0]; c = alive_planets[1] # this will be the right pair regardless
                
            # Get P_ratios by direct division
            true_inner_P_ratio = sim_data[c]['P'].iloc[-1]/sim_data[b]['P'].iloc[-1]
            
            # Swap planets if inner became outer
            if true_inner_P_ratio < 1:
                b, c = c, b
                true_inner_P_ratio = 1/true_inner_P_ratio
                
            # Check Delta (find p, q such that Delta is closest to 0)
            inner_best = mmr_id.find_best_twoBR_pq(metadata['m_star'], sim_data[b], sim_data[c])
            inner_P_ratio = inner_best[0]/inner_best[1]
            inner_Delta = true_inner_P_ratio/inner_P_ratio - 1
            
            # Check libration of 2BR angle
            inner_librates = (mmr_id.check_resonance(metadata['m_star'], sim_data[b], sim_data[c], pomega=pomega, amp_threshold=amp_threshold) != (0,0))

        else:
            inner_P_ratio = np.nan
            true_inner_P_ratio = np.nan
            inner_Delta = np.nan
            inner_librates = False
        
        if len(alive_planets) >= 1:
            m_pl = metadata['m_vals'][0]
            a = 0.65
            P = a**(3/2) # for M = 1
            Sigma = metadata['Sigma_1au']*(AU**2 / Msun) * a**-metadata['alpha']
            h = metadata['h_1au'] * a**metadata['beta']
            tau_a = 1/(2.7+1.1*metadata['alpha']) / m_pl / (Sigma * a**2) * h**2 / (2*np.pi / P) # at 0.65 AU
            log_tau_Omega = np.log10(tau_a * 2*np.pi / (a**(3/2))) # at 0.65 AU
            
            chi_a = 1 / (2.7 + 1.1 * metadata['alpha'])
            chi_e = 1 / 0.780
            log_K2 = np.log10(chi_a/chi_e * h**-2) # for inner ptsml and outer planet
        else:
            tau_a = np.nan
            log_tau_Omega = np.nan
            log_K2 = np.nan

        # Create df to store data
        Sigma_1au = metadata['Sigma_1au']
        h_1au = metadata['h_1au']
        
        rows.append({
            "sim_id": sim_id,
            "Sigma_1au": Sigma_1au,
            "h_1au": h_1au,
            "alive_planets": alive_planets,
            "inner_P_ratio": inner_P_ratio,
            "outer_P_ratio": outer_P_ratio,
            "true_inner_P_ratio": round(true_inner_P_ratio, 3),
            "true_outer_P_ratio": round(true_outer_P_ratio, 3),
            "inner_P_Delta (%)": round(inner_Delta*100, 1),
            "outer_P_Delta (%)": round(outer_Delta*100, 1),
            "inner_librates": inner_librates,
            "outer_librates": outer_librates,
            "inner_res_class": classify_resonance(inner_P_ratio, inner_librates),
            "outer_res_class": classify_resonance(outer_P_ratio, outer_librates),
            "pl_survived": len(alive_planets),
            "tau_a (kyr)": tau_a/1000,
            "log_tau_Omega": log_tau_Omega,
            "log_K2": log_K2
        })
        
    outcomes = pd.DataFrame(rows)
    return outcomes

# Sigma_1au, h_1au, log_tau_Omega, log_K2
def disk_params(dataset_id, sim_ids):
    rows = []
    base_dir = Path.cwd()
    for sim_id in sim_ids:
        file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"
        
        try:
            saved_sim = reb_sims.load_simulation_run(file_path)
        except FileNotFoundError:
            continue

        sim_data, metadata = saved_sim
        
        # Get tau_a_Omega and K2
        m_pl = metadata['m_vals'][0]
        m_em = metadata['m_em'] * m_earth
        a1 = 0.5 # middle of disk
        P1 = a1**(3/2) # for M = 1
        P2 = P1#*2 # or whatever ratio
        a2 = P2**(2/3)
        
        Sigma1 = metadata['Sigma_1au']*(AU**2 / Msun) * a1**-metadata['alpha']
        h1 = metadata['h_1au'] * a1**metadata['beta']
        Sigma2 = metadata['Sigma_1au']*(AU**2 / Msun) * a2**-metadata['alpha']
        h2 = metadata['h_1au'] * a2**metadata['beta']

        tau_a1 = 1/(2.7+1.1*metadata['alpha']) / m_em / (Sigma1 * a1**2) * h1**2 / (2*np.pi / P1) 
        tau_a2 = 1/(2.7+1.1*metadata['alpha']) / m_pl / (Sigma2 * a2**2) * h2**2 / (2*np.pi / P2) 
        tau_a = (tau_a2**(-1)-tau_a1**(-1))**(-1)
        
        log_tau_Omega = np.log10(tau_a * 2*np.pi / (a1**(3/2))) 
        
        chi_a = 1 / (2.7 + 1.1 * metadata['alpha'])
        chi_e = 1 / 0.780
        tau_e2 = chi_e/chi_a * h2**2 * tau_a2
        log_K2 = np.log10(tau_a / tau_e2)
        
        rows.append({
            "sim_id": sim_id,
            "Sigma_1au": metadata['Sigma_1au'],
            "h_1au": metadata['h_1au'],
            "tau_a (kyr)": tau_a/1000,
            "log_tau_Omega": round(log_tau_Omega, 2), # prevent errors in plotting
            "log_K2": round(log_K2, 2)
        })
    
    outcomes = pd.DataFrame(rows)
    return outcomes
        
# P ratio, libration, and Delta between one planet and one embryo (unused)
def pl_em_res(dataset_id, sim_ids, pomega='mixed', amp_threshold=90, snapshot=-1):
    rows = []
    base_dir = Path.cwd()
    for sim_id in sim_ids:
        file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"
        
        try:
            saved_sim = reb_sims.load_simulation_run(file_path)
        except FileNotFoundError:
            continue

        sim_data, metadata = saved_sim

        # find alive embryos
        alive_embryos = []
        for name in [f"embryo {i}" for i in range(metadata['num_em'])]:
            if not np.isnan(sim_data[name]['a'].iloc[snapshot]):
                alive_embryos.append(name)
                
        if len(alive_embryos) >= 1:
            for embryo_name in alive_embryos:
                b = embryo_name
                c = 'planet c'

            # Get P_ratios by direct division
            true_inner_P_ratio = sim_data[c]['P'].iloc[snapshot]/sim_data[b]['P'].iloc[snapshot]
            
            # Don't consider embryos outside of planet c's orbit
            if true_inner_P_ratio < 1:
                continue
                    
            # Check libration of 2BR angle
            inner_librates = (mmr_id.check_resonance(metadata['m_star'], sim_data[b], sim_data[c], pomega=pomega, amp_threshold=amp_threshold) != (0,0))
            if inner_librates:
                resonant_embryos += 1
        else:
            resonant_embryos = 0
        
        rows.append({
            "sim_id": sim_id,
            "resonant_embryos": resonant_embryos     
        })
        
    outcomes = pd.DataFrame(rows)
    return outcomes

# Find embryos AND planetesimals between b and c; detect resonance with inner planet
def embryos_between_planets(dataset_id, sim_ids, inner='planet b', outer='planet c', N=15, amp_threshold=90, snapshot=-1):
    rows = []
    base_dir = Path.cwd()
    for sim_id in sim_ids:
        file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"
        
        try:
            saved_sim = reb_sims.load_simulation_run(file_path)
        except FileNotFoundError:
            continue

        sim_data, metadata = saved_sim
    
        a_inner = sim_data[inner]["a"].iloc[-1]
        a_outer = sim_data[outer]["a"].iloc[-1]
        
        inside_found = 0
        inside_names = []

        for name, df in sim_data.items():
            if not (
                name.startswith("embryo")
                or name.startswith("ptsml")
            ):
                continue

            a_em = df["a"].iloc[snapshot]

            if np.isnan(a_em):
                continue

            if a_inner*1.02 <= a_em <= a_outer*0.98:  # 2% buffer to avoid edge cases (co-orbital embryos)
                inside_found += 1
                inside_names.append(name)
        
        b_resonant_embryos = 0
        b_resonant_names = []
        c_resonant_embryos = 0
        c_resonant_names = []
        
        if inside_found >= 1:
            for embryo_name in inside_names:
                
                # Check libration of 2BR angle of inner pair
                inner_librates = (mmr_id.check_resonance(metadata['m_star'], sim_data[inner], sim_data[embryo_name], N=N, pomega='mixed', amp_threshold=amp_threshold) != (0,0))
                if inner_librates:
                    b_resonant_names.append(embryo_name)
                    b_resonant_embryos += 1
                
                # Check libration of 2BR angle of outer pair
                outer_librates = (mmr_id.check_resonance(metadata['m_star'], sim_data[embryo_name], sim_data[outer], N=N, pomega='mixed', amp_threshold=amp_threshold) != (0,0))
                if outer_librates:
                    c_resonant_names.append(embryo_name)
                    c_resonant_embryos += 1
        
        rows.append({
            "sim_id": sim_id,
            "embryos_inside": inside_found,
            "inside_names": inside_names,
            "b_resonant_embryos": b_resonant_embryos,
            "b_resonant_names": b_resonant_names,
            "c_resonant_embryos": c_resonant_embryos,
            "c_resonant_names": c_resonant_names,
        })
    
    outcomes = pd.DataFrame(rows)
    return outcomes

# Embryo mass
def additional_params(dataset_id, sim_ids):
    rows = []
    base_dir = Path.cwd()
    for sim_id in sim_ids:
        file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"
        
        try:
            saved_sim = reb_sims.load_simulation_run(file_path)
        except FileNotFoundError:
            continue

        sim_data, metadata = saved_sim
        
        rows.append({
            "sim_id": sim_id,
            "m_em": metadata['m_em']
        })
    
    outcomes = pd.DataFrame(rows)
    return outcomes

# Planetesimal (and embryo) accretion data
def get_ptsml_outcomes(dataset_id, sim_ids, snapshot=None):
    rows = []
    
    for sim_id in sim_ids:
        base_dir = Path.cwd()
        file_path = base_dir.parent / f"sim_results/dataset{dataset_id}" / f"sim{sim_id}.h5"
        
        try:
            saved_sim = reb_sims.load_simulation_run(file_path)
        except FileNotFoundError:
            # print(f"Sim {sim_id} not found")
            continue
        
        sim_data, metadata = saved_sim
        
        # Get survival fraction
        if snapshot:
            snapshot_time = sim_data["planet b"]["time"].iloc[snapshot] / 1000  # kyr
        else:
            snapshot_time = None
        # print(f"Processing Sim {sim_id}", end='\r', flush=True)
        
        if not metadata['collision_log']: # no collisions occurred
            em_surv_rate, ptsml_surv_rate, ejec_rate = 100, 100, 0
        else:
            collision_df = get_collision_log(metadata, snapshot_time)
            fates_df = get_particle_fates(metadata, snapshot_time)
            summary = summarize_accretion(metadata, collision_df, fates_df)
            em_surv_rate = summary['embryo %']['Survived']
            ptsml_surv_rate = summary['ptsml %']['Survived']
            
            # Get ejection fraction (both embryos and planetesimals; ejected due to being removed from sim)
            ejec_rate = fates_df['outcome'].str.contains('ejected').sum()/len(fates_df)

        # Find median of semimajor axes of embryos
        em_a_finals = [sim_data[f'embryo {i}']["a"].iloc[-1] for i in range(metadata['num_em'])]
        em_a_med = np.nanmedian(em_a_finals)
        
        # Find median e of embryos
        em_e_finals = [sim_data[f'embryo {i}']["e"].iloc[-1] for i in range(metadata['num_em'])]
        em_e_med = np.nanmedian(em_e_finals)

        # Find median of semimajor axes of ptsmls
        ptsml_a_finals = [sim_data[f'ptsml {i}']["a"].iloc[-1] for i in range(metadata['num_ptsml'])]
        ptsml_a_med = np.nanmedian(ptsml_a_finals)
        
        # Find median e of ptsmls
        ptsml_e_finals = [sim_data[f'ptsml {i}']["e"].iloc[-1] for i in range(metadata['num_ptsml'])]
        ptsml_e_med = np.nanmedian(ptsml_e_finals)

        rows.append({
            "sim_id": sim_id,
            "em_surv_rate (%)": em_surv_rate,
            "ptsml_surv_rate (%)": ptsml_surv_rate,
            "ejec_rate (%)": ejec_rate,
            "ptsml_a_median": ptsml_a_med,
            "ptsml_e_median": ptsml_e_med,
            "em_a_median": em_a_med,
            "em_e_median": em_e_med,
        })
        
    outcomes = pd.DataFrame(rows)
    return outcomes


if __name__ == "__main__":
    dataset_id = '2pl_AdaptiveDrag'
    snapshot = -1
    sim_ids = list(range(9600))

    outcomes_path = Path(f"dfs/outcomes{dataset_id}_{snapshot}.h5")
    sim_dir = Path.cwd().parent / f"sim_results/dataset{dataset_id}"

    merge = outcomes_path.exists()
    existing_outcomes = pd.read_hdf(outcomes_path, key="df") if merge else None
    already_analyzed = set(existing_outcomes['sim_id']) if merge else set()

    # Sims whose output file isn't on disk yet (not run, or still running)
    missing = {s for s in sim_ids if not (sim_dir / f"sim{s}.h5").exists()}

    new_sim_ids = [s for s in sim_ids if s not in already_analyzed and s not in missing]

    print(f"{len(already_analyzed)} sims already analyzed")
    print(f"{len(new_sim_ids)} new sims to analyze")
    print(f"{len(missing)} sims have no output file yet (of {len(sim_ids)} total)")

    sim_ids = new_sim_ids
    if len(sim_ids) == 0:
        print("No new sims to analyze")
        sys.exit()

    disk_df = disk_params(dataset_id=dataset_id, sim_ids=sim_ids)
    print("  Disk df collected", flush=True)
    embryo_df = embryos_between_planets(dataset_id=dataset_id, sim_ids=sim_ids, inner='planet b', outer='planet c', amp_threshold=120, snapshot=snapshot)
    print("  Embryos df collected", flush=True)
    additional_df = additional_params(dataset_id=dataset_id, sim_ids=sim_ids)
    print("  Additional df collected", flush=True)
    accretion_df = get_ptsml_outcomes(dataset_id=dataset_id, sim_ids=sim_ids, snapshot=snapshot)
    print("  Accretion df collected", flush=True)

    merged_outcomes = reduce(lambda left, right: pd.merge(left, right, on='sim_id', how='outer'), [disk_df, embryo_df, additional_df, accretion_df])

    # Drop unneeded columns
    merged_outcomes = merged_outcomes.drop(columns=[
                                                    'ptsml_surv_rate (%)',
                                                    'ejec_rate (%)',
                                                    'ptsml_a_median',
                                                    'ptsml_e_median'
                                                ])

    if merge:
        merged_outcomes = pd.concat([existing_outcomes, merged_outcomes], ignore_index=True)
        print("Merged with existing outcomes")
    merged_outcomes.to_hdf(outcomes_path, key="df", mode="w") # Save to disk
    print(f"  {len(merged_outcomes)} total sims analyzed", flush=True)