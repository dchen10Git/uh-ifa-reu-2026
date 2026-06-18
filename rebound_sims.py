# === IMPORTS ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rebound
import reboundx
from astropy import units as u
import re
import warnings
warnings.filterwarnings('ignore')

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm) # cm per AU
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) # g per Msun
yr = u.yr.to(u.s) # s per yr
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# === DATA MANAGEMENT ===
    
def concatenate_data(stages):
    if type(stages) == dict:
        return stages
    
    # Concatenate first two
    all_stage_data = {
            name: pd.concat(
                [stages[0][name], stages[1][name]],
                ignore_index=True
            )
            for name in stages[0]
        }
    
    # Concatenate the rest
    for i in range(2, len(stages)):
        next_stages = stages[i]
        all_stage_data = {
            name: pd.concat(
                [all_stage_data[name], next_stages[name]],
                ignore_index=True
            )
            for name in all_stage_data
        }
    return all_stage_data
    
def save_simulation_run(stage_data, sim_id, file_path, sim_metadata=None):
    """Save all planets from one simulation run into HDF5.
    
    Parameters:
        stage_data (dict): Data in a dict in the form {planet_name: DataFrame}.
        sim_id (int): Simulation ID.
        file_path (str): File path for data storage.
        sim_metadata (dict, optional): e.g. {"m_star": 1.0, "integrator": "whfast"}
    """
    with pd.HDFStore(file_path, mode="a") as store:
        # Save planet list
        planet_list = list(stage_data.keys())
        store.put("planet_list",
                  pd.Series(planet_list))
        
        # Save simulation metadata
        if sim_metadata is not None:
            store.put("metadata",
                      pd.Series(sim_metadata))
        
        # Save each planet
        for planet_name, df in stage_data.items():
            
            key = f"{planet_name}"
            store.put(key, df, format="table")
            
            # Attach attributes
            storer = store.get_storer(key)
            storer.attrs.planet_name = planet_name
            storer.attrs.sim_id = sim_id
    
    print("Simulation saved")
    
def load_simulation_run(file_path):
    '''Load a simulation for analysis.
    
    Parameters:
        file_path (str): Location of simulation data (saved as hdf5)
    
    Returns:
        tuple:
            * result (dict[str, pd.DataFrame]): Loaded simulation.
            * metadata (dict): Metadata containing additional information (planet_names, sim_id, IDE params, etc.).
    
    '''
    result = {}
    
    with pd.HDFStore(file_path, mode="r") as store:
        
        planet_list = store["planet_list"].tolist()
        
        for planet_name in planet_list:
            key = f"{planet_name}"
            df = store[key]
            
            # Pull HDF5 attributes
            storer = store.get_storer(key)
            df.attrs["planet_name"] = storer.attrs.planet_name
            df.attrs["sim_id"] = storer.attrs.sim_id
            
            result[planet_name] = df
        
        metadata = store[f"metadata"].to_dict()
    
    return result, metadata

# === RUNNING THE SIM ===

def tau_gas(rock, parameters):
    """Calculates damping timescales given parameters for one rock.
    Based on formulas from Adachi 1976. Should be applied on planetesimals.
    Note that negative tau indicates damping for modify_orbits_forces.

    Args:
        rock (rebound.particle.Particle): REBOUND particle object to 
        calculate damping timescales for.
        parameters (dict): Dictionary containing planet and star parameters.

    Returns:
        tuple: Tuple containing tau_a, tau_e, tau_i for the given object,
        in units of years
    """                                     
    Sigma_1au, h_1au, beta = parameters['Sigma_1au'], parameters['h_1au'], parameters['beta']
    
    m_p = rock.m * u.Msun.to(u.g) # convert to g
    r = rock.d * AU # distance to the star, converted to cm
    R_p = rock.r * AU # cm # radius, converted to cm
    e = rock.e
    inc = rock.inc
    Omega_k = 2*np.pi/rock.P / yr # Keplerian orbital frequency, converted to 1/s
        
    h = h_1au * (r/AU)**beta # scale height in AU
    v_K = r*Omega_k # cm/s
    
    rho_p = m_p / (4/3 * np.pi * R_p**3) # Ptsml density (mass / vol) in g/cm^3
    C_d = 0.44 # for km-sized ptsmls and small Mach number (see Brasser 2007) 
    rho_g = Sigma_1au/(np.sqrt(np.pi)*(h_1au*AU)) * (r/AU)**(-11/4) # g/cm^3
    eta = (11/16)*(h*AU/r)**2
    
    K = 2.157
    E = 1.211
    alpha_rho = 3 # In Adachi, gas density index (not surface density)
    
    t_0 = (8*rho_p*R_p) / (3*C_d*rho_g*v_K) / yr # yr
    t_a = t_0 / (2 * ((2*(2*E+K)/(3*np.pi)*e + 2/np.pi*inc + eta) * eta + 
                        (((2*E+K)/(9*np.pi)) * alpha_rho + (68*E-11*K)/(54*np.pi))* (e**3) +
                        (inc**3)/(2*np.pi))) # Adachi 1976 Eq. 4.11
    t_e = t_0 / ((2*E)/(np.pi)*e + 2/np.pi*inc + eta) # Adachi 1976 Eq. 4.12
    t_i = t_0 / (1/2 * (2*E/np.pi*e + 8/(3*np.pi)*inc + eta)) # Adachi 1976 Eq. 4.13
    
    return -t_a, -t_e, -t_i # Negative so damping
 
def integrate_sim(sim, sim_id, rock_names, parameters, years, particle_fate, hash_to_name, start_time=0):
    '''Integrates a REBOUND simulation over a given number of years and
    saves the new state of the sim.
    
    Parameters:
        sim (rebound.Simulation): Simulation object to integrate.
        sim_id: ID of the simulation to keep track.
        rock_names (list[str]): List containing names of the rocks (non-stellar particles).
        parameters (dict): Dictionary containing planet and star parameters.
        years (float): Number of years to integrate.
        particle_fate (dict): Dictionary containing particle fates for keeping track.
        hash_to_name (dict): Dictionary containing rebound hashes as keys and rock names as values. 
        start_time (float, optional): Start time of the integration, defaults to 0.
        
    Returns:
        tuple:
            * stage_data (pd.Dataframe): Simulation data.
            * completed_sim (bool): Whether the integration was fully compelted.
    '''
    m_vals, r_vals, a_vals = parameters["m_vals"], parameters["r_vals"], parameters["a_vals"]
    pebble_flux, m_ptsml, r_ptsml = parameters['pebble_flux'], parameters['m_ptsml'], parameters['r_ptsml']
    num_pl, num_em, num_ptsml = parameters["num_pl"], parameters["num_em"], parameters["num_ptsml"]
    num_rocks = len(rock_names)
    
    # Set up times for integration & data collection
    n_out = 1000 # number of data points to collect
    stage_times = np.linspace(start_time, years+start_time, n_out, endpoint=False)  # all times to integrate over
    tau_pl = years / 6 # planet formation timescale (for 3 planets)
    
    # Remove outer planet(s) at the beginning
    if num_pl > 1:
        sim.remove(hash='planet c')
        c_added = False
    if num_pl > 2:
        sim.remove(hash='planet d')
        d_added = False
    
    max_ptsml_added = int(pebble_flux * years)  # upper bound over the whole run
    num_rocks_max = num_rocks + max_ptsml_added
    pebble_accumulator = 0.0
    
    hist = {
        "time": stage_times,
        "a": np.full((n_out, num_rocks_max), np.nan),
        "e": np.full((n_out, num_rocks_max), np.nan),
        "inc": np.full((n_out, num_rocks_max), np.nan),
        "P": np.full((n_out, num_rocks_max), np.nan),
        "l": np.full((n_out, num_rocks_max), np.nan),
        "pomega": np.full((n_out, num_rocks_max), np.nan),
        "tau_a": np.full((n_out, num_rocks_max), np.nan),
        "tau_e": np.full((n_out, num_rocks_max), np.nan),
        "tau_i": np.full((n_out, num_rocks_max), np.nan),
    }
        
    sim.random_seed = 16 # For reproducibility
   
    removed_names = set() # To keep track of removed rocks during sim

    for i, t in enumerate(stage_times):
        if num_pl > 1:
            if not c_added and t > tau_pl:
                sim.add(m=m_vals[1], r=r_vals[1], a=a_vals[1], hash=rock_names[1], primary=sim.particles[0])
                c_added = True
                sim.N_active += 1
                hash_to_name[int(sim.particles[-1].hash.value)] = rock_names[1]
                removed_names.discard('planet c')
        if num_pl > 2:
            if not d_added and t > 2*tau_pl:
                sim.add(m=m_vals[2], r=r_vals[2], a=a_vals[2], hash=rock_names[2], primary=sim.particles[0])
                d_added = True
                sim.N_active += 1
                hash_to_name[int(sim.particles[-1].hash.value)] = rock_names[2]
                removed_names.discard('planet d')

        alive_rock_names = []
        min_P = 1 # period at 1 AU

        for j, name in enumerate(rock_names):
            if name in removed_names:
                continue
            try:
                rock = sim.particles[name]
            except rebound.ParticleNotFound:
                removed_names.add(name)
                continue

            orb = rock.orbit(primary=sim.particles[0]) # single conversion, reused below

            if orb.a < 0.01 or orb.a > 100:
                fate = 'fell into star (d < 0.01)' if orb.a < 0.01 else 'ejected from system (d > 100)'
                particle_fate[name] = fate
                hist["a"][i,j], hist["e"][i,j], hist["inc"][i,j] = orb.a, orb.e, orb.inc
                hist["P"][i,j], hist["l"][i,j], hist["pomega"][i,j] = orb.P, orb.l, orb.pomega
                sim.remove(hash=name)
                removed_names.add(name)
                print(f"{name} removed; {fate}")
                continue

            if 'ptsml' in name:
                tau_a, tau_e, tau_i = tau_gas(rock, parameters)
                rock.params["tau_a"] = tau_a
                rock.params["tau_e"] = tau_e
                rock.params["tau_inc"] = tau_i
                hist["tau_a"][i,j], hist["tau_e"][i,j], hist["tau_i"][i,j] = tau_a, tau_e, tau_i

            hist["a"][i,j], hist["e"][i,j], hist["inc"][i,j] = orb.a, orb.e, orb.inc
            hist["P"][i,j], hist["l"][i,j], hist["pomega"][i,j] = orb.P, orb.l, orb.pomega

            alive_rock_names.append(name)
            if orb.P < min_P:
                min_P = orb.P

        if not alive_rock_names:
            print(f"Sim {sim_id:<2d} | All particles removed")
            break
        
        sim.dt = min_P / 30

        print(f"Sim {sim_id:<2d} | Step {i} of {len(stage_times)}      ", end="\r", flush=True)
        
        try:
            sim.integrate(t)
        except RuntimeError:
            print(f"\nSim {sim_id} | NaN at step {i}, t={t:.1f} yr")
            print(f"  alive: {alive_rock_names}")
            for name in alive_rock_names:
                try:
                    p = sim.particles[name]
                    orb = p.orbit(primary=sim.particles[0])
                    print(f"  {name}: a={orb.a:.4f}, e={orb.e:.4f}")
                except:
                    print(f"  {name}: orbit computation failed")
            break # Abort simulation

        # Replenish planetesimals
        pebble_accumulator += pebble_flux * (years / n_out)
        num_added = int(pebble_accumulator)
        pebble_accumulator -= num_added

        if num_added > 0:
            ptsml_locs = np.random.uniform(0.4, 0.9, size=num_added)
            for k in range(num_added):
                new_name = f"ptsml {num_ptsml + k}"
                rock_names.append(new_name)
                particle_fate[new_name] = "alive" # track fate for new ptsmls
                sim.add(m=m_ptsml*m_earth, r=r_ptsml*r_earth, a=ptsml_locs[k],
                        hash=new_name, primary=sim.particles[0],
                        M=np.random.uniform(0, 2*np.pi),
                        inc=np.random.uniform(1e-4, 1e-3))
                hash_to_name[int(sim.particles[-1].hash.value)] = new_name # register hash to name
            num_ptsml += num_added
            sim.move_to_com()
        
    # Convert to df
    stage_data = {}
    for j, name in enumerate(rock_names):
        stage_data[name] = pd.DataFrame({
            "time": hist["time"],
            "a": hist["a"][:, j],
            "e": hist["e"][:, j],
            "inc": hist["inc"][:, j],
            "P": hist["P"][:, j],
            "l": hist["l"][:, j],
            "pomega": hist["pomega"][:, j],
            "tau_a": hist["tau_a"][:, j],
            "tau_e": hist["tau_e"][:, j],
            "tau_i": hist["tau_i"][:, j],
        })
    
    return stage_data

def simulate_system(sim_id, file_path, rock_names, parameters, years=None, integrator="trace"):
    """Creates a REBOUND simulation, runs the simulation, and saves it to disk.

    Args:
        sim_id (int): Simulation ID.
        file_path (str): Name of the file path for data storage.
        rock_names (list[str]): Names of the rocks (non-stellar particles).    
        parameters (dict): Dictionary containing planet and star parameters.
        years (float, optional, defaults to None): Number of years to integrate the simulation.
            If none is given, will use a factor of tau_a of the first planet and clip within (30kyr, 10Myr).
        integrator (str, optional, defaults to whfast): Name of the REBOUND integrator to use.
    """    
    m_vals, m_star, r_vals, r_star, a_vals = parameters["m_vals"], parameters["m_star"], parameters["r_vals"], parameters["r_star"], parameters["a_vals"]
    num_pl, num_em, num_ptsml = parameters["num_pl"], parameters["num_em"], parameters["num_ptsml"]
    
    # Create the simulation
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = integrator
    
    if integrator == 'trace':
        sim.ri_trace.r_crit_hill = 5
    
    # Add the star
    sim.add(m=m_star, r=r_star, hash='star')
    num_rocks = len(rock_names)
    
    # Add rocks 
    hash_to_name = {int(sim.particles[0].hash.value): 'star'} # initialize dict with star hash
    for i in range(num_rocks): 
        sim.add(
            m=m_vals[i], 
            r=r_vals[i], 
            a=a_vals[i], 
            hash=rock_names[i], 
            primary=sim.particles[0], 
            M=np.random.uniform(0, 2*np.pi), 
            inc=np.random.uniform(1e-4, 1e-3)
        )
        
        # Sync hashes to names
        h = int(sim.particles[-1].hash.value)
        hash_to_name[h] = rock_names[i]
        
    sim.N_active = 1 + 1 + num_em # Star + first planet + embryos
    sim.testparticle_type = 1 # Ptsmls will not interact with each other
        
    # === Collision Handling ===
    sim.collision = "line" # "direct" might miss too many collisions
    
    # Tracking stats
    collision_log = []
    particle_fate = {name: "alive" for name in rock_names}
    
    def collision_resolve(sim_pointer, collided_particles_index):
        sim = sim_pointer.contents
        ps = sim.particles

        i = collided_particles_index.p1
        j = collided_particles_index.p2

        # Determine survivor (bigger particle) and victim (smaller particle)
        if ps[i].m >= ps[j].m:
            survivor_idx = i
            victim_idx = j
            remove_code = 2
        else:
            survivor_idx = j
            victim_idx = i
            remove_code = 1
            
        survivor_name = hash_to_name[int(ps[survivor_idx].hash.value)]
        victim_name = hash_to_name[int(ps[victim_idx].hash.value)]
                    
        # Log collision
        collision_log.append({
            "time": sim.t,
            "survivor": survivor_name,
            "victim": victim_name,
            "victim mass": ps[victim_idx].m,
        })

        # Track fate
        particle_fate[victim_name] = f"accreted by {survivor_name}"

        # Merge
        ps[survivor_idx] = (ps[survivor_idx] * ps[survivor_idx].m + ps[victim_idx] * ps[victim_idx].m) / (ps[survivor_idx].m + ps[victim_idx].m)
        ps[survivor_idx].m = ps[survivor_idx].m + ps[victim_idx].m
        ps[survivor_idx].r = (ps[survivor_idx].r**3 + ps[victim_idx].r**3)**(1/3)
            
        # print(f"Collision; survivor: {survivor_name}; victim: {victim_name}")
        
        return remove_code
    
    sim.collision_resolve = collision_resolve 
    # sim.collision_resolve = 'merge' # can use merge or halt for debugging

    # Move to center of momentum
    sim.move_to_com()
    
    # Debug for overlapping
    for i in range(1, sim.N):
        for j in range(i+1, sim.N):
            pi, pj = sim.particles[i], sim.particles[j]
            dx = pi.x - pj.x
            dy = pi.y - pj.y
            dz = pi.z - pj.z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist < (pi.r + pj.r):
                print(f"WARNING: overlap at init between {hash_to_name.get(int(pi.hash.value))} "
                      f"and {hash_to_name.get(int(pj.hash.value))}: dist={dist:.4e} AU, "
                      f"sum of radii={pi.r+pj.r:.4e} AU")

    # Reboundx effects
    rebx = reboundx.Extras(sim)
    
    # Only add mof for gas drag if ptsmls exist
    if num_ptsml > 0:
        mof = rebx.load_force("modify_orbits_forces")
        rebx.add_force(mof)
    
    mig = rebx.load_force("type_I_migration")
    rebx.add_force(mig)
    
    # REBOUNDx Type I migration implementation
    mig.params["tIm_surface_density_1"] = parameters['Sigma_1au'] * AU**2 / Msun # Converted to Msun/AU^2 from g/cm^2
    mig.params["tIm_surface_density_exponent"] = parameters['alpha']
    mig.params["tIm_scale_height_1"] = parameters['h_1au']
    mig.params["tIm_flaring_index"] = parameters['beta']
    
    mig.params["ide_position"] = parameters["ide_position"]
    mig.params["ide_width"] = mig.params["tIm_scale_height_1"]*mig.params["ide_position"]**mig.params["tIm_flaring_index"]
    # print('Planet will stop within {0:.3f} AU of the inner disk edge at {1} AU'.format(mig.params["ide_width"], mig.params["ide_position"]))   

    # Clip years if needed
    if years < 1e3 or years > 10e6:
        years = np.clip(years, 1e3, 10e6)
        print(f"Sim {sim_id:<2d} | Years clipped to {years:.3e}")
            
    print(f"Sim {sim_id:<2d} | {years/1000:.1f} kyr | Sigma_1au: {parameters['Sigma_1au']:.0f} | h_1au: {parameters['h_1au']:.3f}", flush=True)
    data = integrate_sim(sim, sim_id, rock_names, parameters, years, particle_fate, hash_to_name, start_time=0)
    print(f"Sim {sim_id} completed           ")
    
    # Save data
    save_simulation_run(data, sim_id, file_path, 
                        sim_metadata={"num_rocks": num_rocks, 
                                      "collision_log": collision_log,
                                      "particle_fate": particle_fate,
                                      "ide_width": mig.params["ide_width"]} | parameters)