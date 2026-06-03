import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

'''
Code to identify mean motion resonances in planetary systems.
Formulas used from section 2.3 in Keller et al. (2026).
Alternatively, we can use zeta as given in Eq. 11 in Fabrycky et al. (2014) to determine the period ratio.
'''

def find_best_twoBR_pq(m_star, b, c, p_max=10, crit="Delta"):
    '''Finds the best values for p and q at the end of simulation for 
       two planets given some MMR criterion'''
    if crit == 'Delta':
        best_Delta = 100
    elif crit == 'zeta': # Only 1st order formula implemented
        best_zeta = 100
        
    best_p, best_q = 100, 100
    
    for p in range(1, p_max+1):
        for q in range(1, p):
            # Use Kepler's law to find period values
            P_b = (b['a'].iloc[-1]**3 / m_star)**(1/2)
            P_c = (c['a'].iloc[-1]**3 / m_star)**(1/2)
            
            if crit == 'Delta':
                Delta = (P_c/P_b)/(p/q) - 1
                if np.abs(Delta) < np.abs(best_Delta):
                    best_Delta = Delta
                    best_p, best_q = p, q
            elif crit == 'zeta':
                zeta = 3*(1/(P_b/P_c-1) - round(1/(P_b/P_c-1)))
                if np.abs(zeta) < np.abs(best_zeta):
                    best_zeta = zeta
                    best_p, best_q = p, q
                    
    return best_p, best_q

def find_best_threeBR_pq(m_star, b, c, d, p_max=10, crit="Delta"):
    best_p_bc, best_q_bc = find_best_twoBR_pq(m_star, b, c, p_max, crit)
    best_p_cd, best_q_cd = find_best_twoBR_pq(m_star, c, d, p_max, crit)          
    return best_p_bc, best_q_bc, best_p_cd, best_q_cd

with open("fg_library.pkl", "rb") as f:
    fg_lib = pickle.load(f)
    
def twoBR_angle(b, c, p, q):
    f, g = fg_lib[(p, q)]
    pomega_hat = np.arctan2((f*b['e']*np.sin(b['pomega']) + g*c['e']*np.sin(c['pomega'])), (f*b['e']*np.cos(b['pomega']) + g*c['e']*np.cos(c['pomega'])))
    return q*b['l'] - p*c['l'] + (p-q)*pomega_hat

def threeBR_angle(b, c, d, p_bc, q_bc, p_cd, q_cd):
    return (p_cd-q_cd)*(q_bc*b['l'] - p_bc*c['l']) + (p_bc-q_bc)*(-q_cd*c['l'] + p_cd*d['l'])

def libration_amp(angles, N):
    '''Calculates libration amplitude of angles using last N samples'''
    mean_angle = np.average(angles[-N:])
    return np.sqrt(2/N * np.sum((angles[-N:]-mean_angle)**2))

def last_P(m_star, b):
    '''Returns the period value given m_star and planet dateframe'''
    return float(b['a'].iloc[-1]**3 / m_star)**(1/2)

def plot_libration(m_star, b, c, d=None, t_units='kyr', N=100, forced_pq=None):
    b_name = b.attrs["planet_name"]
    c_name = c.attrs["planet_name"]
    times = b['time']
    
    # 2BR angle
    if d is None:
        print(f"True period ratio: {last_P(m_star, c) / last_P(m_star, b):.4f}")

        # Calculate two-body resonant angle between b and c
        p, q = find_best_twoBR_pq(m_star, b, c)
    
        # If we want to analyze a period ratio that isn't the final configuration
        if forced_pq: # should be a tuple
            p, q = forced_pq

        assert p != 100 # if not, then the planets are definitely not in resonance
        print(f"p, q: {p}, {q}")
        
        twoBR = np.rad2deg(twoBR_angle(b, c, p, q)) % 360 # mod 360 so it wraps
        amp = libration_amp(twoBR, N)
        print(f"Libration amplitude: {amp:.1f} deg")

        plt.figure(figsize=(8,4))
        if t_units == 'kyr':
            plt.scatter(times/1000, twoBR, s=3) 
        plt.axhline(180, color='gray', ls='--', alpha=0.7)
        plt.xlabel("Time (yrs)")
        plt.ylabel(f"Two-body resonant angle of {b_name} & {c_name} $\phi$ (degrees)")
        plt.ylim(0,360)
        plt.grid(True)
        plt.show()
    
    # 3BR angle
    else:
        d_name = d.attrs["planet_name"]
        print(f"True period ratios: b+c: {last_P(m_star, c) / last_P(m_star, b):.4f}, c+d: {last_P(m_star, d) / last_P(m_star, c):.4f}")
        
        # Calculate three-body resonant angle between b, c, and d
        p_bc, q_bc, p_cd, q_cd = find_best_threeBR_pq(m_star, b, c, d)
        
        assert p_bc != 100; assert p_cd != 100 # if not, then the planets are definitely not in resonance
        print(f"p_{b_name}{c_name}, q_{b_name}{c_name}, p_{c_name}{d_name}, q_{c_name}{d_name}: {p_bc}, {q_bc}, {p_cd}, {q_cd}")
        
        threeBR = np.rad2deg(threeBR_angle(b, c, d, p_bc, q_bc, p_cd, q_cd)) % 360 # mod 360 so it wraps
        amp = libration_amp(threeBR, N)
        print(f"Libration amplitude: {amp:.1f} deg")

        plt.figure(figsize=(8,4))
        if t_units == 'kyr':
            plt.scatter(times/1000, threeBR%360, s=3) 
        plt.axhline(180, color='gray', ls='--', alpha=0.7)
        plt.xlabel("Time (yrs)")
        plt.ylabel(f"Three-body resonant angle of {b_name}, {c_name}, & {d_name} $\phi$ (degrees)")
        plt.ylim(0,360)
        plt.grid(True)
        plt.show()

def check_resonance(m_star, b, c, d=None, N=100):
    '''
    Given m_star and planet dataframes, returns the resonance ratios if resonance is
    detected, otherwise returns (0,0) for 2BR and ((0,0), (0,0)) for 3BR.
    '''
    # 2BR
    if d is None:
        # Calculate two-body resonant angle between b and c
        p, q = find_best_twoBR_pq(m_star, b, c)

        if p == 100: # if not, then the planets are definitely not in resonance
            return (0,0)
        
        else: # resonance detected
            twoBR = np.rad2deg(twoBR_angle(b, c, p, q)) % 360 # mod 360 so it wraps
            amp = libration_amp(twoBR, N)
            if amp < 90: # criterion for libration
                return (p,q)
            else:
                return (0,0)
    
    # 3BR
    else:
        # Calculate three-body resonant angle between b, c, and d
        p_bc, q_bc, p_cd, q_cd = find_best_threeBR_pq(m_star, b, c, d)
        
        if p_bc == 100 or p_cd == 100: # if not, then the planets are definitely not in resonance
            return ((0,0), (0,0))
        
        else: # resonance detected
            threeBR = np.rad2deg(threeBR_angle(b, c, d, p_bc, q_bc, p_cd, q_cd)) % 360 # mod 360 so it wraps
            amp = libration_amp(threeBR, N)
            if amp < 90:
                return ((p_bc, q_bc), (p_cd, q_cd))
            else:
                return ((0,0), (0,0))

def res_chain_orders(saved_sim, N=100):
    '''
    Given m_star and list of planet dataframes and planet_names in order, returns the list of 
    resonance orders of the RC. 
    '''
    sim_data = saved_sim[0]
    m_star = saved_sim[1]['m_star']
    planet_names = saved_sim[1]['planet_names']
    planets = {p: sim_data[p] for p in planet_names}
    
    orders = []
    for i in range(len(planets)-1):
        b_name = planet_names[i]
        c_name = planet_names[i+1]
        res = check_resonance(m_star, planets[b_name], planets[c_name], N=N)
        orders.append(res[0] - res[1])
            
    return orders

def res_chain_outcome(saved_sim, N=100):
    '''
    Given saved_sim, returns list based on outcome.
    
    Returns a list: whether full 3BRC exists (1 or 0)& list of 2BR orders
     '''
    sim_data = saved_sim[0]
    m_star = saved_sim[1]['m_star']
    planet_names = saved_sim[1]['planet_names']
    planets = {p: sim_data[p] for p in planet_names}        
        
    # Check existence of a full three-body chain
    threeBRC = True
    for i in range(len(planet_names)-2):
        b_name = planet_names[i]
        c_name = planet_names[i+1]
        d_name = planet_names[i+2]
        res = check_resonance(m_star, planets[b_name], planets[c_name], planets[d_name], N=N)
        if res == ((0,0), (0,0)):
            threeBRC = False
            break
            
    return [int(threeBRC)] + res_chain_orders(saved_sim, N)
    
def detect_sequential_capture(saved_sim, N_window=100, amp_threshold=90):
    '''
    Detects whether higher-order resonances formed sequentially by analyzing
    the time evolution of resonant angles.
    
    For each adjacent pair, scans through time to detect:
    1. Initial capture into first-order resonance (libration amp < threshold)
    2. Breaking of that resonance (amp rises above threshold)
    3. Re-capture into higher-order resonance (amp drops below threshold again)
    
    Returns a dict with results for each pair.
    '''
    sim_data = saved_sim[0]
    m_star = saved_sim[1]['m_star']
    planet_names = saved_sim[1]['planet_names']
    planets = {p: sim_data[p] for p in planet_names}
    
    results = {}
    
    for i in range(len(planet_names) - 1):
        b_name = planet_names[i]
        c_name = planet_names[i+1]
        b = planets[b_name]
        c = planets[c_name]
        pair_name = f"{b_name}/{c_name}"
        
        # Find best final p, q for this pair
        p_final, q_final = find_best_twoBR_pq(m_star, b, c)
        final_order = p_final - q_final
        
        # Scan through time using sliding windows
        n_times = len(b)
        times = b['time'].to_numpy()
        
        # For each p/q up to the final order, compute resonant angle time series
        # and track libration amplitude over time
        capture_history = []  # list of (time, p, q, amp) when transitions occur
        
        # Check 1st order resonances first, then up to final order
        orders_to_check = list(range(1, final_order + 1)) if final_order > 0 else [1]
        
        for order in orders_to_check:
            # Find the best p,q for this order
            best_p, best_q = None, None
            best_Delta = 100
            for p in range(order+1, 10+1):
                q = p - order
                if q < 1:
                    continue
                P_b = (b['a'].iloc[-1]**3 / m_star) ** (1/2)
                P_c = (c['a'].iloc[-1]**3 / m_star) ** (1/2)
                Delta = abs((P_c / P_b) / (p / q) - 1)
                if Delta < best_Delta:
                    best_Delta = Delta
                    best_p, best_q = p, q
            
            if best_p is None:
                continue
            
            # Compute resonant angle at each timestep
            angles = np.rad2deg(twoBR_angle(b, c, best_p, best_q)) % 360
            
            # Slide a window across time and compute libration amplitude
            amps = []
            window_times = []
            for t in range(N_window, n_times):
                amp = libration_amp(angles, N_window) if t == n_times - 1 else \
                      np.sqrt(2/N_window * np.sum((angles[t-N_window:t] - 
                              np.mean(angles[t-N_window:t]))**2))
                amps.append(amp)
                window_times.append(times[t])
            
            amps = np.array(amps)
            window_times = np.array(window_times)
            
            # Detect transitions: librating -> not librating -> librating
            librating = amps < amp_threshold
            transitions = np.diff(librating.astype(int))
            capture_times = window_times[1:][transitions == 1]   # entered libration
            breaking_times = window_times[1:][transitions == -1] # broke out of libration
            
            capture_history.append({
                'order': order,
                'p': best_p,
                'q': best_q,
                'capture_times': capture_times,
                'breaking_times': breaking_times,
                'amps': amps,
                'times': window_times,
                'librating': librating
            })
        
        # Determine if sequential capture occurred:
        # A first-order resonance was captured, then broken, then higher order formed
        sequential = False
        sequence_description = []
        
        if len(capture_history) >= 2:
            for j in range(len(capture_history) - 1):
                low_order = capture_history[j]
                high_order = capture_history[j + 1]
                
                # Check: low order was captured, then broken, then high order captured
                if (len(low_order['capture_times']) > 0 and 
                    len(low_order['breaking_times']) > 0 and
                    len(high_order['capture_times']) > 0):
                    
                    t_cap_low = low_order['capture_times'][0]
                    t_break_low = low_order['breaking_times'][0]  
                    t_cap_high = high_order['capture_times'][0]
                    
                    if t_cap_low < t_break_low < t_cap_high:
                        sequential = True
                        sequence_description.append(
                            f"{low_order['p']}/{low_order['q']} captured at t={t_cap_low:.0f}, "
                            f"broken at t={t_break_low:.0f}, "
                            f"{high_order['p']}/{high_order['q']} captured at t={t_cap_high:.0f}"
                        )
        
        results[pair_name] = {
            'sequential': sequential,
            'final_order': final_order,
            'final_p': p_final,
            'final_q': q_final,
            'sequence_description': sequence_description,
            'capture_history': capture_history
        }
    
    return results

def plot_sequential_capture(saved_sim, pair_idx=0, N_window=100):
    '''
    Plots the libration amplitude over time for each resonance order
    for a given planet pair, to visualize sequential capture.
    '''
    sim_data = saved_sim[0]
    planet_names = saved_sim[1]['planet_names']
    
    results = detect_sequential_capture(saved_sim, N_window=N_window)
    pair_name = list(results.keys())[pair_idx]
    pair_result = results[pair_name]
    
    history = pair_result['capture_history']
    if not history:
        print(f"No resonance history found for pair {pair_name}")
        return
    
    fig, axes = plt.subplots(len(history), 1, figsize=(10, 3 * len(history)), sharex=True)
    if len(history) == 1:
        axes = [axes]
    
    for ax, entry in zip(axes, history):
        ax.plot(entry['times'], entry['amps'], lw=1)
        ax.axhline(90, color='red', ls='--', alpha=0.7, label='Libration threshold (90°)')
        
        for t in entry['capture_times']:
            ax.axvline(t, color='green', ls=':', alpha=0.8, label='Capture')
        for t in entry['breaking_times']:
            ax.axvline(t, color='orange', ls=':', alpha=0.8, label='Breaking')
        
        ax.set_ylabel(f"Amp (deg)\n{entry['p']}/{entry['q']} (order {entry['order']})")
        ax.set_ylim(0, 200)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (yrs)")
    fig.suptitle(f"Sequential Resonance Capture: {pair_name}\n"
                 f"Sequential: {pair_result['sequential']}\n" + 
                 '\n'.join(pair_result['sequence_description']),
                 fontsize=11)
    plt.tight_layout()
    plt.show()
    
    return results