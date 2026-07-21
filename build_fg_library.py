import pickle
import celmech

def build_fg_library(p_max, q_max):    
    fg_library = {}
    
    for p in range(1, p_max+1):
        for q in range(1, q_max+1):
            fg = celmech.disturbing_function.get_fg_coefficients(p, q)
            fg_library[(p, q)] = fg

    return fg_library

fg_lib = build_fg_library(p_max=7, q_max=1) # for p:p-q MMRs
print("fg library created")
with open("fg_library2.pkl", "wb") as f:
    pickle.dump(fg_lib, f)