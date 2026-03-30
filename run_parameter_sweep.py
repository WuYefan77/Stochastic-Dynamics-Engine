"""
Deterministic Parameter Sweep Engine
Performs sensitivity analysis on the applied current (I_app) to identify 
the bifurcation point and the "sweet spot" for periodic bursting.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Import the pre-defined biophysical payload and initial states
from models.ca1_pyramidal import ca1_drift, DEFAULT_G_PARAMS, h_inf, n_inf, b_inf, z_inf

def main():
    print("--- Initializing I_app Parameter Sweep ---")
    
    # Define the parameter space for the applied current
    # Scanning from 0.5 to 4.0 µA/cm²
    Iapp_values = np.linspace(0.5, 4.0, 31) 
    
    # Initial conditions: [V, h, n, b, z] at V = -70 mV
    y0 = np.array([-70.0, h_inf(-70), n_inf(-70), b_inf(-70), z_inf(-70)])
    t_span = (0, 2000)  # 2000 ms simulation time
    
    # We will save the plots instead of blocking the loop with plt.show()
    # This is standard practice for automated parameter sweeps.
    for iapp_val in tqdm(Iapp_values, desc="Scanning I_app Phase Space"):
        
        # Wrapper function to adapt our decoupled drift function to scipy's solve_ivp
        # solve_ivp expects f(t, y), but our ca1_drift is f(y, **kwargs)
        def ode_system(t, y):
            return ca1_drift(y, g_params=DEFAULT_G_PARAMS, Iapp=iapp_val)
        
        # Execute RK45 Deterministic Solver
        sol = solve_ivp(
            fun=ode_system, 
            t_span=t_span, 
            y0=y0, 
            method='RK45', 
            dense_output=True
        )
        
        # Visualization
        plt.figure(figsize=(12, 4))
        plt.plot(sol.t, sol.y[0], lw=0.8, color='#1f77b4')
        plt.title(f"Deterministic Trajectory | $I_{{app}} = {iapp_val:.2f}$ µA/cm²", fontweight='bold')
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")
        plt.ylim(-100, 60)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # Save dynamically
        safe_filename = f"figures/sweep_Iapp_{iapp_val:.2f}.png".replace('.', 'p', 1)
        plt.savefig(safe_filename, dpi=150, bbox_inches='tight')
        plt.close()

    print("--- Parameter Sweep Completed. Check the 'figures' directory. ---")

if __name__ == "__main__":
    main()
