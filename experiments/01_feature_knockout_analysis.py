"""
Feature Knockout and Parameter Tuning Analysis
Evaluates the contribution of individual "modulators" and "engine" parameters 
to the systemic bursting behavior using deterministic ablation studies.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

# Import our decoupled architecture
from core.sde_solvers import euler_maruyama
from models.ca1_pyramidal import ca1_drift, ca1_diffusion, DEFAULT_G_PARAMS, h_inf, n_inf, b_inf, z_inf

def run_experiment(experiment_name: str, modified_params: dict):
    """Runs a single 2000ms deterministic simulation with modified conductances."""
    print(f"Running: {experiment_name}")
    
    # 1. Setup Simulation (Deterministic, so sigma_z = 0)
    t_span = (0, 2000)
    dt = 0.05
    y0 = np.array([-70.0, h_inf(-70), n_inf(-70), b_inf(-70), z_inf(-70)])
    
    # Execute the deterministic engine
    t, y = euler_maruyama(
        drift_func=ca1_drift, 
        diffusion_func=ca1_diffusion, 
        y0=y0, 
        t_span=t_span, 
        dt=dt, 
        sigma_z=0.0, # Pure ODE mode
        g_params=modified_params 
    )
    
    # 2. Visualization
    plt.figure(figsize=(12, 4))
    plt.plot(t, y[0, :], lw=0.8, color='#1f77b4')
    plt.title(f"{experiment_name}", fontweight='bold')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.ylim(-100, 60)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Save the figure
    safe_filename = experiment_name.replace(" ", "_").replace(":", "").replace("=", "_").lower()
    plt.savefig(f"figures/{safe_filename}.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # 0. Baseline (Control)
    run_experiment("Control Simulation Baseline Bursting Behavior", DEFAULT_G_PARAMS)
    
    # --- PHASE 1: VIRTUAL KNOCKOUTS (Modulators) ---
    
    # Knockout 1: No M-Current (Burst Terminator removed)
    params_no_m = copy.deepcopy(DEFAULT_G_PARAMS)
    params_no_m['gM'] = 0.0
    run_experiment("Knockout 1 No M-Current (g_M = 0)", params_no_m)
    
    # Knockout 2: No Persistent Sodium (Burst Sustainer removed)
    params_no_nap = copy.deepcopy(DEFAULT_G_PARAMS)
    params_no_nap['gNaP'] = 0.0
    run_experiment("Knockout 2 No Persistent Sodium Current (g_NaP = 0)", params_no_nap)
    
    # Knockout 3: No A-type Current (Burst Fine-Tuner removed)
    params_no_a = copy.deepcopy(DEFAULT_G_PARAMS)
    params_no_a['gA'] = 0.0
    run_experiment("Knockout 3 No A-type Current (g_A = 0)", params_no_a)
    
    # --- PHASE 2: ENGINE TUNING (Core Dynamics) ---
    
    # Tuning 1: Weakened Repolarization
    params_weak_kdr = copy.deepcopy(DEFAULT_G_PARAMS)
    params_weak_kdr['gKdr'] = 3.0  # Halved from 6.0
    run_experiment("Engine Tuning 1 Weakened Kdr Current (g_Kdr = 3.0)", params_weak_kdr)
    
    # Tuning 2: Weakened Depolarization (Excitability loss)
    params_weak_na = copy.deepcopy(DEFAULT_G_PARAMS)
    params_weak_na['gNa'] = 17.5   # Halved from 35.0
    run_experiment("Engine Tuning 2 Weakened Na Current (g_Na = 17.5)", params_weak_na)
    
    print("\nAll knockout and tuning experiments completed. Check the 'figures' directory.")

if __name__ == "__main__":
    main()
