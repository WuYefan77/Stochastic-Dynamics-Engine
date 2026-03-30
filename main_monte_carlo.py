"""
Main Monte Carlo Experiment Pipeline
Evaluates the temporal reliability of bursting under varying stochastic noise levels.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# Modular imports
from core.sde_solvers import euler_maruyama
from models.ca1_pyramidal import (
    ca1_drift, ca1_diffusion, h_inf, n_inf, b_inf, z_inf
)
from utils.signal_processing import detect_spikes, detect_bursts, calculate_ibi_stats

def main():
    np.random.seed(42)
    
    # 1. Simulation Configurations
    dt = 0.05
    t_span = (0, 5000)
    noise_levels = [0.0, 0.2, 0.5, 1.0]
    num_realizations = 50
    
    # Init state: V, h, n, b, z
    y0 = np.array([-70.0, h_inf(-70), n_inf(-70), b_inf(-70), z_inf(-70)])
    
    # Boundary constraints: Voltage unconstrained (None, None), gating variables clamped to [0, 1]
    clip_bounds = [(None, None)] + [(0.0, 1.0)] * 4
    
    # 2. Memory Cache for Plots (Performance Optimization)
    # Avoids re-running heavy SDE simulations just for plotting
    plot_cache = {
        'phase_0': None,
        'phase_0_5': None,
        'traces': {},
        'raster_0_5_spikes': [],
        'ibi_0_5': []
    }
    all_results = []

    # 3. Monte Carlo Loop
    for sigma in noise_levels:
        cv_list = []
        print(f"\n=== Executing Monte Carlo Engine | Noise Intensity σ_z = {sigma} ===")
        
        for i in tqdm(range(num_realizations)):
            # Execute Core Solver
            t, y = euler_maruyama(
                drift_func=ca1_drift, 
                diffusion_func=ca1_diffusion, 
                y0=y0, 
                t_span=t_span, 
                dt=dt, 
                clip_bounds=clip_bounds,
                sigma_z=sigma  # Passed via **kwargs
            )

            # Signal Extraction
            v_trace = y[0, :]
            spike_times = detect_spikes(t, v_trace)
            burst_starts, _ = detect_bursts(spike_times)
            stats = calculate_ibi_stats(burst_starts)
            cv_list.append(stats['cv'])

            # --- Data Caching Logic ---
            if i == 0:
                plot_cache['traces'][sigma] = (t.copy(), v_trace.copy())
                if sigma == 0.0: plot_cache['phase_0'] = (y[4, :].copy(), v_trace.copy(), t.copy())
                if sigma == 0.5: plot_cache['phase_0_5'] = (y[4, :].copy(), v_trace.copy(), t.copy())
            
            if sigma == 0.5:
                plot_cache['raster_0_5_spikes'].append(spike_times)
                if stats['ibis'] is not None:
                    plot_cache['ibi_0_5'].extend(stats['ibis'])

        # Statistical Aggregation
        mean_cv = np.nanmean(cv_list)
        sem_cv = np.nanstd(cv_list, ddof=1) / np.sqrt(num_realizations)
        all_results.append({'sigma': sigma, 'mean_cv': mean_cv, 'sem_cv': sem_cv})

    # 4. Export & Visualization (Using cached data)
    print("\n✅ Simulations complete. Exporting statistical tables and figures...")
    
    # Export summary table
    summary_df = pd.DataFrame(all_results).sort_values("sigma")
    summary_df.to_csv("cv_summary.csv", index=False)
    
    # Plot CV vs Noise
    sigmas = [r['sigma'] for r in all_results]
    cv_means = [r['mean_cv'] for r in all_results]
    cv_sems  = [r['sem_cv']  for r in all_results]
    
    plt.figure(figsize=(6, 4))
    plt.errorbar(sigmas, cv_means, yerr=cv_sems, fmt='o-', capsize=4, label='Raw CV')
    plt.plot(sigmas, uniform_filter1d(cv_means, size=2), 'r--', label='Smoothed Trend')
    plt.xlabel(r"Intrinsic Noise Intensity ($\sigma_z$)")
    plt.ylabel("Coefficient of Variation (IBI)")
    plt.title("Bursting Reliability vs. Intrinsic Noise")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("cv_vs_noise.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Cached Raster (sigma=0.5)
    plt.figure(figsize=(10, 5))
    for idx, st in enumerate(plot_cache['raster_0_5_spikes']):
        plt.scatter(st, np.ones_like(st) * idx, s=2, color='k')
    plt.xlabel("Time (ms)")
    plt.ylabel("Monte Carlo Trial")
    plt.title(r"Spike Timing Raster ($\sigma_z = 0.5$)")
    plt.xlim(1000, 3000)
    plt.savefig("raster_sigma_0p5.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print("All artifacts generated and saved to current directory.")

if __name__ == "__main__":
    main()
