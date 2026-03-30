"""
Core SDE Solver Module
Implements numerical integration schemes for Stochastic Differential Equations.
"""
import numpy as np
from typing import Callable, Tuple, List, Optional

def euler_maruyama(
    drift_func: Callable,
    diffusion_func: Callable,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    clip_bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler-Maruyama method for simulating multidimensional SDEs.
    
    Parameters:
    -----------
    drift_func : Callable
        Function computing the deterministic drift vector. Signature: f(y, **kwargs)
    diffusion_func : Callable
        Function computing the stochastic diffusion vector. Signature: g(y, **kwargs)
    y0 : np.ndarray
        Initial state vector.
    t_span : Tuple[float, float]
        Integration time bounds (t_start, t_end).
    dt : float
        Time step size.
    clip_bounds : List[Tuple[float, float]], optional
        A list of (lower, upper) bounds for each state variable to prevent numerical 
        overflow or enforce physical constraints (e.g., probabilities in [0, 1]).
    **kwargs : dict
        Additional parameters to pass to the drift and diffusion functions.
        
    Returns:
    --------
    t : np.ndarray
        Time array.
    y : np.ndarray
        Simulated state trajectory of shape (n_states, n_steps).
    """
    t0, T = t_span
    N = int((T - t0) / dt)
    n_states = len(y0)
    
    y = np.zeros((n_states, N))
    t = np.linspace(t0, T, N)
    y[:, 0] = y0
    
    for i in range(N - 1):
        y_current = y[:, i]
        
        # Calculate drift and diffusion (passing through external parameters like g_params or sigma_z)
        drift = drift_func(y_current, **kwargs)
        diffusion = diffusion_func(y_current, **kwargs)
        
        # Generate Wiener increment (dW ~ N(0, dt))
        dW = np.random.normal(0, np.sqrt(dt))
        
        # Euler-Maruyama update step
        y_next = y_current + drift * dt + diffusion * dW
        
        # Apply physical or mathematical boundary constraints if provided
        if clip_bounds is not None:
            for j in range(n_states):
                lower, upper = clip_bounds[j]
                if lower is not None or upper is not None:
                    y_next[j] = np.clip(
                        y_next[j], 
                        lower if lower is not None else -np.inf, 
                        upper if upper is not None else np.inf
                    )
                    
        y[:, i+1] = y_next
        
    return t, y
