"""
Biophysical CA1 Pyramidal Cell Model
Implements the Hodgkin-Huxley type 5D conductance model.
"""
import numpy as np

# Biophysical constants
C = 1.0
VNa, VK, VL = 55.0, -90.0, -70.0

# Default conductances
DEFAULT_G_PARAMS = {
    'gNa': 35.0, 'gKdr': 6.0, 'gA': 1.4, 'gM': 1.0, 'gL': 0.05, 'gNaP': 0.25
}

# Steady-state gating functions (Sigmoid forms)
def m_inf(V): return 1.0 / (1.0 + np.exp(-(V + 30.0) / 9.5))
def h_inf(V): return 1.0 / (1.0 + np.exp((V + 45.0) / 7.0))  
def n_inf(V): return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
def a_inf(V): return 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0))
def b_inf(V): return 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))
def z_inf(V): return 1.0 / (1.0 + np.exp(-(V + 39.0) / 5.0))
def p_inf(V): return 1.0 / (1.0 + np.exp(-(V + 47.0) / 3.0))

# Time constants
def tau_h(V): return 0.1 + 0.75 / (1.0 + np.exp(-(V + 40.5) / 6.0))
def tau_n(V): return 0.1 + 0.5 / (1.0 + np.exp(-(V + 27.0) / 15.0))
tau_b, tau_z = 15.0, 75.0

def ca1_drift(y: np.ndarray, g_params: dict = DEFAULT_G_PARAMS, Iapp: float = 1.6, **kwargs) -> np.ndarray:
    """Computes the deterministic drift vector for the CA1 model."""
    V, h, n, b, z = y
    
    INa  = g_params['gNa']  * (m_inf(V)**3) * h * (V - VNa)
    INaP = g_params['gNaP'] * p_inf(V) * (V - VNa)
    IKdr = g_params['gKdr'] * (n**4) * (V - VK)
    IA   = g_params['gA']   * (a_inf(V)**3) * b * (V - VK)
    IM   = g_params['gM']   * z * (V - VK)
    IL   = g_params['gL']   * (V - VL)
    
    dVdt = (-INa - INaP - IKdr - IA - IM - IL + Iapp) / C
    dhdt = (h_inf(V) - h) / tau_h(V)
    dndt = (n_inf(V) - n) / tau_n(V)
    dbdt = (b_inf(V) - b) / tau_b
    dzdt = (z_inf(V) - z) / tau_z
    
    return np.array([dVdt, dhdt, dndt, dbdt, dzdt])

def ca1_diffusion(y: np.ndarray, sigma_z: float = 0.0, **kwargs) -> np.ndarray:
    """Computes the stochastic diffusion vector (noise applied only to the M-gate z)."""
    _, _, _, _, z = y
    # Multiplicative noise: vanishes at boundaries 0 and 1
    dz_diff = sigma_z * np.sqrt(max(z * (1.0 - z), 0.0))
    return np.array([0.0, 0.0, 0.0, 0.0, dz_diff])
