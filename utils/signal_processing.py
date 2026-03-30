"""
Signal Processing Utilities
Extracts event-driven features (spikes, bursts) from continuous time series.
"""
import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Dict, Any

def detect_spikes(t: np.ndarray, v: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Detects spike times from a voltage trace."""
    peaks, _ = find_peaks(v, height=threshold)
    return t[peaks]

def detect_bursts(spike_times: np.ndarray, isi_thresh: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
    """Groups spikes into bursts based on the Inter-Spike Interval (ISI) threshold."""
    if len(spike_times) < 2:
        return np.array([]), np.array([])
        
    ISI = np.diff(spike_times)
    burst_starts = [spike_times[0]]
    
    for i in range(1, len(ISI)):
        # New burst identified if previous ISI is exceptionally large
        if ISI[i - 1] > isi_thresh:
            burst_starts.append(spike_times[i])
            
    return np.array(burst_starts), ISI

def calculate_ibi_stats(burst_starts: np.ndarray) -> Dict[str, Any]:
    """Calculates statistics (mean and CV) of Inter-Burst Intervals (IBIs)."""
    if len(burst_starts) < 2:
        return {'mean': np.nan, 'cv': np.nan, 'ibis': None}
        
    ibis = np.diff(burst_starts)
    mean_ibi = np.mean(ibis)
    # Protection against zero division
    cv_ibi = np.std(ibis) / mean_ibi if mean_ibi > 0 else np.nan
    
    return {'mean': mean_ibi, 'cv': cv_ibi, 'ibis': ibis}
