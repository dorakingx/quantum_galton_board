"""
Distance metrics and bootstrap confidence intervals.

Reference: arXiv:2202.01735 - Universal Statistical Simulator
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Total Variation distance.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        float: Total Variation distance
    """
    return 0.5 * np.sum(np.abs(p - q))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Hellinger distance.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        float: Hellinger distance
    """
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """
    Calculate KL divergence with ε-smoothing.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Smoothing parameter
        
    Returns:
        float: KL divergence
    """
    # Apply ε-smoothing
    p_smooth = p + eps
    q_smooth = q + eps
    
    # Renormalize
    p_smooth = p_smooth / np.sum(p_smooth)
    q_smooth = q_smooth / np.sum(q_smooth)
    
    return entropy(p_smooth, q_smooth)


def wasserstein1(p: np.ndarray, q: np.ndarray, positions: Optional[np.ndarray] = None) -> float:
    """
    Calculate Wasserstein-1 distance.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        positions: Position values (default: 0, 1, 2, ...)
        
    Returns:
        float: Wasserstein-1 distance
    """
    if positions is None:
        positions = np.arange(len(p))
    
    # Calculate cumulative distributions
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    
    # Wasserstein-1 = integral of |CDF_p - CDF_q|
    return np.sum(np.abs(cdf_p - cdf_q)) * np.diff(positions).mean()


def bootstrap_metrics(
    counts: Dict[str, int],
    target: np.ndarray,
    positions: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Calculate bootstrap confidence intervals for metrics with stochastic uncertainty.
    
    This function accounts for stochastic uncertainty in quantum measurements
    by using bootstrap resampling to estimate confidence intervals for distance metrics.
    
    Args:
        counts: Counts dictionary from quantum measurements
        target: Target probability distribution
        positions: Position values for Wasserstein distance
        n_boot: Number of bootstrap samples (default: 1000)
        alpha: Significance level for confidence intervals (default: 0.05 for 95% CI)
        
    Returns:
        Dict: Bootstrap results with confidence intervals for each metric
    """
    # Convert counts to empirical distribution
    total_shots = sum(counts.values())
    
    # Handle different bitstring interpretations
    if len(next(iter(counts.keys()))) > 1:  # Multi-qubit measurement
        # For Gaussian: count number of '1's
        max_bin = max(bitstring.count('1') for bitstring in counts.keys())
        empirical = np.zeros(max_bin + 1)
        for bitstring, count in counts.items():
            bin_idx = bitstring.count('1')
            empirical[bin_idx] += count / total_shots
    else:  # Single qubit measurement
        max_bin = max(int(k, 2) for k in counts.keys())
        empirical = np.zeros(max_bin + 1)
        for bitstring, count in counts.items():
            bin_idx = int(bitstring, 2)
            empirical[bin_idx] = count / total_shots
    
    # Ensure same length as target
    min_len = min(len(empirical), len(target))
    empirical = empirical[:min_len]
    target = target[:min_len]
    
    if positions is not None:
        positions = positions[:min_len]
    
    # Initialize metric functions
    metrics = {
        'tv': lambda p, q: total_variation(p, q),
        'hellinger': lambda p, q: hellinger(p, q),
        'kl': lambda p, q: kl_div(p, q),
        'wasserstein': lambda p, q: wasserstein1(p, q, positions)
    }
    
    # Bootstrap sampling to account for stochastic uncertainty
    bootstrap_samples = {}
    for metric_name, metric_func in metrics.items():
        bootstrap_samples[metric_name] = []
        
        for _ in range(n_boot):
            # Resample from empirical distribution using multinomial distribution
            # This accounts for the finite number of shots and measurement uncertainty
            resampled_counts = np.random.multinomial(total_shots, empirical)
            resampled_probs = resampled_counts / total_shots
            
            # Calculate metric for this bootstrap sample
            metric_value = metric_func(resampled_probs, target)
            bootstrap_samples[metric_name].append(metric_value)
    
    # Calculate confidence intervals and uncertainty measures
    results = {}
    for metric_name, samples in bootstrap_samples.items():
        samples = np.array(samples)
        mean_val = np.mean(samples)
        median_val = np.median(samples)
        lower = np.percentile(samples, alpha/2 * 100)
        upper = np.percentile(samples, (1-alpha/2) * 100)
        std_val = np.std(samples)
        
        # Calculate additional uncertainty measures
        mad = np.median(np.abs(samples - median_val))  # Median Absolute Deviation
        iqr = np.percentile(samples, 75) - np.percentile(samples, 25)  # Interquartile Range
        
        results[metric_name] = {
            'mean': mean_val,
            'median': median_val,
            'lower': lower,
            'upper': upper,
            'std': std_val,
            'mad': mad,
            'iqr': iqr,
            'confidence_level': 1 - alpha,
            'n_bootstrap': n_boot
        }
    
    return results


def calculate_all_metrics(
    empirical: np.ndarray,
    target: np.ndarray,
    positions: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all distance metrics between empirical and target distributions.
    
    This function computes multiple distance metrics to quantify the difference
    between the obtained quantum distribution and the target distribution.
    
    Args:
        empirical: Empirical probability distribution from quantum measurements
        target: Target probability distribution
        positions: Position values for Wasserstein distance calculation
        
    Returns:
        Dict: All metric values including TV, Hellinger, KL divergence, and Wasserstein
    """
    return {
        'tv': total_variation(empirical, target),
        'hellinger': hellinger(empirical, target),
        'kl': kl_div(empirical, target),
        'wasserstein': wasserstein1(empirical, target, positions)
    }
