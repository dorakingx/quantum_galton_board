"""
Angle optimization for Quantum Galton Board.

Reference: arXiv:2202.01735 - Universal Statistical Simulator
"""

import numpy as np
from typing import List, Tuple
from qiskit import QuantumCircuit
from .circuits import build_qgb_tree
from .targets import binomial_target
from .samplers import simulate_noisy
from .metrics import total_variation


def precompensate_angles(
    theta_list: List[float],
    noise_level: str,
    target: np.ndarray,
    shots: int = 10000,
    grid_range: float = 0.1,
    grid_points: int = 5
) -> List[float]:
    """
    Precompensate angles using grid search to reduce TV distance.
    
    Args:
        theta_list: Original angle list
        noise_model: Noise model to use
        target: Target probability distribution
        shots: Number of shots for evaluation
        grid_range: Range for grid search (±range around each angle)
        grid_points: Number of grid points per angle
        
    Returns:
        List[float]: Optimized angle list
    """
    optimized_angles = []
    
    for i, theta in enumerate(theta_list):
        # Create grid around current angle
        grid = np.linspace(theta - grid_range, theta + grid_range, grid_points)
        best_theta = theta
        best_tv = float('inf')
        
        for test_theta in grid:
            # Create test angle list
            test_angles = theta_list.copy()
            test_angles[i] = test_theta
            
            # Build circuit with test angles
            circuit = build_qgb_tree(len(test_angles), test_angles)
            
            # Simulate with noise
            counts = simulate_noisy(circuit, noise_level=noise_level, shots=shots)
            
            # Convert counts to probabilities
            empirical = counts_to_probabilities(counts)
            
            # Ensure same length as target
            min_len = min(len(empirical), len(target))
            empirical = empirical[:min_len]
            target_short = target[:min_len]
            
            # Calculate TV distance
            tv = total_variation(empirical, target_short)
            
            if tv < best_tv:
                best_tv = tv
                best_theta = test_theta
        
        optimized_angles.append(best_theta)
    
    return optimized_angles


def grid_search_optimization(
    initial_angles: List[float],
    target: np.ndarray,
    noise_level: str = "medium",
    shots: int = 5000,
    max_iterations: int = 10,
    tolerance: float = 1e-4
) -> Tuple[List[float], float]:
    """
    Perform grid search optimization for angle compensation.
    
    Args:
        initial_angles: Initial angle list
        target: Target probability distribution
        noise_level: Noise level for simulation
        shots: Number of shots per evaluation
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple[List[float], float]: (optimized_angles, final_tv)
    """
    from .noise import get_noise_model
    
    noise_model = get_noise_model(noise_level)
    current_angles = initial_angles.copy()
    prev_tv = float('inf')
    
    for iteration in range(max_iterations):
        # Optimize each angle
        optimized_angles = precompensate_angles(
            current_angles, noise_level, target, shots
        )
        
        # Evaluate final TV distance
        circuit = build_qgb_tree(len(optimized_angles), optimized_angles)
        counts = simulate_noisy(circuit, noise_level=noise_level, shots=shots)
        empirical = counts_to_probabilities(counts)
        
        min_len = min(len(empirical), len(target))
        empirical = empirical[:min_len]
        target_short = target[:min_len]
        
        current_tv = total_variation(empirical, target_short)
        
        # Check convergence
        if abs(current_tv - prev_tv) < tolerance:
            break
        
        prev_tv = current_tv
        current_angles = optimized_angles
    
    return current_angles, current_tv


def counts_to_probabilities(counts: dict) -> np.ndarray:
    """
    Convert counts to probability distribution.
    
    Args:
        counts: Counts dictionary
        
    Returns:
        np.ndarray: Probability distribution
    """
    total_shots = sum(counts.values())
    max_bin = max(int(k, 2) for k in counts.keys())
    
    probs = np.zeros(max_bin + 1)
    for bitstring, count in counts.items():
        bin_idx = int(bitstring, 2)
        probs[bin_idx] = count / total_shots
    
    return probs


def optimize_for_gaussian(
    layers: int,
    noise_level: str = "medium",
    shots: int = 5000
) -> Tuple[List[float], float]:
    """
    Optimize angles for Gaussian distribution.
    
    Args:
        layers: Number of layers
        noise_level: Noise level
        shots: Number of shots
        
    Returns:
        Tuple[List[float], float]: (optimized_angles, final_tv)
    """
    # Initial unbiased angles
    initial_angles = [np.pi/2] * layers
    
    # Target binomial distribution
    target = binomial_target(layers)
    
    return grid_search_optimization(
        initial_angles, target, noise_level, shots
    )


def optimize_for_exponential(
    layers: int,
    lmbda: float,
    noise_level: str = "medium",
    shots: int = 5000
) -> Tuple[List[float], float]:
    """
    Optimize angles for exponential distribution.
    
    Args:
        layers: Number of layers
        lmbda: Exponential parameter
        noise_level: Noise level
        shots: Number of shots
        
    Returns:
        Tuple[List[float], float]: (optimized_angles, final_tv)
    """
    from .targets import exponential_target
    from .circuits import angles_for_geometric
    
    # Initial geometric angles
    p = 1 - np.exp(-lmbda)
    initial_angles = angles_for_geometric(p, layers)
    
    # Target exponential distribution
    target = exponential_target(lmbda, layers + 1)
    
    return grid_search_optimization(
        initial_angles, target, noise_level, shots
    )
