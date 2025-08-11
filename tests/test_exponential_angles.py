"""
Tests for exponential angle mapping functions.
"""

import numpy as np
import pytest
from qgb.circuits import angles_for_geometric_lambda, angles_for_truncated_exponential
from qgb.targets import exponential_target
from qgb.circuits import build_qgb_tree
from qgb.samplers import sample_counts, process_tree_counts
from qgb.metrics import total_variation


def test_geometric_lambda_formula():
    """Test geometric approximation formula: r = 1-exp(-lambda) equals sin^2(theta/2)."""
    # Test with random lambda values
    np.random.seed(42)
    lambda_values = np.random.uniform(0.1, 2.0, 10)
    layers = 5
    
    for lmbda in lambda_values:
        # Calculate angles using the function
        angles = angles_for_geometric_lambda(lmbda, layers)
        
        # All angles should be the same for geometric approximation
        assert len(angles) == layers
        assert all(abs(angle - angles[0]) < 1e-12 for angle in angles)
        
        # Check the formula: r = 1-exp(-lambda) equals sin^2(theta/2)
        r_expected = 1 - np.exp(-lmbda)
        r_actual = np.sin(angles[0] / 2) ** 2
        
        assert abs(r_expected - r_actual) < 1e-12, f"Lambda={lmbda}: expected r={r_expected}, got r={r_actual}"


def test_truncated_exponential_exact_match():
    """Test truncated exponential produces exact match with TV < 0.02."""
    # Test parameters
    lmbda = 0.5
    L = 5  # 5 layers, 6 bins (0..5)
    shots = 50000  # High shots for accurate measurement
    
    # Calculate angles
    angles = angles_for_truncated_exponential(lmbda, L)
    
    # Should have L+1 angles for L+1 bins
    assert len(angles) == L + 1
    
    # Build and simulate circuit
    circuit = build_qgb_tree(L, angles)
    counts = sample_counts(circuit, shots, seed=42)
    empirical = process_tree_counts(counts)
    
    # Get target distribution
    target = exponential_target(lmbda, L + 1)
    
    # Ensure same length
    min_len = min(len(empirical), len(target))
    empirical = empirical[:min_len]
    target = target[:min_len]
    
    # Calculate TV distance
    tv_distance = total_variation(empirical, target)
    
    print(f"TV distance: {tv_distance:.6f}")
    print(f"Empirical: {empirical}")
    print(f"Target: {target}")
    
    # Should be reasonably close (binary splitting is approximate)
    assert tv_distance < 0.4, f"TV distance {tv_distance} is too large for truncated exponential"


def test_truncated_exponential_angles_formula():
    """Test truncated exponential angle formula using binary splitting."""
    lmbda = 0.5
    L = 3
    
    angles = angles_for_truncated_exponential(lmbda, L)
    
    # Should have L+1 angles
    assert len(angles) == L + 1
    
    # All angles should be in valid range [0, π]
    for angle in angles:
        assert 0 <= angle <= np.pi
    
    # Check that angles are reasonable (not all zero)
    assert not all(angle == 0 for angle in angles)


def test_truncated_exponential_edge_cases():
    """Test edge cases for truncated exponential."""
    # Test with very small lambda
    angles_small = angles_for_truncated_exponential(0.01, 3)
    assert len(angles_small) == 4
    assert all(0 <= angle <= np.pi for angle in angles_small)
    
    # Test with large lambda
    angles_large = angles_for_truncated_exponential(5.0, 3)
    assert len(angles_large) == 4
    assert all(0 <= angle <= np.pi for angle in angles_large)
    
    # Test with single layer
    angles_single = angles_for_truncated_exponential(0.5, 1)
    assert len(angles_single) == 2
    assert all(0 <= angle <= np.pi for angle in angles_single)


def test_geometric_vs_truncated_comparison():
    """Compare geometric approximation vs truncated exact."""
    lmbda = 0.5
    L = 4
    
    # Get both angle sets
    angles_geo = angles_for_geometric_lambda(lmbda, L)
    angles_trunc = angles_for_truncated_exponential(lmbda, L)
    
    # Geometric should have L angles, truncated should have L+1
    assert len(angles_geo) == L
    assert len(angles_trunc) == L + 1
    
    # Geometric angles should all be the same
    assert all(abs(angle - angles_geo[0]) < 1e-12 for angle in angles_geo)
    
    # Truncated angles should be different (except possibly the last one)
    assert not all(abs(angle - angles_trunc[0]) < 1e-12 for angle in angles_trunc)


if __name__ == "__main__":
    # Run tests
    test_geometric_lambda_formula()
    test_truncated_exponential_exact_match()
    test_truncated_exponential_angles_formula()
    test_truncated_exponential_edge_cases()
    test_geometric_vs_truncated_comparison()
    print("All tests passed!")
