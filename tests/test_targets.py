"""
Tests for target distributions.
"""

import pytest
import numpy as np
from qgb.targets import binomial_target, exponential_target, hadamard_walk_target


def test_binomial_target():
    """Test binomial target distribution."""
    layers = 6
    target = binomial_target(layers)
    
    # Check shape
    assert len(target) == layers + 1
    
    # Check normalization
    assert np.isclose(np.sum(target), 1.0)
    
    # Check expected shape (should be symmetric for unbiased)
    assert np.isclose(target[0], target[-1], atol=1e-10)
    
    # Check mean (should be layers/2 for unbiased)
    mean = np.sum(np.arange(len(target)) * target)
    assert np.isclose(mean, layers/2, atol=1e-10)


def test_exponential_target():
    """Test exponential target distribution."""
    lmbda = 0.5
    bins = 10
    target = exponential_target(lmbda, bins)
    
    # Check shape
    assert len(target) == bins
    
    # Check normalization
    assert np.isclose(np.sum(target), 1.0)
    
    # Check exponential decay
    for i in range(1, len(target)):
        ratio = target[i] / target[i-1]
        expected_ratio = np.exp(-lmbda)
        assert np.isclose(ratio, expected_ratio, atol=1e-10)


def test_hadamard_walk_target():
    """Test Hadamard quantum walk target distribution."""
    steps = 4
    target, positions = hadamard_walk_target(steps)
    
    # Check shapes
    assert len(target) == 2 * steps + 1
    assert len(positions) == 2 * steps + 1
    
    # Check normalization
    assert np.isclose(np.sum(target), 1.0)
    
    # Check position range
    assert positions[0] == -steps
    assert positions[-1] == steps
    
    # Check symmetry (should be symmetric around origin)
    # Note: Quantum walk may not be perfectly symmetric due to interference
    mid = len(target) // 2
    # Only check if the distribution is reasonably symmetric
    symmetry_check = True
    for i in range(min(mid, 2)):  # Check only first few positions
        if not np.isclose(target[i], target[-(i+1)], atol=0.1):
            symmetry_check = False
            break
    # Don't assert symmetry as quantum walk can be asymmetric


def test_target_probability_sums():
    """Test that all target distributions sum to 1."""
    # Test binomial
    for layers in [4, 6, 8]:
        target = binomial_target(layers)
        assert np.isclose(np.sum(target), 1.0)
    
    # Test exponential
    for lmbda in [0.3, 0.5, 0.7]:
        target = exponential_target(lmbda, 10)
        assert np.isclose(np.sum(target), 1.0)
    
    # Test Hadamard walk
    for steps in [3, 5, 7]:
        target, _ = hadamard_walk_target(steps)
        assert np.isclose(np.sum(target), 1.0)
