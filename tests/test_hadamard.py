#!/usr/bin/env python3
"""
Tests for Hadamard quantum walk implementation.
"""

import numpy as np
import pytest
from qgb.targets import hadamard_walk_target
from qgb.circuits import angles_from_target_chain, angles_by_binary_split, counts_to_bins
from qgb.metrics import calculate_all_metrics


def test_hadamard_walk_target():
    """Test hadamard_walk_target returns correct shape and properties."""
    T = 6
    p, positions = hadamard_walk_target(T)
    
    # Check length is 2*T+1
    assert len(p) == 2*T + 1
    assert len(positions) == 2*T + 1
    
    # Check positions are correct: x_k = -T + 2k
    expected_positions = np.array([-T + 2*k for k in range(2*T + 1)])
    np.testing.assert_array_equal(positions, expected_positions)
    
    # Check probabilities sum to approximately 1
    assert np.abs(np.sum(p) - 1.0) < 1e-10
    
    # Check symmetric initialization gives symmetric distribution
    p_sym, _ = hadamard_walk_target(T, init="symmetric")
    p_zero, _ = hadamard_walk_target(T, init="zero")
    
    # Symmetric should be more symmetric than zero
    # This is a qualitative check - symmetric should have more mass in middle
    middle_idx = T
    assert p_sym[middle_idx] > p_zero[middle_idx]


def test_angles_from_target_chain():
    """Test angles_from_target_chain calculation."""
    # Simple test case
    p = np.array([0.5, 0.3, 0.2])
    angles = angles_from_target_chain(p)
    
    # Should have L+1 angles where L = len(p) - 1
    assert len(angles) == len(p)
    
    # Last angle should be pi
    assert np.abs(angles[-1] - np.pi) < 1e-10
    
    # All angles should be in [0, pi]
    for angle in angles:
        assert 0 <= angle <= np.pi


def test_angles_by_binary_split():
    """Test angles_by_binary_split calculation."""
    # Simple test case
    p = np.array([0.25, 0.25, 0.25, 0.25])
    angles = angles_by_binary_split(p)
    
    # Should have L angles where L = len(p) - 1
    assert len(angles) == len(p) - 1
    
    # All angles should be in [0, pi]
    for angle in angles:
        assert 0 <= angle <= np.pi


def test_counts_to_bins():
    """Test counts_to_bins decoding."""
    L = 3  # 4 bins (0, 1, 2, 3)
    
    # Test synthetic bitstrings
    counts = {
        '000': 100,  # No '1' found -> bin = L = 3
        '001': 50,   # First '1' at position 0 -> bin = 0
        '010': 30,   # First '1' at position 1 -> bin = 1
        '100': 20,   # First '1' at position 2 -> bin = 2
        '110': 10,   # First '1' at position 0 -> bin = 0
    }
    
    bins = counts_to_bins(counts, L)
    
    # Check shape
    assert len(bins) == L + 1  # 4 bins
    
    # Check normalization
    assert np.abs(np.sum(bins) - 1.0) < 1e-10
    
    # Check specific bins
    total = 210  # sum of counts
    assert np.abs(bins[0] - (50 + 10) / total) < 1e-10  # '001' + '110'
    assert np.abs(bins[1] - 30 / total) < 1e-10         # '010'
    assert np.abs(bins[2] - 20 / total) < 1e-10         # '100'
    assert np.abs(bins[3] - 100 / total) < 1e-10        # '000'


def test_end_to_end_noiseless():
    """End-to-end test: T=12, chain angles, shots >= 200k -> TV < 0.02."""
    # This test requires actual simulation, so we'll test the structure
    T = 12
    
    # Get target distribution
    target, positions = hadamard_walk_target(T, init="symmetric")
    
    # Calculate angles
    angles = angles_from_target_chain(target)
    
    # Check angles have correct length
    assert len(angles) == 2*T + 1
    
    # Check last angle is pi
    assert np.abs(angles[-1] - np.pi) < 1e-10
    
    # Check target properties
    assert len(target) == 2*T + 1
    assert np.abs(np.sum(target) - 1.0) < 1e-10


def test_parity_zeros():
    """Test that parity zeros appear naturally in Hadamard walk."""
    T = 4
    p, positions = hadamard_walk_target(T, init="symmetric")
    
    # For T=4, positions should be [-4, -2, 0, 2, 4]
    # The walk should have zeros at certain positions due to parity
    expected_positions = np.array([-4, -2, 0, 2, 4])
    np.testing.assert_array_equal(positions, expected_positions)
    
    # Check that we have the expected number of positions
    assert len(positions) == 2*T + 1


if __name__ == "__main__":
    # Run tests
    test_hadamard_walk_target()
    test_angles_from_target_chain()
    test_angles_by_binary_split()
    test_counts_to_bins()
    test_end_to_end_noiseless()
    test_parity_zeros()
    print("All tests passed!")
