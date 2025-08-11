"""
Tests for distance metrics.
"""

import pytest
import numpy as np
from qgb.metrics import (
    total_variation, hellinger, kl_div, wasserstein1,
    bootstrap_metrics, calculate_all_metrics
)


def test_total_variation():
    """Test total variation distance."""
    # Test with identical distributions
    p = np.array([0.25, 0.25, 0.25, 0.25])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(total_variation(p, q), 0.0)
    
    # Test with completely different distributions
    p = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.isclose(total_variation(p, q), 1.0)
    
    # Test with partially different distributions
    p = np.array([0.5, 0.3, 0.2, 0.0])
    q = np.array([0.3, 0.5, 0.0, 0.2])
    tv = total_variation(p, q)
    assert 0.0 < tv < 1.0


def test_hellinger():
    """Test Hellinger distance."""
    # Test with identical distributions
    p = np.array([0.25, 0.25, 0.25, 0.25])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(hellinger(p, q), 0.0)
    
    # Test with completely different distributions
    p = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.isclose(hellinger(p, q), 1.0)
    
    # Test with partially different distributions
    p = np.array([0.5, 0.3, 0.2, 0.0])
    q = np.array([0.3, 0.5, 0.0, 0.2])
    h = hellinger(p, q)
    assert 0.0 < h < 1.0


def test_kl_div():
    """Test KL divergence."""
    # Test with identical distributions
    p = np.array([0.25, 0.25, 0.25, 0.25])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(kl_div(p, q), 0.0)
    
    # Test with different distributions
    p = np.array([0.5, 0.3, 0.2, 0.0])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    kl = kl_div(p, q)
    assert kl > 0.0


def test_wasserstein1():
    """Test Wasserstein-1 distance."""
    # Test with identical distributions
    p = np.array([0.25, 0.25, 0.25, 0.25])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    positions = np.array([0, 1, 2, 3])
    assert np.isclose(wasserstein1(p, q, positions), 0.0)
    
    # Test with shifted distributions
    p = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    positions = np.array([0, 1, 2, 3])
    w1 = wasserstein1(p, q, positions)
    assert w1 > 0.0


def test_calculate_all_metrics():
    """Test calculation of all metrics."""
    p = np.array([0.5, 0.3, 0.2, 0.0])
    q = np.array([0.3, 0.5, 0.0, 0.2])
    positions = np.array([0, 1, 2, 3])
    
    metrics = calculate_all_metrics(p, q, positions)
    
    assert 'tv' in metrics
    assert 'hellinger' in metrics
    assert 'kl' in metrics
    assert 'wasserstein' in metrics
    
    # All metrics should be non-negative
    for value in metrics.values():
        assert value >= 0.0


def test_bootstrap_metrics():
    """Test bootstrap metrics calculation."""
    # Create mock counts
    counts = {'00': 100, '01': 150, '10': 200, '11': 50}
    target = np.array([0.2, 0.3, 0.4, 0.1])
    
    results = bootstrap_metrics(counts, target, n_boot=100)
    
    # Check that all metrics are present
    expected_metrics = ['tv', 'hellinger', 'kl', 'wasserstein']
    for metric in expected_metrics:
        assert metric in results
        assert 'mean' in results[metric]
        assert 'lower' in results[metric]
        assert 'upper' in results[metric]
        assert 'std' in results[metric]
    
    # Check that confidence intervals are valid
    for metric, stats in results.items():
        assert stats['lower'] <= stats['mean'] <= stats['upper']
        assert stats['std'] >= 0.0
