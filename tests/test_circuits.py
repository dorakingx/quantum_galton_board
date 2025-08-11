"""
Tests for quantum circuits.
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qgb.circuits import (
    peg_cswap, build_qgb_coherent, build_qgb_tree,
    angles_for_geometric, angles_by_binary_split,
    angles_for_geometric_lambda, angles_for_truncated_exponential
)
from qgb.samplers import sample_counts
from qgb.targets import binomial_target
from qgb.metrics import total_variation


def test_peg_cswap():
    """Test single peg circuit."""
    circuit = peg_cswap(np.pi/2)
    
    # Check circuit properties
    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 0
    
    # Add measurements to make it executable
    circuit.measure_all()
    
    # Check that circuit can be executed
    counts = sample_counts(circuit, shots=1000)
    assert len(counts) > 0


def test_build_qgb_coherent():
    """Test coherent QGB circuit."""
    layers = 4
    circuit = build_qgb_coherent(layers, np.pi/2)
    
    # Check circuit properties
    assert circuit.num_clbits == layers
    
    # Check that circuit can be executed
    counts = sample_counts(circuit, shots=1000)
    assert len(counts) > 0


def test_build_qgb_tree():
    """Test tree QGB circuit."""
    layers = 4
    circuit = build_qgb_tree(layers, np.pi/2)
    
    # Check circuit properties
    assert circuit.num_qubits == layers
    assert circuit.num_clbits == layers
    
    # Check that circuit can be executed
    counts = sample_counts(circuit, shots=1000)
    assert len(counts) > 0


def test_angles_for_geometric():
    """Test geometric angle calculation."""
    p = 0.3
    layers = 6
    angles = angles_for_geometric(p, layers)
    
    # Check shape
    assert len(angles) == layers
    
    # Check angle range (should be between 0 and π)
    for angle in angles:
        assert 0 <= angle <= np.pi
    
    # Check that all angles are the same for geometric
    assert all(np.isclose(angle, angles[0]) for angle in angles)


def test_angles_by_binary_split():
    """Test binary splitting angle calculation."""
    target = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
    angles = angles_by_binary_split(target)
    
    # Check shape
    assert len(angles) == len(target) - 1
    
    # Check angle range (should be between 0 and π)
    for angle in angles:
        assert 0 <= angle <= np.pi


def test_gaussian_convergence():
    """Test that Gaussian QGB converges to binomial distribution."""
    layers = 6
    shots = 10000
    
    # Build circuit
    circuit = build_qgb_tree(layers, np.pi/2)
    
    # Get target distribution
    target = binomial_target(layers)
    
    # Run simulation
    counts = sample_counts(circuit, shots=shots)
    
    # Convert to empirical distribution
    empirical = np.zeros(layers + 1)
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        bin_idx = bitstring.count('1')  # Count number of '1's
        empirical[bin_idx] = count / total_shots
    
    # Calculate TV distance
    tv = total_variation(empirical, target)
    
    # For noiseless simulation with sufficient shots, TV should be reasonable
    assert tv < 0.5, f"TV distance {tv} is too large for Gaussian convergence"


def test_circuit_depth():
    """Test that circuits have reasonable depth."""
    layers = 8
    
    # Test coherent circuit
    coherent_circuit = build_qgb_coherent(layers, np.pi/2)
    assert coherent_circuit.depth() > 0
    
    # Test tree circuit
    tree_circuit = build_qgb_tree(layers, np.pi/2)
    assert tree_circuit.depth() > 0
    
    # Tree circuit should be shallower than coherent circuit
    assert tree_circuit.depth() < coherent_circuit.depth()


def test_circuit_measurements():
    """Test that circuits have correct measurement structure."""
    layers = 4
    
    # Test coherent circuit
    coherent_circuit = build_qgb_coherent(layers, np.pi/2)
    assert coherent_circuit.num_clbits == layers
    
    # Test tree circuit
    tree_circuit = build_qgb_tree(layers, np.pi/2)
    assert tree_circuit.num_clbits == layers
    
    # Check that measurements are present
    coherent_ops = [op.operation.name for op in coherent_circuit.data]
    tree_ops = [op.operation.name for op in tree_circuit.data]
    
    assert 'measure' in coherent_ops
    assert 'measure' in tree_ops


def test_truncated_exponential_absorption():
    """Test that truncated exponential circuit supports absorption semantics."""
    lmbda = 0.5
    L = 3
    
    # Get angles for truncated exponential (L+1 angles for L+1 bins)
    angles = angles_for_truncated_exponential(lmbda, L)
    assert len(angles) == L + 1
    
    # Build circuit with L+1 angles
    circuit = build_qgb_tree(L, angles)
    
    # Should have L+1 qubits and L+1 classical bits for L+1 bins
    assert circuit.num_qubits == L + 1
    assert circuit.num_clbits == L + 1
    
    # Check that circuit can be executed
    counts = sample_counts(circuit, shots=1000)
    assert len(counts) > 0
    
    # Verify that we get the expected number of bins
    from qgb.samplers import process_tree_counts
    empirical = process_tree_counts(counts)
    
    # Should have reasonable probability distribution
    assert len(empirical) > 0
    assert np.sum(empirical) > 0.1  # Some probability should be captured


def test_geometric_lambda_angles():
    """Test geometric lambda angle calculation."""
    lmbda = 0.5
    layers = 4
    
    angles = angles_for_geometric_lambda(lmbda, layers)
    
    # Should have L angles for L layers
    assert len(angles) == layers
    
    # All angles should be the same for geometric approximation
    assert all(np.isclose(angle, angles[0]) for angle in angles)
    
    # Check the formula: r = 1 - exp(-lambda)
    r_expected = 1 - np.exp(-lmbda)
    r_actual = np.sin(angles[0] / 2) ** 2
    assert abs(r_expected - r_actual) < 1e-12
