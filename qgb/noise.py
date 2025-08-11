"""
Noise model implementations for Quantum Galton Board.

Reference: arXiv:2202.01735 - Universal Statistical Simulator
"""

import numpy as np
from typing import Optional
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError,
    pauli_error
)


def get_noise_model(level: str = "medium") -> NoiseModel:
    """
    Create custom noise model.
    
    Args:
        level: Noise level ("low", "medium", "high")
        
    Returns:
        NoiseModel: Custom noise model
    """
    noise_model = NoiseModel()
    
    # Set error rates based on level
    if level == "low":
        gate_error = 0.001
        readout_error = 0.01
    elif level == "medium":
        gate_error = 0.01
        readout_error = 0.05
    elif level == "high":
        gate_error = 0.05
        readout_error = 0.1
    else:
        raise ValueError(f"Unknown noise level: {level}")
    
    # Add depolarizing error to single-qubit gates only
    # Avoid multiple error types on same gates to prevent conflicts
    error_1q = depolarizing_error(gate_error, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz'])
    
    # Add depolarizing error to two-qubit gates
    error_2q = depolarizing_error(gate_error, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
    
    # Add readout error
    readout_error_matrix = np.array([
        [1 - readout_error, readout_error],
        [readout_error, 1 - readout_error]
    ])
    readout_error_obj = ReadoutError(readout_error_matrix)
    noise_model.add_all_qubit_readout_error(readout_error_obj)
    
    return noise_model


def add_twirling(noise_model: NoiseModel, twirl_type: str = "pauli") -> NoiseModel:
    """
    Add twirling to noise model.
    
    Args:
        noise_model: Base noise model
        twirl_type: Type of twirling ("pauli", "clifford")
        
    Returns:
        NoiseModel: Noise model with twirling
    """
    if twirl_type == "pauli":
        # Add Pauli twirling
        pauli_twirl = pauli_error([('X', 0.25), ('Y', 0.25), ('Z', 0.25), ('I', 0.25)])
        noise_model.add_all_qubit_quantum_error(pauli_twirl, ['rx', 'ry', 'rz'])
    
    return noise_model


def add_dynamical_decoupling(noise_model: NoiseModel, dd_sequence: str = "XY4") -> NoiseModel:
    """
    Add dynamical decoupling to noise model.
    
    Args:
        noise_model: Base noise model
        dd_sequence: DD sequence type ("XY4", "XZXZ")
        
    Returns:
        NoiseModel: Noise model with DD
    """
    # This is a simplified implementation
    # In practice, DD would be added during transpilation
    
    if dd_sequence == "XY4":
        # Add XY4 sequence errors
        dd_error = depolarizing_error(0.001, 1)
        noise_model.add_all_qubit_quantum_error(dd_error, ['x', 'y'])
    
    return noise_model


def get_measurement_error_mitigation_matrix(backend_name: Optional[str] = None) -> np.ndarray:
    """
    Get measurement error mitigation matrix.
    
    Args:
        backend_name: Name of backend (optional)
        
    Returns:
        np.ndarray: Mitigation matrix
    """
    # Simplified implementation
    # In practice, this would be calibrated from the backend
    
    # Assume 2% readout error
    readout_error = 0.02
    mitigation_matrix = np.array([
        [1 - readout_error, readout_error],
        [readout_error, 1 - readout_error]
    ])
    
    return np.linalg.inv(mitigation_matrix)
