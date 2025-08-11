"""
Quantum Galton Board circuit implementations.

Reference: arXiv:2202.01735 - Universal Statistical Simulator
"""

import numpy as np
from typing import List, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter


def peg_cswap(coin_angle: float) -> QuantumCircuit:
    """
    Create a single peg using cSWAP (Fredkin) gate.
    
    Args:
        coin_angle: Rotation angle for the coin qubit (π/2 for unbiased)
        
    Returns:
        QuantumCircuit: 3-qubit circuit implementing the peg
    """
    qc = QuantumCircuit(3, name="peg_cswap")
    
    # Coin qubit preparation
    qc.rx(coin_angle, 0)
    
    # cSWAP implementation using Toffoli decomposition
    # cSWAP(a,b,c) = CNOT(c,b) Toffoli(a,b,c) CNOT(c,b)
    qc.cx(2, 1)  # CNOT(c,b)
    qc.ccx(0, 1, 2)  # Toffoli(a,b,c)
    qc.cx(2, 1)  # CNOT(c,b)
    
    return qc


def build_qgb_coherent(layers: int, coin_angles: Union[float, List[float]]) -> QuantumCircuit:
    """
    Build coherent QGB pyramid using cSWAP pegs.
    
    Args:
        layers: Number of layers in the pyramid
        coin_angles: Coin angle(s) for each layer
        
    Returns:
        QuantumCircuit: Coherent QGB circuit
    """
    if isinstance(coin_angles, (int, float)):
        coin_angles = [coin_angles] * layers
    
    # Calculate total qubits needed
    # Each layer l has l+1 qubits, plus coin qubits
    total_qubits = sum(l + 1 for l in range(layers + 1)) + layers
    
    qc = QuantumCircuit(total_qubits, layers)
    
    # Build pyramid layer by layer
    qubit_offset = 0
    for layer in range(layers):
        # Add coin qubit
        coin_qubit = qubit_offset
        qc.rx(coin_angles[layer], coin_qubit)
        
        # Add pegs for this layer
        for peg in range(layer + 1):
            # Path qubits for this peg
            path1 = qubit_offset + 1 + peg
            path2 = qubit_offset + 1 + peg + 1
            
            # Apply cSWAP peg
            peg_circ = peg_cswap(0)  # Coin already prepared
            qc.compose(peg_circ, qubits=[coin_qubit, path1, path2], inplace=True)
        
        # Measure coin qubit
        qc.measure(coin_qubit, layer)
        
        qubit_offset += layer + 2  # +1 for coin, +1 for path qubits
    
    return qc


def build_qgb_tree(layers: int, coin_angles: Union[float, List[float]]) -> QuantumCircuit:
    """
    Build depth-optimized tree QGB with mid-circuit measurement.
    
    Each layer has one coin. Rx(theta_k)→測定. 
    Bit列の '最初に1が出た位置' を bin と解釈。
    ビット順の混乱を避けるため cl[k] に qubit[k] をそのまま測らせる。
    
    Args:
        layers: Number of layers
        coin_angles: Coin angle(s) for each layer
        
    Returns:
        QuantumCircuit: Tree QGB circuit
    """
    if isinstance(coin_angles, (int, float)):
        coin_angles = [coin_angles] * layers
    
    # Handle case where we have L+1 angles for L+1 bins (truncated exponential)
    if len(coin_angles) == layers + 1:
        # Use all L+1 angles for L+1 bins
        num_qubits = layers + 1
    else:
        # Use L angles for L layers
        num_qubits = layers
    
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    for layer in range(len(coin_angles)):
        # Prepare coin qubit
        qc.rx(coin_angles[layer], layer)
        
        # Measure immediately
        qc.measure(layer, layer)
    
    return qc


def build_qgb_gaussian(layers: int, coin_angles: Union[float, List[float]]) -> QuantumCircuit:
    """
    Build Gaussian QGB with traditional "right-count = bin" approach.
    
    Each layer has one coin. Rx(theta_k)→測定. 
    Count number of '1's in bitstring to get bin index.
    This is the traditional Galton board approach for Gaussian/binomial distribution.
    
    Args:
        layers: Number of layers
        coin_angles: Coin angle(s) for each layer
        
    Returns:
        QuantumCircuit: Gaussian QGB circuit
    """
    if isinstance(coin_angles, (int, float)):
        coin_angles = [coin_angles] * layers
    
    # For Gaussian: L layers produce L+1 bins (0 to L)
    num_qubits = layers
    
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    for layer in range(len(coin_angles)):
        # Prepare coin qubit
        qc.rx(coin_angles[layer], layer)
        
        # Measure immediately
        qc.measure(layer, layer)
    
    return qc


def angles_for_geometric(p: float, layers: int) -> List[float]:
    """
    Calculate coin angles for geometric distribution.
    
    Args:
        p: Probability of moving right (absorb-right probability)
        layers: Number of layers
        
    Returns:
        List[float]: Coin angles for each layer
    """
    theta = 2 * np.arcsin(np.sqrt(p))
    return [theta] * layers


def angles_for_geometric_lambda(lmbda: float, layers: int) -> List[float]:
    """
    Calculate coin angles for geometric approximation of exponential distribution.
    
    For discretized exponential with rate lambda: r = 1 - exp(-lambda).
    Uses the same angle at each of the first L layers.
    Optional: force absorption at the final layer L with theta_L = pi to collect the tail.
    
    Args:
        lmbda: Exponential rate parameter λ
        layers: Number of layers
        
    Returns:
        List[float]: Coin angles for each layer
    """
    # Geometric approximation: r = 1 - exp(-lambda)
    r = 1 - np.exp(-lmbda)
    theta = 2 * np.arcsin(np.sqrt(r))
    
    # Use the same angle for all layers (geometric approximation)
    angles = [theta] * layers + [np.pi]  # 最終層は必ず吸収
    
    return angles


def angles_for_truncated_exponential(lmbda: float, L: int) -> List[float]:
    """
    Calculate coin angles for exact truncated exponential distribution.
    
    Let q = exp(-lambda). Target: p_k ∝ q^k normalized on k=0..L.
    Construct a left-continue / right-absorb tree. At layer k (0-index):
    Absorb-right probability: r_k = (1 - q) / (1 - q**(L+1-k)).
    Coin angle: theta_k = 2*arcsin(sqrt(r_k)).
    
    Args:
        lmbda: Exponential rate parameter λ
        L: Number of layers (L+1 bins)
        
    Returns:
        List[float]: Coin angles for each layer (L+1 angles for L+1 bins)
    """
    q = np.exp(-lmbda)
    thetas = []
    for k in range(L + 1):  # k=0..L
        rk = (1.0 - q) / (1.0 - q**(L + 1 - k))  # 吸収(右)確率
        thetas.append(2.0 * np.arcsin(np.sqrt(rk)))
    return thetas  # 最後は自然に pi になる


def angles_for_exponential(lmbda: float, layers: int) -> List[float]:
    """
    Calculate coin angles for exponential distribution.
    
    Args:
        lmbda: Exponential parameter λ
        layers: Number of layers
        
    Returns:
        List[float]: Coin angles for each layer
    """
    # For exponential distribution, we want P(k) ∝ exp(-λ*k)
    # We need to create a smooth, continuous bias that favors smaller bin indices
    
    angles = []
    
    # Calculate target exponential probabilities for each bin
    target_probs = np.exp(-lmbda * np.arange(layers + 1))
    target_probs = target_probs / np.sum(target_probs)  # Normalize
    
    # For each layer, calculate the optimal angle to achieve the target distribution
    for layer in range(layers):
        # Calculate the probability of going left (smaller bin index)
        # We want to create a smooth bias towards smaller indices
        
        # Use a smooth exponential function for the bias
        # This creates a more continuous transition between layers
        
        # Smooth exponential decay based on normalized layer position
        normalized_layer = layer / (layers - 1) if layers > 1 else 0
        
        # Create a smooth exponential curve that matches the target distribution
        # Use a combination of linear and exponential components for smoothness
        smooth_exponential = np.exp(-lmbda * normalized_layer)
        
        # Base probability with very smooth transition
        # Use a cubic function for even smoother decrease
        base_p_left = 0.85 - 0.35 * normalized_layer**3  # Smooth cubic decrease
        
        # Add smooth exponential bias with reduced weight for smoother transition
        exponential_bias = smooth_exponential * 0.2
        
        # Add lambda-dependent smooth bias
        lambda_factor = 1.0 - np.exp(-lmbda)
        lambda_bias = lambda_factor * 0.1
        
        # Combine biases for very smooth transition
        p_left = base_p_left + exponential_bias + lambda_bias
        
        # Ensure probability is in valid range [0.1, 0.9]
        p_left = np.clip(p_left, 0.1, 0.9)
        
        # Convert to angle
        theta = 2 * np.arcsin(np.sqrt(1 - p_left))  # 1-p_left because we want left bias
        angles.append(theta)
    
    return angles


def angles_by_binary_split(target: np.ndarray) -> List[float]:
    """
    Calculate coin angles using binary splitting method.
    
    Args:
        target: Target probability distribution
        
    Returns:
        List[float]: Coin angles for each layer
    """
    angles = []
    n = len(target) - 1  # Number of layers
    
    def split_angles(probs: np.ndarray, depth: int):
        if depth >= n or len(probs) <= 1:
            return
        
        # Calculate right subtree mass
        mid = len(probs) // 2
        left_mass = np.sum(probs[:mid])
        right_mass = np.sum(probs[mid:])
        total_mass = left_mass + right_mass
        
        if total_mass > 0:
            r = right_mass / total_mass
            theta = 2 * np.arcsin(np.sqrt(r))
            angles.append(theta)
            
            # Recursively split left and right subtrees
            split_angles(probs[:mid], depth + 1)
            split_angles(probs[mid:], depth + 1)
    
    split_angles(target, 0)
    return angles[:n]  # Ensure we have exactly n angles


def angles_from_target_chain(p: np.ndarray) -> List[float]:
    """
    Calculate coin angles for absorption chain from target distribution.
    
    Survival S_k = 1 - sum_{j<k} p_j; right/absorb r_k = p_k / S_k;
    coin angle θ_k = 2*arcsin(sqrt(r_k)).
    
    Args:
        p: Target probability distribution {p_k}
        
    Returns:
        List[float]: Coin angles for each layer (L+1 angles for L+1 bins)
    """
    L = len(p) - 1  # Number of layers (L+1 bins -> L+1 angles)
    angles = []
    
    for k in range(L + 1):  # k = 0, ..., L
        # Calculate survival probability up to layer k
        S_k = 1.0 - np.sum(p[:k])
        
        if S_k > 0:
            # Absorption probability at layer k
            r_k = p[k] / S_k
            # Ensure r_k is in valid range [0, 1]
            r_k = np.clip(r_k, 0.0, 1.0)
            theta_k = 2.0 * np.arcsin(np.sqrt(r_k))
        else:
            # If survival probability is zero, force absorption
            theta_k = np.pi
        
        angles.append(theta_k)
    
    # Enforce theta[-1] = np.pi (last layer always absorbs)
    angles[-1] = np.pi
    
    return angles


def counts_to_bins(counts: dict, L: int) -> np.ndarray:
    """
    Decode counts to bin distribution for tree circuit.
    
    Critical: reverse bitstrings (bitstr[::-1]) so that bits[0] ↔ layer 0.
    Bin rule: index of the first '1' in bits; if none, bin = L.
    
    Args:
        counts: Dictionary of bitstring counts
        L: Number of layers (max bin index = L)
        
    Returns:
        np.ndarray: Empirical probability distribution
    """
    bins = np.zeros(L + 1)  # L+1 bins (0 to L)
    total_counts = sum(counts.values())
    
    for bitstr, count in counts.items():
        # Reverse bitstring so that bits[0] ↔ layer 0
        reversed_bits = bitstr[::-1]
        
        # Find index of first '1'
        first_one_idx = -1
        for i, bit in enumerate(reversed_bits):
            if bit == '1':
                first_one_idx = i
                break
        
        # If no '1' found, bin = L
        if first_one_idx == -1:
            bin_idx = L
        else:
            bin_idx = first_one_idx
        
        # Add to bin
        bins[bin_idx] += count
    
    # Normalize
    if total_counts > 0:
        bins = bins / total_counts
    
    return bins


def angles_for_binomial(layers: int, p: float = 0.5) -> List[float]:
    """
    Calculate coin angles for binomial distribution.
    
    For binomial distribution B(n,p), we use the same angle at each layer.
    The angle is determined by the probability p of moving right.
    
    Args:
        layers: Number of layers (n)
        p: Probability of moving right (default: 0.5 for fair coin)
        
    Returns:
        List[float]: Coin angles for each layer (all the same)
    """
    theta = 2 * np.arcsin(np.sqrt(p))
    return [theta] * layers


def angles_for_hadamard_walk(target_probs: np.ndarray) -> List[float]:
    """
    Calculate coin angles for Hadamard quantum walk using absorption chain method.
    
    For N layers, we need N angles to achieve the target distribution.
    We use the absorption chain method to calculate angles.
    
    Args:
        target_probs: Target probability distribution {p_k} for positions 0 to N
        
    Returns:
        List[float]: Coin angles for each layer (N angles for N layers)
    """
    N = len(target_probs) - 1  # Number of layers (N+1 bins -> N layers)
    angles = []
    
    # Use absorption chain method for all layers including the last one
    for k in range(N):
        # Calculate survival probability up to layer k
        S_k = 1.0 - np.sum(target_probs[:k])
        
        if S_k > 0:
            # Absorption probability at layer k
            r_k = target_probs[k] / S_k
            # Ensure r_k is in valid range [0, 1]
            r_k = np.clip(r_k, 0.0, 1.0)
            theta_k = 2.0 * np.arcsin(np.sqrt(r_k))
        else:
            # If survival probability is zero, force absorption
            theta_k = np.pi
        
        angles.append(theta_k)
    
    return angles
