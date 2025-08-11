"""
Sampling utilities for Quantum Galton Board.

Reference: arXiv:2202.01735 - Universal Statistical Simulator
"""

import numpy as np
from typing import Dict, Optional
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def check_gpu_support() -> bool:
    """
    Check if GPU acceleration is supported on this system.
    
    Returns:
        bool: True if GPU is supported, False otherwise
    """
    try:
        # Try to create a simple GPU backend
        backend = AerSimulator(
            method='statevector',
            device='GPU'
        )
        # If no exception is raised, GPU is supported
        return True
    except Exception:
        return False


def sample_counts(
    circuit: QuantumCircuit, 
    shots: int = 20000, 
    seed: int = 123,
    use_gpu: bool = True
) -> Dict[str, int]:
    """
    Sample circuit using Aer simulator with optional GPU acceleration.
    
    Args:
        circuit: Quantum circuit to sample
        shots: Number of shots
        seed: Random seed
        use_gpu: Whether to use GPU acceleration (macOS Metal)
        
    Returns:
        Dict[str, int]: Counts dictionary
    """
    try:
        if use_gpu:
            # Try to use GPU acceleration on macOS
            backend = AerSimulator(
                method='statevector',
                device='GPU',
                seed_simulator=seed
            )
        else:
            # Fallback to CPU
            backend = AerSimulator(seed_simulator=seed)
            
        job = backend.run(circuit, shots=shots)
        result = job.result()
        
        return result.get_counts()
        
    except Exception as e:
        if use_gpu:
            print(f"GPU acceleration failed, falling back to CPU: {e}")
        # Fallback to CPU
        backend = AerSimulator(seed_simulator=seed)
        job = backend.run(circuit, shots=shots)
        result = job.result()
        
        return result.get_counts()


def simulate_noisy(
    circuit: QuantumCircuit,
    backend_name: Optional[str] = None,
    noise_level: str = "medium",
    shots: int = 20000,
    seed: int = 123,
    use_gpu: bool = True
) -> Dict[str, int]:
    """
    Simulate circuit with noise.
    
    Args:
        circuit: Quantum circuit to simulate
        backend_name: Name of fake backend to use
        noise_level: Noise level ("noiseless", "low", "medium", "high")
        shots: Number of shots
        seed: Random seed
        
    Returns:
        Dict[str, int]: Counts dictionary
    """
    # For noiseless simulation, use sample_counts
    if noise_level == "noiseless":
        return sample_counts(circuit, shots=shots, seed=seed, use_gpu=use_gpu)
    try:
        # Try to use fake backend
        if backend_name:
            from qiskit.providers.fake_provider import get_backend
            backend = get_backend(backend_name)
        else:
            # Use default fake backend
            try:
                from qiskit.providers.fake_provider import FakeSherbrookeV2
                backend = FakeSherbrookeV2()
            except ImportError:
                try:
                    from qiskit.providers.fake_provider import FakeManilaV2
                    backend = FakeManilaV2()
                except ImportError:
                    # Fallback to custom noise model directly
                    from .noise import get_noise_model
                    noise_model = get_noise_model(noise_level)
                    backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
                    job = backend.run(circuit, shots=shots)
                    result = job.result()
                    return result.get_counts()
        
        job = backend.run(circuit, shots=shots)
        result = job.result()
        
        return result.get_counts()
        
    except ImportError:
        # Fallback to custom noise model
        from .noise import get_noise_model
        
        noise_model = get_noise_model(noise_level)
        try:
            if use_gpu:
                # Try GPU with noise model
                backend = AerSimulator(
                    method='statevector',
                    device='GPU',
                    noise_model=noise_model,
                    seed_simulator=seed
                )
            else:
                # CPU with noise model
                backend = AerSimulator(
                    noise_model=noise_model,
                    seed_simulator=seed
                )
        except Exception as e:
            if use_gpu:
                print(f"GPU with noise model failed, falling back to CPU: {e}")
            # Fallback to CPU with noise model
            backend = AerSimulator(
                noise_model=noise_model,
                seed_simulator=seed
            )
        
        job = backend.run(circuit, shots=shots)
        result = job.result()
        
        return result.get_counts()


def counts_to_probabilities(counts: Dict[str, int]) -> np.ndarray:
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


def process_tree_counts(counts: Dict[str, int], num_layers: int = None) -> np.ndarray:
    """
    Process counts from tree QGB to get bin distribution.
    
    Qiskit のbit文字列は右端が cl[0] になるので反転して解釈。
    反転後 bits[0] が層0に対応。最初に '1' が出た位置が bin。なければ bin=L。
    
    Args:
        counts: Counts dictionary from tree QGB
        num_layers: Number of layers (if specified, ensures L+1 bins)
        
    Returns:
        np.ndarray: Bin probability distribution
    """
    from collections import Counter
    
    bins = Counter()
    for bitstr, n in counts.items():
        bits = bitstr[::-1]  # 反転
        first1 = bits.find("1")
        b = first1 if first1 != -1 else len(bits)  # 全ゼロなら最後のbin (L)
        bins[b] += n
    
    shots = sum(counts.values())
    
    if num_layers is not None:
        # Ensure we have exactly num_layers + 1 bins (0 to num_layers)
        hist = np.zeros(num_layers + 1)
        for bin_idx in range(num_layers + 1):
            hist[bin_idx] = bins.get(bin_idx, 0) / shots
    else:
        # Original behavior: use max bin found
        max_bin = max(bins.keys()) if bins else 0
        hist = np.array([bins[k] for k in range(max_bin + 1)], dtype=float) / shots
    
    return hist


def process_gaussian_counts(counts: Dict[str, int]) -> np.ndarray:
    """
    Process counts from Gaussian QGB to get bin distribution.
    
    For Gaussian distribution: count number of '1's in each bitstring to get bin index.
    This is the traditional "right-count = bin" approach.
    
    Args:
        counts: Counts dictionary from Gaussian QGB
        
    Returns:
        np.ndarray: Bin probability distribution
    """
    total_shots = sum(counts.values())
    
    # Count number of '1's in each bitstring to get bin index
    bin_counts = {}
    for bitstring, count in counts.items():
        bin_idx = bitstring.count('1')  # Number of right moves
        bin_counts[bin_idx] = bin_counts.get(bin_idx, 0) + count
    
    # Find the maximum bin index that actually has counts
    max_bin = max(bin_counts.keys()) if bin_counts else 0
    
    # Create probability array with correct length
    probs = np.zeros(max_bin + 1)
    for bin_idx, count in bin_counts.items():
        probs[bin_idx] = count / total_shots
    
    return probs
