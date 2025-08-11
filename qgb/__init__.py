"""
Quantum Galton Board implementation based on Carney-Varcoe paper.

Reference: arXiv:2202.01735 - Universal Statistical Simulator
"""

__version__ = "0.1.0"
__author__ = "Quantum Galton Board Team"

from .circuits import (
    peg_cswap,
    build_qgb_coherent,
    build_qgb_tree,
    angles_for_geometric,
    angles_by_binary_split,
)
from .targets import (
    binomial_target,
    exponential_target,
    hadamard_walk_target,
)
from .samplers import sample_counts, simulate_noisy
from .noise import get_noise_model
from .metrics import (
    total_variation,
    hellinger,
    kl_div,
    wasserstein1,
    bootstrap_metrics,
)
from .optimize import precompensate_angles
from .plot import plot_distribution_comparison

__all__ = [
    "peg_cswap",
    "build_qgb_coherent",
    "build_qgb_tree",
    "angles_for_geometric",
    "angles_by_binary_split",
    "binomial_target",
    "exponential_target",
    "hadamard_walk_target",
    "sample_counts",
    "simulate_noisy",
    "get_noise_model",
    "total_variation",
    "hellinger",
    "kl_div",
    "wasserstein1",
    "bootstrap_metrics",
    "precompensate_angles",
    "plot_distribution_comparison",
]
