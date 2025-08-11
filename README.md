# Quantum Galton Board

A Python implementation of the Quantum Galton Board (QGB) based on the Carney-Varcoe paper "Universal Statistical Simulator" (arXiv:2202.01735).

## 🎯 Overview

This project implements a quantum version of the classical Galton board, capable of generating various probability distributions including:

- **Gaussian/Binomial distributions** (unbiased pegs)
- **Exponential distributions** (biased pegs)
- **Hadamard quantum walk distributions** (quantum interference patterns)

The implementation supports both coherent and depth-optimized tree modes, with noise modeling and error mitigation capabilities.

## 🏗️ Architecture

### Core Components

- **`qgb/circuits.py`**: Quantum circuit implementations (coherent and tree modes)
- **`qgb/targets.py`**: Target distribution generators
- **`qgb/samplers.py`**: Quantum simulation and sampling utilities
- **`qgb/noise.py`**: Noise models and error mitigation
- **`qgb/metrics.py`**: Distance metrics and bootstrap confidence intervals
- **`qgb/optimize.py`**: Angle optimization for noise compensation
- **`qgb/plot.py`**: Visualization utilities
- **`qgb/cli.py`**: Command-line interface

### Circuit Modes

1. **Coherent Mode**: Full quantum implementation using cSWAP gates
2. **Tree Mode**: Depth-optimized implementation with mid-circuit measurements

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Install Dependencies

```bash
pip install qiskit>=1.2 qiskit-aer>=0.14 numpy>=1.24 scipy>=1.10 matplotlib>=3.6
```

### Optional Development Dependencies

```bash
pip install pytest>=7.0 black>=23.0 ruff>=0.1.0
```

## 📖 Usage

### Simple Interface

For quick experiments, use the simplified runner:

```bash
# Run all experiments in one execution
python run.py all --layers 8
python run.py all --layers 8 --lambda 0.5 --steps 8

# With noise and plots
python run.py all --layers 6 --noisy --plot

# Generate all visualizations
python run.py visualize
```

### Full Command Line Interface

For advanced usage, use the comprehensive CLI:

```bash
# Gaussian distribution experiment
python -m qgb gaussian --layers 12 --shots 20000 --mode coherent

# Exponential distribution experiment
python -m qgb exponential --layers 10 --lambda 0.35 --mode tree --noisy

# Hadamard quantum walk experiment
python -m qgb hadamard --steps 12 --mode tree --noisy --optimize --plot
```

### CLI Options

#### Common Options
- `--shots`: Number of shots (default: 20000)
- `--mode`: Circuit mode (`coherent` or `tree`, default: `tree`)
- `--noisy`: Enable noisy simulation
- `--backend`: Fake backend name for noise simulation
- `--noise-level`: Noise level (`low`, `medium`, `high`, default: `medium`)
- `--optimize`: Enable angle optimization
- `--plot`: Generate plots
- `--seed`: Random seed (default: 123)

#### Gaussian Experiment
- `--layers`: Number of layers (default: 12)

#### Exponential Experiment
- `--layers`: Number of layers (default: 10)
- `--lambda`: Exponential parameter λ (default: 0.35)
- `--exact`: Use exact truncated exponential (default: geometric approximation)

#### Hadamard Experiment
- `--steps`: Number of walk steps (default: 12)

### Python API

```python
from qgb import (
    build_qgb_tree, binomial_target, exponential_target,
    sample_counts, calculate_all_metrics
)

# Build circuit
circuit = build_qgb_tree(layers=8, coin_angles=np.pi/2)

# Get target distribution
target = binomial_target(layers=8)

# Run simulation
counts = sample_counts(circuit, shots=10000)

# Calculate metrics
metrics = calculate_all_metrics(empirical, target)
```

## 📊 Features

### Distance Metrics

The implementation provides comprehensive distance metrics:

- **Total Variation Distance**: Measures overall distribution difference
- **Hellinger Distance**: Measures distribution similarity
- **KL Divergence**: Information-theoretic distance (with ε-smoothing)
- **Wasserstein-1 Distance**: Earth mover's distance

### Bootstrap Confidence Intervals

All metrics include bootstrap confidence intervals (95% CI) to account for statistical uncertainty:

```python
from qgb import bootstrap_metrics

results = bootstrap_metrics(counts, target, n_boot=1000)
# Returns: {'tv': {'mean': 0.123, 'lower': 0.118, 'upper': 0.128, 'std': 0.002}, ...}
```

### Noise Modeling

Support for realistic quantum hardware noise:

- **Fake Backends**: Integration with Qiskit fake providers
- **Custom Noise Models**: Depolarizing, amplitude/phase damping, readout errors
- **Error Mitigation**: Measurement error mitigation and twirling options

### Angle Optimization

Automatic angle pre-compensation for noise reduction:

```python
from qgb import optimize_for_gaussian

optimized_angles, optimized_tv = optimize_for_gaussian(
    layers=8, noise_level="medium", shots=5000
)
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

### Test Coverage

- **Target Distributions**: Verification of probability normalization and expected shapes
- **Distance Metrics**: Validation of metric properties and edge cases
- **Quantum Circuits**: Circuit construction and Gaussian convergence tests

## 📈 Examples

### Quick Start Scripts

The `examples/` directory contains quick start scripts:

```bash
# Gaussian distribution
./examples/quickstart_gaussian.sh

# Exponential distribution
./examples/quickstart_exponential.sh

# Hadamard quantum walk
./examples/quickstart_hadamard.sh
```

### Expected Results

#### Gaussian Distribution
- **Layers**: 12
- **Expected TV Distance**: < 0.05 (noiseless)
- **Shape**: Bell curve (binomial ≈ Gaussian)

#### Exponential Distribution
- **Layers**: 10
- **Lambda**: 0.35
- **Expected Shape**: Exponential decay
- **Optimization**: Reduces TV distance under noise

#### Hadamard Quantum Walk
- **Steps**: 12
- **Expected Shape**: Characteristic peaks and valleys
- **Features**: Quantum interference patterns

## 🔬 Technical Details

### Circuit Implementation

#### Coherent Mode
- Uses cSWAP (Fredkin) gates for peg implementation
- Full quantum superposition maintained
- Higher circuit depth but preserves quantum coherence

#### Tree Mode
- Mid-circuit measurements for path decisions
- Classical branching with quantum coin flips
- Optimized for depth and noise tolerance

### Target Distribution Synthesis

#### Binary Splitting Method
For arbitrary target distributions, the implementation uses a binary splitting approach:

1. Calculate right subtree mass ratio
2. Set rotation angle: θ = 2·arcsin(√r)
3. Recursively apply to left and right subtrees

#### Exponential Distribution Mapping

The implementation provides two methods for mapping exponential rate λ to quantum circuit angles:

**A. Geometric Approximation (Default)**
For discretized exponential with rate λ:
- Absorb-right probability: r = 1 - exp(-λ)
- Rotation angle: θ = 2·arcsin(√r)
- Uses the same angle at each layer
- Optional: Force absorption at final layer with θ_L = π

**B. Truncated Exponential (Exact Match)**
For exact truncated exponential over bins 0..L:
- Let q = exp(-λ)
- At layer k (0-index): r_k = (1 - q) / (1 - q^(L+1-k))
- Rotation angle: θ_k = 2·arcsin(√r_k)
- Yields exact match to truncated exponential

**CLI Usage:**
```bash
# Geometric approximation (default)
python -m qgb exponential --layers 10 --lambda 0.35 --mode tree --exact false --shots 20000

# Truncated exact match
python -m qgb exponential --layers 10 --lambda 0.35 --mode tree --exact true --shots 20000
```

### Noise Compensation

The optimization module implements grid search for angle pre-compensation:

1. Evaluate TV distance around current angles
2. Select angles that minimize distance under noise
3. Iterate until convergence

## 📁 Project Structure

```
quantum_galton_board/
├── qgb/                    # Main package
│   ├── __init__.py
│   ├── circuits.py         # Quantum circuit implementations
│   ├── targets.py          # Target distribution generators
│   ├── samplers.py         # Simulation and sampling
│   ├── noise.py            # Noise models and mitigation
│   ├── metrics.py          # Distance metrics and bootstrap
│   ├── optimize.py         # Angle optimization
│   ├── plot.py             # Visualization utilities
│   ├── cli.py              # Command-line interface
│   └── __main__.py         # Package entry point
├── tests/                  # Unit tests
├── examples/               # Example scripts
├── outputs/                # Generated plots and results
│   └── YYYYMMDD_HHMMSS/    # Timestamped run folders
├── run.py                  # Simple experiment runner
├── visualize.py            # Unified visualization script
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## 📚 References

- **Primary Reference**: Carney, D., & Varcoe, B. (2022). Universal Statistical Simulator. arXiv:2202.01735
- **Qiskit Documentation**: https://qiskit.org/documentation/
- **Quantum Walks**: Kempe, J. (2003). Quantum random walks - an introductory overview

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Work

- **Hardware Integration**: Real quantum device support
- **Advanced Error Mitigation**: Zero-noise extrapolation, PEC
- **Distribution Synthesis**: Support for arbitrary target distributions
- **Performance Optimization**: Circuit compilation and transpilation
- **Visualization**: Interactive circuit diagrams and real-time plots

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the test examples

---

**Note**: This implementation is based on the theoretical framework described in the Carney-Varcoe paper. For production use, consider hardware-specific optimizations and error mitigation strategies. 
