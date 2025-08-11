# Quantum Galton Board

A Python implementation of the Quantum Galton Board (QGB) based on the Carney-Varcoe paper "Universal Statistical Simulator" (arXiv:2202.01735).

## 🎯 Overview

This project implements a quantum version of the classical Galton board, capable of generating various probability distributions including:

- **Gaussian/Binomial distributions** (unbiased pegs)
- **Exponential distributions** (biased pegs)
- **Hadamard quantum walk distributions** (quantum interference patterns)

The implementation supports both coherent and depth-optimized tree modes, with comprehensive noise modeling, statistical error analysis, and advanced visualization capabilities.

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

### Unified Experiment Runner

For comprehensive experiments with all distributions and visualizations:

```bash
# Run unified experiment with all distributions (layers 2, 4, 6, 8)
python run.py unified

# Run with CPU (if GPU not available)
python run.py unified --use-cpu

# Run with GPU acceleration
python run.py unified --use-gpu
```

### Individual Distribution Experiments

```bash
# Gaussian distribution
python run.py gaussian --layers 8 --shots 20000

# Exponential distribution
python run.py exponential --layers 8 --lambda 0.35 --shots 20000

# Hadamard quantum walk
python run.py hadamard --layers 8 --shots 20000
```

### Visualization

```bash
# Generate all visualizations from latest results
python visualize.py

# Generate specific visualizations
python visualize.py --distance-scaling
python visualize.py --layer-comparison
```

### Full Command Line Interface

For advanced usage, use the comprehensive CLI:

```bash
# Gaussian distribution experiment
python -m qgb gaussian --layers 12 --shots 20000 --mode coherent

# Exponential distribution experiment
python -m qgb exponential --layers 10 --lambda 0.35 --mode tree --noisy

# Hadamard quantum walk experiment
python -m qgb hadamard --steps 12 --mode tree --noisy --plot
```

### CLI Options

#### Common Options
- `--shots`: Number of shots (default: 20000)
- `--mode`: Circuit mode (`coherent` or `tree`, default: `tree`)
- `--noisy`: Enable noisy simulation
- `--backend`: Fake backend name for noise simulation
- `--noise-level`: Noise level (`noiseless`, `low`, `high`, default: `low`)
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
- `--init`: Initial coin state (`symmetric` or `zero`, default: `symmetric`)
- `--tree-kind`: Angle synthesis method (`absorption` or `binary`, default: `absorption`)

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

Support for realistic quantum hardware noise with unified noise levels:

- **Noiseless**: Pure quantum simulation without noise
- **Low Noise**: Minimal noise conditions
- **High Noise**: High noise conditions

The implementation supports:
- **Fake Backends**: Integration with Qiskit fake providers
- **Custom Noise Models**: Depolarizing, amplitude/phase damping, readout errors
- **Error Mitigation**: Measurement error mitigation and twirling options

### Advanced Visualization

Comprehensive visualization system with statistical error bars:

- **Layer Comparison Plots**: Target vs empirical distributions for each layer
- **Distance Layer Scaling**: Distance metrics vs layer count with confidence intervals
- **Combined Visualizations**: All distributions and metrics in unified plots
- **Statistical Error Bars**: 95% confidence intervals for all distance metrics
- **Offset Plotting**: Horizontal offsets prevent error bar overlap

### Angle Optimization

Automatic angle pre-compensation for noise reduction:

```python
from qgb import optimize_for_gaussian

optimized_angles, optimized_tv = optimize_for_gaussian(
    layers=8, noise_level="high", shots=5000
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
- **Hadamard Quantum Walk**: Angle synthesis and target distribution validation

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
- **Layers**: 2, 4, 6, 8
- **Expected TV Distance**: < 0.05 (noiseless)
- **Shape**: Bell curve (binomial ≈ Gaussian)
- **Angle Calculation**: Uses `angles_for_binomial` for correct binomial distribution

#### Exponential Distribution
- **Layers**: 2, 4, 6, 8
- **Lambda**: 0.35
- **Expected Shape**: Exponential decay
- **Optimization**: Reduces TV distance under noise

#### Hadamard Quantum Walk
- **Layers**: 2, 4, 6, 8
- **Expected Shape**: Characteristic peaks and valleys with quantum interference
- **Features**: Absorption chain implementation with N+1 bins for N layers
- **Target Distribution**: Pre-calculated theoretical distributions for specific layers

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

#### Hadamard Quantum Walk Implementation

The Hadamard quantum walk uses an absorption chain approach:

- **Layers**: N layers produce N+1 bins (positions 0 to N)
- **Angle Synthesis**: Absorption chain method for angle calculation
- **Target Distribution**: Pre-calculated theoretical distributions for layers 2, 4, 6, 8
- **Decoding**: `process_tree_counts` with `num_layers` parameter for correct binning

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

### Statistical Error Analysis

All distance metrics include bootstrap confidence intervals:

- **Bootstrap Resampling**: 1000 resamples for robust statistics
- **95% Confidence Intervals**: Lower and upper bounds for each metric
- **Error Bar Visualization**: Statistical uncertainty displayed in all plots
- **Offset Plotting**: Horizontal offsets prevent error bar overlap

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
├── run.py                  # Unified experiment runner
├── visualize.py            # Comprehensive visualization script
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## 🔧 Recent Updates

### Latest Improvements (2025-08-11)

1. **Statistical Error Visualization**:
   - Added bootstrap confidence intervals (95% CI) for all distance metrics
   - Implemented error bar plotting in all distance scaling visualizations
   - Added horizontal offsets to prevent error bar overlap between noise levels

2. **Unified Noise Levels**:
   - Standardized noise levels: `noiseless`, `low`, `high`
   - Updated all plotting functions with unified color scheme
   - Consistent noise level handling across all distributions

3. **Hadamard Quantum Walk Enhancements**:
   - Fixed target distribution calculation for layers 2, 4, 6, 8
   - Implemented absorption chain angle synthesis
   - Corrected bin indexing (N+1 bins for N layers)
   - Added proper statistical error analysis

4. **Gaussian Distribution Corrections**:
   - Implemented `angles_for_binomial` for correct binomial distribution
   - Fixed angle calculation in Gaussian QGB implementation
   - Ensured proper target-empirical distribution matching

5. **Visualization Improvements**:
   - Target distributions displayed as line graphs (not bar charts)
   - Integer x-axis ticks for all layer comparison plots
   - Unified output directory structure with timestamped folders
   - Comprehensive combined visualizations

6. **Data Management**:
   - Bootstrap results saved for all distributions
   - Automatic data regeneration for missing or outdated results
   - Robust data loading with fallback mechanisms

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
- **Interactive Visualization**: Real-time plots and circuit diagrams
- **Extended Statistical Analysis**: Additional confidence interval methods

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the test examples

---

**Note**: This implementation is based on the theoretical framework described in the Carney-Varcoe paper. For production use, consider hardware-specific optimizations and error mitigation strategies. The latest version includes comprehensive statistical error analysis and improved visualization capabilities. 
