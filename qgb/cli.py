"""
Command-line interface for Quantum Galton Board.
"""

import argparse
import json
import numpy as np
from datetime import datetime

try:
    from .circuits import (
        build_qgb_coherent, build_qgb_tree, build_qgb_gaussian, angles_for_geometric, 
        angles_by_binary_split, angles_for_geometric_lambda, angles_for_truncated_exponential,
        angles_for_hadamard_walk, angles_from_target_chain, counts_to_bins
    )
    from .targets import binomial_target, exponential_target, hadamard_walk_target
    from .samplers import sample_counts, simulate_noisy, process_tree_counts, process_gaussian_counts
    from .metrics import bootstrap_metrics, calculate_all_metrics
    from .optimize import optimize_for_gaussian, optimize_for_exponential
    from .plot import plot_distribution_comparison, plot_metrics_with_ci
except ImportError:
    from circuits import (
        build_qgb_coherent, build_qgb_tree, build_qgb_gaussian, angles_for_geometric, 
        angles_by_binary_split, angles_for_geometric_lambda, angles_for_truncated_exponential,
        angles_for_hadamard_walk
    )
    from targets import binomial_target, exponential_target, hadamard_walk_target
    from samplers import sample_counts, simulate_noisy, process_tree_counts, process_gaussian_counts
    from metrics import bootstrap_metrics, calculate_all_metrics
    from optimize import optimize_for_gaussian, optimize_for_exponential
    from plot import plot_distribution_comparison, plot_metrics_with_ci


def counts_to_probabilities(counts):
    """Convert counts to probability distribution."""
    total_shots = sum(counts.values())
    max_bin = max(int(k, 2) for k in counts.keys())
    
    probs = np.zeros(max_bin + 1)
    for bitstring, count in counts.items():
        bin_idx = int(bitstring, 2)
        probs[bin_idx] = count / total_shots
    
    return probs


def run_gaussian_experiment(args):
    """Run Gaussian distribution experiment."""
    print(f"Running Gaussian experiment with {args.layers} layers...")
    
    # Build circuit
    if args.mode == "coherent":
        circuit = build_qgb_coherent(args.layers, np.pi/2)
    else:
        print("Using traditional Gaussian (right-count = bin) approach...")
        circuit = build_qgb_gaussian(args.layers, np.pi/2)
    
    # Get target distribution
    target = binomial_target(args.layers)
    
    # Run simulation
    if args.noisy:
        counts = simulate_noisy(
            circuit, 
            backend_name=args.backend,
            noise_level=args.noise_level,
            shots=args.shots,
            seed=args.seed
        )
    else:
        counts = sample_counts(circuit, args.shots, args.seed)
    
    # Process results
    if args.mode == "tree":
        empirical = process_gaussian_counts(counts)
    else:
        empirical = counts_to_probabilities(counts)
    
    # Ensure same length
    min_len = min(len(empirical), len(target))
    empirical = empirical[:min_len]
    target = target[:min_len]
    
    # Calculate metrics
    metrics = calculate_all_metrics(empirical, target)
    bootstrap_results = bootstrap_metrics(counts, target, n_boot=1000)
    
    # Optimization if requested
    optimization_results = None
    if args.optimize and args.noisy:
        print("Running angle optimization...")
        optimized_angles, optimized_tv = optimize_for_gaussian(
            args.layers, args.noise_level, args.shots // 4
        )
        optimization_results = {
            'original_tv': metrics['tv'],
            'optimized_tv': optimized_tv,
            'original_angles': [np.pi/2] * args.layers,
            'optimized_angles': optimized_angles
        }
    
    return {
        'experiment_type': 'gaussian',
        'layers': args.layers,
        'mode': args.mode,
        'noisy': args.noisy,
        'shots': args.shots,
        'empirical': empirical.tolist(),
        'target': target.tolist(),
        'metrics': metrics,
        'bootstrap_results': bootstrap_results,
        'optimization_results': optimization_results
    }


def run_exponential_experiment(args):
    """Run exponential distribution experiment."""
    print(f"Running exponential experiment with λ={args.lambda_param}...")
    
    # Use truncated exponential as the default method
    print("Using truncated exponential (exact match)...")
    angles = angles_for_truncated_exponential(args.lambda_param, args.layers)
    bins = args.layers + 1  # L+1 bins for L layers
    
    # Build circuit
    if args.mode == "coherent":
        circuit = build_qgb_coherent(args.layers, angles)
    else:
        circuit = build_qgb_tree(args.layers, angles)
    
    # Get target distribution
    target = exponential_target(args.lambda_param, bins)
    
    # Run simulation
    if args.noisy:
        counts = simulate_noisy(
            circuit,
            backend_name=args.backend,
            noise_level=args.noise_level,
            shots=args.shots,
            seed=args.seed
        )
    else:
        counts = sample_counts(circuit, args.shots, args.seed)
    
    # Process results
    if args.mode == "tree":
        empirical = process_tree_counts(counts)
    else:
        empirical = counts_to_probabilities(counts)
    
    # Ensure same length
    min_len = min(len(empirical), len(target))
    empirical = empirical[:min_len]
    target = target[:min_len]
    
    # Calculate metrics
    metrics = calculate_all_metrics(empirical, target)
    bootstrap_results = bootstrap_metrics(counts, target, n_boot=1000)
    
    # Optimization if requested
    optimization_results = None
    if args.optimize and args.noisy:
        print("Running angle optimization...")
        optimized_angles, optimized_tv = optimize_for_exponential(
            args.layers, args.lambda_param, args.noise_level, args.shots // 4
        )
        optimization_results = {
            'original_tv': metrics['tv'],
            'optimized_tv': optimized_tv,
            'original_angles': angles,
            'optimized_angles': optimized_angles
        }
    
    return {
        'experiment_type': 'exponential',
        'layers': args.layers,
        'lambda': args.lambda_param,
        'mode': args.mode,
        'noisy': args.noisy,
        'shots': args.shots,
        'empirical': empirical.tolist(),
        'target': target.tolist(),
        'metrics': metrics,
        'bootstrap_results': bootstrap_results,
        'optimization_results': optimization_results
    }


def run_hadamard_experiment(args):
    """Run Hadamard quantum walk experiment."""
    print(f"Running Hadamard quantum walk experiment with {args.steps} steps...")
    print(f"Initial state: {args.init}")
    print(f"Tree kind: {args.tree_kind}")
    
    # Get target distribution
    target, positions = hadamard_walk_target(args.steps, args.init)
    print(f"Target distribution shape: {len(target)}")
    print(f"Positions: {positions}")
    
    # Calculate angles based on tree kind
    if args.tree_kind == "chain":
        print("Using absorption chain angles...")
        angles = angles_from_target_chain(target)
    else:  # balanced
        print("Using balanced binary split angles...")
        angles = angles_by_binary_split(target)
    
    print(f"Generated {len(angles)} angles: {angles}")
    
    # Build tree circuit (one qubit per layer)
    circuit = build_qgb_tree(len(angles), angles)
    
    # Run simulation
    if args.noisy:
        counts = simulate_noisy(
            circuit,
            backend_name=args.backend,
            noise_level=args.noise_level,
            shots=args.shots,
            seed=args.seed
        )
    else:
        counts = sample_counts(circuit, args.shots, args.seed)
    
    # Decode counts to bins using counts_to_bins
    L = len(angles) - 1  # Number of layers
    empirical = counts_to_bins(counts, L)
    
    # Ensure same length
    min_len = min(len(empirical), len(target))
    empirical = empirical[:min_len]
    target = target[:min_len]
    positions = positions[:min_len]
    
    # Calculate metrics
    metrics = calculate_all_metrics(empirical, target, positions)
    bootstrap_results = bootstrap_metrics(counts, target, positions, n_boot=1000)
    
    return {
        'experiment_type': 'hadamard_walk',
        'steps': args.steps,
        'init': args.init,
        'tree_kind': args.tree_kind,
        'noisy': args.noisy,
        'shots': args.shots,
        'empirical': empirical.tolist(),
        'target': target.tolist(),
        'positions': positions.tolist(),
        'metrics': metrics,
        'bootstrap_results': bootstrap_results
    }


def print_results(results):
    """Print experiment results."""
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    print(f"Experiment Type: {results['experiment_type']}")
    print(f"Mode: {results['mode']}")
    print(f"Noisy: {results['noisy']}")
    print(f"Shots: {results['shots']}")
    
    if 'layers' in results:
        print(f"Layers: {results['layers']}")
    if 'lambda' in results:
        print(f"Lambda: {results['lambda']}")
    if 'steps' in results:
        print(f"Steps: {results['steps']}")
    
    print("\nDistance Metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric.upper()}: {value:.6f}")
    
    print("\nBootstrap Results (95% CI):")
    for metric, stats in results['bootstrap_results'].items():
        print(f"  {metric.upper()}: {stats['mean']:.6f} [{stats['lower']:.6f}, {stats['upper']:.6f}]")
    
    if results.get('optimization_results'):
        opt = results['optimization_results']
        print(f"\nOptimization Results:")
        print(f"  Original TV: {opt['original_tv']:.6f}")
        print(f"  Optimized TV: {opt['optimized_tv']:.6f}")
        print(f"  Improvement: {opt['original_tv'] - opt['optimized_tv']:.6f}")


def save_results(results, args):
    """Save results to files."""
    import os
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"outputs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n📁 Creating output directory: {run_dir}")
    
    # Save JSON results
    json_filename = f"{run_dir}/{results['experiment_type']}_results.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_filename}")
    
    # Create plots
    if args.plot:
        empirical = np.array(results['empirical'])
        target = np.array(results['target'])
        
        # Distribution comparison plot
        plot_filename = f"{run_dir}/{results['experiment_type']}_comparison.png"
        plot_distribution_comparison(
            empirical, target,
            title=f"{results['experiment_type'].title()} Distribution",
            save_path=plot_filename
        )
        print(f"Plot saved to: {plot_filename}")
        
        # Metrics plot
        metrics_filename = f"{run_dir}/{results['experiment_type']}_metrics.png"
        plot_metrics_with_ci(
            results['bootstrap_results'],
            title=f"{results['experiment_type'].title()} Metrics",
            save_path=metrics_filename
        )
        print(f"Metrics plot saved to: {metrics_filename}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum Galton Board CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m qgb gaussian --layers 12 --shots 20000 --mode coherent
  python -m qgb exponential --layers 10 --lambda 0.35 --mode tree --shots 20000
  python -m qgb hadamard --steps 12 --mode tree --noisy true --optimize true
        """
    )
    

    
    # Subparsers for different experiments
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Gaussian subcommand
    gaussian_parser = subparsers.add_parser('gaussian', help='Gaussian distribution experiment')
    gaussian_parser.add_argument('--layers', type=int, default=12, help='Number of layers')
    gaussian_parser.add_argument('--shots', type=int, default=20000, help='Number of shots')
    gaussian_parser.add_argument('--mode', choices=['coherent', 'tree'], default='tree', help='Circuit mode')
    gaussian_parser.add_argument('--noisy', action='store_true', help='Use noisy simulation')
    gaussian_parser.add_argument('--backend', type=str, help='Fake backend name')
    gaussian_parser.add_argument('--noise-level', choices=['low', 'medium', 'high'], default='medium', help='Noise level')
    gaussian_parser.add_argument('--optimize', action='store_true', help='Enable angle optimization')
    gaussian_parser.add_argument('--plot', action='store_true', help='Generate plots')
    gaussian_parser.add_argument('--seed', type=int, default=123, help='Random seed')
    
    # Exponential subcommand
    exp_parser = subparsers.add_parser('exponential', help='Exponential distribution experiment')
    exp_parser.add_argument('--layers', type=int, default=10, help='Number of layers')
    exp_parser.add_argument('--lambda', type=float, default=0.35, dest='lambda_param', help='Exponential parameter λ')
    exp_parser.add_argument('--shots', type=int, default=20000, help='Number of shots')
    exp_parser.add_argument('--mode', choices=['coherent', 'tree'], default='tree', help='Circuit mode')
    exp_parser.add_argument('--noisy', action='store_true', help='Use noisy simulation')
    exp_parser.add_argument('--backend', type=str, help='Fake backend name')
    exp_parser.add_argument('--noise-level', choices=['low', 'medium', 'high'], default='medium', help='Noise level')
    exp_parser.add_argument('--optimize', action='store_true', help='Enable angle optimization')
    exp_parser.add_argument('--plot', action='store_true', help='Generate plots')
    exp_parser.add_argument('--seed', type=int, default=123, help='Random seed')
    
    # Hadamard subcommand
    hadamard_parser = subparsers.add_parser('hadamard', help='Hadamard quantum walk experiment')
    hadamard_parser.add_argument('--steps', type=int, default=12, help='Number of walk steps')
    hadamard_parser.add_argument('--init', choices=['symmetric', 'zero'], default='symmetric', 
                                help='Initial coin state: symmetric=(|0⟩+i|1⟩)/√2, zero=|0⟩')
    hadamard_parser.add_argument('--tree_kind', choices=['chain', 'balanced'], default='chain',
                                help='Tree kind: chain=absorption chain, balanced=binary split')
    hadamard_parser.add_argument('--shots', type=int, default=200000, help='Number of shots')
    hadamard_parser.add_argument('--noisy', action='store_true', help='Use noisy simulation')
    hadamard_parser.add_argument('--backend', type=str, help='Fake backend name')
    hadamard_parser.add_argument('--noise-level', choices=['low', 'medium', 'high'], default='medium', help='Noise level')
    hadamard_parser.add_argument('--plot', action='store_true', help='Generate plots')
    hadamard_parser.add_argument('--seed', type=int, default=123, help='Random seed')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run experiment
    if args.command == 'gaussian':
        results = run_gaussian_experiment(args)
    elif args.command == 'exponential':
        results = run_exponential_experiment(args)
    elif args.command == 'hadamard':
        results = run_hadamard_experiment(args)
    else:
        print(f"Unknown command: {args.command}")
        return
    
    # Print and save results
    print_results(results)
    save_results(results, args)


if __name__ == "__main__":
    main()
