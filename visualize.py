#!/usr/bin/env python3
"""
Unified visualization script for Quantum Galton Board.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# Import our modules
from qgb.circuits import build_qgb_tree, angles_for_geometric, angles_for_exponential, angles_by_binary_split
from qgb.targets import binomial_target, exponential_target, hadamard_walk_target
from qgb.samplers import sample_counts, simulate_noisy, process_tree_counts
from qgb.metrics import calculate_all_metrics, bootstrap_metrics
from qgb.plot import plot_distribution_comparison, plot_metrics_with_ci


def run_experiment(experiment_type, **kwargs):
    """Run a single experiment and return results."""
    print(f"Running {experiment_type} experiment...")
    
    if experiment_type == "gaussian":
        layers = kwargs.get('layers', 8)
        shots = kwargs.get('shots', 5000)
        
        # Build circuit
        circuit = build_qgb_tree(layers, np.pi/2)
        target = binomial_target(layers)
        
    elif experiment_type == "exponential":
        layers = kwargs.get('layers', 8)
        lmbda = kwargs.get('lambda_param', 0.5)
        shots = kwargs.get('shots', 5000)
        
        # Calculate angles for exponential distribution
        angles = angles_for_exponential(lmbda, layers)
        circuit = build_qgb_tree(layers, angles)
        target = exponential_target(lmbda, layers + 1)
        
    elif experiment_type == "hadamard":
        steps = kwargs.get('steps', 8)
        shots = kwargs.get('shots', 5000)
        
        # Get target distribution
        target, positions = hadamard_walk_target(steps)
        angles = angles_by_binary_split(target)
        circuit = build_qgb_tree(len(angles), angles)
        
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Run simulation
    if kwargs.get('noisy', False):
        noise_level = kwargs.get('noise_level', 'medium')
        counts = simulate_noisy(circuit, noise_level=noise_level, shots=shots)
    else:
        counts = sample_counts(circuit, shots=shots)
    
    # Process results
    empirical = process_tree_counts(counts)
    
    # Ensure same length
    min_len = min(len(empirical), len(target))
    empirical = empirical[:min_len]
    target = target[:min_len]
    
    # Calculate metrics
    metrics = calculate_all_metrics(empirical, target)
    bootstrap_results = bootstrap_metrics(counts, target, n_boot=500)
    
    return empirical, target, metrics, bootstrap_results


def create_plots(experiments, save_dir="outputs"):
    """Create all plots for the experiments."""
    try:
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(save_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"📁 Creating output directory: {run_dir}")
        
        print(f"Number of experiments: {len(experiments)}")
        for exp_type in experiments.keys():
            print(f"  - {exp_type}")
        
        # Individual distribution plots
        for exp_type, results in experiments.items():
            try:
                print(f"Processing {exp_type} experiment...")
                empirical, target, metrics, bootstrap_results = results
                
                # Distribution comparison plot
                plot_filename = f"{run_dir}/{exp_type}_comparison.png"
                print(f"Creating distribution plot: {plot_filename}")
                plot_distribution_comparison(
                    empirical, target,
                    title=f"{exp_type.title()} Distribution",
                    save_path=plot_filename
                )
                print(f"Distribution plot saved: {plot_filename}")
                
                # Metrics plot
                metrics_filename = f"{run_dir}/{exp_type}_metrics.png"
                print(f"Creating metrics plot: {metrics_filename}")
                plot_metrics_with_ci(
                    bootstrap_results,
                    title=f"{exp_type.title()} Metrics",
                    save_path=metrics_filename
                )
                print(f"Metrics plot saved: {metrics_filename}")
            except Exception as e:
                print(f"Error processing {exp_type}: {e}")
                import traceback
                traceback.print_exc()
        
            # Metrics comparison table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [['Distribution', 'TV Distance', 'Hellinger Distance', 'KL Divergence', 'Wasserstein Distance']]
        
        for exp_type, results in experiments.items():
            metrics = results[2]
            table_data.append([
                exp_type.title(),
                f"{metrics['tv']:.6f}",
                f"{metrics['hellinger']:.6f}",
                f"{metrics['kl']:.6f}",
                f"{metrics['wasserstein']:.6f}"
            ])
        
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        ax.set_title('Distance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
        
        table_filename = f"{run_dir}/metrics_table.png"
        plt.savefig(table_filename, dpi=300, bbox_inches='tight')
        print(f"Metrics table saved: {table_filename}")
        
        plt.close()
        
        return run_dir
    except Exception as e:
        print(f"Error in create_plots: {e}")
        import traceback
        traceback.print_exc()
        return None
    



def compare_noise_levels(run_dir):
    """Compare different noise levels."""
    print("\nGenerating noise comparison...")
    
    # Ensure run directory exists
    os.makedirs(run_dir, exist_ok=True)
    
    layers = 6
    shots = 3000
    circuit = build_qgb_tree(layers, np.pi/2)
    target = binomial_target(layers)
    
    noise_levels = ['low', 'medium', 'high']
    results = {}
    
    # Noiseless
    counts_noiseless = sample_counts(circuit, shots=shots)
    empirical_noiseless = process_tree_counts(counts_noiseless)
    min_len = min(len(empirical_noiseless), len(target))
    empirical_noiseless = empirical_noiseless[:min_len]
    target_short = target[:min_len]
    metrics_noiseless = calculate_all_metrics(empirical_noiseless, target_short)
    results['noiseless'] = {'empirical': empirical_noiseless, 'metrics': metrics_noiseless}
    
    # Noisy simulations
    for noise_level in noise_levels:
        counts_noisy = simulate_noisy(circuit, noise_level=noise_level, shots=shots)
        empirical_noisy = process_tree_counts(counts_noisy)
        min_len = min(len(empirical_noisy), len(target))
        empirical_noisy = empirical_noisy[:min_len]
        metrics_noisy = calculate_all_metrics(empirical_noisy, target_short)
        results[noise_level] = {'empirical': empirical_noisy, 'metrics': metrics_noisy}
    
    # Create noise comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    x = np.arange(len(target_short))
    
    colors = {'noiseless': 'green', 'low': 'orange', 'medium': 'red', 'high': 'purple'}
    
    for i, (noise_type, result) in enumerate(results.items()):
        row, col = i // 2, i % 2
        axes[row, col].bar(x - 0.2, target_short, width=0.4, label='Theoretical', alpha=0.7, color='blue')
        axes[row, col].bar(x + 0.2, result['empirical'], width=0.4, label=noise_type.title(), alpha=0.7, color=colors[noise_type])
        axes[row, col].set_title(f'{noise_type.title()} Noise')
        axes[row, col].set_xlabel('Bin Index')
        axes[row, col].set_ylabel('Probability')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    noise_filename = f"{run_dir}/noise_comparison.png"
    plt.savefig(noise_filename, dpi=300, bbox_inches='tight')
    print(f"Noise comparison plot saved: {noise_filename}")
    
    # Print noise results
    print("\nNoise Comparison Results:")
    for noise_type, result in results.items():
        metrics = result['metrics']
        print(f"{noise_type.title()}: TV = {metrics['tv']:.6f}, Hellinger = {metrics['hellinger']:.6f}")
    
    plt.close()


def load_latest_results():
    """Load the latest comprehensive results."""
    # Find the most recent results directory with layer scaling data
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        print(f"Outputs directory {outputs_dir} not found")
        return None, None
    
    # Get all timestamped directories
    timestamp_dirs = [d for d in os.listdir(outputs_dir) 
                     if os.path.isdir(os.path.join(outputs_dir, d)) 
                     and d.replace('_', '').replace('-', '').isdigit()]
    
    if not timestamp_dirs:
        print("No timestamped results directories found")
        return None, None
    
    # Find directory with layer scaling results
    results_dir = None
    
    # First, try to find a directory with all three distributions and complete Hadamard data
    # Check current directory first (for newly generated data)
    current_dir = os.path.basename(os.getcwd())
    if current_dir.startswith('outputs/'):
        current_timestamp = current_dir.split('/')[-1]
        if current_timestamp in timestamp_dirs:
            preferred_dirs = [current_timestamp, '20250811_060128', '20250811_060557', '20250811_070148', '20250811_055513']
        else:
            preferred_dirs = ['20250811_060128', '20250811_060557', '20250811_070148', '20250811_055513']
    else:
        preferred_dirs = ['20250811_060128', '20250811_060557', '20250811_070148', '20250811_055513']
    for preferred_dir in preferred_dirs:
        if preferred_dir in timestamp_dirs:
            dir_path = os.path.join(outputs_dir, preferred_dir)
            # Check if this directory has all three distributions
            has_all = True
            for exp_type in ['gaussian', 'exponential', 'hadamard']:
                layer_scaling_file = os.path.join(dir_path, f"{exp_type}_layer_scaling_results.json")
                results_file = os.path.join(dir_path, f"{exp_type}_results.json")
                if not os.path.exists(layer_scaling_file) and not os.path.exists(results_file):
                    has_all = False
                    break
            
            if has_all:
                # For Hadamard, check if it has complete data for layers 2, 4, 6, 8
                hadamard_file = os.path.join(dir_path, "hadamard_layer_scaling_results.json")
                if os.path.exists(hadamard_file):
                    try:
                        with open(hadamard_file, 'r') as f:
                            hadamard_data = json.load(f)
                        layer_results = hadamard_data.get('layer_results', {})
                        # Check if medium and high noise have data for layers 2, 4, 6, 8
                        has_complete_hadamard = True
                        for noise_level in ['medium', 'high']:
                            if noise_level not in layer_results:
                                has_complete_hadamard = False
                                break
                            for layer in [2, 4, 6, 8]:
                                if str(layer) not in layer_results[noise_level]:
                                    has_complete_hadamard = False
                                    break
                            if not has_complete_hadamard:
                                break
                        
                        if has_complete_hadamard:
                            results_dir = dir_path
                            break
                    except:
                        pass
            
            if results_dir is None and has_all:
                results_dir = dir_path
                break
    
    # If preferred directories don't have all data, search from newest to oldest
    if results_dir is None:
        for dir_name in sorted(timestamp_dirs, reverse=True):  # Check from newest to oldest
            dir_path = os.path.join(outputs_dir, dir_name)
            # Check if this directory has layer scaling results
            has_layer_scaling = False
            for exp_type in ['gaussian', 'exponential', 'hadamard']:
                filename = os.path.join(dir_path, f"{exp_type}_layer_scaling_results.json")
                if os.path.exists(filename):
                    has_layer_scaling = True
                    break
            
            if has_layer_scaling:
                results_dir = dir_path
                break
    
    # If no directory with layer scaling results found, try to find any directory with results
    if results_dir is None:
        for dir_name in sorted(timestamp_dirs, reverse=True):
            dir_path = os.path.join(outputs_dir, dir_name)
            # Check if this directory has any results
            has_results = False
            for exp_type in ['gaussian', 'exponential', 'hadamard']:
                filename = os.path.join(dir_path, f"{exp_type}_results.json")
                if os.path.exists(filename):
                    has_results = True
                    break
            
            if has_results:
                results_dir = dir_path
                break
    
    if results_dir is None:
        print("No directory with layer scaling results found")
        return None, None
    
    print(f"Loading results from: {results_dir}")
    
    results = {}
    
    # Load layer scaling results
    for exp_type in ['gaussian', 'exponential', 'hadamard']:
        # Try layer scaling results first
        filename = os.path.join(results_dir, f"{exp_type}_layer_scaling_results.json")
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # For Hadamard, just log the data format for debugging
                if exp_type == 'hadamard':
                    print(f"  Loading Hadamard data...")
                    for noise_level in ['low', 'medium', 'high']:
                        if noise_level in data.get('layer_results', {}):
                            for layer_str in data['layer_results'][noise_level]:
                                layer = int(layer_str)
                                target = data['layer_results'][noise_level][layer_str]['target']
                                actual_bins = len(target)
                                print(f"    Layer {layer}: {actual_bins} bins")
                
                results[exp_type] = data
                print(f"  Loaded {exp_type} layer scaling data")
            except Exception as e:
                print(f"  Error loading {exp_type} layer scaling data: {e}")
        else:
            # Try regular results file
            filename = os.path.join(results_dir, f"{exp_type}_results.json")
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    results[exp_type] = data
                    print(f"  Loaded {exp_type} results data")
                except Exception as e:
                    print(f"  Error loading {exp_type} results data: {e}")
            else:
                print(f"  {exp_type} data not found")
    
    return results, results_dir

def create_gaussian_layer_comparison(results, output_dir):
    """Create Gaussian layer comparison plot with layers 2, 4, 6, 8."""
    
    if 'gaussian' not in results:
        print("Gaussian data not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gaussian Distribution - Layer Comparison (2, 4, 6, 8)', fontsize=16, fontweight='bold')
    
    layer_data = results['gaussian']['layer_results']
    comparison_layers = [2, 4, 6, 8]
    noise_levels = ['noiseless', 'low', 'high']
    colors = {'target': 'black', 'noiseless': 'blue', 'low': 'orange', 'high': 'red'}
    
    for i, layer in enumerate(comparison_layers):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Get target distribution for this layer
        target_dist = None
        for level in noise_levels:
            if level in layer_data and str(layer) in layer_data[level]:
                target_dist = np.array(layer_data[level][str(layer)]['target'])
                break
        
        if target_dist is None:
            ax.text(0.5, 0.5, f'No data for layer {layer}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer}', fontweight='bold')
            continue
        
        # Use position from 0 to layer
        x = np.arange(layer + 1)
        width = 0.2
        
        # Plot target distribution as ideal line
        ax.plot(x, target_dist, 'o-', label='Target', linewidth=2, markersize=6, color=colors['target'])
        
        # Plot each noise level
        for j, level in enumerate(noise_levels):
            if level in layer_data and str(layer) in layer_data[level]:
                empirical = np.array(layer_data[level][str(layer)]['empirical'])
                ax.bar(x + (j-0.5)*width, empirical, width, label=f'{level.title()} Noise', 
                      alpha=0.7, color=colors[level])
        
        ax.set_title(f'Layer {layer}', fontweight='bold')
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Set x-axis ticks to integers only
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'gaussian_layer_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created Gaussian layer comparison plot: {filename}")

def create_exponential_layer_comparison(results, output_dir):
    """Create Exponential layer comparison plot with layers 2, 4, 6, 8."""
    
    if 'exponential' not in results:
        print("Exponential data not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Exponential Distribution - Layer Comparison (2, 4, 6, 8)', fontsize=16, fontweight='bold')
    
    layer_data = results['exponential']['layer_results']
    comparison_layers = [2, 4, 6, 8]
    noise_levels = ['noiseless', 'low', 'high']
    colors = {'target': 'black', 'noiseless': 'blue', 'low': 'orange', 'high': 'red'}
    
    for i, layer in enumerate(comparison_layers):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Get target distribution for this layer
        target_dist = None
        for level in noise_levels:
            if level in layer_data and str(layer) in layer_data[level]:
                target_dist = np.array(layer_data[level][str(layer)]['target'])
                break
        
        if target_dist is None:
            ax.text(0.5, 0.5, f'No data for layer {layer}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer}', fontweight='bold')
            continue
        
        # Use position from 0 to layer
        x = np.arange(layer + 1)
        width = 0.2
        
        # Plot target distribution as ideal line
        ax.plot(x, target_dist, 'o-', label='Target', linewidth=2, markersize=6, color=colors['target'])
        
        # Plot each noise level
        for j, level in enumerate(noise_levels):
            if level in layer_data and str(layer) in layer_data[level]:
                empirical = np.array(layer_data[level][str(layer)]['empirical'])
                ax.bar(x + (j-0.5)*width, empirical, width, label=f'{level.title()} Noise', 
                      alpha=0.7, color=colors[level])
        
        ax.set_title(f'Layer {layer}', fontweight='bold')
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Set x-axis ticks to integers only
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'exponential_layer_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created Exponential layer comparison plot: {filename}")




def adjust_hadamard_data_for_layer(data, layer):
    """Adjust Hadamard data to extract central layer+1 bins from 2*layer+1 bins."""
    if len(data) == 2 * layer + 1:
        # Extract central layer+1 bins
        start_idx = layer
        end_idx = 2 * layer + 1
        return data[start_idx:end_idx]
    else:
        # Data is already in correct format
        return data

def create_hadamard_layer_comparison(results, output_dir):
    """Create Hadamard layer comparison plot with layers 2, 4, 6, 8."""
    
    if 'hadamard' not in results:
        print("Hadamard data not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hadamard Distribution - Layer Comparison (2, 4, 6, 8)', fontsize=16, fontweight='bold')
    
    layer_data = results['hadamard']['layer_results']
    comparison_layers = [2, 4, 6, 8]
    noise_levels = ['noiseless', 'low', 'high']
    colors = {'target': 'black', 'noiseless': 'blue', 'low': 'orange', 'high': 'red'}
    
    for i, layer in enumerate(comparison_layers):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Check if we have data for this layer
        has_data = False
        target_dist = None
        
        for level in noise_levels:
            if level in layer_data and str(layer) in layer_data[level]:
                if target_dist is None:
                    target_dist = np.array(layer_data[level][str(layer)]['target'])
                has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, f'No data for layer {layer}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer}', fontweight='bold')
            continue
        
        # Adjust Hadamard data if needed (extract central layer+1 bins from 2*layer+1 bins)
        original_target = target_dist.copy()
        target_dist = adjust_hadamard_data_for_layer(target_dist, layer)
        
        # Hadamard data now has layer+1 bins by design
        x = np.arange(layer + 1)
        width = 0.2
        
        # Plot target distribution as ideal line
        ax.plot(x, target_dist, 'o-', label='Target', linewidth=2, markersize=6, color=colors['target'])
        
        # Plot each noise level
        for j, level in enumerate(noise_levels):
            if level in layer_data and str(layer) in layer_data[level]:
                empirical = np.array(layer_data[level][str(layer)]['empirical'])
                # Adjust empirical data to match target
                empirical = adjust_hadamard_data_for_layer(empirical, layer)
                
                ax.bar(x + (j-0.5)*width, empirical, width, label=f'{level.title()} Noise', 
                      alpha=0.7, color=colors[level])
        
        ax.set_title(f'Layer {layer}', fontweight='bold')
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Set x-axis ticks to integers only
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'hadamard_layer_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created Hadamard layer comparison plot: {filename}")

def create_combined_layer_comparison(results, output_dir):
    """Create combined layer comparison plot for all distributions."""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Layer Comparison Across All Distributions (2, 4, 6, 8)', fontsize=18, fontweight='bold')
    
    distributions = ['gaussian', 'exponential', 'hadamard']
    comparison_layers = [2, 4, 6, 8]
    noise_levels = ['noiseless', 'low', 'high']
    colors = {'target': 'black', 'noiseless': 'blue', 'low': 'orange', 'high': 'red'}
    
    for i, dist_type in enumerate(distributions):
        if dist_type not in results:
            continue
            
        layer_data = results[dist_type]['layer_results']
        
        for j, layer in enumerate(comparison_layers):
            ax = axes[i, j]
            
            # Check if we have data for this layer
            has_data = False
            target_dist = None
            
            for level in noise_levels:
                if level in layer_data and str(layer) in layer_data[level]:
                    if target_dist is None:
                        target_dist = np.array(layer_data[level][str(layer)]['target'])
                    has_data = True
            
            if not has_data:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dist_type.title()} - Layer {layer}', fontweight='bold')
                continue
            
            # For all distributions, use position from 0 to layer
            if dist_type == 'hadamard':
                original_target = target_dist.copy()
                target_dist = adjust_hadamard_data_for_layer(target_dist, layer)
            x = np.arange(layer + 1)
            xlabel = 'Position'
            
            width = 0.2
            
            # Plot target distribution as ideal line
            ax.plot(x, target_dist, 'o-', label='Target', linewidth=2, markersize=6, color=colors['target'])
            
            # Plot each noise level
            for k, level in enumerate(noise_levels):
                if level in layer_data and str(layer) in layer_data[level]:
                    empirical = np.array(layer_data[level][str(layer)]['empirical'])
                    
                    # For Hadamard, adjust empirical data to match target
                    if dist_type == 'hadamard':
                        empirical = adjust_hadamard_data_for_layer(empirical, layer)
                    else:
                        # Ensure empirical has the same length as target
                        if len(empirical) < len(target_dist):
                            empirical_padded = np.zeros(len(target_dist))
                            empirical_padded[:len(empirical)] = empirical
                            empirical = empirical_padded
                        elif len(empirical) > len(target_dist):
                            empirical = empirical[:len(target_dist)]
                    
                    ax.bar(x + (k-0.5)*width, empirical, width, label=f'{level.title()} Noise', 
                          alpha=0.7, color=colors[level])
            
            ax.set_title(f'{dist_type.title()} - Layer {layer}', fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Probability')
            if i == 0 and j == 0:  # Only show legend for first subplot
                ax.legend()
            ax.grid(True, alpha=0.3)
            # Set x-axis ticks to integers only
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'combined_layer_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created combined layer comparison plot: {filename}")

def create_layer_metrics_comparison(results, output_dir):
    """Create layer metrics comparison plot showing TV distance for each layer."""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Layer Metrics Comparison (TV Distance) - All Distributions', fontsize=18, fontweight='bold')
    
    distributions = ['gaussian', 'exponential', 'hadamard']
    comparison_layers = [2, 4, 6, 8]
    noise_levels = ['low', 'medium', 'high']
    colors = {'low': 'blue', 'medium': 'orange', 'high': 'red'}
    
    for i, dist_type in enumerate(distributions):
        if dist_type not in results:
            continue
            
        layer_data = results[dist_type]['layer_results']
        
        for j, layer in enumerate(comparison_layers):
            ax = axes[i, j]
            
            tv_values = []
            available_levels = []
            
            for level in noise_levels:
                if (level in layer_data and 
                    str(layer) in layer_data[level] and
                    'tv' in layer_data[level][str(layer)]):
                    tv = layer_data[level][str(layer)]['tv']
                    tv_values.append(tv)
                    available_levels.append(level)
            
            if tv_values:
                bars = ax.bar(available_levels, tv_values, color=[colors[level] for level in available_levels], alpha=0.7)
                ax.set_title(f'{dist_type.title()} - Layer {layer}', fontweight='bold')
                ax.set_ylabel('TV Distance')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, tv_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dist_type.title()} - Layer {layer}', fontweight='bold')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'layer_metrics_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created layer metrics comparison plot: {filename}")

def create_distance_layer_scaling_plot(results, output_dir):
    """Create distance layer scaling plots for each distribution with error bars."""
    
    distributions = ['gaussian', 'exponential', 'hadamard']
    metrics = ['tv', 'hellinger', 'kl', 'wasserstein']
    noise_levels = ['noiseless', 'low', 'high']
    colors = {'noiseless': 'green', 'low': 'blue', 'high': 'red'}
    
    for dist_type in distributions:
        if dist_type not in results:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{dist_type.title()} Distribution - Distance Layer Scaling (with 95% CI)', fontsize=16, fontweight='bold')
        
        layer_data = results[dist_type]['layer_results']
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Collect data for layers 1-8
            layers = []
            for level in noise_levels:
                if level in layer_data:
                    layers.extend([int(l) for l in layer_data[level].keys() if 1 <= int(l) <= 8])
            layers = sorted(list(set(layers)))
            
            for level in noise_levels:
                if level not in layer_data:
                    continue
                    
                metric_values = []
                metric_errors = []
                available_layers = []
                
                for layer in layers:
                    if (str(layer) in layer_data[level] and 
                        metric in layer_data[level][str(layer)]):
                        value = layer_data[level][str(layer)][metric]
                        metric_values.append(value)
                        available_layers.append(layer)
                        
                        # Get bootstrap confidence interval if available
                        if 'bootstrap' in layer_data[level][str(layer)]:
                            bootstrap_data = layer_data[level][str(layer)]['bootstrap']
                            if metric in bootstrap_data:
                                lower = bootstrap_data[metric]['lower']
                                upper = bootstrap_data[metric]['upper']
                                # Calculate error bar as half the confidence interval width
                                error = (upper - lower) / 2
                                metric_errors.append(error)
                            else:
                                metric_errors.append(0)
                        else:
                            metric_errors.append(0)
                
                if metric_values:
                    if any(metric_errors):  # If we have error bars
                        ax.errorbar(available_layers, metric_values, yerr=metric_errors, 
                                  fmt='o-', label=f'{level.title()} Noise', color=colors[level], 
                                  linewidth=2, markersize=6, capsize=5, capthick=2)
                    else:  # Fallback to regular plot
                        ax.plot(available_layers, metric_values, 'o-', 
                               label=f'{level.title()} Noise', color=colors[level], 
                               linewidth=2, markersize=6)
            
            ax.set_title(f'{metric.upper()} Distance', fontweight='bold')
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel(f'{metric.upper()} Distance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'{dist_type}_distance_layer_scaling.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created {dist_type} distance layer scaling plot with error bars: {filename}")

def create_combined_distance_layer_scaling_plot(results, output_dir):
    """Create combined distance layer scaling plot with all distributions in one graph with error bars."""
    
    distributions = ['gaussian', 'exponential', 'hadamard']
    metrics = ['tv', 'hellinger', 'kl', 'wasserstein']
    noise_levels = ['noiseless', 'low', 'high']
    colors = {'noiseless': 'green', 'low': 'blue', 'high': 'red'}
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Combined Distance Layer Scaling - All Distributions (with 95% CI)', fontsize=18, fontweight='bold')
    
    for i, dist_type in enumerate(distributions):
        if dist_type not in results:
            continue
            
        layer_data = results[dist_type]['layer_results']
        
        # Collect data for all layers
        layers = []
        for level in noise_levels:
            if level in layer_data:
                layers.extend([int(l) for l in layer_data[level].keys()])
        layers = sorted(list(set(layers)))
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            for level in noise_levels:
                if level not in layer_data:
                    continue
                    
                metric_values = []
                metric_errors = []
                available_layers = []
                
                for layer in layers:
                    if (str(layer) in layer_data[level] and 
                        metric in layer_data[level][str(layer)]):
                        value = layer_data[level][str(layer)][metric]
                        metric_values.append(value)
                        available_layers.append(layer)
                        
                        # Get bootstrap confidence interval if available
                        if 'bootstrap' in layer_data[level][str(layer)]:
                            bootstrap_data = layer_data[level][str(layer)]['bootstrap']
                            if metric in bootstrap_data:
                                lower = bootstrap_data[metric]['lower']
                                upper = bootstrap_data[metric]['upper']
                                # Calculate error bar as half the confidence interval width
                                error = (upper - lower) / 2
                                metric_errors.append(error)
                            else:
                                metric_errors.append(0)
                        else:
                            metric_errors.append(0)
                
                if metric_values:
                    if any(metric_errors):  # If we have error bars
                        ax.errorbar(available_layers, metric_values, yerr=metric_errors, 
                                  fmt='o-', label=f'{level.title()} Noise', color=colors[level], 
                                  linewidth=2, markersize=6, capsize=5, capthick=2)
                    else:  # Fallback to regular plot
                        ax.plot(available_layers, metric_values, 'o-', 
                               label=f'{level.title()} Noise', color=colors[level], 
                               linewidth=2, markersize=6)
            
            ax.set_title(f'{dist_type.title()} - {metric.upper()} Distance', fontweight='bold')
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel(f'{metric.upper()} Distance')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'combined_distance_layer_scaling.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created combined distance layer scaling plot with error bars: {filename}")

def create_metric_comparison_plots(results, output_dir):
    """Create metric comparison plots for each metric."""
    
    metrics = ['tv', 'hellinger', 'kl', 'wasserstein']
    distributions = ['gaussian', 'exponential', 'hadamard']
    noise_levels = ['low', 'medium', 'high']
    colors = {'gaussian': 'blue', 'exponential': 'orange', 'hadamard': 'green'}
    markers = {'low': 'o', 'medium': 's', 'high': '^'}
    
    for metric in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{metric.upper()} Distance - Layer Scaling Comparison', fontsize=16, fontweight='bold')
        
        # Individual distribution plots
        for i, dist_type in enumerate(distributions):
            if dist_type not in results:
                continue
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            layer_data = results[dist_type]['layer_results']
            
            # Collect data for layers 1-8
            layers = []
            for level in noise_levels:
                if level in layer_data:
                    layers.extend([int(l) for l in layer_data[level].keys() if 1 <= int(l) <= 8])
            layers = sorted(list(set(layers)))
            
            for level in noise_levels:
                if level not in layer_data:
                    continue
                    
                metric_values = []
                available_layers = []
                
                for layer in layers:
                    if (str(layer) in layer_data[level] and 
                        metric in layer_data[level][str(layer)]):
                        value = layer_data[level][str(layer)][metric]
                        metric_values.append(value)
                        available_layers.append(layer)
                
                if metric_values:
                    ax.plot(available_layers, metric_values, markers[level], 
                           label=f'{level.title()} Noise', color=colors[dist_type], 
                           linewidth=2, markersize=6)
            
            ax.set_title(f'{dist_type.title()} Distribution', fontweight='bold')
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel(f'{metric.upper()} Distance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # Combined comparison plot (Layer 16)
        ax = axes[1, 1]
        layer_16_data = {}
        
        for dist_type in distributions:
            if dist_type not in results:
                continue
                
            layer_data = results[dist_type]['layer_results']
            layer_16_data[dist_type] = {}
            
            for level in noise_levels:
                if (level in layer_data and 
                    '16' in layer_data[level] and 
                    metric in layer_data[level]['16']):
                    layer_16_data[dist_type][level] = layer_data[level]['16'][metric]
        
        # Create bar plot for Layer 16 comparison
        if layer_16_data:
            x = np.arange(len(distributions))
            width = 0.25
            
            for i, level in enumerate(noise_levels):
                values = []
                for dist_type in distributions:
                    if dist_type in layer_16_data and level in layer_16_data[dist_type]:
                        values.append(layer_16_data[dist_type][level])
                    else:
                        values.append(0)
                
                ax.bar(x + i*width, values, width, label=f'{level.title()} Noise', alpha=0.8)
            
            ax.set_title('Layer 16 Comparison', fontweight='bold')
            ax.set_xlabel('Distribution Type')
            ax.set_ylabel(f'{metric.upper()} Distance')
            ax.set_xticks(x + width)
            ax.set_xticklabels([d.title() for d in distributions])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'{metric}_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created {metric} comparison plot: {filename}")

def create_metric_scaling_plots(results, output_dir):
    """Create metric scaling plots for each distribution."""
    
    distributions = ['gaussian', 'exponential', 'hadamard']
    metrics = ['tv', 'hellinger', 'kl', 'wasserstein']
    noise_levels = ['low', 'medium', 'high']
    colors = {'low': 'blue', 'medium': 'orange', 'high': 'red'}
    
    for dist_type in distributions:
        if dist_type not in results:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{dist_type.title()} Distribution - Metric Scaling', fontsize=16, fontweight='bold')
        
        layer_data = results[dist_type]['layer_results']
        
        # Collect data for layers 1-8
        layers = []
        for level in noise_levels:
            if level in layer_data:
                layers.extend([int(l) for l in layer_data[level].keys() if 1 <= int(l) <= 8])
        layers = sorted(list(set(layers)))
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            for level in noise_levels:
                if level not in layer_data:
                    continue
                    
                metric_values = []
                available_layers = []
                
                for layer in layers:
                    if (str(layer) in layer_data[level] and 
                        metric in layer_data[level][str(layer)]):
                        value = layer_data[level][str(layer)][metric]
                        metric_values.append(value)
                        available_layers.append(layer)
                
                if metric_values:
                    ax.plot(available_layers, metric_values, 'o-', 
                           label=f'{level.title()} Noise', color=colors[level], 
                           linewidth=2, markersize=6)
            
            ax.set_title(f'{metric.upper()} Distance', fontweight='bold')
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel(f'{metric.upper()} Distance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'{dist_type}_metric_scaling.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created {dist_type} metric scaling plot: {filename}")

def create_unified_visualizations(output_dir):
    """Create unified visualizations."""
    print("📊 Creating Unified Visualizations")
    print("=" * 60)
    
    # Check if we have newly generated data in the current output directory
    results = {}
    source_dir = output_dir
    
    # Try to load data from the current output directory first
    for exp_type in ['gaussian', 'exponential', 'hadamard']:
        # Try layer scaling results first
        filename = os.path.join(output_dir, f"{exp_type}_layer_scaling_results.json")
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                results[exp_type] = data
                print(f"  Loaded {exp_type} layer scaling data from current directory")
            except Exception as e:
                print(f"  Error loading {exp_type} layer scaling data: {e}")
        else:
            # Try regular results file
            filename = os.path.join(output_dir, f"{exp_type}_results.json")
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    results[exp_type] = data
                    print(f"  Loaded {exp_type} results data from current directory")
                except Exception as e:
                    print(f"  Error loading {exp_type} results data: {e}")
    
    # If no data in current directory, load from latest results
    if not results:
        results, source_dir = load_latest_results()
    
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded results for: {list(results.keys())}")
    
    # Layer Comparison Visualizations
    print("\n🔍 Creating Layer Comparison Visualizations...")
    
    # Create individual distribution plots for available data
    if 'gaussian' in results:
        try:
            create_gaussian_layer_comparison(results, output_dir)
        except Exception as e:
            print(f"Error in gaussian_layer_comparison: {e}")
    else:
        print("  Skipping Gaussian layer comparison - no data available")
    
    if 'exponential' in results:
        try:
            create_exponential_layer_comparison(results, output_dir)
        except Exception as e:
            print(f"Error in exponential_layer_comparison: {e}")
    else:
        print("  Skipping Exponential layer comparison - no data available")
    
    if 'hadamard' in results:
        try:
            create_hadamard_layer_comparison(results, output_dir)
        except Exception as e:
            print(f"Error in hadamard_layer_comparison: {e}")
    else:
        print("  Skipping Hadamard layer comparison - no data available")
    
    # Create combined plot if we have at least 2 distributions
    available_distributions = [d for d in ['gaussian', 'exponential', 'hadamard'] if d in results]
    if len(available_distributions) >= 2:
        try:
            create_combined_layer_comparison(results, output_dir)
        except Exception as e:
            print(f"Error in combined_layer_comparison: {e}")
    else:
        print("  Skipping combined layer comparison - need at least 2 distributions")
    
    # Metric Comparison Visualizations
    print("\n📈 Creating Metric Comparison Visualizations...")
    
    # Create distance layer scaling plots for available distributions
    if len(available_distributions) >= 1:
        try:
            create_distance_layer_scaling_plot(results, output_dir)
        except Exception as e:
            print(f"Error in distance_layer_scaling_plot: {e}")
        
        if len(available_distributions) >= 2:
            try:
                create_combined_distance_layer_scaling_plot(results, output_dir)
            except Exception as e:
                print(f"Error in combined_distance_layer_scaling_plot: {e}")
        else:
            print("  Skipping combined distance layer scaling - need at least 2 distributions")
    else:
        print("  Skipping distance layer scaling - no data available")

def main():
    """Main function to generate all visualizations."""
    print("🚀 Quantum Galton Board Visualization")
    print("=" * 50)
    
    # Run experiments
    experiments = {}
    
    # Gaussian experiment
    experiments['gaussian'] = run_experiment("gaussian", layers=8, shots=5000)
    
    # Exponential experiment
    experiments['exponential'] = run_experiment("exponential", layers=8, lambda_param=0.5, shots=5000)
    
    # Hadamard experiment
    experiments['hadamard'] = run_experiment("hadamard", steps=8, shots=5000)
    
    # Create plots
    run_dir = create_plots(experiments)
    
    # Compare noise levels (use same run directory)
    compare_noise_levels(run_dir)
    
    # Create unified visualizations (Layer Comparison + Metric Comparison)
    print("\n🔍 Creating Layer Comparison and Metric Comparison Visualizations...")
    create_unified_visualizations(run_dir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("✅ All visualizations generated successfully!")
    print(f"📊 Check the directory: {run_dir}")
    print("=" * 50)
    
    # Print results summary
    print("\nResults Summary:")
    for exp_type, results in experiments.items():
        metrics = results[2]
        print(f"{exp_type.title()}: TV = {metrics['tv']:.6f}, Hellinger = {metrics['hellinger']:.6f}")


if __name__ == "__main__":
    main()
