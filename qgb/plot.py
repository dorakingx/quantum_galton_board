"""
Plotting utilities for Quantum Galton Board.

Reference: arXiv:2202.01735 - Universal Statistical Simulator
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from datetime import datetime
from .targets import binomial_target, exponential_target, hadamard_walk_target


def plot_distribution_comparison(
    empirical: np.ndarray,
    target: np.ndarray,
    title: str = "Distribution Comparison",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot empirical vs target distribution comparison.
    
    Args:
        empirical: Empirical probability distribution
        target: Target probability distribution
        title: Plot title
        save_path: Path to save plot
        show: Whether to show plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ensure same length
    min_len = min(len(empirical), len(target))
    empirical = empirical[:min_len]
    target = target[:min_len]
    x = np.arange(min_len)
    
    # Bar plot comparison
    width = 0.35
    ax1.bar(x - width/2, target, width, label='Target', alpha=0.7, color='blue')
    ax1.bar(x + width/2, empirical, width, label='Empirical', alpha=0.7, color='red')
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'{title} - Bar Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Line plot comparison
    ax2.plot(x, target, 'bo-', label='Target', linewidth=2, markersize=6)
    ax2.plot(x, empirical, 'ro-', label='Empirical', linewidth=2, markersize=6)
    ax2.set_xlabel('Bin Index')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'{title} - Line Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_with_ci(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Distance Metrics with Stochastic Uncertainty",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot distance metrics with confidence intervals accounting for stochastic uncertainty.
    
    This function visualizes the distance metrics between quantum and target distributions,
    including bootstrap confidence intervals that account for measurement uncertainty.
    
    Args:
        metrics: Bootstrap metrics results with confidence intervals
        title: Plot title
        save_path: Path to save plot
        show: Whether to show plot
    """
    metric_names = list(metrics.keys())
    means = [metrics[name]['mean'] for name in metric_names]
    lowers = [metrics[name]['lower'] for name in metric_names]
    uppers = [metrics[name]['upper'] for name in metric_names]
    stds = [metrics[name]['std'] for name in metric_names]
    
    x = np.arange(len(metric_names))
    errors = np.array([means - np.array(lowers), np.array(uppers) - means])
    
    # Ensure errors are non-negative
    errors = np.abs(errors)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confidence intervals plot
    ax1.errorbar(x, means, yerr=errors, fmt='o', capsize=5, capthick=2, markersize=8, 
                color='blue', label='95% Confidence Interval')
    ax1.set_xlabel('Distance Metric')
    ax1.set_ylabel('Distance Value')
    ax1.set_title(f'{title} - Confidence Intervals')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.upper() for name in metric_names])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Uncertainty measures plot
    ax2.bar(x, stds, color='red', alpha=0.7, label='Standard Deviation')
    ax2.set_xlabel('Distance Metric')
    ax2.set_ylabel('Uncertainty Measure')
    ax2.set_title(f'{title} - Uncertainty Measures')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name.upper() for name in metric_names])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Add value labels
    for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
        ax1.text(i, upper + 0.01, f'{mean:.4f}', ha='center', va='bottom')
    
    # Add value labels for uncertainty measures
    for i, std in enumerate(stds):
        ax2.text(i, std + 0.001, f'{std:.4f}', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_optimization_comparison(
    original_angles: list,
    optimized_angles: list,
    original_tv: float,
    optimized_tv: float,
    title: str = "Angle Optimization Comparison",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot angle optimization comparison.
    
    Args:
        original_angles: Original angle list
        optimized_angles: Optimized angle list
        original_tv: Original TV distance
        optimized_tv: Optimized TV distance
        title: Plot title
        save_path: Path to save plot
        show: Whether to show plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(original_angles))
    
    # Angle comparison
    ax1.plot(x, original_angles, 'bo-', label='Original', linewidth=2, markersize=6)
    ax1.plot(x, optimized_angles, 'ro-', label='Optimized', linewidth=2, markersize=6)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Angle (radians)')
    ax1.set_title('Angle Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # TV distance comparison
    tv_values = [original_tv, optimized_tv]
    tv_labels = ['Original', 'Optimized']
    colors = ['blue', 'red']
    
    bars = ax2.bar(tv_labels, tv_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Total Variation Distance')
    ax2.set_title('TV Distance Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, tv_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_timestamp_filename(prefix: str, extension: str = "png") -> str:
    """
    Create timestamped filename.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        str: Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/{prefix}_{timestamp}.{extension}"


def plot_convergence_analysis(
    layers_list: list,
    tv_distances: list,
    title: str = "Convergence Analysis",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot convergence analysis for different layer counts.
    
    Args:
        layers_list: List of layer counts
        tv_distances: List of TV distances
        title: Plot title
        save_path: Path to save plot
        show: Whether to show plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(layers_list, tv_distances, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Layers')
    plt.ylabel('Total Variation Distance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(layers_list, tv_distances):
        plt.text(x, y + 0.01, f'{y:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_distribution_comprehensive(
    empirical: np.ndarray,
    target: np.ndarray,
    bootstrap_results: Dict[str, Dict[str, float]],
    distribution_name: str,
    layer_scaling_data: Optional[Dict] = None,
    noise_comparison_data: Optional[Dict] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Create comprehensive plot with 4 subplots for different layers and layer scaling analysis.
    
    Args:
        empirical: Empirical probability distribution
        target: Target probability distribution
        bootstrap_results: Bootstrap metrics results
        distribution_name: Name of the distribution
        layer_scaling_data: Layer scaling data for different noise levels
        noise_comparison_data: Noise comparison data for different noise levels
        save_path: Path to save plot
        show: Whether to show plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'{distribution_name.title()} Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Define layers to plot
    layers_to_plot = [2, 4, 8, 16]
    
    # 1. Distribution comparison for different layers (4 subplots)
    if layer_scaling_data and 'layer_results' in layer_scaling_data:
        noise_levels = ['low', 'medium', 'high']
        colors = ['green', 'orange', 'red']
        
        for idx, layer in enumerate(layers_to_plot):
            ax = [ax1, ax2, ax3, ax4][idx]
            
            # Get target distribution for this layer
            if distribution_name == "gaussian":
                layer_target = binomial_target(layer)
            elif distribution_name == "exponential":
                layer_target = exponential_target(0.5, layer + 1)  # Using lambda=0.5
            elif distribution_name == "hadamard":
                layer_target, _ = hadamard_walk_target(layer)
            else:
                layer_target = target
            
            # Get empirical data for different noise levels
            width = 0.12
            x_pos = np.arange(len(layer_target))
            
            # Target distribution
            ax.bar(x_pos - 2*width, layer_target, width, label='Target', alpha=0.8, color='blue')
            
            # Noiseless data (from noise comparison data)
            noiseless_data = None
            if noise_comparison_data and 'noise_results' in noise_comparison_data:
                # Get noiseless data from the main experiment
                if 'noiseless' in noise_comparison_data['noise_results']:
                    noiseless_data = noise_comparison_data['noise_results']['noiseless'].get('empirical', [])
                elif 'low' in noise_comparison_data['noise_results']:
                    # Use low noise as noiseless reference
                    noiseless_data = noise_comparison_data['noise_results']['low'].get('empirical', [])
                
                if noiseless_data:
                    # Convert to numpy array if it's a list
                    if isinstance(noiseless_data, list):
                        noiseless_data = np.array(noiseless_data)
                    # Ensure we have enough data and pad/truncate to match layer_target length
                    if len(noiseless_data) >= len(layer_target):
                        ax.bar(x_pos - width, noiseless_data[:len(layer_target)], width, 
                               label='Noiseless', alpha=0.8, color='gray')
                    elif len(noiseless_data) < len(layer_target):
                        # Pad with zeros if shorter
                        padded_data = np.zeros(len(layer_target))
                        padded_data[:len(noiseless_data)] = noiseless_data
                        ax.bar(x_pos - width, padded_data, width, 
                               label='Noiseless', alpha=0.8, color='gray')
            
            # Noisy distributions
            for i, noise_level in enumerate(noise_levels):
                if noise_level in layer_scaling_data['layer_results'] and layer in layer_scaling_data['layer_results'][noise_level]:
                    noisy_data = layer_scaling_data['layer_results'][noise_level][layer].get('empirical', [])
                    if noisy_data:
                        # Convert to numpy array if it's a list
                        if isinstance(noisy_data, list):
                            noisy_data = np.array(noisy_data)
                        
                        # Ensure data length matches layer_target
                        if len(noisy_data) >= len(layer_target):
                            plot_data = noisy_data[:len(layer_target)]
                        else:
                            # Pad with zeros if shorter
                            plot_data = np.zeros(len(layer_target))
                            plot_data[:len(noisy_data)] = noisy_data
                        
                        ax.bar(x_pos + i*width, plot_data, width, 
                               label=f'{noise_level} noise', alpha=0.8, color=colors[i])
            
            ax.set_xlabel('Bin Index')
            ax.set_ylabel('Probability')
            ax.set_title(f'Layer {layer} Distribution Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(range(len(layer_target)))
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        # Fallback: simple comparison without layer data
        for idx, layer in enumerate(layers_to_plot):
            ax = [ax1, ax2, ax3, ax4][idx]
            ax.text(0.5, 0.5, f'Layer {layer} data\nnot available', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Layer {layer} Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # 2. Layer scaling analysis with multiple metrics (separate plot)
    if layer_scaling_data and 'layer_results' in layer_scaling_data:
        noise_levels = ['low', 'medium', 'high']
        colors = ['green', 'orange', 'red']
        metrics = ['tv', 'hellinger', 'kl', 'wasserstein']
        metric_names = ['TV Distance', 'Hellinger', 'KL Divergence', 'Wasserstein']
        
        # Create subplots for each metric
        fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig2.suptitle(f'{distribution_name.title()} - Layer Scaling Analysis by Metrics (Layers 1-16)', fontsize=14, fontweight='bold')
        
        for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[metric_idx // 2, metric_idx % 2]
            
            for i, noise_level in enumerate(noise_levels):
                if noise_level in layer_scaling_data['layer_results']:
                    layer_results = layer_scaling_data['layer_results'][noise_level]
                    
                    # Get all layers from 1 to 16
                    all_layers = sorted([l for l in layer_results.keys() if 1 <= l <= 16])
                    metric_values = [layer_results[l].get(metric, 0) for l in all_layers]
                    
                    ax.plot(all_layers, metric_values, 'o-', color=colors[i], 
                           linewidth=2, markersize=6, label=f'{noise_level} noise')
            
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Layers (1-16)')
            ax.set_xticks(range(1, 17))
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add threshold line for TV distance
            if metric == 'tv':
                ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
        
        plt.tight_layout()
        
        # Save layer scaling plot separately
        layer_scaling_path = save_path.replace('_comprehensive_analysis.png', '_layer_scaling_analysis.png')
        plt.savefig(layer_scaling_path, dpi=300, bbox_inches='tight')
        plt.close()
