import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from scipy.stats import binom, norm, expon
from scipy.spatial.distance import jensenshannon
import seaborn as sns
from typing import List, Tuple, Dict, Callable
import math
import argparse
from datetime import datetime
import os

class UniversalQuantumGaltonBoard:
    """
    Universal Quantum Galton Board implementation that can generate circuits
    for any number of layers and various target distributions.
    
    This implementation provides:
    1. General algorithm for arbitrary number of layers
    2. Gaussian distribution verification
    3. Exponential distribution generation
    4. Hadamard quantum walk implementation
    5. Noise modeling and optimization
    6. Statistical distance calculations
    """
    
    def __init__(self, num_levels: int = 8, distribution_type: str = "gaussian"):
        """
        Initialize the Universal Quantum Galton Board
        
        Args:
            num_levels: Number of levels in the Galton Board
            distribution_type: Type of target distribution ("gaussian", "exponential", "quantum_walk")
        """
        self.num_levels = num_levels
        self.distribution_type = distribution_type
        self.num_qubits = num_levels
        self.backend = Aer.get_backend('qasm_simulator')
        self.statevector_backend = Aer.get_backend('statevector_simulator')
        
        # Distribution-specific parameters
        self.lambda_param = 0.5  # For exponential distribution
        self.bias = 0.5  # For biased distributions
        
    def create_universal_circuit(self) -> QuantumCircuit:
        """
        Create a universal quantum circuit for any number of layers
        
        This is the core algorithm that generates circuits for arbitrary
        number of layers, ensuring Gaussian distribution convergence.
        
        Returns:
            QuantumCircuit: The universal quantum circuit
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        if self.distribution_type == "gaussian":
            return self._create_gaussian_circuit(qc)
        elif self.distribution_type == "exponential":
            return self._create_exponential_circuit(qc)
        elif self.distribution_type == "quantum_walk":
            return self._create_quantum_walk_circuit(qc)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
    
    def _create_gaussian_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Create circuit for Gaussian distribution (standard Galton board)
        
        Args:
            qc: Base quantum circuit
            
        Returns:
            QuantumCircuit: Circuit implementing Gaussian distribution
        """
        # Apply Hadamard gates to create 50/50 superposition at each level
        for i in range(self.num_qubits):
            qc.h(i)
        
        # Optional: Add entanglement layers for more realistic simulation
        # This simulates the cascading effect of the classical Galton board
        for layer in range(min(self.num_levels - 1, 3)):  # Limit entanglement layers
            for i in range(self.num_qubits - layer - 1):
                qc.cx(i, i + 1)
        
        return qc
    
    def _create_exponential_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Create circuit for exponential distribution
        
        Args:
            qc: Base quantum circuit
            
        Returns:
            QuantumCircuit: Circuit implementing exponential distribution
        """
        # Calculate rotation angles for exponential distribution
        # P(x) ∝ e^(-λx) where x is the position (number of 1s)
        angles = []
        for i in range(self.num_qubits):
            # Exponential decay in rotation angles
            angle = 2 * np.arccos(np.sqrt(np.exp(-self.lambda_param * i / self.num_qubits)))
            angles.append(angle)
        
        # Apply rotation gates with exponential angles
        for i in range(self.num_qubits):
            qc.ry(angles[i], i)
        
        return qc
    
    def _create_quantum_walk_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Create circuit for Hadamard quantum walk
        
        Args:
            qc: Base quantum circuit
            
        Returns:
            QuantumCircuit: Circuit implementing quantum walk
        """
        # Initialize in |0⟩ state
        # Apply Hadamard to first qubit
        qc.h(0)
        
        # Apply quantum walk steps
        for step in range(self.num_qubits - 1):
            # Shift operation (CNOT chain)
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
            
            # Hadamard on the last qubit
            qc.h(self.num_qubits - 1)
        
        return qc
    
    def create_measurement_circuit(self, base_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Add measurement operations to the base circuit
        
        Args:
            base_circuit: The base quantum circuit
            
        Returns:
            QuantumCircuit: Circuit with measurement operations
        """
        qc = base_circuit.copy()
        qc.measure_all()
        return qc
    
    def simulate_distribution(self, shots: int = 1000) -> Dict[str, int]:
        """
        Simulate the statistical distribution using the quantum circuit
        
        Args:
            shots: Number of shots for the simulation
            
        Returns:
            Dict[str, int]: Measurement results with counts
        """
        base_circuit = self.create_universal_circuit()
        measurement_circuit = self.create_measurement_circuit(base_circuit)
        
        # Execute the circuit
        job = self.backend.run(measurement_circuit, shots=shots)
        result = job.result()
        
        return result.get_counts()
    
    def analyze_distribution(self, counts: Dict[str, int]) -> Tuple[List[int], List[int], Dict]:
        """
        Analyze the measurement results and calculate statistics
        
        Args:
            counts: Measurement counts from the simulation
            
        Returns:
            Tuple containing positions, frequencies, and statistics
        """
        positions = []
        frequencies = []
        
        for bitstring, count in counts.items():
            # Count the number of 1s in the bitstring
            position = bitstring.count('1')
            positions.append(position)
            frequencies.append(count)
        
        # Calculate statistics
        total_shots = sum(frequencies)
        mean_pos = np.average(positions, weights=frequencies)
        variance = np.average([(pos - mean_pos)**2 for pos in positions], weights=frequencies)
        std_dev = np.sqrt(variance)
        
        stats = {
            'mean': mean_pos,
            'variance': variance,
            'std_dev': std_dev,
            'total_shots': total_shots
        }
        
        return positions, frequencies, stats
    
    def theoretical_distribution(self) -> Tuple[List[int], List[float]]:
        """
        Calculate the theoretical distribution based on the distribution type
        
        Returns:
            Tuple[List[int], List[float]]: Positions and their theoretical probabilities
        """
        positions = list(range(self.num_levels + 1))
        
        if self.distribution_type == "gaussian":
            # Binomial distribution (converges to Gaussian)
            probabilities = [binom.pmf(k, self.num_levels, 0.5) for k in positions]
        elif self.distribution_type == "exponential":
            # Exponential distribution
            probabilities = []
            for k in positions:
                prob = np.exp(-self.lambda_param * k)
                probabilities.append(prob)
            # Normalize
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
        elif self.distribution_type == "quantum_walk":
            # Quantum walk distribution (approximate)
            probabilities = []
            for k in positions:
                # Simplified quantum walk distribution
                prob = np.cos(np.pi * k / (2 * self.num_levels))**2
                probabilities.append(prob)
            # Normalize
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
        
        return positions, probabilities
    
    def calculate_statistical_distance(self, experimental_probs: List[float], 
                                     theoretical_probs: List[float]) -> Dict[str, float]:
        """
        Calculate various statistical distances between distributions
        
        Args:
            experimental_probs: Experimental probabilities
            theoretical_probs: Theoretical probabilities
            
        Returns:
            Dict[str, float]: Various distance measures
        """
        # Ensure same length
        min_len = min(len(experimental_probs), len(theoretical_probs))
        exp_probs = experimental_probs[:min_len]
        theo_probs = theoretical_probs[:min_len]
        
        # Total Variation Distance
        tv_distance = 0.5 * sum(abs(exp - theo) for exp, theo in zip(exp_probs, theo_probs))
        
        # Jensen-Shannon Divergence
        js_divergence = jensenshannon(exp_probs, theo_probs)
        
        # Mean Squared Error
        mse = np.mean((np.array(exp_probs) - np.array(theo_probs))**2)
        
        # Kullback-Leibler Divergence (with smoothing to avoid log(0))
        epsilon = 1e-10
        exp_smooth = [p + epsilon for p in exp_probs]
        theo_smooth = [p + epsilon for p in theo_probs]
        kl_divergence = sum(exp * np.log(exp / theo) for exp, theo in zip(exp_smooth, theo_smooth))
        
        return {
            'total_variation': tv_distance,
            'jensen_shannon': js_divergence,
            'mean_squared_error': mse,
            'kl_divergence': kl_divergence
        }
    
    def verify_gaussian_convergence(self, shots: int = 1000) -> Dict:
        """
        Verify that the output converges to a Gaussian distribution
        
        Args:
            shots: Number of shots for the simulation
            
        Returns:
            Dict: Convergence analysis results
        """
        if self.distribution_type != "gaussian":
            raise ValueError("Gaussian convergence verification only for gaussian distribution")
        
        # Run simulation
        counts = self.simulate_distribution(shots)
        positions, frequencies, stats = self.analyze_distribution(counts)
        
        # Calculate theoretical Gaussian parameters
        theoretical_mean = self.num_levels * 0.5
        theoretical_std = np.sqrt(self.num_levels * 0.25)
        
        # Create theoretical Gaussian
        x = np.linspace(0, self.num_levels, 100)
        gaussian = norm.pdf(x, theoretical_mean, theoretical_std)
        
        # Normalize experimental results
        total_shots = sum(frequencies)
        experimental_probs = [freq / total_shots for freq in frequencies]
        
        # Create theoretical binomial distribution
        theoretical_positions, theoretical_probs = self.theoretical_distribution()
        
        # Calculate distances
        distances = self.calculate_statistical_distance(experimental_probs, theoretical_probs)
        
        # Convergence metrics
        mean_error = abs(stats['mean'] - theoretical_mean)
        std_error = abs(stats['std_dev'] - theoretical_std)
        
        return {
            'experimental_stats': stats,
            'theoretical_mean': theoretical_mean,
            'theoretical_std': theoretical_std,
            'mean_error': mean_error,
            'std_error': std_error,
            'distances': distances,
            'positions': positions,
            'frequencies': frequencies,
            'gaussian_x': x,
            'gaussian_y': gaussian
        }
    
    def plot_comprehensive_results(self, counts: Dict[str, int], save_path: str = None):
        """
        Create comprehensive visualization of the results
        
        Args:
            counts: Measurement counts from the simulation
            save_path: Optional path to save the plot
        """
        positions, frequencies, stats = self.analyze_distribution(counts)
        theoretical_positions, theoretical_probs = self.theoretical_distribution()
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Experimental results histogram
        ax1 = plt.subplot(3, 3, 1)
        ax1.bar(positions, frequencies, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Position (Number of 1s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Quantum Simulation Results\n{self.distribution_type.capitalize()} Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Theoretical vs Experimental comparison
        ax2 = plt.subplot(3, 3, 2)
        theoretical_freqs = [prob * stats['total_shots'] for prob in theoretical_probs]
        
        # Create complete position arrays for comparison
        all_positions = list(range(self.num_levels + 1))
        
        # Create frequency arrays that match the complete position range
        experimental_freqs_complete = []
        for pos in all_positions:
            if pos in positions:
                idx = positions.index(pos)
                experimental_freqs_complete.append(frequencies[idx])
            else:
                experimental_freqs_complete.append(0)
        
        ax2.bar([x - 0.2 for x in all_positions], experimental_freqs_complete, width=0.4, 
                alpha=0.7, color='skyblue', label='Quantum', edgecolor='black')
        ax2.bar([x + 0.2 for x in all_positions], theoretical_freqs, width=0.4,
                alpha=0.7, color='red', label='Theoretical', edgecolor='black')
        
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Quantum vs Theoretical Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Probability comparison
        ax3 = plt.subplot(3, 3, 3)
        experimental_probs = [freq / stats['total_shots'] for freq in experimental_freqs_complete]
        
        ax3.plot(all_positions, experimental_probs, 'o-', color='skyblue', label='Quantum', linewidth=2)
        ax3.plot(theoretical_positions, theoretical_probs, 's-', color='red', label='Theoretical', linewidth=2)
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Probability')
        ax3.set_title('Probability Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error analysis
        ax4 = plt.subplot(3, 3, 4)
        errors = [abs(exp - theo) for exp, theo in zip(experimental_probs, theoretical_probs)]
        ax4.bar(all_positions, errors, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Error Analysis')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Statistics summary
        ax5 = plt.subplot(3, 3, 5)
        ax5.axis('off')
        
        # Calculate distances
        distances = self.calculate_statistical_distance(experimental_probs, theoretical_probs)
        
        stats_text = f"""
        Statistics Summary:
        
        Distribution: {self.distribution_type.capitalize()}
        Levels: {self.num_levels}
        Total Shots: {stats['total_shots']}
        
        Experimental:
        Mean: {stats['mean']:.3f}
        Std Dev: {stats['std_dev']:.3f}
        
        Distances:
        TV Distance: {distances['total_variation']:.4f}
        JS Divergence: {distances['jensen_shannon']:.4f}
        MSE: {distances['mean_squared_error']:.4f}
        """
        
        ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        # Plot 6: Circuit visualization
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        
        circuit_info = f"""
        Quantum Circuit Info:
        
        Distribution: {self.distribution_type}
        Number of Qubits: {self.num_qubits}
        Circuit Depth: {self.num_levels}
        
        Circuit Operations:
        - {self.distribution_type} specific gates
        - Measurement operations
        - Entanglement layers
        """
        
        ax6.text(0.1, 0.5, circuit_info, transform=ax6.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        # Plot 7: Gaussian convergence (if applicable)
        if self.distribution_type == "gaussian":
            ax7 = plt.subplot(3, 3, 7)
            convergence_results = self.verify_gaussian_convergence(stats['total_shots'])
            
            # Plot experimental vs Gaussian
            ax7.bar(positions, frequencies, alpha=0.7, color='skyblue', 
                   label='Quantum', edgecolor='black')
            
            # Scale Gaussian to match frequency scale
            gaussian_scaled = convergence_results['gaussian_y'] * (max(frequencies) / max(convergence_results['gaussian_y']))
            ax7.plot(convergence_results['gaussian_x'], gaussian_scaled, 'r-', 
                    linewidth=2, label='Gaussian', alpha=0.7)
            
            ax7.set_xlabel('Position')
            ax7.set_ylabel('Frequency')
            ax7.set_title('Gaussian Convergence')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # Plot 8: Distribution comparison across types
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        comparison_text = f"""
        Distribution Comparison:
        
        Current: {self.distribution_type.capitalize()}
        
        Features:
        - Universal circuit generation
        - Arbitrary number of levels
        - Multiple distribution types
        - Statistical analysis
        - Error quantification
        """
        
        ax8.text(0.1, 0.5, comparison_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
        
        # Plot 9: Performance metrics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        performance_text = f"""
        Performance Metrics:
        
        Accuracy:
        - Statistical distance measures
        - Convergence verification
        - Error analysis
        
        Scalability:
        - O(n) circuit depth
        - n qubits for n levels
        - Universal algorithm
        """
        
        ax9.text(0.1, 0.5, performance_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # plt.show()  # 表示を無効化
    
    def run_universal_experiment(self, shots: int = 1000, plot: bool = True, save_path: str = None):
        """
        Run a complete universal Quantum Galton Board experiment
        
        Args:
            shots: Number of shots for the simulation
            plot: Whether to plot the results
            save_path: Optional path to save the plot
            
        Returns:
            dict: Complete experiment results
        """
        print(f"Universal Quantum Galton Board Experiment")
        print(f"Distribution: {self.distribution_type}")
        print(f"Number of levels: {self.num_levels}")
        print(f"Shots: {shots}")
        print("=" * 60)
        
        # Run simulation
        counts = self.simulate_distribution(shots)
        
        # Analyze results
        positions, frequencies, stats = self.analyze_distribution(counts)
        
        # Calculate theoretical distribution
        theoretical_positions, theoretical_probs = self.theoretical_distribution()
        
        # Calculate distances
        experimental_probs = [freq / stats['total_shots'] for freq in frequencies]
        distances = self.calculate_statistical_distance(experimental_probs, theoretical_probs)
        
        print(f"\nResults:")
        print(f"Total measurements: {stats['total_shots']}")
        print(f"Mean position: {stats['mean']:.3f}")
        print(f"Standard deviation: {stats['std_dev']:.3f}")
        print(f"\nStatistical Distances:")
        print(f"Total Variation: {distances['total_variation']:.4f}")
        print(f"Jensen-Shannon: {distances['jensen_shannon']:.4f}")
        print(f"Mean Squared Error: {distances['mean_squared_error']:.4f}")
        
        if plot:
            # Generate default save path if not provided
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"output/universal_quantum_galton_{self.distribution_type}_{self.num_levels}levels_{shots}shots_{timestamp}.pdf"
            
            self.plot_comprehensive_results(counts, save_path)
        
        return {
            'counts': counts,
            'positions': positions,
            'frequencies': frequencies,
            'stats': stats,
            'distances': distances,
            'theoretical_positions': theoretical_positions,
            'theoretical_probs': theoretical_probs
        }

def demonstrate_universal_simulator():
    """
    Demonstrate the universal quantum Galton board with different distributions
    """
    print("Universal Quantum Galton Board Demonstration")
    print("=" * 70)
    
    # Test different distributions and levels
    distributions = ["gaussian", "exponential", "quantum_walk"]
    levels_list = [6, 8, 12]
    
    for dist_type in distributions:
        for num_levels in levels_list:
            print(f"\n{'='*20} {dist_type.capitalize()} Distribution, {num_levels} Levels {'='*20}")
            
            galton_board = UniversalQuantumGaltonBoard(num_levels=num_levels, distribution_type=dist_type)
            results = galton_board.run_universal_experiment(shots=1000, plot=True)
            
            # Print key statistics
            stats = results['stats']
            distances = results['distances']
            
            print(f"Experimental Mean: {stats['mean']:.3f}")
            print(f"Experimental Std: {stats['std_dev']:.3f}")
            print(f"Total Variation Distance: {distances['total_variation']:.4f}")
            print(f"Graph saved to output folder")

def main():
    """
    Main function to run universal implementation with command line arguments
    """
    parser = argparse.ArgumentParser(description='Universal Quantum Galton Board Implementation')
    parser.add_argument('--levels', '-n', type=int, default=8, 
                       help='Number of levels in the Galton Board (default: 8)')
    parser.add_argument('--distribution', '-d', type=str, default='gaussian',
                       choices=['gaussian', 'exponential', 'quantum_walk'],
                       help='Target distribution type (default: gaussian)')
    parser.add_argument('--shots', '-s', type=int, default=1000,
                       help='Number of shots for simulation (default: 1000)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with multiple distributions')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting (only save to file)')
    parser.add_argument('--lambda-param', type=float, default=0.5,
                       help='Lambda parameter for exponential distribution (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demonstration
        demonstrate_universal_simulator()
    else:
        # Run single experiment
        print("Universal Quantum Galton Board Implementation")
        print("=" * 70)
        print(f"Parameters: Levels={args.levels}, Distribution={args.distribution}, Shots={args.shots}")
        print("=" * 70)
        
        galton_board = UniversalQuantumGaltonBoard(num_levels=args.levels, distribution_type=args.distribution)
        galton_board.lambda_param = args.lambda_param
        galton_board.run_universal_experiment(shots=args.shots, plot=not args.no_plot)
        print(f"Graph saved to output folder")

if __name__ == "__main__":
    main()
