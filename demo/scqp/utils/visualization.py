"""Visualization utilities for SCQP"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple


def plot_s_entropy_navigation(navigation_path: List[Tuple[float, float, float]], 
                             save_path: str = None):
    """Plot S-entropy navigation path in 3D space."""
    if not navigation_path:
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    path_array = np.array(navigation_path)
    
    ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
            'b-', linewidth=2, label='Navigation Path')
    ax.scatter(path_array[0, 0], path_array[0, 1], path_array[0, 2], 
               c='green', s=100, label='Start')
    ax.scatter(path_array[-1, 0], path_array[-1, 1], path_array[-1, 2], 
               c='red', s=100, label='End')
    
    ax.set_xlabel('S_knowledge')
    ax.set_ylabel('S_time')
    ax.set_zlabel('S_entropy')
    ax.set_title('S-Entropy Navigation Path')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_convergence_analysis(validation_results: Dict[str, Any], save_path: str = None):
    """Plot convergence analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot processing times
    if 'processing_times' in validation_results:
        axes[0, 0].plot(validation_results['processing_times'])
        axes[0, 0].set_title('Processing Times')
        axes[0, 0].set_ylabel('Time (ms)')
    
    # Plot convergence rates
    if 'convergence_rates' in validation_results:
        axes[0, 1].plot(validation_results['convergence_rates'])
        axes[0, 1].set_title('Convergence Rates')
        axes[0, 1].set_ylabel('Success Rate')
    
    # Plot complexity validation
    if 'complexity_data' in validation_results:
        data = validation_results['complexity_data']
        axes[1, 0].scatter(data.get('input_sizes', []), data.get('processing_times', []))
        axes[1, 0].set_title('Complexity Analysis')
        axes[1, 0].set_xlabel('Input Size')
        axes[1, 0].set_ylabel('Time (ms)')
    
    # Plot memory usage
    if 'memory_data' in validation_results:
        data = validation_results['memory_data']
        axes[1, 1].plot(data.get('input_sizes', []), data.get('memory_usage', []))
        axes[1, 1].set_title('Memory Usage')
        axes[1, 1].set_xlabel('Input Size')
        axes[1, 1].set_ylabel('Memory (MB)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_performance_comparison(scqp_times: List[float], traditional_times: List[float], 
                              save_path: str = None):
    """Plot performance comparison between SCQP and traditional methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Processing times comparison
    x = range(len(scqp_times))
    ax1.plot(x, scqp_times, 'b-', label='SCQP', linewidth=2)
    ax1.plot(x, traditional_times, 'r-', label='Traditional', linewidth=2)
    ax1.set_title('Processing Time Comparison')
    ax1.set_xlabel('Test Case')
    ax1.set_ylabel('Time (ms)')
    ax1.legend()
    ax1.grid(True)
    
    # Improvement factor
    improvement_factors = [t / s for s, t in zip(scqp_times, traditional_times)]
    ax2.bar(x, improvement_factors, color='green', alpha=0.7)
    ax2.set_title('Speed Improvement Factor')
    ax2.set_xlabel('Test Case')
    ax2.set_ylabel('Improvement Factor (×)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
