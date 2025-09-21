"""Utility functions for SCQP"""

from .mathematical import (
    s_entropy_distance,
    gibbs_energy_calculation,
    precision_by_difference
)
from .data_processing import (
    normalize_coordinates,
    validate_environmental_context,
    extract_pattern_features
)
from .visualization import (
    plot_s_entropy_navigation,
    plot_convergence_analysis,
    plot_performance_comparison
)

__all__ = [
    "s_entropy_distance",
    "gibbs_energy_calculation", 
    "precision_by_difference",
    "normalize_coordinates",
    "validate_environmental_context",
    "extract_pattern_features",
    "plot_s_entropy_navigation",
    "plot_convergence_analysis",
    "plot_performance_comparison"
]
