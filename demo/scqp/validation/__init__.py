"""
Validation suite for S-Entropy Counterfactual Quantum Processor.

Comprehensive experimental validation of all theoretical claims made in the paper.
"""

from .comprehensive_validation import run_comprehensive_validation
from .s_entropy_validation import validate_s_entropy_navigation
from .counterfactual_validation import validate_counterfactual_reasoning
from .gas_molecular_validation import validate_gas_molecular_processing
from .bmd_validation import validate_cross_modal_bmd
from .temporal_validation import validate_temporal_coordination
from .performance_validation import validate_performance_claims

__all__ = [
    "run_comprehensive_validation",
    "validate_s_entropy_navigation",
    "validate_counterfactual_reasoning", 
    "validate_gas_molecular_processing",
    "validate_cross_modal_bmd",
    "validate_temporal_coordination",
    "validate_performance_claims"
]
