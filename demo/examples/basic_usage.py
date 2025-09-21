"""
Basic usage example for S-Entropy Counterfactual Quantum Processor (SCQP)

This example demonstrates the core functionality of the SCQP system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scqp import SCQProcessor
from scqp.validation import run_comprehensive_validation
import numpy as np


def main():
    print("=== S-Entropy Counterfactual Quantum Processor Demo ===\n")
    
    # Initialize the SCQP system
    print("Initializing SCQP...")
    processor = SCQProcessor(
        s_entropy_dimensions=3,
        counterfactual_depth=5,
        gas_molecular_count=100,
        bmd_threshold=0.75
    )
    
    # Example 1: Simple environmental processing
    print("\\n1. Processing simple environmental context...")
    simple_context = {
        'visual': {
            'facial_expression': {'comprehension_level': 0.8},
            'posture': {'engagement_level': 0.7}
        },
        'audio': {
            'vocal_energy': 0.6,
            'pause_frequency': 0.3
        },
        'semantic': {
            'text': 'I think I understand this concept now',
            'reasoning_quality': 0.7
        }
    }
    
    result = processor.process_environmental_meaning(simple_context)
    
    print(f"Synthesized meaning: {result.meaning}")
    print(f"Consciousness index: {result.consciousness_index:.3f}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"S-entropy coordinates: ({result.s_entropy_coordinates[0]:.3f}, {result.s_entropy_coordinates[1]:.3f}, {result.s_entropy_coordinates[2]:.3f})")
    print(f"Counterfactual depth: {result.counterfactual_depth}")
    print(f"Cross-modal convergence: {result.cross_modal_convergence:.3f}")
    
    # Example 2: Complex environmental processing
    print("\\n2. Processing complex environmental context...")
    complex_context = {
        'visual': {
            'facial_expression': {'comprehension_level': 0.4, 'confusion_indicators': 0.6},
            'eye_tracking': {'attention_focus': 0.3, 'scanning_pattern': 'scattered'},
            'posture': {'engagement_level': 0.5, 'tension_level': 0.7}
        },
        'audio': {
            'vocal_energy': 0.3,
            'pause_frequency': 0.8,
            'hesitation_markers': 0.7,
            'environmental_noise': 0.2
        },
        'semantic': {
            'text': 'This is quite complex. I am not sure I fully grasp the implications of these counterfactual scenarios in relation to the gas molecular processing.',
            'reasoning_quality': 0.4,
            'uncertainty_indicators': 0.8,
            'conceptual_depth': 0.9
        }
    }
    
    result = processor.process_environmental_meaning(complex_context)
    
    print(f"Synthesized meaning: {result.meaning}")
    print(f"Consciousness index: {result.consciousness_index:.3f}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"Formal proof validity: {result.formal_proof_validity}")
    print(f"Gas molecular equilibrium energy: {result.gas_molecular_equilibrium.get('total_semantic_energy', 0):.2f}")
    
    # Example 3: System status
    print("\\n3. System status:")
    status = processor.get_system_status()
    for component, component_status in status.items():
        if component_status:
            print(f"  {component}: {component_status.get('status', 'unknown')}")
    
    # Example 4: Run validation (optional - can take some time)
    run_validation = input("\\nRun comprehensive validation? (y/n): ").lower().strip()
    if run_validation == 'y':
        print("\\n4. Running comprehensive validation...")
        validation_results = run_comprehensive_validation()
        print(validation_results.summary())
    else:
        print("\\nSkipping validation. To run validation later, use:")
        print("  from scqp.validation import run_comprehensive_validation")
        print("  results = run_comprehensive_validation()")
        print("  print(results.summary())")
    
    print("\\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()
