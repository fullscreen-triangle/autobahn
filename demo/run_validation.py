#!/usr/bin/env python3
"""
Run comprehensive SCQP validation suite
"""

import sys
import logging
from scqp.validation import run_comprehensive_validation


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Starting S-Entropy Counterfactual Quantum Processor validation...")
    print("This will validate all theoretical claims made in the paper.")
    print("-" * 60)
    
    try:
        # Run comprehensive validation
        results = run_comprehensive_validation()
        
        # Print detailed summary
        print(results.summary())
        
        # Exit with appropriate code
        if results.overall_success:
            print("\\n✓ All validations passed successfully!")
            sys.exit(0)
        else:
            print("\\n✗ Some validations failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\\nError during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
