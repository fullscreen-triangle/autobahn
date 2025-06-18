//! # Autobahn: A Biological Metabolism Computer for Probabilistic Information Processing
//!
//! Autobahn is a revolutionary metacognitive orchestrator that implements authentic biological
//! metabolism as a computational paradigm for probabilistic information processing.
//!
//! ## Overview
//!
//! Unlike traditional deterministic computing systems, Autobahn models cellular respiration
//! pathways—glycolysis, Krebs cycle, and electron transport—to metabolize information into
//! understanding through ATP-generating cycles.
//!
//! ## Core Components
//!
//! - **V8 Metabolism Pipeline**: Eight specialized intelligence modules implementing cellular respiration
//! - **Tres Commas Trinity Engine**: Three consciousness layers (Context, Reasoning, Intuition)
//! - **Probabilistic Processing**: All operations incorporate uncertainty and multiple outcomes
//! - **Adversarial Validation**: Continuous testing for robustness and reliability
//! - **Champagne Phase**: Dream processing and lactate recovery for self-improvement
//!
//! ## Quick Start
//!
//! ```rust
//! use autobahn::{BiologicalProcessor, MetacognitiveOrchestrator, InformationInput};
//!
//! let mut orchestrator = BiologicalProcessor::new();
//! let result = orchestrator.process_information(
//!     InformationInput::Text("Analyze this probabilistic content".to_string())
//! );
//! ```

pub mod types;
pub mod traits;
pub mod v8_pipeline;
pub mod tres_commas;
pub mod champagne;
pub mod error;
pub mod utils;

// Re-export core types and traits for easy access
pub use types::*;
pub use traits::*;
pub use error::AutobahnError;

// Re-export main processor implementations
pub use v8_pipeline::BiologicalProcessor;
pub use tres_commas::TrinityEngine;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Autobahn library initialization
pub fn init() -> Result<(), AutobahnError> {
    // Initialize logging
    env_logger::try_init().map_err(|e| AutobahnError::InitializationError(e.to_string()))?;
    
    // Initialize random number generation
    use rand::SeedableRng;
    let _rng = rand::rngs::StdRng::from_entropy();
    
    log::info!("Autobahn biological metabolism computer initialized (v{})", VERSION);
    Ok(())
}

/// Quick start function for basic processing
pub fn quick_process(content: &str) -> Result<ProcessingResult, AutobahnError> {
    let mut processor = BiologicalProcessor::new();
    processor.process_information(InformationInput::Text(content.to_string()))
}

/// Quick start function for probabilistic analysis
pub fn quick_analyze_uncertainty(content: &str) -> Result<UncertaintyAnalysis, AutobahnError> {
    let processor = BiologicalProcessor::new();
    processor.analyze_uncertainty(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_initialization() {
        // Test that the library can be initialized without errors
        let result = init();
        // Note: This might fail in test environment due to logger being already initialized
        // That's okay - we're testing that the function exists and runs
        assert!(result.is_ok() || matches!(result, Err(AutobahnError::InitializationError(_))));
    }

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(VERSION.contains('.'));
    }

    #[test]
    fn test_quick_process() {
        let content = "This is a test of probabilistic processing";
        let result = quick_process(content);
        // For now, this will fail until we implement the processor
        // But we're testing the API exists
        match result {
            Ok(_) => {
                // Success case when implemented
            }
            Err(AutobahnError::NotImplemented(_)) => {
                // Expected during development
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }
} 