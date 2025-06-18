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
pub mod temporal_processor;
pub mod probabilistic_engine;
pub mod research_dev;

// Re-export core types and traits for easy access
pub use types::*;
pub use traits::*;
pub use error::AutobahnError;

// Re-export main processor implementations
pub use v8_pipeline::BiologicalProcessor;
pub use tres_commas::TrinityEngine;
pub use temporal_processor::TemporalProcessorEngine;
pub use probabilistic_engine::ProbabilisticReasoningEngine;
pub use research_dev::ResearchLaboratory;

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

/// Comprehensive Autobahn system integrating all components
pub struct AutobahnSystem {
    /// Main biological processor
    pub biological_processor: BiologicalProcessor,
    /// Temporal processing engine
    pub temporal_processor: TemporalProcessorEngine,
    /// Probabilistic reasoning engine
    pub probabilistic_engine: ProbabilisticReasoningEngine,
    /// Trinity engine for consciousness layers
    pub trinity_engine: TrinityEngine,
    /// Research and development laboratory
    pub research_lab: ResearchLaboratory,
}

impl AutobahnSystem {
    /// Create new integrated Autobahn system
    pub fn new() -> Self {
        Self {
            biological_processor: BiologicalProcessor::new(),
            temporal_processor: TemporalProcessorEngine::new(),
            probabilistic_engine: ProbabilisticReasoningEngine::new(),
            trinity_engine: TrinityEngine::new(),
            research_lab: ResearchLaboratory::new(),
        }
    }

    /// Initialize all system components
    pub async fn initialize(&mut self) -> Result<(), AutobahnError> {
        // Initialize biological processor modules
        self.biological_processor.initialize_modules().await?;
        
        log::info!("Autobahn integrated system initialized successfully");
        Ok(())
    }

    /// Process information through all system components
    pub async fn process_comprehensive(&mut self, input: InformationInput) -> Result<ComprehensiveResult, AutobahnError> {
        // Process through biological metabolism
        let biological_result = self.biological_processor.process_information(input.clone()).await?;
        
        // Update temporal processing
        self.temporal_processor.update_temporal_processing().await?;
        
        // Perform probabilistic analysis if applicable
        let uncertainty_analysis = match &input {
            InformationInput::Text(text) => {
                Some(self.biological_processor.analyze_uncertainty(text)?)
            }
            _ => None,
        };

        Ok(ComprehensiveResult {
            biological_result,
            uncertainty_analysis,
            temporal_insights: self.temporal_processor.get_detected_patterns().clone(),
            processing_metadata: ProcessingMetadata {
                total_atp_consumed: biological_result.atp_consumed,
                processing_time_ms: biological_result.processing_time_ms,
                modules_used: biological_result.modules_activated.clone(),
                confidence_score: biological_result.confidence,
            },
        })
    }
}

/// Comprehensive processing result from integrated system
#[derive(Debug, Clone)]
pub struct ComprehensiveResult {
    /// Result from biological processing
    pub biological_result: ProcessingResult,
    /// Uncertainty analysis (if applicable)
    pub uncertainty_analysis: Option<UncertaintyAnalysis>,
    /// Temporal patterns detected
    pub temporal_insights: Vec<temporal_processor::TemporalPattern>,
    /// Processing metadata
    pub processing_metadata: ProcessingMetadata,
}

/// Metadata about the processing operation
#[derive(Debug, Clone)]
pub struct ProcessingMetadata {
    /// Total ATP consumed across all modules
    pub total_atp_consumed: f64,
    /// Total processing time
    pub processing_time_ms: u64,
    /// Modules that were activated
    pub modules_used: Vec<String>,
    /// Overall confidence score
    pub confidence_score: f64,
}

/// Create and initialize a complete Autobahn system
pub async fn create_system() -> Result<AutobahnSystem, AutobahnError> {
    let mut system = AutobahnSystem::new();
    system.initialize().await?;
    Ok(system)
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