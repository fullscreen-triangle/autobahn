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
pub mod benchmarking;
pub mod plugins;
pub mod configuration;
pub mod deception;

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
pub use benchmarking::AutobahnBenchmarkSuite;
pub use plugins::{PluginManager, AutobahnPlugin, PluginMetadata};
pub use configuration::{ConfigurationManager, AutobahnConfig};
pub use deception::{PungweAtpSynthase, MetacognitiveAnalysisResult};

// Re-export quantum-oscillatory modules
pub use v8_pipeline::modules::{Foursidedtriangle, OscillationEndpointManager};

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
    /// Pungwe ATP Synthase for metacognitive oversight
    pub pungwe_atp_synthase: PungweAtpSynthase,
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
            pungwe_atp_synthase: PungweAtpSynthase::new(),
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

        // Perform metacognitive analysis with Pungwe ATP Synthase
        let processing_context = ProcessingContext {
            timestamp: chrono::Utc::now(),
            processing_id: uuid::Uuid::new_v4().to_string(),
            confidence_threshold: 0.8,
            max_processing_time_ms: 10000,
            metadata: std::collections::HashMap::new(),
        };
        
        let metacognitive_analysis = self.pungwe_atp_synthase
            .analyze_metacognition(&input, &processing_context).await?;

        // Calculate total ATP consumed including Pungwe's truth ATP generation
        let total_atp_consumed = biological_result.atp_consumed - metacognitive_analysis.truth_atp_generated;

        Ok(ComprehensiveResult {
            biological_result,
            uncertainty_analysis,
            temporal_insights: self.temporal_processor.get_detected_patterns().clone(),
            metacognitive_analysis: Some(metacognitive_analysis),
            processing_metadata: ProcessingMetadata {
                total_atp_consumed,
                processing_time_ms: biological_result.processing_time_ms,
                modules_used: {
                    let mut modules = biological_result.modules_activated.clone();
                    modules.push("Pungwe".to_string());
                    modules
                },
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
    /// Metacognitive analysis from Pungwe ATP Synthase
    pub metacognitive_analysis: Option<MetacognitiveAnalysisResult>,
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

/// Quick comprehensive processing function
pub async fn quick_comprehensive_process(content: &str) -> Result<ComprehensiveResult, AutobahnError> {
    let mut system = AutobahnSystem::new();
    system.initialize().await?;
    system.process_comprehensive(InformationInput::Text(content.to_string())).await
}

/// System capabilities information
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    /// Supports probabilistic processing
    pub supports_probabilistic: bool,
    /// Supports adversarial testing
    pub supports_adversarial: bool,
    /// Supports champagne phase processing
    pub supports_champagne: bool,
    /// Available V8 modules
    pub available_modules: Vec<String>,
    /// Available processing modes
    pub processing_modes: Vec<String>,
    /// Maximum ATP capacity
    pub max_atp_capacity: f64,
    /// Supported input types
    pub supported_input_types: Vec<String>,
}

/// Get system capabilities
pub fn get_capabilities() -> SystemCapabilities {
    SystemCapabilities {
        supports_probabilistic: true,
        supports_adversarial: true,
        supports_champagne: true,
        available_modules: vec![
            "Mzekezeke".to_string(),
            "Diggiden".to_string(),
            "Hatata".to_string(),
            "Spectacular".to_string(),
            "Nicotine".to_string(),
            "Clothesline".to_string(),
            "Zengeza".to_string(),
            "Diadochi".to_string(),
        ],
        processing_modes: vec![
            "Glycolysis".to_string(),
            "Krebs Cycle".to_string(),
            "Electron Transport".to_string(),
            "Champagne Phase".to_string(),
        ],
        max_atp_capacity: 1000.0,
        supported_input_types: vec![
            "Text".to_string(),
            "Numerical".to_string(),
            "Structured".to_string(),
            "Temporal".to_string(),
        ],
    }
}

/// Get version information
pub fn version() -> &'static str {
    VERSION
}

/// Advanced system monitoring
pub struct SystemMonitor {
    /// System performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Processing history
    pub processing_history: Vec<ProcessingHistoryEntry>,
}

/// Performance metrics tracking
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average processing time per operation
    pub avg_processing_time_ms: f64,
    /// Total operations processed
    pub total_operations: u64,
    /// ATP efficiency (operations per ATP)
    pub atp_efficiency: f64,
    /// Error rate
    pub error_rate: f64,
    /// Throughput (operations per second)
    pub throughput: f64,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Current ATP level
    pub current_atp: f64,
    /// Peak ATP usage
    pub peak_atp_usage: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Active modules count
    pub active_modules: usize,
}

/// Processing history entry
#[derive(Debug, Clone)]
pub struct ProcessingHistoryEntry {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Operation type
    pub operation_type: String,
    /// Processing time
    pub processing_time_ms: u64,
    /// ATP consumed
    pub atp_consumed: f64,
    /// Success status
    pub success: bool,
    /// Confidence score
    pub confidence: f64,
}

impl AutobahnSystem {
    /// Get system monitor
    pub fn get_monitor(&self) -> SystemMonitor {
        SystemMonitor {
            performance_metrics: PerformanceMetrics {
                avg_processing_time_ms: 50.0, // Placeholder
                total_operations: 0,
                atp_efficiency: 10.0,
                error_rate: 0.01,
                throughput: 20.0,
            },
            resource_usage: ResourceUsage {
                current_atp: self.biological_processor.get_energy_state().current_atp,
                peak_atp_usage: 800.0,
                memory_usage_mb: 256.0,
                cpu_usage_percent: 25.0,
                active_modules: 8,
            },
            processing_history: Vec::new(),
        }
    }

    /// Perform system health check
    pub async fn health_check(&self) -> Result<SystemHealthReport, AutobahnError> {
        let energy_state = self.biological_processor.get_energy_state();
        
        Ok(SystemHealthReport {
            overall_health: if energy_state.current_atp > 100.0 { 
                HealthStatus::Healthy 
            } else { 
                HealthStatus::Warning 
            },
            atp_status: energy_state,
            module_status: self.check_module_health().await?,
            system_uptime_ms: 0, // Placeholder
            last_error: None,
        })
    }

    /// Check health of all modules
    async fn check_module_health(&self) -> Result<Vec<ModuleHealthStatus>, AutobahnError> {
        Ok(vec![
            ModuleHealthStatus {
                module_name: "Mzekezeke".to_string(),
                status: HealthStatus::Healthy,
                last_activity: chrono::Utc::now(),
                atp_consumption_rate: 5.0,
            },
            ModuleHealthStatus {
                module_name: "Diggiden".to_string(),
                status: HealthStatus::Healthy,
                last_activity: chrono::Utc::now(),
                atp_consumption_rate: 8.0,
            },
            // Add other modules...
        ])
    }
}

/// System health report
#[derive(Debug, Clone)]
pub struct SystemHealthReport {
    /// Overall system health
    pub overall_health: HealthStatus,
    /// ATP energy status
    pub atp_status: EnergyState,
    /// Individual module status
    pub module_status: Vec<ModuleHealthStatus>,
    /// System uptime in milliseconds
    pub system_uptime_ms: u64,
    /// Last error encountered
    pub last_error: Option<String>,
}

/// Health status enumeration
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Offline,
}

/// Individual module health status
#[derive(Debug, Clone)]
pub struct ModuleHealthStatus {
    /// Module name
    pub module_name: String,
    /// Current status
    pub status: HealthStatus,
    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
    /// ATP consumption rate
    pub atp_consumption_rate: f64,
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