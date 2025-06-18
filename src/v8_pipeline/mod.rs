//! V8 Metabolism Pipeline - The Core Biological Processing Engine
//!
//! This module implements the eight specialized intelligence modules that metabolize
//! information through authentic biological pathways: glycolysis, Krebs cycle, and
//! electron transport chain.

pub mod biological_processor;
pub mod modules;
pub mod energy_manager;
pub mod glycolysis;
pub mod krebs_cycle;
pub mod electron_transport;

// Re-export main components
pub use biological_processor::BiologicalProcessor;
pub use energy_manager::ATPManager;

// Re-export module implementations
pub use modules::{
    mzekezeke::MzekezekerModule,
    diggiden::DiggidenModule,
    hatata::HatataModule,
    spectacular::SpectacularModule,
    nicotine::NicotineModule,
    clothesline::ClotheslineModule,
    zengeza::ZengazerModule,
    diadochi::DiadochiModule,
};

use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::traits::QualityRequirements;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the V8 metabolism pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V8Configuration {
    /// Maximum ATP capacity
    pub max_atp: f64,
    /// ATP regeneration rate per second
    pub atp_regeneration_rate: f64,
    /// Enable champagne phase processing
    pub enable_champagne_phase: bool,
    /// Enable adversarial testing
    pub enable_adversarial_testing: bool,
    /// Processing strategy
    pub processing_strategy: ProcessingStrategy,
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
}

/// Processing strategies for the V8 pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStrategy {
    /// Sequential processing through all modules
    Sequential,
    /// Parallel processing where possible
    Parallel,
    /// Adaptive strategy based on content
    Adaptive,
    /// Energy-optimized processing
    EnergyOptimized,
}

/// Quality thresholds for different processing stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum confidence for glycolysis
    pub glycolysis_confidence: f64,
    /// Minimum confidence for Krebs cycle
    pub krebs_confidence: f64,
    /// Minimum confidence for electron transport
    pub electron_transport_confidence: f64,
}

/// Current stage in the V8 pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStage {
    /// Glycolysis - Context layer processing
    Glycolysis,
    /// Krebs cycle - Reasoning layer processing
    KrebsCycle,
    /// Electron transport - Intuition layer processing
    ElectronTransport,
    /// Champagne phase - Dream processing
    ChampagnePhase,
    /// Complete - Processing finished
    Complete,
}

/// Processing statistics for the V8 pipeline
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    /// Total processing operations
    pub total_operations: u64,
    /// Successful operations
    pub successful_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Total ATP consumed
    pub total_atp_consumed: f64,
    /// Total ATP generated
    pub total_atp_generated: f64,
    /// Average processing time
    pub average_processing_time_ms: f64,
    /// Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Efficiency metrics for the pipeline
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// ATP efficiency (generated/consumed)
    pub atp_efficiency: f64,
    /// Processing speed (operations/second)
    pub processing_speed: f64,
    /// Error rate (failed/total)
    pub error_rate: f64,
    /// Quality score (average confidence)
    pub quality_score: f64,
}

/// ATP allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATPAllocation {
    /// Total ATP required
    pub total_required: f64,
    /// ATP for glycolysis
    pub glycolysis_allocation: f64,
    /// ATP for Krebs cycle
    pub krebs_allocation: f64,
    /// ATP for electron transport
    pub electron_transport_allocation: f64,
    /// Reserve ATP for emergency
    pub reserve_allocation: f64,
    /// Whether allocation is feasible
    pub feasible: bool,
}

impl Default for V8Configuration {
    fn default() -> Self {
        Self {
            max_atp: 1000.0,
            atp_regeneration_rate: 10.0,
            enable_champagne_phase: true,
            enable_adversarial_testing: true,
            processing_strategy: ProcessingStrategy::Adaptive,
            quality_thresholds: QualityThresholds::default(),
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            glycolysis_confidence: 0.6,
            krebs_confidence: 0.7,
            electron_transport_confidence: 0.8,
        }
    }
}

/// Calculate ATP allocation for processing
pub fn calculate_atp_allocation(
    complexity: f64,
    quality_requirements: &QualityRequirements,
    available_atp: f64,
) -> ATPAllocation {
    // Base ATP requirements
    let base_glycolysis = 20.0;
    let base_krebs = 30.0;
    let base_electron_transport = 50.0;
    
    // Scale by complexity
    let complexity_multiplier = 1.0 + (complexity * 0.5);
    
    // Scale by quality requirements
    let quality_multiplier = 1.0 + (quality_requirements.min_confidence * 0.3);
    
    let glycolysis_allocation = base_glycolysis * complexity_multiplier * quality_multiplier;
    let krebs_allocation = base_krebs * complexity_multiplier * quality_multiplier;
    let electron_transport_allocation = base_electron_transport * complexity_multiplier * quality_multiplier;
    
    let total_required = glycolysis_allocation + krebs_allocation + electron_transport_allocation;
    let reserve_allocation = total_required * 0.1; // 10% reserve
    let total_with_reserve = total_required + reserve_allocation;
    
    let feasible = total_with_reserve <= available_atp;
    
    ATPAllocation {
        total_required: total_with_reserve,
        glycolysis_allocation,
        krebs_allocation,
        electron_transport_allocation,
        reserve_allocation,
        feasible,
    }
}

/// Optimize processing strategy based on conditions
pub fn optimize_processing_strategy(
    content_complexity: f64,
    available_atp: f64,
    time_pressure: f64,
    quality_requirements: &QualityRequirements,
) -> ProcessingStrategy {
    // High time pressure favors parallel processing
    if time_pressure > 0.8 {
        return ProcessingStrategy::Parallel;
    }
    
    // Low ATP favors energy optimization
    if available_atp < 200.0 {
        return ProcessingStrategy::EnergyOptimized;
    }
    
    // High complexity with high quality requirements favors sequential
    if content_complexity > 0.8 && quality_requirements.min_confidence > 0.8 {
        return ProcessingStrategy::Sequential;
    }
    
    // Default to adaptive
    ProcessingStrategy::Adaptive
}

/// Validate V8 pipeline configuration
pub fn validate_configuration(config: &V8Configuration) -> AutobahnResult<()> {
    if config.max_atp <= 0.0 {
        return Err(AutobahnError::ConfigurationError(
            "Maximum ATP must be greater than 0".to_string()
        ));
    }

    if config.atp_regeneration_rate <= 0.0 {
        return Err(AutobahnError::ConfigurationError(
            "ATP regeneration rate must be greater than 0".to_string()
        ));
    }

    let thresholds = &config.quality_thresholds;
    if thresholds.glycolysis_confidence < 0.0 || thresholds.glycolysis_confidence > 1.0 {
        return Err(AutobahnError::ConfigurationError(
            "Glycolysis confidence threshold must be between 0 and 1".to_string()
        ));
    }

    if thresholds.krebs_confidence < 0.0 || thresholds.krebs_confidence > 1.0 {
        return Err(AutobahnError::ConfigurationError(
            "Krebs cycle confidence threshold must be between 0 and 1".to_string()
        ));
    }

    if thresholds.electron_transport_confidence < 0.0 || thresholds.electron_transport_confidence > 1.0 {
        return Err(AutobahnError::ConfigurationError(
            "Electron transport confidence threshold must be between 0 and 1".to_string()
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v8_configuration_default() {
        let config = V8Configuration::default();
        assert_eq!(config.max_atp, 1000.0);
        assert_eq!(config.atp_regeneration_rate, 10.0);
        assert!(config.enable_champagne_phase);
        assert!(config.enable_adversarial_testing);
    }

    #[test]
    fn test_configuration_validation() {
        let mut config = V8Configuration::default();
        
        // Valid configuration should pass
        assert!(validate_configuration(&config).is_ok());
        
        // Invalid max_atp should fail
        config.max_atp = 0.0;
        assert!(validate_configuration(&config).is_err());
        
        // Reset and test invalid regeneration rate
        config = V8Configuration::default();
        config.atp_regeneration_rate = -1.0;
        assert!(validate_configuration(&config).is_err());
    }

    #[test]
    fn test_atp_allocation_calculation() {
        let quality_requirements = QualityRequirements::default();
        let allocation = calculate_atp_allocation(0.5, &quality_requirements, 500.0);
        
        assert!(allocation.total_required > 0.0);
        assert!(allocation.glycolysis_allocation > 0.0);
        assert!(allocation.krebs_allocation > 0.0);
        assert!(allocation.electron_transport_allocation > 0.0);
        assert!(allocation.reserve_allocation > 0.0);
    }

    #[test]
    fn test_atp_allocation_feasibility() {
        let quality_requirements = QualityRequirements::default();
        
        // Test with sufficient ATP
        let allocation = calculate_atp_allocation(0.5, &quality_requirements, 1000.0);
        assert!(allocation.feasible);
        
        // Test with insufficient ATP
        let allocation = calculate_atp_allocation(2.0, &quality_requirements, 50.0);
        assert!(!allocation.feasible);
    }

    #[test]
    fn test_processing_strategy_optimization() {
        let quality_requirements = QualityRequirements::default();
        
        // Test high time pressure
        let strategy = optimize_processing_strategy(0.5, 500.0, 0.9, &quality_requirements);
        assert!(matches!(strategy, ProcessingStrategy::Parallel));
        
        // Test low ATP
        let strategy = optimize_processing_strategy(0.5, 100.0, 0.3, &quality_requirements);
        assert!(matches!(strategy, ProcessingStrategy::EnergyOptimized));
        
        // Test high complexity and quality
        let high_quality = QualityRequirements {
            min_confidence: 0.9,
            ..Default::default()
        };
        let strategy = optimize_processing_strategy(0.9, 500.0, 0.3, &high_quality);
        assert!(matches!(strategy, ProcessingStrategy::Sequential));
    }

    #[test]
    fn test_processing_statistics_default() {
        let stats = ProcessingStatistics::default();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.successful_operations, 0);
        assert_eq!(stats.failed_operations, 0);
        assert_eq!(stats.total_atp_consumed, 0.0);
        assert_eq!(stats.total_atp_generated, 0.0);
    }

    #[test]
    fn test_quality_thresholds_default() {
        let thresholds = QualityThresholds::default();
        assert_eq!(thresholds.glycolysis_confidence, 0.6);
        assert_eq!(thresholds.krebs_confidence, 0.7);
        assert_eq!(thresholds.electron_transport_confidence, 0.8);
    }
} 