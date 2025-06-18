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

/// Core V8 metabolism pipeline configuration
#[derive(Debug, Clone)]
pub struct V8Configuration {
    /// Maximum ATP capacity
    pub max_atp: f64,
    /// ATP regeneration rate per second
    pub atp_regeneration_rate: f64,
    /// Energy efficiency factor
    pub efficiency_factor: f64,
    /// Module processing timeouts
    pub module_timeouts_ms: std::collections::HashMap<String, u64>,
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// Enable different processing modes
    pub enable_champagne_phase: bool,
    pub enable_adversarial_testing: bool,
    pub enable_anaerobic_processing: bool,
}

/// Quality thresholds for biological processing
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub min_comprehension_score: f64,
    pub min_confidence_threshold: f64,
    pub max_uncertainty_tolerance: f64,
    pub robustness_requirement: f64,
}

impl Default for V8Configuration {
    fn default() -> Self {
        let mut timeouts = std::collections::HashMap::new();
        timeouts.insert("mzekezeke".to_string(), 5000);  // 5 seconds
        timeouts.insert("diggiden".to_string(), 3000);   // 3 seconds
        timeouts.insert("hatata".to_string(), 4000);     // 4 seconds
        timeouts.insert("spectacular".to_string(), 10000); // 10 seconds (paradigm detection takes time)
        timeouts.insert("nicotine".to_string(), 2000);   // 2 seconds
        timeouts.insert("clothesline".to_string(), 3000); // 3 seconds
        timeouts.insert("zengeza".to_string(), 2000);    // 2 seconds
        timeouts.insert("diadochi".to_string(), 8000);   // 8 seconds (external API calls)

        Self {
            max_atp: 1000.0,
            atp_regeneration_rate: 100.0, // 100 ATP per second
            efficiency_factor: 1.0,
            module_timeouts_ms: timeouts,
            quality_thresholds: QualityThresholds {
                min_comprehension_score: 0.7,
                min_confidence_threshold: 0.6,
                max_uncertainty_tolerance: 0.4,
                robustness_requirement: 0.8,
            },
            enable_champagne_phase: true,
            enable_adversarial_testing: true,
            enable_anaerobic_processing: true,
        }
    }
}

/// Processing pipeline stages
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    /// Context layer processing (Glycolysis)
    Glycolysis {
        nicotine_validation: bool,
        clothesline_comprehension: bool,
        zengeza_noise_reduction: bool,
    },
    /// Reasoning layer processing (Krebs Cycle)
    KrebsCycle {
        current_step: u8, // 1-8 for the 8 steps
        steps_completed: Vec<String>,
        atp_generated: f64,
        nadh_generated: f64,
        fadh2_generated: f64,
    },
    /// Intuition layer processing (Electron Transport)
    ElectronTransport {
        complex_i_complete: bool,
        complex_ii_complete: bool,
        complex_iii_complete: bool,
        complex_iv_complete: bool,
        atp_synthase_active: bool,
    },
    /// Emergency anaerobic processing
    AnaerobicProcessing {
        lactate_accumulation: f64,
        partial_results: Vec<String>,
    },
    /// Dream processing (Champagne Phase)
    ChampagnePhase {
        dream_mode: crate::traits::DreamMode,
        lactate_processing: bool,
    },
}

/// Processing statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct ProcessingStatistics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub average_atp_consumption: f64,
    pub average_processing_time_ms: f64,
    pub efficiency_score: f64,
    pub module_usage_stats: std::collections::HashMap<String, ModuleUsageStats>,
    pub pathway_distribution: PathwayDistribution,
}

/// Statistics for individual modules
#[derive(Debug, Clone)]
pub struct ModuleUsageStats {
    pub invocations: u64,
    pub total_processing_time_ms: u64,
    pub average_atp_cost: f64,
    pub success_rate: f64,
    pub error_count: u64,
}

/// Distribution of processing pathways used
#[derive(Debug, Clone)]
pub struct PathwayDistribution {
    pub aerobic_percentage: f64,
    pub anaerobic_percentage: f64,
    pub champagne_percentage: f64,
    pub hybrid_percentage: f64,
}

impl Default for ProcessingStatistics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            average_atp_consumption: 0.0,
            average_processing_time_ms: 0.0,
            efficiency_score: 0.0,
            module_usage_stats: std::collections::HashMap::new(),
            pathway_distribution: PathwayDistribution {
                aerobic_percentage: 100.0,
                anaerobic_percentage: 0.0,
                champagne_percentage: 0.0,
                hybrid_percentage: 0.0,
            },
        }
    }
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

    if config.efficiency_factor <= 0.0 || config.efficiency_factor > 2.0 {
        return Err(AutobahnError::ConfigurationError(
            "Efficiency factor must be between 0 and 2".to_string()
        ));
    }

    let thresholds = &config.quality_thresholds;
    if thresholds.min_comprehension_score < 0.0 || thresholds.min_comprehension_score > 1.0 {
        return Err(AutobahnError::ConfigurationError(
            "Comprehension score threshold must be between 0 and 1".to_string()
        ));
    }

    if thresholds.min_confidence_threshold < 0.0 || thresholds.min_confidence_threshold > 1.0 {
        return Err(AutobahnError::ConfigurationError(
            "Confidence threshold must be between 0 and 1".to_string()
        ));
    }

    Ok(())
}

/// Calculate optimal ATP allocation for processing
pub fn calculate_atp_allocation(
    content_complexity: f64,
    quality_requirements: &crate::traits::QualityRequirements,
    available_atp: f64,
) -> ATPAllocation {
    let base_allocation = content_complexity * 50.0; // Base cost per complexity unit
    
    // Adjust for quality requirements
    let quality_multiplier = if quality_requirements.robustness_required { 1.5 } else { 1.0 };
    let adversarial_multiplier = if quality_requirements.adversarial_testing { 1.3 } else { 1.0 };
    let confidence_multiplier = quality_requirements.min_confidence * 1.2;
    
    let total_required = base_allocation * quality_multiplier * adversarial_multiplier * confidence_multiplier;
    
    // Distribute across processing layers
    let glycolysis_allocation = total_required * 0.1;   // 10% for initial processing
    let krebs_allocation = total_required * 0.3;        // 30% for reasoning
    let electron_transport_allocation = total_required * 0.6; // 60% for final synthesis
    
    ATPAllocation {
        total_required,
        glycolysis_allocation,
        krebs_allocation,
        electron_transport_allocation,
        reserve_allocation: (available_atp - total_required).max(0.0) * 0.2, // 20% reserve
        feasible: total_required <= available_atp,
    }
}

/// ATP allocation plan for processing
#[derive(Debug, Clone)]
pub struct ATPAllocation {
    pub total_required: f64,
    pub glycolysis_allocation: f64,
    pub krebs_allocation: f64,
    pub electron_transport_allocation: f64,
    pub reserve_allocation: f64,
    pub feasible: bool,
}

/// Determine processing strategy based on available resources and requirements
pub fn determine_processing_strategy(
    atp_allocation: &ATPAllocation,
    energy_state: &EnergyState,
    quality_requirements: &crate::traits::QualityRequirements,
) -> ProcessingStrategy {
    if !atp_allocation.feasible {
        if energy_state.current_atp >= atp_allocation.total_required * 0.6 {
            // Can do partial processing
            return ProcessingStrategy::Anaerobic {
                partial_processing: true,
                lactate_tolerance: 0.3,
                fallback_mode: FallbackMode::ReducedQuality,
            };
        } else {
            // Need to wait or use emergency mode
            return ProcessingStrategy::Emergency {
                minimal_processing: true,
                confidence_reduction: 0.5,
            };
        }
    }

    // Determine optimal pathway
    if quality_requirements.adversarial_testing && quality_requirements.robustness_required {
        ProcessingStrategy::FullAerobic {
            enable_adversarial: true,
            comprehensive_validation: true,
            quality_assurance: QualityAssurance::Maximum,
        }
    } else if quality_requirements.min_confidence > 0.8 {
        ProcessingStrategy::HighQuality {
            enable_champagne_optimization: true,
            iterative_refinement: true,
            quality_assurance: QualityAssurance::High,
        }
    } else {
        ProcessingStrategy::Balanced {
            optimize_for_speed: false,
            quality_assurance: QualityAssurance::Standard,
        }
    }
}

/// Processing strategy options
#[derive(Debug, Clone)]
pub enum ProcessingStrategy {
    FullAerobic {
        enable_adversarial: bool,
        comprehensive_validation: bool,
        quality_assurance: QualityAssurance,
    },
    HighQuality {
        enable_champagne_optimization: bool,
        iterative_refinement: bool,
        quality_assurance: QualityAssurance,
    },
    Balanced {
        optimize_for_speed: bool,
        quality_assurance: QualityAssurance,
    },
    Anaerobic {
        partial_processing: bool,
        lactate_tolerance: f64,
        fallback_mode: FallbackMode,
    },
    Emergency {
        minimal_processing: bool,
        confidence_reduction: f64,
    },
}

/// Quality assurance levels
#[derive(Debug, Clone)]
pub enum QualityAssurance {
    Maximum,
    High,
    Standard,
    Minimal,
}

/// Fallback modes for resource-constrained processing
#[derive(Debug, Clone)]
pub enum FallbackMode {
    ReducedQuality,
    PartialResults,
    CachedResults,
    DeferredProcessing,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v8_configuration_default() {
        let config = V8Configuration::default();
        assert_eq!(config.max_atp, 1000.0);
        assert_eq!(config.atp_regeneration_rate, 100.0);
        assert_eq!(config.efficiency_factor, 1.0);
        assert!(config.enable_champagne_phase);
        assert!(config.enable_adversarial_testing);
        assert!(config.enable_anaerobic_processing);
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
    fn test_atp_allocation() {
        let quality_req = crate::traits::QualityRequirements::default();
        let allocation = calculate_atp_allocation(1.0, &quality_req, 1000.0);
        
        assert!(allocation.total_required > 0.0);
        assert!(allocation.feasible);
        assert!(allocation.glycolysis_allocation > 0.0);
        assert!(allocation.krebs_allocation > 0.0);
        assert!(allocation.electron_transport_allocation > 0.0);
    }

    #[test]
    fn test_processing_strategy_determination() {
        let quality_req = crate::traits::QualityRequirements {
            min_confidence: 0.9,
            max_uncertainty: 0.1,
            robustness_required: true,
            adversarial_testing: true,
        };
        
        let allocation = calculate_atp_allocation(1.0, &quality_req, 1000.0);
        let energy_state = EnergyState::new(1000.0);
        
        let strategy = determine_processing_strategy(&allocation, &energy_state, &quality_req);
        
        match strategy {
            ProcessingStrategy::FullAerobic { enable_adversarial, .. } => {
                assert!(enable_adversarial);
            }
            _ => panic!("Expected FullAerobic strategy for high requirements"),
        }
    }

    #[test]
    fn test_pipeline_stage_progression() {
        let glycolysis = PipelineStage::Glycolysis {
            nicotine_validation: true,
            clothesline_comprehension: true,
            zengeza_noise_reduction: true,
        };
        
        match glycolysis {
            PipelineStage::Glycolysis { nicotine_validation, .. } => {
                assert!(nicotine_validation);
            }
            _ => panic!("Expected Glycolysis stage"),
        }
    }
} 