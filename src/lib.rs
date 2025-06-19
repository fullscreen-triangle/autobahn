//! Autobahn: Oscillatory Bio-Metabolic Computation System
//! 
//! A complete implementation of the theoretical framework described in code.md,
//! providing quantum-enhanced biological computation through oscillatory dynamics.
//! 
//! # Core Principles
//! 
//! 1. **Universal Oscillation Equation**: d²y/dt² + γ(dy/dt) + ω²y = F(t)
//! 2. **Membrane Quantum Computation Theorem**: Biological membranes as quantum computers
//! 3. **Oscillatory Entropy Theorem**: Information content through oscillation termination
//! 4. **10-Level Hierarchy**: From quantum (10⁻⁴⁴s) to cosmic (10¹³s) scales
//! 5. **ENAQT Processing**: Environment-Assisted Quantum Transport
//! 
//! # Architecture
//! 
//! The system implements a complete biological computation architecture:
//! - Quantum membrane processing with ENAQT optimization
//! - ATP-driven metabolic modes (flight, cold-blooded, mammalian, anaerobic)
//! - Multi-scale hierarchy processing across 10 time scales
//! - Three biological layers: Context → Reasoning → Intuition
//! - Oscillatory entropy calculations for information content
//! 
//! # Usage
//! 
//! ```rust
//! use autobahn::{
//!     oscillatory_rag::{OscillatoryRAGSystem, OscillatoryRAGConfig},
//!     hierarchy::HierarchyLevel,
//!     atp::MetabolicMode,
//! };
//! 
//! let config = OscillatoryRAGConfig {
//!     temperature: 285.0, // Cold-blooded advantage
//!     target_entropy: 2.0,
//!     oscillation_dimensions: 8,
//!     hierarchy_levels: vec![
//!         HierarchyLevel::CellularOscillations,
//!         HierarchyLevel::OrganismalOscillations,
//!         HierarchyLevel::CognitiveOscillations,
//!     ],
//!     metabolic_mode: MetabolicMode::ColdBlooded {
//!         temperature_advantage: 1.4,
//!         metabolic_reduction: 0.7,
//!     },
//! };
//! 
//! let mut system = OscillatoryRAGSystem::new(config).await?;
//! let result = system.process_query("Complex biological question").await?;
//! ```

pub mod error;
pub mod types;
pub mod traits;
pub mod configuration;
pub mod plugins;
pub mod benchmarking;
pub mod research_dev;
pub mod temporal_processor;
pub mod deception;
pub mod probabilistic_engine;

// Core oscillatory dynamics and quantum processing
pub mod oscillatory;
pub mod quantum;
pub mod atp;
pub mod hierarchy;
pub mod biological;

// Enhanced AI modules with intelligent optimization
pub mod entropy;
pub mod adversarial;
pub mod models;
pub mod rag;
pub mod consciousness;

// Main system integration
pub mod oscillatory_rag;

// V8 biological processing pipeline
pub mod v8_pipeline;

// Specialized modules
pub mod champagne;
pub mod tres_commas;
pub mod utils;

// Re-export core types and errors
pub use error::{AutobahnError, AutobahnResult};
pub use types::*;
pub use traits::*;

// Re-export main system components
pub use oscillatory_rag::{OscillatoryRAGSystem, OscillatoryRAGConfig, OscillatoryQuery, OscillatoryResponse};
pub use oscillatory::{UniversalOscillator, OscillationState, OscillationProfile, OscillationPhase};
pub use quantum::{QuantumMembraneProcessor, QuantumMembraneState, ENAQTProcessor};
pub use atp::{ATPManager, QuantumATPManager, MetabolicMode, ATPState};
pub use hierarchy::{NestedHierarchyProcessor, HierarchyLevel, HierarchyResult};
pub use biological::{BiologicalLayerProcessor, BiologicalLayer, BiologicalProcessingResult};
pub use consciousness::{
    ConsciousnessEmergenceEngine, FireConsciousnessEngine, 
    FireConsciousnessResponse, FireRecognitionResponse, AgencyDetection,
    IonType, BiologicalMaxwellDemon, BMDSpecialization, FireEnvironment,
    QuantumCoherenceField, UnderwaterFireplaceTest, ConsciousExperience
};

// Re-export utility functions
pub use utils::helpers::*;

// System configuration constants from the specification
pub const PLANCK_TIME_SECONDS: f64 = 1e-44;
pub const QUANTUM_COHERENCE_THRESHOLD: f64 = 1e-12;
pub const MEMBRANE_THICKNESS_NM: f64 = 7.0;
pub const ATP_SYNTHASE_EFFICIENCY: f64 = 0.85;
pub const RADICAL_DAMAGE_THRESHOLD: f64 = 1e-3;
pub const COSMIC_TIME_SCALE_YEARS: f64 = 3.17e5; // 10¹³ seconds in years

/// System-wide configuration for the oscillatory bio-metabolic system
#[derive(Debug, Clone)]
pub struct AutobahnSystemConfig {
    /// Operating temperature in Kelvin
    pub temperature_k: f64,
    /// Target entropy for oscillatory processing
    pub target_entropy: f64,
    /// Number of oscillatory dimensions
    pub oscillation_dimensions: usize,
    /// Hierarchy levels to activate
    pub active_hierarchy_levels: Vec<HierarchyLevel>,
    /// Biological layers to process through
    pub active_biological_layers: Vec<BiologicalLayer>,
    /// Initial metabolic mode
    pub metabolic_mode: MetabolicMode,
    /// Quantum enhancement enabled
    pub quantum_enhancement: bool,
    /// ENAQT optimization enabled
    pub enaqt_optimization: bool,
    /// Maximum processing time per query in milliseconds
    pub max_processing_time_ms: u64,
    /// ATP regeneration rate per second
    pub atp_regeneration_rate: f64,
}

impl Default for AutobahnSystemConfig {
    fn default() -> Self {
        Self {
            temperature_k: 285.0, // Cold-blooded advantage
            target_entropy: 2.0,
            oscillation_dimensions: 8,
            active_hierarchy_levels: vec![
                HierarchyLevel::MolecularOscillations,
                HierarchyLevel::CellularOscillations,
                HierarchyLevel::OrganismalOscillations,
                HierarchyLevel::CognitiveOscillations,
            ],
            active_biological_layers: vec![
                BiologicalLayer::Context,
                BiologicalLayer::Reasoning,
                BiologicalLayer::Intuition,
            ],
            metabolic_mode: MetabolicMode::ColdBlooded {
                temperature_advantage: 1.4,
                metabolic_reduction: 0.7,
            },
            quantum_enhancement: true,
            enaqt_optimization: true,
            max_processing_time_ms: 30000,
            atp_regeneration_rate: 100.0,
        }
    }
}

/// Initialize the complete oscillatory bio-metabolic system
pub async fn initialize_system(config: AutobahnSystemConfig) -> AutobahnResult<OscillatoryRAGSystem> {
    log::info!("🧬 Initializing Autobahn Oscillatory Bio-Metabolic System");
    log::info!("   Temperature: {:.1} K", config.temperature_k);
    log::info!("   Metabolic Mode: {:?}", config.metabolic_mode);
    log::info!("   Hierarchy Levels: {} active", config.active_hierarchy_levels.len());
    log::info!("   Biological Layers: {} active", config.active_biological_layers.len());
    log::info!("   Quantum Enhancement: {}", config.quantum_enhancement);
    log::info!("   ENAQT Optimization: {}", config.enaqt_optimization);
    
    let rag_config = OscillatoryRAGConfig {
        temperature: config.temperature_k,
        target_entropy: config.target_entropy,
        oscillation_dimensions: config.oscillation_dimensions,
        hierarchy_levels: config.active_hierarchy_levels,
        metabolic_mode: config.metabolic_mode,
        quantum_enhancement: config.quantum_enhancement,
        enaqt_optimization: config.enaqt_optimization,
        max_processing_time_ms: config.max_processing_time_ms,
        atp_regeneration_rate: config.atp_regeneration_rate,
    };
    
    let system = OscillatoryRAGSystem::new(rag_config).await?;
    
    log::info!("✅ Autobahn system initialized successfully");
    Ok(system)
}

// Integration tests (conditionally compiled)
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_initialization() {
        let config = AutobahnSystemConfig::default();
        let system = initialize_system(config).await;
        assert!(system.is_ok());
    }
    
    #[tokio::test]
    async fn test_hierarchy_processing() {
        let config = AutobahnSystemConfig::default();
        let mut system = initialize_system(config).await.unwrap();
        let response = system.process_query("Test hierarchical processing").await;
        assert!(response.is_ok());
    }
    
    #[tokio::test]
    async fn test_quantum_enhancement() {
        let mut config = AutobahnSystemConfig::default();
        config.quantum_enhancement = true;
        config.enaqt_optimization = true;
        
        let mut system = initialize_system(config).await.unwrap();
        let response = system.process_query("Test quantum-enhanced processing").await;
        assert!(response.is_ok());
    }
    
    #[tokio::test]
    async fn test_metabolic_modes() {
        for mode in vec![
            MetabolicMode::ColdBlooded { temperature_advantage: 1.4, metabolic_reduction: 0.7 },
            MetabolicMode::SustainedFlight { efficiency_boost: 2.5, oxidative_capacity: 3.0 },
            MetabolicMode::MammalianBurden { quantum_cost_multiplier: 1.2, radical_generation_rate: 1e-5 },
        ] {
            let mut config = AutobahnSystemConfig::default();
            config.metabolic_mode = mode.clone();
            
            let mut system = initialize_system(config).await.unwrap();
            let response = system.process_query(&format!("Test {:?} metabolic mode", mode)).await;
            assert!(response.is_ok());
        }
    }
} 