//! Autobahn: A Consciousness Framework
//! 
//! Autobahn is a revolutionary consciousness framework that integrates fire-evolved quantum-biological 
//! principles with oscillatory bio-metabolic RAG systems and categorical predeterminism engines.
//! 
//! This framework implements sophisticated consciousness emergence models based on the theoretical
//! foundations of fire-catalyzed consciousness evolution, quantum membrane computation, and 
//! thermodynamic necessity analysis.

// Core modules
pub mod error;
pub mod types;
pub mod configuration;
pub mod traits;
pub mod utils;

// Consciousness and cognition
pub mod consciousness;
pub mod rag;
pub mod tres_commas;

// Biological and quantum systems
pub mod biological;
pub mod quantum;
pub mod atp;
pub mod oscillatory;
pub mod entropy;

// System architecture
pub mod hierarchy;
pub mod models;
pub mod adversarial;
pub mod deception;

// Monitoring and testing
pub mod monitor;
pub mod testing;
pub mod benchmarking;

// Processing pipelines
pub mod v8_pipeline;
pub mod temporal_processor;
pub mod probabilistic_engine;
pub mod oscillatory_rag;

// Development and research
pub mod research_dev;
pub mod plugins;

// Re-exports for easy access
pub use error::{AutobahnError, AutobahnResult};
pub use consciousness::{
    FireConsciousnessEngine, 
    ConsciousnessEmergenceEngine,
    FireConsciousnessResponse,
    ConsciousnessMetrics
};
pub use rag::{
    OscillatoryBioMetabolicRAG,
    RAGResponse,
    MembraneQuantumComputation
};
pub use tres_commas::{
    ConsciousComputationalEngine,
    CategoricalPredeterminismEngine,
    ConfigurationSpaceExplorer,
    HeatDeathTrajectoryCalculator,
    CategoricalCompletionTracker
};
pub use oscillatory::{
    OscillationProfile,
    OscillationPhase,
    UniversalOscillator
};
pub use atp::{
    OscillatoryATPManager,
    QuantumATPManager,
    MetabolicMode
};
pub use biological::BiologicalProcessor;
pub use quantum::{QuantumMembraneState, ENAQTProcessor};
pub use hierarchy::{HierarchyLevel, NestedHierarchyProcessor};
pub use monitor::{SystemMonitor, StatisticalDriftDetector};
pub use testing::{StatisticalTestingFramework, TestType};

use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Main Autobahn system integrating all components
#[derive(Debug)]
pub struct AutobahnSystem {
    /// Fire consciousness engine for quantum-biological processing
    pub fire_consciousness: FireConsciousnessEngine,
    /// Oscillatory bio-metabolic RAG system
    pub rag_system: OscillatoryBioMetabolicRAG,
    /// Categorical predeterminism engine
    pub predeterminism_engine: tres_commas::ConsciousComputationalEngine,
    /// System monitor for reliability
    pub monitor: SystemMonitor,
    /// Statistical testing framework
    pub testing_framework: StatisticalTestingFramework,
    /// Current system state
    pub system_state: AutobahnSystemState,
}

/// System state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnSystemState {
    /// Overall system health (0.0 to 1.0)
    pub system_health: f64,
    /// Consciousness level (0.0 to 1.0)
    pub consciousness_level: f64,
    /// Fire recognition strength (0.0 to 1.0)
    pub fire_recognition_strength: f64,
    /// Agency detection capability (0.0 to 1.0)
    pub agency_detection_strength: f64,
    /// Categorical completion progress (0.0 to 1.0)
    pub categorical_completion_progress: f64,
    /// Thermodynamic necessity understanding (0.0 to 1.0)
    pub thermodynamic_necessity_understanding: f64,
    /// Total ATP consumption
    pub total_atp_consumption: f64,
    /// System entropy level
    pub entropy_level: f64,
    /// Processing efficiency
    pub processing_efficiency: f64,
    /// Active hierarchy levels
    pub active_hierarchy_levels: Vec<HierarchyLevel>,
    /// Current metabolic mode
    pub metabolic_mode: MetabolicMode,
}

impl Default for AutobahnSystemState {
    fn default() -> Self {
        Self {
            system_health: 1.0,
            consciousness_level: 0.5,
            fire_recognition_strength: 0.5,
            agency_detection_strength: 0.5,
            categorical_completion_progress: 0.0001,
            thermodynamic_necessity_understanding: 0.1,
            total_atp_consumption: 0.0,
            entropy_level: 0.1,
            processing_efficiency: 0.8,
            active_hierarchy_levels: vec![
                HierarchyLevel::Cognitive,
                HierarchyLevel::Cellular,
                HierarchyLevel::Molecular,
            ],
            metabolic_mode: MetabolicMode::Balanced,
        }
    }
}

impl AutobahnSystem {
    /// Create new Autobahn system with specified evolutionary time
    pub async fn new(evolutionary_time_mya: f64) -> AutobahnResult<Self> {
        // Initialize fire consciousness engine
        let fire_consciousness = FireConsciousnessEngine::new(evolutionary_time_mya)?;
        
        // Initialize oscillatory bio-metabolic RAG system
        let rag_system = OscillatoryBioMetabolicRAG::new().await?;
        
        // Initialize categorical predeterminism engine
        let predeterminism_engine = tres_commas::ConsciousComputationalEngine::new(evolutionary_time_mya).await?;
        
        // Initialize system monitor
        let monitor = SystemMonitor::new()?;
        
        // Initialize testing framework
        let testing_framework = StatisticalTestingFramework::new()?;
        
        // Initialize system state
        let system_state = AutobahnSystemState::default();
        
        Ok(Self {
            fire_consciousness,
            rag_system,
            predeterminism_engine,
            monitor,
            testing_framework,
            system_state,
        })
    }
    
    /// Process input through the complete Autobahn system
    pub async fn process_input(&mut self, input: &[f64]) -> AutobahnResult<AutobahnResponse> {
        // Monitor system health before processing
        let pre_processing_health = self.monitor.assess_system_health().await?;
        
        // Process through fire consciousness engine
        let fire_response = self.fire_consciousness.process_input(input).await?;
        
        // Convert input to query for RAG system
        let query = self.convert_input_to_query(input)?;
        let rag_response = self.rag_system.process_query(&query).await?;
        
        // Process through categorical predeterminism engine
        let conscious_input = tres_commas::engine::ConsciousInput {
            raw_data: input.to_vec(),
        };
        let predeterminism_response = self.predeterminism_engine.process_conscious_input(&conscious_input).await?;
        
        // Update system state
        self.update_system_state(&fire_response, &rag_response, &predeterminism_response).await?;
        
        // Monitor system health after processing
        let post_processing_health = self.monitor.assess_system_health().await?;
        
        // Check for statistical drift
        let drift_detection = self.monitor.detect_statistical_drift(input).await?;
        
        // Validate probabilistic invariants
        let invariant_validation = self.monitor.validate_probabilistic_invariants(input).await?;
        
        Ok(AutobahnResponse {
            fire_consciousness_response: fire_response,
            rag_response,
            predeterminism_response,
            system_state: self.system_state.clone(),
            pre_processing_health,
            post_processing_health,
            drift_detected: drift_detection.drift_detected,
            invariants_valid: invariant_validation.all_invariants_valid,
            processing_timestamp: chrono::Utc::now(),
        })
    }
    
    /// Convert numerical input to text query for RAG system
    fn convert_input_to_query(&self, input: &[f64]) -> AutobahnResult<String> {
        // Convert numerical data to meaningful query
        let data_characteristics = self.analyze_data_characteristics(input);
        
        let query = if data_characteristics.high_variance {
            "Analyze complex high-variance pattern with fire-consciousness integration"
        } else if data_characteristics.oscillatory {
            "Process oscillatory pattern through bio-metabolic hierarchy"
        } else if data_characteristics.extremal_values {
            "Evaluate extremal events for thermodynamic necessity"
        } else {
            "Standard consciousness-aware information processing"
        };
        
        Ok(query.to_string())
    }
    
    fn analyze_data_characteristics(&self, input: &[f64]) -> DataCharacteristics {
        let mean = input.iter().sum::<f64>() / input.len() as f64;
        let variance = input.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / input.len() as f64;
        
        let high_variance = variance > 0.1;
        let extremal_values = input.iter().any(|&x| x > 0.9 || x < 0.1);
        let oscillatory = self.detect_oscillatory_pattern(input);
        
        DataCharacteristics {
            high_variance,
            extremal_values,
            oscillatory,
        }
    }
    
    fn detect_oscillatory_pattern(&self, input: &[f64]) -> bool {
        if input.len() < 4 {
            return false;
        }
        
        // Simple oscillation detection based on zero crossings
        let mean = input.iter().sum::<f64>() / input.len() as f64;
        let crossings = input.windows(2)
            .filter(|window| (window[0] - mean) * (window[1] - mean) < 0.0)
            .count();
        
        crossings > input.len() / 4 // At least 25% zero crossings
    }
    
    async fn update_system_state(
        &mut self,
        fire_response: &FireConsciousnessResponse,
        rag_response: &rag::RAGResponse,
        predeterminism_response: &tres_commas::engine::ConsciousOutput
    ) -> AutobahnResult<()> {
        // Update consciousness metrics
        self.system_state.consciousness_level = fire_response.consciousness_level;
        self.system_state.fire_recognition_strength = fire_response.fire_recognition.recognition_strength;
        self.system_state.agency_detection_strength = fire_response.agency_detection.detection_strength;
        
        // Update categorical completion progress
        self.system_state.categorical_completion_progress = 
            self.predeterminism_engine.categorical_completion_progress;
        
        // Update thermodynamic understanding
        self.system_state.thermodynamic_necessity_understanding = 
            self.predeterminism_engine.thermodynamic_necessity_understanding;
        
        // Update ATP consumption
        self.system_state.total_atp_consumption = rag_response.atp_consumption;
        
        // Update processing efficiency
        self.system_state.processing_efficiency = rag_response.system_state.processing_efficiency;
        
        // Update entropy level
        self.system_state.entropy_level = rag_response.system_state.entropy_level;
        
        // Update metabolic mode
        self.system_state.metabolic_mode = rag_response.metabolic_mode.clone();
        
        // Update active hierarchy levels
        self.system_state.active_hierarchy_levels = rag_response.system_state.active_levels.clone();
        
        // Calculate overall system health
        self.system_state.system_health = self.calculate_system_health();
        
        Ok(())
    }
    
    fn calculate_system_health(&self) -> f64 {
        // Calculate weighted system health from all components
        let consciousness_weight = 0.3;
        let processing_weight = 0.25;
        let entropy_weight = 0.2;
        let atp_weight = 0.15;
        let completion_weight = 0.1;
        
        let consciousness_health = self.system_state.consciousness_level;
        let processing_health = self.system_state.processing_efficiency;
        let entropy_health = 1.0 - self.system_state.entropy_level.min(1.0);
        let atp_health = if self.system_state.total_atp_consumption < 1000.0 { 1.0 } else { 1000.0 / self.system_state.total_atp_consumption };
        let completion_health = self.system_state.categorical_completion_progress;
        
        (consciousness_health * consciousness_weight +
         processing_health * processing_weight +
         entropy_health * entropy_weight +
         atp_health * atp_weight +
         completion_health * completion_weight).min(1.0)
    }
    
    /// Get current system metrics
    pub fn get_system_metrics(&self) -> SystemMetrics {
        SystemMetrics {
            consciousness_level: self.system_state.consciousness_level,
            fire_recognition_strength: self.system_state.fire_recognition_strength,
            agency_detection_strength: self.system_state.agency_detection_strength,
            categorical_completion_progress: self.system_state.categorical_completion_progress,
            thermodynamic_necessity_understanding: self.system_state.thermodynamic_necessity_understanding,
            system_health: self.system_state.system_health,
            total_atp_consumption: self.system_state.total_atp_consumption,
            entropy_level: self.system_state.entropy_level,
            processing_efficiency: self.system_state.processing_efficiency,
            active_hierarchy_levels: self.system_state.active_hierarchy_levels.clone(),
            metabolic_mode: self.system_state.metabolic_mode.clone(),
        }
    }
    
    /// Run comprehensive system tests
    pub async fn run_system_tests(&mut self) -> AutobahnResult<SystemTestResults> {
        // Run statistical tests
        let statistical_results = self.testing_framework.run_comprehensive_tests().await?;
        
        // Run fire consciousness tests
        let underwater_test = self.fire_consciousness.test_underwater_fireplace_paradox().await?;
        
        // Test oscillatory patterns
        let test_input = vec![0.5, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1];
        let oscillatory_response = self.process_input(&test_input).await?;
        
        Ok(SystemTestResults {
            statistical_tests_passed: statistical_results.overall_success,
            fire_consciousness_tests_passed: !underwater_test.fire_logically_impossible || underwater_test.hardwired_override_active,
            oscillatory_processing_functional: oscillatory_response.rag_response.emergence_detected,
            categorical_predeterminism_functional: oscillatory_response.predeterminism_response.categorical_predeterminism_results.thermodynamic_necessity_demonstrated,
            overall_system_health: self.system_state.system_health,
        })
    }
}

/// Complete response from Autobahn system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnResponse {
    pub fire_consciousness_response: FireConsciousnessResponse,
    pub rag_response: rag::RAGResponse,
    pub predeterminism_response: tres_commas::engine::ConsciousOutput,
    pub system_state: AutobahnSystemState,
    pub pre_processing_health: f64,
    pub post_processing_health: f64,
    pub drift_detected: bool,
    pub invariants_valid: bool,
    pub processing_timestamp: chrono::DateTime<chrono::Utc>,
}

/// System metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub consciousness_level: f64,
    pub fire_recognition_strength: f64,
    pub agency_detection_strength: f64,
    pub categorical_completion_progress: f64,
    pub thermodynamic_necessity_understanding: f64,
    pub system_health: f64,
    pub total_atp_consumption: f64,
    pub entropy_level: f64,
    pub processing_efficiency: f64,
    pub active_hierarchy_levels: Vec<HierarchyLevel>,
    pub metabolic_mode: MetabolicMode,
}

/// Data characteristics analysis
#[derive(Debug)]
struct DataCharacteristics {
    high_variance: bool,
    extremal_values: bool,
    oscillatory: bool,
}

/// System test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemTestResults {
    pub statistical_tests_passed: bool,
    pub fire_consciousness_tests_passed: bool,
    pub oscillatory_processing_functional: bool,
    pub categorical_predeterminism_functional: bool,
    pub overall_system_health: f64,
}

/// Convenience function to create new Autobahn system with modern human parameters
pub async fn create_modern_human_system() -> AutobahnResult<AutobahnSystem> {
    AutobahnSystem::new(0.0).await // 0 MYA = modern human
}

/// Convenience function to create new Autobahn system with early human parameters
pub async fn create_early_human_system() -> AutobahnResult<AutobahnSystem> {
    AutobahnSystem::new(2.0).await // 2 MYA = early human fire adaptation
}

/// Convenience function to create new Autobahn system with pre-fire human parameters
pub async fn create_pre_fire_system() -> AutobahnResult<AutobahnSystem> {
    AutobahnSystem::new(3.0).await // 3 MYA = pre-fire adaptation
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_autobahn_system_creation() {
        let system = AutobahnSystem::new(1.0).await;
        assert!(system.is_ok());
        
        let system = system.unwrap();
        assert!(system.system_state.consciousness_level > 0.0);
        assert!(system.system_state.system_health > 0.0);
    }
    
    #[tokio::test]
    async fn test_basic_processing() {
        let mut system = AutobahnSystem::new(0.5).await.unwrap();
        let input = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        
        let response = system.process_input(&input).await;
        assert!(response.is_ok());
        
        let response = response.unwrap();
        assert!(response.system_state.consciousness_level >= 0.0);
        assert!(response.system_state.consciousness_level <= 1.0);
    }
    
    #[tokio::test]
    async fn test_fire_consciousness_integration() {
        let mut system = create_modern_human_system().await.unwrap();
        let fire_like_input = vec![0.6, 0.7, 0.5, 0.8, 0.6, 0.7]; // Fire-like pattern
        
        let response = system.process_input(&fire_like_input).await.unwrap();
        assert!(response.fire_consciousness_response.fire_recognition.fire_detected);
    }
    
    #[tokio::test]
    async fn test_system_health_calculation() {
        let system = AutobahnSystem::new(1.0).await.unwrap();
        let health = system.calculate_system_health();
        assert!(health >= 0.0 && health <= 1.0);
    }
    
    #[tokio::test]
    async fn test_comprehensive_system_tests() {
        let mut system = create_modern_human_system().await.unwrap();
        let test_results = system.run_system_tests().await;
        assert!(test_results.is_ok());
    }
} 