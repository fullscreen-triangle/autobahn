//! Autobahn: Composable Quantum Processor Units
//! 
//! Autobahn provides composable quantum processor units that implement biological Maxwell's demons
//! and fire-consciousness quantum frameworks. These processors can be combined with other units
//! (nebuchadnezzar for intracellular processes, bene-gesserit for membranes) to construct
//! realistic neurons in the imhotep system.
//! 
//! Each processor type (Context, Reasoning, Intuition) can be independently instantiated,
//! configured, and orchestrated. The only constraint should be the number of processors
//! you can manage and orchestrate.

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

// New hardware and optical systems
#[cfg(feature = "hardware-sync")]
pub mod hardware;
#[cfg(feature = "optical-processing")]
pub mod optical;
#[cfg(feature = "environmental-photosynthesis")]
pub mod photosynthesis;

// Development and research
pub mod research_dev;
pub mod plugins;

// Turbulance language integration
pub mod turbulance;

// Re-exports for composable processors
pub use tres_commas::{
    ContextProcessor,
    ReasoningProcessor, 
    IntuitionProcessor,
    TrinityEngine,
    ProcessorConfig,
    ProcessorInput,
    ProcessorOutput,
    ProcessorMetrics,
};

// Core framework re-exports
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
pub use configuration::ConfigurationManager;

// Hardware and optical system re-exports
#[cfg(feature = "hardware-sync")]
pub use hardware::{
    HardwareOscillationCapture, 
    CoherenceSync, 
    FrequencyDomain, 
    OscillationData, 
    SynchronizationState
};
#[cfg(feature = "optical-processing")]
pub use optical::{
    DigitalFireProcessor, 
    OpticalCoherence, 
    LightSource, 
    LightData, 
    DigitalFireCircle
};
#[cfg(feature = "environmental-photosynthesis")]
pub use photosynthesis::{
    EnvironmentalPhotosynthesis, 
    ColorMetabolism, 
    VisualATPManager, 
    AgencyIllusionEngine, 
    ChaosSubstrateGenerator
};

use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Composable Quantum Processor Unit
/// 
/// This is the main composable unit that can be instantiated with different
/// processor types and combined with other units to build complex neural systems.
#[derive(Debug)]
pub struct QuantumProcessorUnit {
    /// Processor type (Context, Reasoning, or Intuition)
    pub processor_type: ProcessorType,
    /// Fire consciousness engine for quantum-biological processing
    pub fire_consciousness: FireConsciousnessEngine,
    /// Oscillatory bio-metabolic RAG system
    pub rag_system: OscillatoryBioMetabolicRAG,
    /// Specialized processor for this unit's type
    pub specialized_processor: SpecializedProcessor,
    /// System monitor for reliability
    pub monitor: SystemMonitor,
    /// Current processor state
    pub processor_state: ProcessorState,
    /// Configuration
    pub config: ProcessorConfig,
}

/// Types of quantum processors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessorType {
    /// Context processing (Glycolysis layer)
    Context,
    /// Reasoning processing (Krebs cycle layer)  
    Reasoning,
    /// Intuition processing (Electron transport layer)
    Intuition,
}

/// Specialized processor implementations
#[derive(Debug)]
pub enum SpecializedProcessor {
    Context(ContextProcessor),
    Reasoning(ReasoningProcessor),
    Intuition(IntuitionProcessor),
}

/// Processor state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorState {
    /// Overall processor health (0.0 to 1.0)
    pub processor_health: f64,
    /// Consciousness level (0.0 to 1.0)
    pub consciousness_level: f64,
    /// Fire recognition strength (0.0 to 1.0)
    pub fire_recognition_strength: f64,
    /// Processing efficiency
    pub processing_efficiency: f64,
    /// Total ATP consumption
    pub total_atp_consumption: f64,
    /// System entropy level
    pub entropy_level: f64,
    /// Active hierarchy levels
    pub active_hierarchy_levels: Vec<HierarchyLevel>,
    /// Current metabolic mode
    pub metabolic_mode: MetabolicMode,
    /// Last processing timestamp
    pub last_processing: chrono::DateTime<chrono::Utc>,
}

impl Default for ProcessorState {
    fn default() -> Self {
        Self {
            processor_health: 1.0,
            consciousness_level: 0.5,
            fire_recognition_strength: 0.5,
            processing_efficiency: 0.8,
            total_atp_consumption: 0.0,
            entropy_level: 0.1,
            active_hierarchy_levels: vec![
                HierarchyLevel::Cognitive,
                HierarchyLevel::Cellular,
                HierarchyLevel::Molecular,
            ],
            metabolic_mode: MetabolicMode::Balanced,
            last_processing: chrono::Utc::now(),
        }
    }
}

impl QuantumProcessorUnit {
    /// Create new quantum processor unit of specified type
    pub async fn new(processor_type: ProcessorType, config: ProcessorConfig) -> AutobahnResult<Self> {
        // Initialize fire consciousness engine
        let fire_consciousness = FireConsciousnessEngine::new(config.evolutionary_time_mya)?;
        
        // Initialize oscillatory bio-metabolic RAG system
        let rag_system = OscillatoryBioMetabolicRAG::new().await?;
        
        // Initialize specialized processor based on type
        let specialized_processor = match processor_type {
            ProcessorType::Context => {
                SpecializedProcessor::Context(ContextProcessor::new(config.clone()).await?)
            },
            ProcessorType::Reasoning => {
                SpecializedProcessor::Reasoning(ReasoningProcessor::new(config.clone()).await?)
            },
            ProcessorType::Intuition => {
                SpecializedProcessor::Intuition(IntuitionProcessor::new(config.clone()).await?)
            },
        };
        
        // Initialize system monitor
        let monitor = SystemMonitor::new();
        
        Ok(Self {
            processor_type,
            fire_consciousness,
            rag_system,
            specialized_processor,
            monitor,
            processor_state: ProcessorState::default(),
            config,
        })
    }
    
    /// Process input through this quantum processor unit
    pub async fn process(&mut self, input: ProcessorInput) -> AutobahnResult<ProcessorOutput> {
        let start_time = std::time::Instant::now();
        
        // Pre-processing health check
        let pre_processing_health = self.calculate_processor_health();
        
        // Fire consciousness processing
        let fire_response = self.fire_consciousness
            .process_input(&input.raw_data).await?;
        
        // RAG system processing
        let rag_response = self.rag_system
            .process_query(&input.content).await?;
        
        // Specialized processor processing
        let specialized_response = match &mut self.specialized_processor {
            SpecializedProcessor::Context(processor) => processor.process(input.clone()).await?,
            SpecializedProcessor::Reasoning(processor) => processor.process(input.clone()).await?,
            SpecializedProcessor::Intuition(processor) => processor.process(input.clone()).await?,
        };
        
        // Update processor state
        self.update_processor_state(&fire_response, &rag_response, &specialized_response).await?;
        
        // Post-processing health check
        let post_processing_health = self.calculate_processor_health();
        
        let processing_time = start_time.elapsed();
        
        Ok(ProcessorOutput {
            processor_type: self.processor_type.clone(),
            fire_consciousness_response: fire_response,
            rag_response,
            specialized_response,
            processor_state: self.processor_state.clone(),
            pre_processing_health,
            post_processing_health,
            processing_time_ms: processing_time.as_millis() as u64,
            processing_timestamp: chrono::Utc::now(),
        })
    }
    
    /// Get processor metrics for monitoring
    pub fn get_metrics(&self) -> ProcessorMetrics {
        ProcessorMetrics {
            processor_type: self.processor_type.clone(),
            processor_health: self.processor_state.processor_health,
            consciousness_level: self.processor_state.consciousness_level,
            fire_recognition_strength: self.processor_state.fire_recognition_strength,
            processing_efficiency: self.processor_state.processing_efficiency,
            total_atp_consumption: self.processor_state.total_atp_consumption,
            entropy_level: self.processor_state.entropy_level,
            active_hierarchy_levels: self.processor_state.active_hierarchy_levels.clone(),
            metabolic_mode: self.processor_state.metabolic_mode.clone(),
            last_processing: self.processor_state.last_processing,
        }
    }
    
    /// Configure processor parameters
    pub fn configure(&mut self, config: ProcessorConfig) -> AutobahnResult<()> {
        self.config = config;
        Ok(())
    }
    
    /// Check if processor is ready for processing
    pub fn is_ready(&self) -> bool {
        self.processor_state.processor_health > 0.1 && 
        self.processor_state.total_atp_consumption < self.config.max_atp_consumption
    }
    
    /// Reset processor to initial state
    pub fn reset(&mut self) -> AutobahnResult<()> {
        self.processor_state = ProcessorState::default();
        Ok(())
    }
    
    fn calculate_processor_health(&self) -> f64 {
        let health_factors = vec![
            self.processor_state.consciousness_level,
            self.processor_state.fire_recognition_strength,
            self.processor_state.processing_efficiency,
            1.0 - (self.processor_state.entropy_level / 2.0).min(1.0),
        ];
        
        health_factors.iter().sum::<f64>() / health_factors.len() as f64
    }
    
    async fn update_processor_state(
        &mut self,
        fire_response: &FireConsciousnessResponse,
        rag_response: &rag::RAGResponse,
        specialized_response: &tres_commas::SpecializedProcessorResponse
    ) -> AutobahnResult<()> {
        // Update consciousness metrics
        self.processor_state.consciousness_level = 
            (self.processor_state.consciousness_level + fire_response.consciousness_level) / 2.0;
        
        self.processor_state.fire_recognition_strength = fire_response.fire_recognition.recognition_strength;
        
        // Update ATP consumption
        self.processor_state.total_atp_consumption += 
            rag_response.atp_consumption + specialized_response.atp_consumed;
        
        // Update processing efficiency
        self.processor_state.processing_efficiency = 
            (rag_response.system_state.processing_efficiency + 0.8) / 2.0; // Default efficiency for fire response
        
        // Update timestamp
        self.processor_state.last_processing = chrono::Utc::now();
        
        // Recalculate health
        self.processor_state.processor_health = self.calculate_processor_health();
        
        Ok(())
    }
}

/// Multi-processor orchestrator for combining multiple quantum processor units
#[derive(Debug)]
pub struct MultiProcessorOrchestrator {
    /// Active processor units
    pub processors: Vec<QuantumProcessorUnit>,
    /// Orchestration strategy
    pub orchestration_strategy: OrchestrationStrategy,
    /// Global state
    pub global_state: GlobalOrchestratorState,
}

/// Orchestration strategies for multi-processor systems
#[derive(Debug, Clone)]
pub enum OrchestrationStrategy {
    /// Sequential processing through all processors
    Sequential,
    /// Parallel processing with result aggregation
    Parallel,
    /// Hierarchical processing with priority levels
    Hierarchical { priority_order: Vec<ProcessorType> },
    /// Adaptive orchestration based on input characteristics
    Adaptive,
    /// Custom orchestration with user-defined logic
    Custom { orchestration_logic: String },
}

/// Global orchestrator state
#[derive(Debug, Clone)]
pub struct GlobalOrchestratorState {
    /// Total processors managed
    pub total_processors: usize,
    /// Active processors
    pub active_processors: usize,
    /// Overall system health
    pub system_health: f64,
    /// Total ATP consumption across all processors
    pub total_atp_consumption: f64,
    /// Average processing efficiency
    pub average_efficiency: f64,
    /// Last orchestration timestamp
    pub last_orchestration: chrono::DateTime<chrono::Utc>,
}

impl MultiProcessorOrchestrator {
    /// Create new multi-processor orchestrator
    pub fn new(orchestration_strategy: OrchestrationStrategy) -> Self {
        Self {
            processors: Vec::new(),
            orchestration_strategy,
            global_state: GlobalOrchestratorState {
                total_processors: 0,
                active_processors: 0,
                system_health: 1.0,
                total_atp_consumption: 0.0,
                average_efficiency: 1.0,
                last_orchestration: chrono::Utc::now(),
            },
        }
    }
    
    /// Add processor unit to orchestrator
    pub fn add_processor(&mut self, processor: QuantumProcessorUnit) {
        self.processors.push(processor);
        self.global_state.total_processors = self.processors.len();
        self.update_global_state();
    }
    
    /// Remove processor unit from orchestrator
    pub fn remove_processor(&mut self, index: usize) -> Option<QuantumProcessorUnit> {
        if index < self.processors.len() {
            let processor = self.processors.remove(index);
            self.global_state.total_processors = self.processors.len();
            self.update_global_state();
            Some(processor)
        } else {
            None
        }
    }
    
    /// Process input through orchestrated processors
    pub async fn orchestrate_processing(&mut self, input: ProcessorInput) -> AutobahnResult<Vec<ProcessorOutput>> {
        match &self.orchestration_strategy {
            OrchestrationStrategy::Sequential => self.process_sequential(input).await,
            OrchestrationStrategy::Parallel => self.process_parallel(input).await,
            OrchestrationStrategy::Hierarchical { priority_order } => 
                self.process_hierarchical(input, priority_order.clone()).await,
            OrchestrationStrategy::Adaptive => self.process_adaptive(input).await,
            OrchestrationStrategy::Custom { orchestration_logic } => 
                self.process_custom(input, orchestration_logic.clone()).await,
        }
    }
    
    async fn process_sequential(&mut self, input: ProcessorInput) -> AutobahnResult<Vec<ProcessorOutput>> {
        let mut results = Vec::new();
        let mut current_input = input;
        
        for processor in &mut self.processors {
            if processor.is_ready() {
                let output = processor.process(current_input.clone()).await?;
                // Chain outputs for sequential processing
                current_input.content = output.specialized_response.content.clone();
                results.push(output);
            }
        }
        
        self.update_global_state();
        Ok(results)
    }
    
    async fn process_parallel(&mut self, input: ProcessorInput) -> AutobahnResult<Vec<ProcessorOutput>> {
        let mut results = Vec::new();
        
        // Process all ready processors in parallel
        let mut futures = Vec::new();
        for processor in &mut self.processors {
            if processor.is_ready() {
                futures.push(processor.process(input.clone()));
            }
        }
        
        // Wait for all to complete
        for future in futures {
            results.push(future.await?);
        }
        
        self.update_global_state();
        Ok(results)
    }
    
    async fn process_hierarchical(&mut self, input: ProcessorInput, priority_order: Vec<ProcessorType>) -> AutobahnResult<Vec<ProcessorOutput>> {
        let mut results = Vec::new();
        
        // Process in priority order
        for processor_type in priority_order {
            for processor in &mut self.processors {
                if processor.processor_type == processor_type && processor.is_ready() {
                    let output = processor.process(input.clone()).await?;
                    results.push(output);
                }
            }
        }
        
        self.update_global_state();
        Ok(results)
    }
    
    async fn process_adaptive(&mut self, input: ProcessorInput) -> AutobahnResult<Vec<ProcessorOutput>> {
        // Analyze input characteristics to determine best strategy
        let strategy = if input.raw_data.len() > 1000 {
            OrchestrationStrategy::Parallel
        } else if input.priority > 0.8 {
            OrchestrationStrategy::Hierarchical { 
                priority_order: vec![ProcessorType::Context, ProcessorType::Reasoning, ProcessorType::Intuition] 
            }
        } else {
            OrchestrationStrategy::Sequential
        };
        
        // Temporarily switch strategy
        let original_strategy = self.orchestration_strategy.clone();
        self.orchestration_strategy = strategy;
        
        let results = self.orchestrate_processing(input).await?;
        
        // Restore original strategy
        self.orchestration_strategy = original_strategy;
        
        Ok(results)
    }
    
    async fn process_custom(&mut self, input: ProcessorInput, _orchestration_logic: String) -> AutobahnResult<Vec<ProcessorOutput>> {
        // For now, default to sequential processing
        // In the future, this could parse and execute custom orchestration logic
        self.process_sequential(input).await
    }
    
    fn update_global_state(&mut self) {
        self.global_state.active_processors = self.processors.iter()
            .filter(|p| p.is_ready())
            .count();
        
        self.global_state.system_health = if self.processors.is_empty() {
            0.0
        } else {
            self.processors.iter()
                .map(|p| p.processor_state.processor_health)
                .sum::<f64>() / self.processors.len() as f64
        };
        
        self.global_state.total_atp_consumption = self.processors.iter()
            .map(|p| p.processor_state.total_atp_consumption)
            .sum();
        
        self.global_state.average_efficiency = if self.processors.is_empty() {
            0.0
        } else {
            self.processors.iter()
                .map(|p| p.processor_state.processing_efficiency)
                .sum::<f64>() / self.processors.len() as f64
        };
        
        self.global_state.last_orchestration = chrono::Utc::now();
    }
    
    /// Get global orchestrator metrics
    pub fn get_global_metrics(&self) -> GlobalOrchestratorState {
        self.global_state.clone()
    }
    
    /// Get individual processor metrics
    pub fn get_processor_metrics(&self) -> Vec<ProcessorMetrics> {
        self.processors.iter()
            .map(|p| p.get_metrics())
            .collect()
    }
}

// Convenience functions for creating different processor types
pub async fn create_context_processor(config: ProcessorConfig) -> AutobahnResult<QuantumProcessorUnit> {
    QuantumProcessorUnit::new(ProcessorType::Context, config).await
}

pub async fn create_reasoning_processor(config: ProcessorConfig) -> AutobahnResult<QuantumProcessorUnit> {
    QuantumProcessorUnit::new(ProcessorType::Reasoning, config).await
}

pub async fn create_intuition_processor(config: ProcessorConfig) -> AutobahnResult<QuantumProcessorUnit> {
    QuantumProcessorUnit::new(ProcessorType::Intuition, config).await
}

/// Create a balanced multi-processor system with all three types
pub async fn create_balanced_system(config: ProcessorConfig) -> AutobahnResult<MultiProcessorOrchestrator> {
    let mut orchestrator = MultiProcessorOrchestrator::new(OrchestrationStrategy::Sequential);
    
    orchestrator.add_processor(create_context_processor(config.clone()).await?);
    orchestrator.add_processor(create_reasoning_processor(config.clone()).await?);
    orchestrator.add_processor(create_intuition_processor(config).await?);
    
    Ok(orchestrator)
}

/// Create a custom multi-processor system
pub async fn create_custom_system(
    processor_configs: Vec<(ProcessorType, ProcessorConfig)>,
    orchestration_strategy: OrchestrationStrategy
) -> AutobahnResult<MultiProcessorOrchestrator> {
    let mut orchestrator = MultiProcessorOrchestrator::new(orchestration_strategy);
    
    for (processor_type, config) in processor_configs {
        let processor = QuantumProcessorUnit::new(processor_type, config).await?;
        orchestrator.add_processor(processor);
    }
    
    Ok(orchestrator)
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn init() -> AutobahnResult<()> {
    env_logger::init();
    log::info!("Autobahn quantum processor system initialized");
    Ok(())
}

/// Get system capabilities
pub fn get_capabilities() -> SystemCapabilities {
    SystemCapabilities {
        available_modules: vec![
            "Context Processor".to_string(),
            "Reasoning Processor".to_string(),
            "Intuition Processor".to_string(),
            "Fire Consciousness Engine".to_string(),
            "Oscillatory Bio-Metabolic RAG".to_string(),
            "Multi-Processor Orchestrator".to_string(),
        ],
        processing_modes: vec![
            "Sequential".to_string(),
            "Parallel".to_string(),
            "Hierarchical".to_string(),
            "Adaptive".to_string(),
            "Custom".to_string(),
        ],
        max_processors_supported: 1000, // The only constraint should be orchestration capacity
    }
}

#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    pub available_modules: Vec<String>,
    pub processing_modes: Vec<String>,
    pub max_processors_supported: usize,
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