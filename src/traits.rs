//! Core traits for the Autobahn biological metabolism computer
//!
//! This module defines the fundamental interfaces that all components of the system implement,
//! providing clean abstractions for biological processing, energy management, and probabilistic reasoning.

use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use async_trait::async_trait;

/// Primary interface for metacognitive orchestration
///
/// This trait defines the core API that external packages use to interact with
/// the Autobahn biological metabolism computer.
#[async_trait]
pub trait MetacognitiveOrchestrator {
    /// Process information through the biological metabolism pipeline
    async fn process_information(&mut self, input: InformationInput) -> AutobahnResult<ProcessingResult>;
    
    /// Analyze uncertainty in content without full processing
    fn analyze_uncertainty(&self, content: &str) -> AutobahnResult<UncertaintyAnalysis>;
    
    /// Metabolize content through biological pathways
    async fn metabolize_content(&mut self, content: &str) -> AutobahnResult<MetabolismResult>;
    
    /// Validate understanding through comprehension testing
    fn validate_understanding(&self, content: &str) -> AutobahnResult<ValidationResult>;
    
    /// Test robustness through adversarial methods
    async fn test_robustness(&mut self, content: &str) -> AutobahnResult<RobustnessReport>;
    
    /// Get current energy state
    fn get_energy_state(&self) -> EnergyState;
    
    /// Regenerate ATP energy
    fn regenerate_atp(&mut self);
    
    /// Enter champagne phase for dream processing
    async fn enter_champagne_phase(&mut self) -> AutobahnResult<ChampagneResult>;
    
    /// Check if ready for processing
    fn is_ready(&self) -> bool;
    
    /// Get processing capabilities
    fn get_capabilities(&self) -> ProcessingCapabilities;
    
    // NEW: Quantum-oscillatory processing methods
    
    /// Measure entropy through oscillation endpoint analysis
    fn measure_entropy_endpoints(&self, content: &str) -> AutobahnResult<OscillationEndpointAnalysis>;
    
    /// Process semantic structures using four-sided triangle geometry
    async fn process_foursided_triangle(&mut self, semantic_structure: SemanticStructure) -> AutobahnResult<GeometricProcessingResult>;
    
    /// Maintain quantum coherence across membrane interfaces
    async fn maintain_quantum_coherence(&mut self) -> AutobahnResult<CoherenceState>;
    
    /// Analyze oscillation patterns for semantic meaning extraction
    fn analyze_oscillation_patterns(&self, content: &str) -> AutobahnResult<OscillationPatternAnalysis>;
    
    /// Actively control information entropy through oscillation manipulation
    async fn control_information_entropy(&mut self, target_entropy: f64, content: &str) -> AutobahnResult<EntropyControlResult>;
}

/// Specialized result type for metabolism operations
#[derive(Debug, Clone)]
pub struct MetabolismResult {
    pub pathway: ProcessingPathway,
    pub atp_yield: f64,
    pub byproducts: Vec<String>,
    pub efficiency: f64,
}

/// Result from champagne phase processing
#[derive(Debug, Clone)]
pub struct ChampagneResult {
    pub lactate_processed: f64,
    pub insights_gained: Vec<String>,
    pub optimization_improvements: f64,
    pub dream_duration_ms: u64,
}

/// Processing capabilities of the orchestrator
#[derive(Debug, Clone)]
pub struct ProcessingCapabilities {
    pub supports_probabilistic: bool,
    pub supports_adversarial: bool,
    pub supports_champagne: bool,
    pub max_atp: f64,
    pub available_modules: Vec<String>,
    pub processing_modes: Vec<ProcessingMode>,
}

/// Interface for biological processing modules
#[async_trait]
pub trait BiologicalModule {
    /// Module name identifier
    fn name(&self) -> &str;
    
    /// Process input through this biological module
    async fn process(&mut self, input: ModuleInput) -> AutobahnResult<ModuleOutput>;
    
    /// Calculate ATP cost for processing
    fn calculate_atp_cost(&self, input: &ModuleInput) -> f64;
    
    /// Check if module is ready for processing
    fn is_ready(&self) -> bool;
    
    /// Get module-specific capabilities
    fn capabilities(&self) -> ModuleCapabilities;
    
    /// Reset module state
    fn reset(&mut self);
}

/// Input for biological modules
#[derive(Debug, Clone)]
pub struct ModuleInput {
    pub content: String,
    pub context: ProcessingContext,
    pub energy_available: f64,
    pub confidence_required: f64,
}

/// Output from biological modules
#[derive(Debug, Clone)]
pub struct ModuleOutput {
    pub result: String,
    pub confidence: f64,
    pub atp_consumed: f64,
    pub byproducts: Vec<String>,
    pub metadata: ModuleMetadata,
}

/// Processing context for modules
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    pub layer: TresCommasLayer,
    pub previous_results: Vec<String>,
    pub time_pressure: f64,
    pub quality_requirements: QualityRequirements,
}

/// Quality requirements for processing
#[derive(Debug, Clone)]
pub struct QualityRequirements {
    pub min_confidence: f64,
    pub max_uncertainty: f64,
    pub robustness_required: bool,
    pub adversarial_testing: bool,
}

/// Layer in the Tres Commas Trinity Engine
#[derive(Debug, Clone, PartialEq)]
pub enum TresCommasLayer {
    Context,    // Glycolysis layer
    Reasoning,  // Krebs cycle layer
    Intuition,  // Electron transport layer
}

/// Module capabilities
#[derive(Debug, Clone)]
pub struct ModuleCapabilities {
    pub supports_async: bool,
    pub energy_efficiency: f64,
    pub processing_speed: f64,
    pub accuracy: f64,
    pub specialized_domains: Vec<String>,
}

/// Module metadata
#[derive(Debug, Clone)]
pub struct ModuleMetadata {
    pub processing_time_ms: u64,
    pub memory_used_mb: f64,
    pub cpu_usage_percent: f64,
    pub cache_hits: u32,
    pub cache_misses: u32,
}

/// Interface for ATP energy management
pub trait EnergyManager {
    /// Get current energy state
    fn get_energy_state(&self) -> &EnergyState;
    
    /// Consume ATP for operation
    fn consume_atp(&mut self, amount: f64) -> AutobahnResult<f64>;
    
    /// Regenerate ATP over time
    fn regenerate_atp(&mut self, time_elapsed_ms: u64);
    
    /// Check if operation can be afforded
    fn can_afford(&self, cost: f64) -> bool;
    
    /// Optimize energy usage
    fn optimize_energy_usage(&mut self);
    
    /// Get energy efficiency score
    fn get_efficiency(&self) -> f64;
}

/// Interface for probabilistic processing
#[async_trait]
pub trait ProbabilisticProcessor {
    /// Create a probabilistic point from content
    fn create_point(&self, content: &str, context: &str) -> AutobahnResult<Point>;
    
    /// Resolve a point through debate platform
    async fn resolve_point(&mut self, point: Point, affirmations: Vec<Evidence>, contentions: Vec<Evidence>) -> AutobahnResult<Resolution>;
    
    /// Analyze uncertainty in content
    fn analyze_uncertainty(&self, content: &str) -> AutobahnResult<UncertaintyAnalysis>;
    
    /// Propagate uncertainty through processing
    fn propagate_uncertainty(&self, input_uncertainty: f64, operation: &str) -> f64;
    
    /// Calculate confidence intervals
    fn calculate_confidence_intervals(&self, data: &[f64]) -> (f64, f64);
}

/// Interface for adversarial testing
#[async_trait]
pub trait AdversarialTester {
    /// Execute adversarial attack
    async fn execute_attack(&mut self, target: &str, attack_type: AttackType) -> AutobahnResult<AttackResult>;
    
    /// Test system robustness
    async fn test_robustness(&mut self, content: &str) -> AutobahnResult<RobustnessReport>;
    
    /// Perform perturbation analysis
    fn perform_perturbation_analysis(&self, content: &str) -> AutobahnResult<PerturbationAnalysis>;
    
    /// Detect vulnerabilities
    fn detect_vulnerabilities(&self, system_state: &SystemState) -> Vec<Vulnerability>;
    
    /// Generate attack recommendations
    fn recommend_attacks(&self, target_analysis: &str) -> Vec<AttackStrategy>;
}

/// Attack types for adversarial testing
#[derive(Debug, Clone)]
pub enum AttackType {
    ContradictionInjection,
    TemporalManipulation,
    SemanticSpoofing,
    ContextHijacking,
    PerturbationAttack,
    BeliefPoisoning,
}

/// Result of adversarial attack
#[derive(Debug, Clone)]
pub struct AttackResult {
    pub attack_type: AttackType,
    pub success: bool,
    pub confidence_change: f64,
    pub vulnerabilities_exposed: Vec<Vulnerability>,
    pub mitigation_suggestions: Vec<String>,
}

/// Attack strategy recommendation
#[derive(Debug, Clone)]
pub struct AttackStrategy {
    pub attack_type: AttackType,
    pub target_component: String,
    pub expected_success_rate: f64,
    pub required_resources: f64,
}

/// System state for vulnerability analysis
#[derive(Debug, Clone)]
pub struct SystemState {
    pub current_atp: f64,
    pub active_modules: Vec<String>,
    pub processing_load: f64,
    pub confidence_levels: Vec<f64>,
    pub recent_errors: Vec<String>,
}

/// Interface for validation and testing
pub trait Validator {
    /// Validate comprehension through testing
    fn validate_comprehension(&self, content: &str) -> AutobahnResult<ValidationResult>;
    
    /// Test context retention
    fn test_context_retention(&self, original: &str, processed: &str) -> AutobahnResult<f64>;
    
    /// Perform stability testing
    fn test_stability(&self, content: &str, perturbations: &[Perturbation]) -> AutobahnResult<f64>;
    
    /// Validate processing quality
    fn validate_quality(&self, input: &str, output: &ProcessingResult) -> AutobahnResult<QualityScore>;
}

/// Perturbation for testing
#[derive(Debug, Clone)]
pub struct Perturbation {
    pub perturbation_type: PerturbationType,
    pub strength: f64,
    pub target: String,
}

/// Quality score for validation
#[derive(Debug, Clone)]
pub struct QualityScore {
    pub overall_score: f64,
    pub accuracy: f64,
    pub completeness: f64,
    pub consistency: f64,
    pub robustness: f64,
}

/// Interface for champagne phase processing
#[async_trait]
pub trait ChampagneProcessor {
    /// Enter dream state
    async fn enter_dream_state(&mut self, user_status: UserStatus) -> AutobahnResult<DreamInitialization>;
    
    /// Process lactate buffer
    async fn process_lactate_buffer(&mut self) -> AutobahnResult<ChampagneResult>;
    
    /// Complete comprehension processing
    async fn complete_comprehension_processing(&mut self, partial: PartialComprehension) -> AutobahnResult<CompletedInsight>;
    
    /// Auto-correct turbulance scripts
    async fn auto_correct_scripts(&mut self, scripts: Vec<String>) -> AutobahnResult<Vec<String>>;
    
    /// Check if champagne phase is available
    fn is_champagne_available(&self, user_status: UserStatus) -> bool;
}

/// User status for champagne phase
#[derive(Debug, Clone, PartialEq)]
pub enum UserStatus {
    Active,
    Idle,
    Away,
    Sleeping,
}

/// Dream initialization result
#[derive(Debug, Clone)]
pub struct DreamInitialization {
    pub success: bool,
    pub dream_mode: DreamMode,
    pub atp_allocated: f64,
    pub estimated_duration_ms: u64,
}

/// Dream processing modes
#[derive(Debug, Clone)]
pub enum DreamMode {
    Light,
    Deep,
    REM,
    Recovery,
}

/// Partial comprehension for completion
#[derive(Debug, Clone)]
pub struct PartialComprehension {
    pub original_text: String,
    pub failed_tests: Vec<String>,
    pub comprehension_score: f64,
    pub processing_attempts: u32,
}

/// Completed insight from champagne processing
#[derive(Debug, Clone)]
pub struct CompletedInsight {
    pub original_partial: PartialComprehension,
    pub completion_method: CompletionMethod,
    pub final_comprehension_score: f64,
    pub insights_gained: Vec<String>,
    pub patterns_discovered: Vec<String>,
}

/// Methods for completing insights
#[derive(Debug, Clone)]
pub enum CompletionMethod {
    DeepProcessing,
    AlternativeStrategy,
    CrossDomainAnalysis,
    TemporalReconsideration,
}

/// Interface for temporal processing and decay
pub trait TemporalProcessor {
    /// Apply temporal decay to evidence
    fn apply_decay(&self, evidence: &Evidence, time_elapsed: f64) -> f64;
    
    /// Calculate evidence strength over time
    fn calculate_temporal_strength(&self, evidence: &Evidence) -> f64;
    
    /// Update temporal state
    fn update_temporal_state(&mut self, new_evidence: Vec<Evidence>);
    
    /// Predict future decay
    fn predict_decay(&self, evidence: &Evidence, future_time: f64) -> f64;
}

/// Default implementations and helpers
impl Default for ProcessingCapabilities {
    fn default() -> Self {
        Self {
            supports_probabilistic: true,
            supports_adversarial: true,
            supports_champagne: true,
            max_atp: 1000.0,
            available_modules: vec![
                "mzekezeke".to_string(),
                "diggiden".to_string(),
                "hatata".to_string(),
                "spectacular".to_string(),
                "nicotine".to_string(),
                "clothesline".to_string(),
                "zengeza".to_string(),
                "diadochi".to_string(),
            ],
            processing_modes: vec![
                ProcessingMode::Deterministic,
                ProcessingMode::Probabilistic,
                ProcessingMode::Adversarial,
                ProcessingMode::Dream,
            ],
        }
    }
}

impl Default for QualityRequirements {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            max_uncertainty: 0.3,
            robustness_required: true,
            adversarial_testing: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processing_capabilities_default() {
        let caps = ProcessingCapabilities::default();
        assert!(caps.supports_probabilistic);
        assert!(caps.supports_adversarial);
        assert!(caps.supports_champagne);
        assert_eq!(caps.max_atp, 1000.0);
        assert_eq!(caps.available_modules.len(), 8);
    }

    #[test]
    fn test_quality_requirements_default() {
        let req = QualityRequirements::default();
        assert_eq!(req.min_confidence, 0.7);
        assert_eq!(req.max_uncertainty, 0.3);
        assert!(req.robustness_required);
        assert!(!req.adversarial_testing);
    }

    #[test]
    fn test_tres_commas_layer_equality() {
        assert_eq!(TresCommasLayer::Context, TresCommasLayer::Context);
        assert_ne!(TresCommasLayer::Context, TresCommasLayer::Reasoning);
    }
} 