//! Core types for the Autobahn biological metabolism computer
//!
//! This module defines all fundamental data structures used throughout the system,
//! including probabilistic types, biological processing states, and energy management.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Information input types for the biological processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InformationInput {
    /// Raw text for processing
    Text(String),
    /// Structured data with metadata
    StructuredData {
        content: String,
        metadata: HashMap<String, String>,
        context: Option<String>,
    },
    /// Genetic sequence data
    GeneticSequence {
        sequence: String,
        sequence_type: SequenceType,
        organism: Option<String>,
    },
    /// Scientific document with specialized processing
    ScientificDocument {
        title: String,
        content: String,
        authors: Vec<String>,
        domain: String,
    },
    /// Multi-modal input combining different types
    MultiModal {
        primary_content: String,
        secondary_data: Vec<InformationInput>,
        integration_strategy: IntegrationStrategy,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SequenceType {
    DNA,
    RNA,
    Protein,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationStrategy {
    Sequential,
    Parallel,
    Hierarchical,
    Contextual,
}

/// Result of information processing through the biological metabolism system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// Primary output from processing
    pub output: ProcessingOutput,
    /// Confidence in the result (0.0 to 1.0)
    pub confidence: f64,
    /// ATP consumed during processing
    pub atp_consumed: f64,
    /// Processing pathway taken
    pub pathway: ProcessingPathway,
    /// Metabolic efficiency
    pub efficiency: f64,
    /// Uncertainty analysis
    pub uncertainty: UncertaintyAnalysis,
    /// Validation results
    pub validation: ValidationResult,
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

/// Output types from biological processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingOutput {
    /// Simple text output
    Text(String),
    /// Probabilistic interpretation with multiple possibilities
    ProbabilisticInterpretation {
        interpretations: Vec<WeightedInterpretation>,
        consensus: Option<String>,
    },
    /// Structured analysis result
    Analysis {
        summary: String,
        details: HashMap<String, String>,
        confidence_intervals: HashMap<String, (f64, f64)>,
    },
    /// Decision recommendation with reasoning
    Decision {
        recommendation: String,
        reasoning: Vec<String>,
        alternatives: Vec<Alternative>,
        risk_assessment: RiskAssessment,
    },
    /// Pattern recognition result
    Patterns {
        detected_patterns: Vec<Pattern>,
        significance_scores: HashMap<String, f64>,
        novel_insights: Vec<String>,
    },
}

/// Weighted interpretation with probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedInterpretation {
    pub content: String,
    pub probability: f64,
    pub supporting_evidence: Vec<String>,
    pub confidence_interval: (f64, f64),
}

/// Alternative decision option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alternative {
    pub description: String,
    pub probability: f64,
    pub pros: Vec<String>,
    pub cons: Vec<String>,
}

/// Risk assessment for decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk: f64,
    pub risk_factors: Vec<RiskFactor>,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor: String,
    pub impact: f64,
    pub probability: f64,
}

/// Detected pattern with significance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub name: String,
    pub description: String,
    pub confidence: f64,
    pub occurrences: Vec<PatternOccurrence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternOccurrence {
    pub location: String,
    pub context: String,
    pub strength: f64,
}

/// Processing pathway through the biological system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingPathway {
    /// Standard aerobic processing through all layers
    Aerobic {
        glycolysis_result: GlycolysisResult,
        krebs_result: KrebsResult,
        electron_transport_result: ElectronTransportResult,
    },
    /// Emergency anaerobic processing
    Anaerobic {
        partial_processing: PartialProcessing,
        lactate_accumulated: f64,
    },
    /// Dream processing during Champagne phase
    ChampagnePhase {
        dream_insights: Vec<String>,
        lactate_processed: f64,
        optimization_gained: f64,
    },
    /// Hybrid processing switching between modes
    Hybrid {
        mode_switches: Vec<ModeSwitch>,
        final_mode: ProcessingMode,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlycolysisResult {
    pub atp_net: f64,
    pub processing_time_ms: u64,
    pub comprehension_validated: bool,
    pub context_preserved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrebsResult {
    pub atp_produced: f64,
    pub nadh_produced: f64,
    pub fadh2_produced: f64,
    pub cycles_completed: u32,
    pub evidence_processed: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronTransportResult {
    pub atp_synthesized: f64,
    pub truth_synthesis_confidence: f64,
    pub metacognitive_alignment: f64,
    pub paradigm_insights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialProcessing {
    pub completed_steps: Vec<String>,
    pub incomplete_steps: Vec<String>,
    pub partial_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeSwitch {
    pub from_mode: ProcessingMode,
    pub to_mode: ProcessingMode,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

/// Processing modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingMode {
    Deterministic,
    Probabilistic,
    Adversarial,
    Dream,
    Emergency,
}

/// Uncertainty analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyAnalysis {
    /// Overall uncertainty score (0.0 = certain, 1.0 = completely uncertain)
    pub uncertainty_score: f64,
    /// Sources of uncertainty
    pub uncertainty_sources: Vec<UncertaintySource>,
    /// Confidence intervals for key metrics
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Epistemic vs aleatoric uncertainty breakdown
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
    /// Temporal decay factors
    pub temporal_decay: TemporalDecay,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintySource {
    pub source: String,
    pub contribution: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDecay {
    pub decay_function: DecayFunction,
    pub half_life: f64,
    pub current_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunction {
    Exponential { lambda: f64 },
    Power { alpha: f64 },
    Logarithmic { base: f64 },
    Weibull { shape: f64, scale: f64 },
}

/// Validation result from various testing mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall validation passed
    pub passed: bool,
    /// Individual test results
    pub tests: Vec<ValidationTest>,
    /// Perturbation analysis result
    pub perturbation_analysis: Option<PerturbationAnalysis>,
    /// Adversarial testing result
    pub adversarial_testing: Option<AdversarialTestResult>,
    /// Robustness score (0.0 to 1.0)
    pub robustness_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationTest {
    pub test_name: String,
    pub passed: bool,
    pub score: f64,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationAnalysis {
    pub stability_score: f64,
    pub perturbation_tests: Vec<PerturbationTest>,
    pub sensitivity_profile: SensitivityProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationTest {
    pub test_type: PerturbationType,
    pub original_confidence: f64,
    pub perturbed_confidence: f64,
    pub stability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerturbationType {
    WordRemoval,
    PositionalRearrangement,
    SynonymSubstitution,
    NegationTesting,
    NoiseAddition,
    GrammaticalVariation,
    PunctuationChanges,
    CaseVariation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityProfile {
    pub content_word_sensitivity: f64,
    pub function_word_sensitivity: f64,
    pub order_sensitivity: f64,
    pub semantic_sensitivity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialTestResult {
    pub attacks_attempted: u32,
    pub attacks_successful: u32,
    pub vulnerability_score: f64,
    pub detected_vulnerabilities: Vec<Vulnerability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    pub vulnerability_type: String,
    pub severity: f64,
    pub description: String,
    pub mitigation: Option<String>,
}

/// Processing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub processing_id: Uuid,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub processing_duration_ms: u64,
    pub modules_used: Vec<String>,
    pub atp_efficiency: f64,
    pub memory_usage: MemoryUsage,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_tokens_per_second: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
}

/// ATP (energy) management types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyState {
    pub current_atp: f64,
    pub max_atp: f64,
    pub regeneration_rate: f64,
    pub efficiency_factor: f64,
    pub last_regeneration: DateTime<Utc>,
}

impl EnergyState {
    pub fn new(max_atp: f64) -> Self {
        Self {
            current_atp: max_atp,
            max_atp,
            regeneration_rate: max_atp * 0.1, // 10% per time unit
            efficiency_factor: 1.0,
            last_regeneration: Utc::now(),
        }
    }

    pub fn energy_percentage(&self) -> f64 {
        (self.current_atp / self.max_atp) * 100.0
    }

    pub fn can_afford(&self, cost: f64) -> bool {
        self.current_atp >= cost
    }
}

/// Robustness testing report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessReport {
    pub overall_robustness: f64,
    pub vulnerability_assessment: VulnerabilityAssessment,
    pub recommendations: Vec<String>,
    pub attack_resistance: AttackResistance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityAssessment {
    pub critical_vulnerabilities: u32,
    pub moderate_vulnerabilities: u32,
    pub low_vulnerabilities: u32,
    pub risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackResistance {
    pub resistance_score: f64,
    pub tested_attack_types: Vec<String>,
    pub successful_defenses: u32,
    pub failed_defenses: u32,
}

/// Points and Resolutions framework types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub id: Uuid,
    pub content: String,
    pub certainty: f64,
    pub context: String,
    pub temporal_state: DateTime<Utc>,
    pub evidence_strength: f64,
    pub positional_metadata: Option<PositionalMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionalMetadata {
    pub position_weights: Vec<f64>,
    pub semantic_roles: Vec<String>,
    pub order_dependencies: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    pub point: Point,
    pub affirmations: Vec<Evidence>,
    pub contentions: Vec<Evidence>,
    pub resolution_strategy: ResolutionStrategy,
    pub consensus_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub content: String,
    pub strength: f64,
    pub source: String,
    pub credibility: f64,
    pub temporal_validity: TemporalDecay,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    Bayesian,
    MaximumLikelihood,
    Conservative,
    Exploratory,
    Consensus,
}

/// NEW: Quantum-Oscillatory Processing Types

/// Analysis of oscillation endpoints for entropy measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationEndpointAnalysis {
    /// Statistical distribution of oscillation termination points
    pub endpoint_distribution: Vec<EndpointFrequency>,
    /// Calculated entropy from endpoint statistics
    pub measured_entropy: f64,
    /// Oscillation frequency patterns
    pub frequency_patterns: Vec<FrequencyPattern>,
    /// Coherence score of oscillations
    pub coherence_score: f64,
    /// Predicted future endpoints
    pub predicted_endpoints: Vec<PredictedEndpoint>,
    /// Processing confidence
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointFrequency {
    pub endpoint_value: f64,
    pub frequency: u64,
    pub probability: f64,
    pub temporal_context: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyPattern {
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub semantic_meaning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedEndpoint {
    pub predicted_value: f64,
    pub probability: f64,
    pub time_to_occurrence: f64,
}

/// Four-sided triangle semantic structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStructure {
    /// Primary semantic vertices (A, B, C)
    pub primary_vertices: [SemanticVertex; 3],
    /// Temporal vertex (D) representing evolution
    pub temporal_vertex: TemporalVertex,
    /// Non-Euclidean properties
    pub geometric_properties: NonEuclideanProperties,
    /// Quantum superposition states
    pub quantum_states: Vec<QuantumSemanticState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticVertex {
    pub id: String,
    pub semantic_content: String,
    pub certainty: f64,
    pub connections: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalVertex {
    pub temporal_content: String,
    pub evolution_rate: f64,
    pub temporal_dependencies: Vec<String>,
    pub future_projections: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonEuclideanProperties {
    pub angular_sum: f64, // > 180 degrees for non-Euclidean
    pub curvature: f64,
    pub topological_genus: i32,
    pub folding_dimensions: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSemanticState {
    pub state_id: String,
    pub superposition_components: Vec<SuperpositionComponent>,
    pub coherence_time: f64,
    pub decoherence_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperpositionComponent {
    pub semantic_content: String,
    pub amplitude: f64,
    pub phase: f64,
}

/// Result of four-sided triangle processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricProcessingResult {
    /// Processed semantic structure
    pub processed_structure: SemanticStructure,
    /// Information compression achieved
    pub compression_ratio: f64,
    /// Paradox resolution success
    pub paradox_resolved: bool,
    /// Energy cost for processing
    pub atp_consumed: f64,
    /// Processing insights
    pub insights: Vec<String>,
    /// Temporal evolution predictions
    pub temporal_predictions: Vec<TemporalPrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPrediction {
    pub timeframe: f64,
    pub predicted_state: String,
    pub confidence: f64,
}

/// Quantum coherence state management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceState {
    /// Current coherence level (0.0 to 1.0)
    pub coherence_level: f64,
    /// Decoherence rate
    pub decoherence_rate: f64,
    /// Time until critical decoherence
    pub time_to_decoherence: f64,
    /// Active quantum states
    pub active_quantum_states: Vec<QuantumState>,
    /// Membrane interface health
    pub membrane_interface_health: f64,
    /// Energy required for coherence maintenance
    pub maintenance_atp_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub state_id: String,
    pub energy_level: f64,
    pub stability: f64,
    pub entanglement_partners: Vec<String>,
}

/// Analysis of oscillation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationPatternAnalysis {
    /// Detected oscillation modes
    pub oscillation_modes: Vec<OscillationMode>,
    /// Harmonic analysis
    pub harmonic_content: Vec<HarmonicComponent>,
    /// Phase relationships
    pub phase_relationships: Vec<PhaseRelationship>,
    /// Synchronization metrics
    pub synchronization_score: f64,
    /// Pattern stability
    pub pattern_stability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationMode {
    pub mode_id: String,
    pub frequency: f64,
    pub amplitude: f64,
    pub damping_factor: f64,
    pub semantic_significance: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicComponent {
    pub harmonic_number: u32,
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseRelationship {
    pub oscillator_a: String,
    pub oscillator_b: String,
    pub phase_difference: f64,
    pub coupling_strength: f64,
}

/// Result of entropy control operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyControlResult {
    /// Target entropy requested
    pub target_entropy: f64,
    /// Achieved entropy
    pub achieved_entropy: f64,
    /// Control accuracy
    pub control_accuracy: f64,
    /// Control method used
    pub control_method: EntropyControlMethod,
    /// Energy cost for control
    pub atp_consumed: f64,
    /// Time to achieve control
    pub control_time_ms: u64,
    /// Side effects of control
    pub side_effects: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntropyControlMethod {
    /// Direct oscillation endpoint manipulation
    EndpointManipulation,
    /// Frequency modulation
    FrequencyModulation,
    /// Phase adjustment
    PhaseAdjustment,
    /// Amplitude control
    AmplitudeControl,
    /// Hybrid approach
    Hybrid(Vec<EntropyControlMethod>),
}

/// Common vector type for numerical data
pub type Vector = Vec<f64>;

/// Common matrix type for 2D data
pub type Matrix = Vec<Vec<f64>>;

/// Configuration parameters
pub type ConfigMap = HashMap<String, ConfigValue>;

/// Configuration value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
}

/// Timestamp type
pub type Timestamp = chrono::DateTime<chrono::Utc>;

/// Probability type (0.0 to 1.0)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(f64);

impl Probability {
    pub fn new(value: f64) -> Result<Self, &'static str> {
        if value >= 0.0 && value <= 1.0 {
            Ok(Probability(value))
        } else {
            Err("Probability must be between 0.0 and 1.0")
        }
    }
    
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Efficiency metric (0.0 to 1.0)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Efficiency(f64);

impl Efficiency {
    pub fn new(value: f64) -> Result<Self, &'static str> {
        if value >= 0.0 && value <= 1.0 {
            Ok(Efficiency(value))
        } else {
            Err("Efficiency must be between 0.0 and 1.0")
        }
    }
    
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Energy level type
pub type EnergyLevel = f64;

/// Frequency type (Hz)
pub type Frequency = f64;

/// Amplitude type
pub type Amplitude = f64;

/// Phase type (radians)
pub type Phase = f64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_state() {
        let mut energy = EnergyState::new(100.0);
        assert_eq!(energy.energy_percentage(), 100.0);
        assert!(energy.can_afford(50.0));
        assert!(!energy.can_afford(150.0));
        
        energy.current_atp = 75.0;
        assert_eq!(energy.energy_percentage(), 75.0);
    }

    #[test]
    fn test_point_creation() {
        let point = Point {
            id: Uuid::new_v4(),
            content: "Test content".to_string(),
            certainty: 0.8,
            context: "test".to_string(),
            temporal_state: Utc::now(),
            evidence_strength: 0.7,
            positional_metadata: None,
        };
        
        assert_eq!(point.content, "Test content");
        assert_eq!(point.certainty, 0.8);
    }

    #[test]
    fn test_serialization() {
        let input = InformationInput::Text("test".to_string());
        let serialized = serde_json::to_string(&input).unwrap();
        let deserialized: InformationInput = serde_json::from_str(&serialized).unwrap();
        
        match deserialized {
            InformationInput::Text(content) => assert_eq!(content, "test"),
            _ => panic!("Expected Text variant"),
        }
    }
} 