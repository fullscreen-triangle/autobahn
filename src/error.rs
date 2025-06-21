//! Comprehensive error handling for the oscillatory bio-metabolic system.
//! Errors are categorized by their biological and physical origins with
//! specialized handling for probabilistic system failures.

use thiserror::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

#[derive(Error, Debug)]
pub enum AutobahnError {
    // ATP and Energy Management Errors
    #[error("Insufficient ATP: required {required:.2}, available {available:.2}")]
    InsufficientATP { required: f64, available: f64 },
    
    #[error("ATP regeneration failed: rate {rate:.2} below minimum threshold")]
    ATPRegenerationFailure { rate: f64 },
    
    #[error("Metabolic mode transition failed: cannot switch from {from:?} to {to:?}")]
    MetabolicTransitionFailure { from: String, to: String },
    
    // Oscillatory Dynamics Errors
    #[error("Oscillation desynchronization: frequency mismatch {observer:.2} Hz vs {system:.2} Hz")]
    DesynchronizationError { observer: f64, system: f64 },
    
    #[error("Oscillation amplitude overflow: {amplitude:.2} exceeds maximum {max_amplitude:.2}")]
    AmplitudeOverflow { amplitude: f64, max_amplitude: f64 },
    
    #[error("Phase coherence lost: coherence time {coherence_time_fs:.2} fs below threshold")]
    CoherenceLoss { coherence_time_fs: f64 },
    
    // Biological Membrane Processing Errors
    #[error("Ion channel coherence failure: {channel_type} channels showing {coherence:.2}% coherence")]
    IonChannelCoherenceFailure { channel_type: String, coherence: f64 },
    
    #[error("Membrane transport optimization failed: coupling strength {coupling:.2} out of bounds")]
    MembraneTransportFailure { coupling: f64 },
    
    #[error("Biological coherence decoherence rate {rate:.2} exceeds transport rate")]
    BiologicalDecoherenceFailure { rate: f64 },
    
    // Hierarchy and Scale Errors
    #[error("Hierarchy level {level} not supported (valid range: 1-10)")]
    UnsupportedHierarchyLevel { level: u8 },
    
    #[error("Cross-scale coupling failed between levels {level1} and {level2}")]
    CrossScaleCouplingFailure { level1: u8, level2: u8 },
    
    #[error("Hierarchy emergence detection failed: insufficient data points {data_points}")]
    EmergenceDetectionFailure { data_points: usize },
    
    // Model Selection and Processing Errors
    #[error("Model selection failed: no resonance found for profile")]
    ModelSelectionFailure,
    
    #[error("Model API error: {model_id} returned status {status}")]
    ModelAPIError { model_id: String, status: u16 },
    
    #[error("Model timeout: {model_id} exceeded {timeout_ms}ms")]
    ModelTimeout { model_id: String, timeout_ms: u64 },
    
    // Entropy and Information Errors
    #[error("Entropy calculation overflow: oscillation endpoints {endpoints}")]
    EntropyOverflow { endpoints: usize },
    
    #[error("Information value calculation failed: negative probability {probability:.2}")]
    NegativeProbability { probability: f64 },
    
    #[error("Oscillation termination distribution invalid: sum {sum:.2} ≠ 1.0")]
    InvalidDistribution { sum: f64 },
    
    // Biological System Errors
    #[error("Radical damage threshold exceeded: {current_damage:.2} > {threshold:.2}")]
    RadicalDamageThreshold { current_damage: f64, threshold: f64 },
    
    #[error("Membrane integrity compromised: thickness {thickness_nm:.2} nm below minimum")]
    MembraneIntegrityFailure { thickness_nm: f64 },
    
    #[error("Biological layer {layer:?} processing failed")]
    BiologicalLayerFailure { layer: String },
    
    // Probabilistic System Critical Errors
    #[error("Probabilistic drift detected: system confidence {confidence:.3} below critical threshold {threshold:.3}")]
    ProbabilisticDrift { confidence: f64, threshold: f64 },
    
    #[error("Bayesian network convergence failure: max iterations {max_iter} reached without convergence")]
    BayesianConvergenceFailure { max_iter: u32 },
    
    #[error("Monte Carlo sampling failure: effective sample size {effective_size} below minimum {required}")]
    MonteCarloSamplingFailure { effective_size: u32, required: u32 },
    
    #[error("Statistical significance lost: p-value {p_value:.6} exceeds threshold {threshold:.6}")]
    StatisticalSignificanceLoss { p_value: f64, threshold: f64 },
    
    #[error("Confidence calibration failure: predicted {predicted:.3} vs observed {observed:.3}, error {error:.3}")]
    ConfidenceCalibrationFailure { predicted: f64, observed: f64, error: f64 },
    
    #[error("Probabilistic invariant violation: {invariant} violated with severity {severity:.3}")]
    ProbabilisticInvariantViolation { invariant: String, severity: f64 },
    
    // System Integration Errors
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Resource exhaustion: {resource} depleted")]
    ResourceExhaustion { resource: String },
    
    #[error("System shutdown initiated: {reason}")]
    SystemShutdown { reason: String },
    
    #[error("Physics error: {message}")]
    PhysicsError { message: String },
    
    // External Integration Errors
    #[error("Serialization error: {message}")]
    SerializationError { message: String },
    
    #[error("Network error: {message}")]
    NetworkError { message: String },
    
    #[error("Database error: {message}")]
    DatabaseError { message: String },
    
    // Initialization errors
    #[error("Initialization failed: {0}")]
    InitializationError(String),

    // Processing errors
    #[error("Processing failed in {layer} layer: {reason}")]
    ProcessingError { layer: String, reason: String },

    #[error("V8 module {module} failed: {error}")]
    V8ModuleError { module: String, error: String },

    #[error("Metabolism pipeline error: {0}")]
    MetabolismError(String),

    // Probabilistic processing errors
    #[error("Uncertainty analysis failed: {0}")]
    UncertaintyError(String),

    #[error("Bayesian network error: {0}")]
    BayesianError(String),

    #[error("Resolution failed: confidence {confidence} below threshold {threshold}")]
    ResolutionError { confidence: f64, threshold: f64 },

    // Comprehension and validation errors
    #[error("Comprehension validation failed: {test} failed with score {score}")]
    ComprehensionError { test: String, score: f64 },

    #[error("Context drift detected: {drift_amount} exceeds threshold {threshold}")]
    ContextDriftError { drift_amount: f64, threshold: f64 },

    #[error("Validation puzzle failed: {puzzle_type}")]
    ValidationPuzzleError { puzzle_type: String },

    // Adversarial and security errors
    #[error("Adversarial attack detected: {attack_type} with strength {strength}")]
    AdversarialAttackError { attack_type: String, strength: f64 },

    #[error("Vulnerability detected: {vulnerability} with score {score}")]
    VulnerabilityError { vulnerability: String, score: f64 },

    #[error("Perturbation test failed: {test_type} showed instability {instability}")]
    PerturbationError { test_type: String, instability: f64 },

    // Champagne phase (dream processing) errors
    #[error("Dream processing failed: {stage} stage error")]
    DreamProcessingError { stage: String },

    #[error("Lactatic acid cycle disrupted: {disruption_type}")]
    LactaticAcidCycleError { disruption_type: String },

    // Timeout and resource errors
    #[error("Operation timeout: {operation} exceeded {timeout_ms}ms")]
    TimeoutError { operation: String, timeout_ms: u64 },

    #[error("Memory limit exceeded: {used_mb}MB > {limit_mb}MB")]
    MemoryLimitExceeded { used_mb: f64, limit_mb: f64 },

    #[error("CPU limit exceeded: {usage_percent}% > {limit_percent}%")]
    CPULimitExceeded { usage_percent: f64, limit_percent: f64 },

    // Not implemented
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
    
    // Hardware Oscillation Synchronization Errors
    #[cfg(feature = "hardware-sync")]
    #[error("Hardware oscillation capture failed: {message}")]
    HardwareError(String),
    
    #[cfg(feature = "hardware-sync")]
    #[error("Frequency domain synchronization failed: {domain} at {frequency}Hz")]
    FrequencyDomainError { domain: String, frequency: f64 },
    
    #[cfg(feature = "hardware-sync")]
    #[error("Phase lock failure: {domain} phase error {error_radians:.3} rad")]
    PhaseLockError { domain: String, error_radians: f64 },
    
    #[cfg(feature = "hardware-sync")]
    #[error("Hardware coherence loss: {domain} coherence {coherence:.3} below threshold {threshold:.3}")]
    HardwareCoherenceError { domain: String, coherence: f64, threshold: f64 },
    
    // Optical Processing Errors
    #[cfg(feature = "optical-processing")]
    #[error("Optical processing failed: {message}")]
    OpticalError(String),
    
    #[cfg(feature = "optical-processing")]
    #[error("Light source failure: {source_id} wavelength {wavelength}nm")]
    LightSourceError { source_id: String, wavelength: u16 },
    
    #[cfg(feature = "optical-processing")]
    #[error("Fire circle disruption: coherence {coherence:.3} insufficient for pattern")]
    FireCircleError { coherence: f64 },
    
    #[cfg(feature = "optical-processing")]
    #[error("650nm consciousness coupling failed: strength {strength:.3} below minimum")]
    ConsciousnessCouplingError { strength: f64 },
    
    // Environmental Photosynthesis Errors
    #[cfg(feature = "environmental-photosynthesis")]
    #[error("Photosynthesis processing failed: {message}")]
    PhotosynthesisError(String),
    
    #[cfg(feature = "environmental-photosynthesis")]
    #[error("Visual ATP conversion failed: wavelength {wavelength}nm efficiency {efficiency:.3}")]
    VisualATPError { wavelength: u16, efficiency: f64 },
    
    #[cfg(feature = "environmental-photosynthesis")]
    #[error("Chaos substrate generation failed: complexity {complexity:.3} insufficient")]
    ChaosSubstrateError { complexity: f64 },
    
    #[cfg(feature = "environmental-photosynthesis")]
    #[error("Agency illusion failure: chaos level {chaos_level:.3} below threshold {threshold:.3}")]
    AgencyIllusionError { chaos_level: f64, threshold: f64 },
}

/// Enhanced error severity classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational - system continues normally
    Info,
    /// Low severity - minor degradation possible
    Low,
    /// Medium severity - noticeable impact on performance
    Medium,
    /// High severity - significant functionality impaired
    High,
    /// Critical severity - system stability at risk
    Critical,
    /// Fatal severity - immediate shutdown required
    Fatal,
}

/// Probabilistic system health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthMetrics {
    /// Overall system confidence
    pub system_confidence: f64,
    /// Statistical convergence indicators
    pub convergence_metrics: ConvergenceMetrics,
    /// Calibration quality metrics
    pub calibration_metrics: CalibrationMetrics,
    /// Drift detection results
    pub drift_metrics: DriftMetrics,
    /// Invariant violation tracking
    pub invariant_metrics: InvariantMetrics,
    /// Resource utilization
    pub resource_metrics: ResourceMetrics,
    /// Timestamp of last health check
    pub last_updated: DateTime<Utc>,
}

/// Convergence tracking for iterative algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    /// Whether algorithms are converging
    pub converged: bool,
    /// Rate of convergence
    pub convergence_rate: f64,
    /// Number of iterations to convergence
    pub iterations_to_convergence: Option<u32>,
    /// Gelman-Rubin statistic for MCMC
    pub gelman_rubin_statistic: f64,
    /// Effective sample size
    pub effective_sample_size: u32,
}

/// Confidence calibration quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Calibration error (ECE - Expected Calibration Error)
    pub expected_calibration_error: f64,
    /// Maximum calibration error
    pub max_calibration_error: f64,
    /// Reliability diagram data points
    pub reliability_bins: Vec<CalibrationBin>,
    /// Brier score
    pub brier_score: f64,
}

/// Individual calibration bin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Predicted confidence range
    pub confidence_range: (f64, f64),
    /// Observed accuracy in this bin
    pub observed_accuracy: f64,
    /// Number of samples in bin
    pub sample_count: u32,
}

/// System drift detection metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMetrics {
    /// Statistical drift detected
    pub drift_detected: bool,
    /// Magnitude of drift
    pub drift_magnitude: f64,
    /// Drift detection confidence
    pub drift_confidence: f64,
    /// Drift trend direction
    pub drift_direction: DriftDirection,
    /// Time since drift started
    pub drift_duration_seconds: f64,
}

/// Direction of detected drift
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDirection {
    Improving,
    Degrading,
    Oscillating,
    Unknown,
}

/// Probabilistic invariant violation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantMetrics {
    /// List of violated invariants
    pub violated_invariants: Vec<InvariantViolation>,
    /// Total violation severity score
    pub total_violation_severity: f64,
    /// Critical violations requiring immediate attention
    pub critical_violations: u32,
}

/// Individual invariant violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantViolation {
    /// Name of violated invariant
    pub invariant_name: String,
    /// Severity of violation (0.0 to 1.0)
    pub severity: f64,
    /// Expected value
    pub expected_value: f64,
    /// Observed value
    pub observed_value: f64,
    /// Violation persistence duration
    pub duration_seconds: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// ATP consumption rate
    pub atp_consumption_rate: f64,
    /// Memory usage percentage
    pub memory_usage_percent: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Network bandwidth utilization
    pub network_usage_mbps: f64,
}

impl AutobahnError {
    /// Determine error severity with probabilistic considerations
    pub fn severity_level(&self) -> ErrorSeverity {
        match self {
            // Fatal errors - immediate shutdown
            AutobahnError::SystemShutdown { .. } => ErrorSeverity::Fatal,
            AutobahnError::MembraneIntegrityFailure { .. } => ErrorSeverity::Fatal,
            AutobahnError::ProbabilisticInvariantViolation { severity, .. } if *severity > 0.9 => ErrorSeverity::Fatal,
            
            // Critical errors - system stability at risk
            AutobahnError::RadicalDamageThreshold { .. } => ErrorSeverity::Critical,
            AutobahnError::ProbabilisticDrift { confidence, threshold } if confidence < threshold * 0.5 => ErrorSeverity::Critical,
            AutobahnError::BayesianConvergenceFailure { .. } => ErrorSeverity::Critical,
            AutobahnError::StatisticalSignificanceLoss { .. } => ErrorSeverity::Critical,
            AutobahnError::ConfidenceCalibrationFailure { error, .. } if *error > 0.3 => ErrorSeverity::Critical,
            
            // High severity - significant functionality impaired
            AutobahnError::InsufficientATP { .. } => ErrorSeverity::High,
            AutobahnError::MonteCarloSamplingFailure { .. } => ErrorSeverity::High,
            AutobahnError::CoherenceLoss { .. } => ErrorSeverity::High,
            AutobahnError::IonChannelCoherenceFailure { coherence, .. } if *coherence < 50.0 => ErrorSeverity::High,
            
            // Medium severity - noticeable impact
            AutobahnError::ModelSelectionFailure => ErrorSeverity::Medium,
            AutobahnError::DesynchronizationError { .. } => ErrorSeverity::Medium,
            AutobahnError::BiologicalDecoherenceFailure { .. } => ErrorSeverity::Medium,
            AutobahnError::ContextDriftError { .. } => ErrorSeverity::Medium,
            
            // Low severity - minor degradation
            AutobahnError::ModelTimeout { .. } => ErrorSeverity::Low,
            AutobahnError::NetworkError { .. } => ErrorSeverity::Low,
            
            // Default to medium for unknown cases
            _ => ErrorSeverity::Medium,
        }
    }
    
    /// Determine if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self.severity_level() {
            ErrorSeverity::Fatal => false,
            ErrorSeverity::Critical => false,
            _ => true,
        }
    }
    
    /// Get probabilistic confidence that this error can be handled gracefully
    pub fn recovery_probability(&self) -> f64 {
        match self {
            AutobahnError::InsufficientATP { .. } => 0.9,
            AutobahnError::ModelTimeout { .. } => 0.8,
            AutobahnError::NetworkError { .. } => 0.7,
            AutobahnError::DesynchronizationError { .. } => 0.6,
            AutobahnError::MonteCarloSamplingFailure { .. } => 0.4,
            AutobahnError::ProbabilisticDrift { .. } => 0.3,
            AutobahnError::BayesianConvergenceFailure { .. } => 0.2,
            AutobahnError::SystemShutdown { .. } => 0.0,
            _ => 0.5,
        }
    }
}

/// Error context for detailed debugging with probabilistic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub module: String,
    pub operation: String,
    pub atp_available: f64,
    pub confidence_level: f64,
    pub processing_layer: String,
    pub timestamp: DateTime<Utc>,
    /// Probabilistic state when error occurred
    pub probabilistic_state: ProbabilisticState,
    /// System health at error time
    pub system_health: SystemHealthMetrics,
    /// Reproducibility information
    pub reproducibility_info: ReproducibilityInfo,
}

/// Snapshot of probabilistic state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticState {
    /// Random seed state
    pub random_seed: Option<u64>,
    /// Current inference state
    pub inference_state: String,
    /// Monte Carlo iteration count
    pub mc_iterations: u32,
    /// Bayesian network state hash
    pub network_state_hash: String,
    /// Confidence distribution parameters
    pub confidence_params: Vec<f64>,
}

/// Information for reproducing probabilistic errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityInfo {
    /// Input that triggered the error
    pub triggering_input: String,
    /// Configuration at error time
    pub config_snapshot: String,
    /// Sequence of operations leading to error
    pub operation_sequence: Vec<String>,
    /// Environmental conditions
    pub environment_snapshot: HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(module: &str, operation: &str) -> Self {
        Self {
            module: module.to_string(),
            operation: operation.to_string(),
            atp_available: 0.0,
            confidence_level: 0.0,
            processing_layer: "unknown".to_string(),
            timestamp: Utc::now(),
            probabilistic_state: ProbabilisticState::default(),
            system_health: SystemHealthMetrics::default(),
            reproducibility_info: ReproducibilityInfo::default(),
        }
    }

    pub fn with_atp(mut self, atp: f64) -> Self {
        self.atp_available = atp;
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence_level = confidence;
        self
    }

    pub fn with_layer(mut self, layer: &str) -> Self {
        self.processing_layer = layer.to_string();
        self
    }
    
    pub fn with_probabilistic_state(mut self, state: ProbabilisticState) -> Self {
        self.probabilistic_state = state;
        self
    }
    
    pub fn with_health_metrics(mut self, health: SystemHealthMetrics) -> Self {
        self.system_health = health;
        self
    }
}

// Default implementations for new types
impl Default for ProbabilisticState {
    fn default() -> Self {
        Self {
            random_seed: None,
            inference_state: "unknown".to_string(),
            mc_iterations: 0,
            network_state_hash: "".to_string(),
            confidence_params: Vec::new(),
        }
    }
}

impl Default for SystemHealthMetrics {
    fn default() -> Self {
        Self {
            system_confidence: 0.5,
            convergence_metrics: ConvergenceMetrics::default(),
            calibration_metrics: CalibrationMetrics::default(),
            drift_metrics: DriftMetrics::default(),
            invariant_metrics: InvariantMetrics::default(),
            resource_metrics: ResourceMetrics::default(),
            last_updated: Utc::now(),
        }
    }
}

impl Default for ConvergenceMetrics {
    fn default() -> Self {
        Self {
            converged: false,
            convergence_rate: 0.0,
            iterations_to_convergence: None,
            gelman_rubin_statistic: 1.0,
            effective_sample_size: 0,
        }
    }
}

impl Default for CalibrationMetrics {
    fn default() -> Self {
        Self {
            expected_calibration_error: 0.0,
            max_calibration_error: 0.0,
            reliability_bins: Vec::new(),
            brier_score: 0.5,
        }
    }
}

impl Default for DriftMetrics {
    fn default() -> Self {
        Self {
            drift_detected: false,
            drift_magnitude: 0.0,
            drift_confidence: 0.0,
            drift_direction: DriftDirection::Unknown,
            drift_duration_seconds: 0.0,
        }
    }
}

impl Default for InvariantMetrics {
    fn default() -> Self {
        Self {
            violated_invariants: Vec::new(),
            total_violation_severity: 0.0,
            critical_violations: 0,
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            atp_consumption_rate: 0.0,
            memory_usage_percent: 0.0,
            cpu_usage_percent: 0.0,
            network_usage_mbps: 0.0,
        }
    }
}

impl Default for ReproducibilityInfo {
    fn default() -> Self {
        Self {
            triggering_input: String::new(),
            config_snapshot: String::new(),
            operation_sequence: Vec::new(),
            environment_snapshot: HashMap::new(),
        }
    }
}

/// Trait for errors that can provide biological context
pub trait BiologicalError {
    fn atp_cost(&self) -> Option<f64>;
    fn affected_modules(&self) -> Vec<String>;
    fn recovery_strategy(&self) -> RecoveryStrategy;
    fn health_impact(&self) -> SystemHealthImpact;
}

/// Recovery strategies for different error types
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry with different parameters
    Retry { max_attempts: u32, backoff_ms: u64 },
    /// Switch to anaerobic processing
    AnaerobicMode,
    /// Enter champagne phase for recovery
    ChampagnePhase,
    /// Reduce complexity and try again
    SimplifyProcessing,
    /// Request more ATP and retry
    RequestMoreATP { amount: f64 },
    /// Degrade gracefully with reduced functionality
    GracefulDegradation,
    /// Increase sampling for better statistics
    IncreaseSampling { factor: f64 },
    /// Reset probabilistic state
    ResetProbabilisticState,
    /// Complete failure - cannot recover
    Fatal,
}

/// Impact on system health
#[derive(Debug, Clone)]
pub struct SystemHealthImpact {
    /// Confidence degradation amount
    pub confidence_impact: f64,
    /// Affected subsystems
    pub affected_subsystems: Vec<String>,
    /// Recovery time estimate (seconds)
    pub recovery_time_estimate: f64,
    /// Permanent damage assessment
    pub permanent_damage_risk: f64,
}

impl BiologicalError for AutobahnError {
    fn atp_cost(&self) -> Option<f64> {
        match self {
            AutobahnError::InsufficientATP { required, .. } => Some(*required),
            AutobahnError::ProcessingError { .. } => Some(10.0),
            AutobahnError::V8ModuleError { .. } => Some(20.0),
            AutobahnError::ResolutionError { .. } => Some(15.0),
            AutobahnError::MonteCarloSamplingFailure { .. } => Some(50.0),
            AutobahnError::BayesianConvergenceFailure { .. } => Some(100.0),
            _ => None,
        }
    }

    fn affected_modules(&self) -> Vec<String> {
        match self {
            AutobahnError::V8ModuleError { module, .. } => vec![module.clone()],
            AutobahnError::ProcessingError { layer, .. } => vec![layer.clone()],
            AutobahnError::MetabolismError(_) => vec!["v8_pipeline".to_string()],
            AutobahnError::ComprehensionError { .. } => vec!["clothesline".to_string()],
            AutobahnError::ContextDriftError { .. } => vec!["nicotine".to_string()],
            AutobahnError::AdversarialAttackError { .. } => vec!["diggiden".to_string()],
            AutobahnError::BayesianError(_) => vec!["mzekezeke".to_string()],
            AutobahnError::ProbabilisticDrift { .. } => vec!["probabilistic_engine".to_string()],
            _ => vec![],
        }
    }

    fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            AutobahnError::InsufficientATP { required, .. } => {
                RecoveryStrategy::RequestMoreATP { amount: *required }
            }
            AutobahnError::ProcessingError { .. } => RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff_ms: 1000,
            },
            AutobahnError::UncertaintyError(_) => RecoveryStrategy::SimplifyProcessing,
            AutobahnError::ContextDriftError { .. } => RecoveryStrategy::ChampagnePhase,
            AutobahnError::AdversarialAttackError { .. } => RecoveryStrategy::AnaerobicMode,
            AutobahnError::ValidationPuzzleError { .. } => RecoveryStrategy::GracefulDegradation,
            AutobahnError::TimeoutError { .. } => RecoveryStrategy::SimplifyProcessing,
            AutobahnError::MonteCarloSamplingFailure { .. } => RecoveryStrategy::IncreaseSampling { factor: 2.0 },
            AutobahnError::BayesianConvergenceFailure { .. } => RecoveryStrategy::ResetProbabilisticState,
            AutobahnError::ProbabilisticDrift { .. } => RecoveryStrategy::ResetProbabilisticState,
            AutobahnError::StatisticalSignificanceLoss { .. } => RecoveryStrategy::IncreaseSampling { factor: 1.5 },
            _ => RecoveryStrategy::Fatal,
        }
    }
    
    fn health_impact(&self) -> SystemHealthImpact {
        match self {
            AutobahnError::ProbabilisticDrift { confidence, threshold } => {
                SystemHealthImpact {
                    confidence_impact: threshold - confidence,
                    affected_subsystems: vec!["probabilistic_engine".to_string(), "bayesian_networks".to_string()],
                    recovery_time_estimate: 300.0, // 5 minutes
                    permanent_damage_risk: 0.1,
                }
            }
            AutobahnError::BayesianConvergenceFailure { .. } => {
                SystemHealthImpact {
                    confidence_impact: 0.3,
                    affected_subsystems: vec!["bayesian_networks".to_string(), "inference_engine".to_string()],
                    recovery_time_estimate: 600.0, // 10 minutes
                    permanent_damage_risk: 0.2,
                }
            }
            AutobahnError::ConfidenceCalibrationFailure { error, .. } => {
                SystemHealthImpact {
                    confidence_impact: *error,
                    affected_subsystems: vec!["calibration_system".to_string()],
                    recovery_time_estimate: 120.0, // 2 minutes
                    permanent_damage_risk: 0.05,
                }
            }
            _ => SystemHealthImpact {
                confidence_impact: 0.1,
                affected_subsystems: vec![],
                recovery_time_estimate: 30.0,
                permanent_damage_risk: 0.01,
            }
        }
    }
}

pub type AutobahnResult<T> = std::result::Result<T, AutobahnError>;

#[derive(Error, Debug)]
pub enum MetabolismError {
    #[error("Glycolysis failed at step {step}: {reason}")]
    GlycolysisFailed { step: u8, reason: String },

    #[error("Krebs cycle failed at step {step}: {reason}")]
    KrebsCycleFailed { step: u8, reason: String },

    #[error("Electron transport failed at complex {complex}: {reason}")]
    ElectronTransportFailed { complex: String, reason: String },

    #[error("ATP synthase failed: {0}")]
    ATPSynthaseFailed(String),
}

impl From<MetabolismError> for AutobahnError {
    fn from(error: MetabolismError) -> Self {
        AutobahnError::MetabolismError(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_severity_classification() {
        let fatal_error = AutobahnError::SystemShutdown { reason: "Critical failure".to_string() };
        assert_eq!(fatal_error.severity_level(), ErrorSeverity::Fatal);
        
        let critical_error = AutobahnError::ProbabilisticDrift { confidence: 0.1, threshold: 0.8 };
        assert_eq!(critical_error.severity_level(), ErrorSeverity::Critical);
        
        let high_error = AutobahnError::InsufficientATP { required: 100.0, available: 10.0 };
        assert_eq!(high_error.severity_level(), ErrorSeverity::High);
    }
    
    #[test]
    fn test_recovery_probability() {
        let recoverable_error = AutobahnError::InsufficientATP { required: 100.0, available: 10.0 };
        assert!(recoverable_error.recovery_probability() > 0.8);
        
        let fatal_error = AutobahnError::SystemShutdown { reason: "Critical failure".to_string() };
        assert_eq!(fatal_error.recovery_probability(), 0.0);
    }
    
    #[test]
    fn test_health_impact_calculation() {
        let drift_error = AutobahnError::ProbabilisticDrift { confidence: 0.3, threshold: 0.8 };
        let impact = drift_error.health_impact();
        assert_eq!(impact.confidence_impact, 0.5);
        assert!(impact.recovery_time_estimate > 0.0);
    }
} 