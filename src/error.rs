//! Comprehensive error handling for the oscillatory bio-metabolic system.
//! Errors are categorized by their biological and physical origins.

use thiserror::Error;

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
    
    // Quantum Mechanical Errors
    #[error("Quantum tunneling probability calculation failed: barrier height {barrier_height:.2} eV")]
    QuantumTunnelingFailure { barrier_height: f64 },
    
    #[error("ENAQT coupling optimization failed: coupling strength {coupling:.2} out of bounds")]
    ENAQTCouplingFailure { coupling: f64 },
    
    #[error("Quantum coherence decoherence rate {rate:.2} exceeds transport rate")]
    QuantumDecoherenceFailure { rate: f64 },
    
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
}

impl From<serde_json::Error> for AutobahnError {
    fn from(err: serde_json::Error) -> Self {
        AutobahnError::SerializationError {
            message: err.to_string(),
        }
    }
}

impl From<reqwest::Error> for AutobahnError {
    fn from(err: reqwest::Error) -> Self {
        AutobahnError::NetworkError {
            message: err.to_string(),
        }
    }
}

pub type AutobahnResult<T> = std::result::Result<T, AutobahnError>;

// Utility functions for error context
impl AutobahnError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Recoverable errors that can be retried or handled gracefully
            AutobahnError::InsufficientATP { .. } => true,
            AutobahnError::ModelTimeout { .. } => true,
            AutobahnError::NetworkError { .. } => true,
            AutobahnError::DesynchronizationError { .. } => true,
            AutobahnError::CoherenceLoss { .. } => true,
            
            // Non-recoverable errors that indicate fundamental problems
            AutobahnError::SystemShutdown { .. } => false,
            AutobahnError::MembraneIntegrityFailure { .. } => false,
            AutobahnError::RadicalDamageThreshold { .. } => false,
            AutobahnError::UnsupportedHierarchyLevel { .. } => false,
            
            // Context-dependent errors
            _ => true,
        }
    }
    
    pub fn severity_level(&self) -> ErrorSeverity {
        match self {
            AutobahnError::SystemShutdown { .. } => ErrorSeverity::Critical,
            AutobahnError::MembraneIntegrityFailure { .. } => ErrorSeverity::Critical,
            AutobahnError::RadicalDamageThreshold { .. } => ErrorSeverity::High,
            AutobahnError::InsufficientATP { .. } => ErrorSeverity::High,
            AutobahnError::ModelSelectionFailure => ErrorSeverity::Medium,
            AutobahnError::CoherenceLoss { .. } => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Main error type for Autobahn operations
#[derive(Error, Debug)]
pub enum AutobahnError {
    /// Initialization errors
    #[error("Initialization failed: {0}")]
    InitializationError(String),

    /// ATP (energy) related errors
    #[error("Insufficient ATP for operation: required {required}, available {available}")]
    InsufficientATP { required: f64, available: f64 },

    #[error("ATP regeneration failed: {0}")]
    ATPRegenerationError(String),

    /// Processing errors
    #[error("Processing failed in {layer} layer: {reason}")]
    ProcessingError { layer: String, reason: String },

    #[error("V8 module {module} failed: {error}")]
    V8ModuleError { module: String, error: String },

    #[error("Metabolism pipeline error: {0}")]
    MetabolismError(String),

    /// Probabilistic processing errors
    #[error("Uncertainty analysis failed: {0}")]
    UncertaintyError(String),

    #[error("Bayesian network error: {0}")]
    BayesianError(String),

    #[error("Resolution failed: confidence {confidence} below threshold {threshold}")]
    ResolutionError { confidence: f64, threshold: f64 },

    /// Comprehension and validation errors
    #[error("Comprehension validation failed: {test} failed with score {score}")]
    ComprehensionError { test: String, score: f64 },

    #[error("Context drift detected: {drift_amount} exceeds threshold {threshold}")]
    ContextDriftError { drift_amount: f64, threshold: f64 },

    #[error("Validation puzzle failed: {puzzle_type}")]
    ValidationPuzzleError { puzzle_type: String },

    /// Adversarial and security errors
    #[error("Adversarial attack detected: {attack_type} with strength {strength}")]
    AdversarialAttackError { attack_type: String, strength: f64 },

    #[error("Vulnerability detected: {vulnerability} with score {score}")]
    VulnerabilityError { vulnerability: String, score: f64 },

    #[error("Perturbation test failed: {test_type} showed instability {instability}")]
    PerturbationError { test_type: String, instability: f64 },

    /// Champagne phase (dream processing) errors
    #[error("Lactate processing failed: {0}")]
    LactateProcessingError(String),

    #[error("Dream processing interrupted: {reason}")]
    DreamProcessingError { reason: String },

    #[error("Champagne phase unavailable: user status {user_status}")]
    ChampagneUnavailableError { user_status: String },

    /// Data and serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Data corruption detected: {0}")]
    DataCorruptionError(String),

    #[error("Invalid input format: expected {expected}, got {actual}")]
    InvalidInputError { expected: String, actual: String },

    /// Network and I/O errors
    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Temporal and time-related errors
    #[error("Temporal decay error: {0}")]
    TemporalDecayError(String),

    #[error("Evidence too old: age {age} exceeds maximum {max_age}")]
    EvidenceExpiredError { age: f64, max_age: f64 },

    /// Configuration and setup errors
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Missing required component: {component}")]
    MissingComponentError { component: String },

    /// Physics and simulation errors
    #[error("Physics error: {message}")]
    PhysicsError { message: String },

    /// Generic errors
    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    #[error("Operation timeout: {operation} exceeded {timeout_ms}ms")]
    TimeoutError { operation: String, timeout_ms: u64 },
}

/// Result type alias for Autobahn operations
pub type AutobahnResult<T> = Result<T, AutobahnError>;

/// Specialized error types for different subsystems
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

/// Error context for detailed debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub module: String,
    pub operation: String,
    pub atp_available: f64,
    pub confidence_level: f64,
    pub processing_layer: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ErrorContext {
    pub fn new(module: &str, operation: &str) -> Self {
        Self {
            module: module.to_string(),
            operation: operation.to_string(),
            atp_available: 0.0,
            confidence_level: 0.0,
            processing_layer: "unknown".to_string(),
            timestamp: chrono::Utc::now(),
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
}

/// Trait for errors that can provide biological context
pub trait BiologicalError {
    fn atp_cost(&self) -> Option<f64>;
    fn affected_modules(&self) -> Vec<String>;
    fn recovery_strategy(&self) -> RecoveryStrategy;
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
    /// Complete failure - cannot recover
    Fatal,
}

impl BiologicalError for AutobahnError {
    fn atp_cost(&self) -> Option<f64> {
        match self {
            AutobahnError::InsufficientATP { required, .. } => Some(*required),
            AutobahnError::ProcessingError { .. } => Some(10.0),
            AutobahnError::V8ModuleError { .. } => Some(20.0),
            AutobahnError::ResolutionError { .. } => Some(15.0),
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
            _ => RecoveryStrategy::Fatal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = AutobahnError::InsufficientATP {
            required: 100.0,
            available: 50.0,
        };
        
        assert!(error.to_string().contains("Insufficient ATP"));
        assert!(error.to_string().contains("100"));
        assert!(error.to_string().contains("50"));
    }

    #[test]
    fn test_biological_error_trait() {
        let error = AutobahnError::ProcessingError {
            layer: "reasoning".to_string(),
            reason: "test failure".to_string(),
        };

        assert_eq!(error.atp_cost(), Some(10.0));
        assert_eq!(error.affected_modules(), vec!["reasoning"]);
        
        match error.recovery_strategy() {
            RecoveryStrategy::Retry { max_attempts, .. } => {
                assert_eq!(max_attempts, 3);
            }
            _ => panic!("Expected Retry strategy"),
        }
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("mzekezeke", "bayesian_update")
            .with_atp(75.5)
            .with_confidence(0.85)
            .with_layer("reasoning");

        assert_eq!(context.module, "mzekezeke");
        assert_eq!(context.operation, "bayesian_update");
        assert_eq!(context.atp_available, 75.5);
        assert_eq!(context.confidence_level, 0.85);
        assert_eq!(context.processing_layer, "reasoning");
    }
} 