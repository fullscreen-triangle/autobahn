//! Temporal Processor - Time-Based Evidence and Pattern Analysis
//!
//! This module implements temporal processing capabilities including evidence decay,
//! temporal pattern recognition, and historical analysis for the biological metabolism computer.

use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::traits::TemporalProcessor;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Temporal processor for time-based analysis
#[derive(Debug, Clone)]
pub struct TemporalProcessorEngine {
    /// Evidence decay manager
    decay_manager: EvidenceDecayManager,
    /// Temporal pattern analyzer
    pattern_analyzer: TemporalPatternAnalyzer,
    /// Historical data manager
    historical_manager: HistoricalDataManager,
    /// Temporal configuration
    config: TemporalConfig,
    /// Processing statistics
    stats: TemporalStats,
}

/// Evidence decay management system
#[derive(Debug, Clone)]
pub struct EvidenceDecayManager {
    /// Decay functions for different evidence types
    decay_functions: HashMap<EvidenceType, DecayFunction>,
    /// Active evidence tracking
    active_evidence: Vec<TimedEvidence>,
    /// Decay parameters
    decay_params: DecayParameters,
}

/// Temporal pattern analysis system
#[derive(Debug, Clone)]
pub struct TemporalPatternAnalyzer {
    /// Pattern detection algorithms
    pattern_detectors: Vec<PatternDetector>,
    /// Detected patterns
    detected_patterns: Vec<TemporalPattern>,
    /// Pattern matching configuration
    pattern_config: PatternConfig,
}

/// Historical data management
#[derive(Debug, Clone)]
pub struct HistoricalDataManager {
    /// Time-series data storage
    time_series_data: HashMap<String, TimeSeries>,
    /// Historical snapshots
    snapshots: VecDeque<HistoricalSnapshot>,
    /// Retention policies
    retention_policies: RetentionPolicies,
}

/// Types of evidence for decay calculation
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Scientific literature evidence
    Literature,
    /// Experimental data
    Experimental,
    /// Clinical observations
    Clinical,
    /// Computational predictions
    Computational,
    /// Expert opinions
    Expert,
    /// Sensor data
    Sensor,
    /// User interactions
    UserInteraction,
}

/// Decay function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Exponential decay: strength * e^(-λt)
    Exponential { lambda: f64 },
    /// Linear decay: max(0, strength - rate*t)
    Linear { rate: f64 },
    /// Power law decay: strength * t^(-α)
    PowerLaw { alpha: f64 },
    /// Step function decay: strength until threshold, then drop
    StepFunction { threshold_hours: f64, drop_factor: f64 },
    /// Custom decay with multiple phases
    Custom { phases: Vec<DecayPhase> },
}

/// Phase in custom decay function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayPhase {
    /// Duration of this phase in hours
    pub duration_hours: f64,
    /// Decay rate during this phase
    pub decay_rate: f64,
    /// Minimum strength at end of phase
    pub min_strength: f64,
}

/// Timed evidence with decay tracking
#[derive(Debug, Clone)]
pub struct TimedEvidence {
    /// The evidence data
    pub evidence: Evidence,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Current strength (after decay)
    pub current_strength: f64,
    /// Original strength
    pub original_strength: f64,
    /// Evidence type for decay calculation
    pub evidence_type: EvidenceType,
    /// Decay metadata
    pub decay_metadata: DecayMetadata,
}

/// Decay calculation metadata
#[derive(Debug, Clone)]
pub struct DecayMetadata {
    /// Last decay calculation time
    pub last_calculated: DateTime<Utc>,
    /// Decay rate applied
    pub decay_rate: f64,
    /// Half-life in hours
    pub half_life_hours: f64,
    /// Minimum viable strength
    pub min_viable_strength: f64,
}

/// Temporal pattern detection
#[derive(Debug, Clone)]
pub struct PatternDetector {
    /// Pattern type this detector finds
    pub pattern_type: PatternType,
    /// Detection algorithm
    pub algorithm: DetectionAlgorithm,
    /// Sensitivity threshold
    pub sensitivity: f64,
    /// Minimum pattern duration
    pub min_duration_hours: f64,
}

/// Types of temporal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Cyclical patterns (daily, weekly, etc.)
    Cyclical { period_hours: f64 },
    /// Trending patterns (increasing/decreasing)
    Trending { direction: TrendDirection },
    /// Seasonal patterns
    Seasonal { season_length_days: f64 },
    /// Anomaly detection
    Anomaly { deviation_threshold: f64 },
    /// Correlation patterns between variables
    Correlation { variables: Vec<String> },
    /// Burst patterns (sudden increases)
    Burst { intensity_threshold: f64 },
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionAlgorithm {
    /// Fast Fourier Transform for frequency analysis
    FFT,
    /// Autocorrelation analysis
    Autocorrelation,
    /// Moving average analysis
    MovingAverage { window_size: usize },
    /// Statistical change point detection
    ChangePoint,
    /// Machine learning based
    MachineLearning { model_type: String },
}

/// Detected temporal pattern
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Detection confidence
    pub confidence: f64,
    /// Pattern start time
    pub start_time: DateTime<Utc>,
    /// Pattern end time (if completed)
    pub end_time: Option<DateTime<Utc>>,
    /// Pattern parameters
    pub parameters: HashMap<String, f64>,
    /// Associated data points
    pub data_points: Vec<DataPoint>,
}

/// Time series data point
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Time series data structure
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Series identifier
    pub id: String,
    /// Data points
    pub data: VecDeque<DataPoint>,
    /// Series metadata
    pub metadata: HashMap<String, String>,
    /// Last update time
    pub last_updated: DateTime<Utc>,
}

/// Historical snapshot
#[derive(Debug, Clone)]
pub struct HistoricalSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,
    /// System state at snapshot time
    pub system_state: SystemState,
    /// Processing metrics
    pub processing_metrics: ProcessingMetrics,
    /// Evidence summary
    pub evidence_summary: EvidenceSummary,
}

/// System state for historical tracking
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Active modules
    pub active_modules: Vec<String>,
    /// Energy state
    pub energy_state: EnergyState,
    /// Processing load
    pub processing_load: f64,
    /// Memory usage
    pub memory_usage: MemoryUsage,
}

/// Memory usage tracking
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total memory used (MB)
    pub total_mb: f64,
    /// Memory by component
    pub by_component: HashMap<String, f64>,
    /// Peak memory usage
    pub peak_mb: f64,
}

/// Processing metrics snapshot
#[derive(Debug, Clone)]
pub struct ProcessingMetrics {
    /// Requests processed
    pub requests_processed: u64,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Error rate
    pub error_rate: f64,
    /// Throughput (requests/second)
    pub throughput: f64,
}

/// Evidence summary for historical tracking
#[derive(Debug, Clone)]
pub struct EvidenceSummary {
    /// Total evidence count
    pub total_evidence: u32,
    /// Evidence by type
    pub by_type: HashMap<EvidenceType, u32>,
    /// Average evidence strength
    pub avg_strength: f64,
    /// Expired evidence count
    pub expired_evidence: u32,
}

/// Decay parameters configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayParameters {
    /// Default decay function
    pub default_decay: DecayFunction,
    /// Minimum evidence strength before removal
    pub min_viable_strength: f64,
    /// Decay calculation interval
    pub calculation_interval_minutes: u64,
    /// Enable adaptive decay rates
    pub adaptive_decay: bool,
}

/// Pattern detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Enable pattern detection
    pub enable_detection: bool,
    /// Maximum patterns to track
    pub max_patterns: usize,
    /// Pattern detection interval
    pub detection_interval_minutes: u64,
    /// Minimum confidence for pattern acceptance
    pub min_confidence: f64,
}

/// Data retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicies {
    /// Maximum age for time series data
    pub max_time_series_age_days: u32,
    /// Maximum number of snapshots
    pub max_snapshots: usize,
    /// Snapshot interval
    pub snapshot_interval_hours: u32,
    /// Compression settings
    pub compression_settings: CompressionSettings,
}

/// Data compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression threshold (days)
    pub compression_threshold_days: u32,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Lossless compression
    Lossless,
    /// Lossy compression with quality factor
    Lossy { quality: f64 },
    /// Adaptive compression
    Adaptive,
}

/// Temporal processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Decay parameters
    pub decay_params: DecayParameters,
    /// Pattern configuration
    pub pattern_config: PatternConfig,
    /// Retention policies
    pub retention_policies: RetentionPolicies,
    /// Enable temporal processing
    pub enable_temporal_processing: bool,
}

/// Temporal processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalStats {
    /// Evidence decay calculations performed
    pub decay_calculations: u64,
    /// Patterns detected
    pub patterns_detected: u64,
    /// Historical snapshots created
    pub snapshots_created: u64,
    /// Data points processed
    pub data_points_processed: u64,
    /// Average processing time
    pub avg_processing_time_ms: f64,
}

impl TemporalProcessorEngine {
    /// Create new temporal processor
    pub fn new() -> Self {
        Self {
            decay_manager: EvidenceDecayManager::new(),
            pattern_analyzer: TemporalPatternAnalyzer::new(),
            historical_manager: HistoricalDataManager::new(),
            config: TemporalConfig::default(),
            stats: TemporalStats::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: TemporalConfig) -> Self {
        Self {
            decay_manager: EvidenceDecayManager::with_config(&config.decay_params),
            pattern_analyzer: TemporalPatternAnalyzer::with_config(&config.pattern_config),
            historical_manager: HistoricalDataManager::with_config(&config.retention_policies),
            config,
            stats: TemporalStats::new(),
        }
    }

    /// Add evidence for temporal tracking
    pub fn add_evidence(&mut self, evidence: Evidence, evidence_type: EvidenceType) -> AutobahnResult<()> {
        let timed_evidence = TimedEvidence {
            evidence,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            current_strength: 1.0,
            original_strength: 1.0,
            evidence_type,
            decay_metadata: DecayMetadata {
                last_calculated: Utc::now(),
                decay_rate: 0.0,
                half_life_hours: 24.0,
                min_viable_strength: self.config.decay_params.min_viable_strength,
            },
        };

        self.decay_manager.active_evidence.push(timed_evidence);
        Ok(())
    }

    /// Update all temporal processing
    pub async fn update_temporal_processing(&mut self) -> AutobahnResult<()> {
        if !self.config.enable_temporal_processing {
            return Ok(());
        }

        // Update evidence decay
        self.update_evidence_decay().await?;

        // Detect temporal patterns
        self.detect_temporal_patterns().await?;

        // Update historical data
        self.update_historical_data().await?;

        // Clean up expired data
        self.cleanup_expired_data().await?;

        Ok(())
    }

    /// Update evidence decay calculations
    async fn update_evidence_decay(&mut self) -> AutobahnResult<()> {
        let now = Utc::now();
        let mut expired_indices = Vec::new();

        for (i, timed_evidence) in self.decay_manager.active_evidence.iter_mut().enumerate() {
            let time_elapsed = now.signed_duration_since(timed_evidence.decay_metadata.last_calculated);
            let hours_elapsed = time_elapsed.num_minutes() as f64 / 60.0;

            if hours_elapsed > 0.0 {
                let decay_function = self.decay_manager.decay_functions
                    .get(&timed_evidence.evidence_type)
                    .unwrap_or(&self.config.decay_params.default_decay);

                let new_strength = self.calculate_decay(
                    timed_evidence.current_strength,
                    hours_elapsed,
                    decay_function,
                );

                timed_evidence.current_strength = new_strength;
                timed_evidence.decay_metadata.last_calculated = now;
                timed_evidence.updated_at = now;

                // Mark for removal if below minimum viable strength
                if new_strength < timed_evidence.decay_metadata.min_viable_strength {
                    expired_indices.push(i);
                }

                self.stats.decay_calculations += 1;
            }
        }

        // Remove expired evidence
        for &index in expired_indices.iter().rev() {
            self.decay_manager.active_evidence.remove(index);
        }

        Ok(())
    }

    /// Calculate decay based on function type
    fn calculate_decay(&self, current_strength: f64, hours_elapsed: f64, decay_function: &DecayFunction) -> f64 {
        match decay_function {
            DecayFunction::Exponential { lambda } => {
                current_strength * (-lambda * hours_elapsed).exp()
            }
            DecayFunction::Linear { rate } => {
                (current_strength - rate * hours_elapsed).max(0.0)
            }
            DecayFunction::PowerLaw { alpha } => {
                if hours_elapsed > 0.0 {
                    current_strength * hours_elapsed.powf(-alpha)
                } else {
                    current_strength
                }
            }
            DecayFunction::StepFunction { threshold_hours, drop_factor } => {
                if hours_elapsed >= *threshold_hours {
                    current_strength * drop_factor
                } else {
                    current_strength
                }
            }
            DecayFunction::Custom { phases } => {
                let mut remaining_time = hours_elapsed;
                let mut strength = current_strength;

                for phase in phases {
                    if remaining_time <= 0.0 {
                        break;
                    }

                    let phase_time = remaining_time.min(phase.duration_hours);
                    strength = (strength - phase.decay_rate * phase_time).max(phase.min_strength);
                    remaining_time -= phase_time;
                }

                strength
            }
        }
    }

    /// Detect temporal patterns in data
    async fn detect_temporal_patterns(&mut self) -> AutobahnResult<()> {
        // Implementation would analyze time series data for patterns
        // This is a simplified version
        self.stats.patterns_detected += 1;
        Ok(())
    }

    /// Update historical data
    async fn update_historical_data(&mut self) -> AutobahnResult<()> {
        // Create periodic snapshots
        let now = Utc::now();
        let should_create_snapshot = self.historical_manager.snapshots.is_empty() ||
            now.signed_duration_since(
                self.historical_manager.snapshots.back().unwrap().timestamp
            ).num_hours() >= self.config.retention_policies.snapshot_interval_hours as i64;

        if should_create_snapshot {
            let snapshot = self.create_system_snapshot();
            self.historical_manager.snapshots.push_back(snapshot);
            self.stats.snapshots_created += 1;

            // Maintain maximum snapshot count
            while self.historical_manager.snapshots.len() > self.config.retention_policies.max_snapshots {
                self.historical_manager.snapshots.pop_front();
            }
        }

        Ok(())
    }

    /// Create system snapshot
    fn create_system_snapshot(&self) -> HistoricalSnapshot {
        HistoricalSnapshot {
            timestamp: Utc::now(),
            system_state: SystemState {
                active_modules: vec!["temporal_processor".to_string()],
                energy_state: EnergyState::new(1000.0),
                processing_load: 0.5,
                memory_usage: MemoryUsage {
                    total_mb: 512.0,
                    by_component: HashMap::new(),
                    peak_mb: 768.0,
                },
            },
            processing_metrics: ProcessingMetrics {
                requests_processed: self.stats.decay_calculations,
                avg_response_time_ms: self.stats.avg_processing_time_ms,
                error_rate: 0.01,
                throughput: 100.0,
            },
            evidence_summary: EvidenceSummary {
                total_evidence: self.decay_manager.active_evidence.len() as u32,
                by_type: HashMap::new(),
                avg_strength: self.calculate_average_evidence_strength(),
                expired_evidence: 0,
            },
        }
    }

    /// Calculate average evidence strength
    fn calculate_average_evidence_strength(&self) -> f64 {
        if self.decay_manager.active_evidence.is_empty() {
            return 0.0;
        }

        let total_strength: f64 = self.decay_manager.active_evidence
            .iter()
            .map(|e| e.current_strength)
            .sum();

        total_strength / self.decay_manager.active_evidence.len() as f64
    }

    /// Clean up expired data
    async fn cleanup_expired_data(&mut self) -> AutobahnResult<()> {
        let cutoff_date = Utc::now() - Duration::days(self.config.retention_policies.max_time_series_age_days as i64);

        // Clean up time series data
        for time_series in self.historical_manager.time_series_data.values_mut() {
            time_series.data.retain(|point| point.timestamp >= cutoff_date);
        }

        Ok(())
    }

    /// Get temporal processing statistics
    pub fn get_stats(&self) -> &TemporalStats {
        &self.stats
    }

    /// Get active evidence count
    pub fn get_active_evidence_count(&self) -> usize {
        self.decay_manager.active_evidence.len()
    }

    /// Get detected patterns
    pub fn get_detected_patterns(&self) -> &Vec<TemporalPattern> {
        &self.pattern_analyzer.detected_patterns
    }
}

impl TemporalProcessor for TemporalProcessorEngine {
    fn apply_decay(&self, evidence: &Evidence, time_elapsed: f64) -> f64 {
        // Default exponential decay
        let lambda = 0.1; // Default decay constant
        (-lambda * time_elapsed).exp()
    }

    fn calculate_temporal_strength(&self, evidence: &Evidence) -> f64 {
        // Find evidence in active tracking
        for timed_evidence in &self.decay_manager.active_evidence {
            if std::ptr::eq(&timed_evidence.evidence, evidence) {
                return timed_evidence.current_strength;
            }
        }
        0.0 // Evidence not found or expired
    }

    fn update_temporal_state(&mut self, new_evidence: Vec<Evidence>) {
        for evidence in new_evidence {
            let _ = self.add_evidence(evidence, EvidenceType::Experimental);
        }
    }

    fn predict_decay(&self, evidence: &Evidence, future_time: f64) -> f64 {
        let current_strength = self.calculate_temporal_strength(evidence);
        self.apply_decay(evidence, future_time) * current_strength
    }
}

// Implementation for sub-components
impl EvidenceDecayManager {
    fn new() -> Self {
        let mut decay_functions = HashMap::new();
        
        // Set default decay functions for different evidence types
        decay_functions.insert(EvidenceType::Literature, DecayFunction::Exponential { lambda: 0.05 });
        decay_functions.insert(EvidenceType::Experimental, DecayFunction::Exponential { lambda: 0.02 });
        decay_functions.insert(EvidenceType::Clinical, DecayFunction::Exponential { lambda: 0.03 });
        decay_functions.insert(EvidenceType::Computational, DecayFunction::Linear { rate: 0.1 });
        decay_functions.insert(EvidenceType::Expert, DecayFunction::PowerLaw { alpha: 0.5 });
        decay_functions.insert(EvidenceType::Sensor, DecayFunction::Exponential { lambda: 0.2 });
        decay_functions.insert(EvidenceType::UserInteraction, DecayFunction::Exponential { lambda: 0.15 });

        Self {
            decay_functions,
            active_evidence: Vec::new(),
            decay_params: DecayParameters::default(),
        }
    }

    fn with_config(config: &DecayParameters) -> Self {
        let mut manager = Self::new();
        manager.decay_params = config.clone();
        manager
    }
}

impl TemporalPatternAnalyzer {
    fn new() -> Self {
        Self {
            pattern_detectors: Vec::new(),
            detected_patterns: Vec::new(),
            pattern_config: PatternConfig::default(),
        }
    }

    fn with_config(config: &PatternConfig) -> Self {
        let mut analyzer = Self::new();
        analyzer.pattern_config = config.clone();
        analyzer
    }
}

impl HistoricalDataManager {
    fn new() -> Self {
        Self {
            time_series_data: HashMap::new(),
            snapshots: VecDeque::new(),
            retention_policies: RetentionPolicies::default(),
        }
    }

    fn with_config(config: &RetentionPolicies) -> Self {
        let mut manager = Self::new();
        manager.retention_policies = config.clone();
        manager
    }
}

impl TemporalStats {
    fn new() -> Self {
        Self {
            decay_calculations: 0,
            patterns_detected: 0,
            snapshots_created: 0,
            data_points_processed: 0,
            avg_processing_time_ms: 0.0,
        }
    }
}

// Default implementations
impl Default for DecayParameters {
    fn default() -> Self {
        Self {
            default_decay: DecayFunction::Exponential { lambda: 0.1 },
            min_viable_strength: 0.01,
            calculation_interval_minutes: 60,
            adaptive_decay: true,
        }
    }
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            max_patterns: 100,
            detection_interval_minutes: 30,
            min_confidence: 0.7,
        }
    }
}

impl Default for RetentionPolicies {
    fn default() -> Self {
        Self {
            max_time_series_age_days: 365,
            max_snapshots: 1000,
            snapshot_interval_hours: 1,
            compression_settings: CompressionSettings {
                enable_compression: true,
                algorithm: CompressionAlgorithm::Adaptive,
                compression_threshold_days: 30,
            },
        }
    }
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            decay_params: DecayParameters::default(),
            pattern_config: PatternConfig::default(),
            retention_policies: RetentionPolicies::default(),
            enable_temporal_processing: true,
        }
    }
}

impl Default for TemporalProcessorEngine {
    fn default() -> Self {
        Self::new()
    }
} 