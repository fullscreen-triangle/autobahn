// src/lib.rs
//! # Oscillatory Bio-Metabolic RAG System
//! 
//! This library implements a biologically-inspired Retrieval-Augmented Generation system
//! based on two fundamental theorems:
//! 
//! 1. **Oscillatory Entropy Theorem**: Entropy represents the statistical distribution 
//!    of oscillation termination points, with biological systems optimizing these endpoints
//!    for maximum information processing efficiency.
//! 
//! 2. **Membrane Quantum Computation Theorem**: Biological membranes function as 
//!    room-temperature quantum computers through Environment-Assisted Quantum Transport (ENAQT),
//!    making life a thermodynamic inevitability rather than an improbable accident.
//! 
//! The system operates across 10 nested hierarchy levels, from quantum oscillations (10⁻⁴⁴ s)
//! to cosmic patterns (10¹³ s), managing ATP resources through quantum-enhanced metabolic
//! pathways while maintaining oscillatory coherence across all scales.

pub mod error;
pub mod oscillatory;
pub mod quantum;
pub mod atp;
pub mod hierarchy;
pub mod biological;
pub mod entropy;
pub mod adversarial;
pub mod models;
pub mod rag;
pub mod utils;

// Re-export main types for convenience
pub use error::{OscillatoryError, Result};
pub use oscillatory::{
    OscillationState, OscillationPhase, OscillationProfile, UniversalOscillator
};
pub use quantum::{
    QuantumMembraneState, ENAQTProcessor, QuantumOscillatoryProfile
};
pub use atp::{
    ATPState, OscillatoryATPManager, QuantumATPManager, MetabolicMode
};
pub use hierarchy::{
    HierarchyLevel, HierarchyResult, NestedHierarchyProcessor
};
pub use rag::OscillatoryBioMetabolicRAG;

use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Utc};
use nalgebra::{DVector, DMatrix};
use serde::{Deserialize, Serialize};

// src/error.rs
//! Comprehensive error handling for the oscillatory bio-metabolic system.
//! Errors are categorized by their biological and physical origins.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum OscillatoryError {
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
    #[error("Configuration error: {parameter} = {value} is invalid")]
    ConfigurationError { parameter: String, value: String },
    
    #[error("Resource exhaustion: {resource} depleted")]
    ResourceExhaustion { resource: String },
    
    #[error("System shutdown initiated: {reason}")]
    SystemShutdown { reason: String },
    
    // External Integration Errors
    #[error("Serialization error: {message}")]
    SerializationError { message: String },
    
    #[error("Network error: {message}")]
    NetworkError { message: String },
    
    #[error("Database error: {message}")]
    DatabaseError { message: String },
}

impl From<serde_json::Error> for OscillatoryError {
    fn from(err: serde_json::Error) -> Self {
        OscillatoryError::SerializationError {
            message: err.to_string(),
        }
    }
}

impl From<reqwest::Error> for OscillatoryError {
    fn from(err: reqwest::Error) -> Self {
        OscillatoryError::NetworkError {
            message: err.to_string(),
        }
    }
}

pub type Result<T> = std::result::Result<T, OscillatoryError>;

// Utility functions for error context
impl OscillatoryError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Recoverable errors that can be retried or handled gracefully
            OscillatoryError::InsufficientATP { .. } => true,
            OscillatoryError::ModelTimeout { .. } => true,
            OscillatoryError::NetworkError { .. } => true,
            OscillatoryError::DesynchronizationError { .. } => true,
            OscillatoryError::CoherenceLoss { .. } => true,
            
            // Non-recoverable errors that indicate fundamental problems
            OscillatoryError::SystemShutdown { .. } => false,
            OscillatoryError::MembraneIntegrityFailure { .. } => false,
            OscillatoryError::RadicalDamageThreshold { .. } => false,
            OscillatoryError::UnsupportedHierarchyLevel { .. } => false,
            
            // Context-dependent errors
            _ => true,
        }
    }
    
    pub fn severity_level(&self) -> ErrorSeverity {
        match self {
            OscillatoryError::SystemShutdown { .. } => ErrorSeverity::Critical,
            OscillatoryError::MembraneIntegrityFailure { .. } => ErrorSeverity::Critical,
            OscillatoryError::RadicalDamageThreshold { .. } => ErrorSeverity::High,
            OscillatoryError::InsufficientATP { .. } => ErrorSeverity::High,
            OscillatoryError::ModelSelectionFailure => ErrorSeverity::Medium,
            OscillatoryError::CoherenceLoss { .. } => ErrorSeverity::Medium,
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

// src/oscillatory/mod.rs
//! Core oscillatory dynamics implementation based on the Universal Oscillation Equation:
//! d²y/dt² + γ(dy/dt) + ω²y = F(t)
//! 
//! This module implements the fundamental oscillatory behavior that underlies all
//! biological information processing, from quantum coherence to cosmic patterns.

pub mod types;
pub mod equations;
pub mod dynamics;
pub mod entropy;

use crate::error::{OscillatoryError, Result};
use nalgebra::{DVector, DMatrix};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationState {
    /// Current amplitude of oscillation
    pub amplitude: f64,
    /// Fundamental frequency in Hz
    pub frequency: f64,
    /// Current phase in radians
    pub phase: f64,
    /// Damping coefficient (γ in universal equation)
    pub damping_coefficient: f64,
    /// Natural frequency (ω in universal equation)
    pub natural_frequency: f64,
    /// Position vector in phase space
    pub position: DVector<f64>,
    /// Velocity vector in phase space
    pub velocity: DVector<f64>,
    /// Timestamp of last update
    pub timestamp: DateTime<Utc>,
    /// External forcing function amplitude
    pub external_forcing_amplitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OscillationPhase {
    /// System is accelerating, building energy
    Acceleration,
    /// System at maximum amplitude/energy
    Peak,
    /// System is decaying, losing energy
    Decay,
    /// System at rest or minimum energy
    Equilibrium,
    /// System in resonance with external forcing
    Resonance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationProfile {
    /// Computational complexity measure
    pub complexity: f64,
    /// Current oscillation phase
    pub phase: OscillationPhase,
    /// Dominant frequency
    pub frequency: f64,
    /// Active hierarchy levels for processing
    pub hierarchy_levels: Vec<u8>,
    /// Strength of cross-scale coupling
    pub coupling_strength: f64,
    /// Distribution of oscillation termination points (entropy measure)
    pub entropy_distribution: HashMap<String, f64>,
    /// Quality factor (Q = ω₀/γ)
    pub quality_factor: f64,
    /// Resonance bandwidth
    pub bandwidth: f64,
}

impl OscillationProfile {
    pub fn new(complexity: f64, frequency: f64) -> Self {
        Self {
            complexity,
            phase: OscillationPhase::Equilibrium,
            frequency,
            hierarchy_levels: vec![6], // Default to cognitive level
            coupling_strength: 0.5,
            entropy_distribution: HashMap::new(),
            quality_factor: 10.0, // Default Q factor
            bandwidth: frequency / 10.0,
        }
    }
    
    /// Calculate information content based on entropy distribution
    pub fn calculate_information_content(&self) -> f64 {
        let mut entropy = 0.0;
        let total_probability: f64 = self.entropy_distribution.values().sum();
        
        if total_probability > 0.0 {
            for probability in self.entropy_distribution.values() {
                if *probability > 0.0 {
                    let normalized_prob = probability / total_probability;
                    entropy -= normalized_prob * normalized_prob.log2();
                }
            }
        }
        
        entropy
    }
    
    /// Update entropy distribution with new termination point
    pub fn add_termination_point(&mut self, endpoint: String, probability: f64) {
        *self.entropy_distribution.entry(endpoint).or_insert(0.0) += probability;
        
        // Normalize distribution
        let total: f64 = self.entropy_distribution.values().sum();
        if total > 1.0 {
            for prob in self.entropy_distribution.values_mut() {
                *prob /= total;
            }
        }
    }
}

/// Universal Oscillator implementing the fundamental equation: d²y/dt² + γ(dy/dt) + ω²y = F(t)
#[derive(Debug, Clone)]
pub struct UniversalOscillator {
    /// Current oscillation state
    pub state: OscillationState,
    /// External forcing function F(t)
    pub external_forcing: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Historical states for analysis
    pub history: Vec<(DateTime<Utc>, OscillationState)>,
    /// Maximum history length to prevent memory bloat
    pub max_history_length: usize,
}

impl UniversalOscillator {
    /// Create new universal oscillator with specified parameters
    pub fn new(
        initial_amplitude: f64,
        natural_frequency: f64,
        damping_coefficient: f64,
        dimensions: usize,
    ) -> Self {
        let position = DVector::from_element(dimensions, initial_amplitude);
        let velocity = DVector::zeros(dimensions);
        
        Self {
            state: OscillationState {
                amplitude: initial_amplitude,
                frequency: natural_frequency,
                phase: 0.0,
                damping_coefficient,
                natural_frequency,
                position,
                velocity,
                timestamp: Utc::now(),
                external_forcing_amplitude: 0.0,
            },
            external_forcing: Box::new(|_t| 0.0), // Default: no external forcing
            history: Vec::new(),
            max_history_length: 10000,
        }
    }
    
    /// Create oscillator with custom forcing function
    pub fn with_forcing<F>(mut self, forcing_fn: F) -> Self 
    where 
        F: Fn(f64) -> f64 + Send + Sync + 'static 
    {
        self.external_forcing = Box::new(forcing_fn);
        self
    }
    
    /// Evolve the oscillator by time step dt using the Universal Oscillation Equation
    pub fn evolve(&mut self, dt: f64) -> Result<()> {
        if dt <= 0.0 {
            return Err(OscillatoryError::ConfigurationError {
                parameter: "dt".to_string(),
                value: dt.to_string(),
            });
        }
        
        // Get current time for forcing function
        let t = self.state.timestamp.timestamp() as f64;
        let forcing = (self.external_forcing)(t);
        self.state.external_forcing_amplitude = forcing;
        
        // Universal Oscillation Equation: d²y/dt² = -γ(dy/dt) - ω²y + F(t)
        let gamma = self.state.damping_coefficient;
        let omega_squared = self.state.natural_frequency.powi(2);
        
        // Calculate acceleration for each dimension
        let acceleration = -gamma * &self.state.velocity 
                          - omega_squared * &self.state.position
                          + DVector::from_element(self.state.position.len(), forcing);
        
        // Velocity Verlet integration for numerical stability
        // v(t + dt/2) = v(t) + a(t) * dt/2
        let velocity_half_step = &self.state.velocity + &acceleration * (dt / 2.0);
        
        // x(t + dt) = x(t) + v(t + dt/2) * dt
        self.state.position += &velocity_half_step * dt;
        
        // Recalculate acceleration at new position
        let new_acceleration = -gamma * &velocity_half_step 
                              - omega_squared * &self.state.position
                              + DVector::from_element(self.state.position.len(), forcing);
        
        // v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        self.state.velocity = velocity_half_step + new_acceleration * (dt / 2.0);
        
        // Update derived quantities
        self.state.amplitude = self.state.position.norm();
        
        // Calculate phase (for 1D case, use position and velocity)
        if self.state.position.len() > 0 {
            self.state.phase = if self.state.velocity.len() > 0 {
                self.state.position[0].atan2(self.state.velocity[0] / self.state.natural_frequency)
            } else {
                0.0
            };
        }
        
        // Update frequency based on current dynamics
        if self.state.velocity.norm() > 1e-10 {
            self.state.frequency = self.state.velocity.norm() / (2.0 * std::f64::consts::PI * self.state.amplitude.max(1e-10));
        }
        
        self.state.timestamp = Utc::now();
        
        // Store history
        self.history.push((self.state.timestamp, self.state.clone()));
        
        // Limit history size to prevent memory issues
        if self.history.len() > self.max_history_length {
            self.history.drain(..1000);
        }
        
        // Check for amplitude overflow
        if self.state.amplitude > 1e6 {
            return Err(OscillatoryError::AmplitudeOverflow {
                amplitude: self.state.amplitude,
                max_amplitude: 1e6,
            });
        }
        
        Ok(())
    }
    
    /// Determine current oscillation phase based on position and velocity
    pub fn calculate_phase(&self) -> OscillationPhase {
        let velocity_magnitude = self.state.velocity.norm();
        let position_magnitude = self.state.position.norm();
        let natural_freq = self.state.natural_frequency;
        
        // Define phase boundaries based on energy distribution
        let kinetic_energy = 0.5 * velocity_magnitude.powi(2);
        let potential_energy = 0.5 * natural_freq.powi(2) * position_magnitude.powi(2);
        let total_energy = kinetic_energy + potential_energy;
        
        if total_energy < 1e-10 {
            return OscillationPhase::Equilibrium;
        }
        
        let kinetic_fraction = kinetic_energy / total_energy;
        
        // Check for resonance first (growing amplitude)
        if self.is_resonating() {
            return OscillationPhase::Resonance;
        }
        
        // Phase classification based on energy distribution
        match kinetic_fraction {
            f if f > 0.8 => OscillationPhase::Acceleration, // Mostly kinetic energy
            f if f > 0.6 => OscillationPhase::Peak,         // High kinetic energy
            f if f > 0.2 => OscillationPhase::Decay,        // Balanced energy
            _ => OscillationPhase::Equilibrium,             // Mostly potential energy
        }
    }
    
    /// Check if the oscillator is in resonance (amplitude growing)
    fn is_resonating(&self) -> bool {
        if self.history.len() < 10 {
            return false;
        }
        
        // Get recent amplitude history
        let recent_amplitudes: Vec<f64> = self.history
            .iter()
            .rev()
            .take(10)
            .map(|(_, state)| state.amplitude)
            .collect();
        
        // Calculate amplitude growth rate
        let slope = self.calculate_amplitude_slope(&recent_amplitudes);
        
        // Resonance detected if amplitude is increasing significantly
        slope > 0.1 && self.state.external_forcing_amplitude.abs() > 0.01
    }
    
    /// Calculate slope of amplitude over time using linear regression
    fn calculate_amplitude_slope(&self, amplitudes: &[f64]) -> f64 {
        if amplitudes.len() < 2 {
            return 0.0;
        }
        
        let n = amplitudes.len() as f64;
        let sum_x: f64 = (0..amplitudes.len()).map(|i| i as f64).sum();
        let sum_y: f64 = amplitudes.iter().sum();
        let sum_xy: f64 = amplitudes.iter().enumerate()
            .map(|(i, &y)| i as f64 * y).sum();
        let sum_x_squared: f64 = (0..amplitudes.len()).map(|i| (i as f64).powi(2)).sum();
        
        let denominator = n * sum_x_squared - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return 0.0;
        }
        
        (n * sum_xy - sum_x * sum_y) / denominator
    }
    
    /// Calculate quality factor Q = ω₀/(2γ)
    pub fn quality_factor(&self) -> f64 {
        if self.state.damping_coefficient > 0.0 {
            self.state.natural_frequency / (2.0 * self.state.damping_coefficient)
        } else {
            f64::INFINITY
        }
    }
    
    /// Get oscillation energy
    pub fn total_energy(&self) -> f64 {
        let kinetic = 0.5 * self.state.velocity.norm_squared();
        let potential = 0.5 * self.state.natural_frequency.powi(2) * self.state.position.norm_squared();
        kinetic + potential
    }
    
    /// Synchronize with another oscillator (mutual coupling)
    pub fn synchronize_with(&mut self, other: &UniversalOscillator, coupling_strength: f64) -> Result<()> {
        if coupling_strength < 0.0 || coupling_strength > 1.0 {
            return Err(OscillatoryError::ENAQTCouplingFailure { 
                coupling: coupling_strength 
            });
        }
        
        // Calculate phase difference
        let phase_diff = other.state.phase - self.state.phase;
        
        // Apply coupling force proportional to phase difference
        let coupling_force = coupling_strength * phase_diff.sin();
        
        // Modify the external forcing to include coupling
        let current_forcing = self.state.external_forcing_amplitude;
        let new_forcing = current_forcing + coupling_force;
        
        // Update forcing amplitude
        self.state.external_forcing_amplitude = new_forcing;
        
        Ok(())
    }
}

// Implement Send + Sync for UniversalOscillator by providing a custom implementation
unsafe impl Send for UniversalOscillator {}
unsafe impl Sync for UniversalOscillator {}

// src/quantum/mod.rs
//! Quantum membrane computation implementation based on the Membrane Quantum Computation Theorem.
//! This module implements Environment-Assisted Quantum Transport (ENAQT) and demonstrates
//! how biological membranes function as room-temperature quantum computers.

pub mod membrane;
pub mod enaqt;
pub mod computation;
pub mod radicals;
pub mod coherence;

use crate::error::{OscillatoryError, Result};
use crate::oscillatory::{OscillationProfile, OscillationPhase};
use nalgebra::{Complex, DVector, DMatrix};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantum state of a biological membrane system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMembraneState {
    /// Quantum coherence time in femtoseconds (FMO complex: ~660 fs)
    pub coherence_time_fs: f64,
    /// Electron tunneling probability through membrane
    pub tunneling_probability: f64,
    /// ENAQT coupling strength (optimal ~0.4)
    pub enaqt_coupling_strength: f64,
    /// Operating temperature in Kelvin
    pub temperature_k: f64,
    /// Electron transport efficiency (biological: >90%)
    pub electron_transport_efficiency: f64,
    /// Rate of oxygen radical generation (quantum leakage)
    pub radical_generation_rate: f64,
    /// Membrane thickness in nanometers (typical: 3-5 nm)
    pub membrane_thickness_nm: f64,
    /// Complex quantum oscillation amplitudes
    pub quantum_oscillations: Vec<Complex<f64>>,
    /// Membrane potential in millivolts
    pub membrane_potential_mv: f64,
    /// Proton gradient strength
    pub proton_gradient: f64,
}

impl QuantumMembraneState {
    pub fn new(temperature_k: f64) -> Self {
        Self {
            coherence_time_fs: 660.0, // FMO complex experimental value
            tunneling_probability: 0.1,
            enaqt_coupling_strength: 0.4, // Optimal coupling from theorem
            temperature_k,
            electron_transport_efficiency: 0.95,
            radical_generation_rate: 1e-6, // Base quantum leakage rate
            membrane_thickness_nm: 4.0,
            quantum_oscillations: vec![Complex::new(1.0, 0.0); 10],
            membrane_potential_mv: -70.0, // Typical resting potential
            proton_gradient: 1.0,
        }
    }
    
    /// Calculate quantum advantage factor over classical systems
    pub fn quantum_advantage_factor(&self) -> f64 {
        // κ = k_quantum / k_classical = 1 / (1 + exp((E_a - ΔG) / k_B T))
        let activation_energy_ev = 0.5; // Typical activation barrier
        let delta_g_ev = 0.3; // Free energy change
        let kb_t_ev = 8.617e-5 * self.temperature_k; // Boltzmann constant × temperature
        
        let exponent = (activation_energy_ev - delta_g_ev) / kb_t_ev;
        1.0 / (1.0 + exponent.exp())
    }
    
    /// Check if membrane is in optimal quantum regime
    pub fn is_quantum_optimal(&self) -> bool {
        self.coherence_time_fs > 500.0 
            && self.electron_transport_efficiency > 0.9
            && self.enaqt_coupling_strength > 0.3
            && self.enaqt_coupling_strength < 0.6
    }
}

/// Environment-Assisted Quantum Transport processor
/// Implements the core theorem: environmental coupling enhances rather than destroys coherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ENAQTProcessor {
    /// System Hamiltonian (H_system)
    pub system_hamiltonian: DMatrix<Complex<f64>>,
    /// Environment Hamiltonian (H_environment)  
    pub environment_hamiltonian: DMatrix<Complex<f64>>,
    /// Interaction Hamiltonian (H_interaction)
    pub interaction_hamiltonian: DMatrix<Complex<f64>>,
    /// Optimized coupling strength
    pub coupling_optimization: f64,
    /// Coherence enhancement factor from environmental coupling
    pub coherence_enhancement_factor: f64,
    /// Spectral density of environmental modes
    pub environmental_spectral_density: Vec<f64>,
}

impl ENAQTProcessor {
    pub fn new(dimensions: usize) -> Self {
        Self {
            system_hamiltonian: DMatrix::zeros(dimensions, dimensions),
            environment_hamiltonian: DMatrix::zeros(dimensions, dimensions),
            interaction_ham
            interaction_hamiltonian: DMatrix::zeros(dimensions, dimensions),
            coupling_optimization: 0.0,
            coherence_enhancement_factor: 1.0,
            environmental_spectral_density: vec![0.0; dimensions],
        }
    }
    
    /// Calculate transport efficiency using ENAQT theorem: η = η₀ × (1 + αγ + βγ²)
    pub fn calculate_transport_efficiency(
        &self,
        coupling_strength: f64,
        temperature: f64,
    ) -> Result<f64> {
        // ENAQT enhancement coefficients (from biological measurements)
        let alpha = 0.8;  // Linear enhancement coefficient
        let beta = -0.2;  // Quadratic optimization coefficient
        let eta_0 = 0.85; // Base efficiency without environmental coupling
        
        let gamma = coupling_strength;
        
        // Verify coupling is in valid range
        if gamma < 0.0 || gamma > 1.0 {
            return Err(OscillatoryError::ENAQTCouplingFailure { coupling: gamma });
        }
        
        // ENAQT efficiency formula
        let efficiency = eta_0 * (1.0 + alpha * gamma + beta * gamma.powi(2));
        
        // Calculate optimal coupling: γ_optimal = α/(2|β|)
        let gamma_optimal = alpha / (2.0 * beta.abs());
        
        // Temperature correction factor (biological systems maintain efficiency at 300K)
        let temperature_factor = if temperature > 0.0 {
            (-0.01 * (temperature - 300.0).abs()).exp()
        } else {
            return Err(OscillatoryError::ConfigurationError {
                parameter: "temperature".to_string(),
                value: temperature.to_string(),
            });
        };
        
        // Store optimization information
        let final_efficiency = efficiency * temperature_factor;
        
        // Biological systems achieve >90% efficiency, artificial systems <40%
        Ok(final_efficiency.min(0.98)) // Cap at 98% (thermodynamic limit)
    }
    
    /// Calculate quantum tunneling probability: P = (16E(V₀-E)/V₀²) × exp(-2κa)
    pub fn calculate_quantum_tunneling_probability(
        &self,
        barrier_height_ev: f64,
        barrier_width_nm: f64,
        electron_energy_ev: f64,
    ) -> Result<f64> {
        if barrier_height_ev <= 0.0 || barrier_width_nm <= 0.0 {
            return Err(OscillatoryError::QuantumTunnelingFailure { 
                barrier_height: barrier_height_ev 
            });
        }
        
        let e = electron_energy_ev;
        let v0 = barrier_height_ev;
        let a = barrier_width_nm * 1e-9; // Convert to meters
        
        // Over-barrier transport (classical regime)
        if e >= v0 {
            return Ok(1.0);
        }
        
        // Quantum tunneling calculation
        // κ = √(2m(V₀-E)/ℏ²)
        let m_electron = 9.109e-31; // kg
        let hbar = 1.055e-34; // J⋅s
        let ev_to_joule = 1.602e-19;
        
        let energy_diff = (v0 - e) * ev_to_joule;
        let kappa = ((2.0 * m_electron * energy_diff) / (hbar * hbar)).sqrt();
        
        // Transmission coefficient
        let pre_factor = (16.0 * e * (v0 - e)) / (v0 * v0);
        let exponential_factor = (-2.0 * kappa * a).exp();
        
        let tunneling_prob = pre_factor * exponential_factor;
        
        Ok(tunneling_prob.min(1.0))
    }
    
    /// Calculate oxygen radical generation rate from quantum leakage
    /// d[O₂⁻]/dt = k_leak × [e⁻] × [O₂] × P_quantum
    pub fn calculate_radical_generation_rate(
        &self,
        electron_density: f64,
        oxygen_concentration: f64,
        quantum_leakage_probability: f64,
    ) -> Result<f64> {
        if electron_density < 0.0 || oxygen_concentration < 0.0 || quantum_leakage_probability < 0.0 {
            return Err(OscillatoryError::NegativeProbability { 
                probability: quantum_leakage_probability 
            });
        }
        
        // Rate constant for electron-oxygen interaction (typical biological value)
        let k_leak = 1e6; // s⁻¹ M⁻¹
        
        let radical_rate = k_leak * electron_density * oxygen_concentration * quantum_leakage_probability;
        
        Ok(radical_rate)
    }
    
    /// Calculate coherence time with environmental enhancement
    pub fn calculate_enhanced_coherence_time(
        &self,
        base_coherence_fs: f64,
        coupling_strength: f64,
        temperature: f64,
    ) -> Result<f64> {
        // Environmental enhancement of coherence (counter-intuitive but experimentally verified)
        let enhancement_factor = if coupling_strength > 0.2 && coupling_strength < 0.6 {
            1.0 + 0.5 * coupling_strength // Optimal coupling enhances coherence
        } else {
            1.0 / (1.0 + coupling_strength) // Sub-optimal coupling reduces coherence
        };
        
        // Temperature decoherence factor
        let temp_factor = (300.0 / temperature).sqrt();
        
        let enhanced_coherence = base_coherence_fs * enhancement_factor * temp_factor;
        
        Ok(enhanced_coherence)
    }
    
    /// Optimize ENAQT coupling for maximum efficiency
    pub fn optimize_coupling(&mut self, target_efficiency: f64) -> Result<f64> {
        let alpha = 0.8;
        let beta = -0.2;
        
        // Optimal coupling from calculus: dη/dγ = 0 → γ_opt = α/(2|β|)
        let optimal_coupling = alpha / (2.0 * beta.abs());
        
        self.coupling_optimization = optimal_coupling;
        
        // Calculate maximum achievable efficiency
        let eta_0 = 0.85;
        let max_efficiency = eta_0 * (1.0 + alpha * optimal_coupling + beta * optimal_coupling.powi(2));
        
        if max_efficiency < target_efficiency {
            log::warn!(
                "Target efficiency {:.2} not achievable. Maximum: {:.2}", 
                target_efficiency, max_efficiency
            );
        }
        
        Ok(optimal_coupling)
    }
}

/// Combined quantum-oscillatory profile integrating both frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumOscillatoryProfile {
    /// Base oscillatory dynamics
    pub base_oscillation: OscillationProfile,
    /// Quantum membrane state
    pub quantum_membrane_state: QuantumMembraneState,
    /// ENAQT processor for quantum calculations
    pub enaqt_processor: ENAQTProcessor,
    /// ATP synthase efficiency (quantum-enhanced)
    pub atp_synthase_efficiency: f64,
    /// Metabolic quantum burden factor
    pub metabolic_quantum_burden: f64,
    /// Predicted longevity based on quantum damage
    pub longevity_prediction: Option<f64>,
    /// Cross-hierarchy quantum coupling matrix
    pub hierarchy_coupling_matrix: DMatrix<f64>,
}

impl QuantumOscillatoryProfile {
    pub fn new(
        base_oscillation: OscillationProfile,
        temperature_k: f64,
    ) -> Self {
        let quantum_membrane_state = QuantumMembraneState::new(temperature_k);
        let enaqt_processor = ENAQTProcessor::new(10);
        
        // Initialize hierarchy coupling matrix (10x10 for 10 hierarchy levels)
        let mut coupling_matrix = DMatrix::zeros(10, 10);
        
        // Set coupling strengths based on frequency separation
        for i in 0..10 {
            for j in 0..10 {
                if i != j {
                    let freq_ratio = ((i + 1) as f64 / (j + 1) as f64).ln().abs();
                    let coupling = 0.8 * (-freq_ratio / 5.0).exp();
                    coupling_matrix[(i, j)] = coupling;
                }
            }
        }
        
        Self {
            base_oscillation,
            quantum_membrane_state,
            enaqt_processor,
            atp_synthase_efficiency: 0.95, // Default high efficiency
            metabolic_quantum_burden: 1.0,
            longevity_prediction: None,
            hierarchy_coupling_matrix: coupling_matrix,
        }
    }
    
    /// Calculate quantum-enhanced ATP cost considering oscillatory and quantum effects
    pub fn calculate_quantum_enhanced_atp_cost(
        &self,
        base_cost: f64,
        metabolic_demand: f64,
    ) -> Result<f64> {
        // Quantum efficiency modulation
        let quantum_efficiency = self.enaqt_processor.calculate_transport_efficiency(
            self.quantum_membrane_state.enaqt_coupling_strength,
            self.quantum_membrane_state.temperature_k,
        )?;
        
        // Metabolic mode factor based on demand
        let metabolic_factor = if metabolic_demand > 10.0 {
            // Sustained flight metabolism - maximum efficiency
            0.6 * quantum_efficiency
        } else if self.quantum_membrane_state.temperature_k < 290.0 {
            // Cold-blooded advantage - temperature-dependent efficiency
            let temp_advantage = (300.0 - self.quantum_membrane_state.temperature_k) / 20.0;
            0.8 * quantum_efficiency * (1.0 + temp_advantage)
        } else {
            // Mammalian quantum burden - higher cost due to warm-blooded metabolism
            1.2 / quantum_efficiency
        };
        
        // Oscillatory phase modulation
        let phase_factor = match self.base_oscillation.phase {
            OscillationPhase::Peak => 0.7,         // Maximum quantum coherence
            OscillationPhase::Resonance => 0.5,    // Optimal ENAQT coupling
            OscillationPhase::Acceleration => 1.3, // Building coherence costs energy
            OscillationPhase::Decay => 1.8,        // Losing coherence is inefficient
            OscillationPhase::Equilibrium => 1.0,  // Baseline efficiency
        };
        
        // Coupling strength optimization
        let coupling_factor = if self.quantum_membrane_state.enaqt_coupling_strength > 0.3 
                                && self.quantum_membrane_state.enaqt_coupling_strength < 0.6 {
            0.9 // Optimal coupling range
        } else {
            1.1 // Sub-optimal coupling penalty
        };
        
        let final_cost = base_cost * metabolic_factor * phase_factor * coupling_factor;
        
        Ok(final_cost)
    }
    
    /// Predict longevity enhancement based on quantum mechanical principles
    pub fn predict_longevity_enhancement(&mut self) -> Result<f64> {
        let base_lifespan = 80.0; // Human baseline years
        
        // Temperature factor (ectothermic advantage from theorem)
        let temp_factor = if self.quantum_membrane_state.temperature_k < 300.0 {
            let temp_reduction = (300.0 - self.quantum_membrane_state.temperature_k) / 10.0;
            // Exponential longevity increase with temperature reduction
            (temp_reduction / 2.0).exp()
        } else {
            1.0
        };
        
        // Metabolic efficiency factor (sustained flight advantage)
        let efficiency_factor = if self.atp_synthase_efficiency > 0.9 {
            2.5 // Sustained flight metabolism advantage
        } else if self.atp_synthase_efficiency > 0.8 {
            1.5 // Good efficiency
        } else {
            1.0 // Standard mammalian burden
        };
        
        // Radical generation burden (quantum mechanical aging)
        let radical_factor = 1.0 / (1.0 + self.quantum_membrane_state.radical_generation_rate * 0.1);
        
        // Quantum coherence advantage
        let coherence_factor = 1.0 + (self.quantum_membrane_state.coherence_time_fs / 1000.0).ln().max(0.0);
        
        // ENAQT coupling optimization bonus
        let coupling_bonus = if self.quantum_membrane_state.enaqt_coupling_strength > 0.35 
                               && self.quantum_membrane_state.enaqt_coupling_strength < 0.55 {
            1.2 // Optimal coupling provides longevity bonus
        } else {
            1.0
        };
        
        let predicted_lifespan = base_lifespan * temp_factor * efficiency_factor 
                               * radical_factor * coherence_factor * coupling_bonus;
        
        self.longevity_prediction = Some(predicted_lifespan);
        
        log::info!(
            "Longevity prediction: {:.1} years (temp: {:.1}x, efficiency: {:.1}x, radicals: {:.1}x, coherence: {:.1}x)",
            predicted_lifespan, temp_factor, efficiency_factor, radical_factor, coherence_factor
        );
        
        Ok(predicted_lifespan)
    }
    
    /// Calculate quantum advantage over classical systems
    pub fn quantum_advantage(&self) -> f64 {
        let classical_efficiency = 0.4; // Maximum classical efficiency
        let quantum_efficiency = self.quantum_membrane_state.electron_transport_efficiency;
        
        quantum_efficiency / classical_efficiency
    }
    
    /// Check if system is operating in optimal quantum regime
    pub fn is_quantum_optimal(&self) -> bool {
        self.quantum_membrane_state.is_quantum_optimal()
            && self.base_oscillation.quality_factor > 5.0
            && self.atp_synthase_efficiency > 0.85
    }
}

// src/atp/mod.rs
//! ATP resource management system implementing quantum-enhanced metabolic pathways.
//! Based on the principle that ATP synthase functions as a biological quantum computer.

pub mod manager;
pub mod quantum_manager;
pub mod optimization;
pub mod economics;

use crate::error::{OscillatoryError, Result};
use crate::oscillatory::{OscillationPhase, OscillationProfile};
use crate::biological::BiologicalLayer;
use crate::quantum::QuantumMembraneState;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// ATP state tracking with quantum considerations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATPState {
    /// Current ATP available
    pub current: f64,
    /// Maximum ATP capacity
    pub maximum: f64,
    /// Reserved ATP for critical functions
    pub reserved: f64,
    /// Emergency threshold for anaerobic activation
    pub emergency_threshold: f64,
    /// ATP regeneration rate (quantum-enhanced)
    pub regeneration_rate: f64,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
    /// Quantum efficiency factor
    pub quantum_efficiency: f64,
    /// Accumulated radical damage
    pub radical_damage: f64,
}

impl ATPState {
    pub fn new(maximum: f64) -> Self {
        Self {
            current: maximum * 0.75,     // Start at 75% capacity
            maximum,
            reserved: maximum * 0.15,    // Reserve 15% for emergencies
            emergency_threshold: maximum * 0.1, // Emergency at 10%
            regeneration_rate: maximum * 0.025,  // 2.5% per minute base rate
            last_update: Utc::now(),
            quantum_efficiency: 0.95,    // Default quantum efficiency
            radical_damage: 0.0,
        }
    }
    
    /// Calculate available ATP (current - reserved)
    pub fn available(&self) -> f64 {
        (self.current - self.reserved).max(0.0)
    }
    
    /// Calculate ATP percentage of maximum
    pub fn percentage(&self) -> f64 {
        (self.current / self.maximum) * 100.0
    }
    
    /// Check if ATP is at critical levels
    pub fn is_critical(&self) -> bool {
        self.current < self.emergency_threshold
    }
    
    /// Check if radical damage is affecting ATP production
    pub fn is_damage_critical(&self) -> bool {
        self.radical_damage > 50.0 // Arbitrary damage threshold
    }
}

/// Metabolic modes based on quantum mechanical principles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetabolicMode {
    /// Sustained flight metabolism - maximum quantum efficiency
    SustainedFlight {
        efficiency_boost: f64,
        radical_suppression: f64,
    },
    /// Cold-blooded metabolism - temperature advantage
    ColdBlooded {
        temperature_advantage: f64,
        metabolic_reduction: f64,
    },
    /// Mammalian burden - high temperature, high metabolic cost
    MammalianBurden {
        quantum_cost_multiplier: f64,
        radical_generation_rate: f64,
    },
    /// Anaerobic emergency - lactate pathway activation
    AnaerobicEmergency {
        lactate_pathway_active: bool,
        efficiency_penalty: f64,
    },
}

/// Layer-specific ATP requirements with quantum considerations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerResourceRequirement {
    pub layer: BiologicalLayer,
    pub base_atp_cost: f64,
    pub complexity_multiplier: f64,
    pub minimum_atp: f64,
    pub maximum_atp: f64,
    pub expected_yield: f64,
    pub oscillation_sensitivity: f64,
    pub quantum_enhancement_factor: f64,
}

/// Basic oscillatory ATP manager
pub struct OscillatoryATPManager {
    state: Arc<RwLock<ATPState>>,
    consumption_history: Arc<RwLock<Vec<(DateTime<Utc>, String, f64, f64)>>>,
    layer_requirements: HashMap<BiologicalLayer, LayerResourceRequirement>,
    cosmic_oscillation_phase: Arc<RwLock<OscillationPhase>>,
}

impl OscillatoryATPManager {
    pub fn new(maximum_atp: f64) -> Self {
        let mut layer_requirements = HashMap::new();
        
        // Context layer - basic information processing
        layer_requirements.insert(
            BiologicalLayer::Context,
            LayerResourceRequirement {
                layer: BiologicalLayer::Context,
                base_atp_cost: 100.0,
                complexity_multiplier: 1.2,
                minimum_atp: 50.0,
                maximum_atp: 300.0,
                expected_yield: 0.7,
                oscillation_sensitivity: 0.3,
                quantum_enhancement_factor: 1.1,
            }
        );
        
        // Reasoning layer - complex logical processing
        layer_requirements.insert(
            BiologicalLayer::Reasoning,
            LayerResourceRequirement {
                layer: BiologicalLayer::Reasoning,
                base_atp_cost: 300.0,
                complexity_multiplier: 1.8,
                minimum_atp: 150.0,
                maximum_atp: 800.0,
                expected_yield: 0.85,
                oscillation_sensitivity: 0.6,
                quantum_enhancement_factor: 1.3,
            }
        );
        
        // Intuition layer - highest-level pattern recognition
        layer_requirements.insert(
            BiologicalLayer::Intuition,
            LayerResourceRequirement {
                layer: BiologicalLayer::Intuition,
                base_atp_cost: 500.0,
                complexity_multiplier: 2.5,
                minimum_atp: 300.0,
                maximum_atp: 1200.0,
                expected_yield: 0.95,
                oscillation_sensitivity: 0.9,
                quantum_enhancement_factor: 1.5,
            }
        );
        
        Self {
            state: Arc::new(RwLock::new(ATPState::new(maximum_atp))),
            consumption_history: Arc::new(RwLock::new(Vec::new())),
            layer_requirements,
            cosmic_oscillation_phase: Arc::new(RwLock::new(OscillationPhase::Equilibrium)),
        }
    }
    
    /// Calculate oscillatory ATP cost with phase and complexity modulation
    pub async fn calculate_oscillatory_atp_cost(
        &self,
        layer: BiologicalLayer,
        query_complexity: f64,
        oscillation_profile: &OscillationProfile,
    ) -> Result<f64> {
        let req = self.layer_requirements.get(&layer)
            .ok_or(OscillatoryError::BiologicalLayerFailure { 
                layer: format!("{:?}", layer) 
            })?;
        
        // Base cost calculation
        let base_cost = req.base_atp_cost;
        let complexity_cost = base_cost * query_complexity.powf(req.complexity_multiplier);
        
        // Oscillatory phase modulation
        let phase_multiplier = match oscillation_profile.phase {
            OscillationPhase::Acceleration => 2.5, // High energy during buildup
            OscillationPhase::Peak => 0.8,         // Maximum efficiency at peak
            OscillationPhase::Decay => 1.5,        // Increased cost during decay
            OscillationPhase::Equilibrium => 1.0,  // Baseline cost
            OscillationPhase::Resonance => 0.6,    // Highly efficient during resonance
        };
        
        // Frequency modulation (biological systems have optimal frequency ranges)
        let frequency_multiplier = if oscillation_profile.frequency > 10.0 {
            1.3 // Higher frequency requires more energy coordination
        } else if oscillation_profile.frequency < 0.1 {
            1.2 // Very low frequency is also less efficient
        } else {
            1.0 // Optimal frequency range
        };
        
        // Coupling strength modulation
        let coupling_multiplier = 1.0 + (oscillation_profile.coupling_strength - 0.5) * 0.4;
        
        // Quality factor consideration (higher Q = more efficient)
        let quality_multiplier = 1.0 / (1.0 + oscillation_profile.quality_factor / 20.0);
        
        let total_cost = complexity_cost * phase_multiplier * frequency_multiplier 
                        * coupling_multiplier * quality_multiplier;
        
        // Apply quantum enhancement
        let quantum_enhanced_cost = total_cost / req.quantum_enhancement_factor;
        
        // Apply bounds
        Ok(quantum_enhanced_cost.clamp(req.minimum_atp, req.maximum_atp))
    }
    
    /// Optimize ATP allocation across multiple layers
    pub async fn optimize_atp_allocation(
        &self,
        oscillation_profile: &OscillationProfile,
        priority_layers: Vec<BiologicalLayer>,
    ) -> Result<HashMap<BiologicalLayer, f64>> {
        let state = self.state.read().await;
        let available_atp = state.available();
        drop(state);
        
        let mut requirements = HashMap::new();
        let mut total_required = 0.0;
        
        // Calculate requirements for each layer
        for layer in &priority_layers {
            let req_atp = self.calculate_oscillatory_atp_cost(
                *layer,
                oscillation_profile.complexity,
                oscillation_profile,
            ).await?;
            
            requirements.insert(*layer, req_atp);
            total_required += req_atp;
        }
        
        if total_required <= available_atp {
            // Sufficient ATP - allocate as requested
            Ok(requirements)
        } else {
            // Insufficient ATP - optimize allocation using efficiency-based prioritization
            self.optimize_constrained_allocation(requirements, available_atp).await
        }
    }
    
    /// Optimize ATP allocation when resources are constrained
    async fn optimize_constrained_allocation(
        &self,
        requirements: HashMap<BiologicalLayer, f64>,
        available_atp: f64,
    ) -> Result<HashMap<BiologicalLayer, f64>> {
        let mut allocations = HashMap::new();
        let mut remaining_atp = available_atp;
        
        // Create efficiency-sorted list (yield per ATP cost)
        let mut layer_efficiency: Vec<(BiologicalLayer, f64, f64)> = requirements
            .iter()
            .map(|(layer, atp_cost)| {
                let req = &self.layer_requirements[layer];
                let efficiency = req.expected_yield / atp_cost;
                (*layer, *atp_cost, efficiency)
            })
            .collect();
        
        // Sort by efficiency (descending) - allocate to most efficient layers first
        layer_efficiency.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        // Allocate ATP starting with most efficient layers
        for (layer, requested_atp, _efficiency) in layer_efficiency {
            let req = &self.layer_requirements[&layer];
            
            if remaining_atp >= req.minimum_atp {
                let allocated = requested_atp.min(remaining_atp);
                allocations.insert(layer, allocated);
                remaining_atp -= allocated;
                
                log::debug!(
                    "Allocated {:.1} ATP to {:?} (efficiency: {:.3})",
                    allocated, layer, _efficiency
                );
            } else {
                allocations.insert(layer, 0.0);
                log::warn!("Insufficient ATP for {:?} (need {:.1}, have {:.1})", 
                          layer, req.minimum_atp, remaining_atp);
            }
        }
        
        Ok(allocations)
    }
    
    /// Consume ATP for a specific operation
    pub async fn consume_atp(
        &self,
        layer: BiologicalLayer,
        amount: f64,
        operation: &str,
    ) -> Result<bool> {
        let mut state = self.state.write().await;
        
        if state.available() >= amount {
            state.current -= amount;
            
            // Record consumption in history
            let mut history = self.consumption_history.write().await;
            history.push((
                Utc::now(),
                format!("{:?}:{}", layer, operation),
                amount,
                state.current,
            ));
            
            // Limit history size to prevent memory bloat
            if history.len() > 10000 {
                history.drain(..1000);
            }
            
            log::debug!("Consumed {:.1} ATP for {:?}:{}, remaining: {:.1}", 
                       amount, layer, operation, state.current);
            
            Ok(true)
        } else {
            // Handle ATP shortage
            drop(state);
            self.handle_atp_shortage(layer, amount, operation).await
        }
    }
    
    /// Handle ATP shortage situations
    async fn handle_atp_shortage(
        &self,
        layer: BiologicalLayer,
        amount: f64,
        operation: &str,
    ) -> Result<bool> {
        let mut state = self.state.write().await;
        
        if state.current >= state.emergency_threshold {
            // Use emergency reserves
            let emergency_amount = amount.min(state.reserved);
            state.current -= emergency_amount;
            state.reserved -= emergency_amount;
            
            log::warn!(
                "Using emergency ATP reserves: {:.1} for {:?}:{} (reserves now: {:.1})",
                emergency_amount, layer, operation, state.reserved
            );
            
            Ok(true)
        } else {
            // Critical ATP shortage - trigger anaerobic processing
            drop(state);
            self.trigger_anaerobic_processing().await;
            
            Err(OscillatoryError::InsufficientATP {
                required: amount,
                available: state.current,
            })
        }
    }
    
    /// Trigger anaerobic processing mode (lactate pathway)
    async fn trigger_anaerobic_processing(&self) {
        log::warn!("ATP critically low - triggering anaerobic processing mode");
        
        // Update cosmic oscillation phase to reflect energy crisis
        let mut phase = self.cosmic_oscillation_phase.write().await;
        *phase = OscillationPhase::Decay;
        
        // In a full implementation, this would:
        // 1. Switch to lighter, less accurate models
        // 2. Reduce processing precision
        // 3. Prioritize only critical functions
        // 4. Activate ATP regeneration protocols
    }
    
    /// Regenerate ATP over time
    pub async fn regenerate_atp(&self, duration_minutes: f64) -> Result<()> {
        let mut state = self.state.write().await;
        
        let time_elapsed = Utc::now()
            .signed_duration_since(state.last_update)
            .num_minutes() as f64;
        
        // Calculate regeneration amount
        let regeneration_time = duration_minutes.min(time_elapsed);
        let base_regeneration = state.regeneration_rate * regeneration_time;
        
        // Apply quantum efficiency boost
        let quantum_regeneration = base_regeneration * state.quantum_efficiency;
        
        // Apply radical damage penalty
        let damage_penalty = 1.0 / (1.0 + state.radical_damage * 0.01);
        let final_regeneration = quantum_regeneration * damage_penalty;
        
        state.current = (state.current + final_regeneration).min(state.maximum);
        state.last_update = Utc::now();
        
        log::debug!("Regenerated {:.1} ATP (base: {:.1}, quantum: {:.1}, damage penalty: {:.2})",
                   final_regeneration, base_regeneration, quantum_regeneration, damage_penalty);
        
        Ok(())
    }
    
    /// Get current ATP state
    pub async fn get_state(&self) -> ATPState {
        self.state.read().await.clone()
    }
    
    /// Get ATP consumption history
    pub async fn get_consumption_history(&self) -> Vec<(DateTime<Utc>, String, f64, f64)> {
    pub async fn get_consumption_history(&self) -> Vec<(DateTime<Utc>, String, f64, f64)> {
        self.consumption_history.read().await.clone()
    }
    
    /// Calculate ATP efficiency metrics
    pub async fn calculate_efficiency_metrics(&self) -> ATPEfficiencyMetrics {
        let state = self.state.read().await;
        let history = self.consumption_history.read().await;
        
        let total_consumed: f64 = history.iter().map(|(_, _, amount, _)| amount).sum();
        let average_consumption = if !history.is_empty() {
            total_consumed / history.len() as f64
        } else {
            0.0
        };
        
        let utilization_rate = if state.maximum > 0.0 {
            (state.maximum - state.current) / state.maximum
        } else {
            0.0
        };
        
        ATPEfficiencyMetrics {
            total_consumed,
            average_consumption,
            utilization_rate,
            quantum_efficiency: state.quantum_efficiency,
            radical_damage_impact: state.radical_damage,
            regeneration_rate: state.regeneration_rate,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATPEfficiencyMetrics {
    pub total_consumed: f64,
    pub average_consumption: f64,
    pub utilization_rate: f64,
    pub quantum_efficiency: f64,
    pub radical_damage_impact: f64,
    pub regeneration_rate: f64,
}

// src/atp/quantum_manager.rs
//! Quantum-enhanced ATP manager implementing the Membrane Quantum Computation Theorem
//! with metabolic mode switching based on energy demand and environmental conditions.

use crate::atp::{OscillatoryATPManager, ATPState, MetabolicMode, LayerResourceRequirement};
use crate::quantum::{QuantumMembraneState, ENAQTProcessor, QuantumOscillatoryProfile};
use crate::oscillatory::{OscillationPhase, UniversalOscillator};
use crate::biological::BiologicalLayer;
use crate::error::{OscillatoryError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Advanced ATP manager with quantum membrane computation
pub struct QuantumATPManager {
    /// Base oscillatory ATP manager
    base_manager: OscillatoryATPManager,
    /// Quantum membrane state
    quantum_membrane: QuantumMembraneState,
    /// ENAQT processor for quantum calculations
    enaqt_processor: ENAQTProcessor,
    /// ATP synthase modeled as quantum oscillator
    atp_synthase_oscillator: UniversalOscillator,
    /// Accumulated radical damage over time
    radical_damage_accumulator: f64,
    /// Current metabolic mode
    metabolic_mode: Arc<RwLock<MetabolicMode>>,
    /// Metabolic mode history for analysis
    mode_history: Arc<RwLock<Vec<(DateTime<Utc>, MetabolicMode)>>>,
    /// Temperature regulation system
    temperature_controller: TemperatureController,
}

impl QuantumATPManager {
    pub fn new(maximum_atp: f64, operating_temperature: f64) -> Self {
        let quantum_membrane = QuantumMembraneState::new(operating_temperature);
        let enaqt_processor = ENAQTProcessor::new(10);
        
        // ATP synthase as quantum oscillator (rotational frequency ~100 Hz)
        let atp_synthase_oscillator = UniversalOscillator::new(
            1.0,    // Initial amplitude
            100.0,  // Natural frequency (Hz) - ATP synthase rotation
            0.1,    // Damping coefficient
            3,      // 3D rotational dynamics
        ).with_forcing(|t| {
            // Proton gradient forcing function
            0.5 * (2.0 * std::f64::consts::PI * 10.0 * t).sin() // 10 Hz proton oscillation
        });
        
        // Determine initial metabolic mode based on temperature
        let initial_mode = if operating_temperature < 290.0 {
            MetabolicMode::ColdBlooded {
                temperature_advantage: Self::calculate_temperature_advantage(operating_temperature),
                metabolic_reduction: 0.5,
            }
        } else if operating_temperature > 310.0 {
            MetabolicMode::MammalianBurden {
                quantum_cost_multiplier: 1.5,
                radical_generation_rate: 1e-5,
            }
        } else {
            MetabolicMode::MammalianBurden {
                quantum_cost_multiplier: 1.2,
                radical_generation_rate: 8e-6,
            }
        };
        
        Self {
            base_manager: OscillatoryATPManager::new(maximum_atp),
            quantum_membrane,
            enaqt_processor,
            atp_synthase_oscillator,
            radical_damage_accumulator: 0.0,
            metabolic_mode: Arc::new(RwLock::new(initial_mode)),
            mode_history: Arc::new(RwLock::new(Vec::new())),
            temperature_controller: TemperatureController::new(operating_temperature),
        }
    }
    
    /// Calculate quantum-enhanced ATP cost with full metabolic modeling
    pub async fn calculate_quantum_atp_cost(
        &self,
        layer: BiologicalLayer,
        query_complexity: f64,
        quantum_profile: &QuantumOscillatoryProfile,
    ) -> Result<f64> {
        // Base oscillatory cost
        let base_cost = self.base_manager.calculate_oscillatory_atp_cost(
            layer,
            query_complexity,
            &quantum_profile.base_oscillation,
        ).await?;
        
        // Quantum membrane efficiency modulation
        let quantum_efficiency = self.enaqt_processor.calculate_transport_efficiency(
            self.quantum_membrane.enaqt_coupling_strength,
            self.quantum_membrane.temperature_k,
        )?;
        
        // Current metabolic mode modulation
        let mode = self.metabolic_mode.read().await;
        let mode_factor = match &*mode {
            MetabolicMode::SustainedFlight { efficiency_boost, .. } => {
                0.4 * efficiency_boost // Sustained flight provides massive efficiency
            },
            MetabolicMode::ColdBlooded { temperature_advantage, metabolic_reduction } => {
                metabolic_reduction / temperature_advantage // Cold-blooded efficiency
            },
            MetabolicMode::MammalianBurden { quantum_cost_multiplier, .. } => {
                *quantum_cost_multiplier // Mammalian quantum burden
            },
            MetabolicMode::AnaerobicEmergency { efficiency_penalty, .. } => {
                2.0 + efficiency_penalty // Emergency metabolism is expensive
            },
        };
        drop(mode);
        
        // ATP synthase quantum oscillation efficiency
        let synthase_phase = self.atp_synthase_oscillator.calculate_phase();
        let synthase_efficiency = match synthase_phase {
            OscillationPhase::Peak => 0.6,      // Maximum quantum efficiency
            OscillationPhase::Resonance => 0.5, // Optimal ENAQT coupling
            OscillationPhase::Acceleration => 0.8,
            OscillationPhase::Decay => 1.2,
            OscillationPhase::Equilibrium => 1.0,
        };
        
        // Radical damage penalty
        let damage_penalty = 1.0 + (self.radical_damage_accumulator * 0.001);
        
        // Temperature optimization
        let temp_factor = self.temperature_controller.calculate_efficiency_factor();
        
        let quantum_cost = base_cost * mode_factor * synthase_efficiency * damage_penalty 
                          / (quantum_efficiency * temp_factor);
        
        Ok(quantum_cost)
    }
    
    /// Process operation with full quantum membrane modeling
    pub async fn process_with_quantum_membrane(
        &mut self,
        operation: &str,
        energy_demand: f64,
        dt: f64,
    ) -> Result<QuantumProcessingResult> {
        // Evolve ATP synthase oscillator
        self.atp_synthase_oscillator.evolve(dt)?;
        
        // Calculate quantum tunneling and radical generation
        let tunneling_prob = self.enaqt_processor.calculate_quantum_tunneling_probability(
            1.0, // 1 eV barrier height (typical for biological systems)
            self.quantum_membrane.membrane_thickness_nm,
            0.3, // 0.3 eV electron energy
        )?;
        
        let radical_rate = self.enaqt_processor.calculate_radical_generation_rate(
            1e-3, // Electron density (M)
            2e-4, // Oxygen concentration (M)
            tunneling_prob,
        )?;
        
        // Accumulate radical damage over time
        self.radical_damage_accumulator += radical_rate * dt;
        
        // Check for metabolic mode transitions
        self.update_metabolic_mode(energy_demand).await?;
        
        // Calculate ATP efficiency with quantum effects
        let quantum_efficiency = self.enaqt_processor.calculate_transport_efficiency(
            self.quantum_membrane.enaqt_coupling_strength,
            self.quantum_membrane.temperature_k,
        )?;
        
        // Calculate actual ATP generated with quantum enhancement
        let base_atp = energy_demand;
        let quantum_enhanced_atp = base_atp * quantum_efficiency;
        
        // Apply metabolic mode efficiency
        let mode = self.metabolic_mode.read().await;
        let final_atp = match &*mode {
            MetabolicMode::SustainedFlight { efficiency_boost, .. } => {
                quantum_enhanced_atp * efficiency_boost
            },
            MetabolicMode::ColdBlooded { temperature_advantage, .. } => {
                quantum_enhanced_atp * temperature_advantage
            },
            MetabolicMode::MammalianBurden { .. } => {
                quantum_enhanced_atp // Standard efficiency
            },
            MetabolicMode::AnaerobicEmergency { efficiency_penalty, .. } => {
                quantum_enhanced_atp / (1.0 + efficiency_penalty)
            },
        };
        drop(mode);
        
        // Consume ATP through base manager
        let consumption_success = self.base_manager.consume_atp(
            BiologicalLayer::Context, // Default layer for quantum processing
            final_atp,
            operation,
        ).await?;
        
        // Update quantum membrane state
        self.update_quantum_membrane_state(energy_demand, dt).await?;
        
        Ok(QuantumProcessingResult {
            atp_generated: final_atp,
            quantum_efficiency,
            radical_damage_rate: radical_rate,
            tunneling_probability: tunneling_prob,
            metabolic_mode: self.metabolic_mode.read().await.clone(),
            consumption_success,
            coherence_time_fs: self.quantum_membrane.coherence_time_fs,
            synthase_phase: self.atp_synthase_oscillator.calculate_phase(),
            temperature_k: self.quantum_membrane.temperature_k,
        })
    }
    
    /// Update metabolic mode based on energy demand and system state
    async fn update_metabolic_mode(&mut self, energy_demand: f64) -> Result<()> {
        let current_atp_state = self.base_manager.get_state().await;
        let current_mode = self.metabolic_mode.read().await.clone();
        drop(current_atp_state);
        
        let new_mode = if energy_demand > 15.0 && current_atp_state.percentage() > 70.0 {
            // High energy demand with sufficient ATP - switch to sustained flight
            MetabolicMode::SustainedFlight {
                efficiency_boost: 2.5,
                radical_suppression: 0.3,
            }
        } else if current_atp_state.is_critical() {
            // ATP critical - emergency anaerobic mode
            MetabolicMode::AnaerobicEmergency {
                lactate_pathway_active: true,
                efficiency_penalty: 1.8,
            }
        } else if energy_demand < 3.0 && self.quantum_membrane.temperature_k < 290.0 {
            // Low demand, low temperature - cold-blooded advantage
            MetabolicMode::ColdBlooded {
                temperature_advantage: Self::calculate_temperature_advantage(self.quantum_membrane.temperature_k),
                metabolic_reduction: 0.6,
            }
        } else {
            // Default mammalian metabolism
            let radical_rate = if self.radical_damage_accumulator > 10.0 { 1.5e-5 } else { 1e-5 };
            MetabolicMode::MammalianBurden {
                quantum_cost_multiplier: 1.2,
                radical_generation_rate: radical_rate,
            }
        };
        
        // Update mode if changed
        if new_mode != current_mode {
            let mut mode = self.metabolic_mode.write().await;
            *mode = new_mode.clone();
            
            // Record mode change in history
            let mut history = self.mode_history.write().await;
            history.push((Utc::now(), new_mode.clone()));
            
            // Limit history size
            if history.len() > 1000 {
                history.drain(..100);
            }
            
            log::info!("Metabolic mode changed from {:?} to {:?}", current_mode, new_mode);
        }
        
        Ok(())
    }
    
    /// Update quantum membrane state based on processing activity
    async fn update_quantum_membrane_state(&mut self, energy_demand: f64, dt: f64) -> Result<()> {
        // Update coherence time based on activity level
        let base_coherence = 660.0; // FMO complex baseline
        let activity_factor = if energy_demand > 10.0 {
            1.2 // High activity can enhance coherence through optimal coupling
        } else {
            1.0
        };
        
        self.quantum_membrane.coherence_time_fs = self.enaqt_processor.calculate_enhanced_coherence_time(
            base_coherence,
            self.quantum_membrane.enaqt_coupling_strength,
            self.quantum_membrane.temperature_k,
        )? * activity_factor;
        
        // Update tunneling probability based on membrane state
        self.quantum_membrane.tunneling_probability = self.enaqt_processor.calculate_quantum_tunneling_probability(
            1.0,
            self.quantum_membrane.membrane_thickness_nm,
            0.3,
        )?;
        
        // Update radical generation rate
        self.quantum_membrane.radical_generation_rate = self.enaqt_processor.calculate_radical_generation_rate(
            1e-3,
            2e-4,
            self.quantum_membrane.tunneling_probability,
        )?;
        
        // Evolve quantum oscillations
        for oscillation in &mut self.quantum_membrane.quantum_oscillations {
            let phase_evolution = 2.0 * std::f64::consts::PI * 100.0 * dt; // 100 Hz typical frequency
            *oscillation *= (Complex::i() * phase_evolution).exp();
        }
        
        Ok(())
    }
    
    /// Calculate temperature advantage for cold-blooded metabolism
    fn calculate_temperature_advantage(temperature_k: f64) -> f64 {
        // Exponential advantage as temperature decreases below 300K
        let temp_diff = (300.0 - temperature_k).max(0.0);
        1.0 + (temp_diff / 20.0).exp() - 1.0
    }
    
    /// Get comprehensive longevity assessment based on quantum damage
    pub async fn get_longevity_assessment(&self) -> LongevityAssessment {
        let damage_rate = self.radical_damage_accumulator;
        let mode = self.metabolic_mode.read().await;
        
        let quantum_burden = match &*mode {
            MetabolicMode::SustainedFlight { .. } => 0.3,
            MetabolicMode::ColdBlooded { .. } => 0.5,
            MetabolicMode::MammalianBurden { .. } => 2.0,
            MetabolicMode::AnaerobicEmergency { .. } => 3.0,
        };
        
        // Predict remaining operational time based on quantum damage accumulation
        let damage_threshold = 100.0; // Arbitrary damage units before critical failure
        let remaining_time = if damage_rate > 0.0 {
            (damage_threshold - damage_rate) / damage_rate
        } else {
            f64::INFINITY
        };
        
        // Generate optimization suggestions
        let suggestions = self.generate_optimization_suggestions(&mode).await;
        
        LongevityAssessment {
            current_damage: damage_rate,
            quantum_burden_factor: quantum_burden,
            predicted_remaining_time: remaining_time,
            metabolic_optimization_suggestions: suggestions,
            coherence_time_fs: self.quantum_membrane.coherence_time_fs,
            transport_efficiency: self.enaqt_processor.calculate_transport_efficiency(
                self.quantum_membrane.enaqt_coupling_strength,
                self.quantum_membrane.temperature_k,
            ).unwrap_or(0.0),
            temperature_optimization: self.temperature_controller.get_optimization_status(),
        }
    }
    
    /// Generate metabolic optimization suggestions
    async fn generate_optimization_suggestions(&self, mode: &MetabolicMode) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        match mode {
            MetabolicMode::MammalianBurden { .. } => {
                suggestions.push("Consider implementing sustained flight protocols during high-demand periods".to_string());
                suggestions.push("Optimize temperature regulation to reduce quantum burden".to_string());
                suggestions.push("Implement query batching to maintain peak quantum coherence".to_string());
                suggestions.push("Consider intermittent fasting protocols to reduce metabolic load".to_string());
            },
            MetabolicMode::AnaerobicEmergency { .. } => {
                suggestions.push("CRITICAL: Reduce processing complexity immediately".to_string());
                suggestions.push("Implement ATP regeneration protocols".to_string());
                suggestions.push("Switch to lighter models to restore aerobic metabolism".to_string());
            },
            MetabolicMode::ColdBlooded { .. } => {
                suggestions.push("Current temperature optimization is excellent".to_string());
                suggestions.push("Maintain current operating temperature for longevity".to_string());
            },
            MetabolicMode::SustainedFlight { .. } => {
                suggestions.push("Excellent metabolic mode - maintain high activity levels".to_string());
                suggestions.push("Monitor for transition back to baseline when demand decreases".to_string());
            }
        }
        
        // Add quantum-specific suggestions
        if self.quantum_membrane.enaqt_coupling_strength < 0.3 {
            suggestions.push("Increase ENAQT coupling strength for better efficiency".to_string());
        } else if self.quantum_membrane.enaqt_coupling_strength > 0.6 {
            suggestions.push("Reduce ENAQT coupling to avoid over-coupling penalties".to_string());
        }
        
        if self.quantum_membrane.coherence_time_fs < 500.0 {
            suggestions.push("Implement coherence enhancement protocols".to_string());
        }
        
        suggestions
    }
    
    /// Optimize ENAQT coupling for current conditions
    pub async fn optimize_enaqt_coupling(&mut self) -> Result<f64> {
        let optimal_coupling = self.enaqt_processor.optimize_coupling(0.95)?;
        self.quantum_membrane.enaqt_coupling_strength = optimal_coupling;
        
        log::info!("Optimized ENAQT coupling to {:.3}", optimal_coupling);
        Ok(optimal_coupling)
    }
    
    /// Get detailed quantum state information
    pub async fn get_quantum_state(&self) -> QuantumStateReport {
        QuantumStateReport {
            membrane_state: self.quantum_membrane.clone(),
            atp_synthase_phase: self.atp_synthase_oscillator.calculate_phase(),
            atp_synthase_energy: self.atp_synthase_oscillator.total_energy(),
            metabolic_mode: self.metabolic_mode.read().await.clone(),
            radical_damage: self.radical_damage_accumulator,
            quantum_advantage_factor: self.quantum_membrane.quantum_advantage_factor(),
            is_quantum_optimal: self.quantum_membrane.is_quantum_optimal(),
        }
    }
}

/// Temperature controller for metabolic optimization
#[derive(Debug, Clone)]
struct TemperatureController {
    target_temperature: f64,
    current_temperature: f64,
    optimization_active: bool,
}

impl TemperatureController {
    fn new(initial_temperature: f64) -> Self {
        Self {
            target_temperature: initial_temperature,
            current_temperature: initial_temperature,
            optimization_active: false,
        }
    }
    
    fn calculate_efficiency_factor(&self) -> f64 {
        // Efficiency increases as temperature decreases below 300K
        if self.current_temperature < 300.0 {
            (310.0 - self.current_temperature) / 10.0
        } else {
            1.0 / (1.0 + (self.current_temperature - 300.0) / 50.0)
        }
    }
    
    fn get_optimization_status(&self) -> String {
        if self.current_temperature < 290.0 {
            "Excellent - Cold-blooded advantage active".to_string()
        } else if self.current_temperature < 305.0 {
            "Good - Near-optimal temperature range".to_string()
        } else {
            "Suboptimal - Consider temperature reduction".to_string()
        }
    }
}

/// Result of quantum processing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumProcessingResult {
    pub atp_generated: f64,
    pub quantum_efficiency: f64,
    pub radical_damage_rate: f64,
    pub tunneling_probability: f64,
    pub metabolic_mode: MetabolicMode,
    pub consumption_success: bool,
    pub coherence_time_fs: f64,
    pub synthase_phase: OscillationPhase,
    pub temperature_k: f64,
}

/// Comprehensive longevity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongevityAssessment {
    pub current_damage: f64,
    pub quantum_burden_factor: f64,
    pub predicted_remaining_time: f64,
    pub metabolic_optimization_suggestions: Vec<String>,
    pub coherence_time_fs: f64,
    pub transport_efficiency: f64,
    pub temperature_optimization: String,
}

/// Detailed quantum state report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumStateReport {
    pub membrane_state: QuantumMembraneState,
    pub atp_synthase_phase: OscillationPhase,
    pub atp_synthase_energy: f64,
    pub metabolic_mode: MetabolicMode,
    pub radical_damage: f64,
    pub quantum_advantage_factor: f64,
    pub is_quantum_optimal: bool,
}
// src/hierarchy/mod.rs
//! Multi-scale hierarchy processing system implementing the 10-level biological hierarchy
//! from quantum oscillations (10⁻⁴⁴ s) to cosmic patterns (10¹³ s).

pub mod levels;
pub mod emergence;
pub mod coupling;
pub mod processing;

use crate::error::{OscillatoryError, Result};
use crate::oscillatory::{OscillationProfile, OscillationPhase};
use crate::quantum::QuantumOscillatoryProfile;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// The 10 hierarchy levels of biological organization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HierarchyLevel {
    /// Level 1: Quantum oscillations (10⁻⁴⁴ s) - Planck scale quantum fluctuations
    QuantumOscillations = 1,
    /// Level 2: Atomic oscillations (10⁻¹⁵ s) - Electronic transitions, nuclear vibrations
    AtomicOscillations = 2,
    /// Level 3: Molecular oscillations (10⁻¹² s) - Molecular vibrations, rotations
    MolecularOscillations = 3,
    /// Level 4: Cellular oscillations (10⁻³ s) - Metabolic cycles, membrane dynamics
    CellularOscillations = 4,
    /// Level 5: Organismal oscillations (10⁰ s) - Heartbeat, breathing, neural firing
    OrganismalOscillations = 5,
    /// Level 6: Cognitive oscillations (10³ s) - Thought processes, decision making
    CognitiveOscillations = 6,
    /// Level 7: Social oscillations (10⁶ s) - Social interactions, group dynamics
    SocialOscillations = 7,
    /// Level 8: Technological oscillations (10⁹ s) - Innovation cycles, technological adoption
    TechnologicalOscillations = 8,
    /// Level 9: Civilizational oscillations (10¹² s) - Rise and fall of civilizations
    CivilizationalOscillations = 9,
    /// Level 10: Cosmic oscillations (10¹³ s) - Stellar evolution, galactic dynamics
    CosmicOscillations = 10,
}

impl HierarchyLevel {
    /// Get the characteristic time scale for this hierarchy level
    pub fn time_scale_seconds(&self) -> f64 {
        match self {
            HierarchyLevel::QuantumOscillations => 1e-44,
            HierarchyLevel::AtomicOscillations => 1e-15,
            HierarchyLevel::MolecularOscillations => 1e-12,
            HierarchyLevel::CellularOscillations => 1e-3,
            HierarchyLevel::OrganismalOscillations => 1e0,
            HierarchyLevel::CognitiveOscillations => 1e3,
            HierarchyLevel::SocialOscillations => 1e6,
            HierarchyLevel::TechnologicalOscillations => 1e9,
            HierarchyLevel::CivilizationalOscillations => 1e12,
            HierarchyLevel::CosmicOscillations => 1e13,
        }
    }
    
    /// Get the characteristic frequency for this hierarchy level
    pub fn characteristic_frequency(&self) -> f64 {
        1.0 / self.time_scale_seconds()
    }
    
    /// Get all hierarchy levels
    pub fn all_levels() -> Vec<HierarchyLevel> {
        vec![
            HierarchyLevel::QuantumOscillations,
            HierarchyLevel::AtomicOscillations,
            HierarchyLevel::MolecularOscillations,
            HierarchyLevel::CellularOscillations,
            HierarchyLevel::OrganismalOscillations,
            HierarchyLevel::CognitiveOscillations,
            HierarchyLevel::SocialOscillations,
            HierarchyLevel::TechnologicalOscillations,
            HierarchyLevel::CivilizationalOscillations,
            HierarchyLevel::CosmicOscillations,
        ]
    }
    
    /// Get adjacent hierarchy levels for coupling calculations
    pub fn adjacent_levels(&self) -> Vec<HierarchyLevel> {
        let all_levels = Self::all_levels();
        let current_index = *self as usize - 1;
        let mut adjacent = Vec::new();
        
        if current_index > 0 {
            adjacent.push(all_levels[current_index - 1]);
        }
        if current_index < all_levels.len() - 1 {
            adjacent.push(all_levels[current_index + 1]);
        }
        
        adjacent
    }
    
    /// Check if this level can couple with another level
    pub fn can_couple_with(&self, other: HierarchyLevel) -> bool {
        let level_diff = (*self as i32 - other as i32).abs();
        level_diff <= 3 // Allow coupling within 3 levels
    }
}

impl fmt::Display for HierarchyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            HierarchyLevel::QuantumOscillations => "Quantum (10⁻⁴⁴s)",
            HierarchyLevel::AtomicOscillations => "Atomic (10⁻¹⁵s)",
            HierarchyLevel::MolecularOscillations => "Molecular (10⁻¹²s)",
            HierarchyLevel::CellularOscillations => "Cellular (10⁻³s)",
            HierarchyLevel::OrganismalOscillations => "Organismal (10⁰s)",
            HierarchyLevel::CognitiveOscillations => "Cognitive (10³s)",
            HierarchyLevel::SocialOscillations => "Social (10⁶s)",
            HierarchyLevel::TechnologicalOscillations => "Technological (10⁹s)",
            HierarchyLevel::CivilizationalOscillations => "Civilizational (10¹²s)",
            HierarchyLevel::CosmicOscillations => "Cosmic (10¹³s)",
        };
        write!(f, "{}", name)
    }
}

/// Result of hierarchy-level processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyResult {
    pub level: HierarchyLevel,
    pub processing_success: bool,
    pub emergence_detected: bool,
    pub coupling_strength: f64,
    pub information_content: f64,
    pub oscillation_phase: OscillationPhase,
    pub cross_scale_interactions: Vec<(HierarchyLevel, f64)>,
    pub computational_cost: f64,
}

/// Multi-scale hierarchy processor
