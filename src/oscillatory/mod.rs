//! Core oscillatory dynamics implementation based on the Universal Oscillation Equation:
//! d²y/dt² + γ(dy/dt) + ω²y = F(t)
//! 
//! This module implements the fundamental oscillatory behavior that underlies all
//! biological information processing, from quantum coherence to cosmic patterns.

pub mod types;
pub mod equations;
pub mod dynamics;
pub mod entropy;

use crate::error::{AutobahnError, AutobahnResult};
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
    pub fn evolve(&mut self, dt: f64) -> AutobahnResult<()> {
        if dt <= 0.0 {
            return Err(AutobahnError::ConfigurationError {
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
            return Err(AutobahnError::PhysicsError {
                message: format!("Amplitude overflow: {:.2}", self.state.amplitude),
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
    
    /// Calculate slope of amplitude over time using simple linear regression
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
    
    /// Get oscillation energy
    pub fn total_energy(&self) -> f64 {
        let kinetic = 0.5 * self.state.velocity.norm_squared();
        let potential = 0.5 * self.state.natural_frequency.powi(2) * self.state.position.norm_squared();
        kinetic + potential
    }
    
    /// Get current oscillation profile
    pub fn get_profile(&self) -> OscillationProfile {
        let mut profile = OscillationProfile::new(self.state.amplitude, self.state.frequency);
        profile.phase = self.calculate_phase();
        profile.quality_factor = if self.state.damping_coefficient > 0.0 {
            self.state.natural_frequency / (2.0 * self.state.damping_coefficient)
        } else {
            f64::INFINITY
        };
        profile
    }
}

// Implement Send + Sync for UniversalOscillator
unsafe impl Send for UniversalOscillator {}
unsafe impl Sync for UniversalOscillator {} 