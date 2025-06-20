//! Utility functions for the Autobahn framework

use crate::error::AutobahnResult;
use std::f64::consts::PI;

/// Mathematical utilities
pub mod math {
    use super::*;
    
    /// Calculate Shannon entropy of a data sequence
    pub fn shannon_entropy(data: &[f64]) -> f64 {
        let mut entropy = 0.0;
        let total: f64 = data.iter().sum();
        
        if total == 0.0 {
            return 0.0;
        }
        
        for &value in data {
            if value > 0.0 {
                let probability = value / total;
                entropy -= probability * probability.log2();
            }
        }
        
        entropy
    }
    
    /// Calculate variance of data
    pub fn variance(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64
    }
    
    /// Calculate standard deviation
    pub fn standard_deviation(data: &[f64]) -> f64 {
        variance(data).sqrt()
    }
    
    /// Normalize data to range [0, 1]
    pub fn normalize(data: &[f64]) -> Vec<f64> {
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max == min {
            return vec![0.5; data.len()];
        }
        
        data.iter()
            .map(|&x| (x - min) / (max - min))
            .collect()
    }
    
    /// Calculate cross-correlation between two sequences
    pub fn cross_correlation(x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = x.len();
        let m = y.len();
        let mut result = vec![0.0; n + m - 1];
        
        for i in 0..n {
            for j in 0..m {
                result[i + j] += x[i] * y[j];
            }
        }
        
        result
    }
    
    /// Calculate autocorrelation of a sequence
    pub fn autocorrelation(data: &[f64]) -> Vec<f64> {
        cross_correlation(data, data)
    }
}

/// Signal processing utilities
pub mod signal {
    use super::*;
    
    /// Generate sine wave
    pub fn sine_wave(frequency: f64, amplitude: f64, phase: f64, samples: usize, sample_rate: f64) -> Vec<f64> {
        (0..samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                amplitude * (2.0 * PI * frequency * t + phase).sin()
            })
            .collect()
    }
    
    /// Generate cosine wave
    pub fn cosine_wave(frequency: f64, amplitude: f64, phase: f64, samples: usize, sample_rate: f64) -> Vec<f64> {
        (0..samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                amplitude * (2.0 * PI * frequency * t + phase).cos()
            })
            .collect()
    }
    
    /// Detect zero crossings in a signal
    pub fn zero_crossings(data: &[f64]) -> Vec<usize> {
        let mut crossings = Vec::new();
        
        for i in 1..data.len() {
            if (data[i-1] >= 0.0 && data[i] < 0.0) || (data[i-1] < 0.0 && data[i] >= 0.0) {
                crossings.push(i);
            }
        }
        
        crossings
    }
    
    /// Apply moving average filter
    pub fn moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
        if window_size == 0 || window_size > data.len() {
            return data.to_vec();
        }
        
        let mut result = Vec::new();
        
        for i in 0..data.len() {
            let start = if i >= window_size / 2 { i - window_size / 2 } else { 0 };
            let end = (i + window_size / 2 + 1).min(data.len());
            
            let sum: f64 = data[start..end].iter().sum();
            result.push(sum / (end - start) as f64);
        }
        
        result
    }
}

/// Time utilities
pub mod time {
    use chrono::{DateTime, Utc, Duration};
    
    /// Get current timestamp
    pub fn now() -> DateTime<Utc> {
        Utc::now()
    }
    
    /// Calculate time difference in milliseconds
    pub fn time_diff_ms(start: DateTime<Utc>, end: DateTime<Utc>) -> i64 {
        (end - start).num_milliseconds()
    }
    
    /// Create timestamp from milliseconds since epoch
    pub fn from_millis(millis: i64) -> Option<DateTime<Utc>> {
        DateTime::from_timestamp_millis(millis)
    }
}

/// Quantum mechanics utilities
pub mod quantum {
    use super::*;
    
    /// Calculate quantum tunneling probability
    pub fn tunneling_probability(mass: f64, barrier_height: f64, barrier_width: f64) -> f64 {
        // Quantum tunneling probability: P = exp(-2 * sqrt(2m(V-E)) * a / ℏ)
        let hbar = 1.054571817e-34; // Reduced Planck constant
        let mass_kg = mass * 1.66053906660e-27; // Convert AMU to kg
        let energy_barrier = barrier_height * 1.602176634e-19; // Convert eV to Joules
        
        let exponent = -2.0 * (2.0 * mass_kg * energy_barrier).sqrt() * barrier_width / hbar;
        exponent.exp().min(1.0) // Cap at 1.0 for probability
    }
    
    /// Calculate coherence time based on temperature and coupling
    pub fn coherence_time(temperature: f64, coupling_strength: f64) -> f64 {
        // Simplified coherence time calculation
        let boltzmann = 1.380649e-23; // Boltzmann constant
        let thermal_energy = boltzmann * temperature;
        
        // Coherence time inversely proportional to thermal energy and coupling
        1.0 / (thermal_energy * coupling_strength + 1e-15) // Add small constant to avoid division by zero
    }
}

/// Biological utilities
pub mod biology {
    use super::*;
    
    /// Calculate ATP hydrolysis energy
    pub fn atp_hydrolysis_energy() -> f64 {
        30.5 // kJ/mol under standard conditions
    }
    
    /// Calculate metabolic rate based on mass and temperature
    pub fn metabolic_rate(mass_kg: f64, temperature_k: f64) -> f64 {
        // Simplified metabolic rate calculation (Kleiber's law)
        let base_rate = 70.0; // W for 70kg human at 37°C
        let mass_factor = (mass_kg / 70.0).powf(0.75);
        let temperature_factor = (temperature_k / 310.0).powf(2.0);
        
        base_rate * mass_factor * temperature_factor
    }
    
    /// Calculate membrane potential
    pub fn membrane_potential(ion_concentrations_inside: &[f64], ion_concentrations_outside: &[f64]) -> f64 {
        // Simplified Nernst equation
        let r = 8.314; // Gas constant
        let t = 310.0; // Body temperature in K
        let f = 96485.0; // Faraday constant
        
        let mut potential = 0.0;
        
        for (i, (&c_in, &c_out)) in ion_concentrations_inside.iter().zip(ion_concentrations_outside.iter()).enumerate() {
            if c_in > 0.0 && c_out > 0.0 {
                let charge = if i == 0 { 1.0 } else { -1.0 }; // Simplified charge assignment
                potential += (r * t / (charge * f)) * (c_out / c_in).ln();
            }
        }
        
        potential
    }
}

/// Data validation utilities
pub mod validation {
    use super::*;
    
    /// Validate that all values are finite
    pub fn validate_finite(data: &[f64]) -> AutobahnResult<()> {
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(crate::error::AutobahnError::ValidationError(
                    format!("Non-finite value at index {}: {}", i, value)
                ));
            }
        }
        Ok(())
    }
    
    /// Validate that all values are in range [min, max]
    pub fn validate_range(data: &[f64], min: f64, max: f64) -> AutobahnResult<()> {
        for (i, &value) in data.iter().enumerate() {
            if value < min || value > max {
                return Err(crate::error::AutobahnError::ValidationError(
                    format!("Value at index {} ({}) outside range [{}, {}]", i, value, min, max)
                ));
            }
        }
        Ok(())
    }
    
    /// Validate that data is not empty
    pub fn validate_not_empty<T>(data: &[T]) -> AutobahnResult<()> {
        if data.is_empty() {
            return Err(crate::error::AutobahnError::ValidationError(
                "Data cannot be empty".to_string()
            ));
        }
        Ok(())
    }
} 